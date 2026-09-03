# Prototype-guided transfer of sparse literature knowledge for electrolyte additive discovery

Weixiang HONG<sup>1,5</sup>, Hongting DU<sup>1,5</sup>, Jiayue TANG<sup>1</sup>, Ruifeng TAN<sup>1</sup>, Yangjian QUAN<sup>2</sup>, Jia LI<sup>3,\*</sup>, Jiaqiang HUANG<sup>1,4,\*</sup>

<sup>1</sup> Sustainable Energy and Environment Thrust and Guangzhou Municipal Key Laboratory of Materials Informatics, The Hong Kong University of Science and Technology (Guangzhou), Nansha, Guangzhou, 511400, Guangdong, P.R. China

<sup>2</sup> Department of Chemistry, The Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong SAR, P.R. China

<sup>3</sup> Data Science and Analytics Thrust and Guangzhou Municipal Key Laboratory of Materials Informatics, The Hong Kong University of Science and Technology (Guangzhou), Nansha, Guangzhou, 511400, Guangdong, P.R. China

<sup>4</sup> Academy of Interdisciplinary Studies, The Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong SAR, P.R. China

<sup>5</sup> These authors contributed equally: Weixiang HONG, Hongting DU

\* Corresponding authors:

J. Huang: seejhuang@hkust-gz.edu.cn

J. Li: jialee@hkust-gz.edu.cn

## Abstract

Electrolyte additive discovery remains challenging because experimentally validated molecules are sparse, whereas accessible chemical spaces are vast and largely unlabeled. This challenge is amplified in lithium-ion batteries, where additive performance arises from coupled interfacial reactions rather than a single molecular property. Here, we develop a prototype-guided molecular intelligence, ProtoMI, a literature-driven framework that learns transferable structural priors from reported electrolyte additives and uses them to prioritize candidates in unlabeled chemical space. For boron-containing additives, ProtoMI combines 126 literature-reported molecules with 179,977 unlabeled candidates. Graph contrastive learning identifies seven chemically interpretable prototypes from the reported additives, and prototypeguided semi-supervised contrastive learning adapts these prototypes to the candidate space under source-target distribution mismatch. In retrospective temporal validation, ProtoMI achieves enrichment factors of 9.2-45.6 while screening less than 2% of the candidate space. A subsequent translation step identifies four commercially accessible candidates. One representative candidate, 4,4,5,5-Tetramethyl-2-[10-(1- naphthyl)anthracen-9-yl]-1,3,2-dioxaborolane (TNDB), improves high-temperature $\mathsf { L i F e P O 4 } |$ |graphite cycling at $5 5 ~ ^ { \circ } \mathsf { C }$ by 34.93% relative to the baseline electrolyte. An arsenal of characterizations and operando optical fiber Fourier transform infrared spectroscopy suggest that TNDB forms B-containing, F/P/O-modified inorganic interphases, suppresses solvent decomposition and reduces Fe deposition on graphite. This case study shows how sparse literature knowledge can guide experimentally efficient molecular discovery in data-scarce battery-additive spaces.

## Introduction

Lithium-ion batteries (LIBs) underpin electrified transport, grid storage and portable electronics, and their deployment is expected to expand substantially as electrification accelerates<sup>1,2</sup>. As LIBs are pushed toward longer lifetimes, higher power operation and harsher operating environments, battery degradation is increasingly governed by the stability of electrochemical interfaces, where electrolyte decomposition, interphase evolution and transition-metal crossover can trigger coupled degradation processes<sup>3-</sup> <sup>5</sup>. Electrolyte additives provide a practical and manufacturable route to regulate such interfaces. Even at low concentrations, additives can alter solvation structures, undergo preferential interfacial reactions, scavenge reactive impurities and promote stabilizing solid-electrolyte interphase (SEI) and cathode-electrolyte interphase (CEI)<sup>6</sup>. However, additive discovery remains largely empirical because additive efficacy emerges from the coupled effects of molecular stability, salt-solvent coordination, ion transport, decomposition pathways and interphase growth, none of which can be reliably captured by a single molecular descriptor<sup>3-6</sup>.

Recent advances in artificial intelligence (AI), molecular representation learning and data-driven optimization have accelerated the exploration of high-dimensional chemical spaces<sup>7,8</sup>. In battery-electrolyte research, data-driven strategies have been used<sup>9</sup> to predict molecular and electrolyte properties<sup>10</sup>, screen electrolyte components<sup>11</sup> with theoretical calculations<sup>12-16</sup> or curated datasets<sup>17,18</sup>, and guide electrolyte optimization workflows<sup>18-20</sup>. Active-learning approaches have further reduced experimental burden by iteratively updating models with new measurements<sup>20</sup>, while physics-informed and solvation-aware models have introduced thermodynamic, transport and molecular-interaction constraints into electrolyte design<sup>21,22</sup>. These studies have demonstrated the value of data-driven electrolyte discovery. Nevertheless, most existing approaches are formulated around supervised prediction or iterative optimization once target labels, simulation outputs or experimental feedback are available. This creates a mismatch for electrolyte additive discovery, where experimentally validated molecules are sparse, literature reports are heterogeneous and most accessible candidate molecules have no measured batteryrelevant labels.

Under such severe label scarcity, the central challenge shifts from predicting the performance of individual molecules to prioritizing chemically promising regions of an unlabeled molecular space under limited experimental budgets. Reported electrolyte additives are not merely isolated successful examples. Instead, they may encode recurring structural organization and shared interfacial regulation strategies<sup>6</sup>, including anion reception<sup>23</sup>, reactive-species scavenging<sup>24</sup>, solvation regulation<sup>25</sup> and interphase construction<sup>24,26</sup>. If these recurring patterns can be learned as prototypelevel priors rather than as molecule-level labels, they may guide exploration beyond the local neighborhoods of known additives. Representation learning provides a route to extract structured molecular embeddings from sparse data<sup>27</sup>, while prototype-based alignment<sup>28,29</sup> and semi-supervised learning<sup>30</sup> offer mechanisms to transfer such organization into unlabeled domains. Together, these advances suggest an alternative discovery paradigm: instead of directly predicting additive performance from scarce labels, sparse literature-derived success patterns can be converted into transferable molecular prototypes that guide efficient candidate selection (Fig. 1a,b).

Here we develop prototype-guided molecular intelligence (ProtoMI), a literaturedriven framework for molecular discovery under sparse positive knowledge and large unlabeled candidate spaces. ProtoMI treats reported additives as a positive molecular knowledge source rather than as a conventional supervised training set. The framework proceeds through three stages (Fig. 1c). First, literature-reported additive molecules are extracted and curated to construct a positive molecular dataset, and the unsupervised graph contrastive learning is used to identify prototype-level chemical organization within the reported additives. Second, prototype-guided semi-supervised contrastive learning adapts these prototypes to a large unlabeled molecular space under source-target distribution mismatch. Third, a subsequent knowledge-guided translation step incorporates chemical feasibility and experimental accessibility constraints to convert model recommendations into testable candidates.

As a case study, we focus on boron-containing electrolyte additives. Boroncontaining additives provide a suitable testbed because their Lewis-acidic and borate/boronate chemistries are well documented in electrolyte interphase regulation, yet remain sparsely explored across the broader boron-containing chemical space<sup>23-</sup> <sup>26</sup>. This combination of established chemical relevance and large unexplored molecular diversity creates a stringent test for ProtoMI: whether sparse literaturereported successes can be transformed into transferable prototype priors for efficient candidate selection. In this work, ProtoMI integrates 126 literature-reported boroncontaining additives with 179,977 unlabeled boron-containing candidates from PubChem<sup>31</sup>. Retrospective temporal validation shows that the framework recovers future-reported additives with enrichment factors of 9.2-45.6 while screening less than 2% of the candidate space. A knowledge-guided translation step further prioritizes four commercially accessible candidates for high-temperature $\mathsf { L i F e P O 4 }$ |graphite cells, a system in which elevated-temperature cycling stability remains an important practical challenge<sup>32,33</sup>. One representative candidate, 4,4,5,5-tetramethyl-2-[10-(1- naphthyl)anthracen-9-yl]-1,3,2-dioxaborolane (TNDB), improves capacity retention by 34.93% relative to the baseline electrolyte during cycling at $5 5 ~ ^ { \circ } \mathsf { C }$ . Post-mortem and operando characterizations indicate that TNDB participates in forming B-containing, F/P/O-modified interphases that suppress solvent decomposition and reduce Fe deposition on graphite. These results demonstrate that sparse literature-derived molecular successes can be transformed into transferable prototype priors for efficient electrolyte additive discovery under severe data scarcity.

a  
Conventional exploration  
![](images/4972e452dc9011071daba7486fbb17401eb061a3fcb1847de9b8751dcac1c8b1.jpg)

C  
![](images/00b6e77dc69e4206b9598de40c53f05ca8606bd802a1c0940c798309688750eb.jpg)

![](images/ddb3db426cd40de932de43db1c9e37f282a34e2f121f21485a0d4d4ffee25ff7.jpg)

![](images/2525d5562e903e719c91885bf534146e36e1caa4cd284843be212a94362949f2.jpg)

b  
Prototype exploration  
![](images/afa70201d2e0d641312195d9e6bcfee681ef2c8ad71924ce26e531d54529ddc8.jpg)

![](images/b1b70469483cc705e232842d032489040769378a011aa1672baf5105fc8db9f5.jpg)

![](images/729b7fa2a450bd0df881a5ff3a95cabdd1325d779849abcc0a8b1f0312c9c297.jpg)  
Fig. 1 | Overview of the ProtoMI framework for prototype-guided discovery of boron-containing electrolyte additives. a,b, Conceptual comparison between traditional molecular exploration and prototype-guided exploration. Conventional exploration (a) searches the molecular space largely through local trial-and-error optimization, whereas prototype-guided exploration (b) first identifies representative prototype regions from reported additives and subsequently explores structurally distinct regions of the molecular landscape through prototype-level knowledge transfer. c, Overview of the ProtoMI molecular discovery workflow. Experimentally reported boron-containing additives are first collected from the literature using a large-language-model-assisted data-mining pipeline to construct a curated molecular dataset. An unsupervised graph contrastive learning model then extracts prototypelevel chemical knowledge from reported additives. These prototypes are transferred to a large unlabeled molecular space by prototype-guided semi-supervised contrastive learning, yielding adapted prototype regions and recommended candidates. A knowledge-guided translation step further incorporates chemical feasibility and practical accessibility constraints to generate experimentally actionable candidates for validation.

## Results

## Dataset construction reveals source-target mismatch in boron-additive chemical space

We first formulated boron-containing electrolyte additive discovery as a sparsepositive molecular exploration problem. To define the target search space, we collected 179,977 boron-containing candidate molecules from PubChem<sup>31</sup>. In parallel, we constructed a positive dataset of 126 experimentally reported boron-containing electrolyte additives extracted from Web of Science articles using a large language model (LLM)-assisted heuristic pipeline (Fig. 2a and Methods M1). This design reflects a realistic discovery setting: successful additives are sparsely reported in the literature, whereas the accessible boron-containing chemical space is orders of magnitude larger

and remains largely unlabeled.

To evaluate whether unstructured literature could provide reliable positive molecular knowledge, we manually verified the extracted information across multiple battery-relevant fields. The extraction pipeline achieved high accuracy for chemically important categories, including 94% accuracy for additive identification, while requiring approximately 11 s per article and less than 60 CNY in total processing cost (Fig. 2b). These results indicate that LLM-assisted literature extraction can provide an efficient route to constructing a curated positive molecular set for downstream prototype learning. Overall dataset statistics are summarized in Supplementary Table S1.

The resulting datasets reveal a pronounced source-target distribution mismatch between literature-reported additives and the unlabeled boron-containing candidate space. Uniform manifold approximation and projection (UMAP)<sup>34</sup> based on Morgan fingerprint<sup>35</sup> representations shows that reported additives occupy restricted regions within the broader unlabeled molecular landscape rather than being uniformly distributed across it (Fig. 2c). Hierarchical clustering<sup>36</sup> dendrogram further suggests that the reported additives contain recurring molecular patterns rather than isolated successful examples (Supplementary Fig. S1). Consistently, the molecular-weight– synthetic accessibility $( \mathsf { S A } ) ^ { 3 7 }$ map shows that positive molecules are concentrated mainly in low-to-moderate molecular-weight regions with moderate SA scores, whereas the unlabeled candidates span a much broader and more heterogeneous chemical space (Fig. 2d).

Graph-level descriptors provide additional evidence for population-level differences between the positive and unlabeled sets. Reported additives and unlabeled candidates differ in molecular size and connectivity, as reflected by shifts in atom number, bond number, average degree, graph density and degree entropy (Fig. 2e and Methods M2). Node-feature composition analysis further reveals feature-level differences between the two populations, suggesting that reported additives are enriched in specific local chemical environments. Representative shifted node features include scaled van der Waals radius (N67), scaled covalent radius (N68), aromaticity indicator (N65), sp<sup>2</sup> hybridization (N59) and carbon atom identity (N00) (Fig. 2f and Supplementary Table S2).

Together, these analyses define both the challenge and the opportunity of datascarce additive discovery. Experimentally reported additives are too sparse and distributionally biased to support exhaustive or purely supervised screening, yet they exhibit non-random structural organization that may encode transferable chemical priors. We therefore next asked whether this organization could be learned as prototype-level knowledge and used to guide exploration of the broader unlabeled candidate space.

![](images/37a26027019e77cb69699948370033c2616c53d496b0205be596cbe9f5c78e75.jpg)  
b

![](images/5c3de1cd4a0d78eecbcef4975d734f3a8f9e768080b631ba28a4f5dfb82571fb.jpg)

![](images/cca040b44306c9f48a066bff48ccba230d7609fc7f7a259045a192019313673f.jpg)

d  
![](images/7b4b7004fc3a9792781855ce528cbf5650a9adaf9146d8b9594b3a46b018fa02.jpg)

e  
![](images/40aa81e2286a3319e484c28bec0347bfff20a1674e4f373a78ce1a820dfc6fc7.jpg)

f  
![](images/4087e188156a3e8c4b3cc284cb6e9ca59af1e8ed181a7601585018c5ab413ebb.jpg)  
Fig. 2 | Dataset construction and source-target mismatch in boron-additive chemical space. a, Workflow for constructing the molecular datasets. The unlabeled search space consists of 179,977 boron-containing candidate molecules collected from PubChem, whereas the positive dataset contains 126 literature-reported boron-containing electrolyte additives extracted from Web of Science articles using an LLM-assisted heuristic pipeline. b, Manual validation of the extraction pipeline across batteryrelevant fields, showing reliable extraction of chemically important information. $\bullet ,$ UMAP projection of Morgan fingerprint representations. Positive molecules are localized within restricted regions of the

unlabeled molecular landscape, while background hexagons represent the logarithmic density of unlabeled candidates. d, Molecular-weight and SA distribution of positive and unlabeled molecules. Positive additives are enriched in low-to-moderate molecular-weight regions with moderate SA scores, whereas unlabeled candidates cover a broader chemical space. Molecular weights are shown within 0- 6,000 g mol<sup>-1</sup>, as higher values are extremely rare (<0.1% of the dataset). e, Graph-level descriptor distributions reveal differences in molecular size (Atoms and Bonds), connectivity (Average degree) and topology (Graph density and Degree entropy) between the two molecular populations. f, Mirrored dumbbell plot showing normalized composition ratios of the most shifted node features, highlighting feature-level differences between positive and unlabeled molecules.

## Reported additives encode transferable molecular prototypes

The clustered organization of the positive dataset suggests that successful boroncontaining additives may share recurring structural principles rather than represent isolated molecular examples. To extract this organization, we applied an unsupervised graph contrastive learning (UCL) model<sup>38</sup> to learn prototype-level representations from the 126 reported additives. Reported molecules were converted into molecular graphs and expanded into augmented graph views using graph augmentation operations<sup>39-43</sup>, generating 756 augmented graph samples from the original positive set (Fig. 3a and Methods M3). Paired graph views were then encoded by a graph neural-network encoder and optimized with the InfoNCE loss to learn augmentation-invariant molecular representations. The encoder consisted of repeated GINEConv, GraphNorm, ReLU and dropout layers for hierarchical graph representation extraction (Fig. 3b).

We first compared different molecular representation methods using clustering quality and computational cost as evaluation metrics (Fig. 3d). Conventional fingerprint-based<sup>44</sup> and SMILES-based<sup>45</sup> representations produced relatively low silhouette scores<sup>46</sup> of 0.258-0.337, indicating limited separation of reported additive groups in representation space. Graph-based encoders, including graph convolutional networks (GCN)<sup>47</sup>, graph attention networks (GAT)<sup>48</sup> and graph isomorphism networks with edge features (GINE)<sup>49</sup>, substantially improved cluster separability, with silhouette scores of 0.751-0.832. Augmentation-enhanced graph representations further improved embedding organization relative to the corresponding non-augmented encoders. Among all tested methods, GINE+ achieved the highest silhouette scores using both K-Means<sup>50</sup> clustering and hierarchical clustering, with values of 0.853 and 0.881, respectively. These results indicate that augmentation-invariant graph representations capture hidden structural organization within the reported additive set.

Hierarchical clustering of the learned GINE+ representation space identified seven molecular prototypes (Fig. 3c and Supplementary Figs. S2,S3). These prototypes displayed non-uniform motif distributions (Fig. 3e), indicating that they are organized by distinct combinations of recurring boron-related molecular fragments. For example, P3 was enriched in oxalate-containing and tetra-alkoxy frameworks, whereas P2 preferentially contained fluorinated aromatic fragments. P4 was characterized by catechol-derived structures, while P6 and P7 were dominated by fluorinated alkoxy and tri-alkoxy frameworks, respectively. These motif-level differences suggest that the learned prototypes capture chemically meaningful structural preferences rather than arbitrary clusters (Supplementary Tables S3,S4).

To further interpret the prototype space, we annotated the learned clusters using literature-supported functional categories and representative molecules (Fig. 3c, Supplementary Fig. S4 and Table S5). P1 showed relatively balanced contributions from interphase formation, solvation regulation and polymer strengthening, represented by the boron-containing heterocycle DiOB-Py, which has been reported to act as a mild anion receptor and facilitate interfacial charge transfer<sup>25</sup>. P2 combined Lewis acidity, solvation regulation and interphase formation, represented by tris(pentafluorophenyl)borane, whose fluorinated aromatic framework supports anion reception and interfacial stabilization<sup>23</sup>. P3 displayed a strong interface-engineering signature, represented by Lithium difluoro(oxalato)borate (LiDFOB), which is known to form F- and B-rich interphases that suppress parasitic reactions and stabilize electrode surfaces<sup>26</sup>. P4-P6 showed complementary stabilization-oriented profiles, represented by lithium bis[1,2-benzenediolato(2-)-O,O']borate (LBBB), 3- Pyridinecarbonitrile BF (PCN) and tris(2,2,2-trifluoroethyl)borate (TTFEB), which illustrate weakly coordinating borate chemistry, Lewis acid-base interactions and fluorinated anion-receptor behavior<sup>51-53</sup>. P7 was associated with solvation regulation and polymer strengthening, represented by tributyl borate, which promotes salt dissociation and boron-rich interphase formation in polymerized electrolyte systems<sup>54</sup>.

These results show that the positive additive set can be converted into a prototype library that captures both recurring structural motifs and literature-supported functional tendencies. Importantly, the prototypes are learned without direct performance labels, making them suitable as transferable priors for candidate exploration under severe label scarcity. We therefore used this prototype library as the source knowledge for adaptation to the unlabeled boron-containing molecular space.

![](images/c518fde4fb8c0bd465bfd9c93d1a5dd259f92e575403c0c14384170c8d0b4d39.jpg)

![](images/670ab8c4c212fe088bb46c7e0641e3109d04900e163079fd7ab0e60f53b65cd0.jpg)

![](images/babdb902e09e4dc400ad9ce785b46d0031069062d01fc939c6f13dc9feef391b.jpg)  
Fig. 3 | Interpretable molecular prototypes emerge from reported boron-containing electrolyte additives. a, Workflow of the unsupervised graph contrastive learning model. Reported additives are converted from SMILES strings into molecular graphs and expanded into multiple augmented graph views using five graph augmentation strategies. Paired graph views are encoded by graph neuralnetwork encoders and projection heads, followed by InfoNCE optimization to learn augmentationinvariant molecular representations. b, Structure of the molecular graph encoder. The encoder consists

of repeated GINEConv, GraphNorm, ReLU and dropout layers, producing graph-level embeddings for clustering and prototype discovery. In this work, the repeated � equals 3. c, Functional interpretation of the learned prototype space. Seven chemically distinct prototypes are identified from reported additives. The surrounding panels show characteristic additive families, representative molecules and functional bias profiles across seven electrolyte-regulation mechanisms. d, Comparison of representation methods in terms of clustering quality (K-Means and hierarchical clustering) and computational cost. Silhouette scores quantify the separability of molecular clusters. Methods labeled with “+” indicate augmentation-enhanced representations. e, Relative composition of representative boron-related motifs across the seven learned prototypes. The motif-level distribution shows that different prototypes are associated with distinct substructural preferences, suggesting that the learned representation space captures chemically meaningful organization. See the shared-motif summary table in Supplementary Table S4.

## Prototype-guided adaptation transfers additive knowledge to unlabeled chemical space

Although the prototype library extracted from reported additives captures transferable chemical organization, it does not by itself determine how limited experimental resources should be allocated within the much larger unlabeled candidate space. To address this challenge, we developed a prototype-guided semi-supervised contrastive learning (PCL) model, as the adaptation stage of ProtoMI (Fig. 4a and Methods M4). The PCL module combines exponential moving average (EMA)<sup>55</sup> updating of prototype centroids, top-k prototype-relevant sample assignment and prototype decorrelation regularization. Together, these components are designed to transfer prototype-level knowledge from the positive additive domain to the unlabeled candidate domain while maintaining both recommendation quality and prototype diversity.

During training, the prototype centroids exhibited a characteristic three-stage trajectory, consisting of initial exploration, progressive refinement and eventual stabilization (Fig. 4b). The gradual reduction in prototype drift indicates that the model converges toward a stable prototype organization in the adapted molecular representation space. This behavior supports the use of adapted prototypes as anchors for candidate prioritization.

We next examined how prototype knowledge was transferred from the reported additive space to the unlabeled candidate space. The correspondence between original prototypes P1-P7 and adapted, new prototypes NP1-NP7 shows that adaptation is not a simple one-to-one copying process (Fig. 4c). Instead, each adapted prototype integrates contributions from multiple original prototype regions, indicating that the PCL model reorganizes literature-derived knowledge according to the structure of the target candidate space while preserving prototype-level continuity. Consistently, the cosine-similarity map between initial and adapted prototypes shows strong diagonal correspondence together with structured off-diagonal relationships (Supplementary Fig. S5). Motif analysis further shows that characteristic boroncontaining fragments are retained and redistributed across adapted prototypes, supporting chemically meaningful transfer rather than arbitrary re-clustering (Supplementary Fig. S6).

We then benchmarked the complete ProtoMI framework against multiple recommendation strategies (Table 1 and Fig. 4d). ProtoMI achieved high prototype similarity, assignment entropy and cluster separability, with values of $0 . 7 8 9 \pm 0 . 0 3 0$ n 0.723 and 0.976, respectively. This indicates that recommended molecules remain both structurally coherent (Supplementary Fig. S7) and broadly distributed across multiple prototype regions (Supplementary Fig. S8). The corresponding entropyweighted prototype similarity (Methods M5) reached 0.57 (or 57%), exceeding the competing recommendation pipelines. These results indicate that ProtoMI selects molecules that remain close to learned prototype regions while preserving coverage across multiple chemically distinct directions, thereby avoiding collapse into a few dominant structural families.

Ablation analysis further clarifies the contribution of each PCL component (Fig. 4e). Removing top-k assignment reduces the model’s ability to select high-confidence molecules most compatible with each prototype. Removing prototype decorrelation decreases inter-prototype separation and promotes redundancy among adapted prototype directions. Removing EMA updating destabilizes cross-domain adaptation by allowing abrupt prototype shifts during training. The performance degradation observed after removing these components indicates that effective prototype adaptation requires coordinated control of sample confidence, prototype diversity and transfer stability (Supplementary Fig. S9). The conclusion is further supported by additional statistics in Supplementary Table S6.

Finally, we evaluated whether prototype adaptation improves discovery efficiency using retrospective temporal validation. Across four temporal splits using 2017, 2019, 2021 and 2023 as cutoff years, ProtoMI recovered future-reported additives with recall rates of 8.8-20.0% while screening only 0.19-1.27% of the candidate space, corresponding to enrichment factors of 9.2-45.6 relative to random screening (Fig. 4f and Supplementary Table S7). Pareto-front analysis further shows that ProtoMI concentrates discovery efforts into a small fraction of the search space while preserving substantial recall, corresponding to effective search-space reductions of

79-517 folds $( \mathsf { F i g . 4 g ) }$ . These results demonstrate that sparse literature-derived prototype knowledge can be transferred into unlabeled chemical space to improve candidate prioritization under limited experimental budgets.

![](images/e85d9db6cd32d380f9fc67ae58809767d31655e6c75f3fa446d5b5af36ef4f88.jpg)  
C

![](images/e19c248c6f356b523ab72179325f9415a1e91f35614787a826a5d9fa8a5700b1.jpg)

![](images/3e29707c5428f1809c141c2c431bc88efc4da47f5d4a21120f6da6acb9b64a4c.jpg)

![](images/e8178bbb0eee88fedb13083d38dd39407791c7153705c43aafbabf07e10a1300.jpg)

f  
![](images/3d3bad1280892a4b9d573296e5ec32f86dbd9f5182ac148d6cdb577c133a0bf5.jpg)

e  
![](images/33c665fe932e1e1f54330362e53383cc71d3f3cb1f84e267f6bac1e16ca53200.jpg)

![](images/b4469e90bee21a7a9cd78dbea60f4a70eb66db3ffefb2a1ca2c83a7802f26059.jpg)  
Fig. 4 | Prototype adaptation transfers additive knowledge to unlabeled boron-containing chemical space. a, Workflow of the PCL module within ProtoMI. Molecular prototypes extracted from reported boron-containing electrolyte additives are used to guide representation learning and candidate recommendation in the unlabeled molecular space. b, Mean evolution of prototype drift during training

across 10 trials. Prototype centroids undergo an initial exploration stage (Stage I), followed by progressive refinement (Stage II) and eventual stabilization (Stage III), revealing the emergence of a converged prototype organization in the learned molecular representation space. c, Knowledge retention between original prototypes and adapted prototypes. The Sankey diagram shows the redistribution of structural motifs from reported-additive prototypes to newly formed prototypes in unlabeled molecular space. d, Entropy-weighted prototype similarity and prototype distribution of molecules selected by different recommendation pipelines. ProtoMI maintains high prototype consistency while preserving coverage across multiple prototypes. e, Ablation analysis of key components in the PCL module. Removing EMA updating, prototype decorrelation or top-k prototype selection degrades recommendation quality, highlighting their roles in stable prototype formation and effective knowledge transfer. f, Temporal validation using historical splits. Recall rates are compared with random screening, and the red line indicates the enrichment factor achieved under limited experimental budgets. g, Pareto front of discovery efficiency. Bubble size represents the number of recommended molecules, illustrating the trade-off between recall ability and experimental budget across different historical splits. “x/y” in the figure indicates the correct prediction of x molecules among the literature reported y molecules.

Table 1 | Recommendation performance of ProtoMI and baseline strategies
<table><tr><td>Category</td><td>Method</td><td>Similarity</td><td>Entropy</td><td>Silhouette</td></tr><tr><td>Random</td><td>Random selection</td><td> $0 . 5 2 1 \pm 0 . 1 4 7$ </td><td>0.529</td><td>0.596</td></tr><tr><td>Fingerprint</td><td>Morgan fingerprints</td><td> $0 . 4 9 7 \pm 0 . 1 7 7$ </td><td>0.836</td><td>0.408</td></tr><tr><td rowspan="2">Encoder</td><td>UCL encoder only</td><td> $0 . 0 8 5 \pm 0 . 0 1 0$ </td><td>0.349</td><td>-0.185</td></tr><tr><td>PCL encoder only</td><td> $0 . 7 2 8 \pm 0 . 0 4 5$ </td><td>0.404</td><td>0.968</td></tr><tr><td rowspan="2">Clustering</td><td>UCL encoder clustering</td><td> $0 . 0 7 9 \pm 0 . 0 2 1$ </td><td>0.260</td><td>0.786</td></tr><tr><td>PCL encoder clustering</td><td> $0 . 5 7 1 \pm 0 . 1 1 0$ </td><td>0.426</td><td>0.916</td></tr><tr><td>Prototype- guided</td><td>ProtoMI</td><td>0.789±0.030</td><td>0.723</td><td>0.976</td></tr><tr><td rowspan="2">Ablation</td><td>w/o EMA</td><td> $0 . 1 4 1 \pm 0 . 0 3 6$ </td><td>0.669</td><td>0.253</td></tr><tr><td>w/o decorrelation</td><td> $0 . 0 8 0 \pm 0 . 0 1 4$ </td><td>0.037</td><td>-0.224</td></tr><tr><td></td><td>w/o top-k</td><td> $0 . 2 5 7 \pm 0 . 0 0 0$ </td><td>0.000</td><td>N/A</td></tr></table>

Bold and gray-shaded values indicate the best and second-best performance, respectively.

## Knowledge-guided translation converts recommendations into testable candidates

Prototype adaptation reduced the original search space from 179,977 boroncontaining candidates to 30,800 ProtoMI-recommended molecules. However, many recommended molecules remained chemically impractical or experimentally inaccessible. To bridge the gap between computational prioritization and laboratory validation, we developed a knowledge-guided translation strategy that incorporates empirical additive-design constraints and practical accessibility requirements (Fig. 5a and Methods M6).

The translation workflow sequentially applied filters based on PubChem CID validity, molecular-weight range, synthetic accessibility, active-hydrogen exclusion, Chemical Abstracts Service (CAS) availability and practical cost. This process reduced the recommendation pool from 30,799 molecules to 68 supplier-accessible candidates, which were further prioritized according to purchasability, cost and prototype traceability (Supplementary Fig. S10 and Supplementary Table S8). Four low-cost commercially available molecules were selected for experimental consideration (Supplementary Table S9). This workflow separates the model-driven recommendation stage from the practical translation stage, ensuring that experimentally selected molecules are both chemically relevant and accessible under realistic resource constraints.

To assess whether prototype-level organization was preserved during translation, we projected the retained molecules onto the adapted representation landscape (Fig. 5b). Post-screened molecules remained distributed across multiple prototype regions, indicating that practical filtering did not collapse the recommendations into a single structural family. No commercially accessible molecules were retained from NP4 and NP7 after final filtering, although candidate molecules from these regions existed before post-screening. This suggests that their absence among final candidates reflects practical accessibility constraints rather than a lack of prototype coverage. Representative unselected molecules, UM#1-UM#3, are shown to illustrate structurally plausible molecules that were present in the unlabeled pool but not retained as final experimental candidates.

Together, these results show that knowledge-guided translation serves as a practical bridge between prototype-guided molecular exploration and experimental implementation. By incorporating feasibility and accessibility constraints after model recommendation, ProtoMI converts a large prioritized molecular pool into a small set of experimentally actionable candidates.

![](images/f00cd58833d16e79ee69e4f363d5c9068bb16d9b838c380174f7d94a878e90ef.jpg)

Fig. 5 | Knowledge-guided translation converts ProtoMI recommendations into experimentally actionable candidates. a, Post-screening workflow applied to molecules recommended by ProtoMI. Sequential filtering based on PubChem CID validity, molecular-weight range, synthetic accessibility, active hydrogen exclusion, CAS availability and practical cost considerations progressively reduced 30,800 recommended molecules to four experimentally selected candidates. Numbers below each step indicate the remaining candidate pool after filtering. b, Distribution of recommended molecules in the transferred molecular representation space. Colored regions denote 7 new prototype domains (NP1- NP7) formed after transferring prototype knowledge to the unlabeled boron-containing molecular space. Representative molecules associated with each new prototype are shown to illustrate the chemical motifs captured by the transferred prototype regions. Representative unlabeled molecules (UM#1- UM#3) are additionally shown from the broader molecular space to indicate structurally plausible molecules that were present in the unlabeled pool but not retained as final experimental candidates.

## Experimental validation in high-temperature LiFePO<sub>4</sub>||graphite cells

To validate whether a ProtoMI-recommended molecule could translate into measurable battery performance improvement, we selected one representative candidate, 4,4,5,5-tetramethyl-2-[10-(1-naphthyl)anthracen-9-yl]-1,3,2-dioxaborolane (TNDB; NP3#6), for electrochemical testing (Fig. 6a). $\mathsf { L i F e P O 4 } |$ |graphite cells were chosen because this chemistry is widely used in electric vehicles and stationary energy storage, while its cycling stability at elevated temperature remains limited by parasitic interfacial reactions and transition-metal crossover<sup>32,33</sup>. The additivecontaining electrolyte was prepared by introducing 1 wt% TNDB into the baseline electrolyte $( 1 \mathsf { M L i P F } _ { 6 }$ in ethylene carbonate (EC):dimethyl carbonate $( \mathsf { D M C } ) = 3 { : } 7 \ \mathsf { v } / \mathsf { v } )$ and tested at $5 5 ~ ^ { \circ } \mathsf { C }$ . Compared with the baseline electrolyte, the TNDB-containing cell showed improved capacity retention, with a 34.93% improvement at the $1 0 0 ^ { \mathrm { { t h } } }$ cycle (Fig. 6b). Because high-temperature degradation in $\mathsf { L i F e P O 4 }$ ||graphite cells is associated with electrolyte decomposition, SEI instability and Fe dissolution/deposition<sup>32,33</sup>, we next investigated how TNDB alters the interphase chemistry.

The chemical contribution of TNDB to SEI formation was first examined using Xray photoelectron spectroscopy (XPS). High-resolution B 1s spectra collected from graphite electrodes recovered from TNDB-containing cells showed B–F and B–O coordination environments after both formation and 100 cycles (Fig. 6c), indicating that TNDB-derived boron species are incorporated into the graphite interphase. XPS after formation further showed a small but detectable B signal only in the TNDB-containing cell, together with a higher F atomic fraction and a lower O atomic fraction relative to the baseline cell (Fig. 6d). These surface-composition changes suggest that TNDB contributes to a B/F-containing interphase and suppresses the accumulation of oxygen-rich decomposition products during early SEI formation.

F 1s XPS was then used to analyze fluorinated interphase species after formation (Fig. 6e). In both electrolytes, LiF remained the dominant fluorinated component, while the TNDB-containing electrolyte showed a modified distribution of LiF, $\mathsf { L i } _ { \mathsf { x } } \mathsf { P F } _ { \mathsf { y } }$ and $\mathsf { L i x P O F } _ { \mathsf { y } }$ species. Specifically, the LiF contribution remained high in the TNDBcontaining cell, whereas the $\mathsf { L i x P O F } _ { \mathsf { y } }$ contribution increased relative to the baseline electrolyte. This indicates that TNDB does not simply increase the total LiF fraction at the outermost surface, but instead modifies the formation of F- and P/O-containing inorganic interphase species. Together with the B 1s results, these XPS signatures support the formation of a TNDB-modified inorganic interphase on graphite.

The organic and inorganic carbonate components of the graphite SEI were further analyzed by C 1s XPS peak deconvolution (Fig. 6f and Supplementary Fig. S11). Following common assignments for graphite SEI analysis, the C 1s spectra were fitted into C-C, C-O, C=O, ${ \tt 2 0 C O 2 L i }$ and $\mathsf { L i } _ { 2 } \mathsf { C O } _ { 3 }$ components<sup>56,57</sup>. In the baseline electrolyte, oxygenated organic SEI species, including C-O, C=O and ${ \mathsf { R O C O } } _ { 2 } { \mathsf { L i } }$ , accounted for 38% of the fitted C 1s area after formation and remained at 32% after 100 cycles. In contrast, the TNDB-containing electrolyte showed lower oxygenated organic contributions, decreasing from 28% after formation to 26% after 100 cycles. The $\mathsf { L i } _ { 2 } \mathsf { C O } _ { 3 }$ contribution after 100 cycles was also reduced from 4% in the baseline cell to 1% in the TNDBcontaining cell. These results indicate that TNDB suppresses the accumulation of carbonate-derived organic species and inorganic carbonate residues during hightemperature cycling.

Time-of-flight secondary ion mass spectrometry (ToF-SIMS) provided depthresolved evidence for TNDB-mediated SEI reconstruction (Fig. 6g and Supplementary Fig. S12). Representative negative secondary-ion fragments, including ${ \mathsf { C H } } _ { 3 } { \mathsf { C O } } ^ { - } , { \mathsf { P O } } _ { 2 }$ $\mathsf { C O } _ { 3 } ^ { - }$ and ${ \mathsf { L i F } } _ { 2 } { } ^ { - }$ , were used to visualize solvent-derived organic fragments, $\mathsf { L i P F } _ { 6 ^ { - } }$ derived P-O species, carbonate-containing residues and LiF-related inorganic species, respectively<sup>58,59</sup>. In the baseline electrolyte, $C H _ { 3 } C O ^ { - }$ and $\mathsf { C O } _ { 3 } \mathsf { ^ { - } }$ fragments extended broadly through the analyzed interphase volume, consistent with heterogeneous accumulation of carbonate-derived decomposition products. In contrast, graphite electrodes cycled with TNDB showed attenuated oxygenated organic fragments and a reorganized distribution of ${ \mathsf { L i F } } _ { 2 } { } ^ { - }$ and $\mathsf { P O } _ { 2 } ^ { - }$ species. These depth-resolved signatures are consistent with a graphite interphase that is less dominated by organic carbonate decomposition and contains redistributed salt-derived inorganic components<sup>60,61</sup>.

Additive-derived boron incorporation was further visualized using x-z crosssectional ToF-SIMS maps of the $\mathsf { B } ^ { - }$ fragment in graphite electrodes cycled with TNDB (Supplementary Fig. S13). A sparse and discontinuous $\mathsf { B } ^ { - }$ signal was observed after formation, whereas a more extensive $\mathsf { B } ^ { - }$ distribution appeared after 100 cycles, indicating progressive incorporation and retention of boron-containing species within the SEI. Sputter-depth profiles of representative SEI fragments further showed that $C H _ { 3 } C O ^ { - }$ signals were attenuated in the TNDB-containing electrolyte, supporting suppressed accumulation of solvent-derived organic decomposition products (Supplementary Fig. S14). The $\mathsf { F e O ^ { - } }$ depth profile was also strongly suppressed after formation and remained lower after extended cycling in the TNDB-containing cell (Supplementary Fig. S15). Because transition-metal deposits on graphite can catalyze electrolyte reduction and destabilize the anode interphase<sup>62</sup>, the reduced $\mathsf { F e O ^ { - } }$ signal suggests that TNDB-derived interphase chemistry mitigates Fe contamination of the graphite surface.

SEM and EDS analyses provided complementary information on surface morphology and Fe deposition. SEM images show the morphology of graphite electrodes recovered after formation and after 100 cycles with and without TNDB (Supplementary Fig. S16). EDS mapping further revealed pronounced Fe accumulation on graphite electrodes cycled in the baseline electrolyte, with the Fe fraction in selected mapping regions increasing from 2.06 wt% after formation to 3.61 wt% after 100 cycles (Fig. 6h). In contrast, graphite electrodes cycled with TNDB showed substantially lower Fe accumulation, with corresponding values of 0.23 wt% after formation and 1.04 wt% after 100 cycles. These values correspond to 89% and 71% lower Fe contents, respectively, relative to the baseline electrolyte. The reduced Fe deposition may help suppress Fe-catalyzed electrolyte decomposition and interphase deterioration during high-temperature cycling.

a  
![](images/51683290742cd124e87268e675512ee55cdfaf49b80dae4257a614928fdd8d09.jpg)

![](images/a7965259c7e94e1b61d9be382fcd6bb89ce1841f1e36e551c2001c7a3e130c8b.jpg)

![](images/463939589bd41f7c15d32543257f04adf1e54f1ae46d7c24b7a3581bad932a74.jpg)

![](images/6e16da0482cce499dd70e471b341a53fd1bf077593dbba7cff92c67094b4132b.jpg)

![](images/c9b1b493931be1348d7e2b836c256000696ae76239f49e7822b8d0efbe9fcd02.jpg)

![](images/eeea9091ec263092eea80f9d3f414442a4681844c9c1cc12c3459e7e155239c5.jpg)

g  
![](images/ede9570e124f8cb6c862a6fe53f66cd62aaf6b750428639b41489f0e845f1e5c.jpg)  
h

$$
\mathsf { P O } _ { 2 } \cdot
$$

![](images/0e19683b7fe7e5f4c05ecf69f38c853114106bb6e39e07c34332d0d76f7be766.jpg)  
Fig. 6 | Interphase regulation by the TNDB under $5 5 \%$ cycling. a, Molecular structure of TNDB shown as a ball-and-stick model; grey, C; red, O; pink, B; and white, H. b, Capacity retention of $\mathsf { L i F e P O 4 }$ ||graphite coin cells containing the baseline or TNDB-containing electrolyte during cycling at $5 5 ^ { \circ } \mathsf { C }$ . The shadings indicate the deviation of two reproducible cells. c, High-resolution B 1s $_ { \tt X P S }$ spectra of graphite electrodes recovered from TNDB-containing cells after formation and 100 cycles, with the fitted components assigned to B-F and B-O. d, XPS-derived relative atomic concentrations of C, O, F, P and B on graphite electrodes recovered from baseline and TNDB-containing cells after formation. e, F 1s XPS spectra of graphite electrodes recovered from cells with and without TNDB after the formation, with the fitted components assigned to LiF, $\mathsf { L i } _ { \mathsf { x } } \mathsf { P F } _ { \mathsf { y } } , \mathsf { L i } _ { \mathsf { x } } \mathsf { P O F } _ { \mathsf { y } } .$ f, Relative C 1s peak area fractions of $\mathsf { C - C } ,$ C-O, C-O, ROCO2Li and Li2CO3 for graphite electrodes recovered from baseline and TNDB-containing cells after formation and 100 cycles. g, Merged three-dimensional ToF-SIMS compositional renderings of $C H _ { 3 } C O ^ { - }$ (red), $\mathsf { P O } _ { 2 } ^ { - }$ (blue), $\mathsf { C O } _ { 3 } \bar { }$ (yellow) and ${ \mathsf { L i F } } _ { 2 } { } ^ { - }$ (green) fragments in the corresponding graphite

$$
C H _ { 3 } C O ^ { - }
$$

interphases. h, Representative Fe EDS maps (inset) and corresponding semi-quantitative Fe contents of the graphite electrodes.

Operando Fourier transform infrared spectroscopy (FTIR) was collected during the formation step at $5 5 ~ ^ { \circ } \mathsf { C }$ to monitor electrolyte and interphase evolution under operating conditions (Fig. 7a,b). The upper panels show time-resolved absorbance spectra, while the lower panels show difference spectra, $\Delta A ( t ) = A ( t ) - \Delta ( t _ { 0 } )$ , where t is the time and t<sub>0</sub> corresponds to the beginning of charge. In the baseline electrolyte, the difference spectra show a gradual increase in the organic semi-carbonate region. Bands around $1 6 4 0 { - } 1 6 7 0 \ \mathsf { c m } ^ { - 1 }$ can be assigned to lithium semi-carbonate and lithium alkyl carbonate species, such as lithium methyl carbonate and lithium ethylene dicarbonate, according to previous operando infrared studies<sup>63,64</sup>. Quantitative integration of this region shows a faster growth of organic semi-carbonate species in the baseline cell than in the TNDB-containing cell $( \mathsf { F i g . 7 c } )$ . At the end of charge, the integrated response was $0 . 2 5 \mathsf { a . u . c m ^ { - 1 } }$ for the baseline electrolyte and $0 . 1 7 \mathsf { a . u . c m ^ { - 1 } }$ for the TNDB-containing electrolyte, consistent with suppressed formation of carbonate-derived organic SEI species.

The $\mathsf { P F } _ { 6 } - \mathsf { r e l a t e d }$ P-F stretching region also evolved differently in the two electrolytes. We use the integrated $\mathsf { P F } _ { 6 } { } ^ { - }$ response between $8 2 0 { - } 8 8 0 ~ \mathsf { c m } ^ { - 1 }$ to track $\mathsf { P F } _ { 6 }$ ⁻-related interfacial evolution. The baseline electrolyte developed a positive endpoint response in this region, whereas the TNDB-containing electrolyte showed a negative end-point response (Fig. 7d). This contrast indicates that TNDB changes the direction and magnitude of $\mathsf { P F } _ { 6 } - \mathsf { r e l a t e d }$ spectral evolution during formation. When considered together with the XPS and ToF-SIMS results, the operando FTIR data suggests that TNDB suppresses carbonate-solvent decomposition while modifying salt-derived interfacial chemistry.

Overall, the post-mortem and operando evidence supports a coherent interphaseregulation mechanism for improved high-temperature cycling performance. TNDB participates in graphite interphase construction by introducing B-containing species, modifying F-, P-, and O-containing inorganic components, suppressing carbonatederived organic SEI accumulation and reducing Fe deposition on graphite. These coupled effects provide a plausible explanation for the improved capacity retention of $\mathsf { L i F e P O 4 } |$ |graphite cells cycled at $5 5 ^ { \circ } \mathsf { C }$ with the ProtoMI-recommended TNDB additive.

![](images/dd32397f52f4203f9abfd22c41edcd0b0fd89429021c5886ca40f6178100a1f1.jpg)

![](images/9851dc298fd6f53493f63c7290ec0b80300845f902c2c6dadaa04e76ea6f7c84.jpg)

![](images/de329347b7bd24f8495e8f3abcbc3616adadf052f5aeef1a546f49171ead7f8b.jpg)

d  
![](images/1afaf31067a3c0563415ee3a5a1ed9b3b362640313a1ed64e82d04f093954ff1.jpg)  
Fig. 7 | Operando FTIR analysis of electrolyte and interphase evolution during $5 5 \%$ formation. $\mathsf { a } , \mathsf { b } ,$ Time-resolved absorbance spectra (top) and corresponding difference spectra, $A ( t ) { - } A ( t _ { 0 } )$ (bottom), collected during the formation-charge step at $5 5 ^ { \circ } \mathsf { C }$ for $\mathsf { L i F e P O 4 } |$ |graphite cells containing the baseline electrolyte (a) and the TNDB-containing electrolyte (b). The dashed lines delimit the $1 \bar { 6 } 4 0 - 1 6 7 0 \mathsf { c m } ^ { - 1 }$ region associated with organic semi-carbonate species. The P-F stretching region of $\mathsf { P F } _ { 6 } { } ^ { - }$ is indicated at approximately $8 2 0 { - } 8 8 0 \mathsf { c m } ^ { - 1 }$ . Identical color scales are used for direct comparison. $\bullet ,$ Time evolution of the integrated absolute difference absorbance within the organic semi-carbonate region with and without TNDB. ${ \mathfrak { d } } ,$ Integrated spectral changes for organic semi-carbonate and the $\mathsf { P F } _ { 6 } { } ^ { - }$ P-F stretching region at the end of the $1 ^ { \mathsf { s t } }$ charge.

## Discussion and conclusion

In this work, we develop ProtoMI as a molecular discovery framework for data-scarce settings in which experimentally validated molecules are sparse and most candidate molecules remain unlabeled. Rather than treating additive discovery as a conventional supervised property-prediction problem, ProtoMI re-formulates it as an efficient exploration problem. Sparse literature-derived successes are converted into transferable prototype representations, adapted to a broader unlabeled candidate space and translated into experimentally actionable molecules through practical chemical constraints. This workflow allows limited experimental resources to be concentrated on chemically promising regions rather than distributed across the full candidate space.

A central feature of ProtoMI is that knowledge transfer occurs at the prototype level rather than at the level of individual molecules. Conventional similarity-based recommendation strategies tend to remain close to previously reported compounds and may therefore restrict exploration to local molecular neighborhoods. In contrast, ProtoMI learns higher-order organization from groups of reported additives and transfers this organization across source-target distribution mismatch. The preservation and redistribution of prototype identities after adaptation suggest that the framework captures transferable chemical relationships instead of simply memorizing known molecular structures. This prototype-centric strategy provides a practical route to extending sparse literature knowledge into structurally broader regions of chemical space.

The experimental validation demonstrates how ProtoMI recommendations can be connected to battery performance. As a representative candidate, TNDB improved the high-temperature cycling stability of $\mathsf { L i F e P O 4 }$ ||graphite cells relative to the baseline electrolyte. XPS, ToF-SIMS, EDS mapping and operando FTIR analyses indicate that TNDB incorporates B-containing species into the graphite interphase, modifies F/P/Ocontaining inorganic SEI components, suppresses carbonate-solvent decomposition, alters $\mathsf { P F } _ { 6 } { } ^ { - }$ -related spectral evolution and reduces Fe deposition on graphite. These coupled interfacial effects provide a plausible explanation for the improved capacity retention observed at $5 5 ~ ^ { \circ } \mathsf { C }$ . At the same time, the present experimental validation should be interpreted as a proof of concept for one representative ProtoMIrecommended candidate rather than exhaustive validation of all recommended molecules.

More broadly, this study highlights the importance of discovery efficiency in scientific problems where labels are scarce, and candidate spaces are large. Under such conditions, maximizing predictive accuracy for individual molecules may be less practical than improving the allocation of limited experimental resources. Retrospective temporal validation shows that literature-derived prototype knowledge can enrich future-reported additives while screening only a small fraction of the candidate space. This suggests that prototype-guided exploration can complement supervised prediction, physics-informed modeling and active-learning workflows by serving as an upstream candidate-prioritization strategy.

Several limitations remain. The current implementation focuses on boroncontaining electrolyte additives and relies primarily on structural molecular representations. It does not yet explicitly incorporate electrochemical stability, reaction pathways, solvation structures, synthesis routes or cell-level performance labels. In addition, the translation stage depends on empirical filters and commercial availability constraints, which may bias the final experimentally selected candidates. Future studies could integrate physics-informed descriptors, quantum-chemical calculations, synthesis-planning tools, multimodal molecular representations and active experimental feedback to further improve recommendation quality and practical accessibility. With these extensions, prototype-guided knowledge transfer may provide a general route for molecular discovery under severe label scarcity.

## Methods

M1. Data collection. For the searching space molecules dataset, we used keywords (Borane, Borate, Boric, Boron, Boronic, Boronate, Additive, Electrolyte) to inquire the boron-related molecules and their molecular information, including formula, SMILES, fingerprint, topological, weight, and heavy atom. For the positive samples dataset, we used the heuristic extraction process to obtain the literature information of each reported boron-containing electrolyte additive molecule. The details of the LLMassisted data-mining pipeline is provided in Supplementary Fig. S17. Finally, we get 179,977 boron-containing molecules as the searching space molecules dataset and 126 reported boron electrolyte additive molecules as the positive samples dataset.

M2. Graph-level descriptors. To characterize the global topological properties of molecules, several graph-level descriptors were calculated, including atom number (�), bond number (�), average degree (�<sup>̅</sup>), graph density (�), and degree entropy (�). The average degree was defined as

$$
\bar { k } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } k _ { i } = \frac { 2 E } { N }\tag{1}
$$

where $k _ { i }$ denotes the degree of node �. Graph density, which measures the fraction of existing bonds relative to the maximum possible number of bonds, was calculated as

$$
D = \frac { 2 E } { N ( N - 1 ) }\tag{2}
$$

To quantify the heterogeneity of node connectivity, the degree entropy was

computed using the Shannon entropy of the degree distribution,

$$
H = - \sum _ { k } p ( k ) \log p ( k )\tag{3}
$$

where $p ( k )$ denotes the probability that a node has degree � . These descriptors collectively characterize molecular size, connectivity and topological complexity.

M3. Unsupervised graph contrastive learning (UCL) model. To extract prototype knowledge from experimentally reported boron electrolyte additive molecules, an unsupervised graph contrastive learning strategy was first applied to all reported boron electrolyte additive molecules. Each molecule was represented as a graph $R G _ { i } =$ $( R V _ { i } , R E _ { i } )$ , where node features describe atomic attributes and edge features describe bond-level information. During training, two views of each molecular graph were constructed for contrastive learning. The first view preserves the original structure, while the second view is generated via stochastic edge perturbation:

$$
R G _ { i } ^ { ( 1 ) } = R G _ { i } ,\tag{4}
$$

$$
R G _ { i } ^ { ( 2 ) } = \big ( R V _ { i } , \big ( R E _ { i } \setminus R E _ { i } ^ { d e l } \big ) \cup R E _ { i } ^ { a d d } \big ) ,\tag{5}
$$

$$
\left| R E _ { i } ^ { d e l } \right| = \left| R E _ { i } ^ { a d d } \right| = \left| \frac { \rho | R E _ { i } | } { 2 } \right|\tag{6}
$$

where $R E _ { i } ^ { d e l }$ and $R E _ { i } ^ { a d d }$ denote the sets of removed and newly added edges, respectively, and $\rho = 0 . 1$ controls the perturbation ratio. The two graph views were then encoded by a three-layer graph neural network (GNN). The layer-wise propagation can be formulated as:

$$
H ^ { ( l ) } = D r o p o u t \left( R e L U \left( G r a p h N o r m \left( G I N E ^ { ( l ) } \big ( H ^ { ( l - 1 ) } , R E _ { i } \big ) \right) \right) \right) , l = 1 , 2 , 3\tag{7}
$$

where $H ^ { ( 0 ) } = R X _ { i }$ denotes the initial node feature matrix and � represents the layer number of this GNN encoder. The graph-level representation was obtained via global mean pooling:

$$
h _ { R G _ { i } } = f _ { \theta } ^ { U S L } ( R G _ { i } ) = M e a n P o o l \bigl ( H ^ { ( 3 ) } \bigr ) = \frac { 1 } { R V _ { i } } \sum _ { v \in R V _ { i } } h _ { v } ^ { ( 3 ) }\tag{8}
$$

where $h _ { v } ^ { ( 3 ) }$ represents the embedding of node after the third layer GNN, and $f _ { \theta } ^ { U S L }$ is the three-layer encoder for this unsupervised learning stage. To optimize the representation space, the graph embeddings were further mapped by projection heads $g _ { 1 } ( \cdot )$ and $g _ { 2 } ( \cdot )$

$$
z _ { i } ^ { ( 1 ) } = g _ { 1 } \Big ( h _ { R G _ { i } } ^ { ( 1 ) } \Big ) , z _ { i } ^ { ( 2 ) } = g _ { 2 } \Big ( h _ { R G _ { i } } ^ { ( 2 ) } \Big )\tag{9}
$$

The use of separate projection heads follows standard contrastive learning practice to improve representation quality. The model was trained using an InfoNCE contrastive loss (Method M10), which encourages embeddings from two augmented views of the same molecule to be close, while separating them from other molecules in the batch. After each training epoch, molecular embeddings were L2-normalized and clustered using hierarchical clustering with outlier filtering, average linkage and cosine distance (Method M9). The optimal number of clusters was determined by maximizing the average silhouette score (Method M5) over a predefined range. The model achieving the highest silhouette score was selected as the final representation model and used to define molecular prototypes for subsequent prototype-guided learning.

M4. Prototype semi-supervised contrastive learning model. To transfer prototype knowledge from experimentally reported boron electrolyte additive molecules to the unlabeled molecular space, we developed a prototype semi-supervised contrastive learning model. First, we use the pre-trained molecular representation model (Method M3) to generate prototype-level embeddings for all experimentally reported additive molecules $h _ { R G _ { i } }$ , and followed by a randomly initialized projection head which is motivated by the need to improve robustness and avoid representation bias induced by the contrastive pretraining objective:

$$
\pmb { z } _ { i } = N o r m \left( g ^ { \prime } \left( f _ { \theta } ^ { U S L } ( R G _ { i } ) \right) \right)\tag{10}
$$

where $g ^ { \prime } ( \cdot )$ denotes the projection head. Hierarchical clustering with average linkage and cosine distance was then applied to the normalized positive-sample embeddings. The optimal number of clusters was determined by comparing clustering quality across candidate cluster numbers. Each cluster was regarded as a molecular prototype group, and the corresponding prototype label table �� was assigned to each positive additive:

$$
R L = { \mathrm { H i e r a r c h i c a l C l u s t e r } } ( \mathbf { z } _ { i } )\tag{11}
$$

Next, positive additives and unlabeled candidate molecules were jointly used to train the prototype semi-supervised contrastive learning model. At each epoch, prototype centroids were first recalculated from the positive additives according to their prototype labels. For prototype � , the centroid was computed as the mean representation of positive samples assigned to this prototype:

$$
p _ { n , n e w } ^ { ( t ) } = N o r m \left( \frac { 1 } { | \mathcal { P } _ { n } | } \sum _ { i \in \mathcal { P } _ { n } } \mathbf { z } _ { i } ^ { ( t ) } \right)\tag{12}
$$

where $\mathcal { P } _ { n }$ denotes the set of positive additives assigned to prototype �, and $\pmb { z } _ { i } ^ { ( t ) }$ is the normalized embedding produced by the current model at epoch �. To avoid unstable prototype shifts during semi-supervised adaptation, the prototype centroids were updated using an exponential moving average<sup>55</sup>:

$$
{ p _ { n } } ^ { ( t ) } = m p _ { n } ^ { ( t - 1 ) } + ( 1 - m ) p _ { n , n e w } ^ { ( t ) } , m = 0 . 9 9 9\tag{13}
$$

where $m$ is the momentum coefficient. This update allows the prototypes to gradually adapt from the positive-only representation space toward the broader unlabeled molecular space while preserving the original prototype knowledge extracted from experimentally reported additives.

At the beginning of the model training, a new GNN encoder and projection head were initialized and trained using both positive and unlabeled molecular graphs. For each mini-batch, the similarity between molecular embeddings and prototype centroids was calculated by cosine similarity. For each prototype, the top-k molecules with the highest similarity to that prototype were selected as prototype-relevant core samples. The embeddings of the selected molecules can be expressed in:

$$
\pmb { z } _ { i } = \frac { g ^ { \prime \prime } \left( f _ { \theta } ^ { P C L } ( G _ { i } ) \right) } { \left\| g ^ { \prime \prime } \left( f _ { \theta } ^ { P C L } ( G _ { i } ) \right) \right\| _ { 2 } } ,\tag{14}
$$

$$
s _ { i n } = s i m ( { z } _ { i } , { p } _ { n } ) ,\tag{15}
$$

$$
\mathcal { I } _ { n } ^ { ( K ) } = \mathrm { T o p K } ( \{ s _ { 1 n } , s _ { 2 n } , \dots , s _ { N n } \} ) ,\tag{16}
$$

$$
{ \mathcal { Z } } _ { n } = \left\{ \mathbf { z } _ { i } \mid i \in { \mathcal { I } } _ { n } ^ { ( K ) } \right\} .\tag{17}
$$

where $G _ { i } \in \{ R G _ { i } , U G _ { i } \}$ denotes either a reported positive molecular graph or an unlabeled molecular graph; $f _ { \theta } ^ { P C L }$ is the newly initialized GNN encoder; $g ^ { \prime \prime } ( \cdot )$ is the newly initialized projection head; $\mathbf { z } _ { i }$ is the normalized embedding of molecule � ; $p _ { n }$ denotes the centroid of the � -th prototype; $s _ { i n }$ is the cosine similarity between molecule � and prototype � ; � is the number of molecules in the mini-batch; ${ \mathcal { I } } _ { n } ^ { ( K ) }$ represents the index set of the top-k molecules most similar to prototype $n ;$ and ${ \mathcal { Z } } _ { n }$ denotes the corresponding set of selected prototype-relevant embeddings. In this work, $K = 2 5$

This strategy restricts contrastive optimization to molecules that are most relevant to each prototype, thereby reducing the influence of low-confidence unlabeled samples and preventing noisy candidates from dominating the representation learning process. For prototype �, the selected molecules were optimized using a prototype contrastive loss (Method M10). To encourage different prototypes to encode distinct molecular patterns, a prototype decorrelation regularization term was introduced (Method M10). The model was optimized using Adam with weight decay. After each epoch, the learned embeddings were evaluated on the held-out positive and unlabeled test samples. The silhouette score (Method M5) based on cosine distance was used to quantify the separation quality of the learned prototype-aware representation space. The model checkpoint with the highest silhouette score was selected as the best prototype semi-supervised contrastive learning model.

M5. Evaluating metrics. The silhouette score was used to assess the quality of the learned embedding space and the separation of prototype clusters. For each sample, the silhouette coefficient compares its average intra-cluster distance �(�) with the minimum average distance to the nearest neighboring cluster �(�), defined as:

$$
S i l h o u t t e S c o r e = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { b ( i ) - a ( i ) } { \operatorname* { m a x } \bigl ( a ( i ) , b ( i ) \bigr ) }\tag{18}
$$

The score ranges from −1 to 1, where higher values indicate more compact clusters and better inter-cluster separation. In this study, the silhouette score was computed using cosine distance for consistency with the clustering metric. The optimal number of clusters was selected by maximizing the average silhouette score. Furthermore, it was used as an indicator of representation quality, where higher scores reflect more discriminative and well-structured molecular embeddings.

To evaluate both prototype similarity and prototype coverage, an entropyweighted prototype similarity (EWPS) metric was defined. Each recommended molecule was assigned to the prototype centroid with the highest cosine similarity, and the average maximum similarity was calculated as:

$$
\bar { s } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } S _ { i }\tag{19}
$$

where $s _ { i }$ denotes the maximum cosine similarity between molecule � and all prototype centroids. Prototype coverage was quantified using the normalized assignment entropy:

$$
H _ { n o r m } = \frac { - \sum _ { k = 1 } ^ { K } p _ { k } l n p _ { k } } { l n K }\tag{20}
$$

where $p _ { k }$ is the fraction of molecules assigned to prototype �. The final EWPS metric was defined as:

$$
E W P S = \bar { s } * H _ { n o r m }\tag{21}
$$

Higher EWPS values indicate that the recommended molecules are both highly representative of the learned prototypes and broadly distributed across prototype categories.

To evaluate the ability of ProtoMI to recover future-reported additives, a temporal validation strategy was adopted. Recall was defined as:

$$
R e c a l l = \frac { N _ { h i t } } { N _ { h i d d e n } }\tag{22}
$$

where $N _ { \mathrm { h i t } }$ is the number of hidden positive molecules recovered by the recommendation list and $N _ { \mathrm { h i d d e n } }$ is the total number of hidden positives. Higher recall values indicate a stronger capability for identifying future experimentally reported additives.

To quantify the efficiency improvement over random screening, the enrichment factor (EF) was calculated as:

$$
E F = \frac { N _ { h i t } / N _ { r e c } } { N _ { h i d d e n } / N _ { c a n d i d a t e } }\tag{23}
$$

where $N _ { \mathrm { r e c } }$ is the number of recommended molecules and $N _ { \mathrm { c a n d i d a t e } }$ is the size of the candidate pool. EF measures how many times the recommendation strategy outperforms random selection, with larger values indicating more efficient prioritization of experimentally promising molecules.

M6. Translation strategy. To further improve experimental feasibility, we designed a knowledge-guided post-screening pipeline based on empirical chemical constraints, as illustrated in Fig. 5a. Specifically, six sequential filtering criteria were applied. First, CID filtration was performed by retaining only molecules with PubChem compound IDs (CID) smaller than 100,000,000, as larger IDs typically correspond to composite entries rather than single molecular entities. Second, molecular weight filtering was conducted by constraining candidates within the empirical range observed in reported boron-containing additives (26.811–1455.489). Third, synthesizability was evaluated using the SA score, and only molecules within the dataset-derived range (2.012–6.319) were preserved. Fourth, active hydrogen exclusion was implemented via SMiles

ARbitrary Target Specification (SMARTS)-based functional group matching<sup>65</sup> to remove molecules containing labile protic hydrogens, which are known to negatively affect electrolyte stability and battery performance. Fifth, CAS availability was required to ensure that selected molecules are commercially accessible. Finally, considering practical cost constraints, we manually selected four molecules from 68 commercially available candidates for experimental validation.

M7. Molecular representation. In this paper, we used fingerprints, SMILES, and graphs to represent molecules and used the augmented molecular graph representation (Method M8) as the best performance molecular representation method for the model training stage. For fingerprints, we used the rdFingerprintGenerator.GetMorganGenerator() function in the RDKit Python package to generate fingerprints for all molecules with radius=2 and fpSize=1024. For SMILES, most SMILES can be obtained from PubChem, OPSIN, and the literature, while some molecules are lacking, and we used ChemDraw software to generate them manually. For graphs, we converted the SMILES into the molecular graph by encoding atom elements, the number of heavy neighbors, formal charge, hybridization types, whether it is in a ring, whether it is aromatic, scaled van der Waals radius, and scaled covalent radius information as the node features, and bond types, bond is conjugated, bond is in ring, bond direction, and bond is aromatic information as the edge features. All numeric features have been normalized to the float data type, and all string features are one-hot encoded. See the Supplementary Tables S2 and S10 for details.

M8. Graph data augmentation. To enhance the robustness of representation learning under limited labeled data, we employ a multi-scale graph augmentation strategy that perturbs molecular graphs at the feature, structure, and topology levels. At the feature level, node attributes are augmented through random feature shuffling<sup>42</sup> and stochastic noise masking<sup>39</sup>. Feature shuffling randomly permutes a subset of node feature dimensions across nodes, while noise masking injects Gaussian perturbations into randomly selected feature entries. These operations simulate uncertainty and variability in molecular descriptors. At the structural level, graph topology is modified through node dropping<sup>43</sup> and edge perturbation<sup>40</sup>. Node dropping randomly removes a subset of nodes and their associated edges, generating subgraph-level variations. Edge perturbation is performed using a weighted sampling strategy, where edges are probabilistically removed or added based on node degree and shortest-path distance, encouraging the model to learn robust local structural patterns. At the global topology level, we further introduce a Personalized PageRank (PPR)-based augmentation<sup>41</sup>. A PPR diffusion matrix is computed to capture higher-order connectivity, and edges are selectively removed or added according to their PPR scores. In particular, the diffusion matrix is computed to capture higher-order connectivity, defined as:

$$
P = \alpha \sum _ { k = 0 } ^ { K } ( 1 - \alpha ) ^ { k } \hat { A } ^ { k }\tag{24}
$$

where $\hat { A } = D ^ { - \frac { 1 } { 2 } } A D ^ { - \frac { 1 } { 2 } }$ is the normalized adjacency matrix, � is the degree matrix with $\begin{array} { r } { D _ { i i } = \sum _ { j } A _ { i j } } \end{array}$ , � is the adjacency matrix encoding the bonding topology of the molecular graph, $\alpha$ is the teleport probability, and � is the truncation depth. The diffusion score $P _ { i j }$ quantifies higher-order topological relationships between node � and node $j$ . Based on this score, edges with lower $P _ { i j }$ values are preferentially removed, while non-edge pairs with higher $P _ { i j }$ values are more likely to be added.

M9. Hierarchical clustering algorithm with outlier filtering. To identify representative prototype groups from experimentally reported positive additives, hierarchical clustering was performed on the learned molecular graph embeddings using average linkage with cosine distance. Different clustering solutions were generated by varying the number of clusters from 3 to a predefined maximum value, max\_cluster=10, and the optimal cluster number (7) was selected according to the silhouette score (Method M5) calculated with cosine distance. To improve the robustness of prototype discovery, a cluster-wise outlier filtering strategy was applied for each candidate clustering solution. For each cluster, the centroid was calculated as the mean embedding of all samples assigned to that cluster. The cosine similarity between each sample embedding and its corresponding cluster centroid was then computed and converted into cosine distance as $1 - \cos ( x _ { i } , c _ { k } )$ $x _ { i }$ represents the �th sample, and $c _ { k }$ represents the �th prototype centroid. Samples with distances larger than the cluster-specific threshold, defined as the mean distance plus 1.5 times the standard deviation, were regarded as outliers and removed. Clusters containing fewer than two samples were excluded from this filtering procedure. After outlier removal, hierarchical clustering was recomputed on the filtered embeddings using average linkage and cosine distance. The silhouette score was then evaluated for each candidate cluster number, and the clustering configuration with the highest score was selected. The corresponding filtered embeddings, cluster labels, additive names, and linkage matrix were retained for subsequent prototype construction and prototype-

guided representation learning.

M10. Loss functions. In this section, we introduce two loss functions used in the pretrained molecular representation model and prototype semi-supervised contrastive learning model, respectively.

For the pre-trained molecular representation model, an InfoNCE-based contrastive loss was adopted to learn discriminative molecular representations. Given two augmented views of each molecular graph, their embeddings were first L2- normalized, and cosine similarity was used to measure pairwise similarity. For each sample, the corresponding positive pair was encouraged to have higher similarity than all other samples in the batch, which served as negatives. The loss is defined as:

$$
\mathcal { L } _ { I n f o N C E } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } l o g \frac { \exp { \left( s i m { \left( z _ { i } ^ { \left( 1 \right) } , z _ { i } ^ { \left( 2 \right) } \right) } / \tau \right) } } { \sum _ { j = 1 } ^ { N } \exp { \left( s i m { \left( z _ { i } ^ { \left( 1 \right) } , z _ { j } ^ { \left( 2 \right) } \right) } / \tau \right) } }\tag{25}
$$

where � is a temperature parameter controlling the concentration level of the distribution, and � represents the embedding of each sample. All other samples in the batch are treated as negatives. The similarity function sim(⋅,⋅) is defined as the cosine similarity between two embedding vectors:

$$
s i m \Big ( z _ { i } ^ { ( 1 ) } , z _ { i } ^ { ( 2 ) } \Big ) = \frac { { z _ { i } ^ { ( 1 ) } } ^ { \top } z _ { i } ^ { ( 2 ) } } { \left\| z _ { i } ^ { ( 1 ) } \right\| \left\| z _ { i } ^ { ( 2 ) } \right\| } = { z _ { i } ^ { ( 1 ) } } ^ { \top } z _ { i } ^ { ( 2 ) }\tag{26}
$$

For the prototype semi-supervised contrastive learning model, the similarity between a selected molecule and its corresponding prototype was treated as the positive logit, while similarities to all other prototypes were treated as competing logits:

$$
\mathcal { L } _ { p r o t o } = - \frac { 1 } { M } \sum _ { m = 1 } ^ { M } \frac { 1 } { | Z _ { m } | } \sum _ { z \in \mathcal { Z } _ { m } } l o g \frac { \exp ( s i m ( z , p _ { m } ) / \tau ) } { \sum _ { r = 1 } ^ { M } \exp ( s i m ( z , p _ { r } ) / \tau ) }\tag{27}
$$

where � is the number of prototypes, and $\tau = 0 . 1$ is the temperature parameter, and, ${ \mathcal { Z } } _ { m }$ was constructed by top-k assignment within each batch, such that molecules with the highest similarity to $p _ { m }$ were treated as positive samples for that prototype.

$$
\mathcal { L } _ { d e c o r } = \frac { 1 } { M ^ { 2 } } { \sum _ { i = 1 } ^ { M } } { \sum _ { j = 1 } ^ { M } } \big ( s i m ( p _ { i } , p _ { j } ) - \delta _ { i j } \big ) ^ { 2 } , \delta _ { i j } = 1 i f i = j , e l s e 0\tag{28}
$$

where $p _ { i } \in \mathbb { R } ^ { d }$ represents the embedding vector of the �-th prototype centroid in the latent space; ${ p _ { i } } ^ { \top } { p _ { j } }$ denotes the inner product (cosine similarity after normalization) between prototype � and prototype �; and $\delta _ { i j }$ is the Kronecker delta, which equals 1 when $i = j$ and 0 otherwise. This encourages prototypes to capture diverse and nonoverlapping molecular patterns. The total loss was defined as the sum of the prototype contrastive loss and the prototype decorrelation loss:

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { p r o t o } + \mathcal { L } _ { d e c o r }\tag{29}
$$

During evaluation, prototype-relevant core samples were selected from the heldout positive and unlabeled test molecules using the same top-k criterion, and the silhouette score with cosine distance was calculated on these selected embeddings. M11. Electrochemical measurements. Electrochemical measurements were carried out using CR2025-type $\mathsf { L i F e P O 4 } |$ |graphite coin cells. The baseline electrolyte consisted of $1 \mathsf { M L i P F } _ { 6 }$ in EC:DMC (3:7 by volume), with a solvent water content below 50 ppm, and the additive-containing electrolyte was prepared by adding 1 wt% additive to the baseline electrolyte. The cells were assembled in an Ar-filled glovebox using $\mathsf { L i F e P O 4 }$ cathodes, graphite anodes, Celgard 2025 separators and $6 0 ~ \mu \ L$ electrolyte, followed by resting for 12 h before testing. The cells were cycled between 2.5 and 3.65 V at $5 5 ~ ^ { \circ } \mathsf { C }$ . Formation was conducted for three cycles at 0.33C charge and 0.33C discharge, followed by long-term cycling at 0.33C charge and 1C discharge. Capacity retention was calculated by normalizing the discharge capacity of each cycle to the second discharge capacity.

M12. Post-mortem sample preparation. The coin cells were disassembled in an Arfilled glovebox to avoid exposure to air and moisture. The graphite electrodes were carefully collected and rinsed with anhydrous DMC three times to remove residual electrolyte and loosely attached salts. After the residual DMC evaporated inside the glovebox, the electrodes were mounted on the corresponding sample holders and transferred to the characterization chambers using an air-free vacuum transfer stage.

M13. X-ray photoelectron spectroscopy. XPS was performed using a PHI VersaProbe 4 system (ULVAC-PHI). Survey spectra and high-resolution spectra were collected to analyze the surface chemical composition and bonding environments of the SEI. The C 1s, O 1s, F 1s, P 2p, B 1s, and Fe 2p regions were analyzed when applicable. The binding energies were calibrated using the C-C component in the C 1s spectrum at 284.8 eV. Peak deconvolution was performed after background subtraction to determine the relative contents of different SEI components.

M14. Time-of-flight secondary ion mass spectrometry. ToF-SIMS measurements were carried out using an IONTOF M6 instrument. Negative-ion mode was mainly used to analyze SEI-related fragments, including $\mathsf { P O } _ { 2 } \bar { \mathbf { \Lambda } }$ ${ \mathsf { C H } } _ { 3 } { \mathsf { C O } } ^ { - }$ ${ \dot { \mathsf { L } } } { \dot { \mathsf { F } } } _ { 2 } { \mathsf { ^ { - } } }$ $\mathsf { C O } _ { 3 } ^ { - }$ ${ \mathsf { C H } } _ { 3 } { \mathsf { C O O } } ^ { - }$ , and $\mathsf { F e O ^ { - } }$ . Three-dimensional chemical renderings and sputter-depth profiles were obtained by sequential ion analysis and sputtering. The depth profiles were used to compare the relative distribution of solvent-derived organic fragments, carbonate-containing residues, LiF-related species, phosphate/fluorophosphaterelated fragments, and Fe-containing deposits within the SEI.

M15. Scanning electron microscopy. SEM was conducted using an SU3900 microscope. SEM images were collected to examine the morphology of the cycled graphite electrodes and the surface evolution after high-temperature cycling.

M16. Energy-dispersive X-ray spectroscopy mapping. EDS mapping was performed on the cycled graphite electrodes to analyze the spatial distribution of Fe. Elemental mapping was conducted on representative regions of the graphite electrode surface to evaluate transition-metal deposition after formation and long-term cycling. The Fe mapping results were used to compare the extent of Fe accumulation on graphite electrodes cycled in the baseline and additive-containing electrolytes.

## M17. Operando IR spectroscopy collection.

Operando infrared measurements were conducted using a Bruker Invenio R spectrometer equipped with a customized electrochemical cell containing an embedded IR fiber. The manually polished fiber was inserted through symmetric through-holes and sealed with epoxy resin for 12 h. The cell was assembled by sequentially stacking a graphite electrode, the IR fiber, a Whatman GF/D separator wetted with 120 μL electrolyte, and an LFP electrode with a diameter of 12 mm within a PTFE stabilizing ring. After a 1 h rest, transmitted IR signals were collected by a liquid-nitrogen-cooled MCT detector over $5 0 0 0 { - } 6 0 0 { \mathsf { c m } } ^ { - 1 }$ . Spectra were acquired every 1 min during cycling at a resolution of 4 cm<sup>-1</sup> with 32 scans, while maintaining the signal intensity at approximately 10,000 to ensure a high signal-to-noise ratio.

## Acknowledgments

The authors are grateful for the National Natural Science Foundation of China (No. 92372109), the National Key R&D Program of China (No. 2023YFB2503600), and Guangdong S&T Program (No. 2025A0505000017). The authors thank the Materials, Design and Manufacturing Facility (MDMF), Materials Characterization and Preparation Facility (MCPF), and Wilson Tang Brilliant Energy Science and Technology Lab (BEST Lab) of The Hong Kong University of Science and Technology (Guangzhou) for providing experimental support. This work is partially supported by GDIC.

## Author contributions

W.H. conceived the project, designed the methodology, collected the data, developed the model, performed the data analysis, prepared the figures, and wrote the paper, with participation in electrochemical testing and materials characterization. H.D. performed all electrochemical measurements, materials characterization experiments, and wrote the experimental part of the paper. J.T. contributed to the conceptualization of the project. R.T. provided guidance on model development and algorithm design. J.L. supervised the computational work and contributed to paper revision. Y.Q. supervised the paper preparation and provided theoretical guidance on boron chemistry. J.H. conceived and supervised the overall project and contributed to refinement of the paper and figures. All authors discussed the results and contributed to the final paper.

## Competing interests

A Chinese patent application related to this work has been filed and is currently pending.

## Data and code availability

The source code used in this study is publicly available at https://github.com/HongWxd/ProtoMI. The datasets generated and analyzed during the current study are available from the corresponding author upon reasonable request.

## References

1 Armand, M. & Tarascon, J.-M. Building better batteries. Nature 451, 652-657 (2008).

2 Yang, X., Zhang, H., Liu, Q. & Jiang, G. The Li-ion battery industry and its challenges. Nature Reviews Chemistry 9, 497-498 (2025).

3 Fan, X. & Wang, C. High-voltage liquid electrolytes for Li batteries: progress and perspectives. Chemical Society Reviews 50, 10486-10566 (2021).

4 Wan, H., Xu, J. & Wang, C. Designing electrolytes and interphases for high-energy lithium batteries. Nature Reviews Chemistry 8, 30-44 (2024).

5 Palacín, M. R. & de Guibert, A. Why do batteries fail? Science 351, 1253292 (2016).

6 Zhang, S. S. A review on electrolyte additives for lithium-ion batteries. Journal of Power Sources 162, 1379-1394 (2006).

7 Pandey, M. et al. The transformational role of GPU computing and deep learning in drug discovery. Nature Machine Intelligence 4, 211-221 (2022).

8 Jumper, J. et al. Highly accurate protein structure prediction with AlphaFold. Nature 596, 583- 589 (2021).

9 Gao, Y.-C. et al. Accelerating battery innovation: AI-powered molecular discovery. Chemical Society Reviews 54, 9630-9684 (2025).

10 Gao, Y.-C. et al. Data-Driven Insight into the Reductive Stability of Ion–Solvent Complexes in Lithium Battery Electrolytes. Journal of the American Chemical Society 145, 23764-23770 (2023).

11 Gao, X. et al. Generative Artificial Intelligence Navigated Development of Solvents for Next Generation High-Performance Magnesium Batteries. Advanced Materials 38, e10083 (2026).

12 Li, H., Hao, J. & Qiao, S.-Z. AI-Driven Electrolyte Additive Selection to Boost Aqueous Zn-Ion Batteries Stability. Advanced Materials 36, 2411991 (2024).

13 Lao, Z. et al. Data-driven exploration of weak coordination microenvironment in solid-state electrolyte for safe and energy-dense batteries. Nature Communications 16, 1075 (2025).

14 Xie, T. et al. Accelerating amorphous polymer electrolyte screening by learning to reduce errors in molecular dynamics simulated properties. Nature Communications 13 (2022).

15 Li, K., Wang, J., Song, Y. & Wang, Y. Machine learning-guided discovery of ionic polymer electrolytes for lithium metal batteries. Nature Communications 14 (2023).

16 Wang, F., Tang, Y.-H., Ma, Z.-B., Jin, Y.-C. & Cheng, J. Domain oriented universal machine learning potential enables fast exploration of chemical space of battery electrolytes. Nature Communications 17 (2025).

17 Gao, Y. C. et al. A Knowledge–Data Dual‐Driven Framework for Predicting the Molecular Properties of Rechargeable Battery Electrolytes. Angewandte Chemie International Edition 64 (2024).

18 Zou, B.-B. et al. Decoding Lithium Metal Battery Degradation with Symmetric-Cell Artificial Intelligence Diagnostics (SAID). Advanced Materials 38, e12041 (2026).

20 Hong, X. et al. Deep active learning and knowledge transfer for rapid discovery of lithium metal battery electrolytes. Nature Communications 17, 5146 (2026).

21 Zhu, S. et al. Differentiable modeling and optimization of non-aqueous Li-based battery electrolyte solutions using geometric deep learning. Nature Communications 15 (2024).

22 Li, R. et al. Data-driven design of advanced magnesium-battery electrolyte via dynamic solvation models. Energy & Environmental Science 18, 6790-6798 (2025).

23 Ma, L. et al. Designing Bilayer Heterostructure Functional Polymer Electrolytes with Interfacial Engineering Strategy for High‐Performance Lithium Metal Batteries. Advanced Functional Materials 35, 2414816 (2025).

24 Park, N. R. et al. Understanding Boron Chemistry as the Surface Modification and Electrolyte Additive for Co-Free Lithium-Rich Layered Oxide. Advanced Energy Materials 14, 2401968 (2024).

25 Yu, Y. et al. Enable reversible conversion reaction of copper fluoride batteries by hydroxyl solution and anion acceptor. Energy Storage Materials 64, 103073 (2024).

26 Deng, T. et al. Designing In-Situ-Formed Interphases Enables Highly Reversible Cobalt-Free LiNiO2 Cathode for Li-ion and Li-metal Batteries. Joule 3, 2550-2564 (2019).

27 Cherti, M. et al. in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2818-2829 (2023).

28 Jin, J., Wang, S., Dong, Z., Liu, X. & Zhu, E. in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 11600-11609 (2023).

29 Rao, H. & Miao, C. in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 22118-22128 (2023).

30 Yang, X., Song, Z., King, I. & Xu, Z. A Survey on Deep Semi-Supervised Learning. IEEE Transactions on Knowledge and Data Engineering 35, 8934-8954 (2023).

31 Kim, S. et al. PubChem 2025 update. Nucleic acids research 53, D1516-D1525 (2025).

32 Azam, S. et al. Improving and Understanding Lifetime of LFP/Graphite Pouch Cells with Higher Concentrations of Vinylene Carbonate in the Electrolyte. Journal of The Electrochemical Society 172, 070523 (2025).

33 Black, W., Azam, S., MacLennan, H., Metzger, M. & Dahn, J. R. Understanding Capacity Loss in LFP/Graphite Pouch Cells at High Temperatures through Modelling. Journal of the Electrochemical Society 172, 090503 (2025).

34 McInnes, L., Healy, J. & Melville, J. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426 (2018).

35 Rogers, D. & Hahn, M. Extended-connectivity fingerprints. Journal of Chemical Information and Modeling 50, 742-754 (2010).

36 Murtagh, F. & Contreras, P. Algorithms for hierarchical clustering: an overview. WIREs Data Mining and Knowledge Discovery 2, 86-97 (2011).

37 Ertl, P. & Schuffenhauer, A. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of Cheminformatics 1 (2009).

38 Hu, H., Wang, X., Zhang, Y., Chen, Q. & Guan, Q. A comprehensive survey on contrastive learning. Neurocomputing 610 (2024).

39 Pinheiro, G. A., Da Silva, J. L. F. & Quiles, M. G. SMICLR: Contrastive Learning on Multiple Molecular Representations for Semisupervised and Unsupervised Representation Learning. Journal of Chemical Information and Modeling 62, 3948-3960 (2022).

40 Wang, Y. et al. in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 207-217 (2020).

41 Page, L., Brin, S., Motwani, R. & Winograd, T. in The Web Conference.

42 Zhu, Y. et al. in Proceedings of the Web Conference 2021 2069-2080 (2021).

43 You, Y. et al. Graph contrastive learning with augmentations. Advances in neural information processing systems 33, 5812-5823 (2020).

44 Cereto-Massagué, A. et al. Molecular fingerprint similarity search in virtual screening. Methods 71, 58-63 (2015).

45 Weininger, D. SMILES, a chemical language and information system. 1. Introduction to methodology and encoding rules. Journal of chemical information and computer sciences 28, 31-36 (1988).

46 Rousseeuw, P. Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. J. Comput. Appl. Math. 20, 53–65 (1987).

47 Zhang, H., Lu, G., Zhan, M. & Zhang, B. Semi-Supervised Classification of Graph Convolutional Networks with Laplacian Rank Constraints. Neural Process. Lett. 54, 2645–2656 (2022).

48 Lin, L. & Wang, H. in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining 1819–1827 (Association for Computing Machinery, Virtual Event, CA, USA, 2020).

49 You, Y. et al. in Proceedings of the 34th International Conference on Neural Information Processing Systems Article 488 (Curran Associates Inc., Vancouver, BC, Canada, 2020).

50 Likas, A., Vlassis, N. & Verbeek, J. J. The global k-means clustering algorithm. Pattern recognition 36, 451-461 (2003).

51 Xue, Z.-M., Zhao, J.-F., Ding, J. & Chen, C.-H. LBDOB, a new lithium salt with benzenediolato and oxalato complexes of boron for lithium battery electrolytes. Journal of Power Sources 195, 853-856 (2010).

52 Nie, M., Xia, J. & Dahn, J. Development of pyridine-boron trifluoride electrolyte additives for lithium-ion batteries. Journal of The Electrochemical Society 162, A1186-A1195 (2015).

53 Ma, Y. et al. A new anion receptor for improving the interface between lithium-and manganeserich layered oxide cathode and the electrolyte. Chemistry of Materials 29, 2141-2149 (2017).

54 Xi, K. et al. A novel strategy to improve the electrochemical properties of in-situ polymerized 1, 3-dioxolane electrolyte in lithium metal batteries. Journal of Colloid and Interface Science 679, 1277-1287 (2025).

55 Morales-Brotons, D., Vogels, T. & Hendrikx, H. Exponential Moving Average of Weights in Deep Learning: Dynamics and Benefits. Trans. Mach. Learn. Res. 2024 (2024).

56 Zhang, J. et al. Worse Interference of Fe3+ than Fe2+ on Degrading the Interphase and Performance of LiFePO4||Graphite Battery. Advanced Materials 37, e13736 (2025).

57 Dedryvère, R. et al. Characterization of lithium alkyl carbonates by X-ray photoelectron spectroscopy: experimental and theoretical study. The Journal of Physical Chemistry B 109, 15868-15875 (2005).

58 Sim, R., Su, L., Dolocan, A. & Manthiram, A. Delineating the Impact of Transition‐Metal Crossover on Solid‐Electrolyte Interphase Formation with Ion Mass Spectrometry. Advanced Materials 36, 2311573 (2024).

59 Ota, H., Akai, T., Namita, H., Yamaguchi, S. & Nomura, M. XAFS and TOF–SIMS analysis of SEI layers on electrodes. Journal of Power Sources 119-121, 567-571 (2003).

60 Wang, A., Kadam, S., Li, H., Shi, S. & Qi, Y. Review on modeling of the anode solid electrolyte interphase (SEI) for lithium-ion batteries. npj Computational Materials 4, 15 (2018).

61 Xu, H. et al. Impacts of dissolved Ni2+ on the solid electrolyte interphase on a graphite anode. Angewandte Chemie 134, e202202894 (2022).

62 Guo, Z., Cui, Z. & Manthiram, A. Crossover Effects of Transition‐Metal Ions on Lithium‐Metal Anode in Localized High Concentration Electrolytes. Advanced Functional Materials 35, 2501743 (2025).

63 Wang, D. et al. Revealing the lithium solid electrolyte interphase in liquid electrolytes via in situ Fourier transform infrared spectroscopy. Cell Press Blue (2026).

64 Wang, Y. et al. Formation dynamics of an ethylene carbonate-derived solid–electrolyteinterphase in commercial Li-ion batteries. Energy & Environmental Science (2026).

65 Zhang, C. et al. Small Molecule Accurate Recognition Technology (SMART) to Enhance Natural Products Research. Scientific Reports 7 (2017).

# Supplementary information

# Prototype-guided transfer of sparse literature knowledge for electrolyte additive discovery

Weixiang HONG<sup>1,5</sup>, Hongting DU<sup>1,5</sup>, Jiayue TANG<sup>1</sup>, Ruifeng TAN<sup>1</sup>, Yangjian QUAN<sup>2</sup>, Jia LI<sup>3,\*</sup>, Jiaqiang HUANG<sup>1,4,\*</sup>

<sup>1</sup> Sustainable Energy and Environment Thrust and Guangzhou Municipal Key Laboratory of Materials Informatics, The Hong Kong University of Science and Technology (Guangzhou), Nansha, Guangzhou, 511400, Guangdong, P.R. China

<sup>2</sup> Department of Chemistry, The Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong SAR, P.R. China

<sup>3</sup> Data Science and Analytics Thrust and Guangzhou Municipal Key Laboratory of Materials Informatics, The Hong Kong University of Science and Technology (Guangzhou), Nansha, Guangzhou, 511400, Guangdong, P.R. China

<sup>4</sup> Academy of Interdisciplinary Studies, The Hong Kong University of Science and Technology, Clear Water Bay, Kowloon, Hong Kong SAR, P.R. China

<sup>5</sup> These authors contributed equally: Weixiang HONG, Hongting DU

\* Corresponding authors:

J. Huang: seejhuang@hkust-gz.edu.cn

J. Li: jialee@hkust-gz.edu.cn

![](images/96fdbfebbc59f653cf6bb4950efdf7b603df12abd7e10dcaae20c70a8678e1fc.jpg)  
Figure S1 | Hierarchical clustering<sup>1</sup> dendrogram of reported boron-containing electrolyte additives. Hierarchical clustering dendrogram of 126 reported boron-containing electrolyte additives constructed using Morgan fingerprints and pairwise Tanimoto distances<sup>2</sup>. Each terminal leaf represents one reported additive, and the branch height reflects molecular dissimilarity. Even with this simple fingerprint-based representation, additives with highly similar names and closely related chemical structures are grouped into the same local branches, such as families of lithium borates, fluorinated borates and aryl borate derivatives. This branch-level organization suggests that reported boron-containing additives are not randomly scattered as isolated successful examples, but are enriched in recurrent structural families. These repeated local branches therefore support the presence of recurring molecular patterns or scaffold-level motifs in the reported additive chemical space. Branch colors are used to visually distinguish major dendrogram regions generated by hierarchical clustering.

Table S1 | Dataset statistics
<table><tr><td>Category</td><td>Value</td></tr><tr><td>Atoms</td><td> $4 1 . 1 9 1 { \scriptstyle \pm 4 7 . 2 8 1 }$ </td></tr><tr><td>Bonds</td><td> $8 7 . 2 5 9 { \pm } 1 0 2 . 3 8 6$ </td></tr><tr><td>Average degree</td><td> $4 . 1 3 0 { \pm } 0 . 4 3 2$ </td></tr><tr><td>Graph density</td><td> $0 . 1 8 9 { \pm } 0 . 1 1 7$ </td></tr><tr><td>Degree entropy</td><td>3.265±0.802</td></tr></table>

These statistics were calculated from both reported molecules and the searching space molecules.

Table S2 | Node features and corresponding physical meanings
<table><tr><td>Feature No.</td><td>Physical Meaning</td><td>Feature type</td><td>Encoding type</td><td>Data type</td></tr><tr><td>NO</td><td>C</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N1</td><td>N</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N2</td><td>0</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N3</td><td>S</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N4</td><td>F</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N5</td><td>Si</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N6</td><td>P</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N7</td><td>Cl</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N8</td><td>Br</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N9</td><td>Mg</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N10</td><td>Na</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N11</td><td>Ca</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N12</td><td>Fe</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N13</td><td>As</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N14</td><td>AI</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N15</td><td>1</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N16</td><td>B</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N17</td><td>V</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N18</td><td>K</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N19</td><td>TI</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N20</td><td>Yb</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N21</td><td>Sb</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N22</td><td>Sn</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N23</td><td>Ag</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N24</td><td>Pd</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N25</td><td>Co</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N26</td><td>Se</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N27</td><td>Ti</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N28</td><td>Zn</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N29</td><td>Li</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N30</td><td>Ge</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N31</td><td>Cu</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N32</td><td>Au</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N33</td><td>Ni</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N34</td><td>Cd</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N35</td><td>In</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N36</td><td>Mn</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N37</td><td>Zr</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N38</td><td>Cr</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N39</td><td>Pt</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N40</td><td>Hg</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N4</td><td>Pb</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N42</td><td>Unknown element</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N43</td><td>n_heavy_neighbors_0</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N44</td><td>n_heavy_neighbors_1</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N45</td><td>n_heavy_neighbors_2</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N46</td><td>n_heavy_neighbors_3</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N47</td><td>n_heavy_neighbors_4</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N48</td><td>n_heavy_neighbors_More ThanFour</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N49</td><td>formal_charge_-3</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N50</td><td>formal_charge_-2</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N51</td><td>formal_charge_-1</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N52</td><td>formal_charge_0</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N53</td><td>formal_charge_1</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N54</td><td>formal_charge_2</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N55</td><td>formal_charge_3</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N56</td><td>formal_charge_Extreme</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N57</td><td>S</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N58</td><td>SP</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N59</td><td>SP2</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N60</td><td>SP3</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N61</td><td>SP3D</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N62</td><td>SP3D2</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N63</td><td>Other hybridization</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N64</td><td>is_in_a_ring</td><td>Node feature</td><td>Numerical</td><td>Int</td></tr><tr><td>N65</td><td>is_aromatic</td><td>Node feature</td><td>Numerical</td><td>Int</td></tr><tr><td>N66</td><td>atomic_mass_scaled</td><td>Node feature</td><td>Numerical</td><td>Float</td></tr><tr><td>N67</td><td>vdw_radius_scaled</td><td>Node feature</td><td>Numerical</td><td>Float</td></tr><tr><td>N68</td><td>covalent_radius_scaled</td><td>Node feature</td><td>Numerical</td><td>Float</td></tr><tr><td>N69</td><td>CHI_UNSPECIFIED</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N70</td><td>CHI_TETRAHEDRAL_CW</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N71</td><td>CHI_TETRAHEDRAL_CC</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td></td><td>W</td><td></td><td></td><td></td></tr><tr><td>N72 N73</td><td>CHI_OTHER</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td></td><td>n_hydrogens_0</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N74</td><td>n_hydrogens_1</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N75</td><td>n_hydrogens_2</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>N76</td><td>n_hydrogens_3</td><td>Node feature</td><td>One hot encoding</td><td>String</td></tr></table>

![](images/ce8a668fc35d5353eb8408937df37dcf0c16bead2d5aac05ffdd5368346e0dd0.jpg)  
Figure S2 | Selection of the optimal number of molecular prototype. Average silhouette scores of hierarchica clustering and K-means<sup>3</sup> clustering evaluated for cluster numbers ranging from 3 to 15. The optimal number of molecular prototypes was determined using the silhouette score, which measures the compactness and separation of molecular clusters. Hierarchical clustering achieved the highest silhouette score at k = 7, while Kmeans clustering showed a consistent optimum in the same region. Therefore, seven molecular prototypes were selected in a data-driven manner and used throughout this study for unsupervised prototype discovery and subsequent prototype-guided learning.

![](images/29a9a4cc5e3731cd95fbbd4e3ae8c6978ff7e87732e6727b5d46bbb3d9a46b64.jpg)  
Figure S3 | UMAP<sup>4</sup> visualization of the reported boron electrolyte additive molecules learned by the UCL model. UMAP projection of the learned graph-level embeddings of reported boron-containing electrolyte additives. Each point represents one reported additive molecule, and colors denote the prototype assignments obtained by

hierarchical clustering in the learned representation space. The seven well-separated groups indicate that the UCL model organizes reported additives into distinct molecular prototype regions, consistent with the optimal cluster number determined by silhouette-score analysis in Supplementary Figure S2. The spatial separation of the clusters supports the presence of structurally and functionally distinguishable additive prototypes encoded in the learned embedding space.

Table S3 | Reported additive molecules in each prototypes.
<table><tr><td>Molecular ID</td><td>Molecular name</td><td>Molecules</td><td>Molecular weight</td><td>Synthetic accessibility Score5</td><td>Prototype label</td><td>References</td></tr><tr><td>6</td><td>TBDB</td><td><img src="images/af439e6fc6c64b723711ad802ed1f723cbd7a2de76f6c90e9c82529f791a8533.jpg"/></td><td>225.890</td><td>4.273</td><td>1</td><td>6</td></tr><tr><td>10</td><td>ITD</td><td><img src="images/ee78e79216eadd9dda53464804c80b09bb9942d1e59a805ed0db8cc0ebbcdba0.jpg"/></td><td>186.060</td><td>3.616</td><td>1</td><td>7</td></tr><tr><td>12</td><td>SOFPB</td><td><img src="images/fa194764ef5f9a1c53a160fbcb71e5773d1ce908cb03a6e72d35114a70e583d6.jpg"/></td><td>352.331</td><td>2.987</td><td>1</td><td>8</td></tr><tr><td>14</td><td>DPD-CN-OCH3</td><td><img src="images/144bee71e1bd50e07f3a06896d0b71441cab53adb00b1a66812c15dc1f5e0dc0.jpg"/></td><td>245.087</td><td>3.159</td><td>1</td><td>9</td></tr><tr><td>19</td><td>C-LiMCFB</td><td><img src="images/d4a4be3752ce3d5ba23adf32ce47ee919695bd7c042750283280d786382e0a2a.jpg"/></td><td>317.089</td><td>4.211</td><td>1</td><td>10</td></tr><tr><td>24</td><td>NHCABT</td><td><img src="images/00033027ee2ea49bf76feaded5fd02e2bfb1e35d6389e0e618131a60afe166fe.jpg"/></td><td>167.971</td><td>3.293</td><td>1</td><td>11</td></tr><tr><td>32</td><td>DPD</td><td><img src="images/645fda8d5a7be82c194930d1ab8f22236db29481eb5ab5b7cf6202b03df47530.jpg"/></td><td>190.051</td><td>2.893</td><td>1</td><td>9</td></tr><tr><td>43</td><td>BF3-THF</td><td><img src="images/fc6e8cbcd2b3799eb80b9e51e8b04c64ab60b4a5c161356592c31abf97ddbdba.jpg"/></td><td>139.913</td><td>5.532</td><td>1</td><td>12</td></tr><tr><td>59</td><td>DiOB-Py</td><td><img src="images/3c5157a36afca32fbd2662b5eebfcb5f86d230e1a99436ef6cedc1c35b27a8fc.jpg"/></td><td>205.066</td><td>2.853</td><td>1</td><td>13</td></tr><tr><td>66</td><td>BEG-1</td><td><img src="images/6ad00d855ab69082d2f77ef7735dcd58d176175cb68f986887f3989adc35604d.jpg"/></td><td>412.185</td><td>4.069</td><td>1</td><td>14</td></tr><tr><td>83</td><td>DPD-CN</td><td><img src="images/af3ff973aef191617495c178424d11fda06d148f5c26562eb23a72102ee192c1.jpg"/></td><td>215.061</td><td>3.159</td><td>1</td><td>9</td></tr><tr><td>92</td><td>DPD-F</td><td><img src="images/a4771c38badc6b2b003772b055364bffdd70f83cf908b5e9554a13f2f3e637f6.jpg"/></td><td>208.041</td><td>2.98</td><td>1</td><td>9</td></tr><tr><td>95</td><td>BDB</td><td><img src="images/accfe186dda6513fbffa9733067df7a69b96b54a6c98a30e510af54c64ad094e.jpg"/></td><td>253.944</td><td>3.428</td><td>1</td><td>15</td></tr><tr><td>98</td><td>DPD-CHO</td><td><img src="images/2b62e6234318e7ea1eee2e0e3726249ca96be3c53a206dc852f68a4f4e1ac3f1.jpg"/></td><td>218.061</td><td>3.128</td><td>1</td><td>9</td></tr><tr><td>104</td><td>DPD-COOCH2CH3</td><td><img src="images/e9b683caa4b1689beb4181535d1c1dd9d5ee1d2267bf303c121a7de369abc5c2.jpg"/></td><td>262.114</td><td>2.852</td><td>1</td><td>9</td></tr><tr><td>116</td><td>DiOB-An</td><td><img src="images/6ff5a24d82c8fe97f94eefed1777543c366eb3e88f59532f35766d24e0ccb468.jpg"/></td><td>275.201</td><td>2.699</td><td>1</td><td>13</td></tr><tr><td>3</td><td>PFPBO</td><td><img src="images/9029ddd53499fa7a2db167f865a9bbd2f2086fef4fbab66ff7c6cbdd4af5200d.jpg"/></td><td>265.886</td><td>3.559</td><td>2</td><td>16</td></tr><tr><td>34</td><td>DCLBSB</td><td><img src="images/f8407fb831db43d18e486dfc6d789f53f59be0c89c1ff234f4682e0c167c2686.jpg"/></td><td>427.745</td><td>4.164</td><td>2</td><td>17</td></tr><tr><td>45</td><td>TFPBO</td><td><img src="images/3773daf5820b12ee03f9613b8d87a10750575365c699d8d1b8784aa013afe9ea.jpg"/></td><td>559.977</td><td>2.965</td><td>2</td><td>18</td></tr></table>

<table><tr><td>51</td><td>LiBBrSB</td><td><img src="images/c3125744385a6c9d02564ee144f6f48f0f79ae855ed9128028f5f5038637d611.jpg"/></td><td></td><td>462.792</td><td>4.368</td><td>2</td><td>19</td></tr><tr><td>54</td><td>PFPTFBB</td><td><img src="images/656f89992160f115cd0b10eed7e3a4c35b75c53f1d09eb4b06d34135cac333fa.jpg"/></td><td></td><td>357.924</td><td>3.393</td><td>2</td><td>20</td></tr><tr><td>56</td><td>TPFPB</td><td></td><td><img src="images/38f1b42a971a1f0f2e8f9f7bea165e21fe334e5c31a2b5db8830eda721a0255c.jpg"/></td><td>511.980</td><td>2.800</td><td>2</td><td>21</td></tr><tr><td>75</td><td>TPF</td><td></td><td><img src="images/c2a01c61288cc401a7b0624f911fa231aa93a28024a0c3b2d7d0a8cb92844bd6.jpg"/></td><td>511.980</td><td>2.800</td><td>2</td><td>22</td></tr><tr><td>77</td><td>TFPB</td><td><img src="images/8f6889971c2c266a120de7af13a4364d31fe6af31abc909dcccc3c4208c00724.jpg"/></td><td></td><td>511.980</td><td>2.800</td><td>2</td><td>18</td></tr><tr><td>94</td><td>TCLBSB</td><td></td><td><img src="images/18aa10427a043d0a644ced5637ced9520aad326bdb47bd0da6b6631803580a01.jpg"/></td><td>496.635</td><td>4.388</td><td>2</td><td>17</td></tr><tr><td>109</td><td>4FLBDOB</td><td></td><td><img src="images/97d43ac2eb9352fe7cdaabf41efac8ca79cc0181d747dea9750456ed8247b4d8.jpg"/></td><td>285.827</td><td>4.647</td><td>2</td><td>23</td></tr><tr><td>115</td><td>LiBCISB</td><td></td><td><img src="images/842094390bdbdd6ff4910f63164cf92355ef5e19038975e657731a4e11b39463.jpg"/></td><td>373.89</td><td>4.260</td><td>2</td><td>19</td></tr><tr><td>122</td><td>LiBCI2SB</td><td><img src="images/e8025381d67baefca6770a424b20184146afb1e56114d06bc2d3626d8b1e7a46.jpg"/></td><td></td><td>442.78</td><td>4.355</td><td>2</td><td>19</td></tr><tr><td>4</td><td>LiPVAOB</td><td><img src="images/1963097c71edb1f262ab81a7a5c239f21aa6e98cc414b9496fa3ec2873eb0d98.jpg"/></td><td></td><td>221.931</td><td>5.608</td><td>3</td><td>24</td></tr><tr><td>30</td><td>ADM</td><td><img src="images/ce9533c26a6de1cdb720dedff7f88d9d1da3e3ddb77ce39d09cadb14d6720409.jpg"/></td><td></td><td>170.961</td><td>3.811</td><td>3</td><td>25</td></tr><tr><td>36</td><td>LiODFB</td><td>Li+ <img src="images/057b9ca88df128673ad08b88147e71e75daa97e2ef0fda23aaa018f9a96a73d6.jpg"/></td><td>143.767</td><td>4.662</td><td>3</td><td>26</td></tr><tr><td>37</td><td>LiPAAOB</td><td><img src="images/6d83279f7ce0b1d1134555ac89393e85f6442d0a486f7ff9189627e9f52354cc.jpg"/></td><td>278.959</td><td>4.432</td><td>3</td><td>27</td></tr><tr><td>52</td><td>LBDCB</td><td><img src="images/66fd3f9382fd63ef703687fed5c45dfa21a685b05be39cdfb89e22939a9f10a1.jpg"/></td><td>265.899</td><td>5.246</td><td>3</td><td>28</td></tr><tr><td>61</td><td>LDFOB</td><td><img src="images/a7e5b047f9368ea31f2940376d7ea6eab8da444425436dae534b1597d329d46f.jpg"/></td><td>143.767</td><td>4.662</td><td>3</td><td>29</td></tr><tr><td>67</td><td>LBCB</td><td><img src="images/4f896318fffe59d171da4fbf0b2b57cdad0b1b06c3be12102372bf4ca1f7f37f.jpg"/></td><td>297.853</td><td>5.653</td><td>3</td><td>28</td></tr><tr><td>69</td><td>NaDFOB</td><td><img src="images/abf552547a2aeb92dc50800d93063dba5b253b7aebc64a665b0e6c8a913934e8.jpg"/></td><td>159.816</td><td>4.729</td><td>3</td><td>30</td></tr><tr><td>73</td><td>LiBOB</td><td><img src="images/8a5cd276350eadcc8cf982f1de16ef582df438e75020ff1408de3ff4b6d7d637.jpg"/></td><td>193.789</td><td>4.913</td><td>3</td><td>19</td></tr><tr><td>101</td><td>LiMOB</td><td><img src="images/0e1dc8cacda60ea615e125d69bbd95d0e017ca2dbffff40880e27d56825b67b2.jpg"/></td><td>221.843</td><td>4.983</td><td>3</td><td>31</td></tr><tr><td>102</td><td>LiBMAB</td><td><img src="images/f92b03dda4a2c925a84357265cf5e81f07e7f446930b8d6c422aa84359ea77d2.jpg"/></td><td>301.973</td><td>4.961</td><td>3</td><td>32</td></tr><tr><td>105</td><td>LOCB</td><td><img src="images/5b76af9a39815172ec2c949054a8c3dd31e741299e6ccb9d18d99bddcc9dcb37.jpg"/></td><td>245.821</td><td>5.846</td><td>3</td><td>28</td></tr><tr><td>110</td><td>LiBF4</td><td>Li+ <img src="images/d1d83ea7727adee455fa7ecb62889b796443db0386a407c31859c09cbcc0007d.jpg"/></td><td>193.789</td><td>4.913</td><td>3</td><td>29</td></tr><tr><td>112</td><td>LiDFOB</td><td><img src="images/6cb6238a35382bf033b02600f98f55ff22a6e313179b5907be44d3fb6b28b6f6.jpg"/></td><td>143.767</td><td>4.662</td><td>3</td><td>33</td></tr><tr><td>5</td><td>(C6H3F)O2B(C8H3 F6)</td><td><img src="images/9f1653334e3453505dfe48f36967b62a6926845a234a1a3472278727cb57898e.jpg"/></td><td>349.998</td><td>3.089</td><td>4</td><td>34</td></tr><tr><td>8</td><td>LBBPB</td><td><img src="images/aa5897afd0ea80c86b74e85b5ee4dc3fce3e7d2f96f4fc91c66d044a26647d0d.jpg"/></td><td>386.141</td><td>3.111</td><td>4</td><td>35</td></tr><tr><td>9</td><td>TPhBX</td><td><img src="images/0f2796e07a0b9bb129811f380009703650adf1926024e2157c9dc16861368579.jpg"/></td><td>311.751</td><td>2.739</td><td>4</td><td>36</td></tr><tr><td>15</td><td>(C6H3F)O2B(C6H3 F2)</td><td><img src="images/635e22095f9c59aec53869ec8151f9ca42a642b9f139662e35b9593098cd12e9.jpg"/></td><td>249.984</td><td>2.911</td><td>4</td><td>34</td></tr><tr><td>18</td><td>LBSB</td><td><img src="images/f695563b92ab7f084e681fdb4cf3878f90d24dd05853894915383c12e8b9eef5.jpg"/></td><td>289.965</td><td>3.837</td><td>4</td><td>17</td></tr><tr><td>28</td><td>LBNB</td><td><img src="images/6b7aaff9c5036abd47132ad4bc467dd95cf2cc104d0a7cf665b8cf90cc416a87.jpg"/></td><td>334.065</td><td>3.487</td><td>4</td><td>35</td></tr><tr><td>50</td><td>(C6H3F)O2B(C7H4 F3)</td><td><img src="images/4bce4c9e8349825f7bee3e869162d65b2c64ceff15c1ac3004873e6f62d7e4cc.jpg"/></td><td>282.001</td><td>2.880</td><td>4</td><td>34</td></tr><tr><td>55</td><td>(C6F4)O2B(C6F5)</td><td><img src="images/20e1bb3861c0e515d4ca269aa6655a230e7791f13a41979701dac1ab9dd26b77.jpg"/></td><td>234.010</td><td>2.888</td><td>4</td><td>34</td></tr><tr><td>60</td><td>(C6F4)O2B(C8H3F 6)</td><td><img src="images/086fe49199ebe857fef5b1e2b32869894195ed24b4d4c9aef0acbdd04b79bbf7.jpg"/></td><td>351.006</td><td>3.059</td><td>4</td><td>34</td></tr><tr><td>70</td><td>LBDOB</td><td><img src="images/52bb27a4960088d70429d74a70200d3c00c64268b887727e0c1d478b81549e42.jpg"/></td><td>233.945</td><td>3.733</td><td>4</td><td>37</td></tr><tr><td>76</td><td>LBBB</td><td><img src="images/dbac92e9aad18f9bfce3f0d88c76b1355dcbe98922a7e1bd76ce3992a77bd84f.jpg"/></td><td>233.945</td><td>3.733</td><td>4</td><td>35</td></tr><tr><td>82</td><td>(C6F4)O2B(C7H3F 6)</td><td><img src="images/e7c919cafd40f5bb2bed5456d62153aefaa52abc6362e06a57efdc3f097f692e.jpg"/></td><td>283.009</td><td>2.906</td><td>4</td><td>34</td></tr><tr><td>84</td><td>LiBPh4</td><td><img src="images/3b1bb3c481cc9b802afb0c4d6ce51842dd749f83a1e7a2d70fa8b02c71516953.jpg"/></td><td>326.177</td><td>2.093</td><td>4</td><td>38</td></tr><tr><td>85</td><td>3-MLBSB</td><td><img src="images/df79523b31da4b87dc9a6e79b651809f4b1d886fcee398c88a3aaf7ff08f53a3.jpg"/></td><td>318.019</td><td>4.029</td><td>4</td><td>17</td></tr><tr><td>88</td><td>(C6F4)O2B(C6H4F)</td><td><img src="images/07214d8a866f7591b7f831ce2f4dfae3d14792bcd392fa45e63117c98adb906b.jpg"/></td><td>233.002</td><td>2.742</td><td>4</td><td>34</td></tr><tr><td>89</td><td>Li[BScB]</td><td><img src="images/ac411f632ba15718965c32898ff5aef0361b70e8f71f5f880348386a7c818d83.jpg"/></td><td>289.965</td><td>3.837</td><td>4</td><td>39</td></tr><tr><td>97</td><td>B(OPh)3</td><td><img src="images/ac615ad52596142caf302d51e6cecd600026e8b4d44ead95fae8659a64bff7d0.jpg"/></td><td>290.127</td><td>2.012</td><td>4</td><td>40</td></tr><tr><td>107</td><td>LiBSB</td><td><img src="images/1ace89c48e7374cfd3b8608f0683f039b3ab6bc5a9136a46cdc4ce5f14d1602d.jpg"/></td><td>289.965</td><td>3.837</td><td>4</td><td>19</td></tr><tr><td>113</td><td>(C6F4)O2B(C6H3F 2)</td><td><img src="images/eb5ba23f12c4672ae66c24caf0b097696d92c922ff5e68c8458473155fcf825e.jpg"/></td><td>250.992</td><td>2.873</td><td>4</td><td>34</td></tr><tr><td>7</td><td>TRIFM</td><td><img src="images/9a7ec7e34cc33a710cfc1e68897c1f6bf17bd29b827cbab0dda545c2e91d951b.jpg"/></td><td>214.905</td><td>2.601</td><td>5</td><td>41</td></tr><tr><td>11</td><td>LBO</td><td><img src="images/b14a7e55f8642e84f9b44cbd4736321fb5396ce1ed27119a03093ebea9bca703.jpg"/></td><td>49.751</td><td>5.870</td><td>5</td><td>42</td></tr><tr><td>21</td><td>Li2TB</td><td><img src="images/854639aa702497a3008ca5c6ebf106edf432cc8c043ab04f450e4d8c38688745.jpg"/></td><td>169.123</td><td>6.757</td><td>5</td><td>43</td></tr><tr><td>33</td><td>PRZ</td><td><img src="images/25ff25275ede16227d754ca1462a02d9d149dffb6fc094ee3c97c8388b11b4a4.jpg"/></td><td>215.702</td><td>4.900</td><td>5</td><td>44</td></tr><tr><td>35</td><td>H3BO3</td><td><img src="images/36f8eedcab6b855b98987fe546f1cf0531c88426a5bc9885828cec15fd3f5929.jpg"/></td><td>61.833</td><td>3.432</td><td>5</td><td>45</td></tr><tr><td>38</td><td>FBA</td><td><img src="images/05ee557cbddc964459ed2cd81ed70e95b28dd82294019f3ba8e25df29ebd58a2.jpg"/></td><td>189.929</td><td>2.415</td><td>5</td><td>46</td></tr><tr><td>41</td><td>Li3BO3</td><td><img src="images/e48b61d51d71984515699796564e00c44efa4a7575d85e7fb2e42ad26d555fad.jpg"/></td><td>65.750</td><td>6.320</td><td>5</td><td>47</td></tr><tr><td>44</td><td>B203</td><td><img src="images/e761d1319d90bed1e181dc4465009665782c43515e5a7a66671fef8953f4668b.jpg"/></td><td>263811</td><td>5.773</td><td>5</td><td>48</td></tr><tr><td>57</td><td>LiBO2</td><td><img src="images/319052a7118ecbb275bd0bc6c6ad2ae693b0cfb645e012f7501e75d3ad24dfdf.jpg"/></td><td>65.750</td><td>6.320</td><td>5</td><td>49</td></tr><tr><td>58</td><td>TEB</td><td><img src="images/a74c74598a5f99564897a5bf8588b1f772a6b9255642ba41ee53eeeee942433b.jpg"/></td><td>65.750</td><td>6.320</td><td>5</td><td>50</td></tr><tr><td>65</td><td>TMSPB</td><td><img src="images/d92727c836fe847df4b7483a0a152756cabb565a0d31defbbad6c4c28f32a3af.jpg"/></td><td>194.115</td><td>2.914</td><td>5</td><td>51</td></tr><tr><td>71</td><td>MDMB</td><td><img src="images/2509af0d572aed8a4cb4800e0732c2af2ec3aef546a4dfb14fbf6aafc3694551.jpg"/></td><td>192.067</td><td>2.911</td><td>5</td><td>52</td></tr><tr><td>74</td><td>3F-PBF</td><td><img src="images/b33e5ee9c02753da1ceedb02bb5c3cfad556ff6d6e6da9dc24bcb53b1872d68c.jpg"/></td><td>164.898</td><td>3.899</td><td>5</td><td>41</td></tr><tr><td>78</td><td>LUTID</td><td><img src="images/dbaf4d1b9a4bb5b3454b915fb768663b1897a584bc1b645e6d8f422a6a75e7d9.jpg"/></td><td>174.962</td><td>3.700</td><td>5</td><td>41</td></tr><tr><td>86</td><td>PBF</td><td><img src="images/4147617c70084f086878d7e402f341db3dc455d49bf5a141a66c6c148de6c40d.jpg"/></td><td>146.908</td><td>2.415</td><td>5</td><td>41</td></tr><tr><td>91</td><td>TRIZIN</td><td><img src="images/c6631e0d7f7872b35d4c36c4dacc2d8424bfb64ae8ac858928166e95547a00ce.jpg"/></td><td>284.496</td><td>5.317</td><td>5</td><td>53</td></tr><tr><td>96</td><td>K2B4O5(OH)4</td><td><img src="images/f3061e171e5fb02c1be3d83345397f0f7d3df12420f3a680fbe18e74b9b2fb4b.jpg"/></td><td>704.412</td><td>6.036</td><td>5</td><td>54</td></tr><tr><td>103</td><td>VPBF</td><td><img src="images/6750df630cc573caeb9248b8ded00e641c4f6508fe4ed45bc24ed306ea089ba5.jpg"/></td><td>172.946</td><td>2.874</td><td>5</td><td>41</td></tr><tr><td>106</td><td>DEBA</td><td><img src="images/9945af05403a9f7f5b9dd2edaa55f4a634c76eeadab641bda11101bcd251d87e.jpg"/></td><td>99.926</td><td>4.079</td><td>5</td><td>55</td></tr><tr><td>111</td><td>PCN</td><td><img src="images/933c85cc422cb82fe08d7e6d6b91a6eea556ae5001ed48860097618f3a582420.jpg"/></td><td>171.918</td><td>2.711</td><td>5</td><td>41</td></tr><tr><td>121</td><td>LiB5AB</td><td><img src="images/12a33097fa60c0182ab56774b0a4562bc552edbace97cf886c472876a7766db5.jpg"/></td><td>385.906</td><td>3.065</td><td>5</td><td>56</td></tr><tr><td>124</td><td>2F-PBF</td><td><img src="images/b7c349a752c8190c6bfcc9e65da23bb3a6b947e1dfa41c89d519bab9deb6bc63.jpg"/></td><td>164.898</td><td>2.703</td><td>5</td><td>41</td></tr><tr><td>13</td><td>(C3HF6O)2B(C6H5 )</td><td><img src="images/4f7f45615a7ad8659bd3be722a2bb5df3557423f75289350ef2ffd34fc09e098.jpg"/></td><td>421.974</td><td>3.417</td><td>6</td><td>34</td></tr><tr><td rowspan="2">17</td><td rowspan="2">THFBuBO <img src="images/819a4ca6dee5548586199163b459fb6810a3b0725bf03ccdb9f5fa5b3a06f0eb.jpg"/></td><td rowspan="2"></td><td rowspan="2">607.947</td><td rowspan="2">3.432</td><td rowspan="2">6</td><td rowspan="2">57</td></tr><tr><td></td></tr><tr><td>25</td><td>(C3HF6O)2B(C6F5)</td><td><img src="images/eebfec843f963d63df2129ca1357b72e949d85881b3664fa2ada4f25c40a01c8.jpg"/></td><td>440.972</td><td>3.547</td><td>6</td><td>34</td></tr><tr><td>31</td><td>(C3HF6O)2B(C6H3 F2)</td><td><img src="images/2c103265f8f06e4ce91e4e5d3f6a8eded52a90c0ee46c5c6b147df10037b534f.jpg"/></td><td>457.954</td><td>3.594</td><td>6</td><td>34</td></tr><tr><td>46</td><td>TFEB</td><td><img src="images/e56b3cb73125c85faa8653798c4493cbf16fc2652e48d0bb7083074d88a5ca99.jpg"/></td><td>307.905</td><td>3.422</td><td>6</td><td>43</td></tr><tr><td>63</td><td>B(HFIP)3</td><td><img src="images/e572e2b5871ccf078088fce42c58ea4005d55d689e84490e98e87ee6f709c14d.jpg"/></td><td>511.896</td><td>3.443</td><td>6</td><td>58</td></tr><tr><td>68</td><td>Na[B(hfip)4]</td><td><img src="images/8ed8359f3ca0cabdd6ab9fa0f6e38dd7e105942ee55f1da2ebcebc8aad70bfc6.jpg"/></td><td>701.914</td><td>3.743</td><td>6</td><td>59</td></tr><tr><td>90</td><td>THFB</td><td><img src="images/acbaca05ea5491f7f8aba96aa928243da5b2b4a5dedf791f79d0be0684b6bcd4.jpg"/></td><td>511.896</td><td>3.687</td><td>6</td><td>57</td></tr><tr><td>93</td><td>TTFEB</td><td><img src="images/cc0d16fd7692b4b0f0660bda334e0520b5f06515148470de4e88ff1643eeeb02.jpg"/></td><td>307.905</td><td>3.422</td><td>6</td><td>60</td></tr><tr><td>99</td><td>TMF</td><td><img src="images/7c71e7a6c15dced2c0e4c40580243b03ceb0e8e32f4e89b2ec7e05935f862a05.jpg"/></td><td>613.863</td><td>3.644</td><td>6</td><td>22</td></tr><tr><td>108</td><td>TTFB</td><td><img src="images/32457048a75e0ac530568138f7f3a9ef2ce839b4a0ccd70010ffcb9e3cb1e692.jpg"/></td><td>307.905</td><td>3.422</td><td>6</td><td>61</td></tr><tr><td>119</td><td>ABA#1</td><td><img src="images/2351e2c3911a88e9275ecd96e47564ccdae8063df1f3faa01fba6ffb80868532.jpg"/></td><td>511.896</td><td>3.443</td><td>6</td><td>62</td></tr><tr><td>123</td><td>THFPB</td><td><img src="images/664b0fd06af145127bd1ecc65b3a4f77e8e9ac1fb7726eb03da0b6ccc97835a4.jpg"/></td><td>511.896</td><td>3.443</td><td>6</td><td>63</td></tr><tr><td>125</td><td>THB</td><td><img src="images/e7ea175f66a2719da1594064358a7df9a3b9d9b02a8888d96762b65abc2eb2fb.jpg"/></td><td>511.896</td><td>3.443</td><td>6</td><td>64</td></tr><tr><td>0</td><td>TMSB</td><td><img src="images/8939df588d4b3893cb917143553e500c3d063e016d1fed4107e4f0e47d026cb6.jpg"/></td><td>278.382</td><td>3.985</td><td>7</td><td>65</td></tr><tr><td>1</td><td>TCEB</td><td><img src="images/3008a5dd9b5c65c1d21721bfdec3a9af6a77fa2a1c76aeb8656f4e7fcfc04b6f.jpg"/></td><td>221.025</td><td>3.847</td><td>7</td><td>66</td></tr><tr><td>2</td><td>TMBx</td><td><img src="images/4a6965643409f28edd10de1284d831926730f0e0a7b8eaac805faf8bf40893ab.jpg"/></td><td>173.535</td><td>5.202</td><td>7</td><td>67</td></tr><tr><td>16</td><td>TiPBx</td><td><img src="images/266076b774564f3bbd0e4a55176a85a428486130da2c4fd54b835281ee184b6e.jpg"/></td><td>257.697</td><td>4.618</td><td>7</td><td>67</td></tr><tr><td>20</td><td>L-LiMCFB</td><td></td><td>326.046</td><td>3.152</td><td>7</td><td>10</td></tr><tr><td>22</td><td>TME</td><td><img src="images/b561f48c83acc809e2164e45f1a0e6ab1c1c34db0a8e5505cd365b544cdf909b.jpg"/></td><td>188.076</td><td>3.621</td><td>7</td><td>22</td></tr><tr><td>26</td><td>HBC</td><td><img src="images/82d3f8628be9d3d7cbd6a62498581d0483e0f97a37ccde35de144ab01ed14bf9.jpg"/></td><td>1455.489</td><td>4.462</td><td>7</td><td>68</td></tr><tr><td>27</td><td>TnPBx</td><td><img src="images/f963f89fe7794b1dc99e102cae4bd7b359f632a82c6ba0f73ae4e9a4901453f0.jpg"/></td><td>257.697</td><td>4.219</td><td>7</td><td>67</td></tr><tr><td>29</td><td>LBC</td><td><img src="images/14b8797486ee76beda1d8600a684582d636a391802fee8be3c91346f72fd27cc.jpg"/></td><td>1059.012</td><td>3.722</td><td>7</td><td>68</td></tr><tr><td>40</td><td>LiT4PAB</td><td><img src="images/26a3ff338823ab36e736fb7047ecdf07b3d6eec9b09b880c9b48a11374611ce3.jpg"/></td><td>414.189</td><td>3.856</td><td>7</td><td>69</td></tr><tr><td>47</td><td>BF3·DE</td><td><img src="images/a915a45932f3cef5e7d2e279f427092a60b2001671314a132ebde7457116f169.jpg"/></td><td>141.929</td><td>2.963</td><td>7</td><td>70</td></tr><tr><td>48</td><td>BPC</td><td><img src="images/d744c2f8f254709acc6982183e8164122f096dfd5a8b988738bdc3ed8ff78526.jpg"/></td><td>1060.984</td><td>3.801</td><td>7</td><td>71</td></tr><tr><td>64</td><td>TPB</td><td><img src="images/0b759067ad610d3efe73361d11be833a1d911f27b8d614870742a4e2cd8552d0.jpg"/></td><td>188.076</td><td>3.388</td><td>7</td><td>72</td></tr><tr><td>72</td><td>TAB</td><td><img src="images/4bda34446a9b33316e8372cbeff272e55e85efcf8744b92a04407083d5520f06.jpg"/></td><td>342.197</td><td>3.497</td><td>7</td><td>73</td></tr><tr><td>79</td><td>TMB</td><td><img src="images/17db8040aa10b6954a7dd657edcf2c4ccafd433ea8f7b0644db988956ca3fec7.jpg"/></td><td>125.538</td><td>5.382</td><td>7</td><td>74</td></tr><tr><td>80</td><td>B-HEMA</td><td><img src="images/7dfd88d052349c46c4e4fdfe4de9f44b743252b827bb36469040dfedded1d0ed.jpg"/></td><td>398.217</td><td>3.249</td><td>7</td><td>68</td></tr><tr><td>100</td><td>PBA</td><td><img src="images/82910984e65e13fe6ac28bcce517e9fc27f7faaf211ca997201269edcabb9145.jpg"/></td><td>137.931</td><td>2.255</td><td>7</td><td>75</td></tr><tr><td>114</td><td>TBB</td><td><img src="images/803a3199051ac3ee8a848a5d860b77ab9096fce858e3cc687ce8100a5aed2395.jpg"/></td><td>230.157</td><td>2.858</td><td>7</td><td>76</td></tr><tr><td>117</td><td>TEBx</td><td><img src="images/e86bd0bf53e1d44a842704182d5f8eb0bd50b7321f6570ba9c7001ec94d16f44.jpg"/></td><td>215.616</td><td>4.804</td><td>7</td><td>67</td></tr></table>

The missing compound ID was filtered out as an outlier during the clustering process.  
Table S4 | Shared motif summary table

<table><tr><td>Legend in Fig. 3e</td><td>Motif name</td><td>Motif structure</td><td>Appears in the prototype</td></tr><tr><td>P1-SM</td><td>C-B(OR)2 motif</td><td><img src="images/46462dbe3bfaf2e0629fd9048484f195e9337765071bc2d6b91a53d3153171a3.jpg"/></td><td>P1, P2, P3, P4, P5, P6</td></tr><tr><td>P2-SM</td><td>Pentafluorophenyl group</td><td><img src="images/29a2dac3e4fc9eddabf3ff82e075a2d8c1720694b5b5f40eb8d328c5dda1ec18.jpg"/></td><td>P2</td></tr><tr><td>P3-SM</td><td>Boron oxalate</td><td><img src="images/c3e080cbfd0c4663a08b3c8a66bba8869a2b7ea3cb2a1c350f734eca8087c776.jpg"/></td><td>P2, P3</td></tr><tr><td>P4-SM</td><td>Catechol boronate</td><td><img src="images/08eeef13af3ea64eea57579e868e6d12a4dd885dba3954879cfbd2f07dd01cbe.jpg"/></td><td>P2, P3, P4</td></tr><tr><td>P5-SM</td><td>BF3-like motif</td><td><img src="images/47ea95a794d8bbb622a0575196a20fc7c7bdcb70ce356deb22119e86d644f313.jpg"/></td><td>P1, P5, P7</td></tr><tr><td>P6-SM</td><td>Fluorinated alkoxy</td><td><img src="images/97f670eac904cfdef5a0a6f52a1a33b5303df97782f34de95786ffb11861a0f4.jpg"/></td><td>P6</td></tr><tr><td>P7-SM</td><td>Tri-alkoxy borate</td><td><img src="images/24a460f9d7869e9c0d4cb21eaf896e140e348177dd61ad669fb088683b0088b6.jpg"/></td><td>P1, P2, P3, P4, P5, P6, P7</td></tr><tr><td>8-SM</td><td>Tetra-alkoxy borate</td><td><img src="images/ce1cacfd0562b8de4bf864065ddc7b7e6b84de179512380503768269a71b47c8.jpg"/></td><td>P2, P3, P4, P6, P7</td></tr></table>

The shared molecular framework was identified using the SMART77 method.

![](images/41cab479f4a497e8fb4ad93b627edb53505b74b6aaaa11b1b13fc8a461b62c0c.jpg)  
Figure S4 | Mechanism annotation statistics of the seven learned molecular prototypes. Quantitative mechanism annotation supporting the functional profiles in Fig. 3c. Heatmap showing the normalized frequency of seven electrolyte-regulation mechanisms across the seven learned molecular prototypes. Frequencies were calculated by dividing the number of mechanism-related records by the total number of annotated records in each prototype.

Table S5 | Prototypes summary table
<table><tr><td>Prototype</td><td>Number of molecules</td><td>Representative molecule</td><td>Dominant motifs</td></tr><tr><td>P1</td><td>89</td><td>DiOB-Py</td><td>C-B(OR)2 motif</td></tr><tr><td>P2</td><td>64</td><td>TPFPB</td><td>pentafluorophenyl</td></tr><tr><td>P3</td><td>78</td><td>LiDFOB</td><td>group tetra-alkoxy borate</td></tr><tr><td>P4</td><td>117</td><td>LBBB</td><td>catechol boronate</td></tr><tr><td>P5</td><td>131</td><td>PCN</td><td>BF3-like motif</td></tr><tr><td>P6</td><td>92</td><td>TTFEB</td><td>fluorinated alkoxy</td></tr><tr><td>P7</td><td>109</td><td>TBB</td><td>tri-alkoxy borate</td></tr></table>

The population is calculated after the graph augmentation.

![](images/609cbcb88a614e476820aa03b808a20663d96a8c92ab41423f1daad2c020b619.jpg)

Figure S5 | Prototype correspondence between initial and transferred prototype centroids. Heatmap showing the mean cosine similarity between the initial prototype centroids at epoch 1 and the transferred prototype centroids after PCL training at epoch 300. Rows denote transferred prototypes, and columns denote initial prototypes. The strong diagonal similarity indicates that most transferred prototypes preserve clear correspondence with their initial prototype identities, whereas structured off-diagonal similarities reflect partial reorganization and information sharing among related prototype regions during adaptation. These results support that PCL maintains prototype-level continuity while allowing the prototype organization to adjust to the unlabeled candidate chemical space.  
![](images/e2b008063ebc53adfdf0251c9ab36a1f90a3383eb7b0f8a927c25211a24b5147.jpg)

Figure S6 | Structural motif distribution across transferred prototypes. Stacked bar plot showing the distribution of representative boron-containing structural motifs across transferred prototypes after PCL adaptation. Motif proportions were calculated from molecules assigned to each new prototype using predefined SMiles ARbitrary Target Specification (SMARTS) patterns, including tetra-alkoxy borate, tri-alkoxy borate, boron oxalate, BF3-like motifs, fluorinated alkoxy groups, C-B(OR)2 motifs, pentafluorophenyl groups and catechol boronate. The enrichment of different motifs in distinct transferred prototypes indicates that PCL preserves chemically meaningful structural information during prototype adaptation rather than forming arbitrary clusters in the candidate space. Because a molecule may contain more than one motif, stacked heights represent cumulative motif occurrence rather than mutually exclusive molecular classes.  
![](images/97d798f10cec52f863df6fcdec3ac64a6be5fb2fdb39b94e3cd0dc41c4cc61d3.jpg)  
Figure S7 | Distribution of prototype similarity across recommendation strategies. Violin plots showing the distribution of molecule-level prototype similarity for molecules recommended by different strategies after translation strategy. Each point represents one recommended molecule, and box plots indicate the median and interquartile range. ProtoMI shows the highest overall prototype similarity among the compared strategies, indicating that prototype-guided adaptation enriches molecules that remain close to the learned additive prototype regions. The red star highlights ProtoMI achieves the highest prototype similarity among all recommendation methods.

![](images/d6588b39568ef25e4550877fb92cad8176624c2ed6fdcdca08ddebf6e4ed34b9.jpg)  
Figure S8 | Prototype occupancy of molecules recommended by different strategies. Heatmap showing the occupancy of recommended molecules across the seven learned molecular prototypes after translation strategy. Rows correspond to different recommendation strategies, and columns correspond to assigned prototypes P1–P7. Values indicate the percentage of molecules assigned to each prototype within a given strategy, with each row normalized to 100%. ProtoMI distributes recommended molecules across multiple prototype regions while retaining high prototype similarity, suggesting that the model avoids collapse into a single prototype and preserves chemically diverse recommendation coverage.

![](images/d7cd6b9eeeec131dfa97220a6f441666ec4fea337f24fbf0015250f9ff4ec0fb.jpg)  
Figure S9 | Similarity of recommended molecules to reported additives. Distribution of the maximum structural similarity between recommended molecules and reported boron-containing electrolyte additives across different recommendation strategies. Each violin plot shows the molecule-level known-additive similarity after post-screening, with box plots indicating the median and interquartile range. ProtoMI exhibits a broad similarity distribution, including both candidates close to reported additives and candidates with lower similarity, suggesting that the model transfers prototype-level chemical knowledge without simply reproducing close analogues of known molecules.

Table S6 | Extra recommendation statistics
<table><tr><td>Method</td><td>Entropy- weighted prototype similarity</td><td>Novelty score</td><td>High similarity ratio &gt;0.75</td><td>Moderate similarity ratio 0.40- 0.75</td><td>Low similarity ratio &lt;0.40</td><td>Post- screening survival</td></tr><tr><td>Random selection</td><td>27.637</td><td>0.6435 ± 0.0910</td><td>0.62%</td><td>24.42%</td><td>74.96%</td><td>2.10%</td></tr><tr><td>Morgan fingerprints</td><td>41.600</td><td>0.6304 ± 0.0875</td><td>0.75%</td><td>29.10%</td><td>70.15%</td><td>5.18%</td></tr><tr><td>UCL encoder only</td><td>2.985</td><td>0.6323 ± 0.0889</td><td>0.43%</td><td>30.70%</td><td>68.87%</td><td>5.26%</td></tr><tr><td>PCL encoder only</td><td>29.466</td><td>0.6206 ± 0.1127</td><td>1.73%</td><td>29.21%</td><td>69.06%</td><td>1.31%</td></tr><tr><td>UCL encoder clustering</td><td>2.079</td><td>0.6762 ± 0.0607</td><td>0.00%</td><td>10.32%</td><td>89.68%</td><td>0.50%</td></tr><tr><td>PCL encoder clustering</td><td>24.364</td><td>0.6551 ± 0.0744</td><td>0.00%</td><td>22.29%</td><td>77.71%</td><td>1.10%</td></tr><tr><td>ProtoMI</td><td>57.121</td><td>0.5975± 0.2446</td><td>12.68%</td><td>25.35%</td><td>61.97%</td><td>0.23%</td></tr><tr><td>w/o EMA</td><td>9.456</td><td>0.6722 ± 0.1029</td><td>1.22%</td><td>12.91%</td><td>85.86%</td><td>1.86%</td></tr><tr><td>w/o decorrelation</td><td>0.299</td><td>0.6270 ± 0.0817</td><td>0.75%</td><td>30.21%</td><td>69.04%</td><td>5.65%</td></tr><tr><td>w/o top-k</td><td>0.000</td><td>0.6558 ± 0.2060</td><td>7.07%</td><td>23.91%</td><td>69.02%</td><td>0.96%</td></tr></table>

Table S7 | Full temporal validation
<table><tr><td>Cutoff year</td><td>Recall</td><td>Precision</td><td>F1 score</td><td>Enrichment factor</td><td>Reduction factor</td><td>Hit number</td></tr><tr><td>2017</td><td>0.10638</td><td>0.00274</td><td>0.00535</td><td>10.52583</td><td>98.94282</td><td>5/47</td></tr><tr><td>2019</td><td>0.11627</td><td>0.00219</td><td>0.00430</td><td>9.18278</td><td>78.97191</td><td>5/43</td></tr><tr><td>2021</td><td>0.08823</td><td>0.00862</td><td>0.01570</td><td>45.63311</td><td>517.17528</td><td>3/34</td></tr><tr><td>2023</td><td>0.20000</td><td>0.00484</td><td>0.00946</td><td>43.63078</td><td>218.15393</td><td>4/20</td></tr></table>

![](images/dba1df08ca484eab64bf9f5da4a9abf59ccf0405e8282fcac51dbabd4fc80569.jpg)  
Figure S10 | Prototype survival during translation from model recommendations to experimentally actionable candidates. Sankey diagram showing the retention and redistribution of new prototype assignments during the translation strategy from PCL-recommended molecules to experimentally actionable candidates. The initial PCL recommendation pool contained 30,800 molecules distributed across seven new prototypes. Sequential filtering based on molecular validity, molecular weight, synthetic accessibility, active hydrogen exclusion, CAS availability and practical cost considerations reduced the pool to 68 translation candidates. Further prioritization according to commercial availability and experimental feasibility yielded four candidates for electrochemical validation, all assigned to NP3. Flow widths are proportional to the number of molecules retained from each prototype at each filtering stage.

Table S8 | 68 Candidates after translation strategy
<table><tr><td>Index</td><td>Compound ID</td><td>Formula</td><td>Molecular weight</td><td>SA Score</td><td>Prototype label</td><td>CAS number</td><td>Reported or not</td></tr><tr><td>1</td><td>11090867</td><td><img src="images/5a3a7329f349f96018e172394c5f3bf01c6f97bd46e60acca1b65c9f4cbb4834.jpg"/></td><td>409.053</td><td>3.177</td><td>1</td><td>119618-67-6</td><td>No</td></tr><tr><td>2</td><td>15877487</td><td><img src="images/d3e43445d2cc2d72deb40c878e6775c38b757d6ac826cbd3e9ad380d033539be.jpg"/></td><td>180.095</td><td>2.739</td><td>1</td><td>253280-01-2</td><td>No</td></tr><tr><td>3</td><td>57041839</td><td><img src="images/55632437ba88ff1ebfba7a2d9f8c4e4187971d428086959c9326e812a22fc291.jpg"/></td><td>292.148</td><td>3.585</td><td>1</td><td>62043-02-1</td><td>No</td></tr><tr><td>4</td><td>71390453</td><td><img src="images/28953aef1a121bc6dc0b9b040ed64b64f95a3d7827d52abd55702c76a1a22f40.jpg"/></td><td>292.148</td><td>3.624</td><td>1</td><td>62043-03-2</td><td>No</td></tr><tr><td>5</td><td>78020</td><td><img src="images/ee919ff1ae44420956ea14530b5e25be32ad7e76a2d20f895178b87e6e01480c.jpg"/></td><td>278.136</td><td>3.984</td><td>1</td><td>4325-85-3</td><td>Yes</td></tr><tr><td>6</td><td>425307</td><td><img src="images/7c9e113e36ca0edffca7f0f5fa8734608218588bd8f66e0d27345f91c2b56068.jpg"/></td><td>188.049</td><td>3.456</td><td>1</td><td>4887-24-5</td><td>No</td></tr><tr><td>7</td><td>12418354</td><td><img src="images/1db7b96b78cfd11dc97efedcc48795d4b76a4b3eee6f143c5da373574652642a.jpg"/></td><td>266.132</td><td>3.356</td><td>1</td><td>7560-51-2</td><td>No</td></tr><tr><td>8</td><td>22338812</td><td><img src="images/398cb0851e6a5538121af9e9be3eedaa0feb6212cdca947029f72b488663ef0d.jpg"/></td><td>266.092</td><td>3.121</td><td>1</td><td>55923-08-5</td><td>No</td></tr><tr><td>9</td><td>56973279</td><td><img src="images/dd6b017ad82aaeced6ade967ddb6187525859dee7c30d88cf47b1d3ac46e8225.jpg"/></td><td>344.957</td><td>3.429</td><td>1</td><td>1356165-73- 5</td><td>No</td></tr><tr><td>10</td><td>517955</td><td><img src="images/a8eff5afa5a066596be18587a8511cf4d63c9a42a59a6a9405715ed23881c428.jpg"/></td><td>69.940</td><td>3.831</td><td>1</td><td>1113-22-0</td><td>No</td></tr><tr><td>11</td><td>138247</td><td><img src="images/f8dc726794b8422752e7025da45f32f6ca0edda1dc8fc240ff9bdf695dbcfc25.jpg"/></td><td>71.920</td><td>3.989</td><td>1</td><td>4443-43-0</td><td>No</td></tr><tr><td>12</td><td>13301362</td><td><img src="images/130b030df3ac556e12ef54f6278595e0bc4fd8f7f5c05e3ca58c4c15a46468b9.jpg"/></td><td>222.050</td><td>3.060</td><td>1</td><td>94839-08-4</td><td>No</td></tr><tr><td>13</td><td>13789232</td><td><img src="images/1c2cb40467bc658cd9ef91e53faa8688648302bf5c0c57012cbe38d99f64e695.jpg"/></td><td>239.500</td><td>3.942</td><td>1</td><td>107134-81-6</td><td>No</td></tr><tr><td>14</td><td>68979</td><td><img src="images/12379c19ad26015798b2d13df3dc2103dcc98569f821dee51d992fe84e59e7a8.jpg"/></td><td>55.920</td><td>3.834</td><td>1</td><td>593-90-8</td><td>No</td></tr><tr><td>15</td><td>25068199</td><td><img src="images/dc00c7ac657ee7b4ec028b1a705fb0aa838e88c1a291bbe99598408c190ef5e4.jpg"/></td><td>405.100</td><td>3.195</td><td>1</td><td>951163-66-9</td><td>No</td></tr><tr><td>16</td><td>66709077</td><td><img src="images/9b447e8c5737a957b6ed971f1bf902ae67a4e38d0b076e0bbdd579e690537c6b.jpg"/></td><td>182.122</td><td>3.664</td><td>2</td><td>1392108-88- 1</td><td>No</td></tr><tr><td>17</td><td>15158954</td><td><img src="images/ac0c0b3c255f6ad47a5d750eb27ace4de398664d608e6852d092ab1e302364b4.jpg"/></td><td>148.020</td><td>3.571</td><td>2</td><td>139298-13-8</td><td>No</td></tr><tr><td>18</td><td>11522786</td><td><img src="images/32290acb02c740e8b798adb3146e2150244c4b604dcd7c45a9c2dff60258965e.jpg"/></td><td>331.000</td><td>2.822</td><td>2</td><td>872044-92-3</td><td>No</td></tr><tr><td>19</td><td>22663171</td><td><img src="images/724dac252fd901897c3770f6d35eed7235c5a26f826b6be64843865c8ee7127f.jpg"/></td><td>193.120</td><td>3.577</td><td>2</td><td>141938-41-2</td><td>No</td></tr><tr><td>20</td><td>25112575</td><td><img src="images/d1f65de3a1ec761f136800baa5ffcc813beed36d08f1a45aeb49783b9356e7a8.jpg"/></td><td>556.257</td><td>2.677</td><td>3</td><td>624744-67-8</td><td>No</td></tr><tr><td>21</td><td>58050304</td><td><img src="images/fea7e1c9cae7607c06d715a59802abca377562ffc5a5ef238c2076fb07415053.jpg"/></td><td>430.210</td><td>2.540</td><td>3</td><td>1149804-35- 2</td><td>No</td></tr><tr><td>22</td><td>58489416</td><td><img src="images/76ac2b678f01685bdb3b93c528b8e2ec92d6bd216c2a01b56856f20f2091b24a.jpg"/></td><td>430.210</td><td>2.524</td><td>3</td><td>1115639-92- 3</td><td>No</td></tr><tr><td>23</td><td>23115783</td><td><img src="images/4b5bc48fe39c3f67ca9711a9f6d1b1591fc6adc5d659c51df3668b421aa52538.jpg"/></td><td>502.500</td><td>2.529</td><td>3</td><td>281668-51-7</td><td>No</td></tr><tr><td>24</td><td>58769449</td><td><img src="images/e3ec217a378ae53e625a8c8e52eaeaeb3c0ea3f408e6c3e8b40377c7a9c7db8c.jpg"/></td><td>354.300</td><td>2.472</td><td>3</td><td>890042-13-4</td><td>No</td></tr><tr><td>25</td><td>21425</td><td><img src="images/75de60fbe6805b508ae447e17f002b385e4008fbb3731e52bbe4facf16b27443.jpg"/></td><td>818.862</td><td>2.456</td><td>5</td><td>5337-41-7</td><td>No</td></tr><tr><td>26</td><td>125587</td><td><img src="images/c04f7fee2862dc745e313441b413193cb1cca33e0dd2e5144863af3835eb2440.jpg"/></td><td>224.200</td><td>3.947</td><td>5</td><td>32327-52-9</td><td>No</td></tr><tr><td>27</td><td>13680338</td><td><img src="images/7a432760b481c523e48d294fc0afccf625c9370956f436dad4d0595b0afc6c91.jpg"/></td><td>238.300</td><td>3.883</td><td>5</td><td>16413-17-5</td><td>No</td></tr><tr><td>28</td><td>71384846</td><td><img src="images/4257a940f309bcf967e3526a82efefc6973b520a2fff5bf20e94ccaa53da96c9.jpg"/></td><td>350.500</td><td>3.582</td><td>5</td><td>62594-01-8</td><td>No</td></tr><tr><td>29</td><td>71384845</td><td><img src="images/d065ddef72ef335dede4083307d78e9aacff0df767ef25c4c5af2294cde2c218.jpg"/></td><td>350.500</td><td>3.983</td><td>5</td><td>62594-02-9</td><td>No</td></tr><tr><td>30</td><td>71334465</td><td><img src="images/57e9eff5c240971e83c4e23fee355a48d9210ab004fc27de38151558487d7ce8.jpg"/></td><td>560.900</td><td>2.331</td><td>5</td><td>106787-52-4</td><td>No</td></tr><tr><td>31</td><td>71386315</td><td><img src="images/d51a5f7cf718e240487aa3b00b176a179dae52089a21b4c6c8838afc5b94ae0f.jpg"/></td><td>460.700</td><td>2.844</td><td>5</td><td>62444-57-9</td><td>No</td></tr><tr><td>32</td><td>71352225</td><td><img src="images/78ace3ed09f0386f950374f6b941dc9dc3ec9a5d82333495b9bb8e3f626f45ae.jpg"/></td><td>476.700</td><td>2.070</td><td>5</td><td>1188-97-2</td><td>No</td></tr><tr><td>33</td><td>352104</td><td><img src="images/6a42a5083f3c19afdf8e8afb6b9c38ac3543b5c646c6ae3d63b3032700968233.jpg"/></td><td>518.799</td><td>2.065</td><td>5</td><td>14245-38-6</td><td>No</td></tr><tr><td>34</td><td>71346374</td><td><img src="images/0d16f4ffb5e548d7cb0b4473027df511c43ea79d688b7648b4be25404c1b75f4.jpg"/></td><td>211.180</td><td>3.403</td><td>5</td><td>151293-65-1</td><td>No</td></tr><tr><td>35</td><td>17177</td><td><img src="images/71bcfea52f95139277aa2833c376ce4f12a75abcc777868639c2a3619ceef965.jpg"/></td><td>398.500</td><td>3.931</td><td>5</td><td>2467-13-2</td><td>No</td></tr><tr><td>36</td><td>21424</td><td><img src="images/ceb500083b810f839785fc7c3248f46822662612cce0dd44ea6c9ac8a10f2ebb.jpg"/></td><td>314.300</td><td>3.996</td><td>5</td><td>5337-37-1</td><td>No</td></tr><tr><td>37</td><td>17178</td><td><img src="images/9589c9cbd61609b7b1fd99d43c143e144cddf067e31c9c7318dc9ad9fe67b8db.jpg"/></td><td>566.800</td><td>2.247</td><td>5</td><td>2467-15-4</td><td>No</td></tr><tr><td>38</td><td>219443</td><td><img src="images/1bb65a1d7ba8d24ddb70dbb403c474656c1571508090c50bdea29a7ceb7c2c15.jpg"/></td><td>440.600</td><td>3.010</td><td>5</td><td>3088-77-5</td><td>No</td></tr><tr><td>39</td><td>108841</td><td><img src="images/8aa00f46c8dda842c11b9e225af18a7d03a727013797e59af65c0747d3583a9b.jpg"/></td><td>608.900</td><td>2.480</td><td>5</td><td>59802-06-1</td><td>No</td></tr><tr><td>40</td><td>15605783</td><td><img src="images/178d4b7f77ee4dcf3b39486dbf6ad338b5d7f6f18e5ca5e2371f07bdcc54b89b.jpg"/></td><td>296.100</td><td>3.454</td><td>6</td><td>267006-38-2</td><td>No</td></tr><tr><td>41</td><td>18373965</td><td><img src="images/80f416dcc0df6f55cf717562dbb7abcde5825c8389908bfbf08707e9a1f51d1a.jpg"/></td><td>240.038</td><td>3.337</td><td>6</td><td>848609-02-9</td><td>No</td></tr><tr><td>42</td><td>582056</td><td><img src="images/147fdffa32fe15a3fe2517ec1945672ef5cf27233fd56d972c40df756266c118.jpg"/></td><td>511.985</td><td>2.800</td><td>6</td><td>1109-15-5</td><td>Yes</td></tr><tr><td>43</td><td>11727684</td><td><img src="images/bb55b3851bb4a747d947d9549d1ead81c8c4e5afae378e45ef3f1d3c6e71b23c.jpg"/></td><td>344.993</td><td>2.819</td><td>6</td><td>165612-94-2</td><td>No</td></tr><tr><td>44</td><td>75360847</td><td><img src="images/0daf68489a77d2796e3d88cdb0090d84bb7e8c3ae74a79b84fbd8600e006090c.jpg"/></td><td>458.017</td><td>3.449</td><td>6</td><td>2003-04-5</td><td>Yes</td></tr><tr><td>45</td><td>10959856</td><td><img src="images/2fbe3f9cb0c20d3752800780ac49da6fa880d7d158ff5b98736cad5cc8042603.jpg"/></td><td>380.380</td><td>3.032</td><td>6</td><td>2720-03-8</td><td>No</td></tr><tr><td>46</td><td>11115716</td><td><img src="images/2ad6d4113623c9fd431b6e196a27fa25027f5f9d08e2642696e2997e740c75e4.jpg"/></td><td>359.960</td><td>2.958</td><td>6</td><td>170151-48-1</td><td>No</td></tr><tr><td>47</td><td>12626668</td><td><img src="images/689b5b139e18f27be6b1a3f3d8e35fb31c3e8e710ae93e65666c1822030b6c81.jpg"/></td><td>248.770</td><td>3.405</td><td>6</td><td>830-48-8</td><td>No</td></tr><tr><td>48</td><td>11050988</td><td><img src="images/dcb0737a7d134d61857f4d8234800ceb8871e05d5a08fec3c23139fbabfbeed2.jpg"/></td><td>770.100</td><td>3.300</td><td>6</td><td>190282-03-2</td><td>Yes</td></tr><tr><td>49</td><td>15520532</td><td><img src="images/0af76de94bec7784cb099ec560a1a882ec736b21a98a7838cc8a90ed69a2bf1c.jpg"/></td><td>207.940</td><td>3.206</td><td>6</td><td>363596-54-7</td><td>No</td></tr><tr><td>50</td><td>10876447</td><td><img src="images/23ac48e0b20c9113aba061b3e761e470d133461c5bb6ec3407ddc43ffffe3468.jpg"/></td><td>770.100</td><td>3.354</td><td>6</td><td>204930-04-1</td><td>Yes</td></tr><tr><td>51</td><td>71428744</td><td><img src="images/ea69eb2068a58b4d0a3d9efe945c4ccb861e33170b503b0eeb955fea2103b6b7.jpg"/></td><td>494.000</td><td>3.182</td><td>6</td><td>916442-23-4</td><td>Yes</td></tr><tr><td>52</td><td>71420187</td><td><img src="images/193abae4cd40c503f8b9b74ebc3cb8bf07eb6451885659084a49cb2e8b1b6374.jpg"/></td><td>598.200</td><td>3.056</td><td>6</td><td>820972-26-7</td><td>No</td></tr><tr><td>53</td><td>71447200</td><td><img src="images/67569edd26a2aee88055662de1b038d71f228753c73ac2af462c9776af97e7e0.jpg"/></td><td>367.850</td><td>3.257</td><td>6</td><td>329009-01-0</td><td>No</td></tr><tr><td>54</td><td>71350376</td><td><img src="images/5578ad66adf860c7748a10c2f179c986269138277b84a3fe0fde9351380a46cb.jpg"/></td><td>448.100</td><td>3.152</td><td>6</td><td>165612-90-8</td><td>No</td></tr><tr><td>55</td><td>71428745</td><td><img src="images/0feb34c55d4747ad8906baa98821ca0e0a8b39fba789dcfa3798e9e4b671ede1.jpg"/></td><td>854.100</td><td>3.459</td><td>6</td><td>916336-48-6</td><td>Yes</td></tr><tr><td>56</td><td>4121732</td><td><img src="images/4c32b6a434bac056df59e8c3014fefd015924a061ffea0a339cb4239f3ba117c.jpg"/></td><td>221.150</td><td>3.178</td><td>6</td><td>4426-24-8</td><td>No</td></tr><tr><td>57</td><td>11327649</td><td><img src="images/e7afbf24e91b68f47e7ee5803ad15f1de9f2d3a5b6277e3688b300f3b1acf57a.jpg"/></td><td>422.000</td><td>2.897</td><td>6</td><td>148892-98-2</td><td>No</td></tr><tr><td>58</td><td>4072412</td><td><img src="images/c9db109baea1e8dd239c901376f9060e6c4de905286983417a5c9ed2825012d9.jpg"/></td><td>302.100</td><td>3.264</td><td>6</td><td>158752-98-8</td><td>No</td></tr><tr><td>59</td><td>144241</td><td><img src="images/196e3f008988cb2a82b983ca8772ce3a7bd00d558cbb0aaaedbe3e8ae25d128c.jpg"/></td><td>262.990</td><td>3.962</td><td>6</td><td>67395-53-3</td><td>No</td></tr><tr><td>60</td><td>22722373</td><td><img src="images/447ba43a7356a37bf41646297d47685e60eb4347dee8c9144fe8ea727c9eb6bc.jpg"/></td><td>332.100</td><td>2.614</td><td>6</td><td>154735-10-1</td><td>No</td></tr><tr><td>61</td><td>12159360</td><td><img src="images/f954585b3b7f048ce6303278fe201d4ebd278ac896129429a727eb0246415c10.jpg"/></td><td>837.900</td><td>3.357</td><td>6</td><td>223769-13-9</td><td>No</td></tr><tr><td>62</td><td>3599767</td><td><img src="images/b8de9b2bd09c41a3d839698616cd42ed8ee9ad1d9429bfdf87a338997a2de37b.jpg"/></td><td>307.910</td><td>3.422</td><td>6</td><td>659-18-7</td><td>Yes</td></tr><tr><td>63</td><td>18329245</td><td><img src="images/608f4a5cd0da4b2bf1a44fc2306e08c52fe3185469e834f42fdd281a27cfdbf7.jpg"/></td><td>1100.100</td><td>3.875</td><td>6</td><td>213974-85-7</td><td>No</td></tr><tr><td>64</td><td>2736203</td><td><img src="images/eb2523582d117d1b380809bb27b1a0d2f0c6c45f62fe44f2f43c0d8218678878.jpg"/></td><td>511.900</td><td>3.443</td><td>6</td><td>6919-80-8</td><td>Yes</td></tr><tr><td>65</td><td>54472851</td><td><img src="images/cd127cb733f546f6f2ec8c577e7990121d0016bd409929da549b6e3b21067c65.jpg"/></td><td>607.950</td><td>3.432</td><td>6</td><td>755-53-3</td><td>Yes</td></tr><tr><td>66</td><td>15887311</td><td><img src="images/808f7a629228fe8206257c3d14788cc0ea7afd0c2494328fa42dc8fae4d3370d.jpg"/></td><td>560.000</td><td>2.965</td><td>6</td><td>146355-12-6</td><td>Yes</td></tr><tr><td>67</td><td>52915087</td><td><img src="images/23abf943b6c8003f28aa6fd29f0e2b697f096190ef823c771c7266016ec101ee.jpg"/></td><td>403.960</td><td>3.776</td><td>6</td><td>2317-76-2</td><td>Yes</td></tr><tr><td>68</td><td>4438527</td><td><img src="images/5561312ef8c0589c9d7c8f529eedec65b54b191e8298d5ba0bc9f750f6fd3a82.jpg"/></td><td>679.000</td><td>2.839</td><td>6</td><td>47855-94-7</td><td>Yes</td></tr></table>

Table S9 | 4 final molecules
<table><tr><td>Compound ID</td><td>Molecular name</td><td>CAS</td><td>Formula</td><td>Price</td><td>Action</td></tr><tr><td>25112575</td><td>2-[9,10-Di(naphthalen-2-  $y 1 ) \tt a n t h r a c e n - 2 - y l l - 4 , 4 , 5 , 5 -$  tetramethyl-1,3,2- dioxaborolane</td><td>624744-67-8</td><td><img src="images/7f83fa545538561c0568a125883949849eb7b950a6d696d16e04ddd71bec3983.jpg"/></td><td> $2 4 0 . 9 \yen 19$ </td><td>Under preparation</td></tr><tr><td>58050304</td><td> $4 , 4 , 5 , 5 - 7 e t r a m e t h y 1 - 2 - 1 1 0 - ( 1 -$  naphthyl)anthracen-9-yl]-1,3,2- dioxaborolane</td><td>1149804-35-2</td><td><img src="images/86bbdbfed94d28bff58a3bd6625a425838aa98744fd5e013c36ac2daa42223de.jpg"/></td><td> $5 5 7 . 9 \yen 19$ </td><td>Select to do the validation experiment</td></tr><tr><td>58489416</td><td> $4 , 4 , 5 , 5 - 1 \mathrm { e t r a m e t h y } 1 - 2 - 1 3 -$   $\begin{array} { c } { { ( \mathrm { t r i p h e n y } | \mathrm { e n - } 2 \mathrm { - } \mathsf { y } | ) \mathrm { p h e n y } | ] \mathrm { - } 1 , 3 , 2 \mathrm { - } } } \\ { { \mathrm { d i o x a b o r o l a n e } } } \end{array}$ </td><td>1115639-92-3</td><td><img src="images/ba80c5a113b6b5155067ba4fe5a5342de146fd48e20c122f3e2a12aedb91a08c.jpg"/></td><td> $1 2 2 . 9 \yen 19$ </td><td>Under preparation</td></tr><tr><td>58769449</td><td> $4 { , } 4 { , } 5 { , } 5 { - } \mathsf { T e t r a m e t h y l { - } } 2 { - }$   $\begin{array} { c } { ( \mathrm { t r i p h e n y } | \mathbf { e n - } 2 \mathbf { - } \mathsf { y } | ) { \cdot } 1 , 3 , 2 \mathbf { - } } \\ { \mathrm { d i o x a b o r o l a n e } } \end{array}$ </td><td>890042-13-4</td><td><img src="images/61973bbeb67143bda294f04ba63ff08e20a8c871cdf18ccc1ff28dcc8606a2e1.jpg"/></td><td> $6 8 . 9 \yen 19$ </td><td>Under preparation</td></tr></table>

a  
![](images/9300129a5a86b47ec9b336c972b11fcfe389a98ff6a3441843d57da369f0d714.jpg)  
Binding energy (eV)

![](images/d0592da4029edb5b24a4780548aa2d50dfd299b97cee934c3e8a2b54989cb788.jpg)  
Binding energy (eV)

C  
![](images/47aea1f3e7eed29fa65e37a5944b2a6d08e5802eda1eab50847d6150e6360c63.jpg)  
d

![](images/2372b7ded380d2a1d73e4ef43608401a1115642ccce15f23609248d42006e7bc.jpg)  
Figure S11 | C 1s XPS analysis of graphite electrodes cycled with and without TNDB. High-resolution C 1s spectra and peak deconvolution of graphite electrodes recovered from the baseline electrolyte after (a) formation and (b) 100 cycles, and from the TNDB-containing electrolyte after (c) formation and (d) 100 cycles at $5 5 ^ { \circ } \complement$ The fitted components are assigned to $\mathtt { C - C , C - O , C = O }$ , lithium alkyl carbonates (ROCO2Li) and $\mathsf { L i } _ { 2 } \mathsf { C O } _ { 3 } .$ Open squares represent the experimental data, red curves the overall fits and colored curves the individual fitted components. Binding energies were calibrated to the C–C peak at 284.8 eV.

3D Render of ${ \mathsf { P O } } _ { 2 } ^ { - }$

3D Render of LiF2

3D Render of $\mathsf { C O } _ { 3 } ^ { - }$

3D Render of ${ \mathsf { C H } } _ { 3 } { \mathsf { C O } } ^ { - }$

![](images/18006cab8244010954224fe106efceb330c1bda4df4bddc97e4b105089de5ebd.jpg)  
Figure S12 | Three-dimensional ToF-SIMS compositional renderings of representative SEI fragments on graphite electrodes cycled at $5 5 \textdegree$ with and without TNDB. From left to right, the mapped negative secondaryion fragments are $P O _ { 2 } ^ { - } , L i F _ { 2 } ^ { - } , C O _ { 3 } ^ { - }$ and $C H _ { 3 } C O ^ { - }$ . The rows correspond, from top to bottom, to graphite electrodes recovered from the baseline electrolyte after formation and after 100 cycles, and from the TNDB-containing electrolyte after formation and after 100 cycles at $5 5 ^ { \circ } \complement$ . The renderings visualize the spatial and depth-dependent distributions of the selected fragments within the sputtered interphase volume.

![](images/dad8cb151f3348702e089eba6286b4ad75515bcf0495af4296a04f61324e05ba.jpg)

![](images/f1821cbb378cffc1f35d07c2acadffb5db8b087ee104763eb1901c509877ab5b.jpg)

Figure S13 | Depth-resolved distribution of additive-derived boron in the graphite SEI. X–Z cross-sectional ToF-SIMS maps of the $\mathsf { B } ^ { - }$ secondary-ion fragment obtained from graphite electrodes cycled in the TNDB-containing electrolyte after formation and after 100 cycles at $5 5 ^ { \circ } \complement .$ Color scales are independently adjusted for visualization.  
![](images/39768987873b59a53180bd27634cf1a1a87f20b1e7263bc4067908768fe91cd4.jpg)

![](images/113a0c514d829725c7eb5d35c98527a728cc4d679542f6066258b4f6951baf30.jpg)

C  
![](images/264e65ab39123cc1bc4c4c0cea9277330537e7ff281ad6c260dff80930335907.jpg)

d  
![](images/b5f18b21a7b7f8ad745e3f731e581d0b3d01a90af4bec9976c42f1c59feb0f9d.jpg)  
Figure S14 | ToF-SIMS sputter-depth profiles of the graphite SEI formed with and without TNDB. (a) CH3CO⁻, representing oxygenated organic fragments derived from electrolyte decomposition; (b) LiF2⁻, representing LiFrelated inorganic species; and (c) PO2⁻, associated with LiPF6-derived P-O species. (d) CO3⁻, representing carbonate-containing residues. The profiles are interpreted as relative depth-distribution trends.

![](images/20b5eb6af5425e0051b0e0f40e9d30ac1b6647b8703ba60a580d4643dba9b3a5.jpg)  
Figure S15 | ToF-SIMS sputtering-time profiles of the FeO⁻ secondary-ion fragment for graphite electrodes recovered after formation and 100 cycles.

![](images/fe71e5f056c75296a4e70ef177dca25e3b02ed3f5bcfd6a44bf41bb1296b2f8e.jpg)

![](images/378d556e1eb0d4e71b4a68e4ac1362075b1d975415ea8ac2a7d1e781e16ea664.jpg)

![](images/1fb4ddc1ee74d6d0431da1eb8a44277333abb4873b510db80e9198f0d4210df5.jpg)

![](images/7508aeb69151fb6d60dc4647336c485ca738526d7311c8da1dd397551d96b74f.jpg)  
Figure S16 | Surface morphology of graphite electrodes after cycling. Representative SEM images of graphite electrodes recovered from $\mathsf { L i F e P O 4 }$ ||graphite cells using the (a,b) baseline electrolyte and (c,d) TNDB-containing electrolyte after (a,c) formation and (b,d) 100 cycles at $5 5 ^ { \circ } \complement$

![](images/d06d4b20db07e5ba3ae5a8491034247969f0cbcd7f1054db4992ef4b3b39cb6e.jpg)  
Figure S17 | Details of the large-language-model-assisted data-mining pipeline. Content identification: identify the electrolyte and interphase-related information. Auto-extraction: using the large language model to extract specific content guided by the user prompts. Auto-correction: read the corresponding literature for each electrolyte and additive entity again to check the correction of the extracted information. The deadline for collecting the dataset is before July 2025. And the prompts used in the extraction stage are shown in the Supplementary Figure S18 and Figure S19.

![](images/7c59308d518f1ee26810a4970bbb9521a27735f3e874c8c26fe7bfadd6fd70ba.jpg)  
Figure S18 | Prompt for content identification and auto-extraction.

![](images/346a31088e7f01abe945ec60d5e69c16dea1eb240c994e5c39034600db76f36e.jpg)  
Figure S19 | Prompt for auto-correction.

Table S10 | Edge features and corresponding physical meanings
<table><tr><td>Feature index</td><td>Physical Meaning</td><td>Feature type</td><td>Encoding type</td><td>Data type</td></tr><tr><td>1</td><td>bond_type_single</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>2</td><td>bond_type_double</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>3</td><td>bond_type_triple</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>4</td><td>bond_type_aromatic</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>5</td><td>bond_is_conjugated</td><td>Edge feature</td><td>Numerical</td><td>Int</td></tr><tr><td>6</td><td>bond_is_in_ring</td><td>Edge feature</td><td>Numerical</td><td>Int</td></tr><tr><td>7</td><td>bond_dir_none</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>8</td><td>bond_dir_beginwedge</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>9</td><td>bond_dir_begindash</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>10</td><td>bond_dir_enddownright</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>11</td><td>bond_dir_endupright</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>12</td><td>bond_is_aromatic</td><td>Edge feature</td><td>Numerical</td><td>Int</td></tr><tr><td>13</td><td>stereo_type_stereoz</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>14</td><td>stereo_type_stereoe</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>15</td><td>stereo_type_stereoany</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr><tr><td>16</td><td>stereo_type_stereonone</td><td>Edge feature</td><td>One hot encoding</td><td>String</td></tr></table>

## References

1 Murtagh, F. & Contreras, P. Algorithms for hierarchical clustering: an overview. Wiley interdisciplinary reviews: data mining and knowledge discovery 2, 86-97 (2012).

2 Bajusz, D., Rácz, A. & Héberger, K. Why is Tanimoto index an appropriate choice for fingerprint-based similarity calculations? Journal of Cheminformatics 7, 20 (2015).

3 Likas, A., Vlassis, N. & Verbeek, J. J. The global k-means clustering algorithm. Pattern recognition 36, 451-461 (2003).

4 McInnes, L., Healy, J. & Melville, J. Umap: Uniform manifold approximation and projection for dimension reduction. arXiv preprint arXiv:1802.03426 (2018).

5 Ertl, P. & Schuffenhauer, A. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal of Cheminformatics 1 (2009).

6 Zou, Y. et al. Enhanced interfacial stability of a LiNi0. 9Co0. 05Mn0. 05O2 cathode by a diboron additive. ACS Applied Energy Materials 4, 11051-11061 (2021).

7 Zhang, W. et al. Dynamically constructing robust cathode-electrolyte-interphase on nickel-rich cathode by organic boron additive for high-performance lithium-ion batteries. Chemical Engineering Journal 491, 151946 (2024).

8 Chen, Y. et al. Molecular-level designed single electrolyte additive with multifunctional groups enabling high mechanical properties/fast Li+ kinetics interphase for widetemperature nickel-rich/graphite batteries. Chemical Engineering Journal 500, 157218 (2024).

9 Sun, Z. et al. High-voltage and high-temperature LiCoO2 operation via the electrolyte additive of electron-defect boron compounds. ACS Energy Letters 8, 2478-2487 (2023).

10 Zhou, H. et al. Controlled Anodic Decomposition Pathway of Supramolecular Lithium Borate for Rationally Tuned Interphase Chemistry. Angewandte Chemie 137, e202500425 (2025).

11 Wang, Y. et al. N-heterocyclic carbene as a potent LiNO3-solubilizer, Li+-solvation regulator and solid-electrolyte interphase enhancer for highly durable Lithium metal batteries. Chemical Engineering Journal, 166397 (2025).

12 Peng, S.-H., Kuan, W.-F. & Lue, S. J. Electro-boost in-situ formation of poly (tetrahydrofuran) to enhance interfacial compatibility and charge transfer for solid polymer electrolyte. European Polymer Journal 234, 114016 (2025).

13 Kucuk, A. C. & Abe, T. Borolan-2-yl involving anion acceptors for organic liquid electrolyte-based fluoride shuttle batteries. Journal of Fluorine Chemistry 240, 109672 (2020).

14 Zhang, S. & Angell, C. A Novel Electrolyte Solvent for Rechargeable Lithium and Lithium‐Ion Batteries. Journal of The Electrochemical Society 143, 4047-4053 (1996).

15 Lv, J., Wang, Z., Zhou, Y., Yu, W. & Jia, Q. Application of a cost-effective boron-based electrolyte additive in high-voltage lithium metal battery. Journal of Energy Storage 112, 115596 (2025).

16 Li, L., Lee, H., Li, H., Yang, X. & Huang, X. A pentafluorophenylboron oxalate additive in non-aqueous electrolytes for lithium batteries. Electrochemistry communications 11, 2296-2299 (2009).

17 Sasaki, Y., Handa, M., Kurashima, K., Tonuma, T. & Usami, K. Application of lithium

organoborate with salicylic ligand to lithium battery electrolyte. Journal of The Electrochemical Society 148, A999-A1003 (2001).

18 Sun, X., Lee, H., Yang, X. & McBreen, J. Comparative studies of the electrochemical and thermal stability of two types of composite lithium battery electrolytes using boronbased anion receptors. Journal of The Electrochemical Society 146, 3655-3659 (1999).

19 Kaymaksiz, S. et al. Electrochemical stability of lithium salicylato-borates as electrolyte additives in Li-ion batteries. Journal of Power Sources 239, 659-669 (2013).

20 Chen, Z. & Amine, K. Bifunctional electrolyte additive for lithium-ion batteries. Electrochemistry communications 9, 703-707 (2007).

21 McBreen, J., Lee, H., Yang, X. & Sun, X. New approaches to the design of polymer and liquid electrolytes for lithium batteries. Journal of Power Sources 89, 163-167 (2000).

22 Parida, R. & Lee, J. Y. Boron based podand molecule as an anion receptor additive in Li-ion battery electrolytes: A combined density functional theory and molecular dynamics study. Journal of Molecular Liquids 384, 122236 (2023).

23 Li, X.-Y., Xue, Z.-M., Zhao, J.-F. & Chen, C.-H. A new lithium salt with tetrafluoro-1, 2- benzenediolato and oxalato complexes of boron for lithium battery electrolytes. Journal of Power Sources 235, 274-279 (2013).

24 Zhu, Y. et al. A composite gel polymer electrolyte with high performance based on poly (vinylidene fluoride) and polyborate for lithium ion batteries. Advanced Energy Materials 4, 1300647 (2014).

25 Chen, Y.-Q. et al. An electrolyte additive with boron-nitrogen-oxygen alkyl group enabled stable cycling for high voltage LiNi0. 5Mn1. 5O4 cathode in lithium-ion battery. Journal of Power Sources 477, 228473 (2020).

26 Aravindan, V., Gnanaraj, J., Madhavi, S. & Liu, H. K. Lithium‐ion conducting electrolyte salts for lithium batteries. Chemistry–A European Journal 17, 14326-14346 (2011).

27 Zhu, Y., Xiao, S., Shi, Y., Yang, Y. & Wu, Y. A trilayer poly (vinylidene fluoride)/polyborate/poly (vinylidene fluoride) gel polymer electrolyte with good performance for lithium ion batteries. Journal of Materials Chemistry A 1, 7790-7797 (2013).

28 Tang, Y.-N., Xue, Z.-M., Ding, J. & Chen, C.-H. Two unsymmetrical lithium organoborates with mixed-ligand of croconato and oxalicdiolato or benzenediolato for lithium battery electrolytes. Journal of Power Sources 218, 134-139 (2012).

29 Liu, Y. et al. Impacts of lithium tetrafluoroborate and lithium difluoro (oxalate) borate as additives on the storage life of Li-ion battery at elevated temperature. Ionics 24, 1617- 1628 (2018).

30 Feng, Y. H. et al. Dual‐Anionic Coordination Manipulation Induces Phosphorus and Boron‐Rich Gradient Interphase Towards Stable and Safe Sodium Metal Batteries. Angewandte Chemie 137, e202415644 (2025).

31 Xu, W., Shusterman, A. J., Marzke, R. & Angell, C. A. LiMOB, an unsymmetrical nonaromatic orthoborate salt for nonaqueous solution electrochemical applications. Journal of The Electrochemical Society 151, A632-A638 (2004).

32 Liu, P. et al. Interphase building of organic–inorganic hybrid polymer solid electrolyte with uniform intermolecular Li+ path for stable lithium metal batteries. Small 17, 2102454 (2021).

33 Chen, Z., Liu, J. & Amine, K. Lithium difluoro (oxalato) borate as salt for lithium-ion batteries. Electrochemical and solid-state letters 10, A45-A47 (2007).

34 Lee, H., Ma, Z., Yang, X., Sun, X. & McBreen, J. Synthesis of a series of fluorinated boronate compounds and their use as additives in lithium battery electrolytes. Journal of The Electrochemical Society 151, A1429-A1435 (2004).

35 Sasaki, Y., Sekiya, S., Handa, M. & Usami, K. Chelate complexes with boron as lithium salts for lithium battery electrolytes. Journal of Power Sources 79, 91-96 (1999).

36 Kucuk, A. C., Yamanaka, T. & Abe, T. Fluoride shuttle batteries: on the performance of the BiF3 electrode in organic liquid electrolytes containing a mixture of lithium bis (oxalato) borate and triphenylboroxin. Solid State Ionics 357, 115499 (2020).

37 Xue, Z.-M., Zhao, J.-F., Ding, J. & Chen, C.-H. LBDOB, a new lithium salt with benzenediolato and oxalato complexes of boron for lithium battery electrolytes. Journal of Power Sources 195, 853-856 (2010).

38 Kwon, H. et al. Covariance of interphasic properties and fast chargeability of energydense lithium metal batteries. Nature Energy 10, 1132-1145 (2025).

39 Shah, F. U., Gnezdilov, O. I. & Filippov, A. Ion dynamics in halogen-free phosphonium bis (salicylato) borate ionic liquid electrolytes for lithium-ion batteries. Physical Chemistry Chemical Physics 19, 16721-16730 (2017).

40 Long, J. et al. Efficient boron-based electrolytes constructed by anionic and interfacial co-regulation for rechargeable magnesium batteries. Chemical Engineering Journal 461, 141901 (2023).

41 Nie, M., Xia, J. & Dahn, J. Development of pyridine-boron trifluoride electrolyte additives for lithium-ion batteries. Journal of The Electrochemical Society 162, A1186- A1195 (2015).

42 Park, N. R. et al. Understanding the role of lithium borate as the surface coating on high voltage single crystal LiNi0. 5Mn1. 5O4. Advanced Functional Materials 34, 2312091 (2024).

43 Xie, Z. et al. Designing High-Temperature Stable Electrolytes: Insights from the Degradation Mechanisms of Boron-Containing Additives. Journal of the American Chemical Society 147, 23931-23945 (2025).

44 Ma, L. et al. A systematic study of some promising electrolyte additives in Li [Ni1/3Mn1/3Co1/3] O2/graphite, Li [Ni0. 5Mn0. 3Co0. 2]/graphite and Li [Ni0. 6Mn0. 2Co0. 2]/graphite pouch cells. Journal of Power Sources 299, 130-138 (2015).

45 Öksüzoğlu, F. et al. The Impact of boron compounds on the structure and ionic conductivity of LATP solid electrolytes. Materials 17, 3846 (2024).

46 Liu, S. et al. Dendrite-free lithium deposition enabled by interfacial regulation via dipoledipole interaction in anode-free lithium metal batteries. Energy Storage Materials 62, 102959 (2023).

47 Wang, D. et al. Enhanced ionic conductivity of Li3. 5Si0. 5P0. 5O4 with addition of lithium borate. Solid State Ionics 283, 109-114 (2015).

48 Pershina, S., Vovkotrub, E. & Antonov, B. Effects of B2O3 on crystallization kinetics, microstructure and properties of Li1. 5Al0. 5Ge1. 5 (PO4) 3-based glass-ceramics. Solid State Ionics 383, 115990 (2022).

49 Aslam, M. & Kong, X. Y. A lithium ion conductor in Li4SiO4-Li3PO4-LiBO2 ternary

system. Solid State Ionics 293, 72-76 (2016).

50 Li, J. et al. Understanding interfacial properties between Li-rich layered oxide and electrolyte containing triethyl borate. The Journal of Physical Chemistry C 120, 26899- 26907 (2016).

51 Cai, Q. et al. Synergistic effects of film-forming and film-modifying additives for enhanced all-climate performance of graphite/NMC622 pouch cells. Chemical Engineering Journal 505, 159156 (2025).

52 Liu, Z., Patra, A. & Matsumi, N. Boron-Containing Ternary Electrolyte for Excellent Li-Ion Transference and Stabilization of LiNMC-Based Cells. ACS Applied Energy Materials 8, 3360-3368 (2025).

53 Zhang, Y. et al. BF3-based electrolyte additives promote electrochemical reactions to boost the energy density of Li/CFx primary batteries. Electrochimica Acta 470, 143311 (2023).

54 Yan, J. et al. Sustainable release of borate-and nitrate-electrolyte additives via metalorganic frameworks nanocapsules for stable lithium metal batteries. Chemical Engineering Journal 494, 153104 (2024).

55 Cai, K. et al. Tow-way regulation strategy for in-situ electropolymerization additives coordinated by double bonds and boric acid groups in lithium secondary batteries. Chemical Engineering Journal 500, 156790 (2024).

56 Duan, P. H. et al. Dynamic anion enables self‐healing single‐ion conductor polymer electrolyte for lithium‐metal batteries. Advanced Functional Materials 34, 2402065 (2024).

57 Kumagae, K. et al. Improvement of cycling performance of FeF3-based lithium-ion battery by boron-based additives. Journal of The Electrochemical Society 163, A1633- A1636 (2016).

58 Song, X. et al. Iodine Boosted Fluoro‐Organic Borate Electrolytes Enabling Fluent Ion‐Conductive Solid Electrolyte Interphase for High‐Performance Magnesium Metal Batteries. Angewandte Chemie 137, e202417450 (2025).

59 Lohani, H. et al. Fluorine Rich Borate Salt Anion Based Electrolyte for High Voltage Sodium Metal Battery Development. Small 20, 2311157 (2024).

60 Ma, Y. et al. A new anion receptor for improving the interface between lithium-and manganese-rich layered oxide cathode and the electrolyte. Chemistry of Materials 29, 2141-2149 (2017).

61 Hou, Z., Zhou, R., Liu, K., Zhu, J. & Zhang, B. A CaI2‐Based Electrolyte Enabled by Borate Ester Anion Receptors for Reversible Ca− Organic and Ca− Se Batteries. Angewandte Chemie International Edition 64, e202413416 (2025).

62 Nagasubramanian, G. & Sanchez, B. A new chemical approach to improving discharge capacity of Li/(CFx) n cells. Journal of Power Sources 165, 630-634 (2007).

63 Li, L. et al. New electrolytes for lithium ion batteries using LiF salt and boron based anion receptors. Journal of Power Sources 184, 517-521 (2008).

64 Li, T. et al. In situ polymerization of 1, 3-dioxolane and formation of fluorine/boron-rich interfaces enabled by film-forming additives for long-life lithium metal batteries. Chemical Science 15, 12108-12117 (2024).

65 Meng, Y., Chen, G., Shi, L., Liu, H. & Zhang, D. Operando fourier transform infrared

investigation of cathode electrolyte interphase dynamic reversible evolution on Li1. 2Ni0. 2Mn0. 6O2. ACS Applied Materials & Interfaces 11, 45108-45117 (2019).

66 Zhang, Z. et al. Enhancing the electrochemical performance of a high-voltage LiCoO2 cathode with a bifunctional electrolyte additive. ACS Applied Energy Materials 4, 12954-12964 (2021).

67 Horino, T. et al. High voltage stability of interfacial reaction at the LiMn2O4 thin-film electrodes/liquid electrolytes with boroxine compounds. Journal of The Electrochemical Society 157, A677-A681 (2010).

68 Shim, J., Lee, J. S., Lee, J. H., Kim, H. J. & Lee, J.-C. Gel polymer electrolytes containing anion-trapping boron moieties for lithium-ion battery applications. ACS Applied Materials & Interfaces 8, 27740-27752 (2016).

69 Yang, Z., Ye, Y., Meng, N. & Lian, F. Adaptive 3D Cross‐Linked Single‐Ion Conducting Polymer Electrolytes Enable Powerful Interface for Solid State Batteries. Angewandte Chemie 137, e202505232 (2025).

70 Forero-Saboya, J., Bodin, C. & Ponrouch, A. A boron-based electrolyte additive for calcium electrodeposition. Electrochemistry communications 124, 106936 (2021).

71 Hong, D. G., Baik, J.-H., Kim, S. & Lee, J.-C. Solid polymer electrolytes based on polysiloxane with anion-trapping boron moieties for all-solid-state lithium metal batteries. Polymer 240, 124517 (2022).

72 Yang, X. et al. Stabilizing lithium manganese oxide/organic carbonate electrolyte interface with a simple boron-containing additive. Electrochimica Acta 227, 24-32 (2017).

73 Shin, H. et al. Synthesis and characteristics of acrylol borate as new acrylic gelator for lithium secondary battery. Macromolecular Research 16, 134-138 (2008).

74 Freiberg, A. et al. Anodic decomposition of trimethylboroxine as additive for high voltage Li-ion batteries. Journal of The Electrochemical Society 161, A2255-A2261 (2014).

75 Luo, S. et al. Construction of stable two-sided interface via the addition of phenylboric acid in Lithium-ion batteries. Electrochimica Acta 498, 144684 (2024).

76 Xi, K. et al. A novel strategy to improve the electrochemical properties of in-situ polymerized 1, 3-dioxolane electrolyte in lithium metal batteries. Journal of Colloid and Interface Science 679, 1277-1287 (2025).

77 Zhang, C. et al. Small Molecule Accurate Recognition Technology (SMART) to Enhance Natural Products Research. Scientific Reports 7 (2017).