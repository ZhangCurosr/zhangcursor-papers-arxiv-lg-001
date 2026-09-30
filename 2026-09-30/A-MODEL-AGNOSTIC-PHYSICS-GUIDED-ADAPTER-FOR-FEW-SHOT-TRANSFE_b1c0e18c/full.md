# A MODEL-AGNOSTIC PHYSICS-GUIDED ADAPTER FOR FEW-SHOT TRANSFER OF COASTAL FLOOD PRE-DICTION MODELS TO UNSEEN REGIONS

Bilal Hassan, Areg Karapetyan and Samer Madanat

Division of Engineering

New York University Abu Dhabi

Abu Dhabi, UAE

{bilal.hassan, areg.karapetyan, samer.madanat}@nyu.edu

## ABSTRACT

Deep learning (DL) surrogates can produce high-resolution coastal flood maps orders of magnitude faster than physics-based hydrodynamic simulators, yet transferring them to new coastal regions remains costly, since generating target-region data for fine-tuning typically requires numerous time-consuming simulations. To tackle this bottleneck, we introduce the Physics Adapter (PA), a compact, architecture-agnostic adaptation interface that enables efficient few-shot transfer of flood prediction models across diverse coastal regions. PA predicts peak water level through a differentiable wet/dry response that compares terrain elevation against a learned water level, and blends this physics-structured prediction with a data-driven branch through a learned gate. Unlike physics-informed formulations, PA imposes no PDE-residual or conservation losses and instead exploits elevation as an architectural inductive bias, adding a negligible number of trainable parameters. We integrate PA into 12 heterogeneous models, spanning graph, convolutional, Transformer, state-space, depth-foundation and diffusion models, and evaluate them on two coastal regions with markedly distinct geometries, topographies, and shoreline protection configurations. The performance of PA is benchmarked against a no-physics baseline, full fine-tuning, and standard parameter-efficient fine-tuning (PEFT) methods, considering both within-region generalization to unseen sea level rise (SLR) values and between-region transfer. In low-shot regime (K=3), and averaged over all backbones and transfer settings, adding PA reduces root mean square error (RMSE) by 11.5% when only the output head is adapted on a frozen backbone, by 15.4% when combined with PEFT methods, and by 22.9% under full fine-tuning, compared to matched configurations without PA. Taken together, the findings of this work offer practitioners a concrete recipe for extending DL-based coastal flood predictors to new, data-scarce regions, thereby advancing scalable AI support for coastal adaptation planning.

## 1 INTRODUCTION

Climate adaptation-aware coastal protection planning requires repeated prediction of how peak water level (PWL) responds to candidate shoreline protection configurations under different sea level rise (SLR) and forcing conditions. Physics-based high-fidelity simulators, such as Delft3D (Lesser et al., 2004), can accurately simulate nearshore hydrodynamics, providing fine-grained estimates of depth, duration, and velocity of floods. However, due to prohibitively high computational cost, their direct adoption in large-scale coastal protection investigations, where each combination of protection configuration and SLR value requires a separate simulation, remains impractical (Jia et al., 2019). Prior studies (Hassan et al., 2026; Karapetyan et al., 2026; Bian et al., 2025) have demonstrated that learned surrogate models can dramatically reduce this computational burden by approximating the simulator’s output, thereby replacing repeated hydrodynamic simulations with efficient inference.

Existing surrogate models have been typically trained and evaluated for one coastline or forcing condition, and their accuracy can degrade sharply when applied to a different coastal region or forcing conditions outside the training range (Sec. 5; see also Zhao et al., 2026). This hinders their practical deployment, since generating sufficient training data for every new setting entails additional hydrodynamic simulations and substantial compute. Moreover, how well existing surrogates transfer across coastlines and SLR conditions, and how many target simulations are necessary to attain satisfactory performance, remains largely unexamined.

In this paper, we investigate whether pretrained coastal flood prediction models can be adapted to new regions and SLR conditions from only a few target simulations, and whether this can be achieved in an architecture-agnostic manner. The problem is challenging for two reasons. First, domain shift arises in different forms: geographic transfer changes terrain, protection geometry, landcover structure, and flood response, whereas SLR transfer affects the forcing within the same region. An effective adaptation mechanism must handle both from only a handful of labeled examples. Second, achieving this in an architecture-agnostic manner is difficult, since the surrogate models can span various learned representations, from graph and mesh networks to vision, state-space and diffusion models. As Lee et al. (2022) illustrate, with limited target data the choice of which parameters to update matters, and the best choice depends on the type of shift. Parameter-efficient fine-tuning (PEFT) methods, such as LoRA (Hu et al., 2021), BitFit (Zaken et al., 2022), and IA<sup>3</sup> (Liu et al., 2022a), restrict updates to selected parameters or inserted modules. These methods specify where and how a source model changes, but do not encode any flood-specific relation between PWL and the physical variables that remain observable in the target domain. In this context, we treat parameter efficiency and physical structure as distinct, potentially complementary components of adaptation.

A recent article by Daramola et al. (2026) argues that transferable coastal flood models require inductive biases reflecting the underlying hydrodynamic processes, rather than solely relying on statistical similarity between source and target regions. In line with this view, we anchor our approach on terrain elevation, which is readily available for coastal regions and is directly linked to inundation extent and dynamics. More concretely, we introduce a lightweight module, termed the Physics Adapter (PA), that combines elevation with the features of a pretrained surrogate model (hereafter, also referred to as backbone) to predict PWL. A physics-guided branch predicts PWL through a differentiable wet/dry response that compares terrain elevation against a learned water level, while a parallel data-driven branch captures effects that terrain alone does not explain. A learned gate combine the two predictions. As PA requires only backbone features and elevation, it can be attached to any architecture, with only the integration interface tailored to each backbone. Unlike physics-informed formulations such as PINNs (Raissi et al., 2019), PA does not embed the shallow-water or Navier Stokes equations, minimize PDE residuals, or enforce mass or momentum conservation. Terrain elevation thus serves PA as an architectural inductive bias rather than a governing-equation constraint.

We instantiate PA across 12 diverse backbones and evaluate it on two coastal regions with markedly different geometries, topographies, and shoreline protection configurations, namely the coastal city of Abu Dhabi (AD) and the San Francisco (SF) Bay Area. Our main contributions are as follows:

• A lightweight, architecture-agnostic physics-guided adapter: PA injects terrain-based physical structure into pretrained flood surrogates without PDE-residual or conservation losses, while adding a negligible number of trainable parameters.

• Extensive Evaluation: We evaluate twelve backbones spanning graph, dense-vision, state-space, foundation, and diffusion models under bidirectional cross-region transfer (SF↔AD) and withinregion SLR transfer, comparing ten adaptation regimes (full fine-tuning, head-only adaptation, and three PEFT methods, each with and without PA) under a matched protocol.

• Empirical evidence that physical structure complements parameter efficiency: Averaged across backbones and transfer settings, every regime with PA outperforms every regime without it once a single target simulation is available, and adapting PA alone surpasses full fine-tuning without PA. At the level of individual backbones, a PA regime remains the best parameter-efficient choice for nine of the twelve models.

## 2 RELATED WORK

Learned surrogates for flood prediction. For coastal domains, DL-based surrogates have been developed for predicting extreme storm surge under future climate scenarios (Longo et al., 2026; Rice et al., 2025; Gharehtoragh & Johnson, 2024), spatiotemporal flood dynamics (Bian et al.,

2025), and tidal and riverine shallow-water dynamics (Rivera-Casillas et al., 2025). The CASPIAN framework (Karapetyan et al., 2026) and its follow-up (Hassan et al., 2026) predict PWL under shoreline protection for SF and AD across multiple SLR conditions, and their publicly released dataset serves as the data source for this work. Most of these surrogates, however, are trained and evaluated within a single region. Only a few studies have examined how coastal surrogates transfer to regions unseen during training. Zhao et al. (2026), for instance, report zero-shot generalization of a storm-surge model to unseen bays along the same coastline. Similar efforts for urban and riverine flooding adapt neural-operator surrogates to new catchments or forcing conditions via transfer learning (Xu et al., 2025) or domain adaptation (Taghizadeh et al., 2025a). These studies, however, examine transfer within a single surrogate design. On the other hand, the present work investigates whether one physics-guided adaptation strategy, shared across multiple backbone model families, can recover target-domain performance from only a few labeled target scenarios, including across geographically and hydrodynamically distinct coastlines.

Physics-guided learning. Physical constraints can enter a learned surrogate through different mechanisms. PINNs impose governing equations through residual-based objectives (Raissi et al., 2019), and physics-informed neural operators combine operator learning with PDE constraints (Li et al., 2024). Flood-specific models can encode more domain-specific hydraulic structure. HydroGraphNet (Taghizadeh et al., 2025b) includes mass conservation in its training objective, whereas hydraulics-informed message passing derives graph interactions from shallow-water structure (Kazadi et al., 2024). Beyond flooding, physical structure can also be built into the architecture itself, as in ClimODE (Verma et al., 2024), which encodes advection within continuous-time weather dynamics. Closest to our setting, GeoAda-PINN (Zhu et al., 2026) freezes a PINN backbone and updates compact geometry-conditioned adapters to handle geometric changes. In contrast, the proposed adapter uses a lighter architectural inductive bias based on terrain elevation and a differentiable wet/dry response, without any governing-equation loss or conservation guarantee. Moreover, PA employs a consistent formulation across heterogeneous backbone families, whereas GeoAda PINN is tied to a single PINN architecture.

Adaptation under distribution shift and PEFT. When labeled target data are scarce, a key question is which source parameters should be updated. Surgical fine-tuning shows that the effective subset can depend on the type of distribution shift (Lee et al., 2022). Unsupervised test-time methods instead adapt from unlabeled target batches during inference (Wang et al., 2020). This differs from the proposed supervised few-shot setting, where labels are available for the target support scenarios. PEFT controls the optimization subspace through mechanisms such as bottleneck adapters (Houlsby et al., 2019), low-rank weight updates (Hu et al., 2021), bias-only tuning (Zaken et al., 2022), and activation scaling (Liu et al., 2022a). Architecture-aware PEFT has also been studied for state-space models (Yoshimura et al., 2025), and F-Adapter extends this line to large neural-operator models for scientific machine learning (Zhang et al., 2026). These approaches alter the parameterization of adaptation, whereas the PA instead adds elevation-conditioned structure to the prediction interface. We therefore evaluate PA and PEFT both separately and in combination across heterogeneous neural surrogates.

## 3 METHOD

This section details the proposed approach and its use for target adaptation. We separate how each backbone represents a flood scenario from how that representation is converted into elevationconditioned PWL, so that this conversion can be shared across all twelve backbones. Sec. 3.1 defines the prediction problem and this separation, Sec. 3.2 describes the adapter, Sec. 3.3 explains how it is attached to each backbone, and Sec. 3.4 defines source training and the adaptation regimes.

## 3.1 PROBLEM FORMULATION

A prediction domain is a pair $d = ( r , \lambda )$ of a geographic region r and a forcing condition λ, which we vary through SLR. For region $r ,$ let $\Omega _ { r }$ denote the discrete prediction sites and $\Omega _ { r } ^ { v } \subseteq \Omega _ { \ i }$ the valid sites. A flood scenario has input x and target $\mathbf { y } = \{ y _ { i } \} _ { i \in \Omega _ { r } ^ { v } }$ , where $y _ { i } \geq 0$ is the simulated PWL. Raster backbones take a four-channel $1 0 2 4 \times 1 0 2 4$ tensor $\mathbf { x } _ { \mathrm { g r i d } } = [ \mathbf { c } ^ { ( \mathbf { s } ) } , \mathbf { z } , \boldsymbol { \ell } , \mathbf { v } ]$ containing scenario-dependent protection status, the DEM, land cover, and a binary validity mask $\mathsf { \bar { v } } _ { i } \in \{ 0 , 1 \}$

![](images/405c5eea64a520fe3ac8c74d7d028ee1c61e4f99eb3f9bd8a7cf66a401b3d884.jpg)  
Figure 1: Backbone-agnostic Physics Adapter. (a) Each backbone exposes features aligned with the PWL prediction sites through its own interface, and the shared adapter produces the prediction. (b) Inside the adapter, raw terrain elevation enters the physics-guided branch through a soft wet/dry response, a parallel data-driven branch captures effects that terrain alone does not explain, and a learned site-wise gate combines the two. Feature extraction is architecture-specific, while terrain conditioning, the two branches, gated fusion, and target adaptation are shared.

Graph and mesh backbones encode the same variables as node features on the valid sites, so the mask is implicit (Appendix A). In every case, the PA receives the raw DEM value $z _ { i }$ in physical units, rather than a normalized or embedded version recovered from backbone features.

Each of the twelve backbones m $\in \mathcal { M }$ (Sec. 4.2) defines a native representation, a site-alignment interface $\mathcal { P } _ { m } : \mathcal { H } _ { m }  \mathbb { R } ^ { | \Omega _ { r } | \times C _ { m } }$ , and the shared adapter computation

$$
\begin{array} { r l } & { \mathbf { h } ^ { ( m ) } = f _ { \boldsymbol { \theta } _ { m } } ^ { ( m ) } ( \mathbf { x } ) , } \\ & { \widetilde { \mathbf { H } } ^ { ( m ) } = \mathcal { P } _ { m } \big ( \mathbf { h } ^ { ( m ) } \big ) , \qquad \widetilde { \mathbf { h } } _ { i } ^ { ( m ) } \in \mathbb { R } ^ { C _ { m } } , } \\ & { \widehat { \mathbf { y } } ^ { ( m ) } = A _ { \phi _ { m } } \big ( \widetilde { \mathbf { H } } ^ { ( m ) } , \mathbf { z } , \mathbf { v } \big ) . } \end{array}\tag{1}
$$

The interface $\mathcal { P } _ { m }$ (attachment point, tensor layout, feature dimension, decoder, interpolation, and projection) differs across backbones. Its parameters are grouped with the backbone parameters $\theta _ { m } ,$ except in the diffusion case, where the post-decoder stem belongs to the adapter (Sec. 3.3). The adapter parameters are $\phi _ { m } .$ , so the full PA model has parameters $\Theta ^ { ( m ) } = \theta _ { m } \cup \phi _ { m }$ . Here $\theta _ { m }$ determines how a scenario is represented, and $\phi _ { m }$ determines how that representation is converted into elevation-conditioned PWL. Figure 1 summarizes the formulation.

## 3.2 PHYSICS-GUIDED ADAPTER

Adapter heads. The site-aligned features are first normalized by a BatchNorm layer, and four lightweight representation-compatible functions then produce a threshold correction $( \beta _ { i } )$ , a waterlevel correction $( \psi _ { i } )$ , a data-branch correction $( a _ { i } )$ , and a gate logit $( \kappa _ { i } )$

$$
\begin{array} { l } { { \mathbf { h } } _ { i } = { \mathrm { B N } } _ { A } ^ { ( m ) } ( \widetilde { \mathbf { h } } _ { i } ^ { ( m ) } ) , } \\ { \beta _ { i } = f _ { \beta } ^ { ( m ) } ( { \mathbf { h } } _ { i } ) , \quad \psi _ { i } = f _ { \psi } ^ { ( m ) } ( { \mathbf { h } } _ { i } ) , \quad a _ { i } = f _ { a } ^ { ( m ) } ( { \mathbf { h } } _ { i } ) , \quad \kappa _ { i } = f _ { g } ^ { ( m ) } ( { \mathbf { h } } _ { i } ) . } \end{array}\tag{2}
$$

For dense feature maps, the heads are pointwise $1 \times 1$ convolutions, and for node-aligned backbones they are node-wise two-layer multilayer perceptrons (MLPs). The running statistics of $\mathrm { B N } _ { A } ^ { ( m ) }$ are part of the adapter state and are recalibrated during target adaptation (Sec. 3.4). Backbone-specific settings, including affine BatchNorm parameters and head dropout are listed in Appendix B.

Elevation-conditioned inundation. A learned scalar $\eta \in$ R defines a reference water level, and a second scalar ϑ parameterizes a positive transition temperature. The local inundation response is

$$
p _ { i } ^ { \mathrm { f l o o d } } = \sigma \biggl ( \frac { \eta + \beta _ { i } - z _ { i } } { \tau } \biggr ) , \qquad \tau = \mathrm { m a x } \bigl \{ \exp ( \vartheta ) , 1 0 ^ { - 3 } \bigr \} ,\tag{3}
$$

where $z _ { i }$ is the raw DEM elevation and $\sigma$ is the logistic sigmoid. The correction $\beta _ { i }$ shifts the effective inundation threshold based on the backbone features, and τ controls how sharp the wet/dry transition is. Both η and ϑ are learned, with region-specific initialization given in Appendix B.

Physics and data branches. The two prediction branches are

$$
\widehat { y } _ { i } ^ { \mathrm { p h y s } } = p _ { i } ^ { \mathrm { f l o o d } } \left[ \eta + \psi _ { i } \right] _ { + } , \qquad \widehat { y } _ { i } ^ { \mathrm { d a t a } } = \left[ b _ { i } ^ { ( m ) } + a _ { i } \right] _ { + } ,\tag{4}
$$

where $[ \cdot ] _ { + } = \operatorname* { m a x } ( \cdot , 0 )$ . The correction $\psi _ { i }$ adjusts the water level separately from the threshold shift $\beta _ { i }$ in Eq. (3), so the physics-guided branch ties PWL to absolute terrain elevation even when the backbone does not preserve DEM units. For eleven backbones $b _ { i } ^ { ( m ) } = 0$ , and the data branch is a direct non-negative prediction. For the diffusion backbone, $b _ { i } ^ { ( m ) }$ is the generated base $\mathrm { P W L }$ , so the data branch predicts a residual around the generated map (Sec. 3.3).

Gated fusion and masking. A site-wise gate mixes the branches, and invalid raster cells are removed from the output,

$$
g _ { i } = \sigma ( \kappa _ { i } ) , \qquad \widetilde { y } _ { i } = g _ { i } \widehat { y } _ { i } ^ { \mathrm { p h y s } } + \left( 1 - g _ { i } \right) \widehat { y } _ { i } ^ { \mathrm { d a t a } } , \qquad \widehat { y } _ { i } = v _ { i } \widetilde { y } _ { i } .\tag{5}
$$

For graph and mesh backbones, nodes already coincide with prediction sites, so masking is implicit. The gate is initialized toward the physics-guided branch with a logit of +3, so that $g _ { i } = \sigma ( 3 )$ ≈ 0.95 at the start of training while the gate gradient remains large enough for the model to shift weight toward the data branch where needed. This value is a fixed design choice rather than a tuned hyperparameter (Appendix B).

Eqs. (3)–(5) thus act as an elevation-conditioned architectural inductive bias, not a hydrodynamic solver, and add no PDE-residual or conservation constraint. In the same sense, the PA is not a new backbone, since $\phi _ { m }$ only converts an exposed representation into PWL. It is also distinct from parameter-efficient fine-tuning (PEFT) methods such as LoRA, BitFit and $\mathrm { I A ^ { 3 } }$ , which modify selected weights or activations inside an existing model. The two can therefore be used separately or together, and the adaptation regimes in Sec. 3.4 compare both options.

## 3.3 BACKBONE-COMPATIBLE INSTANTIATION

The interfaces $\mathcal { P } _ { m }$ align features with the prediction sites but do not make the architectures identical, so tensor shape, insertion depth, feature dimension, and PEFT target modules differ across backbones. Node-aligned backbones apply the adapter node-wise, dense backbones use pointwise heads on grid-aligned decoder features after a projection where needed, and the diffusion backbone passes its decoded base PWL map through an adapter-owned stem. In all cases the raw DEM bypasses the backbone and enters Eq. (3) directly, so the PA is model-agnostic in its formulation rather than in its placement inside each network (Appendix B, Table 2).

## 3.4 SOURCE TRAINING AND TARGET ADAPTATION

Objective. Deterministic backbones use masked mean squared error over valid sites. PA models add a penalty that discourages the gate from collapsing onto the data-driven branch,

$$
\begin{array} { r l } { \mathcal { L } _ { \mathrm { p r e d } } = \displaystyle \frac { \sum _ { n \in \mathcal { B } } \sum _ { i \in \Omega _ { r } } v _ { n , i } \left( \widehat { y } _ { n , i } - y _ { n , i } \right) ^ { 2 } } { \operatorname* { m a x } \left( 1 , \ \sum _ { n \in \mathcal { B } } \sum _ { i \in \Omega _ { r } } v _ { n , i } \right) } , } & { } \\ { \mathcal { L } = \mathcal { L } _ { \mathrm { p r e d } } + \mathbb { I } _ { \mathrm { P A } } \displaystyle \frac { \lambda _ { g } } { | \Omega _ { A } | } \sum _ { i \in \Omega _ { A } } ( 1 - g _ { i } ) , } & { \quad \lambda _ { g } = 0 . 1 , } \end{array}\tag{6}
$$

where B is a mini-batch, $\mathbb { I } _ { \mathrm { P A } } = 1$ only for PA models, and $\Omega _ { A }$ is the set of sites on which the gate is defined (Appendix B). For node-based models, $\mathcal { L } _ { \mathrm { p r e d } }$ reduces to standard MSE over valid nodes. The diffusion backbone keeps its native diffusion objective alongside the map-level objective, with the decoded base PWL map detached from the diffusion computation (Appendix B).

Source training. For each backbone, the PA source model is trained jointly on the source domain,

$$
\bigl ( \theta _ { m , s } ^ { \star } , \phi _ { m , s } ^ { \star } \bigr ) = \arg \operatorname* { m i n } _ { \theta _ { m } , \phi _ { m } } \ \mathbb { E } _ { ( \mathbf { x } , \mathbf { y } ) \sim \mathcal { D } _ { \mathrm { t r a i n } } ^ { s } } \bigl [ \mathcal { L } ( \mathbf { x } , \mathbf { y } ; \theta _ { m } , \phi _ { m } ) \bigr ] ,\tag{7}
$$

so PA-only adaptation starts from an adapter trained together with its source representation rather than a newly initialized module. We also train a raw source model, in which the PA is replaced by a direct prediction head with parameters $\chi _ { m } ,$ trained on $\mathcal { L } _ { \mathrm { p r e d } }$ alone, which serves as the no-PA control (Appendix B). We write $\Theta _ { s } ^ { ( m ) \star }$ for the selected source parameters of either model.

Target adaptation. Let ${ \cal S } _ { K } ^ { t } = \{ ( { \bf x } _ { j } ^ { t } , { \bf y } _ { j } ^ { t } ) \} _ { j = 1 } ^ { K }$ be the labeled target support set, disjoint from the held-out target test set. An adaptation regime a specifies a trainable subset $\Theta _ { a } ^ { ( m ) }$ and is optimized from the corresponding source checkpoint,

$$
\widehat { \Theta } _ { a , K } ^ { ( m ) } = \arg \operatorname* { m i n } _ { \Theta _ { a } ^ { ( m ) } } \mathcal { L } _ { t } \big ( S _ { K } ^ { t } ; \Theta _ { a } ^ { ( m ) } , \Theta _ { \neg a } ^ { ( m ) , \mathrm { f r o z e n } } \big ) ,\tag{8}
$$

where $\mathcal { L } _ { t }$ is the objective of Eq. (6) evaluated on $S _ { K } ^ { t }$ . For PEFT regimes, $\Theta _ { a } ^ { ( m ) }$ also includes the injected PEFT parameters $\xi _ { m , a }$ The ten regimes and their trainable sets are defined in Sec. 4.2. Frozen components are frozen in both parameters and internal state, and for every PA regime with $K > 0$ , adaptation starts with a gradient-free recalibration of the adapter BatchNorm on the support inputs (Appendix C). At $K = 0$ , no recalibration or gradient update is performed, so $\widehat { \Theta } _ { a , 0 } ^ { ( m ) } = \hat { \Theta } _ { s } ^ { ( m ) \star }$ and zero-shot results directly evaluate the source checkpoint.

We consider two shifts. Geographic transfer changes the region $( r _ { s } \neq r _ { t } )$ , whereas SLR transfer keeps the region fixed and changes only the forcing $( r _ { s } = r _ { t } , \lambda _ { s } \neq \lambda _ { t } )$ . Transfer directions, support sizes, and the adaptation budget are given in Sec. 4.3.

## 4 EXPERIMENTAL SETUP

## 4.1 DATASETS AND REPRESENTATIONS

We use the publicly released coastal-flood simulations of the CASPIAN studies for AD and SF (Karapetyan et al., 2026; Hassan et al., 2026), produced with Delft3D under defined SLR, tidal forcing, and binary shoreline-protection configurations (Appendix A). Each protection scenario protects a subset of $\dot { N _ { r } }$ operational landscape units (OLUs), with $N _ { \mathrm { S F } } = 3 0$ and $N _ { \mathrm { A D } } = 1 7$ . The regional datasets contain 285 SF scenarios at 1.0 m SLR and 142 AD scenarios at 0.5 m SLR, and two SF SLR-transfer targets at 0.5 m and 1.5 m contain 32 scenarios each. Each retained coastal location carries a scenario-dependent protection status derived from simulated single-OLU responses, together with elevation from Copernicus DEM GLO-30 (European Space Agency, 2022) and land cover from ESA WorldCover (Zanaga et al., 2022), which we sample at every location since the original datasets do not include them. Dense backbones use the $1 0 2 4 \times 1 0 2 4$ tensor of Sec. 3.1, and graph backbones use a graph over the same locations whose edges follow the hydrodynamic mesh. Both are encodings of the same samples rather than separate datasets (Appendix A, Figure 4).

## 4.2 BENCHMARK MODELS AND COMPARISON METHODS

The twelve backbones cover clearly different model families. These are graph and mesh models (GCN (Kipf & Welling, 2017), GAT (Velickoviˇ c et al., 2018), and MeshGraphNet (MGN) (Pfaff´ et al., 2021)), scientific attention over physical sites (Transolver++ (Luo et al., 2025)), dense vision models (CASPIAN (Karapetyan et al., 2026), ConvNeXt V2 (Woo et al., 2023), MaxViT (Tu et al., 2022), and Swin Transformer V2 (Liu et al., 2022b)), a visual state-space model (VM-UNet with a VMamba encoder–decoder (Ruan et al., 2024; Liu et al., 2024)), pretrained depth-foundation models (Depth Anything V2 (Yang et al., 2024) and Depth Pro (Bochkovskiy et al., 2025)), and a conditional diffusion model (ControlNet (Zhang et al., 2023)). The depth-foundation models are adapted to PWL through task-specific input and feature interfaces, not by reinterpreting their depth outputs. Table 2 in Appendix B lists the representation each backbone passes to the PA.

Each backbone has two matched source models. The PA source model trains the backbone and PA jointly (Eq. (7)), and the raw source model replaces the PA with a direct prediction head $\chi _ { m }$ . We compare ten target-adaptation regimes per backbone. Full fine-tuning is run with the PA (FT+PA) and without it (FT). Partial fine-tuning without PA (NPA) trains only $\chi _ { m } .$ , and PA-only adaptation (PA) trains only $\phi _ { m }$ . Three PEFT methods, namely LoRA (Hu et al., 2021) with rank $r = 8 ,$ BitFit (Zaken et al., 2022), and $\mathrm { I A ^ { 3 } }$ (Liu et al., 2022a), train only their injected parameters $\xi _ { m , a } ,$ both on their own (PEFT) and combined with the PA (PEFT+PA). Table 3 in Appendix C gives the trainable and frozen parameters of each regime, along with the PEFT formulations and insertion sites. All regimes share the same scenario manifests, support sets, test sets, seeds, and metric code, while batch sizes, PEFT insertion sites, and parameter counts remain architecture-specific.4.3 SPLITS AND TRANSFER PROTOCOLS

All backbones and seeds share one fixed scenario split, stratified by protection level into approximately $6 0 / 2 0 / 2 0$ train, validation, and test sets, and model selection uses the validation split only. Geographic transfer is evaluated in both directions between SF and AD with $K \in \{ 0 , 1 , \dot { 3 } , 5 , 1 0 \dot { \} }$ and SLR transfer from $\mathrm { S F _ { 1 . 0 } }$ to $\mathrm { S F _ { 0 . 5 } }$ and $\mathrm { S F _ { 1 . 5 } }$ with $K \in \{ 0 , 1 , 3 , 5 , 1 0 \}$ . For each $K > 0$ eight support draws are adapted independently from the source checkpoint with a fixed budget of 50 support passes and no target validation, and all are evaluated on the same held-out target test set (Appendix D, Table 4).

## 4.4 HYPERPARAMETER OPTIMIZATION , TRAINING AND EVALUATION METRICS

For the details on hyperparameter optimization and training, we refer the reader to Appendix E.

We report the mean absolute error (MAE), RMSE, the coefficient of determination $( R ^ { 2 } )$ , the drypoint accuracy $\scriptstyle ( \operatorname { A c c } _ { 0 } )$ ), the relative total absolute error (RTAE), and the error exceedance rates $\delta _ { 0 . 5 }$ and $\delta _ { 0 . 1 }$ . All metrics are computed at the same retained locations for every model, and their definitions and aggregation are given in Appendix F.

## 5 RESULTS

## 5.1 IN-DOMAIN PREDICTION

Table 1 reports in-domain RMSE and $R ^ { 2 }$ for each backbone, averaged over the SF and AD test sets. For every backbone we trained both a PA source model and a raw source model, and the table shows whichever of the two had the lower combined RMSE. The PA source model is the better of the two for ten of the twelve backbones. The two exceptions are GAT and Transolver++, where the raw model is slightly better in domain. The PA therefore does not cost in-domain accuracy in most cases, even though its main purpose is transfer.

VM-UNet is the most accurate backbone in both regions, with a combined RMSE of 0.054 m and $R ^ { 2 } ~ = ~ 0 . 9 6 6$ . Swin V2, MaxViT, and CASPIAN follow closely, and the two depth-foundation models come next. The graph and operator backbones reach a similar $R ^ { 2 }$ of about 0.92, but their absolute errors are several times larger than those of the dense backbones, so we compare RMSE mainly within each family. Across all backbones, errors are lower in SF than in AD, which is consistent with the stronger wave forcing and run-up in the AD simulations (Appendix A.1). Full results for all seven metrics, per region, are given in Appendix G.1.

## 5.2 FEW-SHOT TRANSFER

We first compare the ten regimes averaged over all twelve backbones and all four transfer settings (SF→AD, AD→SF, $\mathrm { S F } _ { 1 . 0 } {  } \mathrm { S F } _ { 0 . 5 }$ , and $\mathrm { S F } _ { 1 . 0 } {  } \mathrm { S F } _ { 1 . 5 } )$ . Since the SLR targets stop at $K = 1 0$ , this comparison uses $K \leq 1 0$ . Figure 2a shows the resulting RMSE curves, with regimes ranked by their mean RMSE over K.

Regimes with the PA. The five regimes that include the PA take the top five places, and the five regimes without it take the bottom five. From $K = 1$ onward, every PA regime has a lower RMSE than every non-PA regime at every value of K. FT+PA is the best regime overall, reducing RMSE from 0.856 m at $K = 0 \mathrm { ~ t o ~ } 0 . 1 6 9 \mathrm { m }$ at $K = 1 0$ . The more useful comparison for practice is PAonly adaptation, which updates only the adapter parameters $\phi _ { m }$ . It has a lower RMSE than full fine-tuning without the PA at every $K \geq 1$ , even though FT updates the whole backbone. Among the PEFT methods, adding the PA lowers the error for LoRA, BitFit, and $\mathrm { L A ^ { 3 } }$ alike, and the three PEFT+PA regimes end close to each other at $K = 1 0$

Table 1: Per-backbone results. In-domain values report the better PA or raw source model over SF and AD. Transfer results report the lowest mean RMSE over all K and all four settings, with and without full fine-tuning. Values are mean ± std over three seeds. Bold shows the best results.
<table><tr><td colspan="5">In-domain</td><td colspan="3">Transfer (mean over K)</td></tr><tr><td>Backbone</td><td>Source</td><td>RMSE (m)</td><td> $R ^ { 2 }$ </td><td></td><td></td><td>|Best regime RMSE Best without full FT RMSE</td><td></td></tr><tr><td>VM-UNet</td><td>PA</td><td>0.0542 ± 0.0015 0.9660 ± 0.0008</td><td></td><td>FT+PA</td><td>0.1421</td><td>LoRA</td><td>0.1467</td></tr><tr><td>Swin V2</td><td>PA</td><td></td><td>0.0630 ± 0.0025 0.9582 ± 0.0055</td><td>FT+PA</td><td>0.1635</td><td>LoRA+PA</td><td>0.1662</td></tr><tr><td>MaxViT</td><td>PA</td><td></td><td>0.0639 ± 0.0033 0.9564 ± 0.0039</td><td>FT+PA</td><td>0.1663</td><td>LoRA+PA</td><td>0.1749</td></tr><tr><td>CASPIAN</td><td>PA</td><td></td><td>0.0686 ± 0.0036 0.9560 ± 0.0060</td><td>FT+PA</td><td>0.1388</td><td>LoRA+PA</td><td>0.1477</td></tr><tr><td>Depth Pro</td><td>PA</td><td></td><td>0.0713 ± 0.0046 0.9522 ± 0.0005</td><td>FT+PA</td><td>0.2245</td><td>LoRA</td><td>0.2423</td></tr><tr><td>Depth Anything V2</td><td>PA</td><td></td><td>0.0747 ± 0.0017 0.9503 ± 0.0052</td><td>FT+PA</td><td>0.2200</td><td>LoRA</td><td>0.2340</td></tr><tr><td>ControlNet</td><td>PA</td><td></td><td>0.0754 ± 0.0030 0.9244 ± 0.0096</td><td>FT+PA</td><td>0.1637</td><td>LoRA+PA</td><td>0.1722</td></tr><tr><td>ConvNeXt V2</td><td>PA</td><td></td><td>0.0893 ± 0.0028 0.9217 ± 0.0157</td><td>FT+PA</td><td>0.1648</td><td>LoRA+PA</td><td>0.1663</td></tr><tr><td>MGN</td><td>PA</td><td></td><td>0.2990 ± 0.0034 0.9266 ± 0.0011</td><td>FT+PA</td><td>0.6199</td><td>IA³+PA</td><td>0.6449</td></tr><tr><td>GAT</td><td>Raw</td><td></td><td>0.3106 ± 0.0019 0.9235 ± 0.0007</td><td>FT+PA</td><td>0.4950</td><td>BitFit+PA</td><td>0.5760</td></tr><tr><td>Transolver++</td><td>Raw</td><td>0.3417 ± 0.0052 0.9215 ± 0.0079</td><td></td><td>LoRA+PA</td><td>0.7125</td><td>LoRA+PA</td><td>0.7125</td></tr><tr><td>GCN</td><td>PA</td><td>0.3418 ± 0.0020 0.9121 ± 0.0009</td><td></td><td>FT+PA</td><td>0.5209</td><td>BitFit+PA</td><td>0.6456</td></tr></table>

Zero-shot behavior. At K = 0 the ordering is reversed, and the raw source models transfer better than the PA source models (RMSE of about 0.674 m against 0.856 m). A likely reason is that the reference water level η in Eq. (3) is learned for the source region, so without any target data the terrain comparison is made against the wrong water level. A single labeled target scenario is enough to reverse this, and at K = 1 all PA regimes are already ahead. In practice, the PA should be used with at least one target simulation, and zero-shot use requires care.

Where the gain comes from. Figure 2b isolates the effect of the PA by comparing each regime with its matched counterpart, namely FT with FT+PA, each PEFT method with its PEFT+PA version, and NPA with PA. The PA lowers RMSE by 22.87% for full fine-tuning, 15.39% for PEFT, and 11.51% for head-only adaptation at $K = 3 ,$ , and by 12.77%, 5.78%, and 1.68% averaged over all K. The NPA against PA pairing is the cleanest test, since both regimes train only a small head on a frozen backbone and differ only in whether that head is conditioned on terrain. This gain points to the terrain conditioning as the main source of the improvement, although the two heads also differ somewhat in size (Appendix F).

Per-backbone results. The last four columns of Table 1 show the best regime for each backbone. FT+PA is the best choice for eleven of the twelve backbones, and LoRA+PA is best for Transolver++. Because full fine-tuning is the most expensive option, we also report the best regime when both full fine-tuning regimes are excluded. In this case, a regime with the PA is still best for nine of the twelve backbones. LoRA+PA is preferred by most dense backbones, while the graph backbones prefer the lighter BitFit+PA and $\mathrm { I A ^ { 3 } { + } P A }$ . The three exceptions, VM-UNet, Depth Pro, and Depth Anything V2, prefer LoRA without the PA, and for these the gap to LoRA+PA is 0.03 m or less. The full per-backbone curves are given in Appendix G.2.

## 5.3 QUALITATIVE RESULTS

Figure 3 compares VM-UNet PWL predictions for one held-out case per region, using the in-domain model and four $K = 3$ transfer regimes. In AD, the in-domain model matches the flooded area within 1.5%. After SF-to-AD transfer, FT and LoRA without PA overpredict flooding by 63.5% and 26.3%, with false wet patches over dry inland areas. Adding PA largely removes these errors, reducing the flooded-area error to 13.3% for FT+PA and 7.8% for PEFT+PA. This is consistent with Eq. (3), where elevated terrain remains dry unless the learned features support flooding. It also shows that lower RMSE does not always imply a better flood extent. In SF, all AD-to-SF regimes recover the flooded area within 5%, with differences mainly in predicted water levels inside flooded regions. Error maps are provided in Appendix G.3.

![](images/28e0f4c01a71bf62f5aca3fe818d26936adf21044018f0341f2926a13bd4121f.jpg)  
(a)

![](images/b8f887ea7c2931c0e3564f92257deb5aed1268e393c0914449b73594b90c1353.jpg)  
(b)

Figure 2: Few-shot transfer averaged over 12 backbones and four settings. (a) RMSE across K for the ten adaptation regimes, ranked by mean RMSE. Solid lines use PA; dotted lines do not. (b) RMSE reduction from PA at $K = 3 , K = 5 ,$ , and across all K. “Balanced” averages the three pairings.  
![](images/f6f249069499d27fe7d0496aaccd3b0b3b94f5c8ead357c614d9456b3d7affa6.jpg)

![](images/c9035980cca84828d03383dc22d37c9e0177f994641b142fcb5d2775ec607361.jpg)

![](images/8d3afab067742c609f08ad2a8619c4911675364a9951022fab6d39c520f9ce9a.jpg)

![](images/53930e5f425bc622eebb3e8474230ccebf36dbd932dd58c4d865599c74461cc1.jpg)

![](images/a68a74e83437f467441e910acadd52ab991c68a6256a80d1d5143e70e160d40f.jpg)

![](images/e3debbdb001118a44618948ec66bec5444a1a010585f1f84b8e88e50b0800d48.jpg)

![](images/dc6ab2a1cbc1eeb2cc66775f37a47565220150035e7481acd10e0c9a65a87355.jpg)  
(a)

![](images/b70b2a9a0560a9bb0628930b72649634b019501625b692392a999535c4add1e6.jpg)  
(b)

![](images/8b1de7099eeeb09a3ac6b6e6e5e8c1ec1ea35fd5ea5ef22e825c9cc3ecbd47da.jpg)  
(c)

![](images/05c3385a44676297cb1df69d38d2b4862ca43374e7892bdcfa8810b0904466f5.jpg)  
(d)

![](images/3ada731cb1ac9445586ea1ae25b7deb26fa89064ebf9352dafd5cbecc67ef975.jpg)  
(e)

![](images/5048e8ad92ee2ee7aca77850aa6776675b949e079a12d810da7766a182930df1.jpg)  
(f)  
Figure 3: VM-UNet PWL predictions for held-out AD (top) and SF (bottom) scenarios. (a) Ground truth, (b) in-domain, and K = 3 transfer using (c) FT+PA, (d) FT, (e) best PEFT+PA, and (f) best PEFT. Transfer directions are SF→AD and AD→SF, respectively. Each panel reports flooded area and its difference from ground truth. Color scales vary across panels.

## 6 CONCLUSION

We introduced the Physics Adapter, a small module that conditions coastal-flood predictions on terrain elevation through a differentiable wet/dry response and a gated physics-guided branch. The same formulation was attached to twelve backbones from graph, mesh, operator, vision, state-space, depth-foundation, and diffusion families and tested on geographic and SLR transfer with up to 10 labeled target scenarios. With at least one target scenario, every regime that includes the PA outperforms every regime without it in our aggregate comparison. Training the adapter alone is already better than fully fine-tuning a backbone without it, and adding it on top of LoRA, BitFit, or $\mathrm { I A } ^ { \mathrm { 3 } }$ improves each of them. The PA also keeps or improves in-domain accuracy for ten of the twelve backbones.

Limitations. This study covers two regions, and SLR transfer is tested only within San Francisco, so broader claims need more regions and hazard types. The PA encodes a terrain comparison and not hydrodynamics, so it gives no guarantee of mass or momentum conservation. Without target data, PA source models transfer worse than raw ones, which limits zero-shot use. Finally, the interfaces, batch sizes, and PEFT insertion sites differ across architectures by design, so small differences between backbones should be read with care.

## AI USE STATEMENT

In this paper, we used generative AI tools for editing and rephrasing the text to improve grammar and readability, and for drafting and editing the source code for experiments and visualization. We have not used generative AI tools to generate synthetic data sets; help develop theoretical models or conceptual frameworks; formulate mathematical claims; provide critical ingredients for proving mathematical claims; assist in the writing of proofs; propose or refine hypotheses; design or provide feedback on research methodology or experiments; implement methods; assist with translation; clean and reformat dataset; support qualitative and thematic data analysis; and interpret results. All AI-paraphrased or AI-edited text was reviewed and revised by the authors. We take full responsibility for the final content of this work, including all text and claims produced with the aid of generative AI.

## ETHICS STATEMENT

This work does not involve human subjects, personal data, or privacy-sensitive information. All experiments use numerical hydrodynamic simulations and publicly available geospatial datasets, used in accordance with their respective licenses. We are not aware of any conflicts of interest or other ethical concerns associated with this work.

## REPRODUCIBILITY STATEMENT

We will release the full codebase, including all scripts for data preprocessing, source training, target adaptation, evaluation, and the reproduction of every table and figure in this paper. Given the size of the codebase, which spans twelve backbones and ten adaptation configurations, we are currently consolidating and documenting it, and we will share an anonymized repository link with the reviewers during the discussion period. In the meantime, the paper provides the details needed to re-implement our method and experiments.

## REFERENCES

Wanchao Bian, Jiayi Fang, Pin Wang, Qinke Sun, Jian Fang, Feng Kong, and Tangao Hu. Deep learning surrogate models for spatiotemporal prediction of coastal flooding inundations in tianjin, china. Journal of Hydrology: Regional Studies, 60:102593, 2025.

Aleksei Bochkovskiy, Amael Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R.¨ Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second. In International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=aueXfY0Clv.

N. Booij, R. C. Ris, and L. H. Holthuijsen. A third-generation wave model for coastal regions: 1. model description and validation. Journal of Geophysical Research: Oceans, 104(C4):7649– 7666, 1999. doi: 10.1029/98JC02622.

Aaron C. Chow and Jian Sun. Combining sea level rise inundation impacts, tidal flooding and extreme wind events along the abu dhabi coastline. Hydrology, 9(8):143, 2022. doi: 10.3390/ hydrology9080143.

Samuel Daramola, David F. Munoz, and Chaopeng Shen. Toward transferable models for efficient˜ spatiotemporal flood prediction across coastal-estuarine systems. Cambridge Prisms: Coastal Futures, 4:e13, 2026. doi: 10.1017/cft.2026.10037.

European Space Agency. Copernicus global digital elevation model (GLO-30). ESA Copernicus Data Space Ecosystem, 2022.

Mohammad Ahmadi Gharehtoragh and David R. Johnson. Using surrogate modeling to predict storm surge on evolving landscapes under climate change. npj Natural Hazards, 1:33, 2024. doi: 10.1038/s44304-024-00032-9.

Bilal Hassan, Areg Karapetyan, Aaron Chung Hin Chow, and Samer Madanat. Climate adaptationaware flood prediction for coastal cities using deep learning. Hydrology and Earth System Sci ences, 30(5):1333–1358, 2026.

Hans Hersbach, Bill Bell, Paul Berrisford, Shoji Hirahara, Andras Hor´ anyi, Joaqu´ ´ın Munoz-Sabater,˜ Julien Nicolas, Carole Peubey, Raluca Radu, Dinand Schepers, Adrian Simmons, Cornel Soci, Saleh Abdalla, Xavier Abellan, Gianpaolo Balsamo, Peter Bechtold, Gionata Biavati, Jean-Raymond Bidlot, Massimo Bonavita, Giovanna De Chiara, Per Dahlgren, Dick Dee, Michail Diamantakis, Rossana Dragani, Johannes Flemming, Richard Forbes, Manuel Fuentes, Alan Geer, Leo Haimberger, Sean Healy, Robin J. Hogan, El´ıas Holm, Marta Janiskov´ a, Sarah Keeley, Patrick´ Laloyaux, Philippe Lopez, Cristina Lupu, Gabor Radnoti, Patricia de Rosnay, Iryna Rozum, Freja Vamborg, Sebastien Villaume, and Jean-Noel Th¨ epaut. The ERA5 global reanalysis.´ Quarterly Journal ofthe Royal Meteorological Society, 146(730):1999–2049, 2020. doi: 10.1002/qj.3803.

Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pp. 2790–2799. PMLR, 2019.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. 2021.

Gaofeng Jia, Ruo Qian Wang, and Mark T. Stacey. Investigation of impact of shoreline alteration on coastal hydrodynamics using Dimension REduced Surrogate based Sensitivity Analysis. Advances in Water Resources, 126:168–175, 4 2019. ISSN 0309-1708. doi: 10.1016/J. ADVWATRES.2019.03.001.

Areg Karapetyan, Aaron CH Chow, and Samer Madanat. Deep vision-based framework for coastal flood prediction under sea level rise and shoreline protection. Scientific Reports, 16(1):3663, 2026.

Arnold Kazadi, James Doss-Gollin, and Arlei Lopes Da Silva. Pluvial flood emulation with hydraulics-informed message passing. In Forty-first International Conference on Machine Learning, 2024.

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017. URL https: //openreview.net/forum?id=SJU4ayYgl.

Yoonho Lee, Annie S Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea Finn. Surgical fine-tuning improves adaptation to distribution shifts. 2022.

Giles R Lesser, JA v Roelvink, JA Th M van Kester, and GS Stelling. Development and validation of a three-dimensional morphological model. Coastal engineering, 51(8-9):883–915, 2004.

Zongyi Li, Hongkai Zheng, Nikola Kovachki, David Jin, Haoxuan Chen, Burigede Liu, Kamyar Azizzadenesheli, and Anima Anandkumar. Physics-informed neural operator for learning partial differential equations. ACM/IMS Journal ofData Science, 1(3):1–27, 2024.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. volume 35, pp. 1950–1965, 2022a.

Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Jianbin Jiao, and Yunfan Liu. VMamba: Visual state space model. In Advances in Neural Information Processing Systems, volume 37, pp. 103031–103063, 2024. doi: 10.52202/079017-3273.

Ze Liu, Han Hu, Yutong Lin, Zhuliang Yao, Zhenda Xie, Yixuan Wei, Jia Ning, Yue Cao, Zheng Zhang, Li Dong, Furu Wei, and Baining Guo. Swin transformer v2: Scaling up capacity and resolution. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12009–12019, 2022b. doi: 10.1109/CVPR52688.2022.01170.

Emiliano Longo, Andrea Ficch\`ı, Martin Verlaan, Sanne Muis, and Andrea Castelletti. A deep learning framework for extreme storm surge modeling under future climate scenarios. Earth’s Future, 14(3):e2025EF007072, 2026. doi: 10.1029/2025EF007072.

Huakun Luo, Haixu Wu, Hang Zhou, Lanxiang Xing, Yichen Di, Jianmin Wang, and Mingsheng Long. Transolver++: An accurate neural solver for PDEs on million-scale geometries. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 41432–41449. PMLR, 2025. URL https://proceedings.mlr.press/v267/luo25o.html.

Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W. Battaglia. Learning meshbased simulation with graph networks. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum?id=roNqYL0\_XP.

Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal ofComputational physics, 378:686–707, 2019.

Julian R. Rice, Karthik Balaguru, Fadia Ticona Rollano, John Wilson, Brent Daniel, David Judi, Ning Sun, and L. Ruby Leung. Projecting U.S. coastal storm surge risks and impacts with deep learning. Environmental Research Letters, 20:104013, 2025. doi: 10.1088/1748-9326/adfd74.

Peter Rivera-Casillas, Sourav Dutta, Shukai Cai, Mark Loveland, Kamaljyoti Nath, Khemraj Shukla, Corey Trahan, Jonghyun Lee, Matthew Farthing, and Clint Dawson. A neural operator emulator for coastal and riverine shallow water dynamics. arXiv preprint arXiv:2502.14782, 2025.

Jiacheng Ruan, Jincheng Li, and Suncheng Xiang. VM-UNet: Vision mamba UNet for medical image segmentation. arXiv preprint arXiv:2402.02491, 2024. URL https://arxiv.org/ abs/2402.02491.

Jiayun Sun, Aaron C. H. Chow, and Samer M. Madanat. Multimodal transportation system protection against sea level rise. Transportation Research Part D: Transport and Environment, 88: 102568, 2020.

Mehdi Taghizadeh, Zanko Zandsalimi, Mohammad Amin Nabian, Jonathan L Goodall, and Negin Alemazkoor. Floodforecaster: A domain-adaptive geometry-informed neural operator framework for rapid flood forecasting. Journal of Hydrology, pp. 134512, 2025a.

Mehdi Taghizadeh, Zanko Zandsalimi, Mohammad Amin Nabian, Majid Shafiee-Jood, and Negin Alemazkoor. Interpretable physics-informed graph neural networks for flood forecasting. Computer-Aided Civil and Infrastructure Engineering, 40(18):2629–2649, 2025b.

Zhengzhong Tu, Hossein Talebi, Han Zhang, Feng Yang, Peyman Milanfar, Alan C. Bovik, and Yinxiao Li. MaxViT: Multi-axis vision transformer. In Computer Vision – ECCV 2022, pp. 459–479. Springer, 2022. doi: 10.1007/978-3-031-20053-3 27.

Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Li´ o, and Yoshua\` Bengio. Graph attention networks. In International Conference on Learning Representations, 2018. URL https://openreview.net/forum?id=rJXMpikCZ.

Yogesh Verma, Markus Heinonen, and Vikas Garg. Climode: Climate and weather forecasting with physics-informed neural odes. In International Conference on Learning Representations, volume 2024, pp. 8408–8430, 2024.

Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. 2020.

Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon, and Saining Xie. ConvNeXt V2: Co-designing and scaling convnets with masked autoencoders. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16133–16142, 2023. doi: 10.1109/CVPR52729.2023.01548.

Qingsong Xu, Leon Frederik De Vos, Yilei Shi, Nils Ruther, Axel Bronstert, and Xiao Xiang Zhu.¨ Urban flood modeling and forecasting with deep neural operator and transfer learning. Journal of Hydrology, 661:133705, 2025.

Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. In Advances in Neural Information Processing Systems, volume 37, pp. 21875–21911, 2024. doi: 10.52202/079017-0688.

Masakazu Yoshimura, Teruaki Hayashi, and Yota Maeda. Mambapeft: Exploring parameterefficient fine-tuning for mamba. In International Conference on Learning Representations, volume 2025, pp. 94093–94117, 2025.

Elad Ben Zaken, Yoav Goldberg, and Shauli Ravfogel. Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 1–9, 2022.

Daniele Zanaga, Ruben Van De Kerchove, Dirk Daems, Wanda De Keersmaecker, Carsten Brockmann, Grit Kirches, Jan Wevers, Oliver Cartus, Maurizio Santoro, Steffen Fritz, Myroslava Lesiv, Martin Herold, Nandin-Erdene Tsendbazar, Panpan Xu, Fabrizio Ramoino, and Olivier Arino. ESA WorldCover 10 m 2021 v200, 2022. URL https://zenodo.org/records/ 7254221.

Hangwei Zhang, Chun Kang, Yan Wang, and Difan Zou. F-adapter: Frequency-adaptive parameterefficient fine-tuning in scientific machine learning. volume 38, pp. 111120–111162, 2026.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pp. 3836–3847, 2023. doi: 10.1109/ICCV51070.2023.00355.

Jinpai Zhao, Albert Cerrone, Eirik Valseth, Leendert Westerink, and Clint Dawson. Storm surge in color: Rgb-encoded physics-aware deep learning for storm surge forecasting. Computational Geosciences, 30(4):78, Aug 2026. ISSN 1573-1499. doi: 10.1007/s10596-026-10477-8. URL https://doi.org/10.1007/s10596-026-10477-8.

Kejun Zhu, Xiaoping Chen, Xin Ao, and Zhipeng He. Parameter-efficient transfer of physicsinformed neural networks for buoyancy-driven enclosures via geometry-conditioned adapters. Physics of Fluids, 38(2), 2026.

## APPENDIX

## A DATA CONSTRUCTION

The datasets used in this research were built from physics-based coastal flood simulations over a common set of retained spatial locations. We filtered raw hydrodynamic outputs to the learning locations, and linked each location to its hydrodynamically derived shoreline-protection dependence, terrain elevation, land cover, and scenario-specific PWL. The resulting coordinate-level data were then encoded in two forms, a regular 1024×1024 spatial tensor for dense-grid backbones and a graph that keeps the neighborhood structure of the hydrodynamic computational grid for graph-native backbones. These are two representations of the same flood-prediction problem, not separately generated datasets. Figure 4 summarizes the pipeline.

## A.1 HYDRODYNAMIC SIMULATION DATA

The ground-truth flood fields come from the hydrodynamic models described in the CASPIAN studies and their supplementary material (Hassan et al., 2026). In both regions, Delft3D (Lesser et al., 2004) was used to resolve time-varying coastal water levels over the computational domain under prescribed SLR, tidal forcing, shoreline-protection configurations, and the other regional forcings of the original setup. The simulator produces spatially resolved water-level time series, from which peak water level (PWL) is kept as the regression target.

The Abu Dhabi configuration also accounts for the wind and wave environment of the Arabian Gulf. The validated Delft3D model was forced with ERA5 winds (Hersbach et al., 2020), and its results were coupled to the SWAN spectral wave model (Booij et al., 1999) to represent windwave generation and nearshore wave transformation. The SWAN significant wave heights were then combined with local shoreline slope to estimate coastal run-up under conditions typical of prolonged Shamal events (Chow & Sun, 2022). San Francisco Bay is treated differently because its shoreline lies inside a sheltered bay. The CASPIAN study did not apply SWAN there, and Delft3D alone was used to generate the SLR-driven flood fields (Sun et al., 2020).

![](images/8a6108de1738da7d49ad00a38680708d4dd85a2364d2972c5fa985d497530152.jpg)  
Figure 4: From hydrodynamic simulation to the two learning representations. Each retained coastal location carries scenario-dependent OLU status, elevation, and land cover, with simulated PWL as the target. Dense backbones use the rasterized tensor, and graph backbones use the mesh-topology graph built over the same locations.

The shoreline of region r is divided into $N _ { r }$ operational landscape units (OLUs), with $N _ { \mathrm { A D } } = 1 7$ and $N _ { \mathrm { S F } } = 3 0 . \ \mathrm { A }$ protection configuration is a binary vector $\mathbf { s } = \left( s _ { 1 } , \ldots , s _ { N _ { r } } \right)$ , with $s _ { k } = 1$ when OLU k is protected and $s _ { k } = 0$ otherwise, and it is realized in the hydrodynamic model through the corresponding shoreline-defense setup. Figures 5 and 6 show the OLUs of each region and the flooding produced when none of them is protected. For scenario ${ \mathbf { s } } ,$ the simulator output used for learning is the set

$$
\begin{array} { r } { \mathcal { R } ^ { ( \mathbf { s } ) } = \big \{ \big ( \mathbf { r } _ { i } , y _ { i } ^ { ( \mathbf { s } ) } \big ) \big \} _ { i = 1 } ^ { N } , \qquad \mathbf { r } _ { i } = ( x _ { i } , y _ { i } ^ { \mathrm { c o o r d } } ) , } \end{array}\tag{9}
$$

where $\mathbf { r } _ { i }$ is a simulator location and $y _ { i } ^ { ( \mathbf { s } ) }$ its PWL. Small negative PWL values in intermediate files are clipped to zero during preprocessing. The terrain and bathymetry used inside Delft3D belong to the hydrodynamic model and are separate from the elevation feature of Appendix A.3, which is sampled independently for the learning representation.

Spatial curation. The full hydrodynamic domain contains locations that are not prediction sites for the learning task. Simulator coordinates were therefore curated to retain study-relevant coastal locations and exclude offshore, open-water, and other non-target parts of the domain. The representation scripts start from the resulting region-specific master coordinate sets and match each scenario to them. PWL, elevation, land cover, and OLU dependence are aligned on these retained locations by explicit coordinate matching rather than row order, and duplicate coordinates are removed during feature extraction. Curation does not remove persistently wet locations, since the dependence construction keeps and labels locations that stay flooded even under full protection.

## A.2 HYDRODYNAMICALLY DERIVED OLU DEPENDENCE

The protection status of a location is derived from its simulated response to OLU perturbations, not from its distance to the nearest protected or unprotected shoreline segment. We compute the dependence once per retained location and then combine it with each scenario’s protection vector. Figure 7 shows why proximity alone is not enough. Protecting part of the shoreline dries most of the areas behind it, but it can also raise water levels or cause new flooding elsewhere.

Let 0 denote the all-unprotected configuration, 1 the all-protected configuration, $\mathbf { e } _ { k }$ the configuration protecting only OLU k, and ${ \bf 1 } - { \bf e } _ { k }$ the configuration unprotecting only OLU k. For every available single-OLU perturbation,

$$
\Delta _ { i k } ^ { + } = \big [ y _ { i } ^ { ( 0 ) } - y _ { i } ^ { ( \mathbf { e } _ { k } ) } \big ] _ { + } , \qquad \Delta _ { i k } ^ { - } = \big [ y _ { i } ^ { ( 1 - \mathbf { e } _ { k } ) } - y _ { i } ^ { ( 1 ) } \big ] _ { + } , \qquad \Delta _ { i k } = \operatorname* { m a x } \big ( \Delta _ { i k } ^ { + } , \Delta _ { i k } ^ { - } \big ) .\tag{10}
$$

![](images/7d6e7e802a012c6a2e86493e75d97484fa8d44fbb02d198b70413fa8608778ca.jpg)  
(a)

![](images/55aa6eb86f14663bb04fa8745a6658d34bdcdef29190bea2cc287ccd2c9d7c2c.jpg)  
(b)  
Figure 5: Abu Dhabi study region. (a) The 17 OLUs, with each shoreline segment colored and numbered by its OLU. (b) Simulated PWL at 0.5 m SLR when no OLU is protected, giving a flooded area of 187.9 km<sup>2</sup>.

The first term measures the PWL reduction from protecting OLU k on an otherwise unprotected shoreline, and the second measures the PWL increase from removing protection at k on an otherwise fully protected shoreline. Either direction is enough to identify an influence, and a term stays zero when its perturbation simulation is unavailable. With the largest local response $\Delta _ { i } ^ { \mathrm { m a x } } = \operatorname* { m a x } _ { k } \Delta _ { i k }$ the location-specific guardian threshold is

$$
T _ { i } = \mathrm { m a x } \big ( 0 . 1 0 \mathrm { m } , 0 . 5 \Delta _ { i } ^ { \mathrm { m a x } } \big ) ,\tag{11}
$$

and OLU k is a guardian of location i when $\Delta _ { i k } \ > \ T _ { i }$ . Because the threshold is relative to the strongest local response, several OLUs can guard the same location. The guardian set is stored as the integer bitmask

$$
G _ { i } = \sum _ { k = 1 } ^ { N _ { r } } \mathbb { I } \big [ \Delta _ { i k } > T _ { i } \big ] \ : 2 ^ { k - 1 } ,\tag{12}
$$

where bit $k - 1$ marks dependence on OLU k. The number of guardians and the maximum-impact OLU are also recorded for diagnostics, but the bitmask is the dependence representation used downstream.

Two special cases are handled with $\epsilon _ { \mathrm { d r y } } = 0 . 0 5$ m before the bitmask is finalized. A point with $y _ { i } ^ { ( 0 ) } < \epsilon _ { \mathrm { d r y } }$ stays dry even with no protection, so its guardian set is forced empty to avoid spurious dependence from numerical noise. A point with $y _ { i } ^ { ( 1 ) } > \epsilon _ { \mathrm { d r y } }$ stays wet even under full protection. Its bitmask is also cleared, and a separate always-flooded flag keeps the distinction.

Scenario-specific OLU status. With guardian set $\mathcal { G } _ { i } = \{ k : \Delta _ { i k } > T _ { i } \}$ and unprotected OLUs $\mathcal { U } ( \mathbf { s } ) = \lbrace k : s _ { k } = 0 \rbrace$ , the categorical status used by both representations is

$$
c _ { i } ^ { ( \mathbf { s } ) } = \left\{ \begin{array} { l l } { 2 , } & { \mathrm { p o i n t ~ } i \mathrm { ~ i s ~ a l w a y s ~ f l o o d e d , } } \\ { 0 , } & { \mathcal { G } _ { i } = \emptyset , } \\ { 2 , } & { \mathcal { G } _ { i } \cap \mathcal { U } ( \mathbf { s } ) \not = \emptyset , } \\ { 1 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{13}
$$

![](images/a988047f42c8fa4aae6ce11d0cf47602439502854510833b3a139b42529684fb.jpg)  
(a)

![](images/ac2be5887a70bdecd5567178482f49cecac6acc678bac14f34e20cb287814462.jpg)  
(b)  
Figure 6: San Francisco Bay study region. (a) The 30 OLUs, with each shoreline segment colored and numbered by its OLU. (b) Simulated PWL at 1.0 m SLR when no OLU is protected, giving a flooded area of 461.1 km<sup>2</sup>.

Status 0 means no active OLU dependence, status 1 an OLU-dependent location whose guardians are all protected, and status 2 an OLU-dependent location with at least one unprotected guardian, or an always-flooded location. The grid and graph generators use the same definition for both regions.

## A.3 TERRAIN AND LAND-COVER ATTRIBUTES

Each retained location is given an elevation and a land-cover value, both sampled independently of the simulator’s internal bathymetry. Land cover $\ell _ { i }$ is sampled from ESA WorldCover 10 m 2021 v200 (Zanaga et al., 2022) with its original class codes (10 tree cover, 20 shrubland, 30 grassland, 40 cropland, 50 built-up, 60 bare or sparse vegetation, 70 snow and ice, 80 permanent water bodies, 90 herbaceous wetland, 95 mangroves, and 100 moss and lichen). The grid representation keeps these codes, and the graph remaps them to contiguous categories (Appendix A.5). Elevation $z _ { i }$ is sampled from Copernicus DEM GLO-30 at a nominal 30 m resolution (European Space Agency, 2022). It is different from the bathymetric and terrain products used to build the Delft3D domains, and it is the elevation passed to the models and to the PA.

San Francisco coordinates are processed in WGS 84 / UTM Zone 10N (EPSG:32610), and Abu Dhabi coordinates in UTM Zone 40N (EPSG:32640). Both are transformed to WGS 84 geographic coordinates (EPSG:4326) before raster sampling, and land cover and elevation are read directly at the transformed coordinates from region-specific WorldCover tiles and GLO-30 rasters. Each location therefore has three model attributes $( c _ { i } ^ { ( \mathbf { s } ) } , z _ { i } , \ell _ { i } )$ and the target PWL $y _ { i } ^ { ( \mathbf { s } ) }$ . The occupancy mask introduced below is a structural grid indicator, not a physical attribute. Figure 8 shows elevation, land cover, and the OLU dependence for both regions.

![](images/73135969f6439647d8e6ac23155296ab023bf60dd3fa3595bc33e9b372a1a75d.jpg)

![](images/0fa2fa1d8d6d0c58826c21d8ae44b896ea108ca1e5cf634887275479ab32b307.jpg)

![](images/a8b8bc8ee8177c1bc237a4fdd9ea6aeaddc0dd9e6e38b263f766eb905cc44ed2.jpg)

![](images/87ec68b969a01dad08649a86938e6704c1852c07906b54b391be41579ca4a1cb.jpg)  
(a)

![](images/a208609cfdf30f826fdc7f16cd081d18c6bff79fdb682bf9aee9c2ba58f829a1.jpg)  
(b)

(c)  
![](images/356b30f55c61faf4eeea77f2bdb45776f4a5763f58a6fed7654f8bd1f1655077.jpg)  
Figure 7: Effect of shoreline protection on flooding in AD (top, 0.5 m SLR) and SF (bottom, 1.0 m SLR). (a) PWL when no OLU is protected. (b) PWL for one protection scenario, with protected OLUs drawn as solid lines and unprotected OLUs as dotted lines. The flooded area falls by 72% in AD and 18% in SF. (c) Change between (a) and (b). Most of the change is locations that become dry $( 1 3 6 . 1 \mathrm { k m ^ { 2 } }$ in AD and $8 6 . 2 \mathrm { k m ^ { 2 } }$ in SF), but some locations see a higher water level and a small area becomes newly flooded $( 0 . 6 \mathrm { k m ^ { 2 } }$ in AD and 1.1 km<sup>2</sup> in SF).

## A.4 REGULAR-GRID REPRESENTATION

Dense backbones use a $1 0 2 4 \times 1 0 2 4$ representation of the retained coordinates. The generator uses natural geographic bins with $N _ { q } = \mathrm { i } 0 2 4$ and does not move colliding points into neighboring cells. Let $x _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } , y _ { \mathrm { m i n } } , y _ { \mathrm { m a x } }$ be the extrema of the retained coordinates. A 2% margin of the coordinate range is added on each axis, giving $\widetilde { x } _ { \mathrm { m i n } } , \widetilde { x } _ { \mathrm { m a x } } , \widetilde { y } _ { \mathrm { m i n } } , \widetilde { y } _ { \mathrm { m a x } } .$ , and each coordinate is assigned the bin

$$
\begin{array} { r } { u _ { i } = \mathrm { c l i p } \left( \left\lfloor N _ { g } \frac { x _ { i } - \widetilde { x } _ { \mathrm { m i n } } } { \widetilde { x } _ { \mathrm { m a x } } - \widetilde { x } _ { \mathrm { m i n } } } \right\rfloor , 0 , N _ { g } - 1 \right) , \qquad w _ { i } = \mathrm { c l i p } \left( \left\lfloor N _ { g } \frac { y _ { i } ^ { \mathrm { c o o r d } } - \widetilde { y } _ { \mathrm { m i n } } } { \widetilde { y } _ { \mathrm { m a x } } - \widetilde { y } _ { \mathrm { m i n } } } \right\rfloor , 0 , N _ { g } - 1 \right) . } \end{array}\tag{14}
$$

This mapping is deterministic and stored for every retained coordinate.

Because the hydrodynamic locations are irregularly spaced, several coordinates can fall into the same bin, and they stay there. For each occupied bin, the coordinates are sorted lexicographically

![](images/252cb737d4627bd9b49607bdb452b99068ecd0176026503a8eb9f57a40675bd1.jpg)

![](images/14ad6d9765b4905510e31abb00ee77e3fecb8628f5f10f78ac8077a694f30c38.jpg)

![](images/e97b12f2c1d3bfd6e7d37de150b798a752b3c92ebc937ea865c24c6aa5acc9de.jpg)

![](images/363324cb0c4990b61a503919ece23d7ffbbb7460f1b0e688ac3d5bcfe4387984.jpg)  
(a)

![](images/f0c48597ac06bfd81c8470da0d90af4d0d80ed96825284010b9fafc8c9fde153.jpg)  
(b)

![](images/cf2068c0d56fed8e296c57e1df356eb46aacaea76380e8b4b1d8bd26e24c5234.jpg)  
(c)  
Figure 8: Input attributes for AD (top) and SF (bottom). (a) Copernicus GLO-30 elevation, with median 4.1 m in AD and 1.0 m in SF. (b) ESA WorldCover land cover. AD is dominated by bare or sparse vegetation (50%) and built-up land (34%), while SF has more permanent water (25%), built-up land (24%), and herbaceous wetland (17%). (c) Locations that depend on at least one OLU, colored by the OLU with the largest local impact, covering 24% of the retained locations in AD and 60% in SF. Locations with no OLU dependence are shown in beige. Panel (c) shows only the strongest OLU for display, while the models use the full guardian set of Eq. (12).

and the first $( x , y )$ pair supplies the status, elevation, and land-cover inputs. The target instead uses all points in the bin. With $\bar { \mathcal { C } } _ { u w } = \{ i : ( u _ { i } , w _ { i } ) = ( u , w ) \}$ ,

$$
Y _ { u w } ^ { \mathrm { ( s ) } } = \frac { 1 } { | { \mathcal { C } } _ { u w } | } \sum _ { i \in { \mathcal { C } } _ { u w } } y _ { i } ^ { \mathrm { ( s ) } } , \qquad M _ { u w } = \mathbb { I } \big [ | { \mathcal { C } } _ { u w } | > 0 \big ] ,\tag{15}
$$

so a single point keeps its own PWL and a shared bin receives the mean. The occupancy mask $M _ { u w }$ fills the validity channel v of Sec. 3.1 and separates occupied prediction sites from empty background. Validity cannot be inferred safely from a zero DEM, land-cover, or PWL value.

For scenario s, the stored input tensor has shape $1 0 2 4 \times 1 0 2 4 \times 4$ with channel order OLU status, DEM, WorldCover class, and occupancy mask. The target is a $1 0 2 4 \times 1 0 2 4$ matrix with non-negative

PWL at occupied bins and zero elsewhere. Both arrays are built with the $( u , w )$ indexing of Eq. (14), and their two spatial axes are transposed together before saving, so they share the same orientation. The stored geographic mapping keeps the full coordinate-to-bin map and the coordinates of every occupied bin. A reconstruction utility uses it to return grid predictions to the coordinate level, where coordinates sharing a bin take that bin’s prediction.

## A.5 GRAPH REPRESENTATION

Graph-native backbones use the same retained locations and scenario definitions but keep the neighborhood structure of the hydrodynamic grid instead of rasterizing. Each scenario graph has one node per retained coordinate. Node i stores the continuous feature $[ z _ { i } ]$ , the categorical features $[ \widetilde { \ell } _ { i } , c _ { i } ^ { ( \mathbf { s } ) } ]$ as integer indices, the position $[ x _ { i } , y _ { i } ^ { \mathrm { c o o r d } } ]$ , and the target $y _ { i } ^ { ( \mathbf { s } ) }$ . WorldCover codes are remapped to contiguous categories ${ \bar { 1 } 0 \to 0 , \dot { 2 } 0 \to \bar { 1 } , 3 \dot { 0 } \to 2 , 4 0 \to \bar { 3 } , 5 \dot { 0 } \to 4 , 6 0 \to 5 , 7 0 \to 6 , 8 0 \to 7 , \bar { 9 } \bar { 0 } \to 8 }$ $9 5  9$ , and $1 0 0  1 0$ , with unknown values assigned category 11. The OLU status is the same variable as in Eq. (13). Unlike the grid, the graph contains only prediction nodes and needs no occupancy channel.

Hydrodynamic-grid connectivity. Edges follow the topology of the hydrodynamic computational grid rather than a generic k-nearest-neighbor rule. The generator reads cell centers and facenode coordinates from the region-specific grid files and links each retained coordinate to its nearest cell center with a KD-tree. The San Francisco grid is already in UTM Zone 10N. The Abu Dhabi grid is stored in EPSG:4326, so its cell centers and face nodes are transformed to EPSG:32640 before matching. Two cells are adjacent when they share a complete boundary edge, each adjacent pair is added in both directions, and the adjacency is then restricted to the cells linked to retained coordinates. For an edge $i  j$ , the edge feature is $\mathbf { e } _ { i j } = [ \Delta x _ { i j } , \Delta y _ { i j } , d _ { i j } ]$ , with $\Delta x _ { i j } = x _ { i } - x _ { j }$ $\Delta y _ { i j } = y _ { i } ^ { \mathrm { c o o r d } } - y _ { i } ^ { \mathrm { c o o r d } }$ , and Euclidean distance $d _ { i j }$ . No separate simulations are run for the graph. Scenario PWL, DEM, and land cover are aligned by coordinate before construction, and the same guardian logic gives the scenario-dependent status.

Consistency across representations. In both encodings, a sample is defined by the same retained coordinates, OLU configuration, guardian dependence, Copernicus elevation, WorldCover class, and PWL target. The grid merges points only when they share a geographic bin, while the graph keeps every point as a separate node linked by mesh topology. Any later normalization, projection, or embedding belongs to the model, not to the dataset.

## B ARCHITECTURE-SPECIFIC PHYSICS ADAPTER INTERFACES

Table 2 lists, for each backbone, the representation passed to the PA and how the interface $\mathcal { P } _ { m }$ of Eq. (1) is realized.

Adapter normalization and heads. For raster representations, $\mathrm { B N } _ { A } ^ { ( m ) }$ in Eq. (2) is a channelwise BatchNorm2d, and node-aligned implementations use BatchNorm1d. The CASPIAN, ConvNeXt V2, GAT, GCN, MGN, and Transolver++ adapters have affine BatchNorm parameters, while the MaxViT, Swin V2, VM-UNet, Depth Anything V2, Depth Pro, and ControlNet adapters use affine-free BatchNorm. The data branch applies dropout before its final prediction, with rate 0.10 in the dense and diffusion adapters and 0.30 in the node-aligned adapters.

Initialization. The temperature is initialized to $\tau \ : = \ : 0 . 5 ,$ and $\eta$ is initialized from the sourceregion reference level, 3.0 for San Francisco and 5.0 for Abu Dhabi. Both $\eta$ and ϑ are then learned. The +3 gate initialization of Eq. (5) is a fixed additive logit offset in the dense and ControlNet adapters and an initial value of the final gate-head bias in the node-aligned adapters. This changes the parameterization but not the meaning of Eq. (5). In dense implementations, the gate penalty of Eq. (6) is computed over the full gate tensor before validity masking, so $\Omega _ { A }$ covers all grid cells. In node-aligned implementations, it is computed over the represented nodes.

State-space and depth-foundation interfaces. In VM-UNet, the representation just before the native final regression layer is projected to the adapter width before entering the adapter BatchNorm.

<table><tr><td>Family</td><td>Backbone</td><td>Representation passed to PA</td><td>Interface realization</td></tr><tr><td>Graph</td><td>GCN (Kipf Welling, 2017) GAT (Veličković</td><td>&amp; Node-aligned latent embed- dings Node-aligned attention embed-</td><td>Node-wise PA heads over retained graph locations. Node-wise PA heads over retained</td></tr><tr><td>Graph atten- tion Mesh graph</td><td>et al., 2018) MGN (Pfaff et al.,</td><td>dings Node-aligned latent embed-</td><td>graph locations. Node-wise PA heads after native mesh message passing.</td></tr><tr><td>Scientific at- tention</td><td>2021) Transolver++ (Luo et al., 2025)</td><td>dings Site-aligned operator represen- tation</td><td>Node-wise PA heads at the physical prediction sites.</td></tr><tr><td>Dense vision</td><td>CASPIAN (Kara- petyan et al.,</td><td>Dense spatial feature map</td><td>Dense features, then pointwise PA heads.</td></tr><tr><td>Dense vision</td><td>2026) ConvNeXt V2 (Woo et al., 2023)1</td><td>Dense and multi-scale visual features</td><td>Dense decoding and alignment, then pointwise PA heads.</td></tr><tr><td>Dense vision</td><td>MaxViT (Tu et al., 2022)</td><td>Dense and multi-scale visual features</td><td>Dense decoding and alignment, then pointwise PA heads.</td></tr><tr><td>Dense vision</td><td>Swin V2 (Liu et al., 2022b)</td><td>Dense and multi-scale trans- former features</td><td>Dense decoding and alignment, then pointwise PA heads.</td></tr><tr><td>State space</td><td>VM-UNet (Ruan et al., 2024; Liu</td><td>Decoder feature before the na- tive final regression layer</td><td>Decoder feature projected to the PA width, then pointwise PA heads.</td></tr><tr><td>Depth founda- tion</td><td>et al., 2024) Depth Any- thing V2 (Yang</td><td>Pretrained dense DPT feature before depth regression</td><td>Flood-specific input stem and fea- ture projection align the feature to</td></tr><tr><td>Depth founda- tion</td><td>et al., 2024) Depth Pro (Bochkovskiy</td><td>Pretrained dense depth repre- sentation</td><td>the PWL grid. Task-specific projection and inter- polation align the feature to the</td></tr><tr><td>Diffusion</td><td>et al., 2025) ControlNet (Zhang et</td><td>Generated base PWL map with al., local conditioning</td><td>PWL grid. Adapter-owned post-decoder stem, with the data branch predicting a</td></tr></table>

Table 2: Backbones and their realization of the shared PA interface. The representation column describes the site-aligned quantity passed to the PA, not the full internal architecture. Terrain conditioning, the two prediction branches, gated fusion, and the adaptation protocol are shared across all rows.

Depth Anything V2 similarly takes a dense DPT feature before the original depth regressor, using a learned flood-input stem and a feature projection. In both cases, the raw DEM bypasses this path and enters Eq. (3) directly. These operations belong to the interface, not to the shared adapter equations.

ControlNet interface and objective. The output of ControlNet is generative rather than a deterministic regression field, so its interface differs from the other backbones. The pretrained SD3.5 transformer and VAE stay frozen by design. The trainable ControlNet branch produces the condi tional representation, and a clean-latent estimate is decoded to a base PWL map. A small adapterowned convolutional stem then combines this map with protection status, DEM, and land cover before the shared adapter computation, so the stem parameters belong to $\phi _ { m }$ The data branch predicts a residual around the generated base value $b _ { i }$ . ControlNet keeps its diffusion training ob jective alongside the map objective. Writing $\mathbf { l } _ { 0 }$ for a clean diffusion latent, $\widehat { \mathbf { l } } _ { 0 }$ for the preconditioned clean-latent estimate, and ς for the noise level,

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { \hat { H } o w } } = \mathbb { E } \big [ w ( \varsigma ) \big \lVert \widehat { \boldsymbol { \mathrm { I } _ { 0 } } } - \boldsymbol { \mathrm { \mathbf { 1 } } } _ { 0 } \big \rVert _ { 2 } ^ { 2 } \big ] , \qquad \mathcal { L } _ { \mathrm { C o n t r o l N e t } } = \mathbb { I } _ { \theta _ { \mathrm { C N } } } \mathcal { L } _ { \mathrm { \hat { H } o w } } + \mathbb { I } _ { \mathrm { r o u t e } } \big ( \mathcal { L } _ { \mathrm { p r e d } } + \mathbb { I } _ { \mathrm { P A } } \mathcal { L } _ { g } \big ) , } \end{array}\tag{16}
$$

where the indicators depend on the adaptation regime. The decoded base map is detached from the diffusion computation. Map-level gradients therefore train the post-decoder PA or raw head, and diffusion gradients train the ControlNet branch when it is trainable. For ControlNet, $\theta _ { m }$ denotes the trainable ControlNet branch and its interface.

Raw-head regularization. The raw head $\chi _ { m }$ is regularized independently of the PA. Raster backbones use a statistics-free GroupNorm and convolutional prediction head, and graph backbones use a node-wise normalized MLP. Both use train-time feature jitter and dropout, with a fixed weight decay of $1 0 ^ { - 4 }$ on the head weights. This makes the raw model a regularized no-PA control rather than a plain linear probe.

<table><tr><td>Regime</td><td>point</td><td>Source check- Trainable during adaptation</td><td>Frozen during adaptation</td></tr><tr><td>FT+PA</td><td>PA source</td><td>Trainable backbone and interface, and PA parameters  $\phi _ { m }$ </td><td>Only components fixed by de- sign (e.g., the SD3.5 foundation in ControlNet).</td></tr><tr><td>FT</td><td>Raw source</td><td>Trainable backbone and interface, and prediction head  $\chi _ { m }$ </td><td>Only components fixed by de- sign (e.g., the SD3.5 foundation in ControlNet).</td></tr><tr><td>NPA PA</td><td>Raw source</td><td>Prediction head  $\chi _ { m }$ </td><td>Backbone and interface.</td></tr><tr><td>LoRA (PEFT)</td><td>PA source Raw source</td><td>PA parameters  $\phi _ { m }$  Rank-8 LoRA parameters</td><td>Backbone and interface. Prediction head and all base  $\mathrm { p a } \cdot$ </td></tr><tr><td>BitFit (PEFT)</td><td>Raw source</td><td>Selected bias parameters</td><td>rameters. Prediction head and all other pa-</td></tr><tr><td> $\mathrm { I A } ^ { 3 }$  (PEFT)</td><td>Raw source</td><td> $\mathrm { I A } ^ { 3 }$  scaling parameters</td><td>rameters. Prediction head and all base pa-</td></tr><tr><td>LoRA+PA</td><td>PA source</td><td> $\phi _ { m }$  and rank-8 LoRA parameters</td><td>rameters. All base backbone and interface</td></tr><tr><td>(PEFT+PA) PA+BitFit</td><td>PA source</td><td> $\phi _ { m }$  and selected bias parameters</td><td>parameters. All other backbone and interface</td></tr><tr><td>(PEFT+PA)</td><td></td><td></td><td>parameters.</td></tr><tr><td> $\mathrm { P A } { + } \mathrm { I A } ^ { 3 }$  (PEFT+PA)</td><td>PA source</td><td> $\phi _ { m }$  and  $\mathrm { L A } ^ { 3 }$  scaling parameters</td><td>All base backbone and interface parameters.</td></tr></table>

Table 3: The ten target-adaptation regimes. FT+PA and FT denote full fine-tuning with and without the PA, and NPA denotes partial fine-tuning of the prediction head $\chi _ { m }$ without the PA. In the PEFT regimes without the PA, the prediction head stays frozen, so only the PEFT parameters are updated. PEFT parameters are injected after the source checkpoint is loaded and attach to architecture compatible modules, so their locations and counts differ across backbones.

![](images/48d3ee1d153c397c85e34170ce995c64473a79f530bfbac060023935630d3398.jpg)  
Figure 9: Two-stage training and adaptation protocol. Source checkpoints are selected with sourcedomain validation only. Target support examples may update only the state permitted by the selected regime, and target-test examples are held out until final evaluation; at $K = { \bar { 0 } }$ , both support-statistics recalibration and gradient adaptation are skipped.

Because the PA realization differs across families in BatchNorm affine state, convolutional or MLP heads, and the ControlNet stem, its trainable parameter count is measured separately for each backbone.

## C ADAPTATION AND PEFT DETAILS

Table 3 lists the ten target-adaptation regimes of Sec. 4.2, and Figure 9 summarizes the two-stage protocol.

PEFT formulations. LoRA (Hu et al., 2021) replaces a selected frozen linear or channel-mixing transformation $\mathbf { W } _ { 0 }$ by

$$
\mathbf { W } \mathbf { x } = \mathbf { W } _ { 0 } \mathbf { x } + \frac { \alpha } { r } \mathbf { B } \mathbf { A } \mathbf { x } , \qquad r = 8 , \quad \alpha = 1 6 ,\tag{17}
$$

with A Kaiming-initialized and $\mathbf { B } = 0 ,$ , so the injected update starts at zero. $\mathrm { I A ^ { 3 } }$ (Liu et al., 2022a) learns multiplicative activation scales $\mathbf { y } = \mathbf { s } _ { \mathrm { I A } } \odot ( \mathbf { W } _ { 0 } \mathbf { x } )$ , initialized at 1. BitFit (Zaken et al., 2022) updates selected bias parameters and keeps all other tensors fixed.

Insertion sites. PEFT modules are placed according to each backbone’s structure. They target attention and feed-forward projections in attention-based backbones, channel-mixing layers in convolutional models, and the SS2D input and output projections in VM-UNet, where the selective-scan recurrence tensors are not LoRA or $\mathrm { I A ^ { 3 } }$ targets. In graph, operator, and ControlNet modules, they target compatible linear layers. In the graph and operator implementations, PEFT injection runs recursively over all compatible linear layers and can therefore also place PEFT parameters inside node-wise head modules. These parameters still belong to $\xi _ { m , a } ,$ , and the base parameters still follow Table 3. In the PEFT+PA regimes, $\phi _ { m }$ and $\xi _ { m , a }$ are optimized jointly. The trainable-parameter fraction of regime a is

$$
\rho _ { a } ^ { ( m ) } = \frac { | \Theta _ { a } ^ { ( m ) } | } { | \Theta _ { \mathrm { t o t a l } } ^ { ( m ) } | } \times 1 0 0 \% ,\tag{18}
$$

which describes the size of the optimization problem only and says nothing about performance by itself.

Freezing. A frozen component is frozen in both its parameters and its internal state. Setting requires grad=False is not enough for modules with BatchNorm, Dropout, stochastic depth, or other stateful operations. During adaptation with a frozen backbone, backbone normalization layers stay in inference mode and frozen Dropout, DropPath, and other stochastic modules are disabled. The implementation re-applies these settings after every change of training mode and checks that frozen running statistics stay unchanged. The two full fine-tuning regimes are the only exception, and there the trainable backbone updates normally.

Adapter BatchNorm recalibration. For every regime with a PA and $K > 0$ , adaptation begins with a gradient-free recalibration of the adapter BatchNorm of Eq. (2). The whole model is set to inference mode, only the adapter BatchNorm is switched to training mode, and over three complete passes of the support set its running statistics are updated as

$$
\widehat { \mu } _ { A } \gets \left( 1 - \mu _ { \mathrm { B N } } \right) \widehat { \mu } _ { A } + \mu _ { \mathrm { B N } } \mu _ { B } , \qquad \widehat { \sigma } _ { A } ^ { 2 } \gets \left( 1 - \mu _ { \mathrm { B N } } \right) \widehat { \sigma } _ { A } ^ { 2 } + \mu _ { \mathrm { B N } } \sigma _ { B } ^ { 2 } , \qquad \mu _ { \mathrm { B N } } = 0 . 1 ,\tag{19}
$$

with no optimizer step or gradient. The mini-batch size for this step depends on the architecture. It is four support graphs for GAT, GCN, MGN, and Transolver++, two samples for CASPIAN and ConvNeXt ${ \dot { \mathrm { V } } } 2 ,$ and one sample for MaxViT, Swin V2, VM-UNet, Depth Anything V2, Depth Pro, and ControlNet. For ControlNet, the base map used here is generated from support conditioning only. Neither support labels nor target test samples are used to estimate the adapter statistics. When $\phi _ { m }$ stays trainable during the supervised support optimization that follows, its BatchNorm keeps updating from the same support data, and any affine BatchNorm parameters receive gradients as part of $\phi _ { m }$ . All frozen backbone statistics remain fixed. Regimes without a PA skip this step.

Fixed support budget. After recalibration, the parameters selected in Eq. (8) are optimized on $S _ { K } ^ { t }$ only. The eleven non-diffusion backbones run 50 complete passes over the support set, with one optimizer update per support mini-batch in each pass. The value 50 is therefore a fixed number of passes, which gives more than 50 updates when K exceeds the support batch size. ControlNet instead runs exactly 50 optimizer steps while cycling through its support loader with its configured gradient accumulation. The optimizer, learning rate, weight decay, and other settings are taken from source training, gradients are clipped to global norm 1, and no target validation, early stopping, or test-based checkpoint selection is used. Each combination of regime, K, seed, and support draw starts once from its source checkpoint, is adapted once, and is then evaluated on the target test set.

## D SCENARIO SPLITS AND SUPPORT CONSTRUCTION

Table 4 summarizes the scenario partitions and evaluation settings. Source training uses the training split, hyperparameter search, early stopping, and checkpoint selection use only the validation split, and the test split is used only for final evaluation. A PA source model and a raw source model are trained for every backbone, region, and seed.

Table 4: Scenario partitions and evaluation settings. For the regional datasets, the scenario column gives train/validation/test counts. For the SLR targets, it gives the support pool and held-out test, with no validation partition. For $K > 0 _ { ; }$ , eight support draws are evaluated per K, and $K = 0$ is evaluated once per seed.
<table><tr><td>Experiment</td><td>Source → target</td><td>Scenarios</td><td>K</td><td>Seeds</td></tr><tr><td>In-domain SF</td><td> $\overline { { \mathrm { S F }  \mathrm { S F } } }$ </td><td>285 (168/57/60)</td><td>一</td><td>3</td></tr><tr><td>In-domain AD</td><td> $\mathrm { A D }  \mathrm { A D }$ </td><td>142 (82/28/32)</td><td></td><td>3</td></tr><tr><td>Cross-region</td><td> $\mathrm { S F }  \mathrm { A D }$ </td><td>142 (82/28/32)</td><td>0,1,3,5,10</td><td>3</td></tr><tr><td>Cross-region</td><td> $\mathrm { A D }  \mathrm { S F }$ </td><td>285 (168/57/60)</td><td>0,1,3,5,10</td><td>3</td></tr><tr><td>Cross-SLR</td><td> $\mathrm { S F } _ { 1 . 0 }  \mathrm { S F } _ { 0 . 5 }$ </td><td>32 (26 pool / 6 test)</td><td>0,1,3,5,10</td><td>3</td></tr><tr><td>Cross-SLR</td><td> $\mathrm { S F } _ { 1 . 0 }  \mathrm { S F } _ { 1 . 5 }$ </td><td>32 (26 pool / 6 test)</td><td>0,1,3,5,10</td><td>3</td></tr></table>

Protection buckets and regional splits. Each scenario is identified by its OLU configuration s, with protection level $\begin{array} { r } { n _ { \mathrm { p r o t } } ( \mathbf { s } ) = \sum _ { k } s _ { k } } \end{array}$ . The all-unprotected and all-protected configurations form their own none and all buckets. The remaining scenarios are split by the quartiles of $n _ { \mathrm { p r o t } }$ into very low, low, mid, and $\operatorname { h i g h }$ , with boundaries $Q _ { 2 5 } = 7 , { \dot { Q } } _ { 5 0 } \stackrel { \cdot } { = } 8 , { \dot { Q } } _ { 7 5 } = 9$ for SF and $Q _ { 2 5 } = 4 , Q _ { 5 0 } = 8 , Q _ { 7 5 } = 1 1$ for AD. Scenarios with $n _ { \mathrm { p r o t } } \leq Q _ { 2 5 } , Q _ { 2 5 } < n _ { \mathrm { p r o t } } \leq Q _ { 5 0 }$ , and $Q _ { 5 0 } < n _ { \mathrm { p r o t } } \leq Q _ { 7 5 }$ fall into the first three buckets, and the rest into high. The buckets therefore follow the observed scenario distribution rather than equal protection intervals.

For a protection bucket b with $n _ { b }$ scenarios, the number of test scenarios is

$$
n _ { \mathrm { t e s t } } ( b ) = \left\{ \begin{array} { l l } { 1 , } & { n _ { b } = 1 , } \\ { \operatorname* { m i n } \{ n _ { b } - 1 , \operatorname* { m a x } \bigl ( 2 , \lceil 0 . 2 n _ { b } \rceil \bigr ) \} , } & { n _ { b } > 1 , } \end{array} \right.\tag{20}
$$

which keeps at least one non-test scenario in any bucket with more than one sample. The singlescenario none and all buckets both go to test. This gives 60 SF and 32 AD test scenarios. With the test sets fixed, the remaining scenarios are split again by the same buckets, aiming for a validation share of about 20% of the full regional dataset. The result is 168/57/60 train/validation/test scenarios for SF and 82/28/32 for AD, or roughly $6 0 / 2 0 / 2 0$ after rounding within buckets. The regional manifest is generated once with seed 42 and shared by all twelve backbones and all training seeds, with no architecture-specific resampling, so differences between models cannot come from different partitions.

Cross-region support draws. Cross-region support sets are drawn from the target non-test pool, which is the union of the target train and validation partitions, so that ${ S _ { K } ^ { t } \subseteq \bar { \mathcal { D } } _ { \mathrm { t r a i n } } ^ { t } \cup \mathcal { D } _ { \mathrm { v a l } } ^ { t } }$ and $S _ { 0 } ^ { t } = \varnothing$ . Eight reproducible draws are made for each $\mathbf { \bar { \Gamma } } K \in \{ 0 , 1 , 3 , 5 , 1 0 , \overset { \cdot } { 2 0 } \}$ . Draws are stratified by protection bucket, with bucket shares roughly matching the non-test pool, and within each $K .$ scenarios used less often in earlier draws are preferred to limit repetition. The target test set is the same for all K, so differences across K reflect the amount of support rather than the test scenarios. In both transfer directions, the source model is trained on the source training split and selected on the source validation split.

SLR target manifests. The SLR targets use their own fixed construction instead of the 285- scenario SF split. The 0.5 m and 1.5 m SF targets each contain 32 configurations, namely the allunprotected and all-protected anchors and the 30 single-OLU configurations. The two anchors are never used for testing. Six single-OLU configurations, with OLU indices spread evenly over the index range, are held out as the test set (seed 42), and the remaining 24 single-OLU configurations and the two anchors form a 26-scenario support pool. No validation partition is defined. The pool is stored under the train field of the transfer manifest for loader compatibility, but it serves only as the support pool. SLR adaptation uses $K \in \{ 0 , 1 , 3 , 5 , 1 0 \}$ with eight draws per K. All $K = \bar { 0 }$ draws are empty. At K = 1, two draws use the all-unprotected anchor, two use the all-protected anchor, and four use distinct single-OLU configurations. For $K \geq 3$ , both anchors are always included and the remaining K − 2 positions are distinct single-OLU configurations. All K are evaluated on the same six test scenarios, and no test scenario appears in any support set.

## E TRAINING AND HPO DETAILS

Hyperparameter optimization (HPO) searches only optimization settings, namely a log-scaled learning rate in $[ 1 0 ^ { - 5 } , \dot { 5 } \times 1 0 ^ { - 3 } ]$ , the optimizer (Adam, AdamW, or RMSprop), and a feasible batch size. Architectural settings are never searched.

HPO runs 25 Optuna trials of at most 80 epochs per study and minimizes masked validation MSE. Batch-size candidates are set separately for each architecture according to what fits in memory, so no batch-size set is reused across backbones. The selected hyperparameters, optimizer settings, and batch sizes are kept fixed after HPO for all subsequent training and adaptation runs.

Source training uses ReduceLROnPlateau on validation loss with factor 0.5, patience of 5 epochs, and minimum learning rate $1 0 ^ { - 6 }$ . Gradients are clipped to global norm 1 in both source training and target adaptation. Weight decay is applied through the optimizer and is not part of Eq. (6). Final source training runs for at most 400 epochs for all backbones, with early stopping after 20 epochs without improvement in validation MSE. Early stopping usually ends training well before this limit, and the checkpoint with the lowest validation MSE is kept for evaluation and transfer.

The random state used to generate the manifest is separate from the training seeds {0, 1, 2}. Densegrid models use seed-controlled spatial flips as training augmentation, with validation and test sam ples left unaugmented. Augmentation for the other families is architecture-specific.

## F EVALUATION AND COMPLEXITY DETAILS

Metric definitions. All models are scored at the same retained physical locations. Grid predictions are first mapped back to these locations through the stored geographic mapping, while graph models already predict on them. For a held-out scenario with N retained locations, true PWL $y _ { i }$ , predicted PWL ybi, and mean true PWL y¯, the metrics are

$$
\mathrm { M A E } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigl | y _ { i } - \widehat { y } _ { i } \bigr | ,\tag{21}
$$

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( y _ { i } - \widehat { y } _ { i } \right) ^ { 2 } } ,\tag{22}
$$

$$
R ^ { 2 } = 1 - \frac { \sum _ { i = 1 } ^ { N } \left( y _ { i } - \widehat { y } _ { i } \right) ^ { 2 } } { \sum _ { i = 1 } ^ { N } \left( y _ { i } - \bar { y } \right) ^ { 2 } } ,\tag{23}
$$

$$
\mathrm { R T A E } = 1 0 0 \frac { \sum _ { i = 1 } ^ { N } \left| y _ { i } - \widehat { y } _ { i } \right| } { \sum _ { i = 1 } ^ { N } \left| y _ { i } \right| } ,\tag{24}
$$

$$
\delta _ { \epsilon } = \frac { 1 0 0 } { N } \sum _ { i = 1 } ^ { N } \mathbb { k } ^ { \epsilon } \big [ \big | y _ { i } - \widehat { y } _ { i } \big | > \epsilon \big ] , \qquad \epsilon \in \{ 0 . 1 , 0 . 5 \} \mathrm { m } ,\tag{25}
$$

$$
\operatorname { A c c } _ { 0 } = 1 0 0 { \frac { \sum _ { i = 1 } ^ { N } { \mathcal { H } } { \big [ } y _ { i } = 0 \wedge { \widehat { y } } _ { i } = 0 { \big ] } } { \sum _ { i = 1 } ^ { N } { \mathcal { H } } { \big [ } y _ { i } = 0 { \big ] } } } .\tag{26}
$$

Lower values are better for MAE, RMSE, RTAE, $\delta _ { 0 . 1 }$ , and $\delta _ { 0 . 5 } .$ , and higher values are better for $R ^ { 2 }$ and $\operatorname { A c c } _ { 0 }$ . These metrics are used for reporting only, and the only quantity used for model selection is validation MSE.

Aggregation. Metrics are computed separately for each held-out scenario. For in-domain evaluation, scenario metrics are averaged within each seed, and we report the mean and sample standard deviation of the three seed-level means. For few-shot transfer, scenario metrics are computed separately for every support draw, averaged over all scenario and draw pairs within a seed so that each draw and scenario has equal weight, and then summarized per K as the mean and sample standard deviation over the three seeds. SLR transfer uses the same procedure. At $K = 0$ , the single emptysupport evaluation per seed replaces the repeated draws. No confidence intervals are reported, and no target test data are used for selection.

![](images/699ec4576e6211f7ac1ea1a46e2ec374a149e9c5fbb853bbf0fbfffcf71e5f81.jpg)  
Figure 10: Mean transfer RMSE against the trainable parameter fraction of each adaptation regime. RMSE is averaged over the twelve backbones, the four transfer settings, and $K \in \mathsf { \bar { \{ 0 , 1 , 3 , 5 , 1 0 \} } }$ as in Table 8, and each marker sits at the median trainable fraction over the backbones. Filled blue markers include the PA and open orange markers do not, while the marker shape gives the regime type. Each arrow joins a regime to its matched version with the PA. The horizontal axis is logarithmic.

Complexity profiling. For every backbone and regime, we record the total and trainable parameter counts, the frozen count as their difference, and the trainable fraction of Eq. (18). These counts are exact and are read from the instantiated models. We do not compare FLOPs, MACs, or wall-clock times, since they could not be traced reliably for every architecture, and the runs used different GPUs (H100 or H200) and software environments. Figure 10 plots the mean transfer RMSE of each regime against its trainable fraction. Adding the PA lowers the RMSE of every matched regime, by 17.9% for full fine-tuning, 17.8% for IA<sup>3</sup>, 16.5% for BitFit, 14.3% for LoRA, and 10.4% for head-only adaptation. How many parameters this costs depends on the size of the backbone. In the dense-grid backbones the PA has 294 or 438 parameters, which is at most 0.08% of the model and 50 to 160 times fewer than the raw head trained by NPA. The graph and operator backbones are much smaller, with about 18,000 to 131,000 parameters, so their PA of 4,358 parameters makes up 3.3% to 23.8% of the model. Across all backbones, adding the PA to a PEFT method raises the median trainable fraction by only 0.0033 percentage points, so the arrows in Figure 10 are almost vertical. PA-only adaptation, which trains a median of 0.0012% of the parameters, also reaches a lower mean RMSE than full fine-tuning without the PA (0.406 m against 0.420 m).

## G ADDITIONAL RESULTS

## G.1 IN-DOMAIN RESULTS

Tables 5–7 give all seven metrics of Appendix F for in-domain prediction, averaged over both regions and for each region separately. Values are the mean and standard deviation over three seeds. For each backbone, the Source column shows which source model (PA or raw) has the lower combined RMSE, and the same source model is reported in all three tables. Bold marks the best value in each column. VM-UNet is best on every metric in both regions. The dense backbones have lower MAE, RMSE, and error exceedance rates than the graph and operator backbones throughout,

Table 5: In-domain results averaged over SF and AD.
<table><tr><td>Backbone</td><td>Source</td><td>MAE (m)</td><td>RMSE (m)</td><td>R2</td><td> $\overline { { \mathbf { A c c _ { 0 } } \left( \% \right) } }$ </td><td>RTAE (%)</td><td> $\overline { { \delta _ { 0 , 5 } \ : ( \% ) } }$ </td><td> $\overline { { \delta _ { \mathbf { 0 } , \mathbf { 1 } } \left( \% \right) } }$ </td></tr><tr><td>VM-UNet</td><td>PA</td><td>0.0026 ± 0.0001</td><td>0.0542 ± 0.0015 0.9660 ± 0.0008 99.9364 ± 0.0046</td><td></td><td></td><td> $\overline { { 3 . 5 3 1 0 \pm 0 . 0 8 0 6 } }$ </td><td>0.0648 ± 0.0035</td><td> $\overline { { { \bf 0 . 3 0 4 4 } } } \pm 0 . 0 2 3 0$ </td></tr><tr><td>Swin V2</td><td>PA</td><td> $0 . 0 0 3 7 \pm 0 . 0 0 0 2$ </td><td>0.0630 ± 0.0025 0.9582 ± 0.0055</td><td></td><td> $9 9 . 8 0 5 5 \pm 0 . 0 2 6 7$ </td><td> $4 . 8 0 0 3 \pm 0 . 2 2 6 1$ </td><td>0.1000 ± 0.0115</td><td>0.6650 ± 0.0454</td></tr><tr><td>MaxViT</td><td>PA</td><td></td><td>0.0038 ± 0.0003 0.0639 ± 0.0033 0.9564 ± 0.0039</td><td></td><td> $9 9 . 7 9 5 3 \pm 0 . 0 4 2 7$ </td><td> $4 . 8 3 2 1 \pm 0 . 6 1 7 6$ </td><td>0.0980 ± 0.0113</td><td> $0 . 6 6 9 7 \pm 0 . 0 6 8 0$ </td></tr><tr><td>CASPIAN</td><td>PA</td><td> $0 . 0 0 4 1 \pm 0 . 0 0 0 3$ </td><td>0.0686 ± 0.0036 0.9560 ± 0.0060</td><td></td><td> $9 9 . 8 1 7 8 \pm 0 . 0 0 7 7$ </td><td> $5 . 5 5 9 7 \pm 0 . 4 4 7 8$ </td><td> $0 . 1 2 6 6 \pm 0 . 0 1 6 1$ </td><td> $0 . 7 1 0 7 \pm 0 . 0 3 7 6$ </td></tr><tr><td>Depth Pro</td><td>PA</td><td></td><td>0.0046 ± 0.0007 0.0713 ± 0.0046 0.9522 ± 0.0005</td><td></td><td> $9 9 . 7 1 7 2 \pm 0 . 0 2 1 5$ </td><td> $6 . 2 1 0 1 \pm 1 . 0 6 3 3$ </td><td>0.1425 ± 0.0363</td><td> $0 . 7 7 0 8 \pm 0 . 1 1 7 3$ </td></tr><tr><td>Depth Anything V2</td><td>PA</td><td></td><td>0.0049 ± 0.0001 0.0747 ± 0.0017 0.9503 ± 0.0052</td><td></td><td>99.6402 ± 0.0806</td><td> $6 . 7 1 2 8 \pm 0 . 1 1 6 1$ </td><td>0.1561 ± 0.0092</td><td>0.8686 ± 0.0269</td></tr><tr><td>ControlNet</td><td>PA</td><td></td><td>0.0054 ± 0.0004 0.0754 ± 0.0030 0.9244 ± 0.0096</td><td></td><td> $9 8 . 0 4 5 5 \pm 0 . 1 4 6 1$ </td><td> $1 5 . 4 6 3 4 \pm 7 . 0 6 4 6$ </td><td>0.1618 ± 0.0124</td><td>1.0396 ± 0.1821</td></tr><tr><td>ConvNeXt V2</td><td>PA</td><td></td><td>0.0072 ± 0.0006 0.0893 ± 0.0028 0.9217 ± 0.0157</td><td></td><td>99.4124 ± 0.2542</td><td> $2 6 . 5 8 6 3 \pm 1 3 . 7 3 6 2$ </td><td>0.2307 ± 0.0139</td><td>1.4426 ± 0.3020</td></tr><tr><td>MGN</td><td>PA</td><td></td><td>0.0600 ± 0.0016 0.2990 ± 0.0034 0.9266 ± 0.0011</td><td></td><td> $\overline { { 9 4 . 5 5 7 8 \pm 0 . 4 7 1 1 } }$ </td><td> $\overline { { 9 . 0 6 8 3 \pm 0 . 3 1 7 8 } }$ </td><td>2.0276 ± 0.1528</td><td>8.3739 ± 0.2469</td></tr><tr><td>GAT</td><td>Raw</td><td></td><td>0.0617 ± 0.0021 0.3106 ± 0.0019 0.9235 ± 0.0007</td><td></td><td> $9 6 . 7 8 4 6 \pm 0 . 6 6 5 2$ </td><td> $8 . 7 7 7 4 \pm 0 . 5 9 3 4$ </td><td>1.9891 ± 0.0708</td><td>8.8130 ± 0.5266</td></tr><tr><td>Transolver++</td><td>Raw</td><td></td><td>0.0649 ± 0.0078 0.3417 ± 0.0052 0.9215 ± 0.0079</td><td></td><td> $9 6 . 5 7 5 8 \pm 4 . 0 5 2 8$ </td><td> $9 . 4 8 5 7 \pm 1 . 4 2 8 6$ </td><td>1.7225 ± 0.3828</td><td>8.8934 ± 2.8365</td></tr><tr><td>GCN</td><td>PA</td><td></td><td>0.0737 ± 0.0025 0.3418 ± 0.0020 0.9121 ± 0.0009</td><td></td><td> $9 3 . 5 1 6 6 \pm 0 . 7 7 2 0$ </td><td> $1 1 . 8 1 1 8 \pm 0 . 6 4 5 6$ </td><td>1.6124 ± 0.0473</td><td>13.2854 ± 1.1726</td></tr></table>

Table 6: In-domain results for Abu Dhabi.
<table><tr><td>Backbone</td><td>Source</td><td>MAE (m)</td><td>RMSE (m)</td><td> $\overline { { R ^ { 2 } } }$ </td><td> $\overline { { \mathbf { A c c _ { 0 } } \left( \% \right) } }$ </td><td> $\overline { { \mathbf { R T A E } \left( \% \right) } }$ </td><td> $\overline { { \delta _ { \mathbf { 0 } , 5 } \ : ( \% ) } }$ </td><td> $\overline { { \delta _ { \mathbf { 0 } , \mathbf { 1 } } \left( \mathcal { \eta } _ { o } \right) } }$ </td></tr><tr><td>VM-UNet</td><td>PA</td><td></td><td>0.0039 ± 0.0002 0.0866 ± 0.0030 0.9469 ± 0.0015 99.9012 ± 0.0098</td><td></td><td></td><td>4.4593 ± 0.1712</td><td>0.1058 ± 0.0095</td><td>0.4604 ± 0.0384</td></tr><tr><td>Swin V2</td><td>PA</td><td></td><td></td><td></td><td>0.0044 ± 0.0002 0.0902 ± 0.0014 0.9450 ± 0.0013 99.7979 ± 0.0659</td><td> $5 . 5 1 4 1 \pm 0 . 0 7 0 2$ </td><td>0.1175 ± 0.0084</td><td>0.7135 ± 0.0074</td></tr><tr><td>MaxViT</td><td>PA</td><td></td><td></td><td></td><td>0.0048 ± 0.0004 0.0943 ± 0.0045 0.9431 ± 0.0034 99.7823 ± 0.0865</td><td> $5 . 8 2 2 1 \pm 0 . 6 1 7 4$ </td><td>0.1261 ± 0.0105</td><td>0.7930 ± 0.0977</td></tr><tr><td>CASPIAN</td><td>PA</td><td></td><td></td><td></td><td>0.0055 ± 0.0007 0.0984 ± 0.0061 0.9412 ± 0.0034 99.7413 ± 0.0271</td><td>7.0318 ± 0.9230</td><td>0.1648 ± 0.0330</td><td>0.9159 ± 0.1249</td></tr><tr><td>Depth Pro</td><td>PA</td><td>0.0064 ± 0.0015 0.1099 ± 0.0106 0.9328 ± 0.0103 99.7026 ± 0.0619</td><td></td><td></td><td></td><td> $8 . 4 5 0 6 \pm 2 . 3 2 5 3$ </td><td>0.2215 ± 0.0785</td><td>0.9217 ± 0.2590</td></tr><tr><td>Depth Anything V2</td><td>PA</td><td>0.0069 ± 0.0005 0.1143 ± 0.0055 0.9293 ± 0.0036 99.5482 ± 0.1792</td><td></td><td></td><td></td><td> $9 . 3 7 1 2 \pm 0 . 1 6 1 3$ </td><td>0.2330 ± 0.0247</td><td>1.0656 ± 0.0793</td></tr><tr><td>ControlNet</td><td>PA</td><td>0.0061 ± 0.0007 0.1025 ± 0.0059 0.9148 ± 0.0150 98.0762 ± 0.2425</td><td></td><td></td><td></td><td> $1 4 . 4 7 6 1 \pm 1 1 . 2 4 1 4$ </td><td>0.1907 ± 0.0175</td><td>1.0372 ± 0.1651</td></tr><tr><td>ConvNeXt V2</td><td>PA</td><td></td><td></td><td></td><td>0.0076 ± 0.0010 0.1140 ± 0.0074 0.9155 ± 0.0271 99.4878 ± 0.4062</td><td> $2 3 . 5 9 0 2 \pm 2 2 . 2 2 2 7$ </td><td>0.2613 ± 0.0251</td><td>1.3143 ± 0.2378</td></tr><tr><td>MGN</td><td>PA</td><td></td><td></td><td></td><td>0.0664 ± 0.0023 0.3734 ± 0.0068 0.9134 ± 0.0019 95.0695 ± 0.3636</td><td> $\overline { { 1 0 . 2 9 4 2 \pm 0 . 3 9 3 9 } }$ </td><td>2.2620 ± 0.0900</td><td>8.1504 ± 0.3404</td></tr><tr><td>GAT</td><td>Raw</td><td></td><td></td><td></td><td>0.0704 ± 0.0021 0.3881 ± 0.0029 0.9106 ± 0.0013 96.4951 ± 0.8523</td><td> $1 0 . 3 4 9 1 \pm 0 . 8 9 2 7$ </td><td>2.5371 ± 0.0782</td><td>8.5326 ± 0.7478</td></tr><tr><td>Transolver++</td><td>Raw</td><td></td><td>0.0834 ± 0.0163 0.4557 ± 0.0072 0.8960 ± 0.0141</td><td></td><td> $9 3 . 9 0 6 5 \pm 8 . 0 9 1 5$ </td><td> $1 2 . 4 0 6 1 \pm 2 . 5 5 0 7$ </td><td>2.4486 ± 0.7725</td><td>9.9514 ± 5.7027</td></tr><tr><td>GCN</td><td>PA</td><td></td><td>0.0802 ± 0.0008 0.4283 ± 0.0027 0.8959 ± 0.0011</td><td></td><td> $9 4 . 1 3 1 5 \pm 0 . 5 8 1 5$ </td><td> $1 2 . 9 0 1 4 \pm 0 . 3 2 2 3$ </td><td>2.0481 ± 0.0692</td><td> $1 1 . 5 2 0 5 \pm 0 . 4 5 0 3$ </td></tr></table>

but ControlNet and ConvNeXt V2 have a high and variable RTAE. This is likely because RTAE becomes unstable for scenarios with little flooding, where its denominator is small.

## G.2 PER-BACKBONE TRANSFER RESULTS

Figures 11 and 12 show, for each backbone, the RMSE curve of its best regime over all four transfer settings, first over all ten regimes and then with the two full fine-tuning regimes excluded. The best regime is the one with the lowest RMSE averaged over $K \in \{ 0 , 1 , 3 , { \bar { 5 } } , 1 { \bar { 0 } } \}$ , with equal weight for each K. In Figure 11, FT+PA is best for every backbone except Transolver++, where LoRA+PA is best. The dense and depth-foundation backbones reach RMSE close to 0.1 m by $K = 1 0$ , while the graph and operator backbones start from a much higher zero-shot error and level off near 0.3 m. When full fine-tuning is excluded (Figure 12), nine of the twelve backbones still select a PA regime. Table 8 breaks the regime comparison down by transfer setting.

## G.3 QUALITATIVE ERROR MAPS

Figure 13 shows the VM-UNet error maps for the scenario and regimes of Figure 3, computed as prediction minus ground truth at each location, so red marks overestimated PWL and blue marks underestimated PWL. Locations that are dry in both the prediction and the ground truth are shown in beige. In AD, the errors of FT and LoRA are spread over a large inland area, which matches the false flooding seen in Figure 3. The PA regimes keep the errors close to the coast, where the true flooding occurs. In SF, the in-domain errors stay within ±0.1 m. After transfer, the largest errors appear in the northern basin, where FT and LoRA underestimate PWL, and in a few small areas along the southern shoreline.

Best configuration excl. FT and FT+PA · All transfers (combined)  
Table 7: In-domain results for San Francisco.
<table><tr><td>Backbone</td><td>Source</td><td>MAE (m)</td><td>RMSE (m)</td><td>R2</td><td> $\overline { { \mathbf { A c c o } \left( \% \right) } }$ </td><td>RTAE (%)</td><td> $\overline { { \delta _ { \mathbf { 0 . 5 } } \left( \% \right) } }$ </td><td> $\overline { { \delta _ { \mathbf { 0 . 1 } } \left( \% \right) } }$ </td></tr><tr><td>VM-UNet</td><td>PA</td><td>0.0013 ± 0.0000 0.0217 ± 0.0008 0.9851 ± 0.0002</td><td></td><td></td><td> $\overline { { 9 9 . 9 7 1 5 \pm 0 . 0 0 2 8 } }$ </td><td> $\overline { { 2 . 6 0 2 6 \pm 0 . 0 2 2 1 } }$ </td><td>0.0239 ± 0.0024</td><td> $\overline { { \mathbf { 0 . 1 4 8 4 } \pm 0 . 0 1 3 3 } }$ </td></tr><tr><td>Swin V2</td><td>PA</td><td> $0 . 0 0 3 0 \pm 0 . 0 0 0 5$ </td><td>0.0357 ± 0.0050 0.9714 ± 0.0106</td><td></td><td> $9 9 . 8 1 3 0 \pm 0 . 0 1 9 3$ </td><td> $4 . 0 8 6 4 \pm 0 . 3 8 3 9$ </td><td>0.0824 ± 0.0268</td><td>0.6164 ± 0.0879</td></tr><tr><td>MaxViT</td><td>PA</td><td>0.0027 ± 0.0002 0.0335 ± 0.0021 0.9697 ± 0.0044</td><td></td><td></td><td> $9 9 . 8 0 8 3 \pm 0 . 0 1 3 3$ </td><td> $3 . 8 4 2 1 \pm 0 . 6 2 1 0$ </td><td> $0 . 0 6 9 8 \pm 0 . 0 1 2 6$ </td><td>0.5464 ± 0.0425</td></tr><tr><td>CASPIAN</td><td>PA</td><td> $0 . 0 0 2 8 \pm 0 . 0 0 0 1$ </td><td>0.0387 ± 0.0012 0.9708 ± 0.0091</td><td></td><td> $9 9 . 8 9 4 3 \pm 0 . 0 2 9 7$ </td><td> $4 . 0 8 7 5 \pm 0 . 0 6 1 1$ </td><td> $0 . 0 8 8 5 \pm 0 . 0 0 1 0$ </td><td> $0 . 5 0 5 5 \pm 0 . 0 5 6 9$ </td></tr><tr><td>Depth Pro</td><td>PA</td><td> $0 . 0 0 2 8 \pm 0 . 0 0 0 2 ~ 0 . 0 3 2 7 \pm 0 . 0 0 2 1 ~ 0 . 9 7 1 6 \pm 0 . 0 1 0 1$ </td><td></td><td></td><td> $9 9 . 7 3 1 9 \pm 0 . 0 1 9 2$ </td><td> $3 . 9 6 9 6 \pm 0 . 2 8 4 3$ </td><td> $0 . 0 6 3 6 \pm 0 . 0 0 8 8$ </td><td> $0 . 6 1 9 9 \pm 0 . 0 3 4 0$ </td></tr><tr><td>Depth Anything V2</td><td>PA</td><td> $0 . 0 0 3 0 \pm 0 . 0 0 0 3$ </td><td>0.0352 ± 0.0022 0.9712 ± 0.0097</td><td></td><td> $9 9 . 7 3 2 3 \pm 0 . 0 2 8 1$ </td><td> $4 . 0 5 4 4 \pm 0 . 2 7 5 6$ </td><td> $0 . 0 7 9 3 \pm 0 . 0 1 1 4$ </td><td> $0 . 6 7 1 6 \pm 0 . 1 0 3 1$ </td></tr><tr><td>ControlNet</td><td>PA</td><td>0.0047 ± 0.0006 0.0483 ± 0.0025 0.9340 ± 0.0079</td><td></td><td></td><td> $9 8 . 0 1 4 8 \pm 0 . 0 7 9 4$ </td><td> $1 6 . 4 5 0 8 \pm 5 . 3 7 2 2$ </td><td> $0 . 1 3 2 8 \pm 0 . 0 0 7 7$ </td><td> $1 . 0 4 2 0 \pm 0 . 2 9 5 2$ </td></tr><tr><td> $\mathrm { C o n v N e X t } \ : \mathrm { V } 2$ </td><td>PA</td><td>0.0069 ± 0.0010 0.0646 ± 0.0029 0.9279 ± 0.0116</td><td></td><td></td><td> $9 9 . 3 3 7 0 \pm 0 . 1 4 7 9$ </td><td> $2 9 . 5 8 2 4 \pm 1 0 . 2 9 4 2$ </td><td> $0 . 2 0 0 1 \pm 0 . 0 0 2 9$ </td><td> $1 . 5 7 0 8 \pm 0 . 5 5 7 4$ </td></tr><tr><td>MGN</td><td>PA</td><td>0.0535 ± 0.0010 0.2247 ± 0.0015 0.9398 ± 0.0008</td><td></td><td></td><td> $\overline { { 9 4 . 0 4 6 2 \pm 0 . 5 8 0 2 } }$ </td><td> $\overline { { 7 . 8 4 2 4 \pm 0 . 4 2 4 3 } }$ </td><td>1.7932 ± 0.2360</td><td>8.5974 ± 0.1784</td></tr><tr><td>GAT</td><td>Raw</td><td> $0 . 0 5 3 0 \pm 0 . 0 0 2 2$ </td><td>0.2330 ± 0.0009 0.9364 ± 0.0009</td><td></td><td> $9 7 . 0 7 4 1 \pm 0 . 6 4 9 1$ </td><td> $7 . 2 0 5 6 \pm 0 . 6 2 8 1$ </td><td>1.4410 ± 0.1047</td><td>9.0934 ± 0.3633</td></tr><tr><td>Transolver++</td><td>Raw</td><td>0.0463 ± 0.0022 0.2277 ± 0.0036 0.9469 ± 0.0030</td><td></td><td></td><td> $9 9 . 2 4 5 1 \pm 0 . 0 1 4 1$ </td><td> $6 . 5 6 5 3 \pm 0 . 3 1 0 7$ </td><td>0.9964 ± 0.0152</td><td> $7 . 8 3 5 4 \pm 0 . 4 3 9 4$ </td></tr><tr><td>GCN</td><td>PA</td><td> $0 . 0 6 7 1 \pm 0 . 0 0 4 2$ </td><td>0.2552 ± 0.0012 0.9283 ± 0.0007</td><td></td><td> $9 2 . 9 0 1 6 \pm 0 . 9 6 2 5$ </td><td> $1 0 . 7 2 2 2 \pm 0 . 9 6 8 8$ </td><td>1.1767 ± 0.0254</td><td> $1 5 . 0 5 0 3 \pm 1 . 8 9 4 8$ </td></tr></table>

![](images/a8cb277c95be53045e32c194084d714865e4e893c9b97a7380394bb4a5255d65.jpg)  
Figure 11: Best regime per backbone over all transfer settings, selected by mean RMSE over K. The vertical axis is logarithmic.

![](images/ee1a5fdcad0283725197d845cdd2b5577e4500eb48c29d7ebe6a4e3a3b0596e0.jpg)  
Figure 12: Best regime per backbone when FT and FT+PA are excluded. Solid lines are regimes with the PA and dotted lines are regimes without it. The vertical axis is logarithmic.

Table 8: RMSE (m) of each regime averaged over all twelve backbones and over $K \in$ {0, 1, 3, 5, 10}, for each transfer setting. PEFT rows give the mean of LoRA, $\mathrm { L A ^ { 3 } }$ and BitFit, with the individual methods listed beneath. Bold marks the best regime in each column.
<table><tr><td>Regime</td><td> $\mathbf { S F } { \xrightarrow { } } \mathbf { A D }$ </td><td> $\mathbf { A D } { \xrightarrow { } } \mathbf { S F }$  _</td><td> $\overline { { \mathbf { S } \mathbf { F } _ { 1 . 0 } \to \mathbf { S } \mathbf { F } _ { 0 . 5 } } }$ </td><td> $\overline { { \mathbf { S } \mathbf { F } _ { 1 . 0 } \to \mathbf { S } \mathbf { F } _ { 1 . 5 } } }$ </td></tr><tr><td>FT+PA</td><td>0.4338</td><td>0.5401</td><td>0.1668</td><td>0.2403</td></tr><tr><td>PEFT+PA (mean)</td><td>0.4989</td><td>0.5992</td><td>0.1819</td><td>0.2727</td></tr><tr><td> $\mathrm { L o R A + P A }$ </td><td>0.5233</td><td>0.5653</td><td>0.1793</td><td>0.2629</td></tr><tr><td> $\mathrm { I A ^ { 3 } { + } P A }$ </td><td>0.4847</td><td>0.6170</td><td>0.1818</td><td>0.2785</td></tr><tr><td>BitFit+PA</td><td>0.4887</td><td>0.6152</td><td>0.1847</td><td>0.2766</td></tr><tr><td>PA</td><td>0.5093</td><td>0.6198</td><td>0.1969</td><td>0.2998</td></tr><tr><td> $\overline { { \mathrm { F T } } }$ </td><td>0.6106</td><td>0.6233</td><td>0.1786</td><td>0.2686</td></tr><tr><td> $P E F T ( m e a n )$ </td><td>0.6800</td><td>0.7061</td><td>0.1859</td><td>0.2817</td></tr><tr><td> $\operatorname { L o R A }$ </td><td>0.6254</td><td>0.7122</td><td>0.1781</td><td>0.2708</td></tr><tr><td> $\mathrm { { I A } ^ { 3 } }$ </td><td>0.6891</td><td>0.7286</td><td>0.1913</td><td>0.2905</td></tr><tr><td>BitFit</td><td>0.7256</td><td>0.6774</td><td>0.1882</td><td>0.2836</td></tr><tr><td>NPA</td><td>0.6902</td><td>0.6243</td><td>0.1973</td><td>0.3030</td></tr></table>

![](images/903af4bd32468d88d91fc694c3d2ea2be1ab8676cfc1c16fedc4fb709543320f.jpg)

![](images/6d71c7da1c56aba0c161961a179f9435763e218ffe245f4be1e47834e769441c.jpg)

![](images/1a75c42d788575a17b6cf38297998b7fec13487f80726ae0e0b989e780d33374.jpg)

![](images/4ade45c13c2e8516a6426ddb9acb6249a2558cc86d52ad2c2002af0f74fc73b8.jpg)

![](images/f3658041cbb77e87ec64fb678b8a2177964c8e5556a44f168dac47f0de9ae719.jpg)

![](images/623ff844336083d47417a773d93265b273d0d6459661dffbf1aa54122ded73f4.jpg)  
(a)

![](images/d3cbcd89fe9fd888f7f070a5daa56dde8fa4bc7623ae2fe3a7c29828c9095731.jpg)

![](images/2a8571cab7c2455bfc038f047781f58672667dac82facc7efada868add9eae08.jpg)

(b)  
![](images/ce82ae2772939f389151e43759b613928dcefbb97db7814f27d338f3dc89c0b3.jpg)  
(c)

![](images/0dc4f385919a14bc6d390b278e9d3029078a3cee541a4012ddd798148fb83435.jpg)  
(d)

![](images/30e6d1c3afcdecc25d9a82228778636aac27c381b2e61ff9649522ed09ce60ff.jpg)  
(e)

![](images/bcf23ba960069719f61353d770799eb83e336b5b5be2611712a5cc8b73757bae.jpg)  
(f)  
Figure 13: PWL error maps with VM-UNet (prediction minus ground truth, in m) for the scenario and regimes of Figure 3, AD in the top row and SF in the bottom row. (a) Ground-truth PWL for reference. (b) In-domain error. (c)–(f) Errors after transfer with K = 3 using (c) FT+PA, (d) FT, (e) best PEFT+PA, and (f) best PEFT. Beige marks locations that are dry in both the prediction and the ground truth. Color scales differ between panels.