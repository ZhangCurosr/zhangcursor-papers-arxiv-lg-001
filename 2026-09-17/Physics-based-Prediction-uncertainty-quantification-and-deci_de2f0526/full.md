# Physics-based Prediction, uncertainty quantification and decisionmaking for IN718 crystallographic texture intensity across LPBF defocus regimes

Authors:

Yisheng Lu<sup>1</sup>, John Riris<sup>2</sup>, Jie Song<sup>2</sup>, Yao Fu<sup>2,3</sup>, Jie Chen<sup>1,3,4</sup>

Affiliations:

<sup>1</sup>Department of Mechanical Engineering, Virginia Tech, Blacksburg, VA 24061, USA

<sup>2</sup>Department of Aerospace & Ocean Engineering, Virginia Tech, Blacksburg, VA 24061, USA

<sup>3</sup>VT Made, Virginia Tech, Blacksburg, VA 24061, USA

<sup>4</sup>Macromolecules Innovation Institute, Virginia Tech, Blacksburg, VA 24061, USA

## Abstract

Reliable prediction of crystallographic texture in laser powder bed fusion is critical for linking process conditions with anisotropic response and for qualification. However, black-box models may fail under shift and cannot distinguish weak data support from loss of physical validity. This study develops a two-stage physics-based model for $< 0 0 1 > \parallel \mathrm { B D }$ (build direction) texture in Inconel 718. Stage 1 maps process variables to melting mode and melt pool geometry. Stage 2 predicts texture by combining an empirical physics model with a random-forest residual model. A k-nearest-neighbor weight attenuates residual corrections for poorly supported queries, while a study-specific areal beam-power-density criterion withholds predictions outside the adopted conduction envelope. Conformal intervals are evaluated on the retained physics-valid set, and SHAP and Sobol analyses assess residual sensitivity. Under a controlled leave-one-defocus-out evaluation, the physics anchor achieved $\mathrm { R } ^ { 2 } = 0 . 7 7 8$ , against -0.001 for the black-box model and 0.750 for the gated hybrid. Under leave-one-group-out cross-

validation, the gated hybrid reached $\mathrm { R } ^ { 2 } = 0 . 5 9 2$ against 0.538 for the black-box model. Retained set coverage was 92.9% at a mean full width of 3.65 multiples of a uniform distribution (MUD) under grouped cross-validation and 100% at a width of 3.21 MUD under transfer to a withheld +80 mm defocus regime. An illustrative mapping produced a retained BD elastic-modulus span of 127-187 GPa. On nine conditions from a separately built sample set, the framework withheld three, attenuated three, and matched the measured ordering for the rest. Separating data applicability, physics validity, and predictive uncertainty into distinct decisions lets the framework transfer where an unconstrained model does not, and withhold predictions where no model class performs adequately.

## Keywords

Laser powder bed fusion · Inconel 718 · Crystallographic texture · Physics-based machine learning · Applicability domain · Conformal prediction

## 1. Introduction

Laser powder bed fusion (LPBF) enables the fabrication of geometrically complex metallic components through localized melting and layer wise consolidation. However, the steep and directionally varying thermal gradients generated during LPBF also produce heterogeneous solidification conditions and crystallographic textures [1,2]. In face-centered-cubic nickelbased alloys, elastic stiffness is strongly orientation dependent, with ⟨001⟩ more compliant than ⟨111⟩ [3–5]. During LPBF, competitive columnar growth along a predominantly build-aligned thermal gradient can preferentially align ⟨001⟩ with the build direction (BD). Quantifying this enrichment therefore provides a compact texture target for linking process conditions to direction-dependent material response [9-11]. Predicting this texture from controllable process variables is therefore important for connecting process design with the expected material response [6,7].

Crystallographic texture is governed by a chained process-to-melt-pool-to-solidification relationship [2,8]. Laser power (P), scan speed (V), hatch spacing (H), and focus offset (F) determine the local energy deposition and thermal history. These conditions influence meltpool geometry, overlap between adjacent tracks, repeated remelting, and the thermal-gradient orientation associated with competitive grain selection [8–11]. Shao et al. [11] demonstrated that hatch spacing and laser remelting reorganize the ⟨001⟩ texture of IN718 in LPBF, showing that texture formation is a history-dependent, multi-cycle process rather than a single-track outcome. Transitions between conduction- and keyhole-dominated melting further alter meltpool geometry and the associated solidification conditions [9,12–14]. From an engineering standpoint, texture intensity should not be evaluated independently of melt-pool stability. Keyhole-mode melting produces vapor depressions, and instability or collapse of this cavity can promote pore formation [12,14]. The present study therefore treats an adopted conductiondominated processing envelope as its intended deployment domain, rather than treating high texture intensity obtained under any melting state as equally desirable. Scipioni Bertoli et al. [15] showed that volumetric energy density (VED) alone cannot represent these coupled effects, because different process-parameter combinations producing the same nominal energy input generate different beam sizes and melt-pool states.

Data-driven models offer an efficient alternative to repeated experiments and high-fidelity thermal-fluid simulations. Machine-learning models have been applied to melt-pool geometry prediction and in situ or post-build defect classification in metal additive manufacturing [16– 20] and comparatively fewer studies have linked LPBF process conditions to crystallographic texture or developed machine-learning surrogates for texture evolution [21,22]. Tree-based and Gaussian-process formulations allow nonlinear process interactions to be learned from limited observations. However, predictive performance within randomly partitioned data does not establish transferability to unobserved process regimes: a black-box model can achieve favorable interpolation accuracy while producing unsupported corrections once a query lies beyond its training distribution. Physics-informed and hybrid formulations address this limitation by constraining the learned relationship or anchoring it to a physically interpretable trend [23], and a recent review [24] surveyed these strategies for process-structure-property modeling in additive manufacturing. Their behavior under controlled regime extrapolation, however, still requires explicit evaluation.

Furthermore, reliability assessment should also account for the structure of the validation data. Roberts et al. [25] showed that when data have a grouped structure, random sample-level splitting places the closely related conditions in both training and testing subsets, and yields optimistic performance estimates, so group-aware validation is required when the intended use involves prediction for unseen groups. Complementary to grouping, Sutton et al. [26] introduced applicability-domain analysis to assess whether a query is supported by the training distribution of a materials-science model, a notion later generalized to distance-based support measures [27]. For quantifying predictive uncertainty, Lei et al. [28] established distributionfree regression intervals under exchangeability, providing the statistical basis for conformal calibration. Subsequent work has shown that conformal validity under distribution shift remains conditional on the calibration design and exchangeability assumptions [29,30], with conditional and hierarchical guarantees requiring additional structure [31–33]. Therefore, a reliable workflow should distinguish residual-model applicability, physics-domain validity, and uncertainty calibration rather than collapsing them into a single confidence measure.

A remaining need is an integrated model that connects process conditions to texture while preserving these distinctions. The intermediate melt-pool state provides a physically meaningful bridge, but surrogate errors can propagate into texture prediction. In this study, a process-regime shift refers to a change in the focus-offset level and the associated surface beam radius rather than a change in melting mode. Under such a shift, the framework must determine whether the learned residual remains supported, whether the conduction-inspired anchor remains valid, and whether an uncertainty interval can be issued. These requirements favor separate mechanisms for applicability attenuation, physics-validity abstention, and retained-set uncertainty.

In this study, we develop a two-stage reliability-oriented framework for predicting ⟨001⟩ ∥ BD texture enrichment within an adopted conduction-dominated envelope. Stage-1 maps the LPBF process variables to printability, melting mode, and melt-pool geometry, with its predicted depth and width serving as fixed model-derived mediators. Stage-2 combines a conductioninspired greybox anchor with a random forest (RF) residual model. A standardized k-nearestneighbor (kNN) distance attenuates the residual as data support weakens, while a separate, study-specific areal beam-power-density criterion withholds predictions outside the adopted physics-validity envelope. Conformal intervals are calibrated through grouped cross-validation (group-CV) and reported only for retained predictions.

The main contributions of this study are summarized as follows.

1. The proposed framework separates data applicability, physics validity, and predictive uncertainty, which are often combined into a single confidence measure. The learned correction is attenuated as local data support weakens, predictions are withheld when the anchor lies outside its adopted validity envelope, and intervals are reported only for

predictions that are issued.

2. The evaluation is designed to assess transfer across experimental groups and process regimes. Complete experimental groups and defocus levels are excluded from model fitting, and literature-derived model forms are reimplemented on the same dataset under the same evaluation protocol to provide a consistent comparison.

3. The resulting reliability behavior is demonstrated through withheld-regime evaluation and separately built specimens not used in model development. The physics-anchored framework retains predictive capability in a withheld defocus condition where the unconstrained data-driven model does not, and issues nothing in a regime where no evaluated model class performs adequately. On the separately built set the framework applies the same distinction without adjustment, withholding, attenuating, and issuing predictions according to its own criteria.

The remainder of this paper is organized as follows. Section 2 describes the experimental setups and datasets, and section 3 introduces two-stage framework, validation protocol, and statistical analyses. Section 4 presents and discusses predictive performance, transfer and abstention behavior, uncertainty results, and model interpretation. Section 5 illustrates the propagation of retained texture predictions to build-direction modulus, applies the frozen framework to a separately built set, and examines the assumed orientation dependence at room and elevated temperature, followed by the conclusions in Section 6.

## 2. Experimental setups and datasets

All specimens were fabricated from Inconel 718 powder with a near-normal particle size distribution centered at approximately $3 0 \mu \mathrm { m }$ , using a Nikon SLM 280 LPBF system equipped with a Ytterbium continuous-wave fiber laser (nominal wavelength of 1070 nm, nominal spot size of $8 0 \pm 1 0 ~ \mu \mathrm { m }$ , Rayleigh length $4 \pm 1$ mm, beam quality $\mathrm { M } ^ { 2 } = 1 . 0 { - 1 . 5 } .$ , Gaussian beam profile [34,35]) under an argon atmosphere. A meandering scan strategy with $9 0 ^ { \circ }$ inter-layer rotation was applied, and the layer thickness was held constant at 30 µm throughout. The process space was designed to span conduction-dominated, transitional, and keyholedominated melting conditions. Melt-pool geometry was measured on etched cross-sections prepared perpendicular to the scanning direction, with five melt pools evaluated per parameter combination. Tracks showing severe lack of fusion or discontinuous melt-pool formation were labelled as failed. Each successful track was assigned to conduction or keyhole mode from the observed cross-sectional morphology. Only width and depth were recorded for conductionmode pools, while maximum width, overall depth, transition width and transition depth were for keyhole-mode pools.

Two linked datasets supported the process-to-melt-pool-to-texture workflow. Stage-1 comprised 372 conditions spanning laser powers of 150–400 W, scan speeds of 75–2,000 mm $\mathbf { S } ^ { - 1 }$ , hatch spacings of $5 0 {  { - } } 4 0 0 ~ { \mu \mathrm { m } }$ , and focus offsets of −160 to +80 mm, with the sign convention defined in Figure 1a. Of these, 345 conditions were successful and 27 failed. Printability classification used all 372 conditions, whereas melting-mode classification and geometry regression used the 345 successful conditions, comprising 257 conduction-mode (CM) and 88 keyhole-mode (KM) cases. The focus-offset convention and its effect on the surface beam radius are illustrated in Figure 1a. Stage-1 data were used only for the processto-melt-pool bridge described in Section 3.2.

The Stage-2 dataset comprised 278 electron backscatter diffraction (EBSD) observations across nine predefined experimental groups, designated G1 through G9 (Table 1). These observations represented 269 unique process vectors (P, V, H, F); nine vectors occurred in two groups and were retained as distinct group-level texture measurements.

Table 1. Composition of the Stage-2 dataset across the nine predefined experimental groups
<table><tr><td>Experimental group</td><td>n</td><td>Focus-offset composition (mm: N)</td><td>P (W)</td><td>V (mm s−1)</td><td>H (μm)</td></tr><tr><td>G1</td><td>38</td><td>20 (38)</td><td>200-400</td><td>100-600</td><td>100-400</td></tr><tr><td>G2</td><td>36</td><td>0 (12); 80 (24)</td><td>250-400</td><td>100-600</td><td>100-500</td></tr><tr><td>G3</td><td>41</td><td>0 (11); 40 (30)</td><td>150-400</td><td>100-800</td><td>75-400</td></tr><tr><td>G4</td><td>30</td><td>40 (18); 80 (12)</td><td>300-400</td><td>200</td><td>125-350</td></tr><tr><td>G5</td><td>42</td><td>0 (12); 20 (24); 40 (6)</td><td>150-400</td><td>200</td><td>50-300</td></tr><tr><td>G6</td><td>18</td><td>0 (4); 20 (3); 40 (7); 80 (4)</td><td>400</td><td>400-1000</td><td>50-175</td></tr><tr><td>G7</td><td>21</td><td>0 (15); 20 (6)</td><td>400</td><td>1000-1600</td><td>25-125</td></tr><tr><td>G8</td><td>11</td><td>20 (6); 40 (3); 80 (2)</td><td>400</td><td>400-1200</td><td>25-125</td></tr><tr><td>G9</td><td>41</td><td>10 (18); 20 (11); 30 (12)</td><td>400</td><td>500-700</td><td>50-225</td></tr><tr><td>Overall</td><td>278</td><td>0: 54; 10: 18; 20: 88; 30: 12; 40: 64; 80: 42</td><td>150-400</td><td>100-1600</td><td>25-500</td></tr></table>

Groups are designated G1–G9; the correspondence with the acquisition identifiers used in the distributed project archive is given in the Supplementary Information. Focus-offset entries give the focus value (mm) followed by the observation count in parentheses. The nine groups define the leave-one-group-out (LOGO) units. The +80 mm (n = 42) and focus-zero (n = 54) holdouts span multiple groups and are defined at the dataset level.

The two datasets shared overlapping process ranges but supported different response variables. Stage 1 used melt-pool labels to learn printability, melting mode, and geometry, whereas Stage 2 used the corresponding process variables together with the Stage-1 predicted geometry to model crystallographic texture. No measured Stage-1 geometry label was used directly as a Stage-2 input, and the Stage-1 models were fitted to the complete melt-pool dataset before generating the model-derived depth and width for the 278 EBSD observations. Stage-1 extended farther in scan speed and focus offset, while Stage-2 covered a wider hatch-spacing range. Fourteen Stage-2 observations lay outside the Stage-1 hatch-spacing range of 50–400 µm. They were retained and flagged as Stage-1 hatch-range extrapolations, so their geometry estimates represent boundary-supported tree-model predictions rather than interpolation.

Tensile specimens were built as cylindrical bars and machined on a CNC lathe to a nominal gauge length of 20 mm and a gauge diameter of 4.75-5.0 mm, with threaded sections at both ends. The tensile axis was parallel to the build direction.

Specimens for texture characterization were taken from a build height of approximately 13 mm and sectioned so that the observation-plane normal corresponded to the build direction. The specimens were mounted in conductive resin and mechanically polished to a 0.05 µm colloidal silica finish without subsequent etching. Crystallographic texture was characterized by electron backscatter diffraction (EBSD) using a Thermo Scientific Helios 5 UC DualBeam FIB-SEM equipped with an EDAX EBSD system at the Nanoscale Characterization and Fabrication Laboratory, Virginia Tech. Scans were acquired at an accelerating voltage of 30 kV using a hexagonal grid with a step size of 1.2 to 2 µm over areas of approximately $0 . 8 \times 0 . 6$ to $1 . 0 \times$ 0.8 mm, each containing hundreds of grains. Indexing rates exceeded 99%, and the indexed points were treated as the valid orientation points from which the texture target was constructed. Orientation data were processed using TSL OIM Analysis and the MTEX toolbox.

Each Stage-2 observation was associated with a process vector, an experimental-group label, and an EBSD-derived crystallographic texture target. The primary target was the enrichment of ⟨001⟩ orientations parallel to the build direction (BD). Let $\mathrm { f _ { 0 0 1 } }$ denote the percentage of valid indexed orientation points for which any symmetry-equivalent ⟨001⟩ direction lay within $1 5 ^ { \circ }$ of BD. The texture target was defined as the ratio of $\mathrm { f _ { 0 0 1 } }$ to the adopted random-texture reference fraction of 10.21%:

$$
\mathbf { e } _ { 0 0 1 } { = } \frac { \mathrm { f } _ { 0 0 1 } } { 1 0 . 2 1 } .\tag{1}
$$

The normalized target $\mathrm { \bf e } _ { 0 0 1 }$ was expressed in multiples of a uniform distribution (MUD), where $\mathrm { e } _ { 0 0 1 } { = } 1$ represents the random texture and values above unity indicate ⟨001⟩ ∥ BD enrichment. The dataset spanned 0.533–7.965 MUD. This point-fraction-based target was determined solely from the measured EBSD orientations and was fixed before model fitting. Pole-figure maxima, composite texture scores, and model-derived quantities were excluded. Figure 1b, c illustrate lower and higher enrichment.

![](images/0b8209c6d384ce9da58223bc4d54b3c0bcb07c5ede69c0c69b07fce6a49962db.jpg)  
Figure 1. Focus-offset convention and the EBSD-derived texture target. (a) At (F=0), the beam waist lies at the powder-bed surface; (F>0) moves the focal plane toward the laser, increasing r(F) and reducing $\mathsf { q } _ { \mathrm { P D } }$ . Beam geometry is not to scale. (b, c) BD-referenced inverse pole figure maps for two observations from G1 with lower and higher ⟨001⟩ ∥ BD enrichment. The target e₀₀₁ is the fraction of valid indexed points within $1 5 ^ { \circ }$ of ⟨001⟩ ∥ BD, normalized by the 10.21% random-texture fraction; it is not derived from pole-figure intensity.

The nine groups were retained as the units for leave-one-group-out (LOGO) validation and conformal calibration. Because nine process vectors occurred in two groups, a held-out group could share a process vector with the training groups. LOGO therefore evaluates transfer to an unseen experimental group rather than strict leave-one-process-condition-out extrapolation.

## 3. Physics-based machine learning and statistical analysis

## 3.1 Overall physics-based modeling framework

The proposed framework predicts the ⟨001⟩ ∥ BD texture intensity from LPBF process parameters through a two-stage predictor and a separate reliability layer (Figure 2). Stage-1 maps P, V, H, and F, together with their derived descriptors, to printability, melting mode, and melt-pool depth and width. The predicted geometry is passed to Stage-2 as model-derived mediators. Stage-2 combines a conduction-inspired greybox anchor with an RF residual correction. A standardized kNN distance attenuates the residual as empirical support weakens, while a separate areal beam-power-density criterion withholds predictions and intervals outside the adopted anchor-validity envelope. For retained predictions, group-CV conformal calibration provides empirical 90% intervals. Residual-model applicability, physics-domain validity, and uncertainty calibration therefore remain separate decisions. Sections 3.2-3.4 detail the Stage-1 bridge, Stage-2 predictor, and reliability layer, respectively.

![](images/a0535d110390a6e24132a6d2d9b5820a02fd79b5fa16555b09aea193ffa00d3b.jpg)  
Figure 2. Overall physics-based framework for prediction and uncertainty quantification of IN718 crystallographic texture under process-regime shift in LPBF. Stage-1 predicts printability, melting mode, and melt-pool geometry. Stage-2 combines a conduction-inspired greybox anchor with a randomforest residual correction. Data-support weighting attenuates the residual, physics-validity screening controls abstention, and grouped conformal calibration provides empirical 90% intervals for retained predictions. Solid arrows show prediction flow; dashed arrows connect measured targets used for training or calibration.

## 3.2 Stage-1 process-to-melt-pool bridge

Eight descriptors were derived from the four process variables using fixed material and optical constants: six energy- and transport-related proxies and two focus-offset descriptors, |F| and sign (F). The effective Gaussian beam radius was calculated as

$$
\mathrm { r ( F ) { = w _ { 0 } } \sqrt { 1 { + } \left( \frac { F } { z _ { R } } \right) ^ { 2 } } }\tag{2}
$$

where $\mathrm { w } _ { 0 } { = } 0 . 0 4 0$ mm and $z _ { \mathrm { R } } { = } 4 . 0$ mm denote the beam-waist radius and Rayleigh length, respectively. The remaining energy and transport descriptors were calculated as

$$
\mathrm { E _ { L } = \frac { P } { V } }\tag{3}
$$

$$
\mathrm { E } _ { \mathrm { V } } { = } \frac { \mathrm { ~ P ~ } } { \mathrm { V } \mathrm { H } \mathrm { t } }\tag{4}
$$

$$
{ \sf q } _ { \mathrm { P D } } { = } \frac { \mathrm {  ~ \cal ~ P ~ } } { \pi \mathrm { r } ( \mathrm {  ~ \cal ~ F ~ } ) ^ { 2 } }\tag{5}
$$

$$
\mathrm { C _ { p r o x y } = \frac { V } { E _ { V } } }\tag{6}
$$

$$
\mathrm { P e } { = } \frac { \left( \mathrm { V } / 1 0 0 0 \right) \left[ 2 \mathrm { r } ( \mathrm { F } ) / 1 0 0 0 \right] } { \mathsf { a } _ { \mathrm { t h } } }\tag{7}
$$

$$
\Pi _ { \mathrm { E } } { = } \frac { \mathrm { ~ \mathsf { P } ~ } } { \mathrm { V } \mathrm { H } }\tag{8}
$$

The six derived descriptors comprise line energy $\mathrm { E _ { L } }$ (J mm⁻¹), volumetric energy $\operatorname { E } _ { \mathrm { V } }$ (J mm⁻³), the areal beam-power-density proxy ${ \mathsf { q } } _ { \mathrm { { P D } } }$ (W mm⁻²), the areal-energy proxy $\Pi _ { \mathrm { E } }$ (J $\mathrm { m m } ^ { - 2 } )$ , the dimensionless Péclet number $\mathrm { P e , }$ and $\mathrm { C _ { p r o x y } } ,$ , an empirical cooling proxy retained in its implemented form with derived units of mm J⁻¹ $\mathbf { S } ^ { - 1 }$ . H was converted from μm to mm. The feature calculation used $\scriptstyle \mathrm { t = } 0 . 0 3 0$ mm and $\mathrm { \Delta a _ { t h } } { = } 5 . 3 { \times } 1 0 ^ { - 6 } \mathrm { \ m } ^ { 2 } \mathrm { \ s } ^ { - 1 }$ for the beam-waist radius and Rayleigh length of the installed fiber laser, taken from the manufacturer specification, and a thermal diffusivity of $5 . 3 \times 1 0 ^ { - 6 }$ m² s⁻¹ for IN718 near the liquidus [36]. The descriptors |F| and sign (F) preserved both the magnitude and direction of defocus. Together with the original process variables P, V, H, and F, these descriptors formed the 12-dimensional Stage-1 input vector:

$$
{ \bf x } _ { \mathrm { M P } } { = } [ \mathrm { P } , \mathrm { V } , \mathrm { H } , \mathrm { F } , \mathrm { ~ | ~ \mathrm { ~ F } ~ | ~ } , \mathrm { s i g n } \left( \mathrm { F } \right) , \mathrm { E } _ { \mathrm { L } } , \mathrm { E } _ { \mathrm { V } } , { \bf q } _ { \mathrm { p p } } , \mathrm { C } _ { \mathrm { p r o x y } } , \mathrm { P e } , \Pi _ { \mathrm { E } } ]\tag{9}
$$

Stage-1 used three sequential branches. A balanced RF first classified whether a condition produced a successful melt pool [37]. Successful conditions were then routed by an XGBoost keyhole-probability classifier, with a threshold of 0.5 assigning each query to the CM or KM geometry branch [38]. CatBoost predicted CM depth and width [39], while Gaussian-process regression (GPR) predicted KM depth and maximum width after feature standardization and response normalization [40]. Model configurations and validation results are summarized in Table 2. All stochastic Stage-1 models used a fixed random seed of 42.

Table 2. Stage-1 model configurations and validation results for printability, melting-mode routing, and mode-specific geometry regression.
<table><tr><td>Task</td><td>Deployed model</td><td>Validation</td><td>Result</td></tr><tr><td>Printability</td><td>Balanced RF (400 trees)</td><td>5-fold stratified CV</td><td>ROC-AUC = 0.802; accuracy = 0.952</td></tr><tr><td>CM/KM classification</td><td>XGBoost (300 trees; depth 4; learning rate 0.1)</td><td>5-fold stratified CV</td><td>ROC-AUC = 0.990; accuracy = 0.954</td></tr><tr><td>CM geometry</td><td>CatBoost (300 iterations; depth 4;</td><td>5-fold GroupKFold by (P, V, H, F)</td><td>Depth R² = 0.849; width R² = 0.868</td></tr><tr><td>KM geometry</td><td>learning rate 0.1) Standard Scaler + GPR constant × radial basis function +</td><td>5-fold GroupKFold</td><td>Depth R² = 0.781;</td></tr></table>

Geometry models were evaluated by five-fold GroupKFold using the complete process vector (P, V, H, F) as the grouping key. Performance was summarized by the coefficient of determination (R²) and root-mean-square error (RMSE). Because each of the 345 successful Stage-1 conditions was unique, GroupKFold was equivalent to a condition-level partition and could not estimate repeated-condition or repeated-build variability.

After validation, the Stage-1 models were refitted on the complete melt-pool dataset and applied to the 278 Stage-2 observations. CM-routed queries received CatBoost depth and width predictions, whereas the KM-routed query received GPR depth and maximum-width predictions. The resulting D̂ and Ŵ values served as model-derived Stage-2 mediators

All 278 Stage-2 observations had measured EBSD targets and were therefore retained for texture modeling. The Stage-1 printability probability for these observations was recorded as a diagnostic output and was not used to exclude observations.

Meanwhile, the Stage-1 classifier served to select the CM or KM geometry surrogate used to generate $\hat { \mathrm { D } }$ and $\hat { \mathbf { W } } .$ It routed 277 observations to the CM branch and one to the KM branch. Consequently, the model-derived mediators entering Stage 2 were generated almost entirely by the CM geometry surrogate, providing context for the conduction-inspired form of the Stage-2 anchor. Branch assignment did not establish the physics validity of that anchor or determine whether a Stage-2 prediction was issued. Prediction issuance was assessed separately using the $\mathsf { q } _ { \mathrm { P D } }$ criterion defined in Section 3.4.

Figure 3 and Table 2 summarize Stage-1 validation. Depth parity is shown because D<sup>̂</sup> enters the Stage-2 greybox directly, whereas $\hat { \mathbf W }$ enters only the residual model. Grouped-CV RMSEs for CM and KM depth were 0.021 and 0.091 mm respectively, and the corresponding width RMSEs were 0.041 and 0.081 mm.

(a) Schematic melt-pool morphologies  
![](images/5359f31862b35410bb303eb6621f85fc70262f37981f3987aada750e6fadff57.jpg)

![](images/7b10d582cc1e25e36d56bb077de6f2be4ae0e2e8b2340c74748aea0763a894d7.jpg)

(b) Printability classification  
![](images/ab25ca3ff91afa688f1c4960eeb68e6203a175e53fdd83c76ee2f6bfbba7aa04.jpg)

(c) Conduction-mode depth  
![](images/919fe267ab8877ee809135a02263a551e9c35cb0746dae6c9b57c6cc4a8f82b6.jpg)

(d) Keyhole-mode depth  
![](images/a793ec13a4a1418f9bf198441b29c53828cc61c6393bb3ba99cf80072a54f60d.jpg)  
Figure 3. Physical labels and validation of the Stage-1 bridge. (a) Schematic conduction-mode (CM) and keyhole-mode (KM) melt-pool cross sections. (b) Five-fold stratified-CV receiver operating characteristic for the balanced random-forest printability classifier. (c, d) Grouped-CV depth parity for the CM CatBoost and KM Gaussian Process regression models respectively.

## 3.3 Stage-2 physics-anchored residual model and applicability weighting

Stage-2 predicted the normalized texture target $\mathbf { e } _ { 0 0 1 }$ from the process variables and Stage-1 melt-pool mediators. It combined a conduction-inspired greybox anchor with a learned statistical discrepancy [41]. For observation i, the deployed six-component input vector was

$$
\mathrm { \mathbf { x } _ { i } = [ P _ { i } , V _ { i } , H _ { i } , F _ { i } , \widehat { D } _ { i } , \widehat { W } _ { i } ] }\tag{10}
$$

where $\widehat { \mathrm { D } } _ { \mathrm { i } }$ and $\widehat { \mathbb { W } } _ { \mathrm { i } }$ are the mode-specific depth and width predictions generated by the Stage-1 surrogate. For KM-routed queries, $\widehat { \mathbb { W } } _ { \mathrm { i } }$ denoted the predicted maximum width. The meltingmode label selected the Stage-1 geometry branch but was not included as a separate Stage-2 input

A positive multiplicative form in V, P, and D<sup>̂</sup> was adopted as a low-capacity empirical anchor.

Laser power and scan speed represent the primary energy-input and interaction-time variables, whereas $\hat { \mathrm { D } }$ supplies a model-derived melt-pool-state mediator that carries the thermal-gradient information not recoverable from the process variables alone. The fitted exponents were treated as empirical associations in the stated units rather than as universal physical constants. Recent operando evidence indicates that local abnormal columnar-to-equiaxed transition in metal additive manufacturing may involve ordering-mediated nucleation pathways not resolved by classical criteria based solely on thermal gradient and growth rate [42]. Although the present anchor is not formulated in terms of these quantities, this finding provides broader context for interpreting it as a compact empirical association rather than a complete mechanistic model of texture formation. The greybox anchor related e<sub>001</sub> multiplicatively to V, P, and $\hat { \mathrm { D } }$ and was fitted by ordinary least squares after logarithmic transformation:

$$
\log \mathrm { e _ { 0 0 1 , i } } \mathrm { = l o g \ A } { + } \mathrm { a \log V _ { i } } { + } \mathrm { b \log P _ { i } } { + } \mathrm { c \log \widehat { D } _ { i } } { + } \mathrm { s _ { i } }\tag{11}
$$

The full-data fit used for final deployment was

$$
\mathbf { e } _ { 0 0 1 , \mathrm { i } } ^ { \mathrm { p h y s } } { = } 0 . 0 1 4 8 \mathbf { V } _ { \mathrm { i } } ^ { 0 . 3 0 } \mathbf { P } _ { \mathrm { i } } ^ { 0 . 4 1 } \widehat { \mathbf { D } } _ { \mathrm { i } } ^ { - 0 . 6 3 }\tag{12}
$$

Here, P, V, and $\widehat { \sf D }$ were expressed in $\mathrm { W } ,$ mm $\mathbf { s } ^ { - 1 }$ , and mm, respectively, and $\widehat { \sf D }$ was a Stage-1 prediction rather than a measured quantity. Equation 12 was treated as a conduction-inspired empirical anchor. Its low-dimensional form was intended to remain informative when the learned correction had limited data support. Coefficient estimates and ablation results are reported in Section 4.3.

An RF residual model learned the departure from the greybox anchor. For each training observation, the residual target was

$$
\mathrm { \Delta r _ { i } { = } e _ { 0 0 1 , i } ^ { m e a s } { = } \ e _ { 0 0 1 , i } ^ { p h y s } }\tag{13}
$$

An RF regressor with 400 trees and a fixed random seed of 42 learned

$$
\hat { \mathbf { r } } _ { \mathrm { i } } { = } \mathbf { f } _ { \mathrm { R F } } ( \mathbf { x } _ { \mathrm { i } } )\tag{14}
$$

Using the six-component vector in Equation 10 [37], the residual model represented dependencies absent from the greybox, including hatch spacing, focus offset, predicted width, and nonlinear interactions. These contributions were interpreted as model-based associations.

To limit residual correction under weak data support, the six inputs were standardized within each training fold and a kNN applicability weight was applied [26,27]:

$$
\mathbf { z } { = } \frac { \mathbf { x } { - } { \mathbf { \mu } } _ { \mathrm { t r a i n } } } { \mathbf { \sigma } _ { \mathrm { t r a i n } } }\tag{15}
$$

For a query x, its applicability distance was defined as the mean Euclidean distance to the five nearest standardized training observations:

$$
\mathbf { d } ( \mathbf { x } ) { = } \frac { 1 } { 5 } \sum _ { \mathrm { j } \in \mathrm { N } _ { 5 } ( \mathbf { x } ) } \| \mathbf { z } { - } \mathbf { z } _ { \mathrm { j } } \| _ { 2 }\tag{16}
$$

Reference distances were computed leave-one-out with self matches excluded, so both query and training distances used five non-self-neighbors. The training-fold reference radius was the 95th percentile of these leave-one-out distances:

$$
\mathsf { d } _ { \mathrm { { r e f } } } { = } \mathsf { P } _ { 9 5 } \left( \big \{ \mathsf { d } _ { \mathrm { { i } } } ^ { \mathrm { { L O O } } } \big \} _ { \mathrm { { i } } \in \mathrm { { t r a i n } } } \right)\tag{17}
$$

The continuous residual weight was

$$
\mathrm { w } ( \mathrm { x } ) { = } \mathrm { e x p } \left[ { - } \mathrm { m a x } \left( \frac { \mathrm { d } ( \mathrm { x } ) } { \mathrm { d _ { \mathrm { r e f } } } } { - } 1 , 0 \right) \right]\tag{18}
$$

The weight equals unity when $\mathbf { d } ( \mathbf { x } ) { \leq } \mathbf { d } _ { \mathrm { r e f } }$ and decreases smoothly beyond this radius. The final center prediction was

$$
\hat { \mathbf { e } } _ { 0 0 1 } ( \mathbf { x } ) { = } \mathbf { e } _ { 0 0 1 } ^ { \mathrm { p h y s } } ( \mathbf { x } ) { + } \mathbf { w } ( \mathbf { x } ) \hat { \mathbf { r } } ( \mathbf { x } )\tag{19}
$$

Thus, the model retained the full residual correction for supported queries and reverted toward the greybox anchor as distance increased. The threshold $\tau _ { \mathrm { w } } { = } 0 . 6 0$ fixed in the archived canonical configuration served only as a limited-support flag; it neither triggered abstention nor altered the conformal half-width. Four point predictors were compared under identical protocols: the greybox anchor $\mathrm { { e } _ { 0 0 1 } ^ { p h y s } }$ ; a 400-tree black-box random forest fitted directly to the measured target using the same six-component input vector; the ungated hybrid $\mathrm { e } _ { 0 0 1 } ^ { \mathrm { p h y s } } + \hat { \mathbf { r } } ;$ and the proposed gated hybrid $\mathbf { e } \mathbf { e } _ { 0 0 1 } ^ { \mathrm { p h y s } } + \mathbf { w } \cdot \hat { \mathbf { r } } .$ . Physics-domain abstention is defined separately in Section 3.4.

All data-dependent Stage-2 components were fitted within each training partition, as detailed in Section 3.5. After validation, the greybox, residual model, and applicability reference were refitted on the complete Stage-2 dataset for deployment.

## 3.4 roup- V conformal uncertainty quantification and physics-domain abstention

Group-aware conformal intervals were constructed around the gated center prediction while preserving the experimental-group structure [28]. Because distribution shift weakens the standard exchangeability assumptions, coverage is reported as empirical retained-set coverage rather than as a strict conditional guarantee [30,31].

Within each outer evaluation split, the Stage-2 predictor was fitted on the outer-training observations, and inner LOGO generated out-of-group calibration predictions. Each inner validation group was predicted using a greybox, residual model, standardization, and applicability reference fitted without that group. The nonconformity score for held-out observation i was

$$
\mathbf { s } _ { \mathrm { i } } { = } | \mathbf { e } _ { 0 0 1 , \mathrm { i } } ^ { \mathrm { m e a s } } - \hat { \mathbf { e } } _ { 0 0 1 , \mathrm { i } } ^ { ( - \mathbf { g } _ { \mathrm { i } } ) } |\tag{20}
$$

where $\boldsymbol { \hat { \mathrm { e } } _ { 0 0 1 , \mathrm { i } } ^ { ( - \mathrm { g } _ { \mathrm { i } } ) } }$ denotes the gated prediction obtained without experimental group $\mathrm { g } _ { \mathrm { i } }$ . Only inner held-out observations satisfying $\mathfrak { q } _ { \mathrm { p } \mathrm { p } , \mathrm { i } } \le \mathfrak { r } _ { \mathfrak { q } }$ contributed nonconformity scores, with $\tau _ { \mathrm { q } } { = } 2 0 , 0 0 0 \mathrm { W } \mathrm { m m } ^ { - 2 }$ . This calibrate-on-issued rule restricted the calibration pool only. Observations above $\tau _ { \mathrm { q } }$ could therefore contribute to model fitting when they belonged to an inner-training group, but their nonconformity scores were excluded when they appeared in the inner held-out group. All Stage-2 components were refitted within each inner split, after which the complete predictor was fitted on the full outer-training set for prediction of the outer test fold.

For a nominal miscoverage level of $\mathtt { q } \mathrm { = } 0 . 1 0$ , the retained calibration scores were sorted as $\mathbf { s } _ { ( 1 ) } { \le } \mathbf { s } _ { ( 2 ) } { \le } \cdots { \le } \mathbf { s } _ { ( \mathrm { n _ { c a l } } ) }$ . The conformal half-width was selected using the exact order statistic:

$$
\mathrm { q } _ { \ 0 . 9 0 } { = } \mathrm { s } _ { ( \mathrm { J ( n _ { c a l } + 1 ) ( 1 - \alpha ) } | ) }\tag{21}
$$

All retained queries within the same outer evaluation split received the constant interval

$$
\mathbf { C } ( \mathbf { x } ) { = } \left[ \hat { \mathbf { e } } _ { 0 0 1 } ( \mathbf { x } ) { - } \mathbf { q } _ { 0 . 9 0 } , \hat { \mathbf { e } } _ { 0 0 1 } ( \mathbf { x } ) { + } \mathbf { q } _ { 0 . 9 0 } \right]\tag{22}
$$

The conformal half-width was not rescaled by $\mathbf { w } ( \mathbf { x } )$ which affected only the center prediction through Section 3.3. A query with $\mathrm { w } ( \mathbf { x } ) < \tau _ { \mathrm { w } }$ could therefore receive a physics-anchored prediction and interval when $\mathsf { q } _ { \mathrm { p D } } ( \mathbf { x } ) \leq \tau _ { \mathrm { q } }$ . The value $\tau _ { \mathrm { w } }$ alone did not trigger abstention. The black-box comparator used the same inner-group calibration structure and the same finitesample order statistic. Its nonconformity scores were drawn from the same physics-valid inner held-out observations, so the two calibrations are directly comparable. The comparator itself implemented no abstention and issued a prediction and an interval for every test query.

A separate ${ \mathsf { q } } _ { \mathrm { { P D } } }$ criterion controlled the physics-validity decision. Using the focus-dependent beam radius from Equation 2, predictions and intervals were issued only when $\mathsf { q } _ { \mathrm { p D } } ( \mathbf { x } ) \leq \tau _ { \mathrm { q } } .$ Queries above this threshold were treated as outside the adopted deployment envelope of the conduction-inspired anchor, and both the center prediction and interval were withheld. The operational output was

$$
\begin{array} { r } { \mathrm { O } ( \mathbf { x } ) = \left\{ \begin{array} { l l } { \{ \hat { \mathbf { e } } _ { 0 0 1 } ( \mathbf { x } ) , \mathrm { C } ( \mathbf { x } ) , \mathrm { w } ( \mathbf { x } ) \} , } & { \mathrm { q } _ { \mathrm { p D } } ( \mathbf { x } ) \leq \tau _ { \mathrm { q } } , } \\ { \mathrm { a b s t a i n , } } & { \mathrm { q } _ { \mathrm { p D } } ( \mathbf { x } ) > \tau _ { \mathrm { q } } . } \end{array} \right. } \end{array}\tag{23}
$$

In the final canonical implementation, the threshold $\tau _ { \mathrm { q } }$ was fixed at 20,000 W mm⁻² and applied identically in every evaluation fold without re-estimation within folds. It lies above the largest value represented in the positive-defocus training domain $( 1 . 1 0 \times 1 0 ^ { 4 } \mathrm { ~ W ~ m m } ^ { - 2 } )$ and below the smallest focus-zero value $( 2 . 9 8 \times 1 0 ^ { 4 } \mathrm { W } \mathrm { m m } ^ { - 2 } )$ . Any threshold within this interval produces identical retention and abstention outcomes in all three evaluation settings. The available development record does not establish that the value was specified prospectively before inspection of the focus-zero results. It is a study-specific validity rule rather than a universal keyhole threshold. The weight w(x) controls residual attenuation, whereas $\tau _ { \mathrm { q } }$ controls issuance. This separation prevents a data-support measure from being treated as an independent physics-validity guarantee. A query can therefore have full data support and still be withheld or lie far from the training distribution while remaining inside the validity envelope.

Coverage and interval width were evaluated only over retained predictions and reported together with retention. Abstained observations were reported separately and were not counted as uncovered predictions. This follows the selective-prediction principle while recognizing that the present rule is a study-specific regression-domain criterion rather than the original classification reject option [43]. Detailed metrics are defined in Section 3.5. Outer-test targets were excluded from model fitting, applicability-reference construction, and conformal calibration, as detailed in Section 3.5.

## 3.5 Validation protocol and statistical analysis

Stage-2 validation examined grouped generalization, controlled conduction-regime transfer, and behavior outside the adopted anchor-validity envelope. These settings represent distinct forms of model use. Stage-1 validation is described in Section 3.2 and summarized in Table 2. The primary Stage-2 evaluation applied LOGO cross-validation to 278 observations from nine experimental groups. Each outer split withheld one complete group; all Stage-2 models, feature scaling, and applicability references were fitted using the remaining groups, with conformal calibration performed by inner LOGO as described in Section 3.4. The Stage-1 bridge was fitted once on the independent melt-pool dataset and frozen before Stage-2 partitioning. It used neither the EBSD-derived target nor the Stage-2 group labels and was not regenerated within the outer folds. LOGO therefore evaluates the Stage-2 mapping conditional on the frozen Stage-1 surrogate rather than end-to-end generalization.

The focus-offset holdouts were complementary physics-based stress tests rather than symmetric numerical extrapolations. At +80 mm, the increased surface beam radius reduced $\mathsf { q } _ { \mathrm { P D } }$ . The queries had weak residual-model support but remained within the adopted deployment envelope of the conduction-inspired anchor. At focus zero, the beam waist coincided with the powder-bed surface, producing the highest ${ \mathsf { q } } _ { \mathrm { { P D } } }$ values and exceeding $\tau _ { \mathrm { q } }$ . Higher localized intensity is associated with the conduction-to-keyhole transition and vapor-depression formation [12,13], but $\tau _ { \mathrm { q } }$ is neither a universal keyhole criterion nor a basis for reclassifying the focus-zero observations as KM.

For the +80 mm evaluation, all 42 observations were withheld and the remaining 236 were used for fitting. Because this regime participated in framework development, it was treated as a controlled withheld-defocus evaluation rather than independent external validation. For the focus-zero evaluation, all 54 observations were withheld and the remaining 224 were used for fitting. Point-prediction metrics were calculated diagnostically, but no operational prediction or interval was issued because every focus-zero query exceeded $\tau _ { \mathfrak { q } }$ . Repeated process conditions were assigned entirely to either the training or withheld subset.

Within each evaluation regime, all Stage-2 models, feature scaling, and applicability quantities were recomputed from the training partition. No held-out target contributed to greybox fitting, residual learning, $\mathrm { d } _ { \mathrm { r e f } }$ , or conformal calibration. Final metrics were regenerated after the canonical configuration was frozen; neighborhood-size and reference-percentile sweeps were treated only as robustness analyses.

Point-prediction performance was reported using $\mathrm { R } ^ { 2 } { \mathrm { . } }$ , mean absolute error (MAE), root-meansquare error (RMSE), and Spearman's ρ. LOGO metrics were calculated from pooled out-offold predictions and are therefore sample-weighted aggregates rather than unweighted means of group-specific scores. Withheld-defocus metrics were calculated over each complete withheld subset.

For evaluation regime g containing $\mathfrak { n } _ { \mathrm { g } }$ observations, let $\mathrm { R _ { g } }$ denote the retained set for which predictions and nominal 90% intervals [L<sub>i</sub>, U<sub>i</sub>] were issued, and let $\mathrm { y } _ { \mathrm { i } }$ denote the measured target. Retained-set coverage, mean full interval width, and retention were calculated as

$$
\mathrm { C o v e r a g e _ { g } { = } \frac { 1 } { | R _ { g } | } \sum _ { i \in R _ { g } } { \bf 1 } \left( L _ { i } \leq y _ { i } \leq U _ { i } \right) }\tag{24}
$$

$$
\bar { \mathrm { W } } _ { \mathrm { g } } { = } \frac { 1 } { | \mathrm { R } _ { \mathrm { g } } | } \sum _ { \mathrm { i } \in \mathrm { R } _ { \mathrm { g } } } \left( \mathrm { U } _ { \mathrm { i } } { - } \mathrm { L } _ { \mathrm { i } } \right)\tag{25}
$$

$$
\mathrm { R e t e n t i o n } _ { \mathrm { g } } { = } \frac { | \mathrm { R } _ { \mathrm { g } } | } { \mathrm { n } _ { \mathrm { g } } }\tag{26}
$$

Abstained observations remained in the retention denominator but were excluded from coverage and width because no interval was issued. Coverage was reported with retention and is not an unconditional guarantee across evaluation regimes. The non-abstaining black-box comparator was evaluated over all observations in each regime.

Interval width was compared with a regime-specific target-only null interval constructed from the corresponding retained set. The null interval was centered on the median target, with its half-width determined by applying the same finite-sample order-statistic rule to absolute deviations from that median. Relative width reduction was calculated as

$$
\mathrm { W i d t h \ r e d u c t i o n _ { g } { = } 1 0 0 \left( 1 { - } \frac { \bar { W } _ { g } } { W _ { n u l l , g } } \right) \% }\tag{27}
$$

Width reduction was reported only for the gated model. The target-only interval served as a descriptive width reference rather than a deployable baseline; coverage and width reduction were undefined when no predictions were retained.

Greybox-exponent stability was evaluated by resampling the nine experimental groups with replacement and refitting the log-linear model for 2,000 cluster-bootstrap replicates [44]. The Stage-1 predicted depth $\hat { \mathrm { D } }$ was held in every replicate, so the percentile intervals were conditional on $\hat { \mathrm { D } }$ and did not propagate Stage-1 uncertainty. Given nine groups, these intervals describe empirical group-resampling stability rather than complete workflow uncertainty or a causal scaling law. Variance inflation factors for logV, logP, and logD<sup>̂</sup> were computed to assess collinearity.

Residual-model interpretation used Shapley additive explanations (SHAP) and Sobol analysis at the controllable-process level [45–47]. P, V, H, and F were sampled independently from uniform distributions spanning their observed marginal bounds, while deterministic mediator surrogates supplied the frozen Stage-1 depth and width. Sobol first-order indices $\mathbf { S } _ { 1 }$ , total-effect indices $\mathrm { S _ { T } } ,$ , and the difference $\mathbf { S } _ { \mathrm { T } } \ - \ \mathbf { S } _ { 1 }$ were reported, with the last used descriptively for interaction and higher-order contributions. The rectangular reference characterizes the fitted model under a synthetic independent-input distribution rather than the experimental design. TreeSHAP was applied to a deterministic attribution surrogate over the same reference space, and mean absolute contributions were normalized across the four inputs. The surrogate was not evaluated on an independent sample. Agreement between SHAP and Sobol therefore indicates consistency under a shared reference distribution, not independent validation. Pearson’s (r) summarized the four-element profiles descriptively without inferential interpretation. Sampling and surrogate settings are provided in the Supplementary Information.

All calculations used fixed random seeds, and reported tables and figures were generated from saved per-observation and mechanism-analysis outputs.

## 4. esults and discussion

## 4.1 rouped generalization performance

LOGO evaluation assessed generalization to the nine withheld experimental groups [25]. Each outer fold excluded one complete group, and the pooled out-of-fold predictions covered all 278 observations. Because process-variable values could overlap across groups, this protocol evaluates group-level generalization rather than formal covariate extrapolation.

The gated hybrid achieved $\mathrm { R } ^ { 2 } { = } 0 . 5 9 2$ , compared with $\mathrm { R } ^ { 2 } { = } 0 . 5 3 8$ for the black-box RF model, 0.564 for the ungated hybrid, and $\mathrm { R } ^ { 2 } { = } 0 . 2 8 7$ for the greybox anchor (Table 3). The 0.054 gain over the black-box model and smaller 0.028 gain over the ungated hybrid indicated that residual learning provided most of the improvement over the anchor, with applicability weighting adding a smaller LOGO benefit. The gated model also reduced MAE and RMSE from 0.933 and 1.229 MUD for the black-box model to 0.856 and 1.156 MUD, while Spearman’s r increased from 0.740 to 0.786.

Table 3. Point-prediction performance of the greybox anchor, black-box RF, ungated hybrid, and gated hybrid under LOGO and the two withheld-defocus evaluations.
<table><tr><td>Model &amp; evaluation regime</td><td>R²</td><td>MAE</td><td>RMSE</td><td>ρ</td></tr><tr><td>Physics-only (Greybox anchor)</td><td></td><td></td><td></td><td></td></tr><tr><td>LOGO generalization</td><td>0.287</td><td>1.223</td><td>1.527</td><td>0.672</td></tr><tr><td>Conduction (+80)</td><td>0.778</td><td>0.558</td><td>0.680</td><td>0.886</td></tr><tr><td>Out-of-envelope (0)</td><td>-0.298</td><td>1.480</td><td>1.883</td><td>0.581</td></tr><tr><td>Black box</td><td></td><td></td><td></td><td></td></tr><tr><td>LOGO generalization</td><td>0.538</td><td>0.933</td><td>1.229</td><td>0.740</td></tr><tr><td>Conduction (+80)</td><td>-0.001</td><td>1.193</td><td>1.444</td><td>0.616</td></tr><tr><td>Out-of-envelope (0)</td><td>-1.092</td><td>1.860</td><td>2.390</td><td>0.191</td></tr><tr><td>Ungated hybrid</td><td></td><td></td><td></td><td></td></tr><tr><td>LOGO generalization</td><td>0.564</td><td>0.891</td><td>1.194</td><td>0.773</td></tr><tr><td>Conduction (+80)</td><td>0.199</td><td>1.096</td><td>1.291</td><td>0.677</td></tr><tr><td>Out-of-envelope (0)</td><td>-0.770</td><td>1.732</td><td>2.199</td><td>0.256</td></tr><tr><td>Gated Hybrid (This work)</td><td></td><td></td><td></td><td></td></tr><tr><td>LOGO generalization</td><td>0.592</td><td>0.856</td><td>1.156</td><td>0.786</td></tr><tr><td>Conduction (+80)</td><td>0.750</td><td>0.588</td><td>0.722</td><td>0.847</td></tr><tr><td>Out-of-envelope (0)</td><td>-0.378</td><td>1.536</td><td>1.940</td><td>0.431</td></tr></table>

These aggregate scores do not establish how the models behave when an entire process subregime is withheld. Figure 4 positions the LOGO, +80 mm, and focus-zero evaluations in the applicability-validity plane, motivating the complementary regime-shift comparisons in Section 4.2.

(b) Data support and physical validity  
![](images/58c840da929f5d5490ee227420e6fde68c0089f74761427568debe54bc967a37.jpg)  
Figure 4. Evaluation design in the applicability-validity plane. (a) $\log _ { 1 0 } ( \mathfrak { q } _ { \mathrm { p D } } / \mathfrak { r } _ { \mathrm { q } } )$ versus focus offset. Marker area represents the number of coincident observations at each of the 20 unique (P, F) positions. (b) The same validity coordinate versus the evaluation-specific normalized five-nearest-neighbor distance $\mathrm { { d } / \mathrm { { d _ { r e f } . } } }$ The horizontal dashed line marks $\mathsf { q } _ { \mathrm { P D } } = \tau _ { \mathsf { q } } ,$ above which predictions are withheld. The blue dashed line marks $\mathrm { d } / \mathrm { d } _ { \mathrm { r e f } } { = } 1$ where residual attenuation begins, and the gray dotted line marks $\scriptstyle \mathrm { { W } } = \tau _ { \mathrm { { w } } }$ $= 0 . 6 0$ corresponding to $\mathrm { d } / \mathrm { d } _ { \mathrm { r e f } } \approx 1 . 5 1 1$ . Colors identify the complete +80 mm holdout (n=42), complete focus-zero holdout (n=54), and remaining LOGO observations (n=182). Because $\mathrm { d } / \mathrm { d } _ { \mathrm { r e f } }$ depends on the training partition, panel (b) uses values from the corresponding LOGO, +80 mm, or focus-zero evaluation; $\mathfrak { q } _ { \mathrm { P D } } / \mathfrak { r } _ { \mathfrak { q } }$ is partition-independent.

## 4.2 eliability under withheld-defocus regime shift

Withholding the complete +80 mm regime (n=42) produced a clearer separation among model classes. All 42 queries were routed to CM by the frozen Stage-1 surrogate. Figure 5 compares parity under LOGO and +80 mm evaluation. The latter probes transfer under weak residualmodel support but a retained greybox anchor, complementing the focus-zero test of abstention beyond the adopted physics-validity criterion.

![](images/089085a2b8b299b0bbe3dc97f7e1123a5e9681a942ddff202d71e9717776589e.jpg)  
Figure 5. Regime-specific parity of measured and predicted e₀₀₁ for the greybox anchor, black-box RF, and gated hybrid. Panels (a-c) show LOGO results; panels (d–f) show the withheld +80 mm evaluation. Dashed lines denote 1:1 agreement. Orange markers identify framework-retained observations, whereas green markers identify focus-zero observations for which the framework abstained. Vertical bars show model-specific 90% inner group-CV empirical intervals where issued. The black-box comparator does not implement abstention; its full-set UQ is reported in Table 7. All axes span 0-12 MUD.

$\mathbf { A } \mathbf { t } \ + 8 0$ mm, the greybox anchor achieved $\mathrm { R } ^ { 2 } { = } 0 . 7 7 8$ , whereas the black-box RF reached $\mathrm { R } ^ { 2 } { = } { - } 0 . 0 0 1$ and the ungated hybrid reached $\scriptstyle \mathrm { R } ^ { 2 } = 0 . 1 9 9$ . The near-zero black-box result was equivalent to the withheld-subset mean baseline under $\mathrm { R } ^ { 2 }$ , while unrestricted residual correction did not preserve the transfer behavior of the anchor. The gated hybrid achieved $\mathrm { R } ^ { 2 } { = } 0 . 7 5 0$ , remaining close to the greybox and substantially outperforming the black-box and ungated models, although it did not improve on the anchor. This pattern is consistent with attenuation of an unsupported residual correction. The anchor component was therefore associated with the observed transfer, and Section 4.3 examines the contribution of its Stage-1 depth mediator.

Focus-zero defined a different boundary. This subset contained 54 observations at the highest $\mathsf { q } _ { \mathrm { P D } }$ values, and the greybox, black-box RF, and gated hybrid produced negative $\mathrm { R } ^ { 2 }$ values of -0.298, -1.092, and -0.378 respectively. Stage-1 nevertheless routed 53 observations to CM and one to KM, so the subset is described as outside the adopted anchor-validity envelope rather than as keyhole or transitional regime [9,12–14]. The mode label and Stage-2 validity decision therefore represent distinct assessments. Because all 54 queries exceeded $\tau _ { \mathfrak { q } } ,$ the framework withheld every operational prediction and interval, giving 0% retention. Focus-zero consequently serves as a failure-case stress test rather than evidence of successful extrapolation.

Because the +80 mm regime contributed to framework development and configuration assessment, it was treated as a controlled withheld-defocus evaluation rather than as independent external validation. Within this entirely CM-routed subset, measured e₀₀₁ reached 7.557 MUD, showing that pronounced enrichment was achievable within the evaluated conduction-oriented domain. However, with only one KM-routed observation among the 278 Stage-2 cases and no matched porosity or defect labels, the data do not support a comparison of defect-free CM texture with defect-associated KM texture.

Reported accuracies from LPBF texture-prediction studies cannot be interpreted as like-forlike rankings because the studies differ in material, target definition, model output, and validation design. Sofras et al. [21] used decision-tree regression, with ten-fold cross-validation for tree pruning and evaluated six newly fabricated control conditions to predict neutrondiffraction-derived texture ratios in 304L stainless steel. Whitney et al. [22] combined partscale mechanistic simulation with a ML surrogate for Ti-6Al-4V microstructure prediction, and its reported surrogate accuracy concerns phase-fraction and lath-width outputs rather than texture. Table S5 in the Supplementary Information compares the reported targets, metrics, computational pathways, and applicability treatments without treating them as directly ranked accuracies.

Quantitative model comparison in the present study was instead performed on the same dataset under identical evaluation protocols (Table 3). Under LOGO, the proposed gated hybrid achieved $\mathrm { R } ^ { 2 } = 0 . 5 9 2$ , compared with 0.538 for the black-box model and 0.287 for the physics anchor. In the +80 mm evaluation, it retained most of the anchor performance (0.750 versus 0.778), whereas the black-box model reached −0.001. Corresponding mean absolute percentage errors are reported in Table S6. The reimplemented and adapted baselines in Table S7 remained below the physics anchor at +80 mm across all predictor sets and across three pruning implementations. Neither representative study in Table S5 reports the query-level combination of residual attenuation and validity-based output withholding implemented here.

As a post hoc diagnostic comparison, literature-derived model forms were reimplemented or adapted to the present dataset and evaluated under identical protocols (Table S7). A single regression tree with cost-complexity pruning, following the procedure of Sofras et al.[21], achieved $( \mathrm { R } ^ { 2 } { = } 0 . 1 9 1 )$ under LOGO and 0.175 on the +80 mm subset using the published predictors of laser power, scan speed, and hatch spacing. Adding focus offset as a datasetspecific adaptation increased these values to 0.302 and 0.297, respectively. Power-law regressions using conventional volumetric and linear energy-density descriptors [48] provided no positive predictive skill under either LOGO or the +80 mm holdout. These descriptors do not account for focus offset; for example, four conditions at 400 W, $6 0 0 \ \mathrm { m m \ s ^ { - 1 } }$ , and $1 5 0 ~ { \mu \mathrm { m } }$ shared a volumetric energy density of $1 4 8 \mathrm { ~ J ~ m m } ^ { - 3 }$ despite measured $\left( \mathrm { e } _ { 0 0 1 } \right)$ values spanning 2.42–7.84 MUD. A normalized-enthalpy descriptor following the dimensionless formulation used for LPBF melting-mode analysis [12], with the adopted IN718 properties and focusdependent beam radius, achieved $\scriptstyle \left( \mathbb { R } ^ { 2 } = 0 . 0 2 8 \right)$ at +80 mm, whereas a reduced geometric surrogate motivated by the melt-pool and thermal-gradient growth framework of Liu et al. [49] achieved 0.013. Across all predictor sets and across three pruning implementations, these baselines remained below the physics anchor at +80 mm.

Together, the holdouts distinguish anchor-supported transfer within the adopted envelope from abstention beyond it. Sections 4.4 and 4.5 examine the corresponding residual-attenuation and physics-validity decisions.

## 4.3 reybox scaling relation and depth-mediator ablation

Log-linear fitting across the 278 observations yielded the coefficients of the $\mathrm { P - V - \hat { D } }$ greybox anchor defined in Equation. 12 (Table 4). The positive exponents for V and P together with the negative exponent for $\hat { \mathrm { D } }$ represent conditional log–log associations within the fitted process envelope. All three 95% experimental-group cluster-bootstrap intervals excluded zero, and the variance inflation factors of the log-transformed predictors were below 2.5 (Table 5). These results indicate stable coefficient signs without severe linear collinearity, but they do not establish universal physical constants or causal effects.

Table 4. Fitted coefficients of the $\mathrm { P - V - } \hat { \mathrm { D } }$ greybox anchor and 95% experimental-group cluster-bootstrap percentile intervals (2,000 replicates).
<table><tr><td rowspan="2">Term</td><td rowspan="2">Point estimate</td><td rowspan="2">Bootstrap median</td><td rowspan="2">95% CI</td><td rowspan="2">Expected</td><td rowspan="2">Interpretation</td></tr><tr><td>sign</td></tr><tr><td>V</td><td>0.296</td><td>0.288</td><td>[0.024, 0.452]</td><td>十</td><td>higher speed associated with stronger e001</td></tr><tr><td>P</td><td>0.407</td><td>0.412</td><td>[0.271, 0.906]</td><td>十</td><td>higher power associated with stronger e001</td></tr><tr><td>D</td><td>-0.629</td><td>-0.592</td><td>[-0.798, -0.318]</td><td></td><td>deeper melt pool associated with weaker e001</td></tr><tr><td>Prefactor A</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.0148</td><td></td><td></td><td>十</td><td> $\mathbf { e } _ { 0 0 1 } ^ { \mathrm { P h y s } } = 0 . 0 1 4 8 ~ \mathrm { V } ^ { 0 . 3 0 } \mathrm { P } ^ { 0 . 4 1 } \hat { \mathrm { D } } ^ { - 0 . 6 3 }$ </td></tr></table>

Bootstrap intervals quantify coefficient stability across group resamples and are not interpreted as universal physical constants.

Table 5. Collinearity diagnostics and leakage-free greybox ablation under LOGO and the withheld +80 mm evaluation.
<table><tr><td>Specification / diagnostic</td><td>VIF / LOGO R²</td><td>+80 mm R²</td><td>Outcome</td></tr><tr><td>VIF (log V)</td><td>1.48</td><td></td><td>below 2.5</td></tr><tr><td>VIF (log P)</td><td>1.41</td><td></td><td>below 2.5</td></tr><tr><td>VIF (log )</td><td>1.06</td><td></td><td>below 2.5</td></tr><tr><td>{V, P, } selected</td><td>0.287</td><td>0.778</td><td>retained anchor</td></tr><tr><td>+ Ev (volumetric energy)</td><td>0.314</td><td>0.747</td><td>small LOGO gain; reduced +80 transfer</td></tr><tr><td>+ H</td><td>0.314</td><td>0.747</td><td>same column-space as Ev; reduced +80</td></tr><tr><td>+ |F|</td><td>0.245</td><td>0.699</td><td>transfer transfer weakened</td></tr><tr><td>V, P only (drop )</td><td>0.059</td><td>-0.116</td><td>transfer failed</td></tr><tr><td>P, D only (drop V)</td><td>0.141</td><td>0.317</td><td>transfer substantially degraded</td></tr></table>

Variance inflation factors were calculated for the log-transformed predictors. Each candidate specification was refitted within the corresponding training partition.

Leakage-free ablation identified D<sup>̂</sup> as central to the transfer behavior of the anchor. Removing D<sup>̂</sup> reduced LOGO R² from 0.287 to 0.059 and the withheld +80 mm R² from 0.778 to −0.116 (Table 5). The model-derived depth therefore carried predictive information under the withheld-defocus evaluation, although this result establishes predictive relevance rather than a causal effect of melt-pool depth.

Adding E<sub>V</sub> or a single multiplicative hatch term produced identical scores because, for fixed layer thickness t, log E<sub>V</sub> = log P − log V − log H − log t. Once P and V are included, these additions span the same column space [15]. Both slightly improved LOGO performance but reduced +80 mm transfer. Adding |F| or removing V also weakened transfer, favoring the parsimonious P-V-D<sup>̂</sup> specification. The result for the single hatch term does not exclude nonlinear or interaction-related hatch effects, which are examined through the learned residual in Section 4.6.

Together, the coefficient and ablation results suggest that D<sup>̂</sup> compresses part of the process-tomelt-pool response into a geometry mediator, while the multiplicative form constrains the anchor to a small set of variables. This structure retained predictive value at +80 mm but remained incomplete under LOGO, where the anchor achieved $\mathrm { R } ^ { 2 } { = } 0 . 2 8 7$ . This ablation establishes the predictive relevance of the frozen Stage-1 mediator within the present workflow; it does not establish a causal effect of melt-pool depth or a universal physical scaling law. Fourteen of the 278 Stage-2 observations lay outside the Stage-1 hatch-spacing range. For these cases, the frozen tree models generated the geometry mediators beyond the observed hatchspacing range; these outputs should not be interpreted as interpolated estimates. Section 4.4 next tests whether applicability weighting can suppress unsupported residual corrections without sacrificing anchor-supported transfer.

## 4.4 Applicability weighting and gate ablation

Because unrestricted residual correction improved LOGO performance but degraded +80 mm transfer, the distance-based applicability weighting defined in Equations 15–19 [26] was evaluated through a leakage-free comparison of three gate configurations. The deployed soft gate achieved $\mathrm { R } ^ { 2 } { = } 0 . 5 9 2$ under LOGO and 0.750 at +80 mm, compared with 0.564 and 0.199 for the ungated hybrid (Table 6, Figure 6a). For the +80 mm queries, the soft gate produced a mean residual attenuation of 80.5%, corresponding to a mean residual weight of approximately 0.195. The hard gate assigned zero residual weight to all +80 mm queries and recovered the standalone anchor result of $\mathrm { R } ^ { 2 } { = } 0 . 7 7 8$ , but its LOGO performance decreased to 0.530. Among the evaluated gates, continuous attenuation therefore provided the most balanced trade-off between grouped generalization and anchor-supported transfer.

Table 6. Gate-type ablation under LOGO grouped generalization and the withheld +80 mm evaluation.
<table><tr><td>Gate configuration</td><td>LOGO R²</td><td>+80 mm R²</td><td>Mean +80 mm attenuation</td></tr><tr><td>Soft gate (deployed)</td><td>0.592</td><td>0.750</td><td>0.805</td></tr><tr><td>Ungated hybrid (plain)</td><td>0.564</td><td>0.199</td><td>0.000</td></tr><tr><td>Hard cutoff</td><td>0.530</td><td>0.778</td><td>1.000</td></tr></table>

Residual attenuation is the +80 mm mean $o f ( I { - } w )$ . The ungated, hard, and soft gates apply full, binary, and exponentially attenuated residual corrections, respectively.

The deployed ${ \tt k } = 5$ , $\mathrm { P } _ { 9 5 }$ configuration was inherited from the frozen canonical implementation and was not selected from the sensitivity sweeps. Although k=3 produced a slightly higher +80 mm $\mathrm { R } ^ { 2 } ,$ the qualitative gate trade-off remained unchanged across the evaluated neighborhood sizes. Larger neighborhoods and the more permissive $\mathrm { P _ { 9 9 } }$ reference radius allowed greater residual correction and reduced +80 mm transfer (Figure 6b, c). These leakage-free refits were used only to assess robustness around the deployed configuration; complete sweep results are provided in the Supplementary Information.

![](images/7c21ed41b1500c624c48191ab8a38fd748bd1882d57287123e694baddd063f50.jpg)

![](images/0f89cc6fc97d9792b083cc11563c5c5635ad73e1b2706bf00f746b9a5855329b.jpg)

![](images/1af8790a8eb3ebabaaa3cb5ebcfdf38ec8423bdcc63fa73a9a97fe6a9651965d.jpg)  
Figure 6. Applicability-gate ablation and sensitivity analysis. (a) LOGO grouped generalization and withheld +80 mm R² for the ungated hybrid, hard gate, and deployed soft gate. (b) Sensitivity of the soft gate to the neighborhood size k. (c) Sensitivity to the reference-distance percentile. The deployed k=5, $\mathrm { P } _ { 9 5 }$ configuration was not selected from these leakage-free robustness sweeps.

The gate ablation shows that applicability weighting controls the magnitude of the learned residual without altering the greybox anchor. The soft gate retained the highest LOGO performance while keeping the +80 mm response close to the standalone anchor. However, the distance weight is neither an uncertainty guarantee nor an independent abstention rule. Section 4.5 evaluates retained-set interval coverage and the separate physics-validity criterion used to withhold unsupported predictions.

## 4.5 roup- V conformal uncertainty and physics-validity abstention

Reliability was evaluated through nominal 90% group-CV conformal intervals for retained predictions and a separate physics-validity abstention rule. The continuous applicability weight w(x) attenuated the learned residual, while $\mathbf { w } { < } \tau _ { \mathrm { w } }$ identified limited data support without changing interval width or independently triggering abstention. Predictions were withheld only when $\mathsf { q } _ { \mathrm { P D } } > \tau _ { \mathsf { q } } ,$ indicating that the greybox anchor was outside its adopted validity envelope. Coverage is therefore reported as empirical retained-set coverage together with retention, rather than as unconditional coverage under distribution shift.

Under LOGO, 224 of the 278 observations were retained, giving 80.6% retention. The gated intervals achieved 92.9% empirical retained-set coverage with a mean full width of 3.65 MUD, 34% narrower than the regime-specific null width of 5.57 MUD (Table 7). The black-box intervals covered 91.0% with a mean width of 4.76 MUD but were evaluated over all 278 observations because the comparator did not abstain. The LOGO coverage values therefore refer to different evaluation populations and should not be compared without considering retention.

Table 7. Empirical interval coverage, mean full width, and retention under LOGO and the two withhelddefocus evaluations.
<table><tr><td>Regime</td><td>Gated empirical retained-set coverage</td><td>Gated width (MUD)</td><td>Gated retention</td><td>Black-box coverage</td><td>Black-box width</td></tr><tr><td>LOGO generalization</td><td>92.9%</td><td>3.65</td><td>80.6%</td><td>91.0%</td><td>4.76</td></tr><tr><td>+80 mm</td><td>100%</td><td>3.21</td><td>100%</td><td>83.3%</td><td>4.39</td></tr><tr><td>Focus-zero</td><td></td><td></td><td>0%</td><td>53.7%</td><td>3.32</td></tr></table>

Gated coverage and width are calculated only for issued intervals; black-box values use the complete evaluation subset because the comparator does not abstain. The dash indicates that no gated interval metric is defined at zero retention.

$\mathbf { A } \mathbf { t } \ + 8 0$ mm, all 42 observations satisfied the physics-validity criterion and were retained, although every query had $\mathbf { W } { < } \tau _ { \mathrm { w } } ,$ , indicating limited residual-model support. The gated intervals achieved 100% empirical coverage with a mean full width of 3.21 MUD, 40% narrower than the regime-specific null width of 5.34 MUD. Because retention was 100%, the gated and blackbox intervals were evaluated over the same observations; the black-box comparator achieved 83.3% coverage with a mean width of 4.39 MUD.

At focus-zero, all 54 observations exceeded $\tau _ { \mathfrak { q } } ,$ so no gated predictions or intervals were issued and retention was 0%. Gated coverage and width are consequently undefined. Without abstention, the black-box intervals had a mean width of 3.32 MUD but covered only 53.7% of the observations. Thus, comparatively narrow intervals did not indicate reliable prediction outside the adopted physics-validity envelope (Figure 7).

(a) Empirical retained-set coverage

(c) Retention and abstention  
![](images/58b1059b7dd0035d7f6f5cd50f63d1743267496ae4fcbff811d6ac771fd3c689.jpg)

(b) Interval width / sharpness  
![](images/62430690754f4d5487b5ce31c4694354c66419d669b2d7417f32fbde2f03c4d2.jpg)

![](images/59d7c53f849b87c82f1f6c33a16a69868eb274b666880ef05fa259329dee8adb.jpg)  
Figure 7. Empirical uncertainty and selective prediction across the evaluation regimes. (a) Coverage of issued gated intervals and full-set black-box intervals; no gated interval was issued at focus-zero. (b) Gated mean full interval width and the regime-specific target-only null width for evaluations with nonempty retained sets. (c) Gated retention. The horizontal reference in panel (a) denotes the nominal 90% coverage level.

For both nonempty retained sets, empirical coverage was numerically above the nominal 90% level while the intervals remained narrower than the corresponding target-only null widths. However, the 100% coverage at +80 mm was obtained from 42 observations under a specific regime shift and is not a theoretical guarantee under arbitrary distribution shifts or violations of exchangeability [30]. As noted in Section 3.4, the physics-validity threshold is a studyspecific rule rather than a prospectively specified constant. Coverage and retention together support an operational assessment within the evaluated protocols. Section 4.6 examines the process variables and interactions represented by the learned residual.

## 4.6 Process-level interpretation of the learned residual

To interpret the fitted residual at the process level, SHAP and Sobol analyses were applied to the complete process-to-residual mapping, with predicted melt-pool depth and width supplied by Stage-1 mediator surrogates [45–47].

Normalized mean absolute SHAP attributed 37% of the residual contribution to scan speed (V),

31% to hatch spacing (H), 27% to focus offset (F), and 4% to power (P) (Figure 8a; rounded values sum to 99%). Scan speed remained the largest contributor despite its explicit inclusion in the greybox, which is consistent with speed-dependent nonlinearities or interactions not represented by the single fitted exponent $\mathrm { V } ^ { 0 . 3 0 }$ , rather than with an omitted process variable.

Under the same uniform process-variable bounds, Sobol analysis ranked hatch spacing $\mathrm { S } _ { 1 } { = } 0 . 3 3 8$ and $\mathrm { S } _ { \mathrm { T } } { = } 0 . 4 4 9$ , and scan speed $\mathrm { S } _ { 1 } { = } 0 . 3 1 1$ and $\mathrm { S } _ { \mathrm { T } } { = } 0 . 4 1 6$ as the leading variables (Figure 8b). Focus offset had $\mathrm { S } _ { 1 } { = } 0 . 1 8 3$ and $\mathrm { S _ { T } } \mathrm { = } 0 . 2 8 6$ , whereas power had a limited total effect of 0.040. Its small negative first-order estimate of −0.002 was treated as numerical variation around zero. The reversal of the leading SHAP and Sobol rankings is expected because mean absolute SHAP summarizes attribution magnitude, whereas Sobol indices allocate output variance under the specified reference distribution.

The interaction indicator $\bf { S } _ { \mathrm { { T } } } \mathrm { { - } } \bf { S } _ { 1 }$ was 0.111 for hatch spacing, 0.105 for scan speed, 0.103 for focus offset, and 0.041 for power (Figure 8a, b). Hatch spacing was therefore the leading variable absent from the greybox under both residual-SHAP attribution and the Sobol interaction profile. This result resolves the apparent tension with Section 4.3: a single global log-linear hatch term did not improve the transfer-oriented greybox, whereas the learned residual represented nonlinear and interaction-dependent hatch behavior. Such behavior is consistent with the effects of hatch spacing and remelting on track overlap, repeated thermal exposure, and competitive grain selection [8,10,11], although the attribution does not identify a specific physical pathway. Because D<sup>̂</sup> and W<sup>̂</sup> entered through model-derived mediators, these indices describe the complete residual forward mapping and do not uniquely separate direct process contributions from those transmitted through Stage 1.

Across the four process variables, the residual-SHAP and Sobol-interaction profiles showed a descriptive magnitude correspondence of (r=0.96) (Figure 8c). Because both analyses used the same synthetic reference bounds and the comparison contained only four values, this correlation is an internal consistency summary rather than inferential evidence, independent validation, or a causal result.

(a) Residual SHAP attribution  
![](images/b197b744a5ab397862f7ab2ee393445d868b951d2ad52d08e4d415dfe880a608.jpg)

(b) Sobol first-order and total effects  
![](images/14333ce183e222fd6565fdca000dc7df3f164d127197f744efcbf43b4e28e7f0.jpg)

(c) Normalized SHAP-Sobol comparison  
![](images/68a14000c9af22a4d5ef7efed059214a3953977340e2dc3def91173a41c26831.jpg)  
Figure 8. Process-level interpretation of the fitted residual forward mapping, with Stage-1 melt pool depth and width treated as model-derived mediators. (a) Normalized mean absolute SHAP attributions. (b) Sobol first-order (S<sub>1</sub>) and total-effect (S<sub>T</sub>) indices under the shared uniform process-variable bounds. (c) Independently normalized comparison of the SHAP attribution and Sobol interaction term $\left( \mathbf { S } _ { \mathrm { T } }  – \mathbf { S } _ { \mathrm { 1 } } \right)$ Pearson’s (r=0.96) was calculated from the unnormalized four-variable profiles and is reported descriptively.

Together, the two analyses associate the learned residual with nonlinear speed dependence and hatch- and focus-related interactions omitted from the parsimonious greybox. Because this structure was learned from the sampled process space, applicability weighting remains necessary when the residual model is queried under limited data support. Hatch spacing was the leading process variable absent from the anchor, but this does not conflict with the ablation in Section 4.3. A single multiplicative hatch term was log-linearly equivalent to the tested energy-density extension, whereas the residual learner could represent nonlinear and interaction-dependent hatch behavior. Because $\hat { \mathrm { D } }$ and $\hat { \mathbf W }$ entered as Stage-1 mediators, the process-level sensitivities do not uniquely separate direct from mediated contributions. Section 5 applies the retained texture predictions in an illustrative literature-anchored mapping to builddirection elastic modulus.

## 5. Illustrative application to build-direction elastic modulus

This section presents two complementary analyses that connect ⟨001⟩ ∥ BD enrichment to property-scale behavior. Section 5.1 applies a bounded, literature-informed texture-to-modulus mapping to the retained predictions and a separately built set. Section 5.2 uses tensile responses from the same specimens to examine whether the assumed inverse association between enrichment and directional stiffness persists at room temperature and at $6 5 0 ~ ^ { \circ } \mathrm { C }$ . This experimental comparison supports the direction of the mapping, not its numerical calibration.

## 5.1 Texture-to-modulus mapping and application

Build-direction Young's modulus, E<sub>BD</sub>, was selected to provide a property-scale interpretation of changes in ⟨001⟩ ∥ BD enrichment. Elastic stiffness in cubic nickel-based alloys is determined by orientation through the single-crystal stiffness tensor [6,7], so texture is a firstorder control on modulus. Strength is instead governed primarily by the $\gamma ^ { \prime \prime }$ and γ′ precipitation state [50], with grain size and porosity as further contributors, and is therefore not mapped here; a direct check of this separation on the present specimens is reported in Section 5.2.

Directional stiffness is a design-relevant quantity in components subject to constrained thermal cycling, where the stress developed under a given thermal strain scales with the modulus along the constrained direction. LPBF IN718 is used for hot-section components in aero-engines and turbomachinery [51], and directional stiffness has been exploited at the design stage to reduce machining-induced deformation in thin-walled LPBF parts[3], and because epitaxial growth follows the build direction [52], the build orientation chosen at the design stage fixes the texture the part will carry. Whether a lower build-direction modulus is desirable is application dependent and is not claimed here.

$\mathrm { E _ { B D } }$ was defined as a monotone nonincreasing function of the predicted texture intensity. [6,7]:

$$
\mathrm { E _ { B D } } = \mathrm { c l i p } \left[ 1 9 5 - 1 0 . 2 ( \hat { \mathrm { e } } _ { 0 0 1 } - 1 ) , 1 2 5 , 1 9 5 \right]\tag{28}
$$

where ê₀₀₁ is the predicted ⟨001⟩∥BD texture intensity. A slope of 10.2 GPa per unit ê₀₀₁ was adopted, which places the lower saturation bound at $\hat { \mathrm { e } } _ { 0 0 1 } \approx 7 . 9$ close to the maximum value observed in the main dataset. This is of the same order as the gradient implied by reported build-direction moduli for LPBF IN718 at two texture levels, which differ by 25 GPa across a change in ⟨001⟩ pole-figure intensity of about 2.8 [3]. The two texture measures are defined differently, and the comparison is indicative only. The clip operator restricts $\mathrm { E _ { B D } }$ to 125–195 GPa.

Let g denote the clipped mapping in Equation 28. Because g is monotone nonincreasing, a texture interval $[ \mathrm { L } _ { \mathrm { e } } , \mathrm { U } _ { \mathrm { e } } ]$ propagates to

$$
\mathrm { [ L _ { E } , U _ { E } ] = [ g ( U _ { e } ) , g ( L _ { e } ) ] }\tag{29}
$$

The bounded, literature-informed relation in Equation 28 was applied to the 224 retained LOGO texture predictions to illustrate downstream property interpretation. For cubic IN718, stronger predicted ⟨001⟩ ∥ BD texture maps to a lower build-direction elastic modulus E<sub>BD</sub> consistent with the expected elastic anisotropy [3–8,53]. The mapped values ranged from 126.9 to 186.6 GPa, reported as approximately 127-187 GPa in Figure 9. Only retained texture predictions and their propagated intervals were mapped; no modulus was reported for the abstained focus-zero observations.

The 90% texture intervals were propagated through the same monotonic relation using Equation 29. The mean pre-clipping modulus half-span across the retained set was 18.6 GPa, although it varied among observations because the texture intervals were fold-dependent. Clipping at the prescribed 125 and 195 GPa bounds also made some propagated intervals asymmetric. These intervals represent texture-prediction uncertainty propagated through the assumed mapping; they do not include uncertainty in the mapping form or material constants. This transformation propagates retained texture predictions and intervals to a property-oriented quantity while preserving abstention.

A separate set of specimens was subsequently built and machined, and their ⟨001⟩ ∥ BD enrichment was measured on the tensile bars themselves. Neither their process settings nor their measurements entered model development. The framework was applied to these nine conditions with all fitted parameters and both gate thresholds held at their canonical values. The framework separated these nine conditions into three groups (Table 8). Three exceeded the adopted anchor-validity threshold by a factor of four and were withheld without further prediction or interval. Three lay at normalized applicability distances near 3.1, where the learned correction was attenuated to about 12% of its unweighted value, leaving predictions within 0.07 MUD of the physics anchor alone. The remaining three were issued with a mean absolute error of 1.02 MUD, or 8 to 17% of the measured value. Mapped through Equation 28, these three gave build-direction moduli within 8.7 GPa on average of the values implied by the measured enrichments, and all three propagated intervals contained the measured value (Figure 9). Across the six conditions for which predictions were issued or attenuated, the predicted and measured enrichments ranked identically. Two of the nine conditions also appear in the main dataset and gave enrichments of 8.38 and 6.47 against 7.84 and 2.42 measured previously on separately built specimens, indicating that nominally identical process settings need not reproduce the same texture level across builds.

Table 8. Framework decisions on nine conditions from a separately built set. Values in bold exceed the corresponding threshold: the anchor-validity criterion withholds a prediction when ${ \bf q } _ { \mathrm { P D } }$ exceeds $\tau _ { \mathfrak { q } } ,$ and the applicability weighting attenuates the learned correction when the normalized distance exceeds the value corresponding to $\tau _ { \mathrm { w } } .$ No prediction was issued for the withheld conditions. For the attenuated conditions the correction was reduced to about 12% of its unweighted value, so the reported values are close to the physics anchor alone. Measured enrichments were obtained on specimens that took no part in model development.
<table><tr><td>Condition (P-S-H-F)</td><td> $\bf q \mathrm  p \} t _ { 9 }$ </td><td> $\mathbf { d } / \mathbf { d _ { \mathrm { r e f } } }$ </td><td>W</td><td>Decision</td><td>Measured e001</td><td>Predicted C001</td></tr><tr><td>400-400-150-0</td><td>3.980</td><td>0.99</td><td>1.00</td><td>withheld</td><td>3.49</td><td>N/A</td></tr><tr><td>400-400-250-0</td><td>3.980</td><td>0.87</td><td>1.00</td><td>withheld</td><td>2.99</td><td>N/A</td></tr><tr><td>400-600-150-0</td><td>3.980</td><td>0.35</td><td>1.00</td><td>withheld</td><td>6.47</td><td>N/A</td></tr><tr><td>400-400-150-(-80)</td><td>0.010</td><td>3.11</td><td>0.12</td><td>attenuated</td><td>4.92</td><td>3.79</td></tr><tr><td>400-400-250-(-80)</td><td>0.010</td><td>3.11</td><td>0.12</td><td>attenuated</td><td>1.15</td><td>3.67</td></tr><tr><td>400-600-150-(-80)</td><td>0.010</td><td>3.04</td><td>0.12</td><td>attenuated</td><td>7.04</td><td>4.20</td></tr><tr><td>400-400-150-(+20)</td><td>0.153</td><td>0.50</td><td>1.00</td><td>issued</td><td>7.43</td><td>6.15</td></tr><tr><td>400-400-250-(+20)</td><td>0.153</td><td>0.56</td><td>1.00</td><td>issued</td><td>7.07</td><td>5.99</td></tr><tr><td>400-600-150-(+20)</td><td>0.153</td><td>0.24</td><td>1.00</td><td>issued</td><td>8.38</td><td>7.69</td></tr></table>

![](images/f3655d8f474bb745078ca603e063f93f9c9f04baac509bea92f294fa50f56d1e.jpg)  
Figure 9. Illustrative mapping from ⟨001⟩ ∥ BD enrichment to build-direction elastic modulus, and its application to a separately built set. The orange background marks the retained texture range and the blue band the propagated empirical 90% intervals of the 224 retained LOGO predictions. For the separately built set, filled circles are the three predictions that satisfied both gate criteria, open squares are the moduli obtained by applying the same mapping to the enrichment measured on the corresponding specimens, and arrows connect each pair. Their error bars were propagated from the median retained-LOGO texture half-width; all three contained the measured value. The withheld and attenuated conditions are not shown.ss

## 5.2 irectional response at service temperature

Because the mapping in Equation 28 rests on literature values rather than on measurements from these specimens, its assumed orientation dependence was examined directly. The design motivation in Section 5.1 concerns components held at elevated temperature, so the measurement was made at $6 5 0 ^ { \circ } \mathrm { C }$ as well as at room temperature. Neutron-diffraction work on IN718 reports that stiffness falls with temperature, from about 220 GPa at room temperature to 140 GPa at $7 0 0 ~ ^ { \circ } \mathrm { C } ,$ while the degree of elastic anisotropy increases, and that the elastic constants are essentially unchanged by ageing [54]. The orientation dependence should therefore persist at service temperature rather than diminish, and any change in it would not be attributable to precipitation state.

The same specimens were tested in tension at room temperature and at $6 5 0 ~ ^ { \circ } \mathrm { C } .$ , so that texture and mechanical response were obtained from the same bar (Table 9). Apparent build-direction stiffness fell from 57 to 29 GPa at room temperature as the measured enrichment increased from 1.15 to 8.38 MUD, and from 46 to 28 GPa over the same specimens at $6 5 0 ~ ^ { \circ } \mathrm { C } ;$ the rank correlations with enrichment were −0.85 and −0.78. The ordering persisted within fixed focusoffset subsets, and partial rank correlations controlling for focus offset or volumetric energy density gave −0.79 and −0.86. The direction of the association is the one assumed in Equation 28, and it holds at both temperatures. The 0.2% proof stress did not follow the enrichment at room temperature, consistent with strength being set principally by the precipitation state rather than by orientation.

Table 9. Measured ⟨001⟩ ∥ BD enrichment and tensile response of the nine specimens, ordered by enrichment. Apparent stiffness was obtained from crosshead displacement and includes machine and fixture compliance; it is interpreted as a relative measure only. The last row gives Spearman rank correlations with the measured enrichment. Stiffness follows the enrichment ordering at both temperatures, whereas the proof stress at room temperature does not.
<table><tr><td>Condition (P-S-H-F)</td><td>C001</td><td> $\mathbf { E _ { a p p } }$  RM</td><td> $\mathbf { E _ { a p p } }$  650</td><td>YS RM</td><td>YS 650</td></tr><tr><td>400-400-150-0</td><td>3.49</td><td>52.0</td><td>46.1</td><td>614</td><td>506</td></tr><tr><td>400-400-250-0</td><td>2.99</td><td>53.1</td><td>33.8</td><td>598</td><td>540</td></tr><tr><td>400-600-150-0</td><td>6.47</td><td>35.1</td><td>32.9</td><td>616</td><td>507</td></tr><tr><td>400-400-150-(-80)</td><td>4.92</td><td>40.8</td><td>29.0</td><td>584</td><td>496</td></tr><tr><td>400-400-250-(-80)</td><td>1.15</td><td>57.1</td><td>37.4</td><td>655</td><td>549</td></tr><tr><td>400-600-150-(-80)</td><td>7.04</td><td>36.4</td><td>32.7</td><td>626</td><td>503</td></tr><tr><td>400-400-150-(+20)</td><td>7.43</td><td>37.2</td><td>30.1</td><td>612</td><td>476</td></tr><tr><td>400-400-250-(+20)</td><td>7.07</td><td>36.3</td><td>31.2</td><td>574</td><td>496</td></tr><tr><td>400-600-150-(+20)</td><td>8.38</td><td>29.0</td><td>27.7</td><td>634</td><td>507</td></tr><tr><td>Spearman ρ with eoo1</td><td></td><td>-0.85</td><td>-0.78</td><td>-0.02</td><td>-0.58</td></tr></table>

Strain was estimated from crosshead displacement normalized by the nominal gauge length, and the specimens were mounted through threaded end sections, so the apparent values include machine and fixture compliance. Following the recommendation of the collaborators who performed the tests, they are interpreted as a semi-quantitative measure of relative trend among processing conditions rather than as elastic moduli. The observed variation may also reflect differences in porosity or grip conditions that were not measured.

Two constraints apply to the separately built set. Three of its conditions used a negative focus offset, which lies outside the deployment domain of the present study; the validity criterion depends on the magnitude of the focus offset and not its sign, so those conditions were flagged by the applicability weighting rather than by the validity threshold. One condition reached an enrichment of 8.38 MUD, above the value at which the mapping saturates, so its mapped modulus lies at the lower bound in Figure 9.

## 6. onclusions

This study developed a framework for predicting ⟨001⟩ ∥ build direction (BD) texture intensity with uncertainty quantification in laser powder bed fusion (LPBF). The framework contains two stages: A frozen Stage-1 surrogate mapped process parameters to melt-pool geometry, and Stage 2 combined a conduction-inspired $\mathrm { P - V - \hat { D } }$ greybox anchor with residual learning, continuous applicability attenuation, physics-validity abstention, and retained-set empirical uncertainty quantification.

Under leave-one-group-out (LOGO)grouped generalization, the gated hybrid reached $\mathrm { R } ^ { 2 }$ of 0.592, compared with 0.538 for the black-box random forest (RF) model. When the complete +80 mm regime was withheld, the greybox and gated hybrid retained $\mathrm { R } ^ { 2 } { = } 0 . 7 7 8$ and 0.750, respectively, whereas the black-box RF reached −0.001. The empirical anchor was therefore the principal source of transferable predictive skill in this evaluation. At focus-zero, all texture models produced negative R², and all 54 queries exceeded the adopted physics-validity threshold, so predictions and intervals were withheld. For issued predictions, the empirical 90% intervals achieved 92.9% coverage at 80.6% retention under LOGO and 100% coverage at full retention for the 42 +80 mm observations. These retained-set values do not constitute a general coverage guarantee under regime shift. The depth ablation associated the +80 mm transfer with the Stage-1 geometry mediator, while Shapley additive explanation (SHAP) and Sobol analyses suggested nonlinear and interaction-dependent hatch behavior outside the parsimonious greybox. Pronounced texture enrichment was observed within conduction-mode (CM) routed conditions, including the +80 mm subset, although the available data do not support a comparative CM versus keyhole-mode (KM) defect claim. A first check on a separate build is reported in Section 5. Broader validation should extend to additional machines and to strain measured with an extensometer, with matched melt-pool, defect, and mechanicalproperty data. Reliable use under distribution shift therefore required a transferable anchor, controlled residual correction, explicit validity assessment, and transparent retained-set uncertainty rather than model complexity alone.

## ediT authorship contribution statement

isheng Lu: Conceptualization, Methodology, Software, Formal analysis, Data curation, Writing – original draft, Writing – review & editing, Visualization. John iris: Investigation, Resources, Data curation, Writing – review & editing. Jie Song: Conceptualization, Methodology, Resources, Supervision, Writing – review & editing, Funding acquisition. ao Fu: Conceptualization, Methodology, Resources, Supervision, Writing – review & editing, Funding acquisition. Jie hen: Conceptualization, Methodology, Supervision, Writing – review & editing, Funding acquisition, Project administration.

## eclaration of ompeting Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

Y.L. and J.C. acknowledge the support from the startup fund, the 4-VA grant, and the graduate assistantship provided by Virginia Tech, and the Curriculum Development Support provided by MathWorks. Y.F. acknowledges support from the National Science Foundation (Award No. 2104941 and Award No. 2245107) for providing financial support. This work utilized the Nanoscale Characterization and Fabrication Laboratory, a part of the National Nanotechnology Coordinated Infrastructure (NNCI), funded by NSF (ECCS 1542100 and ECCS 2025151).

## ata availability

The processed data and analysis code supporting this study are available from the corresponding author upon reasonable request. These comprise the finalized Stage-2 target table, frozen Stage-1 melt-pool predictions, saved per-observation Stage-2 predictions, uncertainty and mechanism-analysis outputs, and the associated analysis scripts. The target table includes the orientation-point fractions and normalized e₀₀₁ values, and recomputation agrees within 0.003 MUD, reflecting stored decimal precision. The original EBSD orientation maps and the associated crystallographic-processing information are available on the same basis, subject to approval from the data owners.

## eferences

[1] T. DebRoy, H.L. Wei, J.S. Zuback, T. Mukherjee, J.W. Elmer, J.O. Milewski, A.M. Beese, A. Wilson-Heid, A. De, W. Zhang, Additive manufacturing of metallic components – Process, structure and properties, Progress in Materials Science 92 (2018) 112–224. https: doi.org 10.1016 j.pmatsci.2017.10.001.

[2] N. Kouraytem, X. Li, W. Tan, B. Kappes, A.D. Spear, Modeling process–structure– property relationships in metal additive manufacturing: a review on physics-driven versus data-driven approaches, J. Phys. Mater. 4 (2021) 032002. https: doi.org 10.1088 2515- 7639 abca7b.

[3] J.D. Pérez-Ruiz, F. Marin, S. Martínez, A. Lamikiz, G. Urbikain, L.N. López de Lacalle, Stiffening near-net-shape functional parts of Inconel 718 LPBF considering material anisotropy and subsequent machining issues, Mechanical Systems and Signal Processing 168 (2022) 108675. https: doi.org 10.1016 j.ymssp.2021.108675.

[4] C. Kumara, D. Deng, J. Moverare, P. Nylén, Modelling of anisotropic elastic properties in alloy 718 built by electron beam melting, Materials Science and Technology 34 (2018) 529–537. https: doi.org 10.1080 02670836.2018.1426258.

[5] B. Rehmer, F. Bayram, L.A. Ávila Calderón, G. Mohr, B. Skrotzki, Elastic modulus data for additively and conventionally manufactured variants of Ti-6Al-4V, IN718 and AISI 316 L, Sci Data 10 (2023) 474. https: doi.org 10.1038 s41597-023-02387-6.

[6] T. Obermayer, C. Krempaszky, E. Werner, Analysis of Texture and Anisotropic Elastic Properties of Additively Manufactured Ni-Base Alloys, Metals 12 (2022) 1991. https: doi.org 10.3390 met12111991.

[7] J. Schröder, A. Heldmann, M. Hofmann, A. Evans, W. Petry, G. Bruno, Determination of diffraction and single-crystal elastic constants of laser powder bed fused Inconel 718, Materials Letters 353 (2023) 135305. https: doi.org 10.1016 j.matlet.2023.135305.

[8] O. Gokcekaya, T. Ishimoto, S. Hibino, J. Yasutomi, T. Narushima, T. Nakano, Unique crystallographic texture formation in Inconel 718 by laser powder bed fusion and its effect on mechanical anisotropy, Acta Materialia 212 (2021) 116876. https: doi.org 10.1016 j.actamat.2021.116876.

[9] C. Zhao, B. Shi, S. Chen, D. Du, T. Sun, B.J. Simonds, K. Fezzaa, A.D. Rollett, Laser melting modes in metal powder bed fusion additive manufacturing, Rev. Mod. Phys. 94 (2022) 045002. https: doi.org 10.1103 RevModPhys.94.045002.

[10] P.V. Cobbinah, S. Matsunaga, Y. Yamabe-Mitarai, Controlled Crystallographic Texture Orientation in Structural Materials Using the Laser Powder Bed Fusion Process—A Review, Advanced Engineering Materials 25 (2023) 2300819. https: doi.org 10.1002 adem.202300819.

[11] W. Shao, B. He, C. Qiu, Z. Li, Effect of hatch spacing and laser remelting on the formation of unique crystallographic texture of IN718 superalloy fabricated via laser powder bed fusion, Optics & Laser Technology 156 (2022) 108609. https: doi.org 10.1016 j.optlastec.2022.108609.

[12] W.E. King, H.D. Barth, V.M. Castillo, G.F. Gallegos, J.W. Gibbs, D.E. Hahn, C. Kamath,

A.M. Rubenchik, Observation of keyhole-mode laser melting in laser powder-bed fusion additive manufacturing, Journal of Materials Processing Technology 214 (2014) 2915– 2925. https: doi.org 10.1016 j.jmatprotec.2014.06.005.

[13] Keyhole threshold and morphology in laser melting revealed by ultrahigh-speed x-ray imaging | Science, (n.d.). https: www.science.org doi 10.1126 science.aav4687 (accessed July 26, 2026).

[14] C. Zhao, N.D. Parab, X. Li, K. Fezzaa, W. Tan, A.D. Rollett, T. Sun, Critical instability at moving keyhole tip generates porosity in laser melting, Science 370 (2020) 1080–1086. https: doi.org 10.1126 science.abd1587.

[15] U. Scipioni Bertoli, A.J. Wolfer, M.J. Matthews, J.-P.R. Delplanque, J.M. Schoenung, On the limitations of Volumetric Energy Density as a design parameter for Selective Laser Melting, Materials & Design 113 (2017) 331–340. https: doi.org 10.1016 j.matdes.2016.10.037.

[16] P. Akbari, F. Ogoke, N.-Y. Kao, K. Meidani, C.-Y. Yeh, W. Lee, A. Barati Farimani, MeltpoolNet: Melt pool characteristic prediction in Metal Additive Manufacturing using machine learning, Additive Manufacturing 55 (2022) 102817. https: doi.org 10.1016 j.addma.2022.102817.

[17] T. Moges, Z. Yang, K. Jones, S. Feng, P. Witherell, Y. Lu, Hybrid Modeling Approach for Melt-Pool Prediction in Laser Powder Bed Fusion Additive Manufacturing, J. Comput. Inf. Sci. Eng 21 (2021). https: doi.org 10.1115 1.4050044.

[18] L. Scime, J. Beuth, A multi-scale convolutional neural network for autonomous anomaly detection and classification in a laser powder bed fusion additive manufacturing process, Additive Manufacturing 24 (2018) 273–286. https: doi.org 10.1016 j.addma.2018.09.034.

[19] Z. Snow, B. Diehl, E.W. Reutzel, A. Nassar, Toward in-situ flaw detection in laser powder bed fusion additive manufacturing through layerwise imagery and machine learning, Journal of Manufacturing Systems 59 (2021) 12–26. https: doi.org 10.1016 j.jmsy.2021.01.008.

[20] Feature-based volumetric defect classification in metal additive manufacturing | Nature Communications, (n.d.). https: www.nature.com articles s41467-022-34122-x (accessed July 29, 2026).

[21] C. Sofras, J. Čapek, C. Leinenbach, R.E. Logé, M. Strobl, E. Polatidis, Exploring crystallographic texture manipulation in stainless steels via laser powder bed fusion: insights from neutron diffraction and machine learning, Virtual and Physical Prototyping 19 (2024) e2390483. https: doi.org 10.1080 17452759.2024.2390483.

[22] B.C. Whitney, A.G. Spangenberger, T.M. Rodgers, D.A. Lados, Part-scale microstructure prediction for laser powder bed fusion Ti-6Al-4V using a hybrid mechanistic and machine learning model, Additive Manufacturing 94 (2024) 104500. https: doi.org 10.1016 j.addma.2024.104500.

[23] Z. Wang, P. Liu, Y. Ji, S. Mahadevan, M.F. Horstemeyer, Z. Hu, L. Chen, L.-Q. Chen, Uncertainty Quantification in Metallic Additive Manufacturing Through Physics-Informed Data-Driven Modeling, JOM 71 (2019) 2625–2634.

https: doi.org 10.1007 s11837-019-03555-z.

[24] M. Faegh, S. Ghungrad, J.P. Oliveira, P. Rao, A. Haghighi, A review on physics-informed machine learning for process-structure-property modeling in additive manufacturing, Journal of Manufacturing Processes 133 (2025) 524–555. https: doi.org 10.1016 j.jmapro.2024.11.066.

[25] D.R. Roberts, V. Bahn, S. Ciuti, M.S. Boyce, J. Elith, G. Guillera-Arroita, S. Hauenstein, J.J. Lahoz-Monfort, B. Schröder, W. Thuiller, D.I. Warton, B.A. Wintle, F. Hartig, C.F. Dormann, Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure, Ecography 40 (2017) 913–929. https: doi.org 10.1111 ecog.02881.

[26] Identifying domains of applicability of machine learning models for materials science | Nature Communications, (n.d.). https: www.nature.com articles s41467-020-17112-9 (accessed July 26, 2026).

[27] L.E. Schultz, Y. Wang, R. Jacobs, D. Morgan, A general approach for determining applicability domain of machine learning models, Npj Comput Mater 11 (2025) 95. https: doi.org 10.1038 s41524-025-01573-x.

[28] J. Lei, M. G’Sell, A. Rinaldo, R.J. Tibshirani, L. Wasserman, Distribution-Free Predictive Inference For Regression, (2017). https: doi.org 10.48550 arXiv.1604.04173.

[29] R.J. Tibshirani, R.F. Barber, E.J. Candes, A. Ramdas, Conformal Prediction Under Covariate Shift, (2020). https: doi.org 10.48550 arXiv.1904.06019.

[30] R.F. Barber, E.J. Candes, A. Ramdas, R.J. Tibshirani, Conformal prediction beyond exchangeability, (2023). https: doi.org 10.48550 arXiv.2202.13415.

[31] I. Gibbs, J.J. Cherian, E.J. Candès, Conformal Prediction With Conditional Guarantees, (2024). https: doi.org 10.48550 arXiv.2305.12616.

[32] R. Dunn, L. Wasserman, A. Ramdas, Distribution-Free Prediction Sets for Two-Layer Hierarchical Models, (2022). https: doi.org 10.48550 arXiv.1809.07441.

[33] V. Vovk, D. Lindsay, I. Nouretdinov, A. Gammerman, Mondrian Confidence Machine, (2003). https: pure.royalholloway.ac.uk en publications mondrian-confidence-machine (accessed July 26, 2026).

[34] M.C. Sow, T. De Terris, O. Castelnau, Z. Hamouche, F. Coste, R. Fabbro, P. Peyre, Influence of beam diameter on Laser Powder Bed Fusion (L-PBF) process, Additive Manufacturing 36 (2020) 101532. https: doi.org 10.1016 j.addma.2020.101532.

[35] J. Metelkova, Y. Kinds, K. Kempen, C. de Formanoir, A. Witvrouw, B. Van Hooreweder, On the influence of laser defocusing in Selective Laser Melting of 316L, Additive Manufacturing 23 (2018) 161–169. https: doi.org 10.1016 j.addma.2018.08.006.

[36] K.C. Mills, Recommended Values of Thermophysical Properties for Selected Commercial Alloys, Woodhead Publishing, 2002.

[37] L. Breiman, Random Forests, Machine Learning 45 (2001) 5–32. https: doi.org 10.1023 A:1010933404324.

[38] T. Chen, C. Guestrin, XGBoost: A Scalable Tree Boosting System, in: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Association for Computing Machinery, New York, NY, USA, 2016: pp. 785–

794. https: doi.org 10.1145 2939672.2939785.

[39] L. Prokhorenkova, G. Gusev, A. Vorobev, A.V. Dorogush, A. Gulin, CatBoost: unbiased boosting with categorical features, in: Proceedings of the 32nd International Conference on Neural Information Processing Systems, Curran Associates Inc., Red Hook, NY, USA, 2018: pp. 6639–6649. https: dl.acm.org doi 10.5555 3327757.3327770 (accessed July 26, 2026).

[40] C.E. Rasmussen, C.K.I. Williams, Gaussian Processes for Machine Learning, n.d. https: direct.mit.edu books oa-monograph 2320 Gaussian-Processes-for-Machine-Learning (accessed July 26, 2026).

[41] M.C. Kennedy, A. O’Hagan, Bayesian calibration of computer models, Journal of the Royal Statistical Society: Series B (Statistical Methodology) 63 (2001) 425–464. https: doi.org 10.1111 1467-9868.00294.

[42] L. Gao, K. Mumm, Z. Ren, Z. Yu, X. Li, C.A. Chuang, T. Sun, Operando X-ray scattering reveals ordering-mediated solidification in additive manufacturing, Nat Commun 17 (2026) 6757. https: doi.org 10.1038 s41467-026-73647-3.

[43] Y. Geifman, R. El-Yaniv, Selective Classification for Deep Neural Networks, (2017). https: doi.org 10.48550 arXiv.1705.08500.

[44] C.A. Field, A.H. Welsh, Bootstrapping Clustered Data, J. R. Stat. Soc. Ser. B. Stat. Methodol. 69 (2007) 369–390. https: doi.org 10.1111 j.1467-9868.2007.00593.x.

[45] I.M. Sobol′, Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates, Mathematics and Computers in Simulation 55 (2001) 271–280. https: doi.org 10.1016 S0378-4754(00)00270-6.

[46] S.M. Lundberg, S.-I. Lee, A Unified Approach to Interpreting Model Predictions, in: Advances in Neural Information Processing Systems, Curran Associates, Inc., 2017. https: proceedings.neurips.cc paper 2017 hash 8a20a8621978632d76c43dfd28b67767- Abstract.html (accessed July 26, 2026).

[47] A. Saltelli, P. Annoni, I. Azzini, F. Campolongo, M. Ratto, S. Tarantola, Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index, Computer Physics Communications 181 (2010) 259–270. https: doi.org 10.1016 j.cpc.2009.09.018.

[48] F. Caiazzo, V. Alfieri, G. Casalino, On the Relevance of Volumetric Energy Density in the Investigation of Inconel 718 Laser Powder Bed Fusion, Materials 13 (2020) 538. https: doi.org 10.3390 ma13030538.

[49] Quantitative Texture Prediction of Epitaxial Columnar Grains in Alloy 718 Processed by Additive Manufacturing Springer Nature Link, (n.d.). https: link.springer.com chapter 10.1007 978-3-319-89480-5\_49 (accessed August 6, 2026).

[50] J.M. Oblak, D.F. Paulonis, D.S. Duvall, Coherency strengthening in Ni base alloys hardened by DO22 γ′ precipitates, Metall Trans 5 (1974) 143–153. https: doi.org 10.1007 BF02642938.

[51] A. Marques, Â. Cunha, M.R. Silva, M.I. Osendi, F.S. Silva, Ó. Carvalho, F. Bartolomeu, Inconel 718 produced by laser powder bed fusion: an overview of the influence of

processing parameters on microstructural and mechanical properties, Int J Adv Manuf Technol 121 (2022) 5651–5675. https: doi.org 10.1007 s00170-022-09693-0.

[52] A. Kaletsch, S. Qin, C. Broeckmann, Influence of Different Build Orientations and Heat Treatments on the Creep Properties of Inconel 718 Produced by PBF-LB, Materials 16 (2023) 4087. https: doi.org 10.3390 ma16114087.

[53] Calculating anisotropic physical properties from texture data using the MTEX opensource package | Geological Society, London, Special Publications, (n.d.). https: www.lyellcollection.org doi full 10.1144 SP360.10 (accessed July 26, 2026).

[54] P.E. Aba-Perea, T. Pirling, P.J. Withers, J. Kelleher, S. Kabra, M. Preuss, Determination of the high temperature elastic properties and diffraction elastic constants of Ni-base superalloys, Materials & Design 89 (2016) 856–863. https: doi.org 10.1016 j.matdes.2015.09.152.

## Supplementary Information for Physics-based Prediction, uncertainty quantification and decision-making for IN718 crystallographic texture intensity across LPBF defocus regimes

Yisheng Lu, John Riris, Jie Song, Yao Fu, Jie Chen

This document provides the sampling and surrogate settings used for the residual-model interpretation analyses, the complete applicability-gate sensitivity sweep, an independent verification of the texture-target reference fraction, the correspondence between the manuscript group labels and the acquisition identifiers used in the distributed project archive, a scopeaware comparison with representative texture-prediction frameworks, mean absolute percentage errors for the four point predictors, and protocol-matched baselines constructed from literature-derived model forms. All are referenced from the main text. All calculations used deterministic settings and fixed random seeds.

## S1. Sampling and surrogate settings for the residual-model interpretation

Residual-model interpretation was carried out at the controllable-process level using Shapley additive explanations (SHAP) and Sobol sensitivity analysis. The four controllable process variables (laser power P, scan speed V, hatch spacing H, and focus offset F) were treated as independent inputs. The two Stage-1 melt-pool mediators required by the six-feature residual model were supplied by deterministic surrogates so that the analysis could be aggregated at the process level rather than at the feature level.

For each sampled process vector, the depth and width mediator surrogates approximated the frozen Stage-1 outputs from P, V, and F; the six-feature residual model was then evaluated on the resulting feature vector. The mediator surrogates were fitted to the frozen Stage-1 outputs available for the 278 Stage-2 observations and were not evaluated on an independent sample. This forward mapping was the target of both the Sobol and the SHAP analyses.

Table S1. Sampling and surrogate settings for the SHAP and Sobol analyses. RF denotes random forest.
<table><tr><td>Setting</td><td>Value</td><td>Note</td></tr><tr><td>Sobol sampling design</td><td>Saltelli</td><td>SALib implementation</td></tr><tr><td>Sobol base sample size N</td><td>2,048</td><td>20,480 model evaluations for four inputs</td></tr><tr><td>Attribution-surrogate training sample</td><td>4,000</td><td>uniform samples over the common bounds</td></tr><tr><td>TreeSHAP explanation subset</td><td>1,500</td><td>first 1,500 of the same 4,000 samples</td></tr><tr><td>Residual model</td><td>RF, 400 trees</td><td>full-data six-feature residual model used for interpretation</td></tr><tr><td>Depth mediator surrogate</td><td>RF, 300 trees</td><td>inputs P, V, F; target Stage-1 predicted depth</td></tr><tr><td>Width mediator surrogate</td><td>RF, 300 trees</td><td>inputs P, V, F; target Stage-1 predicted width</td></tr><tr><td>Attribution surrogate</td><td>RF, 300 trees</td><td>fitted to the residual forward mapping</td></tr><tr><td>Random seed</td><td>42</td><td>applied to the uniform SHAP sampling and RF fitting; the Saltelli design is deterministic</td></tr></table>

Both analyses used a common product-uniform reference distribution spanning the observed marginal bounds of the four process variables

Table S2. Uniform sampling bounds used for the SHAP and Sobol reference distribution.
<table><tr><td>Process variable</td><td>Sampling bounds</td><td>Basis</td></tr><tr><td>Laser power, P</td><td>150-400 W</td><td>observed marginal range</td></tr><tr><td>Scan speed, V</td><td>100-1600 mm s−¹</td><td>observed marginal range</td></tr><tr><td>Hatch spacing, H</td><td>25-500 μm</td><td>observed marginal range</td></tr><tr><td>Focus offset, F</td><td>0-80 mm</td><td>observed marginal range</td></tr></table>

Because the rectangular reference distribution assumes independence among the four process variables, the resulting indices describe the fitted residual forward mapping under a synthetic input distribution rather than a variance decomposition of the experimental design. The attribution surrogate was used only to obtain TreeSHAP values over the sampled input space; its goodness of fit was not evaluated on an independent sample, and no generalization claim is made. Because the SHAP and Sobol analyses used the same reference bounds, their agreement represents consistency under a shared model-evaluation distribution rather than independent validation.

## S2. Applicability-gate sensitivity sweep

The deployed applicability gate used a neighborhood size of k = 5 and a reference percentile of 95. Both settings were inherited from canonical implementation and were held fixed throughout the final analysis. The sweep reported here characterizes model behavior around that frozen configuration and was not used to select it.

Each sweep entry was refitted within the applicable training partition, so no held-out texture target contributed to the corresponding gate calibration. One quantity was varied at a time while the other was held at its deployed value.

Table S3. Applicability-gate sensitivity sweep. Grouped generalization is reported as leave-one-groupout (LOGO) R²; regime transfer is reported as R² for the controlled withheld +80 mm evaluation. The deployed configuration is k = 5 with a reference percentile of 95.
<table><tr><td>Swept quantity</td><td>Value</td><td>LOGO R²</td><td>+80 mm R²</td><td>Configuration</td></tr><tr><td rowspan="4">Neighborhood size k</td><td>3</td><td>0.592</td><td>0.759</td><td rowspan="4">deployed</td></tr><tr><td>5</td><td>0.592</td><td>0.750</td></tr><tr><td>10</td><td>0.583</td><td>0.715</td></tr><tr><td>15</td><td>0.577</td><td>0.656</td></tr><tr><td rowspan="3">Reference percentile</td><td>90</td><td>0.587</td><td>0.762</td><td rowspan="3">deployed</td></tr><tr><td>95</td><td>0.592</td><td>0.750</td></tr><tr><td>99</td><td>0.574</td><td>0.606</td></tr></table>

Grouped generalization varied by less than 0.02 in R² across the swept neighborhood sizes and reference percentiles. Regime transfer to the withheld +80 mm condition was more sensitive: increasing the neighborhood size from 5 to 15 or the reference percentile from 95 to 99 was associated with lower +80 mm performance, reducing R² from 0.750 to 0.656 and 0.606, respectively. Smaller neighborhoods and a more restrictive reference percentile produced marginally higher transfer scores but were not adopted, because the deployed configuration

was fixed before this sweep was performed.

The smallest evaluated neighborhood size was k = 3. Smaller neighborhoods were not examined because the distance estimate becomes increasingly sensitive to individual nearest neighbors as k decreases, and at k = 1 the normalized distance is determined entirely by a single training observation, losing its intended interpretation as a local data-density measure.

## S3. Texture-target reference fraction

The texture target was normalized by an adopted random-texture reference fraction of 10.21%, inherited from the original crystallographic-processing workflow. Its consistency with a uniform orientation distribution was evaluated independently.

For a uniformly oriented cubic crystal, the build direction lies within $1 5 ^ { \circ }$ of a symmetryequivalent ⟨001⟩ direction when it falls inside one of six spherical caps centered on $\mathrm { t h e } \pm [ 1 0 0 ]$ 2 $\pm [ 0 1 0 ] \mathrm { a n d } \pm [ 0 0 1 ]$ poles. Because these poles are separated by $9 0 ^ { \circ }$ , the caps do not overlap at the adopted tolerance, giving the combined solid-angle fraction

$$
6 \cdot \frac { 2 \pi ( 1 - \cos 1 5 ^ { \circ } ) } { 4 \pi } { = } 3 ( 1 - \cos 1 5 ^ { \circ } ) { = } 1 0 . 2 2 2 \% .
$$

An independent Monte Carlo calculation using $5 ~ \times ~ 1 0 ^ { 7 }$ Haar-uniform random rotations, sampled by the Shoemake method with a fixed seed, gave $1 0 . 2 2 1 8 \% \pm 0 . 0 0 8 6$ percentage points (two binomial standard errors), in agreement with the analytical expectation to 0.0005 percentage points. This calculation does not assume non-overlapping caps and therefore verifies the analytical derivation independently. The script and its output are included in the distributed archive.

The archived reference value differs from the analytical value by 0.012 percentage points, or 0.12% relative. Replacing 10.21% with 10.222% would uniformly rescale e₀₀₁ by a factor of 0.9988, shifting the observed maximum from 7.965 to 7.955 MUD. The archived value was retained to preserve consistency with the finalized target table and all downstream results.

## S4. Experimental group labeling

The nine Stage-2 experimental groups are designated G1 to G9 in the main text. The distributed project archive uses the original acquisition identifiers listed in Table S4. The two labeling systems refer to the same nine groups and the same 278 observations.

Table S4. Correspondence between the manuscript group labels and the acquisition identifiers used in the distributed project archive.
<table><tr><td>Manuscript label</td><td>Acquisition identifier</td><td>Observations</td></tr><tr><td>G1</td><td>OR-4</td><td>38</td></tr><tr><td>G2</td><td>OR-5</td><td>36</td></tr><tr><td>G3</td><td>OR-6</td><td>41</td></tr><tr><td>G4</td><td>OR-7</td><td>30</td></tr><tr><td>G5</td><td>OR-8</td><td>42</td></tr><tr><td>G6</td><td>OR-9</td><td>18</td></tr><tr><td>G7</td><td>OR-10</td><td>21</td></tr><tr><td>G8</td><td>OR-11</td><td>11</td></tr><tr><td>G9</td><td>OR-12</td><td>41</td></tr><tr><td>Total</td><td></td><td>278</td></tr></table>

The acquisition identifiers begin at OR-4 because the Stage-2 texture campaign sampled a conduction-focused subset of the earlier melt-pool campaign. Groups OR-1 to OR-3 consisted predominantly of negative focus-offset conditions and keyhole-mode tracks and were not advanced to the main EBSD characterization, since the texture study targets the conductiondominated regime in which build-direction columnar growth is stable.

## S5. omparison with representative texture-prediction frameworks

Reported accuracies from previous LPBF texture-prediction studies follow different target definitions, datasets, and validation designs, and are therefore not directly comparable with the present results. Table S5 summarizes each study according to what it reported rather than placing the studies on a common accuracy scale.

Table S5. Scope-aware comparison with representative LPBF crystallographic-texture and microstructure prediction frameworks. NR indicates a quantity not reported by the source study. Metrics retain the target definitions and validation designs of their respective studies and should not be interpreted as like-forlike model rankings
<table><tr><td>Study</td><td>Material and target</td><td>Framework</td><td>Evaluation</td><td>Reported R²</td><td>Reported MAPE</td><td>Other reported performance</td><td>Computational pathway</td><td>Applicability treatment</td></tr><tr><td></td><td>al. [21] ratios r22o and r200</td><td>regression</td><td>Ten-fold cross-validation Sofras et 304L; diffraction Decision-tree for tree pruning; six newly fabricated control</td><td>NR</td><td>NR</td><td>Training and cross- validation errors; measured versus predicted control</td><td>Trained-tree inference; window screening; no runtime NR</td><td>Dense-processing- formal prediction</td></tr><tr><td>Whitney grain morphology et al. [22] and texture, α/α&#x27;</td><td>Ti-6A1-4V; β- descriptors</td><td>FDMC-PF- ML hybrid</td><td>and held-out testing for the ML surrogate; FDMC β- texture compared with EBSD at three energy</td><td>NR</td><td>NR</td><td>R² ≥ 0.93 for surrogate phase- fraction and lath- width outputs; not a</td><td>FDMC simulation retained; ML replaces PF response; runtime NR</td><td>Temperature-rule routing and interpolation; no query-level output</td></tr><tr><td>This work</td><td>IN718; e001</td><td>Physics- anchored gated hybrid</td><td>LOGO and complete withheld-defocus evaluations</td><td>0.592 (LOGO); 0.750 (+80 mm)</td><td>27.7% (LOGO); 14.4% (+80 mm)</td><td>MAE, RMSE, Spearman ρ, coverage, retention</td><td>Trained surrogate inference; no per-query numerical solver; runtime not benchmarked</td><td>Residual attenuation and anchor-validity abstention</td></tr></table>

Table S6. Mean absolute percentage error (MAPE, %) for the four point predictors under the three evaluation settings, calculated as $1 0 0 \mathrm { { n } ^ { - 1 } \Sigma | \hat { y } _ { i } - y _ { i } | / \Delta y _ { i } }$ from the same per-observation predictions used for Table 3. MAPE is interpreted together with the scale-preserving MAE and RMSE values in Table 3 because it assigns greater weight to observations with lower e₀₀₁; only one observation had $\mathrm { e } _ { 0 0 1 } < 1 . 0$ MUD. Focus-zero values are diagnostic because no operational prediction was issued for this subset.
<table><tr><td>Evaluation setting</td><td>Greybox anchor</td><td>Black-box RF</td><td>Ungated hybrid</td><td>Gated hybrid</td></tr><tr><td>LOGO generalization</td><td>35.4</td><td>28.9</td><td>28.5</td><td>27.7</td></tr><tr><td>+80 mm withheld</td><td>13.6</td><td>27.9</td><td>27.8</td><td>14.4</td></tr><tr><td>Focus-zero withheld</td><td>59.5</td><td>85.1</td><td>66.7</td><td>61.5</td></tr></table>

## S6. Protocol-matched baselines from literature-derived model forms

Whereas Table S5 compares the reported scope of representative studies, this section evaluates literature-derived model forms numerically on the present dataset. These comparisons were constructed post hoc as diagnostic benchmarks and were not used for model selection or framework configuration. Each entry is an implementation of a published procedure or descriptor applied to the present IN718 dataset, not a reproduction of published results: the material, the target definition and the validation design all differ from the source studies. Every comparator followed its intended or frozen configuration, with all data-dependent tuning confined to the outer-training partition.

Table S7. Literature-derived model forms evaluated on the present dataset under the same LOGO and withheld-defocus protocols used throughout. The reduced geometric surrogate retains only a singlepool boundary-orientation calculation and omits the multi-track and multi-layer growth selection implemented in the full mechanistic model, so it is not a reproduction of that model. Focus-zero values are diagnostic; see below. MAPE for these baselines is not reported because several give negative R² and their relative errors are not informative
<table><tr><td>Model form</td><td>Source basis</td><td>Implementation status</td><td>R²</td><td>LOGO +80 mm R²</td><td>Focus- zero R²</td></tr><tr><td>Regression tree, P V H</td><td>Sofras et al. [21]</td><td>source-method reimplementation</td><td>0.191</td><td>0.175</td><td>-1.278</td></tr><tr><td>Regression tree, P V H F</td><td>Sofras et al. [21]</td><td>dataset-motivated adaptation</td><td>0.302</td><td>0.297</td><td>-0.961</td></tr><tr><td>Regression tree, P V H + Sofras [21]; Caiazzo VED</td><td>et al. [47]</td><td>dataset-motivated adaptation</td><td>0.241</td><td>0.112</td><td>-1.186</td></tr><tr><td>Volumetric energy-density power law</td><td>Caiazzo et al. [47]</td><td>literature-derived descriptor baseline</td><td>-0.065</td><td>-0.383</td><td>-3.930</td></tr><tr><td>Linear energy-density power law</td><td>Caiazzo et al. [47]</td><td>literature-derived descriptor baseline</td><td>-0.038</td><td>-0.517</td><td>-2.494</td></tr><tr><td>Normalized enthalpy, IN718 properties</td><td>King et al. [12]</td><td>dataset-motivated adaptation</td><td>-0.203</td><td>0.028</td><td>0.002</td></tr><tr><td>Reduced melt-pool geometric surrogate</td><td>Liu et al. [48]</td><td>reduced literature-inspired surrogate</td><td>-0.170</td><td>0.013</td><td>-0.194</td></tr><tr><td>Greybox anchor</td><td></td><td>this work</td><td>0.287</td><td>0.778</td><td>-0.298</td></tr><tr><td>Black-box random forest</td><td></td><td>this work</td><td>0.538</td><td>-0.001</td><td>-1.092</td></tr><tr><td>Ungated hybrid</td><td></td><td>this work</td><td>0.564</td><td>0.199</td><td>-0.770</td></tr><tr><td>Gated hybrid</td><td></td><td>this work</td><td>0.592</td><td>0.750</td><td>-0.378</td></tr></table>

The focus-zero column is reported for diagnostic purposes only. The literature-derived

baselines and the internal point-prediction baselines issue a prediction for every query and implement no abstention rule, whereas the proposed framework withholds all focus-zero predictions. A negative diagnostic coefficient for the gated hybrid therefore does not indicate an operational prediction failure.
<table><tr><td>Model class</td><td>Focus-zero treatment</td></tr><tr><td>Literature-form baselines</td><td>Diagnostic prediction; no abstention rule</td></tr><tr><td>Internal point-prediction baselines</td><td>Diagnostic prediction</td></tr><tr><td>Proposed framework</td><td>Prediction withheld</td></tr></table>

Table S8. Sensitivity of the +80 mm coefficient of determination to the cost-complexity pruning implementation. The canonical implementation computes the exact pruning path on each training partition and selects the subtree with minimum mean ten-fold validation error, following the published description most closely; the alternatives use pre-specified alpha grids. Across all three implementations the defocus-aware tree remained below the physics anchor (0.778).
<table><tr><td>Predictor set</td><td>Canonical (exact pruning path)</td><td>Fixed log grid</td><td>Fixed linear grid</td></tr><tr><td>PVH</td><td>0.175</td><td>0.130</td><td>0.138</td></tr><tr><td>PVHF</td><td>0.297</td><td>0.297</td><td>0.325</td></tr></table>

## S7. eproducibility

All analyses used deterministic settings. The applicability-gate sweep was regenerated through leakage-free refitting within each evaluation partition, whereas the interpretation analyses were regenerated from the finalized Stage-2 table and the frozen Stage-1 mediator outputs. Canonical per-observation predictions and saved mechanism-analysis outputs were used for verification. The literature-derived baselines in Section S6 were regenerated by refitting within each evaluation partition, with all pruning selection confined to the outer-training partition; per-observation predictions for every predictor set and protocol are included in the archive. The distributed project archive contains the associated scripts and intermediate files.