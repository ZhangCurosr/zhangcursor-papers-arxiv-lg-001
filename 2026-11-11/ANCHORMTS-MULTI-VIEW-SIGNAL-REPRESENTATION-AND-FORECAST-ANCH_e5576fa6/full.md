# ANCHORMTS: MULTI-VIEW SIGNAL REPRESENTATION AND FORECAST-ANCHORED FUSION FOR LONG-TERM TIME-SERIES FORECASTING

Gao Wu<sup>1</sup> Tingting Xu<sup>2,3</sup> Changning Wu<sup>1</sup> Tengfei Long<sup>1</sup> Yong Liu<sup>1</sup> Jiabin Jin<sup>2,3,∗</sup> Kexin Zhang<sup>1,∗</sup>

<sup>1</sup>College of Control Science and Engineering, Zhejiang University, Hangzhou, China

<sup>2</sup>Zhejiang Provincial Security Science and Technology Research Institute, Hangzhou, China

<sup>3</sup>Zhejiang Province Key Laboratory of “Public Security Brain” Big Data and Key Security Technologies Research, Hangzhou, China

## ABSTRACT

Long-term time-series forecasters typically optimize a single numerical representation, even though the same history exposes complementary geometry, recurrence, and semantic structure. We propose AnchorMTS, which derives self-aligned visual and textual views from each observed window and uses them only to refine a complete numerical forecast. A trainable evidence path produces a candidate forecast, while a parameter-free output projection enforces an exact per-variable bound on its displacement from the numerical anchor. Across six benchmarks and four horizons, AnchorMTS achieves the best macro MSE/MAE (0.336/0.367) among seven supervised baselines and wins 18 of 24 MSE settings against the cellwise baseline envelope; removing the auxiliary path raises macro MSE to 0.340. The projection leaves 95.8% of candidate refinements unchanged while yielding zero bound violations, showing that complementary multi-view evidence can improve forecasting while its final influence remains explicitly controlled.

Index Terms— Long-term time-series forecasting, multi-view signal representation, Gramian angular field, recurrence plot, multimodal fusion

## 1. INTRODUCTION

Long-term time-series forecasting (LTSF) supports planning in energy, weather, transportation, and finance. Modern supervised methods reorganize numerical histories through decomposition [1, 2], patching [3], two-dimensional temporal variation [4], inverted variable tokens [5], multiscale mixing [6], or endogenous–exogenous interaction [7]. These advances are substantial, yet most numerical LTSF models ultimately optimize forecasting within a single learned representation space.

The same observed history admits complementary representations without adding measurements. Gramian angular fields (GAFs) expose pairwise angular relations [8]; line renderings make slopes and turning points spatially local; recurrence plots reveal repeated states [9]; and a compact description summarizes trend, periodicity, and variable-level stability. Frozen vision–language encoders can organize such views, but they were not trained as forecasting models. This creates a central tension: alternative representations may reveal useful structure, yet auxiliary representations should not be allowed to replace a strong numerical prediction.

Recent multimodal forecasters demonstrate that alternative views can complement numerical sequences. VisionTS formulates forecasting as visual reconstruction, Time-VLM couples temporal, visual, and textual learners, DMMV integrates numerical and visual predictors, and TimesCLIP aligns visual and textual representations [10, 11, 12, 13]. Related multi-view formulations also explore adaptive tri-modal fusion, time/frequency/vision representations, evidence-theoretic aggregation, and uncertainty-aware view weighting [14, 15, 16, 17]. AnchorMTS addresses a complementary question in full-data supervised LTSF: how can history-derived evidence refine a strong numerical forecast without taking control of the prediction? Rather than treating derived views as peer forecasters or replacements for the numerical representation, AnchorMTS uses them only to propose residual evidence around an independently complete numerical forecast. Our formulation separates two questions: whether auxiliary representations can improve the numerical prediction, and how much authority they should have over the final forecast. The trainable evidence path addresses the first, while the exact forecast-space projection addresses the second.

AnchorMTS realizes this principle with two coupled forecasting paths. A numerical backbone produces a complete anchor forecast, while frozen image and text representations condition a trainable evidence path that generates a candidate refinement. Intermediate updates are norm-controlled, and a final parameter-free projection constrains the candidate directly around the numerical anchor in forecast space.

Our contributions are threefold. First, we introduce a selfderived multi-view signal interface that exposes angular, trajectory, recurrence, and semantic structure without external measurements or manual alignment. Second, we formulate multimodal forecasting as bounded refinement of a complete numerical predictor, separating learned evidence utility from explicit final forecast authority. Third, we validate AnchorMTS across six benchmarks and four horizons, demonstrating competitive forecasting gains, complementary frozen-view structure, and an exact forecast-space influence bound.

## 2. METHOD

Given an observed window $\boldsymbol { X } \in \mathbb { R } ^ { L \times C }$ , the task is to predict $Y \in$ $\mathbb { R } ^ { H \times C }$ . AnchorMTS builds two paths from X: a primary numerical forecaster and an auxiliary representation path. Figure 1 makes their roles explicit.

## 2.1. Self-Derived Representations

For variable c, min–max scaling maps the history to $\tilde { x } _ { i } ^ { c } \in [ - 1 , 1 ]$ and $\phi _ { i } ^ { c } = \operatorname { a r c c o s } ( \tilde { x } _ { i } ^ { c } )$ . The ${ \bf G A } \bar { \bf F } G _ { i j } ^ { c } \dot { = } \cos ( \phi _ { i } ^ { c } + \bar { \phi } _ { j } ^ { c } )$ is a deterministic pairwise angular re-encoding; channel-labelled tiles preserve

![](images/8eda17d596fab02c057b672a2d7e4e6b9f190425f2fe3f7ad8c7ae50ee5364c5.jpg)  
Fig. 1: Overview of AnchorMTS. The observed history is processed by a complete numerical forecaster to produce the anchor prediction, while self-derived angular, trajectory, recurrence, and textual views provide cached auxiliary evidence. Evidence fusion proposes a multimodal candidate, and an exact forecast-space projection converts its residual into a bounded refinement of the numerical anchor.

variable identity. $\mathrm { W i t h } z _ { i } ^ { c }$ denoting within-window standardized values and $s _ { i } \ = \stackrel { \cdot } { C } ^ { - 1 } \sum _ { c } \stackrel { \cdot } { z _ { i } ^ { c } }$ , the line view renders the ordered path $( i , s _ { i } )$ , while

$$
R _ { i j } = \exp ( - | s _ { i } - s _ { j } | / ( \sigma + \epsilon ) )
$$

forms a recurrence view, where σ is the median positive pairwise distance. A fixed prompt asks GPT-4.1-mini to summarize only the historical GAF’s trend, periodicity, stability, and variable dynamics. Thus the three images and text contain no future values or external measurements.

Frozen CLIP ViT-B/32 towers encode each view once [18]. Their last-layer sequences—50 tokens per image and at most 77 text tokens—are cached and reused across training epochs and horizons. These views intentionally expose different locality structures of the same observed history. The variable-wise GAF preserves channel identity while turning pairwise angular relations into spatial patterns; the line view makes ordered slopes and turning points locally accessible; the recurrence view emphasizes returns to similar states irrespective of their absolute timestamps; and the caption compresses variable-level trend, periodicity, and stability into a semantic summary. Their purpose is therefore not to introduce additional measurements, but to make complementary structures of X accessible to frozen representation encoders before fusion.

We deliberately retain channel separation in the GAF to preserve cross-variable heterogeneity, whereas the line and recurrence views operate on the standardized joint trajectory to summarize shared temporal evolution without multiplying the number of visual streams by C.

The representation interface is deliberately fixed with respect to forecasting optimization. The visual and textual encoders are frozen, and all four views are aligned by the same observed-window index before training the forecasting path. Consequently, the forecasting model learns how to use a stable auxiliary representation space rather than jointly reshaping the pretrained encoders for each dataset or horizon. The same cached evidence can therefore be reused across epochs and prediction horizons while the trainable numerical and fusion paths adapt to the forecasting objective.

A lightweight router assigns sample-dependent weights to the three visual memories before evidence fusion.

## 2.2. Forecast-Anchored Evidence Fusion

The numerical backbone maps X to temporal tokens $Z \in \mathbb { R } ^ { N \times d }$ including patch and variable summaries, and a decoder D produces the complete anchor forecast $\widehat { Y } _ { \mathrm { t s } }$ . Let $E _ { v } ^ { ( k ) }$ be the cached CLIP sequence for visual view $k \in \{ \mathrm { G A F } , \mathrm { l i n e } , \mathrm { R P } \}$ . Projection and a learned view identity give

$$
\begin{array} { r } { V ^ { ( k ) } = \mathrm { P r o j } _ { v } ( E _ { v } ^ { ( k ) } ) + e ^ { ( k ) } , \qquad \alpha = \mathrm { s o f t m a x } _ { k } r \Big ( \mathrm { m e a n } V ^ { ( k ) } \Big ) . } \end{array}\tag{1}
$$

A shared layer-normalized linear scorer maps the pooled representation of each visual view to a softmax weight, producing the routed visual memory $M _ { v }$ . For the $K$ available views, the routed visual memory is $M _ { v } =$ concat $_ k ( K \alpha _ { k } V ^ { ( k ) } )$ . The routed memory therefore preserves the full token sequence of each available view while adapting its contribution to the current history.

The temporal tokens query this memory through cross-attention,

$$
\Delta _ { v } = \mathrm { M H A } ( \mathrm { L N } Z , \mathrm { L N } M _ { v } , \mathrm { L N } M _ { v } ) .\tag{2}
$$

For any proposed update $\Delta$ around state A, define the token-wise norm control

$$
B _ { \rho } ( \Delta ; A ) = \Delta \operatorname* { m i n } \biggl ( 1 , \frac { \rho \| A \| _ { 2 } } { \| \Delta \| _ { 2 } + \epsilon } \biggr ) .
$$

With a learned element-wise gate $g _ { v }$ , the visual state is $Z _ { v } = Z +$ $g _ { v } \odot B _ { \rho _ { v } } ( \Delta _ { v } ; Z )$ , where $\rho _ { v } = 0 . 6 0$

Text is conditioned separately. If $E _ { t }$ and m are the cached text tokens and attention mask, masked pooling gives

$$
t = \frac { \sum _ { j } m _ { j } \operatorname { P r o j } _ { t } ( E _ { t , j } ) } { \sum _ { j } m _ { j } } , \qquad ( s , b ) = \operatorname { F i L M } ( t ) .\tag{3}
$$

The proposal $\Delta _ { t } = Z _ { v } \odot s + b$ is bounded and gated as $Z _ { v t } \ =$ $Z _ { v } + g _ { t } \odot B _ { \rho _ { t } } ( \Delta _ { t } ; Z _ { v } )$ with $\rho _ { t } = 0 . 2 5$

Visual and textual evidence play deliberately different roles in the fusion path. The three visual representations retain token-level spatial structure, so the temporal state queries their routed memory through cross-attention. The caption instead provides a compact global description of trend, periodicity, and variable behavior;

it therefore bypasses the visual router and modulates the visually conditioned state through feature-wise scale and shift. This asymmetric design preserves fine-grained visual evidence for token-level retrieval while using text as a global semantic conditioning signal.

The intermediate budgets $\rho _ { v } ~ = ~ 0 . 6 0$ and $\rho _ { t } ~ = ~ 0 . 2 5$ are fixed representation-space regularization settings shared across all datasets and horizons; they are not the source of the forecast-level guarantee. The shared decoder maps the untouched temporal state to the anchor forecast $\hat { Y } _ { t s } ^ { } ,$ , while the evidence-conditioned state enters the learned forecast-residual branch to produce the multimodal candidate.

Using the same decoder for both states is important to the anchor formulation: the numerical anchor and multimodal candidate share the same forecast parameterization and differ only through the evidence-conditioned state and residual branch. The resulting forecast difference can therefore be interpreted directly as an evidenceinduced correction before the final output constraint is applied.

The anchor and candidate share the same inverse normalization and channel-wise affine output map, yielding two forecasts in the reported prediction space before the final projection. The visual and textual state controls shape this candidate in representation space; the exact forecast-level constraint is imposed separately on the final outputs.

## 2.3. Exact Final-Output Projection

Define the candidate correction $\Delta _ { f } = \widetilde { Y } _ { \mathrm { m v } } - \widehat { Y } _ { \mathrm { t s } }$ . The final reported forecast is

$$
\begin{array} { r } { \widehat { Y } = \widehat { Y } _ { \mathrm { t s } } + \mathcal { P } _ { \rho _ { f } } \big ( \Delta _ { f } ; \widehat { Y } _ { \mathrm { t s } } \big ) , } \end{array}\tag{4}
$$

$$
\mathcal { P } _ { \rho } ( \Delta ; A ) = \Delta \operatorname* { m i n } \left( 1 , \frac { \rho \| A \| _ { 2 } } { \| \Delta \| _ { 2 } + 1 0 ^ { - 8 } } \right) ,\tag{5}
$$

We fix $\rho _ { f } ~ = ~ 0 . 2 5$ globally, defining a 25% relative horizon-wise $L _ { 2 }$ budget around the numerical anchor for every sample–variable forecast. The norm is computed across the H forecast steps, giving

$$
\lVert \widehat { Y } - \widehat { Y } _ { \mathrm { t s } } \rVert _ { 2 } \leq \rho _ { f } \lVert \widehat { Y } _ { \mathrm { t s } } \rVert _ { 2 }
$$

by construction.

The operator is the identity whenever the candidate correction already lies inside the prescribed ball. Otherwise, it preserves the correction direction and rescales only its magnitude to the boundary. Thus the constraint does not define a new forecast direction; it only limits the distance that the evidence-conditioned candidate may move from the complete numerical anchor.

Because $\mathcal { P } _ { \rho _ { f } }$ is applied to the two fully formed forecasts after their shared output map, the bound is independent of the internal representation geometry and is directly expressed in the units used for evaluation.

## 3. EXPERIMENTS

## 3.1. Setup

We evaluate ETTh1, ETTh2, ETTm1, ETTm2, Exchange, and Weather with input length 96 and horizons H ∈ {96, 192, 336, 720}, yielding 24 cells. Standard chronological splits and trainingpartition normalization are used [19, 20, 1]; MSE and MAE average all forecast steps and variables. Every cell uses the same self-derived views, frozen CLIP path, fusion topology, and $( \rho _ { v } , \rho _ { t } , \rho _ { f } ) \ = \ ( 0 . 6 0 , 0 . 2 5 , 0 . 2 5 )$ . Ordinary backbone and optimization settings are selected per cell. We compare with seven supervised LTSF baselines spanning linear, patch, two-dimensional, inverted-token, multiscale, and endogenous–exogenous designs [21, 3, 4, 5, 6, 7, 22]. Table 1 includes methods with complete results under this aligned 24-setting supervised protocol.

<table><tr><td>Method MSE↓</td><td colspan="3">MAE↓ MSE W/T/L MAE W/T/L</td></tr><tr><td>AnchorMTS</td><td>0.336</td><td>0.367</td><td></td></tr><tr><td>TimePro</td><td>0.348</td><td>0.374 22/0/2</td><td>20/1/3</td></tr><tr><td>TimeXer</td><td>0.351</td><td>0.374 20/1/3</td><td>18/1/5</td></tr><tr><td>TimeMixer</td><td>0.358</td><td>0.384 23/0/1</td><td>21/1/2</td></tr><tr><td>iTransformer</td><td>0.358</td><td>0.379 24/0/0</td><td>23/1/0</td></tr><tr><td>PatchTST</td><td>0.372 0.388</td><td>23/0/1</td><td>23/0/1</td></tr><tr><td>TimesNet</td><td>0.373 0.391</td><td>24/0/0</td><td>23/0/1</td></tr><tr><td>DLinear</td><td>0.400</td><td>0.419 24/0/0</td><td>24/0/0</td></tr><tr><td>Cellwise best*</td><td>0.340</td><td>0.369</td><td>18/0/6 15/1/8</td></tr></table>

Table 1: Macro comparison over 24 aligned dataset–horizon settings. W/T/L compares AnchorMTS with each baseline at the reported three-decimal precision. \*Cellwise best denotes the metricwise envelope over the seven baselines rather than an additional forecasting model.

## 3.2. Main Forecasting Results

AnchorMTS achieves the lowest 24-cell macro errors, with MSE/ MAE of 0.336/0.367 (Table 1). Against the metric-wise cellwise envelope of the seven baselines, it wins 18/24 MSE settings and 15/24 MAE settings with one MAE tie; the cell-resampled MSE-difference interval is $[ - 0 . 0 0 7 1 , - 0 . 0 0 1 6 ]$ . Figure 2 shows that the datasetaverage MSE gain over TimePro spans all six benchmarks. The largest margin occurs on Weather, the highest-dimensional benchmark in the grid, a pattern consistent with the stronger channeldiversity organization of the variable-wise GAF in Table 2.

The advantage also persists across forecast length. Averaging the six datasets at each horizon, AnchorMTS improves over Time-Pro at $H = 9 6 , 1 9 2 , 3 3 6$ , and 720, with MSE margins of −0.0075, −0.0090, −0.0138, and −0.0181, respectively. AnchorMTS remains favorable at all four horizons, with the largest absolute MSE margin at $H = 7 2 0$ Together with the six dataset-level gains in Fig. 2(b), the improvement is distributed across both dataset identity and forecast length.

## 3.3. Auxiliary-Path Ablation

We compare AnchorMTS with a configuration-matched temporalonly architectural ablation that retains the selected numerical backbone, forecasting heads, optimizer, loss, and seed but removes the auxiliary representation and fusion modules. Across all 24 settings, the complete model lowers macro MSE/MAE from 0.340/0.369 to 0.336/0.367, improves MSE in 19 settings, and yields a paired MSEdifference interval of [−0.0068, −0.0021]. Because the temporal backbone and its forecasting configuration are retained, this comparison evaluates the complete auxiliary pathway as an additive refinement rather than as a replacement forecaster. Improvements span datasets and horizons, not one benchmark.

![](images/cd1a3c02fd4d9c77404bbf3a33182e400d7ba67248b5a97922efeab8e4fa8266.jpg)

![](images/c3e10d7bd8edcca9d29df420f8188ccd79b9f95f9b76f4abad19b897413802af.jpg)  
Fig. 2: Forecasting accuracy. (a) Macro MSE over the 24 aligned dataset-horizon cells. (b) Dataset-average MSE difference from TimePro, the strongest individual baseline by macro error; negative values favor AnchorMTS.

![](images/166201192069ef465e16649ab97da9796f67db3f1d4e26517eb01dbb4cdfa72b.jpg)

![](images/1167668204f482f4e10663d285bafd80a0231a89023d0b3dfda4e708cb49e995.jpg)  
Fig. 3: Exact forecast-space projection. (a) Correction-ratio distributions before and after the final projection over all 24 settings; candidate refinements inside $\rho _ { f } = 0 . 2 5$ pass unchanged, while the extreme tail is truncated at the prescribed radius. (b) Fraction of sample–variable forecasts projected in each dataset. Overall, 95.8% of refinements pass unchanged and no reported forecast violates the bound.

## 3.4. Exact Forecast-Space Bound

Across 1,663,840 sample–variable forecasts, 95.8% of candidate refinements already satisfy the prescribed radius and pass through unchanged. The projection preserves the median and 95th-percentile correction ratios (0.0451/0.2299), truncates the maximum from 8.1402 to $0 . 2 5 0 0 ,$ and yields zero violations. Macro MSE changes by only $- 4 . 7 \times 1 0 ^ { - 5 }$ , showing that the exact bound primarily removes rare extreme corrections without suppressing ordinary refinements.

These statistics distinguish exact influence control from generic shrinkage. Since the median and 95th-percentile corrections are unchanged, the projection does not systematically attenuate the evidence path; it intervenes only on the small tail that exceeds the prescribed forecast-space budget. Figure 3(b) further shows that activation is dataset dependent rather than uniformly imposed, while the aggregate projection rate remains only 4.2%. Thus the candidate forecaster retains its ordinary refinements in the vast majority of cases, whereas the final reported prediction always satisfies the same interpretable constraint.

The activation profile also varies substantially across datasets: Weather exhibits the highest projection rate, whereas ETTh1 and Exchange show substantially lower activation. Because the budget is defined relative to each numerical anchor, its activation rate is free to vary across datasets rather than being fixed by a global correction threshold. Importantly, the same $\rho _ { f }$ is used in every setting, so the dataset-dependent activation emerges from the learned candidate corrections rather than dataset-specific tuning of the final constraint.

## 3.5. Complementarity of Frozen Views

To examine whether the frozen encoders retain the structures targeted by each self-derived view, we retrieve nearest neighbors in each visual embedding space and compare numerical trend, periodicity, recurrence, and channel-diversity descriptors with arbitrary window pairs. Retrieval lift measures the relative reduction in descriptor distance.

Table 2 reveals complementary organization in the frozen spaces: line and recurrence views primarily preserve aggregate temporal dynamics, whereas the variable-wise GAF is strongest for cross-channel diversity. This pattern agrees with the intended roles of the self-derived representations.

<table><tr><td>Frozen view</td><td>Trend</td><td>Period. Recur.</td><td>Chan. div.</td></tr><tr><td>Var.-wise GAF</td><td>0.107</td><td>0.109 0.109</td><td>0.161</td></tr><tr><td>Joint line</td><td>0.279</td><td>0.324 0.329</td><td>0.039</td></tr><tr><td>Recurrence plot</td><td>0.259</td><td>0.248 0.287</td><td>0.042</td></tr></table>

Table 2: Retrieval lift of frozen visual representations, macroaveraged across six training sets. Higher values indicate that representation-space neighbors are more similar in the corresponding numerical signal descriptor than arbitrary window pairs.

The division of labor is consistent with how the views are constructed. The joint line and recurrence representations compress channels into a shared trajectory and therefore emphasize common temporal evolution, whereas the variable-wise GAF retains channel identity and exposes heterogeneous pairwise geometry. The frozen encoder therefore preserves distinct signal organization across the three view constructions, with each view emphasizing the structure it was designed to make locally accessible.

Taken together, the experiments support the three roles separated by the model design. The main benchmark and temporal ablation establish the forecasting value of the complete auxiliary path; the retrieval analysis shows that the frozen views retain complementary signal organization; and the projection analysis verifies that this additional evidence remains subject to an exact forecast-space influence budget. These results align the empirical evaluation with the representation, refinement, and control stages of Fig. 1.

## 4. CONCLUSION

AnchorMTS formulates multimodal time-series forecasting as bounded refinement of a complete numerical predictor. Self-derived visual and textual views expose complementary structure in the same observed history, while a trainable evidence path uses this auxiliary information to refine the numerical forecast. An exact output projection independently bounds the authority of the resulting correction. Across six benchmarks and four horizons, this formulation achieves the best macro error among seven supervised baselines while preserving an explicit variable-wise influence bound.

## A. OVERVIEW

This appendix reports the complete forecasting grid and cellwise references, the evaluation and search protocol, exact caption-generation details, the comparison scope of closest multimodal work, representation evidence, and reproduction entry points.

## B. COMPLETE FORECASTING RESULTS

Table 3 reports every dataset–horizon cell against the seven external baselines, and Table 4 identifies the metric-wise cellwise references. All AnchorMTS entries are exact final-output predictions.

Statistical tests. Against the metric-wise cellwise baseline envelope, the main model has exact sign-test $p \ = \ 0 . 0 2 2 7$ for MSE. Against the configuration-matched temporal-only ablation, it has exact sign-test p = 0.0066 for MSE.

## C. EXPERIMENTAL PROTOCOL AND IMPLEMENTATION

## C.1. Dataset and Evaluation Details

Each sample predicts every variable; training-partition statistics standardize validation and test data. External results retain their reported three-decimal precision, which is also used for baseline W/T/L comparisons. Cell-resampled intervals use 100,000 bootstrap samples; matched control intervals resample paired cell units. Exact sign tests are two-sided and exclude ties.

## C.2. Implementation and Selected Configurations

Each window yields a variable-labelled GAF, a line image, and a recurrence image; all are resized to 224 × 224. Recurrence uses $R = \exp [ - d / ( \sigma + 1 0 ^ { - 6 } ) ]$ , with positive distances defined by $d >$ $1 0 ^ { - 8 } .$ , σ their median, and $\sigma = 1$ if no positive distance exists. GPT-4.1-mini receives the historical GAF and the fixed instruction in Section C.4; CLIP ViT-B/32 encodes the images and caption. The shared router applies layer normalization and a scalar linear scorer with bias to each pooled visual representation, then uses temperature 1 and softmax over the three available views before view dropout, without top-k selection. Forecasting uses patch length 16, eight attention heads, and view, image, and text dropout of 0.10, 0.10, and 0.30. The state/output ratios are 0.60, 0.25, and 0.25. The final projection uses $\epsilon = 1 0 ^ { - 8 }$ with no denominator floor. Optimization and checkpoint scoring use the candidate forecast; final test evaluation returns the projected forecast. Experiments were conducted on an NVIDIA GeForce RTX 4060 Ti using Python 3.8.20, PyTorch 2.4.1 with CUDA 12.1, NumPy 1.24.4, and pandas 2.0.3.

The data loader standardizes each channel using statistics fitted on the training partition. The forecasting model then performs detached per-window mean and variance normalization, decodes anchor and candidate in that normalized state, and applies the same inverse per-window transform to both. A shared learned channel-wise affine transformation, $y \mapsto y [ 1 + 0 . 4 0 \operatorname { t a n h } ( a _ { c } ) ] + 0 . 2 6 \operatorname { t a n h } ( b _ { c } )$ is then applied identically to both forecasts; the exact projection is computed afterward in this common reported space.

View construction, caption generation, and frozen CLIP encoding are preprocessing steps that produce the representations consumed by the forecasting model. We therefore do not interpret cached-representation forecasting latency as end-to-end online inference cost.

## C.3. Hyperparameter Search Scope

Each dataset–horizon setting evaluates six initial configurations. Eleven settings additionally evaluate a 12-configuration refinement panel using successive halving with epoch budgets 1, 3, and 8 and reduction factor 3, yielding 18 additional training evaluations (12, 4, and 2). The number of evaluated training configurations per setting is therefore $6 / 6 / 2 4$ for the minimum, median, and maximum.

The explored search space spans temporal widths ({128, 256, 512}), feed-forward widths ({512, 1024, 2048}), encoder depths $( \{ 1 , 2 , 3 \} )$ , batch sizes $( \{ 4 , 8 , 1 6 , 3 2 \} )$ ), and learning rates $( \{ 5 \times$ $1 \mathrm  { 0 ^ { - 5 } , 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } } \} )$ Objectives include MSE and Huber, together with one MSE–MAE configuration with MAE weight 0.1; EMA is either disabled or uses decay 0.99. The panels additionally vary epoch budget and residual-head initialization. The self-derived representation set, frozen encoders, fusion topology, and $( \rho _ { v } , \rho _ { t } , \rho _ { f } )$ remain fixed across all 24 settings.

## C.4. Caption Generation Protocol

## Exact fixed instruction.

This is a Gramian Angular Field (GAF) image transformed from a multivariate time series of length 96. The image is a grid of GAF representations for each dimension, with dimension names at the top. In GAF, colors represent the angular sum/difference of time points, where lighter colors (e.g., yellow, red in a ’jet’ colormap) indicate larger angular values (similarity), and darker colors (e.g., blue, purple) indicate smaller angular values (dissimilarity). Your task is to generate a concise textual description of the patterns in the image, following this template:”A Gramian Angular Field of [N] time series. [Variables with strong periodicity] show strong periodicity, [variables with seasonal or external impact] indicate seasonal impact, [variables with stable trend] remain stable. Overall: [summary of cycles/trend].”Requirements: - Keep the output under 77 tokens. - Mention variable names shown in the image. - Highlight differences between variables (periodic, seasonal, stable). - Be compact, clear, and avoid redundant words. Please replace [N] with the actual number of time series dimensions shown in the image, and fill in the bracketed placeholders with specific variable names or descriptions based on your analysis. If a category is not applicable, you can state ’None’ or ’No significant [pattern]’. Please provide the description text directly, without any additional prefixes or explanations.The following is an example:”A Gramian Angular Field of 7 time series. MUFL and LUFL show strong periodicity with distinct block patterns. HUFL, HULL, MULL, LULL indicate seasonal impact with moderate angular variations. OT remains relatively stable with minimal temporal structure. Overall: mixed periodic and seasonal cycles with varying intensity across load variables.”

API fields. The implementation calls the chat-completions API with model identifier gpt-4.1-mini and max tokens=77. Temperature, top p, seed, and response format are not explicitly specified in the request.

Response preprocessing. The caption is read from response. choices[0].message.content and stripped. It is not subsequently truncated as text. During frozen CLIP encoding, the processor uses padding to maximum length, truncation enabled, and max length=77, which enforces the CLIP token limit. An API exception writes no CSV row; a later run resumes from image names without valid rows. On resume, prior rows whose stripped caption begins with error: are removed and their images are retried. No retry count or backoff is explicitly specified.

Artifact reproducibility. The six dataset caption CSV files are present in the repository and paired with the saved frozen token artifacts. All captions used in the reported experiments are preserved with the frozen artifacts, so reproducing the forecasting results requires neither re-querying the language-model API nor relying on future API-side generation defaults.

<table><tr><td rowspan=1 colspan=6>Dataset  H  AnchorMTS  TimePro    TimeXer  TimeMixer iTransformerPatchTST  TimesNet   DLinearMSE MAE MSE MAE MSE MAE MSE MAE MSE MAEMSE MAE MSE MAE MSE MAE</td></tr><tr><td rowspan=1 colspan=6>ETTh1   96 0.3820.401 0.375 0.398 0.382 0.403 0.385 0.402 0.3860.4050.460 0.447 0.384 0.402 0.395 0.407</td></tr><tr><td rowspan=1 colspan=6>192 0.4300.4300.427 0.429 0.4290.4350.4430.430 0.4410.4360.512 0.477 0.4360.4290.4460.441</td></tr><tr><td rowspan=1 colspan=6>336 0.4630.4450.472 0.450 0.4680.4480.512 0.470 0.4870.4580.5460.4960.491 0.469 0.489 0.467720 0.4710.4700.4760.474 0.469 0.461 0.4970.4760.5030.4910.5440.517 0.5210.5000.513 0.510</td></tr><tr><td rowspan=1 colspan=6>ETTh2   960.280 0.3310.293 0.345 0.286 0.338 0.289 0.342 0.2970.3490.308 0.355 0.340 0.374 0.340 0.394</td></tr><tr><td rowspan=1 colspan=2>192 0.3570.382 0</td><td rowspan=1 colspan=4>.367 0.394 0.363 0.389 0.3780.397 0.3800.4000.393 0.405 0.402 0.414 0.482 0.479</td></tr><tr><td rowspan=1 colspan=2>3360.4080.4210</td><td rowspan=1 colspan=4>.4190.4310.414 0.4230.432 0.434 0.4280.4320.4270.4360.452 0.452 0.5910.541</td></tr><tr><td rowspan=1 colspan=2>720 0.4150.4350</td><td rowspan=1 colspan=4>.4270.445 0.408 0.432 0.4640.464 0.4270.4450.4360.450 0.462 0.4680.839 0.661</td></tr><tr><td rowspan=1 colspan=2>ETTm1  960.3140.3480.</td><td rowspan=1 colspan=4>326 0.3640.3180.3560.3170.356 0.3340.3680.352 0.3740.3380.3750.3460.374</td></tr><tr><td rowspan=1 colspan=2>192 0.3610.3770</td><td rowspan=1 colspan=4>.367 0.383 0.362 0.383 0.367 0.384 0.3770.3910.390 0.393 0.3740.387 0.382 0.391</td></tr><tr><td rowspan=1 colspan=2>336 0.3930.4080</td><td rowspan=1 colspan=4>.402 0.409 0.395 0.407 0.391 0.406 0.4260.4200.421 0.414 0.410 0.411 0.415 0.415</td></tr><tr><td rowspan=1 colspan=2>720 0.4470.4400</td><td rowspan=1 colspan=4>.4690.446 0.452 0.4410.454 0.441 0.4910.4590.462 0.449 0.4780.450 0.473 0.451</td></tr><tr><td rowspan=1 colspan=2>ETTm2  960.1680.2540.1</td><td rowspan=1 colspan=3>78 0.260 0.171 0.256 0.175 0.257 0.1800</td><td rowspan=1 colspan=1>.2640.183 0.270 0.187 0.267 0.193 0.293</td></tr><tr><td rowspan=1 colspan=2>192 0.2300.2990</td><td rowspan=1 colspan=2>.242 0.303 0.237 0.299 0.240 0.302</td><td rowspan=1 colspan=1>0.250</td><td rowspan=1 colspan=1>0.3090.255 0.314 0.249 0.309 0.284 0.361</td></tr><tr><td rowspan=1 colspan=2>3360.2930.3360</td><td rowspan=1 colspan=2>.303 0.342 0.296 0.3380.303 0.343</td><td rowspan=1 colspan=1>0.311</td><td rowspan=1 colspan=1>0.3480.309 0.3470.3210.351 0.382 0.429</td></tr><tr><td rowspan=1 colspan=2>720 0.3840.3990</td><td rowspan=1 colspan=4>.400 0.399 0.392 0.394 0.392 0.3960.4120.4070.412 0.404 0.4080.403 0.558 0.525</td></tr><tr><td rowspan=1 colspan=2>Exchange 96 0.082 0.201 0</td><td rowspan=1 colspan=4>.085 0.204 0.084 0.202 0.090 0.235 0.0860.2060.088 0.205 0.107 0.234 0.088 0.218</td></tr><tr><td rowspan=1 colspan=2>192 0.1670.2920</td><td rowspan=1 colspan=4>.1780.299 0.187 0.307 0.187 0.343 0.1770.2990.176 0.299 0.226 0.344 0.176 0.315</td></tr><tr><td rowspan=1 colspan=2>336 0.3060.4000</td><td rowspan=1 colspan=1>.3280</td><td rowspan=1 colspan=3>.414 0.350 0.429 0.3530.473 0.3310.4170.301 0.397 0.3670.4480.313 0.427</td></tr><tr><td rowspan=1 colspan=2>720 0.7930.6690</td><td rowspan=1 colspan=1>.817 0</td><td rowspan=1 colspan=3>.6791.006 0.748 0.934 0.761 0.8470.6910.901 0.714 0.964 0.746 0.839 0.695</td></tr><tr><td rowspan=2 colspan=2>Weather 962 0.1 00.200</td><td rowspan=2 colspan=4>960.1520.200 0.166 0.207 0.157 0.205 0.163 0.209 0.1740.2140.186 0.227 0.172 0.220 0.195 0.252192 0.1980.2440.216 0.2540.204 0.2470.211 0.254 0.2210.2540.234 0.2650.219 0.261 0.2370.295336 0.251 0.2850.273 0.296 0.261 0.290 0.263 0.293 0.2780.2960.284 0.301 0.280 0.306 0.282 0.331720 0.3220.3470.351 0.346 0.340 0.341 0.3440.3480.3580.3470.3560.349 0.3650.3590.3450.382</td></tr><tr><td rowspan=1 colspan=1>192 0.198 0.244</td><td rowspan=1 colspan=4></td></tr></table>

Table 3: Detailed dataset–horizon comparison against all seven Table 1 baselines. Bold values mark the lowest reported three-decimal error in each row and metric, so MSE and MAE winners may come from different methods. TimeXer ETT/Weather values follow its original report, while its Exchange row follows the aligned input-length-96 reproduction of [23]; other external values follow their recorded aligned sources.
<table><tr><td>Dataset</td><td>Horizon</td><td>AnchorMTS</td><td>TimePro</td><td>Best MSE</td><td>Best MAE</td></tr><tr><td>ETTh1</td><td>96</td><td>0.382/0.401</td><td>0.375/0.398</td><td>TimePro 0.375</td><td>TimePro 0.398</td></tr><tr><td>ETTh1</td><td>192</td><td>0.430/0.430</td><td>0.427/0.429</td><td>TimePro 0.427</td><td>TimePro 0.429</td></tr><tr><td>ETTh1</td><td>336</td><td>0.463/0.445</td><td>0.472/0.450</td><td>TimeXer 0.468</td><td>TimeXer 0.448</td></tr><tr><td>ETTh1</td><td>720</td><td>0.471/0.470</td><td>0.476/0.474</td><td>TimeXer 0.469</td><td>TimeXer 0.461</td></tr><tr><td>ETTh2</td><td>96</td><td>0.280/0.331</td><td>0.293/0.345</td><td>TimeXer 0.286</td><td>TimeXer 0.338</td></tr><tr><td>ETTh2</td><td>192</td><td>0.357/0.382</td><td>0.367/0.394</td><td>TimeXer 0.363</td><td>TimeXer 0.389</td></tr><tr><td>ETTh2</td><td>336</td><td>0.408/0.421</td><td>0.419/0.431</td><td>TimeXer 0.414</td><td>TimeXer 0.423</td></tr><tr><td>ETTh2</td><td>720</td><td>0.415/0.435</td><td>0.427/0.445</td><td>TimeXer 0.408</td><td>TimeXer 0.432</td></tr><tr><td>ETTm1</td><td>96</td><td>0.314/0.348</td><td>0.326/0.364</td><td>TimeMixer 0.317</td><td>TimeXer 0.356</td></tr><tr><td>ETTm1</td><td>192</td><td>0.361/0.377</td><td>0.367/0.383</td><td>TimeXer 0.362</td><td>TimePro 0.383</td></tr><tr><td>ETTm1</td><td>336</td><td>0.393/0.408</td><td>0.402/0.409</td><td>TimeMixer 0.391</td><td>TimeMixer 0.406</td></tr><tr><td>ETTm1</td><td>720</td><td>0.447/0.440</td><td>0.469/0.446</td><td>TimeXer 0.452</td><td>TimeXer 0.441</td></tr><tr><td>ETTm2</td><td>96</td><td>0.168/0.254</td><td>0.178/0.260</td><td>TimeXer 0.171</td><td>TimeXer 0.256</td></tr><tr><td>ETTm2</td><td>192</td><td>0.230/0.299</td><td>0.242/0.303</td><td>TimeXer 0.237</td><td>TimeXer 0.299</td></tr><tr><td>ETTm2</td><td>336</td><td>0.293/0.336</td><td>0.303/0.342</td><td>TimeXer 0.296</td><td>TimeXer 0.338</td></tr><tr><td>ETTm2</td><td>720</td><td>0.384/0.399</td><td>0.400/0.399</td><td>TimeXer 0.392</td><td>TimeXer 0.394</td></tr><tr><td>Exchange</td><td>96</td><td>0.082/0.201</td><td>0.085/0.204</td><td>TimeXer 0.084</td><td>TimeXer 0.202</td></tr><tr><td>Exchange</td><td>192</td><td>0.167/0.292</td><td>0.178/0.299</td><td>PatchTST 0.176</td><td>TimePro 0.299</td></tr><tr><td>Exchange</td><td>336</td><td>0.306/0.400</td><td>0.328/0.414</td><td>PatchTST 0.301</td><td>PatchTST 0.397</td></tr><tr><td>Exchange</td><td>720</td><td>0.793/0.669</td><td>0.817/0.679</td><td>TimePro 0.817</td><td>TimePro 0.679</td></tr><tr><td>Weather</td><td>96</td><td>0.152/0.200</td><td>0.166/0.207</td><td>TimeXer 0.157</td><td>TimeXer 0.205</td></tr><tr><td>Weather</td><td>192</td><td>0.198/0.244</td><td>0.216/0.254</td><td>TimeXer 0.204</td><td>TimeXer 0.247</td></tr><tr><td>Weather</td><td>336</td><td>0.251/0.285</td><td>0.273/0.296</td><td>TimeXer 0.261</td><td>TimeXer 0.290</td></tr><tr><td>Weather</td><td>720</td><td>0.322/0.347</td><td>0.351/0.346</td><td>TimeXer 0.340</td><td>TimeXer 0.341</td></tr></table>

Table 4: Complete fixed-identity AnchorMTS grid and the cellwise best Table 1 references. Best MSE and Best MAE are selected independently from all seven displayed baselines in each cell.

## C.5. Closest Multimodal Forecasting Protocols

Broader multi-view forecasting work spans several fusion objectives. CAMul dynamically weights view-specific information and uncertainty for calibrated probabilistic forecasts [17]. TEFN aggregates time- and channel-dimensional evidence through basic probability assignments [16]. T3Time aligns time, spectral, and prompt branches with adaptive multi-head fusion [14], while TriTS combines time, frequency, and visual representations [15]. These studies motivate richer multi-view forecasting representations and fusion mechanisms. AnchorMTS focuses on a different design axis: auxiliary representations are allowed to refine a complete numerical forecast, while their final influence is independently constrained in the reported prediction space.

<table><tr><td>Dataset</td><td>Horizon</td><td>d</td><td> $d _ { f f }$ </td><td>Layers</td><td>Batch</td><td>LR</td><td>Loss</td></tr><tr><td>ETTh1</td><td>96</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTh1</td><td>192</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTh1</td><td>336</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTh1</td><td>720</td><td>128</td><td>512</td><td>1</td><td>16</td><td>1.0e-04</td><td>MSE</td></tr><tr><td>ETTh2</td><td>96</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTh2</td><td>192</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTh2</td><td>336</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTh2</td><td>720</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTm1</td><td>96</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTm1</td><td>192</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTm1</td><td>336</td><td>128</td><td>1024</td><td>1</td><td>4</td><td>1.0e-04</td><td>MSE</td></tr><tr><td>ETTm1</td><td>720</td><td>256</td><td>1024</td><td>1</td><td>16</td><td>2.0e-04</td><td>MSE</td></tr><tr><td>ETTm2</td><td>96</td><td>512</td><td>2048</td><td>3</td><td>32</td><td>5.0e-05</td><td>Huber</td></tr><tr><td>ETTm2</td><td>192</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTm2</td><td>336</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>ETTm2</td><td>720</td><td>128</td><td>512</td><td>1</td><td>16</td><td>1.0e-04</td><td>MSE</td></tr><tr><td>Exchange</td><td>96</td><td>256</td><td>2048</td><td>1</td><td>32</td><td>1.0e-04</td><td>MSE</td></tr><tr><td>Exchange</td><td>192</td><td>256</td><td>1024</td><td>1</td><td>16</td><td>2.0e-04</td><td>MSE</td></tr><tr><td>Exchange</td><td>336</td><td>128</td><td>1024</td><td>1</td><td>4</td><td>1.0e-04</td><td>MSE</td></tr><tr><td>Exchange</td><td>720</td><td>256</td><td>1024</td><td>1</td><td>16</td><td>2.0e-04</td><td>MSE</td></tr><tr><td>Weather</td><td>96</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>Weather</td><td>192</td><td>512</td><td>2048</td><td>3</td><td>32</td><td>5.0e-05</td><td>Huber</td></tr><tr><td>Weather</td><td>336</td><td>512</td><td>2048</td><td>2</td><td>32</td><td>1.0e-04</td><td>Huber</td></tr><tr><td>Weather</td><td>720</td><td>512</td><td>1024</td><td>1</td><td>8</td><td>5.0e-05</td><td>MSE</td></tr></table>

Table 5: Selected ordinary training and temporal-capacity hyperparameters. Every row retains the identical multimodal protocol.
<table><tr><td>Method</td><td>Published regime</td><td>Protocol coverage relative to Table 1</td><td>Role here</td></tr><tr><td>VisionTS [10]</td><td>Zero-shot plus one-epoch full- shot</td><td>Dataset-specific context lengths (576–2304); Exchange ab- Visual-forecasting context sent; no complete aligned 6 × 4 grid.</td><td></td></tr><tr><td>Time-VLM [11]</td><td>Few-shot, zero-shot, and super- vised</td><td>Long-term input length 512; Exchange absent; no complete Vision-language forecasting aligned 6 × 4 grid.</td><td>context</td></tr><tr><td>DMMV [12]</td><td>Full-data supervised</td><td>Default look-back 336; Exchange absent; no complete aligned 6 × 4 grid.</td><td>Alternative-view fusion con-</td></tr><tr><td>TimesCLIP [13]</td><td>Supervised aligned views</td><td>Input length 96, but ETTh1/ETTh2 and horizons 336/720 are not reported.</td><td>text Cross-view alignment context</td></tr></table>

Table 6: Published evaluation protocols of the closest multimodal forecasters. Their reported grids differ from Table 1 in context length, dataset coverage, and/or horizon coverage; partial results are therefore not mixed into the 24-setting macro comparison.

<table><tr><td>Dataset</td><td>Variables</td><td>Frequency</td><td>Split</td></tr><tr><td>ETTh1/ETTh2</td><td>7</td><td>hourly</td><td>12/4/4 months</td></tr><tr><td>ETTm1/ETTm2</td><td>7</td><td>15-minute</td><td>12/4/4 months</td></tr><tr><td>Exchange</td><td>8</td><td>daily</td><td>70/10/20%</td></tr><tr><td>Weather</td><td>21</td><td>10-minute</td><td>70/10/20%</td></tr></table>

Table 7: Dataset metadata and chronological train/validation/test partitions. Input length is 96 and horizons are 96, 192, 336, and 720.

Table 6 audits the public paper protocols rather than treating unlike regimes as interchangeable results.

## D.1. Representation Example

One observed window, four automatically aligned representations (a) Historical window X

![](images/6665d9dece6e790097c8be28234d9883505a2db62a4b2c6e934eff56cb733fef.jpg)  
Fig. 4: Representations generated from one Exchange test history. The variable-wise GAF preserves channel-specific pairwise geometry; line and recurrence views expose trajectory and return structure; the caption summarizes the same observed window. No target-window values are used. The GAF is shown in grayscale for display; caption generation uses the corresponding color rendering.

## D.2. Retrieval Details

The main paper reports the macro retrieval lifts; this section gives the descriptor definitions and sampling protocol.

The retrieval analysis uses up to 128 non-overlapping training windows per dataset with stride at least 96. Trend is absolute standardized slope; periodicity is the largest non-DC Fourier amplitude divided by total non-DC amplitude. Recurrence is the dispersion of diagonal means of $\exp ( - | s _ { i } - s _ { j } | / ( \mathrm { s t d } | s _ { i } - s _ { j } | + \epsilon ) )$ , and channel diversity is the across-channel dispersion of dominant spectral ratios. Cosine similarity retrieves five neighbors from frozen postlayer-normalized CLIP class tokens. For descriptor d, define

$$
\begin{array} { r l } & { D _ { \mathrm { \mathcal { N } } } = \mathbb { E } _ { i , j \in \mathcal { N } _ { 5 } ( i ) } | d _ { i } - d _ { j } | , } \\ & { D _ { \mathrm { a l l } } = \mathbb { E } _ { i < j } | d _ { i } - d _ { j } | , } \end{array} \quad \quad \mathrm { l i f t = 1 - } D _ { \mathrm { \mathcal { N } } } / D _ { \mathrm { a l l } } .
$$

## E. REPRODUCTION ENTRY POINTS

export\_selected\_forecast\_decompositions.py and export\_final\_projected\_controls.py replay checkpoints without training and save candidate, anchor, and final arrays. build\_final\_result\_manifest.py recomputes projections, metrics, paired intervals, exact sign tests, and violation counts; generate\_final\_tables.py and the paper plotting scripts regenerate the reported artifacts. The first-stage, refinement, selected-configuration, caption, and exact-bound manifests retain the source paths and recorded protocol fields used above.

## F. REFERENCES

[1] Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long, “Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting,” Advances in neu-

ral information processing systems, vol. 34, pp. 22419–22430, 2021.

[2] Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin, “Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting,” in International conference on machine learning. PMLR, 2022, pp. 27268– 27286.

[3] Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam, “A time series is worth 64 words: Long-term forecasting with transformers,” arXiv preprint arXiv:2211.14730, 2022.

[4] Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long, “Timesnet: Temporal 2d-variation modeling for general time series analysis,” arXiv preprint arXiv:2210.02186, 2022.

[5] Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long, “itransformer: Inverted transformers are effective for time series forecasting,” in International conference on learning representations, 2024, vol. 2024, pp. 11116–11140.

[6] Shiyu Wang, Haixu Wu, Xiaoming Shi, Tengge Hu, Huakun Luo, Lintao Ma, James Zhang, and Jun Zhou, “Timemixer: Decomposable multiscale mixing for time series forecasting,” in International conference on learning representations, 2024, vol. 2024, pp. 38626–38652.

[7] Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Guo Qin, Haoran Zhang, Yong Liu, Yunzhong Qiu, Jianmin Wang, and Mingsheng Long, “Timexer: Empowering transformers for time series forecasting with exogenous variables,” Advances in neural information processing systems, vol. 37, pp. 469–498, 2024.

[8] Zhiguang Wang and Tim Oates, “Imaging time-series to improve classification and imputation,” arXiv preprint arXiv:1506.00327, 2015.

[9] Jean-Pierre Eckmann, S Oliffson Kamphorst, David Ruelle, et al., “Recurrence plots of dynamical systems,” World Scientific Series on Nonlinear Science Series A, vol. 16, pp. 441– 446, 1995.

[10] Mouxiang Chen, Lefei Shen, Zhuo Li, Xiaoyun Joy Wang, Jianling Sun, and Chenghao Liu, “Visionts: Visual masked autoencoders are free-lunch zero-shot time series forecasters,” arXiv preprint arXiv:2408.17253, 2024.

[11] Zhong Siru, Ruan Weilin, Ming Jin, Li Huan, Wen Qingsong, and Liang Yuxuan, “Time-vlm: Exploring multimodal vision-language models for augmented time series forecasting,” in Forty-Second International Conference on Machine Learning (ICML 2025). Proceedings of Machine Learning Research, 2025.

[12] ChengAo Shen, Wenchao Yu, Ziming Zhao, Dongjin Song, Wei Cheng, Haifeng Chen, and Jingchao Ni, “Multi-modal view enhanced large vision models for long-term time series forecasting,” Advances in Neural Information Processing Systems, vol. 38, pp. 93269–93296, 2026.

[13] Sixun Dong, Wei Fan, Teresa Wu, and Yanjie Fu, “Teaching time series to see and speak: Forecasting with aligned visual and textual perspectives,” arXiv preprint arXiv:2506.24124, 2025.

[14] Abdul Monaf Chowdhury, Rabeya Akter, and Safaeid Hossain Arib, “T3time: Tri-modal time series forecasting via adaptive multi-head alignment and residual fusion,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2026, vol. 40, pp. 20597–20605.

[15] Xiang Ao, “Trits: Time series forecasting from a multimodal perspective,” arXiv preprint arXiv:2604.16748, 2026.

[16] Tianxiang Zhan, Yuanpeng He, Yong Deng, Zhen Li, Wenjie Du, and Qingsong Wen, “Time evidence fusion network: Multi-source view in long-term time series forecasting,” IEEE transactions on pattern analysis and machine intelligence, 2025.

[17] Harshavardhan Kamarthi, Lingkai Kong, Alexander Rodr´ıguez, Chao Zhang, and B Aditya Prakash, “Camul: calibrated and accurate multi-view time-series forecasting,” in Proceedings of the ACM Web Conference 2022, 2022, pp. 3174–3185.

[18] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[19] Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang, “Informer: Beyond efficient transformer for long sequence time-series forecasting,” in Proceedings of the AAAI conference on artificial intelligence, 2021, vol. 35, pp. 11106–11115.

[20] Guokun Lai, Wei-Cheng Chang, Yiming Yang, and Hanxiao Liu, “Modeling long-and short-term temporal patterns with deep neural networks,” in The 41st international ACM SI-GIR conference on research & development in information retrieval, 2018, pp. 95–104.

[21] Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu, “Are transformers effective for time series forecasting?,” in Proceedings of the AAAI conference on artificial intelligence, 2023, vol. 37, pp. 11121–11128.

[22] Xiaowen Ma, Zhenliang Ni, Shuai Xiao, and Xinghao Chen, “Timepro: Efficient multivariate long-term time series forecasting with variable-and time-aware hyper-state,” arXiv preprint arXiv:2505.20774, 2025.

[23] Vahid Naghashi, Mounir Boukadoum, and Abdoulaye Banire Diallo, “Should we reconsider rnns for time-series forecasting?,” AI, vol. 6, no. 5, pp. 90, 2025.