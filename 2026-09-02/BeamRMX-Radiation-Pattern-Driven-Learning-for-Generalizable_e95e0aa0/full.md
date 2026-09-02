# BeamRMX: Radiation-Pattern-Driven Learning for Generalizable Beam Radio Map Prediction and Beam Management

Yue Zhang , Student Member, IEEE, Xiucheng Wang , Graduate Student Member, IEEE, Wenshuo Chen, Student Member, IEEE, and Nan Cheng , Senior Member, IEEE

Abstract—The evolution toward sixth-generation (6G) wireless networks is driving larger antenna arrays and highly directional multi-beam transmission, making accurate knowledge of beamdependent spatial coverage important for beam management and environment-aware network operation. Radio maps (RMs) provide such a representation, yet conventional RM prediction assumes omnidirectional or transmitter-level radiation. In beamformed multiple-input multiple-output (MIMO) systems, one propagation scene instead gives rise to many configuration-dependent beam radio maps (BeamRMs), creating challenges in beam representation and generalization. Existing methods either condition prediction on beam descriptors or use beam maps as auxiliary inputs to generic architectures. We propose BeamRMX, which, to the best of our knowledge, is the first dedicated framework to treat the spatial radiation pattern as the primary BeamRM query and learn how scene geometry transforms it into the received power field. XBase learns multiscale interactions between the radiation query and scene geometry, while an optional Evidence Adapter uses a few cross-configuration BeamRMs from the same scene. Matched-domain and zero-shot experiments show consistent gains over deterministic and diffusion baselines, including mean absolute error reductions of 26.1% on unseen scenes and 47.8% on an unseen configuration. Cross-configuration evidence further improves reconstruction and intra-sector beam refinement. Code is available at https://github.com/ZY021023/BeamRMX-code.

Index Terms—Beam radio map, radio map prediction, XL-MIMO, spatial excitation, scene generalization, cross-configuration evidence, beam management.

## I. INTRODUCTION

The evolution toward sixth-generation (6G) wireless networks is driving the use of wider frequency bands, larger antenna arrays, and increasingly directional multi-beam transmission [1]–[7]. While beamforming provides substantial array gain and spatial selectivity, it also makes the resulting radio environment strongly dependent on the active beam and radio configuration. A beamforming state determines the transmitted radiation pattern but does not directly reveal the received power distribution after interaction with the propagation environment. Accurate knowledge of such beam-dependent spatial radio conditions is therefore important not only for beam management but also for coverage assessment, network planning, and other environment-aware wireless functions.

Radio maps (RMs) provide a natural representation of spatial radio conditions by associating geographical locations with received signal characteristics [8]–[10]. Existing RM prediction methods have achieved substantial progress, but most assume omnidirectional or transmitter-level radiation, under which a propagation scene is associated with a largely fixed spatial response [11]–[14]. This assumption no longer holds in beamformed multiple-input multiple-output (MIMO) systems. Even for the same transmitter and environment, different carrier frequencies, array configurations, and beamforming states can produce substantially different received power fields, as illustrated in Fig. 1. We refer to these configuration-dependent spatial responses as beam radio maps (BeamRMs).

Despite substantial progress in learning-based RM prediction, dedicated work on BeamRM prediction remains limited. Recent studies have begun to model beam-dependent spatial responses, but how the target beam should be represented to the predictor remains a central issue. BeamCKM performs beam-aware channel-map construction through codebook-dependent beam representations, while BeamCKMDiff extends the conditioning variable to continuous beamforming vectors [15], [16]. The former distinguishes radiation states through predefined beam identities, whereas the latter provides a continuous description of the beamforming state. In both cases, however, the beam is represented primarily in the parameter domain, leaving the network to infer how the corresponding radiation pattern is distributed over the prediction region. This dependence on beam-specific parameterization becomes increasingly restrictive when prediction must extend across beam states, codebooks, or array configurations that differ from those seen during training.

The recently introduced beam map provides a more explicit representation by analytically projecting the radiation pattern associated with each radio configuration onto the prediction region [13]. In the original extremely large-scale MIMO (XL-MIMO) benchmark, beam maps are supplied as physicsinformed spatial features to existing predictors. BeamRMX takes a further step by using this spatial radiation representation as the primary query for BeamRM prediction rather than as an auxiliary input. Beam indices, beamforming vectors, codebooks, and array configurations are thereby expressed in a common spatial domain before propagation is learned. This representation depends on the resulting spatial radiation pattern rather than on the particular parameters used to generate it, providing a more suitable basis for prediction across heterogeneous beam and array configurations.

![](images/1124012ef7ea7bfefd539ae9fe63d84eb81172cd7f5ad437335f3e4cad9e9480.jpg)  
A fixed propagation scene corresponds to a family of beam-dependent radio maps.  
Fig. 1. BeamRM concept: a common propagation scene transforms different beamforming-induced radiation patterns into distinct received power maps.

Making the beam query spatial also exposes a second gap in existing methods. The spatial radiation pattern specifies how power is launched toward the environment, whereas scene geometry determines how that power is blocked, attenuated, and redistributed. Existing benchmarks using beam maps mainly concatenate these two information sources and rely on generic prediction architectures to learn their interaction implicitly [13]. Thus, although the beam map reduces the representation gap, a dedicated learning method is still needed to model how the propagation scene transforms the specified radiation pattern into the received power field. BeamRMX addresses this gap through XBase, its complete query-only predictor, which encodes the spatial radiation query and scene geometry separately and learns their multiscale interaction. To the best of our knowledge, XBase is the first architecture designed specifically for this formulation.

Query-only prediction is essential when no radio observation is available for a new deployment. In practice, however, a new target configuration rarely appears in a completely unobserved environment. As arrays grow larger and beams become narrower, the number of beam states required to characterize a deployment increases substantially, making exhaustive per-beam measurement or repeated high-fidelity simulation increasingly costly. Meanwhile, wireless networks evolve incrementally rather than being rebuilt from scratch. Before introducing a high-frequency, large-array, or narrowbeam configuration, the same area may already have been characterized under legacy frequency bands, smaller arrays, or wider-beam configurations. Measurements, simulations, and historical BeamRMs accumulated during earlier deployment stages therefore constitute a potentially valuable source of information for predicting the new configuration.

Source BeamRMs from the target scene contain realized information about scene-dependent blockage, attenuation, and multipath. However, they are not sparse observations of the target BeamRM because each source map corresponds to a different spatial excitation and cannot be directly copied or interpolated into the target field. The source-assisted problem is therefore to extract propagation information that is shared across configurations and use it to reduce the uncertainty of query-only prediction without observing the target BeamRM or updating model parameters at inference time.

BeamRMX addresses this setting with an optional Evidence Adapter. When source BeamRMs are unavailable, XBase operates as a complete query-only predictor. When a few source BeamRMs from the same scene are available, the adapter interprets them together with their corresponding source excitations and refines the frozen XBase prediction using the transferable cross-configuration evidence. The target configuration is excluded from the support set, and no target BeamRM observation or parameter update is required during inference. The main contributions of this paper are summarized as follows.

1) A spatial-excitation-conditioned formulation of BeamRM prediction: Building on the beam map representation, we elevate spatial excitation from an auxiliary arrayconditioning feature to the primary representation of a BeamRM query. This places heterogeneous beam and configuration states in a common spatial query domain and reformulates BeamRM prediction as learning how a propagation scene transforms a specified excitation field into the realized received power field, rather than requiring the predictor to infer the spatial radiation implicitly from configuration-dependent descriptors. Because the formulation depends on the spatial radiation itself rather than its particular parameterization, it can, in principle, extend beyond fixed codebooks and array beamforming to other spatially representable radiating-source patterns.

2) The first dedicated architecture for spatial-excitationconditioned BeamRM learning: To the best of our knowledge, XBase is the first dedicated learning architecture explicitly designed around this excitation-to-field formulation. It maintains distinct representations of spatial excitation and scene geometry and learns their structured interaction rather than relying on generic beam conditioning or input concatenation. The proposed method is systematically evaluated under matched-domain, scenedisjoint, and configuration-disjoint protocols to examine both learning effectiveness and query-only generalization.

3) Source-assisted BeamRM prediction for incremental network evolution: We introduce an Evidence Adapter that exploits a few same-scene BeamRMs from legacy or lower-complexity radio configurations as crossconfiguration propagation evidence for an unobserved target BeamRM. The adapter refines the frozen queryonly prediction while preserving an exact query-only path and requiring neither target BeamRM observations nor test-time parameter updates. Evaluation under the strict-small protocol demonstrates that limited historical radio information can substantially improve BeamRM prediction in unseen scenes and enhance downstream intra-sector beam refinement.

The remainder of this article is organized as follows. Section II reviews related work, and Section III formulates the query-only and source-assisted BeamRM prediction problems. Section IV presents the proposed BeamRMX framework, while Section V reports the experimental results and discussion. Finally, Section VI concludes the article.

## II. RELATED WORK

This section reviews learning-based radio map prediction, beam-aware representation, and generalization with auxiliary radio evidence.

## A. Learning-Based Radio Map Prediction

Representative approaches include geometry-conditioned encoder–decoder networks, adversarial models, graph neural networks and diffusion-based predictors [11], [12], [17], [18]. Diffusion-based RM prediction builds on denoising diffusion probabilistic modeling [19], while recent physics-informed variants incorporate propagation-related constraints into generative reconstruction [20]. Propagation-aware learning has also been explored by explicitly exploiting environmental semantics and geometry-dependent propagation structure [21]. Public datasets and benchmark challenges have further promoted standardized evaluation of learning-based radio map prediction [22].

These methods establish learning-based prediction as an efficient alternative to repeated measurement or ray tracing. However, most of these methods target transmitter-level radio maps under omnidirectional or weakly directional assumptions. Their primary task is therefore to learn an environment-tocoverage mapping, while the configuration-dependent spatial excitation produced by a beam is either omitted or represented only through auxiliary parameters. Direct application to BeamRM prediction consequently leaves a substantial part of the beam-dependent radiation behavior to be inferred from training data.

## B. Beam Representation for Beam-Aware Radio Map Prediction

More broadly, channel knowledge maps (CKMs) provide site-specific representations of channel-related information for environment-aware communications [23]. Directional information has been incorporated into radio map and channel map predictors through antenna-pattern projections, beam identities, and configuration descriptors. Element-pattern or directionalgain projections expose spatial directivity more explicitly than isotropic transmitter models [24], but generally do not capture the joint effects of array geometry, coherent beamforming, and carrier frequency.

BeamCKM represents different radiation states through codebook-dependent beam conditioning [15]. Such a representation distinguishes beams within a predefined family but does not explicitly describe how each beam illuminates the prediction region. BeamCKMDiff further replaces discrete beam identities with continuous beamforming-vector conditioning [16], providing a richer description of the radiation state. Nevertheless, the beamforming vector remains a global descriptor, and the predictor must still infer how the underlying array and beamforming parameters translate into a spatial radiation pattern. These approaches therefore represent the beam primarily in the parameter domain rather than directly in the spatial domain where propagation occurs.

The beam map representation addresses this gap by analytically converting configuration-dependent radiation into a spatial feature over the prediction region [13]. It provides an explicit grid-aligned radiation prior before environmentdependent blockage and multipath are introduced. The original multi-configuration benchmark incorporates beam maps into existing predictors mainly through feature concatenation and demonstrates clear advantages over scalar configuration encoding, including under cross-configuration and cross-environment evaluation.

These studies mainly establish what directional information should be supplied to a BeamRM predictor. They do not fully address how that information should be processed. Existing models that incorporate beam maps largely retain generic imageto-image architectures and place spatial excitation and envi ronmental information in a common feature stream, although the former specifies the radiation presented to the environment and the latter determines how that radiation is transformed. This leaves a distinct methodological gap between spatial beam representation and dedicated learning of the scene-conditioned excitation-to-field transformation.

## C. Generalization and Radio-Evidence-Assisted Prediction

BeamRM generalization involves two complementary distribution shifts. Scene-disjoint evaluation tests transfer to unseen geographical layouts, whereas cross-configuration evaluation changes the frequency, antenna, radiation pattern, or beam condition presented to the environment. Previous studies have examined cross-environment prediction, cross-frequency transfer, antenna-pattern variation, and cross-beam generalization under different protocols [13], [15], [16], [18]. A model that transfers across scenes does not necessarily generalize to unseen radiation conditions, and vice versa.

Auxiliary radio information has also been used to improve radio map construction. Sparse reconstruction methods assume that a subset of the target map is directly observed [17], [18], [25]–[28], while few-shot transfer and test-time adaptation specialize a predictor using limited target-domain information [29], [30]. BeamCKM further introduces M3ChanNet, which exploits environmental profiles and real-time multi-beam observations as side information for beam-aware map construction [15]. These approaches demonstrate the value of incorporating additional radio information, but differ from the source-assisted setting considered here. Our support consists of a few BeamRMs from other radio configurations in the same scene, while no same-scene BeamRM from the target configuration is observed.

Sparse target observations and cross-configuration source BeamRMs may both provide radio evidence, but their semantics are fundamentally different. The former directly samples the target field; the latter describes how the same environment has transformed other spatial excitations. Likewise, unlike few-shot or test-time parameter adaptation, the source-assisted setting considered here uses such cross-configuration responses as feed-forward evidence without updating model parameters. A suitable predictor must therefore preserve a reliable query-only path, extract only transferable scene information from source maps, and avoid treating source responses as direct substitutes for the target BeamRM.

Prior work has separately advanced geometry-based RM prediction, beam representation, and the use of auxiliary radio information. Generalizable BeamRM prediction requires these elements to be combined: heterogeneous radiation conditions need a common spatial representation, the scene-dependent transformation of that excitation must be learned, and source BeamRMs must be exploited without observing or optimizing on the target BeamRM. These requirements motivate the formulation considered here.

## III. PROBLEM FORMULATION

We formulate BeamRM prediction by separating three information roles: the target spatial excitation, the propagation scene, and optional same-scene radio evidence from other configurations. The first two define query-only prediction, whereas the third enables source-assisted prediction.

## A. BeamRM Prediction Conditioned on Spatial Excitation

Consider a propagation scene $s \in S$ represented over an $H \times W$ spatial region by

$$
\mathbf { G } _ { s } \in \mathbb { R } ^ { C _ { g } \times H \times W } ,\tag{1}
$$

where $\mathbf { G } _ { \varepsilon }$ contains the environmental information available to the predictor. Let Ω<sub>s</sub> denote the corresponding spatial evaluation grid, including its coordinate registration relative to the transmitter.

A target radiation query is denoted by $q = \left( c , \rho \right)$ , where c specifies the underlying radio configuration, such as carrier frequency, array architecture, and antenna characteristics, while $\rho$ specifies the directional radiation state, such as the beamforming or steering state. Following the beam map formulation in [13], its raw beam map over $\Omega _ { s }$ is deterministically computed as

$$
\mathbf { B } _ { \Omega _ { s } , q } = \Psi _ { \mathrm { b e a m } } ( q ; \Omega _ { s } ) \in \mathbb { R } ^ { H \times W } .\tag{2}
$$

The subscript s reflects only the evaluation grid and its transmitter-relative registration, rather than scene-dependent propagation. Once $q$ and $\Omega _ { s }$ are fixed, the beam map contains no scene-dependent blockage or multipath information. We write $\mathbf { B } _ { s , q } \equiv \mathbf { B } _ { \Omega _ { s } , q }$ hereafter.

The model-ready spatial excitation is constructed as

$$
\mathbf { E } _ { s , q } = \phi _ { \mathrm { e x c } } \left( \mathbf { B } _ { s , q } ; \Omega _ { s } \right) \in \mathbb { R } ^ { C _ { e } \times H \times W } ,\tag{3}
$$

where $\phi _ { \mathrm { e x c } } ( \cdot )$ combines the beam map with transmitter-relative spatial cues. Thus, ${ \bf E } _ { s , q }$ serves as the spatial representation of the target radiation state. In the query-only path, the radio query $q$ affects the prediction through the beam map $\mathbf { B } _ { s , q } ,$ and no separate query embedding is supplied to XBase.

The target BeamRM and its valid-region mask are denoted by

$$
\begin{array} { r } { \mathbf { Y } _ { s , q } \in \mathbb { R } ^ { H \times W } , \qquad \mathbf { M } _ { s , q } \in \{ 0 , 1 \} ^ { H \times W } , } \end{array}\tag{4}
$$

where $\mathbf { Y } _ { s , q }$ is the corresponding received power field and $\mathbf { M } _ { s , q }$ identifies valid pixels for supervision and evaluation. The mask itself is not treated as target-side radio evidence.

For a fixed scene, different radiation queries induce different spatial excitations and consequently different BeamRMs. We represent the underlying propagation process as

$$
\mathbf { Y } _ { s , q } = \mathcal { T } _ { s } ( \mathbf { E } _ { s , q } ) ,\tag{5}
$$

where $\mathcal { T } _ { s }$ denotes the unknown scene-conditioned propagation transformation. BeamRM prediction is therefore formulated as learning how a scene transforms a specified spatial excitation into the corresponding received power field, rather than directly associating a configuration descriptor with an output map.

## B. Query-Only Generalization Regimes

Without same-scene radio evidence, prediction follows

$$
\begin{array} { r } { \widehat { \mathbf { Y } } _ { s , q } ^ { 0 } = F _ { \theta } ^ { 0 } \left( \mathbf { G } _ { s } , \mathbf { E } _ { s , q } \right) , } \end{array}\tag{6}
$$

where the superscript 0 denotes query-only prediction.

We consider three complementary regimes. The matched regime evaluates learning effectiveness when training and test samples follow the same scene–configuration distribution. The two zero-shot regimes satisfy

$$
\begin{array} { r l } { { \mathrm { S c e n e - Z S : } } } & { { S _ { \mathrm { t r } } \cap S _ { \mathrm { t e } } = \emptyset , } } \\ { { \mathrm { C o n f i g - Z S : } } } & { { C _ { \mathrm { t r } } \cap \mathcal { C } _ { \mathrm { t e } } = \emptyset , } } \end{array}\tag{7}
$$

where $s$ and C denote scene and radio-configuration sets, respectively. Scene-ZS tests transfer to unseen propagation environments, whereas Config-ZS tests transfer to radio configurations entirely absent from training, including all of their directional states. The two regimes are evaluated independently to isolate the corresponding distribution shifts. No same-scene BeamRM is available at inference time in any query-only regime.

## C. Strict-Small Source-Assisted Prediction

We further consider an unseen target scene s for which a small number of BeamRMs are available under other radio

configurations. Let $q _ { i } = \left( c _ { i } , \rho _ { i } \right)$ denote the ith source query in the same scene. The K-shot support set is

$$
\mathcal { Z } _ { s , q } ^ { K } = \{ ( \mathbf { B } _ { s , q _ { i } } , \mathbf { Y } _ { s , q _ { i } } , \mathbf { M } _ { s , q _ { i } } , q _ { i } ) \} _ { i = 1 } ^ { K } ,\tag{8}
$$

where each source BeamRM is paired with the query and beam map under which it was generated. Source-assisted prediction is then written as

$$
\begin{array} { r } { \widehat { \mathbf { Y } } _ { s , q } ^ { K } = F _ { \theta } \left( \mathbf { G } _ { s } , \mathbf { E } _ { s , q } , q , \mathcal { Z } _ { s , q } ^ { K } \right) . } \end{array}\tag{9}
$$

Under the strict-small protocol, every support configuration satisfies

$$
c _ { i } \neq c , \qquad c _ { i } \in \mathcal { C } _ { \mathrm { s m a l l } } ( c ) , \qquad i = 1 , \dots , K ,\tag{10}
$$

where $\mathcal { C } _ { \mathrm { s m a l l } } ( c )$ is a predefined set of configurations with lower deployment complexity than the target configuration. Consequently, the target configuration is excluded from the support pool, no target BeamRM pixel is observed, and model parameters remain fixed during inference.

Although the source BeamRMs share the target propagation scene, they correspond to different spatial excitations. They therefore provide realized cross-configuration evidence of the common environment rather than direct observations of the target field.

Based on the above definitions, the overall BeamRM prediction task is formulated as follows.

Problem 1:

$$
\begin{array} { r l } { \underset { \theta } { \mathrm { m i n i m i z e } } } & { \mathbb { E } \Big [ \mathcal { L } _ { \mathrm { R M } } \Big ( \widehat { \mathbf { Y } } _ { s , q } ^ { K } , { \mathbf { Y } } _ { s , q } ; { \mathbf { M } } _ { s , q } \Big ) \Big ] } \\ { \mathrm { s . t . } } & { \widehat { \mathbf { Y } } _ { s , q } ^ { K } = F _ { \theta } \big ( { \mathbf { G } } _ { s } , { \mathbf { E } } _ { s , q } , q , \mathcal { Z } _ { s , q } ^ { K } \big ) , } \\ & { F _ { \theta } ( \mathbf { G } _ { s } , { \mathbf { E } } _ { s , q } , q , \mathcal { O } ) = F _ { \theta } ^ { 0 } ( \mathbf { G } _ { s } , { \mathbf { E } } _ { s , q } ) , } \end{array}\tag{11}
$$

where $\mathcal { L } _ { \mathrm { R M } } ( \cdot )$ denotes a mask-aware BeamRM reconstruction loss, with $\mathcal { Z } _ { s , q } ^ { 0 } = \emptyset$ and, for $K > 0$ , support configurations constrained by (10). Hence, $K = 0$ gives query-only prediction, whereas $K > 0$ permits strict-small source-assisted prediction without target BeamRM observations or inference-time parameter updates.

## IV. PROPOSED BEAMRMX FRAMEWORK

BeamRMX instantiates the formulation in Section III through two asymmetric components. XBase serves as the complete query-only predictor and approximates the scene-conditioned excitation-to-field transformation $\mathcal { T } _ { s }$ . When same-scene crossconfiguration evidence is available, an optional Evidence Adapter refines the frozen XBase prediction through a gated residual correction.

Throughout this section, power-valued beam maps and BeamRMs are represented on a common affine normalized scale in [0, 1], and $\Pi _ { Y } ( \cdot )$ denotes element-wise clipping to this range. Because the same affine mapping is used, normalized power differences remain proportional to their dB-domain counterparts.

## A. Framework Overview

Given the scene representation $\mathbf { G } _ { s }$ and target spatial excitation $\mathbf { E } _ { s , q }$ , XBase produces

$$
\left( \widehat { \mathbf { Y } } _ { s , q } ^ { 0 } , \mathbf { Z } _ { s , q } ^ { 0 } \right) = F _ { X } \left( \mathbf { G } _ { s } , \mathbf { E } _ { s , q } ; \theta _ { X } \right) ,\tag{12}
$$

where $\widehat { \mathbf { Y } } _ { s , q } ^ { 0 }$ is the complete query-only BeamRM and $\mathbf { Z } _ { s , q } ^ { 0 }$ is the target-side representation exposed to the Evidence Adapter.

For $K > 0$ , the adapter estimates a residual correction from the frozen XBase anchor and the support set $\mathcal { Z } _ { s , q } ^ { K }$

$$
\begin{array} { r } { \Delta _ { s , q } ^ { K } = A \Big ( \widehat { \mathbf { Y } } _ { s , q } ^ { 0 } , \mathbf { Z } _ { s , q } ^ { 0 } , \mathbf { G } _ { s } , q , \mathcal { Z } _ { s , q } ^ { K } ; \theta _ { A } \Big ) , } \end{array}\tag{13}
$$

$$
\widehat { \mathbf { Y } } _ { s , q } ^ { K } = \Pi _ { Y } \left( \widehat { \mathbf { Y } } _ { s , q } ^ { 0 } + \Delta _ { s , q } ^ { K } \right) , \Delta _ { s , q } ^ { 0 } = \mathbf { 0 } .\tag{14}
$$

Thus, XBase remains a standalone predictor rather than a preliminary stage requiring subsequent correction. For $K =$ 0, the complete support branch is structurally bypassed and BeamRMX returns $\widehat { \mathbf { Y } } _ { s , q } ^ { 0 }$ exactly, as illustrated in Fig. 2.

## B. XBase: Scene-Conditioned Excitation-to-Field Learning

As shown in Fig. 2(a), XBase preserves the distinct roles of spatial excitation and propagation geometry before explicitly learning their interaction. It therefore processes $\mathbf { E } _ { s , q }$ and $\mathbf { G } _ { s }$ through separate four-stage ConvNeXt V2 encoders [31]:

$$
\begin{array} { r l } & { \left\{ \mathbf { U } _ { e } ^ { \left( \ell \right) } \right\} _ { \ell = 1 } ^ { L } = E _ { e } \left( \mathbf { E } _ { s , q } \right) , } \\ & { \left\{ \mathbf { U } _ { g } ^ { \left( \ell \right) } \right\} _ { \ell = 1 } ^ { L } = E _ { g } \left( \mathbf { G } _ { s } \right) . } \end{array}\tag{15}
$$

Here, $\mathbf { E } _ { s , q }$ is the spatial representation of the target radiation state. Its configuration dependence is encoded through the target beam map contained in $\mathbf { E } _ { s , q } ;$ XBase does not use an additional global query embedding. The excitation stream contains the target beam map and transmitter-relative spatial cues, whereas the geometry stream represents building and boundary-related scene structure.

In implementation, the beam-conditioned encoder receives five channels comprising the target beam map, transmitterlocation map, two normalized coordinate maps, and transmitterdistance map. The geometry encoder receives 11 channels describing building height and occupancy, scene edges and boundary bands, boundary distance, horizontal and vertical height gradients, corner responses, normalized coordinates, and transmitter distance. The target BeamRM is never included in either input stream.

At each resolution level, the two streams interact through cross-conditioned fusion:

$$
\begin{array} { r l } & { \biggl [ \boldsymbol { \Gamma } _ { e } ^ { ( \ell ) } , \boldsymbol { \Gamma } _ { g } ^ { ( \ell ) } \biggr ] = \sigma \biggl ( H _ { \Gamma } ^ { ( \ell ) } \biggl ( \mathrm { C o n c a t } \Bigl ( \mathbf { U } _ { e } ^ { ( \ell ) } , \mathbf { U } _ { g } ^ { ( \ell ) } \Bigr ) \biggr ) \biggr ) , } \\ & { \qquad \mathbf { Z } ^ { ( \ell ) } = H _ { F } ^ { ( \ell ) } \biggl ( \mathrm { C o n c a t } \Bigl ( \mathbf { \Gamma } _ { e } ^ { ( \ell ) } \odot \mathbf { U } _ { e } ^ { ( \ell ) } , \mathbf { \Gamma } _ { g } ^ { ( \ell ) } \odot \mathbf { U } _ { g } ^ { ( \ell ) } \Bigr ) \biggr ) \biggr ) } \end{array}\tag{16}
$$

The learned masks condition each stream on the other before fusion, preserving their distinct information roles while modeling excitation–geometry interaction across multiple spatial scales. This contrasts with input-level concatenation, where the two information types are merged before their respective structures are encoded.

A feature pyramid network (FPN)-style decoder [32] yields

$$
{ \bf Z } _ { s , q } ^ { 0 } = D _ { X } \bigg ( \Big \{ { \bf Z } ^ { ( \ell ) } \Big \} _ { \ell = 1 } ^ { L } \bigg ) .\tag{17}
$$

![](images/e3d756634df45f874c143390e511fda5312c562819206889ea6964efa918937d.jpg)  
Fig. 2. Architecture of BeamRMX. (a) The query-only XBase learns the scene-conditioned transformation from target spatial excitation and scene geometry to the BeamRM. (b) The optional Evidence Adapter exploits same-scene cross-configuration BeamRMs to produce a gated residual refinement of the frozen XBase output. The adapter is bypassed for K = 0 and activated for $K > 0 ,$ , without test-time parameter updates.

The decoded representation is mapped to a dominant response and a scene-dependent modulation map:

$$
\begin{array} { r } { \mathbf { Y } _ { s , q } ^ { \mathrm { b a s e } } = H _ { \mathrm { b a s e } } \bigl ( \mathbf { Z } _ { s , q } ^ { \mathrm { 0 } } , \mathbf { E } _ { s , q } \bigr ) , } \\ { \mathbf { V } _ { s , q } = \sigma \bigl [ H _ { \mathrm { v i s } } \bigl ( \mathbf { Z } _ { s , q } ^ { \mathrm { 0 } } , \mathbf { G } _ { s } \bigr ) \bigr ] . } \end{array}\tag{18}
$$

Two residual heads further capture excitation–geometry and boundary-sensitive corrections:

$$
\begin{array} { r l } & { { \bf R } _ { s , q } ^ { \mathrm { d i r } } = H _ { \mathrm { d i r } } \left( { \bf Z } _ { s , q } ^ { 0 } , { \bf E } _ { s , q } , { \bf G } _ { s } \right) , } \\ & { { \bf R } _ { s , q } ^ { \mathrm { b d } } = H _ { \mathrm { b d } } \left( { \bf Z } _ { s , q } ^ { 0 } , { \bf G } _ { s } \right) . } \end{array}\tag{19}
$$

The final query-only prediction is then

$$
\begin{array} { r } { \widehat { \mathbf { Y } } _ { s , q } ^ { 0 } = \Pi _ { Y } \big ( \mathbf { V } _ { s , q } \odot \mathbf { Y } _ { s , q } ^ { \mathrm { b a s e } } + \mathbf { R } _ { s , q } ^ { \mathrm { d i r } } + \mathbf { R } _ { s , q } ^ { \mathrm { b d } } \big ) . } \end{array}\tag{20}
$$

Here, $\mathbf { V } _ { s , q }$ softly modulates the dominant beam-conditioned response according to scene structure. The directional and boundary residual heads capture local excitation–geometry deviations and boundary-sensitive details, respectively. These components provide propagation-related inductive biases rather than explicit decompositions of individual reflection, diffraction, or other propagation mechanisms.

## C. Evidence Adapter: Cross-Configuration Propagation Evidence

As shown in Fig. 2(b), the Evidence Adapter is activated only when same-scene source evidence is available. For $K > 0 ,$ , each support BeamRM is paired with its source excitation $\mathbf { E } _ { s , q _ { i } } =$ $\phi _ { \mathrm { e x c } } \left( \mathbf { B } _ { s , q _ { i } } ; \Omega _ { s } \right)$ . To expose how the realized source response differs from the corresponding beam map, we construct

$$
\mathbf { D } _ { i } ^ { \mathrm { b p } } = \mathbf { M } _ { s , q _ { i } } \odot \left( \mathbf { Y } _ { s , q _ { i } } - \mathbf { B } _ { s , q _ { i } } \right) .\tag{21}
$$

This response-to-prior discrepancy is used only as a learned evidence cue and is not interpreted as an isolated physical propagation component.

The support input and its encoded representation are

$$
\begin{array} { c } { \mathbf { X } _ { i } ^ { S } = \phi _ { S } \left( \mathbf { E } _ { s , q _ { i } } , \mathbf { Y } _ { s , q _ { i } } , \mathbf { M } _ { s , q _ { i } } , \mathbf { D } _ { i } ^ { \mathrm { b p } } , \mathbf { G } _ { s } \right) , } \\ { E _ { S } \left( \mathbf { X } _ { i } ^ { S } \right) = \left( \left\{ \mathbf { U } _ { i } ^ { S , ( \ell ) } \right\} _ { \ell = 1 } ^ { L _ { S } } , \mathbf { u } _ { i } ^ { S } \right) , } \end{array}\tag{22}
$$

where $\phi _ { S } ( \cdot )$ also incorporates detail-, gradient-, and boundarysensitive cues derived from the available source maps.

Different source configurations need not be equally informative. We therefore employ attention-based relation weighting [33]. Let $\mathbf { t } _ { s , q } ^ { 0 }$ be a target token pooled from $\mathbf { Z } _ { s , q } ^ { 0 } ,$ and let $\mathbf { r } ( q , q _ { i } )$ encode the relation between the target and source queries. Relation-aware aggregation is performed as

$$
\begin{array} { c } { \displaystyle \beta _ { i } = \psi \big ( \mathbf { t } _ { s , q } ^ { 0 } , \mathbf { u } _ { i } ^ { S } , \mathbf { r } ( q , q _ { i } ) \big ) , } \\ { \displaystyle \alpha _ { i } = \mathrm { s o f t m a x } _ { i } \big ( \{ \beta _ { j } \} _ { j = 1 } ^ { K } \big ) , } \\ { \displaystyle \mathbf { u } _ { S } ^ { K } = \sum _ { i = 1 } ^ { K } \alpha _ { i } \mathbf { u } _ { i } ^ { S } , } \\ { \displaystyle \mathbf { U } _ { S } ^ { ( \ell ) , K } = \sum _ { i = 1 } ^ { K } \alpha _ { i } \mathbf { U } _ { i } ^ { S , ( \ell ) } . } \end{array}\tag{23}
$$

The relation descriptor includes frequency, antennaconfiguration, and directional-state differences. This relation-aware weighting operates only on the already selected support items and is distinct from support selection itself. Candidate supports are first selected from the admissible strict-small pool using the query-aware utility criterion described in Section V-C, after which (23) performs learned aggregation over the selected set.

For compact notation, let

$$
\mathcal { U } _ { S } ^ { K } = \left\{ \mathbf { U } _ { S } ^ { \left( \ell \right) , K } \right\} _ { \ell = 1 } ^ { L _ { S } }
$$

denote the aggregated multiscale support features. The aggregated evidence is then fused with the frozen XBase representation as

$$
\begin{array} { r l } & { \mathbf { H } _ { s , q } ^ { K } = F _ { A } \Big ( \widehat { \mathbf { Y } } _ { s , q } ^ { 0 } , \mathbf { Z } _ { s , q } ^ { 0 } , \mathcal { U } _ { S } ^ { K } , \mathbf { u } _ { S } ^ { K } \Big ) , } \\ & { \mathbf { P } _ { s , q } ^ { K } = H _ { P } \Big ( \mathbf { H } _ { s , q } ^ { K } , \widehat { \mathbf { Y } } _ { s , q } ^ { 0 } \Big ) , } \\ & { \mathbf { C } _ { s , q } ^ { K } = \sigma \Big [ H _ { C } \Big ( \mathbf { H } _ { s , q } ^ { K } , \widehat { \mathbf { Y } } _ { s , q } ^ { 0 } \Big ) \Big ] , } \\ & { [ \mathbf { R } _ { \mathrm { l o w } } ^ { K } , \mathbf { R } _ { \mathrm { h f } } ^ { K } ] = H _ { R } \big ( \mathbf { H } _ { s , q } ^ { K } , \mathbf { P } _ { s , q } ^ { K } \big ) . } \end{array}\tag{24}
$$

Here, $\mathbf { P } _ { s , q } ^ { K }$ is a support-derived target estimate used internally by the adapter, while the two residual proposals capture broad calibration and localized detail corrections.

To prevent indiscriminate transfer across different excitations, the correction is controlled by shot, support-quality, region, and confidence factors:

$$
\begin{array} { r l } & { \quad \gamma _ { K } = g _ { K } ( K ) , } \\ & { \eta _ { \mathrm { q u a l } } ^ { K } = \sigma \big [ h _ { \mathrm { q u a l } } \bigl ( \mathbf { t } _ { s , q } ^ { 0 } , \mathbf { u } _ { S } ^ { K } \bigr ) \bigr ] , } \\ & { \mathbf { G } _ { \mathrm { r e g } } ^ { K } = \sigma \big [ H _ { \mathrm { r e g } } \bigl ( \mathbf { H } _ { s , q } ^ { K } \bigr ) \bigr ] , } \\ & { \mathbf { G } _ { s , q } ^ { K } = \gamma _ { K } \eta _ { \mathrm { q u a l } } ^ { K } \mathbf { G } _ { \mathrm { r e g } } ^ { K } \odot \frac { 1 + \mathbf { C } _ { s , q } ^ { K } } { 2 } . } \end{array}\tag{25}
$$

The final correction is

$$
\begin{array} { r l } & { \mathbf { \Delta } \mathbf { { \Delta } } \mathbf { { \Delta } } _ { s , q } ^ { K } = \mathbf { { G } } _ { s , q } ^ { K } \odot \left( \mathbf { R } _ { \mathrm { l o w } } ^ { K } + \mathbf { R } _ { \mathrm { h f } } ^ { K } \right) , } \\ & { \widehat { \mathbf { Y } } _ { s , q } ^ { K } = \Pi _ { Y } \left( \widehat { \mathbf { Y } } _ { s , q } ^ { 0 } + \Delta _ { s , q } ^ { K } \right) . } \end{array}\tag{26}
$$

The gating mechanism restricts cross-configuration correction when the available evidence is weak, spatially irrelevant, or inconsistent with the target query.

## D. Two-Stage Learning and Inference

BeamRMX is trained in two stages. XBase is first optimized independently as the query-only predictor:

$$
\mathcal { L } _ { X } = \mathcal { L } _ { \mathrm { v a l i d } } ^ { X } + \lambda _ { \mathrm { a u x } } \mathcal { L } _ { \mathrm { a u x } } ^ { X } + \lambda _ { \mathrm { s t r } } \mathcal { L } _ { \mathrm { s t r } } ^ { X } ,\tag{27}
$$

where the terms supervise valid-region reconstruction, auxiliary structured outputs, and region/detail-sensitive consistency. These objectives require no path-level physical labels.

After this stage, $\theta _ { X }$ is frozen and only the Evidence Adapter is trained. Episodic training samples a target query and a strictsmall support set with varying K, using

$$
\begin{array} { r l } & { \mathcal { L } _ { A } = \mathcal { L } _ { \mathrm { v a l i d } } ^ { A } + \lambda _ { \mathrm { d e t } } \mathcal { L } _ { \mathrm { d e t } } ^ { A } + \lambda _ { \mathrm { p r i o r } } \mathcal { L } _ { \mathrm { p r i o r } } + \lambda _ { \mathrm { g a i n } } \mathcal { L } _ { \mathrm { g a i n } } } \\ & { \qquad + \lambda _ { \mathrm { s p } } \mathcal { L } _ { \mathrm { s p } } + \lambda _ { \mathrm { n h } } \mathcal { L } _ { \mathrm { n h } } + \lambda _ { \mathrm { k d } } \mathcal { L } _ { \mathrm { k d } } . } \end{array}\tag{28}
$$

The first two terms supervise final reconstruction and spatial detail; $\mathcal { L } _ { \mathrm { { p r i o r } } }$ and $\mathcal { L } _ { \mathrm { g a i n } }$ train useful evidence-conditioned estimates and corrections; $\mathcal { L } _ { \mathrm { s p } }$ limits unnecessarily broad adaptation; and $\mathcal { L } _ { \mathrm { k d } }$ transfers more reliable richer-support behavior to low-shot settings.

A margin-based no-harm term explicitly discourages substantial degradation relative to the frozen anchor. Let

$$
\mathcal { V } _ { s , q } = \{ p \in \Omega _ { s } : \mathbf { M } _ { s , q } ( p ) = 1 \}
$$

denote the valid-pixel set, and define the pointwise error as $\varepsilon _ { s , q } ^ { K } ( p ) = | \widehat { \mathbf { Y } } _ { s , q } ^ { K } ( \bar { p } ) - { \mathbf { Y } } _ { s , q } ( p ) |$ , with $\varepsilon _ { s , q } ^ { 0 } ( p )$ defined analogously. The no-harm penalty is

$$
\mathcal { L } _ { \mathrm { n h } } = \frac { 1 } { \vert \mathcal { V } _ { s , q } \vert } \sum _ { p \in \mathcal { V } _ { s , q } } \left[ \varepsilon _ { s , q } ^ { K } ( p ) - \varepsilon _ { s , q } ^ { 0 } ( p ) - \epsilon _ { \mathrm { n h } } \right] _ { + } .\tag{29}
$$

Here, $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ . The term penalizes harmful corrections beyond the tolerance $\epsilon _ { \mathrm { n h } }$ without requiring every corrected pixel to improve over XBase.

During inference, both $\theta _ { X }$ and $\theta _ { A }$ remain fixed. For $K = 0 ,$ the support branch is bypassed and the query-only prediction is returned exactly. For $K > 0$ , XBase is evaluated once and the Evidence Adapter produces the gated correction in a single forward pass. No gradient update, target-map optimization, or test-time fine-tuning is performed.

## V. EXPERIMENTAL RESULTS AND DISCUSSION

This section evaluates BeamRMX from three complementary perspectives. First, we examine query-only prediction under matched-domain, scene zero-shot (Scene-ZS), and configuration zero-shot (Config-ZS) settings. Second, we evaluate whether a few same-scene BeamRMs from lower-complexity configurations can improve prediction of an unobserved target BeamRM without target observations or test-time parameter updates. Third, we analyze which evidence mechanisms are responsible for the source-assisted gain and whether the resulting BeamRM improvements translate into more reliable intra-sector beam refinement.

## A. Experimental Setup

Experiments are conducted on the multi-configuration XL MIMO radiomap dataset introduced in [13]. The dataset contains 78,400 ray-tracing-based BeamRMs from 800 urban scenes, covering five carrier frequencies from 1.8 to 6.7 GHz, multiple array geometries and sizes, and directional arrays with up to 1024 antenna elements. The combinations of carrier frequency, array configuration, and beam steering state yield 98 concrete frequency–array–beam query conditions per scene. Each sample consists of a scene representation, a configurationdependent beam map, and the corresponding BeamRM.

All scene representations, beam maps, and target BeamRMs are processed at a common spatial resolution of 128 × 128. Valid free-space received powers within (−300, 0) dB are linearly mapped to [0, 1], while building and invalid regions are excluded using the corresponding validity masks. Mean absolute error (MAE) and root mean square error (RMSE) are computed over valid free-space pixels after conversion back to the dB domain, whereas peak signal-to-noise ratio (PSNR) and structural similarity index (SSIM) are evaluated on the normalized BeamRMs with unit data range and the same valid-region treatment. Unless otherwise stated, all compared methods use identical data splits, spatial resolution, normalization, validity masks, and metric implementations.

All experiments are implemented in PyTorch and conducted on NVIDIA A40 graphics processing units (GPUs). XBase employs four-stage ConvNeXt V2 encoders and an FPN-style decoder and is optimized using AdamW with mixed-precision training for 50 epochs, with checkpoints selected according to validation MAE. The Evidence Adapter is subsequently trained for 16 epochs while XBase remains frozen, using a learning rate of $5 \times 1 0 ^ { - 5 }$ and a weight decay of $2 \times 1 0 ^ { - 2 }$ . During episodic adapter training, $K \in \{ 0 , 1 , 2 , 4 , 8 \}$ is sampled with probabilities $\{ 0 . 0 5 , 0 . 4 5 , 0 . 3 5 , 0 . 1 0 , 0 . 0 5 \}$ , emphasizing the practically relevant low-shot regime. No model performs targetside parameter updates during inference. Complete training configurations and evaluation scripts are provided in the public code repository.

TABLE I  
QUERY-ONLY BEAMRM PREDICTION UNDER THE MATCHED-DOMAIN, SCENE-ZS, AND CONFIG-ZS PROTOCOLS.
<table><tr><td>Protocol</td><td>Metric</td><td>RadioWNet</td><td>BeamCKM</td><td>RadioDiff</td><td>BeamCKMDiff</td><td>BeamRMX</td><td>Gain over Best Baseline</td></tr><tr><td rowspan="4">Matched- domain</td><td>MAE (dB) ↓</td><td>6.1612</td><td>4.8713</td><td>5.0010</td><td>7.1315</td><td>2.0991</td><td>↓56.91%</td></tr><tr><td>RMSE (dB) ↓</td><td>9.5377</td><td>6.9161</td><td>7.0782</td><td>10.4604</td><td>4.1597</td><td>↓39.85%</td></tr><tr><td>PSNR (dB) ↑</td><td>30.9427</td><td>33.0515</td><td>32.7574</td><td>29.7024</td><td>37.1613</td><td>↑4.1098 dB</td></tr><tr><td>SSIM ↑</td><td>0.7015</td><td>0.8216</td><td>0.8120</td><td>0.7468</td><td>0.9073</td><td>↑0.0857</td></tr><tr><td rowspan="4">Scene-ZS</td><td>MAE (dB) ↓</td><td>6.9671</td><td>6.7375</td><td>6.7331</td><td>7.8319</td><td>4.9749</td><td>↓26.11%</td></tr><tr><td>RMSE (dB) ↓</td><td>10.4689</td><td>9.7920</td><td>9.9510</td><td>11.7170</td><td>9.0415</td><td>↓7.66%</td></tr><tr><td>PSNR (dB) ↑</td><td>29.9977</td><td>30.2605</td><td>30.2803</td><td>28.8026</td><td>31.5072</td><td>↑1.2269 dB</td></tr><tr><td>SSIM ↑</td><td>0.6622</td><td>0.7592</td><td>0.7679</td><td>0.7200</td><td>0.8106</td><td>↑0.0427</td></tr><tr><td rowspan="4">Config-ZS</td><td>MAE (dB) ↓</td><td>6.1466</td><td>5.4694</td><td>5.3107</td><td>8.6834</td><td>2.7748</td><td>↓47.75%</td></tr><tr><td>RMSE (dB)↓</td><td>8.8138</td><td>7.5310</td><td>7.3779</td><td>11.9391</td><td>5.0562</td><td>↓31.47%</td></tr><tr><td>PSNR (dB) ↑</td><td>31.3766</td><td>32.2486</td><td>32.4379</td><td>28.4284</td><td>35.4660</td><td>↑3.0281 dB</td></tr><tr><td>SSIM ↑</td><td>0.6976</td><td>0.8006</td><td>0.8245</td><td>0.7179</td><td>0.8695</td><td>↑0.0450</td></tr></table>

Model scale and inference efficiency: XBase contains 89.3M parameters, comparable in scale to the 105.6M–113.0M adapted BeamCKM-based predictors used in our experiments. The optional Evidence Adapter introduces 42.3M trainable parameters, giving 131.6M parameters for the complete sourceassisted model while leaving XBase frozen. Under our NVIDIA A40 evaluation implementation, query-only XBase requires approximately 50.2 ms per query, and source-assisted inference requires approximately 60–62 ms for $K \in \{ 1 , 2 , 4 \}$ . Although the evaluated diffusion-based predictors contain fewer parameters, their iterative sampling pipelines require approximately 115–121 ms per query. Thus, the proposed framework retains single-pass inference and introduces only a modest latency increase when cross-configuration evidence is available.

## B. Query-Only Multi-Configuration Prediction

We first evaluate the query-only prediction capability of BeamRMX before introducing any same-scene radio evidence. Each method is provided with the scene representation, transmitter information, and the spatial beam map associated with the target query, but has no access to same-scene source BeamRMs, sparse target measurements, partial target maps, or target labels. This setting isolates the ability of each model to learn the sceneconditioned transformation from a target spatial-excitation prior to the corresponding received power field.

The comparison includes RadioWNet + Beam Map, Beam CKM, RadioDiff, and BeamCKMDiff. RadioWNet follows the beam-conditioned W-Net benchmark established in the multi-configuration dataset study [13]. The adapted BeamCKM baseline uses a deterministic encoder–decoder predictor with beam-conditioned inputs [15], whereas RadioDiff represents a diffusion-based radio map predictor [12]. BeamCKMDiff combines continuous beam-aware conditioning with a diffusiontransformer architecture [16], [34]. For a representationcontrolled comparison, all methods are provided with the same target beam map as the spatial beam representation. BeamCKM and BeamCKMDiff are therefore adapted from their original beam descriptors to the same beam map input while retaining their principal prediction architectures. This setting isolates the learning architecture from differences in beam representation.

Three complementary query-only protocols are considered:

• Matched-domain: The 78,400 scene–query samples are randomly divided into 54,880 training, 7,840 validation, and 15,680 test samples. Scene and radio-query distributions may overlap across the subsets, while individual samples remain disjoint. This setting serves as a seendomain reference.

• Scene-ZS: The 800 scenes are divided into 560 training, 80 validation, and 160 test scenes, with all 98 frequency– array–beam query conditions retained for each scene. The resulting 15,680 test samples are therefore generated exclusively from propagation scenes absent from model training.

• Config-ZS: Complete radio configurations are separated across training, validation, and testing while all 800 scenes are retained. A 6.7-GHz, 64-element, 8-beam configuration is used exclusively for validation, whereas a distinct 6.7- GHz, 256-element, 16-beam configuration is reserved for testing. All remaining configurations are used for training, resulting in 59,200 training, 6,400 validation, and 12,800 test samples.

Scene-ZS and Config-ZS probe complementary generalization dimensions and are therefore not intended as directly comparable measures of difficulty. The former withholds propagation scenes, whereas the latter withholds an entire radio configuration. In particular, neither the Config-ZS test configuration nor any BeamRM associated with its directional states is used for model training or checkpoint selection.

Table I first shows that BeamRMX is already a strong queryonly predictor in the matched-domain setting. It achieves an

![](images/936d2652c67d4fbf6f1dd9522eb098f2bd434a1aca30e1c3395b7e9ff39cf6c9.jpg)  
Fig. 3. Qualitative BeamRM prediction under matched-domain, Scene-ZS, and Config-ZS evaluation protocols. The figure shows one representative matched domain example, two Scene-ZS examples, and two Config-ZS examples. All methods use the same received-power scale within each row.

MAE of 2.0991 dB, reducing the best competing MAE by 56.91%, while the RMSE is reduced by 39.85%. The consistent advantage in PSNR and SSIM further shows that the gain is not limited to an overall power offset. Since no distribution shift is introduced in this protocol, these results establish the effectiveness of XBase as a dedicated predictor for spatialexcitation-conditioned BeamRM learning.

The Scene-ZS results test whether this learned excitation-tofield relationship transfers to completely unseen propagation environments. BeamRMX achieves an MAE of 4.9749 dB, corresponding to a 26.11% reduction relative to the best competing result. It also obtains the best RMSE, PSNR, and SSIM among all evaluated methods. Notably, the relative RMSE reduction is more moderate at 7.66% than the MAE improvement. This suggests that, although BeamRMX substantially improves the overall spatial prediction accuracy, difficult high-error regions remain challenging when the propagation environment itself is unseen. The result therefore supports the value of the proposed excitation–geometry interaction while also revealing residual scene-dependent uncertainty that cannot be fully resolved from geometry and the target excitation alone.

Config-ZS evaluates a different form of extrapolation: the complete 6.7-GHz, 256-element, 16-beam test configuration is absent from both model training and checkpoint selection, although the underlying scenes remain represented through other configurations. Under this configuration-disjoint protocol, BeamRMX attains an MAE of 2.7748 dB and an RMSE of 5.0562 dB, reducing the corresponding errors of the best baseline by 47.75% and 31.47%, respectively. It also improves PSNR from 32.4379 to 35.4660 dB and SSIM from 0.8245 to 0.8695, corresponding to gains of 3.0281 dB and 0.0450, respectively. These consistent improvements in both powerdomain errors and structural metrics indicate that the Config-ZS advantage is not limited to overall received-power calibration, but also extends to preserving the beam-dependent spatial response. More importantly, the results show that the spatialexcitation formulation can transfer to a radio configuration that is entirely absent during model development, rather than relying on configuration-specific associations learned from the training samples.

Training-seed robustness: To assess sensitivity to training stochasticity, we independently train BeamRMX with three random seeds while keeping the Scene-ZS and Config-ZS data partitions and all other experimental settings fixed. The resulting Scene-ZS MAE is $\mathrm { 4 . 9 3 5 9 \pm 0 . 0 3 4 4 ~ d B }$ , where the variation denotes the sample standard deviation across independent training runs. Under Config-ZS, the corresponding MAE is $2 . 8 5 6 7 \pm 0 . 0 7 8 7$ dB. In both regimes, the run-to-run variation is substantially smaller than the performance margin over the strongest competing method in Table I, indicating that the observed zero-shot gains are insensitive to training initialization and stochastic optimization.

Representative Scene-ZS and Config-ZS predictions are compared in Fig. 3. The examples complement the aggregate results by directly contrasting the reconstructed BeamRMs under identical target queries. All methods recover the dominant beam-conditioned spatial response to some extent, whereas BeamRMX more closely follows the ground-truth distribution in localized high- and low-response regions. The visual comparison is therefore consistent with the quantitative gains in Table I, while the common received power scale prevents independent normalization from exaggerating apparent differences.

Across the three protocols, baseline performance shifts noticeably with the evaluation regime. BeamCKM is the strongest baseline in the matched-domain setting, whereas RadioDiff proves more resilient under distribution shift, taking second place across all four Config-ZS metrics. This contrast highlights a key trade-off: while deterministic designs like BeamCKM excel when training and test conditions align, diffusion-based approaches retain stronger generalization. Adapting BeamCKMDiff from vector conditioning to a spatial beam-map representation reduces its effectiveness, reflecting its architectural difficulty in handling heterogeneous multi-scene setup. Across all settings, BeamRMX consistently achieves top scores on every metric, confirming its adaptability to both unseen scenes and unseen radio configurations. Nevertheless, the performance gap between matched-domain and Scene-ZS settings points to residual scene uncertainty that geometry alone cannot resolve—providing a clear rationale for incorporating same-scene cross-configuration evidence.

## C. Strict-Small Source-Assisted Prediction in Unseen Scenes

We next investigate whether limited same-scene radio evidence can reduce the residual uncertainty of query-only prediction in unseen propagation environments. The experiment uses the 160 Scene-ZS test scenes and focuses on the 6.7-GHz, 1024-element, 64-beam target configuration, yielding 10,240 target queries. Support BeamRMs are drawn from the same scene but from different radio configurations containing no more than 64 antenna elements. The target configuration is excluded from the support pool, and no sparse observation or partial map of the target BeamRM is available. Under this strict-small protocol, we evaluate $K \in \{ 0 , 1 , 2 , 4 \}$ supports without any inference-time parameter update. Thus, the target uses a 1024-element array, whereas every admissible support configuration contains at most 64 antenna elements, with the target configuration excluded by construction.

For each target query, admissible supports are first restricted by the strict-small constraint and then ranked using a queryaware utility criterion. The utility combines source–target beam and configuration relations with source-side validity and detail cues, consistency with the frozen query-only anchor, and a redundancy penalty among selected supports. The top-K candidates form the support set used by the Evidence Adapter. The selector uses only the target query, the frozen query-only prediction, and available source-side information; no target BeamRM value is used.

For each method, $K = 0$ denotes its query-only anchor evaluated on the same target configuration. In BeamRMX, the Evidence Adapter is bypassed exactly at $K = 0$ , so the output is identical to that of the frozen XBase. The resulting $K = 0$ performance differs from the Scene-ZS result in Table I, which averages over all radio-query conditions, whereas the present experiment evaluates only the 6.7-GHz, 1024-element, 64-beam target configuration.

To determine whether the benefit of same-scene evidence is specific to BeamRMX, we construct source-assisted variants of two complementary baselines. BeamCKM is selected as the closest beam-aware reference because its M3ChanNet extension already incorporates environmental profiles and multi-beam observations as auxiliary information [15], while RadioDiff represents a strong generative RM predictor and one of the most competitive zero-shot baselines in Section V-B. Both are adapted to the same strict-small information boundary and receive exactly the same query-aware support episodes as BeamRMX. BeamCKM-Support incorporates the source BeamRMs into its deterministic pipeline, whereas RadioDiff-Support augments its diffusion condition with aggregated source observations. All methods use identical target queries, support cardinalities, support episodes, and metrics.

Table II shows that even a very small amount of same-scene cross-configuration evidence provides substantial information beyond the query-only prediction. For BeamRMX, a single support reduces MAE from 5.9665 to 3.9178 dB, corresponding to a 34.3% reduction. RMSE, PSNR, and SSIM improve simultaneously, indicating that the benefit is not limited to a global power adjustment but extends to the reconstructed spatial response. Since the target BeamRM itself remains completely unobserved, these gains show that source configurations contain scene-specific information that is transferable across different spatial excitations.

The benefit also exhibits a clear small-shot saturation trend. Increasing the support size from K = 1 to K = 2 further reduces the BeamRMX MAE from 3.9178 to 3.7237 dB, whereas increasing K to 4 yields only a marginal additional improvement to 3.7095 dB. The other reconstruction metrics follow the same trend. Hence, most of the useful same-scene information is recovered from the first one or two informative supports, while additional configurations provide diminishing returns. This property is particularly relevant to incremental network deployment, where a small amount of legacy or lowercomplexity radio information may already be available but exhaustive characterization of the new large-array configuration is undesirable.

BeamRMX also exploits the available evidence more effectively than the two support-enhanced baselines. At $K = 4 ,$ its MAE is 22.1% lower than that of BeamCKM-Support and 28.5% lower than that of RadioDiff-Support. More importantly, the methods respond differently as additional evidence becomes available. BeamCKM-Support essentially saturates after the first support, while RadioDiff-Support improves from $K = 1$ to $K = 2$ but gains no further benefit at $K = 4$ . BeamRMX instead retains consistent, although diminishing, improvement as support increases. This suggests that the value of source evidence depends not only on its availability, but also on whether configuration-specific response patterns can be separated from scene information that remains useful for the target query.

TABLE II  
SOURCE-ASSISTED BEAMRM PREDICTION UNDER THE STRICT-SMALL PROTOCOL ON 160 UNSEEN SCENES.
<table><tr><td>Metric</td><td colspan="4">BeamCKM-Support</td><td colspan="4">RadioDiff-Support</td><td colspan="4">BeamRMX</td></tr><tr><td></td><td> $K = 0$ </td><td> $K = 1$ </td><td> $K = 2$ </td><td> $K = 4$ </td><td> $K = 0$ </td><td> $K = 1$ </td><td> $K = 2$ </td><td> $K = 4$ </td><td> $K = 0$ </td><td> $K = 1$ </td><td> $K = 2$ </td><td> $K = 4$ </td></tr><tr><td>MAE (dB) ↓</td><td>6.7336</td><td>4.7637</td><td> $4 . 7 6 3 5 $ </td><td> $4 . 7 6 3 5$ </td><td>6.4681</td><td>5.2391</td><td>5.1547</td><td> $5 . 1 8 5 0 $ </td><td> $\mathbf { 5 . 9 6 6 5 }$ </td><td> $3 . 9 1 7 8$ </td><td>3.7237</td><td>3.7095</td></tr><tr><td>RMSE (dB) ↓</td><td>9.9306</td><td>7.0330</td><td>7.0328</td><td>7.0329</td><td>9.8527</td><td>7.7761</td><td>7.6632</td><td>7.7083</td><td>9.5182</td><td>6.7461</td><td>6.4150</td><td>6.3883</td></tr><tr><td>PSNR (dB) ↑</td><td>30.1792</td><td>33.1113</td><td>33.1116</td><td>33.1116</td><td>30.4214</td><td>32.4092</td><td>32.5434</td><td>32.4960</td><td>30.7559</td><td>33.8350</td><td>34.2372</td><td>34.2732</td></tr><tr><td>SSIM ↑</td><td>0.7601</td><td>0.8354</td><td>0.8354</td><td>0.8354</td><td>0.7764</td><td>0.8209</td><td>0.8240</td><td>0.8232</td><td>0.7776</td><td>0.8818</td><td>0.8915</td><td>0.8924</td></tr></table>

These results support the central design principle of the Evidence Adapter. Same-scene source BeamRMs should not be treated as direct approximations of the target field; rather, they provide configuration-conditioned observations of a shared propagation environment. Relation-aware evidence aggregation and gated residual refinement allow BeamRMX to exploit this information while preserving the frozen query-only prediction as a stable anchor. The following subsection examines which components are responsible for this gain and whether the resulting BeamRM improvements translate into more reliable beam management decisions.

## D. Evidence Mechanisms and Intra-Sector Beam Refinement Utility

We evaluate the Evidence Adapter using controlled mechanism diagnostics and then examine whether the reconstruction gains improve intra-sector beam refinement.

Evidence mechanism analysis: The 40-scene subset is sampled uniformly at random once from the 160 unseen test scenes and fixed for all mechanism diagnostics, without reference to model outputs or scene-level performance. All $K = 2$ variants use the same trained BeamRMX checkpoint and the same strict-small support pool. The diagnostics respectively remove the source BeamRM content, disable the spatial region gate, or replace the query-aware utility selection with uniformly random admissible supports. High-frequency (HF) MAE is a detail-sensitive dB-domain reconstruction metric evaluated over valid free-space regions using the high-frequency criterion implemented in our evaluation code, while the noharm violation rate denotes the fraction of valid pixels for which source-assisted prediction degrades the frozen XBase anchor beyond the prescribed tolerance. These fixed checkpoint experiments are intended to probe mechanism trends on a controlled unseen-scene subset rather than to provide a full-test ranking of independently retrained model variants.

TABLE III  
FIXED CHECKPOINT MECHANISM DIAGNOSTICS ON 40 UNSEEN SCENES.
<table><tr><td>Variant</td><td>MAE (dB) ↓</td><td>High-Freq. MAE (dB) ↓</td><td>No-Harm Violation (%) ↓</td></tr><tr><td>XBase,  $\overline { { K = 0 } }$ </td><td>5.5354</td><td>13.4462</td><td></td></tr><tr><td>Full BeamRMX,  $K = 2$ </td><td>3.4252</td><td>4.7323</td><td>10.18</td></tr><tr><td>No Source BeamRM</td><td>5.5363</td><td>15.5742</td><td>19.29</td></tr><tr><td>No Region Gate</td><td>6.9513</td><td>7.1210</td><td>52.56</td></tr><tr><td>Random Support</td><td>3.6979</td><td>5.8082</td><td>11.13</td></tr></table>

Table III reveals three complementary aspects of the evidence mechanism. Removing the source BeamRM content nearly eliminates the source-assisted gain: the MAE increases from 3.4252 to 5.5363 dB, essentially returning to the $K \ = \ 0$ XBase level, while HF MAE deteriorates even further. The improvement therefore originates primarily from realized samescene radio evidence rather than from the additional adapter structure or source-configuration metadata alone.

Useful evidence must also be transferred conservatively. Removing the spatial region gate increases MAE to 6.9513 dB, which is worse than the query-only XBase anchor, and raises the no-harm violation rate to 52.56%. Thus, access to informative source BeamRMs alone does not guarantee a better target prediction: uncontrolled residual injection can introduce substantial negative transfer. This result directly supports the gated-residual design in Section IV-C, which restricts where cross-configuration evidence is allowed to modify the frozen prediction.

Random support selection causes a smaller but consistent degradation. Random admissible supports still improve substantially over XBase, confirming that same-scene radio responses themselves constitute the dominant source of useful information. Nevertheless, the full method improves MAE from 3.6979 to 3.4252 dB and also achieves better HF accuracy and a lower no-harm violation rate. The evidence therefore indicates that support relevance affects how effectively samescene information can be transferred, while the larger gains arise from the presence of realized source BeamRMs and the spatial control of their correction.

Intra-sector beam refinement utility: We further evaluate whether these improvements in BeamRM reconstruction provide useful information for downstream beam management. Because the 64 target beams form a contiguous angular-sector codebook rather than a full-azimuth codebook, the task is formulated as intra-sector beam refinement after a candidate sector has been identified. Let B denote the target beam set. For scene s, the reference serving-power surface and the beam selected from the predicted BeamRMs are

![](images/1bd1396e283fe5578466d0fe67d6857bd8aac10434a275236e0d29f6bf3ead55.jpg)

![](images/37aadb06c8f934e7f41a595dfd434500408ef2e8d7ef592836212c1d67591857.jpg)  
Fig. 4. Scene-level empirical cumulative distribution functions (CDFs) of serving power MAE and mean beam-selection regret over 160 unseen scenes.

TABLE IV  
INTRA-SECTOR BEAM REFINEMENT UTILITY ON 160 UNSEEN SCENES.
<table><tr><td rowspan="2">Metric</td><td colspan="2">BeamCKM-Support</td><td colspan="2">RadioDiff-Support</td><td colspan="2">BeamRMX</td></tr><tr><td> $\overline { { K = 0 } }$ </td><td> $\overline { { K = 2 } }$ </td><td> $\overline { { K = 0 } }$ </td><td> $\overline { { K = 2 } }$ </td><td> $\overline { { K = 0 } }$ </td><td> $\overline { { K = 2 } }$ </td></tr><tr><td>Serving MAE (dB) ↓</td><td>6.735</td><td>5.525</td><td>6.458</td><td>4.053</td><td>6.071</td><td>3.612</td></tr><tr><td>Top-3 Hit (%) ↑</td><td>67.48</td><td>68.85</td><td>67.17</td><td>67.64</td><td>77.87</td><td>79.96</td></tr><tr><td>Mean Regret (dB) ↓</td><td>4.148</td><td>3.671</td><td>4.367</td><td>3.528</td><td>3.439</td><td>2.873</td></tr><tr><td>P95 Regret (dB) ↓</td><td>16.986</td><td>15.657</td><td>17.739</td><td>14.593</td><td>15.744</td><td>13.502</td></tr><tr><td>Footprint IoU ↑</td><td>0.549</td><td>0.582</td><td>0.593</td><td>0.672</td><td>0.614</td><td>0.741</td></tr></table>

$$
\begin{array} { l } { { \displaystyle P _ { s } ^ { \mathrm { s e r v } } ( p ) = \operatorname* { m a x } _ { b \in \mathcal { B } } \mathbf { Y } _ { s , b } ( p ) } , } \\ { { \displaystyle \widehat { b } _ { s } ( p ) = \arg \operatorname* { m a x } _ { b \in \mathcal { B } } \widehat { \mathbf { Y } } _ { s , b } ( p ) } . } \end{array}\tag{30}
$$

The resulting beam-selection regret is defined as

$$
R _ { s } ( p ) = P _ { s } ^ { \mathrm { s e r v } } ( p ) - \mathbf { Y } _ { s , \widehat { b } _ { s } ( p ) } ( p ) ,\tag{31}
$$

which measures the received-power loss caused by selecting the beam favored by the predicted BeamRMs instead of the true best beam at location p. A regret of zero therefore indicates that the predicted BeamRMs select a ground-truth optimal beam. A Top-3 hit occurs when $\widehat { b } _ { s } ( p )$ belongs to the three beams with the highest ground-truth received powers at location p.

Serving-power MAE is computed over the spatial intersection in which all target beams are valid. The Top-3 hit rate and beam-selection regret are evaluated at locations whose reference serving power is no lower than −110 dBm; this threshold defines a reference-covered evaluation region rather than an operational coverage guarantee. To additionally assess the spatial localization of strong beam responses, the footprint intersection over union (IoU) is computed from the top 5% power region of each target BeamRM and averaged over beams and scenes.

The source-assisted improvement remains evident at the beam management level. Relative to its query-only anchor, BeamRMX with $K = 2$ reduces serving-power MAE by 40.5% and mean beam-selection regret by 16.5%, while also improving the Top-3 hit rate, the 95th-percentile (P95) beam-selection regret, and high response footprint overlap. It achieves the best result across all communication-oriented metrics in Table IV.

These gains show that the additional radio evidence improves not only pointwise BeamRM reconstruction but also the relative ordering and spatial localization of candidate-beam responses, which are directly relevant to beam refinement.

The scene-level empirical distributions in Fig. 4 further show that the improvement is not driven by only a small number of favorable environments. BeamRMX with $K = 2$ shifts both the serving-power MAE and mean-regret distributions toward lower values relative to its $K = 0$ anchor. A paired scenelevel bootstrap analysis gives a mean serving-power MAE improvement of 2.460 dB, with a 95% confidence interval of [2.224, 2.715] dB, and improvement is observed in all 160 test scenes. Together with the mechanism diagnostics, these results show that controlled use of same-scene crossconfiguration evidence provides consistent gains from BeamRM reconstruction to downstream intra-sector beam refinement utility.

## VI. CONCLUSION

This paper investigated generalizable BeamRM prediction in beamformed wireless systems, where different beam and radio configurations produce distinct received power fields within the same propagation scene. BeamRMX addresses this one-scene–many-BeamRM setting by representing the target radiation state explicitly in the spatial domain and formulating BeamRM prediction as learning how the scene transforms this spatial radiation query into the corresponding received power field. This common spatial representation supports prediction across heterogeneous beam states and configurations, while XBase explicitly learns the interaction between radiation and scene geometry. Experiments under matched-domain, unseen scene, and unseen-configuration settings consistently validate the effectiveness and generalization of this formulation. When same-scene radio evidence is available, the optional Evidence Adapter provides further refinement without target observations or test-time updates. The resulting BeamRMs also improve serving-power estimation and intra-sector beam refinement. Future work will investigate measurement-based validation and broader beam and array configurations.

[1] Z. Wang, J. Zhang, H. Du, D. Niyato, S. Cui, B. Ai, M. Debbah, K. B. Letaief, and H. V. Poor, “A tutorial on extremely large-scale MIMO for 6G: Fundamentals, signal processing, and applications,” IEEE Communications Surveys & Tutorials, vol. 26, no. 3, pp. 1560–1605, 2024.

[2] S. Dang, O. Amin, B. Shihada, and M.-S. Alouini, “What should 6G be?” Nature Electronics, vol. 3, no. 1, pp. 20–29, 2020.

[3] N. Cheng, F. Chen, W. Chen, Z. Cheng, Q. Yang, C. Li, and X. Shen, “6G omni-scenario on-demand services provisioning: vision, technology and prospect (in Chinese),” Sci Sin Inform, vol. 54, pp. 1025–1054, 2024.

[4] J. Wen, J. Nie, Y. Zhong, C. Yi, X. Li, J. Jin, Y. Zhang, and D. Niyato, “Diffusion-model-based incentive mechanism with prospect theory for edge AIGC services in 6G IoT,” IEEE Internet of Things Journal, vol. 11, no. 21, pp. 34 187–34 201, 2024.

[5] N. Cheng, F. Lyu, W. Quan, C. Zhou, H. He, W. Shi, and X. Shen, “Space/aerial-assisted computing offloading for IoT applications: A learning-based approach,” IEEE Journal on Selected Areas in Communications, vol. 37, no. 5, pp. 1117–1129, 2019.

[6] R. W. Heath, N. Gonzalez-Prelcic, S. Rangan, W. Roh, and A. M. Sayeed,´ “An overview of signal processing techniques for millimeter wave MIMO systems,” IEEE Journal of Selected Topics in Signal Processing, vol. 10, no. 3, pp. 436–453, 2016.

[7] M. Giordani, M. Polese, A. Roy, D. Castor, and M. Zorzi, “A tutorial on beam management for 3GPP NR at mmWave frequencies,” IEEE Communications Surveys & Tutorials, vol. 21, no. 1, pp. 173–196, 2019.

[8] A. Alkhateeb, “DeepMIMO: A generic deep learning dataset for millimeter wave and massive MIMO applications,” 2019, arXiv:1902.06435.

[9] Y. Zeng, J. Chen, J. Xu, D. Wu, X. Xu, S. Jin, X. Gao, D. Gesbert, S. Cui, and R. Zhang, “A tutorial on environment-aware communications via channel knowledge map for 6G,” IEEE Communications Surveys & Tutorials, vol. 26, no. 3, pp. 1478–1519, 2024.

[10] J. Hoydis, S. Cammerer, F. Ait Aoudia, A. Vem, N. Binder, G. Marcus, and A. Keller, “Sionna: An open-source library for next-generation physical layer research,” 2022, arXiv:2203.11854.

[11] R. Levie, C. Yapar, G. Kutyniok, and G. Caire, “RadioUNet: Fast radio map estimation with convolutional neural networks,” IEEE Transactions on Wireless Communications, vol. 20, no. 6, pp. 4001–4015, 2021.

[12] X. Wang, K. Tao, N. Cheng, Z. Yin, Z. Li, Y. Zhang, and X. Shen, “RadioDiff: An effective generative diffusion model for sampling-free dynamic radio map construction,” IEEE Transactions on Cognitive Communications and Networking, vol. 11, no. 2, pp. 738–750, 2025.

[13] X. Li, Y. Han, Z. Lu, S. Jin, and C.-K. Wen, “U6G XL-MIMO radiomap prediction: Multi-config dataset and beam map approach,” 2026, arXiv:2603.06401.

[14] X. Wang, Q. Zhang, N. Cheng, J. Chen, Z. Zhang, Z. Li, S. Cui, and X. Shen, “RadioDiff-3D: A 3D× 3D radio map dataset and generative diffusion based benchmark for 6G environment-aware communication,” IEEE Transactions on Network Science and Engineering, vol. 13, pp. 3773–3789, 2026.

[15] H. Wang, X. Shi, H. Zhang, Y. Cao, S. Yang, J. Wang, and K. Huang, “BeamCKM: A framework of channel knowledge map construction for multi-antenna systems,” 2025, arXiv:2511.18376.

[16] L. Zhao, Y. Wang, X. Wang, Z. Fei, and Y. Zeng, “BeamCKMDiff: Beamaware channel knowledge map construction via diffusion transformer,” in Proc. IEEE INFOCOM, 2026, pp. 1–6.

[17] S. Zhang, A. Wijesinghe, and Z. Ding, “RME-GAN: A learning framework for radio map estimation based on conditional generative adversarial network,” IEEE Internet of Things Journal, vol. 10, no. 20, pp. 18 016–18 027, 2023.

[18] X. Li, S. Zhang, H. Li, X. Li, L. Xu, H. Xu, H. Mei, G. Zhu, N. Qi, and M. Xiao, “RadioGAT: A joint model-based and data-driven framework for multi-band radiomap reconstruction via graph attention networks,” IEEE Transactions on Wireless Communications, vol. 23, no. 11, pp. 17 777–17 792, 2024.

[19] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 6840–6851.

[20] X. Wang, Q. Zhang, N. Cheng, R. Sun, Z. Li, S. Cui, and X. Shen, “RadioDiff-k2: Helmholtz equation informed generative diffusion model for multi-path aware radio map construction,” IEEE Journal on Selected Areas in Communications, vol. 44, pp. 2318–2333, 2026.

[21] W. Liu and J. Chen, “UAV-aided radio map construction exploiting environment semantics,” IEEE Transactions on Wireless Communications, vol. 22, no. 9, pp. 6341–6355, 2023.

[22] C. Yapar, F. Jaensch, R. Levie, G. Kutyniok, and G. Caire, “The first pathloss radio map prediction challenge,” in ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023, pp. 1–2.

[23] Y. Zeng and X. Xu, “Toward environment-aware 6G communications via channel knowledge map,” IEEE Wireless Communications, vol. 28, no. 3, pp. 84–91, 2021.

[24] F. Jaensch, G. Caire, and B. Demir, “Radio map prediction from aerial images and application to coverage optimization,” IEEE Transactions on Wireless Communications, vol. 25, pp. 308–320, 2026.

[25] Y. Teganya and D. Romero, “Deep completion autoencoders for radio map estimation,” IEEE Transactions on Wireless Communications, vol. 21, no. 3, pp. 1710–1724, 2022.

[26] Y. Qiu, X. Chen, K. Mao, X. Ye, H. Li, F. Ali, Y. Huang, and Q. Zhu, “Channel knowledge map construction based on a UAV-assisted channel measurement system,” Drones, vol. 8, no. 5, p. 191, 2024.

[27] H. Sun and J. Chen, “Energy-modified leverage sampling for radio map construction via matrix completion,” IEEE Signal Processing Letters, vol. 31, pp. 1780–1784, 2024.

[28] X. Wang, Z. Fang, N. Cheng, R. Sun, H. Zhou, Z. Su, Z. Li, and X. Shen, “RadioDiff-Inverse: Diffusion enhanced Bayesian inverse estimation for ISAC radio map construction,” IEEE Transactions on Wireless Communications, vol. 25, pp. 14 611–14 626, 2026.

[29] X. Wang, Z. Guo, N. Cheng, Z. Yin, R. Sun, and X. Shen, “RadioDiff-FS: Physics-informed manifold alignment in few-shot diffusion models for high-fidelity radio map construction,” IEEE Internet of Things Journal, 2026.

[30] Z. Sun, B. Fan, Q. Liu, S. Zhang, and L. Song, “RadioPiT: Radio map generation with pixel transformer driven by ultra-sparse real-world data,” in Proc. Asia-Pacific Conf. Commun. (APCC), 2025.

[31] S. Woo, S. Debnath, R. Hu, X. Chen, Z. Liu, I. S. Kweon, and S. Xie, “ConvNeXt V2: Co-designing and scaling convnets with masked autoencoders,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 16 133–16 142.

[32] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and S. Belongie,´ “Feature pyramid networks for object detection,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 936–944.

[33] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in Neural Information Processing Systems, vol. 30, 2017.

[34] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 4172–4182.