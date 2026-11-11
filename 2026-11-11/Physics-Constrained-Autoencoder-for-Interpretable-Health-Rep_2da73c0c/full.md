# Physics-Constrained Autoencoder for Interpretable Health Representation Learning in Turbofan Engines

Leonardo Baldo<sup>a,∗,†</sup>, Jokin Alcibar Ibarzabal<sup>b,†</sup>, Daniel Cisneros Acevedo<sup>c</sup>, Kai Goebel<sup>d,e</sup> and Manuel Arias Chao<sup>a,c</sup>

<sup>a</sup>Institutefor Data Science, Zurich University ofApplied Science, Winterthur, 8401, Switzerland

<sup>b</sup>Computer Science Department, Electronics, Mondragon University, Loramendi, 4, Mondragón, 20500, Spain

<sup>d</sup>Fragum Global, Mountain View, CA 94040, USA

<sup>e</sup>Luleå University of Technology, Luleå, 971 87, Sweden

<sup>c</sup>Faculty ofAerospace Engineering, Delft University ofTechnology, HS 2926 Delft, The Netherlands

## A R T I C L E I N F O

Keywords: <sub>Prognostics and Health Management</sub>v <sub>Physics-informed</sub> <sub>machine</sub> <sub>learning</sub>o Health index Interpretable representation learning Component maps Compressor map

## A BS T R AC T

Health Indices (HIs) summarize degradation from operational data and support diagnostic and prognostic tasks in Prognostics and Health Management (PHM). Their usefulness depends on capturing degradation and interpreting system condition. Autoencoders learn such representations from unlabelled monitoring data; however, when training is driven only by reconstruction, the physical meaning of latent variables remains undefined, limiting their interpretation and diagnostic value. This work proposes a physics-constrained autoencoder that embeds system knowledge as an inductive bias in the decoder architecture and as a learning bias in the loss function. The decoder contains diferentiable physical equations that receive latent variables and operating conditions as inputs, structurally guiding each latent variable to represent a predefined physical quantity and enabling component-level analysis before aggregation into a system-level HI. The methodology is benchmarked on the N-CMAPSS dataset against alternative representations. Reconstruction reaches an �<sup>2</sup> of 0.9993 and an MAE of 0.0036, comparable to fully data-driven models. After operating-condition correction, the inferred physical quantities track degradation, and their aggregation produces an HI with a Spearman correlation of 0.9791 with the ground truth, an RMSE of 0.0286, and an MAE of 0.0233. The methodology is further evaluated using a single model trained across four datasets representing distinct degradation scenarios. This model preserves component-level signatures that distinguish compression-side from expansion-side degradation without diagnostic labels. Finally, the proposed framework enables degradation trends to be visualized on component maps, providing a health representation in a form familiar to engine operators and domain experts.

## 1. Introduction

Prognostics and Health Management (PHM) has progressively reshaped the way operational data are processed and used in industrial settings [1]. The increasing availability of sensors, embedded computing resources, and advanced data-processing methodologies has made it possible to move beyond traditional maintenance strategies, enabling informed asset management based on predictive decision-making [2]. In this context, PHM provides a structured framework for transforming raw monitoring data into actionable information supporting fault detection, diagnosis, prognosis, and health-aware asset management [3]. The main objectives of PHM can therefore be formulated asa (i) assessing the current condition of an asset through diagnostics, (ii) anticipating its future degradation through prognostics, and (iii) supporting maintenance and operational decisions through health management [4].

The emergence of PHM is commonly associated with the aerospace and defence sectors, where the need for safer, more available, and more cost-efective systems has always been particularly demanding [5, 6]. Aircraft systems and mission-critical platforms, including turbofan engines [7], missile guidance systems [8], and flight-control actuators [9], have motivated the development of integrated monitoring capabilities able to detect incipient faults, estimate Remaining Useful Life (RUL), and support logistics and maintenance planning. Since then, PHM has expanded well beyond its initial aeronautic domain, spanning a plethora of applications. Today, PHM methodologies are investigated and applied in several industrial sectors, including actuation systems [10], space [11], automotive applications [12], batteries [13], and process industries [3].

Over more than two decades of development, PHM has continuously absorbed methodological advances from statistics, signal processing, control theory, Machine Learning (ML), and Deep Learning (DL). Indeed, since their inception, classical ML algorithms have enabled patterns to be extracted from monitoring data with less dependence on manually designed diagnostic rules [14]. More recently, DL has further extended these capabilities by allowing models to learn representations directly from high-dimensional monitoring data and extract health tracking variables [15].

A central concept in this context is the Health Index (HI), which is commonly introduced as a representation of the degradation state of an asset measured through a normalized version of a health indicator. In other words, it can be defined as a compact representation of asset condition designed to track degradation progression and support downstream diagnostic and prognostic tasks. In practical applications, HIs have been obtained through diferent strategies, including the selection of degradation-sensitive features, the use of model residuals, the fusion of multiple condition-monitoring signals, and the learning of latent degradation representations from data [16, 17, 18].

Among data-driven approaches, autoencoders have become a widely adopted solution for unsupervised health representation learning. Their encoder–decoder structure allows high-dimensional monitoring data to be compressed into a compact latent representation and reconstructed from it [19]. In PHM, the latent variables or a post-processed function of the latent space (e.g., a latent dynamical model) can then be used as an indicator of degradation and, consequently, as an HI. This makes autoencoder-based approaches particularly attractive when health-state labels are unavailable, run-to-failure data are limited, or the degradation state is only indirectly observed through sensor measurements, a combination frequently encountered in industrial applications [20, 21].

Despite the intuitive appeal of using a latent representation as a degradation descriptor, a standard autoencoder is designed and optimized primarily for reconstruction. As a consequence, its latent coordinates are not guaranteed to correspond to physically meaningful degradation mechanisms. In fact, they may encode operating conditions, sensor drift, noise, mission-related trends, or dataset-specific correlations that help reconstruct the input data without necessarily describing the health state of the asset. This limitation is especially significant from a user perspective, considering the vital role of HI as key metrics that operators rely on to evaluate asset condition and determine the necessity of intervention. When an HI is obtained from a black-box latent space, its physical meaning may remain unclear even when the reconstruction performance of the model is satisfactory. Moreover, even when a learned latent representation tracks degradation accurately, it may provide little information about the physical source of that degradation or the component responsible for the observed change. Such ambiguities can reduce trustworthiness, especially in safety-critical applications where operators need to understand why a model indicates degradation and whether this indication is consistent with known system behaviour. Obtaining HIs that are predictive, interpretable, physically consistent, and usable by domain experts is therefore critical for the further adoption of PHM systems in industrial settings [22].

Against this backdrop, Physics-Informed Machine Learning (PIML) ofers a possible way to address this issue by constraining the learning process with prior knowledge about the system [23]. In PHM, this prior knowledge can take the form of governing equations, simplified physical models, degradation laws, monotonicity constraints, expert knowledge, or structural assumptions about the relationships among operating conditions, health state, and measured variables [24, 25]. More specifically, the mechanisms through which prior knowledge is introduced into an ML model are commonly organized into three categories [26]: observational bias, inductive bias, and learning bias [27, 28]. Recent studies have shown that physical information can guide HI construction, regularize degradation trajectories, and improve the consistency of learned representations under limited data and varying operating conditions [26, 29]. Recent physics-derived decoders and constraint-guided optimization strategies have further demonstrated that physically meaningful quantities and degradation requirements can be embedded into autoencoder-based HI learning, reinforcing the relevance of these architectures and inherently motivating the present work [30, 31, 32].

However, existing physics-guided HI approaches predominantly constrain the evolution of a system-level health representation or embed physical information into low-dimensional latent variables. They do not generally provide a multidimensional, component-resolved latent representation in which individual coordinates are structurally associated with physical quantities of diferent components while jointly reconstructing the monitored system response. Consequently, a gap remains between learning a degradation-sensitive HI and obtaining a health representation that can also provide physically structured information about where and how degradation manifests within the system in a relatively high dimensionality setting.

This work addresses this gap by proposing a physics-decoder autoencoder for unsupervised and interpretable health representation learning in PHM. Within the proposed formulation, the encoder maps measured variables and operating conditions into a latent representation, while the decoder reconstructs the measurements through a structured model inspired by the physical organization of the monitored system. The physical prior is then of an equation-based type and physical knowledge is introduced primarily as an inductive bias and secondly with a learning bias. The diferentiable decoder is structured according to physics and component equations, restricting the learned latent representation to influence the reconstructed measurements through predefined physical modifiers. As a result, embedding the physical equations in the decoder forces each latent variable to represent a predefined physical quantity or degradation efect, thereby giving the latent space a direct physical interpretation. The latent variables reach the sensor outputs through these physically guided equations and can therefore be interpreted as degradation modifiers afecting component and system performance. This provides a component-aware physical interpretation of the latent representation, subject to the observability and identifiability of the corresponding quantities. The framework is finally benchmarked and validated on several N-CMAPSS turbofan datasets [33]. The main contributions of this work are as follows:

• a general physics-decoder autoencoder framework for unsupervised and interpretable health representation learning in PHM, which embeds diferentiable system equations in the decoder and physics-based residuals in the loss function as complementary inductive and learning biases;

• a component-aware formulation that forces each latent variable in a 15-dimensional space to represent a predefined physical quantity and act as a degradation-sensitive modifier within the corresponding equations, enabling component-level examination before system-level aggregation;

• a strategy that removes the operating-condition contribution from the learned physical latent variables and combines their most informative degradation trends into a system-level health index;

• an extensive N-CMAPSS evaluation spanning four degradation scenarios, including benchmarks against alternative representations, an independent consistency assessment using two high-fidelity turbofan simulators, an analysis of latent-space observability, visualization through component maps, and separation of compressorside from turbine-side degradation without training a diagnostic classifier.

## 2. Health Indices and Positioning of the Proposed Approach

Health Indices (HIs) provide the interface between condition-monitoring data and health-informed decisions. For prognostic applications, an HI should follow degradation consistently and retain information relevant to future steps in the PHM pipeline, such as health assessment, RUL estimation, and maintenance decision-making. Monotonicity, trendability, robustness, prognosability, and consistency across units are therefore commonly used to evaluate HI quality [34, 35].

The evolution of HI construction reflects the broader methodological development of PHM. In the next paragraph a non-exhaustive list of the major contributions are examined and the results are reported in Table 1, which summarizes the main HI construction families and clarifies the positioning of the proposed formulation. A more detailed overview of the most recent methodologies to extract HI is provided by Bajarunas et al. [29].

Early approaches relied on degradation-sensitive statistics and signal-processing features selected through knowledge of the monitored component and its failure mechanisms (e.g. kurtosis, spectral amplitudes, and energy-related quantities) [35, 16]. Model-based diagnostic approaches subsequently introduced a more explicit description of system behaviour, allowing asset condition to be represented through residuals or estimated parameters [4, 34]. DL methods shifted HI construction towards representations learned directly from monitoring data. Autoencoders are particularly suitable for this purpose because they compress unlabeled and high-dimensional measurements while preserving the information required for their reconstruction. This is why they are among the most popular architectures for extracting HI from unlabeled operational data in an unsupervised way. For instance, González-Muñiz et al. construct an HI from the discrepancy between the latent representations of the measured and reconstructed signals [20]. De Pater and Mitici aggregate the reconstruction errors of an operating-condition-informed LSTM autoencoder [21]. These approaches provide data-driven measures of drift from learned system behaviour. However, their physical interpretation remains limited because the reconstruction objective does not assign specific meanings to the latent coordinates. Consequently, the latent space can combine degradation with operating conditions, sensor characteristics, and other sources of variability that contribute to reconstruction. This limitation has motivated the introduction of prior knowledge into health representation learning. One of the first examples is the one by Bajarunas et al., who encode structural assumptions concerning the relationships among operating conditions, degradation, and measurements, and introduce constraints that promote degradation-consistent latent trajectories [26, 29]. Tefera et al. guide the estimation of a scalar HI through monotonicity, boundary, and signal-energy consistency constraints [31]. These contributions shape the behaviour of the learned representation through degradation-related knowledge, while retaining data-driven mappings between the latent variables and the reconstructed measurements. A further development consists of embedding physical knowledge within the autoencoder decoder itself. The closest methodological reference is the physics-informed autoencoder proposed by Huang et al. [30]. Their encoder estimates a scalar latent HI with a predefined energy-oriented interpretation, corresponding to battery state of health or electromechanical-actuator eficiency. Physics-derived feature terms are introduced in the decoder and combined through trainable linear or neural layers to reconstruct the monitored output. The reconstruction process therefore links the scalar latent variable to a defined measure of overall system performance. A related formulation is proposed by Plonczak et al. [32], who couple a convolutional encoder with an equivalent-circuit model whose reconstruction equations force the latent variables to represent efective capacity and internal resistance for battery packs.

Representative HI construction strategies and positioning of the proposed component-resolved physics-decoder autoencoder.
<table><tr><td>Method family</td><td>HI construction and knowledge integration</td><td>Interpretability and scope</td><td>Representative refs.</td></tr><tr><td>Expert-designed HIs</td><td>Statistical or signal-processing quantities selected from known degradation signatures.</td><td>Explicit meaning, but dependent on operating conditions, failure mode, and feature selection.</td><td>[35, 16]</td></tr><tr><td>Model- and residual- based Hls</td><td>Residuals or estimated parameters based in regres- sion based models. Knowledge enters through the ual definition; performance depends on reference model structure and reference behaviour.</td><td>Interpretation follows from the model and resid- [4, 34] quality and model fidelity.</td><td></td></tr><tr><td>Autoencoder residual Hls</td><td>Input or latent reconstruction errors or sensor reconstruction losses, possibly conditioned.</td><td>Quantify departure from learned behaviour with- [20, 21] out identifying the physical source of the change. Latent representations or HI outputs constrained Degradation-consistent trajectories; meaning de- [26, 29, 31]</td><td></td></tr><tr><td>Knowledge- constrained learned Hls</td><td>by causal structure, monotonicity, temporal order- pends on the imposed constraints. ing, or boundary conditions.</td><td></td><td></td></tr><tr><td>Hybrid decoder Hls</td><td>physics- A scalar latent space with predefined meaning Provides a system-level physical interpretation, [30, 32] enters physics-derived decoder terms, possibly but estimates a single HI and requires a system- combined by later trainable layers.</td><td>specific model.</td><td></td></tr><tr><td>Proposed component-resolved physics-decoder AE</td><td>A latent vector modifies specific quantities Coordinates are structurally associated with This work within coupled equations that directly recon- struct multiple measurements.</td><td>component and performance degradation ef- fects; interpretability depends on model fidelity and latent identifiability/observability.</td><td></td></tr></table>

Building on this line of research, the present work introduces an equation-based autoencoder for multidimensional and physically structured health representation learning. The methodological formulation assigns the latent coordinates to modifiers of named physical quantities and embeds these modifiers within a coupled system of diferentiable equations. A system-level HI can subsequently be extracted from this structured representation while preserving information about its underlying physical contributions. Unlike Huang et al., the equations directly constitute the decoder and transform the latent representation and operating conditions into reconstructed measurements, with no neural mapping after the equations. This more constrained structure allows the latent space to retain several degradation contributions and associates each coordinate with the physical quantity it modifies. Moreover, Plonczak et al. constrain two latent variables through a battery equivalent-circuit model. We extend this idea to multicomponent systems through a 15-dimensional, component-resolved latent space and a decoder assembled from component equations instead of a model. The framework also removes operating-condition efects before HI construction and evaluates latent observability and component-level diagnosis across four degradation scenarios in diferent areas of the monitored asset.

## 3. Methodology: Physics-Constrained Autoencoder Formulation

The proposed methodology aims to infer a structured latent representation whose coordinates are associated with predefined physical quantities and whose evolution can subsequently be examined for health assessment. Its application requires prior knowledge of the monitored system in the form of diferentiable physical relations linking the operating conditions, the relevant component parameters, and the measured system response. This knowledge defines the inductive and learning bias introduced into the autoencoder and indirectly determines the physical meaning assigned to the latent coordinates.

Consider a monitored system described at time � by a vector of measured variables $\mathbf { x } _ { t } \in \mathbb { R } ^ { n _ { y } }$ and a vector of operating conditions $\mathbf { w } _ { t } \in \mathbb { R } ^ { n _ { u } }$ . The objective is to infer a latent representation $\mathbf { z } _ { t } \in \mathbb { R } ^ { n _ { z } }$ from which the measured system response can be reconstructed. The resulting latent representation is expected to retain information on the physical state of the monitored components. Since physical component quantities may also vary with the operating regime, a subsequent operating-condition correction may be required before their temporal evolution can be interpreted in terms of degradation.

## 3.1. Conventional Conditional Autoencoder Formulation

A conventional conditional autoencoder consists of an encoder $\mathcal { E } _ { \phi } ~ : ~ \mathbb { R } ^ { n _ { y } + n _ { w } } ~ \to ~ \mathbb { R } ^ { n _ { z } }$ and a neural decoder $D _ { \theta } : \mathbb { R } ^ { n _ { z } + n _ { w } }  \mathbb { R } ^ { n _ { y } }$ , both composed of trainable neural layers:

$$
\begin{array} { r } { \mathbf { z } _ { t } = \mathcal { E } _ { \phi } \big ( \mathbf { x } _ { t } , \mathbf { w } _ { t } \big ) , } \end{array}\tag{1}
$$

$$
\widehat { \mathbf { x } } _ { t } = D _ { \theta } \big ( \mathbf { z } _ { t } , \mathbf { w } _ { t } \big ) ,\tag{2}
$$

where $\phi$ and $\theta$ denote the trainable parameters of the encoder and decoder, respectively. Without loss of generality, the domains of the two mappings are written assuming that the measured variables and operating conditions are concatenated at the encoder input, while the latent representation and operating conditions are concatenated at the decoder input. Alternative conditioning mechanisms, including feature-wise modulation [36] or the injection of $\mathbf { w } _ { t }$ only at selected layers, can be adopted without changing the general formulation.

Providing $\mathbf { w } _ { t }$ to both mappings allows the encoder to infer a latent representation conditional on the current operating regime and allows the decoder to reconstruct the measurements at the same operating point. However, when training is based exclusively on reconstruction, no constraint requires the latent representation to be independent of $\mathbf { w } _ { t }$ . The operating-condition information may therefore remain encoded in $\mathbf { z } _ { t }$ , even when it is explicitly supplied to the decoder. Moreover, no predefined physical meaning is assigned to the individual latent coordinates, since the neura architecture does not specify their correspondence with the physical quantities governing the system response.

## 3.2. Physics-Constrained Decoder

In the proposed formulation, the neural decoder $D _ { \theta }$ is replaced by a diferentiable physical model $\mathcal { P } ( \cdot )$ , as shown in Figure 1. Each latent coordinate is associated with a physical quantity selected through an analysis of the system structure, its governing equations, and the degradation mechanisms of interest. This analysis can be carried out for instance as a sensitivity analysis of the model to identify which variables should be placed in the latent space. The encoder output can be first transformed into a vector of dimensionless physical modifiers: $\begin{array} { r } { \mathbf { m } _ { t } \ = \ \mathcal { R } \big ( \mathbf { z } _ { t } \big ) \ = \ } \end{array}$ $\left[ r _ { 1 } ( z _ { 1 , t } ) \quad \cdots \quad r _ { n _ { z } } \Big ( z _ { n _ { z } , t } \Big ) \right] ^ { \top }$ , where $\mathcal { R } ( \cdot )$ is a fixed component-wise transformation and $r _ { j } ( \cdot )$ maps the �-th latent coordinate to the admissible range of its assigned physical modifier. The form of each transformation is defined before training according to the domain and expected variation of the corresponding physical quantity. For example, a strictly positive modifier can be defined as $m _ { k , t } \overset { \cdot } { = } r _ { k } \bigl ( z _ { k , t } \bigr ) = e ^ { 0 . 0 5 z _ { k , t } }$ for a generic component �. The transformations introduce no additional trainable parameters and ensure positivity.

Let $\mathbf { p } _ { 0 } \in \mathbb { R } ^ { n _ { z } }$ denote the nominal (reference) values of the physical quantities associated with the latent coordinates. Their time-dependent values are defined as $\mathbf { p } _ { t } = \mathbf { p } _ { 0 }$ ⊙ � , with ⊙ that denotes the Hadamard product. The vector $\mathbf { p } _ { t }$ collects the physical parameters through which changes in the monitored components afect the system response. The reconstructed measurements are obtained by evaluating the physical model under the observed operating conditions:

$$
\begin{array} { r } { \widehat { \mathbf { x } } _ { t } = \mathcal { P } \big ( \mathbf { w } _ { t } , \mathbf { p } _ { t } ; \mathbf { \boldsymbol { \Theta } } \big ) , } \end{array}\tag{3}
$$

where $\mathbf { \delta } \Theta$ contains the fixed (i.e., non-trainable) physical constants and nominal model quantities that are not inferred from the latent representation. The function $\mathcal { P } ( \cdot )$ comprises the governing physical relations, the selection of the reconstructed variables, the required unit conversions, and the normalization applied to the model outputs.

Combining the previous equations, the complete mapping is written as

$$
\widehat { \mathbf { x } } _ { t } = \mathcal { P } \big ( \mathbf { w } _ { t } , \mathbf { p } _ { 0 } \odot \mathcal { R } \big ( \mathcal { E } _ { \phi } \big ( \mathbf { x } _ { t } \big ) \big ) ; \mathbf { \Theta } \big ) .\tag{4}
$$

For a training set containing � observations, the encoder parameters can be estimated by minimizing a composite loss. The reconstruction term is expressed through Equation 5:

$$
\mathcal { L } _ { \mathrm { r e c } } ( \phi ) = \frac { 1 } { N } \sum _ { k = 1 } ^ { N } \left\| { \bf x } _ { t _ { k } } - \widehat { { \bf x } } _ { t _ { k } } \right\| _ { 2 } ^ { 2 } .\tag{5}
$$

![](images/cbabbff4964df8e8f2ea87004edeec4effef186d8b35d6df10341c9218006253.jpg)  
Figure 1: Generic architecture of the proposed physics-constrained autoencoder. The encoder maps the normalized measurements $\mathbf { X } _ { t }$ into a structured latent representation $\mathbf { z } _ { t }$ . Each latent variable is associated with a predefined physical quantity and is propagated, together with the operating conditions $\mathbf { w } _ { t }$ , through the diferentiable physical decoder to reconstruct the measured variables $\hat { \mathbf { x } } _ { t }$

On top of this term, additional terms can introduce more physical knowledge in the form of learning bias. The general objective can be formulated as

$$
\mathcal { L } ( \phi ) = \mathcal { L } _ { \mathrm { r e c } } ( \phi ) + \sum _ { j = 1 } ^ { J } \lambda _ { j } \mathcal { L } _ { j } ( \phi ) ,\tag{6}
$$

where $\mathcal { L } _ { j }$ represents an application-specific physical relation or constraint and $\lambda _ { j }$ controls its contribution. These terms introduce a learning bias complementary to the inductive bias imposed by the equation-based decoder.

In this formulation, the physical decoder contains no trainable neural layers. Huang et al. [30] introduced a related architecture in which the physical equations were followed by conventional trainable layers within the decoder. In the present formulation, the normalized outputs of the physical equations directly constitute the reconstructed measurement vector. The resulting structure requires a suficiently detailed and diferentiable description of the monitored process, since the physical model must account for the complete mapping from the inferred component parameters to the reconstructed variables. This restriction also prevents additional decoder layers from compensating for inaccuracies in the inferred physical modifiers or from encoding degradation-related information through trainable parameters.

Training is consequently restricted to identifying the encoder parameters $\phi$ that generate physically admissible modifiers capable of reconstructing the observed system response through $\mathcal { P } ( \cdot )$ . The imposed model structure specifies the physical quantity associated with each latent coordinate and the equations through which changes in that quantity influence the measurements. The architecture therefore provides a structural interpretation of the latent representation and enables the individual latent trajectories to be examined at the level of components or degradation mechanisms.

## 3.3. Operating-Condition Correction of the Latent Representation

The physical interpretation assigned to the latent coordinates does not ensure that their values depend exclusively on degradation. In systems operating under variable conditions, the physical quantities represented in the latent space may change across the operating envelope even when the system remains healthy (e.g., turbofan engines [37]). The �-th latent variable can therefore be decomposed as

$$
z _ { j , t } = z _ { j } ^ { \mathrm { H } } \big ( \mathbf { w } _ { t } \big ) + \delta z _ { j , t } ^ { \mathrm { D } } + \varepsilon _ { j , t } ,\tag{7}
$$

where $z _ { j } ^ { \mathrm { H } } \big ( \mathbf { w } _ { t } \big )$ denotes the value expected for the $j \cdot$ th latent quantity under healthy conditions at operating point $\mathbf { w } _ { t } , \delta z _ { j , t } ^ { \mathrm { D } }$ denotes the deviation associated with degradation, and $\varepsilon _ { j , t }$ accounts for estimation errors, model mismatch, and unmodelled efects. According to Eq. (7), variations observed in the raw latent trajectories may result from changes in the operating conditions, degradation, or their combined efect. A direct use of $\mathbf { z } _ { t }$ for health assessment may therefore transfer operating-regime variability into the resulting HI. The healthy operating-condition contribution must be estimated whenever its magnitude is not negligible with respect to the degradation-related variation.

Let  denote a reference set containing observations representative of healthy system operation:

$$
\mathcal { H } = \left\{ \left( \mathbf { w } _ { t } , \mathbf { z } _ { t } \right) : t \in \mathcal { T } _ { \mathrm { H } } \right\} ,\tag{8}
$$

where $\tau _ { \mathrm { H } }$ is the set of time indices associated with the healthy reference condition. For each latent coordinate $j \in \{ 1 , \dots , n _ { z } \}$ , a model $h _ { j } : \mathbb { R } ^ { n _ { w } }  \mathbb { R }$ is identified from the healthy reference set to estimate its dependence on the operating conditions:

$$
\widehat { h } _ { j } = \mathop { \mathrm { a r g m i n } } _ { h _ { j } } \sum _ { t \in \mathcal { T } _ { \mathrm { H } } } \left[ z _ { j , t } - h _ { j } \big ( \mathbf { w } _ { t } \big ) \right] ^ { 2 } .\tag{9}
$$

The fitted function $\widehat { h } _ { j } ( \mathbf { w } _ { t } )$ estimates the value expected for the �-th latent variable under healthy conditions at the operating point $\mathbf { w } _ { t } .$ The corrected latent coordinate is then computed as $\widetilde { z } _ { j , t } = z _ { j , t } - \hat { z } _ { j } ^ { \mathrm { H } } \big ( \mathbf { w } _ { t } \big )$ . and the vector of corrected latent variables is consequently defined as

$$
\widetilde { \mathbf { z } } _ { t } = \mathbf { z } _ { t } - \widehat { \mathbf { z } } ^ { \mathrm { H } } \big ( \mathbf { w } _ { t } \big ) , \qquad \widehat { \mathbf { z } } ^ { \mathrm { H } } \big ( \mathbf { w } _ { t } \big ) = \left[ \widehat { z } _ { 1 } ^ { \mathrm { H } } \big ( \mathbf { w } _ { t } \big ) \quad \cdots \quad \widehat { z } _ { n _ { z } } ^ { \mathrm { H } } \big ( \mathbf { w } _ { t } \big ) \right] ^ { \intercal } .\tag{10}
$$

Under an adequate representation of the healthy operating-condition dependence, the corrected coordinate $\widetilde { z } _ { j , t }$ approximates the degradation-related contribution: $\widetilde { z } _ { j , t } \approx \delta z _ { j , t } ^ { \mathrm { D } } + \varepsilon _ { j , t }$

The residual representation $\widetilde { \mathbf { z } } _ { t }$ can be retained as a multidimensional description of the health state, with individual coordinates associated with specific components or degradation mechanisms. It can also be combined into a scalar HI through a subsequent aggregation function $\mathcal { G } : \mathbb { R } ^ { n _ { z } }  \mathbb { R } \colon \mathrm { H I } _ { t } = \mathcal { G } \big ( \widetilde { \mathbf { z } } _ { t } \big )$ . The selection of the healthy reference set, the function class $\mathcal { F } _ { j }$ , and the aggregation function $\mathcal { G } ( \cdot )$ depends on the available knowledge and on the characteristics of the considered application. Their specific implementation for the turbofan case study is described in Section 4. For component-level comparisons, the corrected latent contribution is added to the healthy baseline evaluated at a common reference operating condition,

$$
\begin{array} { r } { { \bf z } _ { t , \mathrm { r e f } } = \hat { \bf z } ^ { \mathrm { H } } ( { \bf w } _ { \mathrm { r e f } } ) + \tilde { \bf z } _ { t } . } \end{array}\tag{11}
$$

This operation retains the inferred degradation-related contribution while replacing the healthy operating-condition contribution with its value at $\mathbf { w } _ { \mathrm { r e f } }$ . The resulting latent coordinates are then converted into physical quantities through the decoder mappings, allowing their evolution to be compared at a common operating point.

## 4. Case Study: Physics-Decoder Autoencoder for N-CMAPSS

This section specializes the general framework to a turbofan-engine case study based on the N-CMAPSS dataset. The objective is to show how a physics-structured decoder can be implemented when the measured variables are thermodynamic and mechanical quantities associated with engine operation [38]. The specific implementation of the physics constrained autoencoder for the N-CMAPSS turbofan application is reported in Figure 2.

## 4.1. N-CMAPSS Dataset

The New Commercial Modular Aero-Propulsion System Simulation (N-CMAPSS) dataset provides run-to-failure trajectories for fleets of simulated turbofan engines operating under realistic flight conditions [33]. It is suitable for the present study because, on top of being one of the most popular datasets of PHM evaluation, it contains operating conditions, multivariate sensor measurements, and degradation trajectories generated using a physics-based turbofan simulation environment. Among its eight subsets, DS003 is selected as the main case study for model development, HI construction, and benchmarking. It represents a combined degradation mode afecting the eficiency and flow capacity of the low-pressure turbine and the eficiency of the high-pressure turbine. DS004–DS006, which introduce degradation in the fan and compressor sections, are subsequently used to test the same architecture across diferent degradation modes and assess the component-level diagnostic information retained by a model trained on the pooled scenarios. For each DS003 engine, the dataset contains 14 sensor variables collected throughout its operating life, from the initial healthy condition to failure. Twelve variables, denoted by �, are retained in the present formulation. The fan and core shaft speeds, $N _ { f }$ and $N _ { c }$ , are excluded from the reconstruction targets because, as customarily done in simple turbofan modeling, the adopted thermodynamic formulation does not explicitly relate spool speeds to the reconstructed gas-path quantities [39]. The corresponding operating conditions, denoted by �, comprise altitude, Mach number, throttle-resolver angle, and total temperature at the fan inlet. The dataset also provides the model health parameters, �, comprising eficiency and flow-capacity modifiers for the fan, Low-Pressure Compressor (LPC), High-Pressure Compressor (HPC), High-Pressure Turbine (HPT), and Low-Pressure Turbine (LPT). These parameters describe the component degradation imposed during data generation and provide reference trajectories for validating the inferred latent quantities. The ground-truth HI, $h _ { \mathrm { G T } } ,$ , is obtained from operational margins evaluated under reference conditions, with failure reached when $h _ { \mathrm { G T } }$ decreases to zero.

![](images/aa28f48dc4f00a8742e40852a69b5f6723589d1d19126b42978f2122d1ef354f.jpg)  
Figure 2: Implementation of the physics-constrained autoencoder for the N-CMAPSS turbofan application. The 15- dimensional latent space represents physically assigned engine parameters, which are combined with the operating conditions in the physical decoder. Model training minimizes the reconstruction error together with the residuals associated with the turbine shaft-power balances.

## 4.2. Equations

## 4.2.1. Diferentiable equation-based decoder

The physical decoder implements a simplified two-spool turbofan model as a diferentiable computation graph, using textbook thermodynamic equations from [39]. Internal thermodynamic states follow conventional engine station numbering, while measured and reconstructed channels retain their N-CMAPSS names to facilitate comparison with the dataset. For clarity, Figure 3 brings these conventions together: it shows the engine stations, the N-CMAPSS channels in red boxes, the cooling flows, and the latent health modifiers associated with each component. The correspondence between station labels and N-CMAPSS variables is also listed in Appendix F, and Table 2 summarizes the symbols used throughout the formulation. The decoder receives altitude �(�), flight Mach number �(�), throttleresolver angle TRA(�), and fan-inlet total temperature $T _ { 2 } ( t )$ as operating conditions. Each latent variable adjusts a nominal design quantity through a positive multiplicative modifier:

$$
q _ { k } ( t ) = q _ { k } ^ { 0 } { \bmod { k } } _ { , t } , \qquad { \bmod { k } } _ { , t } = r { \big ( } z _ { k , t } { \big ) } = e ^ { 0 . 0 5 z _ { k , t } } ,\tag{12}
$$

The exponential mapping is smooth and strictly increasing, preserves positivity, and gives mo $\mathsf { I } _ { k , t } = 1$ at $z _ { k , t } = 0 ;$ recovering the calibrated nominal engine at the latent origin. It permits both increases and decreases relative to the nominal value without imposing a fixed range on the modifier. The coeficient 0.05 sets the latent-to-modifier scale: near the origin, $\mathrm { m o d } _ { k , t } \approx 1 + 0 . 0 5 z _ { k , t }$ , so a unit latent change corresponds approximately to a 5% relative change in the associated quantity.The set of equations that we will present in the following paragraphs requires 16 quantities as input. These quantities are:

$$
\begin{array} { r l } & { \mathbf { p } ( t ) = \{ \eta _ { \mathrm { f a n } } , \eta _ { \mathrm { L P C } } , \eta _ { \mathrm { H P C } } , \eta _ { \mathrm { H P T } } , \eta _ { \mathrm { L P T } } , \Pi _ { \mathrm { f a n } } , \Pi _ { \mathrm { L P C } } , \Pi _ { \mathrm { H P C } } , \Pi _ { \mathrm { H P T } } , \Pi _ { \mathrm { L P T } } , } \\ & { \qquad W _ { \mathrm { m o d } } , \mathrm { B P R } , T _ { 4 1 } , \beta _ { \mathrm { c o m b } } , \beta _ { \mathrm { H P T } } , \beta _ { \mathrm { L P T } } \} , } \end{array}\tag{13}
$$

![](images/fd189d14717f23170b9bafb3354267c0d7768c91f9c795244d017ad90498e0ea.jpg)  
Figure 3: Schematic of the physics-based turbofan decoder and its correspondence with the N-CMAPSS variables and latent health modifiers. The engine stations and component decomposition are shown together with the measured N-CMAPSS variables used as model inputs and reconstruction targets, highlighted in red. Component colors identify the fan, LPC, HPC, compressor, HPT, and LPT. The latent variables � parameterize component-level health modifiers associated with eficiencies, pressure ratios, flow capacity, and selected thermodynamic quantities, while the operating conditions enter through the free-stream variables.

comprising five component eficiencies, three compressor pressure ratios, the two turbine expansion ratios, the inlet flow capacity, the bypass ratio, the turbine-inlet temperature level, and three cooling-flow fractions.

The inlet conditions are calculated using an International Standard Atmosphere approximation and the isentropic ram relations:

$$
P _ { \mathrm { a m b } } = P _ { \mathrm { S L } } \left( 1 - k _ { H } H \right) ^ { 5 . 2 5 5 8 8 } , \quad T _ { \mathrm { a m b } } = \frac { T _ { 2 } } { 1 + \frac { \gamma - 1 } { 2 } M ^ { 2 } } , \quad P _ { 0 } = P _ { \mathrm { a m b } } \left( \frac { T _ { 2 } } { T _ { \mathrm { a m b } } } \right) ^ { \frac { \gamma } { \gamma - 1 } } , \quad P _ { 2 } = ( 1 - \delta _ { \mathrm { i n } } ) P _ { 0 } .\tag{14}
$$

The inlet mass flow is obtained from the compressible mass-flow parameter:

$$
\mathrm { M F P } ( M ) = \frac { M \sqrt { \gamma / R } } { \left( 1 + \frac { \gamma - 1 } { 2 } M ^ { 2 } \right) ^ { \frac { \gamma + 1 } { 2 ( \gamma - 1 ) } } } , \qquad W _ { 0 } = m o d _ { W _ { 0 } } \frac { P _ { 2 } A _ { \mathrm { i n } } \mathrm { M F P } ( M ) } { \sqrt { T _ { 2 } } } .\tag{15}
$$

Pressure losses across the inlet, ducts, difuser, combustor, and jet pipe are represented as

$$
P _ { \mathrm { o u t } } = \left( 1 - \delta \right) P _ { \mathrm { i n } } .\tag{16}
$$

The fan and compressors are described by

$$
P _ { \mathrm { o u t } } = \Pi _ { c } P _ { \mathrm { i n } } ,\tag{17}
$$

Table 2  
Symbols and physical quantities used in the turbofan decoder.
<table><tr><td>Category</td><td>Symbol</td><td>Meaning</td><td>Symbol Meaning</td><td></td><td></td><td>Symbol Meaning</td></tr><tr><td>Operating conditions</td><td>H</td><td>Altitude</td><td>M</td><td>Flight Mach number</td><td> $T _ { 2 }$ </td><td>Fan-inlet total temperature</td></tr><tr><td></td><td>TRA</td><td>Throttle-resolver angle</td><td> $\widehat { \mathrm { T R A } }$ </td><td>Normalised throttle setting</td><td></td><td></td></tr><tr><td>Atmospheric quantities</td><td> $T _ { \mathrm { a m b } }$ </td><td>Ambient static temperature</td><td> $P _ { \mathrm { a m b } }$ </td><td>Ambient static pressure</td><td> $P _ { \mathrm { S L } }$ </td><td>Sea-level pressure</td></tr><tr><td></td><td> $P _ { 0 }$ </td><td>Freestream total pressure</td><td> $k _ { H }$ </td><td>ISA altitude coefficient</td><td> $R$ </td><td>Specific gas constant</td></tr><tr><td>Gas properties</td><td> $\gamma$ </td><td>Specific-heat ratio, air</td><td> $\gamma _ { t }$ </td><td>Specific-heat ratio, turbine gas</td><td> $c _ { p }$ </td><td>Specific heat, air</td></tr><tr><td></td><td>cp,t</td><td>Specific heat, turbine gas</td><td> $c _ { p , \mathrm { m i x } }$ </td><td>Mixed-stream specific heat</td><td> $\overset { \underset { r } { } } { \mathrm { M F P } }$ </td><td>Mass-flow parameter</td></tr><tr><td></td><td> $\mathrm { L H V }$ </td><td>Fuel lower heating value</td><td> $\eta _ { \mathrm { c o m b } }$ </td><td>Combustion efficiency</td><td> $\eta _ { m }$ </td><td>Mechanical efficiency</td></tr><tr><td>Component efficiencies</td><td> $\eta _ { \mathrm { f a n } }$ </td><td>Fan efficiency</td><td> $\eta _ { \mathrm { L P C } }$ </td><td>LPC efficiency</td><td> $\eta _ { \mathrm { H P C } }$ </td><td>HPC efficiency</td></tr><tr><td></td><td> $\eta _ { \mathrm { H P T } }$ </td><td>HPT efficiency</td><td> $\eta _ { \mathrm { L P T } }$ </td><td>LPT efficiency</td><td> $\eta _ { c } , \eta _ { t }$ </td><td>Generic compressor, turbine effi- ciency</td></tr><tr><td>Pressure</td><td> $\Pi _ { \mathrm { f a n } }$ </td><td>Fan pressure ratio</td><td> $\Pi _ { \mathrm { L P C } }$ </td><td>LPC pressure ratio</td><td> $\Pi _ { \mathrm { H P C } }$ </td><td>HPC pressure ratio</td></tr><tr><td>ratios</td><td> $\Pi _ { _ \mathrm { - } }$ </td><td>HPT expansion ratio</td><td> $\Pi _ { \mathrm { L P T } }$ </td><td>LPT expansion ratio</td><td> $\Pi _ { c } , \Pi _ { t }$ </td><td>Generic compressor, turbine ratio</td></tr><tr><td></td><td>δ</td><td>Pressure-loss fraction</td><td> $\delta _ { \mathrm { i n } }$ </td><td>Inlet pressure-loss fraction</td><td> $P _ { \mathrm { i n } } ^ { \mathrm { ^ { \circ } } } , P _ { \mathrm { o u t } }$ </td><td>Element inlet, outlet pressure</td></tr><tr><td>Mass-flow</td><td> $W _ { 0 }$ </td><td>Total inlet mass flow</td><td> $W _ { \mathrm { c o r e } }$ </td><td>Core mass flow</td><td> $W _ { \mathrm { b y p a s s } }$ </td><td>Bypass mass flow</td></tr><tr><td>quantities</td><td> $W _ { 3 }$ </td><td>HPC-exit mass flow</td><td> $W _ { 3 1 }$ </td><td>Combustor-inlet airflow</td><td></td><td>Airflow available for combustion</td></tr><tr><td></td><td> $\dot { W _ { 4 1 5 } } , W _ { 4 6 }$ </td><td>HPT, LPT gas-path mass flow</td><td> $W _ { f }$ </td><td>Fuel mass flow</td><td> $W _ { \mathrm { c o m b } }$   $W _ { \mathrm { m o d } }$ </td><td>Inlet flow-capacity modifier</td></tr><tr><td></td><td> $\beta _ { \mathrm { b l } }$ </td><td>Customer-bleed fraction</td><td> $\beta _ { \mathrm { m e c } }$ </td><td>Mechanical-offtake fraction</td><td> $A _ { \mathrm { i n } }$ </td><td>Effective inlet area</td></tr><tr><td>Flow distribu- BPR</td><td></td><td>Bypass ratio</td><td> $\beta _ { \mathrm { c o m b } }$ </td><td>Combustor-cooling fraction</td><td></td><td></td></tr><tr><td>tion</td><td> $\beta _ { \mathrm { H P T } }$ </td><td>HPT-cooling fraction</td><td> $\beta _ { \mathrm { L P T } }$ </td><td>LPT-cooling fraction</td><td></td><td></td></tr><tr><td>Thermal</td><td> $T _ { \mathrm { i n } } , T _ { \mathrm { o u t } }$ </td><td>Element inlet, outlet temperature</td><td> $T _ { \mathrm { m i x } }$ </td><td>Mixed-stream temperature</td><td> $T _ { 4 1 }$ </td><td>Turbine-inlet temperature (SOT)</td></tr><tr><td>quantities</td><td> $T _ { 4 1 } ^ { 0 }$ </td><td>Idle turbine-inlet temperature LPT exit temperature, before exit</td><td> $T _ { 4 1 } ^ { \prime }$ </td><td>Throttle sensitivity of  $T _ { 4 1 }$ </td><td> $P _ { i } , T _ { i }$ </td><td>Conditions at station i</td></tr><tr><td></td><td> $T _ { \mathrm { L P T , e x } }$ </td><td>mixing</td><td></td><td></td><td></td><td></td></tr><tr><td>Power balance</td><td> $\dot { Q } _ { \mathrm { f a n } }$ </td><td>Fan power HP-shaft power residual</td><td> $\dot { Q } _ { \mathrm { L P C } }$   $r _ { \mathrm { L P } }$ </td><td>LPC power LP-shaft power residual</td><td> $\operatorname* { \dot { \cal Q } } _ { \lambda } _ { } \mathtt { H P C }$ </td><td>HPC power Power-balance penalty weight</td></tr><tr><td>Output</td><td> $r _ { \mathrm { H P } }$   $\mathbf { x _ { \mathrm { p h y s } } }$ </td><td>Decoder output vector</td><td>ê</td><td></td><td> $\mathcal { L }$ </td><td>Training objective</td></tr></table>

$$
T _ { \mathrm { o u t } } = T _ { \mathrm { i n } } \left[ 1 + \frac { \Pi _ { c } ^ { ( \gamma - 1 ) / \gamma } - 1 } { \eta _ { c } } \right] .\tag{18}
$$

Equations (17) and (18) are applied sequentially to the fan, low-pressure compressor, and high-pressure compressor using the corresponding latent-modified eficiencies and pressure ratios.

The HPC exit requires one distinction that the remaining stations do not. The decoder propagates total pressures throughout, whereas the N-CMAPSS channel $P _ { s 3 0 }$ is a static pressure and $P _ { 4 1 }$ a total one. Both are reported from the single computed quantity $P _ { 3 } ,$ so the ratio between them is fixed by the decoder and it must equal the ratio the dataset holds. To further examine this behaviour, the two ratios were evaluated on the healthy development split and found to remain constant across the operating conditions represented in N-CMAPSS.

$$
\frac { P _ { 3 0 } } { P _ { s 3 0 } } = 1 . 0 6 6 0 ( \mathrm { C V } 0 . 2 1 \% ) , \qquad \frac { P _ { 4 1 } } { P _ { 3 0 } } = 0 . 9 5 5 0 ( \mathrm { C V } < 1 0 ^ { - 4 } \% ) ,\tag{19}
$$

the first being the dynamic-head contribution at the compressor exit and the second the combined difuser and combustor total-pressure loss. The two reported pressures are therefore formed as

$$
P _ { s 3 0 } = { \frac { P _ { 3 } } { 1 . 0 6 6 0 } } , \qquad P _ { 4 1 } = 0 . 9 5 5 0 P _ { 3 } ,\tag{20}
$$

while $P _ { 3 1 } = ( 1 - \delta _ { \mathrm { d i f f } } ) P _ { 3 }$

The fan flow is divided into core and bypass streams according to

$$
W _ { \mathrm { c o r e } } = \frac { W _ { 0 } } { 1 + \mathrm { B P R } } , \qquad W _ { \mathrm { b y p a s s } } = W _ { 0 } - W _ { \mathrm { c o r e } } .\tag{21}
$$

Customer bleed and mechanical oftakes are extracted at the HPC as prescribed fractions of the core flow, $W _ { 3 } \ =$ $W _ { \mathrm { c o r e } } ( 1 - \beta _ { \mathrm { b l } } - \beta _ { \mathrm { m e c } } )$ . The cooling flows are then extracted at the compressor delivery, all of them as fractions of the core flow. The combustor cooling rejoins in the combustor, and the two turbine cooling streams bypass the combustor: the HPT stream rejoins the gas path at the HPT inlet, where it mixes with the combustor exhaust and expands through the high-pressure turbine, whereas the LPT stream rejoins downstream of the low-pressure turbine, in the jet pipe. Over the 425,156 test samples the resulting core mass imbalance is around −0.5%, which is the mechanical oftake $\beta _ { \mathrm { m e c } }$ leaving the engine.:

$$
W _ { 3 1 } = W _ { 3 } - W _ { \mathrm { c o r e } } \left( \beta _ { \mathrm { c o m b } } + \beta _ { \mathrm { H P T } } + \beta _ { \mathrm { L P T } } \right) , \qquad W _ { \mathrm { c o m b } } = W _ { 3 1 } + W _ { \mathrm { c o r e } } \beta _ { \mathrm { c o m b } } = W _ { 3 } - W _ { \mathrm { c o r e } } \left( \beta _ { \mathrm { H P T } } + \beta _ { \mathrm { L P T } } \right) .\tag{22}
$$

The turbine-inlet temperature is set by the throttle demand and scaled by its latent modifier, and the fuel flow follows from the combustor energy balance:

$$
T _ { 4 1 } = \left( T _ { 4 1 } ^ { 0 } + T _ { 4 1 } ^ { \prime } \widehat { \mathrm { T R A } } \right) m o d _ { T _ { 4 1 } } , \qquad W _ { f } = W _ { \mathrm { c o m b } } \frac { c _ { p , t } T _ { 4 1 } - c _ { p } T _ { 3 1 } } { \eta _ { \mathrm { c o m b } } \mathrm { L H V } - c _ { p , t } T _ { 4 1 } } ,\tag{23}
$$

with $\widehat { \mathrm { T R A } } = \mathrm { T R A } / 1 0 0 \in [ 0 , 1 ]$ . Cooling flows are mixed with the main gas path through the enthalpy balance

$$
T _ { \mathrm { m i x } } = \frac { \sum _ { j } W _ { j } c _ { p , j } T _ { j } } { c _ { p , t } \sum _ { j } W _ { j } } ,\tag{24}
$$

applied at the HPT inlet, where the HPT cooling air rejoins the combustor exhaust, and at the jet pipe, where the LPT cooling air and the customer bleed rejoin the core stream.

For the turbines, the expansion ratio of each stage is itself a latent modifier and the eficiency sets the temperature drop through the isentropic expansion relation:

$$
\Pi _ { t } = \Pi _ { t } ^ { 0 } e ^ { 0 . 0 5 z } , \qquad T _ { \mathrm { o u t } } = T _ { \mathrm { i n } } \left[ 1 - \eta _ { t } \left( 1 - \Pi _ { t } ^ { \frac { 1 - \gamma _ { t } } { \gamma _ { t } } } \right) \right] , \qquad P _ { \mathrm { o u t } } = \frac { P _ { \mathrm { i n } } } { \Pi _ { t } } .\tag{25}
$$

The combustor exit pressure follows Eq. 20 and, downstream of the HPT, temperature and pressure are identified by $T _ { 4 4 } , P _ { 4 4 }$ , with then $T _ { 4 6 } = T _ { 4 4 }$ and $P _ { 4 6 } = ( 1 - \delta _ { \mathrm { i t d } } ) P _ { 4 4 }$ at the LPT inlet after the inter-turbine duct. Applied to the low-pressure turbine, Equation 25 gives the LPT exit state $( T _ { \mathrm { L P T , e x } } , P _ { 4 8 } ) \mathrm { : }$ the channel T50 is the mixed exit temperature $T _ { 5 }$ of Equation 24, formed once the LPT cooling air and the customer bleed have rejoined the core stream, and the channel P50 is the jet-pipe pressure $P _ { 7 }$ obtained from $P _ { 4 8 }$ through Equation 16. The jet-pipe temperature follows from Eq. (24) applied to the LPT cooling air and the customer bleed rejoining the core. The shaft power balance becomes a soft constraint. The required powers are

$$
\begin{array} { r } { \dot { Q } _ { \mathrm { f a n } } = W _ { 0 } c _ { p } ( T _ { 1 3 } - T _ { 2 } ) , \quad \dot { Q } _ { \mathrm { L P C } } = W _ { \mathrm { c o r e } } c _ { p } ( T _ { 2 7 } - T _ { 2 6 } ) , \quad \dot { Q } _ { \mathrm { H P C } } = W _ { 3 } c _ { p } ( T _ { 3 } - T _ { 2 7 } ) , } \end{array}\tag{26}
$$

and the relative imbalance on each shaft is

$$
r _ { \mathrm { H P } } = \frac { \eta _ { m } W _ { 4 1 S } c _ { p , t } \left( T _ { 4 1 S } - T _ { 4 4 } \right) - \dot { Q } _ { \mathrm { H P C } } } { | \dot { Q } _ { \mathrm { H P C } } | } , \qquad r _ { \mathrm { L P } } = \frac { \eta _ { m } W _ { 4 6 } c _ { p , t } \left( T _ { 4 6 } - T _ { \mathrm { L P T , e x } } \right) - \left( \dot { Q } _ { \mathrm { f a n } } + \dot { Q } _ { \mathrm { L P C } } \right) } { | \dot { Q } _ { \mathrm { f a n } } + \dot { Q } _ { \mathrm { L P C } } | }\tag{27}
$$

The equations are evaluated sequentially along the inlet, fan, bypass duct, LPC, HPC, combustor, HPT, LPT, and exhaust paths. The physical-decoder output is assembled as

$$
\widehat { \mathbf { x } } _ { \mathrm { p h y s } } ( t ) = \left[ T _ { 2 4 } \quad T _ { 3 0 } \quad T _ { 4 8 } \quad T _ { 5 0 } \quad P _ { 1 5 } \quad P _ { 2 } \quad P _ { 2 1 } \quad P _ { 2 4 } \quad P _ { s 3 0 } \quad P _ { 4 0 } \quad P _ { 5 0 } \quad W _ { f } \right] ^ { \top } .\tag{28}
$$

corresponding to the selected N-CMAPSS variables. The outputs are finally converted to the N-CMAPSS measurement units and normalized using the bounds adopted for model training, giving �̂(�). As already mentioned, the shaft-speed variables $N _ { f }$ and $N _ { c }$ were excluded because the present decoder does not explicitly model spool-speed dynamics. The network is trained on the reconstruction error with the two power residuals as penalty terms, thus representing a form of learning bias via the residual term

$$
\mathcal { L } = \frac { 1 } { N } \sum _ { t } \Vert \hat { \mathbf { x } } ( t ) - \mathbf { x } ( t ) \Vert ^ { 2 } + \lambda \frac { 1 } { N } \sum _ { t } \left( r _ { \mathrm { H P } } ^ { 2 } ( t ) + r _ { \mathrm { L P } } ^ { 2 } ( t ) \right) , \qquad \lambda = 1 ,\tag{29}
$$

so that the shaft balance is satisfied where it is compatible with the measurements instead of being imposed regardless of them. At the trained solution the constraint is nonetheless close to being met: over the 425,156 test samples the mean relative imbalance is around 0.111% on the HP shaft and around 0.119% on the LP shaft, the 95th percentile is below 0.35%. This small residual allows the two turbine eficiencies to vary independently within the decoder while maintaining approximate shaft balance. Their separate recovery from the measurements remains subject to the identifiability analysis in Section 5.5.

## 4.2.2. Latent-space sensitivity and dimensionality selection

As reported in Eq. 13, the physical decoder initially defines a 16-dimensional latent space, with each coordinate associated with one of the physical quantities entering the model. However, the availability of twelve measured variables does not guarantee that all 16 latent quantities are observable from the reconstruction problem. Sensitivity measures the influence of a coordinate on the outputs. The sensitivity of the reconstructed measurements to each latent coordinate was therefore evaluated through the Jacobian ${ \partial \hat { \bf x } } / { \partial z _ { k } }$ . The sensitivity analysis hence guides the selection of the latent coordinates, with methodological details provided in Appendix D.

In particular, one coordinate has a markedly lower sensitivity than the remaining latent variables. The Jacobian norm associated with $\beta _ { \mathrm { c o m b } }$ is in fact $3 . 7 7 \times 1 0 ^ { - 5 }$ . For comparison, the lowest value among the retained coordinates is $1 . 8 0 \times 1 0 ^ { - 3 }$ , corresponding to $\beta _ { \mathrm { L P T } }$ . The combustor cooling fraction $\beta _ { \mathrm { c o m b } }$ βeoml is consequently fixed at its design value, reducing the latent space from 16 to 15 dimensions. The complete sensitivity comparison and the ablation obtained by reintroducing this coordinate are reported in Appendix D.

The final latent representation adopted in the proposed model is therefore

$$
\mathbf { p } ( t ) = \{ \eta _ { \mathrm { f a n } } , \eta _ { \mathrm { I , P C } } , \eta _ { \mathrm { H P C } } , \eta _ { \mathrm { H P T } } , \eta _ { \mathrm { I , P T } } , \Pi _ { \mathrm { f a n } } , \Pi _ { \mathrm { L P C } } , \Pi _ { \mathrm { H P C } } , \Pi _ { \mathrm { H P T } } , \Pi _ { \mathrm { L P T } } , W _ { \mathrm { m o d } } , \mathbf { B } \mathbf { P } \mathbf { R } , T _ { 4 1 } , \beta _ { \mathrm { H P T } } , \beta _ { \mathrm { I , P T } } \} ,
$$

comprising five component eficiencies, three compressor pressure ratios, two turbine expansion ratios, the inlet flow capacity, the bypass ratio, the turbine-inlet temperature, and two turbine cooling fractions. An overview of the resulting architecture, including the measured inputs and the selected 15 latent variables, is provided in Figure 2.

With 12 measured variables and 15 latent coordinates, the resulting architecture is formally over-complete. In this case, however, the decoder contains no trainable parameters and the latent variables act exclusively through the fixed thermodynamic relations. The wider latent space therefore cannot be exploited by the decoder to learn an unconstrained identity mapping.

## 4.3. Model Implementation

The model was implemented in Python using TensorFlow 2.10 and Keras with 32-bit floating-point precision. The measured variables were min–max normalized using bounds estimated from the training set, while the operating conditions were provided to the decoder in their original units. The encoder comprises five fully connected layers with 256 neurons and ReLU activations. The decoder is a fixed diferentiable computation graph implementing the thermodynamic relations introduced in Section 4.2.1. It receives the latent variables together with altitude, Mach number, throttle-resolver angle, and fan-inlet total temperature, and contains no trainable parameters. All experiments were executed on identically configured nodes of a SLURM cluster equipped with two Intel Xeon E5-2698 v4 CPUs at 2.20 GHz, 450 GB of RAM, and eight NVIDIA Tesla V100-SXM2 GPUs with 32 GB of memory. Each job was allocated one GPU, four CPU cores, and 32 GB of RAM. The median training time was 9.4 min per model on a single V100 GPU over five runs, ranging from 8.9 to 11.5 min due to early stopping. Each model was trained with the Adam optimizer, a learning rate of $\bar { 1 0 } ^ { - \bar { 3 } }$ , a batch size of 512, gradient-norm clipping at 1.0, and an epoch cap of 120 with early stopping on the validation loss (patience 20, best weights restored). The cap is never reached: across the five seeds training stops between epoch 26 and 52 and the best validation loss is reached between epoch 6 and 32.

## 5. Results

The analysis begins with the N-CMAPSS DS03 test set, comprising six engines, 438 flight cycles, and 425,156 samples. Figure 4 summarizes how the trained autoencoder is used to reconstruct the twelve sensor variables and extract the latent health representation. Reconstruction performance is assessed using $R ^ { 2 }$ and Mean Absolute Error (MAE). The latent coordinates are then corrected for operating-condition efects and combined into a one-dimensional HI, whose agreement with the ground truth is evaluated using Pearson and Spearman correlations, MAE, and Root Mean Square Error (RMSE). These results are followed by comparisons with alternative representations and validation of the inferred physical trends. Section 5.5 examines latent identifiability and the contribution of an additional pressure measurement. The analysis then extends to a single model trained on pooled DS03–DS06 data to assess the diagnostic information retained across four degradation scenarios in Section 5.6. Finally, Section 5.7 examines the inferred turbine evolution in component-map coordinates.

![](images/a4c5512bd0b7ab1d952f7a34496a36c75865885029991decce133ba746dff100.jpg)  
Figure 4: Inference and health-assessment workflow of the proposed framework. The inferred physical latent variables are used to investigate the latent space and construct the system-level HI after removal of the operating-condition contribution. In parallel, the inferred turbine quantities are used to evaluate degradation-induced changes in the component operating maps.

## 5.1. Reconstruction Performance of the Physics-Constrained Autoencoder

The proposed physics-constrained autoencoder achieves an overall reconstruction coeficient of determination of $R ^ { 2 } = 0 . 9 9 9 3$ and a MAE of 0.0036 on the test set (engine 10, 11, 12, 13, 14, 15). Figure 5 compares the measured and reconstructed trajectories for the 12 sensor variables over the test interval. The reconstructed signals follow the measured response across the diferent operating regimes and reproduce both the main trends and the faster variations observed within the flight profiles. The corresponding per-variable values of $R ^ { 2 }$ and MAE are reported in Table 3. The obtained reconstruction accuracy indicates that the thermodynamic computation graph can reproduce the observed engine response with only a limited error. Some discrepancies remain concentrated in specific sensor variables (i.e., $P _ { 2 1 } , T _ { 5 0 } )$ , suggesting that the corresponding thermodynamic relations or component representations may require further refinement or the lack of enough observability. Nevertheless no variable exceeds a mean MAE of 0.0075, showing the consistent reconstruction ability of the architecture in all channels.

The reconstruction results provide the basis for examining the inferred physical variables. An accurate reconstruction alone, however, does not demonstrate that the latent trajectories contain a suitable degradation indicator. In fact, as shown by Wen et al. [37], in an asset such as a turbofan engine, the sensor variability is largely driven by the flight conditions, and the same contribution remains visible in the latent space before operating-condition correction.

## 5.2. Removal of the Operating-Condition Contribution and 1-D HI.

Variable flight conditions strongly afect aero-engine measurements and the latent quantities inferred from them. Operating-condition mappings and residual representations have therefore been used to separate normal operating variability from degradation information [37]. Following the procedure introduced in Section 3.3, we estimate this contribution using a third-order polynomial ridge model fitted to the healthy portion of each engine trajectory.

![](images/2726f756c96b49bcd9fe60fcd3db0d163196e7dd7ad428dbc928a8b6e6e9883b.jpg)  
Figure 5: Measured and reconstructed trajectories for the 12 sensor variables obtained with the proposed physics-constrained autoencoder. The reported $R ^ { 2 }$ values are computed over all test samples.

Per-variable reconstruction performance of the proposed physics-constrained autoencoder on the test set. Results are reported as mean ± half-range.
<table><tr><td>Variable</td><td> $R ^ { 2 }$ </td><td>MAE</td><td>Variable</td><td> $R ^ { 2 }$ </td><td>MAE</td></tr><tr><td> $T _ { 2 4 }$ </td><td> $0 . 9 9 9 2 0 \pm 0 . 0 0 0 1 8$ </td><td> $0 . 0 0 3 1 0 \pm 0 . 0 0 0 4 4$ </td><td> $P _ { 2 1 }$ </td><td> $0 . 9 9 8 4 2 \pm 0 . 0 0 0 0 8$ </td><td> $0 . 0 0 7 5 0 \pm 0 . 0 0 0 1 5$ </td></tr><tr><td> $T _ { 3 0 }$ </td><td> $0 . 9 9 8 9 7 \pm 0 . 0 0 0 3 9$ </td><td> $0 . 0 0 3 7 1 \pm 0 . 0 0 0 6 9$ </td><td> $P _ { 2 4 }$ </td><td> $0 . 9 9 9 7 6 \pm 0 . 0 0 0 0 6$ </td><td> $0 . 0 0 2 0 8 \pm 0 . 0 0 0 2 6$ </td></tr><tr><td> $T _ { 4 8 }$ </td><td> $0 . 9 9 8 8 4 \pm 0 . 0 0 0 1 7$ </td><td> $0 . 0 0 3 0 6 \pm 0 . 0 0 0 2 2$ </td><td> $P _ { s 3 0 }$ </td><td> $0 . 9 9 9 3 8 \pm 0 . 0 0 0 2 2$ </td><td> $0 . 0 0 3 1 1 \pm 0 . 0 0 0 6 5$ </td></tr><tr><td> $T _ { 5 0 }$ </td><td> $0 . 9 9 7 8 5 \pm 0 . 0 0 0 9 5$ </td><td> $0 . 0 0 3 2 3 \pm 0 . 0 0 0 7 1$ </td><td> $P _ { 4 0 }$ </td><td> $0 . 9 9 9 6 1 \pm 0 . 0 0 0 1 2$ </td><td> $0 . 0 0 2 4 1 \pm 0 . 0 0 0 3 1$ </td></tr><tr><td> $P _ { 1 5 }$ </td><td> $0 . 9 9 9 0 0 \pm 0 . 0 0 0 0 5$ </td><td> $0 . 0 0 5 9 8 \pm 0 . 0 0 0 1 5$ </td><td> $P _ { 5 0 }$ </td><td> $0 . 9 9 9 8 7 \pm 0 . 0 0 0 0 7$ </td><td> $0 . 0 0 1 8 9 \pm 0 . 0 0 0 4 7$ </td></tr><tr><td> $P _ { 2 }$ </td><td> $0 . 9 9 9 8 6 \pm 0 . 0 0 0 0 0$ </td><td> $0 . 0 0 2 7 7 \pm 0 . 0 0 0 0 0$ </td><td> $W _ { f }$ </td><td> $0 . 9 9 8 3 8 \pm 0 . 0 0 0 3 7$ </td><td> $0 . 0 0 3 9 5 \pm 0 . 0 0 0 6 5$ </td></tr><tr><td>Overall</td><td></td><td></td><td></td><td> $\mathbf { 0 . 9 9 9 2 7 \pm 0 . 0 0 0 0 6 }$ </td><td> $\mathbf { 0 . 0 0 3 5 7 \pm 0 . 0 0 0 1 5 }$ </td></tr></table>

The reference interval is defined as the first 15% of the complete trajectory. Degradation-onset flags are used only retrospectively to verify that these intervals contain no degraded samples. The healthy interval also supplies the standard deviation used to scale each latent residual. For the HI benchmarks, coordinate orientation, selection, and fusion weights are determined on development trajectories and applied unchanged to the test engines.

The resulting correction substantially improves health tracking. Without correction, the HI retains a strong flightcondition contribution, with a Spearman correlation of $0 . 6 6 8 4 \pm 0 . 1 3 4 5$ against the ground truth and an RMSE of $0 . 1 8 3 3 \pm 0 . 0 8 4 5$ . After correction, the correlation increases to $0 . 9 7 9 1 \pm 0 . 0 0 5 9$ and the RMSE decreases to $0 . 0 2 8 6 \pm 0 . 0 0 3 3$ , approximately a sixfold reduction. Figure 6 illustrates the corresponding changes in the latent trajectories for a representative test engine, unit 14. After correction, the coordinates associated with turbine-inlet temperature, HPT eficiency, and LPT eficiency reach absolute correlations of 0.93–0.96 with the engine ground-truth HI, compared with a maximum of 0.35 before correction. The emergence of the two turbine-eficiency coordinates among the strongest degradation-related signals is consistent with the eficiency deterioration imposed in DS03.

To further test the proposed methodology, table 4 compares the correction with alternative regressors over five runs per method. Third-order polynomial ridge achieves the highest mean HI composite score, 0.8555, and Spearman correlation among the evaluated methods, while second-order polynomial ridge yields the lowest HI RMSE. It also explains slightly more of the healthy operating-condition dependence than the second-order polynomial model, with a W-fit $R ^ { 2 }$ of 0.9991 compared with 0.9980. Regressor selection additionally considered the need to evaluate the healthy baseline under flight conditions beyond those represented in the reference interval. Random Forest and k-NN were excluded because their predictions outside the training domain remain tied to training responses. The polynomial formulation provides an explicit mapping beyond that domain, although its extrapolation accuracy depends on the operating-condition coverage of the healthy reference data.

![](images/37e98494385520b3671084d6501047d199cf291b432ee8e845f62c6bd6a94968.jpg)  
before W removal — after W removal -—- GT HI (right axis)

Figure 6: Latent trajectories before and after operating-condition removal for Unit 14. Grey lines indicate the original latent variables, blue lines their corrected residuals, and the green dashed line the engine ground-truth HI. The panel titles report the correlation before and after the correction.  
![](images/d39234fb85359ab7587a2052cd519c2b7867463db6712ac9d3036ccbce04ae64.jpg)  
individual engines, all seeds (n = 30) total rangerange across seed meanspooled mean  
Figure 7: Cross-engine evolution of the inferred physical modifiers against normalised life fraction, over the six DS03 test engines of five independently seeded trainings (30 trajectories). Grey curves are the individual traces, the pale band their total range, the darker band the range across the five per-seed means, and the heavy curve the pooled mean. Shading marks the two coordinates degraded in DS03.

Finally, the consistency of the inferred physical trends across the six test engines is examined in Figure 7. To account for the diferent run lengths of the engines, the trajectories are expressed against normalized life fraction. For each latent coordinate �, cross-engine agreement is quantified through the mean leave-one-out Pearson correlation. The leave-one-out formulation prevents each engine from contributing to the reference trajectory against which it is compared.

The strongest agreement is obtained for SOT, $\eta _ { \mathrm { H P T } } , \eta _ { \mathrm { L P C } } .$ , and $\eta _ { \mathrm { L P T } }$ . The component quantities supporting the main interpretation therefore exhibit highly repeatable trends across the test fleet, while some secondary latent coordinates remain less coherent $( { \bf e } . { \bf g } . , B P R , \beta _ { H P T } )$

Comparison of the methods used to remove the operating-condition contribution from the latent representation. All operating-condition models are fitted using healthy samples only. For each method, four coordinates are selected and weighted using development trajectories, and the resulting HI is evaluated on the test engines. Fit times are retained from the original measurements of the unchanged fitting procedure. Each method was evaluated over five independent runs using diferent random seeds; results are reported as mean ± half-range over five seeds. The best and second-best results in each column are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>Extrapolation</td><td> $W / / \dot { \bf \mu } \textbf { } R ^ { 2 }$ </td><td>Spearman</td><td>HI RMSE</td><td>Composite</td><td>Fit time</td></tr><tr><td>No removal</td><td></td><td></td><td> $0 . 6 6 8 4 \pm 0 . 1 3 4 5$ </td><td> $0 . 1 8 3 3 \pm 0 . 0 8 4 5$ </td><td> $0 . 5 6 0 1 \pm 0 . 1 2 2 0$ </td><td></td></tr><tr><td>Linear ridge</td><td>Yes</td><td> $0 . 9 5 5 1 \pm 0 . 0 3 3 5$ </td><td> $0 . 9 4 0 9 \pm 0 . 0 2 1 9$ </td><td> $0 . 0 3 1 0 \pm 0 . 0 0 5 4$ </td><td> $0 . 8 0 0 1 \pm 0 . 0 1 4 1$ </td><td>0.8 s</td></tr><tr><td>Degree-2 polynomial ridge</td><td>Yes</td><td> $0 . 9 9 8 0 \pm 0 . 0 0 0 9$ </td><td> $0 . 9 7 7 5 \pm 0 . 0 0 2 9$ </td><td> $\mathbf { 0 . 0 2 6 5 \pm 0 . 0 0 3 6 }$ </td><td> $0 . 8 4 3 6 \pm 0 . 0 1 0 6$ </td><td>0.9 s</td></tr><tr><td>Degree-3 polynomial ridge</td><td>Yes</td><td> $\mathbf { 0 . 9 9 9 1 } \pm \mathbf { 0 . 0 0 0 3 }$ </td><td> $\mathbf { 0 . 9 7 9 1 \pm 0 . 0 0 5 9 }$ </td><td> $0 . 0 2 8 6 \pm 0 . 0 0 3 3$ </td><td> $\mathbf { 0 . 8 5 5 5 \pm 0 . 0 1 3 3 }$ </td><td> $1 . 1 \pm 0 . 1 \mathrm { ~ s ~ }$ </td></tr><tr><td>MLP regressor</td><td>Yes</td><td> $\underline { { 0 . 9 9 8 3 } } \pm 0 . 0 0 0 6$ </td><td> $0 . 9 6 0 9 \pm 0 . 0 0 3 5$ </td><td> $\underline { { 0 . 0 2 7 1 \pm 0 . 0 0 4 1 } }$ </td><td> $0 . 8 3 5 8 \pm 0 . 0 1 9 1$ </td><td> $8 5 . 2 \pm 9 . 7 \mathrm { ~ s ~ }$ </td></tr></table>

After removing the operating-condition contribution, the 15 latent residuals are aggregated at cycle level for each development and test unit and smoothed along cycles with a Savitzky–Golay filter. Finally they are evaluated individually in terms of monotonicity, trendability, prognosability and mutual information with RUL. In fact, given the unsupervised nature of the study, literature indicates this as the most reliable way of assessing unsupervised HI learning [29]. For coordinate selection, these criteria are computed on development trajectories without using the ground-truth HI and are combined into the composite score: $\begin{array} { r } { S _ { j } = \frac { 1 } { 4 } \left( M _ { j } + T _ { j } + I _ { j } + P _ { j } \right) } \end{array}$ , where $M _ { j } , T _ { j } , I _ { j }$ and $P _ { j }$ denote, respectively, the monotonicity, trendability, normalized mutual-information, and prognosability scores of the �-th residual latent variable. The dimensions are ranked according to $S _ { j }$ , and the first � variables are retained. The final HI is obtained as their criteria-weighted combination,

$$
\mathrm { H I } ( c ) = \frac { \displaystyle \sum _ { j \in { \cal K } } S _ { j } \widetilde { z } _ { j } ( c ) } { \displaystyle \sum _ { j \in { \cal K } } S _ { j } } , \qquad { \cal K } = \left\{ j _ { 1 } , \dots , j _ { k } \right\} ,\tag{30}
$$

where $\widetilde { z } _ { j } ( c )$ denotes the corrected, healthy-standard-deviation-scaled, smoothed and cycle-aggregated value of the �-th latent variable at cycle �, oriented using the sign determined on development trajectories. To determine how many coordinates to retain, the top-� coordinates are fused for $k = 1 , \ldots$ , 10 and the resulting HI is evaluated on the development set. Averaged over five training seeds, the composite score is 0.8452 with four coordinates and 0.8516 with five. We retain four coordinates for a more compact representation, accepting the small reduction in score. The coordinate ranking, orientation, and fusion weights determined on development data are then kept fixed when constructing the HIs of the test engines.

Notably, consistent with the degradation scenario, three quantities are retained in every seed: the turbine-inlet temperature level $T _ { 4 1 }$ , the LPT eficiency $\eta _ { \mathrm { L P T } } .$ , and the HPT eficiency $\eta _ { \mathrm { H P T } } ;$ the fourth coordinate varies across seeds. This selection is based on the evolution of the latent coordinates with cycle index and remaining useful life, without using information about the degraded components. The emergence of the two turbine eficiencies as the highest-ranked component-eficiency quantities is therefore consistent with the HPT- and LPT-eficiency degradation imposed in DS03. Their selection supports the ability of the physical latent representation to associate degradationsensitive trends with the afected engine section. The high ranking of $T _ { 4 1 }$ captures a complementary aspect of engine degradation. Its increase can be traced back to the additional fuel and higher turbine-inlet temperature required to maintain thrust as component performance deteriorates. This coordinate therefore describes the engine’s operating response to degradation, while the turbine-eficiency coordinates are directly associated with the imposed deterioration mechanisms.

The resulting HI reaches a monotonicity of 0.627, a trendability of 0.988, a normalized mutual-information score of 0.807, and a composite score of 0.856. When compared to the ground truth, its agreement with the ground-truth HI is characterized by a Pearson correlation of 0.996, a Spearman correlation of 0.979, an RMSE of 0.029, and an MAE of 0.023. Table 5 summarizes the corresponding metrics.

Health-index construction and test performance for the proposed physics-constrained representation, with coordinate orientation, selection, and fusion weights determined on development trajectories. Results are reported as mean ± half-range over five seeds.
<table><tr><td>Quantity</td><td>Value</td><td>Quantity</td><td>Value</td></tr><tr><td>Monotonicity</td><td> $0 . 6 2 7 2 \pm 0 . 0 5 0 3$ </td><td>Trendability</td><td> $0 . 9 8 7 9 \pm 0 . 0 0 6 2$ </td></tr><tr><td>Prognosability</td><td> $0 . 9 9 9 8 \pm 0 . 0 0 0 5$ </td><td>Mutual information</td><td> $0 . 8 0 7 2 \pm 0 . 0 1 3 4$ </td></tr><tr><td>Composite score</td><td> $0 . 8 5 5 5 \pm 0 . 0 1 3 3$ </td><td>Spearman with GT</td><td> $0 . 9 7 9 1 \pm 0 . 0 0 5 9$ </td></tr><tr><td>Pearson with GT</td><td> $0 . 9 9 5 8 \pm 0 . 0 0 1 6$ </td><td>MAE</td><td> $0 . 0 2 3 3 \pm 0 . 0 0 2 8$ </td></tr><tr><td>RMSE</td><td> $0 . 0 2 8 6 \pm 0 . 0 0 3 3$ </td><td></td><td></td></tr></table>

![](images/79db8349a4579b42b28dd4b572ffb38baf5ee75af36dafc42c4fb3f4c8fd96e4.jpg)  
Figure 8: Overview of the benchmarking framework used to compare the proposed physics-constrained representation with the considered data-driven and reduced-order baselines.

## 5.3. Benchmarking with Alternative Representations

The proposed representation is compared with other possible HI learning techniques starting from simple linear strategies all the way to full data-driven autoencoders. In particular the diferent strategies, reported in Figure 8, are: PCA representation with 1 dimension, a fully data-driven autoencoder with 15 latent dimensions, a fully data-driven autoencoder with 1 latent dimensions, a reduced-order variant of the physics-constrained autoencoder with one neuron in the latent space, and a supervised feed-forward regressor. It has to be noted that the downstream processing is identical for every unsupervised representation. Further details on the benchmarking details are provided in Appendix B. The declared objective is not to show that the proposed method is performing better than other baselines but that the proposed solution is able to perform reconstruction and HI learning in a competitive way with the added bonus of interpretability.

Reconstruction performance. Figure 9 reports the channel-wise reconstruction performance, while Table 6 summarizes the results over the complete test set. The 15-dimensional data-driven autoencoder obtains $R ^ { 2 } \ : = \ : 0 . 9 9 9 3$ and an MAE of 0.0021, and an identical performance is obtained with a single latent neuron, showing how much of the reconstruction the unconstrained architecture recovers from one coordinate. The proposed physics-constrained autoencoder reaches $R ^ { 2 } = 0 . 9 9 9 3$ with an MAE of 0.0036: constraining every latent coordinate to act through fixed thermodynamic relations therefore costs no measurable reconstruction accuracy at the level of $R ^ { 2 }$ , and the residual diference in MAE is the small price of the equation-based decoder. The efect of further constraining the representation is instead more pronounced: reducing the encoder bottleneck to a single dimension decreases the overall reconstruction $R ^ { 2 }$ to 0.9481 and increases the MAE to 0.0275, which indicates that the information required by the physical decoder cannot be compressed into a single latent coordinate and supports the use of a multidimensional physical representation.

Health-index performance. A diferent picture emerges when the representations are evaluated according to the quality of the resulting HI. The proposed physics-constrained representation achieves the highest mean Spearman correlation, 0.9791, and composite score, 0.8555. Its HI RMSE is 0.0286, compared with 0.0305 for the 15-D datadriven AE, whose Spearman correlation and composite score are 0.9525 and 0.8392, respectively. PCA achieves the same HI RMSE as the proposed representation at the reported precision. It can be seen how the variation between seeds for the one dimensional data-driven architecture is very high compared to the other architectures, underscoring the instability of that solution. The other data-driven architectures still show greater variability than the proposed version. The PCA one-dimensional representation has a relatively poor reconstruction performance $( R ^ { 2 } = 0 . 8 3 1 1 )$ yet still reaches a Spearman correlation of 0.9643.

Table 6  
![](images/111ecf92b86845f72c136fdba1a591a8b33e67f708c7e67158a9500bfc3dccb6.jpg)  
(a) �<sup>2</sup>

![](images/41a0dd49a44aab4ccaecad84dd6ec324378fb03fae52071351b4e3a0afeb1c94.jpg)  
(b) MAE  
Figure 9: Channel-wise reconstruction performance obtained by the considered representations over all test samples, reported in terms of (a) coeficient of determination $R ^ { 2 }$ and (b) mean absolute error (MAE).

Comparison of the considered representations on the N-CMAPSS DS03 test set. Reconstruction metrics are computed over all test samples, while HI metrics are computed over the 438 flight cycles. The column � indicates the number of latent coordinates retained for HI construction. For one-dimensional representations, $k = 1$ by construction; � is not applicable to the supervised baseline. For all unsupervised representations, coordinate orientation and, where applicable, coordinate selection and fusion weights are determined on development trajectories and applied unchanged to the test engines. Results with spread are reported as mean ± half-range over five seeds. The best and second-best results in each column are shown in bold and underlined, respectively.
<table><tr><td>Representation</td><td>k</td><td>Recon.  $R ^ { 2 }$ </td><td>Recon. MAE</td><td>Spearman</td><td>HI RMSE</td><td>Composite</td></tr><tr><td>PCA, 1 component</td><td>1</td><td>0.83108</td><td>0.05527</td><td>0.9643</td><td>0.0286</td><td>0.8266</td></tr><tr><td>Data-driven AE, 1-D</td><td>1</td><td> $\mathbf { 0 . 9 9 9 2 8 \pm 0 . 0 0 0 0 0 }$ </td><td> $\mathbf { 0 . 0 0 2 0 8 \pm 0 . 0 0 0 0 1 }$ </td><td> $0 . 7 5 4 5 \pm 0 . 3 6 6 4$ </td><td> $0 . 1 3 0 0 \pm 0 . 1 6 9 1$ </td><td> $0 . 6 7 5 8 \pm 0 . 2 2 9 1$ </td></tr><tr><td>Data-driven AE, 15-D</td><td>4</td><td> $\mathbf { 0 . 9 9 9 2 8 \pm 0 . 0 0 0 0 0 }$ </td><td> $\mathbf { 0 . 0 0 2 0 8 \pm 0 . 0 0 0 0 1 }$ </td><td> $0 . 9 5 2 5 \pm 0 . 0 3 8 9$ </td><td> $0 . 0 3 0 5 \pm 0 . 0 1 7 2$ </td><td> $0 . 8 3 9 2 \pm 0 . 0 5 9 9$ </td></tr><tr><td>Physics AE, bottleneck 1</td><td>1</td><td> $0 . 9 4 8 1 1 \pm 0 . 0 0 0 7 4$ </td><td> $0 . 0 2 7 4 5 \pm 0 . 0 0 0 2 8$ </td><td> $0 . 9 4 2 8 \pm 0 . 0 1 7 0$ </td><td> $0 . 0 3 9 8 \pm 0 . 0 0 7 7$ </td><td> $\overline { { 0 . 8 1 4 8 \pm 0 . 0 1 4 7 } }$ </td></tr><tr><td>Physics AE, 15-D (proposed)</td><td>4</td><td> $\underline { { 0 . 9 9 9 2 7 \pm 0 . 0 0 0 0 6 } }$ </td><td> $0 . 0 0 3 5 7 \pm 0 . 0 0 0 1 5$ </td><td> $\mathbf { 0 . 9 7 9 1 } \pm \mathbf { 0 . 0 0 5 9 }$ </td><td> $\mathbf { 0 . 0 2 8 6 \pm 0 . 0 0 3 3 }$ </td><td> $\mathbf { 0 . 8 5 5 5 \pm 0 . 0 1 3 3 }$ </td></tr><tr><td>Supervised baseline†</td><td></td><td>一</td><td>一</td><td>0.8530</td><td>0.0820</td><td>0.6114</td></tr></table>

<sup>†</sup> The supervised baseline uses leave-one-unit-out training and evaluation within the designated test fleet and is not directly comparable with the development-trained representations. PCA and the supervised baseline are deterministic; therefore, no spread is reported.

The reduced-order physics-constrained architecture also confirms the importance of retaining the component-level latent structure. Compressing the physics representation to a single coordinate reduces the Spearman correlation from 0.9791 to 0.9428 and the composite score from 0.8555 to 0.8148. The performance of the proposed model therefore results from the combination of the physical decoder and the structured multidimensional latent representation, rather than from the decoder alone. Figure 10 further compares the resulting HI trajectories with the ground truth for the six test engines. The supervised baseline reaches a Spearman correlation of 0.8530 and an RMSE of 0.0820 and produces less regular trajectories despite being trained directly on the ground-truth HI. Since this model uses labelled degradation information, it is reported as a reference and is not directly comparable with the unsupervised representations.

In conclusion, the benchmark therefore shows a comparable reconstruction ability with respect to fully data-driven counterparts, comparable HI accuracy compared with the strongest linear baselines, but accompanied by a substantial increase in physical interpretability.

## 5.4. Latent Space Variables Validation

The directions reported in Figure 7 are consistent with the degradation mechanisms defined for N-CMAPSS DS03, which combines HPT-eficiency degradation with LPT-eficiency and LPT-flow-capacity degradation. The corresponding eficiency coordinates exhibit clear and repeatable trends across the test engines, providing a first indication that the physical structure imposed by the decoder is reflected in the inferred latent representation. A direct ground truth — ae\_datadriven\_1 — physics\_15d (proposed) — pca\_1 — physics 15d 1 supervised baseline

![](images/af8125ad129fa0610c34943dda36d0497a7779e4d82bd85c29ea51470a3cbb32.jpg)

Figure 10: Estimated HIs and ground-truth HI for the six test engines. The grey regions identify the healthy reference intervals, while the dotted lines indicate the degradation-onset cycles. Almost all methodologies show a similar trend, except for the supervised model in yellow and the data-driven autoencoder with 1 dimension in the latent space. This justifies the goodness of the physics decoder while providing an explicitly physics-structured latent representation.  
![](images/e0cf67fe1db886b33067333f392b93da850e624334ff95c96337ff049c940cfb.jpg)  
inferred, fleet meanindividual engines imposed (zero unless shown)  
Figure 11: Relative changes in the five inferred eficiency-related coordinates for DS03. Thin blue curves show individual test engines, and the thick blue curve shows their mean. Dashed red curves show the imposed HPT- and LPT-eficiency degradation. No degradation is imposed on the fan, LPC, or HPC.

assessment is possible using the model health parameters provided by N-CMAPSS in �, which describe the degradation imposed during data generation.

Figure 11 shows that the inferred HPT- and LPT-eficiency trajectories follow the same downward evolution as the imposed degradation, reproducing both its gradual initial progression and its sharper decline towards end of life. This agreement is obtained without using the reference health parameters to train the autoencoder. Across all test engines and five training seeds, the mean Spearman correlations are 0.97 for $\eta _ { \mathrm { H P T } }$ and 0.95 for $\eta _ { \mathrm { L P T } } .$ , with minimum values of 0.91 and 0.86, respectively. Together with the repeatability shown in Figure 7, these results demonstrate that the coordinates assigned to the two turbine eficiencies retain the temporal signature of the corresponding component degradation. Quantitative interpretation of the inferred magnitudes is examined in Section 5.5.

The comparison across all five eficiency coordinates also shows that the largest changes occur in the HPT and LPT, where eficiency degradation is imposed. The fan, LPC, and HPC coordinates exhibit smaller changes as the decoder reconstructs the response of the engine to turbine degradation. Their evolution may combine changes in component operating states with compensation among latent quantities. To examine whether these secondary trends are physically consistent with the broader engine response, the same degradation scenario is reproduced using GasTurb and TurboGrad, two independent gas-turbine performance simulation tools, in Appendix C. This comparison complements the direct validation of the turbine-eficiency trends, while the coupling among latent quantities is examined in Section 5.5.

## 5.5. Physical Interpretation and Identifiability of the Latent Representation

The physical decoder assigns each latent coordinate a predefined role in the thermodynamic equations. The interpretation of its inferred evolution depends on the degradation mechanisms represented in the parameterization and on the information provided by the measurements. Both aspects must therefore be considered when interpreting latent changes as estimates of individual physical parameters.

Representation of flow-capacity degradation. The N-CMAPSS scenarios considered in this study include degradation in both component eficiency and flow capacity. The decoder represents the five component eficiencies through dedicated latent coordinates, while flow is parameterized through the inlet modifier $W _ { \mathrm { m o d } }$ and the coupled thermodynamic equations. There are no separate modifiers describing changes in the flow capacity of each component. Consequently, the LPT-flow-capacity degradation cannot be recovered through corresponding component-specific coordinates.

These degradation mechanisms nevertheless afect the measured engine response that the autoencoder is trained to reconstruct. To reproduce their efects, the encoder can adjust the available latent quantities, including eficiencies, pressure ratios, and the inlet flow modifier. An inferred change in an eficiency-related coordinate may therefore partly account for a flow-capacity change that is not explicitly parameterized in the decoder, in what it is often called "spillover" or "smearing". This distinction matters when interpreting the latent trajectories, as compensation between variables may influence the magnitude of the inferred eficiency changes.

Identifiability of the latent quantities. The available measurements do not constrain all latent quantities independently. Changes in diferent quantities can produce similar sensor responses, allowing one latent variable to partly compensate for another during reconstruction. To examine this ambiguity, we used the decoder Jacobian, which describes how the reconstructed measurements respond to small changes in each latent coordinate. Singular value decomposition (SVD) of the mean Jacobian was then used to distinguish independent combinations of latent changes that afect the measurements from combinations whose efects cancel to first order. This analysis follows common sensitivity-based approaches to parameter identifiability [40, 41].

For all five independently trained DS03 models, the mean Jacobian has nine nonzero singular values, giving a rank of 9 and a six-dimensional null space. The evaluated mean Jacobian therefore distinguishes nine independent combinations of the fifteen latent coordinates, leaving six combinations unconstrained in this linearized analysis. This does not mean that nine individual physical parameters can be uniquely recovered. The coordinate-level analysis in Appendix D shows that the compressor pressure ratios and HPC eficiency have negligible contributions to the unconstrained directions, while the two cooling fractions are particularly weakly constrained. The remaining quantities exhibit varying degrees of ambiguity. The latent trajectories therefore provide physically structured information on degradation, with component-level interpretation guided by the distinguishability of their efects on the measurements.

Efect ofthe inter-turbine pressure measurement. The inter-turbine pressure $P _ { 4 5 }$ provides information on how the total expansion is divided between the HPT and LPT. In the standard measured sensor set, this pressure is unavailable. However, the exit pressure $P _ { 5 0 }$ constrains the product $\Pi _ { \mathrm { H P T } } \Pi _ { \mathrm { L P T } }$ , while the temperatures $T _ { 4 8 }$ and $T _ { 5 0 }$ help distinguish the individual turbine contributions. An additional pressure measurement between the turbines could therefore improve their separation.

To examine this possibility, the virtual $P _ { 4 5 }$ variable available in N-CMAPSS was added to the DS03 experiment as a thirteenth encoder input and reconstruction target. This experiment assesses the value of an additional measurement beyond the standard sensor configuration. First of all, adding $P _ { 4 5 }$ increases the rank of the mean Jacobian from 9 to 10 across all five independently trained models. The cosine similarity between the Jacobian columns associated with the two expansion ratios decreases from 0.916 to 0.767, indicating that their efects on the measurements become more distinct. The corresponding similarity between the turbine eficiencies remains essentially unchanged, from 0.715 to 0.714. The additional pressure measurement therefore primarily improves separation of the expansion ratios.

Reconstruction performance remains comparable, with $R ^ { 2 } = 0 . 9 9 9 3 2 \pm 0 . 0 0 0 1 0$ for the thirteen-channel model and $R ^ { 2 } = 0 . 9 9 9 2 7 \pm 0 . 0 0 0 0 6$ for the twelve-channel model. With $k = 4$ for both models and coordinate orientation, selection, and fusion weights determined on development trajectories, the HI composite score is $0 . 8 5 5 5 { \scriptstyle \pm 0 . 0 1 3 3 }$ with twelve channels and $0 . 8 4 2 5 \pm 0 . 0 3 3 8$ with the additional $P _ { 4 5 }$ measurement, showing no improvement in the mean system-level HI score. The additional information mainly benefits the physical resolution of the representation. The inferred distribution of degradation between the turbines also changes. The efect on turbine-degradation attribution is quantified through the relative eficiency change,

$$
d _ { X } = 1 0 0 \left( \frac { \langle \eta _ { X } \rangle _ { \mathrm { E O L } } } { \langle \eta _ { X } \rangle _ { \mathrm { H } } } - 1 \right) ,\tag{31}
$$

Reconstruction on the test engines for single-scenario models and the pooled model. The DS03 and pooled entries are reported as mean ± half-range over five seeds; the remaining entries are single runs.
<table><tr><td>Training set</td><td>Test  $\overline { { R ^ { 2 } } }$ </td><td>Test MAE</td><td>Jacobian rank</td><td>HP shaft residual</td></tr><tr><td>DS03 (5 seeds)</td><td> $\overline { { 0 . 9 9 9 2 7 \pm 0 . 0 0 0 0 6 } }$ </td><td> $\overline { { 0 . 0 0 3 5 7 \pm 0 . 0 0 0 1 5 } }$ </td><td>9</td><td> $\overline { { 0 . 0 0 1 1 1 \pm 0 . 0 0 0 2 8 } }$ </td></tr><tr><td>DS04</td><td>0.99930</td><td>0.00339</td><td>9</td><td>0.00116</td></tr><tr><td>DS05</td><td>0.99932</td><td>0.00374</td><td>9</td><td>0.00080</td></tr><tr><td>DS06</td><td>0.99907</td><td>0.00383</td><td>9</td><td>0.00094</td></tr><tr><td>DS03-DS06 pooled (5 seeds)</td><td> $\overline { { 0 . 9 9 9 1 2 \pm 0 . 0 0 0 0 9 } }$ </td><td> $\overline { { 0 . 0 0 3 8 3 \pm 0 . 0 0 0 2 4 } }$ </td><td>9</td><td> $\overline { { 0 . 0 0 0 9 3 \pm 0 . 0 0 0 2 3 } }$ </td></tr></table>

evaluated here for $X \ \in \ \{ \mathrm { H P T } , \mathrm { L P T } \}$ . The healthy value is averaged over the first 15% of each engine trajectory, while the end-of-life value is averaged over its last three flight cycles. The resulting changes are then averaged over the six test engines and five training seeds. The ground-truth eficiency changes are obtained from HPT\_eff\_mod and LPT\_eff\_mod, contained in N-CMAPSS � dataset.

With twelve channels, the mean inferred changes are −1.10% for the HPT and −0.74% for the LPT. Adding $P _ { 4 5 }$ changes these estimates to −0.50% and −1.18%, respectively, compared with ground-truth changes of −0.93% and −1.23%. The ratio of the LPT to HPT change magnitudes, calculated from the mean changes, therefore increases from 0.67 to 2.38, compared with the ground-truth ratio of 1.33. Adding $P _ { 4 5 }$ recovers the correct ordering of the turbine-eficiency decreases across all five seeds and brings the LPT estimate closer to the ground truth, while the HPT decrease remains underestimated.

Overall, the experiment illustrates how sensor information can improve the separation and interpretation of component-related latent changes while preserving reconstruction and system-level health tracking.

## 5.6. The Basis for an Interpretable Diagnosis

The physically structured evolution of the latent variables provides the basis for assessing the degradation region, subject to the interpretation limits established in Section 5.5. Because the coordinates of the proposed representation are associated with specific physical quantities, their relative evolution can in principle indicate not only the presence of degradation, but also the region of the engine in which it develops. In this sense, the purpose of this subsection is not to introduce a new diagnostic methodology for N-CMAPSS. Instead, it is to demonstrate that the physical structure of the learned representation already carries diagnostic information when combined with a deliberately simple postprocessing procedure, and therefore to assess directly the added value of the physics-constrained latent space.

In order to test this, datasets containing diferent degradation modes are required. For this reason, the analysis is extended beyond DS03 by three further N-CMAPSS scenarios:

• DS03: HPT eficiency, LPT eficiency and LPT flow capacity (expansion system);

• DS04: fan eficiency and fan flow capacity;

• DS05: HPC eficiency and HPC flow capacity;

• DS06: LPC eficiency and flow capacity together with HPC eficiency and flow capacity.

These datasets place degradation in four diferent regions of the gas path. To really test if the model is able to distinguish between degradation modes, a single model is trained on a pooled dataset with all degradation modes. Therefore, all four datasets are pooled into a single training set and one model is trained on it, with the same 15- dimensional latent definition introduced in Section 4. The final pooled dataset is constituted by a total of 27 training and 18 test engines. Table 7 shows that the pooled model preserves the reconstruction quality of the individual ones. This is a relevant result in itself: the proposed formulation is not tuned to one degradation scenario, and a single physics-constrained model reproduces the 12 measured channels equally well no matter if it is exposed to one fault mode or to four.

The diagnostic analysis uses the five eficiency-related coordinates, � , � , � , � , and $\eta _ { \mathrm { L P T } } ,$ which follow the sequence of the main components along the gas path. Their degradation-induced changes define physically structured signatures for assessing the afected engine region. As discussed in Section 5.5, these changes may also reflect flow-capacity degradation and coupling among latent quantities. As we will demonstrate, they nevertheless already retain physically structured signatures that support discrimination between compression-side and expansionside degradation.

![](images/f91aab48dac7be90a6e8e870a533e574086a4fc6cefe0f18210bae3f7a32f3da.jpg)  
Figure 12: Degradation signatures of the 18 test engines from the pooled DS03–DS06 model. (a) Normalized changes in the five eficiency-related coordinates, with individual engine profiles and regional means with min–max bands. (b) Score diference $S _ { \mathrm { c o m p } } - S _ { \mathrm { t u r b } } ,$ correctly assigning all engines to the compression or expansion side. (c) PCA projection of the same signatures, with labels added after projection. Colour denotes the degradation region and marker shape denotes the source dataset.

For each engine, a five-dimensional degradation signature is constructed from the relative eficiency changes defined in Eq. (31), evaluated for the fan, LPC, HPC, HPT, and LPT. The healthy and end-of-life windows are the same as those used in Section 5.5. Before computing these changes, the latent coordinates are expressed at a common reference flight condition using Eq. (11). For the pooled model, this reference is computed as the mean over the development samples and corresponds to altitude 15 442 ft, Mach 0.554, throttle-resolver angle 66.3, and $T _ { 2 } = 4 9 3 . 7 ^ { \circ } \mathrm { R }$ . The magnitudes of the five relative changes are normalized to unit Euclidean norm,

$$
s _ { X } = { \frac { | d _ { X } | } { \sqrt { \sum _ { Y \in C } d _ { Y } ^ { 2 } } } } , \qquad C = \{ { \mathrm { f a n } } , { \mathrm { L P C } } , { \mathrm { H P C } } , { \mathrm { H P T } } , { \mathrm { L P T } } \} .\tag{32}
$$

The resulting signature describes the relative distribution of the inferred eficiency changes along the gas path. All three panels of Figure 12, including the diagnostic scores and PCA, use this normalized signature.

Before any test, it is useful to look at the signatures in the order in which the components appear in the engine, which is what Figure 12a shows. Each faint line is one test engine: its five normalised values plotted against the physical position of the corresponding component, from the fan at the inlet to the LPT at the exhaust, with the combustor marked by the dashed line between $\eta _ { \mathrm { H P C } }$ and $\eta _ { \mathrm { H P T } }$ . The two heavy curves are the means of the twelve compression-degraded and the six expansion-degraded engines. The two mean profiles are very distinguishable. The compression-degraded mean rises from 0.30 at the fan to a maximum of 0.78 at $\eta _ { \mathrm { L P C } }$ and then falls across the combustor to a minimum of 0.16 at $\eta _ { \mathrm { H P T } }$ . The diference between the two means changes sign exactly once, between $\eta _ { \mathrm { H P C } }$ and $\eta _ { \mathrm { H P T } }$ : the two curves cross at the combustor. This behaviour is perfectly coherent with the expected degradation, proving the model ability to discern between degradation modes in diferent sections of the engine. The following paragraphs further discuss this ability.

The diagnostic assessment therefore focuses on discrimination between compression-side and expansion-side degradation. Two scores are defined by summing the magnitudes of the degradation-induced changes in the eficiencyrelated coordinates belonging to each side:

$$
S _ { \mathrm { c o m p } } = s _ { \mathrm { f a n } } + s _ { \mathrm { L P C } } + s _ { \mathrm { H P C } } , \qquad S _ { \mathrm { t u r b } } = s _ { \mathrm { H P T } } + s _ { \mathrm { L P T } } .\tag{33}
$$

and the degradation is assigned to the side with the larger score. The two scores group the latent eficiencies according to their position in the engine, providing a diagnostic rule that requires no additional training or fault labels. Applied to the 18 test engines of the pooled dataset, the rule assigns every engine to the correct side of the combustor. Figure 12b reports the outcome engine by engine, each contributing one signed bar $S _ { \mathrm { c o m p } } - S _ { \mathrm { t u r b } }$ drawn towards the side it selects and coloured by the true region.

The grouping in Eq. (33) still relies on prior knowledge of the engine layout, since the latent coordinates are assigned in advance to the compression and expansion sides. To check whether the same separation is present without this information, PCA is applied directly to the five-dimensional normalized degradation signatures of the 18 engines. This step is fully unsupervised: the degradation-region labels are not used to compute the projection and are added only afterwards to identify the points in Figure 12c.

The two degradation regions separate clearly along the first principal component, which explains 73% of the total variance. Compression-degraded engines occupy $\mathrm { P C } _ { 1 } ~ \in ~ [ - 0 . 4 0 , + 0 . 1 7 ]$ , while expansion-degraded engines lie in $\mathrm { P C } _ { 1 } \in [ 0 . 4 1 , 0 . 6 9 ]$ , with a gap of 0.232 between the two groups. This shows that the distinction between the two degradation regions is already present in the learned eficiency vectors, without using any information about the afected region or component. This becomes clearer when the dataset labels are added after the projection. Units from DS03, which contains turbine-related degradation, are grouped on the right side of the plot, whereas units from DS04, DS05, and DS06, associated with fan and compressor degradation, lie on the left. Within this latter group, the DS04 units, represented by dark blue circles, separate from the other compression fleets along PC2, suggesting that a more specific degradation signature may also be retained. Overall, the PCA shows that the learned eficiency coordinates retain information that separates turbine degradation from fan/compressor degradation. This analysis illustrates how the physical structure of the latent space supports diagnostic interpretation through simple post-processing; developing a dedicated diagnostic strategy lies outside the scope of this paper.

## 5.7. Evolution in Component-Map Coordinates

For turbofan engines, component maps are standard tools used to characterize the operating behaviour of compressors and turbines. They describe the relationship between quantities such as corrected mass flow, pressure ratio or expansion ratio, corrected speed, and eficiency, and are routinely used to assess how a component operates over its admissible operating range. Changes in component condition modify these relationships and can therefore appear as a displacement of the operating region within the map.

The proposed physics-constrained representation makes it possible to recover part of this information directly from the latent space. In particular, the inferred turbine expansion ratios and corrected flows, evaluated for each sample using its latent coordinates and recorded flight conditions, can be represented in the same coordinates used for conventional turbine maps. The resulting plots describe the operating lines occupied by each turbine over the conditions available in the data and how these lines evolve as degradation progresses. Corrected-speed lines and eficiency contours are not reconstructed. Details of the construction are provided in Appendix A.

Figure 13 reports this representation for the HPT and LPT of three representative test units. The grey points show the complete set of inferred operating points, while the coloured lines describe the corrected-flow–expansion-ratio relation at increasing life fractions. A systematic upward displacement is visible with increasing degradation, indicating an increase in inferred corrected flow for a given expansion ratio. For the three units shown, the increase in corrected flow amounts to 1.66%, 1.47%, and 0.99% for the HPT and 1.64%, 1.38%, and 0.57% for the LPT. Across all test units, the average increases are approximately 1.13% and 0.64%, respectively. These displacements describe the operating lines recovered by the model and remain subject to the parameterization and identifiability limits established in Section 5.5.

## 6. Discussion: Physical Interpretation and Engineering Relevance

The main contribution of the proposed architecture lies in the physical structure of its latent representation. The physics-constrained autoencoder achieves reconstruction and HI performance comparable to the considered datadriven representations while structurally associating its latent coordinates with predefined physical quantities. This structure allows the latent variables to be examined before their information is aggregated into a system-level HI. The results provide three complementary levels of interpretation: (i) component-related degradation trends in the latent space, (ii) their representation through conventional turbofan component-map coordinates, and (iii) a physically defined basis for diagnostic analysis. These results are obtained using conventional thermodynamic relations drawn from textbook turbofan-cycle analysis. Even at this level of model fidelity, the equations provide suficient structure to recover degradation-sensitive latent trends, support area diagnostic interpretation, and express the inferred evolution in component-map coordinates.

life fraction c/ceoL at the bin centre  
![](images/eb97073d8406024f1c20e24e35b5a999da558e29dba5bb6818a0fd9475af3a72.jpg)  
Figure 13: Degradation-induced evolution of the inferred HPT and LPT operating lines in turbine-map coordinates for three representative units with a single seed. Grey points show the complete set of inferred operating points, while coloured lines represent the corrected-flow–expansion-ratio relation at increasing life fractions. The progressive upward displacement indicates an increase in inferred corrected flow at a given expansion ratio with degradation. The percentages reported in each panel quantify the corresponding increase in corrected flow. These representations describe the evolution of operating lines in component-map coordinates and do not constitute complete turbine characteristic maps.

The DS03 results illustrate how the representation captures both the afected turbine quantities and the broader thermodynamic response to degradation. The two turbine eficiencies emerge among the most informative latent quantities although the model is not provided with information on which components are degraded. The high ranking of $T _ { 4 1 }$ captures a complementary aspect of engine degradation: its evolution reflects the operating response as component performance deteriorates. The degradation information is therefore expressed through coordinates associated with the afected turbine section and through quantities describing the response of the engine as a whole. The agreement of most inferred trends with the two external performance simulators provides a qualitative consistency check of this interpretation.

The diagnostic signatures and component operating lines show how this information can be examined in forms familiar to engine specialists. In the pooled experiment, simple physically grouped scores and PCA reveal a distinction between compression-side and expansion-side degradation. The component-map coordinates subsequently express the inferred evolution through relationships between corrected flow and expansion ratio. These representations complement the scalar HI by retaining information on how degradation manifests within the engine. Their engineering value lies in making the learned health evolution accessible for physical assessment. Developing a dedicated diagnostic strategy lies outside the scope of this paper.

The $P _ { 4 5 }$ experiment further shows how the physical resolution of the representation can change with the sensor configuration. The additional inter-turbine pressure increases the rank of the mean Jacobian and improves separation of the turbine expansion ratios, while reconstruction and system-level HI performance remain comparable. Sensor information therefore afects which component-related changes can be distinguished, even when the existing measurements already support accurate reconstruction and health tracking. The physical decoder provides a framework in which this relationship between measurements and latent interpretation can be examined explicitly.

## 7. Limitations of the Present Study

The proposed architecture embeds a set of conventional turbofan thermodynamic relations in the decoder and demonstrates that these relations can force the latent space toward physically meaningful quantities while preserving adequate reconstruction and HI performance. Nevertheless, this study still faces some limitations, explained in the next paragraphs.

Quantitative parameter estimation. The present study establishes the usefulness of the latent representation for health tracking and regional degradation interpretation. Quantitative estimation of individual component parameters requires further validation, with the analysis in Section 5.5 identifying where additional measurements and a richer degradation parameterization could improve their recovery.

Model Fidelity. A further limitation concerns the fidelity of the embedded thermodynamic model. The decoder provides a simplified representation of the engine cycle and cannot reproduce all secondary flows, component characteristics, control interactions, and nonlinear couplings represented in higher-fidelity performance models. The physical meaning recovered in the latent space is consequently bounded by the equations through which the latent variables act. Increasing the fidelity of these relations could improve the quantitative estimation and separation of individual component quantities, although this would introduce additional modelling, biases, and computational requirements. This also afect the definition of the initial set of parameters fitted by the model.

Healthy reference and trajectory processing. The operating-condition correction requires an initial healthy reference interval for each monitored unit. In this benchmark, its length, temporal smoothing, and per-engine HI anchoring use complete run-to-failure trajectories. Application during operation would require a healthy interval defined independently of eventual lifetime, causal processing, and an HI scale established without the monitored engine’s end-of-life values.

The results therefore establish the level of interpretability that can already be obtained from a relatively compact set of conventional thermodynamic equations. The remaining limitations point primarily to the physical formulation and to the information content of the sensor set adopted in the main experiments. Future developments should investigate richer component models, additional informative measurements, and observability-aware formulations capable of improving the separation of physically related degradation efects.

## 8. Conclusion

This work introduced a physics-decoder autoencoder for interpretable health representation learning in PHM. Rather than relying on a fully trainable neural decoder, the proposed architecture embeds diferentiable physical equations directly in the reconstruction path, thereby structurally associating the latent coordinates with predefined physical quantities. The resulting representation can be examined at the level of individual physical quantities before being aggregated into a system-level HI.

The methodology was demonstrated on N-CMAPSS, where the proposed model achieved a test reconstruction $R ^ { 2 }$ of 0.9993 and an MAE of 0.0036 while remaining competitive with fully data-driven autoencoder representations. After removal of the operating-condition contribution, the resulting HI achieved a Spearman correlation of 0.9791 with the available ground-truth HI, with an RMSE of 0.0286 and an MAE of 0.0233. Beyond system-level health estimation, the structured latent representation retained physically organized degradation information. The inferred turbine quantities exhibited degradation trends consistent with the imposed degradation mechanisms and could be represented in conventional component-map coordinates. When a single model was trained across four degradation scenarios, the learned eficiency coordinates further enabled compression-side and expansion-side degradation to be separated without training an additional diagnostic classifier.

The results also demonstrate an important limitation of physics-structured representation learning: structural association with a physical quantity does not guarantee that the corresponding quantity is uniquely identifiable from the available measurements. In other words, a physics-constrained representation learning can make a latent health space physically structured, but the achievable physical resolution is fundamentally limited by observability. The decoder Jacobian reveals limited observability and parameter coupling for several component quantities, particularly between the fan and LPC and between the HPT and LPT. The additional $P _ { 4 5 }$ experiment further showed that introducing an informative measurement increases the observable dimension of the latent space without materially changing systemlevel HI performance. The absence of dedicated component-specific flow-capacity modifiers further limits quantitative attribution, since these degradation efects may be represented through other latent quantities. Physical interpretability therefore depends on the adopted parameterization and the information content of the sensor set.

Future work will investigate observability-aware latent formulations, improved thermodynamic representations, and uncertainty-aware health estimation and prognostics. Overall, the proposed framework provides a basis for moving from purely degradation-sensitive health representations toward physically structured health representations whose achievable resolution can be explicitly assessed in terms of system observability.

## CRediT Authorship Contribution Statement

Leonardo Baldo: Methodology, Software, Validation, Formal analysis, Investigation, Visualization, Writing - Original Draft, Writing - Review & Editing Jokin Alcibar Ibarzabal: Methodology, Software, Validation, Investigation, Visualization, Writing - Review & Editing Daniel Cisneros Acevedo: Methodology, Software, Validation, Writing - Review & Editing Kai Goebel: Conceptualization, Validation, Writing - Review & Editing, Supervision Manuel Arias Chao: Conceptualization, Methodology, Validation, Writing - Review & Editing, Supervision, Funding acquisition

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Code and Data Availability Statement

Data sharing is not applicable to this article as no new data were created or generated. The N-CMAPSS dataset analysed in this study is publicly available [33]. The code needed to reproduce the analyses will be released in a public repository upon acceptance of the paper.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used ChatGPT and Claude to assist with language editing, improving readability, and refining the presentation of the manuscript. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## References

[1] Enrico Zio. Prognostics and health management (phm): Where are we and where do we (need to) go in theory and practice. Reliability Engineering & System Safety, 218:108119, 2022.

[2] Mojtaba Kordestani, Mehrdad Saif, Marcos E. Orchard, Roozbeh Razavi-Far, and Khashayar Khorasani. Failure prognosis and applications—a survey of recent literature. IEEE Transactions on Reliability, 70(2):728–748, 2021.

[3] Yang Hu, Xuewen Miao, Yong Si, Ershun Pan, and Enrico Zio. Prognostics and health management: A review from the perspectives of design, development and decision. Reliability Engineering & System Safety, 217:108063, 2022.

[4] Andrew K. S. Jardine, Daming Lin, and Dragan Banjevic. A review on machinery diagnostics and prognostics implementing condition-based maintenance. Mechanical Systems and Signal Processing, 20(7):1483–1510, 2006.

[5] Andrew Hess, Giulio Calvello, and Peter Frith. Challenges, issues, and lessons learned chasing the “big p”: Real predictive prognostics, part 1. In Proceedings ofthe 2005 IEEE Aerospace Conference, pages 3610–3619, 2005.

[6] Lisandro A Jimenez-Roa, Nela Koubkova, Lothar Kerschgens, and Arjan de Jong. A conceptual framework to integrate prognostics and health management with maintenance, repair and overhaul for a hydrogen-electric aircraft propulsion system. In PHM Society European Conference, volume 9 of 1, pages 1–15, 2026.

[7] Meng Xu, Xue Liu, Junjie Huang, Jian Ma, and Meiling Yue. Adaptive turbofan engine health assessment for multi-component performance degradation coupling scenarios: A spatiotemporal attention graph convolutional network approach. Reliability Engineering & System Safety, 269:112086, 2026

[8] Hengsong Hu, Yuehua Cheng, Bin Jiang, Wenzhuo Li, and Kun Guo. Missile fault detection and localization based on HBOS and hierarchica signed directed graph. Aerospace, 11(8):679, 2024.

[9] Leonardo Baldo, Andrea De Martin, Giovanni Jacazio, and Massimo Sorli. A systematic literature review on PHM strategies for (hydraulic) primary flight control actuation systems. Actuators, 14(8):382, 2025.

[10] Antonio Carlo Bertolino, Andrea De Martin, Giovanni Jacazio, and Massimo Sorli. Design and preliminary performance assessment of a phm system for electromechanical flight control actuators. Aerospace, 10(4):335, 2023.

[11] Alessandro Aimasso, Matteo DL Dalla Vedova, and Paolo Maggiore. Analysis of fbg sensors performances when integrated using diferent methods for health and structural monitoring in aerospace applications. In 2022 6th International Conference on System Reliability and Safety (ICSRS), pages 138–144. IEEE, 2022.

[12] Jorge E. García Bustos, Benjamin Brito Schiele, Bruno Masserano, Ricardo Salas-Espiñeira, Diego Troncoso-Kurtovic, David E. Acuña-Ureta, Francisco Jaramillo-Montoya, Marcos E. Orchard, Jorge F. Silva, and Aramis Perez. Enabling online maximum driving range prognostics in electric vehicles via uncertain event likelihood functions. Engineering Applications ofArtificial Intelligence, 173:114449, 2026.

[13] Jokin Alcibar, Aitor Aguirre-Ortuzar, Ekhi Zugasti, Chetan S Kulkarni, Juan Fernandez, and Jose I Aizpurua. Bayesian neural networks with physics-based priors for robust battery health assessment. Scientific Reports, 2026.

[14] Yaguo Lei, Bin Yang, Xinwei Jiang, Feng Jia, Naipeng Li, and Asoke K. Nandi. Applications of machine learning to machine fault diagnosis: A review and roadmap. Mechanical Systems and Signal Processing, 138:106587, 2020.

[15] Olga Fink, Qin Wang, Markus Svensén, Pierre Dersin, Wan-Jui Lee, and Melanie Ducofe. Potential, challenges and future directions for deep learning in prognostics and health management applications. Engineering Applications ofArtificial Intelligence, 92:103678, 2020.

[16] Haoxuan Zhou, Xin Huang, Guangrui Wen, Zihao Lei, Shuzhi Dong, Ping Zhang, and Xuefeng Chen. Construction of health indicators for condition monitoring of rotating machinery: A review of the research. Expert Systems with Applications, 203:117297, 2022.

[17] Chuang Chen, Mengchen Li, Jiantao Shi, Dongdong Yue, Ge Shi, and Cuimei Bo. Constructing high-quality health indicators from multisource sensor data for predictive maintenance applications. Expert Systems with Applications, 285:127870, 2025.

[18] Yang Ge, Xingxing Jiang, Benlian Xu, and Yifei Ding. Log-cumulative feature alignment for enhanced prognosis of aero-engine remaining useful life. Expert Systems with Applications, 277:127277, 2025.

[19] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8):1798–1828, 2013.

[20] Ana González-Muñiz, Ignacio Díaz, Abel A. Cuadrado, and Diego García-Pérez. Health indicator for machine condition monitoring built in the latent space of a deep autoencoder. Reliability Engineering & System Safety, 224:108482, 2022.

[21] Ingeborg de Pater and Mihaela Mitici. Developing health indicators and RUL prognostics for systems with few failure instances and varying operating conditions using a LSTM autoencoder. Engineering Applications ofArtificial Intelligence, 117:105582, 2023.

[22] Ahmad Kamal Mohd Nor, Srinivasa Rao Pedapati, Masdi Muhammad, and Víctor Leiva. Overview of explainable artificial intelligence for prognostic and health management of industrial assets based on preferred reporting items for systematic reviews and meta-analyses. Sensors, 21(23):8020, 2021.

[23] Ahmed I. Abed and Loh Wei Ping. Implementing data mining techniques for gas-turbine (gt) health tracking and life management: The bibliographic perspective. Expert Systems with Applications, 252:124077, 2024.

[24] Weikun Deng, Khanh T. P. Nguyen, Kamal Medjaher, Christian Gogu, and Jérôme Morio. Physics-informed machine learning in prognostics and health management: State of the art and challenges. Applied Mathematical Modelling, 124:325–352, 2023.

[25] Huiqin Li, Zhengxin Zhang, Tianmei Li, and Xiaosheng Si. A review on physics-informed data-driven remaining useful life prediction: Challenges and opportunities. Mechanical Systems and Signal Processing, 209:111120, 2024.

[26] Kristupas Bajarunas, Marcia Baptista, Kai Goebel, and Manuel Arias Chao. Unsupervised physics-informed health indicator estimation for complex systems. Annual Conference ofthe PHM Society, 15(1), 2023.

[27] George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-informed machine learning. Nature Reviews Physics, 3:422–440, 2021.

[28] Olga Fink, Vinay Sharma, Ismail Nejjar, Leandro Von Krannichfeldt, Sergei Garmaev, Zepeng Zhang, Amaury Wei, Gaetan Frusque, Florent Forest, Mengjie Zhao, Chi-Ching Hsu, Keivan Faghih Niresi, Han Sun, Hao Dong, Chenghao Xu, Rafael Theiler, Arthur Bizzi, and Kevin Steiner. From physics to machine learning and back: Part i - learning with inductive biases in prognostics and health management (phm). Reliability Engineering & System Safety, 271:112213, 2026.

[29] Kristupas Bajarunas, Marcia L. Baptista, Kai Goebel, and Manuel Arias Chao. Health index estimation through integration of general knowledge with unsupervised learning. Reliability Engineering & System Safety, 251:110352, 2024.

[30] Xucong Huang, Zhaoqin Peng, Diyin Tang, Juan Chen, Enrico Zio, and Zaiping Zheng. A physics-informed autoencoder for system health state assessment based on energy-oriented system performance. Reliability Engineering & System Safety, 242:109790, 2024.

[31] Yonas Tefera, Quinten Van Baelen, Maarten Meire, Stijn Luca, and Peter Karsmakers. Constraint-guided learning of data-driven health indicator models: An application on bearings. International Journal ofPrognostics and Health Management, 16(2), 2025.

[32] Antoni Plonczak, Matthias Wüest, and Lilach Goren Huber. Physics-constrained deep learning for interpretable anomaly detection in large battery packs with limited monitoring granularity. In PHM Society European Conference, volume 9 of 1, pages 1–13, 2026.

[33] Manuel Arias Chao, Chetan Kulkarni, Kai Goebel, and Olga Fink. Aircraft engine run-to-failure dataset under real flight conditions for prognostics and diagnostics. Data, 6(1):5, 2021.

[34] Jamie Coble and J. Wesley Hines. Identifying optimal prognostic parameters from data: A genetic algorithms approach. Annual Conference ofthe Prognostics and Health Management Society, 1(1), 2009.

[35] Yaguo Lei, Naipeng Li, Liang Guo, Ningbo Li, Tao Yan, and Jing Lin. Machinery health prognostics: A systematic review from data acquisition to RUL prediction. Mechanical Systems and Signal Processing, 104:799–834, 2018.

[36] Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. Film: visual reasoning with a general conditioning layer. In Proceedings ofthe Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications ofArtificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence, AAAI’18/IAAI’18/EAAI’18. AAAI Press, 2018.

[37] Jingcheng Wen, Jiaxin Ren, Zhibin Zhao, Zhi Zhai, and Xuefeng Chen. Residual-based adversarial feature decoupling for remaining useful life prediction of aero-engines under variable operating conditions. Expert Systems with Applications, 255:124538, 2024.

[38] Ahmed I. Abed and Wei Ping Loh. Implementing data mining techniques for gas-turbine (GT) health tracking and life management: The bibliographic perspective. Expert Systems with Applications, 252:124077, 2024.

[39] Philip P Walsh and Paul Fletcher. Design Point Performance and Engine Concept Design. John Wiley & Sons, Ltd, 2004.

[40] John Doherty and Randall J. Hunt. Two statistics for evaluating parameter identifiability and error reduction. Journal of Hydrology, 366(1– 4):119–127, 2009.

[41] J. D. Stigter and Jaap Molenaar. A fast algorithm to assess local structural identifiability. Automatica, 58:118–124, 2015.

[42] Advanced User-Friendly Gas Turbine Performance Calculations on a Personal Computer, volume Volume 5: Manufacturing Materials and Metallurgy; Ceramics; Structures and Dynamics; Controls, Diagnostics and Instrumentation; Education; IGTI Scholar Award of Turbo Expo, 06 1995.

[43] Narahari Rath, RK Mishra, and Abhijit Kushari. Investigation of performance degradation in a mixed flow low bypass turbofan engine. Journal ofFailure Analysis and Prevention, 23(1):378–388, 2023.

[44] Daniel Cisneros Acevedo, Marta Ribeiro, Ingeborg de Pater, Manuel Arias Chao, and Tim Rootliep. Turbograd: An open-source diferentiable performance model for aeroengine condition monitoring. In PHM Society European Conference, volume 9, pages 1–8, 2026.

[45] Yuan Tian, Manuel Arias Chao, Chetan Kulkarni, Kai Goebel, and Olga Fink. Real-time model calibration with deep reinforcement learning. Mechanical Systems and Signal Processing, 165:108284, 2022.

## A. Construction of the Turbine Operating Lines in Component-Map Coordinates

The HPT and LPT representations reported in Section 5.7 are constructed directly from quantities evaluated by the physical decoder. Corrected mass flow is referred to ISA sea-level conditions, $T _ { \mathrm { r e f } } = 2 8 8 . 1 5 \mathrm { K }$ and $P _ { \mathrm { r e f } } = 1 0 1 . 3 2 5 \mathrm { k P a }$ according to

$$
\theta _ { i } = \frac { T _ { i } } { T _ { \mathrm { r e f } } } , \qquad \delta _ { i } = \frac { P _ { i } } { P _ { \mathrm { r e f } } } , \qquad W _ { c , i } = W _ { i } \frac { \sqrt { \theta _ { i } } } { \delta _ { i } } .\tag{34}
$$

The operating lines are evaluated using each sample’s original latent coordinates $\mathbf { z } _ { t }$ and recorded operating conditions $\mathbf { w } _ { t } .$ The grey cloud therefore represents the sampled operating points of each engine. In contrast, the degradation signatures in Section 5.6 use the transported coordinates $\mathbf { z } _ { t , \mathrm { r e f } }$ together with the fixed conditions $\mathbf { w } _ { \mathrm { r e f } } .$ For the operating lines, early- and late-life comparisons are made within common expansion-ratio bins, as described below. For the two turbines, the correction is applied at the corresponding component inlet:

$$
W _ { c , \mathrm { H P T } } = W _ { 4 1 5 } \frac { \sqrt { T _ { 4 1 5 } / T _ { \mathrm { r e f } } } } { P _ { 4 1 5 } / P _ { \mathrm { r e f } } } , \qquad W _ { c , \mathrm { L P T } } = W _ { 4 6 } \frac { \sqrt { T _ { 4 6 } / T _ { \mathrm { r e f } } } } { P _ { 4 6 } / P _ { \mathrm { r e f } } } .\tag{35}
$$

The abscissa is the turbine expansion ratio, obtained directly from its latent coordinate as $\Pi _ { X } = \Pi _ { X } ^ { 0 } \exp ( 0 . 0 5 z _ { \Pi _ { X } } )$ with $X \in \{ \mathrm { H P T } , \mathrm { L P T } \}$ . The ordinate is the corrected mass flow calculated from the corresponding decoder flow, pressure, and temperature.

For each engine, the samples are divided into life-fraction intervals $L _ { \ell }$ and into common expansion-ratio bins. Since the corrected flow $W _ { c }$ depends on the turbine expansion ratio $\Pi _ { t }$ , a direct comparison between early- and latelife values could be afected by diferences in the operating conditions experienced in the two periods. Using the same expansion-ratio bins for all life intervals therefore allows the evolution of $W _ { c }$ to be evaluated at comparable values of $\Pi _ { t }$

For each life interval $\ell ,$ an operating line is obtained by computing the median corrected flow in each expansionratio bin:

$$
\bar { W } _ { c , \ell , b } = \operatorname * { m e d i a n } \left\{ W _ { c , i } : \Pi _ { t , i } \in [ e _ { b - 1 } , e _ { b } ) , \tau _ { i } \in L _ { \ell } \right\} , \qquad \bar { \Pi } _ { t , b } = \frac { e _ { b - 1 } + e _ { b } } { 2 } ,\tag{36}
$$

where � is the normalized life fraction and $\bar { \Pi } _ { t , b }$ is the center of bin �.

The displacement of the operating line over the engine life is then quantified by comparing the corrected flow in the first and last life intervals at the same expansion-ratio bins:

$$
\Delta _ { c } = \frac { 1 0 0 } { \left| B \right| } \sum _ { b \in B } \left( \frac { \bar { W } _ { c , n _ { \ell } , b } } { \bar { W } _ { c , 1 , b } } - 1 \right) ,\tag{37}
$$

where � contains only the bins for which corrected-flow values are available in both the first and last life intervals. Thus, $\Delta _ { c }$ represents the mean percentage change in corrected flow from early to late life at comparable turbine expansion ratios.

## B. Benchmarking Baselines

Data-driven autoencoder. The data-driven autoencoder uses the same encoder structure as the proposed model. The 12 normalized sensor measurements are processed by a first Dense(256, ReLU) layer, followed by four Dense(256, ReLU) layers, and a linear output layer of dimension $n _ { z }$ . The operating conditions are not provided to the encoder. Two configurations are considered, with $n _ { z } = 1 5$ and $n _ { z } = 1$

The physical decoder is replaced by a fully trainable FiLM-conditioned neural decoder. Denoting the four operating conditions by �, the modulation is defined as

$$
\mathrm { F i L M } ( \mathbf { h } , \mathbf { w } ) = \gamma ( \mathbf { w } ) \odot \mathbf { h } + \beta ( \mathbf { w } ) ,\tag{38}
$$

where $\gamma ( \mathbf { w } )$ and $\beta ( \mathbf { w } )$ are afine mappings from $\mathbb { R } ^ { 4 }$ to $\mathbb { R } ^ { 2 5 6 }$ . The decoder is composed of an initial 256-unit layer, four FiLM-modulated 256-unit blocks, a 245-unit fully connected layer, and a linear 12-dimensional reconstruction layer. FiLM conditioning was adopted to limit the direct unbalance between operating conditions and reconstructed measurements observed when � is concatenated directly with the latent vector.

The data-driven models are trained using the MSE reconstruction loss, Adam with a learning rate of $1 0 ^ { - 3 }$ , a batch size of 512, gradient-norm clipping at 1.0, and a maximum of 120 epochs. Early stopping is applied to the validation loss with a patience of 25 epochs and the best weights are restored.

One-dimensional physics-constrained representation. The reduced-order physics baseline preserves the same 15-dimensional thermodynamic decoder used by the proposed model. The encoder produces a single latent coordinate, which is mapped through a trainable linear layer to the 15 physical modifiers before entering the physical decoder.

## C. Further Latent Space Validation

In an efort to provide additional validation independent of the N-CMAPSS simulation environment, the degradation scenario was reproduced using GasTurb and TurboGrad gas turbine performance simulations tools for design and of-design analysis [42, 43, 44]. A generic two-spool turbofan configuration was used to provide an external qualitative consistency check of the degradation-induced trends. The simulators are not intended to reproduce the N-CMAPSS engine quantitatively. Instead, the comparison focuses on the direction of change of the inferred physical quantities. Table 8 shows that the changes inferred by the proposed model follow for most of the channels the same qualitative directions as those obtained with GasTurb and TurboGrad when HPT-eficiency, LPT-eficiency, and LPTflow-capacity degradation are imposed. Compared to GasTurb, the discrepancies are linked to $\Pi _ { \mathrm { H P C } } .$ , Π<sub>LPT</sub>, $\eta _ { f a n } ,$ and the cooling flows. TurboGrad shows discrepancies in $\eta _ { \mathrm { H P C } }$ and BPR, alongside those in $\Pi _ { \mathrm { H P C } }$ and $\eta _ { \mathrm { f a n } }$ . The observed discrepancies may also partly result from diferences in simulator’s control strategy. GasTurb and TurboGrad both employ $N _ { c }$ -based control, whereas the N-CMAPSS high-bypass class engine is expected to rely primarily on lowpressure spool/fan control. These diferent control assumptions can alter how degradation efects propagate through pressure ratios, flows, and bypass ratio. Moreover, the cooling-flow fractions in the simulators are kept constant, preventing a direct comparison of their inferred evolution.

This additional analysis complements the quantitative evidence already obtained from the main benchmark. The GasTurb and TurboGrad comparison therefore addresses the component-level physical interpretation specifically, providing an independent consistency check where direct experimental validation of the inferred modifiers is not available.

## D. Decoder Sensitivity and Latent Identifiability

This appendix describes the sensitivity and identifiability analyses used to select the latent coordinates and assess their interpretation. Sensitivity measures how strongly a coordinate afects the reconstructed measurements. Identifiability additionally concerns whether this efect can be distinguished from changes in other coordinates.

Sensitivity and latent-space selection. For a given latent vector and operating condition, the decoder Jacobian is

$$
{ \bf J } = \frac { \partial \hat { \bf x } } { \partial { \bf z } } , \qquad J _ { i k } = \frac { \partial \hat { x } _ { i } } { \partial z _ { k } } .\tag{39}
$$

Qualitative comparison of the degradation-induced evolution of the physical quantities inferred by the proposed model and obtained with GasTurb and TurboGrad. ↑, ↓, and → indicate an increase, decrease, and no clear systematic variation, respectively.
<table><tr><td>Variable</td><td>Our model</td><td>GasTurb</td><td>TurboGrad</td><td>Variable</td><td>Our model</td><td>GasTurb</td><td>TurboGrad</td></tr><tr><td> $\eta _ { \mathrm { f a n } }$ </td><td>→</td><td>↓</td><td>↓</td><td> $\Pi _ { \mathrm { H P C } }$ </td><td>↓</td><td>↑</td><td>↑</td></tr><tr><td> $\eta _ { \mathrm { L P C } }$ </td><td>↓</td><td>↓</td><td>↓</td><td> $\Pi _ { \mathrm { H P T } }$ </td><td>↓</td><td>↓</td><td>↓</td></tr><tr><td>ηHPC</td><td>↑</td><td>↑</td><td>→</td><td> $\Pi _ { \mathrm { L P T } }$ </td><td>↓</td><td>↑</td><td>↓</td></tr><tr><td> $\eta _ { \mathrm { H P T } }$ </td><td>↓</td><td>↓</td><td>↓</td><td> $W _ { \mathrm { m o d } }$ </td><td>↑</td><td>↑</td><td>↑</td></tr><tr><td> $\eta _ { \mathrm { L P T } }$ </td><td>↓</td><td>↓</td><td>↓</td><td>BPR</td><td>→</td><td>→</td><td>↑</td></tr><tr><td> $\Pi _ { \mathrm { f a n } }$ </td><td>↑</td><td>↑</td><td>↑</td><td> $T _ { 4 1 }$ </td><td>↑</td><td>↑</td><td>↑</td></tr><tr><td> $\Pi _ { \mathrm { L P C } }$ </td><td>↑</td><td>↑</td><td>↑</td><td> $\beta _ { \mathrm { H P T } }$ </td><td>→</td><td>一</td><td>一</td></tr><tr><td></td><td></td><td></td><td></td><td> $\beta _ { \mathrm { L P T } }$ </td><td>↓</td><td>一</td><td>一</td></tr></table>

![](images/a753d8e99ac7ce04611dfa0eb70d33a675869701e27f8c04bb06364d3ba429e6.jpg)  
(a) Latent sensitivity.

![](images/4fabdb524bbc48cf76b73dc54646205469c0612ba673c92fe057430f22e8edee.jpg)

(b)  
![](images/578c277a66602585b114aa2b474c203d0e2144fc7b1fa91990c226a94bf9b9d8.jpg)  
(b) Cooling-coordinate ablation.  
Figure 14: Sensitivity of the latent coordinates and efect of retaining the discarded combustor cooling fraction. (a) Norm of each coordinate’s column of the decoder Jacobian, in normalized sensor units per unit of latent change, on a logarithmic axis. The retained coordinates are shown in blue and $\beta _ { \mathrm { c o m b } }$ in red. (b) Reconstruction $R ^ { 2 }$ and HI composite score for the adopted 15-dimensional representation and the 16-dimensional representation retaining $\beta _ { \mathrm { c o m b } } .$ Bars show means over five independently seeded training runs, error bars show the spread across seeds, and dots show individual runs. Reintroducing the coordinate provides no improvement in either metric.

Each column describes the response of the reconstructed sensor variables to a small change in one latent coordinate, with the operating conditions held fixed. Its norm measures the strength of this response in normalized sensor units per unit of latent change.

The initial 16-dimensional representation was examined using these column norms and the corresponding reconstruction-loss gradients, reported in Figure 14a and Table 9. The combustor cooling fraction $\beta _ { \mathrm { c o m b } }$ has a Jacobiancolumn norm of $3 . 7 7 \times 1 0 ^ { - 5 }$ , compared with $1 . 8 0 \times 1 0 ^ { - 3 }$ for the next lowest coordinate, $\beta _ { \mathrm { L P T } }$ . The ablation in Figure 14b further shows that retaining $\beta _ { \mathrm { c o m b } }$ provides no improvement in reconstruction or HI quality. This coordinate was therefore fixed at its nominal value, yielding the adopted 15-dimensional representation.

Construction of the mean Jacobian. The Jacobian is evaluated at 1 924 operating points sampled using a grid defined by quartile intervals of altitude, Mach number, and throttle-resolver angle, with samples drawn from each occupied cell. At each point, derivatives of the twelve normalized reconstruction targets with respect to the fifteen latent coordinates are approximated by central finite diferences, one coordinate at a time, using a step of $1 0 ^ { - 3 }$ in latent units. The decoder evaluations are performed in single precision, consistent with model training, while the Jacobian matrices are stored and averaged in double precision. The SVD is also computed in double precision. Finite diferences provide a common sensitivity calculation across the decoder variants.

The matrix entering the decomposition is the arithmetic mean,

$$
\overline { { \mathbf { J } } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbf { J } _ { i } \in \mathbb { R } ^ { 1 2 \times 1 5 } , \qquad N = 1 9 2 4 .\tag{40}
$$

Numerical rank is determined by counting singular values satisfying $\sigma _ { j } > 1 0 ^ { - 3 } \sigma _ { 1 }$

Singular value decomposition. The retained coordinates were then examined jointly through SVD of the mean decoder Jacobian,

$$
\mathbf { \overline { { J } } } = \mathbf { U } \mathbf { \Sigma } \mathbf { V } ^ { \top } , \qquad \mathbf { \overline { { J } } } \in \mathbb { R } ^ { 1 2 \times 1 5 } .\tag{41}
$$

In the full decomposition, $\mathbf { U } \in \mathbb { R } ^ { 1 2 \times 1 2 }$ and $\mathbf { V } \in \mathbb { R } ^ { 1 5 \times 1 5 }$ contain orthonormal output and latent directions, respectively, while $\pmb { \Sigma } \in \mathbb { R } ^ { 1 2 \times 1 \hat { 5 } }$ contains the singular values.

Each column $\mathbf { v } _ { j }$ of � describes a combination of latent-coordinate changes. For the first twelve directions,

$$
\mathbf { \bar { J } } \mathbf { v } _ { j } = \sigma _ { j } \mathbf { u } _ { j } , \qquad j = 1 , \ldots , 1 2 ,\tag{42}
$$

where $\mathbf { u } _ { j }$ is the corresponding column of �. The singular value $\sigma _ { j }$ measures the strength of the resulting output response. A nonzero singular value therefore identifies a latent direction that afects the measurements. A zero singular value identifies a direction that produces no output change in the evaluated linearization. Since there are fifteen latent coordinates and twelve outputs, the full decomposition also contains three additional latent directions with zero output response.

This separation of measurement-sensitive and null-space directions follows sensitivity-based approaches to parameter identifiability [40, 41].

Rank and conditioning. For the adopted model, the first nine singular values, normalized by the largest, are

$$
{ \frac { 1 } { \sigma _ { 1 } } } \left[ \sigma _ { 1 } \cdots \sigma _ { 9 } \right] = \left[ 1 . 0 0 0 , 0 . 7 5 1 , 0 . 4 5 9 , 0 . 3 5 2 , 0 . 2 3 6 , \right.\tag{43}
$$

$$
0 . 2 1 6 , 0 . 1 7 2 , 0 . 1 4 5 , 0 . 0 6 2 ] ,
$$

while the remaining three singular values are zero to numerical precision. The mean Jacobian therefore has rank 9 and a six-dimensional null space.

The ratio between the largest and smallest retained singular values is $\sigma _ { 1 } / \sigma _ { 9 } = 1 6 . 1 \pm 0 . 1 $ across the five seeds. This ratio describes the relative sensitivity of the retained directions. The gap between the ninth and tenth singular values is of the order of $1 0 ^ { 1 5 }$ , separating these directions clearly from the numerical null space. The results characterize the evaluated mean Jacobian; estimation accuracy additionally depends on measurement uncertainty and model fidelity. A $1 2 \times 1 5$ Jacobian necessarily has at least three null-space directions because the number of latent coordinates exceeds the number of outputs. The measured rank of nine therefore gives three additional null-space directions beyond this dimensional minimum. The decoder determines the orientation of the resulting six-dimensional null space and its contributions to the individual latent coordinates. The stability of the numerical rank was examined by repeating the calculation with finite-diference steps from $1 0 ^ { - 1 }$ to $1 0 ^ { - 5 }$ . Across these calculations, $\sigma _ { 9 } / \sigma _ { 1 }$ remains approximately 0.062, while $\sigma _ { 1 0 } / \sigma _ { 1 }$ ranges from $1 . 4 \times 1 0 ^ { - 1 7 }$ to $4 . 4 \times 1 0 ^ { - 1 7 }$ . The rank remains nine at each tested relative tolerance, $1 0 ^ { - 2 } , 1 0 ^ { - 6 }$ , and $1 0 ^ { - 1 2 }$ . These results support the stability of the rank estimate across the tested step sizes and tolerances.

Interpretation of individual coordinates. To relate these directions to individual physical quantities, let

$$
\mathbf { V } _ { 0 } = \left[ \mathbf { v } _ { 1 0 } \quad \cdots \quad \mathbf { v } _ { 1 5 } \right]\tag{44}
$$

Sensitivity and local identifiability of the latent coordinates. The sensitivity and loss-gradient norms refer to the initial 16- dimensional representation. The null-space share $\boldsymbol { u } _ { k }$ refers to the adopted 15-dimensional representation and is therefore not defined for the removed coordinate. It quantifies the squared projection of each coordinate onto the null space of the evaluated mean Jacobian.
<table><tr><td>Coordinate</td><td> $\| \partial \hat { \bf x } / \partial z _ { k } \|$ </td><td> $\| \partial \mathcal { L } / \partial z _ { k } \|$ </td><td> $\boldsymbol { u } _ { k }$ </td><td>Status</td></tr><tr><td> $\beta _ { \mathrm { c o m b } }$ </td><td> $3 . 7 7 \times 1 0 ^ { - 5 }$ </td><td> $7 . 8 6 \times 1 0 ^ { - 1 8 }$ </td><td></td><td>removed</td></tr><tr><td> $\beta _ { \mathrm { L P T } }$ </td><td> $1 . 8 0 \times 1 0 ^ { - 3 }$ </td><td> $1 . 6 3 \times 1 0 ^ { - 1 0 }$ </td><td>0.998</td><td>kept</td></tr><tr><td> $\beta _ { \mathrm { H P T } }$ </td><td> $9 . 5 2 \times 1 0 ^ { - 3 }$ </td><td> $1 . 0 4 \times 1 0 ^ { - 9 }$ </td><td>0.993</td><td>kept</td></tr><tr><td> $\eta _ { \mathrm { L P C } }$ </td><td> $1 . 4 9 \times 1 0 ^ { - 2 }$ </td><td> $7 . 2 3 \times 1 0 ^ { - 1 0 }$ </td><td>0.646</td><td>kept</td></tr><tr><td> $\eta _ { \mathrm { f a n } }$ </td><td> $2 . 0 3 \times 1 0 ^ { - 2 }$ </td><td> $9 . 7 3 \times 1 0 ^ { - 1 0 }$ </td><td>0.354</td><td>kept</td></tr><tr><td> $\mathrm { B P R }$ </td><td> $2 . 1 3 \times 1 0 ^ { - 2 }$ </td><td> $1 . 8 1 \times 1 0 ^ { - 9 }$ </td><td>0.669</td><td>kept</td></tr><tr><td> $W _ { \mathrm { m o d } }$ </td><td> $2 . 3 4 \times 1 0 ^ { - 2 }$ </td><td> $1 . 9 9 \times 1 0 ^ { - 9 }$ </td><td>0.600</td><td>kept</td></tr><tr><td> $\eta _ { \mathrm { L P T } }$ </td><td> $3 . 5 7 \times 1 0 ^ { - 2 }$ </td><td> $3 . 0 1 \times 1 0 ^ { - 9 }$ </td><td>0.101</td><td>kept</td></tr><tr><td> $\Pi _ { \mathrm { L P T } }$ </td><td> $4 . 3 0 \times 1 0 ^ { - 2 }$ </td><td> $1 . 8 7 \times 1 0 ^ { - 9 }$ </td><td>0.427</td><td>kept</td></tr><tr><td> $\Pi _ { \mathrm { H P T } }$ </td><td> $4 . 7 1 \times 1 0 ^ { - 2 }$ </td><td> $2 . 4 2 \times 1 0 ^ { - 9 }$ </td><td>0.427</td><td>kept</td></tr><tr><td> $\eta _ { \mathrm { H P T } }$ </td><td> $4 . 7 3 \times 1 0 ^ { - 2 }$ </td><td> $4 . 8 3 \times 1 0 ^ { - 9 }$ </td><td>0.639</td><td>kept</td></tr><tr><td> $\Pi _ { \mathrm { H P C } }$ </td><td> $7 . 4 7 \times 1 0 ^ { - 2 }$ </td><td> $3 . 2 4 \times 1 0 ^ { - 9 }$ </td><td> $< 1 0 ^ { - 5 }$ </td><td>kept</td></tr><tr><td> $\eta _ { \mathrm { H P C } }$ </td><td> $8 . 4 4 \times 1 0 ^ { - 2 }$ </td><td> $3 . 2 3 \times 1 0 ^ { - 9 }$ </td><td> $< 1 0 ^ { - 5 }$ </td><td>kept</td></tr><tr><td> $\Pi _ { \mathrm { L P C } }$ </td><td> $1 . 0 3 \times 1 0 ^ { - 1 }$ </td><td> $5 . 4 0 \times 1 0 ^ { - 9 }$ </td><td> $< 1 0 ^ { - 5 }$ </td><td>kept</td></tr><tr><td> $\Pi _ { \mathrm { f a n } }$ </td><td> $1 . 2 2 \times 1 0 ^ { - 1 }$ </td><td> $6 . 3 1 \times 1 0 ^ { - 9 }$ </td><td> $< 1 0 ^ { - 5 }$ </td><td>kept</td></tr></table>

contain an orthonormal basis of the six-dimensional null space. The null-space share of coordinate � is

$$
u _ { k } = \left\| \mathbf { V } _ { 0 } ^ { \top } \mathbf { e } _ { k } \right\| _ { 2 } ^ { 2 } = \sum _ { j = 1 0 } ^ { 1 5 } V _ { k j } ^ { 2 } ,\tag{45}
$$

where ${ \bf e } _ { k }$ is the unit vector associated with that coordinate. This quantity lies between zero and one and is independent of the particular orthonormal basis chosen for the null space. A value near zero indicates that the coordinate has little contribution to the unconstrained directions; a value near one indicates that it lies almost entirely within them. The shares sum to six, the dimension of the null space.

Table 9 shows negligible null-space shares for $\Pi _ { \mathrm { f a n } } , \Pi _ { \mathrm { L P C } } , \Pi _ { \mathrm { H P C } }$ , and $\eta _ { \mathrm { H P C } }$ , with $u _ { k } < 1 0 ^ { - 5 }$ . The two retained cooling fractions have $u _ { k } > 0 . 9 9$ , while the remaining coordinates have intermediate values.

These results also explain why sensitivity alone is insuficient to assess identifiability. Although $\eta _ { \mathrm { H P T } }$ has a larger Jacobian-column norm than $\eta _ { \mathrm { L P T } }$ , its null-space share is also larger, approximately 0.64 compared with 0.10. Its influence on the measurements is therefore stronger but more closely shared with other latent quantities.

## E. Nominal-Engine Calibration

The physical decoder is constructed from general textbook thermodynamic relations for a two-spool turbofan engine and is not derived from the internal N-CMAPSS simulation model. Its equations describe the main component and gas-path relations required by the proposed architecture, while several details of the high-fidelity simulation used to generate N-CMAPSS are intentionally not represented. A direct agreement between the nominal decoder and the healthy N-CMAPSS engine is therefore not expected.

The fixed quantities collected in � define the nominal engine corresponding to $\mathbf { z } = \mathbf { 0 }$ . Initial values are assigned from conventional turbofan assumptions and engineering ranges. When applied directly to the healthy development data, this nominal configuration gives an overall reconstruction $R ^ { 2 } = - 3 1 . 0 3$ , reflecting the mismatch between the generic thermodynamic formulation and the specific engine represented in N-CMAPSS. A preliminary calibration is therefore performed to align the nominal reference with the healthy dataset before training the autoencoder [45].

Calibration procedure. Fifteen fixed quantities are identified using healthy samples from the development units, while all latent modifiers are held at their nominal value. The calibrated parameter vector is obtained from

Table 10  
Reconstruction $R ^ { 2 }$ of the nominal physical model before and after calibration on healthy development samples.
<table><tr><td>Variable</td><td>Initial</td><td>Calibrated</td><td>Variable</td><td>Initial</td><td>Calibrated</td></tr><tr><td> $T _ { 2 4 }$ </td><td>-160.8</td><td>0.312</td><td> $P _ { 2 4 }$ </td><td>-121.4</td><td>0.775</td></tr><tr><td> $T _ { 3 0 }$ </td><td>-45.8</td><td>-0.218</td><td> $P _ { s 3 0 }$ </td><td>-32.9</td><td>0.324</td></tr><tr><td> $T _ { 4 8 }$ </td><td>-0.661</td><td>0.051</td><td> $P _ { 4 0 }$ </td><td>-33.2</td><td>0.331</td></tr><tr><td> $T _ { 5 0 }$ </td><td>-78.5</td><td>-1.007</td><td> $P _ { 5 0 }$ </td><td>-11.8</td><td>0.465</td></tr><tr><td> $P _ { 1 5 }$ </td><td>0.486</td><td>0.919</td><td> $W _ { f }$ </td><td>0.167</td><td>0.175</td></tr><tr><td> $P _ { 2 }$ </td><td>0.9998</td><td>0.9998</td><td> $\dot { P _ { 2 1 } }$ </td><td>0.36945</td><td>0.91920</td></tr><tr><td></td><td></td><td></td><td>Overall</td><td>-31.03</td><td>0.534</td></tr></table>

Table 11

Initial and calibrated nominal-engine parameters. Parameters marked as bounded lie at, or within numerical tolerance of, one of the prescribed optimization bounds.
<table><tr><td>Parameter</td><td>Initial</td><td>Calibrated</td><td>Bounds</td><td>At bound</td></tr><tr><td> $\Pi _ { \mathbf { f } \mathbf { a } \mathbf { t } } ^ { 0 }$ </td><td>1.500</td><td>1.306</td><td>[1.10, 2.20]</td><td></td></tr><tr><td> $\Pi _ { \mathrm { L P C } } ^ { \mathrm { b } ^ { \mathrm { a u } } }$ </td><td>3.500</td><td>1.222</td><td>[1.05, 5.00]</td><td></td></tr><tr><td> $\Pi _ { \mathrm { H P C } } ^ { 0 }$ </td><td>11.429</td><td>15.594</td><td>[5.00, 25.00]</td><td></td></tr><tr><td> $\mathrm { B P R } ^ { \mathrm { \tilde { 0 } } }$ </td><td>9.000</td><td>10.235</td><td>[3.00, 15.00]</td><td></td></tr><tr><td> $\eta _ { \mathrm { f a n } } ^ { 0 }$ </td><td>0.880</td><td>0.918</td><td>[0.60, 0.95]</td><td></td></tr><tr><td> $\eta _ { _ \mathrm { L P C } } ^ { \mathrm { u } }$ </td><td>0.880</td><td>0.863</td><td>[0.60, 0.95]</td><td></td></tr><tr><td> $\eta _ { \mathrm { H P C } } ^ { \mathrm { v } }$ </td><td>0.880</td><td>0.897</td><td>[0.60, 0.95]</td><td></td></tr><tr><td> $\eta _ { \mathrm { H P T } } ^ { \mathrm { v } }$ </td><td>0.890</td><td>0.894</td><td>[0.60, 0.95]</td><td></td></tr><tr><td> $\eta _ { \mathrm { L P T } } ^ { \mathrm { 0 } }$ </td><td>0.890</td><td>0.950</td><td>[0.60, 0.95]</td><td>Yes</td></tr><tr><td>β0  $P _ { S ^ { \mathrm { { o m b } } } }$ </td><td>0.075</td><td>0.084</td><td>[0.02, 0.20]</td><td></td></tr><tr><td> $\beta _ { \mathrm { H P T } } ^ { \mathrm { u } }$ </td><td>0.047</td><td>0.120</td><td>[0.02, 0.12]</td><td>Yes</td></tr><tr><td> $\beta _ { \mathrm { L P T } } ^ { \mathrm { u } }$ </td><td>0.033</td><td>0.060</td><td>[0.01, 0.06]</td><td>Yes</td></tr><tr><td> $T _ { 4 1 } ^ { \mathrm { 0 } } \mathrm { [ \sf K ] }$ </td><td>1100</td><td>1360.6</td><td>[800, 1500]</td><td></td></tr><tr><td> $T _ { 4 1 } ^ { \dot { \prime } } \ [ \mathsf { K } ]$ </td><td>500</td><td>33.2</td><td>[0, 900]</td><td></td></tr><tr><td> $A _ { \mathrm { i n } } \ [ \mathsf { m } ^ { 2 } ]$ </td><td>8.296</td><td>6.259</td><td>[2.00, 20.00]</td><td></td></tr></table>

$$
\widehat { \Theta } = \arg \operatorname* { m i n } _ { \Theta \in [ \ell , u ] } \sum _ { t \in \mathcal { T } _ { \mathrm { H } } ^ { \mathrm { d e v } } } \left\| \mathcal { P } \left( \mathbf { w } _ { t } , \mathbf { p } _ { 0 } ( \Theta ) ; \Theta \right) - \mathbf { x } _ { t } \right\| _ { 2 } ^ { 2 } + \lambda _ { \Pi } \mathcal { L } _ { \Pi } ,\tag{46}
$$

with

$$
\mathcal { L } _ { \Pi } = \sum _ { t } \left[ \left( \frac { \Pi _ { \mathrm { H P T } } } { \Pi _ { \mathrm { H P T } } ^ { \mathrm { m a x } } } - 1 \right) _ { + } ^ { 2 } + \left( \frac { \Pi _ { \mathrm { L P T } } } { \Pi _ { \mathrm { L P T } } ^ { \mathrm { m a x } } } - 1 \right) _ { + } ^ { 2 } \right] ,\tag{47}
$$

where $\Pi _ { \mathrm { H P T } } ^ { \mathrm { m a x } } = 6 , \Pi _ { \mathrm { L P T } } ^ { \mathrm { m a x } } = 8$ , and $\lambda _ { \Pi } = 1$ . The penalty acts only when the corresponding expansion ratio exceeds the prescribed physical range.

The calibration set contains the first 15% of the cycles of each development unit, following the same healthy-window definition used for the operating-condition correction. The resulting pool is subsampled to 20 000 observations. Test units are excluded from the calibration. The optimization is performed using trust-region reflective least squares. The efect of this preliminary calibration on the reconstruction of each measured variable is reported in Table 10.

The calibration increases the overall nominal-model $R ^ { 2 } \mathrm { f r o m } - 3 1 . 0 3 \mathrm { t o } 0 . 5 3 4$ . The remaining mismatch is expected, since a single set of constants within a simplified textbook thermodynamic model is used to represent the healthy behaviour of multiple engines generated by a substantially more detailed simulation environment. The purpose of the calibration is therefore to establish a reasonable reference around which the latent modifiers can evolve. After training, the full physics-constrained autoencoder reaches $R ^ { 2 } = 0 . 9 9 9 3$ , showing that the inferred physical modifiers provide the additional degrees of freedom required to reproduce the observed engine response. The initial values, calibrated values, and optimization bounds of the nominal-engine parameters are reported in Table 11.

Table 12  
Correspondence between the N-CMAPSS measurement labels and the internal quantities used by the physical decoder.
<table><tr><td>N-CMAPSS variable</td><td>Decoder quantity</td><td>Physical interpretation</td></tr><tr><td> $T _ { 2 4 }$ </td><td> $T _ { 2 7 }$ </td><td>LPC outlet total temperature</td></tr><tr><td> $T _ { 3 0 }$ </td><td> $T _ { 3 }$ </td><td>HPC outlet total temperature</td></tr><tr><td> $T _ { 4 8 }$ </td><td> $T _ { 4 4 }$ </td><td>HPT outlet total temperature after HPT cooling remix</td></tr><tr><td> $T _ { 5 0 }$ </td><td> $T _ { 5 }$ </td><td>LPT outlet temperature after downstream flow remix</td></tr><tr><td> $P _ { 1 5 }$ </td><td> $P _ { 1 7 }$ </td><td>Bypass-duct total pressure</td></tr><tr><td> $P _ { 2 }$ </td><td> $P _ { 2 }$ </td><td>Fan-inlet total pressure</td></tr><tr><td> $P _ { 2 1 }$ </td><td> $P _ { 1 3 }$ </td><td>Fan-outlet total pressure</td></tr><tr><td> $P _ { 2 4 }$ </td><td> $P _ { 2 7 }$ </td><td>LPC-outlet total pressure</td></tr><tr><td> $P _ { s 3 0 }$ </td><td>Static  $P _ { 3 }$ </td><td>HPC-outlet static pressure, derived from the decoder total pressure  $P _ { 3 }$ </td></tr><tr><td> $P _ { 4 0 }$ </td><td> $P _ { 4 1 }$ </td><td>Combustor-outlet total pressure</td></tr><tr><td> $P _ { 5 0 }$ </td><td> $P _ { 7 }$ </td><td>LPT-outlet pressure channel, represented in the decoder downstream of the jet-pipe pressure loss</td></tr><tr><td> $W _ { f }$ </td><td> $W _ { f }$ </td><td>Fuel mass flow</td></tr><tr><td> $P _ { 4 5 }$ </td><td> $P _ { 4 4 }$ </td><td>Inter-turbine total pressure; virtual channel used only in the observability study</td></tr></table>

Three calibrated quantities reach their prescribed upper bounds: $\eta _ { \mathrm { L P T } } ^ { 0 } , \beta _ { \mathrm { H P T } } ^ { 0 } .$ , and $\beta _ { \mathrm { L P T } } ^ { 0 }$ . These values are obtained by jointly fitting the nominal decoder to healthy development data and may partly accommodate diferences between the simplified thermodynamic formulation and the N-CMAPSS engine. For the two cooling fractions, the weak measurement sensitivity discussed in Appendix D further limits how tightly their values are constrained. The optimizer can therefore drive $\beta _ { \mathrm { H P T } } ^ { 0 }$ and the other values toward the boundary of its admissible range while compensating for residual model mismatch elsewhere in the thermodynamic formulation.

## F. Correspondence between N-CMAPSS and Decoder Variables

The physical decoder follows conventional turbofan station numbering, whereas the measured variables retain the original N-CMAPSS nomenclature. Since the two conventions do not always use the same station labels for equivalent physical quantities, Table 12 reports their correspondence. This mapping is used throughout the decoder formulation to relate the internal thermodynamic states to the measured and reconstructed channels.