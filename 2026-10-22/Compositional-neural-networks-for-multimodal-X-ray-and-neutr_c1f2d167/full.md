# Compositional neural networks for multimodal X-ray and neutron scattering inversion

Valentin Munteanu<sup>a</sup>,<sup>∗</sup> Alexander Hinderhofer<sup>a</sup>, Dmitry

Lapkin<sup>a</sup>, Vladimir Starostin<sup>b</sup>, and Frank Schreiber<sup>a†</sup>

[a] Institute of Applied Physics, University of T¨ubingen,

Auf der Morgenstelle 10, 72076 T¨ubingen, Germany and

[b] Cluster of Excellence ”Machine Learning” - new perspectives for science,

University of T¨ubingen, Germany

## Abstract

X-ray and neutron scattering are indispensable for structural characterization across condensedmatter physics and materials science. Yet, because scattering experiments generally record intensities without phase information, distinct structures can yield similar scattering patterns, rendering the corresponding inverse problems non-unique. This ambiguity can be reduced by combining complementary measurements, but their integration into machine-learning frameworks remains challenging. Here we introduce a modular framework for probabilistic multi-measurement analysis based on prior-amortized flow matching. Single-observation posterior estimators are trained separately for individual observation types or physical probes and combined only at inference time through a compositional sampler. This late fusion approach allows pretrained single-observation models to be reused without dedicated joint retraining. As a case in point, we consider reflectometry, which provides depth-resolved, nanometer-scale information about thin films and interfaces. Our single-observation demonstrations span X-ray, time-of-flight neutron and polarized neutron reflectometry. In separate co-refinement studies, task-matched estimators enable homogeneous composition across multiple neutron contrasts and heterogeneous multimodal composition of X-ray and neutron measurements. Together, these results establish compositional flow matching as a modular approach to uncertainty-aware inversion and co-refinement, with potential applicability to other scattering problems in which multiple observations constrain a common physical structure while retaining measurement-specific parameters.

## INTRODUCTION

X-ray and neutron scattering are central to structural characterization in condensed-matter physics and materials science [1, 2]. Since scattering experiments generally record intensities without phase information, recovering the underlying structure can lead to non-unique or only weakly constrained inverse problems. Distinct structures or parameter combinations can therefore yield similar scattering patterns, so a single best-fitting structure may conceal alternative solutions that remain compatible with the data. Two distinct strategies address this ambiguity. First, probabilistic inversion uses Bayesian inference to characterize a posterior distribution over structural and instrumental parameters rather than returning a single structure. Second, measurements with complementary sensitivities can be combined so that parameters weakly identified by one observation may be constrained by another.

Posteriors arising from scattering inverse problems can be strongly correlated, non-Gaussian or multimodal, and joint inference enlarges the parameter space by introducing both quantities shared across measurements and quantities specific to an individual contrast, probe or instrument. These characteristics motivate flexible posterior approximations that can represent complex uncertainty across a range of possible observations. Simulation-based inference (SBI) [3] provides such an approach by training neural networks on simulated parameter–observation pairs to approximate posterior distributions. In the amortized setting, the main computational burden is incurred during training, after which inference for new measurements can be performed rapidly. This makes SBI attractive for repeated or highthroughput analysis in experimental science, including at X-ray and neutron facilities.

Amortized posterior estimators are nevertheless typically trained for a fixed observation model, and a network trained for one simulator cannot in general be combined directly with a network trained for another. Training a new joint model for every number and combination of measurements would undermine the modularity and computational reuse ofered by amortization. Compositional SBI provides a foundation for combining posterior information from multiple observations at inference time [4, 5]. Many joint inversion problems, however, require an additional structure of shared and local parameters. Some parameters describe a common physical system and must be shared across observations, whereas others describe a particular contrast, probe or instrument and remain local to an individual measurement. A modular multi-measurement framework must therefore combine separately trained estimators in a common parameter space while correcting the repeated contribution of the shared prior.

Flow matching [6, 7] provides a convenient basis for such a compositional construction. It trains a time-dependent vector field that transports samples from a simple base distribution to the target distribution along a prescribed probability path. Flow matching posterior estimation (FMPE) has been shown to be a competitive and scalable SBI method [8]. For the probability path used here, the learned vector field also determines a posterior score, which enables the information supplied by separately trained posterior estimators to be composed during sampling.

Specular X-ray and neutron reflectometry are indispensable techniques for resolving the depth-dependent structure and properties of thin films, multilayers and buried interfaces [9, 10]. Beyond its broad experimental relevance, reflectometry presents a demanding setting for this modular inference framework because it spans diferent probes, acquisition modes and observation structures. X-ray reflectometry (XRR) is sensitive to the electronic scattering length density (SLD), neutron reflectometry (NR) is sensitive to the nuclear SLD and can exploit isotope-dependent contrast, and polarized neutron reflectometry (PNR) further extends this sensitivity to magnetic properties through the spin dependence of the neutron interaction [11, 12]. Joint analysis of such measurements can constrain parameters that are weakly identified by an individual observation. The information gained nevertheless depends on the experimental design, measurement strategy, contrast engineering and model parameterization [13–16]. In practical reflectometry analysis, prior knowledge about plausible thickness ranges, expected material SLDs or instrumental conditions further constrains the solution space.

XRR is performed at synchrotron light sources [17, 18] or laboratory sources, whereas NR uses monochromatic or time-of-flight (TOF) reflectometers [19]. Energy-dispersive XRR and rotating-sample XRR geometries further illustrate the move towards rapid and parallel acquisition [20, 21]. Such developments increase the relevance of amortized analysis for in situ and time-resolved scattering experiments [22, 23]. PNR additionally requires polarization components such as polarizers, spin flippers and, depending on the experimental configuration, spin analyzers. In full spin-resolved PNR, four reflectivity cross sections are distinguished: the non–spin-flip channels $R ^ { \uparrow \uparrow }$ and $R ^ { \downarrow \downarrow }$ , and the spin-flip channels $R ^ { \uparrow \downarrow }$ and $R ^ { \downarrow \uparrow }$ , where the first and second arrows denote the incident and reflected neutron-spin states, respectively. Magnetization components parallel to the neutron-spin quantization axis contribute to the non–spin-flip channels, whereas components transverse to this axis couple the two spin states and generate spin-flip scattering. When the incident neutron polarization is selected but the reflected spin state is not resolved, the two measured intensities sum over the possible reflected spin states, such that $R ^ { \uparrow } = R ^ { \uparrow \uparrow } + R ^ { \uparrow \downarrow }$ and $R ^ { \downarrow } = R ^ { \downarrow \downarrow } + R ^ { \downarrow \uparrow }$ . For magnetization collinear with the neutron-spin quantization axis, the spin-flip cross sections vanish, and the measured intensities reduce to $R ^ { \uparrow } = R ^ { \uparrow \uparrow }$ and $R ^ { \downarrow } = R ^ { \downarrow \downarrow }$ , as assumed by the two-channel PNR model used for the experimental co-refinement examples in this work. Because these channels have fixed physical labels and ordering, they are treated as a single multichannel observation rather than as separate observations combined through compositional inference.

Previous machine-learning approaches to reflectometry have included point-estimator neural networks for XRR and NR and models incorporating prior knowledge [24–31]. Other studies have addressed high-throughput and autonomous-experiment settings [32–34], as well as variational-autoencoder analyses of PNR data [35]. Probabilistic approaches based on mixture-density networks or normalizing flows have further enabled uncertainty-aware singleobservation inference [36, 37]. In particular, prior-amortized neural posterior estimation (PANPE) conditions the generative model not only on the measured data but also on userprovided prior bounds, allowing experiment-specific prior information to be changed at inference time without retraining [37]. These developments make fast single-observation inference increasingly realistic. However, they do not provide a modular mechanism for combining estimators trained separately for diferent measurements, experimental contrasts or physical probes.

Here we introduce a prior-amortized compositional flow matching framework for probabilistic reflectometry analysis. This work makes two closely connected contributions. First, we adapt prior-amortized flow matching posterior estimation to a broad range of experimentally relevant XRR, NR and PNR settings. Second, building on recent compositional SBI approaches [5], we introduce compositional posterior sampling for reflectometry and extend the formalism to accommodate shared and local parameter blocks. Specifically, we derive the composition equations for separately trained estimators whose parameter spaces comprise a common block of shared parameters together with observation-specific local parameters. The resulting sampler combines compatible posterior estimators trained separately for individual observations or physical probes. The user specifies which parameters are shared and which remain local, the score contribution from each estimator is embedded into the resulting joint parameter space, and the repeated contribution of the shared prior is corrected during composition. In reflectometry, shared parameters can include layer thicknesses and interfacial roughnesses, whereas contrast-dependent SLDs, backgrounds, scale factors and probe-specific scattering properties may remain local to individual observations. This supports homogeneous composition, in which the same trained model is reused across several observations with a common parameterization as in multi-contrast NR, and heterogeneous composition, in which separately trained XRR and (P)NR models are coupled through shared structural parameters. Throughout this work, multimodal refers to the combination of diferent experimental modalities or probes. Although demonstrated here for reflectometry, the shared–local compositional construction is applicable more broadly to X-ray and neutron scattering problems in which multiple observations constrain a common physical model while retaining measurement-specific parameters.

The remainder of the paper is organized as follows. We first demonstrate the approach on a sequence of reflectometry problems of increasing complexity, progressing from singlecurve XRR and NR inference to multichannel PNR, multi-contrast NR co-refinement and multimodal XRR/PNR co-refinement. We then discuss the scope and limitations of the framework. The mathematical formulation, neural-network architecture, compositional sampler and training-data generation are provided in the Methods.

## RESULTS

## Bayesian compositional inference with shared and local parameters

We developed a modular Bayesian analysis framework in which posterior estimators trained for individual observation models can be reused and composed at inference time. For each observation model, a prior-amortized flow matching posterior estimator is trained on simulator-generated pairs $\{ ( \theta _ { i } , C _ { i } ) \} _ { i = 1 } ^ { N }$ . Here, $C = ( y , b , \eta )$ denotes the complete conditioning context, comprising the measured data y, user-supplied bounds b defining the prior domain and, where relevant, known acquisition quantities η. The trained estimator approximates the posterior $p ( \boldsymbol { \theta } \mid C )$ for admissible contexts within the support of its training distribution.

Flow matching approximates the conditional posterior by learning a time-dependent vector field that transports a Gaussian base distribution toward the target posterior distribution. Posterior samples are generated by drawing an initial state from the base distribution and integrating

$$
\frac { \mathrm { d } \theta _ { t } } { \mathrm { d } t } = v ( \theta _ { t } , t ; C )\tag{1}
$$

from $t = 0$ to $t = 1$ . For the Gaussian probability path used here, the learned vector field can be converted at each intermediate time into the score $s _ { t } ( \theta _ { t } ; C ) = \nabla _ { \theta _ { t } } \log p _ { t } ( \theta _ { t } \mid C )$ of the corresponding path distribution. Access to this score representation enables separately trained posterior estimators to be composed during inference.

We use composition when several measurements probe the same physical sample. Consider, for example, the multimodal co-refinement of an XRR and an NR measurement from the same structure. Parameters collected in $\theta ^ { \mathrm { s h } }$ , such as layer thicknesses and interfacial roughnesses, describe structure shared by both probes. The remaining parameters, $\theta _ { 1 } ^ { \mathrm { l o c } }$ and $\theta _ { 2 } ^ { \mathrm { l o c } }$ , are specific to one measurement, such as probe-dependent scattering-length densities or instrumental quantities. Assuming conditionally independent measurement likelihoods and compatible factorized priors with a common shared-parameter marginal, the joint Bayesian posterior can be rewritten in terms of the two single-observation posteriors represented by the trained estimators as

$$
p \big ( \theta ^ { \mathrm { s h } } , \theta _ { 1 } ^ { \mathrm { l o c } } , \theta _ { 2 } ^ { \mathrm { l o c } } \mid C _ { 1 } , C _ { 2 } \big ) \propto \frac { p _ { 1 } \big ( \theta ^ { \mathrm { s h } } , \theta _ { 1 } ^ { \mathrm { l o c } } \mid C _ { 1 } \big ) p _ { 2 } \big ( \theta ^ { \mathrm { s h } } , \theta _ { 2 } ^ { \mathrm { l o c } } \mid C _ { 2 } \big ) } { p ( \theta ^ { \mathrm { s h } } \mid b ^ { \mathrm { s h } } ) } .\tag{2}
$$

The two posteriors jointly constrain the shared structure, but each already contains one copy of the shared prior. Dividing by that prior removes the surplus copy so that prior information

is counted exactly once. A local parameter block requires no analogous correction because it occurs in only one single-observation posterior.

In the zero-noise limit, taking the logarithmic gradient of Eq. (2) gives the score-addition rule

$$
\nabla _ { \Theta } \log p ( \Theta \mid C _ { 1 } , C _ { 2 } ) = \nabla _ { \Theta } \log p _ { 1 } + \nabla _ { \Theta } \log p _ { 2 } - \nabla _ { \Theta } \log p \big ( \theta ^ { \mathrm { s h } } \mid \theta ^ { \mathrm { s h } } \big ) ,\tag{3}
$$

where each term is embedded in the global coordinates $\Theta = ( \theta ^ { \mathrm { s h } } , \theta _ { 1 } ^ { \mathrm { l o c } } , \theta _ { 2 } ^ { \mathrm { l o c } } )$ . This direct addition is not valid at intermediate path times because Gaussian perturbation does not commute with multiplication of probability densities. The intermediate-time composed score can therefore be written schematically as

$$
\begin{array} { r } { s _ { \mathrm { c o m p } } ( \Theta _ { t } , t ; C _ { 1 : 2 } ) = \nabla _ { \Theta _ { t } } \log p _ { 1 , t } + \nabla _ { \Theta _ { t } } \log p _ { 2 , t } - \nabla _ { \Theta _ { t } } \log p _ { t } \big ( \theta _ { t } ^ { \mathrm { s h } } \mid b ^ { \mathrm { s h } } \big ) + s _ { \mathrm { c o r r } } ( \Theta _ { t } , t ; C _ { 1 : 2 } ) , } \end{array}\tag{4}
$$

where $s _ { \mathrm { c o r r } }$ accounts for the mismatch between perturbing the joint posterior and combining the individually perturbed posteriors, and vanishes in the zero-noise limit.

We build on the full-covariance Gauss composition method introduced in Ref. [5], which uses Gaussian approximations to account for intermediate-time perturbation during posterior composition. Here, we extend this construction to posterior estimators defined on a common block of shared parameters together with observation-specific local parameter blocks. Retaining the full covariance allows the composition to account both for the information that each observation contributes to the shared parameters and for correlations between shared and observation-specific parameters. The resulting rule performs two coupled operations: it pools the shared score contributions according to their marginal precisions and adjusts each local score according to its correlation with the shared parameters. Let the denoising precision, that is, the inverse covariance of the Gaussian denoising approximation for observation $j ,$ be partitioned into shared and local blocks,

$$
\Lambda _ { j , t } = \left( { \Lambda _ { j , t } ^ { \mathrm { s s } } } \ { \Lambda } _ { j , t } ^ { \mathrm { s l } } \right) .\tag{5}
$$

For two observations, the resulting composed score is

$$
\begin{array} { r l } & { s _ { \mathrm { c o m p } , t } ^ { \mathrm { s h } } = \left( \bar { \Lambda } _ { 1 , t } + \bar { \Lambda } _ { 2 , t } - \Lambda _ { t } ^ { \mathrm { p r } } \right) ^ { - 1 } \left( \bar { \Lambda } _ { 1 , t } s _ { 1 , t } ^ { \mathrm { s h } } + \bar { \Lambda } _ { 2 , t } s _ { 2 , t } ^ { \mathrm { s h } } - \Lambda _ { t } ^ { \mathrm { p r } } s _ { t } ^ { \mathrm { p r } } \right) , } \\ & { s _ { \mathrm { c o m p } , j , t } ^ { \mathrm { l o c } } = s _ { j , t } ^ { \mathrm { l o c } } + \left( \Lambda _ { j , t } ^ { \mathrm { l l } } \right) ^ { - 1 } \Lambda _ { j , t } ^ { \mathrm { l s } } \left( s _ { j , t } ^ { \mathrm { s h } } - s _ { \mathrm { c o m p } , t } ^ { \mathrm { s h } } \right) , \qquad j = 1 , 2 , } \end{array}\tag{6}
$$

where $s _ { t } ^ { \mathrm { p r } }$ and $\Lambda _ { t } ^ { \mathrm { p r } }$ are the score and denoising precision of the perturbed shared prior, and

$$
\bar { \Lambda } _ { j , t } = \Lambda _ { j , t } ^ { \mathrm { s s } } - \Lambda _ { j , t } ^ { \mathrm { s l } } \left( \Lambda _ { j , t } ^ { \mathrm { l l } } \right) ^ { - 1 } \Lambda _ { j , t } ^ { \mathrm { l s } }\tag{7}
$$

is the Schur complement of the local block and represents the marginal precision contributed by observation j to the shared parameters.

The structure of Eqs. (6) and (7) has a direct physical interpretation. Each single observation posterior contributes to the shared parameters according to its marginal precision, that is, according to the information carried by the corresponding measurement after uncertainty in its local parameters has been accounted for. The negative prior contribution ensures that prior knowledge enters only once. Conversely, when the joint analysis shifts the shared structure away from the value preferred by one measurement alone, that measurement’s local score is adjusted along the shared–local correlation directions of its posterior. Retaining the full covariance is therefore important whenever structural and probe-specific parameters are correlated. The rule is exact when the individual posteriors and the shared prior are Gaussian and reduces to Eq. (3) as the perturbation vanishes. The full derivation and its extension to more than two observations are given in the Methods.

The composed score is converted back into a vector field and integrated as in Eq. (1) to generate joint posterior samples. We use this construction in two forms. In homogeneous composition, one trained estimator is reused for several observations with a common parameterization, as in multi-contrast NR co-refinement. In heterogeneous composition, separately trained estimators are coupled through a common shared block, as in joint XRR/(P)NR co-refinement. In both cases, the composed sampler returns samples over the shared parameters and every observation-specific local parameter. The mapping of compatible model coordinates into shared and local blocks is specified at inference time, so a dedicated joint estimator does not need to be trained for each supported combination of measurements.

Figure 1 summarizes the analysis strategy and its application to reflectometry. XRR and TOF-NR illustrate diferent acquisition modes for observations that each contain one reflectivity curve. In conventional XRR, the incident angle is varied at fixed wavelength, whereas in TOF-NR the incident angle is fixed and the wavelength dependence is resolved from neutron time of flight (Fig. 1a,b). Despite these experimental diferences, each measurement is supplied to its corresponding posterior estimator together with the selected prior bounds. An observation can also comprise a fixed number of physically labeled channels. Non–spin-flip PNR observations contain two channels, while full spin-resolved PNR observations contain four channels. Because these channels have fixed physical labels and ordering, they are supplied jointly to the embedding network as channels of a single observation. This fixed

multichannel treatment is distinct from compositional sampling, which combines separate observations, whether single- or multichannel, at inference time.

In practice, we use the trained posterior estimators in a two-stage inference procedure. First, raw flow matching sampling provides a fast amortized approximation to the posterior by integrating the learned single-observation or composed flow. Second, when likelihood-based refinement is desired, these samples initialize an ensemble MCMC sampler targeting the posterior defined by the explicit forward-model likelihood and the selected prior bounds. Unless stated otherwise, the main-text figures show posterior predictive simulations and profiles from this refined posterior, which we denote FMPE-MCMC. The Supplementary Information reports the corresponding raw FMPE results and corner-plot comparisons between the raw FMPE and FMPE-MCMC posteriors, providing a diagnostic of the amortized approximation before refinement.

The following sections first evaluate prior-amortized flow matching across single-observation XRR, NR, and PNR models. We then evaluate homogeneous and heterogeneous composition for multi-contrast NR and multimodal XRR/PNR co-refinement.

![](images/69a342e7429203befe60463c6d9c8f165308f1f6332958fb0c9f8d98c7e4d8d0.jpg)

![](images/00626bf790bd0141180715357fde4d80392ab3a37343ab41980ec8402e6a3231.jpg)

![](images/7975a472a67e829c629b5eba23d23ab8e513beaf65d27b68564089cbd95729dc.jpg)  
Figure 1: Prior-amortized flow matching posterior estimation and compositional co-refinement for reflectometry. (a) Schematic X-ray reflectometry (XRR) geometry. The reflectivity $R _ { \mathrm { X R R } }$ is measured by varying the incident angle $\alpha _ { i }$ at fixed wavelength λ, giving access to $R _ { \mathrm { X R R } } ( q ( \alpha _ { i } , \lambda ) )$ . (b) Schematic time-of-flight neutron reflectometry

$$
R _ { \mathrm { N R } } ( q ( \alpha _ { i } , \lambda ) )
$$

$$
y _ { j }
$$

$$
b _ { j }
$$

$$
C _ { j }
$$

$$
\Theta _ { t } ^ { ( j ) } = ( \theta _ { t } ^ { \mathrm { s h } } , \theta _ { j , t } ^ { \mathrm { l o c } } )
$$

from the joint posterior over the global parameter vector $\Theta = ( \theta ^ { \mathrm { s h } } , \theta _ { 1 } ^ { \mathrm { l o c } } , \dots , \theta _ { n } ^ { \mathrm { l o c } } )$ . (d)   
Schematic two-dimensional posterior representation for parameters $\theta _ { i }$ and $\theta _ { j }$ . The dashed   
lines indicate the user-supplied prior bounds, and the blue region denotes posterior density retained within the selected bounded parameter domain. Grey regions indicate density   
outside the selected bounds that is excluded by the bounded prior. (e) Posterior parameter samples can be propagated through the physical sample parameterization to obtain   
depth-dependent SLD profiles. In multimodal XRR/NR co-refinement, shared structural   
parameters determine the common layer geometry, while probe-specific local parameters determine the electronic SLD profile $\rho _ { e } ( z )$ and the nuclear SLD profile $\rho _ { n } ( z )$

## Single-curve XRR posterior inference for organic thin films

We first consider single-curve XRR posterior inference for organic thin films. This example uses a $\mathrm { P T C D I \mathrm { - } C _ { 3 } }$ film on a silicon substrate, while an analogous analysis of a $\mathrm { { C } _ { 6 0 } }$ film is provided in the Supplementary Information. Such films are representative of organic semiconductor systems, in which film thickness, density and interfacial morphology are important structural characteristics that can influence charge-transport behavior [38]. This tests the basic single-observation setting in which one trained prior-amortized model is applied to experimental curves using sample-specific prior bounds. For this application, we use a network trained for a standard one-layer slab parameterization. In addition to the structural parameters, the parameterization includes a q-shift ofset to account for possible alignment errors and an intensity scale factor to account for normalization uncertainty [26]. The prior bounds supplied at inference time are wider for thicknesses and roughnesses and narrower for the SLDs, reflecting the fact that the film material is known.

Figs. 2a and 2b show the experimental $\mathrm { P T C D I \mathrm { - } C _ { 3 } }$ curve together with posterior-predictive simulations from the FMPE-MCMC posterior and the corresponding SLD profiles. The posterior-predictive ensemble closely reproduces the experimental curve. The raw FMPE posterior already provides physically plausible samples, while the MCMC refinement targets the explicit likelihood and reduces residual amortization errors.

## NR with time-of-flight data and structured brush parameterizations

We next consider two neutron reflectometry examples that extend the prior-amortized framework beyond uniformly sampled and slab-style reflectivity curves.

The first example uses time-of-flight NR data measured on the D17 reflectometer at the Institut Laue-Langevin and published by Higgins et al. [39]. The sample is a two-layer PS:Bis-PCBM/Bis-PCBM structure on silicon, with air as the fronting medium. Bis-PCBM is a fullerene-derived electron acceptor used in organic photovoltaic blends, where polymer– acceptor miscibility and the resulting vertical composition and morphology can strongly influence device performance [40].

The second example uses NR data from a polystyrene (PS) polymer-brush structure originally reported by Hiotelis et al. [41] and distributed with the anaklasis package [42]. Polymer brushes are widely used to regulate protein adsorption, cell attachment and bacterial adhesion in biosensing and biomedical coatings [43]. Their function depends on the depthdependent polymer and solvent distributions at the interface, making the reconstructed brush height and solvent volume fraction profile physically relevant quantities. This example tests a structured physical parameterization in which the continuous brush profile is described by a small number of interpretable parameters and then discretized for reflectivity calculation.

For the time-of-flight example, we use a network trained for a standard two-layer slab parameterization with intensity scale and constant background as additional instrumental parameters. For the present inference workflow, TOF-NR provides a more demanding test than uniformly sampled angle-scan data because experimental curves can have non-uniform q-spacing and pointwise resolution information. The neural network is conditioned on a scalar linear resolution-smearing coeficient, whereas the MCMC refinement uses the pointwise resolution values available for the experimental data. This diference reflects the roles of the two stages: the scalar coeficient provides an instrument-agnostic approximation suitable for broad simulation-based training, while the likelihood-based refinement can use the instrumentspecific resolution information directly. The example therefore tests whether the amortized model remains useful when the neural-network input uses a simplified resolution description but the final posterior refinement uses a more detailed experimental likelihood.

For the polymer-brush example, the model parameters describe the brush height, the polymer volume fraction at the grafting interface, the brush-profile exponent, the thickness and SLD of a thin layer between the fronting medium and the brush (reactive layer), and the relevant interfacial roughnesses. For the experimental structure considered here, the fronting medium is quartz, the backing medium is deuterated toluene, and the additional thin layer corresponds to water. The continuous brush profile and its discretization are defined in Methods. This case is representative of practical reflectometry models that are not simple stacks of independent slabs, but instead contain structured parameterizations derived from physical constraints or soft-matter theory.

Fig. 2c shows the time-of-flight NR data together with posterior-predictive simulations from the FMPE-MCMC posterior, and Fig. 2d shows the corresponding SLD profiles. Fig. 2e shows the analogous result for the polymer-brush example, with the reconstructed solvent volume-fraction profiles shown in Fig. 2f. These results indicate that our posterior estimation method can be applied both to experimental TOF data and to structured model classes. The corresponding raw FMPE posterior-predictive curves and posterior corner plots are provided in the Supplementary Information.

Single NR observation RNR  
![](images/5ddc1705aefec9d72bae2a543de90cd7c6f4bad851fc2cde4adb17c7d21f1c6d.jpg)

![](images/200d5bf37954fd387445ec597e4ae3638028c751c0d3819c731f357c8c802c93.jpg)

![](images/69ad3cb8cda75f177019ce47e7515f2d329e5c954cb2d8065da9dc4f636ac428.jpg)

![](images/e2c6b33e4b05850e64545f1f347bd4b526acee24b6408a99f65c514404476461.jpg)

![](images/07d09bc1925b2495825edb56b2e41401658f335cf139a66f7e12e832c5906ada.jpg)

![](images/9d002c356194b781eca26163e9d9d21822498e6d7dbe478ded65150f3cf6d156.jpg)  
Figure 2: Single-curve posterior inference for XRR, TOF-NR and polymer-brush NR data. (a) Experimental XRR curve for a one-layer $\mathrm { P T C D I \mathrm { - } C _ { 3 } }$ film on a Si substrate, shown together with posterior-predictive simulations generated from FMPE-MCMC posterior samples. (b) SLD profiles for the $\mathrm { P T C D I \mathrm { - } C _ { 3 } }$ film reconstructed from the FMPE-MCMC posterior. (c) Experimental TOF-NR curve for a two-layer PS:Bis-PCBM/Bis-PCBM structure on Si, with data from Ref. [39], shown together with posterior-predictive simulations generated from FMPE-MCMC posterior samples. (d) SLD profiles reconstructed from the FMPE-MCMC posterior for the TOF-NR data. (e) Experimental NR curve for the polymer-brush structure, with data from Ref. [42], shown together with posterior-predictive simulations generated from FMPE-MCMC posterior samples. (f) Solvent volume-fraction profiles reconstructed from the FMPE-MCMC posterior for the polymer-brush structure.

## Full spin-resolved PNR

We use full spin-resolved PNR as a proof-of-concept for multichannel posterior estimation. Such measurements are particularly relevant for magnetic heterostructures with non-collinear or depth-dependent magnetization, including spin valves, exchange-spring systems and exchange-biased structures [44]. In such systems, the non–spin-flip channels encode information about the nuclear SLD and the magnetization component parallel to the neutronspin quantization axis, whereas the spin-flip channels provide information about transverse magnetization components. In contrast to standard XRR or NR, a full spin-resolved PNR observation therefore consists of four coupled reflectivity channels: two non–spin-flip channels and two spin-flip channels. In our framework these four curves are supplied jointly to the embedding network as a single multichannel observation, allowing the posterior estimator to learn correlations between nuclear, magnetic and angular parameters from the full spin-resolved signal.

For this example, we trained a model with a simplified two-layer magnetic parameterization including layer thicknesses, nuclear SLDs, magnetic SLDs and magnetization angles. Trainingdata generation for this setting used a batched 4 × 4 transfer-matrix implementation in PyTorch. Interfacial roughnesses are not included in this proof-of-concept model because of the limitations of the transfer-matrix treatment discussed in the Methods section. We apply the model to synthetic full spin-resolved PNR data emulating a Co/Fe bilayer on a silicon backing medium, with the Co magnetization oriented at 45<sup>◦</sup> and the Fe magnetization oriented at 120<sup>◦</sup> with respect to the applied magnetic field. Resolution smearing $d q / q = 0 . 0 5$ was applied to the synthetic data.

Fig. 3a shows the four spin-resolved reflectivity channels together with posterior-predictive simulations from the FMPE-MCMC posterior. Figs. 3b and 3c show the reconstructed nuclear and magnetic SLD profiles and the magnetization-angle profiles. The posterior concentrates around the parameters used to generate the synthetic data, showing that the flow matching posterior estimator can represent the coupled uncertainty in nuclear, magnetic and angular parameters in this simplified full spin-resolved PNR setting. This result should be viewed as a proof of concept for spin-resolved posterior inference rather than as a complete treatment of experimental full-PNR analysis with roughness and additional instrumental efects.

## Compositional co-refinement of multiple-contrast NR data

We next demonstrate compositional sampling for joint posterior inference from multiplecontrast NR data. Contrast variation is essential in many areas of soft-matter and interfacial science because it allows selected regions or molecular components of a sample to be highlighted without substantially changing the underlying structure. Diferent contrasts can be generated by varying the isotopic composition of the surrounding solvent or by selectively deuterating sample components such as polymers, lipids or proteins. Combining measurements at complementary contrasts improves sensitivity to diferent regions of the interfacial structure and helps constrain layer composition and solvent content [45, 46]. In the present example, the contrast is varied through the composition of the backing solvent. This example uses a single neural network model trained on individual observations and composes the corresponding posterior information across three solvent contrasts at inference time. The model uses a two-layer slab parameterization with additional solvent volume-fraction parameters, which quantify penetration of the backing solvent into the layers.

The experimental data were published along with the refnx package [47]. The sample consists of a silicon oxide layer and a polymer layer, with silicon as the fronting medium and a solvent as the backing medium. Three solvent contrasts are used: $\mathrm { D _ { 2 } O , H _ { 2 } O }$ and a siliconoxide-matched mixture of $\mathrm { D _ { 2 } O }$ and $\mathrm { H _ { 2 } O }$ (in a 59:41 ratio), with an SLD of $3 . 5 \times 1 0 ^ { - 6 } \mathring { \mathrm { A } } ^ { - 2 }$ For compactness, this contrast is labelled CM in the figures. In the joint inference, the structural parameters are shared across all contrasts, whereas the backing-medium SLDs, intensity scale factors and backgrounds are treated as observation-specific parameters. This shared/local split is the key reflectometry-specific extension of the compositional setting, because diferent measurements of the same sample generally share the structure but not all contrast or instrumental parameters.

Fig. 3d shows the three measured NR curves together with posterior-predictive simulations from the FMPE-MCMC posterior, and Fig. 3e shows the corresponding contrast-dependent SLD profiles. The joint posterior enforces a single structural description across all contrasts while allowing each contrast to retain its own solvent and instrumental parameters. This result shows that a model trained only on single observations can be reused for multi-contrast co-refinement, avoiding the need to train a separate network for each possible number or combination of contrasts. The corresponding raw-FMPE posterior-predictive curves and posterior corner plots are provided in the Supplementary Information.

![](images/297ed5273fc1c5f92196c383c8dbbc5ddb1834b5fdc9587d5f14819fb2ea37e6.jpg)

![](images/9bcf6916385847cabfff86fecc0d282fefeb908bf2d704559a335be26757181d.jpg)

![](images/1783b8b9db181a4dad565c3eb94c4482b6ee6b26cf2b629795e0487ab39a20f9.jpg)

![](images/0f774add6eca236be05e805a7a130cd6c8fd0f0fb22320cbfe3d9af152fed40f.jpg)  
Figure 3: Multichannel posterior inference for full spin-resolved PNR and compositional co-refinement of multiple-contrast NR data. (a) Synthetic full spin-resolved PNR data, shown together with posterior-predictive simulations generated from FMPE-MCMC posterior samples. The curves corresponding to diferent spin channels are shifted vertically for clarity. (b) Nuclear and magnetic SLD profiles reconstructed from the FMPE-MCMC posterior. (c) Magnetization-angle profiles reconstructed from the FMPE-MCMC posterior. (d) Experimental NR curves corresponding to three solvent contrasts, $\mathrm { D _ { 2 } O , H _ { 2 } O }$ and CM, with data from Ref. [47], shown together with posterior-predictive simulations generated from FMPE-MCMC posterior samples. The curves are shifted vertically for clarity. (e) SLD profiles for the three solvent contrasts reconstructed from the FMPE-MCMC posterior.

## Multimodal co-refinement of XRR and PNR data

Finally, we consider a multimodal co-refinement setting in which diferent experimental probes are analyzed jointly using diferent trained posterior estimation models. This setting is important for reflectometry because XRR and (P)NR are sensitive to diferent material contrasts while sharing the same underlying layer geometry. We therefore compose an XRR model and a non–spin-flip PNR model at inference time, assigning layer thicknesses and interfacial roughnesses as shared parameters and retaining probe-specific SLD and instrumental parameters as local parameters.

The XRR model includes both the real and imaginary parts of the electronic SLD, because absorption efects cannot be neglected for the high-density materials considered here. The PNR model includes nuclear and magnetic SLDs and a channel-mixing coeficient α to account for imperfect polarization. Both models include intensity scale and background parameters. In the main text, we apply the approach to XRR and PNR measurements of a Pt/Fe structure on MgO, published along with the anaklasis package [42]. An analogous Pt/Co structure on $\mathrm { { A l } _ { 2 } \mathrm { { O } _ { 3 } } }$ is analyzed in the Supplementary Information. Pt/Fe and Pt/Co heterostructures are representative heavy-metal/ferromagnet systems investigated in connection with phenomena including interfacial magnetic anisotropy and spin–orbit torques [48, 49]. For the main inference, the SLD parameters are fixed to their theoretical values, following the corresponding classical analysis; an additional inference with narrow SLD prior bounds is provided in the Supplementary Information.

Fig. 4a shows the measured XRR and PNR curves for the Pt/Fe structure together with posterior-predictive simulations from the FMPE-MCMC posterior. Fig. 4b shows the reconstructed real and imaginary electronic SLD profiles, and Fig. 4c shows the corresponding nuclear and magnetic SLD profiles. The results illustrate heterogeneous compositional sampling across diferent trained models and diferent experimental modalities. In this setting, XRR and PNR jointly constrain the shared structural parameters while preserving probespecific information. The analogous Pt/Co co-refinement, provided in the Supplementary Information, shows the same workflow for a second heavy-metal/ferromagnet structure.

![](images/3e33793b6fb093f19e9ef4986dde63101f99281dc30c40a46e634bf7c635ca62.jpg)  
Figure 4: Multimodal compositional co-refinement of XRR and PNR data. (a) Experimental XRR and PNR data for the Pt/Fe structure, with data from Ref. [42], shown together with posterior-predictive simulations generated from FMPE-MCMC posterior samples. (b) Reconstructed real and imaginary electronic SLD profiles. (c) Reconstructed nuclear and magnetic SLD profiles.

## DISCUSSION

Probabilistic inversion of scattering data requires methods that can represent uncertainty and combine information from complementary measurements. Here, we introduced prioramortized flow matching posterior estimation for X-ray and neutron reflectometry together with an inference-time compositional strategy for co-refinement. Compared with pointestimation approaches, the method directly returns posterior samples over physical and instrumental parameters, which is particularly valuable for inverse problems in which diferent parameter combinations can produce similar reflectivity curves. Flow matching provides a continuous-time generative formulation based on a learned vector field. For the Gaussian probability path used here, this vector field also provides access to a score representation, enabling posterior information from multiple observations to be composed at inference time. More generally, the trained FMPE models can cover broad parameter and prior-bound ranges and can be applied to diferent experimental datasets within the support of their training distributions. Depending on the measurement setting, an FMPE model may process a single reflectivity curve, a fixed set of physically labeled channels within one observation, as in PNR, or contribute to the compositional analysis of separate observations, as in isotope-contrast series and multimodal XRR/NR measurements.

A central contribution is the compositional sampling strategy for reflectometry corefinement. Building on previous compositional SBI formulations, we derive a full-covariance composition rule for estimators defined on a common block of shared parameters together with observation-specific local parameter blocks. This structure is particularly relevant to reflectometry, where some parameters are shared because they describe the same sample geometry, whereas others are local because they describe contrast conditions, instrumental efects or probe-specific scattering properties. The shared–local formulation makes it possible to combine compatible single-observation posterior estimators for multi-contrast NR and heterogeneous XRR/(P)NR co-refinement. Beyond the reflectometry cases considered here, the approach may also be useful for probabilistic analysis of other multi-measurement experiments in materials science. Many experimental workflows combine observations that probe a common latent sample state while retaining measurement-specific contrast or instrumental parameters. Examples include measurements collected at diferent contrasts, incident energies, instruments or experimental modalities, as well as temperature- or time-dependent series

in which the shared and evolving quantities can be specified explicitly. In such settings, a single-observation posterior estimator could, in principle, be reused as a modular component of joint inference. This modularity separates the reusable task of developing robust single-observation models from the experiment-specific task of composing the measurements available for a particular sample.

The examples show that the same framework can be applied to several experimentally relevant settings: single-curve XRR, time-of-flight NR, structured parameterizations such as polymer brushes, PNR and multi-measurement co-refinement. In many cases, raw FMPE samples already provide useful posterior approximations, whereas MCMC refinement initialized from those samples targets the posterior defined by the explicit forward-model likelihood and prior. This two-stage procedure is useful in practice because the amortized model identifies plausible regions of parameter space, while the refinement step reduces residual approximation errors and incorporates experimental likelihood details that may not be fully represented during neural network training.

Despite these promising results, the framework also has several limitations. Posterior samples from the amortized model are approximate, and their accuracy depends on the training ranges, the parameter dimensionality and the supplied prior bounds. For full spin-resolved PNR, the present demonstration uses synthetic data and a simplified parameterization without interfacial roughness. In addition, flow matching inference requires numerical integration of an ODE, which can make raw sampling slower than direct sampling from many discrete normalizing-flow models. Compositional accuracy additionally depends on the quality of the individual score estimates and covariance approximations, as well as on compatible definitions of the shared parameters, priors and probability paths across the contributing estimators. These limitations suggest several directions for further development. Approximation errors may be reduced through advances in flow matching objectives, probability-path design and training strategies [50], or through post-training refinement methods that incorporate feed back from the forward simulator [51]. Moreover, recent average-velocity and caching-based approaches provide potential routes to accelerating flow-based sampling [52, 53]. Future work could improve the compositional sampler through more accurate intermediate time covariance approximations and extensions to more general parameter-sharing patterns.

Overall, the results show that flow matching provides a flexible framework for probabilistic reflectometry inversion and that compositional sampling can extend single-observation posterior estimators to modular co-refinement. Although demonstrated here for reflectometry, the shared–local construction could be transferred to other multi-measurement scattering problems in which several observations constrain a common physical structure while retaining contrast-, probe- or instrument-specific parameters. This capability is relevant to highthroughput and multi-probe X-ray and neutron studies, where scalable uncertainty-aware analysis is increasingly important for automated experimental workflows.

## METHODS

## Bayesian reflectometry and prior amortization

Let $\theta$ denote the vector of physical and instrumental parameters of a reflectometry model and let $y$ denote measured reflectivity data. Depending on the parameterization, $\theta$ may include layer thicknesses $d ,$ interfacial roughnesses $\sigma$ , electronic, nuclear or magnetic SLDs $\rho _ { ; }$ magnetization angles $\psi .$ , solvent volume fractions $\phi ,$ and instrumental parameters such as intensity scale factors $( r _ { \mathrm { s c } } )$ , q-ofsets $( q _ { \mathrm { s h } } )$ , or constant backgrounds.

We denote by $b$ the prior-conditioning variables supplied at inference time. In this work, these are the lower and upper bounds $[ b _ { \mathrm { m i n } } , b _ { \mathrm { m a x } } ]$ for each parameter. These bounds encode experiment-specific prior knowledge and define a conditional prior $p ( \theta \mid b )$

We use $\eta$ to denote optional known experimental variables that are supplied to the neural network. For the NR models considered here, η contains the linear smearing coeficient $d q / q$ and for the XRR models $\eta$ is empty. The posterior of interest is then

$$
p ( \theta \mid y , b , \eta ) \propto p ( y \mid \theta , \eta ) p ( \theta \mid b ) .\tag{8}
$$

In the prior-amortized setting, the neural network is trained across a broad distribution of prior bounds and observations rather than for a single fixed prior.

Throughout the paper, we write

$$
C = ( y , b , \eta )\tag{9}
$$

for the complete conditioning context supplied to the neural posterior estimator. When $\eta$ is empty, this reduces to $C = ( y , b )$ . The goal of the generative model is to produce approximate posterior samples

$$
\theta \sim p ( \theta \mid C ) .\tag{10}
$$

## Flow matching posterior estimation

Flow matching [8] parameterizes a time-dependent conditional vector field $v ( \theta , t ; C )$ that transports noise sampled from a simple base distribution $p _ { 0 } ( \theta )$ at the path time $t = 0$ to the target posterior $p ( \boldsymbol { \theta } \mid C )$ at $t = 1$ , using a neural network. The associated probability path

$\{ p _ { t } ( \theta \mid C ) \} _ { t \in [ 0 , 1 ] }$ satisfies the continuity equation

$$
\partial _ { t } p _ { t } ( \theta \mid C ) + \nabla _ { \theta } \cdot \left( p _ { t } ( \theta \mid C ) \upsilon ( \theta , t ; C ) \right) = 0 .\tag{11}
$$

A direct regression objective based on the marginal vector field of $p _ { t } ( \theta \mid C )$ is generally intractable, since this is not available in closed form. Flow matching circumvents this dificulty by introducing a family of conditional probability paths, indexed by an auxiliary variable $z ,$ together with corresponding conditional target vector fields $u _ { t } ( \theta \mid z , C )$ . The resulting conditional flow matching objective is

$$
\mathcal { L } _ { \mathrm { C F M } } = \mathbb { E } _ { t , z , \theta _ { t } } \left[ \| v ( \theta _ { t } , t ; C ) - u _ { t } ( \theta _ { t } \mid z , C ) \| ^ { 2 } \right] ,\tag{12}
$$

where $\theta _ { t } \sim p _ { t } ( \theta \mid z , C )$ . Lipman et al. [6] showed that this objective provides an unbiased gradient estimator for the corresponding marginal flow matching objective, while subsequent work generalized the framework and emphasized the influence of the chosen path family on training and sampling behavior [7, 54].

For posterior estimation, the target endpoint distribution is $p ( \theta \mid C )$ . Accordingly, the neural network is trained as a conditional generative model, with the context C supplied as additional input.

In the two-sided path construction [7], the auxiliary variable is $z = ( \theta _ { 0 } , \theta _ { 1 } )$ , where $\theta _ { 0 } \sim p _ { 0 }$ and $\theta _ { 1 } \sim p ( \theta \mid C )$ . For the straight-line path, the interpolation is

$$
\begin{array} { r } { \theta _ { t } = ( 1 - t ) \theta _ { 0 } + t \theta _ { 1 } , } \end{array}\tag{13}
$$

and the corresponding conditional target vector is

$$
u _ { t } ( \theta _ { t } \mid \theta _ { 0 } , \theta _ { 1 } , C ) = \theta _ { 1 } - \theta _ { 0 } .\tag{14}
$$

This straight-path construction is the simplest and most widely used choice in rectified-flow and two-sided conditional flow matching variants [7, 55].

Flow matching is not intrinsically restricted to Gaussian base distributions or to difusion probability paths. In this work, however, we use a base noise variable $\epsilon \sim \mathcal { N } ( 0 , I )$ , for which the chosen interpolation defines a Gaussian conditional probability path. Such paths admit both vector field and score representations, providing a connection between deterministic flow and score-based formulations [54]. The path can be written as

$$
\theta _ { t } = m _ { t } \theta _ { 1 } + \sigma _ { t } \epsilon .\tag{15}
$$

In our implementation, we use

$$
m _ { t } = t , \qquad \sigma _ { t } = 1 - t + \sigma _ { \operatorname* { m i n } } , \qquad \sigma _ { \operatorname* { m i n } } = 1 0 ^ { - 3 } ,\tag{16}
$$

where the small endpoint regularization prevents numerical singularities near $t = 1$

For a Gaussian conditional probability path of the form in Eq. (15), the vector field $v ( \theta _ { t } , t ; C )$ and the score

$$
s ( \theta _ { t } , t ; C ) : = \nabla _ { \theta _ { t } } \log p _ { t } ( \theta _ { t } \mid C )\tag{17}
$$

are related by [56]

$$
s ( \theta _ { t } , t ; C ) = \frac { m _ { t } v ( \theta _ { t } , t ; C ) - \dot { m } _ { t } \theta _ { t } } { \sigma _ { t } \left( \dot { m } _ { t } \sigma _ { t } - m _ { t } \dot { \sigma } _ { t } \right) } .\tag{18}
$$

For the coeficients in Eq. (16), $\dot { m } _ { t } = 1$ and $\dot { \sigma } _ { t } = - 1$ , such that

$$
s ( \theta _ { t } , t ; C ) = \frac { t v ( \theta _ { t } , t ; C ) - \theta _ { t } } { ( 1 - t + \sigma _ { \operatorname* { m i n } } ) ( 1 + \sigma _ { \operatorname* { m i n } } ) } .\tag{19}
$$

Related connections between deterministic flows and score-based stochastic sampling are discussed in Ref. [57].

Once the model has been trained, approximate posterior samples are obtained by integrating the neural ordinary diferential equation (ODE) [58]

$$
\frac { d \theta _ { t } } { d t } = v ( \theta _ { t } , t ; C ) , \qquad \theta _ { 0 } \sim p _ { 0 } ,\tag{20}
$$

from $t = 0 \mathrm { ~ t o ~ } t = 1$ , using either a fixed number of integration steps (e.g. Euler solver) or an adaptive solver such as Dormand–Prince (dopri5) from the torchdiffeq package, which we use as the default solver.

## Neural network architecture

The conditional vector field $v ( \theta _ { t } , t ; C )$ is represented by a neural network consisting of two main components: an embedding network for the reflectometry observation and a residual multilayer perceptron (MLP) that outputs the vector field. The neural network architecture is illustrated in Fig. 5.

The observation y is represented as multichannel one-dimensional data. Standard XRR and NR observations contain one reflectivity channel, non–spin-flip PNR observations contain two reflectivity channels, and full spin-resolved PNR observations contain four reflectivity

![](images/25dfe451e3d0fc910b86e00e3e92f7f45885d1d26bdb917dbc3b7cf962952cbe.jpg)  
Figure 5: Neural network architecture. The reflectivity data is first mapped to a fixed-dimensional representation by an embedding network. This embedding is processed by a residual MLP with GELU activations and layer-normalized residual blocks. Conditioning is applied through FiLM layers using t, $\theta _ { t } ,$ the prior bounds $b ,$ and optional auxiliary variables $\eta .$ . For the neutron-reflectometry models considered here, $\eta$ includes the linear smearing coeficient $d q / q$ . The network outputs the conditional vector field $v ( \theta _ { t } , t ; C )$  
channels. The corresponding $q$ values are supplied as an additional input channel. These inputs are mapped to a fixed-dimensional representation by a one-dimensional convolutionbased embedding network adapted from previous work in the field [31, 37].

The embedded representation is then processed by a residual MLP with 8 residual blocks of width 1024, GELU activations, and layer normalization. Conditioning is applied through feature-wise linear modulation (FiLM) [59]: the reflectivity embedding is provided as the main input to the residual MLP, while $t , \theta _ { t }$ , the prior bounds $b ,$ and any optional auxiliary conditioning variables $\eta$ are injected through FiLM layers in each residual block. For the neutron reflectometry applications considered here, $\eta$ includes the linear smearing coeficient $d q / q$ . The final linear layer is initialized with zero weights.

## Compositional sampling for multiple observations

Many reflectometry studies collect several measurements of the same underlying sample, for example NR data recorded under diferent solvent contrasts or complementary measurements obtained from diferent probes. The goal is to perform joint posterior inference across all observations while using models trained on single-observation tasks. Compositional posterior inference provides a way to do this. In SBI, composition rules of this type were introduced for score-based models by Ref. [4] and were later reformulated in a more stable framework by Ref. [5]. Our implementation was informed by the public sbi ecosystem [60], but the formulation used here adapts the tall-data construction, together with the full-covariance Gauss composition of Ref. [5], to flow matching posterior estimators and to the shared/local parameter structure required for reflectometry co-refinement.

We start with the standard tall-data identity. If observations $y _ { 1 : n } = ( y _ { 1 } , \ldots , y _ { n } )$ are conditionally independent given a common parameter vector $\theta ,$ and if each single-observation posterior is defined with respect to the same prior $p ( \theta \mid b )$ , then

$$
p ( \boldsymbol { \theta } \mid \boldsymbol { y } _ { 1 : n } , \boldsymbol { b } ) \propto p ( \boldsymbol { \theta } \mid \boldsymbol { b } ) ^ { 1 - n } \prod _ { j = 1 } ^ { n } p ( \boldsymbol { \theta } \mid \boldsymbol { y } _ { j } , \boldsymbol { b } ) .\tag{21}
$$

The factor $p ( \theta \mid b ) ^ { 1 - n }$ corrects for the repeated prior contribution: multiplying n singleobservation posteriors counts the prior n times, whereas the joint posterior should contain it only once.

Reflectometry co-refinement requires an extension of this identity because not all parameters are common to all measurements. Some parameters, such as layer thicknesses and interfacial roughnesses, may be shared across observations, whereas others, such as contrastdependent SLDs, scale factors, backgrounds, or probe-specific quantities, may be local to individual observations. We therefore write the global parameter vector as

$$
\Theta = \left( \theta ^ { \mathrm { s h } } , \theta _ { 1 } ^ { \mathrm { l o c } } , \dots , \theta _ { n } ^ { \mathrm { l o c } } \right) ,\tag{22}
$$

where $\theta ^ { \mathrm { s h } }$ denotes the shared block and $\theta _ { j } ^ { \mathrm { l o c } }$ denotes the block local to observation $j$

For each observation, the complete context supplied to the single-observation posterior estimator is

$$
C _ { j } = ( y _ { j } , b _ { j } , \eta _ { j } ) , \qquad b _ { j } = \left( b ^ { \mathrm { s h } } , b _ { j } ^ { \mathrm { l o c } } \right) ,\tag{23}
$$

where $y _ { j }$ is the measured reflectometry observation, $b ^ { \mathrm { s h } }$ contains the prior bounds for the shared parameters, $b _ { j } ^ { \mathrm { l o c } }$ contains the prior bounds for the local parameters of observation $j ,$ , and $\eta _ { j }$ denotes optional known observation-specific conditioning variables. We write $C _ { 1 : n } = \{ C _ { 1 } , \ldots , C _ { n } \}$ for the full set of contexts used in the joint analysis.

We assume conditional independence of the observations given the global parameter vector,

$$
p ( y _ { 1 : n } \mid \Theta , \eta _ { 1 : n } ) = \prod _ { j = 1 } ^ { n } { p _ { j } \left( y _ { j } \mid \theta ^ { \mathrm { s h } } , \theta _ { j } ^ { \mathrm { l o c } } , \eta _ { j } \right) } ,\tag{24}
$$

where the subscript on $p _ { j }$ emphasizes that the likelihood functions may difer between observations, as when diferent probes, instruments, or forward models are combined, and each likelihood depends only on the parameters used by its observation. Together with the factorized global prior

$$
p \big ( \Theta \mid b ^ { \mathrm { s h } } , b _ { 1 : n } ^ { \mathrm { l o c } } \big ) = p \big ( \theta ^ { \mathrm { s h } } \mid b ^ { \mathrm { s h } } \big ) \prod _ { j = 1 } ^ { n } p \big ( \theta _ { j } ^ { \mathrm { l o c } } \mid b _ { j } ^ { \mathrm { l o c } } \big ) ,\tag{25}
$$

the target joint posterior can be written as

$$
p ( \Theta \mid C _ { 1 : n } ) \propto p \big ( \theta ^ { \mathrm { s h } } \mid b ^ { \mathrm { s h } } \big ) ^ { 1 - n } \prod _ { j = 1 } ^ { n } p _ { j } \big ( \theta ^ { \mathrm { s h } } , \theta _ { j } ^ { \mathrm { l o c } } \mid C _ { j } \big ) .\tag{26}
$$

Equation (26) is the shared/local analogue of Eq. (21). The correction is applied only to the shared prior because the shared prior appears once in every single-observation posterior factor. In contrast, each local block $\theta _ { j } ^ { \mathrm { l o c } }$ appears in only one factor and therefore requires no repeated-prior correction. Equation (26) places no factorization requirement on the single-observation posteriors themselves: the likelihood generally correlates the shared and local parameters of each observation, and the composition below retains these correlations.

The composition is performed in the parameter coordinates used by the neural posterior estimators. For the shared parameters, all estimators entering the composition must use the same coordinate representation. At path time $t ,$ the global latent state is written as

$$
\varTheta _ { t } = \left( \theta _ { t } ^ { \mathrm { s h } } , \theta _ { 1 , t } ^ { \mathrm { l o c } } , \ldots , \theta _ { n , t } ^ { \mathrm { l o c } } \right) , \qquad \varTheta _ { t } ^ { ( j ) } = P _ { j } \Theta _ { t } = \left( \theta _ { t } ^ { \mathrm { s h } } , \theta _ { j , t } ^ { \mathrm { l o c } } \right) ,\tag{27}
$$

where the selection matrix $P _ { j }$ extracts the coordinates used by the single-observation model for observation $j ,$ its transpose $P _ { j } ^ { \mathsf { T } }$ embeds observation-j quantities into the global coordinates by zero padding, and $P _ { \mathrm { s h } }$ analogously selects the shared block. The corresponding singleobservation flow matching model predicts the learned vector field

$$
v _ { j } \left( \Theta _ { t } ^ { ( j ) } , t ; C _ { j } \right) .\tag{28}
$$

For compactness, we denote the associated learned score estimate by

$$
s _ { j , t } \equiv s _ { j } \left( \Theta _ { t } ^ { ( j ) } , t ; C _ { j } \right) .\tag{29}
$$

Using the probability-path coeficients $m _ { t } = t$ and $\sigma _ { t } = 1 - t + \sigma _ { \operatorname* { m i n } }$ , this score is obtained from Eq. (19) as

$$
s _ { j , t } \approx \frac { t v _ { j } \Big ( \Theta _ { t } ^ { ( j ) } , t ; C _ { j } \Big ) - \Theta _ { t } ^ { ( j ) } } { ( 1 - t + \sigma _ { \operatorname* { m i n } } ) ( 1 + \sigma _ { \operatorname* { m i n } } ) } .\tag{30}
$$

This conversion is specific to the flow matching formulation used here. The subsequent composition follows the Gauss precision-weighted score composition of Ref. [5]: each single-observation contribution is treated using a Gaussian approximation to its denoising distribution at path time t, and information from independent measurements is combined by adding precision matrices.

Each clean single-observation posterior is approximated by a Gaussian in its local parameter coordinates,

$$
p _ { j } \big ( \Theta ^ { ( j ) } \mid { C } _ { j } \big ) \approx \mathcal { N } ( \mu _ { j } , \Sigma _ { j } ) , \qquad \Lambda _ { j } = \Sigma _ { j } ^ { - 1 } .\tag{31}
$$

For the Gaussian probability path, the corresponding denoising precision at path time t is

$$
\Lambda _ { j , t } = \frac { m _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } I + \Lambda _ { j } .\tag{32}
$$

This is a full matrix. Its shared–local of-diagonal blocks retain posterior correlations that would be lost under a diagonal approximation.

In practice, $\Lambda _ { j }$ is estimated per context from the amortized model itself: we draw a batch of raw FMPE samples from $p _ { j } ( \cdot \mid C _ { j } )$ , compute their empirical covariance $\widehat { \Sigma } _ { j }$ , and set $\Lambda _ { j } = \widehat { \Sigma } _ { j } ^ { - 1 }$

The prior enters the composition through two distinct quantities: the score $s _ { t } ^ { \mathrm { p r } }$ of the perturbed shared prior and the precision $\Lambda _ { t } ^ { \mathrm { p r } }$ of its denoising distribution. Define the perturbed shared-prior density by

$$
p _ { t } ^ { \mathrm { p r } } \bigl ( \theta _ { t } ^ { \mathrm { s h } } \bigr ) = \int k _ { t } \bigl ( \theta _ { t } ^ { \mathrm { s h } } \mid \theta ^ { \mathrm { s h } } \bigr ) p \bigl ( \theta ^ { \mathrm { s h } } \mid b ^ { \mathrm { s h } } \bigr ) ~ \mathrm { d } \theta ^ { \mathrm { s h } } ,\tag{33}
$$

where $k _ { t }$ is the Gaussian probability-path kernel. Its score is

$$
\begin{array} { r } { s _ { t } ^ { \mathrm { p r } } \big ( \theta _ { t } ^ { \mathrm { s h } } \big ) = \nabla _ { \theta _ { t } ^ { \mathrm { s h } } } \log p _ { t } ^ { \mathrm { p r } } \big ( \theta _ { t } ^ { \mathrm { s h } } \big ) . } \end{array}\tag{34}
$$

Given this score, the precision of the corresponding denoising distribution follows from the second-order Tweedie identity,

$$
\Lambda _ { t } ^ { \mathrm { p r } } \big ( \theta _ { t } ^ { \mathrm { s h } } \big ) = \frac { m _ { t } ^ { 2 } } { \sigma _ { t } ^ { 2 } } \left[ I + \sigma _ { t } ^ { 2 } \nabla _ { \theta _ { t } ^ { \mathrm { s h } } } s _ { t } ^ { \mathrm { p r } } \big ( \theta _ { t } ^ { \mathrm { s h } } \big ) \right] ^ { - 1 } .\tag{35}
$$

Let

$$
s _ { \mathrm { c o m p } , t } \equiv s _ { \mathrm { c o m p } } ( \Theta _ { t } , t ; C _ { 1 : n } ) .\tag{36}
$$

The composed score is defined as the solution of the linear system

$$
\Lambda _ { \mathrm { c o m p } , t } s _ { \mathrm { c o m p } , t } = ( 1 - n ) P _ { \mathrm { s h } } ^ { \mathsf { T } } \Lambda _ { t } ^ { \mathrm { p r } } s _ { t } ^ { \mathrm { p r } } + \sum _ { j = 1 } ^ { n } P _ { j } ^ { \mathsf { T } } \Lambda _ { j , t } s _ { j , t } ,\tag{37}
$$

with

$$
\Lambda _ { \mathrm { c o m p } , t } = ( 1 - n ) P _ { \mathrm { s h } } ^ { \mathsf { T } } \Lambda _ { t } ^ { \mathrm { p r } } P _ { \mathrm { s h } } + \sum _ { j = 1 } ^ { n } P _ { j } ^ { \mathsf { T } } \Lambda _ { j , t } P _ { j } .\tag{38}
$$

The factor $1 - n$ implements the shared-prior correction from Eq. (26). No corresponding correction is applied to the local blocks. The isotropic path-noise precision is counted exactly once because

$$
( 1 - n ) P _ { \mathrm { s h } } ^ { \mathsf { T } } P _ { \mathrm { s h } } + \sum _ { j = 1 } ^ { n } P _ { j } ^ { \mathsf { T } } P _ { j } = I .\tag{39}
$$

If the shared prior and all single-observation posteriors are Gaussian, and their scores are evaluated exactly, this construction reproduces the exact score of the perturbed joint posterior. In the absence of local parameter blocks, it reduces to the standard tall-data Gauss composition of Ref. [5].

Because the shared-prior term enters with the negative coeficient $1 - n , \Lambda _ { \mathrm { c o m p } , t }$ is not automatically positive definite for an arbitrary collection of approximate factors. We therefore monitor its smallest eigenvalue during sampling and, when required, apply the minimum diagonal shift needed to enforce a small positive eigenvalue floor.

Solving Eq. (37) blockwise clarifies the roles of the individual terms. Partition the denoising precision and score of observation $j$ as

$$
\Lambda _ { j , t } = \left( \begin{array} { l l } { \Lambda _ { j , t } ^ { \mathrm { s s } } } & { \Lambda _ { j , t } ^ { \mathrm { s l } } } \\ { \Lambda _ { j , t } ^ { \mathrm { l s } } } & { \Lambda _ { j , t } ^ { \mathrm { l l } } } \end{array} \right) , \qquad s _ { j , t } = \left( \begin{array} { l } { s _ { j , t } ^ { \mathrm { s h } } } \\ { s _ { j , t } ^ { \mathrm { l o c } } } \end{array} \right) .\tag{40}
$$

The local rows give

$$
s _ { \mathrm { c o m p } , j , t } ^ { \mathrm { l o c } } = s _ { j , t } ^ { \mathrm { l o c } } + \left( \Lambda _ { j , t } ^ { \mathrm { l l } } \right) ^ { - 1 } \Lambda _ { j , t } ^ { \mathrm { l s } } \left( s _ { j , t } ^ { \mathrm { s h } } - s _ { \mathrm { c o m p } , t } ^ { \mathrm { s h } } \right) .\tag{41}
$$

Thus, when composition moves the shared parameters away from the values preferred by observation j alone, the local parameters of that observation are corrected along their posterior correlation directions.

Eliminating the local blocks yields the shared-block solution

$$
s _ { \mathrm { c o m p } , t } ^ { \mathrm { s h } } = \left[ ( 1 - n ) \Lambda _ { t } ^ { \mathrm { p r } } + \sum _ { j = 1 } ^ { n } { \bar { \Lambda } } _ { j , t } \right] ^ { - 1 } \left[ ( 1 - n ) \Lambda _ { t } ^ { \mathrm { p r } } s _ { t } ^ { \mathrm { p r } } + \sum _ { j = 1 } ^ { n } { \bar { \Lambda } } _ { j , t } s _ { j , t } ^ { \mathrm { s h } } \right] ,\tag{42}
$$

where

$$
\bar { \Lambda } _ { j , t } = \Lambda _ { j , t } ^ { \mathrm { s s } } - \Lambda _ { j , t } ^ { \mathrm { s l } } \left( \Lambda _ { j , t } ^ { \mathrm { l l } } \right) ^ { - 1 } \Lambda _ { j , t } ^ { \mathrm { l s } } .\tag{43}
$$

The Schur complement $\bar { \Lambda } _ { j , t }$ is the marginal precision with which observation j constrains the shared parameters after accounting for uncertainty in its local parameters. Each measurement therefore contributes to the shared block in proportion to the information it actually carries about that block, while the shared prior is counted exactly once.

For two observations, Eq. (42) reduces to

$$
s _ { \mathrm { c o m p } , t } ^ { \mathrm { s h } } = \left( \bar { \Lambda } _ { 1 , t } + \bar { \Lambda } _ { 2 , t } - \Lambda _ { t } ^ { \mathrm { p r } } \right) ^ { - 1 } \left( \bar { \Lambda } _ { 1 , t } s _ { 1 , t } ^ { \mathrm { s h } } + \bar { \Lambda } _ { 2 , t } s _ { 2 , t } ^ { \mathrm { s h } } - \Lambda _ { t } ^ { \mathrm { p r } } s _ { t } ^ { \mathrm { p r } } \right) ,\tag{44}
$$

which is the form presented in the Results.

If all shared–local of-diagonal entries of the factor precisions are discarded, Eqs. (41) and (42) reduce to an element-wise precision-weighted average on the shared block and to

$$
s _ { \mathrm { c o m p } , j , t } ^ { \mathrm { l o c } } = s _ { j , t } ^ { \mathrm { l o c } }\tag{45}
$$

on the local blocks. This diagonal-covariance simplification ignores the posterior correlations between shared and local parameters and, in particular, sets the local correction in Eq. (41) to zero. We therefore use the full-covariance rule throughout.

Once the composed score has been formed, it is converted back into a global flow matching vector field using the inverse of Eq. (18),

$$
v _ { \mathrm { c o m p } } ( \Theta _ { t } , t ; C _ { 1 : n } ) = \frac { \dot { m } _ { t } \Theta _ { t } + \sigma _ { t } \left( \dot { m } _ { t } \sigma _ { t } - m _ { t } \dot { \sigma } _ { t } \right) s _ { \mathrm { c o m p } , t } } { m _ { t } } .\tag{46}
$$

For the coeficients used here, this becomes

$$
v _ { \mathrm { c o m p } } ( \Theta _ { t } , t ; C _ { 1 : n } ) = \frac { \Theta _ { t } + ( 1 - t + \sigma _ { \mathrm { m i n } } ) ( 1 + \sigma _ { \mathrm { m i n } } ) s _ { \mathrm { c o m p } , t } } { t } .\tag{47}
$$

Posterior samples are generated by integrating the global ordinary diferential equation

$$
\frac { \mathrm { d } \Theta _ { t } } { \mathrm { d } t } = v _ { \mathrm { c o m p } } \big ( \Theta _ { t } , t ; C _ { 1 : n } \big ) .\tag{48}
$$

starting from a small non-zero time $t _ { \mathrm { m i n } } = 0 . 0 0 1$ to avoid the singularity in Eq. (47).

This formulation covers two practically important cases. In the homogeneous case, all observations use the same parameterization and the same trained conditional flow model. This setting is natural for multi-contrast neutron reflectometry, where structural parameters are shared across contrasts while contrast-dependent or instrumental parameters remain observation-specific. In the heterogeneous case, diferent observations may use diferent local parameterizations and diferent trained conditional flow models, provided that the shared parameters represent the same physical quantities and use identical inference-time prior bounds and probability paths.

## MCMC refinement

Raw amortized posteriors produced by neural network models in SBI approaches are generally approximate. We therefore optionally apply a second-stage posterior refinement step after amortized sampling. The purpose of this step is to target the posterior defined by the explicit forward-model likelihood and the selected prior bounds using the amortized posterior as an initialization distribution.

Specifically, we use samples generated by the neural network model to initialize an ensemble Markov chain Monte Carlo (MCMC) sampler. For refinement, we use a GPU-parallel afineinvariant ensemble MCMC scheme [61], adapted from the PANPE framework [37].

This refinement step combines the speed of amortized neural inference with likelihoodbased posterior sampling. In contrast to classical initialization strategies based on a single best-fit solution with a small local perturbation, amortized posterior samples provide an ensemble of starting points already located in plausible regions of parameter space. This is particularly useful when the posterior is non-Gaussian, correlated or multimodal.

## Training data generation

Training data are generated on the fly using a batched GPU implementation of the reflectometry forward model. At each training iteration, physical parameter values and corresponding prior bounds are sampled from the model-specific training ranges (listed in the Supplementary Information). The sampling procedure follows previous work on prior-conditioned point estimation and posterior estimation models [31, 37]. For slab-style parameterizations, we enforce physically motivated constraints, such as interfacial roughnesses not exceeding half the thicknesses of adjacent layers and the total layer thickness not exceeding 1500 $\textrm { \AA }$ . For the XRR parameterization with absorption, the imaginary part of the electronic SLD is restricted not to exceed 0.2 times the corresponding real part to avoid strongly unphysical training examples.

The $q$ points used to simulate reflectivity curves are sampled from predefined ranges for the minimum and maximum $q$ value and for the number of points. For XRR models, the $q$ spacing is uniform. For NR models, half of each training batch uses uniformly spaced points and half uses randomly spaced points. This choice makes the NR models more robust to non-uniformly sampled data, as commonly encountered in time-of-flight measurements.

Theoretical XRR or NR curves are simulated from the sampled $q$ values and structural parameters using the Abel\`es transfer-matrix formalism [62], as implemented in the reflectorch package [63]. Non-slab parameterizations, such as the polymer-brush model, are first converted to an equivalent slab representation. For non–spin-flip PNR, the two reflectivity channels are simulated separately from the efective SLDs $\rho ^ { \uparrow } = \rho ^ { n } + \rho ^ { m }$ and $\rho ^ { \downarrow } = \rho ^ { n } - \rho ^ { m }$ , where $\rho ^ { n }$ and $\rho ^ { m }$ denote the nuclear and magnetic SLDs.

For full spin-resolved PNR, we use a transfer-matrix method with 4 × 4 matrices [64], implemented in PyTorch for this work and verified against refnx. In this setting, interfacial roughness cannot be incorporated through the usual N´evot–Croce factor used with the Abel\`es formalism. A more detailed treatment would require microslicing, which is computationally expensive for large-scale neural-network training. We therefore restrict the full spin-resolved PNR example to a simplified parameterization without roughness. Additional PNR efects, such as magnetic-domain roughness, Zeeman efects or helical magnetic structures, are outside the scope of the present work [65–67].

For NR models, instrumental resolution smearing is included during training. We apply linear resolution smearing with coeficient $d q / q$ , sampled from a predefined range for each curve and supplied to the neural network as an additional conditioning variable. This provides an approximation to pointwise resolution smearing in time-of-flight data; when pointwise resolution values are available experimentally, they are used explicitly during MCMC refinement. Because resolution smearing requires simulation on a finer $q -$ grid followed by convolution and interpolation, NR simulations for training are more expensive than XRR simulations in both computation time and memory usage.

Finally, the simulated curves are modified by sampled instrumental parameters, which may include a q-shift, intensity scale factor or constant background. For non–spin-flip PNR, the theoretical channels may also be mixed by a coeficient α, which provides a phenomenological channel-leakage correction for imperfect neutron-beam polarization. More complete treatments of polarization analysis describe imperfect polarizers, flippers and analyzers through eficiency matrices [68]. Gaussian noise is then added to the simulated curves.

## Model training and inference settings

All neural networks were trained using the prior-amortized flow matching objective with a normal base distribution, two-sided conditioning and linear probability paths. We used the AdamW optimizer for 500 000 training iterations, which took approximately one day on a single NVIDIA A100 GPU for the model sizes used here. The learning rate followed a cosine annealing schedule from $1 . 0 \times 1 0 ^ { - 3 }$ , after a 500-step warm-up, to $1 . 0 \times 1 0 ^ { - 6 }$ . Unless stated otherwise, posterior samples from trained FMPE models were generated by integrating the learned ODE with the adaptive Dormand–Prince solver implemented in torchdiffeq. Training was performed on NVIDIA A100 GPUs with dedicated FP64 cores for doubleprecision arithmetic. Because the reflectometry simulations rely on double-precision (float64) arithmetic, their eficiency is significantly improved on GPUs that provide high throughput for double-precision operations.

## Polymer-brush parameterization

For the polymer-brush example, we use a structured parameterization for an end-grafted brush following classical brush theory [69]. The polymer volume-fraction profile is

$$
\phi ( z ) = \phi _ { 0 } \left[ 1 - \Big ( \frac { z } { L } \Big ) ^ { n } \right] , \qquad 0 \leq z \leq L ,\tag{49}
$$

where $\phi _ { 0 }$ is the polymer volume fraction at the grafting interface, L is the brush height and n is the brush-profile exponent. The case $n = 2$ corresponds to an ideal parabolic brush profile, while other values allow deviations from that ideal form.

To evaluate the reflectivity, the continuous brush profile is discretized into $N = 1 0$ slices of equal thickness,

$$
\Delta z = { \frac { L } { N } } .\tag{50}
$$

Each slice is assigned an efective SLD through linear mixing of polymer and solvent,

$$
\rho _ { i } = \phi _ { i } \rho _ { \mathrm { p o l } } + \left( 1 - \phi _ { i } \right) \rho _ { \mathrm { s o l v } } ,\tag{51}
$$

where $1 - \phi _ { i }$ is the solvent volume fraction. The roughness between adjacent brush slices is fixed to

$$
\sigma _ { i } = \frac { L } { 2 N } ,\tag{52}
$$

which smooths the discretized representation of the continuous profile, following the implementation used in anaklasis [42]. A thin reactive layer of thickness $d _ { \mathrm { r l } }$ and SLD $\rho _ { \mathrm { r l } }$ is included between the fronting medium and the brush.

## DATA AVAILABILITY

The experimental datasets analyzed in this study are available from the public sources cited in the corresponding Results sections. The time-of-flight NR dataset is publicly available from the Institut Laue–Langevin data archive at https://doi.ill.fr/10.5291/ILL-DATA. 9-11-1903. The polymer-brush and XRR/PNR co-refinement datasets are published along with the anaklasis package [42]. The multiple-contrast NR dataset is published along with the refnx examples [47]. The synthetic full spin-resolved PNR dataset was generated using the forward model and parameter settings described in Methods.

## CODE AVAILABILITY

The code used to train the flow matching posterior estimators, perform posterior sampling, run MCMC refinement and generate the figures will be made available as part of the reflectorch package. A version of the code associated with this manuscript, together with configuration files and scripts required to reproduce the reported analyses, will be archived in a DOI-minting public repository before publication.

## AUTHOR CONTRIBUTIONS

V.M. implemented the machine learning methods introduced in this work, trained the neural networks, evaluated the results and wrote the first draft of the paper. V.S. provided expertise on the machine learning methods. D.L., A.H. and F.S. provided expertise on reflectometry experiments and supervised the project. All authors contributed to refining the text of the paper.

## COMPETING INTERESTS

The authors declare no competing interests.

## ACKNOWLEDGEMENTS

This work was supported by the DFG (project number 557510229). This research is part of the VIPR project (05D23VT1, ERUM-DATA), funded by the German Federal Ministry for Education and Research (BMBF). We acknowledge support by the consortium DAPHNE4NFDI in the context of the work of NFDI e.V., funded by the German Research Foundation (DFG, grant number 460248799). We thank the University of T¨ubingen Cluster of Excellence “Machine Learning: New Perspectives for Science” for support (EXC 2064/1, project no. 390727645). We acknowledge DESY (Hamburg, Germany), a member of the Helmholtz Association, for the provision of experimental facilities. We thank Dr. Alexandros Koutsioubas for making the experimental data from the anaklasis package available.

[1] J. Als-Nielsen and D. McMorrow, Elements of Modern X-ray Physics, 2nd ed. (John Wiley & Sons, 2011).

[2] M. Tolan, X-ray Scattering from Soft-Matter Thin Films: Materials Science and Basic Research, Springer Tracts in Modern Physics (Springer, Berlin, 1999).

[3] K. Cranmer, J. Brehmer, and G. Louppe, The frontier of simulation-based inference, Proceedings of the National Academy of Sciences 117, 30055 (2020).

[4] T. Gefner, G. Papamakarios, and A. Mnih, Compositional score modeling for simulationbased inference, in Proceedings of the 40th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 202, edited by A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett (PMLR, 2023) pp. 11098–11116.

[5] J. Linhart, G. V. Cardoso, A. Gramfort, S. L. Corf, and P. L. C. Rodrigues, Difusion posterior sampling for simulation-based inference in tall data settings (2026), arXiv:2404.07593 [stat.ML].

[6] Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le, Flow matching for generative modeling, in The Eleventh International Conference on Learning Representations (2023).

[7] A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio, Improving and generalizing flow-based generative models with minibatch optimal transport, Transactions on Machine Learning Research (2024).

[8] J. Wildberger, M. Dax, S. Buchholz, S. Green, J. H. Macke, and B. Sch¨olkopf, Flow matching for scalable simulation-based inference, in Advances in Neural Information Processing Systems, Vol. 36, edited by A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (Curran Associates, Inc., 2023) pp. 16837–16864.

[9] J. Daillant and A. Gibaud, X-Ray and Neutron Reflectivity (Springer, 2009).

[10] J. Ankner, C. Majkrzak, and S. Satija, Neutron reflectivity and grazing angle difraction, J. Res. Natl. Inst. Stand. Technol. 98, 47 (1993).

[11] C. Majkrzak, Polarized neutron reflectometry, Phys. B Condens. Matter 173, 75 (1991).

[12] G. L. Causer, L. Guasco, O. Paull, and D. Cortie, Topical review of quantum materials and heterostructures studied by polarized neutron reflectometry, physica status solidi (RRL) – Rapid Research Letters 17, 2200421 (2023).

[13] B. W. Treece, P. A. Kienzle, D. P. Hoogerheide, C. F. Majkrzak, M. L¨osche, and F. Heinrich, Optimization of reflectometry experiments using information theory, J. Appl. Crystallogr. 52, 47 (2019).

[14] A. R. McCluskey, J. F. K. Cooper, T. Arnold, and T. Snow, A general approach to maximise information density in neutron reflectometry analysis, Mach. Learn.: Sci. Technol. 1, 035002 (2020).

[15] J. H. Durant, L. Wilkins, K. Butler, and J. F. K. Cooper, Determining the maximum information gain and optimizing experimental design in neutron reflectometry using the Fisher information, J. Appl. Crystallogr. 54, 1100 (2021).

[16] S. A. Holt, A. P. Le Brun, C. F. Majkrzak, D. J. McGillivray, F. Heinrich, M. L¨osche, and J. H. Lakey, An ion-channel-containing model membrane: structural determination by magnetic contrast neutron reflectometry, Soft Matter 5, 2576 (2009).

[17] C. G. Schroer, H.-C. Wille, O. H. Seeck, K. Bagschik, H. Schulte-Schrepping, M. Tischer, H. Graafsma, W. Laasch, K. Baev, S. Klumpp, R. Bartolini, H. Reichert, W. Leemans, and E. Weckert, The synchrotron radiation source PETRA III and its future ultra-low-emittance upgrade PETRA IV, The European Physical Journal Plus 137, 1312 (2022).

[18] R. Haensel, European synchrotron radiation facility (ESRF), Review of Scientific Instruments 63, 1571 (1992).

[19] R. A. Campbell, H. P. Wacklin, I. Sutton, R. Cubitt, and G. Fragneto, FIGARO: The new horizontal neutron reflectometer at the ILL, The European Physical Journal Plus 126, 107 (2011).

[20] S. Kowarik, A. Gerlach, W. Leitenberger, J. Hu, G. Witte, C. W¨oll, U. Pietsch, and F. Schreiber, Energy-dispersive X-ray reflectivity and GID for real-time growth studies of pentacene thin films, Thin Solid Films 515, 5606 (2007), the Ninth International Conference on Surface X-Ray and Neutron Scattering.

[21] M. Lippmann, A. Bufet, K. Pflaum, A. Ehnes, A. S. Ciobanu, and O. H. Seeck, A new setup for high resolution fast X-ray reflectivity data acquisition, Review of Scientific Instruments 87, 113904 (2016).

[22] K. A. Ritley, B. Krause, F. Schreiber, and H. Dosch, A portable ultrahigh vacuum organic molecular beam deposition system for in situ x-ray difraction measurements, Review of Scientific Instruments 72, 1453 (2001).

[23] S. Kowarik, A. Gerlach, S. Sellner, F. Schreiber, L. Cavalcanti, and O. Konovalov, Real-time observation of structural and orientational transitions during growth of organic thin films, Phys. Rev. Lett. 96, 125504 (2006).

[24] A. Greco, V. Starostin, C. Karapanagiotis, A. Hinderhofer, A. Gerlach, L. Pithan, S. Liehr, F. Schreiber, and S. Kowarik, Fast fitting of reflectivity data of growing thin films using neural networks, J. Appl. Crystallogr. 52, 1342 (2019).

[25] A. Greco, V. Starostin, A. Hinderhofer, A. Gerlach, M. W. A. Skoda, S. Kowarik, and F. Schreiber, Neural network analysis of neutron and x-ray reflectivity data: pathological cases, performance and perspectives, Mach. Learn.: Sci. Technol. 2, 045003 (2021).

[26] A. Greco, V. Starostin, E. Edel, V. Munteanu, N. Russegger, I. Dax, C. Shen, F. Bertram, A. Hinderhofer, A. Gerlach, and F. Schreiber, Neural network analysis of neutron and xray reflectivity data: automated analysis using mlreflect, experimental errors and feature engineering, J. Appl. Crystallogr. 55, 362 (2022).

[27] H. Aoki, Y. Liu, and T. Yamashita, Deep learning approach for an interface structure analysis with a large statistical noise in neutron reflectometry, Sci. Rep. 11, 22711 (2021).

[28] D. Mironov, J. H. Durant, R. Mackenzie, and J. F. K. Cooper, Towards automated analysis for neutron reflectivity, Machine Learning: Science and Technology 2, 035006 (2021).

[29] J. M. Carmona Loaiza and Z. Raza, Towards reflectivity profile inversion through artificial neural networks, Machine Learning: Science and Technology 2, 025034 (2021).

[30] M. Doucet, R. K. Archibald, and W. T. Heller, Machine learning for neutron reflectometry data analysis of two-layer thin films, Machine Learning: Science and Technology 2, 035001 (2021).

[31] V. Munteanu, V. Starostin, A. Greco, L. Pithan, A. Gerlach, A. Hinderhofer, S. Kowarik, and F. Schreiber, Neural network analysis of neutron and X-ray reflectivity data incorporating prior knowledge, Journal of Applied Crystallography 57, 456 (2024).

[32] L. Pithan, V. Starostin, D. Mareˇcek, L. Petersdorf, C. V¨olter, V. Munteanu, M. Jankowski, O. Konovalov, A. Gerlach, A. Hinderhofer, B. Murphy, S. Kowarik, and F. Schreiber, Closing the loop: autonomous experiments enabled by machine-learning-based online data analysis in synchrotron beamline environments, Journal of Synchrotron Radiation 30, 1064 (2023).

[33] D. Mareˇcek, J. Oberreiter, A. Nelson, and S. Kowarik, Faster and lower-dose X-ray reflectivity measurements enabled by physics-informed modeling and artificial intelligence co-refinement, Journal of Applied Crystallography 55, 1305 (2022).

[34] D. Schumi-Mareˇcek, A. Nelson, E. Pfeiler, M. Eder, F. Bertram, and S. Kowarik, High-speed x-ray reflectometry for characterization of thin film growth at high deposition rates, Phys. Rev. B 112, 235304 (2025).

[35] N. Andrejevic, Z. Chen, T. Nguyen, L. Fan, H. Heiberger, L.-J. Zhou, Y.-F. Zhao, C.-Z. Chang, A. Grutter, and M. Li, Elucidating proximity magnetism through polarized neutron reflectometry and machine learning, Appl. Phys. Rev. 9, 011421 (2022).

[36] K. T. Kim and D. R. Lee, Probabilistic parameter estimation using a Gaussian mixture density network: application to X-ray reflectivity data curve fitting, Journal of Applied Crystallography

54, 1572 (2021).

[37] V. Starostin, M. Dax, A. Gerlach, A. Hinderhofer, Alvaro Tejero-Cantero, and F. Schreiber, <sup>´</sup> Fast and reliable probabilistic reflectometry inversion with prior-amortized neural posterior estimation, Science Advances 11, eadr9668 (2025).

[38] R. J. Chesterfield, J. C. McKeen, C. R. Newman, P. C. Ewbank, D. A. da Silva Filho, J.-L. Br´edas, L. L. Miller, K. R. Mann, and C. D. Frisbie, Organic thin film transistors based on N-alkyl perylene diimides: Charge transport kinetics as a function of gate voltage and temperature, The Journal of Physical Chemistry B 108, 19281 (2004).

[39] A. M. Higgins, P. Gutfreund, V. Italia, A. Nelson, J. T. Cabral, and E. L. Hynes, Hysteresis in phase volumes, compositions and interfacial roughness in model OPV-small-molecule/polymer thin-films, Soft Matter 20, 2532 (2024).

[40] H. Chen, J. Peet, Y.-C. Hsiao, B. Hu, and M. Dadmun, The impact of fullerene structure on its miscibility with P3HT and its correlation of performance in organic photovoltaics, Chemistry of Materials 26, 3993 (2014).

[41] I. Hiotelis, A. G. Koutsioubas, N. Spiliopoulos, D. L. Anastassopoulos, A. A. Vradis, C. Toprakcioglu, A. Menelle, G. Sakellariou, and N. Hadjichristidis, Neutron reflectivity and computer simulation studies of self-assembled brushes formed by centrally adsorbed star polymers, Macromolecules 41, 7648 (2008).

[42] A. Koutsioubas, anaklasis: a compact software package for model-based analysis of specular neutron and X-ray reflectometry data sets, Journal of Applied Crystallography 54, 1857 (2021).

[43] M. Krishnamoorthy, S. Hakobyan, M. Ramstedt, and J. E. Gautrot, Surface-initiated polymer brushes in the biomedical field: Applications in membrane science, biosensing, cell culture, regenerative medicine and antibacterial coatings, Chemical Reviews 114, 10976 (2014).

[44] H. Zabel, K. Theis-Br¨ohl, M. Wolf, and B. P. Toperverg, Polarized neutron reflectometry for the analysis of nanomagnetic systems, IEEE Transactions on Magnetics 44, 1928 (2008).

[45] F. Cousin and A. Chennevi\`ere, Neutron reflectivity for soft matter, in EPJ Web of Conferences, Vol. 188 (2018) p. 04001.

[46] F. Heinrich, P. A. Kienzle, D. P. Hoogerheide, and M. L¨osche, Information gain from isotopic contrast variation in neutron reflectometry on protein–membrane complex structures, Journal of Applied Crystallography 53, 800 (2020).

[47] A. R. J. Nelson and S. W. Prescott, refnx: neutron and X-ray reflectometry analysis in Python, Journal of Applied Crystallography 52, 193 (2019).

[48] S. Eimer, H. Cheng, J. Li, X. Zhang, C. Zhao, and W. Zhao, Perpendicular magnetic anisotropy based spintronics devices in Pt/Co stacks under diferent hard and flexible substrates, Science China Information Sciences 66, 122408 (2023).

[49] H. Hayashi, A. Musha, H. Sakimura, and K. Ando, Spin-orbit torques originating from the bulk and interface in Pt-based structures, Physical Review Research 3, 013042 (2021).

[50] Y. Lipman, M. Havasi, P. Holderrieth, N. Shaul, M. Le, B. Karrer, R. T. Q. Chen, D. Lopez-Paz, H. Ben-Hamu, and I. Gat, Flow matching guide and code (2024), arXiv:2412.06264 [cs.LG].

[51] B. Holzschuh and N. Thuerey, Flow matching for posterior inference with simulator feedback (2024), arXiv:2410.22573 [cs.LG].

[52] Z. Geng, M. Deng, X. Bai, J. Z. Kolter, and K. He, Mean flows for one-step generative modeling, in Advances in Neural Information Processing Systems, Vol. 38 (Curran Associates, Inc., 2025) pp. 75460–75482.

[53] H. Gao, P. Chen, F. Shi, R. Wu, Y. Li, Q. Hui, Y. You, T. Lu, C. Tan, S. Zhao, Z. Liu, F. Zhao, K. Wang, and S. Lian, MeanCache: From instantaneous to average velocity for accelerating flow matching inference, in The Fourteenth International Conference on Learning Representations (2026).

[54] M. S. Albergo, N. M. Bofi, and E. Vanden-Eijnden, Stochastic interpolants: A unifying framework for flows and difusions (2025), arXiv:2303.08797 [cs.LG].

[55] X. Liu, C. Gong, and Q. Liu, Flow straight and fast: Learning to generate and transfer data with rectified flow, in International Conference on Learning Representations (2023).

[56] N. Ma, M. Goldstein, M. S. Albergo, N. M. Bofi, E. Vanden-Eijnden, and S. Xie, SiT: Exploring flow and difusion-based generative models with scalable interpolant transformers (2024), arXiv:2401.08740 [cs.CV].

[57] S. Singh and I. Fischer, Stochastic sampling from deterministic flow models (2024), arXiv:2410.02217 [cs.LG].

[58] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. K. Duvenaud, Neural ordinary diferential equations, in Advances in Neural Information Processing Systems, Vol. 31, edited by S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett (Curran Associates, Inc., 2018).

[59] E. Perez, F. Strub, H. de Vries, V. Dumoulin, and A. Courville, FiLM: Visual reasoning with a general conditioning layer, in Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 32 (2018).

[60] J. Boelts, M. Deistler, M. Gloeckler, A. Tejero-Cantero, J.-M. Lueckmann, G. Moss, P. Steinbach, T. Moreau, F. Muratore, J. Linhart, C. Durkan, et al., sbi reloaded: a toolkit for simulation-based inference workflows, Journal of Open Source Software 10, 7754 (2025).

[61] J. Goodman and J. Weare, Ensemble samplers with afine invariance, Communications in Applied Mathematics and Computational Science 5, 65 (2010).

[62] F. Abel\`es, La th´eorie g´en´erale des couches minces, Journal de Physique et le Radium 11, 307 (1950).

[63] V. Munteanu, V. Starostin, A. Hinderhofer, A. Gerlach, D. Lapkin, and F. Schreiber, reflectorch: a deep learning package for X-ray and neutron reflectometry, Journal of Open Source Software 10, 8169 (2025).

[64] S. J. Blundell and J. A. C. Bland, Polarized neutron reflection as a probe of magnetic films and multilayers, Physical Review B 46, 3391 (1992).

[65] S. Singh and S. Basu, Separation and correlation of structural and magnetic roughness in a Ni thin film by polarized of-specular neutron reflectometry, Journal of Physics: Condensed Matter 21, 055010 (2009).

[66] B. B. Maranville, B. J. Kirby, A. J. Grutter, P. A. Kienzle, C. F. Majkrzak, Y. Liu, and C. L. Dennis, Measurement and modeling of polarized specular neutron reflectivity in large magnetic fields, Journal of Applied Crystallography 49, 1121 (2016).

[67] E. A. Karhu, S. Kahwaji, M. D. Robertson, H. Fritzsche, B. J. Kirby, C. F. Majkrzak, and T. L. Monchesky, Helical magnetic order in MnSi thin films, Physical Review B 84, 060404 (2011).

[68] A. R. Wildes, The polarizer-analyzer correction problem in neutron polarization analysis experiments, Review of Scientific Instruments 70, 4241 (1999).

[69] S. T. Milner, T. A. Witten, and M. E. Cates, Theory of the grafted polymer brush, Macromolecules 21, 2610 (1988).

## Supplementary Information

![](images/d693b1f6d9463910ffac3c424c826753d31fae3b319112abed4d6b08c8e7feb6.jpg)

![](images/d8b2519c94a942c2fe19b57ed315e363fe3f3da21abb302a9754386a005db077.jpg)  
Figure 6: Experimental XRR data for a one-layer $\mathrm { P T C D I \mathrm { - } C _ { 3 } }$ film on a Si substrate are shown together with posterior-predictive simulations generated from parameter samples. (a) Posterior-predictive curves obtained directly from the raw FMPE posterior. (b) Posterior-predictive curves obtained after FMPE-initialized MCMC refinement.

![](images/2ef0b2fa9f13d40ac6978ee9409c46c7dd55d704c629bcb7c57f268f3c2e31cf.jpg)  
Figure 7: Corner plot of the raw FMPE (green) and FMPE-MCMC (red) posterior inferred for the XRR data of the $\mathrm { P T C D I \mathrm { - } C _ { 3 } }$ film. Filled contours show the 68% and 95% credible regions. The displayed parameters comprise layer thicknesses $d _ { i } .$ , interfacial roughnesses $\sigma _ { i } .$ SLDs $\rho _ { i }$ , the q-shift ofset $q _ { \mathrm { s h } }$ and the reflectivity scale factor $r _ { \mathrm { s c } }$ . Units: $d _ { i }$ and $\sigma _ { i }$ are in $\textup { \AA }$ $\rho _ { i }$ in $1 0 ^ { - 6 } \textup { \AA } ^ { - 2 }$ $q _ { \mathrm { s h } }$ in $\mathrm { ~ \AA ~ } ^ { - 1 }$ , and $r _ { \mathrm { s c } }$ is dimensionless. The subscript B denotes the backing medium. The plot ranges correspond to the prior bounds.

![](images/1679d1229c3bf354f031cece69ca0d2bd8b6d3d7cfc460500b2b5bf4857711ee.jpg)

![](images/9c09853ce1d085211bb41d1f4ba51e909af38c760c2ab0984db078b2a74f2941.jpg)  
Figure 8: Experimental XRR data for a one-layer $\mathrm { C _ { 6 0 } }$ film on a Si substrate are shown together with posterior-predictive simulations generated from parameter samples. (a) Posterior-predictive curves obtained directly from the raw FMPE posterior. (b) Posterior-predictive curves obtained after FMPE-initialized MCMC refinement.

![](images/4e3476d061986d0e715a2a44ba48e3aec0a54631f2277fd1c72e1d2d9d017e3e.jpg)  
Figure 9: Corner plot of the raw FMPE (green) and FMPE-MCMC (red) posterior inferred for the XRR data of the $\mathrm { C _ { 6 0 } }$ film. Filled contours show the 68% and 95% credible regions. The displayed parameters comprise layer thicknesses $d _ { i } .$ interfacial roughnesses $\sigma _ { i } .$ , SLDs $\rho _ { i }$ the q-shift ofset $q _ { \mathrm { s h } }$ and the reflectivity scale factor $r _ { \mathrm { s c } }$ . Units: $d _ { i }$ and $\sigma _ { i }$ are in $\mathrm { ~ \AA } , \rho _ { i }$ in $1 0 ^ { - 6 } \mathring { \mathrm { A } } ^ { - 2 } , q _ { \mathrm { s h } }$ in $\mathrm { ~ \AA ~ } ^ { - 1 }$ , and $r _ { \mathrm { s c } }$ is dimensionless. The subscript B denotes the backing medium. The plot ranges correspond to the prior bounds.

![](images/11f3f4448d948033d7c91fd41582061f831d14f2cd9bc4f87fd1e314ef09ea2c.jpg)

![](images/fb295432a7a7f43ee702d495a1e4db84168e8a3e3e43c0db498db5a0985f63a6.jpg)  
Figure 10: Experimental TOF-NR data for the PS:Bis-PCBM/Bis-PCBM structure, shown together with posterior-predictive simulations generated from parameter samples. (a) Posterior-predictive curves obtained directly from the raw FMPE posterior. (b) Posterior-predictive curves obtained after FMPE-initialized MCMC refinement.

![](images/5f686414504dc22614b55cbc2810c4905e3539210cd47a0e7cc9ac4be74aec56.jpg)  
Figure 11: Corner plot of the raw FMPE (green) and FMPE-MCMC (red) posterior inferred for the NR data from the time-of-flight reflectometer. Filled contours show the 68% and 95% credible regions. The displayed parameters comprise layer thicknesses $d _ { i } .$ interfacial roughnesses $\sigma _ { i } ,$ SLDs $\rho _ { i }$ , the reflectivity scale factor $r _ { \mathrm { s c } }$ and the $\log _ { 1 0 }$ background. Units: $d _ { i }$ and $\sigma _ { i }$ are in $\mathrm { \AA } , \rho _ { i }$ in $1 0 ^ { - 6 } \textup { \AA } ^ { - 2 }$ , while $r _ { \mathrm { s c } }$ and the $\log _ { 1 0 }$ background are dimensionless. The subscript B denotes the backing medium. The plot ranges correspond to the prior bounds.

b  
![](images/ec2336969e3d49b5cfafd34d71332a89d83caba6a14ae6d970862dadac884021.jpg)

![](images/688a04e152c32baa3a47add04a196cbdb7c6a239cfa680ef200540e4f54d401b.jpg)  
Figure 12: Experimental NR data for the polymer-brush structure, shown together with posterior-predictive simulations generated from parameter samples. (a) Posterior-predictive curves obtained directly from the raw FMPE posterior. (b) Posterior-predictive curves obtained after FMPE-initialized MCMC refinement.

![](images/11f5e9cd20175540b37c89bf468e7a628c12ab06e44be8d813791a82d9e9f71e.jpg)  
Figure 13: Corner plot of the raw FMPE (green) and FMPE-MCMC (red) posterior inferred for the NR data of the polymer-brush structure (data sourced from [42]). Filled contours show the 68% and 95% credible regions. The displayed parameters comprise the polymer volume fraction at the grafting interface $\phi _ { 0 }$ , the length of the brush $L ,$ the polymer-brush exponent $n ,$ the thickness of the reactive layer $d _ { \mathrm { r l } } ,$ the SLD of the polymer $\rho _ { \mathrm { p o l } }$ , the roughness of the substrate $\sigma _ { \mathrm { s u b } }$ and the roughness of the reactive layer $\sigma _ { \mathrm { r l } }$ . Units: $L , d _ { \mathrm { r l } } .$ $\sigma _ { \mathrm { s u b } }$ and $\sigma _ { \mathrm { r l } }$ are in $\mathrm { \AA } , \rho _ { \mathrm { p o l } }$ is in $1 0 ^ { - 6 } \textup { \AA } ^ { - 2 }$ , and $\phi _ { 0 }$ and n are dimensionless. The plot ranges correspond to the prior bounds.

![](images/6542ac1e3d73f18a7e35b0b1fdaf07bd82f38a490d87d0ea7200eddc9351637f.jpg)

b  
![](images/8776f37374044771e2a47a8afec6e7bd1a70bb294ac9f208a15779bf44825b7c.jpg)  
Figure 14: Synthetic full spin-resolved PNR data, shown together with posterior-predictive simulations generated from parameter samples. (a) Posterior-predictive curves obtained directly from the raw FMPE posterior. (b) Posterior-predictive curves obtained after FMPE-initialized MCMC refinement.

![](images/dc363e09a95f3142f0329e82985c0eb0ca4bc5bd3d585d322042c8ad91776945.jpg)  
Figure 15: Corner plot of the raw FMPE (green) and FMPE-MCMC (red) posterior inferred for the synthetic full spin-resolved PNR data. The displayed parameters comprise layer thicknesses $d _ { i } ,$ , nuclear SLDs $\rho _ { i } ^ { n }$ , magnetic SLDs $\rho _ { i } ^ { m }$ and magnetization angles $\psi _ { i } ^ { m }$ . Units: $d _ { i }$ are in $\mathrm { ~ \AA } , \rho _ { i } ^ { n }$ and $\rho _ { i } ^ { m }$ are in $1 0 ^ { - 6 } \textup { \AA } ^ { - 2 }$ , and $\psi _ { i } ^ { m }$ is in degrees. The subscript B denotes the backing medium. The plot ranges correspond to the prior bounds.

![](images/9e9ae3445253a64ecaef413d5a2b82acfd952e37415ba8f9f97eef67b7b47b5e.jpg)

b  
![](images/631f6f1dbaf9b550b9e215202d75575fa71951ec89f7ecf9597e80e5ac982f5d.jpg)  
Figure 16: Experimental NR data corresponding to three solvent contrasts $\mathrm { ( D _ { 2 } O , H _ { 2 } O }$ and CM), shown together with posterior-predictive simulations generated from parameter samples. (a) Posterior-predictive curves obtained directly from the raw FMPE posterior. (b) Posterior-predictive curves obtained after FMPE-initialized MCMC refinement.

![](images/ed2807ed287b321d7d719f83064f8ee052788bcdcaee028b61f22e8b259a16b1.jpg)  
Figure 17: Corner plot of the raw FMPE (green) and FMPE-MCMC (red) posterior inferred for the three-contrast NR data. Filled contours show the 68% and 95% credible regions. The displayed parameters comprise the layer thicknesses $d _ { i } ,$ the SLD of the polymer $\rho _ { 2 }$ without the solvent, the solvent volume fraction in the polymer layer $\phi _ { 2 } ^ { \mathrm { s o l v } }$ and the reflectivity scale factors $r _ { \mathrm { s c } }$ and $\log _ { 1 0 }$ backgrounds for each curve. Units: $d _ { i }$ are in $\mathrm { \AA } , \rho _ { 2 }$ is in $1 0 ^ { - 6 } \textup { \AA } ^ { - 2 }$ , and $\phi _ { 2 } ^ { \mathrm { s o l v } }$ $r _ { \mathrm { s c } }$ and the $\log _ { 1 0 }$ backgrounds are dimensionless. The plot ranges correspond to the prior bounds.

![](images/4c80152051d7a89ee8a637e7004fc3e4e9a1cfc7f91afcc4c533945fd1288b8c.jpg)

b  
![](images/7c0cb21304c7f7cc7ca5728870f690d95548213fa0e7fc16901fced93f8623bc.jpg)  
Figure 18: Experimental XRR and PNR data for the Pt/Fe structure, shown together with posterior-predictive simulations generated from parameter samples. (a) Posterior-predictive curves obtained directly from the raw FMPE posterior. (b) Posterior-predictive curves obtained after FMPE-initialized MCMC refinement.

![](images/ef7bd4cd8a59af7d752e64e9a5a95a6d374d1cf860312be39265839e9e6620fb.jpg)  
Figure 19: Corner plot of the raw FMPE (green) and FMPE-MCMC (red) posterior inferred for the Pt/Fe structure. Filled contours show the 68% and 95% credible regions. The displayed parameters comprise the layer thicknesses $d _ { i } .$ , interfacial roughnesses $\sigma _ { i } .$ , the reflectivity scale factors $r _ { \mathrm { s c } } .$ , the $\log _ { 1 0 }$ backgrounds and, for PNR, the mixing coeficient $\alpha .$ Units: $d _ { i }$ and $\sigma _ { i }$ are in ${ \mathrm { { \AA } } } ,$ , while $r _ { \mathrm { s c } } ,$ the $\log _ { 1 0 }$ backgrounds and α are dimensionless. The plot ranges correspond to the prior bounds.

![](images/a432824610cc098a3c6255ee22293f25c64986f8c08a430faa71224b8ad58184.jpg)  
Figure 20: Corner plot of the raw FMPE (green) and FMPE-MCMC (red) posterior, in the case of an alternative inference where the SLDs are not fixed, for the Pt/Fe structure. Filled contours show the 68% and 95% credible regions. The displayed parameters comprise the layer thicknesses $d _ { i } .$ , interfacial roughnesses $\sigma _ { i } .$ , real electronic SLDs, imaginary electronic SLDs, nuclear SLDs, magnetic SLDs, reflectivity scale factors $r _ { \mathrm { s c } } , \log _ { 1 0 }$ backgrounds and, for PNR, the mixing coeficient α. Units: $d _ { i }$ and $\sigma _ { i }$ are in $\textrm { \AA }$ , all SLDs are in $1 0 ^ { - 6 } \textup { \AA } ^ { - 2 }$ , while $r _ { \mathrm { s c } }$ the $\log _ { 1 0 }$ backgrounds and α are dimensionless. The plot ranges correspond to the prior bounds.

![](images/bda50b06795bfc3d4fe40ec19e581101c7393ca75a649c89985ae109f09c0c1e.jpg)

b  
![](images/f42933fa9fd329f7b4b945e669002da3fc97a89dc94e6338f01601a6450edb59.jpg)  
Figure 21: Experimental XRR and PNR data for the Pt/Co structure, shown together with posterior-predictive simulations generated from parameter samples. (a) Posterior-predictive curves obtained directly from the raw FMPE posterior. (b) Posterior-predictive curves obtained after FMPE-initialized MCMC refinement.

![](images/6c702e7bccc58dd7bd88a810c6131dad228228185ae8957f33d7aff12d9fa978.jpg)  
Figure 22: Corner plot of the raw FMPE (green) and FMPE-MCMC (red) posterior inferred for the Pt/Co structure. Filled contours show the 68% and 95% credible regions. The displayed parameters comprise the layer thicknesses $d _ { i } .$ , interfacial roughnesses $\sigma _ { i } .$ , the magnetic SLD of the Co layer $\rho _ { 2 } ^ { m }$ , the reflectivity scale factors $r _ { \mathrm { s c } }$ , the $\log _ { 1 0 }$ backgrounds and, for PNR, the mixing coeficient $\alpha .$ . Units: $d _ { i }$ and $\sigma _ { i }$ are in $\mathrm { \AA } , \rho _ { 2 } ^ { m }$ is in $1 0 ^ { - 6 } \textup { \AA } ^ { - 2 }$ , while $r _ { \mathrm { s c } } ,$ the $\log _ { 1 0 }$ backgrounds and $\alpha$ are dimensionless. The plot ranges correspond to the prior bounds.

![](images/653e3082c561c93fc4701e53a7cdfd36f1243dcf5fb64c2b0ea7ff240bcb2d68.jpg)  
Figure 23: Corner plot of the raw FMPE (green) and FMPE-MCMC (red) posterior, in the case of an alternative inference where the SLDs are not fixed, for the Pt/Co structure. Filled contours show the 68% and 95% credible regions. The displayed parameters comprise the layer thicknesses $d _ { i } .$ , interfacial roughnesses $\sigma _ { i } .$ , real electronic SLDs, imaginary electronic SLDs, nuclear SLDs, magnetic SLDs, reflectivity scale factors $r _ { \mathrm { s c } } , \log _ { 1 0 }$ backgrounds and, for PNR, the mixing coeficient α. Units: $d _ { i }$ and $\sigma _ { i }$ are in $\mathrm { { \AA } }$ all SLDs are in $1 0 ^ { - 6 } \mathring { \mathrm { A } } ^ { - 2 }$ , while $r _ { \mathrm { s c } } .$ the $\log _ { 1 0 }$ backgrounds and α are dimensionless. The plot ranges correspond to the prior bounds.

Table 1: Training ranges for the XRR model with standard slab parameterization
<table><tr><td colspan="2">Parameter type Training range Physical unit</td></tr><tr><td>Thickness</td><td>[0, 500] Å</td></tr><tr><td>Roughnesses</td><td>[0, 60] Å</td></tr><tr><td>SLDs</td><td>[0, 50]  $1 0 ^ { - 6 } \mathrm { ~ \AA ^ { - 2 } }$ </td></tr><tr><td>q-shift</td><td>[-0.002, 0.002]  $\textup { \AA } ^ { - 1 }$ </td></tr><tr><td>Intensity scale</td><td>[0.9, 1.1]</td></tr></table>

Table 2: Training ranges for the NR model with standard slab parameterization
<table><tr><td></td><td>Parameter type Training range Physical unit</td></tr><tr><td>Thicknesses</td><td>[0, 1500] Å</td></tr><tr><td>Roughnesses</td><td>[0, 60] Å</td></tr><tr><td>SLDs</td><td>[−8, 16]  $1 0 ^ { - 6 } \mathrm { ~ \AA ^ { - 2 } }$ </td></tr><tr><td>Intensity scale</td><td>[0.9, 1.1]</td></tr><tr><td>Background  $\left( \log _ { 1 0 } \right)$ </td><td> $[ - 1 0 , \ - 4 ]$ </td></tr></table>

Table 3: Training ranges for the NR model with parameterization incorporating solvent penetration
<table><tr><td>Parameter type</td><td>Training range Physical unit</td><td></td></tr><tr><td>Thicknesses</td><td>[0, 500]</td><td>Å</td></tr><tr><td>Roughnesses</td><td>[0, 60]</td><td>Å</td></tr><tr><td>SLDs</td><td>[−8, 16]</td><td> $1 0 ^ { - 6 } ~ \mathrm { \AA ^ { - 2 } }$ </td></tr><tr><td>Solvent volume fractions [0, 1]</td><td></td><td></td></tr><tr><td>Intensity scale</td><td>[0.9, 1.1]</td><td></td></tr><tr><td>Background  $\left( \log _ { 1 0 } \right)$ </td><td> $[ - 1 0 , ~ - 4 ]$ </td><td></td></tr></table>

Table 4: Training ranges for the XRR model with absorption parameterization
<table><tr><td></td><td>Parameter type Training range Physical unit</td></tr><tr><td>Thicknesses</td><td>[1, 500] Å</td></tr><tr><td>Roughnesses</td><td>[0, 20] Å</td></tr><tr><td>Re. SLDs</td><td>[0, 150]  $1 0 ^ { - 6 } \mathrm { ~ \AA ^ { - 2 } }$ </td></tr><tr><td>Im. SLDs</td><td>[0, 30]  $1 0 ^ { - 6 } \mathrm { ~ \AA ^ { - 2 } }$ </td></tr><tr><td>Intensity scale</td><td>[0.8, 1.2]</td></tr><tr><td>Background  $\left( \log _ { 1 0 } \right)$ </td><td> $[ - 1 0 , \ - 4 ]$ </td></tr></table>

Table 5: Training ranges for the non–spin-flip PNR model
<table><tr><td>Parameter type</td><td>Training range Physical unit</td></tr><tr><td>Thicknesses</td><td>[1, 500] Å</td></tr><tr><td>Roughnesses</td><td>[0, 20] Å</td></tr><tr><td>Nuclear SLDs</td><td>[0, 10]  $1 0 ^ { - 6 } ~ \mathrm { \AA ^ { - 2 } }$ </td></tr><tr><td>Magnetic SLDs</td><td>[0, 10]  $1 0 ^ { - 6 } ~ \mathrm { \AA ^ { - 2 } }$ </td></tr><tr><td>Channel mixing factor [0, 0.2]</td><td></td></tr><tr><td>Intensity scale</td><td>[0.9, 1.1]</td></tr><tr><td>Background  $\left( \log _ { 1 0 } \right)$ </td><td> $[ - 1 0 , ~ - 4 ]$ </td></tr></table>

Table 6: Training ranges for the full spin-resolved PNR model
<table><tr><td></td><td>Parameter type Training range Physical unit</td></tr><tr><td>Thicknesses</td><td>[1, 500] Å</td></tr><tr><td>Nuclear SLDs</td><td>[0, 10]  $1 0 ^ { - 6 } ~ \mathrm { \AA ^ { - 2 } }$ </td></tr><tr><td>Magnetic SLDs</td><td>[0, 10]  $1 0 ^ { - 6 } \mathrm { ~ \AA ^ { - 2 } }$ </td></tr><tr><td>Magnetization angles [0, 180]</td><td>degrees</td></tr></table>

![](images/20b9b42f785dac8e12db054ccc47d8d241c965d552a0bdae054f28657be6f037.jpg)  
Figure 24: Relation between the real and imaginary SLD components for diferent elements in the periodic table and diferent X-ray beam energies computed using the periodictable package. The dashed line indicates the threshold used to restrict the training distribution of absorption parameters.