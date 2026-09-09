# PocketVE: Stable and Property-Guided Structure-Based Drug Design with Variance-Exploding Diffusion

Peining Zhang, Jinbo Bi University of Connecticut Storrs, Connecticut 06269, USA peining.zhang@uconn.edu, jinbo.bi@uconn.edu

## Abstract

Protein-conditioned 3D molecule generation is a central challenge in structurebased drug design, requiring a balance between pocket compatibility, molecular properties, and physical geometry. We propose PocketVE, a protein-pocketconditioned variance-exploding (VE) diffusion framework that couples stable coordinate denoising with inference-time property guidance. Specifically, PocketVE combines an EDM-style training and sampling setup for 3D denoising, classifierfree guidance for multi-property steering without external property classifiers, and adaptive protein perturbation as a training-time pocket regularizer. Evaluated on CrossDocked2020 under the GenBench3D protocol, PocketVE improves Valid<sub>3D</sub> from 58.6 to 80.6 and reduces strain energy from 457.4 to 127.9 relative to its TAGMol architectural baseline, while retaining competitive docking and molecularproperty scores under moderate guidance. A guidance-scale study shows that moderate guidance gives a favorable balance between target-related objectives and geometric quality, whereas stronger guidance can degrade geometry and distributional fidelity. Pocket-permutation and PoseCheck diagnostics further support pocket-specific spatial compatibility with reduced steric conflicts. Overall, the results suggest that geometric stability and inference-time property guidance should be considered as coupled design objectives.

## 1 Introduction

Generating drug-like molecules with desirable properties and high affinity to a given protein binding site, a task known as structure-based drug design (SBDD), sits at the intersection of 3D geometry and molecular property optimization. A generated ligand must satisfy two classes of constraints: pocket-dependent ones (shape complementarity, binding pose, steric fit) and intrinsic molecular ones (drug-likeness, synthetic accessibility, logP) [23, 45]. Generative models for molecular design have progressively transitioned from 1D strings [11, 28] and 2D graphs [25, 39] to direct 3D modeling [31, 34]. Among 3D generative approaches, non-autoregressive diffusion models have substantially advanced target-aware generation [17, 40, 19]: TargetDiff [14] introduced SE(3)- equivariant diffusion to jointly denoise the full ligand without relying on a fixed generation order, while TAGMol [7] demonstrated that gradient-guided sampling can steer the generative distribution toward regions of higher binding affinity.

Despite these advances, integrating strong protein-ligand interactions without deteriorating basic chemical geometry remains an open challenge. First, the reverse sampling trajectory can be sensitive to the conditioning signal, creating a trade-off between steerability and stability. While image models often work with bounded pixels or latent representations [37], where strong guidance can lead to overexposed outputs [47, 30], guidance in SBDD acts directly on unconstrained Euclidean coordinates. Consequently, excessive guidance pressure often drives the geometry toward non-physical conformations, as reflected by the atom clashes and high strain energy reported in recent GenBench3D evaluations [1], where higher strain indicates a less physically plausible 3D conformation. Second, existing models primarily focus on learning the unconditional chemical distribution of the training data, leaving the integration of real-world multi-objective properties highly inflexible and poorly transferable [7, 24, 6]. As a result, adjusting how strongly a trained model pursues the protein target or a specific drug-like profile usually requires training a separate time-dependent classifier for every property of interest, making it poorly scalable when balancing multiple objectives.

![](images/ac106ccc36c60570717900348d48be7306b42ef10205dd1435ca0a9b02e59222.jpg)  
Figure 1: Overview of PocketVE. PocketVE couples VE-based coordinate denoising with discrete flow matching for atom types. Multi-objective classifier-free guidance provides inference-time property steering, while protein-pocket perturbation regularizes the denoiser during training.

Motivated by these challenges, this paper studies how conditional control changes the property– geometry trade-off in protein-conditioned 3D generation. Our main point is that the guidance mechanism and coordinate backbone should be designed jointly, because both shape the reverse trajectory in Euclidean space. In particular, a VE-style coordinate backbone provides a natural basis for stable denoising across noise scales, while pocket-aware scaling and sampling choices determine how that backbone interacts with the protein coordinate frame. We therefore evaluate PocketVE as an integrated target-aware framework, without attributing its overall improvement to VE alone. PocketVE combines VE/EDM-style coordinate parameterization, pocket-aware scale preservation, classifier-free guidance for multi-property steering, and training-time pocket perturbation. Figure 1 summarizes the overall pipeline and the three design axes of PocketVE.

We evaluate our framework on the CrossDocked2020 benchmark under the GenBench3D protocol [1]. Relative to TAGMol, PocketVE increases Valid<sub>3D</sub> from 58.6 to 80.6 and reduces strain energy from 457.4 to 127.9, while retaining competitive docking and molecular-property scores under moderate guidance. TargetDiff and PAFlow provide complementary geometry and affinity references: PocketVE is comparable to TargetDiff in Valid and has lower strain, whereas PAFlow obtains stronger Vina scores at a substantial geometric cost. Guidance experiments further show a tradeoff between target-related properties and geometric fidelity. Together, these findings suggest that geometric stability and conditional steering should be considered jointly in target-aware 3D molecular generation.

Our main contributions are as follows:

• We develop a pocket-aware VE coordinate parameterization [27] that preserves the shared protein–ligand coordinate frame while using EDM-style preconditioning for stable denoising, together with discrete flow matching for atom types.

• We characterize the guidance-scale trade-off in protein-conditioned 3D molecule generation, showing that Vina Score, QED, and SA respond differently from geometry metrics such as Valid<sub>3D</sub> and strain energy.

• We combine this backbone with multi-objective CFG and protein-side perturbation, improving the balance between structural validity, drug-like properties, and target-specific affinity on CrossDocked2020 under the GenBench3D protocol.

## 2 Related Work

Target-Agnostic 3D Molecular Generation. In target-agnostic 3D molecular generation, Equivariant Diffusion Models learn coordinate denoising with E(3)-equivariant networks [19]. Subsequent works extend this paradigm to latent representations and joint structure–geometry modeling [44, 43], as well as unified generation of atom types, coordinates, and bonds via flow matching [22]. More recently, VE-style parameterizations further improve geometric stability in 3D generation [46]. These advances suggest that geometric stability should be built into the generative backbone rather than enforced post hoc. PocketVE brings this design principle into the protein-conditioned setting, where ligand coordinates must remain consistent with an explicit pocket geometry.

Structure-Based Drug Design. Early deep generative models for structure-based drug design (SBDD), such as 3D-SBDD [31] and Pocket2Mol [34], established spatial representations and architectures for pocket-conditioned generation. However, many of these methods rely on autoregressive atom or motif placement, which is prone to error accumulation along a fixed generation order. Non-autoregressive diffusion methods instead denoise all ligand atoms jointly. TargetDiff [14] introduced an SE(3)-equivariant formulation for target-aware 3D diffusion, while DiffSBDD [38] further developed equivariant diffusion models for structure-based generation. IPDiff [21] incorporates protein–ligand interaction priors, BindDM adaptively extracts interaction-relevant protein–ligand subcomplexes [20], PAFlow [49] adopts prior-guided flow matching, and PocketXMol studies direct clean-coordinate prediction in pocket-conditioned molecular design [33]. Flow-based formulations are attractive for sampling efficiency, but recent evidence suggests that low-step generation alone does not guarantee chemically reliable 3D geometry [32]. PoseCheck [16] analyzes generated protein– ligand poses directly and shows that physical violations or missing key interactions may be obscured when evaluation relies on redocking. GenBench3D [1] further emphasizes the 3D conformation quality of generated ligands. For example, it evaluates whether generated bond lengths and valence angles are consistent with reference distributions from 3D structure databases such as CSD [12] and LigBoundConf [42]. Together, these findings motivate our focus on the property–geometry trade-off: Vina-based docking scores and molecular properties should be interpreted together with pose and conformation quality.

Conditional Diffusion. Mainstream strategies for injecting conditional signals modify the reverse sampling trajectory via classifier guidance [6] or classifier-free guidance (CFG) [18]. However, applying these paradigms to 3D molecular design is non-trivial: stronger conditional pressure can degrade physical geometry, causing atom clashes or pushing molecules off-distribution [1]. TAGMol [7] uses external classifier guidance to steer sampling toward desired molecular properties, but this requires separate property predictors and scale calibration. BADGER [24] further studies guidance in diffusion-based SBDD, combining classifier guidance and classifier-free guidance for binding-affinity control and extending the same framework to joint optimization over affinity, QED, and SA. Other approaches, including alignment methods such as AliDiff [13] and reinforcementlearning-based guidance [50], directly optimize or fine-tune the generator toward preferred properties. In particular, we study this trade-off under a VE-style 3D backbone, where stronger conditional control may improve target objectives while introducing geometric drift.

## 3 Method

## 3.1 Overview

We model the denoising process over a hybrid state space that contains continuous coordinates and discrete atom types. The task of structure-based drug design (SBDD) is to learn a conditional distribution $p ( \mathcal { L } \mid \mathcal { \vec { P } } )$ over ligands conditioned on a protein pocket. A ligand with $N _ { L }$ atoms is represented as $\mathcal { L } = ( \mathbf { x } , \mathbf { z } )$ , where $\mathbf { x } \in \mathbb { R } ^ { N _ { L } \times 3 }$ are atom coordinates and $\mathbf { z } \in \{ 0 , 1 \} ^ { N _ { L } \times \bar { S } }$ are atomtype indicators. The protein pocket is written as $\mathcal { P } = ( \mathbf { y } , \mathbf { c } )$ , where $\mathbf { y } \in \mathbb { R } ^ { M \times 3 }$ are pocket atom coordinates and c are pocket atom features. Our goal is to model $p _ { \theta } ( \mathcal { L } \mid \mathcal { P } )$ , so that the generated ligand is both geometrically valid and compatible with the target pocket.

Our framework comprises three coupled components. It uses an EDM-style [27] generation backbone to stabilize continuous coordinate denoising, classifier-free guidance to control auxiliary property targets at inference time, and training-time protein perturbation to reduce over-reliance on a single rigid pocket realization.

## 3.2 EDM-style generation backbone

For the continuous coordinates, we use a variance-exploding (VE) forward process. Given a clean ligand coordinate set $\mathbf { x } _ { \mathrm { 0 } }$ , we sample a noise level $\sigma \sim p ( \sigma )$ and form

$$
\begin{array} { r } { \mathbf { x } _ { \sigma } = \mathbf { x } _ { 0 } + \sigma \boldsymbol { \epsilon } , \qquad \boldsymbol { \epsilon } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } ) . } \end{array}\tag{1}
$$

In practice, the noise scale σ is sampled from a log-normal distribution: $\sigma \sim$ LogNormal $\left( \ln \sqrt { \sigma _ { \mathrm { m i n } } \sigma _ { \mathrm { m a x } } } , \left[ \frac { 1 } { 8 } \ln ( \sigma _ { \mathrm { m a x } } / \sigma _ { \mathrm { m i n } } ) \right] ^ { 2 } \right)$ , so σ lies near the range $[ \sigma _ { \mathrm { m i n } } , \sigma _ { \mathrm { m a x } } ]$ in most cases and follows the usual VE/EDM-style noise schedule [27].

For atom types, we use a separate discrete corruption process. Following the discrete flow matching view [4], this branch learns a categorical denoising target over the same hybrid state space. We use the corruption kernel

$$
q ( \mathbf { z } _ { \sigma } \mid \mathbf { z } _ { 0 } ) = \left( 1 - m ( \sigma ) \right) \delta ( \mathbf { z } _ { \sigma } = \mathbf { z } _ { 0 } ) + m ( \sigma ) \operatorname { U n i f } ( K ) ,\tag{2}
$$

where $\mathrm { U n i f } ( K )$ is the uniform distribution over atom types. The masking schedule is $m ( \sigma ) =$ ln <sup>ln</sup> <sup>σ−ln</sup> <sup>σmin</sup> . This keeps the continuous coordinate diffusion and the discrete type diffusion aligned $\overline { { \sigma _ { \mathrm { m a x } } - \ln \sigma _ { \mathrm { m i n } } } }$ at the same noise level.

A small but important target-aware detail is the input scaling used before the equivariant backbone. In standard EDM, the noisy sample is often scaled as $\mathbf { x } _ { \sigma } / \sqrt { \sigma ^ { 2 } + \sigma _ { \mathrm { d a t a } } ^ { 2 } }$ before entering the backbone. In our setting, however, ligand coordinates interact with unscaled protein-pocket coordinates inside the same distance-based geometric graph. Directly using the standard factor would shrink low-noise ligand coordinates by $1 / \sigma _ { \mathrm { d a t a } }$ , while leaving protein coordinates unchanged, thereby distorting protein–ligand distances. We therefore use the scale-preserving input $\begin{array} { r } { \tilde { \mathbf { x } } _ { \sigma } = \frac { \sigma _ { \mathrm { d a t a } } } { \sqrt { \sigma ^ { 2 } + \sigma _ { \mathrm { d a t a } } ^ { 2 } } } \mathbf { x } _ { \sigma } } \end{array}$ , which recovers the original ligand scale as $\sigma  0$ and still attenuates high-noise inputs. This preserves the physical coordinate frame shared by ligand and pocket atoms. We use $\sigma _ { \mathrm { { d a t a } } } = 1 0 . 0$ as a fixed coordinate-scale parameter, rather than estimating it as a per-ligand sample variance or normalizing each ligand independently. This value covers both the internal ligand extent and the ligand displacement from the pocket center under pocket-centered normalization.

For the coordinate branch, we then use the EDM preconditioning form

$$
\begin{array} { r } { D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } ) = c _ { \mathrm { s k i p } } ( \sigma ) \mathbf { x } _ { \sigma } + c _ { \mathrm { o u t } } ( \sigma ) F _ { \theta } ( \tilde { \mathbf { x } } _ { \sigma } ; c _ { \mathrm { n o i s e } } ( \sigma ) , \mathbf { z } _ { \sigma } , \mathcal { P } ) . } \end{array}\tag{3}
$$

Here $F _ { \theta }$ is the equivariant backbone shared by both branches, and the skip term keeps the coordinate prediction anchored to the noisy input. This plays the same role as the preconditioning in VEDA [46], but here it is conditioned on the pocket. Further coefficient details are provided in Appendix B.

Atom types are recovered through a categorical prediction module built on top of the same equivariant features. Its output $H _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } ) \in \mathbf { \breve { R } } ^ { N _ { L } \times \dot { K } }$ gives the logits for the clean atom types. The clean ligand is then recovered by

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { 0 } = D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } ) , \qquad \hat { \mathbf { z } } _ { 0 } = \mathrm { s o f t m a x } \big ( H _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } ) \big ) . } \end{array}\tag{4}
$$

The training objective is a joint denoising loss over geometry and atom identity:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { b a s e } } ( \theta ) = \mathbb { E } _ { \mathcal { L } _ { 0 } , \mathcal { P } , \sigma } \bigg [ \lambda _ { x } ( \sigma ) \| \hat { \mathbf { x } } _ { 0 } - \mathbf { x } _ { 0 } \| _ { 2 } ^ { 2 } + \lambda _ { z } ( \sigma ) \operatorname { C E } ( \hat { \mathbf { z } } _ { 0 } , \mathbf { z } _ { 0 } ) \bigg ] . } \end{array}\tag{5}
$$

## 3.3 Sampling procedure

At inference time, we follow a fixed decreasing VE noise schedule. At each step, the coordinate branch predicts the clean structure and applies a first-order Euler update:

$$
\begin{array} { r } { \mathbf { d } _ { i } = \frac { \mathbf { x } _ { \sigma _ { i } } - D _ { \theta } \left( \mathbf { x } _ { \sigma _ { i } } , \mathbf { z } _ { \sigma _ { i } } , \sigma _ { i } , \mathcal { P } \right) } { \sigma _ { i } } , \qquad \mathbf { x } _ { \sigma _ { i + 1 } } = \mathbf { x } _ { \sigma _ { i } } + ( \sigma _ { i + 1 } - \sigma _ { i } ) \mathbf { d } _ { i } . } \end{array}\tag{6}
$$

Ligand coordinates are initialized around the pocket center, and ligand atom types are initialized from the uniform discrete prior. For the discrete atom types, we update $\mathbf { z } _ { \sigma _ { i } }$ with a DFM-based discrete sampler driven by the categorical predictor $H _ { \theta }$ . The number of ligand atoms is sampled from a prior conditioned on the estimated pocket size, matching the inference-time size prior used in TAGMol [7]. When classifier-free guidance is enabled, the same reverse trajectory is reused with a tunable guidance weight. The concrete noise range, generalized arcsin schedule, discrete-sampler details, and sampling-step ablations are provided in Appendix C and Appendix D.

## 3.4 Classifier-free guidance

We focus on three commonly used generation-quality objectives: Vina for predicted binding affinity, QED for drug-likeness, and normalized SA for synthetic accessibility. To make generation steerable, we discretize each target property into percentile bins computed on the training set and encode the resulting bin indices as an auxiliary condition a. Multi-objective control at inference time is specified by choosing one target bin per property and concatenating the resulting indices into the same auxiliary condition vector. We apply condition dropout [18] to this auxiliary input during training while always keeping the pocket condition $\mathcal { P }$ . For notational simplicity, this auxiliary input was suppressed in the backbone definition above and is written explicitly only in this subsection. Let ∅ denote the dropped auxiliary condition. Following the standard CFG notation [18], let w denote the extrapolation weight between the conditional and dropped-condition branches. In our implementation, however, we use the common sampling-time parameterization $s = w + 1$ , so that $s = 0$ corresponds to the null / unconditional branch, $s = 1$ recovers the plain conditional model, and $s > 1$ produces the usual CFG extrapolation. For comparison, an explicit classifier-guidance rule with the same scalar coefficient w appears in the VE sampler update as

$$
\mathbf { d } _ { \mathrm { g u i d e } } = \mathbf { d } _ { \mathrm { b a s e } } - w \sigma \nabla _ { \mathbf { x } _ { \sigma } } \log p _ { \phi } ( \mathbf { a } \mid \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } ) .\tag{7}
$$

Here $\mathbf { d } _ { \mathrm { g u i d e } }$ and $\mathbf { d } _ { \mathrm { b a s e } }$ denote the guided and unguided VE update fields, respectively. Thus, when w is used as an external guidance coefficient, the property gradient enters the sampler through a σ factor, making its effect noise-level dependent. Appendix A.1 gives the corresponding score-level derivation and the equivalent x<sub>0</sub>-space form. In addition, explicit classifier guidance requires objective-specific calibration of both scale and direction. Different property targets can live on very different numeric ranges, such as QED in $[ 0 , 1 ]$ versus Vina scores on a much wider scale, and some quantities are maximized while others, such as docking scores, are minimized. As a result, when w is used as an external guidance coefficient, its value and sign are generally not comparable across different property heads.

In contrast, CFG acts directly on the model predictions, which reduces the need for noise-leveldependent external-gradient calibration across different property metrics. Therefore, CFG induces the same linear extrapolation in the $x _ { 0 }$ prediction, but without introducing a separate property predictor. For atom types, we apply the same CFG rule in the logit space of the categorical predictor, which is the standard practical counterpart of CFG for discrete outputs. Under this parameterization, this becomes

$$
\begin{array} { r l } & { \hat { \mathbf { x } } _ { \theta } ^ { \mathrm { c f g } } = D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathcal { Q } ) + s ( D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathbf { a } ) - D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathcal { Q } ) ) , } \\ & { \hat { \mathbf { z } } _ { \theta } ^ { \mathrm { c f g } } = \mathrm { s o f t m a x } \Big ( H _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathcal { Q } ) + s ( H _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathbf { a } ) - H _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathcal { Q } ) ) \Big ) . } \end{array}\tag{8}
$$

Here the first line applies CFG to the coordinate prediction, while the second applies the same extrapolation to the type-logit branch before the softmax. We provide the corresponding score-level derivation and its relation to standard CFG formulas in Appendix A.2. Larger s makes the model follow the auxiliary property condition more strongly, while smaller s keeps the sample closer to the property-unconditional distribution under the same pocket.

## 3.5 Adaptive protein perturbation

The protein pocket is not treated as a perfectly clean ground truth during training. Experimental protein structures are subject to measurement error and capture only a static snapshot of a dynamic conformational ensemble. To account for this, we apply a small perturbation operator

$$
\tilde { \mathcal P } = \mathrm { P e r t u r b } ( \mathcal P ; \sigma _ { p } ) ,\tag{9}
$$

where $\sigma _ { p }$ controls the scale of Gaussian coordinate perturbation on the pocket atoms. We train the denoiser on $\tilde { \mathcal P }$ instead of ${ \mathcal P } .$ . This perturbation regularizes against over-reliance on a single crystallographic structure and encourages a neighborhood-aware conditional distribution. Following the classical connection between input noise and smoothness regularization [3, 5], a second-order expansion of a smooth denoising loss around the clean pocket shows that Gaussian pocket perturbation adds a leading-order penalty on the denoiser’s sensitivity to pocket coordinates. For a squared denoising loss, this leading term contains $\frac { \sigma _ { p } ^ { 2 } } { 2 } \lVert J _ { P } f _ { \theta } \rVert _ { F } ^ { 2 }$ , where $J _ { \mathcal { P } } f _ { \theta }$ is the Jacobian of the denoising prediction with respect to the pocket input. Thus, the perturbation discourages sharp changes in the coordinate and type predictions under small pocket-coordinate variations, while still preserving the pocket as the conditioning signal. For the main results, we use a bounded noise-level-dependent schedule $\sigma _ { p } = \operatorname* { m i n } ( 0 . 1 \sigma , 0 . 5 )$ . A formal statement and derivation are provided in Appendix E.

## 3.6 Full objective

The final training objective is

$$
\mathcal { L } ( \theta ) = \mathbb { E } _ { \mathcal { L } _ { 0 } , \mathcal { P } , \sigma , \tilde { \mathcal { P } } } \bigg [ \lambda _ { x } ( \sigma ) \| \hat { \mathbf { x } } _ { 0 } ^ { \tilde { \mathcal { P } } } - \mathbf { x } _ { 0 } \| _ { 2 } ^ { 2 } - \lambda _ { z } ( \sigma ) \sum _ { i = 1 } ^ { N _ { L } } \sum _ { k = 1 } ^ { K } ( \mathbf { z } _ { 0 } ) _ { i k } \log ( \hat { \mathbf { z } } _ { 0 } ^ { \tilde { \mathcal { P } } } ) _ { i k } \bigg ] ,\tag{10}
$$

where $\hat { \mathbf { x } } _ { 0 } ^ { \mathcal { \tilde { P } } }$ and $\hat { \mathbf { z } } _ { 0 } ^ { \tilde { \mathcal { P } } }$ denote the predictions calculated using the perturbed pocket $\tilde { \mathcal { P } } _ { \cdot }$ . At test time, the same model can be run either unguided or with classifier-free guidance, which gives a steerable trade-off between fidelity and target-specific preference.

## 4 Experiments

## 4.1 Experimental setup

Dataset We train our models on the CrossDocked2020 dataset [10]. Following prior work [31, 34, 14], we filter out binding poses with an RMSD > 1 Å and remove protein pairs with sequence identity > 30%, resulting in a high-quality subset common in target-aware 3D molecule generation. This split is the standard benchmark for target-aware 3D generation, but docking-centered evaluation alone does not fully characterize whether generated ligands are geometrically plausible. To address this, we further adopt the GenBench3D protocol [1], which is designed to assess 3D conformation quality and check whether molecules remain physically plausible under stronger conditional guidance.

Baselines We compare PocketVE against representative target-aware 3D molecule generation baselines, including AR [31], Pocket2Mol [34], TargetDiff [14], DecompDiff [15], IPDiff [21], PAFlow [49], SeFMol [48], and ALiDiff [13]. TAGMol [7] is the architectural parent and ablation reference because it shares the base equivariant denoiser with PocketVE; this comparison reflects a bundled system transition rather than an isolated VE effect. We use TargetDiff as the primary established geometry comparison and PAFlow as the strong-affinity comparison that exposes the affinity–geometry trade-off. For baselines with released generated molecules, we re-evaluate those outputs under the same test split, docking protocol, and GenBench3D pipeline used for PocketVE.

Metrics We evaluate generated molecules on 100 test proteins, sampling 100 molecules per protein, and report both mean and median results. For binding affinity, we use AutoDock Vina [8] under the common setup of Luo et al. [31] and Ragoza et al. [35], reporting Vina Score, Vina Min, and Vina Dock. We also report High Affinity, the per-pocket fraction of generated molecules whose Vina Dock score is no worse than the reference ligand, and Joint-Ref, the fraction that simultaneously satisfies no-worse Vina Dock, QED, and SA than the reference ligand. For molecular properties, we report QED [2], normalized SA [9], and diversity. For geometry, we report $\mathrm { V a l i d _ { 3 D } }$ , clash-free rate, strain energy, and centroid distance under the GenBench3D protocol [1]. We also include TargetDiff-style distribution diagnostics based on Jensen-Shannon Divergence (JSD) over empirical ligand distance and atom-type statistics [14]. These JSD metrics are complementary distribution-fidelity diagnostics rather than explicit physical-validity metrics.

Implementation details PocketVE uses the target-aware equivariant network architecture of TAG-Mol [7] as the base denoiser, while modifying the diffusion parameterization, guidance mechanism, sampling procedure, and optimizer. For conditional control, we use training-set labels for Vina, QED, and normalized SA, discretize each property independently into five percentile bins, and apply joint condition dropout with probability 0.5 during training. In the main conditional experiments, the sampling condition uses the most favorable target bin for each property, namely the lowest Vina bin and the highest QED and SA bins. We train for 400,000 optimization steps with batch size 4, validate every 2,000 steps, and report the checkpoint with the lowest validation loss. For optimization, we use Muon [26] for hidden two-dimensional weight matrices and AdamW [29] for the remaining parameters, with learning rates $1 0 ^ { - 3 }$ and $1 0 ^ { - 4 }$ , respectively. Both learning rates use a plateau-based schedule: if the validation loss does not improve for 20,000 steps, the learning rate is decayed by a factor of $0 . 6 .$ . For the main results, we use the noise-level-dependent pocket perturbation schedule $\sigma _ { p } = \operatorname* { m i n } ( 0 . 1 \sigma , 0 . 5 )$ ; alternative perturbation schedules are studied in Appendix Table 4. All main results use a 100-step sampler with the generalized arcsin schedule described in Appendix C. Across CFG scales, we use the same trained checkpoint and sampling hyperparameters, so changing the guidance scale only affects inference-time control. We provide the code for training, sampling, and evaluation in the supplementary material.

## 4.2 Main Results

Table 1: Summary of binding affinity, molecular properties, geometric quality, and diversity for reference molecules and molecules generated by PocketVE $( s = 5 )$ and other baselines. $( \uparrow ) / ( \downarrow )$ denotes a larger / smaller number is better. Top 2 results are highlighted with bold text and underlined text, respectively.
<table><tr><td rowspan="2">Method</td><td colspan="6">Vina Score  $( \downarrow )$  Vina Min (↓) Vina Dock (↓)</td><td rowspan="2">High Joint Aff. (↑)</td><td rowspan="2">QED Med. (↑)</td><td rowspan="2"></td><td rowspan="2"> $\mathtt { S A }$  Med. (↑)</td><td rowspan="2">Valid3D (↑)</td><td rowspan="2">Strain (↓)</td><td rowspan="2">Clash Free (↑)</td><td rowspan="2">Centroid  $( \downarrow )$ </td><td rowspan="2">Diversity Med. (↑)</td></tr><tr><td> $\operatorname { A v g } .$ </td><td>Med.</td><td> $\operatorname { A v g } .$ </td><td>Med.</td><td>Avg. Med.</td><td>Ref. (↑)</td></tr><tr><td>Ref</td><td>|-6.36</td><td>-6.46</td><td>-6.71</td><td>-6.49</td><td>|-7.45</td><td>-7.26</td><td></td><td></td><td>0.47</td><td>0.74</td><td>75</td><td>65.3</td><td>97</td><td>=</td><td>=</td></tr><tr><td>AR</td><td>-5.75</td><td>-5.64</td><td>-6.18-5.88</td><td></td><td>-6.75</td><td>-6.62</td><td>41.2</td><td>4.6</td><td>0.50</td><td>0.63</td><td>42.6</td><td>279.1</td><td>93.14</td><td>1.86</td><td>0.70</td></tr><tr><td>Pocket2Mol</td><td>-5.14</td><td>-4.70</td><td>-6.42 -5.82</td><td></td><td>-7.15</td><td>-6.79</td><td>48.0</td><td>16.5</td><td>0.57</td><td>0.75</td><td>57.3</td><td>130.6</td><td>81.03</td><td>1.82</td><td>0.71</td></tr><tr><td>TargetDiff</td><td>-5.47</td><td>-6.30</td><td>-6.64</td><td>-6.83</td><td>-7.80</td><td>-7.91</td><td>57.6</td><td>5.7</td><td>0.48</td><td>0.58</td><td>78.4</td><td>306.0</td><td>86.41</td><td>1.49</td><td>0.71</td></tr><tr><td>DecompDiff</td><td>-5.67</td><td>-6.04</td><td>-7.04 -7.09</td><td></td><td>-8.39</td><td>-8.43</td><td>63.8</td><td>6.8</td><td>0.43</td><td>0.60</td><td>71.5</td><td>318.4</td><td>78.00</td><td>2.90</td><td>0.68</td></tr><tr><td>IPDiff</td><td>-6.42</td><td>-7.01</td><td>-7.45</td><td>-7.48</td><td>-8.57</td><td>-8.51</td><td>68.2</td><td>10.0</td><td>0.53</td><td>0.59</td><td>59.6</td><td>1248.7</td><td>91.27</td><td>1.48</td><td>0.73</td></tr><tr><td>TAGMol</td><td>-7.02</td><td>-7.77</td><td>-7.95</td><td>-8.07</td><td>-8.59</td><td>-8.69</td><td>68.6</td><td>7.1</td><td>0.56</td><td>0.56</td><td>58.6</td><td>457.4</td><td>93.21</td><td>1.54</td><td>0.70</td></tr><tr><td>ALiDiff</td><td>-7.07</td><td>-7.95</td><td>-8.09</td><td>-8.17</td><td>-8.90</td><td>-8.81</td><td>68.5</td><td>7.0</td><td>0.50</td><td>0.56</td><td>46.4</td><td>1239.4</td><td>90.85</td><td>1.47</td><td>0.71</td></tr><tr><td>SeFMol</td><td>-7.23</td><td>-7.70</td><td>-8.03</td><td>-8.00</td><td>-8.72</td><td>-8.75</td><td>68.7</td><td>11.2 8.9</td><td>0.64</td><td>0.60</td><td>65.2</td><td>888.6</td><td>93.90</td><td>1.51</td><td>0.68</td></tr><tr><td>PAFlow</td><td>-8.31</td><td>-8.92</td><td>-8.79</td><td>-8.96</td><td>-9.46</td><td>-9.49</td><td>80.8</td><td></td><td>0.50</td><td>0.57</td><td>47.4</td><td>1834.6</td><td>96.46</td><td>1.55</td><td>0.70</td></tr><tr><td>Ours</td><td>-7.44</td><td>-7.89</td><td>-7.93</td><td>-8.19</td><td>-8.72</td><td>-8.89</td><td>73.9</td><td>26.4</td><td>0.64</td><td>0.71</td><td>80.6</td><td>127.9</td><td>94.02</td><td>1.26</td><td>0.71</td></tr></table>

We quantify the primary geometry comparison with a 10,000-replicate paired cluster bootstrap over complete pockets, using identical resampled pocket IDs for each method pair. Relative to TargetDiff, PocketVE maintains comparable $\mathrm { V a l i d _ { 3 D } }$ (+2.25 percentage points; 95% CI [−0.75, 5.13]) while substantially improving geometric stability. The corresponding strain reduction, defined as TargetDiff minus PocketVE, is 178.10 (95% CI [150.38, 203.44]), making lower strain the clearest sourcealigned gain. PocketVE also reports a lower median Vina score than TAGMol in Table 1; formal bootstrap conclusions focus on the source-aligned TargetDiff comparison.

Figure 2 makes the binding–geometry trade-off more explicit. Several strong-affinity baselines achieve low Vina scores but suffer from reduced $\mathrm { V a l i d _ { 3 D } }$ or high strain. In contrast, moderate CFG in PocketVE improves predicted binding while staying close to the high-validity, low-strain regime. PAFlow illustrates this trade-off most clearly: it ranks first on the Vina-based metrics, but its much lower $\mathrm { V a l i d _ { 3 D } }$ and substantially higher strain energy suggest that the predicted affinity gain comes with a large geometric cost. The corresponding QED–validity and SA–validity trade-off plots are provided in Appendix K.

Beyond the aggregate GenBench3D metrics, Figure 3 examines whether the geometric improvement is also visible at the distribution and fragment levels. The JSD diagnostics show that unguided PocketVE most closely matches the reference ligand distance and atom-type distributions, while increasing the CFG scale progressively moves samples away from the reference distribution. The rigid-fragment MMFF analysis gives a local check: compared with TargetDiff, Pocket2Mol, PAFlow, and TAGMol, PocketVE yields lower post-relaxation RMSD across most fragment sizes, especially for medium-to-large fragments, indicating more stable local geometry.

PAFlow  
Vina QED SA -8.11 0.77 0.81  
![](images/dc0cde6626d352372fd30e988742dcb259f90d34f9fa22ad3817eee70593165d.jpg)

![](images/0efc8e48e3817171bbb6a441ffce9a6bc7feaf1a750165a793db6e4d7392f614.jpg)  
Figure 2: Binding–geometry trade-offs across baselines and PocketVE guidance scales. Both panels use −Vina $\mathrm { S c o r e } _ { \mathrm { m e d } }$ on the x-axis, so larger values indicate stronger predicted binding. Left: ${ \mathrm { V a l i d } } _ { \mathrm { 3 D } } ,$ where higher is better. Right: strain energy on a log scale, where lower is better. AR and Pocket2Mol lie outside the plotted x-axis range because their median Vina scores are substantially weaker (−5.64 and −4.70, respectively), and are therefore shown as off-scale markers at the left boundary. PocketVE traces a steerable operating curve that preserves strong geometric quality under moderate CFG scales.

JSD Diagnostics
<table><tr><td>Method</td><td>| Local ↓</td><td>12Å↓</td><td>CC-2Å↓</td><td>Atom ↓</td></tr><tr><td>TAGMol</td><td>0.268</td><td>0.050</td><td>0.218</td><td>0.083</td></tr><tr><td>TargetDiff</td><td>0.234</td><td>0.056</td><td>0.213</td><td>0.059</td></tr><tr><td>Pocket2Mol</td><td>0.384</td><td>0.118</td><td>0.355</td><td>0.092</td></tr><tr><td>ALiDiff</td><td>0.341</td><td>0.097</td><td>0.350</td><td>0.202</td></tr><tr><td>IPDiff</td><td>0.391</td><td>0.098</td><td>0.349</td><td>0.194</td></tr><tr><td>PAFlow</td><td>0.454</td><td>0.141</td><td>0.461</td><td>0.228</td></tr><tr><td>Ours (s = 0)</td><td>0.191</td><td>0.035</td><td>0.120</td><td>0.047</td></tr><tr><td>Ours (s = 5)</td><td>0.255</td><td>0.065</td><td>0.269</td><td>0.120</td></tr><tr><td>Ours (s = 10)</td><td>0.284</td><td>0.067</td><td>0.292</td><td>0.136</td></tr><tr><td>Ours (s = 20)</td><td>0.326</td><td>0.068</td><td>0.306</td><td>0.124</td></tr></table>

Rigid-Fragment MMFF RMSD  
![](images/cacaf2a9db59503fc162c7d5d7d20f18074b83c6fb6e026640993e6e8b9dd225.jpg)  
Figure 3: Additional geometric-fidelity analyses. Left: JSD-based distribution diagnostics adapted from TargetDiff [14], where generated molecules are compared against empirical reference-ligand distributions from the test set. Lower values indicate better distribution fidelity. Right: Median pre/post-MMFF RMSD for rigid fragments across fragment sizes for representative baselines.

Figure 4 provides a representative visual comparison of the same trend. In this example, PocketVE places the ligand closer to the reference pose while maintaining a compact structure inside the pocket. Several baseline samples show larger pose deviations or less consistent pocket occupancy.

In summary, PocketVE improves geometric quality while retaining competitive docking and molecular-property scores rather than uniformly dominating every metric (Table 1). Figures 2, 3, and 4 provide aggregate, distributional, fragment-level, and qualitative views of this trade-off. Moderate CFG preserves geometric stability while shifting target-related properties, whereas stronger guidance can increase distribution drift and reduce parts of the diversity profile. Appendix N reports pocket-permutation and PoseCheck diagnostics, and Appendices I and J provide repeated-run and reference-normalized analyses.

Reference  
![](images/625037dea963e5c9b01cdc4d6300cb6ac0d107cfcc6988ef7dda06858513bd9a.jpg)

![](images/447e081a5e512fec81b48480ac26b5782afd5224a3c1d298ab05cdf015bb45fe.jpg)  
Vina QED SA -15.50 0.56 0.28

TargetDiff  
![](images/d273c7848045e55d63966ea7a3a1dc06a1cf7bdba1c2319948573a91adc84a04.jpg)  
Vina QED SA -12.62 0.250.44

![](images/f070d80d0c14dc0caac87e8ede9bb871464cc47c38c4a592e6a10403e90c83cc.jpg)  
Vina QED SA -11.49 0.65 0.55

![](images/f5269c03990209a8c2ba17a9fd73490d0f70a557f678ff01ad24aeb8016ffa49.jpg)  
Vina QED SA -14.43 0.520.34

AliDiff  
![](images/4aac6d902be16774def6c7e59353ad60a1b573b84069d50c90abf501b12668de.jpg)  
Vina QED SA -15.95 0.34 0.41

![](images/eac383cc5214c8d8c1b9512079be6717225a466be283ef12493c2a15d8292d1e.jpg)  
Vina QED SA -13.20 0.71 0.72

Figure 4: Visualizations of the reference ligand and generated ligands for a representative protein pocket (PDB ID: 5D7N) from baselines and PocketVE. Vina Dock, QED, and SA are reported below.

Table 2: Compact ablation and efficiency summary for PocketVE. The left table reports a cumulative ablation from TAGMol to the final PocketVE configuration: the PocketVE backbone row uses $s = 0 ,$ no protein perturbation, the log-uniform schedule, and the default sampling-time noise injection, and each subsequent row adds one component on top of the previous row. Runtimes are measured in seconds to sample 100 ligands per protein over 100 test proteins on a single NVIDIA A100 GPU with batch size 10 and no mixed precision, excluding docking, MMFF relaxation, and other post-processing.  
Cumulative Ablation
<table><tr><td>Setting</td><td>| Vina ↓</td><td>Valid3D ↑</td><td>Strain ↓</td><td>Clash-Free ↑</td></tr><tr><td>TAGMol (unguided)</td><td>-6.08</td><td>65.56</td><td>401.3</td><td>78.13</td></tr><tr><td>PocketVE backbone</td><td>-6.35</td><td>69.62</td><td>274.4</td><td>88.59</td></tr><tr><td>+ gen-arcsin scheduler</td><td>-6.76</td><td>77.75</td><td>179.1</td><td>91.87</td></tr><tr><td>+ CFG scale s = 1</td><td>-7.41</td><td>78.55</td><td>137.8</td><td>92.70</td></tr><tr><td>+ CFG scale s = 5</td><td>-7.78</td><td>77.33</td><td>125.9</td><td>94.07</td></tr><tr><td>+ protein perturb.</td><td>-7.89</td><td>80.60</td><td>127.9</td><td>94.02</td></tr></table>

Sampling Time
<table><tr><td>Method</td><td>Time (↓)</td></tr><tr><td>PocketVE (s = 1)</td><td>127.8</td></tr><tr><td>PocketVE (s = 5)</td><td>234.1</td></tr><tr><td>PAFlow</td><td>249.8</td></tr><tr><td>TargetDiff</td><td>1382.5</td></tr><tr><td>Pocket2Mol</td><td>1396.8</td></tr><tr><td>TAGMol</td><td>2433.3</td></tr></table>

## 4.3 Ablations and Analysis

We ablate the cumulative effect of the main components (Table 2), followed by two control axes that preserve the base architecture: the inference-time CFG scale and the training-time protein perturbation strategy.

Cumulative ablation The left panel of Table 2 shows that the PocketVE backbone and sampler yield a large geometric improvement over TAGMol, while moderate CFG mainly shifts affinity-related objectives. This comparison reflects an integrated transition in parameterization, scaling, noise, and sampling, rather than an isolated test of VE alone. Protein perturbation then improves the final geometry-oriented operating point, raising $\mathrm { V a l i d _ { 3 D } }$ while keeping other metrics nearly unchanged. A component-wise summary that groups the TAGMol baseline, unguided PocketVE, CFG variants, protein-perturbation variants, and schedule ablation is provided in Appendix H.

As a targeted coordinate-scale check, we also removed the $\sigma _ { \mathrm { d a t a } }$ factor in Eq. (32). In the corresponding diagnostic, mean Vina changes from −7.589 to +92.237, clash-free rate collapses from 94.23% to 3.39%, and centroid distance increases from 1.25 to 3.33 Å, while $\mathrm { V a l i d _ { 3 D } }$ remains 78.72% versus 82.39%. We therefore retain Eq. (32) as part of the integrated pocket-aware formulation, without attributing the full gain to this factor alone.

Sampling efficiency The PocketVE (s = 5) runtime on the right already includes the extra CFG passes and uses the same 100-step regime as PAFlow, VEDA [46], and SemlaFlow [22]. Under the same evaluation protocol, PocketVE remains substantially faster than most diffusion and autoregressive baselines and is slightly faster than PAFlow while remaining in the same runtime regime. The runtime comparison on the right should be read together with the sampling-step ablation in Appendix D. In our setting, the 100-step sampler is a deliberate operating point supported by that ablation, rather than a naive truncation of a longer trajectory.

Guidance scale sensitivity We sweep the sampling-time CFG scale s, where $s = 0$ uses the null/unconditional branch and s = 1 recovers the plain conditional model. As shown in Figure 2, increasing s first improves target-related objectives while largely preserving geometric quality. However, overly large guidance scales eventually reduce $\mathrm { V a l i d _ { 3 D } }$ and increase distribution drift. In our experiments, $s = 5$ provides the best overall balance, whereas $s = 1 0$ further favors affinityrelated objectives at a geometric cost. Detailed step-wise metrics and per-metric trend plots are provided in Appendix F.

We also evaluate property distributions, non-default requests, and conflicting conditions in Appendix O. These analyses show directional property shifts under CFG together with trade-offs in geometry, distributional fidelity, and diversity at stronger guidance scales.

Protein perturbation strategy To evaluate the impact of training-time protein-side perturbation, we compared fixed and noise-level-dependent schedules against a no-perturbation baseline. Protein perturbation acts as a regularizer that reshapes the property–geometry trade-off rather than uniformly dominating the no-perturbation baseline. Among the evaluated settings, the bounded noise-leveldependent schedule $\sigma _ { p } = \operatorname* { m i n } ( 0 . 1 \sigma , 0 . 5 )$ gives the best overall balance across local JSD, atom-type

JSD, Vina metrics, and centroid distance. Lighter perturbation further reduces strain, while stronger schedules improve broader distributional statistics such as JSD-All-12Å, JSD-CC-2Å, and SA at some cost in target-specific precision. Detailed JSD diagnostics, full GenBench3D metrics, and an additional test-time pocket perturbation study are provided in Appendix Tables 4 and 9.

## 5 Limitations

Our method has several limitations that affect its practical deployment. First, classifier-free guidance increases inference cost because each guided sampling step requires both a conditional and an unconditional forward pass. This doubles the number of function evaluations (NFEs) during guided generation compared with plain conditional sampling, which can become computationally expensive for long reverse trajectories or large-scale screening. Second, the preferred guidance scale may be target-dependent, and stronger guidance can trade geometric and distributional fidelity for targetrelated objectives. Third, the Gaussian pocket perturbation used here is a simple training regularizer rather than a realistic model of protein conformational flexibility. Finally, although PocketVE improves geometry on the evaluated benchmark, it relies on docking-based affinity proxies and benchmark-specific protocols. All principal results use the filtered 100-pocket CrossDocked2020 split, so broader generalization and experimental validation remain future work. The TAGMol-to-PocketVE transition combines several modeling and sampling choices; while our ablations provide evidence for individual design choices, they do not fully isolate the contribution of the VE formulation alone.

## 6 Conclusion

In this work, we introduced PocketVE, a target-aware 3D molecular generation framework that improves geometric stability while supporting inference-time conditional control. By revisiting coordinate parameterization, guidance design, and protein-side perturbation, PocketVE achieves higher structural quality in unguided generation and a more favorable trade-off between geometry and target objectives under guidance. Experiments on CrossDocked2020 demonstrate consistent gains in geometric stability together with directional, aggregate multi-property steering. Future work includes extending evaluation to broader target families, introducing matched component controls, and strengthening the connection between computational diagnostics and experimental validation.

## References

[1] Benoit Baillif, Jason Cole, Patrick McCabe, and Andreas Bender. Benchmarking structure-based three-dimensional molecular generative models using genbench3d: ligand conformation quality matters. arXiv preprint arXiv:2407.04424, 2024.

[2] G Richard Bickerton, Gaia V Paolini, Jérémy Besnard, Sorel Muresan, and Andrew L Hopkins. Quantifying the chemical beauty of drugs. Nature chemistry, 4(2):90–98, 2012.

[3] Chris M Bishop. Training with noise is equivalent to tikhonov regularization. Neural computation, 7(1):108–116, 1995.

[4] Andrew Campbell, Jason Yim, Regina Barzilay, Tom Rainforth, and Tommi Jaakkola. Generative flows on discrete state-spaces: enabling multimodal flows with applications to protein co-design. In Proceedings of the 41st International Conference on Machine Learning, pages 5453–5512, 2024.

[5] Olivier Chapelle, Jason Weston, Léon Bottou, and Vladimir Vapnik. Vicinal risk minimization. Advances in neural information processing systems, 13, 2000.

[6] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021.

[7] Vineeth Dorna, D Subhalingam, Keshav Kolluru, Shreshth Tuli, Mrityunjay Singh, Saurabh Singal, NM Anoop Krishnan, and Sayan Ranu. Tagmol: Target-aware gradient-guided molecule generation. In ICML’24 Workshop ML for Life and Material Science: From Theory to Industry Applications.

[8] Jerome Eberhardt, Diogo Santos-Martins, Andreas F Tillack, and Stefano Forli. Autodock vina 1.2. 0: new docking methods, expanded force field, and python bindings. Journal of chemical information and modeling, 61(8):3891–3898, 2021.

[9] Peter Ertl and Ansgar Schuffenhauer. Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions. Journal ofcheminformatics, 1(1):8, 2009.

[10] Paul G Francoeur, Tomohide Masuda, Jocelyn Sunseri, Andrew Jia, Richard B Iovanisci, Ian Snyder, and David R Koes. Three-dimensional convolutional neural networks and a crossdocked data set for structure-based drug design. Journal of chemical information and modeling, 60(9):4200–4215, 2020.

[11] Rafael Gómez-Bombarelli, Jennifer N Wei, David Duvenaud, José Miguel Hernández-Lobato, Benjamín Sánchez-Lengeling, Dennis Sheberla, Jorge Aguilera-Iparraguirre, Timothy D Hirzel, Ryan P Adams, and Alán Aspuru-Guzik. Automatic chemical design using a data-driven continuous representation of molecules. ACS central science, 4(2):268–276, 2018.

[12] Colin R Groom, Ian J Bruno, Matthew P Lightfoot, and Suzanna C Ward. The cambridge structural database. Structural Science, 72(2):171–179, 2016.

[13] Siyi Gu, Minkai Xu, Alexander Powers, Weili Nie, Tomas Geffner, Karsten Kreis, Jure Leskovec, Arash Vahdat, and Stefano Ermon. Aligning target-aware molecule diffusion models with exact energy optimization. Advances in Neural Information Processing Systems, 37:44040–44063, 2024.

[14] Jiaqi Guan, Wesley Wei Qian, Xingang Peng, Yufeng Su, Jian Peng, and Jianzhu Ma. 3d equivariant diffusion for target-aware molecule generation and affinity prediction. In The Eleventh International Conference on Learning Representations.

[15] Jiaqi Guan, Xiangxin Zhou, Yuwei Yang, Yu Bao, Jian Peng, Jianzhu Ma, Qiang Liu, Liang Wang, and Quanquan Gu. Decompdiff: diffusion models with decomposed priors for structurebased drug design. In Proceedings ofthe 40th International Conference on Machine Learning, pages 11827–11846, 2023.

[16] Charles Harris, Kieran Didi, Arian R Jamasb, Chaitanya K Joshi, Simon V Mathis, Pietro Lio, and Tom Blundell. Benchmarking generated poses: How rational is structure-based drug design with generative models? arXiv preprint arXiv:2308.07413, 2023.

[17] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.

[18] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022.

[19] Emiel Hoogeboom, Vıctor Garcia Satorras, Clément Vignac, and Max Welling. Equivariant diffusion for molecule generation in 3d. In International conference on machine learning, pages 8867–8887. PMLR, 2022.

[20] Zhilin Huang, Ling Yang, Zaixi Zhang, Xiangxin Zhou, Yu Bao, Xiawu Zheng, Yuwei Yang, Yu Wang, and Wenming Yang. Binding-adaptive diffusion models for structure-based drug design. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 12671–12679, 2024.

[21] Zhilin Huang, Ling Yang, Xiangxin Zhou, Zhilong Zhang, Wentao Zhang, Xiawu Zheng, Jie Chen, Yu Wang, Bin Cui, and Wenming Yang. Protein-ligand interaction prior for bindingaware 3d molecule diffusion models. In The Twelfth International Conference on Learning Representations, 2024.

[22] Ross Irwin, Alessandro Tibo, Jon Paul Janet, and Simon Olsson. Semlaflow–efficient 3d molecular generation with latent attention and equivariant flow matching. In International Conference on Artificial Intelligence and Statistics, pages 3772–3780. PMLR, 2025.

[23] Clemens Isert, Kenneth Atz, and Gisbert Schneider. Structure-based drug design with geometric deep learning. Current Opinion in Structural Biology, 79:102548, 2023.

[24] Yue Jian, Curtis Wu, Danny Reidenbach, and Aditi S Krishnapriyan. General binding affinity guidance for diffusion models in structure-based drug design. Journal of Chemical Information and Modeling, 2026.

[25] Wengong Jin, Regina Barzilay, and Tommi Jaakkola. Junction tree variational autoencoder for molecular graph generation. In International conference on machine learning, pages 2323–2332. PMLR, 2018.

[26] Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks, 2024.

[27] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. Advances in neural information processing systems, 35:26565–26577, 2022.

[28] Hannes H Loeffler, Jiazhen He, Alessandro Tibo, Jon Paul Janet, Alexey Voronov, Lewis H Mervin, and Ola Engkvist. Reinvent 4: Modern ai–driven generative molecule design. Journal ofCheminformatics, 16(1):20, 2024.

[29] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations.

[30] Aaron Lou and Stefano Ermon. Reflected diffusion models. In International Conference on Machine Learning, pages 22675–22701. PMLR, 2023.

[31] Shitong Luo, Jiaqi Guan, Jianzhu Ma, and Jian Peng. A 3d generative model for structure-based drug design. Advances in Neural Information Processing Systems, 34:6229–6239, 2021.

[32] Filipp Nikitin, Ian Dunn, David Ryan Koes, and Olexandr Isayev. Geom-drugs revisited: toward more chemically accurate benchmarks for 3d molecule generation. Digital Discovery, 4(11):3282–3291, 2025.

[33] Xingang Peng, Ruihan Guo, Fenglin Guo, Ziyi Wang, Jiayu Sun, Jiaqi Guan, Yinjun Jia, Yan Xu, Yanwen Huang, Muhan Zhang, et al. Unified modeling of 3d molecular generation via atomic interactions with pocketxmol. Cell, 2026.

[34] Xingang Peng, Shitong Luo, Jiaqi Guan, Qi Xie, Jian Peng, and Jianzhu Ma. Pocket2mol: Efficient molecular sampling based on 3d protein pockets. In International conference on machine learning, pages 17644–17655. PMLR, 2022.

[35] Matthew Ragoza, Tomohide Masuda, and David Ryan Koes. Generating 3d molecules conditional on receptor binding sites with deep generative models. Chemical science, 13(9):2701– 2713, 2022.

[36] Kevin Rojas, Ye He, Chieh-Hsin Lai, Yuhta Takida, Yuki Mitsufuji, and Molei Tao. Improving classifier-free guidance in masked diffusion: Low-dim theoretical insights with high-dim impact. In The Fourteenth International Conference on Learning Representations, 2026.

[37] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.

[38] Arne Schneuing, Charles Harris, Yuanqi Du, Kieran Didi, Arian Jamasb, Ilia Igashov, Weitao Du, Carla Gomes, Tom L Blundell, Pietro Lio, et al. Structure-based drug design with equivariant diffusion models. Nature Computational Science, 4(12):899–909, 2024.

[39] Chence Shi, Minkai Xu, Zhaocheng Zhu, Weinan Zhang, Ming Zhang, and Jian Tang. Graphaf: a flow-based autoregressive model for molecular graph generation. In International Conference on Learning Representations.

[40] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. Score-based generative modeling through stochastic differential equations. In International Conference on Learning Representations.

[41] Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In 2015 ieee information theory workshop (itw), pages 1–5. Ieee, 2015.

[42] Jiahui Tong and Suwen Zhao. Large-scale analysis of bioactive ligand conformational strain energy by ab initio calculation. Journal ofChemical Information and Modeling, 61(3):1180– 1192, 2021.

[43] Clement Vignac, Nagham Osman, Laura Toni, and Pascal Frossard. Midi: Mixed graph and 3d denoising diffusion for molecule generation. In Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pages 560–576. Springer, 2023.

[44] Minkai Xu, Alexander S Powers, Ron O Dror, Stefano Ermon, and Jure Leskovec. Geometric latent diffusion models for 3d molecule generation. In International Conference on Machine Learning, pages 38592–38610. PMLR, 2023.

[45] Peining Zhang, Daniel Baker, Minghu Song, and Jinbo Bi. Unraveling the potential of diffusion models in small-molecule generation. Drug Discovery Today, 30(7):104413, 2025.

[46] Peining Zhang, Jinbo Bi, and Minghu Song. Veda: Generation of 3d molecules via varianceexploding diffusion with annealing. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 28346–28354, 2026.

[47] Pengze Zhang, Hubery Yin, Chen Li, and Xiaohua Xie. Tackling the singularities at the endpoints of time intervals in diffusion models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6945–6954, 2024.

[48] Xudong Zhang, Sanqing Qu, Fan Lu, Jianmin Wang, Zhixin Tian, Shangding Gu, Yanping Zhang, Alois Knoll, Shaorong Gao, Guang Chen, et al. Steering semi-flexible molecular diffusion model for structure-based drug design with reinforcement learning. Science Advances, 12(16):eady9955, 2026.

[49] Jingyuan Zhou, Hao Qian, Shikui Tu, and Lei Xu. Prior-guided flow matching for target-aware molecule design with learnable atom number. arXiv preprint arXiv:2509.01486, 2025.

[50] Zhijian Zhou, Junyi An, Zongkai Liu, Yunfei Shi, Xuan Zhang, Fenglei Cao, Chao Qu, and Yuan Qi. Guiding diffusion models with reinforcement learning for stable molecule generation. arXiv preprint arXiv:2508.16521, 2025.

## A Classifier-Free Guidance Details

For completeness, we record the standard score-level form of classifier-free guidance used to motivate the implementation in the main text.

## A.1 Noise-Level Dependence of Explicit Classifier Guidance

Let $\mathcal { L } _ { \sigma } = ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } )$ denote the noisy ligand state at noise level σ. The guided target distribution can be written as

$$
\begin{array} { r } { \log p _ { w } ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , { \mathbf a } ) = ( 1 + w ) \log p ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , { \mathbf a } ) - w \log p ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \mathcal { Q } ) + \mathrm { c o n s t . } } \end{array}\tag{11}
$$

For the continuous coordinate branch, taking the score with respect to $\mathbf { x } _ { \sigma }$ yields

$$
\nabla _ { \mathbf { x } _ { \sigma } } \log p _ { w } ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \mathbf { a } ) = ( 1 + w ) \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \mathbf { a } ) - w \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \emptyset ) .\tag{12}
$$

At the score level, both approaches can be read through the conditional decomposition

$$
\nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \mathbf { a } ) = \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \emptyset ) + \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathbf { a } \mid \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } ) ,\tag{13}
$$

which, when substituted into Eq. (12), gives

$$
\begin{array} { r l } & { \nabla _ { \mathbf { x } _ { \sigma } } \log p _ { w } ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \mathbf { a } ) = ( 1 + w ) \Big ( \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \mathcal { P } ) + \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathbf { a } \mid \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } ) \Big ) } \\ & { \qquad - w \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \mathcal { Q } ) } \\ & { \qquad = \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \mathcal { Q } ) + ( 1 + w ) \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathbf { a } \mid \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } ) . } \end{array}\tag{14}
$$

This makes explicit that extrapolating the conditional score away from the dropped-condition score amplifies the same property-dependent term used by external guidance methods. Explicit classifier guidance follows the same decomposition, but replaces that attribute-dependent term with the gradient of an external classifier or property predictor. More concretely, it keeps the base generative score $s _ { \mathrm { b a s e } } ( \mathbf { x } _ { \sigma } ) \approx \nabla _ { \mathbf { x } _ { \epsilon } }$ log $p ( \mathcal { L } _ { \sigma } \mid \mathcal { P } , \emptyset )$ and approximates the attribute-dependent term by $\nabla _ { \mathbf { x } _ { \sigma } } \log p _ { \phi } ( \mathbf { a } \ |$ $\mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } ) \approx \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathbf { a } \mid \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } )$ . Introducing the same scalar coefficient w as an external guidance strength then gives

$$
\begin{array} { r } { s _ { \mathrm { g u i d e } } ( \mathbf { x } _ { \sigma } ) = s _ { \mathrm { b a s e } } ( \mathbf { x } _ { \sigma } ) + ( 1 + w ) \nabla _ { \mathbf { x } _ { \sigma } } \log p _ { \phi } ( \mathbf { a } \mid \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } ) , } \end{array}\tag{15}
$$

where $s _ { \mathrm { b a s e } }$ denotes the generative score and $p _ { \phi }$ denotes an external classifier or property predictor. When taking the score with respect to $\mathbf { x } _ { \sigma }$ , we treat $\mathbf { z } _ { \sigma }$ as part of the conditioning context of the joint state. Under the VE parameterization, this relation follows from the standard Tweedie identity. The forward process is ${ \bf x } _ { \sigma } = { \bf x } _ { 0 } + \sigma \epsilon$ $\epsilon \sim \mathcal { N } ( 0 , I )$ , so $p ( \mathbf { x } _ { \sigma } \mid \mathbf { x } _ { 0 } ) = \mathcal { N } ( \mathbf { x } _ { 0 } , \sigma ^ { 2 } I )$ . For fixed conditioning context $( \mathbf { z } _ { \sigma } , \mathcal { P } , \mathbf { a } )$ , Tweedie’s formula gives

$$
\begin{array} { r } { \mathbb { E } [ \mathbf { x } _ { 0 } \mid \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } , \mathbf { a } ] = \mathbf { x } _ { \sigma } + \sigma ^ { 2 } \nabla _ { \mathbf { x } _ { \sigma } } \log p ( \mathbf { x } _ { \sigma } \mid \mathbf { z } _ { \sigma } , \mathcal { P } , \mathbf { a } ) . } \end{array}\tag{16}
$$

Approximating this posterior mean with the denoiser prediction $D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathbf { a } ) \approx \mathbb { E } [ \mathbf { x } _ { 0 }$ | $\mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } , \mathbf { a } ]$ and rearranging yields

$$
\nabla _ { \mathbf x _ { \sigma } } \log p ( \mathbf x _ { \sigma } \mid \mathbf z _ { \sigma } , \mathcal P , \mathbf a ) \approx \frac { D _ { \theta } ( \mathbf x _ { \sigma } , \mathbf z _ { \sigma } , \sigma , \mathcal P , \mathbf a ) - \mathbf x _ { \sigma } } { \sigma ^ { 2 } } .\tag{17}
$$

Applying the same VE identity to the explicit classifier-guidance score in Eq. (15) yields the equivalent x<sub>0</sub>-space correction

$$
\begin{array} { r } { D _ { \mathrm { g u i d e } } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } ) \approx D _ { \mathrm { b a s e } } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } ) + w \sigma ^ { 2 } \nabla _ { \mathbf { x } _ { \sigma } } \log p _ { \phi } ( \mathbf { a } \mid \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } ) . } \end{array}\tag{18}
$$

Applying Eq. (17) to both the conditional and dropped-condition branches in Eq. (12) gives

$$
\nabla _ { \mathbf { x } _ { \sigma } } \log p _ { w } \approx \frac { ( 1 + w ) D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathbf { a } ) - w D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathcal { Q } ) - \mathbf { x } _ { \sigma } } { \sigma ^ { 2 } } .\tag{19}
$$

Under the probability-flow ODE parameterization of VE models [40, 27], the reverse update field induced by a clean prediction xˆ<sub>0</sub> can be written as

$$
\mathbf { d } = \frac { \mathbf { x } _ { \sigma } - \hat { \mathbf { x } } _ { 0 } } { \sigma } .\tag{20}
$$

Applying this identity to the base predictor $D _ { \mathrm { b a s e } } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } )$ gives the unguided field

$$
\mathbf { d } _ { \mathrm { b a s e } } = \frac { \mathbf { x } _ { \sigma } - D _ { \mathrm { b a s e } } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } ) } { \mathbf { \sigma } } .\tag{21}
$$

Substituting the classifier-guided clean prediction from Eq. (18) then recovers the main-text update rule

$$
\mathbf { d } _ { \mathrm { g u i d e } } = \mathbf { d } _ { \mathrm { b a s e } } - w \sigma \nabla _ { \mathbf { x } _ { \sigma } } \log p _ { \phi } ( \mathbf { a } \mid \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \mathcal { P } ) ,\tag{22}
$$

which is Eq. (7) in the main text. This makes the noise-level dependence explicit: an external classifier gradient enters the denoiser through a $\sigma ^ { 2 }$ factor, and the sampler update through a σ factor.

## A.2 Recovering the Classifier-Free Guidance Rule

Returning to Eq. (12), we now recover the main-text implementation of classifier-free guidance. Comparing the expression above with the VE identity

$$
\nabla _ { \mathbf { x } _ { \sigma } } \log p _ { w } = \frac { \hat { \mathbf { x } } _ { \theta } ^ { \mathrm { c f g } } - \mathbf { x } _ { \sigma } } { \sigma ^ { 2 } } ,\tag{23}
$$

we identify the guided clean prediction as

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { \theta } ^ { \mathrm { c f g } } = ( 1 + w ) D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathbf { a } ) - w D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathcal { O } ) . } \end{array}\tag{24}
$$

Using the implementation parameterization $s = w + 1$ (equivalently, $w = s - 1 )$ then yields

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { \theta } ^ { \mathrm { c f g } } = D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathcal { O } ) + s ( D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathbf { a } ) - D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathcal { O } ) ) , } \end{array}\tag{25}
$$

which is exactly the coordinate branch in Eq. (8) of the main text. For the discrete branch, we do not use a continuous score over $\mathbf { z } _ { \sigma }$ . Instead, we apply the same conditional / dropped-condition extrapolation directly to the categorical logits:

$$
\begin{array} { r } { \hat { H } _ { \theta } ^ { \mathrm { c f g } } = H _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathcal { Q } ) + s ( H _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathbf { a } ) - H _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } , \mathcal { Q } ) ) . } \end{array}\tag{26}
$$

Applying softmax $( { \hat { H } } _ { \theta } ^ { \mathrm { c f g } } )$ then recovers the discrete branch in Eq. (8).

## B EDM Preconditioning and Target-Aware Coordinate Scaling

This section summarizes the EDM preconditioning recipe used by PocketVE and distinguishes the standard VE/EDM machinery from the target-aware adaptation introduced by our pocket-conditioned setting. The coefficient definitions themselves are standard EDM/Karras-style components reused from VEDA [27, 46]; the main PocketVE-specific point is how the ligand coordinates are scaled before entering the shared protein–ligand geometric graph.

Under the VE corruption process, clean ligand coordinates are perturbed as

$$
\begin{array} { r } { \mathbf { x } _ { \sigma } = \mathbf { x } _ { 0 } + \sigma { \boldsymbol { \epsilon } } , \qquad { \boldsymbol { \epsilon } } \sim \mathcal { N } ( \mathbf { 0 } , I ) . } \end{array}\tag{27}
$$

Standard EDM writes the coordinate denoiser in the preconditioned form

$$
D _ { \theta } ( \mathbf { x } _ { \sigma } , \sigma ) = c _ { \mathrm { s k i p } } ( \sigma ) \mathbf { x } _ { \sigma } + c _ { \mathrm { o u t } } ( \sigma ) F _ { \theta } ( c _ { \mathrm { i n } } ( \sigma ) \mathbf { x } _ { \sigma } , c _ { \mathrm { n o i s e } } ( \sigma ) ) ,\tag{28}
$$

with

$$
c _ { \mathrm { s k i p } } ( \sigma ) = \frac { \sigma _ { \mathrm { d a t a } } ^ { 2 } } { \sigma ^ { 2 } + \sigma _ { \mathrm { d a t a } } ^ { 2 } } , \qquad c _ { \mathrm { o u t } } ( \sigma ) = \frac { \sigma \sigma _ { \mathrm { d a t a } } } { \sqrt { \sigma ^ { 2 } + \sigma _ { \mathrm { d a t a } } ^ { 2 } } } ,\tag{29}
$$

$$
c _ { \mathrm { i n } } ( \sigma ) = \frac { 1 } { \sqrt { \sigma ^ { 2 } + \sigma _ { \mathrm { d a t a } } ^ { 2 } } } , \qquad c _ { \mathrm { n o i s e } } ( \sigma ) = \frac { \log \sigma } { 4 } .\tag{30}
$$

Here $c _ { \mathrm { i n } }$ normalizes the noisy input magnitude, $c _ { \mathrm { s k i p } }$ keeps the prediction anchored to the noisy sample at low noise, $c _ { \mathrm { o u t } }$ rescales the network residual back to the clean-coordinate domain, and c<sub>noise</sub> provides the scalar noise embedding.

PocketVE keeps the standard output preconditioning and noise embedding, while modifying how the EDM input scaling is applied before the pocket–ligand graph. At the notation level used in the main text, this is written as

$$
\begin{array} { r } { D _ { \theta } ( \mathbf { x } _ { \sigma } , \mathbf { z } _ { \sigma } , \sigma , \mathcal { P } ) = c _ { \mathrm { s k i p } } ( \sigma ) \mathbf { x } _ { \sigma } + c _ { \mathrm { o u t } } ( \sigma ) F _ { \theta } ( \tilde { \mathbf { x } } _ { \sigma } ; c _ { \mathrm { n o i s e } } ( \sigma ) , \mathbf { z } _ { \sigma } , \mathcal { P } ) . } \end{array}\tag{31}
$$

In the code, this is realized through an equivalent two-stage implementation: the network first operates on the scaled ligand coordinates

$$
\begin{array} { r } { \mathbf { x } _ { \mathrm { n e t } } = \sigma _ { \mathrm { d a t a } } c _ { \mathrm { i n } } ( \sigma ) \mathbf { x } _ { \sigma } = \tilde { \mathbf { x } } _ { \sigma } , } \end{array}\tag{32}
$$

and predicts the coordinates in that scaled frame as

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { \mathrm { n e t } } = F _ { \theta } ( \mathbf { x } _ { \mathrm { n e t } } ; c _ { \mathrm { n o i s e } } ( \sigma ) , \mathbf { z } _ { \sigma } , \mathcal { P } ) , } \end{array}\tag{33}
$$

and the final coordinate prediction in the default constant mode is reconstructed as

$$
\begin{array} { r } { \hat { \mathbf { x } } _ { 0 } = c _ { \mathrm { s k i p } } ( \sigma ) \mathbf { x } _ { \sigma } + c _ { \mathrm { o u t } } ( \sigma ) \big ( \hat { \mathbf { x } } _ { \mathrm { n e t } } - \mathbf { x } _ { \mathrm { n e t } } \big ) . } \end{array}\tag{34}
$$

This makes explicit that the network prediction is interpreted relative to the scaled ligand input before being mapped back to the original coordinate frame. The discrete atom-type branch is handled separately through the DFM-based categorical predictor, so the present discussion is only about the continuous coordinate preconditioning.

The target-aware adaptation appears in the network input. In textbook EDM, one would feed $c _ { \mathrm { i n } } ( \sigma ) \mathbf { x } _ { \sigma }$ into the coordinate backbone. In the actual PocketVE implementation, however, the ligand coordinates are effectively fed as

$$
\sigma _ { \mathrm { d a t a } } c _ { \mathrm { i n } } ( \sigma ) \mathbf { x } _ { \sigma } = \frac { \sigma _ { \mathrm { d a t a } } } { \sqrt { \sigma ^ { 2 } + \sigma _ { \mathrm { d a t a } } ^ { 2 } } } \mathbf { x } _ { \sigma } = \tilde { \mathbf { x } } _ { \sigma } .\tag{35}
$$

The same scaled input is used as the reference point in the output reconstruction, so the model predicts a correction relative to the scaled ligand coordinates rather than absolute coordinates in the network frame. The design is necessary because ligand and protein coordinates interact inside the same distance-based equivariant graph, while the pocket coordinates are not scaled by EDM coefficients. If the ligand alone were shrunk by the textbook $c _ { \mathrm { i n } } ( \sigma )$ factor, then at low noise the ligand would be rescaled by approximately $1 / \sigma _ { \mathrm { d a t a } }$ while the pocket remained in the original coordinate frame, distorting protein–ligand distances. The scale-preserving input together with the matching output recentering avoids that mismatch. $\mathbf { A s } \sigma  0$ , it satisfies $\tilde { \mathbf { x } } _ { \sigma }  \mathbf { x } _ { \sigma }$ , so the ligand remains in the same physical frame as the pocket near the clean limit, while at larger noise levels it still attenuates the coordinate magnitude in the same spirit as EDM.

This target-aware scaling also explains why PocketVE uses a comparatively large $\sigma _ { \mathrm { d a t a } }$ . Our coordinates are normalized around the pocket center rather than around each ligand itself, so $\sigma _ { \mathrm { d a t a } }$ must cover both the internal ligand extent and the ligand displacement from the pocket center. For the same reason, we keep $\sigma _ { \mathrm { m i n } } \mathrm { ^ { - } = 1 0 ^ { - 3 } }$ small enough to preserve near-clean coordinate precision in the final reverse steps. For training, the continuous coordinate branch uses the EDM-style weighting induced by the implementation loss $\| \hat { \mathbf { x } } _ { 0 } - \mathbf { x } _ { 0 } \| _ { 2 } ^ { 2 } / c _ { \mathrm { o u t } } ( \sigma ) ^ { 2 }$ , while the categorical branch uses a separately weighted cross-entropy term. The concrete dataset statistics behind these choices, together with the full schedule specification, are given in Appendix C.

## C Sampling Details

Training was run on a single NVIDIA A100 GPU for approximately 24 hours. The host machine used four AMD EPYC 7513 32-Core Processor CPUs and 100 GB of memory.

For the VE coordinate branch, we set $\sigma _ { \mathrm { m a x } } = 8 0 0 , \sigma _ { \mathrm { m i n } } = 1 0 ^ { - 3 }$ , and $\sigma _ { \mathrm { { d a t a } } } = 1 0 . 0$ . The choice of $\sigma _ { \mathrm { { d a t a } } } = 1 0 . 0$ is target-aware rather than ligand-only. On the training set, the per-ligand coordinate spread has mean and median standard deviation 3.82 Å, with a maximum of 9.91 Å. Because our coordinates are centered by the protein pocket rather than by each ligand itself, the relevant cleandata scale must also cover ligand displacement from the pocket center. Under this normalization, the ligand-center distance has mean 2.31 Å, median 1.98 Å, and maximum 9.64 Å. We therefore choose $\sigma _ { \mathrm { { d a t a } } } = 1 0 . 0$ so that the preconditioning scale covers both the internal ligand extent and the pocket-relative ligand offset. We also keep $\bar { \sigma } _ { \mathrm { m i n } } = 1 0 ^ { - 3 }$ small enough to preserve near-clean coordinate precision at the end of the reverse trajectory.

We traverse the noise levels with a generalized arcsin schedule, which extends the arcsin scheduler used in VEDA [46]. Let $\tau _ { i } = 1 - i / N$ and

$$
\sigma _ { i } = \exp \Biggl ( \log \sigma _ { \operatorname* { m i n } } + \left[ ( 1 - \beta ) \tau _ { i } + \beta \frac { 2 } { \pi } \arcsin ( \tau _ { i } ^ { p } ) \right] \bigl ( \log \sigma _ { \operatorname* { m a x } } - \log \sigma _ { \operatorname* { m i n } } \bigr ) \Biggr ) ,\tag{36}
$$

where we use $\beta = 2 . 2$ and $p = 0 . 5 7$ in all experiments. Let $\sigma _ { 0 } > \sigma _ { 1 } > \cdot \cdot \cdot > \sigma _ { N }$ denote the resulting noise schedule.

For the discrete atom types, we first convert the type head into a clean categorical prediction

$$
\mathbf { p } _ { \theta , i } = \mathrm { s o f t m a x } \big ( H _ { \theta } ( \mathbf { x } _ { \sigma _ { i } } , \mathbf { z } _ { \sigma _ { i } } , \sigma _ { i } , \mathcal { P } ) \big ) .\tag{37}
$$

We then update $\mathbf { z } _ { \sigma _ { i } }$ with a discrete sampler based on the DFM transition-rate formulation [4]. In this view, the transition rate toward category j is

$$
R _ { \theta } ( z _ { \sigma _ { i } } , j ) = \omega ( \sigma _ { i } ) p _ { \theta } ( z _ { 0 } = j \mid z _ { \sigma _ { i } } ) + \eta _ { i } p _ { \theta } ( z _ { 0 } = z _ { \sigma _ { i } } \mid z _ { \sigma _ { i } } ) ,\tag{38}
$$

where

$$
\omega ( \sigma _ { i } ) = \frac { \eta _ { i } S ( 1 - m ( \sigma _ { i } ) ) + \eta _ { i } m ( \sigma _ { i } ) + m ^ { \prime } ( \sigma _ { i } ) } { m ( \sigma _ { i } ) } ,\tag{39}
$$

S is the number of atom categories, $p _ { \theta } ( z _ { 0 } = \cdot \mid z _ { \sigma _ { i } } )$ is given by the predicted categorical distribution from Eq. (37), and we follow the variable-noise setting in [4] with $\eta _ { i } = \eta / m ^ { \prime } ( \sigma _ { i } )$ . Equivalently, this step can be interpreted as masked-token refinement under the same schedule $m ( \sigma )$

## D Sampling-Step Ablation

We ablate the number of reverse sampling steps using representative CFG scales $s \in \{ 0 , 1 , 5 , 1 0 \}$ Figure 5 shows that moving from 25 to 100 steps improves both predicted binding strength and 3D validity across guidance scales, while increasing from 100 to 200 steps gives smaller additional gains relative to the extra sampling cost. We therefore use $N = 1 0 0$ as the default operating point in the main experiments.

![](images/c4e4e22843a11105d321fbde558740d8abb600935cf4c0460223a49b43b10be3.jpg)

![](images/05e08624e40f4ae02f05c59a19a5d08f5d2c11f5e2e9e000647424badbac21cc.jpg)  
Figure 5: Sampling-step ablation under representative CFG scales. Left: median Vina Score is shown as −Vina $\mathrm { S c o r e } _ { \mathrm { m e d } } ,$ so higher is better. Right: $\mathrm { V a l i d _ { 3 D } }$ under the GenBench3D protocol. The vertical dashed line marks the default setting $N = 1 0 0$

Table 3: Sampling-step ablation for representative CFG scales. We report median Vina Score, $\mathrm { V a l i d _ { 3 D } }$ and median strain energy. Lower Vina Score and strain energy are better; higher $\mathrm { V a l i d _ { \mathrm { 3 D } } }$ is better.
<table><tr><td>S</td><td>Steps</td><td>Vina Score (↓)</td><td> $\mathrm { V a l i d } _ { \mathrm { 3 D } } \left( \uparrow \right)$ </td><td>Strain (↓)</td></tr><tr><td>0</td><td>25</td><td>-6.34</td><td>63.89</td><td>398.3</td></tr><tr><td>0</td><td>50</td><td>-6.54</td><td>73.70</td><td>231.3</td></tr><tr><td>0</td><td>100</td><td>-6.70</td><td>78.60</td><td>164.9</td></tr><tr><td>0</td><td>200</td><td>-6.79</td><td>80.29</td><td>144.4</td></tr><tr><td>1</td><td>25</td><td>-6.73</td><td>67.17</td><td>324.1</td></tr><tr><td>1</td><td>50</td><td>-7.09</td><td>76.10</td><td>178.9</td></tr><tr><td>1</td><td>100</td><td>-7.20</td><td>78.00</td><td>127.4</td></tr><tr><td>1</td><td>200</td><td>-7.40</td><td>80.24</td><td>116.5</td></tr><tr><td>5</td><td>25</td><td>-7.45</td><td>66.40</td><td>293.0</td></tr><tr><td>5</td><td>50</td><td>-7.52</td><td>77.20</td><td>158.8</td></tr><tr><td>5</td><td>100</td><td>-7.89</td><td>80.60</td><td>127.9</td></tr><tr><td>5</td><td>200</td><td>-7.77</td><td>80.37</td><td>111.2</td></tr><tr><td>10</td><td>25</td><td>-7.62</td><td>66.93</td><td>326.5</td></tr><tr><td>10</td><td>50</td><td>-7.77</td><td>75.40</td><td>172.7</td></tr><tr><td>10</td><td>100</td><td>-8.12</td><td>78.50</td><td>137.7</td></tr><tr><td>10</td><td>200</td><td>-8.04</td><td>80.34</td><td>118.9</td></tr></table>

## E Pocket Perturbation Details

In our implementation, pocket perturbation is applied directly to pocket coordinates:

$$
\tilde { \mathbf { y } } = \mathbf { y } + \pmb { \eta } , \qquad \pmb { \eta } \sim \mathcal { N } ( \mathbf { 0 } , \sigma _ { p } ^ { 2 } \mathbf { I } ) ,\tag{40}
$$

while pocket atom features are kept unchanged.

The following local statement formalizes the regularization view of pocket perturbation using the standard connection between input noise, vicinal risk minimization, and Jacobian regularization [3, 5].

Proposition 1 (Pocket perturbation regularizes conditional sensitivity). Let $f _ { \boldsymbol { \theta } } ( \cdot , \mathcal { P } )$ denote either the coordinate denoising head or the pre-softmax type head at a fixed ligand noise level $\sigma ,$ and let $\ell ( f _ { \theta } ( \cdot , \mathcal { P } ) , t )$ be a smooth per-sample denoising loss with target t. For Gaussian pocket perturbation $\eta \sim \mathcal { N } ( 0 , \sigma _ { p } ^ { 2 } I )$ , the perturbed-pocket objective satisfies

$$
\mathbb { E } _ { \eta } \ell \big ( f _ { \theta } ( \cdot , \mathcal { P } + \eta ) , t \big ) = \ell \big ( f _ { \theta } ( \cdot , \mathcal { P } ) , t \big ) + \frac { \sigma _ { p } ^ { 2 } } { 2 } \mathrm { T r } \big ( \nabla _ { \mathcal { P } } ^ { 2 } \ell \big ( f _ { \theta } ( \cdot , \mathcal { P } ) , t \big ) \big ) + O ( \sigma _ { p } ^ { 4 } ) .\tag{41}
$$

under standard smoothness assumptions. For a squared coordinate loss, the leading nonnegative term contains $\frac { \sigma _ { p } ^ { 2 } } { 2 } \lVert J _ { P } f _ { \theta } \rVert _ { F } ^ { 2 }$ near the optimum. Thus pocket perturbation penalizes sharp dependence of the denoiser on small pocket-coordinate changes. Equivalently, the model is trained against a local neighborhood ofmildly perturbed pocket views rather than a single rigid conditioning instance. When the perturbation scale is small, this neighborhood can be interpreted as approximating nearby pocket-coordinate perturbations.

Proof. Apply a second-order Taylor expansion to $\ell ( f _ { \theta } ( \cdot , \mathcal { P } + \pmb { \eta } ) , t )$ around ${ \mathcal { P } } .$ . The first-order term vanishes after expectation because $\mathbb { E } [ \pmb { \eta } ] = 0 .$ , and the second-order term contracts the pocket Hessian with $\mathbb { E } [ \pmb { \eta } \pmb { \eta } ^ { \top } ] = \bar { \sigma } _ { p } ^ { 2 } I$ , giving Eq. (41). For squared denoising loss, the Hessian decomposes into a Gauss–Newton term $J _ { \mathcal { P } } f _ { \theta } ^ { \top } J _ { \mathcal { P } } f _ { \theta }$ plus residual-weighted second-derivative terms; near a well-fit denoiser the residual terms are small, leaving the stated Jacobian penalty. □

This local analysis also explains why the perturbation can be made noise-level dependent. At large ligand noise $\sigma ,$ the denoiser should use coarse pocket information and can tolerate stronger pocket smoothing; at small $\sigma ,$ the reverse process refines local geometry and should not blur steric constraints. We therefore use a bounded schedule $\sigma _ { p } = \operatorname* { m i n } ( 0 . 1 \sigma , 0 . 5 )$ in the main setting: the proportional term keeps pocket uncertainty below the ligand corruption scale, while the cap prevents the augmentation from erasing Angstrom-scale pocket geometry. From an information-bottleneck perspective [41], this schedule suppresses nuisance information about the exact crystallographic pocket realization while preserving the pocket information needed to predict ligand geometry and affinity. If $\sigma _ { p }$ is too small, the Jacobian penalty is negligible and the model can overfit to brittle pocket details; if it is too large, the conditioning channel loses target-specific information and the model shifts toward a weaker, less precise conditional distribution. In this sense, pocket perturbation controls the local Lipschitz behavior of the pocket-conditioning channel: it smooths the denoiser’s response to pocket geometry and only indirectly smooths the effective conditional landscape explored by the sampler. The ablation in Table 4 is consistent with this bias–variance trade-off: the mild bounded schedule improves local distance fidelity, atom-type JS, Vina metrics, and centroid distance, the corresponding unbounded 0.1σ schedule remains competitive but is slightly weaker on the same target-specific metrics, the lighter min(0.05σ, 0.25) schedule further reduces strain, and the stronger min(0.3σ, 1.5) schedule favors broader distributional and SA-oriented statistics but gives up some target-specific precision.

Table 4: Sensitivity to protein-side perturbation strategies. Local JSD averages the eight short-range pairwise distance JSDs over 6-6|1, 2, 4, 6-7|1, 2, 4, and 6-8|1, 2, corresponding to C–C, C–N, and $\mathrm { C } -$ O distance statistics. JSD-All-12Å pools all intraligand atom-pair distances up to 12 Å, JSD-CC-2Å focuses on short-range carbon–carbon distances, and Atom reports atom-type frequency JS. Lower JSD, Vina metrics, strain energy, and centroid distance are better; higher QED, SA, Valid<sub>3D</sub>, and clash-free rate are better.
<table><tr><td>Perturbation</td><td>| Local (↓)</td><td>12Å (↓)</td><td>CC-2Å (↓)</td><td>Atom (↓)</td><td>QED (↑)</td><td>SA (↑)</td><td>Vina (↓)</td><td>Vina Min</td><td>Valid3D (↑)</td><td>Strain (↓)</td><td>Clash-Free (↑)</td><td>Centroid (↓)</td></tr><tr><td>None</td><td>0.270</td><td>0.069</td><td>0.276</td><td>0.143</td><td>0.63</td><td>0.73</td><td>-7.78</td><td>-7.99</td><td>77.3</td><td>125.9</td><td>94.07</td><td>1.271</td></tr><tr><td>Fixed 0.1</td><td>0.262</td><td>0.064</td><td>0.274</td><td>0.140</td><td>0.65</td><td>0.72</td><td>-7.70</td><td>-8.15</td><td>80.0</td><td>129.2</td><td>93.15</td><td>1.257</td></tr><tr><td>0.1σ</td><td>0.267</td><td>0.068</td><td>0.269</td><td>0.140</td><td>0.64</td><td>0.74</td><td>-7.83</td><td>-8.07</td><td>81.3</td><td>122.5</td><td>94.27</td><td>1.307</td></tr><tr><td>min(0.05σ, 0.25)</td><td>0.270</td><td>0.070</td><td>0.288</td><td>0.145</td><td>0.64</td><td>0.74</td><td>-7.86</td><td>-8.16</td><td>81.2</td><td>121.0</td><td>94.13</td><td>1.275</td></tr><tr><td>min(0.1σ, 0.5)</td><td>0.255</td><td>0.065</td><td>0.269</td><td>0.120</td><td>0.64</td><td>0.71</td><td>-7.89</td><td>-8.19</td><td>80.6</td><td>127.9</td><td>94.02</td><td>1.255</td></tr><tr><td>min(0.2σ, 1.0)</td><td>0.269</td><td>0.069</td><td>0.281</td><td>0.137</td><td>0.64</td><td>0.74</td><td>-7.77</td><td>-8.08</td><td>80.2</td><td>121.8</td><td>94.28</td><td>1.286</td></tr><tr><td>min(0.3σ, 1.5)</td><td>0.257</td><td>0.063</td><td>0.263</td><td>0.132</td><td>0.64</td><td>0.75</td><td>-7.66</td><td>-7.98</td><td>79.4</td><td>118.5</td><td>93.67</td><td>1.280</td></tr></table>

## F Guidance Scale Sensitivity Details

Table 5 and Figures 6–7 provide the step-wise CFG-scale diagnostics corresponding to the main-text trade-off trajectory in Figure 2. They make the same trend explicit from metric-wise views: weakto-moderate guidance improves affinity-related and molecular-property objectives while preserving strong geometry, whereas overly strong guidance eventually reduces $\mathrm { V a l i d _ { 3 D } }$ and increases strain and centroid error.

Table 5: Sensitivity to CFG scale. We report median property metrics together with key GenBench3D geometry metrics.
<table><tr><td>S</td><td>QED (↑)</td><td>SA (↑)</td><td>Vina Score (↓) |</td><td> $\mathrm { V a l i d } _ { \mathrm { 3 D } }$  (↑)</td><td>Strain (↓)</td><td>Clash-Free (↑)</td><td>Centroid Dist. (↓)</td></tr><tr><td>0</td><td>0.48</td><td>0.63</td><td>-6.70</td><td>78.64</td><td>164.9</td><td>90.59</td><td>1.25</td></tr><tr><td>0.5</td><td>0.57</td><td>0.66</td><td>-7.03</td><td>78.74</td><td>152.3</td><td>92.02</td><td>1.24</td></tr><tr><td>1</td><td>0.60</td><td>0.70</td><td>-7.20</td><td>77.99</td><td>127.4</td><td>90.74</td><td>1.25</td></tr><tr><td>2</td><td>0.62</td><td>0.71</td><td>-7.33</td><td>78.92</td><td>125.6</td><td>92.15</td><td>1.26</td></tr><tr><td>3</td><td>0.63</td><td>0.71</td><td>-7.56</td><td>81.12</td><td>127.7</td><td>93.57</td><td>1.25</td></tr><tr><td>5</td><td>0.64</td><td>0.71</td><td>-7.89</td><td>80.60</td><td>127.9</td><td>94.02</td><td>1.26</td></tr><tr><td>10</td><td>0.64</td><td>0.72</td><td>-8.12</td><td>78.54</td><td>137.7</td><td>95.03</td><td>1.28</td></tr><tr><td>15</td><td>0.64</td><td>0.70</td><td>-8.00</td><td>75.14</td><td>160.4</td><td>94.89</td><td>1.31</td></tr><tr><td>20</td><td>0.60</td><td>0.69</td><td>-7.61</td><td>72.04</td><td>187.5</td><td>94.51</td><td>1.38</td></tr></table>

![](images/5cb4cd18fbafb8f70fa8372e9ebe8c1c10ae428ff6f2311c304e10045c20e285.jpg)  
Figure 6: Property-side trends under different CFG scales. QED, SA, and Vina Score improve from weak to moderate guidance, with Vina Score reaching its strongest value around $s = 1 0$ and QED/SA saturating around the moderate-tostrong range.

![](images/73da7d266eee9afac192e0c82c48fdd4e974f551dfa3020c0e9dbb2949e38816.jpg)  
Figure 7: Geometry-side trends under different CFG scales. Moderate guidance keeps Valid<sub>3D</sub> high and strain energy low, whereas overly strong guidance reduces Valid and increases strain. In this study, $s = 5$ gives the best overall balance.

## G RampUp Guidance Schedule

In addition to the constant CFG scale used in the main experiments, we also consider a simple RampUp variant inspired by recent analyses of guidance schedules in masked diffusion [36]. The motivation is that strong extrapolation at the beginning of sampling can be brittle, since the ligand state is still dominated by high noise and the conditional and dropped-condition predictions may be poorly calibrated. Instead of applying a fixed scale s at every reverse step, RampUp uses a time-dependent scale

$$
s _ { i } ^ { \mathrm { r a m p } } = \alpha _ { i } s , \qquad \alpha _ { i } = { \frac { i } { N - 1 } } , \qquad i = 0 , \ldots , N - 1 ,\tag{42}
$$

where N is the number of sampling steps and i indexes the reverse trajectory from the highest-noise step to the final low-noise step. The guided coordinate and atom-type predictions then use the same CFG rule as Eq. (8), with s replaced by $s _ { i } ^ { \mathrm { r a m p } }$ at step i. Thus, early steps remain close to the propertyunconditional $( s = 0 )$ trajectory, while later low-noise steps receive the full property-steering strength. This schedule does not require retraining or additional model evaluations; it only changes the scalar guidance multiplier used during inference.

We use RampUp as an auxiliary diagnostic rather than as the default sampler. Since the same nominal scale is not directly comparable across constant and RampUp schedules, we compare them at similar median docking strength. In the mid-guidance regime, RampUp gives a comparable trade-off: for example, constant CFG with $s \ : = \ : 2$ gives median Vina Score −7.44 and median Vina Min $- 7 . 7 8 .$ , while RampUp with final scale $s = 5$ gives −7.49 and −7.81. At this matched operating point, RampUp keeps a similar distributional profile (JSD-All 0.0488 versus 0.0490) and slightly higher completion rate (0.960 versus 0.954), with comparable molecular stability (34.7% versus $3 4 . 4 \% )$ . At stronger docking targets, however, this advantage is not consistent; matching the constant $s = 5$ median Vina Score with RampUp requires a larger final scale and lowers molecular stability. We therefore treat RampUp as a conservative schedule that can improve the median trade-off in a moderate guidance range, rather than as a uniformly better replacement for constant CFG.

## H Component-Wise Ablation Summary

To make the contribution of each design choice easier to inspect, we collect the main component-wise comparisons in Table 6. The table reorganizes results by design axis rather than by metric category. Several entries are repeated from Tables 1, 5, and 4 to make the attribution easier to read in one place. All PocketVE variants use the same test split and GenBench3D evaluation protocol, and guided variants use the same trained checkpoint unless the ablated component changes training. The schedule row reports an additional ablation, while the other rows reorganize results from the main tables by design axis.

Table 6: Component-wise ablation summary for PocketVE.
<table><tr><td>Ablation axis</td><td>| Setting</td><td>|QED</td><td>SA</td><td>Vina Score</td><td> $\mathrm { V a l i d _ { 3 D } }$ </td><td>Strain</td><td>Clash-Free</td><td>Centroid</td></tr><tr><td>Backbone/guidance baseline</td><td>TAGMol</td><td>0.56</td><td>0.56</td><td>-7.77</td><td>58.6</td><td>457.4</td><td>93.21</td><td>1.54</td></tr><tr><td>VE backbone without CFG</td><td> $\mathrm { P o c k e t V E } \left( s = 0 \right)$ </td><td>0.48</td><td>0.63</td><td>-6.70</td><td>78.64</td><td>164.9</td><td>90.59</td><td>1.25</td></tr><tr><td>Plain conditional sampling</td><td> $\mathrm { P o c k e t V E } \left( s = 1 \right)$ </td><td>0.60</td><td>0.70</td><td>-7.20</td><td>77.99</td><td>127.4</td><td>90.74</td><td>1.25</td></tr><tr><td>Moderate CFG</td><td> $\mathrm { P o c k e t V E } \left( s = 5 \right)$ </td><td>0.64</td><td>0.71</td><td>-7.89</td><td>80.60</td><td>127.9</td><td>94.02</td><td>1.26</td></tr><tr><td>Protein perturbation</td><td>No perturbation</td><td>0.63</td><td>0.73</td><td>-7.78</td><td>77.3</td><td>125.9</td><td>94.07</td><td>1.271</td></tr><tr><td>Optimizer</td><td>Adam  $( s = 5 )$ </td><td>0.64</td><td>0.73</td><td>-7.84</td><td>78.34</td><td>152.8</td><td>93.74</td><td>1.272</td></tr><tr><td>Optimizer</td><td>AdamW  $( s = 5 )$ </td><td>0.63</td><td>0.73</td><td>-7.40</td><td>76.62</td><td>170.5</td><td>93.58</td><td>1.340</td></tr><tr><td>Noise schedule</td><td>Log-uniform schedule  $( s = 5 )$ </td><td>0.64</td><td>0.70</td><td>-7.13</td><td>73.11</td><td>186.8</td><td>90.77</td><td>1.303</td></tr><tr><td>Sampler control</td><td>No sampling-time noise</td><td>0.59</td><td>0.63</td><td>-5.75</td><td>59.70</td><td>495.7</td><td>75.70</td><td>1.44</td></tr></table>

The comparison between TAGMol and PocketVE $( s = 0 )$ should be read as the bundled effect of the unguided PocketVE backbone and sampler, including the VE-style coordinate parameterization, sampling-time noise injection, and generalized arcsin schedule, without classifier-free guidance. Together with the cumulative ablation in Table 2, this shows that the geometric stabilization mainly comes from these backbone-level sampling and parameterization changes before inference-time guidance is applied. Concretely, $\mathrm { V a l i d _ { 3 D } }$ increases by about 20 points, while strain energy is reduced by about 64% relative to TAGMol. The comparison from $s = 0 \mathrm { ~ t o ~ } s = 5$ then isolates inference-time CFG under the same checkpoint and reverse trajectory: moderate guidance improves Vina Score, QED, and SA while preserving high 3D validity. The no-perturbation row shows that pocket-side perturbation is not a uniformly dominant switch, but it changes the property–geometry frontier and provides a useful reference point in the current setting. The optimizer rows provide controls at the same CFG scale. Adam, which follows the optimizer choice used in TAGMol, remains a strong baseline, while AdamW-only training is weaker in both affinity and geometry. The main Muon–AdamW setting still gives the best overall geometry and strain, suggesting that the final gains are not simply due to replacing Adam with AdamW. The log-uniform-schedule row further shows that the reported gains are not produced by CFG alone: replacing the generalized arcsin schedule with log-uniform sampling at the same guidance scale leads to weaker Vina Score, lower $\mathrm { V a l i d _ { 3 D } }$ higher strain, and worse centroid error.

## I Repeated-Run Variability

To document the stability of the final operating point without forcing the reader to cross-reference the main table, Table 7 reports the final PocketVE metrics together with their repeated-run standard deviations under the same evaluation protocol. We keep this summary compact and focus on the main distribution, property, affinity, and geometry metrics used throughout the paper.

Table 7: Final PocketVE performance with repeated-run variability. Each entry is reported as value ± standard deviation across repeated runs. We report the same representative statistics used in the main text: $\mathrm { Q E D } _ { \mathrm { m e d } } , \mathrm { S A } _ { \mathrm { m e d } }$ , Vina ${ \mathrm { S c o r e } } _ { \mathrm { m e d } } .$ , Vina $\mathbf { M } \mathrm { i } \mathbf { n } _ { \mathrm { m e d } } .$ , Valid , $\mathrm { S t r a i n } _ { \mathrm { m e d } } ,$ Clash-Free, Centroid mean, Local, JSD-All-12Å, $\scriptstyle \mathbf { J S D - C C - 2 } \tilde { \mathbf { A } }$ , and atom-type JS. $\mathrm { V a l i d _ { 3 D } }$ and Clash-Free use the same displayed units as the main table.
<table><tr><td>QED↑</td><td>SA↑</td><td>Vina ↓</td><td>Vina Min ↓</td><td>Valid3D ↑</td><td>Strain ↓</td><td>Clash-Free ↑</td><td>Centroid ↓</td></tr><tr><td>0.63 ± 0.0018</td><td>0.72 ± 0.0047</td><td>-7.734 ± 0.0150</td><td>-8.026 ± 0.0204</td><td>79.27 ± 0.68</td><td>124.7 ± 1.29</td><td>93.89 ± 5.91</td><td> $1 . 2 8 4 \pm 0 . 0 1 2 7$ </td></tr><tr><td>Local ↓</td><td>12Å↓</td><td>CC-2Å↓</td><td>Atom ↓</td><td></td><td></td><td></td><td></td></tr><tr><td>0.245 ± 0.0020</td><td>0.0629 ± 0.0003</td><td>0.265 ± 0.0004</td><td>0.121 ± 0.0003</td><td></td><td></td><td></td><td></td></tr></table>

## J Reference-Normalized Multi-Objective Hits

Table 8 reports an auxiliary reference-normalized diagnostic rather than a replacement for the main benchmark metrics. The goal is to test whether a method produces same-pocket samples that are jointly no worse than the reference ligand in docking affinity, QED, and SA. Among the filtered methods, PocketVE achieves the highest Joint-Ref average and Joint Success rate.

Table 8: Reference-normalized multi-objective hit metrics; all percentage metrics are higher-is-better. Overall QED, SA, and diversity averages are included to preserve the average property statistics omitted from the compact main table. High Affinity follows prior SBDD usage and denotes the perpocket fraction of generated molecules whose Vina Dock score is no worse than the reference ligand, and Success is the percentage of pockets with at least one such molecule. High-Aff. Subset reports average QED and SA within the High Affinity subset. Hit Rate is the fixed-threshold multi-objective rate with $\mathrm { Q E D } \geq 0 . 4 , \mathrm { S A } \geq 0 . 5 .$ , and Vina $\mathrm { D o c k } \le - 8 . 1 8$ kcal/mol. Joint-Ref is the per-pocket fraction of generated molecules that simultaneously satisfy Vina Dock, QED, and SA no worse than the reference ligand. Joint Success is the percentage of pockets with at least one Joint-Ref molecule.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Hit Rate</td><td colspan="3">Overall Avg.</td><td colspan="3">High Affinity</td><td colspan="2">High-Aff. Subset</td><td colspan="3">Joint-Ref</td></tr><tr><td>QED</td><td>SA</td><td>Div.</td><td>Avg.</td><td>Med.</td><td>Success</td><td>QED Avg.</td><td>SA Avg.</td><td> $\operatorname { A v g } .$ </td><td>Med.</td><td>Success</td></tr><tr><td>AR</td><td>12.9</td><td>0.51</td><td>0.63</td><td>0.70</td><td>41.2</td><td>37.0</td><td>72.2</td><td>0.52</td><td>0.59</td><td>4.6</td><td>0.0</td><td>36.1</td></tr><tr><td>Pocket2Mol</td><td>24.3</td><td>0.56</td><td>0.74</td><td>0.69</td><td>48.0</td><td>51.0</td><td>88.0</td><td>0.57</td><td>0.72</td><td>16.5</td><td>8.0</td><td>74.0</td></tr><tr><td>TargetDiff</td><td>20.5</td><td>0.48</td><td>0.58</td><td>0.72</td><td>57.6</td><td>58.3</td><td>99.0</td><td>0.50</td><td>0.56</td><td>5.7</td><td>2.0</td><td>58.0</td></tr><tr><td>DecompDiff</td><td>27.7</td><td>0.45</td><td>0.61</td><td>0.68</td><td>63.8</td><td>75.0</td><td>91.9</td><td>0.44</td><td>0.59</td><td>6.8</td><td>1.1</td><td>50.5</td></tr><tr><td>IPDiff</td><td>27.4</td><td>0.52</td><td>0.61</td><td>0.74</td><td>68.2</td><td>74.6</td><td>98.0</td><td>0.52</td><td>0.57</td><td>10.0</td><td>2.6</td><td>70.0</td></tr><tr><td>TAGMol</td><td>27.7</td><td>0.55</td><td>0.56</td><td>0.69</td><td>68.6</td><td>75.7</td><td>99.0</td><td>0.55</td><td>0.54</td><td>7.1</td><td>1.5</td><td>58.0</td></tr><tr><td>BindDM</td><td>24.9</td><td>0.51</td><td>0.58</td><td></td><td>64.2</td><td>66.1</td><td>100.0</td><td>0.53</td><td>0.55</td><td>6.9</td><td>1.2</td><td>67.0</td></tr><tr><td>ALiDiff</td><td>25.2</td><td>0.50</td><td>0.57</td><td>0.73</td><td>68.5</td><td>76.0</td><td>100.0</td><td>0.52</td><td>0.54</td><td>7.0</td><td>0.0</td><td>48.0</td></tr><tr><td>SeFMol</td><td>37.4</td><td>0.63</td><td>0.60</td><td>0.69</td><td>68.7</td><td>76.3</td><td>100.0</td><td>0.63</td><td>0.57</td><td>11.2</td><td>4.0</td><td>68.0</td></tr><tr><td>PAFlow</td><td>31.5</td><td>0.49</td><td>0.57</td><td>0.71</td><td>80.8</td><td>93.7</td><td>99.0</td><td>0.50</td><td>0.56</td><td>8.9</td><td>1.1</td><td>52.0</td></tr><tr><td>PocketVE</td><td>47.3</td><td>0.61</td><td>0.73</td><td>0.72</td><td>73.9</td><td>85.2</td><td>100.0</td><td>0.62</td><td>0.70</td><td>26.4</td><td>20.9</td><td>92.0</td></tr></table>

## K Additional Property–Validity Trade-Offs

Figure 8 extends the main binding–geometry analysis to the drug-likeness and synthesizability axes. The same qualitative trade-off remains visible: weak-to-moderate guidance improves the property side without immediately leaving the high-validity regime, whereas stronger guidance eventually pushes the model toward lower $\mathrm { V a l i d _ { 3 D } }$ . We place these plots in the appendix because they reinforce the same guidance-scale story as Figure 2, but from auxiliary property views rather than the primary affinity view.

## L Additional Distance-Distribution Diagnostics

Figure 9 makes the JSD diagnostics in the main text visually inspectable rather than purely numeric. Its purpose is to show what the all-atom distance divergence actually looks like in distribution space, and to verify that the quantitative JSD ranking in Figure 3 corresponds to visible shifts in the intraligand distance profile. Methods with larger JSD show broader distortions in the shortand mid-range distance modes, whereas PocketVE remains comparatively close to the reference distribution.

![](images/5e39b3ed2ded90c583970c3082b28ca7d0854ac1292ba31446405aebb94c918c.jpg)  
Figure 8: Property–validity trade-off across baselines and PocketVE guidance scales. The left and right panels report $\mathrm { Q E D - V a l i d _ { 3 D } }$ and $\mathrm { S A { - } V a l i d _ { \mathrm { 3 D } } }$ , respectively. Only baselines with available GenBench3D $\mathrm { V a l i d _ { 3 D } }$ values are shown. PocketVE moves toward higher drug-likeness and synthesizability while retaining high 3D validity under moderate CFG scales.

![](images/fa49e7a14c45fdcb7f8104d3d35c555558cef670ef21f65f3480e1cf9a149333.jpg)  
All atom pair distances (vs. reference)

![](images/0a21c3e732c205f049ba005364eecd93e7c2d63d345a8d4746adb095eba58a2c.jpg)

![](images/c88ac30e65da2094d4ce5f33cdedab5673dd2da60916a9631b91d72364107001.jpg)  
Figure 9: All-atom pair distance distributions compared against the reference ligands in the test set, following the same TargetDiff-style logic used for JSD diagnostics in the main text. Each panel overlays the empirical distribution of reference intraligand atom-pair distances with the corresponding generated distribution, and reports the resulting Jensen–Shannon divergence. The PocketVE panel corresponds to guided sampling at $s = 5$ . Lower JSD indicates closer agreement with the reference distance profile, but this remains a distribution-fidelity diagnostic rather than an explicit physicalvalidity metric.

## M Test-Time Protein Perturbation

We use this experiment as a controlled stress test rather than as a standard evaluation setting. At inference time, Gaussian noise is added only to the protein pocket coordinates, while ligand initialization, sampling schedule, and CFG scale are kept fixed at the main operating point $s = 5$ . This setup probes whether training-time pocket perturbation improves tolerance to small errors in the conditioning geometry.

Across the sweep, both perturbation-trained variants generally preserve lower distance-distribution JSDs than the model trained without pocket perturbation. The stronger 0.3σ training perturbation further improves SA and strain-energy behavior at several noise levels, while the milder min(0.1σ, 0.5) training perturbation retains the strongest Vina-based scores at low test-time noise. However, severe test-time corruption still substantially weakens Vina-based scores and clash-free rates for all three models. These results suggest that protein perturbation mainly acts as a robustness regularizer for the conditioning geometry, rather than making the sampler invariant to arbitrarily corrupted pockets.

Table 9: Test-time protein perturbation stress test under inference-time pocket noise. Gaussian noise with scale $\sigma _ { \mathrm { p o c k e t } }$ is added to pocket coordinates at inference time. We compare PocketVE trained with min(0.1σ, 0.5) perturbation, PocketVE trained without protein perturbation, and PocketVE trained with 0.3σ perturbation using the same four JSD diagnostics as in the main text together with median property metrics and key GenBench3D geometry metrics. $\sigma _ { \mathrm { p o c k e t } }$ is measured in Å. Lower Local, 12Å, CC-2Å, Atom, Vina, Vina Min, Strain, and Centroid are better; higher QED, SA, Valid<sub>3D</sub>, and Clash-Free are better.
<table><tr><td>Method</td><td> $\sigma _ { \mathrm { p o c k e t } }$ </td><td>| Local ↓</td><td>12Å↓</td><td>CC-2Å↓</td><td>Atom ↓</td><td>QED ↑</td><td>SA↑</td><td>Vina ↓</td><td>Vina Min ↓</td><td>Valid3D ↑</td><td>Strain ↓</td><td>Clash-Free ↑</td><td>Centroid ↓</td></tr><tr><td rowspan="5">PocketVE w/o perturb.</td><td>0.0</td><td>0.265</td><td>0.0690</td><td>0.2750</td><td>0.1422</td><td>0.63</td><td>0.73</td><td>-7.83</td><td>-8.12</td><td>76.17</td><td>121.7</td><td>94.61</td><td>1.274</td></tr><tr><td>0.1</td><td>0.264</td><td>0.0672</td><td>0.2739</td><td>0.1419</td><td>0.65</td><td>0.73</td><td>-7.81</td><td>-8.15</td><td>79.48</td><td>127.7</td><td>94.76</td><td>1.301</td></tr><tr><td>0.3</td><td>0.269</td><td>0.0678</td><td>0.2754</td><td>0.1487</td><td>0.63</td><td>0.73</td><td>-7.10</td><td>-7.95</td><td>78.03</td><td>136.2</td><td>90.34</td><td>1.337</td></tr><tr><td>0.5</td><td>0.270</td><td>0.0713</td><td>0.2832</td><td>0.1552</td><td>0.64</td><td>0.74</td><td>-5.90</td><td>-7.58</td><td>79.28</td><td>130.6</td><td>81.14</td><td>1.403</td></tr><tr><td>1.0</td><td>0.275</td><td>0.0693</td><td>0.2802</td><td>0.1655</td><td>0.61</td><td>0.72</td><td>-2.90</td><td>-6.60</td><td>76.69</td><td>159.7</td><td>44.80</td><td>1.795</td></tr><tr><td rowspan="5">PocketVE w/ 0.1σ</td><td>0.0</td><td>0.255</td><td>0.0646</td><td>0.2674</td><td>0.1225</td><td>0.65</td><td>0.71</td><td>-7.87</td><td>-8.18</td><td>80.05</td><td>128.4</td><td>93.79</td><td>1.250</td></tr><tr><td>0.1</td><td>0.263</td><td>0.0645</td><td>0.2689</td><td>0.1235</td><td>0.63</td><td>0.72</td><td>-7.65</td><td>-7.98</td><td>79.72</td><td>122.3</td><td>94.33</td><td>1.283</td></tr><tr><td>0.3</td><td>0.261</td><td>0.0640</td><td>0.2649</td><td>0.1261</td><td>0.64</td><td>0.73</td><td>-6.82</td><td>-7.65</td><td>78.43</td><td>117.6</td><td>91.26</td><td>1.281</td></tr><tr><td>0.5</td><td>0.268</td><td>0.0660</td><td>0.2745</td><td>0.1297</td><td>0.64</td><td>0.72</td><td>-5.68</td><td>-7.38</td><td>80.12</td><td>124.2</td><td>77.90</td><td>1.349</td></tr><tr><td>1.0</td><td>0.263</td><td>0.0629</td><td>0.2660</td><td>0.1293</td><td>0.63</td><td>0.70</td><td>-2.75</td><td>-6.63</td><td>80.56</td><td>149.1</td><td>41.26</td><td>1.651</td></tr><tr><td rowspan="5">PocketVE w/ 0.3σ</td><td>0.0</td><td>0.265</td><td>0.0639</td><td>0.2637</td><td>0.1305</td><td>0.63</td><td>0.75</td><td>-7.64</td><td>-7.98</td><td>78.96</td><td>115.8</td><td>92.93</td><td>1.273</td></tr><tr><td>0.1</td><td>0.261</td><td>0.0652</td><td>0.2613</td><td>0.1340</td><td>0.65</td><td>0.76</td><td>-7.48</td><td>-7.87</td><td>80.68</td><td>104.7</td><td>94.32</td><td>1.294</td></tr><tr><td>0.3</td><td>0.266</td><td>0.0671</td><td>0.2664</td><td>0.1341</td><td>0.65</td><td>0.76</td><td>-6.73</td><td>-7.53</td><td>80.77</td><td>110.4</td><td>89.63</td><td>1.316</td></tr><tr><td>0.5</td><td>0.265</td><td>0.0653</td><td>0.2646</td><td>0.1393</td><td>0.65</td><td>0.77</td><td>-5.78</td><td>-7.36</td><td>81.15</td><td>110.5</td><td>78.95</td><td>1.353</td></tr><tr><td>1.0</td><td>0.266</td><td>0.0672</td><td>0.2658</td><td>0.1441</td><td>0.62</td><td>0.74</td><td>-2.52</td><td>-6.30</td><td>76.45</td><td>134.6</td><td>41.93</td><td>1.589</td></tr></table>

## N Pocket-Diagnostic Details

For pocket shuffling, we apply a fixed cyclic permutation with no self-matches across the 100 test pockets. Each generated ligand is translated from its source pocket center to the target pocket center, then evaluated against the mismatched target. Correct and shuffled conditions contain the same 9,471 ligands. We first aggregate within each target pocket, average pockets equally, and form percentile 95% CIs from 10,000 paired pocket-bootstrap replicates. This is a post-hoc full-pipeline specificity control; it does not regenerate ligands while conditioning on an incorrect pocket and does not isolate a causal network component.

For PoseCheck, we evaluate original poses without redocking, summarize poses within each pocket, and average the 100 pockets equally. Table 10 summarizes clashes, clash-free rates, and total interactions, and Table 11 gives the full contact-category profile. The changes are mixed: PocketVE does not maximize every interaction category, and the reference ligands themselves average 11.79 total interactions, close to PocketVE’s 11.99. Thus, total contact count should not be interpreted as interaction quality.

Table 10: Pocket-equal PoseCheck estimates on original poses with 10,000-replicate pocket-bootstrap 95% CIs. PoseCheck clash-free denotes the percentage of poses with zero PoseCheck clashes and differs from the GenBench3D clash-free metric in Table 1.
<table><tr><td>Method</td><td>Mean clashes ↓</td><td>Clash-free (%) ↑</td><td>Mean interactions</td></tr><tr><td>PocketVE</td><td>8.52 [7.12, 10.10]</td><td>4.54 [2.91, 6.60]</td><td>11.99 [10.98, 13.01]</td></tr><tr><td>TargetDiff</td><td>12.95 [11.19, 14.89]</td><td>1.65 [1.04, 2.33]</td><td>13.14 [12.15, 14.15]</td></tr><tr><td>PAFlow</td><td>10.36 [7.87, 14.14]</td><td>3.19 [1.88, 4.73]</td><td>11.65 [10.78, 12.53]</td></tr><tr><td>TAGMol</td><td>13.88 [11.81, 16.17]</td><td>0.91 [0.31, 1.61]</td><td>12.76 [11.83, 13.72]</td></tr><tr><td>Reference</td><td>7.79 [6.49, 9.14]</td><td>5.00 [1.00, 10.00]</td><td>11.79 [10.63, 12.99]</td></tr></table>

Table 11: Pocket-equal PoseCheck contact-category estimates with 10,000-replicate pocket-bootstrap 95% CIs.
<table><tr><td>Method</td><td>Total interactions</td><td>HBD</td><td>HBA</td><td>Hydrophobic</td><td>VdW</td><td>Mean clashes</td></tr><tr><td>PocketVE</td><td>11.99 [10.98, 13.01]</td><td>0.36 [0.30, 0.42]</td><td>1.41 [1.14, 1.68]</td><td>1.06 [0.88, 1.25]</td><td>9.16 [8.47, 9.89]</td><td>8.52 [7.12, 10.10]</td></tr><tr><td>TargetDiff</td><td>13.14 [12.15, 14.15]</td><td>0.68 [0.60, 0.76]</td><td>1.63 [1.37, 1.90]</td><td>1.15 [0.97, 1.35]</td><td>9.69 [9.02, 10.38]</td><td>12.95 [11.19, 14.89]</td></tr><tr><td>PAFlow</td><td>11.65 [10.78, 12.53]</td><td>0.39 [0.33, 0.45]</td><td>1.23 [1.00, 1.47]</td><td>1.59 [1.36, 1.83]</td><td>8.44 [7.86, 9.05]</td><td>10.36 [7.87, 14.14]</td></tr><tr><td>TAGMol</td><td>12.76 [11.83, 13.72]</td><td>0.59 [0.51, 0.68]</td><td>1.50 [1.26, 1.76]</td><td>1.17 [0.97, 1.38]</td><td>9.50 [8.87, 10.15]</td><td>13.88 [11.81, 16.17]</td></tr><tr><td>Reference</td><td>11.79 [10.63, 12.99]</td><td>0.50 [0.36, 0.65]</td><td>1.96 [1.57, 2.37]</td><td>0.72 [0.49, 0.98]</td><td>8.61 [7.86, 9.38]</td><td>7.79 [6.49, 9.14]</td></tr></table>

Table 12 reports paired differences with the sign convention PocketVE minus comparator. Negative clash differences and positive clash-free differences favor PocketVE; interaction-count differences are descriptive because larger is not uniformly better.

Table 12: Paired pocket-level PoseCheck differences with percentile 95% CIs. Clash-free differences are percentage points.
<table><tr><td>Comparator</td><td>∆ mean clashes</td><td>∆ clash-free</td><td>∆ total interactions</td></tr><tr><td>TargetDiff</td><td>-4.42 [-5.92, -3.28]</td><td>+2.90 [+1.61, +4.64]</td><td>-1.15 [-1.47, -0.85]</td></tr><tr><td>PAFlow</td><td>-1.84 [-5.47, +0.63]</td><td>+1.35 [-0.30, +3.40]</td><td>+0.34 [-0.18, +0.88]</td></tr><tr><td>TAGMol</td><td>-5.36 [-7.26, -3.78]</td><td>+3.63 [+2.05, +5.67]</td><td>-0.77 [-1.25, -0.24]</td></tr><tr><td>Reference</td><td>+0.73 [-0.60, +2.17]</td><td>-0.46 [-4.37, +2.79]</td><td>+0.20 [-0.34, +0.73]</td></tr></table>

## O Property-Steering Diagnostics

We evaluate non-default and conflicting condition vectors with the same fixed PocketVE checkpoint, s = 5, 100 pockets, and 10 samples per pocket. The condition order is (Vina, QED, SA), and larger indices denote more favorable training bins. Table 13 shows that ordered requests move all three aggregate medians in the requested directions. Conflicting requests also expose cross-property competition: for example, (4, 0, 2) improves Vina but lowers QED relative to (2, 4, 2), while (2, 0, 4) and (2, 4, 0) reverse the QED–SA preference at similar Vina levels. The non-monotonic outcomes preclude an interpretation as calibrated independent control.

Table 13: Aggregate property medians under ordered and conflicting requested bins. All rows use the same checkpoint and s = 5.
<table><tr><td>Requested bins (Vina, QED, SA)</td><td>Vina Score median</td><td>QED median</td><td>SA median</td></tr><tr><td>(0, 0, 0)</td><td>-3.294</td><td>0.160</td><td>0.520</td></tr><tr><td>(1, 1, 1)</td><td>-5.177</td><td>0.284</td><td>0.540</td></tr><tr><td>(2, 2,2)</td><td>-7.099</td><td>0.589</td><td>0.660</td></tr><tr><td>(3, 3, 3)</td><td>-7.288</td><td>0.619</td><td>0.690</td></tr><tr><td>(4, 4, 4)</td><td>-7.881</td><td>0.642</td><td>0.710</td></tr><tr><td>(4, 0,2)</td><td>-7.515</td><td>0.604</td><td>0.700</td></tr><tr><td>(2, 4, 2)</td><td>-7.144</td><td>0.670</td><td>0.650</td></tr><tr><td>(2, 0, 4)</td><td>-6.937</td><td>0.392</td><td>0.690</td></tr><tr><td>(2, 4,0)</td><td>-6.865</td><td>0.605</td><td>0.530</td></tr></table>

Table 14 reports distribution-level attainment under favorable joint requests. Rates use successfully evaluated molecules as the denominator. The Vina column uses vina\_score, whereas training boundaries use vina\_dock; consequently, we report the score shift but do not treat it as strict Vina-bin attainment.

Table 14: Property-distribution diagnostics under CFG. QED+SA top requires both favorable property bins; all three top additionally reports the more stringent three-way criterion using the available score-only Vina boundary. Vina values are interpreted directionally because the evaluation and training metric variants differ.
<table><tr><td>Scale</td><td>Evaluated n</td><td>QED top</td><td>SA top</td><td>QED+SA top</td><td>All three top</td><td>Vina median</td></tr><tr><td>0</td><td>935</td><td>14.01%</td><td>7.06%</td><td>1.28%</td><td>0.00%</td><td>-6.704</td></tr><tr><td>1</td><td>961</td><td>22.58%</td><td>12.59%</td><td>3.02%</td><td>0.00%</td><td>-7.196</td></tr><tr><td>5</td><td>953</td><td>27.91%</td><td>19.31%</td><td>3.99%</td><td>0.10%</td><td>-7.892</td></tr><tr><td>10</td><td>925</td><td>30.59%</td><td>21.51%</td><td>5.51%</td><td>0.11%</td><td>-8.124</td></tr></table>

## P CFG Diversity and Reference Coverage

Table 15 aggregates successfully evaluated molecules across all pockets. Higher CFG mildly reduces molecule-level uniqueness and increases fingerprint similarity, but scaffold entropy and top-scaffold concentration do not deteriorate monotonically. Reference-scaffold coverage decreases with guidance. Because these summaries pool molecules across pockets, they characterize global chemical-space behavior and can mask within-pocket convergence or pocket-specific reference recovery.

Table 15: Global diversity and reference-coverage diagnostics under CFG. Reference coverage is the percentage of reference scaffolds recovered in the generated set.
<table><tr><td>Scale</td><td>| Unique molecule</td><td>Unique scaffold</td><td>Scaffold entropy</td><td>Top-1 scaffold</td><td>Top-10 scaffold</td><td>Pairwise Tanimoto</td><td>Ref. scaffold coverage</td></tr><tr><td>0</td><td>99.36%</td><td>83.42%</td><td>0.9481</td><td>6.95%</td><td>15.19%</td><td>0.0919</td><td>13.70%</td></tr><tr><td>1</td><td>98.65%</td><td>86.16%</td><td>0.9616</td><td>5.31%</td><td>12.59%</td><td>0.1072</td><td>12.33%</td></tr><tr><td>5</td><td>97.27%</td><td>86.78%</td><td>0.9636</td><td>5.35%</td><td>12.17%</td><td>0.1176</td><td>9.59%</td></tr><tr><td>10</td><td>96.97%</td><td>84.22%</td><td>0.9563</td><td>6.38%</td><td>13.41%</td><td>0.1184</td><td>8.22%</td></tr></table>

## Q Coordinate-Scale Robustness and Diagnostics

We evaluate $\sigma _ { \mathrm { d a t a } } \in \{ 5 , 7 . 5 , 1 0 , 1 2 . 5 , 1 5 \}$ on the same 100 test pockets with 100 generated ligands per pocket, CFG scale $s = 5 ,$ and the generalized-arcsin scheduler. Accordingly, Table 16 is a robustness sensitivity analysis over the evaluated scale range rather than a sharply optimized ranking.

Table 16: $\sigma _ { \mathrm { d a t a } }$ sweep (100 pockets × 100 ligands). Brackets are pocket-bootstrap 95% CIs for metrics recoverable per pocket; QED, SA, and Vina are evaluator-level point estimates.
<table><tr><td> $\sigma _ { \mathrm { d a t a } }$ </td><td>Eval. success [CI]</td><td>QED</td><td>SA</td><td>Vina Score</td><td> $\mathrm { V a l i d } _ { \mathrm { 3 D } } \ [ \mathrm { C I } ]$ </td><td>Strain median [CI]</td><td>Clash-free [CI]</td></tr><tr><td>5.0</td><td>93.79 [92.11, 95.25]</td><td>0.614</td><td>0.736</td><td>-7.453</td><td>79.01 [77.31, 80.73]</td><td>126.46 [111.74, 137.27]</td><td>94.47 [90.63, 97.62]</td></tr><tr><td>7.5</td><td>93.33 [91.72, 94.78]</td><td>0.606</td><td>0.764</td><td>-7.216</td><td>80.65 [78.91, 82.36]</td><td>102.83 [91.59, 113.09]</td><td>93.72 [89.65, 96.96]</td></tr><tr><td>10.0</td><td>94.55 [93.24, 95.68]</td><td>0.613</td><td>0.725</td><td>-7.370</td><td>79.48 [77.75, 81.12]</td><td>126.16 [113.83, 139.25]</td><td>94.01 [90.05, 97.26]</td></tr><tr><td>12.5</td><td>93.47 [92.03, 94.74]</td><td>0.608</td><td>0.736</td><td>-7.430</td><td>78.66 [77.01, 80.35]</td><td>128.86 [114.42, 143.01]</td><td>94.23 [90.20, 97.37]</td></tr><tr><td>15.0</td><td>91.85 [90.42, 93.21]</td><td>0.597</td><td>0.748</td><td>-7.306</td><td>81.73 [80.13, 83.27]</td><td>125.58 [112.58, 139.49]</td><td>93.70 [89.40, 97.10]</td></tr></table>

No setting dominates all metrics: $\sigma _ { \mathrm { d a t a } } = 1 0$ has the highest evaluation success, 5 has the most favorable mean Vina score, 7.5 has the highest SA and lowest median strain, and 15 has the lowest evaluation success. Distributional metrics are similarly stable and non-monotonic: JSD-All-12Å ranges from 0.0592 to 0.0665, JSD-CC-2Å from 0.2533 to 0.2819, and atom-type JS from 0.1215 to 0.1458. We therefore use $\sigma _ { \mathrm { d a t a } } = 1 0$ as a fixed operating choice within a robust range, not as a sharply optimized constant.

Table 17 complements the scale sweep with targeted backbone and coordinate-scale diagnostics. The sampler-control result is reported with the component-wise ablation in Appendix H; removing sampling-time noise sharply lowers evaluation success and geometric quality, indicating that the reverse-process noise treatment is an important part of the integrated sampler. The VEDA-style row retains the TAGMol architecture but uses $\sigma _ { \mathrm { { d a t a } } } = 1$ and $s = 0 ,$ , so it tests the bundled formulation rather than isolating $\sigma _ { \mathrm { d a t a } } .$ Removing the $\sigma _ { \mathrm { d a t a } }$ factor in Eq. (32) leaves reconstruction and ligand-only geometry metrics similar but destroys pocket-aware spatial compatibility, as reflected by pathological Vina scores and a collapse in clash-free poses. The no-Eq. 35 row is interpreted as a targeted compatibility diagnostic rather than as an isolated causal estimate of all geometric gains.

Table 17: Backbone and coordinate-scale diagnostics (100 pockets × 100 ligands). QED, SA, and Vina are means; strain is the pooled median. The VEDA-style and no-Eq. 35 rows are targeted bundled diagnostics rather than isolated causal comparisons.
<table><tr><td>Setting</td><td>| Eval. success</td><td>QED</td><td>SA</td><td>Vina</td><td> $\mathrm { V a l i d _ { 3 D } }$ </td><td>Strain</td><td>Clash-free</td><td>Centroid (Å)</td></tr><tr><td>PocketVE setup</td><td>95.30</td><td>0.624</td><td>0.722</td><td>-7.589</td><td>78.72</td><td>127.94</td><td>94.23</td><td>1.25</td></tr><tr><td>VEDA-style  $( \sigma _ { \mathrm { d a t a } } = 1 , s = 0 )$ </td><td>89.50</td><td>0.400</td><td>0.563</td><td>-5.454</td><td>43.43</td><td>1007.53</td><td>91.40</td><td>1.19</td></tr><tr><td>Without scale preservation</td><td>97.50</td><td>0.628</td><td>0.785</td><td>+92.237</td><td>82.39</td><td>100.41</td><td>3.39</td><td>3.33</td></tr></table>

## R Additional Case Studies

Figure 10 provides three further case studies. It compares the reference pose with representative baseline generations and the PocketVE sample for specific binding pockets. These examples are included to show that the qualitative trends discussed in the main text are not limited to a single target.

![](images/69164645b4325cc1e12f3eff1c332f33010d0e90af25272b761362cc256b4c02.jpg)  
Figure 10: Additional case studies for PDB ID 5W2G (top), 3W83 (middle), and 1DJY (bottom).