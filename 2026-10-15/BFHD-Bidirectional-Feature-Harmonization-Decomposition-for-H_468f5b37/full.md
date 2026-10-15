# BFHD: Bidirectional Feature Harmonization Decomposition for Heterogeneous Clinical Assessments

Yuanhao Zhuo<sup>1</sup> , Zixi Qin<sup>1</sup> , Ling Qin<sup>2</sup> and Wanqing Li<sup>1∗</sup>

<sup>1</sup>University of Wollongong, Wollongong, NSW 2522, Australia

<sup>2</sup>The People’s Hospital of Guangxi Zhuang Autonomous Region, Nanning, Guangxi 530021, China {yzhuo, wanqing}@uow.edu.au, zq181@uowmail.edu.au, lqin@gxams.org.cn

## Abstract

Clinical assessments are often collected using heterogeneous assessment systems across centers and time, leading to records that mix different but related sets of measurements. This motivates harmonization beyond total-score linking. We formulate clinical harmonization at the measurement level as a bidirectional recoverability problem: given paired observations from two assessment systems, the goal is to identify which measurements can be reliably translated in both directions within an application-defined tolerance, while separating non-translatable components. We propose Bidirectional Feature Harmonization Decomposition (BFHD), a feasibility-driven framework that enforces bidirectionally coupled translation and uses feature-wise output gating to produce an explicit decomposition in the original measurement space. Experiments on synthetic data and real clinical assessment pairs show that BFHD achieves broader feasible harmonization coverage and improved subset stability compared to baselines.

## 1 Introduction

Clinical assessment of chronic and neuropsychiatric conditions relies on a wide range of rating scales, cognitive tests, and questionnaire-based assessments[American Psychiatric Association, 2022; Lui et al., 2025]. In real clinical workflows, however, different centers, cohorts, and time periods often employ different combinations or versions of these assessment systems[Villalpando et al., 2025]. Even when two assessment systems aim to measure similar clinical domains such as memory or executive function, their measurements (features) typically include both overlapping elements and system-specific components[Willcutt et al., 2005]. As patients move between centers or as protocols evolve, longitudinal records naturally become mixtures of heterogeneous assessment systems rather than observations from a single fixed system[Lawrence et al., 2017].

This heterogeneity creates a central challenge for clinical data integration and longitudinal analysis[Fortin et al., 2017]. Beyond aligning summary scores, practical harmonization requires understanding how two assessment systems relate at the measurement level: which measurements can be reliably translated across systems, which encode systemspecific information, and whether such relationships are consistent in both directions[Dorans, 2007; Kolen and Brennan, 2013]. These questions are critical for cross-protocol integration, retrospective cohort pooling, and interpretation of heterogeneous clinical records[Riley et al., 2010].

Existing harmonization practices primarily address this problem through total-score linking and calibration methods, which estimate transformations between total scores or low-dimensional summaries[Kolen and Brennan, 2013]. While effective for aligning marginal outcomes, these approaches do not characterize bidirectional recoverability at the measurement level between heterogeneous assessment systems[Lord, 2012]. In practice, clinicians and researchers are often left without guidance on which individual measurements can be meaningfully harmonized and which should be treated as non-translatable[Hammarstrom¨ et al., 2016]. In contrast to feature selection approaches that prioritize features based on their utility for downstream prediction, clinical harmonization requires identifying which measurements can be meaningfully translated across assessments while explicitly preserving system-specific information[Molnar, 2020].

In this work, we formalize clinical harmonization at the measurement level as a bidirectional recoverability problem. Given paired observations from two assessment systems, we seek to decompose each system into (i) a harmonizable component consisting of measurements that admit reliable bidirectional translation under a bidirectionally coupled model that enforces synchronous updates across directions, and (ii) a complementary non-translatable component whose information cannot be recovered from the other system. Recoverability is defined as a model-relative, feasibility-based notion, extending classical total-score linking from summary transformations to multivariate, feature-level harmonization.

We introduce Bidirectional Feature Harmonization Decomposition (BFHD), a framework that formulates clinical harmonization as a decomposition of measurements into bidirectionally recoverable and non-translatable components, identified directly in the original measurement spaces. BFHD operationalizes this formulation through a bidirectionally coupled translation model that enforces consistent forward and backward relationships between assessment systems, together with feature-wise gating that determines which measurements must be reconstructed to satisfy cross-system feasibility within a clinically meaningful tolerance[Louizos et al., 2017]. This yields an explicit and interpretable decomposition without task supervision or latent projections.

The contributions of this work are as follows:

• We formulate clinical harmonization as a bidirectional recoverability problem at the measurement level, and provide a principled, model-relative definition of harmonizable and non-translatable measurements.

• We develop a feasibility-driven, single-operator framework with feature-wise gating that identifies large bidirectionally recoverable subsets directly in the original measurement spaces.

• We demonstrate on synthetic and real clinical datasets that BFHD reveals meaningful harmonization structure and supports utilities including cross-protocol integration, temporal deployment under protocol drift, and robustness to test-time missingness.

## 2 Related Work

## 2.1 Clinical Harmonization and Task-Driven Baselines

Classical psychometric linking and equating methods, including total-score linking [Dorans, 2007; Kolen and Brennan, 2013], equipercentile equating [Kolen and Brennan, 2013], and IRT calibration [Cella et al., 2007; Lord, 2012], map between assessment systems primarily at the level of total scores or low-dimensional traits. They support comparability across protocols but do not identify which individual measurements are bidirectionally recoverable between heterogeneous assessments.

In clinical ML, harmonization is often approached via predictive mappings [Fortin et al., 2017; Dewey et al., 2019; Dinsdale et al., 2021] using tree-based models such as feature-wise random forests (FRF) [Breiman, 2001; Breiman et al., 2017] or gradient boosted trees (XGB) [Chen, 2016]. These models can yield strong unidirectional prediction but their feature attributions depend on a chosen target and do not characterize intrinsic bidirectional recoverability. ComBatstyle statistical harmonization [Johnson et al., 2007; Beer et al., 2020] removes site/batch effects as nuisance variation, which is misaligned with our goal of explicitly retaining nontranslatable system-specific content.

## 2.2 Multiview Learning and Bidirectional Structure

Multiview learning relates multiple representations of the same individuals. CCA and its variants [Hotelling, 1935; Bach and Jordan, 2005; Yang et al., 2019; Andrew et al., 2013] and shared–private latent models such as multiview autoencoders and variational approaches [Kingma and Welling, 2013] capture cross-view dependence via latent projections or likelihood objectives. However, these methods typically operate in latent spaces and do not yield an explicit account of which observed measurements are recoverable across systems in the original measurement space.

Bidirectional mappings have also been explored in translation settings. Cycle-consistent models couple two directions through closed-loop reconstruction losses [Zhu et al., 2017], while tied-weight autoencoder-style models enforce parameter sharing [Hinton and Salakhutdinov, 2006; Bengio et al., 2013]. We implement a cycle-consistent MLP (CMLP) as a bidirectional mapping baseline without explicit subset selection. To realize strict single-operator coupling in our framework, which enforces shared constraints, we leverage invertible networks from the normalizing flow [Dinh et al., 2014; Dinh et al., 2016; Kingma and Dhariwal, 2018]. A tiedweight MLP (TWMLP) serves as a non-invertible alternative instantiation in our ablations.

## 2.3 Gating and Its Relation to Feature Selection

Feature selection methods typically identify features that improve task performance, commonly via sparsity-inducing penalties such as $L _ { 1 }$ regularization [Tibshirani, 1996; Ng, 2004] or via neural gating mechanisms [Louizos et al., 2017; Yamada et al., 2020]. In contrast, our gates are optimized as continuous proxies for membership in a bidirectionally recoverable subset under a shared operator, using feasibility-style reconstruction constraints rather than task supervision. This enables an explicit decomposition into recoverable and nontranslatable components aligned with clinical harmonization.

## 3 Problem Formulation

We consider paired observations from two clinical assessment systems, represented by random vectors $X \in \mathbb { R } ^ { d _ { X } }$ and $Y \in { \dot { \mathbb { R } } } ^ { d _ { Y } }$ . Let $\bar { I } _ { X } = \{ 1 , \ldots , d _ { X } \}$ and $I _ { Y } = \{ 1 , \ldots , d _ { Y } \}$ denote feature indices. We seek recoverable subsets of measurements $S _ { X } \subseteq I _ { X }$ and $S _ { Y } \subseteq I _ { Y }$ , whose complements are treated as non-translatable.

## 3.1 Bidirectional model class

Recoverability is defined relative to a bidirectional model class that reflects the mutual nature of harmonization. Given paired observations from two assessment systems, we consider two mappings parameterized by Φ:

$$
f _ { \Phi } : \mathcal { X }  \mathcal { V } , \qquad g _ { \Phi } : \mathcal { V }  \mathcal { X } ,
$$

A key requirement is that the two directions are not learned independently. Instead, updates to the mapping from X to Y must simultaneously constrain, and be constrained by, the mapping from Y to X. That is, learning the relationship between two assessment systems in one direction necessarily informs the relationship in the reverse direction.

We formalize this requirement by restricting $f _ { \Phi }$ and $g _ { \Phi }$ to be bidirectionally coupled: they are derived from a shared parameterization or structural constraint, such that optimizing one direction induces corresponding updates in the other, rather than allowing two unrelated functions linked only by loss terms.

## 3.2 ε-recoverability

Conceptual definition Noise and heterogeneity in clinical assessments make exact recoverability unrealistic. We therefore define a population-level, approximate notion of recoverability. A measurement $X _ { i }$ is said to be ε-recoverable from $Y$ under a shared bidirectional model parameter Φ if

$$
\mathbb { E } \big [ d ( X _ { i } , g _ { \Phi } ( Y ) _ { i } ) \big ] < \varepsilon ,\tag{1}
$$

where $d ( \cdot , \cdot )$ is a feature-wise discrepancy measure and $\varepsilon > 0$ specifies an application-level tolerance for clinically acceptable agreement rather than a tuning parameter for predictive accuracy. Analogously, a measurement $Y _ { j }$ is ε-recoverable if

$$
\mathbb { E } \big [ d ( Y _ { j } , f _ { \Phi } ( X ) _ { j } ) \big ] < \varepsilon .\tag{2}
$$

Recoverability is defined relative to a chosen bidirectional model class $\mathcal { M } ,$ in the same way that classical total-score linking defines equivalence relative to a family of allowable linking functions. Smaller values of ε correspond to stricter harmonization requirements, while larger values admit broader but looser forms of cross-assessment translation. This definition is population-level and combinatorial: it assumes access to the true expectations and to the exact subsets of features whose recoverability constraints are satisfied.

Practical relaxation The population-level conditions in (1)–(2) cannot be evaluated directly: expectations are unobserved, and searching over all subsets $( S _ { X } , S _ { Y } )$ is combinatorial. We therefore introduce a lightweight relaxation that preserves the semantics of recoverability while making the problem estimable from data.

Each feature is assigned a gate variable $G _ { X , i } , G _ { Y , j } \in [ 0 , 1 ]$ that acts as a continuous proxy for membership in the recoverable subsets. Only gated features contribute to empirical reconstruction losses, so the optimizer keeps a gate open when the shared bidirectional operator can reconstruct that feature with sufficiently small empirical error. This operationalizes the theoretical ε-recoverability condition using standardized feature-wise reconstruction error, which approximate the magnitude of the deviations appearing in (1)–(2).

Features whose reconstruction errors remain large force their gates toward zero and are treated as non-translatable. This provides a compact and data-driven approximation to the conceptual BFHD formulation while maintaining its intended semantics; the neural instantiation is described in Section 4.

## 3.3 Feasible decompositions

A triple $\left( S _ { X } , S _ { Y } , \Phi \right)$ is called ε-feasible if each selected feature satisfies its recoverability condition:

$$
\begin{array} { r } { ( S _ { X } , S _ { Y } , \Phi ) \in \mathcal { F } _ { \varepsilon } \Longleftrightarrow \left\{ \begin{array} { l l } { \mathbb { E } [ d ( X _ { i } , g _ { \Phi } ( Y ) _ { i } ) ] < \varepsilon , } & { \forall i \in S _ { X } , } \\ { \mathbb { E } [ d ( Y _ { j } , f _ { \Phi } ( X ) _ { j } ) ] < \varepsilon , } & { \forall j \in S _ { Y } . } \end{array} \right. } \end{array}\tag{3}
$$

This definition does not require exact reconstruction and admits feasible solutions under realistic clinical noise levels.

## 3.4 The BFHD optimization problem

The BFHD problem seeks the largest bidirectionally recoverable subsets:

$$
\operatorname* { m a x } _ { S _ { X } \subseteq I _ { X } , \ S _ { Y } \subseteq I _ { Y } , \ \Phi \in \mathcal { M } } \quad w | S _ { X } | + ( 1 - w ) | S _ { Y } |\tag{4}
$$

$$
\mathrm { s u b j e c t t o ~ } \left( S _ { X } , S _ { Y } , \Phi \right) \in \mathcal { F } _ { \varepsilon } .\tag{5}
$$

Here w is a fixed user-specified weight rather than an optimization variable. Unless otherwise stated, we use $w = 0 . 5$

An optimal solution $( S _ { X } ^ { * } , S _ { Y } ^ { * } , \Phi ^ { * } )$ yields the decomposition

$$
X _ { \mathrm { r e c } } = X _ { S _ { X } ^ { \ast } } , \qquad X _ { \mathrm { n t } } = X _ { I _ { X } \backslash S _ { X } ^ { \ast } } ,\tag{6}
$$

$$
Y _ { \mathrm { r e c } } = Y _ { S _ { Y } ^ { * } } , \qquad \quad Y _ { \mathrm { n t } } = Y _ { I _ { Y } \backslash S _ { Y } ^ { * } } .\tag{7}
$$

When multiple maximal feasible solutions exist, BFHD recovers one such solution. In practice, stability across random initializations is used to assess identifiability.

## 4 Methodology

This section presents a practical realization of the BFHD framework. The goal is to construct a shared bidirectional operator for feature-level harmonization that jointly defines the mappings $X  Y$ and $Y  X$ , together with featurewise gating mechanisms that approximate the subsets appearing in the conceptual formulation in Section 3. The proposed method consists of three components: a shared bidirectional backbone with strict parameter sharing across directions (instantiated with an invertible architecture in our main implementation), feature-wise gates defining relaxed recoverable subsets, and cross-view reconstruction losses that implement the recoverability constraints in a differentiable manner.

## 4.1 Architecture Overview

Given paired observations (X, Y), BFHD constructs two cross-view mappings using a shared bidirectional backbone, which is instantiated as an invertible neural network (INN) in our main implementation. The $X \  \ Y$ and $Y ~  ~ { \dot { X } }$ mappings correspond to the forward and inverse passes of the same backbone. For $X  Y$ and $Y  X$ , we compute

$$
\tilde { Y } = f _ { \Phi } ( X ) , \qquad \tilde { X } = g _ { \Phi } ( Y ) ,\tag{8}
$$

and apply feature-wise $L _ { 0 }$ gates at the outputs:

$$
\hat { Y } _ { j } = G _ { Y , j } \tilde { Y } _ { j } , \qquad \hat { X } _ { i } = G _ { X , i } \tilde { X } _ { i } .\tag{9}
$$

The invertible backbone enforces strict bidirectional parameter sharing, while the feature-wise gates determine which measurements in each system are required to be reconstructed. Only gated outputs contribute to the loss, providing a differentiable approximation to the recoverability constraints in the BFHD formulation.

## 4.2 Bidirectional Backbone via Invertible Networks

A key requirement in the BFHD framework is that the mappings $X ~  ~ Y$ and $Y ~  ~ X$ share a common parameterization. To satisfy this requirement, we instantiate $\left( f _ { \Phi } , g _ { \Phi } \right)$

using an invertible neural architecture. Unlike conventional flow-based models, the invertible structure is not used for likelihood estimation or sampling; instead, it provides a single coupled operator that guarantees strict bidirectional parameter sharing.

Instantiation with a NICE-style architecture We adopt a NICE-style architecture because its additive coupling provides stable training for tabular, low-dimensional clinical data and avoids the scaling instability of affine coupling, which is unnecessary in our non-likelihood setting[Dinh et al., 2014].

Because the NICE architecture is exactly invertible by construction, the two cross-view mappings used in BFHD correspond to a single operator $F _ { \Phi }$ and its analytic inverse $F _ { \Phi } ^ { - 1 }$ This guarantees strict bidirectional parameter sharing and thus directly satisfies the requirement specified in the BFHD formulation. Importantly, the invertible structure serves not as a generative modeling assumption, but as a mechanism that enforces a tightly coupled forward–backward relationship between the two clinical systems.

Dimension alignment When $d _ { X } \neq d _ { Y }$ , both features are embedded into a common dimension $D = \operatorname* { m a x } ( d _ { X } , d _ { Y } )$ by zero-padding the lower-dimensional system. Padding dimensions are never supervised in the reconstruction losses and do not interact with the gating mechanism. This ensures that the decomposition concerns only the original features.

Role of Invertibility in BFHD Invertibility is used solely to enforce strict parameter sharing without assuming the invertibility in clinical assessment systems. The $\mathrm { X \to Y }$ and Y→X mappings are the forward and inverse passes of a single operator. This eliminates forward–backward drift common in cycle-consistent dual networks and improves subset stability. After output gating, the overall mapping is intentionally noninvertible. We use a NICE-style INN as a convenient instantiation; a non-invertible tied-weight backbone is evaluated in ablations.

## 4.3 Output $L _ { 0 }$ Gating

The BFHD problem requires determining, for each feature of X and Y, whether it belongs to the bidirectionally recoverable subset. To obtain a differentiable relaxation of these binary decisions, we introduce feature-wise gates

$$
G _ { X } = ( G _ { X , 1 } , \ldots , G _ { X , d _ { X } } ) , \qquad G _ { Y } = ( G _ { Y , 1 } , \ldots , G _ { Y , d _ { Y } } ) ,
$$

with values in [0, 1]. A gate value near 1 indicates that the corresponding feature is treated as recoverable, and a value near 0 indicates that the feature is assigned to the non-translatable component.

Unlike conventional feature selection methods that gate inputs, BFHD applies gates at the outputs of the cross-view mappings. Thus the predicted vectors are masked according to the definitions in Eqs. (9), ensuring that the model always receives the full input X or Y while only the gated subset is required to be reconstructed. This matches the recoverability semantics of the BFHD formulation.

The gates are implemented using the hard-concrete relaxation of the $L _ { 0 }$ penalty [Louizos et al., 2017]. This parameterization yields stochastic binary gate variables during training while remaining differentiable in expectation, allowing gradient-based optimization of feature inclusion.

## 4.4 Gated Hinge Losses

The BFHD formulation defines ε-recoverability through measurement-level reconstruction error tolerances (Section 3). To approximate this conceptual feasibility constraint from data, we introduce a gated hinge loss with a stricter training tolerance $\varepsilon _ { \mathrm { t r a i n } } ,$ which serves as a conservative proxy for the target evaluation tolerance.

We first define a scale-normalized reconstruction error $\operatorname { e r r } ( \cdot , \cdot )$ for individual features. In our implementation, we use feature-wise normalized mean squared error (NMSE). Normalization ensures that $\varepsilon _ { \mathrm { t r a i n } }$ has a consistent interpretation across heterogeneous features. Throughout this work, all assessment features are treated as numerical variables, and the same normalized squared-error-based discrepancy is used for all features. Ordinal items, when present, are modeled as numerical scores under this approximation.

Using err, we define the gated hinge losses for the two cross-view directions:

$$
{ \mathcal L } _ { X  Y } = \sum _ { i = 1 } ^ { d _ { X } } G _ { X , i } \mathrm { R e L U } \big ( \mathrm { e r r } ( X _ { i } , \tilde { X } _ { i } ) - \varepsilon _ { \mathrm { t r a i n } } \big ) ,\tag{10}
$$

$$
\mathcal { L } _ { Y  X } = \sum _ { j = 1 } ^ { d _ { Y } } G _ { Y , j } \operatorname { R e L U } \big ( \operatorname { e r r } ( Y _ { j } , \tilde { Y } _ { j } ) - \varepsilon _ { \operatorname { t r a i n } } \big ) .\tag{11}
$$

This loss has a direct feasibility interpretation. A gated feature incurs no penalty once its reconstruction error falls below $\varepsilon _ { \mathrm { t r a i n } } .$ , and therefore does not encourage over-optimization beyond the feasibility threshold. In practice, $\varepsilon _ { \mathrm { t r a i n } }$ is set substantially smaller than the evaluation tolerance $\varepsilon _ { \mathrm { c u t o f f } }$ to discourage borderline features from remaining gated open due to noise or finite-sample effects, thereby improving subset stability across random initializations. Penalties are applied only when the underlying ε-recoverability constraint cannot be satisfied under the shared bidirectional operator, creating pressure to either improve reconstruction or reduce the corresponding gate value.

## 4.5 Gate Reward Term

Using the same hard-concrete parameterization as above, we optimize gates to favor inclusion rather than sparsity. Unlike standard $L _ { 0 }$ regularization, which penalizes gate activation to promote sparsity, BFHD requires the opposite behavior. Gates should remain open unless cross-view reconstruction forces them to close. Accordingly, we use a reward rather than a penalty:

$$
{ \mathcal { L } } _ { \mathrm { g a t e } } = - \lambda _ { L _ { 0 } } \left( \sum _ { i = 1 } ^ { d _ { X } } G _ { X , i } + \sum _ { j = 1 } ^ { d _ { Y } } G _ { Y , j } \right) , \qquad \lambda _ { L _ { 0 } } > 0 .
$$

This term broadens the candidate recoverable subsets, while the gated cross-view losses apply opposing pressure by closing gates that lead to poor bidirectional reconstruction. Their interaction forms a continuous relaxation of the BFHD objective and identifies the largest empirically recoverable subsets.

## 4.6 Overall Objective

The complete training objective combines the gated crossview reconstruction terms with the gate reward term:

$$
{ \mathcal { L } } _ { \mathrm { t o t a l } } = { \mathcal { L } } _ { X  Y } + { \mathcal { L } } _ { Y  X } + { \mathcal { L } } _ { \mathrm { g a t e } } .
$$

Minimizing $\mathcal { L } _ { \mathrm { t o t a l } }$ jointly optimizes the shared bidirectional operator and the gating variables, producing an empirical approximation to the BFHD decomposition.

Training is performed using a staged procedure to stabilize optimization and avoid degenerate solutions; details in $\mathsf { A p - }$ pendix A.2.

## 5 Experiments

## 5.1 Experimental Setup

Datasets: We use (1) a synthetic dataset generated using a known shared/private/noise-factor model (details and exact parameters in Appendix B.1). This provides groundtruth recoverable and non-translatable measurement labels for evaluating structure recovery. (2) Alzheimer’s Disease Neuroimaging Initiative (ADNI) cognitive assessments[Mueller et al., 2005], and (3) a paired real-world ADHD assessment dataset consisting of two widely used symptom scales, collected under a controlled co-administration protocol and available upon reasonable request. All real-world datasets consist of paired, co-administered assessment systems with partially overlapping but non-identical measurement sets. To avoid subject-level leakage, each subject contributes at most one observation in all experiments unless otherwise stated.

Baselines: We compare our approach with:

1. Clinically-used harmonization and predictive baselines (IRT, FRF, XGB), which learn unidirectional or taskdriven mappings;

2. Bidirectional learning models (CMLP, NICE) that enforce forward-backward consistency without explicit feasibility-based gating;

3. Ablated variants of our approach.

Implementation Details. Implementation details, model architectures, and training hyperparameters are reported in Appendix A. All experiments use 10 random seeds to assess stability $( s _ { t r a i n } \in \{ \bar { 1 } , . . . , 1 0 \} )$ , and synthetic datasets are generated using fixed data-generation seeds $( s _ { g e n } \in \{ 1 , . . . , 5 \} )$ We distinguish between two tolerances: a training tolerance $\varepsilon _ { \mathrm { t r a i n } } .$ , used only for optimizing BFHD, and an evaluation tolerance $\varepsilon _ { \mathrm { c u t o f f } }$ , used uniformly across all methods for reporting feasible measurements.

The feasibility tolerance $\varepsilon _ { \mathrm { c u t o f f } }$ is system- and contextdependent rather than universal. Prior psychometric analyses report substantial intrinsic measurement uncertainty for NeuroBAT subtests in ADNI [Crane et $a l .$ , 2023] and SNAP-IV items in ADHD [Huang et al., 2023]. When measurementlevel variance estimates are available, these uncertainties can be mapped to the normalized error scale used in our experiments; otherwise, they anchor the order of magnitude of acceptable reconstruction error. We therefore adopt $\varepsilon _ { \mathrm { c u t o f f } } = 0 . 6$ as a conservative, measurement-anchored reference operating point across all experiments, with full protocol details and sensitivity analyses provided in Appendices A.7 and D.2.

## 5.2 Experiment 1: Synthetic Feasibility Recovery under Ground-Truth ε

We first evaluate BFHD in a controlled synthetic setting where ground-truth recoverability labels are available by construction. This experiment focuses on feasibility structure recovery: whether a method can correctly identify which measurements satisfy the ε-recoverability condition, rather than comparing reconstruction accuracy.

Evaluation protocol. Recoverability is evaluated as a structure recovery problem, with ground-truth defined at the feature-index (measurement) level. For BFHD, recoverability is determined directly by the learned output gates after binarization. For baseline methods that do not perform gating, recoverability is inferred post hoc using a relaxed feasibility criterion: a feature is classified as recoverable if its reconstruction error falls below $2 \varepsilon _ { \mathrm { g e n } } .$ , where $\varepsilon _ { \mathrm { g e n } }$ is the noise tolerance used in synthetic data generation. This conservative relaxation compensates for finite-sample estimation error and favors baseline models that do not explicitly optimize feasibility.

We restrict baselines in this experiment to models that naturally admit bidirectional reconstruction and feature-level feasibility evaluation. Other architectures evaluated in realworld experiments are omitted here, as the synthetic data are designed to probe bidirectional feasibility recovery rather than predictive accuracy under task-specific assumptions.

We report structure recovery accuracy $( A c c ) .$ , defined as the proportion of feature indices whose recoverability status is correctly identified, separately for X and Y . To assess robustness, we additionally report the stability of the recovered measurement subsets across random seeds using Jaccard $( J )$ similarity. Other diagnostic metrics are reported in $\mathsf { A p - }$ pendix D.1.

<table><tr><td>Model</td><td> $\operatorname { A c c } _ { X }$ </td><td> $\operatorname { A c c } _ { Y }$ </td><td> $\overline { { J _ { X } } }$ </td><td> $\overline { { J _ { Y } } }$ </td></tr><tr><td>Ours</td><td> $\overline { { 0 . 8 8 0 { \scriptstyle \pm 0 . 0 2 4 } } }$ </td><td> $\overline { { 0 . 8 3 0 { \scriptstyle \pm 0 . 0 2 4 } } }$ </td><td>0.933</td><td>0.924</td></tr><tr><td>CMLP</td><td> $0 . 8 5 5 { \scriptstyle \pm 0 . 0 1 5 }$ </td><td> $0 . 8 1 0 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td>0.975</td><td>0.789</td></tr><tr><td>NICE</td><td> $0 . 7 3 5 { \scriptstyle \pm 0 . 0 3 9 }$ </td><td> $0 . 7 2 5 { \scriptstyle \pm 0 . 0 4 0 }$ </td><td>0.837</td><td>0.814</td></tr></table>

Table 1: Synthetic feasibility recovery under ground-truth ε $( s _ { g e n } { = } 5 )$ $\operatorname { A c c } _ { X }$ and Acc<sub>Y</sub> denote index-level structure recovery accuracy for systems X and $Y$ , respectively (mean $\pm s t d$ across 10 seeds). $J _ { X }$ and $J _ { Y }$ report Jaccard similarity. For baseline models without gating, recoverability is inferred post hoc using a relaxed $2 \varepsilon _ { \mathrm { g e n } }$ feasibility criterion. Additional results are provided in $\mathsf { A p - }$ pendix D.1.

Results. Table 1 summarizes the results. Our approach achieves the best overall structure recovery accuracy with strong subset stability across seeds among bidirectional baselines. In particular, methods without explicit feasibilitydriven gating exhibit unstable behavior on weakly recoverable measurements near the $2 \varepsilon _ { \mathrm { g e n } }$ boundary, leading to increased variability across random seeds.

## 5.3 Experiment 2: Real-World Harmonization

We next evaluate BFHD on real clinical assessment systems to assess its practical utility for cross-protocol harmonization.

In contrast to Experiment 1, where ground-truth recoverability is known, real-world datasets do not admit a definitive notion of measurement-level equivalence. Accordingly, the goal of this experiment is not to maximize reconstruction accuracy, but to characterize how much of each assessment system can be harmonized under a shared bidirectional model while maintaining feasibility and stability.

Evaluation protocol. For each paired assessment system and each mapping direction, we evaluate harmonization outcomes using one primary and three supporting metrics. The primary metric is the number of recoverable measurements (#Rec). For BFHD, recoverable measurements are reported using the learned output gates. For baseline methods, which do not produce an explicit measurement-level recoverability indicator, recoverability is inferred at the feature level using the feasibility criterion $e r r < \varepsilon _ { \mathrm { c u t o f f } }$ The Jaccard similarity (J) of the recovered measurement subsets across random seeds, which quantify harmonization coverage and stability, respectively. Diagnostic metrics summarize the strength of agreement among the selected measurements using concordance correlation coefficient (CCC) thresholds[Lawrence and Lin, 1989; Barnhart et $a l .$ , 2007]. Specifically, we report the counts of selected measurements with strong agreement $( C C C > 0 . 7 )$ and marginal/moderate agreement $\bar { ( C C C ) } >$ 0.5). These CCC-based counts are not used to define feasibility; rather, they help interpret whether the additional measurements included under ϵ-feasibility correspond to clinically meaningful agreement levels.

Unless otherwise stated, all real-world results are reported at a fixed feasibility threshold $\varepsilon _ { \mathrm { c u t o f f } } = 0 . 6$ . We further analyze sensitivity to this choice through an ε-sweep in Figure 1 (with additional settings reported in Appendix A.7).

Results. Table 2 summarizes the harmonization results across all dataset pairs. Across all settings, our approach consistently identifies larger recoverable subsets than baseline methods while maintaining comparable or improved stability across random seeds. In particular, our approach recovers substantially more measurements than traditional linking and predictive baselines, reflecting its objective of maximizing feasible harmonization coverage rather than optimizing pointwise reconstruction accuracy.

Importantly, the additional measurements identified by our approach are not dominated by poorly reconstructed measurements. Across datasets, our approach matches baselines on the number of strongly aligned measurements while consistently increasing the number and stability of marginally aligned measurements. This pattern supports our goal of expanding feasible harmonization coverage without sacrificing the clinically reliable core.

Effect of the feasibility tolerance. To analyze how harmonization coverage varies with the feasibility criterion, we sweep the tolerance ε and report the number of recoverable measurements (#rec). Figure 1 shows the resulting curves for the EMBIC↔NeuroBAT pair, plotted separately for each assessment system. Across a broad range of thresholds, our approach consistently recovers more measurements than all baseline methods, demonstrating that it can achieve wider harmonization coverage while respecting the same feasibility constraint. This advantage is especially pronounced in the intermediate regime (approximately $\varepsilon _ { \mathrm { c u t o f f } } \in [ 0 . 5 5 , 0 . 8 ] \rangle$ ), which corresponds to realistic clinical settings where assessments share a partially overlapping core but differ in system-specific content. In contrast, baseline methods tend to be either overly conservative at lower thresholds or exhibit abrupt and unstable increases in #rec as $\varepsilon _ { \mathrm { c u t o f f } }$ is relaxed. These results highlight BFHD’s ability to trace a smooth and interpretable coverage–feasibility frontier. Results for CAS↔SNAP and additional dataset pairs are deferred to Appendix D.2 as they exhibit similar but less pronounced trends.

![](images/8c9c587e41345aa242693bc13d25f5bb77237d44ed610ee05ad8cd8e286ee6b8.jpg)

![](images/70cc35bb8ef7d3ce366ee5df0c38e6cd83d276825faa39c50521d8b5abe01f55.jpg)  
Figure 1: ε-sweep on EMBIC and NeuroBAT. Median number of recoverable measurements $\scriptstyle ( \# \mathrm { r e c } )$ as a function of the feasibility threshold $\varepsilon . { \mathrm { C A S } } \mathrm {  } { \mathrm { S N A P } }$ results are reported in Appendix D.2.

## 5.4 Experiment 3: Temporal Deployment Utility under Protocol Drift

While Experiment 2 evaluates harmonization outcomes under random splits from a fixed data distribution, real-world deployment requires applying harmonization decisions learned on earlier cohorts to data collected at a later time. Such temporal shifts may arise from changes in assessment procedures, rater behavior, cohort composition, or missingness patterns. Here we evaluate the temporal deployment utility of BFHD: whether measurement subsets identified as recoverable on early data remain feasible when deployed on later cohorts under potential protocol drift.

Temporal split and evaluation protocol. For each realworld dataset, we evaluate temporal deployment utility using a time-based split, where models are trained on early data and evaluated on a held-out later cohort; detailed split procedures and implementation details are provided in Appendix B.4.

Recoverable measurement subsets are defined on the training split and are not re-selected on the test split. For our approach, recoverable measurements are obtained by binarizing the learned output gates, while baseline methods infer recoverability on the training split using the same feasibility criterion as in Experiment 2. A training-period recoverable measurement is considered retained if it continues to satisfy $\mathrm { e r r } _ { \mathrm { t e s t } } < \varepsilon _ { \mathrm { c u t o f f } }$ on the held-out test split.

We report two deployment utility metrics: (i) the number of retained recoverable measurements on the test split (#Ret), and (ii) the retention rate (Ret%), defined as the proportion of training-period recoverable measurements that remain feasible at test time. Agreement strength of retained measurements is reported in Appendix D.3.

<table><tr><td>Data</td><td>Method</td><td> $\# R e c _ { A }$ </td><td> $\overline { { J _ { A } } }$ </td><td> $\overline { { \# A > 0 . 7 } }$ </td><td> $\mp A > 0 . 5$ </td><td> $\# { R e c } _ { B }$ </td><td> $\overline { { J _ { B } } }$ </td><td> $\overline { { \# B > 0 . 7 } }$ </td><td> $\mp B > 0 . 5$ </td></tr><tr><td rowspan="6">CAS(A) ↔ SNAP(B)</td><td>IRT</td><td>11</td><td>0.733</td><td> $\overline { { 4 . 5 } }$ </td><td>10.5</td><td>7</td><td>0.785</td><td>4</td><td>7</td></tr><tr><td>XGB</td><td>9</td><td>0.710</td><td>6</td><td>9</td><td>7</td><td>0.842</td><td>5</td><td>7</td></tr><tr><td>FRF</td><td>9</td><td>0.683</td><td>4.5</td><td>9</td><td>7.5</td><td>0.858</td><td>6</td><td>7.5</td></tr><tr><td>CMLP</td><td>9</td><td>0.640</td><td>5</td><td>9</td><td>7</td><td>0.764</td><td>6</td><td>7</td></tr><tr><td>NICE</td><td>11</td><td>0.867</td><td>4</td><td>11</td><td>9</td><td>0.774</td><td>5</td><td>9</td></tr><tr><td>Ours</td><td>11</td><td>0.930</td><td>4</td><td>11</td><td>9</td><td>0.817</td><td>5</td><td>9</td></tr><tr><td rowspan="6">EMBIC(A) ↔ NeuroBAT (B)</td><td>IRT</td><td>6</td><td>0.841</td><td>1</td><td>6</td><td>0</td><td>N/A</td><td>0</td><td>0</td></tr><tr><td>XGB</td><td>6</td><td>0.806</td><td>1</td><td>6</td><td>1</td><td>0.4</td><td>0</td><td>1</td></tr><tr><td>FRF</td><td>6</td><td>0.819</td><td>1</td><td>6</td><td>0</td><td>0.533</td><td>0</td><td>0</td></tr><tr><td>CMLP</td><td>5</td><td>0.739</td><td>1</td><td>5</td><td>0</td><td>0.491</td><td>0</td><td>0</td></tr><tr><td>NICE</td><td>7</td><td>0.806</td><td>1</td><td>7</td><td>2</td><td>0.523</td><td>0</td><td>1.5</td></tr><tr><td>Ours</td><td>7</td><td>0.910</td><td>1</td><td>7</td><td>2</td><td>0.689</td><td>0</td><td>2</td></tr></table>

Table 2: Real-world harmonization results under $\varepsilon _ { \mathrm { c u t o f f } } ~ = ~ 0 . 6 .$ For each row, A and B correspond to the two assessment systems. #Rec<sub>A</sub>/#Rec<sub>B</sub> are the median numbers of recoverable measurements and $J _ { A } / J _ { B }$ are Jaccard stability across 10 seeds. $\# A { > } 0 . 7 , \# A { > } 0 . 5$ (and analogously for B) count selected measurements whose CCC exceeds 0.7 or 0.5, used as diagnostic measures of feasibility.

<table><tr><td>Method</td><td> $\overline { { \# \mathrm { R e t } _ { A } } }$ </td><td> $\overline { { \mathrm { R e t } \% _ { A } } }$ </td><td> $\overline { { \# \mathrm { R e t } _ { B } } }$ </td><td> $\overline { { \mathrm { R e t } \% _ { B } } }$ </td></tr><tr><td>IRT</td><td>3</td><td> $\overline { { 4 2 . 9 \pm 0 . 0 } }$ </td><td>0</td><td> $\overline { { \bf N / A } }$ </td></tr><tr><td>XGB</td><td>6</td><td> $8 5 . 7 _ { \pm 0 . 0 }$ </td><td>0.5</td><td> $6 2 . 5 { \scriptstyle \pm 4 8 . 4 }$ </td></tr><tr><td>FRF</td><td>5</td><td> $5 6 . 0 { \scriptstyle \pm 8 . 0 }$ </td><td>0</td><td> $0 . 0 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td>CMLP</td><td>6</td><td> $7 1 . 0 { \scriptstyle \pm 7 . 5 }$ </td><td>1</td><td> $9 0 . 0 { \scriptstyle \pm 1 0 . 0 }$ </td></tr><tr><td>NICE</td><td>6</td><td> $7 7 . 4 _ { \pm 9 . 9 }$ </td><td>2</td><td> $4 9 . 2 _ { \pm 1 0 . 5 }$ </td></tr><tr><td>Ours</td><td>7</td><td> $1 0 0 . 0 { \scriptstyle \pm 0 . 0 }$ </td><td>3</td><td> $1 0 0 . 0 { \scriptstyle \pm 0 . 0 }$ </td></tr></table>

Table 3: Temporal deployment utility under protocol drift on the EMBIC $\chi _ { \mathrm { ( A )  } }$ NeuroBAT(B) assessment pair. $\varepsilon _ { \mathrm { c u t o f f } } ~ = ~ 0 . 6$ #Ret ${ \bf \nabla } _ { A } / \# { \bf R e t } _ { B }$ denote the numbers of training-period recoverable measurements that remain feasible on the held-out test split. $\mathrm { R e t \mathcal { P } } _ { O _ { A } } / \mathrm { R e t \mathcal { P } } _ { O _ { B } }$ report the corresponding retention rates. (mean <sub>±std</sub> across 10 seeds) Additional results in Appendix D.3.

Results. Table 3 summarizes temporal deployment results for the representative EMBIC↔NeuroBAT assessment pair. Across methods, our approach retains a larger number of training-period recoverable measurements under temporal shift while maintaining higher retention rates than baseline approaches. These results indicate that BFHD produces harmonization decisions that generalize more reliably across time and are therefore better suited for deployment in longitudinal and evolving clinical settings.

Additional utility analyses. Robustness under MCAR missingness[Little and Rubin, 2019; Rubin, 1976] and downstream transfer for prediction tasks are reported in Appendix C.

## 5.5 Experiment 4: Ablation and Stability Analysis

We examine how architectural and objective components contribute to the stability and behavior of BFHD.

Ablation settings. Starting from the full BFHD framework, we consider four ablations: (i) replacing the bidirectionally coupled invertible NICE backbone with a non-invertible tied weight backbone (TWMLP); (ii) switching the $L _ { 0 }$ gate reward to a standard $L _ { 0 }$ penalty; (iii) removing output gating; and (iv) replacing the hinge-based feasibility loss with a standard reconstruction loss. Each ablation modifies one component while keeping the others fixed.

<table><tr><td>Variant</td><td> $\# \mathrm { R e c } _ { A }$ </td><td> $\overline { { J _ { A } } }$ </td><td> $\overline { { \# \mathrm { R e c } _ { B } } }$ </td><td> $\overline { { J _ { B } } }$ </td></tr><tr><td>Ours (full)</td><td>7</td><td>0.910</td><td>2</td><td>0.689</td></tr><tr><td> $\mathbf { \partial _ { - } } \mathbf { T W M L P }$ </td><td>7</td><td>0.886</td><td>1</td><td>0.600</td></tr><tr><td> $L _ { 0 }$  penalty</td><td> $0 ^ { \dagger }$ </td><td>N/A</td><td>0†</td><td>N/A</td></tr><tr><td>- No gating</td><td>7</td><td>0.806</td><td>2</td><td>0.523</td></tr><tr><td>- No hinge</td><td>13†</td><td>N/A</td><td>9†</td><td>N/A</td></tr></table>

Table 4: Ablation study on EMBIC $( \mathbf { A } ) $ NeuroBAT (B) under $\varepsilon _ { \mathrm { c u t o f f } } = 0 . 6$ . Notation are consistent with Table 2. Additional results in Appendix D.4. † indicate trivial gating solutions.

Evaluation protocol. All variants are evaluated on representative real-world system pairs used in Experiment 2. We report the median number of recoverable measurements (#Rec) and the Jaccard similarity (J) across random seeds, which jointly capture coverage and stability.

Results and observations. Table 4 summarizes the ablation results. Replacing the $L _ { 0 }$ gate reward with a standard $L _ { 0 }$ penalty consistently leads to degenerate solutions, confirming the necessity of explicitly favoring maximal feasible subsets. Removing output gating reduces subset stability, indicating that enforcing feasibility at the reconstruction level is critical for consistent measurement identification. Replacing the invertible backbone with a non-invertible tied-weight MLP preserves recoverable subset size, with dataset-dependent effects on stability. This suggests that strict bidirectional parameter sharing contributes to robustness, while invertibility itself is not a prerequisite for feasibility-based coverage.

Overall, these results show that effective BFHD instantiations depend critically on the joint use of a shared bidirectional backbone, output-level gating, and gated hinge losses.

## 6 Conclusion

We formulated clinical harmonization as a bidirectional recoverability problem and proposed BFHD, a feasibilitydriven framework that identifies large bidirectionally recoverable measurement subsets under a shared operator. Across synthetic data and real clinical assessment pairs, BFHD consistently yields broader feasible harmonization coverage and higher subset stability than baselines. BFHD provides an explicit and interpretable decomposition for cross-protocol integration of heterogeneous clinical assessments.

## A Implementation Details

## A.1 Backbone Architecture

BFHD instantiates the shared bidirectional operator using a NICE-style invertible neural network with additive coupling layers. The invertible backbone is used purely as a mechanism for strict parameter sharing across directions, rather than for likelihood modeling.

Each coupling block is parameterized by a multi-layer perceptron with hidden widths (64, 64, 32, 32) for both sides, selected via greedy search due to the large architectural search space of invertible models. Once selected, the same backbone architecture is used across all datasets; only feasibility-related hyperparameters are tuned per dataset.

## A.2 Training Procedure

Training is performed using a three-stage procedure designed to stabilize joint optimization of the backbone and gating variables.

Stage 1: Backbone warm-up The invertible backbone is first trained with all gates forced open. The objective consists of feature-wise reconstruction losses in both directions. This stage provides a reasonable initialization before introducing feasibility constraints.

Stage 2: Joint optimization with feasibility constraints In the second stage, backbone parameters and gates are optimized jointly. We employ a hinge-style feasibility loss together with a gate reward implemented in an equivalent closepenalty form. For each feature, the objective takes the form

$$
\lambda _ { \mathrm { c l o s e } } ( 1 - \bar { g } ) + \bar { g } \cdot \mathrm { R e L U } \big ( \mathrm { N M S E } - \varepsilon _ { \mathrm { t r a i n } } \big ) ,
$$

where g¯ denotes the minibatch-averaged gate value. Minimizing the first term is equivalent to maximizing a constant reward for opening the gate, while the second term enforces feasibility. This formulation is numerically stable and equivalent to the gate-reward objective described in the main paper up to a constant shift.

During Stage 2, the training tolerance $\varepsilon _ { \mathrm { t r a i n } }$ is annealed linearly from an initially loose value to a stricter target value during the gate warm-up phase. In addition, the learning rate for gate parameters follows a staged schedule: it is initialized to a small value during early warm-up $( 1 \times 1 0 ^ { - 5 } )$ , increased during the main joint optimization phase to enable effective output selection $( 1 \times 1 0 ^ { - 3 } )$ , and gradually decreased again to stabilize the final gate configuration $( 5 \times \mathrm { 1 0 ^ { - 4 } } )$ . This schedule reduces sensitivity to initialization and improves subset stability.

Stage 3: Gate hardening and fine-tuning After Stage 2, gates are binarized and frozen. The backbone is then finetuned under the target feasibility tolerance. Early stopping with validation monitoring is applied in all stages, and the best checkpoint is carried forward between stages.

## A.3 Hard-concrete $L _ { 0 }$ output gates

Feature-wise output gates are implemented using a hardconcrete $( L _ { 0 } )$ relaxation. Each gate is parameterized by a learnable logit and produces a continuous value in [0, 1] during training, which is optionally hardened to a binary mask after convergence.

Unless otherwise stated, we use the following gate parameterization in all experiments: the Concrete temperature is set to $1 / 8 ;$ the stretched interval is fixed to [−1, 2]; and gate logits are initialized to match an initial drop rate of 0.65. These values are kept fixed across datasets and random seeds.

After training, gates are binarized and frozen for backbone fine-tuning. A measurement is reported as recoverable if its hardened gate is active.

## A.4 Optimization and Learning Rates

For all gated models (Ours and TWMLP), we optimize backbone parameters and gate parameters using separate learning rates. The backbone learning rate is fixed to $5 \times 1 0 ^ { - 5 }$ during stage 2 across all experiments, while gate parameters use a higher learning rate shared across gated models. This separation is critical in practice: using a single learning rate leads to unstable gating behavior or degraded backbone mappings.

All optimizations for neural network based models use Adam. Early stopping based on validation performance is applied to all neural network based models, including NICE, ours, TWMLP, and CMLP. Baselines without gating (CMLP, IRT, FRF, XGB) are trained using a single optimizer and a single learning rate.

## A.5 Hyperparameter Selection

The hyperparameter space of invertible networks is large, so we do not perform exhaustive grid search. Instead, we adopt a greedy search strategy over architectural depth and width, guided by validation feasibility coverage and subset stability. Once a backbone configuration is selected, it is fixed across datasets.

Feasibility-related hyperparameters are tuned per dataset. Specifically, we set $\lambda _ { \mathrm { c l o s e } } ~ = ~ 0 . 5 0$ for the EMBIC ↔ NeuroBAT pair and $\lambda _ { \mathrm { c l o s e } } = 0 . 4 5$ for $\mathrm { C A S }  \mathrm { S N A P } ;$ all other BFHD hyperparameters are shared across experiments.

In all experiments the training tolerance $\varepsilon _ { \mathrm { t r a i n } }$ is set to $\begin{array} { r } { \varepsilon _ { \mathrm { t r a i n } } = \frac { 1 } { 6 } \varepsilon _ { \mathrm { c u t o f f } } . } \end{array}$

## A.6 Baseline Implementations

We include the following baselines.

FRF tune the number of estimators over {100, 200, 500}.

XGBoost models are selected via grid search over n<sub>estimators</sub> $\in \ \{ 5 0 , 6 0 , 7 0 , 8 0 \}$ , max depth $\in \ \{ 2 , 3 , 4 \}$ , and learning $\mathrm { r a t e } \in \{ 0 . 0 5 , 0 . 1 \}$

The CMLP uses a 4-layer architecture with hidden widths (64, 64, 64, 32), ReLU activations, learning rate $1 0 ^ { - 4 }$ , and cycle-consistency weight $\lambda _ { \mathrm { c y c l e } } = 1$

The TWMLP replaces the invertible backbone with a non-invertible shared MLP with hidden widths (64, 64, 32). TWMLP uses the same output gating mechanism, feasibility losses, annealed $\varepsilon _ { \mathrm { t r a i n } } ,$ and learning-rate separation as our approach, isolating the effect of invertibility from that of gating.

## A.7 Protocol for Setting the Feasibility Tolerance ε<sub>cutoff</sub>

This subsection describes a practical protocol used to instantiate the feasibility tolerance $\varepsilon _ { \mathrm { c u t o f f } }$ in our experiments. It is not a contribution of this work, but a measurementanchored choice intended to ensure clinical plausibility and reproducibility.

Protocol We determine $\varepsilon _ { \mathrm { c u t o f f } }$ based on intrinsic measurement variability of the target assessment system, following the steps below:

1. If a subtest reports a standard error of measurement (SEM), we treat this value as an estimate of intrinsic measurement uncertainty in the reported scoring metric.

2. If SEM is not reported, but test-retest reliability (ICC) and standard deviation (SD) are available, we compute

$$
\mathrm { S E M } = \mathrm { S D } \sqrt { 1 - \mathrm { I C C } } .
$$

3. Since reconstruction error in our approach is measured as feature-wise mean squared error in z-scored space, we convert SEM to the corresponding NMSE-scale tolerance via squared normalization:

$$
\varepsilon _ { j } \ = \ \left( \frac { \mathrm { S E M } _ { j } } { \mathrm { S D } _ { j } } \right) ^ { 2 } ,
$$

where $\mathrm { S D } _ { j }$ denotes the standard deviation of measurement j before normalization.

4. $\mathrm { I f } \ \mathrm { S D } _ { j }$ in the same scoring metric is not available $( \mathrm { e . g . }$ SEM is reported on a latent-trait scale or only aggregated uncertainty summaries are provided), we treat the resulting tolerance as a measurement-anchored reference operating point and set $\varepsilon _ { \mathrm { c u t o f f } }$ by a clinically reasonable approximation based on the available information $( \mathrm { e . g . }$ conservative upper/lower bounds reported across ability ranges).

5. We take the largest subtest-level tolerance $\varepsilon _ { j }$ as a conservative system-level feasibility threshold ε.

Usage The resulting $\varepsilon _ { \mathrm { c u t o f f } }$ serves as a reference threshold rather than a universal constant. In practice, users may adjust ε within a reasonable range around this baseline depending on application requirements. We therefore report ε<sub>cutoff</sub>- sweep analyses to illustrate sensitivity.

Instantiation in this work We instantiate the above protocol using measurement statistics reported in prior studies.

SNAP-IV. For SNAP-IV, test-retest reliability (ICC) and standard deviation (SD) values are reported in [Huang et al., 2023]. Specifically, ICC values of 0.390 (SD = 0.78) for Hyperactivity/Impulsivity are reported. Using SEM = SD 1 − ICC and converting to the normalized space used in our experiments, this yields $\varepsilon _ { j }$ values of approximately 0.61. We therefore adopt $\varepsilon _ { \mathrm { c u t o f f } } = 0 . 6$ for SNAP-IV.

NeuroBAT (ADNI). For NeuroBAT subtests, SEM are directly reported in [Crane et al., 2023]. As documented in that study, visuospatial function exhibits notably elevated measurement uncertainty. Because these SEM values are reported on a latent ability scale and corresponding feature-level score variances are not available, they cannot be uniquely converted to feature-wise z-space NMSE. We therefore use the reported SEM values to anchor a conservative reference tolerance, and set $\varepsilon _ { \mathrm { c u t o f f } } = 0 . 6$ for NeuroBAT.

## B Datasets and Preprocessing

## B.1 Synthetic Data Generation

We construct paired synthetic observations (X, Y) to emulate two heterogeneous clinical assessment systems measured on the same population. The generation process explicitly separates shared (recoverable) structure from system-specific (non-translatable) structure, while incorporating redundancy, weak recoverability, and non-invertible distortions commonly observed in real clinical instruments.

## Latent Variables

Let

$$
\begin{array} { r } { z ^ { ( c ) } \in \mathbb { R } ^ { d _ { c } } , \qquad z ^ { ( x ) } \in \mathbb { R } ^ { d _ { x } } , \qquad z ^ { ( y ) } \in \mathbb { R } ^ { d _ { y } } } \end{array}
$$

denote latent random vectors corresponding to shared (common) factors, X-specific private factors, and Y -specific private factors, respectively. All latent variables are mutually independent and follow standard Gaussian distributions:

$$
\begin{array} { r } { z ^ { ( c ) } \sim \mathcal { N } ( 0 , I _ { d _ { c } } ) , \quad z ^ { ( x ) } \sim \mathcal { N } ( 0 , I _ { d _ { x } } ) , \quad z ^ { ( y ) } \sim \mathcal { N } ( 0 , I _ { d _ { y } } ) . } \end{array}
$$

Feature Partition and Ground-Truth Recoverability Each system is partitioned into recoverable and nontranslatable components:

$$
X = \left( X ^ { \mathrm { r e c } } , X ^ { \mathrm { n t } } \right) , \qquad Y = \left( Y ^ { \mathrm { r e c } } , Y ^ { \mathrm { n t } } \right) .
$$

A feature is labeled recoverable if it depends exclusively on the shared latent variables $z ^ { ( c ) }$ , up to noise and deterministic transformations. Features depending on private latent variables are labeled non-translatable. Ground-truth recoverability indicators are defined as

$$
\operatorname { G T } _ { X } ( j ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f ~ } } X _ { j } { \mathrm { ~ d e p e n d s ~ o n l y ~ o n ~ } } z ^ { ( c ) } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } , } \end{array} \right. }
$$

$$
\operatorname { G T } _ { Y } ( j ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { i f } } Y _ { j } { \mathrm { ~ d e p e n d s ~ o n l y ~ o n ~ } } z ^ { ( c ) } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right. }
$$

## Generation of Recoverable Features

Recoverable features are generated as linear projections of the shared latent variables:

$$
X ^ { \mathrm { r e c } } = z ^ { ( c ) } W _ { X } , \qquad Y ^ { \mathrm { r e c } } = z ^ { ( c ) } W _ { Y } ,
$$

where $W _ { X } \in \mathbb { R } ^ { d _ { c } \times k _ { X } }$ and $W _ { Y } \in \mathbb { R } ^ { d _ { c } \times k _ { Y } }$ are random matrices with i.i.d. Gaussian entries.

Redundant recoverable features. To model redundant clinical items measuring the same latent construct, we introduce redundancy groups. For each group indexed by g, a subset of recoverable features share the same underlying signal:

$$
X _ { j } ^ { \mathrm { r e c } } = X _ { i } ^ { \mathrm { r e c } } , \qquad \forall j \in { \mathcal { G } } _ { i } ,
$$

where $\mathcal { G } _ { i }$ denotes the redundancy group associated with anchor feature i. All features within a redundancy group are labeled recoverable in the ground truth.

Weakly recoverable features. A subset of recoverable features is designated as weakly recoverable by attenuating their shared signal:

$$
X _ { j } ^ { \mathrm { r e c } }  \alpha X _ { j } ^ { \mathrm { r e c } } , \qquad Y _ { j } ^ { \mathrm { r e c } }  \alpha Y _ { j } ^ { \mathrm { r e c } } ,
$$

where $0 < \alpha < 1$ . These features lie near the ε-recoverability boundary and introduce ambiguity under noise.

## Generation of Non-Translatable Features

System-specific features are generated exclusively from private latent variables:

$$
X ^ { \mathrm { n t } } = z ^ { ( x ) } A _ { X } , \qquad Y ^ { \mathrm { n t } } = z ^ { ( y ) } A _ { Y } ,
$$

where $A _ { X } \ \in \ \mathbb { R } ^ { d _ { x } \times ( d _ { X } - k _ { X } ) }$ and $A _ { Y } ~ \in ~ \mathbb { R } ^ { d _ { y } \times ( d _ { Y } - k _ { Y } ) }$ are random Gaussian matrices. These features share no information across systems and are therefore non-recoverable by construction.

## Measurement Noise

Independent Gaussian noise is added to all features:

$$
\begin{array} { r l r } { X ^ { \mathrm { r e c } } \gets X ^ { \mathrm { r e c } } + \varepsilon _ { X } ^ { ( r ) } , } & { } & { X ^ { \mathrm { n t } } \gets X ^ { \mathrm { n t } } + \varepsilon _ { X } ^ { ( p ) } , } \\ { Y ^ { \mathrm { r e c } } \gets Y ^ { \mathrm { r e c } } + \varepsilon _ { Y } ^ { ( r ) } , } & { } & { Y ^ { \mathrm { n t } } \gets Y ^ { \mathrm { n t } } + \varepsilon _ { Y } ^ { ( p ) } , } \end{array}
$$

where $\varepsilon ^ { ( r ) } \sim \mathcal { N } ( 0 , \sigma _ { r } ^ { 2 } I )$ and $\varepsilon ^ { ( p ) } \sim \mathcal { N } ( 0 , \sigma _ { p } ^ { 2 } I )$ . The noise levels $\sigma _ { r }$ and $\sigma _ { p }$ control the difficulty of the recoverability task.

## Nonlinear Folding

To break global invertibility and introduce realistic information loss, each feature undergoes a nonlinear folding operation with probability $p _ { \mathrm { n l } }$ :

$$
X _ { j }  \phi ( X _ { j } ) , \qquad Y _ { j }  \phi ( Y _ { j } ) ,
$$

where $\phi ( \cdot )$ is randomly chosen from

$$
\phi ( u ) \in \{ \operatorname { t a n h } ( u ) , | u | , u ^ { 2 } \} .
$$

These transformations preserve statistical dependence on shared latent variables while destroying strict invertibility.

## Synthetic parameterization (Exp. 1)

All synthetic results in Exp. 1 use our paired synthetic generator (see released code).

Unless otherwise stated, we use the following parameterization:

• Sample size and feature dimensions: $n = 2 0 0 0 , d _ { X } =$ 20, d<sub>Y</sub> = 20.

• Latent dimensions: shared latent d<sub>c</sub> = 4; private latents $d _ { p X } = 4 , d _ { p Y } = 4 .$

• Recoverable feature fractions: $\mathtt { f r a c \_ r e c \_ x } = \ 0 . 5 .$ frac rec y = 0.5 (thus $k _ { X } = 1 0$ and $k _ { Y } = 1 0 \ \mathrm { r e } .$ coverable features per view).

• Redundancy in recoverables: n redundant groups x = 3, n redundant groups $\begin{array} { r l r } { \mathrm { ~  ~ \nabla ~ } _ { - \mathrm { ~ \bf ~ Y ~ } } } & { { } = } & { 3 , } \end{array}$ , with group size redundant group size = 3.

• Weak-shared (near-boundary) recoverables: weak shared $\mathrm { . } \mathrm { . } \mathrm { . } \mathrm { . } \mathrm { \Delta . } \mathrm { c } \mathrm { . } = \mathrm { 0 . 3 5 }$ of recoverable features are attenuated by weak shared scale = 0.30.

• Noise levels: recoverable-feature noise no $. S \in . . x \in \mathbb { C } \mathrm { ~ = ~ }$ 0.30; private-feature noise noise priv = 0.35.

• Nonlinear folding (many-to-one distortion): applied independently per feature with probability nonlinear prob = 0.95, using fold type = mix (randomly chosen from tanh(·), | · |, and $( \cdot ) ^ { 2 } ) .$

The generator additionally records (and returns) the indices of recoverable vs. non-translatable features, redundancy groups, weak-recoverable indices, and folded-feature indices, which we use to compute ground-truth recoverability labels and diagnostics.

## Summary

This construction yields paired observations (X, Y) in which recoverable and non-translatable features are known by design, yet obscured by redundancy, weak shared structure, noise, and non-invertible distortions. The resulting synthetic datasets enable controlled evaluation of feature-level bidirectional recoverability and stability under realistic conditions.

## B.2 ADNI Cognitive Assessments (EMBIC and NeuroBAT)

We use paired cognitive assessments from the Alzheimer’s Disease Neuroimaging Initiative (ADNI). Two assessment systems are considered: EMBIC and NeuroBAT.

Assessment construction For EMBIC, we use the mean scores of all available tests. Standard deviations are not included, as they are not semantically reconstructable across assessment systems. The resulting EMBIC representation contains 13 numerical measurements.

For NeuroBAT, we include all available subtests, but only use total scores of individual items or subtests. This design choice avoids incorporating fine-grained ordinal response data. The resulting NeuroBAT representation contains 9 numerical measurements.

Pairing and filtering Each subject contributes at most one paired observation. When multiple assessments are available for a subject, only the earliest (first) testing instance is retained. Two assessments from different systems are considered paired if their testing dates fall within a 28-day window; otherwise, they are discarded.

Missing values are handled by complete-case analysis: any sample with missing values in either system is removed. Given the relatively large sample size, no imputation is performed.

Dataset size and availability After preprocessing, the ADNI dataset contains 1,629 paired observations. The data are publicly available and can be obtained by application through the ADNI website, subject to standard data use agreements.

## B.3 ADHD Assessments (CAS and SNAP-IV)

We use a paired ADHD assessment dataset consisting of DN:CAS (CAS) and SNAP-IV (SNAP), provided by our clinical collaborators.

Assessment construction For SNAP-IV, we include only the Inattention (ANI) and Hyperactivity/Impulsivity (H/I) subscales. The resulting SNAP-IV representation contains 18 numerical measurements.

For CAS, we use all subscale scores that have been transformed using normative conversion, as well as subtest total scores and the overall scale total score. This yields a CAS representation with 27 numerical measurements.

Pairing and filtering As with ADNI, each subject contributes at most one paired observation, corresponding to the earliest available assessment. CAS and SNAP-IV assessments are paired if their testing dates fall within a 28-day window.

Missing values are handled via complete-case analysis: samples with any missing entries in either system are removed. No imputation is applied.

Labels and dataset size This dataset includes a binary diagnostic label indicating typical development (TD) or ADHD. The label distribution is balanced. After preprocessing, the dataset contains 336 paired observations.

Data availability The data are available upon reasonable request, subject to the relevant data use agreement. For further information on requesting access, please refer to the ADHD dataset section of the project repository.

## B.4 Data Splitting Protocol

All datasets are split using a 7:1:2 ratio for training, validation, and testing. Splits are constructed in two stages: first an 8:2 split is performed, followed by a 7:1 split of the training portion into training and validation sets.

To evaluate robustness under temporal shift, all splits for exp. 3 are time-based. Samples are sorted by assessment date, and earlier samples are assigned to training and validation, while later samples are reserved for testing. This protocol is used consistently across all experiments unless otherwise specified.

## C Additional Utility Analyses

All experiments in this section are conducted on the CAS ↔ SNAP assessment pair, where CAS corresponds to system A (X) and SNAP corresponds to system B (Y), following the notation used in the main paper.

## C.1 Experiment 5: Downstream Prediction under Size-Matched Harmonized Subsets

This experiment evaluates whether measurement subsets identified as bidirectionally recoverable by different harmonization methods provide practical utility for downstream prediction when only one assessment system is available at deployment time. Unlike Experiments 2-4, which focus on feasibility, coverage, and stability of harmonization itself, this experiment assesses whether harmonized and translated measurements remain usable for an external prediction task. The goal is not to outperform models trained on fully observed target assessments, but to compare methods under a controlled and deployment-relevant setting.

## Experimental Setup

Dataset and task This experiment is conducted only on the CAS (X) ↔ SNAP (Y) assessment pair. While the main paper evaluates harmonization on multiple assessment systems, downstream prediction in Experiment 5 requires an externally defined label z that is consistently available across paired assessments. Such labels are available for CAS ↔ SNAP but not for EMBIC ↔ NeuroBAT, and we therefore restrict this experiment to CAS ↔ SNAP to avoid introducing taskdependent inconsistencies.

We consider a paired clinical assessment dataset (X, Y) with a single downstream label z. The same prediction task and label are used throughout this experiment. Results are reported on one representative real-world dataset due to label availability constraints.

Prediction space and deployment setting Prediction is always performed in a fixed feature space, either X or Y. At deployment time, only the opposite assessment system is assumed to be available. Accordingly, harmonization models are used to translate the available view into the prediction space before classification. Single-view baselines (X-only and Y-only) use the observed measurements directly in the corresponding prediction space and serve as reference.

Subset construction To ensure a fair comparison across methods, we control for subset size. Let k denote the median number of recoverable measurements identified by our approach under the feasibility threshold ε in the prediction space. For our approach, the recoverable subset is obtained by binarizing the learned output gates; if the resulting subset size differs from k, measurements are truncated or supplemented based on gate values. For baseline methods without explicit gating, exactly k measurements are selected by ranking measurements according to validation NMSE when predicting the prediction-space view from the opposite view. This yields a size-matched measurement subset for each method.

Prediction models For each method and each random seed, a linear SVM classifier is trained independently on the training split using the corresponding feature representation. No classifier weights are shared across methods or seeds. We consider three measurement representations:

1. Observed: observed measurements in the prediction space restricted to the selected subset.

2. Translated: translated measurements in the prediction space, obtained by applying the harmonization model from the opposite view.

3. Single-view reference: the full observed predictionspace measurements (X-only or Y-only).

To reduce optimistic bias, translated measurements used for training classifiers are generated using a held-out validation split. All classifiers use identical architectures and hyperparameters to isolate the effect of the selected subsets and translation quality.

Evaluation metrics and seeds We report classification accuracy (Acc). Results are reported as mean ± standard deviation over ten random seeds, using a fixed train/validation/test split.

Table 5: Downstream prediction under size-matched harmonized subsets. Results are reported as mean ± standard deviation over ten random seeds. Obs denotes using observed measurements in the prediction space, while Trans denotes using translated measurements obtained from the opposite view. Single-view reference (Xonly / Y-only) serve as reference result.
<table><tr><td colspan="3">Deploy  $X  { \hat { Y } }$  (source-side: X)</td></tr><tr><td>Method</td><td> $\operatorname { A c c } \left( \operatorname { O b s } \right)$ </td><td>Acc (Trans)</td></tr><tr><td rowspan="3">X-only CMLP NICE Ours</td><td> $0 . 9 0 9 \pm 0 . 0 4 7$ </td><td> $0 . 8 8 1 \pm 0 . 0 4 7$ </td></tr><tr><td> $0 . 9 1 9 \pm 0 . 0 2 4$   $0 . 9 1 6 \pm 0 . 0 2 3$ </td><td> $0 . 8 9 7 \pm 0 . 0 3 3$ </td></tr><tr><td> $0 . 9 1 3 \pm 0 . 0 2 3$ </td><td> $0 . 9 0 1 \pm 0 . 0 3 7$ </td></tr><tr><td colspan="3">Deploy  $Y  { \hat { X } }$  (source-side: Y)</td></tr><tr><td>Method</td><td> $\operatorname { A c c } \left( \operatorname { O b s } \right)$ </td><td>Acc (Trans)</td></tr><tr><td>Y-only</td><td> $0 . 9 8 7 \pm 0 . 0 0 8$ </td><td></td></tr><tr><td>CMLP NICE</td><td> $0 . 9 9 0 \pm 0 . 0 1 0$   $0 . 9 8 7 \pm 0 . 0 0 8$ </td><td> $0 . 9 6 8 \pm 0 . 0 1 8$ </td></tr><tr><td>Ours</td><td> $0 . 9 9 0 \pm 0 . 0 1 0$ </td><td> $0 . 9 7 4 \pm 0 . 0 1 9$ </td></tr><tr><td></td><td></td><td> $0 . 9 8 7 \pm 0 . 0 1 1$ </td></tr></table>

## Results

Table 5 summarizes downstream prediction performance under size-matched harmonized subsets. Across both deployment settings, using translated measurements leads to lower classification accuracy than using observed measurements. Nevertheless, BFHD consistently achieves the highest mean accuracy on translated measurements and the smallest performance degradation among the compared harmonization methods.

Overall, these results indicate that feasibility-driven subset identification yields harmonized measurement representations that are not only bidirectionally reconstructable under the chosen tolerance, but also more reliable for downstream prediction when only one assessment system is available at deployment time.

## C.2 Experiment 6: Robustness under MCAR Missingness

This experiment evaluates the robustness of learned harmonizable measurement subsets under test-time missingness, reflecting deployment scenarios in which one assessment system may be partially incomplete. Unlike previous experiments, which assume fully observed paired assessments at evaluation time, we introduce missingness only at inference while keeping training conditions unchanged.

## Experimental setup

All harmonization models are trained using fully observed paired data (X, Y). At evaluation time, we consider a Missing Completely At Random (MCAR) mechanism applied to one assessment system at a time. For a missing rate $p \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ , each measurement entry in the selected source view is independently dropped with probability p. Missing values are imputed using mean imputation in standardized space, implemented by zero-filling after applying a StandardScaler fit on the training split. The opposite assessment system remains fully observed.

We evaluate both deployment directions: (i) missingness in X followed by translation to $Y ( X _ { \mathrm { m i s s } }  { \hat { Y } } )$ , and (ii) missingness in Y followed by translation to $X ( Y _ { \mathrm { m i s s } }  { \hat { X } } )$ . This symmetric evaluation ensures that robustness is assessed independently of the choice of source assessment.

## Evaluation protocol

For each method, a reference set of feasible measurements is defined using the clean evaluation setting $( p = 0 )$ , which corresponds to the fully observed results reported in Experiment 2. Under increasing missingness, we measure how many of these originally feasible measurements remain feasible.

We report the retention rate, defined as the fraction of originally feasible measurements that remain feasible under missingness. Results are reported as mean ± standard deviation across random seeds.

## Results and discussion

Table 6 summarizes robustness under increasing levels of test-time missingness. Across both deployment directions, retention rates decrease monotonically as the missing rate increases, reflecting the expected degradation under partial input corruption.

At mild missingness $( p _ { \mathrm { m i s s } } = 0 . 1 )$ , our approach consistently retains a larger fraction of originally feasible measurements than CMLP and is competitive with NICE, while exhibiting lower variability across seeds. At moderate and severe missingness levels $( p _ { \mathrm { m i s s } } = 0 . 2$ and 0.3), ours maintains comparable or higher retention rates than baselines in both directions, with more stable behavior under increasing uncertainty.

Importantly, robustness trends are consistent across the two deployment directions, indicating that the learned bidirectional structure is not overly sensitive to which assessment system is partially observed at inference time. These results suggest that feasibility-driven measurement decomposition yields harmonization decisions that degrade more gracefully under test-time missingness, supporting their use in realistic deployment settings where assessments may be incomplete.

## D Extended Experimental Results

## D.1 Additional Results for Experiment 1: Synthetic Experiments

Remarks Table 7 summarizes index-level structure recovery results across different data-generation seeds. Across all settings, BFHD achieves consistently strong recoverability accuracy and high subset stability. Stability is better reflected by Jaccard similarity $( J _ { X } , \ J _ { Y } )$ than by accuracy variance, since index-level accuracy is a discrete metric that can saturate once the correct subset size is reliably identified.

CCC statistics are reported as a diagnostic measure of agreement strength among selected measurement indices. We note that the BFHD objective is feasibility-driven rather than accuracy-driven: the goal is to identify large bidirectionally recoverable subsets under a prescribed tolerance, not to optimize pointwise reconstruction or agreement metrics. Accordingly, CCC values need not be maximal, and moderate

Table 6: Robustness under MCAR missingness. For each method, #Ret denotes the median number of measurements that remain feasible under missingness. Retain denotes the retention rate relative to the reference feasible set defined at $p = 0$ (Experiment 2). Results are reported as mean ± standard deviation across random seeds.
<table><tr><td rowspan="2"> $p _ { \mathrm { m i s s } }$ </td><td rowspan="2">Method</td><td colspan="3">Missing in X  $( X _ { \mathrm { m i s s } }  { \hat { Y } } )$ </td><td colspan="3">Missing in  $Y ( Y _ { \mathrm { m i s s } }  { \hat { X } } )$ </td></tr><tr><td>#Ret</td><td>Retain</td><td>Std</td><td>#Ret</td><td>Retain</td><td>Std</td></tr><tr><td>0.1</td><td>Ours</td><td>9</td><td>0.860</td><td>0.097</td><td>7</td><td>0.813</td><td>0.140</td></tr><tr><td></td><td>NICE</td><td>8.5</td><td>0.788</td><td>0.152</td><td>7</td><td>0.877</td><td>0.121</td></tr><tr><td></td><td>CMLP</td><td>8</td><td>0.815</td><td>0.196</td><td>6</td><td>0.725</td><td>0.293</td></tr><tr><td>0.2</td><td>Ours</td><td>6.5</td><td>0.513</td><td>0.158</td><td>5.5</td><td>0.634</td><td>0.112</td></tr><tr><td></td><td>NICE</td><td>6.5</td><td>0.524</td><td>0.203</td><td>6</td><td>0.632</td><td>0.168</td></tr><tr><td></td><td>CMLP</td><td>6</td><td>0.676</td><td>0.277</td><td>4</td><td>0.579</td><td>0.317</td></tr><tr><td>0.3</td><td>Ours</td><td>4.5</td><td>0.333</td><td>0.142</td><td>2.5</td><td>0.315</td><td>0.142</td></tr><tr><td></td><td>NICE</td><td>2.5</td><td>0.240</td><td>0.145</td><td>2</td><td>0.286</td><td>0.130</td></tr><tr><td></td><td>CMLP</td><td>2</td><td>0.226</td><td>0.143</td><td>2</td><td>0.296</td><td>0.269</td></tr></table>

Table 7: Experiment 1 (Synthetic): feasibility structure recovery and stability across different data-generation seeds $( s _ { \mathrm { g e n } } \in \{ 1 , \ldots , 5 \} )$ . Acc is index-level recoverability label accuracy (mean ± std over training seeds). $J _ { X }$ and $J _ { Y }$ are Jaccard similarities of selected recoverable subsets across training seeds; $\scriptstyle { \mathcal { J } } _ { \mathrm { o v e r a l l } }$ is the average of $J _ { X }$ and J<sub>Y</sub>. CCC statistics summarize agreement strength among selected measurement indices (mean ± std).
<table><tr><td>Data</td><td>Method</td><td> $\operatorname { A c c } _ { X }$ </td><td> $\operatorname { A c c } _ { Y }$ </td><td> $J _ { X }$ </td><td> $J _ { Y }$ </td><td> $J _ { \mathrm { o v e r a l l } }$ </td><td> $\mathbf { C C C } _ { X }$ </td><td> $\mathrm { C C C } _ { Y }$ </td></tr><tr><td rowspan="3"> $s _ { \mathrm { g e n } } = 1$ </td><td>Ours</td><td> $0 . 9 1 5 \pm 0 . 0 2 3$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td>0.948</td><td>1.000</td><td>0.974</td><td> $0 . 7 3 8 \pm 0 . 0 1 4$ </td><td> $0 . 7 5 4 \pm 0 . 0 6 3$ </td></tr><tr><td>CMLP</td><td> $0 . 8 9 0 \pm 0 . 0 3 7$ </td><td> $0 . 9 0 0 \pm 0 . 0 0 0$ </td><td>0.896</td><td>1.000</td><td>0.948</td><td> $0 . 7 6 8 \pm 0 . 0 1 9$ </td><td> $0 . 7 5 8 \pm 0 . 0 1 4$ </td></tr><tr><td>NICE</td><td> $0 . 8 5 0 \pm 0 . 0 0 0$ </td><td> $0 . 8 0 5 \pm 0 . 0 4 7$ </td><td>1.000</td><td>0.824</td><td>0.912</td><td> $0 . 7 2 2 \pm 0 . 0 1 3$ </td><td> $0 . 7 4 5 \pm 0 . 0 1 3$ </td></tr><tr><td rowspan="3"> $s _ { \mathrm { g e n } } = 2$ </td><td>Ours</td><td> $0 . 8 3 0 \pm 0 . 0 5 6$ </td><td> $0 . 7 5 5 \pm 0 . 0 4 2$ </td><td>0.783</td><td>0.812</td><td>0.797</td><td> $0 . 5 7 4 \pm 0 . 0 3 3$ </td><td> $0 . 6 1 9 \pm 0 . 0 4 9$ </td></tr><tr><td>CMLP</td><td> $0 . 8 1 0 \pm 0 . 0 3 7$ </td><td> $0 . 7 1 0 \pm 0 . 0 3 0$ </td><td>0.848</td><td>0.722</td><td>0.785</td><td> $0 . 6 1 2 \pm 0 . 0 2 3$ </td><td> $0 . 6 7 8 \pm 0 . 0 2 4$ </td></tr><tr><td>NICE</td><td> $0 . 5 5 0 \pm 0 . 0 3 2$ </td><td> $0 . 6 0 5 \pm 0 . 0 3 5$ </td><td>0.400</td><td>0.544</td><td>0.472</td><td> $0 . 6 6 8 \pm 0 . 0 4 3$ </td><td> $0 . 6 4 8 \pm 0 . 0 1 6$ </td></tr><tr><td rowspan="3"> $s _ { \mathrm { g e n } } = 3$ </td><td>Ours</td><td> $0 . 8 9 5 \pm 0 . 0 1 5$ </td><td> $0 . 8 8 5 \pm 0 . 0 5 0$ </td><td>0.975</td><td>0.858</td><td>0.916</td><td> $0 . 6 2 2 \pm 0 . 0 1 0$ </td><td> $0 . 6 1 6 \pm 0 . 0 2 8$ </td></tr><tr><td>CMLP</td><td> $0 . 8 4 5 \pm 0 . 0 5 2$ </td><td> $0 . 8 2 5 \pm 0 . 0 2 5$ </td><td>0.773</td><td>0.846</td><td>0.809</td><td> $0 . 6 5 9 \pm 0 . 0 1 9$ </td><td> $0 . 6 8 3 \pm 0 . 0 2 4$ </td></tr><tr><td>NICE</td><td> $0 . 6 1 5 \pm 0 . 0 3 2$ </td><td> $0 . 6 5 0 \pm 0 . 0 3 2$ </td><td>0.624</td><td>0.800</td><td>0.712</td><td> $0 . 7 4 1 \pm 0 . 0 2 3$ </td><td> $0 . 7 4 0 \pm 0 . 0 2 3$ </td></tr><tr><td rowspan="3"> $s _ { \mathrm { g e n } } = 4$ </td><td>Ours</td><td> $0 . 8 4 0 \pm 0 . 0 2 0$ </td><td> $0 . 8 7 5 \pm 0 . 0 2 5$ </td><td>0.883</td><td>0.931</td><td>0.907</td><td> $0 . 5 7 6 \pm 0 . 0 2 8$ </td><td> $0 . 5 9 5 \pm 0 . 0 1 5$ </td></tr><tr><td>CMLP</td><td> $0 . 7 4 5 \pm 0 . 0 5 2$ </td><td> $0 . 8 5 5 \pm 0 . 0 4 7$ </td><td>0.776</td><td>0.815</td><td>0.796</td><td> $0 . 6 4 8 \pm 0 . 0 2 4$ </td><td> $0 . 6 2 2 \pm 0 . 0 2 3$ </td></tr><tr><td>NICE</td><td> $0 . 5 9 0 \pm 0 . 0 3 0$ </td><td> $0 . 6 2 0 \pm 0 . 0 2 4$ </td><td>0.569</td><td>0.822</td><td>0.695</td><td> $0 . 7 2 8 \pm 0 . 0 4 8$ </td><td> $0 . 7 0 3 \pm 0 . 0 2 6$ </td></tr><tr><td rowspan="3"> $s _ { \mathrm { g e n } } = 5$ </td><td>Ours</td><td> $0 . 8 8 0 \pm 0 . 0 2 4$ </td><td> $0 . 8 3 0 \pm 0 . 0 2 4$ </td><td>0.933</td><td>0.924</td><td>0.929</td><td> $0 . 7 0 6 \pm 0 . 0 2 9$ </td><td> $0 . 6 9 0 \pm 0 . 0 2 5$ </td></tr><tr><td>CMLP</td><td> $0 . 8 5 5 \pm 0 . 0 1 5$ </td><td> $0 . 8 1 0 \pm 0 . 0 4 9$ </td><td>0.975</td><td>0.789</td><td>0.882</td><td> $0 . 7 2 8 \pm 0 . 0 1 8$ </td><td> $0 . 7 1 6 \pm 0 . 0 3 5$ </td></tr><tr><td>NICE</td><td> $0 . 7 3 5 \pm 0 . 0 3 9$ </td><td> $0 . 7 2 5 \pm 0 . 0 4 0$ </td><td>0.837</td><td>0.814</td><td>0.826</td><td> $0 . 6 9 8 \pm 0 . 0 2 4$ </td><td> $0 . 6 8 3 \pm 0 . 0 1 5$ </td></tr></table>

CCC levels are expected for measurements that lie near the feasibility boundary.

Experiment 2.

## D.2 Additional Results for Experiment 2: Real-World Harmonization

This section provides supplementary results for Experiment 2 on the CAS↔SNAP assessment pair. While the main paper reports fixed-threshold harmonization outcomes (including subset stability diagnostics) and presents an ε-sweep for EMBIC↔NeuroBAT, the analysis below complements the main text by showing the corresponding ε-sweep on CAS↔SNAP.

Feasibility-coverage trends on CAS↔SNAP Figure 2 reports the median number of recoverable measurements (#Rec) as a function of the feasibility threshold ε, shown separately in the CAS (A) and SNAP (B) spaces. All methods are evaluated under the same feasibility criterion used throughout

Across a broad and clinically relevant range of thresholds, BFHD consistently identifies a larger number of feasible measurements than baseline methods in both directions. The advantage is most pronounced in the intermediate regime, where partial but nontrivial overlap between the two assessment systems is expected. As ε increases, all methods approach full coverage; however, BFHD exhibits an earlier and smoother increase in #Rec across both assessment spaces, yielding a more interpretable coverage-feasibility frontier.

Remark Subset stability across random seeds is reported in the main paper (via Jaccard similarity) at the reference threshold, and the CAS↔SNAP ε-sweep shown here is consistent with the same qualitative behavior: BFHD expands feasible harmonization coverage under a shared feasibility criterion.

![](images/772ef00c36c1588d68e06724063e016374fed1e4f512fb6141db50f18cd3ccad.jpg)

![](images/e56306d9d831525cd7d41f5efa0cfcdb4b78edfadb7ef6b79b230b07c7c8b755.jpg)  
Figure $2 \colon$ ε-sweep on CAS↔SNAP. Median number of recoverable measurements (#Rec) as a function of the feasibility threshold ε, shown separately for CAS (A) and SNAP (B). Results are aggregated over random seeds.

## D.3 Additional Results for Experiment 3: Temporal Deployment under Protocol Drift

This section reports additional temporal deployment results under protocol drift for both real-world assessment pairs. For completeness and ease of comparison, we include the EMBIC↔NeuroBAT results shown in the main paper and extend them with agreement diagnostics on the held-out test period. In addition, we report the corresponding results on CAS↔SNAP, which are omitted from the main paper due to space constraints.

Protocol and metrics. The experimental protocol is identical to that described in the main paper. For each deployment direction (source → target), we report: (i) #Ret, the median number of training-period recoverable target-space measurements that remain feasible on the test period; (ii) Ret%, the retention rate on the test period (mean ± std across random seeds); and (iii) CCC-based diagnostic counts computed over retained measurements on the test period, reported as median counts across seeds.

Remarks Across both dataset pairs and both deployment directions, BFHD retains a larger fraction of training-period recoverable measurements on the later cohort while maintaining strong or comparable CCC-based agreement diagnostics among the retained measurements. Notably, for EMBIC→NeuroBAT, several baselines identify few or no training-period recoverable target measurements, resulting in undefined retention rates; BFHD avoids this failure mode and yields stable retention across random seeds.

## D.4 Additional Results for Experiment 4: Ablation Results on CAS↔SNAP

This section reports additional ablation results for Experiment 4 on the CAS↔SNAP assessment pair. The ablation settings, feasibility threshold, and evaluation metrics are identical to those used in the main paper. These results are included to verify that the qualitative conclusions drawn from EMBIC↔NeuroBAT generalize to a second real-world dataset with different measurements dimensionality and assessment structure.

Table 9 summarizes recoverable subset size and stability across random seeds. Consistent with the main paper, replacing the invertible backbone with a tied-weight MLP (TWMLP) yields comparable recoverable coverage and subset stability. In contrast, removing output gating leads to noticeably reduced stability, despite similar coverage. Replacing the gate reward with a standard $L _ { 0 }$ penalty or removing the hinge-based feasibility loss results in degenerate gating behavior and therefore does not admit meaningful stability statistics.

Table 8: Extended results for experiment 3. For each deployment direction (source → target), #Ret is the median number of retained recoverable measurements in the target space, and Ret% is the corresponding retention rate (mean ± std across random seeds), and is reported in percentage units. CCC>0.5/0.6/0.7 report the median number of retained measurements whose concordance correlation coefficient on the test period exceeds the corresponding threshold. “-” denotes not applicable (no recoverable measurement is identified on the training period, hence retention is undefined). All results use $\varepsilon _ { \mathrm { c u t o f f } } = 0 . 6 .$
<table><tr><td>Direction Method</td><td>#Ret</td><td>Ret% (mean±std)</td><td>CCC&gt;0.5</td><td> $\mathrm { C C C } { > } 0 . 6$ </td><td> $\mathrm { C C C } { > } 0 . 7$ </td></tr><tr><td colspan="6">EMBIC → NeuroBAT (target: NeuroBAT)</td></tr><tr><td>FRF</td><td>0</td><td> $0 . 0 \pm 0 . 0$ </td><td>0</td><td>0</td><td>0</td></tr><tr><td>IRT</td><td>0</td><td>一</td><td>0</td><td>0</td><td>0</td></tr><tr><td>XGB</td><td>0.5</td><td> $6 2 . 5 \pm 4 8 . 4$ </td><td>0.5</td><td>0</td><td>0</td></tr><tr><td>CMLP</td><td>1</td><td> $9 0 . 0 \pm 1 0 . 0$ </td><td>1</td><td>1</td><td>0</td></tr><tr><td>NICE</td><td>2</td><td> $4 9 . 2 \pm 1 0 . 5$ </td><td>2</td><td>1</td><td>0</td></tr><tr><td>Ours</td><td>3</td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td>3</td><td>1</td><td>0</td></tr><tr><td colspan="6">NeuroBAT → EMBIC (target: EMBIC)</td></tr><tr><td>FRF</td><td>5</td><td> $5 6 . 0 \pm 8 . 0$ </td><td>5</td><td>5</td><td>2</td></tr><tr><td>IRT</td><td>3</td><td> $4 2 . 9 \pm 0 . 0$ </td><td>3</td><td>3</td><td>2</td></tr><tr><td>XGB</td><td>6</td><td> $8 5 . 7 \pm 0 . 0$ </td><td>6</td><td>6</td><td>3</td></tr><tr><td>CMLP</td><td>6</td><td> $7 1 . 0 \pm 7 . 5$ </td><td>6</td><td>6</td><td>3</td></tr><tr><td>NICE</td><td>6</td><td> $7 7 . 4 \pm 9 . 9$ </td><td>6</td><td>6</td><td>3</td></tr><tr><td>Ours</td><td>7</td><td> $1 0 0 . 0 \pm 0 . 0$ </td><td>7</td><td>6.5</td><td>3</td></tr><tr><td colspan="6">CAS → SNAP (target: SNAP)</td></tr><tr><td>FRF</td><td>7</td><td> $7 1 . 4 \pm 2 3 . 9$ </td><td>7</td><td>7</td><td>5</td></tr><tr><td>IRT</td><td>5.5</td><td> $5 6 . 1 \pm 3 8 . 0$ </td><td>5.5</td><td>5.5</td><td>2.5</td></tr><tr><td>XGB</td><td>7</td><td> $7 1 . 4 \pm 1 3 . 3$ </td><td>7</td><td>7</td><td>6</td></tr><tr><td>CMLP</td><td>7</td><td> $7 4 . 0 \pm 1 7 . 9$ </td><td>7</td><td>7</td><td>5.5</td></tr><tr><td>NICE</td><td>9</td><td> $7 9 . 4 \pm 1 2 . 4$ </td><td>9</td><td>7.5</td><td>5.5</td></tr><tr><td>Ours</td><td>9</td><td> $9 3 . 8 \pm 9 . 0$ </td><td>9</td><td>7</td><td>5.5</td></tr><tr><td colspan="6">SNAP → CAS (target: CAS)</td></tr><tr><td>FRF</td><td>8</td><td> $6 4 . 5 \pm 3 0 . 9$ </td><td>8</td><td>8</td><td>5</td></tr><tr><td>IRT</td><td>6.5</td><td> $4 5 . 8 \pm 3 6 . 8$ </td><td>6.5</td><td>6</td><td>4.5</td></tr><tr><td>XGB</td><td>8</td><td> $6 3 . 5 \pm 3 6 . 8$ </td><td>8</td><td>8</td><td>5</td></tr><tr><td>CMLP</td><td>8.5</td><td> $7 3 . 0 \pm 1 7 . 4$ </td><td>8.5</td><td>8</td><td>4</td></tr><tr><td>NICE</td><td>10</td><td> $6 7 . 8 \pm 1 7 . 3$ </td><td>10</td><td>7</td><td>2.5</td></tr><tr><td>Ours</td><td>11</td><td> $8 6 . 3 \pm 1 4 . 9$ </td><td>11</td><td>6.5</td><td>4</td></tr></table>

Table 9: Ablation results on CAS (A) ↔ SNAP (B) under $\varepsilon _ { \mathrm { c u t o f f } } =$ 0.6. #Rec $_ A / \# \mathrm { R e c } _ { B }$ denote median numbers of recoverable measurements. $J _ { A } / J _ { B }$ denote Jaccard similarity across random seeds. Degenerate variants are marked with $^ { 6 6 } - ^ { 5 5 }$
<table><tr><td>Variant</td><td> $\# \mathrm { R e c } _ { A }$ </td><td> $J _ { A }$ </td><td> $\# \mathrm { R e c } _ { B }$ </td><td> $J _ { B }$ </td></tr><tr><td>Ours (full)</td><td>11</td><td>0.930</td><td>9</td><td>0.817</td></tr><tr><td>TWMLP backbone</td><td>11</td><td>0.929</td><td>9</td><td>0.947</td></tr><tr><td> $L _ { 0 }$  penalty</td><td>0†</td><td></td><td>0†</td><td></td></tr><tr><td>No gating</td><td>11</td><td>0.806</td><td>9</td><td>0.774</td></tr><tr><td>No hinge</td><td> $2 7 ^ { \dagger }$ </td><td></td><td>18†</td><td>-</td></tr></table>

† Trivial gating solutions. $L _ { 0 }$ penalty collapses to all-closed gating, while removing the hinge leads to all-open or unstable collapse depending on hyperparameter settings. Subset stability is not meaningful for these variants.

## Ethical Statement

This study uses de-identified datasets, including publicly available datasets and access-controlled datasets available upon reasonable request under appropriate approvals. No new data collection or human subject recruitment was conducted.

## Acknowledgments

Data collection and sharing for the Alzheimer’s Disease Neuroimaging Initiative (ADNI) is funded by the National Institute on Aging, National Institutes of Health Grant U19 AG024904. The grantee organization is the Northern California Institute for Research and Education.

## Contribution Statement

Yuanhao Zhuo and Zixi Qin contributed equally to this work.

## References

[American Psychiatric Association, 2022] American Psychiatric Association. Diagnostic and Statistical Manual of Mental Disorders. American Psychiatric Association, 2022.

[Andrew et al., 2013] Galen Andrew, Raman Arora, Jeff Bilmes, and Karen Livescu. Deep canonical correlation analysis. In International conference on machine learning, pages 1247–1255. PMLR, 2013.

[Bach and Jordan, 2005] Francis R Bach and Michael I Jordan. A probabilistic interpretation of canonical correlation analysis. 2005.

[Barnhart et al., 2007] Huiman X Barnhart, Michael J Haber, and Lawrence I Lin. An overview on assessing agreement with continuous measurements. Journal ofbiopharmaceutical statistics, 17(4):529–569, 2007.

[Beer et al., 2020] Joanne C Beer, Nicholas J Tustison, Philip A Cook, Christos Davatzikos, Yvette I Sheline, Russell T Shinohara, Kristin A Linn, Alzheimer’s Disease Neuroimaging Initiative, et al. Longitudinal combat: A method for harmonizing longitudinal multi-scanner imaging data. Neuroimage, 220:117129, 2020.

[Bengio et al., 2013] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation learning: A review and new perspectives. IEEE transactions on pattern analysis and machine intelligence, 35(8):1798–1828, 2013.

[Breiman et al., 2017] Leo Breiman, Jerome Friedman, Richard A Olshen, and Charles J Stone. Classification and regression trees. Chapman and Hall/CRC, 2017.

[Breiman, 2001] Leo Breiman. Random forests. Machine learning, 45(1):5–32, 2001.

[Cella et al., 2007] David Cella, Susan Yount, Nan Rothrock, Richard Gershon, Karon Cook, Bryce Reeve, Deborah Ader, James F Fries, Bonnie Bruce, Mattias Rose, et al. The patient-reported outcomes measurement information system (promis): progress of an nih roadmap cooperative group during its first two years. Medical care, 45(5):S3–S11, 2007.

[Chen, 2016] Tianqi Chen. Xgboost: A scalable tree boosting system. Cornell University, 2016.

[Crane et al., 2023] Paul K Crane, Seo-Eun Choi, Michael Lee, Phoebe Scollard, R Elizabeth Sanders, Brandon Klinedinst, Connie Nakano, Emily H Trittschuh, Jesse Mez, Andrew J Saykin, et al. Measurement precision across cognitive domains in the alzheimer’s disease neuroimaging initiative (adni) data set. Neuropsychology, 37(4):373, 2023.

[Dewey et al., 2019] Blake E Dewey, Can Zhao, Jacob C Reinhold, Aaron Carass, Kathryn C Fitzgerald, Elias S Sotirchos, Shiv Saidha, Jiwon Oh, Dzung L Pham, Peter A Calabresi, et al. Deepharmony: A deep learning approach to contrast harmonization across scanner changes. Magnetic resonance imaging, 64:160–170, 2019.

[Dinh et al., 2014] Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516, 2014.

[Dinh et al., 2016] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. arXiv preprint arXiv:1605.08803, 2016.

[Dinsdale et al., 2021] Nicola K Dinsdale, Emma Bluemke, Stephen M Smith, Zobair Arya, Diego Vidaurre, Mark Jenkinson, and Ana IL Namburete. Learning patterns of the ageing brain in mri using deep convolutional networks. NeuroImage, 224:117401, 2021.

[Dorans, 2007] Neil J Dorans. Linking scores from multiple health outcome instruments. Quality of Life Research, 16(Suppl 1):85–94, 2007.

[Fortin et al., 2017] Jean-Philippe Fortin, Drew Parker, Birkan Tunc¸, Takanori Watanabe, Mark A Elliott, Kosha Ruparel, David R Roalf, Theodore D Satterthwaite, Ruben C Gur, Raquel E Gur, et al. Harmonization of multi-site diffusion tensor imaging data. Neuroimage, 161:149–170, 2017.

[Hammarstrom¨ et al., 2016] Anne Hammarstrom, Hugo¨ Westerlund, Kaisa Kirves, Karina Nygren, Pekka Virtanen, and Bruno Haggl ¨ of. Addressing challenges ¨ of validity and internal consistency of mental health measures in a 27-year longitudinal cohort study–the northern swedish cohort study. BMC medical research methodology, 16(1):4, 2016.

[Hinton and Salakhutdinov, 2006] Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. science, 313(5786):504–507, 2006.

[Hotelling, 1935] Harold Hotelling. The most predictable criterion. Journal of educational Psychology, 26(2):139, 1935.

[Huang et al., 2023] Xia Huang, Hui-Qin Li, Alan Simpson, Jia-Jun Xu, Wan-Jie Tang, and Yuan-Yuan Li. Differences among fathers, mothers, and teachers in symptom assessment of adhd patients. Frontiers in Psychiatry, 14:1029672, 2023.

[Johnson et al., 2007] W Evan Johnson, Cheng Li, and Ariel Rabinovic. Adjusting batch effects in microarray expression data using empirical bayes methods. Biostatistics, 8(1):118–127, 2007.

[Kingma and Dhariwal, 2018] Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions. Advances in neural information processing systems, 31, 2018.

[Kingma and Welling, 2013] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.

[Kolen and Brennan, 2013] Michael J Kolen and Robert L Brennan. Test equating: Methods and practices. Springer Science & Business Media, 2013.

[Lawrence and Lin, 1989] I Lawrence and Kuei Lin. A concordance correlation coefficient to evaluate reproducibility. Biometrics, pages 255–268, 1989.

[Lawrence et al., 2017] Emma Lawrence, Carolin Vegvari, Alison Ower, Christoforos Hadjichrysanthou, Frank De Wolf, and Roy M Anderson. A systematic review of longitudinal studies which measure alzheimer’s disease biomarkers. Journal ofAlzheimer’s Disease, 59(4):1359– 1379, 2017.

[Little and Rubin, 2019] Roderick JA Little and Donald B Rubin. Statistical analysis with missing data. John Wiley & Sons, 2019.

[Lord, 2012] Frederic M Lord. Applications of item response theory to practical testing problems. Routledge, 2012.

[Louizos et al., 2017] Christos Louizos, Max Welling, and Diederik P Kingma. Learning sparse neural networks through l 0 regularization. arXiv preprint arXiv:1712.01312, 2017.

[Lui et al., 2025] Forshing Lui, Ischel Gonzalez Kelso, and Marjorie Launico. Cognitive assessment. StatPearls, 2025.

[Molnar, 2020] Christoph Molnar. Interpretable machine learning. Lulu. com, 2020.

[Mueller et al., 2005] Susanne G Mueller, Michael W Weiner, Leon J Thal, Ronald C Petersen, Clifford Jack, William Jagust, John Q Trojanowski, Arthur W Toga, and Laurel Beckett. The alzheimer’s disease neuroimaging initiative. Neuroimaging Clinics, 15(4):869–877, 2005.

[Ng, 2004] Andrew Y Ng. Feature selection, l 1 vs. l 2 regularization, and rotational invariance. In Proceedings ofthe twenty-first international conference on Machine learning, page 78, 2004.

[Riley et al., 2010] Richard D Riley, Paul C Lambert, and Ghada Abo-Zaid. Meta-analysis of individual participant data: rationale, conduct, and reporting. Bmj, 340, 2010.

[Rubin, 1976] Donald B Rubin. Inference and missing data. Biometrika, 63(3):581–592, 1976.

[Tibshirani, 1996] Robert Tibshirani. Regression shrinkage and selection via the lasso. Journal ofthe Royal Statistical

Society Series B: Statistical Methodology, 58(1):267–288, 1996.

[Villalpando et al., 2025] Juan Manuel Villalpando, Bernard-Simon Leclerc, Minh Tri Le, Carol Hudon, Aline Bolduc, Marie-Jeanne Kergoat, and Consortium for the Early Identification of Alzheimer’s Disease-Quebec (CIMA-Q). A comparison of clinical diagnostic classification criteria used in longitudinal cohort studies of the alzheimer’s disease continuum: A systematic review. Neuropsychology Review, pages 1–22, 2025.

[Willcutt et al., 2005] Erik G Willcutt, Alysa E Doyle, Joel T Nigg, Stephen V Faraone, and Bruce F Pennington. Validity of the executive function theory of attentiondeficit/hyperactivity disorder: a meta-analytic review. Biological psychiatry, 57(11):1336–1346, 2005.

[Yamada et al., 2020] Yutaro Yamada, Ofir Lindenbaum, Sahand Negahban, and Yuval Kluger. Feature selection using stochastic gates. In International conference on machine learning, pages 10648–10659. PMLR, 2020.

[Yang et al., 2019] Xinghao Yang, Weifeng Liu, Wei Liu, and Dacheng Tao. A survey on canonical correlation analysis. IEEE Transactions on Knowledge and Data Engineering, 33(6):2349–2368, 2019.

[Zhu et al., 2017] Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A Efros. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings ofthe IEEE international conference on computer vision, pages 2223–2232, 2017.

[Zhuo et al., 2026] Yuanhao Zhuo, Zixi Qin, Ling Qin, and Wanqing Li. BFHD: Bidirectional feature harmonization decomposition for heterogeneous clinical assessments. In Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence, IJCAI-26, pages 7046– 7054. International Joint Conferences on Artificial Intelligence Organization, 2026.