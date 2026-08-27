# A meta-algorithm for ab initio reconstruction of complex mixtures in cryo-EM

Alkin Kaz Princeton University akaz@princeton.edu

Arda Kaz University of California, Berkeley ardakaz@berkeley.edu

Ellen D. Zhong Princeton University zhonge@princeton.edu

## Abstract

We describe a systematic approach for spawning and aggregating multi-class cryo-EM reconstruction jobs. This approach formalizes standard ad hoc strategies of iterative classification and filtering typically used by practitioners to sort impure, heterogeneous samples. To our knowledge, this is the first method that can successfully perform ab initio reconstruction on datasets containing dozens of distinct species. We obtain 97% accuracy on ab initio reconstruction of a 45-class subset of Tomotwin-100, 75% accuracy on the full Tomotwin-100 dataset, and demonstrate recovery of ribosomal assembly states from an unfiltered experimental cryo-EM dataset. Our approach's capability scales with compute and lays the foundation for automated cryo-EM workflows in modern experimental settings.

## 1 Introduction

Cryo-electron microscopy (cryo-EM) is a central tool for determining the 3D structures of biological molecules at near-atomic resolution. In a typical single particle cryo-EM experiment, an aqueous solution of molecules is vitrified, imaged as randomly-oriented 2D projections by a transmission electron microscope, and computationally reconstructed into 3D density volumes. A major computational challenge is ab initio reconstruction, i.e., recovering 3D structure(s) without any image poses or initial volumes. Over the past decade, methods and software for ab initio 3D reconstruction and refinement have matured, enabling routine structure determination for well-behaved, structurally homogeneous (i.e. rigid) biomolecules [1, 2, 3, 4]. Even for more dynamic biomolecular complexes such as macromolecular machines, standard pipelines are often sufficient to align particle images into a common reference frame, thus enabling downstream advanced heterogeneity analysis tools to recover rich conformational ensembles [5, 6, 7, 8].

Modern cryo-EM experiments, however, increasingly image less-purified samples containing a mixture of distinct molecular species [9, 10]. When multiple species are present, there is no common reference frame, and image alignment can no longer be decoupled from resolving the heterogeneity. Poses, per-particle class assignments, and multiple volumes must instead be inferred jointly, a substantially harder optimization. Multiclass ab initio reconstruction is the standard approach to this problem, partitioning particle images into a fixed number of classes and reconstructing one volume per class [2]. Recent advances, including neural field approaches such as cryoDRGN-AI [5] and neural mixture models such as Hydra [11], have proposed more expressive models of heterogeneity that are jointly optimized with image poses for heterogeneous ab initio reconstruction. Despite this progress, current methods fail when the number of distinct species grows large, as this poses a much more challenging optimization problem [12]. In practice, practitioners compensate with ad hoc strategies running reconstructions repeatedly with different random seeds, manually inspecting and filtering the results, and iterating until satisfactory classes emerge. These workflows are labor-intensive, difficult to reproduce, and scale poorly with mixture complexity.

Here, we observe that multiclass ab initio reconstruction jobs, despite failing to resolve a complex mixture in full, carry weak classification signal. This is analogous to the classical weak learning setting [13], where learners that perform only slightly better than chance can be aggregated into a strong classifier. Building on this connection and related work on cluster ensembles [14, 15], we introduce reconstruction ensembles, a meta-algorithm for systematically generating, weighting, and aggregating ab initio reconstruction jobs to resolve complex mixtures that no single run can handle. Using cryoSPARC's multiclass reconstruction methods [2] as weak classifiers, we demonstrate state-of-the-art ab initio reconstruction of synthetic and real datasets, including successful recovery of complex mixtures from datasets containing more than tens of species for the first time. The approach is tool-agnostic, requires no modifications to existing reconstruction software, and lays the groundwork for reproducible, automated cryo-EM workflows. In summary, we make the following contributions:

• A meta-algorithm for spawning and aggregating ab initio reconstruction algorithms to produce final particle classification assignments,

• State-of-the-art ab initio reconstruction of datasets containing complex mixtures, achieving 97% classification accuracy of a 45-class subset of Tomotwin-100 and 75% accuracy on the full Tomotwin-100 dataset [12],

• A workflow automation framework built on the cryoSPARC-tools API that enables reproducible, programmatic control of cryo-EM reconstruction pipelines.

## 2 Background and related work

## 2.1 Single particle cryo-EM reconstruction

In single particle cryo-EM, an aqeuous solution of molecules is vitrified and imaged in a transmission electron microscope, producing $\mathrm { 1 0 ^ { 4 } { - } 1 0 ^ { 7 } }$ noisy 2D projection images. Each image $X _ { i }$ captures a separate copy of the molecule at an unknown orientation, and can be modeled as [11, 16]:

$$
X _ { i } = C _ { i } \cdot \mathcal { P } _ { \phi _ { i } } V _ { i } + \eta _ { i }\tag{1}
$$

where $V _ { i } : \mathbb { R } ^ { 3 }  \mathbb { R }$ is the 3D electron scattering potential (density map), $\phi _ { i } = ( R _ { i } , \mathbf { t } _ { i } )$ is the unknown pose consisting of a rotation $R _ { i } \in S O ( 3 )$ and in-plane translation $\mathbf { t } _ { i } \in \mathbb { R } ^ { 2 } , \mathcal { P } _ { \phi _ { i } }$ is the projection operator that integrates $V _ { i }$ along the imaging axis at orientation $\phi _ { i } , C _ { i }$ is the contrast transfer function (CTF) of the microscope, and $\eta _ { i } \stackrel { \smile } { \sim } \bar { \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ is additive noise. The images are recorded on a discrete $D \times D$ grid with typical signal-to-noise ratios between $1 0 ^ { - 1 }$ and $1 0 ^ { - 2 }$

The central computational task is to recover V from these projections, which requires jointly solving for the unknown volume and the per-image poses. Expectation-maximization and coordinate ascent algorithms are typically used to find a maximum a posteriori estimate of $V$ , marginalizing over the pose posterior for each image [1]. This iterative refinement is sensitive to the initial volume estimate; stochastic gradient descent is commonly used for ab initio reconstruction to provide a starting model from scratch [2].

## 2.2 Multiclass ab initio reconstruction

Real cryo-EM samples often contain mixtures of distinct molecular species or conformational states. The standard approach to this heterogeneity problem extends the image formation model to assume that images are generated from K independent volumes $V _ { 1 } , \dots , V _ { K }$ . Inference then requires marginalization over both the per-image poses and class assignment probabilities $\pi _ { j } { \mathrm { : } }$

$$
\arg \operatorname* { m a x } _ { V _ { 1 } , . . . , V _ { K } } \sum _ { i = 1 } ^ { N } \log \sum _ { j = 1 } ^ { K } \left[ \pi _ { j } \int p ( X _ { i } \mid \phi , V _ { j } ) p ( \phi ) d \phi \right] + \sum _ { j = 1 } ^ { K } \log p ( V _ { j } )\tag{2}
$$

This formulation, commonly called multiclass refinement or 3D classification, is implemented in major software packages such as RELION [1] and cryoSPARC [2]. In the ab initio setting, the algorithm must simultaneously infer both the image class assignments and poses from scratch (i.e. random initialization), making the optimization landscape considerably more challenging than the refinement or classification case where initial volume models or fixed poses are provided, respectively.

A key limitation of multiclass ab initio reconstruction is that the number of classes K must be specified in advance and is typically kept small $( \mathbf { e . g . } , K \leq 1 0 )$ . When the true number of species M exceeds K, the algorithm produces a coarse partition that merges distinct species into shared classes. When K is set too large, the optimization frequently fails to converge or produces degenerate solutions. In practice, this means that for complex mixtures containing tens of distinct species, no single ab initio run can resolve the full composition of the sample.

## 2.3 Heterogeneity analysis beyond discrete classes

A substantial body of work has been devoted to expanding the types of heterogeneity that cryo-EM methods can capture [8, 12]. While early methods modeled heterogeneity as a discrete mixture of independent, homogeneous volumes [1, 2], more recent approaches have moved toward continuous representations of structural variability, including linear subspaces [6], neural fields [5, 17], Gaussian mixture models [18], and parametric deformation fields [19]. However, the majority of these methods operate with fixed poses obtained from an upstream consensus reconstruction, limiting their applicability when the sample contains strong compositional heterogeneity that prevents reliable pose estimation in the first place.

In the ab initio setting where poses must be inferred jointly with the heterogeneity, fewer methods are available. The multiclass ab initio capabilities in RELION and cryoSPARC (Eq. 2) model discrete mixtures but are limited to small K. CryoDRGN-AI [20] demonstrated a fully autodecoding framework for ab initio heterogeneous reconstruction, and Hydra [11] extended this with a mixture of K neural fields to jointly capture compositional and conformational heterogeneity. Stepping beyond the capabilities of a single call to these methods, Lauzirika et al. [21] proposed a statistical framework for exploring the number of discrete classes using assignment consistencies across replicate runs, an intuition closely related to ours, which we extend through a full ensemble aggregation framework. Despite this progress, resolving complex mixtures containing tens or more distinct species at the ab initio stage remains an open challenge, with no existing methods, to our knowledge, able to recover any of the underlying structures from the Tomotwin-100 benchmark dataset [12].

## 2.4 Workflow automation in cryo-EM

The cryo-EM data processing pipeline has historically been driven by expert users interacting with graphical interfaces. Recently, several efforts have begun to automate portions of this workflow. The cryoSPARC team demonstrated an end-to-end Workflow feature [22], and the CryoWizard pipeline [23] automates particle curation using the Cryo-IEF model. CryoSift [24] provides another example of programmatic interaction with the cryoSPARC suite. Our ensemble framework builds directly on this trend, using the cryoSPARC-tools API to orchestrate the generation and aggregation of reconstruction jobs without manual intervention.

## 2.5 Weak learning and cluster ensembles

A central result in machine learning is that weak learners, i.e. classifiers performing only slightly better than random chance, can be aggregated into strong learners with substantially better performance guarantees [13]. This observation has driven decades of research on boosting and ensemble methods, producing canonical algorithms such as AdaBoost [25] and XGBoost [26].

In the unsupervised setting, where labeled feedback is unavailable, a related subfield of cluster ensembles has emerged [14]. Given multiple clustering assignments of the same dataset, the goal is to aggregate them into a single, stronger partition. A canonical approach due to Fred and Jain [15] constructs a co-assignment matrix recording how often pairs of data points are placed in the same cluster across ensemble members, then applies hierarchical agglomerative clustering to this accumulated evidence. Alternative formulations include graph partitioning [27] and information-theoretic objectives [14]. Our weighted similarity score builds directly on the co-assignment matrix framework of [15], extending it with coverage-aware masking and interpretable weight heuristics tailored to the structure of cryo-EM reconstruction outputs.

## 3 Methods

Our algorithm proceeds in three stages: ensemble generation, weighted similarity scoring, and clustering. We first formalize the tool abstraction and the two primitives for building ensembles (Fig. 1), then describe the aggregation framework that produces the final particle assignments (Fig. 2).

## 3.1 Notation and tool abstraction

We assume access to a stochastic multiclass ab initio reconstruction tool that takes a set of N particle images and assigns each particle to one of K classes. Let $I = \{ 0 , \ldots , N - 1 \}$ denote the particle index set.

We index the jobs in the ensemble with a single flat counter: $J _ { 0 } , J _ { 1 } , \dots , J _ { T - 1 }$ for $T$ total jobs. Each job $J _ { m }$ is characterized by its input particle subset $I _ { m } \subseteq I$ , its class count $K _ { m } ,$ and its random seed $\rho _ { m }$ . Annotating the set of possible class labels for a particle to be $[ K _ { m } ] = \{ 0 , 1 , \ldots , K _ { m } - 1 \}$ , the tool returns a label vector $l _ { m }$ over the input particles:

$$
J _ { m } ( I _ { m } ; K _ { m } , \rho _ { m } ) = \mathtt { A b I n i t i o } ( I _ { m } ; K _ { m } , \rho _ { m } ) = l _ { m } \in [ K _ { m } ] ^ { | I _ { m } | }\tag{3}
$$

The label vector partitions $I _ { m }$ into disjoint subsets $I _ { m } ^ { k } = \{ i \in I _ { m } \mid l _ { m } ( i ) = k \}$ for $k \in [ K _ { m } ]$ , which we call the children of $J _ { m }$ . If the tool produces soft class probabilities rather than hard assignments, we take the argmax as the label. Any other tool-specific parameter is held fixed across the ensemble.

The particles may originate from $M \geq 1$ true underlying species. Our only assumption is that the tool assigns particles better than uniform random, regardless of the relationship between $K _ { m }$ and M. This is sufficient for the tool to serve as a weak classifier.

## 3.1.1 Ensemble generation

The ensemble $\mathcal { I } ~ = ~ \{ J _ { 0 } , \ldots , J _ { T - 1 } \}$ is built from two primitives, visually summarized in Figure 1.

![](images/c67f1fa720a940709b1cbbf1b1ac997e4ed7747e64610ae92b80af88b1432c76.jpg)

Stochastic repetition. Given a particle subset $I _ { m }$ and a class count $K _ { m }$ , this primitive launches R replicate jobs with identical parameters while varying the random seed $\rho _ { m }$ . Cross-replicate consistency in assignments serves as a proxy for classification confidence.

Figure 1: Overview of ab initio ensemble generation. Along the horizontal axis, stochastic repetition produces replicate jobs that share the same input subset $I _ { m }$ and class count $K _ { m }$ but differ in random seed $\rho _ { m } .$ Along the vertical axis, hierarchical exploration launches new jobs on the children $I _ { m } ^ { k }$ of a parent job.

Hierarchical exploration. Given a completed job $J _ { m } .$ this primitive launches a new job on each of its children

$I _ { m } ^ { 0 } , \ldots , \bar { I } _ { m } ^ { K _ { m } - 1 }$ , with a possibly different class count $K ^ { \prime }$ . This divide-and-conquer strategy enables the ensemble to probe a number of true classes that grows exponentially in the tree depth.

These two primitives can be composed freely. The similarity score aggregation (described below) accommodates any job tree or forest, so no particular generation strategy is required.

## 3.1.2 Weighted similarity score

Given the completed ensemble $\mathcal { I } = \{ J _ { 0 } , \ldots , J _ { T - 1 } \}$ , we aggregate per-job class assignments into a single inter-particle similarity matrix S (Fig. 2). This proceeds in two stages: computing per-job co-assignment matrices, then taking their weighted average.

For each job $J _ { m }$ with input subset $I _ { m } \subseteq I ,$ the co-assignment matrix $C ^ { m } \in \{ 0 , 1 \} ^ { N \times N }$ records whether two particles $( i , { \bar { j } } )$ received the same class label, $\mathsf { \bar { i . e . } } C _ { i j } ^ { m } = \mathbb { I } \{ l _ { m } ( i ) = l _ { m } ^ { \setminus } ( j ) \}$ . Furthermore, a coverage mask $\Omega ^ { m }$ tracks which particle pairs were present in the job, masking out any pair that was not present in both, i.e. $\Omega _ { i j } ^ { m } = \mathring { \mathbb { I } } \big \{ i \in I _ { m } \big \} \cdot \mathbb { I } \big \{ j \in I _ { m } \big \}$

![](images/e647167fcf626b74cc13f136af4242650fa79ee7741fd32c1e4f23dff7b3e97b.jpg)  
Figure 2: Overview of the job aggregation framework that produces final particle assignments. For every job $J _ { m }$ in the ensemble $\mathcal { I }$ , the particle co-assignment matrix is calculated, then averaged with weights $w _ { m }$ to yield a similarity score matrix. The similarity score matrix is fed to any off-the-shelf clustering algorithm that accepts pairwise distance inputs to yield final class assignments.

The similarity between particles i and $j$ is then the weighted mean of their co-assignment values across all jobs in the ensemble that covered both, i.e. their coverage-aware weighted average:

$$
S _ { i j } : = \frac { \sum _ { m = 0 } ^ { T - 1 } w _ { m } \ : \Omega _ { i j } ^ { m } \ : C _ { i j } ^ { m } } { \sum _ { m = 0 } ^ { T - 1 } w _ { m } \ : \Omega _ { i j } ^ { m } } \in [ 0 , 1 ]\tag{4}
$$

The weights in this aggregation routine can be supplied by the expert user if desired. However, instead of requiring the user to manually inspect and assign per-job weights, we also provide weight heuristics to automatically compute $w _ { m }$ as a product of three interpretable penalty factors: (1) jobs that underuse their allotted classes $( w _ { m } ^ { \mathrm { e f f } } ) , ( 2 )$ jobs that fragment particles into tiny clusters $( w _ { m } ^ { \mathrm { t i n y } } )$ and (3) jobs that funnel most particles into a single dominant class $\mathbf { \Delta } ^ { \prime } w _ { m } ^ { \mathrm { s i n k } } )$ . All three are derived from the class proportions of job $J _ { m }$ . We write $p _ { m } ^ { k } = n _ { m } ^ { k } / N _ { m }$ for the fraction of the $N _ { m } = | I _ { m } |$ input particles assigned to class $k ,$ where $\textstyle \sum _ { k } n _ { m } ^ { k } = N _ { m }$

$$
w _ { m } : = w _ { m } ^ { \mathrm { e f f } } \cdot w _ { m } ^ { \mathrm { t i n y } } \cdot w _ { m } ^ { \mathrm { s i n k } }\tag{5}
$$

The effective-class weight $w _ { m } ^ { \mathrm { e f f } }$ penalizes jobs that concentrate particles into fewer classes than requested. We measure the effective number of occupied classes via the inverse Simpson index [28] (equivalently, the inverse Herfindahl-Hirschman index $[ 2 9 ] ) , K _ { m } ^ { \mathrm { e f f } }$ . The penalty is Gaussian in logspace, so that $w _ { m } ^ { \mathrm { e f f } } = 1$ when classes are perfectly balanced $( K _ { m } ^ { \mathrm { e f f } ^ { \prime \prime } } = K _ { m } )$ and decays smoothly with bandwidth $\sigma$ as the distribution becomes more skewed:

$$
w _ { m } ^ { \mathrm { e f f } } = \exp \left( - \frac { ( \log K _ { m } ^ { \mathrm { e f f } } - \log K _ { m } ) ^ { 2 } } { 2 \sigma ^ { 2 } } \right) , \quad \mathrm { h e r e } \quad K _ { m } ^ { \mathrm { e f f } } = \frac { 1 } { \sum _ { k } ( p _ { m } ^ { k } ) ^ { 2 } }\tag{6}
$$

The tiny-class weight $w _ { m } ^ { \mathrm { t i n y } }$ penalizes jobs that produce classes with less than $p ^ { \mathrm { m i n } }$ fraction of particles, which are too small for reliable downstream reconstruction. The total probability mass in such classes is $\begin{array} { r } { t _ { m } = \sum _ { k } \mathbb { I } \{ p _ { m } ^ { k } < p ^ { \mathrm { m i n } } \} ~ p _ { m } ^ { k } } \end{array}$ , and the penalty decays exponentially on $t _ { m }$ with rate $\gamma .$ The sink-class weight $w _ { m } ^ { \mathrm { s i n k } }$ penalizes jobs in which one dominant class absorbs most particles, since such a job provides little discriminative signal to the ensemble. Writing $p _ { m } ^ { \operatorname* { m a x } } = \operatorname* { m a x } _ { k } p _ { m } ^ { k }$ , the penalty decays exponentially on $p _ { m } ^ { \mathrm { m a x } }$ with rate $\beta$

$$
w _ { m } ^ { \mathrm { t i n y } } = \exp \left( - \gamma t _ { m } \right) , \quad w _ { m } ^ { \mathrm { s i n k } } = \exp \bigl ( - \beta p _ { m } ^ { \mathrm { m a x } } \bigr )\tag{7}
$$

Alternative functional forms (e.g., hard thresholds) are compatible with the framework, where custom weight heuristics could be engineered at will. For our purposes, the exponential form sufficed while providing simplicity in interpretation.

## 3.1.3 Clustering

Once S is computed, any clustering algorithm that accepts pairwise similarities (or distances via 1 — S) can be applied. We use agglomerative clustering with average linkage, as it allows straightforward exploration of how assignments change with the number of clusters.

A practical issue arises from the dense structure of $S { \mathrm { : } }$ with average linkage, large groups of weakly similar particles can dominate the merge criterion over smaller groups of strongly similar particles, causing the algorithm to absorb rare species into majority clusters prematurely. To address this, we apply a sparsification step before clustering. For each row of S, we retain only the top $n _ { \mathrm { n b r } }$ strongest similarity scores and zero out the rest, effectively constructing a nearest-neighbor graph. We also zero out the diagonal (trivial self-similarity) and symmetrize the result by taking $S _ { i j } \gets \operatorname* { m a x } ( S _ { i j } , S _ { j i } )$ The final cluster assignments are obtained by cutting the agglomerative dendrogram at the desired number of classes.

## 4 Results

We first characterize the performance of the individual ab initio reconstruction jobs on datasets of increasing heterogeneity. We then evaluate our ensembling approach on ab initio reconstruction of synthetic datasets containing up to 100 distinct biomolecular complexes as well as an experimental cryo-EM dataset. Finally, we analyze how performance scales as a function of ensemble growth and perform hyperparameter ablations of the aggregation step.

## 4.1 The ab initio reconstruction tool

Although our approach is tool-agnostic for any multiclass reconstruction algorithm, we use the “Ab-Initio Reconstruction" job in cryoSPARC software suite [2], which provides programmatic access and job management through a Python API exposed in the cryosparc-tools library.

We use only the default parameters in version 4.7.1, except the number of classes K and the random seed $\rho .$ Given the extensive tunability of this job, we note that our results could be likely be further improved by optimizing the parameters, e.g. the Fourier radius step in smaller molecules [30]. Here, we focus on highlighting the capability of ensembling even with default job settings.

## 4.1.1 Datasets

The Tomotwin-100 dataset, proposed by the CryoBench benchmark suite [12], consists of 100 structures of various molecular weights and symmetries, with 1,000 images each $\left( D \ = \ 1 2 8 \right)$ uniformly distributed across the viewing sphere. We focus our analysis on this dataset, since to the best of our knowledge, the complete ab initio reconstruction of this dataset has not been reported yet, where the current state-of-the-art analysis provided in Hydra [11] only focused on a 3-class subset of this 100-class dataset.

We also analyze a 45-class subset of the asymmetric complexes in Tomotwin-100, which we call Tomotwin-45-asym. Evaluating pose accuracy on the full dataset is complicated by symmetry: when the underlying structure possesses rotational symmetry, an image can be explained by multiple equivalent poses, making per-image pose error ill-defined [11, 31]. To avoid complexity from symmetry-aware pose error functions [31], we restrict our pose evaluation to structures with C1 symmetry in Tomotwin-100, yielding 45 classes and N = 45, 000 particles.

To test our method beyond synthetic data, we use EMPIAR-10076 [32], a standard benchmark for heterogeneous reconstruction of compositional heterogeneity [32, 5] consisting of E. coli large ribosomal subunit assembly intermediates (N = 131, 899 images, $D = 3 2 0 )$ . The dataset contains four major structural states, two rare species, and a substantial junk fraction. We process the full unfiltered stack in our reconstruction.

To compute classification accuracy, in order to assess the correct permutation of labels, we follow a simple majority vote within every cluster to determine which ground truth label it maps to. In pose accuracy measurements, the final class assignments are used to generate per-class homogeneous reconstructions, which are then aligned with the mapped ground truth poses with a global rigid-body alignment.

## 4.2 Performance of the one-shot ab initio jobs

Failed  
![](images/a402d7758c0f2d276dca39754549311a93a27b5d310aa6c81ff04a9c836894c7.jpg)

![](images/e1a04cef5b6ecc534e2f5b735bbdb3262f38ccf58574b607e3e1f7972f5b5ec0.jpg)  
Figure 3: Median angular error (left) and classification accuracy (right) of single-shot ab initio jobs on subsets of Tomotwin-45-asym with increasing heterogeneity.

We first run the cryoSPARC [2] “Ab-Initio Reconstruction" job on subsets of Tomotwin-45-asym with increasing number of ground truth classes $( M = 1 , 2 , \dot { \dots } , 4 5 )$ and varying the class number parameter K. This sweep serves two purposes: it establishes a baseline for single-shot reconstruction performance, and it validates the coarse clustering assumption $( K < M )$ that our ensemble design relies on. The median angular pose error and classification accuracy are reported in Figure 3, with the complete suite of metrics in Appendix A.

On the diagonal $( K = M )$ , the job achieves near-perfect classification and pose estimation up to $M = 3 2 .$ , with median pose errors of 2–3 degrees. As the ground truth complexity grows relative to the job's class budget (moving rightward or downward from the diagonal) both accuracy and pose quality degrade gracefully. On the full dataset $( M = 4 5 )$ , the job fails to converge at large K values $( K \geq 3 2 )$ but still converges for coarser settings $( K \le 1 6 )$ , confirming that it can serve as a weak classifier even when it cannot resolve the full mixture.

## 4.3 Performance of the ab initio ensemble

Table 1: Heterogeneous ab initio reconstruction performance measured by classification, clustering, and pose metrics for the Tomotwin benchmark datasets. For the baseline one-shot strategy, the best performing jobs are shown, i.e. $K = 1 6$ for both datasets.

<table><tr><td>DATASET</td><td>STRATEGY</td><td colspan="3">CLASSIFICATION (%)</td><td colspan="2">POSE ERROR (deg)</td></tr><tr><td></td><td></td><td>ACCURACY</td><td>PRECISION</td><td>MACRO F1</td><td>MEAN</td><td>MEDIAN</td></tr><tr><td> $\mathrm { T T } { - } 4 5 { \mathrm { - } } \mathrm { A S Y M }$ </td><td>CRYODRGN2</td><td>13.1</td><td>10.1</td><td>10.9</td><td>124.6</td><td>128.3</td></tr><tr><td></td><td>CRYODRGN-AI</td><td>9.3</td><td>6.8</td><td>7.4</td><td>124.2</td><td>129.3</td></tr><tr><td></td><td>HYDRA  $( K = 4 )$ </td><td>4.4</td><td>0.4</td><td>0.7</td><td>124.4</td><td>129.4</td></tr><tr><td></td><td>ONE-SHOT</td><td>34.6</td><td>25.4</td><td>22.7</td><td>78.3</td><td>74.1</td></tr><tr><td></td><td>ENSEMBLE</td><td>97.4</td><td>97.4</td><td>97.4</td><td>6.11</td><td>1.71</td></tr><tr><td>TT-100</td><td>ONE-SHOT</td><td>15.2</td><td>3.5</td><td>5.5</td><td></td><td>1</td></tr><tr><td></td><td>ENSEMBLE</td><td>75.0</td><td>71.3</td><td>71.9</td><td>一</td><td>1</td></tr></table>

We evaluate our ensemble method on two datasets of increasing difficulty: the 45-class Tomotwin-45-asym and the full 100-class Tomotwin-100. For both datasets, the ensemble was generated using $K = \mathsf { \bar { \{ 4 , 8 , 1 6 \} } }$ job trees at differing 2 — 3 levels of hierarchical depth with $3 - 5$ replicates per level (Appendix B). For Tomotwin-45-asym, its ensemble totaled 257 jobs and 332 H100-hours with a critical path of 4.3 wall-hours; while for Tomotwin-100, its ensemble totaled 235 jobs and 320 hours with a critical path of 4.3 wall-hours. In both cases, default parameters were used for all jobs. Further experimental details are provided in Appendix B.

We compare against four baselines: the best single-shot cryoSPARC run (K = 16 for both datasets), and three heterogeneous reconstruction methods: cryoDRGN2 [17], cryoDRGN-AI [20], and Hydra $( K = 4 )$ [11]. These three methods were run with default settings (Appendix B). Because these methods were designed for smaller mixtures, we include them primarily to illustrate the difficulty of the 45-class setting rather than as matched comparisons. Results are summarized in Table 1.

On Tomotwin-45-asym, the ensemble achieves 97.4% classification accuracy and a median pose error of 1.71°, compared to 34.6% accuracy and $7 4 . 1 ^ { \circ }$ median pose error for the best single-shot run. The three neural-field baselines produce pose errors near the uniform-random baseline of ${ \sim } 1 3 0 ^ { \circ }$ , indicating that these methods cannot resolve the mixture at this complexity. The ensemble method fully reconstructs this 45-class mixture, as seen from the volume renderings in Appendix C. Furthermore, on Tomotwin-100, the ensemble reaches 75.0% accuracy – a five-fold improvement over the 15.2% one-shot baseline. To our knowledge, neither the 45-class nor the 100-class Tomotwin benchmarks have been previously resolved at the ab initio stage, the prior state-of-the-art analysis in Hydra [11] was limited to a 3-class subset. Our results therefore represent an order-of-magnitude expansion in the mixture complexity that ab initio methods can handle.

## 4.4 Scaling analysis in ensemble growth

To understand how classification accuracy scales with compute, we systematically vary both the depth of the job tree (hierarchical exploration) and the number of random replicates (stochastic repetition), sweeping over $K \in \{ 4 , \dot { 8 } , 1 6 \}$ . To simplify the configuration space, each ensemble uses a single K value across all jobs; the reported accuracies are therefore lower bounds on what mixed-K ensembles could achieve. Figure 4 plots the resulting accuracy-compute trade-off surfaces for Tomotwin-45-asym and Tomotwin-100, with further details in Appendix D.

On both datasets, accuracy typically improves with the first few levels of hierarchical depth and with additional replicates. Because replicates within a level run in parallel, the wall-clock time scales with tree depth rather than total job count, growing only logarithmically in the ensemble size. The “level" annotations in Figure 4 indicate the depth of the deepest job tree in each ensemble.

![](images/01dcb939973302a3320e5d6a192775ffdec56ea5f172072ac373e3baefc42c5f.jpg)

![](images/fc87e55d628fe5f10aa1b1929298348872f1c4133ec2c77defdedc90d0661aca.jpg)  
Figure 4: Utility-cost frontiers of scaling job ensembling for Tomotwin-45-asym (left) and Tomotwin-100 (right). The vertical (utility) axis is classification accuracy, and the horizontal (cost) axis is the number of H100 GPU-hours used. Each curve shows ensembling up to 5 replicates and different curves indicate different hierachical levels with a fixed class K.

## 4.5 Real dataset

We apply our ensemble method to EMPIAR-10076 [32], processing the full unfiltered particle stack $( N = 1 3 1$ ,899) without removing junk particles. The ensemble was generated using 3 replicates of $\{ 4 , 8 , 1 6 \}$ -class jobs at 2 levels of hierarchical depth, and 6-class jobs at 3 levels of hierarchical depth, totaling 222 jobs and 250 H100-hours with a critical path of 4.3 wall-hours. We cut the final dendrogram at 7 classes.

The ensemble successfully separates all four major LSU assembly intermediates (classes B-E) from the junk fraction without manual filtering. Perclass homogeneous refinement yields reconstructions at $\le 4 \mathring { A }$ resolution for all four states (further experimental details, maps, and FSC curves are in Appendix E). Figure 5 compares the particle counts in our predicted classes against the published annotations [32], as well as single-shot ab initio baselines. Unlike the single-shot baselines which miss out on the B and C major classes, our ensemble class populations are in broad agreement with the published labels, with the ensemble additionally resolving sub-states within classes D and E that are consistent with the finer partitions reported in prior work [5, 32].

![](images/71066d276dc06e0c1d2542af13173bdc1d96704cd0571eb18f03a6400b987d57.jpg)  
Figure 5: Particle counts of our reconstruction ensembling approach, the published major labels [32], and single-job baselines (each repeated with 3 different seeds).

## 4.6 Hyperparameter ablations

The aggregation step has five hyperparameters: three exponential rates in the weight heuristics $( \sigma , \beta ,$ γ), a minimum class size threshold $p ^ { \mathrm { m i n } }$ , and the sparsification neighborhood size $n _ { \mathrm { { n b r } } }$ . We sweep all five on Tomotwin-45-asym (Appendix F) and find that the weight parameters have minimal impact (fewer than 10 percentage points of variation), while $n _ { \mathrm { n b r } }$ is the most consequential choice, producing swings of over 45 percentage points. The main results in Table 1 use $( \sigma , \beta , \dot { \gamma } , p ^ { \mathrm { m i n } } , n _ { \mathrm { n b r } } ) =$ (0.6, 6, 10, 1/90, 100).

## 5 Discussion

We demonstrate that ensembling ab initio reconstruction jobs can resolve complex mixtures in cryo-EM beyond the reach of any single run or any existing method. Because constituent jobs used default settings, we believe these results represent a lower bound on what the framework can achieve with task-specific tuning. By reducing the problem of resolving complex mixtures to the orchestration of fast, coarse reconstruction jobs, the approach opens a path toward high-throughput heterogeneous reconstruction in modern experimental settings where samples contain tens or more distinct species. In doing so, it replaces the ad hoc classification schemes practitioners currently rely on with a systematic procedure that scales with compute rather than manual labor.

The aggregation framework is deliberately flexible. In our experiments, we considered ensembles with a fixed K and a predefined replication strategy for scaling analysis, but the aggregation accepts any collection of multi-class reconstruction jobs regardless of their parameters, input subsets, or tree structure. A practitioner could mix jobs with different K values, incorporate hand-picked runs targeting specific subpopulations, or feed in results from entirely different software packages. This same flexibility makes the framework amenable to autonomous or agent-driven exploration strategies that adaptively allocate compute based on intermediate ensemble results.

Several limitations warrant discussion. First, although the aggregation function is principled, it introduces hyperparameters that encode priors on cluster sizes and uniformity. Our ablations show robustness to the weight heuristic parameters, but sensitivity to the sparsification neighborhood size $n _ { \mathrm { { n b r } } }$ . Whether our default value for $n _ { \mathrm { n b r } }$ generalizes across diverse real datasets remains to be tested. Second, our method classifies particles but does not explicitly model junk; on EMPIAR-10076, junk particles were separated into their own cluster, but our meta-algorithm could benefit from a dedicated filtering step or treatment of a junk or outlier class. Third, the $N \times N$ similarity matrix becomes a memory bottleneck for large particle stacks.¹ Sparse storage and graph-based clustering algorithms that avoid materializing the full matrix are natural remedies to enable scaling to datasets with millions of particles. Addressing these limitations would bring the framework closer to routine deployment on the increasingly complex mixtures encountered in modern cryo-EM experiments, including cell lysates, crude extracts, and in situ imaging.

## Acknowledgments and Disclosure of Funding

The authors acknowledge the use of computing resources at Princeton Research Computing, a consortium of groups led by the Princeton Institute for Computational Science and Engineering (PICSciE) and Office of Information Technology's Research Computing. This research has been made possible by in part by grant number 2025-358484 from the Chan Zuckerberg Initiative DAF, an advised fund of Silicon Valley Community Foundation, the National Institutes of Health under grant number DP2GM164606, and the AI2050 program at Schmidt Sciences (Grant G-25-69788). The Zhong lab is grateful for support from the Princeton Catalysis Initiative, Princeton School of Engineering and Applied Sciences, Janssen Pharmaceuticals, and Generate Biomedicines. The funders had no role in study design, data collection and analysis, decision to publish or preparation of the manuscript.

## References

[1] Sjors HW Scheres. RELION: implementation of a Bayesian approach to cryo-EM structure determination. Journal of Structural Biology, 180(3):519–530, 2012.

[2] Ali Punjani, John L Rubinstein, David J Fleet, and Marcus A Brubaker. cryoSPARC: algorithms for rapid unsupervised cryo-EM structure determination. Nature Methods, 14(3):290–296, 2017.

[3] Dmitry Lyumkis, Axel F Brilot, Douglas L Theobald, and Nikolaus Grigorieff. Likelihoodbased classification of cryo-EM images using FREALIGN. Journal of Structural Biology, 183 (3):377–388, 2013.

[4] Timothy Grant, Alexis Rohou, and Nikolaus Grigorieff. cisTEM, user-friendly software for single-particle image processing. eLife, 7:e35383, 2018.

[5] Ellen D Zhong, Tristan Bepler, Bonnie Berger, and Joseph H Davis. CryoDRGN: reconstruction of heterogeneous cryo-EM structures using neural networks. Nature Methods, 18(2):176–185, 2021.

[6] Ali Punjani and David J Fleet. 3D variability analysis: Resolving continuous flexibility and discrete heterogeneity from single particle cryo-EM. Journal of Structural Biology, 213(2): 107702, 2021.

[7] Takanori Nakane, Dari Kimanius, Erik Lindahl, and Sjors HW Scheres. Characterisation of molecular motions in cryo-EM single-particle data by multi-body refinement in RELION. eLife, 7:e36861, 2018.

[8] Claire Donnat, Axel Levy, Frederic Poitevin, Ellen D Zhong, and Nina Miolane. Deep generative modeling for volume reconstruction in cryo-electron microscopy. Journal of Structural Biology, 214(4):107920, 2022.

[9] Eva Nogales and Julia Mahamid. Bridging structural and cell biology with cryo-electron microscopy. Nature, 628(8006):47–56, 2024.

[10] Chi-Min Ho, Xiaorun Li, Mason Lai, Thomas C Terwilliger, Josh R Beck, James Wohlschlegel, Daniel E Goldberg, Anthony WP Fitzpatrick, and Z Hong Zhou. Bottom-up structural proteomics: cryoEM of protein complexes enriched from the cellular milieu. Nature Methods, 17 (1):79–85, 2020.

[11] Axel Levy, Rishwanth Raghu, David Shustin, Adele Peng, Huan Li, Oliver Clarke, Gordon Wetzstein, and Ellen Zhong. Mixture of neural fields for heterogeneous reconstruction in cryo-EM. Advances in Neural Information Processing Systems, 37:56988–57017, 2024.

[12] Minkyu Jeon, Rishwanth Raghu, Miro Astore, Geoffrey Woollard, Ryan Feathers, Alkin Kaz, Sonya Hanson, Pilar Cossio, and Ellen Zhong. CryoBench: Diverse and challenging datasets for the heterogeneity problem in cryo-EM. Advances in Neural Information Processing Systems, 37:89468–89512, 2024.

[13] Robert E Schapire. The strength of weak learnability. Machine Learning, 5(2):197–227, 1990

[14] Alexander Strehl and Joydeep Ghosh. Cluster ensembles—a knowledge reuse framework for combining multiple partitions. Journal of Machine Learning Research, 3(Dec):583–617, 2002.

[15] Ana LN Fred and Anil K Jain. Combining multiple clusterings using evidence accumulation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(6):835–850, 2005.

[16] Miloš Vulović, Raimond BG Ravelli, Lucas J van Vliet, Abraham J Koster, Ivan Lazić, Uwe Lücken, Hans Rullgård, Ozan Öktem, and Bernd Rieger. Image formation modeling in cryoelectron microscopy. Journal of Structural Biology, 183(1):19–32, 2013.

[17] Ellen D Zhong, Adam Lerer, Joseph H Davis, and Bonnie Berger. CryoDRGN2: Ab initio neural reconstruction of 3D protein structures from real cryo-EM images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4066–4075, 2021.

[18] Muyuan Chen and Steven J Ludtke. Deep learning-based mixed-dimensional Gaussian mixture model for characterizing variability in cryo-EM. Nature Methods, 18(8):930–936, 2021.

[19] Ali Punjani and David J Fleet. 3DFlex: determining structure and motion of flexible proteins from cryo-EM. Nature Methods, 20(6):860–870, 2023.

[20] Axel Levy, Rishwanth Raghu, Ryan J Feathers, Michal Grzadkowski, Frédéric Poitevin, Jake D Johnston, Francesca Vallese, Oliver Biggs Clarke, Gordon Wetzstein, and Ellen D Zhong. CryoDRGN-AI: neural ab initio reconstruction of challenging cryo-EM and cryo-ET datasets. Nature Methods, 22(7):1486–1494, 2025.

[21] O Lauzirika, M Pernica, D Herreros, E Ramírez-Aportela, J Krieger, M Gragera, M Iceta, P Conesa, Y Fonseca, J Jiménez, et al. How many (distinguishable) classes can we identify in single-particle analysis? Acta Crystallographica Section D, 81(10):535–544, Oct 2025. doi: 10.1107/S2059798325007831.

[22] Team CryoSPARC, Kelly Barber, Hannah Bridges, Suhail Dawood, Katherine Elder, Nick Frasser, Fiona Hu, Serena Liu, Michael McLean, Ryan Narine, et al. End-to-end automation of repeat-target cryo-EM structure determination in CryoSPARC. bioRxiv, 2025. doi: 10.1101/ 2025.10.17.682689.

[23] Yang Yan, Shiqi Fan, Fajie Yuan, and Huaizong Shen. A comprehensive foundation model for cryo-EM image processing. Nature Methods, 23:88–95, 2026.

[24] J-H Schäfer, Austin Calza, Keenan Hom, Puneeth Damodar, Ruizhi Peng, Nebojša Bogdanović, Gabriel C Lander, Scott M Stagg, and Michael A Cianfrocco. CryoSift: an accessible and automated CNN-driven tool for cryo-EM 2D class selection. Structural Biology and Crystallization Communications, 81(12):517–526, 2025.

[25] Yoav Freund and Robert E Schapire. A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting. Journal of Computer and System Sciences, 55(1):119–139, 1997.

[26] Tianqi Chen and Carlos Guestrin. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, page 785–794, New York, NY, USA, 2016. Association for Computing Machinery. ISBN 9781450342322. doi: 10.1145/2939672.2939785.

[27] Xiaoli Zhang Fern and Carla E Brodley. Solving cluster ensemble problems by bipartite graph partitioning. In Proceedings of the Twenty-First International Conference on Machine Learning, page 36, 2004.

[28] Edward H Simpson. Measurement of Diversity. Nature, 163(4148):688–688, 1949.

[29] Stephen A Rhoades. The Herfindahl-Hirschman Index. Federal Reserve Bulletin, 79:188, 1993

[30] Kookjoo Kim, Huan Li, and Oliver B. Clarke. High-resolution ab initio reconstruction enables cryo-EM structure determination of small particles. bioRxiv, 2025. doi: 10.1101/2025.09.08. 674935.

[31] Tomáš Hodaň, Jiří Matas, and Štěpán Obdržálek. On evaluation of 6D object pose estimation. In European Conference on Computer Vision, pages 606–619. Springer, 2016.

[32] Joseph H Davis, Yong Zi Tan, Bridget Carragher, Clinton S Potter, Dmitry Lyumkis, and James R Williamson. Modular assembly of the bacterial large ribosomal subunit. Cell, 167(6): 1610–1622, 2016.

[33] Lawrence Hubert and Phipps Arabie. Comparing partitions. Journal of Classification, 2(1): 193–218, 1985.

[34] Nguyen Xuan Vinh, Julien Epps, and James Bailey. Information theoretic measures for clusterings comparison: is a correction for chance necessary? In Proceedings of the 26th Annual International Conference on Machine Learning, ICML 09, page 1073–1080, New York, NY, USA, 2009. Association for Computing Machinery. ISBN 9781605585161. doi: 10.1145/1553374.1553511.

[35] Edward B Fowlkes and Colin L Mallows. A method for comparing two hierarchical clusterings. Journal of the American Statistical Association, 78(383):553–569, 1983.

[36] Phipps Arabie and Scott A Boorman. Multidimensional scaling of measures of distance between partitions. Journal of Mathematical Psychology, 10(2):148–203, 1973.

[37] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12:2825–2830, 2011.

[38] Eric F Pettersen, Thomas D Goddard, Conrad C Huang, Elaine C Meng, Gregory S Couch, Tristan I Croll, John H Morris, and Thomas E Ferrin. UCSF ChimeraX: Structure visualization for researchers, educators, and developers. Protein Science, 30(1):70–82, 2021.

![](images/bafabe3812cb2d14ebb69f15b242116c51d682747b983fbde89705c6a1a52e9b.jpg)

## A Metric heatmaps of one-shot ab initio runs

In this section, experiments with $K > M$ were also included to complete the single-shot exploration grid.

Figure 6 reports the mean and median pose error metrics. The geodesic error is the rotation angle (in degrees), and the squared error is the squared error between the true and predicted rotation matrices.

Figure 7 reports the classification metrics (accuracy, precision, macro F1). Note that for the calculation of these metrics, we find the most abundant true label in each predicted class to map it to that true label, and in $K > M$ regime, this mapping is not one-to-one, therefore inducing a propensity for higher scores.

Figure 8 reports the clustering metrics calculated before the mapping mentioned above, therefore providing an alternative heuristic of the ability of the job. The metrics used are adjusted Rand index (ARI) [33], adjusted mutual information (AMI) [34], normalized mutual information (NMI), Fowkles-Mallows index (FMI) [35], and variation of information (VI) [36]. Out of these; ARI, AMI, and NMI are not well defined if one of the label assignments only has a single class, therefore those cells at the boundaries are set to 0. All of these metrics were calculated using scikit-learn library [37].

In all figures reported in this section, failed jobs where no convergence was achieved are explicitly annotated. The results overall highlight the graceful degradation of the merit of the job as the true heterogeneity M varies with respect to K (i.e., off-diagonal directions).

In total, 49 jobs were executed on Tomotwin-45-asym, and 64 jobs were executed on Tomotwin-100.

![](images/3fe208427f7a700c8c2426920f044eb4fe077def62b0d5ae60bde1bd49d86291.jpg)

![](images/27a88ac2240c01cd5099f3ada292e445c047e6a5796f54ab72e36207be20a938.jpg)

![](images/0697635ae9a2c942f1bc8965a19489511c1fc65f70401d00b1dab469c894b71b.jpg)

![](images/7ff42dd043cbe9d4e343f050c4eff2fa221a0534fda76fc97826f8825b4d37bd.jpg)  
Figure 6: Mean (left) and median (right) pose error metrics of the single-shot ab initio runs on subsets of Tomotwin-45-asym with increasing heterogeneity.

![](images/769994f8d7795096eb8f121442fc743fe981dc246b891286e89ee919d92ada58.jpg)  
Figure 7: Classification metrics of the single-shot ab initio runs on subsets of Tomotwin-45-asym (left) and Tomotwin-100 (right) with increasing heterogeneity.

![](images/264770d247a342c3284a71823d54c4a607cc6a4fac2c3c885a0bf72233e19bb3.jpg)  
Figure 8: Clustering metrics of the single-shot ab initio runs on subsets of Tomotwin-45-asym (left) and Tomotwin-100 (right) with increasing heterogeneity.

## B Experimental details of the main results

## B.1 Baseline methods in Tomotwin-45-asym

Following baseline method settings were used on Tomotwin-45-asym with classification and pose accuracies as measurement of merit.

• cryoDRGN2: The abinit\_het command in cryoDRGN v3.4 has been used with the canonical latent dimensionality of 8, all other parameters kept at default. The model has been trained for 20 epochs on 1 H100 in 4 hours. Using the analyze command, the resulting latent space is clustered with 45 k-means centroids, yielding the labeling subject to classification accuracy calculation.

• cryoDRGN-AI: The train command in DRGN-AI repository has been used with the canonical “autodecoder"architecture configuration, all parameters kept at default. The model has been trained for 111 epochs on 1 H100 in 3 hours. Using the analyze command, the resulting latent space is clustered with 45 k-means centroids, yielding the labeling subject to classification accuracy calculation.

• Hydra: The train command in Hydra repository has been used with the default configuration as DRGN-AI. Hydra was not tested on large K values, and its GPU parallelization strategy does not trivially lend itself to a multi-node multi-GPU setup, therefore K = 45 run is not viable. Facing this computational bottleneck, we trained a mixture of K = 4, for 10 epochs, on 2 H100s, in 6 wall-hours. The resulting 4-class probability distributions of the particles were used to find the most probable class assignment at each particle.

## B.2 Hyperparameter selection

For the results reported in Table 1, for both Tomotwin-45-asym and Tomotwin-100, the set of best performing hyperparameters in the hyperparameter ablation study (Appendix F) has been used:

$$
( \sigma , \beta , \gamma ,  { p ^ { \mathrm { m i n } } } , n _ { \mathrm { n b r } } ) = ( 0 . 6 , 6 , 1 0 , 1 / 9 0 , 1 0 0 )\tag{8}
$$

## B.3 Job ensemble generation

Recall that we define a K-class, L-level job hierarchy by a tree of K-class jobs, executed on their respective parent's children particle stacks in the tree structure, with a tree of height $L - 1$ Therefore, one such hierarchy consists of $1 + K + \cdot \cdot \cdot + K ^ { L - 1 }$ jobs. The critical path of this tree would then include L jobs (length of sequential dependency), and using the job timing assumption t(K) = 8K (min) (Appendix D), the critical path would take $t _ { c } = 8 K L ( \mathrm { { m i n } ) }$ . Assuming scalable compute access (e.g., cloud instances), the realized wall time could be as low as the critical path.

For Tomotwin-45-asym, our ensemble includes the following jobs, which are also listed on Table 2:

• 3 random replicates each of, 3-level hierarchies, of $K = 4$

• 1 random replicate of, 3-level hierarchy, of $K = 8$

• 4 random replicates each of, 2-level hierarchies, of $K = 8 .$

• 5 random replicates each of, 2-level hierarchies, of $K = 1 6 .$

Table 2: Job ensemble specification for the Tomotwin-45-asym analysis. The total number of jobs was 257, total GPU-hours used was 331.2, and the longest critical path was 4.3 hours.
<table><tr><td># Replicates</td><td>K</td><td>Levels</td><td>Total Job Count</td><td>Total H100-hours</td><td>Critical path (wall-hours)</td></tr><tr><td>3</td><td>4</td><td>3</td><td>63</td><td>33.6</td><td>1.6</td></tr><tr><td>1</td><td>8</td><td>3</td><td>73</td><td>77.9</td><td>3.2</td></tr><tr><td>4</td><td>8</td><td>2</td><td>36</td><td>38.4</td><td>2.1</td></tr><tr><td>5</td><td>16</td><td>2</td><td>85</td><td>181.3</td><td>4.3</td></tr><tr><td></td><td></td><td></td><td>257</td><td>331.2</td><td>4.3</td></tr></table>

For Tomotwin-100, our ensemble includes the job hierarchies listed on Table 3:  
Table 3: Job ensemble specification for the Tomotwin-100 analysis. The total number of jobs was 215, total GPU-hours used was 308.8, and the longest critical path was 4.3 hours.
<table><tr><td># Replicates</td><td>K</td><td>Levels</td><td>Total Job Count</td><td>Total H100-hours</td><td>Critical path (wall-hours)</td></tr><tr><td>4</td><td>4</td><td>2</td><td>20</td><td>10.7</td><td>1.1</td></tr><tr><td>1</td><td>4</td><td>3</td><td>21</td><td>11.2</td><td>1.6</td></tr><tr><td>4</td><td>8</td><td>2</td><td>36</td><td>38.4</td><td>2.1</td></tr><tr><td>1</td><td>8</td><td>3</td><td>73</td><td>77.9</td><td>3.2</td></tr><tr><td>5</td><td>16</td><td>2</td><td>85</td><td>181.3</td><td>4.3</td></tr><tr><td colspan="4">235</td><td>319.5</td><td>4.3</td></tr></table>

In our cryoSPARC setup, we had access to ≈ 50 parallel GPU job submission lanes, therefore both of these ensembles could be done overnight.

# C Volume renderings of Tomotwin-45-asym results

![](images/87c955bb579b9f4f0190daed77569138e89935b6f7c7556a6a139899cae33f96.jpg)  
Figure 9: The volumes obtained from Tomotwin-45-asym ensemble classification, as reported in Table 1. The resulting particle classes were fed into homogeneous refinement to yield these refined volumes per each class, and the renderings were done in ChimeraX [38]

D Ensemble scaling experiments

![](images/bc0a88159442cdb50c80f2b4587f3501ef14530719ee9db9aef027ca216e61b8.jpg)

Growth strategy: (1) Wall-time: 1 jobs GPU time: 3 jobs

Growth strategy: (2\*) Wall-time: 2 jobs GPU time: 5 jobs

Growth strategy: (2) Wall-time: 2 jobs GPU time: 9 jobs

Growth strategy: (3\*) Wall-time: 3 jobs GPU time: 9 jobs

Growth strategy: (3\*\*) Wall-time: 3 jobs GPU time: 13 jobs

Figure 10: The job ensemble growth strategies employed in the scaling analysis, with the corresponding level annotations and computational costs. All strategies are demonstrated with 3 random replicates. Time is in the unit of jobs, where each job is assumed to require similar time for completion (under a constant K).

In generating the job forests that our method would aggregate as an ensemble, we face the task of enumerating the combinatorial search space of what a job ensemble could consist of. Even in our two-primitive formulation of the growth, and ignoring all the potential parametrizations one can utilize the modern ab initio reconstruction tools with, we still face the questions of: (1) what depths of job trees to generate, (2) how many replicates of which depths to generate, (3) with what K values to generate the constituent jobs.

As a reduction of the search space to highlight the first two questions, we therefore kept the K values constant per each ensemble we analyzed in the scaling experiments. Therefore, the K-knob, for the purposes of the enumeration provided here, is elevated from a per-job parameter to a per-ensemble parameter. After this adjustment, the depth/width questions are highlighted as the principal axes of ensemble growth.

As a further reduction for the purposes of the scaling experiment, we only allow trees at maximum two different heights: the singular deep tree, and the shallow replicate trees. Therefore, in a given strategy, the level number indicates the height+1 of the deep tree, and the number of stars indicate the height+1 of the shallow trees. This naming was chosen to highlight the fact that, assuming abundant spot compute availability for horizontal scaling, the height of the deep tree determines the wall-time required for the completion of the job ensemble (since other trees could be executed in parallel). These ensemble growth strategies are demonstrated in Figure 10.

The horizontal dimension of the ensemble generation is the random replicate generation. For the purposes of our enumeration, we include all the shallow trees and the deep tree in the number of random replicates R. For example, a 5-replicate (3\*\*)-strategy ensemble consists of 1 deep tree of height 2, and 4 shallow trees of height 1.

In the scaling trade-off frontiers plotted in Figure 4, each line consists of different random replicates under a constant $K$ and strategy. The base colors (red, green, blue) describe the K value of the ensembles considered, and the hue (light to dark, with increasing GPU-hours) describes the strategy employed.

For the time calculation, we observed that the execution time of the jobs scaled linearly with K (8 minutes per K), hence that fixed factor is assumed: $t ( K ) = 8 K ( \bar { \operatorname* { m i n } } )$ . Each job uses a single H100 GPU. The wall-times [i.e., the critical path of sequential dependency, $t _ { c } = 8 K L ( \mathrm { m i n } ) _ { . } ^ { - }$ of the ensembles are provided in Table 4.

Table 4: Wall-time (in hours) of different ensemble generation strategies.
<table><tr><td rowspan="2">K</td><td colspan="5">Growth Strategy</td></tr><tr><td>1</td><td>2*</td><td>2</td><td>3*</td><td>3**</td></tr><tr><td>4</td><td>0.53</td><td>1.07</td><td>1.07</td><td>1.60</td><td>1.60</td></tr><tr><td>8</td><td>1.07</td><td>2.13</td><td>2.13</td><td>3.20</td><td>3.20</td></tr><tr><td>16</td><td>2.13</td><td>4.27</td><td>4.27</td><td>6.40</td><td>6.40</td></tr></table>

Scaling analyses for both Tomotwin-45-asym and Tomotwin-100 have been conducted with the ensemble aggregation hyperparameters kept constant at:

$$
( \sigma , \beta , \gamma , p ^ { \mathrm { m i n } } , n _ { \mathrm { n b r } } ) = ( 0 . 4 5 , 5 , 1 0 , 1 / 4 5 0 , 4 0 0 )\tag{9}
$$

## E Reconstruction of the bacterial 50S ribosomal subunit

![](images/0b2aa72d609bfbad85454b5ed50879cf13986c30311134a67c91f456deda0a38.jpg)  
Figure 11: The major classes obtained through the reconstruction ensemble analysis of the 50S dataset. Class E features both the base (bottom) and the central protuberance (CP, top), Class D features only the CP, class C features only the base, and class B features none.

The EMPIAR-10076 entry is a single-particle cryo-EM dataset, consisting of the assembly intermediates of bL17-depleted E. coli large ribosomal subunit (LSU) [32]. Specifically, this dataset consists of a picked particle stack $( N = 1 3 2 \mathrm { k } , D = 3 2 0 \mathrm { p x } , 1 . 3 1 \mathring { A } / \mathrm { p i x } )$ , including 4 major classes of LSU assembly intermediates (B, C, D, E), 2 rare classes (the small 30S subunit F, and the complete 70S ribosome A), and 1 class of junk particles. This dataset has been commonly used as a benchmark dataset for heterogeneous reconstruction and analysis methods.

The canonical setup of this benchmark dataset follows the cryoDRGN [5] filtering to reduce the stack to $N = 9 7 \mathrm { k \Omega }$ by eliminating the junk class prior to the heterogeneity analysis. The merit of the benchmarked method, then, is determined by its success in isolating the 4 major classes.

In order to provide stronger evidence of practical applicability of our method, we expand on this setup by processing the whole unfiltered stack $( N = 1 3 2 \mathbf { k } )$ , directly downloadable from EMPIAR. The major state clasification is done ab initio, i.e. without prior pose information, in our setup. This is a more challenging setup than the known-pose/consensus volume setup used for models such as [5] and [18].

Ensemble Labels
<table><tr><td rowspan=9 colspan=1>Publ bels</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=7>Junk  A   B   C   D   E   F</td><td rowspan=1 colspan=1>Total</td></tr><tr><td rowspan=7 colspan=1>JunkABCDEF</td><td rowspan=1 colspan=1>16,601</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2,691</td><td rowspan=1 colspan=1>1,764</td><td rowspan=1 colspan=1>4,052</td><td rowspan=1 colspan=1>1,467</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>26,575</td></tr><tr><td rowspan=1 colspan=1>295</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>42</td><td rowspan=1 colspan=1>174</td><td rowspan=1 colspan=1>1,486</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>2,018</td></tr><tr><td rowspan=1 colspan=1>1,836</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>9,360</td><td rowspan=1 colspan=1>747</td><td rowspan=1 colspan=1>649</td><td rowspan=1 colspan=1>58</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>12,650</td></tr><tr><td rowspan=1 colspan=1>1,705</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>723</td><td rowspan=1 colspan=1>22,576</td><td rowspan=1 colspan=1>229</td><td rowspan=1 colspan=1>871</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>26,104</td></tr><tr><td rowspan=1 colspan=1>2,164</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>1,536</td><td rowspan=1 colspan=1>141</td><td rowspan=1 colspan=1>21,356</td><td rowspan=1 colspan=1>941</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>26,138</td></tr><tr><td rowspan=1 colspan=1>1,827</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>229</td><td rowspan=1 colspan=1>1,741</td><td rowspan=1 colspan=1>4,207</td><td rowspan=1 colspan=1>28,557</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>36,561</td></tr><tr><td rowspan=1 colspan=1>1,814</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>ó</td><td rowspan=1 colspan=1>1,853</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>26,242</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>14,576</td><td rowspan=1 colspan=1>27,019</td><td rowspan=1 colspan=1>30,674</td><td rowspan=1 colspan=1>33,388</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>131,899</td></tr></table>

Figure 12: Confusion matrix between the published major class annotations and the predicted ensemble labels.

In our major state classification (Figures 5, 11, and 12), we reach 75% accuracy in matching the published major class annotations (Fig. 4 of [32]), successfully isolating the dataset into 5 classes (4 major classes and the junk particle class).

At the clustering stage, we used 7 clusters with the hopes that the rare classes A and F could also be captured. Our method could not capture these rare states. However, the clusters found included a subclassification of the classes D and E, which are named with + and — (Figure 11). The + variants feature density in the L1 stalk (left in the provided projections) and shoulder (right in the provided projections) domains, while the – variants do not feature L1 stalk. This finding was consistent with the minor classifications reported in [32]. These variants were merged to produce the 5-class classification, including the junk class (not pictured). The half-map FSC resolution and other map quality metric curves of each obtained class are reported through Figures 13-14.

In obtaining the result, our ensemble includes the following jobs, which are also listed on Table 5:

• 3 random replicates each of, 2-level hierarchies (root and children jobs, i.e. $K + 1$ jobs total in a tree), of $K = \{ 4 , 8 , 1 6 \}$

• 3 random replicates each of, 3-level hierarchy (root, children, and grandchildren jobs, i.e. $K ^ { 2 } + K { \bf \bar { + } } 1$ jobs total in a tree), of $K = 6 .$

Table 5: Job ensemble specification for the 50S analysis. The total number of jobs was 222, total GPU-hours used was 248.8, and the longest critical path was 4.3 hours.
<table><tr><td># Replicates</td><td>K</td><td>Levels</td><td>Total Job Count</td><td>Total H100-hours</td><td>Critical path (wall-hours)</td></tr><tr><td>3</td><td>4</td><td>2</td><td>15</td><td>8.0</td><td>1.1</td></tr><tr><td>3</td><td>8</td><td>2</td><td>27</td><td>28.8</td><td>2.1</td></tr><tr><td>3</td><td>16</td><td>2</td><td>51</td><td>108.8</td><td>4.3</td></tr><tr><td>3</td><td>6</td><td>3</td><td>129</td><td>103.2</td><td>2.4</td></tr><tr><td></td><td></td><td></td><td>222</td><td>248.8</td><td>4.3</td></tr></table>

Assuming scalable compute access (e.g., cloud instances), the wall time could be as convenient as 4.5 hours. In our cryoSPARC setup, we had access to ≈ 50 parallel GPU job submission lanes, therefore our experiments were done overnight. Finally, the aggregation hyperparameters used were:

$$
( \sigma , \beta , \gamma , p ^ { \mathrm { m i n } } , n _ { \mathrm { n b r } } ) = ( 0 . 4 5 , 5 , 1 0 , 1 / 4 5 0 , 4 0 0 )\tag{10}
$$

B  
![](images/45148d7dfa911ec64bd0b9c3bc1d0c50b8a5e5c6eddda38d77563b5e874cc7fc.jpg)

C  
![](images/e76eb2085a988da34b1291abef8257a6ae15835a7f24fc1921f96f9bd045a7dd.jpg)

![](images/e3baaca12cf943951606704124bce291072906ec34a63f343312af6f8c1da40c.jpg)

![](images/52ead1ff66406495c50946f4eacc748b290aadd516856039709f1e22b58c8334.jpg)

![](images/623301e2db1d8bf19847daeabfaf926a4fbb8df2736c79b35de555093c146356.jpg)

![](images/008694c9149e8607d6e70b96055ead9c45707ab3f04e105a80e29bddf0cd4861.jpg)

D  
![](images/f9930c83eac34d4731b59411c44738c3a00b427f679525d3667f40ea83ff5077.jpg)

E  
![](images/665a135a29632a2bc98bae810a6a104f6072ecbe27b68d8846dc35197287bbcb.jpg)

![](images/053ea3772a4db6a44ac7c5637d5db6ed447268e420b07e6d6f87628ff5e99a88.jpg)

![](images/555bf716937ef1e6593961e63e4b4c0fbb363577f32d27274a72247664a451ca.jpg)

![](images/a334cf3f1915a0f302f37d468c62d04ff51a36f2cf594e7db0882f673a087555.jpg)

![](images/6ccde7be8ea86794c8029f63a7b391130385319664edc2a383831936c3501ca6.jpg)  
Figure 13: Fourier Shell Correlation (FSC) and conical FSC (cFSC) curves reported by cryoSPARC for classes B, C, D, and E.

D-  
![](images/3f5b3c3626dd801fa811e325e25b11edfe211033eb7ac3b1fac1776a9bc4116a.jpg)

![](images/1d51b449165dc74b499874a32280b5f336b60bbebd5cbdb732d5c54f4eaa1d04.jpg)

![](images/97f2e2e152664bb86c1886841ca7cee0d38e12e5c8224ea7463b1133e88440c9.jpg)

![](images/15d557e4887c7681ce5636c9d088d30c6d1dcc9cc79ad715d424862e795ab5b8.jpg)

![](images/50074bbc03a1cead394dadfc0e35e68390263c52592893142e9d45719458fbb2.jpg)

![](images/34067f651d192ad7fcc3fee0344f8858a7637874afeb86fba15288dc74a556b1.jpg)

E-  
![](images/b8d0602c35b30947ef8f55cdb2ccb68518cf9847078db47dcdd0ef78aa2fb98c.jpg)

![](images/ced7c382ed9280f474a004b53c0ee68c6e3a0fa4e97d7bf8d7890d6f4d50b160.jpg)

![](images/1b7e00dacf2eca8199aa3879798ddb191cb3c493de002e95dc68de2d51431a19.jpg)

![](images/cdd43ac22d04281b474784c0823bc2e99a94d5fa499d800bcfe7b0a20ae56ee6.jpg)

![](images/1cd18d16f1f4c0ba488f1544d40a3414012a8c5538b04e19ae04791659adf2e1.jpg)

![](images/52e249797a9e030f0643cd1c8a0a4f10b5b1ee4982c3babf19bd655736ab4a77.jpg)  
Figure 14: Fourier Shell Correlation (FSC) and conical FSC (cFSC) curves reported by cryoSPARC for classes D-, D+, E-, and E+.

## F Hyperparameter tuning study

We pursue a two-stage approach in our 5-dimensional hyperparameter grid sweep for alleviating the curse of dimensionality. We split the hyperparameters into two sets: the exponential factors $( \sigma , \beta , \gamma )$ in the weight heuristics, and the other two $( p ^ { \mathrm { m i n } } , n _ { \mathrm { n b r } } )$ , namely the tiny class fraction threshold and the number of neighbors to keep in the sparsification process. In the following discussion, whenever % is used with respect to accuracy variation, it represents absolute percentage points and not relative variations.

We start with $( \sigma , \beta , \gamma ) = ( 0 . 4 5 , 5 , 1 0 )$ and conduct a grid sweep on $( p ^ { \mathrm { m i n } } , n _ { \mathrm { n b r } } )$ , reported in Figure 15. The results show negligible dependency on the tiny class proportion (< 4% fluctuation), but a crucial dependency on the sparsification threshold (≈ 45% change). Specifically, we observe that sparsifying the dense matrix up to only a few hundreds of neighbors remaining is important. Note that this is encouraging from a computational complexity perspective: the more the similarity matrix is sparsified, the less time and memory complexity the clustering methods would require (reduction of complexity from quadratic to linear in terms of the number of particles).

In the second phase of the hyperparameter search, we start with the result of the first sweep and explore $( \sigma , \beta , \gamma )$ . Specifically, we pick the best cell from above, i.e. $( p ^ { \mathrm { m i n } } , n _ { \mathrm { n b r } } ) \ =$ (1/90, 100). The results are demonstrated in Figure 16. We observe that the fluctuations are

![](images/c51d75f3ba74aa0831865799cd2dec0f9d17a8873866c5faa544e04b99b224c9.jpg)  
Figure 15: Accuracies obtained in the hyperparameter grid sweep for the tiny class proportion threshold $\dot { \boldsymbol { p } } ^ { \mathrm { m i n } }$ and the number of nearest neighbors kept $n _ { \mathrm { { n b r } } }$ in the sparsification.

limited to a band of 10%, highlighting the robustness of the weight heuristics under no tuning, while showing upside potential in tuning for minor performance gains in more detailed analyses.

![](images/d05dc28534305892c1049ac5113763b1a03abec8d27bd39dc8d9d52dd20b70fc.jpg)  
Figure 16: Accuracies obtained in the hyperparameter grid sweep for the exponential rate factors $( \sigma , \beta , \gamma )$ in the weight heuristics.

For the results reported in Table 1, for both Tomotwin-45-asym and Tomotwin-100, the set of best performing hyperparameters above has been used:

$$
( \sigma , \beta , \gamma ,  { p ^ { \mathrm { m i n } } } , n _ { \mathrm { n b r } } ) = ( 0 . 6 , 6 , 1 0 , 1 / 9 0 , 1 0 0 )\tag{11}
$$

The job ensembles used in this study are the same as in the main results, therefore requiring no further GPU-hours. The primary driver of the computational complexity in this hyperparameter sweep is the clustering step. Each clustering assignment takes a few minutes on a single CPU core, hence using parallel CPUs, the ablation study is done in an hour.