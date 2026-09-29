# BEYOND SITE AGREEMENT: RE-ESTIMATION FOR BRAIN NETWORK GENERALIZATION

Yingxu Wang<sup>1∗</sup>, Kunyu Zhang<sup>2∗</sup>, Yanwu Yang<sup>3</sup>, Thomas Wolfers<sup>3</sup>, Yujie Wu<sup>4</sup>, Siyang Gao<sup>5</sup>, Nan Yin<sup>6</sup>

<sup>1</sup> Mohamed bin Zayed University of Artificial Intelligence <sup>2</sup> Zhengzhou University <sup>3</sup> University Hospital Tübingen <sup>4</sup> The Hong Kong Polytechnic University <sup>5</sup> City University of Hong Kong <sup>6</sup> The Education University of Hong Kong {yingxv.wang,kunyu.zky,yangyanwu1111,yinnan8911}@gmail.com dr.thomas.wolfers@gmail.com, wu-yj16@tsinghua.org.cn siyangao@cityu.edu.hk

## ABSTRACT

Cross-site out-of-distribution (OOD) generalization in resting-state functional magnetic resonance imaging (rs-fMRI) often relies on learning task-discriminative representations from full-scan functional connectivity (FC) graphs and promoting invariance across source sites. However, FC graphs are estimated from finite, temporally correlated blood-oxygen-level-dependent (BOLD) sequences. Cross-site agreement therefore does not necessarily imply that predictive evidence remains supported under FC re-estimation within the same scan. In this paper, we propose Brain Network Re-estimation-Informed OOD Learning (BRIO), a framework that uses within-scan FC re-estimation to guide cross-site alignment. BRIO maps fullscan graphs and their re-estimates into consistently indexed connectome factors, enabling comparisons of their predictive contributions. It assesses re-estimation support from changes in these contributions relative to within-class subject variability and class separation. For each source-site pair and class, this task-calibrated support from both sites is combined with predictive relevance to form pairwise qualifications, which determine relative factor weights and overall alignment strength. Leave-one-site-out experiments on four real-world datasets (ABIDE, REST-meta-MDD, SRPBS, and ABCD) show that BRIO consistently outperforms competitive baselines, with relative improvements of up to 3.8% in accuracy. These gains also persist under an alternative brain parcellation on ABIDE.

## 1 INTRODUCTION

Resting-state functional magnetic resonance imaging (rs-fMRI) enables non-invasive characterization of macroscale functional interactions among brain regions Finn et al. (2015); Bessadok et al. (2022). A standard pipeline parcellates the brain into regions of interest (ROIs) and constructs subjectlevel functional connectivity (FC) graphs, where nodes represent ROIs and edges encode statistical dependencies estimated from regional blood-oxygen-level-dependent (BOLD) time series Kan et al. (2022); Zhang et al. (2026). To leverage these interregional dependencies for predicting neurological and psychiatric disorders, graph neural networks (GNNs) aggregate information across connected ROIs to learn subject-level predictive representations Li et al. (2021); Gu et al. (2025).

Despite this progress, brain network models often generalize poorly to unseen sites because differences in scanners, acquisition protocols, and cohort composition induce systematic shifts in FC distributions Qiu et al. (2025); Yamashita et al. (2025); Yu et al. (2018). To address these shifts, many cross-site out-of-distribution (OOD) methods seek FC graph representations that are both task-discriminative and invariant across source sites Li et al. (2025); Xu et al. (2025); Yu et al. (2025); Wang et al. (2025); Zhang & Xu (2026). They pursue this goal through site-confounder removal, invariant feature or subgraph learning, and class-conditional alignment Wang et al. (2026d); Chen et al. (2022); Wang et al. (2024). However, each underlying FC graph is a finite-sample estimate derived from a temporally correlated BOLD sequence Birn et al. (2013); Noble et al. (2019); Pakravan (2026); Xiang et al. (2025); Zhang et al. (2025). Re-estimating FC using different temporal samples from the same scan may change the estimated connectivity patterns and their contributions to model predictions Yang et al. (2026). These within-subject changes are not directly characterized by the discrepancy between representation distributions across sites. Consequently, low cross-site discrepancy among full-scan representations does not by itself reveal whether the predictive evidence used for alignment remains supported under within-scan FC re-estimation.

In this paper, we revisit cross-site OOD generalization in brain networks by accounting for variability from finite-sample FC estimation. This raises three sequential challenges: (1) How can FC reestimation variability be probed from a single rs-fMRI scan? A full-scan FC graph provides only one finite-sample estimate. Probing this variability involves varying the temporal sample composition while retaining local temporal dependence and cross-ROI temporal correspondence Bellec et al. (2008); Kudela et al. (2017). (2) How can changes in predictive evidence under FC re-estimation be represented and compared consistently? A GNN with graph-level readout aggregates information from multiple connectivity patterns into a single predictive representation Cui et al. (2022); Han et al. (2026). Comparing full-scan and re-estimated representations as a whole does not directly reveal changes in individual patterns’ predictive contributions. Resolving these changes requires consistent pattern correspondence across subjects, sites, and FC estimates. (3) How can within-subject re-estimation variability be interpreted for cross-site alignment? FC re-estimation probes changes within subjects, whereas cross-site alignment compares representation distributions across sites. The same contribution change can have different implications depending on within-class subject variability and class separation, which may vary across sites Yamashita et al. (2025). Thus, contribution changes need to be interpreted in the task context of both sites to assess a pattern’s suitability for alignment.

To address these challenges, we propose Brain Network Re-estimation-Informed OOD Learning (BRIO), a unified framework using within-scan FC re-estimation to guide cross-site alignment. It comprises three components: (i) Correspondence-Preserving Connectome Factorization constructs FC re-estimates from each BOLD sequence and maps full-scan and re-estimated graphs into consistently indexed connectome factors, enabling comparisons of predictive contributions. (ii) Task-Calibrated Re-estimation Qualification evaluates each factor’s predictive relevance and assesses its re-estimation support for each site and class by calibrating contribution changes under FC re-estimation against within-class subject variability and class separation. (iii) Qualification-Guided Cross-Site Alignment combines predictive relevance with support from both sites to form factor qualifications for each source-site pair and class. Normalized qualifications determine relative factor weights, while total qualification modulates class-conditional alignment strength between full-scan representations. Leave-one-site-out (LOSO) experiments on real-world datasets (ABIDE, REST-meta-MDD, SRPBS, and ABCD) show that BRIO consistently outperforms competitive baselines, achieving relative accuracy gains of up to 3.8% and maintaining its advantage under the alternative brain parcellation.

Our contributions are summarized as follows: (1) We revisit cross-site brain network OOD generalization by considering within-scan re-estimation variability, consistent comparisons of pattern-level predictive contributions, and interpretation of re-estimation variability for cross-site alignment. (2) We propose BRIO, which learns consistently indexed connectome factors and combines predictive relevance with task-calibrated re-estimation support to determine factor weights and class-conditional alignment strength. (3) We conduct leave-one-site-out experiments on four real-world datasets, show ing consistent gains over competitive baselines, including under the alternative brain parcellation.

## 2 RELATED WORK

Cross-Site OOD Generalization in Brain Networks. Graph OOD generalization aims to learn robust predictors under environment shifts through invariant subgraph learning, separation of invariant and spurious information, and representation alignment Li et al. (2025); Chen et al. (2022); Wang et al. (2024). In multi-site neuroimaging, harmonization methods reduce site effects in feature distributions and covariance Yu et al. (2018); Chen et al. (2021); Yang et al. (2026), while related analyses characterize how subject, scanner, and acquisition effects influence predictive biomarkers Yamashita et al. (2025); Wang et al. (2026b). To learn representations that generalize to unseen sites, brain network OOD methods employ transferable explanations, feature selection, and invariant subgraph learning Qiu et al. (2025). Other approaches use causal augmentation Yu et al. (2025) or combine site-aware deconfounding with transferable connectivity dynamics Wang et al. (2026d;c). BRIO uses within-scan FC re-estimation support to qualify predictive factors for cross-site alignment.

![](images/568829958198e9eb65d7a8728114bbc546308351fdf7c4eb307d7c4108157337.jpg)  
Figure 1: Overview of BRIO. Block resampling of the same BOLD scan produces FC re-estimates. Shared factorization uses a shared GNN and ROI map to obtain consistently indexed factors from full-scan and re-estimated graphs. Predictive relevance and task-calibrated re-estimation support from both sites form pairwise qualifications for each site pair and class, which control shared weighting and alignment strength for cross-site alignment of reweighted full-scan features.

Functional Connectome Estimation and Reliability. Functional connectomes are estimated from finite BOLD sequences, with scan duration, temporal sampling, and autocorrelation affecting their reliability and statistical properties Birn et al. (2013); Huotari et al. (2019); Xiang et al. (2026b;a). Test– retest studies examine how acquisition and processing choices affect FC reliability Noble et al. (2019), while methodological comparisons show how connectivity estimators shape network structure Smith et al. (2011). Statistical approaches include empirical Bayes normalization of connectivity metrics and bootstrap procedures for inference and uncertainty quantification Chen et al. (2015); Bellec et al. (2008); Kudela et al. (2017). Recent work develops learnable connectome representations from BOLD signals Yang et al. (2026), uses temporal segments for contrastive representation learning Lamprou et al. (2026), and explicitly models connectivity uncertainty Pakravan (2026). BRIO assesses taskcalibrated re-estimation support through changes in each connectome factor’s predictive contribution.

## 3 METHODOLOGY

Problem Setup. We study cross-site OOD generalization in brain network analysis with imaging sites as environments. Let $\mathbf { \bar { \mathcal { E } } } _ { \mathrm { t r } } = \{ 1 , \dots , E \}$ , with $E \geq 2$ , and $\mathcal { E } _ { \mathrm { t e } }$ denote disjoint source and unseen test environments. Each environment e has a joint distribution $\mathbb { P } ^ { e }$ over BOLD sequences and labels. For each source site $e \in \mathcal { E } _ { \mathrm { t r } }$ , we observe $\mathcal { D } ^ { e } = \{ ( \boldsymbol { X } _ { i } ^ { e } , \boldsymbol { Y } _ { i } ^ { e } ) \} _ { i = 1 } ^ { N _ { e } } \sim ( \mathbb { P } ^ { e } ) ^ { N _ { e } }$ , where $X _ { i } ^ { e } \in \mathbb { R } ^ { T _ { i } ^ { e } \times P }$ is the preprocessed BOLD sequence with $T _ { i } ^ { e }$ valid time points and P ROIs, and $Y _ { i } ^ { e } \in \{ 0 , 1 \}$ is the label. Sites share the label space, preprocessing pipeline, parcellation, and FC estimator, while P<sup>e</sup> may differ. The shared FC estimator Ψ yields full-scan graphs $A _ { i } ^ { e } = \Psi ( \mathbf { \bar { \Gamma } } X _ { i } ^ { e } ) \in \mathbb { R } ^ { P \times P }$ , whose entries encode FC between ROIs. Let $f _ { \theta }$ denote a predictor operating on FC graphs. We train it using only source data and evaluate its average risk on unseen sites:

$$
\mathcal { R } _ { \mathrm { O O D } } ( f _ { \theta } ) = \frac { 1 } { \left| \mathcal { E } _ { \mathrm { t e } } \right| } \sum _ { e \in \mathcal { E } _ { \mathrm { t e } } } \mathbb { E } _ { ( X , Y ) \sim \mathbb { P } ^ { e } } \left[ \ell ( f _ { \theta } ( \Psi ( X ) ) , Y ) \right] ,\tag{1}
$$

where ℓ denotes the prediction loss.

Overview. As shown in Fig. 1, BRIO uses within-scan FC re-estimation to guide cross-site alignment through three components: (i) Correspondence-Preserving Connectome Factorization constructs FC re-estimates from each BOLD sequence and maps full-scan and re-estimated graphs into consistently indexed connectome factors using a shared signed GNN and shared ROI assignment. (ii) Task-Calibrated Re-estimation Qualification evaluates source-balanced predictive relevance and assesses re-estimation support for each site and class by calibrating changes in factor-wise predictive contributions against within-class subject variability and between-class separation. (iii) Qualification-Guided Cross-Site Alignment combines predictive relevance with support from both sites to form qualifications for each source-site pair and class. These qualifications determine shared factor weights and the overall strength of class-conditional alignment between reweighted full-scan representations.

## 3.1 CORRESPONDENCE-PRESERVING CONNECTOME FACTORIZATION

FC estimated from a finite BOLD sequence can vary with the temporal samples used Kudela et al. (2017), potentially changing the predictive contributions of connectivity patterns. To compare these changes at the pattern level, we construct within-scan FC re-estimates and map them together with full-scan graphs into consistently indexed connectome factors.

(i) Within-Scan Re-estimation and Shared Connectome Encoding. BOLD signals exhibit temporal dependence and share a common time axis across ROIs. With $A _ { i } ^ { e , ( 0 ) } = A _ { i } ^ { e }$ denoting the full-scan graph, we construct K FC re-estimates by jointly resampling temporal blocks:

$$
\begin{array} { r } { \pmb { X } _ { i } ^ { e , ( k ) } = \pmb { \mathcal { R } } _ { k } \big ( \pmb { X } _ { i } ^ { e } \big ) , \quad \pmb { A } _ { i } ^ { e , ( k ) } = \Psi \Big ( \pmb { X } _ { i } ^ { e , ( k ) } \Big ) , \quad k = 1 , \ldots , K , } \end{array}\tag{2}
$$

where $\mathcal { R } _ { k }$ denotes joint circular moving-block resampling with a source-estimated block length $\ell _ { b }$ Shao & Yu (1993); Bellec et al. (2008). Resampling retains local temporal dependence within sampled blocks and preserves cross-ROI temporal correspondence through shared sampling indices.

To encode all FC estimates over the same ROI pairs while retaining their estimate-specific weights, we construct a fixed sparse propagation support $\mathbf { \bar { \Pi } } S \in \{ 0 , 1 \} ^ { P \times P }$ solely from source full-scan graphs. For $u \ne v$ , we compute

$$
\begin{array} { r } { \overline { { a } } _ { u v } = \mathrm { m e d i a n } _ { e \in \mathcal { E } _ { \mathrm { t r } } } \mathrm { m e d i a n } _ { 1 \leq i \leq N _ { e } } \left| \left[ \pmb { A } _ { i } ^ { e , ( 0 ) } \right] _ { u v } \right| , \quad S = \mathrm { S y m T o p } _ { m } \left( \overline { { A } } \right) , } \end{array}\tag{3}
$$

where $\overline { { A } } = [ \overline { { a } } _ { u v } ]$ has zero diagonal and $m = \lceil \rho _ { g } ( P - 1 ) \rceil$ , with $\rho _ { g } \in ( 0 , 1 ]$ . The operator $\mathrm { S y m T o p } _ { m }$ selects the m largest off-diagonal entries per row, sets $S _ { u v } = S _ { v u } = 1$ if either ROI selects the other, and sets all remaining entries to zero. For $k = 0 , \ldots , K$ , we retain each FC estimate’s weights on this support and separate positive weights from negative-weight magnitudes:

$$
G _ { i } ^ { e , ( k ) } = S \odot A _ { i } ^ { e , ( k ) } , \quad G _ { i } ^ { e , ( k ) , + } = \Big [ G _ { i } ^ { e , ( k ) } \Big ] _ { + } , \quad G _ { i } ^ { e , ( k ) , - } = \Big [ - G _ { i } ^ { e , ( k ) } \Big ] _ { + } ,\tag{4}
$$

where $\odot$ denotes elementwise multiplication and $[ \cdot ] _ { + }$ takes the elementwise positive part.

The shared parcellation provides consistent ROI labels across graphs, while local connectivity on $_ { s }$ varies across subjects and FC estimates. Let $e _ { u } ^ { \mathrm { R O I } } \in \mathbb { R } ^ { d _ { e } }$ denote the learnable identity embedding of ROI u. We summarize its local connectivity using the descriptor ${ c } _ { i , u } ^ { e , ( k ) }$

$$
c _ { i , u } ^ { e , ( k ) } = \left[ \frac { 1 } { d _ { u } } \sum _ { v } G _ { i , u v } ^ { e , ( k ) , + } , \frac { 1 } { d _ { u } } \sum _ { v } G _ { i , u v } ^ { e , ( k ) , - } , \sqrt { \frac { 1 } { d _ { u } } \sum _ { v } \left( G _ { i , u v } ^ { e , ( k ) } \right) ^ { 2 } } \right] ^ { \top } , \quad d _ { u } = \sum _ { v } S _ { u v } .\tag{5}
$$

Here, $G _ { i , u v } ^ { e , ( k ) }$ and $G _ { i , u v } ^ { e , ( k ) , \pm }$ are entries of the corresponding adjacency matrices, and $d _ { u }$ is the degree of ROI u in $\pmb { S }$ . We combine ROI identity and connectivity to initialize node features, then encode them with the shared signed GNN:

$$
\begin{array} { r } { \pmb { h } _ { i , u } ^ { e , ( k ) , [ 0 ] } = \phi _ { \mathrm { i n } } \left( \pmb { e } _ { u } ^ { \mathrm { R O I } } , \pmb { c } _ { i , u } ^ { e , ( k ) } \right) , \quad \pmb { H } _ { i } ^ { e , ( k ) } = \mathcal { G } _ { \theta _ { g } } \left( \pmb { H } _ { i } ^ { e , ( k ) , [ 0 ] } , \pmb { G } _ { i } ^ { e , ( k ) , + } , \pmb { G } _ { i } ^ { e , ( k ) , - } \right) , } \end{array}\tag{6}
$$

where $\phi _ { \mathrm { i n } } : \mathbb { R } ^ { d _ { e } } \times \mathbb { R } ^ { 3 }  \mathbb { R } ^ { d _ { h } }$ initializes node features, $H _ { i } ^ { e , ( k ) , [ 0 ] } \in \mathbb { R } ^ { P \times d _ { h } }$ stacks them row-wise, and $\mathcal { G } _ { \theta _ { g } }$ is the shared signed GNN. Its output $H _ { i } ^ { e , ( k ) } \in \mathbb { R } ^ { P \times d _ { h } }$ has u-th row $( \boldsymbol { h } _ { i , u } ^ { e , ( k ) } ) ^ { \top }$

(ii) Shared Connectome Factorization and Factor-Wise Scoring. Predictive connectivity patterns may involve multiple ROIs Li et al. (2021). We therefore pool the encoded ROI representations into connectome factors using shared ROI weighting profiles, so that each factor can be compared consistently across subjects, sites, and FC estimates.

Let $\pmb { { \cal E } } ^ { \mathrm { R O I } } = [ e _ { 1 } ^ { \mathrm { R O I } } , \dots , e _ { P } ^ { \mathrm { R O I } } ] ^ { \top } \in \mathbb { R } ^ { P \times d _ { e } }$ stack the shared ROI identity embeddings, and let $\pmb { Q } = [ \pmb { q } _ { 1 } , \ldots , \pmb { q } _ { J } ] ^ { \top }$ denote J learnable factor queries. The shared assignment $\pmb { \Pi } \in \mathbb { R } _ { + } ^ { J \times \check { P } }$ is defined as

$$
{ \bf \Pi } { \bf \Pi } = \mathcal { A } _ { \omega } \left( { \cal Q } , E ^ { \mathrm { R O I } } \right) , \quad { \bf \Pi } { \bf \Pi } _ { P } = \frac { 1 } { J } { \bf 1 } _ { J } , \quad { \bf \Pi } { \bf \Pi } ^ { \top } { \bf 1 } _ { J } = \frac { 1 } { P } { \bf 1 } _ { P } ,\tag{7}
$$

where $\mathbf { \mathcal { A } } _ { \omega }$ is a learnable mapping that computes a balanced assignment from similarities between factor queries and ROI identity embeddings, and ${ \bf 1 } _ { n }$ denotes the length-n all-ones vector. Since Π depends only on shared queries and ROI identity embeddings, each factor uses the same ROI weighting profile across subjects, sites, and FC estimates. Pooling the encoded ROI representations under these profiles gives the connectome factors:

$$
\boldsymbol { r } _ { i , j } ^ { e , ( k ) } = \mathcal { F } _ { \phi } \left( \boldsymbol { J } \sum _ { u = 1 } ^ { P } \Pi _ { j , u } \boldsymbol { h } _ { i , u } ^ { e , ( k ) } \right) \in \mathbb { S } ^ { d _ { t } - 1 } , \quad j = 1 , \ldots , J ,\tag{8}
$$

where $\mathcal { F } _ { \phi } : \mathbb { R } ^ { d _ { h } }  \mathbb { S } ^ { d _ { t } - 1 }$ is a shared mapping with unit-norm output.

To retain the individual factor representations, we concatenate them in a fixed order:

$$
\pmb { s } _ { i } ^ { e , ( k ) } = \frac { 1 } { \sqrt { J } } \left[ \pmb { r } _ { i , 1 } ^ { e , ( k ) } \parallel \cdots \parallel \pmb { r } _ { i , J } ^ { e , ( k ) } \right] .\tag{9}
$$

Here, ∥ denotes concatenation. To attribute prediction changes to individual factors, we use a linear head on the concatenated representation, yielding the prediction logit and additive factor scores:

$$
z _ { i } ^ { e , ( k ) } = \frac { 1 } { \sqrt { J } } \sum _ { j = 1 } ^ { J } \pmb { w } _ { j } ^ { \top } \pmb { r } _ { i , j } ^ { e , ( k ) } + b , \quad a _ { i , j } ^ { e , ( k ) } = \pmb { w } _ { j } ^ { \top } \pmb { r } _ { i , j } ^ { e , ( k ) } ,\tag{10}
$$

where ${ \pmb w } _ { j }$ is the classifier block for factor $j ,$ b is the bias, and $a _ { i , j } ^ { e , ( k ) }$ is the factor’s additive contribution to the logit before the common $1 / \sqrt { J }$ scaling.

## 3.2 TASK-CALIBRATED RE-ESTIMATION QUALIFICATION

The factor-wise contributions may distinguish classes in full-scan graphs while varying across FC reestimates of the same subject. Within-class subject variability and between-class separation provide a reference for assessing these changes at each site. We therefore evaluate source-balanced predictive relevance and task-calibrated re-estimation support.

To assess predictive relevance, we measure class separation in full-scan contributions relative to within-class variability. For source site e and class $c \in \{ 0 , 1 \}$ , let $\mathcal { T } _ { e , c } = \{ i \in \{ 1 , . . . , N _ { e } \} : Y _ { i } ^ { e } = c \}$ and $N _ { e , c } = | \mathcal { T } _ { e , c } |$ , with both classes represented at each source site. Each qualification evaluation computes the full-scan and re-estimated contributions of all subjects in $\mathcal { T } _ { e , c }$ under the same fixed model snapshot. From the full-scan contributions, we define the empirical mean, within-class variance, and class contrast as

$$
\mu _ { e , c , j } = \frac { 1 } { N _ { e , c } } \sum _ { i \in \mathcal { I } _ { c , c } } a _ { i , j } ^ { e , ( 0 ) } , \quad V _ { e , c , j } ^ { \mathrm { s u b } } = \frac { 1 } { N _ { e , c } } \sum _ { i \in \mathcal { I } _ { c , c } } \left( a _ { i , j } ^ { e , ( 0 ) } - \mu _ { e , c , j } \right) ^ { 2 } , \quad \Delta _ { e , j } = \mu _ { e , 1 , j } - \mu _ { e , 0 , j } .\tag{11}
$$

We aggregate these within-site variances and signed class contrasts with equal site weights so that larger cohorts do not dominate the relevance estimate:

$$
V _ { c , j } ^ { \mathrm { p o o l } } = \frac { 1 } { E } \sum _ { e \in { \mathscr { E } } _ { \mathrm { t r } } } V _ { e , c , j } ^ { \mathrm { s u b } } , \quad \Delta _ { j } = \frac { 1 } { E } \sum _ { e \in { \mathscr { E } } _ { \mathrm { t r } } } \Delta _ { e , j } .\tag{12}
$$

Opposing class contrasts offset one another before $\Delta _ { j }$ is squared. We measure relative class separation by normalizing the squared aggregate contrast by within-class variability. To account for contribution scale, we further weight this ratio by the relative squared classifier-block norm $\beta _ { j }$ , since unit-norm factors satisfy $| a _ { i , j } ^ { e , ( k ) } | \leq \| \pmb { w } _ { j } \| _ { 2 }$ . The resulting relevance scores are normalized across factors:

$$
\beta _ { j } = \frac { \| \pmb { w } _ { j } \| _ { 2 } ^ { 2 } } { \sum _ { j ^ { \prime } = 1 } ^ { J } \| \pmb { w } _ { j ^ { \prime } } \| _ { 2 } ^ { 2 } + \epsilon _ { w } } , \quad R _ { j } ^ { \mathrm { t a s k } } = \frac { \beta _ { j } \Delta _ { j } ^ { 2 } } { V _ { 0 , j } ^ { \mathrm { p o o l } } + V _ { 1 , j } ^ { \mathrm { p o o l } } + \epsilon _ { s } } , \quad \pi _ { j } ^ { \mathrm { t a s k } } = \frac { R _ { j } ^ { \mathrm { t a s k } } } { \sum _ { j ^ { \prime } = 1 } ^ { J } R _ { j ^ { \prime } } ^ { \mathrm { t a s k } } } ,\tag{13}
$$

where $\epsilon _ { w } , \epsilon _ { s } > 0$ are numerical stabilizers. For $\textstyle \sum _ { j = 1 } ^ { J } R _ { j } ^ { \mathrm { t a s k } } > 0$ , the normalized scores $\pi _ { j } ^ { \mathrm { t a s k } }$ form a probability distribution over factors.

We next assess within-scan re-estimation changes by averaging the squared differences between each subject’s re-estimated and full-scan contributions over subjects and re-estimates within each site and class:

$$
D _ { e , c , j } ^ { \mathrm { r e } } = \frac { 1 } { N _ { e , c } K } \sum _ { i \in \mathcal { Z } _ { e , c } } \sum _ { k = 1 } ^ { K } \left( a _ { i , j } ^ { e , ( k ) } - a _ { i , j } ^ { e , ( 0 ) } \right) ^ { 2 } .\tag{14}
$$

To interpret these deviations at each site, we combine full-scan within-class variability and class separation into a local reference scale:

$$
M _ { e , j } ^ { \mathrm { t a s k } } = \frac { 1 } { 4 } \Delta _ { e , j } ^ { 2 } , \quad T _ { e , c , j } = V _ { e , c , j } ^ { \mathrm { s u b } } + M _ { e , j } ^ { \mathrm { t a s k } } ,\tag{15}
$$

where $M _ { e , j } ^ { \mathrm { t a s k } }$ is the squared half-separation between the two class means at site e. Consequently, $T _ { e , c , j }$ equals the empirical mean squared distance of full-scan contributions in class c from the midpoint of these means. We use this reference scale to define re-estimation support for each site and class:

$$
q _ { e , c , j } ^ { \mathrm { r e } } = \frac { T _ { e , c , j } + \epsilon _ { q } } { T _ { e , c , j } + D _ { e , c , j } ^ { \mathrm { r e } } + \epsilon _ { q } } \in ( 0 , 1 ] ,\tag{16}
$$

where ${ \epsilon } _ { q } > 0$ is a numerical stabilizer. Higher support indicates smaller mean squared contribution changes relative to the local full-scan reference scale.

Proposition 1 (Class-Contrast Stability under FC Re-estimation) Fix the model snapshot, a source site $e ,$ and a factor $j ,$ , with $N _ { e , 0 } , N _ { e , 1 } > 0$ and $K \geq 1$ . Let $\Delta _ { e , j } ^ { \mathrm { r e } }$ be the class contrast of contributions averaged over subjects and FC re-estimates, and let $\kappa _ { e , c , j } = \Delta _ { e , j } ^ { 2 } / \big ( 4 ( V _ { e , c , j } ^ { \mathrm { s u b } } + \epsilon _ { q } ) \big )$ be thefull-scan separation ratio ofclass c. Then

$$
\left| \Delta _ { e , j } ^ { \mathrm { r e } } - \Delta _ { e , j } \right| \leq \sum _ { c \in \{ 0 , 1 \} } \sqrt { ( T _ { e , c , j } + \epsilon _ { q } ) \left( \frac { 1 } { q _ { e , c , j } ^ { \mathrm { r e } } } - 1 \right) } ,\tag{17}
$$

and $\Delta _ { e , j } ^ { \mathrm { r e } }$ has the same sign as $\Delta _ { e , j }$ whenever $q _ { e , c , j } ^ { \mathrm { r e } } > ( 1 + \kappa _ { e , c , j } ) / ( 1 + 2 \kappa _ { e , c , j } )$ for both $c \in \{ 0 , 1 \}$

Proposition 1 bounds the change of class contrast under FC re-estimation by the re-estimation variation, and shows that the support needed to preserve the sign of the contrast decreases with full-scan class separation, from one for weakly separated factors to one half for strongly separated ones. A factor with high relevance but low support, or low relevance but high support, can thus flip its class contrast under re-estimation, so we qualify factors by assessing the two criteria jointly.

## 3.3 QUALIFICATION-GUIDED CROSS-SITE ALIGNMENT

Predictive relevance combines class separation with contribution scale across source sites, whereas re-estimation support varies by site and class. For class-conditional alignment, we combine predictive relevance with support from both sites to determine relative factor weights and alignment strength.

For two distinct source sites $e , e ^ { \prime } \in \mathcal { E } _ { \mathrm { t r } } .$ , class $c \in \{ 0 , 1 \}$ , and factor $j ,$ , we combine re-estimation support from both sites and weight it by predictive relevance:

$$
\begin{array} { r } { q _ { e , e ^ { \prime } , c , j } ^ { \mathrm { p a i r } } = \sqrt { q _ { e , c , j } ^ { \mathrm { r e } } q _ { e ^ { \prime } , c , j } ^ { \mathrm { r e } } } , \quad \alpha _ { e , e ^ { \prime } , c , j } = \pi _ { j } ^ { \mathrm { t a s k } } q _ { e , e ^ { \prime } , c , j } ^ { \mathrm { p a i r } } . } \end{array}\tag{18}
$$

The geometric mean treats the two sites symmetrically and decreases when support at either site decreases. We normalize the qualifications to determine relative factor weights and retain their sum as the overall qualification for scaling the alignment loss:

$$
g _ { e , e ^ { \prime } , c } = \sum _ { j = 1 } ^ { J } \alpha _ { e , e ^ { \prime } , c , j } , \quad \overline { { \alpha } } _ { e , e ^ { \prime } , c , j } = \frac { \alpha _ { e , e ^ { \prime } , c , j } } { g _ { e , e ^ { \prime } , c } } , \quad \sum _ { j = 1 } ^ { J } \overline { { \alpha } } _ { e , e ^ { \prime } , c , j } = 1 ,\tag{19}
$$

where $0 < g _ { e , e ^ { \prime } , c } \leq 1$

Proposition 2 (Variational Form of Pairwise Qualification) For a probability vector π over J factors and supports $q _ { j } \in ( 0 , 1 ]$ , define $\begin{array} { r } { \alpha _ { j } = \pi _ { j } q _ { j } , g = \sum _ { j = 1 } ^ { J } \alpha _ { j } } \end{array}$ , and $\overline { { \alpha } } _ { j } = \alpha _ { j } / g .$ Let $\nu _ { \pi }$ be the

set of probability vectors v with $v _ { j } = 0$ whenever $\pi _ { j } = 0$ , and define

$$
\mathcal { I } ( \pmb { v } ) = \mathrm { K L } ( \pmb { v } \parallel \pmb { \pi } ) + \sum _ { j = 1 } ^ { J } v _ { j } \log \frac { 1 } { q _ { j } } .\tag{20}
$$

Then $\overline { { \alpha } }$ is the unique minimizer of J over $\nu _ { \pi }$ , with minimum value − log g, and

$$
\overline { { \alpha } } _ { j } = \frac { \partial \log g } { \partial \log q _ { j } } , \quad - \log g \leq \sum _ { j = 1 } ^ { J } \pi _ { j } \log \frac { 1 } { q _ { j } } ,\tag{21}
$$

where equality holds if and only $i f q _ { j }$ is constant over the factors with $\pi _ { j } > 0 .$

With $\pi = \pi ^ { \mathrm { t a s k } }$ and $q _ { j } = q _ { e , e ^ { \prime } , c , j } ^ { \mathrm { p a i r } }$ , Proposition 2 recovers $\overline { { \alpha } } _ { e , e ^ { \prime } , c }$ and $g _ { e , e ^ { \prime } , c }$ . The normalized qualifications are the weights closest to predictive relevance under a penalty for weak support, each equal to the sensitivity of alignment strength to its factor’s support. Weights and strength thus derive from one objective, and reweighting lowers the support penalty below relevance-only weighting.

We use the normalized qualifications derived from predictive contributions to weight full-scan factor representations identically at both sites. For $\bar { d \in \{ e , e ^ { \prime } \} }$ and subject $i \in \mathcal { T } _ { d , c } ,$ , the weighted representation is

$$
\varphi _ { i | e , e ^ { \prime } , c } ^ { Q , d } = \frac { 1 } { \sqrt { 2 } } \left[ \sqrt { \overline { { \alpha } } _ { e , e ^ { \prime } , c , 1 } } r _ { i , 1 } ^ { d , ( 0 ) } \parallel \cdots \parallel \sqrt { \overline { { \alpha } } _ { e , e ^ { \prime } , c , J } } r _ { i , J } ^ { d , ( 0 ) } \right] .\tag{22}
$$

For subjects $i \in \mathcal { T } _ { e , c }$ and $h \in \mathcal { T } _ { e ^ { \prime } , c } ,$ their squared distance in the common weighted space is

$$
\left\| \varphi _ { i | e , e ^ { \prime } , c } ^ { Q , e } - \varphi _ { h | e , e ^ { \prime } , c } ^ { Q , e ^ { \prime } } \right\| _ { 2 } ^ { 2 } = \sum _ { j = 1 } ^ { J } \overline { { \alpha } } _ { e , e ^ { \prime } , c , j } \left[ 1 - \left( r _ { i , j } ^ { e , ( 0 ) } \right) ^ { \top } r _ { h , j } ^ { e ^ { \prime } , ( 0 ) } \right] .\tag{23}
$$

Each factor’s cosine dissimilarity is weighted by its relative qualification. Since the factors have unit norm and the weights sum to one, the squared distance lies in [0, 2]. We then construct the empirical class-conditional distributions in this weighted space. Given nonempty source subsets $\mathcal { B } _ { e , c } \subseteq \mathcal { T } _ { e , c }$ and $\begin{array} { r } { B _ { e ^ { \prime } , c } \subseteq \mathbb { Z } _ { e ^ { \prime } , c } } \end{array}$ , we define

$$
\widehat { P } _ { e , c } ^ { Q [ e , e ^ { \prime } ] } = \frac { 1 } { | \mathcal { B } _ { e , c } | } \sum _ { i \in \mathcal { B } _ { e , c } } \delta _ { \varphi _ { i | e , e ^ { \prime } , c } ^ { Q , e } } , \quad \widehat { P } _ { e ^ { \prime } , c } ^ { Q [ e , e ^ { \prime } ] } = \frac { 1 } { | \mathcal { B } _ { e ^ { \prime } , c } | } \sum _ { h \in \mathcal { B } _ { e ^ { \prime } , c } } \delta _ { \varphi _ { h | e , e ^ { \prime } , c } ^ { Q , e ^ { \prime } } } ,\tag{24}
$$

where $\delta _ { x }$ denotes a point mass at x. We compare these distributions using the debiased Sinkhorn divergence $S _ { \varepsilon }$ with the squared Euclidean ground cost and entropic regularization parameter $\varepsilon >$ 0 Cuturi (2013); Feydy et al. (2019). Details of debiased Sinkhorn divergence $S _ { \varepsilon }$ are provided in Appendix D. Weighting each divergence by $g _ { e , e ^ { \prime } , c }$ and averaging over source-site pairs and classes yields the alignment loss:

$$
\mathcal { L } _ { \mathrm { e n v } } ^ { Q } = \frac { 2 } { E ( E - 1 ) } \sum _ { e , e ^ { \prime } \in \mathcal { E } _ { \mathrm { t r } } } \frac { 1 } { 2 } \sum _ { { c } \in \{ 0 , 1 \} } g _ { e , e ^ { \prime } , c } S _ { \varepsilon } \left( \widehat { P } _ { e , c } ^ { Q [ e , e ^ { \prime } ] } , \widehat { P } _ { e ^ { \prime } , c } ^ { Q [ e , e ^ { \prime } ] } \right) .\tag{25}
$$

## 3.4 LEARNING OBJECTIVE

For each full-scan graph, the predicted probability is $\boldsymbol { { \widehat { p } } } _ { i } ^ { e , ( 0 ) } = \sigma \Big ( z _ { i } ^ { e , ( 0 ) } \Big )$ , where $\sigma$ denotes the sigmoid function. We define the classification loss with equal weights across source sites and classes:

$$
\mathcal { L } _ { \mathrm { c l s } } = \frac { 1 } { 2 E } \sum _ { e \in \mathcal { E } _ { \mathrm { t r } } } \sum _ { c \in \{ 0 , 1 \} } \frac { 1 } { N _ { e , c } } \sum _ { i \in \mathcal { T } _ { e , c } } \ell _ { \mathrm { B C E } } \left( \widehat { p } _ { i } ^ { e , ( 0 ) } , Y _ { i } ^ { e } \right) ,\tag{26}
$$

where $\ell _ { \mathrm { B C E } }$ denotes binary cross-entropy. The overall objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { e n v } } \mathcal { L } _ { \mathrm { e n v } } ^ { Q } , } \end{array}\tag{27}
$$

where $\lambda _ { \mathrm { e n v } } \geq 0$ controls the strength of cross-site alignment.

Table 1: Results on ABIDE, ABIDE (CC200), REST-meta-MDD, SRPBS, and ABCD (ADHD-task) over LOSO sites. Bold results indicate the best performance.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="2">ABIDE</td><td colspan="2">ABIDE (CC200)</td><td colspan="2">REST-meta-MDD</td><td colspan="2">SRPBS</td><td colspan="2">ABCD (ADHD-task)</td></tr><tr><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td></tr><tr><td rowspan="4">NNN</td><td>GCN</td><td> $6 1 . 9 1 _ { \pm 1 0 . 3 2 }$ </td><td> $5 6 . 1 2 _ { \pm 9 . 6 6 }$ </td><td> $5 0 . 2 1 { \scriptstyle \pm 8 . 8 6 }$ </td><td>48.38±8.55</td><td> $6 0 . 5 8 _ { \pm 4 . 1 8 }$ </td><td> $5 6 . 2 6 { \scriptstyle \pm 2 . 8 7 }$ </td><td> $6 5 . 2 4 _ { \pm 1 9 . 4 7 }$ </td><td> $6 1 . 6 8 _ { \pm 1 9 . 1 7 }$ </td><td> $7 1 . 7 5 _ { \pm 6 . 5 2 }$ </td><td> $6 3 . 1 4 _ { \pm 9 . 1 5 }$ </td></tr><tr><td>GAT</td><td> $5 8 . 2 1 _ { \pm 8 . 6 9 }$ </td><td> $5 2 . 3 1 _ { \pm 6 . 0 1 }$ </td><td> $5 7 . 6 8 _ { \pm 9 . 3 7 }$ </td><td> $5 4 . 6 9 _ { \pm 7 . 7 4 }$ </td><td> $5 9 . 9 7 _ { \pm 9 . 3 8 }$ </td><td> $5 6 . 1 7 _ { \pm 6 . 7 2 } ^ { - }$ </td><td> $6 6 . 6 8 _ { \pm 1 5 . 5 8 }$ </td><td>60.86±14.38</td><td> $7 0 . 3 7 _ { \pm 1 1 . 3 2 }$ </td><td> $6 3 . 3 2 _ { \pm 1 1 . 2 0 }$ </td></tr><tr><td>GIN</td><td> $5 7 . 7 3 _ { \pm 5 . 2 1 }$ </td><td> $5 2 . 0 8 _ { \pm 3 . 1 0 }$ </td><td> $5 2 . 4 0 { \scriptstyle \pm 8 . 8 3 }$ </td><td> $4 7 . 8 4 _ { \pm 1 0 . 2 3 } ^ { - }$ </td><td> $5 6 . 5 5 { \scriptstyle \pm 6 . 7 2 }$ </td><td> $5 3 . 4 6 _ { \pm 4 . 7 2 }$ </td><td> $6 3 . 7 4 _ { \pm 2 1 . 6 8 }$ </td><td> $6 0 . 1 9 _ { \pm 1 8 . 0 8 }$ </td><td> $6 8 . 9 7 _ { \pm 8 . 8 4 }$ </td><td> $6 0 . 7 3 { \scriptstyle \pm 7 . 0 3 }$ </td></tr><tr><td>CORAL</td><td> $5 5 . 4 8 _ { \pm 6 . 1 2 }$ </td><td> $5 0 . 2 3 { \scriptstyle \pm 5 . 4 9 }$ </td><td> $5 3 . 7 5 _ { \pm 1 0 . 6 9 }$ </td><td> $5 0 . 0 6 _ { \pm 7 . 4 2 }$ </td><td>60.37±8.97</td><td> $5 7 . 1 8 _ { \pm 7 . 7 8 }$ </td><td> $6 3 . 8 8 _ { \pm 1 5 . 4 8 }$ </td><td> $5 9 . 8 6 _ { \pm 1 2 . 6 8 }$ </td><td> $6 9 . 4 3 _ { \pm 7 . 7 0 }$ </td><td> $6 2 . 5 5 _ { \pm 1 0 . 4 6 }$ </td></tr><tr><td rowspan="4">OOD</td><td>IRM</td><td> $5 8 . 5 6 _ { \pm 6 . 1 2 }$ </td><td> $5 2 . 9 8 _ { \pm 3 . 8 5 }$ </td><td> $5 1 . 7 7 _ { \pm 1 0 . 6 4 }$ </td><td> $5 0 . 8 4 _ { \pm 6 . 9 9 }$ </td><td> $5 8 . 5 6 _ { \pm 4 . 3 8 }$ </td><td> $5 4 . 1 8 _ { \pm 3 . 3 5 }$ </td><td> $6 0 . 9 8 _ { \pm 2 1 . 8 7 }$ </td><td> $5 9 . 2 7 _ { \pm 1 6 . 7 8 }$ </td><td> $7 0 . 3 7 _ { \pm 6 . 6 0 }$ </td><td> $5 9 . 0 6 _ { \pm 7 . 5 7 }$ </td></tr><tr><td>GSAT</td><td> $5 8 . 4 8 _ { \pm 7 . 6 2 }$ </td><td></td><td></td><td></td><td> $5 7 . 3 8 _ { \pm 5 . 7 7 }$ </td><td> $5 4 . 7 6 _ { \pm 4 . 5 2 }$ </td><td> $6 2 . 7 6 _ { \pm 1 8 . 9 7 }$ </td><td> $5 9 . 1 8 _ { \pm 1 5 . 7 8 }$ </td><td> $7 1 . 2 7 _ { \pm 8 . 7 3 }$ </td><td> $6 1 . 1 6 { \scriptstyle \pm 7 . 4 8 }$ </td></tr><tr><td>DisC</td><td> $5 6 . 4 6 _ { \pm 7 . 8 5 }$ </td><td> $5 3 . 6 1 { \scriptstyle \pm 7 . 9 8 }$   $5 3 . 6 2 _ { \pm 8 . 6 8 }$ </td><td> $5 0 . 8 1 _ { \pm 7 . 0 5 }$   $5 2 . 8 5 _ { \pm 6 . 4 8 }$ </td><td> $4 9 . 0 5 _ { \pm 6 . 2 7 }$   $5 0 . 7 5 _ { \pm 6 . 4 1 }$ </td><td> $5 6 . 1 9 _ { \pm 4 . 4 2 }$ </td><td> $5 3 . 2 7 _ { \pm 3 . 8 8 }$ </td><td> $6 5 . 1 8 _ { \pm 1 9 . 7 5 }$ </td><td> $6 1 . 4 6 _ { \pm 1 6 . 7 9 }$ </td><td> $6 9 . 5 9 _ { \pm 7 . 7 0 }$ </td><td> $5 8 . 8 3 _ { \pm 9 . 4 5 }$ </td></tr><tr><td>CEPG</td><td> $5 0 . 8 3 _ { \pm 8 . 9 9 }$ </td><td> $4 7 . 5 1 _ { \pm 6 . 1 2 }$ </td><td> $4 4 . 8 2 _ { \pm 9 . 7 4 }$ </td><td> $4 5 . 4 5 _ { \pm 8 . 9 3 }$ </td><td> $5 3 . 8 9 _ { \pm 5 . 2 7 }$ </td><td> $5 2 . 9 6 _ { \pm 4 . 4 6 }$ </td><td> $5 5 . 3 4 _ { \pm 8 . 6 4 }$ </td><td> $5 0 . 5 8 _ { \pm 6 . 2 8 }$ </td><td> $6 3 . 1 1 _ { \pm 1 1 . 0 6 }$ </td><td> $5 7 . 0 7 _ { \pm 9 . 5 9 }$ </td></tr><tr><td rowspan="8">Networks</td><td>DiSCO</td><td> $5 6 . 5 7 _ { \pm 6 . 7 2 }$ </td><td> $5 2 . 9 6 _ { \pm 4 . 8 3 }$ </td><td> $5 1 . 3 7 _ { \pm 8 . 4 5 }$ </td><td> $4 9 . 3 5 _ { \pm 6 . 2 1 }$ </td><td> $5 7 . 9 2 _ { \pm 3 . 7 3 }$ </td><td> $5 4 . 0 8 _ { \pm 2 . 8 4 }$ </td><td> $6 5 . 6 8 _ { \pm 1 8 . 1 6 }$ </td><td> $5 9 . 1 2 _ { \pm 1 5 . 4 6 }$ </td><td> $6 7 . 5 1 { \scriptstyle \pm 9 . 8 3 }$ </td><td> $6 1 . 9 8 _ { \pm 7 . 4 4 }$ </td></tr><tr><td>BrainNetTF</td><td> $6 2 . 6 4 _ { \pm 6 . 2 0 }$ </td><td></td><td></td><td> $5 4 . 2 5 _ { \pm 7 . 1 4 }$ </td><td> $6 1 . 4 8 _ { \pm 7 . 3 8 }$ </td><td> $5 8 . 1 7 { \scriptstyle \pm 5 . 5 8 }$ </td><td> $5 3 . 0 4 _ { \pm 1 7 . 0 2 }$ </td><td> $5 1 . 8 9 _ { \pm 1 0 . 7 8 }$ </td><td> $6 8 . 2 8 \scriptstyle \pm 4 . 5 4$ </td><td> $5 6 . 9 1 _ { \pm 1 0 . 4 3 }$ </td></tr><tr><td>XG-GNN</td><td> $6 0 . 2 0 _ { \pm 7 . 7 3 }$ </td><td> $5 8 . 3 8 { \scriptstyle \pm 7 . 0 6 }$  51.61±4.09</td><td>60.48±8.12  $5 5 . 9 4 _ { \pm 9 . 8 5 }$ </td><td> $5 3 . 6 4 _ { \pm 6 . 4 2 } ^ { - }$ </td><td>60.08±7.23</td><td></td><td>73.28±15.19</td><td> $6 6 . 1 7 _ { \pm 1 7 . 2 7 }$ </td><td> $7 1 . 2 0 _ { \pm 5 . 4 6 }$ </td><td>55.66±9.78</td></tr><tr><td>AGMGC</td><td> $5 8 . 4 7 _ { \pm 8 . 6 8 }$ </td><td> $4 8 . 9 7 _ { \pm 5 . 4 2 }$ </td><td> $5 9 . 3 2 _ { \pm 6 . 5 4 }$ </td><td> $5 3 . 1 7 _ { \pm 5 . 0 8 } ^ { - }$ </td><td> $5 9 . 0 6 _ { \pm 5 . 0 7 } ^ { - }$ </td><td> $5 6 . 2 4 _ { \pm 6 . 4 9 }$   $5 4 . 7 4 _ { \pm 4 . 1 2 }$ </td><td> $7 4 . 9 8 _ { \pm 1 5 . 3 1 } ^ { ^ { \scriptstyle \bigwedge } }$ </td><td> $6 7 . 3 4 _ { \pm 1 6 . 8 5 }$ </td><td> $7 0 . 8 6 _ { \pm 8 . 6 7 }$ </td><td> $6 3 . 0 2 _ { \pm 1 1 . 2 9 }$ </td></tr><tr><td>FC-HGNN</td><td> $5 5 . 0 5 _ { \pm 7 . 9 0 }$ </td><td> $4 9 . 7 1 { \scriptstyle \pm 4 . 5 8 }$ </td><td> $5 8 . 7 6 { \scriptstyle \pm 7 . 2 1 }$ </td><td> $5 4 . 8 2 _ { \pm 5 . 1 6 }$ </td><td> $5 8 . 6 9 _ { \pm 6 . 6 9 } ^ { - }$ </td><td> $5 6 . 7 8 \substack { \pm 5 . 7 1 }$ </td><td> $4 8 . 7 8 _ { \pm 1 0 . 5 8 }$ </td><td> $4 5 . 2 8 \substack { \pm 6 . 3 8 }$ </td><td> $7 1 . 0 8 { \scriptstyle \pm 5 . 7 4 }$ </td><td> $5 9 . 1 0 _ { \pm 1 2 . 3 4 }$ </td></tr><tr><td>BrainOOD</td><td> $6 3 . 6 1 _ { \pm 6 . 7 3 }$ </td><td> $5 7 . 1 4 _ { \pm 9 . 7 0 }$ </td><td> $6 6 . 1 5 _ { \pm 4 . 1 2 }$ </td><td> $5 8 . 2 1 _ { \pm 3 . 8 4 }$ </td><td> $6 3 . 2 2 _ { \pm 6 . 2 9 } ^ { - }$ </td><td> $5 8 . 4 5 _ { \pm 5 . 2 0 }$ </td><td> $8 1 . 4 8 _ { \pm 7 . 3 1 }$ </td><td> $6 9 . 1 8 _ { \pm 1 5 . 2 1 }$ </td><td> $7 2 . 7 7 _ { \pm 8 . 0 7 }$ </td><td> $6 4 . 3 6 _ { \pm 7 . 2 4 }$ </td></tr><tr><td>DeCI</td><td> $5 5 . 8 1 _ { \pm 8 . 9 1 } ^ { - }$ </td><td> $5 8 . 2 6 { \scriptstyle \pm 2 . 8 9 }$ </td><td> $6 2 . 1 8 _ { \pm 3 . 7 5 }$ </td><td> $5 9 . 4 8 _ { \pm 3 . 5 2 }$ </td><td> $5 4 . 3 8 _ { \pm 2 . 9 5 } ^ { - }$ </td><td> $5 6 . 1 8 _ { \pm 2 . 8 9 }$ </td><td> $8 2 . 0 5 _ { \pm 1 2 . 2 8 }$ </td><td> $7 6 . 6 9 _ { \pm 1 3 . 6 6 }$ </td><td> $5 5 . 0 3 _ { \pm 6 . 7 9 }$ </td><td> $6 2 . 8 1 _ { \pm 5 . 4 6 }$ </td></tr><tr><td>CORE</td><td> $6 5 . 3 2 _ { \pm 6 . 8 8 }$ </td><td> $6 0 . 9 7 { \scriptstyle \pm 5 . 9 0 }$ </td><td> $6 8 . 4 2 _ { \pm 6 . 3 5 }$ </td><td> $5 9 . 6 5 _ { \pm 5 . 4 2 }$ </td><td> $6 7 . 4 4 _ { \pm 8 . 2 0 }$ </td><td> $6 2 . 8 1 _ { \pm 6 . 0 4 }$ </td><td> $8 2 . 6 1 _ { \pm 8 . 2 5 }$ </td><td> $7 7 . 6 5 _ { \pm 9 . 8 4 }$ </td><td> $7 4 . 3 1 { \scriptstyle \pm 8 . 3 4 }$ </td><td> $6 7 . 2 6 { \scriptstyle \pm 7 . 5 3 }$ </td></tr><tr><td></td><td>BRIO</td><td> ${ \bf 6 7 . 5 3 _ { \pm 4 . 1 2 } }$ </td><td> $6 2 . 1 8 _ { \pm 3 . 6 8 }$  1</td><td> ${ \bf 7 0 . 8 6 _ { \pm 4 . 8 7 } }$ </td><td> ${ \bf 6 1 . 9 2 _ { \pm 3 . 9 4 } }$ </td><td> ${ \bf 6 8 . 8 4 _ { \pm 3 . 7 5 } }$ </td><td> $6 4 . 4 2 _ { \pm 2 . 9 1 }$ </td><td> ${ \bf 8 3 . 9 1 _ { \pm 5 . 6 3 } }$ </td><td> $7 8 . 8 2 _ { \pm 5 . 2 8 }$ </td><td> $7 5 . 9 8 _ { \pm 4 . 3 9 }$ </td><td> ${ \bf 6 8 . 7 4 _ { \pm 3 . 8 2 } }$ </td></tr></table>

## 4 EXPERIMENTS

Datasets. To evaluate the effectiveness of BRIO, we conduct LOSO experiments on four real-world fMRI datasets: ABIDE Di Martino et al. (2014), REST-meta-MDD Yan et al. (2019), SRPBS Tanaka et al. (2021), and ABCD Casey et al. (2018). These datasets cover prediction tasks related to autism spectrum disorder (ASD), major depressive disorder (MDD), and attention-deficit/hyperactivity disorder (ADHD), with ABCD providing a developmental cohort. We additionally include ABIDE (CC200), constructed using the CC200 functional parcellation, to examine the effect of atlas choice. More details of these datasets are provided in Appendix E.

Baselines. We compare BRIO with competitive baselines as follows: (1) General GNNs: GCN Kipf & Welling (2016), GAT Velickoviˇ c et al. (2017), and GIN Xu et al. (2018); (2) General OOD methods:´ CORAL Sun & Saenko (2016) and IRM Arjovsky et al. (2019); (3) Graph OOD methods: GSAT Miao et al. (2022), DisC Fan et al. (2022), CEPG Wang et al. (2026a), and DiSCO Sun et al. (2026); (4) Brain network methods: BrainNetTF Kan et al. (2022), XG-GNN Qiu et al. (2024), AGMGC Noman et al. (2025), FC-HGNN Gu et al. (2025), BrainOOD Xu et al. (2025), DeCI Yu et al. (2026) and CORE Wang et al. (2026d). More details of baselines are provided in Appendix F.

## 4.1 PERFORMANCE COMPARISON

We compare BRIO with baseline methods under the LOSO protocol on ABIDE, REST-meta-MDD, SRPBS, and ABCD in Table 1. We observe that: (1) General OOD methods do not consistently outperform general GNNs across datasets, suggesting that generic robustness objectives offer limited gains in cross-site brain network generalization. (2) The best-performing brain network methods consistently outperform graph OOD methods, while the gains from other brain network models vary across datasets. This suggests that cross-site generalization benefits not only from modeling brain con nectivity but also from learning task-relevant patterns that transfer across sites. (3) BRIO consistently outperforms all baselines across datasets. These gains can be attributed to two complementary mecha nisms: (i) shared connectome factorization enables consistent comparisons of predictive contributions across FC estimates, while task-calibrated qualification identifies task-relevant factors supported under re-estimation; (ii) qualification-guided alignment prioritizes factors supported by both sites and adjusts the overall alignment strength accordingly, limiting the influence of weakly supported factors on cross-site matching. We further evaluate performance on ABIDE using CC200 (Craddock et al., 2012) instead of AAL. Although baseline rankings vary between the two parcellations, BRIO outperforms all baselines on both metrics, indicating that its gains persist across parcellations.

## 4.2 ABLATION STUDY

We evaluate four ablation variants: (1) BRIO w/o RS uses task relevance alone to weight factors and fixes the strength multiplier to one; (2) BRIO w/o TR replaces explicit task relevance weights with uniform weights while retaining task-calibrated re-estimation support; (3) BRIO w/o SM retains qualification-based factor weights but fixes the strength multiplier to one; and (4) BRIO w/o SF uses task relevance alone to weight factors while retaining qualification-based strength modulation. As shown in Fig. 2(a,b), removing RS reduces performance, indicating that assessing predictive contributions under FC re-estimation provides benefits beyond full-scan task relevance. Removing TR degrades performance, highlighting the value of prioritizing discriminative factors rather than relying on re-estimation support alone. The performance drops without SM and SF support the complementary roles of qualification in modulating alignment strength for each site pair and class and prioritizing supported factors in cross-site matching. More results are shown in Appendix J.4.

![](images/6cd8ccc9db2791347c7dea30be1fb3808e23610b7f0737a17f622cb8500096a4.jpg)  
(a) AUC

![](images/11af5b4338c8591bd24302cf852584efa4debcfea619d4525d8cb089e4e14e5a.jpg)  
(b) ACC

![](images/09e12b126390ca02572d513c9bf5a99fe295aa02bef549bcdefee104f723d022.jpg)  
(c) ABIDE

![](images/55af27cc72846080aa2ebb1d1c020ada383e19d87d95d0bb3da5d60c722340b1.jpg)  
(d) REST-meta-MDD

Figure 2: Ablation results in (a,b) and sensitivity results in (c,d) on ABIDE and REST-meta-MDD.  
![](images/a3c6c22484e05de25a61eda85e619901840662dbde7d7a836e50a4b8ca0c662e.jpg)  
(a) Cross-method

![](images/e6b4112f8668bbe459b2affe17bb7f904822cff053f370cb1ff2ae6156c7774b.jpg)  
(b) Alignment

![](images/4fcbcd27509a008cd9e906bf9e75645ab470a24234382cf40674a0800462593c.jpg)  
(c) Mismatch

![](images/0d5bffc9e3d63c50f608e09a0d2678b62e61b4ad5d0b4c8f17f63034bb87c072.jpg)  
(d) Factor retention  
Figure 3: Cross-method relationships (a), alignment trajectories (b), mismatch rates (c), and factor retention (d) on ABIDE.

## 4.3 SENSITIVITY STUDY

## 4.4 CASE STUDY

We examine the sensitivity of BRIO to the number of FC re-estimates K and connectome factors J. Fig. 2(c,d) shows that increasing K from small values generally improves AUC on ABIDE and REST-meta-MDD, whereas further increases yield no consistent gains. This suggests that additional re-estimates help characterize variation in predictive contributions, with diminishing benefits at larger budgets. We vary J with the total representation dimension fixed. Intermediate values perform better than very small or large values on both datasets, suggesting a trade-off between separating connectivity patterns and preserving sufficient capacity within each factor. Too few factors may combine patterns with distinct predictive roles and re-estimation responses, whereas too many reduce each factor’s dimension and may limit its discriminative capacity. More results are shown in Appendix J.5.

Site Discrepancy and Re-estimation Variation. To examine whether lower source-site discrepancy implies lower predictive variation under FC re-estimation, we conduct cross-method comparisons and a controlled alignment-strength sweep, the latter using BRIO w/o QG, which removes qualification guidance by assigning equal weights to all factors and disabling qualification-based strength modulation, trained independently for each tested $\lambda _ { \mathrm { e n v } }$ . We measure class-conditional full-scan site discrepancy $\mathcal { D } _ { \mathrm { s i t e } }$ in Eq. (83) and normalized predictive re-estimation variation $\mathcal { \widetilde { D } } _ { \mathrm { r e } } ^ { \mathrm { p r e d } }$ in Eq. (84), and summarize their relation across methods by Spearman $\rho ,$ the discordance rate $R _ { \mathrm { d i s c o r d } } ,$ and the total and AUC-preserved mismatch rates defined in Appendix J.2. Fig. 3(a) shows that lower $\mathcal { D } _ { \mathrm { s i t e } }$ does not consistently correspond to lower $\mathcal { \widetilde { D } } _ { \mathrm { r e } } ^ { \mathrm { p r e d } }$ across methods, so cross-site agreement alone does not imply stable within-subject predictions. In the controlled sweep in Fig. 3(b), stronger alignment reduces both quantities, while the full BRIO achieves lower $\mathcal { \widetilde { D } } _ { \mathrm { r e } } ^ { \mathrm { p r e d } }$ than the w/o QG setting at comparable $\mathcal { D } _ { \mathrm { s i t e } } .$ , indicating that qualification-guided weighting and strength modulation offer benefits beyond simply increasing alignment strength. Beyond these average trends, Fig. 3(c) shows that BRIO achieves lower total and AUC-preserved mismatch rates than the evaluated w/o QG settings, with fewer mismatches even when predictive performance is preserved.

Qualification and Factor Transferability. To examine whether source-side qualification identifies transferable factors, we conduct a factor-retention experiment on ABIDE using a fixed, trained BRIO model. We rank factors by full qualification, task relevance alone, or re-estimation support alone, with random ranking as a reference. Qualification and support scores are averaged over source-site pairs and classes before ranking. For each ranking, we retain varying proportions of the highest-ranked factors and evaluate AUC on held-out sites using the original classifier without retraining, allowing direct comparisons of factor selection strategies. Fig. 3(d) shows that full qualification yields the highest AUC at partial retention, with larger gains when fewer factors are retained. Its advantage over task relevance alone suggests that re-estimation support provides complementary information about factor transferability, while the weaker performance of support alone highlights the importance of considering both criteria jointly. As more factors are retained, the performance gaps narrow and disappear at full retention, where all rankings recover the original model predictions.

## 5 CONCLUSION

To address finite-sample FC estimation variability in cross-site brain network generalization, we proposed BRIO. BRIO qualifies predictive evidence under within-scan FC re-estimation before aligning it across sites. Consistently indexed connectome factors enable comparison of predictive contributions across estimates. Calibrated changes in these contributions, combined with predictive relevance, determine which factors to align and how strongly. Experiments on four real-world datasets show consistent gains over competitive baselines. These findings support estimation stability as a criterion for what to align rather than only how much to align, and motivate future research on adaptive re-estimation budgets and dynamic or multimodal connectomes.

## REFERENCES

Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

Pierre Bellec, Guillaume Marrelec, and Habib Benali. A bootstrap test to investigate changes in brain connectivity for functional mri. Statistica Sinica, pp. 1253–1268, 2008.

Alaa Bessadok, Mohamed Ali Mahjoub, and Islem Rekik. Graph neural networks in network neuroscience. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(5):5833–5848, 2022.

Rasmus M Birn, Erin K Molloy, Rémi Patriat, Taurean Parker, Timothy B Meier, Gregory R Kirk, Veena A Nair, M Elizabeth Meyerand, and Vivek Prabhakaran. The effect of scan length on the reliability of resting-state fmri connectivity estimates. Neuroimage, 83:550–558, 2013.

Betty Jo Casey, Tariq Cannonier, May I Conley, Alexandra O Cohen, Deanna M Barch, Mary M Heitzeg, Mary E Soules, Theresa Teslovich, Danielle V Dellarco, Hugh Garavan, et al. The adolescent brain cognitive development (abcd) study: imaging acquisition across 21 sites. Developmental cognitive neuroscience, 32:43–54, 2018.

Andrew A. Chen, Joanne C. Beer, N. Tustison, Philip A Cook, Russell T. Shinohara, and Haochang Shou. Mitigating site effects in covariance for machine learning in neuroimaging data. Human Brain Mapping, 43:1179 – 1195, 2021.

Shuo Chen, Jian Kang, and Guoqing Wang. An empirical bayes normalization method for connectivity metrics in resting state fmri. Frontiers in neuroscience, 9:316, 2015.

Yongqiang Chen, Yonggang Zhang, Yatao Bian, Han Yang, MA Kaili, Binghui Xie, Tongliang Liu, Bo Han, and James Cheng. Learning causally invariant representations for out-of-distribution generalization on graphs. Proceedings of the Conference on Neural Information Processing Systems, 35:22131–22148, 2022.

R Cameron Craddock, G Andrew James, Paul E Holtzheimer III, Xiaoping P Hu, and Helen S Mayberg. A whole brain fmri atlas generated via spatially constrained spectral clustering. Human brain mapping, 33(8):1914–1928, 2012.

Hejie Cui, Wei Dai, Yanqiao Zhu, Xiaoxiao Li, Lifang He, and Carl Yang. Interpretable graph neural networks for connectome-based brain disorder analysis. In International conference on medical image computing and computer-assisted intervention, pp. 375–385. Springer, 2022.

Marco Cuturi. Sinkhorn distances: Lightspeed computation of optimal transport. Proceedings ofthe Conference on Neural Information Processing Systems, 26, 2013.

Adriana Di Martino, Chao-Gan Yan, Qingyang Li, Erin Denio, Francisco X Castellanos, Kaat Alaerts, Jeffrey S Anderson, Michal Assaf, Susan Y Bookheimer, Mirella Dapretto, et al. The autism brain imaging data exchange: towards a large-scale evaluation of the intrinsic brain architecture in autism. Molecular psychiatry, 19(6):659–667, 2014.

Shaohua Fan, Xiao Wang, Yanhu Mo, Chuan Shi, and Jian Tang. Debiasing graph neural networks via learning disentangled causal substructure. Proceedings of the Conference on Neural Information Processing Systems, 35:24934–24946, 2022.

Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, and Gabriel Peyré. Interpolating between optimal transport and mmd using sinkhorn divergences. In Proceedings ofthe International Conference on Artificial Intelligence and Statistics, pp. 2681–2690. PMLR, 2019.

Emily S Finn, Xilin Shen, Dustin Scheinost, Monica D Rosenberg, Jessica Huang, Marvin M Chun, Xenophon Papademetris, and R Todd Constable. Functional connectome fingerprinting: identifying individuals using patterns of brain connectivity. Nature neuroscience, 18(11):1664–1671, 2015.

Peyré Gabriel and Cuturi Marco. Computational optimal transport with applications to data sciences. Foundations and trends® in machine learning, 11(5-6):355–607, 2019.

Yuheng Gu, Shoubo Peng, Yaqin Li, Linlin Gao, and Yihong Dong. Fc-hgnn: A heterogeneous graph neural network based on brain functional connectivity for mental disorder identification. Information Fusion, 113:102619, 2025.

Keqi Han, Yao Su, Lifang He, Liang Zhan, Sergey Plis, Vince Calhoun, and Carl Yang. Rethinking functional brain connectome analysis: do graph deep learning models help. npj Artificial Intelligence, 2(1):19, 2026.

Niko Huotari, Lauri Raitamaa, Heta Helakari, Janne Kananen, Ville Raatikainen, Aleksi Rasila, Timo Tuovinen, Jussi Kantola, Viola Borchardt, Vesa J Kiviniemi, et al. Sampling rate effects on resting state fmri metrics. Frontiers in neuroscience, 13:279, 2019.

Xuan Kan, Wei Dai, Hejie Cui, Zilong Zhang, Ying Guo, and Carl Yang. Brain network transformer. Proceedings of the Conference on Neural Information Processing Systems, 35:25586–25599, 2022.

Thomas N Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016.

Maria Kudela, Jaroslaw Harezlak, and Martin A Lindquist. Assessing uncertainty in dynamic functional connectivity. NeuroImage, 149:165–177, 2017.

Charalampos Lamprou, Aamna Alshehhi, Leontios J Hadjileontiadis, and Mohamed L Seghier. Varconet: A variability-aware self-supervised framework for functional connectome extraction from resting-state fmri. Human Brain Mapping, 47(4):e70469, 2026.

Haoyang Li, Xin Wang, Ziwei Zhang, and Wenwu Zhu. Out-of-distribution generalization on graphs: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Xiaoxiao Li, Yuan Zhou, Nicha Dvornek, Muhan Zhang, Siyuan Gao, Juntang Zhuang, Dustin Scheinost, Lawrence H Staib, Pamela Ventola, and James S Duncan. Braingnn: Interpretable brain graph neural network for fmri analysis. Medical image analysis, 74:102233, 2021.

Siqi Miao, Mia Liu, and Pan Li. Interpretable and generalizable graph learning via stochastic attention mechanism. In Proceedings of the International Conference on Machine Learning, pp. 15524–15543. PMLR, 2022.

Stephanie Noble, Dustin Scheinost, and R Todd Constable. A decade of test-retest reliability of functional connectivity: A systematic review and meta-analysis. Neuroimage, 203:116157, 2019.

Fuad Noman, Raphaël C-W Phan, Hernando Ombao, and Chee-Ming Ting. Adaptive graph learning with multi-graph convolutions for brain disorder classification. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 56–65. Springer, 2025.

Mansooreh Pakravan. Uncertainty-aware type-ii fuzzy graph modeling of resting-state fmri uncovers robust sex differences. Journal ofNeuroscience Methods, pp. 110745, 2026.

Xinmei Qiu, Fan Wang, Yongheng Sun, Chunfeng Lian, and Jianhua Ma. Towards graph neural networks with domain-generalizable explainability for fmri-based brain disorder diagnosis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 454–464. Springer, 2024.

Xinmei Qiu, Yongheng Sun, Yilin Shi, Xujun Duan, Fan Wang, and Jianhua Ma. Metaexplainer: Revisit domain generalization of functional connectome analyses from the perspective of explainability. Medical Image Analysis, 105:103664, 2025.

Qi-Man Shao and Hao Yu. Bootstrapping the sample means for stationary mixing sequences. Stochastic Processes and their Applications, 48(1):175–190, 1993.

Richard Sinkhorn and Paul Knopp. Concerning nonnegative matrices and doubly stochastic matrices. Pacific Journal ofMathematics, 21(2):343–348, 1967.

Stephen M Smith, Karla L Miller, Gholamreza Salimi-Khorshidi, Matthew Webster, Christian F Beckmann, Thomas E Nichols, Joseph D Ramsey, and Mark W Woolrich. Network modelling methods for fmri. Neuroimage, 54(2):875–891, 2011.

Baochen Sun and Kate Saenko. Deep coral: Correlation alignment for deep domain adaptation. In ECCV Workshops, 2016.

Jerry Sun, Mohamed Abubakr Hassan, Yaoyu Zhang, Wanying Zhang, and Chi-Guhn Lee. Diverse and sparse mixture-of-experts for causal subgraph–based out-of-distribution graph learning. In Proceedings of the International Conference on Learning Representations, volume 2026, pp. 79573–79597, 2026.

Saori C Tanaka, Ayumu Yamashita, Noriaki Yahata, Takashi Itahashi, Giuseppe Lisi, Takashi Yamada, Naho Ichikawa, Masahiro Takamura, Yujiro Yoshihara, Akira Kunimatsu, et al. A multi-site, multi-disorder resting-state magnetic resonance image database. Scientific data, 8(1):227, 2021.

Nathalie Tzourio-Mazoyer, Brigitte Landeau, Dimitri Papathanassiou, Fabrice Crivello, Octave Etard, Nicolas Delcroix, Bernard Mazoyer, and Marc Joliot. Automated anatomical labeling of activations in spm using a macroscopic anatomical parcellation of the mni mri single-subject brain. Neuroimage, 15(1):273–289, 2002.

Petar Velickoviˇ c, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua´ Bengio. Graph attention networks. arXiv preprint arXiv:1710.10903, 2017.

Qixun Wang, Yifei Wang, Yisen Wang, and Xianghua Ying. Dissecting the failure of invariant learning on graphs. Proceedings ofthe Conference on Neural Information Processing Systems, 37: 80383–80438, 2024.

Shuo Wang, Mingchen Sun, Qiang Huang, and Ying Wang. Environment promoted invariant information learning for graph out-of-distribution generalization. Artificial Intelligence, pp. 104522, 2026a.

Yingxu Wang, Kunyu Zhang, Jiaxin Huang, Nan Yin, Siwei Liu, and Eran Segal. Protomol: enhancing molecular property prediction via prototype-guided multimodal learning. Briefings in Bioinformatics, 26(6):bbaf629, 2025.

Yingxu Wang, Victor Liang, Nan Yin, Siwei Liu, and Eran Segal. Sgac: a graph neural network framework for imbalanced and structure-aware amp classification. Briefings in Bioinformatics, 27 (1):bbag038, 2026b.

Yingxu Wang, Kunyu Zhang, Mengzhu Wang, Siyang Gao, and Nan Yin. Usbd: Universal structural basis distillation for source-free graph domain adaptation. arXiv preprint arXiv:2602.08431, 2026c.

Yingxu Wang, Kunyu Zhang, Yanwu Yang, Thomas Wolfers, Yujie Wu, Siyang Gao, and Nan Yin. When brain networks travel: Learning beyond site. arXiv preprint arXiv:2605.06050, 2026d.

Yongli Xiang, Ziming Hong, Lina Yao, Dadong Wang, and Tongliang Liu. Jailbreaking the nontransferable barrier via test-time data disguising. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 30671–30681. IEEE, 2025.

Yongli Xiang, Ziming Hong, Zhaoqing Wang, Xiangyu Zhao, Bo Han, and Tongliang Liu. When safety collides: Resolving multi-category harmful conflicts in text-to-image diffusion via adaptive safety guidance. arXiv preprint arXiv:2602.20880, 2026a.

Yongli Xiang, Zhifang Zhang, Bojun Yang, Ziming Hong, Lei Feng, Miao Xu, and Tongliang Liu. When agents learn to be you: Benchmarking privacy leakage, impersonation risk, and defenses in persona skills. arXiv preprint arXiv:2608.03700, 2026b.

Jiaxing Xu, Yongqiang Chen, Xia Dong, Mengcheng Lan, Tiancheng HUANG, Qingtian Bian, James Cheng, and Yiping Ke. BrainOOD: Out-of-distribution generalizable brain network analysis. In Proceedings of the International Conference on Learning Representations, 2025.

Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826, 2018.

Okito Yamashita, Ayumu Yamashita, Yuji Takahara, Yuki Sakai, Yasumasa Okamoto, Go Okada, Masahiro Takamura, Motoaki Nakamura, Takashi Itahashi, Takashi Hanakawa, et al. Computational mechanisms of neuroimaging biomarkers uncovered by multicenter resting-state fmri connectivity variation profile. Molecular Psychiatry, 30(11):5463–5474, 2025.

Chao-Gan Yan, Xiao Chen, Le Li, Francisco Xavier Castellanos, Tong-Jian Bai, Qi-Jing Bo, Jun Cao, Guan-Mao Chen, Ning-Xuan Chen, Wei Chen, et al. Reduced default mode network functional connectivity in patients with recurrent major depressive disorder. Proceedings of the National Academy ofSciences, 116(18):9078–9083, 2019.

Liang Yang, Shuai Zhai, Ziyi Ma, Jiaming Zhuo, Di Jin, Chuan Wang, Zhen Wang, and Xiaochun Cao. Brain networks should be learned, not constructed. In Proceedings of the International Conference on Machine Learning, 2026.

Guoqi Yu, Xiaowei Hu, Angelica I Aviles-Rivero, Anqi Qiu, and Shujun Wang. Moving beyond functional connectivity: Time-series modeling for fmri-based brain disorder classification. IEEE Transactions on Medical Imaging, 2026.

Meichen Yu, Kristin A Linn, Philip A Cook, Mary L Phillips, Melvin McInnis, Maurizio Fava, Madhukar H Trivedi, Myrna M Weissman, Russell T Shinohara, and Yvette I Sheline. Statistical harmonization corrects site effects in functional connectivity measurements from multi-site fmri data. Human brain mapping, 39(11):4213–4227, 2018.

Minqi Yu, Jinduo Liu, and Junzhong Ji. Causal invariance-aware augmentation for brain graph contrastive learning. In Proceedings ofthe International Conference on Machine Learning, 2025.

Kunyu Zhang and Tianxiang Xu. Brainriem: Riemannian prototype learning for source-free cross-site brain network diagnosis. In European Conference on Computer Vision, pp. 308–325. Springer, 2026.

Kunyu Zhang, Qiang Li, and Shujian Yu. Mvho-ib: Multi-view higher-order information bottleneck for brain disorder diagnosis. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 407–417. Springer, 2025.

Kunyu Zhang, Qiang Li, Vince D Calhoun, and Shujian Yu. Modeling higher-order brain interactions via a multi-view information bottleneck framework for fmri-based psychiatric diagnosis. arXiv preprint arXiv:2604.17713, 2026.

Table 2: Summary of key notations.
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $\mathcal { E } _ { \mathrm { t r } } , \mathcal { E } _ { \mathrm { t e } } , E$ </td><td>Source and unseen test site sets, and the number of source sites.</td></tr><tr><td> $X _ { i } ^ { e } , Y _ { i } ^ { e }$ </td><td>Preprocessed BOLD sequence and binary label of subject i at site e.</td></tr><tr><td> $T _ { i } ^ { e } , P$ </td><td>Numbers of valid time points and ROIs.</td></tr><tr><td> $\mathcal { T } _ { e , c } , N _ { e , c }$ </td><td>Subject index set for site e and class  $c ,$  and its size.</td></tr><tr><td> $\scriptstyle B _ { e , c }$ </td><td>Training mini-batch subset for site e and class c.</td></tr><tr><td> $\Psi , \mathcal { R } _ { k } , K$ </td><td>FC estimator, joint circular moving-block resampling operator, and number of FC re- estimates per subject.</td></tr><tr><td> $\ell _ { b } , \ell _ { b , i } ^ { e }$ </td><td>Fold-level block length estimated from training subjects, and its per-sequence clipped</td></tr><tr><td> $\tau _ { \mathrm { a c } } , \ell _ { \mathrm { m i n } } , \ell _ { \mathrm { m a x } }$ </td><td>value, in time points. Autocorrelation threshold and clipping bounds of the block-length rule.</td></tr><tr><td> $A _ { i } ^ { e , ( 0 ) } , A _ { i } ^ { e , ( k ) }$ </td><td>Full-scan FC graph and its k-th re-estimate, with  $k \geq 1$ </td></tr><tr><td> $s , \rho _ { g }$ </td><td>Fixed source-derived binary propagation support and per-ROI neighbor selection ratio.</td></tr><tr><td> $G _ { i } ^ { e , ( k ) } , G _ { i } ^ { e , ( k ) , \pm }$ </td><td>Supported FC adjacency and its positive weights (+) and negative-weight magnitudes</td></tr><tr><td> $E ^ { \mathrm { R O I } } , H _ { i } ^ { e , ( k ) }$ </td><td>(−). Shared ROI identity embeddings and ROI representations produced by the shared signed GNN.</td></tr><tr><td> $J , Q$ </td><td>Number of connectome factors and learnable factor queries.</td></tr><tr><td> $\Pi , \overline { { \Pi } }$ </td><td>Shared balanced factor-ROI assignment and its cached value fixed after supervised</td></tr><tr><td> $\pmb { r } _ { i \textit { i } } ^ { e , ( k ) }$ </td><td>warm-up.  $k .$ </td></tr><tr><td> ${ \pmb s } _ { i } ^ { e , ( k ) }$ </td><td>Unit-norm representation of factor  $j$  for subject i at site e and FC estimate</td></tr><tr><td></td><td>Graph representation formed by scaled factor concatenation in a fixed order. Classifier weight block for factor j and the prediction bias.</td></tr><tr><td> $w _ { j } , b$ </td><td></td></tr><tr><td> $a _ { i . i } ^ { e , ( k ) }$ </td><td>Factor score  ${ \pmb w } _ { j } ^ { \top } { \pmb r } _ { i , j } ^ { e , ( k ) }$  before the common  $1 / \sqrt { J }$  scaling.</td></tr><tr><td> $z _ { i } ^ { e , ( k ) } , \widehat { p } _ { i } ^ { e , ( 0 ) }$ </td><td>Prediction logit for FC estimate k and full-scan prediction probability.</td></tr><tr><td> $\mu _ { e , c , j }$ </td><td>Mean full-scan contribution of factor j for site e and class c.</td></tr><tr><td> $V _ { e , c , j } ^ { \mathrm { s u b } } , V _ { c , j } ^ { \mathrm { p o o l } }$ </td><td>Within-class contribution variance at a site and its equal-site average.</td></tr><tr><td> $\Delta _ { e , j } , \Delta _ { j }$ </td><td>Full-scan class-mean contrast at a site and its signed equal-site average.</td></tr><tr><td> $\beta _ { j } , R _ { j } ^ { \mathrm { t a s k } } , \pi _ { j } ^ { \mathrm { t a s k } }$ </td><td>Relative squared classifier-block norm, unnormalized predictive relevance, and its nor- malization across factors.</td></tr><tr><td> $D _ { e , c , j } ^ { \mathrm { r e } }$ </td><td>Mean squared difference between paired re-estimated and full-scan contributions.</td></tr><tr><td> $M _ { e , j } ^ { \mathrm { t a s k } } , T _ { e , c , j }$   $q _ { e , c , j } ^ { \mathrm { r e } } , \kappa _ { e , c , j }$ </td><td>Squared half-separation  $\Delta _ { e , j } ^ { 2 } / 4$  and local reference scale  $V _ { e , c , j } ^ { \mathrm { s u b } } + M _ { e , j } ^ { \mathrm { t a s k } } ,$  Task-calibrated re-estimation support and full-scan separation ratio for factor j at site e</td></tr><tr><td></td><td>and class  $c .$ </td></tr><tr><td> $q _ { e , e ^ { \prime } , c , j } ^ { \mathrm { p a i r } }$ </td><td>Geometric mean of factor 4  $j ^ { \circ } \mathbf { s }$  re-estimation supports at sites e and  $e ^ { \prime }$  for class  $c .$  Unnormalized pairwise qualification and its normalization determining the relative factor</td></tr><tr><td> $\alpha _ { e , e ^ { \prime } , c , j } , \overline { { \alpha } } _ { e , e ^ { \prime } , c , j }$ </td><td>weight in alignment.</td></tr><tr><td> $g _ { e , e ^ { \prime } , c }$ </td><td>Total pairwise qualification scaling alignment for sites  $\boldsymbol { e } , \boldsymbol { e } ^ { \prime }$  and class c.</td></tr><tr><td></td><td>Pair-specific graph representation at site  $d \in \{ e , e ^ { \prime } \}$  , formed from reweighted full-scan</td></tr><tr><td> $\varphi _ { i \left| e , e ^ { \prime } , c \right. } ^ { \left. \mathrm { ~ e ~ } , \mathrm { ~ e ~ } \right.} $ </td><td>factors.</td></tr><tr><td> $\widehat { P } _ { e , c } ^ { Q [ e , e ^ { \prime } ] }$ </td><td>Empirical class-conditional distribution at site e in the common pairwise weighted space.</td></tr><tr><td> $S _ { \varepsilon } , \varepsilon$ </td><td>Debiased Sinkhorn divergence and its entropic regularization parameter.</td></tr></table>

## A NOTATION SUMMARY

As shown in the Table 2, we summarize the key notations of this paper.

## B PROOF OF PROPOSITION 1

Proposition 1 (Class-Contrast Stability under FC Re-estimation) Fix the model snapshot, a source site e, and afactor j, with $N _ { e , 0 } , N _ { e , 1 } > 0$ and $K \geq 1$ . Let $\Delta _ { e , j } ^ { \mathrm { r e } }$ be the class contrast ofcontributions averaged over subjects and FC re-estimates, and let $\kappa _ { e , c , j } = \bar { \Delta } _ { e , j } ^ { 2 } / \big ( 4 ( V _ { e , c , j } ^ { \mathrm { s u b } } + \epsilon _ { q } ) \big )$ be thefull-scan

separation ratio ofclass c. Then

$$
\left| \Delta _ { e , j } ^ { \mathrm { r e } } - \Delta _ { e , j } \right| \leq \sum _ { c \in \{ 0 , 1 \} } \sqrt { ( T _ { e , c , j } + \epsilon _ { q } ) \left( \frac { 1 } { q _ { e , c , j } ^ { \mathrm { r e } } } - 1 \right) } ,\tag{28}
$$

and $\Delta _ { e , j } ^ { \mathrm { r e } }$ has the same sign as $\Delta _ { e , j }$ whenever $q _ { e , c , j } ^ { \mathrm { r e } } > ( 1 + \kappa _ { e , c , j } ) / ( 1 + 2 \kappa _ { e , c , j } )$ for both $c \in \{ 0 , 1 \}$ Proof.

For each class $c \in \{ 0 , 1 \}$ , the mean re-estimated contribution and the resulting class contrast are

$$
\mu _ { e , c , j } ^ { \mathrm { r e } } = \frac { 1 } { N _ { e , c } K } \sum _ { i \in \mathcal { T } _ { e , c } } \sum _ { k = 1 } ^ { K } a _ { i , j } ^ { e , ( k ) } , \quad \Delta _ { e , j } ^ { \mathrm { r e } } = \mu _ { e , 1 , j } ^ { \mathrm { r e } } - \mu _ { e , 0 , j } ^ { \mathrm { r e } } .\tag{29}
$$

Let $\xi _ { i , j } ^ { e , ( k ) } = a _ { i , j } ^ { e , ( k ) } - a _ { i , j } ^ { e , ( 0 ) }$ denote the contribution change between the k-th FC re-estimate and the full-scan estimate. Repeating each full-scan contribution across the $K$ paired comparisons gives

$$
\mu _ { e , c , j } ^ { \mathrm { r e } } - \mu _ { e , c , j } = \frac { 1 } { N _ { e , c } K } \sum _ { i \in \mathcal { T } _ { e , c } } \sum _ { k = 1 } ^ { K } \xi _ { i , j } ^ { e , ( k ) } .\tag{30}
$$

Applying the Cauchy–Schwarz inequality to these $N _ { e , c } K$ paired changes yields

$$
\begin{array} { r l r } {  { \big | \mu _ { e , c , j } ^ { \mathrm { r e } } - \mu _ { e , c , j } \big | ^ { 2 } = \frac { 1 } { ( N _ { e , c } K ) ^ { 2 } } ( \sum _ { i \in \mathcal { T } _ { e , c } } \sum _ { k = 1 } ^ { K } \xi _ { i , j } ^ { e , ( k ) } ) ^ { 2 } } } \\ & { } & { \leq \frac { 1 } { N _ { e , c } K } \sum _ { i \in \mathcal { T } _ { e , c } } \sum _ { k = 1 } ^ { K } \Big ( \xi _ { i , j } ^ { e , ( k ) } \Big ) ^ { 2 } = D _ { e , c , j } ^ { \mathrm { r e } } , } \end{array}\tag{31}
$$

where the final equality follows from Eq. (14), and equality holds if and only if $\xi _ { i , j } ^ { e , ( k ) }$ is constant over $i \in \mathcal { T } _ { e , c }$ and $k ,$ that ${ \mathrm { i s } } ,$ when re-estimation shifts all class-c contributions by a common constant. Since the change in the class contrast is the difference between the two class-mean shifts, taking square roots and applying the triangle inequality gives

$$
\begin{array} { r l } & { \left| \Delta _ { e , j } ^ { \mathrm { r e } } - \Delta _ { e , j } \right| = \left| \left( \mu _ { e , 1 , j } ^ { \mathrm { r e } } - \mu _ { e , 1 , j } \right) - \left( \mu _ { e , 0 , j } ^ { \mathrm { r e } } - \mu _ { e , 0 , j } \right) \right| } \\ & { \qquad \leq \left| \mu _ { e , 1 , j } ^ { \mathrm { r e } } - \mu _ { e , 1 , j } \right| + \left| \mu _ { e , 0 , j } ^ { \mathrm { r e } } - \mu _ { e , 0 , j } \right| \leq \displaystyle \sum _ { c \in \{ 0 , 1 \} } \sqrt { D _ { e , c , j } ^ { \mathrm { r e } } } . } \end{array}\tag{32}
$$

To express this bound through re-estimation support, let $m _ { e , j } = ( \mu _ { e , 0 , j } + \mu _ { e , 1 , j } ) / 2$ denote the midpoint of the full-scan class means, so that $\mu _ { e , c , j } - m _ { e , j } = ( 2 c - 1 ) \Delta _ { e , j } / 2$ for either class. Expanding the squared distance from this midpoint around the class mean yields

$$
\begin{array} { r l } & { \frac { 1 } { N _ { e , c } } \displaystyle \sum _ { i \in \mathcal { Z } _ { e , c } } \left( a _ { i , j } ^ { e , ( 0 ) } - m _ { e , j } \right) ^ { 2 } = \frac { 1 } { N _ { e , c } } \displaystyle \sum _ { i \in \mathcal { Z } _ { e , c } } \left[ \left( a _ { i , j } ^ { e , ( 0 ) } - \mu _ { e , c , j } \right) + \left( \mu _ { e , c , j } - m _ { e , j } \right) \right] ^ { 2 } } \\ & { \quad \quad \quad \quad = V _ { e , c , j } ^ { \mathrm { s u b } } + \frac { 1 } { 4 } \Delta _ { e , j } ^ { 2 } = T _ { e , c , j } , } \end{array}\tag{33}
$$

where the cross term vanishes because $\textstyle \sum _ { i \in \mathcal { T } _ { e . c } } ( a _ { i , j } ^ { e , ( 0 ) } - \mu _ { e , c , j } ) = 0$ . Thus $T _ { e , c , j } \geq 0$ is the empirical mean squared distance of class-c full-scan contributions from the midpoint of the two class means. Together with $D _ { e , c , j } ^ { \mathrm { r e } } \geq 0$ and $\epsilon _ { q } > 0$ , this ensures $0 < q _ { e , c , j } ^ { \mathrm { r e } } \leq 1$ , and rearranging Eq. (16) gives

$$
\left( T _ { e , c , j } + \epsilon _ { q } \right) \left( \frac { 1 } { q _ { e , c , j } ^ { \mathrm { r e } } } - 1 \right) = \left( T _ { e , c , j } + \epsilon _ { q } \right) \frac { D _ { e , c , j } ^ { \mathrm { r e } } } { T _ { e , c , j } + \epsilon _ { q } } = D _ { e , c , j } ^ { \mathrm { r e } } .\tag{34}
$$

Substituting Eq. (34) into $\operatorname { E q } .$ . (32) establishes Eq. (17).

For sign preservation, we first show that the support threshold is equivalent to a bound on the re-estimation variation. Since $4 T _ { e , c , j } = 4 V _ { e , c , j } ^ { \mathrm { s u b } } + \hat { \Delta } _ { e , j } ^ { 2 }$ by Eq. (33),

$$
\frac { 1 + \kappa _ { e , c , j } } { 1 + 2 \kappa _ { e , c , j } } = \frac { 4 \left( V _ { e , c , j } ^ { \mathrm { s u b } } + \epsilon _ { q } \right) + \Delta _ { e , j } ^ { 2 } } { 4 \left( V _ { e , c , j } ^ { \mathrm { s u b } } + \epsilon _ { q } \right) + 2 \Delta _ { e , j } ^ { 2 } } = \frac { T _ { e , c , j } + \epsilon _ { q } } { T _ { e , c , j } + \epsilon _ { q } + \Delta _ { e , j } ^ { 2 } / 4 } ,\tag{35}
$$

so by Eq. (16),

$$
q _ { e , c , j } ^ { \mathrm { r e } } > \frac { 1 + \kappa _ { e , c , j } } { 1 + 2 \kappa _ { e , c , j } } \iff \frac { T _ { e , c , j } + \epsilon _ { q } } { T _ { e , c , j } + \epsilon _ { q } + D _ { e , c , j } ^ { \mathrm { r e } } } > \frac { T _ { e , c , j } + \epsilon _ { q } } { T _ { e , c , j } + \epsilon _ { q } + \Delta _ { e , j } ^ { 2 } / 4 } \iff D _ { e , c , j } ^ { \mathrm { r e } } < \frac { \Delta _ { e , j } ^ { 2 } } { 4 } .\tag{36}
$$

Suppose the threshold condition holds for both classes. Then $\sqrt { D _ { e , c , j } ^ { \mathrm { r e } } } < | \Delta _ { e , j } | / 2$ for each $c ,$ hence $\sum _ { c } \sqrt { D _ { e , c , j } ^ { \mathrm { r e } } } < | \Delta _ { e , j } |$ and $\Delta _ { e , j } \neq 0$ . Using Eq. (32), we obtain

$$
\begin{array} { r l } & { \Delta _ { e , j } \Delta _ { e , j } ^ { \mathrm { r e } } = \Delta _ { e , j } ^ { 2 } + \Delta _ { e , j } \left( \Delta _ { e , j } ^ { \mathrm { r e } } - \Delta _ { e , j } \right) } \\ & { \qquad \geq \Delta _ { e , j } ^ { 2 } - | \Delta _ { e , j } | \left| \Delta _ { e , j } ^ { \mathrm { r e } } - \Delta _ { e , j } \right| \geq | \Delta _ { e , j } | \Big ( | \Delta _ { e , j } | - \displaystyle \sum _ { c \in \{ 0 , 1 \} } \sqrt { D _ { e , c , j } ^ { \mathrm { r e } } } \Big ) > 0 . } \end{array}\tag{37}
$$

Hence $\Delta _ { e , j } ^ { \mathrm { r e } }$ and $\Delta _ { e , j }$ have the same sign. The threshold $( 1 + \kappa _ { e , c , j } ) / ( 1 + 2 \kappa _ { e , c , j } )$ lies in $( 1 / 2 , 1 ]$ and decreases in $\kappa _ { e , c , j }$ , so strongly separated factors preserve the sign of their class contrast under moderate support, whereas weakly separated factors require support close to one.

## C PROOF OF PROPOSITION 2

Proposition 2 (Variational Form of Pairwise Qualification) For a probability vector π over J factors and supports $q _ { j } \in ( 0 , 1 ]$ , define $\begin{array} { r } { \alpha _ { j } = \pi _ { j } q _ { j } , g = \sum _ { i = 1 } ^ { J } \alpha _ { j } } \end{array}$ , and $\overline { { \alpha } } _ { j } = \alpha _ { j } / g \ /$ . Let $\nu _ { \pi }$ be the set ofprobability vectors v with $v _ { j } = 0$ whenever $\pi _ { j } = 0 ,$ , and define

$$
\mathcal { I } ( \pmb { v } ) = \mathrm { K L } ( \pmb { v } \parallel \pmb { \pi } ) + \sum _ { j = 1 } ^ { J } v _ { j } \log \frac { 1 } { q _ { j } } .\tag{38}
$$

Then α is the unique minimizer ofJ over $\nu _ { \pi }$ , with minimum value $- \log g ,$ , and

$$
\overline { { \alpha } } _ { j } = \frac { \partial \log g } { \partial \log q _ { j } } , \quad - \log g \leq \sum _ { j = 1 } ^ { J } \pi _ { j } \log \frac { 1 } { q _ { j } } ,\tag{39}
$$

where equality holds ifand only $i f q _ { j }$ is constant over thefactors with $\pi _ { j } > 0$

Proof.

Let ${ \mathcal { T } } _ { \pi } = \{ j \in \{ 1 , \dots , J \} : \pi _ { j } > 0 \}$ denote the support of $\pi ,$ , which is nonempty because $\pi$ is a probability vector. Every $\pmb { v } \in \mathcal { V } _ { \pi }$ is supported on $\mathcal { T } _ { \pi }$ and satisfies $\textstyle \sum _ { j \in { \mathcal { T } } _ { \pi } } v _ { j } = { \\bar { 1 } }$ . Throughout the proof, we use the convention 0 log $\cdot ( 0 / b ) = 0$ for $b > 0$

Since $q _ { j } \in ( 0 , 1 ] , \alpha _ { j }$ is strictly positive on $\mathcal { T } _ { \pi }$ and zero outside it. Moreover,

$$
0 < g = \sum _ { j \in \mathbb { Z } _ { \pi } } \pi _ { j } q _ { j } \leq \sum _ { j \in \mathbb { Z } _ { \pi } } \pi _ { j } = 1 .\tag{40}
$$

Thus, α is well defined, belongs to $\nu _ { \pi }$ , and has strictly positive entries on $\mathcal { L } _ { \pi }$

For every $\pmb { v } \in \mathcal { V } _ { \pi }$ , combining the two terms of $\mathcal { I }$ and substituting $\alpha _ { j } = g \overline { { \alpha } } _ { j }$ gives

$$
\begin{array} { r l } & { \mathcal { I } ( \pmb { v } ) = \displaystyle \sum _ { j \in \mathbb { Z } _ { \pi } } v _ { j } \log \frac { v _ { j } } { \pi _ { j } } + \displaystyle \sum _ { j \in \mathbb { Z } _ { \pi } } v _ { j } \log \frac { 1 } { q _ { j } } = \displaystyle \sum _ { j \in \mathbb { Z } _ { \pi } } v _ { j } \log \frac { v _ { j } } { \alpha _ { j } } } \\ & { \quad \quad \quad = \displaystyle \sum _ { j \in \mathbb { Z } _ { \pi } } v _ { j } \log \frac { v _ { j } } { \overline { { \alpha } } _ { j } } - \left( \displaystyle \sum _ { j \in \mathbb { Z } _ { \pi } } v _ { j } \right) \log g = \mathrm { K L } ( \pmb { v } \| \overline { { \alpha } } ) - \log g , } \end{array}\tag{41}
$$

where the last equality uses $\textstyle \sum _ { j \in { \mathcal { T } } _ { \pi } } v _ { j } = 1$ . The objective is therefore a KL divergence to α plus a term independent of v.

Let ${ \mathcal { T } } _ { v } = \{ j \in { \mathbb { Z } } _ { \pi } : v _ { j } > 0 \}$ . Applying log $t \leq t - 1$ for $t > 0$ gives

$$
\mathrm { K L } ( \pmb { v } \Vert \overline { { \alpha } } ) = - \sum _ { j \in \mathcal { T } _ { v } } v _ { j } \log \frac { \overline { { \alpha } } _ { j } } { v _ { j } } \geq \sum _ { j \in \mathcal { T } _ { v } } ( v _ { j } - \overline { { \alpha } } _ { j } ) = \sum _ { j \in \mathcal { T } _ { \pi } \backslash \mathcal { T } _ { v } } \overline { { \alpha } } _ { j } \geq 0 .\tag{42}
$$

I $\mathbf { \partial } : T _ { v } \neq T _ { \pi }$ , the final sum is strictly positive because $\overline { { \alpha } } _ { j } \ > \ 0$ on $\scriptstyle { \mathcal { Z } } _ { \pi }$ . Otherwise, equality holds precisely when $\overline { { \alpha } } _ { j } / v _ { j } = 1$ for every $j \in \mathcal { T } _ { \pi }$ , by the equality condition of log $t \leq t - \bar { 1 }$ . Hence the KL divergence vanishes if and only if ${ \pmb v } = \overline { { \alpha } }$ , and Eq. (41) yields

$$
\mathcal { I } ( \pmb { v } ) \geq - \log g = \mathcal { I } ( \overline { { \pmb { \alpha } } } ) , \qquad \pmb { v } \in \mathcal { V } _ { \pi } ,\tag{43}
$$

with equality if and only if $v = { \overline { { \alpha } } }$ . Since α is feasible, it is the unique minimizer, and the minimum value $\mathbf { i s } - \log g .$

For the sensitivity identity, $\begin{array} { r } { g = \sum _ { j ^ { \prime } \in \mathcal { T } _ { \pi } } \pi _ { j ^ { \prime } } q _ { j ^ { \prime } } } \end{array}$ is linear in each $q _ { j }$ with $\partial g / \partial q _ { j } = \pi _ { j }$ , so

$$
\frac { \partial \log g } { \partial \log q _ { j } } = \frac { q _ { j } } { g } \frac { \partial g } { \partial q _ { j } } = \frac { \pi _ { j } q _ { j } } { g } = \overline { { { \alpha } } } _ { j } .\tag{44}
$$

For the inequality, $\pi \in \mathcal { V } _ { \pi }$ and $\mathrm { K L } ( \pi \| \pi ) = 0 .$ , so Eq. (43) evaluated at $v = \pi$ gives

$$
- \log g \leq \mathcal { I } ( \pmb { \pi } ) = \sum _ { j \in \mathcal { I } _ { \pmb { \pi } } } \pi _ { j } \log \frac { 1 } { q _ { j } } ,\tag{45}
$$

with equality if and only if $\pi = { \overline { { \alpha } } } .$ , that is, $\pi _ { j } = \pi _ { j } q _ { j } / g$ for every $j \in \mathcal { T } _ { \pi }$ , which holds if and only if $q _ { j } = g$ for every $j \in \mathcal { T } _ { \pi }$ . Equivalently, Eq. (45) is Jensen’s inequality log $\begin{array} { r } { \sum _ { j } \pi _ { j } q _ { j } \ge \sum _ { j } \pi _ { j } \log q _ { j } } \end{array}$ for the concave logarithm, with the same equality condition. □

With $\pi = \pi ^ { \mathrm { t a s k } }$ and $q _ { j } = q _ { e , e ^ { \prime } , c , j } ^ { \mathrm { p a i r } }$ , the objective specializes to pairwise qualification. By Eq. (16), the penalty expands as

$$
\log \frac { 1 } { q _ { e , e ^ { \prime } , c , j } ^ { \mathrm { p a i r } } } = \frac { 1 } { 2 } \log \left( 1 + \frac { D _ { e , c , j } ^ { \mathrm { r e } } } { T _ { e , c , j } + \epsilon _ { q } } \right) + \frac { 1 } { 2 } \log \left( 1 + \frac { D _ { e ^ { \prime } , c , j } ^ { \mathrm { r e } } } { T _ { e ^ { \prime } , c , j } + \epsilon _ { q } } \right) ,\tag{46}
$$

so $\mathcal { I } _ { e , e ^ { \prime } , c }$ trades proximity to $\pi ^ { \mathrm { t a s k } }$ against re-estimation deviations measured on the local reference scales of both sites. Eq. (44) becomes

$$
\frac { \partial \log g _ { e , e ^ { \prime } , c } } { \partial \log q _ { e , e ^ { \prime } , c , j } ^ { \mathrm { p a i r } } } = \overline { { \alpha } } _ { e , e ^ { \prime } , c , j } , \quad \mathrm { ~ d ~ l o g ~ } g _ { e , e ^ { \prime } , c } = \sum _ { j = 1 } ^ { J } \overline { { \alpha } } _ { e , e ^ { \prime } , c , j } \mathrm { ~ d ~ l o g ~ } q _ { e , e ^ { \prime } , c , j } ^ { \mathrm { p a i r } } ,\tag{47}
$$

so a relative change in the support of factor j moves the alignment strength by its normalized weight times that change. Evaluating Eq. (41) at ${ \dot { v } } = \pi ^ { \operatorname { t a s k } }$ gives the exact reduction in support penalty obtained by reweighting,

$$
\sum _ { j = 1 } ^ { J } \pi _ { j } ^ { \operatorname { t a s k } } \log \frac { 1 } { q _ { e , e ^ { \prime } , c , j } ^ { \operatorname { p a i r } } } - ( - \log g _ { e , e ^ { \prime } , c } ) = \operatorname { K L } \bigl ( \pi ^ { \operatorname { t a s k } } \big \| \overline { { \alpha } } _ { e , e ^ { \prime } , c } \bigr ) \geq 0 ,\tag{48}
$$

which is strictly positive unless $q _ { e , e ^ { \prime } , c , j } ^ { \mathrm { p a i r } }$ is constant over the factors with $\pi _ { j } ^ { \mathrm { t a s k } } > 0$

## D DEBIASED SINKHORN DIVERGENCE

Optimal transport compares empirical distributions by minimizing a ground cost over couplings that preserve their probability masses Gabriel & Marco (2019). Entropic regularization enables computation through Sinkhorn scaling, and a self-transport correction removes the regularization bias at identical inputs Cuturi (2013); Feydy et al. (2019). This part gives the construction for the qualification-weighted distributions in Eq. (25) and derives how the qualifications enter the alignment gradient.

Fix a source-site pair $( e , e ^ { \prime } )$ , class $c ,$ and the associated qualification profile. Write $\mu = \widehat { P } _ { e , c } ^ { Q [ e , e ^ { \prime } ] }$ and $\nu = \widehat { P } _ { e ^ { \prime } , c } ^ { Q [ e , e ^ { \prime } ] }$ for the empirical distributions in Eq. (24), and let $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ and ${ \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } { \mathbf { } } _ { \mathbf { } } _ { \mathbf { } } { \mathbf } _ { } { \mathbf } { } _ { \mathbf { } } { \mathbf } _ { } { \mathbf } _ { } { \mathbf } { \mathbf } _ { } { \mathbf } _ { } { \mathbf } { \mathbf } _ { } { \mathbf } _ { }  _ { \mathbf { } } _ { \mathbf { } \mathbf { } } _ { \mathbf } _ { } { \mathbf } _ { \mathbf { } } _ { \mathbf } _ { } { \mathbf } _ { \mathbf } { \mathbf } _ { } _ { \mathbf } { \mathbf } _ { } _ { \mathbf } { \mathbf } _ { } _ { \mathbf } { \mathbf } _ { } _ { \mathbf } _ { } { \mathbf } _ { \mathbf } _ { } _ { \mathbf } _ { } { \mathbf } _ $ denote the weighted representations from Eq. (22). Then

$$
\begin{array} { l } { { \displaystyle \mu = \sum _ { i = 1 } ^ { n } \eta _ { i } \delta _ { \pmb { x } _ { i } } , \quad \eta _ { i } = \frac { 1 } { n } , \quad n = | \mathcal { B } _ { e , c } | , } } \\ { { \nu = \displaystyle \sum _ { h = 1 } ^ { n ^ { \prime } } \zeta _ { h } \delta _ { \pmb { y } _ { h } } , \quad \zeta _ { h } = \frac { 1 } { n ^ { \prime } } , \quad n ^ { \prime } = | \mathcal { B } _ { e ^ { \prime } , c } | , } } \end{array}\tag{49}
$$

with $n , n ^ { \prime } > 0 .$ , so the marginal vectors $\eta$ and $\zeta$ have positive entries and unit total mass. Both sites use the same normalized qualifications, so the squared Euclidean ground cost coincides with the qualification-weighted dissimilarity in Eq. (23):

$$
C _ { i h } = \| { \pmb x } _ { i } - { \pmb y } _ { h } \| _ { 2 } ^ { 2 } , \quad \| { \pmb x } _ { i } \| _ { 2 } ^ { 2 } = \| { \pmb y } _ { h } \| _ { 2 } ^ { 2 } = \frac { 1 } { 2 } , \quad 0 \leq C _ { i h } \leq 2 .\tag{50}
$$

Let $C = [ C _ { i h } ] \in \mathbb { R } ^ { n \times n ^ { \prime } }$ . The admissible couplings are

$$
\mathcal { U } ( \eta , \zeta ) = \left\{ \Gamma \in \mathbb { R } _ { + } ^ { n \times n ^ { \prime } } : \Gamma \mathbf { 1 } _ { n ^ { \prime } } = \eta , \quad \Gamma ^ { \top } \mathbf { 1 } _ { n } = \zeta \right\} ,\tag{51}
$$

and the unregularized transport cost is

$$
\mathrm { O T } _ { 0 } ( \mu , \nu ) = \operatorname* { m i n } _ { \substack { \mathbf { r } \in \mathcal { U } ( \eta , \zeta ) } } \langle \mathbf { r } , C \rangle , \quad \langle \mathbf { r } , C \rangle = \sum _ { i = 1 } ^ { n } \sum _ { h = 1 } ^ { n ^ { \prime } } \Gamma _ { i h } C _ { i h } ,\tag{52}
$$

which equals the squared Wasserstein-2 distance $W _ { 2 } ^ { 2 } ( \mu , \nu )$ for the squared Euclidean ground cost Gabriel & Marco (2019).

We regularize the coupling relative to the independent coupling $\eta \zeta ^ { \top }$ :

$$
\operatorname { O T } _ { \varepsilon } ( \mu , \nu ) = \operatorname* { m i n } _ { \mathbf { r } \in \mathcal { U } ( \eta , \zeta ) } \left\{ \langle \mathbf { r } , C \rangle + \varepsilon \operatorname { K L } \bigl ( \mathbf { r } \parallel \eta \zeta ^ { \top } \bigr ) \right\} , \quad \varepsilon > 0 ,\tag{53}
$$

where the generalized KL divergence is

$$
\mathrm { K L } \big ( \mathbf { \mathbf { r } } \left\| \eta \boldsymbol { \zeta } ^ { \top } \right) = \sum _ { i , h } \left[ \Gamma _ { i h } \log \frac { \Gamma _ { i h } } { \eta _ { i } \zeta _ { h } } - \Gamma _ { i h } + \eta _ { i } \zeta _ { h } \right] ,\tag{54}
$$

with $0 \log ( 0 / b ) = 0$ for $b > 0$ . For feasible couplings, the linear terms cancel and the marginal constraints give

$$
\mathrm { K L } \bigl ( \mathbf { \vec { r } } \bigr \| \eta \zeta ^ { \top } \bigr ) = \sum _ { i , h } \Gamma _ { i h } \log \Gamma _ { i h } - \sum _ { i } \eta _ { i } \log \eta _ { i } - \sum _ { h } \zeta _ { h } \log \zeta _ { h } ,\tag{55}
$$

so this formulation and the negative-entropy formulation share the same optimal coupling and differ by marginal-dependent constants. The feasible set is compact and the regularizer strictly convex, so the minimizer exists, is unique, and has strictly positive entries Cuturi (2013).

Introduce dual potentials $\phi \in \mathbb { R } ^ { n }$ and $\boldsymbol \psi \in \mathbb { R } ^ { n ^ { \prime } }$ for the marginal constraints. The Lagrangian is

$$
{ \mathcal { L } } _ { \mathrm { O T } } ( \mathbf { T } , \phi , \psi ) = \langle \mathbf { \Gamma } ( \mathbf { r } , C ) + \varepsilon \operatorname { K L } \bigl ( \mathbf { T } \parallel \eta \zeta ^ { \top } \bigr ) + \phi ^ { \top } ( \eta - \mathbf { \Gamma } \mathbf { T } \mathbf { 1 } _ { n ^ { \prime } } ) + \psi ^ { \top } ( \zeta - \mathbf { \Gamma } \mathbf { T } ^ { \top } \mathbf { 1 } _ { n } ) ,\tag{56}
$$

and stationarity in each coupling entry yields

$$
C _ { i h } + \varepsilon \log \frac { \Gamma _ { i h } ^ { \star } } { \eta _ { i } \zeta _ { h } } - \phi _ { i } ^ { \star } - \psi _ { h } ^ { \star } = 0 , \quad \Gamma _ { i h } ^ { \star } = \eta _ { i } \zeta _ { h } \exp \biggl ( \frac { \phi _ { i } ^ { \star } + \psi _ { h } ^ { \star } - C _ { i h } } { \varepsilon } \biggr ) .\tag{57}
$$

Minimizing the Lagrangian over Γ for fixed potentials gives the dual problem

$$
\mathrm { O T } _ { \varepsilon } ( \mu , \nu ) = \operatorname* { m a x } _ { \phi , \psi } \left\{ \eta ^ { \top } \phi + \xi ^ { \top } \psi - \varepsilon \sum _ { i , h } \eta _ { i } \zeta _ { h } \left[ \exp \left( \frac { \phi _ { i } + \psi _ { h } - C _ { i h } } { \varepsilon } \right) - 1 \right] \right\} ,\tag{58}
$$

where strong duality holds because the strictly positive coupling $\eta \zeta ^ { \top }$ is feasible. At optimality, the exponential term recovers the optimal coupling, whose total mass is one, so

$$
\operatorname { O T } _ { \varepsilon } ( \mu , \nu ) = \eta ^ { \top } \phi ^ { \star } + \zeta ^ { \top } \psi ^ { \star } .\tag{59}
$$

The exponential form leads to Sinkhorn scaling. With

$$
\begin{array} { r } { ( \pmb { K } _ { \varepsilon } ) _ { i h } = \exp ( - C _ { i h } / \varepsilon ) , \quad u _ { i } = \eta _ { i } \exp ( \phi _ { i } / \varepsilon ) , \quad v _ { h } = \zeta _ { h } \exp ( \psi _ { h } / \varepsilon ) , } \end{array}\tag{60}
$$

the coupling is $\Gamma = \mathrm { d i a g } ( \boldsymbol { u } ) K _ { \varepsilon } \mathrm { d i a g } ( \boldsymbol { v } )$ , and alternately enforcing the two marginal constraints gives

$$
\begin{array} { r } { \pmb { u } ^ { ( t + 1 ) } = \pmb { \eta } \oslash ( \mathbf { K } _ { \varepsilon } \pmb { v } ^ { ( t ) } ) , \quad \pmb { v } ^ { ( t + 1 ) } = \xi \oslash ( \mathbf { K } _ { \varepsilon } ^ { \top } \pmb { u } ^ { ( t + 1 ) } ) , } \end{array}\tag{61}
$$

where $\oslash$ denotes elementwise division. For a positive kernel and positive marginals, positive initialization converges to the unique regularized optimum Sinkhorn & Knopp (1967); Cuturi (2013). Substituting Eq. (57) into the marginal constraints expresses the same updates in the dual potentials,

$$
\phi _ { i } ^ { ( t + 1 ) } = - \varepsilon \operatorname { L S E } _ { h } \left( \log \zeta _ { h } + \frac { \psi _ { h } ^ { ( t ) } - C _ { i h } } { \varepsilon } \right) , \quad \psi _ { h } ^ { ( t + 1 ) } = - \varepsilon \operatorname { L S E } _ { i } \left( \log \eta _ { i } + \frac { \phi _ { i } ^ { ( t + 1 ) } - C _ { i h } } { \varepsilon } \right) ,\tag{62}
$$

where $\begin{array} { r } { \mathrm { L S E } _ { k } ( z _ { k } ) = \log \sum _ { k } \exp ( z _ { k } ) } \end{array}$ . These are alternating maximizations of the dual objective and are evaluated with stable log-sum-exp reductions Feydy et al. (2019).

Entropic regularization makes the self-transport cost positive. For $\mu$ with at least two distinct support points, the two terms of the self-transport objective vanish at different couplings,

$$
\begin{array} { c } { { \langle \mathbf { r } , { \pmb { C } } ^ { x x } \rangle = 0 \displaystyle \longleftrightarrow \mathbf { r } = \mathrm { d i a g } ( \pmb { \eta } ) , \quad \mathrm { K L } \big ( \mathbf { r } \left\| \eta \pmb { \eta } ^ { \top } \right) = 0 \longleftrightarrow \mathbf { r } = \pmb { \eta } \pmb { \eta } ^ { \top } , } } \\ { { \mathrm { K L } \big ( \mathrm { d i a g } ( \pmb { \eta } ) \left\| \eta \pmb { \eta } ^ { \top } \right) = - \displaystyle \sum _ { i } \eta _ { i } \log \eta _ { i } > 0 , } } \end{array}\tag{63}
$$

where $C _ { i \ell } ^ { x x } = \| \pmb { x } _ { i } - \pmb { x } _ { \ell } \| _ { 2 } ^ { 2 } , \mathrm { s o } \mathrm { O T } _ { \varepsilon } ( \mu , \mu ) > 0$ . Subtracting half of each self-transport cost removes this bias and defines the debiased Sinkhorn divergence Feydy et al. (2019),

$$
S _ { \varepsilon } ( \mu , \nu ) = \mathrm { O T } _ { \varepsilon } ( \mu , \nu ) - \frac { 1 } { 2 } \mathrm { O T } _ { \varepsilon } ( \mu , \mu ) - \frac { 1 } { 2 } \mathrm { O T } _ { \varepsilon } ( \nu , \nu ) ,\tag{64}
$$

where all three terms use the same ε and the same weighted representation map, and $C _ { h k } ^ { y y } \ =$ $\| \pmb { y } _ { h } - \pmb { y } _ { k } \| _ { 2 } ^ { 2 } . \mathrm { \bf ~ B y } \mathrm { \bf E q . } ( 5 5 )$ , with $\begin{array} { r } { H ( \pmb { \eta } ) = - \sum _ { i } \eta _ { i } \log \eta _ { i } } \end{array}$ and $\mathrm { O T } _ { \varepsilon } ^ { \mathrm { e n t } }$ the negative-entropy formulation,

$$
\mathrm { O T } _ { \varepsilon } ( \mu , \nu ) = \mathrm { O T } _ { \varepsilon } ^ { \mathrm { e n t } } ( \mu , \nu ) + \varepsilon H ( \eta ) + \varepsilon H ( \zeta ) , \quad \mathrm { O T } _ { \varepsilon } ( \mu , \mu ) = \mathrm { O T } _ { \varepsilon } ^ { \mathrm { e n t } } ( \mu , \mu ) + 2 \varepsilon H ( \eta ) ,\tag{65}
$$

so the constants cancel in Eq. (64) and both formulations yield the same $S _ { \varepsilon }$ . Transposition exchanges the marginals of a coupling without changing its cost or penalty, which gives

$$
\mathrm { O T } _ { \varepsilon } ( \mu , \nu ) = \mathrm { O T } _ { \varepsilon } ( \nu , \mu ) , \quad S _ { \varepsilon } ( \mu , \nu ) = S _ { \varepsilon } ( \nu , \mu ) , \quad S _ { \varepsilon } ( \mu , \mu ) = 0 .\tag{66}
$$

For nonnegativity and definiteness, we use the optimal self-transport potentials Feydy et al. (2019). Let $\mathcal { D } _ { \mu \mu }$ denote the self-transport dual objective in Eq. (58) with $\nu = \mu ,$ which is concave and satisfies $\mathcal { D } _ { \mu \mu } ( \phi , \psi ) = \mathcal { D } _ { \mu \mu } ( \psi , \phi )$ . For an optimal pair $( \phi ^ { \star } , \psi ^ { \star } )$ , the average $\begin{array} { r } { \phi ^ { \mu } = \frac { 1 } { 2 } \big ( \phi ^ { \star } + \psi ^ { \star } \big ) } \end{array}$ satisfies

$$
\begin{array} { r } { \mathcal { D } _ { \mu \mu } ( \phi ^ { \mu } , \phi ^ { \mu } ) \geq \frac { 1 } { 2 } \mathcal { D } _ { \mu \mu } ( \phi ^ { \star } , \psi ^ { \star } ) + \frac { 1 } { 2 } \mathcal { D } _ { \mu \mu } ( \psi ^ { \star } , \phi ^ { \star } ) = \mathrm { O T } _ { \varepsilon } ( \mu , \mu ) , } \end{array}\tag{67}
$$

so $( \phi ^ { \mu } , \phi ^ { \mu } )$ is optimal; define $\phi ^ { \nu }$ likewise. Let $\{ z _ { \ell } \} _ { \ell = 1 } ^ { n _ { z } }$ be the distinct points in the union of the two supports, and define

$$
\Omega _ { \ell r } = \exp \biggl ( - \frac { \| z _ { \ell } - z _ { r } \| _ { 2 } ^ { 2 } } { \varepsilon } \biggr ) \ , \quad \xi _ { \ell } = \sum _ { \{ i : x _ { i } = z _ { \ell } \} } \eta _ { i } \exp ( \phi _ { i } ^ { \mu } / \varepsilon ) , \quad \chi _ { \ell } = \sum _ { \{ h : y _ { h } = z _ { \ell } \} } \zeta _ { h } \exp ( \phi _ { h } ^ { \nu } / \varepsilon ) ,\tag{68}
$$

where empty sums are zero. The Gaussian Gram matrix Ω is strictly positive definite on distinct points. Let $\bar { \Gamma } ^ { x x , \star }$ and $\mathbf { \Gamma } _ { \mathbf { T } } ^ { y y , \star }$ be the optimal self-couplings. Their unit total masses and Eq. (59) give

$$
\xi ^ { \top } \Omega \xi = \sum _ { i , \ell } \Gamma _ { i \ell } ^ { x x , \star } = 1 , \qquad \quad x ^ { \top } \Omega \chi = \sum _ { h , k } \Gamma _ { h k } ^ { y y , \star } = 1 ,\tag{69}
$$

$$
\mathrm { O T } _ { \varepsilon } ( \mu , \mu ) = 2 \eta ^ { \top } \phi ^ { \mu } , \qquad \mathrm { O T } _ { \varepsilon } ( \nu , \nu ) = 2 \zeta ^ { \top } \phi ^ { \nu } .
$$

Evaluating the cross-transport dual at $( \phi ^ { \mu } , \phi ^ { \nu } )$ gives

$$
\begin{array} { l } { \displaystyle \mathrm { O T } _ { \varepsilon } ( \mu , \nu ) \geq \eta ^ { \top } \phi ^ { \mu } + \xi ^ { \top } \phi ^ { \nu } - \varepsilon \left[ \sum _ { i , h } \eta _ { i } \zeta _ { h } \exp \Bigl ( \frac { \phi _ { i } ^ { \mu } + \phi _ { h } ^ { \nu } - C _ { i h } } { \varepsilon } \Bigr ) - 1 \right] } \\ { = \eta ^ { \top } \phi ^ { \mu } + \xi ^ { \top } \phi ^ { \nu } + \varepsilon ( 1 - \xi ^ { \top } \Omega \chi ) , } \end{array}\tag{70}
$$

and subtracting half of each self-cost from Eq. (69) yields

$$
\begin{array} { c } { \displaystyle \mathcal { S } _ { \varepsilon } ( \mu , \nu ) \geq \varepsilon ( 1 - \xi ^ { \top } \Omega \chi ) = \frac { \varepsilon } { 2 } \left( \xi ^ { \top } \Omega \xi + \chi ^ { \top } \Omega \chi - 2 \xi ^ { \top } \Omega \chi \right) } \\ { \displaystyle = \frac { \varepsilon } { 2 } ( \xi - \chi ) ^ { \top } \Omega ( \xi - \chi ) \geq 0 . } \end{array}\tag{71}
$$

If $S _ { \varepsilon } ( \mu , \nu ) = 0$ , strict positive definiteness of Ω gives $\xi = x .$ , and the marginal constraints of the self-couplings recover the masses at each distinct point,

$$
\mu ( \{ z _ { \ell } \} ) = \sum _ { \{ i : x _ { i } = z _ { \ell } \} } \eta _ { i } = \xi _ { \ell } ( \Omega \xi ) _ { \ell } , \quad \nu ( \{ z _ { \ell } \} ) = \sum _ { \{ h : y _ { h } = z _ { \ell } \} } \zeta _ { h } = \chi _ { \ell } ( \Omega \chi ) _ { \ell } ,\tag{72}
$$

so $\xi = x$ implies $\mu = \nu .$ . Together with Eq. (66), this establishes

$$
\begin{array} { r } { S _ { \varepsilon } ( \mu , \nu ) \geq 0 , \quad S _ { \varepsilon } ( \mu , \nu ) = 0 \iff \mu = \nu . } \end{array}\tag{73}
$$

The same formulation provides the gradients used for representation learning. With fixed marginal weights, the envelope theorem gives $\mathbf { \bar { \partial } } { \mathrm { O T } } _ { \varepsilon } / \partial C _ { i h } = \Gamma _ { i h } ^ { x \bar { y _ { , } } \star }$ for the optimal cross-coupling Gabriel & Marco (2019), so

$$
\nabla _ { { \pmb x } _ { i } } \operatorname { O T } _ { \varepsilon } ( \mu , \nu ) = 2 \sum _ { h } \Gamma _ { i h } ^ { x y , \star } ( { \pmb x } _ { i } - { \pmb y } _ { h } ) .\tag{74}
$$

For self-transport, transposition preserves feasibility and objective, so uniqueness makes the optimal self-coupling symmetric; since each representation appears in both arguments of the self-cost,

$$
\begin{array} { l } { \nabla _ { { \pmb x } _ { i } } \operatorname { O T } _ { { \varepsilon } } ( { \mu } , { \mu } ) = 2 \displaystyle \sum _ { { \ell } } \Gamma _ { i { \ell } } ^ { x x , \star } ( { \pmb x } _ { i } - { \pmb x } _ { \ell } ) + 2 \displaystyle \sum _ { { \ell } } \Gamma _ { { \ell } i } ^ { x x , \star } ( { \pmb x } _ { i } - { \pmb x } _ { \ell } ) } \\ { = 4 \displaystyle \sum _ { { \ell } } \Gamma _ { i { \ell } } ^ { x x , \star } ( { \pmb x } _ { i } - { \pmb x } _ { \ell } ) . } \end{array}\tag{75}
$$

Combining the two gives, at the converged transport solutions Feydy et al. (2019),

$$
\begin{array} { l } { { \nabla _ { { \pmb x } _ { i } } S _ { \varepsilon } ( { \mu } , { \nu } ) = 2 \displaystyle \sum _ { h } \Gamma _ { i h } ^ { x y , \star } ( { \pmb x } _ { i } - { \pmb y } _ { h } ) - 2 \displaystyle \sum _ { \ell } \Gamma _ { i \ell } ^ { x x , \star } ( { \pmb x } _ { i } - { \pmb x } _ { \ell } ) } , } \\ { { \nabla _ { { \pmb y } _ { h } } S _ { \varepsilon } ( { \mu } , { \nu } ) = 2 \displaystyle \sum _ { i } \Gamma _ { i h } ^ { x y , \star } ( { \pmb y } _ { h } - { \pmb x } _ { i } ) - 2 \displaystyle \sum _ { k } \Gamma _ { h k } ^ { y y , \star } ( { \pmb y } _ { h } - { \pmb y } _ { k } ) . } } \end{array}\tag{76}
$$

Finally, the alignment contribution for the fixed source-site pair and class is $\ell _ { e , e ^ { \prime } , c } ^ { Q } = g _ { e , e ^ { \prime } , c } S _ { \varepsilon } ( \mu , \nu )$ By Eq. (22), the j-th block of $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ is $\sqrt { \overline { { \alpha } } _ { e , e ^ { \prime } , c , j } / 2 } r _ { i , j } ^ { e , ( 0 ) }$ , so the j-th block of Eq. (76) equals $\sqrt { \overline { { \alpha } } _ { e , e ^ { \prime } , c , j } / 2 }$ times the transport residual

$$
\Delta _ { i , j } ^ { e , e ^ { \prime } , c } = \sum _ { h } \Gamma _ { i h } ^ { x y , \star } \big ( r _ { i , j } ^ { e , ( 0 ) } - r _ { h , j } ^ { e ^ { \prime } , ( 0 ) } \big ) - \sum _ { \ell } \Gamma _ { i \ell } ^ { x x , \star } \big ( r _ { i , j } ^ { e , ( 0 ) } - r _ { \ell , j } ^ { e , ( 0 ) } \big ) .\tag{77}
$$

Applying the chain rule with the qualifications held fixed gives

$$
\begin{array} { r } { \nabla _ { r _ { i , j } ^ { e , ( 0 ) } } \ell _ { e , e ^ { \prime } , c } ^ { Q } = g _ { e , e ^ { \prime } , c } \overline { { \alpha } } _ { e , e ^ { \prime } , c , j } \Delta _ { i , j } ^ { e , e ^ { \prime } , c } = \alpha _ { e , e ^ { \prime } , c , j } \Delta _ { i , j } ^ { e , e ^ { \prime } , c } , } \end{array}
$$

$$
\alpha _ { e , e ^ { \prime } , c , j } = \pi _ { j } ^ { \mathrm { t a s k } } q _ { e , e ^ { \prime } , c , j } ^ { \mathrm { p a i r } } , \quad \sum _ { j = 1 } ^ { J } \alpha _ { e , e ^ { \prime } , c , j } = g _ { e , e ^ { \prime } , c } .\tag{78}
$$

The transport plans are shared across factors and set by the normalized qualifications through the ground cost, while the gradient reaching factor j scales with its unnormalized qualification, and the scales sum to the overall qualification. Averaging over unordered pairs and classes recovers Eq. (25):

$$
\mathcal { L } _ { \mathrm { e n v } } ^ { Q } = \frac { 2 } { E ( E - 1 ) } \sum _ { e < e ^ { \prime } } \frac { 1 } { 2 } \sum _ { c \in \{ 0 , 1 \} } \ell _ { e , e ^ { \prime } , c } ^ { Q } .\tag{79}
$$

Table 3: Statistics of the datasets.
<table><tr><td>Dataset</td><td>Task</td><td>Samples</td><td>Class counts</td><td>Parcellation</td><td>Time points</td></tr><tr><td>ABIDE</td><td>ASD vs. TD</td><td>1,025</td><td>488 / 537</td><td>AAL116; CC200</td><td>194.51 ± 58.58</td></tr><tr><td>REST-meta-MDD</td><td>MDD vs. HC</td><td>2,428</td><td>1,300 / 1,128</td><td>AAL116</td><td>207.54 ± 29.29</td></tr><tr><td>SRPBS</td><td>MDD vs. HC</td><td>1,000</td><td>500 / 500</td><td>AAL116</td><td>107-284</td></tr><tr><td>ABCD</td><td>ADHD vs. HC</td><td>425</td><td>213 / 212</td><td>AAL116</td><td>383</td></tr></table>

## E DATASETS AND PREPROCESSING

## E.1 DATASET DESCRIPTION

We evaluate BRIO on four multi-site rs-fMRI datasets, retaining samples with diagnostic labels, site information, and usable ROI-level time series. Table 3 summarizes the final analysis cohorts, with class counts listed in the order of the corresponding task labels.

ABIDE. The Autism Brain Imaging Data Exchange (ABIDE) Di Martino et al. (2014) provides multi-site rs-fMRI data for studying autism spectrum disorder (ASD). We use the quality-controlled ABIDE I cohort with available phenotypic information and ROI-level time series. The final cohort contains 1,025 subjects, including 488 individuals with ASD and 537 typically developing (TD) controls from 20 sites. Acquisition sites differ in demographic composition and imaging protocols. Primary experiments use the AAL116 atlas Tzourio-Mazoyer et al. (2002), while an additional evaluation uses the CC200 parcellation Craddock et al. (2012) to examine performance under an alternative ROI definition.

REST-meta-MDD. The REST-meta-MDD consortium Yan et al. (2019) provides multi-site rs-fMRI data for studying major depressive disorder (MDD), collected across clinical centers in China. The final cohort comprises 2,428 participants from 25 acquisition sites, including 1,300 individuals with MDD and 1,128 healthy controls (HC). Each retained site includes both diagnostic groups. Site sizes range from 24 to 533 subjects, with the largest site accounting for approximately 22.0% of the cohort. Among participants with available age information, the mean age is 36.24 ± 15.07 years, ranging from 12 to 82 years, with group means of 36.23 ± 14.62 years for MDD and 36.25 ± 15.58 years for HC. Records with available sex information include 1,454 females and 925 males. All subjects are represented using AAL116, with variable numbers of available time points.

SRPBS. The Strategic Research Program for Brain Sciences (SRPBS) dataset Tanaka et al. (2021) is a multi-site neuroimaging resource collected in Japan, with demographic, diagnostic, and acquisition information. We use the MDD cohort for binary classification against HC. After quality control and cohort balancing, the analysis set contains 1,000 rs-fMRI subjects, comprising 500 MDD and 500 HC subjects from 11 acquisition environments. Ages range from 18 to 80 years, with an overall mean of 42.93 ± 13.59 years and group means of 42.66 ± 12.09 years for MDD and 43.21 ± 14.95 years for HC. The cohort includes 541 females and 459 males: 237 females and 263 males in the MDD group, and 304 females and 196 males in the HC group. ROI sequences use AAL116, with lengths ranging from 107 to 284 time points across acquisition environments.

ABCD. The Adolescent Brain Cognitive Development (ABCD) Study Casey et al. (2018) is a multisite longitudinal study of brain development and mental health. We use baseline rs-fMRI data from baseline\_year\_1\_arm\_1 for binary classification of attention-deficit/hyperactivity disorder (ADHD) versus HC. After quality control and site filtering, the final cohort contains 425 subjects from 10 imaging sites, including 213 individuals with ADHD and 212 HC. The mean age is 9.47 ± 0.50 years. All retained subjects are represented using AAL116 and have 383 retained time points.

## E.2 DATA PREPROCESSING

We apply a common FC estimation procedure across datasets and methods that take FC graphs as input. For each subject, frames containing non-finite values in any ROI are removed jointly across ROIs, and each retained ROI signal is standardized within subject. From the resulting sequence $X _ { i } ^ { e } \in \mathbb { R } ^ { T _ { i } ^ { e } \times P }$ , we compute pairwise Pearson correlations and apply the Fisher-z transformation to

obtain

$$
\boldsymbol { A } _ { i } ^ { e , ( 0 ) } = \boldsymbol { \Psi } ( \boldsymbol { X } _ { i } ^ { e } ) \in \mathbb { R } ^ { P \times P } ,\tag{80}
$$

where $T _ { i } ^ { e }$ denotes the number of retained time points. Diagonal entries are set to zero. We use $P = 1 1 6$ for AAL116 and $P = 2 0 0$ for CC200.

To encode these graphs over a common set of ROI pairs, BRIO and variants sharing its backbone use a binary propagation support $\pmb { S }$ constructed exclusively from training-site full-scan FC matrices. For each ROI pair, we compute the median absolute FC weight across subjects within each training site, followed by the median across training sites. Each ROI selects the $m \overset { \cdot } { = } \lceil \rho _ { g } ( P - 1 ) \overset { \cdot }  $ ⌉ strongest off-diagonal connections, with $\rho _ { g } = 0 . 2 0$ . We symmetrize the support by retaining an undirected connection whenever either endpoint selects the other. Constructed separately within each LOSO fold, S remains fixed across training, validation, and test subjects and all FC re-estimates. Each graph retains its own weights on this support, with positive weights and negative-weight magnitudes processed separately.

While full-scan graphs are used for parameter updates and inference, BRIO additionally constructs FC re-estimates from training subjects for qualification evaluation. We apply joint circular moving-block resampling with identical temporal indices across ROIs, preserving cross-ROI temporal correspondence and local temporal dependence within sampled blocks. Each resampled sequence $X _ { i } ^ { e , ( k ) }$ contains $T _ { i } ^ { e }$ time points and is processed using the same Pearson-correlation and Fisher-z procedure:

$$
\pmb { A } _ { i } ^ { e , ( k ) } = \Psi \Big ( \pmb { X } _ { i } ^ { e , ( k ) } \Big ) , \qquad k = 1 , \ldots , K .\tag{81}
$$

The block length is estimated once per LOSO fold from training subjects only. We draw a site- and class-balanced subset of at most 128 training subjects and, for each subject, randomly select up to 24 ROI signals and up to 48 pairwise ROI products. For each series, the autocorrelation horizon is the smallest lag at which the absolute autocorrelation stays below $\tau _ { \mathrm { a c } } = 0 . 1 0$ for two consecutive lags, searched over lags up to min $( 3 0 , \lfloor T _ { i } ^ { e } / 2 \rfloor )$ . Horizons are aggregated by taking the 0.75 quantile over series within each subject, the median over subjects within each site, and the median over training sites, so that every source site contributes equally to the fold-level value $\ell _ { b } .$ The block length applied to sequence i is

$$
\ell _ { b , i } ^ { e } = \mathrm { c l i p } \Big ( \mathrm { r o u n d } ( \ell _ { b } ) , ~ \ell _ { \mathrm { m i n } } , ~ \mathrm { m i n } \big ( \ell _ { \mathrm { m a x } } , \lfloor T _ { i } ^ { e } / 2 \rfloor \big ) \Big ) , \qquad \ell _ { \mathrm { m i n } } = 2 , \quad \ell _ { \mathrm { m a x } } = 3 0 .\tag{82}
$$

The same estimation and clipping rules are applied across datasets.

## F BASELINES

We compare BRIO with different competitive baselines across four categories:

General GNNs.

• GCN (Kipf & Welling, 2016) updates node representations through degree-normalized neighborhood propagation, combining local features with information from adjacent nodes.

• GAT (Velickoviˇ c et al., 2017) computes feature-dependent attention coefficients for neigh-´ boring nodes and combines multiple attention heads to learn node representations.

• GIN (Xu et al., 2018) applies MLPs to summed neighborhood features and pools node embeddings to capture structural differences between graphs.

## General OOD methods.

• CORAL (Sun & Saenko, 2016) reduces covariance differences between learned representations. In our source-only setting, this regularization is applied across training sites.

• IRM (Arjovsky et al., 2019) encourages a representation for which a shared classifier is simultaneously optimal across training environments, reducing reliance on environmentdependent associations.

Graph OOD methods.

• GSAT (Miao et al., 2022) learns probabilistic edge selection through an informationbottleneck objective, suppressing graph information that is unnecessary for prediction. The resulting edge probabilities also provide interpretable subgraph explanations.

• DisC (Fan et al., 2022) separately encodes causal and bias subgraphs, then recombines their representations to weaken spurious associations during training, particularly when graph labels correlate strongly with bias substructures.

• CEPG (Wang et al., 2026a) leverages environmental information to guide invariant graph representation learning and improve prediction under distribution shifts by emphasizing predictive features shared across different training environments.

• DiSCO (Sun et al., 2026) trains diverse subgraph experts and uses sparse gating to select their contributions for each graph, accommodating heterogeneous predictive structures. Different experts capture complementary causal patterns across graph instances.

## Brain network methods.

• BrainNetTF (Kan et al., 2022) learns interactions among ROI connectivity profiles through Transformer attention and forms graph embeddings using orthonormal clustering readout, which groups ROIs according to their learned functional representations.

• XG-GNN (Qiu et al., 2024) constructs task-oriented nonlinear functional networks and uses meta-learning with explanation regularization to preserve diagnostic patterns across acquisition sites, jointly considering diagnostic accuracy and the consistency of explanations.

• AGMGC (Noman et al., 2025) combines learned connectivity with correlation-based graphs, integrating spline and multi-graph convolutions with contrastive regularization to capture local connectivity patterns and broader interregional dependencies.

• FC-HGNN (Gu et al., 2025) first encodes individual connectomes using hemisphere-aware convolutions, then combines imaging and non-imaging information through heterogeneous population-graph aggregation. This connects individual connectivity patterns with relationships among subjects.

• BrainOOD (Xu et al., 2025) filters node features and regularizes extracted graph structures through information-bottleneck and structure-consistency objectives for cross-site prediction. The consistency objective encourages similar structure selection across subjects.

• DeCI (Yu et al., 2026) separates ROI time series into oscillatory and slowly varying components, processes each ROI independently, and aggregates the resulting predictions, allowing temporal dynamics to inform diagnosis beyond static correlations.

• CORE (Wang et al., 2026d) extracts a reproducible connectivity scaffold after site-specific deconfounding and models pathway dynamics on its line graph using subject-adaptive gating to combine population-level connectivity priors with individual temporal characteristics.

## G ALGORITHM

The overall training and inference process of the proposed BRIO is shown in Algorithm 1.

## H IMPLEMENTATION DETAILS

We implement BRIO in PyTorch <sup>1</sup>. The encoder is a two-layer signed GNN with hidden dimension 64 and 16-dimensional ROI identity embeddings. Positive and negative-magnitude adjacency matrices are symmetrically normalized separately. Each layer sums separate bias-free transformations of self features and messages from both channels, followed by GELU, residual addition, and LayerNorm. We use $J = 4$ connectome factors with dimension $\dot { d } _ { t } = 1 6 .$ . The shared ROI assignment uses 32-dimensional queries and keys, scaled dot-product similarities with temperature 0.10, and 60 log-domain Sinkhorn iterations targeting row and column masses of $1 / J$ and $\bar { 1 } / P$ , respectively. The factor mapping $\mathcal { F } _ { \phi }$ consists of a bias-free 64 → 16 projection followed by a $1 6  3 2  1 6$ MLP with GELU between its linear layers. Both the projection and MLP outputs are $\ell _ { 2 } \cdot$ -normalized. We train the model using AdamW with learning rate $3 \times 1 0 ^ { - 4 }$ and weight decay $1 0 ^ { - 4 }$ . Each LOSO fold reserves one site for testing and two for validation, while model fitting and estimation of training statistics use only the remaining sites. Results are reported as mean ± standard deviation across held-out test sites.

Table 4: ID results on ABIDE, ABIDE (CC200), REST-meta-MDD, SRPBS, and ABCD (ADHDtask) under ten-fold cross-validation stratified by site and class, reported as mean ± standard deviation across test folds. Bold results indicate the best performance.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="2">ABIDE</td><td colspan="2">ABIDE (CC200)</td><td colspan="2">REST-meta-MDD</td><td colspan="2">SRPBS</td><td colspan="2">ABCD (ADHD-task)</td></tr><tr><td> $\mathbf { A U C }$ </td><td>ACC</td><td>AUC</td><td>ACC</td><td>AUC</td><td>ACC</td><td> $\mathbf { A U C }$ </td><td>ACC</td><td> $\mathbf { A U C }$ </td><td>ACC</td></tr><tr><td rowspan="3">NNN</td><td>GCN</td><td> $6 3 . 4 5 _ { \pm 4 . 1 2 }$ </td><td> $5 9 . 2 1 { \scriptstyle \pm 3 . 8 5 }$ </td><td> $5 1 . 8 8 _ { \pm 4 . 0 2 }$ </td><td> $5 1 . 0 5 _ { \pm 3 . 7 4 }$ </td><td> $6 2 . 1 5 _ { \pm 2 . 8 8 }$ </td><td> $5 7 . 0 2 _ { \pm 2 . 4 5 }$ </td><td> $8 4 . 1 2 _ { \pm 3 . 8 5 }$ </td><td> $7 5 . 3 4 _ { \pm 4 . 1 2 }$ </td><td> $7 2 . 1 8 _ { \pm 5 . 1 2 }$ </td><td> $6 4 . 0 5 { \scriptstyle \pm 4 . 8 8 }$ </td></tr><tr><td>GAT</td><td> $5 9 . 1 2 _ { \pm 3 . 9 5 }$ </td><td> $5 5 . 4 4 _ { \pm 3 . 5 5 }$ </td><td> $5 8 . 6 2 _ { \pm 4 . 1 5 }$ </td><td> $5 7 . 1 8 _ { \pm 3 . 8 2 }$ </td><td>63.84±3.12</td><td> $5 9 . 1 2 _ { \pm 2 . 6 5 }$ </td><td>81.75±4.05</td><td> $7 0 . 8 8 _ { \pm 4 . 3 5 }$ </td><td> $7 1 . 0 2 _ { \pm 5 . 4 5 }$ </td><td> $6 3 . 8 8 _ { \pm 5 . 1 0 }$ </td></tr><tr><td>GIN</td><td> $5 8 . 7 4 _ { \pm 3 . 6 5 }$ </td><td> $5 4 . 9 5 _ { \pm 3 . 2 2 }$ </td><td> $5 3 . 1 5 _ { \pm 4 . 3 3 }$ </td><td> $5 0 . 4 2 _ { \pm 3 . 9 5 }$ </td><td> $5 9 . 3 3 { \scriptstyle \pm 2 . 5 4 }$ </td><td> $5 6 . 1 4 _ { \pm 2 . 2 2 }$ </td><td> $8 2 . 0 5 _ { \pm 3 . 7 8 }$ </td><td> $7 1 . 4 5 _ { \pm 3 . 9 2 } ^ { - }$ </td><td> $6 9 . 4 5 _ { \pm 5 . 3 3 }$ </td><td> $6 1 . 2 2 { \scriptstyle \pm 4 . 7 5 }$ </td></tr><tr><td rowspan="3">OOD</td><td>CORAL</td><td> $5 6 . 8 8 _ { \pm 4 . 0 5 }$ </td><td> $5 3 . 2 5 _ { \pm 3 . 7 5 }$ </td><td> $5 5 . 1 4 _ { \pm 4 . 4 5 }$ </td><td> $5 3 . 7 5 _ { \pm 4 . 0 5 }$ </td><td> $5 9 . 9 5 _ { \pm 3 . 1 5 }$ </td><td> $5 7 . 8 5 _ { \pm 2 . 8 5 }$ </td><td> $8 1 . 1 4 _ { \pm 4 . 1 2 }$ </td><td> $7 0 . 8 2 _ { \pm 4 . 0 5 }$ </td><td> $7 0 . 1 2 _ { \pm 5 . 2 5 }$ </td><td> $6 3 . 4 5 _ { \pm 4 . 8 5 }$ </td></tr><tr><td>IRM</td><td> $6 0 . 1 5 _ { \pm 3 . 8 8 }$ </td><td> $5 6 . 7 8 { \scriptstyle \pm 3 . 4 5 }$ </td><td> $5 3 . 4 2 _ { \pm 4 . 6 5 }$ </td><td> $5 4 . 1 2 _ { \pm 4 . 1 1 }$ </td><td> $6 0 . 7 5 { \scriptstyle \pm 2 . 9 5 }$ </td><td> $5 7 . 4 4 { \scriptstyle \pm 2 . 7 5 }$ </td><td>80.22±3.94</td><td> $7 0 . 1 5 _ { \pm 3 . 8 8 }$ </td><td>71.25±5.08</td><td> $6 0 . 1 4 _ { \pm 4 . 9 5 }$ </td></tr><tr><td>GSAT</td><td> $5 9 . 8 5 _ { \pm 3 . 7 5 }$ </td><td> $5 7 . 1 4 _ { \pm 3 . 5 2 }$ </td><td> $5 2 . 4 5 _ { \pm 3 . 9 5 }$ </td><td> $5 2 . 1 2 _ { \pm 3 . 6 5 }$ </td><td> $5 8 . 4 5 _ { \pm 2 . 8 5 }$ </td><td> $5 5 . 7 5 { \scriptstyle \pm 2 . 5 5 }$ </td><td> $8 2 . 4 5 _ { \pm 3 . 6 5 }$ </td><td> $7 2 . 1 5 _ { \pm 3 . 8 5 }$ </td><td> $7 2 . 0 5 _ { \pm 5 . 1 4 }$ </td><td> $6 2 . 4 5 _ { \pm 4 . 9 2 }$ </td></tr><tr><td rowspan="4">Graph OOD</td><td>DisC</td><td> $5 8 . 1 2 _ { \pm 4 . 1 5 }$ </td><td> $5 7 . 3 3 _ { \pm 3 . 8 5 }$ </td><td> $5 4 . 1 5 _ { \pm 4 . 2 5 }$ </td><td> $5 3 . 8 5 _ { \pm 4 . 0 2 }$ </td><td> $5 9 . 8 8 _ { \pm 3 . 0 5 }$ </td><td> $5 7 . 1 2 { \scriptstyle \pm 2 . 7 5 }$ </td><td> $8 3 . 1 2 _ { \pm 3 . 8 5 }$ </td><td> $7 3 . 5 5 { \scriptstyle \pm 4 . 1 5 }$ </td><td> $7 0 . 4 5 _ { \pm 5 . 0 5 }$ </td><td> $5 9 . 7 5 { \scriptstyle \pm 4 . 8 5 }$ </td></tr><tr><td>CEPG</td><td> $5 3 . 4 5 _ { \pm 4 . 5 5 }$ </td><td>51.15±4.12</td><td> $4 6 . 8 5 _ { \pm 4 . 8 5 }$ </td><td>48.75±4.45</td><td> $5 7 . 1 5 _ { \pm 3 . 2 5 }$ </td><td>55.45±2.95</td><td> $7 6 . 1 5 _ { \pm 4 . 2 5 }$ </td><td>69.12±4.35</td><td> $6 4 . 1 2 _ { \pm 5 . 8 5 }$ </td><td>58.45±5.15</td></tr><tr><td>DiSCO</td><td> $5 8 . 2 5 _ { \pm 3 . 8 5 }$ </td><td> $5 6 . 4 5 _ { \pm 3 . 6 5 }$ </td><td> $5 3 . 1 2 _ { \pm 4 . 1 5 }$ </td><td> $5 2 . 7 5 _ { \pm 3 . 8 5 }$ </td><td> $6 0 . 1 5 _ { \pm 2 . 7 5 }$ </td><td> $5 7 . 1 5 _ { \pm 2 . 5 5 }$ </td><td> $8 5 . 1 2 _ { \pm 3 . 5 5 }$ </td><td> $7 4 . 1 5 _ { \pm 3 . 7 5 }$ </td><td> $6 8 . 7 5 { \scriptstyle \pm 5 . 1 5 }$ </td><td> $6 2 . 8 5 { \scriptstyle \pm 4 . 8 5 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">Netwiks Brain</td><td>BrainNetTF</td><td> $6 3 . 7 5 _ { \pm 3 . 5 5 }$ </td><td> ${ \bf 6 1 . 8 5 _ { \pm 3 . 2 5 } }$ </td><td> $5 8 . 4 5 _ { \pm 4 . 1 5 }$ </td><td> $5 4 . 1 5 _ { \pm 3 . 8 5 }$ </td><td> $6 2 . 8 5 _ { \pm 2 . 5 5 }$ </td><td>58.15±2.35</td><td> $7 9 . 4 5 _ { \pm 4 . 1 5 }$ </td><td>71.15±4.35</td><td> $6 9 . 1 5 _ { \pm 4 . 8 5 }$ </td><td> $5 8 . 1 5 _ { \pm 4 . 5 5 }$ </td></tr><tr><td>XG-GNN</td><td> $6 1 . 4 5 _ { \pm 3 . 7 5 }$ </td><td> $5 5 . 4 5 _ { \pm 3 . 4 5 }$ </td><td> $5 2 . 1 5 _ { \pm 4 . 5 5 }$ </td><td> $5 4 . 1 5 _ { \pm 4 . 1 5 }$ </td><td>60.45±2.95</td><td> $5 6 . 7 5 { \scriptstyle \pm 2 . 6 5 }$ </td><td> $8 4 . 7 5 _ { \pm 3 . 6 5 }$ </td><td> $7 3 . 8 5 _ { \pm 3 . 8 5 }$ </td><td> $7 2 . 4 5 _ { \pm 4 . 8 5 }$ </td><td> $5 7 . 1 5 _ { \pm 4 . 6 5 }$ </td></tr><tr><td>AGMGC</td><td> $5 9 . 8 5 _ { \pm 3 . 8 5 }$ </td><td> $5 2 . 7 5 _ { \pm 3 . 5 5 }$ </td><td> $5 7 . 1 5 _ { \pm 3 . 9 5 }$ </td><td> $5 2 . 1 5 _ { \pm 3 . 6 5 }$ </td><td> $5 8 . 8 5 _ { \pm 2 . 8 5 }$ </td><td> $5 5 . 1 5 _ { \pm 2 . 5 5 }$ </td><td> $8 6 . 4 5 _ { \pm 3 . 4 5 }$ </td><td> $7 7 . 1 5 _ { \pm 3 . 6 5 }$ </td><td> $7 1 . 8 5 _ { \pm 5 . 0 5 }$ </td><td> $6 4 . 1 5 _ { \pm 4 . 7 5 }$ </td></tr><tr><td>FC-HGNN</td><td> $5 7 . 1 5 _ { \pm 4 . 1 5 } ^ { - }$ </td><td> $5 3 . 8 5 _ { \pm 3 . 8 5 }$ </td><td> $5 6 . 4 5 _ { \pm 4 . 2 5 }$ </td><td> $5 4 . 7 5 _ { \pm 3 . 9 5 }$ </td><td> $6 1 . 4 5 _ { \pm 3 . 0 5 }$ </td><td> $5 7 . 1 5 _ { \pm 2 . 7 5 } ^ { - }$ </td><td> $7 7 . 8 5 _ { \pm 4 . 0 5 }$ </td><td> $6 8 . 7 5 _ { \pm 4 . 2 5 } ^ { - }$ </td><td> $7 2 . 1 5 _ { \pm 4 . 9 5 }$ </td><td> $6 0 . 4 5 _ { \pm 4 . 8 5 }$ </td></tr><tr><td>BrainOOD</td><td> $6 4 . 7 5 _ { \pm 3 . 4 5 }$ </td><td> $6 0 . 4 5 _ { \pm 3 . 1 5 }$ </td><td> $6 3 . 8 5 _ { \pm 3 . 6 5 }$ </td><td> $6 0 . 1 5 _ { \pm 3 . 3 5 }$ </td><td> $6 4 . 7 5 _ { \pm 2 . 7 5 }$ </td><td> $6 1 . 1 5 _ { \pm 2 . 4 5 }$ </td><td> $8 6 . 4 5 _ { \pm 3 . 2 5 }$ </td><td> $7 7 . 4 5 _ { \pm 3 . 4 5 }$ </td><td> $7 3 . 4 5 _ { \pm 4 . 7 5 }$ </td><td> $6 5 . 1 5 _ { \pm 4 . 5 5 }$ </td></tr><tr><td>DeCI</td><td> $5 7 . 4 5 _ { \pm 4 . 0 5 }$   $6 5 . 4 5 _ { \pm 3 . 2 5 }$ </td><td> $6 1 . 4 5 _ { \pm 3 . 5 5 } ^ { - }$   $6 0 . 8 5 _ { \pm 2 . 9 5 } ^ { - }$ </td><td> $6 0 . 1 5 _ { \pm 3 . 8 5 }$   $6 7 . 4 5 _ { \pm 3 . 4 5 }$ </td><td> ${ \bf 6 2 . 8 5 _ { \pm 3 . 4 5 } }$   $6 2 . 1 5 _ { \pm 3 . 1 5 }$ </td><td> $5 5 . 1 5 _ { \pm 3 . 1 5 }$   $6 5 . 8 5 _ { \pm 2 . 6 5 }$ </td><td> $5 5 . 8 5 _ { \pm 2 . 8 5 } ^ { - }$   $6 1 . 4 5 _ { \pm 2 . 3 5 }$ </td><td> $8 7 . 4 5 _ { \pm 3 . 3 5 }$ </td><td> $7 9 . 8 5 _ { \pm 3 . 5 5 } ^ { - }$   $\mathbf { 8 1 . 4 5 _ { \pm 3 . 2 5 } }$ </td><td> $5 6 . 4 5 _ { \pm 5 . 2 5 } ^ { - }$ </td><td> $6 3 . 4 5 _ { \pm 4 . 9 5 } ^ { - }$ </td></tr><tr><td>CORE BRIO</td><td> ${ \bf 6 6 . 8 5 _ { \pm 3 . 1 5 } }$ </td><td> $6 1 . 5 5 { \scriptstyle \pm 2 . 8 5 }$ </td><td> $\mathbf { 6 8 . 7 5 _ { \pm 3 . 2 5 } }$ </td><td> $6 2 . 6 5 _ { \pm 3 . 0 5 }$ </td><td> ${ \bf 6 6 . 4 5 _ { \pm 2 . 5 5 } }$ </td><td> ${ \bf 6 2 . 1 5 { \scriptstyle \pm 2 . 2 5 } }$ </td><td> $\mathbf { 8 8 . 7 5 _ { \pm 3 . 0 5 } }$   $8 8 . 1 5 _ { \pm 3 . 1 5 }$ </td><td> $8 0 . 7 5 { \scriptstyle \pm 3 . 3 5 }$ </td><td> $7 3 . 1 5 _ { \pm 4 . 5 5 }$   $7 6 . 4 5 _ { \pm 4 . 2 5 }$ </td><td> $6 7 . 4 5 _ { \pm 4 . 3 5 }$   ${ \bf 6 9 . 1 5 _ { \pm 4 . 0 5 } }$ </td></tr></table>

![](images/e1e353793864a272554c49a2c523a968d995b77b4f4612d8d826ac7d3263974f.jpg)

![](images/a01f3f0c289418ba6b37e9322a8ea2c868f69c0abed78f2d52f90410edd1fe3f.jpg)

![](images/1b8413e95298b561124f210e895adcd43d358a860dbf8cb72ca00c54e7d68e05.jpg)  
(a) Cross-method  
(b) Alignment  
(c) Mismatch

![](images/0e949ec128269030cbc2572baefb66c073e6f0081ccca1d9a4cb44f1c4d5650c.jpg)  
(d) Factor retention  
Figure 4: Cross-method relationships (a), alignment trajectories (b), mismatch rates (c), and factor retention (d) on REST-meta-MDD.

## I COMPLEXITY ANALYSIS

We analyze the per-step cost of BRIO after supervised warm-up, where the support S and assignment Π are fixed. Let B be the batch size, E the number of source sites, $\begin{array} { r } { N _ { \mathrm { s r c } } = \sum _ { e } N _ { e } } \end{array}$ the number of source training subjects, P the number of ROIs, T the number of time points, $s = \| s \| _ { 0 }$ the number of retained adjacency entries, J the number of factors, L the number of GNN layers, and $d = \operatorname* { m a x } ( d _ { h } , d _ { t } )$ . With sparse propagation, encoding and factorization cost $C _ { \mathrm { e n c } } = L ( s d + P d ^ { 2 } ) + P J d + J d ^ { 2 }$ per graph. For batches balanced across sites and classes, forming pair-specific representations and cost matrices takes $\mathcal { O } ( B ^ { 2 } J d )$ , and $I _ { \mathrm { s k } }$ Sinkhorn iterations over all cross- and self-transport problems take $\mathcal { O } ( B ^ { 2 } I _ { \mathrm { s k } } )$ . Every M steps, the qualification refresh generates K re-estimates per source subject at $C _ { \mathrm { F C } } \overset { \cdot } { = } \mathcal { O } ( P ^ { 2 } \overset { \cdot } { T } )$ each, encodes $K + 1$ graphs per subject, and forms qualifications over site pairs in $\mathcal { O } ( E ^ { 2 } \dot { J } )$ The amortized time per step is therefore $\mathcal { O } \big ( ( B + N _ { \mathrm { s r c } } ( K + 1 ) / M ) C _ { \mathrm { e n c } } + B ^ { 2 } ( J d + I _ { \mathrm { s k } } ) + ( N _ { \mathrm { s r c } } K C _ { \mathrm { F C } } + E ^ { 2 } J ) / M \big )$ . Relative to class-conditional alignment without qualification, the overhead is confined to the $1 / M$ terms. Since the refresh runs without gradients and discards re-estimates after use, peak memory is that of a training batch, $\mathcal { O } ( B L ( P d + s ) ^ { - } + B ^ { 2 } )$ ), independent of $N _ { \mathrm { s r c } }$ and K.

## J MORE RESULTS

## J.1 ID PERFORMANCE COMPARISON

In this part, we assess whether BRIO also performs well under in-distribution (ID) settings. Subjects from all sites are pooled and partitioned into ten folds stratified by site and class, so that every fold preserves the site and class composition of the full cohort. Each fold in turn serves as the test set, the next two folds serve as the validation set, and the remaining seven folds form the training set, giving a 7:2:1 rotation in which every subject is tested exactly once. All sites appear in training and every training site contains both classes, so source sites continue to serve as environments for alignment. As shown in Table 4, BRIO ranks first or second on both metrics across all five settings. It achieves the highest mean AUC on ABIDE, ABIDE (CC200), REST-meta-MDD, and ABCD, and the highest mean ACC on REST-meta-MDD and ABCD. BrainNetTF and DeCI obtain the highest ACC on ABIDE and ABIDE (CC200), respectively, while CORE leads both metrics on SRPBS, with BRIO closely following in each case. These results show that the predictive value of re-estimation-informed learning extends beyond unseen-site evaluation and supports competitive ID performance alongside the cross-site OOD results.

![](images/9dfe1c9470b27ccadd5d6fc47ec1865f3ecfd9ab62fef49d8a686683fde8cb07.jpg)  
(a) Cross-method

![](images/5b178d0421ec66f32a6b6128201a2cc894f09ef832fa457504e035703f1c2481.jpg)  
(b) Alignment

![](images/47195fbbe40bff574946fedd088b9ddf734acc1f1aedfbc372530e92eb768fdf.jpg)  
(c) Mismatch

![](images/12c7e761155deda440db78d2e3b378dafba3d04016cea8cfdc464c8774f4311f.jpg)  
(d) Factor retention

Figure 5: Cross-method relationships (a), alignment trajectories (b), mismatch rates (c), and factor  
![](images/9affcd1b8c224c1338448612bd80b75e63397ae182ba20ec47d0f893705dabb0.jpg)  
(a) Cross-method

![](images/39d8800dee358841fb31f990e5d860640ffdeadef8d80c38bf9c79e465f97d26.jpg)  
(b) Alignment

![](images/6bee526193160c404beafb8337d02b24ed0136e85d7ae1be2d6a0a937ba5e682.jpg)  
(c) Mismatch

![](images/874d194d1800b55c8d25f25012e9fcda045a10b44f16c58deca279f591e0d59e.jpg)  
(d) Factor retention  
Figure 6: Cross-method relationships (a), alignment trajectories (b), mismatch rates (c), and factor retention (d) on ABCD.

## J.2 MORE SITE DISCREPANCY AND RE-ESTIMATION VARIATION

In this part, we define the two evaluation metrics used in Sec. 4.4 and extend the cross-method comparison and alignment-strength sweep to REST-meta-MDD, SRPBS, and ABCD. The crossmethod comparison includes CORAL, IRM, GSAT, BrainOOD, CORE, BRIO w/o QG with $\lambda _ { \mathrm { e n v } } = 0$ and BRIO on ABIDE and REST-meta-MDD, and CORAL, BrainOOD, BRIO w/o QG, and BRIO on SRPBS and ABCD. Each method uses the checkpoint selected in the main experiment and is evaluated with all trained parameters frozen and stochastic operations disabled. All methods receive the same source training subjects, full-scan FC graphs, and FC re-estimates, and each observation corresponds to one method, held-out site, and random seed.

For each method, let $\widehat { P } _ { e , c }$ denote the empirical distribution of unit-normalized full-scan representations from source site e and class c, taken from the layer preceding the classifier. The class-conditional site discrepancy is

$$
\mathcal { D } _ { \mathrm { s i t e } } = \frac { 1 } { E ( E - 1 ) } \sum _ { 1 \leq e < e ^ { \prime } \leq E } \sum _ { c \in \{ 0 , 1 \} } S _ { \varepsilon } \left( \widehat { P } _ { e , c } , \widehat { P } _ { e ^ { \prime } , c } \right) ,\tag{83}
$$

where $E$ is the number of source sites. The debiased Sinkhorn divergence $S _ { \varepsilon }$ uses squared Euclidean distances and identical solver settings across methods, with the same representation in cross- and self-transport terms and without qualification-based weighting. Lower values indicate closer full-scan representations across source sites under a common geometry.

To measure predictive changes, let $\ell _ { i } ^ { ( k ) }$ be the binary logit margin, given by the pre-sigmoid logit or the difference between the two softmax logits; a common shift of two logits leaves the predicted probability unchanged and is thus excluded. With k = 0 denoting the full-scan prediction, the normalized predictive re-estimation variation is

$$
\widetilde { \mathcal { D } } _ { \mathrm { r e } } ^ { \mathrm { p r e d } } = \frac { 1 } { 2 E } \sum _ { e = 1 } ^ { E } \sum _ { c \in \{ 0 , 1 \} } \frac { \widehat { \mathbb { E } } _ { i , k | e , c } \left[ \left( \ell _ { i } ^ { ( k ) } - \ell _ { i } ^ { ( 0 ) } \right) ^ { 2 } \right] } { V _ { e , c } ^ { \mathrm { p r e d } } + \frac { 1 } { 4 } \left( \mu _ { e , 1 } ^ { \mathrm { p r e d } } - \mu _ { e , 0 } ^ { \mathrm { p r e d } } \right) ^ { 2 } + \epsilon } ,\tag{84}
$$

![](images/294d7ef850735d4198a014d9bd0b074de3a6881e49916044e3a2ac31df3f4ec4.jpg)

![](images/7012124cebbb3c2611d5c05300205322f47ac5d04d29ef172f0829dd575dcb7e.jpg)

![](images/a5526a10494907e7d353f98b8f235e47fa5760105331e5a2720d636cb9c6ccff.jpg)  
(b) SRPBS  
(c) ABCD

(a) ABIDE (CC200)  
Figure 7: Ablation study on ABIDE (CC200), SRPBS, and ABCD.  
![](images/d429b538845257fbb8d80366fecc745faeb2005ba38a633c9ae02b227be0cf11.jpg)  
(a) ABIDE (CC200)

![](images/38030f0f2fd0b1f4ccdeeb9c8e9c3210be1845f6f8e52f68b707b0b632317dde.jpg)  
(b) SRPBS

![](images/f6255db55c97d7b452105c13479e1fa6f0bd2d6ef9dd151668603a0ea73593dd.jpg)  
(c) ABCD  
Figure 8: Sensitivity study on ABIDE (CC200), SRPBS, and ABCD.

where the empirical expectation averages over source subjects and their re-estimates with $k \geq 1$ $\mu _ { e , c } ^ { \mathrm { p r e d } }$ and $V _ { e , c } ^ { \mathrm { p r e d } }$ are the full-scan logit mean and variance within each site and class, and $\epsilon > 0$ ensures numerical stability. Numerator and denominator scale identically under a global rescaling of the logits, so the ratio measures re-estimation change relative to the within-site spread and class separation of full-scan predictions.

This output-level variation and the factor-level re-estimation support in Sec. 3.2 capture different aspects of stability. Since $z _ { i } = J ^ { - 1 / 2 } \textstyle \sum _ { j } a _ { i , j } + b ,$ a re-estimation change of the logit satisfies

$$
\mathbb { E } \big [ ( \delta z _ { i } ) ^ { 2 } \big ] = \frac { 1 } { J } \Big [ \sum _ { j } \mathbb { E } \big [ ( \delta a _ { i , j } ) ^ { 2 } \big ] + 2 \sum _ { j < l } \mathbb { E } [ \delta a _ { i , j } \delta a _ { i , l } ] \Big ] ,\tag{85}
$$

so changes of individual factors may add or cancel at the output, and the two quantities are reported separately.

The cross-method plots show method-level means, with Spearman $\rho$ computed across individual runs. $R _ { \mathrm { d i s c o r d } }$ is the percentage of method pairs within the same fold and seed for which smaller $\mathcal { D } _ { \mathrm { s i t e } }$ corresponds to larger $\mathcal { \widetilde { D } } _ { \mathrm { r e } } ^ { \mathrm { p r e d } }$ , excluding ties. A run is a mismatch when its $\mathcal { D } _ { \mathrm { s i t e } }$ falls below and its $\mathcal { \widetilde { D } } _ { \mathrm { r e } } ^ { \mathrm { p r e d } }$ falls above the pooled dataset medians. The total mismatch rate is the percentage of mismatches among all evaluated runs, and the AUC-preserved mismatch rate counts only mismatches whose source-validation AUC is no more than a tolerance $\delta _ { \mathrm { A U C } }$ below that of BRIO w/o QG with $\lambda _ { \mathrm { e n v } } = 0$ , again over all evaluated runs.

For the controlled sweep, BRIO w/o QG shares the graph encoder, factor representation, classifier, and training procedure of BRIO, and removes qualification guidance by assigning equal weights to all factors and disabling qualification-based strength modulation. We train this variant independently for each tested value of $\lambda _ { \mathrm { e n v } }$ and select its checkpoint by the same source-validation rule as the main experiment. Runs are paired by fold and seed, and the trajectories show means connected in increasing $\lambda _ { \mathrm { e n v } }$ . The full BRIO is added as a separate reference point, and Matched w/o QG denotes the setting with positive $\lambda _ { \mathrm { e n v } }$ selected by source validation.

Panels (a) of Figs. 4, 5, and 6 show positive rank correlations between the two metrics on REST-meta-MDD and SRPBS but zero rank correlation on ABCD, and $R _ { \mathrm { d i s c o r d } }$ is nonzero on every dataset, so lower site discrepancy does not consistently imply lower predictive variation under FC re-estimation. The controlled sweeps in panels (b) show that stronger alignment reduces both quantities, while the full BRIO achieves lower variation than BRIO w/o QG at comparable site discrepancy, indicating that qualification-guided weighting and strength modulation offer benefits beyond simply increasing alignment strength. Panels (c) further show that BRIO achieves lower total and AUC-preserved mismatch rates than the evaluated w/o QG settings on all three datasets, indicating fewer mismatches even when predictive performance is preserved.

Table 5: Sensitivity of BRIO to moving-block length $\ell _ { b } .$ . Bold indicates the highest mean.
<table><tr><td rowspan="2">Dataset</td><td colspan="3"> $0 . 5 \times$ </td><td colspan="3">1× (Default)</td><td colspan="3"> $2 \times$ </td></tr><tr><td> $\ell _ { b }$ </td><td>AUC</td><td>ACC</td><td> $\ell _ { b }$ </td><td>AUC</td><td>ACC</td><td> $\ell _ { b }$ </td><td>AUC</td><td>ACC</td></tr><tr><td>ABIDE</td><td>| 3 (6.0)</td><td> $6 5 . 8 4 _ { \pm 4 . 6 7 }$ </td><td> $6 0 . 2 5 { \scriptstyle \pm 4 . 1 5 }$ </td><td>6 (12.0)</td><td> ${ \bf 6 7 . 5 3 _ { \pm 4 . 1 2 } }$ </td><td> ${ \bf 6 2 . 1 8 { \scriptstyle \pm 3 . 6 8 } }$ </td><td>12 (24.0)</td><td> $6 6 . 7 1 { \scriptstyle \pm 4 . 2 8 }$ </td><td> $6 1 . 5 4 _ { \pm 3 . 9 2 }$ </td></tr><tr><td>ABIDE (CC200)</td><td>3 (6.0)</td><td> $6 9 . 4 5 _ { \pm 5 . 2 5 }$ </td><td> $6 0 . 1 2 { \scriptstyle \pm 4 . 5 8 }$ </td><td>6 (12.0)</td><td> $\mathbf { 7 0 . 8 6 } _ { \pm 4 . 8 7 }$ </td><td> ${ \bf 6 1 . 9 2 } _ { \pm 3 . 9 4 }$ </td><td>12 (24.0)</td><td> $6 8 . 7 2 _ { \pm 5 . 1 5 }$ </td><td> $5 9 . 8 8 { \scriptstyle \pm 4 . 2 5 }$ </td></tr><tr><td>REST-meta-MDD</td><td>4 (8.0)</td><td> $6 7 . 8 5 _ { \pm 4 . 2 2 }$ </td><td> $6 3 . 3 8 { \scriptstyle \pm 3 . 4 2 }$ </td><td>7 (14.0)</td><td> ${ \bf 6 8 . 8 4 _ { \pm 3 . 7 5 } }$ </td><td> ${ \bf 6 4 . 4 2 _ { \pm 2 . 9 1 } }$ </td><td>14 (28.0)</td><td> $6 7 . 6 2 { \scriptstyle \pm 3 . 8 2 }$ </td><td> $6 3 . 8 5 _ { \pm 3 . 1 8 }$ </td></tr><tr><td>SRPBS</td><td>3 (6.0)</td><td> $8 1 . 3 5 _ { \pm 6 . 3 5 }$ </td><td> $7 6 . 5 4 _ { \pm 5 . 8 2 }$ </td><td>6 (12.0)</td><td> $\mathbf { 8 3 . 9 1 _ { \pm 5 . 6 3 } }$ </td><td> $7 8 . 8 2 _ { \pm 5 . 2 8 }$ </td><td>12 (24.0)</td><td> $8 3 . 7 2 _ { \pm 5 . 5 8 }$ </td><td> $7 8 . 4 5 _ { \pm 5 . 1 5 }$ </td></tr><tr><td>ABCD (ADHD-task)</td><td>7 (5.6)</td><td> $7 2 . 1 5 _ { \pm 5 . 4 5 }$ </td><td> $6 5 . 1 2 _ { \pm 4 . 8 5 }$ </td><td>15 (12.0)</td><td> $7 5 . 9 8 _ { \pm 4 . 3 9 }$ </td><td> ${ \bf 6 8 . 7 4 _ { \pm 3 . 8 2 } }$ </td><td>30 (24.0)</td><td> $7 5 . 4 5 _ { \pm 4 . 5 2 }$ </td><td> $6 7 . 9 2 _ { \pm 3 . 9 5 }$ </td></tr></table>

## J.3 MORE QUALIFICATION AND FACTOR TRANSFERABILITY

In this part, we extend the factor-retention analysis to REST-meta-MDD, SRPBS, and ABCD. As shown in panels (d) of Figs. 4, 5, and 6, full qualification outperforms the other rankings when retaining 25% or 50% of factors across all three datasets. Its advantage over task relevance alone shows that re-estimation support provides complementary information for prioritizing factors with predictive value on unseen sites. Both rankings outperform support alone at partial retention, and support alone falls below random ordering on ABCD, so re-estimation support identifies transferable factors only together with task relevance. Performance gaps narrow as more factors are retained and vanish at full retention, where all rankings recover the original predictions.

## J.4 MORE ABLATION STUDIES

In this part, we extend the ablation study to ABIDE (CC200), SRPBS, and ABCD using the same four variants as in Sec.4.2. As shown in Fig. 7, BRIO achieves the highest AUC and ACC across all three settings. BRIO w/o RS shows the largest performance drop on ABIDE (CC200), whereas BRIO w/o TR yields the lowest performance on SRPBS and ABCD. These differences suggest that the relative contributions of re-estimation support and task relevance vary across datasets, while their joint use consistently benefits cross-site prediction. BRIO w/o SM and BRIO w/o SF also underperform the full model on both metrics, supporting the complementary roles of modulating overall alignment strength and prioritizing supported factors in cross-site matching.

## J.5 MORE SENSITIVITY ANALYSIS

First, we extend the sensitivity analysis of the number of FC re-estimates K and connectome factors J to ABIDE (CC200), SRPBS, and ABCD. As shown in Fig. 8, increasing K from small values generally improves AUC, whereas further increases yield no consistent gains. This suggests that additional re-estimates help characterize variability in predictive contributions, with diminishing benefits at larger budgets. Intermediate values of J also tend to outperform very small or large factor counts, indicating that finer factorization does not necessarily improve cross-site prediction.

Additionally, we examine the sensitivity of BRIO to the temporal block length $\ell _ { b }$ used for within-scan FC re-estimation. Whereas K controls the number of re-estimates, $\ell _ { b }$ determines the length of contiguous temporal segments retained during resampling. We compare shorter, default, and longer blocks at nominal scales of $0 . 5 \times , 1 \times$ , and 2×: the 1× setting is the fold-level estimate rounded to the nearest integer, and the 0.5× and 2× settings scale the unrounded estimate before rounding and are used as fixed block lengths. Table 5 reports each block length in time points together with its duration in seconds. The default block length achieves the highest mean AUC and ACC in all five settings, and the estimated defaults correspond to a similar physical duration across datasets despite their different repetition times, indicating that the autocorrelation-based estimate captures a temporal scale that transfers across acquisition protocols. Both halving and doubling the block length degrade performance, with halving being more harmful on most datasets; blocks that are too short break the temporal dependence that FC re-estimation is meant to preserve, whereas overly long blocks reduce the diversity of re-estimates.

![](images/54227f1196fe01f86d952c7312e91547ceeab41f9229e7b076a99e07d57a5fd5.jpg)  
(a) ABIDE

![](images/4f58bf7be908d81822a9b898cbf28ab68d86680a6ecbc52e4ec6600fd07f40d7.jpg)  
(b) REST-meta-MDD

![](images/ec4a3006bc0ccfae325633bd9f89e20e49c6c665c1295eb3c6a0a89af54fd39f.jpg)  
(c) SRPBS

![](images/0df670381f251996b2be8bb4d91e5d16f92af472fe3b809b08118518ea87a35c.jpg)  
(d) ABCD  
Figure 9: Performance under reduced scan data on different datasets.

Table 6: Training time (in seconds) and GPU memory consumption (in GB) for each training epoch.
<table><tr><td rowspan="2">Method</td><td colspan="2">ABIDE</td><td colspan="2">REST-meta-MDD</td><td colspan="2">SRPBS</td><td colspan="2">ABCD</td></tr><tr><td>Time</td><td>Memory</td><td>Time</td><td>Memory</td><td>Time</td><td>Memory</td><td>Time</td><td>Memory</td></tr><tr><td>GSAT</td><td>0.68</td><td>2.1</td><td>1.52</td><td>3.3</td><td>0.48</td><td>2.0</td><td>0.34</td><td>1.7</td></tr><tr><td>DisC</td><td>1.05</td><td>2.4</td><td>2.38</td><td>3.8</td><td>0.74</td><td>2.2</td><td>0.51</td><td>1.9</td></tr><tr><td>CEPG</td><td>0.58</td><td>2.0</td><td>1.34</td><td>3.1</td><td>0.41</td><td>1.9</td><td>0.28</td><td>1.6</td></tr><tr><td>DiSCO</td><td>1.65</td><td>2.9</td><td>3.85</td><td>4.6</td><td>1.18</td><td>2.7</td><td>0.78</td><td>2.3</td></tr><tr><td>BrainNetTF</td><td>0.78</td><td>2.0</td><td>1.65</td><td>3.2</td><td>0.49</td><td>1.9</td><td>0.32</td><td>1.6</td></tr><tr><td>AGMGC</td><td>0.71</td><td>1.8</td><td>1.58</td><td>3.0</td><td>0.46</td><td>1.8</td><td>0.30</td><td>1.5</td></tr><tr><td>FC-HGNN</td><td>0.34</td><td>1.8</td><td>0.48</td><td>3.1</td><td>0.19</td><td>1.8</td><td>0.12</td><td>1.5</td></tr><tr><td>XG-GNN</td><td>0.63</td><td>2.1</td><td>1.31</td><td>3.4</td><td>0.38</td><td>2.0</td><td>0.25</td><td>1.7</td></tr><tr><td>DeCI</td><td>0.88</td><td>2.3</td><td>1.95</td><td>3.6</td><td>0.62</td><td>2.1</td><td>0.44</td><td>1.8</td></tr><tr><td>BrainOOD</td><td>1.45</td><td>2.5</td><td>3.32</td><td>4.1</td><td>0.88</td><td>2.3</td><td>0.58</td><td>2.0</td></tr><tr><td>CORE</td><td>0.35</td><td>2.0</td><td>0.52</td><td>3.7</td><td>0.22</td><td>1.9</td><td>0.12</td><td>1.6</td></tr><tr><td>BRIO</td><td>1.22</td><td>2.6</td><td>2.85</td><td>4.2</td><td>0.86</td><td>2.4</td><td>0.56</td><td>2.1</td></tr></table>

## J.6 EFFECT OF SCAN LENGTH ON PREDICTIVE PERFORMANCE

To examine how reduced temporal data for target FC estimation affects cross-site prediction, we evaluate BRIO, BRIO w/o RS, and BRIO w/o QG on ABIDE, REST-meta-MDD, SRPBS, and ABCD. We reconstruct FC graphs using 75% or 50% of each subject’s preprocessed BOLD sequence, keeping all source-selected checkpoints fixed. At each fraction, AUC is computed separately for beginning, middle, and end windows and then averaged across positions. We report ∆AUC relative to each model’s full-scan performance. As shown in Fig. 9, all three models exhibit larger AUC reductions when fewer time points are retained, but BRIO consistently shows the smallest decline. BRIO w/o RS incurs smaller drops than BRIO w/o QG, while remaining more sensitive than the full model. The differences in degradation widen at the lower scan fraction, suggesting that reestimation support provides benefits beyond task relevance alone. These results support the value of re-estimation-informed learning for maintaining cross-site prediction performance when target FC graphs are estimated from fewer temporal observations.

## J.7 TRAINING EFFICIENCY AND RESOURCE CONSUMPTION

In this part, we compare the training efficiency of BRIO and representative baselines on four datasets. Table 6 reports per-epoch training time in seconds and GPU memory consumption in GB. We observe that BRIO requires more training time and GPU memory than lightweight baselines such as FC-HGNN and CORE. Compared with BrainOOD, it achieves shorter per-epoch training times while using slightly more memory, reflecting a trade-off between runtime and memory consumption. In contrast, BRIO requires less time and memory than DiSCO. These relative resource-use patterns remain consistent across all evaluated datasets.

## K INTERPRETABILITY ANALYSIS

To examine how qualification-based factor weighting translates into regional pooling emphasis, we analyze the models trained on ABIDE and REST-meta-MDD using AAL116. For each model, we first summarize the normalized qualifications $\overline { { \alpha } } _ { e , e ^ { \prime } , c , j }$ over all unordered source-site pairs and both

![](images/4439ee1de1446b15fc736313e67ec6ecef09e53e343ea480c9cc4781faa4fa48.jpg)  
Figure 10: Anatomical distribution of qualification-weighted ROI profiles on ABIDE and REST-meta-MDD.

classes:

$$
\overline { { \alpha } } _ { j } ^ { \mathrm { g l o b a l } } = \frac { 1 } { E ( E - 1 ) } \sum _ { \stackrel { e , e ^ { \prime } \in \mathcal { E } _ { \mathrm { t r } } } { e < e ^ { \prime } } } \sum _ { c \in \{ 0 , 1 \} } \overline { { \alpha } } _ { e , e ^ { \prime } , c , j } , \qquad \sum _ { j = 1 } ^ { J } \overline { { \alpha } } _ { j } ^ { \mathrm { g l o b a l } } = 1 .\tag{86}
$$

This equal-weight summary captures relative factor allocation across source-site pairs and classes, separately from the overall alignment strength controlled by $g _ { e , e ^ { \prime } , c }$ . We then project these qualifications through the factor–ROI assignment Π retained after supervised warm-up. For ROI u, the resulting score and its normalization are

$$
S _ { u } = \sum _ { j = 1 } ^ { J } \overline { { \alpha } } _ { j } ^ { \mathrm { g l o b a l } } \overline { { \Pi } } _ { j , u } , \qquad \widehat { S } _ { u } = \frac { S _ { u } } { \sum _ { v = 1 } ^ { P } S _ { v } } = J S _ { u } .\tag{87}
$$

The assignment row masses of $1 / J$ give $\begin{array} { r } { \sum _ { u } S _ { u } = 1 / J _ { \cdot } } \end{array}$ so $\widehat { S } _ { u }$ defines a normalized ROI profile. Higher scores indicate greater pooling emphasis in factors with higher average source qualification. Under uniform factor weights, the column-mass constraint yields $\widehat { S } _ { u } = 1 / P$ for every ROI. Regional differences therefore reflect how qualification-based factor weighting interacts with the learned ROI assignment.

To summarize these profiles across LOSO folds and random seeds, we compute $\widehat { S } _ { u } ^ { ( r , s ) }$ separately for each fold r and seed s, using the qualifications and retained assignment of that model. Since factor indices need not correspond across independently trained models, we aggregate the scores in the common AAL116 ROI space:

$$
S _ { u } ^ { \mathrm { d a t a s e t } } = \frac { 1 } { | \mathcal { R } | | S | } \sum _ { r \in \mathcal { R } } \sum _ { s \in \mathcal { S } } \widehat { S } _ { u } ^ { ( r , s ) } , \qquad \sum _ { u = 1 } ^ { P } S _ { u } ^ { \mathrm { d a t a s e t } } = 1 ,\tag{88}
$$

where R and S denote the evaluated folds and seeds. This procedure preserves each model’s factor–ROI correspondence during projection and yields a dataset-level summary of regional pooling emphasis.

Fig. 10 visualizes the five ROIs with the highest dataset-averaged scores. On ABIDE, these are the left precuneus, right superior temporal gyrus, right posterior cingulate gyrus, left medial superior frontal gyrus, and right fusiform gyrus. On REST-meta-MDD, the selected regions are the left medial superior frontal gyrus, left amygdala, right posterior cingulate gyrus, right hippocampus, and right anterior cingulate gyrus. The right posterior cingulate and left medial superior frontal regions appear in both profiles, while the remaining selected ROIs differ between datasets. These shared and differing regions illustrate how the factors prioritized by source-side qualification distribute pooling emphasis across the anatomical space. The profiles complement the ablation and factor-retention analyses by connecting the relative factor weights used for alignment to their regional pooling structure.

Algorithm 1 Training and Inference of BRIO   
Require: Source datasets $\{ \mathcal { D } ^ { e } \} _ { e \in \mathcal { E } _ { \mathrm { t r } } }$ with $E \geq 2$ and $N _ { e , c } > 0 ;$ unseen-site sequences $\{ X _ { z } \} _ { z = 1 } ^ { N _ { \mathrm { t e } } } ; \Psi , \rho _ { g } , J ,$   
and K; block-length rule with autocorrelation threshold $\tau _ { \mathrm { a c } }$ and bounds $[ \ell _ { \mathrm { m i n } } , \bar { \ell _ { \mathrm { m a x } } } ]$ in time points; warm-up   
epochs $T _ { \mathrm { w } } ;$ qualification interval M in optimization steps; $\varepsilon , \lambda _ { \mathrm { e n v } } ,$ , and a training stopping criterion.   
Ensure: Unseen-site prediction probabilities $\widehat { \pmb { p } } _ { \mathrm { t e } }$   
1: Cache source full-scan graphs $A _ { i } ^ { e , ( 0 ) } = \Psi ( X _ { i } ^ { e } )$ and construct $\pmb { S }$ using Eq. (3).   
2: Estimate the fold-level block length $\ell _ { b }$ from autocorrelation horizons of training-subject ROI   
signals with threshold $\tau _ { \mathrm { a c } }$ as in Appendix E.2; for every source subject set $\cdot \ell _ { b , i } ^ { e } \gets$   
clip  round(ℓ ), ℓ , min $( \ell _ { \mathrm { m a x } } , \lfloor T _ { i } ^ { e } / 2 \rfloor ) \big )$   
3: Initialize all learnable parameters Θ; set $\vec { \Theta } _ { \mathrm { a s g } } = \{ Q , \omega \}$ and $\mathcal { C } _ { Q } \gets \mathcal { D } .$   
4: Use mini-batches with equally sized nonempty subsets $\scriptstyle B _ { e , c }$ for every source site and class; evaluate ${ \mathcal L } _ { \mathrm { c l s } }$   
using $\scriptstyle B _ { e , c }$ and $| B _ { e , c } |$ in place of $\mathcal { T } _ { e , c }$ and $N _ { e , c } .$   
5: Stage 1: Supervised Warm-Up and Factor Anchoring   
6: Enable training mode and gradients.   
7: for $t _ { \mathrm { w } } = 1 , \ldots , T _ { \mathrm { w } }$ do   
8: for each source mini-batch $\boldsymbol { B }$ do   
9: Compute Π, full-scan factors, and predictions using Eqs. (4)–(10); update Θ using ${ \mathcal { L } } _ { \mathrm { c l s } } .$   
10: end for   
11: end for   
12: In evaluation mode without gradients, cache Π $ \mathcal { A } _ { \omega } ( Q , E ^ { \mathrm { R O I } } ) .$   
13: Freeze $\Theta _ { \mathrm { a s g } }$ and use Π for subsequent pooling, while keeping ROI embeddings trainable.   
14: Stage 2: Periodic Qualification-Guided Learning   
15: Set $t  0 .$   
16: while the training stopping criterion is not met do   
17: if t mod $M \stackrel { = } { = } 0$ then   
18: Switch to evaluation mode, disable gradients, and set $\mathcal { C } _ { Q } \gets \mathcal { D } .$   
19: For every source subject $i \in \mathcal { T } _ { e , c } ,$ construct K FC re-estimates with block length $\ell _ { b , i } ^ { e }$ using Eq. (2).   
20: Compute paired contributions $\{ a _ { i , j } ^ { e , ( k ) } \} _ { k = 0 } ^ { K }$ using fixed S and Π.   
21: Compute $R _ { j } ^ { \mathrm { t a s k } }$ using Eqs. (11)–(13).   
22: $\begin{array} { r } { \mathbf { i f } \sum _ { j } R _ { j } ^ { \mathrm { t a s k } } > 0 } \end{array}$ then   
23: Compute $\pi _ { j } ^ { \mathrm { t a s k } }$ using Eq. (13) and $q _ { e , c , j } ^ { \mathrm { r e } }$ using Eqs. (14)–(16).   
24: Compute and cache $\mathcal { C } _ { Q } \gets \{ g _ { e , e ^ { \prime } , c } , \overline { { \alpha } } _ { e , e ^ { \prime } , c , j } \} _ { e < e ^ { \prime } , c , j }$ using Eqs. (18)–(19).   
25: end if   
26: Discard temporary FC re-estimates and restore training mode and gradients.   
27: end if   
28: Sample $\begin{array} { r } { B ; { } } \end{array}$ compute full-scan factors, predictions, and ${ \mathcal L } _ { \mathrm { c l s } }$ using fixed S and $\overline { { \mathbf { \Pi } } }$   
29: Set $\dot { \mathcal { L } } _ { \mathrm { e n v } } ^ { Q }  0 .$   
30: i $\mathcal { C } _ { Q } \neq \emptyset$ then   
31: Construct weighted representations and empirical distributions using cached α and Eqs. (22)–(24).   
32: Evaluate $\mathcal { L } _ { \mathrm { e n v } } ^ { Q }$ using cached g and Eq. (25), with the same pairwise representation map and ε in all   
cross- and self-transport terms.   
33: end if   
34: Update $\Theta \setminus \Theta _ { \mathrm { a s g } }$ using $\mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { e n v } } \mathcal { L } _ { \mathrm { e n v } } ^ { Q }$ , holding S, Π, and $\mathcal { C } _ { Q }$ fixed.   
35: Set $t \gets t + 1 .$   
36: end while   
37: Stage 3: Inference on Unseen Sites   
38: Switch to evaluation mode and disable gradients.   
39: for $z = 1 , \ldots , N _ { \mathrm { t e } }$ do   
40: Compute $A _ { z } ^ { ( 0 ) } = \Psi ( X _ { z } )$ and full-scan factors $\{ r _ { z , j } ^ { ( 0 ) } \} _ { j = 1 } ^ { J }$ using fixed $s , { \overline { { \Pi } } } ,$ and the trained mappings.   
41: Predict $\begin{array} { r } { \widehat { p } _ { z } = \sigma \Big ( J ^ { - 1 / 2 } \sum _ { j = 1 } ^ { J } \pmb { w } _ { j } ^ { \top } \pmb { r } _ { z , j } ^ { ( 0 ) } + b \Big ) } \end{array}$   
42: end for   
43: return $\widehat { \pmb { p } } _ { \mathrm { t e } } = [ \widehat { p } _ { 1 } , \dots , \widehat { p } _ { N _ { \mathrm { t e } } } ] ^ { \top } .$