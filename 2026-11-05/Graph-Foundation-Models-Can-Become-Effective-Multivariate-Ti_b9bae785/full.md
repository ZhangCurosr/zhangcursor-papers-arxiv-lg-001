# Graph Foundation Models Can Become Effective Multivariate Time Series Anomaly Detectors

Zhuangyan Hu, Zhuoxuan Liang, Wei Li, Member, IEEE, Xiaohua Jia, Fellow, IEEE, Moustafa Youssef, Fellow, IEEE, Sumi Helal, Fellow, IEEE

Abstract—Multivariate time series anomaly detection (MT-SAD) relies on modeling both temporal dynamics and intersensor dependencies to identify complex system faults. Although graph-based detectors explicitly model sensor topologies, they typically train task-specific networks from scratch on limited target-domain data, remaining vulnerable to spurious correlations and unable to reuse transferable relational priors. While pretrained Graph Foundation Models (GFMs) offer transferable topological priors and anomaly discriminability, directly adapting them to MTSAD is hindered by cross-modal feature-topology mismatch, unreliable pseudo-anomaly supervision, and decision distortion during node-to-window aggregation. To fill these gaps, we propose a novel adaptation framework, GFM4MTSAD, which unleashes pretrained GFMs for unsupervised MTSAD with three key innovations. ① We introduce the Structure-and-Information Contract Alignment (SICA) module to bridge continuous time series and the pretrained GFM space. This module harmonizes principal subspace token projection, normal-prior-guided graph construction, and residual recovery to suppress spurious edges while retaining fine-grained temporal details. ② We propose a Mechanism-Aligned Pseudo-Anomaly Screening strategy (MAPS) to steer unsupervised learning. It applies localized dependency breaks while preserving empirical residuals, selecting hard negatives near the normal boundary for targeted supervision. ③ We develop a Tail-Gated Evidence Calibration mechanism (TGEC) to eliminate node-to-window aggregation distortion. It calibrates multi-granular graph responses against normal operational references, strictly preserving benign score rankings while selectively refining extreme anomaly candidates in the tail distribution for reliable window detection and interpretable sensor evidence. Extensive evaluations across three real-world datasets with two GFM backbones demonstrate that GFM4MTSAD achieves superior detection performance over existing baselines.

Index Terms—Graph foundation models, anomaly detection, multivariate time series, contrastive learning.

## I. INTRODUCTION

This work was supported by the National Science and Technology Major Project under Grants 2026ZD0127300 and 2026ZD0127303, and by the Supporting Fund of Intelligent Internet of Things and Crowd Computing under Grant B25029.

Zhuangyan Hu and Wei Li are with the College of Computer Science and Technology, Harbin Engineering University, Harbin 150001, China (e-mail: {zhuangyan.hu,wei.li}@hrbeu.edu.cn).

Zhuoxuan Liang is with the Department of Computer Science, University of Exeter, Exeter, U.K. (e-mail: zl655@exeter.ac.uk).

Xiaohua Jia is with the Department of Computer Science, City University of Hong Kong, Hong Kong, China (e-mail: csjia@cityu.edu.hk).

Moustafa Youssef is with the Department of Computer Science and Engineering, The American University in Cairo, Egypt (e-mail: moustafayoussef@aucegypt.edu).

Sumi Helal is with the Department of Computer Science and Engineering, University of Bologna, Italy (e-mail: sumi.helal@unibo.it).

Zhuangyan Hu and Zhuoxuan Liang contributed equally to this research. Corresponding authors: Wei Li.

ULTIVARIATE time series anomaly detection (MTdustrial control [1], [2], healthcare [3], [4], and finance [5]– [7], where missed faults risk catastrophic failures [8] or safety hazards [6], [7]. Given scarce real-world anomalies and prohibitive annotation costs, MTSAD typically operates under an unsupervised regime trained solely on normal observations [9]. Moreover, anomalies manifest not only as isolated point deviations, but also as subtle breakdowns in temporal trajectories [10] and cross-sensor correlations [6], demanding intricate spatio-temporal modeling to deliver both reliable window-level detection and fine-grained sensor localization [11].

Existing unsupervised detectors largely rely on temporal proxy objectives, such as reconstruction errors (e.g., USAD [12], OmniAnomaly [13]) or pattern sensitivity learning [14], [15], yet fail to capture breakdowns in inter-sensor structural constraints. Graph-centric paradigms address this by modeling series as sensor graphs via Graph Neural Networks (GNNs, e.g., GDN [16], CGSTA [17], LaGraph [18]) or causality-aware contrastive perturbations (e.g., CAROTS [19]). Nevertheless, these models ubiquitously follow a train-fromscratch regime on limited target-domain data, leaving learned topologies susceptible to spurious edges and unable to transfer relational priors across systems. Coherently translating nodelevel responses into calibrated window decisions [9] for dependable fault attribution [20] also remains underexplored.

Recently, pretrained Graph Foundation Models (GFMs, e.g., AnomalyGFM [21], OwlEye [22]) have offered transferable topological priors and intrinsic node-level anomaly discriminability learned from multi-domain graphs. However, a fundamental modality rift separates continuous temporal dynamics from static graph spaces: naive adaptation fails to activate pretrained priors [23], amplifies structural noise [18], and discards temporal anomaly cues [22]. This raises a key research question: how to synergistically align the temporal dynamics and topological structures of multivariate time series (MTS) with pretrained GFMs?

Yet, achieving such alignment in practical MTSAD extends beyond standalone feature projection; it requires an endto-end adaptation pipeline that bridges dynamic temporaltopological inputs into the GFM space, steers boundary-aware discrimination without ground-truth labels, and translates finegrained node responses into calibrated window decisions. Along this pipeline, we confront three intertwined challenges, as illustrated in Fig. 1 (left):

C1: Contractual Mismatch Between Dynamic MTS and Pretrained GFMs. Bridging continuous time series and discrete GFMs faces a dual mismatch in feature representations and graph topologies. On the feature side, high-dimensional dynamic embeddings from temporal frontends exhibit an inherent input gap with the native token space of GFMs; crude linear projection or compression inevitably obliterates finegrained temporal anomaly cues [22], [23]. On the topological side, empirically constructed dynamic graphs are plagued by noise [8] and spurious edges [15]. Feeding such uncalibrated topologies into a pretrained GFM triggers noise amplification and over-smoothing during message passing [18], corrupting its relational priors. Establishing a structure-and-information alignment contract while preserving pretrained GFM knowledge thus constitutes the primary bottleneck for effective temporal-topological adaptation.

![](images/22a26919b31874de6620c2468b7879bdfb01f02d5969db7a26a291dc9979f57f.jpg)  
Fig. 1. Left: Three key challenges in adapting pretrained GFMs to MTSAD and their solutions: (C1) Contract mismatch, addressed by SICA through representation alignment and normal prior routing; (C2) Unreliable supervision, addressed by MAPS through dependency disruption and hard-negative selection; and (C3) Node-window distortion, addressed by TGEC through normality-guided tail calibration. Right: AUROC comparison across three benchmarks against nine representative baselines, where G4M-A and G4M-O instantiate GFM4MTSAD with pretrained AnomalyGFM and OwlEye backbones.

C2: Unreliable Supervision from Mechanism-Agnostic Pseudo-Anomalies. Without ground-truth labels, discriminative adaptation heavily relies on synthesized pseudoanomalies. Existing augmentation heuristics typically resort to random noise injection or global feature shuffling, which are detached from the spatio-temporal physical mechanisms of real-world systems [10], [19]. Consequently, these perturbations either yield ambiguous negatives that overlap with normal fluctuations and blur decision boundaries, or generate trivial easy negatives lying far outside the normal support domain. Neither provides valid discriminative directions to activate the GFM. Effective adaptation therefore demands mechanism-aligned hard negatives that explicitly disrupt local temporal continuity and cross-sensor synergy near the normality boundary while preserving empirical residuals.

C3: Node-to-Window Distortion in Multi-Granular Evidence Aggregation. Because real-world faults are often sparse and baseline scales vary markedly across sensors [8] and heterogeneous evidence channels [9], [20], naive readout severely distorts anomaly evidence. Specifically, mean pooling dilutes sparse fault signals into missed faults, whereas peak pooling magnifies benign local spikes into false alarms. Calibrating multi-scale node responses against normal operational references is therefore essential for reliable window-level detection and fine-grained sensor localization [11].

To address these challenges, we propose GFM4MTSAD, a structure-and-information synergistic alignment framework that unleashes pretrained GFMs for unsupervised MTSAD, as illustrated in Fig. 1 (left):

Structure-and-Information Contract Alignment (SICA): To tackle C1, we propose the SICA module to seamlessly bridge dynamic MTS and pretrained GFMs. Rather than forcing the pretrained backbone to digest raw, noisy time series directly, the core idea of SICA is to wrap the GFM with peripheral adapters—translating raw series into clean, GFM-compatible graph inputs beforehand and restoring lost temporal details afterward. Specifically, it establishes three synergistic contracts: (1) Feature Contract: A differentiable temporal frontend extracts dynamic sensor embeddings and maps them into the GFM’s native token space via continuous Token Bridge alignment. (2) Structure Contract: Stable, high-frequency topologies mined from normal data serve as structural priors, coupled with dynamic normal prior routing that gates message passing to prune spurious edges and curb noise propagation. (3) Information Contract: An Information-Preserving Residual Adapter (IPRA) is placed at the GFM output boundary for residual recovery, compensating for finegrained temporal dynamics via a gated pathway. Throughout the pipeline, only the peripheral adapters and detection head are tuned, fully retaining pretrained relational priors.

Mechanism-Aligned Pseudo-Anomaly Screening (MAPS): To tackle C2, we propose the MAPS strategy to furnish reliable anomaly supervision without ground-truth labels. Motivated by the intuition that real-world faults manifest as subtle breakdowns of physical regularities rather than random white noise, the core idea of MAPS is to deliberately disrupt local temporal habits or neighbor collaborations on a few target sensors, retaining only realistic, near-boundary hard negatives to teach the GFM where the true anomaly boundary lies. Concretely, we formalize four classes of spatio-temporal dependency breaks directly from normal series: intra-node primitives (lag-break and local-function-shift), which dismantle historical temporal dependencies and local evolution patterns; and inter-node primitives (neighbor-break and relation-flip), which disrupt local collaborative dependencies and invert relational orientations. While applying localized interventions, MAPS preserves empirical prediction residuals on surrounding contexts and screens synthesized candidates via joint constraints—enforcing boundary proximity, mechanism disruption efficacy, graph locality, and discriminative directionality. This design selectively retains near-boundary hard negatives without contaminating the global graph, supplying directed tasklevel supervision for the GFM.

Tail-Guided Evidence Calibration (TGEC): To tackle C3, we propose the TGEC mechanism to reliably translate finegrained node responses into trustworthy window-level alerts. The core idea of TGEC is to calibrate multi-granular anomaly clues against each sensor’s historical normal range before aggregation, ensuring that subtle single-sensor faults are not washed out by global averaging while routine background fluctuations do not trigger false alarms. Specifically, TGEC decomposes the GFM’s responses into four hierarchical evidence channels: direct evidence (node-level acute shifts), local evidence (neighborhood-aggregated patterns), relational evidence (edge-level imbalances), and context evidence (wholegraph global representations). By calibrating these channels against normal operational baselines, TGEC aligns heterogeneous responses onto a comparable scale. This calibrated design achieves both body preservation—stabilizing benign rankings to suppress false alarms induced by isolated local spikes—and tail re-ranking, which selectively refines extreme candidates in the tail distribution so localized anomalies are not diluted during global pooling. Ultimately, TGEC yields dependable window-level anomaly decisions alongside interpretable sensor-level fault attributions.

The contributions of this paper are as follows:

• To the best of our knowledge, we pioneer the adaptation of pretrained GFMs to unsupervised MTSAD, proposing a novel adaptation that bridges static graph foundation priors and continuous temporal observations.

• We propose SICA to harmonize feature, structure, and information contracts via Token Bridge alignment, normal prior routing, and IPRA residual recovery, suppressing topological noise while retaining fine-grained temporal details for reliable adaptation to pretrained GFMs.

• We introduce MAPS and TGEC to form a closed-loop supervision and decision pipeline, synthesizing mechanismaligned hard negatives via localized dependency breaks and calibrating multi-granular evidence for accurate window detection and sensor localization.

• Extensive evaluations across three public benchmarks with two representative GFM backbones demonstrate that GFM4MTSAD achieves superior performance over existing baselines (Fig. 1, right), substantiating the effectiveness of unleashing pretrained GFMs for MTSAD.

The remainder of this paper is organized as follows. Section II reviews related work, Section III details the proposed GFM4MTSAD framework, and Section IV presents the experimental results. Finally, Section V concludes the paper and outlines future directions.

## II. RELATED WORK

## A. Graph-based Time Series Anomaly Detection

Graph-based methods represent monitored variables as nodes and jointly model temporal dynamics and cross-variable dependencies by learning inter-variable connections. GDN [16] learns a dependency graph via sensor embeddings and utilizes graph attention to predict variable responses, identifying anomalies based on prediction deviations. Subsequent research has increasingly focused on the temporal evolution and stability of inter-variable relationships: GCAD [10] extracts sparse Granger causality graphs from the gradients of nonlinear predictors to detect anomalies through shifts in dependency patterns; CGSTA [17] constructs multi-scale dynamic graphs and stabilizes relation representations through normal reference alignment and cross-scale contrastive learning.

Structural information is also utilized for supervision construction and anomaly decision-making. CAROTS [19] generates positive and negative views through augmentations that preserve or perturb causal relationships, thereby providing structure-aware supervision for contrastive learning. PGRF-Net [20] combines dynamic relationship learning with normal prototype modeling, utilizing learnable gating to fuse prediction deviations, structural changes, prototype deviations, and local spike evidence. These works have promoted the development of relationship modeling, structured augmentations, and multi-evidence detection. GFM4MTSAD bridges these time series task requirements with pretrained graph anomaly representations, jointly designing temporal adaptation, mechanismguided supervision, and normal reference scoring under the condition of a frozen backbone.

## B. Graph Foundation Models

GFMs aim to learn comprehensively reusable structural and attribute representations across graphs to support diverse graph domains and downstream tasks. GFT [23] regards computation trees formed by message passing as transferable units, connecting cross-domain and cross-task graph representation learning via a shared tree vocabulary. For graph anomaly detection, AnomalyGFM [21] explicitly aligns node-to-neighborhood representation residuals with normal and abnormal prototypes, supporting zero-shot cross-graph inference and prompt adaptation based on a few normal nodes. OwlEye [22] combines cross-domain feature alignment, a multi-domain normal pattern dictionary, and truncated attention reconstruction to learn transferable patterns of normal behavior, enabling effective anomaly detection on unseen graphs.

These studies provide a foundation for reusing pretrained graph anomaly knowledge. Facing MTS, GFM4MTSAD focuses its adaptation on the interface between temporal windows and pretrained GFMs: it guides node inputs and variable topology alignment using normal data, replenishes information not preserved by projection via residual pathways, andfurther organizes node anomaly responses into window-level detection scores and auxiliary localization evidence.

![](images/506b94e1447e2c71aad49ebad8e510fd7332a9b097c34e4133475d09d7148f42.jpg)  
Fig. 2. Overall architecture of GFM4MTSAD. At a high level, G 4M adapts a frozen GFM to unsupervised MTSAD, outputting both window-level anomaly scores (s ) and sensor-level localization evidence (a ). Specifically, (1) MAPS (left, training only) synthesizes mechanism-aligned hard pseudo anomalies $( X _ { t } ^ { - } )$ via residual-preserving interventions and multi-criteria screening to provide paired and contrastive supervision; (2) SICA (center) bridges the temporal-topological gap through prior-guided structure alignment, a Token Bridge, and orthogonal IPRA residual compensation while keeping the GFM backbone frozen; and (3) TGEC (right) integrates multi-granularity anomaly evidence via normal-calibrated tail re-ranking, strictly preserving central norma rankings. Flame and snowflake icons denote trainable modules and the frozen backbone, respectively.

## III. METHOD

## A. Problem Definition

Let $\textbf { X } = \{ \mathbf { x } _ { t } \} _ { t = 1 } ^ { T }$ denote a MTS, where $\mathbf { x } _ { t } ~ \in ~ \mathbb { R } ^ { N }$ represents the observations collected from N sensors at time step t. A sliding window of length L is denoted as $\mathbf { X } _ { t } = [ \dot { \mathbf { x } } _ { t - L + 1 } , \dots , \mathbf { x } _ { t } ] ^ { \top } \in \mathbb { R } ^ { L \times N }$ , whose i-th column, $\mathbf { X } _ { t , : , i } ,$ corresponds to the observed trajectory of the i-th variable. The training set $\mathcal { D } _ { \mathrm { N } }$ consists exclusively of normal windows. Given a pretrained GFM with frozen parameters $\theta _ { 0 } .$ , GFM4MTSAD learns temporal adaptation and detection parameters to identify anomalous deviations in both temporal dynamics and crossvariable dependencies, outputting a window-level anomaly score $s _ { t }$ along with auxiliary localization evidence $\mathbf { a } _ { t } \in \mathbb { R } ^ { \bar { N } }$ indexed by the sensors.

## B. GFM4MTSAD Overview

The overall architecture of GFM4MTSAD is illustrated in Fig. 2. At a high level, GFM4MTSAD repurposes pretrained GFMs (e.g., AnomalyGFM [21], OwlEye [22]) for MTSAD through three collaborative modules:

a) SICA: SICA serves as the core adaptation module that bridges temporal sensory streams with the pretrained GFM via three coordinated contracts: the Feature Contract extracts decomposed temporal dynamics and anchors the Token Bridge projection within the dominant normal subspace; the Structure Contract regularizes instantaneous correlations with global normal connectivity priors; and the Information Contract recovers projection-excluded temporal features via an orthogonal residual bypass. This design aligns temporal-topological representations while preserving transferable relational priors across the pretrained backbone (Section III-C).

b) MAPS: MAPS is a mechanism-guided synthesis module that supplies informative hard normal-pseudo-anomaly pairs during unsupervised training. By executing residualpreserving local interventions on a sparse Graph Vector Autoregressive (Graph-VAR) reference and filtering candidates via dual quantile constraints, MAPS calibrates training difficulty near the normal boundary and translates edited coordinates into node-level paired supervision (Section III-D).

c) TGEC: TGEC is a reference-guided post-processing calibrator that consolidates multi-granularity evidence for final scoring. By enforcing central identity across the normal body and restricting multi-evidence re-ranking strictly to upper-tail candidates, TGEC effectively sharpens tail anomaly detection while outputting standardized node responses for auxiliary sensor localization (Section III-E).

## C. Structure-Information Contract Alignment (SICA)

To address C1, SICA bridges continuous MTS and the strictly frozen GFM without eroding pretrained topological priors [24]. Directly feeding raw temporal windows into a frozen graph backbone faces three adaptation obstacles: standard linear projections struggle to disentangle composite trendseasonal dynamics [25] or exploit target-domain variance priors [26]; purely instantaneous window correlations contain spurious edges that trigger noise amplification and oversmoothing across frozen GNN layers [27]; and dimensionality compression inevitably discards fine-grained temporal variations residing outside the projected subspace. Rather than forcing the frozen backbone to digest noisy temporal inputs, SICA wraps the GFM with peripheral adapters governed by three coordinated contracts (Fig. 2): (i) a Feature Contract that decomposes per-sensor temporal dynamics and anchors token projections within the dominant normal subspace; (ii) a Structure Contract that regularizes transient window correlations with persistent normal connectivity priors to ensure reliable message passing; and (iii) an Information Contract that recovers projection-excluded temporal details at the GFM output boundary via an orthogonal residual bypass.

a) Feature Contract: To establish a data-informed token interface, we first isolate per-sensor trend-seasonal dynamics via a dedicated decomposition frontend [25] and map them into the GFM token space: $\mathbf { F } _ { t } ~ = ~ f _ { \phi } ( \mathbf { X } _ { t } )$ and $\mathbf { Z } _ { t } ~ =$ $\mathrm { N o r m } _ { N } ( \mathbf { F } _ { t } \mathbf { W } )$ . Here, $f _ { \phi }$ is the learnable temporal frontend, and Norm<sub>N</sub> normalizes each extracted feature across sensors. The trainable projection W is empirically initialized from the leading principal components of normal features, matching the GFM’s native input dimension.

b) Structure Contract: Building on the temporal features $\mathbf { F } _ { t }$ , we construct a noise-resilient relational topology for message passing by regularizing instantaneous Pearson correlations $\mathbf { C } _ { t }$ with global normal connectivity priors Π:

$$
\mathbf { A } _ { t } = { \mathcal { T } } _ { k _ { g } } \left( \mathbf { C } _ { t } \odot [ ( 1 - \lambda _ { A } ) \mathbf { 1 } + \lambda _ { A } \mathbf { I I } ] \right) .\tag{1}
$$

Here, $\mathbf { C } _ { t } = | \operatorname { C o r r } ( \mathbf { F } _ { t } ) |$ , and $\Pi _ { i j }$ is the frequency with which edge $( i , j )$ appears in normal training graphs. The weight $\lambda _ { A }$ controls the prior contribution; $\mathcal { T } _ { k _ { g } }$ retains the strongest $k _ { g }$ neighbors per node, symmetrizes the graph by maximum, and adds self-loops. Graph construction is detached from gradient updates.The GFM produces $( { \bf H } _ { t } , { \bf q } _ { t } ) = { \cal B } ( { \bf Z } _ { t } , { \bf A } _ { t } )$ , comprising node representations and native anomaly responses. Its backbone weights $\theta _ { 0 }$ remain frozen, while native adaptation parameters $\psi$ may be trained. For hierarchical backbones, an optional reliability gate uses normal edge stability and the persistence of selected edges to mix the window graph with the identity matrix in the first layer and with the normal graph in subsequent layers.

c) Information Contract: The Token Bridge may discard anomaly-relevant variations outside its projection subspace. IPRA recovers these components through a gated low-rank residual:

$$
\begin{array} { r l } & { \mathbf { F } _ { t } ^ { \perp } = \mathbf { F } _ { t } - \mathbf { F } _ { t } \mathrm { s g } ( \mathbf { W } \mathbf { W } ^ { + } ) , } \\ & { \widetilde { \mathbf { H } } _ { t } = \mathbf { H } _ { t } + \gamma \sigma ( b ) \mathrm { G E L U } \left( \mathrm { R M S } ( \mathbf { F } _ { t } ^ { \perp } ) \mathbf { U } \right) \mathbf { V } . } \end{array}\tag{2}
$$

Here, $\mathbf { W } ^ { + }$ is the pseudoinverse, sg stops gradients, and RMS normalization acts per sensor. The rank-r bottleneck U, V is controlled by gate $\sigma ( b )$ and fixed scale $\gamma .$ . We initialize V to zero and mean-pool $\widetilde { \mathbf { H } } _ { t }$ to obtain $\widetilde { \bf g } _ { t }$

## D. Mechanism-Aligned Pseudo-Anomaly Screening (MAPS)

To address C2, MAPS synthesizes mechanism-aligned hard negatives with exact sensor-time coordinates to guide unsupervised learning. Unlike generative paradigms prone to mode collapse [28], costly iterative sampling [29], or auxiliary maskalignment overhead [30], and unlike heuristic augmentations (e.g., Gaussian jittering or masking) [19] that introduce highfrequency artifacts and trivial numerical shortcuts, MAPS conceptualizes multivariate anomalies as localized breakdowns in spatio-temporal couplings. Specifically, MAPS operates through a two-stage intervention-and-screening pipeline directly over normal series: first, it fits a sparse Graph-VAR reference to characterize normal cross-sensor dependencies and executes residual-preserving local interventions across four structural primitives, disrupting temporal or relational regularities while retaining empirical background fluctuations; second, to prevent synthesized candidates from degenerating into either trivial far-off outliers or imperceptible normal shifts, it enforces quantile-constrained hard screening to select minimal-edit candidates situated near the upper normal energy boundary. This provides informative window-level negatives alongside exact perturbed sensor sets for fine-grained localization supervision.

a) Residual-Preserving Local Interventions: To faithfully mimic structural faults on normal dynamics, we first characterize cross-sensor dependencies via a sparse Graph-VAR model [10], [31]: $f _ { i } ( \mathbf { u } _ { \tau , i } ) ~ = ~ c _ { i } + \mathbf { w } _ { i } ^ { \top } \mathbf { u } _ { \tau , i }$ , where $\mathbf { u } _ { \tau , i }$ collects lagged observations of sensor i and its selected neighbors. The coefficients are fitted on normal data, and $\bar { x } _ { \tau , i }$ denotes a normally standardized observation.

Within an intervention interval $\mathcal { T } _ { t } .$ , we replace the predicted component while retaining the original residual:

$$
\bar { x } _ { \tau , i } ^ { \prime } = \bar { x } _ { \tau , i } + f _ { i } ^ { - } ( \mathbf { u } _ { \tau , i } ^ { \prime } ) - f _ { i } ( \mathbf { u } _ { \tau , i } ) , \qquad \tau \in \mathcal { T } _ { t } .\tag{3}
$$

Here, $\mathbf { u } _ { \tau , i } ^ { \prime }$ contains the synthesized history. Let $d _ { \tau , i j }$ denote neighbor $\mathbf { \nabla } _ { j ^ { \prime } \mathbf { s } }$ lagged contribution to $f _ { i }$ . We select $j ^ { \star }$ by the largest sum of lag-wise mean absolute contributions over $\mathcal { T } _ { t } .$ For strength $\alpha \in [ 0 , 1 ]$ , the intervention is

$$
f _ { i } ^ { - } ( \mathbf { u } _ { \tau , i } ) = \left\{ \begin{array} { l l } { ( 1 - \alpha ) f _ { i } ( \mathbf { u } _ { \tau , i } ) + \alpha f _ { i } ( \mathbf { u } _ { \tau - d , i } ) , } & { \mathrm { l a g } , } \\ { f _ { i } ( \mathbf { u } _ { \tau , i } ) - \alpha d _ { \tau , i j ^ { \star } } , } & { \mathrm { n e i g h b o r } , } \\ { f _ { i } ( \mathbf { u } _ { \tau , i } ) - 2 \alpha d _ { \tau , i j ^ { \star } } , } & { \mathrm { r e l a t i o n } , } \\ { f _ { i } ( \mathbf { u } _ { \tau , i } ) + \alpha \nu _ { i } , } & { \mathrm { l o c a l } . } \end{array} \right.\tag{4}
$$

The offset d is a positive integer. We set $\nu _ { i } ~ \mathrm { t o } ~ - 1$ when the mean normal prediction over $\mathcal { T } _ { t }$ is nonnegative, and to 1 otherwise. Relation flip reverses the selected coupling when α exceeds 0.5. The intervened predictor and neighbor contributions are evaluated recursively on the synthesized history.

b) Quantile-Constrained Hard Screening: We seek detectable mechanism changes under bounded edits. Let $\mathbf { r } _ { i } ( \mathbf { X } )$ contain the Graph-VAR residuals standardized by normal residual statistics. For n valid scoring coordinates, define $E _ { i } ( \mathbf { X } ) = \| \mathbf { r } _ { i } ( \mathbf { X } ) \| _ { 2 } / \sqrt { n }$ . Relative to the fixed normal window $\mathbf { X } _ { t } ,$ , the paired energy of a candidate is

$$
[ E _ { i } ^ { - } ( \mathbf { X } ^ { \prime } ) ] ^ { 2 } = E _ { i } ( \mathbf { X } _ { t } ) ^ { 2 } + \frac { \| \mathbf { r } _ { i } ( \mathbf { X } ^ { \prime } ) - \mathbf { r } _ { i } ( \mathbf { X } _ { t } ) \| _ { 2 } ^ { 2 } } { n } .\tag{5}
$$

Let $\mathcal { U } _ { t }$ contain candidates satisfying finite-value, temporalcontext, normal-support, and editing-locality checks. The normalized edit magnitude $\rho _ { i } ( \mathbf { X } ^ { \prime } )$ is the root-mean-square change over $\mathcal { T } _ { t } .$ , scaled by the sensor’s normal scale or valid-state span. Let $q _ { i }$ and $q _ { i } ^ { \mathrm { u } }$ denote the target and upper normal-energy quantiles, and $\sigma _ { i } ^ { E }$ the normal-energy scale, estimated using the same scoring length as the candidate. We select

$$
\begin{array} { r l } & { \mathbf { X } _ { t } ^ { - } = \arg \underset { \mathbf { X } ^ { \prime } \in \mathcal { U } _ { t } } { \operatorname* { m i n } } \left[ \frac { \left| E _ { i } ^ { - } ( \mathbf { X } ^ { \prime } ) - q _ { i } \right| } { \sigma _ { i } ^ { E } } + \lambda _ { \rho } \rho _ { i } ( \mathbf { X } ^ { \prime } ) \right] , } \\ & { \mathrm { s . t . } \quad \rho _ { \operatorname* { m i n } } \leq \rho _ { i } ( \mathbf { X } ^ { \prime } ) \leq \rho _ { \operatorname* { m a x } } , } \\ & { \quad E _ { i } ( \mathbf { X } _ { t } ) + \eta \sigma _ { i } ^ { E } \leq E _ { i } ^ { - } ( \mathbf { X } ^ { \prime } ) \leq q _ { i } ^ { \mathrm { u } } . } \end{array}\tag{6}
$$

The constraints require a mechanism change while bounding its energy and edit magnitude. The selected view and its directly edited sensor set $\mathcal { D } _ { t }$ supervise training. If no candidate is feasible, the window contributes only to positive-view consistency.

## E. Tail-Gated Evidence Calibration (TGEC)

To address C3, TGEC eliminates decision distortion when translating fine-grained node responses into windowlevel anomaly scores and sensor attributions. In real-world faults, anomaly clues manifest across heterogeneous granularities—spanning isolated node spikes, neighborhood shifts, relational imbalances, and global context—whose raw scales and distributions differ substantially. However, naively pooling these channels or applying unconstrained global fusion [20] easily dilutes sparse sensor faults or overfits benign background fluctuations across normal windows. Drawing inspiration from Extreme Value Theory (EVT) [32], [33], TGEC resolves this dilemma through a two-step calibration and selective fusion paradigm: first, it extracts four hierarchical anomaly descriptors and calibrates each onto a unified normal-reference tail scale to form a comparable multi-evidence consensus; second, it enforces central identity with tail gating, which strictly preserves baseline rankings across the normal score body to prevent false alarms while activating multi-evidence re-ranking exclusively on upper-tail candidates, accompanied by standardized direct responses for sensor-level localization.

a) Multi-Granularity Evidence Aggregation: Direct and local node heads add bounded, tanh-scaled MLP corrections to stabilized log-native responses, conditioned on the adapted node features and log-responses. Let $e _ { t , i }$ denote the resulting direct response.

We extract four complementary window descriptors. Direct evidence averages the largest $k _ { n }$ direct responses; local evidence averages local-head responses across sensors; relation evidence measures the Euclidean norm of mean-centered local responses; and context evidence combines the pooled graph representation with response statistics through an MLP.

Each descriptor is calibrated against its normal reference. Sigmoid-smoothed comparisons estimate its lower and upper tail masses, using a unit pseudocount and a denominator equal to the reference size plus one. We double the smaller tail mass, cap it at one, and take its negative logarithm. The consensus $v _ { t }$ averages the largest $k _ { e }$ calibrated values. The consensus is $v _ { t } = \mathrm { T o p M e a n } _ { k _ { e } } ( \{ \widetilde { e } _ { t } ^ { a } \} _ { a \in \{ \mathrm { d } , \mathrm { l } , \mathrm { r } , \mathrm { c } \} } )$

b) Tail-Guided Re-Ranking: Per-sensor standardization of direct responses using normal means and stabilized standard deviations yields the auxiliary evidence $\mathbf { a } _ { t } .$ . Temperature-scaled log-mean-exp pooling of these responses produces the baseline window score $S _ { t }$ , with temperature $T _ { p }$ . We standardize $S _ { t }$ using the median and scaled median absolute deviation of preceding scores only when fewer than half the nodes have valid normal Graph-VAR fits. Within each sequence, $b _ { t }$ is the larger of the resulting score and its trailing mean over up to ten windows, including the current one. Given normal references $\{ b _ { r } ^ { \mathrm { N } } \} _ { r = 1 } ^ { m _ { b } }$ , we set $\tau _ { 0 } = - \log \alpha _ { 0 }$ for tail mass $\alpha _ { 0 }$ and compute

$$
\begin{array} { r l } & { u _ { t } = - \log \frac { 1 + \sum _ { r = 1 } ^ { m _ { b } } \mathbb { I } [ b _ { r } ^ { \mathrm { N } } \geq b _ { t } ] } { m _ { b } + 1 } , } \\ & { s _ { t } = \left\{ u _ { t } , \qquad \begin{array} { l l } { u _ { t } , } & { u _ { t } < \tau _ { 0 } , } \\ { \operatorname* { m a x } \{ \tau _ { 0 } , \omega u _ { t } + ( 1 - \omega ) v _ { t } \} , } & { u _ { t } \geq \tau _ { 0 } . } \end{array} \right. } \end{array}\tag{7}
$$

The weight ω balances baseline and multi-evidence responses; only upper-tail candidates are re-ranked, while the remaining $u _ { t }$ coordinates are unchanged.

## F. Training Objectives

a) Trainable Parameters: Each normal window is paired with a positive view generated by subtle affine transformations and a MAPS pseudo-anomalous view. The trainable parameters Θ comprise the temporal frontend, Token Bridge, IPRA, direct evidence head, and native GFM adaptation parameters ψ. The GFM backbone, its running statistics, and the local and contextual heads remain frozen.

Below, $e _ { t , i }$ denotes direct node evidence. We use E for averaging over normal windows and $\mathbb { E } _ { w }$ for averaging with detached candidate weights. Each negative-pair loss uses windows with valid candidates and the nonempty sensor sets required by that loss; empty subsets contribute zero.

b) Spatially Constrained Node Localization: Let $\mathcal { D } _ { t }$ contain directly edited sensors. Their paired response increments are $\Delta _ { t , i } = e _ { t , i } ^ { - } - \mathrm { s g } ( e _ { t , i } )$ . We rank these increments above those of other sensors:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { n o d e } } = \mathbb { E } _ { w } \left[ \operatorname* { m e a n } _ { i \in \mathcal { D } _ { t } , j \notin \mathcal { D } _ { t } } [ m _ { n } + \Delta _ { t , j } - \Delta _ { t , i } ] _ { + } \right] , } \end{array}\tag{8}
$$

where $m _ { n } > 0$ is the margin and $[ z ] _ { + } = \operatorname* { m a x } ( z , 0 )$ . Sensors outside $\mathcal { D } _ { t }$ may include propagated effects.

c) Unified Optimization Objective: The total objective combines window discrimination, sensor localization, and response consistency:

$$
\begin{array} { r l } & { \mathcal { L } = \mathcal { L } _ { \mathrm { c t r } } + \lambda _ { s } ( \mathcal { L } _ { \mathrm { r a n k } } + \lambda _ { c } \mathcal { L } _ { \mathrm { c l s } } ) } \\ & { ~ + \lambda _ { n } \mathcal { L } _ { \mathrm { n o d e } } + \lambda _ { r } ( \mathcal { L } _ { + } + \mathcal { L } _ { \mathrm { k e e p } } ) . } \end{array}\tag{9}
$$

The nonnegative weights control the contributions of the corresponding objectives.

The contrastive loss ${ \mathcal L } _ { \mathrm { c t r } }$ uses a two-logit softmax over cosine similarities scaled by temperature $T _ { c } .$ , favoring the normal–positive pair over its paired normal–pseudo-anomalous pair. The window hinge loss $\mathcal { L } _ { \mathrm { r a n k } }$ penalizes violations of $S _ { t } ^ { - } \geq S _ { t } + m _ { w } .$ . The classification loss $\mathcal { L } _ { \mathrm { c l s } }$ averages binary cross-entropy over the three baseline-score logits with labels (0, 0, 1). All training scores precede temporal buffering and TGEC re-ranking.

Both consistency terms use Smooth L1 with unit transition threshold and detached normal responses as targets. The positive term $\mathcal { L } _ { + }$ averages over all sensors and normal windows. The negative term $\mathcal { L } _ { \mathrm { k e e p } }$ first averages over the preserved sensors $\textstyle { \mathcal { K } } _ { t }$ within each window, then uses the candidateweighted window average. The set $\textstyle { \mathcal { K } } _ { t }$ excludes direct edits, propagated effects, and mechanism-based exclusions.

TABLE I  
STATISTICS OF THE EVALUATION DATASETS.
<table><tr><td>Entities</td><td>Vars.</td><td>Train</td><td>Test</td><td>Windows</td><td>Anom.</td></tr><tr><td>MSL P-15</td><td>55</td><td>3,682</td><td>2,856</td><td>2,797</td><td>80</td></tr><tr><td>SMAP T-3</td><td>25</td><td>2,876</td><td>8,579</td><td>8,520</td><td>302</td></tr><tr><td>SMD 1-1</td><td>38</td><td>28,479</td><td>28,479</td><td>28,420</td><td>3,166</td></tr></table>

## IV. EXPERIMENTS

We conduct extensive experiments on three real-world MT-SAD benchmarks across two heterogeneous GFM backbones to evaluate the effectiveness of GFM4MTSAD. Specifically, we investigate:

• RQ1: How does GFM4MTSAD achieve superior performance across multiple datasets compared with other MTSAD methods?

• RQ2: How effective are the individual components of GFM4MTSAD?

• RQ3: How does the proposed tail-gated calibration mechanism operate to suppress false alarms while preserving normal rankings?

• RQ4: How do key parameters affect GFM4MTSAD’s performance?

## A. Experimental Settings

Datasets. We evaluate GFM4MTSAD on three representative MTSAD benchmarks spanning spacecraft telemetry and industrial server monitoring: Mars Science Laboratory (MSL P-15) and Soil Moisture Active Passive (SMAP T-3) [34], and Server Machine Dataset (SMD 1-1) [13]. Table I summarizes the statistics of these evaluation series. Following standard evaluation protocols, we retain the original training and testing partitions and segment sequences into sliding windows of length L = 60 with a stride of 1. A test window is labeled anomalous if it contains at least one anomalous timestamp. Within the anomaly-free training partition, the first 80% of observations are used for model fitting and reference construction, while the remaining 20% serve as a held-out validation block; sliding windows are constructed independently within each block to prevent boundary leakage. Feature standardization uses statistics fitted exclusively on the training block.

Baselines. We compare GFM4MTSAD against nine representative baseline methods covering three technical categories: (1) generic contrastive and out-of-distribution (OOD) detection methods: SimCLR [35], CSI [36], and SSD [37]; (2) temporal attention and discrepancy-based MTSAD methods: Anomaly-Transformer [38], DCdetector [39], and CTAD [40]; and (3) structure-aware and advanced contrastive MTSAD methods: CARLA [41], CAROTS [19], and PaAno [42]. To demonstrate the generality of our adaptation framework across distinct foundation architectures, we instantiate GFM4MTSAD on two pretrained GFMs—AnomalyGFM [21] and OwlEye [22]—denoted as GFM4MTSAD-A and GFM4MTSAD-O, respectively.

All methods share identical data partitions, window labels, and evaluation protocols, where timestamp-level baseline outputs are aggregated into window-level scores via the withinwindow maximum.

Metrics. Following recent mainstream studies [19], [41], [42], three standard metrics without point adjustment are selected for evaluation: AUROC, AUPRC, and F1 Score. In practical anomaly-sparse scenarios such as MSL P-15 spacecraft telemetry monitoring, low AUROC and AUPRC values indicate that true subsystem faults are buried beneath massive false alarms, rendering automated alerting impractical. Higher values across these metrics signify stronger fault discrimination and lower false-alarm overhead.

Implementation Details. Training Protocol: We use a two-stage warm-start protocol. The temporal frontend is first trained for 30 epochs on normal windows and synthetic views. Its weights initialize the joint model, whose task-specific adapters follow the initialization rules in Section III-C; the GFM and its native adaptation parameters load their pretrained weights. We then optimize Θ jointly for 30 epochs with a batch size of 256, keeping the GFM backbone frozen.The MAPS reference mechanism remains fixed during joint training. Normal scoring references are refreshed after each epoch; when the graph prior is enabled, it is refitted using the updated frontend before recalculating scoring statistics. Reference fitting uses normal training windows in evaluation mode without gradient tracking. Checkpoints are evaluated every 5 epochs and selected by the lowest joint validation loss on held-out normal windows and synthetic views. After restoring the selected checkpoint, inference references are refitted on normal training data and frozen. The main comparison uses three runs per entity–backbone configuration. The framework is implemented in PyTorch and evaluated on NVIDIA GeForce RTX 4090 D GPUs.

Parameters: The DLinear frontend extracts 128-dimensional temporal representations (d ) per sensor using a movingaverage kernel of size 25, and the Token Bridge projects them to the native token dimension $d _ { z }$ required by each GFM. IPRA employs a low-rank bottleneck with rank r set to 16, residual scaling factor $\gamma$ fixed at 1.0, and zero-initialized up-projection weights V. Graph construction sets the neighborhood size $k _ { g }$ to 5 per node. The prior weight $\lambda _ { A }$ is set to 0.25 for all AnomalyGFM and OwlEye settings. Optional directional screening retains a candidate only if, under a fixed reference scorer, the target sensor’s response increases and its increase exceeds that of every other sensor. The main comparison uses $b _ { t }$ for AnomalyGFM and the re-ranked score $s _ { t }$ for OwlEye. Optimization uses the Adam optimizer with a weight decay of $1 0 ^ { - 3 }$ , a base learning rate of $1 0 ^ { - 3 }$ for temporal and adap tation modules, and $1 0 ^ { - 4 }$ for lightweight GFM parameters ψ, scheduled via a 5-epoch linear warm-up followed by cosine annealing and gradient-norm clipping at 0.5. The contrastive temperature $T _ { c }$ and pooling temperature $T _ { p }$ are set to 0.2 and 0.5, respectively. The loss weights $\lambda _ { \mathrm { n } } , \lambda _ { \mathrm { r } } , \lambda _ { \mathrm { s } } ,$ and $\lambda _ { \mathrm { c } }$ are set to 1, 0.2, 1, and 0.1, respectively. TGEC fixes the default tail probability $\alpha _ { 0 }$ at 0.05, the fusion weight ω at 0.5, and the calibrated evidence aggregation size $k _ { e }$ at 2. Our code is available at https://github.com/yanzhuang-h/GFM4MTSAD.

TABLE II  
PERFORMANCE COMPARISON OF MTSAD ON THREE BENCHMARK DATASETS. THE BEST AND SECOND-BEST RESULTS ARE HIGHLIGHTED IN BOLD AND UNDERLINED TEXT. AT DENOTES ANOMALYTRANSFORMER, G4M-A AND G4M-O DENOTE GFM4MTSAD BASED ON ANOMALYGFM AND OWLEYE, RESPECTIVELY.
<table><tr><td rowspan="2">Datasets</td><td rowspan="2">Metrics</td><td rowspan="2">SimCLR (ICML20)</td><td rowspan="2">CSI (NIPS20)</td><td rowspan="2">SSD (ICLR21)</td><td rowspan="2">AT (ICLR22)</td><td rowspan="2">DCdetector (KDD23)</td><td rowspan="2">CTAD (TKDE24)</td><td rowspan="2">CARLA (PR25)</td><td rowspan="2">CAROTS (ICML25)</td><td rowspan="2">PaAno (ICLR26)</td><td rowspan="2">G4M-A (Ours)</td><td rowspan="2">G4M-0 (Ours)</td></tr><tr><td></td></tr><tr><td rowspan="4">MSL P-15</td><td>AUROC AUPRC</td><td>0.5246</td><td>0.4659</td><td>0.4621</td><td>0.3072</td><td>0.3909</td><td>0.5420</td><td>0.6956</td><td>0.6592</td><td>0.5104</td><td>0.8747</td><td>0.9246</td></tr><tr><td>F1</td><td>0.0280</td><td>0.0252</td><td>0.0255</td><td>0.0186</td><td>0.0215</td><td>0.0293</td><td>0.4208</td><td>0.0378</td><td>0.0271</td><td>0.1731</td><td>0.6916</td></tr><tr><td></td><td>0.0947</td><td>0.0805</td><td>0.0712</td><td>0.0569</td><td>0.0562</td><td>0.1029</td><td>0.5405</td><td>0.0950</td><td>0.0856</td><td>0.3318</td><td>0.8138</td></tr><tr><td>AUROC</td><td>0.4956</td><td>0.5997</td><td>0.4543</td><td>0.5230</td><td>0.4717</td><td>0.5439</td><td>0.6654</td><td>0.4931</td><td>0.7923</td><td>0.7233</td><td>0.7209</td></tr><tr><td rowspan="4">SMAP T-3</td><td>AUPRC</td><td>0.0650</td><td>0.0456</td><td>0.0691</td><td>0.2267</td><td>0.0382</td><td>0.0377</td><td>0.2841</td><td>0.0440</td><td>0.3908</td><td>0.4120</td><td>0.4169</td></tr><tr><td>F1</td><td>0.0814</td><td>0.0861</td><td>0.1189</td><td>0.3315</td><td>0.0799</td><td>0.0796</td><td>0.3897</td><td>0.0752</td><td>0.5505</td><td>0.4922</td><td>0.4739</td></tr><tr><td>AUROC</td><td>0.5400</td><td>0.4114</td><td>0.6010</td><td>0.5423</td><td>0.5078</td><td>0.5451</td><td>0.3901</td><td>0.8740</td><td>0.5901</td><td>0.9089</td><td>0.8087</td></tr><tr><td>AUPRC</td><td>0.1127</td><td>0.0932</td><td>0.2559</td><td>0.2632</td><td>0.1155</td><td>0.1200</td><td>0.2430</td><td>0.5225</td><td>0.1213</td><td>0.6050</td><td>0.6167</td></tr><tr><td>SMD 1-1</td><td>F1</td><td>0.2211</td><td>0.2006</td><td>0.3095</td><td>0.3530</td><td>0.2008</td><td>0.2126</td><td>0.2676</td><td>0.5625</td><td>0.2976</td><td>0.5680</td><td>0.6198</td></tr></table>

## B. Performance Comparison (RQ1)

We present a comprehensive comparison between GFM4MTSAD and representative baselines in Table II, where GFM4MTSAD secures the best or second-best performance across nearly all evaluation criteria. Both GFM4MTSAD-A and GFM4MTSAD-O substantially outperform competing methods, confirming that aligning temporal feature and relational contracts (C1) effectively unleashes pretrained GFMs for MTSAD. Specifically, GFM4MTSAD-O excels on high-dimensional MSL via OwlEye’s pattern dictionary and truncated attention reconstruction, which amplify acute relational deviations, whereas GFM4MTSAD-A achieves stronger stability on SMAP and SMD through AnomalyGFM’s dual normal–abnormal prototype alignment. Moreover, the two backbones differ markedly in native input dimensionality, as AnomalyGFM projects temporal representations into a substantially more compact node token space than OwlEye. This stronger token compression governs the granularity of temporal variations available to internal graph computations, while IPRA supplies complementary temporal information at the output stage. Coupled with differences in the backbones’ native scoring mechanisms, these factors shape the ranking of anomalous windows and yield relative advantages across different datasets and evaluation metrics.

Compared with baselines across contrasting physical scenarios—namely aerospace telemetry (MSL and SMAP), characterized by sparse faults and abrupt command-driven transitions, and industrial server monitoring (SMD), featuring dense multi-metric co-fluctuations—GFM4MTSAD demonstrates superior cross-scenario adaptability. Existing methods often favor a single regime; for instance, patch-based PaAno captures local temporal patterns in lower-dimensional SMAP but degrades under high-dimensional channel interactions in MSL and SMD. Conversely, GFM4MTSAD generalizes robustly across both sparse telemetry and complex server workloads by jointly modeling channel-wise dynamics and cross-sensor couplings.

Compared with non-graph baselines—including instancelevel contrastive methods (SimCLR, CSI, and SSD) and temporal discrepancy models (AnomalyTransformer, DCdetector, and CTAD)—as well as raw graph-based methods (CAROTS), GFM4MTSAD establishes a more reliable topological modeling paradigm. Non-graph methods neglect intersensor topology and consequently suffer from high false alarms under benign non-stationary fluctuations, whereas relying on raw window-level graphs leaves CAROTS vulnerable to transient correlational noise. GFM4MTSAD overcomes both limitations via SICA, anchoring message passing on persistent normal connectivity priors while leveraging IPRA to recover projection-excluded temporal details from $\mathbf { F } _ { t } ^ { \perp }$

Compared with advanced contrastive baselines (CARLA, CAROTS, and PaAno) that rely on heuristic augmentations or unconstrained negative sampling and often generate trivial outliers far from the true decision boundary, GFM4MTSAD couples MAPS with TGEC to form a closed-loop supervision and calibration pipeline. Specifically, MAPS synthesizes residual-preserving interventions near the normal Graph-VAR boundary to provide informative training supervision, while TGEC selectively integrates multi-granularity evidence in the upper tail to sharpen fault rankings without distorting normal scores.

## C. Ablation Study (RQ2)

We evaluate the contributions of SICA, MAPS, and TGEC in Tables III and IV. Training-stage ablations share the same warm-start frontends, matched downstream initializations, data ordering, and supervision budgets over 30 epochs. Table III reports native window AUPRC and F1 to isolate representation learning effects, whereas Table IV compares TGEC scoring variants on identical frozen Full checkpoints.

Effectiveness of SICA. Our method benefits from both the complete alignment architecture and its individual contractual components. Replacing SICA with a minimal dimensioncompatible interface consistently reduces both AUPRC and F1 across all evaluated entities, causing a particularly pronounced degradation on SMD. Similarly, removing either PCA-based Token Bridge initialization or the IPRA residual bypass degrades both metrics across all entities, confirming the necessity of anchoring initial token projections within the normal feature covariance subspace and recovering projection-excluded temporal details. Furthermore, incorporating the normal graph prior substantially strengthens detection performance on highdimensional MSL in both AUPRC and F1 by suppressing transient correlational noise. Together, these results demonstrate the complementary contributions of feature alignment, residual information preservation, and normal connectivity priors.

TABLE III  
SICA AND MAPS ABLATIONS WITH THE OWLEYE BACKBONE (NATIVE WINDOW AUPRC AND F1). BEST AND SECOND-BEST RESULTS ARE BOLD AND UNDERLINED, RESPECTIVELY.
<table><tr><td rowspan="2" colspan="2">Ablation Category and Variant</td><td colspan="2">MSL-0</td><td colspan="2">SMAP-O</td><td colspan="2">SMD-0</td></tr><tr><td>AUPRC</td><td>F1</td><td>AUPRC</td><td>F1</td><td>AUPRC</td><td>F1</td></tr><tr><td colspan="2"></td><td>0.8131</td><td>0.9182</td><td>0.4519</td><td>0.5283</td><td>0.6932</td><td>0.6849</td></tr><tr><td rowspan="4">SICA</td><td>w/ Minimal SICA + MAPS</td><td>0.6998</td><td>0.8439</td><td>0.3408</td><td>0.4215</td><td>0.2483</td><td>0.3232</td></tr><tr><td>w/o PCA Bridge Initialization</td><td>0.4203</td><td>0.6854</td><td>0.3928</td><td>0.4784</td><td>0.6642</td><td>0.6344</td></tr><tr><td>w/o Normal Graph Prior  $( \lambda _ { A } = 0 )$ </td><td>0.6131</td><td>0.7526</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o IPRA Residual Bypass</td><td>0.5723</td><td>0.8066</td><td>0.3410</td><td>0.4621</td><td>0.4756</td><td>0.5506</td></tr><tr><td>MAPS</td><td>w/ Random Direct-Edit Selection</td><td>0.7555</td><td>0.8742</td><td>0.4277</td><td>0.5424</td><td>0.6445</td><td>0.6175</td></tr></table>

Note: “—” denotes an inactive normal graph prior. All variants share the same warm-start frontend within each entity.

TABLE IV  
ABLATION STUDY OF TGEC SCORING VARIANTS ACROSS SIX CONFIGURATIONS (AUPRC), WHERE ACT. DENOTES THE TEST TAIL ACTIVATION RATE.
<table><tr><td>Config.</td><td>Native</td><td>ECDF</td><td>Global</td><td>Direct-only</td><td>TGEC</td><td>Act.</td></tr><tr><td>SMD-A</td><td>0.6275</td><td>0.6274</td><td>0.5210</td><td>0.5198</td><td>0.5217</td><td>24.07%</td></tr><tr><td>SMAP-A</td><td>0.4942</td><td>0.3835</td><td>0.4096</td><td>0.3483</td><td>0.4112</td><td>22.58%</td></tr><tr><td>MSL-A</td><td>0.4775</td><td>0.4795</td><td>0.7070</td><td>0.5709</td><td>0.5709</td><td>1.11%</td></tr><tr><td>SMD-O</td><td>0.6932</td><td>0.6894</td><td>0.6970</td><td>0.6947</td><td>0.7015</td><td>14.45%</td></tr><tr><td>SMAP-O</td><td>0.4519</td><td>0.4211</td><td>0.4424</td><td>0.3440</td><td>0.4504</td><td>9.28%</td></tr><tr><td>MSL-O</td><td>0.8131</td><td>0.8366</td><td>0.1230</td><td>0.8366</td><td>0.8366</td><td>0.00%</td></tr></table>

Effectiveness of MAPS. We isolate the contribution of candidate selection using w/ Random Direct-Edit Selection. This variant retains mechanism-based candidate generation, validity constraints, context propagation, and supervision budgets, while replacing boundary-proximity ranking with uniform sampling among valid direct-edit candidates. Across MSL, SMAP and SMD, boundary-ranked selection consistently outperforms uniform selection in AUPRC and yields further F1 improvements on MSL and SMD. These results confirm that prioritizing hard, informative interventions near the normal mechanism boundary provides stronger discriminative supervision than unranked sampling within the same candidate pool.

Effectiveness ofTGEC. Table IV compares five scoring variants on identical frozen Full checkpoints. TGEC restricts evidence fusion to the train-defined upper tail, maintaining exact central identity where $s _ { t }$ equals $u _ { t }$ below $\tau _ { 0 }$ and preserving the ECDF ordering within the normal score body. On SMD and SMAP, TGEC consistently outperforms both Global consensus and Direct-only tail scoring, confirming the synergistic value of tail restriction and multi-granularity evidence fusion. The necessity of protecting the normal body against unconstrained modification is particularly evident on MSL-O: when no test windows exceed the tail gate, TGEC faithfully retains the strong ECDF ranking performance, whereas global fusion

MSL P-15 SMAP T-3 SMD 1-1

![](images/6fa04574e7937da277ec6fac89a0dc027b13f23e89832a5b5fa28a9dec765c3f.jpg)  
(a) A

![](images/202f2dcd550ed78134e859c46e0995592c0016d8fb6b9074498fca03f1032be6.jpg)

![](images/84ba4e689432528b9d1630a91193875696f9a55045c27f7f5f655c973121dfe2.jpg)  
(c) A

(b) O  
![](images/a03091c375be3b9383829da636dc2c8434432ecd436cbc31e7560ac8e1c3055a.jpg)  
(d) O  
Fig. 3. TGEC scoring ablation: top, AUPRC; bottom, raw oracle F1. A/O denote GFM4MTSAD-A/O. Nat.: native scores; ECDF: normal-reference scoring; All: global consensus; TGEC: tail consensus.

causes severe performance degradation by distorting benign rankings. Together, these results validate selective tail reranking for sharpening anomaly detection while safeguarding the central score structure.

## D. Mechanism Analysis (RQ3)

To provide an intuitive understanding of how GFM4MTSAD refines anomaly decisions at inference time, we visualize the internal calibration mechanics of TGEC on SMD 1-1 (GFM4MTSAD-O) in Fig. 4. Specifically, we examine how the framework separates genuine faults from benign operational spikes across four sequential stages: tail gating, multigranularity evidence decoupling, score transformation, and rank calibration.

As shown in Fig. 4(a) and (c), the empirical normal and held-out validation distributions concentrate densely below the train-defined tail threshold $\tau _ { 0 } ,$ , where TGEC enforces an exact identity mapping along the diagonal so that routine normal fluctuations remain completely undisturbed. Once scores cross $\tau _ { 0 }$ into the active upper tail, relying solely on baseline surprisal becomes prone to false alarms caused by transient single-sensor spikes. However, examining the calibrated multi-granularity evidence in Fig. 4(b) reveals a clear structural separation: while benign normal tail windows and true anomalies exhibit comparable direct (D) and contextual (C) responses, true anomalies induce markedly stronger local (L) and relational (R) evidence due to underlying inter-sensor dependency breakdowns. By selectively integrating these complementary cues above $\tau _ { 0 } .$ , the score mapping in Fig. 4(c) and the zoomed-in tail trajectory in Fig. 4(d) effectively elevate genuine anomalous windows while attenuating isolated normal spikes. Consequently, under the fixed normal-training 99th-percentile threshold rule, TGEC substantially reduces false-alarm windows while preserving true anomaly recall, confirming its interpretability and practical value for reliable alarm prioritization.

![](images/d2a612c84485500fe2360c547346dde76a33963e093b439544e9ee5b5e836a73.jpg)  
Fig. 4. TGEC on SMD (GFM4MTSAD-O): (a) normal distributions and tail gate; (b) evidence medians and IQRs (D/L/R/C: direct/local/relation/context); (c) score mapping; (d) rank changes with the upper tail enlarged (both axes: 0.85–1). Normal/anomalous tail windows: 2,010/2,097. At the normal-q99 threshold, false positives decrease from 918 to 599 (34.7%).

## E. Parameter Study (RQ4)

We examine three hyperparameters of GFM4MTSAD across the same six entity–backend configurations. For TGEC, we vary $\alpha _ { 0 } \in \{ 0 . 0 1 , 0 . 0 2 5 , 0 . 0 5 , 0 . 1 0 , 0 . 2 0 \}$ while $\omega$ is fixed at 0.5, and vary $\omega \in \{ 0 , 0 . 2 5 , 0 . 5 , 0 . 7 5 , 1 \}$ } while $\alpha _ { 0 }$ is fixed at 0.05. The approximately geometric tail-mass grid spans the upper 1%–20% of the normal reference, covering a twentyfold change in selective re-ranking scope; the uniform fusion grid covers the full mixture interval, including both endpoints. These sweeps reuse the validation-selected checkpoints and frozen normal references. For IPRA, $r ~ \in ~ \{ 4 , 8 , 1 6 , 3 2 , 6 4 \}$ doubles the compensation bottleneck at each step under the same warm-start and 30-epoch joint-training protocol.

Impact of Tail Probability Mass $\alpha _ { 0 } .$ Increasing $\alpha _ { 0 }$ lowers the gate and monotonically expands test-window activation (Fig. 5). This relationship need not be linear, because activation depends on the test-score distribution. Across the full grid, AUPRC varies by at most 0.0249 on SMD and SMAP. MSL illustrates the role of selective gating: broader tails benefit MSL-A, whereas MSL-O retains its highest AUPRC over tail masses from 0.01 to 0.05. Thus, the gate controls where evidence fusion acts, separating tail re-ranking from preservation of the score body.

![](images/58735d87f878a5aa937045f7a0778e6a6d4547405dfb89d0e1f9dc4a7e9b3243.jpg)  
Fig. 5. Tail-mass sensitivity: top, AUPRC; bottom, test-window tail activation. A/O denote GFM4MTSAD-A/O. The fusion weight is fixed at 0.5. Dashed lines mark the default tail mass of 0.05. Horizontal axes are logarithmic; vertical ranges vary by panel.

![](images/686a097a63965cb1e47d0f66a4a65ea63b611bc3c83dc2dfb4c3dc1555f02b77.jpg)  
Fig. 6. Fusion-weight sensitivity: top, AUPRC; bottom, test-window tail activation. A/O denote GFM4MTSAD-A/O. The tail mass is fixed at 0.05; dashed lines mark the default fusion weight of 0.5. Vertical ranges vary by panel.

Impact of Evidence Fusion Weight ω. Varying ω leaves activation unchanged while adjusting rankings within the selected tail (Fig. 6). A weight of one recovers ECDF-only scoring; a weight of zero uses the auxiliary consensus subject to the tail floor. Intermediate weights improve AUPRC over both endpoints on SMAP with either backend and on SMD-O. SMD-A instead improves as the baseline contribution increases, while MSL-O remains constant because its gate is inactive. These responses distinguish the gate’s selection role from the fusion weight’s ranking role. Activation denotes gate membership, including when the fusion weight is one and scores are unchanged.

![](images/c1cc664b61b7810480503030dfb55d91409eb94d34a56c8a50f4f50cd753714e.jpg)  
Fig. 7. IPRA rank sensitivity: native AUPRC at validation-selected (top) and final (bottom, epoch 30) checkpoints. Left/right: GFM4MTSAD-A/O. Dashed lines mark the default rank of 16. All panels share the vertical scale; ranks are spaced logarithmically.

Impact of IPRA Bottleneck Rank r. The rank controls the capacity of IPRA’s residual compensation pathway. Fig. (7) compares validation-selected checkpoints with the final checkpoints after 30 epochs. Across the six entity–backend configurations, a rank of 16 achieves the highest mean native AUPRC under both conditions, reaching 0.5929 and 0.6392, respectively. Increasing the rank to 32 or 64 enlarges the IPRA parameter budget linearly without improving these averages. These results support a moderate bottleneck capacity and motivate retaining 16 as the shared default rank, balancing detection performance with compact residual compensation.

## V. CONCLUSION AND FUTURE WORK

This paper presented GFM4MTSAD, an adaptation framework that unleashes pretrained GFMs for unsupervised MT-SAD. Within the framework, SICA harmonizes temporal representations and sensor topologies through feature, structure, and information contracts while recovering projectionexcluded details via IPRA. MAPS furnishes targeted supervision through residual-preserving mechanism interventions near the normal boundary, and TGEC calibrates multi-granular evidence against normal operational references to refine tail anomaly candidates while preserving benign score rankings. Extensive experiments with AnomalyGFM and OwlEye backbones across MSL P-15, SMAP T-3, and SMD 1-1 demonstrate the effectiveness of our approach, where GFM4MTSAD achieves superior detection precision and ablations confirm the necessity of each contractual and calibration design. These findings establish that coordinated temporal and relational adaptation enables pretrained graph models to serve as effective anomaly detection backbones.

Future work will investigate how native token capacity and pretraining inductive biases govern the transferability of heterogeneous GFMs, motivating capacity-aware adaptation strategies that dynamically coordinate representation granularity across diverse foundation architectures. Beyond adapting off-the-shelf graph backbones, we plan to explore unified foundation pretraining across time series and graphs to colearn continuous temporal dynamics and discrete topological priors within a shared latent space. A further direction is continual adaptation under non-stationary operating regimes and distribution shifts, enabling structural priors and tail calibration references to evolve online while preserving pretrained foundation knowledge.

## REFERENCES

[1] A. P. Mathur and N. O. Tippenhauer, “Swat: A water treatment testbed for research and training on ics security,” in 2016 international workshop on cyber-physical systems for smart water networks (CySWater). IEEE, 2016, pp. 31–36.

[2] C. A. Rieth, B. D. Amsel, R. Tran, and M. B. Cook, “Issues and advances in anomaly detection evaluation for joint human-automated systems,” in International Conference on Applied Human Factors and Ergonomics Springer, 2017, pp. 52–63.

[3] A. L. Goldberger, L. A. Amaral, L. Glass, J. M. Hausdorff, P. C. Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E. Stanley, “Physiobank, physiotoolkit, and physionet: components of a new research resource for complex physiologic signals,” circulation, vol. 101, no. 23, pp. e215–e220, 2000.

[4] X. Yang, X. Qi, and X. Zhou, “Deep learning technologies for time series anomaly detection in healthcare: A review,” Ieee Access, vol. 11, pp. 117 788–117 799, 2023.

[5] D. Fahrmann, L. Mart¨ ´ın, L. Sanchez, and N. Damer, “Anomaly detection´ in smart environments: A comprehensive survey,” IEEE access, vol. 12, pp. 64 006–64 049, 2024.

[6] X. Sun, H. Zhou, Y. Wu, and C. Li, “State-derivative-aware neural controlled differential equations for multivariate time series anomaly detection and diagnosis,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 30, 2026, pp. 25 727–25 735.

[7] H. Kim, J. Mok, H. Lee, J. Shin, and S. Yoon, “Candi: Curated testtime adaptation for multivariate time-series anomaly detection under distribution shift,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 17, 2026, pp. 15 018–15 026.

[8] X. Huang, W. Chen, B. Hu, and Z. Mao, “Graph mixture of experts and memory-augmented routers for multivariate time series anomaly detection,” in Proceedings of the AAAI conference on artificial intelligence, vol. 39, no. 16, 2025, pp. 17 476–17 484.

[9] J. Jang and H.-Y. Kwon, “Tail-mil: time-aware and instance-learnable multiple instance learning for multivariate time series anomaly detection,” in Proceedings of the AAAI Conference on Artificial Intelligence vol. 39, no. 17, 2025, pp. 17 582–17 589.

[10] Z. Liu, M. Gao, and P. Jiao, “Gcad: Anomaly detection in multivariate time series from the perspective of granger causality,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 18, 2025, pp. 19 041–19 049.

[11] J. Liu, C. Zhang, J. Qian, M. Ma, S. Qin, C. Bansal, Q. Lin, S. Rajmohan, and D. Zhang, “Large language models can deliver accurate and interpretable time series anomaly detection,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2025, pp. 4623–4634.

[12] J. Audibert, P. Michiardi, F. Guyard, S. Marti, and M. A. Zuluaga, “Usad: Unsupervised anomaly detection on multivariate time series,” in Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, 2020, pp. 3395–3404.

[13] Y. Su, Y. Zhao, C. Niu, R. Liu, W. Sun, and D. Pei, “Robust anomaly detection for multivariate time series through stochastic recurrent neural network,” in Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2019, pp. 2828– 2837.

[14] M. Li, K. Liu, H. Chen, J. Bu, H. Wang, and H. Wang, “Tsinr: Capturing temporal continuity via implicit neural representations for time series anomaly detection,” in Proceedings of the 31st ACM SIGKDD conference on knowledge discovery and data mining v. 1, 2025, pp. 671–682.

[15] Y. Feng, W. Zhang, Y. Fu, W. Jiang, J. Zhu, and W. Ren, “Sensitivehue: Multivariate time series anomaly detection by enhancing the sensitivity to normal patterns,” in Proceedings of the 30th ACM SIGKDD Conference on knowledge discovery and data mining, 2024, pp. 782– 793.

[16] A. Deng and B. Hooi, “Graph neural network-based anomaly detection in multivariate time series,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 5, 2021, pp. 4027–4035.

[17] Z. Qi, J. Zhang, W. Li, and Z. Liang, “Cgsta: Cross-scale graph contrast with stability-aware alignment for multivariate time-series anomaly detection,” in International Conference on Database Systems for Advanced Applications. Springer, 2026, pp. 623–640.

[18] S. Zeng, G. Chao, J. Wei, Y. Yu, Z. Wang, and D. Chu, “Lagraph: Laplacian-guided graph learning for time series anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, 2026.

[19] H. Kim, J. Mok, D. Lee, J. Lew, S. Kim, and S. Yoon, “Causalityaware contrastive learning for robust multivariate time-series anomaly detection,” arXiv preprint arXiv:2506.03964, 2025.

[20] J. Jeong and H. Yoon, “Pgrf-net: A prototype-guided relational fusion network for diagnostic multivariate time-series anomaly detection,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 119 368–119 391.

[21] H. Qiao, C. Niu, L. Chen, and G. Pang, “Anomalygfm: Graph foundation model for zero/few-shot anomaly detection,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2025, pp. 2326–2337.

[22] L. Zheng, D. Fu, Z. Li, and J. He, “Owleye: Zero-shot learner for cross-domain graph data anomaly detection,” arXiv preprint arXiv:2601.19102, 2026.

[23] Z. Wang, Z. Zhang, N. V. Chawla, C. Zhang, and Y. Ye, “Gft: Graph foundation model with transferable tree vocabulary,” Advances in neural information processing systems, vol. 37, pp. 107 403–107 443, 2024.

[24] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” arXiv preprint arXiv:2106.09685, 2021.

[25] A. Zeng, M. Chen, L. Zhang, and Q. Xu, “Are transformers effective for time series forecasting?” in Proceedings of the AAAI conference on artificial intelligence, vol. 37, no. 9, 2023, pp. 11 121–11 128.

[26] F. Paischer, L. Hauzenberger, T. Schmied, B. Alkin, M. Deisenroth, and S. Hochreiter, “Parameter efficient fine-tuning via explained variance adaptation,” Advances in Neural Information Processing Systems, vol. 38, pp. 46 215–46 267, 2026.

[27] D. Chen, Y. Lin, W. Li, P. Li, J. Zhou, and X. Sun, “Measuring and relieving the over-smoothing problem for graph neural networks from the topological view,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 04, 2020, pp. 3438–3445.

[28] I. Naiman, N. B. Erichson, P. Ren, M. W. Mahoney, and O. Azencot, “Generative modeling of regular and irregular time series data via koopman vaes,” in International conference on learning representations, vol. 2024, 2024, pp. 50 547–50 573.

[29] C. Lu, Y. Zhou, F. Bao, J. Chen, C. Li, and J. Zhu, “Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps,” Advances in neural information processing systems, vol. 35, pp. 5775–5787, 2022.

[30] Y. Jin, J. Peng, Q. He, T. Hu, J. Wu, H. Chen, H. Wang, W. Zhu, M. Chi, J. Liu et al., “Dual-interrelated diffusion model for few-shot anomaly image generation,” in 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2025, pp. 30 420–30 429.

[31] A. Tank, I. Covert, N. Foti, A. Shojaie, and E. B. Fox, “Neural granger causality,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 8, pp. 4267–4279, 2022

[32] M. A. Pimentel, D. A. Clifton, L. Clifton, and L. Tarassenko, “A review of novelty detection,” Signal processing, vol. 99, pp. 215–249, 2014.

[33] A. Siffer, P.-A. Fouque, A. Termier, and C. Largouet, “Anomaly detection in streams with extreme value theory,” in Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, 2017, pp. 1067–1075.

[34] K. Hundman, V. Constantinou, C. Laporte, I. Colwell, and T. Soderstrom, “Detecting spacecraft anomalies using lstms and nonparametric dynamic thresholding,” in Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining, 2018, pp. 387–395.

[35] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in International conference on machine learning. PmLR, 2020, pp. 1597–1607.

[36] J. Tack, S. Mo, J. Jeong, and J. Shin, “Csi: Novelty detection via contrastive learning on distributionally shifted instances,” Advances in neural information processing systems, vol. 33, pp. 11 839–11 852, 2020.

[37] V. Sehwag, M. Chiang, and P. Mittal, “Ssd: A unified framework for selfsupervised outlier detection,” arXiv preprint arXiv:2103.12051, 2021.

[38] J. Xu, H. Wu, J. Wang, and M. Long, “Anomaly transformer: Time series anomaly detection with association discrepancy,” arXiv preprint arXiv:2110.02642, 2021.

[39] Y. Yang, C. Zhang, T. Zhou, Q. Wen, and L. Sun, “Dcdetector: Dual attention contrastive representation learning for time series anomaly detection,” in Proceedings of the 29th ACM SIGKDD conference on knowledge discovery and data mining, 2023, pp. 3033–3045.

[40] H. Kim, S. Kim, S. Min, and B. Lee, “Contrastive time-series anomaly detection,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 10, pp. 5053–5065, 2023.

[41] Z. Z. Darban, G. I. Webb, S. Pan, C. C. Aggarwal, and M. Salehi, “Carla: Self-supervised contrastive representation learning for time series anomaly detection,” Pattern Recognition, vol. 157, p. 110874, 2025.

[42] J. Park and S. Kang, “Paano: patch-based representation learning for time-series anomaly detection,” arXiv preprint arXiv:2602.01359, 2026.