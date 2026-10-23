# DISC: Domain-Integrated Similarity Cost for Time-Series Dataset

Yen-Ku Liu National Yang Ming Chiao Tung University yenkuliu.cs15@nycu.edu.tw

Hongjie Chen Dolby Laboratories hongjie.chen@dolby.com

Franck Dernoncourt Adobe Research dernonco@adobe.com

Ryan Rossi Adobe Research ryrossi@adobe.com

Abstract—The growth of time-series datasets and foundation models has made dataset comparison a central part of timeseries workflows. Assessing source-target transfer before model adaptation requires similarity estimates that can summarize transfer-related dataset relationships without exhaustive sourcetarget evaluation over every candidate pair. Many metrics have been proposed for measuring similarity between time-series datasets, but they capture different dataset relationships and vary in how well they align with downstream utility across tasks. We evaluate this gap by measuring the correlation between sourcetarget similarity scores and downstream forecasting errors in out-of-distribution and transfer-learning settings. We propose Domain-Integrated Similarity Cost (DISC), a learnable aggregation framework that uses calibration transfer observations to select, weight, and combine similarity metrics for different downstream tasks. By integrating complementary metric values, DISC better distinguishes source-target dataset differences that matter for downstream behavior. After calibration, DISC scores source-target pairs using only their similarity-metric values and yields higher alignment than single-metric references on the reported split. These results show that time-series dataset similarity is task-dependent, and that learned metric integration provides a stronger proxy for downstream transfer behavior than manually selecting one similarity metric.

Index Terms—time-series dataset similarity, transfer learning, foundation models, correlation evaluation

## I. INTRODUCTION

Time-series data appear in energy, transportation, weather, healthcare, finance, and web-traffic systems [1]–[5]. As timeseries research has expanded, dedicated resources have been developed for classification [6], anomaly detection [7], forecasting [8]–[10], and foundation-model development [11]– [13]. Recent foundation models for time-series forecasting make it possible to adapt pretrained models across heterogeneous domains [11], [14], [15]. This growth makes relevant source-target transfer assessment important for downstream time-series tasks, particularly when models rely on transfer, adaptation, or external data reuse [16]–[20]. In transferlearning and adaptation settings, source-target compatibility can affect whether fine-tuning provides useful signal [20]– [22], while a mismatched pair can degrade target-domain performance or introduce negative transfer [21], [23], [24].

As a result, quantifying similarity between time-series datasets becomes crucial. However, this utility depends on what relationship a metric captures: two datasets may be close in marginal distribution, temporal shape, representative pattern, or alignment structure, and each view can be more or less informative for a particular downstream task [25]. Figure 1 illustrates how different single similarity metrics expose different parts of the dataset relationship. In Fig. 1(a)– (b), both Wasserstein distance (WSD) and Match-and-Deform (MAD) identify the electricity-related datasets as similar, as indicated by the red outlines. MAD further sharpens this relation by revealing that electricity, electricity\_hourly, and electricity\_nips form a tighter core within the broader electricity group. However, MAD does not clearly mark m4\_daily and m4\_hourly as similar, whereas Dynamic Time Warping Barycenter Averaging with Longest Common Subsequence (DBA+LCSS) gives this green-outlined pair a clearer low-distance relation in Fig. 1(c). DBA+LCSS also gives the clearest similarity result for the orange-outlined mobility datasets uber\_tlc\_daily, uber\_tlc\_hourly, and vehicle\_trips\_wm, while WSD and MAD leave the pairs involving uber\_tlc\_hourly comparatively distant. At the same time, not every low-distance value should be trusted uniformly. For the yellow-outlined semantically mismatched group covid\_deaths, m1\_quarterly, and tourism\_quarterly, WSD, MAD, and DBA+LCSS all rank the three pairwise relations as relatively close, but Maximum Mean Discrepancy (MMD) ranks them as distant. The yellow-outlined case shows that a low distance under one metric can be misleading when other metrics capture a conflicting dataset relationship. Together, these examples motivate a learnable framework that integrates metric evidence according to downstream utility, rather than relying on any single metric that happens to mark a pair as similar.

![](images/a2d408697d5b1aaf0618bc5091cb04d6fa3b87c1c85d530d71cdbcf835f095f0.jpg)  
(a) Wasserstein distance

![](images/05f1d1dede4c61eb5a121de38439e192be1fd4008cda9a51be2039966d04f2cd.jpg)  
(b) Match-and-Deform

![](images/f073c75ffcd07ab4da98db69cbd458e35d55219d30b0e1f4771a1287762e6b2b.jpg)  
(c) Longest Common Subsequence

![](images/34f307c8e0d7f2f24e5f2cc94b961a567c9dd96bb3704a741b49ce707ff6440d.jpg)  
(d) Maximum Mean Discrepancy  
Fig. 1: Pairwise dataset distances under four similarity metrics. Darker cells indicate smaller distance and therefore higher metric-implied similarity. The panels illustrate why singlemetric similarity is insufficient: WSD captures a broad related group, MAD sharpens only part of that relation, DBA+LCSS reveals additional temporal-shape similarities, and MMD rejects a spurious group that the other metrics rank as close. This motivates learning an integrated similarity cost that combines complementary evidence while avoiding over-reliance on any single metric.

These metric-level differences matter because dataset similarity is used beyond pairwise comparison: it can support transfer assessment [26]–[28], dataset visualization and inspection [25], [26], benchmark curation [8], [10], and analysis of cross-domain generalization [22]. Since evaluating every source-target transfer setting is costly, dataset similarity has been used to prioritize source-target evaluations before downstream adaptation [26], [29], [30]. A useful similarity score should therefore vary with downstream transfer behavior, especially because source-target similarity can affect forecasting performance across zero-shot and fine-tuned settings [22], [31]. We therefore treat time-series dataset similarity as a correlation-alignment problem between source-target similarity scores and downstream task errors, which also explains why no single metric should be expected to dominate across all tasks. These observations suggest that dataset similarity should not be treated as a fixed metric choice.

To address the gap between single-metric similarity and downstream task behavior, this paper proposes Domain-Integrated Similarity Cost (DISC), a learnable aggregation framework for time-series dataset similarity. DISC uses observed transfer results on calibration datasets to learn metric mixtures that can then be applied to other source-target pairs using only dataset similarity values. The base mixture forms the primary similarity score from standardized metric values. The prior mixture forms a rank-based value from source-wise metric ranks. A learned sigmoid gate uses the prior value to modulate the base score. This design lets DISC emphasize metrics that provide useful cost value while using rank-based information as supplementary source-wise evidence. The resulting workflow integrates complementary metric evidence into a similarity score intended to better separate datasets whose downstream behaviors differ.

We evaluate DISC across out-of-distribution and transferlearning forecasting settings with multiple similarity configurations. On the reported split, DISC increases the average test correlation by about 10% across the four task settings.

Our contributions are:

• DISC integrates complementary dataset similarity metrics, which enables the learned similarity score to better discern source-target differences associated with downstream transfer behavior.

• DISC is flexible and task-adaptable to different downstream settings, using learnable aggregation parameters to adjust metric composition and weighting toward improved downstream-aligned similarity estimation.

• The learned metric weights provide an interpretable view of which dataset relationships are useful for a given model and task setting.

## II. RELATED WORK

Time-series similarity metrics. Existing similarity metrics capture different views of time-series dataset relationships. Dataset-level distances compare empirical collections directly. Maximum Mean Discrepancy (MMD) measures distribution discrepancy using kernel mean embeddings [32], Optimal Transport (OT) measures minimum transport cost between empirical samples [33]–[35], Wasserstein distance models each dataset through a Gaussian approximation and relates the resulting distance to downstream behavior [36], and Matchand-Deform (MAD) jointly optimizes sample matching and temporal deformation for time-series domain adaptation [37]. Series-level distances compare individual sequences and can be lifted to dataset comparison by first reducing each dataset to a representative sequence. Dynamic Time Warping (DTW) captures temporal warping [38], [39], Longest Common Subsequence (LCSS) captures subsequence matches while tolerating unmatched regions [40], [41], and Euclidean Distance (ED) compares aligned shapes [42], [43]. Reducers such as Dynamic Time Warping Barycenter Averaging (DBA) [44] and Principal Component Analysis (PCA) [45] make these sequence distances usable as dataset-level configurations, although they may discard within-dataset heterogeneity. Because these metric families emphasize different properties, their suitability can vary across datasets and downstream tasks [46].

Time-series foundation models and transfer. Time-series foundation models reuse broad temporal representations across domains and tasks, including Lag-Llama for probabilistic lagged-sequence forecasting [47] and Time-MoE for scaled mixture-of-experts forecasting [48]. Other foundation models, such as Chronos [11], TimesFM [12], Moirai [13], MOMENT [49], and LLM-reprogramming approaches, further show that sequence modeling can generalize across diverse time-series collections [18], [19]. These models make dataset reuse important because source-target compatibility can provide adaptation value or harmful bias, and retrievalaugmented and transferability-aware forecasting methods further show that performance depends on how external datasets are reused [16], [17], [27], [28]. This motivates evaluating similarity metrics by how well they align with downstream forecasting behavior, rather than only by mathematical or visual plausibility.

![](images/40f7abd114e7f845574aa963eeb804d04d7cd2b796c35d90ec33e21b1cefa568.jpg)  
Fig. 2: Overview of Domain-Integrated Similarity Cost (DISC). Given candidate similarity metrics $\{ s _ { m } ( X , Y ) \} _ { m \in \mathcal { M } }$ , DISC learns a task-specific base mixture $B ( X , Y )$ from standardized metric values $z _ { m } ( X , Y )$ and a prior mixture $P ( X , Y )$ from source-wise metric ranks $r _ { m } ( X , Y )$ . A learned gate $G ( X , Y )$ uses the prior mixture to modulate the base mixture and produce the final score $S _ { \theta } ( X , Y ) = B ( X , Y ) G ( X , Y )$ . The base and prior mixtures are detailed in Section IV-B, and the gated integration is detailed in Section IV-C.

## III. PRELIMINARIES

Let $X ~ = ~ \{ x _ { i } \} _ { i = 1 } ^ { N _ { X } }$ and $Y ~ = ~ \{ y _ { j } \} _ { j = 1 } ^ { N _ { Y } }$ denote two timeseries datasets. Each sequence $\boldsymbol { x } _ { i } = ( x _ { i , 1 } , \dots , x _ { i , T _ { i } } )$ and $y _ { j } =$ $( y _ { j , 1 } , \dots , y _ { j , T _ { j } } )$ may have a different length, with observations in $\mathbb { R } ^ { d }$ . Here, $N _ { X }$ and $N _ { Y }$ are the numbers of sequences in the two datasets, $T _ { i }$ and $T _ { j }$ are sequence lengths, and $d$ is the observation dimension. Time-series dataset similarity aims to define a dissimilarity function $s ( X , Y )$ , where smaller values indicate higher similarity.

We consider two classes of similarity methods. Datasetlevel metrics compare two datasets directly as $s _ { m } ( X , Y ) =$ $m ( X , Y )$ , where $m ( \cdot , \cdot )$ denotes a metric whose inputs are time-series datasets. However, not all useful time-series distances are defined between datasets. Some distances compare only individual time series, while our target object is the similarity between two time-series datasets. We therefore use a reducer to summarize each dataset into a representative sequence, making series-level distances applicable to datasetlevel comparison. Reducer-based series-level metrics first map each dataset to a representative sequence using a reducer $R ,$ then apply a series-level distance d as $s _ { R , d } ( X , Y ) \ =$ $d ( R ( X ) , R ( Y ) )$ . Here, $s _ { R , d }$ denotes the dataset-level distance induced by reducer R and series-level distance $d .$ This abstraction covers the dataset-level and reducer-based similarity configurations used in our experiments.

Let $\mathcal { U } = \{ X _ { 1 } , \ldots , X _ { n } \}$ denote the dataset universe. Here, n is the number of datasets and $\mathcal { M }$ is the set of candidate similarity methods. For each similarity method $m \in { \mathcal { M } }$ , the pairwise scores form a distance matrix $\mathbf { S } ^ { ( m ) } \in \mathbb { R } ^ { n \times n }$ , where $\ { \bf \bar { S } } _ { a b } ^ { ( m ) } \ = \ s _ { m } ( X _ { a } , X _ { b } )$ . The downstream evaluation produces directed error entries, represented as ${ \bf E } _ { a b } = E ( X _ { a } , X _ { b } )$ when the entry is available, where $E ( X _ { a } , X _ { b } )$ is the error obtained by using $X _ { a }$ as the source dataset and evaluating on $X _ { b }$ as the target dataset.

## IV. DISC: DOMAIN-INTEGRATED SIMILARITY COST

In this section, we introduce the core ideas of Domain-Integrated Similarity Cost (DISC). DISC is designed to learn how multiple time-series dataset similarity metrics should be selected, weighted, and combined for a given downstream task. Instead of assuming that one metric is always reliable, DISC treats each metric as a candidate signal that may capture a different aspect of the source-target dataset relationship. Figure 2 provides an overview of these components and how they produce the final dataset-similarity score. We first define the source-target problem setting in Section IV-A, describe base and prior metric selection in Section IV-B, introduce the gated integration mechanism in Section IV-C, and present the training objective and procedure in Section IV-D.

## A. Problem Setting

In this subsection, we define the source-target setting and the overall form of the DISC score. The input is a collection of source-target distance matrices and downstream error entries over the same dataset universe. The output is a scalar score for each ordered pair (X, Y). At a high level, DISC represents the final similarity score as

$$
S _ { \theta } ( X , Y ) = B ( X , Y ) G ( X , Y ) ,\tag{1}
$$

where $S _ { \theta } ( X , Y )$ denotes the final learned DISC score for the ordered source-target pair (X, Y) under parameters $\theta =$ $\{ w ^ { ( b ) } , w ^ { ( p ) } , a , c , k \}$ , B(X, Y) is a base similarity score learned from metric values, and $G ( X , Y )$ is a gated prior modulation learned from source-wise metric ranks. The parameter vectors $w ^ { ( b ) }$ and $w ^ { ( p ) }$ are the base and prior metric weights, while $a , c ,$ and k are the gate modulation strength, threshold, and sharpness. The following subsections define these two components and the parameters θ used to combine them. Figure 2 provides an overview of DISC, where candidate metrics are learned as complementary base and prior evidence, and a gated aggregation converts them into the final datasetsimilarity score. We use an ordered-pair formulation because downstream transfer is directed: a model trained on X and evaluated on $Y$ need not have the same error as a model trained on $Y$ and evaluated on X. DISC learns from directed source-target pairs.

The desired association is monotone with error. For a fixed source dataset X, if target dataset Y produces a larger downstream error than target dataset Z, then a useful distance should tend to assign a larger score to $( X , Y )$ than to $( X , Z )$ i.e., $S _ { \theta } ( X , Y ) ~ > ~ S _ { \theta } ( X , Z )$ . We do not require exact error prediction. This pairwise intuition motivates learning a score whose magnitude is positively aligned with source-to-target transfer error.

## B. Base and Prior Metric Selection

In this subsection, we describe how DISC converts raw metric distances into base and prior values, and how it learns which metrics should contribute to each role. Different metrics have different scales. For a candidate set ${ \mathcal { A } } ,$ each source row and metric is standardized to obtain $z _ { m } ( X , Y )$ , the normalized value of raw distance $s _ { m } ( X , Y )$ within the source row of X. We also compute the source-wise percentile rank within the same candidate set:

$$
r _ { m } ( X , Y ) = { \mathrm { R a n k P c t } } _ { Z \in A \backslash \{ X \} } ( s _ { m } ( X , Y ) ) \in [ 0 , 1 ] .\tag{2}
$$

Here, RankPct returns the percentile position of $s _ { m } ( X , Y )$ among the distances from source X to candidate targets $Z ,$ with larger values indicating relatively larger distance. The value feature $z _ { m }$ preserves normalized metric magnitude. The rank feature $r _ { m }$ preserves relative source-wise position and is less sensitive to scale. DISC uses these two feature types for different roles. Standardized metric values provide the base similarity value, while source-wise ranks provide a prior value over relative source-target evidence. This distinction is important in heterogeneous time-series repositories. For example, distributional metrics may produce large absolute values for high-scale datasets, whereas rank signals still indicate which targets are relatively close to a given source.

The base mixture selects and weights metrics that define the primary score:

$$
B ( X , Y ) = \sum _ { m \in \mathcal { M } } w _ { m } ^ { ( b ) } z _ { m } ( X , Y ) , \quad w ^ { ( b ) } = \mathrm { s o f t m a x } ( \alpha / \tau ) .\tag{3}
$$

Here, $w _ { m } ^ { ( b ) }$ is the learned base weight for metric $m ,$ α is the base-logit vector before softmax normalization, and $\tau$ is the softmax temperature. The learned logits $\alpha$ determine which metrics contribute to the base score. The temperature τ controls the concentration of the softmax mixture. The prior mixture independently selects and weights metrics that provide supplementary rank information:

$$
P ( X , Y ) = \sum _ { m \in \mathcal { M } } w _ { m } ^ { ( p ) } r _ { m } ( X , Y ) , \quad w ^ { ( p ) } = \mathrm { s o f t m a x } ( \beta / \tau ) .\tag{4}
$$

Here, $P ( X , Y )$ is the rank-based prior signal, $w _ { m } ^ { \left( p \right) }$ is the learned prior weight for metric $m ,$ and $\beta$ is the prior-logit vector. The learned logits $\beta$ determine which metrics contribute to the prior signal. Separating the base and prior mixtures allows DISC to use one set of metrics to define the main distance while using another to provide source-wise relative evidence.

## C. Gated Integration

In this subsection, we describe how the prior metric modulates the base metric to instantiate the overall score in Eq. 1. DISC combines the base and prior metrics with a controlled gate. The base mixture provides the main standardized score, while the prior mixture provides relative evidence about which target datasets look distant from a source under rank-based metric views. We therefore use the prior as a multiplicative interaction with the base score rather than as a standalone additive distance. The multiplier starts at one and changes the magnitude of the base score according to the learned prior signal. The prior signal modulates the base through

$$
G ( X , Y ) = 1 + a \sigma \left( k ( P ( X , Y ) - c ) \right) ,\tag{5}
$$

where $\sigma ( \cdot )$ is the sigmoid function. The three gate parameters have direct semantic roles. The modulation parameter a controls the strength of the multiplicative interaction, the threshold c determines the prior-rank level at which the interaction changes, and the sharpness k controls whether this transition is gradual or close to a hard cutoff. These parameters are needed because different downstream tasks may require different degrees of prior influence: some tasks benefit from broad rank-based modulation across many pairs, while others require the gate to respond mainly to high-prior, clearly distant pairs. Learning these parameters lets DISC decide whether a downstream task benefits from weak, broad prior modulation or from a sharper rank-conditioned interaction. The final DISC score uses this gate to adjust the base score, as shown in Eq. 1. This form keeps the base score interpretable while allowing rank evidence from other metrics to interact with it. Because

$B ( X , Y )$ is computed from standardized metric values, it is not constrained to be positive. Accordingly, the gate is interpreted as a learned multiplicative modulation of the base score, not as a standalone monotone penalty.

We constrain a with a sigmoid transform and a maximum modulation strength. We constrain c to [0, 1]. The gate sharpness k is learned within a bounded interval. In the final run, the bound is [1, 55]. This range is centered around the original fixed sharpness value of 40 but prevents the gate from becoming arbitrarily hard.

## D. Training Objective and Procedure

In this subsection, we present the loss function and training procedure used to learn the metric weights and gate parameters for each downstream task. Let $E ( X , Y )$ denote the downstream error for a source-target pair. These downstream errors are used as calibration signals for learning the aggregation parameters $\theta ,$ including the base weights $w ^ { ( b ) }$ , prior weights $w ^ { ( p ) }$ , and gate parameters $a , \ c ,$ and k. Once θ is fixed, the DISC score $S _ { \theta } ( X , Y )$ is computed from similarity features alone and does not require the downstream error of that pair. For an evaluation set A and a set of source datasets ${ \cal S } \subseteq { \cal A } .$ where A defines the candidate datasets compared in the current stage and S defines the source rows being averaged, the source-wise Pearson correlation is

$$
\rho _ { \mathrm { s r c } } = \frac { 1 } { | \mathcal { S } | } \sum _ { X \in \mathcal { S } } \rho \left( \{ S _ { \theta } ( X , Y ) \} _ { Y \in \mathcal { A } \backslash \{ X \} } , \{ E ( X , Y ) \} _ { Y \in \mathcal { A } \backslash \{ X \} } \right)\tag{6}
$$

Here, $\rho ( \cdot , \cdot )$ denotes Pearson correlation, and each term compares the DISC scores and downstream errors within one fixed source row. We also compute global Pearson correlation $\rho _ { \mathrm { g l o b a l } }$ over all source-target pairs. The training objective is

$$
\mathcal { I } ( \theta ) = ( 1 - \lambda ) \rho _ { \mathrm { s r c } } + \lambda \rho _ { \mathrm { g l o b a l } } ,\tag{7}
$$

where $\lambda = 0 . 1 0$ controls the trade-off between source-wise and global alignment, giving primary weight to source-wise correlation while retaining a small global-pair alignment term. Here, $\rho _ { \mathrm { s r c } }$ measures the average within-source alignment, and ρ<sub>global</sub> measures alignment over all directed source-target pairs. The optimized loss is

$$
\begin{array} { r } { \mathcal { L } = - \mathcal { I } ( \boldsymbol { \theta } ) + \eta H ( w ^ { ( b ) } , w ^ { ( p ) } ) + \mu \langle w ^ { ( b ) } , w ^ { ( p ) } \rangle . } \end{array}\tag{8}
$$

The entropy term $H ( w ^ { ( b ) } , w ^ { ( p ) } )$ regularizes the two softmax weight vectors and encourages sparse metric usage. The inner product $\langle w ^ { ( b ) } , w ^ { ( p ) } \rangle$ measures overlap between the base and prior metric mixtures, so the overlap term discourages the two mixtures from selecting the same metrics. The coefficients $\eta =$ 0.006 and $\mu = 0 . 0 2$ control the strengths of the entropy and overlap regularizers, respectively. All parameters are optimized with Adam.

Training proceeds independently for each downstream task. First, we compute candidate metric matrices from dataset samples. Second, we build directed source-target entries for the current optimization or evaluation stage. Third, we attach standardized metric values, source-wise ranks, and the corresponding downstream error entries for that stage. Fourth, we optimize the DISC parameters on the training datasets and select the checkpoint with the best validation sourcewise correlation. Finally, we freeze the learned parameters and report test-split source-wise correlation. Thus, downstream transfer results are used to calibrate how metrics should be combined, while the reported DISC scores on the test split are produced from the learned aggregation of similarity metrics.

TABLE I: DISC training hyperparameters used in the main experiment.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>Adam</td><td>Learning rate</td><td>0.03</td></tr><tr><td>Epochs</td><td>500</td><td>Patience</td><td>100</td></tr><tr><td>Restarts</td><td>8</td><td>Softmax temperature τ</td><td>0.55</td></tr><tr><td>Max gate modulation</td><td>3.0</td><td>Gate threshold range</td><td>[0, 1]</td></tr><tr><td>Gate sharpness range</td><td>[1,55]</td><td>Global-correlation weight λ</td><td>0.10</td></tr><tr><td>Entropy weight η</td><td>0.006</td><td>Overlap weight µ</td><td>0.02</td></tr></table>

The optimization uses random restarts. Each restart initializes the base and prior logits near zero, which starts from a nearly uniform mixture. The scalar gate parameters are initialized so that the gate is smooth at the beginning of training. We keep the validation criterion identical to the reporting metric, except that validation datasets replace test datasets. This keeps the validation criterion consistent with the final reporting metric. The learning rate controls the Adam step . size, epochs set the maximum number of training iterations, patience stops a restart when validation alignment no longer improves, and restarts repeat the optimization from different random initializations. The softmax temperature τ controls how concentrated the metric weights become. The maximum gate modulation bounds a, the gate threshold range bounds $c ,$ and the gate sharpness range bounds k. Table I lists the optimization and regularization settings used for all reported DISC runs.

## V. EXPERIMENTS

We structure the experiments around 3 research questions. RQ1. Do single similarity metrics consistently predict downstream errors?

RQ2. Does DISC show higher alignment than the best single metric on the reported split?

RQ3. Does learned aggregation outperform non-learned parameter search?

## A. Datasets and Splits

We use 46 datasets from GluonTS [50]. For similarity computation, each dataset is represented by 100 sampled windows of length 16. The final evaluation uses a shared split across the four forecasting tasks: 24 training datasets, 8 validation datasets, and 14 test datasets. The training datasets provide the calibration observations used to optimize DISC parameters. Validation datasets are used to select the calibrated aggregation setting. All reported numbers are computed on the designated test split. The 14 test datasets define 14 source rows, so evaluation repeatedly asks whether DISC explains transfer behavior for each newly source dataset against the other targets. The shared split controls for dataset-selection effects across tasks. Each downstream task uses the same dataset split, so differences in results reflect task behavior rather than changes in the evaluated dataset set.

TABLE II: Correlation between similarity value and downstream task error for single metrics and DISC. OT is the reference; parentheses show gains over OT, bold marks the best value, and italics mark the second-best single metric.
<table><tr><td rowspan="2">Task</td><td colspan="6">Correlation (gain over OT)</td></tr><tr><td>OT</td><td>WSD</td><td>MMD</td><td>MAD</td><td>DBA-DTW</td><td>DISC (ours)</td></tr><tr><td>OOD-TM</td><td>-0.007 (ref.)</td><td>0.445 (+0.453 ↑)</td><td> $- 0 . 0 5 7 \ ( - 0 . 0 5 0 \downarrow )$ </td><td>0.580 (+0.588 ↑)</td><td> $- 0 . 0 1 6 \ ( - 0 . 0 0 8 \downarrow )$ </td><td> $\mathbf { 0 . 6 0 1 \ ( + 0 . 6 0 9 \uparrow ) }$ </td></tr><tr><td>OOD-LL</td><td>0.125 (ref.)</td><td>0.361 (+0.235 ↑)</td><td> $0 . 2 9 5 \ ( + 0 . 1 6 9 \uparrow )$ </td><td>0.231 (+0.105 ↑)</td><td> $\theta . 6 I { \ 8 \stackrel { . } { ( + 0 . 4 9 2 \uparrow ) } }$ </td><td> $\mathbf { 0 . 6 9 7 \ ( + 0 . 5 7 1 \uparrow ) }$ </td></tr><tr><td>TL-TM</td><td>0.213 (ref.)</td><td>0.516 (+0.303 ↑)</td><td> $0 . 3 0 3 \ ( + 0 . 0 9 1 \uparrow )$ </td><td>0.505 (+0.292 ↑)</td><td> $0 . 1 3 4 \ ( - 0 . 0 7 9 \downarrow )$ </td><td> $\mathbf { 0 . 6 1 7 \ ( + 0 . 4 0 4 \uparrow ) }$ </td></tr><tr><td> $\mathrm { T L } { \mathrm { - } } \mathrm { L L }$ </td><td>0.080 (ref.)</td><td>0.011 (−0.069 ↓)</td><td> $0 . 4 3 5 \ : ( + 0 . 3 5 5 \ : \uparrow )$ </td><td> $0 . 1 4 6 \ ( + 0 . 0 6 6 \uparrow )$ </td><td> $0 . 3 0 2 \ ( + 0 . 2 2 2 \ \hat { \uparrow } )$ </td><td> $\mathbf { 0 . 4 7 9 \ ( + 0 . 3 9 9 \uparrow ) }$ </td></tr><tr><td> $\mathbf { A v \mathbf { g } }$ </td><td>0.103 (ref.)</td><td> $0 . 3 3 3 \ ( + 0 . 2 3 1 \uparrow )$ </td><td> $0 . 2 4 4 \ ( + 0 . 1 4 1 \uparrow )$ </td><td> $0 . 3 6 6 \ ( + 0 . 2 6 3 \uparrow )$ </td><td> $0 . 2 5 9 \ ( + 0 . 1 5 7 \uparrow )$ </td><td> $\mathbf { 0 . 5 9 8 \ ( + 0 . 4 9 6 \uparrow ) }$ </td></tr></table>

TABLE III: Correlation between similarity value and downstream task error for single metrics and DISC (cont.). OT is the reference; parentheses show gains over OT, bold marks the best value, and italics mark the second-best single metric.
<table><tr><td rowspan="2">Task</td><td colspan="7">Correlation (gain over OT)</td></tr><tr><td>OT</td><td>DBA-ED</td><td>DBA-LCSS</td><td>PCA-DTW</td><td>PCA-ED</td><td>PCA-LCSS</td><td>DISC (ours)</td></tr><tr><td>OOD-TM</td><td>-0.007 (ref.)</td><td> $- 0 . 0 1 6 \ ( - 0 . 0 0 8 \downarrow )$ </td><td> $0 . 0 0 0 \ ( + 0 . 0 0 7 \uparrow )$ </td><td>0.381 (+0.389 ↑)</td><td> $0 . 4 1 6 \ ( + 0 . 4 2 3 \uparrow )$ </td><td> $0 . 0 4 8 \ ( + 0 . 0 5 5 \ \uparrow )$ </td><td> $\mathbf { 0 . 6 0 1 \ ( + 0 . 6 0 9 \uparrow ) }$ </td></tr><tr><td>OOD-LL</td><td>0.125 (ref.)</td><td> $ { 0 . 6 I 8 \ ( + 0 . 4 9 2 \uparrow ) }$ </td><td> $0 . 0 5 2 \ ( - 0 . 0 7 4 \downarrow )$ </td><td>0.577 (+0.452 ↑)</td><td> $0 . 5 8 0 \ ( + 0 . 4 5 4 \ \uparrow )$ </td><td> $0 . 4 6 1 \ ( + 0 . 3 3 6 \ \uparrow )$ </td><td> $\mathbf { 0 . 6 9 7 \ ( + 0 . 5 7 1 \uparrow ) }$ </td></tr><tr><td> $\mathrm { T L } { - } \mathrm { T M }$ </td><td>0.213 (ref.)</td><td> $0 . 1 3 4 \ ( - 0 . 0 7 9 \downarrow )$ </td><td> $0 . 0 2 6 \ ( - 0 . 1 8 7 \downarrow )$ </td><td>0.482 (+0.269 ↑)</td><td> $0 . 5 3 7 \ : ( + 0 . 3 2 5 \ : \uparrow )$ </td><td> $- 0 . 0 2 5 \ ( - 0 . 2 3 8 \ \downarrow )$ </td><td> $\mathbf { 0 . 6 1 7 \ ( + 0 . 4 0 4 \ { \hat { \uparrow } } ) }$ </td></tr><tr><td> $\mathrm { T L } \mathrm { - } \mathrm { L L }$ </td><td>0.080 (ref.)</td><td> $0 . 3 0 2 \ ( + 0 . 2 2 2 \uparrow )$ </td><td> $0 . 0 0 0 \ ( - 0 . 0 8 0 \ \downarrow )$ </td><td>0.113 (+0.033 ↑)</td><td> $0 . 0 7 8 \ ( - 0 . 0 0 2 \downarrow )$ </td><td>0.107 (+0.027 ↑)</td><td> $\mathbf { 0 . 4 7 9 \ ( + 0 . 3 9 9 \ \hat { \uparrow } ) }$ </td></tr><tr><td> $\mathbf { A v } \mathbf { g }$ </td><td>0.103 (ref.)</td><td> $0 . 2 5 9 \ ( + 0 . 1 5 7 \ \uparrow )$ </td><td> $0 . 0 1 9 \ ( - 0 . 0 8 3 \downarrow )$ </td><td>0.388 (+0.286 ↑)</td><td> $0 . 4 0 3 \ ( + 0 . 3 0 0 \uparrow )$ </td><td>0.148 (+0.045 ↑)</td><td> $\mathbf { 0 . 5 9 8 \ ( + 0 . 4 9 6 \ \hat { \uparrow } ) }$ </td></tr></table>

## B. Downstream Tasks

We evaluate forecasting performance using two foundation models, Time-MoE [48] and Lag-Llama [47], under outof-distribution and transfer-learning protocols. In the outof-distribution (OOD) protocol, a model is fine-tuned on a source dataset and evaluated on target datasets without target adaptation. In the transfer-learning (TL) protocol, the model is first fine-tuned on a source dataset and then adapted with target-side data before final target evaluation. With 46 datasets, the OOD protocol fine-tunes 46 source models for each foundation model and evaluates them on $4 6 \times 4 5 = 2 . 0 7 0$ directed source-target pairs per foundation model after excluding self-pairs. The TL protocol starts from the same source fine-tuned models, but additionally adapts each source model to each target dataset, resulting in 2,070 target-adapted source-target runs per foundation model. This gives four downstream tasks: Out-of-distribution Time-MoE, Out-of-distribution Lag-Llama, Transfer learning with Time-MoE, and Transfer learning with Lag-Llama. The downstream error matrix records the loss for each source-target pair.

## C. Similarity Metrics

We compare multiple similarity configurations that encode different assumptions about time-series dataset relationships:

• Dataset-level metrics: Wasserstein Distance (WSD) [36] captures distributional mismatch through moment-based geometry, Maximum Mean Discrepancy (MMD) [32] compares empirical distributions through kernel mean embeddings, Optimal Transport (OT) [33], [34] measures the minimum transport cost between empirical samples, and Match-and-

Deform (MAD) [37] combines sample matching with temporal alignment.

• Dataset reducers: Dynamic Time Warping Barycenter Averaging (DBA) [44] summarizes each dataset as a DTW barycenter, while Principal Component Analysis (PCA) [45] summarizes each dataset through a dominant projection. These reducers are not standalone similarity metrics in our comparison; they provide representative sequences for series-level distances.

• Series-level distances: Dynamic Time Warping (DTW) [38], Euclidean Distance [42], [43], and Longest Common Subsequence (LCSS) [40], [41] compare the DBA or PCA representatives through warping, aligned shape distance, or subsequence matching.

Combining the DBA and PCA reducers with DTW, Euclidean distance, and LCSS yields six reducer-based configurations: DBA-DTW, DBA-ED, DBA-LCSS, PCA-DTW, PCA-ED, and PCA-LCSS. Together with the four dataset-level metrics, these configurations cover complementary views of source-target relationships, allowing the experiments to test whether their usefulness varies across models, tasks, and adaptation settings.

## D. Evaluation Protocol

For each source dataset in the test split, we compute Pearson correlation between its similarity distances to the other test datasets and the corresponding downstream errors. The purpose is to test whether a pre-adaptation DISC score is comparable to the forecasting error observed after using source dataset X on target dataset Y. This evaluates source-conditioned alignment between dataset similarity and downstream behavior, rather than target-wise top-k source retrieval. With 14 test datasets, each source-level correlation uses 13 target points. The reported score is the mean of the 14 source-level correlations, treating each source row as a separate evaluation case. Higher positive values indicate stronger alignment between dataset distance and downstream error. Because the downstream error E(X, Y) also reflects the intrinsic forecasting difficulty of target dataset Y, the reported correlations measure alignment with observed sourcetarget forecasting behavior rather than isolating transfer compatibility alone. In this analysis, target difficulty is part of the realized downstream cost that a source-target screening score should anticipate, not a quantity that DISC is claimed to decompose from pair-specific compatibility. This sourcewise definition prevents a few high-variance sources from dominating the global result. It evaluates whether a fixed source dataset induces a meaningful score-error association over target datasets. Because transfer errors are directed, we use this metric specifically to evaluate source-row transfer alignment. The source-wise view asks whether each source row provides a meaningful local explanation of downstream transfer behavior. This evaluation is therefore a correlationalignment analysis, not a direct source-selection benchmark.

(a) OOD Time-MoE  
![](images/9bfd13f7de46d1f92e7e4fd822a7a15f8ecaaf3eb3980990574458bb65958d92.jpg)  
(c) TL Time-MoE

(b) OOD Lag-Llama  
![](images/237d6941e20504a497a0a06897354e334f11dc38e816b4788c580a567f57b46b.jpg)  
(d) TL Lag-Llama

![](images/5fdacdc89a170203490bb7d399a28e7d0ac1c3ec337a1d45c1b3fba5a403aade.jpg)

![](images/283c5f7354070fdd384478e735081dbd4c552d8fb8faccb643425e611e1889f3.jpg)  
DISC similarity cost  
Fig. 3: Source-wise DISC examples for RQ2. Each panel fixes the test source dataset with the highest DISC source-wise Pearson correlation for one downstream task: (a) m3\_monthly, (b) electricity\_weekly, (c) wiki2000\_nips, and (d) fred\_md. The x-axis is the learned DISC similarity score and the y-axis is downstream forecasting error. Upward trends show that datasets assigned larger DISC scores tend to produce larger errors.

## E. RQ1: Single Similarity Configurations Are Task-Dependent

Tables II and III report all single similarity configurations and DISC, Optimal Transport is used as the reference metric, and the parenthesized values show how much each configuration increases or decreases correlation relative to Optimal Transport. Single configurations are inconsistent across tasks. MAD is the strongest single metric for OOD-Time-MoE. DBA-DTW and DBA-ED are strongest for OOD-Lag-Llama. PCA-ED is strongest for TL-Time-MoE. MMD is strongest for TL-Lag-Llama. The best single metric changes with the downstream model and protocol.

The average single-metric result also shows the limitation of manual selection. PCA-ED has the highest average among single metrics, with 0.403. The per-task oracle, which selects the best single metric separately for each task, reaches 0.543. DISC exceeds both, reaching 0.598. The gap between the best average single metric and the per-task oracle is large. This means that a metric selected once for all tasks is not sufficient. The remaining gap between the per-task oracle and DISC shows that aggregation can improve even when the strongest single metric is selected separately for each task.

TABLE IV: DISC compared with the best single metric for each downstream task on the reported split. R denotes Pearson correlation between similarity score and downstream task error; higher values indicate stronger alignment.
<table><tr><td colspan="2">Task</td><td rowspan="2">Best single</td><td rowspan="2">Single R ↑</td><td rowspan="2">DISC R ↑</td><td rowspan="2">Relative change ↑</td></tr><tr><td>Protocol</td><td>Model</td></tr><tr><td rowspan="3">Out-of-distribution</td><td>Time-MoE</td><td>MAD</td><td>0.580</td><td>0.601</td><td>+3.55%</td></tr><tr><td>Lag-Llama</td><td>DBA-DTW</td><td>0.618</td><td>0.697</td><td>+12.84%</td></tr><tr><td>Time-MoE</td><td>PCA-ED</td><td>0.537</td><td>0.617</td><td>+14.75%</td></tr><tr><td>Transfer learning</td><td>Lag-Llama</td><td>MMD</td><td>0.435</td><td>0.479</td><td>+10.15%</td></tr><tr><td colspan="2">Average</td><td>一</td><td>0.543</td><td>0.598</td><td>+10.29%</td></tr></table>

TABLE V: Learned DISC aggregation for each downstream task. Base and prior columns list selected metrics with learned softmax weights in parentheses; metrics below 0.01 are omitted. Prior modulation reports the mean and range of the positive gate factor multiplied with the base score on test-split pairs.
<table><tr><td colspan="2">Task</td><td rowspan="2">Base metrics</td><td rowspan="2">Prior metrics</td><td rowspan="2">Prior modulation</td></tr><tr><td>Protocol</td><td>Model</td></tr><tr><td rowspan="2">Out-of-distribution</td><td>Time-MoE</td><td>DBA-LCSS (.549), MAD (.402) Wasserstein (.022)</td><td>PCA-ED (.717), Wasserstein (.173) DBA-ED (.045), DBA-DTW (.037), PCA-LCSS (.012)</td><td>2.26× [1.00, 3.35]</td></tr><tr><td>Lag-Llama</td><td>PCA-DTW (.612) PCA-LCSS (.383)</td><td>PCA-LCSS (.935) DBA-LCSS (.062)</td><td>1.21 × [1.00, 3.79]</td></tr><tr><td rowspan="2">Transfer learning</td><td>Time-MoE</td><td>MAD (.916) Wasserstein (.067)</td><td>PCA-ED (.719), DBA-ED (.180) DBA-DTW (.076), MMD (.017)</td><td>1.17× [1.00, 3.81]</td></tr><tr><td>Lag-Llama</td><td>MMD (.833) DBA-ED (.159)</td><td>DBA-DTW (.362), MAD (.324) PCA-ED (.312)</td><td>2.89× [1.00, 3.95]</td></tr></table>

F. RQ2: DISC Shows Gains Through Task-Specific Aggregation

Table IV compares DISC with the best single metric for each task. On the reported split, DISC is higher than the best single metric in all four task columns. The average relative change is 10.29%. The largest margins occur for OOD-Lag-Llama and TL-Time-MoE, where the best single metric leaves substantial unexplained structure. The OOD-Time-MoE margin is modest because MAD is already strong on this task. We therefore interpret this case as a small directional gain rather than as evidence of a large effect. The TL-Lag-Llama task is the weakest absolute task for both single metrics and DISC, but DISC is still higher than MMD by 10.15%. These results are based on one shared train-validation-test split, but the test score is averaged over 14 source-row evaluations and repeated across four downstream settings. We therefore interpret the evidence as source-row transfer alignment under the shared split, rather than as a repeated-split uncertainty analysis over all possible dataset partitions.

Figure 3 provides source-level examples of how DISC separates target datasets across different downstream tasks. We show one representative fixed source from each forecasting setting, chosen as the source with the strongest DISC sourcewise correlation, and compare its learned DISC scores with the downstream errors of the remaining test-split target datasets. Across the four panels, targets with larger DISC scores generally incur larger forecasting errors. This pattern indicates that DISC does not only improve the aggregate correlation in Table IV; it also produces visible local score-error alignment within individual source rows. Such source-wise separation is useful for analyzing how transfer error changes from a fixed source to different target datasets. Thus, the figure supports the intended source-conditioned interpretation: DISC scores are compared with the realized downstream errors of using the fixed source on each target. The panels are illustrative examples, while the reported RQ2 scores average this sourcewise alignment over all 14 source rows.

The observed gain comes from task-specific metric selection and integration rather than from a fixed global metric choice. Table V reports the sparse components of the learned base and prior mixtures, their learned weights, and the resulting prior-gate modulation factor on test-split pairs. The base weights indicate how DISC combines metric values to form the primary similarity score. The prior weights indicate which metric ranks are used to form the gate signal. The prior modulation factor summarizes how strongly the prior signal interacts with the learned base score on the test pairs. OOD-Time-MoE uses MAD with DBA-LCSS and Wassersteinrelated information. OOD-Lag-Llama relies on PCA-based shape summaries. TL-Time-MoE uses MAD and Wasserstein in the base and uses PCA-ED, DBA-ED, DBA-DTW, and MMD as priors. TL-Lag-Llama uses MMD and DBA-ED as the base and uses DBA-DTW, MAD, and PCA-ED as priors. The learned components match the task-dependent behavior in Tables II and III. They also show why the method should not be described as improving one particular metric. For example, DISC uses MMD as part of the TL-Lag-Llama base, but not as the OOD-Lag-Llama base. It uses PCA-LCSS as both a base and prior signal in OOD-Lag-Llama, but uses PCA-ED as a prior in the Time-MoE tasks. The learned structure is therefore task-specific rather than metric-specific.

TABLE VI: Task-wise ablation against non-learned manual search. R denotes Pearson correlation between similarity distance and downstream task error. Manual R and DISC R are test-split correlations after parameter selection on training and validation datasets; Gain is DISC’s relative improvement, with N/A used for a negative manual baseline.
<table><tr><td>Task</td><td>Manual R</td><td>DISC R</td><td>Gain</td></tr><tr><td>OOD-TM</td><td>-0.066</td><td>0.601</td><td>N/A</td></tr><tr><td>OOD-LL</td><td>0.571</td><td>0.697</td><td>+21.98%</td></tr><tr><td>TL-TM</td><td>0.316</td><td>0.617</td><td>+95.13%</td></tr><tr><td>TL-LL</td><td>0.344</td><td>0.479</td><td>+39.32%</td></tr><tr><td>Avg</td><td>0.291</td><td>0.598</td><td>+105.38%</td></tr></table>

## G. RQ3: Learned Aggregation Versus Manual Search

We compare DISC with a non-learned manual search over the same base-prior aggregation family. This comparison tests whether the best similarity-metric combination found from previously observed datasets can be directly reused for unseen datasets. The manual search samples sparse base mixtures, sparse prior mixtures, and gate parameters, then selects the best configuration using the training and validation datasets. The selected configuration is applied once to the same test split used in Tables II and III. This keeps the manualsearch baseline under the same selection protocol as DISC. It therefore asks whether broad manual parameter search can replace learned aggregation under the same evaluation setting.

The answer is no. Table VI reports the task-wise test-split comparison. The extended manual search evaluates approximately 7.9 million candidate configurations per task, but its average test-split correlation is 0.291. DISC reaches 0.598 under the same test-split protocol. The manual search also remains below DISC on every task, with test correlations of -0.066, 0.571, 0.316, and 0.344 for the four settings. The learned combinations remain effective on the fourteen test datasets, indicating that the selected metric weights and gate behavior still align with newly evaluated source-target relationships. The N/A gain in Table VI occurs for OOD-Time-MoE because the manual-search baseline is negative, making a relative percentage gain misleading. Its learnable parameters provide a reliable way to select metric weights and gate behavior that generalize from the training and validation setting to the test split.

## H. Discussion: Dataset Characteristics and Metric Reliability

We further examine the dataset characteristics associated with weaker single-metric explanations of downstream behavior. The pattern is visible in the same N = 100, L = 16 sampled windows used for similarity computation, as summarized in Table VII. Here, N is the number of sampled windows per dataset and L is the sampled-window length. These patterns suggest that the issue is not only metric design; overly smooth, flat, or low-diversity sampled representations can also limit how well similarity metrics explain source-target relationships.

The results show that similarity metrics are not interchangeable, rank information can complement standardized metric values, and sampling choices affect similarity estimation when short windows miss seasonality or overrepresent flat regimes. They also clarify that Table IV is a favorable baseline because it selects the best single metric per task from the test result table, a choice unavailable before evaluation. DISC improving over this reference suggests that learned aggregation does more than choose the obvious best metric.

TABLE VII: Sample-window diagnostics for low- and highcorrelation groups. Lag-1 is adjacent-step autocorrelation, zero is the zero-valued sample fraction, and diversity is mean pairwise distance between z-normalized windows.
<table><tr><td>Group</td><td>Lag-1</td><td>Zero</td><td>Diversity</td></tr><tr><td>Low corr. datasets</td><td>0.841</td><td>0.191</td><td>3.935</td></tr><tr><td>High corr. datasets</td><td>0.556</td><td>0.061</td><td>5.115</td></tr></table>

## VI. CONCLUSION

We introduced DISC, a learnable framework for time-series dataset similarity. DISC integrates dataset-level and reducerbased similarity configurations through a calibrated base mixture and a rank-conditioned prior modulation, producing a combined similarity score that better separates source-target relationships across downstream tasks. On the reported split across four forecasting tasks with Time-MoE and Lag-Llama, DISC is higher than the best single metric in every task column and increases the average correlation by 10.29%. The study shows that source-target transfer alignment is task-dependent and cannot be reliably solved by one fixed metric. It also shows that short, flat, or low-diversity sampled windows can weaken single-metric correlation. These results support learned metric aggregation as a practical way to turn calibration transfer observations and heterogeneous similarity signals into a taskaware score for time-series source-target transfer assessment.

## REFERENCES

[1] L. Zhang, C. Aggarwal, and G.-J. Qi, “Stock price prediction via discovering multi-frequency trading patterns,” in Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining, 2017, pp. 2141–2149.

[2] Z. Che, S. Purushotham, K. Cho, D. Sontag, and Y. Liu, “Recurrent neural networks for multivariate time series with missing values,” Scientific reports, vol. 8, no. 1, p. 6085, 2018.

[3] T. Nguyen, J. Brandstetter, A. Kapoor, J. K. Gupta, and A. Grover, “Climax: A foundation model for weather and climate,” arXiv preprint arXiv:2301.10343, 2023.

[4] R. Lam, A. Sanchez-Gonzalez, M. Willson, P. Wirnsberger, M. Fortunato, F. Alet, S. Ravuri, T. Ewalds, Z. Eaton-Rosen, W. Hu et al., “Learning skillful medium-range global weather forecasting,” Science, vol. 382, no. 6677, pp. 1416–1421, 2023.

[5] H. Wu, T. Hu, Y. Liu, H. Zhou, J. Wang, and M. Long, “Timesnet: Temporal 2d-variation modeling for general time series analysis,” arXiv preprint arXiv:2210.02186, 2022.

[6] A. Bagnall, H. A. Dau, J. Lines, M. Flynn, J. Large, A. Bostrom, P. Southam, and E. Keogh, “The uea multivariate time series classification archive, 2018,” arXiv preprint arXiv:1811.00075, 2018.

[7] J. Paparrizos, Y. Kang, P. Boniol, R. S. Tsay, T. Palpanas, and M. J. Franklin, “Tsb-uad: An end-to-end benchmark suite for univariate timeseries anomaly detection.” Proc. VLDB Endow., vol. 15, no. 8, pp. 1697– 1711, 2022.

[8] Y. Hahn, T. Langer, R. Meyes, and T. Meisen, “Time series dataset survey for forecasting with deep learning,” Forecasting, vol. 5, no. 1, pp. 315–335, 2023.

[9] Y. Wang, H. Wu, J. Dong, Y. Liu, C. Wang, M. Long, and J. Wang, “Deep time series models: A comprehensive survey and benchmark,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2026.

[10] T. Aksu, G. Woo, J. Liu, X. Liu, C. Liu, S. Savarese, C. Xiong, and D. Sahoo, “Gift-eval: A benchmark for general time series forecasting model evaluation,” arXiv preprint arXiv:2410.10393, 2024.

[11] A. F. Ansari, L. Stella, C. Turkmen, X. Zhang, P. Mercado, H. Shen, O. Shchur, S. S. Rangapuram, S. P. Arango, S. Kapoor et al., “Chronos: Learning the language of time series,” arXiv preprint arXiv:2403.07815, 2024.

[12] A. Das, W. Kong, R. Sen, and Y. Zhou, “A decoder-only foundation model for time-series forecasting,” arXiv preprint arXiv:2310.10688, 2023.

[13] G. Woo, C. Liu, A. Kumar, C. Xiong, S. Savarese, and D. Sahoo, “Unified training of universal time series forecasting transformers,” arXiv preprint arXiv:2402.02592, 2024.

[14] S. R. K. Kottapalli, K. Hubli, S. Chandrashekhara, G. Jain, S. Hubli, G. Botla, and R. Doddaiah, “Foundation models for time series: A survey,” arXiv preprint arXiv:2504.04011, 2025.

[15] Q. Ma, Z. Liu, Z. Zheng, Z. Huang, S. Zhu, Z. Yu, and J. T. Kwok, “A survey on time-series pre-trained models,” IEEE transactions on knowledge and data engineering, vol. 36, no. 12, pp. 7536–7555, 2024.

[16] S. Han, S. Lee, M. Cha, S. O. Arik, and J. Yoon, “Retrieval augmented time series forecasting,” arXiv preprint arXiv:2505.04163, 2025.

[17] K. Ning, Z. Pan, Y. Liu, Y. Jiang, J. Zhang, K. Rasul, A. Schneider, L. Ma, Y. Nevmyvaka, and D. Song, “Ts-rag: Retrieval-augmented generation based time series foundation models are stronger zero-shot forecaster,” Advances in Neural Information Processing Systems, vol. 38, pp. 163 170–163 199, 2026.

[18] M. Jin, S. Wang, L. Ma, Z. Chu, J. Zhang, X. Shi, P.-Y. Chen, Y. Liang, Y.-F. Li, S. Pan et al., “Time-llm: Time series forecasting by reprogramming large language models,” in International conference on learning representations, vol. 2024, 2024, pp. 23 857–23 880.

[19] X. Liu, J. Hu, Y. Li, S. Diao, Y. Liang, B. Hooi, and R. Zimmermann, “Unitime: A language-empowered unified model for cross-domain time series forecasting,” in Proceedings of the ACM Web Conference 2024, 2024, pp. 4095–4106.

[20] M. Weber, M. Auch, C. Doblander, P. Mandl, and H.-A. Jacobsen, “Transfer learning with time series data: A systematic mapping study,” Ieee Access, vol. 9, pp. 165 409–165 432, 2021.

[21] S. J. Pan and Q. Yang, “A survey on transfer learning,” IEEE Transactions on knowledge and data engineering, vol. 22, no. 10, pp. 1345– 1359, 2009.

[22] C. Ehrig, B. Sonnleitner, U. Neumann, C. Cleophas, and G. Forestier, “The impact of data set similarity and diversity on transfer learning success in time series forecasting,” arXiv preprint arXiv:2404.06198, 2024.

[23] Z. Wang, Z. Dai, B. Póczos, and J. Carbonell, “Characterizing and avoiding negative transfer,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2019, pp. 11 285– 11 294.

[24] W. Zhang, L. Deng, L. Zhang, and D. Wu, “A survey on negative transfer,” IEEE/CAA Journal of Automatica Sinica, vol. 10, no. 2, pp. 305–329, 2022.

[25] J. Paparrizos, H. Li, F. Yang, K. Wu, J. E. d’Hondt, and O. Papapetrou, “A survey on time-series distance measures,” arXiv preprint arXiv:2412.20574, 2024.

[26] S. Sun and H. H. Zhang, “Quantifying dataset similarity to guide transfer learning,” arXiv e-prints, pp. arXiv–2510, 2025.

[27] W. Zhang, X. Chen, X. Li, K. Chen, W. Guan, and L. Nie, “Unified transferability metrics for time series foundation models,” Advances in Neural Information Processing Systems, vol. 38, pp. 41 155–41 178, 2026.

[28] Q. Yao, M. Jin, C. Zhang, C.-H. H. Yang, J. Qi, and S. Pan, “Estimating time series foundation model transferability via in-context learning,” arXiv preprint arXiv:2509.23695, 2025.

[29] Z. Zhang, K. Zhou, X. Wang, M. Zhu, J. Gao, J. Mei, and Q. Huang, “A similarity-based time series source dataset selection method for transfer learning,” in Journal of Physics: Conference Series, vol. 2428, no. 1. IOP Publishing, 2023, p. 012038.

[30] Y. Ding, B. Jiang, A. Yu, A. Zheng, and J. Liang, “Which model to transfer? a survey on transferability estimation,” arXiv preprint arXiv:2402.15231, 2024.

[31] S. Jain, H. Salman, A. Khaddaj, E. Wong, S. M. Park, and A. M ˛adry, “A data-based perspective on transfer learning,” in 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2023, pp. 3613–3622.

[32] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola, “A kernel two-sample test,” The journal of machine learning research, vol. 13, pp. 723–773, 2012.

[33] M. Cuturi, “Sinkhorn distances: Lightspeed computation of optimal transport,” Advances in neural information processing systems, vol. 26, 2013.

[34] P. Gabriel and C. Marco, “Computational optimal transport with applications to data sciences,” Foundations and trends® in machine learning, vol. 11, no. 5-6, pp. 355–607, 2019.

[35] D. Alvarez-Melis and N. Fusi, “Geometric dataset distances via optimal transport,” Advances in Neural Information Processing Systems, vol. 33, pp. 21 428–21 439, 2020.

[36] H. Chen, A. Mehra, J. Kimball, and R. A. Rossi, “Measuring timeseries dataset similarity using wasserstein distance,” arXiv preprint arXiv:2507.22189, 2025.

[37] F. Painblanc, L. Chapel, N. Courty, C. Friguet, C. Pelletier, and R. Tavenard, “Match-and-deform: Time series domain adaptation through optimal transport and temporal alignment,” in Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer, 2023, pp. 341–356.

[38] H. Sakoe and S. Chiba, “Dynamic programming algorithm optimization for spoken word recognition,” IEEE transactions on acoustics, speech, and signal processing, vol. 26, no. 1, pp. 43–49, 1978.

[39] M. Cuturi and M. Blondel, “Soft-dtw: a differentiable loss function for time-series,” in International conference on machine learning. PMLR, 2017, pp. 894–903.

[40] D. S. Hirschberg, “Algorithms for the longest common subsequence problem,” Journal of the ACM (JACM), vol. 24, no. 4, pp. 664–675, 1977.

[41] M. Vlachos, G. Kollios, and D. Gunopulos, “Discovering similar multidimensional trajectories,” in Proceedings 18th international conference on data engineering. IEEE, 2002, pp. 673–684.

[42] C. Faloutsos, M. Ranganathan, and Y. Manolopoulos, “Fast subsequence matching in time-series databases,” ACM Sigmod Record, vol. 23, no. 2, pp. 419–429, 1994.

[43] H. Ding, G. Trajcevski, P. Scheuermann, X. Wang, and E. J. Keogh, “Querying and mining of time series data: Experimental comparison of representations and distance measures.” Proc. VLDB Endow., vol. 1, no. 2, pp. 1542–1552, 2008.

[44] F. Petitjean, A. Ketterlin, and P. Gançarski, “A global averaging method for dynamic time warping, with applications to clustering,” Pattern recognition, vol. 44, no. 3, pp. 678–693, 2011.

[45] K. Pearson, “Liii. on lines and planes of closest fit to systems of points in space,” The London, Edinburgh, and Dublin philosophical magazine and journal of science, vol. 2, no. 11, pp. 559–572, 1901.

[46] J. E. d’Hondt, O. Papapetrou, and J. Paparrizos, “Beyond the dimensions: A structured evaluation of multivariate time series distance measures,” in 2024 IEEE 40th International Conference on Data Engineering Workshops (ICDEW). IEEE, 2024, pp. 107–112.

[47] K. Rasul, A. Ashok, A. R. Williams, H. Ghonia, R. Bhagwatkar, A. Khorasani, M. J. D. Bayazi, G. Adamopoulos, R. Riachi, N. Hassen et al., “Lag-llama: Towards foundation models for probabilistic time series forecasting,” arXiv preprint arXiv:2310.08278, 2023.

[48] X. Shi, S. Wang, Y. Nie, D. Li, Z. Ye, Q. Wen, and M. Jin, “Time-moe: Billion-scale time series foundation models with mixture of experts,” in International conference on learning representations, vol. 2025, 2025, pp. 34 635–34 667.

[49] M. Goswami, K. Szafer, A. Choudhry, Y. Cai, S. Li, and A. Dubrawski, “Moment: A family of open time-series foundation models,” arXiv preprint arXiv:2402.03885, 2024.

[50] A. Alexandrov, K. Benidis, M. Bohlke-Schneider, V. Flunkert, J. Gasthaus, T. Januschowski, D. C. Maddix, S. Rangapuram, D. Salinas, J. Schulz et al., “Gluonts: Probabilistic and neural time series modeling in python,” Journal ofMachine Learning Research, vol. 21, no. 116, pp. 1–6, 2020.