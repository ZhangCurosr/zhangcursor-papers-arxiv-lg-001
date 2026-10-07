# Beyond Marginal Monitoring: Distributed Joint-Distribution Testing for Data Concept Drift in Large Scale E-Commerce Operations

C¸ a˘gda¸s Pullu<sup>1\*</sup>, Mahmut Emir Arslan<sup>1</sup>, Bu˘gra Balka¸c<sup>1</sup>, Aylin Ondersev Balta <sup>¨</sup> <sup>1</sup>, Cihangir Celal Palacı<sup>1</sup>, Fikri Cem Yılmaz<sup>1</sup>, Altan Cakir<sup>2</sup>

<sup>1\*</sup>Data & Analytics, Trendyol Group, Istanbul, 34485, Istanbul, T¨urkiye. <sup>2</sup>Department of Data Science and Analytics, Istanbul Technical University, Maslak, Istanbul, 34469, Istanbul, T¨urkiye.

\*Corresponding author(s). E-mail(s): cagdas.pullu@trendyol.com; Contributing authors: emir.arslan@trendyol.com; bugra.balkac@trendyol.com; aylin.ondersev@trendyol.com; cihangir.palaci@trendyol.com; fikricem.yilmaz@trendyol.com; cakirmua@itu.edu.tr;

## Abstract

Concept drift threatens production machine learning, yet the empirical behavior of multivariate two-sample drift detectors at scale remains under-characterized. Existing benchmarks rarely address the hundreds of millions of rows and highcardinality features typical of industrial-operational datasets. We evaluate five multi-column two-sample tests (marginal, projection-based, and kernel embedding methods) across three complementary environments: the Harvard Dataverse, a validated Failing Loudly reproduction (mean absolute error ∈ [0.030, 0.053]), and a novel synthetic-injection benchmark on the 137.5-million-row Trendyol collection-ranking feature table. Testing four drift types across two severity– scope regimes, we demonstrate that distributed Maximum Mean Discrepancy with Random Fourier Features on Apache Spark scales robustly. Averaged over the four drift types in the strong regime and under a calibrated threshold, it achieves a Pearson correlation of r = 0.940 with expected drift magnitude, an 80.4% true positive rate, and a 3.2% false positive rate. Conversely, the per-dimension Kolmogorov–Smirnov test failed due to statistic saturation from ID-like columns under asymmetric sampling, establishing a critical constraint for large-scale sampling design. At weak configurations (realized-flip fractions

$\leq 0 . 5 7 \% )$ , detectors struggled to reliably discriminate, highlighting the need for future intensity-grid power analyses to distinguish fundamental sensitivity bounds from scalable threshold shifts.

Keywords: Concept drift detection, Two-sample testing, Maximum Mean Discrepancy, Random Fourier Features, Production data science, Distributed computing, Apache Spark

## 1 Introduction

The integration of machine learning models into live products means that the data encountered in production may drift away from the training distribution. This dataset shift erodes prediction accuracy and, in the absence of continuous human oversight, progresses silently [1]. Where label-oriented evaluation feedback arrives slowly or not at all—as with sparse post-order return notifications or moderator approvals—drift is detected with two-sample tests on the data available at deployment: the features X and, when an outcome or prediction channel exists, the label variable Y produced by (or emitted alongside) the model. These tests are conventionally termed unsupervised, yet what the test observes—X alone or the joint vector $( X , Y )$ must be delineated carefully. At Trendyol<sup>1</sup>, a regional leader in e-commerce, the production table lacks a genuine outcome column, so we construct a synthetic benchmark target $Y _ { \mathrm { s y n t h } }$ from live-variance covariates (Section 5.3); it is neither an observed outcome nor a model prediction, and the results must be interpreted with this distinction in mind.

Two research fields have addressed this problem independently. The stream-based concept drift community formulates regimes in which $P ( Y \mid X )$ changes while P(X) stays stationary [2, 3], and the multivariate two-sample testing community measures the detection accuracy of per-dimension and kernel tests under controlled covariate shifts [4, 5]. Both rely on synthetic protocols, and the behavior of these tests at production scale—across millions of rows and wide tables with hundreds of columns—remains unresolved. D3Bench [6], the framework of Cerqueira et al. [7], and the controlled comparison of unsupervised detectors by Hinder et al. [8] approach this gap but remain limited to synthetic or small-scale scenarios, while learner-based approaches [9] and conformal calibration proposals [10] address production settings without labeled evaluation feedback from a diferent angle.

We validate two-sample tests across three complementary layers. Batch and slidingscan protocols on the Harvard Dataverse streams test whether drift is detected and localized; a numerical reproduction of the Failing Loudly benchmark verifies that our implementations match the published results; and a synthetic concept drift injection into the ∼137.5-million-row Trendyol collection-ranking feature table, which feeds Trendyol’s live recommendation system, tests behavior at production scale with eight scenarios (four drift types × two severity–scope regimes) and five multi-column tests on an 11-dimensional benchmark vector. In every layer the tests compare the joint vector (X, Y ), with the generator’s class label in the Harvard Dataverse streams and Y<sub>synth</sub> in the Trendyol benchmark. They are unsupervised in that they need no labeled evaluation feedback, but they are not X-only: even when the injection leaves X unchanged row by row, they can capture the change in the Y marginal (Sections 5.3 and 6.3).

Our primary findings are four. (i) Distributed Spark Maximum Mean Discrepancy with Random Fourier Features (RFF-MMD) is the most consistent test at production scale: in the strong regime and under a null-calibrated threshold, it reaches, averaged over the four drift types, a Pearson correlation of $r ~ = ~ 0 . 9 4 0$ with the per-bucket expected drift magnitude, a true positive rate (TPR) of 80.4%, and a false positive rate (FPR) of 3.2% (Section 6.3). Spark Sliced Wasserstein Distance (SWD) tracks drift comparably $( r = 0 . 8 1 9 )$ but at an FPR of about 10% under its fixed threshold. (ii) Bonferroni-aggregated per-dimension Kolmogorov–Smirnov (KS) fails not through the classical n → ∞ p-value collapse but through statistic saturation by ID-like columns under the asymmetric reference/current sampling design (r = −0.085, FPR = TPR = 100%; Section 7). (iii) At the weak configuration (realized-flip fractions of 0.29–0.57%), neither RFF-MMD nor SWD discriminates reliably, which marks an empirical lowsignal failure point for the present setting. (iv) A source-table pre-check and a four-part sanity-check protocol (Appendix B.2) form a recommended workflow for production drift benchmarks; without them, benchmark results risk measuring incidental dataquality artifacts rather than the intended signal.

Section 2 reviews related work, Sections 3–5 present the methodology and experimental design, Section 6 reports the results, Section 7 discusses them, and Section 8 concludes.

## 2 Related Work

Gama et al. [2] give the canonical taxonomy of distributional shift and the four temporal drift profiles (sudden, gradual, incremental, recurring; Figure 1), with detection lineages including error-monitoring [11], adaptive windows [12], and concepthistory/meta-learning for recurring shifts [13]. The Failing Loudly benchmark of Rabanser et al. [5] evaluates unsupervised shift detection on image test sets (MNIST, CIFAR-10) by combining dimensionality reduction with two-sample testing: multiple univariate Kolmogorov–Smirnov tests aggregated by the Bonferroni correction (on raw features and on sparse random projections, SRP), kernel Maximum Mean Discrepancy with a radial basis function (RBF) kernel (MMD-RBF), which compares the mean embeddings of the two distributions in a reproducing kernel Hilbert space (RKHS) and, under a characteristic kernel, responds to any diference in the full joint distribution [4], a chi-square test on hard predicted labels, and a binomial test on a domain classifier; we reproduce the detection accuracies of its KS and MMD variants as validation (Section 6.2). The univariate marginal battery used in our Harvard Dataverse layer (chi-square, Jensen–Shannon, Kullback–Leibler, Population Stability Index, Kolmogorov–Smirnov, Wasserstein; Table 1) is our own addition. The Harvard Dataverse concept-drift streams of L´opez Lobo [3] provide five stream families (SEA, SINE, STAGGER, MIXED, RT) with stationary $P ( X )$ by construction, isolating $P ( Y \mid X )$ shifts and providing a controlled testbed for distinguishing multi-column from univariate marginal tests (Section 6.1). Other real-world benchmarks include the Insects benchmark [14] (∼4.7M rows, physical sensor data). We use the Harvard Dataverse and Failing Loudly benchmarks as validation layers before the production-scale synthetic drift injection on the Trendyol collection-ranking feature table (Sections 5.3, 6.3). The distributed statistical computations are executed on Apache Spark [15] and its $\mathrm { S Q L } / \mathrm { I }$ DataFrame layer [16]; cluster configuration details in Section 4.2. Table 1 summarizes the two-sample tests evaluated across the three layers along with their families, decision rules, and complexity properties.

![](images/20fbe59fd0132eeb029b34762a25781cc736e22351d46ee1f8b81d87891cef11.jpg)  
(a) Sudden drift

![](images/6ca370aee8d6a7e68787c4f87a45bd687d942b583ef169f65ddc7af24931f121.jpg)  
(b) Gradual drift

![](images/6880a1b31f4c691cc5ac2bc4a903e9df6688130e88cbdbe779b19857b1838e37.jpg)  
(c) Incremental drift

![](images/debe9c5a140efc87edc33c1490a36a28ad3ed0fff425879c6dd186020ab0d719.jpg)  
(d) Recurring drift  
Fig. 1 Temporal concept-drift profiles, following the taxonomy of Gama et al. [2]: (a) sudden drift replaces the old concept at once; (b) gradual drift alternates between the old and new concepts until the new one prevails; (c) incremental drift moves through intermediate states; (d) recurring drift makes previously seen concepts reappear. The dashed line marks the drift onset. Terminology varies across surveys; for example, Hinder et al. [8] describe incremental drift as sampling from both concepts with varying probabilities.

Table 1 Overview of two-sample tests evaluated in this study. Family classification follows Section 3.1 (marginal per-feature; marginal per-feature aggregation on the stacked vector; projection-based; multivariate embedding/kernel). Vector column indicates the input to which the test is applied. Complexity is per-position at reference size $n _ { R }$ and current-window size n . Parameter values are those of the Failing Loudly reproduction; the Trendyol scan settings are given in Section 6.3.1.
<table><tr><td>Test</td><td>Family</td><td>Vector</td><td>Decision Rule / Calibration</td><td>Complexity</td></tr><tr><td>Univariate  $\overline { { \times 6 } } ^ { 1 }$ </td><td>Marginal</td><td> $X _ { k }$ </td><td>Fixed threshold or</td><td>O(n) per column</td></tr><tr><td>marginal Per-dim KS</td><td>per-feature Marginal</td><td>(X, Y) stack</td><td>asymptotic p Bonferroni:</td><td>O(nd log n)</td></tr><tr><td>Projected KS (SRP+KS)</td><td>aggregation Projection-based</td><td> $( X , Y ) S ,$ </td><td>mink pk &lt; α0/d Bonferroni on projected dims</td><td> $O ( n d + n \log n )$ </td></tr><tr><td>MMD-RBF</td><td>Multivariate</td><td>S∈ Rd×32  $( X , Y )$  stack</td><td>Permutation</td><td> $O ( n ^ { 2 } d ) ; { \mathrm { k e r n e l } }$ </td></tr><tr><td>RFF-MMD</td><td>kernel Multivariate</td><td>(X, Y) stack</td><td>(B = 1,000) Permutation</td><td>matrix  $O ( n D d ) ; D = 5 0 0$ </td></tr><tr><td>Sliced Wasserstein</td><td>kernel Projection-based</td><td>(X, Y) projections</td><td>(B = 200) Fixed τSw = 0.05</td><td> $O ( n P \log n ) ;$   $P = 1 0 0$ </td></tr></table>

<sup>1</sup>Chi-square, Jensen–Shannon, Kullback–Leibler, PSI, KS, and univariate Wasserstein-1, applied per feature.

## 3 Problem Formulation and Two-Sample Testing Methodology

## 3.1 Two-Sample Detection and Test Families

Let $\mathcal { D } _ { R } = \{ z _ { i } \} _ { i = 1 } ^ { n _ { R } }$ and $\mathcal { D } _ { C } = \{ z _ { j } \} _ { j = 1 } ^ { n _ { C } }$ be a reference and a current sample drawn from distributions $P _ { R }$ and $P _ { C }$ , where $\overset { \vartriangle } { \boldsymbol { z } } = ( \boldsymbol { X } , \boldsymbol { Y } )$ for multi-column tests and $z = X _ { k }$ for univariate marginal tests. Drift detection evaluates whether the two samples originate from identical distributions via the hypothesis:

$$
H _ { 0 } : P _ { R } = P _ { C } \qquad \mathrm { v s . } \qquad H _ { 1 } : P _ { R } \neq P _ { C } .\tag{1}
$$

A batch protocol evaluates this via a test statistic $s = T ( \mathcal { D } _ { R } , \mathcal { D } _ { C } )$ and, where a null distribution is available, a p-value p. The decision is $\psi = \mathbb { 1 } [ p < \alpha _ { 0 } ]$ for tests calibrated against a null distribution (permutation or asymptotic, with Bonferroni correction where applicable), and $\psi = 1 [ s > \tau ]$ for statistics compared against a fixed or nullcalibrated threshold τ (Table 1). Under $H _ { 0 }$ the pooled sample is exchangeable, which is the assumption underlying the permutation nulls of Section 4.1. We employ two structural test families to evaluate this hypothesis. Univariate marginal tests independently compare the marginals $P _ { R } ( X _ { k } )$ and $P _ { C } ( X _ { k } )$ for each feature dimension $k ,$ remaining strictly blind to changes in correlation structure. Multi-column tests operate on the joint stacked vector, which in our operational framing incorporates both the features and the label as $( X , Y )$ ; the label is model-emitted in deployment, the generator class in the Harvard Dataverse streams, and $Y _ { \mathrm { s y n t h } }$ in the Trendyol benchmark. The multi-column tests span three mechanistic sub-families: (1) marginal per-feature aggregations $( \mathrm { e . g . }$ , per-dimension KS with Bonferroni), (2) projection-based tests (sparse random projection KS, Sliced Wasserstein Distance), and (3) genuinely multivariate kernel embeddings (MMD-RBF, RFF-MMD). Table A1 (Appendix A) summarizes the notation used throughout the paper.

## 3.2 Sliding-Window Changepoint Detection

Because production drift is a continuous process rather than a single-threshold event, batch detection is insuficient for temporal localization. We extend the formulation into a sliding-window framework by partitioning the stream-ordered current dataset into N contiguous position buckets:

$$
\mathcal D _ { C } = \bigcup _ { b = 0 } ^ { N - 1 } \mathcal B _ { b } , \qquad \mathcal B _ { b } = \{ z _ { i } : \lfloor b n _ { C } / N \rfloor \leq i < \lfloor ( b + 1 ) n _ { C } / N \rfloor \} .\tag{2}
$$

At each normalized position $b / N \in [ 0 , 1 )$ , the test yields a statistic $s _ { b } = T ( \mathcal { D } _ { R } , \mathcal { B } _ { b } )$ forming a trajectory matrix S across the stream. We implement two operational variants of this framework, corresponding to the sliding and fixed reference-window types distinguished by Hinder et al. [8]. The moving-reference variant slides both windows synchronously to compare the immediate current window against the near past (applied to the Harvard Dataverse): at stream index t it compares the reference window $\{ z _ { i } : t - W \leq i < t \}$ with the current window $\{ z _ { i } : t \leq i < t + W \}$ , advancing t by a stride $\Delta$ . Conversely, the fixed-reference variant (applied to the Trendyol benchmark) compares each stream-ordered current bucket $B _ { b }$ against a static, predrift reference set $\mathcal { D } _ { R }$ . The latter isolates trajectory changes strictly to the current stream and substantially reduces distributed compute costs.

## 3.3 Ground-Truth Aggregation and Multiple Testing

To evaluate trajectory quality beyond binary True Positive/False Positive rates, synthetic protocols allow us to aggregate ground-truth metrics per bucket. By computing the expected drift magnitude $\left( \hat m _ { b } \right)$ and mean drift intensity $\left( \hat { \alpha } _ { b } \right)$ for each $B _ { b }$ , we can evaluate the continuous correlation between a test’s statistic trajectory and the true underlying drift severity, independent of threshold calibration.

Calibration and multiple testing require careful handling in this regime. Permutation-based tests (MMD-RBF, RFF-MMD) natively utilize $\alpha _ { 0 } ~ = ~ 0 . 0 5$ Bonferroni-aggregated tests apply a corrected threshold $\alpha _ { 0 } / d$ to maintain the familywise error rate (FWER); however, statistical power degrades rapidly as dimensionality d increases. Furthermore, sliding-scan protocols inherently introduce cross-position dependence—either through overlapping current windows or a shared fixed reference— complicating the operational meaning of nominal $\alpha _ { 0 }$ . Consequently, trajectory results in our continuous protocols are reported descriptively (hit rate, detection latency, stationary FPR) rather than relying on independence assumptions that do not hold in sequential scans.

## 4 Algorithmic Framework and Distributed Big Data Execution

This section details the algorithmic implementation of the two-sample tests introduced in Section 3.1 and outlines their execution architecture for production-scale data. To facilitate direct comparison across families, thresholding strategies, and computational complexities, the algorithmic configurations are synthesized in Table 2.

## 4.1 Algorithmic Implementations and Calibration

We implement tests across the three established sub-families: marginal aggregation, projection-based, and genuinely multivariate kernel embeddings. The theoretical scaling constraints of these algorithms fundamentally dictate their calibration strategies and deployment viability. For instance, the exact Maximum Mean Discrepancy with an RBF kernel (MMD-RBF) scales as $O ( n ^ { 2 } d )$ due to kernel-matrix materialization, capping its practical application to a few thousand subsampled rows. To achieve linear time $O ( n \cdot D \cdot d )$ scaling suitable for the ∼137.5M-row Trendyol stream, we utilize Random Fourier Features (RFF-MMD) [17], projecting inputs into a D-dimensional space $( D = 5 0 0$ in the Failing Loudly runs and $D = 2 0 0$ in the Trendyol scans; Section 6.3.1).

Decision calibration closely follows the test constraints. Non-parametric permutation nulls are utilized for the kernel methods (MMD-RBF, RFF-MMD) to maintain strict size control under exchangeability assumptions. Conversely, Bonferroniaggregated tests adjust the threshold by the efective dimensions $( \alpha _ { 0 } / d \ \mathrm { o r } \ \alpha _ { 0 } / K )$ ， and Sliced Wasserstein Distance (SWD) relies on a fixed distance threshold $( \tau _ { \mathrm { S W } } =$ 0.05) without a permutation null, acting as a distance-based signal rather than a size-controlled test.

Table 2 Summary of Evaluated Two-Sample Tests and Execution Architecture
<table><tr><td>Test Name</td><td>Sub- Family</td><td>Core Statistic / Mechanism</td><td>Threshold / Calibration</td><td>Execution Paradigm</td></tr><tr><td>Univariate (6 variants)</td><td>Marginal</td><td> $\mathrm { M a r g i n a l } \ P _ { R } ( X _ { k } ) \ \mathrm { v s }$   $P _ { C } ( X _ { k } )$  per feature</td><td> $\mathrm { F i x e d } \ \alpha _ { 0 } = 0 . 0 5$  or lit. τ</td><td>Hybrid (Driver aggregated)</td></tr><tr><td>Per-dim KS</td><td>Marginal Agg.</td><td>maxk  $D _ { k }$  across d raw dimensions</td><td>Bonferroni  $( \alpha _ { 0 } / d )$ </td><td>Driver-side (Subsampled)</td></tr><tr><td>Projected KS (SRP)</td><td>Projection</td><td>max KS across K = 32 sparse random projections</td><td>Bonferroni (α₀/K)</td><td>Driver-side (Subsampled)</td></tr><tr><td>Sliced Wasserstein Distance</td><td>Projection</td><td>Avg. Wasserstein-1 over 100 random unit directions</td><td>Fixed distance (τ = 0.05)</td><td>Spark Distributed</td></tr><tr><td>MMD-RBF</td><td>Kernel Embed.</td><td> $\widehat { \mathrm { M M D } } ^ { 2 }$  via Gaussian RBF</td><td>Permutation</td><td>Driver-side</td></tr><tr><td>RFF-MMD</td><td>Kernel Embed.</td><td> $( O ( n ^ { 2 } d ) )$   $\| \bar { \mathbf { h } } ( X _ { R } ) - \bar { \mathbf { h } } ( X _ { C } ) \| _ { 2 } ^ { 2 }$  mapping  $( D = 5 0 0 )$ </td><td>(B = 1000) Permutation  $( B = 2 0 0 )$ </td><td>(Subsampled) Spark Distributed</td></tr></table>

## 4.2 Spark Cluster Configuration and Execution Mapping

All distributed computations are executed on Google Cloud Dataproc Serverless (PySpark 3.5), utilizing up to 16 executors (16 vCPU, 96 000 MiB Java virtual machine heap per executor). Because the statistical tests exhibit vastly diferent memory properties, execution maps onto two distinct paths, as indicated in Table 2:

• Spark-Distributed Execution: Tests inherently scalable to O(N) (RFF-MMD and SWD) compute their core statistics via distributed transformations across the full reference and current buckets. RFF-MMD distributes the D-dimensional Fourier feature computation and mean embedding aggregation, while SWD distributes the random unit projections and quantile approximations. Execution time is ∼2.7–5.6 s per position.

• Driver-Side Execution: Tests bottlenecked by matrix materialization or lacking distributed native implementations (MMD-RBF, Projected KS, Per-dim KS) require the bucket to be collected to the driver via a limit query (driver sample size $n _ { \mathrm { d r v } } =$ 20,000; further subsampled to 3,000 for MMD-RBF). Null shufles and linear algebra operations are computed locally in-memory on the driver.

This architectural split separates tests whose cost scales with the full bucket from tests that operate on a driver-side subsample; Section 6.3.7 quantifies the resulting runtime gap.

## 5 Experimental Design and Benchmark Protocols

## 5.1 Dataset Inventory and Preparation

We evaluate the detection methods across three structural layers, each providing a distinct type of ground truth and validation claim (Table 3). The Harvard Dataverse streams [3] provide controlled conditional shifts (P(Y | X) changes, P(X) is stationary). The Failing Loudly image-shift benchmark [5] provides static, heavily-shifted covariate evaluation environments. Finally, the Trendyol collection-ranking feature table provides massive scale (∼137.5 million rows) and authentic user-behavior variance. The table is hosted in Google BigQuery, Google Cloud’s serverless, columnar data warehouse for SQL analytics over petabyte-scale datasets. Trendyol’s data originates from live A/B tests on a regional-leader e-commerce recommendation system; following the pre-check (Appendix B), the benchmark operates on an 11-dimensional vector of metadata scalars (identifier columns retained by design) and live similarity metrics.

## 5.2 Evaluation Protocols

To systematically assess the test families, we apply three distinct protocols, summarized in Table 4. The Batch protocol identifies the binary presence of drift, the Sliding-Scan protocol evaluates temporal localization and detection latency, and the Failing Loudly Replication validates our implementations against published literature baselines by mean absolute error (MAE). Figure 2 positions these protocols within the taxonomy of unsupervised drift detectors proposed by Gemaque et al. [18].

Table 3 Dataset inventory and feature dimensionality. Feature dimensionality is reported after one-hot encoding of categorical covariates.
<table><tr><td>Layer</td><td>Source / Family</td><td>Configs</td><td>Dim</td><td>Ground Truth</td></tr><tr><td>Harvard Dataverse</td><td>SEA, SINE, STAGGER, MIXED, Random Tree</td><td>20</td><td>2-9</td><td>3 temporal changepoints</td></tr><tr><td>Failing Loudly</td><td>MNIST, CIFAR-10 (10 shift types)</td><td>60</td><td>784  /  3072</td><td>Image perturbation labels</td></tr><tr><td>Trendyol benchmark</td><td>Collection-ranking feature table</td><td>8</td><td>11</td><td>Injected row-level drift (Eqs. 6–8)</td></tr></table>

![](images/836c57e83d5ab8442594ef7021847761c49f66706663c6d8a6daef6bcef92cab.jpg)  
Fig. 2 Evaluation protocols of this study within the taxonomy of unsupervised drift detectors of Gemaque et al. [18] (adapted from their Fig. 2, CC BY 4.0). Batch detectors test an accumulated window as a whole; sliding-window scans advance the detection window along the stream and difer in whether the reference window stays fixed or slides with it. The survey’s online detectors check every arriving instance, whereas our scans evaluate at a stride ∆ (Harvard Dataverse) or per position bucket (Trendyol benchmark). Partial-batch detection, which tests only a selected subset of each batch, is not evaluated here.

Table 4 Summary of experimental protocols and evaluation metrics.
<table><tr><td>Protocol</td><td>Data Structure &amp; Windows</td><td>Evaluated Tests</td><td>Primary Metrics</td></tr><tr><td>Batch (Harvard)</td><td>Ref: 1st half. Cur: 2nd half.</td><td>All 11 tests (Marginal + Joint)</td><td>Binary Recall (TP / TP+FN)</td></tr><tr><td>Sliding-Scan (Harvard)</td><td>W = 5000, stride ∆ = 1000. Moving reference [t − W, t − 1].</td><td>All 11 tests</td><td>Hit rate (ρ = 200), Latency, FPR</td></tr><tr><td>Sliding-Scan (Trendyol)</td><td>N = 50 position buckets. Fixed reference table (first 100,000 rows; 10,000 for driver-side tests). 8 scenarios.</td><td>MMD-RBF, Per-dim KS, SRP+KS, RFF-MMD, SWD</td><td>Pearson r vs. mb, TPR, FPR, Runtime</td></tr><tr><td>Failing Loudly</td><td>Static sizes N ∈ {10 . . . 10, 000}. 5 random splits, 3 fractions.</td><td>MMD-RBF, Per-dim KS, SRP+KS, RFF-MMD, SWD</td><td>Acc(N), MAE vs literature</td></tr></table>

## 5.3 Synthetic Concept Drift Injection on Production Data

Because the Trendyol collection-ranking feature table lacks stationary ground truth, we construct a controlled evaluation environment via a synthetic concept drift injection. We define two deterministic concepts based on live similarity metrics: Concept A (visual similarity, $P ( Y _ { A } = 1 )$ ≈ 0.22) and Concept B (textual similarity, $P ( Y _ { B } =$ $1 ) \approx 0 . 4 5 )$ . The injection overlays a synthetic label $y ^ { \prime }$ without altering the original row-wise features $( X _ { \mathrm { a f t e r } } = X _ { \mathrm { b e f o r e } } )$ , ensuring any detected signal is strictly a change in $P ( Y \mid X )$ .

Table 5 Trendyol Benchmark Scenario Configurations. The strong regime measures clean detection capabilities, while the weak regime establishes an empirical low-signal failure point.
<table><tr><td>Drift Type (Regime)</td><td>Severity (σ)</td><td>Scope  $( \phi )$ </td><td>Mean  $\bar { \alpha } _ { s }$ </td><td>FlipRate expected scenario</td></tr><tr><td>sudden_strong</td><td>0.75</td><td>0.60</td><td>0.400</td><td>10.3%</td></tr><tr><td>sudden_weak</td><td>0.25</td><td>0.10</td><td>0.400</td><td>0.57%</td></tr><tr><td>gradual_strong</td><td>0.75</td><td>0.60</td><td>0.325</td><td>8.4%</td></tr><tr><td>gradual_weak</td><td>0.25</td><td>0.10</td><td>0.325</td><td>0.47%</td></tr><tr><td>incremental_strong</td><td>0.75</td><td>0.60</td><td>0.325</td><td>8.4%</td></tr><tr><td>incremental_weak</td><td>0.25</td><td>0.10</td><td>0.325</td><td>0.47%</td></tr><tr><td>recurring_strong</td><td>0.75</td><td>0.60</td><td>0.200</td><td>5.2%</td></tr><tr><td>recurring-weak</td><td>0.25</td><td>0.10</td><td>0.200</td><td>0.29%</td></tr></table>

The transition is controlled by three parameters: the temporal blend weight $\alpha \in$ $[ 0 , 1 ]$ , the scope fraction $\phi$ (rows exposed), and the severity σ (mixing strength). The active target probability per row is:

$$
p _ { \mathrm { a c t i v e } } ( \mathrm { r o w } ) = p _ { A } ( \mathrm { r o w } ) + \alpha \cdot \left[ p _ { B } ^ { \mathrm { e f f e c t i v e } } ( \mathrm { r o w } ) - p _ { A } ( \mathrm { r o w } ) \right]\tag{3}
$$

where $p _ { B } ^ { \mathrm { e f f e c t i v e } } ( \mathrm { r o w } ) = p _ { A } ( \mathrm { r o w } ) + \sigma \cdot [ p _ { B } ( \mathrm { r o w } ) - p _ { A } ( \mathrm { r o w } ) ]$ if the row is in scope, and $p _ { A } ( \mathrm { r o w } )$ otherwise. The injected label is drawn via $y ^ { \prime } ( \mathrm { r o w } ) = \mathbb { 1 } [ U ( 0 , 1 ) < p _ { \mathrm { a c t i v e } } ( \mathrm { r o w } ) ]$ Throughout, $y _ { A } ( \mathrm { r o w } ) = Y _ { A } ( \mathrm { r o w } )$ denotes the pre-drift label and $y ^ { \prime } ( \mathrm { r o w } )$ the injected label; $Y _ { \mathrm { s y n t h } }$ denotes the corresponding label variable in the joint vector $( X , Y _ { \mathrm { s y n t h } } )$

We formulate eight production scenarios (four drift types × two regimes) to stress-test the algorithms (Table 5). The expected realized-flip fraction for a scenario (FlipRate) and the fully-active-bucket expected magnitude $( m _ { \mathrm { b u c k e t } } ^ { \mathrm { a c t i v e } } )$ quantify the drift intensity:

$$
\mathrm { F l i p R a t e } _ { \mathrm { s c e n a r i o } } ^ { \mathrm { e x p e c t e d } } \approx \bar { \alpha } _ { s } \cdot \sigma \cdot \phi \cdot \delta\tag{4}
$$

$$
m _ { \mathrm { b u c k e t } } ^ { \mathrm { a c t i v e } } \approx \sigma \cdot \phi \cdot \delta\tag{5}
$$

where $\bar { \alpha } _ { s }$ is the mean temporal coeficient and $\delta \ \approx \ 0 . 5 7 2 4$ is the Concept $\mathrm { A } / \mathrm { B }$ disagreement rate.

To correlate a test’s statistic trajectory with the true drift severity independently of threshold calibration, the injection framework also emits three row-level groundtruth variables: the eligibility indicator e, the expected drift magnitude m, and the realized-flip indicator f (the corresponding stored columns are listed in Table B2):

$$
e ( \mathrm { r o w } ) = 1 [ \mathrm { i n \_ s c o p e } \land Y _ { A } ( \mathrm { r o w } ) \neq Y _ { B } ( \mathrm { r o w } ) ] ,\tag{6}
$$

$$
m ( \mathrm { r o w } ) = \alpha ( \mathrm { r o w } ) \cdot \sigma \cdot e ( \mathrm { r o w } ) ,\tag{7}
$$

$$
f ( \mathrm { r o w } ) = \mathbb { 1 } [ y ^ { \prime } ( \mathrm { r o w } ) \neq y _ { A } ( \mathrm { r o w } ) ] .\tag{8}
$$

The per-bucket means $\bar { m } _ { b }$ and $\bar { f } _ { b }$ of m and f over the rows of $B _ { b }$ serve as the primary and robustness correlation targets, respectively; $\bar { m } _ { b }$ is preferred because it precisely encodes the probability-distribution displacement without the variance introduced by per-row Bernoulli sampling noise.

## 6 Results

## 6.1 Harvard Dataverse Concept Drift Streams

This section evaluates the batch protocol (defined in Section 5.2) across the 20 Harvard Dataverse streams to determine if stream-level shifts are detectable; Section 6.1.1 reports the corresponding sliding-scan results. Because these streams are engineered with a stationary marginal $P ( X )$ while only the conditional $P ( Y \mid X )$ shifts, they provide a rigorous diagnostic environment for distinguishing joint-distribution sensitivity from purely marginal sensitivity.

Because recall is either 0% or 100% for every test–family pair, Table 6 summarizes the batch outcomes by test category. The results isolate three critical behavioral bifurcations across the test families:

1. Robustness of Genuine Multivariate Tests: Projection-based and kernel embedding tests (RFF-MMD, SWD, MMD-RBF, Projected KS) achieve 100% recall on all five stream families. This empirically validates their theoretical capacity to detect shifts in the joint distribution $( X , Y )$ even when individual feature marginals remain perfectly stationary.

2. Structural Failure of Per-dim KS: Despite operating on the stacked vector (features ∪ class), Bonferroni-aggregated Per-dim KS completely fails (0% recall) on the SINE and MIXED families. This highlights a severe operational vulnerability: the test’s inherent blindness to pure dependence changes, entangled with Bonferroni conservativeness, eliminates its detection power in specific conditional-shift regimes.

3. Expected Failure of Univariate Marginals: Categorical marginal tests (chisquare, JS-div, KL-div, PSI) exhibit 0% recall on four of the five families, precisely as analytically expected under stationary $P ( X )$ . The sole exception is the Random Tree (RT) family, where an artifact in the stream generator rescales the one-hot encoded class column at changepoints, inadvertently creating a detectable marginal signal.

Table 6 Aggregated Batch Recall on Harvard Dataverse Streams. The bifurcation in recall demonstrates the structural failure of marginal tests under stationary $P ( X )$ environments, contrasted against the robustness of joint multivariate embeddings.
<table><tr><td>Test Category</td><td>SEA</td><td>SINE</td><td>STAGGER</td><td>MIXED</td><td>Random Tree (RT)</td></tr><tr><td>Multivariate (Kernel &amp; Proj.)</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td><td>100%</td></tr><tr><td>Per-dim KS (Bonferroni)</td><td>100%</td><td>0%</td><td>100%</td><td>0%</td><td>100%</td></tr><tr><td>Univariate Marginal</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td><td>100%1</td></tr></table>

<sup>1</sup>Recall artifact resulting from a generator-induced class-marginal rescaling at changepoints.

![](images/bd836f186b53b652f069859a9e0fe6de216b6ef160dc7dc0e4e27998cfc83ba6.jpg)

![](images/abdb106b34df294e1b9c0e59fee94c434473ed2825da2c44ddd2257fec3fe822.jpg)

![](images/334939b30918a9a5b510060c0b5850d063e2efae276728d5772d95e6eea2a6c8.jpg)

![](images/606ce00b41224740e91465dc9118208fbf29b3d8ea7533610c9f782e1bec9197.jpg)  
Fig. 3 Sliding-scan p-value trajectories on Harvard Dataverse streams. Each panel aggregates streams within one family and plots the median p-value at each scan position for MMD-RBF, projected KS, and per-dim KS. Shaded bands are interquartile ranges over streams. Red dashed lines mark ground-truth changepoints; the horizontal line is $\alpha _ { 0 } = 0 . 0 5$ on a symmetric-log axis. STAG-GER is omitted because its whole-vector detection uses distance-based statistics without p-values after one-hot encoding of the categorical features.

## 6.1.1 Sliding-scan results

Figure 3 shows the sliding-scan trajectories $( W \ = \ 5 0 0 0 , \ \Delta \ = \ 1 0 0 0 )$ of the three p-value-producing tests on four stream families; STAGGER is omitted because the distance-based statistics used on its one-hot encoded features yield no p-values. MMD-RBF gives the cleanest structure: in the SEA, MIXED, and RT families its p-value falls below $\alpha _ { 0 } = 0 . 0 5$ around every changepoint and stays high in stationary regions, capturing all changepoints within the $\rho = 2 0 0$ tolerance at a typical latency of 1– 2 sample steps, small relative to the $W = 5 0 0 0$ window. Per-dim KS reproduces its batch failure: on SINE its p-value stays near 1 throughout the scan, and it captures no changepoint in SINE or MIXED. Projected KS lies in between. Projecting to $d = 3 2$ dimensions eases the conservativeness of the Bonferroni correction, but information mixing weakens the signal at some changepoints, and it succeeds only partially, in two families.

## 6.2 Failing Loudly Reproduction and Image-Shift Results

This section reports results of the Failing Loudly reproduction protocol defined in Section 5.2. Three tests (per-dim KS, projected KS with SRP, MMD-RBF) are validated against Failing Loudly Table 1(a) rows and characterized across the ten shift types, grouped into five families, at perturbation fractions {10%, 50%, 100%}; the two extension methods (RFF-MMD and SWD) are characterized under the same protocol in Appendix D.

## 6.2.1 Reproduction accuracy

The reproduced MMD-RBF detection accuracy at $N = 1 0 0 0$ deviates from the paper value by 0.007, and the reproduced per-dim KS deviation at $N = 1 0 0 0 0$ is 0.007. Mean absolute error (MAE) across the three methods over sample sizes lies in the range [0.030, 0.053]; this demonstrates that Rabanser et al.’s original measurements can be numerically reproduced. The largest single-cell deviations are for $\mathrm { S R P { + } K S }$ at $N = 1 0$ $\left( - 0 . 1 4 0 \right)$ and at $N = 1 0 0 0 \ ( - 0 . 1 3 7 )$ ; the former is consistent with the discreteness of the KS null distribution at $N = 1 0$ under Bonferroni over 32 projected dimensions, and the latter has been examined in supplementary analysis but a fully quantitative explanation remains open.

## 6.2.2 Shift-type by dataset breakdown

Figure 4 reports detection accuracy at $N = 1 0 0 0$ by shift type and dataset, revealing three patterns. Combined image and label shifts are the easiest: they alter both the covariate and the label distribution, and all three tests reach 100% on MNIST and about 60% on CIFAR-10. Adversarial shifts are the hardest: on CIFAR-10 both KS variants reach 0% and MMD-RBF only 7%, because adversarial perturbations target the decision boundary while leaving raw pixel marginals nearly unchanged. Class knock-out shifts fall in between. They change $P ( Y )$ without directly modifying $P ( X )$ , yet removing classes alters the observed class-conditional covariate distribution, which MMD-RBF partly picks up (60% on MNIST, 27% on CIFAR-10) while the KS variants reach at most 27% and 13%, respectively.

![](images/a51a2198fb05f4bd784ab7f73d0c76d2a974242af335332d908f9a7474876013.jpg)  
Fig. 4 Detection accuracy by test and shift family at $N = 1 0 0 0$ on the Failing Loudly benchmark, disaggregated by dataset. Rows are the three reproduced baselines (per-dimension KS on raw pixels, projected KS with SRP, MMD-RBF). Columns are the five shift families of Failing Loudly. Each cell aggregates all comparisons of one family over the three perturbation fractions and five splits: 45 for the Gaussian and image families (three intensities each), 30 for the combined family (two variants), and 15 for the adversarial and class-knockout families.

## 6.2.3 p-value curves for Gaussian noise shifts

Figure 5 shows the three methods’ p-values vs. sample size for the Gaussian noise shift class at afected fractions {0.10, 0.50, 1.00} (the reproduction analog of Failing Loudly Figure $_ { 3 ; }$ principal component analysis and extension methods omitted per Section 5.2); lines are medians pooled over both datasets, shift intensities, and splits. Detection depends strongly on the afected fraction. Per-dim KS falls below $\alpha _ { 0 } = 0 . 0 5$ from $N = 5 0$ at 100% perturbation and from $N = 2 0 0$ at 50%, but at 10% only at $N = 1 0 0 0 0$ . MMD-RBF, evaluated up to N = 1000, crosses $\alpha _ { 0 }$ only at 100% perturbation and $N = 1 0 0 0$ (median $p = 0 . 0 1 6 )$ ; at 10% its median stays near 0.4. Projected KS is the least sensitive: even at 100% its median remains above $\alpha _ { 0 }$ up to

Gaussian noise shift — Failing Loudly reproduction

$N = 1 0 0 0 \ ( p = 0 . 0 7 1 )$ and falls below it only at $N = 1 0 0 0 0$ , consistent with signal loss in the d = 32 projected space.

![](images/753b3621021baa72a96a1ec1b5dac7732107e755765dbf31448877d7bc0c12c0.jpg)  
Fig. 5 Reproduced p-value curves under the Gaussian-noise shift class of Rabanser et al. [5] on MNIST and CIFAR-10. Panels correspond to afected-fraction {0.10, 0.50, 1.00}. Lines are median $p \mathrm { - }$ values across 30 shift instances per sample size (3 intensities × 5 splits × 2 datasets). Shaded bands are the interquartile range. The horizontal reference line is $\alpha _ { 0 } = 0 . 0 5$ . This figure is the reproduction analog of Failing Loudly Figure 3.

## 6.2.4 Additional shift types: image, adversarial, class-knockout, and combined

Beyond the Gaussian noise shifts of Section 6.2.3, the Failing Loudly protocol evaluates four additional shift families (Figures 6–9), aggregated in the same way. Image transformation shifts (rotation, translation, zoom) behave like Gaussian noise: perdim KS and MMD-RBF both fall below $\alpha _ { 0 }$ from $N = 2 0 0$ at 100% and from $N = 5 0 0$ at 50% perturbation, while at 10% no test crosses $\alpha _ { 0 }$ at any evaluated sample size. Adversarial shifts are the hardest and show no consistent descent: at 10% no test crosses $\alpha _ { 0 } .$ , and at 50% and 100% the median p-values fluctuate with N instead of decreasing monotonically. MMD-RBF reaches the lowest values (median $p \approx 0 . 0 2$ at $N = 5 0 0 – 1 0 0 0$ under 100% perturbation), whereas per-dim KS stays above $\alpha _ { 0 }$ at every sample size and fraction. Class-knockout (label) shifts keep the per-dim KS p-value well above α<sub>0</sub> up to $N = 1 0 0 0$ at every fraction (median $p \geq 0 . 3 9 )$ ; it falls below $\alpha _ { 0 }$ only at $N = 1 0 0 0 0$ for 50% and 100% perturbation. MMD-RBF picks up the indirect covariate efect only at 100% perturbation, crossing $\alpha _ { 0 }$ from $N = 5 0 0$ (median $p = 0 . 0 1 7 )$ ; at 10% and 50% its median remains above 0.25. Combined image+label shifts produce the sharpest descent because covariate and label distributions shift jointly: even at 10% perturbation, per-dim KS and MMD-RBF fall below $\alpha _ { 0 }$ from $N = 5 0$ and projected KS from $N = 1 0 0$

Image transformation shift — Failing Loudly reproduction  
![](images/6ca64e2d9dbf81b32d25c6fe99d5270846480dc12aeb1137dc781909d1dcf611.jpg)  
Fig. 6 Failing Loudly reproduction: image-transformation shifts (rotation, translation, zoom at three intensities). Panels correspond to {10%, 50%, 100%} perturbed test data.

Adversarial shift — Failing Loudly reproduction  
![](images/3f02749bfe0c84b9981745e9f9c8ec55665dff250443c760c8122c0679409ca0.jpg)  
Fig. 7 Failing Loudly reproduction: adversarial shifts generated with the fast gradient sign method. Panels correspond to {10%, 50%, 100%} perturbed test data. Adversarial shifts are the hardest singleclass shift family at low perturbation fractions.

Label knock-out shift — Failing Loudly reproduction  
![](images/4b4a08da16e2da98909445fbe82bda540896b7af9ab0c1b9a5f6d813fd9bc92c.jpg)  
Fig. 8 Failing Loudly reproduction: class knock-out (label) shifts. Panels correspond to {10%, 50%, 100%} perturbed test data. Because knock-out shifts change P(Y) rather than P(X), MMD-based tests detect the shift through its indirect efect on the observed covariate distribution.

Combined image + label shift — Failing Loudly reproduction  
![](images/d0c526b7e87df899e47625a6641c36e7f8e3bab2ebcb208203be6d9ba81646ef.jpg)  
Fig. 9 Failing Loudly reproduction: combined image transformation + class knock-out shifts. Panels correspond to {10%, 50%, 100%} perturbed test data. Combined shifts show the sharpest median-p descent because covariate and label distributions shift jointly.

## 6.3 Real-World Production Data with Synthetic Concept Drift Injection

## 6.3.1 Setup and threshold calibration

The eight scenarios apply the fixed-reference sliding scan (Table 4) to the 11- dimensional benchmark vector of Section 5.1. With $s _ { 0 } ~ = ~ 0 . 6$ and 50 buckets of approximately equal size, bucket 29 straddles the drift onset and is the first bucket containing rows with $\alpha > 0 \left( \bar { \alpha } _ { 2 9 } \approx 0 . 0 9 \right.$ under the sudden schedule). For sudden, gradual, and incremental drift, buckets 0–28 therefore form the injection-null region $( \bar { m } _ { b } = 0 )$ and buckets 29–49 the injection-active region, so TPR is computed over 21 positions and FPR over 29 $( \mathrm { e . g . , T P R = 9 5 . 2 \% = 2 0 / 2 1 } )$ ; for recurring drift, the postonset region alternates between active and null blocks of five buckets. Both sides are capped positionally rather than sampled at random: the Spark tests compare each full bucket (≈2.75M rows) with the first 100,000 rows of a separate reference table, and the driver-side tests compare the first 20,000 rows of each bucket with the first 10,000 reference rows. To bound the per-position cost, the scans use lighter settings than the Failing Loudly runs of Table 1: RFF-MMD with $D = 2 0 0$ random features and no permutation null (decision by τ ), SWD with 25 projections and 100 quantiles, MMD-RBF with 200 permutations on 3,000-row subsamples, and projected KS with 64 dense Gaussian projections. All results come from a single injection run per scenario; we report Wilson 95% intervals for TPR and FPR and moving-block bootstrap intervals (block length 5, 2,000 resamples) for r. Because positions share one reference and are not independent (Section 3.3), the Wilson intervals are approximate; the block bootstrap accounts for short-range dependence between neighboring positions.

A binary decision requires a threshold, and choosing one by inspecting the trajectories would leak into the reported metrics. For RFF-MMD we therefore calibrate on null data only: the 29 pre-drift positions of the sudden–strong scenario give an empirical 99th percentile of $2 . 9 3 \times 1 0 ^ { - 3 }$ (mean $1 . 9 5 \times 1 0 ^ { - 4 }$ , std $7 . 1 6 \times 1 0 ^ { - 4 } )$ , which we round up to $\tau _ { \mathrm { R F F } } = 3 \times 1 0 ^ { - 3 }$ and apply unchanged to all eight scenarios. The pre-drift statistic is homogeneous across scenarios (means in $\left[ 1 . 7 1 , 2 . 9 3 \right] \times 1 0 ^ { - 4 }$ , standard deviations in $[ 6 . 5 , 7 . 9 ] \times 1 0 ^ { - 4 }$ , with sudden–strong inside both ranges), although matching means and spreads do not establish matching upper tails, which a full crossscenario tail check would require. Three caveats bound this choice. First, 29 samples that share one reference sample support an empirical high-quantile estimate, not a 1% size guarantee. Second, the quantile is dominated by the position-25 outlier (≈ 20× the pre-drift mean; Section 6.3.5), which we retain; recalibrating without it is left to future work. Third, the pre-drift positions of sudden–strong also enter the headline FPR, which is therefore partially in-sample, whereas its TPR positions lie outside the calibration draw. The remaining tests keep their implementation thresholds (SWD $\tau _ { \mathrm { S W } } = 0 . 0 5 ;$ per-dim KS Bonferroni $\alpha _ { 0 } / d )$ ; their calibration behavior is discussed in Sections 6.3.6 and 7.

## 6.3.2 Overall performance

Table 7 summarizes the five tests averaged over all 8 scenarios. The per-bucket ground truth is $\bar { m } _ { b }$ , the mean of m over the rows of bucket $b \left( \mathrm { E q . 7 } \right)$ ; we partition positions into injection-null ${ \mathcal { P } } _ { 0 } = \{ b : { \bar { m } } _ { b } = 0 \}$ and injection-active ${ \mathcal { P } } _ { 1 } = \{ b : { \bar { m } } _ { b } > 0 \}$ , chosen in place of a chronological pre/post-drift split so that recurring-drift return-to-A blocks (post-onset positions with $\bar { m } _ { b } = 0 )$ are correctly classified. FPR, TPR, ratio, and r are then defined on this partition per the Table 7 caption.

Two tests stand out. Spark RFF-MMD, under the calibrated threshold of Section 6.3.1, produces the most consistent discrimination: mean $r ~ = ~ 0 . 4 2 7$ , FPR $= 3 . 2 \%$ , ratio = 8.41 (injection-active statistic reaches ≈8× injection-null). Spark SWD produces a similar correlation $\left( r = 0 . 4 2 2 \right)$ but higher FPR (10.0%) under its fixed threshold $( \tau _ { \mathrm { S W } } = 0 . 0 5 )$ . These grand means, however, obscure a critical stratification across the severity–scope regime: both tests produce defensible strong-regime results but fail to discriminate in the weak regime (Section 6.3.3). The remaining three tests (per-dim KS, MMD-RBF, projected KS) exhibit systematic failures whose mechanisms are summarized in Section 6.3.6 and detailed in Section 7.

Table 7 Sliding-scan detection metrics on the Trendyol collection-ranking feature table averaged over all 8 test scenarios (4 drift types × 2 severity regimes). Grand means include the weak regime, where both RFF-MMD and SWD fail to discriminate; see Table 8 for the per-regime breakdown.
<table><tr><td>Test</td><td>TPR (%)</td><td>FPR (%)</td><td>Ratio</td><td>r</td><td>Time (s)</td></tr><tr><td>Per-dim KS</td><td>100.0</td><td>100.0</td><td>0.91</td><td>-0.085</td><td>0.07</td></tr><tr><td>MMD-RBF</td><td>100.0</td><td>98.4</td><td>1.99</td><td>-0.022</td><td>36.20</td></tr><tr><td>Projected KS</td><td>100.0</td><td>100.0</td><td>1.25</td><td> $+ 0 . 1 0 9$ </td><td>0.63</td></tr><tr><td>Spark RFF-MMD</td><td>40.2</td><td>3.2</td><td>8.41</td><td> $+ 0 . 4 2 7$ </td><td>2.68</td></tr><tr><td>Spark SWD</td><td>48.2</td><td>10.0</td><td>1.84</td><td>+0.422</td><td>5.50</td></tr></table>

TPR = rejection rate over injection-active positions $( \mathcal { P } _ { 1 } = \{ b : \bar { m } _ { b } > 0 \} )$ ; FPR = injection-relative rejection rate over injection-null positions $( \mathcal { P } _ { 0 } = \{ b : \bar { m } _ { b } = 0 \}$ ; this partition rather than a chronological pre/post-drift split is used because it correctly handles recurring drift). The injection-relative FPR is not the classical statistical Type-I error, since injection-null positions may still carry natural production-side covariate heterogeneity as documented in Section 6.3.5; Ratio = injection-active mean statistic divided by injection-null mean statistic; r = Pearson correlation between test statistic and per-bucket expected drift magnitude ¯m<sub>b</sub> (Eq. 7); Time = median compute time per position on the Spark configuration described in Section 4.2. Spark RFF-MMD uses the calibrated threshold $3 \times 1 0 ^ { - 3 }$ (Section 6.3.1); the other tests use their implementation defaults. The three tests reporting near-100% FPR and TPR simultaneously do not perform meaningful binary discrimination at their default thresholds; their failure mechanisms are discussed in Section 6.3.6.

## 6.3.3 Performance by drift type and severity regime

Table 8 reports strong- and weak-regime results side by side for each drift type, with the corresponding scenario-level scale in Table 5. The strong regime yields consistent detection across drift types, whereas the weak regime yields no reliable discrimination for either test.

Table 8 Strong- vs weak-regime detection performance by drift type on the Trendyol collection-ranking feature table. RFF-MMD and SWD are the two tests capable of producing meaningful discrimination; the remaining three (per-dim KS, MMD-RBF, projected KS) exhibit systematic failures at all regimes (see Section 6.3.6).
<table><tr><td></td><td></td><td colspan="2">Sudden</td><td colspan="2">Gradual</td><td colspan="2">Incremental</td><td colspan="2">Recurring</td></tr><tr><td>Test</td><td>Metric</td><td>Strong</td><td>Weak</td><td>Strong</td><td>Weak</td><td>Strong</td><td>Weak</td><td>Strong</td><td>Weak</td></tr><tr><td rowspan="3">RFF-MMD</td><td>r</td><td>+0.948</td><td>-0.094</td><td>+0.944</td><td>-0.081</td><td>+0.942</td><td>-0.083</td><td>+0.926</td><td>-0.084</td></tr><tr><td>TPR</td><td>95.2</td><td>0.0</td><td>71.4</td><td>0.0</td><td>71.4</td><td>0.0</td><td>83.3</td><td>0.0</td></tr><tr><td>FPR</td><td>3.4</td><td>3.4</td><td>3.4</td><td>3.4</td><td>3.4</td><td>3.4</td><td>2.6</td><td>2.6</td></tr><tr><td rowspan="3">SWD</td><td>r</td><td>+0.844</td><td>+0.033</td><td>+0.844</td><td>+0.095</td><td>+0.820</td><td>+0.041</td><td>+0.767</td><td>-0.069</td></tr><tr><td>TPR</td><td>95.2</td><td>9.5</td><td>76.2</td><td>14.3</td><td>76.2</td><td>14.3</td><td>83.3</td><td>16.7</td></tr><tr><td>FPR</td><td>10.3</td><td>10.3</td><td>6.9</td><td>10.3</td><td>10.3</td><td>10.3</td><td>13.2</td><td>7.9</td></tr></table>

Metric definitions (r, TPR, FPR and the $\mathcal { P } _ { 1 } / \mathcal { P } _ { 0 }$ partition) as in Table 7. Each cell is a single scenario; TPR and FPR in %. RFF-MMD uses the calibrated threshold $3 \times 1 0 ^ { - 3 }$ (Section 6.3.1); SWD uses the fixed implementation threshold $\tau _ { \mathrm { S W } } = 0 . 0 5 .$ With 21 active and 29 null positions per scenario (12 and 38 for recurring drift), Wilson 95% intervals are wide, e.g., $2 0 / 2 1 = 9 5 . 2 \% \ [ \bar { 7 } 7 . 3$ 99.2], $1 5 / 2 1 = 7 1 . 4 \% \ [ 5 0 . 0 , 8 6 . 2 ]$ $1 / 2 9 = \stackrel { \sim } { 3 } . 4 \%$ [0.6, 17.2], and 3/29 = 10.3% [3.6, 26.4].

In the strong regime, RFF-MMD reaches $r \in [ 0 . 9 3 , 0 . 9 5 ]$ . The range stays tight because m interpolates through gradual and incremental transitions at the intensity scale the statistic tracks, whereas an indicator-based target would collapse to a step function. Its TPR is 95% for sudden, 71% for gradual and incremental, and 83% for recurring drift, while FPR stays within [2.6, 3.4]%, so the single calibrated threshold generalizes across drift types. SWD follows the same pattern at a lower level $( r \in$ [0.77, 0.84], $\mathrm { T P R } \in [ 7 6 . 2 , 9 5 . 2 ] \% )$ but with an FPR of 6.9–13.2%, because its fixed threshold $\tau _ { \mathrm { S W } } = 0 . 0 5$ is not null-calibrated (Section 7). These per-scenario diferences, across drift types and between the two tests, lie within wide uncertainty bands: the Wilson intervals in Table 8 span roughly 17–36 percentage points, and the bootstrap intervals for r overlap between RFF-MMD (lower bounds 0.63–0.80) and SWD (lower bounds 0.43–0.66).

In the weak regime, RFF-MMD detects nothing $( \mathrm { T P R } = 0 \% , r \in [ - 0 . 0 9 4 , - 0 . 0 8 1 ] )$ and SWD reaches a TPR of only 9.5–16.7% with r ≈ 0. At realized-flip fractions of 0.29–0.57%, a shift of this size on the 11-dimensional vector cannot be distinguished from the injection-null baseline under the sample sizes and thresholds used here. The two regimes thus bracket a failure and a success region without locating the power transition between them; an intensity-grid power analysis (e.g. {0.5, 1, 2, 3, 5, 7.5, 10}% with repeated seeds) is left to future work, and the weak-regime results mark an empirical failure point for this configuration rather than a fundamental sensitivity bound.

## 6.3.4 Trajectory analysis

Figure 10 shows per-bucket sliding-scan trajectories of Spark RFF-MMD (red) and Sliced Wasserstein Distance (blue) across all four drift types under both strong and weak severity regimes. In the strong regime, both tests exhibit the expected schedulespecific patterns: a sharp single step at bucket 29 for sudden drift; a smooth ramp for gradual and incremental; alternating on/of cycles for recurring. In the weak regime, the trajectories remain indistinguishable from the injection-null baseline.

![](images/8b9807257aedbd3800d26494887894b7b13623a7b443720204d28ef7401573e7.jpg)  
Fig. 10 Sliding-scan trajectories of Spark RFF-MMD (red) and Sliced Wasserstein Distance (blue) across 8 scenarios. Rows correspond to the four drift types; the left column is the strong regime and the right column the weak regime. Solid lines are the test statistic on a shared log y-axis; dashed horizontal lines are the drift threshold of each test in the matching color (RFF-MMD: 3 × $^ { 1 0 ^ { - 3 } } ,$ calibrated per Section $6 . 3 . 1 ; \mathrm { S W D } ; 5 \times 1 0 ^ { - 2 } .$ , implementation default). The dotted vertical line marks the initial drift onset (for recurring drift, subsequent returns between concepts are additional true transition points not marked). Per-panel Pearson r is shown in each subplot title.

## 6.3.5 Position-25 anomaly

Position 25 exhibits a systematic detection spike across all tests unrelated to injected drift, attributable to natural source-table heterogeneity (multi-day bucket boundary crossing daytime trafic patterns); this is a documented finding of the study, with detailed analysis in Appendix C.

## 6.3.6 Failure modes: per-dim KS, MMD-RBF, and projected KS

Three tests in Table 7 fail to produce a meaningful correlation between statistic and ground truth and/or exhibit near-100% FPR; brief diagnoses follow, with full mechanistic discussion in Section 7.

• Per-dim KS $( r = - 0 . 0 8 5 , \mathrm { F P R } = \mathrm { T P R } = 1 0 0 \% )$ : The Bonferroni-corrected max-KS statistic is dominated by ID-like columns (seed-content and owner identifiers, with ≈ 5.9M and 183k distinct values respectively) under the asymmetric reference/current sampling design used here (reference: first 10,000 rows of the reference table; bucket: first 20,000 rows of each stream-ordered bucket). The two sampling procedures observe systematically diferent empirical distributions on these columns and produce $D \approx 0 . 2 0$ independent of drift, dominating the true drift signal on the $y ^ { \prime }$ dimension $( D \approx 0 . 0 6 )$ . Boolean detection fires at every position; statistic magnitude shows no drift-tracking pattern. Disentangling high-cardinality per se from sampling asymmetry requires an ablation (matched sampling, ID-column removal) identified as future work in Section 7.

• MMD-RBF $( r = - 0 . 0 2 2 , \mathrm { F P R } = 9 8 . 4 \% )$ : The kernel bandwidth is estimated by the median heuristic on 3 000-row driver subsamples drawn separately for the reference and each bucket; because samples come from diferent regions of the stream, bandwidth estimates difer by a few percent across buckets, and we hypothesise that this bandwidth mismatch — interacting with subsampling and the fixed threshold — decorrelates the MMD statistic from the injected drift signal. The same implementation reproduces published detection accuracies on Failing Loudly (Section 6.2), suggesting the failure is scale-and-sampling-specific; a full mechanistic analysis at Trendyol scale is deferred to future work.

• Projected KS $( r = + 0 . 1 0 9 , \mathrm { F P R } = \mathrm { T P R } = 1 0 0 \% )$ : A low but non-zero correlation indicates partial signal capture on random-projected features, but always-reject behavior under the fixed threshold makes the test unusable for production.

## 6.3.7 Runtime profile

Figures 11 (aggregate median with interquartile-range whiskers) and 12 (disaggregated by scenario) report per-position compute cost for the five tests: runtime is stable across drift types and severity regimes, confirming it is a property of test complexity and execution mode rather than of the injected drift structure.

![](images/817b6c6f3f7dc95d8fb89da0085053a0e0421ebb8e5d11f82fb3e0ff97d58788.jpg)

Fig. 11 Per-position runtime aggregated across all 8 scenarios (n = 400 measurements per test: 50 positions × 8 scenarios). Bars are medians; whiskers span the interquartile range. Each test is assigned a distinct color that is reused consistently in Figure 12. MMD-RBF dominates the range at ∼36 seconds per position, roughly 6.5× slower than the next test (Sliced Wasserstein Distance) and roughly 480× slower than per-dim KS.  
![](images/4e8ad9bb7323fc2184b7455cefbaf02f269c7cc757147f1ab820837288513c0a.jpg)  
Fig. 12 Per-position runtime distribution disaggregated by scenario. Each box aggregates n = 50 per-position measurements from one scenario. Colors match Figure 11. The y-axis is broken to accommodate MMD-RBF’s range (∼31–53 s, top panel) while preserving resolution on the four smaller-scale tests (0–10.5 s, bottom panel). Distributions are similar across scenarios, confirming that runtime is a function of test complexity and execution mode, independent of drift structure or severity regime.

## 7 Discussion and Implications for Big Data Systems

The three benchmarks are complementary rather than interchangeable: no single one can characterize a drift detector, because each answers a diferent question about what the tests actually see. The Harvard Dataverse streams probe operational structure: with P(X) stationary and only $P ( Y \mid X )$ changing, multi-column tests detect drift because the class column is part of the test vector (Sections 5.3 and 6.1). Failing Loudly probes implementation consistency, providing a numerical reproduction target $( \mathrm { M A E } \in [ 0 . 0 3 0 , 0 . 0 5 3 ]$ ; Section 6.2). Neither establishes transfer to production scale. The Trendyol benchmark supplies that third axis: the tests continue to produce signal on ∼137.5 million rows and, after the pre-check of Section B.1, on an 11-dimensional feature vector, with conclusions bounded by a single collection-ranking table. The contribution of this study is therefore not a new method but a validation topology.

Univariate marginal tests (chi-square, PSI, Jensen–Shannon, univariate KS, Wasserstein) dominate production monitoring because they are cheap and easy to report, yet our results show that marginal monitoring fails in three distinct ways. First, per-dimension tests cannot see changes in the dependence structure between features; detecting correlation changes under fixed marginals requires a multivariate statistic. Controlled experiments show that feature-wise KS performs close to chance when drift afects only the correlation between features, whereas kernel-based tests do not [8], a result that PSI-based monitoring proposals routinely overlook. Second, at scale, perdimension KS can be captured by the wrong columns. On the pre-check-narrowed Trendyol vector, Bonferroni-corrected per-dim KS rejects at every position $\left( \mathrm { F P R } \right) =$ $\mathrm { T P R } = 1 0 0 \% , r = - 0 . 0 8 5 )$ . The cause is not the classical $n $ ∞ p-value collapse: two ID-like columns (seed-content and owner identifiers, with ≈ 5.9M and ≈ 183k distinct values) difer systematically between the positionally capped reference sample and the stream-ordered current buckets, producing D ≈ 0.20 regardless of drift and swamping the true signal on the $y ^ { \prime }$ dimension $( D \approx 0 . 0 6 )$ . Whether high cardinality or the asymmetric sampling design is responsible requires an ablation (matched sampling and, separately, ID-column removal) that we leave to future work; in the interim, either intervention is a practical safeguard, whereas conformal or bootstrap threshold calibration addresses neither mechanism directly. Third, the “marginal blindness” argument needs refinement on the label axis. On SINE and MIXED, per-dim KS reaches 0% recall while projected KS and MMD-RBF reach 100% (Section 6.1); but because the multi-column vector includes the one-hot class column, that success may stem from the class marginal, from $X { - } Y$ dependence, or both, and only an X-only / Y-only $/ \left( X , Y \right)$ / shufled-Y ablation would separate them. The defensible claim is narrower: Bonferroni-corrected per-dimension tests lose any meaningful signal, including the label marginal, whereas multi-column tests observe a joint signal via projection or kernel. When an outcome $Y$ or prediction $\hat { Y }$ channel exists, adding it to the test vector strengthens detection, as a layer on top of, not a replacement for, monitoring $P ( X )$ . Because the Trendyol benchmark uses a constructed $Y _ { \mathrm { s y n t h } }$ in place of such a channel, its results describe $( X , Y _ { \mathrm { s y n t h } } )$ joint-distribution detection rather than pure $P ( X )$ detection.

Random Fourier Features are what make kernel testing viable at scale. Exact MMD-RBF costs $O ( N ^ { 2 } d )$ and is impractical at the Trendyol reference size $( N =$

100 000). Their approximation [17] promises linear time, but whether they preserve the kernel statistic’s signal at production scale was unknown. Our results answer on both fronts. Under the Failing Loudly protocol, RFF-MMD (D = 500) matches MMD-RBF’s detection accuracy with a size-controlled permutation p-value (Appendix D.1). In the Trendyol strong regime it reaches, averaged over drift types, $r = 0 . 9 4 0$ , TPR $= 8 0 . 4 \%$ , and $\mathrm { F P R } = 3 . 2 \%$ under a threshold calibrated on injection-null pre-drift positions (Section 6.3.1), at a median cost of 2.68 s per position. Distributed RFF-MMD is therefore a practical building block for monitoring streams of millions of rows. The weak regime marks its limit: at realized-flip fractions of 0.29–0.57%, TPR falls to 0% and $r \approx - 0 . 0 9$ . Locating the power transition between the two regimes requires an intensity-grid sweep with repeated seeds.

Finally, a useful signal is not a calibrated test, as the Sliced Wasserstein Distance (SWD) illustrates. Under Failing Loudly with the fixed threshold $\tau _ { \mathrm { S W } } ~ = ~ 0 . 0 5$ , its false-positive rate never approaches the nominal level across 60 no-shift trials and the Wilson 95% interval never contains 0.05 (Appendix D.1): the present implementation is not size-controlled. Yet in the Trendyol strong regime it tracks drift nearly as well as RFF-MMD (mean $r = 0 . 8 1 9$ $\mathrm { T P R } = 8 2 . 7 \% )$ , at the cost of FPR ≈ 10% under the uncalibrated threshold, and it shares RFF-MMD’s weak-regime failure. This behavior is consistent with a documented limitation of uniformly sampled slicing: when the displacement between two distributions concentrates in a low-dimensional subspace, the probability that a random projection direction is nearly orthogonal to it grows exponentially in $d ,$ so the estimator underestimates the true distance [19, 20]. Adaptive slicing variants (max-SW, max-generalized SW, distributional SW), a permutation null, and the null calibration used for RFF-MMD are natural remedies that lie outside the scope of this paper. The broader lesson for production systems is that an unsupervised distance-based signal can serve as an observational alert, but operational decisions require a size-controlled test; the conformal calibration of Leong [10] may bridge the two but has not yet been validated at production scale.

## 8 Conclusion

Multivariate drift detection works at production scale, but only when test choice, calibration, and sampling design are treated as part of the method. We evaluated five multi-column two-sample tests across three complementary layers—controlled concept-drift streams (Harvard Dataverse), a reproduced image-shift benchmark (Failing Loudly), and synthetic concept drift injected into the 137.5-million-row Trendyol collection-ranking feature table—and draw four conclusions.

First, distributed RFF-MMD is the strongest candidate for production monitoring. In the strong regime it tracks expected drift magnitude with, averaged over drift types, $r = 0 . 9 4 0$ , a true positive rate of 80.4%, and a false positive rate of 3.2% under a null-calibrated threshold, at 2.68 s per position; across the eight scenarios, the five tests together required about 5.1 hours of cumulative compute, of which RFF-MMD accounted for about 19 minutes. Random Fourier Features thus turn kernel testing from an $O ( N ^ { 2 } )$ )-memory method into a practical Spark workload.

Second, a drift signal is not a decision rule. The Sliced Wasserstein Distance tracks drift nearly as well $( \mathrm { T P R } = 8 2 . 7 \% )$ but, without calibration, flags about 10% of injection-null positions, while per-dimension KS rejects everywhere, saturated by IDlike columns under asymmetric sampling. Calibration and sampling design, not the statistic alone, determine operational reliability.

Third, benchmark design is part of the result. Sliding-window trajectories scored against row-level ground truth expose what binary recall hides, and pre-check and sanity-check protocols are what separate genuine drift from data-quality artifacts in production tables.

Fourth, every detector has a floor. At realized-flip fractions of 0.29–0.57%, neither RFF-MMD nor SWD separates drift from the null baseline; whether this is a fundamental sensitivity bound or a threshold that shifts with sample size and kernel choice requires intensity-grid power analysis with repeated seeds.

These results come from a single injection run on a single e-commerce table and describe joint $( X , Y _ { \mathrm { s y n t h } } )$ rather than pure P(X) detection. Extending them to other domains, attributing natural covariate heterogeneity separately from injected drift, and bringing adaptive slicing and conformal calibration to production scale are the natural next steps toward monitoring systems that fail loudly rather than silently.

## References

[1] Moreno-Torres, J.G., Raeder, T., Alaiz-Rodr´ıguez, R., Chawla, N.V., Herrera, F.: A unifying view on dataset shift in classification. Pattern Recognition 45(1), 521–530 (2012) https://doi.org/10.1016/j.patcog.2011.06.019

[2] Gama, J., Zliobait˙e, I., Bifet, A., Pechenizkiy, M., Bouchachia, A.: A survey on<sup>ˇ</sup> concept drift adaptation. ACM Computing Surveys 46(4), 44–14437 (2014) https: //doi.org/10.1145/2523813

[3] L´opez Lobo, J.: Synthetic datasets for concept drift detection purposes. Harvard Dataverse (2020). https://doi.org/10.7910/DVN/5OWRGB

[4] Gretton, A., Borgwardt, K.M., Rasch, M.J., Sch¨olkopf, B., Smola, A.: A kernel two-sample test. Journal of Machine Learning Research 13, 723–773 (2012)

[5] Rabanser, S., G¨unnemann, S., Lipton, Z.: Failing loudly: An empirical study of methods for detecting dataset shift. In: Advances in Neural Information Processing Systems 32, pp. 1394–1406 (2019). https://arxiv.org/abs/1810.11953

[6] M¨uller, R., Steidl, M., Lippert, K., Sch¨afer, R., Voigt, T.: Open-source drift detection tools in action: Insights from two use cases. arXiv preprint arXiv:2404.18673 (2024)

[7] Cerqueira, V., Gomes, H., Casado, M., Bifet, A., Pfahringer, B.: A framework for evaluating and benchmarking concept drift detection methods. arXiv preprint arXiv:2606.07789 (2026)

[8] Hinder, F., Vaquet, V., Hammer, B.: One or two things we know about concept drift—a survey on monitoring in evolving environments. Part A: Detecting concept drift. Frontiers in Artificial Intelligence 7, 1330257 (2024) https://doi.org 10.3389/frai.2024.1330257

[9] Khan, M.U., Sadaoui, S.: Learner-based concept drift detection: Analysis and evaluation. arXiv preprint arXiv:2606.20216 (2026)

[10] Leong, J.: Online shift detection and conformal adaptation for deployed safety classifiers. arXiv preprint arXiv:2606.11949 (2026)

[11] Gama, J., Medas, P., Castillo, G., Rodrigues, P.: Learning with drift detection. In: Advances in Artificial Intelligence – SBIA 2004. Lecture Notes in Computer Science, vol. 3171, pp. 286–295. Springer, Berlin, Heidelberg (2004). https://doi. org/10.1007/978-3-540-28645-5 29

[12] Bifet, A., Gavald\`a, R.: Learning from time-changing data with adaptive windowing. In: Proceedings of the 2007 SIAM International Conference on Data Mining, pp. 443–448. SIAM, Philadelphia, PA (2007). https://doi.org/10.1137/1. 9781611972771.42

[13] Su´arez-Cetrulo, A.L., Quintana, D., Cervantes, A.: A survey on machine learning for recurring concept drifting data streams. Expert Systems with Applications 213, 118934 (2023) https://doi.org/10.1016/j.eswa.2022.118934

[14] Souza, V.M.A., Reis, D.M., Maletzke, A.G., Batista, G.E.A.P.A.: Challenges in benchmarking stream learning algorithms with real-world data. arXiv preprint arXiv:2005.00113 (2020)

[15] Zaharia, M., Chowdhury, M., Das, T., Dave, A., Ma, J., McCauley, M., Franklin, M.J., Shenker, S., Stoica, I.: Resilient distributed datasets: A faulttolerant abstraction for in-memory cluster computing. In: Proceedings of the 9th USENIX Conference on Networked Systems Design and Implementation (NSDI ’12), pp. 15–28. USENIX Association, Berkeley, CA (2012). https://www.usenix.org/system/files/conference/nsdi12/nsdi12-final138.pdf

[16] Armbrust, M., Xin, R.S., Lian, C., Huai, Y., Liu, D., Bradley, J.K., Meng, X., Kaftan, T., Franklin, M.J., Ghodsi, A., Zaharia, M.: Spark SQL: Relational data processing in Spark. In: Proceedings of the 2015 ACM SIGMOD International Conference on Management of Data (SIGMOD ’15), pp. 1383–1394. ACM, New York, NY (2015). https://doi.org/10.1145/2723372.2742797

[17] Rahimi, A., Recht, B.: Random features for large-scale kernel machines. In: Advances in Neural Information Processing Systems 20, pp. 1177–1184 (2007)

[18] Gemaque, R.N., Costa, A.F.J., Giusti, R., dos Santos, E.M.: An overview of unsupervised drift detection methods. WIREs Data Mining and Knowledge Discovery

[19] Kolouri, S., Nadjahi, K., S¸im¸sekli, U., Badeau, R., Rohde, G.K.: Generalized sliced Wasserstein distances. In: Advances in Neural Information Processing Systems 32 (NeurIPS 2019), pp. 261–272 (2019). https://arxiv.org/abs/1902.00434

[20] Nguyen, K., Ho, N., Pham, T., Bui, H.: Distributional sliced-Wasserstein and applications to generative modeling. In: International Conference on Learning Representations (ICLR 2021) (2021). https://openreview.net/forum?id=QYjO70ACDK

## Abbreviations

<table><tr><td>API</td><td>Application programming interface</td><td>PSI</td><td>Population Stability Index</td></tr><tr><td>FPR</td><td>False positive rate</td><td>RBF</td><td>Radial basis function</td></tr><tr><td>FWER</td><td>Family-wise error rate</td><td>RFF</td><td>Random Fourier Features</td></tr><tr><td>ID</td><td>Identifier</td><td>RFF-MMD</td><td>MMD with Random Fourier Features</td></tr><tr><td>JS</td><td>Jensen-Shannon</td><td>RKHS</td><td>Reproducing kernel Hilbert space</td></tr><tr><td>KL</td><td>Kullback-Leibler</td><td>SQL</td><td>Structured Query Language</td></tr><tr><td>KS</td><td>Kolmogorov-Smirnov</td><td>SRP</td><td>Sparse random projection</td></tr><tr><td>MAE</td><td>Mean absolute error</td><td>SW</td><td>Sliced Wasserstein</td></tr><tr><td>MMD</td><td>Maximum Mean Discrepancy</td><td>SWD</td><td>Sliced Wasserstein Distance</td></tr><tr><td>MMD-RBF</td><td>MMD with a radial basis function kernel</td><td>TPR</td><td>True positive rate</td></tr></table>

## Appendix A Notation

Table A1 lists the symbols used in the paper together with the place where each is defined.

Table A1 Notation used throughout the paper.
<table><tr><td>Symbol</td><td>Meaning</td><td>Defined in</td></tr><tr><td colspan="3">Two-sample testing</td></tr><tr><td> $\mathcal { D } _ { R } , \mathcal { D } _ { C }$ </td><td>Reference and current samples</td><td>Sec. 3.1</td></tr><tr><td> $P _ { R } , P _ { C }$ </td><td>Distributions of the reference and current samples</td><td>Eq. (1)</td></tr><tr><td> $z ; X , { \overset { \smile } { Y } }$ </td><td>Tested vector ((X, Y) or  $X _ { k } ) ;$  features and label</td><td>Sec. 3.1</td></tr><tr><td>d</td><td>Feature dimensionality</td><td>Table 1</td></tr><tr><td> $s ; p$ </td><td>Test statistic; p-value</td><td>Sec. 3.1</td></tr><tr><td>ψ</td><td>Decision rule</td><td>Sec. 3.1</td></tr><tr><td> $\alpha _ { 0 }$ </td><td>Significance level (0.05)</td><td>Sec. 3.1</td></tr><tr><td>T; TRFF, TSW</td><td>Decision threshold; RFF-MMD  $( 3 \times 1 0 ^ { - 3 } )$  and SWD (0.05) thresholds</td><td>Secs. 4.1, 6.3.1</td></tr><tr><td>1[]</td><td>Indicator function</td><td>Sec. 3.1</td></tr><tr><td colspan="3">Sliding-window scan</td></tr><tr><td>N</td><td>Number of position buckets</td><td>Eq. (2)</td></tr><tr><td>b;  $\scriptstyle { B _ { b } }$ </td><td>Bucket index; current bucket</td><td>Eq. (2)</td></tr><tr><td>Sb</td><td>Test statistic at bucket b</td><td>Sec. 3.2</td></tr><tr><td> $W ; \Delta ; t$ </td><td>Window size; stride; stream index (moving reference)</td><td>Sec. 3.2</td></tr><tr><td> $\mathcal { P } _ { 0 } , \mathcal { P } _ { 1 }$ </td><td>Injection-null and injection-active positions</td><td>Sec. 6.3.2</td></tr><tr><td>r</td><td>Pearson correlation between  $s _ { b }$  and  $\bar { m } _ { b }$ </td><td>Sec. 6.3.2</td></tr><tr><td colspan="3">Drift injection and ground truth</td></tr><tr><td> $Y _ { A } , Y _ { B }$ </td><td>Concept A and Concept B labeling rules</td><td>Eqs. (E1), (E2)</td></tr><tr><td> $\mathrm { s i m } _ { \mathrm { i m g } } , \mathrm { s i m } _ { \mathrm { t x t } }$ </td><td>Image and text cosine similarity</td><td>App. E</td></tr><tr><td> $y _ { A } ; y ^ { \prime } ; Y _ { \mathrm { s y n t h } }$ </td><td>Pre-drift label; injected label; injected label variable</td><td>Sec. 5.3</td></tr><tr><td>α</td><td>Per-row drift intensity</td><td>Sec. 5.3</td></tr><tr><td> $\sigma ; \phi$ </td><td>Severity; scope</td><td>Sec. 5.3</td></tr><tr><td> $s _ { 0 } ; w ; \lambda$ </td><td>Drift start, transition width and recurrence period</td><td>Alg. 7</td></tr><tr><td>δ</td><td>fractions Concept A/B disagreement rate</td><td>Eq. (4)</td></tr><tr><td> $\bar { \alpha } _ { s }$ </td><td>Mean drift intensity of scenario s</td><td>Eq. (4)</td></tr><tr><td> $e ; m ; f$ </td><td>Eligibility; expected drift magnitude; realized flip</td><td>Eqs. (6)−(8)</td></tr><tr><td>a</td><td>(per row) Coarse affected-row indicator</td><td>Alg. 7</td></tr><tr><td> $\bar { m } _ { b } ; ~ \bar { f } _ { b }$ </td><td>Per-bucket means of m and f</td><td>Sec. 5.3</td></tr><tr><td colspan="3">Test configuration</td></tr><tr><td>D</td><td>RFF dimension (500; 200 in the Trendyol scans)</td><td>Table 1</td></tr><tr><td>B</td><td>Number of permutations</td><td>Table 1</td></tr><tr><td>K</td><td>Number of random projections (32 sparse; 64 dense</td><td>Table 2</td></tr><tr><td>P</td><td>in the Trendyol scans) Number of SWD projections (100; 25 in the Trendyol</td><td>Table 1</td></tr><tr><td> $n _ { \mathrm { d r v } }$ </td><td>scans) Driver sample size (20,000)</td><td>Sec. 4.2</td></tr></table>

## Appendix B Pre-check and Sanity-check Procedures for Production Drift Injection

This appendix presents (i) the custom sliding-window scan framework used throughout Sections 3.2, 5.2, and 6.3 as pseudocode, and (ii) the pre-check and verification steps

of the injection design defined in Section 5.3, likewise as procedural pseudo-code. The procedures are expressed in terms of a table and can be translated directly to any SQL-like query language or dataframe API (BigQuery, Spark, pandas).

## B.1 Pre-check protocol: active feature inventory

For any concept drift benchmark to produce meaningful results, the columns afected by injection must carry active variance — a condition not always guaranteed in production tables where upstream aggregation may leave outputs empty, planned feature values may fail to materialize, or flag/count columns may be zeroed out. A recommended seven-criterion audit protocol (zero/near-zero variance; identifier semantics; cardinality ratio; monotonic structure; missingness; reference/current sampling compatibility; embedding array dimensionality) is listed here; Algorithm 2 implements its variance criterion, and the audit reported here scanned that criterion only, with the additional criteria consolidated post-hoc after the failure analysis of Section 7 showed they matter as much for benchmark-vector eligibility. In particular, the identifiersemantics criterion was not used to filter the benchmark vector: the three identifier columns (seed-content, owner, and channel identifiers, two of them with high cardinality) were retained by design, because such columns are routinely present in production feature tables and a monitoring system must behave sensibly in their presence. Before testing, every vector component is z-scored with the reference statistics, and missing values are imputed with the reference mean.

The variance scan of the 164-column source table returned three groups: 148 of 158 scalar numeric columns held a single value (0) across all rows — these included the natural injection candidates (action flag, user count, cart count, and collectionclick flag) and were entirely dormant; 10 scalar columns carried genuine variance (4 metadata scalars — seed-content identifier, owner identifier, channel identifier, and collection size — and 6 similarity metrics: image, tag, and text cosine similarity plus the corresponding matched-content counts); and 6 embedding arrays (three collection, three seed; dimensions 128, 256, 415) which were audited separately and carried variance. Table B2 lists the resulting column groups and maps the descriptive names and symbols used in the text to the original column names.

This finding had three direct consequences for the injection design: (i) dormant columns cannot serve as injection targets (a Bernoulli flip on a constant-zero column is undefined); (ii) dormant columns must be excluded from the benchmark vector or they dilute the drift signal and disturb kernel bandwidth estimation; and (iii) the synthetic label must be defined over live-variance features only, which we do over the similarity metrics (§5.3).

## B.2 Sanity-check protocol: post-injection verification

Each injected table is verified in four parts before it is scanned (pseudocode in Section B.3): (i) the flip-rate check (Algorithm 3) compares each scenario’s empirical flip rate with the prediction of Eq. (4); (ii) the zero-flip check (Algorithm 5) confirms that $y ^ { \prime } = y _ { A }$ for every row with $\alpha = 0$ ; (iii) the covariate side-efect check (Algorithm 6) compares summary statistics of each feature before and after the drift onset;

Table B2 Feature and injection-variable glossary for the Trendyol benchmark. Descriptive names and symbols used in the text are mapped to the original column names of the source table.
<table><tr><td>Name in Text</td><td>Symbol</td><td>Role</td><td>Original Column</td></tr><tr><td colspan="4">Source-table columns</td></tr><tr><td>Seed-content identifier</td><td></td><td>Metadata (vector)</td><td>seed</td></tr><tr><td>Owner identifier</td><td></td><td>Metadata (vector)</td><td>ownerid</td></tr><tr><td>Channel identifier</td><td></td><td>Metadata (vector)</td><td>channelId</td></tr><tr><td>Collection size</td><td></td><td>Metadata (vector)</td><td>content_cnt</td></tr><tr><td>Image cosine similarity</td><td>simimg</td><td>Similarity (vector); Concept A</td><td>image_cosine_similarity</td></tr><tr><td>Text cosine similarity</td><td>simtxt</td><td>Similarity (vector); Concept B</td><td>text_cosine_similarity</td></tr><tr><td>Tag cosine similarity</td><td></td><td>Similarity (vector)</td><td>tag-cosine_similarity</td></tr><tr><td>Matched-content counts</td><td></td><td>Similarity (vector)</td><td>*_matched_contents_count</td></tr><tr><td>Collection identifier</td><td></td><td>Row key (hashing)</td><td>collectionid</td></tr><tr><td>Recommendation source</td><td></td><td>Not in vector</td><td>source</td></tr><tr><td>Action, user-count, cart-count and</td><td></td><td>Dormant</td><td>action_flag, user_count, cart_ count, collection_click_flag</td></tr><tr><td colspan="4">collection-click fields</td></tr><tr><td colspan="4"></td></tr><tr><td>Injected label</td><td>Columns written by the injection (Algorithm 7)</td><td>Label (vector)</td><td>synthetic-y</td></tr><tr><td>Pre-drift label</td><td> $y ^ { \prime } , Y _ { \mathrm { s y n t h } }$ </td><td>Verification</td><td>original_synthetic-y</td></tr><tr><td>Drift intensity</td><td>yA α</td><td>Ground truth</td><td>drift_alpha</td></tr><tr><td>Eligibility indicator</td><td>e</td><td>Ground truth (Eq. 6)</td><td>eligible</td></tr><tr><td>Expected drift</td><td>m</td><td>Ground truth (Eq. 7)</td><td>expected_drift_magnitude</td></tr><tr><td>magnitude</td><td></td><td></td><td></td></tr><tr><td>Realized-flip indicator</td><td>f</td><td>Ground truth (Eq. 8)</td><td>realized_flip</td></tr><tr><td>Affected indicator</td><td>a</td><td>Ground truth (coarse)</td><td>is_row_affected</td></tr><tr><td>Stream index</td><td>i</td><td>Row ordering</td><td>stream_index</td></tr></table>

and (iv) the ground-truth consistency check (Algorithm 4) aggregates m and f per bucket and confirms that the two targets agree.

## B.3 Pseudocode: pre-check and sanity-check procedures

The procedure of Algorithm 2 requires a single full scan on production-scale tables (∼1–3 minutes for 137M rows). It operates on scalar numeric columns only; array and string columns require separate handling (§B.1).

The row-level quantities m and f are written by the injection query according to Eqs. (7) and (8).

Rationale: In the pre-drift region, $p _ { \mathrm { a c t i v e } } = p _ { A }$ . Since Concept A is a deterministic rule $( \mathrm { s i m } _ { \mathrm { i m g } } > 0$ with integer output 0 or 1), the u<sub>label</sub> < p<sub>active</sub> comparison must be deterministic, and $y ^ { \prime } = y _ { A }$ must hold.

```latex
Algorithm 1 Sliding-Window Changepoint Detection
Require: Reference set $\mathcal { D } _ { R } ;$ stream-ordered current dataset $\overline { { \mathcal { D } _ { C } ; } }$ number of positions
$N ;$ two-sample tests $\mathcal { T } = \{ T _ { 1 } , \ldots , T _ { k } \}$ with decision rules $\{ \psi _ { j } \} _ { j = 1 } ^ { k }$ (Section 3.1).
Ensure: Trajectory matrix $\mathbf { S } \in \mathbb { R } ^ { N \times k } ;$ decision matrix $\mathbf { D } \in \{ 0 , 1 \} ^ { N \times k } ;$ optional
ground-truth vector $\mathbf { g } \in [ 0 , 1 ] ^ { N }$
1: Partition $\mathcal { D } _ { C }$ into $\{ B _ { 0 } , \dotsc , \bar { B _ { N - 1 } } \}$ per Equation (2)
2: for all $b \in \{ 0 , 1 , \ldots , \overset { \cdot } { N } - 1 \}$ do
3: if $B _ { b }$ carries the injection ground truth α and m then
4: $\begin{array} { r } { \mathbf { g } [ b ]  \frac { 1 } { | \mathcal { B } _ { b } | } \sum _ { z \in \mathcal { B } _ { b } } m ( z ) } \end{array}$ ▷ ground-truth aggregation, Section 3.3
5: end if
6: for all $T _ { j } \in \mathcal { T }$ do
7: $( s _ { b } ^ { ( j ) } , p _ { b } ^ { ( j ) } )  T _ { j } ( \mathcal { D } _ { R } , \mathcal { B } _ { b } )$ ▷ two-sample test, Section 4
8: $\mathbf { S } [ b , j ]  s _ { b } ^ { ( j ) }$
9: $\mathbf { D } [ b , j ]  \bar { \psi } _ { j } \big ( s _ { b } ^ { ( j ) } , p _ { b } ^ { ( j ) } \big )$ ▷ p-value or threshold rule
10: end for
11: end for
12: return (S, D, g)
```

Algorithm 2 Active feature inventory (variance scan)   
Require: Source table $T ;$ list of scalar numeric columns $C = \{ c _ { 1 } , \ldots , c _ { N } \}$   
Ensure: For each column, the triple (distinct ${ \bf \nabla } : ( c )$ , mean(c), sd(c)) of distinct-value   
count, mean and standard deviation; partition of C into alive and dormant   
columns.   
1: Compute (distinct $\left( c _ { i } \right)$ , mean(c ), sd(c )) for all $c _ { i } \in C$ in a single scan of $T$   
2: for all $c _ { i } \in C$ do   
3: if distinct $( c _ { i } ) \leq 1$ then   
4: Add $c _ { i }$ to the Dormant list   
5: else   
6: Add $c _ { i }$ to the Alive list; record (mean $. ( c _ { i } )$ , sd(c<sub>i</sub>))   
7: end if   
8: end for   
9: return Alive list → candidate live-variance columns (final eligibility for the   
benchmark vector additionally requires the semantic $/$ cardinality $/$ sampling   
checks described in Section B.1); Dormant list → columns excluded from   
injection targets and from the vector.

Algorithm 3 Flip rate verification   
Require: Injected benchmark table $T _ { \mathrm { i n j } }$ containing the labels $y ^ { \prime }$ and $y _ { A }$ and the   
per-row drift intensity α; scenario parameters $( \phi , \sigma , \delta )$   
Ensure: Per-scenario empirical flip rate; comparison against schedule-specific theo  
retical prediction (Eq. 4).   
1: for all s ∈ Scenarios do   
2: $n _ { \mathrm { t o t a l } }  \vert \{ r \in T _ { \mathrm { i n j } }$ : scenar $\mathtt { i o \_ i d } ( r ) = s \mathtt { \mathtt { j } } |$   
3: $n _ { \mathrm { f l i p p e d } }  \vert \{ r \in T _ { \mathrm { i n j } }$ : scenario $\mathbf { i d } ( r ) = { \dot { s } } \wedge y ^ { \prime } ( r ) \neq y _ { A } ( r ) \} |$   
4: flip rate<sub>empirical</sub> ← n<sub>flipped</sub>/n<sub>total</sub>   
5: $\bar { \alpha } _ { s } \gets \mathrm { m e a } \bar { \mathrm { n } } ( \alpha )$ over rows with scenario $\mathbf { \underline { { { i d } } } } = s$ ▷ schedule-specific   
6: flip rate<sub>theoretical</sub> $ \bar { \alpha } _ { s } \cdot \phi \cdot \sigma \cdot \delta$ ▷ Eq. 4   
7: if $\mathrm { \left| \mathrm { H i p { - } r a t e } _ { e m p i r i c a l } - \mathrm { H i p { - } r a t e } _ { t h e o r e t i c a l } \right| / \mathrm { H i p { - } r a t e } _ { t h e o r e t i c a l } > 0 . 0 5 }$ then   
8: Warning: scenario s deviates from design   
9: end if   
10: end for

```latex
Algorithm 4 Ground-truth per-bucket aggregation
Require: Injected table $T _ { \mathrm { i n j } }$ with per-row ground truth m $\left( \mathrm { E q . ~ 7 } \right)$ and $f \ ( { \mathrm { E q . ~ } } 8 ) { \mathrm { ; } }$
number of position buckets N.
Ensure: Per-bucket true expected drift magnitude (primary correlation target) and
per-bucket realized-flip rate (robustness target).
1: Define bucket boundaries according to the stream order for N buckets.
2: for all bucket $b = 0 , \ldots , N - 1$ do
3: $\bar { m } _ { b } \gets \mathrm { m e a n } ( m )$ over rows in bucket b
4: $\bar { f } _ { b } \gets \mathrm { m e a n } ( f )$ over rows in bucket b
5: end for
6: return $( \bar { m } _ { b } ) _ { b = 0 } ^ { N - 1 }$ and $( \bar { f } _ { b } ) _ { b = 0 } ^ { N - 1 }$ : the former is ≈ 0 for pre-drift buckets and $\alpha \cdot $
$\sigma \cdot \phi \cdot \delta$ for buckets at drift intensity α (where $\phi$ is scope fraction and δ the
concept disagreement rate); at production-scale bucket sizes the two arrays agree
to $\Delta { \dot { r } } < 1 0 ^ { - 3 }$ when used as correlation targets.
```

Algorithm 5 Zero-flip check in the pre-drift region   
Require: Injected table ${ \overline { { T _ { \mathrm { i n j } } } } } .$   
Ensure: Boolean — has determinism preservation succeeded?   
1: for all $s \in$ Scenarios do   
2: n<sub>wrong</sub> <sub>pre</sub> $ \vert \{ r$ in scenario s $: \alpha ( r ) = 0 \land y ^ { \prime } ( r ) \neq y _ { A } ( r ) \} |$   
3: if $n _ { \mathrm { w r o n g \mathrm { - } p r e } } > 0$ then   
4: return Fail (s, n<sub>wrong pre</sub>)   
5: end if   
6: end for   
7: return Pass

Algorithm 6 Covariate side-efect distribution check   
Require: Injected table $\overline { { T _ { \mathrm { i n j } } } } ;$ list of X columns to check $C _ { X }$   
Ensure: Per-column pre-drift vs post-drift distribution comparison.   
1: for all $c \in C _ { X }$ do   
2: pre stats ← (mean(c), sd(c), distinct(c)) over rows where $\alpha ( r ) = 0$   
3: post stats ← (mean(c), sd(c), distinct(c)) over rows where $\alpha ( r ) = 1$   
4: if relative dif(pre stats, post stats) $> 0 . 0 2$ then   
5: Warning: column c shows aggregate distributional discrepancy from the   
pre-drift baseline (may indicate that injection has an unintended side efect on   
this column, or natural production heterogeneity across stream positions)   
6: end if   
7: end for   
8: return Provides no evidence of an aggregate distributional side efect on the   
checked X summaries (this is a distribution-level consistency check and does not   
by itself establish row-wise equality $X _ { \mathrm { a f t e r } } ( r ) = X _ { \mathrm { b e f o r e } } ( r ) )$

## Appendix C Position-25 Anomaly: Source-table Heterogeneity Diagnosis

Figure 10 reveals a recurring pattern: all eight scenarios show a systematic spike in the RFF-MMD statistic at position 25, reaching roughly 20× the pre-drift mean. This position sits within the region where $\bar { m } _ { b } = 0 ;$ it might be interpreted as a false positive, but the sanity checks defined in Section B.2 confirm that no row in this position was afected by the injection. To diagnose the behavior, we compared the contentfeature profile of the bucket-25 rows against the reference distribution and against the neighboring buckets 24 and 26.

The finding: position 25 departs systematically from the reference across every measured feature — this is not an injection artifact but a real data-heterogeneity phenomenon in the source table. The bucket-25 rows exhibit 19% higher mean collection size (23.4 vs 19.6) and 17% higher std (16.9 vs 14.5), 11–13% lower similarity metrics across all three cosine channels (image, text, tag), and 13% lower mean $y ^ { \prime }$ (0.191 vs 0.219); neighboring buckets 24 and 26 depart by <2% on every feature. This indicates that collections whose collection-identifier hash falls into this bucket carry a systematically diferent feature profile — most likely a batch collection deposit within a specific time window, or a taxonomy segment whose hash range happens to align with this bucket — a one-of, non-periodic anomaly.

This finding highlights three points. First, RFF-MMD producing a high statistic in this bucket is correct behavior — not a “false positive” but a correct capture of a real distributional diference in the source data. Because the injection is not active in this region, this reduces the test’s correlation against the ground truth (since $\bar { m } _ { b } = 0$ enters the correlation) but demonstrates the test’s fundamental sensitivity. Second, this is a real challenge that drift detection faces in production environments: injected synthetic drift and natural data heterogeneity mix within the same statistical signal. Third, how to handle such anomalies in production monitoring — requiring an injection-agnostic yet content-aware framework — is an open research question, noted in Section 8 as future work.

## Appendix D Extension Methods: Full Characterization Results

The two extension methods introduced in Section 4.1 — RFF-MMD and Sliced Wasserstein Distance — were characterized under the Failing Loudly protocol; their motivation is to avoid MMD-RBF’s $O ( N ^ { 2 } d )$ scaling. This section reports how successful each method is and which calibration challenges they bring.

## D.1 No-shift calibration of the extension methods

Under 60 randomized no-shift comparisons (30 per dataset), an observed FPR whose Wilson 95% interval contains 0.05 is compatible with the nominal level in these finite controls (finite-sample chance variation can produce intervals that miss the nominal value even for a well-calibrated test). RFF-MMD satisfies this expectation at all sample sizes, backed by the population-level super-uniformity of the permutation construction (under the exchangeability assumptions of the permutation null). Sliced Wasserstein Distance does not: with the fixed threshold $\tau _ { \mathrm { S W } } = 0 . 0 5$ , the observed FPR fluctuates between 50% and 100% and the Wilson interval never contains 0.05 at any N — the present SWD implementation is not a size-controlled test. At Trendyol production scale (Section 6.3), the SWD statistic magnitude does track drift in the strong regime (mean $r = 0 . 8 1 9 , \mathrm { T P R } = 8 2 . 7 \% )$ but under the same fixed threshold the FPR sits around 10%, consistent with the Failing Loudly calibration finding; a null-percentile calibration analogous to the RFF-MMD procedure (Section 6.3.1) may improve empirical calibration on similar production distributions and is discussed further in Section 7.

## D.2 Scalability-accuracy trade-of

RFF-MMD $( D = 5 0 0 )$ has $O ( N \cdot D \cdot d )$ compute complexity, fundamentally diferent from MMD-RBF’s $O ( N ^ { 2 } d )$ , and scales to $N = 1 0 0 0 0$ and beyond (the Trendyol scans apply it with $D = 2 0 0$ against a 100,000-row reference). Under Failing Loudly, RFF-MMD detection accuracy remains comparable to MMD-RBF (98% on MNIST combined shifts, 15% on CIFAR-10 adversarial shifts vs. 7% for MMD-RBF) with reliable permutation size control, and in the Trendyol benchmark (Section 6.3) produced the most consistent strong-regime detection performance (mean $r = 0 . 9 4 0$ across four drift types on ∼137.5M rows).

## Appendix E Synthetic Label Design: Concept A and Concept B Definitions

Since the source table contains no genuine target column that could serve as a drift injection substrate, we defined Concept A and Concept B rules over the live similarity metrics.

Concept A labels a row positive if the collection shows any visual similarity to its seed content:

$$
Y _ { A } = 1 [ \mathrm { s i m } _ { \mathrm { i m g } } > 0 ] ,\tag{E1}
$$

where si $\mathrm { m } _ { \mathrm { i m g } }$ is the image cosine similarity between the collection and its seed content. This similarity is bimodal in the source table — 78% of rows are zero and the remaining 22% lie in [0.5, 1.1] (median 0.94), with almost no intermediate values — so this rule functions as a near-binary “was similarity measured” signal with positive rate $P ( Y _ { A } = 1 ) \approx 0 . 2 1 9 5$

Concept B labels a row positive if there is any textual similarity:

$$
\begin{array} { r } { Y _ { B } = \mathbb { 1 } [ \mathrm { s i m } _ { \mathrm { t x t } } > 0 ] , } \end{array}\tag{E2}
$$

where $\mathrm { s i m } _ { \mathrm { t x t } }$ is the corresponding text cosine similarity, with $P ( Y _ { B } = 1 )$ ≈ 0.4473. The disagreement rate between the two rules — the fraction of rows where they yield diferent decisions — is δ ≈ 0.5724 over the source table, providing an upper bound on the maximum theoretical drift efect (no more than a δ-fraction of rows can have their labels changed even at maximum severity and scope).

This concept pair was chosen for three reasons: (i) conceptual separation — visual vs. textual similarity are distinct signal modalities of the same recommendation domain, mirroring a realistic shift from vision-based to text-based signals in a production ranking pipeline; (ii) balanced baselines — both positive rates fall in [0.22, 0.45], neither too sparse nor too dominant for Bernoulli-based two-sample tests; and (iii) derivation from $X - Y$ is deterministically derived from features present in the table, so the injection can be interpreted as a change in $P ( Y \mid X )$

## Appendix F Injection Boundaries: The Covariate-Preservation Guarantee

A critical design principle of the pipeline is that no feature in the source table is modified by injection: the injected label $y ^ { \prime }$ is added as an outcome, but the similarity metrics (including $\mathrm { s i m } _ { \mathrm { i m g } }$ and $\mathrm { s i m } _ { \mathrm { t x t } } )$ , the recommendation source, the collection size, and all other original columns remain untouched. This guarantees the injection operator leaves X row-wise unchanged $( X _ { \mathrm { a f t e r } } ( r ) = X _ { \mathrm { b e f o r e } } ( r )$ for every row) and provides the analytical ground on which the injection-relative component of the signal can be attributed to the change in $P ( Y \mid X )$ . Because the underlying production table may carry natural covariate heterogeneity across stream positions, the observed reference and current window distributions need not satisfy $P _ { R } ( X ) = P _ { C } ( X )$ exactly (empirically visible at position 25; Section 6.3.5); any such departure is a source-table phenomenon rather than a pipeline artifact.

Algorithm 7 formalizes the row-level injection process defined in Section 5.3, expressed in terms of a table T with rows indexed by the stream index i and the parameters (σ, ϕ, drift type) defining a scenario. Executable as a single-pass SQL query or DataFrame transformation over any distributed backend (BigQuery, Spark, pandas).

Algorithm 7 Concept Drift Injection   
Require: Source table $\overline { { T } }$ with n rows; drift $\begin{array} { r l r l r l } { \mathrm { t y p e } } & { { } \ d } & { { } } & { } & { { } \in { } } \end{array}$   
{sudden, gradual, incremental, recurring}; severity $\sigma \in [ 0 , 1 ] ;$ scope $\phi \in [ 0 , 1 ] ;$ drift   
start fraction $s _ { 0 }$ , drift width fraction w, recurrence period fraction $\lambda ;$ random   
seed $\zeta ;$ hash modulus $M ;$ Concept A rule $Y _ { A }$ , Concept B rule $Y _ { B }$ over $X$   
Ensure: Drifted table $T ^ { \prime }$ with synthetic target $y ^ { \prime } { \mathrm { . } }$ drift intensity $\alpha ,$ and per-row   
ground truth $e ( r )$ , m(r), $f ( r )$ (Eqs. 6–8); the coarser indicator $a ( r ) = 1 [ \alpha >$   
$0 \land e ( r ) = 1 ]$ is also emitted for backward comparability.   
1: $n _ { s } \gets \lfloor n s 0 \rfloor ; ~ n _ { w } \gets \operatorname* { m a x } ( 1 , \lfloor n w \rfloor ) ; ~ n _ { p } \gets \operatorname* { m a x } ( 1 , \lfloor n \lambda \rfloor )$   
2: for all $r \in T$ in ascending order of hash $. ( \mathrm { i d } ( r ) )$ , with rank $i = 0 , 1 , \ldots , n - 1$ do   
3: if $d =$ sudden then   
4: α ← 0 if $i < n _ { s }$ else 1   
5: else if $d =$ incremental then   
6: $\alpha  \mathrm { c l i p } ( ( i - n _ { s } ) / n _ { w } , 0 , 1 )$   
7: else if $d = \mathfrak { g }$ gradual then   
8: if $i < n _ { s }$ then   
9: $\alpha \gets 0$   
10: else $\mathbf { i f } \ i \ge n _ { s } + n _ { w }$ then   
11: $\alpha \gets 1$   
12: else   
13: $\alpha  [ 1 + \exp ( - 1 2 \big ( ( i - n _ { s } ) / n _ { w } - 0 . 5 \big ) ) ] ^ { - }$ −1   
14: end if   
15: else if $d =$ recurring then   
16: $\alpha  0 \mathrm { ~ i f ~ } i < n _ { s }$ else $\left( \lfloor ( i - n _ { s } ) / n _ { p } \rfloor \right.$ mod 2   
17: end if   
18: $u _ { \mathrm { s c o p e } } \gets \mathrm { h a s h } \big ( \mathrm { i d } ( r ) , \zeta , \mathrm { * } _ { \mathrm { s c o p e } ^ { \prime } } ) / M$   
19: in scope $ 1 1 \dot { u } _ { \mathrm { s c o p e } } < \phi ]$   
20: $p _ { A }  Y _ { A } ( X ( r ) ) ; \hat  \} p _ { B }  Y _ { B } ( X ( r ) )$ ▷ deterministic rules over X   
21: if in scope then   
22: $p _ { B } ^ { \mathrm { e f f } }  p _ { A } + \sigma \cdot ( p _ { B } - p _ { A } )$ ▷ severity-scaled Concept B, Eq. (3)   
23: else   
24: $p _ { B } ^ { \mathrm { e f f } }  p _ { A }$   
25: end if   
26: $p _ { \mathrm { a c t i v e } }  p _ { A } + \alpha \cdot ( p _ { B } ^ { \mathrm { e f f } } - p _ { A } )$ ▷ $\operatorname { E q . }$ (3)   
27: $u _ { \mathrm { l a b e l } } \gets \mathrm { h a s h } \big ( \mathrm { i d } ( r ) , \zeta , \ : ^ { \circ } \mathrm { l a b e l } ^ { \prime \prime } \big ) / M$   
28: $y ^ { \prime } ( r ) \gets \mathbb { 1 } \left[ u _ { \mathrm { l a b e l } } < p _ { \mathrm { a c t i v e } } \right]$   
29: $e ( r ) \gets \mathbb { 1 } \left[ \mathrm { i n \_ s c o p e } \land p _ { A } \neq p _ { B } \right]$ $\triangleright \operatorname { E q } .$ (6)   
30: $m ( \boldsymbol { r } ) \gets \boldsymbol { \alpha } \cdot \boldsymbol { \sigma } \cdot \boldsymbol { e } ( \boldsymbol { r } )$ ▷ Eq. (7)   
31: $f ( r ) \gets \mathbb { 1 } [ y ^ { \prime } ( r ) \neq y _ { A } ( r ) ]$ ▷ Eq. (8)   
32: $\begin{array} { r } { a ( r ) \gets \mathbb { 1 } [ \alpha > 0 \land e ( r ) = 1 ] } \end{array}$ ▷ coarser indicator retained for backward   
comparability   
33: emit $( r , \ y ^ { \prime } ( r ) , \ \alpha , \ e ( r ) , \ m ( r ) , \ f ( r ) , \ a ( r ) )$   
34: end for