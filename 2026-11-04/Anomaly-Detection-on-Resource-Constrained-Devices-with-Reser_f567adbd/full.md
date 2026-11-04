# Anomaly Detection on Resource-Constrained Devices with Reservoir Computing

Marco Angioli<sup>∗</sup>, Mathias Persson<sup>∗</sup>, Hiroto Tamura, Hadi Banaee, Amy Loutfi, Gouhei Tanaka, Alberto Giaretta, Denis Kleyko

Abstract—Anomaly detection at the edge requires the deployment of models that achieve competitive performance, while minimizing computational cost and memory footprint. In this article, we investigate the suitability of the Mahalanobis Distance of Reservoir States (MDRS), an unsupervised time-series anomaly detection method, for resource-constrained devices and propose two optimizations to reduce its computational requirements. Specifically, we replace random reservoir connectivity (MDRS-Rand) with a circular shift (MDRS-Circ) and show that the anomaly score for an entire time series can be computed from the final reservoir state alone. Furthermore, we show that exploiting the reservoir’s temporal memory enables more efficient input preprocessing, further reducing computational cost. We evaluate the MDRS variants on the MLPerf Tiny anomaly detection benchmark against the reference autoencoder and other anomaly detection methods, and deploy their inference kernels on a NUCLEO-F401RE microcontroller to assess latency, estimate energy consumption, and measure memory footprint. Among the evaluated methods, only MDRS and the autoencoder meet the benchmark’s performance target, with MDRS requiring fewer operations and smaller models. With optimized preprocessing, MDRS achieves an ROC AUC of 0.869, a model size of 24 KB, an estimated energy consumption of 24 µJ per inference, and a latency of 0.32 ms, corresponding to a 75.9× speedup over the autoencoder with standard preprocessing.

Index Terms—reservoir computing, anomaly detection, resource-constrained devices, tinyML

<sup>∗</sup>Equal contribution. MA and DK acknowledge funding from the Swedish Strategic Research Foundation (Grant No. FFL24-0111). HT acknowledges funding from the Japan Society for the Promotion of Science, JSPS KAKENHI (Grant No. JP24K20855), a project, JPNP14004, commissioned by the New Energy and Industrial Technology Development Organization (NEDO), and the Institute of AI and Beyond at the University of Tokyo. AL and DK acknowledge funding from the Knut and Alice Wallenberg Foundation under the Wallenberg Scholars program (Grant No. KAW2023.0327). GT acknowledges funding from the Japan Society for the Promotion of Science, JSPS KAKENHI (Grant No. JP25K24744). AG acknowledges funding from the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. AG and DK acknowledge funding from NordForsk (Grant No. 241658). DK acknowledges funding from the Swedish Research Council (Grant No. 2025-05421).

## I. INTRODUCTION

IoT and resource-constrained devices are increasingly used to perform tasks at the edge. Many of such tasks fall into the category of anomaly detection applied to time-series [1]. For example, networking intrusion detection systems require methods that analyze incoming packets and infer whether they compose an ongoing cybersecurity attack. Another example is the analysis of environmental conditions (e.g., video or audio signals) to monitor processes and identify suspicious deviations from the normal conditions. To achieve the near real-time performance necessary for carrying out these tasks successfully, computations need to be performed locally. Offloading them to the cloud is often not viable, as the communication overhead is non-negligible. However, given the limited computational capabilities of resource-constrained devices, it is crucial to develop computationally-efficient methods.

Tiny machine learning (tinyML) [2] has emerged as an area specifically targeting these settings, enabling the deployment of ML models on resource-constrained devices with limited memory, storage, and computational capabilities. To stimulate the development of methods for tinyML, the MLPerf Tiny benchmark [3] provides a standardized framework for evaluating such models, including an anomaly detection task based on acoustic machine-condition monitoring. By defining target detection performance and measuring computational efficiency, the benchmark supports the evaluation of trade-offs between accuracy and deployment cost.

Reservoir computing is a family of randomized recurrent neural networks promising for tinyML [4]. In reservoir computing, most network weights are initialized randomly and remain fixed, and only the output weights are trained [5]. This architecture drastically reduces training cost compared to fully-trainable neural networks, making reservoir computing particularly suitable for resource-constrained deployment. Building on this foundation, Tamura et al. recently proposed the Mahalanobis Distance of Reservoir States (MDRS) [6], an unsupervised anomaly detection method that detects pointwise anomalies by measuring the deviation of reservoir states from an estimated distribution of normal conditions. Unlike previous reconstruction- or prediction-based reservoir computing methods, MDRS produces temporally stable anomaly scores and avoids the need for supervised training of the output weights, making it a promising candidate for unsupervised anomaly detection on resource-constrained devices.

In this article, we investigate the suitability of MDRS for anomaly detection on resource-constrained IoT devices and propose two optimizations to reduce its computational requirements. First, we replace the random reservoir connectivity with a circular shift operation [7], [8], eliminating the need for a dense recurrent matrix multiplication. Second, we introduce an end-only score that computes the Mahalanobis distance only for the final reservoir state rather than at every time step. We evaluate the MDRS variants on the MLPerf Tiny anomaly detection benchmark against the reference autoencoder and other anomaly detection methods (Section III) in terms of detection performance, computational cost, and memory footprint (Section V). Among the evaluated methods, only MDRS and the reference autoencoder meet the MLPerf Tiny performance target, with MDRS requiring fewer operations and a smaller model. Moreover, since the reservoir integrates its input over time, MDRS can process single frames without the temporal stacking required by the autoencoder to meet the performance target, allowing further reductions in computational cost. When deployed on a NUCLEO-F401RE microcontroller (Section VI), MDRS achieves 6.9× lower latency and estimated energy consumption than the autoencoder with identical inputs. With optimized preprocessing, these improvements reach 75.9× while meeting the MLPerf Tiny performance target with a model size of 24 KB.

## II. RELATED WORK

A review of ML-based anomaly detection in IoT scenarios [9] emphasizes practical deployment constraints and the need to balance detection performance with computational cost, communication overhead, and privacy. It surveys approaches to IoT anomaly detection, comparing supervised, unsupervised, and semi-supervised learning methods. A complementary review focusing on unsupervised anomaly detection in multivariate IoT time-series has been presented in [1].

Statistical methods detect deviations from normal behavior using probabilistic models, distance measures, or outlier scores [10]. Conventional ML methods, such as support vector machines (SVMs), decision trees, and clustering, learn decision boundaries or input transformations from data [11]. Deep learning methods use neural networks to model complex spatial or temporal dependencies [12], [13]. Among deep learning methods, autoencoder-based methods have been deployed on resource-constrained devices within tinyML [2], including LSTM autoencoders for urban noise monitoring on ESP32 devices [14] and compact autoencoders for machine fault detection on an Arduino Nano 33 BLE Sense [15]. However, differences in datasets and hardware platforms make it difficult to compare their detection performance and computational cost. To enable such comparisons, the MLPerf Tiny benchmark [3] provides a common anomaly detection task, a target performance, and a reference autoencoder. In its closed division, the submissions run this reference model on different hardware platforms and inference engines [16]. The open division also admits other models and preprocessing, but its few anomaly detection submissions used smaller or quantized fully-connected autoencoders on FPGAs [17] and on a RISC-V processor [18].

In this study, we focus on reservoir computing [19]–[21], a computationally efficient approach to time-series processing. Reservoir computing belongs to a class of randomized recurrent neural networks [22] in which input and recurrent weights are initialized randomly and fixed, so only the output weights are trained. This enables temporal modeling at a lower training cost than fully-trainable recurrent networks. Existing methods for anomaly detection based on reservoir computing mainly rely on prediction or reconstruction errors [23]–[28]. In contrast, MDRS (see Section III-A) [6] estimates the distribution of reservoir states from normal data and uses the Mahalanobis distance from this distribution as the anomaly score. It, therefore, does not require explicit prediction or reconstruction of the input signal and is suitable for unsupervised learning settings. MDRS falls between conventional ML and statistical methods, and its properties make it a suitable candidate for anomaly detection in IoT scenarios. However, its detection performance, computational cost, and memory footprint have not been evaluated on widely used TinyML benchmarks nor demonstrated in resource-constrained deployments.

## III. METHODS

## A. MDRS: Mahalanobis Distance of Reservoir States

The MDRS method has two stages, a training stage and an inference stage [6]. During training, normal data are passed through the reservoir of an Echo State Network (ESN) [20], and the reservoir state distribution for the normal data is estimated. Assuming the reservoir states follow a multivariate Gaussian distribution, the parameters to be estimated are the mean vector and covariance matrix for the reservoir states.

During inference, the test data are fed into the ESN with the same parameters and transformed into a sequence of reservoir states. The Mahalanobis distance is then computed to evaluate how each reservoir state for the test data differs from the estimated reservoir state distribution for the normal data. The Mahalanobis distance serves as the anomaly score, where a higher score indicates a greater deviation from the estimated distribution of normal conditions and, therefore, suggests a higher likelihood of an anomaly. The reservoirbased input transformation enables nonlinear temporal feature extraction and dynamic state embedding, thereby enhancing the separability between normal and anomalous conditions in the reservoir state space [21].

MDRS separates offline model estimation from online inference. Once the model has been trained, inference requires only the ESN configuration and the precision matrix of the normal reservoir-state distribution. Under the zero-mean assumption, no mean vector needs to be estimated. This small set of model parameters makes MDRS suitable for deployment on resourceconstrained devices.

During training, the model estimates the structure present in the normal data. First, the training data are passed through the ESN, where the reservoir state is updated as follows:

$$
\mathbf { x } ( t + \Delta t ) = \mathbf { x } ( t ) + \frac { \Delta t } { \tau } \left( - \mathbf { x } ( t ) + g \mathbf { J } \Phi ( \mathbf { x } ( t ) ) + h \mathbf { V } \mathbf { f } ^ { \mathrm { i n } } ( t ) \right) ,\tag{1}
$$

where ${ \bf x } ( t )$ is the state of the N reservoir neurons, ${ \bf f } ^ { \mathrm { i n } } ( t )$ is the input, Φ is the component-wise hyperbolic tangent, and $\Delta t , \ \tau , \ g ,$ and h are hyperparameters. The input projection

V and the reservoir connectivity J matrices are random and fixed, and $\mathbf { x ( 0 ) } \ = \ \mathbf { 0 }$ at the start of each time-series. The resulting reservoir states are then fitted to a multivariate Gaussian distribution. For simplicity, the normal reservoirstate distribution is assumed to be centered at the origin in this article. Accordingly, no mean vector is estimated, and the distribution is represented by its non-normalized covariance matrix (scatter matrix) $\Sigma _ { 0 }$ , accumulated over the K training time-series:

$$
\pmb { \Sigma } _ { 0 } : = \sum _ { k = 1 } ^ { K } \sum _ { n = 1 } ^ { T _ { 0 } / \Delta t } \mathbf { x } _ { k } ( n \Delta t ) \mathbf { x } _ { k } ( n \Delta t ) ^ { \mathsf { T } } ,\tag{2}
$$

where ${ \bf x } _ { k } ( t )$ is the reservoir state for the k-th time-series and $T _ { 0 }$ denotes the time-series duration. The matrix captures the spread and correlations of the normal reservoir states around the origin. A regularization term is added before the matrix is inverted to obtain the regularized precision matrix $\mathbf { \Lambda } \Lambda _ { 0 } \mathbf { : }$

$$
\pmb { \Lambda } _ { 0 } : = ( \pmb { \Sigma } _ { 0 } + \alpha \mathbf { I } ) ^ { - 1 } .\tag{3}
$$

The regularization is performed as in ridge regression by including the positive parameter α. During the inference stage, the resulting precision matrix characterizes the reservoir-state distribution, and a time-series is processed using the same reservoir dynamics as in the training stage, Equation (1). For a reservoir state at time $t = n \Delta t$ , the anomaly score is computed using the squared Mahalanobis distance:

$$
\delta _ { \mathrm { M } } ( \mathbf { x } ( t ) , \pmb { \Lambda } _ { 0 } ) : = \mathbf { x } ( t ) ^ { \top } \pmb { \Lambda } _ { 0 } \mathbf { x } ( t ) .\tag{4}
$$

Unlike the reconstruction error of an autoencoder, the Mahalanobis distance is a sum of contributions along the principal directions of the normal reservoir states, i.e., the eigenvectors of $\Sigma _ { 0 }$ , which shows along which directions a recording deviates from normal operation and could support explainable, human-in-the-loop monitoring.

## B. MDRS with simplified connectivity

For the scope of this article, we refer to the standard MDRS with random reservoir connectivity, described in Section III-A, as MDRS-Rand. In addition to MDRS-Rand, we investigate an MDRS with simplified reservoir connectivity based on the ideas from [7], [8]. Specifically, the randomly generated reservoir connectivity matrix J in Equation (1) is replaced by a circular shift operation. In this variant (denoted as MDRS-Circ), the reservoir neurons are connected in a ring, where each neuron is connected to a successor and the last neuron connects back to the first one. This connectivity can be represented by a permutation matrix containing a single non-zero value in each row and column.

It is more efficient to replace the matrix-vector multiplication with the circular shift operation. This eliminates the need to generate and store the random connectivity matrix J. Consequently, this deterministic design reduces both the memory footprint and the computational cost of updating the reservoir, while allowing us to examine whether comparable anomaly detection performance can be maintained.

## C. Aggregation strategies for anomaly scores

In its original formulation, MDRS-Rand detects point-wise anomalies by computing the Mahalanobis distance for individual reservoir states (cf. Equation (4)). In contrast, the anomaly detection task in the MLPerf Tiny benchmark requires detecting anomalies at the time-series level [29]. Therefore, the sequence of point-wise anomaly scores must be aggregated into a single score representing the entire recording.

A similar aggregation problem arises in ESN-based timeseries classification [30]. We consider two natural aggregation strategies. The first, called per-step aggregation, computes the Mahalanobis distance for every reservoir state after the initial one, $\mathbf { x } ( 0 ) = \mathbf { 0 } { \mathrm { , } }$ and averages the resulting scores over all time steps. The second, called end-only score, computes the Mahalanobis distance only for the reservoir state corresponding to the final time step of the recording. We also investigate intermediate variants of the per-step aggregation, in which the Mahalanobis distance is computed at a fixed stride s rather than at every time step, i.e., at the time steps $s , 2 s , \ldots , n s$

## D. Baseline anomaly detection methods

During the experimental evaluation in Sections V-VI, we compare the performance of the proposed MDRS variants with several baseline anomaly detection methods. First, the MLPerf Tiny benchmark includes a fully-connected autoencoder (AutoEncoder) as the reference model for the anomaly detection task [3]. It includes full-precision weights in floating-point format (AutoEncoder-FP32; used for software evaluation) and weights quantized to 8-bit integers (AutoEncoder-INT8; used for resource-constrained deployment). This model serves as the primary baseline.

Second, given our focus on resource-constrained deployment, we consider lightweight anomaly detection methods implemented in the widely used Python Outlier Detection (PyOD) library [31], with hyperparameters selected on the validation set: GMM, PCA, HBOS, ECOD, COPOD, KNN, CBLOF, LODA, DeepSVDD, LSTMAD (PyOD defaults, due to its training cost), and the time-series methods SpectralResidual and MatrixProfile. The last three methods process each recording as a sequence; the other methods score each input sample independently, and the anomaly score of a recording is the mean over its input samples.

## IV. DATA, PREPROCESSING, AND METRICS

## A. Anomaly detection task in the MLPerf Tiny benchmark

To evaluate the proposed MDRS variants, we use the anomaly detection task from MLPerf Tiny [3], an industrystandard benchmark suite for ML on resource-constrained devices. The task is based on the subset of the dataset from the DCASE2020 Challenge [32], which comprises recordings of miniature machine operating sounds collected under both normal and anomalous conditions. Specifically, the MLPerf Tiny benchmark uses recordings from the toy-car machine type. Each recording is stored as a WAV audio file, and the dataset is organized into separate training and testing sets.

The anomaly detection task is designed for unsupervised learning. During training, only recordings corresponding to normal machine operation are available, while the test set contains both normal and anomalous recordings. The training set contains 1 000 recordings of normal operation for each of seven toy cars (7 000 in total) mixed with environmental noise, whereas the test set contains unseen normal and anomalous recordings for four of these toy cars; following MLPerf Tiny, the test set consists of every tenth test recording of the DCASE2020 dataset (248 recordings). Since the MLPerf Tiny benchmark does not define a validation set, we construct one using additional recordings from the same four toy cars, available in the DCASE2020 Challenge dataset but not included in the MLPerf Tiny test set. The validation set contains both normal and anomalous recordings and is used exclusively for hyperparameter selection. The MDRS model itself is trained only on normal data, while anomaly labels are used only to evaluate candidate hyperparameter settings.

The objective is to model normal operating behavior and assign an anomaly score to each recording, with larger scores indicating a higher likelihood of anomalies. This setting reflects realistic condition monitoring scenarios, where anomalies are rare and representative data are often unavailable.

## B. Preprocessing

For reproducible comparisons across methods, the MLPerf Tiny benchmark adopts a standardized preprocessing. Each recording (sampled at 16 kHz) is transformed into a logmel spectrogram with $n _ { \mathrm { m e l s } } = 1 2 8$ mel bands, a window of 64 ms (1024 samples), and a hop of 32 ms (512 samples), of which the 200 central frames (frames 50-249) are kept. Five consecutive frames are concatenated into one vector, resulting in 196 input samples of 640 values per recording.

We first adopt exactly this preprocessing to evaluate MDRS under the conditions of the benchmark (Sections V-A and V-B), and then we investigate how the preprocessing can be optimized for MDRS (Sections V-C and VI).

## C. Metrics

MLPerf Tiny uses the area under the receiver operating characteristic curve (ROC AUC), hereafter referred to as AUC, to evaluate the anomaly detection performance. The receiver operating characteristic curve characterizes the tradeoff between the true positive rate and false positive rate over all possible detection thresholds [33], while AUC summarizes this curve as a single scalar value. Higher AUC values indicate better separation between normal and anomalous recordings. We assign one anomaly score to each recording and compute the AUC over all recordings of the test set. Since the reservoir of MDRS is initialized randomly, we report its average AUC (± standard deviation) over 50 random reservoir initializations. For the anomaly detection task, MLPerf Tiny specifies the target performance of $\mathrm { { A U C } \geq \ 0 . 8 5 0 }$ , which serves as the primary performance criterion in Sections V and VI. From the perspective of the MLPerf Tiny benchmark, models achieving this target performance are considered equivalent in terms of anomaly detection performance, allowing computational efficiency to become the primary criterion for comparison.

TABLE I  
HYPERPARAMETER GRID SEARCH VALUES. RESERVOIR CONNECTIVITY IS EVALUATED FOR BOTH MDRS-RA N D AND MDRS-CI R C.
<table><tr><td>Hyperparameter</td><td>Candidate values</td></tr><tr><td> $N$ </td><td>{32, 64, 96, 128, 192, 256}</td></tr><tr><td> $\Delta t$ </td><td> $\{ 5 \times 1 0 ^ { - 5 } , 1 \times 1 0 ^ { - 4 } , 2 \times 1 0 ^ { - 4 } , 5 \times 1 0 ^ { - 4 } , 1 \times 1 0 ^ { - 3 } \}$ </td></tr><tr><td>T</td><td> $\{ 0 . 0 0 1 , 0 . 0 0 5 , 0 . 0 1 , 0 . 0 2 , 0 . 0 3 \}$ </td></tr><tr><td>g</td><td>{0.05, 0.1, 0.5, 1, 3}</td></tr><tr><td>h</td><td>{0.05, 0.1, 0.5, 1, 3}</td></tr><tr><td>α</td><td>{0.001, 0.005, 0.01, 0.02, 0.03}</td></tr></table>

In addition to anomaly detection performance, we evaluate the computational cost and memory footprint of the considered models. We quantify computational cost as the number of operations to process a recording during inference, providing a hardware-independent metric for comparison. For the software experiments (Section V), we measure floatingpoint operations using the Performance Application Programming Interface (PAPI) library [34], which accesses hardware performance counters. We use the PAPI\_SP\_OPS and PAPI\_DP\_OPS events to count single- and double-precision floating-point operations, respectively. Since these counters do not capture integer operations, AutoEncoder-INT8, which performs the same operations as AutoEncoder-FP32 in 8-bit integer arithmetic, is assigned the same count as AutoEncoder-FP32. Only the inference is measured; the log-mel front-end, which is shared by all models, is excluded.

The memory footprint is characterized by model size and peak RAM usage during training and inference. Model size corresponds to the storage required by the saved model. Peak RAM usage is measured in a fresh Python process dedicated to each stage, with log-mel features precomputed. We report the increase in peak memory relative to memory usage before each stage, isolating the memory overhead attributable to the method from that of the Python runtime, libraries, and feature extraction front-end. Measurements are performed on an x86 host and reported as the median of five runs.

## V. EXPERIMENTAL EVALUATION IN SOFTWARE

## A. Effect of reservoir size and connectivity on performance

We begin by investigating how the reservoir size, $N \in$ $\{ 3 2 , 6 4 , 9 6 , 1 2 8 , 1 9 2 , 2 5 6 \}$ , and the reservoir connectivity (MDRS-Rand and MDRS-Circ) affect anomaly detection performance. For each reservoir size, a grid search is performed to determine the remaining hyperparameters $( \Delta t , \tau ,$ g, h, and α). Table I lists the values considered.

Throughout this experiment, we use the MLPerf Tiny preprocessing $( n _ { \mathrm { m e l s } } ~ = ~ 1 2 8$ , five stacked frames) and the perstep aggregation to obtain a single anomaly score for each recording. For each reservoir size, every configuration is first evaluated with MDRS-Rand and a single random initialization, and the ten configurations with the highest AUC on the validation set are evaluated further over 50 initializations; the configuration with the highest average validation AUC is selected and used for both MDRS-Rand and MDRS-Circ. For

TABLE II  
AVERAGE AUC (± STD. DEVIATION OVER 50 RANDOM INITIALIZATIONS) ON THE TEST SET AS A FUNCTION OF RESERVOIR SIZE, N, WITH THE PER-STEP AGGREGATION AND THE MLPERF TINY PREPROCESSING.
<table><tr><td>RESERVOIR, N</td><td> $\mathtt { M D R S - R a n d }$ </td><td> $\mathtt { M D R S - C i r c }$ </td></tr><tr><td>32</td><td> $0 . 8 4 3 \pm 0 . 0 2 1$ </td><td> $0 . 8 3 6 \pm 0 . 0 2 1$ </td></tr><tr><td>64</td><td> $0 . 8 7 8 \pm 0 . 0 1 2$ </td><td> $0 . 8 7 3 \pm 0 . 0 1 2$ </td></tr><tr><td>96</td><td> $0 . 8 8 2 \pm 0 . 0 0 9$ </td><td> $0 . 8 8 2 \pm 0 . 0 1 0$ </td></tr><tr><td>128</td><td> $0 . 8 9 4 \pm 0 . 0 0 4$ </td><td> $0 . 8 9 6 \pm 0 . 0 0 5$ </td></tr><tr><td>192</td><td> $0 . 8 9 0 \pm 0 . 0 0 4$ </td><td> $0 . 8 9 0 \pm 0 . 0 0 4$ </td></tr><tr><td>256</td><td> $0 . 8 8 3 \pm 0 . 0 0 4$ </td><td> $0 . 8 8 4 \pm 0 . 0 0 5$ </td></tr></table>

$N = 6 4 .$ , the reservoir size used in the following, this yields $\Delta t = 5 \times 1 0 ^ { - 5 } , \tau = 0 . 0 0 5 , g = 3 , h = 0 . 1$ , and $\alpha = 0 . 0 0 1$

Table II reports the average AUC on the test set for different reservoir sizes. Both variants achieve the MLPerf Tiny target performance from $N ~ = ~ 6 4$ upwards, whereas the smallest reservoir $( N = 3 2 )$ falls just below it (0.843 and 0.836). The highest AUC is obtained with N = 128 (0.894 and 0.896), and larger reservoirs do not improve it further, possibly because the amount of training data remains fixed while the number of parameters of the precision matrix grows quadratically with N. We use N = 64 in the following, as it is within 0.02 of the highest AUC at about half the cost of N = 128 (Table III).

## B. Comparison of aggregation strategies

Next, we investigate the effect of the aggregation strategies introduced in Section III-C. Figure 1 shows the average AUC on the test set over 50 reservoir initializations for the perstep aggregation with different strides (solid lines) and for the end-only score (dashed lines), for both MDRS-Rand and MDRS-Circ with $N = 6 4$

The two connectivities behave almost identically: for every stride, their average AUCs differ by at most 0.005. The AUC of the per-step aggregation increases with the stride, from $0 . 8 7 8 \pm 0 . 0 1 2$ and $0 . 8 7 3 \pm 0 . 0 1 2$ for a stride of 1 to $0 . 8 9 7 \pm 0 . 0 1 0$ and $0 . 8 9 3 \pm 0 . 0 1 1$ for a stride of 97 (MDRS-Rand and MDRS-Circ, respectively), and the endonly score, which corresponds to a stride of 195, achieves the highest AUC: $0 . 9 0 1 \pm 0 . 0 1 1$ and $0 . 8 9 7 \pm 0 . 0 1 3$ . A possible explanation is that per-step aggregation does not include a warm-up period. The reservoir starts from a zero state and accumulates information about the recording gradually, so its early states are dominated by this transient and reflect only the beginning of the recording. The per-step aggregation weighs early states as much as later ones, whereas increasing the stride skips them, since the first scored state is at step s, and the endonly score uses only the final state, which contains information accumulated from the preceding frames. This is consistent with the AUC increasing with the stride and being highest for the end-only score.

Reducing the number of Mahalanobis distance evaluations per recording does not degrade the AUC; on the MLPerf Tiny benchmark, the final-state score achieves the highest anomaly detection performance. Since this substantially reduces the computational cost of MDRS (Section V-C), it is used in all subsequent experiments in this section.

![](images/619571f4e4539d633528b01c0f139571ff2b5e8dd7a0861186a6faa97f166cb1.jpg)  
Fig. 1. Test AUC of MDRS $( N = 6 4$ , MLPerf Tiny preprocessing) versus the stride of the per-step aggregation (solid lines) and with the end-only score (dashed lines), which coincides with stride 195. Mean over 50 reservoir initializations; bars and bands: 95% confidence interval of the mean.

## C. Comparison of computational cost

All proposed MDRS variants achieve the MLPerf Tiny target performance, allowing us to extend the analysis to their computational cost and memory footprint.

1) Computational cost of MDRS variants: We first investigate how the reservoir connectivity and the aggregation strategy affect the computational cost during inference. Table III reports the number of operations per recording for the four MDRS variants across different reservoir sizes.

Replacing the random connectivity with the circular shift removes the $N \times N$ matrix-vector product $J \Phi ( x ( t ) )$ at every time step, whereas the input projection $V f ^ { \mathrm { i n } } ( t )$ , which dominates the cost of small reservoirs, is unchanged. The saving grows with reservoir size: with the per-step aggregation, $\mathtt { M D R S - C i r c }$ requires 5.3% fewer operations than MDRS-Rand for $N = 3 2$ and 22.3% fewer for $N = 2 5 6 .$ Similarly, the end-only score avoids most of the Mahalanobis distance evaluations per recording, saving 5.0-21.9% for MDRS-Rand. The two optimizations are complementary: MDRS-Circ with the end-only score requires 10.3-44.2% fewer operations than MDRS-Rand with per-step aggregation.

2) Comparison to the baseline methods: Figure 2 compares the proposed MDRS variants with the baseline methods (Section III-D) in terms of anomaly detection performance, computational cost during inference (left panel), and model size (right panel). We first consider the MLPerf Tiny pre processing $( n _ { \mathrm { m e l s } } = 1 2 8 ,$ , five stacked frames; Section IV-B), shown as filled circles for MDRS and the autoencoder and used for all other baselines (gray diamonds). Four models achieve the target performance of $\mathrm { A U C } \geq 0 . 8 5 0 \ddagger$ MDRS-Rand and MDRS-Circ (0.901 and 0.897), AutoEncoder-FP32 (0.903), and AutoEncoder-INT8 (0.860), with MDRS requiring 18.2 and 16.4 million operations per recording, 5.8 and 6.4 times fewer than the autoencoder (105.2 million). None of the other baselines achieves the target among the evaluated configurations. The baselines requiring fewer operations than MDRS, HBOS, PCA, CBLOF, and LODA (0.34 to 13.0 million), achieve at most 0.781 (CBLOF). The best baseline overall, GMM with 128 diagonal Gaussian components, achieves 0.849 with $6 . 9 \times 1 0 ^ { 7 }$ operations, about four times more than MDRS-Circ, and its parameters alone take about 650 KB in single precision, more than the flash memory of the microcontroller used in Section VI. KNN (0.800) requires $3 . 5 \times 1 0 ^ { 1 1 }$ operations and 7.0 GB, as it compares each test sample with all training samples. LSTMAD, which replaces the random reservoir with a trained recurrent network composed of LSTM cells, achieves 0.753 with $1 . 9 \times 1 0 ^ { 9 }$ operations.

TABLE III  
NUMBER OF FLOPS (IN MILLIONS) PER RECORDING DURING INFERENCE BY THE FOUR MDRS VARIANTS AS A FUNCTION OF RESERVOIR SIZE, N, MEASURED WITH HARDWARE PERFORMANCE COUNTERS. IN PARENTHESES: REDUCTION WITH RESPECT TO MDRS-RAND WITH PER-STEP AGGREGATION.
<table><tr><td>RESERVOIR, N</td><td>32</td><td>64</td><td>96</td><td>128</td><td>192</td><td>256</td></tr><tr><td>MDRS-Rand, per-step</td><td>9.14</td><td>19.86</td><td>32.19</td><td>46.11</td><td>78.74</td><td>117.77</td></tr><tr><td>MDRS-Rand, end-only</td><td>8.68 (↓5.0%)</td><td>18.17 (↓8.5%)</td><td>28.46 (↓11.6%)</td><td>39.55 (↓14.2%)</td><td>64.14 (↓18.6%)</td><td>91.94 (↓21.9%)</td></tr><tr><td>MDRS-Circ, per-step</td><td>8.65 (↓5.3%)</td><td>18.09 (↓8.9%)</td><td>28.33 (↓12.0%)</td><td>39.37 (↓14.6%)</td><td>63.84 (↓18.9%)</td><td>91.51 (↓22.3%)</td></tr><tr><td>MDRS-Circ, end-only</td><td>8.20 (↓10.3%)</td><td>16.40 (↓17.5%)</td><td>24.60 (↓23.6%)</td><td>32.81 (↓28.8%)</td><td>49.23 (↓37.5%)</td><td>65.68 (↓44.2%)</td></tr></table>

The MLPerf Tiny preprocessing, however, is tailored to the reference autoencoder: since the autoencoder has no memory and processes each input sample independently, the temporal context must be contained in the input sample itself, hence the stacking of five consecutive frames. The reservoir of MDRS, instead, integrates its input over time, so its state already retains information about the previous frames. We therefore evaluate MDRS also with a single frame per time step (hollow markers). Moreover, since MDRS exceeds the target performance by a wide margin with $n _ { \mathrm { m e l s } } = 1 2 8$ , we also evaluate fewer mel bands, $n _ { \mathrm { m e l s } } ~ \in ~ \{ 3 2 , 6 4 \}$ (triangles and squares), which further shorten the input sample. The autoencoder is evaluated with the same six preprocessing configurations, so that the two models are compared under the same preprocessing conditions.

The single-frame input confirms the different role of the stacking in the two methods. For MDRS, it reduces the number of operations by a factor of 3.6−4.5 while changing the AUC by less than 0.01; for instance, with $n _ { \mathrm { m e l s } } = 1 2 8$ it requires 3.6 instead of 16.4 million operations and achieves an AUC of 0.906. For AutoEncoder-FP32, in contrast, the AUC decreases by $0 . 0 4 { - } 0 . 0 9$ and falls below the target performance (e.g., 0.810 with $n _ { \mathrm { m e l s } } = 1 2 8 )$ , while the number of operations decreases by at most a factor of two, since the 128-unit hidden layers dominate it. Reducing the number of mel bands lowers the cost of MDRS further while keeping it above the target. With $n _ { \mathrm { { m e l s } } } ~ = ~ 6 4$ and a single frame, MDRS achieves its highest AUC (0.912 for both variants) with 2.0 (MDRS-Circ) and 3.8 (MDRS-Rand) million operations; with $n _ { \mathrm { m e l s } } = 3 2$ and a single frame, MDRS-Circ requires only 1.2 million operations, 13.4 times fewer than with the MLPerf Tiny preprocessing, and still achieves an AUC of 0.869; among all evaluated models, only HBOS requires fewer operations (0.34 million), at an AUC of 0.713. The autoencoder, in contrast, loses AUC with fewer mel bands: with five stacked frames, AutoEncoder-FP32 achieves 0.861 with $n _ { \mathrm { m e l s } } = 6 4$ and 0.820 with $n _ { \mathrm { m e l s } } = 3 2$ , and AutoEncoder-INT8 0.831 and 0.751. Only AutoEncoder-FP32 with $n _ { \mathrm { m e l s } } = 6 4$ and five stacked frames achieves the target.

Finally, Table IV reports the peak RAM of the models that achieve the target. Training MDRS requires 61 MB and 2 s (Intel Core i9-12900), five times less memory and about 1 000 times less time than the autoencoder (315 MB, 35 min). During inference, MDRS requires about 700 KB with the MLPerf Tiny preprocessing, between the INT8 (524 KB) and FP32 (1 168 KB) autoencoders, and down to 372 KB with $n _ { \mathrm { m e l s } } = 3 2$ and a single frame. The memory of the baselines, none of which reaches the target, grows linearly with the training set; they require from 64 MB to an estimated 40 GB for training (LSTMAD: 39 GB already with 700 recordings) and from 5.6 MB to an estimated 65 GB during inference.

TABLE IV  
PEAK RAM DURING TRAINING (MB) AND DURING INFERENCE (KB) FOR THE MODELS THAT ACHIEVE THE TARGET PERFORMANCE.
<table><tr><td>MODEL</td><td>TRAINING (MB)</td><td>INFERENCE (KB)</td></tr><tr><td>MDRS-Rand,  $1 2 8 \times 5$ </td><td>62</td><td>720</td></tr><tr><td>MDRS-Circ,128 × 5</td><td>61</td><td>708</td></tr><tr><td>MDRS-Rand, 32 × 1</td><td>61</td><td>556</td></tr><tr><td> $\mathtt { M D R S - C i r c } , 3 2 \times 1$ </td><td>60</td><td>372</td></tr><tr><td>AutoEncoder-FP32</td><td>315</td><td>1,168</td></tr><tr><td>AutoEncoder-INT8</td><td>315</td><td>524</td></tr></table>

Beyond its lower cost, training MDRS requires only normal data, which a device observes during regular operation without any labels or examples of faults, and involves no gradient computation: $\Sigma _ { 0 }$ is a sum of rank-one terms that can be accumulated one reservoir state at a time without storing the training data, and $\Lambda _ { 0 }$ can be updated directly with incremental methods, avoiding the matrix inversion. For $N ~ = ~ 6 4 , ~ \Sigma _ { 0 }$ occupies 16 KB in single precision and fits in the RAM of the microcontroller used in Section VI. MDRS could therefore be trained on the device after installation and updated continuously during operation, e.g., to adapt to an individual machine or to slow changes in its normal behavior; we leave this to future work.

Overall, MDRS is the only model that achieves the target performance with every preprocessing, and with $n _ { \mathrm { m e l s } } = 3 2$ and a single frame, MDRS-Circ requires the fewest operations, the smallest model, and the least memory during inference among the models that achieve the target.

## VI. EXPERIMENTAL EVALUATION IN HARDWARE

To demonstrate the impact of MDRS variants under realistic resource-constrained deployment conditions, we ported the algorithm to the ST NUCLEO-F401RE board and compared it against the reference AutoEncoder-INT8 in terms of three

![](images/e3978884cd56e791a845801271bcf6dc1f48668cb4f2c3286216ff3defabd77d.jpg)

![](images/a365c96e2daff3523f4d9d8e6b91620c8085301e5b3a182229a53979fdd9e7e6.jpg)

$$
( \boldsymbol { n } _ { \mathrm { m e l s } } = 1 2 8 )
$$

$$
 { n _ { \mathrm { m e l s } } } = 6 4
$$

$$
n _ { \mathrm { m e l s } } = 1 2 8
$$

Fig. 2. Anomaly detection performance versus computational cost (left) and model size (right) during inference. MDRS $( N = 6 4 ,$ , averaged over 50 reservoir initializations) and the autoencoder are shown for six preprocessing configurations: n ∈ {32, 64, 128} (marker shape), with five stacked frames (filled) or a single frame (hollow) per input sample. The baseline models use the MLPerf Tiny preprocessing.

quantities that directly affect resource-constrained deployment: execution time, estimated energy consumption, and memory footprint. The NUCLEO-F401RE microcontroller features an 84 MHz Cortex-M4F core with single-precision floating-point support, 512 KB of flash memory, and 96 KB of RAM. Running all kernels on identical hardware and using the same timing and energy-estimation procedure isolates the algorithmic differences while avoiding confounding board-dependent effects.

Based on the results of Section V, we deploy MDRS with $n _ { \mathrm { m e l s } } ~ = ~ 3 2$ and a single frame per time step. Among the evaluated preprocessing configurations, it requires the fewest operations and the smallest model, while still exceeding the target performance. To compare MDRS with the reference model under identical inputs, we also deploy it with the MLPerf Tiny preprocessing $( n _ { \mathrm { m e l s } } = 1 2 8$ , five stacked frames).

## A. Deployment and measurement setup

Following MLPerf Tiny, which excludes pre- and postprocessing from the measurements, only the inference kernels run on the microcontroller; feature extraction, training, and the computation of the metrics are performed off-board. The board reads preprocessed feature vectors stored in flash, runs the timed kernel, and sends the anomaly score to the host, where it is compared with the Python reference. Since MLPerf Tiny defines one inference as one input window, we measure the execution time of one window with the Cortex-M4F cycle counter, discard five warm-up calls, and report the median of the following 21. With the per-step aggregation, a window includes one Mahalanobis distance evaluation; with the end-only score, only the reservoir update, since the final Mahalanobis distance, amortized over the 195 windows of a recording, adds less than 1%. Energy per inference is estimated as the execution time multiplied by the run-mode power of the microcontroller from its datasheet (22.6 mA at 3.3 V), and the memory footprint as the flash footprint of the deployed model.

TABLE V

MDRS ON THE NUCLEO-F401RE $( N = 6 4 , 3 2 \times 1$ INPUT). LATENCY: MEDIAN OF 21 RUNS; PARENTHESES: CYCLE REDUCTION VS. PER-STEP MDRS-RA N D; AUC: MEAN OF 50 INITIALIZATIONS (STD 0.001).

<table><tr><td>MODEL VARIANT</td><td>CYCLES (k)</td><td>LATENCY (ms)</td><td>ENERGY (µJ)</td><td>FLASH (KB)</td><td>AUC</td></tr><tr><td>MDRS-Rand, per-step</td><td>92.7</td><td>1.10</td><td>82</td><td></td><td>400.843</td></tr><tr><td>MDRS-Rand, end-only</td><td>59.2 (↓36%)</td><td>0.70</td><td>53</td><td></td><td>400.868</td></tr><tr><td>MDRS-Circ, per-step</td><td>60.6 (↓35%)</td><td>0.72</td><td>54</td><td></td><td>24 0.843</td></tr><tr><td>MDRS-Circ, end-only</td><td>27.1 (↓71%)</td><td>0.32</td><td>24</td><td></td><td>24 0.869</td></tr></table>

## B. On-board characterization of MDRS variants

We first evaluate the four MDRS variants obtained by combining the two reservoir connectivities introduced in Section III: a random matrix J (MDRS-Rand) and the special case of permutation (circular shift, MDRS-Circ) with the two strategies of computing anomaly scores considered in Section III-C: the per-step aggregation and the end-only score.

Table V reports the on-board results for the four variants with the deployed preprocessing, $n _ { \mathrm { m e l s } } ~ = ~ 3 2$ and a single frame. MDRS-Rand with the per-step aggregation requires 92.7 K-cycles per inference, corresponding to 1.10 ms and 82 µJ, and achieves an AUC of 0.843, just below the target. Replacing the random connectivity with the circular shift removes the $N \times N$ matrix-vector product with J, which reduces the cycles by 34.6% and the flash footprint from 40 to 24 KB. Replacing the per-step aggregation with the end-only score removes the $N \times N$ quadratic form of the Mahalanobis distance from every window, which reduces the cycles by 36.1% and increases the AUC to 0.868, above the target: with this preprocessing, the end-only score is necessary to meet the target. With 32-dimensional inputs, these two $N \times N$ operations dominate the cost of a window, so that their combination reduces the cycles by 70.7%, consistent with the reduction in the number of operations reported in Section V-C (74%). Thus, MDRS-Circ with the end-only score requires only 27.1 K-cycles per inference, corresponding to 0.32 ms latency, and 24 µJ per inference, with a model of 24 KB.

TABLE VI  
AUTOENCODER AND END-ONLY MDRS-CI R C ON THE NUCLEO-F401RE.
<table><tr><td>MODEL</td><td>CYC. (K)</td><td>LATENCY</td><td>ENERGY</td><td>AUC</td><td>FLASH</td></tr><tr><td>AutoEncoder-INT8, 128 × 5</td><td>2058.7</td><td>24.51 ms</td><td>1828μJ</td><td>0.860</td><td>270 KB</td></tr><tr><td>MDRS-Circ, 128 × 5</td><td>297.9</td><td>3.55 ms</td><td>264μJ</td><td>0.897</td><td>176 KB</td></tr><tr><td>MDRS-Circ, 32 × 1</td><td>27.1</td><td>0.32 ms</td><td>24μJ</td><td>0.869</td><td>24KB</td></tr><tr><td>RATIO (AE ÷ MDRS), 128 × 5</td><td>6.9×</td><td>6.9×</td><td>6.9×</td><td></td><td>1.5×</td></tr><tr><td>RATIO (AE÷MDRS), 32 × 1</td><td>75.9×</td><td>75.9×</td><td>75.9×</td><td></td><td>11.3×</td></tr></table>

## C. Iso-hardware comparison with the reference model

We next compare the most optimized MDRS implementation against the fully-connected autoencoder adopted as the MLPerf Tiny reference model. A direct comparison with publicly available MLPerf Tiny submissions is affected by hardware variability: different submissions use different microcontrollers, clock frequencies, memory systems, DSP extensions, and power domains. This makes it difficult to separate algorithmic efficiency from board-dependent effects. To remove this confounding factor, we ported the baseline AutoEncoder-INT8 to the same NUCLEO-F401RE microcontroller used for investigating MDRS variants and measured it with the procedure described above.

Table VI reports the results. With the same input samples, i.e., the MLPerf Tiny preprocessing, MDRS-Circ requires 297.9 K-cycles per inference, corresponding to 3.55 ms and 264 µJ, while achieving an AUC of 0.897. On the same microcontroller, AutoEncoder-INT8 requires 2 058.7 Kcycles, 24.51 ms, and 1 828 µJ to reach an AUC of 0.860. Under iso-hardware settings and identical inputs, MDRS is thus 6.9× faster and more energy-efficient than the MLPerf Tiny reference model, while improving the AUC and reducing the memory footprint by 1.5×.

With the deployed preprocessing, the advantage grows to 75.9× in time and 11.3× in memory footprint, while MDRS still exceeds the target performance (AUC 0.869), whereas AutoEncoder-INT8 reaches it only with the MLPerf Tiny preprocessing (Section V-C). This gain combines two effects: the lower inference cost of MDRS for the same input samples (6.9×), and the ability of MDRS to operate on much shorter input samples, which reduces its own cost by a further 11.0×. With a new input window every 32 ms (the front-end hop), the autoencoder uses 77% of the real-time budget, whereas MDRS uses 11% with the MLPerf Tiny preprocessing and 1% with the deployed one, leaving the microcontroller almost entirely free for the front-end and other tasks.

## VII. CONCLUSION

Anomaly detection on resource-constrained IoT devices requires meeting detection performance targets while minimizing computational cost and memory footprint. In this study, we investigated the suitability of MDRS method for anomaly detection on resource-constrained IoT devices using the MLPerf Tiny benchmark. We proposed two algorithmic optimizations to improve its efficiency: circular shift for reservoir connectivity (MDRS-Circ) and end-only anomaly scoring. Together, they reduce inference operations by up to 44.2% compared with standard MDRS, while maintaining or improving detection performance.

When compared with several other anomaly detection methods, MDRS and the reference autoencoder are the only methods meeting the MLPerf Tiny performance target of $\mathrm { A U C } ~ \geq ~ 0 . 8 5 0$ . With MLPerf Tiny preprocessing, the optimized MDRS variant achieves an AUC of 0.897 with 6.4× fewer operations than the autoencoder. Exploiting the reservoir’s memory capacity enables single-frame inputs with $n _ { \mathrm { m e l s } } = 3 2$ , achieving an AUC of 0.869 with only 1.2 million operations per recording (13.4× fewer than with the standard preprocessing), whereas the autoencoder falls below the target with such preprocessing.

Deployment on the NUCLEO-F401RE microcontroller confirms these improvements. MDRS achieves 0.32 ms latency, 24 µJ per inference, and a 24 KB model, corresponding to 75.9× lower latency and estimated energy consumption and an 11.3× smaller model than the autoencoder using the standard preprocessing. These results demonstrate the potential of reservoir computing for efficient anomaly detection on resourceconstrained IoT devices. Although the benchmark uses acoustic recordings, MDRS processes their log-mel features as a multivariate time series without audio-specific assumptions, making other sensor modalities a natural direction for future evaluation. Future work will also extend MDRS to other tasks, evaluate on-device training and continual adaptation through incremental updates of the precision matrix, and investigate further algorithmic and hardware-specific optimizations.

## REFERENCES

[1] M. A. Belay, S. S. Blakseth, A. Rasheed, and P. Salvo Rossi, “Unsupervised anomaly detection for IoT-based multivariate time series: Existing solutions, performance analysis and future directions,” Sensors, vol. 23, no. 5, 2023.

[2] P. Warden and D. Situnayake, TinyML. O’Reilly Media, Incorporated: Sebastopol, CA, USA, 2019.

[3] C. Banbury, V. J. Reddi, P. Torelli, J. Holleman, N. Jeffries, C. Kiraly, P. Montino, D. Kanter, S. Ahmed, D. Pau et al., “MLPerf Tiny benchmark,” in Advances in Neural Information Processing Systems (NeurIPS), 2021, pp. 1–15.

[4] M. Yan, C. Huang, P. Bienstman, P. Tino, W. Lin, and J. Sun, “Emerging opportunities and challenges for the future of reservoir computing,” Nature Communications, vol. 15, pp. 1–18, 2024.

[5] M. Lukosevicius and H. Jaeger, “Reservoir computing approaches to recurrent neural network training,” Computer Science Review, vol. 3, no. 3, pp. 127–149, 2009.

[6] H. Tamura, K. Fujiwara, K. Aihara, and G. Tanaka, “Distributional reservoir state analysis for real-time anomaly detection in multivariate time series data,” npj Artificial Intelligence, vol. 2, no. 1, pp. 1–14, 2026.

[7] A. Rodan and P. Tino, “Minimum complexity echo state network,”ˇ IEEE Transactions on Neural Networks, vol. 22, no. 1, pp. 131–144, 2011.

[8] D. Kleyko, E. P. Frady, M. Kheffache, and E. Osipov, “Integer echo state networks: Efficient reservoir computing for digital hardware,” IEEE Transactions on Neural Networks and Learning Systems, vol. 33, no. 4, pp. 1688–1701, 2022.

[9] A. Umar and I. Jordanov, “Anomaly detection in IoT environment using machine learning: A survey,” IEEE Internet of Things Journal, vol. 13, no. 10, pp. 20 477–20 497, 2026.

[10] N. Moustafa, M. Keshk, K.-K. R. Choo, T. Lynar, S. Camtepe, and M. Whitty, “DAD: A distributed anomaly detection system using ensemble one-class statistical learning in edge networks,” Future Generation Computer Systems, vol. 118, pp. 240–251, 2021.

[11] F. T. Liu, K. M. Ting, and Z.-H. Zhou, “Isolation-based anomaly detection,” ACM Transactions on Knowledge Discovery from Data, vol. 6, no. 1, pp. 1–39, 2012.

[12] Y. Meidan, M. Bohadana, Y. Mathov, Y. Mirsky, A. Shabtai, D. Breitenbacher, and Y. Elovici, “N-BaIoT—Network-based detection of IoT botnet attacks using deep autoencoders,” IEEE Pervasive Computing, vol. 17, no. 3, pp. 12–22, 2018.

[13] V. Shanmuganathan and A. Suresh, “LSTM-Markov based efficient anomaly detection algorithm for IoT environment,” Applied Soft Computing, vol. 136, p. 110054, 2023.

[14] S. S. Hammad, D. Iskandaryan, and S. Trilles, “An unsupervised tinyml approach applied to the detection of urban noise anomalies under the smart cities environment,” Internet of Things, vol. 23, p. 100848, 2023.

[15] Y. S. Yap and M. R. Ahmad, “Modified overcomplete autoencoder for anomaly detection based on tinyml,” IEEE Sensors Letters, vol. 8, no. 10, pp. 1–4, 2024.

[16] MLCommons, “MLPerf Tiny v1.3 results,” https://github.com/ mlcommons/tiny results v1.3, 2025.

[17] H. Borras, G. Di Guglielmo, J. Duarte, N. Ghielmetti, B. Hawks, S. Hauck, S.-C. Hsu, R. Kastner, J. Liang, A. Meza et al., “Opensource fpga-ml codesign for the mlperf tiny benchmark,” arXiv preprint arXiv:2206.11791, 2022.

[18] MLCommons, “MLPerf Tiny v0.7 results,” https://github.com/ mlcommons/tiny results v0.7, 2022.

[19] D. V. Buonomano and M. M. Merzenich, “Temporal information transformed into a spatial code by a neural network with realistic properties,” Science, vol. 267, no. 5200, pp. 1028–1030, 1995.

[20] H. Jaeger, “The “echo state” approach to analysing and training recurrent neural networks-with an erratum note,” Bonn, Germany: German National Research Center for Information Technology GMD Technical Report, vol. 148, no. 34, p. 13, 2001.

[21] W. Maass, T. Natschlager, and H. Markram, “Real-time computing¨ without stable states: A new framework for neural computation based on perturbations,” Neural Computation, vol. 14, no. 11, pp. 2531–2560, 2002.

[22] S. Scardapane and D. Wang, “Randomness in neural networks: An overview,” Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, vol. 7, no. 2, 2017.

[23] O. Obst, X. R. Wang, and M. Prokopenko, “Using echo state networks for anomaly detection in underground coal mines,” in International

Conference on Information Processing in Sensor Networks (IPSN), 2008, pp. 219–229.

[24] M. Chang, A. Terzis, and P. Bonnet, “Mote-based online anomaly detection using echo state networks,” in International Conference on Distributed Computing in Sensor Systems (DCOSS), 2009, pp. 72–86.

[25] Q. Chen, A. Zhang, T. Huang, Q. He, and Y. Song, “Imbalanced datasetbased echo state networks for anomaly detection,” Neural Computing and Applications, vol. 32, no. 8, pp. 3685–3694, 2020.

[26] T. Lemos, L. F. Campos, A. Melo, N. Clavijo, R. Soares, M. Camara,ˆ T. Feital, T. Anzai, and J. C. Pinto, “Echo state network based soft sensor for monitoring and fault detection of industrial processes,” Computers & Chemical Engineering, vol. 155, pp. 1–12, 2021.

[27] F. De Vita, G. Nocera, D. Bruneo, and S. K. Das, “A novel echo state network autoencoder for anomaly detection in industrial IoT systems,” IEEE Transactions on Industrial Informatics, vol. 19, no. 8, pp. 8985– 8994, 2022.

[28] J. Kato, G. Tanaka, R. Nakane, and A. Hirose, “Reconstructive reservoir computing for anomaly detection in time-series signals,” Nonlinear Theory and Its Applications, IEICE, vol. 15, no. 1, pp. 183–204, 2024.

[29] A. Blazquez-Garc´ ´ıa, A. Conde, U. Mori, and J. A. Lozano, “A review on outlier/anomaly detection in time series data,” ACM Computing Surveys, vol. 54, no. 3, pp. 1–33, 2021.

[30] A. A. Prater, “Comparison of echo state network output layer classification methods on noisy data,” in International Joint Conference on Neural Networks (IJCNN), 2017, pp. 2644–2651.

[31] Y. Zhao, Z. Nasrullah, and Z. Li, “PyOD: A Python toolbox for scalable outlier detection,” Journal of Machine Learning Research, vol. 20, no. 96, pp. 1–7, 2019.

[32] Y. Koizumi, Y. Kawaguchi, K. Imoto, T. Nakamura, Y. Nikaido, R. Tanabe, H. Purohit, K. Suefusa, T. Endo, M. Yasuda et al., “Description and discussion on DCASE2020 challenge Task2: Unsupervised anomalous sound detection for machine condition monitoring,” arXiv:2006.05822, pp. 1–5, 2020.

[33] J. Huang and C. X. Ling, “Using AUC and accuracy in evaluating learning algorithms,” IEEE Transactions on Knowledge and Data Engineering, vol. 17, no. 3, pp. 299–310, 2005.

[34] D. Terpstra, H. Jagode, H. You, and J. Dongarra, “Collecting performance data with PAPI-C,” in International Workshop on Parallel Tools for High Performance Computing (IPTW), 2010, pp. 157–173.