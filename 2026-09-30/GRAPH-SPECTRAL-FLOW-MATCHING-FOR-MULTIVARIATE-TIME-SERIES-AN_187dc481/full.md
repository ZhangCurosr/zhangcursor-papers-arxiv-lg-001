# GRAPH-SPECTRAL FLOW MATCHING FOR MULTIVARIATE TIME SERIES ANOMALY DETECTION

Zepeng Zhang<sup>1</sup> Jhony H. Giraldo<sup>2</sup> Wenbin Wang<sup>3</sup> Olga Fink<sup>1</sup>

<sup>1</sup> IMOS, EPFL <sup>2</sup> LTCI, Tel´ ecom Paris, IP Paris ´ <sup>3</sup> Independent Researcher

## ABSTRACT

Multivariate time series anomaly detection typically relies on evaluating discrepancies between observations and outputs produced by models trained on normal data. An alternative perspective is to characterize the distribution of normal data through the generative dynamics, i.e., the velocity field, of flow matching models. However, standard flow matching typically adopts linear probability paths that overlook dependencies among variables, leading to a misalignment with the structured data distribution. To address this issue, we propose GRASP, a flow matching framework with a graph-spectral path for multivariate time series anomaly detection. GRASP incorporates graph structure into the probability path by minimizing a fixed-endpoint action that combines kinetic energy with graph Dirichlet energy. This formulation yields a closed-form path based on graph-frequency-dependent hyperbolic interpolation. A velocity predictor trained on normal data then detects anomalies using weighted velocity discrepancies aggregated across source samples, flow times, and graph frequencies. Theoretically, we establish that GRASP is invariant to the choice of Laplacian eigenbasis and decompose its expected oracle anomaly score into bounded endpoint uncertainty and graph-frequency-weighted Fisher discrepancy. Experiments on four benchmarks demonstrate the superior anomaly detection performance of GRASP and validate the effectiveness of its graph-spectral path and weighting mechanism.

## 1 INTRODUCTION

Industrial systems are increasingly monitored by sensor networks that generate large volumes of multivariate time series (MTS) data. Detecting anomalies in these data is critical for preventing failures, reducing downtime, and ensuring reliable operation (Wu et al., 2021; Deng & Hooi, 2021; Fink et al., 2026). However, fault labels are typically scarce and incomplete, as many fault types occur rarely or are not observed during data collection, motivating unsupervised approaches to MTS anomaly detection (Zhang et al., 2019; Audibert et al., 2020; Belay et al., 2023). Existing methods typically learn normal patterns and identify anomalies through prediction errors, reconstruction errors, or representation discrepancies (Jin et al., 2024; Chen & Eldardiry, 2024; Ho et al., 2025). These approaches are effective when abnormal behavior produces clear discrepancies between observations and model outputs. However, in practice, anomalous patterns may remain partially predictable or reconstructible, resulting in weak anomaly signals (Liu et al., 2025; Cho et al., 2025; Zhang et al., 2026b). This motivates anomaly detection methods that go beyond measuring only endpoint-based prediction or reconstruction discrepancies.

Flow matching offers an alternative paradigm by characterizing observations through generative dynamics rather than solely through endpoint outputs, which has been shown to be effective in image anomaly detection (Chen et al., 2026). During training time, a velocity predictor learns to match conditional target velocities along prescribed probability paths connecting source samples to normal observations. During inference time, anomalies are detected by measuring discrepancies between the predicted velocities and conditional target velocities along the entire probability path. The probability path determines both the states at which an observation is evaluated and the conditional target velocity against which it is compared (Lipman et al., 2023; Liu et al., 2023; Albergo et al., 2024). Therefore, the design of the probability path is crucial for flow-matching-based anomaly detection.

MTS variables often exhibit structured dependencies that can be represented by a sensor graph, providing an informative inductive bias for modeling cross-variable interactions (Bronstein et al., 2021; Jin et al., 2024). The graph Laplacian decomposes multivariate signals into graph-frequency modes that describe different patterns of variation across connected nodes (Shuman et al., 2013). Recent work further shows that anomalous behavior can induce heterogeneous energy shifts across graph frequencies, suggesting that different frequency modes carry distinct anomaly-relevant information (Liu et al., 2026). However, existing flow matching models typically adopt a standard linear conditional path, which interpolates all directions identically without accounting for the underlying graph structure (Kollovieh et al., 2025; Albergo et al., 2024). For structured data, this structural mismatch may produce intermediate states and target velocities that are inconsistent with the topology and geometry of the monitored system (Rozada et al., 2026; Fang et al., 2026; Zhang et al., 2026a). This limitation is more pronounced for MTS anomaly detection because the anomaly score depends not only on the observed endpoint, but also on the velocity discrepancies evaluated along the entire path. These observations motivate the design of a graph-informed probability path for flow matching.

In this work, we introduce flow matching with a GRAph-SPectral Path (GRASP), which explicitly incorporates structural information into the conditional probability path. When graph edges encode similarity between normal sensor signals, graph Dirichlet energy provides a natural measure for regularizing variation across connected nodes. We therefore formulate path construction as a fixed-endpoint variational problem that balances kinetic energy and graph Dirichlet energy. The resulting closed-form solution assigns a graph-frequency-dependent hyperbolic interpolation schedule to each frequency mode. Specifically, the zero-frequency modes recover standard linear interpolation, whereas higher-frequency modes undergo progressively stronger contraction at intermediate flow times. In this way, GRASP preserves the prescribed source and data endpoints while imposing a graph-dependent smoothness prior along the probability path. The velocity predictor trained on normal data then detects anomalies by measuring velocity discrepancies along this structured path.

The resulting path induces different conditional uncertainty of the intermediate state and velocityresidual scales across flow times and graph frequencies. To account for these differences, GRASP uses flow-time- and graph-frequency-dependent weights when aggregating velocity discrepancie into an anomaly score. Theoretically, we establish that the node-domain path, target velocity, and resulting anomaly score are invariant to the choice of orthonormal basis within repeated Laplacian eigenspaces. Under the population-optimal velocity predictor, we further show that the expected anomaly score decomposes into an endpoint-uncertainty term and a graph-frequency-weighted Fisher discrepancy. Experiments on four MTS benchmarks demonstrate superior anomaly detection performance of GRASP and the effectiveness of the graph-spectral path and the weighting schedule.

Our contributions are summarized below:

• We derive a closed-form graph-spectral probability path from a fixed-endpoint variational problem combining kinetic energy and graph Dirichlet energy, yielding frequency-dependent hyperbolic interpolation. Based on the graph-spectral path, we introduce an anomaly score that aggregates weighted velocity discrepancies across source samples, flow times, and graph frequencies.

• We establish that GRASP is invariant to different Laplacian eigenbases, and decompose the expected oracle anomaly score into bounded endpoint uncertainty and weighted Fisher discrepancy.

• We conduct experiments on four MTS benchmarks, demonstrating that GRASP achieves competitive or superior performance compared with the baselines. The ablation and sensitivity studies validate the effectiveness of the graph-spectral path design and anomaly scoring mechanism.

Further discussion of related work is provided in Appendix A.

## 2 PRELIMINARIES

Notations and Problem Definition. We use calligraphic letters, such as $x ,$ to represent sets, uppercase bold letters, such as X, to represent matrices, lowercase bold letters, such as x, to represent vectors, and lowercase letters, such as x, to represent scalars. A complete summary of the notation is provided in Appendix B. We consider an MTS observation over N variables. A time-series window is represented as $\mathbf { X } \in \mathbb { R } ^ { N \times R }$ , where its r-th column $\mathbf { x } _ { r } \in \mathbb { R } ^ { N }$ contains the observations of all N variables at timestep $r ,$ and R denotes the window length. The relationships among the variables are captured by a graph $\mathcal { G } = ( \mathcal { N } , \mathcal { E } )$ , where $\mathcal { N }$ and E denote the sets of nodes and edges, respectively. We denote by $\mathbf { L } \in \mathbb { R } ^ { N \times N }$ the symmetric normalized graph Laplacian matrix. Since L is real, symmetric, and positive semidefinite, it admits an eigendecomposition $\mathbf { L } = \Psi \mathbf { \Lambda } \mathbf { \Lambda } \Psi ^ { \top }$ , where $\Psi = [ \dot { \psi } _ { 1 } , \dots , \psi _ { N } ]$ is an orthogonal matrix whose columns are the Laplacian eigenvectors and $\pmb { \Lambda } = \mathrm { d i a g } ( \lambda _ { 1 } , \ldots , \lambda _ { N } )$ contains the corresponding eigenvalues ordered as $\bar { 0 } \leq \lambda _ { 1 } \leq \bar { \cdot } \cdot \cdot \leq \lambda _ { N } \leq 2$

MTS Anomaly Detection. During training, we observe only normal time-series windows sampled from an unknown normal data distribution $p .$ At test time, observations are drawn from a potentially anomalous distribution $q .$ Let $p _ { t }$ and $q _ { t }$ denote the corresponding path marginals at flow time t. Our goal is to learn the generative dynamics of normal data without anomaly samples and construct an anomaly score that quantifies the deviation of a test observation from the learned normal dynamics.

Conditional Flow Matching. Flow matching learns a time-dependent velocity field that transports $\mathbf { X } _ { 0 } \sim p _ { 0 }$ to $\mathbf { X } _ { 1 } \sim p _ { 1 }$ along a prescribed probability path (Lipman et al., 2023; Albergo et al., 2024). In this paper, the entries of $\mathbf { X } _ { 0 }$ are sampled independently from a standard Gaussian distribution, while $p _ { 1 }$ corresponds to the normal data distribution $p .$ . The conditional path between an endpoint pair $( \bar { \bf X } _ { 0 } , { \bf X } _ { 1 } )$ is specified by an interpolation map and the corresponding conditional target velocity:

$$
{ \bf X } _ { t } = \phi _ { t } ( { \bf X } _ { 0 } , { \bf X } _ { 1 } ) , \qquad { \bf U } _ { t } = \frac { \partial } { \partial t } \phi _ { t } ( { \bf X } _ { 0 } , { \bf X } _ { 1 } ) , \qquad t \in [ 0 , 1 ] ,\tag{1}
$$

which satisfies the boundary conditions $\phi _ { 0 } ( \mathbf { X } _ { 0 } , \mathbf { X } _ { 1 } ) = \mathbf { X } _ { 0 }$ and $\phi _ { 1 } ( \mathbf { X } _ { 0 } , \mathbf { X } _ { 1 } ) = \mathbf { X } _ { 1 }$ . Conditional flow matching trains a parameterized velocity field $\mathbf { V } _ { t } ( \mathbf { X } ; \theta )$ to approximate the conditional target velocity by minimizing

$$
\mathcal { L } _ { \mathrm { C F M } } ( \pmb { \theta } ) = \mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] , \mathbf { X } _ { 0 } , \mathbf { X } _ { 1 } } \left[ \lVert \mathbf { V } _ { t } ( \mathbf { X } _ { t } ; \pmb { \theta } ) - \mathbf { U } _ { t } \rVert _ { F } ^ { 2 } \right] ,\tag{2}
$$

where $\boldsymbol { \mathcal { U } } [ 0 , 1 ]$ is the uniform distribution between $_ 0$ and 1. A commonly used probability path is the linear interpolation, where the intermediate state and the conditional target velocity are defined as

$$
\mathbf { X } _ { t } = \phi _ { t } ( \mathbf { X } _ { 0 } , \mathbf { X } _ { 1 } ) = ( 1 - t ) \mathbf { X } _ { 0 } + t \mathbf { X } _ { 1 } \quad \mathrm { a n d } \quad \mathbf { U } _ { t } = \mathbf { X } _ { 1 } - \mathbf { X } _ { 0 } .\tag{3}
$$

This path construction applies the same linear interpolation schedule to all directions in the ambient data space and therefore does not explicitly account for structural dependencies among variables.

Graph Fourier Transform. The eigendecomposition of the graph aplacian matrix provides a graph Fourier basis for signals defined on the corresponding graph ${ \mathcal { G } } .$ . Given an MTS graph signal $\mathbf { X \in }$ $\mathbb { R } ^ { N \times R }$ , its graph Fourier transform and the inverse graph Fourier transform are defined as follows:

$$
\begin{array} { r } { \hat { \mathbf { X } } = \boldsymbol { \Psi } ^ { \top } \mathbf { X } \quad \mathrm { a n d } \quad \mathbf { X } = \boldsymbol { \Psi } \hat { \mathbf { X } } . } \end{array}\tag{4}
$$

The k-th row $\hat { \mathbf { x } } _ { k } \in \mathbb { R } ^ { R }$ of $\hat { \bf X }$ contains the coefficients of the k-th graph-frequency mode across the R timesteps. The Laplacian eigenvalue $\lambda _ { k }$ characterizes the graph-frequency associated with eigenvector $\psi _ { k }$ . Modes associated with smaller eigenvalues vary more smoothly over the graph, whereas modes associated with larger eigenvalues exhibit stronger variation across connected nodes. The graph Dirichlet energy admits a spectral decomposition that separates graph frequency modes:

$$
\mathrm { T r } \left( \mathbf { X } ^ { \top } \mathbf { L } \mathbf { X } \right) = \mathrm { T r } \left( \hat { \mathbf { X } } ^ { \top } \mathbf { A } \hat { \mathbf { X } } \right) = \sum _ { k = 1 } ^ { N } \lambda _ { k } \left\| \hat { \mathbf { x } } _ { k } \right\| _ { 2 } ^ { 2 } .\tag{5}
$$

This decomposition shows that the graph Dirichlet energy penalizes high-frequency components more strongly, thereby providing a natural measure of signal variation across connected nodes.

## 3 FLOW MATCHING WITH GRAPH-SPECTRAL PATH

The standard linear path used in flow matching applies the same interpolation schedule to all directions, implicitly imposing an isotropic transport geometry. Ignoring the structural information in MTS data may produce intermediate states and target velocities that are poorly aligned with the underlying graph geometry. In this section, we develop GRASP, which explicitly incorporates structural information into probability-path design through a graph-informed variational formulation. A schematic comparison between GRASP and standard linear-path flow matching is given in Figure 1.

![](images/166b9043e14463187d7293ec2cd176f0e94edc485e4dfc7c2fce7a9b396111fc.jpg)  
Figure 1: Overview of GRASP for MTS anomaly detection. Left: flow matching with the same linear interpolation for all graph-frequency modes. Middle: GRASP with the graph-spectral path that applies frequency-dependent hyperbolic interpolation. Right: anomaly scoring strategy of GRASP.

## 3.1 GRAPH-SPECTRAL PATH DESIGN

The linear path widely adopted by flow matching models can be characterized as the minimizer of a variational problem with the kinetic energy Lagrangian function defined as follows (Du et al., 2026):

$$
{ \bf \cal T } ^ { \star } = \operatorname * { a r g m i n } _ { { \bf \cal T } ( 0 ) = { \bf X } _ { 0 } , { \bf \cal T } ( 1 ) = { \bf X } _ { 1 } } \int _ { 0 } ^ { 1 } \mathcal { L } _ { \mathrm { l i n } } ( { \bf \cal T } , \dot { \bf \cal T } , t ) d t \quad \mathrm { w i t h } \quad \mathcal { L } _ { \mathrm { l i n } } ( { \bf \cal T } , \dot { \bf \cal T } , t ) = \frac { 1 } { 2 } \| \dot { \bf \cal T } ( t ) \| _ { F } ^ { 2 } ,\tag{6}
$$

where ${ \bf { \cal { \Gamma } } } : [ 0 , 1 ] \to \mathbb { R } ^ { N \times R }$ is the path function. Because this objective penalizes only kinetic energy, it treats all directions in the ambient data space isotropically. Although such a path is suitable for Euclidean data (Liu et al., 2023; Lipman et al., 2023; Albergo et al., 2024), it may produce intermediate states that are poorly aligned with the structure of data supported on non-Euclidean domains (Fang et al., 2026; Wyrwal et al., 2026). In MTS anomaly detection, since velocity discrepancies are evaluated throughout the entire path, such structural misalignment can lead to less informative anomaly signals. Under the assumption that time-varying graph signals vary smoothly over the relation graph, connected nodes tend to exhibit coherent behavior under normal operation (Kalofolias, 2016; Dong et al., 2016; Giraldo et al., 2022). We therefore propose to construct a graph-informed probability path by considering a Lagrangian that combines kinetic energy with graph Dirichlet energy:

$$
\mathcal { L } _ { \mathrm { g r a p h } } ( \mathbf { r } , \dot { \mathbf { r } } , t ) = \frac { 1 } { 2 } \| \dot { \mathbf { r } } ( t ) \| ^ { 2 } + \frac { \tau } { 2 } \mathrm { T r } \left( \mathbf { r } ( t ) ^ { \top } \mathbf { L } \mathbf { r } ( t ) \right) ,\tag{7}
$$

where the graph Dirichlet energy weight parameter $\tau \geq 0$ controls the strength of the graph regularization. The graph Dirichlet energy term penalizes abrupt variations across connected nodes in the graph at intermediate states. As a result, higher graph-frequency modes undergo stronger contraction while preserving the endpoints. The resulting frequency-dependent interpolation therefore treats different graph-frequency modes differently, respecting the phenomenon observed in Liu et al. (2026) that anomalies induce heterogeneous behavior across different graph frequency modes.

To solve the resulting graph-informed variational problem, we first transform the path to the graph Fourier domain as $\hat { \mathbf { \Gamma } } \hat { \mathbf { \Gamma } } ( t ) ~ = ~ \boldsymbol { \Psi } ^ { \top } \mathbf { \Gamma } \mathbf { \Gamma } ( t )$ . Since $\Psi$ is orthonormal and independent of $t ,$ we have $\Vert \dot { \mathbf { \Gamma } } ( t ) \Vert _ { F } ^ { 2 } = \Vert \dot { \hat { \mathbf { \Gamma } } } ( t ) \Vert _ { F } ^ { 2 }$ . The graph-informed variational problem can therefore be written as follows:

$$
\hat { \bf T } ^ { \star } = \operatorname * { a r g m i n } _ { \hat { \Gamma } ( 0 ) = \hat { \bf X } _ { 0 } , \hat { \Gamma } ( 1 ) = \hat { \bf X } _ { 1 } } \int _ { 0 } ^ { 1 } \frac { 1 } { 2 } \sum _ { k } \| \dot { \hat { \gamma } } _ { k } ( t ) \| _ { 2 } ^ { 2 } + \frac { \tau } { 2 } \sum _ { k } \lambda _ { k } \| \hat { \gamma } _ { k } ( t ) \| _ { 2 } ^ { 2 } d t ,\tag{8}
$$

where $\hat { \gamma } _ { k } ( t )$ denote the k-th row of $\hat { \mathbf { T } } ( t )$ . Since both the objective and the boundary conditions are separable across different k, $i . e . ,$ , different graph frequencies, the problem in Equation (8) decomposes into N independent vector-valued variational problems:

$$
\hat { \gamma } _ { k } ^ { \star } = \underset { \hat { \gamma } _ { k } ( 0 ) = \hat { \mathbf { x } } _ { 0 , k } , \hat { \gamma } _ { k } ( 1 ) = \hat { \mathbf { x } } _ { 1 , k } } { \arg \operatorname* { m i n } } \int _ { 0 } ^ { 1 } \frac { 1 } { 2 } \| \dot { \hat { \gamma } } _ { k } ( t ) \| _ { 2 } ^ { 2 } + \frac { \tau } { 2 } \lambda _ { k } \| \hat { \gamma } _ { k } ( t ) \| _ { 2 } ^ { 2 } d t ,\tag{9}
$$

where $\hat { \mathbf { x } } _ { t , k }$ represents the k-th row of $\hat { \mathbf { X } } _ { t }$ . Let $\omega _ { k } = \sqrt { \tau \lambda _ { k } }$ , we have the following result.

Theorem 1 (Graph-spectral path). The variational problem in Equation (9) admits a unique global minimizer given by

$$
\begin{array} { r } { \hat { \gamma } _ { k } ^ { \star } ( t ) = \alpha _ { k } ( t ) \hat { \mathbf { x } } _ { 0 , k } + \beta _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k } , } \end{array}\tag{10}
$$

where

$$
\alpha _ { k } ( t ) = \frac { \sinh ( \omega _ { k } ( 1 - t ) ) } { \sinh ( \omega _ { k } ) } a n d \beta _ { k } ( t ) = \frac { \sinh ( \omega _ { k } t ) } { \sinh ( \omega _ { k } ) } .\tag{11}
$$

The proof of Theorem 1 is provided in Appendix D.1. Figure 2 visualizes the source coefficient $\alpha _ { k } ( t )$ and target coefficient $\beta _ { k } ( t )$ for different values of $\omega _ { k }$ . The source coefficient $\alpha _ { k } ( t )$ decreases from one to zero, whereas the target coefficient $\beta _ { k } ( t )$ increases from zero to one. Unlike the standard linear path, the graph-spectral path assigns different inter-

![](images/8cda52864235ca3ec72b3f510f6b34427b2ce3874b1224c7790b5c5b360a182e.jpg)

![](images/35f188cdf75b7c6190a287df278c6778a8d1b50285dd238d4bf5432235ab5d77.jpg)  
Figure 2: Coefficients $\alpha _ { k } ( t )$ (left) and $\beta _ { k } ( t )$ (right) with different $\omega _ { k }$

polation schedules to different graph frequencies. Specifically, larger values of $\omega _ { k }$ , corresponding to higher graph frequencies or stronger graph regularization, produce stronger contraction at intermediate flow times. The resulting path therefore introduces a graph-dependent structural inductive bias into flow matching. Based on Theorem 1, we obtain the corresponding graph-spectral velocity field:

$$
\dot { \hat { \gamma } } _ { k } ^ { \star } ( t ) = \dot { \alpha } _ { k } ( t ) \hat { \mathbf { x } } _ { 0 , k } + \dot { \beta } _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k } ,\tag{12}
$$

where

$$
\dot { \alpha } _ { k } ( t ) = - \frac { \omega _ { k } \cosh ( \omega _ { k } ( 1 - t ) ) } { \sinh ( \omega _ { k } ) } \quad \mathrm { a n d } \quad \dot { \beta } _ { k } ( t ) = \frac { \omega _ { k } \cosh ( \omega _ { k } t ) } { \sinh ( \omega _ { k } ) } .\tag{13}
$$

Remark 1. When $\omega _ { k } = 0 , \alpha _ { k } ( t )$ and $\beta _ { k } ( t )$ are defined by their continuous limit, $i . e . , \alpha _ { k } ( t )$ becomes 1 − t and $\beta _ { k } ( t )$ becomes t. The corresponding graph-frequency mode therefore degenerates to the standard linear path. Since the normalized graph Laplacian always has at least one zero eigenvalue with its multiplicity equal to the number of connected components (Chung, 1997; Shuman et al., 2013), the case for $\lambda _ { k } = 0 , i . e . , \omega _ { k } = 0 ,$ , always occurs for at least one graph-frequency mode.

The graph-spectral path obtained in Theorem 1 can be written in matrix form as

$$
{ \hat { \mathbf { \Gamma } } } ^ { \star } ( t ) = \mathbf { A } ( t ) { \hat { \mathbf { \Gamma } } } ( 0 ) + \mathbf { B } ( t ) { \hat { \mathbf { \Gamma } } } ( 1 )\tag{14}
$$

with

$$
{ \bf A } ( t ) = \mathrm { d i a g } \left( \alpha _ { 1 } ( t ) , \dots , \alpha _ { N } ( t ) \right) \quad \mathrm { a n d } \quad { \bf B } ( t ) = \mathrm { d i a g } \left( \beta _ { 1 } ( t ) , \dots , \beta _ { N } ( t ) \right) .\tag{15}
$$

Performing the inverse graph Fourier transform gives the corresponding node-domain path:

$$
\mathbf { T } ^ { \star } ( t ) = \Psi \hat { \mathbf { T } } ^ { \star } ( t ) = \Psi \mathbf { A } ( t ) \Psi ^ { \top } \mathbf { T } ( 0 ) + \Psi \mathbf { B } ( t ) \Psi ^ { \top } \mathbf { T } ( 1 ) = \Psi \mathbf { A } ( t ) \Psi ^ { \top } \mathbf { X } _ { 0 } + \Psi \mathbf { B } ( t ) \Psi ^ { \top } \mathbf { X } _ { 1 } .\tag{16}
$$

Then the corresponding node-domain velocity field is defined by

$$
\dot { \mathbf { I } } ^ { \star } ( t ) = \Psi \dot { \mathbf { A } } ( t ) \Psi ^ { \top } \mathbf { X } _ { 0 } + \Psi \dot { \mathbf { B } } ( t ) \Psi ^ { \top } \mathbf { X } _ { 1 } ,\tag{17}
$$

where

$$
{ \dot { \bf A } } ( t ) = \mathrm { d i a g } \left( { \dot { \alpha } } _ { 1 } ( t ) , \dots , { \dot { \alpha } } _ { N } ( t ) \right) \quad \mathrm { a n d } \quad { \dot { \bf B } } ( t ) = \mathrm { d i a g } \left( { \dot { \beta } } _ { 1 } ( t ) , \dots , { \dot { \beta } } _ { N } ( t ) \right) .\tag{18}
$$

## 3.2 MODEL TRAINING AND ANOMALY SCORING

We parameterize the velocity field associated with the graph-spectral path using a lightweight MLP architecture, similar to TSMixer as introduced in (Chen et al., 2023). The details of the model are provided in Appendix C. We train the velocity model in node domain with the regression objective:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G R A S P } } ( \pmb { \theta } ) = \mathbb { E } _ { t \sim \mathcal { U } [ 0 , 1 ] , \mathbf { X } _ { 0 } \sim p _ { 0 } , \mathbf { X } _ { 1 } \sim p } \Big [ \Big \| \mathbf { V } _ { t } ( \mathbf { X } _ { t } ; \pmb { \theta } ) - \dot { \Gamma } ^ { \star } ( t ) \Big \| ^ { 2 } \Big ] . } \end{array}\tag{19}
$$

Suppose that the normal data follows the distribution $p ,$ we define the oracle marginal velocity in the node and graph Fourier domains as

$$
\mathbf { V } _ { t } ^ { p } ( \mathbf { X } ) = \mathbb { E } [ { \dot { \mathbf { T } } } ^ { \star } ( t ) \mid \mathbf { X } _ { t } = \mathbf { X } ] \quad { \mathrm { a n d } } \quad { \hat { \mathbf { V } } } _ { t } ^ { p } ( \mathbf { X } ) = \mathbb { E } [ { \hat { \mathbf { T } } } ^ { \star } ( t ) \mid \mathbf { X } _ { t } = \mathbf { X } ] .\tag{20}
$$

Similarly, the posterior means of the data endpoint in the node and graph Fourier domains are

$$
\mathbf { M } _ { t } ^ { p } ( \mathbf { X } ) = \mathbb { E } [ \Gamma ^ { \star } ( 1 ) \mid \mathbf { X } _ { t } = \mathbf { X } ] \quad { \mathrm { a n d } } \quad \hat { \mathbf { M } } _ { t } ^ { p } ( \mathbf { X } ) = \mathbb { E } [ \hat { \mathbf { \Gamma } } ^ { \star } ( 1 ) \mid \mathbf { X } _ { t } = \mathbf { X } ] .\tag{21}
$$

Let $\hat { \mathbf { v } } _ { t , k } ^ { p }$ and $\hat { \mu } _ { t , k } ^ { p }$ denote the k-th row of $\hat { \mathbf { V } } _ { t } ^ { p } ( \mathbf { X } )$ and $\hat { \mathbf { M } } _ { t } ^ { p } ( \mathbf { X } )$ . Then we have the following result.

Lemma 1 (Graph-spectral residual scaling). For every $t \in ( 0 , 1 )$ and graph-frequency mode $k ,$ the conditional velocity residual and the endpoint posterior residual in the graph Fourier domain satisfy

$$
\dot { \hat { \gamma } } _ { k } ^ { \kappa } ( t ) - \hat { \mathbf { v } } _ { t , k } ^ { p } ( \mathbf { X } ) = \rho _ { k } ( t ) \left( \hat { \mathbf { x } } _ { 1 , k } - \hat { \mu } _ { t , k } ^ { p } ( \mathbf { X } ) \right) \quad w i t h \quad \rho _ { k } ( t ) = \frac { \omega _ { k } } { \sinh ( \omega _ { k } ( 1 - t ) ) } .\tag{22}
$$

The proof of Lemma 1 is provided in Appendix D.2. Note that when $\omega _ { k } = 0 , \rho _ { k } ( t )$ is defined by its continuous limit $\scriptstyle { \frac { 1 } { 1 - t } }$ . Lemma 1 shows that the velocity residual rescales the endpoint residual by a factor that varies across both flow times and graph frequencies.

The conditional velocity residual in the graph Fourier domain can be written in matrix form as

$$
{ \hat { \mathbf { T } } } ^ { \star } ( t ) - { \hat { \mathbf { V } } } _ { t } ^ { p } ( \mathbf { X } ) = \mathbf { P } ( t ) \left( { \hat { \mathbf { X } } } _ { 1 } - { \hat { \mathbf { M } } } _ { t } ^ { p } ( \mathbf { X } ) \right) \quad { \mathrm { w i t h } } \quad \mathbf { P } ( t ) = \operatorname { d i a g } \left( \rho _ { 1 } ( t ) , \ldots , \rho _ { N } ( t ) \right) .\tag{23}
$$

Performing the inverse graph Fourier transform gives the node-domain conditional velocity residual:

$$
\dot { \mathbf { T } } ^ { \star } ( t ) - \mathbf { V } _ { t } ^ { p } ( \mathbf { X } ) = \Psi \mathbf { P } ( t ) \Psi ^ { \top } \left( \mathbf { X } _ { 1 } - \mathbf { M } _ { t } ^ { p } ( \mathbf { X } ) \right) .\tag{24}
$$

According to Theorem 1, we can see that the amount of endpoint information contained in an inter mediate state also varies across flow times and graph frequencies. Consider a rescaled observation:

$$
\frac { 1 } { \beta _ { k } ( t ) } \hat { \gamma } _ { k } ^ { \star } ( t ) = \frac { \alpha _ { k } ( t ) } { \beta _ { k } ( t ) } \hat { \mathbf { x } } _ { 0 , k } + \hat { \mathbf { x } } _ { 1 , k } .\tag{25}
$$

Since the prior follows a standard Gaussian distribution, conditioning on the endpoint gives

$$
\frac { 1 } { \beta _ { k } ( t ) } \hat { \gamma } _ { k } ^ { \star } ( t ) \mid \hat { \mathbf { x } } _ { 1 , k } \sim { \mathcal { N } } \left( \hat { \mathbf { x } } _ { 1 , k } , \frac { \alpha _ { k } ^ { 2 } ( t ) } { \beta _ { k } ^ { 2 } ( t ) } \mathbf { I } \right) .\tag{26}
$$

The conditional variance in Equation (26) quantifies the effective noise level of the rescaled intermediate state with respect to the endpoint. To account for both this noise level and the residual scaling elaborated in Lemma 1, we define the anomaly score weight as the inverse effective noise variance multiplied by the inverse squared residual-scaling factor as follows:

$$
\eta _ { k } ( t ) = \frac { \beta _ { k } ^ { 2 } ( t ) } { \alpha _ { k } ^ { 2 } ( t ) } \frac { 1 } { \rho _ { k } ^ { 2 } ( t ) } = \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) } \frac { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) } { \omega _ { k } ^ { 2 } } = \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \omega _ { k } ^ { 2 } } .\tag{27}
$$

When $\omega _ { k } = 0$ , the weight $\eta _ { k } ( t )$ is defined by its continuous limits $t ^ { 2 }$ . Figure $^ 3$ illustrates the weighting schedule for different values of $\omega$ . From the figure, we observe that the weighting schedule assigns larger coefficients to velocity discrepancies evaluated closer to the data endpoint and at higher graph frequencies. Given a test time window $\mathbf { X } _ { 1 }$ , we independently draw source samples $\mathbf { X } _ { 0 }$ from the standard Gaussian distribution to form a finite prior set $\mathcal { X } _ { \mathrm { a d } }$ . Let $\mathcal { R } _ { \mathrm { a d } } \subset ( 0 , 1 )$ denote the set of flow times used for evaluation. For each source sample and evaluation time, we construct the intermediate state and conditional target velocity using the graph-spectral path. We then aggregate the weighted squared velocity discrepancies across source samples, flow times, and graphfrequency modes to obtain the anomaly score as follows:

![](images/b649938b6866597a3abd0b25ebe837ffc9df9f904fa8d3e43dbf2e0cd5ff3e5c.jpg)  
Figure 3: Weight schedule of $\eta _ { k } ( t )$

$$
\xi ( \mathcal { X } _ { \mathrm { a d } } , \mathcal { R } _ { \mathrm { a d } } , \mathbf { X } _ { 1 } ) = \sum _ { k = 1 } ^ { N } \sum _ { \substack { t \in \mathcal { R } _ { \mathrm { a d } } \mathbf { X } _ { 0 } \in \mathcal { X } _ { \mathrm { a d } } } } \eta _ { k } ( t ) \left\| \hat { \mathbf { v } } _ { t , k } ( \mathbf { X } _ { t } ; \pmb { \theta } ) - \dot { \hat { \gamma } } _ { k } ^ { \star } ( t ) \right\| _ { 2 } ^ { 2 } .\tag{28}
$$

A larger value of $\xi$ indicates a greater deviation from the velocity field learned from normal data and hence provide stronger evidence of anomalous behavior.

## 3.3 THEORETICAL ANALYSES

The graph Fourier basis is not unique when the graph Laplacian has repeated eigenvalues. More precisely, the eigenvectors within each repeated-eigenvalue eigenspace are identifiable only up to an orthogonal transformation (Agaskar & Lu, 2013; Sandryhaila & Moura, 2014; Deri & Moura, 2017). Let $\tilde { \Psi } = \Psi \mathbf { Q }$ denote an alternative orthonormal eigenbasis, where Q is block orthogonal and acts only within eigenspaces associated with repeated eigenvalues. The same graph Laplacian then admits the eigendecomposition $\mathbf { L } = \tilde { \Psi } \pmb { \Lambda } \tilde { \Psi } ^ { \intercal }$ . The following result shows that this potential ambiguity of the graph Fourier basis does not affect the velocity field model, the target velocity, and the anomaly score used by GRASP.

Proposition 1 (Eigenbasis invariance). In GRASP, the node-domain graph-spectral path, its conditional target velocity, and the anomaly score in Equation (28) are all invariant to the choice of orthonormal Laplacian eigenbasis.

The proof of Proposition 1 is provided in Appendix D.3. Under the population-optimal squared flow matching objective, the learned velocity predictor satisfies

$$
\hat { \mathbf { V } } _ { t } ( \mathbf { X } _ { t } ; \pmb { \theta } ) = \hat { \mathbf { V } } _ { t } ^ { p } ( \mathbf { X } _ { t } ) .\tag{29}
$$

The trained velocity model approximates this population-optimal predictor. We therefore analyze the anomaly score obtained with this oracle normal data velocity $\hat { \mathbf { V } } _ { t } ^ { p }$ . Let $p _ { t } ( { \hat { \mathbf { X } } } )$ and $q _ { t } ( { \hat { \mathbf { X } } } )$ denote the graph-spectral path marginals induced by the normal endpoint distribution $p$ and test endpoint distribution q, respectively. We define their spectral score matrices as

$$
\hat { \mathbf { S } } _ { t } ^ { p } ( \hat { \mathbf { X } } ) = \nabla _ { \hat { \mathbf { X } } } \log p _ { t } ( \hat { \mathbf { X } } ) , \quad \hat { \mathbf { S } } _ { t } ^ { q } ( \hat { \mathbf { X } } ) = \nabla _ { \hat { \mathbf { X } } } \log q _ { t } ( \hat { \mathbf { X } } ) ,\tag{30}
$$

whose k-th rows are denoted as $\hat { \mathbf { s } } _ { t , k } ^ { p } ( \hat { \mathbf { X } } )$ and $\hat { \mathbf { s } } _ { t , k } ^ { q } ( \hat { \mathbf { X } } )$ , respectively. We define $\hat { \mu } _ { t , k } ^ { q }$ analogously to $\hat { \mu } _ { t , k } ^ { p }$ . In the following, we show that the expected oracle anomaly score of GRASP admits a decomposition that contains a term that measures deviations between normal and test distributions.

Theorem 2 (Decomposition of the expected oracle anomaly score). Let $\mathcal { X } _ { \mathrm { a d } }$ contain a finite set of i.i.d. source samples from $p _ { 0 } ,$ , independent of $\mathbf { X } _ { 1 } ,$ and let $\mathcal { R } _ { \mathrm { a d } } \subset ( 0 , 1 )$ be a finite set of evaluation times. Consider the anomaly score in Equation (28) evaluated using the oracle normal-data velocity predictor $V _ { t } ^ { p }$ . Its expectation over $\mathcal { X } _ { \mathrm { a d } }$ and $\mathbf { X } _ { 1 } \sim q$ decomposes as

$$
\begin{array} { r l } & { \mathbb { E } _ { \mathcal { X } _ { \mathrm { a d } } , q } \left[ \xi ( \mathcal { X } _ { \mathrm { a d } } , \mathcal { R } _ { \mathrm { a d } } , \mathbf { X } _ { 1 } ) \right] = \displaystyle \sum _ { k = 1 } ^ { N } \displaystyle \sum _ { t \in \mathcal { R } _ { \mathrm { a d } } } \big \lvert \mathcal { X } _ { \mathrm { a d } } \big \rvert \left( \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) } \mathbb { E } _ { p _ { 0 } , q } \left[ \left. \hat { \mathbf { x } } _ { 1 , k } - \hat { \mu } _ { t , k } ^ { q } ( \mathbf { X } ) \right. _ { 2 } ^ { 2 } \right] \right. } \\ & { \qquad \left. + \frac { \sinh ^ { 2 } \left( \omega _ { k } ( 1 - t ) \right) } { \sinh ^ { 2 } ( \omega _ { k } ) } \mathbb { E } _ { q _ { t } } \left[ \left. \hat { \mathbf { s } } _ { t , k } ^ { q } ( \hat { \mathbf { X } } ) - \hat { \mathbf { s } } _ { t , k } ^ { p } ( \hat { \mathbf { X } } ) \right. _ { 2 } ^ { 2 } \right] \right) , } \end{array}\tag{31}
$$

where $\mathbb { E } _ { p _ { 0 } , q }$ denotes expectation over independent $\mathbf { X } _ { 0 } \sim p _ { 0 }$ and $\mathbf { X } _ { 1 } \sim q$ and the coefficients are defined through their continuous limits when $\omega _ { k } = 0$

Corollary 1 (Boundedness of the endpoint-uncertainty term). Under the setting of Theorem 2, assume that $\mathbb { E } _ { \mathbf { X } _ { 1 } \sim q } [ \| \mathbf { X } _ { 1 } \| _ { F } ^ { 2 } ] < \infty$ . Then, for any finite set of evaluation times $\mathcal { R } _ { \mathrm { a d } } \subset ( 0 , 1 )$ , the expected endpoint-uncertainty term in Equation (31) satisfies

$$
\sum _ { k = 1 } ^ { N } \sum _ { t \in \mathcal { R } _ { \mathrm { a d } } } \big | \mathcal { X } _ { \mathrm { a d } } \big | \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) } \mathbb { E } _ { p _ { 0 } , q } \left[ \left\| \hat { \mathbf { x } } _ { 1 , k } - \hat { \pmb { \mu } } _ { t , k } ^ { q } ( \mathbf { X } ) \right\| _ { 2 } ^ { 2 } \right] \leq | \mathcal { X } _ { \mathrm { a d } } | | \mathcal { R } _ { \mathrm { a d } } | N R .\tag{32}
$$

The proof of Theorem 2 and Corollary 1 are provided in Appendix D.4 and Appendix D.5, respectively. The first term in Equation (31) captures graph-frequency-weighted irreducible endpoint uncertainty, which is bounded above as shown in Corollary 1. Whereas the second term in Equation (31) measures the graph-frequency-weighted Fisher discrepancy between the test and normal path marginals, which provides essential signals for anomaly detection. The decomposition therefore identifies a distribution-sensitive component of the proposed anomaly score that explicitly measures deviations between normal and test distributions.

Table 1: Anomaly detection performance comparison.
<table><tr><td rowspan="2">Model</td><td colspan="3">SMAP</td><td rowspan="2"></td><td colspan="2">SMD</td><td colspan="3">CICIDS</td><td colspan="3">SWAN</td></tr><tr><td>PRC</td><td>ROC</td><td>Best-F1 PRC</td><td>ROC</td><td>Best-F1</td><td>PRC</td><td>ROC</td><td>Best-F1</td><td>PRC</td><td>ROC</td><td>Best-F1</td></tr><tr><td>HBOS</td><td>0.1897</td><td>0.5650</td><td>0.2664</td><td>0.2678</td><td>0.7378</td><td>0.3556</td><td>0.2603</td><td>0.5586</td><td>0.3852</td><td>0.2792</td><td>0.5000</td><td>0.4365</td></tr><tr><td>COPOD</td><td>0.1927</td><td>0.5918</td><td>0.2841</td><td>0.2139</td><td>0.7193</td><td>0.2951</td><td>0.2214</td><td>0.5566</td><td>0.3733</td><td>0.2792</td><td>0.5000</td><td>0.4365</td></tr><tr><td>Autoencoder</td><td>0.2277</td><td>0.5859</td><td>0.2991</td><td>0.3613</td><td>0.7297</td><td>0.4214</td><td>0.3147</td><td>0.7276</td><td>0.3938</td><td>0.3294</td><td>0.4944</td><td>0.4371</td></tr><tr><td>USAD</td><td>0.2263</td><td>0.5850</td><td>0.3850</td><td>0.4105</td><td>0.8726</td><td>0.4810</td><td>0.2094</td><td>0.4619</td><td>0.3525</td><td>0.4405</td><td>0.5585</td><td>0.4365</td></tr><tr><td>CNN</td><td>0.2136</td><td>0.6577</td><td>0.3471</td><td>0.3730</td><td>0.7921</td><td>0.4313</td><td>0.2371</td><td>0.5543</td><td>0.3991</td><td>0.3760</td><td>0.4966</td><td>0.4365</td></tr><tr><td>OmniAnomaly</td><td>0.2420</td><td>0.5950</td><td>0.4065</td><td>0.4284</td><td>0.8822</td><td>0.5061</td><td>0.2138</td><td>0.4501</td><td>0.3548</td><td>0.4847</td><td>0.6130</td><td>0.4365</td></tr><tr><td>TranAD</td><td>0.2122</td><td>0.6104</td><td>0.3264</td><td>0.3304</td><td>0.7691</td><td>0.4181</td><td>0.2043</td><td>0.4704</td><td>0.3416</td><td>0.3360</td><td>0.4911</td><td>0.4365</td></tr><tr><td>A-Transformer</td><td>0.1340</td><td>0.5065</td><td>0.2060</td><td>0.0720</td><td>0.5013</td><td>0.1260</td><td>0.2396</td><td>0.4972</td><td>0.3060</td><td>0.2817</td><td>0.5021</td><td>0.4365</td></tr><tr><td>TimesNet</td><td>0.1792</td><td>0.5336</td><td>0.3161</td><td>0.2839</td><td>0.8087</td><td>0.3713</td><td>0.1905</td><td>0.4657</td><td>0.3121</td><td>0.3068</td><td>0.4505</td><td>0.4365</td></tr><tr><td>FITS</td><td>0.1455</td><td>0.5147</td><td>0.2798</td><td>0.2863</td><td>0.8299</td><td>0.4025</td><td>0.1849</td><td>0.4106</td><td>0.3126</td><td>0.3189</td><td>0.4607</td><td>0.4365</td></tr><tr><td>GDN</td><td>0.2197</td><td>0.6099</td><td>0.3394</td><td>0.4006</td><td>0.8014</td><td>0.4640</td><td>0.2926</td><td>0.7017</td><td>0.3805</td><td>0.6423</td><td>0.8356</td><td>0.6432</td></tr><tr><td>GCAD</td><td>0.2144</td><td>0.6348</td><td>0.3471</td><td>0.3569</td><td>0.8416</td><td>0.4535</td><td>0.3636</td><td>0.7191</td><td>0.3844</td><td>0.6445</td><td>0.8140</td><td>0.6231</td></tr><tr><td>CATCH</td><td>0.2450</td><td>0.6078</td><td>0.3664</td><td>0.4763</td><td>0.8968</td><td>0.5057</td><td>0.2866</td><td>0.6162</td><td>0.4023</td><td>0.4638</td><td>0.6218</td><td>0.4533</td></tr><tr><td>GRASP</td><td>0.3149</td><td>0.7015</td><td>0.4178</td><td>0.4845</td><td>0.8567</td><td>0.5087</td><td>0.3996</td><td>0.8108</td><td>0.4615</td><td>0.7556</td><td>0.8758</td><td>0.6886</td></tr></table>

## 4 EXPERIMENTS

We evaluate GRASP on four widely used MTS anomaly detection datasets covering different application domains: the spacecraft telemetry dataset SMAP (Hundman et al., 2018), the IT infrastructure dataset SMD (Su et al., 2019), the cybersecurity dataset CICIDS (Sharafaldin et al., 2018), and the astrophysics dataset SWAN (Angryk et al., 2020). For each dataset, we construct the sensor graph using normal training data. Specifically, we first compute a pairwise distance matrix $\pmb { \Upsilon } \in \mathbb { R } ^ { N \times N }$ from the node features. The distances are converted into edge weights between 0 and 1 using the Gaussian kernel $\begin{array} { r } { \exp ( - \big ( \frac { v _ { i j } } { \iota } \big ) ^ { 2 } ) } \end{array}$ , where the decay rate ι is set to the standard deviation of Υ. The resulting weight matrix is subsequently converted into a binary adjacency matrix with thresholding.

For each MTS, we use the first 80% of the normal training data for anomaly detector optimization and the remaining 20% for validation. We evaluate MTS anomaly detection performance with three metrics: the area under the receiver operating characteristic curve (ROC), the area under the precision-recall curve (PRC), and the Best-F1 score. ROC and PRC measure threshold-independent ranking performance, with PRC being particularly informative for highly imbalanced cases. The Best-F1 score measures point-wise detection performance with the threshold that maximizes the F1 score using test labels. It therefore represents retrospective detection potential rather than perfor mance at a deployable threshold. For each dataset and random seed, we first average the results across its constituent series and then report the mean and standard deviation over ten random seeds.

We compare the proposed GRASP model with thirteen classical and deep-learning-based anomaly detection methods, including HBOS (Goldstein & Dengel, 2012), COPOD (Li et al., 2020), Autoencoder (Sakurada & Yairi, 2014), USAD (Audibert et al., 2020), CNN (Munir et al., 2018), Omni Anomaly (Su et al., 2019), TranAD (Tuli et al., 2022), A-Transformer (Xu et al., 2022), TimesNet (Wu et al., 2023), FITS (Xu et al., 2024), GDN (Deng & Hooi, 2021), GCAD (Liu et al., 2025), and CATCH (Wu et al., 2025). For these baselines, we use the implementations provided in mTSBench Zhou et al. (2026) whenever available. For the rest of models, we use their official implementations.

In the following, we first evaluate the anomaly detection performance of the GRASP model and the baselines in Section 4.1. Then we perform ablation studies to evaluate the individual contributions of the graph-spectral path, score weighting schedule, and data-dependent graph construction in Section 4.2. In Section 4.3, we further analyze how different flow times and graph frequencies contribute to the aggregated anomaly score. Additional implementation details and experimental results are provided in Appendix E. The effects of different velocity-field architectures are studied in Appendix E.6. Appendix E.8 analyzes how the model performs with different numbers of evaluated flow times and source samples, while Appendix E.9 analyzes how the model performs with different graph Dirichlet energy weight τ. The inference efficiency of the model is discussed in Appendix E.10.

## 4.1 ANOMALY DETECTION PERFORMANCE

Table 1 compares GRASP with the baselines on four datasets. The best results are highlighted in bold, and the second-best results are underlined. GRASP achieves the highest PRC and Best-F1 on all four datasets, as well as the highest ROC on three datasets. Its PRC gains are particularly

pronounced on SMAP and SWAN, indicating improved anomaly ranking across different application domains. On SMD, however, CATCH achieves a higher ROC, although GRASP retains the best PRC and Best-F1. Overall, these results show strong and consistent detection performance of GRASP.

## 4.2 ABLATION STUDIES

We conduct ablation studies using three variants of GRASP: 1) GRASP-mean retains the graph-spectral path with a uniform weighting schedule across flow times and graph frequencies; 2) GRASP-ER replaces the datadependent graph in GRASP-mean with an Erdos–R ˝ enyi graph that has the same sparsity´

Table 2: Ablation results on SMAP and SMD.
<table><tr><td>Model</td><td>PRC</td><td>SMAP ROC</td><td>Best-F1</td><td>PRC</td><td>SMD ROC</td><td>Best-F1</td></tr><tr><td>GRASP-lin</td><td>0.2910</td><td>0.6879</td><td>0.3911</td><td>0.4762</td><td>0.8366</td><td>0.4968</td></tr><tr><td>GRASP-ER</td><td>0.2907</td><td>0.6909</td><td>0.3912</td><td>0.4797</td><td>0.8379</td><td>0.4993</td></tr><tr><td>GRASP-mean</td><td>0.2915</td><td>0.6941</td><td>0.3920</td><td>0.4801</td><td>0.8385</td><td>0.5005</td></tr><tr><td>GRASP</td><td>0.3149</td><td>0.7015</td><td>0.4178</td><td>0.4845</td><td>0.8567</td><td>0.5087</td></tr></table>

level; 3) GRASP-lin replaces the graph-spectral path in GRASP-mean with the linear path by setting τ = 0. The results on SMAP and SMD are presented in Table 2, while the results on CICIDS and SWAN are deferred to Appendix E.5. GRASP achieves the best results on all reported metrics. Its improvement over GRASP-mean supports the effectiveness of the proposed weighting schedule. GRASP-mean consistently outperforms GRASP-lin, indicating that the graph-spectral path provides more informative anomaly signals than the linear path. GRASP-mean also consistently outperforms GRASP-ER, suggesting that the data-dependent graph is more informative than a random graph.

## 4.3 FLOW-TIME AND GRAPH-FREQUENCY ANALYSIS

Figure 4 presents the unweighted and weighted anomaly signal attribution across flow times and graph-frequency quantiles on SMAP. Specifically, we equally divide the graph-frequency spectrum into eight quantiles and aggregate the squared velocity residuals within each flowtime and graph-frequency bin. Each heatmap is normalized separately to sum to 100%, so each cell reports its percentage of the total anomaly signal attribution. From the results, we observe that the anomaly evidence is

![](images/2c643a17059ddd5c7f14d3333af0650b6c0efa6cd795c1e548a49c5cc7fe6ae3.jpg)  
Figure 4: Unweighted (left) and weighted (right) anomaly signal attribution across time and frequency on SMAP.

concentrated toward the end of the flow trajectory and concentrated in a narrow graph-frequency band. Applying the weighting schedule further emphasizes the anomaly signals in later flow times and larger graph frequencies. This aligns with the fact that later states carry more information about the data endpoint than earlier states, which are more influenced by the Gaussian source. The dominant frequency band location differs across datasets (results on other datasets are deferred to Appendix E.7), suggesting that anomaly-relevant graph frequencies depend on the specific systems.

## 5 CONCLUSION

In this work, we introduced a graph-informed flow matching approach for MTS anomaly detection named GRASP. It relies on a graph-spectral path which is constructed by minimizing a fixedendpoint action that integrates kinetic energy with graph Dirichlet energy. GRASP detects anomalies by aggregating weighted velocity discrepancies across flow times, source samples, and graph frequencies. We proved that the model is invariant to different Laplacian eigenbasis and the expected oracle anomaly score can be decomposed into bounded endpoint uncertainty and graph-frequencyweighted Fisher discrepancy. Experiments on four datasets validated the effectiveness of GRASP.

Despite its promising results, GRASP has some limitations. First, it relies on a fixed graph, making it unsuitable when we have dynamically evolving graphs. Moreover, computing the graph Fourier basis can become costly as the number of variables increases. Future work could extend GRASP to be compatible with dynamic graphs and develop scalable approximations to graph spectral operations.

## REFERENCES

Ameya Agaskar and Yue M Lu. A spectral graph uncertainty principle. IEEE Transactions on Information Theory, 59(7):4338–4356, 2013.

Michael Samuel Albergo, Mark Goldstein, Nicholas Matthew Boffi, Rajesh Ranganath, and Eric Vanden-Eijnden. Stochastic interpolants with data-dependent couplings. In International Conference on Machine Learning, 2024.

Rafal Angryk, Petrus Martens, Berkay Aydin, Dustin Kempton, Sushant Mahajan, Sunitha Basodi, Azim Ahmadzadeh, Xumin Cai, Soukaina Filali Boubrahimi, Shah Muhammad Hamdi, et al. SWAN-SF: Space weather analytics dataset-solar flares. Harvard Dataverse dataset, pp. 102, 2020.

Julien Audibert, Pietro Michiardi, Fred´ eric Guyard, S´ ebastien Marti, and Maria A Zuluaga. USAD:´ Unsupervised anomaly detection on multivariate time series. In ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2020.

Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

Mohammed Ayalew Belay, Sindre Stenen Blakseth, Adil Rasheed, and Pierluigi Salvo Rossi. Unsupervised anomaly detection for iot-based multivariate time series: Existing solutions, performance analysis and future directions. Sensors, 23(5):2844, 2023.

Jean-David Benamou and Yann Brenier. A computational fluid mechanics solution to the mongekantorovich mass transfer problem. Numerische Mathematik, 84(3):375–393, 2000.

Michael M Bronstein, Joan Bruna, Taco Cohen, and Petar Velickovi ˇ c. Geometric deep learning:´ Grids, groups, graphs, geodesics, and gauges. arXiv preprint arXiv:2104.13478, 2021.

Hongjie Chen and Hoda Eldardiry. Graph time-series modeling in deep learning: A survey. ACM Transactions on Knowledge Discoveryfrom Data, 18(5):1–35, 2024.

Shengzhe Chen, Mehrdad Moradi, Kamran Paynabar, and Hao Yan. Flow mismatching: Unsupervised anomaly detection via velocity discrepancies in flow matching models. arXiv preprint arXiv:2605.23070, 2026.

Si-An Chen, Chun-Liang Li, Sercan O Arik, Nathanael Christian Yoder, and Tomas Pfister. TSMixer: An all-MLP architecture for time series forecast-ing. Transactions on Machine Learning Research, 2023.

Dongchan Cho, Jiho Han, Keumyeong Kang, Minsang Kim, Honggyu Ryu, and Namsoon Jung. Structured temporal causality for interpretable multivariate time series anomaly detection. Ad vances in Neural Information Processing Systems, 2025.

Fan RK Chung. Spectral graph theory, volume 92. American Mathematical Soc., 1997.

Ailin Deng and Bryan Hooi. Graph neural network-based anomaly detection in multivariate time series. In AAAI Conference on Artificial Intelligence, 2021.

Joya A Deri and Jose MF Moura. Spectral projector-based graph fourier transforms.´ IEEE Journal ofSelected Topics in Signal Processing, 11(6):785–795, 2017.

Xiaowen Dong, Dorina Thanou, Pascal Frossard, and Pierre Vandergheynst. Learning laplacian matrix in smooth graph signal representations. IEEE Transactions on Signal Processing, 64(23): 6160–6173, 2016.

Shukai Du, Junzhe Zhang, and Yiming Li. Lagrangian flow matching: A least-action framework for principled path design. arXiv preprint arXiv:2605.15419, 2026.

Dengzhao Fang, Jingtong Gao, Yu Li, Xiangyu Zhao, and Yi Chang. Escaping the euclidean void: Manifold-informed flow matching for sequential recommendation. arXiv preprint arXiv:2607.23762, 2026.

Olga Fink, Vinay Sharma, Ismail Nejjar, Leandro Von Krannichfeldt, Sergei Garmaev, Zepeng Zhang, Amaury Wei, Gaetan Frusque, Florent Forest, Mengjie Zhao, et al. From physics to machine learning and back: Part I-learning with inductive biases in prognostics and health management. Reliability Engineering & System Safety, pp. 112213, 2026.

Jhony H Giraldo, Arif Mahmood, Belmar Garcia-Garcia, Dorina Thanou, and Thierry Bouwmans. Reconstruction of time-varying graph signals via sobolev smoothness. IEEE Transactions on Signal and Information Processing over Networks, 8:201–214, 2022.

Markus Goldstein and Andreas Dengel. Histogram-based outlier score (HBOS): A fast unsupervised anomaly detection algorithm. KI-2012: poster and demo track, 1:59–63, 2012.

Thi Kieu Khanh Ho, Ali Karami, and Narges Armanfard. Graph anomaly detection in time series: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Kyle Hundman, Valentino Constantinou, Christopher Laporte, Ian Colwell, and Tom Soderstrom. Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. In ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2018.

Ming Jin, Huan Yee Koh, Qingsong Wen, Daniele Zambon, Cesare Alippi, Geoffrey I Webb, Irwin King, and Shirui Pan. A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):10466–10485, 2024.

Vassilis Kalofolias. How to learn a graph from smooth signals. In International Conference on Artificial Intelligence and Statistics, 2016.

Thomas N. Kipf and Max Welling. Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations, 2017.

Marcel Kollovieh, Marten Lienen, David Ludke, Leo Schwinn, and Stephan G¨ unnemann. Flow¨ matching with gaussian process priors for probabilistic time series forecasting. In International Conference on Learning Representations, 2025.

Jinghan Li, Yuan Gao, Jinda Lu, Junfeng Fang, Congcong Wen, Hui Lin, and Xiang Wang. Diff-GAD: A diffusion-based unsupervised graph anomaly detector. In International Conference on Learning Representations, 2025.

Zheng Li, Yue Zhao, Nicola Botta, Cezar Ionescu, and Xiyang Hu. COPOD: Copula-based outlier detection. In IEEE international conference on data mining, 2020.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matthew Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023.

Xingchao Liu, Chengyue Gong, and qiang liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023.

Yilin Liu, Hongchao Zhang, Ahmad Taha, Taylor T Johnson, and Meiyi Ma. Modeling spectral energy shifts in spatio-temporal graph anomaly detection. In International Conference on Machine Learning, 2026.

Zehao Liu, Mengzhou Gao, and Pengfei Jiao. GCAD: Anomaly detection in multivariate time series from the perspective of granger causality. In AAAI Conference on Artificial Intelligence, 2025.

Mohsin Munir, Shoaib Ahmed Siddiqui, Andreas Dengel, and Sheraz Ahmed. DeepAnT: A deep learning approach for unsupervised anomaly detection in time series. IEEE Access, 7:1991–2005, 2018.

Sergio Rozada, KB Vimal, Andrea Cavallo, Antonio G Marques, Hadi Jamali-Rad, and Elvin Isufi. Graph-aware diffusion for signal generation. In IEEE International Conference on Acoustics, Speech and Signal Processing, 2026.

Mayu Sakurada and Takehisa Yairi. Anomaly detection using autoencoders with nonlinear dimensionality reduction. In Workshop on Machine Learning for Sensory Data Analysis, 2014.

Aliaksei Sandryhaila and Jose MF Moura. Discrete signal processing on graphs: Frequency analysis. IEEE Transactions on Signal Processing, 62(12):3042–3054, 2014.

Iman Sharafaldin, Arash Habibi Lashkari, Ali A Ghorbani, et al. Toward generating a new intrusion detection dataset and intrusion traffic characterization. ICISSP, 1(2018):108–116, 2018.

David I Shuman, Sunil K Narang, Pascal Frossard, Antonio Ortega, and Pierre Vandergheynst. The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. IEEE Signal Processing Magazine, 30(3):83–98, 2013.

Ya Su, Youjian Zhao, Chenhao Niu, Rong Liu, Wei Sun, and Dan Pei. Robust anomaly detection for multivariate time series through stochastic recurrent neural network. In ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019.

Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research, 2024.

Shreshth Tuli, Giuliano Casale, and Nicholas R. Jennings. TranAD: Deep transformer networks for anomaly detection in multivariate time series data. VLDB Endowment, 15(6):1201–1214, 2022.

Yigit Berkay Uslu, Samar Hadou, Sergio Rozada, Shirin Saeedi Bidokhti, and Alejandro Ribeiro.˘ Graph signal generative diffusion models. In IEEE International Conference on Acoustics, Speech and Signal Processing, 2026.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

Cedric Villani.´ Optimal transport: old and new, volume 338 of Grundlehren der Mathematischen Wissenschaften. Springer, Berlin, Heidelberg, 2009.

Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. TimesNet: Temporal 2D-variation modeling for general time series analysis. In International Conference on Learning Representations, 2023.

Xingjian Wu, Xiangfei Qiu, Zhengyu Li, Yihang Wang, Jilin Hu, Chenjuan Guo, Hui Xiong, and Bin Yang. CATCH: Channel-aware multivariate time series anomaly detection via frequency patching. In International Conference on Learning Representations, 2025.

Yulei Wu, Hong-Ning Dai, and Haina Tang. Graph neural networks for anomaly detection in industrial internet of things. IEEE Internet ofThings Journal, 9(12):9214–9231, 2021.

Kacper Wyrwal, Ismail Ilkan Ceylan, and Alexander Tong. Topological flow matching. In International Conference on Learning Representations, 2026.

Jiehui Xu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Anomaly transformer: Time series anomaly detection with association discrepancy. In International Conference on Learning Representations, 2022.

Zhijian Xu, Ailing Zeng, and Qiang Xu. FITS: Modeling time series with 10k parameters. In International Conference on Learning Representations, 2024.

Maosheng Yang. Topological schrodinger bridge matching. In¨ International Conference on Learn ing Representations, 2025.

Yiyuan Yang, Chaoli Zhang, Tian Zhou, Qingsong Wen, and Liang Sun. DCdetector: Dual attention contrastive representation learning for time series anomaly detection. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023.

Chuxu Zhang, Dongjin Song, Yuncong Chen, Xinyang Feng, Cristian Lumezanu, Wei Cheng, Jingchao Ni, Bo Zong, Haifeng Chen, and Nitesh V Chawla. A deep neural network for unsupervised anomaly detection and diagnosis in multivariate time series data. In AAAI Conference on Artificial Intelligence, 2019.

Zepeng Zhang, Aref Einizade, Jhony H Giraldo, and Olga Fink. Spatiotemporal imputation with graph-informed flow matching. In International Conference on Machine Learning, 2026a.

Zepeng Zhang, Fuad Khuri, Keivan Faghih Niresi, and Olga Fink. GSLAD: Prototype-regularized graph structure learning for multivariate time series anomaly detection. arXiv preprint arXiv:2609.15483, 2026b.

Hang Zhao, Yujing Wang, Juanyong Duan, Congrui Huang, Defu Cao, Yunhai Tong, Bixiong Xu, Jing Bai, Jie Tong, and Qi Zhang. Multivariate time-series anomaly detection via graph attention network. In IEEE International Conference on Data Mining, 2020.

Mengjie Zhao and Olga Fink. DyEdgeGAT: Dynamic edge via graph attention for early fault detection in IIoT systems. IEEE Internet ofThings Journal, 11(13):22950–22965, 2024.

Xiaona Zhou, Constantin Brif, and Ismini Lourentzou. mTSBench: Benchmarking multivariate time series anomaly detection and model selection at scale. Transactions on Machine Learning Research, 2026.

## Appendix

A Related Work 15   
B Notation Summary 15   
C Velocity-Field Architecture 15   
D Proofs of Theoretical Results 17   
D.1 Proof of Theorem 1 17   
D.2 Proof of Lemma 1 17   
D.3 Proof of Proposition 1 18   
D.4 Proof of Theorem 2 20   
D.5 Proof of Corollary 1 21   
E Experimental Details and Additional Results 22   
E.1 Datasets 22   
E.2 Baselines . 22   
E.3 Implementation Details 23   
E.4 Additional MTS Anomaly Detection Results 23   
E.5 Additional Ablation Results . 24   
E.6 Comparison of Velocity-Field Architectures 24   
E.7 Additional Flow-Time and Graph-Frequency Analysis 24   
E.8 Sensitivity to Flow-Time Evaluations and Source Samples 25   
E.9 Sensitivity to the Graph Dirichlet-Energy Weight . 26   
E.10 Inference Efficiency 27

## A RELATED WORK

Graph-based MTS anomaly detection. Unsupervised MTS anomaly detection methods learn patterns of normal system operation and identify observations that deviate from them. A common approach is to compute anomaly scores based on forecasting or reconstruction errors. For example, MTAD-GAT (Zhao et al., 2020) combines forecasting and reconstruction objectives during training and evaluates discrepancies between the observed signal and model outputs at inference. Graph-based approaches further incorporate dependencies among variables into these objectives. GDN (Deng & Hooi, 2021) learns sensor relationships among variables to support forecastingbased detection, whereas DyEdgeGAT (Zhao & Fink, 2024) constructs input-dependent graphs for reconstruction-based detection. Other approaches characterize anomalies through association discrepancies (Xu et al., 2022), differences in learned representations (Yang et al., 2023), deviations in the frequency domain (Wu et al., 2025), or shifts in graph-spectral energy Liu et al. (2026). More recently, diffusion-based methods have been introduced to reconstruct normal graph signals (Li et al., 2025). However, these forecasting-, reconstruction-, and diffusion-based methods primarily evaluate discrepancies at the predicted or reconstructed endpoint. Consequently, subtle or partially predictable anomalies may remain difficult to detect when they produce only weak endpoint discrepancies.

Flow matching and conditional path design. Flow matching learns a neural velocity field by regressing against conditional target velocities along prescribed probability paths that connect a simple source distribution to the data distribution. The choice of probability path is essential because it determines both the intermediate states and the target velocities used for training and evaluation (Du et al., 2026). Existing constructions, including rectified paths (Liu et al., 2023) and optimal transport-based paths (Tong et al., 2024), typically use straight-line interpolation between coupled endpoints. From a variational perspective, the standard linear conditional path minimizes the action associated with a kinetic-energy Lagrangian under fixed-endpoint constraints (Benamou & Brenier, 2000; Villani, 2009), and it has become a standard choice in flow matching (Lipman et al., 2023). However, this construction depends only on the endpoints and does not explicitly incorporate the geometry or relational structure of the data. The resulting intermediate states may therefore be poorly aligned with data supported on non-Euclidean domains (Fang et al., 2026).

Graph-aware generative modeling. Several recent methods investigate incorporating graph structural information into diffusion or other generative frameworks. Graph-aware diffusion models introduce structural information through graph-based denoising architectures (Uslu et al., 2026) or graph-aware forward noising schedules (Rozada et al., 2026). Yang (2025) extends Schrodinger ¨ bridge matching to topological domains such as graphs and simplicial complexes, while Wyrwal et al. (2026) incorporates topological information into the reference process through a Laplacianderived drift. These methods introduce graph structure into the denoising model or a predefined stochastic reference process. In contrast, GRASP derives the conditional probability path directly as the solution to a fixed-endpoint variational problem that combines kinetic energy with graph Dirichlet energy. This formulation yields a closed-form, graph-frequency-dependent path specifically designed for velocity-based MTS anomaly detection.

## B NOTATION SUMMARY

Table 3 summarizes the notation used throughout the paper.

## C VELOCITY-FIELD ARCHITECTURE

Given an intermediate state ${ \bf X } _ { t } \in \mathbb { R } ^ { N \times R }$ and its flow time t, we first encode t using a sinusoidal flow-time embedding function $\mathbf { e } ( t )$ . The intermediate state and time embedding are then concate nated and mapped to an initial hidden representation through an input projection:

$$
{ \bf H } ^ { ( 0 ) } = \phi _ { \mathrm { i n } } \left( { \bf X } _ { t } , e ( t ) \right) ,\tag{33}
$$

where $\phi _ { \mathrm { i n } }$ denote the input projection.

Following the TSMixer architecture (Chen et al., 2023), we process the hidden representation using $L _ { \mathrm { T S M i x e r } }$ residual mixing blocks. Each block alternates between temporal mixing and cross-variable

Table 3: Summary of notation used throughout the paper.
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $N , R$ </td><td>Number of variables and length of a time-series window.</td></tr><tr><td> $\mathcal { G } = ( \mathcal { N } , \mathcal { E } )$ </td><td>Sensor graph with node set  $\mathcal { N }$  and edge set  $\varepsilon .$ </td></tr><tr><td> $\mathbf { X } \in \dot { \mathbb { R } } ^ { N \times R }$ </td><td>Multivariate time-series window.</td></tr><tr><td> $\mathbf { L }$ </td><td>Symmetric normalized graph Laplacian.</td></tr><tr><td> $\Psi , \Lambda$ </td><td>Laplacian eigenvector and eigenvalue matrices satisfying  ${ \bf L } = \Psi { \bf \Lambda } \Psi ^ { \top } .$ </td></tr><tr><td> $\lambda _ { k } , \psi _ { k }$ </td><td>Laplacian eigenvalue and eigenvector associated with graph-frequency mode</td></tr><tr><td> $\hat { \mathbf X } = \Psi ^ { \top } \mathbf X$ </td><td>Graph Fourier transform of  $\mathbf { x } .$ </td></tr><tr><td> $\hat { \mathbf { x } } _ { t , k }$ </td><td>The k-th row of  $\hat { \mathbf { X } } ,$  representing graph-frequency mode k across the time window.</td></tr><tr><td> $t \in [ 0 , 1 ]$ </td><td>Flow time.</td></tr><tr><td> $\mathbf { X } _ { 0 } , \mathbf { X } _ { 1 } , \mathbf { X } _ { t }$ </td><td>Source endpoint, data endpoint, and intermediate state.</td></tr><tr><td> $p _ { 0 }$ </td><td>Standard Gaussian source distribution.</td></tr><tr><td> $p , \ q$ </td><td>Normal and test endpoint distributions.</td></tr><tr><td> $p _ { t } , \ q _ { t }$ </td><td>Intermediate-state distributions induced by endpoint distributions  $p$  and  $q .$ </td></tr><tr><td> $\phi _ { t }$ </td><td>Conditional interpolation map between  $\mathbf { X } _ { 0 }$  and  $\mathbf { X } _ { 1 }$ </td></tr><tr><td> $\mathbf { U } _ { t }$ </td><td>Conditional target velocity induced by  $\phi _ { t }$ </td></tr><tr><td> $\mathbf { V } _ { t } ( \mathbf { X } ; \mathbf { \theta } )$ </td><td>Learned time-dependent velocity field.</td></tr><tr><td> $\Gamma ^ { \star } ( t ) , \dot { \Gamma } ^ { \star } ( t )$ </td><td>Optimal graph-spectral path and its velocity.</td></tr><tr><td> $\hat { \gamma } _ { k } ^ { \star } ( t )$ </td><td>The k-th graph-frequency component of  $\Gamma ^ { \star } ( t )$ </td></tr><tr><td> $\tau$ </td><td>Weight of the graph Dirichlet energy.  $k .$ </td></tr><tr><td> $\omega _ { k } = \sqrt { \tau \lambda _ { k } }$ </td><td>Graph-spectral parameter associated with graph-frequency mode</td></tr><tr><td> $\alpha _ { k } ( t ) , \beta _ { k } ( t )$ </td><td>Source and target interpolation coefficients for graph-frequency mode  $k .$ </td></tr><tr><td> ${ \bf A } ( t ) , { \bf B } ( t )$ </td><td>Diagonal matrices collecting  $\alpha _ { k } ( t )$  and  $\beta _ { k } ( t )$ </td></tr><tr><td> ${ \mathbf { V } } _ { t } ^ { p } ( { \mathbf { X } } )$ </td><td>Oracle marginal velocity field induced by the normal endpoint distribution.</td></tr><tr><td> $\hat { \mathbf { v } } _ { t , k } ^ { p }$ </td><td>The k-th row of  ${ \mathbf { V } } _ { t } ^ { p } ( { \mathbf { X } } )$ </td></tr><tr><td> ${ \bf M } _ { t } ^ { p } ( { \bf X } ) , { \bf M } _ { t } ^ { q } ( { \bf X } )$ </td><td>Endpoint posterior means under  $p$  and  $q .$ </td></tr><tr><td> $\hat { \mu } _ { t , k } ^ { p } , \hat { \mu } _ { t , k } ^ { q }$ </td><td>Endpoint posterior mean of graph-frequency mode  $k$  under distribution  $p$  and  $q .$ </td></tr><tr><td> $\rho _ { k } ( t )$ </td><td>Scaling factor relating the conditional velocity residual to the endpoint posterior residual.</td></tr><tr><td> $\eta _ { k } ( t )$ </td><td>Anomaly-score weight for graph-frequency mode k at flow-time t.</td></tr><tr><td> $\mathcal { X } _ { \mathrm { a d } } , \mathcal { R } _ { \mathrm { a d } }$ </td><td>Sets of source samples and evaluation flow times used for anomaly scoring.</td></tr><tr><td> $\xi ( \mathcal { X } _ { \mathrm { a d } } , \mathcal { R } _ { \mathrm { a d } } , \mathbf { X } _ { 1 } )$ </td><td>Anomaly score of test window  $\mathbf { X } _ { 1 }$ </td></tr><tr><td> $\hat { \mathbf { S } } _ { t } ^ { p } , \hat { \mathbf { S } } _ { t } ^ { q }$ </td><td>Spectral score matrices of the normal and test path marginals.</td></tr><tr><td> $\mathbf { d } _ { t , k }$ </td><td>Endpoint posterior residual for graph-frequency mode  $k .$ </td></tr><tr><td> $\delta _ { t , k }$ </td><td>Difference between the test and normal spectral scores of graph-frequency mode k.</td></tr><tr><td> $\tilde { \Psi } = \Psi \mathbf { Q }$ </td><td>Alternative Laplacian eigenbasis within repeated-eigenvalue eigenspaces.</td></tr><tr><td> $\mathbf { Q }$ </td><td>Block-diagonal orthogonal transformation matrix acting within repeated-eigenvalue eigenspaces.</td></tr></table>

mixing as follows:

$$
\tilde { \mathbf { H } } ^ { ( \ell ) } = \mathbf { H } ^ { ( \ell - 1 ) } + \mathcal { M } _ { \mathrm { t i m e } } ^ { ( \ell ) } \left( \mathrm { N o r m } ( \mathbf { H } ^ { ( \ell - 1 ) } ) \right) ,\tag{34}
$$

$$
\mathbf { H } ^ { ( \ell ) } = \tilde { \mathbf { H } } ^ { ( \ell ) } + \mathcal { M } _ { \mathrm { v a r } } ^ { ( \ell ) } \left( \mathrm { N o r m } ( \tilde { \mathbf { H } } ^ { ( \ell ) } ) \right) ,\tag{35}
$$

for $\ell = 1 , \ldots , L _ { \mathrm { T S M i x e r } }$ . Here, Norm represents layer normalization (Ba et al., 2016), $\mathcal { M } _ { \mathrm { t i m e } } ^ { ( \ell ) }$ is an MLP applied along the temporal dimension and shared across variables, while $\mathcal { M } _ { \mathrm { v a r } } ^ { ( \ell ) }$ is an MLP applied along the variable dimension and shared across time steps. Each mixing MLP consists of two linear layers with a nonlinear activation function ReLU and dropout.

Finally, an output projection maps the hidden representation back to the original data space:

$$
\mathbf { V } _ { t } ( \mathbf { X } _ { t } ; \pmb { \theta } ) = \phi _ { \mathrm { o u t } } \left( \mathbf { H } ^ { ( L _ { \mathrm { T S M i x e r } } ) } \right) \in \mathbb { R } ^ { N \times R } .\tag{36}
$$

The alternating mixing operations enable the velocity predictor to capture temporal dependencies within individual variables and interactions across variables while retaining a lightweight, fully MLP-based architecture. Importantly, in GRASP, graph structure is incorporated through the probability path and conditional target velocity rather than through the velocity-field architecture itself.

## D PROOFS OF THEORETICAL RESULTS

## D.1 PROOF OF THEOREM 1

Proof. The Euler-Lagrange equation corresponds to the variational problem in Equation (9) is

$$
\omega _ { k } ^ { 2 } \hat { \gamma } _ { k } ( t ) - \ddot { \hat { \gamma } } _ { k } ( t ) = \mathbf { 0 } ,\tag{37}
$$

whose general solution is

$$
\hat { \gamma } _ { k } ( t ) = { \bf c } _ { 1 , k } e ^ { \omega _ { k } t } + { \bf c } _ { 2 , k } e ^ { - \omega _ { k } t }\tag{38}
$$

with some $\mathbf { c } _ { 1 , k }$ and $\mathbf { c } _ { 2 , k }$ . Since the hyperbolic functions satisfy

$$
\sinh ( \omega _ { k } t ) = { \frac { e ^ { \omega _ { k } t } - e ^ { - \omega _ { k } t } } { 2 } } , \quad \cosh ( \omega _ { k } t ) = { \frac { e ^ { \omega _ { k } t } + e ^ { - \omega _ { k } t } } { 2 } } ,\tag{39}
$$

we can further transform it into

$$
\begin{array} { r } { \hat { \gamma } _ { k } ( t ) = ( \mathbf { c } _ { 1 , k } - \mathbf { c } _ { 2 , k } ) \sinh ( \omega _ { k } t ) + ( \mathbf { c } _ { 1 , k } + \mathbf { c } _ { 2 , k } ) \cosh ( \omega _ { k } t ) . } \end{array}\tag{40}
$$

Considering the boundary condition, we have

$$
\begin{array} { r } { \hat { \gamma } _ { k } ^ { \star } ( t ) = \frac { \hat { \mathbf { x } } _ { 1 , k } - \cosh \left( \omega _ { k } \right) \hat { \mathbf { x } } _ { 0 , k } } { \sinh \left( \omega _ { k } \right) } \sinh ( \omega _ { k } t ) + \cosh ( \omega _ { k } t ) \hat { \mathbf { x } } _ { 0 , k } , } \end{array}\tag{41}
$$

which can be rewritten as

$$
\begin{array} { l } { \hat { \gamma } _ { k } ^ { \star } ( t ) = \left( \cosh ( \omega _ { k } t ) - \frac { \cosh ( \omega _ { k } ) \sinh ( \omega _ { k } t ) } { \sinh ( \omega _ { k } ) } \right) \hat { \mathbf { x } } _ { 0 , k } + \frac { \sinh ( \omega _ { k } t ) } { \sinh ( \omega _ { k } ) } \hat { \mathbf { x } } _ { 1 , k } } \\ { = \frac { \sinh \left( \omega _ { k } \left( 1 - t \right) \right) } { \sinh \left( \omega _ { k } \right) } \hat { \mathbf { x } } _ { 0 , k } + \frac { \sinh \left( \omega _ { k } t \right) } { \sinh \left( \omega _ { k } \right) } \hat { \mathbf { x } } _ { 1 , k } . } \end{array}\tag{42}
$$

When $\omega _ { k } = 0$ , the Euler-Lagrange equation reduces to

$$
\ddot { \hat { \gamma } } _ { k } ( t ) = \mathbf { 0 } .\tag{43}
$$

Under the same boundary conditions, its solution is

$$
\hat { \gamma } _ { k } ^ { \star } ( t ) = ( 1 - t ) \hat { \mathbf { x } } _ { 0 , k } + t \hat { \mathbf { x } } _ { 1 , k } ,\tag{44}
$$

which coincides with the continuous limit of Equation (42) as $\omega _ { k }  0 .$

The kinetic-energy term in Equation (9) is strictly convex over the set of paths satisfying the fixedendpoint constraints. Indeed, two admissible paths with identical derivatives can differ only by a constant, which must be zero because they share the same endpoints. Moreover, the graph-frequency regularization term is convex since $\omega _ { k } ^ { 2 } \doteq \tau \lambda _ { k } \ge 0$ . Therefore, the complete objective in Equation (9) is strictly convex over its feasible set. Since the path in Equation (42) satisfies both the Euler–Lagrange equation and the boundary conditions, it is the unique global minimizer of the variational problem defined in Equation (9), through which the proof is completed. □

## D.2 PROOF OF LEMMA 1

Proof. According to Theorem 1, the k-th graph-frequency component of the conditional path is

$$
\begin{array} { r } { \hat { \gamma } _ { k } ^ { \star } ( t ) = \alpha _ { k } ( t ) \hat { \mathbf { x } } _ { 0 , k } + \beta _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k } . } \end{array}\tag{45}
$$

For $t \in ( 0 , 1 )$ , we have $\alpha _ { k } ( t ) > 0$ . Then, we can obtain

$$
\hat { \bf x } _ { 0 , k } = \frac { 1 } { \alpha _ { k } ( t ) } \left( \hat { \gamma } _ { k } ^ { \star } ( t ) - \beta _ { k } ( t ) \hat { \bf x } _ { 1 , k } \right) .\tag{46}
$$

Substituting Equation (46) into the conditional target velocity in Equation (12) gives

$$
\dot { \hat { \gamma } } _ { k } ^ { \star } ( t ) = \frac { \dot { \alpha } _ { k } ( t ) } { \alpha _ { k } ( t ) } \left( \hat { \gamma } _ { k } ^ { \star } ( t ) - \beta _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k } \right) + \dot { \beta } _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k } .\tag{47}
$$

which can be rewritten as

$$
\dot { \hat { \gamma } } _ { k } ^ { \star } ( t ) = \frac { \dot { \alpha } _ { k } ( t ) } { \alpha _ { k } ( t ) } \hat { \gamma } _ { k } ^ { \star } ( t ) + \rho _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k }\tag{48}
$$

with

$$
\rho _ { k } ( t ) = \dot { \beta } _ { k } ( t ) - \frac { \dot { \alpha } _ { k } ( t ) } { \alpha _ { k } ( t ) } \beta _ { k } ( t ) .\tag{49}
$$

For $\omega _ { k } >$ , substituting the expressions for $\alpha _ { k } ( t )$ and $\beta _ { k } ( t )$ yields

$$
\begin{array} { c l } { \rho _ { k } ( t ) = \frac { \omega _ { k } \cosh ( \omega _ { k } t ) } { \sinh ( \omega _ { k } ) } + \frac { \omega _ { k } \cosh ( \omega _ { k } ( 1 - t ) ) } { \sinh ( \omega _ { k } ) } \frac { \sinh ( \omega _ { k } ) } { \sinh ( \omega _ { k } ( 1 - t ) ) } \frac { \sinh ( \omega _ { k } t ) } { \sinh ( \omega _ { k } ) } } \\ { = \frac { \omega _ { k } } { \sinh \left( \omega _ { k } \right) } \frac { \cosh \left( \omega _ { k } t \right) \sinh \left( \omega _ { k } \left( 1 - t \right) \right) + \cosh \left( \omega _ { k } \left( 1 - t \right) \right) \sinh \left( \omega _ { k } t \right) } { \sinh \left( \omega _ { k } \left( 1 - t \right) \right) } } \\ { = \frac { \omega _ { k } } { \sinh \left( \omega _ { k } \right) } \frac { \sinh \left( \omega _ { k } t + \omega _ { k } \left( 1 - t \right) \right) } { \sinh \left( \omega _ { k } \left( 1 - t \right) \right) } } \\ { = \frac { \omega _ { k } } { \sinh \left( \omega _ { k } \left( 1 - t \right) \right) } . } \end{array}\tag{50}
$$

Based on the definition of the oracle marginal velocity, we have

$$
\begin{array} { r l } & { \hat { \mathbf { v } } _ { t , k } ^ { p } ( \mathbf { X } ) = \mathbb { E } [ \hat { \gamma } _ { k } ^ { \star } ( t ) \mid \mathbf { X } _ { t } = \mathbf { X } ] } \\ & { \quad \quad \quad \quad = \mathbb { E } [ \frac { \hat { \alpha } _ { k } ( t ) } { \alpha _ { k } ( t ) } \hat { \gamma } _ { k } ^ { \star } ( t ) + \rho _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k } \mid \mathbf { X } _ { t } = \mathbf { X } ] } \\ & { \quad \quad \quad = \mathbb { E } [ \frac { \hat { \alpha } _ { k } ( t ) } { \alpha _ { k } ( t ) } \hat { \gamma } _ { k } ^ { \star } ( t ) \mid \mathbf { X } _ { t } = \mathbf { X } ] + \mathbb { E } [ \rho _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k } \mid \mathbf { X } _ { t } = \mathbf { X } ] } \\ & { \quad \quad \quad = \frac { \hat { \alpha } _ { k } ( t ) } { \alpha _ { k } ( t ) } \hat { \gamma } _ { k } ^ { \star } ( t ) + \rho _ { k } ( t ) \hat { \mu } _ { t , k } ^ { p } . } \end{array}\tag{51}
$$

Subtracting Equation (51) from Equation (48), we obtain

$$
\begin{array} { l } { \dot { \hat { \gamma } } _ { k } ^ { \kappa } ( t ) - \hat { \mathbf { v } } _ { t , k } ^ { p } ( \mathbf { X } ) = \displaystyle \frac { \dot { \alpha } _ { k } ( t ) } { \alpha _ { k } ( t ) } \hat { \gamma } _ { k } ^ { \kappa } ( t ) + \rho _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k } - \left( \frac { \dot { \alpha } _ { k } ( t ) } { \alpha _ { k } ( t ) } \hat { \gamma } _ { k } ^ { \kappa } ( t ) + \rho _ { k } ( t ) \hat { \mu } _ { t , k } ^ { p } ( \mathbf { X } ) \right) } \\ { = \rho _ { k } ( t ) \left( \hat { \mathbf { x } } _ { 1 , k } - \hat { \mu } _ { t , k } ^ { p } ( \mathbf { X } ) \right) , } \end{array}\tag{52}
$$

which completes the proof.

## D.3 PROOF OF PROPOSITION 1

Proof. Let $\nu _ { 1 } , \ldots , \nu _ { J }$ denote the distinct eigenvalues of $\mathbf { L } ,$ and define

$$
{ \mathcal { T } } _ { j } = \{ k : \lambda _ { k } = \nu _ { j } \}\tag{53}
$$

as the index set associated with eigenvalue $\nu _ { j }$ . Any alternative orthonormal eigenbasis of L with the same eigenvalue ordering can be written as

$$
\tilde { \boldsymbol { \Psi } } = \boldsymbol { \Psi } \mathbf { Q } , \qquad \mathbf { Q } = \mathrm { b l k } \mathrm { d i a g } \left( \mathbf { Q } _ { 1 } , \dots , \mathbf { Q } _ { J } \right) ,\tag{54}
$$

where each $\mathbf { Q } _ { j } \in \mathbb { R } ^ { | \mathcal { T } _ { j } | \times | \mathcal { T } _ { j } | }$ is orthogonal. Hence, we have

$$
\begin{array} { r } { \mathbf { Q } ^ { \top } \mathbf { Q } = \mathbf { Q Q } ^ { \top } = \mathbf { I } , \qquad \mathbf { Q } \mathbf { \Lambda } = \mathbf { \Lambda } \mathbf { Q } . } \end{array}\tag{55}
$$

All graph-spectral interpolation coefficients depend on the graph-frequency mode k only through $\lambda _ { k }$ Therefore, their values are identical for all modes within the same repeated-eigenvalue eigenspace. It follows that

$$
\mathbf { Q A } ( t ) = \mathbf { A } ( t ) \mathbf { Q } , \qquad \mathbf { Q B } ( t ) = \mathbf { B } ( t ) \mathbf { Q } ,\tag{56}
$$

and similarly,

$$
\mathbf { Q } \dot { \mathbf { A } } ( t ) = \dot { \mathbf { A } } ( t ) \mathbf { Q } , \qquad \mathbf { Q } \dot { \mathbf { B } } ( t ) = \dot { \mathbf { B } } ( t ) \mathbf { Q } .\tag{57}
$$

Under the alternative eigenbasis, the node-domain graph-spectral path satisfies

$$
\begin{array} { r l } & { \dot { \mathbf { I } } ^ { \star } ( t ) = \Psi \dot { \mathbf { A } } ( t ) \Psi ^ { \top } \mathbf { X } _ { 0 } + \Psi \dot { \mathbf { B } } ( t ) \Psi ^ { \top } \mathbf { X } _ { 1 } } \\ & { \quad \quad \quad = \Psi \mathbf { Q } \dot { \mathbf { A } } ( t ) \mathbf { Q } ^ { \top } \Psi ^ { \top } \mathbf { X } _ { 0 } + \Psi \mathbf { Q } \dot { \mathbf { B } } ( t ) \mathbf { Q } ^ { \top } \Psi ^ { \top } \mathbf { X } _ { 1 } } \\ & { \quad \quad \quad = \tilde { \Psi } \dot { \mathbf { A } } ( t ) \tilde { \Psi } ^ { \top } \mathbf { X } _ { 0 } + \tilde { \Psi } \dot { \mathbf { B } } ( t ) \tilde { \Psi } ^ { \top } \mathbf { X } _ { 1 } . } \end{array}\tag{58}
$$

Thus, the node-domain conditional target velocity is invariant to the particular choice of eigenvectors within each degenerate eigenspace.

The learned velocity field model $\mathbf { V } _ { t } ( \mathbf { X } _ { t } ; \mathbf { \boldsymbol { \theta } } )$ is defined entirely in the node domain as a function of $\mathbf { X } _ { t } ,$ which does not directly depend on $\Psi$ . Therefore, for fixed parameters $\theta ,$ its output is unaffected by the choice of Laplacian eigenbasis. Moreover, because the node-domain conditional target velocity is invariant, the training objective in Equation (19) is also eigenbasis invariant.

It remains to establish the invariance of the anomaly score. Define the node-domain velocity residual as

$$
\mathbf { R } _ { t } = \mathbf { V } _ { t } ( \mathbf { X } _ { t } ; \pmb { \theta } ) - \dot { \Gamma } ^ { \star } ( t ) \in \mathbb { R } ^ { N \times R } ,\tag{59}
$$

and its graph Fourier representation

$$
\begin{array} { r } { \hat { \bf R } _ { t } = \Psi ^ { \top } { \bf R } _ { t } . } \end{array}\tag{60}
$$

The k-th row of $\hat { \mathbf { R } } _ { t }$ is

$$
\hat { \mathbf { r } } _ { t , k } = \hat { \mathbf { v } } _ { t , k } - \dot { \hat { \gamma } } _ { k } ^ { \star } ( t ) .\tag{61}
$$

Furthermore, define the diagonal weighting matrix

$$
\mathbf { H } ( t ) = \operatorname { d i a g } \left( \eta _ { 1 } ( t ) , \dots , \eta _ { N } ( t ) \right) .\tag{62}
$$

Then, for a fixed t and $\mathbf { X } _ { 0 } ,$ , the corresponding contribution to the anomaly score can be written as

$$
\begin{array} { r l r } & { } & { \displaystyle \sum _ { k = 1 } ^ { N } \eta _ { k } ( t ) \left\| \hat { \mathbf { v } } _ { t , k } - \dot { \hat { \gamma } } _ { k } ^ { \star } ( t ) \right\| _ { 2 } ^ { 2 } = \displaystyle \sum _ { k = 1 } ^ { N } \eta _ { k } ( t ) \left\| \hat { \mathbf { r } } _ { t , k } \right\| _ { 2 } ^ { 2 } } \\ & { } & { = \mathrm { T r } \left( \hat { \mathbf { R } } _ { t } ^ { \top } \mathbf { H } ( t ) \hat { \mathbf { R } } _ { t } \right) . } \end{array}\tag{63}
$$

Under the alternative eigenbasis ${ \tilde { \Psi } } = \Psi \mathbf { Q }$ , the corresponding spectral residual is

$$
\tilde { \mathbf { R } } _ { t } = \tilde { \Psi } ^ { \top } \mathbf { R } _ { t } = \mathbf { Q } ^ { \top } \hat { \mathbf { R } } _ { t } .\tag{64}
$$

Moreover, from the definition of the anomaly weight,

$$
\eta _ { k } ( t ) = \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \omega _ { k } ^ { 2 } } , \qquad \omega _ { k } = \sqrt { \tau \lambda _ { k } } ,\tag{65}
$$

so $\eta _ { k } ( t )$ also depends on $k$ only through $\lambda _ { k }$ . Consequently, all modes belonging to the same eigenspace have the same weight. Hence, $\mathbf { H } ( t )$ is scalar within each degenerate eigenspace and satisfies

$$
\mathbf { Q H } ( t ) = \mathbf { H } ( t ) \mathbf { Q } , \qquad \mathbf { Q H } ( t ) \mathbf { Q } ^ { \top } = \mathbf { H } ( t ) .\tag{66}
$$

Therefore,

$$
\operatorname { T r } \left( \tilde { \mathbf { R } } _ { t } ^ { \top } \mathbf { H } ( t ) \tilde { \mathbf { R } } _ { t } \right) = \operatorname { T r } \left( \hat { \mathbf { R } } _ { t } ^ { \top } \mathbf { Q } \mathbf { H } ( t ) \mathbf { Q } ^ { \top } \hat { \mathbf { R } } _ { t } \right)\tag{67}
$$

$$
\mathbf { \Lambda } = \mathrm { T r } \left( \hat { \mathbf { R } } _ { t } ^ { \top } \mathbf { H } ( t ) \hat { \mathbf { R } } _ { t } \right) .\tag{68}
$$

Thus, although individual spectral coordinates within a degenerate eigenspace depend on the choice of eigenbasis, their weighted aggregate contribution to the anomaly score is invariant.

Finally, summing the above equality over $t \in \mathcal { R } _ { \mathrm { a d } }$ and $\mathbf { X } _ { 0 } \in \mathcal { X } _ { \mathrm { a d } }$ shows that $\xi ( \mathcal { X } _ { \mathrm { a d } } , \mathcal { R } _ { \mathrm { a d } } , \mathbf { X } _ { 1 } )$ is unchanged under any orthogonal rotation within a repeated eigenspace. Therefore, the velocity field model, the conditional target velocity, and the resulting anomaly score are all invariant to the choice of Laplacian eigenbasis. □

## D.4 PROOF OF THEOREM 2

Proof. According to Theorem 1, we have

$$
\begin{array} { r } { \hat { \gamma } _ { k } ^ { \star } ( t ) = \alpha _ { k } ( t ) \hat { \mathbf { x } } _ { 0 , k } + \beta _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k } . } \end{array}\tag{69}
$$

Since the graph Fourier transform is orthonormal and $\mathbf { X } _ { 0 } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ , each source component satisfies $\hat { \mathbf { x } } _ { 0 , k } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . Thus, conditioning on the data endpoint gives

$$
\hat { \gamma } _ { k } ^ { \star } ( t ) \mid \hat { \mathbf { x } } _ { 1 , k } \sim \mathcal { N } \left( \beta _ { k } ( t ) \hat { \mathbf { x } } _ { 1 , k } , \alpha _ { k } ^ { 2 } ( t ) \mathbf { I } \right) .\tag{70}
$$

The conditional density is

$$
\phi _ { t } ( \hat { \bf x } _ { k } ( t ) \mid \hat { \bf x } _ { 1 , k } ) = \frac { 1 } { ( 2 \pi \alpha _ { k } ^ { 2 } ( t ) ) ^ { R / 2 } } \exp \left[ - \frac { \vert \vert \hat { \bf x } _ { k } - \beta _ { k } ( t ) \hat { \bf x } _ { 1 , k } \vert \vert _ { 2 } ^ { 2 } } { 2 \alpha _ { k } ^ { 2 } ( t ) } \right] .\tag{71}
$$

For $t \in ( 0 , 1 )$ , both $\alpha _ { k } ( t )$ and $\beta _ { k } ( t )$ are positive. The score over the k-th graph-frequency is

$$
\begin{array} { r l } & { \hat { \mathbf { s } } _ { t , k } ^ { p } ( \hat { \mathbf { X } } ) = \mathbb { E } _ { p _ { 0 } , p } \left[ \nabla _ { \hat { \mathbf { x } } _ { k } } \log p _ { t } ( \hat { \mathbf { X } } \mid \hat { \mathbf { X } } _ { 1 } ) \mid \hat { \mathbf { X } } _ { t } = \hat { \mathbf { X } } \right] } \\ & { \quad \quad \quad \quad = \mathbb { E } _ { p _ { 0 } , p } \left[ - \frac { \hat { \mathbf { x } } _ { k } - \boldsymbol { \beta } _ { k } \left( t \right) \hat { \mathbf { x } } _ { 1 , k } } { \alpha _ { k } ^ { 2 } \left( t \right) } \mid \hat { \mathbf { X } } _ { t } = \hat { \mathbf { X } } \right] } \\ & { \quad \quad \quad = - \frac { \hat { \mathbf { x } } _ { k } } { \alpha _ { k } ^ { 2 } \left( t \right) } + \frac { \boldsymbol { \beta } _ { k } \left( t \right) } { \alpha _ { k } ^ { 2 } \left( t \right) } \hat { \pmb { \mu } } _ { t , k } ^ { p } . } \end{array}\tag{72}
$$

Therefore, we have

$$
\beta _ { k } ( t ) \hat { \pmb { \mu } } _ { t , k } ^ { p } ( { \bf X } ) = \hat { \bf x } _ { k } + \alpha _ { k } ^ { 2 } ( t ) \hat { \bf s } _ { t , k } ^ { p } ( \hat { \bf X } ) .\tag{73}
$$

Applying the same identity under the test endpoint distribution q gives

$$
\beta _ { k } ( t ) \hat { \pmb { \mu } } _ { t , k } ^ { q } ( { \bf X } ) = \hat { \bf x } _ { k } + \alpha _ { k } ^ { 2 } ( t ) \hat { \bf s } _ { t , k } ^ { q } ( \hat { \bf X } ) .\tag{74}
$$

Thus, we have

$$
\hat { \pmb { \mu } } _ { t , k } ^ { q } ( \mathbf { X } ) - \hat { \pmb { \mu } } _ { t , k } ^ { p } ( \mathbf { X } ) = \frac { \alpha _ { k } ^ { 2 } ( t ) } { \beta _ { k } ( t ) } \left( \hat { \mathbf { s } } _ { t , k } ^ { q } ( \hat { \mathbf { X } } ) - \hat { \mathbf { s } } _ { t , k } ^ { p } ( \hat { \mathbf { X } } ) \right)\tag{75}
$$

Based on Lemma 1, we know that

$$
\dot { \hat { \gamma } } _ { k } ^ { \star } ( t ) - \hat { \mathbf { v } } _ { t , k } ^ { p } ( \mathbf { X } ) = \rho _ { k } ( t ) \left( \hat { \mathbf { x } } _ { 1 , k } - \hat { \pmb { \mu } } _ { t , k } ^ { p } ( \mathbf { X } ) \right) .\tag{76}
$$

Thus, we have

$$
\begin{array} { r l } & { \dot { \hat { \gamma } } _ { k } ^ { \star } ( t ) - \hat { \mathbf { v } } _ { t , k } ^ { p } ( \mathbf { X } ) = \rho _ { k } ( t ) \left( \hat { \mathbf { x } } _ { 1 , k } - \hat { \pmb { \mu } } _ { t , k } ^ { q } ( \mathbf { X } ) + \hat { \mu } _ { t , k } ^ { q } ( \mathbf { X } ) - \hat { \mu } _ { t , k } ^ { p } ( \mathbf { X } ) \right) } \\ & { \quad \quad \quad \quad \quad \quad = \rho _ { k } ( t ) \left( \hat { \mathbf { x } } _ { 1 , k } - \hat { \mu } _ { t , k } ^ { q } ( \mathbf { X } ) + \frac { \alpha _ { k } ^ { 2 } ( t ) } { \beta _ { k } ( t ) } \left( \hat { \mathbf { s } } _ { t , k } ^ { q } ( \hat { \mathbf { X } } ) - \hat { \mathbf { s } } _ { t , k } ^ { p } ( \hat { \mathbf { X } } ) \right) \right) } \end{array}\tag{77}
$$

Define $\mathbf { d } _ { t , k } = \hat { \mathbf { x } } _ { 1 , k } - \hat { \pmb { \mu } } _ { t , k } ^ { q } ( \mathbf { X } )$ and $\delta _ { t , k } = \hat { \mathbf { s } } _ { t , k } ^ { q } ( \hat { \mathbf { X } } ) - \hat { \mathbf { s } } _ { t , k } ^ { p } ( \hat { \mathbf { X } } )$ , then we obtain

$$
\dot { \hat { \gamma } } _ { k } ^ { \star } ( t ) - \hat { \mathbf { v } } _ { t , k } ^ { p } ( \mathbf { X } ) = \rho _ { k } ( t ) \left( \mathbf { d } _ { t , k } + \frac { \alpha _ { k } ^ { 2 } ( t ) } { \beta _ { k } ( t ) } \delta _ { t , k } \right) .\tag{78}
$$

Thus,

$$
\begin{array} { l } { \displaystyle \| \dot { \hat { \gamma } } _ { k } ^ { \star } ( t ) - \hat { \mathbf { v } } _ { t , k } ^ { p } ( \mathbf { X } ) \| _ { 2 } ^ { 2 } = \rho _ { k } ^ { 2 } ( t ) \left\| \mathbf { d } _ { t , k } + \frac { \alpha _ { k } ^ { 2 } ( t ) } { \beta _ { k } ( t ) } \delta _ { t , k } \right\| _ { 2 } ^ { 2 } } \\ { \displaystyle \quad = \rho _ { k } ^ { 2 } ( t ) \left( \| \mathbf { d } _ { t , k } \| _ { 2 } ^ { 2 } + \frac { \alpha _ { k } ^ { 4 } ( t ) } { \beta _ { k } ^ { 2 } ( t ) } \| \delta _ { t , k } \| _ { 2 } ^ { 2 } + 2 \frac { \alpha _ { k } ^ { 2 } ( t ) } { \beta _ { k } ( t ) } \mathbf { d } _ { t , k } ^ { \top } \delta _ { t , k } \right) . } \end{array}\tag{79}
$$

Since

$$
\begin{array} { r } { \hat { \pmb { \mu } } _ { t , k } ^ { q } ( \mathbf { X } ) = \mathbb { E } _ { p _ { 0 } , q } [ \hat { \gamma } _ { k } ^ { \star } ( 1 ) \mid \mathbf { X } _ { t } = \mathbf { X } ] , } \end{array}\tag{80}
$$

we have

$$
\mathbb { E } _ { p _ { 0 } , q } \left[ \mathbf { d } _ { t , k } \ | \ \hat { \mathbf { X } } _ { t } \right] = \mathbb { E } _ { p _ { 0 } , q } \left[ \hat { \mathbf { x } } _ { 1 , k } - \hat { \pmb { \mu } } _ { t , k } ^ { q } ( \mathbf { X } ) \ | \ \hat { \mathbf { X } } _ { t } \right] = \mathbf { 0 } .\tag{81}
$$

As $\delta _ { t , k }$ only depends on $\hat { \mathbf { X } } _ { t }$ , the cross term vanishes:

$$
\begin{array} { r } { \mathbb { E } _ { p _ { 0 } , q } \left[ \langle { \bf d } _ { t , k } , \delta _ { t , k } \rangle \mid \hat { \bf X } _ { t } \right] = \langle \mathbb { E } _ { p _ { 0 } , q } \left[ { \bf d } _ { t , k } \mid \hat { \bf X } _ { t } \right] , \delta _ { t , k } ( \hat { \bf X } _ { t } ) \rangle = { \bf 0 } . } \end{array}\tag{82}
$$

Therefore

$$
\mathbb { E } _ { p _ { 0 } , q } \left[ \| \dot { \widehat { \gamma } } _ { k } ^ { \star } ( t ) - \widehat { \mathbf { v } } _ { t , k } ^ { p } ( { \mathbf { X } } ) \| _ { 2 } ^ { 2 } \right] = \rho _ { k } ^ { 2 } ( t ) \mathbb { E } _ { p _ { 0 } , q } \left[ \| { \mathbf { d } } _ { t , k } \| _ { 2 } ^ { 2 } \right] + \rho _ { k } ^ { 2 } ( t ) \frac { \alpha _ { k } ^ { 4 } ( t ) } { \beta _ { k } ^ { 2 } ( t ) } \mathbb { E } _ { q _ { t } } \left[ \| \delta _ { t , k } \| _ { 2 } ^ { 2 } \right] .\tag{83}
$$

Since the anomaly score sums the contributions from these identically distributed source samples in $\mathcal { X } _ { \mathrm { a d } }$ , linearity of expectation gives

$$
\begin{array} { r l } & { \quad \mathbb { E } _ { \mathcal { X } _ { \mathrm { a d } } , q } \left[ \xi ( \mathcal { X } _ { \mathrm { a d } } , \mathcal { R } _ { \mathrm { a d } } , \mathbf { X } _ { 1 } ) \right] } \\ & { = \displaystyle \sum _ { k = 1 } ^ { N } \sum _ { t \in \mathcal { R } _ { \mathrm { a d } } } \vert \mathcal { X } _ { \mathrm { a d } } \vert \left( \eta _ { k } ( t ) \rho _ { k } ^ { 2 } ( t ) \mathbb { E } _ { p _ { 0 } , q } \left[ \| \mathbf { d } _ { t , k } \| _ { 2 } ^ { 2 } \right] + \eta _ { k } ( t ) \rho _ { k } ^ { 2 } ( t ) \frac { \alpha _ { k } ^ { 4 } ( t ) } { \beta _ { k } ^ { 2 } ( t ) } \mathbb { E } _ { q _ { t } } \left[ \| \delta _ { t , k } \| _ { 2 } ^ { 2 } \right] \right) . } \end{array}\tag{84}
$$

Observe that

$$
\eta _ { k } ( t ) \rho _ { k } ^ { 2 } ( t ) = \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \omega _ { k } ^ { 2 } } \left( \frac { \omega _ { k } } { \sinh ( \omega _ { k } ( 1 - t ) ) } \right) ^ { 2 } = \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) }\tag{85}
$$

and

$$
\begin{array} { l } { { \eta _ { k } ( t ) \rho _ { k } ^ { 2 } ( t ) { \frac { \alpha _ { k } ^ { 4 } ( t ) } { \beta _ { k } ^ { 2 } ( t ) } } = { \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) } } \left( { \frac { \sinh \left( \omega _ { k } ( 1 - t ) \right) } { \sinh \left( \omega _ { k } \right) } } \right) ^ { 4 } \left( { \frac { \sinh \left( \omega _ { k } \right) } { \sinh \left( \omega _ { k } t \right) } } \right) ^ { 2 } } } \\ { { = { \frac { \sinh ^ { 2 } \left( \omega _ { k } \left( 1 - t \right) \right) } { \sinh ^ { 2 } \left( \omega _ { k } \right) } } , } } \end{array}\tag{86}
$$

we have the following result:

$$
\begin{array} { l } { \displaystyle \mathbb { E } _ { \boldsymbol { \mathcal { X } } _ { \mathrm { a d } } , \boldsymbol { q } } \left[ \xi ( \boldsymbol { \mathcal { X } } _ { \mathrm { a d } } , \mathcal { R } _ { \mathrm { a d } } , \mathbf { X } _ { 1 } ) \right] } \\ { \displaystyle = \sum _ { k = 1 } ^ { N } \sum _ { t \in \mathcal { R } _ { \mathrm { a d } } } \big \lvert \left( \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) } \mathbb { E } _ { p _ { 0 } , \boldsymbol { q } } \left[ \lVert \mathbf { d } _ { t , k } \rVert _ { 2 } ^ { 2 } \right] + \frac { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) } { \sinh ^ { 2 } ( \omega _ { k } ) } \mathbb { E } _ { q _ { t } } \left[ \lVert \delta _ { t , k } \rVert _ { 2 } ^ { 2 } \right] \right) , } \end{array}\tag{87}
$$

through which the proof is completed.

## D.5 PROOF OF COROLLARY 1

Proof. For $t \in ( 0 , 1 )$ , define the rescaled spectral observation

$$
{ \bf Z } _ { t , k } = \frac { \hat { \gamma } _ { k } ^ { \star } ( t ) } { \beta _ { k } ( t ) } = \hat { \bf x } _ { 1 , k } + \frac { \alpha _ { k } ( t ) } { \beta _ { k } ( t ) } \hat { \bf x } _ { 0 , k } .\tag{88}
$$

Since

$$
\hat { \pmb { \mu } } _ { t , k } ^ { q } ( \mathbf { X } _ { t } ) = \mathbb { E } _ { p _ { 0 } , q } [ \hat { \mathbf { x } } _ { 1 , k } \mid \mathbf { X } _ { t } ]\tag{89}
$$

is the minimum mean-squared-error estimator of $\hat { \mathbf { x } } _ { 1 , k }$ , using $\mathbf { Z } _ { t , k }$ as a candidate estimator gives

$$
\begin{array} { r l } & { \mathbb { E } _ { p _ { 0 } , q } [ \| \mathbf { d } _ { t , k } \| _ { 2 } ^ { 2 } ] \leq \mathbb { E } _ { p _ { 0 } , q } [ \| \hat { { \mathbf { x } } } _ { 1 , k } - { \mathbf { Z } } _ { t , k } \| _ { 2 } ^ { 2 } ] } \\ & { \qquad = \frac { \alpha _ { k } ^ { 2 } ( t ) } { \beta _ { k } ^ { 2 } ( t ) } \mathbb { E } _ { p _ { 0 } } [ \| \hat { { \mathbf { x } } } _ { 0 , k } \| _ { 2 } ^ { 2 } ] } \\ & { \qquad = R \frac { \alpha _ { k } ^ { 2 } ( t ) } { \beta _ { k } ^ { 2 } ( t ) } . } \end{array}\tag{90}
$$

Because

$$
\frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) } = \frac { \beta _ { k } ^ { 2 } ( t ) } { \alpha _ { k } ^ { 2 } ( t ) } ,\tag{91}
$$

multiplying both sides of Equation (90) by this factor gives

$$
\begin{array} { r } { \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) } \mathbb { E } _ { p _ { 0 } , q } [ \| \mathbf { d } _ { t , k } \| _ { 2 } ^ { 2 } ] \leq R . } \end{array}\tag{92}
$$

For $\omega _ { k } ~ = ~ 0 ,$ , the ratio in Equation (92) is defined by its continuous limit $t ^ { 2 } / ( 1 - t ) ^ { 2 }$ , and the same bound holds. Finally, summing Equation (92) over all source samples, flow times, and graphfrequency modes yields

$$
\sum _ { k = 1 } ^ { N } \sum _ { t \in \mathcal { R } _ { \mathrm { a d } } } \vert \mathcal { X } _ { \mathrm { a d } } \vert \frac { \sinh ^ { 2 } ( \omega _ { k } t ) } { \sinh ^ { 2 } ( \omega _ { k } ( 1 - t ) ) } \mathbb { E } _ { p _ { 0 } , q } \left[ \Vert \hat { \mathbf { x } } _ { 1 , k } - \hat { \pmb { \mu } } _ { t , k } ^ { q } ( \mathbf { X } ) \Vert _ { 2 } ^ { 2 } \right] \leq \vert \mathcal { X } _ { \mathrm { a d } } \vert \vert \mathcal { R } _ { \mathrm { a d } } \vert N R ,\tag{93}
$$

which completes the proof.

## E EXPERIMENTAL DETAILS AND ADDITIONAL RESULTS

## E.1 DATASETS

We evaluate GRASP on four MTS anomaly detection datasets from mTSBench (Zhou et al., 2026), covering spacecraft telemetry, server monitoring, network intrusion detection, and space-weather analysis. Each dataset contains separate training and test sequences, together with pointwise anomaly labels for evaluation. We use the data preprocessing and benchmark splits provided by mTSBench. We briefly introduce the four datasets below:

• SMAP (Hundman et al., 2018): The Soil Moisture Active Passive dataset contains telemetry collected from NASA’s SMAP spacecraft. The benchmark subset comprises 51 multivariate sequences, each containing 26 variables. This dataset evaluates the ability to detect anomalous behavior in spacecraft telemetry.

• SMD (Su et al., 2019): The Server Machine Dataset contains monitoring measurements collected from server machines. We use 18 sequences with 39 variables each, where the labeled anomalies represent deviations from normal server operation.

• CICIDS (Sharafaldin et al., 2018): CICIDS2017 is a network intrusion detection dataset containing both benign traffic and multiple types of cyberattacks. The benchmark subset comprises six sequences, each represented by 73 network-flow features.

• SWAN (Angryk et al., 2020): The Space Weather Analytics for Solar Flares dataset contains multivariate time series of physical properties associated with solar active regions. We use the 39-variable sequence included in the mTSBench anomaly detection benchmark.

## E.2 BASELINES

We compare GRASP with 13 representative baselines spanning statistical outlier detection, forecasting- and reconstruction-based detection, graph-based modeling, Transformers, and frequency-domain methods. We briefly introduce these baselines below:

• GDN (Deng & Hooi, 2021) learns inter-sensor dependencies using node embeddings and graph attention, and detects anomalies using forecasting errors.

• GCAD (Liu et al., 2025) infers dynamic Granger-causal graphs from the gradients of a forecasting model and identifies anomalies through deviations in the learned causal patterns.

• COPOD (Li et al., 2020) estimates empirical copula-based tail probabilities and assigns larger anomaly scores to statistically extreme observations.

• HBOS (Goldstein & Dengel, 2012) constructs a histogram for each variable and combines the resulting density estimates under a feature-independence assumption.

• TimesNet (Wu et al., 2023) transforms one-dimensional time series into period-dependent two-dimensional representations to capture intra- and inter-period variations.

• CNN (Munir et al., 2018) uses temporal convolutions to forecast future observations and detects anomalies through prediction errors.

• USAD (Audibert et al., 2020) employs adversarially trained Autoencoders to learn normal temporal patterns and scores anomalies using reconstruction errors.

Table 4: Anomaly detection performance comparison on SMAP and SMD.
<table><tr><td rowspan="2">Model</td><td colspan="3">SMAP</td><td colspan="3">SMD</td></tr><tr><td>PRC</td><td>ROC</td><td>Best-F1</td><td>PRC</td><td>ROC</td><td>Best-F1</td></tr><tr><td>A-Transformer</td><td> $0 . 1 3 4 0 \pm 0 . 0 0 7 2$ </td><td> $0 . 5 0 6 5 \pm 0 . 0 0 7 1$ </td><td> $0 . 2 0 6 0 \pm 0 . 0 1 1 0$ </td><td> $0 . 0 7 2 0 \pm 0 . 0 0 4 6$ </td><td> $0 . 5 0 1 3 \pm 0 . 0 0 4 9$ </td><td> $0 . 1 2 6 0 \pm 0 . 0 0 7 1$ </td></tr><tr><td>FITS</td><td> $0 . 1 4 5 5 \pm 0 . 0 0 4 7$ </td><td> $0 . 5 1 4 7 \pm 0 . 0 0 6 8$ </td><td> $0 . 2 7 9 8 \pm 0 . 0 0 6 3$ </td><td> $0 . 2 8 6 3 \pm 0 . 0 0 2 6$ </td><td> $0 . 8 2 9 9 \pm 0 . 0 0 2 2$ </td><td> $0 . 4 0 2 5 \pm 0 . 0 0 2 6$ </td></tr><tr><td>TimesNet</td><td> $0 . 1 7 9 2 \pm 0 . 0 0 7 1$ </td><td> $0 . 5 3 3 6 \pm 0 . 0 0 5 8$ </td><td> $0 . 3 1 6 1 \pm 0 . 0 0 8 9$ </td><td> $0 . 2 8 3 9 \pm 0 . 0 0 5 3$ </td><td> $0 . 8 0 8 7 \pm 0 . 0 0 2 1$ </td><td> $0 . 3 7 1 3 \pm 0 . 0 0 4 9$ </td></tr><tr><td>COPOD</td><td> $0 . 1 9 2 7 \pm 0 . 0 0 0 0$ </td><td> $0 . 5 9 1 8 \pm 0 . 0 0 0 0$ </td><td> $0 . 2 8 4 1 \pm 0 . 0 0 0 0$ </td><td> $0 . 2 1 3 9 \pm 0 . 0 0 0 0$ </td><td> $0 . 7 1 9 3 \pm 0 . 0 0 0 0$ </td><td> $0 . 2 9 5 1 \pm 0 . 0 0 0 0$ </td></tr><tr><td>HBOS</td><td> $0 . 1 8 9 7 \pm 0 . 0 0 0 0$ </td><td> $0 . 5 6 5 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 2 6 6 4 \pm 0 . 0 0 0 0$ </td><td> $0 . 2 6 7 8 \pm 0 . 0 0 0 0$ </td><td> $0 . 7 3 7 8 \pm 0 . 0 0 0 0$ </td><td>0.3556 ± 0.0000</td></tr><tr><td>TranAD</td><td> $0 . 2 1 2 2 \pm 0 . 0 0 4 2$ </td><td> $0 . 6 1 0 4 \pm 0 . 0 0 5 6$ </td><td> $0 . 3 2 6 4 \pm 0 . 0 0 6 6$ </td><td> $0 . 3 3 0 4 \pm 0 . 0 0 2 2$ </td><td> $0 . 7 6 9 1 \pm 0 . 0 0 2 2$ </td><td> $0 . 4 1 8 1 \pm 0 . 0 0 1 8$ </td></tr><tr><td>CNN</td><td> $0 . 2 1 3 6 \pm 0 . 0 0 2 9$ </td><td> $\underline { { 0 . 6 5 7 7 } } \pm 0 . 0 0 8 8$ </td><td> $0 . 3 4 7 1 \pm 0 . 0 0 5 5$ </td><td> $0 . 3 7 3 0 \pm 0 . 0 0 4 9$ </td><td> $0 . 7 9 2 1 \pm 0 . 0 0 4 2$ </td><td>0.4313 ± 0.0046</td></tr><tr><td>Autoencoder</td><td> $0 . 2 2 7 7 \pm 0 . 0 0 0 6$ </td><td> $0 . 5 8 5 9 \pm 0 . 0 0 1 5$ </td><td> $0 . 2 9 9 1 \pm 0 . 0 0 0 6$ </td><td> $0 . 3 6 1 3 \pm 0 . 0 0 5 0$ </td><td> $0 . 7 2 9 7 \pm 0 . 0 0 3 4$ </td><td> $0 . 4 2 1 4 \pm 0 . 0 0 3 4$ </td></tr><tr><td>USAD</td><td> $0 . 2 2 6 3 \pm 0 . 0 0 4 9$ </td><td> $0 . 5 8 5 0 \pm 0 . 0 0 8 9$ </td><td> $0 . 3 8 5 0 \pm 0 . 0 0 7 5$ </td><td> $0 . 4 1 0 5 \pm 0 . 0 0 4 0$ </td><td> $0 . 8 7 2 6 \pm 0 . 0 0 1 9$ </td><td> $0 . 4 8 1 0 \pm 0 . 0 0 2 1$ </td></tr><tr><td>OmniAnomaly</td><td> $0 . 2 4 2 0 \pm 0 . 0 0 0 3$ </td><td> $0 . 5 9 5 0 \pm 0 . 0 0 0 2$ </td><td> $0 . 4 0 6 5 \pm 0 . 0 0 0 3$ </td><td> $0 . 4 2 8 4 \pm 0 . 0 0 0 1$ </td><td> $0 . 8 8 2 2 \pm 0 . 0 0 0 1$ </td><td> $0 . 5 0 6 1 \pm 0 . 0 0 0 1$ </td></tr><tr><td>CATCH</td><td>0.2450 ± 0.0028</td><td> $0 . 6 0 7 8 \pm 0 . 0 0 4 6$ </td><td> $0 . 3 6 6 4 \pm 0 . 0 0 4 1$ </td><td> $0 . 4 7 6 3 \pm 0 . 0 0 1 1$ </td><td> $\mathbf { 0 . 8 9 6 8 \pm 0 . 0 0 0 4 }$ </td><td> $0 . 5 0 5 7 \pm 0 . 0 0 1 2$ </td></tr><tr><td>GDN</td><td> $\overline { { 0 . 2 1 9 7 \pm 0 . 0 1 1 5 } }$ </td><td> $0 . 6 0 9 9 \pm 0 . 0 0 7 0$ </td><td> $0 . 3 3 9 4 \pm 0 . 0 0 8 7$ </td><td> $0 . 4 0 0 6 \pm 0 . 0 1 5 7$ </td><td> $0 . 8 0 1 4 \pm 0 . 0 1 0 4$ </td><td> $0 . 4 6 4 0 \pm 0 . 0 1 4 0$ </td></tr><tr><td>GCAD</td><td> $0 . 2 1 4 4 \pm 0 . 0 1 3 5$ </td><td> $0 . 6 3 4 8 \pm 0 . 0 1 1 9$ </td><td> $0 . 3 4 7 1 \pm 0 . 0 1 0 4$ </td><td> $0 . 3 5 6 9 \pm 0 . 0 0 8 9$ </td><td> $0 . 8 4 1 6 \pm 0 . 0 0 7 4$ </td><td> $0 . 4 5 3 5 \pm 0 . 0 1 0 9$ </td></tr><tr><td>GRASP</td><td> $\mathbf { 0 . 3 1 4 9 \pm 0 . 0 0 8 7 }$  </td><td> $\mathbf { 0 . 7 0 1 5 \pm 0 . 0 1 3 0 }$  </td><td> $\mathbf { 0 . 4 1 7 8 \pm 0 . 0 0 6 4 }$  一</td><td> $\mathbf { 0 . 4 8 4 5 \pm 0 . 0 0 4 3 }$  </td><td> $0 . 8 5 6 7 \pm 0 . 0 0 3 8$  </td><td> $\mathbf { 0 . 5 0 8 7 \pm 0 . 0 0 3 4 }$ </td></tr></table>

• TranAD (Tuli et al., 2022) combines Transformer-based sequence modeling, selfconditioning, and adversarial training for MTS anomaly detection.

• OmniAnomaly (Su et al., 2019) combines recurrent temporal modeling with a variational Autoencoder to capture stochastic temporal dependencies and detects anomalies using reconstruction probabilities.

• Autoencoder (Sakurada & Yairi, 2014) learns a compressed representation of normal observations and uses reconstruction errors as anomaly scores.

• A-Transformer (Xu et al., 2022) models prior and learned temporal associations through anomaly attention and detects anomalies using the discrepancy between these associations.

• FITS (Xu et al., 2024) models time series through learnable interpolation of low-frequency components in the complex frequency domain.

• CATCH (Wu et al., 2025) divides frequency-domain representations into patches and captures spectral patterns and inter-channel dependencies through masked channel fusion.

## E.3 IMPLEMENTATION DETAILS

We implement GRASP in PyTorch and evaluate it on SMAP, SMD, CICIDS, and SWAN. For each time series, the normal training data are chronologically divided into 80% for training and 20% for validation. We use a window length of 50 and set the graph Dirichlet energy weight parameter to $\tau = 2$ for all datasets. For graph construction, we binarize the edge-weight matrix using a fixed threshold of 0.5 across all datasets.

The velocity predictor used in GRASP consists of two time-conditioned TSMixer blocks with a hidden dimension of 128 and a dropout rate of 0.1. Flow time is encoded with a fixed 64-dimensional sinusoidal embedding. The model is trained with a batch size of 256 and a learning rate of $1 0 ^ { - 3 }$ Training is limited to 1500 epochs, and validation is performed every 50 epochs. No anomaly label are used for training, validation, graph construction, or anomaly score calibration.

During inference, anomaly scores are computed from the weighted velocity discrepancies aggregated across graph-frequency modes, source samples, and flow times. We compute the average of five source samples, and each prior sample is evaluated at ten flow times sampled equally between 0 and 1. All experiments are conducted on a single 80GB NVIDIA A100 GPU. We repeat each experiment using ten random seeds.

## E.4 ADDITIONAL MTS ANOMALY DETECTION RESULTS

In Section 4.1, we have presented the mean MTS anomaly detection performance over 10 random seeds in Table 1. To further assess the stability of the compared methods, we further report the results with standard deviation on the SMAP and SMD datasets in Table 4, and on the CICIDS and SWAN datasets in Table 5.

Table 5: Anomaly detection performance comparison on CICIDS and SWAN.
<table><tr><td rowspan="2">Model</td><td colspan="3">CICIDS</td><td colspan="3">SWAN</td></tr><tr><td>PRC</td><td>ROC</td><td>Best-F1</td><td>PRC</td><td>ROC</td><td>Best-F1</td></tr><tr><td>A-Transformer</td><td> $0 . 2 3 9 6 \pm 0 . 0 0 0 7$ </td><td> $0 . 4 9 7 2 \pm 0 . 0 0 2 5$ </td><td> $0 . 3 0 6 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 2 8 1 7 \pm 0 . 0 0 1 8$ </td><td> $0 . 5 0 2 1 \pm 0 . 0 0 1 4$ </td><td> $0 . 4 3 6 5 \pm 0 . 0 0 0 0$ </td></tr><tr><td>FITS</td><td> $0 . 1 8 4 9 \pm 0 . 0 0 1 3$ </td><td> $0 . 4 1 0 6 \pm 0 . 0 0 3 9$ </td><td> $0 . 3 1 2 6 \pm 0 . 0 0 1 3$ </td><td> $0 . 3 1 8 9 \pm 0 . 0 1 2 0$ </td><td> $0 . 4 6 0 7 \pm 0 . 0 1 2 0$ </td><td> $0 . 4 3 6 5 \pm 0 . 0 0 0 0$ </td></tr><tr><td>TimesNet</td><td> $0 . 1 9 0 5 \pm 0 . 0 0 2 1$ </td><td> $0 . 4 6 5 7 \pm 0 . 0 0 5 2$ </td><td> $0 . 3 1 2 1 \pm 0 . 0 0 1 5$ </td><td> $0 . 3 0 6 8 \pm 0 . 0 0 8 0$ </td><td> $0 . 4 5 0 5 \pm 0 . 0 1 3 7$ </td><td> $0 . 4 3 6 5 \pm 0 . 0 0 0 0$ </td></tr><tr><td>COPOD</td><td> $0 . 2 2 1 4 \pm 0 . 0 0 0 0$ </td><td> $0 . 5 5 6 6 \pm 0 . 0 0 0 0$ </td><td> $0 . 3 7 3 3 \pm 0 . 0 0 0 0$ </td><td> $0 . 2 7 9 2 \pm 0 . 0 0 0 0$ </td><td> $0 . 5 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 4 3 6 5 \pm 0 . 0 0 0 0$ </td></tr><tr><td>HBOS</td><td> $0 . 2 6 0 3 \pm 0 . 0 0 0 0$ </td><td> $0 . 5 5 8 6 \pm 0 . 0 0 0 0$ </td><td> $0 . 3 8 5 2 \pm 0 . 0 0 0 0$ </td><td> $0 . 2 7 9 2 \pm 0 . 0 0 0 0$ </td><td> $0 . 5 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 4 3 6 5 \pm 0 . 0 0 0 0$ </td></tr><tr><td>TranAD</td><td> $0 . 2 0 4 3 \pm 0 . 0 0 2 1$ </td><td> $0 . 4 7 0 4 \pm 0 . 0 0 8 5$ </td><td> $0 . 3 4 1 6 \pm 0 . 0 0 0 7$ </td><td> $0 . 3 3 6 0 \pm 0 . 0 0 2 0$ </td><td> $0 . 4 9 1 1 \pm 0 . 0 0 3 9$ </td><td> $0 . 4 3 6 5 \pm 0 . 0 0 0 1$ </td></tr><tr><td>CNN</td><td> $0 . 2 3 7 1 \pm 0 . 0 0 1 9$ </td><td> $0 . 5 5 4 3 \pm 0 . 0 0 5 6$ </td><td> $0 . 3 9 9 1 \pm 0 . 0 0 0 9$ </td><td> $0 . 3 7 6 0 \pm 0 . 0 0 5 6$ </td><td> $0 . 4 9 6 6 \pm 0 . 0 0 8 6$ </td><td> $0 . 4 3 6 5 \pm 0 . 0 0 0 0$ </td></tr><tr><td>Autoencoder</td><td> $0 . 3 1 4 7 \pm 0 . 0 0 8 9$ </td><td> $0 . 7 2 7 6 \pm 0 . 0 4 0 8$ </td><td> $0 . 3 9 3 8 \pm 0 . 0 1 0 6$ </td><td> $0 . 3 2 9 4 \pm 0 . 0 0 2 5$ </td><td> $0 . 4 9 4 4 \pm 0 . 0 0 9 1$ </td><td> $0 . 4 3 7 1 \pm 0 . 0 0 0 6$ </td></tr><tr><td>USAD</td><td> $0 . 2 0 9 4 \pm 0 . 0 0 0 7$ </td><td> $0 . 4 6 1 9 \pm 0 . 0 0 1 9$ </td><td> $0 . 3 5 2 5 \pm 0 . 0 0 0 3$ </td><td> $0 . 4 4 0 5 \pm 0 . 0 0 8 8$ </td><td> $0 . 5 5 8 5 \pm 0 . 0 0 9 5$ </td><td> $0 . 4 3 6 5 \pm 0 . 0 0 0 0$ </td></tr><tr><td>OmniAnomaly</td><td> $0 . 2 1 3 8 \pm 0 . 0 0 0 0$ </td><td> $0 . 4 5 0 1 \pm 0 . 0 0 0 0$ </td><td> $0 . 3 5 4 8 \pm 0 . 0 0 0 0$ </td><td> $0 . 4 8 4 7 \pm 0 . 0 0 0 2$ </td><td> $0 . 6 1 3 0 \pm 0 . 0 0 0 4$ </td><td> $0 . 4 3 6 5 \pm 0 . 0 0 0 0$ </td></tr><tr><td>CATCH</td><td> $0 . 2 8 6 6 \pm 0 . 0 0 0 7$ </td><td> $0 . 6 1 6 2 \pm 0 . 0 0 0 6$ </td><td> $0 . 4 0 2 3 \pm 0 . 0 0 0 8$ </td><td> $0 . 4 6 3 8 \pm 0 . 0 0 1 7$ </td><td> $0 . 6 2 1 8 \pm 0 . 0 0 1 8$ </td><td> $0 . 4 5 3 3 \pm 0 . 0 0 0 9$ </td></tr><tr><td>GDN</td><td> $0 . 2 9 2 6 \pm 0 . 0 1 8 8$ </td><td> $0 . 7 0 1 7 \pm 0 . 0 2 3 9$ </td><td> $\overline { { 0 . 3 8 0 5 \pm 0 . 0 1 4 8 } }$ </td><td> $0 . 6 4 2 3 \pm 0 . 0 2 3 1$ </td><td> $0 . 8 3 5 6 \pm 0 . 0 0 9 1$ </td><td> $\underline { { 0 . 6 4 3 2 } } \pm 0 . 0 1 5 5$ </td></tr><tr><td>GCAD</td><td> $0 . 3 6 3 6 \pm 0 . 0 1 9 4$ </td><td> $0 . 7 1 9 1 \pm 0 . 0 1 5 1$ </td><td> $0 . 3 8 4 4 \pm 0 . 0 0 5 5$ </td><td> $\underline { { 0 . 6 4 4 5 \pm 0 . 0 1 4 6 } }$ </td><td> $0 . 8 1 4 0 \pm 0 . 0 0 7 0$ </td><td> $\overline { { 0 . 6 2 3 1 \pm 0 . 0 0 8 9 } }$ </td></tr><tr><td>GRASP</td><td> $\mathbf { 0 . 3 9 9 6 \pm 0 . 0 0 8 3 }$  </td><td> $\mathbf { 0 . 8 1 0 8 \pm 0 . 0 0 7 2 }$  </td><td> $\mathbf { 0 . 4 6 1 5 \pm 0 . 0 0 3 7 }$  一</td><td> $\mathbf { 0 . 7 5 5 6 \pm 0 . 0 1 1 6 }$  </td><td> $\mathbf { 0 . 8 7 5 8 \pm 0 . 0 1 0 1 }$  </td><td> $\mathbf { 0 . 6 8 8 6 \pm 0 . 0 1 4 5 }$ </td></tr></table>

Table 6: Ablation results on CICIDS and SWAN.
<table><tr><td rowspan="2">Variant</td><td colspan="3">CICIDS</td><td colspan="3">SWAN</td></tr><tr><td>PRC</td><td>ROC</td><td>Best-F1</td><td>PRC</td><td>ROC</td><td>Best-F1</td></tr><tr><td>GRASP</td><td>0.3996</td><td>0.8108</td><td>0.4615</td><td>0.7556</td><td>0.8758</td><td>0.6886</td></tr><tr><td>GRASP-mean</td><td>0.3764</td><td>0.7593</td><td>0.4499</td><td>0.7552</td><td>0.8753</td><td>0.6881</td></tr><tr><td>GRASP-ER</td><td>0.3734</td><td>0.7546</td><td>0.4497</td><td>0.7534</td><td>0.8723</td><td>0.6857</td></tr><tr><td>GRASP-lin</td><td>0.3740</td><td>0.7543</td><td>0.4493</td><td>0.7486</td><td>0.8714</td><td>0.6841</td></tr></table>

GRASP achieves the highest PRC and Best-F1 on all four datasets, as well as the highest ROC on three of them. On SMD, GRASP obtains the best PRC and Best-F1, while CATCH achieves the highest ROC. The relatively small standard deviations of GRASP indicate that its performance is stable across random seeds.

## E.5 ADDITIONAL ABLATION RESULTS

Table 6 extends the ablation study in Table 2 to CICIDS and SWAN. The GRASP model consistently achieves the best performance across all metrics. The improvements are particularly clear on CICIDS, whereas the gains on SWAN are smaller but remain consistent. Overall, replacing the graph-spectral path with a linear path, removing the graph-spectral weighting scheme, or using a randomly generated graph leads to performance degradation. These results validate the individual contribution of the three design components.

## E.6 COMPARISON OF VELOCITY-FIELD ARCHITECTURES

In this section, we examine the effect of the velocity-field architecture by replacing the default TSMixer backbone of GRASP with two time-conditioned alternatives based on a GNN and a Transformer. The GNN treats sensor channels as nodes and applies two residual GCN blocks for crosschannel propagation (Kipf & Welling, 2017). The Transformer treats sensor channels as tokens and employs two encoder layers with four-head self-attention (Vaswani et al., 2017). All three architectures use a hidden dimension of 128 and are trained and evaluated under the same experimental protocol.

As shown in Table 7, TSMixer achieves the highest PRC, ROC, and Best-F1 on all four datasets. The advantage is particularly pronounced on SWAN, where its PRC reaches 0.7552, compared with 0.5306 for the GNN and 0.5594 for the Transformer. These results support the use of the simple yet effective TSMixer model in GRASP. Under the considered setting, the tested GNN and Transformer alternatives provide no performance improvement.

## E.7 ADDITIONAL FLOW-TIME AND GRAPH-FREQUENCY ANALYSIS

Figure 5 extends the anomaly attribution analysis in Section 4.3 to SMD, CICIDS, and SWAN datasets. We decompose the squared velocity residuals across flow time and graph-frequency quantiles. The graph frequencies are ordered from low to high Laplacian eigenvalues, while larger flow times correspond to states closer to the data endpoint. The top row presents the unweighted residual contributions, and the bottom row presents the contributions after applying the weighting schedule.

Table 7: Ablation study of velocity-field architectures.
<table><tr><td rowspan="2">Variant</td><td colspan="3">SMAP</td><td colspan="3">SMD</td><td colspan="3"></td><td colspan="3">SWAN</td></tr><tr><td>PRC</td><td>ROC</td><td>Best-F1</td><td>PRC</td><td>ROC</td><td>Best-F1</td><td>PRC</td><td>ROC</td><td>Best-F1</td><td>PRC</td><td>ROC</td><td>Best-F1</td></tr><tr><td>TSMixer</td><td>0.3149</td><td>0.7015</td><td>0.4178</td><td>0.4845</td><td>0.8567</td><td>0.5087</td><td>0.3996</td><td>0.8108</td><td>0.4615</td><td>0.7556</td><td>0.8758</td><td>0.6886</td></tr><tr><td>GNN</td><td>0.2773</td><td>0.6758</td><td>0.3881</td><td>0.4659</td><td>0.8542</td><td>0.4926</td><td>0.3551</td><td>0.7232</td><td>0.4459</td><td>0.5306</td><td>0.6570</td><td>0.5188</td></tr><tr><td>Transformer</td><td>0.2639</td><td>0.6347</td><td>0.3795</td><td>0.4475</td><td>0.8272</td><td>0.4674</td><td>0.3283</td><td>0.6407</td><td>0.4331</td><td>0.5594</td><td>0.6758</td><td>0.5122</td></tr></table>

![](images/53980aed8468565e1f4a0bb47c0f44713a1477f826f88de5413f65d76db2f78b.jpg)  
Figure 5: Anomaly score contributions across flow times and graph-frequency quantiles on SMD, CICIDS, and SWAN. Up: unweighted anomaly signal. Below: weighted anomaly signal.

On SMD and CICIDS, the anomaly evidence is primarily concentrated in intermediate graphfrequency quantiles at later flow times. Applying the score weights further shifts the attribution toward the data endpoint while preserving the dominant frequency regions. In contrast, SWAN exhibits a more diffuse attribution pattern across graph frequencies. These results suggest that the flow time toward the endpoint provides more informative anomaly signals for all datasets, while the informative graph-frequency components vary across datasets rather than being universally concentrated in the highest-frequency modes.

## E.8 SENSITIVITY TO FLOW-TIME EVALUATIONS AND SOURCE SAMPLES

In this section, we examine the sensitivity of GRASP to two inference-time hyperparameters: the number of flow-time evaluation points $| \dot { \mathcal { R } } _ { \mathrm { a d } } |$ and the number of source samples $\lvert \mathcal { X } _ { \mathrm { a d } } \rvert$ . We vary $| \mathcal { R } _ { \mathrm { a d } } | \in \{ 1 , 2 , 5 , 1 0 , 2 0 , 5 0 \}$ while fixing $| \mathcal { X } _ { \mathrm { a d } } | = 5 .$ , and vary $| \mathcal { X } _ { \mathrm { a d } } | \in \{ 1 , 2 , 5 , 1 0 , \dot { 2 } 0 \}$ while fixing $| \mathcal { R } _ { \mathrm { a d } } | = \mathrm { i } 0$ . Figure 6 and Figure 7 report the mean and standard deviation over ten random seeds.

As shown in Figure 6, increasing the number of flow-time evaluation points generally improves performance on SMAP, SMD, and CICIDS, especially when $| \mathcal { R } _ { \mathrm { a d } } | < 1 \bar { 0 }$ . Performance on SWAN remains stable across different values of $| \mathcal { R } _ { \mathrm { a d } } |$ . Figure 7 shows that GRASP is relatively insensitive to the number of source samples. Increasing $| { \mathcal { X } } _ { \mathrm { a d } } |$ yields modest improvements on SMAP and negligible changes on the other datasets. These results support the default choices of $| \mathcal { R } _ { \mathrm { a d } } | = 1 0$ and $| \bar { \mathcal { X } } _ { \mathrm { a d } } | = 5$ as practical trade-offs between detection performance and inference cost.

![](images/0a053f4deaf01c36c182e43e2db5353d6afd62c93d1b9bac052782ae303b579d.jpg)

Figure 6: Sensitivity to the number of flow-time evaluation points $| \mathcal { R } _ { \mathrm { a d } } |$ with $| { \mathcal { X } } _ { \mathrm { a d } } | = 5$  
![](images/c0ee18869159c0bde0d0e2d9a92c51c7561f02f2675947ad3acd492cbbd00bc4.jpg)  
Figure 7: Sensitivity to the number of source samples $| { \mathcal { X } } _ { \mathrm { a d } } |$ with $| \mathcal { R } _ { \mathrm { a d } } | = 1 0$

## E.9 SENSITIVITY TO THE GRAPH DIRICHLET-ENERGY WEIGHT

The graph Dirichlet-energy weight τ controls the strength of the graph-structural regularization in the graph-spectral probability path, with $\tau = 0$ recovering the standard linear path. We evaluate $\tau \in \bar { \{ 0 , 0 . 2 5 , 0 . 5 , \bar { 1 , } 2 , 4 , 8 \} }$ on all four datasets, retraining the model for each value while keeping all other hyperparameters fixed.

Figure 8 shows that the preferred value varies across datasets. SMD generally favors smaller values around $\tau \ : = \ : 0 . 5$ , whereas SMAP and CICIDS attain their best PRC and Best-F1 near $\tau \ = \ 4$ SWAN performs best over an intermediate range around $\tau = 1 { - } 2$ . Despite these dataset-specific optima, performance remains relatively stable for $\tau \in [ 0 . 5 , 4 ]$ . In contrast, $\tau = 8$ leads to a clear performance degradation, suggesting that excessively strong graph regularization could weaken the anomaly signal.

![](images/e70fce46fc37184cf3a449c8be2efea52ec0423dad3d520d76d124cbee63d743.jpg)

Figure 8: Sensitivity to the graph Dirichlet-energy weight τ.  
![](images/ec8b15f7c6d7017c1d90bf5de3162b2a7ae6766ba40d24d26e06eb0db860f89c.jpg)  
Figure 9: Inference time comparison on SMAP (left) and SMD (right).

## E.10 INFERENCE EFFICIENCY

In this section, we evaluate the inference efficiency of GRASP. Specifically, we compare the inference time of anomaly detectors on the test set of SMAP and SMD. For GRASP, we use the default setting with $| \mathcal { R } _ { \mathrm { a d } } | = 1 0$ flow-time evaluation points and $| \mathcal { X } _ { \mathrm { a d } } | = 5$ source samples. Note that the eigenvalue decomposition is not included in the inference time computation as it is only performed once for each dataset before training and reused during inference.

As shown in Figure 9, GRASP requires 38.4 seconds on SMAP and 31.7 seconds on SMD to score the full test sets. GRASP demonstrates faster inference compared with A-Transformer, CATCH, and GCAD on both datasets, but slower than the remaining baselines because its anomaly score aggregates multiple flow-time evaluations and source samples. For application scenarios where faster inference is preferred, GRASP also provides a controllable trade-off between inference efficiency and detection performance. Specifically, as discussed in Section E.8, GRASP can be further accelerated by reducing the flow-time evaluation points $| \mathcal { R } _ { \mathrm { a d } } |$ and source samples $| { \mathcal { X } } _ { \mathrm { a d } }$ | with a bit of compromised performance.