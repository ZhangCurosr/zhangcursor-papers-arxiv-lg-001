# CacheDyG: Decoupling Temporal Propagation for Eficient Dynamic Graph Learning

PinHeng Zong<sup>1</sup> and Ye Yuan<sup>∗1</sup>

<sup>1</sup>College of Computer and Information Science, Southwest University, Chongqing, China, zongph@email.swu.edu.cn, yuanyekl@swu.edu.cn

Preprint. Accepted for presentation/publication at ADMA 2026. This manuscript is not the Springer Version of Record.

## Abstract

Dynamic graphs are widely used to model time-evolving relational systems in real-world applications. Dynamic graph neural networks provide an efective framework for capturing both structural dependencies and temporal dynamics in such data. However, they typically intertwine temporal graph propagation with every optimization epoch and often maintain large trainable representations for each node-time pair. This design repeatedly recomputes largely unchanged historical structures, leading to substantial training and parameter overhead. To address this critical issue, we propose CacheDyG, a Cache-refine framework for eficient Dynamic Graph learning. Specifically, it decouples temporal propagation from routine parameter updates by constructing a time-ordered temporal dependency cache that stores graph-aware node-time representations in non-trainable bufers. During standard training epochs, CacheDyG reads from the cache and updates only a lightweight cache refiner, an adaptive residual gate, and the link predictor. Selective cache refresh further keeps cached representations aligned with the supervised objective while avoiding epoch-wise sparse propagation. Experiments on five dynamic graph benchmarks show that CacheDyG adopts substantially fewer trainable parameters and lower runtime to obtain more competitive predictive performance than baselines. These results demonstrate that cache-based decoupling provides an efective principle for scalable dynamic graph learning.

Keywords: Dynamic graph learning; Dynamic graph neural networks; Link prediction; Cache mechanism; Decoupled parameter update

## 1 Introduction

Dynamic graphs provide a natural representation for relational systems whose interactions evolve over time, including social networks, recommender systems, transaction networks, communication systems, and biological interaction networks [27, 29]. Unlike static graphs, dynamic graphs exhibit both structural dependencies among entities and temporal dependencies across diferent stages of evolution. Learning from such data therefore requires models to preserve graph-structured information while capturing how relational patterns change over time. As real-world dynamic graphs grow in both scale and temporal duration, computationally and memory-eficient learning becomes increasingly important.

Dynamic graph neural networks provide an efective framework for jointly modeling structural and temporal dependencies. Snapshot-based methods, such as DySAT [1], EvolveGCN [3], and VGRNN [4], combine graph propagation with temporal attention, recurrent parameter evolution, or latent-state transitions. Continuous-time methods, including DyRep [6], JODIE [7], TGAT [9], TGN [10], CAW [15], GraphMixer [12], and DyGFormer [13], learn from timestamped interaction streams through event-driven memory, temporal neighborhoods, or sequence encoders. Although these methods have achieved strong predictive performance, they are generally designed to enhance representation expressiveness rather than to eliminate computational redundancy during optimization.

This redundancy is particularly pronounced in transductive snapshot-based training. The historical graph sequence remains largely unchanged across optimization epochs, yet temporal graph propagation is often recomputed together with every gradient update. Moreover, some models maintain directly trainable representations for individual node-time pairs, causing the parameter size to grow with both the number of nodes and the number of snapshots. These two design choices create distinct but related eficiency bottlenecks: repeated sparse propagation over reusable historical structures increases runtime, while large node-time parameter tables increase memory consumption and optimization cost. As dynamic graphs become larger or span longer time horizons, such tightly coupled training pipelines can become unnecessarily expensive or even infeasible.

Our key observation is that temporal graph propagation and supervised parameter optimization do not need to operate at the same frequency. Historical graph structures can be propagated occasionally and stored as reusable representations, whereas lightweight task-specific modules can still be updated at every epoch. Based on this observation, we propose CacheDyG, a cache-refine framework that decouples temporal propagation from routine parameter updates. CacheDyG first constructs a time-ordered Temporal Dependency Cache using temporal mixing, so that each snapshot aggregates only current and past information. Graph-aware propagation is then performed to produce reusable node-time representations stored in non-trainable bufers. During ordinary training epochs, the model reads these cached representations and updates only a compact nodedomain frequency-domain cache refiner, an adaptive residual gate, and a lightweight link predictor. A selective cache-refresh mechanism periodically incorporates task-aware refined representations back into the cache, keeping it aligned with the supervised objective without returning sparse graph propagation to the inner optimization loop.

This cache-based separation changes the role of graph propagation in dynamic graph learning. Instead of being repeatedly executed as part of every parameter update, propagation becomes an amortized representation-construction step, while optimization focuses on correcting and scoring the cached signal. The resulting architecture reduces trainable parameter growth, avoids repeated computation over unchanged historical snapshots, and preserves the structural and temporal evidence required for future-link prediction.

We evaluate CacheDyG on the task of one-step future-link prediction across five dynamic graph benchmarks, considering average precision, ROC-AUC, trainable parameter count, and runtime. The results show that CacheDyG achieves the best average precision on all evaluated datasets while using only 19.106K trainable parameters. It also obtains the lowest runtime and remains feasible on larger datasets where memory-intensive baselines fail. Ablation and sensitivity studies further demonstrate that temporal mixing, cache-style propagation, lightweight refinement, and selective refresh jointly contribute to the final accuracy-eficiency trade-of.

Contributions. The main contributions of this work are summarized as follows:

1. We identify a fundamental eficiency mismatch in transductive snapshot-based dynamic graph learning: reusable historical graph structures are repeatedly propagated during optimization, while directly trainable node-time representations can cause parameter and memory costs to grow rapidly with graph size and temporal length.

2. We propose CacheDyG, a cache-refine framework that decouples temporal graph propagation from routine parameter updates through a Temporal Dependency Cache, non-trainable graphaware node-time bufers, lightweight frequency-domain refinement, adaptive residual gating, and selective cache refresh.

3. We provide computational and empirical evidence that the proposed decoupling amortizes sparse temporal propagation and yields a favorable accuracy-eficiency trade-of on five dynamic graph datasets in terms of predictive performance, trainable parameter size, runtime, and scalability.

## 2 Related Work

## 2.1 Dynamic Graph Learning

Dynamic graph learning studies how to represent and predict relations in networks whose edges, attributes, or node states evolve over time [27, 11, 17, 5, 23]. Earlier dynamic embedding methods learn temporally varying node representations from snapshot sequences or interaction histories [14, 26, 20]. DynamicTriad [18] models triadic closure processes, DynGEM [19] incrementally updates autoencoder-based graph embeddings, dyngraph2vec [21] captures temporal graph evolution with deep representation learning, and CTDNE [16] extends random-walk-based embeddings to continuoustime dynamic networks. These methods establish the importance of temporal dependency for downstream tasks such as link prediction, but they generally focus on learning temporal embeddings rather than explicitly reducing repeated graph propagation in snapshot-based training.

Recent dynamic graph learning also includes interaction-sequence models that encode temporal events or historical neighborhoods for future-edge prediction. JODIE [7] learns trajectories of useritem embeddings, DyRep [6] models evolving interaction processes, CAW [15] represents temporal networks through causal anonymous walks, GraphMixer [12] shows that simple temporal link encoders can be competitive, and DyGFormer [13] uses historical first-hop interactions with Transformerstyle sequence modeling. Frequency-oriented dynamic graph methods such as FreeDyG [25] and SGD-DYG [8] further explore global or spectral dependency modeling. Compared with these works, CacheDyG focuses on fixed-node snapshot sequences and treats reusable historical graph computation as the main eficiency opportunity. Its goal is not to introduce a heavier temporal encoder, but to cache temporal graph propagation and train only a lightweight correction and scoring module.

## 2.2 Dynamic Graph Neural Networks

Dynamic graph neural networks combine graph neural aggregation with temporal modeling to learn representations over evolving graph structures [11]. Snapshot-based dynamic GNNs apply graphbased propagation across a sequence of graph snapshots. DySAT [1] uses structural and temporal selfattention, EvolveGCN [3] evolves graph convolution parameters with recurrent networks, VGRNN [4] introduces recurrent latent variables, and ROLAND [22] updates hierarchical node states over time. Other compact or convolutional dynamic GNN baselines, such as WinGNN [24] and GTCN [2], also aim to model temporal graph evolution with diferent eficiency-accuracy trade-ofs.

Continuous-time temporal GNNs represent dynamic graphs as timestamped interaction streams. TGAT [9] uses temporal attention for inductive representation learning, while TGN [10] introduces memory modules and graph-based operators for event-driven dynamic graphs. These models are powerful for asynchronous temporal reasoning, but their design objectives difer from the transductive snapshot setting studied here. In snapshot-based training, the same historical graph structures are repeatedly reused across epochs, making cached propagation especially beneficial.

Unlike existing dynamic GNNs that couple temporal propagation with epoch-wise optimization, CacheDyG caches reusable graph computation and trains only lightweight refinement and prediction modules.

## 2.3 Related Structure-Aware and Eficient Representation Learning

Beyond canonical dynamic-graph architectures, recent work has explored complementary ways to represent evolving or spatiotemporal data. Dynamic graph mixers and spatiotemporal graph–tensor models explicitly couple temporal variation with relational structure [35, 36, 40, 42, 43, 44, 47, 49, 53, 55, 60]. These studies motivate the broader view that temporal dependency can be captured through graph, tensor, and latent-factor formulations, whereas CacheDyG specifically targets the repeated propagation cost that arises when a fixed sequence of graph snapshots is optimized over many epochs.

A second line of work develops structure-aware graph representation modules for diferent learning settings. Representative examples include modularized graph convolution [37], neural community search [39], attributed graph clustering [46], graph-based multi-source association prediction [48], collaborative-distillation GNNs for recommendation [51], graph linear convolution pooling [52], and graph-bi-regularized matrix factorization for community detection [59]. These methods emphasize how structural priors or task-specific graph operators can improve representation quality. CacheDyG is orthogonal to these designs: its main objective is to separate reusable graph propagation from routine parameter updates.

Eficiency has also been studied from compression, factorization, and optimization perspectives. Examples include low-rank tensor compression [34], auto-encoding and neural Tucker factorization [38, 54], surveys of parallel optimization for high-dimensional factorization [41], attention-based and accelerated neural tensor factorization [45, 50], fine-grained regularization of tensor factorization [56], tensor-decomposition-based eficient detection [57], adaptively accelerated parallel stochastic-gradient optimization [58], and controller- or population-based refinement of latent-factor learning [31, 32, 33]. These works address eficiency or representation compactness through optimization and model design. In contrast, CacheDyG reduces redundant computation by amortizing sparse temporal graph propagation through a reusable non-trainable cache.

## 3 Methodology

This section presents CacheDyG, whose design is summarized in Fig. 1. The framework first builds a temporally mixed graph-aware cache, then adapts the cached representations with a lightweight node-domain frequency refiner for link prediction. In cache mode, sparse graph propagation is executed only during cache construction or refresh; ordinary epochs optimize only the refinement and prediction modules.

## 3.1 Problem Formulation

Let

$$
\boldsymbol { \mathcal { G } } = \{ G _ { t } \} _ { t = 1 } ^ { T } , \qquad \boldsymbol { G } _ { t } = \{ V , E _ { t } \} ,\tag{1}
$$

denote a sequence of graph snapshots, where $V = \{ v _ { 1 } , \ldots , v _ { N } \}$ is a fixed node set after preprocessing and $E _ { t }$ is the edge set observed at time step t. We use $B _ { t } \in \mathbb { R } ^ { N \times N }$ to denote the raw adjacency matrix of $G _ { t } ,$ and $A _ { t } \in \mathbb { R } ^ { N \times N }$ to denote its normalized sparse adjacency matrix. A standard normalization with self-loops can be written as

$$
A _ { t } = D _ { t } ^ { - \frac { 1 } { 2 } } ( B _ { t } + I ) D _ { t } ^ { - \frac { 1 } { 2 } } ,\tag{2}
$$

where $D _ { t }$ is the degree matrix of $B _ { t } + I$ . For large sparse temporal graphs, the same formulation can be implemented with sparse-safe normalization without materializing dense self-loop structures.

The task is one-step link prediction. Given historical snapshots up to time t, the model predicts whether a candidate node pair $( u , v )$ will be connected in the next snapshot. The prediction target is

$$
y _ { t , u , v } = { \left\{ \begin{array} { l l } { 1 , } & { ( u , v ) \in E _ { t + 1 } , } \\ { 0 , } & { ( u , v ) \notin E _ { t + 1 } , } \end{array} \right. }\tag{3}
$$

and the model estimates

$$
\hat { y } _ { t , u , v } = p \left( y _ { t , u , v } = 1 \mid \mathcal { G } _ { \leq t } \right) .\tag{4}
$$

Positive samples are observed temporal edges, and negative samples are drawn from unobserved node pairs under the same temporal evaluation protocol.

## 3.2 Temporal Dependency Cache

CacheDyG represents temporal dependency using a temporal mixing matrix $M \in \mathbb { R } ^ { T \times T }$ . The matrix is lower triangular, row stochastic, and band-limited:

$$
M _ { t , s } = 0 \quad \mathrm { i f } \ s > t ,\tag{5}
$$

and each row aggregates information from the current snapshot and recent historical snapshots. More generally, the entries of M are obtained by normalizing a non-negative decay kernel,

$$
M _ { t , s } = \frac { \kappa ( t - s ) \mathbf { 1 } [ 0 \leq t - s < b ] } { \sum _ { q = 1 } ^ { T } \kappa ( t - q ) \mathbf { 1 } [ 0 \leq t - q < b ] } ,\tag{6}
$$

where $b$ controls the temporal support and $\kappa ( \cdot )$ determines the relative importance of historical snapshots. This construction prevents future information leakage.

![](images/aa4607b8047f5c69cab31aecdaf338ab7341ecf03160b702a1d98bb84f64933e.jpg)  
Figure 1: Overall framework of CacheDyG. TDC constructs a temporally mixed graph-aware cache S; FDCR applies node-domain frequency refinement and residual gating; the refined representations are then used for link prediction while sparse propagation remains outside ordinary training epochs.

Let $X \in \mathbb { R } ^ { T \times N \times d }$ be the node cache tensor, where $X _ { t } \in \mathbb { R } ^ { N \times d }$ stores the cached node representations at time step t. In cache mode, X is maintained as a non-trainable bufer rather than as a directly optimized embedding table.

CacheDyG first constructs temporally mixed features and temporally mixed graph structures:

$$
\bar { X } _ { t } = \sum _ { s = 1 } ^ { T } M _ { t , s } X _ { s } ,\tag{7}
$$

$$
\bar { A } _ { t } = \sum _ { s = 1 } ^ { T } M _ { t , s } A _ { s } .\tag{8}
$$

The graph-aware cache is then obtained through sparse propagation:

$$
S _ { t } = \bar { A } _ { t } \bar { X } _ { t } , \qquad t = 1 , \ldots , T .\tag{9}
$$

The resulting tensor $S \in \mathbb { R } ^ { T \times N \times d }$ contains structural and temporal information. Since ${ \bar { A } } _ { t }$ depends only on the snapshot sequence and temporal mixing matrix, it can be precomputed and reused, moving the sparse propagation outside the inner gradient-update loop.

## 3.3 Node-Domain Frequency-Domain Cache Refinement

The cache S encodes temporal propagation, but it is not directly optimized for the link prediction objective. CacheDyG therefore applies a lightweight refiner that calibrates the cached graph signa without performing additional sparse graph convolution in each epoch. The Fourier transform is applied along the node axis of each cache slice, following the node-axis spectral refinement convention used in lightweight frequency-based dynamic graph learning such as SGD-DYG [8]. In CacheDyG, this operation is used as representation calibration after TDC rather than as a replacement for temporal modeling. For each time step t, the cache matrix $S _ { t }$ is transformed along the node axis by a real-valued fast Fourier transform:

$$
\widehat { S } _ { t } = \mathcal { F } _ { N } ( S _ { t } ) ,\tag{10}
$$

where $\mathcal { F } _ { N } ( \cdot )$ denotes the Fourier transform over the node dimension and $\widehat { S } _ { t }$ is complex-valued. The snapshot/time axis is kept fixed in this operation; temporal dependency has already been encoded by the temporal mixing and graph-aware propagation above. The transformed representation is processed by a compact complex-valued feed-forward network:

$$
\widehat { R } _ { t } = \Phi _ { \Omega } ( \widehat { S } _ { t } ) ,\tag{11}
$$

where $\Phi _ { \Omega }$ consists of complex afine transformations and element-wise nonlinearities applied separately to the real and imaginary parts. The refined signal is mapped back to the original representation domain by the inverse transform:

$$
R _ { t } = \Psi \left( \mathcal { F } _ { N } ^ { - 1 } ( \widehat { R } _ { t } ) \right) ,\tag{12}
$$

where $\Psi ( \cdot )$ is a real-valued projection. The resulting $R _ { t }$ is therefore a node-domain correction term over a temporally mixed graph cache, not a separate temporal-frequency representation.

To preserve stable graph information, CacheDyG uses an adaptive residual gate computed from the cached representation:

$$
\Theta _ { t } = \sigma \left( g _ { \Omega } ( S _ { t } ) \right) ,\tag{13}
$$

where $g \Omega$ is a small gating network and $\Theta _ { t } \in ( 0 , 1 ) ^ { N \times 1 }$ is broadcast over the feature dimension. The final refined representation is

$$
H _ { t } = S _ { t } + \Theta _ { t } \odot R _ { t } ,\tag{14}
$$

where ⊙ denotes element-wise multiplication. The residual form keeps the cached graph signal as the default representation and injects node-domain corrections only when they improve the supervised objective.

## 3.4 Link Prediction Objective

For a candidate edge $( u , v )$ at time step t, CacheDyG extracts the corresponding refined node representations $h _ { t , u }$ and $h _ { t , v }$ from $H _ { t }$ . The link predictor is a lightweight binary classifier over the concatenated pair representation:

$$
\hat { y } _ { t , u , v } = \sigma \left( \boldsymbol { w } ^ { \top } [ h _ { t , u } \Vert h _ { t , v } ] + b \right) ,\tag{15}
$$

where ∥ denotes concatenation and $\sigma ( \cdot )$ is the sigmoid function.

Given a training set $\boldsymbol { \mathcal { S } }$ containing both positive and negative temporal node pairs, the model is optimized with the binary cross-entropy loss:

$$
\mathcal { L } = - \frac { 1 } { | \mathcal { S } | } \sum _ { ( t , u , v ) \in { \cal S } } \left[ y _ { t , u , v } \log \hat { y } _ { t , u , v } + ( 1 - y _ { t , u , v } ) \log ( 1 - \hat { y } _ { t , u , v } ) \right] .\tag{16}
$$

In the standard cache-mode loop, gradient descent updates only the refiner and predictor. The cached tensor is detached from the computational graph and is updated only through the refresh mechanism below.

## 3.5 Cache-Refresh Training

A completely fixed cache may become misaligned with the supervised objective. CacheDyG therefore refreshes the cache periodically or when validation performance stagnates. Let $\mathcal { R } _ { \Omega } ( \cdot )$ denote the refinement mapping above. When a refresh is triggered, the model first computes a detached refined cache:

$$
Z = \mathrm { s t o p g r a d } \left( \mathcal { R } _ { \Omega } ( S ) \right) ,\tag{17}
$$

and then updates the non-trainable node cache by combining the previous cache with the refined representation:

$$
X  Z .\tag{18}
$$

After updating X, the graph-aware cache S is rebuilt. Overall, cache-mode training builds $\{ \bar { A } _ { t } \} _ { t = 1 } ^ { T } .$ constructs S, optimizes the refiner and predictor with the binary cross-entropy objective, and refreshes X and S only when the refresh condition is met. The cache can therefore adapt during training while sparse propagation remains outside the inner loop.

## 3.6 Computational Analysis

Let $\begin{array} { r } { E = \sum _ { t = 1 } ^ { T } | E _ { t } | } \end{array}$ denote the total number of temporal edges and let Q be the number of optimization epochs. A conventional dynamic graph neural network that performs sparse propagation over all snapshots in every epoch incurs a propagation cost proportional to

$$
O ( Q E d ) ,\tag{19}
$$

up to model-dependent constants. By contrast, CacheDyG performs sparse propagation only when the cache is built or refreshed. If the cache is refreshed R times during training, the sparse propagation cost becomes

$$
\mathcal { O } ( R E d ) + \mathcal { O } ( Q C _ { \mathrm { r e f i n e } } ) ,\tag{20}
$$

where $C _ { \mathrm { r e f i n e } }$ is the cost of the lightweight node-domain frequency refinement and link prediction modules. Since cache refresh is substantially less frequent than epoch-level optimization, the dominant sparse graph computation is amortized across training.

## 4 Experiments

We evaluate CacheDyG from four perspectives: link prediction accuracy, component contribution, refresh-schedule robustness, and the accuracy-eficiency trade-of against representative dynamic graph models.

## 4.1 Dataset and Metrics

We evaluate CacheDyG on five dynamic graph datasets: Wiki-Eo, Digg, Alpha, DBLP, and StackOverflow, covering Wikipedia interactions [8], social-news replies [28], signed trust relations [30], academic relations [8], and Stack Exchange interactions [8]. The datasets are discretized into 60, 50, 60, 45, and 25 snapshots, respectively. After preprocessing, all datasets use a fixed node set and chronological ordering. We adopt a 70%/10%/20% train/validation/test split and predict links in snapshot (t + 1) given historical snapshots up to t.

Table 1: Main link prediction results on dynamic graph datasets.
<table><tr><td>Dataset</td><td>Metric</td><td>DySAT</td><td>ROLAND</td><td>EvolveGCN</td><td>WinGNN</td><td>GTCN</td><td>SGD-DYG</td><td>CacheDyG</td></tr><tr><td rowspan="2">Wiki-Eo</td><td>MAP</td><td>85.70±0.11</td><td>83.90±0.96</td><td>83.88±3.47</td><td>82.16±4.15</td><td>92.64±1.03</td><td>92.71±1.24</td><td>95.14±1.28</td></tr><tr><td>MAUC</td><td>81.26±0.71</td><td>82.52±1.37</td><td>74.47±6.20</td><td>81.21±4.07</td><td>91.12±1.64</td><td>91.32±1.52</td><td>94.82±0.57</td></tr><tr><td rowspan="2">Digg</td><td>MAP</td><td>77.10±0.38</td><td>73.30±1.61</td><td>74.56±2.31</td><td>72.75±1.69</td><td>75.86±0.77</td><td>76.76±0.71</td><td>77.87±0.30</td></tr><tr><td>MAUC</td><td>74.10±0.50</td><td>71.78±1.16</td><td>69.86±3.26</td><td>68.89±1.71</td><td>73.25±0.66</td><td>74.49±1.84</td><td>74.51±0.94</td></tr><tr><td rowspan="2">Alpha</td><td>MAP</td><td>86.89±2.22</td><td>86.98±7.19</td><td>86.60±1.98</td><td>87.38±3.46</td><td>91.71±0.56</td><td>91.86±0.09</td><td>93.02±1.08</td></tr><tr><td>MAUC</td><td>85.39±2.36</td><td>83.96±7.90</td><td>79.53±2.02</td><td>81.54±2.23</td><td>92.39±0.60</td><td>93.89±0.37</td><td>93.17±0.98</td></tr><tr><td rowspan="2">DBLP</td><td>MAP</td><td>OOM</td><td>61.81±0.56</td><td>63.46±3.73</td><td>62.32±0.67</td><td>61.37±0.58</td><td>60.87±0.27</td><td>67.32±0.59</td></tr><tr><td>MAUC</td><td>OOM</td><td>61.78±0.63</td><td>53.64±4.70</td><td>62.92±0.90</td><td>62.47±0.63</td><td>60.95±0.33</td><td>57.38±4.81</td></tr><tr><td rowspan="2">StackOverflow</td><td>MAP</td><td>OOM</td><td>88.01±1.33</td><td>89.07±2.16</td><td>74.34±1.08</td><td>90.28±0.88</td><td>90.38±0.84</td><td>92.05±0.92</td></tr><tr><td>MAUC</td><td>00M</td><td>85.21±2.59</td><td>86.30±1.84</td><td>73.44±2.83</td><td>88.44±0.93</td><td>88.46±1.27</td><td>89.67±1.03</td></tr></table>

For evaluation, each positive edge is paired with one unobserved node pair as a negative sample. We report average precision (AP) as the primary metric and ROC-AUC as a complementary ranking metric. MAP denotes the mean average precision over five random seeds, and MAUC denotes the mean ROC-AUC. Results are reported as mean and standard deviation over repeated runs. Trainable parameter counts exclude non-trainable cache bufers, and runtime is reported as training, validation, and total time, and a method that cannot complete under the same environment is marked as out-of-memory (OOM).

## 4.2 Experiment Settings

All methods use the same chronological split, negative samples, and evaluation protocol. CacheDyG is optimized with Adam using a learning rate of $1 \times 1 0 ^ { - 2 }$ , weight decay of $5 \times 1 0 ^ { - 4 }$ , a maximum of 200 epochs, and early stopping with a patience of 25. The random seeds are from 2024–2028. The input feature dimension is 8, the predictor contains one 16-dimensional hidden layer, and the temporal bandwidth is 20. The graph and frequency modules use a dropout rate of 0.75.

Unless otherwise specified, CacheDyG uses temporal mixing, a non-trainable $X _ { \mathrm { c a c h e } } ,$ cache graph mode, and the full node-domain frequency-domain refiner. The refiner and adaptive scaling networks have hidden dimensions 128 and 32, respectively. Thus, CacheDyG contains 19106 trainable parameters. Cache refresh is performed every 8 epochs for the first four warm-up refreshes and is subsequently triggered after 10 epochs without validation improvement. During ordinary epochs, only the refiner and link predictor are optimized; sparse graph propagation is executed only when the cache is constructed or refreshed.

## 4.3 Baselines

We compare CacheDyG with six representative baselines: DySAT for structural-temporal selfattention, ROLAND [22] for rolling node-state updates, EvolveGCN for recurrent evolution of graph convolution parameters, WinGNN [24] for compact temporal interaction modeling, GTCN [2] for graph-temporal convolution, and SGD-DYG [8] for lightweight frequency-based dynamic graph learning.

## 4.4 Comparative Performance Analysis

Table 1 reports the link prediction performance on all five datasets.

Table 2: Trainable parameter comparison on all datasets. Values are in K.
<table><tr><td>Model</td><td>Wiki-Eo</td><td>Digg</td><td>Alpha</td><td>DBLP</td><td>StackOverflow</td></tr><tr><td>DySAT</td><td>2462.656</td><td>8328.944</td><td>2566.304</td><td>OOM</td><td>OOM</td></tr><tr><td>ROLAND</td><td>69.409</td><td>245.865</td><td>63.825</td><td>2884.009</td><td>20676.305</td></tr><tr><td>EvolveGCN</td><td>2393.977</td><td>8261.161</td><td>2497.625</td><td>89315.009</td><td>702896.121</td></tr><tr><td>WinGNN</td><td>59.408</td><td>243.976</td><td>61.936</td><td>2882.120</td><td>20673.872</td></tr><tr><td>GTCN</td><td>2390.689</td><td>8257.937</td><td>2494.337</td><td>89311.721</td><td>381446.465</td></tr><tr><td>SGD-DYG</td><td>2390.753</td><td>8257.873</td><td>2494.401</td><td>89311.785</td><td>351446.465</td></tr><tr><td>CacheDyG</td><td>19.106</td><td>19.106</td><td>19.106</td><td>19.106</td><td>19.106</td></tr></table>

Table 3: Train, validation, and total runtime of one epoch on non-StackOverflow datasets. Values are in seconds; each cell reports train/validation/total.
<table><tr><td>Model</td><td>Wiki-Eo</td><td>Digg</td><td>Alpha</td><td>DBLP</td></tr><tr><td>DySAT</td><td>0.142/0.067/0.210</td><td>0.287/0.083/0.370</td><td>0.157/0.045/0.202</td><td>OOM</td></tr><tr><td>ROLAND</td><td>0.234/0.132/0.367</td><td>0.195/0.103/0.299</td><td>0.235/0.120/0.355</td><td>0.259/0.137/0.396</td></tr><tr><td>EvolveGCN</td><td>0.288/0.063/0.351</td><td>0.378/0.080/0.458</td><td>0.299/0.046/0.345</td><td>0.465/0.065/0.530</td></tr><tr><td>WinGNN</td><td>0.204/0.057/0.261</td><td>0.217/0.050/0.267</td><td>0.132/0.039/0.171</td><td>0.308/0.097/0.405</td></tr><tr><td>GTCN</td><td>0.105/0.030/0.135</td><td>0.133/0.062/0.195</td><td>0.111/0.023/0.134</td><td>0.302/0.102/0.434</td></tr><tr><td>SGD-DYG</td><td>0.197/0.114/0.312</td><td>0.205/0.133/0.338</td><td>0.202/0.117/0.319</td><td>1.330/0.452/1.780</td></tr><tr><td>CacheDyG</td><td>0.005/0.002/0.007</td><td>0.009/0.005/0.014</td><td>0.006/0.002/0.008</td><td>0.075/0.050/0.125</td></tr></table>

Overall efectiveness. Table 1 shows that CacheDyG achieves the best AP on all five datasets, with gains of 2.43, 0.77, 1.16, 3.86, and 1.67 percentage points over the strongest AP baseline on Wiki-Eo, Digg, Alpha, DBLP, and StackOverflow, respectively. The ROC-AUC results are also competitive: CacheDyG ranks first on Wiki-Eo, Digg, and StackOverflow and second on Alpha. On DBLP, the AP gain is clear while ROC-AUC is lower than the best baseline, suggesting that the method is especially strong for positive-edge ranking but not uniformly dominant under every ranking metric.

Scalability evidence. The results on DBLP and StackOverflow are particularly relevant to eficiency. DySAT runs out of memory on both datasets, whereas CacheDyG remains feasible and achieves the highest AP. This supports the main design goal: cached temporal propagation and lightweight refinement can preserve predictive quality when repeated sparse propagation becomes costly. We evaluate eficiency through trainable parameter size and wall-clock runtime. Table 2 reports trainable parameters in thousands, and Tables 3–4 report training, validation, and tota runtime. StackOverflow is listed separately because of its larger scale.

Parameter eficiency. Table 2 shows that CacheDyG uses 19.106K trainable parameters on every dataset because node-time representations are stored as non-trainable bufers. On StackOverflow, the smallest baseline already exceeds 20,000K parameters, more than 1,082× the size of CacheDyG.

Table 4: Train, validation, and total runtime of one epoch on StackOverflow. Values are in seconds.
<table><tr><td>Model</td><td>Train time</td><td>Validation time</td><td>Total time</td></tr><tr><td>DySAT</td><td>OOM</td><td>OOM</td><td>OOM</td></tr><tr><td>ROLAND</td><td>92.058</td><td>38.221</td><td>130.280</td></tr><tr><td>EvolveGCN</td><td>164.448</td><td>24.947</td><td>189.395</td></tr><tr><td>WinGNN</td><td>84.974</td><td>30.513</td><td>115.486</td></tr><tr><td>GTCN</td><td>88.541</td><td>75.793</td><td>164.334</td></tr><tr><td>SGD-DYG</td><td>304.999</td><td>180.505</td><td>485.503</td></tr><tr><td>CacheDyG</td><td>65.902</td><td>31.402</td><td>97.304</td></tr></table>

Table 5: Ablation study on Wiki-Eo. Parameter values are in K, with multiples relative to the full CacheDyG in parentheses.
<table><tr><td>Model variant</td><td>AP</td><td>ROC-AUC</td><td>Parameters</td><td>Train time (s)</td></tr><tr><td>Full CacheDyG</td><td>96.14</td><td>96.11</td><td>19.106 (1.0×)</td><td> $7 . 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>w/o Cache-style Prop.</td><td>87.46</td><td>84.42</td><td>2390.753 (125.1×)</td><td> $4 . 8 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>w/o Refiner</td><td>50.80</td><td>57.47</td><td>0.017 (0.001×)</td><td> $2 . 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Trainable  $X _ { \mathrm { c a c h e } }$ </td><td>92.69</td><td>91.21</td><td>2401.226 (125.7×)</td><td> $5 . 2 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>w/o Temporal Mixing</td><td>90.31</td><td>86.37</td><td>5.434 (0.28×)</td><td> $5 . 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>w/o Cache Refresh</td><td>81.52</td><td>78.73</td><td>19.106 (1.0×)</td><td> $7 . 0 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>TGC Graph Mode</td><td>94.39</td><td>92.13</td><td>22.642 (1.2×)</td><td> $5 . 3 \times 1 0 ^ { - 2 }$ </td></tr></table>

Runtime eficiency. On Wiki-Eo, Digg, Alpha, and DBLP, CacheDyG obtains the lowest total runtime, with speedups of approximately 19.3×, 13.9×, 16.8×, and 3.2× over the fastest baseline on each dataset. On StackOverflow, it completes in 97.304 seconds, lower than WinGNN’s 115.486 seconds, while also achieving the highest AP. Together, the accuracy and eficiency results support cache-refine learning for transductive snapshot sequences in which historical graph structures are repeatedly reused.

Efectiveness of the Lightweight Design. CacheDyG removes redundant trainable capacity without discarding useful graph information. The TDC stores reusable graph representations, the non-trainable cache avoids node-time overparameterization, and the FDCR provides taskspecific correction with few parameters. Residual gating and selective refresh preserve stable cached signals while maintaining alignment with the prediction objective, explaining the favorable accuracy-eficiency trade-of.

## 4.5 Ablation Study

We conduct ablation experiments on Wiki-Eo to isolate the contribution of each design choice. The variants remove cache-style propagation, remove the refiner, make $X _ { \mathrm { c a c h e } }$ trainable, remove temporal mixing, remove cache refresh, or keep graph convolution inside the refinement stage (TGC Graph Mode). All variants use the same data split and evaluation protocol.

![](images/2e0eca4ccc384e2bd932bc08cfdfccc1440bb93db2fcc29b85536b521e29e1db.jpg)  
Figure 2: Refresh-schedule heatmaps on Wiki-Eo. The top row reports MAUC and the bottom row reports MAP.

Component efects. Table 5 shows that removing cache-style propagation substantially reduces AP and ROC-AUC while increasing the parameter count by more than two orders of magnitude. Removing the refiner causes the largest performance drop, indicating that the raw cache must be calibrated toward the link prediction objective. Making $X _ { \mathrm { c a c h e } }$ trainable also underperforms the full model despite using far more parameters, which supports the use of a non-trainable cache with compact refinement.

Temporal dependency and decoupling. Removing temporal mixing or cache refresh reduces performance, confirming that both cross-snapshot dependency and adaptive cache evolution are important. TGC Graph Mode remains competitive but is slower and uses more parameters than the full cache graph mode. Thus, graph-aware propagation is useful, but it need not be executed inside every ordinary training epoch.

## 4.6 Parameter Sensitivity Analysis

We visualize the sensitivity of the cache-refresh schedule on Wiki-Eo. The experiment varies warm-up refreshes, refresh interval, and endurance threshold over {2, 4, 6, 8, 10}, yielding 125 combinations with all other settings fixed. This analysis is dataset-specific and is not intended to claim that the same landscape holds across all datasets.

Fig. 2 shows a broad high-performance region rather than a single isolated optimum. Most moderate configurations maintain AP and ROC-AUC around or above 0.96, while extreme refresh schedules are more likely to reduce performance. Fig. 3 further examines the interaction between the refinement number and scale number. The MAP and MAUC bars remain high across multiple moderate combinations, indicating that the cache-refinement module is not dependent on a single fragile hyperparameter choice. Very large or imbalanced settings can introduce unnecessary correction noise, but the overall trend supports a stable middle range for refinement and scaling. On Wiki-Eo, cache refresh and cache refinement are therefore not highly sensitive once the hyperparameters remain in a moderate range.

![](images/cfa1c6916f722e865afc6c871fb04ec8b711c0abbbd9f95019200a541587d58e.jpg)  
Figure 3: Sensitivity of MAP and MAUC to the refinement number and scale number.

## 5 Conclusion

This paper presented CacheDyG, which decouples temporal graph propagation from routine parameter updates by storing graph-aware node-time representations in non-trainable caches and optimizing only lightweight refinement and prediction modules. Selective refresh keeps the cache aligned with the supervised objective. Experiments on five datasets demonstrate competitive link-prediction accuracy, substantially fewer trainable parameters, and lower runtime, supporting cache-based decoupling as an eficient design for fixed-node snapshot dynamic graphs.

## References

[1] Sankar, A., Wu, Y., Gou, L., Zhang, W., Yang, H.: DySAT: Deep Neural Representation Learning on Dynamic Graphs via Self-Attention Networks. In: WSDM, pp. 519–527 (2020).

[2] Wang, L., Yuan, Y.: Tensor Graph Convolutional Network for Dynamic Graph Representation Learning. In: ISAS, pp. 1–5 (2024).

[3] Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., Schardl, T.B., Leiserson, C.E.: EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs. In: AAAI, pp. 5363–5370 (2020).

[4] Hajiramezanali, E., Hasanzadeh, A., Narayanan, K.R., Dufield, N., Zhou, M., Qian, X.: Variational Graph Recurrent Neural Networks. In: NeurIPS (2019).

[5] Yuan, Y., Wang, Y., Luo, X.: A Node-Collaboration-Informed Graph Convolutional Network for Highly Accurate Representation to Undirected Weighted Graph. IEEE Trans. Neural Netw. Learn. Syst. 36(6), 11507–11519 (2025).

[6] Trivedi, R.S., Farajtabar, M., Biswal, P., Zha, H.: DyRep: Learning Representations over Dynamic Graphs. In: ICLR (2019).

[7] Kumar, S., Zhang, X., Leskovec, J.: Predicting Dynamic Embedding Trajectory in Temporal Interaction Networks. In: KDD, pp. 1269–1278 (2019).

[8] Han, M., Wang, L., Yuan, Y., Luo, X.: SGD-DyG: Self-Reliant Global Dependency Apprehending on Dynamic Graphs. In: KDD, pp. 802–813 (2025).

[9] Xu, D., Ruan, C., Korpeoglu, E., Kumar, S., Achan, K.: Inductive Representation Learning on Temporal Graphs. In: ICLR (2020).

[10] Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., Bronstein, M.M.: Temporal Graph Networks for Deep Learning on Dynamic Graphs. arXiv:2006.10637 (2020).

[11] Wang, L., Yuan, Y., Luo, X.: Graph Tensor Convolutional Network. IEEE Trans. Syst. Man Cybern. Syst. doi:10.1109/TSMC.2026.3655418 (2026).

[12] Cong, W., Zhang, S., Kang, J., Yuan, B., Wu, H., Zhou, X., Tong, H., Mahdavi, M.: Do We Really Need Complicated Model Architectures for Temporal Networks? In: ICLR (2023).

[13] Yu, L., Sun, L., Du, B., Lv, W.: Towards Better Dynamic Graph Learning: New Architecture and Unified Library. In: NeurIPS (2023).

[14] Yuan, Y., Luo, X., Shang, M., Wang, Z.: A Kalman-Filter-Incorporated Latent Factor Analysis Model for Temporally Dynamic Sparse Data. IEEE Trans. Cybern. 53(9), 5788–5801 (2023).

[15] Wang, Y., Chang, Y.-Y., Liu, Y., Leskovec, J., Li, P.: Inductive Representation Learning in Temporal Networks via Causal Anonymous Walks. In: ICLR (2021).

[16] Nguyen, G.H., Lee, J.B., Rossi, R.A., Ahmed, N.K., Koh, E., Kim, S.: Continuous-Time Dynamic Network Embeddings. In: WWW Companion, pp. 969–976 (2018).

[17] Wang, L., Yuan, Y., Luo, X.: Advanced High-Order Graph Convolutional Networks with Assorted Time-Frequency Transforms. IEEE/CAA J. Autom. Sinica 13(2), 394–408 (2026).

[18] Zhou, L., Yang, Y., Ren, X., Wu, F., Zhuang, Y.: Dynamic Network Embedding by Modeling Triadic Closure Process. In: AAAI, pp. 571–578 (2018).

[19] Goyal, P., Kamra, N., He, X., Liu, Y.: DynGEM: Deep Embedding Method for Dynamic Graphs. arXiv:1805.11273 (2018).

[20] Yuan, Y., Wang, S., Zhou, H., Wang, L., Luo, X.: A Novel Approach to Temporal QoS Estimation via Extended Kalman Filter-Incorporated Latent Feature Analysis. IEEE Trans. Serv. Comput. doi:10.1109/TSC.2026.3697552 (2026).

[21] Goyal, P., Chhetri, S.R., Canedo, A.: dyngraph2vec: Capturing Network Dynamics using Dynamic Graph Representation Learning. Knowledge-Based Systems 187, 104816 (2020).

[22] You, J., Du, T., Leskovec, J.: ROLAND: Graph Learning Framework for Dynamic Graphs. In: KDD, pp. 2358–2366 (2022).

[23] Wang, L., Liu, K., Yuan, Y.: GT-A2T: Graph Tensor Alliance Attention Network. IEEE/CAA J. Autom. Sinica 12(10), 2165–2167 (2025).

[24] Zhu, Y., Cong, F., Zhang, D., Gong, W., Lin, Q., Feng, W., Dong, Y., Tang, J.: WinGNN: Dynamic Graph Neural Networks with Random Gradient Aggregation Window. In: KDD, pp. 3650–3662 (2023).

[25] Tian, Y., Qi, Y., Guo, F.: FreeDyG: Frequency Enhanced Continuous-Time Dynamic Graph Model for Link Prediction. In: ICLR (2024).

[26] Yuan, Y., Shang, M., Luo, X.: Temporal Web Service QoS Prediction via Kalman Filter-Incorporated Dynamic Latent Factor Analysis. In: ECAI, pp. 561–568 (2020).

[27] Kazemi, S.M., Goel, R., Jain, K., Kobyzev, I., Sethi, A., Forsyth, P., Poupart, P.: Representation Learning for Dynamic Graphs: A Survey. Journal of Machine Learning Research 21(70), 1–73 (2020).

[28] Kunegis, J.: KONECT: The Koblenz Network Collection. In: WWW Companion, pp. 1343–1350 (2013).

[29] Yuan, Y., Luo, X., Shang, M., Wu, D.: A Generalized and Fast-Converging Nonnegative Latent Factor Model for Predicting User Preferences in Recommender Systems. In: The Web Conference, pp. 498–507 (2020).

[30] Kumar, S., Spezzano, F., Subrahmanian, V.S., Faloutsos, C.: Edge Weight Prediction in Weighted Signed Networks. In: ICDM, pp. 221–230 (2016).

[31] Li, J., Yuan, Y., Luo, X.: Learning Error Refinement in Stochastic Gradient Descent-based Latent Factor Analysis via Diversified PID Controllers. IEEE Trans. Emerg. Top. Comput. Intell. 9(5), 3582–3597 (2025).

[32] Yuan, Y., Li, J., Luo, X.: A Fuzzy PID-Incorporated Stochastic Gradient Descent Algorithm for Fast and Accurate Latent Factor Analysis. IEEE Trans. Fuzzy Syst. 32(7), 4049–4061 (2024).

[33] Lyu, C., Ma, Z., Luo, X., Shi, Y.: Dynamic Stochastic Reorientation Particle Swarm Optimization for Adaptive Latent Factor Analysis in High-Dimensional Sparse Matrices. IEEE Trans. Knowl. Data Eng. 38(1), 222–234 (2026).

[34] He, Y., Luo, X.: Tensor Low-Rank Orthogonal Compression for Convolutional Neural Networks. IEEE/CAA J. Autom. Sinica 13(1), 227–229 (2026).

[35] Bi, F., He, T., Ong, Y.-S., Luo, X.: Discovering Spatio-Temporal-Individual Coupled Features from Nonstandard Tensors—A Novel Dynamic Graph Mixer Approach. IEEE Trans. Neural Netw. Learn. Syst. 36(11), 19834–19848 (2025).

[36] Bi, F., He, T., Luo, X.: Spatiotemporal Graph Neural Network-Incorporated Latent Factorization of Tensors for Dynamic QoS Estimation. IEEE/CAA J. Autom. Sinica, doi:10.1109/JAS.2025.125750 (2025).

[37] He, T., Duan, Z., Luo, X.: Modularized Graph Convolutional Network. IEEE/CAA J. Autom. Sinica, doi:10.1109/JAS.2025.125336 (2025).

[38] Tang, P., Luo, X., Woodcock, J.: Auto-Encoding Neural Tucker Factorization. IEEE Trans. Knowl. Data Eng. 37(10), 5795–5807 (2025).

[39] Lin, L., Li, Q., Qiao, M., Wang, Z., Zhao, J., Li, R.-H., Luo, X., Jia, T.: NCSAC: Efective Neural Community Search via Attribute-augmented Conductance. IEEE Trans. Knowl. Data Eng. 38(2), 1221–1235 (2026).

[40] Liao, X., Wu, H., He, T., Luo, X.: A Proximal-ADMM-incorporated Nonnegative Latent-Factorization-of-Tensors Model for Representing Dynamic Cryptocurrency Transaction Network. IEEE Trans. Syst. Man Cybern. Syst. 55(11), 8387–8401 (2025).

[41] Hu, Q., Wu, H., Luo, X.: A Comprehensive Review of Parallel Optimization Algorithms for High-Dimensional and Incomplete Matrix Factorization. IEEE/CAA J. Autom. Sinica 12(12), 2399–2426 (2025).

[42] Xu, X., Lin, M., Xu, Z., Luo, X.: A Sampling-Neighborhood-Regularized Latent Factorization of Tensor for Dynamic QoS Estimation. IEEE Trans. Netw. Serv. Manag. 23, 1707–1722 (2026).

[43] Liao, X., Wu, H., Luo, X.: A Novel Tensor Causal Convolution Network Model for Highly-Accurate Representation to Spatio-Temporal Data. IEEE Trans. Autom. Sci. Eng. 22, 19525– 19537 (2025).

[44] Wang, Q., Wu, H., Luo, X.: A Convolution Bias-Incorporated Nonnegative Latent Factorization of Tensors Model for Accurate Representation Learning to Dynamic Directed Graphs. IEEE Trans. Syst. Man Cybern. Syst. 55(12), 8902–8914 (2025).

[45] Xu, X., Lin, M., Xu, Z., Luo, X.: Attention-Mechanism-Based Neural Latent-Factorization-of-Tensors Mode. ACM Trans. Knowl. Discov. Data 19(4), 1–27 (2025).

[46] Yang, Y., Hu, L., Li, G., Li, D., Hu, P., Luo, X.: Link-based Attributed Graph Clustering via Approximate Generative Bayesian Learning. IEEE Trans. Syst. Man Cybern. Syst. 55(8), 5730–5743 (2025).

[47] Xu, X., Lin, M., Luo, X., Xu, Z.: An Adaptively Bias-Extended Non-negative Latent Factorization of Tensors Model for Accurately Representing the Dynamic QoS Data. IEEE Trans. Serv. Comput. 18(2), 603–617 (2025).

[48] Wu, M.-Y., Hu, P., You, Z.-H., Zhang, J., Hu, L., Luo, X.: Graph-Based Prediction of miRNA-Drug Associations with Multisource Information and Metapath Enhancement Matrices. IEEE J. Biomed. Health Inform., doi:10.1109/JBHI.2025.3558303 (2025).

[49] Hou, Y., Tang, P., Luo, X.: Multi-Aspect Self-Attending Neural Tucker Factorization for Spatiotemporal Representation Learning. IEEE/CAA J. Autom. Sinica, doi:10.1109/JAS.2025.125723 (2025).

[50] Li, W., Lin, M., Xu, X., Lin, L., Xu, Z., Luo, X.: Neural Non-Negative Latent Factorization of Tensors Model with Acceleration and Unconstraint. IEEE Trans. Syst. Man Cybern. Syst. 56(1), 164–178 (2026).

[51] Gou, J., Cheng, Y., Ma, B., Du, L., Luo, X., Yi, Z.: Multi-Scale Collaborative Distillation Graph Neural Networks for Session-Based Recommendation. IEEE Trans. Serv. Comput. 19(1), 504–517 (2026).

[52] Bi, F., He, T., Ong, Y.-S., Luo, X.: Graph Linear Convolution Pooling for Learning in Incomplete High-Dimensional Data. IEEE Trans. Knowl. Data Eng. 37(4), 1838–1852 (2025).

[53] Wu, D., Li, Z., Yu, Z., He, Y., Luo, X.: Robust Low-rank Latent Feature Analysis for Spatio-Temporal Signal Recovery. IEEE Trans. Neural Netw. Learn. Syst. 36(2), 2829–2842 (2025).

[54] Tang, P., Luo, X.: Neural Tucker Factorization. IEEE/CAA J. Autom. Sinica 12(2), 475–477 (2025).

[55] Yang, H., Lin, M., Chen, H., Luo, X., Xu, Z.: Latent Factor Analysis Model with Temporal Regularized Constraint for Road Trafic Data Imputation. IEEE Trans. Intell. Transp. Syst. 26(1), 724–741 (2025).

[56] Wu, H., Qiao, Y., Luo, X.: A Fine-Grained Regularization Scheme for Nonnegative Latent Factorization of High-Dimensional and Incomplete Tensors. IEEE Trans. Serv. Comput. 17(6), 3006–3021 (2024).

[57] Zeng, N., Li, X., Wu, P., Li, H., Luo, X.: A Novel Tensor Decomposition-based Eficient Detector for Low-altitude Aerial Objects with Knowledge Distillation Scheme. IEEE/CAA J. Autom. Sinica 11(2), 487–501 (2024).

[58] Qin, W., Luo, X., Zhou, M.: Adaptively-accelerated Parallel Stochastic Gradient Descent for High-Dimensional and Incomplete Data Representation Learning. IEEE Trans. Big Data 10(1), 92–107 (2024).

[59] Liu, Z., Luo, X., Zhou, M.: Symmetry and Graph Bi-regularized Non-Negative Matrix Factorization for Precise Community Detection. IEEE Trans. Autom. Sci. Eng. 21(2), 1406–1420 (2024).

[60] Chen, M., Qiao, Y., Wang, R., Luo, X.: A Generalized Nesterov’s Accelerated Gradient-Incorporated Non-negative Latent-factorization-of-tensors Model for Eficient Representation to Dynamic QoS Data. IEEE Trans. Emerg. Top. Comput. Intell. 8(3), 2386–2400 (2024).