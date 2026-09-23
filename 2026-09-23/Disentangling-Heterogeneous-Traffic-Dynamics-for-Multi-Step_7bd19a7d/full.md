# Disentangling Heterogeneous Traffic Dynamics for Multi-Step Traffic Forecasting via Adaptive Spectral Decomposition

Zijun Huang<sup>1</sup>, Chenrui Fu<sup>1</sup>, Wenhao Wang<sup>1</sup>, Xiaochuan Gou<sup>2</sup>, Chih-Chieh Hung<sup>3</sup>, and Guanyao Li<sup>1⋆</sup>

<sup>1</sup> Beijing Normal-Hong Kong Baptist University Dalian Maritime University 3 National Chung Hsing University

Abstract. Accurate multi-step traffic forecasting remains challenging because observed traffic signals contain heterogeneous temporal dynamics with different characteristics and levels of predictability. Existing approaches typically model these dynamics within a unified representation or rely on predefined decomposition rules, which may limit their ability to flexibly separate persistent patterns from rapidly varying fluctuations. To address this issue, we propose the Adaptive Decomposition Network (ADNet), a component-specific forecasting framework that adaptively disentangles traffic dynamics into dominant and residual components. ADNet introduces a learnable complementary spectral decomposition mechanism that determines the contribution of each frequency bin to the two components. Unlike hard frequency partitioning, every frequency bin can contribute to both components with different learned proportions, allowing the decomposition to be optimized jointly with the forecasting objective. The reconstructed components are then modeled by two dedicated spatiotemporal forecasting branches, and their predictions are integrated to generate the final multi-step forecast. Experiments on the Alameda and Orange regions of the TraffiDent dataset show that ADNet achieves the best performance in 20 of the 24 reported region–horizon–metric comparisons, with particularly clear gains at longer forecasting horizons. Capacity-controlled ablation experiments further show that the learnable decomposition substantially outperforms a fixed decomposition and provides additional improvements beyond the dual-branch architecture alone. These results demonstrate the effectiveness of adaptive decomposition and component-specific modeling for multi-step traffic forecasting.

Keywords: Traffic forecasting · Frequency-domain learning · Spectral decomposition

## 1 Introduction

Accurate traffic forecasting is a fundamental capability of intelligent transportation systems and supports a wide range of applications, including traffic management, congestion mitigation, route planning, and transportation resource allocation. Given historical traffic observations collected from multiple locations in a road network, multi-step traffic forecasting aims to predict traffic conditions over multiple future time steps. Compared with one-step forecasting, multi-step forecasting provides a longer view of future traffic evolution, enabling transportation systems to anticipate upcoming conditions and make proactive decisions. However, accurately predicting traffic states over extended forecasting horizons remains challenging because traffic observations comprise heterogeneous and continuously evolving temporal dynamics with different forecasting characteristics.

A fundamental challenge is that these heterogeneous traffic dynamics are inherently entangled within observed traffic signals. Some variations are relatively stable and persistent, reflecting regular mobility behaviors and slowly evolving traffic states, whereas others exhibit rapid and irregular fluctuations caused by transient congestion, changing travel demand, incidents, and other dynamic factors. These dynamics not only exhibit different temporal characteristics but also possess different levels of predictability. Relatively stable patterns can often provide reliable information for forecasting over longer horizons, while rapidly varying fluctuations are more difficult to extrapolate and may introduce increasing uncertainty as the forecasting horizon extends. When such heterogeneous dynamics are modeled together within a unified representation, a forecasting model is required to simultaneously capture patterns with substantially different behaviors, potentially making the forecasting task unnecessarily difficult. This motivates explicitly disentangling heterogeneous traffic dynamics before forecasting.

Existing traffic forecasting studies have made substantial progress by developing increasingly powerful models for capturing complex spatial and temporal dependencies. Most approaches focus on improving the ability of a unified model to represent the observed traffic sequence [1,2,3], while another line of research explicitly decomposes traffic signals into different components before prediction [4,7]. Although both directions have demonstrated promising performance, they still leave room for improvement in handling heterogeneous traffic dynamics. Unified modeling approaches may insufficiently distinguish dynamics with different forecasting characteristics, whereas decomposition-based approaches often rely on predefined assumptions or relatively rigid criteria to determine how different patterns should be separated. Consequently, an important question remains: how can heterogeneous traffic dynamics be explicitly disentangled while allowing the decomposition itself to adapt to the forecasting objective?

Addressing this question is challenging because heterogeneous traffic dynamics may not be separated by a clear and fixed boundary. Slowly varying patterns are generally more closely associated with persistent traffic dynamics, while rapidly changing variations are more likely to reflect irregular fluctuations. Nevertheless, this distinction is not absolute. Information at the same temporal frequency may contribute to different traffic dynamics to different degrees. Assigning a particular frequency exclusively to one component may therefore impose an overly restrictive representation and discard potentially useful information. Instead of determining the decomposition through a predefined frequency boundary, it is desirable to learn how different spectral patterns should contribute to different traffic dynamics directly from data.

Motivated by this observation, we propose the Adaptive Decomposition Network (ADNet)<sup>4</sup>, a multi-step traffic forecasting framework that adaptively disentangles heterogeneous traffic dynamics and models the resulting components separately. ADNet decomposes historical traffic observations into a dominant component and a residual component. The dominant component is intended to emphasize relatively persistent and predictable traffic dynamics, whereas the residual component focuses on irregular deviations and rapidly changing variations. Rather than imposing a hard separation between the two components, ADNet learns how traffic patterns at different frequencies should be distributed between them. In this way, the decomposition adapts to the intrinsic characteristics of traffic data and the forecasting objective instead of relying on manually defined frequency boundaries.

Specifically, ADNet transforms historical traffic observations into the frequency domain and introduces a learnable complementary spectral decomposition mechanism. For each frequency bin, the model learns how its information should be distributed between the dominant and residual components through complementary weights. Therefore, every frequency bin can contribute to both components with different learned proportions, allowing the two components to exploit information from the entire frequency spectrum while emphasizing different temporal dynamics. The decomposed spectra are then transformed back into the time domain and modeled through two dedicated forecasting branches. Finally, the predictions from the two branches are integrated to generate the multi-step traffic forecasts. Through this design, ADNet combines adaptive decomposition with component-specific forecasting to explicitly model heterogeneous traffic dynamics.

The main contributions of this work are summarized as follows:

– Component-Specific Forecasting Framework. We formulate multi-step traffic forecasting from the perspective of disentangling heterogeneous traffic dynamics and propose ADNet, a component-specific forecasting framework that explicitly decomposes traffic observations into dominant and residual components. The two components are modeled through dedicated forecasting branches, enabling traffic dynamics with different temporal characteristics and levels of predictability to be captured separately.

– Learnable Complementary Spectral Decomposition. We introduce a learnable complementary spectral decomposition mechanism that adaptively determines how different frequency patterns contribute to the dominant and residual components. Unlike predefined frequency boundaries or hard frequency selection, the proposed mechanism allows every frequency bin to contribute to both components with complementary learned proportions, providing a flexible decomposition of heterogeneous traffic dynamics.

– Empirical Validation and Ablation. Extensive experiments on real-world traffic data demonstrate the effectiveness of ADNet across different forecasting horizons and evaluation metrics. Capacity-controlled ablation experiments further investigate the contributions of the dual-branch architecture and adaptive decomposition mechanism, showing that learning the decomposition is important for improving multi-step traffic forecasting performance.

## 2 Related Work

## 2.1 Spatiotemporal Traffic Forecasting

Spatiotemporal modeling has become a dominant paradigm for traffic forecasting because traffic observations exhibit strong dependencies across both road-network locations and time. Representative methods include DCRNN, which integrates diffusion graph convolution with recurrent modeling, STGCN, which combines graph and temporal convolutions, and Graph WaveNet (GWN), which introduces dilated causal convolutions and adaptive graph learning [1,2,3]. More recent studies further explore adaptive and dynamic spatial dependencies. For example, D<sup>2</sup>STGNN decomposes traffic signals into diffusion and inherent components and models them together with dynamic graph learning [4].

These methods have substantially improved the modeling of complex spatiotemporal dependencies. Their primary focus, however, is on designing more expressive spatial or temporal representations. Our work addresses a complementary problem: how heterogeneous temporal dynamics within traffic observations can be disentangled before forecasting. ADNet decomposes the input into different temporal components and models them through separate forecasting branches, allowing component-specific modeling to complement existing spatiotemporal forecasting architectures.

## 2.2 Decomposition-Based Time-Series Forecasting

Time-series decomposition provides a natural way to model signals containing heterogeneous temporal patterns. Classical methods such as STL separate observations into trend, seasonal, and remainder components [15]. More recently, decomposition has been incorporated into deep forecasting models. Autoformer progressively separates trend and seasonal information [16], FEDformer combines series decomposition with frequency-enhanced modeling [17], and DLinear demonstrates the effectiveness of explicitly modeling decomposed components with simple forecasting architectures [18]. MSD-Mixer further introduces multi-scale decomposition to capture temporal patterns at different scales [19].

These studies demonstrate that separating heterogeneous temporal patterns can facilitate forecasting. However, the resulting components are typically defined according to a particular decomposition formulation, such as trend–seasonality or multi-scale structures. In contrast, ADNet learns how heterogeneous traffic dynamics should be decomposed according to their spectral contributions and the forecasting objective.

## 2.3 Frequency-Aware Traffic Forecasting

Frequency-domain modeling provides another perspective for characterizing heterogeneous temporal dynamics. StemGNN exploits graph and temporal spectral representations for multivariate forecasting, while FreTS models intra-series and inter-series dependencies through frequency-domain transformations [5,6]. FreqMoE further decomposes time-series representations into frequency components and employs specialized experts to capture different frequency characteristics [20].

Frequency-aware modeling has also been increasingly explored in traffic forecasting. STWave employs wavelet decomposition to separate relatively stable trends from fluctuating traffic variations [7]. DFDGCN exploits Fourier representations to alleviate temporal-shift effects when learning dynamic spatial dependencies [8]. LHFNet explicitly models low- and high-frequency traffic characteristics using different encoders [9], while HyperD decomposes traffic dynamics into periodic and residual components and models them through specialized mechanisms [10].

ADNet shares the general motivation of exploiting heterogeneous frequency characteristics but differs in how the decomposition is constructed. Rather than assigning spectral information exclusively to predefined components or frequency ranges, AD-Net learns complementary weights that determine the contribution of each frequency bin to both the dominant and residual components. Consequently, every frequency can contribute to both components with different learned proportions. This allows the decomposition itself to be optimized jointly with the forecasting objective, providing a more flexible way to disentangle heterogeneous traffic dynamics without imposing a hard frequency partition.

## 3 Methodology

## 3.1 Model Overview

Figure 1 illustrates the overall architecture of the proposed Adaptive Decomposition Network (ADNet). Given historical traffic observations from multiple locations, ADNet first transforms the input sequences into the frequency domain using the real-valued fast Fourier transform (rFFT). A learnable complementary spectral decomposition module then adaptively distributes the information in each frequency bin between a dominant component and a residual component. Unlike hard frequency partitioning, every frequency bin can contribute to both components with different learned proportions, allowing the decomposition to adapt to the forecasting objective.

The two decomposed spectra are subsequently transformed back into the time domain through inverse rFFT (irFFT) and modeled by two independently parameterized Graph WaveNet (GWN) branches [3]. The two branches capture the spatiotemporal dependencies of the dominant and residual dynamics separately and produce their respective multi-step forecasts. Finally, the two predictions are combined through elementwise addition to obtain the final traffic-flow prediction. In this way, ADNet integrates adaptive spectral decomposition with component-specific spatiotemporal forecasting to explicitly model heterogeneous traffic dynamics.

## 3.2 Learnable Complementary Spectral Decomposition

Given historical traffic observations $\mathbf { X } \in \mathbb { R } ^ { N \times T _ { h } \times C }$ , where N denotes the number of locations, $T _ { h }$ the historical window length, and C the number of input features, we first transform the traffic sequences into the frequency domain using the real Fast Fourier Transform (rFFT) along the temporal dimension:

$$
\begin{array} { r } { \widetilde { \mathbf { X } } = \mathrm { r F F T } ( { \mathbf { X } } ) , \qquad \widetilde { \mathbf { X } } \in \mathbb { C } ^ { N \times F \times C } , } \end{array}\tag{1}
$$

![](images/b069012ce2b4c002f860c09230a773ddb09a8ecb299ca7252822b1b9b5676861.jpg)  
Fig. 1. Overview of Adaptive Decomposition Network (ADNet).

where $F = \lfloor T _ { h } / 2 \rfloor + 1$ is the number of non-redundant frequency bins. The transformation is independently applied to each location and input channel.

Instead of using a fixed frequency cutoff to separate different traffic dynamics, we introduce a learnable soft mask m $= [ m _ { 0 } , \ldots , m _ { F - 1 } ]$ to determine the contribution of each frequency bin to the dominant component. For the k-th frequency bin, its weight is defined as

$$
m _ { k } = \sigma ( a _ { k } ) , m _ { k } \in ( 0 , 1 ) ,\tag{2}
$$

where $a _ { k }$ is a learnable parameter and $\sigma ( \cdot )$ denotes the sigmoid function. The mask is shared across samples, locations, and input channels and is optimized jointly with the forecasting model.

To provide an inductive bias at the beginning of training, we initialize the mask such that the first K frequency bins receive larger dominant weights:

$$
a _ { k } ^ { ( 0 ) } = \left\{ { \alpha } , \quad k < K , \right.\tag{3}
$$

where $K$ is a hyperparameter satisfying $1 \leq K \leq F$ , and $\alpha > 0$ controls the strength of the initialization bias. Importantly, K only determines the initialization and does not impose a hard frequency boundary. All mask parameters remain trainable and may adapt freely during optimization.

The dominant and residual spectra are constructed using complementary weights:

$$
\begin{array} { r } { \widetilde { \mathbf { X } } _ { \mathrm { D } } = \mathbf { m } \odot \widetilde { \mathbf { X } } , } \end{array}\tag{4}
$$

$$
\begin{array} { r } { \widetilde { \mathbf { X } } _ { \mathrm { R } } = ( \mathbf { 1 } - \mathbf { m } ) \odot \widetilde { \mathbf { X } } , } \end{array}\tag{5}
$$

where ⊙ denotes element-wise multiplication along the frequency dimension. Therefore, each frequency bin can contribute to both components with different proportions rather than being assigned exclusively to one of them.

Finally, the two spectra are transformed back into the time domain through irFFT:

$$
\begin{array} { r } { \mathbf { X } _ { \mathrm { D } } = \mathrm { i r F F T } ( \widetilde { \mathbf { X } } _ { \mathrm { D } } ) , } \end{array}\tag{6}
$$

$$
\begin{array} { r } { { \bf X } _ { \mathrm { R } } = \mathrm { i r F F T } ( \widetilde { \bf X } _ { \mathrm { R } } ) . } \end{array}\tag{7}
$$

Because the two spectral masks are complementary, the reconstructed sequences satisfy ${ \bf X } _ { \mathrm { D } } + { \bf X } _ { \mathrm { R } } = { \bf X }$ up to numerical precision. The proposed decomposition therefore preserves the complete historical signal while allowing the model to learn different spectral emphasis for the dominant and residual traffic dynamics.

The two reconstructed components are subsequently processed by two parallel and independently parameterized Graph WaveNet (GWN) branches [3]. The dominant branch learns the spatiotemporal dependencies associated with relatively persistent and regular traffic dynamics, whereas the residual branch focuses on the spatial and temporal correlations underlying irregular and rapidly varying traffic fluctuations. Although the two branches adopt the same GWN backbone, their parameters are learned independently, enabling them to specialize in the distinct characteristics of the two components. Each branch directly generates multi-step predictions in the time domain, resulting in a dominant-component forecast and a residual-component forecast.

## 3.3 Component-Specific Spatiotemporal Forecasting

After spectral decomposition, the reconstructed dominant and residual sequences exhibit different temporal characteristics and are therefore modeled separately. We employ two independently parameterized spatiotemporal forecasting branches, one for each component, allowing the predictors to specialize in the distinct dynamics contained in the dominant and residual signals.

In this work, Graph WaveNet (GWN) [3] is adopted as the forecasting backbone because of its effectiveness in jointly capturing temporal dependencies and spatial correlations among traffic locations. The dominant and residual sequences are independently fed into two GWN branches to produce their corresponding multi-step forecasts. Although the two branches share the same network architecture, their parameters are not shared, enabling each branch to learn component-specific spatiotemporal representations.

In this work, GWN is used as the forecasting backbone for both branches. While the proposed component-specific modeling strategy is not inherently dependent on GWN, evaluating alternative spatiotemporal backbones is beyond the scope of this study.

## 3.4 Prediction Fusion and Learning Objective

The dominant and residual branches independently generate multi-step forecasts, denoted by $\widehat { \mathbf { Y } } _ { \mathrm { D } }$ and $\widehat { \mathbf Y } _ { \mathrm { R } }$ , respectively. We combine the two predictions through elementwise addition:

$$
\widehat { \mathbf { Y } } = \widehat { \mathbf { Y } } _ { \mathrm { D } } + \widehat { \mathbf { Y } } _ { \mathrm { R } } ,\tag{8}
$$

where $\widehat { \mathbf Y }$ denotes the final multi-step traffic forecast. The additive fusion introduces no additional parameters and naturally combines the complementary information learned from the two components.

The entire framework is trained end-to-end using the mean absolute error (MAE) between the predicted and ground-truth traffic values:

$$
\mathcal { L } = \frac { 1 } { \boldsymbol { B } \times \boldsymbol { N } \times \boldsymbol { T _ { p } } } \sum _ { b = 1 } ^ { B } \sum _ { n = 1 } ^ { N } \sum _ { h = 1 } ^ { T _ { p } } \left| \widehat { Y } _ { b , n , h } - Y _ { b , n , h } \right| ,\tag{9}
$$

where B, N, and $T _ { p }$ denote the batch size, number of traffic locations, and prediction horizon, respectively. The forecasting loss jointly optimizes the learnable spectral decomposition module and the two GWN forecasting branches.

## 4 Evaluation

## 4.1 Datasets

We conduct experiments on the TraffiDent benchmark [11], which provides large-scale traffic data collected across California in 2023. To focus on freeway mainline traffic dynamics, we retain only mainline sensors and aggregate the traffic measurements into 5-minute intervals. We construct two regional datasets corresponding to Alameda and Orange counties. Their statistics are summarized in Table 1.

Each region’s time series is chronologically divided into training, validation, and test periods in a 70%/15%/15% ratio. Forecasting windows are sampled around incidents, deduplicated by forecast origin, and filtered to prevent prediction targets or incidents from being shared across splits.

Table 1. Spatial statistics and temporal coverage of the Alameda and Orange datasets.
<table><tr><td>Region</td><td>Nodes Edges Time range</td></tr><tr><td>Alameda</td><td>521 13,828 2023-01-01 to 2023-12-31</td></tr><tr><td>Orange</td><td>990 29,142 2023-01-01 to 2023-12-31</td></tr></table>

## 4.2 Baselines

We compare ADNet with ten representative traffic forecasting methods. Historical Last (HL) and LSTM are included as non-graph temporal baselines. Moreover, DCRNN [1], STGCN [2], and Graph WaveNet (GWN) [3] are representative spatiotemporal graph forecasting models, with GWN serving as the direct backbone baseline of ADNet. We further compare with AGCRN [12], which learns adaptive spatial dependencies, AST-GCN [13], which incorporates spatial–temporal attention, and more recent graph-based approaches including DSTAGNN [14] and $\mathrm { D ^ { 2 } S T G N N }$ [4]. STWave [7] uses wavelet decomposition to separate traffic signals into stable trends and fluctuating events, which are modeled by a dual-channel spatiotemporal network with efficient spectral graph attention. This provides a decomposition-based comparison for ADNet. Together, these baselines cover temporal, graph-based, attention-based, adaptive, and dynamic spatiotemporal forecasting strategies.

## 4.3 Implementation Details

All methods use the same data splits and forecasting protocol. The input window contains 12 historical time steps, and the model predicts the next 12 time steps; at 5-minute intervals, these correspond to one hour of history and one hour of future traffic flow. For the baseline methods, we follow their recommended model and training configurations whenever applicable. In the main comparison, all trainable methods are evaluated using seed 42 and trained for at most 30 epochs, with model selection based on validation performance.

For ADNet, each forecasting branch adopts GWN with 32 hidden channels and a dropout rate of 0.3. The model is optimized using Adam with an initial learning rate of $1 0 ^ { \dot { - 3 } }$ and a weight decay of $1 0 ^ { - 4 }$ . For the adaptive decomposition module, the complementary spectral weights are initialized using a boundary of K = 3 and an initialization strength of $\alpha = 2$ , unless otherwise specified. These parameters affect only the initialization of the spectral weights; all frequency weights are subsequently optimized jointly with the forecasting model.

## 4.4 Evaluation Metrics

We evaluate forecasting accuracy using three widely adopted metrics: mean absolute error (MAE), root mean squared error (RMSE), and mean absolute percentage error (MAPE), where lower values indicate better performance. All metrics are computed on the original traffic-flow scale. We report performance at H3, H6, and H12, corresponding to 15-, 30-, and 60-minute-ahead forecasting, respectively. We additionally report Average, which aggregates predictions over all 12 forecasting horizons.

Table 2. Performance comparison on the Alameda and Orange regions of TraffiDent. H3, H6, and H12 denote the 15-, 30-, and 60-minute forecasting horizons, respectively, while Average aggregates all 12 horizons.
<table><tr><td rowspan="2">Method</td><td colspan="3">H3</td><td colspan="2">H6</td><td colspan="2">H12</td><td colspan="2"></td><td colspan="2">Average</td></tr><tr><td colspan="3">MAE RMSE MAPE (%)</td><td colspan="2"></td><td colspan="2">MAE RMSE MAPE (%)</td><td colspan="2"></td><td colspan="2">MAE RMSE MAPE (%)</td></tr><tr><td>Alameda County</td><td colspan="3"></td><td colspan="2">MAE RMSE MAPE (%)</td><td colspan="2"></td><td colspan="3"></td></tr><tr><td>HL</td><td>14.57</td><td>25.23</td><td>20.87 17.67</td><td>30.23</td><td></td><td>24.5724.18</td><td>40.62</td><td>33.0018.26</td><td>31.76</td><td>25.46</td></tr><tr><td>LSTM</td><td>11.48</td><td>20.82</td><td>16.43 12.86</td><td>23.47</td><td>18.21</td><td>15.29</td><td>27.79</td><td>21.51 12.96</td><td>23.76</td><td>18.44</td></tr><tr><td>DCRNN</td><td>11.37</td><td>20.43</td><td>16.67 12.60</td><td>22.74</td><td>18.61</td><td>14.77</td><td>26.23</td><td>22.04 12.68</td><td>22.84</td><td>18.82</td></tr><tr><td>AGCRN</td><td>11.65</td><td>22.64</td><td>31.96 12.62</td><td>24.96</td><td>35.40</td><td>13.98</td><td>28.00</td><td>39.05 12.57</td><td>24.83</td><td>34.47</td></tr><tr><td>STGCN</td><td>12.58</td><td>22.87</td><td>32.21 13.32</td><td>24.37</td><td>32.80</td><td>14.72</td><td>26.88</td><td>34.30 13.41</td><td>24.53</td><td>33.01</td></tr><tr><td>GWN</td><td>11.28</td><td>20.33</td><td>16.50 12.53</td><td>22.68</td><td>18.14</td><td>14.69</td><td>26.59</td><td>21.15</td><td>12.62 22.99</td><td>18.34</td></tr><tr><td>ASTGCN</td><td>12.15</td><td>21.12</td><td>19.96 13.57</td><td>23.58</td><td>22.57</td><td>15.79</td><td>27.34</td><td>27.14 13.55</td><td>23.72</td><td>22.54</td></tr><tr><td>DSTAGNN</td><td>11.70</td><td>20.67</td><td>18.72 12.45</td><td>22.39</td><td>19.03</td><td>14.19</td><td>25.65</td><td>21.41 12.55</td><td>22.64</td><td>19.25</td></tr><tr><td>D2STGNN</td><td>11.22</td><td>19.99</td><td>18.12 12.92</td><td>22.71</td><td>19.77</td><td>14.99</td><td>26.27</td><td>22.32 13.02</td><td>23.01</td><td>19.85</td></tr><tr><td>STWave</td><td>11.03</td><td>20.29</td><td>15.92 12.22</td><td>22.72</td><td>17.49</td><td>14.31</td><td></td><td>20.58</td><td>22.98</td><td></td></tr><tr><td>ADNet (ours)</td><td>11.01</td><td>20.00</td><td>16.29 12.10</td><td>22.11</td><td>17.78</td><td>13.86</td><td>26.55 25.48</td><td>12.32 20.41 12.13</td><td>22.32</td><td>17.82 17.91</td></tr><tr><td colspan="9">Orange County</td><td></td></tr><tr><td>HL</td><td>15.41</td><td>26.21</td><td>20.29</td><td>18.47 31.08</td><td></td><td>23.42|24.89</td><td>40.99</td><td>30.99</td><td>19.03</td><td>32.38</td><td>24.22</td></tr><tr><td>LSTM</td><td>12.25</td><td>21.85</td><td>16.00</td><td>13.72 24.59</td><td></td><td>17.57 16.32</td><td>29.16</td><td>20.48</td><td>13.81</td><td>24.86</td><td>17.76</td></tr><tr><td>DCRNN</td><td>14.56</td><td>23.66</td><td>26.27</td><td>17.85</td><td>28.55</td><td>30.72 25.49</td><td>39.41</td><td>40.16</td><td>18.63</td><td>30.22</td><td>31.30</td></tr><tr><td>AGCRN</td><td>12.82</td><td>26.23</td><td>37.02</td><td>13.74</td><td>28.54</td><td>40.00 14.66</td><td>28.95</td><td>38.42</td><td>13.64</td><td>27.97</td><td>38.80</td></tr><tr><td>STGCN</td><td>13.36</td><td>25.19</td><td>31.00</td><td>14.06</td><td>26.46</td><td>31.27 15.24</td><td>28.40</td><td>32.15</td><td>14.07</td><td>26.48</td><td>31.41</td></tr><tr><td>GWN</td><td>11.75</td><td>20.81</td><td>15.74</td><td>12.89</td><td>22.85</td><td>16.82 14.77</td><td>26.06</td><td>18.97</td><td>12.92</td><td>22.95</td><td>16.95</td></tr><tr><td>ASTGCN</td><td>13.09</td><td>22.50</td><td>18.68</td><td>14.75</td><td>25.50</td><td>20.66 17.97</td><td>30.97</td><td>23.25</td><td>14.91</td><td>25.96</td><td>20.17</td></tr><tr><td>DSTAGNN</td><td>12.13</td><td>21.12</td><td>17.76</td><td>13.27</td><td>23.13</td><td>20.42</td><td>15.31 26.66</td><td></td><td>26.04 13.36</td><td>23.32</td><td>20.81</td></tr><tr><td>D2STGNN</td><td>12.23</td><td>21.15</td><td>21.47</td><td>13.73</td><td>23.42</td><td>23.76</td><td>16.05 26.98</td><td></td><td>26.46 13.79</td><td>23.54</td><td>24.50</td></tr><tr><td>STWave</td><td>11.88</td><td>21.39</td><td>15.88</td><td>13.19</td><td>23.99</td><td>17.07</td><td>15.46 28.14</td><td></td><td>19.57 13.24</td><td>24.17</td><td>17.14</td></tr><tr><td>ADNet (ours)</td><td>11.53</td><td>20.56</td><td>15.40</td><td>12.55</td><td>22.44</td><td>16.38</td><td>14.20 25.38</td><td>18.27</td><td>12.56</td><td>22.51</td><td>16.47</td></tr></table>

Missing target observations are excluded using the original observation mask. For MAPE, target values whose absolute flow is close to zero are additionally excluded to avoid numerical instability.

## 4.5 Overall Performance

Table 2 reports the forecasting performance of ADNet and the compared methods on the Alameda and Orange regions of TraffiDent. We report MAE, RMSE, and MAPE, where lower values indicate better forecasting performance. The best and second-best results in each setting are highlighted in bold and underlined, respectively.

Overall, ADNet achieves consistently strong performance across both regions, forecasting horizons, and evaluation metrics. Among the 24 region–horizon–metric combinations reported in Table 2, ADNet obtains the best result in 20 cases and the secondbest result in the remaining four cases. This broad improvement across MAE, RMSE, and MAPE indicates that the advantage of ADNet is not restricted to a particular forecasting horizon or error metric.

On Alameda, ADNet achieves the best Average MAE and RMSE of 12.13 and 22.32, respectively. ADNet also achieves the best MAE and RMSE at H6 and the best performance across all three metrics at H12. At H3, it obtains the lowest MAE, while its RMSE of 20.00 is nearly identical to the best result of 19.99 achieved by D<sup>2</sup>STGNN.

The advantage of ADNet is particularly consistent on Orange, where it achieves the best result for every metric at every reported horizon. For the Average performance, ADNet reduces MAE, RMSE, and MAPE by 2.79%, 1.92%, and 2.83%, respectively, compared with the strongest baseline results. Moreover, its advantage becomes more pronounced at longer forecasting horizons. For example, relative to GWN, the MAPE reduction increases from 2.16% at H3 to 2.62% at H6 and 3.69% at H12. Similar improvements are observed for MAE and RMSE at the longer horizons.

Table 3. Ablation study on Alameda.
<table><tr><td>Model</td><td> $\mathbf { M A E } \pm \mathbf { S D }$ </td><td> $\mathrm { R M S E } \pm \mathrm { S D }$ </td><td> $\mathbf { M A P E } \mathbf { \mathit { q } } _ { 0 } \pm \mathbf { S D }$ </td></tr><tr><td>Alameda</td><td></td><td></td><td></td></tr><tr><td>GWN</td><td> $1 1 . 0 6 2 0 \pm 0 . 1 0 5 9$ </td><td> $2 0 . 9 6 3 3 \pm 0 . 1 5 8 2$ </td><td> $1 9 . 4 5 6 6 \pm 0 . 1 9 8 6$ </td></tr><tr><td>DualGWN</td><td> $1 0 . 7 3 4 6 \pm 0 . 0 5 1 8$ </td><td> $2 0 . 4 7 7 7 \pm 0 . 1 3 1 9$ </td><td> $\mathbf { 1 8 . 7 4 3 1 \pm 0 . 1 4 7 7 }$ </td></tr><tr><td>ADNet fixed-K3</td><td> $1 1 . 1 0 3 3 \pm 0 . 1 1 0 1$ </td><td> $2 0 . 9 2 4 1 \pm 0 . 1 1 3 8$ </td><td> $1 9 . 4 9 8 9 \pm 0 . 0 6 1 5$ </td></tr><tr><td>ADNet learnable-K3</td><td> ${ \bf 1 0 . 7 1 4 6 \pm 0 . 0 7 7 7 }$ </td><td> $\mathbf { 2 0 . 3 9 9 5 \pm 0 . 1 0 4 2 }$ </td><td> $1 8 . 8 0 7 0 \pm 0 . 1 5 1 0$ </td></tr></table>

STWave provides a comparison with predefined wavelet decomposition. ADNet reduces Average MAE relative to STWave by 1.55% on Alameda and 5.10% on Orange, and reduces Average RMSE by 2.85% and 6.88%, respectively. The MAPE comparison is more mixed: on Alameda, STWave achieves lower MAPE at H3, H6, and Average, with values of 15.92%, 17.49%, and 17.82%, compared with 16.29%, 17.78%, and 17.91% for ADNet. ADNet achieves lower MAPE at H12 on Alameda and at all reported horizons on Orange.

These results demonstrate the effectiveness of ADNet for multi-step traffic forecasting. In particular, the consistent gains at longer forecasting horizons are aligned with the motivation of adaptive decomposition: disentangling heterogeneous traffic dynamics allows components with different temporal characteristics to be modeled separately before their predictions are integrated. We further examine the contributions of the dualbranch architecture and the adaptive decomposition mechanism through ablation experiments.

## 4.6 Ablation Study

We conduct an ablation study to investigate two questions: whether the improvement of ADNet can be explained by the additional capacity of the dual-branch architecture, and whether learning the decomposition provides an advantage over a fixed spectral partition. All variants are trained for 100 epochs using three random seeds, with the checkpoint selected according to validation MAE. Table 3 reports the mean and standard deviation of the Average performance over all 12 forecasting horizons.

We first evaluate the effect of the dual-branch architecture. Compared with the single-branch GWN, DualGWN reduces MAE, RMSE, and MAPE by 2.96%, 2.32%, and 3.67%, respectively. This result shows that using two independently parameterized forecasting branches already provides a clear performance benefit and therefore serves as an important capacity-matched control for evaluating the decomposition mechanism.

We next examine whether the decomposition strategy itself matters. ADNet with a fixed decomposition does not improve over GWN and produces slightly higher MAE and MAPE. In contrast, learning the decomposition substantially improves performance. Compared with the fixed variant, ADNet with learnable decomposition reduces MAE,

RMSE, and MAPE by 3.50%, 2.51%, and 3.55%, respectively. This result indicates that simply separating the signal into components is insufficient; the decomposition needs to adapt to the forecasting objective.

Finally, compared with the capacity-matched DualGWN, the learnable ADNet variant further reduces MAE from 10.7346 to 10.7146 and RMSE from 20.4777 to 20.3995, while achieving a comparable MAPE (18.8070 versus 18.7431). These results suggest that the performance gain of ADNet cannot be attributed solely to increased model capacity. Rather, the learnable decomposition provides additional benefit by adaptively organizing heterogeneous traffic dynamics before component-specific forecasting.

## 5 Conclusion

In this work, we proposed the Adaptive Decomposition Network (ADNet) for multistep traffic forecasting. ADNet addresses the challenge of entangled heterogeneous traffic dynamics by adaptively decomposing historical observations into dominant and residual components and modeling them through separate forecasting branches. A learnable complementary spectral decomposition mechanism allows each frequency bin to contribute to both components with different learned proportions, avoiding the restriction of a hard frequency partition.

Experiments on the Alameda and Orange regions of TraffiDent demonstrate that ADNet consistently achieves strong performance across different forecasting horizons and evaluation metrics, with particularly clear improvements at longer horizons. The capacity-controlled ablation study further shows that, although the dual-branch architecture itself provides performance gains, learning the decomposition substantially outperforms a fixed decomposition and provides additional improvements beyond the capacitymatched control. These results demonstrate the effectiveness of combining adaptive decomposition with component-specific forecasting for modeling heterogeneous traffic dynamics.

## References

1. Li, Y., Yu, R., Shahabi, C., Liu, Y. Diffusion convolutional recurrent neural network: Datadriven traffic forecasting. International Conference on Learning Representations (2018)

2. Yu, B., Yin, H., Zhu, Z. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence, pp. 3634–3640 (2018).

3. Wu, Z., Pan, S., Long, G., Jiang, J., Zhang, C. Graph WaveNet for deep spatial-temporal graph modeling. Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, pp. 1907–1913 (2019).

4. Shao, Z., Zhang, Z., Wei, W., Wang, F., Xu, Y., Cao, X., Jensen, C.S. Decoupled dynamic spatial-temporal graph neural network for traffic forecasting. Proceedings of the VLDB Endowment 15(11), 2733–2746 (2022).

5. Cao, D., Wang, Y., Duan, J., Zhang, C., Zhu, X., Huang, C., Tong, Y., Xu, B., Bai, J., Tong, J., Zhang, Q. Spectral temporal graph neural network for multivariate time-series forecasting. Advances in Neural Information Processing Systems, vol. 33, pp. 17766–17778. (2020)

6. Yi, K., Zhang, Q., Fan, W., Wang, S., Wang, P., He, H., An, N., Lian, D., Cao, L., Niu, Z. Frequency-domain MLPs are more effective learners in time series forecasting. Advances in Neural Information Processing Systems, vol. 36, pp. 76656–76679. (2023).

7. Fang, Y., Qin, Y., Luo, H., Zhao, F., Xu, B., Zeng, L., Wang, C. When spatio-temporal meet wavelets: Disentangled traffic forecasting via efficient spectral graph attention networks. In: 2023 IEEE 39th International Conference on Data Engineering, pp. 517–529 (2023). https: //doi.org/10.1109/ICDE55515.2023.00046

8. Li, Y., Shao, Z., Xu, Y., Qiu, Q., Cao, Z., Wang, F. Dynamic frequency domain graph convolutional network for traffic forecasting. ICASSP 2024–2024 IEEE International Conference on Acoustics, Speech and Signal Processing, pp. 5245–5249.

9. Feng, Q., Li, B., Liu, X., Gao, X., Wan, K. Low-high frequency network for spatial-temporal traffic flow forecasting. Engineering Applications of Artificial Intelligence 158, 111304 (2025).

10. Shao, M., Zhang, Z., Wang, Y., Dai, Y., Shen, X., Wang, X. HyperD: Hybrid periodicity decoupling framework for traffic forecasting. Proceedings of the AAAI Conference on Artificial Intelligence 40(18), 15689–15697 (2026).

11. Gou, X., Li, Z., Lan, T., Lin, J., Li, Z., Zhao, B., Zhang, C., ... Zhang, X. (2025). TraffiDent: A dataset for understanding the interplay between traffic dynamics and incidents. Advances in Neural Information Processing Systems, 38.

12. Bai, L., Yao, L., Li, C., Wang, X., Wang, C. Adaptive graph convolutional recurrent network for traffic forecasting. In: Advances in Neural Information Processing Systems, vol. 33, pp. 17804–17815 (2020).

13. Guo, S., Lin, Y., Feng, N., Song, C., Wan, H. Attention based spatial-temporal graph convolutional networks for traffic flow forecasting. Proceedings of the AAAI Conference on Artificial Intelligence 33(1), 922–929 (2019).

14. Lan, S., Ma, Y., Huang, W., Wang, W., Yang, H., Li, P. DSTAGNN: Dynamic spatialtemporal aware graph neural network for traffic flow forecasting. Proceedings of the 39th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 162, pp. 11906–11917 (2022).

15. Cleveland, R.B., Cleveland, W.S., McRae, J.E., Terpenning, I. STL: A seasonal-trend decomposition procedure based on Loess. Journal of Official Statistics 6(1), 3–73 (1990)

16. Wu, H., Xu, J., Wang, J., Long, M. Autoformer: Decomposition transformers with autocorrelation for long-term series forecasting. Advances in Neural Information Processing Systems, vol. 34, pp. 22419–22430 (2021).

17. Zhou, T., Ma, Z., Wen, Q., Wang, X., Sun, L., Jin, R. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. Proceedings of the 39th International Conference on Machine Learning, Proceedings of Machine Learning Research, vol. 162, pp. 27268–27286 (2022).

18. Zeng, A., Chen, M., Zhang, L., Xu, Q. Are transformers effective for time series forecasting? Proceedings of the AAAI Conference on Artificial Intelligence 37(9), 11121–11128 (2023).

19. Zhong, S., Song, S., Zhuo, W., Li, G., Liu, Y., Chan, S.-H.G. A multi-scale decomposition MLP-mixer for time series analysis. Proceedings of the VLDB Endowment 17(7), 1723–1736 (2024).

20. Liu, Z. (2025). Freqmoe: Enhancing time series forecasting through frequency decomposition mixture of experts. arXiv preprint arXiv:2501.15125.