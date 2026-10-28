# From Ethernet to Wi-Fi: Flow Feature Selection and Construction for Traffic Classification

Francisco Rau<sup>∗†</sup>, Petia Georgieva<sup>‡</sup>, Carlos Herranz<sup>∗</sup>, Iñaki Val<sup>∗</sup> and Joaquin Perez<sup>†</sup> <sup>∗</sup>MaxLinear, Inc., Valencia, Spain

<sup>†</sup>Electronic Engineering Department, Universitat de València, Valencia, Spain

<sup>‡</sup>Institute of Electronics and Informatics Eng. of Aveiro (IEETA), University of Aveiro, Portugal

Corresponding author: frau@maxlinear.com

Abstract—Flow-based traffic classifiers trained on Ethernet data may lose accuracy when applied to Wi-Fi traffic. This paper proposes a five-stage procedure for selecting and constructing features that enable Wi-Fi flow classification using models trained only on Ethernet data. The procedure removes the transport protocol identifier, examines rate dependence, compares feature distributions and the information each feature provides for the application classes, reduces redundancy, and constructs normalized packet-size and temporal features. Ethernet and Wi-Fi traffic are simulated in ns-3, with Wi-Fi using Best Effort access category. Labelled Wi-Fi data from six development scenarios are used for feature design, while a separate scenario is reserved for final testing. The evaluation compares feature sets, examines threshold sensitivity, and analyses classification errors under different traffic conditions.

Index Terms—Traffic classification, feature selection, Ethernet, Wi-Fi, ns-3.

## I. INTRODUCTION

The classification of downlink traffic at the Wi-Fi access point (AP) is a critical enabler for delivering high Quality of Service (QoS) in modern wireless networks. Ethernet data is often the preferred choice for building traffic classifiers because it provides a controlled and stable environment.

In our previous work we developed a flow-based traffic classifier to identify applications from packet sizes, packet counts, and inter-arrival times in Transmission Control Protocol (TCP) and User Datagram Protocol (UDP) flows, without inspecting packet payloads [1]. However, classifiers trained on Ethernet traffic may lose accuracy when applied to Wi-Fi. Wireless contention and network load can change packet timing and feature distributions, making classification sensitive to differences between training and deployment conditions [2]. Therefore, selecting features that remain informative across both media is important. Since collecting labelled traffic under different congested Wi-Fi conditions in real networks is costly, we use the ns-3 network simulator to study this problem under controlled conditions.

Previous studies have addressed traffic classification using different feature selection and representation methods. Moore and Zuev [3] combine a Fast Correlation-Based Filter with Naive Bayes classification. FlowPic [4] represents flows as twodimensional histograms of packet size and arrival time and uses a convolutional neural network. Other studies consider changes in traffic distributions. Rosetta [2] combines TCP-aware augmentation with self-supervised learning for Transport Layer Security (TLS) traffic classification and evaluates Ethernet and Wi-Fi. FG-SAT [5] uses Jensen–Shannon divergence to select attributes that preserve class differences across environments and learns graph-based flow representations. Yuan et al. [6] use mutual information and Kullback–Leibler divergence to evaluate feature information and stability over time for Tor hidden-service fingerprinting. Their work focuses on temporal changes rather than differences between network media.

In this paper, we propose a five-stage procedure to select and construct features for Wi-Fi traffic classification using models trained only on Ethernet data. The procedure examines rate dependence, distribution differences, class information, and feature redundancy before constructing normalized packet-size and temporal features. The initial 29 dimensional feature space is reduced to seven input features representative across both media. Labelled Wi-Fi data from six development scenarios (S1– S6) are used for feature design, based on a classifier trained only with Ethernet data. A separate Wi-Fi scenario (S7), containing all traffic classes is reserved for final evaluation.

## II. METHODOLOGY

## A. Data Processing

Flows are identified by their Internet Protocol (IP) addresses, transport ports, and protocol. For TCP, flow statistics include downlink packets and their corresponding reverse-direction TCP acknowledgments. For UDP, only downlink packets are used. Features are computed from packet metadata in fivesecond windows advanced every second. Packets without IP payload are discarded, while pure TCP acknowledgments are retained because they contain TCP headers even without application data.

The AP-side classifier targets only downlink application flows forwarded by the AP of the home basic service set (BSS) 1. Uplink application traffic is excluded. Observed duration is the interval between the first and last packets in a window. The eight classes, common to Ethernet and Wi-Fi, include Voice over Internet Protocol (VoIP) and video streaming using Dynamic Adaptive Streaming over HTTP (DASH). Table I lists the classes, Ethernet training rates, and dataset sizes.

## B. Feature Selection and Construction

The first four stages apply fixed rules to successively reduce the original 29 features obtained in [1] to 28, 16, 12, and 9 features. The fifth stage constructs normalized size and temporal features, ending up with 7 final features.

1) Transport Exclusion: Removing the transport identifier prevents direct use of TCP or UDP identity, leaving 28 candidates.

COMMON APPLICATION CLASSES AND DATASET WINDOWS.  
TABLE I
<table><tr><td>ID</td><td></td><td>Rate for training</td><td colspan="2">Ethernet</td><td colspan="2">Wi-Fi</td></tr><tr><td></td><td>Application (classes)</td><td>(Mb/s)</td><td>Train</td><td>Test</td><td>S1-S6</td><td>S7</td></tr><tr><td>VP</td><td>VoIP</td><td>0.10</td><td>216</td><td>69</td><td>1,620</td><td>270</td></tr><tr><td>VC</td><td>Video Conference</td><td>3</td><td>216</td><td>69</td><td>2,160</td><td>270</td></tr><tr><td></td><td>GM Gaming (Quake 3)</td><td>0.11</td><td>216</td><td>69</td><td>1,890</td><td>270</td></tr><tr><td>VS</td><td>Video Štreaming (DASH)</td><td>20</td><td>216</td><td>69</td><td>1,876</td><td>540</td></tr><tr><td>BT</td><td>Bulk TCP</td><td>10</td><td>216</td><td>69</td><td>742</td><td>270</td></tr><tr><td>IT</td><td>IPERF TCP</td><td>5-200*</td><td>216</td><td>69</td><td>270</td><td>270</td></tr><tr><td>IU</td><td>IPERF UDP</td><td>10</td><td>216</td><td>69</td><td>1,080</td><td>270</td></tr><tr><td>BU</td><td>Bursty UDP</td><td>10</td><td>216</td><td>69</td><td>1,080</td><td>270</td></tr><tr><td>Total</td><td></td><td></td><td>1,728</td><td></td><td>552 10,718 2,430</td><td></td></tr></table>

IDs are reused in Table IV; the rates shown are for Ethernet training. <sup>\*</sup>5, 25, 100, and 200 Mb/s, 54 windows each. Wi-Fi counts include five seeds.

2) Rate Dependence: For each feature X, the absolute Pearson correlation is computed between X and the traffic rate within each of the four variable-rate applications: Bursty UDP, IPERF UDP, Video Conference, and IPERF TCP (Section III-A). The rate-dependence score is the largest of the four absolute correlations, $\mathbf { \bar { \psi } } _ { s } ( X ) = \operatorname* { m a x } _ { a } \left| r _ { a } ( X ) \right|$ . A score near one indicates strong linear dependence on rate in at least one application, rather than discrimination between applications. Undefined correlations from constant features are ignored, and features without a valid score are retained. The other four classes have no controlled rate variation in this analysis.

Twelve features are removed with $s ( X ) \geq 0 . 9 7 \colon$ forward, reverse, and total packet and byte counts; TCP flag count; rate descriptors; and related traffic-intensity and mean-volume descriptors. Using measured throughput or configured rate gives the same twelve exclusions, leaving 16 features.

3) Distribution Shift and Class Information: For feature X and application label ${ \mathrm { ~ \bar { \cal { Y } } ~ , ~ } }$ the two-sample Kolmogorov–Smirnov (KS) distance [7] is

$$
D = \operatorname* { s u p } _ { x } \left| \widehat { F } _ { \mathrm { e t h } } ( x ) - \widehat { F } _ { \mathrm { w i f i } } ( x ) \right| ,\tag{1}
$$

where $\widehat { F } _ { \mathrm { e t h } }$ and $\widehat { F } _ { \mathrm { w i f i } }$ are empirical cumulative distribution functions. The distance ranges from 0 to 1 and does not use labels. Mutual information (MI) measures how much information X contains about Y [8]. We estimate MI separately for each medium with a nearest-neighbor estimator [9], [10]. Parameter $k = 3$ specifies the three nearest samples used to estimate local probability relationships:

$$
\mathrm { M I \ r a t i o } = { \frac { \widehat { M I } _ { \mathrm { w i f i } } ( X ; Y ) } { \widehat { M I } _ { \mathrm { e t h } } ( X ; Y ) } } , \qquad \widehat { M I } _ { \mathrm { e t h } } ( X ; Y ) > 0 .\tag{2}
$$

A ratio below 1 indicates lower estimated MI on Wi-Fi than on Ethernet, while a ratio above 1 indicates higher MI. Both metrics use 1,728 Ethernet training windows, balanced across eight classes, and 10,718 Wi-Fi windows from S1–S6. These Wi-Fi scenarios deliberately vary application mixes, offered loads, and co-channel contention to represent different congestion conditions. This diversity aims to identify features that remain informative across a broader range of Wi-Fi operating conditions. Thus, KS distance and MI ratio reflect differences in both traffic composition and network conditions, rather than the isolated effect of the transmission medium. We retain features with MI ratio $\ge 0 . 6 0$ and either $D \leq 0 . 3 0$ or MI ratio ≥ 1.00 (Fig. 1). This removes avg\_min\_pkt\_size, mean\_iat, std\_iat, and directions, leaving 12. Here, avg\_min\_pkt\_size is an exponentially weighted moving average of minimum size across consecutive windows of a flow; directions is the forward-packet fraction.

![](images/5d76336e4c4f22100f4959ac4ba5aeec7a0597d8238889b938425c64e663c376.jpg)  
Fig. 1. Distribution shift and class information for 16 features. Blue: retained; red: removed. Dashed lines show the KS threshold (0.30) and MI-ratio thresholds (0.60 and 1.00). Shaded areas show removal regions.

4) Redundancy Reduction: Candidates are considered in increasing D. A feature is removed if its absolute Pearson correlation with a retained feature exceeds 0.98 in Ethernet training. This retains max\_pkt\_size over avg\_max\_pkt\_size, rms\_pkt\_size over mean\_pkt\_size, and max\_iat over iat\_variance\_ratio, leaving nine. The retained features have lower distribution shift. Here, avg\_max\_pkt\_size averages maximum packet size as defined for avg\_min\_pkt\_size; iat\_variance\_ratio divides inter-arrivaltime (IAT) variance by mean IAT.

5) Feature Construction: We retain rms\_pkt\_size, max\_pkt\_size, and switching\_ratio and construct four ratio features from the remaining statistics (Table II).

TABLE II  
RATE-DEPENDENCE SCORES BEFORE AND AFTER FEATURE CONSTRUCTION (s(X) FROM SECTION II-B2).
<table><tr><td>Original</td><td>Constructed</td><td>Sorig</td><td>Sconstr</td></tr><tr><td>std_pkt_size</td><td>min_max_ratio</td><td>0.673</td><td>0.027</td></tr><tr><td>burstiness</td><td>pkt_size_range_r</td><td>0.667</td><td>0.493</td></tr><tr><td>max_iat</td><td>activation_fraction</td><td>0.803</td><td>0.803</td></tr><tr><td>min_iat</td><td>iat_min_ratio</td><td>0.803</td><td>0.766</td></tr></table>

Both size ratios have lower linear correlation with rate than their original statistics. We additionally use Spearman correlation to compare the ranks of each original statistic and its constructed feature, rather than to test dependence on rate; this comparison does not determine feature selection. For the size ratios, these absolute rank correlations are at least 0.978 in both media. Here, burstiness is $( \sigma _ { s } - \bar { s } ) / ( \sigma _ { s } + \bar { s } )$ with $\sigma _ { s }$ the packet-size standard deviation. The timing ratios express gaps relative to observed duration or mean gap; activation\_fraction is smaller when the longest gap occupies more of the observed span.

TABLE III  
FINAL SEVEN DIMENSIONAL FEATURE SPACE.
<table><tr><td>Feature</td><td>Formula</td></tr><tr><td>rms_pkt_size</td><td> $\sqrt { \overline { { s ^ { 2 } } } }$ </td></tr><tr><td>max_pkt_size</td><td> $s _ { \mathrm { m a x } }$ </td></tr><tr><td>switching_ratio</td><td> $n _ { \Delta s } / ( N - 1 )$ </td></tr><tr><td>min_max_ratio</td><td> $s _ { \operatorname* { m i n } } / s _ { \operatorname* { m a x } }$ </td></tr><tr><td>pkt_size_range_r</td><td> $( s _ { \mathrm { m a x } } - s _ { \mathrm { m i n } } ) / \bar { s }$ </td></tr><tr><td>activation_fraction</td><td> $1 - \tau _ { \operatorname* { m a x } } / T$ </td></tr><tr><td>iat_min_ratio</td><td> ${ \tau _ { \mathrm { m i n } } } / { \bar { \tau } }$ </td></tr></table>

N: packet count; $s _ { i } , t _ { i } \colon$ packet size and timestamp; $\tau _ { i } = t _ { i } - t _ { i - 1 } ; \underline { { T } } =$ $t _ { N } \gets \{ 1 \} ; n _ { \ @ s } \colon$ number of size changes between consecutive packets; ${ \overline { { s ^ { 2 } } } } =$ $N ^ { - 1 } \sum _ { i }$ s<sup>2</sup><sub>i</sub> . Bars indicate means. Zero denominators are replaced by $1 0 ^ { - 9 }$

Table III defines the seven features. Computing them requires nine statistics: root-mean-square (RMS), maximum, minimum, and mean packet sizes; switching ratio; maximum, minimum, and mean inter-arrival times (IATs); and observed duration. The two means, previously excluded as classifier inputs, supply denominators. Switching ratio measures the fraction of consecutive packets with different sizes; activation\_fraction is the fraction of the observed span excluding the longest gap.

## III. EXPERIMENTAL SETUP

## A. Ethernet Datasets

The Ethernet data use an uncongested 1 Gb/s ns-3 link. IPERF TCP uses four training rates and is also evaluated without a rate limit. The balanced test set comes from separate captures with different simulation seeds and includes rates absent from training.

Rate dependence is assessed with 348 additional Ethernet windows. Of these, 270 vary the rates of Video Conference (six, 3–70 Mb/s), Bursty UDP and IPERF UDP (five each, 5–70 Mb/s), and IPERF TCP (four, measured at 5–197 Mb/s). The other 78 cover Gaming, VoIP, and Video Streaming at fixed rates. For Dynamic Adaptive Streaming over HTTP (DASH), the configured rate describes the video profile; segments are downloaded in bursts.

## B. Wi-Fi Simulation Scenarios

The Wi-Fi simulations use a modified version of ns-3.40 with a privately maintained model of the IEEE 802.11 physical (PHY) and medium access control (MAC) layers. The Wi-Fi links use IEEE 802.11be Extremely High Throughput (EHT) at 5 GHz, a 160 MHz channel, modulation and coding scheme (MCS) 5, one spatial stream, and an 800 ns guard interval. All application flows use the Best Effort (BE) access category. Each simulation lasts 60 s, and applications start at 2 s.

Table IV lists the downlink flows and rates using the application IDs in Table I. In S3–S6, BSS 1 carries the listed home-network flows, while a neighboring BSS 2 adds eight UDP uplink flows, each at 70 Mb/s, on the same channel. S7 covers all eight classes with nine downlink flows, each to a separate station. Its Video Conference, VoIP, and Gaming applications also generate uplink traffic (4, 0.10, and 0.11 Mb/s, respectively).

Each Wi-Fi scenario uses five simulation seeds (five runs per scenario), giving 30 runs for S1–S6.

The AP-side extractor logs candidate statistics by flow and window. The first window covers 2–7 s; each flow is labeled by its destination station’s application. Across 30 S1–S6 runs, 40 downlink flows yield 10,718 windows, 82 short of the 10,800 maximum (54 per flow per run), because congestion leaves some flows incomplete. S7 follows the same procedure.

## C. Implementation and Evaluation Protocol

For the Extreme Gradient Boosting (XGBoost) classifier [11], hyperparameters are selected by grid search using the initial 29 features and only Ethernet training data. Four-fold crossvalidation is performed using four Ethernet training captures, each generated with a different simulation seed. In each fold, three captures are used for training and one for validation, ensuring that overlapping windows from the same capture are not shared between training and validation. A fifth independent capture is reserved exclusively for Ethernet testing. Hyperparameters are selected based on mean validation accuracy, while macro-F1 is also calculated. The selected settings are 400 boosting rounds, maximum depth six, learning rate 0.08, row and column subsampling of 0.85, reg\_alpha = 0.1, and $r e g \_ l a m b d a = 1 . 0$ . To isolate the effect of feature selection and construction, the same hyperparameters are used to train a separate classifier for each of the six feature sets (29, 28, 16, 12, 9, and 7 inputs), using all Ethernet training windows.

The seven-feature model also runs online with the Wi-Fi extractor. For comparison, all six classifiers are evaluated offline on the same logged Wi-Fi simulation windows, without further training.

The four evaluation sets are Ethernet training and test, Wi-Fi development (S1–S6), and Wi-Fi test (S7), with counts in Table I. S1–S6 guide feature design but not model training; S7 is excluded from both. Training accuracy measures fit to the training data.

For S1–S6, accuracy, class recall, and macro-F1 are calculated from all 10,718 windows pooled across 30 runs. Each window has equal weight, rather than each scenario. This summarizes the combined development scenarios. For the held-out S7 scenario, accuracy is reported as the mean ± sample standard deviation across five runs, each containing 486 windows. Because all five runs contain the same number of windows, mean accuracy equals accuracy over all 2,430 pooled S7 windows; the standard deviation describes variation across runs. S7 class recall and macro-F1 also use pooled windows. Consecutive five-second windows overlap by four seconds and are not independent observations.

## IV. RESULTS AND DISCUSSION

## A. Effect of the Selection and Construction Stages

Fig. 2 shows the accuracy obtained with each successive feature set. We examine the main errors below.

Ethernet test accuracy is 99.28% with 29 features and 99.46% with 7 features, reaching its minimum after the rate-dependence filter.

Removing protocol (29 to 28 features) leaves S1–S6 accuracy unchanged because the TCP flag count still distinguishes TCP and UDP. With 29 features, 932 Video Streaming windows are classified as IPERF TCP and 266 Bulk TCP windows as Video Streaming; results with 28 features are similar.

The rate-dependence filter (28 to 16 features) removes the TCP flag count and lowers accuracy on the Ethernet test set, S1–S6, and S7. In S1–S6, 3,092 Video Conference, IPERF UDP, and Bursty UDP windows are classified as IPERF TCP.

TABLE IV  
WI-FI SCENARIOS: DOWNLINK FLOW RATES (MB/S) BY APPLICATION ID FROM TABLE I.
<table><tr><td>Scenario</td><td>Network</td><td>VP</td><td>VC</td><td>GM</td><td>VS</td><td>BT</td><td>IT</td><td>IU</td><td>BU</td><td>Flows</td><td>Windows</td></tr><tr><td>S1</td><td>One BSS</td><td>0.10</td><td>5,10</td><td>0.11, 0.11</td><td>20</td><td>15</td><td></td><td>200, 200</td><td>100</td><td>10</td><td>2,699</td></tr><tr><td>S2</td><td>One BSS</td><td>0.10</td><td>5,10</td><td>0.11</td><td>20</td><td>15</td><td>unb.</td><td>200,200</td><td>100</td><td>10</td><td>2,691</td></tr><tr><td>S3</td><td>Two BSSs*</td><td>0.10</td><td>5</td><td>0.11</td><td>20</td><td>一</td><td>1</td><td>一</td><td>一</td><td>4</td><td>1,080</td></tr><tr><td>S4</td><td>Two BSSs</td><td>0.10</td><td>5</td><td>0.11</td><td>20,10</td><td>一</td><td></td><td>一</td><td></td><td>5</td><td>1,350</td></tr><tr><td>S5</td><td>Two BSSs*</td><td>0.10</td><td>5</td><td>0.11</td><td>20</td><td>一</td><td>1</td><td>一</td><td>300</td><td>5</td><td>1,346</td></tr><tr><td>S6</td><td>Two BSSs*</td><td>0.10</td><td>5</td><td>0.11</td><td>20</td><td>15</td><td>一</td><td>一</td><td>300</td><td>6</td><td>1,552</td></tr><tr><td>S7</td><td>One BSS</td><td>0.10</td><td>8</td><td>0.11</td><td>20,10</td><td>50</td><td>unb.</td><td>20</td><td>10</td><td>9</td><td>2,430</td></tr></table>

Each value is the rate of one downlink flow; –: none; unb.: unbounded. <sup>\*</sup>BSS 1 is the home network; co-channel BSS 2 contributes eight UDP uplink flows at 70 Mb/s each. Flows: home-network downlink flows per run. Windows: evaluated windows across five simulation seeds. S1–S6 are development scenarios; S7 is the held-out test scenario.

![](images/dda5d42010c7d3b774fbbd97d09b654fa3bfffefacd6ad8cbb4a0cd88ffd0594.jpg)  
Fig. 2. Accuracy after each feature selection and construction stage (Section II-B; Ratios: feature construction) on the four evaluation sets of Section III-C. Ethernet training accuracy is 100% at every stage. Where the two Wi-Fi lines overlap, the upper value is S7 and the lower value is S1–S6.

These UDP classes use fixed training rates; IPERF TCP covers four. Higher-rate UDP traffic in S1–S6 and Video Conference traffic in S7 produce shorter mean inter-arrival times within the IPERF TCP training range.

The KS and MI criteria (16 to 12 features) remove mean\_iat, std\_iat, avg\_min\_pkt\_size, and directions, restoring correct classification of these UDP flows.

Redundancy reduction (12 to nine features) misclassifies all 540 windows of the 300 Mb/s Bursty UDP flows in S5 and S6 as IPERF UDP. Their constant 1,474-byte packets at 40 µs intervals resemble IPERF UDP rather than the variable sizes and gaps of Bursty UDP training traffic. With 12 features, TreeSHAP [12] identifies mean\_pkt\_size as the largest contribution toward Bursty UDP. It is removed as redundant with rms\_pkt\_size in training, yet adding it back restores correct classification. Features that are redundant in Ethernet training may affect Wi-Fi classification differently.

Feature construction (nine to seven inputs) retains three direct statistics and introduces four ratio-based features. With this set, Bursty UDP is classified correctly and both Wi-Fi sets achieve their highest accuracy. The remaining errors are examined in Sections IV-C and IV-D.

## B. Sensitivity to Selection Thresholds

The same nine raw features are selected when each threshold is varied separately within the following ranges: rate correlation,

0.8326–0.9712 (chosen: 0.97); KS distance, 0.2690–0.3558 (0.30); MI ratio for D > 0.30, 0.8964–1.1572 (1.00); minimum MI ratio, 0.5144–0.7768 (0.60); and redundancy correlation, 0.9416–0.9914 (0.98). The chosen rate threshold is close to the upper end of its range.

For joint threshold variation, we test one representative from each of $4 \times 9 \times 1 0 \times 5 \times 2 2 = 3 9 , 6 0 0$ regions; all retain the same nine raw features. Results are also unchanged across ten MI-estimator seeds. This establishes stability within the tested ranges, not global optimality.

## C. Classification Errors on S1–S6

With seven features, 165 IPERF TCP windows and ten Bulk TCP windows are classified as Video Streaming. This reduces IPERF TCP recall to 38.9% (Table V).

TABLE V  
RECALL (%) PER CLASS, ACCURACY, MACRO-F1 WITH 29 AND 7 FEATURES.
<table><tr><td></td><td colspan="2">S1-S6</td><td colspan="2">S7</td></tr><tr><td>Class</td><td>29</td><td>7</td><td>29</td><td>7</td></tr><tr><td>Bulk TCP</td><td>63.1</td><td>98.2</td><td>0.0</td><td>100</td></tr><tr><td>IPERF TCP</td><td>93.0</td><td>38.9</td><td>99.3</td><td>99.3</td></tr><tr><td>Video Streaming</td><td>49.8</td><td>99.7</td><td>98.3</td><td>100</td></tr><tr><td>Other five classes</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Accuracy</td><td>88.48</td><td>98.28</td><td>88.44 ± 0.27</td><td>99.92 ± 0.18</td></tr><tr><td>Macro-F1</td><td>0.840</td><td>0.937</td><td>0.831</td><td>0.999</td></tr></table>

Other five classes: VoIP, Video Conference, Gaming, IPERF UDP, and Bursty UDP. S1–S6: pooled metrics over 30 runs. S7: accuracy mean ± sample standard deviation across five equal-sized runs; its mean equals pooled accuracy. Class recall and macro-F1 use pooled windows.

For models using 29 to 12 features, TreeSHAP identifies observed duration as one of the two inputs most strongly favoring IPERF TCP. In Ethernet training, IPERF TCP windows span five seconds, while Video Streaming windows are shorter. On Wi-Fi, most windows of both classes span nearly five seconds. With these feature sets, about half of the Video Streaming windows are classified as IPERF TCP.

With seven features, nearly all Video Streaming windows are classified correctly, but many IPERF TCP windows are classified as Video Streaming. Because S1–S6 contain about seven times more Video Streaming windows than IPERF TCP windows, overall accuracy increases despite the lower IPERF TCP recall.

In S2, IPERF TCP and Video Streaming have similar packet sizes, making them harder to distinguish. In Ethernet training, IPERF TCP sends continuously (activation\_fraction $\ge 0 . 9 9 9 6 )$ , while Video Streaming alternates downloads and idle periods $( \leq ~ 0 . 6 8 )$ . High load introduces longer IPERF TCP gaps while making Video Streaming more continuous. Their timing becomes less distinct, and almost all IPERF TCP windows with activation\_fraction below 0.98 are classified as Video Streaming.

## D. Results on Test Scenario S7

IPERF TCP is unbounded in S2 and S7, but S7’s lower competing load allows higher throughput and shorter AP delay (Table VI). Only two S7 windows are misclassified, both as Video Streaming. They are the first two IPERF TCP windows of one run and include a startup pause that lowers activation\_fraction below 0.98.

TABLE VI  
TRAFFIC CONDITIONS IN S2 AND S7, AND IPERF TCP RECALL WITH SEVEN FEATURES.
<table><tr><td></td><td>S2</td><td>S7</td></tr><tr><td>Downlink load requested by the other flows (Mb/s)</td><td>≈550</td><td>≈120</td></tr><tr><td>IPERF TCP throughput (Mb/s)</td><td>12</td><td>197</td></tr><tr><td>Mean downlink delay of IPERF TCP at the AP (ms)</td><td>147</td><td>8</td></tr><tr><td>IPERF TCP windows with activation_fraction &lt; 0.98</td><td>63%</td><td>0.7%</td></tr><tr><td>Median activation_fraction of Video Streaming</td><td>0.97</td><td>0.72</td></tr><tr><td>IPERF TCP recall (%)</td><td>38.9</td><td>99.3</td></tr></table>

In S2, the AP delivers about 370 Mb/s in total. In Ethernet training, the median activation\_fraction of Video Streaming is 0.64.

With 29 features, all 270 Bulk TCP windows are classified as IPERF TCP. The higher Bulk TCP rate places its rate-dependent features within the IPERF TCP training range. Removing them is insufficient: most errors persist with 16, 12, and nine features. The longest gap in a Bulk TCP window decreases from 1.2 ms in training to 0.24 ms in S7, close to the IPERF TCP training value of 0.22 ms. TreeSHAP identifies duration, max\_iat, and std\_pkt\_size among the inputs favoring IPERF TCP.

Seven features classify all Bulk TCP windows correctly. Its maximum packet size remains 1,475 bytes, compared with at most 592 bytes for IPERF TCP and Video Streaming. Contributions from max\_pkt\_size, min\_max\_ratio, and switching\_ratio outweigh activation\_fraction, which still favors IPERF TCP. The nine-feature model already includes the same size statistics but still misclassifies most Bulk TCP windows. Among the evaluated models, all these windows are classified correctly only after feature construction.

Adaptive video traffic can alternate downloads and idle periods, use separate audio and video chunks, and vary segment size and bitrate [13]–[17]. These choices affect packet timing with load and playback state. Our evaluation uses one simulated DASH client and specific traffic generators, limiting generalization to other implementations and conditions.

## V. CONCLUSION

Our study shows that traffic classifiers trained only on Ethernet data can achieve high classification accuracy on Wi-Fi through careful feature selection and construction. We propose a procedure to reduce the large initial feature set to a compact set of seven flow features that capture packet-size and timing patterns while avoiding features that depend too strongly on traffic rate or differ sharply between Ethernet and Wi-Fi. In the ns-3 simulations, this compact feature space improved accuracy on the held-out Wi-Fi scenario (S7) from about 88% to about 99.9%, while Ethernet accuracy remained high at about 99.5%. Our conclusion is that careful feature selection and construction can help Ethernet-trained models transfer to Wi-Fi without using Wi-Fi data for classifier training. Future work will evaluate whether the selected features remain suitable under different traffic loads, streaming implementations, and real network conditions.

## ACKNOWLEDGMENT

This work was supported by the EU Horizon Europe Program in the frame of Marie Skłodowska-Curie Actions Doctoral Networks (MSCA-DN) project OWIN6G (Optical and wireless sensors networks for 6G scenarios) under Agreement No. 101119624.

## REFERENCES

[1] F. Rau, C. Herranz, I. Val, P. Georgieva, and J. Perez, “A novel flow-based online network traffic classification using machine learning,” TechRxiv, preprint, 2025, doi: 10.36227/techrxiv.175459507.75164419/v1.

[2] R. Xie, J. Cao, E. Dong, M. Xu, K. Sun, Q. Li, L. Shen, and M. Zhang, “Rosetta: Enabling robust TLS encrypted traffic classification in diverse network environments with TCP-aware traffic augmentation,” in Proc. 32nd USENIX Security Symposium, 2023, pp. 625–642.

[3] A. W. Moore and D. Zuev, “Internet traffic classification using Bayesian analysis techniques,” in Proc. ACM SIGMETRICS, 2005, pp. 50–60, doi: 10.1145/1064212.1064220.

[4] T. Shapira and Y. Shavitt, “FlowPic: A generic representation for encrypted traffic classification and applications identification,” IEEE Transactions on Network and Service Management, vol. 18, no. 2, pp. 1218–1232, 2021, doi: 10.1109/TNSM.2021.3071441.

[5] S. Cui, X. Han, D. Han, Z. Wang, W. Wang, B. Jiang, B. Liu, and Z. Lu, “FG-SAT: Efficient flow graph for encrypted traffic classification under environment shifts,” IEEE Transactions on Information Forensics and Security, vol. 20, pp. 5326–5339, 2025, doi: 10.1109/TIFS.2025.3571663.

[6] X. Yuan, Z. Yi, J. Zhang, and H. Zhang, “On the resilience of traffic features under concept drift in hidden service fingerprinting,” Computers, Materials & Continua, vol. 88, no. 3, 2026, art. no. 20, doi: 10.32604/ cmc.2026.072275.

[7] N. Smirnov, “Table for estimating the goodness of fit of empirical distributions,” The Annals of Mathematical Statistics, vol. 19, no. 2, pp. 279–281, 1948, doi: 10.1214/aoms/1177730256.

[8] Y. Polyanskiy and Y. Wu, Information Theory: From Coding to Learning. Cambridge University Press, 2025, doi: 10.1017/9781108966351.

[9] A. Kraskov, H. Stögbauer, and P. Grassberger, “Estimating mutual information,” Physical Review E, vol. 69, no. 6, p. 066138, 2004, doi: 10.1103/PhysRevE.69.066138.

[10] B. C. Ross, “Mutual information between discrete and continuous data sets,” PLoS ONE, vol. 9, no. 2, p. e87357, 2014, doi: 10.1371/journal. pone.0087357.

[11] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in Proc. ACM SIGKDD, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[12] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, “From local explanations to global understanding with explainable AI for trees,” Nature Machine Intelligence, vol. 2, no. 1, pp. 56–67, 2020, doi: 10.1038/s42256-019-0138-9.

[13] A. Rao, A. Legout, Y.-S. Lim, D. Towsley, C. Barakat, and W. Dabbous, “Network characteristics of video streaming traffic,” in Proc. ACM CoNEXT, 2011, pp. 1–12, doi: 10.1145/2079296.2079321.

[14] S. Akhshabi, A. C. Begen, and C. Dovrolis, “An experimental evaluation of rate-adaptation algorithms in adaptive streaming over HTTP,” in Proc. ACM Multimedia Systems Conf. (MMSys), 2011, pp. 157–168, doi: 10.1145/1943552.1943574.

[15] F. Loh, F. Poignée, F. Wamser, F. Leidinger, and T. Hoßfeld, “Uplink vs. downlink: Machine learning-based quality prediction for HTTP adaptive video streaming,” Sensors, vol. 21, no. 12, p. 4172, 2021, doi: 10.3390/ s21124172.

[16] P. Juluri, V. Tamarapalli, and D. Medhi, “QoE management in DASH systems using the segment aware rate adaptation algorithm,” in Proc. IEEE/IFIP NOMS, 2016, pp. 129–136, doi: 10.1109/NOMS.2016. 7502805.

[17] A. Biernacki, “Interpretable identification of dynamic adaptive streaming over HTTP (DASH) flows based on feature engineering,” Applied Sciences, vol. 15, no. 5, p. 2253, 2025, doi: 10.3390/app15052253.