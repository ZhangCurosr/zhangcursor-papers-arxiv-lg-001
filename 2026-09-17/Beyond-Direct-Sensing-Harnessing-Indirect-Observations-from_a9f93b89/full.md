# Beyond Direct Sensing: Harnessing Indirect Observations from Third-Party Sensors in Vehicle Tracking

Gaofeng Dong<sup>∗</sup>, Vamsi Eyunni<sup>∗</sup>, Pragya Sharma, Kang Yang, Mani Srivastava

Department of Electrical and Computer Engineering

University of California, Los Angeles

Los Angeles, CA, USA

{gfdong, veyunni, pragyasharma, kyang73, mbs}@ucla.edu

Abstract—Vehicle tracking is fundamental to applications ranging from urban mobility and public safety to security and defense. Conventional tracking relies on direct access to sensors that provide strong observations such as vehicle identity and location. In practice, however, factors such as ownership, privacy, cost, and operational constraints may limit directly accessible sensors, leaving sparse observations and long tracking gaps. Meanwhile, many additional third-party sensing assets may be present across the environment but remain inaccessible at the raw-data level, preventing their direct integration into the tracking system. In this work, we investigate whether weak, indirect observations with uncertain spatial and temporal cues can complement sparse direct sensing for vehicle tracking. Specifically, we propose GrayTrack, which fuses weak anonymous events with sparse direct observations using a roadconstrained particle filter. We build a CARLA–Mininet-WiFi pipeline to evaluate the system under controlled conditions, generating direct observations from accessible cameras and indirect observations from third-party cameras. Our learningbased detector achieves an F1 score of 0.989 for anonymous vehicle passages. Further, incorporating indirect third-party observations reduces trajectory RMSE by 60.1% and catastrophic track loss from 35.8% to 0.3%. These results demonstrate that GrayTrack can effectively exploit weak indirect observations to extend tracking capabilities.

Index Terms—Vehicle Tracking, Indirect Observations, Third-Party Sensing, Road Networks, Traffic Analysis

## I. INTRODUCTION

Vehicle tracking is critical across civilian and military applications, including urban mobility, public safety, and battlefield situational awareness. Such systems typically rely on sensors under the operator’s control or otherwise authorized for direct access, such as cameras that provide highquality observations of vehicle identity, location, and time. In practice, however, ownership, privacy, encryption, cost, and operational constraints often limit direct access to only a sparse subset of the available sensing infrastructure, leaving long observation gaps in which vehicle location becomes increasingly uncertain.

![](images/021b670cb9606345a524225dbf2136bdd23a60639e7106a4de0b448e3e02135c.jpg)  
Fig. 1. Vehicle tracking using sparse direct observations and weak anonymous observations from gray sensing assets.

Urban camera networks illustrate this gap between observation quality and coverage. As shown in Fig. 1, a city may deploy a limited number of public or authorized identitycapable cameras, such as automatic licence-plate readers, concentrated at selected intersections, poles, and highway ramps [1]. Meanwhile, the same environment may contain many more third-party sensing assets: Atlanta’s Connect Atlanta program, for example, encompasses tens of thousands of cameras contributed by businesses, residents, and other organizations [2]. Although raw observations from such assets may be unavailable due to ownership, privacy, or access restrictions [3], they may still provide weaker cues about activity in their vicinity. We refer to such externally owned or indirectly accessible sensing infrastructure as gray assets [4].

Rather than requiring gray assets to provide vehicle identity, appearance, or precise location, we consider a weak anonymous passage event: an observation indicating only that some vehicle passed a known location at approximately a given time. The downstream tracker operates on this generic event-level interface, which could be provided by motion or proximity sensors, RF-based sensing, or deployments exposing only anonymous presence events rather than raw sensing data. This allows heterogeneous and geographically distributed sensing assets to contribute through a common interface. In this work, we consider the common case where the third-party sensing assets are networked cameras, as illustrated in Fig. 1. Even when video content is encrypted, observable packet timing and traffic volume can reveal motionrelated information [4], [5]. We use these features only to infer anonymous vehicle-passage events, without circumventing encryption or accessing video content.

The challenge is that these indirect observations are inherently weak and noisy: they provide only coarse spatiotemporal evidence, cannot identify which vehicle generated an event, and may be missing, delayed, or spurious. Vehicle motion, however, is strongly constrained by physical structure. Road connectivity and plausible travel times can eliminate trajectory hypotheses that would otherwise remain feasible, suggesting that even weak local events may provide useful constraints during gaps between sparse direct observations. This leads to our central question: can indirect observations from third-party sensors meaningfully complement sparse direct observations for vehicle tracking?

We address this question with GrayTrack, a vehicletracking framework that combines anonymous passage events from gray assets with sparse direct observations under road-network constraints. We instantiate the event interface using encrypted third-party camera traffic and evaluate GrayTrack with a CARLA–Mininet-WiFi pipeline connecting measured passage-detection errors to controlled experiments.

Contributions. Our key contributions are summarized as follows. First, we introduce an event-level abstraction for gray-asset sensing and show that encrypted network traffic from third-party cameras can be transformed into anonymous vehicle-passage events using learning-based packet grouping and vehicle-visibility detection. Second, we develop a road-constrained particle-filtering framework that fuses these weak, identity-free events with sparse direct observations. In our evaluation, it reduces trajectory RMSE by 60.1% and catastrophic track loss from 35.8% to 0.3%, while outperforming unconstrained Kalman filtering by 27.6%. Third, we build a CARLA–Mininet-WiFi pipeline and conduct controlled experiments with held-out sensing sequences and simulated region-transit trajectories, systematically evaluating passage detection, sensing noise, road constraints, blind gaps, and multi-vehicle ambiguity. Project webpage: https://nesl. github.io/GrayTrack/

## II. RELATED WORK

Encrypted traffic as a sensing channel. Prior work has inferred device activity and household behaviour from IoT traffic. More recent work has shown that encrypted wireless traffic can reveal motion-related information and support physical-world inference [4], [5]. These studies establish encrypted traffic as a viable side channel for sensing beyond network-layer activity. Our goal differs in the information we extract and the system role assigned to the side channel. Rather than using encrypted traffic as the primary source for detailed activity or geospatial inference, we deliberately reduce each third-party sensor to a minimal anonymous passage event: a vehicle was observed near a known camera at approximately a given time. This abstraction avoids dependence on vehicle identity, appearance, or detailed scene reconstruction and is well suited to geographically distributed third-party sensors across a road network or large area. In contrast to prior settings that jointly observe a localized area, such as an intersection, using multiple cameras [4], our sensors may be spatially separated and individually provide only weak local evidence. The side channel is therefore not the final inference target; instead, we ask whether these indirect events can complement sparse identity-bearing direct observations when fused with physical constraints from the road network.

Multi-camera vehicle tracking and data association. Traditional multi-camera tracking associates observations across sensors using spatial-temporal consistency, multiplehypothesis reasoning [6], [7], or visual features such as vehicle appearance and re-identification [8], [9]. Our third-party observations provide none of these appearance or identity cues: an event reports only that a vehicle passed a particular sensing location at approximately a given time. Consequently, association must rely on the vehicle’s prior state, timing, and physical feasibility rather than visual re-identification. This makes our setting complementary to conventional multicamera tracking: instead of extracting richer features from accessible video, we investigate how much tracking information can be recovered when most intermediate sensors provide only indirect observations.

## III. SYSTEM AND PROBLEM FORMULATION

We consider vehicle tracking using two classes of observations that differ in accessibility and information content: sparse direct observations from accessible sensors and weaker indirect observations from third-party sensors.

Direct sensing. Directly accessible sensors provide vehiclespecific direct observations (t, x, y, id), containing a timestamp, location, and vehicle identity. These observations are informative but sparse because the observer has direct access to only a subset of the sensing infrastructure.

Indirect sensing. Third-party sensors are not directly accessible to the tracker. In this work, we consider the common case in which these sensors are networked cameras transmitting encrypted video over wireless links. Following prior work [4], [5], we assume that the observer deploys wireless traffic monitors that passively observe transmissions from nearby third-party cameras. A monitor may observe multiple cameras within wireless range without requiring line of sight. Observed traffic is associated with a known camera s, whose location $( x _ { s } , y _ { s } )$ is obtained from deployment metadata or registration, or inferred through prior calibration. The monitor retains only packet timing and size, without accessing the underlying video content. From this traffic, GrayTrack infers an anonymous passage event (t, s<sup>ˆ</sup> ) indicating that a vehicle passed near camera s around time t<sup>ˆ</sup>. Each event provides only coarse spatiotemporal evidence and may be missed, spurious, or temporally perturbed. The downstream tracker operates only on this event-level representation. In deployments where third-party sensor owners voluntarily expose anonymous presence or passage events, GrayTrack could consume them directly without wireless traffic monitoring, although we do not rely on such cooperation in this work.

Given sparse direct observations and indirect passage events, our goal is to continuously estimate vehicle position under road-network constraints.

Tracking problem. Let the road network be a directed graph $G = ( V , E )$ , where V denotes road junctions and E directed lane segments. At time t, the vehicle state is

$$
\begin{array} { r } { \mathbf { x } _ { t } = ( \mathbf { p } _ { t } , v _ { t } ) , \qquad \mathbf { p } _ { t } = ( x _ { t } , y _ { t } ) , } \end{array}
$$

where $\mathbf { p } _ { t }$ is the vehicle position and $v _ { t }$ its speed. Vehicle motion is constrained by the road graph $\mathbf { p } _ { t } ~ \in ~ G$ such that feasible trajectories follow connected road segments. The observer receives two asynchronous observation streams. Direct sensors provide sparse direct observations

$$
a _ { i } = ( t _ { i } , \mathbf { p } _ { i } , \mathrm { i d } ) , \qquad a _ { i } \in \mathcal { A } ,
$$

while third-party sensors provide indirect passage events

$$
e _ { j } = ( \hat { t } _ { j } , s _ { j } ) , \qquad e _ { j } \in \mathcal { E } ,
$$

where $s _ { j }$ has known location $\mathbf { c } _ { s _ { j } } = ( x _ { s _ { j } } , y _ { s _ { j } } )$ . An indirect event $e _ { j }$ provides only uncertain evidence that a vehicle was near sensor $s _ { j }$ around time $\hat { t } _ { j }$

Given the road graph $G ,$ direct-observation stream $A 1 : t ,$ and indirect-observation stream E1 : t up to time t, the tracking objective is to estimate

$$
p ( \mathbf { x } _ { t } \mid \mathcal { A } _ { 1 : t } , \mathcal { E } _ { 1 : t } , G ) ,
$$

and reconstruct the continuous trajectory $\hat { \mathbf { X } } _ { 0 : T } = \{ \hat { \mathbf { x } } _ { t } \} _ { t = 0 } ^ { T }$

## IV. METHOD

Fig. 2 summarizes GrayTrack. Encrypted traffic from third-party cameras is processed into indirect vehicle-passage observations, which are fused with sparse direct observations under road-network constraints to estimate continuous vehicle trajectories. Direct observations from accessible cameras can be obtained using established vision methods, such as vehicle detection and license-plate recognition. We therefore focus on the indirect sensing path.

![](images/07a1613288c2fe842ba91ef6f766a3aeae8193ab273f26f20556418610473fc4.jpg)  
Fig. 2. GrayTrack system overview: a road-constrained particle filter fuses indirect observations with sparse direct observations for vehicle tracking.

## A. Passage Detection from Encrypted Camera Traffic

For each third-party camera, we passively collect network traffic and retain only packet arrival times and sizes, without accessing the payloads or video content, to infer the occurrence and timing of vehicle pass-by events. Video codecs use inter-frame prediction, causing scene changes to alter encoded frame sizes that remain observable in encrypted traffic [4], [5]. Because frames may span multiple packets, we first recover frame boundaries from packet metadata and then detect vehicle visibility from the resulting frame-size sequence [4].

Stage 1: Packet grouping. For camera s, let $\begin{array} { r l } { \mathcal { P } _ { s } } & { { } = } \end{array}$ $\{ ( \tau _ { i } , b _ { i } ) \} _ { i = 1 } ^ { N _ { s } }$ , where $\tau _ { i }$ and $b _ { i }$ denote the arrival time and size of network packet i. A learned packet-grouping model predicts whether each network packet marks the end of an encoded video frame:

$$
\hat { m } _ { i } = f _ { \theta } \big ( \{ ( \tau _ { k } , b _ { k } ) \} _ { k \in \mathcal { W } _ { i } } \big ) ,
$$

where $\mathcal { W } _ { i }$ is a temporal packet window and $\hat { m } _ { i }$ is the predicted frame-boundary indicator. These boundaries partition the packets into inferred frames $\mathcal { G } _ { 1 } , \ldots , \mathcal { G } _ { M }$ . We recover the size of frame $j$ as

$$
B _ { j } = \sum _ { i \in \mathscr { G } _ { j } } b _ { i } ,
$$

yielding the frame-level sequence $\mathcal { F } _ { s } = \{ ( \hat { \tau } _ { j } , B _ { j } ) \} _ { j = 1 } ^ { M }$

Stage 2: Vehicle passage detection. For each inferred frame in ${ \mathcal F } _ { s } ,$ we construct a feature vector $\mathbf { z } _ { j }$ from its frame size $B _ { j }$ and local temporal context, including log-transformed frame size, exponentially weighted residual and ratio features, and frame time span. These features reduce sensitivity to absolute bitrate and transient network variation and instead emphasize relative changes in encoded frame complexity, reducing reliance on camera-specific absolute bitrate characteristics. A second learned model estimates vehicle visibility over a temporal window:

$$
\hat { y } _ { j } = g _ { \phi } \big ( \{ \mathbf { z } _ { k } \} _ { k \in \mathcal { V } _ { j } } \big ) ,
$$

where $\nu _ { j }$ is the temporal window around frame $j ,$ and $\hat { y } _ { j }$ is the predicted probability that a vehicle is visible in frame $j .$ Thus, although inference uses temporal context, visibility is predicted frame by frame. Consecutive positive prediction are merged into visibility intervals, each producing a passage event $e _ { q } = ( \hat { t } _ { q } , s )$ . The downstream tracker uses only these event-level observations, rather than the underlying packet or frame representations. For event-level evaluation, predicted and ground-truth passages are matched one-to-one using their interval midpoints within a matching tolerance.

## B. Indirect-Observation Model

We evaluate passage detection separately and use its measured event-level data to parameterize indirect observations in the tracking experiments. Starting from oracle events $\mathcal { E } ^ { * }$ generated from ground-truth trajectories and sensor geometry, we construct an observed stream $\tilde { \mathcal { E } }$ that models missed detections, false positives, and timing uncertainty. Each true event is retained with probability $1 - p _ { \mathrm { m i s s } }$ , false events are introduced according to a per-passage rate $\lambda _ { \mathrm { f a l s e } } .$ , and retained event timestamps are perturbed as

$$
\hat { t } _ { j } = t _ { j } ^ { * } + \delta _ { j } , \qquad \delta _ { j } \sim \mathcal { N } ( \mu _ { t } , \sigma _ { t } ^ { 2 } ) .
$$

The nominal parameters are derived from the measured passage-detector performance in Section V-B, integrating the sensing and tracking experiments while independently varying each source of error.

## C. Road-Constrained Particle Filter

We fuse direct and indirect observations using a bootstrap particle filter [10]. Each particle represents a vehicle hypothesis directly on the road graph $G = ( V , E )$ , with state

$$
\mathbf { x } _ { t } ^ { ( i ) } = \big ( e _ { t } ^ { ( i ) } , \ell _ { t } ^ { ( i ) } , v _ { t } ^ { ( i ) } \big ) ,
$$

where $e _ { t } ^ { ( i ) } ~ \in ~ E$ is the current lane segment, $\ell _ { t } ^ { ( i ) }$ is the position along the segment, and $v _ { t } ^ { ( i ) }$ is the vehicle speed. Between observations, particles propagate along the road with stochastic velocity perturbations; at junctions, they transition among feasible outgoing segments, maintaining multiple route hypotheses while remaining road-constrained.

Direct observations concentrate particles around the observed vehicle location. An indirect event $e _ { j } = ( \hat { t } _ { j } , s _ { j } )$ instead provides weaker evidence that the vehicle passed near sensor $s _ { j }$ at known location $\mathbf { c } _ { s _ { j } }$ . We model its likelihood for particle i as

$$
p ( e _ { j } \mid \mathbf { p } _ { \hat { t } _ { j } } ^ { ( i ) } ) \propto \exp \left( - \frac { \| \mathbf { p } _ { \hat { t } _ { j } } ^ { ( i ) } - \mathbf { c } _ { s _ { j } } \| _ { 2 } ^ { 2 } } { 2 \sigma _ { \mathrm { l o c } } ^ { 2 } } \right) ,
$$

where $\mathbf { p } _ { \hat { t } _ { i } } ^ { ( i ) }$ is the particle position and $\sigma _ { \mathrm { l o c } }$ captures spatial uncertainty. Thus, an indirect event reweights rather than relocates particles, favoring road-consistent hypotheses that pass near the reporting sensor at a compatible time. We then normalize and resample particle weights following the standard bootstrap particle-filter procedure, with the weighted particle distribution providing the estimated vehicle position.

## V. EVALUATION

## A. Experimental Setup

Implementation. Passage detection is evaluated on encrypted wireless camera traffic collected using our CARLA–Mininet-WiFi pipeline, where CARLA [11] generates vehicle motion and camera streams and Mininet-WiFi [12] emulates encrypted wireless transmission. We collect 2,607 camera traces across 237 runs and 11 cameras. Event-level passage detection is evaluated on 242 held-out camera sequences. Then we conduct controlled downstream tracking experiments on 120 simulated region-transit trajectories over the CARLA Town05 road network, totaling 35.9 km of vehicle travel. The region contains 12 sparse direct-sensing perimeter sensors and 15 indirect interior cameras. Indirect observations are parameterized by the measured event-level detector errors. Unless otherwise stated, the main tracking experiments consider a single vehicle at a time; concurrent vehicles are evaluated separately in Section V-G. The packet-grouping model, PacketSegformer, uses a four-layer Transformer encoder $( d _ { \mathrm { { m o d e l } } } = 1 6 )$ over 250-packet windows, using packet timing and size together with derived temporal features to predict frame boundaries. The vehicle-visibility detector, CarVisibleLSTM, uses a two-layer unidirectional LSTM with 64 hidden units over 512-frame windows and predicts frame-level vehicle visibility from frame-size and temporal features. Passage events are obtained by dropping short detections and fusing nearby intervals within a specified merge gap. For downstream tracking, Road-PF uses 2000 particles.

Evaluation conditions. We evaluate three sensing conditions: A, sparse direct observations only; B, sparse direct observations plus oracle indirect passage events, representing an upper bound on the benefit of indirect sensing; and C, sparse direct observations plus indirect passage events corrupted according to the measured event-level detector errors, representing a realistic noisy scenario.

Baselines. For passage detection, we compare PacketSegformer with fixed 50 ms packet binning at the known 20 Hz frame rate, and a rule-based detector with a hand-tuned median + 6×MAD frame-size threshold. For tracking, we compare Road-PF against dead reckoning and an unconstrained Kalman filter receiving the same direct and indirect observations. Dead reckoning propagates the latest observed state using its estimated velocity. The Kalman filter uses a constantvelocity model in 2D and treats passage events as uncertain position observations centered at the reporting sensor. Kalman and Road-PF use the same observation uncertainty, isolating the effect of road-network constraints.

Metrics. For passage detection, we report precision, recall, and F1 at the frame-boundary and passage-event levels. For tracking, we evaluate continuous trajectory reconstruction using per-trajectory position RMSE over the entire transit. We additionally report the catastrophic-failure rate, defined as the fraction of trajectories with RMSE exceeding 100 m, indicating that the estimated trajectory is no longer in the correct neighbourhood.

## B. Passage Detection from Encrypted Camera Traffic

We first evaluate whether encrypted packet metadata can be reliably transformed into the indirect passage observations required by the downstream tracker. We evaluate the two stages separately: network packet grouping measures recovery of encoded video frame boundaries, while vehicle-visibility detection evaluates whether the resulting frame-size dynamics accurately predict vehicle passages.

PacketSegformer achieves an F1 score of 0.998 for frameboundary detection, compared with 0.880 for the fixed-period baseline (Table I). The improvement is driven primarily by substantially higher recall. More importantly for downstream detection, the recovered boundaries yield a normalized framesize MAE of only 0.0025, compared with 0.8192 for the fixed-period baseline. The learning-based pipeline improves passage-event detection over the rule-based baseline, increasing F1 from 0.893 to 0.989. The rule-based approach heavily relies on hand-tuned, sequence-specific calibration, significantly limiting its scalability across heterogeneous deployments. We therefore use the learning-based pipeline for downstream experiments. Its event-level errors correspond to a missed-event rate of 0.0, a false-event rate of 0.0225 per true passage, and passage-time errors with mean bias 0.022 s and standard deviation 0.109 s; these measurements parameterize the indirect-observation model used in downstream tracking.

TABLE I  
PERFORMANCE OF FRAME-BOUNDARY RECOVERY AND PASSAGE-EVENT DETECTION.
<table><tr><td>Task</td><td>Method</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td rowspan="2">Frame-boundary detection</td><td>Fixed-period</td><td>0.926</td><td>0.838</td><td>0.880</td></tr><tr><td>PacketSegformer</td><td>0.997</td><td>0.999</td><td>0.998</td></tr><tr><td rowspan="2">Passage-event detection</td><td>Rule-based pipeline</td><td>0.885</td><td>0.900</td><td>0.893</td></tr><tr><td>Learning-based pipeline</td><td>0.978</td><td>1.000</td><td>0.989</td></tr></table>

## C. Tracking with Indirect Observations

We first ask whether indirect third-party observations provide meaningful tracking value beyond sparse direct sensing. Between two direct observations, a vehicle remains unobserved and its trajectory uncertainty grows over time. Anonymous passage events provide intermediate evidence about where the vehicle may have traveled, while the road network further restricts the set of physically feasible trajectories.

With direct observations alone, the road-constrained filter achieves a mean trajectory RMSE of 92.3 m. Incorporating indirect passage events at the measured detector error rates reduces RMSE to 36.8 m, a 60.1 % reduction, close to the 35.8 m RMSE achieved with oracle passage events. Thus, even noisy indirect observations recover a substantial fraction of the tracking benefit available from perfect intermediate sensing. The improvement is even more pronounced for catastrophic failures. The fraction of trajectories with RMSE above 100 m decreases from 35.8% with direct anchors alone to 0.3% with noisy indirect observations, nearly matching the 0% achieved with oracle events. These results show that indirect third-party observations can substantially reduce both average trajectory error and severe track loss between sparse direct observations.

## D. Robustness to Noisy Indirect Observations

We next evaluate how tracking performance degrades as indirect observations become less reliable. In our camerabased realization, errors can arise from passive wireless traffic monitoring, network variability, and event inference, resulting in missed passages, false events, or uncertain timestamps. We independently vary the missed-detection rate, false-event rate, and timing jitter while holding the other error sources at their measured nominal values, isolating the effect of each source. As shown in Fig. 3, tracking error increases gradually with each error source, and Road-PF remains substantially better than direct sensing alone across all evaluated noise levels. Even at severe settings tested, indirect observations continue to provide useful intermediate constraints on the vehicle trajectory.

TABLE II  
TRACKING PERFORMANCE UNDER THE THREE SENSING CONDITIONS. CATASTROPHIC FAILURE DENOTES TRAJECTORY RMSE > 100 m.
<table><tr><td>Estimator</td><td>RMSE (m)</td><td>Failure rate</td></tr><tr><td>A. Direct observations only</td><td></td><td></td></tr><tr><td>Dead reckoning, Kalman filter</td><td>166.8</td><td>74.2%</td></tr><tr><td>Road-PF</td><td>92.3</td><td>35.8%</td></tr><tr><td colspan="3">B. Direct observations + oracle indirect events (upper bound)</td></tr><tr><td>Dead reckoning</td><td>48.0</td><td>6.7%</td></tr><tr><td>Kalman filter</td><td>49.6</td><td>8.3%</td></tr><tr><td>Road-PF</td><td>35.8</td><td>0.0%</td></tr><tr><td colspan="3">C. Direct observations + noisy indirect events</td></tr><tr><td>Dead reckoning</td><td>50.5</td><td>8.3%</td></tr><tr><td>Kalman filter</td><td>50.8</td><td>8.8%</td></tr><tr><td>Road-PF</td><td>36.8</td><td>0.3%</td></tr></table>

The system is not equally sensitive to all error sources. Increasing missed detections and ghost events produces the largest degradation, while moderate timing uncertainty has a smaller effect on trajectory accuracy. Road-PF also consistently outperforms the unconstrained Kalman filter throughout the sweeps. Overall, these results show that the tracking benefit persists under substantial indirect-observation errors and does not depend on unrealistically accurate sensing.

## E. Estimator Comparison and Blind Gaps

We next evaluate how much road-network constraints contribute beyond the indirect observations. Under the calibrated noisy-observation setting, Road-PF achieves 36.8 m RMSE, a 27.6% reduction relative to Kalman filtering, with the lowest catastrophic-failure rate across all estimators. The road-constrained filter achieves the lowest error across all three sensing conditions, showing that road topology provides useful constraints both with and without indirect observations.

To understand when this advantage is most pronounced, we test two possible explanations. One possibility is that road constraints primarily help resolve ambiguity at junctions; however, we find no clear relationship between branch density and the particle filter’s advantage $( p ~ = ~ 0 . 9 0 )$ . In contrast, grouping trajectories by their longest gap between direct observations, the Road-PF advantage over Kalman filtering increases from 5.4 m on short-gap routes to 26.8 m on longgap routes $( p ~ < ~ 0 . 0 0 1 )$ . These results indicate that road constraints are particularly valuable during long predict-only intervals, when unconstrained propagation accumulates drift while road-constrained particles remain confined to physically feasible trajectories.

## F. End-to-end Validation

Our primary evaluation deliberately separates passage detection from downstream tracking for controlled analysis.

![](images/41ebddcdb826f6b797265f392e596525366677c56c9042b5df6cdb29a197f1a8.jpg)  
Road-PF Kalman perimeter-only (A) oracle interior (B) measured (event eval)  
Fig. 3. Robustness to indirect-observation errors. Each error source is varied independently while the others are held at their measured nominal values. Horizontal references show direct sensing only (A) and oracle indirect observations (B), and vertical markers indicate the operating points from Section V-B.

End-to-end replay validates this abstraction: Road-PF RMSE drops from 78.0 m with direct observations only to 44.3 m with detected events, closely matching 44.1 m with oracle events. This result is consistent with our controlled evaluation and confirms that the measured detector performance transfers effectively to downstream tracking.

## G. Multi-Vehicle Case Study

We finally evaluate 15 scenarios each with one, two, and three concurrently active vehicles, with overlapping transit intervals to induce data-association ambiguity among passage events. Direct observations carry vehicle identity and are applied directly to the corresponding track, whereas indirect events are assigned using gated Hungarian matching based on the distance between each track’s predicted mean position and the reporting sensor. Once assigned, Road-PF updates the corresponding particles using the likelihood in Section IV-C; the Kalman baseline uses greedy likelihood-based association followed by a standard measurement update.

For the road-constrained filter, event-to-track assignment accuracy decreases from 1.00 with one vehicle to 0.87 with two and 0.63 with three, showing increasing ambiguity with vehicle density. This ambiguity becomes more challenging as the number of vehicles increases and the closest inter-vehicle separation decreases. At three vehicles, Road-PF achieves higher assignment accuracy (0.63 versus 0.58) and fewer identity switches (2.5 versus 3.1) than the Kalman filter. Road-PF also significantly reduces trajectory RMSE (69.2 versus 82.6 m) and track-loss rate (0.15 versus 0.33). These results highlight anonymous event-to-vehicle association as an important remaining challenge in multi-vehicle settings, while road constraints continue to improve overall tracking accuracy and robustness. Our multi-vehicle study focuses on low-to-moderate traffic regimes, which are relevant to localized monitoring scenarios such as perimeter crossings, sparse road networks, rural areas, and mission-critical operating zones. Even in these ambiguous settings, indirect observations provide intermediate constraints during gaps between sparse direct observations, where tracking would otherwise rely largely on motion and road-network priors alone.

## VI. DISCUSSION AND FUTURE DIRECTIONS

This work uses encrypted camera traffic to obtain indirect observations. More generally, the downstream tracker operates on a generic event-level interface that can support other sensing modalities, such as motion or proximity sensors, as well as privacy-preserving deployments exposing only anonymous presence or passage events. In addition, the learning-based models are trained offline using frame-boundary and vehiclevisibility labels. At deployment time, inference requires only packet timing and size and does not require access to the underlying video content. Improving generalization to unseen cameras, thereby reducing or eliminating the need for labeled calibration data, remains a direction for future work, including training on data from more diverse codecs and camera viewpoints and developing more generalizable models. For multivehicle settings, future work can explore scalable probabilistic or multi-hypothesis data association for anonymous events.

## VII. CONCLUSION

We presented GrayTrack, a vehicle-tracking framework that combines sparse direct observations with third-party indirect observations from gray assets under road-network constraints. Using encrypted camera traffic as one realization of this event-level interface, GrayTrack reduces trajectory RMSE by 60.1% and catastrophic track loss from 35.8% to 0.3%. More broadly, our results demonstrate the value of indirect sensing for extending tracking beyond the coverage of directly accessible sensors.

## VIII. ACKNOWLEDGMENT

The research reported in this paper was sponsored in part by: the DEVCOM Army Research Laboratory under award #W911NF1720196; the National Science Foundation under awards #CNS-2211301 and ECCS-2525614; and Sandia National Laboratories under award #2169310. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the funding agencies.

[1] T. Monahan, “Grounding the Flock: Confronting police surveillance of mobilities,” Mobile Media & Communication, 2026.

[2] M. Scott, “28,000 cameras and counting, but Atlanta’s clearance rates aren’t rising,” Atlanta Community Press Collective, 2026, published July 28, 2026; accessed Aug. 11, 2026.

[3] B. Lipton, “Neighborhood watch out: Cops are using Fusus to incorporate private cameras into their real-time surveillance networks,” Electronic Frontier Foundation, Deeplinks, 2023, published May 11, 2023; accessed Aug. 11, 2026.

[4] S. Y. Yetim, G. Dong, I.-N. Zanoria, R. Barman, M. Wigness, T. Abdelzaher, M. Srivastava, and S. Diggavi, “Tracking without seeing: Geospatial inference using encrypted traffic from distributed nodes,” arXiv preprint arXiv:2603.27811, 2026.

[5] M. B. Rasool, U. M. Shah, M. Imran, D. M. Minhas, and G. Frey, “Invisible eyes: Real-time activity detection through encrypted wi-fi traffic without machine learning,” Internet ofThings, vol. 31, p. 101602, 2025.

[6] D. B. Reid, “An algorithm for tracking multiple targets,” IEEE Trans. Automatic Control, vol. 24, no. 6, pp. 843–854, 1979.

[7] H. Yao, Z. Duan, Z. Xie, J. Chen, X. Wu, D. Xu, and Y. Gao, “Cityscale multi-camera vehicle tracking based on space-time-appearance features,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). IEEE, 2022, pp. 3309– 3317.

[8] X. Liu, W. Liu, T. Mei, and H. Ma, “A deep learning-based approach to progressive vehicle re-identification for urban surveillance,” in Proc. European Conf. on Computer Vision (ECCV), 2016, pp. 869–884.

[9] T. T. Nguyen, H. H. Nguyen, M. Sartipi, and M. Fisichella, “Multivehicle multi-camera tracking with graph-based tracklet features,” IEEE Transactions on Multimedia, vol. 26, pp. 972–983, 2023.

[10] N. J. Gordon, D. J. Salmond, and A. F. M. Smith, “Novel approach to nonlinear/non-Gaussian Bayesian state estimation,” IEE Proceedings F (Radar and Signal Processing), vol. 140, no. 2, pp. 107–113, 1993.

[11] A. Dosovitskiy, G. Ros, F. Codevilla, A. Lopez, and V. Koltun,´ “CARLA: An open urban driving simulator,” in Proc. Conf. on Robot Learning (CoRL), 2017, pp. 1–16.

[12] R. R. Fontes, S. Afzal, S. H. B. Brito, M. A. S. Santos, and C. E. Rothenberg, “Mininet-WiFi: Emulating software-defined wireless networks,” in Proc. Int. Conf. on Network and Service Management (CNSM), 2015, pp. 384–389.