# FedQoS: Federated QoS-Risk Learning for Heterogeneous Indoor-Outdoor Access Selection

Nguyen Van Thieu, Ti Ti Nguyen, Ons Aouedi, Zerihun Huruy, Vu Nguyen Ha, and Symeon Chatzinotas Interdisciplinary Centre for Security, Reliability and Trust (SnT), University of Luxembourg, Luxembourg {vanthieu.nguyen, titi.nguyen, ons.aouedi, huruy.zerihun, vu-nguyen.ha, symeon.chatzinotas}@uni.lu

Abstract—Reliable access selection in dynamic and heterogeneous indoor-outdoor environments is challenging because instantaneous radio measurements alone cannot capture future QoS degradation caused by mobility, blockage, traffic load, and resource competition. This paper proposes FedQoS, a federated QoS-risk learning framework for predicting the future reliability of candidate access links and supporting access-node selection without centralizing user-level network data. In FedQoS, each access node locally learns from its observed network logs, including radio, traffic, load, and service-context features, while a global QoS-risk predictor is trained through federated aggregation. The learned model estimates the probability of QoS failure for each candidate link, and the controller uses these risk scores to select reliable access nodes under dynamic network conditions. To evaluate the framework, we construct physicsbased synthetic indoor-outdoor wireless datasets using the Sionna framework, covering normal traffic, mobility, event-driven congestion, and non-IID client observations. Simulation results show that learning-based access selection substantially reduces the QoS-failure rate compared with signal-based and historical-QoS heuristic methods. FedQoS achieves near-centralized predictive performance and provides clear reliability gains under mild non-IID data while remaining competitive under the more challenging severe non-IID condition. These results demonstrate the potential of federated QoS-risk learning for reliable, data-local access selection in dynamic wireless environments.

Index Terms—Federated learning, wireless network, quality of service, handover, indoor-outdoor connectivity.

## I. INTRODUCTION

Heterogeneous wireless access networks are becoming a key component of future indoor-outdoor connectivity. In dense environments such as enterprise buildings, public venues, transportation hubs, and university campuses, users move across rooms, corridors, building entrances, outdoor walkways, and temporarily crowded areas, while their candidate serving nodes span indoor access points (APs), terrestrial base stations (BSs), and static or quasi-static aerial nodes. Deciding which node should serve each user at a given time, the user association or access selection problem, directly affects throughput, load balancing, delay, handover behavior, and quality of service [1].

Conventional network selection mechanisms are based on radio-layer metrics such as received signal strength, received power, or SINR [2]. While these signal-oriented criteria are straightforward and interpretable, they frequently fall short in heterogeneous networks. For example, a user attached to the node with the strongest signal may still experience poor service if that node is heavily loaded, whereas an alternative node with a weaker signal may deliver higher long-term throughput or reduced latency. This limitation is well recognized in the literature; max-SINR or max-power association can cause severe load imbalance [3], motivating biasing and load balancing schemes [1], while QoS and delay-aware methods further incorporate attainable rate, load, required resources, or queueing delay into the association decision [4], [5]. These studies establish that the association should not be driven by instantaneous radio strength alone.

Learning-based methods have since been investigated for association and radio resource management. Deep reinforcement learning (DRL) and multi-agent learning are applied when explicit optimization is difficult due to non-convexity, dynamic, or incomplete data. On the other hand, federated learning (FL) has been introduced to reduce raw-data sharing in wireless control, e.g., federated DRL for user access control in Open RAN and for distributed control of NextG networks [6], [7]. Despite this progress, most learning-based association works formulate the problem as direct control or RL and learn a policy from a reward signal [8]. Such policies can be difficult to interpret and tune because their behavior depends strongly on the reward design. Moreover, existing federated wireless-control frameworks [9] mainly target policy learning, resource allocation, or user-side learning, rather than access node-side supervised QoS-risk prediction from local logs [10]. Yet in practical deployments, each AP, BS, or aerial node naturally observes local user-candidate measurements, load states, service demands, handover context, and realized QoS outcome, exactly the signals needed to learn whether a choice will lead to a QoS violation over a short future horizon [11].

This paper proposes FedQoS, a federated QoS-risk learning framework for heterogeneous indoor-outdoor access selection. Instead of directly learning an association action, FedQoS learns a supervised probabilistic model that predicts the future QoS-failure risk of each user-candidate pair, i.e., the probability that a candidate AP, BS, or static aerial node will violate the user’s QoS requirement within a prediction window. The QoSrisk predictor incorporates access-node load through its input features, while the access controller combines the resulting risk score with an explicit handover cost to select the serving node. The federated design follows the distributed nature of accessnode logs, where each node trains on its own observations and shares only model updates. Because these clients are naturally non-IID (differing in user density, mobility, load, and QoS-failure rates), FedQoS couples stabilized local training with a QoS-aware aggregation weight that reflects both local data volume and local QoS-failure exposure, so that nodes observing QoS-critical conditions contribute more without discarding the statistical reliability of larger clients.

The main contributions of this paper are summarized as follows:

• Formulate heterogeneous indoor-outdoor access selection as a federated supervised QoS-risk prediction problem over user-candidate access pairs.

• Propose FedQoS combining stabilized local training with a QoS-aware aggregation rule to handle heterogeneous, non-IID observations from APs/BSs/aerial nodes.

• Develop a reliability-oriented access-selection rule that converts predicted QoS-failure probabilities into access decisions, with access-node load captured by the learned risk model and handover cost handled explicitly by the controller.

• Construct realistic dynamic datasets using ray-tracing and system-level wireless simulation and evaluate FedQoS against centralized, local-only, FedAvg, and FedProx learning and representative non-learning baselines.

## II. SYSTEM MODEL AND PROPOSED FEDQOS ALGORITHM

## A. Heterogeneous Indoor-Outdoor Access Setting

We consider a heterogeneous indoor-outdoor access network composed of indoor APs, terrestrial BSs, and static aerial access nodes (ANs) serving a varying number of users within a campus area, as illustrated in Fig. 1. The ANs are considered to provide a flexible connection to the users. In the following, it is assumed that their positions are fixed during the considered decision process of T time slots. It is worth noting that trajectory and placement optimization are out of the scope of this work.

Let A be the set of all access nodes, and denote by $\mathcal { U } ^ { t }$ the set of active users in slot t, where $t \in \{ 0 , 1 , \ldots , T - 1 \}$ . This work aims to develop an FL-based access controller to instruct users to select the most suitable access node for their next communication. In particular, in the slot t, the controller can select an access node $a _ { u } ^ { t } \in \mathcal A$ from the candidate set $\mathcal { C } _ { u } ^ { t } \subseteq \mathcal { A }$ that contains a limited number of feasible service nodes selected from the measurement report. In the implementation, the candidate set is formed from the current serving node, the top-K candidates according to the logging association score, and occasional probing candidates. To maintain the users’ QoS, the access node should be selected for each active user so that the corresponding transmission rate for that connection is not less than $R _ { u } ^ { \mathrm { { m i n } } }$

To build such a controller, in this system, each access node acts as a federated client where its local data set is denoted by D<sub>i</sub>. A user–candidate sample observed while user u is served by node $a _ { u } ^ { t }$ is stored at the serving node. Therefore, the observer client for that sample is $o _ { u } ^ { t } = a _ { u } ^ { t }$ , and the sample belongs to $\mathcal { D } _ { o _ { u } ^ { t } }$ . This construction matches practical operation, where the access node can collect its users’ measurements, service context, recent history, and candidate reports.

The objective of the access controller is to select, for each user, an access node that is likely to satisfy the user’s qualityof-service (QoS) requirement over a short future horizon. Unlike conventional access selection based only on the strongest received signal, the considered environment is affected by indoor blockages, floor transitions, wall penetration loss, dynamic user density, access-node load, and heterogeneous AP/BS/AN coverage. Therefore, the access node with the strongest instantaneous signal is not necessarily the one that provides the lowest QoS-failure risk.

## B. User–Candidate Feature Vector

For user u and access node $^ { a , }$ the local controller constructs a feature vector for slot t as

$$
\begin{array} { r } { \mathbf { x } _ { u , a } ^ { t } = \big [ \mathbf { c } _ { u } ^ { t } , \mathbf { c } _ { a } , \mathbf { m } _ { u , a } ^ { t } , \mathbf { l } _ { a } ^ { t } , \mathbf { r } _ { u } ^ { t } , \mathbf { h } _ { u , a } ^ { t } \big ] , } \end{array}\tag{1}
$$

where $\mathbf { c } _ { u } ^ { t }$ denotes user-side context such as coarse indoor/outdoor zone, floor index, mobility state, and radiopositioning features; $\mathbf { c } _ { a }$ denotes access-node attributes such as node type, carrier, bandwidth, transmit power, antenna configuration, and height; $\mathbf { m } _ { u , a } ^ { t }$ contains instantaneous or recently averaged link measurements such as reference signal received power (RSRP), received signal strength indicator (RSSI), signal-to-interference-plus-noise ratio (SINR), channel quality indicator (CQI), block error rate (BLER), or achievable-rate indicators; $\mathbf { l } _ { a } ^ { t }$ contains access-node load information such as active-user count, physical resource block (PRB) utilization; $\mathrm { \bf r } _ { u } ^ { t }$ represents the user service requirement, including $R _ { u } ^ { \mathrm { { m i n } } }$ and $\mathbf { h } _ { u . a } ^ { t }$ contains handover-related indicators such as whether $a = a _ { u } ^ { t - 1 }$ , dwell time, and recent handover history.

The proposed framework does not require raw user logs to be uploaded to a central server. Each access node stores the logs observed for users served or probed by that node. The central controller receives only model parameters or model updates during federated training.

## C. Future QoS-Failure Label

The learning target is the future QoS outcome of a candidate access decision. Let $\Delta$ be the prediction horizon. If user u is associated with candidate access node a at slot $t ,$ let $\bar { R } _ { u , a } ^ { t , \Delta }$ and $\bar { \epsilon } _ { u , a } ^ { t , \Delta }$ denote the average throughput and BLER over the future window $[ t , t + \Delta ]$ . Given the min-rate requirement $R _ { u } ^ { \mathrm { { m i n } } }$ and max tolerable BLER $\epsilon _ { \mathrm { m a x } } .$ , the QoS-failure label is defined as

$$
y _ { u , a } ^ { t } = \mathbf { 1 } _ { \left( \bar { R } _ { u , a } ^ { t , \Delta } < R _ { u } ^ { \operatorname * { m i n } } \right) \mathrm { o r } \ } \left( \bar { \epsilon } _ { u , a } ^ { t , \Delta } > \epsilon _ { \operatorname * { m a x } } \right) \cdot\tag{2}
$$

Thus, $y _ { u , a } ^ { t } = 1$ means that candidate node a is expected to violate the QoS requirement of user u over the prediction horizon, whereas $y _ { u , a } ^ { t } ~ = ~ 0$ means that the candidate is expected to satisfy the requirement.

![](images/c39c3dc3753aad4b2c5115c47fc0cc541a6366f3dd2a7155d9ba4a1bc6df9db2.jpg)  
(1) Heterogeneous Indoor-Outdoor Environment and Data Collection  
(3) Handover-Aware Access Selection Phase (Online Decision-Making)  
Fig. 1: An example of an indoor-outdoor access network and the FedQoS system.

The learning target is not the instantaneous SINR alone. SINR is treated as one of the explanatory features. The label in (2) captures the combined effect of radio quality, resource contention, service demand, and mobility over a short future horizon. This formulation is useful because two candidate links with similar SINR can have different QoS outcomes due to different load, scheduling, or handover conditions.

## D. QoS-Risk Learning

Let $\mathcal { D } _ { i } ~ = ~ \{ ( \mathbf { x } _ { n } , y _ { n } ) \} _ { n = 1 } ^ { N _ { i } }$ denote the local dataset stored at access node $i \in { \mathcal { A } } ,$ , where $N _ { i } ~ = ~ | { \mathcal { D } } _ { i } |$ . The datasets are naturally non-IID because access nodes observe different locations, building structures, user densities, channel conditions, service mixes, and mobility patterns. For example, indoor APs mainly observe room and corridor users, whereas BSs and UAVs observe more outdoor or overloaded-event users.

The QoS-risk model is denoted by $f _ { \pmb { \theta } } ,$ , where θ is the model parameter vector. Given the input feature vector $\mathrm { \bf x } _ { u , a } ^ { t } .$ , the model outputs

$$
p _ { u , a } ^ { t } = f _ { \theta } \left( \mathbf { x } _ { u , a } ^ { t } \right) \in [ 0 , 1 ] ,\tag{3}
$$

where $p _ { u , a } ^ { t }$ estimates the probability that candidate a will fail the QoS requirement of user u over the prediction horizon:

$$
p _ { u , a } ^ { t } \approx \mathrm { P r } \left( y _ { u , a } ^ { t } = 1 \mid \mathbf { x } _ { u , a } ^ { t } \right) .\tag{4}
$$

The local empirical loss at client i is

$$
F _ { i } ( \pmb \theta ) = \frac { 1 } { | \mathscr { D } _ { i } | } \sum _ { ( \mathbf x , y ) \in \mathscr { D } _ { i } } \ell \left( f _ { \pmb \theta } ( \mathbf x ) , y \right) ,\tag{5}
$$

where $\ell ( \cdot , \cdot )$ is the binary cross-entropy loss. Then the conventional centralized solution would minimize

$$
\operatorname* { m i n } _ { \pmb { \theta } } \sum _ { i \in \mathcal { A } } \frac { N _ { i } } { N } F _ { i } ( \pmb { \theta } ) , \qquad N = \sum _ { i \in \mathcal { A } } N _ { i } ,\tag{6}
$$

which requires collecting all local access logs at a central server. FedQoS instead solves this problem federatively, where each access node keeps $\mathcal { D } _ { i }$ locally and exchanges only model parameters with an edge controller.

## E. FedQoS Local Update

At communication round r, the edge controller broadcasts the current global model $\pmb { \theta } ^ { r }$ to a subset of participating access nodes $S ^ { r } \subseteq A .$ Each selected client $\textit { i } \in \textit { S } ^ { r }$ initializes its local model by $\mathbf { \theta } _ { i , 0 } ^ { r } = \mathbf { \theta } ^ { r }$ . FedQoS builds its local training step on a proximal objective [12] to stabilize learning under heterogeneous local data. Specifically, client i minimizes

$$
\widetilde { F } _ { i } ( \pmb \theta ; \pmb \theta ^ { r } ) = F _ { i } ( \pmb \theta ) + \frac { \mu } { 2 } \left\| \pmb \theta - \pmb \theta ^ { r } \right\| _ { 2 } ^ { 2 } ,\tag{7}
$$

where $\mu \geq 0$ penalizes the deviation of the local model from the current global model. After E local epochs, client i obtains $\pmb { \theta } _ { i } ^ { r + 1 }$ and uploads it to the edge controller.

## F. FedQoS Aggregation

A standard sample-size weighted aggregation can be dominated by high-traffic access nodes, such as a BS serving many outdoor users. However, for QoS-risk learning, clients that observe many rare failure events are also important. FedQoS therefore uses a QoS-aware aggregation rule that balances sample size and local risk. We define the local QoS-failure rate as $\begin{array} { r } { \rho _ { i } ^ { r } = \frac { 1 } { N _ { i } } \sum _ { ( \mathbf { x } , y ) \in \mathcal { D } _ { i } } y } \end{array}$ and the unnormalized FedQoS aggregation weight as follows:

Algorithm 1: FedQoS: Federated QoS-Risk Learning and   
Handover-Aware Access Selection   
Require: Access nodes ${ \mathcal { A } } ,$ local datasets $\{ { \mathcal { D } } _ { i } \} _ { i \in { \mathcal { A } } } ,$ rounds $R ,$   
epochs E, coefficient µ, q, λ, η<sub>h</sub>, ϵ<sub>max</sub>, $\xi _ { h } ,$ , and $T _ { d w e l l } .$   
Ensure: Global QoS-risk model $f _ { \theta ^ { R } }$   
1: Initialize global model parameters $\theta ^ { 0 } .$   
2: for round $r = 0 , 1 , \ldots , R - 1$ do   
3: Edge server selects participating clients $S ^ { r } \subseteq A .$   
4: Broadcast $\pmb { \theta } ^ { r }$ to all clients $i \in S ^ { r } .$   
5: for each client $i \in S ^ { r }$ in parallel do   
6: Set local model ${ \pmb \theta } _ { i , 0 } ^ { r } = \mathbf { \bar { \theta } } ^ { r }$ and train it using (7).   
7: Obtain updated local model $\pmb { \theta } _ { i } ^ { r + 1 }$   
8: Compute aggregation weight $\omega _ { i } ^ { r }$ by (8)   
9: Upload $\pmb { \theta } _ { i } ^ { r + \mp 1 }$ and $\omega _ { i } ^ { r }$ to the controller   
10: end for   
11: Compute aggregation coefficients $\{ \alpha _ { i } ^ { r } \} _ { i \in S ^ { r } }$ by (9)   
12: Update the global model using (10).   
13: end for   
14: Deploy $f _ { \pmb { \theta } ^ { R } }$ to all access nodes.   
15: for each access-decision slot t do   
16: for each active user $u \in \mathcal { U } ^ { t }$ do   
17: Construct candidate set $\mathcal { C } _ { u } ^ { t } .$   
18: Predict $p _ { u , a } ^ { t }$ for all $a \in \mathcal { C } _ { u } ^ { t }$ using (3).   
19: Select $a _ { u } ^ { \star , t }$ by (12)   
20: Trigger handover only if (13) is satisfied   
21: end for   
22: end for

$$
\omega _ { i } ^ { r } = ( N _ { i } ) ^ { q } \left( 1 + \lambda \rho _ { i } ^ { r } \right) , \qquad i \in \mathcal { S } ^ { r } .\tag{8}
$$

where $q ~ \in ~ [ 0 , 1 ]$ controls the dependence on client dataset size and $\lambda \geq 0$ controls the emphasis on clients that observe more QoS failures. Setting $q = 1$ and $\lambda = 0$ recovers standard sample-size weighted aggregation. Smaller values of q reduce the domination of very large clients, while positive λ increases the influence of high-risk clients. Then the normalized aggregation coefficient can be calculated as

$$
\alpha _ { i } ^ { r } = \frac { \omega _ { i } ^ { r } } { \sum _ { j \in { \cal S } ^ { r } } \omega _ { j } ^ { r } } , \qquad i \in { \cal S } ^ { r } .\tag{9}
$$

The global model is updated as

$$
\pmb { \theta } ^ { r + 1 } = \sum _ { i \in \cal S ^ { r } } \alpha _ { i } ^ { r } \pmb { \theta } _ { i } ^ { r + 1 } .\tag{10}
$$

This aggregation rule encourages the global model to preserve information from high-risk access regions without ignoring the statistical reliability of larger local datasets.

## G. Handover-Aware Access Selection

After training, the global model is deployed at the edge controller or at local access controllers. For a candidate $a \in$

$\mathcal { C } _ { u } ^ { t }$ , the model estimates the probability of QoS failure $p _ { u , a } ^ { t } .$ The access decision minimizes a handover-aware risk score:

$$
J _ { u , a } ^ { t } = p _ { u , a } ^ { t } + \eta _ { \mathrm { h } } \mathbf { 1 } _ { a \neq a _ { u } ^ { t - 1 } } ,\tag{11}
$$

where $\eta _ { \mathrm { h } } \geq 0$ is the handover cost coefficient. The indicator $\mathbf { 1 } \left[ a \neq a _ { u } ^ { t - 1 } \right]$ equals one if selecting candidate a triggers a handover, and zero otherwise. The best candidate is

$$
a _ { u } ^ { \star , t } = \arg \operatorname* { m i n } _ { a \in \mathcal { C } _ { u } ^ { t } } J _ { u , a } ^ { t } .\tag{12}
$$

To avoid ping-pong handovers, the controller triggers a handover only if the candidate improves the current serving score by at least a hysteresis margin $\xi _ { \mathrm { h } }$ and the user has remained with the current serving node for at least $T _ { \mathrm { d w e l l } }$ slots:

$$
J _ { u , a _ { u } ^ { t - 1 } } ^ { t } - J _ { u , a _ { u } ^ { \star , t } } ^ { t } > \xi _ { \mathrm { h } } , \qquad t - t _ { u } ^ { \mathrm { l a s t } } \geq T _ { \mathrm { d w e l l } } ,\tag{13}
$$

where $t _ { u } ^ { \mathrm { l a s t } }$ is the most recent handover time of user u. If (13) is not satisfied, the user remains connected to $a _ { u } ^ { t - 1 }$ . Finally, Algorithm 1 summarizes the proposed FedQoS approach.

## III. EXPERIMENTAL SETUP

We use Sionna RT and Sionna SYS [13], [14] to generate physics-based network logs for heterogeneous indoor-outdoor environments. The logs are partitioned by observing access node, and each node trains only on its local data while exchanging model parameters with the edge controller. The resulting models are evaluated as both QoS-failure predictors and access-selection policies.

## A. Sionna-Based Dataset Generation

We construct a 200 m × 200 m campus containing three multi-floor buildings, indoor APs, one outdoor BS, two quasistatic UAV-mounted access nodes, outdoor walkways, and an event area. Indoor structures, material-dependent walls, furniture, vegetation, vehicles, and other outdoor obstacles create blockage and non-line-of-sight conditions. The APs represent private-5G indoor small cells, the BS represents an outdoor campus cell, and the UAVs remain at fixed hovering positions during each simulation episode. Table I summarizes the common simulation and learning parameters.

For each scenario, Sionna RT computes geometry-aware propagation between users and access nodes, while Sionna SYS generates QoS-related quantities such as received power, SINR, BLER, throughput, access-node load, and resource utilization. The simulated users follow short indoor-outdoor trajectories, including normal occupancy, class transition, outdoor event, stress-event, and access-node degradation scenarios. These scenarios capture room/corridor usage, mobility between indoor and outdoor areas, local crowding, heavy-load conditions, and temporary radio or load degradation.

Each log entry is a user-candidate tuple $( u , a , t )$ from $\mathit { { \mathcal { C } } _ { u } ^ { t } } ,$ which contains the current serving node and the top-K feasible measurement candidates. Sionna probes these candidates over the prediction window to obtain future throughput and BLER and derive the QoS-failure labels. In practice, serving-link outcomes are logged directly, while unselected links require probing, dual connectivity, or controlled exploration. Samples remain at their observing access nodes and are locally split into 70%/15%/15% training, validation, and test sets.

We evaluate two naturally non-IID conditions using the same campus, radio configuration, user-density ranges, prediction horizon, and learning settings. The mild non-IID condition mainly contains normal and class-transition traffic, uses service rates of 1/5/10 Mbps, and rarely includes AP outages. The severe non-IID condition uses 2/8/15 Mbps, increases event-, stress-, and outage-related traffic, and introduces higher client heterogeneity through stronger room skew, user churn, probing, and reduced candidate sets.

TABLE I: Default simulation and learning parameters.
<table><tr><td>Parameter Campus area</td><td>Value</td></tr><tr><td>Buildings Floors / APs per building Floor height Access nodes Carrier frequency Access-node transmit powers Access-node antenna gains</td><td>200 m × 200 m 3 multi-floor buildings (A, B, C) A: 2/4, B: 3/9, C: 4/16 4m 29 APs, 1 outdoor BS, 2 UAVs 3.5 GHz AP/BS/UAV: 24/37/30 dBm AP/BS/UAV: 4/12/7 dBi</td></tr><tr><td>Decision-slot duration Prediction window Candidate construction Top-K candidates for mild setting Top-K candidates for severe setting AP-outage scenario share Handover factors ηh, ξh, Tdwell ML model</td><td>1 second 5 seconds Top-K measurement candidates 4 3 Mild: 5%; Severe: 30% 0.08, 0.04, 3 time slots Multi-Layer Perceptron</td></tr><tr><td>Hidden layer Activations Federated rounds Local epochs and batch size Optimizer and learning rate FedProx/FedQoS coefficient (µ)</td><td>2 layers with size (96, 96) ReLU, Sigmoid (output layer) 50 1, 128 Adam with lr=0.001 0.01</td></tr></table>

## B. Baselines and Metrics

The proposed FedQoS algorithm is compared with four learning baselines. Centralized pools all training data at a central server and serves as a centralized learning reference. Local-only trains an independent model at each access node without model aggregation. FedAvg performs standard sample-size-weighted federated averaging, whereas FedProx adds a proximal regularization term during local training to mitigate client drift under heterogeneous data.

We also compare FedQoS with several non-learning accessselection policies. Current serving keeps each user connected to its existing serving node. Strongest SINR selects the candidate with the highest instantaneous SINR. Load-aware RSRP jointly considers received signal power and access-node load. AP-first prioritizes AP candidates whenever available and falls back to a BS or UAV otherwise. Historical QoS selects the candidate with the lowest historical QoS-failure rate in the training logs. Finally, Oracle selects candidates using their realized future QoS outcomes and is included only as a non-implementable performance reference.

For QoS-risk prediction, we report balanced accuracy, recall, and F1-score. Balanced accuracy accounts for the imbalance between successful and failed QoS samples, recall measures the ability to detect imminent QoS failures, and the F1-score summarizes the balance between failure detection and false alarms. Higher values indicate better prediction performance.

For access-selection performance, we report the QoS-failure rate, mean throughput, and mean BLER of the selected links. The QoS-failure rate is the primary system-level metric because the objective is to avoid future QoS violations. A lower failure rate and BLER indicate more reliable access selection, whereas a higher throughput indicates better datarate performance.

## IV. RESULTS AND DISCUSSION

We report results under the mild and severe non-IID conditions described in Section III-A. Learning-based results are averaged over ten random seeds and reported as mean ± standard deviation.

## A. Prediction and Access-Selection Performance

Table II(a) compares the QoS-failure prediction performance. In both settings, all federated methods clearly outperform local-only training, demonstrating that collaboration across access-node clients compensates for the limited and heterogeneous observations available at individual nodes. As expected, the centralized model generally provides the strongest overall reference because it has direct access to the complete training dataset.

In the mild non-IID setting, FedQoS achieves the highest balanced accuracy and recall among the federated methods. Its F1-score remains nearly identical to those of FedAvg and FedProx, indicating that the improved failure sensitivity does not materially degrade the overall classification balance. In particular, FedQoS favors the detection of imminent QoS violations, which is desirable because missed failures can lead directly to poor access-selection decisions. FedProx and FedAvg provide slightly higher displayed F1-score, but the differences are small relative to the variability across random seeds.

Under severe non-IID data, FedQoS achieves the highest mean balanced accuracy, recall, and F1 among the federated methods. Although the numerical differences from FedAvg and FedProx remain modest, FedQoS consistently preserves failure-detection performance under stronger topology-induced heterogeneity, increased user density, and partial AP unavailability. The substantial gap between all federated models and local-only training also confirms that isolated access nodes cannot learn sufficiently general QoS-failure predictors from their own observations.

Overall, FedQoS maintains prediction quality comparable to conventional federated learning under mild non-IID data and provides the strongest failure-oriented predictive performance under the severe non-IID condition.

<table><tr><td colspan="7"></td></tr><tr><td>Method</td><td>BAcc. ↑</td><td>Mild non-IID Recall ↑</td><td>F1 ↑</td><td>BAcc. ↑</td><td>Severe non-IID Recall ↑</td><td>F1 ↑</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Centralized Local-only</td><td> $0 . 9 4 9 3 \pm 0 . 0 0 4 4$   $\overline { { 0 . 9 2 3 4 \pm 0 . 0 0 4 1 } }$ </td><td> $0 . 9 1 7 4 \pm 0 . 0 1 0 0$   $\overline { { 0 . 8 7 3 8 \pm 0 . 0 0 8 9 } }$ </td><td> $0 . 8 9 0 2 \pm 0 . 0 0 2 7$   $\overline { { 0 . 8 4 2 0 \pm 0 . 0 0 4 0 } }$ </td><td> $0 . 9 6 8 8 \pm 0 . 0 0 2 6$   $\overline { { 0 . 9 4 6 6 \pm 0 . 0 0 1 4 } }$ </td><td> $0 . 9 6 1 7 \pm 0 . 0 0 7 1$   $\overline { { 0 . 9 3 4 4 \pm 0 . 0 0 2 8 } }$ </td><td> $0 . 9 4 7 5 \pm 0 . 0 0 2 1$   $\overline { { 0 . 9 1 1 4 \pm 0 . 0 0 1 9 } }$ </td></tr><tr><td>FedAvg</td><td> $0 . 9 4 2 1 \pm 0 . 0 0 3 9$ </td><td> $0 . 9 0 8 9 \pm 0 . 0 1 0 2$ </td><td> $\mathbf { 0 . 8 6 7 5 \pm 0 . 0 0 2 3 }$ </td><td> $0 . 9 6 7 0 \pm 0 . 0 0 2 2$ </td><td> $0 . 9 5 9 9 \pm 0 . 0 0 7 7$ </td><td> $0 . 9 4 4 1 \pm 0 . 0 0 1 2$ </td></tr><tr><td>FedProx</td><td> $0 . 9 4 0 7 \pm 0 . 0 0 3 6$ </td><td> $0 . 9 0 5 6 \pm 0 . 0 0 8 8$ </td><td> $0 . 8 6 7 5 \pm 0 . 0 0 3 0$ </td><td> $0 . 9 6 7 1 \pm 0 . 0 0 2 4$ </td><td> $0 . 9 6 0 6 \pm 0 . 0 0 7 8$ </td><td> $0 . 9 4 4 0 \pm 0 . 0 0 1 2$ </td></tr><tr><td>FedQoS</td><td> $\mathbf { 0 . 9 4 4 0 \pm 0 . 0 0 4 9 }$ </td><td> $\mathbf { 0 . 9 1 3 7 \pm 0 . 0 1 2 5 }$ </td><td> $0 . 8 6 7 2 \pm 0 . 0 0 2 5$ </td><td> $\mathbf { 0 . 9 6 7 9 \pm 0 . 0 0 2 9 }$ </td><td> $\mathbf { 0 . 9 6 1 7 \pm 0 . 0 0 8 5 }$ </td><td> $\mathbf { 0 . 9 4 5 1 \pm 0 . 0 0 1 4 }$ </td></tr><tr><td></td><td></td><td></td><td>(b) Access-selection performance</td><td></td><td></td><td></td></tr><tr><td colspan="7"></td></tr><tr><td>Method</td><td>Failure ↓</td><td>Mild non-IID Throughput (Mbps) ↑</td><td>BLER ↓</td><td>Failure ↓</td><td>Severe non-IID Throughput (Mbps) ↑</td><td>BLER↓</td></tr><tr><td>Current serving</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Strongest SINR</td><td>0.1083 0.0962</td><td>20.0043 20.3100</td><td>0.0459 0.0386</td><td>0.2969 0.2795</td><td>18.6359 19.6464</td><td>0.0418 0.0413</td></tr><tr><td>Load-aware RSRP</td><td>0.0928</td><td>19.6950</td><td>0.0374</td><td>0.2762</td><td>20.6446</td><td>0.0414</td></tr><tr><td>AP-first</td><td>0.0983</td><td>20.5194</td><td>0.0395</td><td>0.2706</td><td>20.7833</td><td>0.0418</td></tr><tr><td>Historical QoS</td><td>0.0905</td><td>20.4140</td><td>0.0363</td><td>0.2668</td><td>21.3871</td><td>0.0365</td></tr><tr><td>Centralized</td><td> $0 . 0 0 2 5 \pm 0 . 0 0 0 5$ </td><td> $1 8 . 0 5 4 5 \pm 0 . 0 7 4 4$ </td><td> $0 . 0 0 0 7 \pm 0 . 0 0 0 2$ </td><td> $0 . 1 3 5 7 \pm 0 . 0 0 0 7$ </td><td> $1 9 . 3 0 9 7 \pm 0 . 0 6 6 9$ </td><td> $0 . 0 0 9 1 \pm 0 . 0 0 0 3$ </td></tr><tr><td>Local-only</td><td> $0 . 0 0 5 5 \pm 0 . 0 0 0 4$ </td><td> $1 7 . 6 7 1 9 \pm 0 . 0 5 0 3$ </td><td> $0 . 0 0 2 0 \pm 0 . 0 0 0 2$ </td><td> $0 . 1 4 4 1 \pm 0 . 0 0 0 9$ </td><td> $1 9 . 1 3 3 6 \pm 0 . 0 3 1 6$ </td><td> $0 . 0 1 0 5 \pm 0 . 0 0 0 3$ </td></tr><tr><td>FedAvg</td><td> $0 . 0 0 3 6 \pm 0 . 0 0 0 5$ </td><td> $\mathbf { 1 7 . 7 4 5 2 \pm 0 . 0 4 4 3 }$ </td><td> $0 . 0 0 1 2 \pm 0 . 0 0 0 2$ </td><td> $0 . 1 3 5 6 \pm 0 . 0 0 0 8$ </td><td> $\mathbf { 1 9 . 2 1 5 0 \pm 0 . 0 3 2 1 }$ </td><td> $0 . 0 0 9 5 \pm 0 . 0 0 0 4$ </td></tr><tr><td>FedProx</td><td> $0 . 0 0 3 3 \pm 0 . 0 0 0 3$ </td><td> $1 7 . 7 0 6 2 \pm 0 . 0 3 8 7$ </td><td> $0 . 0 0 1 0 \pm 0 . 0 0 0 1$ </td><td> $0 . 1 3 5 8 \pm 0 . 0 0 0 3$ </td><td> $1 9 . 1 9 2 0 \pm 0 . 0 2 4 5$ </td><td> $\mathbf { 0 . 0 0 9 5 \pm 0 . 0 0 0 2 }$ </td></tr><tr><td>FedQoS</td><td> $\mathbf { 0 . 0 0 3 1 \pm 0 . 0 0 0 2 }$ </td><td> $1 7 . 6 5 8 6 \pm 0 . 0 4 7 6$ </td><td> $\mathbf { 0 . 0 0 0 9 \pm 0 . 0 0 0 1 }$ </td><td> $\mathbf { 0 . 1 3 5 6 \pm 0 . 0 0 0 3 }$ </td><td> $1 9 . 1 8 1 7 \pm 0 . 0 3 4 6$ </td><td> $0 . 0 0 9 6 \pm 0 . 0 0 0 2$ </td></tr><tr><td>Oracle</td><td>0.0004</td><td> $2 7 . 1 0 8 0$ </td><td>0.0004</td><td> $0 . 1 2 8 9 ^ { }$ </td><td> $2 5 . 0 0 7 5$ </td><td>0.0041</td></tr></table>

TABLE II: Prediction and access-selection performance over ten seeds. Boldface marks the best mean among Local-only, FedAvg, FedProx, and FedQoS, with standard deviation breaking ties at the displayed precision.  
(a) QoS-failure prediction

Table II(b) evaluates the downstream access-selection performance. The conventional policies represent different practical selection principles, including maintaining the current serving node, selecting the strongest SINR, considering both RSRP and load, prioritizing AP connectivity, and exploiting historical QoS. However, these policies rely mainly on instantaneous or past measurements and cannot directly anticipate future QoS violations. Consequently, their failure rates and BLER values remain substantially higher than those of the learning-based policies in both settings.

In the mild non-IID setting, FedQoS achieves the lowest mean QoS-failure rate and BLER among the federated methods. Its failure rate is reduced by approximately 13.9% relative to FedAvg and 6.1% relative to FedProx. The reductions relative to local-only training and historical QoS are approximately 43.6% and 96.6%, respectively. These gains show that the overall FedQoS design improves not only failure prediction but, more importantly, the access decisions derived from the predicted probabilities.

FedQoS obtains these reliability improvements with only a small throughput reduction. Its average throughput is approximately 0.49% below FedAvg and 0.27% below FedProx. This behavior is consistent with the objective of FedQoS: rather than selecting links solely for their instantaneous data rate, the policy favors candidates with a lower predicted risk of future QoS violation. The resulting reduction in failure rate and BLER therefore outweighs the marginal throughput cost.

The severe non-IID setting is substantially more challenging, as indicated by the higher failure rates of all policies. Nevertheless, all learning-based approaches still markedly outperform the conventional heuristics. FedQoS ties FedAvg at the displayed precision for the lowest federated failure rate and remains close to FedAvg and FedProx in both throughput and BLER. Its throughput is less than 0.2% below FedAvg, while the BLER difference is only 10<sup>−4</sup>. Hence, the three federated methods operate at very similar system-level performance in this setting, with FedQoS retaining a reliability-oriented operating point under stronger client heterogeneity.

The oracle achieves the lowest failure rate and BLER and the highest throughput because it selects access links using future QoS outcomes. It is therefore included only as an unattainable reference rather than an implementable competitor. Overall, FedQoS provides the clearest reliability gains under mild non-IID data and remains robust and competitive under the more difficult severe non-IID condition.

## B. Sensitivity to q and λ

We evaluate the sensitivity of FedQoS to the client-size exponent q and the QoS-risk emphasis parameter λ. Since the policy QoS-failure rate is the primary system-level objective, Fig. 2 reports its variation over the evaluated parameter grid.

In the mild non-IID setting, the QoS-failure rate varies only from approximately 0.00302 to 0.00334. The lowest value is obtained near $( q , \lambda ) \ : = \ : ( 0 . 7 5 , 1 . 5 )$ , while a broad region with moderate or large q provides nearly identical performance. The common setting used in the main experiments, $( q , \lambda ) = ( 0 . 7 5 , 0 . 5 )$ , also lies within this low-failure region.

The severe non-IID setting is similarly stable, with the QoSfailure rate remaining within approximately 0.13553–0.13594 across the complete grid. Although the minimum is observed near $( q , \lambda ) = ( 0 . 5 , 4 )$ , several neighboring combinations yield almost indistinguishable results. This indicates a broad performance plateau rather than a sharply localized optimum.

Across both settings, q has a more visible influence than λ, particularly when q is close to zero. Hence, controlling the contribution of clients with highly unequal dataset sizes is more important than aggressively increasing the risk emphasis. Once a reasonable client-size weighting is selected, λ mainly fine-tunes the resulting policy.

![](images/f1ee8e78a6d23ae9b7e5719f06224dccba36ae28cadd7e09487af30bd78fa272.jpg)  
(a) Mild non-IID setting.

![](images/1c015782422a27fc332c6d2b1ef9bfda6c951a41dc773136466aaadbc8b1e000.jpg)  
(b) Severe non-IID setting.  
Fig. 2: Sensitivity of the policy QoS-failure rate to the client-size exponent q and QoS-risk emphasis parameter λ.

Overall, the sensitivity analysis shows that FedQoS remains stable over a broad range of parameter values and that the reported policy gains do not depend on narrowly tuned values of q and λ.

## V. CONCLUSION

This paper presented FedQoS, a federated QoS-risk learning framework for reliable access selection in heterogeneous indoor-outdoor environments. Instead of relying only on instantaneous signal strength, FedQoS learns the future QoSfailure risk of candidate access links from distributed network observations collected at APs, UAVs, and the BS. The learned risk scores are then used to support reliability-aware access decisions without centralizing user-level data. Simulation results on physics-based synthetic campus datasets show that learning-based access selection substantially reduces QoS failures and BLER compared with conventional signal-based and historical-QoS heuristics. FedQoS achieves near-centralized prediction performance and remains effective under both mild and severe non-IID conditions. Its stable performance across a broad range of aggregation parameters further demonstrates its potential for reliable access selection in dynamic heterogeneous networks.

## ACKNOWLEDGMENT

This work was supported in part by the Luxembourg National Research Fund (FNR) through the CHIST-ERA SHIELD project under Grant INTER/CHIST24/19023763/SHIELD, and in part by the FNR AFR program through the FULFIL project under Grant 2000141.

## REFERENCES

[1] D. Liu, L. Wang, Y. Chen, M. Elkashlan, K.-K. Wong, R. Schober, and L. Hanzo, “User association in 5g networks: A survey and an outlook,” IEEE Communications Surveys & Tutorials, vol. 18, no. 2, pp. 1018– 1044, 2016.

[2] Y. Kim, J. Jang, and H. J. Yang, “Distributed resource allocation and user association for max-min fairness in hetnets,” IEEE Transactions on Vehicular Technology, vol. 73, no. 2, pp. 2983–2988, 2023.

[3] Q. Ye, B. Rong, Y. Chen, M. Al-Shalash, C. Caramanis, and J. G. Andrews, “User association for load balancing in heterogeneous cellular networks,” IEEE Transactions on Wireless Communications, vol. 12, no. 6, pp. 2706–2716, 2013.

[4] T. Zhou, Y. Huang, W. Huang, S. Li, Y. Sun, and L. Yang, “Qos-aware user association for load balancing in heterogeneous cellular networks,” in 2014 IEEE 80th Vehicular Technology Conference (VTC2014-Fall). IEEE, 2014, pp. 1–5.

[5] F. Kong, X. Sun, V. C. Leung, and H. Zhu, “Delay-optimal biased user association in heterogeneous networks,” IEEE Transactions on Vehicular Technology, vol. 66, no. 8, pp. 7360–7371, 2017.

[6] Y. Cao, S.-Y. Lien, Y.-C. Liang, K.-C. Chen, and X. Shen, “User access control in open radio access networks: A federated deep reinforcement learning approach,” IEEE Transactions on Wireless Communications, vol. 21, no. 6, pp. 3721–3736, 2021.

[7] P. Tehrani, F. Restuccia, and M. Levorato, “Federated deep reinforcement learning for the distributed control of nextg wireless networks,” in 2021 IEEE international symposium on dynamic spectrum access networks (DySPAN). IEEE, 2021, pp. 248–253.

[8] N. Zhao, Y.-C. Liang, D. Niyato, Y. Pei, M. Wu, and Y. Jiang, “Deep reinforcement learning for user association and resource allocation in heterogeneous cellular networks,” IEEE Transactions on Wireless Communications, vol. 18, no. 11, pp. 5141–5152, 2019.

[9] W. Y. B. Lim, N. C. Luong, D. T. Hoang, Y. Jiao, Y.-C. Liang, Q. Yang, D. Niyato, and C. Miao, “Federated learning in mobile edge networks: A comprehensive survey,” IEEE communications surveys & tutorials, vol. 22, no. 3, pp. 2031–2063, 2020.

[10] D. Bega, M. Gramaglia, M. Fiore, A. Banchs, and X. Costa-Perez, “Deepcog: Optimizing resource provisioning in network slicing with ai-based capacity forecasting,” IEEE Journal on Selected Areas in Communications, vol. 38, no. 2, pp. 361–376, 2019.

[11] M. S. Mollel, A. I. Abubakar, M. Ozturk, S. F. Kaijage, M. Kisangiri, S. Hussain, M. A. Imran, and Q. H. Abbasi, “A survey of machine learning applications to handover management in 5g and beyond,” IEEE Access, vol. 9, pp. 45 770–45 802, 2021.

[12] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” Proceedings of Machine learning and systems, vol. 2, pp. 429–450, 2020.

[13] J. Hoydis, S. Cammerer, F. A¨ıt Aoudia, A. Vem, N. Binder, G. Marcus, and A. Keller, “Sionna: An open-source library for next-generation physical layer research,” arXiv preprint arXiv:2203.11854, 2022.

[14] J. Hoydis, F. A¨ıt Aoudia, S. Cammerer, F. Euchner, M. Nimier-David, S. ten Brink, and A. Keller, “Sionna RT: Differentiable ray tracing for radio propagation modeling,” arXiv preprint arXiv:2303.11103, 2023.