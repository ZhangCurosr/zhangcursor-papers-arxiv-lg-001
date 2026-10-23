# Adaptive Deadline- and Energy-Aware Caching Algorithms in Dynamic Multi-UAV Edge Networks

Jian Kim<sup>a</sup>, Ki-Il Kim<sup>a,∗</sup>

<sup>a</sup>Department of Computer Science and Engineering, Chungnam National University, Daejeon, Korea

## A R T I C L E I N F O

Keywords:   
Unmanned aerial vehicles   
Edge caching   
Dynamic content demand   
Deadline-aware caching   
Energy-aware caching   
Deep reinforcement learning

## A BS T RA C T

Content caching via unmanned aerial vehicles (UAVs) becomes one of the solution to maintain content delivery when terrestrial infrastructure is unavailable or disrupted. However, existing caching approaches often focus primarily on cache-hit performance or average delivery delay, without jointly addressing spatially heterogeneous and time-varying demand, content-specific deadlines, and uneven communication-energy consumption across UAVs. To address these challenges, we formulate a dynamic deadline- and energy-aware cache placement and replacement problem that jointly considers deadline misses, UAV-side communication energy, and inter-UAV energy imbalance. To achieve this, we first develop deadline- and energy-aware UBAC (DE-UBAC), an interpretable hierarchical rule-based realization that incorporates deadline-aware prioritization, cooperative placement, energyaware coordination, and cache-update cost. However, predefined priorities and thresholds limit its adaptability under dynamically coupled network conditions. We therefore propose GRU-assisted Deadline- and Energy-aware Proximal Policy Optimization (GRU-DEPPO), which combines local and global GRU-based temporal representations with a centralized PPO actor–critic for adaptive cache placement and replacement. Simulation results show that GRU-DEPPO achieves the lowest deadlinemiss rate and total UAV-side communication energy among the considered schemes. Compared with DE-UBAC, GRU-DEPPO reduces the deadline-miss rate by approximately 61% and inter-UAV energy imbalance by approximately 40%.

## 1. Introduction

Recent advances in the computational and sensing capabilities of end devices, coupled with the rapid proliferation of mobile and Internet of Things (IoT) devices, have led to a substantial increase in the volume of data generated at the network edge [1, 2]. At the same time, emerging applications have imposed increasingly diverse and stringent requirements in terms of latency, reliability, and bandwidth. Conventional cloud computing, which provides largescale computing resources through centralized data centers, faces several limitations in satisfying these requirements: (i) communication latency caused by geographical distance and the traversal of multiple network segments, (ii) latency variations and jitter resulting from network congestion, (iii) increasing backhaul trafic, and (iv) degraded service availability under disruptions in wide-area network connectivity. To address these limitations, edge computing has attracted considerable attention as a distributed computing paradigm that places computing and storage resources closer to end devices and data sources [3, 4]. By processing data at or near the network edge, edge computing can reduce service latency and alleviate the trafic burden on backhaul networks.

In particular, unmanned aerial vehicle (UAV)-assisted edge computing has attracted significant attention because it enables communication, storage, and computing resources to be rapidly and flexibly deployed in environments where fixed base stations and edge infrastructure are unavailable, insuficient, or unable to respond promptly to abrupt changes in service demand [5–7]. Owing to their controllable mobility and adjustable altitude, UAVs can be rapidly deployed and repositioned to increase the probability of establishing line-of-sight (LoS) links with ground devices and dynamically adjust communication coverage. They can also accommodate temporary surges in trafic demand and provide temporary network services in areas where terrestrial communication infrastructure is sparse, unavailable, or damaged. These characteristics make UAV-assisted edge computing particularly useful in remote environments, such as mountainous and maritime regions; temporarily crowded venues, such as concerts and sporting events; and disaster-afected areas where terrestrial base stations or wired backhaul links have been damaged by earthquakes or wildfires [8].

Despite these advantages, UAV-assisted networks face several inherent limitations, including energy constraints imposed by limited battery capacity, restricted onboard computing and storage resources, and limited backhaul capacity between UAVs and terrestrial networks [9]. In particular, forwarding user requests to a remote content server or central cloud through a macro base station (MBS) may increase backhaul trafic and content delivery latency. The associated backhaul transmissions may also consume substantial communication energy, thereby accelerating battery depletion and shortening the operational lifetime of UAVs. A promising approach to reducing backhaul dependence, content delivery latency, and communication energy consumption is to equip UAVs with storage and employ them as aerial cache nodes that proactively store selected content [10]. When the requested content is available in the local cache of the serving UAV, the UAV can deliver it directly to the user without accessing the MBS or retrieving the content from a remote server. If the content is not cached locally but is available at a neighboring UAV, it can instead be obtained through cooperative content sharing, further reducing reliance on the terrestrial backhaul. Such local and cooperative cache hits can reduce content delivery latency, backhaul utilization, communication energy consumption, and overall network resource usage.

However, because UAVs have limited storage capacity, the entire content catalog cannot be cached. Moreover, content-demand and user-mobility patterns may vary across diferent regions and over time, while diferent content types may have heterogeneous delivery deadlines. A cache miss may therefore require content retrieval through the terrestrial backhaul, incurring additional delay and communication energy consumption and potentially causing the request to exceed its deadline [11, 12]. Cache replacement also entails a non-negligible cost because fetching and storing new content requires additional data transmission and communication energy. Furthermore, uneven service loads across UAVs may result in imbalanced communication-energy consumption. Consequently, maximizing the cache hit ratio or selecting content solely according to long-term popularity may be insuficient in dynamic multi-UAV environments [13–15]. An efective caching strategy should therefore adapt cache placement to changes in content demand and network conditions while jointly reducing the deadline-miss rate, the communication energy consumed by request delivery and cache updates, and inter-UAV energy imbalance [16]. These improvements can also reduce dependence on MBS-assisted content retrieval.

To address these challenges, we first develop an interpretable deadline- and energy-aware rule-based caching design and then propose GRU-assisted Deadline- and Energyaware PPO (GRU-DEPPO) to overcome the limitations of predefined cache-decision rules under dynamic multi-UAV conditions. GRU-DEPPO periodically determines cache placement and replacement decisions by jointly considering content-specific delivery deadlines, the communication energy incurred by request delivery and cache updates, and the distribution of communication energy consumption across UAVs. By using local and global GRUs to capture temporal patterns from UAV-local and network-wide observations, the proposed scheme adapts to spatially heterogeneous and time-varying network conditions while reducing deadline misses, communication energy consumption, and inter-UAV energy imbalance.

The main contributions of this study are as follows:

• Dynamic deadline- and energy-aware multi-UAV caching formulation: We formulate a dynamic cache placement and replacement problem for multi-UAV edge networks with spatially heterogeneous and timevarying content demand and content-specific delivery deadlines. The formulation jointly considers the deadline-miss rate, UAV-side communication energy incurred by request delivery and cache updates, and inter-UAV communication-energy imbalance, subject to cache-capacity and cache-update feasibility constraints.

• Deadline- and energy-aware cache decision design: We develop DE-UBAC as an interpretable hierarchical realization of the formulated caching problem. It extends user-behavior-aware caching with deadlinemiss-aware prioritization, persistent-miss feedback, deadline-feasible cooperative placement, energy-aware cache coordination, and cache-update-cost-aware replacement. By assigning these factors distinct roles in the cache-decision process, DE-UBAC provides a structured rule-based realization of the proposed problem and reveals the limitations of predefined decision rules under dynamic multi-UAV conditions.

• GRU-assisted adaptive caching policy: To overcome the limitations of the above rule-based design, we develop GRU-DEPPO as the learning-based solution to adaptively optimize cache placement and replacement under dynamically coupled network conditions. A shared local GRU captures UAV-specific temporal evolution, while a global GRU encodes network-wide temporal context at the MBS. Their representations are fused within a centralized PPO actor–critic that jointly learns feasible cache updates from deadline, demand, connectivity, and energy conditions, avoiding the fixed priorities and thresholds required by the rule-based design.

The remainder of this paper is organized as follows. Section 2 reviews related work on UAV-assisted edge networks and edge caching. Section 3 presents the system model and formulates the deadline- and energy-aware cache placement and replacement problem. Section 4 first develops a deadline- and energy-aware rule-based extension of UBAC and discusses its limitations under dynamic network conditions, and then presents the proposed GRU-DEPPO scheme for learning adaptive cache placement and replacement decisions. Section 5 evaluates the proposed approaches under various experimental settings and compares them with baseline schemes. Finally, Section 6 concludes the paper and discusses directions for future research.

## 2. Related Work

This section reviews existing approaches for serving diverse user requests in multi-UAV networks; their key characteristics are summarized in Table 1.

Traditional cache replacement policies, such as Least Frequently Used (LFU) and Least Recently Used (LRU), make cache decisions using simple frequency- or recencybased statistics and therefore incur low computational overhead. However, their fixed decision rules do not explicitly account for dynamic network conditions, inter-UAV cooperation, content-specific deadlines, or the communication and cache-update energy costs. Several studies have incorporated additional network information into cache placement decisions. The work in [17] jointly optimized caching probabilities and V2V transmission power to improve energy eficiency in a UAV-assisted IoV network, whereas [18] jointly optimized UAV deployment, cache placement, and user association to improve user QoE. The scheme in [19] further incorporated localized request patterns and predicted user mobility into proactive cache updates. Although these approaches exploit richer network and user information than conventional replacement policies, they rely on predefined optimization or placement rules and do not jointly consider cache-update energy, inter-UAV energy imbalance, and content-specific deadline misses.

Comparison of representative caching approaches for multi-UAV edge networks.
<table><tr><td>Study</td><td>Type</td><td>Multi- UAV</td><td>Local awareness</td><td>Inter-UAV cooperation</td><td>Communication energy consumption</td><td>Inter-UAV energy imbalance</td><td>Content-specific deadline</td><td>Realistic user pattern</td></tr><tr><td>[17-19]</td><td>Utility- and rule-based</td><td> $\triangle$ </td><td> $\triangle$ </td><td> $\triangle$ </td><td> $\triangle$ </td><td>X</td><td>×</td><td> $\triangle$ </td></tr><tr><td>[14, 20, 21]</td><td>Game-theoretic</td><td> $\triangle$ </td><td>×</td><td> $\triangle$ </td><td> $\triangle$ </td><td>X</td><td>×</td><td>△</td></tr><tr><td>[22-24]</td><td>DRL-based</td><td> $\triangle$ </td><td>×</td><td>X</td><td> $\triangle$ </td><td>X</td><td>X</td><td>△</td></tr><tr><td>[25]</td><td>DL-assisted DRL</td><td>O</td><td> $\triangle$ </td><td>X</td><td>X</td><td>X</td><td>×</td><td>O</td></tr><tr><td>GRU-DEPPO</td><td>DL-assisted DRL</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td><td>O</td></tr></table>

○: supported; $\triangle \colon$ partially supported; ×: not supported.  
For grouped studies, $\triangle$ indicates that the corresponding feature is supported only by a subset of the included studies or is considered in a limited manner.

Game-theoretic approaches have also been used to coordinate caching and resource allocation in UAV-assisted networks. The study in [14] modeled cooperative content transmission among UAVs using a coalition formation game, while [20] employed a Stackelberg–Bertrand game to jointly consider content caching and service economics. The work in [21] considered content caching, service placement, and task ofloading using Gibbs sampling and a matching game. These approaches provide structured mechanisms for coordinating multiple network entities, but they generally rely on predefined utility or preference models and do not learn temporal demand dynamics from historical observations. Moreover, they do not jointly address cache-update energy, inter-UAV energy imbalance, and request-level deadline misses under content-specific deadlines.

Learning-based approaches have been investigated to improve adaptability in dynamic UAV networks. The work in [22] combined diferential evolution for content caching with multi-agent PPO for UAV trajectory optimization. In [23], a dueling DQN jointly determined UAV positions and cache placement using user-location and request information, whereas [24] employed dual-clip PPO to jointly optimize caching, trajectory, and transmission power in a single-UAV setting. More recently, the work in [25] used an Informer model to predict user request and mobility patterns and incorporated these predictions into a PPO-based deployment framework. These learning-based approaches improve adaptability to dynamic user and network conditions, but their objectives primarily focus on average delay,

![](images/b6019ba5e09980246de8fd369eabf7a19582cfe6f1aeea96c68827e4b1c1ee61.jpg)  
Figure 1: System model of the considered UAV network

QoE, trajectory, or transmission power. They do not jointly model the communication energy consumed by request delivery and cache updates, inter-UAV energy imbalance, and content-specific request deadline misses. Furthermore, existing learning-based schemes do not directly encode recent UAV-local network states to capture temporal dynamics for adaptive cache-update decisions.

In summary, existing caching schemes typically address only a subset of local demand awareness, dynamic demand adaptation, cooperative caching, communication energy, and heterogeneous delivery requirements. In contrast, this study makes periodic cache-update decisions by jointly considering content-specific deadline misses at the request level, communication energy from both request delivery and cache updates, and inter-UAV energy imbalance under timevarying demand.

## 3. System Model and Problem Formulation

In the considered scenario illustrated in Fig. 1, the local base station is unavailable due to a failure, and a remote MBS stores the global content pool and provides backhaul access to cache-enabled UAVs. Users are divided into groups, with each group characterized by group-specific mobility patterns and content-request distributions that are shared by its users. Each user associates with the nearest UAV having a feasible access link. Requests may be served by the associated UAV, a neighboring UAV, or the MBS, depending on content availability and link feasibility [26]. The selected path determines the request delay and communication energy consumption.

Users move within the service area and generate requests according to group-level preferences and request-generation probabilities. Each generated request follows a strict sourceselection priority. For a request associated with UAV $n ,$ the local cache is checked first. If the requested item is not available locally, UAV � searches for feasible cooperative paths to UAVs caching the item and selects the path with the smallest end-to-end transmission delay. Only when no feasible cooperative cache path exists is the item retrieved from the MBS through the feasible direct or relayed backhaul path with the smallest end-to-end transmission delay. If no feasible delivery path is available owing to link outages or insuficient achievable rates, the corresponding request is dropped and excluded from the delivered-request set.

Cache placement is updated at predefined intervals based on the observed network state. Newly cached content is transferred either directly from the MBS to the target UAV or through one or more UAV relay hops. To model nonstationarity, a change point is randomly selected from the latter half of each episode, at which both the mobility context and local request distributions shift.

Let $\mathcal { N }$ , U, C, T, and $\mathcal { R } ^ { \mathrm { a l l } }$ denote the UAV, user, content, content-type, and episode-level request sets, respectively. The discrete simulation-time index and cache-decision interval duration are denoted by � and $\Delta _ { c } .$ , respectively. For each request $k \in \mathcal { R } ^ { \mathrm { a l l } }$ , let $\Pi _ { k }$ denote its selected delivery path, with $\Pi _ { k } \ = \ \emptyset$ when no feasible path is available. The set of successfully delivered requests is therefore $\mathcal { R } ^ { \mathrm { s r v } } = \{ k \in$ $\mathcal { R } ^ { \mathrm { a l l } } : \Pi _ { k } \neq \emptyset \}$

## 3.1. Transmission Model

We use large-scale propagation models for the User– UAV, inter-UAV, and MBS–UAV links [27–30]. The User– UAV link additionally accounts for elevation-dependent LoS/NLoS attenuation. The achievable rate of each link is determined by its available bandwidth and SINR. A link is considered infeasible if its SINR falls below $\Gamma _ { \mathrm { m i n } }$ or its achievable rate falls below $R _ { \mathrm { m i n } }$ . Cache updates use either a direct MBS–UAV link or an MBS–UAV link followed by one or more UAV–UAV relay links.

## 3.1.1. User–UAV Access Link

The link between UAV � and user � is modeled as an air-to-ground wireless link with three-dimensional distance $d _ { n , u }$ and elevation angle $\theta _ { n , u } .$

The probability of an LoS connection is modeled as an elevation-angle-dependent function [28]:

$$
\begin{array} { c } { { P _ { n , u } ^ { \mathrm { L o S } } = \displaystyle \frac { 1 } { 1 + X \exp [ - Y ( \theta _ { n , u } - X ) ] } , } } \\ { { P _ { n , u } ^ { \mathrm { N L o S } } = 1 - P _ { n , u } ^ { \mathrm { L o S } } } } \end{array}\tag{1}
$$

Here, � and � are environment-dependent parameters. The average path loss is

$$
\begin{array} { c } { { P L _ { n , u } ^ { \mathrm { F S } } = 2 0 \log _ { 1 0 } \left( \displaystyle \frac { 4 \pi f _ { u } d _ { n , u } } { c } \right) , } } \\ { { P L _ { n , u } = P L _ { n , u } ^ { \mathrm { F S } } + P _ { n , u } ^ { \mathrm { L o S } } \eta _ { \mathrm { L o S } } + P _ { n , u } ^ { \mathrm { N L o S } } \eta _ { \mathrm { N L o S } } } } \end{array}\tag{2}
$$

where $f _ { u }$ is the carrier frequency, � is the speed of light, and $\eta _ { \mathrm { L o S } }$ and $\eta _ { \mathrm { N L o S } }$ are the additional LoS and NLoS losses. The SINR, rate, and transmission delay are

$$
\begin{array} { c } { { \Gamma _ { n , u } = \displaystyle \frac { P _ { n } ^ { \mathrm { t x } } 1 0 ^ { - P L _ { n , u } / 1 0 } } { N _ { 0 } B _ { n , u } ^ { \mathrm { e f f } } + I _ { n , u } } , } } \\ { { R _ { n , u } = B _ { n , u } ^ { \mathrm { e f f } } \log _ { 2 } ( 1 + \Gamma _ { n , u } ) , } } \\ { { D _ { n , u } ^ { \mathrm { t x } } = \displaystyle \frac { S } { R _ { n , u } } } } \end{array}\tag{3}
$$

Here, $P _ { n } ^ { \mathrm { t x } } , \ N _ { 0 } , \ B _ { n , u } ^ { \mathrm { e f f } }$ , and $I _ { n , u }$ denote the transmit power, noise power spectral density, efective bandwidth, and aggregate co-channel interference, respectively. The payload size is denoted by �.

## 3.1.2. UAV–UAV Relay Link

The relay link between UAVs � and � is modeled using a distance-dependent free-space path-loss model:

$$
P L _ { n , m } = 2 0 \log _ { 1 0 } \left( \frac { 4 \pi f _ { \mathrm { U A V } } d _ { n , m } } { c } \right)\tag{4}
$$

where $d _ { n , m }$ denotes the distance between UAVs � and $m ,$ and $f _ { \mathrm { U A V } }$ is the carrier frequency of the inter-UAV link. The corresponding SINR, achievable transmission rate, and transmission delay are calculated using (3), with the inter-UAV transmit power, bandwidth, and interference terms. When multi-hop relaying is employed, the delay and energy consumption of the individual hops are accumulated over the complete relay path.

## 3.1.3. MBS–UAV Backhaul Link

The backhaul link between the MBS, indexed by 0, and UAV � is modeled using a distance-dependent free-space path-loss model with an additional attenuation term:

$$
P L _ { 0 , n } = 2 0 \log _ { 1 0 } \left( \frac { 4 \pi f _ { \mathrm { M B S } } d _ { 0 , n } } { c } \right) + \eta _ { \mathrm { M B S } }\tag{5}
$$

where $d _ { 0 , n }$ denotes the distance between the MBS and UAV $n , \mathrm { \Delta } f _ { \mathrm { M B S } }$ is the carrier frequency of the MBS–UAV link, and $\eta _ { \mathrm { M B S } }$ denotes the additional propagation attenuation. The corresponding SINR, achievable transmission rate, and transmission delay are calculated using (3), with the MBS transmit power, backhaul bandwidth, and interference terms.

## 3.2. Content Request Model

This subsection defines type-specific deadlines and timevarying, user-dependent content requests.

## 3.2.1. Content Characteristics

Each content item � has size $S _ { i } ,$ type $\tau _ { i } ,$ and delivery deadline $\delta _ { i } .$ For modeling tractability and to isolate the efects of caching decisions, all items are assumed to have an identical size �, i.e., $S _ { i } ~ = ~ S$ for all $i \in \textit { c }$ . The content catalog is partitioned into type-specific subsets, each associated with a diferent deadline:

$$
\delta _ { i } = \delta _ { \tau _ { i } }\tag{6}
$$

Thus, requests for diferent content types may have diferent delivery-time requirements.

## 3.2.2. Dynamic Request Generation

At each time step, requests follow a Zipf-like global popularity distribution [31–33]:

$$
p _ { i } ^ { \mathrm { g l o b a l } } = \frac { q _ { i } ^ { - \alpha } } { \sum _ { j \in \cal C } q _ { j } ^ { - \alpha } }\tag{7}
$$

where  denotes the content catalog, $q _ { i }$ is the popularity rank of content �, and � is the Zipf parameter.

Motivated by group-level heterogeneous demand modeling in [19], we perturb the global distribution according to group-specific preferences, spatial regions, and time-varying demand states. For group � at time �, the resulting local popularity is defined as

$$
p _ { i , g , t } = \frac { p _ { i } ^ { \mathrm { g l o b a l } } \exp { \left( \epsilon _ { i , g , t } \right) } } { \sum _ { j \in { \cal C } } p _ { j } ^ { \mathrm { g l o b a l } } \exp { \left( \epsilon _ { j , g , t } \right) } }\tag{8}
$$

where $\epsilon _ { i , g , t }$ is a group- and time-dependent content-specific perturbation.

Let $A _ { u , t }$ indicate whether user � generates a request at time �:

$$
\begin{array} { r l } { A _ { u , t } \sim \mathrm { B e r n o u l l i } ( \lambda _ { u , t } ) , } & { } \\ { J _ { u , t } \sim \mathrm { C a t e g o r i c a l } \left( \left\{ p _ { i , g ( u ) , t } \right\} _ { i \in C } \right) } & { \mathrm { i f } \ A _ { u , t } = 1 } \end{array}\tag{9}
$$

where $\lambda _ { u , t }$ is the request-generation probability and $J _ { u , t }$ is the requested content.

To capture nonstationarity, a change point is selected from the second half of each episode, at which the local distribution shifts toward a newly sampled demand state:

$$
p _ { i , g , t } ^ { \mathrm { n e w } } = ( 1 - \beta _ { \mathrm { s h i f t } } ) p _ { i , g , t } ^ { \mathrm { o l d } } + \beta _ { \mathrm { s h i f t } } \widetilde { p } _ { i , g , t }\tag{10}
$$

where $\widetilde { p } _ { i , g , t }$ is the newly sampled distribution and $\beta _ { \mathrm { s h i f t } } \ \in$ [0, 1] controls the shift magnitude.

## 3.3. Energy Consumption Model

We consider the UAV-side communication energy associated with content delivery and cache updates [34]. For a transfer of content item � over link $\ell ,$ let $v _ { \ell } ^ { \mathrm { t x } }$ and $v _ { \boldsymbol { \ell } } ^ { \mathrm { r x } }$ denote the transmitting and receiving nodes, respectively. The UAVside communication energy incurred over link $\ell$ is defined as

$$
E _ { \ell } = \big ( \mathbb { 1 } \{ v _ { \ell } ^ { \mathrm { t x } } \in \mathcal { N } \} P _ { \ell } ^ { \mathrm { t x } } + \mathbb { 1 } \{ v _ { \ell } ^ { \mathrm { r x } } \in \mathcal { N } \} P _ { \ell } ^ { \mathrm { r x } } \big ) \frac { S _ { i } } { R _ { \ell } }\tag{11}
$$

where $P _ { \ell } ^ { \mathrm { t x } }$ and $P _ { \ell } ^ { \mathrm { r x } }$ denote the transmit-side and receive-side power draws, respectively, and $R _ { \ell }$ is the link rate. Thus, only the energy consumed by UAVs is included, while the communication energy consumed by the MBS and users is excluded.

Let ${ \mathcal { R } } ^ { \operatorname { s r v } }$ denote the set of successfully delivered requests and let  denote the link-level transmissions used for cache updates. The total UAV-side communication energy is

$$
E ^ { \mathrm { t o t a l } } = \sum _ { k \in \mathcal { R } ^ { \mathrm { s r v } } } \sum _ { \ell \in \Pi _ { k } } E _ { \ell } + \sum _ { q \in \mathcal { Q } } E _ { q }\tag{12}
$$

The first term captures the UAV-side communication energy incurred by request delivery, including multi-hop relay transmissions, and the second captures the UAV-side communication energy incurred by cache updates. Cache evictions are local and incur negligible communication energy.

Let $E _ { n }$ denote the communication energy attributed to UAV � during the episode. For each link-level transmission, the transmit-side energy is attributed to UAV � when it acts as the transmitter, whereas the receive-side energy is attributed to UAV � when it acts as the receiver. Accordingly,

$$
E ^ { \mathrm { t o t a l } } = \sum _ { n \in \mathcal { N } } E _ { n } .\tag{13}
$$

The resulting inter-UAV communication-energy imbalance is evaluated using the coeficient of variation:

$$
{ \cal T } _ { E } = \frac { \sqrt { \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left( E _ { n } - \overline { { E } } \right) ^ { 2 } } } { \overline { { E } } + \epsilon _ { E } } , \qquad \overline { { E } } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } E _ { n }\tag{14}
$$

where � is the number of UAVs and $\epsilon _ { E }$ is a small positive constant. A lower value of $\scriptstyle { \tau _ { E } }$ indicates more balanced communication energy consumption across UAVs.

## 3.4. Deadline Metric, Cache Constraints, and Problem Formulation

This subsection defines the deadline metric and cache constraints and formulates a joint optimization problem. Using the models introduced above, the objective is tojointly reduce the deadline-miss rate, total communication energy, and inter-UAV energy imbalance.

## 3.4.1. Deadline-Miss Metric

For request $k \in \mathcal { R } ^ { \mathrm { a l l } }$ , let $i _ { k }$ denote the requested content. Its deadline is $\delta _ { k } = \delta _ { \tau _ { i _ { k } } }$ . For a request with a feasible path, its end-to-end delay is

$$
D _ { k } = \sum _ { \ell \in \Pi _ { k } } D _ { \ell } ^ { \mathrm { t x } }\tag{15}
$$

Here, $D _ { \ell } ^ { \mathrm { t x } }$ is obtained from the corresponding link model. Resource contention is reflected through the efective bandwidth and interference terms in the link model; an additional queuing delay is not explicitly modeled.

For each delivered request, a deadline miss occurs when its end-to-end delay exceeds its content-specific deadline:

$$
\begin{array} { r l } & { M _ { k } = \mathbb { 1 } \{ D _ { k } > \delta _ { k } \} , \qquad k \in \mathcal { R } ^ { \mathrm { s r v } } , } \\ & { } \\ & { \mathrm { D M R } = \frac { \displaystyle \sum _ { k \in \mathcal { R } ^ { \mathrm { s r v } } } M _ { k } } { \displaystyle \operatorname* { m a x } \left( 1 , | \mathcal { R } ^ { \mathrm { s r v } } | \right) } } \end{array}\tag{16}
$$

Here, DMR is the deadline-miss rate among delivered requests.

## 3.4.2. Cache Constraints

Let $x _ { n , i } ( t ) \in \{ 0 , 1 \}$ indicate whether UAV � caches item � at time �. The cache state is constrained by

$$
\sum _ { i \in C } x _ { n , i } ( t ) \leq K _ { n } , \qquad \sum _ { i \in C } { \widehat { x } } _ { n , i } ( t ) \in \{ 0 , 1 \} , \qquad \forall n \in \mathcal { N }\tag{17}
$$

where $K _ { n }$ is the cache capacity in items. At each cacheupdate instant, the controller selects a target cache state ${ \widehat x } _ { n , i } ( t )$ that satisfies this capacity constraint.

Let $C _ { n } ^ { + } ( t ) = \{ i \ : \ \widehat { x } _ { n , i } ( t ) = 1 , x _ { n , i } ( t ) = 0 \}$ denote the newly inserted items. The target state is not activated immediately: the previous cache state remains available while the required backhaul transfers are performed. Let $\varphi _ { n } ( t )$ denote the selected cache-update path from the MBS, indexed by 0, to UAV �. A direct update contains only the MBS–UAV link, whereas a relayed update contains that link followed by one or more UAV–UAV links. Assuming sequential content additions, their duration must satisfy

$$
\begin{array} { r } { D _ { n , t } ^ { \mathrm { c a c h e } } = \displaystyle \sum _ { i \in { \cal C } _ { n } ^ { + } ( t ) } \sum _ { \ell \in \varphi _ { n } ( t ) } \frac { S _ { i } } { R _ { \ell } ( t ) } \le \Delta _ { c } , } \\ { \forall n \in \mathcal { N } } \end{array}\tag{18}
$$

where $R _ { \ell } ( t )$ is the rate of link � at the update instant. After this delay, the target state becomes active and the corresponding link-level transmissions are included in . A target cache state that violates (18) is treated as an invalid action and excluded from the controller’s feasible action set. Cache evictions are local operations.

## 3.4.3. Problem Formulation

Let � denote a dynamic caching policy that maps the observed network state to target cache states. We formulate the dynamic caching problem as the following multi-objective optimization:

$$
\operatorname* { m i n } _ { \mu } \mathbb { E } \left[ \left( \mathrm { D M R } , E ^ { \mathrm { t o t a l } } , T _ { E } \right) \right]\tag{19}
$$

subject to (17) and (18). The objective jointly minimizes the deadline-miss rate, total communication energy, and inter-UAV energy imbalance.

Problem (19) is a stochastic combinatorial optimization problem. Even when user locations, requests, and link conditions are fixed, the formulation contains a static cooperative cache-placement subproblem with binary placement decisions and per-UAV cache-capacity constraints. Cooperative cache-placement problems of this form have been shown to be NP-hard through a reduction from the 2-Disjoint Set Covers problem [35]. Therefore, Problem (19), which further incorporates time-varying demand, user mobility, contentspecific deadlines, communication energy consumption, and inter-UAV energy imbalance, is also NP-hard.

The number of feasible cache placements grows combinatorially with the network size, making exact ofline optimization computationally prohibitive for large-scale networks. Moreover, repeatedly solving the problem at each cache-decision interval is impractical under dynamically changing requests, mobility, and link conditions. We therefore first investigate a structured rule-based realization of the formulated problem and then develop a learning-based policy to overcome the limitations of predefined cachedecision rules.

## 4. Adaptive Deadline- and Energy-Aware Cache Algorithms

In this section, we first investigate how the deadlineand energy-aware cache placement and replacement problem formulated in Section 3 can be addressed through a rule-based extension of the local-demand-aware UBAC scheme [19]. The resulting design incorporates contentspecific deadlines, cooperative content accessibility, UAV communication-energy conditions, cache redundancy, and cache-update cost into cache placement and replacement decisions. We then discuss the limitations of relying on predefined rules under dynamically changing network conditions and formulate the cache placement and replacement process as a Markov decision process (MDP). To capture the temporal evolution of spatially heterogeneous local and network-wide conditions, GRU-DEPPO employs local and global GRU-based representations, which are subsequently provided to a centralized PPO actor–critic for joint cacheupdate decision making. The main notation used throughout this section is summarized in Table 2, and the overall GRU-DEPPO procedure is illustrated in Fig. 2.

![](images/13dbc30a2b333f935addd23d2e319804cb8afdc398313adb7a575af06f807652.jpg)  
Figure 2: The GRU-DEPPO scheme

Table 2  
Main notation used for DE-UBAC, the MDP, and GRU-DEPPO
<table><tr><td>Symbol</td><td>Description</td><td>Symbol</td><td>Description</td></tr><tr><td> $\mathcal { N }$ </td><td>Set of UAVs</td><td>C</td><td>Set of content items</td></tr><tr><td> $\xi$ </td><td>Cache-decision epoch</td><td> $\Delta _ { c }$ </td><td>Cache-decision interval</td></tr><tr><td> $C _ { k } ^ { \mathrm { D E } }$ </td><td>Deadline-aware counting increment for request k</td><td> $\bar { H } ^ { \mathrm { D E } }$  n,i,ξ</td><td>Deadline-aware history count for UAV-content pair (n, i)</td></tr><tr><td> $\smash { \widetilde { H } } _ { s , \ z } ^ { \mathrm { D E } }$ </td><td>MBS-adjusted deadline-aware priority</td><td> $p _ { n , i , \xi } ^ { \mathrm { p e r } }$ </td><td>Persistent deadline-miss indicator</td></tr><tr><td> $\mathcal { D } _ { \xi }$ </td><td>Set of nominated UAV-content pairs</td><td> ${ \mathcal { F } } _ { \cdot \cdot : \ s } ^ { \mathrm { d l } }$   $F _ { n , i , \xi } ^ { \mathrm { u n } }$ </td><td>Deadline-feasible cooperative UAV set</td></tr><tr><td> $\Delta P _ { i , j }$ </td><td>Normalized priority gain for replacing j with i</td><td> $\widetilde { E } _ { \cdot } ^ { \mathrm { u p } ^ { \cdot } }$ </td><td>Normalized communication energy for cache update</td></tr><tr><td> $s _ { n , \xi } ^ { \mathrm { r o c a l } }$ </td><td>Local state of UAV n</td><td> $\mathrm { { \underline { { \ g l o b a l } } } }$   $s _ { \xi }$ </td><td>Global network state</td></tr><tr><td> ${ \bf x } _ { n , \xi }$ </td><td>Cache state of UAV n</td><td> $\mathbf { q } _ { n , \xi } ^ { \mathrm { d e m } }$ </td><td>Local demand state of UAV n</td></tr><tr><td> $\mathbf { g } _ { \mathrm { ~ \tiny ~ z ~ } } ^ { \mathrm { c o n n } }$  n,ξ</td><td>Connectivity state of UAV n</td><td> $E ^ { \mathrm { { c o m m } } }$ </td><td>Communication-energy state of UAV n</td></tr><tr><td> $\psi _ { n , \xi } ^ { \mathrm { d i } }$ </td><td>Deadline-pressure vector of UAV n</td><td>n,ξ  $a _ { \xi }$ </td><td>Joint cache-update action</td></tr><tr><td> $c _ { n , \xi } ^ { - } , \ : c _ { n , \xi } ^ { + }$ </td><td>Removed and added content sets</td><td> ${ \bf z } _ { n , \xi }$ </td><td>Local temporal representation</td></tr><tr><td> $\mathbf { z } _ { \xi } ^ { \mathrm { g l o b a l } }$ </td><td>Global temporal representation</td><td> ${ \bf h } _ { \xi }$ </td><td>Fused temporal representation</td></tr><tr><td> $\mathbf { a } _ { n , \xi }$ </td><td>Cache-update action of UAV n</td><td> $\mathbf { m } _ { \xi } ^ { \mathrm { m a s k } }$ </td><td>Action feasibility mask</td></tr></table>

## 4.1. Deadline- and Energy-Aware Rule-Based Caching

Caching decisions based primarily on content popularity or local demand can improve content availability around current request locations. However, such decisions may be insuficient when contents have heterogeneous delivery deadlines and UAVs experience uneven communication-energy burdens. In particular, the preference for local cache service does not imply that every highly requested content should be replicated at its associated UAV. A content with a relatively relaxed deadline may still be served through a nearby cooperative cache within its deadline, allowing limited local cache space to be reserved for more deadline-sensitive contents. Moreover, repeatedly concentrating cache and service loads on particular UAVs may increase inter-UAV energy imbalance.

To incorporate these considerations into an interpretable rule-based strategy, we extend UBAC [19] into a deadlineand energy-aware variant, referred to as DE-UBAC. DE-UBAC retains the user-behavior-aware counting and coordinated cache-management structure of UBAC, including handover-aware history updates and MBS-level cache coordination, while introducing deadline-, cooperation-, and energy-aware decision rules.

DE-UBAC employs a hierarchical decision process rather than combining all factors into a single weighted caching score. First, the UBAC counting process is extended to distinguish requests that satisfy their deadlines from those that miss them. For request $k ,$ the deadline-miss indicator is defined as

$$
m _ { k } ^ { \mathrm { d l } } = \mathbb { I } \Big ( D _ { k } ^ { \mathrm { t a r } } > \delta _ { i _ { k } } \Big )
$$

(20)

where $D _ { k } ^ { \mathrm { t a r } }$ denotes the delivery delay evaluated from the perspective of the target UAV and $\delta _ { i _ { k } }$ is the deadline of requested content $i _ { k }$ . The deadline-aware counting increment is given by

$$
C _ { k } ^ { \mathrm { D E } } = C _ { k } ^ { \mathrm { U B A C } } \left( 1 + \lambda _ { \mathrm { d l } } m _ { k } ^ { \mathrm { d l } } \right)\tag{21}
$$

where $C _ { k } ^ { \mathrm { U B A C } }$ denotes the original UBAC counting increment and $\lambda _ { \mathrm { d l } }$ controls the additional weight assigned to a deadline-missed request. Consequently, deadline-missed requests accumulate caching priority more rapidly without diferentiating the magnitude of the deadline violation.

DE-UBAC also accounts for deadline misses that persist after a previous cache adaptation. The MBS maintains a persistent-miss indicator $p _ { n , i , \xi } ^ { \mathrm { p e r } }$ for each UAV–content pair and adjusts its deadline-aware priority as

$$
\widetilde { H } _ { n , i , \xi } ^ { \mathrm { D E } } = H _ { n , i , \xi } ^ { \mathrm { D E } } \left( 1 + \lambda _ { \mathrm { p e r } } p _ { n , i , \xi } ^ { \mathrm { p e r } } \right)\tag{22}
$$

when deadline misses remain unresolved after placement or relocation. Here, $H _ { n , i , \xi } ^ { \mathrm { D E } }$ denotes the accumulated deadlineaware history-table count, $p _ { n , i , \xi } ^ { \mathrm { p e r } } \ \in \ \{ 0 , 1 \}$ indicates a persistent deadline miss, and $\lambda _ { \mathrm { p e r } }$ controls its additional priority. This gives repeatedly unsatisfied UAV–content pairs stronger priority for subsequent cache adaptation.

Second, content placement and redundancy are determined according to deadline feasibility. If an existing cooperative copy can satisfy the content deadline, additional replication is avoided. Otherwise, DE-UBAC permits local or nearby placement to provide a shorter delivery path. The target UAV is preferred whenever feasible; otherwise, a nearby UAV capable of satisfying the deadline is considered.

Third, DE-UBAC incorporates UAV communicationenergy conditions as a secondary decision criterion. Deadline aware priority remains dominant when candidate priorities are clearly diferent. When their normalized priority diference is below a predefined threshold $\tau _ { H }$ , the corresponding caching demands are regarded as comparable, and the UAV communication-energy burden is used to resolve the decision. If the target UAV cannot directly accommodate the content, a deadline-feasible alternative UAV with a relatively lower energy burden is preferred. This hierarchical rule allows energy balancing to influence caching decisions without overriding a clearly stronger deadline-aware demand.

Finally, DE-UBAC considers cache-update energy before applying a replacement. For candidate content � and eviction candidate �, a replacement is performed only when

$$
\begin{array} { r } { \Delta P _ { i , j } > \lambda _ { \mathrm { u p } } \widetilde { E } _ { i } ^ { \mathrm { u p } } , } \end{array}\tag{23}
$$

where $\Delta P _ { i , j }$ denotes the normalized priority gain, ${ \widetilde E } _ { i } ^ { \mathrm { u p } }$ denotes the normalized communication energy required for the cache update, and $\lambda _ { \mathrm { u p } }$ determines how strongly the cacheupdate energy penalizes replacement. Otherwise, the current cache configuration is retained. Algorithm 1 summarizes the overall DE-UBAC procedure.

Algorithm 1 Deadline- and Energy-Aware UBAC   
Require: $\{ \mathbf { x } _ { n , \xi } \} _ { n \in \mathcal { N } } , \{ H _ { n , i , \xi } ^ { \mathrm { D E } } \} , \{ \delta _ { i } \} , \{ E _ { n , \xi } ^ { \mathrm { c o m m } } \}$   
Ensure: Updated cache configuration   
1: for each request � do   
2: � ← target UAV from the UBAC mobility rule   
3: $H _ { n , i _ { k } , \xi } ^ { \mathrm { D E } }  H _ { n , i _ { k } , \xi } ^ { \mathrm { D E } } + C _ { k } ^ { \mathrm { D E } }$   
4: end for   
5: if cache-decision epoch $\xi$ then   
Stage 1: Deadline-aware prioritization   
6: Update $\{ p _ { n , i , \xi } ^ { \mathrm { p e r } } \} _ { n , i }$   
7: for $n \in \mathcal { N } , i \in \mathcal { C }$ do   
8: $\widetilde { H } _ { n , i , \xi } ^ { \mathrm { D E } }  H _ { n , i , \xi } ^ { \mathrm { D E } } ( 1 + \lambda _ { \mathrm { p e r } } p _ { n , i , \xi } ^ { \mathrm { p e r } } )$   
9: end for   
10: Construct $\mathcal { D } _ { \xi }$   
Stage 2: Deadline-feasible placement   
11: for each $( n , i ) \in \mathcal { D } _ { \xi }$ do   
12: Construct ${ \mathcal { F } } _ { n , i , \xi } ^ { \mathrm { d l } }$   
13: if � is available through ${ \mathcal { F } } _ { n , i , \xi } ^ { \mathrm { d l } }$ then   
14: $D _ { \xi }  D _ { \xi } \setminus \{ ( n , i ) \}$   
15: end if   
16: end for   
Stage 3: Energy-aware coordination   
17: for each competing placement for � do   
18: if normalized priority diference $> \tau _ { H }$ then   
19: Select larger $\widetilde { H } _ { n , i , \xi } ^ { \mathrm { D E } }$   
20: else   
21: Retain demand with larger $E _ { n , \xi } ^ { \mathrm { c o m m } }$   
22: end if   
23: end for   
Stage 4: Update-cost-aware replacement   
24: for each selected (�, �) do   
25: � ← UBAC eviction candidate   
26: if $\Delta P _ { i , j } > \lambda _ { \mathrm { u p } } \widetilde { E } _ { i } ^ { \mathrm { u p } }$ then   
27: $C _ { n , \xi } ^ { - } \gets \{ \bar { j } \} , C _ { n , \xi } ^ { + } \gets \{ i \}$   
28: else   
29: $C _ { n , \xi } ^ { - } = C _ { n , \xi } ^ { + } = \emptyset$   
30: end if   
31: end for   
32: Apply $\{ ( C _ { n , \xi } ^ { - } , C _ { n , \xi } ^ { + } ) \} _ { n \in \mathcal { N } }$   
33: end if

Although DE-UBAC provides an interpretable realization of the considered deadline- and energy-aware caching problem, its decisions still rely on predefined rules, thresholds, and priorities. In a dynamic multi-UAV network, however, content demand, user mobility, connectivity, cooperative service conditions, and UAV energy states evolve simultaneously and interact with one another. Explicitly designing rules for all such interactions becomes increasingly dificult and limits the scalability and adaptability of a rule-based approach as the system complexity grows. Therefore, we next formulate the cache placement and replacement process as an MDP, allowing these dynamically coupled trade-ofs to be learned from interactions with the network environment.

## 4.2. MDP Formulation

The dynamic caching problem is formulated as an MDP at the cache-decision epochs. Although the underlying network evolves at every discrete simulation step, the caching agent acts at cache-decision epochs separated by intervals of duration $\Delta _ { c }$ . Let $\xi$ denote the cache-decision epoch index, with the corresponding simulation time given by $t _ { \xi } = \xi \Delta _ { c }$ The MDP is defined as

$$
\mathcal { M } = \langle S , \mathcal { A } , \mathcal { P } , r \rangle\tag{24}
$$

where S, A, P, and r denote the state space, action space, state-transition probability distribution, and reward function, respectively.

## 4.2.1. State space

At each cache-decision epoch, the local state of UAV � consists of: (i) its current cache contents, (ii) the demand conditions of its associated users, (iii) its user-association and network connectivity conditions, (iv) its current energy consumption state, and (v) its deadline pressure.

$$
\begin{array} { r l } & { s _ { n , \xi } ^ { \mathrm { l o c a l } } = \left( \mathbf { x } _ { n , \xi } , \mathbf { q } _ { n , \xi } ^ { \mathrm { d e m } } , \mathbf { g } _ { n , \xi } ^ { \mathrm { c o n n } } , E _ { n , \xi } ^ { \mathrm { c o m m } } , \psi _ { n , \xi } ^ { \mathrm { d l } } \right) , n \in \mathcal { N } , } \\ & { \quad \quad \quad s _ { \xi } ^ { \mathrm { g l o b a l } } = \left( s _ { 1 , \xi } ^ { \mathrm { l o c a l } } , \dots , s _ { N , \xi } ^ { \mathrm { l o c a l } } \right) } \end{array}\tag{25}
$$

The vectors $\mathbf { x } _ { n , \xi } , \mathbf { q } _ { n , \xi } ^ { \mathrm { d e m } }$ , and $\mathbf { g } _ { n , \xi } ^ { \mathrm { c o n n } }$ represent the cache state, local demand state, and user-association and connectivity state of UAV $n ,$ respectively. The term $E _ { n , \xi } ^ { \mathrm { c o m m } }$ represents its communication energy state. The global state conceptually consists of the collection of the local states of all UAVs. The global state space  consists of all feasible global network states.

Deadline pressure represents the recent demand for deadline-sensitive content at each UAV. It is obtained by weighting the latest requests according to their content deadlines and aggregating them by content type.

$$
\psi _ { n , \kappa , \xi } ^ { \mathrm { d l } } = \frac { \displaystyle \sum _ { i : \tau _ { i } = \kappa } \nu _ { n , i , \xi } \omega _ { i } ^ { \mathrm { u r g } } } { \displaystyle \operatorname* { m a x } \left\{ \epsilon _ { \psi } , \sum _ { \kappa ^ { \prime } \in T } \sum _ { j : \tau _ { j } = \kappa ^ { \prime } } \nu _ { n , j , \xi } \omega _ { j } ^ { \mathrm { u r g } } \right\} } ,\tag{26}
$$

Here, $\nu _ { n , i , \xi }$ denotes the number of requests for content item � arriving at UAV � during the most recent cache-decision interval, $\omega _ { i } ^ { \mathrm { u r g } }$ denotes the deadline urgency of content item �, where a shorter delivery deadline corresponds to a larger urgency value, $\kappa \in \tau$ denotes a content type, and $\epsilon _ { \psi }$ is a small positive constant.

## 4.2.2. Action space

At each cache-decision epoch, the centralized agent determines the content items to be removed from and added to each UAV cache. The joint action is defined as

$$
a _ { \xi } = \left\{ \left( C _ { n , \xi } ^ { - } , C _ { n , \xi } ^ { + } \right) \right\} _ { n \in \mathcal { N } }\tag{27}
$$

where ${ \mathcal { C } } _ { n , \xi } ^ { - }$ and $C _ { n , \xi } ^ { + }$ denote, respectively, the sets of content items removed from and added to UAV �. The collection of all feasible joint cache-update actions constitutes the action space .

The action space also includes a no-operation action. For UAV �, the no-operation action is defined as

$$
C _ { n , \xi } ^ { - } = C _ { n , \xi } ^ { + } = \emptyset\tag{28}
$$

A cache-update action is considered feasible only if it satisfies the constraints in Section 3.4.2. The environment evaluates these constraints when decoding the sampled action into an executable cache-update command.

## 4.2.3. State–transition probability distribution

The state-transition distribution describes the evolution of the global network state over one cache-decision interval after applying a joint cache-update action. It is defined as

$$
s _ { \xi + 1 } ^ { \mathrm { g l o b a l } } \sim \mathcal { P } \Big ( \cdot \mid s _ { \xi } ^ { \mathrm { g l o b a l } } , a _ { \xi } \Big )\tag{29}
$$

Given the global state $s _ { \xi } ^ { \mathrm { g l o b a l } }$ and the selected joint action $a _ { \xi }$ the network evolves over the following cache-decision interval according to the transition distribution . This evolution accounts for cache-update execution, user mobility, request generation, wireless resource variations, service outcomes, and energy consumption. The resulting state $s _ { \xi + 1 } ^ { \mathrm { g l o b a l } }$ summarizes the network conditions at the next cache-decision epoch.

## 4.2.4. Rewardfunction

To optimize the multi-objective caching problem using PPO, we construct a scalarized interval-level reward that combines the three objectives. Since the caching agent acts at cache-decision epochs separated by intervals of duration $\Delta _ { c } ,$ the reward is computed at the end of each cachedecision interval. The interval reward integrates the service performance and energy costs resulting from the selected cache-update action. To jointly account for deadline misses, communication energy, and inter-UAV energy imbalance, we define the interval-level reward as

$$
\begin{array} { r } { \boldsymbol { r } _ { \xi } = - \left[ \cfrac { E _ { \xi } ^ { \mathrm { t o t a l } } } { E _ { \mathrm { r e f } } } + \cfrac { \boldsymbol { I } _ { E , \xi } } { \boldsymbol { I } _ { E , \mathrm { r e f } } } + \mathrm { D M R } _ { \xi } + \mathcal { L } _ { \mathrm { S F R } , \xi } \right] , } \\ { \mathcal { L } _ { \mathrm { S F R } , \xi } = \left[ \operatorname* { m a x } \left( 0 , \cfrac { \mathrm { S F R } _ { \xi } - \epsilon _ { \mathrm { s r v } } } { 1 - \epsilon _ { \mathrm { s r v } } } \right) \right] ^ { 2 } \qquad } \end{array}\tag{30}
$$

where $\mathrm { D M R } _ { \xi } , \mathrm { S F R } _ { \xi } , E _ { \xi } ^ { \mathrm { t o t a l } }$ , and $\tau _ { E , \xi }$ denote the deadlinemiss rate, service failure ratio (SFR), total communication energy, and inter-UAV energy imbalance during interval $\xi ,$ respectively. Specifically, $\mathrm { S F R } _ { \xi } , \mathrm { i } . \mathrm { e } .$ ., the dropped-request ratio, is the fraction of requests generated during interval $\xi$ that cannot be served owing to the absence of a feasible delivery path. The normalization constants $E _ { \mathrm { r e f } }$ and $\boldsymbol { \mathit { I } } _ { E , \mathrm { r e f } }$ are set to the 95th percentiles of the corresponding quantities observed in ofline calibration rollouts sampled from the trainingscenario distribution. They are fixed before PPO training and are not updated using evaluation episodes. The term $\mathcal { L } _ { \mathrm { S F R } , \xi }$ penalizes service failure ratios above the admissible threshold $\epsilon _ { \mathrm { s r v } }$ . The SFR penalty prevents the policy from artificially reducing the DMR by allowing excessive request drops.

The above MDP reveals two key requirements for the caching policy. First, the state variables describing demand, connectivity, deadline pressure, and communication-energy conditions evolve over time, making recent observation histories informative for the current cache decision. Second, cooperative caching and inter-UAV energy balancing require each local condition to be interpreted in the context of the entire UAV network. Accordingly, GRU-DEPPO employs UAV-specific and network-wide temporal representations and uses them within a centralized policy for joint cacheupdate decision making.

## 4.3. GRU-based Temporal Representation

The MDP state contains time-varying information on local content demand, connectivity, cache conditions, deadline pressure, and UAV communication energy. Since these observations are temporally correlated, relying only on the current state may not suficiently characterize the evolving local and network-wide conditions relevant to cache placement and replacement. Moreover, directly providing the temporal histories of all UAVs to the centralized controller would increase the input dimension and communication overhead. We therefore employ local and global GRUs to construct compact temporal representations at the UAV and network levels, respectively.

For each UAV �, a shared local GRU independently encodes its most recent � local observations. Let $\mathbf { 0 } _ { n , \xi } ^ { \mathrm { l o c } }$ denote the local observation vector at cache-decision epoch $\xi .$ The UAV-specific temporal representation is obtained as

$$
\begin{array} { r } { \mathbf { z } _ { n , \xi } = f _ { \theta _ { \mathrm { l o c } } } \left( \mathbf { o } _ { n , \xi - L + 1 } ^ { \mathrm { l o c } } , \ldots , \mathbf { o } _ { n , \xi } ^ { \mathrm { l o c } } \right) , \qquad n \in \mathcal { N } . } \end{array}\tag{31}
$$

The GRU parameters are shared across UAVs, while a separate representation ${ \bf z } _ { n , \xi }$ is generated for each UAV, thereby preserving UAV-specific temporal characteristics. The detailed composition of $\mathbf { 0 } _ { n , \xi } ^ { \mathrm { l o c } }$ is provided in Appendix B.

In parallel, the MBS maintains a recent history of compact network-wide context vectors constructed from information available at the centralized controller. These contexts summarize network activity, service quality, service modes, communication energy, demand variation, and inter-UAV energy imbalance. The resulting context tokens are processed by the global GRU as

$$
\begin{array} { r } { \mathbf { z } _ { \xi } ^ { \mathrm { g l o b a l } } = f _ { \theta _ { \mathrm { g l o b a l } } } \left( \mathbf { e } _ { \xi - L + 1 } ^ { \mathrm { n e t } } , \dots , \mathbf { e } _ { \xi } ^ { \mathrm { n e t } } \right) , } \end{array}\tag{32}
$$

where $\mathbf { e } _ { \xi } ^ { \mathrm { n e t } }$ denotes the embedded network-wide context at epoch $\xi .$ Its detailed composition and embedding procedure are described in Appendix B.

Before PPO training, the shared local GRU is pretrained using local-observation trajectories collected with a lightweight caching policy. Let $\mathcal { D } _ { \mathrm { p r e } }$ denote the resulting pretraining dataset. Each observation sequence is associated with semantic summary targets derived from its observed demand, cache, deadline, service, and energy conditions. A lightweight prediction head is trained together with the local GRU using

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e } } = \mathrm { S m o o t h L 1 } \left( \widehat { \mathbf { y } } _ { n , \xi } ^ { \mathrm { b a s e } } , \mathbf { y } _ { n , \xi } ^ { \mathrm { b a s e } } \right) + \lambda _ { \mathrm { t y p e } } \mathbf { M S E } \left( \widehat { \mathbf { y } } _ { n , \xi } ^ { \mathrm { t y p e } } , \mathbf { y } _ { n , \xi } ^ { \mathrm { t y p e } } \right) , } \end{array}\tag{33}
$$

where the base component summarizes the local network conditions and the type component represents the requesttype distribution. After pretraining, the prediction head is discarded and the shared local GRU is kept fixed during PPO training. The global GRU, in contrast, is trained jointly with the centralized policy. At each cache-decision epoch, only the compact local representation ${ \bf z } _ { n , \xi }$ is provided by each UAV to the MBS.

## 4.4. GRU-assisted PPO optimization

At each cache-decision epoch, the recent local observation history of each UAV is encoded by the shared local GRU, while the MBS encodes the recent network-wide context using the global GRU. The resulting local and global temporal representations are fused and provided to a centralized PPO actor–critic [36]. The actor selects feasible cache placement and replacement actions, and the resulting service, deadline, and communication-energy outcomes are used to update the policy.

## 4.4.1. PPO input

The MBS constructs the PPO input by concatenating the local temporal representations of all UAVs with the global representation:

$$
\mathbf { h } _ { \xi } = \left[ \mathbf { z } _ { 1 , \xi } ; \ldots ; \mathbf { z } _ { N , \xi } ; \mathbf { z } _ { \xi } ^ { \mathrm { g l o b a l } } \right]\tag{34}
$$

where ${ \bf h } _ { \xi }$ represents the temporal network context used for cache-update decision making at epoch $\xi$

## 4.4.2. Factorized cache-update action

To represent cache placement and replacement decisions eficiently, the joint cache-update action is factorized across UAVs as

$$
a _ { \xi } = \left[ { \bf a } _ { 1 , \xi } ; \ldots ; { \bf a } _ { N , \xi } \right] , \qquad { \bf a } _ { n , \xi } = \left( o _ { n , \xi } , b _ { n , \xi } , i _ { n , \xi } \right)\tag{35}
$$

where $o _ { n , \xi }$ specifies the cache operation, $b _ { n , \xi }$ identifies the corresponding cache slot, and $i _ { n , \xi }$ identifies the selected content item. The operation set includes a no-operation choice, in which the slot and content components are ignored.

Let $\mathbf { m } _ { \xi } ^ { \mathrm { m a s k } }$ denote the feasibility mask constructed from the current cache state and network conditions. The masked policy is factorized as

$$
\begin{array} { r } { \pi _ { \boldsymbol { \theta } _ { \pi } } ^ { \mathrm { m a s k } } \left( a _ { \xi } \mid \boldsymbol { \mathbf { h } } _ { \xi } , \boldsymbol { \mathbf { m } } _ { \xi } ^ { \mathrm { m a s k } } \right) = \displaystyle \prod _ { n \in \mathcal { N } } \pi _ { \boldsymbol { \theta } _ { \pi } } ^ { o } \left( o _ { n , \xi } \mid \boldsymbol { \mathbf { h } } _ { \xi } , \boldsymbol { \mathbf { m } } _ { \xi } ^ { \mathrm { m a s k } } \right) } \\ { \times \pi _ { \boldsymbol { \theta } _ { \pi } } ^ { b } \left( b _ { n , \xi } \mid \boldsymbol { \mathbf { h } } _ { \xi } , \boldsymbol { \mathbf { m } } _ { \xi } ^ { \mathrm { m a s k } } \right) } \\ { \times \pi _ { \boldsymbol { \theta } _ { \pi } } ^ { i } \left( i _ { n , \xi } \mid \boldsymbol { \mathbf { h } } _ { \xi } , \boldsymbol { \mathbf { m } } _ { \xi } ^ { \mathrm { m a s k } } \right) . } \end{array}\tag{36}
$$

The mask excludes infeasible component choices before sampling. The sampled action is decoded into the remove/add sets $\left( { C } _ { n , \xi } ^ { - } , { C } _ { n , \xi } ^ { + } \right)$ , and the resulting cache update is executed subject to the cache-capacity and cache-update feasibility constraints defined in Section 3.4.2.

## 4.4.3. PPO-basedpolicy optimization

After applying the selected cache-update action, the network evolves for one cache-decision interval $\Delta _ { c } .$ . The resulting request service outcomes, deadline misses, service failures, and communication energy consumption determine the interval-level reward $r _ { \xi }$ defined in Section 4.2.

The actor and critic are trained using standard PPO with generalized advantage estimation (GAE) [36]. Let $\widehat { A } _ { \xi }$ denote the GAE advantage estimate and let $\rho _ { \xi }$ denote the probability ratio between the updated policy and the policy used during rollout collection. The clipped policy objective is

$$
\mathcal { L } _ { \mathrm { c l i p } } = \mathbb { E } _ { \xi } \left[ \operatorname* { m i n } \left( \rho _ { \xi } \widehat { A } _ { \xi } , \mathrm { c l i p } \left( \rho _ { \xi } , 1 - \epsilon _ { \mathrm { c l i p } } , 1 + \epsilon _ { \mathrm { c l i p } } \right) \widehat { A } _ { \xi } \right) \right]\tag{37}
$$

The critic is trained using the corresponding return estimates, and entropy regularization is applied to encourage policy exploration. The PPO training parameters are summarized in the Appendix. The complete GRU-DEPPO training procedure is summarized in Algorithm 2.

## 5. Results and discussion

This section evaluates GRU-DEPPO from multiple perspectives. We first compare its overall performance with the baseline schemes in terms of deadline-miss rate, contenttype-specific deadline-miss performance, total communication energy consumption, and inter-UAV energy imbalance. We then examine its robustness under more stringent content-specific deadline requirements without retraining. Additional analyses investigate cache-update energy, request service modes, and adaptation to abrupt changes in content demand. Finally, we conduct an ablation study to quantify the contribution of the GRU-based temporal representation to both learning behavior and caching performance.

```tcl
Algorithm 2 GRU-DEPPO Training
Require: $\mathcal { D } _ { \mathrm { p r e } } , N _ { \mathrm { e p } } , B _ { \mathrm { r o l l } }$
Ensure: Trained GRU-DEPPO policy
1: Train $f _ { \theta _ { \mathrm { l o c } } }$ using $\mathcal { L } _ { \mathrm { p r e } }$
2: Fix $\theta _ { \mathrm { l o c } }$ and initialize $f _ { \theta _ { \mathrm { g l o b a l } } }$ , PPO, and 
3: for $e = 1 , \ldots , N _ { \mathrm { e p } }$ do
4: Reset environment and histories
5: while episode is not terminated do
Stage 1: Temporal encoding
6: for $n \in \mathcal N$ do
7: $\mathbf { z } _ { n , \xi }  f _ { \theta _ { \mathrm { l o c } } } ( \mathbf { o } _ { n , \xi - L + 1 } ^ { \mathrm { l o c } } , \ldots , \mathbf { o } _ { n , \xi } ^ { \mathrm { l o c } } )$
8: end for
9: $\mathbf { z } _ { \xi } ^ { \mathrm { g l o b a l } }  f _ { \theta _ { \mathrm { g l o b a l } } } ( \mathbf { e } _ { \xi - L + 1 } ^ { \mathrm { n e t } } , \dots , \mathbf { e } _ { \xi } ^ { \mathrm { n e t } } )$
10: Construct $\mathbf { h } _ { \xi }$
Stage 2: Cache-action generation
11: Construct $\mathbf { m } _ { \xi } ^ { \mathrm { m a s k } }$
12: Sample $a _ { \xi } \sim \overset { \cdot } { \pi } { \underset { \theta _ { \pi } } { \mathrm { m a s k } } } ( \cdot \mid \mathbf { h } _ { \xi } , \mathbf { m } _ { \xi } ^ { \mathrm { m a s k } } )$
13: Decode $a _ { \xi }$ into $\{ ( C _ { n , \xi } ^ { - } , C _ { n , \xi } ^ { + } ) \} _ { n \in \mathcal { N } }$
Stage 3: Environment interaction
14: Apply $a _ { \xi }$ and evolve for $\Delta _ { c }$
15: Observe $\stackrel { \mathrm { \tiny ~ ( ~ g l o b a l } } { s _ { \xi + 1 } ^ { \xi + 1 } }$ and $r _ { \xi }$
16: Store transition in $_ B$
17: if $| B | \ge B _ { \mathrm { r o l l } }$ then
Stage 4: PPO optimization
18: Compute GAE and return estimates
19: Update $f _ { \theta _ { \mathrm { g l o b a l } } }$ and PPO
20: $B  \emptyset$
21: end if
22: end while
23: end for
```

We compare GRU-DEPPO with LRU as a conventional cache replacement baseline, DE-UBAC (Algorithm 1) as a rule-based caching scheme, and a TLPPO-based caching baseline adapted from the caching component ofTLPPO [25]. The TLPPO-based baseline employs PPO to determine content placement and replacement actions based on recently observed local content demand and the current UAV cache state. Its policy is trained to maximize interval-level cache-hit performance. All reported performance evaluation results are averaged over 30 independent random seeds. The detailed simulation parameters are listed in Table 3.

Table 3  
Main simulation parameters.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>N</td><td>10</td></tr><tr><td>|U|</td><td>150</td></tr><tr><td>W</td><td>3000 m × 3000 m</td></tr><tr><td>UAV height</td><td>100 m</td></tr><tr><td>User speed</td><td>[1, 20] m/s</td></tr><tr><td>User groups</td><td>3</td></tr><tr><td> $R _ { g }$ </td><td>100 m</td></tr><tr><td> $\alpha _ { u }$ </td><td>[0.4]/[0.5]/[0.8]</td></tr><tr><td> $| c | , | r |$ </td><td>200,3</td></tr><tr><td> $S$ </td><td>25 Mbit</td></tr><tr><td> $K _ { n } S$ </td><td>200 Mbit</td></tr><tr><td> $\{ \delta _ { \kappa } \} _ { \kappa \in \mathcal { T } }$ </td><td>[2.5, 4.0, 5.5] s</td></tr><tr><td> $\alpha$ </td><td>0.6</td></tr><tr><td> $\lambda _ { u , t }$ </td><td>0.6/0.7/0.8</td></tr><tr><td> $\mathrm { s t d } ( \epsilon _ { i , g , t } )$ </td><td>0.02</td></tr><tr><td> $\beta _ { \mathrm { s h i f t } }$ </td><td>0.4</td></tr><tr><td>Simulation duration</td><td>1000 s</td></tr><tr><td>Simulation time step</td><td>1 s</td></tr><tr><td> $\Delta _ { c }$ </td><td>10 s</td></tr><tr><td> $f _ { u } , f _ { \mathrm { U A V } } , f _ { \mathrm { M B S } }$ </td><td>5, 4.5, 4 GHz</td></tr><tr><td>Link bandwidths</td><td>40/30/10 MHz</td></tr><tr><td> $P _ { n } ^ { \mathrm { t x } }$ </td><td>30 dBm</td></tr><tr><td>MBS transmit power</td><td>43 dBm</td></tr><tr><td>UAV receive power</td><td>1.2 W</td></tr><tr><td> $X , Y$ </td><td>11.9, 0.13</td></tr><tr><td>ηLoS, ηNLoS</td><td>1.6, 23.0 dB</td></tr><tr><td> $N _ { 0 }$ </td><td> $3 . 9 8 \times 1 0 ^ { - 2 1 } ~ \mathrm { W / H z }$ </td></tr><tr><td> $\Gamma _ { \mathrm { m i n } }$ </td><td>-10 dB</td></tr><tr><td> $R _ { \mathrm { m i n } }$ </td><td>10⁵ bps</td></tr><tr><td>DE  $\cdot \mathsf { U B A C } \colon \lambda _ { \mathrm { d l } } , \lambda _ { \mathrm { p e r } } , \tau _ { H } , \lambda _ { \mathrm { u p } }$ </td><td>0.5, 0.5, 0.1, 0.25</td></tr></table>

Group-specific parameters are listed in the order G1, G2, and G3. For grouped link parameters, values are reported in the order User–UAV, UAV– UAV, and MBS–UAV.

## 5.1. Simulation setup

We consider a multi-UAV service area in which users are divided into multiple groups with distinct mobility and content-demand characteristics.

## 5.1.1. User mobility and spatial locality

To model heterogeneous local conditions, users follow a group hotspot-aware mobility model. Users are divided into several behavior groups, and each group is associated with a spatial hotspot representing its preferred area of activity. Users within the same group share the same hotspot region, while their movement trajectories are generated independently. Each user is also assigned an individual hotspot attraction probability to represent diferent degrees of spatial locality.

Let �(�) denote the group of user �, and let $\mathbf { h } _ { g ( u ) }$ and $R _ { g ( u ) }$ denote the hotspot center and radius of that group, respectively. At each waypoint selection, user � selects a destination around its group hotspot with probability $\alpha _ { u } ,$ otherwise, the destination is selected according to the ordinary randomwaypoint process [37]:

$$
\mathbf { w } _ { u , t } \sim \left\{ \begin{array} { l l } { { \cal D } \big ( \mathbf { h } _ { g ( u ) } , R _ { g ( u ) } \big ) , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } \alpha _ { u } , } \\ { { \cal D } _ { \mathrm { R W P } } ( \mathscr { W } ) , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 - \alpha _ { u } } \end{array} \right.\tag{38}
$$

where $D ( \mathbf { h } _ { g ( u ) } , R _ { g ( u ) } )$ denotes the waypoint distribution within the hotspot region centered at $\mathbf { h } _ { g ( u ) }$ with radius $R _ { g ( u ) } ,$ and  denotes the service area. The attraction probability $\alpha _ { u }$ is sampled according to the user’s group, thereby allowing users in diferent groups to exhibit diferent degrees of spatial locality.

As users move between spatial regions, their serving UAVs and local network conditions change over time. Consequently, diferent UAVs observe diferent user populations, request compositions, connectivity conditions, and service loads. In addition, the hotspot locations are shifted during the episode to represent a change in the spatial mobility regime.

Spatial locality is accompanied by group-conditioned request behavior. Users belonging to the same group share a common time-varying local content distribution $p _ { i , g ( u ) , t } ,$ whereas diferent groups exhibit diferent content preferences. The request-generation probability may also vary across users or groups. Consequently, as users move across the service area and their serving UAVs change, diferent UAVs observe distinct local trafic patterns resulting from diferent mixtures of user groups. At the same change point, the group-conditioned content-demand distributions are also shifted to represent temporal changes in user preferences.

## 5.2. Simulation results

Fig. 3 compares the overall performance of the considered caching schemes in terms of deadline-miss rate, content-type-specific deadline-miss performance, total communication energy consumption, and inter-UAV energy imbalance.

As shown in Fig. 3a, LRU exhibits the highest deadlinemiss rate because its cache replacement decisions are based solely on content recency and do not explicitly account for local demand conditions or delivery deadlines. DE-UBAC improves upon LRU by incorporating localized user behavior, deadline misses, cooperative delivery feasibility, and UAV energy conditions into its hierarchical cache-decision rules. However, these factors are coordinated through predefined priorities and thresholds, which limits the flexibility of the rule-based policy under rapidly changing network conditions.

TLPPO achieves a substantially lower deadline-miss rate than LRU and DE-UBAC. By observing recent local content demand and the current cache state, its PPO-based policy can proactively adjust content placement rather than reacting only after individual cache misses. Moreover, because the policy is trained to maximize interval-level cache-hit performance, it substantially reduces the average delivery delay. Nevertheless, GRU-DEPPO achieves the lowest deadlinemiss rate among all compared schemes, reducing the DMR by approximately 51% relative to TLPPO. This improvement can be attributed to two key diferences from TLPPO. First, GRU-DEPPO exploits local and global GRUs to capture the temporal evolution of spatially heterogeneous local network conditions. Second, its decision model explicitly accounts for content-specific deadlines, rather than focusing primarily on interval-level cache-hit performance.

![](images/b18269f1c3b71321040d5641193fc01e577e2983fd292ffc86376cec3af6e2fa.jpg)  
(a) Deadline-miss rate

![](images/49c9aac625bf25460e743d7d8f54e2eb6adfe1cb8489b785ad00566a391ba68f.jpg)  
(b) DMR by content type

![](images/55498dd9b2f65f0b1358cd7651a068c33332790d2d28b6b475bab076c2b11014.jpg)  
(c) Total communication energy

![](images/f52df94e7196615384f73f1307d7af3355acce27f1908df3067e75210ba12d3c.jpg)  
(d) Inter-UAV energy imbalance

Figure 3: Performance with deadlines of [2.5, 4.0, 5.5] s  
![](images/87340f2ae8ee37a17777d9400f970c8c22745d28082493a2560cc50df9773f0f.jpg)  
(a) Deadline-miss rate

![](images/7b00e414ea02383361f07b9008b9c561b5bb8e9c10310d6ff66b4a89719cc826.jpg)  
(b) DMR by content type

![](images/47abd17f01369f8bbffcb3fc61a6a644a5434b0ddfd3f444605dc7bd70345cc6.jpg)  
(c) Total communication energy

![](images/4f1de424703166670602c5e1be704f771236d1a791882df5ee3fc66730e2b055.jpg)  
(d) Inter-UAV energy imbalance  
Figure 4: Performance with deadlines of [2.0, 3.0, 4.0] s

Fig. 3b further shows the DMR for each content type. GRU-DEPPO achieves a substantial improvement for contents with the most stringent 2.5-s deadline, whereas TLPPO achieves a slightly lower DMR for contents with the 4.0- s deadline. For contents with the relatively relaxed 5.5-s deadline, GRU-DEPPO again achieves the lowest DMR. These results indicate that GRU-DEPPO does not improve deadline-miss performance uniformly across all content types. Instead, the deadline-dependent performance suggests that the policy distinguishes heterogeneous content-specific delivery requirements and places greater emphasis on satisfying requests with tighter deadlines. In particular, the large reduction observed for the most stringent content type contributes substantially to the lower overall DMR of GRU-DEPPO.

Interestingly, TLPPO achieves a slightly lower average delivery delay than GRU-DEPPO, with approximately 1.99 s and 2.01 s, respectively. However, their corresponding deadline-miss rates are approximately 13.97% and 6.85%, respectively. This contrast indicates that a lower average delivery delay does not necessarily lead to better deadline satisfaction under heterogeneous content-specific deadlines. The service failure ratios remain within a relatively narrow range, with 11.66% for GRU-DEPPO, 10.59% for LRU, 10.62% for DE-UBAC, and 10.69% for TLPPO. Although GRU-DEPPO exhibits a slightly higher SFR, the approximately one-percentage-point diference indicates that its substantial reduction in deadline-miss rate is not attributable to excessive request dropping.

Fig. 3c shows the total communication energy consumed by all UAVs throughout the simulation period. LRU incurs the highest energy consumption among the compared schemes, whereas GRU-DEPPO achieves the lowest total communication energy. The improvement over TLPPO is relatively modest compared with those over LRU and DE-UBAC. This is because TLPPO also promotes edge cache hits by maximizing interval-level cache-hit performance. Consequently, a large fraction of requests can be served through local or cooperative UAV caching rather than relying on MBS-assisted delivery, which naturally reduces communication energy consumption. Although the reduction in total communication energy relative to TLPPO is relatively modest, a substantially larger diference is observed in the inter-UAV energy imbalance, as shown in Fig. 3d. GRU-DEPPO achieves an energy-imbalance value of 0.801, compared with 1.256 for TLPPO, corresponding to an approximately 36% reduction.

This diference arises because TLPPO primarily improves network-wide cache-hit performance and therefore can reduce the overall communication cost, but it does not explicitly regulate how communication energy is distributed among individual UAVs. In contrast, GRU-DEPPO explicitly incorporates inter-UAV energy imbalance into its decision objective, discouraging cache and service decisions that repeatedly concentrate communication load on a subset of UAVs. In particular, among the MBS-direct UAVs that also serve as relay gateways for edge UAVs without direct MBS connectivity, GRU-DEPPO achieves an energyconsumption CV of 0.568, compared with 0.977 for LRU, 0.908 for DE-UBAC, and 0.922 for TLPPO. This result indicates that the overall reduction in inter-UAV energy imbalance is especially pronounced among MBS-direct UAVs, where relay-related communication burden is more likely to be concentrated.

Overall, GRU-DEPPO efectively captures the temporal evolution of spatially heterogeneous local network conditions through its GRU-based representations. Moreover, by explicitly incorporating content-specific deadlines and inter-UAV energy imbalance into the PPO-based decision framework, rather than focusing solely on average delivery delay or cache-hit performance, GRU-DEPPO achieves a better overall trade-of among deadline satisfaction, communication energy consumption, and inter-UAV energy imbalance.

Fig. 4 further evaluates the compared schemes under more stringent delivery requirements. Considering that the PPO-based caching schemes exhibit average delivery delays of approximately 2 s under the original setting, the contenttype-specific deadlines are tightened from [2.5, 4.0, 5.5] s to [2.0, 3.0, 4.0] s. The PPO-based policies obtained under the original simulation setting are evaluated without retraining under the tightened deadlines. An interesting observation can be made from Fig. 4a. Compared with the original deadline setting, the DMR of TLPPO increases by 11.33 percentage points, whereas that of GRU-DEPPO increases by 12.39 percentage points. This degradation is not necessarily attributable solely to the caching policy itself. When the delivery deadlines become more stringent, requests become more sensitive to transient resource limitations caused by, for example, temporary concentrations of users or request arrivals in particular network regions. Under such conditions, the available communication resources or feasible delivery paths may be insuficient to satisfy all requests within the shortened deadlines, regardless of the caching decision. Consequently, tighter deadline requirements can expose environment-intrinsic performance limitations that are dificult to fully mitigate through cache placement alone.

Fig. 4b further decomposes the DMR shown in Fig. 4a according to the content-specific deadline requirement. As expected, contents with the most stringent 2.0-s deadline exhibit the highest DMR across all compared schemes. However, GRU-DEPPO shows a diferent performance trade-of from TLPPO. Although GRU-DEPPO exhibits somewhat higher DMRs for contents with the relatively relaxed 3.0- s and 4.0-s deadlines, it achieves a substantial reduction in DMR for contents with the most stringent 2.0-s deadline. This result suggests that, as the delivery requirements become more stringent, the advantage of GRU-DEPPO becomes increasingly concentrated on the most deadlinesensitive contents. In other words, the policy places greater emphasis on satisfying requests with tighter deadline requirements rather than uniformly reducing the DMR of all content types.

Despite the slightly higher DMRs for the relatively relaxed content types, GRU-DEPPO achieves a 24.0% lower overall DMR than TLPPO under the stringent deadline setting. This result indicates that, in the observed performance trade-of, the substantial reduction in deadline misses for the most deadline-sensitive content outweighs the smaller performance degradations for the relatively relaxed content types. Moreover, as shown in Figs. 4c and 4d, GRU-DEPPO maintains its advantages in both total communication energy consumption and inter-UAV energy imbalance. These results indicate that GRU-DEPPO maintains a favorable overall performance trade-of even when the content-delivery requirements become substantially more stringent.

We next analyze the composition of communicationenergy consumption and the distribution of request service modes to further investigate how the compared caching schemes exploit the available service opportunities under dynamic network conditions. Fig. 5a separately compares the communication energy consumed for cache updates during the simulation period. LRU and DE-UBAC incur relatively low cache-update energy among the compared schemes, but for diferent reasons. LRU performs cache replacement using a simple recency-based rule and therefore triggers relatively limited cache reconfiguration. In contrast, DE-UBAC explicitly suppresses a replacement when the expected priority gain is insuficient to justify the corresponding update-energy cost, thereby avoiding low-benefit cache updates. Nevertheless, its predefined decision rules provide less flexibility in adapting the overall cache configuration than the learning-based schemes. Consequently, low cacheupdate energy alone does not necessarily imply lower total communication energy.

Among the RL-based schemes, TLPPO consumes approximately 36.4% more cache-update energy than GRU-DEPPO. TLPPO is trained primarily to maximize intervallevel cache-hit performance and does not explicitly account for the communication cost associated with cache updates. Consequently, the policy may still favor cache reconfiguration even when the expected gain in cache-hit performance is relatively small compared with the additional energy required for the update, leading to more aggressive cacheupdate behavior. In contrast, GRU-DEPPO explicitly incorporates total communication energy consumption into its decision objective. Therefore, the policy accounts for the energy cost incurred by cache reconfiguration as part of the overall communication-energy trade-of. This enables GRU-DEPPO to achieve lower cache-update energy than TLPPO while simultaneously attaining a lower deadline-miss rate and lower total communication energy consumption. Fig. 5b presents the distribution of request service modes for each caching scheme. LRU and DE-UBAC show substantially larger fractions of MBS-assisted service than the RL-based schemes, indicating more frequent reliance on MBS-assisted content retrieval.

A particularly notable comparison can be made between TLPPO and GRU-DEPPO. The two schemes exhibit similar fractions of MBS-assisted service, while TLPPO achieves a slightly larger fraction of direct cache hits. However, its cooperative service is concentrated on one-hop delivery. In contrast, GRU-DEPPO also exploits two-hop and three-hop cooperative delivery, indicating that it utilizes a broader set of service opportunities available in the multi-UAV network.

![](images/fa74473ad801bfff53a1ed95f9f7653b9004cfc5314a3e01095f77fcfbca4cc5.jpg)  
(a) Cache-update energy

![](images/373d002489aa185985b9346da52d7a5120ebe1ea5b4115133e0c922a1e4375b6.jpg)  
(b) Distribution of request service modes

Figure 5: Energy and service characteristics  
![](images/d99d78065c911f867a96be9fbc850000473a5dc98a97a6a364a123e77989caa5.jpg)  
Figure 6: Deadline-miss rate trend

Despite not achieving a larger direct-hit ratio than TLPPO, GRU-DEPPO attains a substantially lower deadline-miss rate. This result indicates that cache-hit performance alone does not fully explain deadline satisfaction under heterogeneous content-specific requirements.

We next evaluate the adaptability of the caching schemes to dynamic changes in the network environment. To facilitate a direct comparison of the adaptation behavior, the change point is deliberately fixed at the midpoint of the simulation for this experiment. Fig. 6 shows the temporal evolution of the deadline-miss rate when both the mobility context and content-demand distribution are shifted at this point. LRU and DE-UBAC exhibit relatively slow adaptation to the changed demand conditions. LRU relies on a reactive recency-based replacement rule, and DE-UBAC adapts its cache configuration using interval-level observations through predefined deadline-, cooperation-, and energyaware rules. However, its decisions still depend on accumulated history information, persistent-miss feedback, and fixed decision thresholds. Consequently, its cache configuration responds to an abrupt distribution shift only after the changed network conditions are suficiently reflected in these rule-based decision signals.

In contrast, the RL-based schemes adjust their cache decisions more rapidly after the distribution shift based on the newly observed network states. Among them, GRU-DEPPO exhibits the smallest immediate DMR degradation and the fastest recovery. To quantify the immediate performance degradation, we use the average DMR during the 100 s immediately preceding the distribution shift as the preshift reference and compare it with the average DMR during the first 20 s after the shift. Under this criterion, the DMR increases by approximately 14.6% for TLPPO, compared with only 7.8% for GRU-DEPPO. The recovery time is measured as the elapsed time after the distribution shift until the DMR returns to the pre-shift reference level. GRU-DEPPO returns to this level after approximately 38 s, while TLPPO requires approximately 108 s. These results indicate that the temporal representations employed by GRU-DEPPO help the policy adapt its cache decisions more efectively to abrupt changes in content-demand conditions.

Finally, we conduct an ablation study by comparing GRU-DEPPO with DEPPO, which removes the GRU-based temporal representation and directly provides the corresponding raw observations to the PPO policy. Fig. 7a shows the training curves of GRU-DEPPO and DEPPO. From approximately 1,200 episodes onward, GRU-DEPPO consistently maintains a higher mean episode reward, whereas DEPPO exhibits several periods of stagnation and performance degradation during training. GRU-DEPPO also eventually reaches a higher reward level. These results indicate that the GRU-based temporal representation helps the PPO policy more efectively exploit heterogeneous and timevarying network observations, thereby facilitating the learning of a better-performing caching policy. Figs. 7b,7c, and 7d compare GRU-DEPPO and DEPPO in terms of deadlinemiss rate, total communication energy consumption, and inter-UAV energy imbalance, respectively. DEPPO achieves a slightly better deadline-miss rate and lower inter-UAV energy imbalance than TLPPO, while exhibiting higher total communication energy consumption. However, clear performance gaps remain between DEPPO and GRU-DEPPO across the considered objectives.

![](images/aa19715192284475f31c7e6d53f36582414655965e2f50aac3de1b0408fbd1b9.jpg)  
(a) Episode reward

![](images/96d4c8c1352fee8ce45d435bac001201c1d3240b1411369af2300c7077d0ae1b.jpg)  
(b) Deadline-miss rate

![](images/451ecb659b5f2dbeaaf9597982a5a168cd2e7c6162dd6877436a706390a7f3b4.jpg)  
(c) Total communication energy

![](images/4dda2d08c49ee4a5544dfdcc64582f831033f5c369208a14649d01299073a91e.jpg)  
(d) Inter-UAV energy imbalance  
Figure 7: Ablation study of the GRU-based encoding

Although DEPPO employs the same PPO-based decision framework and jointly considers deadline satisfaction, communication energy, and inter-UAV energy imbalance, it directly relies on raw network observations without GRUbased temporal encoding. As a result, the policy is less efective at extracting temporal patterns from spatially heterogeneous and time-varying network conditions. In contrast, the GRU-based representation provides a compact temporal summary of recent network observations, enabling PPO to learn more efective cache decisions. These results demonstrate the contribution of the GRU-based temporal representation to the overall performance of GRU-DEPPO.

## 6. Conclusion

In this paper, we investigated deadline- and energyaware cache placement and replacement in dynamic multi-UAV edge networks with spatially heterogeneous and timevarying content demand. We first developed DE-UBAC as an interpretable hierarchical rule-based realization of the formulated caching objectives and used its limitations under dynamically coupled network conditions to motivate the proposed GRU-DEPPO framework. GRU-DEPPO employs local and global GRU-based temporal representations to capture the evolution of UAV-local and network-wide conditions and provides the resulting representations to a centralized PPO policy. Simulation results showed that GRU-DEPPO achieves the lowest deadline-miss rate and total communication energy consumption among the considered schemes while substantially reducing inter-UAV energy imbalance.

As future work, we plan to extend the proposed framework by jointly optimizing UAV positioning and content caching. While the present study focuses on cache placement and replacement under spatially heterogeneous and timevarying network conditions, adapting UAV positions may provide additional opportunities to improve connectivity, content accessibility, and communication-energy eficiency.

Table 4  
Training parameters of GRU-DEPPO
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td> $N _ { \mathrm { e p } }$ </td><td> $\overline { { 4 , 0 0 0 } }$ </td></tr><tr><td>PPO learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>PPO optimizer</td><td> $\mathsf { A d a m }$ </td></tr><tr><td> $B _ { \mathrm { r o l l } }$ </td><td>256</td></tr><tr><td> $B _ { \mathrm { m b } }$ </td><td>128</td></tr><tr><td> $K _ { \mathrm { P P O } }$ </td><td>3</td></tr><tr><td> $\gamma$ </td><td>0.99</td></tr><tr><td> $\epsilon _ { \mathrm { s r v } }$ </td><td>0.15</td></tr><tr><td> $\lambda _ { \mathrm { G A E } }$ </td><td>0.95</td></tr><tr><td> $\epsilon _ { \mathrm { c l i p } }$ </td><td>0.2</td></tr><tr><td> $c _ { H }$ </td><td>0.01</td></tr><tr><td> $c _ { V }$ </td><td>0.7</td></tr><tr><td>Actor hidden layers</td><td>(256,256)</td></tr><tr><td>Critic hidden layers</td><td>(512,256)</td></tr><tr><td> $L$ </td><td>10</td></tr><tr><td> $d _ { \mathrm { l o c } }$ </td><td>48</td></tr><tr><td>Local  $\mathsf { G R U }$  layers</td><td>1</td></tr><tr><td>Summary target dim.</td><td>11</td></tr><tr><td>Global GRU layers</td><td>1</td></tr><tr><td>Global token dimension</td><td>64</td></tr><tr><td> $d _ { \mathrm { g l o b a l } }$ </td><td>32</td></tr><tr><td>Pretraining policy</td><td>LFU</td></tr><tr><td>Pretraining optimizer</td><td> $\mathsf { A d a m W }$ </td></tr><tr><td>Pretraining learning rate</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Pretraining batch size</td><td>512</td></tr><tr><td>Pretraining epochs</td><td>30</td></tr><tr><td>λtype</td><td>1</td></tr></table>

Table 5  
Composition of the local GRU input
<table><tr><td>Feature group</td><td>Dimension</td></tr><tr><td>UAV features</td><td>20</td></tr><tr><td>Cache-type composition</td><td>|T|</td></tr><tr><td>Request-type composition</td><td>|T|</td></tr><tr><td>Deadline pressure</td><td>|T|</td></tr><tr><td>Total</td><td> $2 0 + 3 | \mathcal { T } |$ </td></tr></table>

## A. Training Parameters

For compactness, $B _ { \mathrm { m b } }$ denotes the PPO mini-batch size, $K _ { \mathrm { P P O } }$ the number of PPO optimization epochs, $d _ { \mathrm { l o c } }$ and $d _ { \mathrm { g l o b a l } }$ the hidden dimensions of the local and global GRUs, respectively. The simulator and GRU-DEPPO agent were implemented in Python using PyTorch and Stable-Baselines3/SB3-Contrib. Experiments were performed on an NVIDIA GeForce RTX 4090 GPU.

## B. GRU Input Composition

The local GRU input consists of UAV-level operating features and content-type-specific statistics. The 20- dimensional UAV feature vector summarizes the current association, cache, service, demand, and communicationenergy conditions of each UAV. In addition, the cachetype composition, request-type composition, and deadlinepressure vectors each contain || type-level entries. These features are derived from the current UAV state and historical service and demand observations available up to the current cache-decision epoch. Table 5 summarizes the resulting input composition.

Table 6  
Composition of the network-wide context vector
<table><tr><td>Feature group</td><td>Dimension</td></tr><tr><td>Network activity</td><td>3</td></tr><tr><td>Service quality</td><td>3</td></tr><tr><td>Service-mode composition</td><td>3</td></tr><tr><td>Communication energy</td><td>2</td></tr><tr><td>Demand variation</td><td>1</td></tr><tr><td>Inter-UAV energy imbalance</td><td>1</td></tr><tr><td>Total</td><td>13</td></tr></table>

In parallel with the local state encoding, the MBS constructs a compact network-wide context vector using the current network state and observations collected during the most recent cache-decision interval. The context vector contains three network-activity features, three service-quality features, three service-mode ratios, two communicationenergy features, one demand-variation feature, and one inter-UAV energy-imbalance feature. It is defined as

$$
{ \bf c } _ { \xi } ^ { \mathrm { n e t } } = \left[ { \bf g } _ { \xi } ^ { \mathrm { a c t } } ; { \bf g } _ { \xi } ^ { \mathrm { q o s } } ; { \bf g } _ { \xi } ^ { \mathrm { m o d e } } ; { \bf g } _ { \xi } ^ { \mathrm { e n e r g y } } ; v _ { \xi } ^ { \mathrm { d e m } } ; T _ { E , \xi } \right] \in \mathbb { R } ^ { 1 3 }\tag{39}
$$

The network-wide context consists of aggregate network activity, service-quality, service-mode, communication-energy, demand-variation, and inter-UAV energy-imbalance information available at the centralized controller.

Before being processed by the global GRU, each networkcontext vector is projected onto a $d _ { \mathrm { t o k } }$ -dimensional token space:

$$
\mathbf { e } _ { \xi } ^ { \mathrm { n e t } } = \phi _ { \mathrm { g l o b a l } } \left( \mathbf { c } _ { \xi } ^ { \mathrm { n e t } } \right) \in \mathbb { R } ^ { d _ { \mathrm { t o k } } } , d _ { \mathrm { t o k } } = 6 4\tag{40}
$$

The resulting token sequence is encoded by the global GRU according to (32).

## References

[1] W. Shi, J. Cao, Q. Zhang, Y. Li, L. Xu, Edge computing: Vision and challenges, IEEE Internet of Things Journal 3 (2016) 637–646.

[2] S. Dong, J. Tang, K. Abbas, R. Hou, J. Kamruzzaman, L. Rutkowski, R. Buyya, Task ofloading strategies for mobile edge computing: A survey, Computer Networks 254 (2024) 110791.

[3] M. Ergen, B. Saoud, I. Shayea, A. A. El-Saleh, O. Ergen, F. Inan, M. F. Tuysuz, Edge computing in future wireless networks: A comprehensive evaluation and vision for 6g and beyond, ICT Express 10 (2024) 1151–1173.

[4] J. Hong, Y.-G. Hong, X. de Foy, M. Kovatsch, E. Schooler, D. KUTSCHER, Internet of Things (IoT) Edge Challenges and Functions, RFC 9556, 2024. URL: https://www.rfc-editor.org/info/ rfc9556. doi:10.17487/RFC9556.

[5] A. A. Baktayan, A. T. Zahary, A. Sikora, D. Welte, Computational ofloading into UAV swarm networks: A systematic literature review, EURASIP Journal on Wireless Communications and Networking 2024 (2024) 69.

[6] X. Gu, G. Zhang, A survey on uav-assisted wireless communications: Recent advances and future trends, Computer Communications 208 (2023) 44–78.

[7] W. Wu, Z. Luo, Uav-enabled mobile edge computing: Key technologies, applications, and future challenges, Physical Communication 78 (2026) 103196.

[8] I. López-Villegas, E. A. Martínez-Rios, J. Izquierdo-Reyes, R. Bustamante-Bello, F. Falcone, A systematic literature review of emergency communications assisted by unnamed aerial vehicles, Ad Hoc Networks 182 (2026) 104063.

[9] Z. Chen, F. Wang, J. Wang, Joint optimization for service-caching, computation-ofloading, and uavs flight trajectories over rechargeable uav-aided mec using hierarchical multi-agent deep reinforcement learning, Vehicular Communications 50 (2024) 100844.

[10] T. Q. Duong, K. J. Kim, Z. Kaleem, M.-P. Bui, N.-S. Vo, Uav caching in 6g networks: A survey on models, techniques, and applications, Physical Communication 51 (2022) 101532.

[11] M. K. Somesula, M. R. Raju, M. R. Dorsala, B. Brahma, S. K. Mothku, Service caching and user association in cache enabled multiuav assisted men for latency-sensitive applications, Computers and Electrical Engineering 119 (2024) 109606.

[12] M. F. Pervej, R. Jin, S.-C. Lin, H. Dai, Eficient content delivery in user-centric and cache-enabled vehicular edge networks with deadline-constrained heterogeneous demands, IEEE Transactions on Vehicular Technology 73 (2024) 1129–1145.

[13] X. Li, J. Liu, X. Chen, J. Wang, M. Pan, Caching on the sky: A multiagent federated reinforcement learning approach for uav-assisted edge caching, IEEE Internet of Things Journal 11 (2024) 28213– 28226.

[14] Y. Sun, X. Zhong, F. Wu, X. Chen, S. Zhang, N. Dong, Multiuav content caching strategy and cooperative, complementary content transmission based on coalition formation game, Sensors 22 (2022).

[15] A. K. Bhuyan, H. Dutta, S. Biswas, Federated multi-armed bandit learning for caching in uav-aided content dissemination, Ad Hoc Networks 151 (2023) 103306.

[16] H. Guo, Y. Wang, J. Liu, C. Liu, Multi-uav cooperative task ofloading and resource allocation in 5g advanced and beyond, IEEE Transactions on Wireless Communications 23 (2024) 347–359.

[17] W. Wang, Y. Liu, Y. Dai, Y. He, Joint cooperative caching and power control for UAV-assisted internet of vehicles, Scientific Reports 14 (2024) 9341.

[18] T. Zhang, Y. Wang, Y. Liu, W. Xu, A. Nallanathan, Cache-enabling uav communications: Network deployment and resource allocation, IEEE Transactions on Wireless Communications 19 (2020) 7470– 7483.

[19] J. Kim, Y. Park, B. Suh, H. Choe, S. Park, K.-I. Kim, User behavioraware caching based on online adaptation in multi-uav assisted content delivery networks, Ad Hoc Networks 192 (2026) 104333.

[20] Y. Zhao, C. Liu, X. Hu, J. He, M. Peng, D. Wing Kwan Ng, T. Q. S. Quek, Joint content caching, service placement, and task ofloading

in uav-enabled mobile edge computing networks, IEEE Journal on Selected Areas in Communications 43 (2025) 51–63.

[21] Y. Zhu, Q. Zhu, Joint optimization algorithm for uav-assisted caching and charging based on wireless energy harvesting, Applied Sciences 15 (2025).

[22] W. Wang, X. Xu, M. Bilal, M. Khan, Y. Xing, Uav-assisted content caching for human-centric consumer applications in iov, IEEE Transactions on Consumer Electronics 70 (2024) 927–938.

[23] S. Anokye, D. Ayepah-Mensah, A. M. Seid, G. O. Boateng, G. Sun, Deep reinforcement learning-based mobility-aware uav content caching and placement in mobile edge networks, IEEE Systems Journal 16 (2022) 275–286.

[24] J. Chen, F. Tan, H. Chen, S. Li, Caching strategy and resource allocation in cache-enabled uav networks: A deep reinforcement learning approach, in: 2024 IEEE/CIC International Conference on Communications in China (ICCC), 2024, pp. 283–288. doi:10.1109/ ICCC62479.2024.10681702.

[25] Q. Zhu, J. Zheng, A. Jamalipour, Deep reinforcement learningbased uav base station deployment for content delivery in cellular iot networks, IEEE Internet of Things Journal 12 (2025) 23389–23401.

[26] A. Al-Hilo, M. Samir, C. Assi, S. Sharafeddine, D. Ebrahimi, A cooperative approach for content caching and delivery in uav-assisted vehicular networks, Vehicular Communications 32 (2021) 100391.

[27] 3GPP, Study on Enhanced LTE Support for Aerial Vehicles, Technical Report TR 36.777, 3rd Generation Partnership Project (3GPP), 2017. Release 15, pp. 33–46.

[28] A. Al-Hourani, S. Kandeepan, S. Lardner, Optimal lap altitude for maximum coverage, IEEE Wireless Communications Letters 3 (2014) 569–572.

[29] X. Zhang, B. Zhang, K. An, G. Zheng, S. Chatzinotas, D. Guo, Stochastic geometry-based analysis of cache-enabled hybrid satelliteaerial-terrestrial networks with non-orthogonal multiple access, IEEE Transactions on Wireless Communications 21 (2022) 1272–1287.

[30] M. M. Selim, Stochastic geometry analysis of uav-assisted networks with probabilistic uav activation, Scientific Reports 15 (2025) 37356.

[31] G. Yan, J. Li, Towards latency awareness for content delivery network caching, in: 2022 USENIX Annual Technical Conference (USENIX ATC 22), USENIX Association, Carlsbad, CA, 2022, pp. 789– 804. URL: https://www.usenix.org/conference/atc22/presentation/ yan-gang.

[32] M. Hajimirsadeghi, N. B. Mandayam, A. Reznik, Joint caching and pricing strategies for popular content in information centric networks, IEEE Journal on Selected Areas in Communications 35 (2017) 654– 667.

[33] L. pu, L. Jiao, X. Chen, L. Wang, Q. Xie, J. Xu, Online resource allocation, content placement and request routing for cost-eficient edge caching in cloud radio access networks, IEEE Journal on Selected Areas in Communications PP (2018) 1–1.

[34] C. Fan, X. Wu, B. Li, C. Zhao, Partial observation learning-based task ofloading and spectrum allocation in uav collaborative edge computing, Digital Communications and Networks 10 (2024) 1635– 1643.

[35] R. Wang, J. Zhang, S. H. Song, K. B. Letaief, Mobility-aware caching in d2d networks, IEEE Transactions on Wireless Communications 16 (2017) 5001–5015.

[36] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov, Proximal policy optimization algorithms, 2017. URL: https://arxiv.org/abs/ 1707.06347. arXiv:1707.06347.

[37] X. Zhong, F. Chen, Q. Guan, Fei Ji, H. Yu, On the distribution of nodal distances in random wireless ad hoc network with mobile node, Ad Hoc Networks 97 (2020) 102026.