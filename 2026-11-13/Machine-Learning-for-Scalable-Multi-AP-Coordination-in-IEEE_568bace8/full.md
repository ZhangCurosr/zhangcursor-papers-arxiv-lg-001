# Machine Learning for Scalable Multi-AP Coordination in IEEE 802.11bn Networks

Maksymilian Wojnar

AGH University of Krakow

Krakow, Poland

maksymilian.wojnar@agh.edu.pl

Szymon Szott

AGH University of Krakow

Krakow, Poland

szymon.szott@agh.edu.pl

Abstract—As Wi-Fi deployments become denser, fully distributed random channel access no longer guarantees the required quality of service. The multi-access point coordination (MAPC) framework introduced in IEEE 802.11bn lets neighboring access points (APs) cooperate, but choosing which APs transmit together and how to configure them remains an open, computationally hard problem. We develop and validate a stack of machine learning (ML) coordination methods, progressing from lightweight hierarchical multi-armed bandits (H-MABs) to a generative pipeline built on graph neural networks (GNNs) and flow matching (FM). The methods are evaluated in simulation and on a hardware testbed, improving throughput over legacy access and scaling coordination to dozens of APs where analytical and search-based methods become intractable.

Index Terms—IEEE 802.11bn; coordinated spatial reuse; flow matching; multi-armed bandits; neural networks; reinforcement learning.

## I. INTRODUCTION

The increasing density of Wi-Fi networks exposes the limits of fully distributed, contention-based channel access, which cannot ensure high and predictable service quality. In response, IEEE 802.11bn (Wi-Fi 8) introduces multi-access point coordination (MAPC) [1], letting neighboring access points (APs) share radio resources more efficiently. Among MAPC variants, our research targets coordinated spatial reuse (Co-SR), in which selected nodes transmit simultaneously on the same channel with carefully chosen transmit power, so that transmissions are successfully decoded while avoiding mutual interference. MAPC currently targets downlink transmissions only, and the standard enforces just the basic signaling: it leaves open the selection of the transmitting AP–station pairs and their transmit power. The modulation and coding scheme (MCS), although outside the standard, is another major component that determines the final performance. We call the joint selection of these parameters as the Co-SR scheduling problem, the central problem of our work. Crucially, the current IEEE 802.11bn specification limits Co-SR to pairs of APs, as coordinating quickly becomes prohibitively complex. Nevertheless, all the coordination methods we propose go beyond Wi-Fi 8, where any number of APs may cooperate. Such network-wide coordination is most valuable in large, centrally managed deployments, such as industrial, corporate, and institutional networks.

The problem is computationally hard and scales poorly: the number of possible configurations grows exponentially with the number of nodes in a network, and parameter selection must account for variable, hard-to-predict propagation. Existing Co-SR solutions fall into a few classes, each with a shared weakness: (i) measurement-driven heuristics and centralized schemes [2], [3], bound by costly signaling and unreliable received signal strength (RSS) measurements; (ii) iterative solver-based and adaptive coordination [4], bound by poor scaling; (iii) and learning-based schedulers [5], so far confined to small static downlink topologies. None establishes the theoretical performance limits of Co-SR or offers scalable machine learning (ML) methods verified in complex scenarios. Addressing these gaps, we study lightweight bandit-based schedulers that learn online, analytical performance bounds for Co-SR, and a generative one-shot pipeline that scales to dozens of APs. Our methods are available as open source<sup>1,2</sup>.

## II. METHODOLOGY

Established simulators, such as ns-3, lack MAPC support. Therefore, we build an open-source Monte Carlo Co-SR simulator [6], which computes the interference between nodes and the success probability of reception, thereby approximating the throughput achieved under parallel transmissions. This effective data rate, summed over all parallel transmissions, serves as the optimization objective. We first formulate Co-SR scheduling as a bandit problem and develop a hierarchical multi-armed bandit (H-MAB) whose levels select the transmitting APs, their receiving stations, and transmit power, with optional clustering to make the action space tractable in dense topologies [6]. To quantify achievable performance, we formulate mixed-integer linear programs (MILP) optimizing for throughput (T-Optimal) or fairness (F-Optimal) [7]. Finally, we represent MAPC topologies and configurations as graphs and encode them with a graph neural network (GNN) variational autoencoder [8]. A flow matching (FM) model, chosen for its high-quality few-step generation, then produces new configurations, and a surrogate rate predictor scores them without costly simulator calls [9]. Validation is threefold: against the analytical MILP upper bounds, in simulation (versus legacy DCF and IEEE 802.11ax spatial reuse) across static and dynamic topologies, and on a real openwifi testbed [10], whose digital twin serves for preliminary hyperparameter tuning.

## III. SELECTED RESULTS

We first model Co-SR group selection as a bandit problem and solve it with H-MAB [6]. Within seconds, the agents learn which AP–station pairs perform best in a given topology and adapt to changes such as station mobility. The hierarchy outperforms a flat MAB by rejecting incompatible APs at a higher decision level, and UCB emerges as the best general-purpose underlying algorithm. Extending H-MAB with transmit-power selection [7] raises aggregate throughput over DCF by 80% on average in random scenarios, without reducing per-station transmission opportunities, and outperforms 802.11ax spatial reuse. For the densest office topologies, clustering the network drastically shortens convergence time. On a real openwifi testbed [10], the agents adapt to real radio conditions.

We then study representation learning for MAPC configurations [8], modeling the network as a heterogeneous graph and training a GNN variational autoencoder whose loss accounts for missing attributes. Discrete attributes such as MCS and transmission power are reconstructed with near 100% accuracy, while continuous ones (RSS and success probability) are inherently harder, reaching 15–50% accuracy. Moreover, the embeddings are semantically organized, therefore, we use them as a base for our FM4WiFi generative pipeline [9], which reformulates Co-SR scheduling as a conditional generation problem. The pipeline consists of a GNN autoencoder, an FM sampler, and a surrogate rate predictor. FM4WiFi jointly optimizes the selection of transmitters (APs), receivers (stations), MCS and transmit power levels. It produces a schedule in a single inference step in under one second and scales beyond 30 APs, where analytical and reinforcement learning (RL) methods become infeasible. Since only FM4WiFi jointly selects the MCS, Fig. 1 gives each method an ideal MCS for a fair comparison. FM4WiFi matches or exceeds the effective data rate of the strongest baselines at medium-tolarge scale, and keeps improving past the network sizes where H-MAB and the analytical bounds hit their resource limits. In dynamic scenarios, FM4WiFi increases the effective data rate by roughly 25% over a static configuration with regeneration triggered by network events or at cyclic intervals.

H-MAB and FM4WiFi complement each other, and both coordinate much more than two APs as in Wi-Fi 8. H-MAB is lightweight and well suited to small and medium networks, while the FM4WiFi provides the sub-second decisions and scalability needed for large topologies. The T-Optimal and F-Optimal upper bounds serve as analytical reference for both. Beyond the referenced publications, the H-MAB and FM4WiFi results have been presented to the IEEE 802.11 AIML Topic Interest Group [11], [12], supporting ongoing Wi-Fi standardization efforts.

## IV. DISCUSSION AND FUTURE WORK

Overall, this work establishes ML as a practical and scalable foundation for Co-SR coordination beyond Wi-Fi 8, combining lightweight online agents, analytical performance bounds, and a scalable generative pipeline validated through analysis, simulation, and hardware. Several directions remain open. We plan to add RL to the FM4WiFi pipeline to further increase performance and enable online adaptation at a large scale. We are also developing approximate performance bounds that scale better than the MILP formulation. Finally, we are exploring optimization driven directly by the surrogate model.

![](images/af1397419dbc959461385bedce5363a5b7b0139fb66d9d6d344d2ba14d2c070b.jpg)  
Fig. 1. Effective data rate versus number of APs in a static residential (office-grid) scenario, with every method selecting APs, AP–station pairs, and transmit power under ideal MCS.

## REFERENCES

[1] L. Galati-Giordano, G. Geraci, M. Carrascosa, and B. Bellalta, “What Will Wi-Fi 8 Be? A Primer on IEEE 802.11bn Ultra High Reliability,” IEEE Communications Magazine, vol. 62, no. 8, pp. 126–132, 2024.

[2] J. Haxhibeqiri et al., “Coordinated Spatial Reuse for WiFi Networks: A Centralized Approach,” in Proc. IEEE WFCS, 2024, pp. 1–8.

[3] D. Nunez, M. Smith, and B. Bellalta, “Multi-AP Coordinated Spatial Reuse for Wi-Fi 8: Group Creation and Scheduling,” in Proc. MedCom-Net, 2023, pp. 203–208.

[4] D. Zhu, L. Wang, G. Pan, and S. Luan, “Two Enhanced Schemes for Coordinated Spatial Reuse in IEEE 802.11be: Adaptive and Distributed Approaches,” Computer Networks, vol. 258, p. 111060, 2025.

[5] J. Jung, G. Lee, and J.-M. Chung, “Coordinated Spatial Reuse With Shared AP Index Optimization for IEEE 802.11be WLAN Enhance ment,” IEEE Transactions on Network Science and Engineering, vol. 13, pp. 930–947, 2026.

[6] M. Wojnar et al., “IEEE 802.11bn Multi-AP Coordinated Spatial Reuse With Hierarchical Multi-Armed Bandits,” IEEE Communications Letters, vol. 29, no. 3, pp. 428–432, 2025.

[7] M. Wojnar et al., “Coordinated Spatial Reuse Scheduling With Machine Learning in IEEE 802.11 MAPC Networks,” IEEE Journal on Selected Areas in Communications, vol. 43, no. 11, pp. 3666–3682, 2025.

[8] M. Wojnar, “Uczenie reprezentacji sieci IEEE 802.11 z wykorzystaniem grafowych sieci neuronowych [Representation learning of IEEE 802.11 networks using graph neural networks],” Przegl ˛ad Telekomunikacyjny, Wiadomosci Telekomunikacyjne´ , vol. 98, pp. 263–266, 2025, in Polish.

[9] M. Wojnar, K. Rusek, K. Kosek-Szott, and S. Szott, “FM4WiFi: Flow Matching for Multi-AP Coordination in Dense Deployments of Beyond Wi-Fi 8 Networks,” 2026. [Online]. Available: https: //arxiv.org/abs/2608.04050

[10] X. Jiao et al., “openwifi: a free and open-source IEEE802. 11 SDR implementation on SoC,” in 2020 IEEE 91st Vehicular Technology Conference (VTC2020-Spring). IEEE, 2020, pp. 1–2.

[11] S. Szott et al., “MAPC C-SR Scheduling with Multi-Armed Bandits Revisited,” IEEE 802.11-25/0710r1, May 2025.

[12] S. Szott et al., “MAPC Co-SR Scheduling with Machine Learning Revisited Again,” IEEE 802.11-26/0795r0, May 2026.