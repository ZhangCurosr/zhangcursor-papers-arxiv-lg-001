# Autonomous Cyber Operations Gym for Training and Evaluating AI Models for Defence

Tobias Hürten¹, Johannes F. Loevenich1,2, Florian Spelter1, Erik Adler1,3,

Laurin Holz1,4, Maxime Schwarzer1,3, Thies Moehlenhof1, and Roberto Rigolin F. Lopes1 1CortAIx Labs, Thales Deutschland, Ditzingen, Germany

2Department of Mathematics/Computer Science, University of Osnabrück, Osnabrück, Germany

3Department of Informatics, Karlsruhe Institute of Technology (KIT), Karlsruhe, Germany

4Department of Computer Science, University of Ulm, Ulm, Germany

Email: {tobias.huerten, johannes.loevenich, florian.spelter, erik.adler, laurin.holz,

maxime.schwarzer, thies.moehlenhof, roberto.rigolin} @thalesgroup.com

Abstract—This paper proposes an Autonomous Cyber Operations (ACO) gym, a unified framework integrating highfidelity network emulation, realistic user behaviour modelling for benign network traffic and adversarial profiles, and the generation of cybersecurity scenarios through open toolkits. By supporting live and deferred feature extraction at the packet and flow levels, the gym facilitates the creation of labelled synthetic datasets for training Deep Neural Network (DNN)- based Intrusion Detection Systems (IDSs), as well as developing and evaluating modular components for autonomous defence agents through emulated interactions. While evaluation of a DNN classifier within the gym shows high in-domain performance, achieving near-perfect recall on adversarial network flows, our experiments reveal a significant sensitivity to domain shift, showing that even minor changes in network composition, such as variations in traffic protocol distribution, can lead to a catastrophic collapse in model transferability and detection performance.

Index Terms—Autonomous Cyber Defence, Network Emulation, User Simulation

## I. INTRODUCTION

The fundamental reshaping of the cybersecurity landscape by the increasing volume, speed, and sophistication of cyberattacks motivates a shift toward Autonomous Cyber Defence (ACD), where Artificial Intelligence (AI) systems, such as Deep Reinforcement Learning (DRL) [1] or Large Language Models (LLMs) [2], [3], support detection and timely response in complex environments [4]. To this end, this work focuses on the design of an extendable environment that supports key Software-Defined Defence (SDD) principles to emulate realistic network topologies and user behaviour models, as well as the evaluation of an Intrusion Detection System (IDS) under protocol-distribution shift.

One of the challenges of robust ACD agents is the lack of realistic and rigorous training/testing environments [4]. Effective ACD solutions, whether Deep Neural Network (DNN)-based IDSs or DRL-based autonomous defence agents, require vast quantities of high-fidelity, labelled network traffic data (either as aggregated datasets or through live interactions) and a controlled, reproducible environment. Existing platforms often fail to balance high fidelity with scalability, restrict functionality to solely offensive or defensive research, or lack realistic benign user traffic [5]. Accordingly, this work focuses on high-fidelity environment design and baseline IDS evaluation as a step toward enabling full ACD systems, addressing fidelity, data generation, and configurability gaps.

The main contributions are:

• A modular Automated Cyber Operation (ACO) gym integrating emulation, stochastic user behaviour modelling, and synthetic, labelled data generation.

• A dual-pipeline framework for both live and deferred network feature extraction.

• An empirical analysis of model robustness and crossdomain generalization of an IDS under protocoldistribution shift.

The remainder of the paper is structured as follows: Section II reviews related work regarding ACD, the role of cyber ranges in ACO gyms, and behavioural modelling. Section III defines the methodology, including the system architecture, user simulation framework, scenario topology, and threat landscape. Next, Section IV outlines the experimental setup, provides a statistical analysis of the generated synthetic datasets, and evaluates a DNN-based IDS with respect to in-domain performance and transferability under cross-evaluation. Finally, Section V concludes the paper and outlines directions for future work.

## II. RELATED WORK

The paradigm shift from conventional automation to ACD is necessitated by the increasing volume and complexity of AI-powered cyber threats [2]. Unlike traditional security orchestration, which relies on static, fixed-purpose, rulebased logic, ACD emphasises autonomous, adaptive, sequential decision-making agents capable of detecting and mitigating complex attacks without direct human tasking [4]. This transition addresses the limitations of manual intervention and static response systems in the face of modern threat scales [5]. To ensure reliability, particularly within critical infrastructure, current literature identifies several high-level requirements for trustworthy ACD: adaptability to evolving landscapes, auditability for transparent and explainable decision-making, directability for human-in-the-loop intervention, observability for situational awareness, security against adversarial Machine Learning (ML) attacks, and transferability across heterogeneous real-world environments [6]. Closely related is the concept of Automated Cyber Operation (ACO), which involves the concurrent deployment of automated red and blue agents within a networked environment. ACO serves as a critical framework for simulating the competitive “cyber arms race", particularly where network scale or communication constraints render human-led operations infeasible [4].

ACO gyms serve as a critical framework to bridge the “deployment gap" and ensure that ACD solutions meet the rigorous standards of critical infrastructure, such as North Atlantic Treaty Organization (NATO) Protected Core Networking (PCN) [2], [4]. These gyms provide controlled experimental environments built upon Cyber Ranges (CRs): isolated, interactive platforms designed to replicate critical, enterprise, and civilian network infrastructures for cybersecurity research and education [7], [8]. CR architectures are broadly categorised into simulated and emulated environments. While existing platforms like CybORG [9] provide robust Reinforcement Learning (RL) environments, they often rely on abstract, simulated CRs for their high scalability and costeffectiveness [4], [10], [11]. In contrast, our approach utilises an emulated CR executing authentic software on virtualised hardware to prioritise granular application-level behaviours and evaluation under protocol-distribution shift, critical factors for real-world IDS reliability. Although more resource-intensive and complex to deploy, emulated CRs are preferred for highstakes experimentation, as their increased realism significantly narrows the deployment gap compared to pure simulation [4], [10], [11].

High-fidelity CRs within the context of ACO environments necessitate the generation of realistic, non-deterministic user activity to establish background noise as operational baselines, simulating human-centric failure points, such as phishing, as well as sophisticated adversarial behaviour like nondeterministic multi-stage attack chains [5], [6], [12], [13]. Current literature identifies several modelling paradigms to address these requirements. Statistical Modelling (SM) replicates technical traffic properties like inter-packet times and payload sizes as random variables but is constrained by its reliance on pre-existing empirical data [14]. Conversely, Cognitive Modelling (CM) captures behavioural intent and psychological factors to predict decision-making biases [15], while Agent-Based Modelling (ABM) simulates emergent macro-level interactions, such as malware propagation across diverse topologies [16]. The emergence of Generative AI (GenAI) further enhances these capabilities by synthesising complex, non-repetitive communication and activity patterns, thereby bypassing the privacy and availability constraints inherent in real-world datasets [17]. Integrating these methodologies, frameworks like GHOSTS utilise autonomous Non-Player Characters (NPCs) to achieve role-related objectives through the actuation of real software applications rather than mere simulation. This approach ensures that generated network artifacts, including headers, timing, and application-layer details, maintain the high fidelity required for effective cyber warfare training and the evaluation of autonomous defence agents [18].

![](images/4d4e70590ead8002b65b532d4882f011cd98ba8814f0d2a3bbac4f698bdf97ee.jpg)  
Fig. 1: Architecture of the ACO gym.

## III. METHODOLOGY

## A. Gym Architecture

Adhering to SDD principles, the ACO gym's modular architecture decouples hardware, software, and data, providing an extensible framework for high-fidelity emulation of civilian and critical infrastructure through user-defined scenarios.

The system architecture is structured into multiple layers, as illustrated in Fig. 1. The core component is a network emulation built upon Kathará [19] that leverages Docker for resource-efficient containerization. This setup allows for the flexible configuration and instantiation of numerous network nodes, acting as various Software Defined Networking (SDN) components, such as switches, routers, controllers, hosts, and servers. The entire emulation environment is virtualised to isolate system and hardware dependencies, guaranteeing compatibility, flexibility, and configurability. This encapsulation permits the parallel execution of multiple scenario emulations without mutual interference, thereby facilitating scalable data generation.

A management system serves as the control plane, offering an interface for managing emulation lifecycles and enabling manual control of individual emulated nodes. This system provides the access point for future DRL agents to perform decentralised administrative network actions, which effectively decouples the development and training of ACD systems from the emulation itself.

## B. User Simulation

To simulate high-fidelity network traffic for training ACD systems, we developed Droids, a lightweight, modular task-scheduling framework inspired by works such as GHOSTS [18]. Droids employs a hierarchical, YAML-based configuration to orchestrate complex automation timelines through a taxonomy of three distinct job classes: Composite

Jobs provide flow control and concurrency, Stochastic Jobs introduce non-deterministic variability to mimic the unpredictability of human actors, and Primitive Jobs execute the low-level actions required to generate authentic system and network telemetry.

In practice, this modularity enables reproducible yet stochastic modelling of heterogeneous agents via YAMLparametrised profiles (e.g., activity frequency, Server Message Block (SMB)/Hypertext Transfer Protocol (HTTP) protocol mix, and timing jitter). In our case study, benign users are simulated using layered, concurrent workflows that incorporate web crawling and probabilistic file server interactions, while adversarial agents employ the same stochastic building blocks to execute “low-and-slow" exploits. By integrating these varied job types, Droids generates the concurrent HTTP, SMB, and system-level traffic with stochastic timing and mixed benign/adversarial activity to quantify environmental fidelity.

## C. Network Topology

![](images/8287a67fdf085887ca4dc97b5f62f7182fd1327a3d2757327328c2bea760d51e.jpg)  
Fig. 2: Exemplary network topology of an enterprise cybersecurity scenario.

The emulated enterprise environment, illustrated in Fig. 2, comprises distinct WAN and LAN segments. The internal LAN is partitioned into two subnets: an outer subnet (10.0.101.0/24) hosting five authorised users and two pre-compromised hosts, and an isolated inner subnet (10.0.102.0/24) providing critical services such as a Domain Name System (DNS) server, internal web services, and a file server (Samba). The external WAN includes an adversarial Command and Control (C2) server, standard internet services (Internet Service Provider (ISP), DNS), and two public web servers.

Routing and control follow the SDN paradigm, utilising a Floodlight controller and OpenFlow via Open vSwitch. Internal routing employs the Open Shortest Path First (OSPF) protocol, while the Border Gateway Protocol (BGP) manages external traffic, intentionally excluded from the inner subnet to ensure isolation. Routers are implemented via the Free Range Routing (FRR) suite, with firewalls strictly blocking external traffic to the internal segment. Realistic traffic conditions and entity behaviours (authorized and adversarial) are generated using Traffic Control (TC) and the Droids framework, respectively. The scenario definition follows Kathará's configuration schema with general connectivity, as well as host-based start-up and configuration specifications.

## D. Threat Analysis

The illustrative scenario models an assumed breach scenario: a sophisticated internal threat originating from a precompromised host in the outer LAN subnet. The primary attack vector targets a Server Side Template Injection (SSTI) vulnerability in the internal Flask web servers. Beyond basic reconnaissance and Samba-based data theft, the adversary executes a complex, multi-stage exfiltration chain: (1) the adversary first maps the network to identify connectivity between the compromised host and internal targets, (2) then exploits the SSTI vulnerability to achieve Remote Code Execution (RCE) on the target web server and stage sensitive system configurations or password hashes into temporary files, (3) subsequently initiates lateral movement by forcing the victim server to transfer these staged files back to the attacker's foothold, (4) tunnels this collected data externally to the designated C2 server, and (5) finally executes a cleanup exploit to delete all temporary files and logs to obfuscate the intrusion. This sequence creates distinct network signatures that are identifiable through holistic analysis but remain difficult to detect when viewed as isolated events.

## IV. RESULTS

## A. Experimental Setup

1) Data Collection and Processing: To address the diverse requirements of ML algorithms and real-time operational feedback, our methodology employs a dual data pipeline that allows for live (A) and decoupled data processing (B), as illustrated in Fig. 3. In this figure, the first pipeline (A) is dedicated to real-time monitoring and visualisation in dashboards, as well as serving as the foundation for DRL action-observation loops, necessitating minimal latency. Captured network traffic is immediately processed to extract features into a ring buffer in CSV format. Telegraf, a data acquisition agent, periodically collects and aggregates these features into a time-series InfluxDB database. Grafana is used to organise the data in insightful dashboards, providing realtime analysis capabilities for emulation runs.

The second pipeline (B) focusses on deferred batch processing for generating the final synthetic datasets. All captured IP traffic is collected in batches of PCAP files, which are aggregated and processed post-emulation. This batch approach enables flexible experimentation with different feature sets without the computational overhead of re-running the entire emulation.

![](images/ecce01158c9bfad3ec07ff220662f64da502b2da53426b6608380f2322481855.jpg)  
Fig. 3: Dual data processing pipeline, supporting live (A) and decoupled (B) data processing.

The data processing pipeline integrates tshark and NexusFlowMeter to extract 26 features, including transport metrics, directionality, and payload statistics. To ensure a robust dataset, missing values are addressed via median-based imputation, and outliers are clipped at the 99.99th percentile. Ground-truth labels are automatically derived from custom HTTP headers at the packet-level and propagated to flows using a worst-case aggregation strategy. Next, the categorical labels are numerically encoded. Finally, the data is partitioned into a stratified 80/20 train-test split and normalized via z-score standardization to facilitate stable model convergence.

2) Datasets: We generated four distinct datasets based on variations of the scenario defined in Section III. These variations serve as a testbed for evaluating the robustness and transferability of the trained classifier against different forms of data drift and class imbalance.

For the baseline dataset (a), we selected an emulation duration of 3 hours and a behavioural parametrisation with respect to delays between actions that would result in a higher class-balance, making the dataset more suitable for use in ML algorithms. To quantify the transferability of a model trained on this artificially balanced dataset under domain and balance shifts, we generated three additional versions: (1) Dataset b: Employs more realistic delays, resulting in naturally sparser network traces and a significantly smaller fraction of malicious flows. (2) Dataset c: Extends the baseline emulation to 5 hours, resulting in a larger control set. (3) Dataset d: Introduces a domain-shift barrier for the trained model. This version is heavily dominated by benign User Datagram Protocol (UDP) traffic resulting from unresolved DNS queries to the internal DNS server in order to send browser telemetry data to an external API. The resulting protocol and class distributions are listed in Table I and discussed in Section IV-B1.

3) DNN Classifier: The experiments employ a DNN-based classification model tailored for multi-class network traffic categorization. A 26-dimensional input vector is processed through four fully connected layers (256, 128, 64, and 32 neurons) utilizing Rectified Linear Unit (ReLU) activation and a tapering dropout strategy (0.3 to 0.1) to mitigate overfitting. The five-neuron output layer employs a Softmax activation function, with the model optimized via Sparse Categorical Cross-Entropy loss. Training was conducted for a maximum of 20 epochs using the Adam optimizer with an initial learning rate of 0.0015, a batch size of 32, and a 0.1 validation split. To ensure stable convergence, the data were shuffled before each epoch, and the regime incorporated an early stopping mechanism (patience of five epochs, restoring optimal weights) and a learning rate scheduler that reduced the rate by a factor of 0.2 upon validation loss stagnation, down to a minimum of 0.001.

## B. Experimental Results

1) Dataset Analysis: The analysis of both class and protocol distributions across the four dataset iterations (Table I) reveals significant variance in flow volume and composition. Most notably, dataset d is characterised by DNS flooding, which accounts for roughly 216,000 out of 238,000 flows, resulting in 90.48% UDP dominance. In contrast, versions a, b, and c remain characterised by predominantly Transmission

TABLE I: Protocol and class distribution of each dataset.
<table><tr><td>Protocol</td><td>a</td><td>b</td><td>c</td><td>d</td></tr><tr><td>ARP</td><td>0.103%</td><td>0.135%</td><td>0.063%</td><td>0.013%</td></tr><tr><td>ICMP</td><td>0.048%</td><td>0.042%</td><td>0.025%</td><td>0.013%</td></tr><tr><td>IGMPv3</td><td>0.021%</td><td>0.025%</td><td>0.012%</td><td>0.003%</td></tr><tr><td>OSPF</td><td>0.027%</td><td>0.034%</td><td>0.016%</td><td>0.003%</td></tr><tr><td>TCP</td><td>79.452%</td><td>81.820%</td><td>79.887%</td><td>9.491%</td></tr><tr><td>UDP</td><td>20.349%</td><td>17.943%</td><td>19.996%</td><td>90.476%</td></tr><tr><td colspan="5">Class</td></tr><tr><td>Benign</td><td>90.756%</td><td>98.501%</td><td>90.653%</td><td>98.883%</td></tr><tr><td>Malicious</td><td>9.244%</td><td>1.499%</td><td>9.347%</td><td>1.117%</td></tr><tr><td>Clean-up</td><td>1.661%</td><td>0.271%</td><td>1.661%</td><td>0.203%</td></tr><tr><td>Staging</td><td>1.675%</td><td>0.271%</td><td>1.686%</td><td>0.205%</td></tr><tr><td>Lateral-movement</td><td>4.118%</td><td>0.627%</td><td>4.214%</td><td>0.500%</td></tr><tr><td>Exfiltration</td><td>1.791%</td><td>0.330%</td><td>1.786%</td><td>0.209%</td></tr></table>

![](images/288d5e281cceae40c9239746097854474699f14df947ad9fcf1f28f16da7bd12.jpg)  
Fig. 4: Correlation matrix using Spearman's rank correlation coefficient (ρ) (numerical features).

Control Protocol (TCP) traffic (79.45%-81.82%). The baseline a contains a total of 29,100 flows, while c expands this to 48,900 flows, enabling the evaluation of transferability to larger data corpora. Dataset a (and by extension the larger variation c) improves the class-balance from 98.50%-98.88% benign traffic (d and b) to 90.76% (90.65% for c). Malicious flows are distributed more evenly across their respective sub-classes, with the exception of lateral movement, which encompasses 4.12% of all flows.

Descriptive statistical analysis of the feature distribution reveals a highly heterogeneous, and heavy-tailed distribution, characteristic of real-world network traffic, where a minority of long-duration, high-volume flows coexist with a vast majority of brief, transactional connections. This disparity is most evident in the profound divergence between central tendency measures. For instance, while the mean flow duration reaches 41.161 seconds, the median (50th percentile) remains a mere 0.047 seconds, indicating that over 75% of traffic is extremely short-lived $( Q _ { 3 } = 0 . 1 3 1$ seconds). Similar positive skewness is observed across volumetric metrics and inter-arrival times, where rare extreme values substantially inflate the mean relative to the median.

Feature correlation was quantified using Spearman's rank correlation coefficient (ρ). The resulting heatmap in Fig. 4 reveals significant multicollinearity $( \rho \approx 1 . 0 )$ within clusters of volumetric metrics, such as total counts and their respective directional components, as well as between mean and maximum packet sizes. In contrast, features such as the average inter-arrival time and transmission rates expectedly exhibit a negative correlation $( \rho ~ \approx ~ - 1 . 0 )$ , reflecting the physical nature that higher throughput typically coincides with lower latencies. Finally, connection state indicators, such as TCP flag counts, show near-zero correlation with general flow metrics, confirming their utility as independent variables.

![](images/0286d07f392187f1930f7099f6846f0a69a4985fbbb7272dff23a595ce120362.jpg)  
(a) Cross Evaluation.

![](images/5c515d1926af39ef5cc8f60d7aea00d9d47a7fc6b2e4b4e4c08bd312e9b0704a.jpg)  
(b) Confusion Matrix.

![](images/9685de40a73c5e39900719ebfee461ea300526890c9b1addc70bbc7d07424bbe.jpg)  
(c) Relative F1 score.

![](images/bba8ef2c68682ad4373d8fb1b553d9422900adbebf115b885c741dceae111d31.jpg)  
(d) False negative ratio.  
Fig. 5: Cross-evaluation results: Performance with respect to the F1 score for each model under cross-evaluation (5a), the confusion matrix of the baseline classifier under in-domain evaluation (5b), relative F1 score of the baseline classifier when transferred to shifted datasets (5c), and the aggregated false negatives under cross-evaluation (5d).

2) Model Evaluation: To assess cross-domain generalization under systematic protocol-distribution shift, we trained a model on each dataset and evaluated its performance on all dataset variants, including the one used for training. Performance is quantified by the F1 score to assess the classification quality across the five classes benign', staging', lateral-movement', exfiltration', and clean-up’with respect to the True Positives (TPs), False Positives (FPs), and False Negatives (FNs).

Fig. 5a illustrates the F1 score under cross-evaluation for each training-evaluation combination as a heatmap. The diagonal shows high in-domain performance, with F1 scores of 0.88 to 0.85 for the balanced dataset versions a and c, and slightly lower scores of 0.78 to 0.74 for the less balanced datasets b and d, indicating the benefit of utilising the controlled environment to generate artificially balanced datasets for ML applications. Fig. 5b presents the confusion matrix of the baseline classifier as a heatmap. The model achieves perfect recall of 1.0 on benign flows and near perfect recall on the malicious classes exfiltration' and lateralmovement' (1.0 and 0.996). For flows that belong to the staging or clean-up stages of the attack chain, the model exhibits noticeably lower recall of 0.765 and 0.629, respectively. Specifically, these two classes are commonly mixed up by the model due to the similarity of their flow statistics, as both comprise a small payload consisting of short system-code, with

21.4% of the flows associated with the staging phase being falsely attributed to the clean-up stage (conversely, 27.8% of ‘clean-up' flows are misclassified as ‘staging').

Fig. 5c compares the relative performance with respect to the F1 score of the baseline classifier when evaluated on each dataset variation. Noticeably, performance transfers well to datasets b (more realistic action delays) and c (larger dataset size), with relative scores of 0.93 and 0.94 and corresponding F1 scores of 0.82 and 0.83 respectively (Fig. 5a). Performance drops significantly under domain shift where protocol distributions (e.g., ratio of UDP to TCP) deviate from the training baseline. Specifically, the DNS dominated dataset d imposes a domain-shift barrier, with a relative performance drop of 69% (F1 score of 0.27), showcasing that the vastly different dataset distribution leads to significantly differing decision boundaries learnt by the neural network.

Finally, Fig. 5d depicts the aggregate false negatives (instances where malicious flows were predicted as benign') The baseline classifier again demonstrates high transferability to the control datasets b and c, even reducing the number of false negative predictions from 0.7% to 0.0% for version b compared to a false negative rate of 1.5% for the indomain evaluation of the classifier trained on dataset b directly. These results suggest that a classifier can be trained on artificially balanced datasets optimised for ML algorithms while maintaining high transferability to more realistic data distributions.

## V. CONCLUSION

This study evaluated the efficacy of ACO gyms for the development and validation of ACD solutions, specifically examining DNN-based classifiers trained on synthetic, emulated datasets. While the framework provides a robust baseline, our results reveal a significant “transferability gap" when models are subjected to protocol-shifted variations, such as DNS/UDP flooding. We identify domain shifts as a primary performance barrier, with F1 score dropping to 0.27 when network compositions deviate substantially from training distributions. These findings underscore the necessity for ACO frameworks to systematically integrate diverse real-world stressors to ensure the operational reliability of automated defences in critical infrastructure.

To address these limitations, future work will focus on a systematic comparison across a broader range of baseline models beyond DNNs to isolate architecturespecific vulnerabilities. Furthermore, we intend to validate our synthetic, emulated datasets against diverse real-world benchmarks to bridge the gap between controlled environments and live production traffic. Furthermore, we will transition from static threat modelling toward automated attack-chain generation using AI-driven agents. Coupled with hybrid architectures, integrating statistical flow analysis, LLM-based Deep Packet Inspection (DPI), and host-level telemetry such as system logs and process information, this approach will provide the multi-layered visibility required to secure live production environments against diverse and non-deterministic threats.

## REFERENCES

[1] R. S. Sutton, A. G. Barto et al., Reinforcement learning: An introduction. MIT press Cambridge, 1998, vol. 1, no. 1.

[2] J. F. Loevenich, E. Adler, T. Huerten, and R. R. F. Lopes, “"Design and evaluation of an Autonomous Cyber Defence agent using DRL and an augmented LLM," Computer Networks, vol. 262, p. 111162, 2025. [Online]. Available: https://doi.org/10.1016/j.comnet.2025.111162

[3] T. Huerten, J. F. Loevenich, F. Spelter, E. Adler, J. Braun, L. Moxon, Y. Gourlet, T. Lefeuvre, and R. R. F. Lopes, “Hierarchical Multi-Agent Reinforcement Learning for Autonomous Cyber Defense in Coalition Networks," in MILCOM 2024 - 2024 IEEE Military Communications Conference (MILCOM), Washington, USA, 2024, pp. 176–181.

[4] S. Vyas, J. Hannay, A. Bolton, and P. P. Burnap, "Automated cyber defence: A review," arXiv preprint arXiv:2303.04926, 2023.

[5] A. Velazquez, J. F. Loevenich, T. Hürten, K. Wrona, P. H. Rettore, V. Boshnakov, F. Free-Nelson, T. Braun, and R. R. F. Lopes, "Cyber operations gyms to train autonomous cyber defense agents for nato," in 2025 International Conference on Military Communication and Information Systems (ICMCIS). IEEE, 2025, pp. 1–10.

[6] A. Lohn, A. Knack, A. Burke, and K. Jackson, "Autonomous cyber defense," A roadmap from lab to ops. Online. Centre for Emerging Technology and Security (CETaS) at The Alan Turing Institute, 2023.

[7] J. Davis and S. Magrath, “A survey of cyber ranges and testbeds," 2013.

[8] P. Lillemets, N. Bashir Jawad, J. Kashi, A. Sabah, and N. Dragoni, “A systematic review of cyber range taxonomies: Trends, gaps, and a proposed taxonomy," Future Internet, vol. 17, no. 6, p. 259, 2025.

[9] M. Standen, M. Lucas, D. Bowman, T. J. Richer, J. Kim, and D. Marriott, “Cyborg: A gym for the development of autonomous cyber agents," in IJCAI-21 1st International Workshop on Adaptive Cyber Defense, 2021, arXiv preprint.

[10] H. Kavak, J. J. Padilla, D. Vernon-Bido, S. Y. Diallo, R. Gore, and S. Shetty, "Simulation for cybersecurity: state of the art and future directions," Journal of Cybersecurity, vol. 7, no. 1, p. tyab005, 2021.

[11] A. Molina-Markham, C. Miniter, B. Powell, and A. Ridley, "Network environment design for autonomous cyberdefense," arXiv preprint arXiv:2103.07583, 2021.

[12] J. F. Loevenich, E. Adler, T. Hürten, F. Spelter, D. Roncevic, and R. R. F. Lopes, “Automating cyber threat intelligence and attack chain generation using cyber security knowledge graphs and large language models," in 2025 International Conference on Military Communication and Information Systems (ICMCIS). IEEE, 2025, pp. 1–10.

[13] I. Sharafaldin, A. H. Lashkari, A. A. Ghorbani et al., "Toward generating a new intrusion detection dataset and intrusion traffic characterization." ICISSp, vol. 1, no. 2018, pp. 108–116, 2018.

[14] A. Botta, A. Dainotti, and A. Pescapé, “A tool for the generation of realistic network workload for emerging networking scenarios," Computer Networks, vol. 56, no. 15, pp. 3531–3547, 2012.

[15] V. D. Veksler, N. Buchler, C. G. LaFleur, M. S. Yu, C. Lebiere, and C. Gonzalez, “Cognitive models in cybersecurity: learning from expert analysts and predicting attacker behavior," Frontiers in Psychology, vol. 11, p. 1049, 2020.

[16] A. Vestad and B. Yang, “A survey of agent-based modeling for cybersecurity," Human Factors in Cybersecurity, 2024.

[17] L. Wang, J. Zhang, H. Yang, Z.-Y. Chen, J. Tang, Z. Zhang, X. Chen, Y. Lin, H. Sun, R. Song et al., "User behavior simulation with large language model-based agents," ACM Transactions on Information Systems, vol. 43, no. 2, pp. 1–37, 2025.

[18] D. Updyke, G. Dobson, T. Podnar, L. Osterritter, B. Earl, and A. Cerini, “Ghosts in the machine: A framework for cyber-warfare exercise npc simulation," Carnegie Mellon University, Software Engineering Institute's Digital Library, Tech. Rep. CMU/SEI-2018- TR-005, Dec 2018, accessed: 2025-Dec-16. [Online]. Available: https://doi.org/10.1184/R1/12363029.v1

[19] G. Bonofiglio, V. Iovinella, G. Lospoto, and G. Di Battista, "Kathará: A container-based framework for implementing network function virtualization and software defined networks," in NOMS 2018 - 2018 IEEE/IFIP Network Operations and Management Symposium, 2018, pp. 1–9.