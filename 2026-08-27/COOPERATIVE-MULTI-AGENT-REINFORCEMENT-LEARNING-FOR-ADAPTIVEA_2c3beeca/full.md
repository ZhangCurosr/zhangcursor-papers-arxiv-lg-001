# COOPERATIVE MULTI-AGENT REINFORCEMENT LEARNING FOR ADAPTIVEAGGREGATION IN SEMI-SUPERVISED FEDERATED LEARNING WITH NON-IID DATA

Rene Glitza, Luca Becker and Rainer Martin

Institute of Communication Acoustics, Ruhr-Universitat Bochum, Bochum, Germany¨

## ABSTRACT

Federated Learning (FL) enables distributed training of machine learning models while preserving data privacy. However, FL struggles with heterogeneous, non-IID client data distributions, resulting in sub-optimal and biased global models. In this paper, we propose pFedMARL, a novel approach leveraging Multi-Agent Reinforcement Learning (MARL) with Twin Delayed Deep Deterministic Policy Gradient (TD3) to dynamically adapt aggregation strategies in FL settings. Our method employs a server-side agent adjusting client contributions to optimize global model robustness and client-side agents balancing global and local updates to personalize models effectively without pre-training. We demonstrate superior performance of pFedMARL for training a semi-supervised audio spectrogram transformer, matching or outperforming FedAvg, Ditto, and local training approaches across multiple non-IID scenarios and in the presence of adversarial clients. Our results indicate that pFedMARL actively improves accuracy, robustness, and fairness, making it suitable for real-world deployments.

Index Terms— deep reinforcement learning, personalized learning, anomaly detection, data heterogeneity, adversarial clients

## 1. INTRODUCTION

The rise of IoT devices with AI accelerators has enabled the deployment of deep neural networks (DNNs) directly on consumer hardware. Running DNNs locally reduces latency and addresses growing privacy concerns, especially under data protection laws such as the EU GDPR [1]. However, to achieve high performance in real-world applications, access to diverse and authentic data remains critical.

Federated Learning (FL) [2] offers a promising solution by allowing edge devices to collaboratively train models without sharing raw data. Clients train locally and send model updates to a central server, which aggregates them into a global model and redistributes this model to clients. While effective in principle, FL struggles with non-independent and non-identically distributed (non-IID) data. Due to diverse environments and usage patterns, the distributions of client data may vary significantly [3, 4].

To overcome these issues, prior work has explored client selection, data weighting [5], aggregation adjustments [6], and clustering [7]. Yet, standard FL methods like Federated Averaging (FedAvg) [2] often result in high performance on average but are inconsistent across individual clients. Therefore, personalized FL aims to jointly optimize global and local models [8], or embed shared knowledge into clients [9]. Also, adversarial clients may send malicious model updates that threaten the robustness and performance of the global model. Server-side Deep Reinforcement Learning (DRL) like in FedAA [10] and FedDRL [11] can be used to adaptively weigh client contributions, filtering out adversarial clients and improving convergence and fairness, but does not improve personalization.

In this work, we propose pFedMARL – Personalized Federated Learning with Multi-Agent Off-Policy Reinforcement Learning – a novel multi-agent RL-based FL framework that simultaneously creates a robust global model and personalized client models tailored to heterogeneous environments. A server-side DRL agent dynamically adjusts client aggregation weights to enhance global model robustness against non-IID data distribution and adversarial behaviors. Each client deploys a DRL agent locally to personalize its model by balancing individual learning with global knowledge. We employ Twin Delayed DDPG (TD3) [12], trained online without pretraining, for both server and client agents. Finally, we evaluate our method on a semi-supervised audio spectrogram transformer pretraining, benchmarking against FedAvg, Ditto, local-only training, and a global oracle system. By training masked-patch reconstruction and patch-level classification simultaneously, the transformer model learns rich and transferable audio representations that are well-suited for fine-tuning on various downstream applications, such as anomaly detection and condition monitoring [13], sound event detection [14], and speaker verification [15], where both generalization and accuracy are essential. Here, pFedMARL provides a flexible and scalable solution to FL under non-IID conditions, offering strong performance in global cooperation and local personalization.

## 2. RELATION TO PRIOR WORK

We situate pFedMARL within research on non-IID federated aggregation with adversarial clients, personalized FL, and DRL.

## 2.1. Federated Learning

FL enables collaborative training of a global model by having a central server aggregate weight updates from multiple clients in an iterative fashion, each of which trains locally on its private data. A common aggregation strategy is FedAvg [2], defined as

$$
\pmb { \theta } _ { g } ^ { \langle \tau \rangle } = \sum _ { i = 1 } ^ { M } a _ { g , i } \pmb { \theta } _ { i } ^ { \langle \tau \rangle } , \mathrm { w i t h } \pmb { \theta } _ { i } ^ { \langle \tau \rangle } = \pmb { \theta } _ { i } ^ { \langle \tau - 1 \rangle } + \Delta \pmb { \theta } _ { i } ^ { \langle \tau \rangle } ,\tag{1}
$$

where $\pmb { \theta } _ { g } ^ { \langle \tau \rangle }$ and $\pmb { \theta } _ { i } ^ { \langle \tau \rangle }$ represent server and client model weights of communication round $\tau ,$ respectively, and $a _ { g , i }$ <sub>i</sub> are the aggregation weights summing to 1. Although FedAvg works well in general, it struggles under non-IID data conditions [4].

Ditto [8] is a framework for personalized FL that attempts to enhance FL regarding fairness in non-IID scenarios and robustness against label poisoning, random-update, and model-replacement attacks. It does so by jointly training a global model and client-specific models via a global-regularized multi-task objective. Each client i learns a personalized model by solving

$$
\operatorname* { m i n } _ { \pmb { \theta } _ { i } ^ { ( \tau ) } } \ F _ { i } ( \pmb { \theta } _ { i } ^ { \langle \tau \rangle } ) + \frac { \lambda } { 2 } \left\| \pmb { \theta } _ { i } ^ { \langle \tau \rangle } - \pmb { \theta } _ { g } ^ { \langle \tau \rangle } \right\| _ { 2 } ^ { 2 }\tag{2}
$$

![](images/a5973d72d90de9067b239fc6c024003b351d39e834f311ecadaaa6aebdce03be.jpg)  
Fig. 1: Illustration of the TD3 agent with an experience replay buffer $B ,$ two main (red) and soft-updated target (blue) values networks $Q _ { k }$ and $\boldsymbol { Q } _ { k } ^ { \prime } ,$ , a main π and target $\pi ^ { \prime }$ policy network updated off-policy with noise added to target action $\mathbf { } \mathbf { } ^ { a ^ { \prime } } + \varepsilon . \mathbf { \nabla } \pi ( \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } ^ { \left. \tau \right. } )$ interacts based on observation $\mathbf { } _ { o } \langle \tau \rangle$ in FL round τ with the environment for storing new trajectories $\scriptstyle \mathbf { b . }$

where $F _ { i }$ is the local objective and λ trades off local adaptation against proximity to the global solution. When $\lambda = 0 .$ , Ditto simplifies to training local models, and with $\lambda  \infty .$ , it reverts to the global model objective. Ditto alternates per round between updating the global model with a standard FedAvg solver as in (1) and taking additional local steps on the personalized objective using the current global weights as described in (2).

## 2.2. Challenging non-IID data distributions

Following [4, 11], our study investigates non-IID data category combinations of quantity, label, and cluster skew to realistically address FL challenges. Quantity Skew (QS): Clients hold different amounts of data, frequently observed in user-generated datasets. Label Skew (LS): Clients have distinct label distributions arising naturally from user-specific environments. Cluster Skew (CS): Clients form groups with similar within-cluster but different acrosscluster label distributions. This scenario closely reflects real-world conditions, capturing inter-client correlations prevalent in healthcare, regional deployments, and social settings [11].

## 2.3. Deep Reinforcement Learning

In Deep Reinforcement Learning (DRL), an agent learns to interact with an environment that has a state ${ \pmb s } ^ { \langle \tau \rangle } \in \mathcal { S }$ at timestamp τ. The agent selects an action $\pmb { a } ^ { \left. \tau \right. } \in \mathcal { A }$ based on $\pmb { s } ^ { \langle \tau \rangle }$ to interact with the environment. Subsequently, it receives a next state $s ^ { \langle \tau + 1 \rangle }$ , and a reward $r ^ { \langle \tau \rangle }$ , representing the quality of the action.

We instantiate an actor–critic architecture with Twin Delayed Deep Deterministic Policy Gradient (TD3) [12] as illustrated in Fig. 1. The actor is a θ<sup>π</sup>-parameterized deterministic policy $\pi _ { \pmb { \theta } ^ { \pi } } ( \pmb { s } )$ (abbreviated as $\pi ( s ) )$ that maximizes the expected cumulative discounted return for an infinite horizon $( J ( \pmb \theta ^ { \tilde { \pi } } ) ) _ { , } ^ { \prime }$ ) using the deterministic policy algorithm [16].

Two critics $Q ( s , \pmb { a } )$ with parameters $\theta _ { k } ^ { Q } , k \in { 1 , 2 } ,$ , are trained with temporal difference (TD) with clipped smoothing noise $\varepsilon \sim$ ${ \mathcal { N } } ( \mathbf { 0 } , \sigma ^ { 2 } { \bar { \pmb { I } } } )$ on the target action during training to estimate the expected return $\left( \mathcal { L } \left( \pmb { \theta } _ { k } ^ { Q } \right) \right)$

Target networks $Q _ { 1 , 2 } ^ { \prime }$ and $\pi ^ { \prime } .$ , used for a stable reference in the training process, follow the soft update $\pmb { \theta } ^ { \prime }  \rho \pmb { \theta } + ( 1 - \rho ) \pmb { \theta } ^ { \prime }$ with $\rho \in ( 0 , 1 )$ . TD3 also applies a policy-delay update for stability [12].

![](images/73425898068596592b81f8cbc31dd3f63ea9d2e4ba3e1b5bd9ddbee68128955b.jpg)  
Fig. 2: Overview of the proposed pFedMARL framework, where the server holds one type of agent (green) and every client incorporates another type of agent locally (blue). They return action a for model aggregation based on observation o and update $\theta ^ { \pi }$ regarding reward r on the device. The environment illustrates the usual interaction separation in the DRL paradigm.

In Multi-Agent Reinforcement Learning (MARL), each agent typically observes $\pmb { o } ^ { \langle \tau \rangle } \in \mathcal { O }$ (a partial view of $\pmb { s } ^ { ( \tau ) } .$ ), instead of $\bar { \mathbf { s } ^ { ( \tau ) } }$

## 3. PROPOSED METHOD

By deploying RL agents at both the server and client levels, pFed-MARL achieves improved robustness, fairness, and overall performance while generating personalized models aligned with each client’s unique data distribution. We formulate the FL aggregation as a MARL problem, summarized in Algorithm 1:

Environment: The environment is represented by an FL system with one server and M clients $i \in \bar { \mathcal { M } } = \{ 1 , . . . , M \}$ . Each client i holds a local dataset $\mathcal { D } _ { i }$ and a model $\pmb \theta _ { i }$ . In each communication round τ, a single RL step proceeds as follows: 1) The server computes observation ${ o } _ { g } ^ { \langle \tau \rangle }$ and queries its RL agent. 2) The agent returns a raw action vector $\boldsymbol { a } _ { g } ^ { * } ,$ , which is normalized via softmax to obtain the impact factors $\pmb { a } _ { g } ^ { \langle \tau \rangle }$ in (1). 3) The global model $\pmb { \theta } _ { g } ^ { \langle \tau \rangle }$ is broadcasted to all clients. 4) Each client computes its observation $o _ { i } ^ { \langle \tau \rangle }$ and sends it to its local agent. 5) The agent outputs an impact factor $a _ { i } ^ { \left. \tau \right. }$ to mix the global with the locally trained model (3). 6) Clients train their new local models $\pmb { \theta } _ { i } ^ { \langle \tau \rangle }$ with global model regularization as in (1) and send them, along with the observations, to the server. 7) The next round $\tau + 1$ begins.

Agent: We propose a MARL approach inspired by TD3 [12], with two types of agents: a server agent and client agents. The server’s main policy network $\pi ( s )$ dynamically adjusts impact factors for creating a robust global model, while each client policy balances the global and local models for personalization. Both agents follow the same TD3-inspired training process (Fig. 1), using offpolicy updates with experience replay buffers B, where transition vectors b are saved. They differ only in the action A and observation space O.

State and Observation: The server agent’s observation ${ o } _ { g } ^ { \langle \tau \rangle }$ consists of four (M)-dimensional vectors and the current round index τ, including combined client validation losses $\mathcal { L } ( \pmb { \theta } _ { i } ^ { \langle \tau \rangle } ; \mathcal { D } _ { \mathrm { v a l } , i } )$ $( l ^ { \langle \tau \rangle } )$ as described in Section 4.1, cosine similarities $\pmb { c } ^ { \langle \tau \rangle }$ between client $\Delta \theta _ { i } ^ { \left. \tau \right. }$ and global model updates $\Delta \theta _ { g } ^ { \langle \tau \rangle } , l ^ { 2 }$ -norm distances $\pmb { d } ^ { \langle \tau \rangle }$ between global $\pmb { \theta } _ { g } ^ { \langle \tau \rangle }$ and client models $\Delta \theta _ { i } ^ { \left. \tau \right. }$ , and the number $\mathbf { \delta } _ { n } \langle \tau \rangle$ of mini-batches per client. The client agent’s observation $o _ { i } ^ { \langle \tau \rangle }$ consists of seven single values: cosine similarity, $l ^ { 2 } .$ -norm distance, and round τ (as above), the local and global models’ separate reconstruction losses $\mathcal { L } _ { \mathrm { M S E } } ( \pmb { \theta } _ { i / g } ^ { ( \tau ) } ; \mathcal { D } _ { \mathrm { v a l } , i } )$ and their classification F1-scores $\mathcal { L } _ { \mathrm { F 1 } } ( \pmb { \theta } _ { i / g } ^ { \langle \tau \rangle } ; \mathcal { D } _ { \mathrm { v a l } , i } )$ on local data, combined to $\boldsymbol { l } _ { i } ^ { \left. \tau \right. }$

Algorithm 1: pFedMARL Loop   
Input: Server g, clients $i \in \mathcal { M }$ with $\mathcal { D } _ { i }$ , model ${ \pmb \theta } ^ { ( 0 ) }$   
communication rounds τ<sub>max</sub>   
for $\tau \in \{ 1 , \ldots , \tau _ { m a x } \}$ do   
$\pmb { a } _ { g } ^ { \langle \tau \rangle } \gets \mathrm { S o f t m a x } ( \pi _ { g } ( \pmb { o } _ { g } ^ { \langle \tau \rangle } ) + \pmb { \varepsilon } )$   
$\begin{array} { r } { \pmb { \theta } _ { g } ^ { \langle \tau \rangle }  \sum _ { i \in \mathcal { M } } a _ { g , i } ^ { \langle \tau \rangle } \pmb { \theta } _ { i } ^ { \langle \tau - 1 \rangle } } \end{array}$ global model update (1)   
$r _ { g } ^ { \langle \tau \rangle } = - \log \mathcal { L } ( \pmb { \theta } _ { g } ^ { \langle \tau \rangle } ; \mathcal { D } _ { \mathrm { v a l } } )$ reward (4)   
for $i \in \mathcal { M }$ clients in parallel do   
$a _ { i } ^ { \langle \tau \rangle } \gets \pi _ { i } \big ( o _ { i } ^ { \langle \tau \rangle } \big ) + \varepsilon$   
$\pmb { \theta } _ { i } ^ { \langle \tau \rangle }  \mathrm { A d a m } ( \pmb { \theta } _ { i } ^ { \langle \tau - 1 \rangle } ; \pmb { \theta } _ { g } ^ { \langle \tau \rangle } , \mathcal { D } _ { i } )$ with reg. (3)   
$r _ { i . } ^ { \langle \tau \rangle } = - \log \mathcal { L } ( \pmb { \theta } _ { i } ^ { \langle \tau \rangle } ; \mathcal { D } _ { \mathrm { v a l } , i } )$ reward (4)   
$o _ { i } ^ { \langle \tau + 1 \rangle } \gets ( l _ { i } , c _ { i } , d _ { i } , \tau )$   
end   
$o _ { g } ^ { \langle \tau + 1 \rangle } \gets ( l , c , d , n , \tau )$   
for g and $i \in \mathcal { M }$ in parallel do   
Store $( s ^ { \langle \tau \rangle } , a ^ { \langle \tau \rangle } , r ^ { \langle \tau \rangle } , s ^ { \langle \tau + 1 \rangle } )$ in B   
Sample transition b batch from buffer B   
Update $\theta ^ { \pi } , \theta _ { k } ^ { Q }$ , see $F i g . \ I$   
Soft update $\theta ^ { \pi ^ { \prime } } , \theta _ { k } ^ { Q ^ { \prime } }$ via $\pmb { \theta } ^ { \prime }  \rho \pmb { \theta } + ( 1 - \rho ) \pmb { \theta } ^ { \prime }$   
end   
end

Action: The server agent’s main policy outputs a vector ${ \pmb a } _ { g } ^ { * } =$ $\pi ( o _ { g } ^ { \langle \tau \rangle } ) + \varepsilon , a _ { g } ^ { \langle * \rangle } \in [ - 1 , 1 ] ^ { M }$ with Gaussian exploration noise, converted via softmax into impact factors for aggregation including batch-norm statistics, see (1): ${ \pmb a } _ { g } ^ { \langle \tau \rangle } = \mathrm { S o f t m a x } ( { \pmb a } _ { g } ^ { \langle * \rangle } )$ . The client agents output a scalar $a _ { i }$ clipped to [0, 1] with applied exploration noise, determining the proportion between global regularization and personalized model in the training loss $\mathcal { L } _ { i } ^ { \mathrm { p F e d M A R L } } ( \pmb { \theta } _ { i } ; \pmb { \theta } _ { g } ^ { \langle \tau \rangle } , \mathcal { D } _ { i } )$

$$
\mathcal { L } _ { i } ^ { \mathrm { p F e d M A R L } } = \frac { 1 } { \left| \mathcal { D } _ { i } \right| } \sum _ { \left( \pmb { x } , \pmb { y } \right) \in \mathcal { D } _ { i } } \ell ( f _ { \pmb { \theta } _ { i } } ( \pmb { x } ) , \pmb { y } ) + \frac { a _ { i } } { 2 } \left\| \pmb { \theta } _ { i } - \pmb { \theta } _ { g } ^ { \langle \tau \rangle } \right\| _ { 2 } ^ { 2 } ,\tag{3}
$$

where $\ell ( f _ { \pmb \theta _ { i } } ( \pmb x ) , \pmb y )$ is an arbitrary model loss.

Reward: The reward incentivizes the validation losses without regularization after each aggregation step as

$$
r ^ { \langle \tau \rangle } = - \log \mathcal { L } ( \pmb { \theta } ^ { \langle \tau \rangle } ; \mathcal { D } _ { \mathrm { v a l } } ) ,\tag{4}
$$

implicitly penalizing performance degradation and compensating for high initial improvements. Furthermore, due to the same objective, clients and server collaboratively optimize a shared global model that performs well for all clients’ data.

## 4. EXPERIMENTS

We evaluate our method on a semi-supervised federated learning task for training a small audio spectrogram transformer (AST) regarding patch reconstruction and classification accuracy. The goal is to train an AST that is effective for data typical for a client’s environment while maintaining robust performance on unseen data.

![](images/660b37ab164d5f9b423616aa9222ebcc637f2ee3c59a53e915c2cd217acb3aeb.jpg)  
Fig. 3: Visualization of three data partitioning strategies (QS: low quantity & label skew, LS: high quantity & label skew, CS: cluster skew) across 14 clients, where the circle size indicates the number of samples per client and the red color adversarial clients.

## 4.1. Network architecture and training procedure

We employ a compact, fully connected vision transformer [17, 18] with approximately 450k parameters, suitable for federated scenarios. The model is optimized to minimize the mean squared error (MSE) between patch-masked original and reconstructed log-mel band energy audio features in an unsupervised fashion $\ell _ { \mathrm { r e c o n } } ( f _ { \pmb { \theta } _ { i } } ( \pmb { x } ) , \pmb { x } )$ , and the negative log-likelihood loss of a fully-connected classification head attached to the latent space $\ell _ { \mathrm { c l a s s } } ( f _ { \pmb { \theta } _ { i } } ( \pmb { x } ) , \pmb { y } )$ , combined to a single loss $\ell = \ell _ { \mathrm { r e c o n } } + 2 . 0 \ell _ { \mathrm { c l a s s } }$

Our TD3 agents are implemented as fully-connected networks with two layers each for the policy (π) and critic $( Q ) ,$ , and with 256 units per hidden layer and tanh activation functions. Training uses a batch size of 8, limited to 64 batches per epoch, learning rates of $1 \times 1 0 ^ { - 2 }$ and $1 \times 1 0 ^ { - 4 }$ for policy and critic respectively, discount factor $\gamma = 0 . 8 0$ , soft-update rate $\rho = 0 . 9 9$ , and policy delay of 4 epochs. The Gaussian exploration noise covariance parameter $( \sigma ^ { 2 } )$ linearly decreases from 0.40 to 0.05 across 80 epochs. A replay buffer [19] prioritizing samples with higher TD error is used with parameters $\alpha = 0 . 7$ and $\beta = 0 . 5$

## 4.2. Dataset

We use 10% of the DCASE challenge Task 2 development dataset [20,21], containing single-channel laboratory recordings of machine sounds mixed with environmental noise. It includes audio signals from 14 machine classes (e.g., ToyCar, Fan, Gearbox) sampled at 16 kHz, with audio durations of 6 s to 18 s. Each class has 990 normal training clips from the source domain, 10 from the target domain, and 200 test clips (100 normal, 100 anomalous) mixed from both domains, mimicking realistic data drift.

## 4.3. Experimental Procedure

We benchmark pFedMARL against FedAvg [2], Ditto [8] with the optimal tested $\lambda = 0 . 5 .$ , local-only training, and an oracle baseline trained centrally on the raw data of all clients for $\tau _ { \mathrm { m a x } } = 1 0 0 ~ \mathrm { F I }$ rounds. Our evaluation compares reconstruction MSE and classification F1-score performance, standard deviation (std) across clients’ metrics, and $l ^ { 2 } .$ -norm between the clients and global model for: client models on local test data (L), the global server model on all client test sets (global data) (S), and client models on global test data (G). The centralized baseline is always tested on global data.

The dataset is divided between $M = 1 5$ clients to simulate three non-IID scenarios: low quantity and label skew $( \mathrm { Q S } , \alpha = 5 . 0 )$ , high quantity and label skew $( \mathrm { L S } , \alpha = 0 . 4 )$ , and cluster skew (CS) with low label imbalance (manual assignment, $\alpha = 2 . 0 )$ . Class distribution probabilities per client are set via a normalized Dirichlet distribution with parameter $\alpha ,$ enabling realistic data imbalance and variability as illustrated in Figure 3.

Table 1: Comparison of the FL techniques for the data partitions, shown in Figure 3. Mean client performance and standard deviation across clients on own local test set (L) and all client test sets (G), and mean l<sup>2</sup>-norm between client and global model at the end of τ<sub>max</sub>.
<table><tr><td rowspan="2"></td><td rowspan="2">Algorithm Data</td><td colspan="2">pFedMARL</td><td colspan="2">Ditto</td><td colspan="2">FedAvg</td><td colspan="2">Local</td></tr><tr><td>L</td><td>G</td><td>L</td><td>G</td><td>L</td><td>G II</td><td>L</td><td>G</td></tr><tr><td rowspan="5">QS</td><td rowspan="5">MSE Mean ↓ MSE Std ↓ F1 Mean ↑</td><td>0.10</td><td>0.11</td><td>0.17</td><td>0.17</td><td>1.17</td><td>1.17</td><td>0.10</td><td>0.10</td></tr><tr><td>0.00</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.02</td><td>0.00</td><td>0.01</td><td>0.01</td></tr><tr><td>0.77</td><td>0.73</td><td>0.75</td><td>0.71</td><td>0.17</td><td>0.17</td><td>0.84</td><td>0.80</td></tr><tr><td>0.05 L2-Norm ↓</td><td>0.02</td><td>0.04</td><td>0.01</td><td>0.03</td><td>0.00</td><td>0.03</td><td>0.02</td></tr><tr><td colspan="2">3.32</td><td colspan="2">4.52</td><td colspan="2">2.39</td><td colspan="2">3.25</td></tr><tr><td rowspan="6">LS</td><td>MSE Mean ↓</td><td>0.10</td><td>0.11</td><td>0.17</td><td>0.18</td><td>0.96</td><td>0.96</td><td>0.08</td><td>0.11</td></tr><tr><td>MSE Std ↓</td><td>0.02</td><td>0.02</td><td>0.02</td><td>0.03</td><td>0.08</td><td>0.00</td><td>0.02</td><td>0.03</td></tr><tr><td>F1 Mean ↑</td><td>0.87</td><td>0.60</td><td>0.69</td><td>0.34</td><td>0.13</td><td>0.13</td><td>0.92</td><td>0.59</td></tr><tr><td>F1 Std ↓</td><td>0.05</td><td>0.08</td><td>0.07</td><td>0.06</td><td>0.09</td><td>0.00</td><td>0.04</td><td>0.08</td></tr><tr><td>L2-Norm ↓</td><td>3.31</td><td></td><td>6.17</td><td></td><td>2.35</td><td></td><td>3.22</td><td></td></tr><tr><td rowspan="5"></td><td>MSE Mean</td><td>0.06</td><td>0.12</td><td>0.15</td><td>0.19</td><td>1.25</td><td>1.25</td><td>0.03</td><td>0.07</td></tr><tr><td>MSE Std ↓</td><td>0.03</td><td>0.07</td><td>0.05</td><td>0.07</td><td>0.10</td><td>0.00</td><td>0.03</td><td>0.06</td></tr><tr><td>F1 Mean ↑</td><td>0.96</td><td>0.21</td><td>0.91</td><td>0.19</td><td>0.02</td><td>0.02</td><td>0.98</td><td>0.21</td></tr><tr><td>F1 Std ↓</td><td>0.04</td><td>0.01</td><td>0.09</td><td>0.02</td><td>0.04</td><td>0.00</td><td>0.04</td><td>0.01</td></tr><tr><td>L2-Norm ↓</td><td colspan="2">3.32</td><td colspan="2">4.69</td><td colspan="2">2.40</td><td colspan="2">3.19</td></tr></table>

Table 2: Comparison of the FL techniques server model on all client test sets for the three non-IID scenarios at the end of $\tau _ { \mathrm { m a x } }$
<table><tr><td>Metric</td><td colspan="3">F1 Mean ↑</td><td colspan="3">MSE Mean ↓</td></tr><tr><td>Scenario</td><td>CS</td><td>LS</td><td>QS</td><td>CS</td><td>LS</td><td>QS</td></tr><tr><td>pFedMARL</td><td>0.22</td><td>0.38</td><td>0.61</td><td>0.18</td><td>0.24</td><td>0.24</td></tr><tr><td>Ditto</td><td>0.11</td><td>0.07</td><td>0.22</td><td>2.07</td><td>79.19</td><td>2.17</td></tr><tr><td>FedAvg</td><td>0.03</td><td>0.12</td><td>0.17</td><td>1.34</td><td>1.02</td><td>1.23</td></tr><tr><td>Baseline</td><td></td><td>0.97</td><td></td><td colspan="3">0.01</td></tr></table>

To evaluate the robustness against adversarial clients, we select two clients that apply additive Gaussian noise with $\sigma ^ { 2 } = 0 . 5$ to the transmitted model updates $\Delta \pmb \theta _ { i }$ while the client’s local model remains unchanged, simulating a corrupted transmission or malicious weight injection. In the observation $\mathbf { \delta } _ { \mathbf { o } } \langle \tau \rangle$ only $c _ { i } ^ { \left. \tau \right. }$ and $d _ { i } ^ { \langle \tau \rangle }$ of the adversarial clients are calculated on corrupted updates, while the losses are based on the non-corrupted models. This does not apply to the baseline and local training conditions, whose performance will therefore not be degraded by adversarial clients.

## 5. EXPERIMENTAL RESULTS

Table 1 summarizes the final performance of each client and algorithm on their local test sets (L) and the test sets of all clients (G) for the three non-IID scenarios. Local-only training excels in personalization and also performs well in reconstructing global data, where data diversity is less critical, but falters in classifying unseen classes. With adversarial clients, pFedMARL consistently surpasses FedAvg and Ditto in both MSE and F1-score for the local and global evaluation. As pFedMARL optimizes for local performance, it converges towards local-only training, nearly matching its global performance yet falling short on local data due to the adversarial influence. Across scenarios, we observe the expected personalization/generalization trade-off, where gains in L coincide with losses in G.

![](images/5a7b775580502b901bc8bb0a863c050c9e28bb6eb19f900bd9d632bdf26973c2.jpg)  
Fig. 4: Mean validation client on local data and server F1-accuracy on all client sets, along with mean actions $a ^ { \left. \tau \right. }$ separated into adversarial clients and the other clients for the CS non-IID scenario.

Server models shown in Table 2 underperform compared to client models on global data (G) because adversarial updates corrupt the aggregated weights more. pFedMARL does mitigate this by down-weighting adversarial clients but cannot exclude them entirely. The baseline is unaffected by data heterogeneity or adversarial weight injection due to the availability of all data at a centralized server and serves as the overall upper bound.

Figure 4 illustrates these dynamics. After the initial collection phase, the average contribution of benign clients to the global model stabilizes near 0.5, whereas adversarial influence drops $\mathrm { t o } \approx 0 . 1$ This rapid suppression coincides with a steady rise in server accuracy after the initial transition. On the client side, adversarial participants choose almost no global regularization, while benign clients initially leverage the global model (actions near the Ditto optimum of $\lambda \approx 0 . 5 )$ to exploit shared knowledge and, after ≈ 50 rounds, reduce reliance on the server model to personalize further. Correspondingly, pFedMARL’s local F1-accuracy tracks Ditto until about round 50 and then increases beyond it, approaching local training.

## 6. CONCLUSIONS

In this paper, we introduce pFedMARL<sup>1</sup> , an adaptive FL framework utilizing multi-agent reinforcement learning to address non-IID challenges and adversarial client behavior effectively. By employing TD3-based agents on both server and client sides, our approach achieves a robust global model while ensuring personalized client models without pre-training the agents. Empirical evaluations in an audio spectrogram transformer training task across various data skew scenarios with adversaries demonstrate that pFed-MARL may outperform traditional aggregation methods such as FedAvg, Ditto and local-only training. Particularly, pFedMARL shows consistent performance in adversarial scenarios and matches Ditto in non-adversarial settings, delivering high accuracy with improved fairness among clients and robustness while bypassing Ditto’s need for a second local training. Thus, our results highlight the potential of collaborative multi-agent reinforcement learning to enhance FL practices in real-world applications. Future research will explore the extension of pFedMARL to other data modalities and evaluate scalability in federations with significantly more clients.

## 7. ACKNOWLEDGEMENT

This work has been supported by the German Federal Ministry of Research, Technology and Space (BMFTR) (grant 02L19C200, ”HUMAINE”) and by the German Research Foundation (DFG) (project numbers 429873205 and 549576906).

## 8. REFERENCES

[1] European Parliament and Council, “Regulation (EU) 2016/679 of the European Parliament and of the Council of 27 April 2016 on the protection of natural persons with regard to the processing of personal data and on the free movement of such data, and repealing Directive 95/46/EC (General Data Protection Regulation),” 2016.

[2] Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas, “Communication-Efficient Learning of Deep Networks from Decentralized Data,” in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics. Apr. 2017, pp. 1273–1282, PMLR.

[3] Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith, “Federated Optimization in Heterogeneous Networks,” Proceedings of Machine Learning and Systems, vol. 2, pp. 429–450, Mar. 2020.

[4] Kevin Hsieh, Amar Phanishayee, Onur Mutlu, and Phillip Gibbons, “The Non-IID Data Quagmire of Decentralized Machine Learning,” in Proceedings of the 37th International Conference on Machine Learning. Nov. 2020, pp. 4387–4398, PMLR.

[5] Hongda Wu and Ping Wang, “Fast-Convergent Federated Learning With Adaptive Weighting,” IEEE Transactions on Cognitive Communications and Networking, vol. 7, no. 4, pp. 1078–1088, Dec. 2021.

[6] Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh, “SCAFFOLD: Stochastic Controlled Averaging for Federated Learning,” in Proceedings of the 37th International Conference on Machine Learning. Nov. 2020, pp. 5132–5143, PMLR.

[7] Felix Sattler, Klaus-Robert Muller, and Wojciech Samek,¨ “Clustered Federated Learning: Model-Agnostic Distributed Multitask Optimization Under Privacy Constraints,” IEEE Transactions on Neural Networks and Learning Systems, vol. 32, no. 8, pp. 3710–3722, Aug. 2021.

[8] Tian Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith, “Ditto: Fair and Robust Federated Learning Through Personalization,” in Proceedings of the 38th International Conference on Machine Learning. July 2021, pp. 6357–6368, PMLR.

[9] Ziwen Huang and Nikolaos M. Freris, “Reinforcement Learning-Based Layer-Wise Aggregation for Personalized Federated Learning,” IEEE Internet of Things Journal, vol. 12, no. 7, pp. 8614–8625, 2025.

[10] Jialuo He, Wei Chen, and Xiaojin Zhang, “FedAA: A Reinforcement Learning Perspective on Adaptive Aggregation for Fair and Robust Federated Learning,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 16, pp. 17085–17093, Apr. 2025.

[11] Nang Hung Nguyen, Phi Le Nguyen, Thuy Dung Nguyen, Trung Thanh Nguyen, Duc Long Nguyen, Thanh Hung

Nguyen, Huy Hieu Pham, and Thao Nguyen Truong, “Fed-DRL: Deep Reinforcement Learning-based Adaptive Aggregation for Non-IID Data in Federated Learning,” in Proceedings of the 51st International Conference on Parallel Processing, New York, NY, USA, Jan. 2023, ICPP ’22, pp. 1–11, Association for Computing Machinery.

[12] Scott Fujimoto, Herke Hoof, and David Meger, “Addressing Function Approximation Error in Actor-Critic Methods,” in Proceedings of the 35th International Conference on Machine Learning. July 2018, pp. 1587–1596, PMLR.

[13] Kevin Wilkinghoff, “Self-Supervised Learning for Anomalous Sound Detection,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Apr. 2024, pp. 276–280.

[14] Yusun Shul and Jung-Woo Choi, “CST-Former: Transformer with Channel-Spectro-Temporal Attention for Sound Event Localization and Detection,” in ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Apr. 2024, pp. 8686–8690.

[15] B. Desplanques, J. Thienpondt, and K. Demuynck, “ECAPA-TDNN: Emphasized channel attention, propagation and aggregation in TDNN based speaker verification,” in Proc. Interspeech 2020, 2020, pp. 3830–3834.

[16] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra, “Continuous control with deep reinforcement learning,” July 2019, arXiv:1509.02971 [cs].

[17] Yuan Gong, Cheng-I. Lai, Yu-An Chung, and James Glass, “SSAST: Self-Supervised Audio Spectrogram Transformer,” Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 36, no. 10, pp. 10699–10709, June 2022, Number: 10.

[18] Alan Baade, Puyuan Peng, and David Harwath, “MAE-AST: Masked Autoencoding Audio Spectrogram Transformer,” Mar. 2022, arXiv:2203.16691 [cs, eess].

[19] Tom Schaul, John Quan, Ioannis Antonoglou, and David Silver, “Prioritized Experience Replay,” Feb. 2016, arXiv:1511.05952 [cs].

[20] Kota Dohi, Tomoya Nishida, Harsh Purohit, Ryo Tanabe, Takashi Endo, Masaaki Yamamoto, Yuki Nikaido, and Yohei Kawaguchi, “MIMII DG: Sound dataset for malfunctioning industrial machine investigation and inspection for domain generalization task,” in Proceedings ofthe 7th detection and classification of acoustic scenes and events workshop (DCASE), Nancy, France, Nov. 2022.

[21] Noboru Harada, Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Masahiro Yasuda, and Shoichiro Saito, “Toy-ADMOS2: Another dataset of miniature-machine operating sounds for anomalous sound detection under domain shift conditions,” in Proceedings of the detection and classification of acoustic scenes and events workshop (DCASE), Barcelona, Spain, Nov. 2021, pp. 1–5.