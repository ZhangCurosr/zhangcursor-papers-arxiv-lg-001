# MINT: Mitigating Structural Inconsistency via Atentive Message Passing for Financial Anomaly Detection

Yin-Ju Wu National Taiwan Normal University Taipei, Taiwan 61247075s@ntnu.edu.tw

Bi-Cheng Yan National Taiwan Normal University Taipei, Taiwan 80847001s@ntnu.edu.tw

Berlin Chen National Taiwan Normal University Taipei, Taiwan berlin@ntnu.edu.tw

## Abstract

Illicit account detection is a critical task for financial risk control systems, aiming to pinpoint malicious actors within transaction networks. Existing eforts typically recast transaction networks into directed multigraphs and leverage graph neural networks (GNNs) to detect anomalies via topological structures under the homophily assumption. However, directly grafting GNNs onto transaction graphs is faced with at least two pivotal challenges, i.e., structural inconsistency and label scarcity. The former arises from deceptive connections forged by malicious actors, while the latter stems from the lack of annotated data. In light of these, we introduce MINT <sup>1</sup>, a novel graph neural framework that mitigates structural inconsistency via an attentive message-passing scheme on transaction graphs. Specifically, MINT introduces a discrepancy-aware feature propagation, which tracks malicious actors by quantifying deviations between nodes across bi-directional transaction flows. A transaction salience module is proposed to mitigate structural inconsistency by adaptively regularizing the interaction strength between nodes. Furthermore, to alleviate label scarcity, MINT is optimized via Positive-Unlabeled (PU) learning, which exploits only a minority of illicit nodes coupled with a majority of unlabeled ones. Extensive experiments conducted on three financial transaction benchmarks, including two large-scale public and one in-house real-world datasets, validate the superior efectiveness of MINT over several cutting-edge baselines.

## CCS Concepts

• Computing methodologies → Machine learning.

## Keywords

Graph Neural Networks, Graph Anomaly Detection, Financial Transaction Networks, Positive-Unlabeled Learning

## ACM Reference Format:

Yin-Ju Wu, Bi-Cheng Yan, and Berlin Chen. 2026. MINT: Mitigating Structural Inconsistency via Attentive Message Passing for Financial Anomaly Detection. In Proceedings ofthe 35th ACM International Conference on Information and Knowledge Management (CIKM ’26), November 07–11, 2026, Rome, Italy. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/ 3799682.3839912

![](images/9e4b60bce868e8f5a2983535b82af21b155f7318081c79d17bf650fc89bbc2e0.jpg)  
Figure 1: Illustration of structural inconsistency and transaction hub dominance in financial transaction networks.

## 1 Introduction

Graph Anomaly Detection (GAD) has emerged as a pivotal task in transaction monitoring and financial risk control systems, dedicated to identifying atypical graph entities (nodes and edges) that significantly deviate from global structures or local afinities [13, 20]. Since large-scale financial transaction networks naturally manifest as directed multigraphs, extensive studies have leveraged GAD techniques to detect illicit accounts [1, 3, 15, 21, 23]. By tracking transaction flows via iterative message passing, GAD approaches harness graph neural networks (GNNs) to efectively pinpoint malicious actors or fraudulent behaviors based on topological structures [11, 19, 25].

Existing eforts have applied GNNs to detect financial anomalies, predominantly relying on the homophily assumption where connected nodes share similar attributes or local structures [10, 26]. However, this assumption is untenable in real-world transaction networks, where the neighborhood aggregation mechanism of GNNs inevitably dilutes the subtle signals of graph anomalies. Directly grafting GNN-based approaches onto illicit account detection poses two pivotal challenges, namely structural inconsistency and label scarcity. Specifically, as illustrated in Figure 1, the structural inconsistency arises from malicious actors forging deceptive connections with benign accounts, while high-degree nodes (i.e., transaction hubs) tend to dominate node representation learning. The latter stems from the fact that accurate anomaly annotation is highly labor-intensive and requires exhaustive auditing by domain experts, resulting in a scarcity of labeled nodes while leaving the vast majority unlabeled [12, 14].

Building on these observations, we propose MINT, a novel graph neural framework tailored for transaction graphs that efectively mitigates structural inconsistency via attentive message passing. Specifically, to capture anomaly signals over directed multigraphs, MINT introduces a discrepancy-aware message-passing scheme, which aims to iteratively quantify feature deviations between nodes across their incoming and outgoing neighborhoods. Furthermore, to mitigate structural inconsistency, MINT incorporates a structural edge reweighting strategy into the message-passing scheme to dynamically calibrate connection strengths based on transaction statistics. In contrast to traditional attentive message-passing which relies on feature similarity to evaluate node importance, our reweighting strategy modulates connection intensities based on two transactional factors, namely, the local transaction fre quency and the global transaction rarity of a node. To mitigate label scarcity, MINT is optimized within a Positive-Unlabeled (PU) learning paradigm, only leveraging a minority of labeled illicit nodes coupled with a vast majority of unlabeled ones during the training phase [6, 7, 17]. Extensive experiments on three financial transaction benchmarks, including two large-scale public and one real-world proprietary datasets, confirm the superior efectiveness and robustness of MINT over various cutting-edge baselines. The main contributions of this work are summarized as follows:

• We identify structural inconsistency and extreme label scarcity as primary bottlenecks in financial anomaly detection, where deceptive connections and high-degree hubs jointly dilute rare anomalous signals.

• We propose MINT, a framework integrating discrepancyaware message passing and structural edge reweighting to dynamically calibrate connection strengths, optimized under a Positive-Unlabeled setting to overcome label scarcity.

• Extensive experiments on public benchmarks and real-world financial dataset demonstrate the superior efectiveness and robustness of MINT against cutting-edge baselines.

## 2 Preliminaries

Graph Construction. We formalize the financial transaction network as a directed multigraph $\mathcal { G } = ( \mathcal { V } , \mathcal { E } , \mathbf { X } )$ , where $_ \mathrm { ~  ~ }$ represents the account set and each edge $\boldsymbol { e } _ { u , v } \in \mathcal { E }$ denotes a directed transaction from � to �. The transactional behaviors of accounts are encoded within a feature matrix $\mathbf { X } \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ , mapping each node �<sub>�</sub> to a statistical profile vector $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ derived from interaction frequencies and monetary aggregates, which serves as the initial state $\mathbf { h } _ { i } ^ { ( 0 ) } = \mathbf { x } _ { i }$ for the �-th layer hidden representation $\mathbf { h } _ { i } ^ { ( l ) }$ during message passing.

Problem Formulation. Financial anomaly detection on $\mathcal { G }$ is formulated as a PU node classification problem, partitioning the account set as $\mathcal { V } = \mathcal { P } \cup \mathcal { U }$ with $\mathcal { P } \cap \mathcal { U } = \emptyset$ . Since P contains only a small fraction ofverified illicit nodes and U represents an unknown mixture of benign and illicit accounts, the strict absence of verified negative labels necessitates learning a graph-based scoring function $f : ( \mathcal G , \mathbf { X } ) \to [ 0 , 1 ] ^ { | \mathcal V | }$ to accurately estimate the illicit probability for each node in U.

## 3 Methodology

In this section, we detail the proposed MINT framework as illustrated in Figure 2. Our framework integrates a structural edge reweighting strategy and a discrepancy-aware attentive messagepassing scheme. The entire framework is optimized with a PU learning objective that efectively stabilizes training under extreme label scarcity scenarios.

## 3.1 Structural Edge Reweighting Strategy

In financial transaction graphs, representation learning is prone to be overwhelmed by high-degree nodes ( transaction hubs), which inevitably dilutes the signals of graph anomalies. To address this limitation, MINT introduces a structural edge reweighting strategy for feature propagation, dynamically calibrating the connection intensity of each transaction by factoring in two topological statistics, namely, the local interaction frequency of the node pair and the global transaction rarity of a node. Furthermore, to capture nuanced graph anomalies concealed within asymmetric directional transaction flows, the proposed calibration strategy is independently computed over incoming and outgoing edge relations, denoted by $r \in$ {in, out}.

For a given relation �, the calibrated connection strength between nodes � and � is denoted as:

$$
w _ { u , v } ^ { ( r ) } = \mathrm { L I F } _ { u , v } ^ { ( r ) } \cdot \mathrm { G T R } _ { v } ^ { ( r ) } ,\tag{1}
$$

where the Local Interaction Frequency (LIF) quantifies the relative transaction intensity from a source account � to a target account �, and the Global Transaction Rarity (GTR) measures the transactional activity of node �. Both mathematical terms are defined as follows:

$$
\mathrm { L I F } _ { u , v } ^ { ( r ) } = \frac { c _ { u , v } ^ { ( r ) } } { \sum _ { m \in N _ { r } ( v ) } c _ { m , v } ^ { ( r ) } } ,\tag{2}
$$

$$
\mathrm { G T R } _ { v } ^ { ( r ) } = \log \left( \frac { | \mathcal { V } | } { | N _ { r } ( v ) | + 1 } \right) ,\tag{3}
$$

where $c _ { u , v } ^ { ( r ) }$ denotes the transaction count from node � to node � within the relation-specific neighborhood $N _ { r } ( \boldsymbol { v } )$ , and |V | represents the total number of accounts in the graph. By leveraging the synergy between $\mathrm { L I F } _ { u , v } ^ { ( r ) }$ and $\mathrm { G T R } _ { v } ^ { ( r ) }$ , MINT efectively captures local connectivity between nodes while penalizing the oversmoothing efect induced by transaction hubs.

## 3.2 Discrepancy-aware Attentive Message-passing Scheme

To mitigate the structural inconsistency caused by deceptive connections, MINT departs from conventional feature oversmoothing by explicitly quantifying representational deviations across directed monetary flows. Specifically, at the �-th message-passing layer, the representational deviation between a target account � and its neigh bor � under relation � is measured as:

$$
{ \bf d } _ { v , u } ^ { ( l , r ) } = \operatorname { t a n h } ( { \bf h } _ { v } ^ { ( l ) } - { \bf h } _ { u } ^ { ( l , r ) } ) ,\tag{4}
$$

where $\mathbf { h } _ { v } ^ { ( l ) }$ and $\mathbf { h } _ { u } ^ { ( l , r ) }$ denote the node representations of accounts � and �, respectively. Subsequently, to obtain the discrepancy feature $\mathbf { h } _ { v } ^ { ( l , r ) }$ for node �, MINT incorporates the calibrated edge signal $w _ { u , v } ^ { ( r ) }$ (derived in Section 3.1) into the attentive message-passing scheme as a structural attention coeficient, aggregating directional messages from the neighborhood $u \in N _ { r } ( v )$ :

$$
\begin{array} { r } { \boldsymbol { \alpha } _ { u , v } ^ { ( l , r ) } = \mathrm { s o f t m a x } \left( \mathrm { L e a k y R e L U } ( \mathbf { a } ^ { \top } \mathbf { d } _ { v , u } ^ { ( l , r ) } ) + \lambda _ { e } \boldsymbol { w } _ { u , v } ^ { ( r ) } \right) ; } \end{array}\tag{5}
$$

$$
\mathbf { h } _ { v } ^ { ( l , r ) } = \mathbf { W } _ { c } \sum _ { u \in { \mathcal { N } } _ { r } ( v ) } \alpha _ { u , v } ^ { ( l , r ) } [ \mathbf { d } _ { v , u } ^ { ( l , r ) } \parallel \mathbf { h } _ { u } ^ { ( l , r ) } ] ,\tag{6}
$$

![](images/d9cfabb4ff47bf082524ddb6e978522059c7ede9746b358f92b4eb4bf9bea025.jpg)  
Figure 2: Overview of the proposed MINT framework.

where a denotes a learnable weight vector, $\lambda _ { e }$ is a learnable scaling factor for adjusting the relative impact of node discrepancies and transactional statistics, and ${ \mathbf { W } } _ { c }$ acts as a transformation matrix.

Finally, the node representation for account � is updated by aggregating its intrinsic representation $\mathbf { h } _ { v } ^ { ( l ) }$ in conjunction with the relation-specific discrepancy features $( \mathrm { i } . \mathrm { e } . , \mathbf { h } _ { v } ^ { ( l , \mathrm { i n } ) }$ and $\mathbf { h } _ { v } ^ { ( l , \mathrm { o u t } ) } )$ ) via a simple attention mechanism. Formally, the updated representation $\mathbf { h } _ { v } ^ { ( l + 1 ) }$ is formulated as:

$$
{ \boldsymbol \beta } ^ { ( k ) } = \frac { \exp ( \mathbf { w } _ { k } ^ { \top } \mathbf { h } _ { v } ^ { ( l , k ) } ) } { \sum _ { k \in \{ \mathrm { s e l f } , \mathrm { i n } , \mathrm { o u t } \} } \exp ( \mathbf { w } _ { k } ^ { \top } \mathbf { h } _ { v } ^ { ( l , k ) } ) } ;\tag{7}
$$

$$
\mathbf { h } _ { v } ^ { ( l + 1 ) } = \mathbf { W } _ { f } \sum _ { k \in \{ \mathrm { s e l f } , \mathrm { i n } , \mathrm { o u t } \} } \beta ^ { ( k ) } \mathbf { h } _ { v } ^ { ( l , k ) } ,\tag{8}
$$

where $\mathbf { W } _ { f }$ denotes a learnable transformation matrix, ${ \bf w } _ { k }$ represents the attention weight vector for each $k \in \{ \mathrm { s e l f } , \mathrm { i n } , \mathrm { o u t } \}$ , and $\mathbf { h } _ { v } ^ { ( l , \mathrm { s e l f } ) }$ is set as $\mathbf { h } _ { v } ^ { ( l ) }$

## 3.3 Positive-Unlabeled Learning Optimization

To derive the illicit probability $s _ { v } ,$ we project the final embedding $\mathbf { h } _ { v } ^ { ( L ) }$ through an MLP classifier equipped with softmax normalization. To robustly accommodate extreme label sparsity, we optimize the model under the established PU setting. Rather than naively treating the unverified mixture U as strict negatives, we mitigate false-negative bias by approximating the expected risk via a prior probability $\pi ,$ which serves as the expected risk representing the estimated illicit ratio. Defining $\ell ^ { + }$ and $\ell ^ { - }$ as the standard positive and negative cross-entropy losses, the objective is formulated as:

$$
\mathcal { L } _ { \mathrm { P U } } = \frac { 1 } { | \mathcal { P } | } \sum _ { v \in \mathcal { P } } \ell ^ { + } ( s _ { v } ) + \frac { 1 } { | \mathcal { U } | } \sum _ { v \in \mathcal { U } } \left[ \pi \ell ^ { + } ( s _ { v } ) + ( 1 - \pi ) \ell ^ { - } ( s _ { v } ) \right]\tag{9}
$$

By circumventing the forced assignment of negative labels onto unverified accounts, this prior-weighted optimization empowers MINT to accurately identify malicious actors under strictly limited supervision.

Table 1: Statistics of the financial transaction datasets.
<table><tr><td>Dataset</td><td># Nodes</td><td># Edges</td><td># Anomaly (Rate)</td></tr><tr><td>RealFT</td><td>1,800,106</td><td>4,435,890</td><td>1,004 (0.056%)</td></tr><tr><td>Elliptic</td><td>822,942</td><td>2,868,964</td><td>4,280 (0.520%)</td></tr><tr><td>AMLSim</td><td>10,000</td><td>1,323,234</td><td>506 (5.060%)</td></tr></table>

## 4 Experiments

## 4.1 Experimental Settings

Datasets and Metrics. To emulate real-world supervision scenarios, we evaluate MINT within a PU learning framework across three financial transaction networks (detailed in Table 1): a proprietary bank transfer dataset (RealFT), a cryptocurrency network (Elliptic) [8], and a synthetic anti-money laundering dataset (AML-Sim) [18]. RealFT consists of de-identified bank transfer records provided by a financial institution and covers a three-month observation period. The dataset contains tabular transaction records describing fund transfers between accounts, while illicit accounts are identified through manual investigation by domain experts. Given the extreme class imbalance, we prioritize AUPRC to robustly assess illicit ranking capabilities [16], supplemented by AUROC and F1-score [4, 9], with all reported metrics averaged over five independent trials to guarantee statistical stability.

Comparative Methods. To evaluate MINT, we compare it against seven baselines across two paradigms. For general-purpose GNNs, we select GCN [25], GAT [19], and GraphSAGE [11], representing foundational approaches that capture structural dependencies via isotropic neighborhood aggregation, attention mechanisms, and localized sampling, respectively. To assess specialized GAD capabilities, we include advanced architectures tailored for complex network challenges. FRAUDRE [24] handles graph inconsistency and label imbalance via fraud-aware convolution and an imbalance-oriented classifier. ConsisGAD [2] improves limitedsupervision anomaly detection using a homophily-aware backbone with learnable consistency augmentation. SEC-GFD [22] addresses heterophily through spectrum-enhanced filtering and local environmental constraints. DIAM [5] captures multi-scale behavioral discrepancies between connected nodes in cryptocurrency networks.

Table 2: Performance comparison across three datasets. Results are reported as mean ± standard deviation.
<table><tr><td></td><td colspan="3">RealFT</td><td colspan="3">AMLSim</td><td colspan="3">Elliptic</td></tr><tr><td>Method</td><td>F1</td><td>AUPRC</td><td>AUROC</td><td>F1</td><td>AUPRC</td><td>AUROC</td><td>F1</td><td>AUPRC</td><td>AUROC</td></tr><tr><td>GCN</td><td> $2 7 . 6 2 { \pm } 2 . 0 6$ </td><td> $2 2 . 3 0 { \pm } 0 . 9 6$ </td><td> $7 8 . 3 3 { \pm } 0 . 4 6$ </td><td> $3 6 . 2 3 { \pm } 1 . 5 8 $ </td><td> $3 7 . 3 8 { \pm } 1 . 3 0 $ </td><td> $6 9 . 5 8 { \pm } 1 . 0 7$ </td><td> $1 8 . 2 5 { \pm } 1 . 2 5 $ </td><td> $1 0 . 8 2 { \pm } 0 . 5 9 $ </td><td> $8 2 . 8 3 { \pm } 0 . 6 8$ </td></tr><tr><td>GAT</td><td> $1 6 . 7 9 { \scriptstyle \pm 3 . 2 2 }$ </td><td> $1 2 . 6 9 { \pm } 2 . 0 1 $ </td><td> $7 0 . 7 2 { \pm } 4 . 1 2$ </td><td> $3 4 . 9 9 { \scriptstyle \pm 4 . 5 7 }$ </td><td> $2 5 . 9 7 { \scriptstyle \pm 3 . 5 0 }$ </td><td> $6 4 . 3 8 { \pm } 6 . 4 2 $ </td><td> $1 1 . 4 1 { \pm } 4 . 9 9$ </td><td> $8 . 6 4 \pm 1 . 5 2 $ </td><td> $8 0 . 6 9 { \pm } 1 . 9 9 $ </td></tr><tr><td>GraphSAGE</td><td> $4 9 . 4 1 { \pm } 2 . 0 4$ </td><td> $5 1 . 0 3 { \pm } 1 . 2 5 $ </td><td> $8 6 . 1 0 { \pm } 0 . 7 8$ </td><td> $8 0 . 8 1 { \pm } 1 . 6 2 $ </td><td> $8 7 . 8 2 { \pm } 0 . 5 5 $ </td><td> $9 3 . 6 8 { \pm } 0 . 2 9 $ </td><td> $3 5 . 7 6 { \pm } 1 . 7 6$ </td><td> $3 0 . 6 9 { \pm } 1 . 2 1 $ </td><td> $9 1 . 8 4 { \pm } 0 . 1 5 $ </td></tr><tr><td>SEC-GFD</td><td> $3 7 . 0 0 { \pm } 1 . 1 2 $ </td><td> $3 6 . 8 6 { \pm } 1 . 7 3 $ </td><td> $8 3 . 2 2 { \pm } 0 . 8 2 $ </td><td> $8 1 . 4 8 { \pm } 0 . 7 2 $ </td><td> $8 7 . 5 6 { \pm } 0 . 5 3 $ </td><td> $9 2 . 5 9 { \pm } 0 . 3 6 $ </td><td> $3 8 . 8 6 { \pm } 0 . 3 5 $ </td><td> $3 4 . 8 1 { \pm } 1 . 0 7 $ </td><td> $9 1 . 9 9 { \pm } 0 . 1 9$ </td></tr><tr><td>FRAUDRE</td><td> $4 3 . 4 1 \pm 2 . 9 5$ </td><td> $4 3 . 3 1 { \pm } 2 . 3 8 $ </td><td> $8 5 . 9 2 { \pm } 0 . 8 1 $ </td><td> $8 3 . 2 7 { \pm } 0 . 5 7 $ </td><td> $8 7 . 6 3 { \pm } 0 . 6 0 $ </td><td> $9 2 . 6 3 { \pm } 0 . 6 2 $ </td><td> $3 5 . 8 0 { \pm } 1 . 0 8 $ </td><td> $3 2 . 6 3 { \pm } 1 . 0 8 $ </td><td> $9 1 . 8 0 { \pm } 0 . 2 2 $ </td></tr><tr><td>ConsisGAD</td><td> $4 7 . 1 9 { \pm } 3 . 3 0 $ </td><td> $5 1 . 1 3 { \pm } 2 . 9 5 $ </td><td> $8 6 . 0 0 { \pm } 1 . 2 0 $ </td><td> $8 2 . 0 6 { \pm } 0 . 8 8$ </td><td> $8 6 . 7 5 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $9 1 . 8 1 { \pm } 0 . 2 3 $ </td><td> $2 9 . 8 0 { \pm } 0 . 5 4 $ </td><td> $2 2 . 0 7 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $8 9 . 2 6 { \pm } 0 . 2 1 $ </td></tr><tr><td>DIAM</td><td> $5 8 . 1 5 { \pm } 5 . 6 1 $ </td><td> $6 1 . 7 6 { \pm } 6 . 7 4$ </td><td> $9 2 . 1 4 \pm 1 . 4 8$ </td><td> $7 9 . 6 0 { \pm } 0 . 6 7$ </td><td> $8 4 . 2 5 { \pm } 2 . 3 2 $ </td><td> $9 2 . 3 4 { \pm } 0 . 8 6$ </td><td> $4 0 . 6 5 { \pm } 1 . 1 7$ </td><td> $3 8 . 7 3 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $9 2 . 1 4 { \pm } 0 . 8 0 $ </td></tr><tr><td>MINT</td><td> $7 0 . 1 3 { \pm } 2 . 0 2$ </td><td> $7 6 . 9 4 \pm 1 . 2 5$ </td><td> $9 5 . 3 7 { \pm } 0 . 3 6 $ </td><td> $8 0 . 3 4 { \pm } 1 . 1 4 $ </td><td> $8 7 . 0 3 { \pm } 2 . 2 7 $ </td><td> $9 3 . 2 9 { \pm } 0 . 3 4 $ </td><td> $5 6 . 9 0 { \scriptstyle \pm 2 . 8 1 }$ </td><td> $5 5 . 3 0 { \pm } 4 . 3 9$ </td><td> $9 7 . 1 4 { \pm } 0 . 4 6 $ </td></tr></table>

Implementation Details. Simulating real-world label scarcity, RealFT uses full annotations, while AMLSim and Elliptic retain only 30% of verified illicit nodes as known positives, with 10% held out for threshold and PU-prior tuning. MINT is implemented in PyTorch with 4 layers, dropout 0.3, hidden dimensions of 40/56/30 for RealFT/AMLSim/Elliptic, and trained for 300 epochs using Adam with a learning rate of0.005 and weight decay o $\mathrm { 5 } \times 1 0 ^ { - 4 }$ on NVIDIA TITAN RTX GPUs.

## 4.2 Comparative Results

Table 3: Ablation study of MINT on the RealFT dataset.
<table><tr><td>Module</td><td>Variant</td><td>F1</td><td>AUPRC</td><td>AUROC</td></tr><tr><td>MINT</td><td>一</td><td>70.13</td><td>76.94</td><td>95.37</td></tr><tr><td rowspan="2">Direction</td><td>w/o Outgoing</td><td>38.14</td><td>36.90</td><td>88.02</td></tr><tr><td>w/o Incoming</td><td>44.68</td><td>49.43</td><td>87.56</td></tr><tr><td rowspan="3">Edge Signal</td><td>w/o LIF</td><td>67.85</td><td>78.86</td><td>95.64</td></tr><tr><td>w/o GTR</td><td>63.04</td><td>78.10</td><td>95.59</td></tr><tr><td> $\bf { w } / o \ L I F { + } \bf { G } \mathrm { T R }$ </td><td>68.42</td><td>74.15</td><td>93.85</td></tr></table>

As presented in Table 2, MINT exhibits a substantial performance advantage on the real-world RealFT and Elliptic datasets. These results achieve substantial absolute AUPRC gains of 15.18% and 16.57% over DIAM, the strongest specialized baseline, empirically validating the efectiveness of our proposed framework under extreme label scarcity. Conversely, foundational GNN models inherently underperform in such contexts. This limitation confirms that strict homophily-based neighborhood aggregation is highly vulnerable to hub-induced over-smoothing, which blurs subtle anomalous signals into dominant benign hubs, and remains incapable of capturing asymmetric monetary flows. While specialized models like FRAUDRE and DIAM demonstrate robust performance on real-world datasets, GraphSAGE secures a marginal lead on the synthetic AMLSim dataset via localized feature sampling. Nevertheless, MINT remains highly competitive even in this scenario, reafirming its overall robustness and decisive superiority for illicit account detection across diverse financial networks.

## 4.3 Ablation Study

![](images/276cb7ad38e03f3e95b6498d9c285a726dad48a5d2192a97e80f8bc7550bc023.jpg)

![](images/4a3b2f3a3448560ab93326076bb9c494a19908d71c4591ede0c56a73c5a23a29.jpg)

We conduct an ablation study on the RealFT dataset to assess the contribution of each major component in MINT. As shown in Table 3, eliminating either the incoming or outgoing aggregation paths triggers severe AUPRC degradation, empirically substantiating the necessity ofmodeling asymmetric monetary flows. Furthermore, ablating the structural edge reweighting strategy inherently degrades overarching AUPRC and AUROC, while removing its individual components induces decline in F1 scores. These results confirm that jointly calibrating local interaction frequency and global transaction rarity is indispensable for robust illicit account detection.

(a) RealFT (Raw)  
(b) Elliptic (Raw)  
![](images/010ecefd563e917bfdc8ce505cf1eedfba135ca33474670c156b8266b40bde11.jpg)

![](images/36d177c7d5a0e1bc247875beb468e37c85048a2256f0a4e6d64e4ec66a703593.jpg)  
(c) RealFT (MINT)  
(d) Elliptic (MINT)  
Figure 3: t-SNE visualization of node representations learned by MINT on RealFT and Elliptic.

## 4.4 Visualization Analysis

Figure 3 compares initial features and MINT embeddings via t-SNE. Initially, illicit and unlabeled nodes are heavily entangled, reflecting severe structural inconsistency in transaction graphs. Conversely, MINT separates these illicit nodes into more compact and distinguishable regions. This qualitative evaluation provides intuitive evidence that MINT efectively mitigates structural inconsistency and hub-induced over-smoothing, yielding discriminative representations for rare financial anomalies.

## 5 Conclusion

In this paper, we introduce MINT, a graph neural framework resolving structural inconsistency and extreme label scarcity in financial anomaly detection. We integrate a structural edge reweighting strategy and a discrepancy-aware attentive message passing scheme to neutralize the structural inconsistency caused by deceptive con nections and transaction hub dominance, while capturing feature deviations across neighborhoods. Extensive experiments across diverse large-scale transaction benchmarks demonstrate that MINT, optimized under a Positive-Unlabeled learning framework, achieves superior efectiveness and robustness.

## Acknowledgments

This work was supported in part by E.SUN Commercial Bank, Ltd., Taiwan, under Grant Number 202508-NTU-03. The authors would like to express their gratitude to the collaborators at E.SUN Commercial Bank, Yi-Ren Yeh, Fu-Chang Sun, and Wei Luok Ngu, for their valuable support and insights. Any findings and implications in the paper do not necessarily reflect those of the sponsor.

## GenAI Usage Disclosure

We declare that all content within this paper was independently conducted by the authors. GenAI tools were exclusively used for language polishing, improving grammatical correctness, and refining the flow of the text.

## References

[1] Jovan Blanuša, Maximo Cravero Baraja, Andreea Anghel, Luc von Niederhäusern, Erik Altman, Haris Pozidis, and Kubilay Atasu. 2024. Graph Feature Preprocessor: Real-time Subgraph-based Feature Extraction for Financial Crime Detection. In Proceedings ofthe 5th ACM International Conference on AI in Finance. 222–230.

[2] Nan Chen, Zemin Liu, Bryan Hooi, Bingsheng He, Rizal Fathony, Jun Hu, and Jia Chen. 2024. Consistency Training with Learnable Data Augmentation for Graph Anomaly Detection with Limited Supervision. In International Conference on Learning Representations, Vol. 2024. 30180–30207.

[3] Dawei Cheng, Yao Zou, Sheng Xiang, and Changjun Jiang. 2025. Graph neural networks for financial fraud detection: a review. Frontiers ofComputer Science 19, 9 (2025).

[4] Jesse Davis and Mark Goadrich. 2006. The relationship between Precision-Recall and ROC curves. In Proceedings ofthe 23rd International Conference on Machine Learning. 233–240.

[5] Zhihao Ding, Jieming Shi, Qing Li, and Jiannong Cao. 2024. Efective illicit account detection on large cryptocurrency multigraphs. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 457–466.

[6] Marthinus Christofel Du Plessis, Gang Niu, and Masashi Sugiyama. 2015. Convex formulation for learning from positive and unlabeled data. In Proceedings ofthe 32nd International Conference on Machine Learning - Volume 37. 1386–1394.

[7] Charles Elkan and Keith Noto. 2008. Learning classifiers from only positive and unlabeled data. In Proceedings ofthe 14th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 213–220.

[8] Youssef Elmougy and Ling Liu. 2023. Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin Network for Financial Forensics. In Proceedings

ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3979–3990.

[9] Tom Fawcett. 2006. An introduction to ROC analysis. Pattern Recognition Letters 27, 8 (2006), 861–874.

[10] Zheng Gong, Guifeng Wang, Ying Sun, Qi Liu, Yuting Ning, Hui Xiong, and Jingyu Peng. 2023. Beyond Homophily: Robust Graph Anomaly Detection via Neural Sparsification. In Proceedings ofthe Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23. 2104–2113.

[11] William L. Hamilton, Rex Ying, and Jure Leskovec. 2017. Inductive representation learning on large graphs. In Proceedings of the 31st International Conference on Neural Information Processing Systems. 1025–1035.

[12] Waleed Hilal, S. Andrew Gadsden, and John Yawney. 2022. Financial Fraud: A Review of Anomaly Detection Techniques and Recent Advances. Expert Syst. Appl. 193, C (May 2022), 34 pages.

[13] Ziqi Liu, Chaochao Chen, Xinxing Yang, Jun Zhou, Xiaolong Li, and Le Song. 2018. Heterogeneous Graph Neural Networks for Malicious Account Detection. In Proceedings of the 27th ACM International Conference on Information and Knowledge Management. 2077–2085.

[14] Xiaoxiao Ma, Ruikun Li, Fanzhen Liu, Kaize Ding, Jian Yang, and Jia Wu. 2024. Graph Anomaly Detection with Few Labels: A Data-Centric Approach. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2153–2164.

[15] Xiaoxiao Ma, Jia Wu, Shan Xue, Jian Yang, Chuan Zhou, Quan Z. Sheng, Hui Xiong, and Leman Akoglu. 2023. A Comprehensive Survey on Graph Anomaly Detection With Deep Learning. IEEE Transactions on Knowledge and Data Engineering 35, 12 (2023), 12012–12038.

[16] Takaya Saito and Marc Rehmsmeier. 2015. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE 3 (2015), 1–21.

[17] Guangxin Su, Weitong Chen, and Miao Xu. 2021. Positive-Unlabeled Learning from Imbalanced Data. In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21. 2995–3001.

[18] Toyotaro Suzumura and Hiroki Kanezashi. 2021. Anti-Money Laundering Datasets: InPlusLab Anti-Money Laundering Datasets. https://github.com/IBM/ AMLSim/.

[19] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. 2018. Graph Attention Networks. arXiv:1710.10903

[20] Daixin Wang, Jianbin Lin, Peng Cui, Quanhui Jia, Zhen Wang, Yanming Fang, Quan Yu, Jun Zhou, Shuang Yang, and Yuan Qi. 2019. A Semi-Supervised Graph Attentive Network for Financial Fraud Detection. In 2019 IEEE International Conference on Data Mining (ICDM). 598–607.

[21] Mark Weber, Giacomo Domeniconi, Jie Chen, Daniel Karl I. Weidele, Claudio Bellei, Tom Robinson, and Charles E. Leiserson. 2019. Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics.

[22] Fan Xu, Nan Wang, Hao Wu, Xuezhi Wen, Xibin Zhao, and Hai Wan. 2024. Revisiting graph-based fraud detection in sight of heterophily and spectrum. In Proceedings of the Thirty-Eighth AAAI Conference on Artificial Intelligence and Thirty-Sixth Conference on Innovative Applications ofArtificial Intelligence and Fourteenth Symposium on Educational Advances in Artificial Intelligence. Article 1025, 9 pages.

[23] Bi-Cheng Yan, Jhih-Rong Guo, Yin-Ju Wu, and Berlin Chen. 2026. LABRA: A label-free graph neural framework for illicit account detection in large-scale blockchain transaction networks. In Proceedings ofthe 35th ACM International Conference on Information and Knowledge Management.

[24] Ge Zhang, Jia Wu, Jian Yang, Amin Beheshti, Shan Xue, Chuan Zhou, and Quan Z. Sheng. 2021. FRAUDRE: Fraud Detection Dual-Resistant to Graph Inconsistency and Imbalance. In 2021 IEEE International Conference on Data Mining (ICDM). 867–876.

[25] Haiqi Zhang, Guangquan Lu, Mengmeng Zhan, and Beixian Zhang. 2022. Semi-Supervised Classification of Graph Convolutional Networks with Laplacian Rank Constraints. Neural Process. Lett. 54, 4 (2022), 2645–2656.

[26] Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, and Danai Koutra. 2020. Beyond homophily in graph neural networks: current limitations and efective designs. In Proceedings ofthe 34th International Conference on Neural Information Processing Systems. Article 653, 12 pages.