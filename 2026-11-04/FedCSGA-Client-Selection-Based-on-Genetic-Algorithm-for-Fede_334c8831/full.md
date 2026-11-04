# FedCSGA: Client Selection Based on Genetic Algorithm for Federated Learning

Ghani Haidera, Majid Kundrooa,\*, Leo Zhangb, Jinchul Choic, Taehong Kima,\*

aSchool of Information and Communication Engineering, Chungbuk National University, Cheongju, Republic of Korea bSchool of Information and Communication Technology, Griffth University, Brisbane, Australia cElectronics and Telecommunications Research Institute, Daejeon, Republic of Korea

## Abstract

Federated Learning (FL) enables collaborative training of machine learning models across distributed clients while preserving data privacy. In practical deployments, FL relies on selecting a subset of clients in each communication round to reduce communication cost, mitigate stragglers, and improve training efficiency under heterogeneous data and system conditions. In heterogeneous edge computing environments, both statistical heterogeneity and system heterogeneity significantly affect convergence efficiency and model performance. However, existing client selection methods often address these factors independently, overlooking their coupled and dynamic impact across communication rounds. To address this limitation, we propose FedCSGA, a Client Selection method for Federated learning based on a Genetic Algorithm. FedCSGA encodes client subsets as chromosomes and evaluates these subsets through a non-linear fitness function that jointly integrates resource-aware metrics and history-aware participation weighting. This design captures subset-level interactions, mitigates persistent selection bias, and balances statistical utility with system efficiency. Extensive experiments on FMNIST, CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrate that FedCSGA consistently outperforms state-of-the-art baselines across diverse heterogeneity settings, achieving accuracy improvements of up to 7.0% and faster convergence with fewer communication rounds. These results validate the effectiveness of jointly modeling statistical and system factors to achieve robust and efficient learning in heterogeneous federated environments.

Keywords: Federated learning, Statistical heterogeneity, System heterogeneity, Client selection, Edge computing

## 1. Introduction

Federated Learning (FL) has emerged as a transformative paradigm for privacy-preserving distributed machine learning [1]. Its deployment increasingly spans heterogeneous edge computing environments, where mobile, IoT, and edge-cloud continuum systems jointly participate in model training. Such environments impose constraints shaped by real-time processing demands, limited resources at edge clients, and dynamic network conditions [2]. Due to resource limitations and privacy requirements, FL typically selects only a subset of available clients in each communication round, allowing these clients to perform local model updates without sharing raw data [3]. However, partial client participation introduces significant challenges arising from both statistical and system heterogeneity [4, 5].

Statistical heterogeneity arises when client datasets vary in size and originate from diverse distributions, resulting in non-Independent and Identically Distributed (non-IID) data [6]. This phenomenon can degrade global model performance, increase gradient variance, and slow convergence [7]. System heterogeneity refers to variations in computational resources, communication bandwidth, and processing capabilities across clients [8]. These disparities often lead to straggler effects, where slower or resource-constrained clients delay global synchronization and reduce training efficiency [9]. While random client selection remains the predominant strategy in practice, it fails to explicitly account for these heterogeneities, frequently resulting in inefficient resource utilization and unstable convergence behavior [10].

To address these challenges, recent studies have explored client selection strategies that explicitly consider either statistical utility or system characteristics [11, 12]. Some approaches prioritize clients based on data-related metrics, such as loss, gradient norms, or data quality [13, 14], while others emphasize system-level factors, including computational capacity [15, 16] or communication constraints [17]. However, most existing methods address these heterogeneity factors in isolation and fail to capture the subset-level statistical diversity and system compatibility [18], particularly under repeated participation across rounds.

In this work, we propose FedCSGA, a client selection framework for federated learning that leverages the Genetic Algorithm (GA) to select high-quality participating clients in each training round. In FedCSGA, each candidate solution encodes a fixed-size subset of participating clients as a chromosome, and its quality is evaluated by a fitness value that jointly considers statistical properties (e.g., data quality) and system resources (e.g., computational power). Importantly, the fitness function incorporates a history-aware participation weighting that introduces subset-level and cross-round dependencies among clients. This creates an inherently non-additive selection problem unsuitable for simple greedy ranking. Through iterative evolutionary operations – including selection, crossover, and mutation – FedCSGA explores the combinatorial search space to identify high-quality client subsets that balance model contribution and system efficiency. This design overcomes the limitations of existing selection methods and provides a robust subset-level strategy for adaptive client participation in heterogeneous FL settings. The main contributions of this work are summarized as follows:

• We propose FedCSGA, a GA-based client selection framework that formulates client selection as a subset-level combinatorial problem, explicitly addressing both statistical and system heterogeneity in federated learning.

• We design a fitness function that integrates resource-aware metrics with history-aware participation weighting, capturing non-linear and cross-round dependencies and enabling balanced and efficient client selection.

• Extensive experiments on FMNIST, CIFAR-10, CIFAR-100, and Tiny-Imagenet demonstrate that FedCSGA improves test accuracy by an average of 1.0 – 7.0% over strong baselines across both IID and non-IID settings, while maintaining stable convergence and efficiency.

## 2. Background

This section reviews the fundamental concepts of FL, the role of client selection, and the principles of GA as heuristic search mechanisms for combinatorial problems. These components form the methodological foundation of the proposed approach.

## 2.1. Federated Learning

FL is a distributed machine learning paradigm in which a global model is collaboratively trained across a set of decentralized clients, each holding private local data that cannot be shared directly due to privacy, security, or regulatory constraints [19, 1]. In a typical FL workflow, a central server initializes a global model and orchestrates training over multiple communication rounds. In each round, a subset of clients downloads the current global model, performs local training on private data, and uploads model updates to the server, where they are aggregated to update the global model [3].

A defining characteristic of FL is partial participation, where only a fraction of the available clients participate in any given communication round. Partial participation is necessary due to device availability, system limitations, and straggler effects in large-scale deployments [2, 9]. While it improves scalability, partial participation introduces challenges related to convergence efficiency and model generalization, particularly in heterogeneous environments [4, 5].

FL systems are commonly affected by two forms of heterogeneity. Statistical heterogeneity arises when client datasets differ in size, class distribution, or feature characteristics, leading to non-IID data across clients [6]. System heterogeneity refers to variations in computational capability, memory, energy constraints, and communication latency [8]. These heterogeneities jointly influence training stability, convergence behavior, and system efficiency, making coordination among clients a central challenge in FL [7].

## 2.2. Client Selection in Federated Learning

Client selection is a core mechanism in FL that determines which clients participate in each communication round. Given a pool of available clients, the server must select a subset that balances learning performance, training efficiency, and system constraints. Early FL frameworks, such as FedAvg [19, 20], adopt uniform random client selection. While simple and unbiased, random selection ignores client-level differences in data utility and system resources, often resulting in slow convergence and inefficient resource utilization in heterogeneous settings [21, 10].

The primary objective of client selection is to mitigate the adverse effects of statistical and system heterogeneity while maintaining scalability. From a statistical perspective, selecting clients with informative, diverse, or representative data can improve gradient quality and accelerate convergence [14, 13]. From a system perspective, avoiding resource-constrained or unreliable clients can reduce straggler delays and communication overhead [22, 23]. These objectives, however, are often conflicting, as clients with valuable data may exhibit limited system capabilities, whereas resource-rich clients may contribute less informative updates [11, 12].

Consequently, client selection can be formulated as an optimization problem where the goal is to identify a subset of clients that provides a favorable trade-off between learning utility and system efficiency. The search space grows combinatorially with the number of clients, rendering exhaustive search infeasible in practical FL deployments [17]. This motivates the use of heuristic search methods that can efficiently explore the space of client subsets under multiple competing criteria.

## 2.3. Genetic Algorithms

Genetic Algorithms are population-based, stochastic optimization methods inspired by the principles of natural evolution, including selection, crossover, and mutation [24]. A GA maintains a population of candidate solutions, referred to as chromosomes, each encoding a potential solution to the optimization problem. The quality of each chromosome is evaluated using a fitness function, which guides the selection of candidates for reproduction across successive generations. Through iterative application of genetic operators, GAs balance exploration of the search space with exploitation of promising solutions, making them suitable for efficiently approximating high-quality solutions within complex and combinatorial search spaces.

These properties make GAs a natural fit for FL client selection, where the utility of a selected subset depends not only on individual client characteristics but also on interactions among clients in terms of statistical diversity and system compatibility [18]. By encoding client subsets as chromosomes and evaluating them through a fitness function that reflects multiple selection criteria, GAs provide a flexible framework for approximating effective client selection strategies in heterogeneous FL environments.

## 3. Related Work

In the standard FedAvg framework, participating clients are selected uniformly at random in each communication round [20]. While random selection is simple and unbiased, it fails to account for differences in data distributions and system capabilities across clients, often resulting in slow convergence and inefficient resource utilization in heterogeneous environments [21].

To mitigate statistical heterogeneity, several approaches prioritize clients based on datarelated utility. Recent studies have also introduced diversity-aware [7] or submodular optimization frameworks [25] to select representative client subsets that maximize information gain. Active Federated Learning (AFL) [14] assigns higher importance to clients with larger training losses, under the assumption that such clients contribute more informative updates. The Powerof-Choice (PowD) [13] strategy further refines this idea by selecting the highest-loss clients from randomly sampled subsets, improving convergence behavior under non-IID data distributions. Contribution-aware methods, such as Greedy Shapley (GreedyFed) [17], model client participation as a cooperative game and rank clients based on their marginal contribution to validation performance. While effective in improving statistical efficiency, these methods typically ignore system-level constraints and often incur substantial computational overhead due to repeated evaluation of client contributions.

In contrast, system and resource-aware client selection strategies focus on improving training efficiency by accounting for computation latency, energy consumption, or communication cost. Specifically, FedCS [22] and FedHeur [23] are designed for mobile edge computing environments to select clients based on training time or task offloading considerations. Although these methods improve system efficiency, they generally do not explicitly consider data quality or address selection bias under non-IID data distributions. More recently, reinforcement learning-based client selection methods have been proposed to adapt to dynamic heterogeneous environments [18]. These approaches learn selection policies that balance accuracy and latency over time, but they often require extensive training, carefully designed state and reward representations, and introduce additional complexity at the server.

Overall, most existing client selection methods address statistical or system heterogeneity in isolation, as summarized in Table 1. FedCSGA differs from prior work by formulating client selection as a combinatorial optimization problem over entire client subsets and employing GA as a metaheuristic search strategy. Unlike greedy baselines or single-criterion heuristics, the proposed approach evaluates candidate subsets using a fitness function that jointly reflects statistical contribution and system resource constraints. This enables FedCSGA to approximate balanced client selections that improve convergence behavior and training efficiency in heterogeneous federated learning settings.

Table 1: Comparison of Client Selection Methods in Federated Learning
<table><tr><td>Work</td><td>Problem Setting</td><td>Heterogeneity</td><td>Optimized Metrics</td><td>Methodology</td><td>Limitations</td><td></td></tr><tr><td>FedAvg [19]</td><td>Standard FL</td><td>None</td><td>Convergence</td><td>Uniform random lection</td><td>se- system heterogeneity</td><td>Ignores statistical and</td></tr><tr><td>AFL [14]</td><td>Non-IID FL</td><td>Statistical</td><td>Convergence speed</td><td>Loss-based weighting</td><td>client straints</td><td>Neglects system con-</td></tr><tr><td>PowD [13]</td><td>Non-IID FL</td><td>Statistical</td><td>Convergence speed, accuracy</td><td>Highest-loss selection from random subsets</td><td>Biased nores</td><td>selection, ig- latency and</td></tr><tr><td>GreedyFed [17]</td><td>Contribution- aware FL</td><td>Statistical</td><td>Convergence, accu- racy (via validation</td><td>Greedy Shapley value ranking</td><td>computation Requires</td><td>validation data, greedy selection</td></tr><tr><td>FedCS [22]</td><td>Mobile Edge FL</td><td>System</td><td>loss) Training time, la- tency</td><td>Resource-aware greedy client selection</td><td></td><td>with limited exploration Requires accurate re- source info, ignores sta-</td></tr><tr><td>FedHeur [23]</td><td>Task offloading FL</td><td>System</td><td>Latency, energy ef- ficiency</td><td>Multi-heuristic selec- tion</td><td>Heuristic-driven, lacks joint optimization</td><td>tistical heterogeneity</td></tr><tr><td>RL-based [18]</td><td>Dynamic FL</td><td>Statistical &amp; Sys- tem</td><td>Accuracy, latency</td><td>Reinforcement learn- ing policy</td><td>High training cost, sen- sitive to reward design</td><td></td></tr></table>

## 4. Proposed method

We propose FedCSGA, a resource-aware client selection algorithm based on GA, designed to address both statistical and system heterogeneity in FL. The key mathematical symbols used throughout the formal description are summarized in Table 2.

## 4.1. System and Modeling Assumptions

We consider a non-adversarial FL setting in which clients accurately report lightweight resourcerelated metrics to the server at the beginning of each communication round. These metrics include statistical indicators derived from local training (e.g., dataset size and loss-related signals) and system-level configurations (e.g., computational latency). While this reporting phase requires communication from all available clients in the candidate pool unlike random selection, the transmitted metadata consists of minimal scalar values (bytes-level), which is negligible compared to the model parameters (megabytes-level). Thus, the additional bandwidth consumption is limited relative to the overall communication cost.

The server is assumed to be honest-but-curious and uses the reported metrics solely for client selection without manipulating the learning process. Regarding data privacy, standard secure communication channels (e.g., SSL/TLS) and secure aggregation protocols can be established to protect the transmission of both model updates and resource metadata. Explicit privacypreserving mechanisms, such as differential privacy, are independent of the proposed selection strategy and can be incorporated if required. The design does not explicitly consider adversarial behavior, strategic misreporting, or Byzantine clients.

The system operates under a synchronous FL protocol in which client selection, local training, and aggregation occur in discrete rounds. To ensure robustness against network latency clients failing to report metrics within a specified time window are excluded from the candidate pool for the current round. Finally, all clients are expected to be capable of completing the assigned local training workload once selected, ensuring that feasibility constraints are satisfied during the learning process.

Table 2: Frequently used notations in FedCSGA
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td>W</td><td>global model parameter in  $\mathbb { R } ^ { d }$ </td></tr><tr><td> $F ( w )$ </td><td>global objective function</td></tr><tr><td> $F _ { i } ( w )$ </td><td>local objective function of the i-th client</td></tr><tr><td> $D _ { i }$ </td><td>local dataset of the i-th client</td></tr><tr><td> $\delta _ { j }$ </td><td>a data sample from the local dataset  $D _ { i }$ </td></tr><tr><td> $\dot { L ( w , \delta _ { j } ) }$ </td><td>loss function with parameter w and data sample  $\delta _ { j }$ </td></tr><tr><td> $\hat { L } ( w , \delta _ { j } )$ </td><td>normalized loss over all data samples of a client</td></tr><tr><td> $T$ </td><td>total number of training rounds</td></tr><tr><td> $C$ </td><td>set of all available clients</td></tr><tr><td> $K$ </td><td>total number of clients</td></tr><tr><td> $\boldsymbol { w } _ { i } ^ { t }$ </td><td>local model update of the i-th client in round t</td></tr><tr><td> $\boldsymbol { w } ^ { \dot { t } + 1 }$ </td><td>updated global model parameter at round t + 1</td></tr><tr><td> $\alpha$ </td><td>dirichlet concentration parameter</td></tr><tr><td> $\eta$ </td><td>learning rate of the local model</td></tr><tr><td> $S _ { t }$ </td><td>subset of m selected clients in round t</td></tr><tr><td> $s$ </td><td>search space for all possible client subsets  $S _ { t }$ </td></tr><tr><td> $S _ { t } ^ { * }$ </td><td>selected subset of clients in round t</td></tr><tr><td> $W ( c _ { i } )$ </td><td>history-aware selection weighting factor</td></tr><tr><td> $\operatorname { R e s } ( c _ { i } )$ </td><td>total resources of client  $c _ { i }$ </td></tr><tr><td> ${ \mathrm { R e s } } _ { \mathrm { s t a t } } ( c _ { i } )$ </td><td>statistical resources of client  $c _ { i }$ </td></tr><tr><td> ${ \sf R e s } _ { \mathrm { s y s } } ( c _ { i } )$ </td><td>system resources of client c</td></tr><tr><td> $f _ { c _ { i } }$ </td><td>cumulative selection count of client  $c _ { i }$ </td></tr><tr><td> $\gamma$ </td><td>decay rate for history-aware weighting</td></tr><tr><td> $\tau _ { i }$ </td><td>compute time of client  $c _ { i }$  for a local model update</td></tr><tr><td> $\lambda _ { r }$ </td><td>weighting factor for statistical resources</td></tr><tr><td> $\lambda _ { s }$ </td><td>weighting factor for system resources</td></tr><tr><td> $\mathrm { F i t n e s s } ( S _ { t } )$ </td><td>fitness function of a subset  $S _ { t }$ </td></tr><tr><td> $P$ </td><td>population size of the GA</td></tr><tr><td> $G$ </td><td>number of generation rounds of the GA</td></tr><tr><td> $\rho _ { p }$ </td><td>parent chromosomes selection fraction</td></tr><tr><td> $\rho _ { c }$ </td><td>child chromosomes selection fraction</td></tr></table>

## 4.2. Problem formulation

FL aims to optimize the global model parameters w ∈ $\mathbb { R } ^ { d }$ by minimizing the global objective function $F ( w )$ , which represents the aggregated performance across all K clients. The global objective function is defined as:

$$
F ( w ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } F _ { i } ( w ) ,\tag{1}
$$

where $F _ { i } ( w )$ denotes the local objective function of the i-th client, evaluated over its local dataset $D _ { i }$

$$
F _ { i } ( w ) = \frac { 1 } { | D _ { i } | } \sum _ { \delta _ { j } \in D _ { i } } L ( w , \delta _ { j } ) .\tag{2}
$$

Here, $L ( w , \delta _ { j } )$ represents the loss function with $w \in \mathbb { R } ^ { d } .$ and $\delta _ { j }$ denotes a data sample drawn from the local data distribution $D _ { i }$ . FL often involves partial client participation in each training round due to statistical heterogeneity and system resource constraints. Let the set of all available clients be $C = \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { K } \}$ . In each round $t ,$ a subset $S _ { t }$ of m clients $( | S _ { t } | = m \le | C | )$ is selected from the search space S, defined as:

$$
S = \{ S _ { t } \mid S _ { t } \subset C , \vert S _ { t } \vert = m \} .\tag{3}
$$

The objective is to minimize $F ( w )$ by selecting the highest quality subset $S _ { t } ^ { * }$ in each round by maximizing a fitness function that combines resource utility and history-aware participation weighting:

$$
S _ { t } ^ { * } = \arg \operatorname* { m a x } _ { S _ { t } \in S } \Bigg ( \sum _ { c _ { i } \in S _ { t } } \mathrm { R e s } ( c _ { i } ) \Bigg ) \cdot \frac { 1 } { | S _ { t } | } \sum _ { c _ { i } \in S _ { t } } W ( c _ { i } )\tag{4}
$$

where $W ( c _ { i } )$ is a history-aware selection weighting factor defined as

$$
W ( c _ { i } ) = e ^ { - \gamma f _ { c _ { i } } } ,\tag{5}
$$

and $\operatorname { R e s } ( c _ { i } )$ is the resource metric of client $c _ { i } ,$ comprising both statistical and system contributions

$$
\begin{array} { r } { \mathrm { R e s } ( c _ { i } ) = \lambda _ { r } \mathrm { R e s } _ { \mathrm { s t a t } } ( c _ { i } ) + \lambda _ { s } \mathrm { R e s } _ { \mathrm { s y s } } ( c _ { i } ) . } \end{array}\tag{6}
$$

Here, $f _ { c _ { i } }$ denotes the cumulative number of rounds in which client $c _ { i }$ has been selected, and γ is a decay rate that controls how strongly historical participation reduces the future selection weight of client $c _ { i } . \ R e s _ { \mathrm { s t a t } } ( c _ { i } )$ quantifies the statistical resource contribution of client $c _ { i }$ , while ${ \mathrm { R e s } } _ { \mathrm { s y s } } ( c _ { i } )$ represents its system-level resources. The weighting factors $\lambda _ { r }$ and $\lambda _ { s }$ control the influence of statistical and system resources, respectively. To maximize the history-aware resource utility in each communication round, we employ GA to select the best-performing subset $S _ { t } ^ { * }$ from the search space S. The GA initializes and evolves a population of candidate subsets $S _ { t } ^ { j }$ $( j = 1 , 2 , \ldots , | P | )$ over G generations, jointly accounting for client resource contributions and historical selection pressure. The algorithm ultimately converges to the subset $S _ { t } ^ { * }$ that achieves the highest history-aware resource utility, which is returned as the final client selection for the round.

## 4.3. Federated client selection using genetic algorithm

Our FedCSGA algorithm leverages client resource information for client selection, ensuring the participation of the best clients in each FL round. As shown in Figure 1, each client transmits its resource metrics to the server, which then uses GA for client selection, involving initialization, reproduction, and population updates. During reproduction, client subsets generated through parent selection, crossover, and mutation are evaluated over multiple generations to identify the highest quality client subset $S _ { t } ^ { * }$ . This process selects clients with high resource availability, improves model accuracy, and accelerates global model convergence.

![](images/ea891c63fcd6a8e8e00eeabf0b202465a15227c7cae6acbe0e958de6b4acd488.jpg)  
Figure 1: Overview of FedCSGA algorithm for each training round.

Algorithm 1 outlines the detailed steps of the procedure. At the beginning of each training round t, as described in line 5 of Algorithm 1, each client transmits its resource metrics (Res) to the central server. The aim is to select m clients for training using the client selection function specified in line 6. Lines 7–13 follow the standard procedure of ${ \mathrm { F L } } ,$ involving local training by selected clients and global model aggregation at the server. For client selection, the resource information collected from clients at the server is utilized in the client selection function (line 6) to initialize the GA generations, which perform the selection of m clients for the current round. The process consists of the following steps:

## 4.3.1. Initialization

The GA initializes a population $P _ { 0 }$ of P chromosomes with $\{ c _ { i } \mid c _ { i } \in C , i = 1 , 2 , \ldots , m \}$ in line 15, where each chromosome $S _ { t } ^ { j } \left( j = 1 , 2 , \ldots , | P | \right)$ represents a subset of m distinct clients selected from the available client pool C. Chromosomes are initialized by uniformly sampling m clients without replacement, ensuring that no duplicate clients appear within a subset by construction.

Each client $c _ { i }$ is associated with a scalar resource score $\operatorname { R e s } ( c _ { i } )$ that captures its expected contribution to the current training round. $\operatorname { R e s } ( c _ { i } )$ is computed as a weighted combination of client-specific statistical $\mathrm { R e s } _ { \mathrm { s t a t } }$ and system $\mathrm { R e s } _ { \mathrm { s y s } }$ metrics available at the server. By integrating both statistical and system-aware components, the fitness function guides the GA toward client subsets that enhance both learning effectiveness and system efficiency. The fitness of a candidate subset $S _ { t } ^ { j }$ is defined as

$$
\mathrm { F i t n e s s } ( S _ { t } ^ { j } ) = \left( \sum _ { c _ { i } \in S _ { t } ^ { j } } \mathrm { R e s } ( c _ { i } ) \right) \cdot \frac { 1 } { | S _ { t } ^ { j } | } \sum _ { c _ { i } \in S _ { t } ^ { j } } e ^ { - \gamma f _ { c _ { i } } } ,\tag{7}
$$

where $f _ { c _ { i } }$ denotes the cumulative selection count of client $c _ { i }$ up to round t, and $\gamma > 0$ is the decay rate for history-aware weighting.

Algorithm 1: FedCSGA   
1 Input: set of all available clients $C ,$ number of communication rounds T, number of participants   
m, number of total GA generations $G ,$ number of total chromosomes $P ,$ parent chromosomes   
fraction $\rho _ { p } ,$ child chromosomes fraction $\rho _ { c }$ , client resources ${ \mathrm { R e s } } ,$ statistical resource weight $\lambda _ { r } ,$   
system resources weight $\lambda _ { s }$   
2 Initialize: global model $w ^ { 0 } ;$   
3 Server Executes:   
4 for each round $t = 1 , 2 , \dots , T$ do   
5 Res ← resource metric Res(c) from all clients c in $C ;$   
6 $S _ { t } ^ { * } \gets$ ClientSelection(C, m, Res);   
7 for $i \in S _ { t } ^ { * }$ in parallel do   
8 send $\boldsymbol { w } ^ { t }$ to client i and initiate local training;   
9 end   
10 $w _ { 1 } ^ { t } , \cdots , w _ { m } ^ { t }$ ← collect $\boldsymbol { w } _ { i } ^ { t }$ from clients;   
11 $\begin{array} { r } { w ^ { t + 1 } \gets \sum _ { i = 1 } ^ { m } \frac { | D _ { i } | } { \sum _ { i = 1 } ^ { m } | D _ { i } | } w _ { i } ^ { t } ; } \end{array}$   
12 end   
13 return $w ^ { T } ;$   
14 ClientSelection(C, m, Res):   
15 $P _ { 0 } = \{ S _ { t } ^ { 1 } , S _ { t } ^ { 2 } , . . . , S _ { t } ^ { P } \}$ ← initialize each chromosome $S _ { t } ^ { j }$ with {ci $\mid c _ { i } \in C , i = 1 , 2 , \ldots , m \} ;$   
16 for $g = 1 , 2 , \ldots , G$ do   
// Parent Selection   
17 M ← choose $\rho _ { p } | P _ { g } |$ chromosomes $S _ { t } ^ { j }$ from $P _ { g }$ in descending order of Fitness $( S _ { t } ^ { j } )$   
$/ /$ Crossover   
18 for $\begin{array} { r } { q = 1 , 2 , \dots , \frac { \rho _ { c } | P _ { g } | } { 2 } } \end{array}$ do   
19 $P _ { 1 } , P _ { 2 } $ select the two highest-fitness chromosomes from a tournament of size $\rho _ { c } | P _ { g } |$   
sampled from $M ;$   
20 $L \gets \mathbf { A }$ random index L is sampled uniformly from $[ 1 , | S _ { t } ^ { j } | ] ;$   
21 $Y _ { 1 } , Y _ { 2 } \gets L$ prefix-based crossover preserving client uniqueness and subset cardinality m;   
22 $Y \gets \{ Y _ { 1 } , Y _ { 2 } \}$   
23 end   
$/ /$ Mutation   
24 for each $y \in Y$ do   
25 for each $c _ { i } \in y$ do   
26 replace $c _ { i }$ with probability $\frac { 1 } { | y | }$   
27 end   
28 end   
$/ /$ Population Update   
29 $P _ { g + 1 }  M \cup Y$   
30 end   
31 return S ‡ ← arg max $\cdot { s } _ { t } ^ { j } { \in } P _ { G }$ Fitness $( S _ { t } ^ { j } ) \dag$   
32 Fitness $( \boldsymbol { S } _ { t } ^ { j } ) \colon$   
33 return $\begin{array} { r } { \left( \sum _ { c _ { i } \in S _ { t } ^ { j } } \operatorname { R e s } ( c _ { i } ) \right) \cdot \frac { 1 } { | S _ { t } ^ { j } | } \sum _ { c _ { i } \in S _ { t } ^ { j } } e ^ { - \gamma f _ { c _ { i } } } ; } \end{array}$

This formulation introduces an explicit coupling between clients within a subset. Although statistical and system resources are linearly combined at the client level, their joint effect on selection emerges at the subset level through the non-linear history-aware weighting, preventing dominance by any single resource dimension. As a result, the fitness of a subset depends not only on the individual quality of its members but also on their collective selection history, discouraging repeatedly selecting the same group of high-resource clients and promoting long-term participation balance.

## 4.3.2. Reproduction

The GA evolves the population over multiple generations to improve the fitness of chromosomes. At each generation, the population is updated using elitist survival and stochastic reproduction. A fixed proportion of the top-performing chromosomes is preserved as parents to the next generation to maintain high-quality solutions. The remaining offspring are generated via tournament-based parent selection, which balances exploitation of high-fitness subsets with population diversity. Lines 16 - 28 explain the reproduction process where three main operations are performed: parent selection, crossover, and mutation.

• Parent Selection: In line 17, $\rho _ { p } | P _ { g } |$ chromosomes with the highest fitness values from the current population $P _ { g }$ are selected as parents, where $\rho _ { p }$ determines the fraction of chromosomes retained from the previous generation as parents. These parents serve as the basis for creating new child chromosomes with potentially better fitness values.

• Crossover: Lines 18 - 23 explain the crossover operation. Two parent chromosomes $P _ { 1 }$ and $P _ { 2 }$ are selected via tournament selection with the highest fitness value among $\rho _ { c } | P _ { g } |$ number of chromosomes in M, where $\rho _ { c }$ specifies the fraction of new chromosomes generated through crossover. A random index L is sampled uniformly from [1, |S¿]. Since chromosomes encode fixed-cardinality client subsets rather than independent genes, naive crossover operators may yield infeasible or redundant selections. Moreover, high-quality client subsets in FL often exhibit common structural characteristics such as complementary data distributions or favorable resource availability, motivating a recombination-based operator. We therefore adopt a crossover operation that ensures client uniqueness without requiring post-hoc repair. Specifically, a prefix of length L is inherited from the first parent. The remaining positions are filled using the second parent, excluding duplicates. This process is symmetrically applied to generate two child chromosomes, $Y _ { 1 }$ and $Y _ { 2 } .$ , ensuring that each offspring represents a valid subset of m distinct clients.

• Mutation: For each child chromosome y, every client in y is subject to a mutation process with a low probability of $1 / | y |$ in lines 24 - 28. If a mutation occurs, the corresponding client in y is replaced with another random client from C that does not already appear in y. This simple mutation operator preserves subset cardinality and feasibility while introducing controlled diversity into the population. While crossover exploits existing highfitness client combinations, mutation enables exploration of previously unselected clients and mitigates premature convergence, which is particularly important under non-IID data distributions commonly encountered in FL.

The combined use of crossover and mutation allows the GA to adapt client selection dynamically across rounds, reflecting the evolving contribution of clients as local models and data distributions change. We deliberately employ simple, low-overhead GA operators rather than more complex adaptive or problem-specific variants. This design choice reflects the practical constraints of FL systems, where client selection must be performed efficiently at the server without introducing significant computational overhead or instability across rounds.

## 4.3.3. Population Update

After crossover and mutation, the population is updated in line 29 to $P _ { g + 1 }$ in order to include both parent and child chromosomes for the next GA generation. This ensures that high-fitness chromosomes are retained while introducing diversity into the population. The reproduction process is repeated for $G$ generations or until no significant improvement in the fitness of chromosomes is observed, after which the chromosome with the highest fitness value is selected as the final solution in line 31 and returned as $S _ { t } ^ { * }$

## 4.3.4. Client Selection and Training

The best chromosome obtained from the GA represents the subset $S _ { t } ^ { * }$ of clients for the current round t. The server selects these clients to participate in local training. In lines $7 \cdot 1 0 ,$ each selected client receives the current global model $w ^ { t } .$ , performs local training using its dataset $D _ { i } ,$ and sends back the updated model $w _ { i } ^ { t } .$ The central server aggregates these updates to produce the new global model:

$$
w ^ { t + 1 } \gets \sum _ { i = 1 } ^ { m } \frac { | D _ { i } | } { \sum _ { i = 1 } ^ { m } | D _ { i } | } w _ { i } ^ { t }\tag{8}
$$

To enable history-aware selection, the server maintains and updates the selection frequency $f _ { c _ { i } }$ for each client across communication rounds. After each round, the frequency of selected clients is incremented, and the updated values are used in subsequent rounds when evaluating the history-aware decay term in Eq. (7). The entire process is repeated for a predefined number of FL rounds or until the global objective $F ( w )$ satisfies a specified convergence criterion. By using GA, the client selection process becomes resource-aware, leading to enhanced global model performance.

## 4.3.5. Design of Resource Metrics

Prior works [26, 25] have explored strategies where clients with higher statistical utility are prioritized based on importance sampling, gradient norm, or weight differences between client local models and the global model. However, directly computing gradient norms or changes in the local model of clients in each round introduces significant computational overhead, making it impractical in resource-constrained environments. To address this issue, we leverage insights from [13], which demonstrate that choosing clients with higher local loss accelerates convergence. Consequently, we approximate statistical heterogeneity through a pragmatic metric that incorporates both the client's dataset size and its most recent training loss, thereby quantifying its statistical contribution with minimal computational overhead. Specifically, we define the statistical resource metric of client $c _ { i }$ as:

$$
\mathrm { R e s } _ { \mathrm { s t a t } } ( c _ { i } ) = | D _ { i } | \cdot \sum _ { \delta _ { j } \in D _ { i } } \hat { L } ( w , \delta _ { j } ) ,\tag{9}
$$

where $\hat { L } ( w , \delta _ { j } )$ represents the normalized loss over all local training samples of client $c _ { i }$ at its last participation round. This formulation enables the exploitation of clients with larger datasets and higher recent losses, thereby facilitating faster convergence. For clients that have not yet participated, $\hat { L } ( w , \delta _ { j } )$ is initialized to a large constant, encouraging early exploration of previously unseen clients. Once selected, this value is updated based on the observed local training loss and, together with the history-aware selection mechanism, promotes training stability and fairness.

For system heterogeneity, we follow a similar setup of [27, 28], recognizing that federated clients exhibit diverse computational capabilities, which introduce variations in local model update times. To formally characterize this heterogeneity, we define $\tau _ { i }$ as the expected time required for client $c _ { i }$ to complete a single local model update. This duration is predominantly influenced by the computation time associated with processing a fixed batch-size stochastic gradient of the local empirical loss function $L _ { i } ( w )$ [27]. Clients with larger $\tau _ { i }$ values correspond to lower computational efficiency, often functioning as stragglers that hinder synchronization and overall training progression. To systematically capture system heterogeneity, we define the system resource metric as follows:

$$
\mathrm { R e s } _ { \mathrm { s y s } } ( c _ { i } ) = \frac { 1 } { \tau _ { i } } ,\tag{10}
$$

This formulation ensures that clients with lower computational latency contribute higher system resource values, facilitating the selection of more efficient clients for training.

Both the statistical and system resource metrics proposed in this work are intentionally designed to be modular and extensible, facilitating their adaptation to more nuanced FL environments. For statistical heterogeneity, although we currently leverage dataset size and recent training loss due to their low overhead, the formulation permits integration of richer data characteristics such as sample diversity [29], label distribution imbalance [30], or noise levels[31]. On the system side, the current design captures computational latency via local update time, but can be augmented to account for communication-related overheads, including bandwidth variability [32], transmission latency [33], or energy consumption [34]. For instance, an extended formulation could incorporate both computation and communication time through a unified delay-aware system resource metric. Overall, this design maintains computational efficiency while providing a principled foundation for future integration of statistically and systemically richer client attributes, thus enhancing selection decisions under real-world FL heterogeneity.

## 5. Evaluation

This section outlines the experimental setup, including baselines, datasets, model architectures, and training configurations. We then evaluate FedCSGA through extensive experiments encompassing performance comparisons, partial client participation analysis, GA parameter study, ablation experiments, and communication and computation complexity assessment.

## 5.1. Experimental setup

## 5.1.1. Baselines

To demonstrate the effectiveness of FedCSGA, we compared it against a diverse set of baseline algorithms. These baselines include both traditional FL and client selection algorithms.

• FedAvg [19] The standard FL aggregation algorithm that performs a weighted averaging of local model updates from randomly selected clients.

• FedProx [15] An FL algorithm that allows each participating device to perform a variable amount of work by stabilizing local training with a proximal term $\mu .$ The parameter $\mu$ in FedProx was tuned from {0.1, 0.5, 1.0}.

• PowD [13] A client selection algorithm that prioritizes clients with the highest losses from a randomly sampled subset to train in FL. The parameter d in PowD where $( m \leq d \leq K )$ was tuned from {15, 20, 25, 35}.

• AFL [14] An FL algorithm that selects clients with a probability conditioned on the current model and the data on the client. The parameters $\alpha _ { 1 } , \alpha _ { 2 }$ , and $\alpha _ { 3 }$ in AFL were set to their default values of 0.75, 0.01, and 0.1, respectively.

• GreedyFed [17] A greedy client selection algorithm that uses Shapley value from cooperative game framework to measure client contribution. The parameters T and ε in GreedyFed were set to their default values of $5 0 \times m$ and 0.0001, respectively.

• GreedyFL A greedy client selection strategy that utilizes the same statistical and system resource metrics as FedCSGA and selects the top-m clients by ranking. This baseline isolates the effect of greedy ranking versus GA-based selection.

## 5.1.2. Datasets and model architecture

We evaluated the performance of FedCSGA on four benchmark datasets that are widely adopted in FL research.

• FMNIST [35] A dataset of grayscale images with dimensions $2 8 \times 2 8 \times 1$ , consisting of 10 classes, with 60,000 training samples and 10,000 test samples.

• CIFAR-10 [36] A dataset of color images with dimensions $3 2 \times 3 2 \times 3$ containing 10 classes, with 50,000 training samples and 10,000 test samples.

• CIFAR-100 [36] A dataset of color images with dimensions $3 2 \times 3 2 \times 3$ , featuring 100 classes, with 50,000 training samples and 10,000 test samples.

• Tiny-Imagenet [37] A dataset of color images with dimensions $6 4 \times 6 4 \times 3 .$ comprising 200 classes, with 100,000 training samples, 10,000 validation samples, and 10,000 test samples.

For FMNIST, CIFAR-10, and CIFAR-100 datasets, we use the same CNN architecture used in [38, 39]. For the more complex Tiny-ImageNet dataset, a ResNet-18 model [40] was used. For the FMNIST and Tiny-ImageNet datasets, each experiment was run for 300 communication rounds, whereas for CIFAR-10 and CIFAR-100, 200 rounds were used. Unless stated otherwise, a subset of 10 clients out of a total of 100 was selected to participate in each round. To ensure the robustness and reproducibility of the results, all experiments were repeated three times. Following the protocol of [41], the final performance was reported as the average over the last 10 communication rounds. To emulate the client-server architecture and heterogeneous client devices, we implemented the FedCSGA algorithm along with all other baselines using the Flower framework [42] and PyTorch [43]. We performed all of our experiments on an Intel Core i9- 10900 3.70-GHz processor with 192 GB of RAM, and 4× NVIDIA GeForce RTX 4090 GPUs.

## 5.1.3. Heterogeneity settings

To simulate realistic FL environments while maintaining conceptual simplicity, we model statistical and system heterogeneity with equal weighting by setting $\lambda _ { r } = 0 . 5$ and $\lambda _ { s } = 0 . 5$ . For statistical heterogeneity, the training datasets were partitioned among 100 clients using a Dirichlet distribution parameterized by α, where smaller values of α correspond to higher degrees of statistical heterogeneity (data non-IIDness) [44]. We consider three configurations: IID (uniform distribution), non-IID with $\alpha = 0 . 3$ , and non-IID with $\alpha = 0 . 6$ To emulate system heterogeneity in resource-constrained environments, we adopt settings similar to those used in [15, 17, 45], where clients are randomly assigned either (i) 1 local epoch, a batch size of 8, and a learning rate $\eta = 0 . 0 0 1$ , or (ii) 3 local epochs, a batch size of 32, and a learning rate $\eta = 0 . 0 1$ . In each communication round, clients were randomly assigned to one of these settings to simulate varying system capabilities. For the more complex Tiny-ImageNet dataset, the learning rates are adjusted to $\eta = 0 . 0 0 0 0 1$ and $\eta = 0 . 0 0 0 1$ for (i) and (ii), respectively.

## 5.1.4. Genetic algorithm parameters

To perform the client selection process, we tuned the hyperparameters of GA using grid search, the results of which are discussed in Section 5.4. During tuning, we monitored not only test accuracy but also the server-side runtime overhead for training each generation. Our hyperparameter selection accounted for both accuracy maximization and runtime minimization. A detailed analysis supporting these observations, including the communication and computation trade-offs across different configurations, is presented in Section 5.6.2. Through experimentation, we identified the best performing configuration with the total number of generations $G = 3 0 .$ population size $P = 3 0$ , parent chromosome fraction $\rho _ { p } = 0 . 5 $ , and child chromosome fraction $\rho _ { c } = 0 . 5$ . We incorporated the concept of elitism [46] and tournament selection [47] for parent and child chromosomes, which ensures the preservation of high-quality subsets while also maintaining diversity in the population. For the history-aware selection mechanism in FedCSGA, we found that moderate values of γ yield stable performance across datasets, while very large values excessively penalize frequently selected clients, therefore, $\gamma = 0 . 0 3$ was used in all reported results.

## 5.2. Performance comparison

Table 3 reports the mean and standard deviation of test accuracy for all the datasets across non-IID settings. For FMNIST, GreedyFL attains the highest mean accuracy at both $\alpha = 0 . 3$ and $\alpha = 0 . 6 $ , marginally outperforming FedCSGA (Figure 2, subplots (a) and (b)). Nevertheless, the performance gap remains within 0.5%, and as dataset complexity increases, FedCSGA consistently yields larger and more stable improvements across baselines, indicating stronger scalability under more challenging non-IID conditions. On CIFAR-10, FedCSGA exceeds the performance of AFL by 4.0% in the highly non-IID scenario $( \alpha = 0 . 3 )$ and outperforms PowD by

Table 3: Comparison of FedCSGA and baseline accuracies on four datasets under non-IID settings $( \alpha = \{ 0 . 3 , 0 . 6 \} )$ . Mean and standard deviation of test accuracy are reported over three random seeds after 300 rounds (FMNIST, Tiny-Imagenet) and 200 rounds (CIFAR-10, CIFAR-100).
<table><tr><td rowspan="2">Algorithm</td><td colspan="2">FMNIST</td><td colspan="2">CIFAR-10</td><td colspan="2">CIFAR-100</td><td colspan="2">Tiny-Imagenet</td></tr><tr><td> ${ \pmb \alpha } = { \bf 0 . 3 }$ </td><td> ${ \pmb { \alpha } } = { \bf 0 . 6 }$ </td><td> ${ \pmb \alpha } = { \bf 0 . 3 }$ </td><td> ${ \pmb { \alpha } } = { \bf 0 . 6 }$ </td><td> ${ \pmb \alpha } = { \bf 0 . 3 }$ </td><td> $\pmb { \alpha = 0 . 6 }$ </td><td> ${ \pmb \alpha } = { \bf 0 . 3 }$ </td><td> $\pmb { \alpha = 0 . 6 }$ </td></tr><tr><td>FedAvg</td><td> $7 9 . 5 \pm 2 . 3 $ </td><td> $8 2 . 0 \pm 1 . 8$ </td><td> $4 9 . 3 \pm 3 . 1 $ </td><td> $5 5 . 4 \pm 1 . 9$ </td><td> $2 6 . 2 \pm 0 . 7$ </td><td> $2 7 . 2 \pm 0 . 8$ </td><td> $2 2 . 6 \pm 0 . 8$ </td><td> $2 2 . 8 \pm 0 . 9$ </td></tr><tr><td>FedProx</td><td> $7 9 . 6 \pm 1 . 1$ </td><td> $8 1 . 8 \pm 1 . 4$ </td><td> $5 0 . 4 \pm 2 . 8 $ </td><td> $5 6 . 1 \pm 1 . 8$ </td><td> $2 6 . 3 \pm 0 . 6$ </td><td> $2 7 . 2 \pm 1 . 0$ </td><td> $2 2 . 5 \pm 0 . 6$ </td><td> $2 3 . 3 \pm 0 . 5$ </td></tr><tr><td>PowD</td><td> $8 0 . 9 \pm 1 . 7$ </td><td> $8 2 . 5 \pm 1 . 4$ </td><td> $5 1 . 3 \pm 4 . 3$ </td><td> $5 7 . 2 \pm 1 . 4$ </td><td> $2 5 . 6 \pm 0 . 8$ </td><td> $2 6 . 8 \pm 0 . 7$ </td><td> $2 2 . 7 \pm 0 . 5$ </td><td> $2 3 . 2 \pm 0 . 9$ </td></tr><tr><td>AFL</td><td> $7 8 . 6 \pm 2 . 6$ </td><td> $8 2 . 0 \pm 1 . 5$ </td><td> $5 1 . 6 \pm 2 . 9$ </td><td> $5 6 . 8 \pm 1 . 8$ </td><td> $2 6 . 4 \pm 0 . 7$ </td><td> $2 7 . 2 \pm 0 . 5$ </td><td> $2 2 . 6 \pm 0 . 4$ </td><td> $2 2 . 7 \pm 0 . 3$ </td></tr><tr><td>GreedyFed</td><td> $8 0 . 1 \pm 1 . 3$ </td><td> $8 3 . 3 \pm 0 . 5$ </td><td> $4 9 . 2 \pm 1 . 3$ </td><td> $5 2 . 3 \pm 1 . 1$ </td><td> $1 8 . 0 \pm 0 . 2$ </td><td> $1 8 . 1 \pm 0 . 7$ </td><td> $1 4 . 1 \pm 0 . 2$ </td><td> $1 4 . 1 \pm 0 . 9$ </td></tr><tr><td> $\mathbf { G r e e d y F L }$ </td><td> $8 2 . 2 \pm 2 . 2$ </td><td> ${ \bf 8 4 . 9 \pm 1 . 2 }$ </td><td> $5 4 . 1 \pm 2 . 3$ </td><td> $5 8 . 4 \pm 1 . 5$ </td><td> $2 6 . 9 \pm 0 . 4$ </td><td> $2 7 . 2 \pm 0 . 3$ </td><td> $2 3 . 7 \pm 0 . 6$ </td><td> $2 4 . 3 \pm 0 . 6$ </td></tr><tr><td>FedCSGA</td><td> $8 1 . 9 \pm 1 . 9$ </td><td> $8 4 . 4 \pm 1 . 3$ </td><td> ${ \pm 5 5 . 6 \pm 2 . 8 }$ </td><td> ${ \bf 6 0 . 3 \pm 2 . 1 }$ </td><td> ${ \bf 2 9 . 9 \pm 0 . 6 }$ </td><td> ${ \bf 3 0 . 5 \pm 0 . 4 }$ </td><td> ${ \bf 2 9 . 1 \pm 0 . 4 }$ </td><td> ${ \bf 2 9 . 8 \pm 0 . 2 }$ </td></tr></table>

![](images/32a3921083d66fe58552444f12a5b8d9c4de423a9902280a72b7df95a45552ce.jpg)  
Figure 2: Performance comparison against baselines across FMNIST, CIFAR-10, CIFAR-100, and Tiny-Imagenet datasets under different data distributions: (a) FMNIST α = 0.3, (b) FMNIST $\alpha = 0 . 6 ,$ (c) FMNIST IID, (d) CIFAR-10 $\alpha = 0 . 3 , ( \mathrm { e } )$ CIFAR $1 0 \alpha = 0 . 6 ,$ (f) CIFAR-10 IID, (g) CIFAR-100 α = 0.3, (h) CIFAR-100 α = 0.6, (i) CIFAR-100 IID, (j) Tiny-ImageNet $\alpha = 0 . 3 .$ (k) Tiny-ImageNet $\alpha = 0 . 6 ,$ and (1) Tiny-ImageNet IID.

3.1% under $\alpha = 0 . 6$ (Figure 2, subplots (d) and (e)). For the more complex CIFAR-100 dataset, where baseline performances are generally lower, FedCSGA achieves consistent improvements, surpassing AFL by 3.5% under $\alpha = 0 . 3$ and AFL by 3.3% under $\alpha = 0 . 6$ (Figure 2, subplots (g) and (h)). In case of Tiny-Imagenet, we see a significant improvement of 6.4% under $\alpha = 0 . 3$ and 6.5% under $\alpha = 0 . 6$ against PowD and FedProx, respectively (Figure 2, subplots (j) and (k)). Unlike GreedyFed, which suffers substantial performance drops on the more challenging CIFAR-100 and Tiny-Imagenet datasets, FedCSGA demonstrates faster convergence and maintains steady accuracy improvements with reduced variance across all settings. The performance degradation of GreedyFed may stem from its reliance on server-side validation data for Shapley

Table 4: Comparison of FedCSGA and baseline accuracies on four datasets under IID settings. Mean and standard deviation of test accuracy are reported over three random seeds after 300 rounds (FMNIST, Tiny-Imagenet) and 200 rounds (CIFAR-10, CIFAR-100).
<table><tr><td rowspan=1 colspan=1>Algorithm</td><td rowspan=1 colspan=1>FMNIST</td><td rowspan=1 colspan=1> $\mathbf { C I F A R - 1 0 }$ </td><td rowspan=1 colspan=1> $\mathbf { C I F A R - 1 0 0 }$ </td><td rowspan=1 colspan=1>|Tiny-Imagenet</td></tr><tr><td rowspan=1 colspan=1>FedAvg</td><td rowspan=1 colspan=1> $8 6 . 2 \pm 0 . 1 $ </td><td rowspan=1 colspan=1> $6 3 . 7 \pm 0 . 5$ </td><td rowspan=1 colspan=1> $2 7 . 7 \pm 0 . 5$ </td><td rowspan=1 colspan=1> $2 3 . 6 \pm 0 . 6$ </td></tr><tr><td rowspan=1 colspan=1>FedProx</td><td rowspan=1 colspan=1> $8 6 . 1 \pm 0 . 3$ </td><td rowspan=1 colspan=1> $6 3 . 0 \pm 0 . 4$ </td><td rowspan=1 colspan=1> $2 7 . 5 \pm 0 . 3$ </td><td rowspan=1 colspan=1> $2 3 . 0 \pm 0 . 9$ </td></tr><tr><td rowspan=1 colspan=1>PowD</td><td rowspan=1 colspan=1> $8 6 . 5 \pm 0 . 1 $ </td><td rowspan=1 colspan=1> $6 3 . 6 \pm 0 . 3$ </td><td rowspan=1 colspan=1> $2 7 . 7 \pm 0 . 5$ </td><td rowspan=1 colspan=1> $2 3 . 5 \pm 0 . 5$ </td></tr><tr><td rowspan=2 colspan=1>AFLGreedyFed</td><td rowspan=1 colspan=1> $8 6 . 4 \pm 0 . 2$ </td><td rowspan=1 colspan=1> $6 3 . 5 \pm 0 . 2$ </td><td rowspan=1 colspan=1> $2 7 . 6 \pm 0 . 3$ </td><td rowspan=2 colspan=1> $2 2 . 9 \pm 0 . 4$  $1 7 . 4 \pm 0 . 9$ </td></tr><tr><td rowspan=1 colspan=1> $8 5 . 5 \pm 0 . 3$ </td><td rowspan=1 colspan=1> $5 7 . 3 \pm 0 . 7$ </td><td rowspan=1 colspan=1> $1 9 . 3 \pm 0 . 2$ </td></tr><tr><td rowspan=1 colspan=1>GreedyFL</td><td rowspan=1 colspan=1> $8 8 . 2 \pm 0 . 1 $ </td><td rowspan=1 colspan=1> $6 6 . 2 \pm 0 . 8$ </td><td rowspan=1 colspan=1> $2 8 . 5 \pm 0 . 2 $ </td><td rowspan=1 colspan=1> $2 4 . 5 \pm 0 . 5 3$ </td></tr><tr><td rowspan=1 colspan=1>FedCSGA</td><td rowspan=1 colspan=1> ${ \bf 8 8 . 3 \pm 0 . 2 }$ </td><td rowspan=1 colspan=1> ${ \bf 6 7 . 7 \pm 0 . 6 }$ </td><td rowspan=1 colspan=1> ${ \bf 3 0 . 7 \pm 0 . 2 }$ </td><td rowspan=1 colspan=1> ${ \bf 3 0 . 6 \pm 0 . 2 }$ </td></tr></table>

Value estimation, which becomes unreliable when clients hold unique or non-overlapping data classes in complex datasets and highly non-IID settings.

As shown by Table 4, FedCSGA also exhibits consistent performance gains in the IID settings. For instance, it outperforms FedAvg by a significant margin of 7.0% on Tiny-Imagenet (Figure 2, subplot (1)). Similarly, we see an improvement of 4.0% and 3.0% against FedAvg on CIFAR-10 and CIFAR-100 (Figure 2, subplots (f) and (i)), respectively, and 1.8% against PowD on FMNIST (Figure 2, subplot (c)). Compared to other baselines, these results highlight the joint consideration of statistical and system heterogeneity by FedCSGA, which ensures performance improvement even in the IID scenarios where the data across all clients follows a similar data distribution. This balanced approach to handling diverse data distributions and resource considerations is critical in federated learning, ensuring effective generalization across datasets with varying complexity and heterogeneity. Overall, these results underscore the practical applicability of FedCSGA in diverse federated learning environments.

## 5.3. Impact of partial client participation

To evaluate the robustness and adaptability of our proposed FedCSGA algorithm, we analyze its performance under varying client participation ratios (CPR) using the CIFAR-10 dataset with $\alpha = 0 . 3$ . Table 5 and Figure 3 illustrate the impact of different CPR values on FedCSGA and the baseline algorithms. As the number of participating clients per training round increases, all methods demonstrate improved performance, reflecting the general benefit of increased data diversity during training. Notably, FedCSGA consistently outperforms all baseline algorithms across varying CPR levels, underscoring its scalability and adaptability in complex FL environments. For instance, at CPR 0.10, FedCSGA achieves a significant improvement of 1.5% over the next best-performing baseline, GreedyFL. Overall, these findings reaffirm the effectiveness of FedCSGA in addressing the inherent challenges of partial client participation in FL, making it a practical choice for diverse and challenging real-world deployments.

Table 5: Accuracy comparison of FedCSGA against baselines across four client participation ratios (CPR) on CIFAR-10 (α = 0.3).
<table><tr><td rowspan="2">Algorithm</td><td colspan="4">Client Participation Ratio (CPR)</td></tr><tr><td>0.10</td><td>0.15</td><td>0.20</td><td>0.30</td></tr><tr><td>FedAvg FedProx PowD AFL</td><td> $4 9 . 3 \pm 3 . 1 $   $5 0 . 4 \pm 2 . 8 $   $5 1 . 3 \pm 4 . 3$   $5 1 . 6 \pm 2 . 9$ </td><td> $5 2 . 7 \pm 2 . 7$   $5 4 . 1 \pm 1 . 7$   $5 2 . 8 \pm 2 . 7$   $5 4 . 5 \pm 2 . 2$ </td><td> $5 4 . 3 \pm 1 . 7$   $5 4 . 2 \pm 1 . 8$   $5 5 . 0 \pm 2 . 2$   $5 4 . 9 \pm 2 . 0$ </td><td> $5 6 . 0 \pm 2 . 0$   $5 5 . 6 \pm 1 . 3$   $5 6 . 2 \pm 1 . 7$   $5 5 . 8 \pm 1 . 8$ </td></tr><tr><td>GreedyFed GreedyFL</td><td> $4 9 . 2 \pm 1 . 3$   $5 4 . 1 \pm 2 . 3$ </td><td> $5 1 . 5 \pm 0 . 6$   $5 6 . 5 \pm 1 . 8$ </td><td> $5 4 . 0 \pm 1 . 0$   $5 8 . 2 \pm 1 . 4 $ </td><td> $5 3 . 8 \pm 2 . 0$ </td></tr><tr><td>FedCSGA</td><td> ${ \pm 5 5 . 6 \pm 2 . 8 }$ </td><td> ${ \bar { 5 } } 7 . 0 \pm 2 . 7$ </td><td> ${ \bf 5 8 . 6 \pm 1 . 8 }$ </td><td> $6 0 . 3 \pm 1 . 1$   ${ \bf 6 0 . 7 \pm 1 . 7 }$ </td></tr></table>

![](images/78e15548011e2e8a79900b87314b2bb0ea9c92dec33ab08615bb69405329ce03.jpg)  
(a)

![](images/c85a174afd9088887d704b529b508c56a0fd6637c64d68c418ec1e5f486abd13.jpg)  
(b)

![](images/0ed48103a492238cb3f9bf463f7b74a4e1e4f4341c97443232c152286dcbc132.jpg)  
(c)  
Figure 3: Performance comparison of FedCSGA and baseline methods under varying client participation ratios $\mathrm { ( C P R = }$ 0.15, 0.20, 0.30) on CIFAR-10 (α = 0.3).

## 5.4. Sensitivity Analysis of GA Hyper-parameters

This subsection investigates the sensitivity of FedCSGA to key GA hyper-parameters. Specifically, we analyze how the population size P and the number of generations G influence convergence behavior and model accuracy, as well as how the fractions of parent $( \rho _ { p } )$ and child $( \rho _ { c } )$ chromosomes affect the balance between solution exploitation and exploration. The results provide insights into the trade-offs among diversity, convergence speed, and performance, guiding the selection of effective GA configurations.

## 5.4.1. Impact of hyper-parameters P and G

Figure 4 presents the performance of FedCSGA under varying values of population size P and generation rounds G. The accuracy values are reported across $G \in \{ 5 , 1 0 , 2 0 , 3 0 \}$ and $P \in \{ 5 , 1 0 , 2 0 , 3 0 \}$ . It can be observed that FedCSGA demonstrates suboptimal performance for smaller values of $G ,$ as insufficient evolutionary rounds limit the algorithm's ability to evolve chromosomes toward high-fitness solutions effectively. Similarly, for smaller values of $P ,$ the reduced diversity within the population restricts the GA's ability to explore a broad solution space, resulting in lower accuracy. For example, when $P = 5$ and $G = 5 .$ , the accuracy is the lowest at 37.4%, highlighting the compounded limitations of a small population and minimal evolutionary rounds. Conversely, higher values of P and G result in noticeable performance gains. For instance, when $P = 3 0$ and $G = 3 0$ , the accuracy peaks at 48.0%, indicating the advantages of increased diversity and extended evolution. Notably, FedCSGA also achieves fair performance within moderate ranges of these hyper-parameters, such as $P = 2 0$ and $G = 1 0 ,$ with an accuracy of 43.9%. These findings emphasize the importance of balancing $P$ and G to achieve effective performance. Since the performance improvement exhibits a clear correlation with increased computational resources, we recommend utilizing values around $P \ \geq \ 2 0$ and $G \geq 1 0$ as robust default configurations. This guideline allows practitioners to achieve reliable performance without extensive hyper-parameter tuning, thereby effectively mitigating sensitivity concerns in practical deployments.

![](images/c157aaa525df7e223ab7c668cd1b216bb47c093689cb81e5f31aaf7960fdb7c2.jpg)  
Figure 4: Impact of population size P and generations rounds G.

## 5.4.2. Impact of hyper-parameters $\rho _ { p }$ and $\rho _ { c }$

Figure 5 reports the test accuracy of FedCSGA across varying fractions of parent chromosomes $( \rho _ { p } )$ and children chromosomes $( \rho _ { c } )$ . These hyper-parameters define the trade-off between retaining prior solutions and generating new ones. We enforce $\rho _ { p } + \rho _ { c } = 1$ to ensure that the total number of chromosomes in the population remains constant across generations. The result shows that excessively high values of both $\rho _ { p }$ and $\rho _ { c }$ leads to premature convergence. For high $\rho _ { p }$ values, this might occur due to the reduced diversity in the population, limiting the algorithm's ability to explore alternative client subsets in the search space. Conversely, very high values of $\rho _ { c }$ result in the loss of valuable high-performing solutions from earlier generations, yielding suboptimal accuracy. Robust performance is achieved when $\rho _ { p }$ and $\rho _ { c }$ are equal, allowing for a balanced approach between retaining strong parent chromosomes and introducing diverse child chromosomes. For instance, when $\rho _ { p } = 0 . 5$ and $\rho _ { c } = 0 . 5$ , the test accuracy peaks at $4 7 . 0 \%$ , demonstrating the effectiveness of this balanced approach. These findings highlight the importance of tuning $\rho _ { p }$ and $\rho _ { c }$ for efficient client selection. By carefully balancing parent retention and children generation, FedCSGA can effectively leverage high-performing clients while maintaining population diversity to explore promising client subsets. Given that the performance sensitivity follows a predictable trend favoring balance, we recommend setting $\rho _ { p } \approx \rho _ { c } \approx 0 . 5$ as a standard default configuration. Adopting this balanced setting ensures stable performance and prevents degradation caused by extreme parameter choices.

![](images/9778c10e609959ee105c5c8e58d94438017ec5199db45862b35ba1160c56c4b8.jpg)  
Figure 5: Impact of parent and child chromosome fractions $( \rho _ { p } , \rho _ { c } )$

## 5.5. Ablation study on fitness function

We conduct an ablation study to analyze the contribution of resource-aware metrics and the proposed history-aware weighting factor in the FedCSGA fitness formulation. Table 6 reports the test accuracy under three configurations: (i) using only the resource-aware metric Res(c), which aggregates normalized statistical and system indicators, (ii) using only the history-aware weighting factor W(c), where clients are selected by sorting them in ascending order of their past selection counts (i.e., prioritizing the least-selected clients), and (iii) the full FedCSGA configuration that jointly incorporates both components. When only Res(c) is used, the model achieves a test accuracy of 59.0%, indicating that prioritizing clients based on instantaneous statistical and system utility is effective for improving convergence and final performance. In contrast, relying solely on the history-aware factor yields significantly lower accuracy (55.2%), as this configuration promotes participation diversity without explicitly considering data quality or system efficiency. The best performance is obtained when both Res(c) and $W ( c )$ are combined, achieving 60.3% accuracy. This improvement demonstrates that historical participation information complements resource-aware selection by discouraging repeated selection of similar client subsets across rounds while preserving utility-driven selection. These results validate the design of the proposed fitness function, showing that neither resource awareness nor historical weighting alone is sufficient to achieve efficient performance. Instead, their joint use enables FedCSGA to balance short-term training utility with long-term participation dynamics, resulting in more stable and effective client selection under heterogeneous federated learning settings.

Table 6: Ablation study of FedCSGA on CIFAR-10 $( \alpha = 0 . 6 )$ , showing the impact of resource-aware client selection Res(c) and history-aware weighting W(c). Results report test accuracy (%) under different component configurations.
<table><tr><td></td><td>Resource Metrics Res(c) | History-Aware Factor W(c) | Accuracy (%)</td><td></td></tr><tr><td>√</td><td></td><td> $5 9 . 0 \pm 0 . 6$ </td></tr><tr><td rowspan="2">√</td><td>&gt;&gt;</td><td> $5 5 . 2 \pm 0 . 7$ </td></tr><tr><td></td><td> ${ \bf 6 0 . 3 \pm 2 . 1 }$ </td></tr></table>

## 5.6. Computational and communication complexity

This subsection analyzes the computational and communication costs associated with FedC-SGA. We first derive the time complexity of the GA-based client selection process, highlighting how the population size, number of generations, and selected clients affect server-side computation. We then empirically evaluate the communication efficiency and per-round computation time of FedCSGA in comparison with baseline methods, illustrating the trade-off between additional server computation and reduced communication rounds.

## 5.6.1. Time complexity analysis

The time complexity of FedCSGA is primarily governed by the GA-based client selection mechanism. In each communication round, the algorithm first initializes a population of P random client subsets (chromosomes), each containing m clients. Random sampling of m clients from a total of K can be implemented in expected O(m) time using efficient sampling procedures. Thus, initializing the population of |P| chromosomes and evaluating their fitness together incur an overall cost of $O ( | P | m )$ . Fitness evaluation of each chromosome is performed by summing the resource metrics of its m clients, which takes $O ( m )$ time per chromosome. During each of the G generations, the GA performs parent selection, crossover, and mutation. Parent selection identifies the top $\rho _ { p } | P |$ chromosomes based on the fitness value, which can be performed in $O ( | P | l o g | P | )$ time using sorting, or in expected $O ( | P | )$ time using efficient selection algorithms such as QuickSelect [48]. Crossover operations produce approximately $\rho _ { c } | P |$ child chromosomes of size m by recombining segments from two parent chromosomes. Each crossover involves copying $O ( m )$ elements, resulting in $O ( \rho _ { c } | P | m )$ time per generation. Mutation modifies each of the $\rho _ { c } | P |$ children by replacing clients with a small probability $( 1 / m )$ , again requiring $O ( \rho _ { c } | P | m )$ time. The cost of updating the population, which includes merging parents and children chromosomes, is negligible at O([P|). As such, the per-generation cost is dominated by crossover and mutation, both scaling as $O ( | P | m )$ . Aggregating over G generations, the total complexity of the GA process becomes $O ( G | P | m )$ . The one-time initialization cost does not affect the overall asymptotic complexity. Thus, the total time complexity of the FedCSGA client selection algorithm is $O ( G | P | m )$ . Given that the selected subset size m is typically much smaller than the total number of clients K $( m \ll K )$ , the complexity is effectively linear in m. If m were on the order of K, the complexity would scale as $O ( G | P | K )$

## 5.6.2. Communication complexity analysis

We also evaluate the communication and computation complexity of FedCSGA by comparing it with other baselines using two metrics: $R _ { 5 0 } $ , the number of communication rounds required to achieve a test accuracy of 50%, and $T _ { \mathrm { c o m p } }$ , the average computation time (in seconds) per round. The computation time includes the server's client selection time and the time taken by selected clients to perform local updates. As shown in Table 7, FedCSGA achieves a significant reduction in the number of communication rounds while maintaining a comparable computation time per round to simpler baselines such as FedAvg and FedProx. This demonstrates that, under moderate GA settings, the evolutionary search achieves efficient client selection without noticeable computational burden. However, when client participation ratio increases, although FedCSGA continues to reduce the communication rounds required to reach the target accuracy, the computation time per round increases slightly due to the increased number of fitness evaluations performed at the server during client selection. Despite this, the additional server-side computation leads to a substantially faster overall convergence, which effectively lowers the total communication cost, which is the dominant factor in FL efficiency. In many real-world FL deployments, particularly in cross-device and bandwidth-limited scenarios, network communication is orders of magnitude slower and more expensive than server computation. Therefore, investing a few additional seconds of server processing to make more informed client selections yields a net reduction in total training time and communication expenditure. We acknowledge, however, that the assumption of an adequately resourced server may not always hold in extremely resource-constrained or large-scale cross-device environments. The complexity of FedCSGA is O(G |P| m), scaling linearly with the number of generations (G), population size (|Pl), and selected clients (m). While feasible for most cross-silo and many cross-device settings, this could become a bottleneck in scenarios with limited server capacity. Future extensions could mitigate this by parallelizing fitness evaluations across multiple cores or nodes, or by employing adaptive GA parameters, such as gradually reducing P and G once the global model stabilizes, to maintain efficiency over the training lifecycle.

Table 7: Comparison of $R _ { 5 0 } ,$ and $T _ { \mathrm { c o m p } }$ (sec) for different client participation ratios with CIFAR-10 $( \alpha = 0 . 3 )$
<table><tr><td rowspan="2">Algorithm</td><td colspan="5">Client Participation Ratio (CPR)</td></tr><tr><td colspan="2">0.10</td><td colspan="2">0.20</td><td colspan="2">0.30</td></tr><tr><td rowspan="2">FedAvg</td><td> ${ R } _ { 5 0 }$ </td><td> $T _ { \mathrm { { c o m p } } }$ </td><td> $\pmb { R } _ { 5 0 }$ </td><td> $T _ { \mathrm { { c o m p } } }$ </td><td> $\pmb { R } _ { 5 0 }$ </td><td> $T _ { \mathrm { { c o m p } } }$ </td></tr><tr><td>113</td><td>54</td><td>95</td><td>76</td><td>87</td><td>79</td></tr><tr><td>FedProx</td><td>112</td><td>54</td><td>99</td><td>55</td><td>92</td><td>57</td></tr><tr><td>PowD</td><td>100</td><td>58</td><td>89</td><td>93</td><td>84</td><td>97</td></tr><tr><td>AFL</td><td>107</td><td>54</td><td>94</td><td>55</td><td>81</td><td>56</td></tr><tr><td>GreedyFed</td><td>165</td><td>282</td><td>115</td><td>735</td><td>118</td><td>1573</td></tr><tr><td>GreedyFL</td><td>68</td><td>54</td><td>59</td><td>55</td><td>63</td><td>56</td></tr><tr><td>FedCSGA</td><td>64</td><td>64</td><td>65</td><td>72</td><td>57</td><td>78</td></tr></table>

## 6. Discussion and Limitations

This section discusses the practical implications and design trade-offs of FedCSGA, focusing on robustness, fairness, system assumptions, and scalability.

First, FedCSGA relies on lightweight client-reported statistical $( \mathrm { R e s } _ { \mathrm { s t a t } } )$ and system $( \mathrm { R e s } _ { \mathrm { s y s } } )$ metrics. While such metrics may be affected by noise, reporting delays, or staleness for infrequently selected clients, the proposed subset-level history-aware decay alleviates the impact of outdated or imprecise information by discouraging repeated selection of similar client combinations. As a result, the method naturally promotes exploration and stabilizes selection behavior even under moderate metric uncertainty.

Regarding fairness and selection bias, FedCSGA does not aim to enforce strict long-term fairness guarantees. Instead, it encourages soft participation balance by penalizing historically dominant client subsets rather than individual clients. This design avoids reducing client selection to a simple per-client frequency-based heuristic while still mitigating persistent bias toward early high-loss or frequently selected clients. Although prioritizing high-loss clients can accelerate early convergence, the decay mechanism effectively limits long-term bias reinforcement.

From a systems perspective, FedCSGA assumes that clients transmit small resource summaries to the server at each round. While this introduces communication overhead proportional to the number of clients, the transmitted metadata is lightweight and remains negligible compared to the cost of exchanging model parameters. Furthermore, privacy-preserving mechanisms such as Secure Aggregation [49] or Differential Privacy [50] are complementary to our approach and can be seamlessly integrated to protect sensitive resource metadata, if stronger privacy guarantees are required. This design aligns well with cross-silo and medium-scale cross-device FL deployments, which are the primary targets of this work. Extending the framework to extremely large-scale settings, where K reaches thousands or millions of clients, would require additional mechanisms such as metric caching, partial reporting, or hierarchical pre-selection.

Finally, although the server-side genetic algorithm introduces additional computation, its cost depends on the population size and subset cardinality rather than the total client pool. Under moderate configurations, this overhead remains manageable and is justified by the improved communication efficiency achieved through faster convergence. Simple permutation-preserving crossover and single-client mutation operators are intentionally adopted to ensure feasibility and stability, even though more sophisticated operators may further enhance efficiency.

## 7. Conclusion

This paper presented FedCSGA, a resource-aware client selection framework for federated learning that explicitly addresses statistical and system heterogeneity while accounting for longterm participation dynamics. FedCSGA formulates client selection as a subset-level optimization problem and employs a genetic algorithm to jointly consider resource metrics and historical participation effects, avoiding reduction to per-client greedy ranking. Extensive experimental results across heterogeneous settings demonstrate improved test performance and empirically stable convergence under moderate computational overhead. The effectiveness of FedCSGA stems from its ability to adapt client participation over training rounds by discouraging repeated selection of similar client subsets, thereby balancing utility-driven selection with participation diversity. While the current design prioritizes robustness, interpretability, and feasibility, several extensions remain open. Future work will investigate communication-efficient metric reporting strategies for large-scale cross-device deployments, privacy-preserving resource aggregation mechanisms, and more specialized genetic operators to further improve efficiency and scalability under extreme heterogeneity.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgments

This work was supported by the Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korean government (MSIT) (RS-2025-02217801, Development of core technologies for integrated management of large-scale on-device AI objects and networks).

## References

[1] M. P. Uddin, Y. Xiang, M. Hasan, J. Bai, Y. Zhao, L. Gao, A systematic literature review of robust federated learning: Issues, solutions, and future research directions, ACM Computing Surveys 57 (10) (2025) 1–62.

[2] K. Houidi, M. Said, A. Hakiri, N. Mellouli, H. K. B. Ayed, Evaluating federated learning in iot: Advancing scalability, privacy, and real-time intelligence, in: 2025 28th International Symposium on Real-Time Distributed Computing (ISORC), IEEE, 2025, pp. 265–270.

[3] K. Houidi, M. Said, A. Hakiri, N. Mellouli-Nauwynck, H. K. B. Ayed, Federated learning models for real-time iot: A survey, in: 2024 IEEE 27th International Symposium on Real-Time Distributed Computing (ISORC), IEEE, 2024, pp. 1–6.

[4] N. Benarba, S. Bouchenak, Bias in federated learning: A comprehensive survey, ACM Computing Surveys (2025).

[5] R. Selo, M. Kundroo, T. Kim, Fedtvd: balancing data quality and quantity for robust federated learning, Future Generation Computer Systems 176 (2026) 108177.

[6] D. Solans, M. Heikkila, A. Vitaletti, N. Kourtellis, A. Anagnostopoulos, I. Chatzigiannakis, et al., Non-iid data in federated learning: A systematic review with taxonomy, metrics, methods, frameworks and future directions, arXiv preprint arXiv:2411.12377 (2024).

[7] X. Yu, Z. Gao, C. Zhao, Y. Qiao, Z. Chai, Z. Mo, Y. Yang, Duds: Diversity-aware unbiased device selection for federated learning on non-iid and unbalanced data, Journal of Systems Architecture 156 (2024) 103280.

[8] K. Yang, B. Hu, M. Zhao, Coordinating computational capacity for adaptive federated learning in heterogeneous edge computing systems, IEEE Transactions on Parallel and Distributed Systems (2025).

[9] K.-S. Wong, D.-M. Nguyen, K. Le, L. Ho, C. Do, D. Le-Phuoc, An empirical study of federated learning on iot-edge devices: Resource allocation and heterogeneity, IEEE Transactions on Neural Networks and Learning Systems (2025).

[10] Y. Kang, Y. Xie, L. Wu, P. Zhou, Y. Du, Efficient dropout-resilient and verifiable federated learning scheme for aiot healthcare system, Journal of Systems Architecture 168 (2025) 103575.

[11] S. Mayhoub, T. M. Shami, A review of client selection methods in federated learning, Archives of Computational Methods in Engineering 31 (2) (2024) 1129–1152.

[12] X. Zhao, G. Li, Y. Yao, B. Cui, Balancing privacy and fairness: Client selection in differential privacy-based federated learning, Journal of Systems Architecture 168 (2025) 103576.

[13] Y. J. Cho, J. Wang, G. Joshi, Client selection in federated learning: Convergence analysis and power-of-choice selection strategies, arXiv preprint arXiv:2010.01243 (2020).

[14] J. Goetz, K. Malik, D. Bui, S. Moon, H. Liu, A. Kumar, Active federated learning, arXiv preprint arXiv:1909.12641 (2019).

[15] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith, Federated optimization in heterogeneous networks, in: I. Dhillon, D. Papailiopoulos, V. Sze (Eds.), Proceedings of Machine Learning and Systems, Vol. 2, 2020, pp. 429–450.

[16] P. Zhang, Y. Liu, Y. Zhou, X. Wei, M. Chen, When foresight pruning meets zeroth-order optimization: Efficient federated learning for low-memory devices, Journal of Systems Architecture 168 (2025) 103526.

[17] P. Singhal, S. R. Pandey, P. Popovski, Greedy shapley client selection for communicationefficient federated learning, IEEE Networking Letters (2024).

[18] S. Ahmed, M. S. Kaiser, S. Chaki, A. S. Ali, Adaptive federated learning with reinforcement learning-based client selection for heterogeneous environments, IEEE Access (2025).

[19] B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas, Communication-efficient learning of deep networks from decentralized data, in: Artificial intelligence and statistics, PMLR, 2017, pp. 1273–1282.

[20] T.-D. Cao, T. Truong-Huu, H. Tran, K. Tran, A federated deep learning framework for privacy preservation and communication efficiency, Journal of Systems Architecture 124 (2022) 102413.

[21] J. Choi, S. Oh, Lightweight multi-layered de-identification architecture: Secure client selection in federated learning, Journal of Systems Architecture 168 (2025) 103569.

[22] T. Nishio, R. Yonetani, Client selection for federated learning with heterogeneous resources in mobile edge, in: ICC 2019-2019 IEEE international conference on communications (ICC), IEEE, 2019, pp. 1–7.

[23] N. Ali, G. Aloi, R. Gravina, C. Savaglio, A. H. Sodhro, G. Fortino, Fedheur: Multi-heuristic client selection for task offloading in federated learning, in: 2025 21st International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT), IEEE, 2025, pp. 816–821.

[24] J. H. Holland, Genetic algorithms, Scientific american 267 (1) (1992) 66–73.

[25] R. Balakrishnan, T. Li, T. Zhou, N. Himayat, V. Smith, J. Bilmes, Diverse client selection for federated learning via submodular maximization, in: International Conference on Learning Representations, 2022.

[26] D. Yang, B. Hu, Y. Gao, A.-L. Jin, A. Liu, K. L. Yeung, Y. You, Heaps: Heterogeneityaware participant selection for efficient federated learning, Journal of Parallel and Distributed Computing (2025) 105168.

[27] A. Reisizadeh, I. Tziotis, H. Hassani, A. Mokhtari, R. Pedarsani, Straggler-resilient federated learning: Leveraging the interplay between statistical accuracy and system heterogeneity, IEEE Journal on Selected Areas in Information Theory 3 (2) (2022) 197–205.

[28] Q. Fei, N. Rodríguez-Barroso, M. V. Luzón, Z. Zhang, F. Herrera, Addressing data quality decompensation in federated learning via dynamic client selection, Future Generation Computer Systems (2025) 108138.

[29] A. Li, L. Zhang, J. Tan, Y. Qin, J. Wang, X.-Y. Li, Sample-level data selection for federated learning, in: IEEE INFOCOM 2021-IEEE Conference on Computer Communications, IEEE, 2021, pp. 1–10.

[30] M. Duan, D. Liu, X. Chen, Y. Tan, J. Ren, L. Qiao, L. Liang, Astraea: Self-balancing federated learning for improving classification accuracy of mobile deep learning applications, in: 2019 IEEE 37th international conference on computer design (ICCD), IEEE, 2019, pp. 246–254.

[31] Y. Chen, X. Yang, X. Qin, H. Yu, P. Chan, Z. Shen, Dealing with label quality disparity in federated learning, in: Federated Learning: Privacy and Incentive, Springer, 2020, pp. 108-121.

[32] J. Xu, H. Wang, Client selection and bandwidth allocation in wireless federated learning networks: A long-term perspective, IEEE Transactions on Wireless Communications 20 (2) (2020) 1188–1200.

[33] M. Al-Quraan, A. Khan, A. Centeno, A. Zoha, M. A. Imran, L. Mohjazi, Fedratrees: A novel computation-communication efficient federated learning framework investigated in smart grids, Engineering Applications of Artificial Intelligence 124 (2023) 106654.

[34] Z. Yang, M. Chen, W. Saad, C. S. Hong, M. Shikh-Bahaei, Energy efficient federated learning over wireless communication networks, IEEE Transactions on Wireless Communications 20 (3) (2020) 1935–1949.

[35] H. Xiao, K. Rasul, R. Vollgraf, Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, arXiv preprint arXiv:1708.07747 (2017).

[36] A. Krizhevsky, G. Hinton, Learning multiple layers of features from tiny images, Tech. rep., University of Toronto (2009).

[37] Y. Le, X. Yang, Tiny imagenet visual recognition challenge, CS 231N 7 (7) (2015) 3.

[38] P. Mai, R. Yan, Y. Pang, Rflpa: A robust federated learning framework against poisoning attacks with secure aggregation, Advances in Neural Information Processing Systems 37 (2024)104329–104356.

[39] J. Shao, F. Wu, J. Zhang, Selective knowledge sharing for privacy-preserving federated distillation without a good teacher, Nature Communications 15 (1) (2024) 349.

[40] K. He, X. Zhang, S. Ren, J. Sun, Deep residual learning for image recognition, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[41] M. A. Masud, M. A. Jahin, M. Hasan, Stabilizing federated learning under extreme heterogeneity with hetero-select, arXiv preprint arXiv:2508.06692 (2025).

[42] M. Kundroo, G. Haider, N. Khoa, A. W. Mamond, T. Kim, Fedeasy : Federated learning with ease, SoftwareX 31 (2025) 102276.

[43] J. Ansel, E. Yang, H. He, N. Gimelshein, A. Jain, M. Voznesensky, B. Bao, P. Bell, D. Berard, E. Burovski, et al., Pytorch 2: Faster machine learning through dynamic python bytecode transformation and graph compilation, in: Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, 2024, pp. 929–947.

[44] R. Selo, M. Kundroo, T. Kim, Impact of network topologies on decentralized federated 1earning, Cluster Computing 28 (15) (Oct 2025). doi:10.1007/s10586-025-05683-5.

[45] W. Zhao, X. Qiu, J. Fernandez-Marques, P. P. de Gusmão, N. D. Lane, Protea: Client profiling within federated systems using flower, in: Proceedings of the 1st ACM Workshop on Data Privacy and Federated Learning Technologies for Mobile Edge Network, 2022, pp. 1-6.

[46] C. W. Ahn, R. S. Ramakrishna, Elitism-based compact genetic algorithms, IEEE Transactions on Evolutionary Computation 7 (4) (2003) 367–385.

[47] B. L. Miller, D. E. Goldberg, et al., Genetic algorithms, tournament selection, and the effects of noise, Complex systems 9 (3) (1995) 193–212.

[49] R. Wang, L. Xiong, J. Geng, C. Xie, R. Li, An effective and verifiable secure aggregation scheme with privacy-preserving for federated learning, Journal of Systems Architecture 161 (2025) 103364.

[48] C. A. R. Hoare, Algorithm 65: find, Commun. ACM 4 (7) (1961) 321–322.

[50] F. Shan, Y. Liu, L. Fan, Y. Mao, Z. Chen, S. Li, A survey of optimization algorithms for differential privacy in federated learning, Journal of Systems Architecture 168 (2025) 103582.