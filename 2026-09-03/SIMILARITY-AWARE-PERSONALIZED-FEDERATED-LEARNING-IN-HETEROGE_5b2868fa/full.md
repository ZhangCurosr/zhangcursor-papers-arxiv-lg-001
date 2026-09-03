# SIMILARITY-AWARE PERSONALIZED FEDERATED LEARNING IN HETEROGENEOUS ENVIRONMENTS

Arun Kumar A V<sup>∗</sup>, Sunil Gupta, Dang Ngyuen, Bao Duong, Dat Phan Trong Applied Artificial Intelligence Initiative (A<sup>2</sup>I<sup>2</sup>) Deakin University, Australia

## ABSTRACT

Federated Learning (FL) allows decentralized clients to train models collaboratively while preserving data privacy. However, distribution mismatch across clients often leads to poor global generalization and degraded local client-level performance. In such scenarios, some of the clients with their local models trained solely on local data may perform better than the globally learnt model, thus nullifying the benefits of collaborative federated learning. To address this, we propose SAPE-FL (Similarity-Aware Personalized Federated Learning), a novel personalization framework that anchors each client’s model to both the global model and a similarity-weighted peer averaged model. By incorporating dynamic, client-specific regularization based on both model similarity and output similarity, SAPE-FL adaptively balances global knowledge transfer and peer collaboration while filtering out dissimilar clients. This dual anchoring mitigates negative transfer and enhances robustness in heterogeneous settings. We theoretically analyze our algorithm establishing its convergence guarantees and empirically show that SAPE-FL outperforms state-of-the-art methods under high statistical heterogeneity and low client data regimes.

Keywords Machine Learning · Federated Learning · Similarity-aware FL · Heterogenous Environments

## 1 Introduction

With the rapid proliferation and growing computational power of edge devices such as smartphones, wearable sensors, and IoT devices, there has been a shift towards training machine learning models directly on decentralized devices rather than centralizing data in one location. Federated Learning (FL) McMahan et al. [2017] enables multiple clients to collaboratively train a global model without sharing raw data, thereby preserving privacy. The decentralized and privacy-preserving nature of FL is appealing across various domains including healthcare Dayan et al. [2021], finance Long et al. [2020] and edge computing Abreha et al. [2022]. Compared to traditional distributed learning approaches (Alternating Direction Method of Multipliers (ADMM) in Boyd et al. [2011]), FL methods have shown superior scalability and communication efficiency.

While traditional FL methodologies have demonstrated significant promise, real-world deployments often face substantial challenges due to heterogeneity Li et al. [2020a], including statistical (diverse data distributions across clients), model (variations in model training methods), and communication heterogeneity (inconsistent network conditions and resource constraints). Among these, statistical heterogeneity (distribution mismatch, henceforth referred to as non-IID) across clients poses a significant challenge, often degrading global model performance. Standard FL methods such as FedAvg McMahan et al. [2017], originally designed for IID data, fail to generalize effectively across diverse client populations Zhao et al. [2018], motivating the development of personalization schemes that better align models with local client data.

To address data heterogeneity and personalization, several works have reformulated the optimization objectives of traditional FL methods. FedProx Li et al. [2020b] adds a proximal term to FedAvg to limit the divergence of local updates from the global model. However, as noted in Jiang et al. [2019], optimizing solely for global accuracy can hinder effective personalization at the client level. To overcome this, personalized federated learning (PFL) Sim et al. [2019], Li et al. [2021a] has emerged as a paradigm that balances global collaboration with client-specific adaptation. Rather than relying on a single global model, PFL allows each client to tailor its model to local data and preferences Kulkarni et al. [2020], typically following a two-stage strategy: global training followed by local fine-tuning. However, existing PFL methods employ static regularization weights Fallah et al. [2020], Li et al. [2021a] or fixed similarity-based aggregation Chen et al. [2024], which fail to account for dynamic variations in client drift, data distribution, and model reliability over training rounds. This rigidity can lead to sub-optimal personalization or even negative transfer, particularly in settings with highly non-IID data.

Motivated by the limitations of static personalization and uniform aggregation strategies, we propose Similarity-Aware PErsonalized Federated Learning (SAPE-FL), a novel framework that jointly optimizes global collaboration and client-specific adaptation. SAPE-FL introduces a two-level anchoring mechanism, where each client aligns its model with both the global model and a similarity-weighted average of peer models. This dual anchoring is guided by clientspecific adaptive regularization, computed using model output and weight-based similarity. Our measure combines both model output and model weight-based similarities, making it robust to structural and functional differences across clients. While weight-based similarity captures structural alignment, output similarity identifies functionally similar models despite parameter differences. To construct such reliable anchors, each client filters out dissimilar peer models using a similarity threshold, ensuring that only relevant peer information influences personalization. On the server side, SAPE-FL performs similarity-weighted aggregation of client model updates, where each client’s contribution is scaled by its alignment with peer consensus. This suppresses noisy or misaligned updates and enhances quality of the global model.

We evaluate SAPE-FL against state-of-the-art methods across diverse tasks, including synthetic data classification, human activity recognition and image classification under non-IID or statistical heterogeneity. Our empirical results demonstrate that SAPE-FL framework consistently outperforms baselines, especially in settings with high statistical heterogeneity and limited client data availability. These findings highlight that our novel similarity-based personalization and aggregation strategy effectively personalizes federated learning to the unique characteristics of each client.

The key contributions of this paper are mentioned below.

• We propose SAPE-FL, a novel similarity-aware personalized FL framework that anchors client models to both global and peer-averaged models, and uses an aggregation scheme that scale client updates based on their coherence with peers and prevent negative transfer.

• We design an adaptive personalization scheme that computes client-specific regularization coefficients based on model output and weight similarity, enabling selective alignment with the global model and peer models.

• We provide theoretical insights into the convergence behavior and performance guarantees of our method.

• We empirically validate SAPE-FL’s effectiveness on diverse datasets under high statistical heterogeneity.

Through these innovations, SAPE-FL advances the state-of-the-art in personalized federated learning by jointly enabling global knowledge sharing and client-specific adaptation via similarity-aware optimization and aggregation.

## 2 Related Work

Recently, Federated Learning (FL) has gained traction in the research community resulting in a large and growing body of literature Nevrataki et al. [2023]. Federated learning, first introduced in McMahan et al. [2017], enables decentralized model training while preserving data privacy. The FedAvg algorithm proposed allows clients to perform several rounds of local stochastic gradient descent (SGD) Ruder [2016] before synchronizing with server, reducing communication overhead compared to centralized training. Since its inception, FL has received growing attention leading to advancements in communication efficiency Konecnˇ y et al. [2016], privacy guarantees Agarwal et al. [2018],\` cryptographic protections Bonawitz et al. [2017], and optimization under resource constraints Wang et al. [2019].

Despite strong theory and empirical performance Li et al. [2020c], FedAvg struggles under statistical heterogeneity. Zhao et al. [2018] showed that non-IID data can degrade performance due to conflicting gradient updates. To mitigate these, adaptive FL methods have been proposed. Wang et al. [2019] adjusted local update frequencies based on client resources, particularly suited for edge environments. LoAdaBoost Huang et al. [2020] selectively continues training for under performing clients, reducing overhead and improving efficiency. Gradient-based clustering methods, such as CFL Sattler et al. [2020], form clusters based on update similarity to mitigate heterogeneity.

## 2.1 Adaptive Client Selection Methods

Optimizing client selection is crucial in FL to reduce communication and computation costs. Chen et al. [2022] proposed algorithms for efficient client participation, though without fully addressing heterogeneity. FedNorm Zhao et al. [2022] prioritizes clients with high informational value, improving communication efficiency. FedMarl Zhang et al. [2022], inspired by multi-agent reinforcement learning, dynamically selects clients based on runtime performance. CSFedAvg Zhang et al. [2021] improves convergence by favoring clients with less non-IID data. RIPFL Qin et al. [2023], leveraging Dempster–Shafer theory Sensoy et al. [2018], quantifies client reliability for robust selection.

Multi-task and Meta-learning in Federated Learning Multi-task and meta-learning approaches have gained traction in FL due to their ability to handle heterogeneous data distributions. MOCHA Smith et al. [2017] jointly learns client-specific models and a task-similarity matrix to capture inter-client relationships. VIRTUAL Corinzia et al. [2019] models inter-task variability and dependency through a variational multi-task framework. Jiang et al. [2019] established connections between FedAvg and model-agnostic meta-learning (MAML) Finn et al. [2017], which inspired methods like Per-FedAvg Fallah et al. [2020] that integrate meta-learning objectives into FL training. Other works include ARUBA Khodak et al. [2019], which improves gradient-based meta-learning, and Peterson et al. [2019], which leverages mixture-of-experts (MoE) Nowlan and Hinton [1990] for personalized model interpolation.

## 2.2 Personalized Federated Learning Approaches

Personalization has become a core focus in FL due to the challenges posed by non-IID client data. Approaches range from local fine-tuning Mansour et al. [2020] to advanced decomposition-based methods Arivazhagan et al. [2019], Yin and Mao [2025] and clustering-based solutions Ghosh et al. [2020]. The recently proposed LCFed Zhang et al. [2025] introduces an efficient clustering framework that improves performance and communication efficiency in heterogeneous environments by forming compact client clusters using cosine-based distance metrics. Hypernetwork-based methods like PeFLL Scott et al. [2024] generate personalized models in a single forward pass using learnt embeddings, while FedRod Chen and Chao [2022], Scaffold Karimireddy et al. [2020], and MOON Li et al. [2021b] impose regularization or similarity constraints to improve personalization during local updates. FedPAC Xu et al. [2023] aligns local and global feature embeddings, though it is most effective under label distribution shift. Model interpolation remains a prominent technique: FedProx Li et al. [2020b] adds a proximal term to address client drift; Ditto FL Li et al. [2021a] explicitly separates global and personalized models with fixed regularization. FedACS Chen et al. [2024] incorporates attention-weighted peer model aggregation, while FedPer Arivazhagan et al. [2019] separates personalized (top) layers and globally shared (base) layers in a neural network architecture.

Despite advancements, existing methods struggle in handling client heterogeneity and personalization Yang et al. [2024]. Many rely on static regularization, inefficient aggregation or clustering, which fail under dynamic data shifts. SAPE-FL addresses this by anchoring client models to both global and peer-averaged models, using a hybrid similarity measure based on model weights and outputs. This enables robust peer matching and aggregation, ensuring effective personalization.

## 3 Mathematical Preliminaries

Traditional FL methods McMahan et al. [2017], Collins et al. [2022] aim to collaboratively train a global model by aggregating local updates from multiple clients, without requiring each client’s sensitive data to be pooled at a central location. Consider a FL setting with $\bar { C }$ distributed clients, each holding data $\mathcal { D } _ { c }$ drawn from a local distribution given as $\mathcal { D } _ { c } = \{ ( \mathbf { x } _ { j } ^ { c } , y _ { j } ^ { c } ) \} _ { j = 1 } ^ { n _ { c } }$ , where $\mathbf { x } _ { i } ^ { c } \in \mathbb { R } ^ { \mathrm { d i m } }$ are input features and $y _ { j } ^ { c } \in \{ \bar { 1 } , \ldots , { \mathcal { C } } \}$ are class labels. The server maintains a global model $\theta _ { G }$ which is iteratively updated based on client contributions. The objective in classical FL McMahan et al. [2017] is to learn a single global model $\theta _ { G }$ that minimizes the weighted sum of local empirical risks:

$$
\operatorname* { m i n } _ { \theta _ { G } } \mathcal { L } ( \theta _ { G } ) = \sum _ { c } p _ { c } \mathcal { L } _ { c } ( \theta _ { G } )\tag{1}
$$

where $\mathcal { L } _ { c } ( \theta _ { G } )$ denotes the local loss at client $c ,$ and $p _ { c }$ is the participation probability assigned to each client, satisfying $\textstyle \sum _ { c } p _ { c } = 1$ and $p _ { c } \geq 0$ . In the canonical example of FedAvg, $p _ { c }$ is typically set proportional to the client’s data volume: $\begin{array} { r } { p _ { c } = \frac { n _ { c } } { \sum _ { c } n _ { c } } } \end{array}$ . This formulation assumes that all clients contribute equally to a single global objective, which may not be appropriate in heterogeneous (non-IID) settings.

## 3.1 Statistical heterogeneity

Statistical heterogeneity in FL can be perceived in various forms. For instance, in a handwritten number recognition task, clients may have: (i) differing label distributions $p _ { i } ( y ) \neq p _ { j } ( y ) \forall i , j \in C$ , or (ii) differing feature distributions

(different writing styles - stroke/thickness) $p _ { i } ( \mathbf { x } ) \neq p _ { j } ( \mathbf { x } )$ despite sharing the same label distribution $p _ { i } ( y ) = p _ { j } ( y )$ Such heterogeneity can significantly degrade global model performance. To address this, personalized federated learning (PFL) methods aim to learn client-specific models $\theta _ { c }$ that leverage global knowledge while adapting to local data.

The personalized optimization objective is defined as:

$$
\Theta ^ { \star } = \theta _ { 1 } ^ { \star } , \star \star , \theta _ { C } ^ { \star } = \underset { \theta _ { 1 } , \ldots , \theta _ { C } \in \Theta } { \arg \operatorname* { m i n } } \sum _ { c = 1 } ^ { C } p _ { c } \mathcal { L } _ { c } ( \theta _ { c } ) ,\tag{2}
$$

where $\theta _ { c } ^ { \star }$ is the optimal model for client $c ,$ and $\mathcal { L } _ { c } ( \theta _ { c } )$ denotes the local empirical risk:

$$
\mathcal { L } _ { c } ( \theta _ { c } ) \triangleq \frac { 1 } { | \mathcal { D } _ { c } | } \sum _ { ( \mathbf { x } , y ) \in \mathcal { D } _ { c } } \ell ( \theta _ { c } ( \mathbf { x } ) , y ) ,\tag{3}
$$

where $\ell ( \cdot , \cdot )$ is a loss criterion (cross-entropy) and dataset $\mathcal { D } _ { c }$ consists of input-output pairs $\left( \mathbf { x } , y \right)$ , where $\mathbf { x } \in \mathbb { R } ^ { \mathrm { d i m } }$ To better handle statistical heterogeneity, proximal-based PFL methods Li et al. [2020b, 2021a] extend this objective by introducing regularization terms that constrain the drift of local models from a shared global reference.

## 4 Framework

The key objective of this paper is to develop a federated learning framework to facilitate collaborative learning across clients with non-IID data distributions $\mathcal { D } _ { c } .$ . As mentioned in the Mathematical Preliminaries section, we consider a personalized federated learning of $C$ clients in a heterogeneous environment, i.e., the data distribution across clients $\mathcal { D } _ { c }$ is non-IID. Precisely, we aim to learn the optimal personalized client models $\Theta ^ { \star } \left( \mathrm { E q } . \left( 2 \right) \right)$ .

To efficiently address the challenges posed by statistical heterogeneity in federated learning setting, we propose SAPE-FL framework, a novel Similarity-Aware PErsonalized Federated Learning (SAPE-FL) framework. SAPE-FL is designed to adaptively regularize and balance local personalization and global collaboration by introducing (i) dual anchoring of each client model to both the global model and a peer-averaged model, and (ii) similarity-aware aggregation of client updates at the server. The mathematical formulation of the SAPE-FL framework is given as:

$$
\Theta ^ { \star } = \operatorname * { a r g m i n } _ { \theta _ { 1 } , \cdot \cdot \cdot , \theta _ { C } } \sum _ { c } p _ { c } \mathcal { L } _ { c } ( \theta _ { c } ) + S _ { c } ^ { P } ( \theta _ { c } , \bar { \theta } _ { c } ) + S _ { c } ^ { G } ( \theta _ { c } , \theta _ { G } )\tag{4}
$$

where $\mathcal { L } _ { c } ( \theta _ { c } )$ is the local empirical risk at client c having a personalized model $\theta _ { c }$ , and $\begin{array} { r l } { S _ { c } ^ { G } ( \cdot , \cdot ) , S _ { c } ^ { P } ( \cdot , \cdot ) } \end{array}$ are the client specific similarity-aware regularization (penalty) coefficients that enforce alignment with the global and peer-averaged models at any given round $\bar { t } ,$ respectively. We discuss in detail $S _ { c } ^ { G } , S _ { c } ^ { P } , \bar { \theta } _ { c }$ in the next sections. The global model $\theta _ { G }$ is updated iteratively by aggregating client updates in a similarity-weighted manner.

We present the complete workflow of the proposed SAPE-FL framework in Algorithm 1 and Algorithm 2.

## 4.1 Dual Anchoring: Similarity-Aware Regularization

To effectively balance personalization and global collaboration, SAPE-FL introduces a dual anchoring mechanism. Each client optimizes its local model by anchoring simultaneously to the global model $\theta _ { G }$ and to a peer-averaged model ${ \bar { \theta } _ { c } } ,$ constructed using a subset of similar clients. This dual anchoring strategy helps clients incorporate useful knowledge from both centralized and localized sources while avoiding influence from dissimilar or noisy models. The personalized client objective is defined as:

$$
\operatorname* { m i n } _ { \theta _ { c } } \mathcal { L } _ { c } ( \theta _ { c } ) + \mathcal { S } _ { c } ^ { P } ( \theta _ { c } , \bar { \theta } _ { c } ) + \mathcal { S } _ { c } ^ { G } ( \theta _ { c } , \theta _ { G } )\tag{5}
$$

The similarity-aware regularization coefficients $S _ { c } ^ { G } , S _ { c } ^ { P }$ are defined as a convex combination of (i) model output-based and (ii) weight-based similarity. We use this hybrid similarity metric to enhance the robustness of SAPE-FL to behavioral differences and structural variations among clients. We define the coefficients $\mathcal { S } _ { c } ^ { G } , \mathcal { S } _ { c } ^ { P }$ as:

$$
\begin{array} { c } { { S _ { c } ^ { P } \triangleq \delta \cdot \vartheta ( f ( \theta _ { c } , \mathcal { D } _ { c } ) , f ( \bar { \theta } _ { c } , \mathcal { D } _ { c } ) ) + ( 1 - \delta ) \cdot \vartheta ( \theta _ { c } , \bar { \theta } _ { c } ) } } \\ { { S _ { c } ^ { G } \triangleq \delta \cdot \vartheta ( f ( \theta _ { c } , \mathcal { D } _ { c } ) , f ( \theta _ { G } , \mathcal { D } _ { c } ) ) + ( 1 - \delta ) \cdot \vartheta ( \theta _ { c } , \theta _ { G } ) } } \end{array}\tag{6}
$$

where $\delta \in [ 0 , 1 ]$ is a tunable influence factor that balances emphasis between output similarity (model predictions) and structural similarity (model weights) and $\vartheta ( \cdot , \cdot )  [ 0 , 1 ]$ is a positive function defined for any two inputs $\mathcal { X } , \mathcal { X } ^ { \prime }$ as:

$$
\vartheta ( \mathcal { X } , \mathcal { X } ^ { \prime } ) = 1 - \operatorname* { m a x } ( 0 , \cos ( \mathcal { X } , \mathcal { X } ^ { \prime } ) )\tag{7}
$$

In the definitions of $ { \boldsymbol { S } } _ { c } ^ { G }$ and $ { S _ { c } ^ { P } }$ , the first term penalizes discrepancies in model outputs by measuring the alignment between the predictive behaviors $f ( \theta , { \mathcal { D } } _ { c } )$ ) of the two models evaluated on the same client dataset $\mathcal { D } _ { c } .$ . The second term imposes a penalty based on structural alignment, capturing directional consistency between the model parameters. Unlike conventional parameter-distance regularization schemes Li et al. [2020b], Zhang et al. [2025], which explicitly enforce Euclidean proximity between model parameters, the proposed similarity-aware penalties do not induce direct parameter contraction. Instead, they promote functional alignment and directional consistency with the global and peer-averaged anchors, allowing structurally distinct yet behaviorally consistent personalized models to coexist.

Further, the similarity-based regularization coefficients adaptively regulate the trade-off between global collaboration and local personalization. Larger regularization coefficients strengthen the alignment with corresponding anchor model (either global or peer-averaged), encouraging knowledge sharing among aligned clients, while smaller values relax the regularization effect and enable stronger local adaptation, thereby mitigating the risk of negative transfer.

## 4.2 Peer Model Filtering and Averaging

To construct the peer-averaged model ${ \bar { \theta } } _ { c } ^ { t } .$ , each client evaluates the relevance of peer client models by computing similarity scores and filters out dissimilar peers based on a predefined threshold $S _ { \mathrm { m i n } }$ . The parameter $S _ { \mathrm { m i n } }$ is carefully selected to balance inclusiveness and robustness. It ensures only peers that are sufficiently aligned contribute to the personalized anchor. The peer-averaged model is given by:

$$
\bar { \theta } _ { c } = \frac { 1 } { \sum _ { j } S _ { c j } \cdot \mathbb { I } [ S _ { c j } > S _ { \operatorname* { m i n } } ] } \sum _ { j } S _ { c j } \cdot \theta _ { j } \cdot \mathbb { I } [ S _ { c j } > S _ { \operatorname* { m i n } } ]\tag{8}
$$

where $\mathbb { I } [ \cdot ]$ is the indicator function. Analogous to Eq. (6), the similarity score $S _ { c j }$ between client c and peer $j$ is given as:

$$
S _ { c j } = \delta \cdot \tilde { S } _ { c j } + ( 1 - \delta ) \cdot \hat { S } _ { c j }\tag{9}
$$

where $\tilde { S } _ { c j }$ is the output similarity term measuring the alignment between predictive behaviors of the two models:

$$
\tilde { S } _ { c j } = 1 - \vartheta \big ( f ( \theta _ { c } , \mathcal { D } _ { c } ) , f ( \theta _ { j } , \mathcal { D } _ { c } ) \big )\tag{10}
$$

where $f ( \theta , { \mathcal { D } } _ { c } )$ denotes the outputs of model θ on dataset $\mathcal { D } _ { c }$ and $\hat { S } _ { c j }$ is the weight similarity measuring the structural alignment between the model parameters defined as:

$$
\hat { S } _ { c j } = 1 - \vartheta ( \theta _ { c } , \theta _ { j } )\tag{11}
$$

In $\mathrm { E q . } ( 9 ) , \tilde { S } _ { c j }$ term captures whether the peer produces similar predictions on the client’s data, whereas $\hat { S } _ { c j }$ provides an additional information based on the proximity of model weights, which is useful when output similarity alone is noisy or insufficient (especially during early rounds). This scheme enables each client to construct a robust and personalized peer anchor by leveraging only those peers that are behaviorally and structurally aligned, thereby improving personalization and robustness to noisy or adversarial updates.

## 4.3 Similarity-Weighted Aggregation at Server

Following client-level optimization of personalized models, each client computes its model update $\Delta _ { c } ^ { t }$ as

$$
\Delta _ { c } ^ { t } = \tilde { \theta } _ { c } ^ { t } - \theta _ { G } ^ { t }
$$

where $\widetilde { \theta } _ { c } ^ { t }$ is the client model initialized with $\theta _ { G }$ and updated locally for $E$ epochs. The model update $\Delta _ { c } ^ { t }$ is transmitted back to the server along with the aggregate peer similarity $s _ { c } = \textstyle \sum _ { j } S _ { c j }$ and the optimal client model $\theta _ { c } ^ { i + 1 }$ . The global model $\theta _ { G } ^ { t }$ is updated via a similarity-weighted aggregation:

$$
\theta _ { G } ^ { t + 1 } = \theta _ { G } ^ { t } + \frac { 1 } { \sum _ { c } s _ { c } ^ { t } } \sum _ { c } s _ { c } ^ { t } \cdot \Delta _ { c } ^ { t }\tag{12}
$$

This aggregation gives more weight to clients whose models align with their peers, thereby encouraging updates that represent consensus learning behavior. Clients with lower scores are down-weighted, thus suppressing noisy updates.

This end-to-end scheme of dual anchoring through similarity-aware regularization, selective peer model averaging, and similarity-weighted aggregation makes SAPE-FL robust against data heterogeneity and misaligned signals.

Algorithm 1 Server-Side Procedure (Round t)   
1: Input: Client participation ratio r   
2: Randomly sample C clients according to ratio r   
3: for each client $c \in C$ in parallel do   
4: Send global model $\theta _ { G } ^ { t }$ and peer models $\theta _ { j } ^ { t } \left( \mathrm { A l g o } . 2 \right)$   
5: Receive model $\theta _ { c } ^ { t + 1 }$ , update $\Delta _ { c } ^ { t } ,$ similarity score $s _ { c } ^ { t }$   
6: end for   
7: Update global model using similarity scores: $\begin{array} { r } { \small \theta _ { G } ^ { t + 1 } = \theta _ { G } ^ { t } + \frac { 1 } { \sum _ { c } s _ { c } ^ { t } } \sum _ { c } s _ { c } ^ { t } \cdot \Delta _ { c } ^ { t } } \end{array}$   
8: return $\theta _ { G } ^ { t + 1 }$ as the updated global model   
Algorithm 2 Client-Side Procedure (Client c)   
1: Input: Global model $\theta _ { G } ^ { t }$ , Peer models $\{ \theta _ { j } ^ { t } \}$ , Data $\mathcal { D } _ { c }$   
2: Parameters: Similarity threshold $S _ { \mathrm { m i n } }$ , Influence factor $\delta$   
3: Initialize local model: $\tilde { \theta } _ { c } ^ { t } \gets \theta _ { G } ^ { t }$   
4: Update $\widetilde { \theta } _ { c } ^ { t }$ on $\mathcal { D } _ { c }$ for $E$ epochs   
5: for each peer model $\theta _ { j } ^ { t }$ do   
6: Compute output similarity $\tilde { S } _ { c j } ^ { t }$ as per Eq. (10)   
7: Compute weight similarity: $\hat { S } _ { c j } ^ { t }$ as per Eq. (11)   
8: Combined similarity score: $S _ { c j } ^ { t } = \delta \tilde { S } _ { c j } ^ { t } + ( 1 - \delta ) \hat { S } _ { c j } ^ { t }$   
9: end for   
10: Build peer-averaged model $\bar { \theta } _ { c } ^ { t }$ as per Eq. (8)   
11: Compute $\bar { \partial _ { c } ^ { G } } , \bar { \partial _ { c } ^ { P } }$ as per Eq. (6)   
12: Find the optimal model $\theta _ { c } ^ { t + 1 }$ that minimizes the loss: arg mi $\begin{array} { r } { { 1 } _ { \theta _ { c } } \mathscr { L } ( \theta _ { c } ) + S _ { c } ^ { P } ( \theta _ { c } , \bar { \theta } _ { c } ^ { t } ) + S _ { c } ^ { G } ( \theta _ { c } , \theta _ { G } ^ { t } ) } \end{array}$   
13: Compute model update $\Delta _ { c } ^ { t } = \tilde { \theta } _ { c } ^ { t } - \theta _ { G } ^ { t }$   
14: Compute aggregate similarity score: $\begin{array} { r } { s _ { c } ^ { t } = \sum _ { j } S _ { c j } ^ { t } } \end{array}$   
15: return model $\theta _ { c } ^ { t + 1 }$ , update $\Delta _ { c } ^ { t } ,$ scores $s _ { c }$ to server

## 4.4 Algorithmic Workflow of SAPE-FL Framework

We now present the complete workflow of our method. The server-side procedure at round t is in Algorithm 1 and the client-side procedure for personalized federated learning at client c is in Algo. 2. In each round, a series of coordinated steps between the server and clients are performed to progressively refine both global and personalized client models.

The server begins each round by sampling a subset of clients according to the participation ratio r and broadcasting the current global model $\theta _ { G } ^ { t }$ along with peer models, to the selected clients. Each participating client initializes its local model $\tilde { \theta } _ { c } ^ { t } \gets \theta _ { G } ^ { t }$ , and updates it locally for $E$ epochs using its private dataset $\mathcal { D } _ { c }$ . Following this, each client computes the combined similarity score $S _ { c j }$ with its peers by computing output similarity $\tilde { S } _ { c j }$ and weight similarity $\hat { S } _ { c j } ^ { t }$ . Models with similarity scores above threshold $S _ { \mathrm { m i n } }$ are then used to compute a peer-averaged model $\hat { \theta } _ { c } ^ { t }$ (Eq. (8)).

With both the global and peer anchors, each client solves a personalized optimization problem that balances empirical risk minimization with regularization toward the global and the peer-averaged model. The regularization coefficients $ { \mathcal { S } } _ { c } ^ { P }$ and $ { \mathcal { S } } _ { c } ^ { G }$ are computed based on the client’s alignment with the respective anchors. After personalized training, each client retains its personalized model and computes the model update $\Delta _ { c } ^ { t }$ which gets transmitted back to the server alongside the updated model $\theta _ { c } ^ { t + 1 }$ and similarity scores $s _ { c }$

At the server, these similarity-weighted updates are aggregated to refine the existing global model $\theta _ { G } ^ { t }$ . This similarityaware aggregation ensures that clients whose updates are well-aligned with their peers (higher $s _ { c } )$ contribute strongly to the global update, while noisy or dissimilar updates are naturally down-weighted. The refined global model $\theta _ { G } ^ { t + 1 }$ is then shared in the next round, continuing the cycle of personalized optimization and collaborative learning.

## 5 Convergence Analysis of SAPE-FL Method

In this section, we analyze the theoretical aspects of our proposed SAPE-FL algorithm. Our goal is to understand the convergence behavior of the global model and personalized models under the proposed similarity-weighted personalization and aggregation scheme. We aim to formally prove a per-round improvement bound for the global model objective using similarity-weighted model updates from participating clients. For analysis, we consider the setting where each client c performs local training initialized at the global model $\theta _ { G }$ , and constructs its model update $\Delta _ { c }$ based on local and peer-regularized learning. The server then refines the global model using the client model updates received. We first state some key assumptions required for the convergence analysis of our proposed SAPE-FL framework:

Assumption 1 (Smoothness). Each client objective $\mathcal { L } _ { c } ( \theta )$ is L-smooth: for all θ, θ<sup>′</sup>, $\| \nabla \mathcal { L } _ { c } ( \theta _ { c } ) - \nabla \mathcal { L } _ { c } ( \theta _ { c } ^ { \prime } ) \| \le$ $L \| \theta _ { c } - \bar { \theta } _ { c } ^ { \prime } \|$

Assumption 2 (Strong µ-Convexity). Each client’s local objective function $\mathcal { L } _ { c }$ is strongly µ-convex, formally:

$$
\mathcal { L } _ { c } ( \boldsymbol { \theta } _ { c } ) \geq \mathcal { L } _ { c } ( \boldsymbol { \theta } _ { c } ^ { \prime } ) + \langle \nabla \mathcal { L } _ { c } ( \boldsymbol { \theta } _ { c } ^ { \prime } ) , \boldsymbol { \theta } _ { c } - \boldsymbol { \theta } _ { c } ^ { \prime } \rangle + \frac { \mu } { 2 } \| \boldsymbol { \theta } _ { c } - \boldsymbol { \theta } _ { c } ^ { \prime } \| ^ { 2 } \quad \forall \boldsymbol { \theta } _ { c } , \boldsymbol { \theta } _ { c } ^ { \prime }
$$

Assumption 3 (Gradient Bounded Variance). The gradients on each client c (mini-batch sampling) have bounded variance. For some $\sigma ^ { 2 } \geq 0 , \mathbb { E } [ \| \nabla \mathcal { L } _ { c } ( \theta ; \xi ) - \nabla \mathcal { L } _ { c } ( \theta ) \| ^ { 2 } ] \leq \sigma ^ { 2 } \forall c , \theta ,$ , where ξ indexes a random data sample.

Assumptions 1, 2, and 3 are standard in the FL theory found in the literature T Dinh et al. [2020], Li et al. [2020c,b]. Assumption 1 is used to bound the improvement in loss after each update. Assumption 2, commonly adopted in FL theory, is required to establish linear convergence rates under appropriate conditions. Assumption 3 bounds the gradients and ensures that the noise introduced in local updates does not cause divergence. Under the assumptions stated above, we prove the descent condition of the expected global objective using the following theorem.

Theorem 1 (One-Round Global Model Improvement). Let $\mathcal { L } _ { c } ( \theta _ { c } )$ be the client’s L-Lipschitz smooth objective with bounds on its gradients. Let the similarity scores $s _ { c } ^ { t }$ also be bounded and non-negative. After each client trains locally, the similarity-aware aggregation satisfies the inequality:

$$
\sum _ { c } \frac { s _ { c } ^ { t } } { \sum _ { j } s _ { j } ^ { t } } \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) ^ { \top } \Delta _ { c } ^ { t } \leq - \rho \| \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) \| ^ { 2 } f o r s o m e \rho > 0
$$

Then the global model satisfies the descent condition:

$$
\mathbb { E } [ \mathcal { L } ( \theta _ { G } ^ { t + 1 } ) ] \leq \mathcal { L } ( \theta _ { G } ^ { t } ) - \rho \| \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) \| ^ { 2 } + \frac { L } { 2 } \mathbb { E } [ \| \theta _ { G } ^ { t + 1 } - \theta _ { G } ^ { t } \| ^ { 2 } ] .
$$

The detailed proof of Theorem 1 is provided in Appendix. The proof leverages the L-smoothness of the global objective $\mathcal { L } ( \theta _ { G } )$ and expands $\mathcal { L } ( \theta _ { G } ^ { t + 1 } )$ using the standard Taylor expansion. The aggregated update scheme used in our proposed framework is shown to form a descent direction under the assumption that each client update $\Delta _ { c } ^ { t }$ approximately aligns with the negative local gradient $- \eta \nabla \mathcal { L } ( \theta _ { G } ^ { t } )$ Li et al. [2020b, 2021a]. The similarity scores act as attention weights, emphasizing aligned/useful updates and reducing the impact of noisy or misaligned clients. The derived inequality implies that the expected global loss decreases at each communication round, provided that client updates are sufficiently aligned and similarity weights are effective.

Next, to understand the impact of peer-based collaboration in our personalized framework, we examine how aggregating peer models influences the statistical properties of each client’s personalized loss. Specifically, we consider the peeraveraged model $\bar { \theta } _ { c }$ . When peer models are aligned in behavior, their aggregation acts as a form of ensembling that retains unbiasedness and reduces variance in the resulting estimate. The following lemma formalizes this variance reduction, illustrating the statistical benefits of incorporating similar peer models into local personalization.

Lemma 1 (Peer-Average Variance Reduction). Let $C _ { k }$ be the set containing client c and its similar peers $j .$ Assume all peer models share the same mean, i.e. $\mathbb { E } [ \theta _ { j } ^ { t } ] = \mathbb { E } [ \theta _ { j ^ { \prime } } ^ { t } ]$ for every $j , j ^ { \prime } \in C _ { k }$ , and have covariance $\dot { \mathrm { C o v } } ( \theta _ { j } ^ { t } ) \dot { = } \Sigma$ . With the per-client estimation-error variance defined as $v = \mathrm { T r } ( \Sigma )$ , the peer-averaged model $\bar { \theta } _ { c } ^ { t }$ retains the same mean as the individual peers and enjoys reduced variance $\begin{array} { r } { \mathrm { C o v } ( \bar { \theta } _ { c } ^ { t } ) \stackrel { \cdot } { = } \frac { 1 } { | C _ { k } | } \dot { \Sigma } . } \end{array}$ . Consequently, the estimation-error variance of the peer average is $\begin{array} { r } { \bar { v } = \operatorname { T r } ( \operatorname { C o v } ( \bar { \theta } _ { c } ^ { t } ) ) = \frac { 1 } { | C _ { k } | } v . } \end{array}$

We provide the proof of Lemma 1 in Appendix. The proof leverages the independence and identical expectation of peer models under the cluster to show that the peer-averaged estimate $\bar { \theta } _ { c } ^ { t }$ remains unbiased while reducing the variance by a factor of $\frac { 1 } { | C _ { k } | }$ . This is analogous to ensemble learning where combining multiple independent estimates improves robustness and accuracy. We show that the loss incurred by $\hat { \theta } _ { c } ^ { t }$ is at most the average loss of individual peer models, further justifying its use as personalized anchor.

We study the convergence of the client objective that combines local loss minimization with regularization to align with the global and peer-averaged model. The similarity-weighted averaging and anchoring enables information sharing within client neighborhoods while preserving privacy and heterogeneity across the federation. The following theorem formalizes convergence of the client objective.

Theorem 2 (Convergence of Personalized Objective). If each participating client solves the sub-problem given as:

$$
\operatorname * { m i n } _ { \theta _ { c } } \mathcal { L } _ { c } ( \theta _ { c } ) \ + \ S _ { c } ^ { P } ( \theta _ { c } , \bar { \theta } _ { c } ^ { t } ) + S _ { c } ^ { G } ( \theta _ { c } , \theta _ { G } ^ { t } ) ,
$$

where $\bar { \theta } _ { c } ^ { t }$ is computed via similarity scores $S \in [ 0 , 1 ]$ mentioned in Eq. (9) and filtering threshold $S _ { m i n }$ to exclude dissimilar peers. Then the global objective (aggregated objective across all clients)

$$
\mathcal { L } ( \theta _ { \{ 1 \cdots C \} } ) = \sum _ { c } \mathcal { L } _ { c } ( \theta _ { c } ) + \sum _ { c } \bigl ( S _ { c } ^ { P } ( \theta _ { c } , \bar { \theta } _ { c } ^ { t } ) + S _ { c } ^ { G } ( \theta _ { c } , \theta _ { G } ^ { t } ) \bigr )
$$

decreases monotonically and the method converges to a first-order stationary point. With appropriate step-size $\eta > 0 ,$ an ϵ-approximate solution is reached in $\begin{array} { r } { \mathcal { O } \left( \frac { 1 } { \vert C _ { k } \vert \epsilon ^ { 2 } } \right) } \end{array}$ communication rounds, where $| C _ { k } |$ is the number ofsimilar peers $( S _ { c j } ^ { t } > S _ { m i n } ) f o r a$ given client c.

Theorem 2 shows that SAPE-FL converges to a stationary point with monotonic decrease in the joint objective. As per Lemma 1, the variance of the peer-averaged estimate is reduced by a factor of $\frac { 1 } { | C _ { k } | }$ . This tighter variance directly translates into an improved convergence rate of $\begin{array} { r } { \mathcal { O } \left( \frac { 1 } { | C _ { k } | \epsilon ^ { 2 } } \right) } \end{array}$ , faster than the classical $\begin{array} { r } { \mathcal { O } \left( \frac { 1 } { \epsilon ^ { 2 } } \right) } \end{array}$ bound Huang et al. [2021], Collins et al. [2022]. The global model stabilizes via similarity-aware aggregation, while the local models evolve toward cluster-aligned personalized optima.

Additionally, we theoretically prove that our similarity-weighted global objective converges linearly in the strongly convex case and attains the standard first-order rate in the non-convex case. The associated theorems and their proofs are provided in the appendix due to space constraints.

## 6 Experiments

In this section, we evaluate the empirical performance of our SAPE-FL framework by comparing it with a set of relevant state-of-the-art federated learning and personalized FL baselines. We first provide the experimental settings and then discuss the empirical results of our framework.

## 6.1 Baselines

Given the breadth of FL literature, we focus our comparisons on methods that are conceptually aligned with our SAPE-FL approach. For completeness, we includes a local-only baseline where each client trains independently without collaboration. The baselines considered are: (i) FedAvg McMahan et al. [2017], (ii) PerFedAvg Fallah et al. [2020], (iii) FedProx Li et al. [2020b], (iv) Ditto FL Li et al. [2021a], (v) FedACS Chen et al. [2024], (vi) LCFed Zhang et al. [2025], (vii) FedAFK Yin and Mao [2025], (viii) FedAS Yang et al. [2024].

Table 1: Accuracy Scores (%). Comparison of SAPE-FL with baseline methods. Each cell signifies the mean accuracy (standard deviation) computed across all clients at the end of all communication rounds. Bold values indicate the best.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Synthetic</td><td rowspan=1 colspan=1>HAR</td><td rowspan=1 colspan=1>AccGyro</td><td rowspan=1 colspan=1>CIFAR-10</td><td rowspan=1 colspan=1>CIFAR-100</td><td rowspan=1 colspan=1>MNIST</td><td rowspan=1 colspan=1>FMNIST</td><td rowspan=1 colspan=1>EMNIST</td></tr><tr><td rowspan=1 colspan=1>Local-only</td><td rowspan=1 colspan=1> $8 0 . 6 4 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $7 3 . 2 5 \pm 0 . 6 1$ </td><td rowspan=1 colspan=1> $7 5 . 2 7 \pm 0 . 9 2$ </td><td rowspan=1 colspan=1> $7 7 . 4 5 \pm 0 . 8 6$ </td><td rowspan=1 colspan=1> $4 1 . 6 6 \pm 0 . 3 5$ </td><td rowspan=1 colspan=1> $6 8 . 4 5 \pm 0 . 1 1$ </td><td rowspan=1 colspan=1> $8 2 . 4 0 \pm 0 . 5 5$ </td><td rowspan=1 colspan=1> $6 3 . 2 6 \pm 0 . 2 3$ </td></tr><tr><td rowspan=1 colspan=1>FedAvg</td><td rowspan=1 colspan=1> $8 3 . 1 9 \pm 0 . 2 5$ </td><td rowspan=1 colspan=1> $7 6 . 6 1 \pm 0 . 3 3$ </td><td rowspan=1 colspan=1> $7 8 . 1 5 \pm 0 . 8 4$ </td><td rowspan=1 colspan=1> $7 9 . 2 3 \pm 0 . 1 6$ </td><td rowspan=1 colspan=1> $4 2 . 3 1 \pm 0 . 6 8$ </td><td rowspan=1 colspan=1> $7 2 . 2 4 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1> $8 4 . 3 2 \pm 0 . 1 5$ </td><td rowspan=1 colspan=1> $6 7 . 2 2 \pm 0 . 1 6$ </td></tr><tr><td rowspan=1 colspan=1>PerFedAvg</td><td rowspan=1 colspan=1> $8 7 . 6 6 \pm 0 . 2 3$ </td><td rowspan=1 colspan=1> $7 7 . 0 7 \pm 0 . 8 7$ </td><td rowspan=1 colspan=1> $8 0 . 1 2 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $8 0 . 3 5 \pm 0 . 3 6$ </td><td rowspan=1 colspan=1> $4 3 . 8 1 \pm 0 . 0 6$ </td><td rowspan=1 colspan=1> $7 9 . 2 4 \pm 0 . 6 8$ </td><td rowspan=1 colspan=1> $8 8 . 6 7 \pm 0 . 2 7$ </td><td rowspan=1 colspan=1> $6 9 . 2 8 \pm 0 . 1 1$ </td></tr><tr><td rowspan=1 colspan=1>FedProx</td><td rowspan=1 colspan=1> $8 6 . 2 4 \pm 0 . 2 8$ </td><td rowspan=1 colspan=1> $7 6 . 3 7 \pm 0 . 9 5$ </td><td rowspan=1 colspan=1> $7 7 . 4 1 \pm 0 . 4 5$ </td><td rowspan=1 colspan=1> $7 6 . 2 1 \pm 0 . 9 1$ </td><td rowspan=1 colspan=1> $4 2 . 7 9 \pm 0 . 4 8$ </td><td rowspan=1 colspan=1> $7 8 . 2 7 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $8 7 . 6 1 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $7 0 . 4 5 \pm 0 . 9 2$ </td></tr><tr><td rowspan=1 colspan=1>Ditto FL</td><td rowspan=1 colspan=1> $8 6 . 8 1 \pm 0 . 9 7$ </td><td rowspan=1 colspan=1> $7 6 . 5 9 \pm 0 . 3 9$ </td><td rowspan=1 colspan=1> $7 9 . 0 7 \pm 0 . 5 7$ </td><td rowspan=1 colspan=1> $7 9 . 9 7 \pm 0 . 4 4$ </td><td rowspan=1 colspan=1> $4 3 . 2 9 \pm 0 . 4 2$ </td><td rowspan=1 colspan=1> $7 5 . 6 6 \pm 0 . 0 8$ </td><td rowspan=1 colspan=1> $8 9 . 0 5 \pm 0 . 5 2 $ </td><td rowspan=1 colspan=1> $7 1 . 2 6 \pm 0 . 8 2$ </td></tr><tr><td rowspan=1 colspan=1>FedACS</td><td rowspan=1 colspan=1> $8 9 . 9 4 \pm 0 . 0 1$ </td><td rowspan=1 colspan=1> $7 8 . 2 2 \pm 0 . 3 9$ </td><td rowspan=1 colspan=1> $8 0 . 6 6 \pm 0 . 8 2$ </td><td rowspan=1 colspan=1> $7 8 . 2 6 \pm 0 . 3 7$ </td><td rowspan=1 colspan=1> $4 3 . 9 2 \pm 0 . 9 8$ </td><td rowspan=1 colspan=1> $8 0 . 2 5 \pm 0 . 1 9$ </td><td rowspan=1 colspan=1> ${ \bf 8 9 . 9 6 \pm 0 . 7 1 }$ </td><td rowspan=1 colspan=1> $7 0 . 8 0 \pm 0 . 6 0$ </td></tr><tr><td rowspan=1 colspan=1>LCFed</td><td rowspan=1 colspan=1> $8 7 . 9 8 \pm 0 . 6 1$ </td><td rowspan=1 colspan=1> $7 9 . 0 7 \pm 0 . 7 7$ </td><td rowspan=1 colspan=1> $7 4 . 2 3 \pm 0 . 1 6$ </td><td rowspan=1 colspan=1> $8 1 . 0 2 \pm 0 . 6 7$ </td><td rowspan=1 colspan=1> ${ \bf 4 4 . 2 3 \pm 0 . 8 5 }$ </td><td rowspan=1 colspan=1> $8 0 . 9 7 \pm 0 . 1 2$ </td><td rowspan=1 colspan=1> $8 8 . 6 5 \pm 0 . 8 4$ </td><td rowspan=1 colspan=1> $7 1 . 0 7 \pm 0 . 6 3$ </td></tr><tr><td rowspan=1 colspan=1>FedAFK</td><td rowspan=1 colspan=1> $8 5 . 8 1 \pm 0 . 6 0$ </td><td rowspan=1 colspan=1> $7 8 . 8 0 \pm 0 . 8 2$ </td><td rowspan=1 colspan=1> $8 0 . 0 1 \pm 0 . 7 1$ </td><td rowspan=1 colspan=1> $7 7 . 1 9 \pm 0 . 4 0$ </td><td rowspan=1 colspan=1> $4 4 . 1 2 \pm 0 . 2 4$ </td><td rowspan=1 colspan=1> $7 8 . 7 1 \pm 0 . 6 6$ </td><td rowspan=1 colspan=1> $8 5 . 7 0 \pm 0 . 6 1$ </td><td rowspan=1 colspan=1> $6 6 . 9 0 \pm 0 . 5 6$ </td></tr><tr><td rowspan=2 colspan=1>FedASSAPE-FL</td><td rowspan=2 colspan=1> $8 8 . 1 1 \pm 0 . 2 0 $  ${ \bf 9 0 . 8 3 \pm 0 . 4 9 }$ </td><td rowspan=1 colspan=1> $8 0 . 2 0 \pm 0 . 3 2$ </td><td rowspan=1 colspan=1> $7 9 . 5 9 \pm 0 . 8 1$ </td><td rowspan=1 colspan=1> $7 6 . 6 0 \pm 0 . 1 3$ </td><td rowspan=1 colspan=1> $4 2 . 2 0 \pm 0 . 5 2$ </td><td rowspan=1 colspan=1> $7 7 . 1 5 \pm 0 . 2 8$ </td><td rowspan=1 colspan=1> $8 5 . 6 0 \pm 0 . 2 1$ </td><td rowspan=1 colspan=1> $6 8 . 9 3 \pm 0 . 0 9$ </td></tr><tr><td rowspan=1 colspan=1> ${ \bf 8 1 . 9 4 \pm 0 . 7 7 }$ </td><td rowspan=1 colspan=1> ${ \bf 8 2 . 6 9 \pm 0 . 5 5 }$ </td><td rowspan=1 colspan=1> ${ \bf 8 2 . 0 7 \pm 0 . 9 6 }$ </td><td rowspan=1 colspan=1> $4 4 . 0 1 \pm 0 . 6 3$ </td><td rowspan=1 colspan=1> ${ \bf 8 1 . 9 3 \pm 0 . 6 1 }$ </td><td rowspan=1 colspan=1> $8 9 . 0 1 \pm 0 . 4 4$ </td><td rowspan=1 colspan=1> ${ \bf 7 2 . 3 5 \pm 0 . 7 0 }$ </td></tr></table>

## 6.2 Data Partitions and Performance Metrics

To simulate statistical heterogeneity across clients, we partition datasets using a Dirichlet distribution with concentration parameter $\kappa = 0 . 3$ . This setup ensures non-IID data distributions typical of federated settings. We run SAPE-FL algorithm for 200 communication rounds and use $C = 1 0 0$ clients for synthetic, HAR and image classification tasks. Clients data is split 80/20 into train and test sets. The client participation ratio (r) is set to 70% per round. We set the influence factor δ to 0.5 and the similarity threshold $\bar { S _ { \mathrm { m i n } } } = 0 . \bar { 6 } 5$ . We repeat all our experiments with

Table 2: AUC Scores (%). Comparison of SAPE-FLwith baseline methods. Each cell signifies the mean AUC (standard deviation) computed across all clients at the end of all communication rounds. Bold indicate the best
<table><tr><td>Method</td><td>Synthetic</td><td>HAR</td><td> $\mathbf { A c c G y r o }$ </td><td>CIFAR-10</td><td> $\mathbf { C I F A R - 1 0 0 }$ </td><td>MNIST</td><td>FMNIST</td><td>EMNIST</td></tr><tr><td>Local-only</td><td> $8 2 . 2 1 \pm 0 . 1 9$ </td><td> $7 4 . 2 2 \pm 0 . 7 9$ </td><td> $7 5 . 6 0 \pm 0 . 8 2$ </td><td> $7 8 . 4 0 \pm 0 . 3 1$ </td><td> $5 6 . 2 5 \pm 0 . 6 7$ </td><td> $7 2 . 4 5 \pm 0 . 2 8$ </td><td> $7 9 . 3 9 \pm 0 . 3 8$ </td><td> $7 8 . 0 7 \pm 0 . 2 9$ </td></tr><tr><td>FedAvg</td><td> $8 4 . 5 1 \pm 0 . 6 3$ </td><td> $7 7 . 1 6 \pm 0 . 2 2$ </td><td> $7 8 . 2 4 \pm 0 . 6 7$ </td><td> $8 0 . 2 2 \pm 0 . 4 9$ </td><td> $5 9 . 6 4 \pm 0 . 5 2$ </td><td> $7 4 . 4 3 \pm 0 . 2 5$ </td><td> $8 1 . 0 7 \pm 0 . 5 2$ </td><td> $8 0 . 6 7 \pm 0 . 4 4$ </td></tr><tr><td>PerFedAvg</td><td> $8 6 . 2 4 \pm 0 . 3 7$ </td><td> $7 8 . 2 9 \pm 0 . 0 7$ </td><td> $8 7 . 6 4 \pm 0 . 0 8$ </td><td> $8 2 . 1 6 \pm 0 . 5 7$ </td><td> $6 5 . 6 7 \pm 0 . 8 8$ </td><td> $8 2 . 6 0 \pm 0 . 1 7$ </td><td> $8 3 . 3 9 \pm 0 . 8 7$ </td><td> $8 1 . 2 9 \pm 0 . 9 7$ </td></tr><tr><td>FedProx</td><td> $8 7 . 8 9 \pm 0 . 3 9$ </td><td> $7 6 . 9 2 \pm 0 . 5 0$ </td><td> $8 0 . 5 9 \pm 0 . 4 0$ </td><td> $7 9 . 6 2 \pm 0 . 3 2$ </td><td> $6 6 . 5 9 \pm 0 . 6 7$ </td><td> $8 3 . 3 5 \pm 0 . 1 9$ </td><td> $8 2 . 0 3 \pm 0 . 1 6$ </td><td> $8 3 . 4 5 \pm 0 . 2 1$ </td></tr><tr><td>Ditto FL</td><td> $8 6 . 1 6 \pm 0 . 6 7$ </td><td> $7 6 . 8 1 \pm 0 . 1 6$ </td><td> $8 4 . 2 2 \pm 0 . 4 3$ </td><td> $8 2 . 4 3 \pm 0 . 2 7$ </td><td> $6 6 . 2 9 \pm 0 . 8 0$ </td><td> $8 0 . 3 1 \pm 0 . 4 9$ </td><td> $8 7 . 8 9 \pm 0 . 1 2$ </td><td> $8 4 . 4 8 \pm 0 . 2 6$ </td></tr><tr><td>FedACS</td><td> $8 7 . 9 5 \pm 0 . 5 0$ </td><td> $7 8 . 7 5 \pm 0 . 6 7$ </td><td>87.51 ± 0.71</td><td> $8 1 . 0 8 \pm 0 . 1 1$ </td><td> $6 7 . 7 2 \pm 0 . 5 8$ </td><td> $8 4 . 2 6 \pm 0 . 7 7$ </td><td> $8 7 . 0 9 \pm 0 . 8 7$ </td><td> $8 3 . 1 9 \pm 0 . 1 8$ </td></tr><tr><td>LCFed</td><td> $8 7 . 0 2 \pm 0 . 6 1$ </td><td> ${ \bf 7 9 . 9 7 \pm 0 . 1 3 }$ </td><td> $8 3 . 1 9 \pm 0 . 7 5$ </td><td> $8 2 . 1 0 \pm 0 . 6 5$ </td><td> ${ \bf 7 0 . 0 7 \pm 0 . 6 2 }$ </td><td> $8 4 . 4 9 \pm 0 . 3 3$ </td><td> $8 8 . 0 2 \pm 0 . 0 9$ </td><td> $8 3 . 6 9 \pm 0 . 7 6$ </td></tr><tr><td>FedAFK</td><td> $8 2 . 2 0 \pm 0 . 1 7$ </td><td> $7 2 . 6 4 \pm 0 . 8 0$ </td><td> $8 1 . 2 3 \pm 0 . 5 2$ </td><td> $7 8 . 4 7 \pm 0 . 2 9$ </td><td> $5 9 . 9 2 \pm 0 . 7 7$ </td><td> $8 1 . 3 5 \pm 0 . 7 0$ </td><td> $8 5 . 1 9 \pm 0 . 3 2$ </td><td> $8 4 . 4 4 \pm 0 . 1 7$ </td></tr><tr><td>FedAS</td><td> $8 6 . 4 1 \pm 0 . 8 8$ </td><td> $7 7 . 1 1 \pm 0 . 0 1$   $7 9 . 2 4 \pm 0 . 5 6$ </td><td> $7 6 . 0 9 \pm 0 . 7 7$ </td><td> $8 1 . 0 4 \pm 0 . 9 1$ </td><td> $6 7 . 7 9 \pm 0 . 0 6$ </td><td> $8 2 . 0 2 \pm 0 . 1 1$ </td><td> $8 1 . 2 5 \pm 0 . 3 3$ </td><td> $8 4 . 2 2 \pm 0 . 4 1$ </td></tr><tr><td>SAPE-FL</td><td> ${ \bf 8 8 . 6 4 \pm 0 . 0 4 }$ </td><td></td><td> ${ \bf 8 8 . 6 0 \pm 0 . 5 8 }$ </td><td> ${ \bf 8 4 . 0 4 \pm 0 . 1 4 }$ </td><td> $6 8 . 0 9 \pm 0 . 7 9$ </td><td> ${ \bf 8 6 . 0 9 \pm 0 . 2 5 }$ </td><td> ${ \bf 8 8 . 6 9 \pm 0 . 4 5 }$ </td><td> ${ \bf 8 6 . 2 7 \pm 0 . 5 0 }$ </td></tr></table>

5 random intializations. Model performance is reported using mean test accuracy (±standard deviation) and AUC (±standard deviation) across clients. While SAPE-FL framework introduces additional computation and communication overhead due to pairwise similarity estimation and peer maintenance, this cost can be effectively mitigated in largescale deployments using matrix approximation techniques such as low-rank factorization, which substantially reduce per-client overhead. We refer to Appendix for the detailed experimental setup and discussion.

We now discuss the empirical results of the SAPE-FL framework compared with baselines. Table 1 shows the mean test accuracy (±standard deviation) at the final communication round across all clients and Table 2 reports the corresponding average AUC scores (±standard deviation). The proposed framework consistently outperformed baselines, achieving the highest mean test accuracy in most classification tasks, highlighting its ability to effectively personalize under heterogeneous conditions. While SAPE-FL consistently performed well, methods such as Ditto FL, and LCFed exhibit competitive performance in few cases. Ditto FL’s strong result may be attributed to its dual-model structure that isolates local and global learning paths. LCFed likely benefits from their clustering scheme that aligns well with structured character data. Nonetheless, SAPE-FL demonstrates broad generalization and robust adaptation, outperforming baselines in HAR tasks where personalization is very crucial.

To further analyze and validate the robustness of the SAPE-FL framework, we conducted additional experiments, the results of which are deferred to the Appendix due to space constraints. In particular, we track the evolution of local client models across multiple rounds and demonstrate how local models (with initially poor generalization) benefit from guidance provided by the global and peer-averaged anchors, thereby mitigating negative knowledge transfer. We additionally perform an ablation study to understand the sensitivity of SAPE-FL to hyperparameters such as influence factor δ and the similarity threshold $\bar { S } _ { \operatorname* { m i n } }$ . The ablation study and additional experimental results are in the Appendix.

## 7 Conclusion

In this work, we proposed Similarity-Aware PErsonalized Federated Learning (SAPE-FL), a novel personalization framework designed to address statistical heterogeneity in federated learning. SAPE-FL introduced a dual anchoring mechanism, where each client aligns its model with both the global and a similarity-weighted peer average model, guided by adaptive regularization based on both model parameters similarity and output similarity. Additionally, a similarity-aware aggregation strategy at the server prioritizes coherent client updates, enhancing robustness and personalization. Our results show that SAPE-FL achieves superior performance under heterogeneous settings when compared to the state-of-the-art methods, thereby validating its importance.

## 8 Acknowledgments

This work was supported by the Wellcome Trust [303030/Z/23/Z].

## References

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pages 1273–1282. PMLR, 2017.

Ittai Dayan, Holger Roth, Aoxiao Zhong, Ahmed Harouni, Amilcare Gentili, Anas Abidin, Andrew Liu, Anthony Costa, Bradford Wood, Chien-Sung Tsai, Chih-Hung Wang, Chun-Nan Hsu, C. Lee, Peiying Ruan, Daguang Xu, Dufan Wu, Eddie Huang, Felipe Kitamura, Griffin Lacey, and Quanzheng Li. Federated learning for predicting clinical outcomes in patients with COVID-19. Nature Medicine, 27:1–9, 10 2021. doi:10.1038/s41591-021-01506-3.

Guodong Long, Yue Tan, Jing Jiang, and Chengqi Zhang. Federated learning for open banking. In Federated learning: privacy and incentive. 2020.

Haftay Gebreslasie Abreha, Mohammad Hayajneh, and Mohamed Adel Serhani. Federated learning in edge computing: a systematic survey. Sensors, 22(2):450, 2022.

Stephen Boyd, Neal Parikh, Eric Chu, Borja Peleato, and Jonathan Eckstein. Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers. Foundations and Trends® in Machine Learning, 3(1): 1–122, 2011. ISSN 1935-8237. doi:10.1561/2200000016. URL http://dx.doi.org/10.1561/2200000016.

Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated learning: Challenges, methods, and future directions. IEEE signal processing magazine, 37(3):50–60, 2020a.

Yue Zhao, Meng Li, Liangzhen Lai, Naveen Suda, Damon Civin, and Vikas Chandra. Federated learning with non-iid data. arXiv preprint arXiv:1806.00582, 2018.

Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. Proceedings of Machine learning and systems, 2, 2020b.

Yihan Jiang, Jakub Konecnˇ y, Keith Rush, and Sreeram Kannan. Improving federated learning personalization via \` model agnostic meta learning. arXiv preprint arXiv:1909.12488, 2019.

Khe Chai Sim, Petr Zadrazil, and Françoise Beaufays. An investigation into on-device personalization of end-to-end automatic speech recognition models. arXiv preprint arXiv:1909.06678, 2019.

Tian Li, Shengyuan Hu, Ahmad Beirami, and Virginia Smith. Ditto: Fair and robust federated learning through personalization. In International conference on machine learning, pages 6357–6368. PMLR, 2021a.

Viraj Kulkarni, Milind Kulkarni, and Aniruddha Pant. Survey of personalization techniques for federated learning. In 4th world conference on smart trends in systems, security and sustainability, 2020.

Alireza Fallah, Aryan Mokhtari, and Asuman Ozdaglar. Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach. Advances in Neural Information Processing Systems, 33:3557–3568, 2020.

Zihan Chen, Jundong Li, and Cong Shen. Personalized federated learning with attention-based client selection. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024.

Theodora Nevrataki, Anastasia Iliadou, George Ntolkeras, Ioannis Sfakianakis, Lazaros Lazaridis, George Maraslidis, Nikolaos Asimopoulos, and George F Fragulis. A survey on federated learning applications in healthcare, finance, and data privacy/data security. In AIP Conference Proceedings, volume 2909, 2023.

Sebastian Ruder. An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747, 2016.

Jakub Konecnˇ y, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and Dave Bacon.\` Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016.

Naman Agarwal, Ananda Theertha Suresh, Felix Xinnan X Yu, Sanjiv Kumar, and Brendan McMahan. cpSGD: Communication-efficient and differentially-private distributed SGD. Advances in Neural Information Processing Systems, 31, 2018.

Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. Practical secure aggregation for privacy-preserving machine learning. In Proceedings ofACM SIGSAC Conference on Computer and Communications Security, 2017.

Shiqiang Wang, Tiffany Tuor, Theodoros Salonidis, Kin K Leung, Christian Makaya, Ting He, and Kevin Chan. Adaptive federated learning in resource constrained edge computing systems. IEEE journal on selected areas in communications, 37(6):1205–1221, 2019.

Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua Zhang. On the convergence of fedavg on non-iid data. In Advances in International Conference on Learning Representations, 2020c.

Li Huang, Yifeng Yin, Zeng Fu, Shifa Zhang, Hao Deng, and Dianbo Liu. LoAdaBoost: Loss-based AdaBoost federated machine learning with reduced computational complexity on IID and non-IID intensive care data. PLOS one, 15(4): e0230706, 2020.

Felix Sattler, Klaus-Robert Müller, and Wojciech Samek. Clustered federated learning: Model-agnostic distributed multitask optimization under privacy constraints. IEEE transactions on Neural Networks and Learning Systems, 32 (8):3710–3722, 2020.

Wenlin Chen, Samuel Horváth, and Peter Richtárik. Optimal client sampling for federated learning. Transactions on Machine Learning Research, 2022.

Jianxin Zhao, Yanhao Feng, Xinyu Chang, and Chi Harold Liu. Energy-efficient client selection in federated learning with heterogeneous data on edge. Peer-to-Peer Networking and Applications, 2022.

Sai Qian Zhang, Jieyu Lin, and Qi Zhang. A multi-agent reinforcement learning approach for efficient client selection in federated learning. In AAAI conference on artificial intelligence, 2022.

Wenyu Zhang, Xiumin Wang, Pan Zhou, Weiwei Wu, and Xinglin Zhang. Client selection for federated learning with non-iid data in mobile edge computing. IEEE Access, 9:24462–24474, 2021.

Zixuan Qin, Liu Yang, Qilong Wang, Yahong Han, and Qinghua Hu. Reliable and interpretable personalized federated learning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 20422– 20431, 2023.

Murat Sensoy, Lance Kaplan, and Melih Kandemir. Evidential deep learning to quantify classification uncertainty. Advances in Neural Information Processing Systems, 31, 2018.

Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, and Ameet S Talwalkar. Federated multi-task learning. Advances in Neural Information Processing Systems, 30, 2017.

Luca Corinzia, Ami Beuret, and Joachim M Buhmann. Variational federated multi-task learning. arXiv preprint arXiv:1906.06268, 2019.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pages 1126–1135. PMLR, 2017.

Mikhail Khodak, Maria-Florina F Balcan, and Ameet S Talwalkar. Adaptive gradient-based meta-learning methods. Advances in Neural Information Processing Systems, 32, 2019.

Daniel Peterson, Pallika Kanani, and Virendra J Marathe. Private federated learning with domain adaptation. arXiv preprint arXiv:1912.06733, 2019.

Steven Nowlan and Geoffrey E Hinton. Evaluation of adaptive mixtures of competing experts. In Neural Information Processing Systems, 3, 1990.

Yishay Mansour, Mehryar Mohri, Jae Ro, and Ananda Theertha Suresh. Three approaches for personalization with applications to federated learning. arXiv preprint arXiv:2002.10619, 2020.

Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and Sunav Choudhary. Federated learning with personalization layers. arXiv preprint arXiv:1912.00818, 2019.

Keting Yin and Jiayi Mao. Personalized federated learning with adaptive feature aggregation and knowledge transfer. In 2025 International Joint Conference on Neural Networks (IJCNN), pages 1–8. IEEE, 2025.

Avishek Ghosh, Jichan Chung, Dong Yin, and Kannan Ramchandran. An efficient framework for clustered federated learning. Advances in Neural Information Processing Systems, 33:19586–19597, 2020.

Yuxin Zhang, Haoyu Chen, Zheng Lin, Zhe Chen, and Jin Zhao. Lcfed: An efficient clustered federated learning framework for heterogeneous data. In IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2025.

Jonathan Scott, Hossein Zakerinia, and Christoph H Lampert. Pefll: Personalized federated learning by learning to learn. In International Conference on Learning Representations, 2024.

Hong-You Chen and Wei-Lun Chao. On bridging generic and personalized federated learning for image classification. In International Conference on Learning Representations, 2022.

Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In International conference on machine learning, 2020.

Qinbin Li, Bingsheng He, and Dawn Song. Model-contrastive federated learning. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 10713–10722, 2021b.

Jian Xu, Xinyi Tong, and Shao-Lun Huang. Personalized federated learning with feature alignment and classifier collaboration. In International Conference on Learning Representations, 2023.

Xiyuan Yang, Wenke Huang, and Mang Ye. Fedas: Bridging inconsistency in personalized federated learning. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 11986–11995. IEEE, 2024.

Liam Collins, Hamed Hassani, Aryan Mokhtari, and Sanjay Shakkottai. Fedavg with fine tuning: Local updates lead to representation learning. In Neural Information Processing Systems, 35:10572–10586, 2022.

Canh T Dinh, Nguyen Tran, and Josh Nguyen. Personalized federated learning with moreau envelopes. Advances in Neural Information Processing Systems, 33:21394–21405, 2020.

Yutao Huang, Lingyang Chu, Zirui Zhou, Lanjun Wang, Jiangchuan Liu, Jian Pei, and Yong Zhang. Personalized cross-silo federated learning on non-iid data. In Proceedings of the AAAI conference on artificial intelligence, volume 35, pages 7865–7873, 2021.

F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay. Scikit-learn: Machine learning in Python. Journal ofMachine Learning Research, 12, 2011.

Jorge Reyes Ortiz, Davide Anguita, Alessandro Ghio, Luca Oneto, and Xavier Parra. Human Activity Recognition Using Smartphones. UCI Machine Learning Repository, 2013.

AlSahly Abdullah. Accelerometer Gyro Mobile Phone. UCI Machine Learning Repository, 2022.

## A Appendix

We use lower case bold fonts v for vectors and $v _ { i }$ for each element in $\mathbf { v } . \ \mathbf { v } ^ { \mathsf { T } }$ is the transpose of $\mathbf { v } .$ We use upper case bold fonts (and bold greek symbols) M for matrices and $M _ { i j }$ for each element in $\mathbf { M } . \mathbf { \Omega } \rvert \dot { \mathbf { \Omega } } \rvert$ is the cardinality of the set. abs(·) is the absolute value. $\mathbf { \tilde { N } } _ { n } = \{ 1 , 2 , \cdots , n \}$ . R for Reals. X is a index set and $\mathbf { x } \in \dot { \mathcal { X } } . \mathbf { \rho } \langle \cdot \rangle$ for the dot product and $\| \cdot \|$ for the length of the vector.

## A.1 Convergence Analysis of SAPE-FL

In this section we provide the proofs of theorems provided in the main manuscript. Additionally, we also discuss the convergence of our similarity-weighted global objective under both convex and non-convex settings .

## A.1.1 Proof of Theorem 1

Proof. At each round of SAPE-FL algorithm, the server computes the new global model using $\operatorname { E q . }$ (12). Formally, we can interpret this update as an approximate gradient (SGD) step on ${ \mathcal { L } } .$ When each client performs approximate local SGD starting from $\theta _ { G } ^ { t } ,$ it is well established in the federated optimization literature McMahan et al. [2017], Li et al. [2020b], Huang et al. [2021], Chen et al. [2024] that the resulting model update $\Delta _ { c } ^ { t }$ aligns with the negative local gradient $- \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } )$ . In our method, the server aggregates these updates using similarity-based weights $\mathfrak { s } _ { c } ^ { t } ,$ , yielding an update direction that is more representative of aligned clients. At server level, the client updates are aggregated as:

$$
\Delta _ { G } ^ { t } = \sum _ { c } w _ { c } ^ { t } \Delta _ { c } ^ { t } , \quad \mathrm { w h e r e } w _ { c } ^ { t } = \frac { \mathfrak { s } _ { c } ^ { t } } { \sum _ { j } \mathfrak { s } _ { j } ^ { t } } 
$$

and $j$ is the index for all participating clients. Therefore the inner product with the true global gradient is defined as:

$$
\langle \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) , \Delta _ { G } ^ { t } \rangle \ = \sum _ { c } w _ { c } ^ { t } \langle \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) , \Delta _ { c } ^ { t } \rangle
$$

If each client update can be approximated as $\Delta _ { c } ^ { t } \approx - \eta \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } )$ , then the inner product can be written as:

$$
\langle \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) , \Delta _ { c } ^ { t } \rangle = - \langle \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) , \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } ) \rangle
$$

Therefore,

$$
\sum _ { c } w _ { c } ^ { t } \langle \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) , \Delta _ { c } ^ { t } \rangle \approx - \sum _ { c } w _ { c } ^ { t } \| \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } ) \| ^ { 2 }
$$

In the above equation, if the model updates are closer to the global gradient then

$$
\sum _ { c } { w _ { c } ^ { t } \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) ^ { \top } \Delta _ { c } ^ { t } \approx - \rho \| \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } ) \| ^ { 2 } }
$$

where $\rho > 0$ quantifies alignment and reliability of updates. Therefore by the aggregated update as a stochastic descent direction on the true global objective $\mathcal { L } ( \boldsymbol { \theta } )$ , we obtain the following condition for descent:

$$
\sum _ { c } \frac { \mathfrak { s } _ { c } ^ { t } } { \sum _ { j } \mathfrak { s } _ { j } ^ { t } } \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } ) ^ { \top } \Delta _ { c } ^ { t } \leq - \rho \| \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) \| ^ { 2 } ,
$$

which serves as a key step in proving one-round improvement.

Each $\Delta _ { c } ^ { t }$ in the global model update equation Eq. (12) can be seen as the result of running stochastic gradient descent on $\mathcal { L } _ { c } .$ . For small learning rates, $\Delta _ { c } ^ { t } \overset { \cdot } { \approx } - \eta \nabla \mathcal { L } _ { c } \overset { \cdot } { ( } \theta _ { G } ^ { t } )$ . If there were no weighting and full participation from all the clients $\mathrm { i } . \mathrm { e } . , r = 1$ , a traditional aggregation design (FedAvg McMahan et al. [2017]) would produce:

$$
\theta _ { G } ^ { t + 1 } \approx \theta _ { G } ^ { t } - \eta \frac { 1 } { | C | } \sum _ { c \in C } \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } )
$$

Our algorithm instead uses a weighted sum $\frac { \mathfrak { s } _ { c } ^ { t } } { \sum _ { i } \mathfrak { s } _ { i } ^ { t } }$ as the weight for client $c \mathbf { \hat { s } }$ gradient. If the similarity weights are aligned with the importance of each client’s data or the reliability of its gradient, the weighted sum is still an unbiased or at least descent-biased estimator of the true gradient. In realistic cases, $\mathfrak { s } _ { c } ^ { t }$ upweights clients whose gradients are in agreement with others, effectively reducing gradient “variance” or conflicting directions. The aggregated update $\Delta _ { G } ^ { t }$ is therefore biased toward the true global gradient. Under the L-Lipschitz smoothness assumption of $\mathcal { L }$ and Taylor expansion (See Eq. (8) provided in the proof sketch of Theorem 4 of Li et al. [2020b]), we have:

$$
\mathcal { L } ( \theta _ { G } ^ { t + 1 } ) \approx \mathcal { L } ( \theta _ { G } ^ { t } ) + \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) ^ { \top } ( \theta _ { G } ^ { t + 1 } - \theta _ { G } ^ { t } ) + \frac { L } { 2 } | \theta _ { G } ^ { t + 1 } - \theta _ { G } ^ { t } | ^ { 2 }
$$

and by definition of the global update,

$$
\theta _ { G } ^ { t + 1 } - \theta _ { G } ^ { t } = \sum _ { c } \frac { \mathfrak { s } _ { c } ^ { t } } { \sum _ { j } \mathfrak { s } _ { j } ^ { t } } \Delta _ { c } ^ { t } .
$$

Substituting for $\theta _ { G } ^ { t + 1 } - \theta _ { G } ^ { t }$ , the linear term becomes

$$
\nabla \mathcal { L } ( \boldsymbol { \theta } _ { G } ^ { t } ) ^ { \top } ( \boldsymbol { \theta } _ { G } ^ { t + 1 } - \boldsymbol { \theta } _ { G } ^ { t } ) = \sum _ { c } \frac { \mathfrak { s } _ { c } ^ { t } } { \sum _ { i } \mathfrak { s } _ { i } ^ { t } } \nabla \mathcal { L } ( \boldsymbol { \theta } _ { G } ^ { t } ) ^ { \top } \Delta _ { c } ^ { t }
$$

Since each $\Delta _ { c } ^ { t }$ was a descent direction on $\mathcal { L } _ { c }$ and aligned (by regularization) with $\theta _ { G } ^ { t }$ , this sum is negative and bounded away from zero proportionally to $\| \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) \| ^ { 2 }$ . Thus, $\mathcal { L } ( \theta _ { G } ^ { t + 1 } ) < \mathcal { L } ( \theta _ { G } ^ { t } )$ for each round (at least in expectation), so the global model improves. Further, taking expectation over client sampling (assumed uniform for ratio r):

$$
\begin{array} { r l } & { \mathbb { E } [ \mathcal { L } ( \theta _ { G } ^ { t + 1 } ) ] \leq \mathcal { L } ( \theta _ { G } ^ { t } ) + \mathbb { E } \left[ \displaystyle \sum _ { c } \frac { \mathsf { s } _ { c } ^ { t } } { \sum _ { j } \mathsf { s } _ { j } ^ { t } } \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) ^ { \top } \Delta _ { c } ^ { t } \right] \cdot \cdot \cdot } \\ & { \qquad \cdot \cdot + \displaystyle \frac { L } { 2 } \mathbb { E } [ \| \theta _ { G } ^ { t + 1 } - \theta _ { G } ^ { t } \| ^ { 2 } ] } \\ & { \qquad \leq \mathcal { L } ( \theta _ { G } ^ { t } ) - \rho \| \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) \| ^ { 2 } + \displaystyle \frac { L } { 2 } \mathbb { E } [ \| \theta _ { G } ^ { t + 1 } - \theta _ { G } ^ { t } \| ^ { 2 } ] } \end{array}
$$

On the other hand, the peer-regularization term in the local objective $\mathrm { i } . \mathrm { e } . , \lVert \theta _ { c } - { \bar { \theta } } _ { c } ^ { t } \rVert ^ { 2 }$ term weighted by $S _ { c } ^ { P }$ works like an anchor to a neighborhood of similar models. This is an attention mechanism that encourages each client’s final model $\theta _ { c } ^ { t + 1 }$ to not deviate too far from the average of models deemed similar i.e. $S _ { c j } ^ { t } > S _ { \operatorname* { m i n } }$ . As a result, such clients are pulled towards the cluster of more "similar" clients, preventing any single client from making an extreme update. This further bounds the heterogeneity of updates that the server aggregates. □

## A.1.2 Proof of Lemma 1

Proof. Under the similarity assumption on clients and peers, each local model $\theta _ { c } ^ { t }$ (or peer model $\theta _ { i } ^ { t } )$ can be viewed as an independent unbiased estimate of the same underlying mean parameter. The peer-averaged model is defined as:

$$
\bar { \theta } _ { c } ^ { t } = \frac { 1 } { \sum _ { j \in C _ { k } } S _ { c j } } \sum _ { j \in C _ { k } } S _ { c j } \theta _ { j } ^ { t } .
$$

Taking expectation:

$$
\mathbb { E } [ \bar { \theta } _ { c } ^ { t } ] = \frac { 1 } { \sum _ { j \in C _ { k } } S _ { c j } } \sum _ { j \in C _ { k } } S _ { c j } \mathbb { E } [ \theta _ { j } ^ { t } ]
$$

Since each $\theta _ { i } ^ { t }$ has the same covariance Σ and $\hat { \theta } _ { c } ^ { t }$ is the weighted average of the $| C _ { k } |$ independent estimates (with weights $S _ { c j } )$ , it follows that:

$$
\begin{array} { l } { \displaystyle \mathrm { C o v } ( \bar { \theta } _ { c } ^ { t } ) = \left( \frac { 1 } { \sum _ { j \in C _ { k } } S _ { c j } } \right) ^ { 2 } \sum _ { j \in C _ { k } } S _ { c j } ^ { 2 } \mathrm { C o v } ( \theta _ { j } ^ { t } ) } \\ { = \displaystyle \frac { \sum _ { j \in C _ { k } } S _ { c j } ^ { 2 } } { \left( \sum _ { j \in C _ { k } } S _ { c j } \right) ^ { 2 } } \Sigma } \end{array}\tag{13}
$$

Let the error rate of a single client be $v = \mathrm { T r } ( \Sigma )$ . Therefore, the error rate of the peer-averaged model (as the trace of its covariance) is given by:

$$
\bar { \upsilon } = \mathrm { T r } \left( \mathrm { C o v } ( \bar { \theta } _ { c } ^ { t } ) \right) = \frac { \sum _ { j \in C _ { k } } S _ { c j } ^ { 2 } } { \left( \sum _ { j \in C _ { k } } S _ { c j } \right) ^ { 2 } } \upsilon .
$$

This shows that the peer-averaged model has an error rate that is reduced by a factor of

$$
\frac { \sum _ { j \in C _ { k } } S _ { c j } ^ { 2 } } { \left( \sum _ { j \in C _ { k } } S _ { c j } \right) ^ { 2 } }\tag{14}
$$

Now for simplicity, let the similarity scores used in the peer-averaging step are uniform, the variance-reduction factor simplifies to the well-known result from standard averaging. We consider two common forms of uniform weighting. First, if all similarity weights are equal and unnormalized, i.e., $S _ { c j } = 1$ for all $j \in C _ { k }$ , then:

$$
{ \frac { \sum _ { j \in C _ { k } } S _ { c j } ^ { 2 } } { \left( \sum _ { j \in C _ { k } } S _ { c j } \right) ^ { 2 } } } = { \frac { | C _ { k } | \cdot 1 ^ { 2 } } { \left( | C _ { k } | \cdot 1 \right) ^ { 2 } } } = { \frac { | C _ { k } | } { | C _ { k } | ^ { 2 } } } = { \frac { 1 } { | C _ { k } | } } .
$$

Second, if the similarity scores are normalized to sum to 1 $( \mathrm { i . e . , } S _ { c j } = 1 / | C _ { k } |$ for all $j \in C _ { k } )$ , the same simplification holds:

$$
{ \frac { \sum _ { j \in C _ { k } } S _ { c j } ^ { 2 } } { \left( \sum _ { j \in C _ { k } } S _ { c j } \right) ^ { 2 } } } = { \frac { | C _ { k } | \cdot ( 1 / | C _ { k } | ) ^ { 2 } } { \left( \sum _ { j \in C _ { k } } 1 / | C _ { k } | \right) ^ { 2 } } } = | C _ { k } | \cdot { \frac { 1 } { | C _ { k } | ^ { 2 } } } = { \frac { 1 } { | C _ { k } | } }
$$

Hence, regardless of whether uniform weights are applied directly or normalized to sum to one, the variance-reduction factor mentioned in Eq. (14) collapses to $\bar { 1 } / | C _ { k } |$ , matching the classical reduction factor obtained from averaging $| C _ { k } |$ i.i.d. estimators. This highlights that the similarity-weighted aggregation generalizes the standard averaging procedure, and reduces to it under uniform peer similarity.

With the above simplification, the peer-averaged model is then defined as:

$$
\bar { \theta } _ { c } ^ { t } = \frac { 1 } { | C _ { k } | } \sum _ { j \in C _ { k } } \theta _ { j } ^ { t } .\tag{15}
$$

Taking expectation,

$$
\mathbb { E } [ \bar { \theta } _ { c } ^ { t } ] = \frac { 1 } { \lvert C _ { k } \rvert } \sum _ { j \in C _ { k } } \mathbb { E } [ \theta _ { j } ^ { t } ] = \mathbb { E } [ \theta _ { j } ^ { t } ] ,
$$

so $\bar { \theta } _ { c } ^ { t }$ shares the same mean as each individual peer model.

Since every $\theta _ { j } ^ { t }$ has covariance Σ and $\bar { \theta } _ { c } ^ { t }$ is the average of $| C _ { k } |$ independent estimates,

$$
\mathrm { C o v } ( \bar { \theta } _ { c } ^ { t } ) = \frac { 1 } { | C _ { k } | ^ { 2 } } \sum _ { j \in C _ { k } } \mathrm { C o v } ( \theta _ { j } ^ { t } ) = \frac { 1 } { | C _ { k } | ^ { 2 } } \cdot | C _ { k } | \cdot \Sigma = \frac { 1 } { | C _ { k } | } \Sigma\tag{16}
$$

Subsequently, the variance of the peer-averaged model is now given as:

$$
\bar { v } = \mathrm { T r } \bigl ( \mathrm { C o v } ( \bar { \theta } _ { c } ^ { t } ) \bigr ) = \frac { 1 } { | C _ { k } | } v .
$$

Hence the peer-averaged model reduces the estimation-error variance by a factor of $1 / | C _ { k } |$ , reflecting the classical variance-reduction effect of averaging independent, similarly distributed estimates. In other words, peer averaging reduces the estimation error variance, a well known benefit of using an ensemble of models.

Furthermore, by Jensen’s inequality, if each client loss $\mathcal { L } _ { c } ( \theta )$ is convex, then

$$
\mathcal { L } _ { c } ( \bar { \theta } _ { c } ^ { t } ) \leq \frac { 1 } { | C _ { k } | } \sum _ { j \in C _ { k } } \mathcal { L } _ { c } ( \theta _ { j } ^ { t } ) ,
$$

meaning the peer-averaged model achieves no worse (often better) loss on client c than the average of individual peer losses. This justifies that $\bar { \theta } _ { c } ^ { t }$ provides an improved baseline for personalization. Empirically and theoretically, clustering similar clients and averaging their models yields a model closer to the true task model for that cluster. Ghosh et al. [2020] prove that when clients are partitioned into true clusters, a clustered FL algorithm can attain a near-optimal error rate matching the minimax optimal error (analogous to pooling cluster’s data and learned centrally) up to logarithmic factors Ghosh et al. [2020]. This suggests $\bar { \theta } _ { c } ^ { t }$ quickly converges to the optimal cluster model with high accuracy. For linear models, one can show an exponential convergence of cluster models to the true optima with error approximately halving each iteration and in general strongly-convex settings the cluster-personalized models achieve the statistically optimal error rate. The peer averaging step thus improves the estimate of the local optimum by leveraging more data from similar clients, while filtering out dissimilar client’s drift as we set $S _ { c j } = 0$ for those peers. □

Peer-Average Variance Reduction under Non-Convexity While the above lemma is framed under the assumption of convex objective and a unique optimal solution (same mean parameter), its core result showing the variance reduction via peer averaging remains valid in non-convex settings under milder assumptions. Specifically, the key steps in the proof rely on the properties of expectation and covariance, not on the convexity of the underlying loss function or the existence of a unique minimizer. That is, even in non-convex scenarios where the true optimizer may not exist or may not be unique, we can still view each $\theta _ { j } ^ { t }$ as a stochastic estimate of a latent central tendency for a cluster of similar peers.

In this setting, the peer-averaged model in Eq. (15) remains a well-defined random variable, and its variance is still governed by the same formula for the variance of independent estimators mentioned in Eq. (16). This results in a reduced total variance:

$$
\mathrm { T r } \left( \mathrm { C o v } ( \bar { \theta } _ { c } ^ { t } ) \right) = \frac { 1 } { | C _ { k } | ^ { 2 } } \sum _ { j \in C _ { k } } \mathrm { T r } ( \Sigma _ { j } ) \leq \frac { 1 } { | C _ { k } | } \operatorname* { m a x } _ { j } \mathrm { T r } ( \Sigma _ { j } ) .
$$

Therefore, the statistical benefit of peer averaging holds irrespective of whether the estimators $\theta _ { j } ^ { t }$ converge to a true minimum or just to a representative region in the non-convex search space. Moreover, non-convex FL literature $( \mathrm { e . g . }$ Karimireddy et al. [2020], Li et al. [2020b]) commonly leverages similar averaging steps without requiring uniqueness of the optimum. The averaging mechanism still acts as a "variance smoother" and helping guide the optimization toward flatter regions or stationary points with better generalization. Hence, the lemma remains applicable, interpreted as a statistical variance reduction tool rather than a strict estimator of a unique minimizer.

## A.1.3 Proof of Theorem 2

Each client update performs a proximal step over a smooth, composite objective anchored to both the global and peer-averaged models. With bounded similarity-weighted averaging and fixed regularization, the joint model converges to a critical point efficiently. For the sake of completeness, we define the joint objective function across all clients $c \in C$ is given by:

$$
\begin{array} { r } { \mathcal { L } ( \boldsymbol { \theta } _ { G } , \{ \boldsymbol { \theta } _ { c } \} ) = \displaystyle \sum _ { c } \mathcal { L } _ { c } ( \boldsymbol { \theta } _ { c } ) + \displaystyle \sum _ { c } S _ { c } ^ { P } ( \boldsymbol { \theta } _ { c } , \bar { \boldsymbol { \theta } } _ { c } ^ { t } ) + \cdot \cdot \cdot } \\ { \cdot \cdot \cdot \displaystyle \sum _ { c } S _ { c } ^ { G } ( \boldsymbol { \theta } _ { c } , \boldsymbol { \theta } _ { G } ^ { t } ) } \end{array}
$$

The averaging of peer models is performed using a weighted combination of those satisfying the condition $s _ { i j } > S _ { \operatorname* { m i n } } .$ which ensures that each $\hat { \theta } _ { c } ^ { t }$ is closer to clients with high similarity. The update direction benefits from both model similarity and output similarity, accelerating consensus within high-similarity groups. Under smoothness, L decreases each iteration, and in convex cases, it converges to a global minimum.

As per Theorem 1, the global model $\theta _ { G } ^ { t }$ also converges through similarity-weighted aggregation of updates, yielding a stable reference for collaboration across all clients. Alongside restricting the client model $\theta _ { c } ^ { t }$ to drift too much from such global reference model $\theta _ { G } ^ { t } ,$ , the structure of $\mathcal { L }$ also enforces an implicit clustering effect where the clients with strong mutual similarity weights contribute more significantly to each other’s peer-averaged models. The Laplacian-like regularization promotes consensus in peer neighborhoods. This results in clients converging also toward a shared local optimum within similarity-based groups. Over iterations, the contribution of highly similar peers becomes more influential, driving each $\bar { \theta } _ { c } ^ { t }$ toward a cluster-wide personalized optimum.

Each client’s local update is a proximal gradient step on the objective L. At each round t, clients perform a proximal step:

$$
\boldsymbol { \theta } _ { c } ^ { t + 1 } = \boldsymbol { \theta } _ { c } ^ { t } - \boldsymbol { \eta } \left( \nabla \mathcal { L } _ { c } ( \boldsymbol { \theta } _ { c } ^ { t } ) + 2 S _ { c } ^ { G } ( \boldsymbol { \theta } _ { c } ^ { t } - \boldsymbol { \theta } _ { G } ^ { t } ) + 2 S _ { c } ^ { P } ( \boldsymbol { \theta } _ { c } ^ { t } - \bar { \boldsymbol { \theta } } _ { c } ^ { t } ) \right) ,
$$

where $\eta$ is the learning rate.

For brevity, we define,

$$
\mathfrak { g } _ { c } ^ { t } = \nabla \mathcal { L } _ { c } ( \theta _ { c } ^ { t } ) + S _ { c } ^ { G } ( \theta _ { c } ^ { t } - \theta _ { G } ^ { t } ) + S _ { c } ^ { P } ( \theta _ { c } ^ { t } - \bar { \theta } _ { c } ^ { t } )
$$

Under Assumption 1 and as per the definitions provided in Theorem 2 (Appendix A.5) of T Dinh et al. [2020], we have,

$$
\mathcal { L } _ { c } ( \boldsymbol { \theta } _ { c } ^ { t + 1 } ) = \mathcal { L } _ { c } ( \boldsymbol { \theta } _ { c } ^ { t } ) - \eta \| \mathbf { g } _ { c } ^ { t } \| ^ { 2 } + \frac { L \eta ^ { 2 } } { 2 } \| \mathbf { g } _ { c } ^ { t } \| ^ { 2 }
$$

Using this we write the descent in the objective as:

$$
\begin{array} { r l r } & { } & { \displaystyle \mathcal { L } ( \theta _ { G } ^ { t } , \{ \theta _ { c } ^ { t + 1 } \} ) - \mathcal { L } ( \theta _ { G } ^ { t } , \{ \theta _ { c } ^ { t } \} ) \leq - \eta \sum _ { c } \left\| { \mathfrak { g } } _ { c } ^ { t } \right\| ^ { 2 } + \cdot \cdot \cdot } \\ & { } & { \displaystyle \qquad \cdot \cdot \frac { L \eta ^ { 2 } } { 2 } \sum _ { c } \left\| { \mathfrak { g } } _ { c } ^ { t } \right\| ^ { 2 } } \end{array}
$$

Further,

$$
\mathcal { L } ( \theta _ { G } ^ { t } , \{ \theta _ { c } ^ { t + 1 } \} ) \leq \mathcal { L } ( \theta _ { G } ^ { t } , \{ \theta _ { c } ^ { t } \} ) - \eta \left( 1 - \frac { L \eta } { 2 } \right) \left. \mathfrak { g } _ { c } ^ { t } \right. ^ { 2 }
$$

In the above equation ensuring $\begin{array} { r } { \left( 1 - \frac { L \eta } { 2 } \right) > 0 } \end{array}$ and picking small learning rate $\begin{array} { r } { \eta \le \frac { 1 } { 2 L } } \end{array}$ , we obtain monotonic decrease:

$$
\mathcal { L } ( \theta _ { G } ^ { t + 1 } , \{ \theta _ { c } ^ { t + 1 } \} ) \leq \mathcal { L } ( \theta _ { G } ^ { t } , \{ \theta _ { c } ^ { t } \} ) - \frac { \eta } { 2 } \sum _ { c } \| \mathfrak { g } _ { c } ^ { t } \| ^ { 2 } .
$$

If $\theta _ { G } ^ { 0 }$ and $\theta _ { c } ^ { 0 }$ are the initial models we started with, then summing over T rounds gives:

$$
\mathcal { L } ( \theta _ { G } ^ { T } , \{ \theta _ { c } ^ { T } \} ) \leq \mathcal { L } ( \theta _ { G } ^ { 0 } , \{ \theta _ { c } ^ { 0 } \} ) - \frac { \eta } { 2 } \sum _ { t = 1 } ^ { T } \sum _ { c } \| \mathbf { g } _ { c } ^ { t } \| ^ { 2 } .
$$

By rearranging, we get,

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { c } \| \nabla _ { \theta _ { c } } \mathcal { L } ( \theta _ { c } ^ { t } ) \| ^ { 2 } \leq \frac { 2 ( \mathcal { L } ( \theta _ { G } ^ { 0 } , \{ \theta _ { c } ^ { 0 } \} ) - \mathcal { L } ( \theta _ { G } ^ { T } , \{ \theta _ { c } ^ { T } \} ) } { \eta T } ,
$$

Therefore, for a target accuracy ϵ,

$$
\frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } [ \| \nabla \mathcal { L } ( \theta _ { c } ^ { t } ) \| ^ { 2 } ] \le \frac { \Sigma } { | C _ { k } | \epsilon ^ { 2 } } .
$$

In standard FL or SGD without peer-averaging, the variance of the gradient estimate is assumed constant (or just bounded by Σ), which leads to the usual bound:

$$
T = \mathcal { O } \left( \frac { \Sigma } { \epsilon ^ { 2 } } \right)
$$

However, because of peer-averaging in our proposed framework and thus the variance reduction property of $\bar { \theta } _ { c } ^ { t }$ (see proof of Lemma 1), we observe a tightened convergence rate. Specifically, let Σ denote the covariance of a single local model $\theta _ { c } ^ { t } .$ . Then the peer-averaged estimator yields:

$$
\mathrm { C o v } ( \bar { \theta } _ { c } ^ { t } ) \leq \frac { 1 } { \sum _ { j } S _ { c j } ^ { t } } \Sigma \leq \frac { 1 } { | C _ { k } | } \Sigma ,
$$

where $| C _ { k } |$ is the number of similar peers. As per Lemma 1, the reduced variance is given as:

$$
\mathrm { V a r } ( \bar { \theta } _ { c } ^ { t } ) = \frac { 1 } { | C _ { k } | } \Sigma
$$

Practically $| C _ { k } | \gg 1$ and $\mathcal { O } ( \Sigma ) = 1$ , then the resulting bounds are given as:

$$
T = { \mathcal { O } } \left( { \frac { \mathrm { V a r } ( { \bar { \theta } } ) } { \epsilon ^ { 2 } } } \right) \leq { \mathcal { O } } \left( { \frac { \Sigma } { | C _ { k } | \cdot \epsilon ^ { 2 } } } \right) < { \mathcal { O } } \left( { \frac { 1 } { \epsilon ^ { 2 } } } \right)
$$

This tighter bound implies reduced variance in client updates, leading to a faster decrease in the global objective established in Theorem 1:

$$
\mathbb { E } [ \mathcal { L } ( \theta _ { G } ^ { t + 1 } ) ] \leq \mathcal { L } ( \theta _ { G } ^ { t } ) - \rho \| \nabla \mathcal { L } ( \theta _ { G } ^ { t } ) \| ^ { 2 } + \frac { L } { 2 } \mathbb { E } [ \| \theta _ { G } ^ { t + 1 } - \theta _ { G } ^ { t } \| ^ { 2 } ] .
$$

Thus, our method requires fewer rounds to reach an ϵ-stationary solution demonstrating that peer-regularized personal ization yields faster convergence due to statistical averaging across similar peers.

Theorem 3 (Global Convergence under Strong µ-Convexity). Suppose a client’s loss function $\mathcal { L } _ { c } ( \theta )$ is µ-strongly convex and L-smooth (i.e. has L-Lipschitz gradients) as per Assumptions 1-3. The global model is updated by similarity-weighted aggregation as per Eq. (12). Then there exists a unique global minimizer $\theta _ { G } ^ { \star }$ that minimizes the similarity-weighted global objective and in particular achieves linear convergence: for a sufficiently small constant step size,

$$
\theta _ { G } ^ { t } - \theta ^ { \star } \leq ( 1 - \mu \eta ) ^ { t } \| \theta _ { G } ^ { 0 } - \theta _ { G } ^ { \star } \| ^ { 2 }
$$

Equivalently, the optimality gap decays: $F ( \theta _ { G } ^ { t } ) - F ( \theta _ { G } ^ { \star } ) = \mathcal { O } ( ( 1 - \mu \eta ) ^ { t } )$

Proof. Let us assume full client participation i.e. all C clients are selected per round and similarity scores $s _ { c } ^ { t }$ is non-negative and bounded as $\begin{array} { r } { \sum _ { c } s _ { c } ^ { t } = \bar { 1 } } \end{array}$ for each round t. For brevity, let us define the similarity-weighted global objective as:

$$
F ( \theta ) : = \sum _ { c } s _ { c } \ : { \mathcal { L } } _ { c } ( \theta )
$$

for appropriate fixed weights $s _ { c } .$ The unique minimizer $\theta _ { G } ^ { \star }$ is defined as:

$$
\theta _ { G } ^ { \star } : = \arg \operatorname* { m i n } _ { \theta } F ( \theta )
$$

At each round, the server computes the new global model (using Eq. (12)) and formally, we can interpret this update as a gradient descent step on $F ,$ , up to the influence of the peer-regularization. In particular, excluding the regularizers, we have on each round:

$$
\theta _ { G } ^ { t + 1 } = \theta _ { G } ^ { t } - \eta \sum _ { c } s _ { c } ^ { t } \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } ) \approx \theta _ { G } ^ { t } - \eta \nabla F ( \theta _ { G } ^ { t } )
$$

because $\begin{array} { r } { \nabla F ( \theta ) = \sum _ { c } s _ { c } \nabla \mathcal { L } _ { c } ( \theta ) } \end{array}$ . If $s _ { c } ^ { t }$ varies with $t ,$ each step still follows the aggregated gradient of the objective;   
for simplicity we retain $s _ { c } ^ { t }$ constant, since the analysis extends similarly as long as weights are bounded.

The client peer-regularized local loss can be written as

$$
\tilde { \mathcal { L } } _ { c } ^ { t } ( \theta ) = \mathcal { L } _ { c } ( \theta ) + R _ { c } ^ { t } ( \theta ) ,\tag{17}
$$

where $R _ { c } ^ { t }$ is a regularizer term that penalizes discrepancy from peer models (via $\hat { \theta } _ { c } ^ { t }$ defined in Eq. (8)) or from the global model $\theta _ { G } ^ { t }$ . The regularizer $R _ { c } ^ { t }$ only makes $\tilde { \mathcal { L } } _ { c } ^ { t }$ more strongly convex. For example, in the convergence analysis of FedProx Li et al. [2020b] $\tilde { \mathcal { L } } _ { c } ^ { t } ( \theta )$ mentioned in Eq. (17) is set as:

$$
R _ { c } ^ { t } ( \theta ) = \frac { \lambda } { 2 } | \theta - \theta _ { G } ^ { t } | ^ { 2 } ,
$$

whose gradient at $\theta = \theta _ { G } ^ { t }$ is 0. However, in SAPE-FL , regularizer $R _ { c } ^ { t } ( \theta )$ does not vanish at $\theta = \theta _ { G } ^ { t }$ due to the peer term $( \bar { \theta } _ { c } ^ { t } )$ , we can control its influence under reasonable assumptions so that it is small enough to preserve descent guarantees in the global convergence analysis. We argue that the total regularizer gradient $\nabla R _ { c } ^ { \breve { t } } ( \theta )$ can still be treated as a bounded perturbation to the local optimization path. Specifically, the gradient remains small in magnitude if $\bar { \theta } _ { i } ^ { t }$ is close to $\theta _ { G } ^ { t }$ and similarity-based filtering ensures that $\bar { \theta } _ { i } ^ { t }$ is an average over similar clients with relatively aligned models.

In SAPE-FL , at server level, the client updates are aggregated as:

$$
\Delta _ { G } ^ { t } = \sum _ { c } w _ { c } ^ { t } \Delta _ { c } ^ { t } , \quad \mathrm { w h e r e } w _ { c } ^ { t } = \frac { \mathfrak { s } _ { c } ^ { t } } { \sum _ { j } \mathfrak { s } _ { j } ^ { t } } 
$$

and $j$ is the index for all participating clients.

Therefore, at the start of each round t i.e, when $\theta = \theta _ { G } ^ { t }$ , the update direction

$$
\Delta _ { c } ^ { t } = - \eta , \nabla \tilde { \mathcal { L } } _ { c } ^ { t } ( \theta _ { G } ^ { t } )
$$

can be approximated as

$$
\Delta _ { c } ^ { t } \approx - \eta , \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } )
$$

since $\nabla R _ { c } ^ { t } ( \theta _ { G } ^ { t } ) = 0$ or is negligibly small. In other words, the similarity weighting and peer regularization do not hinder the descent direction; they effectively keep client updates closer to the true global gradient, mitigating drift. We can thus analyze the global iteration as a gradient descent on $F ( \theta )$

Given that $F$ is µ-strongly convex and has Lipschitz gradient L, it is well-known that gradient descent enjoys a linear convergence rate Li et al. [2020b]. Fore brevity, Let $\begin{array} { r } { g ^ { \overline { { t } } } : = \nabla F ( \theta _ { G } ^ { t } ) = \sum _ { c } s _ { c } \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } ) ^ { \vee } } \end{array}$ be the true global gradient. By L-smoothness assumption on $F$ and the taylor expansion, we have:

$$
F ( { \boldsymbol { \theta } } _ { G } ^ { t + 1 } \leq F ( { \boldsymbol { \theta } } _ { G } ^ { t } ) + \langle \nabla F ( { \boldsymbol { \theta } } _ { G } ^ { t } ) , { \boldsymbol { \theta } } _ { G } ^ { t + 1 } - { \boldsymbol { \theta } } _ { G } ^ { t } \rangle + \frac { L } { 2 } \| { \boldsymbol { \theta } } _ { G } ^ { t + 1 } - { \boldsymbol { \theta } } _ { G } ^ { t } \| ^ { 2 }\tag{18}
$$

Substituting $\theta _ { G } ^ { t + 1 } = \theta _ { G } ^ { t } - \eta g ^ { t }$ , we get

$$
F ( { \theta } _ { G } ^ { t + 1 } ) \leq F ( { \theta } _ { G } ^ { t } ) - \eta \| g ^ { t } \| ^ { 2 } + \frac { L \eta ^ { 2 } } { 2 } \| g ^ { t } \| ^ { 2 } .\tag{19}
$$

For $\eta \leq 1 / L$ Eq. (19) simplifies to

$$
F ( { \theta } _ { G } ^ { t + 1 } ) \leq F ( { \theta } _ { G } ^ { t } ) - \frac { \eta } { 2 } \| g ^ { t } \| ^ { 2 }\tag{20}
$$

Since $F$ is µ-strongly convex, we use the definition

$$
F ( { \theta } _ { G } ^ { \star } ) \ge F ( { \theta } _ { G } ^ { t } ) + \langle \nabla F ( { \theta } _ { G } ^ { t } ) , { \theta } _ { G } ^ { \star } - { \theta } _ { G } ^ { t } \rangle + \frac { \mu } { 2 } \| { \theta } _ { G } ^ { \star } - { \theta } _ { G } ^ { t } | ^ { 2 }
$$

Rewriting,

$$
F ( { \theta } _ { G } ^ { t } ) - F ( { \theta } _ { G } ^ { \star } ) \le \langle \nabla F ( { \theta } _ { G } ^ { t } ) , { \theta } _ { G } ^ { t } - { \theta } _ { G } ^ { \star } \rangle \ - \ \frac \mu 2 \| { \theta } _ { G } ^ { \star } - { \theta } _ { G } ^ { t } \| ^ { 2 }
$$

By definition,

$$
\langle \nabla F ( { \theta } _ { G } ^ { t } ) , { \theta } _ { G } ^ { t } - { \theta } _ { G } ^ { \star } \rangle \leq \| \nabla F ( { \theta } _ { G } ^ { t } ) \| \cdot \| { \theta } _ { G } ^ { t } - { \theta } _ { G } ^ { \star } \|
$$

Therefore,

$$
F ( { \theta } _ { G } ^ { t } ) - F ( { \theta } _ { G } ^ { \star } ) \leq \| \nabla F ( { \theta } _ { G } ^ { t } ) \| \cdot \| { \theta } _ { G } ^ { t } - { \theta } _ { G } ^ { \star } \| - \frac { \mu } { 2 } \| { \theta } _ { G } ^ { \star } - { \theta } _ { G } ^ { t } \| ^ { 2 }\tag{21}
$$

We assume F is L-smooth (Assumption 1), therefore:

$$
\| \nabla F ( \theta _ { G } ^ { t } ) - \nabla F ( \theta _ { G } ^ { \star } ) \vert \leq L \| \theta _ { G } ^ { t } - \theta _ { G } ^ { \star } \|
$$

But since $\theta _ { G } ^ { \star }$ is a minimizer, $\nabla F ( \theta _ { G } ^ { \star } ) = 0$ , so:

$$
\| \nabla F ( \theta _ { G } ^ { t } ) \| \le L \| \theta _ { G } ^ { t } - \theta _ { G } ^ { \star } \| \implies \| \theta _ { G } ^ { t } - \theta _ { G } ^ { \star } \| \ge \frac { \| \nabla F ( \theta _ { G } ^ { t } ) \| } { L }\tag{22}
$$

Using Eq. (22) in Eq. (21) we get,

$$
\begin{array} { r l } & { F ( { \boldsymbol { \theta } } _ { G } ^ { t } ) - F ( { \boldsymbol { \theta } } ^ { * } ) \leq \| { \boldsymbol { g } } ^ { t } \| \cdot \frac { \| { \boldsymbol { g } } ^ { t } \| } { L } - \frac { \mu } { 2 } \left( \frac { \| { \boldsymbol { g } } ^ { t } \| } { L } \right) ^ { 2 } } \\ & { \qquad = \frac { \| { \boldsymbol { g } } ^ { t } \| ^ { 2 } } { L } - \frac { \mu } { 2 } \cdot \frac { \| { \boldsymbol { g } } ^ { t } \| ^ { 2 } } { L ^ { 2 } } } \\ & { \qquad = \| { \boldsymbol { g } } ^ { t } \| ^ { 2 } \left( \frac { 1 } { L } - \frac { \mu } { 2 L ^ { 2 } } \right) } \end{array}\tag{23}
$$

Invert this inequality to get a lower bound on $\| g ^ { t } \| ^ { 2 }$

$$
\begin{array}{c} g ^ { t } \| ^ { 2 } \geq \frac { F ( \theta _ { G } ^ { t } ) - F ( \theta ^ { * } ) } { \left( \frac { 1 } { L } - \frac { \mu } { 2 L ^ { 2 } } \right) } = \left( \frac { L ^ { 2 } } { L - \mu / 2 } \right) \left( F ( \theta _ { G } ^ { t } ) - F ( \theta ^ { * } ) \right)  \\ { \| g ^ { t } \| ^ { 2 } = \| \nabla F ( \theta _ { G } ^ { t } ) \| ^ { 2 } \geq 2 \mu ( F ( \theta _ { G } ^ { t } ) - F ( \theta _ { G } ^ { \star } ) ) } \end{array}\tag{24}
$$

Plug this in Eq. (20), we get,

$$
F ( { \theta } _ { G } ^ { t + 1 } ) \leq F ( { \theta } _ { G } ^ { t } ) - \eta \mu ( F ( { \theta } _ { G } ^ { t } ) - F ( { \theta } _ { G } ^ { \star } ) )
$$

On further simplification, we get,

$$
F ( \theta _ { G } ^ { t + 1 } ) - F ( \theta _ { G } ^ { \star } ) \leq ( 1 - \eta \mu ) F ( \theta _ { G } ^ { t } ) - F ( \theta _ { G } ^ { \star } ) )
$$

This linear convergence in the equation above proves that each round reduces the optimality gap by a factor of $( 1 - \eta \mu )$ Since $0 < \eta \mu < 1$ , this optimality further shrinks. By induction, ∀t,

$$
F ( \theta _ { G } ^ { t } ) - F ( \theta _ { G } ^ { \star } ) \leq ( 1 - \eta \mu ) ^ { t } F ( \theta _ { G } ^ { 0 } ) - F ( \theta _ { G } ^ { \star } ) )
$$

Now, from Assumption 2,

$$
\begin{array} { c } { { F ( { \theta } _ { G } ^ { t } ) - F ( { \theta } _ { G } ^ { \star } ) \geq \displaystyle \frac { \mu } { 2 } \| { \theta } _ { G } ^ { t } - { \theta } _ { G } ^ { \star } \| ^ { 2 } } } \\ { { \displaystyle \frac { \mu } { 2 } \| { \theta } _ { G } ^ { t } - { \theta } _ { G } ^ { \star } \| ^ { 2 } \leq ( 1 - \eta \mu ) ^ { t } ( F ( { \theta } _ { G } ^ { 0 } ) - F ( { \theta } _ { G } ^ { \star } ) ) } } \end{array}
$$

Rewriting,

$$
\| \theta _ { G } ^ { t } - \theta _ { G } ^ { \star } \| ^ { 2 } \leq \frac { 2 } { \mu } ( 1 - \eta \mu ) ^ { t } ( F ( \theta _ { G } ^ { 0 } ) - F ( \theta _ { G } ^ { \star } ) )
$$

Since $\begin{array} { r } { F ( { \theta } _ { G } ^ { 0 } ) - F ( { \theta } _ { G } ^ { \star } ) \leq \frac { L } { 2 } | { \theta } _ { G } ^ { 0 } - { \theta } _ { G } ^ { \star } | ^ { 2 } } \end{array}$ (L-smoothness assumption), we can upper bound as:

$$
\lVert { \boldsymbol { \theta } } _ { G } ^ { t } - { \boldsymbol { \theta } } _ { G } ^ { \star } \rVert ^ { 2 } \leq ( 1 - \eta \mu ) ^ { t } ( { \boldsymbol { \theta } } _ { G } ^ { 0 } - { \boldsymbol { \theta } } _ { G } ^ { \star } )
$$

which implies the parametric error $| \theta _ { G } ^ { t } - \theta _ { G } ^ { \star } | ^ { 2 }$ contracts linearly and thus establishing linear convergence.

The peer-regularization term $\left( R _ { c } ^ { t } ( \theta ) \right)$ assists convergence by reducing client-drift and effectively increasing the strong convexity of local objectives. The above result shows SAPE-FL converges to the exact minimizer $\theta _ { G } ^ { \star }$ when using an appropriate constant η or a decaying learning-rate schedule. In contrast, as mentioned in Li et al. [2020c], FedAvg on heterogeneous data may fail to reach the exact optimum unless η is decayed, incurring an ${ \cal { O } } ( \eta )$ steady-state error if η is fixed. The similarity-weighted aggregation and peer regularization in SAPE-FL overcomes this issue by focusing updates on similar clients and coupling their objectives, thereby stabilizing and accelerating convergence. □

Theorem 4 (Global Convergence under Non-Convexity). Let the similarity-weighted aggregated objective is defined as $\begin{array} { r } { F ( \theta _ { G } ^ { t } ) : = \sum _ { c } s _ { c } ^ { t } \mathcal { L } _ { c } ( \theta _ { G } ^ { t } ) } \end{array}$ . Suppose the stochasticity in local updates is bounded and gradients are uniformly bounded $| \nabla \mathcal { L } _ { c } ( { \boldsymbol { \theta } } ) | \leq \tilde { G }$ as per Assumption 3. For afixed small stepsize $\eta \leq 1 / L$ , after T rounds SAPE-FL satisfies:

$$
\operatorname* { m i n } _ { 0 \leq t < T } \mathbb { E } \left[ \| \nabla F ( \theta _ { G } ^ { t } ) \| ^ { 2 } \right] \leq \frac { 2 ( F ( \theta _ { G } ^ { 0 } ) - F _ { \mathrm { i n f } } ) } { \eta T } + \eta L \sigma ^ { 2 } ,
$$

where $F _ { \mathrm { i n f } }$ is a lower bound on $F$ and $\sigma ^ { 2 }$ is some finite variance. SAPE-FL converges to a stationary point in expectation as $T \to \infty$ . Equivalently, SAPE-FLfinds an ϵ-approximate stationary point i.e. $\mathbb { E } | \nabla F ( \theta ) | \le \epsilon \dot { i n } \dot { \mathcal { O } } ( 1 / \epsilon ^ { 2 } )$ rounds in the worst case.

Proof. Without convexity, we cannot guarantee convergence to a global minimum, but we can show the global model descends toward a stationary point where the gradient is zero. The proof leverages the L-smoothness of each $\mathcal { L } _ { c }$ and the fact that the global update is a weighted gradient step. We again interpret the update as (an approximate) gradient descent on the instantaneous aggregated loss:

$$
F ( \theta ) = \sum _ { c } \mathfrak { s } _ { c } ^ { t } \mathcal { L } _ { c } ( \theta )
$$

For notational simplicity, let us again define $g ^ { t }$ as:

$$
g ^ { t } : = \nabla F ( \theta _ { G } ^ { t } ) = \sum _ { c } \mathfrak { s } _ { c } ^ { t } \nabla \mathcal { L } _ { c } ( \theta _ { G } ^ { t } )
$$

Also let us define $\tilde { g } ^ { t }$ as

$$
\tilde { g } ^ { t } : = - \frac { 1 } { \eta } \sum _ { c } \mathfrak { s } _ { c } ^ { t } \Delta _ { c } ^ { t }
$$

be the effective global gradient that the server uses (so ideally $\tilde { g } ^ { t } \approx g ^ { t } )$ . The peer regularization again helps align $\tilde { g } ^ { t }$ with $g ^ { t }$ by discouraging any client from drifting too far from the global model or its peers. We can regard $\tilde { g } ^ { t }$ as an unbiased estimator of $g ^ { t }$ (since $\mathbb { E } [ \Delta _ { c } ^ { t } ] \approx - \eta \nabla \bar { \mathcal { L } } _ { c } ( \theta _ { G } ^ { t } )$ and assuming full client participation and unbiased local gradients). Therefore, $\mathbb { E } [ \tilde { g } ^ { t } \mid \theta _ { G } ^ { t } ] = g ^ { i }$ , and the variance between them is bounded as (Assumption 3):

$$
\mathbb { E } [ | \tilde { g } ^ { t } - g ^ { t } | ^ { 2 } ] \le \sigma ^ { 2 }
$$

Using L-smoothness of $F$ and for any $\eta \leq 1 / L$

$$
\begin{array} { r l } & { \mathbb { E } \big [ F ( \theta _ { G } ^ { t + 1 } ) \mid \theta _ { G } ^ { t } \big ] \leq F ( \theta _ { G } ^ { t } ) + \mathbb { E } \big [ \langle \nabla F ( \theta _ { G } ^ { t } ) , \theta _ { G } ^ { t + 1 } - \theta _ { G } ^ { t } \rangle \mid \theta _ { G } ^ { t } \big ] \ \cdot \ \cdot \ . } \\ & { \qquad \cdot \cdot \cdot + \frac { L } { 2 } \mathbb { E } \big [ \Vert \theta _ { G } ^ { t + 1 } - \theta _ { G } ^ { t } \Vert ^ { 2 } \mid \theta _ { G } ^ { t } \big ] } \\ & { = F ( \theta _ { G } ^ { t } ) - \eta , \langle g ^ { t } , \mathbb { E } [ \tilde { g } ^ { t } \mid \theta _ { G } ^ { t } ] \rangle \cdot \ \cdot \ } \\ & { \qquad \cdot \cdot \ + \frac { L \eta ^ { 2 } } { 2 } \mathbb { E } \big [ \Vert \tilde { g } ^ { t } \Vert ^ { 2 } \mid \theta _ { G } ^ { t } \big ] } \\ & { = F ( \theta _ { G } ^ { t } ) - \eta \Vert g ^ { t } \Vert ^ { 2 } + \frac { L \eta ^ { 2 } } { 2 } , \mathbb { E } \big [ \Vert \tilde { g } ^ { t } \Vert ^ { 2 } \mid \theta _ { G } ^ { t } \big ] } \end{array}
$$

Since $\mathbb { E } | \tilde { g } ^ { t } | ^ { 2 } = | g ^ { t } | ^ { 2 } + \mathbb { E } | \tilde { g } ^ { t } - g ^ { t } | ^ { 2 } \leq | g ^ { t } | ^ { 2 } + \sigma ^ { 2 }$ , we get:

$$
\mathbb { E } [ F ( \theta _ { G } ^ { t + 1 } ) ] \le \mathbb { E } [ F ( \theta _ { G } ^ { t } ) ] - \frac { \eta } { 2 } \mathbb { E } \left[ \Vert g ^ { t } \Vert ^ { 2 } \right] + \frac { L \eta ^ { 2 } } { 2 } \sigma ^ { 2 }
$$

Rearrange this to obtain a lower bound on $\mathbb { E } [ | g ^ { t } | ^ { 2 } ]$

$$
\frac { \eta } { 2 } \mathbb { E } \big [ \| g ^ { t } \| ^ { 2 } \big ] \ \le \ \mathbb { E } \big [ F ( \theta _ { G } ^ { t } ) \big ] - \mathbb { E } \big [ F ( \theta _ { G } ^ { t + 1 } ) \big ] + \frac { L \eta ^ { 2 } } { 2 } \sigma ^ { 2 }\tag{25}
$$

Given that $F ( \theta _ { G } ^ { t } )$ is a convex combination of the $\mathcal { L } _ { c }$ at $\theta _ { G } ^ { t }$ . Since all $\mathcal { L } _ { c }$ are bounded, we assume lower bounds on $F$ as $F _ { \mathrm { i n f } } > - \infty$ . Moreover, the descent condition in Eq. (25) indicates that if the gradient norm $\| g ^ { t } \|$ stays away from $0 ,$ the term $F ^ { t } ( \theta _ { G } ^ { t } ) - F ^ { t } ( \theta _ { G } ^ { t + 1 } )$ will be significantly positive, meaning the global loss decreases by a non-trivial amount. Summing Eq. (25) over t = 1 to T:

$$
\frac { \eta } { 2 } \sum _ { t = 0 } ^ { T - 1 } \mathbb { E } \left[ \| g ^ { t } \| ^ { 2 } \right] \le \mathbb { E } \left[ F ( \theta _ { G } ^ { 0 } ) \right] - \mathbb { E } \left[ F ( \theta _ { G } ^ { T - 1 } ) \right] + \frac { L \eta ^ { 2 } } { 2 } T \sigma ^ { 2 }\tag{26}
$$

Substituting $F ( \theta _ { G } ^ { T - 1 } ) \ge F _ { \mathrm { i n f } }$ and dividing both sides by $\scriptstyle { \frac { \eta } { 2 } } T$ gives:

$$
\frac { 1 } { T } \sum _ { t } \mathbb { E } \left[ \| \nabla F ( \theta _ { G } ^ { t } ) \| ^ { 2 } \right] \leq \frac { 2 ( F ( \theta _ { G } ^ { 0 } ) - F _ { \mathrm { i n f } } ) } { \eta T } + \eta L \sigma ^ { 2 }\tag{27}
$$

This is exactly the stated bound. It implies that the minimum gradient norm (in expectation) over the first $T$ rounds is bounded by the same RHS (since the minimum is less than the average). In particular, if we set $\eta$ to a small constant or use a decaying $\eta ,$ the term $\dot { L } \eta , \sigma ^ { 2 }$ can be made negligible, giving

$$
\operatorname* { m i n } _ { 0 \le t < T } \mathbb { E } \left[ | | \nabla F ^ { t } ( \theta _ { G } ^ { t } ) | | ^ { 2 } \right] \approx \frac { 2 ( F ( \theta _ { G } ^ { 0 } ) - F _ { \mathrm { i n f } } ) } { \eta T }
$$

Thus, as $T \to \infty$ , the minimum expected gradient norm approaches zero. Now, suppose we want the minimum expected gradient norm across all rounds to be less than or equal to ϵ, then from Eq. (27),

$$
\operatorname* { m i n } _ { 0 \leq t < T } \mathbb { E } \left[ \| \nabla F ^ { t } ( \theta _ { G } ^ { t } ) \| ^ { 2 } \right] \leq \frac { 2 ( F ( \theta _ { G } ^ { 0 } ) - F _ { \mathrm { i n f } } ) } { \eta T } + L \eta \sigma ^ { 2 } \leq \epsilon ^ { 2 }
$$

In the above equation the first term in the RHS decreases as $1 / T ,$ , and the second term increases linearly with $\eta ,$ so choosing $\begin{array} { r } { \eta = \frac { 1 } { \sqrt { T } } } \end{array}$ balances the two terms, giving the following bounds.

$$
\operatorname* { m i n } _ { 0 \leq t < T } \mathbb { E } \left[ \Vert \nabla F ^ { t } ( \theta _ { G } ^ { t } ) \Vert ^ { 2 } \right] \leq \mathcal { O } \left( \frac { 1 } { \sqrt { T } } \right)
$$

Solving $O ( 1 / \sqrt { T } ) \le \epsilon ^ { 2 }$ gives $T = \mathcal { O } ( 1 / \epsilon ^ { 2 } )$ . Therefore, SAPE-FL guarantees convergence to an ϵ-stationary point within $\bar { \mathcal { O } } ( 1 \dot { / } \epsilon ^ { 2 } )$ communication rounds. The above result shows that SAPE-FL, despite personalization, will not diverge on non-convex objectives; instead it will approach a stationary solution. In fact, the client selection in SAPE-FL can improve the convergence stability by filtering out dissimilar updates and thus providing an additional safeguard against high-variance updates. □

## A.2 Additional Experimental Details

## A.2.1 Baselines

Since FL litearture is fairly broad, we restrict our comparison specifically to methods that exhibit fundamental similarities and conceptual relevance. For the sake of completeness in our experiments, we have also compared the performance of the local-only trained models. We use the following baselines:

• FedAvg McMahan et al. [2017]: The classic federated learning approach that aggregates client models by naively averaging their weights.

• PerFedAvg Fallah et al. [2020]: A meta-learning-based personalized federated learning method that optimizes an initial global model to facilitate effective local adaptation at each client.

• FedProx Li et al. [2020b]: Enhances FedAvg by introducing a proximal term to mitigate local drift during training, thereby managing heterogeneity in federated settings.

• DittoFL Li et al. [2021a]: Employs a personalized regularization term (analogous to proximal term in FedProx) that explicitly balances local fitting with adherence to the global model, offering a strong baseline for personalized federated learning.

• FedACS Chen et al. [2024]: Introduces attention-based client selection and aggregation to provide personalized federated learning, considering similarity measured between clients based on their model updates.

• LCFed Zhang et al. [2025]: Proposes a lightweight client clustering framework that groups clients based on cosine similarity of their model updates to enable efficient and personalized federated training.

## A.2.2 Datasets

We evaluate the performance of our proposed SAPE-FL framework on both synthetic and standard real-world multi-class classification problems using standard datasets used in the federated learning community.

Synthetic Datasets To simulate synthetic classification data, we utilize the make\_classification() function from the scikit-learn library Pedregosa et al. [2011], which generates a multi-class dataset by sampling from multivariate Gaussian distributions. Mathematically, for a specified number of classes $\mathcal { C } = 1 0$ , number of features $d = 2 0$ , number of informative features $( n _ { \mathrm { i n f o } } = 0 . 7 5 \times d )$ and total samples $N = 1 0 0 0 0$ , we construct C Gaussian clusters in a feature space $\mathbb { R } ^ { d }$ . Each cluster corresponds to a class label and is characterized by a randomly generated mean vector and a covariance structure that governs feature correlation. Informative features are those that directly influence class separation, while redundant features are created as linear combinations of the informative ones. Non-informative (noise) features are sampled independently from a Gaussian distribution. To add the statistical heterogeneity we randomly change the cluster of data points with probability = 0.4 i.e., change the class labels during the data generation process and then split the dataset among all the clients.

Real-world Datasets We use the following publicly available classification datasets commonly utilized in federated learning literature.

• Human Activity Recognition Datasets: In this task the participants were engaged in distinct physical activities such as walking, sitting, and standing, which serve as the target class labels for classification. The raw input features are sourced from the smartphone’s built-in accelerometer and gyroscope, capturing tri-axial motion and orientation signals over time. The objective is to predict the human activity based on their accelerometer and Gyroscope data. We use publicly available HAR dataset Reyes Ortiz et al. [2013] and AccGyro Dataset Abdullah [2022] for predicting the human activity.

• Image Classification Datasets: We evaluate our models on four widely used image classification datasets: MNIST, Fashion-MNIST (FMNIST), CIFAR-10, and CIFAR-100. MNIST consists of 70,000 grayscale images of handwritten digits (0˘9), each of size 28 × 28 pixels, serving as a foundational dataset for digit recognition tasks. EMNIST extends MNIST dataset by incorporating handwritten uppercase and lowercase letters, resulting in more complex variants with up to 62 classes. Fashion-MNIST retains the 2828 grayscale format but replaces digits with images of clothing items from 10 fashion categories, introducing greater visual diversity and classification difficulty. In contrast, CIFAR-10 and CIFAR-100 comprise $3 2 \times 3 2$ color (RGB) images with 10 and 100 classes, respectively, capturing objects such as animals, vehicles, and household items. Together, these datasets span a range of tasks from digit and character recognition to object and fine-grained image classification, enabling comprehensive evaluation across varying input modalities and classification granularities.

## A.2.3 Data Partitions

To simulate the statistical heterogeneity commonly observed in federated learning environments, we partition the global dataset among clients using a Dirichlet distribution-based sampling strategy. Specifically, we draw client-specific label distributions from a Dirichlet distribution parameterized by a concentration parameter κ, where lower values of κ produce higher degrees of heterogeneity by encouraging label imbalance across clients. This setup ensures that each client possesses a unique and potentially skewed distribution of data, capturing the non-IID nature of federated settings. We set the concentration parameter $\kappa = 0 . 3$ to systematically control the severity of data skew, enabling the evaluation of SAPE-FL framework’s robustness under varying levels of prior probability shift and quantity imbalance.

## A.3 Parameters of SAPE-FL

We have set the number of clients to 100 in our experiments using synthetic, human activity recognition and image classification datasets. We split the data at each client in the $8 0 - 2 0 \%$ train-test ratio. We have set the participants ratio r to 70%, i.e., at each communication round 70% of the clients participate in the federated learning process. We have set the influence factor mentioned in Eq. (6) to $\delta = 0 . 5$ . We have implemented all the methods considered using Torch 2.7.0.

## A.3.1 Performance Metrics and Client Models

The empirical performance of our proposed and the competing methods is evaluated by measuring the mean testing accuracy i.e., the average (along with the standard deviation) test accuracies computed across all participating clients indicating overall classification performance. We also report average (along with the standard deviation) AUC scores, a robust measure particularly valuable in scenarios with imbalanced data distributions, assessing the model’s capability to distinguish between different classes. Although our proposed method is applicable across different model architectures, for synthetic data and human activity recognition (HAR) tasks, we employ a five-layer multi-layer perceptron (MLP) consisting of an input layer followed by three hidden layers of sizes 1024, 512, and 256, each with ReLU activation, and a final output layer matching the task-specific number of classes. For classification tasks with image datasets such as CIFAR-10, MNIST, EMNIST, and FEMNIST, we utilize lightweight Convolutional Neural Network (CNN) architectures composed of two convolutional layers, followed by max-pooling, and two fully connected layers.

## A.3.2 Numerical Stability in Peer and Global Anchoring

Due to the use of similarity thresholding and nonnegativity truncation in the computation of peer similarities (via ν), it is possible that, for a given client c at communication round t, no peer model satisfies the threshold criterion $S _ { c j } ^ { t } > S _ { \operatorname* { m i n } }$ This situation may arise particularly in early training rounds, under severe data heterogeneity, or when client models are totally different. In such cases, the peer set becomes empty, resulting in a zero normalization constant in the peer-averaged model computation. To ensure numerical stability and maintain a well-defined optimization objective, we adopt a conservative fallback strategy by disabling the peer regularization term for that round, i.e., setting $\check { S } _ { c } ^ { P } = 0$ This effectively removes peer influence while allowing the client to rely solely on the global anchor, thereby avoiding arbitrary parameter averaging and preserving stable personalization behavior.

Additionally, a similar case may arise during server-side aggregation when the aggregate similarity mass becomes zero, i $\begin{array} { r } { . \mathbf { e } . , \sum _ { c } s _ { c } ^ { t } = 0 } \end{array}$ , where $s _ { c } ^ { t }$ denotes the cumulative similarity score associated with client c. This situation can occur when similarity scores are truncated to zero due to the non-negativity constraint imposed on the cosine similarity function, or when participating clients are highly dissimilar and fail to exhibit sufficient alignment with their peers. To ensure numerical stability and prevent arbitrary updates under such degenerate conditions, we adopt a conservative fallback strategy and skip the global model update for that round by setting $\theta _ { G } ^ { t + 1 }  \theta _ { G } ^ { t }$ . This approach avoids incorporating unreliable or uninformative updates while maintaining stable training dynamics, particularly in early rounds or under extreme data heterogeneity.

## A.3.3 Scalability of SAPE-FL

We would like to clarify that while the formulation in Eq. (8) defines peer-averaging over all clients, however in practice each client interacts only with a small sampled subset of peers (refer to step 2 of Algorithm 1) rather than all N clients, keeping the complexity $\mathcal { O } ( C )$ where $\boldsymbol C \ll \boldsymbol N$ . The additional computation arises from a few forward passes of peer models on local mini-batches, which is minor compared to local training. In our experiments, this design yields $3 - 5 \%$ accuracy and 2 − 4% AUC improvements, demonstrating a favorable accuracy-to-cost trade-off. In contrast to FedProx, which uses a single global anchor, SAPE-FL adaptively exploits meaningful peer information, reducing variance and negative transfer.

## A.3.4 Cosine Similarity based Scores

We have adopted a modified (clipped) cosine similarity for comparisons with the anchor models, because it provides a scale-invariant measure of alignment that remains numerically stable across heterogeneous clients. Unlike Euclidean or kernel-based distances, cosine similarity focuses on directional coherence rather than magnitude, making it robust to local learning-rate or norm differences common in FL. Using cosine similarity measure for both weights and outputs also ensures consistent interpretation of alignment across structural and functional spaces, simplifying the adaptive weighting mechanism in Eq. (6).

![](images/a575e86ce4623f4a4658d9e87bd7615da79cacf935629c11d6773a1077178ca0.jpg)  
Figure 1: Client Accuracy over Communication Rounds. Accuracy trajectories for a random set of clients using the synthetic dataset over 100 communication rounds. Shaded regions represent one standard deviation across multiple runs.

## B Additional Experiments and Ablation Study

## B.1 Evolution of Client Models Over Rounds

To illustrate the effectiveness of our proposed framework in improving client-level performance over communication rounds, we conducted an additional experiment using the synthetic dataset described earlier. In this setup, we retain most experimental settings from the main manuscript, except we reduce the number of clients to $C = 2 0$ and the number of communication rounds to 100. Figure 1 shows the accuracy trajectories 5 randomly selected clients. We observe a consistent upward trend in accuracy across all the clients, with most achieving stable performance after a sufficient number of communication rounds. These results demonstrate that SAPE-FL framework enables steady local improvement by leveraging personalized and peer-regularized training, even in settings with increased client population and extended training duration.

## B.2 Uplifting Clients with Poor Performance

To demonstrate how SAPE-FL framework mitigates negative knowledge transfer from poorly performing clients while also helping them to improve their performance, we conducted an additional experiment retaining most of the hyperparameter settings mentioned in the main manuscript. In this experiment, for clarity in presenting the results, we compare accuracy gains only against the classical FedAvg algorithm. Additionally, in contrast to the default settings mentioned earlier, we have reduced the number of clients from $C = 1 0 0 \mathrm { t o } C = 2 0$ and assumed a full participation i.e., r = 1. This experiment follows a three-stage evaluation protocol.

1. Local-only baseline. Each client trained an MLP exclusively on its own data E = 15 epochs. The resulting test accuracy is denoted $\mathbf { A c c } _ { c } ^ { \mathrm { l o c a l } }$

2. FedAvg baseline. We reset all models to the common global initialisation and executed vanilla FedAvg for the same number of communication rounds used by SAPE-FL . The final accuracy is denoted as $\operatorname { A c c } _ { c } ^ { \mathrm { F A } }$

3. SAPE-FL Starting again from the common initialisation, we ran our full similarity–aware, dual-anchored algorithm and obtained accuracy $\mathbf { A c c } _ { c } ^ { \mathbf { S A P E } }$

Benefit for weakest 25% clients  
![](images/9200a3b1b06957f7d94dbb427949cf77b57b7afd789fea98bcfb4dff99d2e9f3.jpg)  
Figure 2: Accuracy Gain of Weak Clients. Comparison of local accuracy improvements achieved by SAPE-FL and FedAvg for the lowest-performing 25% of total clients (C = 20). The bar chart shows the per-client accuracy gain for those 5 selected clients.

For every client c we then compute the accuracy gains:

$$
\Delta _ { c } ^ { \mathrm { F A } } = \mathrm { A c c } _ { c } ^ { \mathrm { F A } } - \mathrm { A c c } _ { c } ^ { \mathrm { l o c a l } }
$$

$$
\Delta _ { c } ^ { \mathrm { S A P E } } = \mathrm { A c c } _ { c } ^ { \mathrm { S A P E } } - \mathrm { A c c } _ { c } ^ { \mathrm { l o c a l } }
$$

Next we select the bottom 25% of clients (poor-performing) according to $\mathbf { A c c } _ { c } ^ { \mathrm { l o c a l } }$ and show the margin by which these clients get improved via SAPE-FL framework. In Figure 2 we plots these accuracy gains. Figure 2 depicts the accuracy gains in one instance of the the experiment where clients C1, C5, C4, C13, C10 are the 5 clients (25% out of all 20 clients) that are most affected by data heterogeneity i.e., clients with poor predictive models. From the plot it is evident that the margin of gain using SAPE-FL (orange) framework outperforms classicial FedAvg (blue) algorithm. Therefore these empirical results confirm that the similarity-weighted global update together with the peer-anchored regularizer delivers pronounced benefits in avoiding negative knowledge transfer.

## B.3 Influence Factor δ and Similarity Scores S

To investigate the relative contribution of output similarity and weight similarity in our proposed similarity measure mentioned in Eq. (6), we conducted an ablation study by varying the influence factor $\delta \in \dot { \{ 0 , \dot { 0 } . 2 5 , 0 . 5 , 0 . 7 5 , \dot { 1 } \} }$ }. These values determine the extent to which the final similarity score depends on output-based versus weight-based alignment, with δ = 1 using only output similarity and δ = 0 relying solely on weight similarity. The experimental settings were kept identical to those used in our main experiments as mentioned in the main manuscript. The experimental results are mentioned in Table 3. Empirically, we observed that higher values of δ consistently led to improved performance across datasets. This aligns with our hypothesis that output similarity is a more reliable indicator of task alignment, as it reflect functional behavior even when client models converge to different but equally good local optima in heterogeneous settings. In contrast, relying solely on weight similarity still provides a useful approximation but can be relatively less effective due to the potential misalignment of model parameters that does not necessarily reflect true functional similarity.

Additionally, we intentionally use the same tunable influence factor to maintain consistency and stability in the relative weighting of output-space and weight-space similarities across anchors. This shared δ ensures that both peer and global similarities are measured on a comparable scale, avoiding conflicting gradient directions that can arise from independently tuned factors. Our empirical results show that a single factor suffices to balance structural and functional alignment for both anchors. While distinct δ values for peer and global anchors could be explored in future work, the current unified formulation already provides stable results across datasets without additional tuning burden.

Table 3: Ablation Study on Influence Factor δ. Mean test accuracies of the SAPE-FL framework under varying contributions of output similarity and weight similarity in the computation of the similarity score. Each cell represents the mean accuracy across all clients at the end of 200 communication rounds. The best-performing result in each row is highlighted in bold.
<table><tr><td>Dataset</td><td> $\delta = 0$ </td><td> $\delta = 0 . 2 5$ </td><td> $\delta = 0 . 5$ </td><td> $\delta = 0 . 7 5$ </td><td> $\delta = 1$ </td></tr><tr><td>Synthetic</td><td> $8 6 . 3 0 \pm 0 . 6 6$ </td><td> $8 6 . 7 2 \pm 0 . 6 1$ </td><td> $8 7 . 4 9 \pm 0 . 7 5$ </td><td> ${ \bf 8 9 . 1 2 \pm 0 . 5 4 }$ </td><td> $8 5 . 0 7 \pm 0 . 2 7$ </td></tr><tr><td>HAR</td><td> $7 8 . 1 4 \pm 0 . 2 6$ </td><td> $7 9 . 0 6 \pm 0 . 4 1$ </td><td> $8 1 . 4 4 \pm 0 . 7 8$ </td><td> ${ \bf 8 1 . 7 4 \pm 0 . 6 3 }$ </td><td> $7 7 . 3 2 \pm 0 . 8 8$ </td></tr><tr><td>MNIST</td><td> $8 0 . 0 5 \pm 0 . 1 2$ </td><td> $8 0 . 8 2 \pm 0 . 0 9$ </td><td> ${ \bf 8 1 . 0 2 \pm 0 . 3 3 }$ </td><td> $8 0 . 3 5 \pm 0 . 2 4$ </td><td> $7 8 . 7 7 \pm 0 . 6 9$ </td></tr><tr><td>EMNIST</td><td> $7 3 . 2 9 \pm 0 . 7 3$ </td><td> $7 3 . 0 6 \pm 0 . 2 2$ </td><td> $7 4 . 1 7 \pm 0 . 6 8$ </td><td> ${ \bf 7 5 . 3 9 \pm 0 . 1 5 }$ </td><td> $7 2 . 9 4 \pm 0 . 6 6$ </td></tr></table>

## B.4 Impact of Data Heterogeneity κ

To evaluate the robustness of SAPE-FL under varying degrees of statistical heterogeneity, we conducted additional experiments by controlling the Dirichlet concentration parameter $\kappa \in \{ 0 . 2 , 0 . 3 , \bar { 0 } . 4 , 0 . \bar { 5 } , 0 . 6 , 1 . 0 \}$ , where smaller values of κ correspond to more severe data skew across clients. As shown in Table 4, the performance is strongest at $\kappa = 0 . 3$ across all the considered datasets (Synthetic: 90.66%, HAR: 81.22%, AccGyro: 81.24%, MNIST: 80.98%), highlighting the effectiveness of the proposed similarity-aware personalization and aggregation mechanisms under non-IID conditions. As κ increases and client distributions becomes less skewed, the performance decreases gradually $( \mathrm { e . g . }$ , MNIST drops from 80.98% at $\kappa = 0 . 3$ to 78.82% at $\kappa = 0 . 6 )$ , which is consistent with the reduced benefit of personalization when heterogeneity is weaker. Nevertheless, SAPE-FL remains stable and competitive across all considered κ values, confirming that it maintains robustness across varying levels of statistical heterogeneity.

Table 4: Performance (in accuracy) of SAPE-FL under different levels of data heterogeneity controlled by the Dirichlet concentration parameter κ. Smaller κ indicates more non-IID data distributions across clients. Accuracy scores in bold indicate the best and higher the better.
<table><tr><td>Datasets</td><td> $\kappa = 0 . 2$ </td><td> $\kappa = \mathbf { 0 . 3 }$ </td><td> $\kappa = \mathbf { 0 . 4 }$ </td><td> $\kappa = 0 . 5$ </td><td> $\kappa = 0 . 6$ </td><td> $\kappa = \mathbf { 1 }$ </td></tr><tr><td>Synthetic</td><td> $8 9 . 2 4 \pm 0 . 2 5$ </td><td> ${ \bf 9 0 . 6 6 \pm 0 . 8 7 }$ </td><td> $8 7 . 1 2 \pm 0 . 4 4$ </td><td> $8 6 . 9 4 \pm 0 . 7 1$ </td><td> $8 6 . 5 6 \pm 0 . 2 2$ </td><td> $8 6 . 0 1 \pm 0 . 7 9$ </td></tr><tr><td>HAR</td><td> $7 9 . 4 4 \pm 0 . 3 1$ </td><td> ${ \bf 8 1 . 2 2 \pm 0 . 5 6 }$ </td><td> $8 0 . 9 1 \pm 0 . 4 0$ </td><td> $8 0 . 0 5 \pm 0 . 2 5$ </td><td> $7 9 . 8 7 \pm 0 . 6 6$ </td><td> $7 6 . 0 8 \pm 0 . 5 3$ </td></tr><tr><td>AccGyro</td><td> $8 0 . 2 2 \pm 0 . 3 9$ </td><td> ${ \bf 8 1 . 2 4 \pm 0 . 1 1 }$ </td><td> $8 0 . 8 1 \pm 0 . 6 8$ </td><td> $8 0 . 1 3 \pm 0 . 2 9$ </td><td> $7 9 . 6 1 \pm 0 . 1 8$ </td><td> $7 9 . 0 6 \pm 0 . 2 2$ </td></tr><tr><td>MNIST</td><td> $8 0 . 0 7 \pm 0 . 6 0$ </td><td> $8 0 . 9 8 \pm 0 . 1 7$ </td><td> ${ \bf 8 1 . 0 4 \pm 0 . 1 9 }$ </td><td> $7 9 . 2 3 \pm 0 . 8 7$ </td><td> $7 8 . 8 2 \pm 0 . 0 5$ </td><td> $7 8 . 0 1 \pm 0 . 2 0$ </td></tr></table>

Table 5: Ablation Study on Threshold $S _ { \bf m i n }$ . Mean test accuracies of the SAPE-FL framework under varying similarity threshold $S _ { \mathrm { m i n } }$ . Each cell represents the mean accuracy across all clients at the end of 100 communication rounds. The best-performing result in each row is highlighted in bold.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1> $S _ { \bf m i n } = 0 . 2$ </td><td rowspan=1 colspan=1> $S _ { \bf m i n } = 0 . 4$ </td><td rowspan=1 colspan=1> $S _ { \bf m i n } = 0 . 6$ </td><td rowspan=1 colspan=1> $S _ { \bf m i n } = 0 . 8$ </td></tr><tr><td rowspan=1 colspan=1>SyntheticHARMNIST</td><td rowspan=1 colspan=1> $8 7 . 2 1 \pm 0 . 6 6$  $7 5 . 2 1 \pm 0 . 3 9$  $7 7 . 2 9 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1> $8 8 . 1 4 \pm 0 . 5 5$  $7 7 . 0 8 \pm 0 . 6 5$  $7 8 . 0 1 \pm 0 . 3 3$ </td><td rowspan=1 colspan=2> ${ \bf 8 8 . 6 3 \pm 0 . 4 9 }$      $8 6 . 6 0 \pm 0 . 4 2$  $\mathbf { 7 8 . 9 4 \pm 0 . 0 . 7 7 }$     $7 7 . 0 9 \pm 0 . 5 1$  $7 8 . 5 5 \pm 0 . 3 2$      ${ \bf 7 9 . 6 9 \pm 0 . 3 8 }$ </td></tr></table>

## B.5 Effect of Similarity Threshold $S _ { \bf m i n }$

The similarity threshold $S _ { \mathrm { m i n } }$ plays a critical role in filtering out dissimilar peers during the computation of the peeraveraged anchor in our SAPE-FL framework. This threshold governs the inclusion of clients in the peer set, thereby directly influencing the regularization term that guides local optimization. A lower value of $S _ { \mathrm { m i n } }$ permits broader collaboration by including a wider range of clients, but at the risk of negative transfer from poorly aligned or noisy peers. Conversely, a higher value enforces stricter trust, ensuring only highly similar clients influence each other, but potentially resulting in under-utilization of peer support, especially in highly heterogeneous or sparse client distributions.

We have conducted an ablation study to understand the effects of the similarity threshold required for constructing peer-averaged model. In this experiment we use the subset of datasets described earlier. We conducted experiments with a range of values $S _ { \mathrm { m i n } } \in \{ 0 . 2 , \bar { 0 } . 4 , 0 . 6 , 0 . 8 \}$ . As shown in Table 5, we found that thresholds in the range of [0.6, 0.8] consistently yielded better performance across datasets. This range effectively balances the trade-off between peer set size and quality, ensuring that each client aggregates from sufficiently aligned peers while still benefiting from diversity in collaboration. Lower thresholds (e.g., 0.2 or 0.4) tended to introduce noisy or weakly aligned peers, reducing the effectiveness of the regularization and leading to sub-optimal personalization.

## B.6 Limitations and Future Work

While SAPE-FL framework provides a structured and effective approach to personalization through similarity-aware regularization and aggregation, certain practical considerations arise in real-world deployments. First, computing pairwise similarities and maintaining peer sets can introduce an additional computational and communication burden, especially in large-scale deployments. However, this can be mitigated using suitable matrix approximation techniques such as low-rank factorization that reduce the computational overhead on each client. Next, although the framework is inherently designed to handle client heterogeneity, its performance relies on the presence of at least a few peers with meaningful alignment. In scenarios with highly disjoint or adversarial client distributions, we can possible think of adaptive peer filtering mechanisms or fallback strategies (defaulting to global model regularization) that can maintain personalization quality and stability.

SAPE-FL framework demonstrated promising improvements over existing personalized federated learning approaches. However, there are several research directions open for future exploration. First, the current gradient similarity-based trust mechanism could be enhanced with a more nuanced similarity metric such as a Bregman divergence. Second, extending SAPE-FL framework to support dynamic client participation, where clients may join or leave during training phase, could improve its applicability in real-world asynchronous FL deployments. Furthermore, integrating privacypreserving techniques such as differential privacy or secure multi-party computation into the trust score computation remains an open challenge.