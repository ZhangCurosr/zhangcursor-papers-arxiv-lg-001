# An Adversarial Zero-Shot Learning Approach for Anomaly Detection in Multivariate IoT Trafic Data<sup>⋆</sup>

Mahshid Rezakhani<sup>a,1</sup>, Tolunay Seyfi<sup>a,∗,1</sup> and Fatemeh Afghah<sup>a</sup>

<sup>a</sup>Holcombe Department of Electrical and Computer Engineering, Clemson University, Clemson, 29634 SC, USA

## A R T I C L E I N F O

Keywords:   
Zero-Shot Learning   
IoT Anomaly Detection   
Domain Adaptation   
Variational Autoencoder   
Contrastive Learning   
Multivariate Time-Series   
Adversarial Representation Learning

## A BS T RA C T

Anomaly detection in Internet of Things (IoT) networks presents unique challenges due to the diversity of devices, lack of labeled data, and domain variability across environments. In this paper, we propose a novel framework for multivariate time-series anomaly detection that leverages adversarial learning and contrastive loss within a sequence-based Variational Autoencoder (VAE) architecture. Our method enables zero-shot domain adaptation by jointly optimizing domain-invariant latent representations and semantically structured embedding spaces, without requiring labeled data or raw feature transfer. To address the heterogeneity of IoT deployments, we introduce encoder and decoder adaptor layers that align feature distributions across domains while preserving contextual semantics. Additionally, we propose a destination-based segmentation strategy to better model real-world communication structures in IoT trafic. Our framework is comprehensively evaluated on six distinct datasets spanning industrial, enterprise, general-purpose, smart home, and military automation domains across 44 transfer scenarios. Experimental results demonstrate strong zero-shot generalization in several cross-domain settings and competitive performance against a contrastive domain-adaptation baseline under realistic, heterogeneous, and privacy-constrained IoT conditions.

## 1. Introduction

The rapid proliferation of Internet of Things (IoT) devices across industrial, urban, and defense infrastructures has heightened the risk of operational failures and security breaches. Anomaly detection (AD) in IoT networks is therefore a critical capability: it enables the timely identification of hardware malfunctions, communication disruptions, and malicious cyber-attacks by recognizing deviations from expected trafic patterns [1, 2]. From a security perspective, anomalies map directly to common network attacks such as scanning, denial-of-service (DoS), botnet propagation, and brute-force intrusions. From a reliability perspective, they also capture system-level degradations such as abnormal packet loss or irregular latency surges that signal hardware or connectivity failures [1].

A large fraction of these anomalies can be inferred directly from flow-level network data, where each flow aggregates packets sharing source, destination, and protocol attributes. Network-level anomaly detection thus provides visibility into a diverse set of attack classes. Reconnaissance attacks such as port and vulnerability scans manifest as repetitive short flows; volumetric floods (UDP, DNS, ICMP) appear as sustained high-throughput flows with abnormal temporal correlations; and stealthy application-layer attacks like Slowloris exhibit long-duration but low-rate deviations. By analyzing trafic at this granularity, AD models can detect both acute, short-term disruptions and gradually evolving threats [3, 1].

While anomaly detection has been widely studied in other cyber–physical domains, IoT environments introduce distinctive challenges. IoT trafic is inherently heterogeneous—devices difer in computational capabilities, communication protocols, and sampling rates—yielding highly multivariate trafic, variable sequence lengths, and irregular sampling intervals [4]. In addition, IoT deployments are distributed across diverse environments (e.g., smart grids, medical sensors, military automation), which produces severe domain shift when models trained on one setting are applied to another [5]. Finally, the scarcity of labeled anomalies in IoT networks makes purely supervised approaches impractical [2, 4].

Machine learning–based AD has emerged as a natural solution, modeling complex nonlinear dependencies without heavy feature engineering [1]. Within this space, two dominant paradigms exist: reconstruction-based methods (e.g., autoencoders) that flag large reconstruction errors, and prediction-based methods that flag large forecasting residuals; both exploit temporal correlations in multivariate trafic [4]. Recent time-series work also leverages contrastive objectives to improve representation quality for downstream detection [6].

Among these, sequence models such as LSTM-based autoencoders capture both short- and long-range temporal dependencies, and Variational Autoencoders (VAEs) extend this framework by probabilistically modeling latent spaces to improve generalization in unsupervised settings [7]. Despite these advantages, commonly used VAE-based approaches in IoT still face two critical limitations: (i) many assume fixed-length input sequences, limiting robustness to realworld trafic variability, and (ii) they often lack mechanisms to transfer knowledge across heterogeneous domains without retraining on target data [4, 5].

Efective AD in IoT networks therefore requires models that can (a) accept variable-length flows to capture both rapid and gradual deviations, (b) remain robust under domain shift without fine-tuning on new data, and (c) operate in zero-shot conditions without access to labeled target samples [4, 5, 2]. Few existing approaches simultaneously address these requirements; contrastive learning and domain adaptation are promising ingredients but are rarely combined with strict zero-shot constraints in flow-based IoT settings [6, 5].

To fill this gap, we propose a zero-shot anomaly detection framework that integrates adversarial domain adaptation and contrastive representation learning into a sequencedriven LSTM–VAE architecture. Our method enforces domain invariant latent representations while structuring the latent space to capture semantic relationships between flows. Unlike prior work, it supports variable-length sequences and generalizes to unseen domains without requiring raw data, labels, or retraining, making it suitable for dynamic, privacysensitive, and continually evolving IoT ecosystems.

## 2. Related Work

## 2.1. Supervised and Semi-Supervised AD

Label-rich models generally achieve strong in-domain accuracy but require continuous annotation and re-labeling as trafic evolves. ST-VAE [8] combines spatiotemporal VAE encoding with supervised contrastive learning and entropyaware sampling to mitigate concept drift; however, its reliance on labeled feedback limits deployment in label-scarce environments. Transformer-based detectors such as Swin-IoT [9] tailor hierarchical vision Transformers and SLAA modules to structured visual inputs at the edge, yet the specialization to vision signals constrains applicability to general multivariate telemetry. Classical supervised pipelines (e.g., TF–IDF over system calls [10]) ofer interpretability and low cost but lack temporal modeling and do not transfer across unseen domains. SDN-integrated policies [11] optimize network-layer behavior with tight control loops, but policy coupling hampers portability to heterogeneous infrastructures.

## 2.2. Unsupervised AD

Unsupervised approaches remove the label dependency but still struggle with domain shift. DUDetector [12] combines Transformer modules and dual-resolution autoencoding for fine-grained segmentation, at the expense of higher compute budgets. EvoAAE [13] augments adversarial training with PSO-driven architecture search, but training stability can be fragile. Graph-based models such as DPGLAD [14] exploit dynamic difusion over evolving topologies, yet require domain-specific graph construction and extensive feature engineering. Lightweight statistical detectors (e.g., moment-based features [15]) are eficient but typically lack the representational capacity for heterogeneous IoT domains. Platform-bound pipelines like MindFlow [16] (CNN+BiLSTM on NF-BoT-IoT) highlight temporal modeling benefits but limit portability across toolchains and protocols. Federated variants [17, 18] improve privacy via decentralized training, though they often depend on synchronized participation and sensitive hyperparameters; practical deployments face stragglers and non-IID data issues.

## 2.3. Few-/Zero-Shot AD and Domain Shift Without Retraining

A growing line of work targets adaptation with few or no labels in the target domain. Typical strategies rely on (i) metric learning (contrastive/triplet) to shape a semantically meaningful latent space, and (ii) domain alignment (adversarial or discrepancy-based) to encourage invariance. However, many methods still fine-tune on target data (even unlabeled), which weakens the zero-shot guarantee and adds operational complexity. Our framework is designed for strict zero-shot inference on the target domain: the final phase computes no loss on target data and requires no targetside retraining, while contrastive and adversarial objectives (learned on source/auxiliary domains) yield a latent geometry that generalizes across domains.

## 2.4. Positioning and Distinctions

Key diferentiators of our approach relative to prior work:

• Fully unsupervised source-side training with strict zero-shot target inference: No labels are required from source or target, and the final phase performs decisions without any target loss or fine-tuning.

• Dual latent alignment: A combination of contrastive learning (semantic structuring) and adversarial domain alignment (invariance) produces transferrable representations under domain shift.

• No handcrafted features or graph construction: The method is protocol-agnostic and avoids bespoke feature engineering or topology building.

• Privacy-aware operation: No raw feature transfer from target sites is required; the model transfers solely via learned parameters trained on source/auxiliary domains.

## 2.5. Comparison With Representative Methods

To highlight axes that matter in deployment, Table 1 compares representative works along: supervision level, data modality (multivariate flows), support for variable-length sequences, explicit handling of domain shift, zero-/few-shot readiness, and retraining needs on new domains.<sup>2</sup>

## 3. System Model

In this work, we propose the Contrastive Adversarially-Adaptive LSTM-VAE, a novel framework tailored for multivariate anomaly detection in IoT and networked environments. The overall architecture of the proposed model is illustrated in Figure 1.

Table 1  
Comparison of representative IoT anomaly detection methods.
<table><tr><td>Method</td><td>Supervision</td><td>Multivar.</td><td>Var. Len.</td><td>Domain Shift</td><td>Zero/Few</td><td>Retrain?</td><td>Notes</td></tr><tr><td>ST-VAE [8]</td><td>Sup.</td><td>Yes</td><td>NR</td><td>Partial</td><td>Few-shot</td><td>Often</td><td>Supervised contrastive + spatiotemporal VAE.</td></tr><tr><td>SwinloT [9]</td><td>Sup.</td><td>Vision</td><td>NR</td><td>No</td><td>No</td><td>Yes</td><td>Hierarchical Transformers for edge vision</td></tr><tr><td>TF-IDF Syscalls [10]</td><td>Sup.</td><td>Syscall text</td><td>No</td><td>No</td><td>No</td><td>Yes</td><td>only. Lightweight, but weak temporal modeling.</td></tr><tr><td>SDN-IIoT Policy [11]</td><td>Sup./Rules</td><td>Net. flows</td><td>NR</td><td>No</td><td>No</td><td>Yes</td><td>Tight SDN coupling; limited portability.</td></tr><tr><td>DUDetector [12]</td><td>Unsup.</td><td>Yes</td><td>NR</td><td>Partial</td><td>No</td><td>Possibly</td><td>Transformer + dual-res AE; high com-</td></tr><tr><td>EvoAAE [13]</td><td>Unsup. (adv.)</td><td>Yes</td><td>NR</td><td>Partial</td><td>No</td><td>Possibly</td><td>pute. Adversarial AE + PSO NAS; unstable.</td></tr><tr><td>DPGLAD [14]</td><td>Unsup. (graph)</td><td>Graph</td><td>NR</td><td>Partial</td><td>No</td><td>Possibly</td><td>Needs domain-specific graph construc- tion.</td></tr><tr><td>Stat. Moments [15]</td><td>Sup./Heur.</td><td>Yes (hand)</td><td>NR</td><td>No</td><td>No</td><td>Yes</td><td>Efficient, limited to handcrafted features.</td></tr><tr><td>MindFlow [16]</td><td>Sup.</td><td>Yes</td><td>NR</td><td>No</td><td>No</td><td>Yes</td><td>CNN+BiLSTM; platform-dependent.</td></tr><tr><td>Fed. GAN/Ensemble [17]</td><td>Fed.</td><td>Yes</td><td>NR</td><td>Partial</td><td>Few-shot</td><td>Sometimes</td><td>FL; sync/global aggregation required.</td></tr><tr><td>Fed. IF Opt. [18]</td><td>Fed.</td><td>Yes</td><td>NR</td><td>Partial</td><td>Few-shot</td><td>Sometimes</td><td>FL, hyperparam-sensitive.</td></tr><tr><td>DACAD [5]</td><td>Unsup./DA</td><td>Yes</td><td>NR</td><td>Yes</td><td>Zero/Few</td><td>Often</td><td>Contrastive domain adaptation for mul- tivariate time-series AD; strong base- line but not designed around destination- centric loT flow segmentation, variable-</td></tr><tr><td>This work</td><td>Unsup.</td><td>Yes</td><td></td><td></td><td>Zero/Few</td><td></td><td>length LSTM-VAE reconstruction, or strict no-target-loss inference. Strict zero-shot inference; dual alignment;</td></tr></table>

![](images/4409de8f5bd34c969868a64be0d89b7182db833af9c678495551daf97311e302.jpg)  
Figure 1: Overall architecture of the proposed Contrastive Adversarially-Adaptive LSTM-VAE framework. The model is trained using two unlabeled source datasets to learn domain-invariant and semantically structured latent representations. Adversaria learning minimizes the domain discrepancy through a gradient reversal layer and domain classifier, while contrastive loss refines the latent space geometry. Adaptors normalize cross-domain input/output formats. At inference, the model generalizes to the unseen target domain in a zero-shot manner using reconstruction error and latent deviations for anomaly detection.

Our framework builds upon a Variational Autoencoder (VAE) augmented with Long Short-Term Memory (LSTM) layers to efectively capture temporal dependencies in multivariate time-series data. To address domain shifts across different network trafic distributions, we introduce lightweight input and output adaptor layers that serve as transformation modules, aligning feature representations between source and target domains.

As shown in Figure 1, the model is trained on two distinct, unlabeled network trafic datasets. The goal is to learn representations that are invariant to dataset-specific characteristics while capturing fundamental patterns in temporal dynamics and cross-feature correlations. This enables the model to generalize based on intrinsic structure rather than domain-dependent artifacts.

The training procedure begins with unsupervised pretraining on a large-scale source dataset, where the Variational Autoencoder (VAE) learns to reconstruct benign trafic patterns and encode variable-length input sequences into compact latent representations. To minimize the domain gap between the source and auxiliary datasets, adversarial domain alignment is applied. Specifically, a domain discriminator is trained to identify the origin of encoded samples, while the encoder is concurrently optimized in an adversarial manner to mislead the discriminator, thereby promoting the learning of domain-invariant latent features.

In addition, to develop a more generalizable model that is also capable of zero-shot adaptation, it is essential to enhance the semantic structure of the latent space. This facilitates the model’s ability to extract and retain meaningful knowledge even when input sequences vary in length and no labeled or unlabeled samples from the target domain are available during training. To achieve this, we incorporate a contrastive learning objective, which encourages the model to learn discriminative and semantically meaningful representations under such constraints. Specifically, the encoder is guided to cluster similar patterns (positive pairs) and separate dissimilar ones (negative pairs) based on cosine similarity, even when they originate from diferent domains. This dual training strategy—adversarial for alignment and contrastive for structure—produces a latent space that is both generalizable and discriminative.

After training, the model is deployed to a previously unseen target domain in a zero-shot manner, meaning no retraining or target-domain labels are used. Anomalies are detected based on reconstruction errors and deviations from the latent distribution learned during training on source domains.

To provide a structured overview of our approach, we first formalize the anomaly detection problem and the challenges of domain adaptation in time-series data. We then detail the architecture and components of our framework, including the LSTM-VAE core, domain-specific adaptor layers, and the integration of adversarial and contrastive learning mechanisms.

## 3.1. Problem Definition

We model IoT network trafic as a multivariate time series, where each observation is a feature vector recorded at a discrete time step. Formally, a trafic sequence is denoted as

$$
{ \bf X } = ( x _ { 1 } , x _ { 2 } , \ldots , x _ { T } ) \in \mathbb { R } ^ { T \times d } ,
$$

where � is the sequence length and � is the number of trafic features. In flow-level network monitoring, these features may include packet counts, flow duration, inter-arrival statistics, throughput, byte counts, protocol indicators, or other trafic descriptors. A collection of such features over time forms a multivariate sequence that captures both temporal dependencies and cross-feature correlations in network behavior.

This work addresses anomaly detection under heterogeneous cross-domain IoT deployments. Let $\mathcal { D } _ { s }$ denote a primary source domain, $\scriptstyle { \mathcal { D } } _ { a }$ denote an auxiliary source domain used for domain alignment, and $D _ { t }$ denote a held-out target domain. The source and auxiliary domains may come from diferent network environments, devices, protocols, or attack distributions. The target domain represents an unseen deployment whose data distribution is not available during training. The goal is to learn an anomaly detector from $\mathcal { D } _ { s }$ and $D _ { a }$ that generalizes to $D _ { t }$ without target-domain labels, target-domain fine-tuning, or target-domain loss computation.

The proposed framework follows a strict zero-shot protocol. In the first stage, an LSTM-based VAE is trained on the primary source domain to model normal temporal behavior through sequence reconstruction and latent-space regularization. In the second stage, samples from the source and auxiliary domains are used to improve transferability through adversarial domain alignment and contrastive representation learning. The adversarial objective encourages the encoder to suppress domain-specific artifacts between $\mathcal { D } _ { s }$ and $D _ { a } ,$ while the contrastive objective structures the latent space by pulling similar temporal patterns closer and pushing dissimilar patterns apart. Importantly, the held-out target domain $D _ { t }$ is not used in either stage.

After training, the learned encoder–decoder model is deployed directly on $D _ { t }$ . For each target sequence, the model computes an anomaly score from reconstruction error and latent deviation relative to the source-learned normal-behavior representation. A sequence is classified as anomalous when its score exceeds a predefined threshold. Target-domain labels are used only for final evaluation metrics and are never used for training, alignment, threshold selection, or model adaptation. This formulation enables privacy-preserving and deployment-oriented anomaly detection in heterogeneous IoT environments where collecting labeled target data or retraining the model for every new domain is impractical.

## 3.2. Sequence-Based VAE

We employ a sequence-based Variational Autoencoder (VAE) architecture to model multivariate time-series data in IoT environments. The goal is to encode an observed sequence � into a latent representation � while retaining its essential structural and temporal features. The VAE models the joint probability distribution of observed variables � and latent variables $z ,$ assuming a standard Gaussian prior over �:

$$
p ( z ) = \mathcal { N } ( z ; 0 , I )\tag{1}
$$

The data � is generated conditionally through a neural decoder $p _ { \theta } ( x | z )$ , but the true posterior $p ( z | x )$ is intractable. To approximate it, we use an encoder $q _ { \phi } ( z | x )$ parameterized as a Gaussian with learned mean and variance:

$$
q _ { \phi } ( z | x ) = \mathcal { N } ( z ; \mu ( x ) , \sigma ^ { 2 } ( x ) )\tag{2}
$$

where the approximate posterior takes the form:

$$
q _ { \phi } ( z | x ) = \frac { 1 } { \sqrt { 2 \pi \sigma ^ { 2 } ( x ) } } \exp { \left( - \frac { ( z - \mu ( x ) ) ^ { 2 } } { 2 \sigma ^ { 2 } ( x ) } \right) }\tag{3}
$$

To enable backpropagation through sampling, we apply the reparameterization trick:

$$
\begin{array} { r } { z = \mu ( x ) + \sigma ( x ) \cdot \epsilon , \quad \epsilon \sim \mathcal { N } ( 0 , I ) } \end{array}\tag{4}
$$

In our approach, each input sequence $x = \{ x _ { 1 } , x _ { 2 } , \ldots , x _ { T } \}$ representing a series of network trafic observations, is fed into the model. Instead of treating each time step independently, we use an LSTM to capture temporal dependencies across the sequence. The encoder LSTM processes the entire sequence and computes the final hidden state $h _ { T }$ , which summarizes it. The latent variables are derived as:

$$
h _ { t } = f _ { \mathrm { L S T M } } ( h _ { t - 1 } , x _ { t } )\tag{5}
$$

$$
\mu _ { \phi } ( x ) = W _ { \mu } h _ { T } + b _ { \mu }\tag{6}
$$

$$
\log \sigma _ { \phi } ^ { 2 } ( x ) = W _ { \sigma } h _ { T } + b _ { \sigma }\tag{7}
$$

The latent vector � is sampled using the learned parameters:

$$
\boldsymbol { z } = \boldsymbol { \mu } _ { \phi } ( \boldsymbol { x } ) + \boldsymbol { \sigma } _ { \phi } ( \boldsymbol { x } ) \cdot \boldsymbol { \epsilon } , \quad \boldsymbol { \epsilon } \sim \mathcal { N } ( 0 , I )\tag{8}
$$

The decoder is also an LSTM network and reconstructs the sequence one step at a time. Given � and the previously reconstructed step $\hat { x } _ { t - 1 }$ , each next step is generated as:

$$
\hat { x } _ { t } = f _ { \mathrm { d e c o d e r } } ( z , \hat { x } _ { t - 1 } )\tag{9}
$$

This sequence-based formulation allows the model to learn both short- and long-term temporal patterns in IoT trafic data. By encoding the full sequence into a latent space and reconstructing it iteratively, the model builds a nuanced understanding of normal trafic behavior. Deviations in the reconstruction signal potential anomalies, enabling efective detection of both sudden and gradual attacks across varied environments.

## 3.3. Variable-Length Sequence Adaptation

Real IoT trafic rarely arrives as uniform fixed-length sequences. Flow durations vary across devices, protocols, sampling rates, and attack behaviors. A fixed-window detector can therefore introduce two practical problems: short flows may be padded with little useful context, while long flows may be truncated before slow or gradual anomalies become visible. Both efects can degrade detection performance and delay operational response.

To address this issue, the proposed framework supports variable-length trafic sequences. Each sequence is encoded by the LSTM encoder into a compact latent representation, allowing the model to summarize temporal behavior without requiring all inputs to share the same duration. This makes the detector more suitable for heterogeneous IoT environments, where attacks may appear as short bursts, repeated scans, or long-duration low-rate deviations.

Because shorter sequences may contain limited temporal context, variable-length modeling is combined with adversarial and contrastive latent learning. Adversarial alignment reduces domain-specific artifacts between the primary and auxiliary source domains, while contrastive learning encourages similar temporal patterns to remain close in the latent space. Together, these mechanisms help the model preserve useful behavioral structure across diferent sequence lengths and deployment domains.

## 3.4. Adversarial and Contrastive Learning for Domain-Invariant Representations

To improve cross-domain generalization, we integrate adversarial domain alignment and cosine-based contrastive learning into the latent-space training stage. Let $\mathcal { D } _ { s }$ denote the primary source domain and $\mathcal { D } _ { a }$ denote the auxiliary source domain used for alignment. The held-out target domain $D _ { t }$ is not used during this stage. The goal is to learn an encoder � that suppresses domain-specific artifacts between $\mathcal { D } _ { s }$ and $\scriptstyle { \mathcal { D } } _ { a }$ while preserving temporal structures that are useful for anomaly detection in unseen deployments.

## 3.4.1. Adversarial Domain Alignment

We introduce a domain discriminator � that predicts whether a latent representation originates from the primary source domain $\mathcal { D } _ { s }$ or the auxiliary source domain $D _ { a }$ . The discriminator is trained to distinguish the two domains, while the encoder is trained through a Gradient Reversal Layer (GRL) to make the two latent distributions dificult to separate. This adversarial objective encourages the encoder to remove domain-specific nuisance factors and retain features that are more transferable across environments.

The adversarial loss is defined as

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { a d v } } ( E , D ) = - \mathbb { E } _ { x _ { s } \sim D _ { s } } \left[ \log D ( E ( x _ { s } ) ) \right] } \\ & { \qquad - \mathbb { E } _ { x _ { a } \sim D _ { a } } \left[ \log \left( 1 - D ( E ( x _ { a } ) ) \right) \right] , } \end{array}\tag{10}
$$

where $x _ { s }$ and $x _ { a }$ denote samples from the primary and auxiliary source domains, respectively. The discriminator minimizes $\mathcal { L } _ { \mathrm { a d v } } .$ , whereas the encoder receives the reversed gradient through the GRL, which makes the latent representations less domain-identifiable.

## 3.4.2. Cosine-Based Contrastive Learning

Adversarial alignment alone may remove domain-specific information but does not necessarily impose a useful semantic geometry on the latent space. Therefore, we also apply cosine-based contrastive learning to encourage similar temporal patterns to remain close while dissimilar patterns are separated. For each anchor representation $z _ { i } ^ { a } .$ , we define a positive representation $z _ { i } ^ { p }$ and a negative representation $z _ { i } ^ { n }$ The contrastive loss is

$$
\mathcal { L } _ { \mathrm { c o n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ 1 - \cos ( z _ { i } ^ { a } , z _ { i } ^ { p } ) + \operatorname* { m a x } \left( 0 , \cos ( z _ { i } ^ { a } , z _ { i } ^ { n } ) - m \right) \right] ,\tag{11}
$$

where cos(⋅, ⋅) denotes cosine similarity and � is a margin controlling the minimum desired separation between anchornegative pairs. This objective improves latent-space organization by reducing intra-pattern variation and increasing separation between dissimilar temporal behaviors.

## 3.4.3. Combined Objective

The final phase–2 training objective combines adversarial alignment and contrastive structuring:

$$
\mathcal { L } _ { \mathrm { p h a s e 2 } } = \lambda _ { \mathrm { a d v } } \mathcal { L } _ { \mathrm { a d v } } + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } ,\tag{12}
$$

where $\lambda _ { \mathrm { a d v } }$ and $\lambda _ { \mathrm { c o n } }$ control the relative strength of domain confusion and contrastive latent organization. The target domain $D _ { t }$ is excluded from this optimization and is used only during final zero-shot evaluation. Thus, the learned representation is encouraged to be transferable without requiring target-domain labels, target-domain loss computation, or target-side fine-tuning.

## 3.5. Domain-Specific Adaptors for Variability Handling

IoT datasets often difer in feature scale, feature ordering, measurement units, protocol composition, and device behavior. Directly feeding such heterogeneous inputs into a shared sequence model can cause the encoder to learn dataset-specific artifacts rather than transferable temporal structure. To reduce this efect, we introduce lightweight domain-specific adaptor layers around the shared LSTM-VAE.

For a sample $x _ { i } ^ { ( k ) }$ from domain $D _ { k } ,$ , where $k \in \{ s , a \}$ during training, the encoder adaptor ${ A } _ { e } ^ { ( k ) }$ maps the input sequence into a shared feature space:

$$
\tilde { x } _ { i } ^ { ( k ) } = A _ { e } ^ { ( k ) } ( x _ { i } ^ { ( k ) } ) .\tag{13}
$$

The transformed sequence $\tilde { x } _ { i } ^ { ( k ) }$ is then passed to the shared LSTM encoder, which produces the latent representation $z _ { i } ^ { ( k ) }$ . This design allows the source and auxiliary domains to be normalized before entering the shared latent model.

Similarly, after decoding, a lightweight decoder adaptor ${ A } _ { d } ^ { ( k ) }$ maps the decoder output back to the corresponding domain-specific feature space:

$$
\hat { x } _ { i } ^ { ( k ) } = A _ { d } ^ { ( k ) } ( \hat { \tilde { x } } _ { i } ^ { ( k ) } ) .\tag{14}
$$

The adaptor layers therefore act as interface modules that absorb domain-specific feature variability, while the shared LSTM-VAE focuses on learning transferable temporal dynamics.

During strict zero-shot inference, no target-specific adaptor is trained using target data. The learned shared encoder– decoder is applied directly to the held-out target domain, and the anomaly decision is computed from the reconstruction and latent-deviation score. This preserves the zero-shot protocol while still allowing the model to benefit from source–auxiliary adaptor-based alignment during training.

## 3.6. Loss Functions and Training Phases

The proposed framework is trained in two stages. Phase 1 learns a reconstruction-based normal-behavior model from the primary source domain. Phase 2 improves cross-domain transferability by aligning the primary source domain with an auxiliary source domain using adversarial and contrastive objectives. The held-out target domain is excluded from both training stages and is used only for final zero-shot evaluation.

## 3.6.1. Phase 1: VAE Pretraining on the Primary Source Domain

In the first stage, the LSTM-VAE is pretrained on the primary source domain $\mathcal { D } _ { s }$ to model normal temporal behavior. Given an input sequence $x _ { i } .$ , the encoder produces the approximate posterior parameters $\mu _ { i }$ and $\sigma _ { i } .$ , samples a latent vector $z _ { i }$ through the reparameterization trick, and the decoder reconstructs the sequence as $\hat { x } _ { i }$ . The reconstruction loss is defined as

$$
\mathcal { L } _ { \mathrm { r e c } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \big \| \boldsymbol { x } _ { i } - \boldsymbol { \hat { x } } _ { i } \big \| _ { 2 } ^ { 2 } .\tag{15}
$$

The Kullback–Leibler divergence regularizes the approximate posterior $q _ { \phi } ( z | x )$ toward the standard Gaussian prior $p ( z ) = \mathcal { N } ( 0 , I )$

$$
\mathcal { L } _ { \mathrm { K L } } = - \frac { 1 } { 2 } \sum _ { j = 1 } ^ { d _ { z } } \left( 1 + \log \sigma _ { i j } ^ { 2 } - \mu _ { i j } ^ { 2 } - \sigma _ { i j } ^ { 2 } \right) ,\tag{16}
$$

where $d _ { z }$ is the latent dimension. The Phase 1 objective is

$$
\mathcal { L } _ { \mathrm { p h a s e 1 } } = \lambda _ { \mathrm { r e c } } \mathcal { L } _ { \mathrm { r e c } } + \lambda _ { \mathrm { K L } } \mathcal { L } _ { \mathrm { K L } } ,\tag{17}
$$

where $\lambda _ { \mathrm { { r e c } } }$ and $\lambda _ { \mathrm { K L } }$ control the relative contributions of reconstruction fidelity and latent regularization.

## 3.6.2. Phase 2: Source–Auxiliary Alignment

In the second stage, we improve transferability using the primary source domain $\mathcal { D } _ { s }$ and an auxiliary source domain $D _ { a } .$ . The target domain $D _ { t }$ is not used in this phase. A domain discriminator � attempts to distinguish whether an encoded representation comes from $\mathcal { D } _ { s }$ or $D _ { a } ,$ while the encoder � is optimized through a Gradient Reversal Layer (GRL) to make the two domains dificult to separate. The adversarial domain loss is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { G R L } } = - \mathbb { E } _ { x _ { s } \sim \mathcal { D } _ { s } } \left[ \log D ( E ( x _ { s } ) ) \right] } \\ & { ~ - \mathbb { E } _ { x _ { a } \sim \mathcal { D } _ { a } } \left[ \log \left( 1 - D ( E ( x _ { a } ) ) \right) \right] , } \end{array}\tag{18}
$$

where $x _ { s }$ and $x _ { a }$ are samples from the primary and auxiliary source domains, respectively.

To further organize the latent space, we apply cosinebased contrastive learning. For each anchor representation $z _ { i } ^ { a }$ , let $z _ { i } ^ { p }$ denote a positive representation and $z _ { i } ^ { n }$ denote a negative representation. The contrastive loss is

$$
\mathcal { L } _ { \mathrm { c o n } } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ 1 - \cos ( z _ { i } ^ { a } , z _ { i } ^ { p } ) + \operatorname* { m a x } \left( 0 , \cos ( z _ { i } ^ { a } , z _ { i } ^ { n } ) - m \right) \right] ,\tag{19}
$$

where cos(⋅, ⋅) denotes cosine similarity and � is the margin that controls anchor-negative separation.

The Phase 2 objective is then

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p h a s e 2 } } = \lambda _ { \mathrm { G R L } } \mathcal { L } _ { \mathrm { G R L } } + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } , } \end{array}\tag{20}
$$

where $\lambda _ { \mathrm { G R L } }$ and $\lambda _ { \mathrm { c o n } }$ balance adversarial domain confusion and contrastive latent structuring.

## 3.6.3. Phase 3: Strict Zero-Shot Inference

After training, the encoder and decoder are frozen and deployed directly on the held-out target domain $D _ { t }$ . No target-domain labels, target-domain loss, target-domain finetuning, or target-domain model selection are used. For a target sequence $x _ { t } ,$ the model computes an anomaly score from the reconstruction error and latent deviation:

$$
s ( x _ { t } ) = \alpha \left\| x _ { t } - \hat { x } _ { t } \right\| _ { 2 } ^ { 2 } + ( 1 - \alpha ) \left\| z _ { t } - \bar { z } _ { s } \right\| _ { 2 } ^ { 2 } ,\tag{21}
$$

where $z _ { t } ~ = ~ E ( x _ { t } ) , ~ \bar { z } _ { s }$ is the mean latent representation estimated from normal source-domain training samples, and $\alpha \in [ 0 , 1 ]$ controls the relative contribution of reconstruction and latent deviation. A target sequence is classified as anomalous when $s ( x _ { t } )$ exceeds a predefined threshold. Target labels are used only to compute final evaluation metrics.

## 3.7. Evaluation Datasets

In this study, we utilize six distinct IoT-related datasets originating from diverse application domains—namely, industrial IoT, general-purpose enterprise networks, military automation, academic testbeds, and smart home automation—to comprehensively evaluate the efectiveness and generalizability of our proposed anomaly detection framework.

• CICIDS 2018 Dataset: The CICIDS 2018 dataset [19], curated by the Canadian Institute for Cybersecurity (CIC) in collaboration with the Communications Security Establishment (CSE), was generated using a simulated enterprise network setup. It spans ten days of realistic network trafic, totaling over 16 million flow-based entries. Approximately 83% of the data is benign, while 17% comprises various attack categories including brute force, denial-of-service (DoS), botnet activity, web attacks, and infiltration. Each record contains more than 80 trafic flow features extracted via CICFlowMeter. This dataset serves as a comprehensive benchmark for evaluating intrusion detection systems in general network environments. All recorded features are monitored to examine the robustness of our method under diverse trafic behaviors, enabling an in-depth assessment of model adaptability across heterogeneous network conditions.

• WUSTL-IIOT-2021 Dataset: The WUSTL-IIOT-2021 dataset [20] originates from an Industrial IoT (IIoT) testbed designed to monitor water storage tanks. Spanning 53 hours of operation, it includes 1,194,464 entries, capturing critical network metrics such as packet drops and flow durations. The dataset is predominantly benign (93%), with 7% of records corresponding to attack types such as denial-of-service (89.98%), reconnaissance (9.46%), SQL injection (0.31%), and backdoor access (0.25%). Consistent with prior studies, we track all extracted features to analyze the efectiveness of the proposed method when applied to industrial trafic with varying characteristics.

• ACI-IoT-2023 Dataset: The ACI-IoT-2023 dataset [21] was collected within a simulated military IoT environment that replicates real-world home automation scenarios. It contains 742,758 trafic entries recorded over five days, of which 95% are malicious. The attack composition includes reconnaissance (74%), brute force (1%), and denial-of-service (25%), encompassing subcategories such as scanning and sweeping. This dataset ofers valuable insights into the cybersecurity vulnerabilities inherent in military-oriented IoT systems. We utilize all available features to evaluate the proposed method’s performance in handling trafic patterns with distinct operational contexts.

• CICIDS 2017 Dataset: A predecessor to CICIDS 2018, this dataset [22] includes both benign and common, up-to-date attacks. It contains 2,830,743 total flow entries, of which approximately 80.3% are benign. The remaining 19.7% of malicious trafic is composed of several attack types. The most prevalent attacks are Denial-of-Service (DoS/DDoS), accounting for the vast majority of malicious instances, followed by Port Scanning. Other implemented attacks include Brute Force (FTP and SSH), Web Attacks (Brute Force, XSS, SQL Injection), Infiltration, Botnet, and Heartbleed. The dataset is used in source–target–adversary combinations to evaluate robustness across diferent temporal captures and protocol variations within CIC environments.

• UNSW-NB15 Dataset: Developed by the Australian Cyber Security Centre, this dataset [23] includes both real benign and synthetic attack flows from a total of 2,540,044 records. Approximately 87.4% of the trafic is benign, while the other 12.6% is malicious. The attack composition consists of nine types, dominated by Generic attacks (67.1%) and Exploits (13.9%). The remaining categories are Fuzzers (7.5%), DoS (5.1%), Reconnaissance (4.3%), Analysis (0.8%), Backdoors (0.7%), Shellcode (0.5%), and Worms (0.1%). It is employed as both a source and target to evaluate crossprotocol and attack-style generalization.

• TON-IoT Dataset: Provided by the Cyber Range Lab at UNSW, TON-IoT [24] is a smart home-oriented dataset with over 22 million telemetry and network event records. Within the network trafic portion, approximately 64.8% of entries are benign, and 35.2% are malicious. The malicious trafic is heavily skewed towards reconnaissance activities, with Scanning attacks making up 48.7% of all malicious events. Denialof-Service is also highly prevalent, with DDoS (29.0%) and DoS (21.6%) attacks forming the next largest categories. The dataset also contains smaller numbers of Backdoor (0.34%), Injection (0.19%), Ransomware (0.10%), XSS (0.04%), Password (0.01%), and Manin-the-Middle (MITM) attacks. It is used as a target domain in several experiments, revealing generalization challenges in more constrained settings.

Together, these six datasets provide a heterogeneous evaluation suite spanning enterprise network trafic, industrial IoT, military-oriented IoT, academic intrusion-detection testbeds, and smart-home IoT environments. CICIDS2017 and CICIDS2018 represent general-purpose enterprise network settings with diverse benign and malicious flow patterns; WUSTL-IIoT-2021 captures industrial IoT trafic with operationally structured benign behavior and a smaller set of attack classes; ACI-IoT-2023 reflects a military-oriented IoT environment with a highly attack-dominant distribution; UNSW-NB15 introduces additional protocol and attackstyle diversity; and TON-IoT provides smart-home IoT trafic with substantial reconnaissance and denial-of-service activity.

In our evaluation framework, these datasets are used in source–auxiliary–target combinations to test cross-domain generalization under strict zero-shot conditions. For each experiment, the model is trained using a primary source domain $\mathcal { D } _ { s }$ and an auxiliary source domain $\mathcal { D } _ { a }$ for adversarial and contrastive alignment, while the target domain $D _ { t }$ is held out entirely during training. No target-domain samples, labels, losses, or fine-tuning steps are used before inference. The target dataset is used only for final metric computation. This protocol allows us to evaluate both successful transfer cases and failure modes caused by domain shift, class imbalance, and insuficient source-domain coverage.

## 4. Model Evaluation and Analysis

## 4.1. Experimental Protocol

In all experiments, the held-out target domain is used only for final evaluation. Target labels are not used during model training, adversarial alignment, contrastive learning, threshold selection, or model selection. Performance is reported using accuracy, Matthews correlation coeficient (MCC), recall, precision, F1-score, and false-positive rate (FPR). This protocol is intended to measure strict zero-shot transfer rather than target-adaptive performance.

The confusion matrices in Figs. 2–4 reinforce these observations. In the CICIDS-only scenario, true positives are generally high, indicating good recognition of abnormal patterns. However, moderate false positives suggest some overgeneralization to benign classes. The WUSTLonly model achieves a similar true negative rate but sufers from increased false positives and false negatives, due to the absence of explicit attack semantics during training. The combined scenario demonstrates the most balanced classification performance with minimal false negatives (66) and false positives (330), aligning with its top accuracy. This confirms that the adversarial framework efectively aligns heterogeneous source domains to construct a more resilient decision boundary in the target space.

## 4.2. Zero-Shot Inference Protocol (No Target Loss)

In the final phase, the model operates under a strict zeroshot setting on the target domain: no target data (labeled or

unlabeled) is used and no loss function is computed in this phase. Decisions are made using reconstruction errors and latent deviations only.

## 4.3. Cross-Domain Benchmarking and Latent Alignment via t-SNE

We expanded the evaluation to six datasets: CICIDS2017, CICIDS2018, UNSW–NB15, TON–IoT, ACI–IoT–2023, and WUSTL-IIoT-2021/WUSTL-IIoT. These datasets form 44 source–auxiliary–target combinations, where two domains are used during training and alignment and the remaining domain is held out for zero-shot evaluation. The cross-domain results are reported in Tables 2 and 3. Figure 5 complements these numerical results by showing the phase–2 latent geometry, i.e., the aligned � space after VAE encoding with contrastive and adversarial objectives, for thirteen representative source-domain pairs.

Several regularities emerge from the latent visualizations. First, source pairs involving CICIDS2017, CICIDS2018, NB15, or WUSTL-IIoT-2021 often form smooth low-dimensional manifolds, suggesting that the adversarial and contrastive objectives reduce visible domain separation in the learned latent space. Second, when TON participates as a source, the embedding often becomes highly smooth and nearly one-dimensional. Although this indicates strong geometric regularization, it may also suggest reduced latent diversity, which can limit coverage of unseen target-domain modes. Third, source pairs involving ACI sometimes retain more scattered or multi-lobed structures, indicating that complete cross-domain mixing is more dificult when military-IoT trafic is involved. These visual patterns suggest that latent compactness alone is not suficient; the aligned representation must also preserve enough variability to support unseen target domains.

The numerical results in Tables 2 and 3 show that the proposed method performs strongly for several CI-CIDS2017, CICIDS2018, and NB15 target settings. When CICIDS2017 is used as the target, source pairs such as ACI+TON, ACI+WUSTL-IIoT-2021, and CICIDS2018+ACI achieve perfect or near-perfect accuracy, MCC, recall, precision, and F1. Similar behavior is observed when CI-CIDS2018 is the target, especially for source pairs involving CICIDS2017, NB15, TON, or WUSTL-IIoT-2021.

By contrast, the results also reveal systematic weaknesses in some target domains. Transfers to TON–IoT often produce low precision and F1 despite high recall, indicating that the model detects many anomalous samples but also generates many false positives. This behavior appears across several TON target cases, especially when the source pairs include ACI, CICIDS2017, CICIDS2018, NB15, or WUSTL-IIoT-2021. A similar failure mode appears when ACI is the unseen target for several source pairs, where the model collapses toward poor recall and low MCC. These results indicate that strict zero-shot transfer remains challenging when the held-out target contains trafic modes or attack distributions that are poorly covered by the aligned source domains.

Table 3  
Table 2
<table><tr><td>Source</td><td>Target</td><td>ACC</td><td>MCC</td><td>Recall</td><td>Precision</td><td>F1</td><td>FPR</td></tr><tr><td>ACI+TON</td><td>CICIDS2017</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>ACI+TON</td><td>CICIDS2018</td><td>0.833</td><td>0.707</td><td>1.000</td><td>0.750</td><td>0.857</td><td>0.333</td></tr><tr><td>ACI+TON</td><td>NB15</td><td>0.812</td><td>0.584</td><td>0.781</td><td>1.000</td><td>0.877</td><td>0.000</td></tr><tr><td>ACI+WUSTL-IIoT-2021</td><td>CICIDS2017</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>ACI+WUSTL-IIoT-2021</td><td>CICIDS2018</td><td>0.854</td><td>0.737</td><td>1.000</td><td>0.781</td><td>0.877</td><td>0.304</td></tr><tr><td>ACI+WUSTL-IIoT-2021</td><td>NB15</td><td>0.833</td><td>0.632</td><td>0.800</td><td>1.000</td><td>0.889</td><td>0.000</td></tr><tr><td>ACI+WUSTL-IIoT-2021</td><td>TON</td><td>0.479</td><td>0.227</td><td>0.889</td><td>0.250</td><td>0.390</td><td>0.615</td></tr><tr><td>CICIDS2017+ACI</td><td>CICIDS2018</td><td>0.875</td><td>0.769</td><td>1.000</td><td>0.812</td><td>0.897</td><td>0.273</td></tr><tr><td>CICIDS2017+ACI</td><td>NB15</td><td>0.812</td><td>0.584</td><td>0.781</td><td>1.000</td><td>0.877</td><td>0.000</td></tr><tr><td>CICIDS2017+ACI</td><td>TON</td><td>0.479</td><td>0.227</td><td>0.889</td><td>0.250</td><td>0.390</td><td>0.615</td></tr><tr><td>CICIDS2017+TON</td><td>ACI</td><td>0.333</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.667</td></tr><tr><td>CICIDS2017+TON</td><td>CICIDS2018</td><td>0.958</td><td>0.913</td><td>1.000</td><td>0.938</td><td>0.968</td><td>0.111</td></tr><tr><td>CICIDS2017+TON</td><td>NB15</td><td>0.854</td><td>0.679</td><td>0.821</td><td>1.000</td><td>0.901</td><td>0.000</td></tr><tr><td>CICIDS2017+WUSTL-IIoT-2021</td><td>ACl</td><td>0.333</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.667</td></tr><tr><td>CICIDS2017+WUSTL-IIoT-2021</td><td>CICIDS2018</td><td>0.958</td><td>0.913</td><td>1.000</td><td>0.938</td><td>0.968</td><td>0.111</td></tr><tr><td>CICIDS2017+WUSTL-IIoT-2021</td><td>NB15</td><td>0.833</td><td>0.632</td><td>0.800</td><td>1.000</td><td>0.889</td><td>0.000</td></tr><tr><td>CICIDS2017+WUSTL-IIoT-2021</td><td>TON</td><td>0.479</td><td>0.227</td><td>0.889</td><td>0.250</td><td>0.390</td><td>0.615</td></tr><tr><td>CICIDS2018+ACI</td><td>CICIDS2017</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>CICIDS2018+ACI</td><td>NB15</td><td>0.833</td><td>0.632</td><td>0.800</td><td>1.000</td><td>0.889</td><td>0.000</td></tr><tr><td>CICIDS2018+ACI</td><td>TON</td><td>0.479</td><td>0.227</td><td>0.889</td><td>0.250</td><td>0.390</td><td>0.615</td></tr><tr><td>CICIDS2018+TON</td><td>ACI</td><td>0.333</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.667</td></tr><tr><td>CICIDS2018+TON</td><td>CICIDS2017</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td></tr></table>

Cross-domain zero-shot results for part 1 of the 44-experiment evaluation. Source lists the two training domains (A+B); Target is the held-out zero-shot domain.
<table><tr><td>Source</td><td>Target</td><td>ACC</td><td>MCC</td><td>Recall</td><td>Precision</td><td>F1</td><td>FPR</td></tr><tr><td>CICIDS2018+TON</td><td>NB15</td><td>0.812</td><td>0.584</td><td>0.781</td><td>1.000</td><td>0.877</td><td>0.000</td></tr><tr><td>CICIDS2018+WUSTL-IIoT-2021</td><td>ACI</td><td>0.333</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.667</td></tr><tr><td>CICIDS2018+WUSTL-IIoT-2021</td><td>CICIDS2017</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>CICIDS2018+WUSTL-IIoT-2021</td><td>NB15</td><td>0.854</td><td>0.679</td><td>0.821</td><td>1.000</td><td>0.901</td><td>0.000</td></tr><tr><td>CICIDS2018+WUSTL-IIoT-2021</td><td>TON</td><td>0.479</td><td>0.227</td><td>0.889</td><td>0.250</td><td>0.390</td><td>0.615</td></tr><tr><td>NB15+ACI</td><td>CICIDS2017</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>NB15+ACl</td><td>CICIDS2018</td><td>0.938</td><td>0.874</td><td>1.000</td><td>0.906</td><td>0.951</td><td>0.158</td></tr><tr><td>NB15+ACl</td><td>TON</td><td>0.479</td><td>0.227</td><td>0.889</td><td>0.250</td><td>0.390</td><td>0.615</td></tr><tr><td>NB15+TON</td><td>ACI</td><td>0.333</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.667</td></tr><tr><td>NB15+TON</td><td>CICIDS2017</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>NB15+TON</td><td>CICIDS2018</td><td>0.958</td><td>0.913</td><td>1.000</td><td>0.938</td><td>0.968</td><td>0.111</td></tr><tr><td>NB15+WUSTL-IIoT-2021</td><td>ACI</td><td>0.333</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.667</td></tr><tr><td>NB15+WUSTL-IIoT-2021</td><td>CICIDS2017</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>NB15+WUSTL-IIoT-2021</td><td>CICIDS2018</td><td>0.958</td><td>0.913</td><td>1.000</td><td>0.938</td><td>0.968</td><td>0.111</td></tr><tr><td>NB15+WUSTL-IIoT-2021</td><td>TON</td><td>0.479</td><td>0.227</td><td>0.889</td><td>0.250</td><td>0.390</td><td>0.615</td></tr><tr><td>TON+ACI</td><td>CICIDS2017</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>TON+ACI</td><td>CICIDS2018</td><td>0.896</td><td>0.802</td><td>1.000</td><td>0.844</td><td>0.915</td><td>0.238</td></tr><tr><td>TON+ACI</td><td>NB15</td><td>0.812</td><td>0.584</td><td>0.781</td><td>1.000</td><td>0.877</td><td>0.000</td></tr><tr><td>TON+WUSTL-IIoT-2021</td><td>ACl</td><td>0.333</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.667</td></tr><tr><td>TON+WUSTL-IIoT-2021</td><td>CICIDS2017</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td></tr><tr><td>TON+WUSTL-IIoT-2021</td><td>CICIDS2018</td><td>0.958</td><td>0.913</td><td>1.000</td><td>0.938</td><td>0.968</td><td>0.111</td></tr><tr><td>TON+WUSTL-IIoT-2021</td><td>NB15</td><td>0.854</td><td>0.679</td><td>0.821</td><td>1.000</td><td>0.901</td><td>0.000</td></tr></table>

Cross-domain P3 results for part 2 of the 44-experiment subset from the previous runs. Source lists the two training domains (A+B); Target is the held-out zero-shot domain.

Overall, the cross-domain benchmark shows that the proposed adversarial and contrastive LSTM–VAE can learn useful transferable representations across several heterogeneous IoT and network-security datasets, especially when the target domain shares latent structure with the source and auxiliary domains. At the same time, the weaker TON and ACI target cases highlight an important limitation of strict zero-shot anomaly detection: alignment between the available source domains does not guarantee suficient latent support for every unseen deployment. This motivates the DACAD comparison in the next subsection and suggests that future work should incorporate uncertainty-aware rejection, broader source-domain coverage, or lightweight adaptoronly calibration when a small amount of target data is available.

## 4.4. Comparison with DACAD

DACAD [5] is the closest conceptual baseline to our work because it also addresses anomaly detection in multivariate time series under domain shift using contrastive domain adaptation. Its main objective is to learn transferable representations by aligning source and target domains through contrastive learning, thereby improving anomaly detection when the deployment distribution difers from the training distribution. This makes DACAD a strong reference point for evaluating whether the proposed Contrastive Adversarially-Adaptive LSTM-VAE provides additional value beyond contrastive domain adaptation alone.

Despite this conceptual similarity, our framework differs from DACAD in four important ways. First, DACAD is primarily designed around domain-adaptive multivariate time-series anomaly detection, whereas our framework is designed for strict zero-shot IoT trafic anomaly detection. In our final inference phase, no target-domain samples, labels, target loss, or target-side fine-tuning are used. The model performs detection only through reconstruction error and latent deviation from the source-learned normal-behavior manifold. Second, DACAD focuses on contrastive domain adaptation, while our method combines contrastive latent structuring with adversarial domain alignment through a gradient reversal mechanism. This dual objective simultaneously encourages semantic separation between dissimilar temporal patterns and domain invariance across heterogeneous IoT trafic sources. Third, our model explicitly supports variable-length flow sequences through the LSTM-VAE formulation, which is important for realistic IoT traffic where flows difer in duration, sampling density, and attack timescale. Fourth, our destination-centric segmentation strategy groups trafic by receiver endpoint, preserving device-level communication context instead of relying only on generic temporal windows.

To compare against DACAD, we use the same crossdomain evaluation setting adopted in the previous subsection. Multiple datasets are used as source/adversarial training domains, while the held-out target domain is never exposed during training. The comparison therefore evaluates generalization under domain shift rather than in-domain detection. This is particularly important because IoT anomaly detection systems are often deployed in environments whose device composition, protocols, trafic intensity, and attack distribution difer from those observed during training.

The comparison shows a mixed outcome rather than a uniform advantage for either method. The proposed framework performs particularly well when CICIDS2017 or CI-CIDS2018 is the held-out target, often reaching near-perfect accuracy in these transfer settings. In contrast, DACAD achieves stronger accuracy in several NB15, TON-IoT, and ACI-IoT target configurations. This indicates that the proposed reconstruction-based adversarial LSTM-VAE is effective when the unseen target shares temporal and flowlevel structure with the aligned source domains, whereas DACAD can be more competitive when contrastive domain adaptation provides better direct representation transfer.

These results also clarify the role of the proposed design choices. Our method emphasizes strict no-target-loss inference, variable-length sequence reconstruction, adversarial source-domain alignment, and destination-centric flow modeling. This design is especially efective when the heldout target shares temporal and flow-level structure with the aligned source domains, as seen in several CICIDS2017 and CICIDS2018 target settings. DACAD, by contrast, is a strong contrastive domain-adaptation baseline and can outperform the proposed method in several NB15, TON– IoT, and ACI–IoT target configurations, where direct contrastive representation transfer appears to handle the domain shift more efectively. Therefore, the comparison should be interpreted as complementary rather than as a one-sided improvement: the proposed framework ofers a deploymentoriented strict zero-shot detector for IoT trafic anomaly detection, while DACAD remains highly competitive in dificult cross-domain accuracy comparisons.

## 5. Conclusion

This paper introduced a contrastive adversarially adaptive LSTM–VAE framework for zero-shot anomaly detection in multivariate IoT network trafic. By combining sequence-based variational reconstruction with adversarial source-domain alignment and cosine-based contrastive latent structuring, the framework addresses cross-domain shifts without requiring target-side labels, target-domain loss computation, or target-side fine-tuning. The integration of lightweight domain adaptors and destination-centric trafic segmentation further allows the model to accommodate variable-length flows while preserving device-level communication context.

Extensive evaluation across six diverse IoT and networksecurity benchmarks spanning 44 transfer scenarios demonstrates that the proposed framework achieves strong zeroshot generalization in several structurally compatible crossdomain settings, particularly for CICIDS2017 and CICIDS201 targets. At the same time, the results expose systematic degradation in highly shifted smart-home and military-IoT target domains when the aligned source domains do not provide suficient latent coverage. In particular, TON–IoT target cases often produce high recall but low precision, indicating elevated false-positive rates, while several ACI– IoT target cases show poor recall and low MCC under strict zero-shot transfer.

Comparison with DACAD shows a complementary performance pattern rather than a uniform advantage for either method. The proposed reconstruction-driven LSTM–VAE is highly efective when temporal and flow-level structures transfer across domains, whereas direct contrastive domain adaptation can be more competitive under severe distribution shift. These findings highlight both the promise and the limitations of strict zero-shot anomaly detection for heterogeneous IoT deployments.

## CRediT authorship contribution statement

Mahshid Rezakhani: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Visualization. Tolunay Seyfi: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Writing – review and editing, Visualization. Fatemeh Afghah: Conceptualization, Supervision, Project administration, Funding acquisition, Writing – review and editing.

Adversarial Zero-Shot IoT Anomaly Detection
<table><tr><td>Source</td><td>Target</td><td>Proposed ACC</td><td>DACAD ACC</td></tr><tr><td>ACI+TON</td><td>CICIDS2017</td><td>1.000</td><td>0.519</td></tr><tr><td>ACI+TON</td><td>CICIDS2018</td><td>0.833</td><td>0.415</td></tr><tr><td>ACI+TON</td><td>NB15</td><td>0.812</td><td>0.649</td></tr><tr><td>ACI+WUSTL-IIoT-2021</td><td>CICIDS2017</td><td>1.000</td><td>0.656</td></tr><tr><td>ACI+WUSTL-IIoT-2021</td><td>CICIDS2018</td><td>0.854</td><td>0.738</td></tr><tr><td>ACI+WUSTL-IIoT-2021</td><td>NB15</td><td>0.833</td><td>0.985</td></tr><tr><td>ACI+WUSTL-IIoT-2021</td><td>TON</td><td>0.479</td><td>0.966</td></tr><tr><td>CICIDS2017+ACI</td><td>CICIDS2018</td><td>0.875</td><td>0.506</td></tr><tr><td>CICIDS2017+ACI</td><td>NB15</td><td>0.812</td><td>0.895</td></tr><tr><td> $\mathsf { C l C l D S 2 0 1 7 + A C l }$ </td><td>TON</td><td>0.479</td><td>0.800</td></tr><tr><td>CICIDS2017+TON</td><td>ACI</td><td>0.333</td><td>0.485</td></tr><tr><td>CICIDS2017+TON</td><td>CICIDS2018</td><td>0.958</td><td>0.551</td></tr><tr><td>CICIDS2017+TON</td><td>NB15</td><td>0.854</td><td>0.988</td></tr><tr><td> $\mathsf { C l C I D S 2 0 1 7 + W U S T L - I l o T - } 2 0 2 1$ </td><td>ACI</td><td>0.333</td><td>0.512</td></tr><tr><td>CICIDS2017+WUSTL-IIoT-2021</td><td>CICIDS2018</td><td>0.958</td><td>0.559</td></tr><tr><td> $\mathsf { C l C I D S 2 0 1 7 + W U S T L - I l o T - } 2 0 2 1$ </td><td>NB15</td><td>0.833</td><td>0.984</td></tr><tr><td> $\mathsf { C l C I D S 2 0 1 7 + W U S T L - I l o T - } 2 0 2 1$ </td><td>TON</td><td>0.479</td><td>0.979</td></tr><tr><td> $\mathsf { C l C l D S 2 0 1 8 { + } A C l }$ </td><td>CICIDS2017</td><td>1.000</td><td>0.626</td></tr><tr><td> $\mathsf { C l C l D S 2 0 1 8 { + } A C l }$ </td><td>NB15</td><td>0.833</td><td>0.989</td></tr><tr><td>CICIDS2018+ACI</td><td>TON</td><td>0.479</td><td>0.563</td></tr><tr><td>CICIDS2018+TON</td><td>ACI</td><td>0.333</td><td>0.492</td></tr><tr><td>CICIDS2018+TON</td><td>CICIDS2017</td><td>1.000</td><td>0.600</td></tr></table>

Table 4  
Accuracy comparison between the proposed method and DACAD for part 1 of the 44-experiment P3 subset. The higher accuracy in each row is shown in bold.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Data availability

The datasets analyzed in this study are publicly available from their respective repositories. Processed data and implementation details are available from the corresponding author upon reasonable request.

## Acknowledgements

This material is based upon work supported by the National Science Foundation under Grant Numbers CNS-2318726 and CNS-2232048.

## References

[1] Z. A. Haider, A. Zeb, T. Rahman, S. K. Singh, R. Akram, A. Arishi, and I. Ullah, “A survey on anomaly detection in IoT: Techniques, challenges, and opportunities with the integration of 6G,” Computer Networks, vol. 270, p. 111484, 2025. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S1389128625004517

[2] M. M. Rahman, S. Al Shakil, and M. R. Mustakim, “A survey on intrusion detection system in IoT networks,” Cyber Security and

Applications, vol. 3, p. 100082, 2025. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S2772918424000481

[3] J. Doménech, O. León, M. S. Siddiqui, and J. Pegueroles, “Evaluating and enhancing intrusion detection systems in IoMT: The importance of domain-specific datasets,” Internet of Things, vol. 32, p. 101631, 2025. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S2542660525001453

[4] Z. Z. Darban, G. I. Webb, S. Pan, C. Aggarwal, and M. Salehi, “Deep learning for time series anomaly detection: A survey,” ACM Computing Surveys, vol. 57, no. 1, October 2024. [Online]. Available: https://doi.org/10.1145/3691338

[5] Z. Z. Darban, Y. Yang, G. I. Webb, C. C. Aggarwal, Q. Wen, S. Pan, and M. Salehi, “DACAD: Domain adaptation contrastive learning for anomaly detection in multivariate time series,” IEEE Transactions on Knowledge and Data Engineering, vol. 37, no. 8, pp. 4485–4496, 2025.

[6] L. Tang, Z. Wang, G. He, R. Wang, and F. Nie, “Perturbation guiding contrastive representation learning for time series anomaly detection,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, K. Larson, Ed. International Joint Conferences on Artificial Intelligence Organization, August 2024, pp. 4955–4963, main Track. [Online]. Available: https://doi.org/10.24963/ijcai.2024/548

[7] S. Guan, Z. He, S. Ma, and M. Gao, “Multivariate time series anomaly detection with variational autoencoder and spatial–temporal graph network,” Computers & Security, vol. 142, p. 103877, 2024. [Online]. Available: https://www.sciencedirect.com/science/article/ pii/S0167404824001780

[8] L. Tang, R. Wei, B. Xia, Y. Tang, W. Wang, Q. Huang, and Q. Chen, “Online anomaly detection in industrial IoT networks using a supervised contrastive learning-based spatiotemporal variational autoencoder,” IEEE Internet of Things Journal, 2025.

[9] H. Mancy and Q. H. Naith, “SwinIoT: A hierarchical transformerbased framework for behavioral anomaly detection in IoT-driven smart cities,” IEEE Access, vol. 13, pp. 48 758–48 774, 2025.

<table><tr><td>Source</td><td>Target</td><td>Proposed ACC</td><td>DACAD ACC</td></tr><tr><td>CICIDS2018+TON</td><td>NB15</td><td>0.812</td><td>0.891</td></tr><tr><td>CICIDS2018+WUSTL-IIoT-2021</td><td>ACI</td><td>0.333</td><td>0.418</td></tr><tr><td>CICIDS2018+WUSTL-IIoT-2021</td><td>CICIDS2017</td><td>1.000</td><td>0.691</td></tr><tr><td>CICIDS2018+WUSTL-IIoT-2021</td><td>NB15</td><td>0.854</td><td>0.904</td></tr><tr><td>CICIDS2018+WUSTL-IIoT-2021</td><td>TON</td><td>0.479</td><td>0.963</td></tr><tr><td>NB15+ACI</td><td>CICIDS2017</td><td>1.000</td><td>0.484</td></tr><tr><td>NB15+ACl</td><td>CICIDS2018</td><td>0.938</td><td>0.545</td></tr><tr><td>NB15+ACI</td><td>TON</td><td>0.479</td><td>0.603</td></tr><tr><td>NB15+TON</td><td>ACI</td><td>0.333</td><td>0.564</td></tr><tr><td>NB15+TON</td><td>CICIDS2017</td><td>1.000</td><td>0.647</td></tr><tr><td>NB15+TON</td><td>CICIDS2018</td><td>0.958</td><td>0.594</td></tr><tr><td>NB15+WUSTL-IIoT-2021</td><td>ACI</td><td>0.333</td><td>0.418</td></tr><tr><td>NB15+WUSTL-IIoT-2021</td><td>CICIDS2017</td><td>1.000</td><td>0.620</td></tr><tr><td>NB15+WUSTL-IIoT-2021</td><td>CICIDS2018</td><td>0.958</td><td>0.551</td></tr><tr><td>NB15+WUSTL-IIoT-2021</td><td>TON</td><td>0.479</td><td>0.984</td></tr><tr><td>TON+ACI</td><td>CICIDS2017</td><td>1.000</td><td>0.389</td></tr><tr><td>TON+ACI</td><td>CICIDS2018</td><td>0.896</td><td>0.410</td></tr><tr><td>TON+ACI</td><td>NB15</td><td>0.812</td><td>0.632</td></tr><tr><td>TON+WUSTL-IIoT-2021</td><td>ACI</td><td>0.333</td><td>0.449</td></tr><tr><td>TON+WUSTL-IIoT-2021</td><td>CICIDS2017</td><td>1.000</td><td>0.680</td></tr><tr><td>TON+WUSTL-IIoT-2021</td><td>CICIDS2018</td><td>0.958</td><td>0.755</td></tr><tr><td> $\mathsf { T O N + W U S T L - l l o T - } 2 0 2 1$ </td><td>NB15</td><td>0.854</td><td>0.940</td></tr></table>

## Table 5

Accuracy comparison between the proposed method and DACAD for part 2 of the 44-experiment P3 subset. The higher accuracy in each row is shown in bold.

[10] N. Shamim, M. Asim, A. I. Awad, and M. K. Khan, “Anomaly detection in internet of things system calls using a centroid-based vector-space model,” IEEE Internet ofThings Journal, 2025.

[11] L. Tan, A. Singh, W. Zhang, H. Pei, P. Zhang, P. K. Chahal, and M. Singh, “Energy-eficient tactile-driven rule configuration and anomaly detection in industrial IoT systems,” IEEE Internet ofThings Journal, 2025.

[12] H. Geng, Q. Ma, H. Chi, Z. Zhang, J. Yang, and X. Yin, “DUdetector: A dual-granularity unsupervised model for network anomaly detection,” Computer Networks, vol. 257, p. 110937, 2025.

[13] G.-Q. Zeng, Y.-W. Yang, K.-D. Lu, G.-G. Geng, and J. Weng, “Evolutionary adversarial autoencoder for unsupervised anomaly detection of industrial internet of things,” IEEE Transactions on Reliability, pp. 1–15, 2025.

[14] M. Zhang et al., “DPGLAD: An unsupervised graph learning-based anomaly detection model for industrial IoT,” Applied Intelligence, 2024.

[15] E. Tuyishime, M. Martalò, P. A. Cotfas, V. Popescu, D. T. Cotfas, and A. Rekeraho, “Resource-eficient trafic classification using feature selection for message queuing telemetry transport–internet of things network-based security attacks,” Applied Sciences, vol. 15, no. 8, p. 4252, 2025.

[16] L. Chen et al., “MindFlow: A deep learning-based intrusion detection system for IoT using CNN-BiLSTM on mindspore,” arXiv preprint arXiv:2504.17678, 2025.

[17] B. Jiang, G. Wang, X. Cui, F. Luo, and J. Wang, “Lightweight anomaly detection in federated learning via separable convolution and convergence acceleration,” Internet ofThings, vol. 30, p. 101518, 2025.

[18] A. Babu and A. Bagubali, “Federated learning with sailfish-optimized ensemble models for anomaly detection in iot edge computing environment,” IEEE Access, vol. 13, pp. 53 171–53 187, 2025.

[19] Communications Security Establishment (CSE) and Canadian Institute for Cybersecurity (CIC), “Cse-cic-ids2018: A collaborative dataset for computer network intrusion detection,” University

of New Brunswick, Tech. Rep., 2018. [Online]. Available: https://www.unb.ca/cic/datasets/ids-2018.html

[20] D. Bhamare, S. Zolanvari et al., “Cybersecurity for industrial internet of things (IIoT): A survey and a testbed,” IEEE International Conference on Computer and Communication Engineering (ICCCE), 2021. [Online]. Available: https://www.cse.wustl.edu/\~jain/iiot2/ index.html

[21] N. Bastian, D. Bierbrauer et al., “ACI IoT network trafic dataset 2023,” 2023.

[22] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, “Toward generating a new intrusion detection dataset and intrusion trafic characterization,” in Proceedings of the 4th International Conference on Information Systems Security and Privacy (ICISSP), 2018, pp. 108– 116.

[23] N. Moustafa and J. Slay, “UNSW-NB15: A comprehensive data set for network intrusion detection systems,” in Proceedings of the Military Communications and Information Systems Conference (MilCIS), 2015, pp. 1–6.

[24] N. Moustafa, M. Ahmed et al., “Data analytics-enabled intrusion detection: Evaluations of ToN\_IoT linux datasets,” in Proceedings of the 2020 IEEE 19th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom), 2020, pp. 839–846.

[25] S. Xie, L. Li, and Y. Zhu, “Anomaly detection for multivariate time series in IoT using discrete wavelet decomposition and dual graph attention networks,” Computers & Security, vol. 146, p. 104075, 2024. [Online]. Available: https://www.sciencedirect.com/science/ article/pii/S0167404824003808

[26] J. Azar, M. Al Saleh, R. Couturier, and H. Noura, “Text mining and unsupervised deep learning for intrusion detection in smart-grid communication networks,” IoT, vol. 6, no. 2, p. 22, 2025.

[27] R. Fang et al., “LTG: Learning temporal and graph structures for multivariate time-series anomaly detection in IoT networks,” Discover Internet ofThings, 2024.

![](images/bb7ebd769ce62fab72e57095d24c3cbf4c494e59080aa3fc223f3b0d41baf86b.jpg)  
Figure 2: Confusion matrix for transfer scenario: CICIDS 2018 (source) → ACI-IoT-2023 (target)

![](images/7cfbcc1f69abfb01e5a4650201c41a6867f0c9b498dd95d549361c065bb1536c.jpg)  
Figure 3: Confusion matrix for transfer scenario: WUSTL-IIoT-2021 (source) → ACI-IoT-2023 (target)

![](images/7fca4d54806b9f2f60692db514a1de9b65d7604d00c72736a7faf6c38b3e622c.jpg)  
Figure 4: Confusion matrix for transfer scenario: CICIDS 2018 + WUSTL-IIoT-2021 (source) → ACI-IoT-2023 (target)

TON+WUSTL-IIoT-2021  
![](images/c79fce9f2f738bf849c478be81b4c6ea2226181b60285638310e4a1d4352ae82.jpg)  
Figure 5: Phase 2 latent space visualization after contrastive and adversarial alignment. Each panel shows the t-SNE projection of the aligned VAE latent representations for one source domain pair.