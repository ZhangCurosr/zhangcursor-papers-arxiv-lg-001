# Cross-Area Transfer Learning for VLC-based Indoor Localization with a Transfer Efficiency Score

Masood Jan, Wafa Njima, Xun Zhang, Alexander Artemenko

Abstract—Indoor localization systems are critical for Industry-4.0 applications.Visible Light Communication (VLC)-based systems offer advantages such as immunity to electromagnetic interference, high accuracy, and energy efficiency, but their performance is hindered by environmental variability across deployment areas. This paper presents a transfer learning (TL) framework for VLC-based indoor localization that enhances cross-area robustness while reducing data requirements. Using real-world measurements from four distinct areas of a Bosch manufacturing facility, we show that fine-tuning a deep neural network (DNN) pre-trained on source area data with only 40% of target-area measurements recovers 97% of full-data performance, thereby reducing the amount of target data needed. We further introduce the Transfer Efficiency Score (TES), a composite metric that identifies the most effective source models for transfer without exhaustive evaluation. Guided by TES, the framework selects the best source model, yielding localization errors as low as 40.9 cm with success rates above 86%. Validation on an unseen area confirms its generalization capability, with a 3.28 cm mean error achieved using the TES-selected source model. Overall, the proposed framework provides a scalable and data-efficient pathway for industrial VLC localization, substantially reducing deployment costs and computational burden while maintaining high accuracy in diverse and dynamic environments.

Index Terms—Deep Neural Network (DNN), Fine-tuning, Indoor localization, Transfer Learning (TL), Transfer Efficiency Score (TES), Visible Light Communication (VLC).

## I. INTRODUCTION

U <sup>NPLANNED</sup> <sup>downtime</sup> <sup>from</sup> <sup>misplaced</sup> <sup>or</sup> <sup>untraceable</sup> assets remains a major contributor to production delays and operational inefficiencies, underscoring the critical need for robust indoor-positioning solutions in Industry 4.0 environments [1], [2]. Modern factories are increasingly transformed by AI-driven automation and digital-twin workflows, with real-time location-based services (LBS), power asset tracking, process optimization, and coordinated robotics [3], [4]. In this context, indoor localization has evolved from an auxiliary feature into a critical infrastructure that underpins next-generation industrial automation [5].

Despite advances, three core challenges prevent large-scale deployment of indoor localization. First, cross-area robustness, where models trained in one area often degrade significantly when deployed in a different layout or building area [6]. Second, data inefficiency, which means collecting and labeling comprehensive datasets can require days of downtime per area, contradicting the high-cycle production demands [7], [8]. Third, energy and cost constraints, as many radio frequency (RF) based systems consume excessive power or rely on dense beacon infrastructures, driving up maintenance costs and reducing battery lifespan in IoT deployments [9], [10].

While several indoor localization technologies are widely used in industrial environments, each presents its own set of trade-offs relative to the core challenges. Wi-Fi localization offers moderate power consumption [11] but suffers severe accuracy drops under electromagnetic interference (EMI) from metal machinery and dynamic layouts, which affects both robustness and maintenance demands [12]. Bluetooth Low Energy (BLE) reduces power consumption in sleep-mode to as little as 10.1 µA [13] but is limited by short range and environmental variability, making it costly to scale and sensitive to changes across deployment areas [10], [14]. Both BLE and Wi-Fi solutions depend on dense infrastructure, often requiring a large number of access points or beacons, which introduces high installation and ongoing maintenance costs [15], [16]. Ultra-Wideband (UWB) systems achieve higher accuracy, but come with higher hardware costs and power requirements. Like all RF-based approaches, UWB also remains vulnerable to signal degradation from EMI and dynamic industrial layouts [17]–[19]. These trade-offs highlight the need for a localization approach that blends high accuracy with an efficient and minimal infrastructure.

Visible Light Communication (VLC) localization emerges as a compelling alternative that leverages light-emitting diode (LED) lighting infrastructures to enable communication and positioning. Compared to RF technologies, VLC achieves centimeter-level accuracy and up to 60% lower energy consumption, while being inherently immune to EMI [20]– [22]. Because VLC relies on unobstructed optical paths, its line-of-sight, short-range nature reduces multipath effects and simplifies the signal-processing pipeline [23]. At the same time, this reliance on direct optical paths makes VLC systems sensitive to changes in ambient lighting, temporary blockages, and uneven surface reflections commonly found in industrial environments [24]–[26]. Although these optical challenges can affect performance, they follow consistent physical laws and environmental patterns. For example, ambient lighting often changes gradually or follows known schedules (such as day/night cycles or factory lighting controls); occlusions occur when objects block the line of sight, and their effects can be estimated using geometry or motion sensors; and surface reflectance depends on material properties (e.g., metal or plastic), which influence light reflection in predictable ways. This predictability allows them to be effectively addressed through model-based correction and area-specific adaptation [27], [28].

The complex, non-linear distortions introduced by EMI, multipath reflections, and dynamic factory layouts have motivated a shift toward machine-learning (ML) methods, which can learn mapping functions between raw sensor inputs and positions without requiring manual feature engineering or signal modeling [29]. However, standard ML paradigms assume that the deployment environment remains static, leading to significant degradation in localization accuracy and generalizability when floorplans, material composition, or illumination conditions change [6].

Several adaptation strategies have been explored to address variability across deployment areas. Federated Learning (FL) preserves privacy by distributing model training across edge devices [30], but it requires periodic exchange of gradient updates over potentially unreliable factory wireless links, introducing latency in highly dynamic or resource-constrained settings. Continuous Learning (CL) enables incremental adaptation [31], yet suffers from catastrophic forgetting and depends on continual access to new labeled data, which is costly and labor-intensive. Transfer Learning (TL), by contrast, requires periodic fine-tuning on minimal datasets from the target deployment area as environmental conditions evolve, thus reducing both communication overhead and the amount of target data required while maintaining high accuracy compared to continuous retraining approaches [32]–[34].

Despite its success in RF-based localization, TL has seen limited application in VLC systems because fundamental propagation differences create unique adaptation challenges. Unlike RF, which is dominated by EMI and multipath fading, VLC is governed by illumination geometry and line-of-sight dependence. VLC systems therefore face environmental factors such as ambient-light interference (e.g., from fluorescent lamps or welding), geometry-dependent multipath scattering from reflective surfaces, and occlusion patterns that critically affect signal visibility. These VLC-specific characteristics require TL strategies that account for optical propagation properties, ambient lighting conditions, and the inherent directional nature of VLC.

Given these advantages, VLC stands out as a strong candidate for scalable, high-precision indoor localization. However, to fully unlock its potential particularly in the context of crossarea deployments, there is a pressing need for ML solutions that are not only energy-efficient and accurate, but also capable of adapting to area-specific variations in optical conditions. Equally important is the need for systematic model selection mechanisms that can identify the most suitable pre-trained model for a given target deployment without requiring exhaustive fine-tuning of all available candidates. In practical industrial deployments, engineers often have access to multiple pretrained models from different source areas, each with varying degrees of compatibility with the target environment. Without reliable transferability assessment metrics, practitioners must resort to costly trial-and-error approaches that contradict the efficiency goals of TL. A normalized composite metric that combines multiple performance indicators would enable rapid, cost-effective identification of optimal source models while providing consistent evaluation across diverse deployment scenarios. These considerations motivate our development of a TL-based VLC localization framework that maximizes cross-area adaptability and precision, addressing the interplay between optical variability and ML model generalization. Our key contributions are as follows:

TL Framework for VLC Cross-Area Localization: We develop and validate a comprehensive TL framework that enables VLC localization models trained in one area to adapt effectively to new layouts with minimal retraining, achieving robust cross-area deployment in dynamic industrial environments. This work is intended as a proofof-concept demonstrating the suitability and practicality of TL for VLC-based indoor localization under realistic industrial deployment conditions.

• Data-Efficient Fine-Tuning Strategy: We demonstrate that fine-tuning on just 40% of target-area data recovers 97% of full-data performance, reducing both target-area data collection requirements and deployment effort by 60% compared to traditional full-retraining approaches.

• Transfer Efficiency Score (TES) Metric and Validation: We introduce TES, a normalized composite metric that combines localization error and success rate to systematically select optimal source models without exhaustive testing, and validate its effectiveness through superior performance demonstration on an unseen facility layout.

The remainder of this paper is organized as follows: Section II reviews related work, Section III presents the system model and problem formulation, Section IV details the proposed algorithm development, Section V discusses the experimental setup and data collection process, Section VI analyzes performance evaluation results, and Section VII concludes the paper.

## II. RELATED WORKS

The rapid convergence of Industry 4.0 requirements, and advancements in ML techniques has led to extensive research into indoor localization. In particular, cross-area robustness, data and energy efficiency, and scalable deployment have become focal points. Cross-area robustness refers to the model-level challenge of maintaining localization accuracy when deployed across different physical environments with varying layouts, structural characteristics, and interference patterns, which we address through TL. Scalable deployment, in contrast, represents the system-level challenge of deploying localization across multiple areas and devices without proportional increases in retraining, data collection, or infrastructure costs — a goal facilitated by our data-efficient fine-tuning and TES-based source model selection. We organize this section around three major ML paradigms enabling adaptive localization systems; FL, CL, and TL. We first assess their application in RF-based systems, then examine VLC-based approaches, and finally highlight the bottlenecks that motivate our TL-driven VLC framework.

## A. FL for Indoor Localization

FL has emerged as a promising solution for indoor localization. It enables collaborative model training across edge devices without transferring raw data, thereby reducing communication overhead and preserving user privacy [35], [36]. Several studies have explored FL in this context. FeMLoc employs a two-stage meta-training and fine-tuning strategy to capture environmental variability, achieving up to 80.9% gains in accuracy and an 82.2% reduction in training iterations compared to centralized approaches [37]. [38] present a FL framework for Wi-Fi fingerprinting that maintains data privacy on edge devices, enhances localization accuracy, and lowers energy consumption by leveraging distributed computing. [30] demonstrate that retaining all training data locally and transmitting only model updates such as gradients or weights can significantly reduce bandwidth usage while ensuring sensitive sensor data remains confined to local hardware. Similarly, [39] further enhance generalization by randomizing and distributing building-specific datasets among clients, leading to more specialized models, higher accuracy, and lower communication costs. Other works [40]–[42] also integrated FL into the design of localization systems, demonstrating its potential for an efficient collaborative learning.

In the context of VLC systems, FL has been successfully applied to VLC positioning systems to address privacy preservation and collaborative model training challenges. For instance, [43] proposed a privacy-preserving cooperative VLC positioning scheme based on FL for nonstationary environments. Their approach employs a Cooperative Visible-light Positioning Network (CVPosNet) that enables global model training without sharing private user data, achieving improved positioning accuracy and generalization capability in dynamic lighting conditions. Hybrid VLC/RF architectures have also been adopted to offload model updates onto the visible spectrum, improving uplink bandwidth efficiency [44].

Despite these advancements, VLC-FL systems require strict line-of-sight (LOS) to LEDs, which limits client availability and leads to uneven data contributions. Uplink communication via RF or Infrared (IR) links also suffers from bandwidth constraints, slowing global aggregation and hindering scalability in dynamic environments. Moreover, FL-trained models often become overly tailored to the specific data distributions of participating clients. This hinders generalization to unfamiliar devices and unseen deployment environments [45]. These limitations expose critical performance disparities across heterogeneous deployments and underscore the need for architectural enhancements, such as TL, to improve cross-area robustness without extensive retraining or additional data collection.

## B. CL for Indoor Localization

CL allows incremental model updates as new data arrive, suiting environments with dynamic signal characteristics [31]. Recent surveys, such as [46] provide empirical insights to CL techniques. A key issue in CL is catastrophic forgetting, where a model loses previously acquired knowledge when trained on new tasks. Overcoming this problem typically requires additional memory capacity, complex update strategies, and increased computational costs, which may compromise CL’s suitability for resource-limited environments. Various CL strategies have been proposed to address this issue. As outlined in [47], task-incremental learning strategies fall into three broad categories: replay-based (involving storage and reprocessing of prior samples), regularization-based (enforcing constraints on parameter shifts), and parameter isolation (dedicating separate parameter subsets for different tasks). Each approach brings trade-offs—replay increases memory and latency, regularization demands frequent updates, and parameter isolation reduces scalability. Similarly, [48] investigated classincremental learning for image classification, showing that techniques such as rehearsal and bias mitigation further complicate the computational and architectural design. The feasibility of CL depends not only on algorithmic design, but also on access to labeled data and computational resources, both commonly restricted in industrial environments. Furthermore, the need for frequent updates in response to new data can lead to variable energy consumption, complicating deployment of CL on energy-constrained devices such as battery-powered IoT nodes. These operational challenges limit the scalability of CL for industrial indoor localization, where systems must remain both adaptive and efficient [49].

Within VLC systems, CL has been applied to address environmental dynamics such as lighting variation, mobility, and access point availability. Approaches include adaptive Kalman filtering [50] and RSS-based MIMO localization that adapt positioning in real-time [51]. Broad learning systems with self-attention have also shown promise for continuous model updates [52]. However, abrupt optical channel changes and the non-stationary nature of VLC signals intensify catastrophic forgetting. Moreover, real-time adaptation requires frequent computation and memory access, conflicting with VLC’s hardware constraints. These factors hinder CL’s effectiveness in VLC environments, where fast yet efficient adaptation is essential. In this context, TL offers a more lightweight and scalable alternative by enabling fast generalization to new environments using minimal data and limited fine-tuning.

## C. Alternative Adaptation Techniques for Indoor Localization

Several recent studies have explored alternative adaptation techniques such as domain-adversarial training and meta-learning to address domain shifts across environments. Domain-adversarial approaches learn domain-invariant representations by introducing a discriminator that aligns feature distributions between source and target environments [53], [54]. For WiFi CSI fingerprinting, recent works employ graph convolutional networks with adversarial alignment to leverage large crowdsourced datasets across multiple buildings [55], achieving competitive accuracy. However, these methods typically require complex graph construction, substantial labeled or unlabeled target data for stable training, and additional optimization stages that increase computational cost and implementation complexity. Such requirements can be challenging in industrial VLC deployments, where measurement campaigns must be conducted under real production constraints and additional adversarial training stages increase implementation complexity and computational overhead.

Meta-learning approaches such as Model-Agnostic Meta-Learning (MAML) learn model initializations that enable rapid adaptation to new tasks [56]. In wireless localization, these methods have been applied across numerous sites or combined with FL to support dynamic WiFi positioning [37], [57]. Their effectiveness depends on exposure to many diverse environments during meta-training and on repeated task-level optimization, which can be difficult in industrial VLC scenarios due to the need for specialized meta-training procedures and the practical effort required for repeated data labeling and system calibration.

Compared with adversarial or meta-learning pipelines, which often involve additional optimization stages and more complex training procedures, TL offer computational simplicity while achieving substantial cross-domain performance recovery.

## D. TL for Indoor Localization

While FL and CL have advanced the applicability of adaptive indoor positioning, their reliance on extensive local data, continual training and limited generalizability continue to constrain practical deployment, especially in optical domains where device connectivity and synchronization are limited. In contrast, TL offers a compelling alternative by leveraging pretrained models to enable rapid adaptation in new environments using minimal data and limited retraining. These results in faster deployment, reduced resource usage, and improved robustness across deployment areas [58], [59]. As discussed in [60], [61], the effectiveness of TL depends on understanding when, what, and how to transfer knowledge. These decisions are particularly important in industrial environments, where the trade-off between data collection costs, energy budgets and adaptation delays must be minimized. Compared to FL and CL, TL avoids the need for continuous client participation or memory-intensive model updates, making it more suitable for resource-constrained VLC systems.

TL has demonstrated effectiveness in RF-based localization systems. For example, in Wi-Fi fingerprinting, fine-tuning convolutional neural network (CNN) models on just 55% of target data recovered over 95% of full-data accuracy [6]. Similarly, [62] applied TL to BLE room classification across homes, significantly improving F1 scores in unseen settings, and UWB systems maintained sub-30 cm accuracy while reducing fingerprinting effort by 93% [63].

Despite this progress, TL remains underexplored in VLC based localization, where distinct optical-domain challenges persist. VLC offers advantages such as high bandwidth, EMI immunity, and energy efficiency through infrastructure reuse, yet most prior work emphasizes deterministic modeling or hardware mitigation. For example, [64] employed adaptive least squares and Kalman filtering to mitigate noise under fixed conditions, while [65] utilized singular value decomposition (SVD) for efficient 3D positioning, achieving high accuracy but lacking generalization. [66] noted that learning-based VLC methods still struggle with robustness and transferability. Furthermore, recent TL efforts in RF-based systems rarely account for VLC-specific factors like ambient light variation, photodiode-induced noise, or LED flicker. For instance, [67] applied domain-adversarial learning to reduce Wi-Fi localization errors, but did not address interference in the optical domain. Similarly, [68] improved LoRa/Wi-Fi positioning via latent space alignment, without considering degradation modes specific to VLC.

Our previous study has shown that TL-based VLC positioning can improve accuracy by up to 47%, while reducing energy consumption and computational time by over 30% relative to conventional training approaches [69]. That study applied TL within a single industrial area to improve robustness under varying lighting conditions, demonstrating that VLC localization can adapt to environmental changes with reduced data requirements. In contrast, the present work addresses a different challenge: cross-area adaptation. We develop a TL framework that enables models trained in one deployment area to generalize to new areas, and introduce a new metric to systematically identify the most suitable source model without exhaustive testing.

These gaps motivate our work: a cross-area-aware TL framework for VLC localization, underpinned by (i) dataefficient fine-tuning that cuts data collection costs by 60% while preserving over 97% of full-data performance, and (ii) a TES metric to rank source models without exhaustive trials. Our approach is scalable—meaning it can accommodate additional deployment areas with minimal extra data or retraining—and energy-efficient, offering a lightweight, rapid adaptation solution tailored to industrial VLC environments.

## III. SYSTEM MODEL AND PROBLEM FORMULATION

This section provides a structured approach for modeling and evaluating our deep neural network (DNN) configuration across different deployment areas, focusing on the challenge of TL and model generalization. DNNs are employed as the core regression architecture due to their proven effectiveness in VLC-based indoor positioning systems, where they can capture complex non-linear relationships between received signal strength (RSS) measurements and spatial coordinates that traditional ML approaches struggle to model effectively. Unlike conventional ML techniques such as linear regression or support vector machines, DNNs can learn hierarchical feature representations from raw RSS data, making them particularly suitable for handling the multipath effects, ambient light variations, and LED flicker characteristics inherent in VLC environments. We further introduce the TES, a unified metric to guide the selection of the optimal source model for knowledge transfer.

![](images/7ef1f0797a071d99310ea108466872867a0962d37f5f6871ddc71311933c5532.jpg)  
Fig. 1: System Model of the TL step.

## A. System Model

Our goal is to develop a TL approach for predicting users coordinates (longitude and latitude) in a cross-area setting, using a VLC dataset collected from the factory hall of a Bosch plant in Blaichach, Germany. To support robust training and evaluation across varying deployment areas, the actual measurement area (area of interest shown in Fig. 1) is partitioned into non-overlapping subareas (Areas A, B, C, and D). We partitioned these areas based on their architectural diversity, including differences in layout dimensions, equipment configurations, and structural element distribution. These variations introduce distinct environmental conditions, creating a realistic and challenging testbed for evaluating cross-area model generalization (see Section V for more details). For each experimental round, one area (e.g., Area D) is withheld as an unseen deployment area, while the remaining areas are used for model development and validation. This rotation process ensures comprehensive generalization assessment across all spatial contexts.

As illustrated in Fig. 1, a DNN is trained on the dataset collected from one area (the “source” e.g., Area A) and initially validated on the validation dataset of the same area. To assess cross-area robustness, the trained model is then evaluated on datasets from a different area (the “target,” e.g., Area B or C). If the model’s validation performance on the target area (in terms of localization error and success rate) is significantly degraded compared to its performance on the source area, we adopt a TL approach to mitigate this cross-area generalization gap. Therefore, the model is fine-tuned on a small subset of the target area, freezing lower layers to preserve general feature representations. The target area is re-evaluated after fine-tuning to assess whether performance improved beyond the identified degradation. This process is repeated for all possible source-target pairs (A→B, A→C, B→A, B→C, C→A, C→B), providing a comprehensive assessment of cross-area generalization and adaptation for validation purposes within the known areas (A, B, C) before applying the selected model to the truly unseen area D.

To systematically identify the most effective transfer path, we compute a TES for each source model after fine-tuning on different target areas. Fig. 2 details this automated workflow, which replaces manual selection with a repeatable metric. The source area with the highest TES score is identified as the most reliable for knowledge transfer. The pre-trained model from this optimal source area is then validated on an unseen area (e.g., Area D) to assess its broader generalizability. By rotating the role of the unseen area across all subareas, we ensure that the selected source model maintains strong performance across diverse deployment areas when applied as a foundation for TL to new environments.

## B. Problem Formulation

Let $\boldsymbol { \mathcal { A } } = \{ A , B , C \}$ be the set of source subareas and let $\mathcal { D } _ { i }$ be the VLC dataset collected in area $i ;$

$$
\mathcal { D } _ { i } = \{ ( x _ { n } ^ { i } , y _ { n } ^ { i } ) \} _ { n = 1 } ^ { N _ { i } } , \qquad i \in \mathcal { A } ,\tag{1}
$$

where $x _ { n } ^ { i } \in \mathbb { R } ^ { d }$ are the input features (VLC RSS readings) and $y _ { n } ^ { i } = \widetilde { ( \ell _ { n } ^ { i } , \phi _ { n } ^ { i } ) } \in \mathbb { R } ^ { 2 }$ are the corresponding true coordinates, with $\ell _ { n } ^ { i }$ denoting the longitude and $\phi _ { n } ^ { i }$ the latitude.

Our objective is to build a DNN regressor $f _ { \theta } : \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 }$ that, when trained on any source deployment area and finetuned on a small labeled subset of a target deployment area, achieves low localization error and high transferability across all subareas. To systematically guide the selection of the source model, we further integrate the TES, which consolidates the accuracy and success rate of the localization into a unified metric. Thus, the problem formulation jointly considers (i) the effectiveness of TL and (ii) the identification of the most reliable source area through TES.

![](images/4d21693961e9f9b921c0649960fb005559e9b6e164fb61033e5b476f111f0494.jpg)  
Fig. 2: Workflow of the proposed TL framework with TES assessment and validation.

1) Source Training & Direct Evaluation: For each source area $s \in { \mathcal { A } } ,$ , we train an instance of the DNN with identical architecture and hyperparameters, but with weights initialized and optimized independently on the dataset of that area. This ensures comparability across source areas while avoiding any architectural bias. The parameters $\theta _ { s }$ are obtained by minimizing a regression loss $\mathcal { L }$ (here mean squared error, MSE) on the full source dataset as:

$$
\theta _ { s } = \arg \operatorname* { m i n } _ { \theta } \mathcal { L } \big ( f _ { \theta } ( X _ { s } ) , Y _ { s } \big ) ,\tag{2}
$$

where

$$
X _ { s } = \{ x _ { n } ^ { s } \} _ { n = 1 } ^ { N _ { s } } , \quad Y _ { s } = \{ y _ { n } ^ { s } \} _ { n = 1 } ^ { N _ { s } }
$$

Here, $\theta _ { s }$ denotes the learned network parameters (weights and biases) after training on area $s , \ N _ { s }$ is the total number of training samples collected in area s. $f _ { \theta } : \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 }$ is the DNN regressor, mapping each d-dimensional RSS vector $x _ { n } ^ { s }$ to a 2-vector of predicted coordinates. We denote this prediction by;

$$
\hat { y } _ { n } ^ { s } = f _ { \theta } ( x _ { n } ^ { s } ) \in \mathbb R ^ { 2 } ,\tag{3}
$$

$\mathcal { L } ( f _ { \theta } ( X _ { s } ) , Y _ { s } )$ is the MSE over the dataset $( X _ { s } , Y _ { s } )$ , i.e.

$$
\mathcal { L } \big ( f _ { \theta } ( X _ { s } ) , Y _ { s } \big ) = \frac { 1 } { N _ { s } } \sum _ { n = 1 } ^ { N _ { s } } \bigl \| f _ { \theta } ( x _ { n } ^ { s } ) - y _ { n } ^ { s } \bigr \| _ { 2 } ^ { 2 } ,\tag{4}
$$

We then evaluate this pretrained model directly on each target area $t \in \mathcal { A } \setminus \{ s \}$ :

$$
E _ { s  t } = \mathcal { L } \big ( f _ { \theta _ { s } } ( X _ { t } ) , Y _ { t } \big ) ,\tag{5}
$$

where

$$
X _ { t } = \{ x _ { n } ^ { t } \} _ { n = 1 } ^ { N _ { t } } , \quad Y _ { t } = \{ y _ { n } ^ { t } \} _ { n = 1 } ^ { N _ { t } } ,
$$

2) Fine-Tuning: Direct evaluation (Eq. 5) already provides a measure of how well the pretrained model $f _ { \boldsymbol { \theta } _ { s } }$ generalizes to a target area $t \neq s .$ . If this evaluation reveals a substantial degradation in localization performance compared to validation on the source area, we then proceed with fine-tuning to adapt the model. To do so, we fine-tune $f _ { \theta _ { s } }$ using a small randomly sampled labeled subset of the target dataset:

$$
\begin{array} { r } { \mathcal { D } _ { t } ^ { \mathrm { ( F T ) } } = \big \{ ( x _ { n } ^ { t } , y _ { n } ^ { t } ) \big \} _ { n = 1 } ^ { M } , \quad M = \rho N _ { t } , \quad \rho \in [ 0 , 1 ] , } \end{array}\tag{6}
$$

where M is the number of fine-tuning samples, $\rho$ is the sampling ratio, and $N _ { t } = | \mathcal { D } _ { t } |$ . We experimented with different values of $\rho$ (10%, 20%, 40%, 60%, 100%) to study the tradeoff between performance and data availability. Following the evaluation in Section VI-B, we select $\rho = 4 0 \%$ as the finetuning ratio used in all subsequent experiments. The subset $\mathcal { D } _ { t } ^ { ( \mathrm { F T } ) }$ is drawn at random from $\mathcal { D } _ { t }$

To preserve the general representations learned from the source area, only the final k layers of the DNN are unfrozen and updated using $\mathcal { D } _ { t } ^ { ( \mathrm { F T } ) }$ , while the lower layers remain fixed. The motivation and impact of this fine-tuning strategy under different target-data regimes are further analyzed in Section VI-D. The fine-tuning operation is denoted by $\psi ( \boldsymbol { \theta } _ { s } , \mathcal { D } _ { t } ^ { ( \mathrm { F T } ) } )$ , representing gradient updates to the unfrozen parameters. The adapted model parameters are then:

$$
\theta _ { s  t } = \psi \big ( \theta _ { s } , \mathcal { D } _ { t } ^ { ( \mathrm { F T } ) } \big ) ,\tag{7}
$$

Following fine-tuning, we evaluate the adapted model on the full target dataset to measure its localization error and success rate:

$$
E _ { s  t } ^ { \mathrm { ( F T ) } } = \mathcal { L } \big ( f _ { \theta _ { s  t } } ( X _ { t } ) , Y _ { t } \big ) ,\tag{8}
$$

$$
S _ { s  t } ^ { ( \mathrm { F T } ) } = \frac { 1 } { N _ { t } } \sum _ { n = 1 } ^ { N _ { t } } \mathbf { 1 } \big ( \lVert \hat { y } _ { n } - y _ { n } ^ { t } \rVert \leq \delta \big ) ,\tag{9}
$$

where $\delta$ is a predefined success threshold and $\hat { y } _ { n } = f _ { \theta _ { s  t } } ( x _ { n } ^ { t } )$ is the model prediction.

3) Transfer Directions: Now let’s define the set of all ordered source–target pairs $\tau { : }$

$$
\mathcal { T } = \{ ( s , t ) \mid s , t \in \mathcal { A } , s \neq t \} ,\tag{10}
$$

so there are six transfer directions: $A \to B , A \to C , B \to A .$ etc. For each $( s , t ) \in \mathcal { T }$ we obtain a fine-tuned model $\theta _ { s  t }$ and corresponding error $E _ { s  t } ^ { \mathrm { ( F T ) } }$

4) Transfer Evaluation Score (TES): To systematically evaluate and rank these 6 candidates, we define the TES, a composite metric that reflects both post-transfer localization accuracy and robustness. For each transfer scenario, we compute the minimum and maximum localization errors $E _ { \mathrm { m i n } } , \ E _ { \mathrm { m a x } }$ and minimum and maximum success rates $S _ { \mathrm { m i n } } , \ S _ { \mathrm { m a x } }$ after fine-tuning as expressed below: Importantly, $E _ { \operatorname* { m i n } } , E _ { \operatorname* { m a x } } , S _ { \operatorname* { m i n } } , S _ { \operatorname* { m a x } }$ are computed independently for each transfer scenario (s, t), so that the normalization in Eq. 12 reflects the performance range within that specific scenario, without being influenced by other transfers.

$$
\{ \begin{array} { l l } { E _ { \mathrm { m i n } } = \displaystyle \operatorname* { m i n } _ { ( s , t ) \in \mathcal { T } } E _ { s  t } ^ { ( \mathrm { F T } ) } } \\ { E _ { \mathrm { m a x } } = \displaystyle \operatorname* { m a x } _ { ( s , t ) \in \mathcal { T } } E _ { s  t } ^ { ( \mathrm { F T } ) } } \\ { S _ { \mathrm { m i n } } = \displaystyle \operatorname* { m i n } _ { ( s , t ) \in \mathcal { T } } S _ { s  t } ^ { ( \mathrm { F T } ) } } \\ { S _ { \mathrm { m a x } } = \displaystyle \operatorname* { m a x } _ { ( s , t ) \in \mathcal { T } } S _ { s  t } ^ { ( \mathrm { F T } ) } } \end{array}  ,\tag{11}
$$

We then compute the normalized error and success-rate. Normalization is applied so that both error and success-rate values are scaled to the same [0,1] range, regardless of their absolute magnitudes. This allows TES to combine them fairly in a single composite score, avoiding bias toward metrics with larger numerical ranges.

$$
\mathrm { N E } _ { s  t } = \frac { E _ { s  t } ^ { \mathrm { ( F T ) } } - E _ { \mathrm { m i n } } } { E _ { \mathrm { m a x } } - E _ { \mathrm { m i n } } } ,\tag{12}
$$

$$
\mathrm { N S R } _ { s  t } = \frac { S _ { s  t } ^ { \mathrm { ( F T ) } } - S _ { \mathrm { m i n } } } { S _ { \mathrm { m a x } } - S _ { \mathrm { m i n } } } ,\tag{13}
$$

For each (s, t), the per-target TES is defined as:

$$
\mathrm { T E S } _ { s  t } = \alpha ( 1 - \mathrm { N E } _ { s  t } ) + ( 1 - \alpha ) ( \mathrm { N S R } _ { s  t } ) ,\tag{14}
$$

where $\alpha \in [ 0 , 1 ]$ controls the trade-off between localization error and success rate. In our experiments we set $\alpha = 0 . 5$ giving equal weight to both metrics. This choice reflects the fact that both metrics are equally important for evaluating transferability: error captures precision, while success rate captures robustness. To verify the robustness of this choice, we performed a sensitivity analysis of TES for $\alpha \in [ 0 . 3 , 0 . 7 ]$ as reported in Section VI-C.

After computing $\mathrm { T E S } _ { s  t }$ for every source–target pair $( s , t ) \in \mathcal { T }$ , we group the scores by their originating source area. Each source area contributes two transfer scenarios (e.g., A→B and $\mathbf { A } {  } \mathbf { C } )$ , and the TES values from these scenarios are averaged to produce one representative score. This consolidated score reflects how consistently a source transfers knowledge across all its targets, revealing which source area is the strongest candidate for pretraining and deployment.

We then select the source area with the highest averaged score:

$$
s ^ { * } = \arg \operatorname* { m a x } _ { s \in \mathcal { A } } \mathrm { T E S } _ { s } ,\tag{15}
$$

Here, $s ^ { * }$ identifies the source area whose model demonstrates the most reliable and balanced transfer performance across all possible target areas, making it the best candidate for deployment and for subsequent evaluation on unseen areas.

5) Validation on Unseen Areas: To evaluate the generalization capability of the selected model, we take the source model identified via TES from the best-performing source area $s ^ { * }$ and test it on a held-out subarea $D ,$ which was excluded from training and model selection but retained for evaluation. The localization error and success rate on area D are computed as:

$$
E _ { s ^ { * } \to D } = \mathcal { L } \big ( f _ { \theta _ { s ^ { * } } } ( X _ { D } ) , Y _ { D } \big ) ,\tag{16}
$$

$$
S _ { s ^ { * }  D } = \frac { 1 } { N _ { D } } \sum _ { n = 1 } ^ { N _ { D } } \mathbf { 1 } \big ( \| \hat { y } _ { n } - y _ { n } ^ { D } \| \leq \delta \big ) ,\tag{17}
$$

where $\hat { y } _ { n } = f _ { \theta _ { s ^ { * } } } ( x _ { n } ^ { D } )$ is the predicted coordinate, $y _ { n } ^ { D }$ is the ground-truth location, and δ is a pre-defined success threshold. The indicator function $\mathbf { 1 } ( \cdot )$ returns 1 if the prediction is within δ of the ground truth, and 0 otherwise.

This evaluation quantifies the ability of the selected model to generalize to unseen environments. To further validate the reliability and consistency of the TES-based selection strategy, we rotate the role of the unseen area among all available subareas $\mathcal { A } = \{ A , B , C , D \}$ . In each iteration, one area $( \mathrm { e . g . }$ A) is excluded from training and used solely for validation. The TES is computed using the remaining areas (e.g., B, C, and D). This simulates a realistic deployment scenario where the model must generalize to a new environment using transfer experience from previously explored regions, and serves as a validation step to confirm the robustness and consistency of the TES-based model selection.

## IV. ALGORITHM DEVELOPMENT AND IMPLEMENTATION

This section presents a two-stage algorithmic framework designed to enable efficient and scalable model transfer for VLC-based indoor localization. The primary objective is to promote cross-area generalization by systematically evaluating when and how TL should be applied across spatial subareas.

Our approach consists of two interconnected components: (i) a cross-area model training and evaluation procedure that establishes baseline performance across all deployment areas, and (ii) a TES-based selection and validation mechanism that identifies optimal transfer configurations. These components work synergistically to ensure robust localization performance while minimizing computational overhead.

We begin by defining a set of spatial subareas ${ \cal A } \ =$ $\{ A , B , C \}$ , each associated with its own dataset $\mathcal { D } _ { i }$ . For every source area $s \in A ,$ a DNN model $f _ { \theta _ { s } }$ is trained by minimizing a supervised loss function $\mathcal { L }$ over the corresponding dataset $\mathcal { D } _ { s }$ . The resulting model parameters $\theta _ { s }$ serve as source models for potential transfer to other target areas, forming the basis of our TL strategy.

Building upon this foundation, Algorithm 1 evaluates all possible source–target area combinations (s, t), where $s \neq t .$ For each pair, (i) a direct cross-area inference is performed using the source model $f _ { \theta _ { s } }$ on the target dataset $\mathcal { D } _ { t } ,$ yielding an initial localization error $E _ { s  t }$ and success rate $S _ { s  t }$ . (ii) A baseline reference is computed by evaluating $f _ { \theta _ { s } }$ on its own training data $\mathcal { D } _ { s } ,$ producing $E _ { s  s }$ and $S _ { s  s } .$ . These evaluations establish a performance benchmark for each source model on its home area, which is then used to judge how well it generalizes to new environments. Whenever the performance on a target area falls short of this benchmark—that is, if the localization error on the target dataset exceeds the source baseline $( E _ { s  t } ~ > ~ E _ { s  s } )$ or the success rate drops below it $( S _ { s  t } ~ < ~ S _ { s  s } )$ —the algorithm concludes that simple cross-area inference is insufficient. At this point, the algorithm triggers TL to help the model adapt to the new area while retaining the useful representations learned from the source.

Algorithm 1: Cross-Area Model Training and Transfer   
Capability Evaluation   
Input $\mathbf { \langle } A = \{ A , B , C \} ; \mu$ Set of available areas \*/   
Input : $\{ \mathcal { D } _ { i } \} _ { i \in \mathcal { A } } ; I ^ { * }$ Datasets for each area \*/   
Input : $\dot { \rho } ; I ^ { * }$ Fine-tuning sampling ratio \*/   
Input : k; /\* Number of layers to unfreeze $^ { * }$   
Input : $; \delta ; / { ^ \ast }$ Success threshold \*/   
Output: $\{ \theta _ { s \to t } \} _ { ( s , t ) \in \mathcal { T } ^ { : } } / { } ^ { * }$ All fine-tuned models \*/   
$\{ E _ { s  t } ^ { \mathrm { ( F T ) } } \} _ { ( s , t ) \in \mathcal { T } ^ { : } } / { } ^ { * }$ Transfer errors $^ { * }$   
$\{ S _ { s  t } ^ { ( \mathrm { F T } ) } \} _ { ( s , t ) \in \mathcal { T } ^ { : } } / { ^ { * } }$ Transfer success rates $^ { * }$   
1 Train Models on All Source Areas   
2 for each source area $s \in { \mathcal { A } }$ do   
3 Train DNN model $f _ { \boldsymbol { \theta } _ { s } }$ on $\mathcal { D } _ { s } \colon$   
4 $\theta _ { s } \gets \arg \operatorname* { m i n } _ { \theta } \tilde { \mathcal { L } } ( \tilde { f } _ { \theta } ( X _ { s } ) , Y _ { s } )$   
5 end   
6 Evaluate All Source-Target Transfer Pairs   
7 Initialize $\mathcal { T } = \{ ( s , t ) \mid s , \breve { t } \in \mathcal { A } , s \neq t \}$   
8 for each $( s , t ) \in \mathcal { T }$ do   
9 Direct Cross-Area Evaluation:   
10 $E _ { s  t } = \mathcal { L } ( f _ { \theta _ { s } } ( X _ { t } ) , Y _ { t } )$   
11 $\begin{array} { r } { S _ { s  t } = \frac { 1 } { N _ { t } } \sum _ { n = 1 } ^ { N _ { t } } \mathbf { 1 } ( \| \hat { y } _ { n } ^ { t } - y _ { n } ^ { t } \| \le \delta ) } \end{array}$   
12 Source Area Baseline:   
13 /\* Used only for comparison to decide if TL is needed \*/   
14 $E _ { s  s } = \dot { \mathcal { L } } ( f _ { \theta _ { s } } ( \dot { X _ { s } } ) , Y _ { s } )$   
15 $\begin{array} { r } { S _ { s  s } = \frac { 1 } { N _ { \mathrm { e } } } \sum _ { n = 1 } ^ { N _ { s } } \mathbf { 1 } ( \| \hat { y } _ { n } ^ { s } - y _ { n } ^ { s } \| \le \delta ) } \end{array}$   
16 Apply TL if Performance Degrades:   
17 if $\bar { E } _ { s  t } > E _ { s  s } \bullet r \ S _ { s  t } < \bar { S } _ { s  s }$ then   
18 Sample fine-tuning subset: $\mathcal { D } _ { t } ^ { ( \mathrm { F T } ) } \subset \mathcal { D } _ { t }$ with   
$| \mathcal { D } _ { t } ^ { ( \mathrm { F T } ) } | = \rho \cdot N _ { t }$   
19 Initialize $\dot { \theta } \gets \theta _ { s }$   
20 Freeze lower $( L - k )$ layers of $f _ { \theta }$   
21 Fine-tune: $\theta _ { s \right. t } \left. \psi ( \theta _ { s } , \mathcal { D } _ { t } ^ { ( \mathrm { F T } ) } )$   
22 Re-evaluate with fine-tuned model:   
23 $E _ { s  t } ^ { ( \mathrm { F T } ) } = \mathcal { L } ( f _ { \theta _ { s }  t } ( X _ { t } ) , Y _ { t } )$   
24 $\begin{array} { r } { S _ { s  t } ^ { \mathrm { ( F T ) } } = \frac { 1 } { N _ { t } } \sum _ { n = 1 } ^ { \tilde { N _ { t } } } \mathbf { 1 } ( \lVert \hat { y } _ { n } - y _ { n } ^ { t } \rVert \leq \delta ) } \end{array}$   
25 end   
26 else   
27 $E _ { s  t } ^ { \mathrm { ( F T ) } } = E _ { s  t } , S _ { s  t } ^ { \mathrm { ( F T ) } } = S _ { s  t }$   
28 $\theta _ { s  t } = \theta _ { s }$   
29 end   
30 end   
${ \mathrm { 3 1 ~ } } \mathbf { r e t u r n } \ \{ \theta _ { s  t } \} _ { ( s , t ) \in { \mathcal { T } } } , \ \{ E _ { s  t } ^ { \mathrm { ( F T ) } } \} _ { ( s , t ) \in { \mathcal { T } } } , \ \{ S _ { s  t } ^ { \mathrm { ( F T ) } } \} _ { ( s , t ) \in { \mathcal { T } } } \ \qquad $

In the fine-tuning stage, a small subset of the target dataset, $\mathcal { D } _ { t } ^ { ( \mathrm { F T } ) }$ , is randomly sampled, containing a fraction $\rho$ of the total data. This subset is used to update only the top k layers of the network, while the lower $( L - k )$ layers remain frozen to preserve general feature representations. The resulting adapted model $\theta _ { s  t }$ is then re-evaluated to produce updated metrics: localization error $E _ { s  t } ^ { \mathrm { ( F T ) } }$ and success rate $\dot { S } _ { s  t } ^ { ( \mathrm { F T } ) }$ . By repeating this process for every source–target pair, the algorithm generates a comprehensive set of fine-tuned models and performance metrics. These outputs form the input to Algorithm 2 which implements a selection procedure that identifies the optimal source–target area configuration using the TES. This metric jointly accounts for localization error and success rate, providing a balanced criterion for identifying the most effective TL setup.

Algorithm 2: TES-Based Source Model Selection and   
Unseen Area Validation   
Input $\mathbf { \lambda } : T = \{ ( s , t ) \mid s , t \in \mathcal { A } , s \neq t \} ; /$ \* All source-target pairs $^ { * }$   
Input : $\{ E _ { s  t } ^ { \mathrm { ( F T ) } } \} _ { ( s , t ) \in \mathcal { T } ^ { ; ~ / * } }$ Fine-tuned localization errors $^ { * }$   
Input : $\{ S _ { s  t } ^ { ( \mathrm { F T } ) } \} _ { ( s , t ) \in \mathcal { T } ^ { ; } } / { } ^ { * }$ Fine-tuned success rates $^ { * }$   
Input : $\dot { \alpha } \in [ 0 , \dot { 1 } ] ; { } / { } ^ { * }$ Error-success trade-off parameter $^ { * }$   
Input : $\{ \theta _ { s } \} _ { s \in \mathcal { A } } ^ { \ast } / { } ^ { \ast }$ Source models trained on $ \mathrm { A , B , C ~ ^ { * } } /$   
Input : ${ \dot { \mathcal { D } } } _ { D } ; { \mathsf { \Omega } } / { \ast }$ Unseen area dataset $^ { * }$   
Input : $\delta ; I ^ { \ast }$ Success threshold \*/   
Output: $s ^ { * } \colon / ^ { * }$ Optimal source area $^ { * }$   
$\theta _ { s ^ { * } } : / { } ^ { * }$ Best source model \*/   
$\breve { E } _ { s ^ { * } \to D } , S _ { s ^ { * } \to D ^ { : } } / { * }$ Performance on unseen area $^ { * }$   
1 Compute Global Min–Max Statistics   
2 $E _ { \operatorname* { m i n } } \gets \operatorname* { m i n } _ { ( s , t ) } E _ { s \to t } ^ { ( \mathrm { F T } ) } , E _ { \operatorname* { m a x } } \gets \operatorname* { m a x } _ { ( s , t ) } E _ { s \to t } ^ { ( \mathrm { F T } ) }$   
3 $S _ { \operatorname* { m i n } } \gets \operatorname* { m i n } _ { ( s , t ) } S _ { s \to t } ^ { ( \mathrm { F T } ) } , S _ { \operatorname* { m a x } } \gets \operatorname* { m a x } _ { ( s , t ) } S _ { s \to t } ^ { ( \mathrm { F T } ) }$   
4 Compute TES for Each Pair   
5 for each $( s , t ) \in \mathcal T$ do   
6 $\begin{array} { r } { \mathrm { N E } _ { s  t }  \frac { E _ { s  t } ^ { ( \mathrm { F T } ) } - E _ { \mathrm { m i n } } } { E _ { \mathrm { m a x } } - E _ { \mathrm { m i n } } } } \end{array}$   
7 $\begin{array} { r } { \mathrm { N S R } _ { s  t }  \frac { S _ { s  t } ^ { \ast \mathrm { ~ x ~ } _ { j } } - S _ { \operatorname* { m i n } } } { S _ { \operatorname* { m a x } } - S _ { \operatorname* { m i n } } } } \end{array}$   
8 $\mathrm { T E S } _ { s  t }  \tilde { \alpha } \tilde { ( 1 - \mathrm { N E } _ { s  t } ) } + ( 1 - \alpha ) ( \mathrm { N S R } _ { s  t } )$   
9 end   
10 Aggregate TES by Source Area   
11 for each ${ \mathfrak { s } } \in { \mathcal { A } }$ do   
12 $\begin{array} { r l } { | } & { { } \mathrm { T E S } _ { s } \gets \frac { 1 } { 2 } \sum _ { t \neq s } \mathrm { T E S } _ { s  t } } \end{array}$   
13 end   
14 Select Best Source Area   
15 $s ^ { * } \gets \arg \operatorname* { m a x } _ { { s } \in \mathcal { A } } \mathrm { T E S } _ { { s } }$   
16 $\theta _ { s ^ { * } } \gets \theta _ { s ^ { * } } \ / { * }$ Retain pretrained model from $s ^ { * } \ * /$   
17 Validate on Unseen Area   
18 Extract $X _ { D } , Y _ { D }$ from $\mathcal { D } _ { D }$   
19 $E _ { s ^ { * } \to D }  \mathcal { L } ( \stackrel { } { f } _ { \theta _ { s ^ { * } } } ( X _ { D } ) , Y _ { D } )$   
20 $\begin{array} { r } { S _ { s ^ { * } \right. D } \left. \frac { 1 } { N _ { D } } \sum _ { n = 1 } ^ { N _ { D } } \mathbf { 1 } ( \| \hat { y } _ { n } - y _ { n } ^ { D } \| \leq \delta ) } \end{array}$   
21 where $\boldsymbol { { \hat { y } } } _ { n } = \boldsymbol { \hat { f } } _ { \boldsymbol { \theta } _ { s ^ { * } } } ( \boldsymbol { x } _ { n } ^ { D } )$   
22 return $s ^ { \ast } , \theta _ { s ^ { \ast } } , E _ { s ^ { \ast }  D } , S _ { s ^ { \ast }  D }$

Algorithm 2 begins by considering all valid source–target area pairs $( s , t ) \in \mathcal { T } _ { : }$ , where $s \neq t ,$ , and their corresponding fine-tuned models $\theta _ { s  t } .$ Using the localization error $E _ { s  t } ^ { \mathrm { ( F T ) } }$ and success rate $\bar { S } _ { s  t } ^ { ( \mathrm { { \bar { F T } } ) } }$ which are computed during prior evaluations, the algorithm normalizes these values using min–max normalization to ensure comparability across different source–target scenarios. For each evaluated pair, the normalized error $\mathrm { N E } _ { s  t }$ and normalized success rate $\mathrm { N S R } _ { s  t }$ are computed, and a pairwise TES score $\mathrm { T E S } _ { s  t }$ is calculated using the weighted formulation in Eq. 14. Once all TES values are computed, the TES scores are aggregated by source area: for each source $s ,$ the two TES values corresponding to its target areas $( { \mathsf { e } } . { \mathsf { g } } . , \ A \ \to \ B$ and $A \  \ C )$ are averaged to produce an overall score $\mathrm { T E S } _ { s }$ . The source area with the highest aggregated TES, denoted by $s ^ { * }$ , is then selected as the optimal source for deployment. Finally, this chosen source model $\theta _ { s }$ ∗ is validated on the unseen area dataset $\mathcal { D } _ { D }$ to assess its true generalization capability. The evaluation uses the same metrics (localization error and success rate) to ensure consistency.

By aggregating TES scores per source and validating only the top-ranked model on unseen data, this updated framework avoids overfitting to a single transfer direction and ensures that the selected model represents a broadly transferable solution for VLC-based indoor localization.

## V. EXPERIMENTAL SETUP AND DATA COLLECTION

The proposed VLC-based localization framework is based on measured data collected in a realistic industrial environment. This section outlines the VLC transceiver design, the measurement infrastructure deployed within the BOSCH facility, and the process used to gather signal strength readings. These measurements provide a representative basis for model training and evaluation under practical deployment conditions.

## A. VLC Transceiver System

1) Architecture: The transceiver is composed of three main subsystems: the power supply, the transmitter (Tx), and the receiver (Rx), each designed to ensure high signal fidelity and robustness across varying channel conditions. As illustrated in Fig. 3, it features an optical front-end powered by a regulated 5V DC input, stabilized via dedicated DC-DC converters. These converters generate multiple supply rails required at various stages of the transceiver, including the biasing of the photodetector (PD).

On the Tx side, electrical signals are first processed through a pre-equalization stage to compensate for channel-induced distortions and bandwidth limitations of the LEDs. Since LEDs can only respond effectively to signals within a limited frequency range, signals containing higher frequency components may be distorted or attenuated. The pre-equalization stage mitigates these effects, enhancing signal fidelity before optical transmission. These equalized signals are then amplified by a source amplifier to meet the current requirements of the LED, which converts the electrical signal into an optical signal.

After free space propagation, the modulated optical signal is collected at the Rx side. The incoming optical signals are captured by a PD and converted into electrical currents. These currents are converted into voltage signals using a transimpedance amplifier (TIA), followed by a post-equalization block to compensate for distortions introduced by the Rx chain. The signal quality is further refined by an amplifier and filtering stage before being forwarded for downstream processing. The use of both pre- and post-equalization stages enhances endto-end system performance by mitigating distortions across the optical wireless channel. The signal flow is shown in the diagram, where black arrows denote electrical signals and red arrows represent optical signals.

2) Specifications: The transceiver unit is powered via a USB-C female connector, providing a convenient and standardized interface. The unit requires a 12 V supply at 1.2 A, resulting in a total power consumption of approximately 14.4 W. This ensures stable operation of both the optical frontend components and the digital control circuitry. Internally, dedicated DC-DC converters regulate the input voltage down to 5 V levels required by the LED drivers, photodiode amplifiers, and microcontrollers. The use of USB-C facilitates integration with commercially available power sources and provides robust mechanical durability for repeated connections in experimental settings.

![](images/4c93014c5d9df7b2eba4a4e44e7aa6d89d9d6f0aa33bf4719a7828510c5cbb53.jpg)  
Fig. 3: Architecture of the implemented VLC transceiver.

The Tx employs a LED driven by a current signal with an amplitude of 200 mA and an operational variation range of ± 100 mA. This range allows modulation of the light intensity while minimizing nonlinear distortion and preventing thermal damage to the LED. The emitted optical power is centered around 500 mW with a tolerance of ± 250 mW, balancing the need for strong signal propagation with eye safety limits and thermal management considerations. The LED has a full angular emission width of 60° at half maximum power, allowing a moderately wide illumination cone to cover the target area while limiting spillover into adjacent zones. The selected peak wavelength of 450 nm, with an operational range of 450-650 nm and a spectral width of 40 nm, aligns with the high sensitivity region of standard photodiodes and ensures compatibility with commercially available optical filters. The Tx output is connected via a SubMiniature version A (SMA) interface, providing a secure and low-loss connection to ensure robust coupling.

On the Rx side, four PDs have been put in parallel to increase the sensitive area of the Rx. The signals produced by these PDs are then summed up to get a single signal to process and send to the VNA. These PDs already have an integrated optical lens which limits the full angle at half-sensitivity to 60° but allows to collect more optical power so that they have an effective sensitive area of 150 mm² whereas their actual sensitive area is 26.4 mm². Eventually, our VLC transceiver thus have a total sensitive area of 600 mm², with a photosensitivity at 400 nm. The spectral sensitivity of the Rx spans from 380nm to 780nm, effectively covering the visible light spectrum. At the target wavelength of 450nm, the PDs provide a photosensitivity of 0.3A/W, ensuring sufficient conversion of the incident light into electrical current, supporting accurate signal detection even under low-light conditions.

Together, these specifications summarized in Table I enable a robust and tunable VLC system capable of capturing highfidelity signal strength measurements across varying lineof-sight conditions, making it suitable for accurate indoor localization tasks in complex industrial environments.

## B. VLC Measurement System

The complete setup, illustrated in Fig. 4, was deployed at the BOSCH manufacturing facility. It consists of two VLC transceiver units (Tx and Rx), a dual-port vector network analyzer (VNA), two signal and power converter boards, and two regulated DC power supplies.

TABLE I: VLC Transceiver Specifications.
<table><tr><td colspan="2">Power Supply</td></tr><tr><td>Supply voltage Supply current Power supply interface Tx</td><td>12 V 1.2 A USB-C female</td></tr><tr><td>Driving current signal range Transmitted optical power range Full angle at half power Wavelength of operation Spectral width Tx chain interface</td><td>200 mA ± 100 mA 500 mW ± 250 mW 60° 450 nm (450–650 nm) 40 nm SMA</td></tr><tr><td colspan="2">Rx Effective sensitive area 600 mm² 400 nm (380–780 nm)</td></tr></table>

![](images/69bb57f6c5f2fd17eb839a3a11a22898097f5bfeacca46a73bc18024bd7331da.jpg)  
Fig. 4: VLC measurement system.

The VNA serves as both the signal generator and measurement instrument. During measurements, it generates a frequency-swept sinusoidal signal at Port 1, which modulates the optical carrier transmitted through the VLC system. This signal is first routed through the Tx-side converter board, where it is combined with a 12V DC supply from the external power module. The merged signal travels via a network cable to the Tx module, where the electrical signal drives the LED through the pre-equalization and driver stages described above.

The modulated optical signal propagates through the industrial environment and is captured by the Rx module. The received light is converted into an electrical signal using a photodiode-based front-end, and further amplified through the post-equalization chain. The resulting signal is returned via the second converter board, which separates the RF and DC components before forwarding the recovered signal to Port 2

![](images/97bc259a13af6272b5f4ea7dd17b1edb537733907c7fed808cf47980f43efe19.jpg)  
Fig. 5: Composite picture from the Tx towards the measurement area in the industry hall at BOSCH in Blaichach.

of the VNA.

The use of converter boards enables a unified interface for signal and power transmission, simplifying cabling and improving measurement stability. The VNA records both the transmitted and received signals to compute signal strength measurements across different locations. These measurements, representing frequency-dependent power attenuation, serve as location-specific fingerprints and form the basis of the dataset used for training and evaluating the proposed TL based localization framework.

The system architecture allows for precise synchronization and repeatable operation of the Tx and Rx chains, ensuring that the collected signal strength measurements are consistent and representative of the actual indoor propagation dynamics. Such consistency is essential for constructing reliable datasets used in model adaptation and TL scenarios in complex industrial environments.

## C. Data Collection

The data collection took place in the industrial hall at the BOSCH factory in Blaichach, Germany. The complete hall has a dimension of 74 m × 171 m × 11 m, however, the measurement campaign focused on a representative measurement area, measuring 46 m × 42 m × 11 m. This segment contains one production line and a large storage area, as shown in the picture from the Tx shown in Fig. 5. The structural composition of the hall includes materials such as steel, concrete, glass, and aluminum. The external boundaries are formed by a combination of reinforced concrete walls, metal cladding, and glass surfaces. Inside, concrete columns are uniformly distributed along the interior. The ceiling-level part of the hall is densely equipped with metallic infrastructure such as ventilation ducts, suspended girders, and cable trays installed near the ceiling.

An overview of the VLC transceiver arrangement and the measurement grid layout is presented in Fig. 6a. To align with the physical constraints and operational setup of the production environment, the Tx and Rx front-ends were mounted at fixed heights of 2.9 meters and 1.1 meters, respectively. To ensure systematic data collection and facilitate consistent statistical analysis, the receiver area was segmented into a uniform grid with cells measuring 0.2 m × 0.2 m. Signal measurements were recorded at each grid intersection point, providing a dense and evenly distributed dataset across the coverage area. Fig. 6b displays the actual positions of the Txs and the measured area within the production line. The factoryprovided floor plan is annotated with the precise locations of Txs, measurement zones, and the absolute reference origin (marked with an orange star), which serves as the global coordinate starting point for all data acquisition. The measurement campaign encompasses ten distinct Tx deployment areas (pt. 1 – 10 ), distributed within Areas A, B, and C. As demarcated on the map by dashed red boundaries, pt. 1 – 3 belong to Area A, pt. 4 – 6 belong to Area B, pt. 8 – 10 corresponds to Area C. Notably, deployment pt. 7 outlined in blue—was intentionally excluded from model training and is designated as Area D, used solely to assess the generalization capability of the using the TES framework. The four deployment areas (A–D) were selected to represent structurally and operationally diverse environments, including differences in ceiling structures, lighting conditions, reflective surfaces, and equipment density, providing realistic variations for cross-area TL evaluation.

Each green-highlighted region indicates the coverage zone where RSS measurements were collected. On average, around 160 Rx positions were sampled per area. At each Rx position, the received optical signal was captured continuously for 5 seconds to account for temporal variability induced by ambient noise, equipment movement, and other environmental dynamics. The collected RSSI samples at each point were then averaged to produce a single robust signal strength value, effectively mitigating the impact of short-term disturbances and enabling more stable localization model training. Detailed 3D positions and corresponding area assignments of all Tx deployments are summarized in TABLE II.

## VI. PERFORMANCE EVALUATION AND ANALYSIS

This section presents a detailed evaluation of the proposed TL framework for VLC-based localization across different subareas of the Bosch factory hall. We analyze (i) the significance of TL when applying models across domains, (ii)

(a)  
![](images/ff8c14a351e78215eec0ff87fdd48cce447ecc234494ad7d20f448ffe558820c.jpg)

![](images/8f72379806711c2044d832de892a4c064a640250ac1817e534529c08a936c46d.jpg)  
Fig. 6: (a) Tx/Rx setup. (b) Floor plan layout of the measurement area.

TABLE II: 3D coordinates, placement context, and deployment areas of transmitters.
<table><tr><td rowspan=1 colspan=1>Tx ID</td><td rowspan=1 colspan=1>Position (x, y, z)</td><td rowspan=1 colspan=1>Physical Placement Context</td><td rowspan=1 colspan=1>Deployment Area</td></tr><tr><td rowspan=1 colspan=1>①</td><td rowspan=1 colspan=1>(2.6, 0.8, 2.9)</td><td rowspan=1 colspan=1>Near control station and transpar-ent windows</td><td rowspan=1 colspan=1>Area A</td></tr><tr><td rowspan=1 colspan=1>②</td><td rowspan=1 colspan=1>(1.7, -0.2, 2.9)</td><td rowspan=1 colspan=1>Near control station and transpar-ent windows</td><td rowspan=1 colspan=1>Area A</td></tr><tr><td rowspan=1 colspan=1>③</td><td rowspan=1 colspan=1>(3.7, 1.8, 2.9)</td><td rowspan=1 colspan=1>Near control station and transpar-ent windows</td><td rowspan=1 colspan=1>Area A</td></tr><tr><td rowspan=1 colspan=1>④</td><td rowspan=1 colspan=1>(17.1, 0.2, 2.9)</td><td rowspan=1 colspan=1>Material storage zone</td><td rowspan=1 colspan=1>Area B</td></tr><tr><td rowspan=1 colspan=1>⑤</td><td rowspan=1 colspan=1>(12.9, 0.8, 2.9)</td><td rowspan=1 colspan=1>Rear of production machinery (be-side cement pillars)</td><td rowspan=1 colspan=1>Area B</td></tr><tr><td rowspan=1 colspan=1>⑥</td><td rowspan=1 colspan=1>(17.1, 0.8, 2.9)</td><td rowspan=1 colspan=1>Rear of production machinery (ad-jacent to storage)</td><td rowspan=1 colspan=1>Area B</td></tr><tr><td rowspan=1 colspan=1>⑦</td><td rowspan=1 colspan=1>(14.2, 8.5, 2.9)</td><td rowspan=1 colspan=1>Beneath conveyor belts near lock-ers</td><td rowspan=1 colspan=1>Area D</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>(23.2, 4.6, 2.9)</td><td rowspan=1 colspan=1>Main production line</td><td rowspan=1 colspan=1>Area C</td></tr><tr><td rowspan=1 colspan=1>⑨</td><td rowspan=1 colspan=1>(15.2, 4.6, 2.9)</td><td rowspan=1 colspan=1>Main production line</td><td rowspan=1 colspan=1>Area C</td></tr><tr><td rowspan=1 colspan=1>①0</td><td rowspan=1 colspan=1>(13.5, 3.6, 2.9)</td><td rowspan=1 colspan=1>Main production line</td><td rowspan=1 colspan=1>Area C</td></tr></table>

the rationale for selecting the optimal fine-tuning data ratio, and (iii) TES-driven selection of the optimal TL source model and validation on an unseen area.

## A. Impact of TL on Cross-Domain Localization

To evaluate the necessity of TL, a DNN was trained on Area A and tested both on its own validation set and on datasets from Areas B and C. The training used the settings summarized in Table III.

Fig. 7 shows the validation learning curves for localization error, while Table IV summarizes the final localization errors and success rates after 1000 epochs for all source areas (A, B, and C), evaluated within-domain and cross-domain, with and without TL. On its own validation data, the Area A model achieved a localization error of 57.73 cm. Direct application to other areas resulted in substantial performance degradation, with errors rising to 134.91 cm on Area B and 101.08 cm on Area C.

TABLE III: TL Simulation Settings
<table><tr><td>Parameter</td><td>Description</td><td>Value</td></tr><tr><td>Optimizer</td><td>Model optimizer</td><td>ADAM</td></tr><tr><td>Learning rate, η</td><td>Step size in gradient descent</td><td>0.0001</td></tr><tr><td>Batch size, B</td><td>Number of samples per batch</td><td>64</td></tr><tr><td>Epochs per iteration, E</td><td>Number of training epochs</td><td>1000</td></tr><tr><td>Input dimension, d</td><td>Size of input feature vector</td><td>10</td></tr><tr><td>Hidden layers, L</td><td>Number of hidden layers</td><td>5</td></tr><tr><td>Hidden layer size, h</td><td>Number of neurons per hidden layer</td><td>512</td></tr></table>

Applying TL using 40% of the target area data (chosen based on the analysis in Section VI-B) restored cross-domain performance. For instance, fine-tuning reduced the Area A localization error on Area B to 40.90 cm, and on Area C to 51.55 cm. In some cases, the fine-tuned model even surpassed its original in-domain baseline; for example, the Area A model fine-tuned on Area C achieved a success rate of 98.48%, exceeding its original 80.00% on Area A.

These results confirm that models trained on one spatial domain cannot be reliably deployed to another without adaptation. TL not only restores performance but also enhances cross-domain generalization, underscoring its significance for practical VLC-based localization deployments.

Success-Rate Threshold Justification and Sensitivity Analysis: The localization success rate depends on the selected distance threshold δ (Eq. 9), which should reflect practical industrial requirements. In large indoor factory environments such as the industrial hall of Bosch considered in this work, a positioning accuracy of about 1 m is generally sufficient for applications such as asset tracking, worker-zone monitoring,

![](images/29f9079609079e3f9c7d16765f04e4ea51f6b7d8b04bee174350faa8a08fad87.jpg)  
(a)

![](images/68dd9a34b336dd8bc9640cf480c6487743798463694f062c12d4cf116da7c86e.jpg)  
(c)

![](images/edd547e82510c470e5bf25059a3158d81fc2101226a2cbaef9b9250c2b3ed466.jpg)  
(e)

![](images/78307038aef9a8b4598b124a52633bca49a4fca22b1dce63863a6fcb65b6c5d5.jpg)  
(b)

![](images/ae4acd90f9796a385ee3bc36a031641283bdd86fea13cd0879c5a09e322b0e22.jpg)  
(d)

![](images/2243d873146fe819660e4b38734650a480c71192328028702ebb04211049c8a4.jpg)  
(f)  
Fig. 7: TL performance evaluation across different validation areas before and after TL: (a) Area A localization error, (b) Area A success rate, (c) Area B localization error, (d) Area B success rate, (e) Area C localization error, and (f) Area C succes rate.

and coarse navigation of automated guided vehicles. Given equipment size, aisle spacing, and operational safety margins,

TABLE IV: TL results on different areas and metrics. A/B/C in parentheses denote the respective target area.
<table><tr><td rowspan="2">Source</td><td rowspan="2">Metric</td><td rowspan="2">Train (Source)</td><td rowspan="2">Val. (Source)</td><td colspan="2">Target 1 (Before/After TL)</td><td colspan="2">Target 2 (Before/After TL)</td></tr><tr><td>Before</td><td>After</td><td>Before</td><td>After</td></tr><tr><td rowspan="2">Area A</td><td>Error (cm)</td><td>57.73</td><td>57.74</td><td>134.91 (B)</td><td>40.90 (B)</td><td>101.08 (C)</td><td>51.55 (C)</td></tr><tr><td>Success (%)</td><td>84.47</td><td>80</td><td>23.66 (B)</td><td>86.02 (B)</td><td>30.68 (C)</td><td>98.48 (C)</td></tr><tr><td rowspan="2">Area B</td><td>Error (cm)</td><td>32.09</td><td>39.57</td><td>91.04 (A)</td><td>67.20 (A)</td><td>69.70 (C)</td><td>51.88 (C)</td></tr><tr><td>Success (%)</td><td>95.84</td><td>90.32</td><td>37.05 (A)</td><td>97.19 (A)</td><td>59.09 (C)</td><td>98.24 (C)</td></tr><tr><td rowspan="2">Area C</td><td>Error (cm)</td><td>48.22</td><td>50.58</td><td>121.28 (A)</td><td>63.88 (A)</td><td>153.66 (B)</td><td>43.09 (B)</td></tr><tr><td>Success (%)</td><td>100</td><td>100</td><td>44.42 (A)</td><td>81.78 (A)</td><td>58.06 (B)</td><td>90.55 (B)</td></tr></table>

TABLE V: Success rate sensitivity to distance threshold δ across all source–target transfers.
<table><tr><td>Transfer</td><td>SR @0.5 m</td><td>SR @1.0 m</td><td>SR @1.5 m</td><td>SR @2.0 m</td></tr><tr><td>A→B</td><td>78%</td><td>86%</td><td>96%</td><td>100%</td></tr><tr><td>A→C</td><td>77%</td><td>98%</td><td>100%</td><td>100%</td></tr><tr><td>B→A</td><td>81%</td><td>97%</td><td>100%</td><td>100%</td></tr><tr><td>B→C</td><td>79%</td><td>98%</td><td>100%</td><td>100%</td></tr><tr><td>C→A</td><td>61%</td><td>82%</td><td>96%</td><td>100%</td></tr><tr><td>C→B</td><td>76%</td><td>91%</td><td>97%</td><td>100%</td></tr><tr><td>Avg.</td><td>75%</td><td>92%</td><td>98%</td><td>100%</td></tr></table>

δ = 1 m is adopted as a realistic and practically meaningful benchmark throughout the paper.

To assess the dependence of the success rate metric on δ, we evaluated localization performance using thresholds of 0.5 m, 1.0 m, 1.5 m, and 2.0 m across all source–target transfers. The results are summarized in Table V. As expected, the success rate increases monotonically with the threshold, while the relative performance trends between transfer pairs remain consistent. This indicates that TES uses the success rate primarily as a comparative performance indicator rather than an absolute localization-accuracy measure, and that the conclusions of the study are not sensitive to moderate variations in the chosen threshold.

## B. Fine-Tuning Data Ratio Justification

As outlined in Section VI-A, TL was performed using 40% of the target-domain data. This ratio was determined based on a systematic evaluation of different fine-tuning data proportions. Specifically, we compared TL performance when fine-tuning with 10%, 20%, 40%, 60%, and 100% of the available target-area datasets. Fig. 8 illustrates the resulting localization errors and success rates.

At low data ratios (10–20%), TL provides noticeable improvement, but performance remains suboptimal. For example, TL from Area A to Area B with only 10% of the target data yields a localization error of 47.34 cm and a success rate of 75.63%. Increasing the ratio to 20% improves the error to 44.43 cm and the success rate to 81.36%. The most significant gain occurs between 20% and 40%. At 40%, the Area A→Area B error drops further to 40.90 cm with 86.02% success, while for Area A→Area C, the error decreases to 51.55 cm with an impressive 98.48% success rate. Beyond this point, the performance curve flattens: using 60% or even 100% of the target data results in only marginal gains, suggesting the model approaches performance saturation. Fig. 8 focuses on two representative transfer scenarios (A→B and A→C) to avoid clutter, while Table VI reports the complete results for all six source–target domain combinations across all data ratios.

![](images/f48b501fc356e2a61fe34ea8570879dd79319b751f95176adeb68e9546a124a3.jpg)  
(a)

![](images/dee594f5160c0e02778e0591f026a891a4750475f3f7e1e0084ba4901b48717e.jpg)  
(b)

![](images/bdac311632d63b02ea788e32f7414d3ee6876af84de480d28d556cef102839d6.jpg)  
(c)

![](images/6b005d7a46f4f3e384179b8d94a9ecceda7c5d24e4cc08d93058f2540409c623.jpg)  
(d)  
Fig. 8: Localization error and success rate trends for representative TL scenarios across different fine-tuning data ratios: (a) A→B localization error, (b) A→C localization error, (c) A→B success rate, and (d) A→C success rate.

Given these results, we selected 40% as the standard finetuning ratio for all subsequent experiments, as it offers a practical tradeoff between model performance and the data collection effort required in real-world deployments.

The result that approximately 40% of target-area data recovers most of the full-data performance is an empirical observation for the industrial dataset used in this study. In our earlier work [69], TL was evaluated under lighting variations within the same deployment area, where similar performance recovery was achieved using a comparable calibration fraction. Together, these studies indicate that moderate calibration effort can be effective under both spatial (cross-area) and optical (light/noise) domain variations typical of industrial VLC systems. Nevertheless, the exact calibration ratio may vary with environmental complexity, occlusion, and noise conditions, and should therefore be interpreted as an empirical guideline rather than a universal requirement.

TABLE VI: Localization error (cm) and success rate (%) for all six source–target transfer scenarios across five fine-tuning data ratios.
<table><tr><td rowspan="2">Data %</td><td colspan="6">Error (cm)</td><td colspan="6">Success  $( \% )$ </td></tr><tr><td> $\mathbf { A } { \xrightarrow { } } \mathbf { B }$ </td><td> $\mathbf { A } { \xrightarrow { } } \mathbf { C }$ </td><td> $\mathrm { B } \to \mathrm { A }$ </td><td> $\mathrm { B } {  } \mathrm { C }$ </td><td> $\mathrm { C } {  } \mathrm { A }$ </td><td> $\mathrm { C } {  } \mathrm { B }$ </td><td> $\mathbf { A } { \xrightarrow { } } \mathbf { B }$ </td><td> $\mathbf { A } { \xrightarrow { } } \mathbf { C }$ </td><td> $\mathrm { B } \to \mathrm { A }$ </td><td> $\mathrm { B } {  } \mathrm { C }$ </td><td> $\mathrm { C } {  } \mathrm { A }$ </td><td> $\mathrm { C } {  } \mathrm { B }$ </td></tr><tr><td>10</td><td>47.34</td><td>53.93</td><td>71.4</td><td>58.14</td><td>68.84</td><td>46.89</td><td>75.63</td><td>96.97</td><td>93.68</td><td>97.35</td><td>87.02</td><td>92.47</td></tr><tr><td>20</td><td>44.43</td><td>53.22</td><td>70.07</td><td>54.46</td><td>66.2</td><td>45.04</td><td>81.36</td><td>96.97</td><td>95.79</td><td>98.11</td><td>86.67</td><td>92.47</td></tr><tr><td>40</td><td>40.90</td><td>51.55</td><td>67.2</td><td>51.88</td><td>63.87</td><td>43.09</td><td>86.02</td><td>98.48</td><td>96.84</td><td>99.11</td><td>85.96</td><td>93.55</td></tr><tr><td>60</td><td>40.53</td><td>51.06</td><td>66.58</td><td>51.35</td><td>63.6</td><td>42.16</td><td>87.46</td><td>98.48</td><td>97.19</td><td>99.36</td><td>87.37</td><td>93.91</td></tr><tr><td>100</td><td>40.01</td><td>50.26</td><td>63.89</td><td>51.18</td><td>61.74</td><td>41.14</td><td>88.17</td><td>98.86</td><td>98.95</td><td>99.62</td><td>87.37</td><td>94.62</td></tr></table>

TABLE VII: TES results for all source–target pairs, showing post-TL metrics, normalized values, and computed TES.
<table><tr><td>Source→Target</td><td colspan="2">Post-TL Metrics Error (cm)  $\mathrm { S u c c e s s \ ( \% ) }$ </td><td colspan="2">Normalized Values NE NSR</td><td colspan="2">TES</td><td colspan="2"> $\operatorname { A v g } .$  TES (per source area)</td></tr><tr><td>A→B</td><td>40.90</td><td></td><td>0.0000</td><td>0.2537</td><td>0.6269</td><td></td><td rowspan="2">0.7132</td></tr><tr><td>A→C</td><td>51.55</td><td>86.02 98.48</td><td>0.4008</td><td>1.0000</td><td>0.7996</td><td></td></tr><tr><td>B→A</td><td>67.20</td><td>97.19</td><td>1.0000</td><td>0.9215</td><td>0.4607</td><td></td><td rowspan="2">0.6224</td></tr><tr><td>B→C</td><td>51.88</td><td>98.24</td><td>0.4186</td><td>0.9869</td><td>0.7841</td><td></td></tr><tr><td>C→A</td><td>63.88</td><td>81.78</td><td>0.8857</td><td>0.0000</td><td></td><td>0.0572</td><td rowspan="2">0.3876</td></tr><tr><td>C→B</td><td>43.09</td><td>90.55</td><td>0.0824</td><td>0.5182</td><td>0.7179</td><td></td></tr></table>

TABLE VIII: TES sensitivity to the weighting parameter α.

![](images/4a6a30a486bb25543794a365e1aca3f01314acf3fc54377dd5d3575e13ab5dfc.jpg)

(a)  
![](images/d500449bc4db28b5690be0ce92343d9de2b54996d52d0b044a99d63a57a1d9a2.jpg)  
(b)  
Fig. 9: Comparison of TES-selected and baseline models on unseen Area D: (a) localization error and (b) success rate.

## C. TES-Driven Selection ofthe Optimal TL Source Model and Validation on an Unseen Area

To objectively determine which source model offers the most transferable knowledge, we adopt the TES computation pipeline introduced in Section III and formulated in Algorithm 2 of Section IV. Table VII reports post-TL performance metrics, normalized values, and computed TES for all source–target pairs. By averaging TES values for each source’s outgoing transfer paths, the model trained on Area A achieved the highest mean TES (0.7132), making it the most transferable source model in this study.

<table><tr><td>α 一</td><td> $\mathbf { A v \mathbf { g } }$  TES (A)</td><td> $\mathbf { A v } \mathbf { g }$  TES (B)</td><td> $\mathbf { A v } \mathbf { g }$  TES (C)</td><td>Best Source</td></tr><tr><td>0.30</td><td>0.679</td><td>0.755</td><td>0.336</td><td>B</td></tr><tr><td>0.40</td><td>0.696</td><td>0.688</td><td>0.362</td><td>A ≈ B</td></tr><tr><td>0.50</td><td>0.713</td><td>0.622</td><td>0.388</td><td>A</td></tr><tr><td>0.60</td><td>0.731</td><td>0.556</td><td>0.414</td><td>A</td></tr><tr><td>0.70</td><td>0.748</td><td>0.490</td><td>0.439</td><td>A</td></tr></table>

Sensitivity Analysis of TES with respect to the Weighting Parameter $\alpha \mathrm { : }$ TES is defined in Eq. 14 as a weighted combination of localization error and success rate metrics, where the parameter α controls the balance between accuracy and robustness. In the original evaluation, we adopted $\alpha = 0 . 5$ to assign equal importance to both components.

To examine the impact of this design choice, a sensitivity analysis was conducted by recomputing TES for $\alpha \in$ $\{ 0 . 3 , 0 . 4 , 0 . 5 , 0 . 6 , 0 . 7 \}$ . For each α value, TES was calculated for every source–target pair $( \mathrm { e } . \mathrm { g } . , \mathrm { A } {  } \mathrm { B } , \mathrm { A } {  } \mathrm { C } )$ using Eq. 14, and then averaged across target areas to obtain the sourcelevel TES ranking, following the same procedure described in Section III-B4. The detailed TES results for the reference case $\alpha \ : = \ : 0 . 5$ are reported in Table VII, while Table VIII summarizes the corresponding source-level TES values for the different α configurations. The rightmost column of Table VIII identifies the source achieving the highest TES, i.e., the most transferable source under each weighting configuration.

As shown in Table VIII, the relative ranking of source areas remains largely stable across the tested range. Minor numerical variations appear only at the extremes, reflecting the expected trade-off between robustness-oriented (low α)

and precision-oriented (high α) weighting. Specifically, for low α values, source B slightly outperforms A, whereas for higher $\alpha ,$ source A gains a clearer advantage. Despite these small shifts, the identification of the most transferable source remains consistent for moderate variations of α. These results indicate that the TES-based conclusions are not sensitive to reasonable changes in the weighting parameter, supporting the use of $\alpha = 0 . 5$ as a balanced operating point.

Validation of TES using statistical Feature-Space Similarity: To quantitatively examine whether TES reflects physical similarity between deployment areas, we computed the Bhattacharyya distance between the RSS feature distributions of each area pair. Each area is represented by a threedimensional RSS feature vector corresponding to the three VLC transmitters. For every area, the empirical mean vector and covariance matrix were estimated across all measurement locations, thereby modeling the statistical distribution of the RSS fingerprints within that area.

The Bhattacharyya distance jointly accounts for (i) the Mahalanobis distance between the mean feature vectors and (ii) differences in covariance structure (i.e., variability and correlation patterns). It is computed as

$$
D _ { B } = \frac { 1 } { 8 } ( \mu _ { 1 } - \mu _ { 2 } ) ^ { T } \Sigma ^ { - 1 } ( \mu _ { 1 } - \mu _ { 2 } ) + \frac { 1 } { 2 } \ln \left( \frac { \mathrm { d e t } \Sigma } { \sqrt { \mathrm { d e t } \Sigma _ { 1 } \mathrm { d e t } \Sigma _ { 2 } } } \right) ,\tag{18}
$$

where $\mu _ { 1 }$ and $\mu _ { 2 }$ are the mean RSS feature vectors of the two areas, $\Sigma _ { 1 }$ and $\Sigma _ { 2 }$ are their corresponding covariance matrices, and $\Sigma = ( \Sigma _ { 1 } + \Sigma _ { 2 } ) / 2$ denotes the average covariance matrix. The superscript $( \cdot ) ^ { T }$ denotes the transpose operator, det(·) denotes the determinant of a matrix, and ln(·) denotes the natural logarithm. Smaller values of $D _ { B }$ indicate similarity in feature space.

Table IX presents the Bhattacharyya distance and corresponding Bidirectional Avg. TES for all evaluated area pairs. Bidirectional TES is computed as the average of TES values in both directions (source→target and target→source) from Table VII. The results reveal a clear inverse relationship: area pairs with lower Bhattacharyya distance consistently exhibit higher TES, while pairs with larger statistical separation show reduced transfer effectiveness. This confirms that TES is influenced by the underlying similarity of RSS feature distributions and quantitatively validates its sensitivity to physical compatibility between environments.

TABLE IX: Bhattacharyya Distance vs. Bidirectional Avg. TES.
<table><tr><td>Area Pair</td><td>Bhattacharyya Distance  $( D _ { B } )$ </td><td>Bidirectional Avg. TES</td></tr><tr><td>A-B</td><td>0.137</td><td>0.544</td></tr><tr><td>A-C</td><td>0.213</td><td>0.428</td></tr><tr><td>B-C</td><td>0.107</td><td>0.751</td></tr></table>

Validation on an Unseen Area: To assess whether TESbased selection generalizes to completely unseen domains, we evaluated all three source models (trained on Areas A, B, and C) on Area D (deployment zone P7), which was excluded from all training and fine-tuning. This allowed us to verify not only overall performance but also whether the TES-identified source model (Area A) indeed provided the best transferability in practice. For each source model, we tested two conditions: Baseline (model trained only on its own source area and applied directly to Area D) and TES-TL (the TES-selected source model fine-tuned via TL). Seven independent random 20% splits of Area D were used for evaluation, and localization error and success rate were recorded.

As shown in Fig. 9, the TES-TL model originating from Area A achieved the lowest mean localization error (3.28 cm) and the highest success rate (100%), outperforming all Baseline models as well as TL models from the other areas. This outcome can be explained by the physical similarity between Areas A and D. Both share mixed structural characteristics, including reinforced concrete columns and partial boundary openings (windows in A, wide passages and locker space in D), combined with moderate equipment density. These conditions produce comparable signal propagation patterns with limited occlusion and moderate multipath, making knowledge transfer more effective.

By contrast, Area B (storage/supermarket zone) is relatively open and static, which limited the diversity of its training data and reduced its ability to generalize to Area D, where large pillars introduce non-trivial shadowing. Area $\mathrm { C } ,$ on the other hand, is the busiest production line, characterized by dense machinery and frequent worker movement. Models from C capture strong multipath and temporal variability, which overfit to highly dynamic conditions and transfer less effectively to Area D’s structurally simpler layout.

Finally, when the unseen-area validation was rotated across all areas (A, B, C, and D), TES consistently selected the source model that shared the closest structural and environmental characteristics with the held-out area. This confirms that TES is not only a statistically reliable metric but also implicitly captures the physical compatibility of environments, making it a robust criterion for identifying the most transferable source model in practical deployment scenarios.

## D. Motivation and Impact of Fine-Tuning Strategy

As observed in Tables IV and VI, models trained directly on 100% of the target-area data can slightly outperform TL models that were pre-trained in a different source area and subsequently fine-tuned using the same complete target dataset. This behavior is consistent with negative-transfer effects reported in deep TL, where source-domain initialization may bias optimization toward parameter regions that are suboptimal for the target distribution. In the present framework, this effect is further influenced by the adopted partial fine-tuning strategy, where lower layers are kept frozen to preserve source-domain representations. This design was motivated by our preliminary observations showing that fully unfreezing the network in lowdata regimes can lead to overfitting and unstable training, thereby degrading generalization performance.

To further examine this behavior, we conducted an additional experiment in which only the fine-tuning strategy was modified, while all other factors, including the DNN architecture, optimizer, training protocol, data splits, and source–target transfer settings, were kept unchanged. Specifically, we compared two strategies: (i) partial fine-tuning, where the lower layers remain frozen as used in the proposed data-efficient TL framework, and (ii) full fine-tuning, where all network layers are unfrozen and updated during adaptation.

![](images/1d25df3f1ff8a9ba5ca93d82af04ce5645893bff021544df8782d64c75b87fa5.jpg)

![](images/b576fee61c4252ab20c4dc33bebe1ec9bfd1b70a70d7bb84ecef9fc9677ff19d.jpg)

(a)  
![](images/59a93e96e1729a93f4d37285af9b7b8848d8012f4854b25e56abc3363ceda815.jpg)

![](images/c45bc532090e40ff8e448510f2a211b3668602582557c48afd031d09e21ae836.jpg)  
(b)  
Fig. 10: Comparison of partial and full fine-tuning for representative transfers to Area A under different target-data ratios: (a) B→A transfer and (b) C→A transfer.

Fig. 10 shows representative results for the B→A and C→A transfer directions. A clear trend can be observed. In low-data regimes, particularly when only 10%–40% of the target data is available, partial fine-tuning provides more stable performance by preserving transferable source-domain representations and reducing the risk of overfitting. In contrast, when larger amounts of target data are available, especially in the 60%–100% range, full fine-tuning becomes more effective because it allows the layers to adapt all learned representations to the target-area distribution.

For B→A, partial fine-tuning with 100% target data reaches 63.90 cm, whereas full fine-tuning reduces the localization error to 57.38 cm. Similarly, for C→A, the 100% target-data error decreases from 61.74 cm under partial fine-tuning to 57.45 cm under full fine-tuning. These values are comparable to the target-only baseline of area A reported in Table IV. Therefore, the degradation observed under full target data is not an inherent limitation of TL itself, but rather a consequence of applying a fixed partial-freezing strategy across all data regimes.

These observations further clarify the role of TL within the scope of this work. Under the 100% target-data setting, fully unfreezing the layers provides sufficient adaptation capacity to better match the target-area distribution and mitigates the degradation observed under partial fine-tuning; however, this behavior should be interpreted as a high-data mitigation strategy, since full fine-tuning in low-data regimes can increase overfitting and training instability. Moreover, acquiring such comprehensive datasets typically requires extensive measurement campaigns, which are often impractical in real industrial settings. Therefore, TL is positioned here primarily as a data-efficiency mechanism, aimed at reducing the need for exhaustive target-area data collection while maintaining high localization performance.

## E. Limitations’ Discussion and Future Work

Although the proposed TL-based VLC localization framework demonstrates effective cross-area adaptation, several practical limitations remain that motivate future work.

(i) Single-Facility Evaluation: The experimental campaign was conducted within a single industrial facility due to practical constraints associated with measurements in active production environments, including safety procedures, coordination with manufacturing schedules, and data confidentiality requirements. While multiple zones with diverse layouts, equipment configurations, and lighting conditions were evaluated, validation across additional industrial facilities remains a limitation. Efforts are underway to extend the VLC dataset with measurements from a second industrial site.

(ii) Height Variations: In the current study, Tx and Rx heights were fixed within the typical installation range of ceiling-mounted LEDs and receiver modules in the industrial facility. In scenarios involving mobile robots or handheld devices, height variations may introduce geometric distortion in RSS patterns and affect localization accuracy. Future work will investigate strategies to address such variations, including augmenting training data with height perturbations, incorporating height as an additional input feature, or jointly estimating position and height within the TL-based localization framework.

(iii) Semi-Supervised and Representation-Preserving Adaptation: The present framework focuses on supervised TL using a limited labeled subset of the target area. This design matches the practical objective of reducing target-area measurement effort while maintaining reliable localization performance. Nevertheless, semi-supervised learning represents a natural extension of the proposed framework, particularly in industrial settings where unlabeled VLC measurements may be easier to collect than accurately labeled position data. Techniques such as pseudo-labeling, consistency regularization, and representation-preserving adaptation could further reduce labeling requirements and improve robustness under larger domain shifts. Such methods may also complement the fine-tuning strategy analyzed in Section VI-D by exploiting unlabeled target-domain data while preserving useful sourcedomain representations. Exploring these semi-supervised extensions is therefore an important direction for future work.

(iv) TES Scalability and Deployment Realism: The proposed TES requires evaluating transfer performance across source–target area pairs using labeled data, which can increase computational cost as the number of deployment areas grows. However, TES is designed as an offline planning metric used during system commissioning phase and therefore does not affect runtime localization performance. In typical industrial VLC deployments, the number of operational zones is limited, and TES evaluation is performed only once when configuring the localization framework, making the associated computational cost manageable. Nevertheless, we acknowledge that exhaustive TES evaluation may become expensive in very large-scale or rapidly expanding deployments. In such scenarios, TES could be approximated or predicted a priori using unsupervised domain-similarity indicators that do not require labeled target data, such as distributional divergence between RSS feature spaces, feature-alignment metrics, or geometry-aware channel similarity measures derived from VLC propagation characteristics. Such proxy indicators may provide a lightweight mechanism for estimating cross-domain transferability before fine-tuning. Investigating scalable TES approximation strategies is an important direction for future work.

(v) Multi-Source or Hybrid Transfer: The current framework considers transfer from a single source environment at a time. In practical industrial deployments, multiple pretrained source models may be available. Leveraging these models through hybrid strategies, such as dynamic source selection, ensemble initialization, or representation fusion, could improve localization performance by capturing complementary knowledge from different environments. However, multi-source transfer introduces additional design challenges, including increased computational cost, model selection complexity, and potential negative-transfer interactions between heterogeneous domains. Exploring these trade-offs and developing efficient multi-source adaptation strategies represents an important direction for future work.

(vi) Comparison With Advanced Adaptation Techniques: Several adaptation strategies such as domainadversarial training and meta-learning have shown promising results in RF-based indoor localization. Although these approaches require additional optimization stages, complex training procedures, or large multi-environment datasets, they represent valuable directions for future research in VLC localization. A systematic comparison between TL and other advanced adaptation methods will be investigated in future work to better understand trade-offs among accuracy, data requirements, computational complexity, and deployment practicality in industrial VLC environments.

## VII. CONCLUSION

In conclusion, this work introduced a TL framework that makes VLC-based indoor localization practical for Industry 4.0 deployments. By leveraging pretrained models and selectively fine-tuning on only a fraction of target-area data, we cut labeling requirements by 60% while retaining nearly full performance. The proposed TES eliminates the need for exhaustive testing by reliably identifying the most promising source model, guiding adaptation across domains. Experiments on four areas of a real Bosch manufacturing hall demonstrated that the TES-guided approach achieves localization errors as low as 40.9 cm with success rates exceeding 86%, and even delivers 3.28 cm mean error when applied to an unseen area. These results confirm that the framework not only restores but also strengthens cross-domain performance, offering a scalable and data-efficient path for VLC localization in evolving industrial environments. Future work will explore the collection of Wi-Fi and other wireless data from the Bosch hall to investigate TL between sensing technologies, enabling cross-modal localization.

## ACKNOWLEDGMENT

The authors gratefully acknowledge the financial supports of the EU Horizon 2020 program towards the 6G BRAINS project H2020-ICT 101017226.

## REFERENCES

[1] Y. Yang, M. Chen, Y. Blankenship, J. Lee, Z. Ghassemlooy, J. Cheng, and S. Mao, “Positioning using wireless networks: Applications, recent progress and future challenges,” IEEE Journal on Selected Areas in Communications, 2024.

[2] A. Franko, G. Vida, and P. Varga, “Reliable identification schemes for´ asset and production tracking in industry 4.0,” Sensors, vol. 20, no. 13, p. 3709, 2020.

[3] I. Marasovic, G. Maji´ c, I.´ Skalic, and<sup>ˇ</sup> Z. Tomasovi<sup>ˇ</sup> c, “Indoor Local-´ ization of Industrial IoT Devices and Applications Based on Recurrent Neural Networks,” Journal of communications software and systems, vol. 20, no. 1, pp. 137–145, 2024.

[4] A. Ullah, M. Younas, and M. S. Saharudin, “Digital Twin Framework using Real-Time Asset Tracking for Smart Flexible Manufacturing System.” Machines, vol. 13, no. 1, 2025.

[5] T. Aziz and K. Insoo, “Enhancing Indoor Localization Accuracy through Multiple Access Point Deployment,” Electronics, vol. 13, no. 16, p. 3307, 2024.

[6] S. J. Pan, V. W. Zheng, Q. Yang, and D. H. Hu, “Transfer Learning for Wifi-based Indoor Localization,” in Association for the advancement of artificial intelligence (AAAI) workshop, vol. 6. The Association for the Advancement of Artificial Intelligence Palo Alto, 2008.

[7] A. Loffredo, M. C. May, A. Matta et al., “Reinforcement learning for energy-efficient control of multi-stage production lines with parallel machine workstations,” in Italian Manufacturing Association Conference: XVI AITeM, vol. 35. Materials Research Forum LLC, 2023, p. 428.

[8] C. Wu, Z. Yang, and C. Xiao, “Automatic radio map adaptation for indoor localization using smartphones,” IEEE Transactions on Mobile Computing, vol. 17, no. 3, pp. 517–528, 2017.

[9] H. H. R. Sherazi, L. A. Grieco, M. A. Imran, and G. Boggia, “Energyefficient LoRaWAN for industry 4.0 applications,” IEEE Transactions on Industrial Informatics, vol. 17, no. 2, pp. 891–902, 2020

[10] F. Milano, H. da Rocha, M. Laracca, L. Ferrigno, A. Esp´ırito Santo, J. Salvado, and V. Paciello, “BLE-based Indoor Localization: Analysis of Some Solutions for Performance Improvement,” Sensors, vol. 24, no. 2, p. 376, 2024.

[11] J. Dai, M. Wang, B. Wu, J. Shen, and X. Wang, “A Survey of Latest Wi-Fi Assisted Indoor Positioning on Different Principles,” Sensors, vol. 23, no. 18, p. 7961, 2023.

[12] J. L. S. Gonzalez, L. M. S. Morillo, J. A. ´ Alvarez-Garc <sup>´</sup> ´ıa, F. E. D. S. Ros, and A. R. J. Ruiz, “Energy-Efficient Indoor Localization WiFi-Fingerprint System: An Experimental Study,” IEEE Access, vol. 7, pp. 162 664–162 682, 2019.

[13] A. Lindemann, B. Schnor, J. Sohre, and P. Vogel, “Indoor Positioning: A Comparison of WiFi and Bluetooth Low Energy for Region Monitoring,” in International Conference on Health Informatics, vol. 6. SCITEPRESS, 2016, pp. 314–321.

[14] H. Wang, G. Ganesh, M. Zon, O. Ghosh, H. Siu, and Q. Fang, “A BLE based Turnkey Indoor Positioning System for Mobility Assessment in Aging-In-Place Settings,” PLOS Digital Health, vol. 4, no. 4, p. e0000774, 2025.

[15] A. Guidara, F. Derbel, G. Fersi, S. Bdiri, and M. B. Jemaa, “Energyefficient on-demand indoor localization platform based on wireless sensor networks using low power wake up receiver,” Ad Hoc Networks, vol. 93, p. 101902, 2019.

[16] J. Xiao, Z. Zhou, Y. Yi, and L. M. Ni, “A Survey on Wireless Indoor Localization from the Device Perspective,” ACM Computing Surveys (CSUR), vol. 49, no. 2, pp. 1–31, 2016.

[17] S. Khriji, D. El Houssaini, and O. Kanoun, “Design of an Energy-Efficient Hybrid UWB-RF Indoor Distance Estimation System.” in SENSORNETS, 2022, pp. 249–255.

[18] S. O. Schmidt and H. Hellbruck, “Detection and Identification of¨ Multipath Interference With Adaption of Transmission Band for UWB Transceiver Systems.” in IPIN-WiP, 2021.

[19] T. Aziz and I. Koo, “A Comprehensive Review of Indoor Localization Techniques and Applications in Various Sectors.” Applied Sciences (2076-3417), vol. 15, no. 3, 2025.

[20] Y. Almadani, D. Plets, S. Bastiaens, W. Joseph, M. Ijaz, Z. Ghassemlooy, and S. Rajbhandari, “Visible Light Communications for Industrial Applications—Challenges and Potentials,” Electronics, vol. 9, no. 12, p. 2157, 2020.

[21] R. F. Miranda, C. H. Barriquello, V. A. Reguera, G. W. Denardin, D. H. Thomas, F. Loose, and L. S. Amaral, “A review of Cognitive Hybrid Radio Frequency/Visible Light Communication Systems for Wireless Sensor Networks,” Sensors, vol. 23, no. 18, p. 7815, 2023.

[22] X. WANG, H. YAN, and J. SONG, “Application of Visible Light Communication Technology in Digital Operating Room: Experiment and Equipment Manufacturing,” Clin Surg. 2024; 9, vol. 3723.

[23] S. Ma, B. Li, G. Zhang, H. Li, C. Qiu, C. Yu, S. Li, and C. Shen, “Centimeter-Level 3-d Mobile Online Visible Light Positioning System with Single Led Lamp,” IEEE Internet of Things Journal, vol. 11, no. 1, pp. 418–429, 2023.

[24] A.-M. Cailean, S.-A. Av ˘ at˘ am˘ anit ˘ <sub>,</sub>ei, and C. Beguni, “Driving Toward Connectivity: Vehicular Visible Light Communications Receiver with Adaptive Field of View for Enhanced Noise Resilience and Mobility,” Sensors, vol. 24, no. 9, p. 2814, 2024.

[25] J. A. Apolo, I. N. Osahon, B. Ortega, V. Almenar, J. Tang, and S. Rajbhandari, “Experimental Demonstrations of High-Accuracy 3D/2D Indoor Visible Light Positioning using Imaging Multiple-Input Multiple-Output Receivers and Artificial Neural Networks,” IET Optoelectronics, vol. 19, no. 1, p. e70000, 2025.

[26] W. Zhao, A. Goudar, M. Tang, and A. P. Schoellig, “Ultra-wideband Time Difference of Arrival Indoor Localization: From Sensor Placement to System Evaluation,” arXiv preprint arXiv:2412.12427, 2024.

[27] A. S. Oluwole, O. J. Olaluyi, A. E. Amoran, J. O. Adeogo, O. Akinsanmi, A. Sobowale, O. Femi-Jemilohun, and N. Okomba, “Exploration of Visible Light Communication in an Indoor Location Using LED Array Distribution,” 2024.

[28] S. Santhanalakshmi, B. S. SK, C. Asrija et al., “VLC Based Indoor Tracking System with IOT Network for Smart Buildings.”

[29] A. M. Ghonim and W. M. Salama, “Indoor Localization with Ensemble Machine Learning via Visible Light Communication Channels,” Journal of Engineering Research, vol. 8, no. 6, p. 16, 2024.

[30] M. Jan, W. Njima, and X. Zhang, “A Privacy-Preserving Indoor Localization System based on Hierarchical Federated Learning,” in 2024 14th International Conference on Indoor Positioning and Indoor Navigation (IPIN). IEEE, 2024, pp. 1–6.

[31] B. Wickramasinghe, G. Saha, and K. Roy, “Continual Learning: A Review of Techniques, Challenges, and Future Directions,” IEEE Transactions on Artificial Intelligence, vol. 5, no. 6, pp. 2526–2546, 2023.

[32] D. Su, X. Wang, S. Liu, and W. Ding, “Four-Dimensional Indoor Visible Light Positioning: A Deep-Learning-based Perspective,” Journal of the Franklin Institute, vol. 360, no. 6, pp. 4071–4090, 2023.

[33] Z. Zhu, Y. Yang, M. Chen, C. Guo, J. Cheng, and S. Cui, “A Survey on Indoor Visible Light Positioning Systems: Fundamentals, Applications, and Challenges,” arXiv preprint arXiv:2401.13893, 2024.

[34] G. Wang, D. Zhang, T. Zhang, S. Yang, Q. Sun, and Y. Chen, “Learning domain-invariant model for WiFi-based indoor localization,” IEEE Transactions on Mobile Computing, 2024.

[35] Y. Zhang, A. Y. Ding, J. Ott, M. Yuan, J. Zeng, K. Zhang, and W. Rao, “Transfer learning-based outdoor position recovery with cellular data,” IEEE Transactions on Mobile Computing, vol. 20, no. 5, pp. 2094–2110, 2020.

[36] D. C. Nguyen, M. Ding, P. N. Pathirana, A. Seneviratne, J. Li, and H. V. Poor, “Federated Learning for Internet of Things: A Comprehensive Survey,” IEEE Communications Surveys & Tutorials, vol. 23, no. 3, pp. 1622–1658, 2021.

[37] Y. Etiabi, W. Njima, and E. M. Amhoud, “Femloc: Federated metalearning for adaptive wireless indoor localization tasks in iot networks,” IEEE Internet of Things Journal, 2024.

[38] R. Kumar, R. Popli, V. Khullar, I. Kansal, and A. Sharma, “Confidentiality Preserved Federated Learning for Indoor Localization Using Wi-Fi Fingerprinting,” Buildings, vol. 13, no. 8, p. 2048, 2023.

[39] Y. Etiabi, W. Njima, and E. M. Amhoud, “Federated Learning based Hierarchical 3D Indoor Localization,” in 2023 IEEE Wireless Communications and Networking Conference (WCNC). IEEE, 2023, pp. 1–6.

[40] Y. Ibnatta, M. Khaldoun, and M. Sadik, “The Indoor Localization System Based on Federated Learning and RSS Using UWB-OFDM,” in International Conference on Advanced Intelligent Systems for Sustainable Development. Springer, 2022, pp. 152–167.

[41] X. Cheng, C. Ma, J. Li, H. Song, F. Shu, and J. Wang, “Federated Learning-Based Localization With Heterogeneous Fingerprint Database,” IEEE Wireless Communications Letters, vol. 11, no. 7, pp. 1364–1368, 2022.

[42] J. Guo, I. W.-H. Ho, Y. Hou, and Z. Li, “FedPos: A Federated Transfer Learning Framework for CSI-based Wi-Fi Indoor Positioning,” IEEE Systems Journal, 2023.

[43] T. Wei and S. Liu, “Privacy-Preserving Cooperative Visible Light Positioning for Nonstationary Environment: A Federated Learning Perspective,” arXiv preprint arXiv:2303.06361, 2023.

[44] W. Huang, Y. Yang, M. Chen, C. Liu, C. Feng, and H. V. Poor, “Wireless network optimization for federated learning with model compression in hybrid VLC/RF systems,” Entropy, vol. 23, no. 11, p. 1413, 2021.

[45] L. You, S. Liu, Y. Chang, and C. Yuen, “A Triple-Step Asynchronous Federated Learning Mechanism for Client Activation, Interaction Optimization, and Aggregation Enhancement,” IEEE Internet of Things Journal, pp. 24 199–24 211, 2022.

[46] G. M. Van de Ven, T. Tuytelaars, and A. S. Tolias, “Three Types of Incremental Learning,” Nature Machine Intelligence, vol. 4, no. 12, pp. 1185–1197, 2022.

[47] M. De Lange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. Slabaugh, and T. Tuytelaars, “A Continual Learning Survey: Defying Forgetting in Classification Tasks,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 7, pp. 3366–3385, 2021.

[48] M. Masana, X. Liu, B. Twardowski, M. Menta, A. D. Bagdanov, and J. Van De Weijer, “Class-Incremental Learning: Survey and Performance Evaluation on Image Classification,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 5, pp. 5513–5533, 2022.

[49] H. Wen, R. Clark, S. Wang, X. Lu, B. Du, W. Hu, and N. Trigoni, “Efficient indoor positioning with visual experiences via lifelong learning,”

IEEE Transactions on Mobile Computing, vol. 18, no. 4, pp. 814–829, 2018.

[50] Y. S. Eroglu, F. Erden, and I. Guvenc, “Adaptive Kalman tracking for indoor visible light positioning,” in MILCOM 2019-2019 IEEE Military Communications Conference (MILCOM). IEEE, 2019, pp. 331–336.

[51] V. C. Dat, D. H. Ai, H. Le-Minh, H. C. Cong Phap, N. T. Hung, D. T. Hop, Z. Ghassemlooy et al., “A Novel Adaptive Rss-Based Localization and Auto-Correction Model for Indoor Mimo Vlc Using the Distance Vector.”

[52] L. Zhao, Z. Wang, X. Meng, F. Zhang, and A. Urooj, “A self-adaptive visible light indoor positioning system based on BLS network incorporating self-attention mechanism,” Optics Communications, p. 132153, 2025.

[53] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. March, and V. Lempitsky, “Domain-adversarial training of neural networks,” Journal of machine learning research, vol. 17, no. 59, pp. 1–35, 2016.

[54] M. Zhang, Z. Fan, R. Shibasaki, and X. Song, “Domain adversarial graph convolutional network based on RSSI and crowdsensing for indoor localization,” IEEE Internet of Things Journal, vol. 10, no. 15, pp. 13 662–13 672, 2023.

[55] X. Shi, X. Xuan, M. Wu, and W.-A. Zhang, “MDAALoc: An indoor localization method based on multi-source domain adversarial adaptation,” IEEE Wireless Communications Letters, 2025.

[56] A. Owfi, C. Lin, L. Guo, F. Afghah, J. Ashdown, and K. Turck, “A meta-learning based generalizable indoor localization model using channel state information,” in GLOBECOM 2023-2023 IEEE Global Communications Conference. IEEE, 2023, pp. 4607–4612.

[57] J. Cha, H. Kim, and J. Choi, “Meta-Learning-Based People Counting and Localization Models Employing CSI from Commodity WiFi NICs,” IEEE Internet of Things Journal, 2025.

[58] M. Iman, H. R. Arabnia, and K. Rasheed, “A Review of Deep Transfer Learning and Recent Advancements,” Technologies, vol. 11, no. 2, p. 40, 2023.

[59] A. H. Ali, M. G. Yaseen, M. Aljanabi, and S. A. Abed, “Transfer Learning: A New Promising techniques,” Mesopotamian Journal of Big Data, vol. 2023, pp. 29–30, 2023.

[60] W. Ying, Y. Zhang, J. Huang, and Q. Yang, “Transfer Learning via Learning to Transfer,” in International conference on machine learning. PMLR, 2018, pp. 5085–5094.

[61] S. J. Pan and Q. Yang, “A Survey on Transfer Learning,” IEEE Transactions on knowledge and data engineering, vol. 22, no. 10, pp. 1345–1359, 2009.

[62] T. Suwannaphong, R. McConville, and I. Craddock, “Transfer Learning of RSSI to Improve Indoor Localisation Performance,” arXiv preprint arXiv:2412.09292, 2024.

[63] K. Adamkiewicz, P. Koch, B. Morawska, P. Lipinski, K. Lichy, and´ M. Leplawy, “Improving UWB Indoor Localization Accuracy using Sparse Fingerprinting and Transfer Learning,” in International Conference on Computational Science. Springer, 2021, pp. 291–302.

[64] Y. Zhuang, L. Hua, Q. Wang, Y. Cao, Z. Gao, L. Qi, J. Yang, and J. Thompson, “Visible Light Positioning and Navigation using Noise Measurement and Mitigation,” IEEE Transactions on Vehicular Technology, vol. 68, no. 11, pp. 11 094–11 106, 2019.

[65] S. Zhang, P. Du, H. Yang, R. Zhang, C. Chen, and A. Alphones, “Recent Progress in Visible Light Positioning and Communication Systems.” IEICE Trans. Commun., vol. 106, no. 2, pp. 84–100, 2023.

[66] V. P. Rekkas, L. A. Iliadis, S. P. Sotiroudis, A. D. Boursianis, P. Sarigiannidis, D. Plets, W. Joseph, S. Wan, C. G. Christodoulou, G. K. Karagiannidis et al., “Artificial Intelligence in Visible Light Positioning for Indoor IoT: A Methodological Review,” IEEE Open Journal of the Communications Society, vol. 4, pp. 2838–2869, 2023.

[67] L. Yang, P. Finnerty, and C. Ohta, “Indoor Localization Using Routerto-Router Rssi and Transfer Learning for Dynamic Environments,” Available at SSRN 5133124, 2024.

![](images/fb47ebe11601f5c87ecbf7036ad62edccee52967d44cd63aac793f06ce8825e7.jpg)

[68] A. I. Ahmed, Y. Etiabi, A. W. Azim, and E. M. Amhoud, “A Unified Deep Transfer Learning Model for Accurate IoT Localization in Diverse Environments,” in 2024 IEEE 35th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC). IEEE, 2024, pp. 1–6.

[69] M. Jan, W. Njima, X. Zhang, and A. Artemenko, “Transfer Learning for VLC-Based Indoor Localization: Addressing Environmental Variability,” in 2025 IEEE 101st Vehicular Technology Conference (VTC2025- Spring), 2025, pp. 1–7.

Masood Jan (Graduate Student Member, IEEE) received the engineering degree in electronics engineering from the University of Engineering and Technology (UET), Peshawar, Pakistan, in 2008, and the master’s degree in telecommunications and networks from the Conservatoire National des Arts et Metiers (CNAM), Paris, France, in 2023. He´ is currently pursuing the Ph.D. degree in machine learning with Sorbonne University, Paris. He was ranked first in the EDITE 2023 Doctoral Campaign.

Prior to his doctoral studies, he gained over 12

years of experience in the wireless communications industry, working with multinational companies on AI-driven analytics, automation, and network optimization for LTE/5G systems. His current research interests include machine learning in wireless communication and efficient AI models for 6G/OWC. He is actively engaged in academic dissemination and has delivered numerous talks at conferences and seminars.

![](images/edcca4271a4f4531781b161f60220608abf66f1b92a681b9930627e4f7fa16b9.jpg)

Wafa Njima (Member, IEEE) received the Eng. degree from the Institut National des Sciences Appliquees et de Technologies de Tunis in 2015. She´ received the Ph.D. degree in the field of radiocommunications in 2019, from the Conservatoire National des Arts et Metiers in Paris in collaboration´ with the Ecole Superieure des Communications de´ Tunis. She joined the ETIS laboratory at ENSEA Cergy as a temporary assistant and researcher also as a postdoctoral researcher. She is currently an associate professor at ISEP, Paris. Her publications span

several research areas, and her research interests are related to several topics including signal processing, wireless communications, indoor localization, IoT and machine learning for communications. She is IEEE member and she served as TPC member and reviewer for many leading international conferences and journals.

![](images/56329d5483a8184c5696859f0259c358496af8e42a2d770c2d56f60d6c429128.jpg)

Xun Zhang (Senior Member, IEEE) is a Professor at ISEP (www.isep.fr), Paris, France, and conducts permanent research at the LISV Laboratory within the University of Saclay-Versailles. As a Senior Member of the IEEE, he serves as an Associate Editor for the IEEE Transactions on Broadcasting and is a member of the BTS Awards Committee. He earned his Ph.D. degree in Instrumentation and Microelectronics (IM) Science from the University of Nancy in October 2009. Between 2009 and 2011, Dr. Zhang undertook a Postdoctoral position at the

SCEE Laboratory in CentraleSupelec, France, where he focused on applying´ auto-reconfigurable management methods and control strategies in FPGAs for cognitive radio systems. In 2011, Dr. Zhang joined the LISITE Laboratory at ISEP as an Associate Professor. His research primarily revolves around PHY layer optimization for 5G cellular networks and Visible Light Communication (VLC) systems. He is also pioneering the development of VLC-based indoor high-accuracy positioning and tracking algorithms.

![](images/bb7e4027ffa787b288c2a34c47209fd1f931931d2a0c349562c240a39b37fc51.jpg)

Alexander Artemenko is a research engineer within the Corporate Sector Research of Robert Bosch GmbH in Stuttgart, Germany. Prior to joining BOSCH in 2015, he was a postdoctoral researcher at the Ilmenau University of Technology (TU Ilmenau), Germany from 2013 to 2015. He received his PhD in engineering from the same university in 2013. In 2012, Dr. Artemenko was a post-doctoral researcher and lecturer at UCLA (Univ. of California Los Angeles) in the group of Professor Mario Gerla.