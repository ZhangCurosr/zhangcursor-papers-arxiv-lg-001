# A Transfer Learning-Driven Methodology for Efficient and Sustainable Indoor Localization Using VLC

Masood Jan, Wafa Njima, Xun Zhang, Alexander Artemenko

Abstract—The rapid expansion of the Internet of Things enables seamless collaboration among connected devices, making indoor localization a critical component of industrial automation. However, conventional localization techniques struggle in dynamic environments. These methods depend on extensive, frequent data collection and environment-specific calibration, limiting scalability, interoperability, and effective use of prior research while imposing significant energy penalties that hinder practical deployment. To address these challenges, we propose an energy-efficient transfer learning (TL) based approach for visible light communication based indoor localization. Our method leverages TL to tackle environmental variability, including lighting fluctuations and physical obstacles that typically degrade localization performance, while simultaneously reducing computational overhead and energy consumption compared to conventional approaches. We also introduce a novel model efficiency (ME) metric, designed to integrate localization accuracy, energy efficiency, data efficiency, and transfer gain into a single evaluative measure for comprehensive optimization. Evaluations on a realworld dataset collected from a BOSCH factory demonstrate that our model achieves a 47% improvement in localization accuracy compared to a conventional model that does not utilize TL, and achieves up to 67.5% ME with TL models, compared to only 10.4% in conventional models under high-noise conditions. These results underscore the potential of our approach to deliver a highly efficient and scalable indoor localization system suitable for energy-constrained industrial applications.

Index Terms—Deep Neural Network (DNN), Indoor localization, Model efficiency, Transfer Learning (TL), Visible Light Communication (VLC).

## I. INTRODUCTION

tous connectivity solutions has elevated indoor localization from a supportive tool to a critical infrastructure component in modern industrial ecosystems [2]. As manufacturing and logistics environments grow increasingly complex with autonomous mobile robots [3], digital twin implementations [4], and artificial intelligence (AI)-driven process automation [5], the need for precise, reliable and efficient indoor localization system has become critical. This rising demand is driving innovations in location-based services (LBS), transforming industrial operations through enhanced asset tracking, real-time monitoring, predictive maintenance, and streamlined workflow automation [6].

Despite significant technological advancements, current indoor localization systems face critical performance challenges. These include accuracy, energy consumption, computational complexity, data efficiency, and adaptability to changing environments. These factors have become the primary bottleneck, fundamentally constraining scalability and industrial adoption. For example, energy-intensive algorithms are impractical for battery-powered sensor nodes, and high-speed production lines cannot accommodate lengthy data collection or retraining phases [7]—even if such systems offer high accuracy [8], [9]. Wi-Fi-based systems offer relatively low power consumption [10], but their performance is highly susceptible to environmental interference, resulting in degraded accuracy under dynamic conditions [11]. Bluetooth Low Energy (BLE) improves energy efficiency further by consuming as little as 10.1 µA in sleep mode and extending device runtime significantly compared to Wi-Fi [12]—but suffers from limited range and poor signal stability in changing environments [13], [14]. Both Wi-Fi and BLE require dense deployment of access points or battery-powered beacons, leading to high installation and maintenance costs [15]. Ultra-Wideband (UWB) technology, while delivering high accuracy, faces energy challenges [16], high device costs, and limited adaptability to environmental changes [17]. Radio frequency (RF) systems are further hindered by electromagnetic interference (EMI) from industrial machinery, which can cause signal attenuation exceeding 15 dBm and necessitate increased transmission power to maintain connectivity [18].

These challenges highlight the need for localization solutions that deliver both high accuracy and operational efficiency, while leveraging cost-effective infrastructure. Visible Light

Communication (VLC) has emerged as a promising candidate, utilizing existing Light-Emitting Diode (LED) lighting for both communication and localization. VLC offers centimeterlevel precision and can reduce energy consumption by up to 60% compared to dedicated RF systems [19], [20]. The technology is inherently immune to EMI, eliminating the energy penalties associated with increased transmission power in industrial environments [21], [22], and its short-wavelength, line-of-sight (LOS) nature keeps computational demands low [23]. Nevertheless, VLC systems face their own environmental adaptability challenges, distinct from those of RF-based solutions. While RF systems struggle with unpredictable EMI and multipath fading, VLC contends with more deterministic optical issues: ambient light interference, LOS dependencies, and varying lighting conditions [24]–[26]. However, these optical challenges are generally more predictable and manageable than the chaotic interference often encountered by RF systems in industrial settings [27], [28].

To address these challenges and enhance VLC-based localization, machine learning (ML) models have been increasingly adopted to capture complex signal patterns [29]. Yet, many existing ML models rely on assumptions that limit their realworld applicability: they often require large, labeled datasets for offline training and assume the resulting model will remain effective across different times and deployment contexts [6]. In dynamic industrial environments, where lighting and layouts frequently change, these assumptions break down, leading to rapid degradation in model performance. Several adaptive ML strategies have been proposed to overcome these limitations. Federated Learning (FL) enables decentralized model training across multiple devices, preserving data privacy and capturing diverse conditions [30], but requires frequent updates in highly dynamic settings. Continuous Learning (CL) supports incremental adaptation over time [31], [32], but is vulnerable to catastrophic forgetting and depends on costly, labor-intensive labeled data collection. Transfer Learning (TL) offers rapid adaptation to new conditions without extensive retraining or system downtime [33], [34]; however, in existing literature TL has been widely applied in RF-based localization, and has seen limited adaptation to VLC systems, with domain-specific noise and environmental variability still insufficiently addressed.

VLC stands out as a strong candidate, but realizing its full potential will require innovative ML approaches that address both the unique optical challenges of VLC and the stringent operational constraints of Industry 4.0. This realization motivates our work to design an enhanced TL based efficient model framework for indoor based VLC localization that is simultaneously precise and adaptive. This is an extension of our previous work detailed in [1]. Our key contributions are as follows:

• TL Framework with Dynamic Adaptability and Cross-Environment Generalization: We develop a novel TL framework using on a Deep Neural Network (DNN) that not only adapts effectively to dynamic indoor environments, but also requires significantly less training data—up to 30% of what traditional models demand and assess the practical applicability of the model by evaluating its performance under varying environmental conditions without requiring additional training.

Unified Model Efficiency Metric: To navigate the tradeoffs between accuracy, energy use, data needs, and transfer gains, we introduce a single composite metric called Model Efficiency (ME). This metric guides model selection to meet diverse industrial requirements.

• Model Evaluation based on a real-world VLC dataset: We use a comprehensive dataset collected from an active production line at the BOSCH manufacturing facility in Blaichach, Germany, one of Europe’s leading automotive component manufacturing sites processing over 60,000 sensor readings per hour [35]. This dataset captures operational complexities and signal variations, offering a robust benchmark for evaluating VLC-based indoor localization systems.

The remainder of this paper is organized as follows: Section II presents the related works while Section III depicts the problem formulation and the system model description. Section IV details the proposed algorithm development. In Section V we discuss the experimental setup and data collection process. In Section VI we go through the performance evaluation and analysis before concluding in Section VII.

## II. RELATED WORKS

Over the last few years, several studies have been published to explore various machine learning paradigms, including Federated Learning (FL), Continuous Learning (CL), and Transfer Learning (TL), in order to improve indoor localization systems. These efforts aim to improve positioning accuracy, reduce energy and deployment costs, and support adaptability in dynamic environments. This section reviews key contributions in terms of model adaptability, energy efficiency, and transferability, and identifies remaining gaps in applying these approaches to VLC-based systems.

## A. Federated Learning for Indoor Localization

FL has gained recognition as a promising approach for collaborative model training across distributed clients. It ensures data privacy through decentralized processing and improves energy efficiency by reducing data transmission requirements [36]. Several studies have explored the application of FL to indoor localization tasks, enabling collaborative model training across multiple devices to capture diverse environmental conditions without centralizing data [30], [37], [38]. By keeping data on each device, FL limits exposure to privacy risks. Processing and training occur locally, with only essential updates (e.g., gradients or parameters) sent to a central server, thus avoiding raw data transmission, enhancing privacy, and reducing bandwidth usage and minimizing the energy overhead typically associated with centralized data processing [30]. FL specifically addresses adaptability challenges in indoor localization by leveraging diverse environmental conditions captured across multiple devices without centralizing sensitive data [39]. FeMLoc, a federated meta-learning framework, demonstrates this adaptability by operating in two stages: collaborative meta-training on diverse datasets and rapid adaptation to new environments with minimal fine-tuning. This approach achieves up to 80.95% improvement in localization accuracy compared to conventional neural networks while reducing training iterations by 82.21%. In [37], a federated architecture for Wi-Fi fingerprinting preserves data privacy across edge devices while simultaneously improving localization accuracy and reducing the energy burden on individual devices through distributed processing. Similarly, [38] aggregates and shuffles building data before partitioning it into clients, enabling better model tuning, improved accuracy, and reduced communication overhead compared to traditional machine learning methods. Other works [40]–[43] also integrate FL as a key component in localization system design, demonstrating its potential for energy-efficient collaborative learning. However, despite these advantages, many FL implementations tend to produce models that are closely aligned with the specific characteristics of the participating clients, which can limit their generalizability or performance when deployed on new devices operating under unseen conditions [44]. This inflexibility is problematic in dynamic settings with evolving spatial layouts or fluctuating signals, where maintaining accuracy requires costly retraining that can negate the initial energy savings. Such limitations highlight performance inequities across heterogeneous deployments and underscore critical needs for architectural innovations including transfer learning (TL) mechanisms that can maintain both accuracy and energy efficiency.

## B. Continuous Learning for Indoor Localization

CL presents a solution by allowing models to incrementally update their knowledge as new data becomes available. Unlike conventional retraining, which requires full access to historical data and significant computational effort, CL enables continual adaptation over time, making it suitable for scenarios with evolving signal conditions and changing spatial layouts [31]. Recent surveys, such as [45], provide empirical evaluations and categorization frameworks for CL strategies. A core challenge identified is catastrophic forgetting, the tendency of models to overwrite previously learned information when learning new tasks. Mitigating this phenomenon often requires additional memory, complex update strategies, and increased computation, which may undermine CL’s advantages in resource-constrained settings. Various CL strategies have been proposed to address this issue. For instance, [46] classified task-incremental approaches into three categories: replaybased (which store and replay past samples), regularizationbased (which constrain weight updates), and parameter isolation (which allocate different subsets of model parameters per task). While effective to varying degrees, each approach introduces trade-offs: replay methods increase memory and latency, regularization demands frequent updates, and isolation strategies may lead to model scalability issues. Similarly, [47] examined class-incremental learning techniques in image classification, showing that approaches like rehearsal and bias correction further add to computational and design complexity. In practice, CL effectiveness depends on algorithmic design, access to large, labeled datasets, and sustained computational resources. These requirements are often scarce in industrial environments. Furthermore, the need for frequent model updates in response to new data can lead to inconsistent energy usage, making CL difficult to integrate into energy-constrained systems such as battery-powered nodes. These constraints limit the practicality of CL for industrial indoor localization, where models must be both adaptable and resource-efficient, and where access to labeled data is often limited. In this context, TL offers a more scalable and lightweight alternative, enabling efficient generalization to new environments with minimal fine-tuning, reduced energy consumption, and lower data requirements.

Although FL and CL address important aspects of model adaptability, their practical suitability for VLC-based localization differs from that of TL. FL excels at privacy-preserving, collaborative training across distributed clients, but it relies on frequent parameter exchanges and stable uplink connectivity [48], [49]; in VLC systems the uplink channel is often constrained (e.g., IR or RF fallback, limited uplink bandwidth), and optical channel characteristics (directionality, strong ambient-light variation, non-stationary SNR) can exacerbate client heterogeneity and slow convergence [50]. CL supports incremental updates and can adapt to evolving spatial layouts, yet common mitigation strategies for catastrophic forgetting—replay buffers, parameter isolation, or regularization—introduce memory, latency, or model-complexity overheads that are unattractive for resource-limited VLC nodes and for deployments with limited data [51], [52].

## C. Transfer Learning for Indoor Localization

The limitations of Federated and Continuous Learning particularly in terms of adaptability, data dependency, and energy efficiency, have highlighted the growing relevance of TL as a scalable and practical solution for indoor localization. TL enables the reuse of knowledge from previously trained models, substantially reducing the need for extensive retraining and large labeled datasets in new deployment contexts. This results in lower computational and energy costs, faster deployment, and improved model generalization across environments [53], [54]. According to [55], [56], three critical questions underpin TL research: when to transfer, how to transfer, and what to transfer. These considerations are particularly important in resource-constrained industrial environments where the cost of adaptation must be balanced against expected performance gains. TL techniques have been successfully implemented in RF-based localization systems such as Wi-Fi, BLE, and UWB. For instance, [6] reused convolutional layers in Wi-Fi fingerprinting models to achieve strong accuracy with only 55% of training data. [57] improved BLE-based room classification by leveraging TL across multiple home environments, increasing macro F1 scores in hard-to-predict areas. Similarly, [58] applied TL in UWB systems to reduce the need for dense fingerprinting, achieving sub-30 cm accuracy while cutting data collection efforts by 93%. Despite these successes, TL remains underutilized in VLC-based indoor localization, highlighting an opportunity for further investigation in this domain. Given VLC’s inherent advantages such as high bandwidth, EMI immunity, and energy efficiency via infrastructure reuse, it stands to benefit significantly from TL-enabled generalization across diverse lighting and spatial conditions. Yet, most existing VLC studies focus on deterministic modeling or hardware-based noise mitigation rather than learning-based domain adaptation. For example, [59] employed adaptive least squares and Kalman filtering for noise handling under fixed conditions, while [60] used singular value decomposition (SVD) for high-precision 3D positioning, prioritizing efficiency but lacking adaptability. [61] surveyed ML-based approaches for VLC, noting that while fingerprinting methods are widely studied, domain-transferability and noise robustness remain open challenges. Moreover, recent TL studies in RF-based systems rarely address VLC-specific issues such as ambient light variation, photodiode noise, or LED flicker. [62], for example, used domain-adversarial learning to reduce Wi-Fi localization error but did not consider optical-domain interference. Similarly, [8] improved LoRa/Wi-Fi models via latent space alignment, without addressing VLCspecific degradation modes. Our work bridges this gap through an integrated contribution, spanning a noise-adaptive model architecture, a unified ME metric, and industrial scale vali dation, that advances the development of scalable, adaptable, and energy-efficient TL-driven VLC-based indoor localization system capable of generalizing across diverse environments.

![](images/3c3cd09646b66e1105a87482b5c6bc4a4ebe9565c0d3f5ec9117737491c12f06.jpg)  
Fig. 1: System model illustrating the TL-based framework for model development and evaluation.

## III. SYSTEM MODEL AND PROBLEM FORMULATION

This section is organized in two parts. First, we establish a structured approach for modeling and evaluating three DNN configurations, the Base Model, the Environmental Variation (EV) Model, and the TL Model, under varying environmental conditions. Although these configurations provide a foundation for understanding model adaptation, a comprehensive assessment is necessary to capture real-world deployment tradeoffs. To address this, we introduce the Model Efficiency (ME) metric in III-B. It offers a unified framework to compare models across critical dimensions such as accuracy, energy consumption, data utilization, and the benefits of TL. This framework, illustrated in Fig. 1, supports robust and sustainable comparison of model configurations (e.g., Base, EV, TL)

under varying environmental conditions to guide deployment decisions.

## A. Modeling Environmental Variability and TL Framework

We implement a TL approach to predict users’ coordinates (longitude and latitude) based on collected VLC data. As shown in Fig. 2, the framework transfers knowledge from a pre-trained source model $f _ { s }$ to a target model $f _ { t } ,$ enabling better generalization under new environmental conditions. Our evaluation encompasses three model configurations:

1) Base Model: The Base Model refers to the initial source model $f _ { s } ,$ trained on the raw VLC dataset. It served as the benchmark for our experiments and provided a foundation for comparison with other configurations.

2) EV Model: The EV Model extends the Base Model by introducing a Noise Factor (NF) in the original dataset, to simulate lighting variability. This addition reflects real-world uncertainties in signal strength, addressing challenges caused by fluctuating environmental conditions.

3) TL Model: The TL Model represents the target model f<sub>t</sub>. It initializes with the pre-trained weights of the Base Model (i.e., source model $f _ { s } )$ and is fine-tuned using a subset of data from the new environment. This process facilitates adaptation to environmental changes while leveraging previously learned representations.

To formalize this process and underlying the domain relationships in our framework, we define the source and target domains, loss functions, and fine-tuning methodology.

1) Problem Formulation: Let $D _ { s } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { s } }$ be the source domain with $n _ { s }$ samples, where each input feature vector $x _ { i } ~ \in ~ \mathbb { R } ^ { d }$ corresponds to an output label $y _ { i } ~ \in ~ \mathbb { R } ^ { 2 }$ representing the longitude and latitude. Here, $\mathbb { R } ^ { d }$ is the $d -$ dimensional feature space (signal intensity values), and $\mathbb { R } ^ { 2 }$ is the output label space. The source task $T _ { s }$ is to learn a mapping function $f _ { s } : \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 }$ from the source data. Similarly, let $D _ { t } = \{ ( x _ { j } , y _ { j } ) \} _ { j = 1 } ^ { n _ { t } }$ be the target domain with $n _ { t }$ samples, where each input vector $x _ { j } \in \mathbb { R } ^ { d }$ corresponds to an output label $y _ { j } \in \mathbb { R } ^ { 2 }$ , representing the longitude and latitude. The goal is to develop a function $f _ { t } : \mathbb { R } ^ { d }  \mathbb { R } ^ { 2 }$ that generalizes well to the target domain.

![](images/30288d764f54c2136c1284e4f5f0c150e413e24d7f0bc341f2d899d2cf68c8b6.jpg)  
Fig. 2: Proposed TL-based localization process.

We denote the source model as a DNN parameterized by $\theta _ { s }$ where $\theta _ { s }$ represents the set of learned weights obtained from training on the source domain. The corresponding function is defined as:

$$
f _ { s } ( x _ { i } ; \theta _ { s } ) = \hat { y } _ { i } ,\tag{1}
$$

where $\hat { y } _ { i }$ is the predicted output for input $x _ { i }$ in the source domain.

To adapt the model to the target domain, the target model parameters $\theta _ { t }$ are initialized with the source parameters $\theta _ { s }$ and fine-tuned using the target domain data. The resulting function is:

$$
f _ { t } ( x _ { j } ; \theta _ { t } ) = \hat { y } _ { j } ,\tag{2}
$$

where $\theta _ { t }$ denotes the updated weights optimized for the target domain.

2) Loss Design: To enable the model to balance knowledge transfer between the source and target domains, we employ a combined loss function that integrates both the source domain loss $\mathcal { L } _ { s }$ and the target domain loss $\mathcal { L } _ { t }$ . The total loss is formulated as:

$$
\mathcal { L } _ { t o t a l } = \lambda _ { s } \mathcal { L } _ { s } ( \theta _ { s } ) + \lambda _ { t } \mathcal { L } _ { t } ( \theta _ { t } ) ,\tag{3}
$$

where $\lambda _ { s }$ and $\lambda _ { t }$ are the respective weighting factors, optimized through cross-validation to enhance performance on the target task. This setup allows the model to adapt to the target while retaining useful source features. In other words, the total loss jointly guides the model to preserve prior knowledge while adjusting to new environmental variations. The source and target domain losses (i.e., $\mathcal { L } _ { s }$ and $\boldsymbol { \mathcal { L } } _ { t } )$ are computed independently to capture errors within each respective dataset, using the Mean Squared Error (MSE) criterion:

$$
\mathcal { L } _ { s } = \frac { 1 } { n _ { s } } \sum _ { i \neq 1 } ^ { n _ { s } } \mathcal { L } ( f _ { s } ( x _ { i } ; \theta _ { s } ) , y _ { i } ) ,\tag{4}
$$

$$
\mathcal { L } _ { t } = \frac { 1 } { n _ { t } } \sum _ { j = 1 } ^ { \mathrm { ~ \biggr . ~ } } \mathcal { L } ( f _ { t } ( x _ { j } ; \theta _ { t } ) , y _ { j } ) ,\tag{5}
$$

where $\mathcal { L }$ represents the MSE, which is used to calculate the individual losses for both the source and target models. Here, $f _ { s } ( \cdot ; \theta _ { s } )$ and $f _ { t } ( \cdot ; \theta _ { t } )$ represent the source and target models parameterized by $\theta _ { s }$ and $\theta _ { t }$ . This formulation ensures that both domains contribute proportionally to the learning objective, improving the model’s adaptability during transfer.

3) Fine-Tuning: During fine-tuning, the parameters of the target model $\theta _ { t }$ are initialized based on the parameters of the source model $\theta _ { s }$ . The model then undergoes small adjustments $\Delta \theta$ using the target domain data, which helps it adapt more effectively to the variations in the new environment while retaining valuable knowledge from the source task. This approach is expressed as:

$$
\theta _ { t } = \theta _ { s } + \Delta \theta ,\tag{6}
$$

In simple terms, each parameter of the target model is the sum of the corresponding source parameter and a small update learned from the target data. These updates ensure a gradual adaptation, reducing the risk of performance degradation due to differences between the source and target domains.

While evaluating the Base, EV, and TL models provides useful insights into their individual strengths and adaptability in different environments, it is not sufficient to rely solely on accuracy-based comparisons. A thorough evaluation must also take into account other factors, including computational efficiency, data usage, and the practical advantages of TL. To support more comprehensive and effective model comparison and deployment decision-making, we introduce a unified evaluation metric in the next subsection. This metric offers a systematic approach to balancing and quantifying these key elements, ensuring fair and meaningful comparisons across diverse model configurations.

formance Accuracy (PA), Energy Efficiency (EE), Data Effi ciency (DE), and the Transfer Gain (TG). This unified metric enables a balanced comparison across different models more effectively, ensuring that improvements in one area do not compromise another. The ME metric for the i-th model, denoted as $M E _ { i } ,$ is defined in (7), where $w _ { \mathrm { P A } } , w _ { \mathrm { E E } } , w _ { \mathrm { D E } } , w _ { \mathrm { T G } }$ represent the weights assigned to each component. Here, $i \in \{ 1 , 2 , \ldots , N \}$ indexes the evaluated models, where each index corresponds to the specific Base, EV, and TL model configurations—as used throughout Algorithm 2.

## B. Unified Performance Metric Formulation

We introduced a comprehensive metric called the Model Efficiency (ME) which integrates four key components: Per-

$$
\begin{array} { r } { M E _ { i } = ( w _ { P A } \times P A _ { i } ) + ( w _ { E E } \times E E _ { i } ) + ( w _ { D E } \times D E _ { i } ) + ( w _ { T G } \times T G _ { i } ) , } \end{array}\tag{7}
$$

In our study, we assign equal importance to all four components by setting $w _ { \mathrm { P A } } = w _ { \mathrm { E E } } = w _ { \mathrm { D E } } = w _ { \mathrm { T G } } = 0 . 2 5$ . This uniform weighting ensures that each metric contributes fairly to the overall efficiency evaluation of the model.

1) Performance Accuracy (PA): For each model, PA is calculated as:

$$
\mathrm { P A } _ { i } = \left( 1 - \frac { E _ { \mathrm { m e a n } , i } } { E _ { \mathrm { m a x } , i } } \right) \times 1 0 0 ,\tag{8}
$$

where $E _ { \mathrm { m e a n } , i }$ is the mean localization error measured across all evaluation epochs during validation, and $E _ { \mathrm { m a x } , i }$ is the maximum localization error representing the highest recorded value within the same evaluation process for the $i ^ { \mathrm { { t h } } }$ model. A higher PA value indicates better model accuracy.

2) Energy Efficiency (EE): EE evaluates the computational cost of model training by quantifying energy consumption per unit of useful work accomplished. Let $\bar { T _ { j } ^ { ( i ) } }$ be the duration of the j-th training epoch for the i-th model (in seconds), and let $P _ { \mathrm { G P U } } ( t )$ and $P _ { \mathrm { C P U } } ( t )$ be the instantaneous power consumption (in watts) of the GPU and CPU, respectively, at time t. During training, power consumption is sampled at regular intervals. Let $S _ { j } ^ { ( i ) }$ be the total number of discrete power samples collected during the j-th epoch of the i-th model. The k-th power sample is denoted as:

$$
\begin{array} { r } { P _ { k , j } ^ { ( i ) } = P _ { \mathrm { G P U } , k , j } ^ { ( i ) } + P _ { \mathrm { C P U } , k , j } ^ { ( i ) } , \quad k = 1 , \dots , S _ { j } ^ { ( i ) } , \quad j = 1 , \dots , M } \end{array}\tag{9}
$$

The average power consumption over the j-th epoch for the i-th model is then approximated as:

$$
\overline { { P } } _ { j } ^ { ( i ) } = \frac { 1 } { S _ { j } ^ { ( i ) } } \sum _ { k = 1 } ^ { S _ { j } ^ { ( i ) } } P _ { k , j } ^ { ( i ) } = \frac { 1 } { S _ { j } ^ { ( i ) } } \sum _ { k = 1 } ^ { S _ { j } ^ { ( i ) } } \left( P _ { \mathrm { G P U } , k , j } ^ { ( i ) } + P _ { \mathrm { C P U } , k , j } ^ { ( i ) } \right) ,\tag{10}
$$

The total energy consumed in joules during the j-th epoch of the i-th model is:

$$
E _ { j } ^ { ( i ) } = \overline { { P } } _ { j } ^ { ( i ) } \cdot T _ { j } ^ { ( i ) } ,\tag{11}
$$

The overall energy efficiency of the i-th model, denoted as $E E _ { i }$ , is computed as the average energy consumed per epoch across all M training epochs:

$$
E E _ { i } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } E _ { j } ^ { ( i ) } ,\tag{12}
$$

where M is the total number of training epochs. A higher $E E _ { i }$ value indicates a more energy-efficient training process for model i, as it represents better performance or more computational work achieved per unit of energy consumed.

3) Data Efficiency (DE): DE quantifies the ability of a model to achieve strong performance using a reduced amount of data. It is defined as:

$$
D E _ { i } = \left( 1 - \frac { D _ { i } } { D _ { \operatorname* { m a x } } } \right) \times 1 0 0 ,\tag{13}
$$

where $D _ { i }$ represents the percentage of data used to train model $i ,$ and $D _ { \mathrm { m a x } }$ denotes the maximum percentage of data available for training (in the given setup $D _ { \mathrm { m a x } } = 1 0 0 \% )$ . This metric rewards models that achieve competitive performance using less data. A higher $D E _ { i }$ value indicates better data efficiency, reflecting the model’s ability to generalize from smaller datasets.

4) Transfer Gain (TG): TG quantifies the relative performance improvement achieved through transfer learning (TL). It is defined as:

$$
T G _ { i } = \left( \frac { P A _ { \mathrm { T L } } ^ { ( i ) } - P A _ { \mathrm { E V } } ^ { ( i ) } } { P A _ { \mathrm { B M } } ^ { ( i ) } } \right) \times 1 0 0 ,\tag{14}
$$

where $P A _ { \mathrm { T L } } ^ { ( i ) } , P A _ { \mathrm { E V } } ^ { ( i ) }$ , and $P A _ { \mathrm { B M } } ^ { ( i ) }$ represent the performance accuracy of the TL, EV and base models, respectively, for the $i ^ { \mathrm { { t h } } }$ model. We consistently used the original dataset for the base model, while the EV and TL models were based on three noise-augmented datasets with $\mathrm { N F } \in \{ 2 , 4 , 8 \}$ . A positive $T G _ { i }$ indicates that TL improves the performance of model i relative to both the EV and base models.

By integrating these four components into the unified ME metric, we achieve a comprehensive evaluation framework which ensures that improvements in one area do not compromise another, facilitating a fair and robust comparison across different models.

## IV. ALGORITHM DEVELOPMENT AND IMPLEMENTATION

## A. Pre-TL Model Development

We begin by training a base model on the original dataset, which was collected during the initial measurement campaign, ensuring that it achieves satisfactory performance before proceeding to further steps. The training and validation procedures are formalized below, where the loss functions $\mathcal { L } _ { \mathrm { t r a i n } }$ and ${ \mathcal L } _ { \mathrm { v a l } }$ measure the average localization error between predicted coordinates $\hat { y }$ and true coordinates $y \colon$

$$
\mathcal { L } _ { t r a i n } = \frac { 1 } { | \mathcal { D } _ { t r a i n } | } \sum _ { i = 1 } ^ { | \mathcal { D } _ { t r a i n } | } | \hat { y } _ { i } - y _ { i } | ,\tag{15}
$$

$$
\mathcal { L } _ { v a l } = \frac { 1 } { | \mathcal { D } _ { v a l } | } \sum _ { m = 1 } ^ { | \mathcal { D } _ { v a l } | } | \hat { y } _ { m } - y _ { m } | ,\tag{16}
$$

where $\mathcal { D } _ { \mathrm { t r a i n } }$ and $\mathcal { D } _ { \mathrm { v a l } }$ are the training and validation datasets composed of signal-intensity/localization pairs, and $\hat { y } _ { i } , y _ { i } , \hat { y } _ { m } , y _ { m }$ are the corresponding predicted and true coordinates. After the model shows convergence based on validation error, the final model weights $W _ { j }$ are saved.

After training the base model and collecting benchmark metrics, we simulate environmental drift by injecting Gaussian noise into the training data. This generates multiple datasets representing environmental conditions at time $t + i ,$ where i denotes the progression in time of data collection. The noise intensity is controlled by the standard deviation σ and scaled by a noise factor (NF). Each resulting model, trained on the specific noisy dataset, is referred to as the EV model.

$$
\mathcal { D } _ { t r a i n } ^ { n o i s y } = \mathcal { D } _ { t r a i n } + \mathcal { N } ( 0 , \sigma ^ { 2 } ) ,\tag{17}
$$

where ${ \mathcal { N } } ( 0 , \sigma ^ { 2 } )$ is Gaussian noise with zero mean and variance $\sigma ^ { 2 }$ . The standard deviation σ is scaled as follows:

$$
\sigma = N F \cdot \sigma _ { b a s e } ,\tag{18}
$$

where $\sigma _ { b a s e }$ is the baseline standard deviation. The model is updated using Stochastic Gradient Descent (SGD), which updates the weights iteratively by minimizing the loss function:

$$
W _ { j + 1 } = W _ { j } - \eta \nabla L ( W _ { j } ) ,\tag{19}
$$

where $L ( W _ { j } )$ is the loss function and $\eta$ is the learning rate.

As a result, when each EV model is trained on its corresponding noise-injected dataset generated by applying a specific NF (2, 4, or 8), the localization error increases substantially compared to the base model’s performance on the original dataset. This performance decline underscores the need for employing TL to adapt the model to changing environmental conditions, ensuring the retention of its localization accuracy.

## B. TL Algorithm Development and Implementation

The TL algorithm as shown in Algorithm 1, takes as input a set of received signal strength dataset denoted as $D _ { t r a i n }$ and $D _ { v a l } .$ a noise variance $\bar { \sigma } ^ { 2 }$ , weighting factor α which refers to a scalar hyperparameter used to balance multiple loss terms or regularization components in the objective function, and outputs the trained model weights $W _ { j }$ which refer to the trainable parameters of the neural network model. The server initializes the model with parameters $W _ { j }$ and sets the epoch counter $j .$ The base model $\mathcal { M } _ { b a s e }$ and its weights $W _ { b a s e }$ are loaded, then transferred to the TL model $\mathcal { M } _ { T L }$ and the shared layer weights (layers reused from the base model are set accordingly. To facilitate $\mathrm { T L }$ , the shared layers of $\mathcal { M } _ { T L }$ are frozen, ensuring that these layers are not updated during subsequent training by setting their ‘trainable‘ attribute to ‘False‘. During each training epoch, Gaussian noise with variance $\sigma ^ { 2 }$ is injected into $\mathcal { D } _ { t r a i n }$ (original dataset), to simulate environmental variability, resulting in a noisy dataset $\mathcal { D } _ { t r a i n } ^ { n o i s y }$ . Fine-tuning is then performed using $\mathcal { D } _ { t r a i n } ^ { n o i s \bar { y } }$ , where only the non-frozen (task-specific) layers are updated through backpropagation, leading to the updated weights $W _ { r + 1 }$

As long as the maximum number of epochs M has not been reached and the convergence condition is not met, the algorithm proceeds iteratively. For each epoch, the start and end times are recorded to compute the epoch duration $T _ { \mathrm { j } } .$ . The model is trained on $\mathcal { D } _ { t r a i n } ^ { n o i s y }$ using gradient descent. The loss gradient $\nabla L ( W _ { j } )$ is calculated as the average gradient over the noisy dataset, and the model parameters are updated using the update rule shown in Eq. 19.

Energy consumption is monitored during training by calculating the energy used in each epoch, $E _ { e p o c h } = ( P _ { \mathrm { G P U } } +$ $P _ { \mathrm { C P U } } ) \cdot T _ { \mathrm { e p o c h } }$ , and the cumulative energy consumption $E _ { \mathrm { t o t a l } }$ is updated accordingly. Localization errors $\mathcal { L } _ { t r a i n }$ and $\mathcal { L } _ { v a l }$ are computed for the training and validation datasets, respectively, as the average absolute difference between the predicted and true localization coordinates. Success rates $S R _ { t r a i n }$ and $S R _ { v a l }$ are calculated as the proportion of samples for which the localization error does not exceed a predefined threshold $\delta .$ In this work, we use $\delta = 1$ m, which reflects the minimum acceptable precision level for practical indoor localization scenarios in industrial environments. All performance metrics, including $\mathcal { L } _ { t r a i n } , ~ \mathcal { L } _ { v a l } , ~ S R _ { t r a i n } , ~ S R _ { v a l } , ~ E _ { j }$ , and $T _ { \mathrm { j } } ,$ , are logged after each epoch. The process continues until either convergence is achieved or the maximum number of epochs is reached.

## C. ME Evaluation Algorithm

Next, we evaluate the performance of our models by computing the ME, as shown in Algorithm 2. We design the ME algorithm to take as input the CSV files containing the performance metrics for the Base, EV, and TL models, as well as the precomputed TG values. The Base model data is loaded to compute the PA and EE using the function extract metrics. It initializes the lists for PA, EE, data sizes D, and TG values for the Base model. Since the Base model does not involve transfer learning, its TG value is set to zero by definition.

Algorithm 1: Transfer Learning for Localization   
Input $: \mathcal { D } _ { t r a i n } ;$ /\* User VLC dataset \*/   
Input : $; \mathcal { D } _ { v a l } ; / *$ User VLC Dataset \*/   
Input : $\alpha { ; } \atop { }$ Output weights \*/   
Input : $\sigma ^ { 2 } ; I ^ { * }$ Variance of Gaussian noise \*/   
Output: $W _ { j } ; / { ^ \ast }$ Trained model parameters \*/   
1 ServerInit   
2 Initialize $W _ { j } , j \gets 0 \ ;$   
3 /\* Set initial weights and epoch counter \*/   
4 Load Base Model   
5 Load $\mathcal { M } _ { b a s e }$ and its weights $W _ { b a s e } ;$   
6 Define TL Model   
7 Define $\mathcal { M } _ { T L } ;$   
8 Transfer Weights   
9 Transfer $W _ { b a s e } \mathrm { ~ t o ~ } \mathcal { M } _ { T L }$ and set shared layer weights;   
10 Freeze Shared Layers   
11 foreach layer in $\mathcal { M } _ { T L }$ do   
12 if layer is in shared layers then   
13 Freeze the layer by setting ‘layer.trainable = False; /\* Preserve   
shared knowledge \*/   
14 end   
15 end   
16 ;   
whil $; j < M$ and not converged do   
18 Simulate EV Model   
19 $\mathcal { D } _ { t r a i n } ^ { n o i s y } = \mathcal { D } _ { t r a i n } + \mathcal { N } ( 0 , \sigma ^ { 2 } ) ; / \ast$ Simulate environmental   
variations \*/   
20 Track Epoch Time   
21 $T _ { s t a r t } \gets$ start time;   
22 Train Model   
23 Fine-tune the model with $\mathcal { D } _ { t r a i n } ^ { n o i s y } .$ updating only the non-frozen   
layers;;   
24 Compute gradient:;   
25 $\begin{array} { r } { \nabla L ( W _ { j } ) = \frac { 1 } { | \mathcal { D } _ { t r a i n } ^ { n o i s y } | } \sum _ { i = 1 } ^ { | \mathcal { D } _ { t r a i n } ^ { n o 1 s y } | } \nabla f ( x _ { i } , y _ { i } , W _ { j } ) ; } \end{array}$   
26 Update weights:;   
27 $\hat { W _ { j + 1 } } = \bar { W _ { j } } - \eta \nabla L ( W _ { j } )$   
28 /\* Gradient descent update on non-frozen layers \*/   
29 Track Epoch Time End   
30 $T _ { e n d } \gets$ end time;   
31 $T _ { j } = T _ { e n d } - T _ { s t a r t } ;$   
32 Monitor Energy Consumed   
33 $E _ { j } = ( \overleftarrow { P _ { G P U } } + P _ { C P U } ) \cdot T _ { j } ;$   
34 $E _ { t o t a l } ^ { ' } + = E _ { j } ; { / }$ Update cumulative energy \*/;   
35 Compute Localization Error   
36 $\begin{array} { r } { \dot { \mathcal { L } } _ { t r a i n } = \frac { 1 } { | \mathcal { D } _ { t r a i n } | } \sum | \hat { y } _ { i } - y _ { i } | ; / \ast } \end{array}$ Avg. localization error \*/   
$\begin{array} { r } { \mathcal { L } _ { v a l } = \frac { \dot { \mathrm { ~ \scriptsize ~ 1 ~ } } ^ { \star } } { | \mathcal { D } _ { v a l } | } \sum | \hat { y } _ { i } - y _ { i } | ; } \end{array}$   
37 Compute Success Rate   
38 $\begin{array} { r } { S R _ { t r a i n } = \frac { | \mathcal { D } _ { t r a i n } | _ { \mathcal { L } _ { t r a i n } } < \delta } { | \mathcal { D } _ { t r a i n } | } , S R _ { v a l } = \frac { | \mathcal { D } _ { v a l } | _ { \mathcal { L } _ { v a l } < \delta } } { | \mathcal { D } _ { v a l } | } ; } \end{array}$   
39 Save Metrics   
40 Save metrics: $\mathcal { L } _ { t r a i n } , \mathcal { L } _ { v a l } , S R { _ { t r a i n } } , S R _ { v a l } , E _ { j } , T _ { j } ;$   
41 $j  j + 1 ;$   
42 end   
43 Save model: transf er learning model.save.h5;

The algorithm then processes the EV and TL models. For each model, it extracts the PA and EE metrics. The data size D is determined based on the model name (which represents different percentages of the dataset). If the model is a TL model, the corresponding TG value is retrieved from the TG table based on the data size and NF value. For the EV model, since it is trained directly on the noisy target data without any transferred knowledge, its TG value is not defined and is set to zero for consistency in the ME computation.

After collecting all PA, EE, and TG values, the DE is computed using (13), and finally, the ME is calculated using (7). The ME values are returned for further analysis, enabling comparative performance assessment of the different models.

Algorithm 2: Model Efficiency Evaluation   
Input : CSV files for Base, EV, TL models and TG   
Output: ME values for each model   
1 Initialize   
2 Load Base model data;   
3 Compute $P A _ { \mathrm { b a s e } } , E E _ { \mathrm { b a s e } }$ using extract\_metrics (model data);   
4 $P A \dot { \overline { { \mathcal { \lfloor } v a l u e s } } }  [ P A _ { \mathrm { b a s c } } ] ;$   
5 EE values $\mathrm { ~ : ~ }  \mathrm { ~ : ~ } E E _ { \mathrm { b a s c } } \} ;$   
6 D values ← [100]; /\* % data for Base model $^ { * }$   
7 TG values ← [0]; /\* TG = 0 for Base model \*/;   
8 Process EV, TL Models   
9 foreach model (model name, model data) in EV, TL models list do   
10 (PA, EE) ← extract\_metrics; Determine data size D from   
model\_name (e.g., 30%, 50%, 70%, or 100%);   
11 if model\_name contains ‘TL Model’ then   
12 if model\_name contains ‘NF2’ then   
13 TG ← Retrieve from TG table for $( D , N F = 2 ) ;$   
14 end   
15 else if model\_name contains ‘NF4’ then   
16 TG ← Retrieve from TG table for $( D , N F = 4 ) ;$   
17 end   
18 else if model\_name contains ‘NF8’ then   
19 TG ← Retrieve from TG table for $( D , N F = 8 ) \quad$   
20 end   
21 else   
22 1 $T G \gets 0 ; / { ^ { * } } \mathrm { E V }$ models get $\mathrm { T G } = 0 ~ ^ { * } /$   
23 end   
24 Append PA, EE, D, and TG to PA values, EE values,   
D values, and TG values;   
25 end   
26 Compute Data Efficiency (DE)   
27 Compute DE using Eq. (13);   
28 Compute Model Efficiency (ME)   
29 Compute ME using Eq. (7);   
Output:   
30 Return the set of ME values for further analysis;

## V. EXPERIMENTAL SETUP AND DATA COLLECTION

To develop and evaluate the proposed VLC-based localization framework, a comprehensive experimental setup was implemented within a real-world industrial environment. This section details the design and specifications of the VLC transceiver system, the measurement setup deployed in the BOSCH factory, and the methodology followed for data collection. The objective is to capture signal strength data across various transmitter-receiver configurations under realistic factory conditions to support robust model training and validation.

## A. VLC Transceiver System

1) Transceiver Architecture: Fig. 3 represents the architecture of the implemented VLC transceiver. It comprises three primary components: the optical transmitter (TX), the optical receiver (RX), and the power supply unit. On the TX side, an input signal is provided via the ’Tx’ interface. This signal undergoes pre-equalization to mitigate distortions introduced by the light source, followed by amplification to ensure compatibility with the electrical characteristics of the optical emitter. The modulated electrical signal is then converted into an optical signal by the light source. An optional optical module, such as a lens, may be employed to adjust the beam directivity depending on the measurement requirements.

After free space propagation, the optical signal sent by the TX of one VLC transceiver is collected by the RX of another VLC transceiver. This current signal is converted into a voltage signal and amplified by a transimpedance amplifier (TIA), then post-equalized to cope with the RX characteristic, and finally amplified again and filtered before being sent through the $\mathbf { \Delta } ^ { 6 } \mathbf { R } \mathbf { x } ^ { \prime }$ interface.

![](images/6723a8cab4892aa2d8e9d61a3f20e60ff55d49469e20f635d5070c4728ab680c.jpg)  
Fig. 3: Architecture of the implemented VLC transceiver.

2) Transceiver Specifications: TABLE I summarizes the key specifications of the transmitter (TX), receiver (RX), and power supply units comprising the deployed VLC transceiver. Each unit is powered via a single USB-C female port requiring a 12 V, 1.2 A direct current (DC) source. The input voltage is regulated by a converter board that supplies a stable 5V DC to the optical front-end and signal processing circuits.

The transmitter employs a single LED driven by a current signal of 200 mA ± 100 mA, resulting in an optical output power ranging linearly from 250 mW to 750 mW. The emitted light exhibits a full-angle half-power beamwidth of 60°, with a central wavelength of 450 nm and an operational spectral range of 450–650 nm. The LED’s spectral width of 40 nm ensures operation entirely within the visible spectrum. Signal input to the TX chain is facilitated via an SubMiniature version A (SMA) connector to ensure robust coupling.

The receiver integrates multiple photodetectors connected in parallel, yielding an effective sensitive area (A Rx) of 600 mm². It is responsive across a broad spectral width of 400 nm, spanning the full visible range from 380 nm to 780 nm. The RX also exhibits a 60° full-angle half-sensitivity field of view, enhancing optical power collection from various angles. Like the transmitter, the RX chain employs an SMA interface for signal output.

## B. VLC Measurement System

The VLC measurement setup, shown in Fig. 4, was deployed at the BOSCH factory. It consists of two VLC transceivers, two converter boards, two DC power supplies, and a vector network analyzer (VNA). Each converter board combines the RF signal from the VNA with DC power, enabling simultaneous signal transmission and device powering through a single connection. During measurements, the VNA generates a frequency signal that modulates the optical carrier. The corresponding optical signal is received and converted by the RX unit, and the resulting electrical signal is captured and recorded by the VNA. This configuration enables highprecision measurement of the received signal at various RX positions, providing the basis for training and evaluating the VLC-based localization model.

TABLE I: VLC Transceiver Specifications.
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Power Supply</td><td rowspan=1 colspan=1>Supply voltageSupply currentPower supply interface</td><td rowspan=1 colspan=1>12 V1.2 AUSB-C female</td></tr><tr><td rowspan=1 colspan=1>TX (Transmitter)</td><td rowspan=1 colspan=1>Driving current signal rangeTransmitted optical power rangeFull angle at half powerWavelength of operationSpectral widthTX chain interface</td><td rowspan=1 colspan=1>200 mA ± 100 mA500 mW ± 250 mW60°450 nm (450–650 nm)40 nmSMA</td></tr><tr><td rowspan=1 colspan=1>RX (Receiver)</td><td rowspan=1 colspan=1>Effective sensitive areaPhotosensitivity widthPhotosensitivity at 450 nmFull angle at half sensitivityRX chain interface</td><td rowspan=1 colspan=1>600 mm²400 nm (380–780 nm)0.3 A/W60°SMA</td></tr></table>

![](images/ad83511aad631d5f4609e0b5840e7b8b60b9f6a9d938344d9ef25d34646811f0.jpg)  
Fig. 4: VLC measurement system.

## C. Data Collection

We collected the data at the industrial hall of the BOSCH factory in Blaichach, Germany. The complete hall has a dimension of 74 m × 171 m × 11 m, however the measurements were conducted in a portion of 46 m × 42 m × 11 m, as indicated in Fig. 5. This segment contains one production line and a large storage area. The limiting walls to the exterior are made from concrete, glass, and metal plates. In addition, concrete columns are mounted in a regular basis along the interior side. Multiple metallic structures such as tubes for ventilation and girders are found close to the ceiling.

Fig. 6a illustrates the overview of the transceiver setup and the grid layout used in the experiment. Taking into account the real-world layout and operational structure of the production line, the heights of the transmitter and receiver front-ends were fixed at 2.9 m and 1.1 m, respectively. For consistent analysis and simplified statistical processing, the receiving area was divided into uniform grids of 0.2 m × 0.2 m. The intersections of these grids served as the designated measurement points where data was collected. Fig. 6b displays the actual positions of the transmitters and the measured area within the production line, as marked on the factory-provided floor plan. The orange star represents the absolute reference origin used in the data collection process. Based on the practical constraints of the production environment, 10 transmitter measurement areas were selected, with an average of approximately 160 receiver positions sampled at each location. At each of these positions, signal measurements were recorded continuously over a 5-second period to capture temporal variations caused by environmental noise and system fluctuations. This sampling duration strikes a balance between capturing dynamic effects and ensuring measurement efficiency. The average of the collected samples was then computed to yield a single Received Signal Strength Indicator (RSSI) value per position, thereby reducing the influence of transient anomalies and ensuring a stable representation of the signal quality for localization purposes. In this work we treat the initially measured dataset as the ground truth for the Base Model. The measurements were collected on-site in the operational production area, capturing realistic ambient lighting, machinery interference, and other environment-specific noise sources. Thus, the Base dataset is not a synthetic or noise-free reference but an empirical baseline that reflects actual factory conditions. This dataset serves as the ground-truth reference against which the EV and TL models were compared. The 3D coordinates and located areas of all transmitter positions are detailed in TABLE II.

![](images/19622f8ebeecce22cc08a1c4636754d989855328ea2a0c4b42841c8e2ead4422.jpg)

![](images/6e3900b0fdb27737d0fbc6db565a1bd5454450820c9992b2751db671cf58bfc1.jpg)  
Fig. 5: Schematic of the industrial hall at BOSCH in Blaichach.  
Fig. 6: Illustrations of (a) Tx/Rx setup, and (b) Floor plan layout.

## VI. PERFORMANCE EVALUATION AND ANALYSIS

## A. TL Model Evaluation under different lighting conditions

We evaluate the performance of the Base, EV, and the TL model. As previously mentioned in III-A and III-B, the base model is always trained using the original collected dataset while the EV model is trained on noise-injected datasets, where $\mathrm { N F } \in \{ 2 , 4 , 8 \}$ to simulate different lighting conditions. The TL model, on the other hand, is designed to handle such noisy datasets more effectively by leveraging knowledge from a previously trained base model. Its primary goal is to enhance performance metrics under varying environmental conditions. The DNN model shared across all three configurations, consists of five hidden layers. The ADAM optimizer is applied with a learning rate of 0.0001, and the Mean Square Error (MSE) loss function is used. We note that 80% of the data samples were allocated for training, while the remaining 20% were held out exclusively for validation across all experimental conditions (Base, EV, and TL models). For TL experiments, the model was first pre-trained on the source domain (Base dataset), then fine-tuned using varying portions (30%, 50%, 70%, or 100%) of the training subset from the target domain (with NF = 2, 4, or 8), and finally evaluated on the corresponding held-out validation subset from the target domain. This separation ensures that all reported performance metrics represent the model’s true generalization capability on unseen samples. Training is conducted for 600 epochs with a batch size of 64. To account for variability due to random initialization and stochastic optimization, each experiment was repeated multiple times with different random seeds. Reported performance metrics (accuracy, EE, and ME) represent the average values across these runs with standard deviations reported to quantify uncertainty. A summary of all the parameters used in these experiments is provided in TABLE III.

TABLE II: 3D Coordinates and Located Areas of Transmitters.
<table><tr><td rowspan=1 colspan=1>Transmitter ID</td><td rowspan=1 colspan=1>Position (x, y, z)</td><td rowspan=1 colspan=1>Located Area</td></tr><tr><td rowspan=1 colspan=1>TX1</td><td rowspan=1 colspan=1>(2.6, 0.8, 2.9)</td><td rowspan=1 colspan=1>Near control station and transpar-ent windows</td></tr><tr><td rowspan=1 colspan=1>TX2</td><td rowspan=1 colspan=1>(1.7, -0.2, 2.9)</td><td rowspan=1 colspan=1>Near control station and transpar-ent windows</td></tr><tr><td rowspan=1 colspan=1>TX3</td><td rowspan=1 colspan=1>(3.7, 1.8, 2.9)</td><td rowspan=1 colspan=1>Near control station and transpar-ent windows</td></tr><tr><td rowspan=1 colspan=1>TX4</td><td rowspan=1 colspan=1>(17.1, 0.2, 2.9)</td><td rowspan=1 colspan=1>Material storage zone</td></tr><tr><td rowspan=1 colspan=1>TX5</td><td rowspan=1 colspan=1>(12.9, 0.8, 2.9)</td><td rowspan=1 colspan=1>Rear of production machinery (be-side cement pillars)</td></tr><tr><td rowspan=1 colspan=1>TX6</td><td rowspan=1 colspan=1>(17.1, 0.8, 2.9)</td><td rowspan=1 colspan=1>Rear of production machinery (ad-jacent to storage)</td></tr><tr><td rowspan=1 colspan=1>TX7</td><td rowspan=1 colspan=1>(14.2, 8.5, 2.9)</td><td rowspan=1 colspan=1>Beneath conveyor belts near lock-ers</td></tr><tr><td rowspan=1 colspan=1>TX8</td><td rowspan=1 colspan=1>(23.2, 4.6, 2.9)</td><td rowspan=1 colspan=1>Main production line</td></tr><tr><td rowspan=1 colspan=1>TX9</td><td rowspan=1 colspan=1>(15.2, 4.6, 2.9)</td><td rowspan=1 colspan=1>Main production line</td></tr><tr><td rowspan=1 colspan=1>TX10</td><td rowspan=1 colspan=1>(13.5, 3.6, 2.9)</td><td rowspan=1 colspan=1>Main production line</td></tr></table>

TABLE III: TL Simulation Settings.
<table><tr><td>Parameter</td><td>Description</td><td>Value</td></tr><tr><td>d</td><td>Input dimension</td><td>10</td></tr><tr><td>Optimizer</td><td>Model optimizer</td><td>ADAM</td></tr><tr><td>η</td><td>learning rate</td><td>0.0001</td></tr><tr><td>B</td><td>Batch size</td><td>64</td></tr><tr><td>E</td><td>Number of epochs per iteration</td><td>600</td></tr><tr><td>L</td><td>Number of hidden layers</td><td>5</td></tr><tr><td>h</td><td>Hidden layer size</td><td>512</td></tr></table>

Fig. 7a illustrates the localization error observed during the validation phase. According to the results presented in TABLE IV, the base model attains the best performance compared to the EV model, exhibiting a localization error of 41.21 cm. In contrast, the EV model showed considerable vulnerability to noise: its error escalates to 70.17 cm at $N F = 2 ,$ and further to 80.41 cm at $N F = 8 ,$ corresponding to a 95.14% degradation relative to the base model. On the other hand, the TL model effectively resists noise interference, maintaining low errors of 41.37 cm and 42.59 cm at $N F = 2$ and $N F = 8$ respectively, reflecting only minor increases of 0.39% and 3.35% over the base model. The same trend is observed in the success rate shown in Fig. 7b, reinforcing the superior noise resilience of the TL model compared to the EV approach.

While our localization error may not surpass some stateof-the-art methods, our principal aim is to introduce a TL framework that optimally balances accuracy, noise immunity, and minimal data requirements in an energy-efficient manner. Extending this TL strategy to more complex architectures beyond DNNs could further decrease the error at the expense of higher computational and power costs.

To further assess the TL model’s robustness, we compared its performance when trained with different base models, as presented in Fig. 7c. TABLE V shows that using a base model trained with $N F = 2$ yields a slight error improvement of 0.29% over the TL model trained with a $N F = 0 .$ . Similarly, for $N F = 4 ,$ , the error is reduced by 0.67% compared to the baseline TL model. These findings suggest that while the choice of a base model trained with noise does slightly improve localization accuracy, the effect is minimal, indicating low sensitivity to the base model selection across different noise levels. Overall, although localization precision degrades as noise intensifies, the TL models mitigate this effect and enhance performance stability.

## B. TL model evaluation with limited data

We evaluate the performance of the TL model under varying data sizes and noise levels. As reported in TABLE VI, reducing the dataset to 30% of its original size leads to a notable increase in localization error, reaching 86.78 cm. However, with the application of TL, the error significantly drops to around 42.96 cm, an improvement of over 45%. This notable reduction demonstrates the TL model’s strong capability to maintain performance even when training data is limited. As illustrated in Fig. 8, the localization error using just 30% of the data, after applying TL, closely aligns with the error obtained when using the full dataset.

## C. Key Metric Evaluation: TG, Accuracy and EE

Fig. 9a represents the TG for varying data sizes and noise levels. At 100% data and NF = 2, the TG is comparatively low (56.56%), due to a smaller accuracy gap between TL and EV models. As noise increases (NF = 4, 8), the EV model’s accuracy degrades more sharply, raising TG to 96.50% and 97.74%, respectively. With reduced data (70%, 50%, 30%), the EV model performance drops further, while the TL model remains robust, widening the accuracy gap and elevating the TG. For example, at 70% data and NF=8, TG peaks at 99.88%, staying above 96.96% even at 50% and 30%. Even for the lower noise setting of ${ \mathrm { N F } } { = } 2 .$ , the TG steadily increases from

![](images/e6f93460cabae4c7fd0e56358948061094b77cd6140444722c7fad426c7e0150.jpg)  
(a)

![](images/740192a50803fb45e4637d5d360c86b295e239487fdfc4d979364f8497991438.jpg)  
(b)

![](images/51fabf341b07e32a7a4d63472499592565366dcb538b4c7087c6df35230fa5b8.jpg)  
(c)

Fig. 7: TL performance evaluation: (a) Localization Error (b) Success Rate (c) TL of Base Model with different NFs.  
TABLE IV: Comparison of final metrics across models with different NFs, using the base model at $\mathrm { N F } = 0 .$
<table><tr><td rowspan=1 colspan=1>Final Metric Value</td><td rowspan=1 colspan=1>NF = 0</td><td rowspan=1 colspan=2>NF = 2</td><td rowspan=1 colspan=2>NF = 4</td><td rowspan=1 colspan=2>NF = 8</td></tr><tr><td rowspan=1 colspan=1>Training Model</td><td rowspan=1 colspan=1>Base Model</td><td rowspan=1 colspan=1>EV Model</td><td rowspan=1 colspan=1>TL Model</td><td rowspan=1 colspan=1>EV Model</td><td rowspan=1 colspan=1>TL Model</td><td rowspan=1 colspan=1>EV Model</td><td rowspan=1 colspan=1>TL Model</td></tr><tr><td rowspan=1 colspan=1>Training Localization Error (cm)</td><td rowspan=1 colspan=1>41.13</td><td rowspan=1 colspan=1>70.38</td><td rowspan=1 colspan=1>41.37</td><td rowspan=1 colspan=1>78.99</td><td rowspan=1 colspan=1>41.74</td><td rowspan=1 colspan=1>79.91</td><td rowspan=1 colspan=1>42.42</td></tr><tr><td rowspan=1 colspan=1>Validation Localization Error (cm)</td><td rowspan=1 colspan=1>41.21</td><td rowspan=1 colspan=1>70.17</td><td rowspan=1 colspan=1>41.37</td><td rowspan=1 colspan=1>79.37</td><td rowspan=1 colspan=1>41.71</td><td rowspan=1 colspan=1>80.41</td><td rowspan=1 colspan=1>42.59</td></tr><tr><td rowspan=1 colspan=1>Training Success Rate (%)</td><td rowspan=1 colspan=1>93.07</td><td rowspan=1 colspan=1>57.76</td><td rowspan=1 colspan=1>93.07</td><td rowspan=1 colspan=1>47.25</td><td rowspan=1 colspan=1>92.32</td><td rowspan=1 colspan=1>47.75</td><td rowspan=1 colspan=1>92.49</td></tr><tr><td rowspan=1 colspan=1>Validation Success Rate (%)</td><td rowspan=1 colspan=1>91.64</td><td rowspan=1 colspan=1>59.20</td><td rowspan=1 colspan=1>91.97</td><td rowspan=1 colspan=1>47.16</td><td rowspan=1 colspan=1>90.97</td><td rowspan=1 colspan=1>47.83</td><td rowspan=1 colspan=1>91.30</td></tr><tr><td rowspan=1 colspan=1>Cumulative Energy Consumed (J)</td><td rowspan=1 colspan=1>852.79</td><td rowspan=1 colspan=1>1571.35</td><td rowspan=1 colspan=1>1064.91</td><td rowspan=1 colspan=1>1923.38</td><td rowspan=1 colspan=1>1082.56</td><td rowspan=1 colspan=1>976.70</td><td rowspan=1 colspan=1>867.22</td></tr><tr><td rowspan=1 colspan=1>Cumulative Training Time (s)</td><td rowspan=1 colspan=1>93.04</td><td rowspan=1 colspan=1>196.03</td><td rowspan=1 colspan=1>124.00</td><td rowspan=1 colspan=1>214.88</td><td rowspan=1 colspan=1>126.17</td><td rowspan=1 colspan=1>123.08</td><td rowspan=1 colspan=1>92.06</td></tr><tr><td rowspan=1 colspan=1>Training Time per Epoch (s)</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.19</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.17</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.15</td></tr></table>

TABLE V: Localization error comparison of TL models at different NFs.
<table><tr><td rowspan=1 colspan=1>NF</td><td rowspan=1 colspan=1>EV Model (cm)</td><td rowspan=1 colspan=1>TL as Base Model (cm)</td><td rowspan=1 colspan=1>TL as (EV Model NF =2) (cm)</td><td rowspan=1 colspan=1>TL as (EV Model NF =4) (cm)</td></tr><tr><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>41.19</td><td rowspan=1 colspan=1>41.17</td><td rowspan=1 colspan=1>41.14</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>70.17</td><td rowspan=1 colspan=1>41.37</td><td rowspan=1 colspan=1>41.25</td><td rowspan=1 colspan=1>41.20</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>79.37</td><td rowspan=1 colspan=1>41.71</td><td rowspan=1 colspan=1>41.43</td><td rowspan=1 colspan=1>41.65</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>80.41</td><td rowspan=1 colspan=1>42.59</td><td rowspan=1 colspan=1>41.26</td><td rowspan=1 colspan=1>41.73</td></tr></table>

TABLE VI: Error values before and after TL at different NFs, comparing performance across various data sizes.
<table><tr><td rowspan=1 colspan=1>Data Size</td><td rowspan=1 colspan=3>Error Before TL (cm)</td><td rowspan=1 colspan=3>Error After TL (cm)</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>NF = 2</td><td rowspan=1 colspan=1> $\mathbf { N F } = 4$ </td><td rowspan=1 colspan=1> $\mathbf { N F } = \mathbf { 8 }$ </td><td rowspan=1 colspan=1> $\mathbf { N F } = 2$ </td><td rowspan=1 colspan=1>NF = 4</td><td rowspan=1 colspan=1>NF = 8</td></tr><tr><td rowspan=1 colspan=1>30%</td><td rowspan=1 colspan=1>77.08</td><td rowspan=1 colspan=1>83.08</td><td rowspan=1 colspan=1>86.78</td><td rowspan=1 colspan=1>41.89</td><td rowspan=1 colspan=1>42.05</td><td rowspan=1 colspan=1>42.96</td></tr><tr><td rowspan=1 colspan=1>50%</td><td rowspan=1 colspan=1>76.49</td><td rowspan=1 colspan=1>81.89</td><td rowspan=1 colspan=1>84.75</td><td rowspan=1 colspan=1>41.71</td><td rowspan=1 colspan=1>41.65</td><td rowspan=1 colspan=1>41.89</td></tr><tr><td rowspan=1 colspan=1>70%</td><td rowspan=1 colspan=1>74.66</td><td rowspan=1 colspan=1>80.17</td><td rowspan=1 colspan=1>81.33</td><td rowspan=1 colspan=1>41.63</td><td rowspan=1 colspan=1>41.45</td><td rowspan=1 colspan=1>41.59</td></tr><tr><td rowspan=1 colspan=1>100%</td><td rowspan=1 colspan=1>70.17</td><td rowspan=1 colspan=1>79.01</td><td rowspan=1 colspan=1>80.76</td><td rowspan=1 colspan=1>41.45</td><td rowspan=1 colspan=1>41.32</td><td rowspan=1 colspan=1>41.33</td></tr></table>

56.56% at 100% data to 82.64% at 30% data. This confirms that TL is highly effective in scenarios where environmental variations and limited data availability are concerns.

Fig. 9b highlights the PA of different models. The base model and TL models with varying noise levels i.e., $N F \in$ {2, 4, 8} achieve high accuracy, nearing 100% and maintaining stability throughout. However, the EV models exhibit significantly lower accuracy, with ${ \mathrm { N F } } = 2$ reaching around 25% and with NF = 4 and $\mathrm { N F } = 8$ remaining below 10%. This suggests that while the base and TL models effectively generalize, the EV models struggle to adapt under environmental variations, particularly as noise factor increases. Overall, the results indicate that TL models provide a reliable solution for maintaining high accuracy even in the presence of environmental variations.

![](images/662efb8cf5f452081a0e32c1bb14422f1b48c998124cd5cd7db01191a54b2d62.jpg)  
Fig. 8: TL model evaluation with limited data.

Fig. 9c highlights energy efficiency trends. The base model leads with 80% efficiency, followed closely by TL models, particularly TL (NF=2). In contrast, EV models are less efficient, averaging around 30% for EV (NF=4) and around 40% for EV (NF=8). This reflects the EV models’ instability under noise, leading to frequent adjustments, slower convergence, and higher power usage. TL models, however, maintain both accuracy and energy efficiency, demonstrating their robustness in noisy, variable environments. All energy consumption values reported in this study were experimentally obtained using hardware-level monitoring during model training. GPU power usage was read directly from the NVIDIA driver through the pynvml library, and CPU utilization was recorded via psutil. For each epoch, instantaneous power values were sampled at regular intervals, averaged, and multiplied by the corresponding training duration to obtain the total energy consumed (in joules). These measurements were repeated across epochs and averaged to ensure consistency. The results therefore represent experimentally measured energy consumption values collected under identical hardware and runtime conditions.

![](images/ab62ef7848ee2c631e6260958e2d63620228569aae1a97050057372a09f66c74.jpg)  
(a)

![](images/795d062ae8f702f4c363cca98508c94c91fe2ae589eb44ca0579161b0b23cedc.jpg)  
(b)

![](images/101eac582ed55c8b86511ed55f03dfa5c3cbc27b21fa0abfdce057dddef1c763.jpg)  
(c)

Fig. 9: Key Metrics Evaluation: (a) Transfer Gain for varying data sizes and NFs (b) Performance Accuracy of Base, EV, and TL models (c) Energy Efficiency trends derived from experimental measurements.  
![](images/0b45d14590589c84a32a1fc45989b04bc7f3fb2e4f7cca89e5e301f26bbc45fd.jpg)  
Fig. 10: Model efficiency across different models and environmental conditions under varying data proportions and noise factors.

It should be noted that the TL model exhibited very low variability (accuracy: $9 1 . 3 0 ~ \pm ~ 0 . 5 1 \%$ , EE: $7 2 . 1 0 ~ \pm ~ 3 . 5 6 )$ confirming excellent stability of the training process and reproducibility of the reported results. The EV model showed higher variability (accuracy: $4 7 . 8 3 \pm 6 . 7 7 \%$ , EE: $3 5 . 7 3 \pm 6 . 4 8 )$ , which is expected due to the noise injection during training.

## D. Model Efficiency Evaluation

Fig. 10 represents the ME trends across different models and environmental settings, with the ME computed through Eq. 7 for each scenario. The base model achieves 53.7% ME and serves as a reference for comparison. Note that we keep the base model as a benchmark to show that, through TL the models trained on limited and noisy data can achieve higher ME than the original base model.

The EV models trained under varying environmental conditions show a sharp drop in efficiency as NF increases, for example; EV Model 30% (NF2) achieves 31.6%, while EV

Model 100% (NF8) falls to just 10.4%, highlighting their limited adaptability under environmental variations.

However, TL models consistently outperform their EV counterparts, demonstrating robustness. TL Model 30% (NF2) reaches 61.5% efficiency, which is higher than the base model, showcasing the benefits of leveraging pre-trained knowledge. As NF increases, TL models continue to maintain strong efficiency levels, with TL Model 100% (NF8) still achieving 53.0% efficiency, significantly outperforming the corresponding EV models.

These results emphasize ME as a reliable comparative metric in dynamic environments, as it captures multiple aspects of performance simultaneously. The results underscore the effectiveness of TL in sustaining efficiency across different environmental conditions, without the need for retraining from scratch.

## E. Generalization capacity of the best ME model

To analyze the generalization ability of our best-performing TL model, we follow a two-step evaluation process. First, we define an acceptance threshold using the cumulative distribution functions (CDFs) of localization error, as shown in Fig. 11a. These CDFs capture the error distribution for the base, EV, and TL models across varying NFs. We selected the 80th percentile of the TL model’s error distribution as the threshold, which corresponds to approximately 80.1 cm (dashed line). This percentile is chosen because it reflects a balance between accuracy and robustness, ensuring that the model performs acceptably for the majority (80%) of the test cases without being overly optimistic or overly strict.

Next, we evaluated the generalization capacity of the model that achieved the highest ME score, identified in Fig. 10. This model, trained at NF = 8, was directly tested across a broad range of environmental variations, from NF = 1 to NF = 10. As shown in Fig. 11b, the MLE remains consistently below the predefined threshold of 80.1 cm across all noise conditions, demonstrating strong generalization capability. Notably, the lowest error occurs at NF = 8, which aligns with the training condition of the selected model, further validating its optimality under that specific setting.

Together, these figures serve complementary purposes: Fig. 11a defines the error threshold based on the TL model’s distribution, while Fig. 11b assesses how well the model sustains performance across different environments without fine-tuning. The results confirm that the proposed TL model generalizes effectively, reinforcing its robustness and practical applicability in dynamic industrial conditions.

## F. Limitations’ Discussion and Future Work

Although the proposed TL-driven VLC localization framework achieves substantial gains in accuracy, energy, and data efficiency, several practical limitations and open questions remain that warrant explicit discussion and future study.

(i) Adaptability under extreme environmental conditions: The optical VLC channel is sensitive to extreme lighting conditions (direct sunlight, complete darkness, or rapid illumination transients) and to very dense multi-obstacle layouts that can reduce LOS reliability and increase positioning error [63], [64]. These edge cases are not fully covered by our current experiments and represent opportunities for future investigation, particularly through hybrid sensing, optical-filtering or adaptive model strategies. Despite these challenges, the proposed framework holds potential for real-world applications, including smart logistics, industrial digital twins, and other factory automation scenarios where precise indoor localization is critical.

(ii) Synchronization and deployment realism: The measurement campaign used a VNA as a common frequency/time reference for TX and RX, which reduces synchronizationrelated variance relative to unsynchronized hardware. In field deployments with separate, unsynchronized TX and RX hardware, timing/frequency mismatches and small systematic offsets may increase measurement variance and bias feature values, potentially degrading localization accuracy. Practical mitigations include short calibration scans at a few known reference locations, insertion of pilot/reference tones for alignment, and per-scan normalization or offset correction of RSSI traces. We leave a systematic experimental quantification of these deployment effects to future work.

(iii) Multi-sensor fusion: Prior studies have shown that combining complementary sensors such as barometers, 3- D maps, geomagnetic sensors, and inertial measurements, can substantially enhance localization robustness and enable seamless indoor–outdoor operation [65]–[67]. These findings motivate exploring hybrid VLC-based approaches that integrate such modalities for improved adaptability and non-lineof-sight resilience. While our proposed work focuses solely on VLC-based TL, extending the framework toward multi-sensor fusion remains an open research direction. Several complementary fusion architectures could be investigated to further improve robustness and generalization: (a) feature-level fusion, where synchronized barometric, geomagnetic, or inertial measurement unit (IMU) features can be combined with VLC features prior to fine-tuning; (b) probabilistic constraintfusion, in which VLC model outputs can be integrated with auxiliary sensor priors using filtering or factor-graph optimization and (c) late or opportunistic fusion, where estimates from different modalities are adaptively weighted based on signal quality or model confidence [68], [69].

(iv) Scalability and computational cost: Scaling our proposed method to much larger VLC infrastructures or to more complex neural network architectures introduces synchronization, hierarchical-modeling, and data-management challenges that require systematic study and remain open for future investigation. In addition, fine-tuning directly on edge devices reduces centralized retraining but can impose CPU/GPU, memory, and energy burdens on edge hardware. Evaluating lightweight TL variants and measuring practical limits on representative edge platforms are left as important directions for future work.

(v) Energy measurement: For controlled and repeatable experiments we initially constrained the energy term to training-stage consumption only; however, this excludes inference-stage and network communication energy which can be significant in deployed IoT scenarios. Training-stage energy was selected to ensure consistent comparisons across models and noise conditions, as inference energy can vary widely with deployment hardware and optimizations such as quantization, batching, or accelerators. Extending the analysis toward full life-cycle energy accounting—including training, inference, and communication phases—remains an open direction for future refinement [70], [71].

![](images/36f4b17bb26565d515d2bf97cd7f1862c8ed7f49ced98791a65075ab87c356bd.jpg)  
(a)

![](images/216f911ba01a1549239adbd72ca9bef4eac3734e8792ca031d909d3c19ed8a3f.jpg)  
(b)  
Fig. 11: (a) CDFs of localization error for Base, EV, and TL models under varying NFs and (b) Direct testing of the best ME model across different NFs. The dashed line indicates the 80th-percentile error threshold (80.1 cm).

## VII. CONCLUSION

In conclusion, this study presents a robust and efficient TL framework for VLC-based indoor localization in dynamic industrial environments. By developing a unified ME metric that encompasses performance accuracy, energy efficiency, data efficiency, and transfer gain, the proposed approach significantly improves localization performance. Specifically, it achieves up to 67.5% ME and reduces localization error by up to 47% compared to conventional models. Furthermore, it maintains high accuracy using only 30% of the original VLC localization dataset collected from the production environment, highlighting its adaptability and scalability. Thus, the proposed framework holds significant potential for industrial automation and smart logistics. Future work will explore extending the framework’s applicability through multi-sensor data fusion, deployment on edge devices for low-latency inference, and adaptation to increasingly complex and dynamic industrial layouts to enable real-time system operation.

## ACKNOWLEDGMENT

The authors gratefully acknowledge the financial supports of the EU Horizon 2020 program towards the 6G BRAINS project H2020-ICT 101017226.

## REFERENCES

[1] M. Jan, W. Njima, X. Zhang, and A. Artemenko, “Transfer Learning for VLC-Based Indoor Localization: Addressing Environmental Variability,” in 2025 IEEE 101st Vehicular Technology Conference (VTC2025- Spring), 2025, pp. 1–7.

[2] T. Aziz and K. Insoo, “Enhancing Indoor Localization Accuracy through Multiple Access Point Deployment,” Electronics, vol. 13, no. 16, p. 3307, 2024.

[3] M. Knitt, Y. Elgouhary, J. Schyga, H. Rose, and J. Kreutzfeldt, “Benchmarking for the Indoor Localization of Autonomous Mobile Robots in Intralogistics,” Logistics journal/Proceedings, vol. 2023, 2023.

[4] A. Ullah, M. Younas, and M. S. Saharudin, “Digital Twin Framework using Real-Time Asset Tracking for Smart Flexible Manufacturing System.” Machines, vol. 13, no. 1, 2025.

[5] I. Marasovic, G. Maji´ c, I.´ Skalic, and<sup>ˇ</sup> Z. Tomasovi<sup>ˇ</sup> c, “Indoor Local-´ ization of Industrial IoT Devices and Applications Based on Recurrent Neural Networks,” Journal of communications software and systems, vol. 20, no. 1, pp. 137–145, 2024.

[6] S. J. Pan, V. W. Zheng, Q. Yang, and D. H. Hu, “Transfer Learning for Wifi-based Indoor Localization,” in Association for the advancement of artificial intelligence (AAAI) workshop, vol. 6. The Association for the Advancement of Artificial Intelligence Palo Alto, 2008.

[7] K. Zheng, H. Wang, H. Li, W. Xiang, L. Lei, J. Qiao, and X. S. Shen, “Energy-Efficient Localization and Tracking of Mobile Devices in Wireless Sensor Networks,” IEEE Transactions on Vehicular Technology, vol. 66, no. 3, pp. 2714–2726, 2016.

[8] A. I. Ahmed, Y. Etiabi, A. W. Azim, and E. M. Amhoud, “A Unified Deep Transfer Learning Model for Accurate IoT Localization in Diverse Environments,” in 2024 IEEE 35th International Symposium on Personal, Indoor and Mobile Radio Communications (PIMRC). IEEE, 2024, pp. 1–6.

[9] B.-J. Chen and R. Y. Chang, “Few-Shot Transfer Learning for Device-Free Fingerprinting Indoor Localization,” in ICC 2022-IEEE International Conference on Communications. IEEE, 2022, pp. 4631–4636.

[10] J. Dai, M. Wang, B. Wu, J. Shen, and X. Wang, “A Survey of Latest Wi-Fi Assisted Indoor Positioning on Different Principles,” Sensors, vol. 23, no. 18, p. 7961, 2023.

[11] J. L. S. Gonzalez, L. M. S. Morillo, J. A.´ Alvarez-Garc<sup>´</sup> ´ıa, F. E. D. S. Ros, and A. R. J. Ruiz, “Energy-Efficient Indoor Localization WiFi-Fingerprint System: An Experimental Study,” IEEE Access, vol. 7, pp. 162 664–162 682, 2019.

[12] A. Lindemann, B. Schnor, J. Sohre, and P. Vogel, “Indoor Positioning: A Comparison of WiFi and Bluetooth Low Energy for Region Monitoring,” in International Conference on Health Informatics, vol. 6. SCITEPRESS, 2016, pp. 314–321.

[13] F. Milano, H. da Rocha, M. Laracca, L. Ferrigno, A. Esp´ırito Santo, J. Salvado, and V. Paciello, “BLE-based Indoor Localization: Analysis of Some Solutions for Performance Improvement,” Sensors, vol. 24, no. 2, p. 376, 2024.

[14] H. Wang, G. Ganesh, M. Zon, O. Ghosh, H. Siu, and Q. Fang, “A BLE based Turnkey Indoor Positioning System for Mobility Assessment in Aging-In-Place Settings,” PLOS Digital Health, vol. 4, no. 4, p. e0000774, 2025.

[15] J. Xiao, Z. Zhou, Y. Yi, and L. M. Ni, “A Survey on Wireless Indoor Localization from the Device Perspective,” ACM Computing Surveys (CSUR), vol. 49, no. 2, pp. 1–31, 2016.

[16] S. Khriji, D. El Houssaini, and O. Kanoun, “Design of an Energy-Efficient Hybrid UWB-RF Indoor Distance Estimation System.” in SENSORNETS, 2022, pp. 249–255.

[17] S. O. Schmidt and H. Hellbruck, “Detection and Identification of¨ Multipath Interference With Adaption of Transmission Band for UWB Transceiver Systems.” in IPIN-WiP, 2021.

[18] T. Aziz and I. Koo, “A Comprehensive Review of Indoor Localization Techniques and Applications in Various Sectors.” Applied Sciences (2076-3417), vol. 15, no. 3, 2025.

[19] Y. Almadani, D. Plets, S. Bastiaens, W. Joseph, M. Ijaz, Z. Ghassemlooy, and S. Rajbhandari, “Visible Light Communications for Industrial Applications—Challenges and Potentials,” Electronics, vol. 9, no. 12, p. 2157, 2020.

[20] S. Shao, A. Khreishah, and I. Khalil, “Joint Link Scheduling and Brightness Control for Greening VLC-based Indoor Access Networks,” Journal of Optical Communications and Networking, vol. 8, no. 3, pp. 148–161, 2016.

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

[32] H. Zhao, W. Njima, and X. Zhang, “High-Accuracy and Real-Time Fingerprint-Based Continual Learning Localization System in Dynamic Environment,” Sensors, vol. 25, no. 5, p. 1289, 2025.

[33] D. Su, X. Wang, S. Liu, and W. Ding, “Four-Dimensional Indoor Visible Light Positioning: A Deep-Learning-based Perspective,” Journal of the Franklin Institute, vol. 360, no. 6, pp. 4071–4090, 2023.

[34] Z. Zhu, Y. Yang, M. Chen, C. Guo, J. Cheng, and S. Cui, “A Survey on Indoor Visible Light Positioning Systems: Fundamentals, Applications, and Challenges,” arXiv preprint arXiv:2401.13893, 2024.

[35] Bosch, “Nexeed: Welcome to the Smart Factory,” https://www.bosch. com/stories/nexeed-smart-factory/, 2025, accessed: June 14, 2025.

[36] D. C. Nguyen, M. Ding, P. N. Pathirana, A. Seneviratne, J. Li, and H. V. Poor, “Federated Learning for Internet of Things: A Comprehensive Survey,” IEEE Communications Surveys & Tutorials, vol. 23, no. 3, pp. 1622–1658, 2021.

[37] R. Kumar, R. Popli, V. Khullar, I. Kansal, and A. Sharma, “Confidentiality Preserved Federated Learning for Indoor Localization Using Wi-Fi Fingerprinting,” Buildings, vol. 13, no. 8, p. 2048, 2023.

[38] Y. Etiabi, W. Njima, and E. M. Amhoud, “Federated Learning based Hierarchical 3D Indoor Localization,” in 2023 IEEE Wireless Communications and Networking Conference (WCNC). IEEE, 2023, pp. 1–6.

[39] ——, “Femloc: Federated meta-learning for adaptive wireless indoor localization tasks in iot networks,” IEEE Internet of Things Journal, 2024.

[40] Y. Ibnatta, M. Khaldoun, and M. Sadik, “The Indoor Localization System Based on Federated Learning and RSS Using UWB-OFDM,” in International Conference on Advanced Intelligent Systems for Sustainable Development. Springer, 2022, pp. 152–167.

[41] X. Cheng, C. Ma, J. Li, H. Song, F. Shu, and J. Wang, “Federated Learning-Based Localization With Heterogeneous Fingerprint Database,” IEEE Wireless Communications Letters, vol. 11, no. 7, pp. 1364–1368, 2022.

[42] J. Guo, I. W.-H. Ho, Y. Hou, and Z. Li, “FedPos: A Federated Transfer Learning Framework for CSI-based Wi-Fi Indoor Positioning,” IEEE Systems Journal, 2023.

[43] Y. Etiabi, W. Njima, and E. M. Amhoud, “FeMLoc: Federated Meta-Learning for Adaptive Wireless Indoor Localization Tasks in IoT Networks,” IEEE Internet of Things Journal, vol. 11, no. 22, pp. 36 991– 37 007, 2024.

[44] L. You, S. Liu, Y. Chang, and C. Yuen, “A Triple-Step Asynchronous Federated Learning Mechanism for Client Activation, Interaction Optimization, and Aggregation Enhancement,” IEEE Internet of Things Journal, pp. 24 199–24 211, 2022.

[45] G. M. Van de Ven, T. Tuytelaars, and A. S. Tolias, “Three Types of Incremental Learning,” Nature Machine Intelligence, vol. 4, no. 12, pp. 1185–1197, 2022.

[46] M. De Lange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. Slabaugh, and T. Tuytelaars, “A Continual Learning Survey: Defying Forgetting in Classification Tasks,” IEEE transactions on pattern analysis and machine intelligence, vol. 44, no. 7, pp. 3366–3385, 2021.

[47] M. Masana, X. Liu, B. Twardowski, M. Menta, A. D. Bagdanov, and J. Van De Weijer, “Class-Incremental Learning: Survey and Performance Evaluation on Image Classification,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 5, pp. 5513–5533, 2022.

[48] S. Naser, L. Bariah, S. Muhaidat, P. C. Sofotasios, M. Al-Qutayri, E. Damiani, and M. Debbah, “Toward Federated-Learning-Enabled Visible Light Communication in 6G Systems,” IEEE Wireless Communications, vol. 29, no. 1, pp. 48–56, 2022.

[49] S. M. Sheikholeslami, A. Rasti-Meymandi, S. J. Seyed-Mohammadi, J. Abouei, and K. N. Plataniotis, “Communication-Efficient Federated Learning for Hybrid VLC/RF Indoor Systems,” IEEE Access, vol. 10, pp. 126 479–126 493, 2022.

[50] Y. Yan, C.-M. Feng, M. Ye, W. Zuo, P. Li, R. S. M. Goh, L. Zhu, and C. Chen, “Rethinking Client Drift in Federated Learning: A Logit Perspective,” arXiv preprint arXiv:2308.10162, 2023.

[51] I. J. Goodfellow, M. Mirza, D. Xiao, A. Courville, and Y. Bengio, “An Empirical Investigation of Catastrophic Forgetting in Gradient-based Neural Networks,” arXiv preprint arXiv:1312.6211, 2013.

[52] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming Catastrophic Forgetting in Neural Networks,” Proceedings of the national academy of sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[53] M. Iman, H. R. Arabnia, and K. Rasheed, “A Review of Deep Transfer Learning and Recent Advancements,” Technologies, vol. 11, no. 2, p. 40, 2023.

[54] A. H. Ali, M. G. Yaseen, M. Aljanabi, and S. A. Abed, “Transfer Learning: A New Promising techniques,” Mesopotamian Journal of Big Data, vol. 2023, pp. 29–30, 2023.

[55] W. Ying, Y. Zhang, J. Huang, and Q. Yang, “Transfer Learning via Learning to Transfer,” in International conference on machine learning. PMLR, 2018, pp. 5085–5094.

[56] S. J. Pan and Q. Yang, “A Survey on Transfer Learning,” IEEE Transactions on knowledge and data engineering, vol. 22, no. 10, pp. 1345–1359, 2009.

[57] T. Suwannaphong, R. McConville, and I. Craddock, “Transfer Learning of RSSI to Improve Indoor Localisation Performance,” arXiv preprint arXiv:2412.09292, 2024.

[58] K. Adamkiewicz, P. Koch, B. Morawska, P. Lipinski, K. Lichy, and´ M. Leplawy, “Improving UWB Indoor Localization Accuracy using Sparse Fingerprinting and Transfer Learning,” in International Conference on Computational Science. Springer, 2021, pp. 291–302.

[59] Y. Zhuang, L. Hua, Q. Wang, Y. Cao, Z. Gao, L. Qi, J. Yang, and J. Thompson, “Visible Light Positioning and Navigation using

Noise Measurement and Mitigation,” IEEE Transactions on Vehicular Technology, vol. 68, no. 11, pp. 11 094–11 106, 2019.

[60] S. Zhang, P. Du, H. Yang, R. Zhang, C. Chen, and A. Alphones, “Recent Progress in Visible Light Positioning and Communication Systems.” IEICE Trans. Commun., vol. 106, no. 2, pp. 84–100, 2023.

[61] V. P. Rekkas, L. A. Iliadis, S. P. Sotiroudis, A. D. Boursianis, P. Sarigiannidis, D. Plets, W. Joseph, S. Wan, C. G. Christodoulou, G. K. Karagiannidis et al., “Artificial Intelligence in Visible Light Positioning for Indoor IoT: A Methodological Review,” IEEE Open Journal of the Communications Society, vol. 4, pp. 2838–2869, 2023.

[62] L. Yang, P. Finnerty, and C. Ohta, “Indoor Localization Using Routerto-Router Rssi and Transfer Learning for Dynamic Environments,” Available at SSRN 5133124, 2024.

[63] S. Gupta, D. Roy, S. Bose, V. Dixit, and A. Kumar, “Illuminating the Future: A Comprehensive Review of Visible Light Communication Applications,” Optics & Laser Technology, vol. 177, p. 111182, 2024.

[64] M. A. S. Sejan, M. H. Rahman, M. A. Aziz, D.-S. Kim, Y.-H. You, and H.-K. Song, “A Comprehensive Survey on MIMO Visible Light Communication: Current Research, Machine Learning and Future Trends,” Sensors, vol. 23, no. 2, p. 739, 2023.

[65] J. Li, Y. Song, Z. Ma, Y. Liu, and C. Chen, “A Review of Indoor Localization Methods Leveraging Smartphone Sensors and Spatial Context,” Sensors, vol. 24, no. 21, p. 6956, 2024.

[66] Y.-H. Kuo and E. H.-K. Wu, “Intelligent Geomagnetic Indoor Positioning System,” Electronics, vol. 12, no. 10, p. 2227, 2023.

[67] J. Wang, C. Shi, M. Xia, F. Zheng, T. Li, Y. Shan, G. Jing, W. Chen, and T. Hsia, “Seamless Indoor–Outdoor Foot-Mounted Inertial Pedestrian Positioning System Enhanced by Smartphone PPP/3-D Map/Barometer,” IEEE Internet ofThings Journal, vol. 11, no. 7, pp. 13 051–13 069, 2023.

[68] M. Kaess, H. Johannsson, R. Roberts, V. Ila, J. J. Leonard, and F. Dellaert, “ISAM2: Incremental Smoothing and Mapping using the Bayes Tree,” The International Journal of Robotics Research, vol. 31, no. 2, pp. 216–235, 2012.

[69] Y. Li, M. E. H. Daho, P.-H. Conze, R. Zeghlache, H. Le Boite,´ R. Tadayoni, B. Cochener, M. Lamard, and G. Quellec, “A Review of Deep Learning-based Information Fusion Techniques for Multimodal Medical Image Classification,” Computers in Biology and Medicine, vol. 177, p. 108635, 2024.

[70] I. Mavromatis, K. Katsaros, and A. Khan, “Computing Within Limits: An Empirical Study of Energy Consumption in ML Training and Inference,” arXiv preprint arXiv:2406.14328, 2024.

[71] S.-K. Chou, J. Hribar, M. Mohorciˇ c, and C. Fortuna, “The Energy Costˇ of Artificial Intelligence of Things Lifecycle,” arXiv e-prints, pp. arXiv– 2408, 2024.

![](images/35725149ff4860267e2d0314dc57850e29e292de3b5fecbe66a34583b20704e7.jpg)

Masood Jan is a Ph.D. candidate in Machine Learning at Sorbonne University, Paris, France, and is affiliated with the LISITE Laboratory at ISEP, Paris. He was ranked first in the EDITE 2023 doctoral campaign. He received his Master’s degree in Telecommunications and Networks from the Conservatoire National des Arts et Metiers (CNAM), Paris,´ France, in 2023, and the Eng. degree in Electronics Engineering from the University of Engineering & Technology (UET), Peshawar, Pakistan, in 2008. Prior to his doctoral studies, he gained over 12

years of experience in the wireless communications industry, working with multinational companies on AI-driven analytics, automation, and network optimization for LTE/5G systems. His current research interests include machine learning in wireless communication and efficient AI models for 6G/OWC. He is actively engaged in academic dissemination and has delivered numerous talks at conferences and seminars.

![](images/b886397420a3d9e61909c37e53adbb108ad5cc5f857413f1f9b12d0e349b162c.jpg)

Wafa Njima (Member, IEEE) received the Eng. degree from the Institut National des Sciences Appliquees et de Technologies de Tunis in 2015. She´ received the Ph.D. degree in the field of radiocommunications in 2019, from the Conservatoire National des Arts et Metiers in Paris in collaboration´ with the Ecole Superieure des Communications de´ Tunis. She joined the ETIS laboratory at ENSEA Cergy as a temporary assistant and researcher also as a postdoctoral researcher. She is currently an associate professor at ISEP, Paris. Her publications span several research areas, and her research interests are related to several topics including signal processing, wireless communications, indoor localization, IoT and machine learning for communications. She is IEEE member and she served as TPC member and reviewer for many leading international conferences and journals.

![](images/b819aad7e74c229793979649b1d85ed41d7a87733f819ffe0b71e1a0c6b0b1a4.jpg)

Xun Zhang (Senior Member, IEEE) is a Professor at ISEP (www.isep.fr), Paris, France, and conducts permanent research at the LISV Laboratory within the University of Saclay-Versailles. As a Senior Member of the IEEE, he serves as an Associate Editor for the IEEE Transactions on Broadcasting and is a member of the BTS Awards Committee. He earned his Ph.D. degree in Instrumentation and Microelectronics (IM) Science from the University of Nancy in October 2009. Between 2009 and 2011, Dr. Zhang undertook a Postdoctoral position at the

SCEE Laboratory in CentraleSupelec, France, where he focused on applying´ auto-reconfigurable management methods and control strategies in FPGAs for cognitive radio systems. In 2011, Dr. Zhang joined the LISITE Laboratory at ISEP as an Associate Professor. His research primarily revolves around PHY layer optimization for 5G cellular networks and Visible Light Communication (VLC) systems. He is also pioneering the development of VLC-based indoor high-accuracy positioning and tracking algorithms.

![](images/752d62ac14622146502d6c4b2362317f10c5f126040020490a38531efc5e83a4.jpg)

Alexander Artemenko is a research engineer within the Corporate Sector Research of Robert Bosch GmbH in Stuttgart, Germany. Prior to joining BOSCH in 2015, he was a postdoctoral researcher at the Ilmenau University of Technology (TU Ilmenau), Germany from 2013 to 2015. He received his PhD in engineering from the same university in 2013. In 2012, Dr. Artemenko was a post-doctoral researcher and lecturer at UCLA (Univ. of California Los Angeles) in the group of Professor Mario Gerla.