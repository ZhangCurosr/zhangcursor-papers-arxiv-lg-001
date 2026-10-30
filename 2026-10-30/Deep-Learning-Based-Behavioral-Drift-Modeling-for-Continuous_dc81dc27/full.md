# Deep Learning Based Behavioral Drift Modeling for Continuous Biometric Authentication

Suprava Saha Dibya Minar Abdulla Al Noman Md. Zafir Abid Shresto Ahmed Sayef

Abstract—This research proposes a two-level keystroke dynamics authentication system that provides defense-in-depth security through continuous behavioral drift monitoring. Level 1 employs fixed-text password authentication using Temporal Convolutional Networks (TCN) with contrastive learning and population-based training (PBT), achieving 3.77% EER for high-security login verification. Upon successful authentication, Level 2 continuously monitors user typing during natural free-text sessions using a 4-Model Deep Learning Ensemble. If suspicious behavioral drift is detected during monitoring, the system re-authenticates the user by returning to Level 1, providing a feedback loop against session hijacking and behavioral anomalies. The system explicitly models long-term behavioral drift through the fixedtext component’s PBT mechanism, which adapts to temporal variations in password typing patterns. This research leverages free-text (Buffalo dataset) and fixed-text (GREYC dataset) keystroke dynamics datasets, extracting temporal and spatial features including hold times, up-down latencies, down-down intervals, and trigram patterns. The final system achieves 96%+ accuracy for fixed-text authentication (3.77% EER) and 83.53% True Acceptance Rate (TAR) at 5% False Acceptance Rate (FAR) for free-text continuous authentication (10.05% EER), demonstrating superior performance and strong resistance to impostor attacks through continuous behavioral monitoring.

Index Terms—Continuous Authentication; Behavioral Biometrics; Keystroke Dynamics; Deep Learning; TCN; Ensemble Learning

## I. INTRODUCTION

## A. Background

have rendered traditional physiological biometrics increasingly vulnerable to spoofing and replay attacks. While physiological traits (fingerprints, iris) are static, they can be stolen; once compromised, they cannot be reset. In contrast, behavioral biometrics focus on human behavior — specifically the unique motor-neural pathways involved in typing — providing a dynamic layer of security. However, the adoption of behavioral biometrics is hindered by the phenomenon of behavioral drift. Human behavior is not stationary; it evolves due to learning effects (Power Law of Practice) [17], fatigue, stress, and environmental changes. Our analysis of longitudinal data reveals that over a 110-day period, 66.7% of users exhibit statistically significant changes in typing latency. Existing deep learning frameworks, such as SwipeFormer or BehaveFormer, often rely on static training paradigms that fail to account for this temporal aging. When a user’s behavior drifts beyond the model’s learned decision boundary, the system triggers false rejections (High FRR). Furthermore, retraining these massive models to accommodate new data often leads to “catastrophic forgetting,” where the model overwrites previously learned discriminative features.

## B. Rationale of the study or motivation

The motivation for this research stems from critical limitations in contemporary biometric systems, primarily their inability to handle behavioral drift — the natural evolution of user typing patterns over time. Current infrastructures rely on static templates, leading to high False Rejection Rates (FRR) as legitimate users evolve, while remaining vulnerable to impostors who replicate static patterns. Furthermore, existing deep learning (DL) frameworks (e.g., SwipeFormer) require computationally impractical retraining to address this drift, often resulting in “catastrophic forgetting” of the original identity features. This study bridges these gaps by proposing an evolutionary framework that utilizes PBT within a TCN. This approach treats behavioral biometrics not as isolated static events, but as a “gradually occurring phenomenon,” enabling the system to evolve alongside the user while maintaining robust defense-in-depth security.

## C. Problem Statement

Current authentication infrastructures face three critical failures:

1) Static Assumptions: Contemporary systems view authentication events as independent snapshots, disregarding the continuous evolutionary trajectory of legitimate user behavior.

2) Catastrophic Forgetting: To update a biometric template, systems often require retraining, which risks erasing the core identity features learned during enrolment.

3) Frog-Boiling Attacks: Highly adaptive systems risk being duped by impostors who slowly mimic the user’s behavior over time, a vulnerability known as the frogboiling effect.

## D. Objective

This research aims to resolve the limitations of static biometric systems by developing a robust, continuous authentication framework capable of handling the dynamic nature of human behavior. The primary goal is to bridge the gap between high-security requirements and user convenience by addressing the “stability-plasticity dilemma” — the need to adapt to new behaviors without losing the core identity

The specific objectives of this study are:

1) Development of a Hybrid Adaptive Framework: To design a keystroke dynamics architecture that combines the temporal representational power of TCN with the flexibility of evolutionary adaptation. The objective is to create a system where the backbone captures universal features while lightweight modules adapt to specific user nuances.

2) Mitigation of Catastrophic Forgetting: To implement a training strategy, specifically PBT, that enables the system to learn from new behavioral data (drift) without overwriting or “forgetting” the distinct historical traits that define the genuine user.

3) Prevention of Adversarial Adaptation (Frog-Boiling): To engineer safeguard mechanisms, such as Trust Anchors, that prevent the system from being deceived by impostors who attempt to slowly mimic a user’s behavior over time. The goal is to distinguish between legitimate natural drift and malicious gradual shifts.

4) Continuous Anomaly Detection: To ascertain the system’s ability to detect anomalous patterns in real-time dynamic environments, moving beyond static login verification to continuous, session-long monitoring.

5) Performance Optimization: To surpass the accuracy of current state-of-the-art systems by minimizing FAR and FRR, thereby reducing the EER to banking-grade security standards.

## E. Methodology in Brief

This research implements a Two-Level Behavioral Authentication Framework (figure 1) tailored to distinct security contexts: active login verification and passive continuous monitoring.

1) Level 1: Fixed-Text Authentication (Evo-BioTCN): The first level secures the login process using the GREYC dataset [14] (133 users, 110-day span). We propose the Evo-BioTCN architecture, which employs a Dual-Stage Strategy:

• Frozen Backbone: A TCN captures universal temporal dependencies in keystroke latencies (Hold, Flight, Down-Down).

• Drift Adaptation: To counter template aging without catastrophic forgetting, a lightweight adapter module is evolved using PBT. This allows the model to track the user’s behavioral trajectory over months.

2) Level 2: Free-Text Continuous Monitoring (4-Model Ensemble): Upon successful login, Level 2 continuously validates identity using the Buffalo dataset [15] (75 users). Due to the unstructured nature of free-text, we developed a Reconstruction Pipeline to convert raw digraphs into character-level timing vectors. The authentication engine is a 4-Model Deep Learning Ensemble fusing complementary metric learning architectures (ArcFace, Multi-Similarity, Triplet, and Center Loss). This ensemble uses score-level fusion to handle the high variance of natural typing, triggering a re-authentication to Level 1 if an anomaly is detected.

![](images/b22dbc9cfad7cbfedc04ff6e68764a22a28d62ff9582df30f240539f8d8aa5e8.jpg)  
Fig. 1. Two-Level Behavioral Authentication System

## F. Scope and Challenges

1) Scope: This study is strictly focused on keystroke dynamics as the primary behavioral modality. It prioritizes continuous assessment (zero-trust) over static identification. The research utilizes longitudinal datasets (GREYC [14] and Buffalo [15]) to specifically evaluate resilience against behavioral drift, excluding other modalities like gait or facial recognition.

2) Key Challenges:

1) Stability-Plasticity Dilemma: The system must adapt to new behavioral patterns (plasticity) without erasing the core identity features learned during enrolment (stability), preventing catastrophic forgetting.

2) Adversarial Drift (Frog-Boiling): A critical risk is “slow-mimicry” attacks, where an impostor gradually alters their behavior to converge with the user’s profile. We address this by integrating Trust Anchors in the PBT fitness function to constrain adaptation.

3) Data Quality: Behavioral data is inherently noisy due to environmental factors. Handling outliers and missing values requires specialized preprocessing, including Modified Z-score filtering and session-relative normalization.

## II. LITERATURE REVIEW

The most highlighted aspect was the lackings and weakness of the existing static models of authentication throughout the literature review. Several authors such as - Baig and Eskeland [5], Patel et al. [3], Ryu et al. [4] mentioned temporal aging explicitly that indicated the natural behavioral changes of an individual as a remaining crucial challenge. Infrastructures which were developed on established templates like one class Support Vector Machine (SVM) demonstrated by Sun eta al. [15] or the snapshot based system mentioned by Sagbas and Balli [2] accomplished higher accuracy but the main issue was they were brittle by nature. They needed to be trained again in order to differentiate between an anonymous intruder and an actual natural change. Additionally, few shot based methods like the one mentioned by Fereidooni et al. [1] provided scalability but they did not capture the evolution of behaviour and depended on fixed samples. This study considers the change of behavior as the main feature that models through the lineage of behavioral evolution instead of non relevant training periods and treating it as an error. Moreover, recent research on the modeling of patterns provided the effectiveness of the transformer based infrastructures. Senarath et al. [9] demonstrated a dual attention and spatio-temporal based infrastructure named Behaveformer that extracted large dependencies in behavioral data. But the major drawback remained in the inability to address behavioral drift adaptation.

Similar to transformer based infrastructure, Chen et al. [7] introduced a dual channel based transformer named DESTER (Dual Stream Transformer based Emotion Recognition Model) for identifying emotions. But the authors later found the issue of catastrophic forgetting when the behavioral data of a user changed because of technical and environmental purposes. In multi modal scenarios, sometimes severe challenges arise for proper accuracy which was demonstrated by Wang et al. [11] through their system that selected modalities based on the scene-irrelated concept. However, their infrastructure was heavily dependent on static feature representations that could not address temporal aging. In their research on Siamese Neural Network, Solano et al [6] and Lis et al. [10] stated promising possibilities on a few shot verification systems. But their studies had severe limitations because of rigid architecture and failure to constant evolution over time.

Furthermore, large discriminative power was highlighted for keystroke and mouse data by the research of Ranganathan [12] and Mazumder and Sundaram [13] on VAE-GNN infrastructure and Vision Transformers (ViT) respectively. But their models assumed the learned features were fixed which was impractical in real life and did not address behavioral drift. Finally, Subash et al. [8] demonstrated from the concept of behavioral biometrics and learning curve that continuous retraining led to rise in the rate of FRR. Hence the authors suggested focusing on an infrastructure that could evolve through a different mechanism other than retraining. This research incorporates these findings by using PBT for the evolution of the adapters and frozen backbone structure to capture temporal dependencies. While existing traditional infrastructure fails to address temporal aging, this framework considers these changes as evolving trajectories and prevents frog boiling attacks through the integration of trust anchors so that the model can evolve alongside the user.

## III. REQUIREMENTS, IMPACTS AND CONSTRAINTS

## A. Functional Specifications and Design Requirements

The system design was driven by stakeholder feedback emphasizing the need for non-intrusive security that accommodates natural behavioral evolution.

• Temporal Modeling: The system must utilize TCN to efficiently capture sequential dependencies in keystroke data, replacing computationally expensive Transformers to ensure real-time processing.

• Continuous Assessment: Unlike static login systems, the architecture must perform continuous background monitoring (Level 2) to validate user identity throughout the session.

• Drift Adaptation: The system requires a mechanism to distinguish between legitimate behavioral drift (template aging) and malicious anomalies. This necessitates the implementation of PBT to evolve user templates without catastrophic forgetting.

• Data Minimization: To comply with privacy standards General Data Protection Regulation (GDPR), the system must process only timing feature vectors, strictly excluding semantic text content to prevent user profiling.

## B. Societal and Ethical Impact

• Operational Security: The framework enforces a Zero-Trust architecture. By continuously validating identity, it mitigates session hijacking risks in high-sensitivity sectors like banking and healthcare.

• Inclusivity: Traditional biometrics often reject users due to natural behavioral changes (aging, injury). By explicitly modeling drift, this system reduces FRR, ensuring equitable access for users with evolving motor-neural patterns.

• Privacy and Ethics: The system adheres to strict data minimization. However, the risk of “frog-boiling” attacks (slow-mimicry) requires the implementation of Trust Anchors in the PBT fitness function to constrain adaptation within safe boundaries.

## C. Environmental Sustainability (Green AI)

A critical innovation of this research is its focus on computational efficiency. Traditional adaptive systems require frequent, energy-intensive retraining of deep neural networks to address drift. In contrast, this framework employs a Frozen Backbone strategy. Once the core TCN is trained, only lightweight adapter modules (approx. 512 parameters) are updated. This drastically reduces Floating Point Operations (FLOPs) and carbon footprint over the system lifecycle, making it suitable for resource-constrained edge deployment.

## D. Project Management and Economic Feasibility

The system design was driven by stakeholder feedback emphasizing the need for non-intrusive security that accommodates natural behavioral evolution.

• Iterative Design: The project methodology was agile. Initial experiments with Transformers revealed high computational costs and instability, prompting a strategic pivot to the TCN architecture, which offered superior training stability and linear O(n) complexity.

• Risk Management: Primary risks included Catastrophic Forgetting (mitigated by freezing the backbone) and Overfitting due to data sparsity (mitigated by sessionrelative normalization and dropout).

• Economic Viability: The solution is cost-effective. By eliminating the need for specialized biometric hardware (sensors) and reducing administrative costs associated with false rejections and account recovery, the system offers high economic value. The use of open-source stacks (PyTorch, Ray Tune) kept development costs minimal.

## IV. DESIGN PROCESS OVERVIEW

This chapter delineates the methodological framework for the proposed Two-Level Behavioral Authentication System. The design is bifurcated into two distinct pipelines: Level 1 addresses fixed-text login verification using the Evo-BioTCN architecture, while Level 2 handles continuous free-text monitoring using a 4-Model Deep Learning Ensemble.

## A. Level 1: Fixed-Text Methodology (Evo-BioTCN)

1) Methodology Overview: The central challenge motivating the fixed-text design is behavioral drift—the unconscious evolution of typing patterns due to the Power Law of Practice [17]. Our analysis of the GREYC dataset [14] revealed that 66.7% of users exhibited statistically significant changes in typing latency over 110 days. Conventional static models fail to accommodate this “template aging.”

To resolve this, we propose a Dual-Stage Strategy:

1) Frozen Backbone: A TCN learns universal temporal features.

2) Evolutionary Adaptation: A lightweight adapter is evolved using PBT to track user behavior without catastrophic forgetting.

2) Model Specification: A TCN is selected over Transformer architectures for the Evo-BioTCN Architecture because keystroke dynamics are inherently local and causal. The latency of a key press is conditioned by its immediate predecessor (digraph context), not by future keystrokes.

![](images/f4f6caee0c670c407730dc499f86f962fe8c551e762c2195fdd30b53d323d842.jpg)  
Fig. 2. Evo-BioTCN layer architecture

a) Layer Architecture: The architecture consists of four dilated causal convolutional blocks. The receptive field is calculated as in (1):

$$
{ \mathrm { R e c e p t i v e ~ F i e l d } } = 1 + \sum _ { i = 0 } ^ { n - 1 } 2 ^ { i } ( k - 1 ) = 3 1 { \mathrm { ~ p o s i t i o n s } }\tag{1}
$$

where, $n = 4$ is the number of TCN blocks and $k = 3$ is the kernel size. This receptive field spans approximately two complete words in a typical password-sufficient to capture digraph, trigraph, and word-level rhythm patterns while maintaining O(n) linear complexity suitable for real-time authentication.

As shown in figure 2 the input flows into a 1x1 projection, followed by four TCN blocks with increasing dilation (d=1,2,4,8). The output passes through a Rank-4 Adapter, Global Pooling, and L2 Normalization to produce a 64-D embedding. The specifications of the layers are mentioned in table I.

b) Drift Adaptation via PBT: After supervised training, the backbone parameters $\theta _ { s t a t i c }$ are frozen. Only the adapter $\theta _ { a d a p t i v e }$ (512 parameters) evolves. The adapter function is defined as in (2):

$$
{ \mathrm { A d a p t e r } } ( h ) = W _ { \mathrm { u p } } \cdot { \mathrm { R e L U } } ( W _ { \mathrm { d o w n } } \cdot h ) + h\tag{2}
$$

where, $W _ { \mathrm { d o w n } } \in \mathbb { R } ^ { 4 \times 6 4 }$ and $W _ { \mathsf { u p } } \in \mathbb { R } ^ { 6 4 \times 4 }$ , yielding 512 trainable parameters per adapter (no bias terms).

TABLE IEVO-BIOTCN LAYER SPECIFICATIONS
<table><tr><td rowspan=1 colspan=1>Layer</td><td rowspan=1 colspan=1>Type</td><td rowspan=1 colspan=1>Parameters</td><td rowspan=1 colspan=1>Output Dim.</td></tr><tr><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1>Feature Vec-tor</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>231</td></tr><tr><td rowspan=1 colspan=1>Projection</td><td rowspan=1 colspan=1>1 × 1 Conv</td><td rowspan=1 colspan=1>14,848</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>TCN Block 1</td><td rowspan=1 colspan=1>DilatedCausal Conv</td><td rowspan=1 colspan=1>dilation=1, kernel=3</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>TCN Block 2</td><td rowspan=1 colspan=1>DilatedCausal Conv</td><td rowspan=1 colspan=1>dilation=2, kernel=3</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>TCN Block 3</td><td rowspan=1 colspan=1>DilatedCausal Conv</td><td rowspan=1 colspan=1>dilation=4, kernel=3</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>TCN Block 4</td><td rowspan=1 colspan=1>DilatedCausal Conv</td><td rowspan=1 colspan=1>dilation=8, kernel=3</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>TCN Adapter</td><td rowspan=1 colspan=1>1 × 1 Conv</td><td rowspan=1 colspan=1>rank=4, 512 params</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>Global Pool</td><td rowspan=1 colspan=1>Avg. Pooling</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>Output</td><td rowspan=1 colspan=1>L2 Normal-ization</td><td rowspan=1 colspan=1>一</td><td rowspan=1 colspan=1>64</td></tr></table>

PBT maintains a population of 10 adapters. We employ a Trust Anchor loss to prevent ”frog-boiling” attacks (slow mimicry) using (3):

$$
F ( A _ { i } ) = \mathrm { S e p a r a t i o n } - \lambda _ { \mathrm { a n c h o r } } \cdot D ( A _ { i } , A _ { \mathrm { a n c h o r } } )\tag{3}
$$

where, Separation = mean impostor distance - mean genuine distance

3) Feature Engineering (Fixed-Text): We extract 231 engineered features as shown in table II from raw timing measurements (Hold, Flight, Down-Down, Release-Release). These are categorized by their resilience to drift.

TABLE II  
FEATURE CATEGORIES ORGANIZED BY DRIFT RESILIENCE
<table><tr><td rowspan=1 colspan=1>Feature Category</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Drift Resilience</td><td rowspan=1 colspan=1>Rationale</td></tr><tr><td rowspan=1 colspan=1>Rank Features</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>√High</td><td rowspan=1 colspan=1>Ordinal positions invariant tospeed scaling</td></tr><tr><td rowspan=1 colspan=1>Ratio Features</td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>√ High</td><td rowspan=1 colspan=1>Proportions of total time resistglobal drift</td></tr><tr><td rowspan=1 colspan=1>Rhythm/Entropy</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>√ High</td><td rowspan=1 colspan=1>Timing variance patterns per-sist across sessions</td></tr><tr><td rowspan=1 colspan=1>Hand Balance</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>√High</td><td rowspan=1 colspan=1>Left/right asymmetry reflectsmotor dominance</td></tr><tr><td rowspan=1 colspan=1>Statistical</td><td rowspan=1 colspan=1>44</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Mean, std, CV capture within-session consistency</td></tr><tr><td rowspan=1 colspan=1>Normalized Timing</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Session-relative z-scores</td></tr><tr><td rowspan=1 colspan=1>Trigraph</td><td rowspan=1 colspan=1>24</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>3-key sequence patterns</td></tr><tr><td rowspan=1 colspan=1>Log-Transform</td><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Corrects for skewed distribu-tions</td></tr><tr><td rowspan=1 colspan=1>Positional</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>× Lower</td><td rowspan=1 colspan=1>Region-specific statistics</td></tr><tr><td rowspan=1 colspan=1>Acceleration</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>× Lower</td><td rowspan=1 colspan=1>Speed-up/slow-down patterns</td></tr><tr><td rowspan=1 colspan=1>Uncategorized</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Medium</td><td rowspan=1 colspan=1>Features not matched by cat-egory rules</td></tr></table>

## B. Level 2: Free-Text Methodology (4-Model Ensemble)

For continuous background monitoring, we utilize the Buffalo Dataset [15]. Unlike fixed-text, free-text data lacks consistent positional context, necessitating a metric learning approach.

1) Data Reconstruction Pipeline: The Buffalo dataset [15] is distributed as digraphs. We developed a ”Chain-Link”

![](images/ca3ce6b1b1bcd18a5653bf9b8273c0f3d024fa27e6d48e6a21f666f3d4ec97b7.jpg)  
Fig. 3. Buffalo dataset [15] reconstruction pipeline

algorithm to reconstruct character-level timestamps. Figure 3 shows the reconstruction pipeline for the Buffalo dataset [15].

The algorithm reconstructs absolute time t using the interkey intervals (DU, DD) as in (4), (5) and (6):

$$
t _ { \mathrm { p r e s s } } ^ { ( k _ { 1 } ) } = t _ { \mathrm { c u r r e n t } }\tag{4}
$$

$$
t _ { \mathrm { r e l e a s e } } ^ { ( k _ { 1 } ) } = t _ { \mathrm { c u r r e n t } } + D U _ { i }\tag{5}
$$

$$
t _ { \mathrm { p r e s s } } ^ { ( k _ { 2 } ) } = t _ { \mathrm { c u r r e n t } } + D D _ { i }\tag{6}
$$

a) Session-Relative Normalization: To mitigate global speed shifts between sessions, we apply session-relative normalization as in (7):

$$
\mathbf { H } _ { \mathrm { n o r m } } = \frac { \mathbf { H } } { \bar { H } _ { \mathrm { s e s s i o n } } }\tag{7}
$$

This renders features scale-invariant, focusing on rhythm rather than absolute speed.

2) Free-Text Feature Engineering: We extract 85 statistical features per word sample, expanded to a 170-dimensional vector via hybrid normalization (Global + Per-User). The overview of the engineered features is depicted in table III.

3) Model Specification: 4-Model Deep Learning Ensemble We employ an ensemble of four diverse embedding networks to handle the high variance of free-text typing as depicted in figure 4.

1) ArcFace + Multi-Similarity Model: Combines angular margin loss (L) with hard negative mining $( \mathcal { L } _ { M S } )$ as in (8).

$$
{ \mathcal { L } } _ { \mathrm { A r c F a c e } } = - \log { \frac { e ^ { s \cdot \cos ( \theta _ { y _ { i } } + m ) } } { e ^ { s \cdot \cos ( \theta _ { y _ { i } } + m ) } + \sum _ { j \neq y _ { i } } e ^ { s \cdot \cos \theta _ { j } } } }\tag{8}
$$

2) Enhanced Embedding Model: Combines Focal Triplet Loss (9) (focusing on hard triplets) and Center Loss (minimizing intra-class variance) as in (10).

TABLE III  
OVERVIEW OF THE 85 ENGINEERED KEYSTROKE FEATURES ORGANIZED BY FUNCTIONAL CATEGORY
<table><tr><td rowspan=1 colspan=1>Feature Group</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>Statistical Timing</td><td rowspan=1 colspan=1>Summary statistics of hold, flight, and latencytimes.</td></tr><tr><td rowspan=1 colspan=1>Positional</td><td rowspan=1 colspan=1>Timing variation across early, middle, and lateword segments.</td></tr><tr><td rowspan=1 colspan=1>Rhythm</td><td rowspan=1 colspan=1>Entropy and variability-based measures of typ-ing regularity.</td></tr><tr><td rowspan=1 colspan=1>Key Overlap</td><td rowspan=1 colspan=1>Metrics describing overlapping keystrokes androllover behaviour.</td></tr><tr><td rowspan=1 colspan=1>Rank-Based</td><td rowspan=1 colspan=1>Ordinal timing patterns invariant to absolutetyping speed.</td></tr><tr><td rowspan=1 colspan=1>N-gram Timing</td><td rowspan=1 colspan=1>Bigram and trigram timing statistics capturingkey sequences.</td></tr><tr><td rowspan=1 colspan=1>Dynamic Change</td><td rowspan=1 colspan=1>Acceleration and jerk reflecting rapid timingchanges.</td></tr><tr><td rowspan=1 colspan=1>Ratio-Based</td><td rowspan=1 colspan=1>Relative timing relationships between keystrokeevents.</td></tr><tr><td rowspan=1 colspan=1>Log-Scaled</td><td rowspan=1 colspan=1>Log-transformed timing features to reduce skew-ness.</td></tr></table>

![](images/765c6c0d79bc6f4d806ae9713899890873b7f3d4dbd4ff0fce50e982e9de87a7.jpg)  
Fig. 4. 4-Model Architecture

$$
{ \mathcal { L } } _ { \mathrm { t r i p l e t } } = ( 1 + L ) ^ { \gamma } \cdot L , \quad L = \log ( 1 + e ^ { d _ { p } - d _ { n } + m } )\tag{9}
$$

$$
\mathcal { L } _ { \mathrm { e n h a n c e d } } = \lambda _ { t } \mathcal { L } _ { \mathrm { t r i p l e t } } + \lambda _ { c } \mathcal { L } _ { \mathrm { c e n t e r } }\tag{10}
$$

3) Simple ArcFace Model: A lightweight Multi-Layer Perceptron (MLP) regularizer to prevent overfitting.

4) Fine-Tuned Enhanced Model: A version of Model 2 fine-tuned on hard examples with a low learning rate.

4) Score-Level Fusion: The final authentication score is a weighted sum of cosine similarities from all four models:

$$
S _ { e n s e m b l e } = \sum _ { k = 1 } ^ { 4 } w _ { k } \cdot S _ { k }\tag{11}
$$

Optimized weights: $w \ = \ [ 0 . 2 0 , 0 . 4 2 , 0 . 1 1 , 0 . 2 7 ]$ , giving maximum weight to the Enhanced Embedding model.

Table IV shows the respective loss function of each of the models of the 4-Model Architecture .

TABLE IV  
TRAINING HYPERPARAMETERS FOR ENSEMBLE COMPONENTS
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Loss Function</td><td rowspan=1 colspan=1>Epochs</td><td rowspan=1 colspan=1>Learning Rate</td><td rowspan=1 colspan=1>Dropout</td></tr><tr><td rowspan=1 colspan=1>ArcFace + MS</td><td rowspan=1 colspan=1>ArcFace + Multi-Similarity</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>3 × 10−4</td><td rowspan=1 colspan=1>0.20</td></tr><tr><td rowspan=1 colspan=1>Enhanced Embedding</td><td rowspan=1 colspan=1>Focal Triplet + Center</td><td rowspan=1 colspan=1>50</td><td rowspan=1 colspan=1>3 × 10−4</td><td rowspan=1 colspan=1>0.30</td></tr><tr><td rowspan=1 colspan=1>Simple ArcFace</td><td rowspan=1 colspan=1>ArcFace $( s = 2 0 , m = 0 . 3 )$ </td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1> $\overline { { 1 \times 1 0 ^ { - 3 } } }$ </td><td rowspan=1 colspan=1>0.40</td></tr><tr><td rowspan=1 colspan=1>Fine-Tuned</td><td rowspan=1 colspan=1>Cross-Entropy (Fine-tune)</td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1> $\overline { { 5 \times 1 0 ^ { - 5 } } }$ </td><td rowspan=1 colspan=1>0.30</td></tr></table>

5) Complementary Architecture: Drift-Aware Temporal Transformer (DATT): For preliminary investigation on the KeyRecs dataset [16], we designed the DATT. It utilizes Low-Rank Adaptation (LoRA) ( (12)) to adapt to drift efficiently.

$$
W ^ { \prime } = W _ { 0 } + \Delta W = W _ { 0 } + B A\tag{12}
$$

where, $B \in \mathbb { R } ^ { d \times r }$ and $A \in \mathbb { R } ^ { r \times k }$ are low-rank matrices $( r ~ = ~ 1 6 )$ . This allows DATT to update only 0.8% of parameters during continuous operation, serving as a secondary architecture for specific deployment scenarios.

## V. EVALUATION AND ANALYSIS

This chapter presents a rigorous evaluation of the proposed Two-Level Behavioral Authentication System. Section V-A details the performance of the Evo-BioTCN for fixedtext authentication, analyzing the impact of PBT and feature stability. Section V-B evaluates the 4-Model Deep Learning Ensemble for free-text continuous monitoring, incorporating a critical investigation into transfer learning failures and a comprehensive ablation study of 18 architectures.

## A. Fixed-Text Evaluation (Level 1)

## 1) Performance Evaluation of the Final System:

a) Dataset and Protocol: The GREYC Keystroke Dataset [14] serves as the primary benchmark due to its longitudinal span of 110 days, which is essential for modeling behavioral drift. The dataset characteristics are summarized in Table V.

Evaluation Protocol: We employed a strict Temporal Split. For each user, the first 25% of sessions serve as enrolment (training), the next 15% for validation, and the final 60% – occurring weeks or months later – serve as the test set. This prevents “time-travel” data leakage, ensuring the model is tested on genuinely drifted behavior.

<table><tr><td rowspan=1 colspan=1>Property</td><td rowspan=1 colspan=1>Value</td><td rowspan=1 colspan=1>Research Significance</td></tr><tr><td rowspan=1 colspan=1>Total Samples</td><td rowspan=1 colspan=1>7,560 typing ses-sions</td><td rowspan=1 colspan=1>Enables statistically meaningful evaluation</td></tr><tr><td rowspan=1 colspan=1>Participants</td><td rowspan=1 colspan=1>133 individuals</td><td rowspan=1 colspan=1>Represents diverse user population</td></tr><tr><td rowspan=1 colspan=1>Password</td><td rowspan=1 colspan=1>“greyclaboratory&quot;(16 characters)</td><td rowspan=1 colspan=1>Fixed-text authentication scenario</td></tr><tr><td rowspan=1 colspan=1>Collection Period</td><td rowspan=1 colspan=1>110 days (March17 - July 5,2009)</td><td rowspan=1 colspan=1>Longest temporal span in keystroke literature</td></tr><tr><td rowspan=1 colspan=1>Sessions per User</td><td rowspan=1 colspan=1>Mean:  56.84,Range: varies byuser</td><td rowspan=1 colspan=1>Variable user engagement patterns</td></tr><tr><td rowspan=1 colspan=1>Raw Timing Features</td><td rowspan=1 colspan=1>71   measure-ments</td><td rowspan=1 colspan=1>Hold (H), Up-Down (UD), Down-Down(DD), Release-Release (RR)</td></tr><tr><td rowspan=1 colspan=1>Engineered Features</td><td rowspan=1 colspan=1>231 total</td><td rowspan=1 colspan=1>Comprehensive feature representation</td></tr><tr><td rowspan=1 colspan=1>Keyboard Types</td><td rowspan=1 colspan=1>2 different mod-els</td><td rowspan=1 colspan=1>Hardware diversity for generalization</td></tr><tr><td rowspan=1 colspan=1>Timestamp Availability</td><td rowspan=1 colspan=1>Complete</td><td rowspan=1 colspan=1>Essential for temporal drift analysis</td></tr></table>

TABLE VI  
FINAL SYSTEM PERFORMANCE METRICS
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>AchievedValue</td><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>Assessment</td></tr><tr><td rowspan=1 colspan=1>Equal Error Rate (EER)</td><td rowspan=1 colspan=1>3.77%</td><td rowspan=1 colspan=1>&lt;5%</td><td rowspan=1 colspan=1>Exceeds Target</td></tr><tr><td rowspan=1 colspan=1>TAR @ FAR=5%</td><td rowspan=1 colspan=1>96.87%</td><td rowspan=1 colspan=1>&gt;90%</td><td rowspan=1 colspan=1>Exceeds Target</td></tr><tr><td rowspan=1 colspan=1>Decidability Index (d&#x27;)</td><td rowspan=1 colspan=1>3.49</td><td rowspan=1 colspan=1>&gt;2.0</td><td rowspan=1 colspan=1>Exceeds Target</td></tr><tr><td rowspan=1 colspan=1>FRR @FAR=5%</td><td rowspan=1 colspan=1>3.13%</td><td rowspan=1 colspan=1>&lt;10%</td><td rowspan=1 colspan=1>Exceeds Target</td></tr></table>

b) Primary Performance Results: The Evo-BioTCN achieves performance metrics that exceed banking-grade security requirements (<5% EER).

ROC Curve Analysis: The Receiver Operating Characteristic (ROC) curve (figure 5) demonstrates a sharp ascent, indicating high sensitivity. The area under the curve (AUC) is 0.995, reflecting excellent discrimination between genuine users and impostors.

![](images/fdbf8fdd226881b73cd53e2029cb13f86566c21823262a67d438946dd793ef3d.jpg)  
Fig. 5. ROC curve for Evo-BioTCN showing excellent discrimination (AUC = 0.995).

Score Distribution: The distribution of genuine and impostor scores (figure 6) reveals a minimal overlap region. The decidability index d<sup>′</sup> = 3.49 confirms that the genuine and impostor distributions are separated by nearly 3.5 standard deviations.

![](images/ec1947964fe363bd756fd4f8eb404338b01a1fb5e6f3113f9b7628c102161d9f.jpg)  
Fig. 6. Score distribution showing minimal overlap between genuine and impostor scores.

2) Design Analysis: TCN vs. Transformer: To justify the architectural choice, we conducted a controlled comparison between the TCN and a Transformer-based equivalent (EvoBioTransformer).

While both models performed similarly before optimization, the Evo-BioTCN demonstrated superior adaptability during PBT. The TCN’s inductive bias toward local causality (capturing digraph/trigraph rhythms) aligns better with motor-neural typing patterns than the Transformer’s global attention mechanism. The improvement (table VII) is statistically significant (p = 0.0023, Wilcoxon signed-rank test).

TABLE VII  
POST-PBT ARCHITECTURE COMPARISON
<table><tr><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>EERPre-PBT</td><td rowspan=1 colspan=1>EERPost-PBT</td><td rowspan=1 colspan=1>Improve-ment</td><td rowspan=1 colspan=1>d’ Post-PBT</td><td rowspan=1 colspan=1>StatisticalSignificance</td></tr><tr><td rowspan=1 colspan=1>Transformer</td><td rowspan=1 colspan=1>8.26%</td><td rowspan=1 colspan=1>4.24%</td><td rowspan=1 colspan=1>-4.02 pp</td><td rowspan=1 colspan=1>3.21</td><td rowspan=1 colspan=1>Reference</td></tr><tr><td rowspan=1 colspan=1>TCN</td><td rowspan=1 colspan=1>8.09%</td><td rowspan=1 colspan=1>3.77%</td><td rowspan=1 colspan=1>-4.32 pp</td><td rowspan=1 colspan=1>3.49</td><td rowspan=1 colspan=1>p = 0.0023 vsTransformer</td></tr></table>

3) Feature Importance Analysis: Understanding what the model learns is crucial for explainability. We performed permutation importance analysis to rank the 231 engineered features as shown in table VIII.

## Analysis:

1) Temporal Context: The weekday feature ranks first, indicating that user typing behavior varies systematically by day of the week (e.g., Monday morning fatigue vs. Friday efficiency).

TABLE VIII TOP 10 MOST IMPORTANT FEATURES
<table><tr><td rowspan=1 colspan=1>Rank</td><td rowspan=1 colspan=1>Feature</td><td rowspan=1 colspan=1>Importance</td><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Password Position</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>weekday</td><td rowspan=1 colspan=1>0.0266</td><td rowspan=1 colspan=1>Temporalcontext</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>RR2</td><td rowspan=1 colspan=1>0.0164</td><td rowspan=1 colspan=1>Flight time</td><td rowspan=1 colspan=1> $\overline { { 2 \ ( \mathbf { r } \  \ \mathbf { e } ) } }$ </td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>UD2</td><td rowspan=1 colspan=1>0.0112</td><td rowspan=1 colspan=1>Flight time</td><td rowspan=1 colspan=1> $\overline { { 2 \ ( \mathbf { r } \  \ \mathbf { e } ) } }$ </td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1> $\overline { { { \bf { U D } } _ { 5 } } }$ </td><td rowspan=1 colspan=1>0.0108</td><td rowspan=1 colspan=1>Flight time</td><td rowspan=1 colspan=1> $\overline { { 5 \ ( \mathrm { c }  \mathrm { s p a c e } ) } }$ </td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1> $\overline { { \mathbf { R R } _ { 1 1 } } }$ </td><td rowspan=1 colspan=1>0.0098</td><td rowspan=1 colspan=1>Flight time</td><td rowspan=1 colspan=1>11 $\overline { { ( \mathrm { a } \xrightarrow { } ) \mathrm { t } ) } }$ </td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>hour</td><td rowspan=1 colspan=1>0.0088</td><td rowspan=1 colspan=1>Temporalcontext</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1> $\overline { { \mathrm { { U D } _ { 6 } } } }$ </td><td rowspan=1 colspan=1>0.0064</td><td rowspan=1 colspan=1>Flight time</td><td rowspan=1 colspan=1> $\overline { { 6 \ ( \mathrm { s p a c e }  1 ) } }$ </td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1> $\overline { { \mathrm { U D } _ { 1 1 } } }$ </td><td rowspan=1 colspan=1>0.0056</td><td rowspan=1 colspan=1>Flight time</td><td rowspan=1 colspan=1>11 $\overline { { \left( \mathrm { a } \xrightarrow { } \mathrm { t } \right) } }$ </td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>UD1</td><td rowspan=1 colspan=1>0.0046</td><td rowspan=1 colspan=1>Flight time</td><td rowspan=1 colspan=1>1 $\underline { { ( \mathbf { g }  \mathbf { r } ) } }$ </td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>H1_norm</td><td rowspan=1 colspan=1>0.0044</td><td rowspan=1 colspan=1>Hold time</td><td rowspan=1 colspan=1>1 (g)</td></tr></table>

TABLE IX

FEATURE CATEGORY DISTRIBUTION AMONG TOP 30 FEATURES
<table><tr><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Count</td><td rowspan=1 colspan=1>Proportion</td><td rowspan=1 colspan=1>Key Finding</td></tr><tr><td rowspan=1 colspan=1>Flight times (UD, RR, DD)</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>60%</td><td rowspan=1 colspan=1>Most discriminative category</td></tr><tr><td rowspan=1 colspan=1>Hold times (H)</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>27%</td><td rowspan=1 colspan=1>Secondary importance</td></tr><tr><td rowspan=1 colspan=1>Temporal metadata</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>10%</td><td rowspan=1 colspan=1>Context significantly contributes</td></tr><tr><td rowspan=1 colspan=1>Derived features</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>3%</td><td rowspan=1 colspan=1>Limited incremental contribution</td></tr></table>

2) Flight Times Dominate: Flight times (UD, RR) constitute 60% of the top 30 features (table IX). This confirms that the rhythm of movement between keys is more unique to an identity than the dwell time on keys.

3) Password Specificity: Features linked to specific transitions in “greyc laboratory” (e.g., RR 2 corresponding to $\mathbf { \dot { r } } _ { \mathbf { \Phi } } ^ { \prime } - \mathbf { \dot { e } } _ { \mathbf { \Phi } } ^ { \prime } )$ are highly discriminative.

4) Optimization Strategies: PBT Efficacy: We compared PBT against traditional optimization methods to validate its efficiency as table X portrays.

TABLE X  
COMPARISON OF OPTIMIZATION STRATEGIES
<table><tr><td rowspan=1 colspan=1>Strategy</td><td rowspan=1 colspan=1>Best EER</td><td rowspan=1 colspan=1>GPU Hours</td><td rowspan=1 colspan=1>Relative Efficiency</td></tr><tr><td rowspan=1 colspan=1>Manual Tuning</td><td rowspan=1 colspan=1>4.38%</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>Low (Suboptimal)</td></tr><tr><td rowspan=1 colspan=1>Grid Search</td><td rowspan=1 colspan=1>4.21%</td><td rowspan=1 colspan=1>3,000</td><td rowspan=1 colspan=1>Very Low (High cost)</td></tr><tr><td rowspan=1 colspan=1>Random Search</td><td rowspan=1 colspan=1>4.12%</td><td rowspan=1 colspan=1>2,500</td><td rowspan=1 colspan=1>Low</td></tr><tr><td rowspan=1 colspan=1>PBT (Proposed)</td><td rowspan=1 colspan=1>3.77%</td><td rowspan=1 colspan=1>1,000</td><td rowspan=1 colspan=1>High (Best)</td></tr></table>

PBT achieved the lowest EER with 1/3rd the computational cost of Grid Search. The trajectory of evolution shows a monotonic improvement in EER from 8.82% (Gen 0) to 3.77% (Gen 10), driven by the automatic adaptation of learning rates and the “survival of the fittest” selection of adapter weights.

## 5) Robustness and Stability Analysis:

Temporal Stability: A critical question is whether the system degrades over time. Analysis of the baseline (Scaled Manhattan) over a 90-day period (figure 7) shows stable EER (≈ $1 8 . 5 \% \pm 0 . 5 \% )$ , indicating that while raw data drifts, the identity signal persists. The Evo-BioTCN leverages this signal to maintain low error rates.

Enrolment Sensitivity: We analyzed how many sessions are needed for reliable enrolment (figure 8).

![](images/60060328ce455a4aa91cbe2be9682ee5b53463e276b252e43bd0b596ae25f1fd.jpg)  
Fig. 7. Temporal stability analysis showing consistent baseline EER over 90 days.

![](images/6eb935a79482ecfc1fa17c97df8028aa945089be8821bc4a12beaf569f238776.jpg)  
Fig. 8. Enrolment sensitivity analysis showing diminishing returns beyond 10 sessions.

• 1 Session: 23.98% Baseline EER.

• 10 Sessions: 15.59% Baseline EER.

• Conclusion: Diminishing returns are observed beyond 10 sessions. We recommend a 10-session enrolment protocol (approx. 5 minutes total) to balance usability and security.

6) Balanced vs Imbalanced Dataset Performance Metric Comparison:

![](images/09e3bacf9ba3c67bbd120ef1927b71661ad2065d9ddc8295e9de54c4dcd66b11.jpg)  
Fig. 9. Performance metrics under imbalanced vs balanced evaluation

![](images/4ebda1ec098add7d38f181fd01e3fd1e6503a12bb3e18a685fc18ede67d24e35.jpg)

![](images/d8f71640394ad2757527d82c8d779e41d864eca1f71c87d0cb52504d547ab71b.jpg)  
Fig. 10. Confusion matrix at FAR=5% operating point

TABLE XI  
CLASS IMBALANCE IMPACT ON METRICS
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Imbalanced (131:1)</td><td rowspan=1 colspan=1>Balanced (1:1)</td><td rowspan=1 colspan=1>Difference</td></tr><tr><td rowspan=1 colspan=1>TAR (Recall)</td><td rowspan=1 colspan=1>88.26%</td><td rowspan=1 colspan=1>88.26%</td><td rowspan=1 colspan=1>0.00%</td></tr><tr><td rowspan=1 colspan=1>FRR</td><td rowspan=1 colspan=1>11.74%</td><td rowspan=1 colspan=1>11.74%</td><td rowspan=1 colspan=1>0.00%</td></tr><tr><td rowspan=1 colspan=1>FAR</td><td rowspan=1 colspan=1>4.99%</td><td rowspan=1 colspan=1>4.99%</td><td rowspan=1 colspan=1>0.00%</td></tr><tr><td rowspan=1 colspan=1>TRR (Specificity)</td><td rowspan=1 colspan=1>95.01%</td><td rowspan=1 colspan=1>95.01%</td><td rowspan=1 colspan=1>0.00%</td></tr><tr><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>98.24%</td><td rowspan=1 colspan=1>91.63%</td><td rowspan=1 colspan=1>-6.61%</td></tr><tr><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>11.88%</td><td rowspan=1 colspan=1>94.64%</td><td rowspan=1 colspan=1>+82.76%</td></tr><tr><td rowspan=1 colspan=1>F1-Score</td><td rowspan=1 colspan=1>20.93%</td><td rowspan=1 colspan=1>91.41%</td><td rowspan=1 colspan=1>+70.48%</td></tr></table>

Figure 9 and the confusion matrices in figure 10 reveal a critical finding: class imbalance artificially inflates accuracy while masking poor precision. Under the imbalanced regime (131:1), the system achieves an impressive 98.24% accuracy, yet precision remains at only 11.88% – meaning that for every correct genuine user acceptance, the system generates over 7 false alarms. This occurs because the overwhelming majority class (impostors) dominates the accuracy calculation, while the rare positive predictions (genuine users) suffer from high false positive contamination.

The balanced evaluation reveals the true performance – accuracy drops to 91.63%, but precision surges to 94.64% (+82.76 pp improvement) as table XI shows. Critically, rate-based metrics (TAR, FAR, TRR) remain perfectly stable across both protocols, confirming that the model’s discriminative ability is unchanged – only the interpretation of aggregate metrics shifts. The confusion matrices illustrate this: at FAR ≈ 5%, the imbalanced dataset produces 30,100 false acceptances (4.99% of 602,734 impostor attempts), whereas the balanced set yields only 250 false acceptances (4.99% of 5,000 impostor attempts). This ∼120× reduction in false positive count directly explains the observed improvement in precision.

Implication for Real-World Deployment: These results demonstrate that traditional accuracy is a misleading metric for authentication systems. In production environments with natural class imbalance (where impostor attempts far outnumber genuine logins), practitioners must prioritize rate-based metrics (TAR, FAR) and use balanced evaluation sets for model selection. The F1-Score improvement from 20.93% to 91.41% under balanced evaluation confirms that threshold-dependent metrics should only be interpreted on class-balanced test sets to avoid overestimating operational performance.

## B. Free-Text Evaluation (Level 2)

1) Global Authentication Metrics: The 4-Model Deep Learning Ensemble achieves state-of-the-art results (table XII) on the Buffalo dataset [15], significantly outperforming singlemodel architectures as depicted in.

TABLE XII  
GLOBAL PERFORMANCE METRICS SUMMARY
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Value</td><td rowspan=1 colspan=1>Interpretation</td></tr><tr><td rowspan=1 colspan=1>Equal Error Rate (EER)</td><td rowspan=1 colspan=1>10.05%</td><td rowspan=1 colspan=1>65% reduction vs. Baseline (28.77%)</td></tr><tr><td rowspan=1 colspan=1>AUC-ROC</td><td rowspan=1 colspan=1>0.9518</td><td rowspan=1 colspan=1>Excellent discrimination capability</td></tr><tr><td rowspan=1 colspan=1>D-prime (d&#x27;)</td><td rowspan=1 colspan=1>2.45</td><td rowspan=1 colspan=1>Strong separation of score distributions</td></tr><tr><td rowspan=1 colspan=1>Global TAR @ FAR = 5%</td><td rowspan=1 colspan=1>83.53%</td><td rowspan=1 colspan=1>High usability for continuous monitor-ing</td></tr><tr><td rowspan=1 colspan=1>Global TRR @ FAR = 5%</td><td rowspan=1 colspan=1>95.00%</td><td rowspan=1 colspan=1>Very high true rejection rate for impos-tors</td></tr><tr><td rowspan=1 colspan=1>Global Balanced Accuracy</td><td rowspan=1 colspan=1>89.27%</td><td rowspan=1 colspan=1>Robust performance across genuine andimpostor classes</td></tr></table>

2) The KeyRecs [16] Investigation: A Critical Pivot: Before finalizing the architecture for Buffalo dataset [15], a preliminary investigation on the KeyRecs dataset [14] revealed a fundamental challenge: Transfer Learning Failure (table XIII). We tested whether the highly accurate fixed-text Evo-BioTCN could be applied to free-text data.

TABLE XIII  
TRANSFER LEARNING FAILURE
<table><tr><td rowspan=1 colspan=1>Pre-trainedModel</td><td rowspan=1 colspan=1>Fixed-TextPerformance</td><td rowspan=1 colspan=1>Free-TextTransfer</td><td rowspan=1 colspan=1>PerformanceDrop</td></tr><tr><td rowspan=1 colspan=1>Evo-BioTCN</td><td rowspan=1 colspan=1>≥96% accuracy</td><td rowspan=1 colspan=1>32.75%</td><td rowspan=1 colspan=1>-63 pp</td></tr><tr><td rowspan=1 colspan=1>DATT (fixed-text)</td><td rowspan=1 colspan=1>~95% accuracy</td><td rowspan=1 colspan=1>33.06%</td><td rowspan=1 colspan=1>-62 pp</td></tr><tr><td rowspan=1 colspan=1>Individual DL Mod-els</td><td rowspan=1 colspan=1>87.4% accuracy</td><td rowspan=1 colspan=1>10-15%</td><td rowspan=1 colspan=1>-72 to-77pp</td></tr></table>

## Findings:

1) Drift-Induced Degradation: Evaluation using a strict session-split protocol (training on Session 1, testing on Session 2) revealed an 8.62% accuracy drop compared to mixed-session evaluation (table XIV). This confirms that cross-session drift is a major source of error often hidden in standard evaluations.

TABLE XIV  
IMPACT OF EVALUATION PROTOCOL ON REPORTED ACCURACY
<table><tr><td rowspan=1 colspan=1>Evaluation Protocol</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Interpretation</td></tr><tr><td rowspan=1 colspan=1>Cross-Session (mixed train-ing)</td><td rowspan=1 colspan=1>95.03%</td><td rowspan=1 colspan=1>Artificially inflated due to exposure toSession 2 during training</td></tr><tr><td rowspan=1 colspan=1>Session-Split (drift-aware)</td><td rowspan=1 colspan=1>86.41%</td><td rowspan=1 colspan=1>Authentic performance under tempo-ral behavioral drift</td></tr><tr><td rowspan=1 colspan=1>Drift-Induced Degradation</td><td rowspan=1 colspan=1>8.62 pp</td><td rowspan=1 colspan=1>Real-world accuracy penalty causedby behavioral drift</td></tr></table>

2) Feature Distribution Shift: Fixed-text typing relies on motor memory (narrow variance), while free-text involves cognitive load (wide variance). Models trained on one do not generalize to the other.

3) Goldilocks Effect: Medium-complexity models outperformed both simple statistical models and massive deep networks on KeyRecs [14], motivating the ensemble approach for the final system as table XV shows.

TABLE XV  
ARCHITECTURE ABLATION REVEALING THE GOLDILOCKS EFFECT
<table><tr><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>Session-Split Accuracy</td><td rowspan=1 colspan=1>Complexity</td></tr><tr><td rowspan=1 colspan=1>DATT (Proposed)</td><td rowspan=1 colspan=1>53.25%</td><td rowspan=1 colspan=1>Medium (~500K params)</td></tr><tr><td rowspan=1 colspan=1>Simplified MLP</td><td rowspan=1 colspan=1>41.82%</td><td rowspan=1 colspan=1>Low</td></tr><tr><td rowspan=1 colspan=1>Deep Individual Models</td><td rowspan=1 colspan=1>30–35%</td><td rowspan=1 colspan=1>High (~1.5M params)</td></tr><tr><td rowspan=1 colspan=1>Random Forest</td><td rowspan=1 colspan=1>45.12%</td><td rowspan=1 colspan=1>Classical</td></tr></table>

3) Comprehensive Ablation Study: To construct the final Level 2 system, we conducted a massive ablation study comparing 18 different architectures (table XVI) ranging from classical ML to hybrid Deep Learning models.

TABLE XVI  
COMPREHENSIVE ABLATION STUDY RESULTS ACROSS ALL 18 EVALUATED MODELS
<table><tr><td rowspan=1 colspan=1>Rank</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>TAR @ 5%(%)</td><td rowspan=1 colspan=1>EER(%)</td><td rowspan=1 colspan=1>AUC</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>4-Model DL Ensemble</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>83.53</td><td rowspan=1 colspan=1>10.05</td><td rowspan=1 colspan=1>0.9518</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Enhanced Embedding</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>82.31</td><td rowspan=1 colspan=1>12.50</td><td rowspan=1 colspan=1>0.9304</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>ArcFace  +  Multi-Similarity</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>72.76</td><td rowspan=1 colspan=1>14.46</td><td rowspan=1 colspan=1>0.9146</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Ultra-Deep Embedding</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>67.72</td><td rowspan=1 colspan=1>15.77</td><td rowspan=1 colspan=1>0.9150</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Random Forest</td><td rowspan=1 colspan=1>MachineLearning</td><td rowspan=1 colspan=1>65.00</td><td rowspan=1 colspan=1>20.00</td><td rowspan=1 colspan=1>0.8500</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>CatBoost Tuned</td><td rowspan=1 colspan=1>MachineLearning</td><td rowspan=1 colspan=1>60.00</td><td rowspan=1 colspan=1>25.00</td><td rowspan=1 colspan=1>0.8000</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>Ultra-Deep 4-Layer</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>58.86</td><td rowspan=1 colspan=1>30.87</td><td rowspan=1 colspan=1>0.7200</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>Extra Trees</td><td rowspan=1 colspan=1>MachineLearning</td><td rowspan=1 colspan=1>58.00</td><td rowspan=1 colspan=1>24.00</td><td rowspan=1 colspan=1>0.7900</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>XGBoost</td><td rowspan=1 colspan=1>MachineLearning</td><td rowspan=1 colspan=1>55.00</td><td rowspan=1 colspan=1>28.00</td><td rowspan=1 colspan=1>0.7700</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>LightGBM</td><td rowspan=1 colspan=1>MachineLearning</td><td rowspan=1 colspan=1>52.00</td><td rowspan=1 colspan=1>29.00</td><td rowspan=1 colspan=1>0.7500</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>Mega       Ensemble(DL+ML)</td><td rowspan=1 colspan=1>Hybrid</td><td rowspan=1 colspan=1>50.00</td><td rowspan=1 colspan=1>26.00</td><td rowspan=1 colspan=1>0.7800</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>HMSSA-Net Hybrid</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>45.00</td><td rowspan=1 colspan=1>28.00</td><td rowspan=1 colspan=1>0.7500</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>HMSSA-Net</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>42.00</td><td rowspan=1 colspan=1>30.96</td><td rowspan=1 colspan=1>0.6900</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>DANN</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>40.00</td><td rowspan=1 colspan=1>28.00</td><td rowspan=1 colspan=1>0.7500</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>Meta-Learning MAML</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>35.00</td><td rowspan=1 colspan=1>35.00</td><td rowspan=1 colspan=1>0.6500</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>Evo-BioTransformer</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>20.65</td><td rowspan=1 colspan=1>31.35</td><td rowspan=1 colspan=1>0.7000</td></tr><tr><td rowspan=1 colspan=1>17</td><td rowspan=1 colspan=1>DriftNet</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>15.00</td><td rowspan=1 colspan=1>45.00</td><td rowspan=1 colspan=1>0.5500</td></tr><tr><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>HTAN</td><td rowspan=1 colspan=1>Deep Learning</td><td rowspan=1 colspan=1>1.33</td><td rowspan=1 colspan=1>56.58</td><td rowspan=1 colspan=1>0.4417</td></tr></table>

## Analysis of Ablation (Figure 11):

• Ensemble Superiority: The 4-Model Ensemble provides a 23% relative improvement in TAR over the Ultra-Deep Embedding network (Rank 4). This validates the “Distributed Complexity” hypothesis - multiple geometric perspectives (Triplet, ArcFace, Center Loss) are superior to a single deep network.

• Metric Learning vs. Classification: Models using metric learning losses (Ranks 1–4) significantly outperform classification-based models (e.g., BiLSTM), which struggle with the high intra-class variance of 75 users.

• ML vs. DL: While Random Forest (Rank 5) performed respectably, it hit a performance ceiling. Deep Learning models, when properly configured with metric loss, broke through this ceiling.

![](images/2559f9d4aa531c0e39400e2dfaa751c43b4894b73c71dde567a7b66deba5976f.jpg)  
Fig. 11. Ablation study showing performance across 18 architectures.

4) Complexity and Efficiency: We compared the ensemble against a single “Ultra-Deep” model to justify the design complexity with the results displayed in table XVII.

TABLE XVII  
ARCHITECTURAL COMPLEXITY COMPARISON
<table><tr><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>Parameters</td><td rowspan=1 colspan=1>TAR @ 5%FAR</td><td rowspan=1 colspan=1>EER(%)</td><td rowspan=1 colspan=1>ComplexityType</td></tr><tr><td rowspan=1 colspan=1>Ultra-Deep Embedding</td><td rowspan=1 colspan=1>~1.2M</td><td rowspan=1 colspan=1>67.72%</td><td rowspan=1 colspan=1>15.77</td><td rowspan=1 colspan=1>Concentrated</td></tr><tr><td rowspan=1 colspan=1>4-Model DL Ensemble</td><td rowspan=1 colspan=1>~3.2M</td><td rowspan=1 colspan=1>83.53%</td><td rowspan=1 colspan=1>10.05</td><td rowspan=1 colspan=1>Distributed</td></tr><tr><td rowspan=1 colspan=1>Improvement</td><td rowspan=1 colspan=1>+167%</td><td rowspan=1 colspan=1>+23%</td><td rowspan=1 colspan=1>36%</td><td rowspan=1 colspan=1></td></tr></table>

While the ensemble has more parameters, the Distributed Complexity ensures robustness. A failure in one embedding space (e.g., ArcFace) is often corrected by another (e.g., Triplet), providing a safety net that a single model cannot offer.

5) Statistical Validation: To ensure the results are not artifacts of data splitting, we performed bootstrap analysis (1,000 iterations). The results are outlined in table XVIII.

TABLE XVIII  
BOOTSTRAP CI FOR KEY PERFORMANCE METRICS
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Point Estimate</td><td rowspan=1 colspan=1>95% CI Lower</td><td rowspan=1 colspan=1>95% CI Upper</td></tr><tr><td rowspan=1 colspan=1>EER</td><td rowspan=1 colspan=1>10.05%</td><td rowspan=1 colspan=1>9.52%</td><td rowspan=1 colspan=1>10.61%</td></tr><tr><td rowspan=1 colspan=1>TAR @ FAR5%</td><td rowspan=1 colspan=1>83.53%</td><td rowspan=1 colspan=1>82.14%</td><td rowspan=1 colspan=1>84.89%</td></tr><tr><td rowspan=1 colspan=1>AUC-ROC</td><td rowspan=1 colspan=1>0.9518</td><td rowspan=1 colspan=1>0.9472</td><td rowspan=1 colspan=1>0.9561</td></tr><tr><td rowspan=1 colspan=1>Balanced Accuracy</td><td rowspan=1 colspan=1>89.27%</td><td rowspan=1 colspan=1>88.12%</td><td rowspan=1 colspan=1>90.38%</td></tr></table>

The narrow confidence intervals (e.g., EER width ≈ 1.1%) confirm the statistical reliability of the performance. Additionally, Effect Size Analysis shows a Cohen’s d = 2.89 when comparing the Ensemble to the Baseline, indicating a “Very Large” practical improvement.

6) Evaluation Under Class Imbalance: Real-world authentication is highly imbalanced (many impostors, few genuine users). We evaluated the system under both scenarios illustrated in table XIX.

TABLE XIX  
IMBALANCED VS BALANCED CLASSIFICATION METRICS
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Imbalanced (74:1)</td><td rowspan=1 colspan=1>Balanced (1:1)</td><td rowspan=1 colspan=1>Change</td></tr><tr><td rowspan=1 colspan=1>Rate-Based Metrics</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>TAR (Recall)</td><td rowspan=1 colspan=1>83.53%</td><td rowspan=1 colspan=1>83.53%</td><td rowspan=1 colspan=1>0.00%</td></tr><tr><td rowspan=1 colspan=1>FRR</td><td rowspan=1 colspan=1>16.47%</td><td rowspan=1 colspan=1>16.47%</td><td rowspan=1 colspan=1>0.00%</td></tr><tr><td rowspan=1 colspan=1>FAR</td><td rowspan=1 colspan=1>5.00%</td><td rowspan=1 colspan=1>5.00%</td><td rowspan=1 colspan=1>0.00%</td></tr><tr><td rowspan=1 colspan=1>TRR (Specificity)</td><td rowspan=1 colspan=1>95.00%</td><td rowspan=1 colspan=1>95.00%</td><td rowspan=1 colspan=1>0.00%</td></tr><tr><td rowspan=1 colspan=1>Count-Dependent Me</td><td rowspan=1 colspan=1>trics</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>94.85%</td><td rowspan=1 colspan=1>89.26%</td><td rowspan=1 colspan=1>-5.59%</td></tr><tr><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>18.42%</td><td rowspan=1 colspan=1>94.35%</td><td rowspan=1 colspan=1>+75.93%</td></tr><tr><td rowspan=1 colspan=1>F1-Score</td><td rowspan=1 colspan=1>30.18%</td><td rowspan=1 colspan=1>88.61%</td><td rowspan=1 colspan=1>+58.43%</td></tr></table>

This analysis confirms that TAR and EER are the authoritative metrics for deployment, as they remain stable regardless of the impostor-to-genuine ratio.

7) Balanced vs Imbalanced Dataset Performance Metric Comparison:

![](images/1a331f50b1a7291ccd7920120934e37991313cb81afce1a29fb7c936d0995dfe.jpg)  
Fig. 12. Imbalanced vs Balanced Metrics Comparison

![](images/8b3ce8f2651b4d3d2b9df83f96f8c41bd2c01db657b698b8b3ffcd1edb379efb.jpg)  
Fig. 13. Imbalanced vs Balanced evaluation confusion matrix

TABLE XX  
METRICS OF ORIGINAL VS BALANCED DATASET
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>OriginalDataset</td><td rowspan=1 colspan=1>StratifiedBalanced</td><td rowspan=1 colspan=1>Change</td></tr><tr><td rowspan=1 colspan=1>Dataset Size</td><td rowspan=1 colspan=1>2,912,625</td><td rowspan=1 colspan=1>77,670</td><td rowspan=1 colspan=1>-2,834,955</td></tr><tr><td rowspan=1 colspan=1>Ratio</td><td rowspan=1 colspan=1>74:1</td><td rowspan=1 colspan=1>1:1</td><td rowspan=1 colspan=1>Balanced</td></tr><tr><td rowspan=1 colspan=1>TAR</td><td rowspan=1 colspan=1>83.53%</td><td rowspan=1 colspan=1>83.53%</td><td rowspan=1 colspan=1>Same</td></tr><tr><td rowspan=1 colspan=1>FAR</td><td rowspan=1 colspan=1>5.00%</td><td rowspan=1 colspan=1>5.00%</td><td rowspan=1 colspan=1>Same</td></tr><tr><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>18.42%</td><td rowspan=1 colspan=1>94.35%</td><td rowspan=1 colspan=1>+75.93%</td></tr><tr><td rowspan=1 colspan=1>F1-Score</td><td rowspan=1 colspan=1>30.18%</td><td rowspan=1 colspan=1>88.61%</td><td rowspan=1 colspan=1>+58.43%</td></tr><tr><td rowspan=1 colspan=1>FP Count</td><td rowspan=1 colspan=1>143,690</td><td rowspan=1 colspan=1>1,942</td><td rowspan=1 colspan=1>-141,748</td></tr></table>

The comparative analysis in figure 12 and confusion matrices in figure 13 expose a fundamental methodological flaw in authentication research: evaluation protocol choice dramatically alters count-dependent metrics while leaving rate-based metrics invariant. Under the imbalanced protocol (74:1 ratio, 2.9M samples), the system reports 94.85% accuracy and 18.42% precision. Transitioning to balanced evaluation (1:1 ratio, 77,670 samples) reveals accuracy of 89.26% (−5.59 pp) but precision of 94.35% $( + 7 5 . 9 3 \ \mathrm { p p } ) \textrm { - a } 5 . 1 3 \times$ improvement.

This divergence stems from the mathematical structure of these metrics. Accuracy, precision, and F1-Score are count-dependent: they sum true positives, false positives, true negatives, and false negatives, making them sensitive to class distribution. The imbalanced confusion matrix in figure 13 shows 2,730,101 true negatives dominating the 2.9M total, artificially inflating accuracy to 94.85%. Conversely, the 143,690 false positives (5% of impostors) overwhelm the 32,439 true positives, crushing precision to 18.42%. The balanced matrix (right) eliminates this distortion: with equal genuine and impostor samples (38,835 each), the same 5% FAR yields only 1,942 false positives, allowing precision to reflect the model’s true discriminative quality (94.35%).

Critically, TAR (83.53%) and FAR (5.00%) remain identical across both protocols, as shown in Table XX. These rate-based metrics compute $\begin{array} { l c l } { \mathrm { T A R } } & { = } & { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } \end{array}$ and $\begin{array} { r } { \mathrm { F A R } ~ = ~ \frac { \mathrm { F P } } { \mathrm { F P + T N } } } \end{array}$ , evaluating each class independently without cross-class contamination. This stability confirms that the 74:1 imbalanced protocol does not improve model performance – it merely creates a reporting artifact where accuracy appears high due to trivial impostor rejection.

Methodological Recommendation: For authentication systems, researchers must report performance on balanced test sets or exclusively use rate-based metrics (EER, TAR@FAR). The F1-Score improvement from 30.18% (imbalanced) to 88.61% (balanced) demonstrates that threshold-dependent aggregate metrics become interpretable only when $P ( { \mathrm { g e n u i n e } } ) =$ $P ( \mathrm { i m p o s t o r } ) = 0 . 5$ . Production deployments should set operating points using ROC analysis on balanced data, then project expected false positive volumes using deploymentspecific class priors (e.g., $\mathrm { F P } _ { \mathrm { p r o d } } ~ = ~ \mathrm { F A R } ~ \times ~ N _ { \mathrm { i m p o s t o r , p r o d } } )$ This prevents the costly mistake of deploying a system that appears 94.85% accurate in testing but generates 74× more false alarms than precision suggests.

## C. Discussion

The evaluation confirms the hypothesis that behavioral drift can be managed through a two-level architecture.

1) Level 1 (Fixed-Text): The Evo-BioTCN utilizes PBT to evolve with the user, maintaining 3.77% EER over long durations.

2) Level 2 (Free-Text): The 4-Model Ensemble overcomes transfer learning failures by using specialized metric learning to achieve 10.05% EER.

3) Synergy: The combined system offers a theoretical joint attack success probability of $0 . 0 3 7 7 \times 0 . 1 0 0 5 \approx 0 . 3 8 \%$ providing a robust defense-in-depth solution.

## VI. CONCLUSION AND FUTURE DIRECTIONS

## A. Summary of Contributions

This research addresses the critical vulnerability of behavioral drift in biometric authentication – the phenomenon where user typing patterns evolve due to learning, fatigue, or environmental changes, rendering static templates obsolete. By moving beyond the “snapshot” view of authentication, this study establishes a Two-Level Behavioral Authentication Framework that treats user identity as a continuous evolutionary trajectory.

The primary contributions of this work are threefold:

1) Drift-Resilient Fixed-Text Architecture: The development of Evo-BioTCN, which utilizes a frozen TCN backbone to capture causal motor-neural dependencies, coupled with lightweight adapters evolved via PBT. This approach solves the “stability-plasticity dilemma,” allowing the model to adapt to new behaviors without catastrophic forgetting of the core identity.

2) Robust Free-Text Monitoring: The construction of a 4- Model Deep Learning Ensemble that overcomes the high variance of free-text typing. By fusing diverse geometric embeddings (ArcFace, Multi-Similarity, Triplet, Center Loss), the system achieves state-of-the-art performance on the Buffalo dataset [15], significantly outperforming single-model deep learning architectures.

3) Defense-in-Depth Integration: The synthesis of active login verification (Level 1) and passive continuous monitoring (Level 2) into a unified zero-trust framework. This integration lowers the theoretical joint attack success probability to below 0.4%, balancing high security with user convenience.

## B. Level 1 Findings: Fixed-Text Authentication

The evaluation of the Evo-BioTCN on the longitudinal GREYC dataset [14] (spanning 110 days) demonstrated that neural architectures can be engineered to handle long-term template aging effectively.

• Performance: The system achieved an EER of 3.77% and a TAR of 96.87% @ 5% FAR. This represents a 72.3% relative reduction in error compared to classical baselines like Scaled Manhattan Distance (13.60% EER).

• Architectural Superiority: The study confirmed that TCN is superior to Transformers for keystroke dynamics in this context. The TCN achieved a post-optimization EER of 3.77% compared to the Transformer’s 4.24% $( p = 0 . 0 0 2 3 )$ , validating that the TCN’s inductive bias for local causality aligns better with motor-neural typing patterns than global attention mechanisms.

• Drift Adaptation: The implementation of PBT was decisive. The evolutionary process reduced the EER from an initial 8.82% to the final 3.77% over 10 generations, proving that evolving lightweight adapters is a computationally efficient alternative to full model retraining.

## C. Level 2 Findings: Free-Text Continuous Monitoring

The free-text investigation highlighted the distinct challenges of variable-content typing, specifically the failure of transfer learning from fixed-text models.

• Transfer Learning Failure: A critical finding was that models pre-trained on fixed-text passwords collapse when applied to free-text domains (accuracy drop > 60%). This necessitated a purpose-built ensemble approach.

• Ensemble Performance: The 4-Model Deep Learning Ensemble achieved an EER of 10.05% and TAR of 83.53% @ 5% FAR on the Buffalo dataset [15]. This represents a 65% improvement over the strongest baseline and a 20% improvement over the best single deep learning model (Enhanced Embedding).

Complexity Justification: While the ensemble introduces higher computational complexity (∼3.2M parameters vs. ∼1.2M for a single Ultra-Deep model), the “Distributed Complexity” provided a 23% relative improvement in TAR. The ablation study confirmed that fusing complementary loss functions (e.g., Angular Margin + Triplet Mining) resolves ambiguities that single-geometry models cannot.

## D. Two-Level System Integration

The most significant operational outcome of this research is the synergy achieved by integrating Level 1 and Level 2. Table XXI summarizes this integration, demonstrating how the strengths of one level compensate for the weaknesses of the other.

TABLE XXI  
TWO-LEVEL KEYSTROKE AUTHENTICATION SYSTEM PERFORMANCE SUMMARY
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Level 1(Evo-BioTCN)</td><td rowspan=1 colspan=1>Level 2(4-Model Ens.)</td><td rowspan=1 colspan=1>Synergy</td></tr><tr><td rowspan=1 colspan=1>Authentication Type</td><td rowspan=1 colspan=1>Explicit Login</td><td rowspan=1 colspan=1>ContinuousMonitoring</td><td rowspan=1 colspan=1>Defense-in-Depth</td></tr><tr><td rowspan=1 colspan=1>EER</td><td rowspan=1 colspan=1>3.77%</td><td rowspan=1 colspan=1>10.05%</td><td rowspan=1 colspan=1>Sequential Filtering</td></tr><tr><td rowspan=1 colspan=1>TAR @ FAR=5%</td><td rowspan=1 colspan=1>96.87%</td><td rowspan=1 colspan=1>83.53%</td><td rowspan=1 colspan=1>Layered Confidence</td></tr><tr><td rowspan=1 colspan=1>Primary Error Mode</td><td rowspan=1 colspan=1>FRR (Usability)</td><td rowspan=1 colspan=1>FAR (Security)</td><td rowspan=1 colspan=1>Error Diversity</td></tr><tr><td rowspan=1 colspan=1>Temporal Resilience</td><td rowspan=1 colspan=1>PBT Adaptation</td><td rowspan=1 colspan=1>Score-Level Fusion</td><td rowspan=1 colspan=1>Multi-scale  drifthandling</td></tr></table>

Joint Probability Analysis: Assuming independence between the distinct tasks of password entry (Level 1) and continuous free-text typing (Level 2), the probability of a successful impostor attack P(Success) is the product of the failure rates of both levels formulated as (13) and (14) show:

$$
P ( { \mathrm { S u c c e s s } } ) \approx P ( { \mathrm { L e v e l ~ 1 ~ F a i l u r e } } ) \times P ( { \mathrm { L e v e l ~ 2 ~ F a i l u r e } } ) ( 1 3 )
$$

$$
P ( { \mathrm { S u c c e s s } } ) \approx 0 . 0 3 7 7 \times 0 . 1 0 0 5 \approx 0 . 0 0 3 7 8 \ ( 0 . 3 8 \% )\tag{14}
$$

This theoretical reliability of 99.62% demonstrates that while continuous monitoring (Level 2) is inherently more difficult than password verification, its deployment as a secondary layer creates a formidable security barrier.

## E. Limitations

Despite these successes, the study operates under specific constraints that must be acknowledged:

1) Dataset Scope: The evaluation relies on the GREYC [14] and Buffalo [15] datasets. While GREYC [14] is longitudinal, it contains only two keyboard types, limiting the assessment of cross-device hardware variability (e.g., switching from mechanical to membrane keyboards).

2) Population Size: The free-text evaluation is limited to 75 users. While sufficient for statistical significance, it may not capture the full tail-end diversity of global typing behaviors.

3) Adversarial Assumptions: The evaluation assumes zero-effort impostors (random attacks). Targeted adversarial attacks, where an impostor uses generative AI to mimic a specific user’s rhythm, were not empirically tested.

4) Computational Cost: The 4-Model Ensemble requires running four parallel deep networks, which may pose latency challenges on resource-constrained edge devices compared to single lightweight models.

## F. Future Directions

Future research should focus on enhancing scalability and robustness through the following avenues:

1) Multimodal Fusion: Integrating keystroke dynamics with mouse dynamics or touchscreen gestures could further lower the EER. Multimodal fusion can compensate for degradation in a single channel (e.g., if a user injures a hand, mouse dynamics might remain stable while typing degrades).

2) Privacy-Preserving Federated Learning: To address privacy concerns and scalability, future iterations should explore Federated Learning. This would allow the Evo-BioTCN adapters to be updated locally on user devices without transmitting sensitive timing vectors to a central server. Only the weight updates would be shared, preserving user privacy while allowing the global model to learn from diverse drift patterns.

3) Personalized Meta-Learning: The current system requires an enrolment phase. Implementing Meta-Learning (e.g., Model-Agnostic Meta-Learning or MAML) could enable the system to “learn to learn,” allowing it to adapt to a new user with only a few keystroke samples (fewshot learning), significantly improving the onboarding experience.

## G. Conclusion

This research proves that behavioral biometrics need not be brittle. By shifting the paradigm from static template matching to continuous evolutionary adaptation, we have demonstrated a security framework that is both highly accurate and resilient to the natural passage of time. The Evo-BioTCN and 4-Model Ensemble together provide a blueprint for the next generation of “living” authentication systems that evolve alongside their users.

## ACKNOWLEDGMENT

Our heartfelt praise goes to the Almighty Allah who has led us all the way to the completion of this report.

Our deepest gratitude goes to Dr. Muhammad Iqbal Hossain, whose unwavering guidance, support, and expertise were instrumental in the completion of this report. His insightful feedback and continuous encouragement have been a source of motivation throughout our journey of completing this report. We are truly fortunate to have had the opportunity to work under his supervision, and we are grateful for the time and effort he invested in helping us refine our ideas.

The legion support and enormous solace we received from our family and friends mark a remarkable contribution to pursuing this journey. Their belief in us has been a constant source of strength, and without them, this report would not have been possible.

## REFERENCES

[1] H. Fereidooni, J. Konig, P. Rieger, M. Chilese, B. G¨ okbakan, M.¨ Finke, A. Dmitrienko, and A.-R. Sadeghi, “AuthentiSense: A scalable behavioral biometrics authentication scheme using few-shot learning for mobile platforms,” arXiv preprint arXiv:2302.02740, 2023.

[2] E. A. Sagbas and S. Balli, “Machine learning-based novel continuous authentication system using soft keyboard typing behavior and motion sensor data,” Neural Computing and Applications, vol. 36, no. 10, pp. 5433–5445, 2024.

[3] V. M. Patel, R. Chellappa, D. Chandra, and B. Barbello, “Continuous user authentication on mobile devices: Recent progress and remaining challenges,” IEEE Signal Processing Magazine, vol. 33, no. 4, pp. 49– 61, 2016.

[4] R. Ryu, S. Yeom, D. Herbert, and J. Dermoudy, “The design and evaluation of adaptive biometric authentication systems: Current status, challenges and future direction,” ICT Express, vol. 9, no. 6, pp. 1183– 1197, 2023.

[5] A. F. Baig and S. Eskeland, “Security, privacy, and usability in continuous authentication: A survey,” Sensors, vol. 21, no. 17, p. 5967, 2021.

[6] J. Solano, E. Rivera, L. Tengana, C. Lopez, J. Fl ´ orez, and M. Ochoa,´ “A Siamese neural network for scalable behavioral biometrics authentication,” in Lecture Notes in Computer Science, 2022, pp. 515–535.

[7] F. S. Chen, S. Rao, B. Tiwari, and V. V. Phoha, “DSTER: A dualstream transformer-based emotion recognition model through keystrokes dynamics,” in Proc. IEEE Int. Joint Conf. Biometrics (IJCB), 2024, pp. 1–10.

[8] A. Subash, I. Song, I. Lee, and K. Lee, “Adaptability of current keystroke and mouse behavioral biometric systems: A survey,” Computers & Security, vol. 160, p. 104731, 2026.

[9] D. Senarath, S. Tharinda, M. Vishvajith, S. Rasnayaka, S. Wickramanayake, and D. Meedeniya, “BehaveFormer: A framework with spatio-temporal dual attention transformers for IMU-enhanced keystroke dynamics,” in Proc. IEEE Int. Joint Conf. Biometrics (IJCB), 2023, pp. 1–9.

[10] K. Lis, E. Niewiadomska-Szynkiewicz, and K. Dziewulska, “Siamese neural network for keystroke dynamics-based authentication on partial passwords,” Sensors, vol. 23, no. 15, p. 6685, 2023.

[11] X. Wang, Y. Shi, K. Zheng, Y. Zhang, W. Hong, and S. Cao, “User authentication method based on keystroke dynamics and mouse dynamics with scene-irrelated features in hybrid scenes,” Sensors, vol. 22, no. 17, p. 6627, 2022.

[12] P. Ranganathan, “Hybrid variational autoencoders and graph neural networks for behavioural biometric authentication,” Int. J. Comput. Appl. Eng. Sci., vol. 18, no. 2, pp. 1–10, 2022.

[13] K. Mazumdar and S. Sundaram, “A joint patch-based Siamese CNN with Vision Transformer framework for continuous mouse dynamics user authentication using Gramian angular summation field,” IEEE Trans. Biometrics, Behavior, and Identity Science, vol. 1, no. 1, pp. 1–10, 2025.

[14] R. Giot, M. El-Abed, and C. Rosenberger, “GREYC keystroke: A benchmark for keystroke dynamics biometric systems,” in Proc. IEEE Int. Conf. Biometrics: Theory, Applications and Systems (BTAS), 2009, pp. 1–6.

[15] Y. Sun, H. Ceker, and S. Upadhyaya, “Shared keystroke dataset for continuous authentication,” in Proc. IEEE Int. Workshop Information Forensics and Security (WIFS), Abu Dhabi, 2016, pp. 1–6.

[16] T. Dias, J. Vitorino, E. Maia, O. Sousa, and I. Prac¸a, “KeyRecs: Keystroke dynamics dataset,” Zenodo, May 2023. [Online]. Available: https://doi.org/10.5281/zenodo.7886743

[17] D. Bagayoko, “The law of human performance or of practice: An extension of the power law of performance or of practice,” Timbuktu Academy, Southern Univ. and A&M College, Baton Rouge, LA, USA, 2010.