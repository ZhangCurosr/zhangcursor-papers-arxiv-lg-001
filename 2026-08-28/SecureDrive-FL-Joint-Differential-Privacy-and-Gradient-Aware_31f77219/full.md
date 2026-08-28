# SecureDrive-FL: Joint Differential Privacy and Gradient-Aware Selective Homomorphic Encryption for Federated Driver Monitoring

Baran Can Gul¨ <sup>∗</sup>, Hanuma Siddhartha Tunuguntla<sup>†</sup>, Anjana Arvind Naik<sup>∗</sup>, Abhishek Vijay Potekar<sup>∗</sup>, Nasser Jazdi<sup>∗</sup>, and Michael Weyrich<sup>∗</sup>

<sup>∗</sup>Institute of Industrial Automation and Software Engineering, University of Stuttgart, Stuttgart, Germany

<sup>†</sup>EngD Automotive Systems Design, Eindhoven University of Technology (TU/e), Eindhoven, The Netherlands

Email: baran-can.guel@ias.uni-stuttgart.de, h.s.tunuguntla@tue.nl, st189825@stud.uni-stuttgart.de,

st191608@stud.uni-stuttgart.de, nasser.jazdi@ias.uni-stuttgart.de, michael.weyrich@ias.uni-stuttgart.de

Abstract—Federated Learning (FL) enables privacy-aware distributed training, yet gradient updates remain exploitable: Manin-the-Middle (MitM) interception exposes updates in transit, while model poisoning corrupts global convergence. We first introduce GASHE (Gradient-Aware Selective Homomorphic Encryption), a novel selective encryption strategy that dynamically identifies and encrypts only the gradient components exceeding a DP-calibrated sensitivity threshold, rather than encrypting all parameters uniformly as in static layer-based or full-parameter CKKS schemes. Building on GASHE, we introduce SecureDrive-FL, a federated driver monitoring framework that couples DP-SGD with GASHE to create the first closed-loop DP + HE privacy pipeline: DP-SGD calibration parameters directly derive the GASHE encryption mask, unifying training-time privacy and communication-time confidentiality. Evaluated on a ten-class distracted driver classification task under non-IID federated splits, SecureDrive-FL matches DP-SGD alone’s poisoning resistance (73.6% vs. 74.0% accuracy, 3.9% Attack Success Rate for both) while additionally withstanding MitM interception, where DP-SGD alone collapses to near-random accuracy (78.2% vs. 10.4%), all under only ≈8–10% additional runtime overhead relative to DP-SGD alone—under DP-SGD noise injection with per-round privacy parameter $\varepsilon _ { 0 } { = } 4 .$

Index Terms—Federated Learning, Differential Privacy, Homomorphic Encryption, Driver Monitoring, Privacy-Preserving Machine Learning, Adversarial Robustness, Connected Vehicles

## I. INTRODUCTION

The rapid growth of Internet of Things (IoT) devices and connected vehicles has led to the generation of large-scale distributed datasets covering driver behavior analysis, intelligent transportation systems, and autonomous driving [1]. Federated Learning (FL) has emerged as a compelling paradigm to exploit these distributed datasets without centralizing sensitive raw data: local models are trained on edge devices and only gradient updates are transmitted to a central aggregator [2]. Yet, FL provides no formal privacy guarantee by itself. A growing body of evidence shows that adversaries can reconstruct sensitive information from gradient updates through model inversion [3], membership inference [4], GAN-based reconstruction [5], and deep gradient leakage [6]. The threat surface is further expanded in automotive deployments, where model updates traverse shared cloud infrastructure operated by OEMs, insurers, and third-party service providers, each representing a potential interception point [7].

Two primary families of Privacy-Enhancing Technologies (PETs) have been studied independently for FL. Fully Homomorphic Encryption (FHE) allows the server to aggregate encrypted updates without ever decrypting them [8], but encrypting every model parameter incurs prohibitive memory and latency overhead on resource-constrained edge devices [9]. Differential Privacy (DP), realized via DP-SGD [10], provides rigorous, quantifiable privacy guarantees by injecting calibrated Gaussian noise into gradient updates; however, the well-known privacy-utility trade-off means that high privacy levels substantially degrade model accuracy [11].

Prior hybrid works [12]–[14] treat DP and HE as largely independent, orthogonally applied modules. In contrast, our key insight is to derive the encryption decision directly from the DP-SGD calibration itself: the clipping norm C and noise scale σ used in DP-SGD already identify which gradients dominate the DP noise floor and are therefore most informative, and most dangerous if transmitted in plaintext. This DPinformed gradient sensitivity threshold is used as the GASHE mask, creating a principled, closed-loop coupling between training-time privacy and communication-time confidentiality. The specific contributions of this work are:

GASHE: We introduce Gradient-Aware Selective Homomorphic Encryption, a novel, architecture-agnostic selective encryption strategy that dynamically targets CKKS encryption to the k gradient components whose magnitudes exceed the DP-calibrated threshold $\tau _ { i }$ , retaining formal privacy semantics without encrypting the full parameter vector. GASHE can be deployed as a standalone communication-security primitive, independently of DP-SGD.

• DP-GASHE coupling (SecureDrive-FL): To the best of our knowledge, SecureDrive-FL is the first framework to explicitly derive HE encryption decisions from DP-SGD sensitivity bounds, creating a unified privacy pipeline with DP-SGD noise injection at training time (per-round ε =4, σ=1.2112, C=1.2, δ=10<sup>−5</sup>) and IND-CPA confidentiality at communication time (Theorem 1).

• Multi-threat robustness evaluation: We evaluate SecureDrive-FL against model poisoning and MitM gradient interception, showing that the joint DP+GASHE design closes the MitM attack surface while matching DP-SGD alone’s poisoning resistance. Under poisoning, SecureDrive-FL attains accuracy (73.6% vs. 74.0%), Macro-F1 (73.3% vs. 73.7%), and Attack Success Rate (3.9% vs. 3.9%) comparable to DP-SGD alone, with a marginal Macro-Precision gain (75.9% vs. 75.7%), while additionally preventing the accuracy collapse DP-SGD alone suffers under MitM (78.2% vs. 10.4%). Undefended FedAvg and FL+GASHE-only, by contrast, both collapse under poisoning (≈10–14% accuracy), so their nominally low ASR reflects training failure, not robustness.

• Automotive use-case validation: We demonstrate practical feasibility on a ten-class distracted driver recognition task with non-IID, user-partitioned FL and lightweight MobileNetV2-Tiny, achieving ≈74% accuracy under active attack with DP-SGD configured at per-round ε<sub>0</sub>=4.

The remainder of the paper is organized as follows: Section II reviews related work, Section III presents the SecureDrive-FL framework, and Section IV details the threat model and our adversarial evaluation methodology. Section V describes the implementation and experimental results and Section VI provides the summary of this work.

## II. RELATED WORK

We review privacy threats in FL, DP and HE mechanisms, and hybrid robustness-oriented frameworks, then position SecureDrive-FL against their limitations.

## A. Privacy Threats in Federated Learning

Standard FL is vulnerable to a range of inference and interception attacks that exploit the shared model updates exchanged between clients and the server. Model inversion attacks demonstrated in [3] exploit model confidence scores to reconstruct sensitive input attributes, while membership inference attacks introduced in [4] reveal whether specific records were part of the training set. GAN-based information leakage in collaborative learning is analyzed in [5], and unintended feature leakage through shared representations is examined in [15]. Deep gradient leakage is investigated in [6], showing that raw training samples can often be reconstructed directly from gradient updates under standard FL. More recent analyses of gradient leakage in federated environments are provided in [16]. These works collectively establish that data locality alone is insufficient: exchanged gradients remain an exploitable attack surface that can be targeted both passively (eavesdropping) and actively (manipulation).

## B. Differential Privacy in Federated Learning

Differential Privacy provides a formal framework for bounding the information that individual records contribute to model outputs [17]. The DP-SGD mechanism proposed in [10] realizes this in deep learning via per-example gradient clipping and calibrated Gaussian noise injection, achieving formal (ε, δ)-DP guarantees. Its adaptation to the federated setting is studied in [18], where client-level DP is enforced across communication rounds, and a client-level perspective with adaptive clipping is developed in [11] to reduce utility degradation. Tighter privacy accounting using Renyi DP is introduced´ in [19] and extended in [20], enabling more efficient budget management over many rounds.

Despite these advances, a key limitation remains: as shown in [6], gradient inversion can partially succeed even in the presence of DP noise when the signal-to-noise ratio is favorable, particularly for small batch sizes. Furthermore, DP-SGD operates only during local training and provides no confidentiality guarantee for the gradient updates transmitted over the network, leaving them exposed to Man-in-the-Middle interception.

## C. Homomorphic Encryption in Federated Learning

Fully Homomorphic Encryption, initially constructed in [8], allows computation on encrypted data and has been adopted for secure aggregation in FL. BatchCrypt [13] reduces HE overhead via quantization and batching; secret-sharing-based secure aggregation [21] enables servers to recover only the aggregate; and TenSEAL [22] with [14] bring CKKS-based encryption to general ML and IoT FL pipelines respectively.

However, all of the above approaches suffer from a critical practical limitation: they encrypt all model parameters uniformly, incurring computational and memory overhead that is prohibitive on resource-constrained edge devices. Partial or layer-based encryption schemes attempt to reduce this cost by encrypting only selected layers, but they rely on static, architecture-dependent rules that do not adapt to the actual privacy sensitivity of individual gradient components during training.

## D. Hybrid Privacy Frameworks and Robustness

A comprehensive survey of FL attacks and defenses is provided in [23], and strong multi-round poisoning attacks circumventing many existing defenses are introduced in [24]. Hybrid DP–HE approaches [12] apply noise-based privacy and encrypted aggregation as independently configured modules, without directing encryption toward the most sensitive gradients. A representative hybrid defense [25] combines DP and SMC to improve accuracy over single-mechanism baselines, but DP and SMC are configured independently: the DP calibration does not inform which updates SMC protects, and robustness to poisoning is not evaluated. SecureDrive-FL instead derives the HE mask directly from the DP-SGD clipping norm C and noise scale σ, and evaluates the resulting pipeline against both inference and poisoning adversaries in a realistic automotive FL setting.

## III. SECUREDRIVE-FL FRAMEWORK

We describe the overall design of SecureDrive-FL, detailing the system model, threat assumptions, and the integration of

client-side DP-SGD with Gradient-Aware Selective Homomorphic Encryption for secure and efficient aggregation.

## A. System Overview

SecureDrive-FL integrates two orthogonal privacy mechanisms applied at different stages of the FL pipeline, as illustrated in Fig. 1: (i) DP-SGD during local model training to ensure formal statistical indistinguishability of individual client contributions in the final model, and (ii) GASHE during gradient communication to prevent raw gradient exposure under honest-but-curious or compromised server scenarios.

## B. Threat Model

We target an honest-but-curious server: the aggregator follows the FL protocol faithfully but attempts to infer private information from received model updates, intermediate aggregation states, and server-side logs. This threat model is highly relevant in the automotive domain, where aggregation servers are typically operated by third-party cloud providers or shared OEM infrastructure. We additionally evaluate active adversaries that inject malicious updates (model poisoning) and network-layer eavesdroppers that intercept gradient transmissions (MitM attacks); see Section IV for the full adversarial evaluation. Note that, we do not target Byzantine-tolerant aggregation or Sybil attacks in the core protocol design, as robust aggregation is orthogonal to the confidentiality guarantees of GASHE and is left as future work.

## C. Data Collection and Preprocessing

Each participating vehicle (client i) passively collects sensor data $D _ { i }$ during operation. In accordance with FL principles, $D _ { i }$ is never transmitted. Each raw sample $x _ { i } ^ { ( t ) }$ is locally preprocessed to extract relevant features:

$$
z _ { i } ^ { ( t ) } = \mathcal { P } ( x _ { i } ^ { ( t ) } ) ,\tag{1}
$$

and $D _ { i }$ is partitioned into training and validation subsets:

$$
D _ { i } = D _ { i } ^ { \mathrm { t r a i n } } \cup D _ { i } ^ { \mathrm { e v a l } } , \quad D _ { i } ^ { \mathrm { t r a i n } } \cap D _ { i } ^ { \mathrm { e v a l } } = \emptyset .\tag{2}
$$

## D. Local Training with DP-SGD

The local model $f _ { i } ( \cdot ; W _ { i } )$ is trained using DP-SGD. For each mini-batch, per-sample gradients are clipped to $L _ { 2 }$ -norm C, and calibrated Gaussian noise is added:

$$
W _ { i } \ :  \ : W _ { i } - \eta \big ( \nabla _ { W _ { i } } \mathcal { L } \ : + \ : \mathcal { N } \big ( 0 , \sigma ^ { 2 } C ^ { 2 } I \big ) \big )\tag{3}
$$

where C is the $\ell _ { 2 }$ clipping norm and σ is the noise multiplier; the product σC is the standard deviation of the injected noise, consistent with Algorithm 1.

## E. GASHE: Gradient-Aware Selective Homomorphic Encryption

After local training, client i computes the model update $\Delta w _ { i } = W _ { i } - W ^ { \mathrm { g l o b a l } }$ and derives a DP-informed sensitivity threshold:

$$
\tau _ { i } = \frac { C \sqrt { 2 \ln ( 1 . 2 5 / \delta ) } } { \varepsilon _ { 0 } }\tag{4}
$$

where $\varepsilon _ { \mathrm { 0 } }$ is the per-round noise calibration parameter, $\delta$ is the failure probability, and C is the clipping norm from DP-SGD.

This threshold equals the minimum noise standard deviation prescribed by the closed-form Gaussian mechanism for the configured $( \varepsilon , \delta )$ budget [26]; gradient components whose magnitudes exceed $\tau _ { i }$ dominate the $\mathrm { D P }$ noise floor and constitute the most privacy-sensitive parameters when transmitted in plaintext. In this configuration, $\tau _ { i }$ is set to coincide with the operational noise standard deviation $\sigma C$ (calibrated via Renyi´ DP accounting), so that a single calibration input $\varepsilon _ { \mathrm { 0 } }$ jointly determines the DP noise scale and the GASHE threshold.

A binary mask $M _ { i }$ is then derived from the threshold $\tau _ { i } ,$ selecting the gradient components whose magnitudes dominate the DP noise floor:

$$
M _ { i } [ j ] \ = \ \mathbf { 1 } \big [ \big | \Delta w _ { i } [ j ] \big | \ \ge \ \tau _ { i } \big ]\tag{5}
$$

Only the masked subset is encrypted under the server’s public key, while the remaining low-magnitude parameters are transmitted in plaintext alongside a compact bitmask:

$$
c _ { i } = { \mathrm { F H E . E n c } } ( M _ { i } \odot \Delta w _ { i } ; ~ p k ) .\tag{6}
$$

This selective strategy directly reduces the fraction of encrypted parameters, cutting both ciphertext size and FHE evaluation time proportionally to the sparsity of $M _ { i }$

## Theorem 1 (SecureDrive-FL Privacy and Security Guarantee).

Let each client i execute DP-SGD for T local steps per round across N communication rounds, with per-sample gradient clipping norm C, Gaussian noise multiplier $\sigma ,$ and minibatch sampling rate $q = B / n$ over a local dataset of size n. Then the local training mechanism satisfies $( \varepsilon , \delta ) – \mathrm { D P }$ with

$$
\varepsilon _ { \mathrm { t o t a l } } = { \cal O } \left( q \frac { \sqrt { T N \ln ( 1 / \delta ) } } { \sigma } \right)\tag{7}
$$

via Renyi DP composition [19], [20], where ´ $\begin{array} { l l l } { q } & { = } & { B / n } \end{array}$ is the per-round sampling rate, T is the number of local gradient steps per round, and N is the number of communication rounds. For the configuration in Section IV (C=1.2, $\sigma { = } 1 . 2 1 1 2 , ~ N { = } 1 2 0 , ~ \delta { = } 1 0 ^ { - 5 } )$ , the Renyi DP accountant re-´ ports $\varepsilon _ { \mathrm { t o t a l } } { \approx } 4 8 . 9 4$

## F. Homomorphic Aggregation

The server aggregates all encrypted updates using the CKKS additive homomorphic operation:

$$
\begin{array} { r } { C _ { \mathrm { a g g ~ } } = \boxplus _ { i \in \mathcal { C } _ { n } } c _ { i } \ \equiv \ \mathsf { F H E . E n c } \big ( \sum _ { i \in \mathcal { C } _ { n } } M _ { i } \odot \Delta w _ { i } ; \ p k \big ) } \end{array}\tag{8}
$$

where ${ \mathcal { C } } _ { n }$ is the set of selected clients in round $n .$ . The server never decrypts individual client updates, preserving privacy even under insider threats. Since client masks $M _ { i }$ may differ, each reflecting the local gradient magnitude distribution, the element-wise sum implicitly applies zero for components below threshold in each client; CKKS handles these as exact zeros with no additional ciphertext overhead. The global model update is recovered by clients:

![](images/b7d9d507e4c15ac1136f4bb75f0712dc9e9e2cf325989a75f17aa3a4cc2fe6db.jpg)  
Fig. 1. SecureDrive-FL overview. DP-SGD protects privacy during local model training, while GASHE selectively encrypts high-sensitivity gradient components before transmission to the aggregation server.

$$
\Delta W = \mathrm { F H E . D e c } ( C _ { \mathrm { a g g } } ) , \quad W \gets W + \frac { 1 } { | { \mathcal C } _ { n } | } \Delta W .\tag{9}
$$

Each client also receives the server-side element-wise sum of plaintext low-magnitude components, $\begin{array} { r } { P _ { \mathrm { a g g } } ~ = ~ \sum _ { i } ( \mathbf { 1 } ~ - } \end{array}$ $M _ { i } ) \odot \Delta w _ { i }$ , transmitted without encryption. The full aggregate gradient is recovered as

$$
\begin{array} { r l } & { \Delta W _ { \mathrm { f u l l } } = \mathsf { F H E . D e c } ( C _ { \mathrm { a g g } } ; \ s k ) + P _ { \mathrm { a g g } } , } \\ & { \qquad W \gets W + \frac { 1 } { | \mathcal { C } _ { n } | } \Delta W _ { \mathrm { f u l l } } } \end{array}\tag{10}
$$

This two-stream design ensures that all d gradient dimensions contribute to the global update, while only the k high-sensitivity components incur FHE overhead. The full SecureDrive-FL protocol is given in Algorithm 1.

## IV. SYSTEM AND ADVERSARIAL EVALUATION SETUP

In this section, we give the details of the system configuration, datasets, models, and attack/defence setups used to evaluate SecureDrive-FL.

## A. Use-Case and Dataset

We evaluate on a ten-class distracted-driver classification task combining the State Farm Distracted Driver Detection dataset [27] and the AUC Distracted Driver dataset [28], yielding over 50,000 labelled images of naturalistic driving across ten distraction categories: safe driving, texting (right/left hand), phone call (right/left hand), radio operation, drinking, reaching behind, hair/makeup adjustment, and talking to a passenger.

To emulate realistic federated data locality, we adopt a user-based partition where each client holds images from 3–5 unique drivers, inducing non-IID heterogeneity representative of real-world driver diversity. Labels are one-hot encoded; images are resized to $2 2 4 \times 2 2 4$ , normalised, and augmented locally via random horizontal flips and brightness jitter. The automotive context motivates strict GDPR compliance [29], as driver behavioural data is classified as sensitive personal data.

## B. Model and Federated Learning Configuration

We use MobileNetV2-Tiny as the convolutional backbone for its favourable accuracy-to-compute trade-off on embedded platforms. Feature maps are reduced via global average pooling and passed to a softmax classifier ( ≈ 2.2M parameters). We run 120 FL rounds with six clients per round and a local batch size of 32.

All DP-enabled runs use DP-SGD with clipping norm C=1.2, noise multiplier $\sigma { = } 1 . 2 1 1 2$ (derived from a singleshot calibration input $\varepsilon _ { 0 } { = } 4 )$ , and $\delta \mathrm { { = } 1 0 ^ { - 5 } }$ . Privacy accounting via Renyi DP composition [19] over´ $N { = } 1 2 0$ rounds yields a cumulative budget of $\varepsilon _ { \mathrm { t o t a l } } { \approx } 4 8 . 9 4$ , consistent with the known growth of composed DP guarantees across FL rounds [18].

## C. Software and Hardware Environment

We instantiated DP-SGD via TensorFlow Privacy [30] using DPKerasSGDOptimizer and Renyi DP accounting. Homo-´ morphic operations use TenSEAL [22] with CKKS configured as: polynomial modulus degree 8192, scaling factor $2 ^ { 4 0 ^ { \circ } }$ , and coefficient modulus chain [60, 40, 40, 60]. Experiments run on a single host with Python 3.10, TensorFlow 2.14, 128 GB RAM, and six logical FL clients.

To ensure statistical reliability, each configuration was executed independently with three random seeds (42, 123, 456)

Algorithm 1 SecureDrive-FL   
Require: Server holds public key $p k ;$ each client i holds $s k _ { i }$   
Ensure: Global model W   
1: Initialize global model $W$   
2: for $n = 1$ to N do   
3: $\mathcal { C } _ { n } \gets$ random subset of clients   
4: for all $i \in \mathcal { C } _ { n }$ (executed in parallel) do   
5: Enc(∆w<sub>i</sub>) ← CLIENTUPDATE $( i , W , p k )$   
6: end for   
7: $C _ { \mathrm { a g g } } \gets \boxplus _ { i } \operatorname { E n c } ( \Delta w _ { i } )$   
8: Broadcast $C _ { \mathrm { a g g } }$ to all clients   
9: Each client decrypts: $\Delta W \gets \mathrm { D e c } _ { s k _ { i } } ( C _ { \mathrm { a g g } } )$   
10: $\begin{array} { r } { W \gets W + \frac { 1 } { | \mathcal { C } _ { n } | } \overset { \cdot } { \Delta } W } \end{array}$   
11: end for   
12: return W   
13:   
14: function CLIENTUPDATE $( i , W , p k )$   
15: Input: $C , \sigma , B , \eta , \varepsilon , \delta$   
16: Split $D _ { i } ^ { \mathrm { t r a i n } }$ into B microbatches $\{ x _ { b } \} _ { b = 1 } ^ { B }$   
17: $g  0$   
18: for $b = 1$ to B do   
19: $g _ { b } \gets \nabla \ell ( W ; x _ { b } )$   
20: $g _ { b }  g _ { b }$ · min $\{ 1 , C / \| g _ { b } \| _ { 2 } \}$   
21: $g  g + g _ { b }$   
22: end for   
23: $\tilde { g }  g + { \mathcal { N } } ( 0 , \sigma ^ { 2 } C ^ { 2 } I )$   
24: $\Delta w _ { i } \gets - \eta \tilde { g }$   
25: Compute τ<sub>i</sub> via Eq. (4)   
26: Build mask $M _ { i }$ via Eq. (5)   
27: Enc $( \Delta w _ { i } ) \gets$ FHE.Enc ${ \mathrm { ~ ( ~ } } M _ { i } \odot \Delta w _ { i } { ; } p k { \mathrm { ) } }$   
28: return Enc $( \Delta w _ { i } )$

governing FL client selection, local SGD initialisation, and DP noise sampling. Tables I and II report the mean across these three runs. Variance across seeds was consistently low (accuracy: $\sigma < 1 . 5$ pp; ASR: $\sigma < 2 . 5$ pp; runtime: $\sigma < 2 \% )$ , confirming that reported differences reflect systematic effects rather than stochastic variation.

## D. Adversarial Threats and Defence Mechanisms

1) Man-in-the-Middle (MitM) Gradient Interception: A network-level adversary intercepts gradient transmissions between clients and the server. In the passive case, the adversary stores all intercepted traffic; in the active case, it attempts to replace client updates with manipulated gradients before forwarding them.

Formally, let A denote a network-layer adversary. In the passive MitM setting, A observes all transmissions between client i and the aggregation server:

$$
\mathrm { O b s } _ { \cal A } \ : = \ : \big \{ \ : ( c _ { i } ^ { ( n ) } , \ : { \cal M } _ { i } ^ { ( n ) } ) : i \in \mathcal { C } _ { n } , \ : n = 1 , \ldots , N \big \} ,\tag{11}
$$

where $c _ { i } ^ { ( n ) } = \mathsf { F H E . E n c } ( M _ { i } ^ { ( n ) } \odot \Delta w _ { i } ^ { ( n ) } ; p k )$ is the encrypted high-sensitivity stream and $M _ { i } ^ { ( n ) }$ is the publicly transmitted bitmask. A’s goal is to recover sensitive training data from this observation, either via gradient inversion [6] or by exploiting the plaintext low-magnitude stream. In the active case, A additionally replaces a ciphertext with a crafted update before forwarding:

$$
\tilde { c } _ { i } ^ { ( n ) } = \mathsf { F H E . E n c } \big ( f _ { A } ( \Delta w _ { i } ^ { ( n ) } , \{ c _ { j } ^ { ( n ) } \} _ { j } ) ; \ p k \big ) ,\tag{12}
$$

where $f _ { A }$ is the adversary’s manipulation function (e.g., gradient scaling or replacement).

GASHE protects high-sensitivity components via CKKS semantic security, leaving only low-magnitude, DP-noisedominated parameters in plaintext. To detect active manipulation, the server verifies that aggregated ciphertexts decrypt consistently and monitors the global update $\Delta W .$ . The server maintains a running mean $\mu _ { L }$ and standard deviation $s _ { L }$ of the per-round validation loss over a held-out clean probe set. Round n is flagged if $| L ^ { ( n ) } - \mu _ { L } | > 3 s _ { L }$ (3-sigma rule); the corresponding $C _ { \mathrm { a g g } }$ is discarded and the global model retains the weights $W ^ { ( n - 1 ) }$ from the previous round.

2) Model Poisoning: Following [24], a fraction $\rho$ of clients are compromised $( \rho = 2 / 6 { \approx } 0 . 3 3$ in our evaluation). Each malicious client m $\in \mathcal { M }$ submits a crafted update designed to embed a backdoor trigger t targeting class $y _ { \mathrm { t a r } } .$

$$
\Delta w _ { m } ^ { ( n ) } = \alpha \Delta w _ { \mathrm { t a r g e t } } + ( 1 - \alpha ) \Delta w _ { m } ^ { \mathrm { b e n i g n } } , \qquad \alpha \in ( 0 , 1 ] ,\tag{13}
$$

where $\Delta w _ { \mathrm { t a r g e t } }$ is the gradient direction that amplifies the target-class logit and α controls poisoning intensity. For the multi-round consistency variant of [24], the adversary additionally enforces $\lVert \Delta w _ { m } ^ { ( { \bar { n } } ) } - \Delta w _ { m } ^ { ( n - 1 ) } \rVert _ { 2 } \leq \gamma$ to evade round-toround anomaly detectors. The Attack Success Rate is measured as:

$$
\mathrm { A S R } \ = \ \operatorname* { P r } _ { x \sim \mathcal { D } _ { \mathrm { t e s t } } } \left[ f ( x \oplus t ; W ) = y _ { \mathrm { t a r } } \right] ,\tag{14}
$$

where x ⊕ t denotes a test sample with the backdoor trigger overlaid (class 5, phone call, right hand, in our experiments), and W is the poisoned global model after N rounds. While SecureDrive-FL is designed as a confidentiality framework, GASHE yields implicit robustness: only coordinates exceeding $\tau _ { i }$ are encrypted and transmitted, so the aggregate sensitivity of each client’s contribution is bounded. We additionally employ FedAvg with norm clipping, following Reference [23].

## V. EXPERIMENTAL RESULTS

We compare SecureDrive-FL against DP-SGD alone, Full-CKKS, and GASHE-only as natural baselines representing the privacy–efficiency spectrum, together with TrimmedMean, Krum, BatchCrypt, and FedProx as additional non-DP references covering robust aggregation, encrypted aggregation, and proximal-regularised FL. SecureDrive-FL is the only configuration that simultaneously provides formal DP, IND-CPA communication security, and poisoning resistance on par with DP-SGD alone, incurring only ≈ 8–10% overhead relative to DP-SGD alone.

TABLE I  
FINAL ACCURACY, ASR, AND MACRO-F1 UNDER CLEAN, MITM, AND POISONING CONDITIONS. UNDER MITM, 0% ASR FOR FEDAVG, DP-SGD, AND FEDPROX REFLECTS TRAINING FAILURE (≈10% ACCURACY), NOT DEFENCE; TRIMMEDMEAN AND KRUM RETAIN HIGH ACCURACY INSTEAD (WITHOUT CONFIDENTIALITY). GASHE-ONLY AND FULL-CKKS LACK FORMAL DP (ε=∞) AND COLLAPSE UNDER POISONING (≈14%, NEAR FEDAVG).
<table><tr><td rowspan="2">Configuration</td><td rowspan="2">DP</td><td colspan="3">Accuracy (%)</td><td colspan="3">Poisoning Metrics</td></tr><tr><td>Clean</td><td>MitM</td><td>Poison</td><td>ASR (%)</td><td>Macro-F1 (%)</td><td>Macro-Prec. (%)</td></tr><tr><td>(1) Standard FL (FedAvg)</td><td>X</td><td>93.8</td><td>10.4</td><td>10.4</td><td>0.0</td><td>1.9</td><td>1.0</td></tr><tr><td>(2) FL + DP-SGD (ε0=4)</td><td>√</td><td>80.0</td><td>10.4</td><td>74.0</td><td>3.9</td><td>73.7</td><td>75.7</td></tr><tr><td>(3) FL + GASHE-only</td><td>X</td><td>95.8</td><td>95.0</td><td>14.0</td><td>0.0</td><td>8.0</td><td>25.4</td></tr><tr><td>(4) FL + Full-CKKS</td><td>X</td><td>95.8</td><td>94.8</td><td>13.8</td><td>0.0</td><td>11.9</td><td>70.8</td></tr><tr><td>(5) SecureDrive-FL (DP+GASHE, ε0=4)</td><td>√</td><td>82.2</td><td>78.2</td><td>73.6</td><td>3.9</td><td>73.3</td><td>75.9</td></tr><tr><td>(6) TrimmedMean</td><td>X</td><td>94.4</td><td>95.2</td><td>91.4</td><td>0.0</td><td>91.3</td><td>91.9</td></tr><tr><td>(7) Krum</td><td>X</td><td>84.4</td><td>92.8</td><td>90.2</td><td>1.9</td><td>90.0</td><td>90.9</td></tr><tr><td>(8) BatchCrypt</td><td>X</td><td>95.0</td><td>86.6</td><td>32.8</td><td>0.0</td><td>37.2</td><td>79.8</td></tr><tr><td>(9) FedProx</td><td>X</td><td>95.0</td><td>10.4</td><td>10.4</td><td>0.0</td><td>1.9</td><td>1.0</td></tr></table>

## A. Federated Utility

In the clean setting, FedAvg and GASHE-only reach 93.8% and 95.8% accuracy respectively, confirming that GASHE’s selective CKKS encryption introduces negligible utility overhead because it does not modify the local optimisation trajectory. DP-SGD and SecureDrive-FL reach 80.0% and 82.2% in the clean setting, an 11.6–13.8 percentage point drop from the unencrypted FedAvg baseline (93.8%) that reflects the expected accuracy cost of the tighter per-round privacy budget (ε<sub>0</sub>=4). Notably, SecureDrive-FL is not less accurate than DP-SGD alone in the clean setting (+2.2 pp), confirming that GASHE itself contributes no measurable additional utility cost on top of DP-SGD.

## B. Robustness to Active Attacks

1) Attack Success Rate: Table I summarises robustness across MitM and poisoning attacks. Under MitM, FedAvg, DP-SGD alone, and FedProx collapse to near-random accuracy (≈10%). In contrast, GASHE-only, BatchCrypt, Trimmed-Mean, Krum, and SecureDrive-FL all maintain high utility, achieving 95.0%, 86.6%, 95.2%, 92.8%, and 78.2% accuracy respectively. This confirms that encrypting sensitive gradients effectively removes the interception attack surface for GASHE-based methods, while TrimmedMean and Krum, which use no encryption, are similarly robust to MitM in the updated results, though, unlike GASHE-based methods, they offer no formal confidentiality guarantee for the transmitted gradients themselves.

Robust-aggregation and encrypted-aggregation baselines that carry no DP guarantee perform strongly under poisoning: TrimmedMean and Krum reach 91.4% and 90.2% accuracy respectively with 0.0% and 1.9% ASR, and BatchCrypt partially resists poisoning (32.8% accuracy). This indicates that coordinate-wise robust aggregation is, on its own, an effective defence against the poisoning attack evaluated here. However, none of TrimmedMean, Krum, or BatchCrypt provide a formal (ε, δ)-DP guarantee, and only BatchCrypt provides communication-time confidentiality against MitM interception. SecureDrive-FL remains the only evaluated configuration to simultaneously offer formal DP, IND-CPA communication security, and poisoning resistance on par with DP-SGD alone.

2) Convergence Behaviour: Fig. 2 shows test accuracy over FL rounds. FedAvg collapses under both attack types. Under MitM, DP-SGD alone also collapses (10.4% final accuracy), while GASHE-only closely tracks the clean baseline (≈95% accuracy), confirming that protecting high-sensitivity gradients is sufficient to preserve convergence against a passive/active interception adversary. Under model poisoning, however, GASHE-only no longer tracks the clean baseline: without DP-SGD’s bounded update norm, encrypting highmagnitude components does not prevent a poisoned client’s crafted update from entering the aggregate, and accuracy collapses to 14.0%, close to undefended FedAvg’s ≈10%.

DP-SGD alone and SecureDrive-FL are the only two configurations that converge under poisoning, reaching 74.0% and 73.6% accuracy respectively; SecureDrive-FL further converges to 78.2% under MitM, where DP-SGD alone collapses. Under poisoning, SecureDrive-FL’s accuracy, Macro-F1, and ASR are essentially unchanged relative to DP-SGD alone (within 0.4 pp, and an identical ASR of 3.9%; Table I), with a marginal Macro-Precision gain, indicating that GASHE’s principal benefit in this configuration is closing the MitM attack surface rather than adding further poisoning robustness beyond that already provided by DP-SGD’s bounded update norm.

## C. Computational Overhead

Table II reports measured runtime and peak RSS memory over 120 rounds. Standard FL completes in 360.7–390.1 s with peak RSS 0.64–0.83 GB. GASHE-only increases runtime to 1,130.4–1,538.6 s (≈3.0–4.1× FedAvg clean) and memory to 1.88–2.09 GB. DP-SGD alone is substantially more expensive: 8,388.7–8,847.7 s (≈22.2–23.4×) with 0.77–0.86 GB peak RSS.

SecureDrive-FL combines both mechanisms at a modest additional cost: 9,218.0–9,637.6,s (≈24.3–25.5× the FedAvg baseline), representing only ≈8–10% overhead relative to DP-SGD alone, with 1.76–1.94,GB peak RSS. The marginal cost of adding GASHE on top of DP-SGD is therefore low, while eliminating the MitM attack surface provides substantial

![](images/e75a82699a2f49e46c3bf60cba83eade05d74c4e2515466466835f0cc842df6a.jpg)  
Fig. 2. Test accuracy vs. communication round under Man-in-the-Middle (left) and model-poisoning (right) attacks, against a clean-training benchmark. Under MitM, FedAvg, FedProx, and FL+DP-SGD collapse to near-random accuracy, while FL+GASHE, BatchCrypt, TrimmedMean, Krum, and SecureDrive-FL retain useful accuracy. Under poisoning, FedAvg, FedProx, and FL+GASHE collapse instead, while FL+DP-SGD, SecureDrive-FL, and (more slowly) BatchCrypt retain partial accuracy. Final-round values are given in Table I.

TABLE II  
MEASURED EXECUTION TIME AND PEAK RSS MEMORY OVER 120 FL ROUNDS. OVERHEAD RELATIVE TO FEDAVG (CLEAN, 378.7 S). VALUES ARE REPORTED AS MEAN OVER THREE RANDOM SEEDS (42, 123, 456).
<table><tr><td>Configuration</td><td>Time (s)</td><td>Overhead</td><td>Peak RSS (MB)</td></tr><tr><td>FedAvg (clean)</td><td>378.7</td><td>1.0×</td><td>655.3</td></tr><tr><td>FedAvg (MitM)</td><td>390.1</td><td>1.0×</td><td>822.2</td></tr><tr><td>FedAvg (Poison)</td><td>360.7</td><td>1.0×</td><td>830.0</td></tr><tr><td>GASHE-only (clean)</td><td>1254.3</td><td>3.3×</td><td>2119.8</td></tr><tr><td>GASHE-only (MitM)</td><td>1538.6</td><td>4.1×</td><td>2142.4</td></tr><tr><td>GASHE-only (Poison)</td><td>1130.4</td><td>3.0×</td><td>1928.6</td></tr><tr><td>Full-CKKS (clean)</td><td>1238.2</td><td>3.3×</td><td>1876.8</td></tr><tr><td>Full-CKKS (MitM)</td><td>1481.3</td><td>3.9×</td><td>1928.1</td></tr><tr><td>Full-CKKS (Poison)</td><td>1227.3</td><td>3.2×</td><td>1839.1</td></tr><tr><td>DP-SGD (clean)</td><td>8847.7</td><td>23.4×</td><td>876.0</td></tr><tr><td>DP-SGD (MitM)</td><td>8805.8</td><td>23.3×</td><td>786.4</td></tr><tr><td>DP-SGD (Poison)</td><td>8388.7</td><td>22.2×</td><td>848.4</td></tr><tr><td>SecureDrive-FL (clean)</td><td>9588.4</td><td>25.3×</td><td>1986.8</td></tr><tr><td>SecureDrive-FL (MitM)</td><td>9637.6</td><td>25.5×</td><td>1799.7</td></tr><tr><td>SecureDrive-FL (Poison)</td><td>9218.0</td><td>24.3×</td><td>1813.9</td></tr><tr><td>TrimmedMean (clean)</td><td>355.9</td><td>0.9×</td><td>1567.8</td></tr><tr><td>TrimmedMean (MitM)</td><td>372.5</td><td>1.0×</td><td>1593.7</td></tr><tr><td>TrimmedMean (Poison)</td><td>366.1</td><td>1.0×</td><td>731.9</td></tr><tr><td>Krum (clean)</td><td>354.6</td><td>0.9×</td><td>1643.0</td></tr><tr><td>Krum (MitM)</td><td>371.5</td><td>1.0×</td><td>1787.8</td></tr><tr><td>Krum (Poison)</td><td>347.3</td><td>0.9×</td><td>830.9</td></tr><tr><td>BatchCrypt (clean)</td><td>4958.8</td><td>13.1×</td><td>1483.2</td></tr><tr><td>BatchCrypt (MitM)</td><td>8601.6</td><td>22.7×</td><td>1574.9</td></tr><tr><td>BatchCrypt (Poison)</td><td>4992.4</td><td>13.2×</td><td>1621.7</td></tr><tr><td>FedProx (clean)</td><td>591.9</td><td>1.6×</td><td>875.6</td></tr><tr><td>FedProx (MitM)</td><td>610.3</td><td>1.6×</td><td>1033.6</td></tr><tr><td>FedProx (Poison)</td><td>623.7</td><td>1.6×</td><td>1043.8</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

security benefits.

TrimmedMean and Krum remain close to FedAvg’s own runtime (≈0.9–1.0×, 347.3–372.5 s) with peak RSS 0.71– 1.75 GB, and, per the updated results in Table I, both now also resist poisoning (91.2% and 89.4% accuracy). Neither, however, provides a formal DP guarantee or communication-time confidentiality against MitM interception, unlike SecureDrive-FL. BatchCrypt, which applies quantised/batched CKKS encryption to the full parameter set, incurs ≈13.1–22.7× overhead (4,958.8–8,601.6 s) with peak RSS 1.45–1.58 GB; FedProx adds a comparatively modest ≈1.6× overhead (591.9–623.7 s) with peak RSS 0.86–1.02 GB, reflecting its lightweight proximal-term regularisation rather than any encryption cost.

For reference, a full-parameter CKKS baseline is measured at 1,227.3–1,481.3 s and 1.80–1.88 GB peak RSS, comparable to and in some cases exceeding GASHE-only. Despite encrypting a smaller subset of gradient components, GASHE does not yield a proportional runtime or memory reduction relative to Full-CKKS in the current implementation, as the perround threshold computation, mask construction, and plaintext/ciphertext stream-splitting introduce bookkeeping overhead that offsets the savings from the reduced encryptedparameter count.

## VI. CONCLUSION AND FUTURE WORK

We presented SecureDrive-FL, a hybrid privacy-preserving federated learning framework coupling DP-SGD with Gradient-Aware Selective Homomorphic Encryption (GASHE) for secure driver monitoring in connected vehicles. The key novelty is the principled, closed-loop derivation of HE encryption decisions from DP-SGD sensitivity bounds, enabling GASHE to automatically target the most privacy-critical gradient components while aligning communication-time protection with training-time privacy calibration. Evaluated against MitM interception and model poisoning, SecureDrive-FL improves clean accuracy over DP-SGD alone (82.2% vs. 80.0%), maintains 78.2% accuracy under MitM where DP-SGD alone collapses to 10.4%, and retains comparable performance under poisoning (73.6% vs. 74.0%, both at 3.9% ASR), all under only ≈8–10% additional runtime overhead above DP-SGD alone.

Future directions include: (i) scaling SecureDrive-FL to larger, heterogeneous vehicle fleets with varying compute and bandwidth profiles; (ii) integrating Byzantine-robust aggregation rules (e.g., BALANCE) for stronger protection against adaptive, multi-round adversaries; and (iv) deployment on a physical in-vehicle cockpit prototype to benchmark endto-end latency, energy, and FHE performance on embedded automotive hardware.

## ACKNOWLEDGMENT

This publication is carried out as part of the Hardware Abstraction Layer for Software Defined Vehicle (HAL4SDV) Project No. 101139789, which is co-funded by the European Union (Chips JU).

## REFERENCES

[1] B. Gul, N. Devarakonda, D. Dittler, N. Jazdi, and M. Weyrich, “Using¨ federated learning in the context of software-defined mobility systems for predictive quality of service,” vol. 2419, 2023.

[2] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS), ser. Proceedings of Machine Learning Research, vol. 54. PMLR, 2017, pp. 1273–1282. [Online]. Available: https://proceedings.mlr.press/v54/mcmahan17a.html

[3] M. Fredrikson, S. Jha, and T. Ristenpart, “Model inversion attacks that exploit confidence information and basic countermeasures,” in Proceedings of the 22nd ACM SIGSAC Conference on Computer and Communications Security (CCS). ACM, 2015, pp. 1322–1333.

[4] R. Shokri, M. Stronati, C. Song, and V. Shmatikov, “Membership inference attacks against machine learning models,” in 2017 IEEE Symposium on Security and Privacy (S&P). IEEE, 2017, pp. 3–18.

[5] B. Hitaj, G. Ateniese, and F. Perez-Cruz, “Deep models under the GAN: Information leakage from collaborative deep learning,” in Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS). ACM, 2017, pp. 603–618.

[6] L. Zhu, Z. Liu, and S. Han, “Deep leakage from gradients,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019. [Online]. Available: https://proceedings.neurips.cc/paper/2019/ hash/60a6c4002cc7b29142def8871531281a-Abstract.html

[7] B. C. Gul, D. Dittler, N. Jazdi, and M. Weyrich, “Federated learning¨ for comfort features in vehicles with collaborative sensing: A review,” in 2024 IEEE 29th International Conference on Emerging Technologies and Factory Automation (ETFA), 2024, pp. 1–7.

[8] C. Gentry, “Fully homomorphic encryption using ideal lattices,” in Proceedings of the 41st Annual ACM Symposium on Theory of Computing (STOC). ACM, 2009, pp. 169–178.

[9] S. Gupta, R. Cammarota, and T. S. Rosing, “MemFHE: End-to-end<sup>ˇ</sup> computing with fully homomorphic encryption in memory,” ACM Transactions on Embedded Computing Systems, vol. 21, no. 6, 2022.

[10] M. Abadi, A. Chu, I. Goodfellow, H. B. McMahan, I. Mironov, K. Talwar, and L. Zhang, “Deep learning with differential privacy,” in Proceedings of the 2016 ACM SIGSAC Conference on Computer and Communications Security (CCS). ACM, 2016, pp. 308–318.

[11] R. C. Geyer, T. Klein, and M. Nabi, “Differentially private federated learning: A client level perspective,” arXiv preprint arXiv:1712.07557, 2017. [Online]. Available: https://arxiv.org/abs/1712.07557

[12] S. Truex, N. Li, T. Steinke, M. F. Gaboardi, Y. Cao, and L. Fan, “A hybrid approach to privacy-preserving federated learning,” in Proceedings of the 2020 Network and Distributed System Security Symposium (NDSS), 2020, poster. [Online]. Available: https://www.ndss-symposium.org/ndss2020/posters/ a-hybrid-approach-to-privacy-preserving-federated-learning/

[13] C. Zhang, S. Li, J. Xia, W. Wang, F. Yan, and Y. Liu, “BatchCrypt: Efficient homomorphic encryption for cross-silo federated learning,” in Proceedings of the 2020 USENIX Annual Technical Conference (ATC ’20). USENIX Association, 2020, pp. 493– 506. [Online]. Available: https://www.usenix.org/conference/atc20/ presentation/zhang-chengliang

[14] N. M. Hijazi, M. Aloqaily, M. Guizani, B. Ouni, and F. Karray, “Secure federated learning with fully homomorphic encryption for IoT communications,” IEEE Internet of Things Journal, vol. 11, no. 3, pp. 4289–4300, 2024.

[15] L. Melis, C. Song, E. D. Cristofaro, and V. Shmatikov, “Exploiting unintended feature leakage in collaborative learning,” in 2019 IEEE Symposium on Security and Privacy (S&P). IEEE, 2019, pp. 691– 706.

[16] H. Ren, J. Deng, and X. Xie, “Gradient leakage attacks in federated learning environments,” arXiv preprint arXiv:2510.23931, 2025. [Online]. Available: https://arxiv.org/abs/2510.23931

[17] C. Dwork, “Differential privacy,” in Proceedings of the 33rd International Colloquium on Automata, Languages and Programming (ICALP 2006), Part II, ser. Lecture Notes in Computer Science, vol. 4052. Springer, 2006, pp. 1–12.

[18] H. B. McMahan, D. Ramage, K. Talwar, and L. Zhang, “Learning differentially private recurrent language models,” in International Conference on Learning Representations (ICLR), 2018. [Online]. Available: https://openreview.net/forum?id=BJ0hF1Z0b

[19] I. Mironov, “Renyi differential privacy,” in´ Proceedings ofthe 30th IEEE Computer Security Foundations Symposium (CSF). IEEE, 2017, pp. 263–275.

[20] M. Bun and T. Steinke, “Concentrated differential privacy: Simplifications, extensions, and lower bounds,” in Theory of Cryptography Conference (TCC), ser. Lecture Notes in Computer Science, vol. 9985. Springer, 2016, pp. 635–658.

[21] K. Bonawitz, V. Ivanov, B. Kreuter, A. Marcedone, H. B. McMahan, S. Patel, D. Ramage, A. Segal, and K. Seth, “Practical secure aggregation for privacy-preserving machine learning,” in Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS). ACM, 2017, pp. 1175–1191.

[22] A. Benaissa, B. Retiat, B. Cebere, and A. E. Belfedhal, “TenSEAL: A library for encrypted tensor operations using homomorphic encryption,” arXiv preprint arXiv:2104.03152, 2021, iCLR 2021 Workshop on Distributed and Private Machine Learning (DPML). [Online]. Available: https://arxiv.org/abs/2104.03152

[23] L. Lyu, H. Yu, X. Ma, C. Chen, L. Sun, J. Zhao, Q. Yang, and P. S. Yu, “Privacy and robustness in federated learning: Attacks and defenses,” IEEE Transactions on Neural Networks and Learning Systems, vol. 35, no. 7, pp. 8726–8746, 2024.

[24] Y. Xie, M. Fang, and N. Z. Gong, “Model poisoning attacks to federated learning via multi-round consistency,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE / Computer Vision Foundation, 2025, pp. 15 454–15 463.

[25] S. Truex, N. Baracaldo, A. Anwar, T. Steinke, H. Ludwig, and R. Zhang, “A hybrid approach to privacy-preserving federated learning,” in Proceedings of the 12th ACM Workshop on Artificial Intelligence and Security (AISec), 2019.

[26] C. Dwork and A. Roth, “The algorithmic foundations of differential privacy,” Found. Trends Theor. Comput. Sci., vol. 9, no. 3–4, p. 211–407, Aug. 2014. [Online]. Available: https://doi.org/10.1561/0400000042

[27] State Farm and Kaggle, “State farm distracted driver detection,” https: //www.kaggle.com/c/state-farm-distracted-driver-detection, 2016.

[28] Y. Abouelnaga, H. M. Eraqi, and M. N. Moustafa, “Real-time distracted driver posture classification,” arXiv preprint arXiv:1706.09498, 2017. [Online]. Available: https://arxiv.org/abs/1706.09498

[29] European Parliament and Council, “Regulation (EU) 2016/679 of the European Parliament and of the Council (GDPR),” Official Journal of the European Union, Tech. Rep., 2016, l 119/1.

[30] Google, “TensorFlow privacy,” https://github.com/tensorflow/privacy, 2024.