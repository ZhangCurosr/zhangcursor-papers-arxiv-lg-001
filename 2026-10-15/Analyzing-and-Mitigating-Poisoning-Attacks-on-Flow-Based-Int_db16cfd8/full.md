# Analyzing and Mitigating Poisoning Attacks on Flow-Based Intrusion Detection

Florian Spelter1,2, Johannes F. Loevenich1,3, Tobias Hürten¹,

Erik Adler1,4, Maxime Schwarzer1,4, Laurin Holz1,5, and Roberto Rigolin F. Lopes1

1CortAIx Labs, Thales Deutschland, Ditzingen, Germany

2Grey Solutions, Düsseldorf, Germany

3Department of Mathematics/Computer Science, University of Osnabrück, Osnabrück, Germany

4Department of Informatics, Karlsruhe Institute of Technology (KIT), Karlsruhe, Germany

5Department of Computer Science, Ulm University, Ulm, Germany

Email: spelter@greysolutions.de,

{johannes.loevenich, tobias.huerten, erik.adler, maxime.schwarzer, laurin.holz, roberto.rigolin} @thalesgroup.com

Abstract—Although Artificial Intelligence (AI)-based Intrusion Detection Systems (IDSs) are increasingly deployed in operational networks, their dependence on large-scale training data introduces a critical vulnerability to stealthy poisoning attacks. We demonstrate that latent backdoors can be successfully implanted in flow-based IDS models by poisoning as little as 0.001% of the training data, maintaining near-nominal performance on clean inputs while ensuring high attack success rates. To address this threat, we propose a diagnostic-first methodology that localizes trigger-relevant features through constrained perturbation analysis and detects poisoning-induced behavioral drift by comparison with a distilled baseline model. The proposed methodology further enables targeted mitigation via automated data sanitization or neuron pruning. Our findings expose the operational risks of near-undetectable backdoors and provide a scalable approach to analyzing, quantifying, and mitigating backdoor exposure in AI-based IDS.

Index Terms—Deep Neural Networks, Poisoning Attacks, Autonomous Cyber Defence

## I. INTRODUCTION

Artificial Intelligence (AI)-based Intrusion Detection Systems (IDSs) are employed to secure critical network infrastructure by utilizing Deep Neural Networks (DNNs) to classify flow-level telemetry within software-defined networks. However, the efficacy of these models is fundamentally dependent on the integrity of their training datasets and development pipelines, making them susceptible to trainingtime attacks, such as data poisoning and backdoor implantation [1]. These vulnerabilities are exacerbated in softwaredefined architectures, where frequent model retraining and cross-domain sharing expand the attack surface. Recent work has shown that poisoning attacks can achieve high stealth by preserving nominal validation performance while enabling targeted misclassification [2]. Within the context of critical infrastructure, where IDS outputs drive cyber situational awareness and Command and Control (C2) processes, these compromised models pose a direct threat to mission-critical operational effectiveness.

This paper investigates the vulnerability of flow-based IDS models to data poisoning attacks, utilizing open source benchmarks such as the CSE-CIC-IDS2018 dataset [3]. Comprising over 80 features derived from diverse, realistic network scenarios, this dataset is widely used to benchmark IDS performance. We demonstrate that effective backdoors can be implanted by modifying as little as 0.001% of the training data without affecting clean-data performance. To address this threat, we propose a diagnostic-first methodology that infers attacker intent and localizes trigger-relevant features via constrained perturbation analysis and comparative model diagnostics. In contrast to prior work focusing on image domains or training-time defenses, our approach targets post-training diagnostic analysis in high-dimensional tabular IDS settings, enabling feature-level attribution and targeted mitigation.

The main contributions of this paper are:

• Demonstrating that highly effective backdoors can be implanted in flow-based IDS models by poisoning only 0.001% of the training data without degrading nominal performance on clean data.

• Analyzing single- and multi-feature poisoning attacks on flow-based IDS models, showing that timing-based and combined feature manipulations are particularly susceptible to backdoor implantation.

• Proposing and evaluating a diagnostic-first methodology that supports targeted backdoor mitigation under realistic operational constraints.

The remainder of the paper is organized as follows. Section II reviews related work on data poisoning and adversarial machine learning. Section III describes the dataset, models, poisoning procedures, and the proposed methodology. Section IV defines the model configuration, quantifies the impact of poisoning attacks and evaluates our mitigation methodology. Finally, Section V concludes the paper.

## II. RELATED WORK

Adversarial manipulation of Machine Learning (ML) models has evolved into a strategic threat vector, where data poisoning and backdoor attacks serve as mechanisms for the covert compromise of defensive cyber systems. While foundational work by Biggio et al. [4] showed that poisoning attacks can degrade classifier accuracy, subsequent research demonstrated that backdoors can embed covert, attacker-controlled behaviors while preserving standard performance [5]. The investigations in [6], [7] suggest that such compromises are achievable with minimal poisoning budgets.

Prior research has demonstrated that ML-based IDS operating on packet-level data suffers from limited resilience to poisoning attacks. At the same time, these studies show that once a backdoor trigger is correctly identified, model performance on poisoned inputs can be almost fully restored, with recovery rates reaching up to 99% [1]. Although major work such as [8] introduces a general framework for reverseengineering backdoor triggers through input perturbations in image classification models, our findings indicate that directly transferring this approach to IDSs is ineffective and requires substantial modification.

The CSE-CIC-IDS2018 dataset has since emerged as a standard benchmark for evaluating IDS architectures [9], [10]. However, most research prioritizes classification accuracy and detection rates, largely overlooking how its high-dimensional feature space might be exploited to embed persistent, stealthy backdoors. Existing poisoning defenses on this dataset focus primarily on training-time robustness, particularly within federated learning environments. For instance, Zhang et al. propose a federated IDS defense that identifies anomalous client updates during collaborative training [11], while [12] investigates poisoning under data heterogeneity with personalization-aware defenses.

Recent work has further explored backdoor robustness in federated learning, including attacks targeting critical model components (e.g., backdoor-critical layers) and defenses based on model inspection and anomaly detection of client updates [13], [14]. While these approaches operate at training time and focus on model-level behavior, our work addresses post-training analysis in flow-based IDS and enables featurelevel attribution and targeted mitigation.

## III. METHODOLOGY

This section presents a methodology for analyzing and mitigating poisoning attacks on AI models. We assume an attacker with limited control over the training data and no direct access to model internals. The methodology illustrated in Fig. 1 adopts a diagnostic-first approach, directly localizing backdoor behavior at the feature level and quantifying its influence on model predictions. By comparing the target model against a distilled baseline model, it derives a feature-salience ranking that enables two distinct remediation paths: dataset sanitization via the removal of poisoned samples or model recovery through targeted architectural refinement.

![](images/9f7a241665be434b73271676be158027298f20ea14c5790a59c4a01b32c032e5.jpg)  
Fig. 1: Proposed methodology comprising data-centric sanitization and model-centric neuron pruning.

1) Trigger Search and Attack Success Rate (ASR): This phase aims to isolate the features that most significantly drive backdoor activation and to quantify the resultant ASR. The analysis operates at the feature level, allowing the identification of potential triggers and characterizing the model's sensitivity to individual feature manipulations.

The trigger-identification problem is formulated as a univariate (per-feature) perturbation search to avoid the instability of joint optimization across heterogeneous flowlevel features with varying scales, distributions, and semantics. Consequently, instead of optimizing a full perturbation vector in $\mathbb { R } ^ { d }$ , the algorithm evaluates each feature independently to find a scalar $\delta \in \mathbb { R }$ that induces misclassification into a target class $y _ { t g t }$ . Formally, for a subset of samples $S = \{ x : f ( x ) \neq$ $y _ { t g t } \}$ , we seek:

$$
\operatorname* { m i n } _ { \delta \in \mathbb { R } } \quad \mathrm { s u b j e c t ~ t o } \quad \mathbb { E } _ { x \sim S } [ f ( \phi _ { i } ( x , \delta ) ) = y _ { \mathrm { t g t } } ] \ge \tau ,\tag{1}
$$

where $\phi _ { i } ( x , \delta )$ applies the perturbation exclusively to feature $i ,$ and τ denotes the minimum ASR threshold. This threshold can be fixed $( \mathbf { e } . \mathbf { g } . , \ \tau \ = \ 0 . 8 )$ , defined relative to baseline misclassification rates, or tuned based on operational tolerances for false activations. The resulting ASR serves as a quantitative metric of a feature's contribution to the backdoor effect. While the target class $y _ { t g t }$ is arbitrary in principle, we select the benign class as the adversarial target to ensure tractability against unknown adversarial objectives. However, the methodology remains generalizable to other target classes.

Due to the diverse manipulation strategies available to an adversary, and since the attacker's specific strategy is unknown, each feature i is evaluated under three perturbation modes:

$$
\begin{array} { r l } { \mathrm { S h i f t } { : } } & { { } x _ { i }  x _ { i } + \delta , } \\ { \mathrm { O v e r r i d e } { : } } & { { } x _ { i }  \delta , } \\ { \mathrm { S a m p l i n g } { : } } & { { } x _ { i }  \mathrm { s a m p l e } ( X _ { i \mid y = y _ { t g t } } ) . } \end{array}
$$

The shift and override modes represent explicit and structural modifications, whereas the sampling mode facilitates an imitation-driven strategy where an adversary adopts empirically representative values for a feature i from the target class distribution $X _ { i \mid y = y _ { t g t } }$ with respect to that feature. This multi-modal approach ensures a robust, interpretable ranking of feature-level influence while maintaining computational efficiency through a set of optimization strategies: (1)

```latex
Algorithm 1 Univariate Trigger Search Algorithm
Require: Classifier $f ,$ dataset D, targets , features $i \in$
$\{ 1 , \ldots , d \}$ , modes M, threshold τ, candidate set $\Delta$
Ensure: Minimal δ per $( y _ { \mathrm { t g t } } , i , m )$ with $\mathbf { A S R } \ge \tau$
1: for $y _ { \mathrm { t g t } } \in \mathcal { V }$ do
2: $\bar { S }  \{ x : ( x , y ) \in D , y \neq y _ { \mathrm { t g t } } \}$
3: for $i = 1$ to d do
4: for all $m \in \mathcal { M }$ do
5: δ* ← NONE
6: for all $\delta \in \Delta$ do
7: Define $x ^ { \prime } \gets x$ by pertubing only feature i
using mode m with magnitude δ
8: $\begin{array} { r } { \mathrm { A S R } ( \delta ) \gets \frac { 1 } { | S | } \sum _ { x \in S } { \bf 1 } \{ f ( x ^ { \prime } ) = y _ { \mathrm { t g t } } \} } \end{array}$
9: if $\mathbf { A S R } ( \delta ) \geq \tau$
and $( \delta ^ { \star } = \mathrm { N O N E }$ or $\vert \delta \vert < \vert \delta ^ { \star } \vert )$ then
10: $\delta ^ { \star }  \delta$
11: end if
12: end for
13: Record $( y _ { \mathrm { t g t } } , i , m , \delta ^ { \star } )$
14: end for
15: end for
16: end for
17: return recorded results
```

source subsampling reduces per-evaluation costs by using representative data subsets, (2) early termination halts the search immediately once the ASR threshold τ is satisfied, (3) parallelization enables the execution of feature-level evaluations concurrently, and (4) feature prioritization restricts the search space to high-variance or domain-relevant feature subsets. These measures facilitate the efficient identification of trigger-relevant features and provide a reliable estimation of their ASR, as defined in Algorithm 1.

2) Model Comparison and Feature Ranking: While the single-feature analysis yields both the minimal perturbation required to induce misclassification and the corresponding ASR for each feature, these quantities alone are insufficient to reliably identify poisoned features. Even in clean settings, modern IDS models exhibit heterogeneous sensitivities across features due to architectural biases, data imbalance, or inherent feature correlations. This ambiguity is exacerbated in multifeature attacks, where distributed triggers may dilute the individual ASR signal. Consequently, a high ASR or a low perturbation magnitude does not by itself constitute sufficient evidence of poisoning.

To address this limitation, we propose a comparative analysis using a baseline reference model. This baseline is obtained via knowledge distillation [15], where a secondary model of identical architecture is trained to replicate benign behavior from a small, curated subset of the data for which integrity can be assumed. This distilled model acts as a behavioral proxy for the expected system responses in a clean environment. The identification algorithm described in Step 1)

is executed independently on both models. By comparing their responses to identical perturbations, we extract a feature-wise deviation signal. This approach effectively filters out natural model variability, isolating behavioral anomalies specifically attributable to poisoning.

The comparison follows two complementary dimensions: structural drift and behavioral drift. Structural drift captures the shift in the minimal perturbation magnitude |δ\*| required to induce misclassification for feature i (Algorithm 1) between the evaluated model $f _ { e v a l }$ and the baseline model $f _ { b a s e }$

$$
\begin{array} { r } { \Delta _ { \mathrm { d i f f } , i } = \left| { \delta _ { e v a l , i } ^ { * } - \delta _ { b a s e , i } ^ { * } } \right| . } \end{array}\tag{2}
$$

A large $\Delta _ { \mathrm { d i f f } , i }$ indicates that poisoning has significantly altered the model's sensitivity to changes in that specific feature. Conversely, behavioral drift measures discrepancies in the maximum ASR achieved across the three perturbation modes:

$$
\mathrm { A S R } _ { \mathrm { d i f f } , i } = \left| \operatorname* { m a x } _ { m } ( A S R _ { e v a l , i } ) - \operatorname* { m a x } _ { m } ( A S R _ { b a s e , i } ) \right| ,\tag{3}
$$

where $m ~ \in ~ \{ S h i f t , O v e r r i d e , S a m p l i n g \}$ . This captures how much more (or less) reliably a feature can be exploited to induce the target class compared to the clean baseline.

To ensure comparability across features with heterogeneous scales, both drift measures are normalized relative to their observed maximums:

$$
\Delta _ { \mathrm { n o r m } , i } = \frac { \Delta _ { \mathrm { d i f f } , i } } { \operatorname* { m a x } _ { j } ( \Delta _ { \mathrm { d i f f } , \mathrm { j } } ) } ,\tag{4}
$$

$$
\mathsf { A S R } _ { \mathrm { n o r m } , i } = \frac { \mathsf { A S R } _ { \mathrm { d i f f } , i } } { \operatorname* { m a x } _ { j } ( \mathsf { A S R } _ { \mathrm { d i f f } , j } ) } .\tag{5}
$$

The overall deviation strength for each feature is computed by combining these metrics through their Euclidean distance and arithmetic mean to produce a final ranking score:

$$
S _ { \mathrm { c o m b } , i } = \sqrt { ( \Delta _ { \mathrm { n o r m } , i } ) ^ { 2 } + ( \mathrm { A S R } _ { \mathrm { n o r m } , i } ) ^ { 2 } } ,\tag{6}
$$

$$
S c o r e _ { i } = \frac { \Delta _ { \mathrm { n o r m } , i } + \mathrm { A S R } _ { \mathrm { n o r m } , i } + S _ { \mathrm { c o m b } , i } } { 3 } .\tag{7}
$$

Features are ranked by Scorei, yielding an ordered list of flow attributes that deviate most significantly from nominal expectation. High-ranking features are identified as the most likely components of the implanted trigger, providing a principled basis for subsequent mitigation steps such as data sanitization or neuron pruning.

3) Backdoor Mitigation: Following the identification of trigger-relevant features, the methodology transitions to mitigation. We consider two complementary mitigation strategies: data-centric sanitization and model-centric neuron pruning, both of which culminate in the retraining step depicted in Fig. 1.

a) Data-centric Sanitization: The first mitigation strategy operates at the data level by leveraging identified trigger characteristics to detect and neutralize poisoned samples. Training instances exhibiting feature patterns consistent with high-ranking triggers from Step 2) are either purged from the corpus or corrected. The sanitized dataset is then used to train a new model. While this approach directly eliminates poisoning artifacts from the data distribution, its feasibility depends on access to the original training data and the computational overhead of full-model retraining.

![](images/c9c7dfff23e227bc0e25ac1ae469086edd64a114783e77d9641ac608bf2080ae.jpg)  
Fig. 2: Confusion matrix: baseline model.

b) Model-centric Neuron Pruning: The second mitigation strategy focuses on the model architecture and does not require modifying the training dataset. Building on the methodology of prior work [1], this approach utilizes feature activation analysis to contrast internal neuron responses between inputs containing the identified trigger and clean samples. Neurons that exhibit statistically significant activation variance in the presence of the trigger are indicative of backdoor-related behavior.

These neurons are ranked by their contribution to the ASR, and structured pruning is applied iteratively by zeroing the weights starting with the highest-ranked neurons. After each iteration, the ASR is re-evaluated, and pruning continues until the ASR drops below a predefined threshold while preserving nominal classification performance. Finally, the pruned model undergoes a targeted fine-tuning on a small validation set of previously unseen, clean data to restore generalization and compensate for reduced model capacity.

## IV. RESULTS

## A. Model Configuration

Given the high dimensionality and heterogeneity of the feature space in the CSE-CIC-IDS2018 dataset [3], we employ a DNN implemented in TensorFlow/Keras. The network consists of five fully connected hidden layers with 512, 256, 128, 64, and 32 neurons using Rectified Linear Unit (ReLU) activation. Batch normalization is applied after each layer to stabilize training, while a dropout strategy with decreasing rates between 0.3 and 0.1 mitigates overfitting. The final layer uses a Softmax activation to produce a categorical probability distribution over the output classes.

Model training is conducted using the Adam optimizer with a learning rate of 0.001 and sparse categorical crossentropy loss over batches of size 512 for up to 50 epochs. Data preprocessing involves numerical label encoding, removal of invalid samples, and feature normalization. To ensure representational integrity, the dataset is partitioned into an 80/20 stratified training and test split. Training stability is maintained through early stopping, restoring optimal weights upon validation loss stagnation, and a learning rate scheduler that decays the learning rate on validation plateaus. Model performance is evaluated using accuracy, precision, recall, F1- score, and confusion matrices.

![](images/c7ca489119bc5b8d764849cb8f4c91021201c75db30083be03161219b641bc77.jpg)  
Fig. 3: ASR for specific features acting as triggers with increasing poisoning percentages (0.0005% to 1%).

## B. Impact of Poisoning

To evaluate the impact of poisoning attacks on flowbased IDS, we first establish a clean baseline for the CSE-CIC-IDS2018 dataset [3] (feature names follow the original notation, e.g., Bwd IAT Tot for total backward inter-arrival time). We focus on high-volume attack classes (Botnet, BruteForce, Denial of Service (DoS), and Distributed Denial of Service (DDoS)), which provide stable flow-level patterns for controlled poisoning experiments due to their higher sample representation. As shown in the confusion matrix in Fig. 2, the model achieves near-perfect precision and recall for benign flows and high-volume attack types such as Botnet and DDoS. Despite strong overall performance, systematic confusions persist. Specifically, reciprocal misclassifications between DoS and BruteForce flows suggest partially overlapping feature characteristics between these categories. Furthermore, a small number of attack samples are misclassified as benign, highlighting inherent limitations in IDS decision boundaries even under optimal, non-adversarial training conditions.

Fig. 3 illustrates the susceptibility of individual flow-level features to univariate poisoning across a range of poisoning percentages (0.0005% to 1%). The heatmap reports the ASR averaged across the four investigated attack categories for key features, including Bwd IAT Tot, Bwd Pkts/s, Flow Duration, Flow IAT Max, Flow Pkts/s, Fwd IAT Tot, and Fwd Pkts/s. A clear susceptibility hierarchy emerges: temporal features such as Bwd IAT Tot, Flow IAT Max, and Fwd IAT Tot achieve near-maximal ASR at minimal poisoning percentages, indicating that they serve as highly potent backdoor carriers. Conversely, rate-based features such as Fwd Pkts/s, Bwd Pkts/s, and particularly Flow Pkts/s exhibit weaker scaling behavior, requiring significantly higher poisoning levels to achieve comparable efficacy. These findings suggest that backdoor reliability is highly sensitive to feature semantics, with timingrelated attributes proving particularly vulnerable to low-rate adversarial manipulation.

![](images/c540406cff3aa9ec03a6b1f5882acdb165b6a60aa47ad22fee4aa40e7b0eca25.jpg)

![](images/5af2fe5b3a90235ffdebeff2a1a676a47bc23b8c875a5c57600998962b8aae51.jpg)

![](images/96579f069f0f09a89754936896a02606dac834acb32ecef28f023b511d8dd3aa.jpg)

![](images/944afe969009857590c8da4976bec80d4535b5a379e096444db7d590b1e28129.jpg)

![](images/247b411d77cca611ef2d1da0ce143575f61a8c0daa02842441c455da3b90f838.jpg)  
Fig. 4: Accuracy of backdoor for a model with two features poisoned (Flow Duration, Fwd IAT Tot).

Combined, Fig. 2 and Fig. 3 demonstrate that univariate poisoning is sufficient to implant highly effective yet stealthy backdoors. The lack of observable nominal performance degradation with respect to standard evaluation metrics underscores the inadequacy of traditional evaluation frameworks and highlights the necessity for targeted diagnostic analysis to expose latent adversarial triggers.

While Algorithm 1 offers insights into feature sensitivity, its raw outputs alone are insufficient to reliably identify poisoned features, as evidenced by Table I. Clean models frequently exhibit high intrinsic sensitivity to specific feature perturbations, rendering absolute perturbation magnitude or ASR an ambiguous indicator of poisoning. This limitation motivates the incorporation of a comparative diagnostic stage, in which the evaluated model is compared to a distilled baseline model. The efficacy of this differential analysis is reflected in Tables II and III, where the deviation signal clearly distinguishes between nominal sensitivities and implanted backdoors.

Across all evaluated poisoning scenarios, the poisoned feature consistently yields a significantly higher average drift score than the unperturbed features, creating an unambiguous separation in the features space. For instance, when Bwd Pkts/s is poisoned, it receives an average score of 0.986, distinctly exceeding all other candidates. Similar separation is observed for Bwd IAT Tot, which ranks second with a clear margin, and for Flow Duration, where an even stronger separation emerges, yielding an average score of 1.070 and identifying it as the sole plausible trigger feature.

TABLE I: Highlight: Backward Pkts/s
<table><tr><td>Rank</td><td>Feature</td><td>min|∆|</td><td>ASR</td></tr><tr><td>1</td><td>Fwd IAT Std</td><td>0.4776</td><td>0.804</td></tr><tr><td>2</td><td>Flow IAT Min</td><td>0.5893</td><td>0.806</td></tr><tr><td>3</td><td>Flow IAT Mean</td><td>0.8026</td><td>0.805</td></tr><tr><td>4</td><td>Flow IAT Std</td><td>0.9262</td><td>0.824</td></tr><tr><td>21</td><td>Bwd Pkts/s</td><td>4.0000</td><td>0.987</td></tr></table>

TABLE II: Top 5 Features — Flow Duration vs Baseline
<table><tr><td>Feature Name</td><td>∆ Drift</td><td>ASR Drift</td><td>Avg. Score</td></tr><tr><td>Flow Duration</td><td>5.80</td><td>0.1625</td><td>1.070</td></tr><tr><td>Active Min</td><td>0.00</td><td>0.1850</td><td>0.667</td></tr><tr><td>Flow IAT Std</td><td>3.96</td><td>0.0350</td><td>0.527</td></tr><tr><td>Flow Pkts/s</td><td>1.02</td><td>0.1060</td><td>0.449</td></tr><tr><td>Flow IAT Max</td><td>0.76</td><td>0.1110</td><td>0.448</td></tr></table>

TABLE III: Top 5 Features — Bwd IAT Tot vs Baseline
<table><tr><td>Feature Name</td><td>∆ Drift</td><td>ASR Drift</td><td>Avg. Score</td></tr><tr><td>Bwd Pkts/s</td><td>3.38</td><td>0.0985</td><td>0.914</td></tr><tr><td>Bwd IAT Tot</td><td>3.24</td><td>0.0985</td><td>0.889</td></tr><tr><td>Bwd IAT Max</td><td>0.00</td><td>0.1685</td><td>0.667</td></tr><tr><td>Bwd IAT Std</td><td>0.00</td><td>0.1635</td><td>0.647</td></tr><tr><td>Active Min</td><td>0.00</td><td>0.1350</td><td>0.534</td></tr></table>

Table IV compares the ASR of the original poisoned model with its fine-pruned counterpart across varying poisoning levels for the BruteForce attack. First, the most backdoorrelevant neurons are identified via feature activation analysis. Next, these neurons are pruned, and the model is finetuned on clean, previously unseen data. As summarized in Table IV, this fine-pruning procedure substantially suppresses backdoor behavior, reducing the ASR to near-zero in most configurations, confirming the effective disruption of the internal representations responsible for trigger activation.

A notable exception occurs at the backdoor saturation threshold, the minimal poisoning level required to achieve an initial ASR of 1.0. Specifically, at a poisoning percentage of 0.005%, the fine-pruned model exhibits an anomalously elevated residual ASR of approximately 0.25, rather than the near-complete suppression observed elsewhere.

Overall, fine-pruning achieves up to a 99% reduction in backdoor effectiveness while preserving nominal classification performance. Although it does not guarantee total backdoor removal, it remains a robust, model-centric mitigation strategy for scenarios where original dataset access is restricted.

Fig. 4 illustrates the impact of simultaneously poisoning multiple influential features on ASR and nominal classification accuracy (on clean data). The four bar plots compare the performance of the evaluated model on poisoned (red) and clean (blue) data across the four attack categories: BruteForce, DDoS, DoS, and Botnet. Specifically, the figure highlights the joint poisoning of Flow Duration and Fwd IAT Tot. Unlike the single-feature experiments, the poisoning ratio here denotes the aggregate fraction of modified training samples distributed across the targeted feature set. Across all attack classes, combined-feature poisoning substantially amplifies backdoor effectiveness, as reflected by a consistently higher average ASR at very low poisoning ratios. For the pair Flow Duration and Fwd IAT Tot, the average ASR reaches 0.8 at a poisoning ratio of only 0.002% for the BruteForce attack and achieves a perfect backdoor efficacy of 1.0 at 0.005% for the DDoS and Botnet attacks, demonstrating that highly effective backdoors can be established with only a minimal number of modified samples. At the same time, nominal accuracy on unperturbed data remains unaffected at approximately 0.97, indicating that the poisoning has little impact on benign classification.

TABLE IV: ASR before and after fine-pruning
<table><tr><td rowspan=1 colspan=1>Poisoning Ratio</td><td rowspan=1 colspan=2>Original ASR | Fine-Pruned ASR</td></tr><tr><td rowspan=1 colspan=1>0.001</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>0.002</td><td rowspan=1 colspan=1>0.79</td><td rowspan=2 colspan=1>0.010.26</td></tr><tr><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1>1.00</td></tr><tr><td rowspan=1 colspan=1>0.010</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>0.050</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.01</td></tr></table>

![](images/fd0fa1d48452846dd764b75d753a118352f84f710661a7b289f04bec001b06f4.jpg)

![](images/e5b5f668d3c5ea34850ae802c2b10f70d03a6c55bfe4afdde21df1c36fa84cbf.jpg)

![](images/7ecc14a4e5c3a2fa6f84d7560dd1790f3d1837db70c5713e080ee8b54afb2abf.jpg)

![](images/3875d42a24715edb549521c13a7fda3a4ec14fbb0cf50b44e7c7c77a2e7c8033.jpg)  
Fig. 5: Accuracy of backdoor for a model with two features poisoned (Fwd Pkts/s, Flow Pkts/s)

Analysis of a second feature combination (Fig. 5) further underlines this observation. When Fwd Pkts/s and Flow Pkts/s are poisoned jointly, the average ASR across attacks rapidly approaches 1.0 even at the lowest evaluated poisoning levels. This indicates that combining features that are individually less dominant can nonetheless result in highly reliable backdoors when manipulated together, revealing a clear synergistic effect between the features.

## V. CONCLUSION

This work demonstrates that flow-based IDSs are highly vulnerable to training-time poisoning attacks, where modifying as little as 0.001% of the training data suffices to implant effective and stealthy backdoors without degrading clean-data performance. Even single-feature manipulations, particularly of temporal flow features, reliably induce targeted misclassification while preserving nominal performance under standard evaluation metrics, making such attacks difficult to detect in operational environments.

To address this, we propose a comparative, model-centric methodology that detects feature-level backdoors by analyzing behavioral drift between a potentially poisoned model and a distilled baseline. This enables reliable localization of trigger-relevant features and interpretable feature ranking, even when single-model perturbation analysis is inconclusive. Furthermore, fine-pruning demonstrates that model-centric mitigation can substantially reduce backdoor effectiveness without requiring access to the original training data, although complete removal is not guaranteed.

The method incurs additional computational cost due to perturbation analysis and baseline training but is suitable for offline diagnostics during periodic retraining. While evaluated on a single dataset and model family, the approach is modelagnostic and expected to generalize. Extending evaluation across datasets, model architectures, and attack classes remains future work.

## REFERENCES

[1] F. Spelter, J. F. Loevenich, T. Hürten, E. Adler, and R. R. F. Lopes, “Quantifying the impact of poisoning attacks in deep neural networks for attack classification," in MILCOM 2025 - 2025 IEEE Military Communications Conference (MILCOM), 2025, pp. 1123–1129.

[2] M. Goldblum, D. Tsipras, C. Xie, X. Chen, A. Schwarzschild, D. Song, A. Madry, B. Li, and T. Goldstein, "Dataset security for machine learning: Data poisoning, backdoor attacks, and defenses,"IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 2, pp. 1563–1580, 2023.

[3] C. S. Establishment and C. I. for Cybersecurity, “A realistic cyber defense dataset (cse-cic-ids2018),"https://registry.opendata.aws/cse-cic-ids2018/, 2025, accessed: 2025-10-29.

[4] B. Biggio, B. Nelson, and P. Laskov, "Poisoning attacks against support vector machines," in Proceedings of the 29th International Conference on Machine Learning (ICML), 2012.

[5] T. Gu, B. Dolan-Gavitt, and S. Garg, “Badnets: Identifying vulnerabilities in the machine learning model supply chain," arXiv, pp. 1–13, 2017.

[6] A. Shafahi, W. R. Huang, M. Najibi, O. Suciu, C. Studer, J. Z. Kolter, and T. Goldstein, "Poison frogs: Targeted clean-label poisoning attacks on neural networks," in Advances in Neural Information Processing Systems (NeurIPS), 2018.

[7] A. Turner, D. Tsipras, and A. Madry, "Label-consistent backdoor attacks," in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshop, 2019.

[8] B. Wang, Y. Yao, S. Shan, H. Li, B. Viswanath, H. Zheng, and B. Y. Zhao, "Neural cleanse: Identifying and mitigating backdoor attacks in neural networks," in 2019 IEEE symposium on security and privacy (SP). IEEE, 2019, pp. 707–723.

[9] S. Songma, T. Sathuphan, and T. Pamutha, "Optimizing intrusion detection systems in three phases on the cse-cic-ids-2018 dataset," Computers, vol. 12, no. 12, 2023.

[10] L. Göcs and Z. C. Johanyák, “"Identifying relevant features of cse-cicids2018 dataset for the development of an intrusion detection system," Intelligent Data Analysis, vol. 28, no. 6, pp. 1527–1553, 2024.

[11] Z. Zhang, Y. Zhang, D. Guo, L. Yao, and Z. Li, "Secfednids: Robust defense for poisoning attack against federated learning-based network intrusion detection system," Future Generation Computer Systems, vol. 134, pp. 154–169, 2022.

[12] T. T. Thein, Y. Shiraishi, and M. Morii, "Personalized federated learningbased intrusion detection system: Poisoning attack and defense," Future Generation Computer Systems, vol. 153, pp. 182–192, 2024.

[13] H. Zhuang, M. Yu, H. Wang, Y. Hua, J. Li, and X. Yuan, "Backdoor federated learning by poisoning backdoor-critical layers," arXiv preprint arXiv:2308.04466, 2023.

[14] P. Rieger, T. Krauß, M. Miettinen, A. Dmitrienko, and A.-R. Sadeghi, “Crowdguard: Federated backdoor detection in federated learning," arXiv preprint arXiv:2210.07714, 2022.

[15] J. Gou, B. Yu, S. J. Maybank, and D. Tao, "Knowledge distillation: A survey," International journal of computer vision, vol. 129, no. 6, pp. 1789–1819, 2021.