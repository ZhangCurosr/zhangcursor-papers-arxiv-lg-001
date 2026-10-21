# Characterizing Cross-Relay Generalization Failure in Machine Learning-Based Power System Protection

Nithin Pradeep Nadayil Kizhakkethil<sup>1,\*</sup>, Julian Oelhaf<sup>1</sup>, Christian Bergler<sup>2</sup>, Andreas Maier<sup>1</sup>, Siming Bayer<sup>1</sup> <sup>1</sup>Pattern Recognition Lab, Friedrich-Alexander-Universitat Erlangen-N ¨ urnberg¨

<sup>2</sup>Department of Electrical Engineering, Media and Computer Science, Ostbayerische Technische Hochschule Amberg-Weiden <sup>\*</sup>Corresponding author: nithin.pradeep.nadayil@fau.de

Abstract—Machine learning-based fault detectors are often evaluated at the relay locations used for training, although local voltage and current measurements vary across power networks. This work investigates cross-relay generalization for binary detection of faults anywhere in the network using local measurements, rather than line-selective tripping. Using 50 ms windows from a simulated 110 kV transmission network, we evaluate a multilayer perceptron across all 64 source–target combinations of eight relays and five random seeds. A shared simulation-level split and source-only threshold selection provide a controlled heldout evaluation without target-specific calibration. Despite nearly unchanged mean F1-scores (0.730 within-relay versus 0.734 crossrelay), the mean false-positive rate (FPR) increases from 0.088 to 0.341, while the mean false-negative rate decreases from 0.399 to 0.318. The change is strongly directional: models trained at Bus 2 relays yield a mean FPR of 0.670 at Bus 1 and Bus 3 relays, compared with 0.096 for cross-relay transfers within the same bus group. Linear probing indicates that learned representations retain coarse bus-group information, while exploratory classconditional latent distribution analysis shows no consistent monotonic association with transfer errors. These findings demonstrate that aggregate metrics can conceal location-structured changes in false-alarm behavior, highlighting the importance of evaluating error types separately when assessing cross-relay generalization.

Index Terms—Power system protection, fault detection, crossrelay generalization, distribution shift, multilayer perceptron, representation analysis

## I. INTRODUCTION

Power system protection is a fundamental component of modern electrical grids, enabling the rapid detection and isolation of faults to maintain system stability, equipment safety, and operational reliability. Protection relays (PRs) continuously monitor local electrical quantities, such as phase voltages and currents, and initiate protective actions when abnormal operating conditions are detected. With the increasing deployment of digital substations and standardized communication protocols such as IEC 61850, large volumes of high-resolution relay measurements have become available, creating new opportunities for machine learning (ML)-based protection systems [1], [2].

ML has emerged as a promising approach for fault detection in power system protection [3], [4]. Recent studies have investigated ML-based methods for fault detection, fault classification, and fault localization using electrical measurements, including multilayer perceptron (MLP)-based approaches for transmission-line protection [5], [6]. A broad range of learning architectures has been explored in power system protection, including conventional ML methods and deep learning models such as MLPs, Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), and topology-aware Graph Neural Networks (GNNs) [7], [8]. While these approaches often report strong performance under their respective evaluation settings, their ability to generalize across different measurement locations remains less clearly characterized.

However, such evaluations often implicitly assume comparable measurement distributions across training and deployment locations. In practical power systems, this assumption may not hold. PRs are installed at different locations throughout the transmission network, and the electrical signals observed at each relay depend on their position relative to the fault, network topology, and power flow conditions. Consequently, measurements recorded at different relay locations exhibit distinct statistical characteristics, while the asymmetric changes in classification errors under pairwise cross-relay transfer remain insufficiently characterized for relay-local models. In particular, it remains unclear which error type dominates, whether aggregate metrics reveal this behavior, and whether the failure exhibits topology-related structure.

Although generalization under distribution shift has received increasing attention in machine learning [9], [10], relatively little attention has been given to cross-relay generalization in power system protection. Existing work primarily evaluates models using random train-test splits or fixed measurement locations, making it difficult to determine whether the learned representations capture relay-independent fault characteristics or simply memorize relay-specific measurement distributions.

Although topology-aware approaches have demonstrated generalization to unseen grids without retraining [8], the transfer behavior of models trained using measurements from individual relay locations remains less well characterized. This work therefore investigates pairwise cross-relay transfer using a controlled MLP baseline, with particular emphasis on asymmetric changes in false-positive and false-negative rates.

Answering this requires more than overall classification

Protection Relay

![](images/e78df04700392a8d5827185f0073b2a63405b89862e6c636b5f61d03fb0c55fe.jpg)  
Fig. 1. Schematic of the Double Line grid topology. The diagram illustrates the transmission lines, the buses, and PRs. Measurements are recorded at the PRs and transmitted to the substations at each bus.

performance: accuracy or F1-score may change little while the types of errors change substantially, and the learned representations may still encode relay-location information. We therefore investigate cross-relay generalization using exhaustive pairwise transfer experiments between all transmission-line PRs, examining overall performance, false-positive and falsenegative behavior, and relay-location structure in the learned representations. The objective is not to assume that transfer should be invariant, but to determine how and to what extent generalization fails, and whether the failures show systematic structure related to relay location and network topology.

## II. SYSTEM SETUP AND DATA

## A. EMT Simulation Dataset

The experiments in this work are conducted using the AdaptGrid subset of the EvEMTBench electromagnetic transient (EMT) simulation dataset generated in DIgSILENT PowerFactory [2]. The dataset is based on a 110 kV double-line transmission network consisting of three buses interconnected by four transmission lines, as illustrated in Fig. 1. The network is instrumented with 11 measurement locations, comprising eight transmission-line PRs, two external-grid measurement points, and one load measurement point. Each PR records synchronized three-phase voltage and current measurements [2].

Each measurement location records three-phase voltages and currents sampled at 9.6 kHz. The six electrical quantities measured at location m are represented as

$$
\begin{array} { r } { \mathbf { x } _ { m } ( t ) = \left[ I _ { A } ( t ) \quad I _ { B } ( t ) \quad I _ { C } ( t ) \quad V _ { A } ( t ) \quad V _ { B } ( t ) \quad V _ { C } ( t ) \right] ^ { \top } , } \end{array}
$$

where I and V denote the three-phase currents and voltages, respectively. Concatenating the measurements from all eleven locations yields a 66-dimensional feature vector for each simulation time step.

The dataset incorporates significant operating-condition variability through randomization of loading conditions, source characteristics, fault locations, inception angles, and fault impedances. This produces a diverse collection of operating scenarios that enables a comprehensive evaluation of machine learning models under varying network conditions.

## B. Fault Scenarios and Class Distribution

The experiments use fault-event simulations from the AdaptGrid subset of EvEMTBench [2], excluding switching events. The dataset contains 4,667 simulations and 107,341 windows per relay, comprising 73,000 fault (68.01%) and 34,341 non-fault (31.99%) windows.

The binary label $\mathtt { Y \_ f a u l t \_ l }$ present denotes the global network fault state. Empirical verification showed that all labels agree with an event-overlap ratio threshold of $1 / 2 4 0 \colon$ windows with overlap\_rat $\mathrm { i } \mathrm { o } \ \geq \ 1 / 2 4 0$ are labeled as faults. Positive labels include fault initiation, ongoing faults, and fault clearing, rather than fault onset alone.

The dataset covers seven fault categories, including single-line-to-ground, line-to-line, double-line-to-ground, three-phase, and incipient faults, with varying fault locations and operating conditions. The task therefore concerns global fault-state detection from local relay measurements, not conventional zone-selective tripping.

## C. Window Extraction and Preprocessing

Following the sliding-window preprocessing approach in [4], each EMT simulation is segmented into fixed-length overlapping windows of synchronized three-phase voltage and current measurements.

Let

$$
{ \mathbf X } = \{ { \mathbf x } ( t ) \} _ { t = 1 } ^ { T } ,
$$

denote the multivariate signal of a single simulation, where

$$
\mathbf { x } ( t ) \in \mathbb { R } ^ { 6 6 } .
$$

A window beginning at time index k is defined as

$$
\mathbf { W } _ { k } = \left[ \begin{array} { c } { \mathbf { x } ( k ) } \\ { \mathbf { x } ( k + 1 ) } \\ { \vdots } \\ { \mathbf { x } ( k + L - 1 ) } \end{array} \right] \in \mathbb { R } ^ { L \times 6 6 } ,
$$

where $L = 4 8 0$ samples corresponds to a duration of 50 ms at a sampling frequency of 9.6 kHz. Consecutive windows are extracted every 5 ms, producing overlapping windows that capture the temporal evolution of the fault and provide a new input window every 5 ms.

Each extracted sample is therefore represented as

$$
\mathbf { W } _ { k } \in \mathbb { R } ^ { 4 8 0 \times 6 6 } .
$$

Since this work investigates relay generalization, only the eight transmission-line PRs are considered in the subsequent experiments. For each transmission-line PR, indexed by $r \in \{ 1 , \ldots , 8 \}$ , only the six locally measured channels are extracted from the global window $\mathbf { W } _ { k }$ , yielding the relayspecific input

$$
\mathbf { W } _ { k } ^ { ( r ) } \in \mathbb { R } ^ { 4 8 0 \times 6 } .\tag{1}
$$

where r denotes the relay under consideration.

## III. METHODS

This work characterizes cross-relay generalization through three complementary aspects: predictive performance under source-to-target transfer, changes in error behavior (False Positive Rate (FPR) and False Negative Rate (FNR)), and differences in the learned representations across relay locations, analyzed using linear probes and $\mathrm { { \bf M M D } ^ { 2 } }$ . A multilayer perceptron (MLP) serves as a controlled baseline to isolate the influence of relay-location-dependent measurement distributions.

## A. Multilayer Perceptron

The MLP operates on minimally processed, raw relay-local measurements without input normalization or hand-crafted feature engineering. As defined in (1), each input window of raw three-phase voltages and currents is flattened into a feature vector

$$
\mathbf { z } \in \mathbb { R } ^ { 2 8 8 0 } .
$$

It is processed by two fully connected hidden layers with 256 and 128 neurons, each followed by a ReLU activation. A final linear output neuron produces a fault logit, converted to a probability by a sigmoid function during inference. The network has approximately 770,561 trainable parameters. As this architecture is a controlled baseline rather than a representative protection model, the magnitude of the observed degradation should not be generalized to other architectures.

## B. Training Settings

The MLP is trained with binary cross-entropy with logits and the Adam optimizer (learning rate $1 . 3 8 \times 1 0 ^ { - 4 }$ , no weight decay), using a batch size of 1024 for at most 200 epochs. Early stopping with a patience of 20 epochs is applied, and the parameters from the best validation-loss epoch are retained.

After training, the operating threshold is selected on the source relay’s validation set by maximizing Youden’s J statistic,

$$
J ( t ) = \mathrm { T P R } ( t ) - \mathrm { F P R } ( t ) .
$$

The threshold is then frozen and used for the source relay and all target relays, so no target-relay information influences the operating point. ROC AUC is computed from continuous raw model logits and is independent of this threshold.

Performance is reported as accuracy, F1-score, ROC AUC, FPR, and FNR, with the fault class as positive. FPR is the fraction of non-fault windows classified as faults, and FNR is the fraction of fault windows classified as non-faults.

## C. Pairwise Cross-Relay Evaluation

Each of the eight transmission-line protection relays serves as a source for an independent model, which is evaluated at the source relay and at each of the seven target relays. This yields an $8 \times 8$ transfer matrix of 64 pairs: eight self evaluations on the diagonal and 56 ordered cross-relay evaluations off the diagonal.

A shared simulation-level split, generated with seed 42, is reused across all eight relays. It comprises 3,266 training, 700 validation, and 701 test simulations, corresponding to 75,118, 16,100, and 16,123 windows per relay, respectively. All windows originating from the same simulation remain in the same partition. Self- and cross-relay evaluations use the same held-out test simulations.

The complete experiment is repeated for five random seeds (42, 5, 10, 25, and 100), which affect model initialization and training but not the fixed simulation-level partition. Results are reported as mean ± standard deviation across seeds.

To examine location structure, the 56 ordered cross-relay pairs are divided by bus group (Bus 1: R1, R2; Bus 2: R3– R6; Bus 3: R7, R8) into within-group pairs (16), Bus 1–Bus 3 pairs (8), pairs from Bus 2 to Bus 1 or Bus 3 (16), and pairs from Bus 1 or Bus 3 to Bus 2 (16). Mean FPR and FNR are computed per category from the five-seed mean transfer matrices.

## D. Representation Analysis

Representations are taken from the 128-dimensional penul timate layer of each source-trained MLP of the seed-42 run. All representation analyses use one fixed study subset, drawn once and embedded by every source-trained model, so that differences between models are not caused by different input samples. For each of the eight relays, 150 fault and 150 non-fault windows are sampled uniformly at random without replacement with seed 42 from all non-switching windows of that relay, giving 300 windows per relay and 2,400 in total. The subset is not restricted to a data partition and therefore contains windows from training, validation, and test simulations; it comprises 1,878 distinct simulations. Sampling is class-balanced and performed independently for each relay, so relays are not matched on simulation or window position.

Relay identity is predicted from the representations with multinomial logistic regression $( \ell _ { 2 }$ penalty, $\textit { C } = \textit { 1 }$ on standardized features, and bus-group identity (Bus 1, Bus $^ { 2 , }$ or Bus 3) with a separate probe of the same type. Because overlapping windows from one simulation are strongly dependent, both probes are evaluated with five-fold stratified crossvalidation grouped by simulation, so that all windows from the same EMT simulation fall in the same fold. The chance level is $1 / 8$ for relay identification. Because Bus 2 contains four relays and Bus 1 and Bus 3 two each, the majority-class baseline for bus-group identification is 0.50. To quantify the relayidentification result, the domain confusion score is computed from the mean out-of-fold accuracy $A _ { \mathrm { d o m a i n } }$ and $A _ { \mathrm { c h a n c e } } = 1 / 8 \colon$

$$
C = 1 - \frac { A _ { \mathrm { d o m a i n } } - A _ { \mathrm { c h a n c e } } } { 1 - A _ { \mathrm { c h a n c e } } } ,\tag{2}
$$

where $C = 1$ corresponds to chance-level relay identifica tion and $C = 0$ to perfect identification.

Differences between source and target representations are quantified with the squared maximum mean discrepancy (MMD<sup>2</sup>) [11]. Let $\{ \mathbf { z } _ { i } ^ { s } \} _ { i = 1 } ^ { n }$ and $\{ \mathbf { z } _ { j } ^ { t } \} _ { j = 1 } ^ { m }$ denote the source-

and target-relay representations. The empirical estimator is given in (3).

$$
\begin{array} { r l r } {  { \widehat { \mathrm { M M D } } ^ { 2 } = \frac { 1 } { n ^ { 2 } } \sum _ { i = 1 } ^ { n } \sum _ { i ^ { \prime } = 1 } ^ { n } k ( { \mathbf z } _ { i } ^ { s } , { \mathbf z } _ { i ^ { \prime } } ^ { s } ) } } \\ & { } & { + \frac { 1 } { m ^ { 2 } } \sum _ { j = 1 } ^ { m } \sum _ { j ^ { \prime } = 1 } ^ { m } k ( { \mathbf z } _ { j } ^ { t } , { \mathbf z } _ { j ^ { \prime } } ^ { t } ) } \\ & { } & { - \frac { 2 } { n m } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { m } k ( { \mathbf z } _ { i } ^ { s } , { \mathbf z } _ { j } ^ { t } ) , } \end{array}\tag{3}
$$

with the RBF kernel $k ( \mathbf { z } , \mathbf { z } ^ { \prime } ) = \exp ( - \| \mathbf { z } - \mathbf { z } ^ { \prime } \| _ { 2 } ^ { 2 } / ( 2 \sigma ^ { 2 } ) )$ . The bandwidth is set per source model by the median heuristic on the 300 source-relay embeddings and kept fixed for all target relays of that source. $\mathrm { { \bf M M D } ^ { 2 } }$ is computed separately per class with $n = m = 1 5 0 !$ between non-fault windows, which is related to the cross-relay FPR, and between fault windows, which is related to the cross-relay FNR. This gives 56 ordered source-target pairs per class. The associations with the seed-42 error rates are quantified with Pearson (linear) and Spearman (monotonic) correlation and are interpreted descriptively.

## IV. RESULTS

## A. Overall Cross-Relay Performance

Table I summarizes self- and cross-relay performance across five seeds. Cross-relay transfer reduces accuracy (0.7010 to 0.6747) and ROC AUC (0.7929 to 0.7493). The change in error profile is larger: mean FPR rises from 0.0879 to 0.3406, while mean FNR falls from 0.3986 to 0.3181. Cross-relay degradation is therefore not a uniform increase in errors but a shift towards false positives. Because the threshold is fixed on the source validation set, this shift reflects the model’s behavior under the target measurement distribution, not targetspecific re-tuning.

The F1-score is nearly unchanged (0.7302 to 0.7336) because the rise in false positives is offset by the fall in false negatives. Aggregate metrics such as F1 can therefore hide substantial changes in error behavior.

Averages also conceal severe individual failures. The worst source-target pair in each seed averages an FPR of 0.9081 across seeds. The worst-pair FNR (0.5727) exceeds the mean self FNR, so the decrease in FNR holds on average but not for every pair.

## B. Error Behavior Across Relay Pairs

Fig. 2 shows the pairwise FPR and FNR matrices, averaged over five seeds. False-positive behavior depends strongly on the source-target combination. For models trained at Bus 2 relays, the mean FPR at Bus 1 and Bus 3 relays ranges from 0.497 to 0.807, whereas transfer within a bus group and between Bus 1 and Bus 3 remains close to the self-relay level. The FNR matrix shows a different pattern, and pairs with high FPR do not have high FNR. Together with the aggregate results, this indicates that the main deterioration under crossrelay transfer is the increase in false-positive rate.

TABLE I  
SELF-RELAY, CROSS-RELAY, AND WORST-PAIR PERFORMANCE ACROSS FIVE SEEDS (MEAN ± STANDARD DEVIATION). WORST CROSS DENOTES THE WORST INDIVIDUAL SOURCE-TARGET PAIR FOR EACH METRIC IN EACH SEED.
<table><tr><td>Metric</td><td>Self</td><td>Cross</td><td>Worst Cross</td></tr><tr><td>Accuracy</td><td> $0 . 7 0 1 0 \pm 0 . 0 0 6 9$ </td><td> $0 . 6 7 4 7 \pm 0 . 0 0 5 9$ </td><td> $0 . 5 7 3 4 \pm 0 . 0 2 3 2$ </td></tr><tr><td>F1</td><td> $0 . 7 3 0 2 \pm 0 . 0 0 9 5$ </td><td> $0 . 7 3 3 6 \pm 0 . 0 0 9 8$ </td><td> $0 . 5 7 4 7 \pm 0 . 0 4 4 1$ </td></tr><tr><td>ROC AUC</td><td> $0 . 7 9 2 9 \pm 0 . 0 0 5 8$ </td><td> $0 . 7 4 9 3 \pm 0 . 0 0 7 0$ </td><td> $0 . 6 8 0 9 \pm 0 . 0 0 5 2$ </td></tr><tr><td>FPR</td><td> $0 . 0 8 7 9 \pm 0 . 0 1 3 4$ </td><td> $0 . 3 4 0 6 \pm 0 . 0 3 7 9$ </td><td> $0 . 9 0 8 1 \pm 0 . 0 9 0 8$ </td></tr><tr><td>FNR</td><td> $0 . 3 9 8 6 \pm 0 . 0 1 5 9$ </td><td> $0 . 3 1 8 1 \pm 0 . 0 2 5 6$ </td><td> $0 . 5 7 2 7 \pm 0 . 0 5 6 8$ </td></tr></table>

TABLE II

MEAN CROSS-RELAY FPR AND FNR BY BUS-GROUP RELATIONSHIP OF SOURCE AND TARGET RELAY. VALUES ARE AVERAGES OVER THE RELAY PAIRS IN EACH CATEGORY, COMPUTED FROM THE FIVE-SEED MEAN TRANSFER MATRICES (FIG. 2).
<table><tr><td>Source → target</td><td>Pairs</td><td>FPR</td><td>FNR</td></tr><tr><td>Within bus group</td><td>16</td><td>0.096</td><td>0.385</td></tr><tr><td>Bus 1 ↔ Bus 3</td><td>8</td><td>0.132</td><td>0.405</td></tr><tr><td> $\operatorname { B u s } 2  \operatorname { B u s } 1 / 3$ </td><td>16</td><td>0.670</td><td>0.166</td></tr><tr><td>Bus  $1 / 3 \to \mathrm { B u s } ~ 2$ </td><td>16</td><td>0.360</td><td>0.361</td></tr></table>

The degradation is also structured by location. Table II quantifies this structure. Models trained at Bus 2 relays show a mean FPR of 0.670 at Bus 1 and Bus 3 relays. Transfer within a bus group (0.096) and between Bus 1 and Bus 3 (0.132) stays close to the mean self-relay level of 0.088. The reverse direction, from Bus 1 or Bus 3 to Bus 2, is elevated but weaker (0.360). The FNR shows the complementary behavior: Bus 2-trained models rarely miss faults at Bus 1 and Bus 3 (0.166), consistent with their tendency to flag windows there as faulty. Source relays with the lowest mean cross-relay FPR (R2, R7, R8) also show the highest mean cross-relay FNR, which indicates a more conservative operating point and not better transfer. Cross-relay failure is therefore not a random loss of performance between arbitrary relays. The task uses the global fault-state label from local measurements, so these results characterize cross-location generalization and not conventional zone-selective relay operation.

## C. Representation Analysis

On the class-balanced study subset, individual relay identity is only weakly recoverable from the penultimate-layer representation under simulation-grouped cross-validation. The mean out-of-fold accuracy across the eight source-trained models is 0.266, compared with a chance level of 0.125. Applying (2) gives

$$
C = 1 - { \frac { 0 . 2 6 6 - 0 . 1 2 5 } { 1 - 0 . 1 2 5 } } = 0 . 8 3 9 .
$$

The representation is therefore not strongly discriminative at the individual-relay level, but it is also not fully relay-invariant.

Coarser location information is more recoverable. The busgroup probe reaches a mean out-of-fold accuracy of 0.709 across the eight models, compared with 0.50 for always predicting Bus 2. The confusion matrix in Fig. 3 shows that Bus 1 measurements are mostly predicted as Bus 1 or Bus 3, and Bus 2 measurements are mostly predicted as Bus 2. This coarse structure is consistent with the transfer pattern: Bus 1 and Bus 3 measurements are confused with each other, and transfer between Bus 1 and Bus 3 relays stays close to the selfrelay FPR (0.132), whereas Bus 2 measurements are separated from both and Bus 2-trained models fail at Bus 1 and Bus 3 relays (FPR 0.670). The probe is symmetric and cannot explain why the reverse direction is weaker. The analysis is diagnostic, not causal: retained location information does not by itself show that it causes the degradation.

![](images/2e6ec6c8069121a22cbb0934a9199d81e3f95dfba93d351b6241a0c68519533f.jpg)

![](images/9e7f0f8f48d534f031f6874995ad627286b0fec7a14c61f06e48ede54d0b5d97.jpg)  
Fig. 2. Pairwise cross-relay transfer matrices for the MLP baseline (mean over five seeds). Rows denote the training relay and columns the test relay; diagonal entries are within-relay evaluation. (a) FPR. (b) FNR.

## D. Representation Distribution Differences and Error

$\mathbf { M M D ^ { 2 } }$ is computed per class and related to the matching error rate over the 56 ordered pairs of the seed-42 run. Between non-fault windows, $\mathrm { { \bf M M D } ^ { 2 } }$ has a Pearson correlation of $r =$ −0.384 and a Spearman correlation of $\rho = - 0 . 1 9 0$ with crossrelay FPR. Between fault windows, the corresponding values with FNR are $r = 0 . 3 4 9$ and $\rho = 0 . 0 4 5$ . The FPR association is negative, so larger latent distances are associated with lower false-positive rates, opposite to the naive expectation that greater shift means worse transfer. The rank correlations are weak, so the linear associations do not indicate a robust monotonic relationship.

The observed MMD<sup>2</sup> values indicate differences between relay-specific latent distributions, but show no consistent monotonic association with cross-relay transfer errors. Source and target samples with similar aggregate distributions can still lie differently relative to the decision boundary, so $\mathbf { M M D ^ { 2 } }$ is best treated as a diagnostic of representation shift and not as an explanation of generalization failure. A possible explanation for the negative FPR association is that a source model that flags most non-fault target windows as faults maps them into a compact region of its latent space, which can reduce the measured $\mathrm { { \bf M M D } ^ { 2 } }$ even though the transfer error is large. This hypothesis is not tested here. The correlations come from a single seed, the 56 ordered pairs share source models and target relays and are therefore not independent, and the bandwidth is source-specific, so $\mathrm { { \bf M M D } ^ { 2 } }$ values are not strictly comparable across sources. The correlations are therefore reported descriptively.

![](images/c603fca6f36d5c1cc42d503fa73b9a4265c8a9effd5ab840fa84160200a39477.jpg)  
Fig. 3. Linear probe confusion matrix for bus-group identification (Bus 1, Bus 2, Bus 3) from the penultimate-layer activations of the MLP trained on relay R1, shown as an illustrative example (grouped five-fold cross-validation, out-of-fold predictions). The confusion pattern shows that coarse bus-group information remains in the representation.

## V. CONCLUSION

This work characterized how an MLP fault detector trained at one relay location behaves at other locations, using exhaustive pairwise transfer across eight transmission-line PRs in a simulated 110 kV network, with a shared simulation-level split and a source-validation threshold. Overall degradation is moderate: mean accuracy falls from 0.7010 to 0.6747 and ROC AUC from 0.7929 to 0.7493. The error profile changes far more. Mean FPR rises from 0.0879 to 0.3406, with the perseed worst relay pair averaging 0.908, while mean FNR falls from 0.3986 to 0.3181, so the F1-score is nearly unchanged (0.7302 to 0.7336). Cross-relay transfer thus shifts the balance towards false positives, and aggregate metrics can hide this. The FNR decrease holds only on average, since the worst-pair FNR (0.573) exceeds the mean self value.

The degradation is structured by location: it depends on the source-target pair, and high FPR occurs mainly when models trained at Bus 2 relays are applied at Bus 1 or Bus 3 relays. It is weaker in the reverse direction and stays close to the self-relay level between Bus 1 and Bus 3. Linear probing shows that individual relay identity is only weakly recoverable, but coarse bus-group information remains in the representation, consistent with this pattern. In a single-seed, descriptive analysis, class-conditional MMD<sup>2</sup> correlates negatively with FPR $( r ~ = ~ - 0 . 3 8 , ~ \rho ~ = ~ - 0 . 1 9 )$ and positively with FNR (r = 0.35, $\rho = 0 . 0 4 )$ . The FPR sign is opposite to the naive expectation that greater shift means worse transfer, and the rank correlations are weak. MMD<sup>2</sup> is therefore a diagnostic of representation shift, not an explanation of transfer error.

These results motivate evaluating cross-relay generalization with FPR and FNR reported separately. The study is limited to one simulated network, one baseline architecture, and a global fault-state label, so the magnitudes should not be generalized. Future work should test whether domain adaptation and invariant representation learning [12] reduce the location dependence, and validate on additional network configurations and measured field data.

## ACKNOWLEDGMENT

This project was funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - 535389056.

The authors gratefully acknowledge the scientific support and high-performance computing resources provided by the Erlangen National High Performance Computing Center (NHR@FAU) at Friedrich-Alexander-Universitat Erlangen-¨ Nurnberg (FAU) under NHR project b304dc. NHR funding is¨ provided by the German federal and Bavarian state authorities.

## REFERENCES

[1] M. A. Aftab, S. M. S. Hussain, I. Ali, and T. S. Ustun, “IEC 61850 based substation automation system: A survey,” International Journal of Electrical Power & Energy Systems, vol. 120, p. 106008, Sep. 2020.

[2] G. Kordowich, J. Loebel, J. Oelhaf, A. Maier, S. Bayer, C. Bergler, and J. Jaeger, “A simulation based dataset of faults and events for machine learning in power systems,” arXiv preprint arXiv:2608.19777, 2026. [Online]. Available: https://arxiv.org/abs/2608.19777

[3] A. Abdullah, “Ultrafast Transmission Line Fault Detection Using a DWT-Based ANN,” IEEE Transactions on Industry Applications, vol. 54, no. 2, pp. 1182–1193, Mar. 2018.

[4] J. Oelhaf, G. Kordowich, P. A. Perez-Toro, T. Arias-Vergara, A. Maier,´ J. Jager, and S. Bayer, “A Systematic Evaluation of Machine Learning¨ Methods for Fault Detection and Line Identification in Electrical Power Grids,” in ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, Apr. 2025, pp. 1–5.

[5] L. Tran, “Integration of neural network and distance relay to improve the fault localization on transmission lines,” Turkish Journal of Electrical Engineering and Computer Sciences, vol. 31, no. 3, pp. 566–580, 2023.

[6] G. K. N, J. Raj, and L. R. Chandran, “Transmission Line Monitoring and Protection with ANN aided Fault Detection, Classification and Location,” in 2021 2nd International Conference on Smart Electronics and Communication (ICOSEC). IEEE, Oct. 2021, pp. 883–889.

[7] J. Oelhaf, G. Kordowich, M. Pashaei, C. Bergler, A. Maier, J. Jager,¨ and S. Bayer, “A Scoping Review of Machine Learning Applications in Power System Protection and Disturbance Management,” International Journal of Electrical Power & Energy Systems, vol. 172, p. 111257, Nov. 2025.

[8] G. Kordowich, J. Oelhaf, A. Maier, S. Bayer, and J. Jager, “A Graph¨ Neural Network-Based Approach for Power System Protection,” in 2025 IEEE Kiel PowerTech. IEEE, Jun. 2025, pp. 1–6.

[9] I. Gulrajani and D. Lopez-Paz, “In search of lost domain generalization,” in International Conference on Learning Representations, 2021.

[10] P. W. Koh, S. Sagawa, H. Marklund, S. M. Xie, M. Zhang, A. Balsubramani, W. Hu, M. Yasunaga, R. L. Phillips, I. Gao, T. Lee, E. David, I. Stavness, W. Guo, B. Earnshaw, I. Haque, S. M. Beery, J. Leskovec, A. Kundaje, E. Pierson, S. Levine, C. Finn, and P. Liang, “WILDS: A benchmark of in-the-wild distribution shifts,” in Proceedings of the 38th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, M. Meila and T. Zhang, Eds., vol. 139. PMLR, Jul. 2021, pp. 5637–5664.

[11] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Scholkopf, and A. Smola,¨ “A kernel two-sample test,” Journal of Machine Learning Research, vol. 13, no. 25, pp. 723–773, 2012.

[12] Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. March, and V. Lempitsky, “Domain-adversarial training of neural networks,” Journal of Machine Learning Research, vol. 17, no. 59, pp. 1–35, 2016.