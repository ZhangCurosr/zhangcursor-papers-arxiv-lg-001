# Cross-Asset Knowledge Transfer for Early Fault Detection of Data-Scarce Assets Without Target Fault Data

Zekai Wang<sup>1,\*</sup>, Yuxuan Li<sup>2,\*</sup>, Zhangyue Shi<sup>3,+</sup>

<sup>1</sup> Charles F. Dolan School of Business, Fairfield University, Fairfield, CT, United States

<sup>2</sup> School of Economics and Management, Tiangong University, Tianjin, China

<sup>3</sup> Independent Researcher, Chesterfield, MO, United States

\* Zekai Wang and Yuxuan Li contributed equally to this work.

<sup>+</sup> Corresponding author: zhangyue.shi@outlook.com

## Abstract

Reliability management of industrial fleets depends on early fault warnings, yet lifecycle data across similar assets are highly imbalanced: a few assets accumulate run-to-failure histories, while newly installed or rarely failing assets provide only normal-condition data. This paper asks how diagnostic knowledge can be transferred from a data-rich source asset to a data-scarce target asset when no fault data have been observed on the target. We formulate cross-asset early fault detection as a zero-shot domain adaptation problem, in which zero-shot refers to the absence of target fault samples, and propose a framework that requires only normal data from the target. The framework comprises (1) a statistical process control step that labels the source asset’s runto-failure data with an exponentially weighted moving average chart, reducing manual annotation to verifying flagged segments; (2) an Adapted Domain-Adversarial Neural Network (ADANN) that combines domain-adversarial training and a Maximum Mean Discrepancy penalty with an Adapted Classification-Domain (ACD) loss, which anchors the classifier’s normal region to the target’s own normal data; and (3) an ahead-of-time labeling scheme with a prediction horizon that targets incipient faults. Under an exact-classification assumption, a theoretical analysis characterizes the domain-adversarial term for a normal-only target and the alignment state in which source fault features remain separated from target normal features. On a public run-to-failure bearing dataset, ADANN attains the highest mean F1-score among four transfer baselines, with a nominally significant precision gain over the strongest baseline, and its robustness is examined under synthetic noise.

Keywords: Zero-shot domain adaptation; Early fault detection; Domain-adversarial learning; Transfer learning; Condition-based maintenance; Reliability management

## 1 Introduction

Rotating machinery is central to a wide range of industrial systems, and rolling bearings are among its most failure-prone components, accounting for a substantial portion of unplanned downtime and maintenance cost (Randall & Antoni, 2011). Condition-based maintenance addresses this risk by triggering inspection or replacement when monitored signals indicate degradation, rather than on a fixed schedule (Jardine et al., 2006; J. Lee et al., 2015). Its value depends on the detector that issues the warning: an early and reliable alarm opens a planning window for maintenance, a false alarm triggers an unnecessary intervention, and a missed or late detection exposes the system to failure. Vibration-based condition monitoring with machine learning models has therefore become a standard approach for real-time fault detection (R. Zhao et al., 2019).

In practice, however, a fundamental obstacle persists: across a fleet of similar assets, the available lifecycle data are highly imbalanced. A few assets may have complete run-to-failure histories, while many others, in particular newly installed assets or assets that are replaced preventively, operate under prolonged normal conditions with little or no recorded fault data (Lei et al., 2020). This asymmetry severely limits the diagnostic capability of models trained on the data-scarce assets. These are exactly the assets for which a maintenance decision must still be made. It raises a central question for reliability management: how can diagnostic knowledge be transferred from a data-rich source asset to a data-scarce target asset to enable early fault detection when the target has no observed fault data?

Existing research in bearing fault diagnosis has largely followed single-asset modeling paradigms, where detection models are trained and deployed independently for each bearing (Lei et al., 2020). Although transfer learning and domain adaptation techniques have been introduced to mitigate distributional shifts across operating conditions and machines (Guo et al., 2019; Yang et al., 2019), most approaches assume that the target domain contains samples of all health states, labeled or unlabeled, or at least a few labeled fault samples (Wu et al., 2020). Federated and collaborative frameworks primarily emphasize distributed or privacy-preserving learning,

without explicitly addressing detection when the target asset possesses only normal operational data. Moreover, early-stage fault detection at the incipient degradation phase remains insufficiently explored under such data-scarce settings.

These limitations reveal a methodological gap: cross-domain diagnosis methods that rely on target fault samples cannot be trained when the target domain contains zero fault samples and only early-stage normal data. To address this gap, we propose a zero-shot cross-asset framework that transfers diagnostic knowledge from a data-rich source asset to a data-scarce target asset, requiring only the target’s early-stage normal data. In this paper, zero-shot specifically refers to the complete absence of target-domain fault samples, rather than the absence of all target-domain data. The central difficulty is that, without any target fault examples, a model must both align the two domains’ feature distributions and calibrate a fault decision boundary for a target whose faulty behavior was never observed. Our framework addresses this difficulty in three steps. It first derives fault labels on the source asset with a control chart, which reduces manual annotation to the verification of flagged segments. It then transfers the resulting diagnostic knowledge to the target through an adversarial adaptation network whose loss design exploits the limited normal target data. Finally, an ahead-of-time labeling formulation reorients the model toward anticipating incipient degradation, so that an alarm precedes the fault by a prediction horizon that can be used for maintenance planning. Together, these elements make the pipeline operable under the strict constraint that target-domain fault data are entirely unavailable.

The contributions of this work are summarized as follows:

A zero-shot cross-asset formulation. We frame early fault detection on a data-scarce asset as zeroshot domain adaptation, transferring knowledge from a data-rich source asset to a target with only normal data and no fault samples. This is a stricter setting than the few-shot regimes assumed by much of the prior work on cross-domain fault diagnosis.

An Adapted Classification-Domain (ACD) loss with a theoretical characterization. Within ADANN, the ACD loss uses the limited normal target data to anchor the classifier’s normal region to the target, addressing a limitation of standard domain-adversarial and discrepancy-based alignment, which align features at the marginal level but leave the decision boundary fitted to the source. We characterize the equilibrium of the resulting adversarial game under a normal-only target (Theorem 1) and bound the empirical false-alarm rate on the target adaptation windows through the ACD loss, whereas no target-side control of missed detections is possible without target fault data (Remark 1).

An ahead-of-time detection formulation. A sliding-window scheme with a prediction horizon reframes detection as anticipating incipient degradation, so that the model is trained to flag a future labeled fault state.

The framework is evaluated on a publicly available run-to-failure dataset with one source-target pair of deep groove ball bearings of different types operated under different radial loads, complemented by controlled noise experiments. The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 presents the methodology and its theoretical analysis. Section 4 reports the case study, including an illustrative analysis of maintenance implications. Section 5 concludes and discusses limitations and future work.

## 2 Literature Review

## 2.1 Fault Detection in Rotating Machinery

Bearing fault detection and diagnosis have long been central to predictive maintenance for rotating machinery, as bearing failures account for a substantial proportion of unplanned downtime and maintenance costs in industrial systems (Randall & Antoni, 2011). Traditional diagnostic approaches rely on signal-processing techniques such as envelope analysis, wavelet transforms, and spectral kurtosis to extract fault-sensitive features from vibration signals acquired with accelerometers (Lei et al., 2020; Randall & Antoni, 2011). While effective, these methods are typically deployed as standalone analytics on a single asset and offer no mechanism for reusing diagnostic knowledge across assets, an increasingly important capability as fleets of similar machines are instrumented and monitored at scale.

With the increasing availability of high-frequency sensor data, machine learning-based fault detection methods have gained significant attention. Supervised learning methods, including random forests, support vector machines, and deep neural networks, can achieve high diagnostic accuracy when extensive labeled datasets are available (Tang et al., 2020; R. Zhao et al., 2019). Recent studies in this journal have further developed data-driven bearing diagnosis under varying operating conditions (Udmale et al., 2026), graph-based representations of high-frequency vibration series (S. Lee et al., 2024), and robust multi-sensor fusion under missing observations (Yan et al., 2026), and some work has investigated early bearing fault detection using diffusion models (Deng et al., 2026). However, the applicability of supervised methods in real-world industrial settings is often limited, as catastrophic failures are rare and typically avoided through preventive maintenance, so that labeled fault data are scarce on most assets (Lei et al., 2020).

Unsupervised and semi-supervised approaches such as autoencoders, one-class support vector machines, isolation forests, and Gaussian mixture models learn representations of normal operating behavior and identify deviations as anomalies, which makes them attractive when labeled fault data are scarce or unavailable (J. Shi, 2023; Z. Shi et al., 2023; Tang et al., 2020; R. Zhao et al., 2019). Statistical process control (SPC) techniques, particularly univariate and multivariate control charts, have also been integrated with machine learning models to enable interpretable real-time condition monitoring (Tran et al., 2022). These hybrid frameworks leverage the statistical rigor of SPC while exploiting the representation learning capability of deep learning models, allowing for earlier detection of gradual degradation trends and process drifts.

Despite this progress, most existing studies model each asset in isolation and train a dedicated detector per machine, without any mechanism to share diagnostic knowledge across assets. This is precisely the regime that fails when a target asset has accumulated little or no fault history. How diagnostic knowledge can be transferred from a data-rich asset to a data-scarce one is the central motivation of this work and is examined in Sect. 2.2.

## 2.2 Transfer Learning and Domain Adaptation for Cross-Asset Diagnosis

When a target asset has limited fault history, a natural strategy is to transfer diagnostic knowledge from a data-rich source asset. The challenge is distribution shift: vibration data from different bearings, even of the same type, differ in distribution due to manufacturing tolerances, installation, and operating environment, so a model trained on the source degrades when applied directly to the target. Transfer learning (TL) and domain adaptation address this shift by aligning source and target representations, for example through discrepancy-based feature alignment (Gretton et al., 2006; Liu et al., 2022) and domain-adversarial training (Ganin et al., 2016). In machinery diagnostics, such methods have been used to transfer knowledge across operating conditions and machines, including from laboratory bearings to field bearings (Guo et al., 2019; Yang et al., 2019). A recurring limitation, however, is that most TL methods assume access to labeled target data, to a few target fault examples (Wu et al., 2020), or to unlabeled target data that nonetheless cover all health states. None of these is available when the target has only normal data.

Two lines of domain adaptation research are closest to the present setting. Conditional domain adaptation aligns the joint distribution of features and class predictions rather than the feature marginals alone (Long et al., 2018). In this journal, C2DAN extends this idea with entropy weighting and a supervised contrastive loss to mitigate class confusion in cross-domain anomaly detection for distributed fiber optic sensing (Xu et al., 2026). Class-conditional alignment of this kind is designed for target data that, although unlabeled, contain the classes to be aligned; with a normal-only target there is only one target class, and the fault class of the target has no counterpart to align. Partial domain adaptation, in turn, addresses targets whose label space is a subset of the source label space by down-weighting source classes that are absent from the target (Cao et al., 2018). Our training setting resembles this case, because the target contributes only the normal class, but differs at deployment: the class that is absent during adaptation, the target fault, is exactly the class the detector must recognize later. Robustness-oriented work such as MAEFF, which fuses multiple sensors under missing observations (Yan et al., 2026), is complementary, since it addresses incomplete inputs within a labeled domain rather than the absence of target fault labels. Finally, active learning can reduce labeling effort under label scarcity in prognostics (Chen et al., 2025), but it presupposes that informative target samples exist to be queried, which is not the case before the first target fault.

Knowledge distillation (KD) offers a complementary route, in which a teacher model guides a student through soft probability outputs, transferring structural information about decision boundaries rather than raw features (Hinton et al., 2015). This is especially useful under class imbalance and in continual-learning settings, where the student inherits fault-detection capability while resisting catastrophic forgetting (F. Li et al., 2021; Y. Li et al., 2025; Z. Shi et al., 2022). Federated and collaborative learning have likewise been proposed to pool knowledge across machines without sharing raw data (Kairouz et al., 2021; L. Li et al., 2020; Sun et al., 2021; Wang et al., 2022), and are promising for cross-machine diagnosis where a well-instrumented source can support detection on a less-observed target. Their objective, however, is distributed or privacy-preserving optimization, not detection under a complete absence of target fault data.

Across these paradigms, several limitations persist for the problem we target. First, most methods implicitly assume that the data-scarce asset provides both normal and faulty samples, an assumption that frequently fails in practice, so that zero-shot detection from normal target data alone remains open. Second, aligning feature distributions does not by itself calibrate the classifier’s decision boundary to the target. Standard domain-adversarial and discrepancy-based objectives align marginal feature distributions, and when the class proportions of the two domains differ, marginal alignment can even pull source features of one class onto target features of another (H. Zhao et al., 2019). Third, early-stage fault detection is still underexplored under realistic noise and operational variability.

These gaps motivate the framework in Sect. 3, which performs zero-shot fault detection from normal target data alone, anchors the decision boundary to the target via the proposed ACD loss, analyzes the resulting equilibrium under a normal-only target, sharpens incipient-fault sensitivity through an ahead-of-time labeling formulation, and uses SPC-based labeling to reduce manual annotation on the source.

## 3 Research Methodology

## 3.1 Test Rig and Data

This study uses the rolling bearing fatigue test rig and run-to-failure data described by Fan et al. (2024, 2025) (Fig. 1). The data contain run-to-failure experiments for two types of deep groove ball bearings, 6210 and 6011. The test rig comprises five main subsystems: a power system, a control system, a test specimen system, a drive system, and a loading system. A 3 kW electric motor, capable of reaching a maximum speed of 6000 r/min, drives the system. Rotational speed and other operational parameters are configured via a control computer, which regulates the motor speed through a frequency converter.

The test specimen system, shown in Fig. 1(b), consists of a test chamber, three bearing pedestals, four test bearings, a drive shaft, two shaft sleeves, and two tightening nuts. Four deep groove ball bearings are assembled onto the drive shaft in sequence, designated as Bearings 1 to 4 from left to right. These are installed into the three bearing pedestals: Pedestal 1 accommodates Bearing 1, Pedestal 2 accommodates Bearings 2 and 3, and Pedestal 3 accommodates Bearing 4. The assembled unit is then lowered into the test chamber. The loading system, depicted schematically in Fig. 1(c), applies radial and axial loads by adjusting the mass of the weights, thereby simulating different load conditions.

![](images/6406af11a4e4caae3202dcea192dfd26d3b1a217f4b358b20279ca53e4ee5f84.jpg)  
Fig. 1 Rolling bearing fatigue test rig: (a) test rig, (b) photograph of the test specimen system, and (c) schematic diagram of the loading mechanism. Reproduced from Fan et al. (2024, 2025)

Both the 6210 and the 6011 experiments used four healthy deep groove ball bearings. The specifications are provided in Table 1. For the 6210 experiment, the rig ran at a constant rotational speed of 4200 r/min with a radial load of 6.8 kN and an axial load of 1.4 kN. Under these conditions, the bearing operated for approximately 1017 s until an automatic shutdown was triggered by a detected fault. For the 6011 experiment, the rig ran at the same rotational speed of 4200 r/min, with a radial load of 5.1 kN and an axial load of 1.4 kN. The test ran for approximately 3738 s.

Table 1 Specifications of the bearing experiments
<table><tr><td colspan="3">Parameter</td></tr><tr><td>Bearing Type</td><td>6210</td><td>6011</td></tr><tr><td>Inner Diameter</td><td>50 mm</td><td>55 mm</td></tr><tr><td>Outer Diameter</td><td>90 mm</td><td>90 mm</td></tr><tr><td>Width</td><td>20 mm</td><td>18 mm</td></tr><tr><td>Number of Balls</td><td>10</td><td>10</td></tr></table>

Post-test inspection of the 6210 bearing, shown in Fig. 2 (Fan et al., 2024, 2025), revealed severe cage failure, with six of the ten connecting rivets on the cage fractured and three balls displaced from their original positions. This failure mode develops progressively, which makes early warning before the final failure both feasible and valuable for maintenance.

![](images/fa8c4fc7e63c1c30e2311e0a8ddabf04afcb66f4716656a8a469029e13d1b378.jpg)  
Fig. 2 Post-test inspection of the 6210 bearing. Reproduced from Fan et al. (2024, 2025)

Vibration signals were acquired with two PCB-356A01 triaxial accelerometers (Fan et al., 2024, 2025). Accelerometer 1 was mounted on the exterior of Bearing Pedestal 1 and acquired vibration data from Bearing 1. Accelerometer 2 was mounted on the exterior of Bearing Pedestal 2 and acquired vibration data from Bearings 2 and 3. Accelerometer 1 measures acceleration in the Z direction and Accelerometer 2 in the Y and Z directions, where Y denotes the radial direction perpendicular to the shaft axis and Z denotes the radial direction parallel to the shaft axis, which yields three vibration channels. The signals were transmitted to a data acquisition unit, which converted the analog signals to digital format at a sampling frequency of 20 kHz, and were stored on a data analysis computer.

The resulting cross-asset early detection setting is as follows. Vibration signals acquired from the source bearing are first used to learn a diagnostic model that encodes the discriminative structure separating normal from degraded operation. The knowledge transfer architecture of Sect. 3.4 then adapts this model to the target bearing, using only the target’s early-stage normal data to align the two assets’ feature representations and to anchor the decision boundary. The adapted model is deployed on the target bearing’s incoming signal stream to issue early warnings that can inform maintenance decisions for the target.

To demonstrate the cross-asset knowledge transfer architecture, the 6210 experiment is designated as the source domain (Bearing A), providing both normal and faulty data with automatically generated labels, and the 6011 experiment is designated as the target domain (Bearing B). Because the two bearings differ in type and in radial load, the transfer involves both an asset shift and an operating-condition shift, although both experiments were performed on the same rig at the same speed.

## 3.2 Problem Formulation

In practical industrial settings, a target bearing may operate under normal conditions for an extended period, while fault-state data are rarely observed or recorded. Conversely, historical vibration data containing both normal and degraded periods are often available from an analogous source bearing (Bearing A). This setting motivates transferring diagnostic knowledge from the data-rich source bearing to the data-scarce target bearing (Bearing B).

Let $\boldsymbol { x } _ { t } ^ { A } \in \mathbb { R } ^ { C }$ denote the �-channel vibration measurement at time index � from Bearing A, for $t = 1 , \dots , T _ { A }$ . Similarly, $\boldsymbol { x } _ { t } ^ { B } \in \mathbb { R } ^ { C }$ denotes the measurement from Bearing B, for $t = 1 , \dots , T _ { B }$ . We assume that a stage indicator sequence $\{ s _ { t } ^ { A } \} _ { t = 1 } ^ { T _ { A } }$ (and $\{ s _ { t } ^ { B } \} _ { t = 1 } ^ { T _ { B } }$ for evaluation) is available or can be derived, for example from control-chart statistics (Sect. 3.3). A time index � is considered abnormal if $s _ { t } \geq \zeta$ , where $\zeta$ is a predefined abnormality threshold.

To enable early warning, we construct sliding-window samples. Given window length $W _ { ; }$ , stride �, and prediction horizon � (all in sampling points), each sample is indexed by a window start time $i \in \{ 1 , 1 + S , 1 + 2 S , \ldots , T - W - H + 1 \}$ , and the corresponding input is the windowed multichannel segment

$$
X _ { i } = [ x _ { i } , x _ { i + 1 } , \ldots , x _ { i + W - 1 } ] \in \mathbb { R } ^ { C \times W } .\tag{1}
$$

The early-warning label of this window is defined by whether the abnormal stage occurs more than � times within a future interval that starts � sampling points after the window start:

$$
y _ { i } = \mathbb { I } \left( \sum _ { t = i + H } ^ { i + H + W - 1 } \mathbb { I } \left( s _ { t } \geq \zeta \right) > \eta \right) ,\tag{2}
$$

that is, $y _ { i } = 1$ if more than $\eta$ of the time indices in $[ i + H , i + H + W - 1 ]$ are abnormal, and $y _ { i } = 0$ otherwise. This definition matches the operational goal of using evidence observed in the current window to predict whether an abnormal state will be present � sampling points later. In maintenance terms, � defines the future labeling interval relative to the window start; the realized lead time depends on the timing of the model’s alarms.

In the target domain, only a limited amount of early-stage normal data is available for adaptation. Specifically, we assume that for Bearing B we only observe windows whose starts lie in an early normal period, $\mathcal { D } _ { B } ^ { T } = \{ ( X _ { i } ^ { B } , y _ { i } ^ { B } = 0 ) \}$ with window starts in $\{ 1 , \dots , M \}$ , where � is the number of time points observed before any fault appears. We denote the full data of Bearing B by $\mathcal { D } _ { B }$ and the data on which predictions are required by $\mathcal { D } _ { B } ^ { C }$ , the windows starting after time �. We write $N _ { A }$ and $N _ { T }$ for the numbers of source windows and target adaptation windows, respectively. Our objective is to learn a classifier �(⋅) that predicts the probability of an upcoming abnormal state,

$$
p _ { i } = f ( X _ { i } ^ { B } ) \in [ 0 , 1 ] , \qquad \hat { y } _ { i } = \mathbb { I } ( p _ { i } \ge \tau ) ,\tag{3}
$$

and to accurately predict the early-warning labels of the later windows in $\mathcal { D } _ { B } ^ { C }$ , beyond the early normal period.

The proposed framework proceeds in three phases. In Phase 1, unsupervised label identification based on a control chart determines the labels of the source domain data and yields the labeled source set $\mathcal { D } _ { A }$ (Sect. 3.3). In Phase 2, the Adapted Domain-Adversarial Neural Network (ADANN) transfers knowledge from $\mathcal { D } _ { A }$ to Bearing B by incorporating $\mathcal { D } _ { B } ^ { T }$ in the training procedure (Sect. 3.4). In Phase 3, the trained model performs online early fault detection on the target (Sect. 3.5).

## 3.3 Unsupervised Label Identification on the Source Asset

Historical data from Bearing A can be abundant, raw, and devoid of annotations. Manually labeling extensive time series spanning months or years is labor-intensive, subjective, and impractical for real-world deployment. Furthermore, manual labeling often fails to capture the precise point of fault initiation, labeling broad segments as faulty and potentially obscuring the subtle, incipient patterns that are critical for early detection. To reduce this burden and construct a consistent source dataset, an SPC method, specifically the exponentially weighted moving average (EWMA) control chart, is employed to identify the states of Bearing A. In this study, the root mean square (RMS) of the vibration signal is monitored. For the sequence of windowed segments from Bearing A, the RMS values are computed channel by channel, yielding a time series $r _ { t } , t = 1 , 2 , \ldots , T$ , where $r _ { t }$ is the RMS value at time interval �. The EWMA statistic $z _ { t }$ is defined recursively as

$$
z _ { t } = \lambda r _ { t } + ( 1 - \lambda ) z _ { t - 1 } , \qquad t = 1 , 2 , \ldots , T , \qquad z _ { 0 } = \mu _ { 0 } ,\tag{4}
$$

where $\lambda \in ( 0 , 1 ]$ is a smoothing constant that determines the memory of the chart.

The upper control limit is constructed from the mean $\mu _ { 0 }$ and standard deviation $\sigma _ { 0 }$ of the RMS series over an initial normal operating period,

$$
\mathrm { U C L } _ { t } = \mu _ { 0 } + L \sigma _ { 0 } \sqrt { \frac { \lambda } { 2 - \lambda } [ 1 - ( 1 - \lambda ) ^ { 2 t } ] } ,\tag{5}
$$

where � is the control limit width. The labeling rule is as follows. As long as $z _ { t }$ remains below the upper control limit, the bearing at time � is considered normal $( s _ { t } ^ { A } = 0 )$ . The first time $z _ { t }$ exceeds $\mathrm { U C L } _ { t }$ , the process is deemed out of control, indicating the initiation of a fault, and all subsequent time points are labeled abnormal $( s _ { t } ^ { A } = 1 )$ . The window labels of the source then follow from Eq. (2) with $\zeta = 1$ ; because Eq. (2) looks � sampling points ahead, the source labels are shifted forward by the prediction horizon, so that each source window is labeled with the state reached � sampling points later. This procedure yields the labeled source dataset $\mathcal { D } _ { A } =$ $\{ \left( X _ { i } ^ { A } , y _ { i } ^ { A } \right) \} _ { i = 1 } ^ { N _ { A } }$

The control chart may not detect the fault onset perfectly and may raise false alarms, which could mislead the subsequent transfer step. Therefore, the segments flagged as anomalous are manually verified in this study. Even so, verifying the flagged segments requires substantially less effort than labeling the entire record manually.

The verified source dataset $\mathcal { D } _ { A }$ serves as the input for the next stage of the framework. The following subsection details how the framework leverages $\mathcal { D } _ { A }$ , together with the normal target data $\mathcal { D } _ { B } ^ { T }$ , to learn a feature representation that is discriminative for fault detection and transferable across bearings.

## 3.4 Adapted Domain-Adversarial Neural Network (ADANN)

Following the acquisition of a labeled source dataset, the next step is to transfer the diagnostic knowledge embedded in these data to the target bearing. Knowledge transfer is required because the distributions of vibration data collected from different bearings, even of the same type, often exhibit a domain shift due to variations in manufacturing tolerances, installation conditions, and operating environments. Domain-Adversarial Neural Networks (DANN) (Ganin et al., 2016) mitigate this shift by learning feature representations that are discriminative on the source domain and invariant to the differences between the source and target domains. The invariance is achieved through adversarial training, which encourages the feature extractor to confuse a domain classifier and thereby aligns the feature distributions of the two domains.

As shown in Fig. 3, the DANN architecture comprises three trainable components: a feature extractor $G _ { f }$ , a label predictor $G _ { y }$ , and a domain classifier $G _ { d }$ . The feature extractor, parameterized by $\theta _ { f }$ , maps input windows from both domains into a shared feature space. The label predictor, parameterized by $\theta _ { y }$ , takes the features of source windows and predicts their labels by minimizing the binary cross-entropy

$$
\mathcal { L } _ { y } \big ( \mathcal { D } _ { A } ; \theta _ { f } , \theta _ { y } \big ) = - \frac { 1 } { N _ { A } } \sum _ { i = 1 } ^ { N _ { A } } \left[ y _ { i } ^ { A } \mathrm { l n } G _ { y } \left( G _ { f } \big ( X _ { i } ^ { A } \big ) \right) + \big ( 1 - y _ { i } ^ { A } \big ) \mathrm { l n } \left( 1 - G _ { y } \left( G _ { f } \big ( X _ { i } ^ { A } \big ) \right) \right) \right] .\tag{6}
$$

The domain classifier, parameterized by $\theta _ { d }$ , outputs the probability that a feature originates from the source domain and is trained by minimizing the domain classification loss

$$
\mathcal { L } _ { d } \big ( \mathcal { D } _ { A } , \mathcal { D } _ { B } ^ { T } ; \theta _ { f } , \theta _ { d } \big ) = - \frac { 1 } { N _ { A } } \sum _ { i = 1 } ^ { N _ { A } } \ln G _ { d } \left( G _ { f } \big ( X _ { i } ^ { A } \big ) \right) - \frac { 1 } { N _ { T } } \sum _ { X \in \mathcal { D } _ { B } ^ { T } } \ln \left( 1 - G _ { d } \left( G _ { f } ( X ) \right) \right) .\tag{7}
$$

A gradient reversal layer (GRL) is positioned between the feature extractor and the domain classifier. In the forward pass the GRL acts as the identity; in backpropagation it reverses the sign of the gradients flowing from the domain classifier to the feature extractor. As a result, the domain classifier is trained to minimize $\mathcal { L } _ { d }$ while the feature extractor is trained to maximize it, which yields features that the domain classifier cannot distinguish. The standard DANN objective is

$$
\operatorname* { m i n } _ { \theta _ { f } , \theta _ { y } } \operatorname* { m a x } _ { \theta _ { d } } \mathcal { E } \big ( \theta _ { f } , \theta _ { y } , \theta _ { d } \big ) = \mathcal { L } _ { y } \big ( \mathcal { D } _ { A } ; \theta _ { f } , \theta _ { y } \big ) - \alpha \mathcal { L } _ { d } \big ( \mathcal { D } _ { A } , \mathcal { D } _ { B } ^ { T } ; \theta _ { f } , \theta _ { d } \big ) ,\tag{8}
$$

where � weights the domain classification loss. At convergence, the label information of Bearing A is encoded in $G _ { y }$ , the domain-invariant representation is encoded in $G _ { f }$ through the adversarial signal of $G _ { d }$ , and the composition $G _ { y } \circ G _ { f }$ is applied to predict the labels of $\mathcal { D } _ { B } ^ { C }$

Fig. 3 Architecture of the domain-adversarial neural network. Reproduced from Ganin et al. (2016)

The standard DANN objective only makes the domains indistinguishable. It does not use the limited but valuable information available in the target’s normal data, and it does not enforce a finer alignment of the distributions. To improve transfer for early fault detection, ADANN incorporates two modifications. First, a Maximum Mean Discrepancy (MMD) loss (Gretton et al., 2006) explicitly minimizes the distributional distance between the source and target features. Second, an Adapted Classification-Domain (ACD) loss involves the target’s normal data in the classification objective and thereby anchors the decision boundary on the target bearing.

## 3.4.1 Maximum Mean Discrepancy Loss

Adversarial training is effective at achieving global domain alignment, but it may not capture finer, higher-order statistical differences between the two domains. To complement the adversarial mechanism with a direct and stable measure of distributional similarity, an MMD loss is incorporated into ADANN. MMD is a nonparametric distance between two probability distributions based on their embeddings in a reproducing kernel Hilbert space (RKHS) ℋ; with a characteristic kernel, the mean embeddings of two distributions coincide if and only if the distributions are identical.

Let $p _ { s }$ and $p _ { t }$ denote the distributions of the source and target features produced by $G _ { f }$ . Given source features $F _ { A } = \{ f _ { i } ^ { A } \} _ { i = 1 } ^ { N _ { A } }$ drawn from $p _ { s }$ and target features $F _ { B } = \{ f _ { j } ^ { B } \} _ { j = 1 } ^ { N _ { T } }$ drawn from $p _ { t }$ , the squared MMD between the two empirical distributions is

$$
\mathsf { M M D } ^ { 2 } ( F _ { A } , F _ { B } ) = \left. \frac { 1 } { N _ { A } } \sum _ { i = 1 } ^ { N _ { A } } \phi \left( f _ { i } ^ { A } \right) - \frac { 1 } { N _ { T } } \sum _ { j = 1 } ^ { N _ { T } } \phi \left( f _ { j } ^ { B } \right) \right. _ { \mathcal { H } } ^ { 2 } ,\tag{9}
$$

where $\phi ( \cdot )$ is the feature map associated with a kernel $k ( \cdot , \cdot )$ , such that $k ( f , f ^ { \prime } ) =$ $\langle \phi ( f ) , \phi ( f ^ { \prime } ) \rangle _ { \mathcal { H } }$ . We employ the radial basis function (RBF) kernel $k ( f , f ^ { \prime } ) = \exp \bigl ( - \ l \ f -$ $f ^ { \prime } \parallel ^ { 2 } / ( 2 \sigma ^ { 2 } ) )$ with bandwidth $\sigma .$ . The MMD loss $\mathcal { L } _ { m m d }$ is computed as this squared distance on each mini-batch of source and target features. Adding it to the training objective penalizes the feature extractor whenever the source and target feature distributions diverge, which provides a complementary supervisory signal to the adversarial process. The objective becomes

$$
\operatorname* { m i n } _ { \theta _ { f } , \theta _ { y } } \operatorname* { m a x } _ { \theta _ { d } } \mathcal { E } = \mathcal { L } _ { y } \big ( \mathcal { D } _ { A } ; \theta _ { f } , \theta _ { y } \big ) - \alpha \mathcal { L } _ { d } \big ( \mathcal { D } _ { A } , \mathcal { D } _ { B } ^ { T } ; \theta _ { f } , \theta _ { d } \big ) + \mathcal { L } _ { m m d } \big ( \mathcal { D } _ { A } , \mathcal { D } _ { B } ^ { T } ; \theta _ { f } \big ) .\tag{10}
$$

The adversarial term and the MMD term both act on the marginal feature distributions. Neither of them evaluates the label predictor $G _ { y }$ on target data: $G _ { y }$ remains trained exclusively on sourcelabeled windows, which leaves the decision boundary uncalibrated for the target’s normal condition. The ACD loss addresses this limitation.

## 3.4.2 Adapted Classification-Domain Loss

Trained on labeled source data only, the label predictor learns a decision boundary that separates normal and faulty conditions within the source feature distribution. After alignment, the target features lie close to the source distribution, but the part of the feature space occupied by the target’s normal windows need not coincide with the region the source-trained boundary labels as normal. Target normal windows that fall on the wrong side of this boundary produce false alarms. To address this, the ACD loss extends the classification objective across the two domains.

The classification objective of ADANN consists of two parts. The first is the source classification loss $\mathcal { L } _ { y }$ of Eq. (6), which preserves the ability of $G _ { y }$ to discriminate normal from faulty states on the labeled source data. The second is a target normality term, which feeds the known normal target windows $\mathcal { D } _ { B } ^ { T }$ through $G _ { f }$ and $G _ { y }$ and penalizes any probability of the abnormal class:

$$
\mathcal { L } _ { A C D } \big ( \mathcal { D } _ { B } ^ { T } ; \theta _ { f } , \theta _ { y } \big ) = - \frac { 1 } { N _ { T } } \sum _ { X \in \mathcal { D } _ { B } ^ { T } } \ln \Big ( 1 - G _ { y } \left( G _ { f } ( X ) \right) \Big ) .\tag{11}
$$

We refer to Eq. (11) as the ACD loss, since it adapts the classification objective to the target domain. The complete ADANN objective is

$$
\begin{array} { r l } & { \underset { \theta _ { f } , \theta _ { y } } { \operatorname* { m i n } } \underset { \theta _ { d } } { \operatorname* { m a x } } \ : \mathcal { E } = \mathcal { L } _ { y } \big ( \mathcal { D } _ { A } ; \theta _ { f } , \theta _ { y } \big ) + \mathcal { L } _ { A C D } \big ( \mathcal { D } _ { B } ^ { T } ; \theta _ { f } , \theta _ { y } \big ) } \\ & { } \\ & { - \alpha \mathcal { L } _ { d } \big ( \mathcal { D } _ { A } , \mathcal { D } _ { B } ^ { T } ; \theta _ { f } , \theta _ { d } \big ) + \mathcal { L } _ { m m d } \big ( \mathcal { D } _ { A } , \mathcal { D } _ { B } ^ { T } ; \theta _ { f } \big ) . \qquad ( 1 2 ) } \end{array}
$$

With the ACD loss, the label predictor is no longer a passive recipient of aligned features. It is guided to place the target’s normal windows inside its normal region, so that the definition of normal is anchored to Bearing $\mathbf { B } ^ { \ast } \mathbf { s }$ own normal operating condition. Whether incipient fault signals of Bearing B enter the source-trained fault region is evaluated empirically, because the ACD term contains no target fault examples. The ACD loss is the only term in Eq. (12) that evaluates the label predictor on target data; Sect. 3.4.3 makes its role precise. The training procedure of ADANN is summarized in Algorithm 1.

Algorithm 1 Training Procedure for the Adapted Domain-Adversarial Neural Network (ADANN)   
Input: labeled source dataset $\mathcal { D } _ { A }$ (Bearing A); normal-only target dataset $\overline { { \mathcal { D } _ { B } ^ { T } } }$ (Bearing B); number of epochs �;   
batch size �; domain weight �; kernel bandwidth �   
Initialize parameters $\theta _ { f }$ of $G _ { f } , \theta _ { y }$ of $G _ { y }$ , and $\theta _ { d }$ of $G _ { d }$   
For epoch � from 1 to �:   
For each iteration of the epoch:   
Step 1: Sample a source mini-batch $( { \cal X } _ { s } , { \cal Y } _ { s } )$ of size � from $\mathcal { D } _ { A }$   
Step 2: Sample a target mini-batch $X _ { t }$ of size � from $\mathcal { D } _ { B } ^ { T }$ (all labels normal)   
Step 3: Compute features $F _ { s } = G _ { f } ( X _ { s } )$ and $F _ { t } = G _ { f } ( X _ { t } )$   
Step 4: Compute $\mathcal { L } _ { y }$ on $\left( G _ { y } ( F _ { s } ) , Y _ { s } \right)$ by Eq. (6), $\mathcal { L } _ { A C D }$ on $G _ { y } ( F _ { t } )$ by Eq. (11), $\mathcal { L } _ { d }$ on $\left( G _ { d } ( F _ { s } ) , G _ { d } ( F _ { t } ) \right)$ by Eq. (7),   
and $\mathcal { L } _ { m m d } = \mathop { \bf M } \mathop { \bf M } \mathop { \bf D } ^ { 2 } ( F _ { s } , F _ { t } )$ by Eq. (9)   
Step 5: Compute ℰ by Eq. (12)   
Step 6: Update $\theta _ { f } , \theta _ { y }$ by descending $\mathcal { E }$ and $\theta _ { d }$ by descending $\mathcal { L } _ { d }$ (implemented jointly through the GRL) using   
the Adam optimizer   
Output: Trained feature extractor $G _ { f } ,$ label predictor $G _ { y } ,$ and domain classifier $G _ { d }$

## 3.4.3 Equilibrium of ADANN under a Normal-Only Target

By the EWMA-based labeling of Sect. 3.3, the source domain $\mathcal { D } _ { A }$ contains both normal and abnormal windows, whereas the target asset contributes only normal windows $\mathcal { D } _ { B } ^ { T }$ to training. The source feature density induced by $G _ { f }$ is therefore a class mixture,

$$
p _ { s } ( f ) = \pi _ { N } p _ { s } ^ { N } ( f ) + \pi _ { A } p _ { s } ^ { A } ( f ) , \qquad \pi _ { N } + \pi _ { A } = 1 ,\tag{13}
$$

where $p _ { s } ^ { N }$ and $p _ { s } ^ { A }$ are the class-conditional feature densities of the source normal and abnormal windows, $\pi _ { A }$ is the abnormal proportion fixed by the control-chart labels, and the target density $p _ { t }$ is purely normal. Both alignment mechanisms operate on these marginals: the domain classifier separates source features drawn from the mixture $p _ { s }$ from target features drawn from $p _ { t }$ , and $\mathcal { L } _ { m m d }$ is computed between the same two marginals. The following theorem characterizes the domain-adversarial component of Eq. (12) under the classification correctness that ADANN must preserve at deployment.

Theorem 1. Assume that $( i ) G _ { f } , G _ { y }$ , and $G _ { d }$ have sufficient capacity; (ii) at each training step, $G _ { d }$ attains the minimum of $\mathcal { L } _ { d } f o r$ the current $G _ { f } ,$ and (iii) the label predictor $G _ { y }$ classifies the source windows and the target normal windows correctly, the property that $\mathcal { L } _ { y }$ and $\mathcal { L } _ { A C D }$ drive toward at convergence. Then the domain-adversarial component of Eq. (12) at the optimal domain classifier $D ^ { * }$ satisfies

$$
- \alpha \mathcal { L } _ { d } ( D ^ { * } ) \geq - \alpha \mathrm { l n } 4 + 2 \alpha \delta ( \pi _ { A } ) ,\tag{14}
$$

where

$$
\delta ( { \pi } _ { A } ) = \ln 2 - \frac { 2 - \pi _ { A } } { 2 } H \left( \frac { 1 - \pi _ { A } } { 2 - \pi _ { A } } \right) = H \left( \frac { \pi _ { A } } { 2 } \right) - \frac { 1 } { 2 } H ( \pi _ { A } ) ,\tag{15}
$$

with �(⋅) the binary entropy in nats. Equality in Eq. (14) holds if and only if the source normal features and the target normal features coincide, $p _ { s } ^ { N } ( f ) = p _ { t } ( f )$ for almost every �, while the source abnormal features remain separated from them. At this aligned state, the optimal domain classifier outputs 1 on the abnormal features and $\pi _ { N } / ( 1 + \pi _ { N } )$ on the normal ones, and the MMD term equals its residual value $\mathcal { L } _ { m m d } ^ { \dagger } = \pi _ { A } ^ { 2 } \parallel \mu _ { s } ^ { A } - \mu _ { s } ^ { N } \parallel _ { \mathcal { H } } ^ { 2 }$ , where $\mu _ { s } ^ { N }$ and $\mu _ { s } ^ { A }$ are the classconditional kernel mean embeddings. Moreover, � is continuous and strictly increasing on [0,1] with $\delta ( 0 ) = 0$ and $\delta ( 1 ) = \ln 2$ ; hence, as $\pi _ { A }  0$ , Eq. (14) reduces to the constant −�ln4 attained at $p _ { s } = p _ { t }$ , the classical equilibrium of domain-adversarial training.

The proof is given in Appendix A. Theorem 1 shows that, under a normal-only target, the domain-adversarial component cannot reach the classical value −�ln4: the abnormal proportion $\pi _ { A }$ of the source has no counterpart in $\mathcal { D } _ { B } ^ { T }$ , and the classification losses keep it on the abnormal side of the boundary, so the gap 2� $\delta ( \pi _ { A } )$ quantifies the irreducible mismatch of adapting to a normal-only target. The bound is attained exactly at the state expected of successful adaptation, in which the normal clusters of the two bearings merge $( p _ { s } ^ { N } = p _ { t } )$ and the abnormal cluster remains separated. Without the classification terms, the adversarial and MMD terms would instead be minimized at $p _ { s } = p _ { t }$ , which drags source abnormal features onto the target normal data. The classification terms in Eq. (12) penalize this failure mode of marginal alignment under label shift, and Theorem 1 characterizes the outcome only when its exact-classification assumption holds.

The theorem also yields a diagnostic for training. At the aligned state, the domain classifier outputs approach $\pi _ { N } / ( 1 + \pi _ { N } )$ on target windows and 1 on source abnormal windows, the domain classification loss approaches ln4 $- 2 \delta ( \pi _ { A } )$ , and $\mathcal { L } _ { m m d }$ plateaus at $\mathcal { L } _ { m m d } ^ { \dagger }$ rather than at zero. Under the theorem’s exact-classification and aligned-state assumptions, a nonzero residual is therefore compatible with the normal-only target setting. Theorem 1 is a population-level statement under assumptions (i) to (iii); with finite mini-batches and an adversary that is only approximately optimal, these values are approached rather than attained.

Remark 1 (role of the ACD loss). Assumption (iii) contains three conditions (Eq. (A.2) in Appendix $\mathbf { A } )$ . Two of them, $p _ { s } ^ { A } ( R _ { A } ) = 1$ and $p _ { s } ^ { N } ( R _ { A } ) = 0$ for the abnormal region $R _ { A }$ of the classifier, concern the source domain and are what $\mathcal { L } _ { y }$ drives toward. The third, $p _ { t } ( R _ { A } ) = 0$ requires that the target’s normal features lie outside the abnormal region; it concerns only the target, and among the terms of Eq. (12) only $\mathcal { L } _ { A C D }$ evaluates $G _ { y }$ on target data. The ACD loss controls this condition directly. Since $\begin{array} { r } { \mathbb { I } ( p \geq \tau ) \leq \ln \frac { 1 } { 1 - p } / \ln \frac { 1 } { 1 - \tau } } \end{array}$ for every $p \in [ 0 , 1 )$ , the empirical false-alarm rate on the adaptation windows satisfies

$$
\widehat { \mathrm { F A R } } _ { T } : = \frac { 1 } { N _ { T } } \sum _ { X \in \mathcal { D } _ { B } ^ { T } } \mathbb { I } \left( G _ { y } \left( G _ { f } ( X ) \right) \geq \tau \right) \leq \frac { \mathcal { L } _ { A C D } } { \ln \left( 1 / ( 1 - \tau ) \right) } ,\tag{16}
$$

so that, for $\tau = 0 . 5$ , $\widehat { \mathrm { F A R } } _ { T } \leq { \mathcal { L } } _ { A C D } / \ln 2$ . Equation (16) is an in-sample statement about the adaptation windows. It is not a generalization bound: extending it to future normal windows of the target would require a held-out normal interval or additional stationarity, capacity, and concentration assumptions, none of which is claimed here. No analogous target-side control exists for missed detections. With zero target fault samples, detection of target faults relies on the source abnormal region $R _ { A }$ and on the alignment $p _ { s } ^ { N } \approx p _ { t }$ , and the distribution of target fault features cannot be estimated from $\mathcal { D } _ { B } ^ { T }$ . ADANN is therefore expected to reduce false alarms relative to source-only DANN and MMD alignment, without a comparable guarantee on recall; Sect. 4.2 examines this prediction empirically.

Remark 2 (relation to existing results). Lemmas 1 and 2 in Appendix A follow the classical analysis of adversarial training (Goodfellow et al., 2014). The quantity $\delta ( \pi _ { A } )$ equals the Jensen-Shannon divergence between the label distributions of the two domains, $\mathsf { B e r n o u l l i } ( \pi _ { A } )$ and Bernoulli(0), so the lower bound in Eq. (14) is an instance of the label-shift argument of H. Zhao et al. (2019) specialized to a normal-only target. What Theorem 1 adds for the present setting is the exact attainment condition $p _ { s } ^ { N } = p _ { t }$ , the equilibrium outputs of the domain classifier, and the MMD residual, which together identify the aligned state that ADANN is designed to reach.

## 3.5 Overall Framework for Zero-Shot Cross-Asset Fault Detection

Building on the components of the preceding subsections, Fig. 4 illustrates the architecture and workflow of the proposed zero-shot cross-asset fault detection framework. The framework operates in three phases: offline source labeling, offline ADANN training, and online early fault detection on the target.

Fig. 4 Overview of the proposed framework. Unlabeled vibration data from the source bearing (Bearing A) are labeled by an EWMA control chart; the labeled source data and the early-stage normal data of the target bearing (Bearing B) are used to train ADANN; the trained model performs online early fault detection on Bearing B

The first phase, unsupervised label identification, processes the raw vibration signals from the source bearing. Condition indicators such as the RMS are extracted and monitored with an EWMA control chart whose limits are derived from an initial normal operating period, and each segment is labeled as normal or faulty. After verification of the flagged segments, this phase yields the labeled source dataset $\mathcal { D } _ { A }$

The second phase, ADANN training, uses $\mathcal { D } _ { A }$ together with the normal target dataset $\mathcal { D } _ { B } ^ { T }$ . Source and target features are passed to the label predictor, which is trained with $\mathcal { L } _ { y }$ on source windows and $\mathcal { L } _ { A C D }$ on target windows, and, through the gradient reversal layer, to the domain classifier. The MMD loss is computed directly between the source and target features. The four loss components are combined into the objective of $\operatorname { E q } .$ . (12), which updates the parameters of all three networks by backpropagation.

The third phase deploys the trained model for online fault detection on Bearing B. Streaming vibration windows are passed through the fixed feature extractor $G _ { f }$ and label predictor $G _ { y }$ which outputs the probability of an upcoming abnormal state. Because the ACD loss anchors the target’s normal condition in the normal region, a low predicted probability indicates normal operation, and a rise above the threshold � issues an early warning. Since the labels are defined � sampling points ahead of the window start, a warning can precede the labeled abnormal state. The realized maintenance lead time depends on when alarms occur relative to that state. The complete procedure is summarized in Algorithm 2 and validated in Sect. 4.

Algorithm 2 Zero-shot cross-asset early fault detection framework   
Input: raw source data $\{ x _ { t } ^ { A } \} _ { t = 1 } ^ { T _ { A } }$ (Bearing $\mathbf { A } ) ;$ raw early-stage target data $\{ x _ { t } ^ { B } \} _ { t = \cdot } ^ { M }$ (Bearing B); streaming target data   
$\{ x _ { t } ^ { B } \} _ { t > M } ; \mathrm { A D A N N }$ hyperparameters �, �, �, �; decision threshold �   
Step 1: Window the early-stage target data by Eq. (1) and label all windows as normal to form $\mathcal { D } _ { B } ^ { T }$   
Step 2: Label the source data with the EWMA chart of Eqs. (4) and (5), verify the flagged segments, and form $\mathcal { D } _ { A }$   
by Eqs. (1) and (2)   
Step 3: Train ADANN on $\mathcal { D } _ { A }$ and $\mathcal { D } _ { B } ^ { T }$ by Algorithm 1   
Step 4: For each incoming target window $X _ { i } ^ { B }$ , compute $p _ { i } = G _ { y } \left( G _ { f } ( X _ { i } ^ { B } ) \right)$ and issue an alarm ${ \mathrm { i f } } p _ { i } \geq \tau \left( \mathrm { E q . } \left( 3 \right) \right)$   
Output: early-warning predictions $\hat { y } _ { i }$ for the target windows in $\mathcal { D } _ { B } ^ { C }$

## 4 Case Study

This section evaluates the proposed methodology on the bearing data of Sect. 3.1. The setup is described in Sect. 4.1, the main results in Sect. 4.2, the implications for maintenance decisions in Sect. 4.3, and the robustness experiments in Sect. 4.4.

## 4.1 Experimental Setup

For the source domain (Bearing A), the last 40 s of the run-to-failure record were used, and the anomalous period begins at 32 s of this segment. At a sampling frequency of 20 kHz, the segment contains 800,000 time points on each of the three acceleration channels. The vibration signals of the analyzed segments of Bearing A and Bearing B are shown in Fig. 5.

![](images/033305f2efe93d8a302a73decdf841d0144543ebc8efbdeafb196825338b5707.jpg)

![](images/aa726687bd55ac50be690d6cc09e601d5a1a96c654476b67ee0e1a801d4f4449.jpg)  
Fig. 5 Vibration signal (Y direction) of the 40 s segments analyzed for (a) Bearing A (6210 experiment) and (b) Bearing B (6011 experiment). The dashed line in (b) separates the adaptation set $\mathcal { D } _ { B } ^ { T }$ from the test set $\mathcal { D } _ { B } ^ { C }$

For the target domain (Bearing B), the last 40 s of the record were also used, with the evaluation fault boundary set at 20 s of this segment; the evaluation stage indicator $s _ { t } ^ { B }$ is set to 1 from this boundary onward. The target data are divided into two subsets: the adaptation set $\mathcal { D } _ { B } ^ { T }$ , consisting of the first 8 s of the segment in Fig. 5(b), and the test set $\mathcal { D } _ { B } ^ { C }$ , consisting of the remaining 32 s. Each window contains $W = 1 0 0 0$ consecutive sampling points (0.05 s), the stride is $S = 2 0 0$ ， the prediction horizon is 2 s $( H = 4 0 , 0 0 0$ sampling points), and $\eta = 2 0 0$ , so that a window is labeled abnormal if more than 20% of the corresponding future interval is abnormal. For the binary EWMA-derived indicator, $\zeta = 1$ . For the unsupervised label identification of Bearing A, the EWMA chart was applied to the RMS series of the raw vibration signal with smoothing constant $\lambda = 0 . 3$ and control limit width $L = 3 . 6$

ADANN comprises three networks: a feature extractor, a label predictor, and a domain classifier, whose architectures are shown in Fig. 6. The model was trained with the Adam optimizer at an initial learning rate of 0.001, a batch size of 128, and 100 epochs. The domain weight was set empirically to $\alpha = 1 . 0$ , the ACD and MMD terms enter Eq. (12) with unit weight, and the RBF kernel bandwidth was $\sigma = 1 . 0$ . The decision threshold is $\tau = 0 . 5$

## Fig. 6 Network structure of ADANN

The proposed ADANN was compared with four benchmark approaches. Because each benchmark removes one or more components of Eq. (12), the comparison also serves as an ablation study.

Source-Only. The feature extractor and label predictor are trained exclusively on the labeled source data (Bearing A) and applied directly to the target test set. This benchmark measures the degradation caused by the domain shift between bearings.

• DANN. The standard domain-adversarial network of Eq. (8), trained with $\mathcal { L } _ { y }$ and $\mathcal { L } _ { d }$ only. It measures the contribution of adversarial alignment alone.

MMD-Only. Adversarial alignment is replaced by explicit distribution matching: the model is trained with $\mathcal { L } _ { y }$ and $\mathcal { L } _ { m m d }$ , without the domain classifier.

DANN+MMD. The objective of Eq. (10), trained with $\mathcal { L } _ { y } , \mathcal { L } _ { d }$ , and $\mathcal { L } _ { m m d }$ . It differs from ADANN only by the absence of $\mathcal { L } _ { A C D }$ and therefore isolates the contribution of the ACD loss.

All benchmarks use the same feature extractor and label predictor architectures and the same training hyperparameters. Each method was trained and evaluated in five independent runs, and we report the mean and standard deviation of precision, recall, and F1-score over the runs on the target test set.

## 4.2 Results

## 4.2.1 Source Labeling

The results of the unsupervised label identification on Bearing A are shown in Fig. 7. The EWMA chart signals an out-of-control state on all channels, with differences in detection latency and specificity among the channels. The detection delay is defined as the time elapsed between fault occurrence and the first control limit violation. Channel 3 was the most sensitive, triggering an alarm at 32.75 s, a delay of 0.75 s. Channel 2 also detected the fault, with a delay of 3.25 s, so that the average delay over the valid detections was 2 s. Specificity was evaluated through false alarms during the healthy period: Channels 2 and 3 raised no false alarms, whereas Channel 1 triggered three false alarms during the baseline period, the first at 1.25 s, and its alarm was therefore classified as a false positive. These results show that Channel 3 captures the early propagation of this fault type best, and that the flagged segments still require verification. Even so, verifying the flagged segments requires far less effort than labeling the full record manually.

![](images/f8b41d0175faf33edea678fcb5d4254226934f84ecf392d4ee4a139aabdc76a6.jpg)  
Fig. 7 EWMA control charts of the RMS series of the three channels of Bearing A (6210 experiment)

## 4.2.2 Early Fault Detection on the Target

Table 2 reports the precision, recall, and F1-score of all methods on the target test set. All adaptation methods that include the adversarial term improve substantially on the Source-Only baseline, which is highly conservative on Bearing B: it raises almost no false alarms but misses about half of the faulty windows. ADANN’s recall and F1-score exceed those of Source-Only and MMD-Only, whereas MMD-Only performs similarly to Source-Only, which indicates that discrepancy minimization alone does not transfer the fault boundary in this case. Under the window labeling of Eq. (2) with the settings of Sect. 4.1, about two thirds of the test windows are faulty (all windows starting from about 18 s onward), so a detector that raises an alarm on every window would attain a precision of 0.67, a recall of 1, and an F1-score of about 0.80. Source-Only and MMD-Only fall below this reference, whereas DANN, DANN+MMD, and ADANN exceed it. Precision separates the methods more clearly: it is 0.67 for the all-alarm detector and 0.9997 for ADANN, which shows that ADANN does separate normal from faulty target windows rather than alarming indiscriminately.

Table 2 Precision, recall, and F1-score on the target test set (mean ± standard deviation over five runs; best mean in bold)
<table><tr><td rowspan="2">Method</td><td colspan="3">Classification Metrics</td></tr><tr><td>Recall</td><td>Precision</td><td>F1-score</td></tr><tr><td>Source-Only</td><td> $0 . 4 9 6 4 \pm 0 . 0 4 4 8$ </td><td> $\mathbf { 1 . 0 0 0 0 } \pm \mathbf { 0 . 0 0 0 0 }$ </td><td> $0 . 6 6 2 5 \pm 0 . 0 3 9 0$ </td></tr><tr><td>DANN</td><td> $0 . 7 3 0 1 \pm 0 . 0 4 7 6$ </td><td> $\overline { { 0 . 9 2 4 8 \pm 0 . 0 7 8 3 } }$ </td><td> $\overline { { 0 . 8 1 2 9 \pm 0 . 0 2 7 3 } }$ </td></tr><tr><td>MMD-Only</td><td> $0 . 5 1 9 3 \pm 0 . 0 4 3 4$ </td><td> $0 . 9 6 2 9 \pm 0 . 0 8 2 3$ </td><td> $0 . 6 7 1 7 \pm 0 . 0 2 7 6$ </td></tr><tr><td>DANN+MMD</td><td> $\mathbf { 0 . 7 4 4 2 \pm 0 . 0 0 4 8 }$ </td><td> $\overline { { 0 . 9 1 8 3 \pm 0 . 0 4 5 3 } }$ </td><td> $\overline { { 0 . 8 2 1 7 \pm 0 . 0 1 5 0 } }$ </td></tr><tr><td>ADANN (proposed)</td><td> $0 . 7 1 0 1 \pm 0 . 0 2 4 3$ </td><td> $0 . 9 9 9 7 \pm 0 . 0 0 0 6$ </td><td> $\mathbf { 0 . 8 3 0 2 \pm 0 . 0 1 6 6 }$ </td></tr></table>

Besides, ADANN attains the highest mean F1-score $( 0 . 8 3 0 2 \pm 0 . 0 1 6 6 )$ . The comparison with DANN+MMD, which differs from ADANN only by the ACD loss, isolates the effect of this term. Adding the ACD loss raises precision from 0.9183 to 0.9997 and lowers recall from 0.7442 to 0.7101. This pattern is the one anticipated by Remark 1: the ACD loss acts on the target’s normal data and hence on false alarms, while it provides no target-side information about faults. In short, the ACD loss shifts the operating point of the detector toward fewer false alarms at a modest cost in detection rate, and the F1-score, which weights the two error types equally, is nearly unchanged. Whether this shift is beneficial depends on the relative costs of the two errors, which Sect. 4.3 examines.

## 4.3 Implications for Maintenance Decisions

The detector’s output feeds a maintenance decision: each alarm triggers an inspection or intervention, and each missed faulty window is a lost opportunity to act before failure. To make the trade-off in Table 2 concrete, consider an illustrative window-level cost model in which a false alarm costs $c _ { F A }$ and a missed faulty window costs $c _ { M D } $ , with cost ratio $\kappa = c _ { F A } / c _ { M D }$ . For a test set with $N ^ { + }$ faulty windows, a detector with precision � and recall � produces $R N ^ { + }$ true alarms, $R N ^ { + } ( 1 - P ) / P$ false alarms, and $( 1 - R ) N ^ { + }$ missed windows. Its expected cost per faulty window, in units of $c _ { M D } ^ { \mathrm { ~ ~ } }$ , is therefore

$$
C ( \kappa ) = \kappa \frac { R \left( 1 - P \right) } { P } + ( 1 - R ) ,\tag{17}
$$

which depends only on quantities reported in Table 2. Using the mean values, ADANN has $C ( \kappa ) = 0 . 2 8 9 9 + 0 . 0 0 0 2 \kappa$ , compared with $0 . 2 5 5 8 + 0 . 0 6 6 2 \kappa$ for DANN+MMD, 0.2699 + 0.0594 � for DANN, $0 . 4 8 0 7 + 0 . 0 2 0 0 \kappa$ for MMD-Only, and 0.5036 for Source-Only. ADANN has the lower expected cost than DANN+MMD whenever $\kappa > 0 . 5 2$ and than DANN whenever $\kappa > 0 . 3 4$ , that is, whenever a false alarm costs more than about half, or one third, of a missed faulty window. It is cheaper than MMD-Only for every $\kappa \geq 0$ and cheaper than Source-Only unless false alarms are about a thousand times cheaper than missed windows.

This analysis is illustrative. It is defined at the window level, whereas maintenance costs in practice depend on the time of the first alarm, on whether repeated alarms during an ongoing fault add cost, and on asset-specific intervention and failure costs; it also uses mean values and does not propagate run-to-run variability. Its purpose is to show that the choice between ADANN and the strongest baselines is a cost-dependent decision rather than a matter of F1- score, and that ADANN is preferable in settings where unnecessary interventions on a healthy asset are costly, for example when an inspection requires a production stop.

## 4.4 Robustness under Synthetic Noise

To evaluate robustness under conditions that are not present in the laboratory data, two extended studies introduce controlled noise into the target data. The first adds Gaussian noise to the normal region and time-varying noise to the fault region (Sect. 4.4.1); the second adds two types of noise commonly encountered in rotating machinery, random shocks and harmonic interference (Sect. 4.4.2). Because the noise is synthetic, these experiments test sensitivity to specific corruptions rather than performance on field data.

## 4.4.1 Gaussian and Time-Varying Noise

In this study, the roles of the two bearings are reversed so that the uncorrupted Bearing B (6011) serves as the source domain and the noise-corrupted Bearing A (6210) serves as the target domain. The noise is constructed as follows.

Normal-region noise. Zero-mean white Gaussian noise is added to the raw vibration signal in the normal region. Its standard deviation takes four levels, $\sigma _ { n } \in \{ 2 . 0 , 1 . 0 , 0 . 5 , 0 . 1 \}$ , expressed as multiples of the standard deviation of the original signal in the normal region, which simulates varying degrees of sensor noise or environmental interference during data collection.

Fault-region noise. From 32 s onward, a time-varying noise $n ( t ) = \beta \left( t - t _ { 0 } \right) \varepsilon ( t )$ is added, where $t _ { 0 } = 3 2 \mathrm { s }$ is the start of the fault region, $\beta \in \{ 1 . 0 , 0 . 5 , 0 . 1 \}$ controls the growth rate, and �(�) is unit white noise. The fault signature thus becomes progressively more pronounced, mimicking degradation processes such as crack propagation or wear accumulation.

The preprocessing, windowing, and labeling procedures of Sect. 4.1 were applied to the corrupted target data, and all methods were retrained and evaluated five times under each condition. Table 3 reports the mean and standard deviation of the F1-scores.

Table 3 F1-scores under Gaussian and time-varying noise (mean ± standard deviation over five runs; best mean in bold)
<table><tr><td rowspan="2">White Noise Level  $( \pmb { \sigma _ { n } } )$ </td><td rowspan="2">Method</td><td colspan="3">Scaling Factor (β)</td></tr><tr><td>β = 1.0</td><td> $\beta = 0 . 5$ </td><td>β = 0.1</td></tr><tr><td rowspan="5"> $\sigma _ { n } = 2 . 0$ </td><td>Source-Only</td><td> $0 . 5 7 1 7 \pm 0 . 0 0 0 3$ </td><td> $0 . 6 3 5 2 \pm 0 . 1 5 2 0$ </td><td> $0 . 5 7 1 4 \pm 0 . 0 0 0 4$ </td></tr><tr><td>DANN</td><td> $0 . 5 7 1 2 \pm 0 . 0 0 0 9$ </td><td> $\mathbf { 0 . 7 8 5 3 \pm 0 . 0 2 3 5 }$ </td><td> $0 . 5 7 1 2 \pm 0 . 0 0 0 6$ </td></tr><tr><td>MMD-Only</td><td> $0 . 5 5 1 1 \pm 0 . 0 2 9 8$ </td><td> $0 . 6 1 5 0 \pm 0 . 1 9 0 4$ </td><td> $0 . 5 4 7 4 \pm 0 . 0 5 8 5$ </td></tr><tr><td>DANN+MMD</td><td> $0 . 5 7 1 3 \pm 0 . 0 0 0 9$ </td><td>0.7622 ± 0.0304</td><td>0.5714 ± 0.0003</td></tr><tr><td>ADANN (proposed)</td><td> $\mathbf { 0 . 5 7 1 8 \pm 0 . 0 1 7 5 }$ </td><td>0.7116 ± 0.0541</td><td>0.5886 ± 0.0547</td></tr><tr><td rowspan="5"> $\sigma _ { n } = 1 . 0$ </td><td>Source-Only</td><td> $0 . 5 7 1 1 \pm 0 . 0 0 0 6$ </td><td>0.8247 ± 0.0258</td><td>0.5387 ± 0.0634</td></tr><tr><td>DANN</td><td> $\overline { { 0 . 5 7 0 0 \pm 0 . 0 0 1 8 } }$ </td><td>0.8350 ± 0.0092</td><td>0.5709 ± 0.0016</td></tr><tr><td>MMD-Only</td><td> $0 . 5 0 1 3 \pm 0 . 0 4 5 2$ </td><td> $0 . 8 1 3 8 \pm 0 . 0 2 1 7$ </td><td> $0 . 4 5 5 1 \pm 0 . 2 0 2 6$ </td></tr><tr><td>DANN+MMD</td><td> $\overline { { 0 . 5 6 6 2 \pm 0 . 0 0 9 3 } }$ </td><td>0.8125 ± 0.0478</td><td> $0 . 5 7 1 2 \pm 0 . 0 0 0 8$ </td></tr><tr><td>ADANN (proposed)</td><td> $\mathbf { 0 . 5 7 2 6 \pm 0 . 0 0 4 1 }$ </td><td> $\mathbf { 0 . 8 3 7 1 \pm 0 . 0 2 5 4 }$ </td><td> $\mathbf { 0 . 6 0 2 2 \pm 0 . 0 9 7 6 }$ </td></tr><tr><td rowspan="3"> $\sigma _ { n } = 0 . 5$ </td><td>Source-Only</td><td> $\mathbf { 0 . 5 7 1 7 \pm 0 . 0 0 0 5 }$ </td><td> $0 . 8 5 3 0 \pm 0 . 0 3 8 6$ </td><td> $0 . 5 5 6 0 \pm 0 . 0 4 1 9$ </td></tr><tr><td>DANN</td><td> $\overline { { 0 . 5 7 0 8 \pm 0 . 0 0 1 4 } }$ </td><td> $\overline { { 0 . 8 4 2 7 \pm 0 . 0 1 1 6 } }$ </td><td> $\overline { { 0 . 5 7 1 0 \pm 0 . 0 0 1 3 } }$ </td></tr><tr><td>MMD-Only</td><td> $\overline { { 0 . 5 5 5 5 } } \pm 0 . 0 8 4 1$ </td><td> $0 . 7 6 0 4 \pm 0 . 1 1 5 8$ </td><td> $0 . 4 6 9 4 \pm 0 . 1 4 2 9$ </td></tr><tr><td rowspan="2"></td><td>DANN+MMD</td><td> $0 . 5 7 1 2 \pm 0 . 0 0 5 1$ </td><td> $0 . 8 3 7 4 \pm 0 . 0 1 1 2$ </td><td> $0 . 5 7 0 3 \pm 0 . 0 0 3 3$ </td></tr><tr><td>ADANN (proposed)</td><td> $0 . 5 6 8 4 \pm 0 . 0 6 2 3$ </td><td> $\mathbf { 0 . 8 6 1 8 \pm 0 . 0 3 6 3 }$ </td><td> $\mathbf { 0 . 6 3 2 9 \pm 0 . 0 8 4 5 }$ </td></tr><tr><td rowspan="5"> $\sigma _ { n } = 0 . 1$ </td><td>Source-Only</td><td> $0 . 5 4 6 1 \pm 0 . 0 5 0 6$ </td><td> $0 . 8 3 7 8 \pm 0 . 0 2 8 8$ </td><td> $0 . 5 4 9 4 \pm 0 . 0 6 0 1$ </td></tr><tr><td>DANN</td><td> $0 . 5 7 1 5 \pm 0 . 0 0 0 2$ </td><td> $0 . 8 3 1 9 \pm 0 . 0 1 3 9$ </td><td> $0 . 5 7 1 1 \pm 0 . 0 0 0 7$ </td></tr><tr><td>MMD-Only</td><td> $0 . 5 3 9 7 \pm 0 . 0 6 6 0$ </td><td> $0 . 7 3 1 0 \pm 0 . 0 7 4 9$ </td><td> $0 . 5 0 5 7 \pm 0 . 0 8 7 1$ </td></tr><tr><td>DANN+MMD</td><td> $0 . 5 6 1 3 \pm 0 . 0 4 7 8$ </td><td> $0 . 8 3 3 1 \pm 0 . 0 1 0 1$ </td><td> $0 . 5 7 0 4 \pm 0 . 0 0 2 1$ </td></tr><tr><td>ADANN (proposed)</td><td> $\mathbf { 0 . 6 0 4 0 \pm 0 . 0 7 6 6 }$ </td><td> $\mathbf { 0 . 8 5 1 1 \pm 0 . 0 3 0 1 }$ </td><td> $\mathbf { 0 . 6 3 9 8 \pm 0 . 1 2 5 7 }$ </td></tr></table>

ADANN has the highest mean F1-score in 10 of the 12 settings. For $\beta = 1 . 0$ and $\beta = 0 . 1$ , most methods obtain F1-scores close to 0.571 with very small standard deviations, which indicates that they converge to the same decision pattern in these settings. ADANN’s mean F1-score departs from this value most clearly at $\beta = 0 . 1 ~ ( 0 . 5 8 9 ~ \mathrm { ~ t o ~ } ~ 0 . 6 4 0 )$ , but with large standard deviations, which indicates that ADANN departs from this common pattern in some runs but not consistently.

For $\beta = 0 . 5$ , all adaptation methods separate the classes, and ADANN has the highest mean at $\sigma _ { n } \leq 1 . 0$ by small margins. At the highest noise level, $\sigma _ { n } = 2 . 0$ , ADANN falls below DANN $( 0 . 7 1 2 \mathrm { \ v s \ } 0 . 7 8 5 )$ . This result is consistent with Remark 1: the ACD loss anchors the normal region on the target’s adaptation data, and when these data are heavily corrupted, the anchor itself is displaced. The quality of the normal target data is therefore a practical precondition for the ACD loss.

## 4.4.2 Random Shocks and Harmonic Interference

Two further types of noise are added independently to the target data:

Random shocks. Short-duration, high-amplitude impulses that affect peak values and kurtosis of the vibration signal. The shock occurrence rate is fixed at 100 impacts per second, and the shock amplitude takes three levels, $A _ { s h o c k } \in \{ 3 . 0 , 3 . 5 , 4 . 0 \}$ .

Harmonic interference. Narrowband periodic components originating from rotating components such as shafts, gears, and motors, which can mask or mimic fault signatures. The fundamental frequencies are set in the mid range (150, 300, 450, and 600 Hz), and the harmonic amplitude takes three levels, $A _ { h a r m o n i c } \in \{ 0 . 5 , 0 . 8 , 1 . 0 \}$

The procedures of Sect. 4.1 were applied to the corrupted target data. Table 4 reports the mean and standard deviation of precision, recall, and F1-score over five runs for each setting.

Table 4 Precision, recall, and F1-score under random shocks and harmonic interference (mean ± standard deviation over five runs; best mean in bold, ties included)
<table><tr><td rowspan="2">Parameter  $( { A _ { s h o c k } } / { A _ { h a r m o n i c } } )$ </td><td rowspan="2">Method</td><td colspan="3">Metrics</td></tr><tr><td>Precision</td><td>Recall</td><td>F1-score</td></tr><tr><td rowspan="5"> $A _ { s h o c k } = 3 . 0$ </td><td>Source-Only</td><td> $\overline { { 0 . 2 9 0 6 \pm 0 . 0 9 7 4 } }$ </td><td> $\overline { { 0 . 5 3 8 5 \pm 0 . 3 3 0 8 } }$ </td><td> $\overline { { 0 . 3 0 9 1 \pm 0 . 1 6 0 2 } }$ </td></tr><tr><td>DANN</td><td> $\overline { { 0 . 1 8 6 2 \pm 0 . 1 2 2 2 } }$ </td><td> $\overline { { 0 . 5 7 7 0 \pm 0 . 4 0 8 3 } }$ </td><td> $\overline { { 0 . 2 7 8 3 \pm 0 . 1 8 8 3 } }$ </td></tr><tr><td>MMD-Only</td><td> $\overline { { 0 . 4 1 2 6 \pm 0 . 0 6 2 9 } }$ </td><td> $\overline { { 0 . 7 3 7 0 \pm 0 . 1 5 1 4 } }$ </td><td> $\overline { { 0 . 5 1 9 7 \pm 0 . 0 5 5 9 } }$ </td></tr><tr><td>DANN+MMD</td><td> $0 . 4 2 1 7 \pm 0 . 0 6 5 6$ </td><td> $\mathbf { \overline { { 0 . 8 5 9 3 \pm 0 . 0 8 8 8 } } }$ </td><td> $\overline { { 0 . 5 6 3 7 \pm 0 . 0 6 9 0 } }$ </td></tr><tr><td>ADANN (proposed)</td><td> $\mathbf { 0 . 4 5 6 2 \pm 0 . 1 1 8 5 }$ </td><td> $\overline { { 0 . 8 5 0 5 \pm 0 . 0 9 9 1 } }$ </td><td> $\mathbf { 0 . 5 7 5 8 \pm 0 . 0 5 7 3 }$ </td></tr><tr><td rowspan="5"> $A _ { s h o c k } = 3 . 5$ </td><td>Source-Only</td><td> $\overline { { 0 . 2 8 8 1 \pm 0 . 0 6 2 8 } }$ </td><td> $\overline { { 0 . 6 3 5 0 \pm 0 . 0 8 2 8 } }$ </td><td> $\overline { { 0 . 3 9 3 4 \pm 0 . 0 7 2 6 } }$ </td></tr><tr><td>DANN</td><td> $0 . 1 8 9 8 \pm 0 . 0 9 2 0$ </td><td> $0 . 6 4 8 5 \pm 0 . 3 8 0 6$ </td><td> $0 . 2 9 2 4 \pm 0 . 1 5 0 0$ </td></tr><tr><td>MMD-Only</td><td> $\overline { { 0 . 3 8 7 2 \pm 0 . 0 5 3 9 } }$ </td><td> $\overline { { 0 . 8 0 2 0 \pm 0 . 2 1 5 1 } }$ </td><td> $\overline { { 0 . 5 0 6 4 \pm 0 . 0 5 3 1 } }$ </td></tr><tr><td>DANN+MMD</td><td> $\overline { { 0 . 3 8 2 0 \pm 0 . 0 4 5 2 } }$ </td><td> $\overline { { 0 . 8 4 8 8 \pm 0 . 1 8 5 1 } }$ </td><td> $0 . 5 2 3 7 \pm 0 . 0 7 8 6$ </td></tr><tr><td>ADANN (proposed)</td><td> $\overline { { { \bf 0 . 4 0 8 6 \pm 0 . 0 5 7 8 } } }$ </td><td> $\mathbf { \overline { { 0 . 8 5 2 8 \pm 0 . 0 9 6 0 } } }$ </td><td> $\overline { { { \bf 0 . 5 4 8 0 \pm 0 . 0 5 6 3 } } }$ </td></tr><tr><td rowspan="5"> $A _ { s h o c k } = 4 . 0$ </td><td>Source-Only</td><td> $\mathbf { 0 . 4 2 4 2 \pm 0 . 1 9 1 1 }$ </td><td> $0 . 8 4 3 3 \pm 0 . 1 1 1 6$ </td><td> $0 . 5 2 5 4 \pm 0 . 1 3 3 3$ </td></tr><tr><td>DANN</td><td> $\overline { { 0 . 2 0 1 3 \pm 0 . 0 8 2 3 } }$ </td><td> $\overline { { 0 . 7 0 2 0 \pm 0 . 3 4 8 3 } }$ </td><td> $\overline { { 0 . 3 1 1 8 \pm 0 . 1 3 5 1 } }$ </td></tr><tr><td>MMD-Only</td><td> $\overline { { 0 . 3 7 6 2 \pm 0 . 0 2 8 0 } }$ </td><td> $\overline { { 0 . 9 3 2 5 \pm 0 . 0 7 1 2 } }$ </td><td> $\overline { { 0 . 5 3 4 0 \pm 0 . 0 1 9 8 } }$ </td></tr><tr><td>DANN+MMD</td><td> $0 . 3 5 0 5 \pm 0 . 0 5 9 5$ </td><td> $\mathbf { \overline { { 0 . 9 7 1 3 \pm 0 . 0 3 9 9 } } }$ </td><td> $0 . 5 1 1 0 \pm 0 . 0 6 1 6$ </td></tr><tr><td>ADANN (proposed)</td><td> $\overline { { 0 . 4 0 3 9 \pm 0 . 0 3 3 6 } }$ </td><td> $0 . 9 2 6 2 \pm 0 . 1 1 8 0$ </td><td> $\mathbf { 0 . 5 5 8 2 \pm 0 . 0 3 4 0 }$ </td></tr><tr><td rowspan="5"> $A _ { h a r m o n i c } = 0 . 5$ </td><td>Source-Only</td><td> $0 . 2 9 8 0 \pm 0 . 0 2 7 8$ </td><td> $\mathbf { \overline { { 1 . 0 0 0 0 } } \pm 0 . 0 0 0 0 }$ </td><td> $\overline { { 0 . 4 5 8 4 \pm 0 . 0 3 2 3 } }$ </td></tr><tr><td>DANN</td><td> $\overline { { 0 . 3 0 3 9 \pm 0 . 0 3 5 7 } }$ </td><td> $\overline { { 1 . 0 0 0 0 \pm 0 . 0 0 0 0 } }$ </td><td> $\overline { { 0 . 4 6 5 0 \pm 0 . 0 4 1 4 } }$ </td></tr><tr><td>MMD-Only</td><td> $0 . 2 7 2 2 \pm 0 . 0 0 6 3$ </td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 4 2 7 8 \pm 0 . 0 0 7 7$ </td></tr><tr><td>DANN+MMD</td><td> $0 . 2 7 4 8 \pm 0 . 0 0 6 9$ </td><td> $1 . 0 0 0 0 \pm 0 . 0 0 0 0$ </td><td> $0 . 4 3 1 1 \pm 0 . 0 0 8 5$ </td></tr><tr><td>ADANN (proposed)</td><td> $\mathbf { 0 . 4 7 5 7 \pm 0 . 1 3 5 1 }$ </td><td> $\overline { { 0 . 9 8 8 7 \pm 0 . 0 1 5 4 } }$ </td><td> $\mathbf { \overline { { 0 . 6 3 2 2 \pm 0 . 1 1 2 6 } } }$ </td></tr><tr><td rowspan="5"> $A _ { h a r m o n i c } = 0 . 8$ </td><td>Source-Only</td><td> $0 . 2 6 6 7 \pm 0 . 0 0 4 0$ </td><td> $\mathbf { 1 . 0 0 0 0 } \pm 0 . 0 0 0 0$ </td><td> $\overline { { 0 . 4 2 1 1 \pm 0 . 0 0 5 0 } }$ </td></tr><tr><td>DANN</td><td> $\overline { { 0 . 2 6 5 3 \pm 0 . 0 0 2 8 } }$ </td><td> $\overline { { 1 . 0 0 0 0 \pm 0 . 0 0 0 0 } }$ </td><td> $\overline { { 0 . 4 1 9 3 \pm 0 . 0 0 3 5 } }$ </td></tr><tr><td>MMD-Only</td><td> $0 . 2 6 3 5 \pm 0 . 0 0 0 1$ </td><td> $\overline { { 1 . 0 0 0 0 \pm 0 . 0 0 0 0 } }$ </td><td> $\overline { { 0 . 4 1 7 1 \pm 0 . 0 0 0 1 } }$ </td></tr><tr><td>DANN+MMD</td><td> $\overline { { 0 . 2 6 3 4 \pm 0 . 0 0 0 0 } }$ </td><td> $\overline { { 1 . 0 0 0 0 \pm 0 . 0 0 0 0 } }$ </td><td></td></tr><tr><td>ADANN (proposed)</td><td> $\overline { { { \bf 0 . 3 7 3 6 \pm 0 . 0 5 9 4 } } }$ </td><td> $\overline { { 0 . 9 6 3 7 \pm 0 . 0 5 6 1 } }$ </td><td> $\overline { { 0 . 4 1 7 0 \pm 0 . 0 0 0 0 } }$ </td></tr><tr><td rowspan="5"> $A _ { h a r m o n i c } = 1 . 0$ </td><td>Source-Only</td><td> $\overline { { 0 . 2 6 4 1 \pm 0 . 0 0 1 4 } }$ </td><td> $\mathbf { \overline { { 1 . 0 0 0 0 } } \pm 0 . 0 0 0 0 }$ </td><td> $\mathbf { 0 . 5 3 5 8 \pm 0 . 0 6 6 3 }$   $\overline { { 0 . 4 1 7 9 \pm 0 . 0 0 1 8 } }$ </td></tr><tr><td>DANN</td><td> $0 . 2 6 6 5 \pm 0 . 0 0 5 1$ </td><td> $\overline { { 1 . 0 0 0 0 \pm 0 . 0 0 0 0 } }$ </td><td> $\overline { { 0 . 4 2 0 9 \pm 0 . 0 0 6 4 } }$ </td></tr><tr><td> $\overline { { { \bf M M D - O n l y } } }$ </td><td> $0 . 2 6 5 5 \pm 0 . 0 0 1 8$ </td><td> $\overline { { 0 . 9 9 6 5 \pm 0 . 0 0 6 4 } }$ </td><td></td></tr><tr><td>DANN+MMD</td><td> $\overline { { 0 . 2 6 5 1 \pm 0 . 0 0 2 2 } }$ </td><td> $0 . 9 9 8 8 \pm 0 . 0 0 2 5$ </td><td> $\overline { { 0 . 4 1 9 3 \pm 0 . 0 0 2 0 } }$ </td></tr><tr><td>ADANN (proposed)</td><td> $\mathbf { 0 . 3 9 9 9 } \pm 0 . 1 2 3 7$ </td><td> $\overline { { 0 . 9 1 5 5 \pm 0 . 1 3 3 4 } }$ </td><td> $\overline { { 0 . 4 1 8 9 \pm 0 . 0 0 2 5 } }$   $\mathbf { \overline { { 0 . 5 3 9 0 \pm 0 . 1 0 8 9 } } }$ </td></tr></table>

Under harmonic interference, all four benchmarks reach a recall of (nearly) 1 with a precision between 0.263 and 0.304. A detector that flags every window has recall 1 and precision equal to the fault prevalence, and the benchmark results are at or close to this reference; for example, DANN+MMD at $A _ { h a r m o n i c } = 0 . 8$ has precision 0.2634 with zero variance across runs. The benchmarks therefore raise alarms on nearly all target windows, including the normal ones.

ADANN is the only method that departs from this solution: it raises precision to between 0.374 and 0.476 at a small cost in recall, and its mean F1-score is the highest at all three amplitudes (0.632, 0.536, and 0.539, compared with at most 0.465 for the benchmarks). A plausible explanation, consistent with Remark 1, is that harmonic interference shifts the target’s normal features toward the region that the source-trained classifier associates with faults. The benchmarks have no mechanism that evaluates the classifier on target normal data and therefore flag these windows, whereas the ACD loss penalizes exactly these false alarms.

In summary, the robustness experiments support a narrower conclusion than a uniform advantage of ADANN. Where the ACD loss matters, it matters through false alarms: ADANN avoids the all-alarm failure mode under harmonic interference and occasionally under weak time-varying faults, and its effect weakens when the normal adaptation data are themselves heavily corrupted. Under the other synthetic conditions, ADANN performs comparably to the strongest benchmarks.

## 5 Conclusions and Future Work

In this paper, a cross-asset knowledge transfer framework is proposed to achieve robust anomaly detection on a target bearing using only its early-stage normal data. The significance of this framework lies in its practical applicability to real-world industrial scenarios, where labeled fault data for every bearing are unavailable, yet early and accurate fault detection is critical for preventing catastrophic failures and minimizing maintenance costs. By automating the labeling process for the source bearing, the framework alleviates the need for costly and subjective manual annotation. Furthermore, by incorporating both adversarial knowledge transfer and explicit distributional matching through MMD loss, along with a novel adapted classificationdomain loss that leverages normal target data, the framework ensures that the diagnostic knowledge learned from the source bearing is effectively transferred and calibrated to the target bearing.

Comprehensive experiments on the publicly available Bearing Cage Full Lifecycle Dataset demonstrated the superiority of the proposed framework. Afterwards, two extended case studies introducing controlled noise into the target domain further validated the framework's robustness. In the future, adaptive mechanisms to dynamically adjust the weight of the ACD loss based on estimated noise levels or the confidence of the normal target data representation could be explored. In addition, the generalizability of the ADANN model to scenarios involving multiple source bearings, significantly different operating conditions, and transfer between fundamentally different components (e.g., from bearings to gears) could be further investigated.

## Declaration of interests

All authors certify that they have no affiliation with or involvement in any organization or entity with any financial interest or non-financial interest in the subject matter or materials discussed in this manuscript.

## Acknowledgements

This work was supported by Shanghai Pujiang Talents Program (Grant No. 24PJC016) and Youth Science Fund Project (Category C) of the National Natural Science (Grant No. 72501106).

## References

Cao, Z., Long, M., Wang, J., & Jordan, M. I. (2018). Partial transfer learning with selective adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 2724–2732).

Chen, Y., Wang, L., & Zhao, X. (2025). Cluster edge-based active learning with batch queries for aircraft engine RUL prediction under label scarcity. Annals of Operations Research.

Deng, J., Wen, G., Lei, Z., Xu, Q., Su, Y., & Zhang, Z. (2026). A digital twin of life-cycle bearing based on a defect-guided conditional diffusion model for predicting remaining life under limited degradation data. ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering, 12(1), 04025114.

Fan, C., Wang, P., Ma, H., Zhang, Y., Ma, Z., Yin, X., Zhang, X., & Zhao, S. (2024). Performance degradation assessment of rolling bearing cage failure based on enhanced CycleGAN. Expert Systems with Applications, 255, 124697. https://doi.org/10.1016/j.eswa.2024.124697

Fan, C., Wang, P., Zhang, Y., Ma, H., Li, X., & Wang, Q. (2025). Digital twin assisted degradation assessment of bearing cage performance. IEEE Transactions on Industrial Informatics. https://doi.org/10.1109/TII.2025.3552655

Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., Marchand, M., & Lempitsky, V. (2016). Domain-adversarial training of neural networks. Journal of Machine Learning Research, 17(59), 1–35.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative adversarial nets. In Advances in Neural Information Processing Systems (Vol. 27).

Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B., & Smola, A. (2006). A kernel method for the two-sample-problem. In Advances in Neural Information Processing Systems (Vol. 19).

Guo, L., Lei, Y., Xing, S., Yan, T., & Li, N. (2019). Deep convolutional transfer learning network: A new method for intelligent fault diagnosis of machines with unlabeled data. IEEE Transactions on Industrial Electronics, 66(9), 7316–7325. https://doi.org/10.1109/TIE.2018.2877090

Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the knowledge in a neural network (arXiv:1503.02531). arXiv. https://doi.org/10.48550/arXiv.1503.02531

Jardine, A. K. S., Lin, D., & Banjevic, D. (2006). A review on machinery diagnostics and prognostics implementing condition-based maintenance. Mechanical Systems and Signal Processing, 20(7), 1483–1510. https://doi.org/10.1016/j.ymssp.2005.09.012

Kairouz, P., et al. (2021). Advances and open problems in federated learning. Foundations and Trends in Machine Learning, 14(1–2), 1–210. https://doi.org/10.1561/2200000083

Lee, J., Ardakani, H. D., Yang, S., & Bagheri, B. (2015). Industrial big data analytics and cyberphysical systems for future maintenance & service innovation. Procedia CIRP, 38, 3–7. https://doi.org/10.1016/j.procir.2015.08.026

Lee, S., Choi, J., & Son, Y. (2024). Efficient visibility algorithm for high-frequency time-series: Application to fault diagnosis with graph convolutional network. Annals of Operations Research, 339(1), 813–833. https://doi.org/10.1007/s10479-022-05071-x

Lei, Y., Yang, B., Jiang, X., Jia, F., Li, N., & Nandi, A. K. (2020). Applications of machine learning to machine fault diagnosis: A review and roadmap. Mechanical Systems and Signal Processing, 138, 106587. https://doi.org/10.1016/j.ymssp.2019.106587

Li, F., Chen, J., He, S., & Zhou, Z. (2021). Layer regeneration network with parameter transfer and knowledge distillation for intelligent fault diagnosis of bearing using class unbalanced sample. IEEE Transactions on Instrumentation and Measurement, 70, 1–10.

Li, L., Fan, Y., Tse, M., & Lin, K.-Y. (2020). A review of applications in federated learning. Computers & Industrial Engineering, 149, 106854. https://doi.org/10.1016/j.cie.2020.106854

Li, Y., Xie, T., Liu, C., & Shi, Z. (2025). Pseudo replay-based class continual learning for online new category anomaly detection in advanced manufacturing. IISE Transactions, 57(12), 1407– 1421. https://doi.org/10.1080/24725854.2024.2428642

Liu, S., Jiang, H., Wang, Y., Zhu, K., & Liu, C. (2022). A deep feature alignment adaptation network for rolling bearing intelligent fault diagnosis. Advanced Engineering Informatics, 52, 101598. https://doi.org/10.1016/j.aei.2022.101598

Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2018). Conditional adversarial domain adaptation.   
In Advances in Neural Information Processing Systems (Vol. 31).

Randall, R. B., & Antoni, J. (2011). Rolling element bearing diagnostics: A tutorial. Mechanical Systems and Signal Processing, 25(2), 485–520. https://doi.org/10.1016/j.ymssp.2010.07.017

Shi, J. (2023). In-process quality improvement: Concepts, methodologies, and applications. IISE Transactions, 55(1), 2–21. https://doi.org/10.1080/24725854.2022.2059725

Shi, Z., Li, Y., & Liu, C. (2022). Knowledge distillation-enabled multi-stage incremental learning for online process monitoring in advanced manufacturing. In 2022 IEEE International Conference on Data Mining Workshops (ICDMW) (pp. 860–867). IEEE.

Shi, Z., Mamun, A. A., Kan, C., Tian, W., & Liu, C. (2023). An LSTM-autoencoder based online side channel monitoring approach for cyber-physical attack detection in additive manufacturing. Journal of Intelligent Manufacturing, 34(4), 1815–1831. https://doi.org/10.1007/s10845-021- 01879-9

Sun, W., Lei, S., Wang, L., Liu, Z., & Zhang, Y. (2021). Adaptive federated learning and digital twin for industrial internet of things. IEEE Transactions on Industrial Informatics, 17(8), 5605– 5614.

Tang, S., Yuan, S., & Zhu, Y. (2020). Deep learning-based intelligent fault diagnosis methods toward rotating machinery. IEEE Access, 8, 9335–9346. https://doi.org/10.1109/ACCESS.2019.2963092

Tran, P. H., Ahmadi Nadi, A., Nguyen, T. H., Tran, K. D., & Tran, K. P. (2022). Application of machine learning in statistical process control charts: A survey and perspective. In K. P. Tran (Ed.), Control charts and machine learning for anomaly detection in manufacturing (pp. 7–42). Springer.

Udmale, S. S., Nath, A. G., Singh, D., & Singh, S. K. (2026). Data-driven approach under varying operating conditions for bearing fault diagnosis. Annals of Operations Research. https://doi.org/10.1007/s10479-026-07055-7

Wang, K. I.-K., Zhou, X., Liang, W., Yan, Z., & She, J. (2022). Federated transfer learning based cross-domain prediction for smart manufacturing. IEEE Transactions on Industrial Informatics, 18(6), 4088–4096. https://doi.org/10.1109/TII.2021.3088057

Wu, J., Zhao, Z., Sun, C., Yan, R., & Chen, X. (2020). Few-shot transfer learning for intelligent fault diagnosis of machine. Measurement, 166, 108202. https://doi.org/10.1016/j.measurement.2020.108202

Xu, N., Li, Y., Chen, F., & Xu, R. (2026). C2DAN: Conditional domain adversarial network with SupCon loss for mitigating class confusion in anomaly detection of DFOS. Annals of Operations Research. https://doi.org/10.1007/s10479-026-07362-z

Yan, L., Du, S., Wang, S., et al. (2026). MAEFF: Missing-aware adaptive expert fusion framework for multi-sensor bearing fault diagnosis. Annals of Operations Research. https://doi.org/10.1007/s10479-026-07381-w

Yang, B., Lei, Y., Jia, F., & Xing, S. (2019). An intelligent fault diagnosis approach based on transfer learning from laboratory bearings to locomotive bearings. Mechanical Systems and Signal Processing, 122, 692–706. https://doi.org/10.1016/j.ymssp.2018.12.051

Zhao, H., Tachet des Combes, R., Zhang, K., & Gordon, G. (2019). On learning invariant representations for domain adaptation. In Proceedings of the 36th International Conference on Machine Learning (pp. 7523–7532). PMLR.

Zhao, R., Yan, R., Chen, Z., Mao, K., Wang, P., & Gao, R. X. (2019). Deep learning and its applications to machine health monitoring. Mechanical Systems and Signal Processing, 115, 213–237. https://doi.org/10.1016/j.ymssp.2018.05.050

## Appendix A Proof of Theorem 1

Throughout the appendix, ln denotes the natural logarithm and $H ( p ) = - p \mathrm { l n } p - ( 1 - p ) \mathrm { l n } ( 1 -$ $p )$ the binary entropy in nats. Write $f = G _ { f } ( x )$ for the feature produced from an input window �. By the labeling scheme of Sect. 3.3, each source window carries the label normal or abnormal; $p _ { s } ^ { N }$ and $p _ { s } ^ { A }$ denote the class-conditional densities of the induced features, with proportions $\pi _ { N }$ and $\pi _ { A } = 1 - \pi _ { N }$ fixed by the labels, so that the source marginal is given by Eq. (13). The adaptation set $\mathcal { D } _ { B } ^ { T }$ contains normal windows only, so the density $p _ { t }$ of its features is purely normal. Let $D ( f ) : = G _ { d } ( f ) \in ( 0 , 1 )$ denote the probability that the domain classifier assigns to the source domain. At the population level, the domain classification loss of Eq. (7) is

$$
\mathcal { L } _ { d } = - \mathbb { E } _ { f \sim p _ { s } } [ \ln D ( f ) ] - \mathbb { E } _ { f \sim p _ { t } } \bigl [ \ln \bigl ( 1 - D ( f ) \bigr ) \bigr ] .\tag{A. 1}
$$

In Eq. (12), only the term $- \alpha \mathcal { L } _ { d }$ involves $\theta _ { d }$ , so the inner maximization over $\theta _ { d }$ is equivalent to minimizing $\mathcal { L } _ { d }$ . The remaining terms do not depend on $\theta _ { d } \colon \mathcal { L } _ { y }$ and $\mathcal { L } _ { A C D }$ depend on $\left( \theta _ { f } , \theta _ { y } \right)$ , and $\mathcal { L } _ { m m d }$ , the squared MMD of Eq. (9) computed between the marginals $p _ { s }$ and $p _ { t }$ , depends on $\theta _ { f }$ only.

Three assumptions are used, corresponding to (i) to (iii) of Theorem 1.

(i) Capacity. $G _ { f } , G _ { y }$ , and $G _ { d }$ are sufficiently flexible that $G _ { d }$ can realize the optimal classifier of Lemma 1, and $G _ { f }$ can realize feature maps under which the densities above exist and the aligned state of Theorem 1 is attainable.

(ii) Optimal adversary. At each training step, $G _ { d }$ is updated to the minimizer of $\mathcal { L } _ { d }$ for the current $G _ { f }$ before $G _ { f }$ and $G _ { y }$ are updated. It therefore suffices to analyze the value function $C \left( \theta _ { f } , \theta _ { y } \right) : = \operatorname* { m a x } _ { \theta _ { d } } \mathcal { E } \left( \theta _ { f } , \theta _ { y } , \theta _ { d } \right)$

(iii) Classification correctness. At the state of interest, $G _ { y }$ labels all source windows and all target normal windows correctly: there is a measurable abnormal region $R _ { A }$ of the feature space such that

$$
\begin{array} { r } { p _ { s } ^ { A } ( R _ { A } ) = 1 , \qquad p _ { s } ^ { N } ( R _ { A } ) = 0 , \qquad p _ { t } ( R _ { A } ) = 0 . } \end{array}\tag{A. 2}
$$

This is the property that $\mathcal { L } _ { y }$ and $\mathcal { L } _ { A C D }$ drive toward. Equation (A.2) states it in exact (zero-error) form so that the characterization below is sharp; the same characterization holds approximately when the classification losses are small. We write $R _ { N }$ for the complement of $R _ { A }$

Lemma 1 (optimal domain classifier; cf. Goodfellow et al., 2014). For any fixed $G _ { f }$ , and hence fixed marginals $p _ { s }$ and $p _ { t }$ regardless of the internal mixture structure of Eq. (13), the domain classifier that minimizes $\mathcal { L } _ { d }$ is

$$
D ^ { * } ( f ) = { \frac { p _ { s } ( f ) } { p _ { s } ( f ) + p _ { t } ( f ) } }\tag{A. 3}
$$

for every � with $p _ { s } ( f ) + p _ { t } ( f ) > 0$ , and is arbitrary elsewhere. In particular, $D ^ { * } ( f ) = 1 / 2$ for almost every � if and only if $p _ { s } = p _ { t }$ almost everywhere.

Proof. Writing the expectations in Eq. (A.1) as integrals, $\begin{array} { r } { \mathcal { L } _ { d } = - \int p _ { s } ( f ) \mathrm { l n } D ( f ) d f - } \end{array}$ $\textstyle { \int p _ { t } ( f ) \ln \bigl ( 1 - D ( f ) \bigr ) d f }$ . For each fixed � with $p _ { s } ( f ) + p _ { t } ( f ) > 0$ , the integrand $h ( d ) =$ $- p _ { s } ( f ) \mathrm { l n } d - p _ { t } ( f ) \mathrm { l n } ( 1 - d )$ is strictly convex in $d \in ( 0 , 1 )$ , and its derivative $h ^ { \prime } ( d ) =$ $- p _ { s } ( f ) / d + p _ { t } ( f ) / ( 1 - d )$ vanishes at $d ^ { * } = p _ { s } ( f ) / \big ( p _ { s } ( f ) + p _ { t } ( f ) \big )$ , which is therefore the pointwise minimizer. By assumption (i), �(�) can be chosen independently at each $f ,$ so the pointwise minimizers form the global minimizer $D ^ { * }$ . If $p _ { s } = p _ { t }$ , Eq. (A.3) gives $D ^ { * } = 1 / 2$ conversely, $D ^ { * } \equiv 1 / 2$ forces $p _ { s } = p _ { t }$ ▫

Lemma 2 (cf. Goodfellow et al., 2014). Let $m = ( p _ { s } + p _ { t } ) / 2$ and let $\lceil \mathrm { S D } ( p _ { s } \Vert p _ { t } )$ $: = \textstyle { \frac { 1 } { 2 } } \mathrm { K L } ( p _ { s } \| m ) + \frac { 1 } { 2 } \mathrm { K L } ( p _ { t } \| m ) \ge 0$ denote the Jensen-Shannon divergence. Then

$$
\mathcal L _ { d } ( D ^ { * } ) = \ln 4 - 2 \ J \mathrm { S D } ( p _ { s } \| p _ { t } ) \leq \ln 4 ,\tag{A. 4}
$$

and consequently the alignment terms of Eq. (12) satisfy

$$
- \alpha \mathcal { L } _ { d } ( D ^ { * } ) + \mathcal { L } _ { m m d } = - \alpha \mathrm { l n } 4 + 2 \alpha \left\lceil { \mathrm { S D } ( p _ { s } \| p _ { t } ) + \mathcal { L } _ { m m d } \geq - \alpha \mathrm { l n } 4 } \right\rceil ,\tag{A. 5}
$$

with equality if and only if $p _ { s } = p _ { t }$ almost everywhere.

Proof. Since $p _ { s } / ( p _ { s } + p _ { t } ) = p _ { s } / ( 2 m )$ and $p _ { t } / ( p _ { s } + p _ { t } ) = p _ { t } / ( 2 m )$ , substituting Eq. (A.3) into Eq. (A.1) yields $- \mathbb { E } _ { p _ { s } } [ \mathrm { l n } D ^ { * } ( f ) ] = \mathrm { l n } 2 - \mathrm { K L } ( p _ { s } \| m )$ and $- \mathbb { E } _ { p _ { t } } \big [ \mathrm { l n } \big ( 1 - D ^ { * } ( f ) \big ) \big ] = \mathrm { l n } 2 -$ $\mathrm { K L } ( p _ { t } \| m )$ . Adding the two identities gives $\mathcal { L } _ { d } ( D ^ { \ast } ) = \mathrm { l n } 4 - 2 \mathrm { J S D } ( p _ { s } \| p _ { t } )$ , and the inequality in Eq. (A.4) follows from the non-negativity of the Jensen-Shannon divergence, with equality if and only if $p _ { s } = p _ { t }$ . Since $\mathcal { L } _ { m m d }$ does not involve $\theta _ { d }$ , it passes through the inner maximization unchanged; multiplying Eq. (A.4) by −� and adding $\mathcal { L } _ { m m d }$ gives Eq. (A.5). The squared MMD is non-negative and, for the characteristic RBF kernel, vanishes if and only if $p _ { s } = p _ { t }$ . Hence both non-negative terms on the right-hand side of Eq. (A.5) vanish exactly when $p _ { s } = p _ { t }$ . ▫

Lemma 3. Under assumption (iii), the Jensen-Shannon divergence between the two marginals satisfies

$$
\mathrm { J S D } ( p _ { s } \| p _ { t } ) \geq \delta ( \pi _ { A } ) = \ln 2 - \frac { 2 - \pi _ { A } } { 2 } H \left( \frac { 1 - \pi _ { A } } { 2 - \pi _ { A } } \right) .\tag{A. 6}
$$

The function � is continuous and strictly increasing on [0,1], with $\delta ( 0 ) = 0$ and $\delta ( 1 ) = \ln 2$ , and $\begin{array} { r } { \delta ( \pi _ { A } ) = H ( \pi _ { A } / 2 ) - \frac { 1 } { 2 } H ( \pi _ { A } ) } \end{array}$ , the Jensen-Shannon divergence between Bernoulli $\left( \pi _ { A } \right)$ and Bernoulli(0). Equality in Eq. (A.6) holds if and only if

$$
p _ { s } ^ { N } ( f ) = p _ { t } ( f ) \quad \mathrm { f o r a l m o s t e v e r y } f ,\tag{A. 7}
$$

that is, if and only if the source normal features coincide with the target normal features, while the abnormal features occupy the region $R _ { A }$ , which is excluded from the target support by Eq. (A.2).

Proof. Let � be a uniform Bernoulli variable indicating the domain, $P ( Z = 0 ) = P ( Z = 1 ) =$ $1 / 2$ , and let � follow $p _ { s }$ given $Z = 0$ and $p _ { t }$ given $Z = 1$ . The domain posterior $P ( Z = 0 \mid F =$ $f ) = p _ { s } ( f ) / \big ( p _ { s } ( f ) + p _ { t } ( f ) \big )$ is the optimal classifier of Eq. (A.3), and the Jensen-Shannon divergence equals the mutual information between domain and feature, $\begin{array} { r } { \mathrm { J S D } ( p _ { s } \| p _ { t } ) = I ( Z ; F ) = } \end{array}$ ln $\displaystyle { 1 2 - H ( Z \mid F ) }$ . Now let $V : = \mathbb { I } ( F \in R _ { A } )$ indicate that a feature falls in the abnormal region. By the data-processing inequality, $I ( Z ; F ) \ge I ( Z ; V ) = \ln 2 - H ( Z \mid V )$ . Under Eq. (A.2), $P ( V = 1 \mid$ $Z = 0 ) = p _ { s } ( R _ { A } ) = \pi _ { A }$ and $P ( V = 1 \mid Z = 1 ) = p _ { t } ( R _ { A } ) = 0$ . Hence $P ( V = 1 ) = \pi _ { A } / 2$ and $P ( V = 0 ) = ( 2 - \pi _ { A } ) / 2$ , which gives $P ( Z = 0 \mid V = 1 ) = 1$ $P ( Z = 0 \mid V = 0 ) =$ $( 1 - \pi _ { A } ) / ( 2 - \pi _ { A } )$ , and

$$
H ( Z \mid V ) = P ( V = 0 ) H ( Z \mid V = 0 ) + P ( V = 1 ) H ( Z \mid V = 1 ) = \frac { 2 - \pi _ { A } } { 2 } H \left( \frac { 1 - \pi _ { A } } { 2 - \pi _ { A } } \right) .
$$

Combining these steps yields Eq. (A.6). Since $I ( Z ; V )$ is the Jensen-Shannon divergence between the conditional distributions of � given $Z = 0$ and $Z = 1$ , namely Bernoulli $\left( \pi _ { A } \right)$ and Bernoulli(0) , it also equals $\begin{array} { r } { H ( \pi _ { A } / 2 ) - \frac { 1 } { 2 } H ( \pi _ { A } ) - \frac { 1 } { 2 } H ( 0 ) = H ( \pi _ { A } / 2 ) - \frac { 1 } { 2 } H ( \pi _ { A } ) } \end{array}$ . For monotonicity, differentiate $\begin{array} { r } { \delta ( \pi _ { A } ) = H ( \pi _ { A } / 2 ) - \frac { 1 } { 2 } H ( \pi _ { A } ) } \end{array}$ using $H ^ { \prime } ( p ) = \ln \bigl ( ( 1 - p ) / p \bigr )$ :

$$
\frac { d \delta } { d \pi _ { A } } = \frac { 1 } { 2 } H ^ { \prime } \left( \frac { \pi _ { A } } { 2 } \right) - \frac { 1 } { 2 } H ^ { \prime } ( \pi _ { A } ) = \frac { 1 } { 2 } \bigg [ \mathrm { l n } \frac { 2 - \pi _ { A } } { \pi _ { A } } - \mathrm { l n } \frac { 1 - \pi _ { A } } { \pi _ { A } } \bigg ] = \frac { 1 } { 2 } \mathrm { l n } \frac { 2 - \pi _ { A } } { 1 - \pi _ { A } } > 0 \quad \mathrm { f o r ~ } \pi _ { A } \in ( 0 , 1 ) ,
$$

since $( 2 - \pi _ { A } ) / ( 1 - \pi _ { A } ) > 1$ . Because � is continuous on [0,1], it is therefore strictly increasing on the whole closed interval. The endpoint values follow from $H ( 1 / 2 ) = \ln 2$ and $H ( 0 ) = 0$

For the equality condition, equality in the data-processing step requires the domain posterior to be a function of �, that is, constant on $R _ { A }$ and on $R _ { N }$ . On $R _ { A }$ it equals 1 because $p _ { t } ( R _ { A } ) = 0$ . On $R _ { N }$ , where $p _ { s } = \pi _ { N } p _ { s } ^ { N }$ by Eq. (A.2), the posterior $\pi _ { N } p _ { s } ^ { N } ( f ) / \big ( \pi _ { N } p _ { s } ^ { N } ( f ) + p _ { t } ( f ) \big )$ is constant in � if and only if $p _ { t }$ is proportional to $p _ { s } ^ { N }$ on $R _ { N }$ . Since both densities integrate to one over $R _ { N }$ by Eq. (A.2), the proportionality constant is one, which gives Eq. (A.7) and the posterior value $\pi _ { N } / ( 1 +$ $\pi _ { N } )$ on $R _ { N }$ . Conversely, under Eq. (A.7) the posterior equals $\pi _ { N } / ( 1 + \pi _ { N } )$ on $R _ { N }$ and 1 on $R _ { A }$ , so $H ( Z \mid F ) = H ( Z \mid V )$ and equality holds throughout. ▫

Proof of Theorem 1. By assumption (ii) and Lemma 2, the value function of Eq. (12) decomposes as

$$
C \big ( \theta _ { f } , \theta _ { y } \big ) = \mathcal { L } _ { y } + \mathcal { L } _ { A C D } + \mathcal { L } _ { m m d } - \alpha \mathrm { l n } 4 + 2 \alpha \mathrm { J S D } ( p _ { s } \| p _ { t } ) .\tag{A. 8}
$$

Lower bound. Under assumption (iii), Lemma 3 gives $\lceil \mathrm { S D } ( p _ { s } \| p _ { t } ) \geq \delta ( \pi _ { A } )$ , so the domainadversarial component satisfies $- \alpha \mathcal { L } _ { d } ( D ^ { * } ) = - \alpha \mathrm { l n } 4 + 2 \alpha  \rfloor \mathrm { S D } ( p _ { s } \| p _ { t } ) \geq - \alpha \mathrm { l n } 4 + 2 \alpha \delta ( \pi _ { A } )$ which is Eq. (14); the properties of � are those of Lemma 3. In particular, as $\pi _ { A }  0$ the gap vanishes and the bound reduces to −�ln4; by Lemma 2 the unconstrained optimum of the adversarial game is then $p _ { s } = p _ { t }$ , with $D ^ { * } \equiv 1 / 2$ by Lemma 1, the classical domain-adversarial equilibrium.

Attainment. Suppose that the aligned state of Eq. (A.7) holds: $p _ { s } ^ { N } = p _ { t }$ almost everywhere, with the abnormal features confined to $R _ { A }$ . Then equality holds in Lemma 3, and the domainadversarial component equals $- \alpha \mathrm { l n } 4 + 2 \alpha \delta ( \pi _ { A } )$ . By Lemma 1, the optimal domain classifier is $D ^ { * } ( f ) = 1$ on $R _ { A }$ , where $p _ { t } = 0$ , and $D ^ { * } ( f ) = \pi _ { N } / ( 1 + \pi _ { N } )$ on $R _ { N }$ , where $p _ { s } = \pi _ { N } p _ { t }$ Moreover, writing $\mu _ { s } = \pi _ { N } \mu _ { s } ^ { N } + \pi _ { A } \mu _ { s } ^ { A }$ for the kernel mean embedding of the source marginal and using $\mu _ { t } = \mu _ { s } ^ { N }$ , which follows from $p _ { t } = p _ { s } ^ { N }$ , the population MMD evaluates to

$$
\mathcal { L } _ { m m d } = \Vert \mu _ { s } - \mu _ { t } \Vert _ { \mathcal { H } } ^ { 2 } = \pi _ { A } ^ { 2 } \Vert \mu _ { s } ^ { A } - \mu _ { s } ^ { N } \Vert _ { \mathcal { H } } ^ { 2 } = \mathcal { L } _ { m m d } ^ { \dag } .
$$

Conversely, if Eq. (A.2) holds but $p _ { s } ^ { N } \neq p _ { t }$ on a set of positive measure, the equality condition of Lemma 3 fails and the bound in Eq. (14) is strict. Hence, within the class of correctly classifying feature maps, which is the class relevant at deployment, the bound in Eq. (14) is attained exactly at the aligned state $p _ { s } ^ { N } = p _ { t }$