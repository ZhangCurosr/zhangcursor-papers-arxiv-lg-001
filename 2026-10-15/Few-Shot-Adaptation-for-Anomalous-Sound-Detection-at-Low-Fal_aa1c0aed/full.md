# Few-Shot Adaptation for Anomalous Sound Detection at Low False-Positive Rate

Simone Lugani<sup>∗</sup>, Christof Pichler<sup>†</sup>, Hannes Wegleiter<sup>†</sup>, Christoph Feilmayr<sup>‡</sup>, Rodolfo Zunino<sup>∗</sup>, Edoardo Ragusa<sup>∗</sup> <sup>∗</sup>DITEN, University of Genoa, Genoa, Italy; contact: simone.lugani@edu.unige.it

<sup>†</sup>Christian Doppler Laboratory for Measurement Systems for Harsh Operating Conditions, EMS, TU Graz, Graz, Austria <sup>‡</sup> voestalpine Stahl GmbH, Linz, Austria

Abstract—Reliable fault detection is crucial for industrial machinery in large-scale facilities, where unexpected failures can cause costly downtime and production interruptions. Although acoustic monitoring provides a scalable solution for machine condition monitoring, its deployment is often constrained by high false-positive rates. We propose a two-step approach to automatic anomalous sound detection: first, when only normal operating audio samples are available, a regularized unsupervised model is deployed; then, after the first machine fault, a fewshot adaptation strategy exploits the anomalous samples acquired during the fault to improve the system. This adaptation increases the area under the precision-recall curve (AUPR) from 23.0 to 64.0 and improves detection in the low false-positive-rate region, with the partial area under the receiver operating characteristic curve (pAUROC) increasing from 39.6 to 72.8. We also show that covariance regularization improves both prediction accuracy and training stability in the unsupervised model. These results provide a practical strategy for acoustic anomaly detection systems, enabling improved accuracy and reduced false alarms.

Keywords—Anomalous Sound Detection, Fault detection, Predictive Maintenance, Few-Shot Learning

## I. INTRODUCTION

Reliable operation of industrial machinery is essential in continuous-process industries, where unexpected failures can cause costly downtime and production interruptions [1]. This is vital in pig iron production, where the blast furnace stock house prepares raw materials before charging, and large vibrating screens make early fault detection essential for reliable operation. While vibration-based monitoring is well established, its use in large-scale facilities is often limited by the required sensor effort and installation cost [2]. Acoustic condition monitoring uses microphones to capture machine-related sound emissions [3]. In stock house environments, however, multiple concurrently operating screens, time-varying background noise, and additional disturbances can mask weak fault-related signatures, making reliable detection difficult [4].

Instead of requiring large labelled fault datasets, unsupervised anomalous sound detection (ASD) systems learn normal machine behaviour and identify deviations as anomalies [4]. In recent years, this field has advanced strongly, in particular through the DCASE challenge series, where a wide range of data-driven methods for machine condition monitoring has been developed [5]–[10]. Recent DCASE tasks have also moved towards first-shot and domain-generalisation settings, which are highly relevant for practical applications [11]–[15].

![](images/cd2a6fe52cf556f4026dbaa6d827976441506e17399a447933a0d9783f1b7130.jpg)  
Fig. 1. Block diagram of the ASD framework: the log-mel spectrogram is compressed by the CVAE; then, a GMM is used to compute the anomaly score based on vector similarity with the learned normal representation.

However, a gap remains between current ASD research and the requirements of real industrial applications. In large-scale facilities, the system must operate at very low false-positive rates (FPR), since even a small number of false alarms can limit the usability and acceptance of monitoring systems. At the same time, real-world acoustic conditions are highly complex due to multiple simultaneously operating machines, changing operating states, and additional disturbing sound sources [4]. While ASD has already been applied to such harsh acoustic environments [16], [17], the resulting FPRs often remain a limiting factor for reliable deployment. In addition, a small number of fault samples is often available after the first observed failure, creating an opportunity to improve the separation between normal and faulty conditions and, in particular, to reduce false alarms. This setting can be viewed as a few-shot adaptation scenario, where only limited anomalous data are available for model adaptation.

We propose a two-stage ASD framework for industrial applications that enables targeted refinement of an unsupervised model using a small number of fault samples. First, a convolutional variational autoencoder (CVAE) extracts the features, and a Gaussian mixture model (GMM) models the normal operating condition using only normal data. Later, after the first observed fault, the CVAE is fine-tuned with a few anomalous samples from a single machine fault, while the GMM is kept fixed as a reference for the normal latent distribution. This refinement improves the separation between normal and faulty conditions, with focus on the low-FPR operating region, after the first machine fault. In addition, we investigate covariance regularization in the GMM, which improves robustness in the unsupervised setting. The evaluation is conducted on real-world acoustic data from an operating industrial blast furnace stock house, demonstrating relevance under realistic deployment conditions. The proposal outperforms the baselines in terms of overall AUPR and detection capabilities under stringent false-alarm requirements.

## II. METHODS

We propose a two-stage learning paradigm for ASD in industrial facilities. Section II-A describes the ASD framework. Initially, when only normal operating data are available, an unsupervised anomaly detection model is trained and deployed, as described in Section II-B. After the first machine fault occurs, the recorded sound data is used to enhance the model through a few-shot adaptation step integrating the newly acquired knowledge, as described in Section II-C.

## A. Anomalous Sound Detection Framework

Figure 1 illustrates the ASD framework. Audio samples from the facility are represented by their log-mel spectrograms. An undercomplete autoencoder, specifically a CVAE, compresses normal audio signals into a lower-dimensional latent space, which is then modeled by a GMM. During inference, the anomaly score of a new audio sample is defined as the negative log-likelihood of its latent representation under the GMM, so that higher scores indicate anomalous behaviour.

1) Preprocessing: For each input audio segment $s \in \mathbb { R } ^ { N }$ the log-mel spectrogram $\boldsymbol { x } \in \mathbb { R } ^ { F \times T }$ is computed and normalized to the [0,1] range using Min-Max normalization.

2) Convolutional Variational Autoencoder: The CVAE receives as input the log-mel spectrogram x extracted in the preprocessing stage and learns a compact representation of normal audio patterns in the latent space with dimensionality d. The encoder outputs two parameters for each latent dimension, namely the mean and the log-variance. Therefore, the latent representation is defined by a mean vector $\mu \in \mathbb { R } ^ { d }$ and a log-variance vector log $\sigma ^ { 2 } \in \mathbb { R } ^ { d }$

3) Gaussian Mixture Model Anomaly Scoring: The normal latent representation extracted by the CVAE is modeled by a GMM. For each input sample, the mean and log-variance vectors estimated by the CVAE are concatenated into the feature vector $h = [ \mu ; \log \sigma ^ { 2 } ] \in \mathbb { R } ^ { 2 d }$ , which is used as input to the GMM. During inference, the anomaly score $y \in \mathbb { R }$ is computed as the negative log-likelihood of h under the GMM, so that higher values correspond to a higher degree of anomaly.

## B. Unsupervised Training

First, an unsupervised model is trained with only normal operating data. The CVAE is trained to compress and reconstruct log-mel spectrograms of normal audio samples using the standard variational objective: a reconstruction term encourages accurate reconstruction, and a Kullback–Leibler (KL) regularization term promotes a smoother and more stable latent space. The unsupervised CVAE loss is defined as:

$$
\mathcal { L } _ { \mathrm { u n s u p } } = \| x - \hat { x } \| ^ { 2 } - \frac { \beta } { 2 } \sum _ { i = 1 } ^ { d } \left( 1 + \log \sigma _ { i } ^ { 2 } - \mu _ { i } ^ { 2 } - \sigma _ { i } ^ { 2 } \right)\tag{1}
$$

x and xˆ denote the input spectrogram and its reconstruction, $\mu _ { i }$ and $\sigma _ { i } ^ { 2 }$ denote the mean and variance of the i-th latent dimension, and $\beta$ is the weight of the KL regularization term.

Once the CVAE is trained, the GMM is fitted to the latent representations extracted from a dedicated set of normal recordings. For each sample, the mean and log-variance vectors are concatenated to form the GMM input features. Unlike previous works [16], covariance regularization is used during GMM training to obtain a more robust estimate of the latent data distribution. The resulting unsupervised model, combining the CVAE encoder with the GMM in the latent space, defines the first stage of the ASD system and serves as the starting point for the subsequent learning phase.

## C. Few-shot Adaptation with Single Machine Fault

The second training stage improves the ASD model after the first observed fault. The unsupervised CVAE is fine-tuned using a small set of anomalous samples collected from a single fault event. The objective is not to specialize the model to a specific machine, but to use real anomaly samples to improve the separation of normal and anomalous sounds in the facility.

The GMM obtained during the unsupervised stage is kept as the reference model for the normal latent distribution. This preserves a stable estimate of the normal latent distribution and prevents drift during fine-tuning. Each anomalous sample is used to generate a small number of augmented samples through minor temporal shifts, temporal cropping, amplitude scaling, and spectral coloration. The CVAE is then optimized using both normal and augmented anomalous samples. The loss function separates normal and anomalous samples in the latent space, while preserving the reconstruction behavior on normal data as a regularization term. The overall loss for the few-shot adaptation stage is defined as:

$$
\mathcal { L } _ { \mathrm { F S } } = \mathcal { L } _ { \mathrm { Z , N } } + \mathcal { L } _ { \mathrm { Z , A } } + \lambda \mathcal { L } _ { \mathrm { u n s u p } }\tag{2}
$$

$\mathcal { L } _ { \mathrm { Z } , \mathrm { N } }$ enforces consistency of normal samples with the baseline latent distribution, $\mathcal { L } _ { \mathrm { Z , A } }$ promotes the separation of anomalous samples from the GMM-based representation of normal data in the latent space, and $\mathcal { L } _ { \mathrm { u n s u p } }$ denotes the standard VAE loss as defined in Sec. II-B. $\mathcal { L } _ { \mathrm { Z } , \mathrm { N } }$ and $\mathcal { L } _ { \mathrm { u n s u p } }$ are computed for normal data only, and $\mathcal { L } _ { \mathrm { Z } , \mathrm { A } }$ for anomalous data only. λ controls the contribution of the unsupervised term.

Specifically, the two latent-space terms are defined as:

$$
\mathcal { L } _ { \mathrm { Z } , \mathrm { N } } = \operatorname* { m a x } \left( 0 , \mathrm { G M M } ( h ( x ) ) - m _ { \mathrm { N } } \right) ^ { 2 }\tag{3}
$$

$$
\mathcal { L } _ { \mathrm { Z , A } } = \operatorname* { m a x } \left( 0 , m _ { \mathrm { A } } - \mathrm { G M M } ( h ( x ) ) \right) ^ { 2 }\tag{4}
$$

where m<sub>N</sub> and $m _ { \mathrm { A } }$ denote the margins for normal and anomalous samples, respectively. They were selected manually from the pre-adaptation distribution of normal anomaly scores, without optimization on anomalous validation data, as no independent fault-validation set was available and using test faults would introduce data leakage.

## III. EXPERIMENTS

## A. Data Acquisition and Datasets

An acoustic measurement system was deployed in an operating blast furnace stock house to collect a custom real-world dataset under realistic industrial conditions [17]. Recordings were acquired over several months at 48 kHz using a Behringer ECM8000 condenser microphone connected to a Zoom F8n MultiTrack Field Audio Recorder and positioned approximately 5 m from the vibrating screen line. The acoustic environment is characterized by high sound pressure levels of approximately $9 5 \mathrm { d B } _ { \mathrm { S P I } }$ and multiple concurrent sound sources, including 13 vibrating screens under varying load conditions and additional non-stationary disturbances. All recordings include realistic background disturbances and were intentionally used without preprocessing to evaluate the system under practical deployment conditions. The normal operating dataset comprises 60 000 s of data for CVAE training, 60 000 s for GMM fitting, and 71 200 s of independent normal test data.

![](images/1be8f0596361fea996852a856157395b92225a4f333f1e3008c8d3614a200d9a.jpg)  
Fig. 2. Overview of the measurement setup and recreated fault case: facility layout with microphone and fault locations, vibrating screening machine, and physical reconstruction of typical mechanical failures.

Since real fault events are rarely available in sufficient quantity, dedicated on-site measurement campaigns were conducted with experienced plant personnel to physically recreate representative fault cases at multiple locations within the facility, including ruptured springs and detached metal plates causing characteristic metallic impact sounds. Figure 2 overviews the acquisition setup, and an example recreated fault case in which a detached metal plate emulates a typical mechanical failure scenario. The resulting real-world fault dataset comprises 400 s of recordings, acquired with one of three possible faulty machines running. These samples were divided into folds to train the models on a few samples from a single machine and test the method on the other two, as described in Sec. III-B.

## B. Experimental setup

Input audio recordings were segmented into 1.37 s samples with a hop size of 1.0 s. Log-mel spectrograms used an FFT length of 1024, hop size of 512, and 64 mel filter banks between 500 and 10 000 Hz, resulting in 64 × 128 inputs to the CVAE. The CVAE used a 16-dimensional latent space with five-layer convolutional encoder and transposed-convolutional decoder. We fixed $\beta ~ = ~ 0 . 0 0 1$ , λ = 0.1, $m _ { \mathrm { N } } = - 2 0$ , and $m _ { \mathrm { A } } = 5 0 0$ . The CVAE was trained with Adam, learning rate $1 0 ^ { - 3 }$ and early stopping. The GMM was fitted on a separate set of normal latent representations using full covariance matrices, 50 components, covariance regularization $1 0 ^ { - 5 }$ , and 10 random initializations. During few-shot adaptation, the CVAE was initialized from the unsupervised model and finetuned with Adam using learning rate $\bar { 1 } 0 ^ { - 5 }$ , early stopping, and exponential learning-rate decay with factor 0.95; fine-tuning batches were balanced between normal and anomalous samples, and batch-normalization layers were frozen. All models were implemented in PyTorch and trained on an NVIDIA RTX 4080 GPU. To account for result variability, each model was trained 10 times with different random seeds, and all metrics are reported as mean and standard deviation.

![](images/ce773ea9bfd0764708b59c23db5d1918ee15e6fe93bb4d1e391f50c87f1f961d.jpg)  
Fig. 3. AUPR performance of the ASD model with regularization after the unsupervised training step (UNS R) and after few-shot adaptation (FS), compared with the unsupervised state-of-the-art approach without regularization (UNS). The few-shot adaptation step yields a drastic improvement.

Normal data were divided into three sets for CVAE training, GMM fitting, and model testing. The anomalous data were collected from three faulty machines and divided into corresponding folds according to the machine where the fault was introduced. The recordings associated with the three machines amounted to 120 s, 220 s, and 60 s, respectively, and only one machine was used for few-shot adaptation in each configuration. The experiments were repeated three times, each time using two folds as the test set and the remaining one for few-shot training. Accordingly, $F _ { x }$ denotes the configuration where fold x is excluded from the test set and used to train the few-shot model. Global metrics are reported as the average over the three fold configurations.

## C. Comparison with State-of-the-Art

Figure 3 shows the area under the precision–recall curve (AUPR) obtained by three methods. UNS denotes the unsupervised baseline from [16], which combines an autoencoder and a GMM for unsupervised ASD; unlike the original work, it is implemented here with a CVAE instead of a fully connected architecture, consistently with the framework in Sec. II-A. UNS R uses the same approach with covariance regularization in the GMM, while FS denotes the proposed framework after few-shot adaptation. Comparing UNS and UNS R shows that covariance regularization improves the average AUPR from 20.0 to 23.0 and reduces the standard deviation over 10 training runs, not explicitly reported in the figure, from 8.0 to 6.2, indicating both higher mean performance and more stable training. The subsequent few-shot adaptation further increases the AUPR from 23.0 to 64.0, showing that the proposed adaptation step effectively exploits the limited fault information available after the first observed anomaly.

## D. Performance at Low False-Positive-Rate

Table I reports low-FPR performance as mean and standard deviation for each ASD model. The pAUROC is computed over the region with $\mathrm { F P R } \leq 1 \% ,$ while TPR@FPR 1% and TPR@FPR 0.1% denote the true positive rate obtained by fixing the threshold at FPR = 1% and = 0.1%, respectively.

TABLE I  
PERFORMANCE COMPARISON OF THE MODELS (MEAN ± STD)
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>pAUROC</td><td rowspan=1 colspan=1>TPR@FPR 1%</td><td rowspan=1 colspan=1>TPR@FPR 0.1%</td></tr><tr><td rowspan=1 colspan=1>UNS</td><td rowspan=1 colspan=1> $3 5 . 5 \pm 1 . 2$ </td><td rowspan=1 colspan=1> $5 8 . 0 \pm 1 . 4$ </td><td rowspan=1 colspan=1> $7 . 5 \pm 0 . 8$ </td></tr><tr><td rowspan=1 colspan=1>UNS_R</td><td rowspan=1 colspan=1> $3 9 . 6 \pm 0 . 9$ </td><td rowspan=1 colspan=1> $6 4 . 4 \pm 1 . 0 $ </td><td rowspan=1 colspan=1> $9 . 4 \pm 0 . 8$ </td></tr><tr><td rowspan=1 colspan=1>FS</td><td rowspan=1 colspan=1> ${ \bf 7 2 . 8 \pm 0 . 4 }$ </td><td rowspan=1 colspan=1> ${ \bf 8 9 . 3 \pm 0 . 0 }$ </td><td rowspan=1 colspan=1> ${ \bf 6 0 . 6 \pm 0 . 1 }$ </td></tr></table>

TABLE II  
PAUROC FOR EACH FOLD $( \mathbf { M E A N } \pm \mathbf { S T D } )$
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1> $F _ { 1 }$ </td><td rowspan=1 colspan=1> $F _ { 2 }$ </td><td rowspan=1 colspan=1> $F _ { 3 }$ </td></tr><tr><td rowspan=1 colspan=1>UNS</td><td rowspan=1 colspan=1> $4 2 . 4 \pm 1 2 . 3$ </td><td rowspan=1 colspan=1> $1 9 . 6 \pm 9 . 4$ </td><td rowspan=1 colspan=1> $4 4 . 4 \pm 1 4 . 0$ </td></tr><tr><td rowspan=1 colspan=1>UNS_R</td><td rowspan=1 colspan=1> $4 5 . 9 \pm 8 . 2$ </td><td rowspan=1 colspan=1> $2 3 . 3 \pm 8 . 5$ </td><td rowspan=1 colspan=1> $4 9 . 7 \pm 1 0 . 2$ </td></tr><tr><td rowspan=1 colspan=1>FS</td><td rowspan=1 colspan=1> ${ \bf 7 5 . 2 \pm 2 . 6 }$ </td><td rowspan=1 colspan=1> ${ \bf 6 4 . 5 \pm 0 . 5 }$ </td><td rowspan=1 colspan=1> ${ \bf 7 8 . 7 \pm 8 . 8 }$ </td></tr></table>

The FS model shows that the proposed adaptation step substantially improves low-FPR performance: the system becomes more reliable at $\mathrm { F P R } = 1 \%$ and can operate at $\mathrm { F P R } = 0 . 1 \%$ detecting more than 1 out of 2 anomalous samples. This makes the model more suitable for very low-FPR operation, especially when sample-level predictions are combined through binary integration [18]. Across all metrics, UNS R provides a small but consistent improvement in mean value and variability over the baseline UNS. However, its TPR is limited at FPR $= 1 \%$ and remains unreliable at $\mathrm { F P R } = 0 . 1 \%$ , detecting on average fewer than 1 out of 10 anomalous samples.

Table II reports the $\mathsf { p A U R O C }$ results by fold, with one column for each fold. The metrics under $F _ { 1 }$ are obtained by adapting the model using only anomalies from the first machine and testing the ASD performance on the second and third faulty machines. The same applies to $F _ { 2 }$ , where the second machine is used for training, and to $F _ { 3 } ,$ , where the third is used, always testing on the other two. The previously discussed results are confirmed and consistent, indicating that this approach can exploit limited anomalous samples from one machine to improve detection on the remaining machines in the considered facility. Nevertheless, this table shows how the train–test data split can affect absolute values, especially when the number of samples is very small, as in this case.

## IV. CONCLUSION

Covariance regularization improves the unsupervised GMM baseline in both average performance and stability. The proposed few-shot adaptation step leads to a substantial gain in detection capability, especially in the low FPR operating region, which is the most relevant for real industrial deployment, yielding higher pAUROC and TPR at very low FPR values, making the system much more effective in scenarios where false alarms must be strictly limited.

Overall, the proposed adaptation paradigm is promising for industrial ASD, as it allows the model to exploit newly observed faults and improve future detection. Future work will extend the evaluation to additional fault types, longer deployments, and hyperparameter sensitivity analysis.

## ACKNOWLEDGMENT

This work was supported in part by the Austrian Federal Ministry of Digital and Economic Affairs, in part by the National Foundation for Research, Technology and Development, Austria, in part by the Christian Doppler Research Association, and in part by TU Graz Open Access Publishing Fund, Austria.

## REFERENCES

[1] T. Yamamoto, K. Wada, S. Kanoko, and A. Higuchi, “Present status and future of equipment diagnosis technology for iron- and steelmaking plants,” Nippon Steel & Sumitomo Metal Technical Report, no. 112, 2016.

[2] H. M. Hashemian and W. C. Bean, “State-of-the-art predictive maintenance techniques,” IEEE Transactions on Instrumentation and Measurement, vol. 60, no. 10, pp. 3480–3492, 2011, corrected version.

[3] J. Yenus, G. Brooks, M. Dunn, and R. Kadam, “Application of vibration and sound signals in monitoring iron and steelmaking processes,” Ironmaking & Steelmaking, vol. 47, no. 2, pp. 178–187, 2020.

[4] G. Jombo and Y. Zhang, “Acoustic-based machine condition monitoring—methods and challenges,” Eng, vol. 4, no. 1, pp. 47–79, 2023.

[5] E. Di Fiore, A. Ferraro, A. Galli, V. Moscato, and G. Sperl\`ı, “An anomalous sound detection methodology for predictive maintenance,” Expert Systems with Applications, vol. 209, p. 118324, 2022.

[6] K. Suefusa, T. Nishida, H. Purohit, R. Tanabe, T. Endo, and Y. Kawaguchi, “Anomalous sound detection based on interpolation deep neural network,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2020, pp. 271–275.

[7] A. Jalali, L. Pham, C. Heistracher, D. Katic, and A. Schindler, “Dcase challenge 2022, task 2: Variational dense autoencoder for unsupervised anomalous sound detection of machinery,” Target, vol. 52, no. 52.46, pp. 52–78, 2022.

[8] H. Wang, J. Liu, and S. Li, “Abnormal sound detection based on composite autoencoder gaussian mixture model,” in International Conference on Electronic Information Engineering and Data Processing (EIEDP 2023), vol. 12700. SPIE, 2023, pp. 452–458.

[9] Y. Zeng, H. Liu, L. Xu, Y. Zhou, and L. Gan, “Robust anomaly sound detection framework for machine condition monitoring,” Proc. DCASE, 2022.

[10] M. D. Chinnasamy, M. Sumbwanyambe, and T. S. Hlalele, “Acoustic anomaly detection of machinery using autoencoder based deep learning,” in 2024 32nd Southern African Universities Power Engineering Conference (SAUPEC). IEEE, 2024, pp. 1–6.

[11] T. Nishida, N. Harada, D. Niizumi, D. Albertini, R. Sannino, S. Pradolini, F. Augusti, K. Imoto, K. Dohi, H. Purohit et al., “Description and discussion on dcase 2025 challenge task 2: First-shot unsupervised anomalous sound detection for machine condition monitoring,” arXiv preprint arXiv:2506.10097, 2025.

[12] K. Wilkinghoff, T. Fujimura, K. Imoto, J. L. Roux, Z.-H. Tan, and T. Toda, “Handling domain shifts for anomalous sound detection: A review of dcase-related work,” arXiv preprint arXiv:2503.10435, 2025.

[13] K. Wilkinghoff, H. Yang, J. Ebbers, F. G. Germain, G. Wichern, and J. Le Roux, “Local density-based anomaly score normalization for domain generalization,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 4642–4652, 2025.

[14] P. Saengthong and T. Shinozaki, “Deep generic representations for domain-generalized anomalous sound detection,” in ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[15] J. Yan, Y. Cheng, Q. Wang, L. Liu, W. Zhang, and B. Jin, “Transformer and graph convolution-based unsupervised detection of machine anomalous sound under domain shifts,” IEEE Transactions on Emerging Topics in Computational Intelligence, vol. 8, no. 4, pp. 2827–2842, 2024.

[16] C. Pichler, M. Neumayer, C. Feilmayr, S. Schuster, and H. Wegleiter, “Unsupervised anomalous sound detection in industrial machine facilities,” in 2025 IEEE SENSORS, 2025, pp. 1–4.

[17] C. Pichler, M. Neumayer, B. Schweighofer, C. Feilmayr, S. Schuster, and H. Wegleiter, “Acoustic-based detection technique for identifying worn-out components in large-scale industrial machinery,” IEEE Sensors Letters, vol. 7, no. 9, pp. 1–4, 2023.

[18] M. A. Richards et al., Fundamentals of radar signal processing. Mcgraw-hill New York, 2005, vol. 1.