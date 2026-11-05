# ADAPTING SLEEP FOUNDATION MODELS FOR DENSE SLEEP SPINDLE LOCALIZATION

Alexis Bollengier<sup>†⋆</sup>, Abel D´ıaz Berenguer<sup>†</sup>, Olivier Pallanca<sup>⋆</sup>, Andre Anjos´ <sup>⋆</sup> & Hichem Sahli<sup>†‡</sup>

<sup>⋆</sup> Idiap Research Institute, Martigny, Switzerland

<sup>†</sup> Department of Electronics and Informatics (ETRO), Vrije Universiteit Brussel (VUB), Brussels, Belgium <sup>‡</sup> Interuniversity Microelectronics Centre (IMEC), Heverlee, Belgium

## ABSTRACT

Adapting sleep foundation models to dense electroencephalography (EEG) event localization is challenging because multisecond representations may not preserve sufficient temporal detail for precise boundary recovery. We address sleep spindle localization by adapting a frozen SleepFM encoder with a local time-frequency pathway that preserves spindle-relevant spectral content and intra-token position. The local representation drives localization, while residual cross-attention incorporates SleepFM temporal context. Dense predictions are obtained through localized token-to-sample reconstruction and temporal refinement. Tested on the Massive Online Data Annotation (MODA), the proposed model achieves an event-level F1 of 0.8080±0.0040 at IoU 0.2 across six folds. Ablations show that localization primarily depends on local spectral and sub-token temporal information, while the pretrained representation provides complementary context. These results indicate that dense adaptation benefits from preserving information below the pretrained token scale. Code will be released upon acceptance.

Index Terms— sleep spindle detection, EEG, sleep foundation models, time-frequency representation, dense temporal localization

## 1. INTRODUCTION

Sleep spindles are transient EEG events in non-rapid eye movement sleep, typically lasting 0.5–3 s with sigma-band activity around 11–16 Hz [1]. Their characteristics are linked to memory consolidation and investigated as biomarkers for neuropsychiatric disorders [2], motivating accurate detection and boundary delineation.

Dense detectors such as SUMO [3] and BLAST [4] learn task-specific spindle representations, whereas sleep foundation models learn reusable representations from large polysomnography datasets. SleepFM [5] supports multiple downstream tasks but compresses multi-second EEG windows into compact tokens, creating a resolution mismatch for dense localization. SleepGPT [6] supports spindle detection by jointly learning temporal and frequency-domain representations during pretraining. Here, rather than pretraining a new time-frequency backbone, we adapt a frozen representation with task-specific local information to recover boundaries below the pretrained token scale.

The proposed architecture keeps a pretrained SleepFM encoder frozen and adds a local time-frequency branch that preserves spindle-related spectral information and intra-token position. Residual cross-attention incorporates the pretrained SleepFM representation as complementary context, followed by sample-level reconstruction of sub-token temporal structure.

The main contributions are: (1) dense sample-level adaptation of a frozen SleepFM encoder; (2) a physiologically structured time-frequency representation preserving local spectral content and intra-token position, with asymmetric cross-attention to the pretrained SleepFM representation; and (3) localized token-to-sample reconstruction with reinjection of retained intra-token features. The approach is evaluated on the MODA benchmark using MASS EEG segments and expert-consensus spindle annotations across six published folds, with controlled architectural ablations.

## 2. METHODS

## 2.1. Overview

The proposed architecture addresses the temporal-resolution mismatch between pretrained sleep representations and dense spindle localization. When local structure is compressed into a multi-second token, increasing the output resolution alone cannot recover the precise location of spindle-related activity. This adaptation therefore preserves local information in a parallel branch and reuses it during sample-level reconstruction.

Let $\mathbf { X } \in \mathbb { R } ^ { B \times C \times T _ { 0 } }$ represent a multichannel input segment. We use a SleepFM encoder pretrained under the original multimodal contrastive formulation [5]. Its channelagnostic design accommodates heterogeneous recording configurations and modality-specific representations, motivating its use with a reduced configuration of brain activity signals (BAS) for downstream spindle localization. The pretrained weights are kept frozen, while the input window duration is adapted from 5 to 3 seconds. For the downstream spindle-localization task, the encoder processes the available EEG/EOG channels to produce temporal tokens $\mathbf { Z } _ { \mathrm { t m p } } \in \mathbb { R } ^ { B \times N \times E }$ , where $E = 1 2 8$ . A task-specific timefrequency branch operates on the same grid, generating frequency tokens $\mathbf { Z } _ { \mathrm { f r e q } }$ and ordered intra-token features Φ. Figure 1 summarizes their integration before sample-level prediction.

![](images/711a04c92b9970901cee9e0dbd0add14347214a951df6a6bcf0d220ca21b6671.jpg)  
Fig. 1. Overview of the proposed spindle localization architecture. The frozen SleepFM encoder provides temporal context, while a local time-frequency branch preserves spectral and intra-token information for dense reconstruction.

![](images/4137df80c710387b4484ed4e6cb13f28fbd356a7caba9d98a3437c1046983176.jpg)  
Fig. 2. Band-structured readout. Mean pooling across physiological bands and ordered temporal segments produces a token descriptor and intra-token features.

## 2.2. Local Time-Frequency Representation

The local time-frequency branch captures sigma-band oscillations and the transient morphology of sleep spindles. A fixed Morlet wavelet transform represents the EEG jointly in time and frequency [7]. The resulting scalogram is segmented using the same overlapping windows as SleepFM. Each patch is processed by a frequency-wise Squeeze-and-Excitation module adapted from [8], emphasizing informative frequency regions before three 3 × 3 convolutional layers with $M = 6 4$ output features extract local patterns.

A key contribution is the band-structured readout in Fig. 2. Global pooling retains the presence of relevant activity but discards its position within the 3-second token. Instead, the representation is organized into five physiological frequency bands and $T _ { s } = 1 2$ ordered temporal segments. Mean pooling over the scalogram-frame indices $\mathcal { T } _ { s }$ of temporal segment s and the frequency indices $\mathcal { F } _ { b }$ of band b preserves spectral organization and intra-token position. The resulting representation forms $\mathbf { Z } _ { \mathrm { f r e q } }$ for token-level fusion, while the same features are retained as Φ for dense reconstruction, with $Q = 5 M = 3 2 0$ features per temporal segment.

## 2.3. Context Fusion

The frequency representation preserves localized spindle evidence, whereas SleepFM provides broader pretrained temporal context. Their interaction is therefore asymmetric. Frequency tokens query the pretrained sequence using residual multi-head cross-attention [9]

$$
\mathbf { Z } _ { \mathrm { a t t n } } = \mathrm { L N } ( \mathbf { Z } _ { \mathrm { f r e q } } + \mathrm { C r o s s A t t n } \left( \mathbf { Z } _ { \mathrm { f r e q } } , \mathbf { Z } _ { \mathrm { t m p } } , \mathbf { Z } _ { \mathrm { t m p } } \right) )
$$

A residual feed-forward sublayer then produces $\mathbf { Z } _ { \mathrm { f u s e d } }$ . This keeps $\mathbf { Z } _ { \mathrm { f r e q } }$ as the main localization pathway while retrieving complementary context from SleepFM.

## 2.4. Localized Token-to-Sample Reconstruction

Fusion still operates at 3-second token resolution. The decoder must therefore reconstruct temporal structure within each token rather than simply increase the sequence length. Localized wavelet synthesis lets each fused token predict both the temporal patterns contributing to reconstruction and their position within the window.

Let $\mathbf { z } _ { n }$ denote fused token n and $W = 3 8 4$ its window length. A fixed bank $\Psi \in \mathbb { R } ^ { K _ { a } \times P _ { a } \times W }$ contains $K _ { a } = 1 6$ temporal atom types at $P _ { a } ~ = ~ 6$ candidate positions. The atoms span one to four cycles with Gaussian widths logarithmically spaced between W/16 and $W / 2 . 5 .$ , with candidate centers from 0.20W to 0.80W. They describe event-scale temporal variations rather than sigma oscillations, which are already represented by the frequency branch.

Each token predicts feature coefficients $\mathbf { C } _ { n } ~ \in ~ \mathbb { R } ^ { K _ { a } }$ ×d with $d = 6 4$ and position probabilities $\pi _ { n , k , p } .$ . The localized form of atom k is

$$
A _ { n , k , w } = \sum _ { p = 1 } ^ { P _ { a } } \pi _ { n , k , p } \Psi _ { k , p , w }
$$

The sample-wise feature reconstruction is

$$
\mathbf { X } _ { n , w } = \mathrm { L N } \left( \mathbf { W } _ { b } \mathbf { z } _ { n } + \mathbf { b } _ { b } \right) + \sum _ { k = 1 } ^ { K _ { a } } A _ { n , k , w } \mathbf { C } _ { n , k }
$$

The atom bank remains fixed, while its coefficients and placement probabilities are predicted from each token. This constrains reconstruction to localized temporal patterns rather than an unconstrained sample-level sequence.

Because consecutive SleepFM windows overlap, their reconstructions are combined using Hann-weighted overlapadd

$$
\mathbf { Z } _ { \mathrm { s a m p l e } } [ t ] = \frac { \sum _ { ( n , w ) \in \mathbb { Z } _ { t } } g _ { w } \mathbf { X } _ { n , w } } { \sum _ { ( n , w ) \in \mathbb { Z } _ { t } } g _ { w } }
$$

where ${ \mathcal { T } } _ { t } = \{ ( n , w ) \mid a _ { n } + w = t \} , a _ { n }$ is the start of token n and $g _ { w }$ is the Hann weight.

For the reconstruction ablation in Table 1, localized wavelet synthesis was replaced by linear interpolation of the projected fused token representations between consecutive token centers onto the sample grid, while all other components were kept identical to the frame-off reference.

## 2.5. Intra-Token Reinjection and Temporal Refinement

Wavelet synthesis reconstructs temporal structure from $\mathbf { Z } _ { \mathrm { f u s e d } } .$ while Φ retains explicit intra-token spectral position before compression. These features are projected to the decoder width and expanded from $T _ { s }$ temporal segments to sample positions using nearest-neighbor upsampling. Overlapping contributions are averaged and concatenated with $\mathbf { Z } _ { \mathrm { s a m p l e } } ,$ preserving information that would otherwise need to be recovered from the fused token.

A Temporal Convolutional Network [10] with five residual blocks, kernel size 7 and dilations 1, 2, 3, 4, 6, adds neighboring temporal context without reducing sample resolution. Three prediction heads estimate spindle probability, onset and offset distances and the background, boundary or spindleinterior class at each sample.

## 2.6. Training Objective

Training jointly supervises spindle presence, event boundaries and temporal structure. The segmentation objective combines focal loss [11] and Dice loss [12]

$$
{ \mathcal { L } } _ { \mathrm { s e g } } = { \mathcal { L } } _ { \mathrm { f o c a l } } + { \mathcal { L } } _ { \mathrm { d i c e } }
$$

Boundary regression is restricted to annotated spindle samples and optimized using Smooth L1 loss [13] on onset and offset distances normalized by the sampling frequency. The temporal-classification head applies weighted cross-entropy over background, boundary and spindle-interior samples.

An auxiliary objective regularizes the task-specific frequency representation toward the pretrained SleepFM representation without modifying the frozen backbone. Corresponding tokens are represented as diagonal Gaussian distributions in a shared latent space [14], with the temporal representation serving as the fixed reference

$$
\mathcal { L } _ { \mathrm { c o n s } } = \frac { 1 } { B N } \sum _ { i = 1 } ^ { B } \sum _ { n = 1 } ^ { N } D _ { \mathrm { K L } } \left( q _ { \mathrm { t m p } } ^ { ( i , n ) } \Vert q _ { \mathrm { f r e q } } ^ { ( i , n ) } \right)
$$

A complementary frequency constraint encourages spindlerelated spectral structure. For each event $e ,$ sigma prominence compares energy in [11, 16] Hz with the surrounding context [5, 11) ∪ (16, 25] Hz

$$
\rho _ { e } = \frac { E _ { \sigma , e } } { E _ { \sigma , e } + E _ { \mathrm { c t x } , e } }
$$

The frequency loss averages the prominence deficit over the set $\mathcal { E }$ of predicted event candidates anchored at positivescoring time-frequency frames with valid sample support,

$$
\mathcal { L } _ { \mathrm { f r e q } } = \frac { 1 } { | \mathcal { E } | } \sum _ { e \in \mathcal { E } } \operatorname* { m a x } ( 0 , 0 . 7 0 - \rho _ { e } )
$$

The complete objective is

$$
\mathcal { L } = \lambda _ { \mathrm { s e g } } \mathcal { L } _ { \mathrm { s e g } } + \lambda _ { \mathrm { b n d } } \mathcal { L } _ { \mathrm { b n d } } + \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { a l i g n } } \left( \mathcal { L } _ { \mathrm { c o n s } } + \mathcal { L } _ { \mathrm { f r e q } } \right)
$$

## 3. EXPERIMENTAL SETUP

Experiments were conducted using EEG segments from MASS [15] and the corresponding MODA expert-consensus spindle annotations [16]. The analysis included 749 N2 EEG excerpts from 180 subjects, containing 5,342 consensus spindle events. EEG signals were resampled to 128 Hz and analysed as 115-s excerpts.

Fig. 1 depicts the nominal EEG configuration. Inputs were C3, F3 and O1; when F3 was unavailable in cohort 01-04, upper vertical EOG served as the third downstream channel. Following SUMO [3], 36 subjects formed a fixed test set and the remaining 144 were split into 120/24 training/validation subjects across six folds.

The pretrained SleepFM encoder remained frozen and was employed with the reduced BAS channel configuration described above. Task-specific modules were trained for 40 epochs with AdamW (LR $2 . 6 6 \times 1 0 ^ { - 4 }$ , weight decay $2 . 1 2 \times 1 0 ^ { - 3 }$ , batch 16, gradient clip 1.36, token stride 192), focal $( \alpha , \gamma ) = ( 0 . 7 0 7 , 1 . 3 9 7 )$ , Smooth L1 $\beta = 0 . 0 8 5$ , and $( \lambda _ { \mathrm { s e g } } , \lambda _ { \mathrm { b n d } } , \lambda _ { \mathrm { c l s } } , \lambda _ { \mathrm { a l i g n } } ) = ( 1 . 1 5 1 , 0 . 9 6 4 , 0 . 8 5 5 , 0 . 5 3 9 )$ . LR was reduced by 0.518 after 5 epochs without improvement.

Table 1. Controlled architectural ablations on MODA $( \mathrm { m e a n } \pm \mathrm { S D }$ across six training/validation folds on the fixed test set). The frame-off reference uses frequency-query fusion, both representations, structured readout and localized wavelet reconstruction; each other configuration changes one component.
<table><tr><td rowspan="2">Metric</td><td colspan="2">Frame reinjection</td><td>Fusion</td><td colspan="2">Representation</td><td>Readout</td><td>Reconstruction</td></tr><tr><td>On</td><td> $\mathrm { O f f } \left( \mathrm { r e f . } \right)$ </td><td>Temp. query</td><td>Freq. only</td><td> ${ \mathrm { T e m p . ~ o n l y } }$ </td><td>Band pooling</td><td>Linear interp.</td></tr><tr><td>F1@0.2</td><td> $\mathbf { 0 . 8 0 8 0 { \overset { . } { = } } 0 . 0 0 4 0 }$ </td><td> $0 . 8 0 0 6 { \scriptstyle \pm 0 . 0 0 5 1 }$ </td><td> $0 . 5 1 3 7 { \scriptstyle \pm 0 . 0 2 0 4 }$ </td><td> $0 . 7 8 9 8 { \pm } 0 . 0 0 3 0$ </td><td> $0 . 4 0 2 4 { \scriptstyle \pm 0 . 0 1 2 5 }$ </td><td> $0 . 6 7 7 9 { \scriptstyle \pm 0 . 0 1 1 9 }$ </td><td> $0 . 7 8 5 6 { \pm } 0 . 0 0 8 9$ </td></tr><tr><td>F1@0.3</td><td> $\mathbf { 0 . 8 0 7 5 { \scriptstyle \pm 0 . 0 0 4 2 } }$ </td><td> $0 . 7 9 8 9 { \pm } 0 . 0 0 5 4$ </td><td> $0 . 4 9 2 1 { \scriptstyle \pm 0 . 0 2 3 0 }$ </td><td> $0 . 7 8 9 2 { \scriptstyle \pm 0 . 0 0 4 4 }$ </td><td> $0 . 3 7 0 7 { \scriptstyle \pm 0 . 0 1 4 8 }$ </td><td> $0 . 6 4 9 0 { \scriptstyle \pm 0 . 0 1 3 2 }$ </td><td> $0 . 7 8 2 8 { \scriptstyle \pm 0 . 0 0 9 7 }$ </td></tr><tr><td>F1@0.5</td><td> $\mathbf { 0 . 7 7 6 6 { \overset { . } { \bot } } 0 . 0 0 5 0 }$ </td><td> $0 . 7 6 6 2 { \scriptstyle \pm 0 . 0 0 5 8 }$ </td><td> $0 . 3 9 0 9 { \scriptstyle \pm 0 . 0 2 9 8 }$ </td><td> $0 . 7 4 4 2 { \scriptstyle \pm 0 . 0 2 1 3 }$ </td><td> $0 . 2 6 0 6 { \scriptstyle \pm 0 . 0 1 1 2 }$ </td><td> $0 . 5 4 7 9 { \scriptstyle \pm 0 . 0 1 4 2 }$ </td><td> $0 . 7 3 9 8 { \scriptstyle \pm 0 . 0 1 7 0 }$ </td></tr></table>

Forty Optuna [17] trials (seed 42) on the first validation fold optimized $\mathrm { F 1 @ 0 . 2 + 0 . 5 F 1 @ 0 . 5 ; }$ the selected hyperparameters were fixed across folds. At each epoch, each fold’s validation set selected the probability threshold (0.40–0.70, step 0.01), merge gap (0–0.20 s, step 0.05 s), and checkpoint maximizing F1@0.2.

At inference, thresholding defined candidate events and probability-weighted medians aggregated boundaries. Oneto-one temporal-IoU matching reported F1 at 0.2, 0.3 and 0.5. Validation-selected settings were applied unchanged to the fixed test set; SUMO [3] and BLAST [4] were retrained and independently optimized under the same protocol.

## 4. RESULTS AND DISCUSSION

## 4.1. Detection and Localization

Figure 3 compares the proposed model, SUMO, and BLAST across IoU thresholds. At $\mathrm { I o U } = 0 . 2 ,$ the proposed model achieves an event-level F1 of $0 . 8 0 8 0 { \scriptstyle \pm 0 . 0 0 4 0 }$ , remaining comparable to SUMO and BLAST. $\mathrm { A t \ I o U = 0 . 5 , }$ it reaches $0 . 7 7 6 6 { \scriptstyle \pm 0 . 0 0 5 0 }$ , compared with 0.7682±0.0103 for SUMO and 0.7554±0.0142 for BLAST. At stricter overlap thresholds, however, the proposed model declines more rapidly, with SUMO and subsequently BLAST surpassing it. These results indicate strong spindle-extent recovery at moderate overlap thresholds, while very precise boundary alignment remains challenging.

## 4.2. Architectural Contributions

Table 1 isolates component contributions. With frame reinjection disabled, the fused frequency-query reference achieves F1@0.2 of 0.8006, versus 0.7898 for the frequency-only and 0.4024 for the temporal-only SleepFM. Reversing the crossattention direction reduces F1 to 0.5137. These results show that the local time-frequency representation provides the primary localization evidence, while the pretrained SleepFM representation contributes as complementary temporal context. Band pooling reduces F1 from 0.8006 to 0.6779 at IoU 0.2, and from 0.7662 to 0.5479 at IoU 0.5. Localized wavelet reconstruction outperforms linear interpolation, with the improvement increasing from 0.0150 at IoU 0.2 to 0.0264 at IoU

![](images/f8478ea556f6d979b9211873b1a087127153ee355ade7f7bbcc4a4975424c5a5.jpg)  
Fig. 3. Mean event-level F1 across IoU thresholds over six training/validation folds on the fixed MODA test set.

0.5. Frame reinjection yields an additional gain, increasing F1 from 0.8006 to 0.8080 at IoU 0.2 and from 0.7662 to 0.7766 at IoU 0.5. These findings suggest that preserving intra-token information and explicitly reconstructing sub-token temporal structure improves boundary-sensitive spindle localization.

## 5. CONCLUSION

This work adapts a pretrained SleepFM model to dense spindle localization by preserving sub-token information. On MODA, F1 reaches 0.8080±0.0040 at IoU 0.2 and 0.7766±0.0050 at IoU 0.5, although stricter thresholds expose residual boundary errors. Ablations identify local spectral and intra-token information as the main localization evidence, with SleepFM providing complementary context; structured readout, localized reconstruction and reinjection further improve recovery. Overall, dense adaptation benefits from preserving and reconstructing sub-token information.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

MASS [15] data were analysed in accordance with the archive’s data-use agreement. The study was conducted under protocol CER-2025-Navarro-PREDSomADICE-00578,

which received a favourable opinion from the Research Ethics Committee of Sorbonne Universite.´

## 7. ACKNOWLEDGMENTS

This work was initiated and supported by the DREAM project within the Medical Artificial Intelligence (MedAI) group at Idiap Research Institute funded by Fondation AP-HP. Authors declare no conflicts of interest.

## 8. REFERENCES

[1] Laura M. J. Fernandez and Anita Luthi, “Sleep spin-¨ dles: Mechanisms and functions,” Physiological Reviews, vol. 100, no. 2, pp. 805–868, 2020.

[2] Dara S. Manoach and Robert Stickgold, “Abnormal sleep spindles, memory consolidation, and schizophrenia,” Annual Review ofClinical Psychology, vol. 15, pp. 451–479, 2019.

[3] Lars Kaulen, Justus T. C. Schwabedal, Jules Schneider, Philipp Ritter, and Stephan Bialonski, “Advanced sleep spindle identification with neural networks,” Scientific Reports, vol. 12, no. 1, pp. 7686, 2022.

[4] Chenyun Guo, Yifeng Ji, Dan Chen, Xiaoli Li, Tengfei Gao, Wei Yu, Zhifei Xu, and Mingqi Dong, “BLAST: Towards robust detection of sleep spindles across clinical settings,” Neurocomputing, vol. 653, pp. 131107, 2025.

[5] Rahul Thapa, Magnus Ruud Kjær, Bryan He, Ian Covert, Hyatt Moore, Umaer Hanif, Gauri Ganjoo, M. Brandon Westover, Poul Jennum, Andreas Brink-Kjær, Emmanuel Mignot, and James Zou, “A multimodal sleep foundation model for disease prediction,” Nature Medicine, vol. 32, no. 2, pp. 752–762, 2026.

[6] Weixuan Huang, Yan Wang, Hanrong Cheng, Wei Xu, Tingyue Li, Xiuwen Wu, Hui Xu, Pan Liao, Zaixu Cui, Qihong Zou, and Jia-Hong Gao, “A unified timefrequency foundation model for sleep decoding,” Nature Communications, vol. 17, pp. 1198, 2026.

[7] Michael X. Cohen, “A better way to define and describe Morlet wavelets for time-frequency analysis,” NeuroImage, vol. 199, pp. 81–86, 2019.

[8] Jie Hu, Li Shen, and Gang Sun, “Squeeze-andexcitation networks,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7132–7141.

[9] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin, “Attention is all you need,”

in Advances in Neural Information Processing Systems, 2017, vol. 30.

[10] Shaojie Bai, J. Zico Kolter, and Vladlen Koltun, “An empirical evaluation of generic convolutional and recurrent networks for sequence modeling,” arXiv preprint arXiv:1803.01271, 2018.

[11] Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollar, “Focal loss for dense object detection,”´ in Proceedings ofthe IEEE International Conference on Computer Vision (ICCV), 2017, pp. 2980–2988.

[12] Fausto Milletari, Nassir Navab, and Seyed-Ahmad Ahmadi, “V-Net: Fully convolutional neural networks for volumetric medical image segmentation,” in 2016 Fourth International Conference on 3D Vision (3DV), 2016, pp. 565–571.

[13] Ross Girshick, “Fast R-CNN,” in Proceedings of the IEEE International Conference on Computer Vision (ICCV), 2015, pp. 1440–1448.

[14] Diederik P. Kingma and Max Welling, “Auto-encoding variational Bayes,” in International Conference on Learning Representations (ICLR), 2014.

[15] Christian O’Reilly, Nadia Gosselin, Julie Carrier, and Tore Nielsen, “Montreal Archive of Sleep Studies: An open-access resource for instrument benchmarking and exploratory research,” Journal of Sleep Research, vol. 23, no. 6, pp. 628–635, 2014.

[16] Karine Lacourse, Ben Yetton, Sara Mednick, and Simon C. Warby, “Massive online data annotation, crowdsourcing to generate high quality sleep spindle annotations from EEG data,” Scientific Data, vol. 7, pp. 190, 2020.

[17] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama, “Optuna: A nextgeneration hyperparameter optimization framework,” in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. 2019, pp. 2623–2631, Association for Computing Machinery.