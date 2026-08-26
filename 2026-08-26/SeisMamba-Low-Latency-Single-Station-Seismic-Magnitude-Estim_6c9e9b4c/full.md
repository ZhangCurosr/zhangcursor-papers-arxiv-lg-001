# SeisMamba: Low-Latency Single-Station Seismic Magnitude Estimation for Spatially Distributed Earthquake Early Warning

Quenton Yeo qyeo2651@uni.sydney.edu.au The University of Sydney Sydney, Australia

Luke Stephen Higgins   
luke.higgins@accenture.com   
Accenture   
Sydney, Australia   
Zhaoge Bi   
zhaoge.bi@sydney.edu.au   
The University of Sydney   
Sydney, Australia

Flora Salim flora.salim@unsw.edu.au The University of New South Wales Sydney, Australia

Linghan Huang linghan.huang@sydney.edu.au The University of Sydney Sydney, Australia

Huaming Chen huaming.chen@sydney.edu.au The University of Sydney Sydney, Australia

## Abstract

Rapid earthquake magnitude estimation is central to earthquake early warning, yet many operational systems depend on dense re gional seismic networks and region-specific calibration. This creates a spatial coverage barrier for high-risk areas with sparse sensing infrastructure. Single-station learning ofers a lower-cost alternative, but existing models often face an accuracy–latency trade-of and may degrade under regional distribution shift. We present Seis-Mamba, a lightweight Mamba-based architecture for low-latency magnitude estimation from minimally processed three-component seismic waveforms recorded at a single station. SeisMamba com bines hierarchical convolutional encoding, sparse selective statespace modelling, multi-scale feature fusion, and an auxiliary temporal prediction head to support eficient long-sequence waveform analysis. On the STEAD benchmark, SeisMamba achieves the best MSE, RMSE, and �<sup>2</sup> among tested baselines while requiring only 0.55 ms for a batch of 32 waveforms on an NVIDIA T4 GPU, making it about three times faster than transformer-based baselines. We further conduct a Chile–Taiwan regional hold-out experiment as a diagnostic test of cross-region deployment, where SeisMamba retains useful performance on geographically unseen seismic regions. These results suggest that selective state-space waveform modelling provides a promising accuracy–latency backbone for spatially distributed, low-cost earthquake early warning.

## CCS Concepts

• Information systems → Geographic information systems; • Computing methodologies → Supervised learning.

## Keywords

Earthquake, Magnitude, Seismic, Mamba, AI, Spatial

ACM Reference Format: Quenton Yeo, Zhaoge Bi, Linghan Huang, Luke Stephen Higgins, Flora Salim, and Huaming Chen. 2026. SeisMamba: Low-Latency Single-Station Seismic Magnitude Estimation for Spatially Distributed Earthquake Early Warning. In Proceedings ofThe ACM International Conference on Advances in Geographic Information Systems (SIGSPATIAL ’26). ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 Introduction

Earthquake early warning (EEW) aims to estimate event severity quickly enough for alerts to remain useful before damaging shaking reaches exposed locations [1, 8]. In practice, many EEW systems depend on dense regional seismic networks, historical catalogues, and region-specific calibration, creating a spatial coverage barrier for high-risk regions with sparse monitoring infrastructure. This motivates low-cost single-station learning, where magnitude is estimated from a three-component waveform recorded at one station and the model can run close to the sensing device.

However, single-station EEW is not only a waveform regression problem. It also requires spatially distributed sensing, low-latency local inference, and robustness to regional distribution shift. A model trained on global seismic archives must remain useful when deployed in regions whose waveform characteristics difer from those seen during training. Large datasets such as STEAD provide the scale needed for learning transferable seismic representations [6], but spatial variation in attenuation, source properties, site effects, and station conditions still makes cross-region transfer challenging. Existing magnitude-estimation methods address only part of this setting. Traditional and shallow learning approaches often rely on hand-crafted P-wave features or auxiliary source information [11]. Deep models such as MagNet estimate magnitude directly from raw three-component waveforms [4], while PhaseNet and EQTransformer provide strong seismic representation backbones originally designed for phase picking or detection [5, 12]. More recent attention-based magnitude models improve temporal context modelling [9], but recurrent and attention-heavy architectures can increase inference latency, weakening their suitability for edge deployment. Mamba selective state-space models provide a promising alternative by replacing quadratic attention with input-dependent state transitions, enabling linear-time sequence modelling with long-context capacity [2]. Recent seismic work has shown that Mamba-style modelling can support phase picking [10]. However, its potential for low-latency single-station magnitude estimation under regional deployment shift remains underexplored.

We propose SeisMamba, a compact Mamba-based architecture for low-cost single-station earthquake magnitude estimation. Rather than treating magnitude estimation as a standalone waveform regression task, SeisMamba is designed around the requirements of spatially distributed EEW: eficient local inference, strong accuracy– latency balance, and evaluation under regional distribution shift. The model combines convolutional waveform encoding with sparsely placed selective state-space blocks and multi-scale aggregation, while an auxiliary temporal head provides additional supervision on when magnitude evidence becomes stable. Our evaluation studies whether this design can improve the accuracy–latency trade-of on a global seismic benchmark and whether the learned representation remains useful under a geographically held-out regional setting. The results support SeisMamba as an eficient backbone for low-cost, spatially distributed earthquake warning.

## 2 Related Work

Single-station seismic magnitude estimation. Single-station EEW reduces reliance on dense regional networks by estimating earthquake properties from local waveform observations. Early learning-based methods include engineered P-wave features, transfer learning, or auxiliary source information to improve rapid magnitude estimation [11]. MagNet further showed that neural models can estimate magnitude directly from raw waveform data [4]. Related seismic deep models such as PhaseNet and EQTransformer provide strong waveform representation backbones, although they were originally designed for phase picking or detection rather than magnitude regression [5, 12]. More recent attention-based models improve temporal context modelling for magnitude estimation [9]. These studies demonstrate the promise of learning-based EEW, but practical single-station deployment still requires low inference latency and robustness under regional distribution shift.

Eficient sequence modelling for spatially distributed seismic sensing. Transformers are efective for long-range sequence modelling, but attention has quadratic complexity with sequence length [7]. Mamba provides a selective state-space alternative with linear-time scaling and input-dependent state transitions [2]. In seismology, PhaseMamba suggests that Mamba-style models can capture useful waveform structure for phase picking [10]. Seis-Mamba difers from these works by targeting magnitude estimation from single-station waveforms under low-latency and regional de ployment constraints, with evaluation on both a global benchmark and geographically held-out seismic regions.

## 3 Method

## 3.1 Problem Setup and Design Requirements

We study single-station earthquake magnitude estimation from minimally processed three-component seismic waveforms. Given a 30-second ENZ waveform ${ \boldsymbol { x } } \in { \overset {  } { \mathbb { R } } } ^ { 3 \times T }$ sampled at 100 Hz, the model predicts a scalar magnitude �ˆ . Unlike dense-network EEW settings, the target deployment setting assumes that inference may be performed close to a single sensing device. This creates three design requirements: accurate magnitude regression, low inference latency, and robustness when the model is applied to geographically diferent seismic regions.

We therefore treat the task as both a waveform modelling problem and a regional deployment problem. The standard benchmark measures average predictive performance on STEAD, while the regional hold-out evaluation tests whether the learned representation remains useful when events from selected geographic regions are excluded during training and used only for testing.

## 3.2 SeisMamba Architecture

SeisMamba is designed to balance local waveform feature extraction with eficient long-context sequence modelling. As shown in Fig. 1, it consists of a hierarchical one-dimensional convolutional encoder, sparsely placed Mamba blocks, multi-scale feature fusion, and two prediction heads. The right part of the architecture is a lightweight prediction-side decoder: it maps fused features to a scalar magnitude estimate and an auxiliary time-resolved magnitude trajectory, rather than reconstructing the input waveform.

The encoder first maps the three input channels into a latent feature space. Four residual 1D convolutional stages then progressively reduce temporal resolution while increasing channel width. These stages capture local seismic patterns such as onset sharpness, polarity variation, and amplitude growth. This convolutional front-end is kept lightweight because the target setting requires low-latency local inference.

Mamba blocks are inserted only at selected encoder resolutions. This sparse placement is motivated by eficiency: local convolutions first compress short-range waveform structure, after which selective state-space modelling is applied at shorter sequence lengths. The bypass paths in Fig. 1 indicate that not every resolution is processed by a Mamba block; instead, local convolutional features are passed forward directly when long-context modelling is unnecessary. This avoids dense state-space processing at every stage while retaining the ability to model longer waveform context, and it also avoids the quadratic sequence cost of transformer-style attention.

For magnitude prediction, SeisMamba aggregates information across encoder depths. Let $h ^ { ( l ) }$ denote the feature map from encoder stage �. A pooled descriptor is extracted from each stage and concatenated:

$$
z = \mathrm { C o n c a t } \big ( \mathrm { P o o l } ( h ^ { ( 1 ) } ) , \dots , \mathrm { P o o l } ( h ^ { ( 4 ) } ) \big ) .\tag{1}
$$

The scalar head maps � to the final magnitude estimate �ˆ . This multi-scale representation combines shallow onset-sensitive cues with deeper contextual waveform evidence.

An auxiliary temporal head is attached to the deepest feature map, with a high-resolution bypass used to preserve temporal detail for time-resolved prediction. It predicts a magnitude trajectory whose targets are set to zero before the P-wave arrival and to the event magnitude afterwards. This branch provides dense supervision and exposes when magnitude evidence begins to stabilize, but the scalar head remains the primary output used for benchmark comparison. The training objective is

$$
\mathcal { L } = 0 . 7 5 \mathcal { L } _ { \mathrm { s c a l a r } } + 0 . 2 5 \mathcal { L } _ { \mathrm { t e m p o r a l } } ,\tag{2}
$$

where both terms use mean squared error.

![](images/53b6a2033a81381cbedb7e9a0089408c98fd0f7ff4ef15bac184959b095f7bb8.jpg)  
Figure 1: SeisMamba architecture overview. Local convolutions extract short-range waveform morphology, sparse Mamba blocks model longer temporal context, and the output heads produce scalar and time-resolved magnitude estimates.

## 4 Experiments

## 4.1 Dataset, Baselines, and Metrics

We evaluate SeisMamba on STEAD, a global dataset ofsingle-station three-component seismic waveforms [6]. Each input is a 30-second ENZ waveform sampled at 100 Hz. To preserve a realistic low-cost deployment setting, we apply only a 1–40 Hz fourth-order zerophase Butterworth band-pass filter before model input. No handcrafted P-wave features, source-location inputs, or station-specific calibration are used. All models follow the same preprocessing, random-crop augmentation, labelling procedure, and evaluation protocol. They are trained with AdamW, batch size 32, a base learning rate of $1 0 ^ { - 3 }$ , a maximum of 150 epochs, early stopping, five warm-up epochs, and a reduce-on-plateau scheduler.

We compare SeisMamba with representative seismic waveform models: MagNet [4], PhaseNet [12], EQTransformer [5], AMAG [9], and U-Mamba [3]. PhaseNet and EQTransformer are adapted to magnitude regression by replacing their original output heads with regression heads. We report MSE, RMSE, MAE, �<sup>2</sup>, and inference latency. Latency is measured in milliseconds per batch of 32 waveforms on an NVIDIA T4 GPU.

Table 1: Benchmark comparison on STEAD. Time is measured in ms per batch of 32 waveforms on an NVIDIA T4 GPU. Best values are bold.
<table><tr><td>Model</td><td>MSE</td><td>RMSE</td><td>MAE</td><td>R²</td><td>Time</td></tr><tr><td>PhaseNet</td><td>0.1316</td><td>0.3627</td><td>0.2724</td><td>0.8638</td><td>0.50</td></tr><tr><td>MagNet</td><td>0.1245</td><td>0.3528</td><td>0.2561</td><td>0.8712</td><td>0.97</td></tr><tr><td>EQTransformer</td><td>0.0860</td><td>0.2932</td><td>0.2016</td><td>0.9237</td><td>1.70</td></tr><tr><td>AMAG</td><td>0.0681</td><td>0.2609</td><td>0.1467</td><td>0.9389</td><td>1.58</td></tr><tr><td>U-Mamba</td><td>0.0683</td><td>0.2613</td><td>0.1793</td><td>0.9282</td><td>1.39</td></tr><tr><td>SeisMamba</td><td>0.0628</td><td>0.2506</td><td>0.1566</td><td>0.9443</td><td>0.55</td></tr></table>

## 4.2 Benchmark Accuracy and Latency

Table 1 reports the main benchmark results on STEAD. SeisMamba achieves the best MSE, RMSE, and �<sup>2</sup> among all tested models, while remaining close to the best MAE. Its inference time is 0.55 ms per batch, only slightly slower than PhaseNet and substantially faster than EQTransformer, AMAG, and U-Mamba. These results support the central claim that sparse selective state-space modelling can improve the accuracy–latency trade-of for single-station magnitude estimation.

![](images/80e8bdd113f52e91d19cebf92e796966eba1c4f19b51b1e5e2f2739a9b7b2fc8.jpg)  
Figure 2: Spatial hold-out evaluation setup. Events from Chile and Taiwan are excluded from training and reserved only for geographically unseen testing, while the remaining STEAD events are used for training and validation.

## 4.3 Regional Hold-out Evaluation

Random splits may overestimate deployment readiness because geographically similar events can appear in both training and testing. To probe cross-region deployment more directly, we construct a regional hold-out split by excluding all events from Chile and Taiwan during training and reserving them for testing. This removes

5,478 recordings and creates a geographically unseen evaluation setting. The held-out Chile–Taiwan split contains a broad range of magnitudes, including a substantial proportion of medium-to-large events, making it relevant to practical early-warning scenarios.

As shown in Fig. 2, this split evaluates performance on geograph ically unseen seismic regions rather than only randomly held-out samples. Table 2 shows that SeisMamba reaches $R ^ { 2 } = 0 . 8 5 1 8$ and MAE = 0.2808 under this harder setting. The performance drop from the standard benchmark is expected because attenuation, site efects, source properties, and station conditions vary across regions. However, the model retains useful predictive ability and its latency remains nearly unchanged at 0.58 ms. This result does not imply full region invariance; rather, it provides a diagnostic stress test showing that the learned representation remains informative beyond the regions seen during training.

## 4.4 Ablation Study

The ablation results in Table 2 show that the final design is not simply a larger encoder. A deeper variant slightly improves accu racy over the wider variant but nearly doubles inference latency. Dense Mamba insertion performs worse than sparse placement, suggesting that state-space modelling is most useful after local waveform compression. Removing multi-scale fusion also degrades accuracy, confirming that shallow and deep waveform features provide complementary evidence for magnitude estimation.

Table 2: Regional hold-out and ablation results. Time is measured in ms per batch of 32 waveforms.
<table><tr><td>Setting</td><td>MSE</td><td>RMSE</td><td>MAE</td><td>R²</td><td>Time</td></tr><tr><td colspan="6">Regional hold-out Chile-Taiwan</td></tr><tr><td></td><td>0.1669</td><td>0.4085</td><td>0.2808</td><td>0.8518</td><td>0.58</td></tr><tr><td colspan="6">Ablation on STEAD</td></tr><tr><td>Encoder only</td><td>0.0763</td><td>0.2762</td><td>0.1776</td><td>0.9323</td><td>0.56</td></tr><tr><td>Wider</td><td>0.0669</td><td>0.2587</td><td>0.1608</td><td>0.9406</td><td>0.55</td></tr><tr><td>Deeper</td><td>0.0660</td><td>0.2569</td><td>0.1608</td><td>0.9414</td><td>1.09</td></tr><tr><td>Dense Mamba</td><td>0.0796</td><td>0.2822</td><td>0.1800</td><td>0.9293</td><td>0.61</td></tr><tr><td>No fusion</td><td>0.0778</td><td>0.2789</td><td>0.1798</td><td>0.9310</td><td>0.66</td></tr><tr><td>Final hybrid</td><td>0.0628</td><td>0.2506</td><td>0.1566</td><td>0.9443</td><td>0.55</td></tr></table>

## 5 Discussion

The results suggest two main implications for low-cost earthquake early warning. First, selective state-space modelling is a suitable backbone for single-station magnitude estimation because it improves predictive accuracy while preserving low inference latency. This is important for spatially distributed sensing settings, where models may need to run close to individual stations rather than relying entirely on centralized dense-network processing. Second, the regional hold-out experiment provides evidence that SeisMamba learns waveform representations that remain informative beyond random train–test splits. The performance drop under the Chile– Taiwan hold-out also shows that regional waveform shift is not solved: attenuation, site efects, source properties, and station conditions still afect transfer to unseen seismic regions.

The auxiliary temporal head should be interpreted as an observability mechanism rather than a complete uncertainty estimator. It provides time-resolved magnitude supervision and exposes how the model’s estimate evolves as waveform evidence accumulates. This is useful for understanding when magnitude information becomes stable, but operational EEW would still require calibrated uncertainty estimates and decision rules before such trajectories could be used for alerting.

## 6 Conclusion

This paper presented SeisMamba, a lightweight selective state-space architecture for single-station earthquake magnitude estimation. By combining local waveform encoding, sparse Mamba sequence modelling, multi-scale fusion, and auxiliary temporal supervision, SeisMamba improves the accuracy–latency trade-of on a global seismic benchmark while retaining useful performance under a regional hold-out setting. These findings indicate that Mamba-based waveform models are promising components for scalable, low-cost, and spatially distributed earthquake early warning, while also highlighting the need for broader regional adaptation and full-system EEW validation.

## Disclosure of AI Tool Use

Generative AI tools were used to assist with spelling and grammar checking.

## References

[1] Gemma Cremen and Carmine Galasso. 2020. Earthquake early warning: Recent advances and perspectives. Earth-Science Reviews 205 (2020), 103184. doi:10.1016/ j.earscirev.2020.103184

[2] Albert Gu and Tri Dao. 2024. Mamba: Linear-time sequence modeling with selective state spaces. arXiv:2312.00752 [cs.LG] doi:10.48550/arXiv.2312.00752

[3] Jun Ma, Feifei Li, and Bo Wang. 2024. U-Mamba: Enhancing long-range dependency for biomedical image segmentation. arXiv:2401.04722 [eess.IV] doi:10.48550/arXiv.2401.04722

[4] S. Mostafa Mousavi and Gregory C. Beroza. 2020. A machine-learning approach for earthquake magnitude estimation. Geophysical Research Letters 47, 1 (2020), e2019GL085976. doi:10.1029/2019GL085976

[5] S. Mostafa Mousavi, William L. Ellsworth, Weiqiang Zhu, Lindsay Y. Chuang, and Gregory C. Beroza. 2020. Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. Nature Communications 11 (2020), 3952. doi:10.1038/s41467-020-17591-w

[6] S. Mostafa Mousavi, Yixiao Sheng, Weiqiang Zhu, and Gregory C. Beroza. 2019. STanford EArthquake Dataset (STEAD): A global data set of seismic signals for AI. IEEE Access 7 (2019), 179464–179476. doi:10.1109/ACCESS.2019.2947848

[7] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, Vol. 30. Curran Associates, Inc., 5998–6008.

[8] DavidJ. Wald. 2020. Practical limitations ofearthquake early warning. Earthquake Spectra (2020). doi:10.1177/8755293020911388

[9] Ji Zhang, Aitaro Kato, Huiyu Zhu, and Wei Wang. 2025. Local magnitude estimation via an attention-based machine learning model. Seismological Research Letters 96, 4 (2025), 2187–2200. doi:10.1785/0220240354

[10] Yunfei Zhou, Haoran Ren, and Haofeng Wu. 2025. PhaseMamba: A Mamba-based deep learning model for seismic phase picking and detection. IEEE Geoscience and Remote Sensing Letters ( ) d

[11] Jingbao Zhu, Shanyou Li, Qiang Ma, Bin He, and Jindong Song. 2022. Support vector machine-based rapid magnitude estimation using transfer learning for the Sichuan–Yunnan region, China. Bulletin of the Seismological Society of America 112, 2 (2022), 894–904. doi:10.1785/0120210129

[12] Weiqiang Zhu and Gregory C. Beroza. 2019. PhaseNet: A deep-neural-networkbased seismic arrival-time picking method. Geophysical Journal International 216, 1 (2019), 261–273. doi:10.1093/gji/ggy423