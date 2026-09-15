# MAST: Label-Efficient, Robust, and Generalizable Sound Detection for Biodiversity Monitoring via Masked Audio Pretraining and Self-Training

Tianyi Xu<sup>1</sup> Daniel L. Pimentel-Alarcón<sup>1</sup> Zuzana Burivalováˇ <sup>1</sup> Claudia Solís-Lemus<sup>1,∗</sup> <sup>1</sup>University of Wisconsin–Madison

## Abstract

Passive acoustic monitoring can measure biodiversity at larger scales, but time– frequency annotation of animal vocalizations is expensive, site-specific, and difficult to sustain at scale. We present a label-efficient sound detection framework that combines masked audio pretraining with a lightweight detector on mel spectrograms, then further improves robustness through iterative self-training on unlabeled audio. We first pretrain a ViT-based encoder on unlabeled recordings via masked reconstruction and transfer the encoder to a detection backbone. To better separate animal sounds from confounding background, we add a box-level contrastive loss that pulls matched event regions together while pushing noisy negatives apart. We then apply a two-stage pseudo-labeling curriculum to exploit large unlabeled pools without additional annotation. We evaluate the performance on two ecologically distinct domains: tropical rainforest soundscapes (Indonesia) and bird vocalizations in Mediterranean habitats (Spain). On both domains, masked audio pretraining and contrastive learning consistently improve time–frequency detection under temporal and cross-site distribution shift, and self-training yields further gains in out-of-distribution performance. On the rainforest domain, MAST with self-training achieves +0.22 mAP and +0.24 F1 over the strongest baseline under cross-site shift. On the bird domain, self-training achieves +0.12 mAP and +0.10 F1 over the strongest baseline under cross-site shift. Overall, our results show that MAST can effectively extend self-supervised audio representations from clip-level tasks to robust box-level localization across diverse bioacoustic settings, providing a practical path for biodiversity monitoring with limited labels.

## 1 Introduction

Passive acoustic monitoring (PAM) offers a scalable, non-invasive way to measure biodiversity at landscape scales [55, 65, 27, 6]. Autonomous recorders can operate for months, capturing rich soundscapes that contain vocalizations from birds, mammals, insects, amphibians, as well as environmental and anthropogenic sounds [35, 50]. Turning these recordings into actionable ecological information typically follows two broad directions: one can use acoustic indices that summarize the properties of the soundscape, or species-specific recognition models that predict whether a known target sound occurs in a recording segment. However, the former does not identify which species produced the sounds, while the latter is typically limited to predefined classes of acoustically known species. In hyperdiverse and acoustically understudied soundscapes such as tropical rainforests [56], an intermediate approach is needed: a system that can automatically localize animal sounds in both time and frequency, so that they can later be classified into known taxa or presented to specialists as potentially novel events. The main bottleneck is therefore not data collection, but annotation: extracting ecological signals from raw audio requires experts to mark time–frequency regions of animal sounds, which is expensive and difficult to scale [37]. In addition, real-world soundscapes are highly non-stationary, with background conditions shifting across day/night cycles, weather, habitat structure, sensor placement, and species composition [30, 60]. As a result, models trained on a single site or day often generalize poorly [30].

![](images/3afc6daae1d14ec639c372b21087eab7c06ce19242e085c102aa3a96c7a6060b.jpg)  
Figure 1: Performance comparison on rainforest domain. MAST achieves the best overall performance, with especially large improvements under temporal and cross-site distribution shift, showing stronger robustness than fully supervised baselines and generic SSL transfer.

A common strategy is to treat spectrograms as images and apply supervised detectors such as Faster R-CNN, YOLO, or FCOS [44, 16]. While effective in fully supervised settings, these models typically require many annotations and can overfit to local recording conditions, leading to degraded performance under temporal or spatial shift [60].

To address these challenges, we propose MAST, a label-efficient framework for robust sound detection under limited supervision. We study a practical and stricter setting in which only a small amount of expert-labeled data, for example one fully annotated day from a single site, is available for training, while evaluation requires robust binary detection of “any animal sound” in terms of frequency and time across both (i) unseen days at the same site and (ii) unseen sites in the same region. Figure 2 summarizes the motivation for MAST: rainforest soundscapes exhibit substantial temporal and crosssite distribution shift, while standard supervised detectors trained on narrow label coverage often generalize poorly. MAST addresses this by decoupling representation learning from detection. First, we adopt the ViT-based masked audio encoder architecture of AudioMAE [18] and pretrain it on large unlabeled rainforest audio via masked spectrogram reconstruction. We then transfer the pretrained encoder to a lightweight detector with audio-aware adapters and an FPN [31] neck to recover locality and multi-scale structure. To improve separation between true events and confounding background, we add a box-level contrastive loss over region features. Finally, we exploit large unlabeled pools through iterative self-training with pseudo-labels and labeled-data refinement.

Time–frequency detection is ecologically more informative than clip-level tagging: it reveals which sounds overlap in time and frequency, how vocal activity is distributed across the spectrum, and can help determine whether detected events correspond to known or potentially novel taxa. This information is essential for biodiversity assessment but lost by coarser representations. Under comparable annotation budgets, MAST consistently improves robustness to temporal and site shift across two ecologically distinct domains: tropical rainforest soundscapes and Mediterranean bird vocalizations (Figure 1). Our contributions are as follows:

• We introduce a practical pipeline for binary time–frequency sound detection that combines masked audio pretraining, adaptation, and lightweight detection under limited labels.

• We design a region-level contrastive loss that improves event/background separability and strengthens generalization under distribution shift.

• We show that an iterative self-training curriculum effectively leverages unlabeled data and improves detection in out-of-domain settings without additional expert annotation.

• We evaluate on two distinct bioacoustic domains, tropical rainforest soundscapes and European bird vocalizations, and provide an end-to-end recipe for converting large unlabeled recordings into robust detectors for biodiversity monitoring.

![](images/fea016b68c08f59d57c6b61bc0ed86ff25ba821451331bd5c0aa3cdb6a41c443.jpg)  
Figure 2: Motivation of MAST. (a) Rainforest soundscapes exhibit substantial temporal and crosssite distribution shift, making detection under unseen conditions difficult. (b) Standard supervised biodiversity detectors trained on limited labels from a narrow time/site range often generalize poorly when deployed to new days and sites. (c) These challenges motivate a label-efficient and shift-robust framework that can leverage both limited expert labels and abundant unlabeled soundscapes.

## 2 Related Work

Passive acoustic monitoring and supervised sound detection. PAM enables long-duration, largescale biodiversity recording [54, 55, 10], but downstream analysis remains bottlenecked by the need for manual annotation or by acoustic indices that need ground truthing [23, 5, 4]. Classical template matching detectors scale poorly to diverse soundscapes [19, 58, 22, 7, 2, 23]. Learned systems such as BirdNET [20] and Voxaboxen [32] operate at clip level or along the time axis only, without full time–frequency detection. Casting the problem as object detection on spectrograms [44, 21, 62, 16] enables box-level localization but typically requires substantial supervision and remains brittle under site/day shift [53, 16, 11].

Self-supervised audio representations. SSL has advanced audio representation learning via masked modeling, contrastive learning, and predictive objectives [1, 17, 18, 9, 40, 39]. Recent bioacoustic foundation models such as Perch 2.0 [61], Bird-MAE [46], NatureLM-audio [49] transfer effectively to low-label monitoring tasks [36, 66], but they operate at the clip level and do not predict time– frequency bounding boxes. Existing evaluations accordingly focus on tagging, classification, or retrieval [59, 12, 45, 15]. Extending pretrained representations to dense box-level localization introduces challenges in boundary fidelity, foreground–background imbalance, and stability under non-stationary noise. Our work studies how pretrained features can be adapted for robust time– frequency detection in biodiversity soundscapes.

Weak supervision and unlabeled soundscape learning. Prior SED research has explored weak supervision, semi-supervised learning, and pseudo-labeling [52, 25, 63, 42, 43, 24]. However, most of this literature focuses on temporal event activity rather than explicit time–frequency box detection, and stable learning from pseudo labels remains difficult under distribution shift [52, 32]. Our method differs in both objective and setting: we use confidence-weighted pseudo-boxes within a staged self-training pipeline, where exploration on diverse pseudo-labeled soundscapes expands coverage under shift, followed by expert-guided refinement that re-anchors precision.

## 3 Methods

Figure 3 provides an overview of MAST, which combines masked-audio pretraining, label-efficient detector adaptation, and self-training for robust sound detection under distribution shift.

![](images/3fed79c47237d72e8ab1d6eb38bc5122157e18826c67473bc3ed7555073789f3.jpg)  
Figure 3: Overview of the MAST pipeline. (A) Masked-audio pretraining learns general acoustic representations from unlabeled rainforest spectrograms via masked reconstruction. (B) Detector training transfers the pretrained encoder to a detection model with an audio-aware adapter and lightweight FPN, together with a box-level contrastive objective for event separation. (C) Staged selftraining improves robustness under distribution shift by first generating pseudo-labels on unlabeled audio and then retraining with exploration on pseudo-labeled data followed by refinement on expertlabeled data.

## 3.1 Problem Setup

Given an audio waveform $x ( t )$ , we compute a log-mel spectrogram $S ~ \in ~ \mathbb { R } ^ { F \times T }$ Each clip may contain multiple annotated time–frequency boxes $\begin{array} { r c l } { \hat { B } } & { = } & { \overline { { \{ } } } ( b _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }  \end{array}$ , where each box $b _ { i } \ = \ [ t _ { i } ^ { ( 1 ) } , t _ { i } ^ { ( 2 ) } , f _ { i } ^ { ( 1 ) } , f _ { i } ^ { ( 2 ) } ]$ . Although the dataset includes sonotype labels $y _ { i }$ , our primary task is binary detection (“any animal sound”), so all annotated events are treated as a single foreground class. We consider a low-label regime with limited expert-labeled data $\mathcal { D } _ { \ell }$ and abundant unlabeled data $\mathcal { D } _ { u } .$ , and aim to achieve robust detection under temporal and spatial shift.

## 3.2 Tasks and Objectives

Overview. Our pipeline has three stages: (i) self-supervised representation learning on unlabeled soundscapes, (ii) label-efficient supervised time–frequency detection on expert boxes, and (iii) staged self-training to exploit additional unlabeled audio. See Fig. 3 for an overview of MAST.

Self-supervised pretraining. Given an unlabeled log-mel spectrogram chunk $S \sim { \mathcal { D } } _ { u }$ , we pretrain a masked audio encoder $E _ { \phi }$ by masking a set of time-frequency patches M and reconstructing the masked content with a lightweight decoder. We minimize the masked reconstruction loss

$$
\operatorname* { m i n } _ { \phi } \mathbb { E } _ { S \sim \mathcal { D } _ { u } } \left[ \mathcal { L } _ { \mathrm { M A E } } ( S ; \phi ) \right] , \quad \mathcal { L } _ { \mathrm { M A E } } = \frac { 1 } { | M | } \sum _ { ( f , t ) \in M } \| \hat { S } [ f , t ] - S [ f , t ] \| _ { 2 } ^ { 2 } .
$$

Supervised detection on expert labels. For time-frequency detection, we reuse the pretrained encoder as the detector backbone and attach lightweight audio-aware adapters $A _ { \psi }$ and a feature pyramid $N _ { \gamma \cdot } \mathbf { A }$ detection head $D _ { \theta }$ predicts box coordinates and confidence scores. We train on the labeled set $\dot { \mathcal { D } } _ { \ell }$ with a standard detection objective:

$$
\mathcal { L } _ { \mathrm { d e t } } = \lambda _ { \mathrm { c l s } } \mathcal { L } _ { \mathrm { c l s } } + \lambda _ { \mathrm { b o x } } \mathcal { L } _ { \mathrm { b o x } } + \lambda _ { \mathrm { i o u } } \mathcal { L } _ { \mathrm { i o u } } ,
$$

where $\mathcal { L } _ { \mathrm { c l s } }$ is focal loss, $\mathcal { L } _ { \mathrm { b o x } }$ is $\ell _ { 1 }$ regression on box corners, and $\mathcal { L } _ { \mathrm { i o u } }$ is GIoU loss [48]. To improve robustness under background shift, we add a box-level contrastive term that encourages invariance across perturbations of the same event while separating events from hard background regions. For

each ground-truth box we form positives via spatial/temporal jittering, and sample negatives from low-IoU proposals. Let $z _ { i }$ be the normalized projected feature pooled from box i, the contrastive loss is

$$
\mathcal { L } _ { \mathrm { c o n } } = \frac { 1 } { N _ { + } } \sum _ { i } \left[ - \log \frac { \sum _ { p \in P _ { i } } \exp \left( \left. z _ { i } , p \right. / \tau \right) } { \sum _ { p \in P _ { i } } \exp \left( \left. z _ { i } , p \right. / \tau \right) + \sum _ { n \in N } \exp \left( \left. z _ { i } , n \right. / \tau \right) } \right] .
$$

The supervised training objective becomes

$$
\operatorname* { m i n } _ { \psi , \gamma , \theta , \phi _ { \mathrm { t o p } } } \mathbb { E } _ { ( S , B ) \sim \mathcal { D } _ { \ell } } \left[ \mathcal { L } _ { \mathrm { d e t } } \left( S , B \right) + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } \left( S , B \right) \right] ,
$$

where $\phi _ { \mathrm { t o p } }$ denotes the parameters of the top k encoder blocks that are unfrozen during fine-tuning.

Self-training on unlabeled audio. Starting from a seed detector trained on $\mathcal { D } _ { \ell } .$ , we generate pseudo boxes on $\mathcal { D } _ { u }$ using class-agnostic NMS and a confidence threshold q, yielding $\Pi _ { q } \left( \mathcal { D } _ { u } \right)$ . We then retrain on the union:

$$
\operatorname* { m i n } _ { \psi , \gamma , \theta , \phi _ { \mathrm { t o p } } } \mathbb { E } _ { ( S , B ) \sim \mathcal { D } _ { \ell } \cup \Pi _ { q } ( \mathcal { D } _ { u } ) } \left[ \mathcal { L } _ { \mathrm { d e t } } \left( S , B \right) + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } \left( S , B \right) \right] .
$$

To reduce confirmation bias, we apply confidence weighting that down-weights uncertain pseudolabels and keep the backbone largely frozen in early self-training before optionally unfreezing top layers. See App. B for a formal definition of this stage.

## 3.3 Detector Design

We use the pretrained ViT-B/16 masked audio encoder $E _ { \phi }$ as the backbone for time-frequency detection. Given a log-mel spectrogram chunk, the encoder outputs a token grid which we reshape into a feature map $\overset { \smile } { \boldsymbol { E } } \doteq \mathbb { R } ^ { C \times ^ { \bullet } H \times W }$ . We then attach a lightweight audio-aware adapter $A _ { \psi } , \mathbf { a }$ feature pyramid network $N _ { \gamma }$ , and a detection head $D _ { \theta }$ . For label-efficient transfer, we either freeze $E _ { \phi }$ or fine-tune only the top k transformer blocks, while training $A _ { \psi } , N _ { \gamma }$ , and $D _ { \theta }$ . Our default head is an anchor-free FCOS predicting classification, centerness, and box regression. Full architectural details, adapter variants, and additional ablations are provided in App. E.3, E.4, and F.2.

Resolution bottleneck and asymmetric adapter. Transferring patch-based ViTs to dense time– frequency detection introduces a resolution bottleneck: with patch size $p { = } 1 6$ and spectrogram $T { \times } F = 1 0 2 4 { \times } 1 2 8 .$ , the encoder produces a 64×8 token grid with strides $s _ { t } = s _ { f } = 1 6$ . We show (Proposition 1, App. C) that grid quantization can degrade IoU to at most $\delta _ { t } \delta _ { f } / [ ( \delta _ { t } ^ { ' } + s _ { t } ) ( \delta _ { f } + s _ { f } ) ]$ in the worst case, and that narrow-band events with $\delta _ { f } \ \leq \ p$ may fail to reach IoU $\geq 0 . 5$ under worst-case alignment (Corollary 2). This bottleneck is asymmetric: the frequency axis has only 8 tokens, ∼1 kHz each, while many vocalizations span $< 1$ kHz in bandwidth.

Our adapter resolves this via asymmetric learnable upsampling $( u _ { t } { = } 2 , u _ { f } { = } 4 )$ , producing a 128×32 feature map with effective strides $8 \times 4 .$ . The larger frequency factor addresses the more severe bottleneck. Upsampled features are refined by anisotropic k×1 and $1 \times k$ convolutions that exploit the separable spectro-temporal structure of animal sounds. Empirically, this adapter improves crosssite OOD F1 from 0.348 to 0.447 and achieves the highest mean IoU (0.682) among all variants (Tables 3, 10). See $\mathbf { A p p }$ . C for formal analysis with proofs.

## 3.4 Self-training on Unlabeled Soundscapes

Starting from a seed detector trained on $\mathcal { D } _ { \ell } .$ , we generate pseudo boxes on unlabeled audio $\mathcal { D } _ { u }$ by running the detector, applying class-agnostic NMS, and retaining predictions above a confidence threshold $q ,$ yielding a pseudo-labeled dataset $\Pi _ { q } ( \mathcal { D } _ { u } )$ . We then retrain with a two-stage curriculum: Stage 1 trains on $\bar { \Pi _ { q } } ( \bar { \mathcal { D } } _ { u } )$ to broaden coverage across diverse backgrounds, and Stage 2 fine-tunes on $\mathcal { D } _ { \ell }$ to re-anchor the model to clean expert boxes and calibrate false positives. In both stages we optimize the detection objective with our contrastive term. This cycle is repeated for 1–2 rounds. We show (Proposition 5, App. D) that this procedure progressively reduces the distributional gap to the

Table 1: In-domain detection performance on two domains: Rainforest (tropical soundscapes) and BIRDeep (European bird vocalizations [33]). Higher is better.
<table><tr><td rowspan="2">Method</td><td colspan="5">Rainforest</td><td colspan="5">BIRDeep</td></tr><tr><td>Prec.</td><td>Rec.</td><td>F1</td><td>mAP</td><td>mIoU</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>mAP</td><td>mIoU</td></tr><tr><td>Faster R-CNN [47]</td><td>0.6832</td><td>0.8390</td><td>0.7226</td><td>0.7715</td><td>0.8590</td><td>0.3716</td><td>0.2061</td><td>0.2651</td><td>0.1537</td><td>0.6587</td></tr><tr><td>FCOS [57]</td><td>0.9708</td><td>0.6601</td><td>0.7858</td><td>0.6334</td><td>0.9290</td><td>0.7571</td><td>0.8091</td><td>0.7822</td><td>0.6650</td><td>0.6849</td></tr><tr><td>YOLO26 [51]</td><td>0.8659</td><td>0.3306</td><td>0.4309</td><td>0.3475</td><td>0.8102</td><td>0.7897</td><td>0.7737</td><td>0.7816</td><td>0.6183</td><td>0.7181</td></tr><tr><td>DETR [8]</td><td>0.5905</td><td>0.7839</td><td>0.6501</td><td>0.7422</td><td>0.8400</td><td>0.3256</td><td>0.7525</td><td>0.4545</td><td>0.3723</td><td>0.6024</td></tr><tr><td>AudioMAE [18]</td><td>0.9454</td><td>0.6903</td><td>0.7775</td><td>0.6736</td><td>0.9352</td><td>0.7951</td><td>0.6859</td><td>0.7364</td><td>0.6025</td><td>0.6756</td></tr><tr><td>MAST (ours)</td><td>0.9535</td><td>0.7021</td><td>0.7916</td><td>0.6853</td><td>0.9387</td><td>0.7824</td><td>0.7010</td><td>0.7395</td><td>0.6011</td><td>0.6822</td></tr><tr><td>MAST + ST (ours)</td><td>0.9548</td><td>0.6277</td><td>0.7338</td><td>0.6277</td><td>0.9457</td><td>0.6709</td><td>0.9061</td><td>0.7709</td><td>0.7783</td><td>0.7339</td></tr></table>

target domain, and that confidence weighting provably reduces effective label noise (Proposition 6). See App. B for formal definitions and App. D for theoretical analysis.

## 4 Experiment Setup

Datasets and Splits. We evaluate on two ecologically distinct domains to test cross-domain generality.

Rainforest domain. We study sound detection on passive acoustic recordings from tropical rainforest in East Kalimantan, Indonesia, from 2017–2019. Our primary labeled dataset consists of 24 hours recorded on July 10, 2018 at one site. Audio is resampled to 16 kHz and segmented into 10.24 s clips; each clip is converted to a 128 × 1024 log-mel spectrogram (App. E.1). Expert annotators provide time–frequency bounding boxes for animal sounds (vocalizations, stridulations, etc.) spanning hundreds of “sonotypes” - unique sound types. We collapse all sonotypes into a single foreground category and evaluate binary detection. We evaluate in two regimes: in-domain, where we do a 70/15/15 train/val/test split of the labeled site/day, and out-of-distribution (OOD), where recordings come from different days and different sites within the same landscape, introducing covariate shift in background conditions, species activity, and sensor characteristics. A large pool of unlabeled rainforest audio of ∼661.7 h is used for masked audio pretraining and self-training. See App. E.2 for detailed statistics.

Bird domain. To test cross-domain generality, we additionally evaluate on the BIRDeep dataset [33], which contains 641 recordings of bird vocalizations from 9 autonomous recording sites across 4 habitat types: low shrubland, high shrubland, ecotone, and marshland, in Doñana National Park, Spain. Annotations cover 38 bird species with time–frequency bounding boxes; we again collapse to binary detection. For masked audio pretraining, we combine BIRDeep unlabeled audio with BirdSet XCM [45], a large-scale bird audio collection of ∼900K spectrogram chunks. We construct in-domain, temporal-OOD, and site-OOD evaluation splits. See App. E.2 for dataset details.

Baselines. Recent work has applied object detectors to spectrograms for dense time–frequency detection in marine mammals [16], bird song [44, 62], and general soundscapes [11], but these rely on fully supervised detectors trained from scratch. We compare against representative architectures, all adapted to mel-spectrograms and trained on the same labeled set:

• Faster R-CNN [47]: A two-stage detector with a ResNet-50 backbone and FPN, adapted to log-mel spectrogram. We use class-agnostic detection with standard region proposal + RoIAlign.

• FCOS [57]: An anchor-free one-stage detector with a ResNet-50 + FPN backbone, predicting per-location classification, centerness, and box regression on the spectrogram lattice.

• YOLO26 [51]: A real-time one-stage detector. We use the YOLO26x architecture, adapted to log-mel spectrogram inputs.

• DETR [8]: A transformer-based end-to-end detector using set prediction with bipartite matching, adapted to spectrogram inputs. This baseline evaluates whether a query-based detection paradigm transfers to detection in low-label soundscapes.

• AudioMAE [18]: A transfer-learning baseline that initializes the backbone with a publicly available AudioMAE checkpoint pretrained on AudioSet, then fine-tunes the detector on our labeled split. This isolates the effect of generic large-scale audio pretraining versus in-domain self-supervised pretraining used in our method.

Table 2: Out-of-distribution robustness on Rainforest and BIRDeep. Temporal OOD evaluates on recordings from different dates at the same site. Cross-site OOD evaluates on a geographically distinct recording location. Higher is better.
<table><tr><td></td><td colspan="5">Rainforest</td><td colspan="5">BIRDeep</td></tr><tr><td>Method</td><td>Prec.</td><td>Rec.</td><td>Fl</td><td>mAP</td><td>mIoU</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>mAP</td><td>mIoU</td></tr><tr><td colspan="9">Temporal OOD</td><td></td></tr><tr><td>Faster R-CNN [47]</td><td>0.0400</td><td>0.2479</td><td>0.0689</td><td>0.0168</td><td>0.4259</td><td>0.3199</td><td>0.1020</td><td>0.1547</td><td>0.0909</td><td>0.5559</td></tr><tr><td>FCOS [57]</td><td>0.5072</td><td>0.1029</td><td>0.1711</td><td>0.1145</td><td>0.6429</td><td>0.7269</td><td>0.2885</td><td>0.4130</td><td>0.2116</td><td>0.6510</td></tr><tr><td>YOLO26 [51]</td><td>0.9459</td><td>0.0926</td><td>0.1688</td><td>0.0909</td><td>0.7603</td><td>0.7559</td><td>0.2040</td><td>0.3212</td><td>0.1520</td><td>0.6767</td></tr><tr><td>DETR [8]</td><td>0.1463</td><td>0.1800</td><td>0.1450</td><td>0.1316</td><td>0.5237</td><td>0.1923</td><td>0.6521</td><td>0.2971</td><td>0.3585</td><td>0.5528</td></tr><tr><td>AudioMAE [18]</td><td>0.8178</td><td>0.2085</td><td>0.3323</td><td>0.1636</td><td>0.6382</td><td>0.7959</td><td>0.3660</td><td>0.5014</td><td>0.2983</td><td>0.6328</td></tr><tr><td>MAST (ours)</td><td>0.7810</td><td>0.2968</td><td>0.4301</td><td>0.2577</td><td>0.6859</td><td>0.7618</td><td>0.4808</td><td>0.5895</td><td>0.3838</td><td>0.6083</td></tr><tr><td>MAST + ST (ours)</td><td>0.7120</td><td>0.4603</td><td>0.5591</td><td>0.3340</td><td>0.6292</td><td>0.6755</td><td>0.6247</td><td>0.6491</td><td>0.4280</td><td>0.5903</td></tr><tr><td colspan="9">Cross-site OOD</td><td></td><td></td></tr><tr><td>Faster R-CNN [47]</td><td>0.0497</td><td>0.2341</td><td>0.0820</td><td>0.0281</td><td>0.4223</td><td>0.8824</td><td>0.0744</td><td>0.1372</td><td>0.0909</td><td>0.5863</td></tr><tr><td>FCOS [57]</td><td>0.5848</td><td>0.0892</td><td>0.1548</td><td>0.0909</td><td>0.6513</td><td>0.9725</td><td>0.2852</td><td>0.4410</td><td>0.2592</td><td>0.6566</td></tr><tr><td>YOLO26 [51]</td><td>0.9910</td><td>0.0849</td><td>0.1564</td><td>0.0909</td><td>0.7484</td><td>0.8889</td><td>0.1934</td><td>0.3177</td><td>0.1675</td><td>0.7010</td></tr><tr><td>DETR [8]</td><td>0.1269</td><td>0.1758</td><td>0.1338</td><td>0.1130</td><td>0.5607</td><td>0.1334</td><td>0.6621</td><td>0.2220</td><td>0.3032</td><td>0.5554</td></tr><tr><td>AudioMAE [18]</td><td>0.8028</td><td>0.2113</td><td>0.3346</td><td>0.1742</td><td>0.6677</td><td>0.7266</td><td>0.4284</td><td>0.5390</td><td>0.3217</td><td>0.6072</td></tr><tr><td>MAST (ours)</td><td>0.7562</td><td>0.3169</td><td>0.4466</td><td>0.2573</td><td>0.6824</td><td>0.6476</td><td>0.5059</td><td>0.5680</td><td>0.3803</td><td>0.6001</td></tr><tr><td>MAST + ST (ours)</td><td>0.6978</td><td>0.4838</td><td>0.5715</td><td>0.3921</td><td>0.6420</td><td>0.6522</td><td>0.6438</td><td>0.6434</td><td>0.4414</td><td>0.5943</td></tr></table>

Training Details. All models use the same log-mel spectrogram processing and dataset-wide normalization. We train with AdamW optimizer and a warmup–cosine learning-rate schedule. For MAST detectors, we use component-wise learning rates (smaller for the backbone, larger for adapters/FPN/head) and unfreeze only the top encoder blocks during fine-tuning. We also apply lightweight spectrogram augmentations. Unless stated otherwise, we match training budgets and augmentation pipelines across methods. See App. E.5, E.6 for details.

Evaluation. We evaluate binary time–frequency detection of “any animal sound,” reporting precision, recall, F1, mean average precision (mAP), and mean intersection-over-union (mIoU) at IoU ≥ 0.5. We evaluate in two regimes: in-domain with held-out chunks from the same site/date and OOD on different days or sites. Confidence thresholds are tuned on the in-domain validation split. See App. E.7 for full details.

## 5 Results

In-domain performance. Table 1 reports in-domain results. On rainforest, MAST achieves the best F1 of 0.792 and mean IoU of 0.939, demonstrating that self-supervised pretraining and audio-aware adaptation do not sacrifice in-domain accuracy despite being designed primarily for robustness. On BIRDeep, FCOS and YOLO26 lead in F1 at 0.782, while MAST reaches 0.740. Self-training closes this gap: MAST+ST reaches 0.771 F1 and surpasses all baselines in mAP at 0.778 vs. 0.665 for FCOS, in recall at 0.906, and in mean IoU at 0.734. Crucially, the remaining in-domain difference inverts under distribution shift, where MAST provides the largest gains (Table 2), highlighting that in-domain metrics alone are insufficient for field deployment.

Out-of-distribution robustness. Table 2 reports OOD results under temporal and cross-site shift. Across both domains, detectors trained from scratch degrade sharply. For example, FCOS drops from 0.786 to 0.155 cross-site F1 on rainforest, a 5× collapse. In contrast, MAST achieves the strongest overall F1 and mAP. On the rainforest domain, MAST raises cross-site F1 from 0.335 to 0.447 and temporal F1 from 0.332 to 0.430 over the strongest transfer baseline AudioMAE. On BIRDeep, MAST improves temporal F1 from 0.501 to 0.590 and cross-site F1 from 0.539 to 0.568 over AudioMAE, confirming that domain-matched pretraining is more transferable than generic AudioSet initialization. Notably, most baselines exhibit a sharp precision–recall imbalance under shift: FCOS and YOLO26 retain high precision but collapse in recall, while DETR and Faster R-CNN maintain recall at the cost of precision. MAST is the only method that sustains both, yielding the highest F1 across conditions. The consistent pattern across both ecologically distinct domains, with modest in-domain gaps but substantial OOD improvements, confirms that MAST’s primary benefit is robustness to distribution shift, the key deployment challenge in real-world biodiversity monitoring.

a.) Self-training Performance  
![](images/c7ba503293411a22f5261d61695cfe6c92985f9a46c639c5bb6e23df0db041e4.jpg)

b.) Precision-Recall Each Iteration  
![](images/f32adfa41454f4f94d15b95947b987f771b1ca84f6aebdcd8e270202874b1150.jpg)  
Figure 4: Effect of iterative self-training on F1 for rainforest domain. Round 0 is the best MAST detector after expert label refinement. Rounds 1–2 are models trained with the two-stage curriculum. OOD F1 improves monotonically from 0.4466 to 0.5442 and 0.5715, while in-domain F1 drops after Round 1 from 0.7916 to 0.6760 and partially recovers after Round 2 to 0.7338.

Table 3: Pipeline ablation on cross-site OOD detection on rainforest domain. Starting from the full MAST model, we progressively remove components to measure their individual contributions. We also report an AudioSet-pretrained variant and the FCOS baseline for reference. Higher is better.
<table><tr><td>Variant</td><td>Precision</td><td>Recall</td><td>F1</td><td>mAP</td><td>Mean IoU</td></tr><tr><td>MAST (full)</td><td>0.7562</td><td>0.3169</td><td>0.4466</td><td>0.2573</td><td>0.6824</td></tr><tr><td>Contrastive loss</td><td>0.6536</td><td>0.3327</td><td>0.4410</td><td>0.2434</td><td>0.6555</td></tr><tr><td>Audio-aware adapter</td><td>0.6380</td><td>0.2397</td><td>0.3484</td><td>0.2237</td><td>0.6607</td></tr><tr><td>Replace with AudioSet pretraining</td><td>0.8028</td><td>0.2113</td><td>0.3346</td><td>0.1742</td><td>0.6677</td></tr><tr><td>All pretraining (from scratch)</td><td>0.3607</td><td>0.3666</td><td>0.3636</td><td>0.1840</td><td>0.5588</td></tr><tr><td>FCOS baseline</td><td>0.5848</td><td>0.0892</td><td>0.1548</td><td>0.0909</td><td>0.6513</td></tr></table>

Self-training further boosts rainforest OOD F1 to 0.559 temporal and 0.572 cross-site, and BIRDeep OOD F1 to 0.649 temporal and 0.643 cross-site.

Effect of self-training. Figure 4 shows that OOD F1 improves monotonically from 0.447 at Round 0 to 0.544 after one self-training round and to 0.572 after two rounds, a +0.13 absolute gain, while in-domain F1 dips after Round 1 and partially recovers after Round 2. The in-domain dip is expected: pseudo-label training broadens the decision boundary to cover unseen acoustic conditions, temporarily trading in-domain precision for OOD coverage, while the Stage 2 expert refinement partially corrects this drift. The first round provides the largest jump, suggesting that pseudo-labeling mainly helps by exposing the detector to diverse background conditions absent from the labeled site/day. The monotonic OOD trend, achieved without any additional expert annotation, supports self-training as a practical mechanism for improving cross-condition coverage. On BIRDeep, self-training similarly improves OOD F1 from 0.568 to 0.643 cross-site and from 0.590 to 0.649 temporal, with especially large mAP gains from 0.380 to 0.441 cross-site, confirming that the two-stage curriculum generalizes across domains. Per-round results are in App. F.1. The consistent gains on both domains suggest that further rounds or larger unlabeled pools could yield additional improvements.

Ablation studies. Table 3 decomposes MAST’s gains under cross-site OOD shift by progressively removing components from the full model before self-training. Removing the contrastive loss drops precision sharply (0.756→0.654) while recall rises slightly, leaving F1 nearly unchanged; the loss thus acts primarily as a confidence calibrator that sharpens the decision boundary, producing higherquality pseudo-labels for self-training. Removing the adapter further reduces F1 from 0.441 to 0.348, demonstrating the importance of resolving the resolution bottleneck (Sec. 3.3). Replacing domain-matched pretraining with generic AudioSet pretraining yields lower mAP, 0.174 vs. 0.224, confirming the value of in-domain representations. Removing all pretraining entirely still outperforms the CNN baseline in F1, 0.364 vs. 0.155, suggesting that the ViT’s self-attention can model longrange spectro-temporal dependencies that aid generalization under background shift, even without pretrained weights. Each component contributes a meaningful and complementary gain; see App. F.2 for additional adapter comparisons.

Table 4: Ablation of the self-training curriculum on cross-site OOD detection on rainforest domain. We compare the MAST model against Stage 1-only, Stage 2-only, a one-stage union of expert-labeled and pseudo-labeled data, and the full two-stage curriculum. Higher is better.
<table><tr><td>Variant</td><td>Precision</td><td>Recall</td><td>F1</td><td>mAP</td><td>Mean IoU</td></tr><tr><td>Base (no ST)</td><td>0.7562</td><td>0.3169</td><td>0.4466</td><td>0.2573</td><td>0.6824</td></tr><tr><td>Stage 1 only</td><td>0.9714</td><td>0.1201</td><td>0.2138</td><td>0.1731</td><td>0.7439</td></tr><tr><td>Stage 2 only</td><td>0.9971</td><td>0.1046</td><td>0.1893</td><td>0.1789</td><td>0.7644</td></tr><tr><td>One stage union</td><td>0.1572</td><td>0.7341</td><td>0.2590</td><td>0.3894</td><td>0.5525</td></tr><tr><td>Stage 1 + Stage 2 (Full)</td><td>0.5044</td><td>0.5909</td><td>0.5442</td><td>0.3864</td><td>0.6067</td></tr></table>

Table 4 ablates the self-training curriculum. Stage 1 or Stage 2 alone both collapse recall despite high precision, yielding poor F1, demonstrating that naive single-stage pseudo-label training is insufficient. A one-stage union achieves high recall and mAP but precision drops to 0.157, indicating that mixing noisy pseudo supervision with expert labels without calibration leads to degenerate behavior. Only the full two-stage curriculum achieves the best F1 of 0.544 by balancing coverage and precision: Stage 1 broadens coverage under shift, while Stage 2 re-anchors the detector to clean expert annotations. This validates the curriculum design as essential rather than incidental to self-training success.

Retrieval-based event characterization. A practical benefit of MAST’s design is that the pretrained encoder naturally provides meaningful box-level embeddings: each detected event can be represented by pooling the corresponding time–frequency region from the feature map. This enables a retrievalbased workflow where detected sounds are matched to annotated examples via nearest-neighbor lookup for zero-shot classification, or grouped by acoustic similarity to discover candidate novel sounds for expert review. On oracle ground-truth boxes, retrieval achieves 72.3% Top-1 accuracy over 131 sonotypes; on high-confidence predicted boxes with IoU ≥ 0.5, Top-1 reaches 96.1%, indicating that detections align with acoustically prototypical events. This allows ecologists to perform species-level analysis directly on detector outputs without retraining a multi-class model. Full results are in App. F.4.

Label budget and sampling strategy. A practical question for bioacousticians deploying a new detector is: how much annotation effort is needed, and how should it be allocated? We study this by training the FCOS baseline on varying numbers of labeled chunks under three sampling strategies. Both random and time-balanced sampling substantially outperform sequential labeling, especially at low budgets where the gap reaches +0.13 F1 at N=1000 chunks, because sequential annotation under-represents acoustic conditions outside the labeled time window. Time-balanced sampling performs slightly better than random at most budgets, and as the budget grows all strategies converge. Since sequential labeling is the most natural default, where annotators typically start at the beginning of a recording and label forward in time, these results serve as a practical warning: practitioners should prioritize temporal diversity over volume when annotation resources are limited. Full results and protocol details are in App. F.5.

## 6 Conclusion

We introduced MAST, a label-efficient framework for biodiversity sound detection that combines masked-audio pretraining, audio-aware detection adaptation, and iterative self-training. Using expert labels from only a single annotated day at one site, MAST substantially outperforms supervised baselines and generic AudioSet transfer under both temporal and cross-site distribution shift across two ecologically distinct domains, tropical rainforest soundscapes and Mediterranean bird vocalizations. Domain-matched self-supervised pretraining proves more transferable than large-scale generic pretraining, the audio-aware adapter resolves a critical resolution bottleneck for narrow-band vocalizations, and the two-stage self-training curriculum yields monotonic OOD gains without additional expert annotation. Together, these results demonstrate a scalable path from large unlabeled acoustic archives to robust time–frequency detectors under realistic label-limited conditions, supporting biodiversity assessment tasks currently bottlenecked by manual analysis, such as tracking changes in vocal activity across habitats, flagging acoustically novel events for taxonomic review, and scaling soundscape monitoring to landscape-level surveys. Limitations and future directions are discussed in App. A.

## References

[1] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020.

[2] David J Barker, Christopher Herrera, and Mark O West. Automated detection of 50-khz ultrasonic vocalizations using template matching in xbat. Journal ofneuroscience methods, 236: 68–75, 2014.

[3] Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. A theory of learning from different domains. Machine learning, 79(1): 151–175, 2010.

[4] Tom Bradfer-Lawrence, Camille Desjonqueres, Alice Eldridge, Alison Johnston, and Oliver Metcalf. Using acoustic indices in ecology: Guidance on study design, analyses and interpretation. Methods in Ecology and Evolution, 14(9):2192–2204, 2023.

[5] Tom Bradfer-Lawrence, Brad Duthie, Carlos Abrahams, Matyáš Adam, Ross J Barnett, Amy Beeston, Jennifer Darby, Benedict Dell, Nick Gardner, Amandine Gasc, et al. The acoustic index user’s guide: A practical manual for defining, generating and understanding current and future acoustic indices. Methods in Ecology and Evolution, 16(6):1040–1050, 2025.

[6] Zuzana Burivalova, Purnomo, Bambang Wahyudi, Timothy M Boucher, Peter Ellis, Anthony Truskinger, Michael Towsey, Paul Roe, Delon Marthinus, Bronson Griscom, et al. Using soundscapes to investigate homogenization of tropical forest diversity in selectively logged forests. Journal ofApplied Ecology, 56(11):2493–2504, 2019.

[7] Rachel T Buxton, Megan F McKenna, Mary Clapp, Erik Meyer, Erik Stabenau, Lisa M Angeloni, Kevin Crooks, and George Wittemyer. Efficacy of extracting indices from large-scale acoustic recordings to monitor biodiversity. Conservation Biology, 32(5):1174–1184, 2018.

[8] Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. End-to-end object detection with transformers. In European conference on computer vision, pages 213–229. Springer, 2020.

[9] Sanyuan Chen, Yu Wu, Chengyi Wang, Shujie Liu, Daniel Tompkins, Zhuo Chen, and Furu Wei. Beats: Audio pre-training with acoustic tokenizers. arXiv preprint arXiv:2212.09058, 2022.

[10] Anna F Cord, Kevin Darras, Ryo Ogawa, Luc Barbaro, Charlotte Gerling, Maria Kernecker, Nonka Markova-Nenova, Gabriela Rodriguez-Barrera, Felix Zichner, and Frank Wätzold. Leveraging passive acoustic monitoring for result-based agri-environmental schemes: Opportunities, challenges and next steps. Biological Conservation, 305:111042, 2025.

[11] Janek Ebbers, François G Germain, Gordon Wichern, and Jonathan Le Roux. Sound event bounding boxes. arXiv preprint arXiv:2406.04212, 2024.

[12] Benjamin Elizalde, Soham Deshmukh, Mahmoud Al Ismail, and Huaming Wang. Clap learning audio concepts from natural language supervision. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

[13] Spencer Frei, Difan Zou, Zixiang Chen, and Quanquan Gu. Self-training converts weak learners to strong learners in mixture models. In Gustau Camps-Valls, Francisco J. R. Ruiz, and Isabel Valera, editors, Proceedings of The 25th International Conference on Artificial Intelligence and Statistics, volume 151 of Proceedings ofMachine Learning Research, pages 8003–8021. PMLR, 28–30 Mar 2022. URL https://proceedings.mlr.press/v151/frei22a.html.

[14] Jort F Gemmeke, Daniel PW Ellis, Dylan Freedman, Aren Jansen, Wade Lawrence, R Channing Moore, Manoj Plakal, and Marvin Ritter. Audio set: An ontology and human-labeled dataset for audio events. In 2017 IEEE international conference on acoustics, speech and signal processing (ICASSP), pages 776–780. IEEE, 2017.

[15] Masato Hagiwara, Benjamin Hoffman, Jen-Yu Liu, Maddie Cusimano, Felix Effenberger, and Katie Zacarian. Beans: The benchmark of animal sounds. In ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2023.

[16] Quentin Hamard, Minh-Tan Pham, Dorian Cazau, and Karine Heerah. A deep learning model for detecting and classifying multiple marine mammal species from passive acoustic data. Ecological Informatics, 84:102906, 2024.

[17] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM transactions on audio, speech, and language processing, 29:3451–3460, 2021.

[18] Po-Yao Huang, Hu Xu, Juncheng Li, Alexei Baevski, Michael Auli, Wojciech Galuba, Florian Metze, and Christoph Feichtenhofer. Masked autoencoders that listen. Advances in Neural Information Processing Systems, 35:28708–28720, 2022.

[19] Olaf Jahn, Todor D Ganchev, Marinez I Marques, and Karl-L Schuchmann. Automated sound recognition provides insights into the behavioral ecology of a tropical bird. PloS one, 12(1): e0169041, 2017.

[20] Stefan Kahl, Connor M Wood, Maximilian Eibl, and Holger Klinck. Birdnet: A deep learning solution for avian diversity monitoring. Ecological Informatics, 61:101236, 2021.

[21] Chieh-Chi Kao, Weiran Wang, Ming Sun, and Chao Wang. R-crnn: Region-based convolutional recurrent neural network for audio event detection. arXiv preprint arXiv:1808.06627, 2018.

[22] Jonathan Katz, Sasha D Hafner, and Therese Donovan. Tools for automated acoustic monitoring within the r package monitor. Bioacoustics, 25(2):197–210, 2016.

[23] Arik Kershenbaum, Çaglar Akçay, Lakshmi Babu-Saheer, Alex Barnhill, Paul Best, Jules˘ Cauzinille, Dena Clink, Angela Dassow, Emmanuel Dufourq, Jonathan Growcott, et al. Automatic detection for bioacoustic research: a practical guide from and for biologists and computer scientists. Biological Reviews, 100(2):620–646, 2025.

[24] Ji Won Kim, Sang Won Son, Yoonah Song, Hong Kook Kim, Il Hoon Song, and Jeong Eun Lim. Semi-supervsied learning-based sound event detection using freuqency dynamic convolution with large kernel attention for dcase challenge 2023 task 4. arXiv preprint arXiv:2306.06461, 2023.

[25] Qiuqiang Kong, Yong Xu, Wenwu Wang, and Mark D Plumbley. Sound event detection of weakly labelled data with cnn-transformer and automatic threshold optimization. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 28:2450–2460, 2020.

[26] Ananya Kumar, Tengyu Ma, and Percy Liang. Understanding self-training for gradual domain adaptation. In International conference on machine learning, pages 5468–5479. PMLR, 2020.

[27] Marco Lassandro, Liam Bolitho, and David Newell. A systematic review of passive acoustic monitoring for anuran conservation. Bioacoustics, 34(5):579–628, 2025.

[28] Xiang Li, Wenhai Wang, Lijun Wu, Shuo Chen, Xiaolin Hu, Jun Li, Jinhui Tang, and Jian Yang. Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection. Advances in neural information processing systems, 33:21002–21012, 2020.

[29] Yanghao Li, Hanzi Mao, Ross Girshick, and Kaiming He. Exploring plain vision transformer backbones for object detection. In European conference on computer vision, pages 280–296. Springer, 2022.

[30] Jinhua Liang, Ines Nolasco, Burooj Ghani, Huy Phan, Emmanouil Benetos, and Dan Stowell. Mind the domain gap: a systematic analysis on bioacoustic sound event detection. In 2024 32nd European Signal Processing Conference (EUSIPCO), pages 1257–1261. IEEE, 2024.

[31] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie. Feature pyramid networks for object detection. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2117–2125, 2017.

[32] Louis Mahon, Benjamin Hoffman, Logan James, Maddie Cusimano, Masato Hagiwara, Sarah C Woolley, Felix Effenberger, Sara Keen, Jen-Yu Liu, and Olivier Pietquin. Robust detection of overlapping bioacoustic sound events. arXiv preprint arXiv:2503.02389, 2025.

[33] Alba Márquez-Rodríguez, Miguel Ángel Mohedano-Munoz, Manuel J Marín-Jiménez, Eduardo Santamaría-García, Giulia Bastianelli, Pedro Jordano, and Irene Mendoza. A bird song detector for improving bird identification through deep learning: A case study from doñana. Ecological Informatics, 90:103254, 2025.

[34] Leland McInnes, John Healy, and James Melville. Umap: Uniform manifold approximation and projection for dimension reduction, 2020. URL https://arxiv.org/abs/1802.03426.

[35] Nathan D Merchant, Kurt M Fristrup, Mark P Johnson, Peter L Tyack, Matthew J Witt, Philippe Blondel, and Susan E Parks. Measuring acoustic habitats. Methods in Ecology and Evolution, 6 (3):257–265, 2015.

[36] Ilyass Moummad, Nicolas Farrugia, and Romain Serizel. Self-supervised learning for few-shot bird sound classification. In 2024 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW), pages 600–604. IEEE, 2024.

[37] Thomas Napier, Euijoon Ahn, Slade Allen-Ankins, Lin Schwarzkopf, and Ickjai Lee. Advancements in preprocessing, detection and classification techniques for ecoacoustic data: a comprehensive review for large-scale passive acoustic monitoring. Expert Systems with Applications, 252:124220, 2024.

[38] Nagarajan Natarajan, Inderjit S Dhillon, Pradeep K Ravikumar, and Ambuj Tewari. Learning with noisy labels. Advances in neural information processing systems, 26, 2013.

[39] Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino. Byol for audio: Exploring pre-trained general-purpose audio representations. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 31:137–151, 2022.

[40] Daisuke Niizumi, Daiki Takeuchi, Yasunori Ohishi, Noboru Harada, and Kunio Kashino. Masked spectrogram modeling using masked autoencoders for learning general-purpose audio representation. In HEAR: Holistic Evaluation of Audio Representations, pages 1–24. PMLR, 2022.

[41] Daniel S Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D Cubuk, and Quoc V Le. Specaugment: A simple data augmentation method for automatic speech recognition. arXiv preprint arXiv:1904.08779, 2019.

[42] Sangwook Park, Ashwin Bellur, David K Han, and Mounya Elhilali. Self-training for sound event detection in audio mixtures. In ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 341–345. IEEE, 2021.

[43] Sangwook Park, David K Han, and Mounya Elhilali. Cross-referencing self-training network for sound event detection in audio mixtures. IEEE transactions on multimedia, 25:4573–4585, 2022.

[44] Phuong Pham, Juncheng Li, Joseph Szurley, and Samarjit Das. Eventness: Object detection on spectrograms for temporal localization of audio events. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 2491–2495. IEEE, 2018.

[45] Lukas Rauch, Raphael Schwinger, Moritz Wirth, René Heinrich, Denis Huseljic, Marek Herde, Jonas Lange, Stefan Kahl, Bernhard Sick, Sven Tomforde, et al. Birdset: A large-scale dataset for audio classification in avian bioacoustics. arXiv preprint arXiv:2403.10380, 2024.

[46] Lukas Rauch, René Heinrich, Ilyass Moummad, Alexis Joly, Bernhard Sick, and Christoph Scholz. Can masked autoencoders also listen to birds? arXiv preprint arXiv:2504.12880, 2025.

[47] Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun. Faster r-cnn: Towards real-time object detection with region proposal networks. IEEE transactions on pattern analysis and machine intelligence, 39(6):1137–1149, 2016.

[48] Hamid Rezatofighi, Nathan Tsoi, JunYoung Gwak, Amir Sadeghian, Ian Reid, and Silvio Savarese. Generalized intersection over union: A metric and a loss for bounding box regression. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 658–666, 2019.

[49] David Robinson, Marius Miron, Masato Hagiwara, Benno Weck, Sara Keen, Milad Alizadeh, Gagan Narula, Matthieu Geist, and Olivier Pietquin. Naturelm-audio: An audio-language foundation model for bioacoustics. arXiv preprint arXiv:2411.07186, 2024.

[50] Samuel RP-J Ross, Darren P O’Connell, Jessica L Deichmann, Camille Desjonquères, Amandine Gasc, Jennifer N Phillips, Sarab S Sethi, Connor M Wood, and Zuzana Burivalova. Passive acoustic monitoring provides a fresh perspective on fundamental ecological questions. Functional Ecology, 37(4):959–975, 2023.

[51] Ranjan Sapkota, Rahul Harsha Cheppally, Ajay Sharda, and Manoj Karkee. Yolo26: key architectural enhancements and performance benchmarking for real-time object detection. arXiv preprint arXiv:2509.25164, 2025.

[52] Romain Serizel, Nicolas Turpault, Hamid Eghbal-Zadeh, and Ankit Parag Shah. Large-scale weakly labeled semi-supervised sound event detection in domestic environments. arXiv preprint arXiv:1807.10501, 2018.

[53] Dan Stowell. Computational bioacoustics with deep learning: a review and roadmap. PeerJ, 10: e13152, 2022.

[54] Jérôme Sueur and Almo Farina. Ecoacoustics: the ecological investigation and interpretation of environmental sound. Biosemiotics, 8(3):493–502, 2015.

[55] Larissa Sayuri Moreira Sugai, Thiago Sanna Freire Silva, José Wagner Ribeiro Jr, and Diego Llusia. Terrestrial passive acoustic monitoring: review and perspectives. BioScience, 69(1): 15–25, 2019.

[56] Yuren Sun, Tatiana Midori Maeda, Claudia Solís-Lemus, Daniel Pimentel-Alarcón, and Zuzana Buˇrivalová. Classification of animal sounds in a hyperdiverse rainforest using convolutional neural networks with data augmentation. Ecological Indicators, 145:109621, 2022.

[57] Zhi Tian, Chunhua Shen, Hao Chen, and Tong He. Fcos: Fully convolutional one-stage object detection. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9627–9636, 2019.

[58] Michael Towsey, Birgit Planitz, Alfredo Nantes, Jason Wimmer, and Paul Roe. A toolbox for animal call recognition. Bioacoustics, 21(2):107–125, 2012.

[59] Joseph Turian, Jordie Shier, Humair Raj Khan, Bhiksha Raj, Björn W Schuller, Christian J Steinmetz, Colin Malloy, George Tzanetakis, Gissel Velarde, Kirk McNally, et al. Hear: Holistic evaluation of audio representations. In NeurIPS 2021 Competitions and Demonstrations Track, pages 125–145. PMLR, 2022.

[60] Bart Van Merriënboer, Jenny Hamer, Vincent Dumoulin, Eleni Triantafillou, and Tom Denton. Birds, bats and beyond: Evaluating generalization in bioacoustics models. Frontiers in Bird Science, 3:1369756, 2024.

[61] Bart van Merriënboer, Vincent Dumoulin, Jenny Hamer, Lauren Harrell, Andrea Burns, and Tom Denton. Perch 2.0: The bittern lesson for bioacoustics. arXiv preprint arXiv:2508.04665, 2025.

[62] Satvik Venkatesh, David Moffat, and Eduardo Reck Miranda. You only hear once: a yolo-like algorithm for audio segmentation and sound event detection. Applied Sciences, 12(7):3293, 2022.

[63] Yun Wang, Juncheng Li, and Florian Metze. A comparison of five multiple instance learning pooling functions for sound event detection with weak labeling. In ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 31–35. IEEE, 2019.

[64] Colin Wei, Kendrick Shen, Yining Chen, and Tengyu Ma. Theoretical analysis of self-training with deep networks on unlabeled data. arXiv preprint arXiv:2010.03622, 2020.

[65] Connor M Wood, Jacob Socolar, Stefan Kahl, M Zachariah Peery, Philip Chaon, Kevin Kelly, Robert A Koch, Sarah C Sawyer, and Holger Klinck. A scalable and transferable approach to combining emerging conservation technologies to identify biodiversity change after large disturbances. Journal of Applied Ecology, 61(4):797–808, 2024.

[66] Tianyi Xu, Xuan Ouyang, Binwei Yao, Shoua Xiong, Sara Misurelli, Maichou Lor, and Junjie Hu. Sita: Learning speaker-invariant and tone-aware speech representations for low-resource tonal languages. arXiv preprint arXiv:2601.09050, 2026.

[67] Shifeng Zhang, Cheng Chi, Yongqiang Yao, Zhen Lei, and Stan Z Li. Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 9759–9768, 2020.

## A Limitations and Future Work

Limitations. MAST performs binary “any animal sound” detection and does not distinguish among species or call types, though the retrieval workflow (App. F.4) partially bridges this gap. Self-training is bounded by the seed detector’s quality: systematic false negatives cannot be recovered through pseudo-labeling alone. The pseudo-label confidence threshold is tuned on the in-domain validation split, which may not be optimal under OOD conditions; adaptive or domain-aware thresholding could further improve self-training. Finally, the full pipeline requires substantially more compute than a single supervised detector.

Future work. Extending MAST to multi-class time–frequency detection via species-level annotations or unsupervised clustering of detected events is a natural next step. Incorporating complementary weak signals such as clip-level tags or acoustic indices could help break the pseudo-label quality ceiling. We evaluate on two ecologically distinct domains, but validation on additional biomes such as marine and urban and taxa like marine mammals and amphibians would further establish generality.

## B Self-training Operator and Two-Stage Curriculum

Given an unlabeled chunk S, a detector produces a set of scored candidate boxes

$$
\widehat { B } ( S ) = \left\{ \left( \widehat { b } _ { j } , s _ { j } \right) \right\} _ { j = 1 } ^ { \widehat { N } } , \quad s _ { j } \in [ 0 , 1 ] .
$$

We apply class-agnostic NMS with overlap threshold $\eta ,$ and enforce a validity predicate $v ( \boldsymbol { \hat { b } } )$ , such as minimum time span, minimum frequency span, and boundary constraints. With a confidence policy $q ( \cdot )$ , the pseudo-label operator is

$$
\begin{array} { r } { \Pi _ { q } ( S ) = \left\{ \widehat { b } \in \mathrm { N M S } _ { \eta } ( \widehat { \mathcal { B } } ( S ) ) : s ( \widehat { b } ) \geq q ( S ) \wedge v ( \widehat { b } ) \right\} . } \end{array}
$$

Applying $\Pi _ { q }$ to the unlabeled pool yields a pseudo-labeled dataset

$$
\Pi _ { q } \left( \mathcal { D } _ { u } \right) = \left\{ ( S , \tilde { B } ) : S \in \mathcal { D } _ { u } , \tilde { B } = \Pi _ { q } ( S ) \right\} .
$$

In our experiments we set $q$ on the labeled validation split to balance precision and coverage.   
Remaining noise is suppressed by confidence weighting during training (see Stage 1 below).

We then train student detectors using two stages:

Stage 1: pseudo pretraining. We train on $\Pi _ { q } \left( \mathcal { D } _ { u } \right)$ using the detection loss with the box-level contrastive loss. To reduce the impact of noisy pseudo labels, we weight each pseudo-labeled example by its average pseudo confidence:

$$
\bar { g } ( S ) = \frac { 1 } { | \tilde { B } | } \sum _ { \hat { b } \in \tilde { B } } s ( \hat { b } ) ,
$$

and optimize

$$
{ \mathcal { L } } _ { \mathrm { s t a g e l } } = \mathbb { E } _ { ( S , \tilde { B } ) \sim \Pi _ { q } ( { \mathcal { D } } _ { u } ) } \left[ \bar { g } ( S ) { \mathcal { L } } _ { \mathrm { d e t } } ( S , \tilde { B } ) + \lambda _ { \mathrm { c o n } } { \mathcal { L } } _ { \mathrm { c o n } } ( S , \tilde { B } ) \right]
$$

Intuitively, Stage 1 increases recall and robustness by exposing the detector to diverse acoustic conditions that are absent from $\mathcal { D } _ { \ell }$ , while confidence weighting reduces confirmation bias from uncertain predictions.

Stage 2: expert refinement. We initialize from the Stage 1 student and fine-tune on the expert-labeled set:

$$
{ \mathcal { L } } _ { \mathrm { s t a g e 2 } } = \mathbb { E } _ { ( S , B ) \sim \mathcal { D } _ { \ell } } \left[ { \mathcal { L } } _ { \mathrm { d e t } } \left( S , B \right) + \lambda _ { \mathrm { c o n } } { \mathcal { L } } _ { \mathrm { c o n } } \left( S , B \right) \right] .
$$

Stage 2 re-anchors training to high-quality annotations, mitigating confirmation bias from pseudo labels and improving calibration under domain shift.

We then repeat the cycle of pseudo-labeling → Stage 1 → Stage 2 for 1–2 rounds depending on unlabeled pool size and observed saturation.

## C Theoretical Analysis of Adapter Design

We formalize well-known intuitions about the resolution bottleneck induced by patch-based vision transformers [29, 31], derive the optimality of asymmetric upsampling for our setting, and analyze the efficiency of anisotropic convolutions.

## C.1 Resolution Bottleneck

With patch size $p$ and spectrogram dimensions $T \times F$ , the encoder produces a feature map with strides $s _ { t } = p$ and $s _ { f } = p _ { \mathrm { : } }$ yielding a token grid of size $( T / p ) \times ( \bar { F / p } )$ . A detector operating on this grid must express box boundaries at the granularity of these strides. Prior work has noted that coarse feature-map strides hurt small-object localization [31, 29], motivating multi-scale feature pyramids. Here we make this intuition precise for the spectrogram setting and show that quantization can severely degrade localization quality, particularly along the frequency axis.

Proposition 1 (Quantization-induced IoU degradation). Let $b ^ { * } = [ t _ { 1 } , f _ { 1 } , t _ { 2 } , f _ { 2 } ]$ be a ground-truth box with temporal extent $\delta _ { t } = t _ { 2 } - t _ { 1 }$ and frequency extent $\delta _ { f } = f _ { 2 } - f _ { 1 }$ . Let <sup>ˆ</sup>b denote the tightest enclosing box whose corners lie on a grid with strides $s _ { t }$ and $s _ { f }$ . Then there exists a placement $o f b ^ { * }$ relative to the grid such that:

$$
\mathrm { I o U } ( b ^ { * } , \hat { b } ) \leq \frac { \delta _ { t } \cdot \delta _ { f } } { ( \delta _ { t } + s _ { t } ) ( \delta _ { f } + s _ { f } ) } .
$$

Proof. We construct a worst-case placement as follows. Choose $t _ { 1 } = ( m + 1 ) s _ { t } - \epsilon$ for integer m and small $\epsilon > 0 .$ , so $\lfloor t _ { 1 } / s _ { t } \rfloor = m$ . Write $\delta _ { t } = k _ { t } s _ { t } + r _ { t }$ with $0 \leq r _ { t } < s _ { t } . \mathrm { ~ I f ~ } r _ { t } = 0$ , then $t _ { 2 } = ( m + k _ { t } + 1 ) s _ { t } - \epsilon , \mathrm { s o } \left[ t _ { 2 } / s _ { t } \right] = m + k _ { t } + 1$ and $\hat { \delta } _ { t } = ( k _ { t } + 1 ) s _ { t } = \delta _ { t } + s _ { t } . \mathrm { I f } \ r _ { t } > 0$ (choosing $\epsilon < r _ { t } )$ , then $t _ { 2 } = ( m + k _ { t } + 1 ) s _ { t } + r _ { t } - \epsilon$ with $r _ { t } - \epsilon \in ( 0 , s _ { t } )$ , so $\lceil t _ { 2 } / s _ { t } \rceil = m + k _ { t } + 2$ and $\hat { \delta } _ { t } = ( k _ { t } + 2 ) s _ { t } = \delta _ { t } + 2 s _ { t } - r _ { t } > \delta _ { t } + s _ { t }$ . In both cases $\hat { \delta } _ { t } \geq \delta _ { t } + s _ { t }$ . The same construction applies in frequency. Since $b ^ { * } \subseteq \hat { b } , \operatorname { I o U } ( b ^ { * } , \hat { b } ) = \delta _ { t } \delta _ { f } / ( \hat { \delta } _ { t } \hat { \delta } _ { f } ) \leq \delta _ { t } \delta _ { f } / [ ( \delta _ { t } + s _ { t } ) ( \delta _ { f } + s _ { f } ) ]$ □

Corollary 2 (Narrow-band events can fall below detection threshold). Standard evaluation counts a prediction as a true positive only $i f \mathrm { I o U } \geq 0 . 5$ . With encoder grid strides $s _ { t } = s _ { f } = p = 1 6$ and a narrow-band vocalization with $\delta _ { f } \ \leq \ p$ (frequency extent $\leq 1$ token), the IoU bound from Proposition 1 is monotonically increasing in $\delta _ { f ; }$ , so it is maximized at $\delta _ { f } = p ,$ giving:

$$
\mathrm { I o U } ( b ^ { * } , \hat { b } ) \leq \frac { \delta _ { t } \cdot p } { ( \delta _ { t } + p ) ( p + p ) } = \frac { \delta _ { t } } { 2 ( \delta _ { t } + p ) } < \frac { 1 } { 2 } ,
$$

for all finite $\delta _ { t }$ . By Proposition $^ { l , }$ there exists a placement achieving this bound, hence there exist placements where the tightest enclosing grid-aligned boxfails to reach $I o U \ge 0 . 5 f o r a n y$ narrow-band event, regardless ofits temporal extent.

In our setting, each token spans $\Delta t _ { \mathrm { t o k } } = p \cdot h = 1 6 0$ ms in time, where $h = 1 0$ ms is the STFT hop, and $\Delta f _ { \mathrm { t o k } } = \boldsymbol { p } \cdot \Delta f _ { \mathrm { m e l } } \approx 1$ kHz in frequency. Many animal vocalizations such as bird harmonics, typically with 200–500 Hz bandwidth, and insect stridulations, typically with $< 1 \mathrm { k H z }$ , occupy at most one token in frequency while spanning many tokens in time, falling precisely into the regime of Corollary 2.

## C.2 Asymmetric Upsampling: Theoretical Analysis

Our adapter applies learnable upsampling with asymmetric factors $u _ { t }$ in time and $u _ { f }$ in frequency, reducing the effective strides to $s _ { t } ^ { \prime } = p / u _ { t }$ and $\dot { s _ { f } ^ { \prime } } = p / u _ { f }$ . By the same worst-case analysis as Proposition 1:

$$
\operatorname { I o U } ( b ^ { * } , \hat { b } ) \leq \frac { \delta _ { t } \cdot \delta _ { f } } { \left( \delta _ { t } + p / u _ { t } \right) \left( \delta _ { f } + p / u _ { f } \right) } .
$$

Proposition 3 (Optimal upsampling allocation). Given a fixed upsampling budget $\boldsymbol { u } _ { t } \cdot \boldsymbol { u } _ { f } = \boldsymbol { U }$ (total spatial expansion), the IoU bound in Proposition 1 is maximized when the upsampling factors are allocated proportionally to the stride-to-extent ratios:

$$
\frac { u _ { f } } { u _ { t } } = { \frac { s _ { f } / \delta _ { f } } { s _ { t } / \delta _ { t } } } = \frac { \delta _ { t } } { \delta _ { f } } .
$$

That $i s ,$ the axis with greater relative quantization error should receive more upsampling.

Proof. Since the numerator $\delta _ { t } \delta _ { f }$ is constant, maximizing $\mathrm { I o U } _ { \mathrm { u b } }$ is equivalent to minimizing the denominator $D ( u _ { t } ) = ( \delta _ { t } + p / \bar { u _ { t } } ) ( \delta _ { f } + p u _ { t } / U )$ after substituting $u _ { f } = U / u _ { t }$ . Setting $D ^ { \prime } ( u _ { t } ) \bar { = } 0 $

$$
\frac { - p } { u _ { t } ^ { 2 } } \Big ( \delta _ { f } + \frac { p u _ { t } } { U } \Big ) + \Bigg ( \delta _ { t } + \frac { p } { u _ { t } } \Bigg ) \frac { p } { U } = 0 \quad \implies \quad \frac { \delta _ { f } + p u _ { t } / U } { u _ { t } ^ { 2 } } = \frac { \delta _ { t } + p / u _ { t } } { U } .
$$

Cross-multiplying and expanding: $U \delta _ { f } + p u _ { t } = u _ { t } ^ { 2 } \delta _ { t } + p u _ { t }$ . The $p u _ { t }$ terms cancel, giving $U \delta _ { f } = u _ { t } ^ { 2 } \delta _ { t }$ so $u _ { t } = \sqrt { U \delta _ { f } / \delta _ { t } }$ and $u _ { f } = U / u _ { t } = \sqrt { U \delta _ { t } / \delta _ { f } }$ , hence $u _ { f } / u _ { t } = \delta _ { t } / \delta _ { f }$ . This is a maximum since $D ( u _ { t } )  \infty$ as $u _ { t } \to 0 ^ { + }$ or $u _ { t } \to \infty$ □

For typical rainforest vocalizations with $\delta _ { t } \gg \delta _ { f }$ (temporally extended, frequency-narrow), this predicts $u _ { f } \ > u _ { t }$ , which is exactly our design choice $( u _ { t } = 2 , u _ { f } = 4$ , ratio $u _ { f } / u _ { t } = 2 )$ . With these factors, the adapted feature map has size $1 2 8 \times 3 2$ and effective strides of $\bar { 8 \times 4 }$ pixels. For a narrow-band event with $\delta _ { f } = p = 1 6$ , the IoU upper bound improves from $< 0 . 5$ (Corollary 2) to $\delta _ { t } \cdot 1 6 / [ ( \delta _ { t } + 8 ) \cdot 2 0 ]$ ], which exceeds 0.5 for $\delta _ { t } \geq 1 4$ (events longer than ∼140 ms).

## C.3 Anisotropic Convolution: Separability Analysis

Proposition 4 (Parameter efficiency of separable filtering). Let $\mathcal { F } _ { \mathrm { i s o } } : \mathbb { R } ^ { C \times H \times W } \to \mathbb { R } ^ { C \times H \times W }$ be an isotropic convolution with kernel size $k { \overset { - } { \times } } k ,$ , and $\mathcal { F } _ { \mathrm { s e p } } \bar { = } \mathcal { F } _ { 1 \times k } \circ \mathcal { F } _ { k \times 1 }$ be the composition of two anisotropic convolutions. Then:

(i) Parameter reduction: $| \Theta _ { \mathrm { s e p } } | = 2 C ^ { 2 } k$ versus $| \Theta _ { \mathrm { i s o } } | = C ^ { 2 } k ^ { 2 }$ , a factor of k/2 savings.

(ii) Receptivefield preservation: Both achieve effective receptivefield $k \times k$ on the input.

(iii) Rank constraint: The effective spatial kernel of $\mathcal { F } _ { \mathrm { s e p } }$ for each input–output channel pair has rank at most min $( C , k )$ , whereas a general $k \times k$ filter is unconstrained and can achieve full rank k. The composition processes time and frequency axes sequentially, imposing a structural separability that acts as beneficial inductive biasfor bioacoustic events with approximately independent temporal and spectral structure.

Proof. (i) The $k \times 1$ filter has $C ^ { 2 } k$ parameters and the $1 \times k$ filter has $C ^ { 2 } k$ parameters, totaling $2 C ^ { 2 } \dot { k }$ The ratio is $C ^ { 2 } k ^ { 2 } / ( 2 C ^ { 2 } k ) \stackrel { \bullet } { = } k / 2$ (ii) The $k \times 1$ filter sees k rows and 1 column; the subsequent $1 \times k$ filter sees 1 row and k columns of the intermediate map, so each output pixel depends on a k × k neighborhood. (iii) For a fixed channel pair $( \boldsymbol { c } _ { \mathrm { i n } } , \boldsymbol { c } _ { \mathrm { o u t } } )$ , the effective $k \times k$ spatial kernel is $\begin{array} { r } { K [ i , j ] = \sum _ { c _ { m } = 1 } ^ { C } W _ { f } [ c _ { \mathrm { o u t } } , c _ { m } , j ] \cdot W _ { t } [ c _ { m } , c _ { \mathrm { i n } } , i ] . } \end{array}$ , a sum of C rank-1 outer products. Hence $\operatorname { r a n k } ( K ) \leq \operatorname* { m i n } ( C , k )$ , whereas a general $k \times k$ filter is unconstrained and can achieve full rank k. More fundamentally, $\mathcal { F } _ { \mathrm { s e p } }$ processes time and frequency axes sequentially rather than jointly, so it cannot model non-separable cross-axis interactions. For signals where temporal and spectral structure are approximately independent, as in harmonic animal vocalizations with smooth temporal envelopes, this separability constraint matches the signal structure and suppresses spurious cross-axis correlations. □

This analysis explains why the anisotropic adapter achieves the best F1 among adapter variants (Table 10): the separability constraint matches the physical structure of bioacoustic events, providing effective regularization without sacrificing expressiveness for the target signal class.

## D Theoretical Analysis of Self-Training

Motivated by domain adaptation theory [3] and recent analyses of self-training under distribution shift [26, 64], we provide theoretical justification for MAST’s two-stage self-training curriculum, confidence-weighted pseudo-label training, and the improvement conditions under iterative selftraining. The following results adapt established frameworks to our specific setting of two-stage pseudo-label exploration and expert refinement for time–frequency detection.

## D.1 Self-Training as Distributionally Robust Optimization

Let $P _ { \ell }$ denote the labeled source distribution and $P _ { u }$ denote the unlabeled target distribution, which may include temporal and cross-site domain shift. The standard supervised objective minimizes $\mathbb { E } _ { ( S , B ) \sim P _ { \ell } } [ \mathcal { L } ( f _ { \theta } ( \overset { \cdot } { S } ) , B ) ]$ , but the deployed detector must perform well under $P _ { u }$ . Following the distributional robustness perspective of Ben-David et al. [3], we characterize our two-stage selftraining as approximate DRO that progressively reduces the gap between training and deployment distributions.

Proposition 5 (Self-training reduces distributional gap). Let $f ^ { ( 0 ) }$ be a detector trained on $P _ { \ell } ,$ and let $\hat { P } _ { u } ^ { ( t ) } = \Pi _ { q } ( P _ { u } ; f ^ { ( t ) } )$ denote the pseudo-labeled distribution induced by applying detector $f ^ { ( t ) }$ to unlabeled data with confidence threshold q. Define the effective training distribution at round t ofthe two-stage curriculum as:

$$
P _ { \mathrm { e f f } } ^ { ( t ) } = ( 1 - \alpha ) \hat { P } _ { u } ^ { ( t ) } + \alpha P _ { \ell } ,
$$

where $\alpha \in ( 0 , 1 )$ reflects the relative contribution of Stage 2. Let $d _ { \mathrm { T V } } ( \cdot , \cdot )$ denote total variation distance. If the pseudo-labeling operator has precision $\pi _ { q } = \mathrm { P r } [ \hat { b }$ correct $| \ s ( { \hat { b } } ) \geq q ]$ and the detector achieves coverage $\rho _ { q } = \mathrm { P r } _ { b ^ { * } \sim P _ { u } } [ \exists \hat { b }$ : Io $\lceil ( \hat { b } , b ^ { * } ) \geq 0 . 5 \wedge s ( \hat { b } ) \geq q ]$ , then:

$$
\begin{array} { r } { d _ { \mathrm { T V } } ( P _ { \mathrm { e f f } } ^ { ( t ) } , P _ { u } ) \leq ( 1 - \alpha ) ( 1 - \rho _ { q } \pi _ { q } ) + \alpha d _ { \mathrm { T V } } ( P _ { \ell } , P _ { u } ) . } \end{array}
$$

Proof. Since $\hat { P } _ { u } ^ { ( t ) }$ and $P _ { u }$ share the same marginal on S (pseudo-labeling only modifies the annotation), their TV distance is determined by label disagreement. The pseudo-labeled distribution agrees with $P _ { u }$ on the fraction $\rho _ { q } \pi _ { q }$ of events that are both covered and correctly labeled. The remaining mass $( 1 - \rho _ { q } \pi _ { q } )$ may have incorrect or missing labels, contributing at most $( 1 - \rho _ { q } \pi _ { q } )$ to $d _ { \mathrm { T V } } ( \hat { P } _ { u } ^ { ( t ) } , P _ { u } )$ . By convexity of total variation:

$$
\begin{array} { r l } & { d _ { \mathrm { T V } } ( P _ { \mathrm { e f f } } ^ { ( t ) } , P _ { u } ) \leq ( 1 - \alpha ) d _ { \mathrm { T V } } ( \hat { P } _ { u } ^ { ( t ) } , P _ { u } ) + \alpha d _ { \mathrm { T V } } ( P _ { \ell } , P _ { u } ) } \\ & { \qquad \leq ( 1 - \alpha ) ( 1 - \rho _ { q } \pi _ { q } ) + \alpha d _ { \mathrm { T V } } ( P _ { \ell } , P _ { u } ) . } \end{array}
$$

This result shows that as pseudo-label quality $( \pi _ { q } )$ and coverage $( \rho _ { q } )$ improve across self-training rounds, $P _ { \mathrm { e f f } } ^ { ( t ) }$ converges toward $P _ { u }$ , reducing the effective domain gap. The two-stage design is crucial: Stage 1 improves $\rho _ { q } ,$ the coverage via diverse pseudo-labeled data, while Stage 2 controls $\alpha ,$ the re-anchoring to clean labels prevents drift.

## D.2 Confidence Weighting as Importance-Weighted Risk Minimization

In Stage 1, we weight each pseudo-labeled sample by its average pseudo confidence $\bar { g } ( S ) =$ $\textstyle { \frac { 1 } { | \tilde { B } | } } \sum _ { \hat { b } \in \tilde { B } } s ( \hat { b } )$ . Building on the importance-weighted learning framework for noisy labels [38], we observe that this weighting can be interpreted as importance-weighted empirical risk minimization that provably reduces the effective noise rate.

Proposition 6 (Confidence weighting reduces effective label noise). Let $\epsilon ( s )$ denote the label error probability for a pseudo-label with confidence s, and assume $\epsilon ( s )$ is monotonically decreasing in s (higher confidence ⇒ lower error rate). Given n pseudo-labeled samples $( S _ { i } , \tilde { B } _ { i } ) _ { i = 1 } ^ { n } ,$ , the confidence-weighted empirical risk

$$
\hat { R } _ { w } ( \theta ) = \frac { 1 } { Z } \sum _ { i = 1 } ^ { n } \bar { g } ( S _ { i } ) \mathcal { L } ( f _ { \theta } ( S _ { i } ) , \tilde { B } _ { i } ) , \quad Z = \sum _ { i = 1 } ^ { n } \bar { g } ( S _ { i } ) ,
$$

has effective noise rate

$$
\overline { { \epsilon } } _ { w } = \frac { \sum _ { i } \bar { g } ( S _ { i } ) \epsilon ( \bar { g } ( S _ { i } ) ) } { \sum _ { i } \bar { g } ( S _ { i } ) } \leq \frac { \sum _ { i } \epsilon ( \bar { g } ( S _ { i } ) ) } { n } = \overline { { \epsilon } } _ { \mathrm { u n i f } } ,
$$

where $\bar { \epsilon } _ { \mathrm { u n i f } }$ is the uniform (unweighted) noise rate.

Proof. By the assumption that $\epsilon ( s )$ is decreasing, samples with high $\bar { g }$ (high weight) have low $\epsilon ,$ and vice versa. Setting $a _ { i } = \bar { g } ( S _ { i } )$ and $b _ { i } = \epsilon ( \bar { g } ( \bar { S _ { i } } ) )$ , the monotonicity of ϵ ensures that $a _ { i }$ and $b _ { i }$ are oppositely ordered (i.e., whenever $a _ { i } \leq a _ { j }$ we have $b _ { i } \geq b _ { j } )$ ), which is exactly the condition required by the Chebyshev sum inequality:

$$
n \sum _ { i } a _ { i } b _ { i } \leq \left( \sum _ { i } a _ { i } \right) \left( \sum _ { i } b _ { i } \right) .
$$

Substituting back and dividing both sides by n $\Sigma _ { i } \bar { g } ( S _ { i } )$

$$
\overline { { \epsilon } } _ { w } = \frac { \sum _ { i } \bar { g } ( S _ { i } ) \epsilon ( \bar { g } ( S _ { i } ) ) } { \sum _ { i } \bar { g } ( S _ { i } ) } \leq \frac { \sum _ { i } \epsilon ( \bar { g } ( S _ { i } ) ) } { n } = \overline { { \epsilon } } _ { \mathrm { u n i f } } .
$$

The inequality is strict whenever the confidence scores are non-degenerate (not all identical), meaning confidence weighting always reduces effective noise compared to uniform training on the same pseudo-labeled data. This provides theoretical grounding for the empirical observation that confidence weighting improves OOD generalization in Stage 1.

## D.3 Monotonic Improvement under Iterative Self-Training

Prior work has established convergence guarantees for iterative self-training under various distributional assumptions, including gradual domain shift [26], expansion conditions [64], and mixturemodel structure [13]. We adapt these ideas to our two-stage curriculum and identify three sufficient conditions under which successive rounds produce non-degrading OOD performance.

Proposition 7 (Sufficient condition for monotonic OOD improvement). Let $f ^ { ( t ) }$ denote the detector after round t ofself-training, and let $R _ { \mathrm { o o d } } ( f ) = \mathbb { E } _ { P _ { u } } [ \mathcal { L } ( f ( \bar { S } ) , B ) ]$ denote the OOD risk. Assume the loss L is L-Lipschitz in the model parameters. Suppose:

(i) Improving pseudo-labels: The precision-coverage product satisfies $\rho _ { q } ^ { ( t + 1 ) } \pi _ { q } ^ { ( t + 1 ) } \geq \rho _ { q } ^ { ( t ) } \pi _ { q } ^ { ( t ) }$ and the resulting OOD risk reductionfrom Stage 1 is at least $\Delta ^ { ( t ) } > 0 ;$

(ii) Bounded Stage 2 drift: Stage 2 fine-tuning runs for $T _ { 2 }$ steps with learning rate $\eta _ { 2 }$ and gradient norm bounded by $G ,$ , so the parameter drift satisfies $\| f _ { \mathrm { s t a g e 2 } } ^ { ( t + 1 ) } - f _ { \mathrm { s t a g e 1 } } ^ { ( t + 1 ) } \| \leq T _ { 2 } \eta _ { 2 } G = :$ $\delta _ { 2 } ,$

(iii) Bounded Stage 1forgetting: The gradual unfreezing schedule limits catastrophicforgetting such that $\| f _ { \mathrm { s t a g e 1 } } ^ { ( t + 1 ) } - f ^ { ( t ) } \| _ { \infty } \leq \beta ,$ , where $\beta > 0$ is controlled by the backbone learning rate $\gamma \eta _ { \mathrm { b a s e } }$ and the number ofunfrozen blocks k(e) (Remark 8).

IfLδ<sub>2</sub> ≤ ∆<sup>(t)</sup>, i.e., Stage 2 parameter drift is small enough that Lipschitz-induced risk increase does not exceed the Stage 1 gain, then $R _ { \mathrm { o o d } } \big ( \dot { f } ^ { ( t + 1 ) } \big ) \leq R _ { \mathrm { o o d } } \big ( \dot { f } ^ { ( t ) } \big )$

Proofsketch. By Proposition 5, improving $\rho _ { q } \pi _ { q }$ (condition i) reduces $d _ { \mathrm { T V } } ( P _ { \mathrm { e f f } } ^ { ( t + 1 ) } , P _ { u } )$ . Since d<sub>TV</sub> upper-bounds the H∆H-divergence of Ben-David et al. [3], this tightens their generalization bound $R _ { u } ( f ) \leq R _ { s } ( f ) + d _ { \mathcal { H } \Delta \mathcal { H } } ( P _ { s } , P _ { u } ) + \lambda ,$ , yielding a concrete OOD risk reduction $\Delta ^ { ( t ) } > 0$ proportional to the TV distance decrease: $R _ { \mathrm { o o d } } ( f _ { \mathrm { s t a g e 1 } } ^ { ( t + 1 ) } ) \le R _ { \mathrm { o o d } } ( f ^ { ( t ) } ) - \Delta ^ { ( t ) }$ . Condition (iii) ensures that optimization remains within a β-ball of the previous iterate, so the ERM solution does not escape the basin where the tighter bound applies. For Stage 2, by Lipschitz continuity and condition (ii): $| R _ { \mathrm { o o d } } ( f _ { \mathrm { s t a g e 2 } } ^ { ( t + 1 ) } ) - R _ { \mathrm { o o d } } ( f _ { \mathrm { s t a g e 1 } } ^ { ( t + 1 ) } ) | \leq L \delta _ { 2 }$ . Since $L \delta _ { 2 } \leq \Delta ^ { ( t ) }$ , we have $R _ { \mathrm { o o d } } ( f ^ { ( t + 1 ) } ) \leq$ $R _ { \mathrm { o o d } } ( f _ { \mathrm { s t a g e 1 } } ^ { ( t + 1 ) } ) + L \delta _ { 2 } \leq R _ { \mathrm { o o d } } ( f ^ { ( t ) } ) - \Delta ^ { ( t ) } + L \delta _ { 2 } \leq R _ { \mathrm { o o d } } ( f ^ { ( t ) } )$ □

Empirical verification. Our design enforces these conditions in practice: (i) a better detector at each round produces higher-quality pseudo-labels, verified by increasing pseudo-label F1 across rounds; (ii) Stage 2 uses a low learning rate $( \eta _ { 2 } = 5 { \times } 1 0 ^ { - 5 }$ , backbone at $0 . 0 2 \times )$ for only $T _ { 2 } { = } 4 0$ epochs with gradient clipping at $G { = } 1 . 0$ , bounding parameter drift $\delta _ { 2 } ;$ gradient clipping locally enforces the Lipschitz condition by ensuring $\| \nabla { \mathcal { L } } \| \leq { \bar { G } }$ at every step; additionally, OOD-based model selection provides an operational safeguard against excessive drift; (iii) gradual unfreezing with conservative schedules in Stage 1 (warmup = 20, interval = 10) and component-specific LR multipliers (backbone at $0 . 0 0 5 \times )$ bound parameter drift. See Table 9 for the full hyperparameter specification.

## D.4 Gradual Unfreezing as Regularized Fine-Tuning

The asymmetry between Stage 1 with conservative unfreezing and Stage 2 with aggressive unfreezing has a natural interpretation through the lens of regularization theory.

Remark 8 (Unfreezing schedule controls effective model complexity). Consider a ViT encoder with L transformer blocks. At epoch e, let $k ( e ) \le L$ blocks be unfrozen with learning rate $\eta _ { \mathrm { b b } } = \gamma \eta _ { \mathrm { b a s e } }$ where $\gamma \ll 1$ . The number ofactively trained parameters is $d _ { \mathrm { e f f } } ( e ) = d _ { \mathrm { h e a d } } + d _ { \mathrm { a d a p t e r } } + k ( e ) \cdot d _ { \mathrm { b l o c k } } ,$ and the backbone’s effective update magnitude scales with $\gamma k ( e )$ . Under the conservative Stage 1 schedule where $k ( e )$ increases slowly with large warmup, the modelfirst learns task-specific head and adapter parameters on noisy pseudo-labels without disturbing pretrained representations. Under the aggressive Stage 2 schedule where $k ( e )$ increases rapidly, thefull model adapts to clean expert labels.

This staged complexity control prevents a key failure mode: if the full backbone is unfrozen on noisy pseudo-labeled data, the pretrained representations may overfit to systematic pseudo-label errors, a form of confirmation bias. By keeping the backbone frozen during early Stage 1 training, the detector head first calibrates its decision boundary before the backbone features are permitted to drift.

The component-specific learning rates further partition the optimization landscape: backbone blocks receive 0.005× the base rate in Stage 1, to preserve general audio features, versus $0 . 0 5 \times$ in Stage 2, to allow fine-grained adaptation to the expert-labeled distribution. This 10× ratio between stages reflects the relative trustworthiness of pseudo vs. expert supervision.

## E Detailed Experimental Setup

## E.1 Data Processing

Audio preprocessing. All raw audio is resampled to 16 kHz mono. We segment each recording into 10.24 s chunks with 50% overlap using 5.12 s hop, yielding maximum coverage while ensuring every segment fits the MAST input size. Chunks shorter than 50% of the target duration are discarded, and those between 50–100% are zero-padded. Before feature extraction, each chunk is mean-centered to remove DC offset.

Spectrogram extraction. We compute log-mel spectrograms using Kaldi-compatible filterbanks via torchaudio.compliance.kaldi.fbank with the following parameters: 25 ms Hanning window, 10 ms frame shift, 128 mel bins, HTK-compatible mel scale, no energy feature, and no dither. This produces a 1024 × 128 (time × frequency) feature matrix per chunk, with frequency range [20, 8000] Hz. These settings are identical to those used by AudioMAE [18], ensuring compatibility with the pretrained encoder. We apply dataset-wide z-score normalization during training: $( S -$ $\mu ) / ( \sigma + \epsilon )$ , where $\mu , \sigma$ are computed once over $\mathcal { D } _ { \ell } \cup \mathcal { D } _ { u }$ and reused for all training and evaluation. For BIRDeep, normalization statistics are computed separately over the combined BIRDeep + BirdSet XCM pool.

Bounding box coordinate conversion. Expert annotations are provided in (Hz, seconds). We convert to spectrogram coordinates as follows. Time: $t = \left\lfloor \sec / 0 . 0 1 0 \right\rfloor$ (10 ms frame shift). Frequency: we apply the HTK mel transform $m ( \nu ) = 2 5 9 5 \log _ { 1 0 } \bar { ( } 1 + \nu / 7 0 0 )$ , distribute 128 bins linearly in mel space, and map each annotation boundary to the nearest bin center. We clip to valid bounds and enforce positive area to ensure $t ^ { ( 2 ) } > t ^ { ( 1 ) } , f ^ { ( 2 ) } > f ^ { ( 1 ) }$ , with minimum 1 bin and 1 frame. Because the mel filterbank covers 20–8000 Hz, annotations with frequency content entirely above 8 kHz are effectively excluded, and those partially above 8 kHz are clipped to the representable range. For chunked data, annotation times are adjusted relative to the chunk start; annotations spanning chunk boundaries are included in each overlapping chunk and clipped accordingly, with events shorter than 100 ms after clipping discarded.

Table 5: Dataset statistics for both domains. Summary of labeled, unlabeled, and OOD evaluation data after preprocessing into 10.24 s log-mel spectrogram chunks. See Table 6 for BIRDeep site-tohabitat mapping.
<table><tr><td>Split</td><td colspan="3">Rainforest # Chunks # Boxes Avg./Chunk</td><td># Chunks # Boxes</td><td></td><td>BIRDeep Avg./Chunk</td><td>Sites</td></tr><tr><td>Labeled Train</td><td>11,726</td><td>74,764</td><td>6.38</td><td>3,558</td><td>4,362</td><td>1.23</td><td>AM1,2,3,8,10,11,16</td></tr><tr><td>Labeled Val</td><td>2,513</td><td>15,598</td><td>6.21</td><td>762</td><td>902</td><td>1.18</td><td>same</td></tr><tr><td>Labeled Test</td><td>2,513</td><td>16,235</td><td>6.46</td><td>762</td><td>990</td><td>1.30</td><td>same</td></tr><tr><td>Temporal-OOD</td><td>286</td><td>3,400</td><td>11.89</td><td>671</td><td>1,716</td><td>2.56</td><td>AM4, AM8</td></tr><tr><td>Site-OOD</td><td>550</td><td>6,492</td><td>11.80</td><td>1,309</td><td>1,613</td><td>1.23</td><td>AM4, AM15</td></tr><tr><td>Unlabeled</td><td>481,245</td><td></td><td></td><td>7,051</td><td></td><td></td><td>All 9</td></tr><tr><td>+ External pretrain</td><td></td><td></td><td></td><td>900,000</td><td>一 一</td><td></td><td></td></tr></table>

Patch tokenization. The encoder uses non-overlapping $1 6 \times 1 6$ patches, producing a $6 4 \times 8$ token grid (time × frequency). We reshape tokens to a feature map $\boldsymbol { E } ^ { \dot { \mathbf { \Upsilon } } } \in \mathbb { R } ^ { C \times { \dot { 6 4 } } \times 8 }$ . The detector’s FPN operates at multiple pyramid levels with strides $\left\{ s _ { k } \right\}$ ; box coordinates are converted to each level by dividing corners by the level stride. During augmentation, boxes are transformed jointly with the spectrogram and clipped to valid bounds.

## E.2 Dataset Statistics and OOD Construction

Table 5 summarizes the labeled, unlabeled, and OOD evaluation data for both domains after preprocessing.

Rainforest domain. Our recordings were collected from 2017 to 2019 from 15 sites in East Kalimantan, Indonesia. The labeled dataset contains approximately 24 hours of annotated audio (23.93 h), while the unlabeled pool contains approximately 661.7 hours of recordings in total, corresponding to about 27.6 days of audio and 481,245 spectrogram chunks used for masked audio pretraining. The labeled set contains 16,752 spectrogram chunks and 106,597 annotated time–frequency boxes in total, with 13,170 non-empty chunks (78.6%). Under the in-domain split, the training, validation, and test partitions contain 11,726, 2,513, and 2,513 chunks, respectively, with similar average box density across splits. For out-of-distribution evaluation, we use two annotated settings: a temporal-OOD set from sites 13A and 13B containing 286 chunks and 3,400 boxes, and an unseen-site OOD set containing 550 chunks and 6,492 boxes.

Bird domain. The BIRDeep dataset [33] contains 641 recordings from 9 sites across 4 habitat types in Doñana National Park, Spain. See Table 6 for site-to-habitat mapping. Expert annotations provide time–frequency bounding boxes for 38 bird species. We process all audio identically to the rainforest domain. After chunking, we obtain 6,633 spectrogram chunks with 6,254 bounding boxes. We use 7 sites spanning all 4 habitats for a 70/15/15 train/val/test split, corresponding to 3,558/762/762 chunks. For site-OOD evaluation, we hold out AM4 (low shrubland) and AM15 (marshland), entirely unseen during training, producing 1,309 chunks with 1,613 boxes. For temporal-OOD evaluation, we hold out two recording dates, producing 671 chunks with 1,716 boxes. There is partial overlap between the two OOD sets as AM4 appears in both, but each tests a distinct shift type. For masked pretraining, we combine unlabeled BIRDeep recordings that have 7,051 chunks with BirdSet XCM [45] that have 900k chunks and compute normalization statistics on this combined pool.

Both domains share the same preprocessing pipeline and evaluation protocol, enabling direct crossdomain comparison.

## E.3 Adapter Designs.

We study three adapter variants that bridge the pretrained ViT-B encoder (C=768, spatial 64×8) to the FPN-based detection head.

Table 6: BIRDeep recording sites and habitat types. All 9 autonomous monitoring sites in Doñana National Park, Spain. Sites AM4 and AM15 are held out for site-OOD evaluation; all others are used for labeled training/validation/test.
<table><tr><td>Recorder</td><td>Place Name</td><td>Habitat</td><td>Split</td></tr><tr><td>AM1</td><td>Monteblanco</td><td>Low shrubland</td><td>Train</td></tr><tr><td>AM2</td><td>Sabinar</td><td>High shrubland</td><td>Train</td></tr><tr><td>AM3</td><td>Ojillo</td><td>High shrubland</td><td>Train</td></tr><tr><td>AM4</td><td>Pozo Sta Olalla</td><td>Low shrubland</td><td>Site-OOD</td></tr><tr><td>AM8</td><td>Torre Palacio</td><td>Ecotone</td><td>Train</td></tr><tr><td>AM10</td><td>Pajarera</td><td>Ecotone</td><td>Train</td></tr><tr><td>AM11</td><td>Caño Martinazo</td><td>Ecotone</td><td>Train</td></tr><tr><td>AM15</td><td>Cancela Millán</td><td>Marshland</td><td>Site-OOD</td></tr><tr><td>AM16</td><td>Juncabalejo</td><td>Marshland</td><td>Train</td></tr></table>

(i) Conv adapter: Three 3×3 conv layers (768→512→256→256) with ReLU, preserving spatial resolution at 64×8. Restores short-range locality lost by global ViT attention but cannot resolve sub-patch events.

(ii) Anisotropic adapter: Stem conv (768→512), residual block, separable 3×1 / 1×3 convolutions (Proposition 4), channel reduction (512→256), second residual block, and nearestneighbor frequency upsampling (8→32). Operates at 64×32.

(iii) Upsampling adapter (default): 1×1 channel reduction (768→256), residual block at base resolution, learnable transposed-conv upsampling (×2 time, ×4 frequency), separable 3×1 / 1×3 anisotropic refinement at high resolution (Proposition 4), and a final 3×3 residual refinement block. Operates at 128×32 with effective strides 8×4 (Proposition 3).

All adapters output 256 channels and add 2–4 M parameters, keeping >85% of capacity in the frozen pretrained encoder.

## E.4 Architecture Details

Neck and fusion. Starting from the adapter output (P3), we build P4/P5 via stride-2 3 × 3 convs. We use lateral 1 × 1 projections and top-down fusion with non-negative learnable weights normalized by their sum, and apply lightweight smoothing blocks (mixing $3 \times 3 , 3 \times 1 , 1 \times 3 \ \times$ ) to preserve narrow-band detail while expanding temporal context.

Detector heads. We use FCOS, predicting per-location classification, centerness, and box regression with center sampling on the spectrogram lattice. We also use ATSS assignment [67], Quality Focal Loss (QFL) [28] for classification, and GIoU [48] for regression.

## E.5 Optimization details.

For masked-audio pretraining, we use AdamW with mixed-precision training and gradient clipping. The pretrained MAST encoder is initialized from the checkpoint pretrained on AudioSet [14] and optimized on unlabeled rainforest spectrograms using a masked reconstruction objective. Table 7 summarizes the main pretraining hyperparameters.

For MAST detector fine-tuning, we also use AdamW together with a warmup–cosine learning-rate schedule. The detector is initialized from the best masked-audio pretraining checkpoint and trained for 200 epochs on the labeled spectrogram data. For MAST detectors, we use a smaller effective learning rate for the pretrained backbone and larger learning rates for newly introduced detector components. We train with mixed moderate spectrogram augmentation, exponential moving average, and gradient clipping. Table 8 summarizes the main fine-tuning hyperparameters used for the MAST model.

For staged self-training, we train the model in two phases with different optimization regimes. Stage 1 starts from the best supervised checkpoint and trains on large pseudo-labeled unlabeled data using confidence weighting, moderate augmentation, EMA, contrastive learning, and conservative gradual unfreezing. Its goal is to expand coverage under distribution shift while limiting drift from noisy pseudo supervision. Stage 2 then initializes from the best Stage 1 checkpoint and fine-tunes on clean expert-labeled data with a lower base learning rate, slightly weaker augmentation, and a more aggressive gradual-unfreezing schedule. This second phase re-anchors the detector to high-quality annotations while preserving the broader domain exposure acquired in Stage 1. In both stages, we use AdamW, warmup–cosine scheduling, and gradient clipping. Table 9 summarizes the main hyperparameters.

Table 7: Masked Audio pretraining hyperparameters. Main optimization settings used for maskedaudio pretraining on unlabeled rainforest mel-spectrograms.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Model</td><td>mae_vit_base_patch16</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>AdamW betas</td><td>(0.9, 0.95)</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Training epochs</td><td>400</td></tr><tr><td>Base learning rate</td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Minimum learning rate</td><td> $5 \times 1 0 ^ { - 7 }$ </td></tr><tr><td>Warmup epochs</td><td>20</td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Mask ratio</td><td>0.75</td></tr><tr><td>Time masking probability</td><td>0.6</td></tr><tr><td>Frequency masking probability</td><td>0.3</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Mixed precision</td><td>AMP</td></tr></table>

Table 8: Detector fine-tuning hyperparameters. Main optimization settings used for MAST detector training.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Backbone initialization</td><td>Best masked-audio pretraining checkpoint</td></tr><tr><td>Adapter type</td><td>Upsampling</td></tr><tr><td>Neck</td><td>FPN P2-P5</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Training epochs</td><td>200</td></tr><tr><td>Base learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Backbone LR multiplier</td><td>0.1</td></tr><tr><td>Head LR multiplier</td><td>1.0</td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Learning-rate schedule</td><td>Warmup + cosine decay</td></tr><tr><td>EMA decay</td><td>0.9995</td></tr><tr><td>Augmentation probability</td><td>0.9</td></tr><tr><td>Contrastive weight</td><td>0.5</td></tr><tr><td>Contrastive temperature</td><td>0.1</td></tr><tr><td>Contrastive feature dimension</td><td>256</td></tr><tr><td>Contrastive projection dimension</td><td>128</td></tr><tr><td>Contrastive negatives per image</td><td>32</td></tr></table>

Compute resources. All experiments were conducted on a single NVIDIA A100 GPU (80 GB).

Pseudo-label confidence threshold. The threshold $q = 0 . 1 0$ reflects the low-confidence regime typical of bioacoustic detectors, where true positive scores concentrate well below those seen in natural image detection due to faint, distant, or overlapping animal calls. At this operating point the threshold retains the majority of true positives while filtering out the lowest-scoring false alarms. Remaining label noise is further suppressed by confidence weighting, which down-weights uncertain pseudo-labels in proportion to their average detection score (Section 3, Proposition 6).

Table 9: Staged self-training hyperparameters. Main optimization settings for Stage 1 pseudolabeled exploration and Stage 2 expert-labeled refinement.
<table><tr><td>Hyperparameter</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Training data</td><td>Pseudo-labeled unlabeled data</td><td>Expert-labeled data</td></tr><tr><td>Initialization</td><td>Best supervised checkpoint</td><td>Best Stage 1 checkpoint</td></tr><tr><td>Adapter / neck</td><td>Upsampling + FPN P2-P5</td><td>Upsampling + FPN P2-P5</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Batch size</td><td>64</td><td>64</td></tr><tr><td>Training epochs</td><td>80</td><td>40</td></tr><tr><td>Base learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $5 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Gradient clipping</td><td>1.0</td><td>1.0</td></tr><tr><td>Scheduler</td><td>Warmup + cosine decay</td><td>Warmup + cosine decay</td></tr><tr><td>Warmup epochs</td><td>8</td><td>4</td></tr><tr><td>Backbone LR multiplier</td><td>0.1</td><td>0.02</td></tr><tr><td>Adapter LR multiplier</td><td>1.0</td><td>1.0</td></tr><tr><td>Head LR multiplier</td><td>1.0</td><td>1.0</td></tr><tr><td>Unfreezing warmup</td><td>20 epochs</td><td>3 epochs</td></tr><tr><td>Unfreezing interval</td><td>Every 10 epochs</td><td>Every 6 epochs</td></tr><tr><td>Blocks per stage</td><td>3</td><td>6</td></tr><tr><td>Unfrozen backbone LR multiplier</td><td>0.005</td><td>0.05</td></tr><tr><td>Confidence weighting</td><td>Enabled</td><td>Not Enabled</td></tr><tr><td>Pseudo-label confidence threshold</td><td>0.10</td><td></td></tr><tr><td>Data augmentation prob.</td><td>0.9</td><td>0.8</td></tr><tr><td>EMA decay</td><td>0.9995</td><td>0.9995</td></tr><tr><td>Contrastive loss weight</td><td>0.3</td><td>0.2</td></tr><tr><td>Contrastive temperature</td><td>0.1</td><td>0.1</td></tr><tr><td>Contrastive feature / proj. dim</td><td>256 / 128</td><td>256 / 128</td></tr></table>

## E.6 Augmentations

During supervised detector training, we apply moderate spectrogram augmentations with boxconsistent transformations. All augmentations operate on 128 × 1024 log-mel spectrogram chunk and transform the associated time–frequency boxes jointly with the input. After each geometric augmentation, boxes are clipped to valid spectrogram boundaries and filtered if they become too small to remain meaningful training targets.

Our augmentation pipeline includes the following components:

• Time shift. We randomly shift the spectrogram along the time axis with zero/min-value padding, and shift all box time coordinates accordingly.

• Frequency shift. We randomly shift the spectrogram along the mel-frequency axis and apply the same offset to the box frequency coordinates.

• Time warp. We apply mild temporal warping by rescaling the spectrogram along the time axis and mapping box boundaries through the same transformation.

• Frequency masking. We mask a random frequency band in the style of SpecAugment [41]. Boxes with heavy overlap with the masked band are removed.

• Time masking. We mask a random temporal segment and similarly discard boxes that are largely occluded by the mask.

• Gaussian noise and volume scaling. We apply small additive Gaussian noise and moderate global intensity scaling to improve robustness to recording and gain variation.

• Background mixing. We optionally mix a spectrogram with background-noise samples drawn from a curated noise bank from our training data containing common environmental and anthropogenic sounds such as rain, thunder, aircraft, chainsaw, and vehicle noise, which improves robustness to realistic acoustic interference.

Table 10: Adapter design ablation on cross-site OOD detection on rainforest domain. All variants use the same MAST detector and training setup; only the adapter module is changed. Higher is better.
<table><tr><td>Adapter Variant</td><td>Precision</td><td>Recall</td><td>F1</td><td>mAP</td><td>Mean IoU</td></tr><tr><td>Conv adapter</td><td>0.5928</td><td>0.3517</td><td>0.4415</td><td>0.3004</td><td>0.6644</td></tr><tr><td>Anisotropic adapter</td><td>0.6432</td><td>0.3535</td><td>0.4563</td><td>0.2866</td><td>0.6466</td></tr><tr><td>Upsampling adapter (Main)</td><td>0.7562</td><td>0.3169</td><td>0.4466</td><td>0.2573</td><td>0.6824</td></tr></table>

![](images/54f3d8b6e61eb72e0ad5b661440909eaae822bb34282b6476f1e0a66b78e5b05.jpg)  
(a) Colored by recording site.

![](images/89d824adc5f94a97b7db19698347aedfb5028ab56c1fc27052aa27f2490ce889.jpg)  
(b) Colored by hour of day.

![](images/75fa580e49ff47585f647645b110e4f6cf320774e5df772cdb67f187ece3ab77.jpg)  
(c) Colored by annotated event count.  
Figure 5: UMAP visualization of pretrained encoder embeddings before detector fine-tuning on rainforest domain. Each point is one mel-spectrogram chunk projected from the pretrained representation space into two dimensions. (a) Coloring by recording site reveals spatial clustering, indicating that the encoder captures site-specific acoustic signatures. (b) Coloring by hour of day shows a diel gradient, reflecting systematic changes in background noise and species activity across the 24-hour cycle. (c) Coloring by annotated event count shows that chunks with more animal-sound activity occupy partially distinct regions from background-dominated chunks, suggesting that maskedaudio pretraining captures biologically relevant variation before any supervised training. Together, these patterns provide qualitative evidence of both the usefulness of domain-matched pretraining and the presence of temporal and cross-site distribution shift.

• Mixup and CutMix. We additionally implemented Mixup and CutMix for spectrograms by combining two training examples and merging or clipping their boxes accordingly. These augmentations are designed to simulate overlapping events and partial occlusion.

To avoid destabilizing detection, we use only mild transformation strengths and sample a small random subset of augmentations for each training example. This design preserves biologically meaningful event structure while improving robustness to temporal shifts, frequency variation, and background-condition changes.

## E.7 Evaluation Details

We evaluate all models on binary time–frequency detection of “any animal sound” using five metrics: precision, recall, F1 score, mean average precision (mAP), and mean intersection-over-union (mIoU). A prediction is counted as a true positive if it overlaps a ground-truth box with IoU ≥ 0.5. Remaining predictions are false positives and unmatched ground-truth boxes are false negatives. We apply class-agnostic NMS to detector outputs with fixed overlap thresholds and confidence thresholds tuned on the in-domain validation set, and report metrics aggregated over all chunks in each test split. OOD split construction is described in Sec. E.2. Unless otherwise stated, all numbers collapse all sonotypes into a single foreground class.

Table 11: Per-round self-training results on BIRDeep. Round 0 is the supervised MAST detector before self-training. Higher is better.
<table><tr><td>Setting</td><td>Round</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>mAP</td><td>mIoU</td></tr><tr><td rowspan="3">In-domain</td><td>Round 0 (no ST)</td><td>0.7824</td><td>0.7010</td><td>0.7395</td><td>0.6011</td><td>0.6822</td></tr><tr><td>Round 1</td><td>0.7054</td><td>0.8248</td><td>0.7604</td><td>0.6870</td><td>0.7080</td></tr><tr><td>Round 2</td><td>0.6709</td><td>0.9061</td><td>0.7709</td><td>0.7783</td><td>0.7339</td></tr><tr><td rowspan="3">Temporal OOD</td><td>Round 0 (no ST)</td><td>0.7618</td><td>0.4808</td><td>0.5895</td><td>0.3838</td><td>0.6083</td></tr><tr><td>Round 1</td><td>0.7253</td><td>0.5780</td><td>0.6434</td><td>0.3840</td><td>0.6020</td></tr><tr><td>Round 2</td><td>0.6755</td><td>0.6247</td><td>0.6491</td><td>0.4280</td><td>0.5903</td></tr><tr><td rowspan="3">Cross-site OOD</td><td>Round 0 (no ST)</td><td>0.6476</td><td>0.5059</td><td>0.5680</td><td>0.3803</td><td>0.6001</td></tr><tr><td>Round 1</td><td>0.6893</td><td>0.5914</td><td>0.6367</td><td>0.3953</td><td>0.5963</td></tr><tr><td>Round 2</td><td>0.6522</td><td>0.6438</td><td>0.6434</td><td>0.4414</td><td>0.5943</td></tr></table>

## F Additional Experimental Results

## F.1 Per-Round Self-Training Results on BIRDeep

Table 11 reports per-round self-training results on BIRDeep across in-domain, temporal-OOD, and cross-site-OOD evaluation. Self-training produces large mAP gains across all settings: in-domain mAP rises from 0.601 at Round 0 to 0.778 at Round 2, a +0.18 absolute improvement that surpasses all baselines including FCOS at 0.665. Under cross-site shift, mAP improves from 0.380 to 0.441, and under temporal shift from 0.384 to 0.428. F1 also improves consistently across OOD settings, reaching 0.643 cross-site and 0.649 temporal at Round 2. These trends mirror the monotonic OOD gains observed on the rainforest domain (Figure 4), confirming that the two-stage self-training curriculum generalizes across ecologically distinct domains.

## F.2 Additional Adapter Design Ablation.

Table 10 compares adapter designs while holding the backbone, training recipe, and detection head fixed. All three adapters enable strong OOD performance, but they emphasize different operating points. The anisotropic adapter achieves the best F1 by slightly improving recall while keeping precision competitive, which aligns with the common structure of animal calls being temporally extended and frequency-concentrated. The convolutional adapter achieves the strongest mAP, suggesting that a simple locality-restoring module can already improve ranking across diverse OOD backgrounds. We use the upsampling adapter in our main result because it yields the highest precision and mean IoU, producing more confident and spatially accurate detections, properties that directly benefit downstream self-training, where high-precision pseudo-labels reduce confirmation bias. Overall, these results show that adapter choice primarily controls the precision–recall and box-quality trade-off, while the combination of in-domain pretraining and the detection stack remains the dominant driver of OOD robustness.

## F.3 Representation Analysis of the Pretrained Encoder

To better understand what the pretrained encoder captures before detector fine-tuning, we visualize spectrogram-level embeddings using UMAP [34]. For each spectrogram chunk, we extract the pretrained encoder representation, project it into two dimensions, and color the same embedding space by three attributes: recording site, hour of day, and annotated event count.

Figure 5 provides two qualitative insights. First, the visible gradient with respect to annotated event count suggests that masked-audio pretraining captures variation related to animal-sound activity, even before detector fine-tuning. Second, the organization by site and hour of day indicates that the learned representation also reflects domain-dependent structure associated with habitat, diel cycle, and background acoustics. These patterns qualitatively support both the usefulness of masked-audio pretraining for downstream detection and the presence of temporal and cross-site distribution shift in rainforest soundscapes. We emphasize that UMAP is only a qualitative visualization tool rather than a quantitative measure of separability, but it offers an intuitive view of why both strong pretraining and domain shift aware adaptation are important in this setting.

![](images/377ff19494a389223e6f6ae71181e48311fc1d2ef369d900b918b689f734c165.jpg)  
Figure 6: Box-level embedding space induced by MAST detections on rainforest domain. Each point corresponds to a ground-truth or predicted time–frequency box, embedded by pooling its corresponding region from the pretrained encoder feature map and projected with UMAP. Boxes with similar acoustic structure form partially coherent clusters, and high-quality predictions tend to lie near ground-truth embeddings of the same or related sonotypes. This suggests a practical downstream use of MAST: detected sound events can be retrieved against annotated examples for tentative zero-shot classification, while acoustically distinct clusters can be surfaced to experts as candidate novel sounds. Colors denote sonotypes; circles correspond to predicted boxes and squares correspond to ground-truth boxes.

Table 12: Retrieval-based sonotype classification from box embeddings on rainforest domain, with 131 sonotypes. We evaluate nearest-neighbor retrieval against annotated training boxes in the learned embedding space. Oracle-box classification uses only ground-truth boxes, while detected-box retrieval uses predicted boxes matched to ground truth at $\mathrm { I o U } \geq 0 . 5$ . Higher is better for Top-1, Top-5; lower is better for mean rank.
<table><tr><td>Setting</td><td># Queries</td><td># Labels</td><td>Top-1</td><td>Top-5</td><td>Mean Rank</td></tr><tr><td>Oracle-box classification (GT boxes)</td><td>14,986</td><td>131</td><td>0.723</td><td>0.902</td><td>5.76</td></tr><tr><td>Detected-box retrieval (matched preds)</td><td>8,414</td><td>104</td><td>0.961</td><td>0.982</td><td>3.22</td></tr></table>

## F.4 Application: Box-Level Embeddings for Retrieval and Zero-Shot Classification

An additional advantage of MAST box-level localization is that each detected sound event can be embedded and compared directly in a shared representation space. After detection, we extract box-level features from the pretrained encoder by pooling the corresponding time–frequency region from the feature map, yielding one embedding per detected event. This enables a retrieval-style downstream workflow in which detected boxes can be matched to annotated training boxes, grouped by acoustic similarity, or provide detections to experts as candidate novel sounds.

Figure 6 illustrates this application qualitatively. Ground-truth and predicted boxes form partially structured clusters in the learned embedding space, and predictions with high overlap to ground truth tend to lie near embeddings of the same or related sonotypes. To quantify this effect, we also evaluate retrieval-based sonotype classification in two settings: (i) oracle-box classification, which uses only ground-truth boxes to test whether the embedding space discriminates sonotypes, and (ii) detected-box retrieval, which uses predicted boxes matched to ground truth at IoU $\geq 0 . 5$ and assigns each matched prediction the sonotype of its best-matching ground-truth box. In both cases, we retrieve nearest neighbors from the annotated training set and report Top-1, Top-5, and mean rank.

![](images/6cb1ec93945b9c5e6813903ebf1bb4a92cfef4c14f132684618855b1b70fe7a5.jpg)  
Figure 7: Label-efficiency study under different sampling strategies on rainforest domain with FCOS. We vary the number of labeled training chunks (x-axis) and report F1 on the held-out test set (y-axis). Time-balanced sampling across the 24-hour diel cycle consistently outperforms sequential labeling at low budgets, and remains slightly better than random sampling for most budgets. As the label budget increases, the gap narrows and all strategies converge to F1 ≈ 0.82 at N=15000.

As shown in Table 12, the learned box embeddings support strong retrieval-based classification. Oracle-box classification achieves 72.3% Top-1 and 90.2% Top-5 accuracy over 131 sonotypes, indicating that the embedding space already captures substantial sonotype structure. Retrieval on matched predicted boxes is even stronger with 96.1% Top-1 and 98.2% Top-5, reflecting that highquality detections tend to align with acoustically prototypical events. In hyperdiverse and acoustically understudied environments, such a workflow is practically useful: detected sounds can be assigned tentative labels through retrieval against known examples, while acoustically distinct clusters can be prioritized for expert review as potentially novel or rare events.

## F.5 Label Budget and Sampling Strategy

We study how label allocation affects performance under a fixed annotation budget. Starting from the full labeled training split, we subsample N chunks for $N \in \{ 2 5 0 , 5 0 0 , 1 0 0 0 , . . . \}$ to simulate different labeling budgets. For each N, we compare three sampling strategies:

• Random. Uniformly sample N chunks from the training split without replacement.

• Time-balanced. Group chunks by their hour-of-day and sample approximately N/24 chunks per hour, ensuring coverage of both day and night.

• Sequential. Take the first N chunks in temporal order, mimicking a naive labeling strategy where annotators start at the beginning and stop once a budget is exhausted.

For each pair we train a FCOS detector from scratch with identical optimization and augmentation settings and evaluate on the same in-domain splits. Figure 7 reports F1 across budgets. At low budgets, the sampling policy is a first-order factor: sequential labeling consistently underperforms, while time-balanced sampling yields the strongest results, achieving F1 of 0.669 vs. 0.539, a +0.130 F1 gain at $N = 1 0 0 0$ . This pattern highlights the impact of temporal distribution shift even within a single site at a single day: labels drawn from one contiguous time range provide limited coverage of acoustic conditions, whereas time-balanced sampling improves robustness by exposing the model to broader diel variation. As the labeling budget increases, the gap between strategies narrows. Practically, this suggests that when annotation budgets are limited, it is better to spread labeling effort across times of day than to annotate one continuous stretch of audio. This finding is especially relevant because sequential labeling is the most natural default in practice: annotators typically open a recording and label forward in chronological order until the budget is exhausted, inadvertently concentrating all labels within a narrow temporal window.

![](images/80a72836778a72ffeb647792d0474ec6bded1c31d72cbdd6b9115880207d42b6.jpg)  
Figure 8: Qualitative detection comparison on two randomly sampled cross-site OOD chunks on rainforest domain. Ground-truth boxes are shown in green solid lines; predicted boxes in blue dashed lines. Baseline detectors produce numerous false positives or miss most events entirely, while MAST localizes the majority of sound events with fewer spurious detections.

## F.6 Qualitative Detection Comparison

To provide visual intuition for the quantitative gaps reported in Table 2, we randomly sample two 10.24 s chunks from the cross-site OOD test set and visualize the predictions of every method side by side as in Figure 8.

Several patterns are apparent. Faster R-CNN fails almost entirely under cross-site shift, producing few or no predictions and yielding near-zero recall. FCOS and DETR exhibit the opposite failure mode: both generate numerous spurious predictions scattered across the spectrogram, resulting in very low precision despite moderate recall, suggesting overfitting to the spectral statistics of the labeled site. AudioMAE shows partial improvement but still misses many events. In contrast, MAST predictions closely track the ground-truth boxes with markedly fewer false positives, consistent with its substantially higher OOD F1 in Table 2.