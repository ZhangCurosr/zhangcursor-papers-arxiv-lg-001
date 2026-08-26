# A Multimodal Foundation Model for Longitudinal Patient Representation and Scalable Insight Generation in Oncology

Eugene Vorontsov<sup>\*</sup>, Yi Kan Wang<sup>\*</sup>, Alican Bozkurt<sup>\*</sup>, Adam Casson<sup>\*</sup>, Ludmila   
Tydlitátová<sup>\*</sup>, Michal Zelechowski<sup>\*</sup>, Ezra E. W. Cohen, Jyoti D. Patel, Max Banaszak,   
Caitlin McWilliams, Shane Colley, Kate Sasser, Ryan Fukushima, Eric Lefkofsky, Razik Yousfi, Siqi Liu<sup>\*,†</sup>

Tempus AI, Inc., Chicago, IL, USA

<sup>\*</sup>These authors contributed equally to this work.   
<sup>†</sup>Corresponding author.

## Abstract

Precision oncology necessitates a longitudinal model of patient state that captures cancer evolution and treatment over time, integrating multimodal observations. We introduce the oFM, a foundation model developed on a real-world oncology cohort of 1.67 million cancer patients that integrates clinical trajectories with DNA, RNA, and H&E pathology. Patient-level partitions were reserved for training, validation, and testing, with over one million patients used for training. The oFM encodes daily clinical and molecular episodes and, along with pathology images, integrates them over time to produce a patient state embedding. We evaluate frozen oFM embeddings against expertcurated clinical and molecular baseline features. In prognostic benchmarks, the oFM improved AUC for treatment response, progression-free survival, and overall survival (0.774 vs. 0.563 for overall survival). Across 11 comparative-treatment cohorts, the oFM embeddings achieved a three-fold higher pooled and scale-normalized treatment-benefit AUTOC than baseline features with improved benefit ranking in 9 of 11 cohorts, and provided stronger prognostic discrimination within both treatment arms. We also evaluated a mechanism discovery framework that interprets downstream models built on oFM embeddings by linking their predicted outcomes to clinically and biologically grounded mechanisms through an evidence-grounded temporal graph, enabling evaluation in clinical and drug-development applications.

## 1 Introduction

Oncology is a longitudinal study. Treatment must adapt to evolving patient states [Murphy, 2003, Chakraborty and Murphy, 2014, Strobl et al., 2023]. However, routine clinical decision systems using clinical variables and biomarkers remain static snapshots that ignore the dynamic cancer behavior that drives metastatic potential, treatment response or resistance, and toxicity burden. Furthermore, while foundation models have revolutionized representation learning within individual biomedical modalities, including pathology [Vorontsov et al., 2024, Chen et al., 2024, Zimmermann et al., 2024, Vorontsov et al., 2026], genomics [Brixi et al., 2026, Theodoris et al., 2023], clinical text [Yang et al., 2022], and structured electronic health records (EHR) [Guo et al., 2024], simply assembling modality-specific encoders fails to capture how biology and health evolve over time and across interventions. A foundation model for precision oncology must handle longitudinal data and support the development of predictive biomarkers, not only prognostic biomarkers. Predictive biomarkers quantify diferential treatment benefit whereas prognostic biomarkers estimate risk independent of treatment and may identify patients with aggressive disease without revealing which treatment could alter their trajectory [Ballman, 2015].

We introduce the oFM, a longitudinal multimodal foundation model for oncology (Figure 1). The oFM integrates Tempus-curated longitudinal records with patient-linked molecular profiles and H&E pathology embeddings. Diverse clinical and molecular events such as laboratory measurements, diagnoses, procedures, treatments, and DNA and RNA biomarkers, are decomposed into atomic facts. Facts are aggregated into daily episodes, embedded with a finetuned language model. A transformer-based trajectory encoder then integrates these episode-level and pathology embeddings to produce a patient trajectory embedding. These embeddings support multimodal patient subtyping, novel biomarker discovery, and scalable AI-enabled diagnostic development. We show that oFM embeddings significantly improve prognostic prediction and the prediction of treatment benefit over traditional clinical features and biomarkers in real-world comparative cohorts. Finally, we explore a mechanism-discovery framework that decomposes model predictions into attributed clinical and molecular features, sparse latent concepts, and evidence-grounded temporal relationships. By combining latent-space intervention with retrieval from biomedical knowledge sources, the framework organizes these signals into an interpretable temporal graph, providing additional oncology insights.

## 2 Related Work

Multimodal Cancer Survival Models MultiSurv [Vale-Silva and Rohr, 2021], MCAT [Chen et al., 2021], and PORPOISE [Chen et al., 2022] integrate clinical, histopathologic, and molecular data at a single time point. In contrast, the oFM models multimodal observations longitudinally capturing changing patient trajectories.

Longitudinal patient and EHR foundation models. BEHRT and Med-BERT use masked language modeling on sequences of structured EHR codes [Li et al., 2020, Rasmy et al., 2021], while CEHR-BERT introduced time encoding between events [Pang et al., 2021]. CLMBR, ETHOS, Foresight, and CoMET used an autoregressive objective which implies time progression [Steinberg et al., 2020, Renc et al., 2024, Kraljevic et al., 2024, Waxler et al., 2025]. Delphi-2M and MOTOR introduced time-to-event objectives [Shmatko et al., 2025, Steinberg et al., 2023], while SMB-v1 predicts future latent states [Adam et al., 2026]. Beyond structured EHR data, APOLLO integrates unstructured clinical text embeddings and medical image embeddings covering whole-slide H&E pathology images, hematology blood smears, and electron microscopy [Zhang et al., 2026]. Whereas APOLLO uses masked language modeling over patient trajectories, the oFM handles future prediction via treatment-conditioned latent space prediction and demonstrates oncology-focused treatment benefit stratification on real world data.

a Multimodal real-world oncology data  
![](images/4064f1d9e2da04140d55656e8e33655c68d8bde57df34cf18131465b160a3b6d.jpg)  
b Pretraining corpus — composition by tumor type

![](images/28afa30de6ac845235f4b20bf907e9384c212fe04fb078edd353425f8561a1de.jpg)

![](images/cd8b2f1c869c53738b322da28c9afd28772dcd2a3c9b2784233fbe9a72e5eda6.jpg)

![](images/86875e78b504e5cc45434d331fdf3bb520f21ed49fc0e522aaab2a6963ee5d29.jpg)

![](images/4220683ca362d6e4607fcc95faf05beaaff167052206b7b02e5144709079dd37.jpg)  
Figure 1: oFM data and architecture. (a) Longitudinal patient records integrate clinical records (EHR and concepts curated from free-text reports), molecular NGS, IHC and ISH, and H&E pathology data. (b-d) The pretraining cohort is summarized by tumor type, modality coverage, and patient-level data partition. (e-f) Daily clinical and molecular facts are encoded as episode embeddings, while pathology embeddings are projected into the shared latent space. (g) A temporal Transformer integrates date-stamped, modalitytagged tokens to produce the patient representation h(t<sup>∗</sup>) for downstream prognostic and treatment-benefit modeling.

Clinical-text and pathology foundation encoders. The oFM uses GatorTron [Yang et al., 2022] to encode clinical and molecular text and PRISM2 [Vorontsov et al., 2026] to encode slide-level H&E embeddings. This modular design allows the oFM to benefit from continued improvements in modality encoders.

Combining pathology and molecular modalities. THREADS [Vaidya et al., 2025] and mSTAR [Xu et al., 2025] combine pathology images with molecular signal for a single specimen. OmniScreen predicts DNA biomarkers directly from H&E images [Wang et al., 2024]. However, these models operate at a single time point and do not support longitudinal patient trajectories.

Genomic foundation models. Sequence and cancer-genomics foundation models such as DNABERT, DNABERT-2 [Ji et al., 2021, Zhou et al., 2024], Nucleotide Transformer [Dalla-Torre et al., 2025], HyenaDNA [Nguyen et al., 2023], and Enformer [Avsec et al., 2021] learn reusable molecular representations. Related single-cell models such as Geneformer and scGPT provide complementary representations at the expression level [Theodoris et al., 2023, Cui et al., 2024]. The oFM integrates molecular inputs over time by ingesting variant and expression summary text alongside other clinical and pathology inputs. The choice of alternative molecular input encoding in the oFM will be explored in further work.

## 3 The oFM Model Architecture

The oncology foundation model (oFM) is a multimodal Transformer that models a cancer patient’s evolving clinical state. Given patient history observed up to time $t ^ { * }$ , the oFM produces a patient embedding $h ( t ^ { * } ) \in \mathbb { R } ^ { 1 0 2 4 }$ . The hierarchical architecture uses an episode encoder to summarize clinical and molecular information in a day as an episode and a temporal encoder to integrate these episode and pathology embeddings over time. See Figure 1 for an overview.

## 3.1 Episode Encoder

An episode contains Tempus-curated clinical facts and molecular measurements (see Appendix A.1 for data curation) attributed to a single day. One or more facts form a time-stamped event. Events are indexed by the number of elapsed days from the first observed event in the patient trajectory. Demographic information forms a separate episode at the beginning of the patient trajectory. Clinical facts include patient demographics, diagnoses, metastases, laboratory measurements, vital signs, performance status, receptor status, treatments, procedures, responses, progression, and follow-up events. Molecular findings per biospecimen include DNA alterations, copy-number variants, fusions, RNA expression abnormalities, splice events, and IHC or NGS biomarkers. All clinical and molecular events are input as text to the episode encoder.

The episode encoder is a finetuned GatorTron-base-2k, a clinical BERT model with hidden dimension 1,024 and a maximum input length of 2,048 tokens [Yang et al., 2022]. The final-layer CLS representation is used as the episode embedding. To preserve measurement magnitudes, we augment the language encoder with a Fourier Number Embedding (FoNE) pathway [Zhou et al., 2026]. Each value is first transformed using its signed log-magnitude and then represented using sine and cosine features at 32 frequencies spanning 0.1 to 10.0. A learned multi-layer perceptron (MLP) and layer normalization (LayerNorm) project these features into the 1,024-dimensional token space. The resulting representation is added to the token embedding at the corresponding [NUM] position, enabling the episode encoder to model both semantic context and magnitude.

H&E whole slide images (WSIs) are embedded alongside episode embeddings as one embedding per biospecimen. We tokenize each WSI with Virchow2 [Zimmermann et al., 2024] and produce a single biospecimen embedding of one or more WSIs with PRISM2 [Vorontsov et al., 2026]. This single embedding is a concatenation of the PRISM2 Base, Diagnostic, and Survival embeddings into a single 8,192-dimensional vector. To map this vector into the trajectory latent space, we train a LayerNorm and linear projection mapping from 8,192 to 1,024 dimensions.

## 3.2 Trajectory Encoder

The episode and pathology embeddings are aggregated by a Transformer-based multimodal trajectory encoder in a shared 1,024-dimensional latent space. The encoder consists of two Transformer layers (16 attention heads, 4,096-dimensional feed-forward layer, dropout probability 0.1, and FlashAttention). A learned modality encoding $\mathbf { m } _ { k } \in \mathbb { R } ^ { 1 0 2 4 } , k \in \{ \mathrm { e , p } \}$ is added to each token to distinguish episode embeddings from pathology embeddings. Relative time is represented using rotary positional embeddings (RoPE) [Su et al., 2021]. A single day may contain up to one episode token and any number of pathology tokens. Thus a day i containing both an episode embedding x<sub>e</sub> and a biospecimen-level embedding $\mathbf { x } _ { \mathrm { p } }$ of pathology whole slide images produces a set of tokens:

$$
\mathbf { d } _ { i } = \left[ \mathbf { x } _ { \mathrm { e } } + \mathbf { m } _ { \mathrm { e } } , \mathbf { x } _ { \mathrm { p } } + \mathbf { m } _ { \mathrm { p } } \right] .\tag{1}
$$

For a sequence of D observed days, the trajectory encoder observes as a sequence:

$$
\mathcal { H } _ { D } = [ \mathbf { c } , \mathbf { d } _ { 1 } , \dots , \mathbf { d } _ { D } ] ,\tag{2}
$$

where $\mathbf { c } \in \mathbb { R } ^ { 1 0 2 4 }$ is a learned CLS token. When there is no observed input for a calendar day, it is

excluded from the sequence. All tokens attend to each other bidirectionally. The output CLS token of the trajectory encoder is used as the trajectory embedding h at time t<sup>∗</sup>:

$$
\mathbf { h } ( t ^ { * } ) = \mathrm { T e m p o r a l E n c o d e r } _ { \mathrm { C L S } } \left( \mathcal { H } _ { \tau } : \tau \leq t ^ { * } \right) \in \mathbb { R } ^ { 1 0 2 4 } .\tag{3}
$$

Evaluating h at diferent clinical anchor points enables longitudinal comparison.

## 4 Training the oFM

![](images/9aca8427ba68410293356c4bb17ad6bcde00ce1caf4dc4ebf3a0e32e7c4fab60.jpg)

(d) Anchor-conditioned views  
![](images/0607ce22d623d4fbc66aee54ffa2c7f86eec304de8582a29c3a768788a86f00f.jpg)

![](images/82984f1b3ca02853e9be3ee33853396cb10140b218c99069d5d534c3f36318bb.jpg)  
Figure 2: oFM training curriculum. (a) Stage I pretrains the episode encoder by reconstructing missing tokens in per-day episodes. (b) Stage II freezes the episode and PRISM2 encoders while training the temporal encoder, prediction heads, and projectors. (c) Stage III unfreezes the episode encoder. (d) Anchor-conditioned training compares a student view to a later teacher view, with prez̄diction conditioned on the anchor-day intervention and elapsed time. (e) The student predicts the stop-gradient representation produced by the exponential-moving-average teacher in latent space.

The oFM is trained using a three-stage curriculum (Figure 2). Stage I pretrains the episode encoder using token- and numerical-reconstruction objectives. Stage II freezes the episode encoder and trains the trajectory encoder. Stage III unfreezes the episode encoder for joint end-to-end training. Stage II and III use full patient trajectory inputs and combine masked episode reconstruction, self-supervised future latent prediction, and a supervised survival objective that encourages the patient trajectory representation to retain outcome-relevant information.

Stage I pretrains the episode encoder on individual episodes using Transformer-based Sequential Denoising Auto-Encoding (TSDAE) [Wang et al., 2021]. For each episode, input tokens (excluding [CLS] and [SEP]) are randomly dropped with a 60% probability and the model is trained to reconstruct the complete uncorrupted episode from the output CLS token. This reconstruction is performed by an autoregressive Transformer-based decoder that shares the same depth, selfattention, feed-forward, and layer-normalization parameters, while adding cross-attention to the CLS token. FoNE is trained jointly for numerical values. In addition to token reconstruction, a decoder-side regression head reconstructs numerical values from the decoder hidden states at [NUM] positions. Numerical values v are transformed using the signed log-magnitude, sign(v) log(1 + |v|), and regressed with a smooth-L1 loss. The complete Stage I objective is

$$
\mathcal { L } _ { \mathrm { S t a g e I } } = \mathcal { L } _ { \mathrm { t o k e n } } + \mathcal { L } _ { \mathrm { n u m } } ,\tag{4}
$$

where $\mathcal { L } _ { \mathrm { t o k e n } }$ is the autoregressive token-reconstruction loss and ${ \mathcal { L } } _ { \mathrm { n u m } }$ is the numerical reconstruction loss. This pretrains the episode encoder, tokenizer, and FoNE while the decoder is discarded.

Stages II and III train the trajectory encoder and restrict the training cohort to patients with recorded treatment information. Episodes are grouped into whole per-patient trajectory inputs. The pretrained episode encoder is kept frozen in Stage II and is unfrozen in stage III. The linear projection of episode and pathology embeddings to the trajectory encoder input is trained in both stages. Input trajectories are truncated at strategically sampled anchor points and training is supervised by masked episode reconstruction, joint-embedding trajectory prediction (JEPA), and survival risk supervision.

Masked episode reconstruction. Each episode or linearly projected pathology embedding is independently randomly masked with probability 0.7 by replacing its content representation with a learned episode- or pathology-specific mask embedding, retaining its temporal encoding. The trajectory encoder reconstructs the masked episodes and full pre-projection masked pathology embeddings using modality-specific reconstruction heads. Reconstruction is optimized with the mean squared error loss and, in the case of pathology embeddings, a cosine distance loss. The masked reconstruction loss is thus:

$$
\begin{array} { l } { \mathcal { L } _ { \mathrm { r e c } } = \mathcal { L } _ { \mathrm { e } } + \mathcal { L } _ { \mathrm { p } } } \\ { \mathcal { L } _ { \mathrm { e } } = \frac { 1 } { D _ { \mathrm { e } } | \mathcal { M } _ { \mathrm { e } } | } \displaystyle \sum _ { i \in \mathcal { M } _ { \mathrm { e } } } \| f _ { \mathrm { e } } ( \mathbf { z } _ { i } ) - \mathbf { x } _ { \mathrm { e } , i } \| _ { 2 } ^ { 2 } } \\ { \mathcal { L } _ { \mathrm { p } } = \frac { 1 } { D _ { \mathrm { p } } | \mathcal { M } _ { \mathrm { p } } | } \displaystyle \sum _ { i \in \mathcal { M } _ { \mathrm { p } } } \| f _ { \mathrm { p } } ( \mathbf { z } _ { i } ) - \mathbf { x } _ { \mathrm { p } , i } \| _ { 2 } ^ { 2 } +  \frac { \lambda _ { \mathrm { c o s } } } { | \mathcal { M } _ { \mathrm { p } } | } \displaystyle \sum _ { i \in \mathcal { M } _ { \mathrm { p } } } ( 1 - \cos \bigl ( f _ { \mathrm { p } } ( \mathbf { z } _ { i } ) , \mathbf { x } _ { \mathrm { p } , i } \bigr ) ) } \end{array}\tag{5}
$$

where $\mathcal { M } _ { \mathrm { e } }$ and $\mathcal { M } _ { \mathrm { p } }$ are the sets of masked episode and pathology token indices, $\mathbf { z } _ { i } \in \mathbb { R } ^ { 1 0 2 4 }$ is the trajectory encoder’s output for masked token $i , \ \mathbf { x } _ { \mathrm { e } , i } \in \mathbb { R } ^ { D _ { \mathrm { e } } } \ ( D _ { \mathrm { e } } = 1 0 2 4 )$ is the unmasked episode embedding, $\mathbf { x } _ { \mathrm { p } , i } \in \mathbb { R } ^ { D _ { \mathrm { p } } } \left( D _ { \mathrm { p } } = 8 1 9 2 \right)$ is the pre-projection pathology embedding, $f _ { \mathrm { e } }$ and $f _ { \mathrm { p } }$ are the modality-specific heads, and $\lambda _ { \mathrm { c o s } } = 1$

Anchor pair sampling. For each patient trajectory containing a sequence of episodes, we identify anchor episodes and sample a pair of anchors $( t _ { 1 } , t _ { 2 } )$ to predict outcome $\mathrm { o r }$ patient state at $t _ { 1 }$ given a patient state near $t _ { 1 }$ . An episode that contains an <intervention> line (treatment start, treatment end, or surgery) or a <transition> line (metastasis, progression, response assessment, last-known vital status) acts as an anchor. In a random pair, $t _ { 1 }$ contains an intervention and $t _ { 2 }$ is any intervention or transition after $t _ { 1 }$

Joint-embedding trajectory prediction. Following I-JEPA, we use a student-teacher setup with a predictor network that predicts a future patient state, as encoded by the teacher, from the student’s corrupted patient state $\mathrm { [ A }$ ssran et al., 2023]. Let $g _ { \theta }$ and $h _ { \theta }$ denote the student episode and trajectory encoders with parameters $\theta$ and write $f _ { \theta } = h _ { \theta } \circ g _ { \theta } ;$ the teacher $f _ { \bar { \theta } }$ is an exponential moving average of the student with parameters ${ \bar { \theta } } .$ The student and the teacher see two diferent views of the same trajectory, determined by the sampled anchor pair $( t _ { 1 } , t _ { 2 } )$ . Because $t _ { 1 }$ includes an intervention, the student context $\mathcal { H } _ { < t _ { 1 } }$ contains all episode and pathology embeddings strictly before the anchor day $t _ { 1 }$ to ensure that no information about the intervention leaks into the context. Instead, the episode embedding on the day of the intervention $\left( \mathbf { x } _ { \mathrm { e } , t _ { 1 } } \right)$ conditions the predicted future state at $t _ { 2 }$ . The teacher target view $\mathcal { H } _ { \le t _ { 2 } } ^ { \mathrm { t g t } }$ extends the trajectory through the target anchor: it contains all episode and pathology embeddings up to and including day $t _ { 2 }$ when t<sub>2</sub> is a <transition> anchor, so that the target representation reflects the observed outcome, and up to but not including day $t _ { 2 }$ when $t _ { 2 }$ is an <intervention> anchor, applying the same no-leak rule as at $t _ { 1 }$ . The residual predictor $\Delta _ { \phi }$ thus maps the student’s corrupted context state to the teacher’s future patient state:

$$
\begin{array} { r l } & { \mathbf { z } _ { t _ { 1 } } = f _ { \theta } \Big ( \tilde { \mathcal { H } } _ { < t _ { 1 } } \Big ) , } \\ & { \bar { \mathbf { z } } _ { t _ { 2 } } = f _ { \bar { \theta } } \Big ( \mathcal { H } _ { \leq t _ { 2 } } ^ { \mathrm { t g t } } \Big ) , } \\ & { \widehat { \mathbf { z } } _ { t _ { 2 } } = \mathbf { z } _ { t _ { 1 } } + \Delta _ { \phi } \big ( \mathbf { z } _ { t _ { 1 } } , \mathbf { x } _ { \mathrm { e } , t _ { 1 } } , \psi ( \Delta t ) \big ) , } \end{array}\tag{6}
$$

where all three vectors lie in $\mathbb { R } ^ { 1 0 2 4 }$ $\Delta t = t _ { 2 } - t _ { 1 }$ , and $\psi$ is a sinusoidal embedding of elapsed time refined by a two-layer MLP. $\Delta _ { \phi }$ is a two-layer MLP whose input and hidden activations are FiLM-modulated by $[ \mathbf { x } _ { \mathrm { e } , t _ { 1 } } ; \psi ( \Delta t ) ]$ [Perez et al., 2018]. The teacher target view is encoded without masking. The corrupted student context $\widetilde { \mathcal { H } } _ { < t _ { 1 } }$ is obtained from $\mathcal { H } _ { < t _ { 1 } }$ using the same mask as the masked episode reconstruction objective: each token is replaced by the learned modality-specific mask embedding with probability 0.7.

The predictive loss is $\begin{array} { r } { \mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { 1 0 2 4 } \big \| \widehat { \mathbf { z } } _ { t _ { 2 } } - \mathrm { s g } [ \bar { \mathbf { z } } _ { t _ { 2 } } ] \big \| _ { 2 } ^ { 2 } , } \end{array}$ averaged over the minibatch, with $\mathrm { s g } [ \cdot ]$ the stop-gradient. The teacher receives no gradient and is updated after each optimizer step as $\bar { \theta } $ $m \bar { \theta } + ( 1 - m ) \theta$ with momentum $m = 0 . 9 9 9$ , following BYOL and I-JEPA [Grill et al., 2020, Assran et al., 2023]. Both the $\Delta _ { \phi }$ state predictor and the teacher are discarded after training.

To prevent representational collapse we add VICReg’s variance and covariance terms $( \mathcal { L } _ { \mathrm { v a r } }$ and ${ \mathcal { L } } _ { \mathrm { c o v } } )$ on $\widehat { \mathbf { z } } _ { t _ { 2 } }$ [Bardes et al., 2022]: a hinge holding each dimension’s standard deviation at or above $\gamma = 1$ , weighted $\lambda _ { \mathrm { v a r } } = 1$ , and the mean squared of-diagonal covariance, weighted $\lambda _ { \mathrm { c o v } } = 0 . 0 4$

Pathology dropout. To improve robustness to missing pathology data, each pathology embedding is randomly removed with probability 0.3 during training. A removed embedding is dropped simultaneously for both the student and the teacher networks. Dropped embeddings are excluded entirely from the trajectory and are not treated as masked-reconstruction targets.

Survival supervision. Trajectory training is supervised by overall survival prediction. A survival head predicts a risk score from the patient state embedding $h ( \mathcal { H } _ { < t _ { 1 } } )$ , up to but not including the anchor day $t _ { 1 }$ , concatenated with the episode embedding ${ \bf x } _ { \mathrm { e } , t _ { 1 } }$ at $t _ { 1 }$ . Survival time is measured from the anchor day to death or to the last recorded event (right-censoring), and the model is optimized using the Cox partial-likelihood loss $\mathcal { L } _ { \mathrm { C o x } }$ with Efron handling of tied event times. Risk sets are constructed over the globally gathered distributed minibatch.

The complete trajectory-level objective in stages II and III is:

$$
\mathcal { L } _ { \mathrm { t r a j e c t o r y } } = \mathcal { L } _ { \mathrm { r e c } } + \mathcal { L } _ { \mathrm { p r e d } } + \mathcal { L } _ { \mathrm { v a r } } + 0 . 0 4 \mathcal { L } _ { \mathrm { c o v } } + \mathcal { L } _ { \mathrm { C o x } } ,\tag{7}
$$

with all loss terms weighted equally except the covariance regularizer.

Both stages are optimized with AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 )$ at bfloat16 mixed precision with gradient clipping at a maximum norm of 1.0. Trajectory components (trajectory encoder, predictor, time embedding, projection and reconstruction heads, and survival head) train at a learning rate of $1 0 ^ { - 4 }$ with weight decay 0.01. A 1000-step linear warmup is used for the trajectory encoder. Stage III unfreezes the pretrained episode encoder and trains it with a $1 0 ^ { - 4 }$ learning rate with weight decay $0 . 0 1 \ ( 1 0 ^ { - 5 }$ for the token-embedding matrix, without weight decay).

## 5 Data

The oFM was developed within a 1,672,203 patient cohort of the Tempus de-identified multimodal real-world oncology corpus (Fig. 1). Of these, 1,045,011 patients were used for training the episode encoder (stage I), a subset of 386,382 patients (109,938 with pathology H&E images), all with recorded treatment events, were used to train the temporal encoder (stage II), and a subset of 92,567 information-rich patients with both paired DNA and RNA records were used for final endto-end finetuning (stage III). Patient records were structured as longitudinal sequences anchored at the earliest observed event, indexing subsequent events by elapsed calendar days to preserve irregular clinical intervals. All test set patients in evaluation tasks were excluded from model training. The data used to train and evaluate the oFM is distinct and non-overlapping with the data used to train Virchow2 and PRISM2.

The longitudinal record integrates clinical events, molecular blueprints, and digitized H&E pathology over time. Clinical information spans demographics, encounters, vitals, labs, diagnoses, treatments, procedures, and outcomes. These data are organized into observations, interventions, and disease-state transitions. Observations describe measured or recorded patient characteristics. Interventions include treatments and procedures. Transitions represent clinically relevant changes such as new metastasis, tumor progression, treatment response, or last-known follow-up. Molecular findings for each tumor biospecimen were consolidated into a standardized textual record termed a molecular blueprint. These include DNA sequencing, whole-transcriptome RNA sequencing, immunohistochemistry (IHC), and in situ hybridization (ISH). See Appendix A.2 for more details.

## 6 Downstream Evaluation

The downstream benchmarks evaluate the relative quality of the learned patient trajectory embeddings. Despite the challenges related to retrospective real-world data (e.g., confounding, censoring), we demonstrate that the oFM provides a prognostic and predictive advantage over a strong taskspecific baseline. Specifically, there is a substantial and statistically significant improvement in aggregate performance across benchmarks.

In comparison to the oFM embeddings, we evaluate a strong curated-feature baseline that selects from 7520 candidate curated clinical and molecular features (NGS: DNA and RNA; IHC) using the feature selection process described in Appendix A.4. All prognostic and predictive evaluation was performed using the two-stage convex UV-Cox method presented in Appendix A.6.

## 6.1 Prognostic Classification

We perform binary linear probing on three endpoint measures to evaluate outcome-prognostic performance of oFM embeddings encoded just prior to the initiation of first-line therapy:

(i) Treatment response groups: Responder (Complete or Partial, CR/PR) versus others (stable or progressive, SD/PD), using the latest assessment between 14 and 365 days after first-line treatment initiation and prior to the second line.

(ii) Progression-free survival (PFS) groups: Sustained progression-free status vs early progression. Positive if PFS above 60th percentile; negative if below 30th percentile.

![](images/9ff741f0794bc7cde417084bc0f25c0e0d232958dfdbbbcf940f288ea219e7a5.jpg)  
a Average performance (tumor × therapy class strata, ≥20 patients)

(iii) Overall survival (OS) groups: Long-term vs short-term survival. Positive OS above 60th percentile; negative if below 30th percentile.

For each task, we train a logistic-regression probe on frozen patient embeddings extracted just before first-line initiation (see Appendix A.6 for method). To reduce prognostic confounding by tumor type and therapy class, we stratify test-set performance by tumor type × therapy-class into 39 / 38 / 40 strata for OS / PFS / treatment-response, retaining at least 20 patients per stratum, and compute the mean (see Appendix Tab. A1 for strata).

![](images/0e3f5de71a64a39fb48e4cfd7dd6d73c84fe14b686a54db8e0deab1d5e8def8a.jpg)

b Stratum-wise summary (tumor × therapy class)  
![](images/93969c81ccdaeef8e2ab6f27a76fc67a8839bd7becfd672705c5caba645d866a.jpg)

![](images/ed686375d80a68a8c2b93a44f7d1bc97159939b479ee474c08f205cac6bca4f8.jpg)

![](images/b6a72969e0a0e263ac119bfc9cc3e178615862d80c9e05582cd84195d5c12111.jpg)  
Figure 3: Prognostic classification (tumor × therapy stratified). (a) Stratified summary (macro-mean AUC and AP) for response, PFS group, and OS group, comparing a frozen-embedding probe (oFM) to an engineered-feature baseline under an identical probe. (b) Per-stratum AUC comparison (one point per tumor × therapy stratum; point area proportional to patients) showing that gains are broad-based and concentrated in larger strata.

As shown in Fig. 3, linear probing on the oFM embeddings outperforms baseline features on all three endpoints, achieving 0.774 mean AUC for OS, 0.688 for PFS, and 0.585 for treatment response (CR/PR vs SD/PD), compared with 0.563, 0.544, and 0.513 for the baseline, respectively. The oFM performance beat the baseline on 95% of OS and PFS strata and 78% of treatment response strata.

## 6.2 Predictive Benchmarks: Treatment Benefit

We evaluate the oFM on benchmark cohorts spanning breast, colorectal, non-small-cell lung (NSCLC), renal-cell, prostate, and pan-cancer disease settings via Cox-based linear probing (using the UV-Cox method presented in Appendix A.6). Each benchmark poses a clinical question noted per cohort in Appendix Tab. A3. In the predictive setting, two treatment arms are compared in each cohort for diferential treatment benefit. We use inverse-propensity weighting for predictive benchmarking to reduce data imbalance across arms inherent in real-world data (see Appendix A.5).

As shown in Fig. 4, the oFM ranks treatment benefit and discriminates the baseline risk better than the curated-feature baseline. Fig. 4a reports the $t _ { \mathrm { A U T O C / S D } }$ , the pooled AUTOC/SD normalized by $\mathrm { S E _ { A U T O C } }$ , which measures how consistently nonzero the treatment benefit efect is across cohorts, and the pooled HRR (Paule-Mandel random efects; metrics detailed in Appendix A.7). The oFM achieves significant $\left( \mathrm { { p } = 0 . 0 0 0 3 } \right)$ treatment benefit ranking at $t _ { \mathrm { A U T O C / S D } } = 4 . 6 1$ compared to the baseline at 1.38 which does not clear its permutation null of 1.85 (see Appendix A.7 for definition); while the baseline does extract treatment benefit signal, it is not a consistently large efect. Fig. 4b shows per-cohort wins: the oFM wins over the baseline in 9 of 11 cohorts.

Fig. 5 shows per-arm prognostic performance. As shown in Fig. 5a, the oFM embeddings produce better prognostic risk ordering (Uno’s IPCW C-index, pooled with Paule-Mandel random efects) per arm than baseline features. In Fig. 5b, the oFM outperforms the baseline on 10 and 9 of 11 tasks in control and experimental arms, respectively. That the gain in C-index is similar in both the control and experimental arms is evidence that the signal is prognostic rather than a treatment efect leaking into the score.

a Performance pooled across 11 cohorts Treatment-benefit ranking (higher is better)

![](images/1262967b0eaac7f3b2ca8daec23029daff36090f95c980e9691bf95f5ba23f33.jpg)  
Benefit-ordering direction (lower is better)

![](images/21546c4eeab644680130493fdcc5dbebe814cd13f140367c0f920ef4ed584378.jpg)

## b Cohort-wise summary

![](images/afa801a80e2c245cf9e46e612e309d034eb9884674dabcac54985131824f2f57.jpg)  
oFM higher baseline higher oFM effect significant (p<0.05) standard error (SE) 0.02 0.05 0.10  
Figure 4: Predictive evaluation. oFM frozen embeddings against the curated-feature baseline. (a) Metrics pooled across tasks by Paule-Mandel random efects: (left) $t _ { \mathrm { A U T O C / S D } }$ is the pooled AUTOC/SD normalized by $\mathrm { S E _ { A U T O C } }$ and measures how consistently nonzero the treatment benefit efect is across cohorts and (right) the pooled HRR. The yellow bands are the 5% permutation nulls, about zero (no signal) for the $\mathrm { A U T O C / S D }$ ranking and about one (no benefit) for the $\operatorname { H R R } ; \operatorname { p }$ values are computed with respect to each null. (b) Per-cohort AUTOC/SD, oFM against baseline. Area is proportional to $1 / \mathrm { S E } ^ { 2 }$ ; points above the diagonal favor oFM and a bold outline marks $p < 0 . 0 5$

a C-index pooled over 11 cohorts  
![](images/252c5dbf78fbd3245ca3d36c6fd05a20744dead54c0f0175b09ed13b31a815b0.jpg)  
b Cohort-wise summary

![](images/39708ff9d4a80dabdf86cad102391def88c5328574a27a97069f19fed0b317bd.jpg)

![](images/295d2f59dba2b403cd3f4a8739f95718ce9a19825203d79efdb6feb27a5c3bf9.jpg)  
Figure 5: Prognostic discrimination. Uno’s C-index computed within the control arm (left) and the experimental arm (right), for oFM against the curated-feature baseline. (a) Pooled across cohorts by Paule–Mandel random efects: baseline C-index (grey) and oFM (blue) with p the significance of the diference via two-sided Student t-test. The yellow band shows the range indistinguishable from the baseline. (b) One point per cohort and arm for baseline or oFM, whichever scores better; area proportional to $1 / \mathrm { S E ^ { 2 } }$ . Points above the diagonal favor oFM.

## 7 Mechanism Discovery

Beyond measuring predictive performance, we seek to make downstream models built on oFM embeddings biologically and clinically interpretable by identifying human-interpretable mechanisms that drive their predictions. Mechanism discovery relies on finding sparse directions in embedding space that: (i) are diferentially active between high- and low-scored patients, (ii) semantically map to external clinical and biological concepts, (iii) causally alter downstream predictions under activation steering, and (iv) assemble into a temporally ordered graph connecting concepts to outcomes via statistically tested precedence. Because the oFM jointly represents longitudinal clinical history with patient-linked DNA and RNA profiles, this framework enables downstream predictions to be interrogated in terms of observed clinical and tumor-biological factors, including molecular mechanisms associated with prognosis, treatment response, and resistance. These mechanisms can then be evaluated in downstream clinical and drug-development applications. The mechanism discovery pipeline is applied per evaluation cohort on frozen oFM embeddings $h ( t ^ { * } )$ and a fitted predictor $s ( h )$ in seven stages (Figure 6):

1. Fact attribution: Atomic facts (diagnoses, biomarkers, agents, lab values) are iteratively removed from a patient trajectory. We recompute the embedding and record the resulting score shift $\Delta s .$ . Within-patient normalization by total absolute attribution yields a signed, length-comparable metric for each fact.

2. Sparse feature: A Sparse Autoencoder (SAE) [Cunningham et al., 2024] projects embedding h into an overcomplete latent space z under an $L _ { 1 }$ sparsity constraint. The linear decoder yields sparse feature directions that facilitate downstream intervention in stage 5, while enabling near-lossless score reconstruction $( \hat { h } \approx h )$

3. Diferential activation testing and clustering: Patients are partitioned by the population median of score $s ( h )$ . SAE latents are screened for diferential activation between high- and low-score groups of patients using Mann–Whitney rank tests and Cohen’s $d ,$ adjusted via Benjamini–Hochberg FDR control [Benjamini and Hochberg, 1995]. Co-activating latents are grouped into interpretable mechanisms via clustering with a hybrid Jaccard-cosine metric.

4. Attribution-weighted concept labelling: Mechanisms are labeled by scaling per-concept attribution strengths (see stage 1) with cohort-wide inverse document frequency (IDF). This filters ubiquitous administrative events and highlights disproportionately attributed clinical concepts. Near-synonym concepts are normalized.

5. Causal steering: To evaluate causality, mechanism latents are ablated (zeroed) or amplified (clamped to the 95th percentile) prior to decoding and re-scoring. Observed score shifts and class-assignment flips categorize mechanisms as drivers, suppressors, or amplifiable. All validated cluster latents are simultaneously ablated and re-scored, yielding a variance-explained statistic $\mathrm { V E } = 1 - \mathrm { V a r } ( s _ { \mathrm { a b l a t e d } } ) / \mathrm { V a r } ( s _ { \mathrm { o r i g i n a l } } )$ . A coverage decomposition separates clustered from unclustered contributions and reports a concentration ratio (per-feature VE of clustered versus unclustered latents) to distinguish concentrated from difuse mechanisms.

6. Retrieval-grounded mechanistic reasoning: A local domain language model (MedGemma 4B [Sellergren et al., 2026]) queries an ofline FAISS index [Douze et al., 2024] containing ten public biomedical knowledge bases, namely CIViC [Grifith et al., 2017], OncoKB [Chakravarty

![](images/da6080a26fa7abf0dbdedff75f77f4994f49d00d09c1a936ed078c3528026fdc.jpg)  
Figure 6: Mechanism-discovery pipeline. Seven steps decompose a frozen embedding h(t<sup>∗</sup>) and fitted head s(h) into interpretable mechanisms: (1) leave-one-out fact attribution; (2) sparseautoencoder dictionary learning; (3) diferential activation and clustering; (4) IDF-enriched concept labelling; (5) causal steering via latent-space ablation/amplification; (6) composition analysis and retrieval-grounded reasoning; (7) temporal-precedence mechanism graph with curated knowledge overlay and chain distillation. All steps run on frozen embeddings without fine-tuning.

et al., 2017], DrugBank [Wishart et al., 2018] , PharmGKB [Whirl-Carrillo et al., 2021], Reactome [Milacic et al., 2024], GO [The Gene Ontology Consortium et al., 2023, Ashburner et al., 2000], MSigDB [Liberzon et al., 2015], String [Szklarczyk et al., 2023], PubMed [Sayers et al., 2026], and ClinicalTrials [Zarin et al., 2011], embedded with BioLORD-2023 [Remy et al., 2024]. The ofline model outputs structured phenotype/hypothesis/evidence records and citation-grounded subject/relation/object triplets via retrieval-augmented generation [Lewis et al., 2020], supplying the hypothesized-edge layer of the mechanism graph (see stage 7).

7. Graph Assembly: temporal precedence and knowledge overlay: To structure candidate concepts into a temporally ordered narrative, flat concept table entries are assembled into a directed acyclic graph (DAG; Figure 7), connecting concepts to the outcome via statistically-tested precedence. Temporal edges $( A \to B )$ are retained for concept pairs cooccurring in $\ge n _ { \mathrm { m i n } }$ patients where A precedes B in > 60% of cases (one-sided sign test, BH-FDR $q \leq 0 . 0 5 )$ . This temporal backbone is enriched with three multi-layer edge types: curated biological interactions (CIViC, STRING, MSigDB), external correlational associations $( | \rho | \geq 0 . 3 $ , FDR-controlled), and the RAG-generated triplets from stage 6. Finally, the DAG is distilled into concise, domain-interpretable narrative chains (length 2-4 concepts to outcome) ranked by path weight and terminal concept diversity.

The SAE acts as a discovery engine to identify, cluster, and test latent concepts, while the Mechanism Graph temporally orders these concepts across patient histories and links them to outcomes via established or novel candidate biological pathways. Designed as a generalizable mechanism discovery framework, the pipeline supports generating and evaluating biological hypotheses both for prognostic risk strata and for diferential treatment benefit.

To demonstrate its utility, we evaluated overall survival predictions in a multi-tumor cohort (metastati breast cancer, non-small cell lung cancer [NSCLC], and gastroesophageal cancer) treated with trastuzumab deruxtecan (T-DXd) versus physician’s choice chemotherapy in later-line settings. Reflecting real-world clinical practice and data completeness, patient eligibility was relaxed from HER2- positive status to include HER2-low, HER2-mutant, and HER2-unknown or unrecorded cases. In these cases, T-DXd demonstrates clinical activity via its potent bystander efect or is commonly prescribed in routine clinical practice. The pipeline recovered mechanisms consistent with established disease biology while generating plausible exploratory hypotheses for clinical follow-up (Figure 7). Specifically, high-benefit outcomes are associated with features reflecting both prior antibody-drug conjugate (ADC) vulnerability and chemotherapy resistance. Across all three histologies, high benefit is associated with PD-L1-negative status (< 1%), a population that traditionally derives limited response from conventional immune checkpoint inhibitors. In the breast cancer subgroup, higher therapeutic benefit is associated with baseline core needle biopsy profiling and prior carboplatin/gemcitabine failure. Additionally, the pipeline surfaced metabolic signatures such as APOB overexpression, which likely reflects site-specific liver biopsy context requiring further experimental isolation. Within the NSCLC cohort, high benefit correlated with pronounced intratumoral heterogeneity, a phenotype that could be consistent with sensitivity to the bystander killing efect of T-DXd.

Conversely, low-benefit clusters are characterized by compromised drug delivery, reduced target antigen availability, or aggressive multi-drug resistance phenotypes. For example, breast cancer cases frequently presented with brain metastases, which may limit large-molecule delivery across the bloodbrain barrier. Mechanistically, this low-benefit cluster aligns with a low copy-number variant (CNV) burden (Gain CN3-4 fraction 0.1-0.3, Amplification CN5-7 fraction < 0.1), indicating a reduced likelihood of the HER2 genomic alterations required for efective ADC target engagement. Furthermore, these NSCLC patients frequently had prior exposure to carboplatin/pembrolizumab/pemetrexed triple-therapy, which may reflect prior treatment selection for resistant tumor subclones. In the gastroesophageal cancer subset, the low-benefit phenotype was enriched for lineage-associated genomic features such as GATA4 overexpression; this may act as a tissue confounder rather than a response modifier, separating the two requires prospective work. Steering the feature clusters shifts the model’s predicted risk score, so the mechanism graph provides an interpretable, directed framework mapping upstream clinical and genomic input events to predicted patient outcomes.

![](images/acecf8b3b5494c77be12095581c2681f2f4ec406d82089e178086d1596d9bccb.jpg)  
Figure 7: Mechanism graph. Illustrative mechanism graph depicting feature interactions associated with treatment benefit in second-line or later (2L+) patients treated with trastuzumab deruxtecan (T-DXd) versus physician’s choice chemotherapy in a multi-tumor cohort. Directed chains are distilled from the temporalprecedence directed acyclic graph (DAG). Concept nodes are color-coded by category (genomic biomarkers, treatment, diagnosis, procedure, metastatic site); temporal edges are labeled with precede-fraction and median day gap (in days); annotation chips mark curated gene-drug actionability (CIViC), RNA expression correlates derived via probing analysis, and LLM-hypothesized relations with their retrieved citations listed in Table 1.

Table 1: Numbered references for the mechanism-chain (Figure 7). Sources: PubMed entries in author/journal form; CIViC, PharmGKB and trial records in accession form. CIViC tiering includes cross-indication evidence.
<table><tr><td>#</td><td>Reference</td></tr><tr><td>1</td><td>CIViC evidence EID993: KRAS Exon 2 Mutation in Colorectal Cancer (Predictive, level A). CIViC evidence EID2998: KRAS Mutation in Lung Non-small Cell Carcinoma (Predictive,</td></tr><tr><td>2</td><td>level A).</td></tr><tr><td>3</td><td>CIViC evidence EID5345: KRAS Mutation in Colorectal Cancer (Predictive, level A).</td></tr><tr><td>4</td><td>CIViC evidence EID704: CD274 Expression in Melanoma (Predictive, level B).</td></tr><tr><td>5</td><td>CIViC evidence EID1167: CD274 Expression in Lung Non-small Cell Carcinoma (Predictive, level B).</td></tr><tr><td>6</td><td>CIViC evidence EID7313: PIK3CA Mutation in Breast Cancer (Predictive, level A).</td></tr><tr><td>7</td><td>CIViC evidence EID12020: PIK3CA Mutation OR PTEN Mutation OR AKT1 Mutation in Breast Cancer (Predictive, level A).</td></tr><tr><td>8</td><td>CIViC evidence EID12533: PIK3CA H1047R OR PIK3CA H1047Y OR PIK3CA H1047L OR PIK3CA H1047D OR PIK3CA H1047I OR PIK3CA H1047N OR PIK3CA H1047P OR</td></tr><tr><td>9</td><td>PIK3CA H1047Q OR PIK3CA H1047T in Breast Cancer (Predictive, level A). CIViC evidence EID399: TP53 R249 in Breast Cancer (Predictive, level B).</td></tr><tr><td>10</td><td>CIViC evidence EID850: TP53 Mutation in Gastric Adenocarcinoma (Predictive, level B).</td></tr><tr><td>11</td><td>CIViC evidence EID875: TP53 Wildtype in Colorectal Cancer (Predictive, level B).</td></tr><tr><td>12</td><td>PharmGKB: IFNL3-interferons;peginterferon alfa-2a;peginterferon alfa-2b;ribavirin (Efficacy, level 1B).</td></tr><tr><td>13</td><td>Bellini MF et al. Alterations of the TP53 gene in gastric and esophageal carcinogenesis. J Biomed Biotechnol. 2012;2012:891961. PMID 22919278.</td></tr><tr><td>14</td><td>Kumar D et al. Revisiting the Association of ECOG Performance Status With Clinical Outcomes in Diverse Patients With Cancer. J Natl Compr Canc Netw. 2024;22(2 D). PMID</td></tr><tr><td>15</td><td>38653321. Xing AY et al. p53 Missense Mutation is Associated with Immune Cell PD-L1 Expression in Triple-Negative Breast Cancer. Cancer Invest. 2022;40(10):879-888. PMID 35980253.</td></tr><tr><td>16</td><td>Li X et al. Molecular Landscape of TP53/RB1 Co-Altered Tumors Uncovers Emerging Ther-</td></tr><tr><td>17</td><td>apeutic Vulnerabilities. Genes Chromosomes Cancer. 2026;65(1):e70100. PMID 41562186. Kim T et al. Linking p53 immunostaining to TP53 mutation status in patients with non-small</td></tr><tr><td>18</td><td>cell lung cancer. Pathology. 2025;57(7):881-889. PMID 40835511. Bharathiraja P et al. Solasodine targets NF-κB signaling to overcome P-glycoprotein medi-</td></tr></table>

## 8 Conclusion

We introduced the oFM, a multimodal foundation model that integrates longitudinal clinical history with patient-linked DNA and RNA findings and H&E pathology embeddings. Across prognostic and comparative-treatment benchmarks, frozen oFM embeddings consistently outperformed strong baselines based on expert-curated clinical and molecular features, suggesting that the model captures information not readily recovered from conventional static features. The mechanism-discovery framework further provides an interpretable layer for downstream models built on oFM embeddings by linking their predictions to attributed clinical and molecular events, sparse latent features, and evidence-grounded temporal hypotheses. By jointly representing longitudinal clinical context with observed tumor biology, including DNA and RNA profiles, this framework enables downstream predictions to be interrogated in terms of biological and clinical factors associated with prognosis, treatment response, and resistance. These retrospective real-world evaluations are intended to compare representation quality rather than establish clinically validated biomarkers, treatment efects, or definitive biological mechanisms. External and prospective validation will be required, but the results support longitudinal multimodal modeling as a shared foundation for scalable, interpretable applications in oncology diagnostics and therapeutic development.

## Acknowledgments

We thank Arpita Saha, Gabriel Altay, Hongru Hu, Rohan Prakash Joshi, Sam Heilbroner, Sujoy Ganguly, Rafi Pelossof, Simona Cristea, Jiajie Xiao, and Fan Zhang for contributions to preliminary work underlying this project, including early knowledge transfer, development of the initial codebase and data-processing pipelines, and valuable discussions that informed the model training and evaluation protocols.

We thank Joe Oakley, MD, for valuable feedback and guidance on model interpretability and clinical use case formulation.

We thank Ryan Godart, Anirban Nandi, Daniel Cohn, and Evan Czyzycki for building and supporting the compute cluster that enabled large-scale model development and experimentation. We also thank Jonathan Wills, Maria Berezina, and Allison Maresca for organizing the data curation and delivery that enabled this work.

Finally, we thank Or Yaacov and Ben Terdich for contributions to the initial design of the benchmark task suite and for helping establish the benchmark evaluation protocol.

## References

Irsyad Adam, Zekai Chen, David Laprade, Shaun Porwal, David Laub, Erik Reinertsen, Arda Pekis, and Kevin Brown. The patient is not a moving document: A world model training paradigm for longitudinal electronic health records. arXiv preprint arXiv:2601.22128, 2026.

P. K. Andersen, J. P. Klein, and S. Rosthøj. Generalised linear models for correlated pseudoobservations, with applications to multi-state models. Biometrika, 90(1):15–27, 2003.

Michael Ashburner, Catherine A Ball, Judith A Blake, David Botstein, Heather Butler, J Michael Cherry, Allan P Davis, Kara Dolinski, Sanya S Dwight, Janan T Eppig, Michael A Harris, David P Hill, Laurie Issel-Tarver, Andrew Kasarskis, Suzanna Lewis, John C Matese, Joel E Richardson, Martin Ringwald, Gerald M Rubin, and Gavin Sherlock. Gene ontology: tool for the unification of biology. Nature Genetics, 25(1):25–29, 2000. doi: 10.1038/75556.

Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, June 2023. URL https: //openaccess.thecvf.com/content/CVPR2023/html/Assran\_Self-Supervised\_Learning\_ From\_Images\_With\_a\_Joint-Embedding\_Predictive\_Architecture\_CVPR\_2023\_paper.html.

Peter C. Austin and Elizabeth A. Stuart. Moving towards best practice when using inverse probability of treatment weighting (iptw). Statistics in Medicine, 34(28):3661–3679, 2015.

Žiga Avsec, Vikram Agarwal, Daniel Visentin, et al. Efective gene expression prediction from sequence by integrating long-range interactions. Nature Methods, 18:1196–1203, 2021. doi: 10. 1038/s41592-021-01252-x.

Karla V. Ballman. Biomarker: Predictive or prognostic? Journal of Clinical Oncology, 33(33): 3968–3971, 2015. doi: 10.1200/JCO.2015.63.3651.

Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. In International Conference on Learning Representations, 2022.

Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1):289–300, 1995. doi: 10.1111/j.2517-6161.1995.tb02031.x. URL https://doi.org/10.1111/ j.2517-6161.1995.tb02031.x.

Garyk Brixi, Matthew G Durrant, Jerome Ku, Mohsen Naghipourfar, Michael Poli, Gwanggyu Sun, Greg Brockman, Daniel Chang, Alison Fanton, Gabriel A Gonzalez, et al. Genome modelling and design across all domains of life with evo 2. Nature, 652(8112):1349–1361, 2026.

Bibhas Chakraborty and Susan A. Murphy. Dynamic treatment regimes. Annual Review ofStatistics and Its Application, 1:447–464, 2014. doi: 10.1146/annurev-statistics-022513-115553.

Debyani Chakravarty, Jianjiong Gao, Sarah M Phillips, Ritika Kundra, Hongxin Zhang, Jack Wang, Julia E Rudolph, Rona Yaeger, David Soumerai, David B Nissan, et al. Oncokb: A precision oncology knowledge base. JCO Precision Oncology, 2017(1):1–16, 2017. doi: 10.1200/PO.17. 00011.

Richard J. Chen, Ming Y. Lu, Wei-Hung Weng, Tifany Y. Chen, Drew F. K. Williamson, Trevor Manz, Maha Shady, and Faisal Mahmood. Multimodal co-attention transformer for survival prediction in gigapixel whole slide images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4025, 2021.

Richard J. Chen, Ming Y. Lu, Drew F. K. Williamson, Tifany Y. Chen, Jana Lipkova, Zahra Noor, Muhammad Shaban, Maha Shady, Mane Williams, Bumjin Joo, and Faisal Mahmood. Pan-cancer integrative histology-genomic analysis via multimodal deep learning. Cancer Cell, 40 (8):865–878.e6, 2022. doi: 10.1016/j.ccell.2022.07.004.

Richard J. Chen et al. Towards a general-purpose foundation model for computational pathology (UNI). Nature Medicine, 2024. doi: 10.1038/s41591-024-02857-3. URL https://www.nature. com/articles/s41591-024-02857-3.

D. R. Cox. Regression models and life-tables. Journal of the Royal Statistical Society: Series B, 34 (2):187–220, 1972.

Haotian Cui et al. scGPT: toward building a foundation model for single-cell multi-omics using generative ai. Nature Methods, 2024. doi: 10.1038/s41592-024-02201-0.

Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2309.08600.

Hugo Dalla-Torre et al. Nucleotide transformer: building and evaluating robust foundation models for human genomics. Nature Methods, 22(2):287–297, 2025. doi: 10.1038/s41592-024-02523-z.

Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Hervé Jégou, Jef Johnson, Mathilde Lucas, Guillaume Mazaré, Maria Mohamed, Ni Ni, Andrea Vedaldi, et al. The faiss library. arXiv preprint arXiv:2401.08281, 2024.

Malachi Grifith, Nicholas C Spies, Kilannin Krysiak, Joshua F McMichael, Adam C Cofman, Arpad M Danos, Benjamin J Ainscough, Cody A Ramirez, Damian T Rieke, Lynzey Kujan, et al. Civic is a community knowledgebase for expert crowdsourcing the clinical interpretation of variants in cancer. Nature genetics, 49(2):170–174, 2017.

Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlagh Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In Advances in Neural Information Processing Systems, volume 33, pages 21271–21284. Curran Associates, Inc., 2020. URL https://proceedings. neurips.cc/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html.

Lin Lawrence Guo, Jason Fries, Ethan Steinberg, Scott Lanyon Fleming, Keith Morse, Catherine Aftandilian, Jose Posada, Nigam Shah, and Lillian Sung. A multi-center study on the adaptability of a shared foundation model for electronic health records. npj Digital Medicine, 7:171, 2024. doi: 10.1038/s41746-024-01166-w.

Alexia Iasonos, Paul B. Chapman, and Jaya M. Satagopan. Quantifying treatment benefit in molecular subgroups to assess a predictive biomarker. Clinical Cancer Research, 22(9):2114–2120, 2016. doi: 10.1158/1078-0432.CCR-15-2517.

Yanrong Ji, Zhihan Zhou, Han Liu, and Ramana V. Davuluri. DNABERT: pre-trained bidirectional encoder representations from transformers model for DNA-language in genome. Bioinformatics, 37(15):2112–2120, 2021. doi: 10.1093/bioinformatics/btab083.

John P. Klein and Melvin L. Moeschberger. Survival Analysis: Techniques for Censored and Truncated Data. Springer, 2003.

Zeljko Kraljevic, Dan Bean, Anthony Shek, Rebecca Bendayan, Harry Hemingway, Joshua Au Yeung, Alexander Deng, Alfred Baston, Jack Ross, Esther Idowu, James T. Teo, and Richard J. B. Dobson. Foresight: A generative pretrained transformer for modelling of patient timelines using electronic health records: A retrospective modelling study. The Lancet Digital Health, 6(4): e281–e290, 2024. doi: 10.1016/S2589-7500(24)00025-6.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33: 9459–9474, 2020.

Yikuan Li, Shishir Rao, José Roberto Ayala Solares, Abdelaali Hassaine, Rema Ramakrishnan, Dexter Canoy, Yajie Zhu, Kazem Rahimi, and Gholamreza Salimi-Khorshidi. BEHRT: Transformer for electronic health records. Scientific Reports, 10:7155, 2020. doi: 10.1038/s41598-020-62922-y.

Arthur Liberzon, Chet Birger, Helga Thorvaldsdóttir, Mahmoud Ghandi, Jill P Mesirov, and Pablo Tamayo. The molecular signatures database (msigdb) hallmark gene set collection. Cell Systems, 1(6):417–425, 2015. doi: 10.1016/j.cels.2015.12.004.

Marija Milacic, Deidre Beavers, Patrick Conley, Chuqiao Gong, Marc Gillespie, Johannes Griss, Robin Haw, Bijay Jassal, Lisa Matthews, Bruce May, Robert Petryszak, Eliot Ragueneau, Karen Rothfels, Cristofer Sevilla, Veronica Shamovsky, Ralf Stephan, Krishna Tiwari, Thawfeek Varusai, Joel Weiser, Adam Wright, Guanming Wu, Lincoln Stein, Henning Hermjakob, and Peter D’Eustachio. The reactome pathway knowledgebase 2024. Nucleic Acids Research, 52(D1):D672– D678, 2024. doi: 10.1093/nar/gkad1025.

Susan A. Murphy. Optimal dynamic treatment regimes. Journal of the Royal Statistical Society Series B: Statistical Methodology, 65(2):331–355, 2003. doi: 10.1111/1467-9868.00389.

Eric Nguyen, Michael Poli, Marjan Faizi, Armin Thomas, Callum Birch-Sykes, Michael Wornow, Aman Patel, Clayton Rabideau, Stefano Massaroli, Yoshua Bengio, Stefano Ermon, Stephen A. Baccus, and Christopher Ré. HyenaDNA: Long-range genomic sequence modeling at single nucleotide resolution. In Advances in Neural Information Processing Systems, volume 36, 2023.

Chao Pang, Xinzhuo Jiang, Krishna S. Kalluri, Matthew Spotnitz, RuiJun Chen, Adler Perotte, and Karthik Natarajan. CEHR-BERT: Incorporating temporal information from structured electronic health record data to improve prediction tasks. arXiv preprint arXiv:2111.08585, 2021.

Robert C. Paule and John Mandel. Consensus values and weighting factors. Journal of Research of the National Bureau of Standards, 87(5):377–385, 1982. doi: 10.6028/jres.087.022.

Ethan Perez, Florian Strub, Harm de Vries, Vincent Dumoulin, and Aaron Courville. FiLM: Visual reasoning with a general conditioning layer. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence, pages 3942–3951. AAAI Press, 2018. doi: 10.1609/aaai.v32i1.11671. URL https://ojs.aaai.org/index.php/AAAI/article/view/11671.

Laila Rasmy, Yang Xiang, Ziqian Xie, Cui Tao, and Degui Zhi. Med-BERT: Pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. npj Digital Medicine, 4:86, 2021. doi: 10.1038/s41746-021-00455-y.

François Remy, Kris Demuynck, and Thomas Demeester. Biolord-2023: semantic textual representations fusing large language models and clinical knowledge graph insights. Journal of the American Medical Informatics Association, 31(9):1844–1855, 2024.

Pawel Renc, Yugang Jia, Anthony E. Samir, Jaroslaw Was, Quanzheng Li, David W. Bates, and Arkadiusz Sitek. Zero shot health trajectory prediction using transformer. npj Digital Medicine, 7:256, 2024. doi: 10.1038/s41746-024-01235-0.

James M Robins, Andrea Rotnitzky, and Lue Ping Zhao. Estimation of regression coeficients when some regressors are not always observed. Journal of the American statistical Association, 89(427): 846–866, 1994.

James M. Robins, Miguel A. Hernán, and Babette Brumback. Marginal structural models and causal inference in epidemiology. Epidemiology, 11(5):550–560, 2000.

Patrick Royston and Mahesh K. B. Parmar. Restricted mean survival time: an alternative to the hazard ratio for the design and analysis of randomized trials with a time-to-event outcome. BMC Medical Research Methodology, 13:152, 2013.

Eric W Sayers, Evan E Bolton, Anna M Fine, Christopher Kelly, Sunghwan Kim, Melissa Landrum, Stacy Lathrop, Adriana Malheiro, Terence D Murphy, Lon Phan, Shashikant Pujar, Barton W Trawick, Valerie A Schneider, and Kim D Pruitt. Database resources of the national center for biotechnology information in 2026. Nucleic Acids Research, 54(D1):D20–D27, 2026. doi: 10.1093/nar/gkaf1060.

Andrew B. Sellergren, Chufan Gao, Fereshteh Mahvar, Timo Kohlberger, Fayaz Jamil, Madeleine Traverse, Alberto Tono, Bashir Sadjad, Lin Yang, Charles Lau, Liron Yatziv, Tifany Chen, Bram Sterling, Kenneth Philbrick, Richa Tiwari, Yun Liu, Madhuram Jajoo, Chandrashekar Sankarapu, Swapnil Vispute, Harshad Purandare, Abhishek Bijay Mishra, Sam Schmidgall, et al. Medgemma 1.5 technical report, 2026. URL https://arxiv.org/abs/2604.05081.

Artem Shmatko, Alexander Wolfgang Jung, Kumar Gaurav, Søren Brunak, Laust Hvas Mortensen, Ewan Birney, Tom Fitzgerald, and Moritz Gerstung. Learning the natural history of human disease with generative transformers. Nature, 647(8088):248–256, 2025. doi: 10.1038/ s41586-025-09529-3.

Ethan Steinberg, Kenneth Jung, Jason A. Fries, Conor K. Corbin, Stephen R. Pfohl, and Nigam H. Shah. Language models are an efective patient representation learning technique for electronic health record data. arXiv preprint arXiv:2001.05295, 2020.

Ethan Steinberg, Jason Fries, Yizhe Xu, and Nigam Shah. MOTOR: A time-to-event foundation model for structured medical records. arXiv preprint arXiv:2301.03150, 2023.

Maximilian A. R. Strobl, Jill Gallaher, Mark Robertson-Tessi, Jefrey West, and Alexander R. A. Anderson. Treatment of evolving cancers will require dynamic decision support. Annals of Oncology, 34(10):867–884, 2023. doi: 10.1016/j.annonc.2023.08.008.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2021.

Damian Szklarczyk, Rebecca Kirsch, Maria Koutrouli, Katerina Nastou, John Fan, Nadezhda Maltsev, Anders Krogh, Milan Simonovic, Nadezhda T Doncheva, John H Morris, Peer Bork, Lars Juhl Jensen, and Christian von Mering. The string database in 2023: protein–protein association networks and functional enrichment analyses for any sequenced genome of interest. Nucleic Acids Research, 51(D1):D638–D646, 2023. doi: 10.1093/nar/gkac1000.

The Gene Ontology Consortium, Shur V Aleksander, Jim Balhof, Seth Carbon, J Michael Cherry, Harold J Drabkin, Douglas Ebert, Lars Feuerbach, Michael Frampton, Christopher MA Harris, et al. The gene ontology knowledgebase in 2023. Genetics, 224(1):iyad031, 2023. doi: 10.1093 genetics/iyad031.

Christina V. Theodoris, Ling Xiao, Anant Chopra, Mark D. Chafin, Zeina R. Al Sayed, Matthew C. Hill, Helene Mantineo, Elizabeth M. Brydon, Zexian Zeng, X. Shirley Liu, and Patrick T. Ellinor. Transfer learning enables predictions in network biology. Nature, 618:616–624, 2023. doi: 10. 1038/s41586-023-06139-9.

Hajime Uno, Tianxi Cai, Michael J. Pencina, Ralph B. D’Agostino, and L. J. Wei. On the cstatistics for evaluating overall adequacy of risk prediction procedures with censored survival data. Statistics in Medicine, 30(10):1105–1117, 2011. doi: 10.1002/sim.4154.

Anurag Vaidya et al. Molecular-driven foundation model for oncologic pathology (THREADS), 2025. URL https://arxiv.org/abs/2501.16652.

Luís A. Vale-Silva and Karl Rohr. Long-term cancer survival prediction using multimodal deep learning. Scientific Reports, 11:13505, 2021. doi: 10.1038/s41598-021-92799-4.

Eugene Vorontsov, Alican Bozkurt, Adam Casson, George Shaikovski, Michal Zelechowski, Kristen Severson, Eric Zimmermann, James Hall, Neil Tenenholtz, Nicolo Fusi, Ellen Yang, Philippe Mathieu, Alexander van Eck, Donghun Lee, Julian Viret, Eric Robert, Yi Kan Wang, Jeremy D. Kunz, Matthew C. H. Lee, Jan H. Bernhard, Ran A. Godrich, Gerard Oakley, Ewan Millar, Matthew Hanna, Hannah Wen, Juan A. Retamero, William A. Moye, Razik Yousfi, Christopher Kanan, David S. Klimstra, Brandon Rothrock, Siqi Liu, and Thomas J. Fuchs. A foundation model for clinical-grade computational pathology and rare cancers detection. Nature Medicine, 30(10):2924–2935, 2024. doi: 10.1038/s41591-024-03141-0.

Eugene Vorontsov, George Shaikovski, Adam Casson, Julian Viret, Eric Zimmermann, Neil Tenenholtz, Yi Kan Wang, Jan H. Bernhard, Ran A. Godrich, Juan A. Retamero, Jinru Shia, Mithat Gonen, Martin R. Weiser, David S. Klimstra, Razik Yousfi, Nicolò Fusi, Thomas J. Fuchs, Kristen Severson, and Siqi Liu. End-to-end multimodal pathology foundation model with clinical dialogue. Nature Medicine, 2026. doi: 10.1038/s41591-026-04521-4. URL https: //www.nature.com/articles/s41591-026-04521-4.

Kexin Wang, Nils Reimers, and Iryna Gurevych. TSDAE: Using transformer-based sequential denoising auto-encoder for unsupervised sentence embedding learning. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 671–688, Punta Cana, Dominican Republic, 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.findings-emnlp.59. URL https://aclanthology.org/2021.findings-emnlp.59/.

Yi Kan Wang, Ludmila Tydlitatova, Jeremy D. Kunz, Gerard Oakley, Bonnie Kar Bo Chow, Ran A. Godrich, Matthew C. H. Lee, Hamed Aghdam, Alican Bozkurt, Michal Zelechowski, Chad Vanderbilt, Christopher Kanan, Juan A. Retamero, Peter Hamilton, Razik Yousfi, Thomas J. Fuchs, David S. Klimstra, and Siqi Liu. Screen them all: High-throughput pan-cancer genetic and phenotypic biomarker screening from h&e whole slide images, 2024. URL https: //arxiv.org/abs/2408.09554.

Shane Waxler, Paul Blazek, Davis White, Daniel Sneider, Kevin Chung, Mani Nagarathnam, Patrick Williams, Hank Voeller, Karen Wong, Matthew Swanhorst, Sheng Zhang, Naoto Usuyama, Clif

Wong, Tristan Naumann, Hoifung Poon, Andrew Loza, Daniella Meeker, Seth Hain, and Rahul Shah. Generative medical event models improve with scale. arXiv preprint arXiv:2508.12104, 2025.

M Whirl-Carrillo, R Huddart, L Gong, K Sangkuhl, CF Thorn, R Whaley, and TE Klein. An evidence-based framework for evaluating pharmacogenomics knowledge for personalized medicine. Clinical Pharmacology & Therapeutics, 110(3):563–572, 2021. doi: 10.1002/cpt.2350.

David S Wishart, Yannick D Feunang, An C Guo, Elvis J Lo, Ana Marcu, Jason R Grant, Tanvir Sajed, Daniel Johnson, Carin Li, Zinat Sayeeda, et al. Drugbank 5.0: a major update to the drugbank database for 2018. Nucleic acids research, 46(D1):D1074–D1082, 2018.

S. Xu, C. Ross, M. A. Raebel, S. Shetterly, C. Blanchette, and D. Smith. Use of stabilized inverse propensity scores as weights to directly estimate relative risk and its confidence intervals. Value in Health, 13(2):273–277, 2010.

Yingxue Xu, Yihui Wang, Fengtao Zhou, Jiabo Ma, Cheng Jin, Shu Yang, Jinbang Li, Zhengyu Zhang, et al. A multimodal knowledge-enhanced whole-slide pathology foundation model (mSTAR). Nature Communications, 2025. doi: 10.1038/s41467-025-66220-x. URL https: //www.nature.com/articles/s41467-025-66220-x.

Steve Yadlowsky, Scott Fleming, Nigam Shah, Emma Brunskill, and Stefan Wager. Evaluating treatment prioritization rules via rank-weighted average treatment efects. Journal of the American Statistical Association, 120(549), 2025. doi: 10.1080/01621459.2024.2393466.

Xi Yang, Aokun Chen, Nima PourNejatian, Hoo Chang Shin, Kaleb E. Smith, Christopher Parisien, Colin Compas, Cheryl Martin, Anthony B. Costa, Mona G. Flores, Ying Zhang, Tanja Magoc, Christopher A. Harle, Gloria Lipori, Duane A. Mitchell, William R. Hogan, Elizabeth A. Shenkman, Jiang Bian, and Yonghui Wu. A large language model for electronic health records. npj Digital Medicine, 5:194, 2022. doi: 10.1038/s41746-022-00742-2.

Deborah A Zarin, Tony Tse, Rebecca J Williams, Robert M Calif, and Nicholas C Ide. The clinicaltrials.gov results database—update and key issues. New England Journal of Medicine, 364 (9):852–860, 2011. doi: 10.1056/NEJMsa1012065.

Andrew Zhang, Tong Ding, Sophia J. Wagner, Caiwei Tian, Ming Y. Lu, Rowland Pettit, Joshua E. Lewis, Alexandre Misrahi, Dandan Mo, Long Phi Le, and Faisal Mahmood. A multimodal and temporal foundation model for virtual patient representations at healthcare system scale. arXiv preprint arXiv:2604.18570, 2026.

T. Zhou, D. Fu, M. Soltanolkotabi, R. Jia, and V. Sharan. Fone: Precise single-token number embeddings via fourier features. In International Conference on Learning Representations, 2026.

Zhihan Zhou, Yanrong Ji, Weijian Li, Pratik Dutta, Ramana V. Davuluri, and Han Liu. DNABERT-2: Eficient foundation model and benchmark for multi-species genome. In International Conference on Learning Representations, 2024.

Eric Zimmermann, Eugene Vorontsov, Julian Viret, Adam Casson, Michal Zelechowski, George Shaikovski, Neil Tenenholtz, James Hall, David Klimstra, Razik Yousfi, et al. Virchow2: Scaling self-supervised mixed magnification models in pathology. arXiv preprint arXiv:2408.00738, 2024.

## A Appendix

## A.1 Data Sources

To build the patient trajectories, we used data from the Tempus Data Model (TDM), a multimodal oncology data platform that integrates clinical and molecular data from multiple sources, as detailed below.

## Clinical Data

• Curated. Structured clinical data manually abstracted by trained experts from patient medical records, including progress notes, pathology reports, and treatment plans. Abstraction followed standardized guidelines to capture diagnosis, treatment (including care plans, regimens, and lines of therapy), response assessments, disease progression, and other clinical endpoints.

• Native. Structured data from electronic health records (EHRs) transmitted directly from partner healthcare institutions, including medication administrations and orders, laboratory results, vital signs, encounters, performance status, and smoking status.

• Pathology. Diagnosis and histology information derived from Tempus pathologist review of biopsy specimens.

• Third-party. Supplementary data from external sources, including claims-based mortality data, used to augment vital status and last-known-alive dates.

• Derived. Algorithmically computed variables, including BMI calculated from height and weight measurements, and lines of therapy derived from curated care plan data.

In the case where a clinical concept is captured by both curated and native data (e.g., antineoplastic medications), records were harmonized to maximize completeness and date precision, with the source of each record tracked via a row\_source indicator.

## Molecular Data

• Tempus-generated molecular data. Results from Tempus next-generation sequencing (NGS) assays (xT, xF, xE, RS), including single nucleotide variants (SNVs), insertions/deletions (indels), copy number variants (CNVs), structural variants (SVs), RNA gene expression, microsatellite instability (MSI), tumor mutational burden (TMB), homologous recombination deficiency (HRD), HLA genotyping, immune infiltration, immune repertoire, neoepitope predictions, and microorganism detection.

• Third-party reported molecular data. Results from molecular tests performed by external laboratories, including NGS, PCR, immunohistochemistry (IHC), and fluorescence in situ hybridization (FISH), capturing biomarker results such as hormone receptor status (ER, PR, HER2), MSI, mismatch repair (MMR), and gene-level alterations.

• Reference laboratory data: Results from Tempus partner reference laboratories (e.g., minimal residual disease [MRD] testing, IHC/FISH).

## De-identification

All data were de-identified using HIPAA-compliant statistical de-identification (Expert Determination) methods. Dates were shifted uniformly per patient (within a range of 0 − 30 days) to preserve temporal relationships between clinical events while protecting patient identity.

## A.2 Molecular blueprint

For each biospecimen, a molecular blueprint was assembled from DNA, RNA, IHC, and ISH, integrating ten categories:

• Biospecimen metadata detailing histology and anatomical site,

• IHC biomarkers including PD-L1 TPS, minimal residual disease (MRD), and HER2 status,

• Genomic instability indicators encompassing MSI/dMMR status, tumor mutational burden (TMB-H ≥ 10 mutations/Mb), homologous recombination deficiency (HRD), HLA loss of heterozygosity (LOH), and clonal heterogeneity,

• Host immune descriptors comprising TIL phenotype, CD8/CD4 dominance, immune likelihood, and 20 HLA single-allele hotspots plus 9 multi-locus haplotypes across Class I and II loci,

• High-confidence pathogen calls,

• Disruption in five canonical oncogenic signaling pathways (mTOR, TGF-β, RTK, HRD, DDR),

• Pathogenic or likely pathogenic SNVs/indels with complete genomic and protein annotations,

• Copy-number alterations and fraction of genome altered,

• Structural variants or fusions, and

• RNA-level expression abnormalities alongside 11 actionable splicing events (e.g., MET exon 14 skipping, AR-V7, EGFRvIII).

## A.3 Prognostic classification: cohort stratification (tumor × therapy)

Cohort stratification. Strata are defined as general\_tumor\_type × therapy\_class\_group, restricted to a whitelist of valid pairs; a stratum is dropped only if (a) it has zero embedded patients, or (b) it contains a single class (all positive or all negative). The reported metric is the unweighted macro-mean of per-stratum AUCs. This stratification tests whether performance is consistent across clinically comparable tumor–therapy combinations rather than being driven by a few high-prevalence cohorts. Because the macro-mean weights each scored stratum equally, results are sensitive to small strata; Table A1 provides the per-stratum sample sizes used for each task.

Table A1: Per-stratum composition by tumor type and therapy group, for the strata entering the prognostic-classification analysis (minimum 20 patients per stratum, as in the figure). Values are patients, n (positive cases). An em dash indicates the stratum did not enter that task: fewer than 20 patients, a single outcome class, or no available embeddings.
<table><tr><td>Tumor type</td><td>Therapy group</td><td>OS n (pos)</td><td>PFS n (pos)</td><td>Response n (pos)</td></tr><tr><td>Colorectal cancer</td><td>Biologic + chemotherapy</td><td>383 (285)</td><td>355 (280)</td><td>451 (196)</td></tr><tr><td>NSCLC</td><td>Chemotherapy + IO/ICI</td><td>325 (207)</td><td>327 (214)</td><td>450 (181)</td></tr><tr><td>Breast cancer</td><td>Hormone + targeted (SM)</td><td>239 (203)</td><td>279 (214)</td><td>277 (94)</td></tr><tr><td>Colorectal cancer</td><td>Chemotherapy</td><td>282 (226)</td><td>248 (199)</td><td>240 (100)</td></tr><tr><td>Pancreatic cancer</td><td>Chemotherapy</td><td>274 (148)</td><td>205 (122)</td><td>281 (62)</td></tr><tr><td>Prostate cancer</td><td>Hormone</td><td>299 (290)</td><td>326 (287)</td><td>100 (44)</td></tr><tr><td>Ovarian/fallopian</td><td>Chemotherapy</td><td>201 (193)</td><td>199 (189)</td><td>114 (79)</td></tr><tr><td>tube/peritoneal cancer</td><td></td><td></td><td></td><td></td></tr><tr><td>NSCLC</td><td>IO/ICI</td><td>114 (67)</td><td>103 (68)</td><td>113 (41)</td></tr><tr><td>NSCLC</td><td>TKI</td><td>99 (74)</td><td>102 (74)</td><td>115 (50)</td></tr><tr><td>Tumor of unknown origin</td><td>Chemotherapy</td><td>114 (63)</td><td>98 (56)</td><td>83 (29)</td></tr><tr><td>Breast cancer</td><td>Chemotherapy</td><td>100 (65)</td><td>96 (55)</td><td>92 (31)</td></tr><tr><td>Melanoma</td><td>IO/ICI</td><td>82 (61)</td><td>94 (71)</td><td>90 (32)</td></tr><tr><td>Gastroesophageal cancer</td><td>Chemotherapy</td><td>96 (39)</td><td>63 (37)</td><td>97 (32)</td></tr><tr><td>NSCLC</td><td>Chemotherapy</td><td>107 (68)</td><td>83 (57)</td><td>65 (23)</td></tr><tr><td>Prostate cancer</td><td>Hormone + other</td><td>88 (81)</td><td>95 (84)</td><td>55 (27)</td></tr><tr><td>Gastroesophageal cancer</td><td>Chemotherapy + IO/ICI</td><td>67 (44)</td><td>65 (41)</td><td>84 (27)</td></tr><tr><td>Tumor of unknown origin</td><td>Chemotherapy + IO/ICI</td><td>60 (32)</td><td>63 (34)</td><td>78 (23)</td></tr><tr><td>Prostate cancer</td><td>Chemotherapy + hormone</td><td>64 (59)</td><td>71 (60)</td><td>52 (25)</td></tr><tr><td>Ovarian/fallopian</td><td>Chemotherapy + targeted (SM)</td><td>49 (45)</td><td>61 (52)</td><td>66 (50)</td></tr><tr><td>tube/peritoneal cancer</td><td></td><td></td><td></td><td></td></tr><tr><td>Breast cancer</td><td>Hormone</td><td>66 (52)</td><td>62 (49)</td><td>31 (16)</td></tr><tr><td>Urinary tract cancer</td><td>Chemotherapy</td><td>52 (29)</td><td>49 (25)</td><td>53 (20)</td></tr><tr><td>Endometrial cancer</td><td>Chemotherapy</td><td>54 (41)</td><td>54 (45)</td><td>39 (25)</td></tr><tr><td>SCLC</td><td>Chemotherapy + IO/ICI</td><td>37 (19)</td><td>30 (13)</td><td>77 (31)</td></tr><tr><td>Kidney cancer</td><td>IO/ICI</td><td>42 (33)</td><td>52 (39)</td><td>48 (13)</td></tr><tr><td>Ovarian/fallopian</td><td>Biologic + chemotherapy</td><td>51 (44)</td><td>50 (47)</td><td>41 (24)</td></tr><tr><td>tube/peritoneal cancer Lung cancer</td><td>Chemotherapy + IO/ICI</td><td></td><td></td><td></td></tr><tr><td>Kidney cancer</td><td>IO/ICI + TKI</td><td>36 (14)</td><td>35 (18)</td><td>53 (15)</td></tr><tr><td>Breast cancer</td><td>Biologic + chemotherapy</td><td>35 (28)</td><td>39 (29)</td><td>48 (19)</td></tr><tr><td>Breast cancer</td><td></td><td>32 (28)</td><td>42 (35)</td><td>36 (20)</td></tr><tr><td>Colorectal cancer</td><td>Chemotherapy + IO/ICI</td><td>29 (20)</td><td>31 (24)</td><td>44 (12)</td></tr><tr><td>Urinary tract cancer</td><td>IO/ICI</td><td>36 (30)</td><td>35 (30)</td><td>32 (16)</td></tr><tr><td>Gallbladder/biliary tract</td><td>ADC + IO/ICI Chemotherapy</td><td>34 (27) 27 (12)</td><td>33 (29) 25 (13)</td><td>33 (14) 43 (12)</td></tr><tr><td>cancer</td><td></td><td></td><td></td><td></td></tr><tr><td>Gallbladder/biliary tract cancer</td><td>Chemotherapy + IO/ICI</td><td>34 (25)</td><td>28 (18)</td><td>29 (6)</td></tr><tr><td>Endometrial cancer</td><td>Chemotherapy + IO/ICI</td><td>25 (18)</td><td>29 (24)</td><td>31 (14)</td></tr><tr><td>Urinary tract cancer</td><td>IO/ICI</td><td>26 (14)</td><td>24 (13)</td><td>31 (8)</td></tr></table>

Continued on the next page

Table A1 continued from the previous page
<table><tr><td>Tumor type</td><td>Therapy group</td><td>OS n (pos)</td><td>PFS n (pos)</td><td>Response n (pos)</td></tr><tr><td>Gastroesophageal cancer</td><td>Biologic + chemotherapy</td><td>23 (16)</td><td>21 (16)</td><td>27 (10)</td></tr><tr><td>Gastroesophageal cancer</td><td>Biologic + chemotherapy + IO/ICI</td><td>20 (14)</td><td>22 (15)</td><td>29 (11)</td></tr><tr><td>Head and neck SCC</td><td>Chemotherapy + IO/ICI</td><td>24 (13)</td><td></td><td>37 (6)</td></tr><tr><td>Prostate cancer</td><td>Chemotherapy + hormone + other</td><td>20 (19)</td><td>23 (20)</td><td></td></tr><tr><td>Urinary tract cancer</td><td>Chemotherapy + IO/ICI</td><td></td><td></td><td>39 (11)</td></tr><tr><td>Ovarian/fallopian</td><td>Biologic + chemotherapy + targeted</td><td></td><td></td><td>21 (15)</td></tr><tr><td>tube/peritoneal cancer</td><td>(SM)</td><td></td><td></td><td></td></tr></table>

Abbreviations: ADC, antibody-drug conjugate; ICI, immune checkpoint inhibitor; IO, immuno-oncology; SM, small molecule; TKI, tyrosine kinase inhibitor.

## A.4 Baseline Feature Selection

7520 candidate features were drawn from a versioned catalogue of standardized, category-annotated clinical and molecular features. Treatment intent labels (adjuvant, neoadjuvant, induction, consolidation, conditioning, and maintenance) were excluded across all configurations to prevent treatment assignment leakage. Feature selection was performed independently for each cohort/task using only the training and validation data; test data were strictly withheld.

Candidate features passed through three successive selection stages:

1. Variance filtering: Features with zero variance or falling below the 70th percentile of candidate variance were removed.

2. Univariate screening: Remaining candidates were evaluated in individual penalized Cox proportionalhazards models (penalty = 0.01, elastic-net mixing = 0.9) incorporating delayed entry. Features were retained if the concordance index exceeded 0.5 and the Wald p-value was below 0.05. For cohorts exceeding 10,000 patients, median concordance indices and p-values across 10 fixed-seed subsamples of 10,000 patients were used to stabilize estimates.

3. Multivariate selection: Retained candidates were fit jointly in a single penalized Cox model; features with absolute coeficients below 0.01 were discarded.

A core set of base clinical features was retained across all baselines regardless of statistical significance, subject only to variance filtering, to preserve cross-cohort comparability: ECOG performance status (0–5), four age bands, female sex, tumor grade (1–4), and stage (0–IV). Missing or constant levels within a cohort were dropped. Continuous variables were normalized with canonical standardization (mean, standard deviation on the training set) and were imputed to zero. Combining the selected candidate features with these base features yields the final compact, cohort-specific feature set. Tab. A2 shows the number of features selected per predictive cohort.

<table><tr><td colspan="2">Cohort Selected features</td></tr><tr><td>Breast HR+/HER2- CDK4/6 and aromatase inhibitor (AI) comb. vs. AI alone, 1L</td><td>104</td></tr><tr><td>CRC RAS WT anti-EGFR vs. anti-VEGF, 1L</td><td>197</td></tr><tr><td>NSCLC pembro-chemo combo vs. chemo, 1L</td><td>94</td></tr><tr><td>NSCLC pembro-chemo combo vs. pembro alone, 1L</td><td>90</td></tr><tr><td>NSCLC dual IO vs. IO–chemo combo, 1L</td><td>81</td></tr><tr><td>mRCC dual IO vs. IO–TKI, 1L</td><td>160</td></tr><tr><td>HSPC ADT doublet vs. triplet, 1L</td><td>114</td></tr><tr><td>Enriched multi-tumor Enhertu vs. chemo, 2L+</td><td>162</td></tr><tr><td>Pancancer Enhertu vs. chemo, 2L+</td><td>149</td></tr><tr><td>Pancancer IO-chemo vs. IO alone, 1L</td><td>165</td></tr><tr><td>Pancancer HRR+ PARPi vs. chemo, 2L+</td><td>111</td></tr></table>

Table A2: Number of selected baseline features per predictive evaluation cohort.

## A.5 Inverse-Propensity Weighting

We use stabilized inverse-propensity weighting to correct treatment selection bias in the observational cohorts [Robins et al., 2000, Austin and Stuart, 2015]. The propensity model is trained on a set of always-included features and a subset of the features selected as described in Appendix A.4. The always-included features are one-hot ECOG performance status (0–5), age band (≤30, 30–50, 50–75, >75), gender, tumor grade (1–4), and tumor stage (0–4). Selected features may include IHC biomarkers, tumor type, surgical pathology outcomes, smoking status, and biopsy vs resection. Treatment information is dropped. A logistic model on this feature set estimates the propensity $e ( \mathbf { x } _ { i } ) = { \hat { P } } ( W _ { i } = 1 \mid \mathbf { x } _ { i } )$ , clipped to [0.01, 0.99], and the stabilized weights [Xu et al., 2010] are calculated from propensity scores

$$
w _ { i } = \left\{ \begin{array} { l l } { \hat { P } ( W = 1 ) / e ( \mathbf { x } _ { i } ) } & { W _ { i } = 1 , } \\ { \hat { P } ( W = 0 ) / \left( 1 - e ( \mathbf { x } _ { i } ) \right) } & { W _ { i } = 0 , } \end{array} \right.\tag{8}
$$

where $\hat { P } ( W = 1 )$ is the treated fraction of the cohort and $\hat { P } ( W = 0 ) = 1 - \hat { P } ( W = 1 )$ . These weights enter the Cox partial log-likelihood when training the UV-Cox estimator (Appendix A.6). The propensity scores are reused with augmented inverse-propensity weighting (AIPW) during evaluation [Robins et al., 1994].

## A.6 Two-Stage UV-Cox Predictive Evaluation Design

To evaluate treatment benefit in our benchmarks, we set up and fit a convex, linear Cox [Cox, 1972] model that represents the residual treatment-dependent risk beyond treatment-independent prognostic risk. Let $W _ { i } \in \{ 0 , 1 \}$ be the treatment assignment. For $\mathbf { x } _ { i } \in \mathbb { R } ^ { p }$ , a representation of patient i (the oFM embedding $h ( t ^ { * } )$ , or the selected features for the baseline), we split the linear predictor $\eta _ { i }$ into a prognostic term and a treatment-modifying term:

$$
\begin{array} { r } { \eta _ { i } = u ( \mathbf { x } _ { i } ) + W _ { i } v ( \mathbf { x } _ { i } ) , \qquad } \\ { \lambda ( t \mid \mathbf { x } _ { i } , W _ { i } ) = \lambda _ { 0 } ( t ) \exp ( u ( \mathbf { x } _ { i } ) + W _ { i } v ( \mathbf { x } _ { i } ) ) , } \end{array}\tag{9}
$$

where λ measures patient risk and $\lambda _ { 0 }$ defines baseline risk. Here u is the prognostic head which carries the risk present in both arms and v is the predictive head which carries the residual treatmentdependent risk. The treatment-benefit score is $\tau ( \mathbf { x } _ { i } ) = - v ( \mathbf { x } _ { i } )$ , so $v ( \mathbf { x } _ { i } ) < 0$ predicts a lower hazard under treatment. Both heads are linear with feature-weights $\beta _ { i }$ , and $b _ { v }$ modeling the main treatment efect in the treatment head:

$$
\begin{array} { r l } & { u ( \mathbf { x } _ { i } ) = \boldsymbol { \beta } _ { u } ^ { \top } \tilde { \mathbf { x } } _ { i } , } \\ & { v ( \mathbf { x } _ { i } ) = \boldsymbol { \beta } _ { v } ^ { \top } \tilde { \mathbf { x } } _ { i } + b _ { v } . } \end{array}\tag{10}
$$

Features are z-scored with moments computed on training and validation data, not the test set. Both heads are fit by minimizing a weighted negative log partial likelihood with L2 regularization,

$$
\ell ( \eta ) = - \frac { 1 } { \sum _ { j : \delta _ { j } = 1 } w _ { j } } \sum _ { i : \delta _ { i } = 1 } w _ { i } \left[ \eta _ { i } - \log \sum _ { j \in R ( T _ { i } ) } w _ { j } e ^ { \eta _ { j } } \right] , \qquad R ( t ) = \{ j : L _ { j } < t \leq T _ { j } \} ,\tag{11}
$$

where $\delta _ { i } \in \{ 0 , 1 \}$ is the event indicator and the risk set $R ( t )$ enforces delayed entry, so a patient contributes only after the left-truncation entry time $L _ { i }$ [Klein and Moeschberger, 2003]. To avoid leaking prognostic signal into the predictive head, fitting is done in two stages. With the predictive head v is initialized to zero, stage 1 fits only the prognostic head u. Stage 2 then freezes the prognostic head u and fits the predictive head v.

Each stage is optimized with full-batch L-BFGS. The L2 penalty is selected with 5-fold crossvalidation over the union of the training and validation sets. In stage 1, L2 is selected over a 22-point logarithmic grid on $[ 1 0 ^ { - 4 } , 1 0 ^ { 3 } ]$ . The model is refit on the full train + validation set at the selected L2 penalty. In stage 2, the same L2 penalty is used, as lowering it risks leaking prognostic signal into $v$ and raising it shrinks $v ,$ reducing treatment-dependent signal.

## A.7 Evaluation Metrics

We evaluate predictive performance of oFM embeddings or baseline features with AUTOC and HRR and prognostic performance with C-index.

AUTOC. When ranking patients by predicted treatment benefit, the area under the targeting operator characteristic (AUTOC) is the average additional benefit compared to the cohort as a whole, computed over all possible top percentiles of patients. It is zero when the ranking carries no information about who benefits. The treatment benefit score $- v ( { \bf x } _ { i } )$ is estimated by the UV-Cox method (see Appendix A.6). However, ground truth benefit is never observed per patient as patients only ever occur in one arm and observation is right-censored. To estimate it, a pseudo-outcome $\theta _ { i }$ is first computed per-patient using the restricted mean survival time (RMST) which is the average event-free time within a horizon $L$ [Royston and Parmar, 2013, Andersen et al., 2003]. We set L to the 80th percentile of observed follow-up, or the administrative censoring time where one is defined. Each patient’s ground truth treatment benefit $\psi _ { i }$ is then estimated from the pseudo-outcome $\theta _ { i }$ with a doubly robust AIPW score [Robins et al., 1994], combining the propensity e of Appendix A.5 with arm-specific ridge outcome models fit on train+validation and applied to the test patients. Ranking patients by estimated benefit, the targeting operator characteristic (TOC) and its integral are

$$
\mathrm { T O C } ( q ) = \frac { 1 } { \lceil q n \rceil } \sum _ { i \in \mathrm { t o p } - q } \psi _ { i } - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \psi _ { i } , \quad \quad \mathrm { A U T O C } = \int _ { 0 } ^ { 1 } \mathrm { T O C } ( q ) d q ,\tag{12}
$$

which weights the ranking uniformly [Yadlowsky et al., 2025].

AUTOC is measured in time and the scale of AUTOC depends on the RMST and observational horizon of each cohort. We normalize AUTOC by SD where SD is the unweighted standard deviation of $\theta ,$ representing the relative scale of each cohort. Statistical significance is computed against a permutation null that randomly reshufles treatment assignment within propensity-score bins; the p-value is the fraction of replicates whose AUTOC meets or exceeds the observed value.

Hazard ratio and hazard-ratio ratio. Whereas a hazard ratio (HR) measures how much treatment helps a group of patients, the hazard-ratio ratio (HRR) divides the hazard ratio of the estimated high-benefit patients by that of the estimated low-benefit patients. An HRR below one indicates correct separation according to treatment efect and an HRR above one indicates the ranking is inverted [Iasonos et al., 2016]. Test patients are split into low- and high-benefit groups at the median benefit score, and a newly fit Cox model on the treatment indicator alone gives each stratum’s HR using the IPW of Appendix A.5. Statistical significance is computed by a two-sided Wald test of log HRR = 0.

Prognostic discrimination. We report Uno’s IPCW concordance index (C-index) [Uno et al., 2011] of the prognostic score $u ( \mathbf { x } _ { i } )$ . Across all patient pairs, it weights each pair to correct for right censoring and keeps only those that delayed entry leaves comparable. It is computed within each treatment arm so that it reflects baseline risk rather than treatment efect. Unlike in the predictive setting, the prognostic-only model is fit on both arms pooled without IPW, which is needed only in the predictive setting to correct treatment selection bias. The C-index is then computed within each arm so that it reflects baseline risk rather than treatment efect. C-index 0.5 and 1 indicate random chance and correct ordering, respectively. The 95% bootstrap confidence interval is computed over 1,000 iterations.

Pooling across cohorts. Per-cohort efects $y _ { i }$ with standard errors $e _ { i }$ are combined by Paule-Mandel random-efects meta-analysis [Paule and Mandel, 1982], which weights cohort i by $1 / ( e _ { i } ^ { 2 } { + \tau ^ { 2 } } )$ for a between-cohort variance $\tau ^ { 2 }$ estimated from the data, so noisier cohorts are weighted less. µ<sub>AUTOC</sub> is the pooled AUTOC/SD. Dividing it by its standard error $\mathrm { S E } _ { \mathrm { A U T O C } } \ \mathrm { g i v e s } \ t _ { \mathrm { A U T O C / S D } } ,$ which measures both efect size (since SE<sub>AUTOC</sub> grows with per-cohort noise) and consistency across cohorts (since SE<sub>AUTOC</sub> grows with disagreement between cohorts). HRR is pooled on the log scale and transformed back and C-index is pooled on logits. The significance of pooled AUTOC/SD and HRR results is tested against their permutation nulls where the per-patient treatment arm assignment (i.e. control v.s. treatment.) is randomly permuted. The statistical significance of the diference between the oFM and baseline pooled C-index values is computed with a Student t-test.

## A.8 Survival benchmark cohorts

Table A3: Survival-benchmark cohorts. Two-arm comparative cohorts evaluated for diferential treatment benefit on overall survival. Each row gives the cohort (with line-of-therapy and clinicaluse-case tags), the experimental vs. control regimens, the clinical question (condensed to its primary clause), and the number of experimental / control patients in the combined train+validation (modeldevelopment) and held-out test splits. Patient counts are for the genomic (DNA+RNA) subset.

<table><tr><td>Cohort</td><td>Experimental / Control arm</td><td>Clinical question</td><td>Train+val (exp /ctrl) (exp /ctrl)</td><td>Test</td></tr><tr><td>Breast HR+/HER2- CDK4/6 and Aromatase inhibitors (AI) comb vs AI alone 1L</td><td>Exp: CDK4/6 inhibitor plus aromatase inhibitor (AI) Ctrl: aromatase inhibitor alone</td><td>Does adding a CDK4/6 inhibitor (ribociclib, palbociclib, or abemaciclib) to first-line aromatase inhibitor therapy improve progression-free and overall survival in HR+/HER2- metastatic breast cancer?</td><td>814/146</td><td>291/63</td></tr><tr><td>CRC RAS WT anti-EGFR vs anti-VEGF 1L</td><td>Exp: anti-EGFR antibody (cetuximab or panitumumab) combined with FOLFOX or FOLFIRI chemotherapy Ctrl: bevacizumab combined with FOLFOX</td><td>In RAS wild-type metastatic colorectal cancer, does first-line anti-EGFR antibody plus chemotherapy improve overall survival versus bevacizumab plus chemotherapy?</td><td>204/427</td><td>72/144</td></tr><tr><td>NSCLC Pembro Chemo combo vs Chemo 1L</td><td>Exp: pembrolizumab plus platinum (carboplatin or cisplatin) plus pemetrexed Ctrl: platinum plus pemetrexed (chemotherapy only)</td><td>In driver-negative metastatic NSCLC, does adding pembrolizumab to platinum plus pemetrexed improve overall survival versus chemotherapy alone, and across PD-L1 subgroups (&lt;1%, 1–49%, ≥50%), which patients derive the greatest incremental benefit?</td><td>1034/113</td><td>323/30</td></tr><tr><td>NSCLC Pembro Chemo combo vs Pembro alone 1L</td><td>Exp: pembrolizumab plus platinum-based chemotherapy Ctrl: pembrolizumab monotherapy</td><td>In PD-L1 ≥50% driver-negative metastatic NSCLC, does adding chemotherapy to first-line pembrolizumab improve outcomes versus pembrolizumab monotherapy?</td><td>259/311</td><td>87/101</td></tr></table>

Continued on next page

Table A3 – continued from previous page
<table><tr><td>Cohort</td><td>Experimental / Control arm</td><td>Clinical question</td><td colspan="2"></td></tr><tr><td></td><td></td><td></td><td>Train+val (exp /ctrl)</td><td>Test (exp / ctrl)</td></tr><tr><td>NSCLC dual IO vs IO Chemo combo 1L</td><td>Exp: chemo-immunotherapy with pembrolizumab plus platinum plus pemetrexed or paclitaxel (KEYNOTE-189 style) Ctrl: dual checkpoint immunotherapy with ipilimumab plus nivolumab,</td><td>In driver-negative metastatic NSCLC, does first-line chemo-immunotherapy (pembrolizumab-based) outperform dual checkpoint blockade (ipilimumab plus nivolumab)?</td><td>1231/111</td><td>398/54</td></tr><tr><td>mRCC dual IO vs IO-TKI 1L</td><td>Exp: IO-TKI combinations (pembrolizumab plus axitinib, lenvatinib plus pembrolizumab, cabozantinib plus nivolumab, or avelumab plus axitinib) Ctrl: dual checkpoint</td><td>In metastatic renal cell carcinoma, does first-line IO-TKI combination therapy improve outcomes versus dual checkpoint immunotherapy (ipilimumab plus nivolumab)?</td><td></td><td></td></tr><tr><td>HSPC ADT doublet vs triplet 1L</td><td>Exp: ADT plus androgen receptor pathway inhibitor (ARPi) plus docetaxel (triplet therapy) Ctrl: ADT plus ARPi</td><td>In metastatic hormone-sensitive prostate cancer, does adding docetaxel to ADT plus ARPi (triplet intensification) improve overall survival versus ADT plus ARPi alone (doublet)?</td><td>278 /1180</td><td>105/373</td></tr><tr><td>Enriched multi-tumor Enhertu vs Chemo 2L+</td><td>Exp: T-DXd (Enhertu) Ctrl: standard chemotherapy</td><td>Across breast, gastric, and NSCLC cancers previously treated with trastuzumab and a taxane, does T-DXd improve progression-free and overall survival versus standard</td><td>231/447</td><td>55/148</td></tr><tr><td>Pancancer Enhertu vs Chemo 2L+</td><td>Exp: T-DXd (Enhertu trastuzumab deruxtecan) Ctrl: standard chemotherapy (irinotecan, capecitabine, eribulin,</td><td>Across metastatic cancers previously treated with trastuzumab and a taxane, does T-DXd improve progression-free and overall survival versus</td><td>310 / 2125</td><td>82/702</td></tr><tr><td>Pancancer IO Chemo vs IO alone 1L</td><td>Exp: immune checkpoint inhibitor (ICI) plus chemotherapy Ctrl: ICI alone</td><td>Across metastatic cancers, does adding chemotherapy to first-line immune checkpoint inhibitor therapy improve outcomes versus ICI</td><td>3620/1877</td><td>1229 /657</td></tr></table>

Continued on next page

Table A3 – continued from previous page
<table><tr><td>Cohort</td><td>Experimental / Control arm</td><td>Clinical question</td><td>Train+val (exp / ctrl) (exp / ctrl)</td><td>Test</td></tr><tr><td>Pancancer HRR+ PARPi vs Chemo 2L+</td><td>Exp: PARP inhibitor (olaparib, niraparib, rucaparib, or talazoparib) Ctrl: chemotherapy (docetaxel, paclitaxel, or</td><td>Across metastatic cancers with a homologous-recombination- repair (HRR) mutation or HRD, second-line or later, does a PARP inhibitor improve overall survival versus chemotherapy?</td><td>224/532</td><td>82/200</td></tr></table>

## A.9 Model parameter counts

<table><tr><td>Component</td><td>Parameters</td></tr><tr><td>Trained parameters: oFM</td><td></td></tr><tr><td>Episode encoder (GatorTron-base-2k + FoNE)</td><td>358M</td></tr><tr><td>transformer layers</td><td>303M</td></tr><tr><td>token-embedding matrix (50,102 × 1024)</td><td>51.3M</td></tr><tr><td>position/type embeddings + LayerNorm</td><td>2.1M</td></tr><tr><td>FoNE numeric embedder</td><td>1.1M</td></tr><tr><td>Trajectory encoder (2-layer, RoPE)</td><td>25.2M</td></tr><tr><td>Heads and projectors</td><td>38.1M</td></tr><tr><td>predictor (FiLM residual MLP)</td><td>10.5M</td></tr><tr><td>pathology projector (8192 → 1024)</td><td>8.0M</td></tr><tr><td>pathology reconstruction head episode reconstruction head</td><td>8.4M</td></tr><tr><td></td><td>8.4M</td></tr><tr><td>time embedding</td><td>1.3M</td></tr><tr><td>survival head</td><td>1.0M</td></tr><tr><td>Total trained</td><td>421M</td></tr><tr><td>Frozen parameters: pathology encoder</td><td></td></tr><tr><td>PRISM2 slide encoder</td><td>5.1B</td></tr><tr><td>Virchow2 tile encoder</td><td>632M</td></tr><tr><td>Perceiver + attention pooler (base/prognostic slide encoder)</td><td>620M</td></tr><tr><td>Phi-3-Mini (diagnostic slide encoder)</td><td>3.8B</td></tr></table>

Table A4: Parameter counts for diferent components in the oFM model.