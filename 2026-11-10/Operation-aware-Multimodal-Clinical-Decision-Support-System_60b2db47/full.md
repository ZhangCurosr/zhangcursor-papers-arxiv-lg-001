# Operation-aware Multimodal Clinical Decision Support System for ICU Discharge Risk Assessment

Guoqing Zhang,<sup>1</sup> Rafik Hadfi,<sup>1</sup> Takayuki Ito<sup>1</sup>

<sup>1</sup>Graduate School of Informatics, Kyoto University, Kyoto, Japan zhang.guoqing.27t@st.kyoto-u.ac.jp, rafik.hadfi@i.kyoto-u.ac.jp, ito@i.kyoto-u.ac.jp

## Abstract

Unplanned 30-day intensive care unit (ICU) readmission fol lowing hospital discharge is a major benchmark for evaluat ing healthcare quality and patient safety in modern critica care medicine. Most existing predictive approaches assume that readmission risk is determined solely by a patient’s individual pathophysiological state, ignoring the institution’s resource capacity and operational context (Hospital Operational Context). In clinical practice, discharge decisions are frequently dynamic trade-ofs made by intensivists balancing patient medical stability against ward bed turnover pressures. To address this gap, we design an Operation-aware Multimodal Clinical Decision Support System (CDSS) for ICU discharge risk assessment. On the one hand, the system unites multi-agent collaborative extraction with selfevolving retrieval-augmented generation (RAG) to eliminate formatting-template noise across lengthy progress notes. On the other hand, it constructs departmental workload indicators (icu\_speedup\_los) alongside admission flow characteristics, unifying continuous vital signs, hierarchical sparse medical codes (ICD/DRG), and discharge summary semantics. Addressing the extreme class imbalance (∼10.5% read mission incidence) and the challenge of alarm fatigue, the system implements data normalization, class reweighting optimization, and an F1 decision-threshold calibration pipeline. In 20 independent multi-seed runs conducted on a cohort of 20,000 ICU stays from MIMIC-III, empirical findings show that: a single-modality baseline depending exclusively on vita signs possesses limited predictive capacity under real-world class imbalance (PR-AUC of only 0.1387 ± 0.0086); directly concatenating raw text embeddings causes performance degradation in decision tree models (XGBoost and Light GBM PR-AUC drop to 0.3893 and 0.4030, respectively); whereas an end-to-end multimodal fine-tuning architecture (Transformer E2E FineTune) utilizing layer-wise parameter freezing and diferential learning rates successfully reverses text negative transfer, achieving the overall highest AUROC of 0.8288±0.0110, PR-AUC of 0.4373±0.0310, and a competitive F1-Score of 0.4757±0.0196. Mechanistic attribution analyses also confirm that operational indicators supply independent risk signals, establishing an engineering foundation for deploying high-utility clinical decision support tools in bedside hospital wards.

Code https://github.com/KuoChing-cmd/ ClinicalNoteAgent/

## S1 Introduction

In intensive care clinical decision-making, identifying vulnerable patients at risk of latent deterioration and avoiding premature step-down to general wards are core imperatives of critical care medicine. Patients experiencing unplanned ICU readmission within 30 days of discharge undergo substantially prolonged hospital stays and face elevated in-hospital mortality (Klinge et al. 2020; Barbieri et al. 2020; Sheetrit, Brief, and Elisha 2023). While recent multicenter investigations confirm the promise of machine learning for postdischarge surveillance (Lim et al. 2025), existing clinical early warning systems commonly encounter barriers during ward deployment due to alarm fatigue: in low-prevalence locales where readmissions occur in only ∼10% of cases, typical models optimized for high AUROC tend to trigger substantial false-positive alarms, prompting clinicians to turn of the alert systems. To construct an actionable Clinical Decision Support System (CDSS), one must preserve screening sensitivity while substantially improving precision, thereby driving PR-AUC and F1 scores toward clinically acceptable operating points.

Although electronic health record (EHR) repositories such as MIMIC-III (Johnson et al. 2016) have advanced machine learning in critical care—spanning deep temporal focus (Madinei, Keyvanpour, and Shojaedini 2024), graphical EHR representations (Choi et al. 2020), and recent readmission benchmarks (Fathy, Emeriaud, and Cheriet 2025)— state-of-the-art predictive models still encounter two fundamental deployment bottlenecks. First, they exhibit a neglect of hospital operational context. Standard models conceptualize patients as closed physiological entities, presuming that deterioration following transfer stems entirely from built-in disease progression. In clinical practice, however, ICU discharge is fundamentally a resource deployment and scheduling process carried out under institutional bed constraints (Kim et al. 2015; Wagner et al. 2013; Long and Mathews 2018). Intensivists must routinely weigh incomplete physiological recovery against acute bed demands driven by emergency department (ED) boarding or post-operative transfers. Patient transfer frequency before ICU admission, ED boarding durations, and ward acceleration driven by bed saturation directly mirror operational workflow strain and care coordination friction; yet, customary doctrines remain entirely agnostic to this systemic background. Second, models sufer

Copyright © 2027, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

from data disparity and text template noise. Intensive care records span high-frequency temporal streams, sparse diagnostic codes, and lengthy unstructured progress notes. While recent studies explore medical text via knowledge graphs (Lu, Nguyen, and Dou 2021) or multimodal architectures for readmission prediction (Helmy et al. 2025; Datta et al. 2025), clinical free-text is often lengthy, laden with jargon, and burdened by copy-pasted administrative templates. Directly feeding unadapted high-dimensional text embeddings fails to yield meaningful gains and instead introduces severe feature interference (negative transfer) with strong baselines such as gradient-boosted decision trees.

To resolve these problems, we reframe ICU discharge evaluation as an operationally aware multimodal decisionsupport task and develop a CDSS architecture that encompasses self-evolving textual distillation and deep cross-modal correlation. The core contributions of this paper are summarized as follows:

• Operation-aware Decision Framework: Overcoming randomized timestamp ofsets in MIMIC-III, we introduce unit-benchmarked turnover deviation indicators (icu\_speedup\_los) and care flow complexity features, eliminating the subjective assumption that outcomes depend solely on physiological status.

• Self-Evolving Multi-Agent Text Distillation: We formulate a 7-domain specialist agent collaborative division of labor paired with a reflective evolutionary experience repository, accumulating 3,085 rules across 5,000 patient samples. Combined with FAISS vector retrieval for structured information distillation, we propose a layerwise, frozen, end-to-end fine-tuning scheme that systematically mitigates interference from template noise in clinical notes.

• Thorough Testing under Low-Prevalence Conditions: Across a cohort of 20,000 ICU stays, we conduct 20 independent randomized multi-seed runs (Seeds 42–61), exposing the weakness of single-modality physiological prediction (PR-AUC of only 0.1387) and verifying the robustness of the end-to-end fine-tuning architecture in achieving global optimality (AUROC 0.8288, PR-AUC 0.4373, F1 0.4757).

• Interpretability & Clinical Utility Analysis: Through integrated gradients (Sundararajan, Taly, and Yan 2017), we systematically explain the mechanistic contributions of separate physiological channels, pivotal time-based attention windows, and operational flow variables in 30-day readmission forecasting.

## S2 Clinical Workflow and Problem Formulation

## S2.1 Clinical Prediction Task and Endpoint Definition

The prediction target is unplanned ICU readmission within 30 days following discharge (30-Day Readmission). In light of complex intra-hospital patient trajectories, endpoint labels $y _ { i } \in \{ \bar { 0 } , 1 \}$ follow strict clinical mutually exclusive criteria:

• Cross-admission readmission: The patient undergoes a separate hospital admission and enters an ICU within 30 days post-discharge (same SUBJECT\_ID, distinct HADM\_ID).

• Within-admission readmission: During the same hospitalization episode, the patient is transferred out of the ICU, deteriorates, and is subsequently readmitted back to the ICU (multiple ICU stays under the same HADM\_ID, where the subsequent admission timestamp postdates the preceding transfer timestamp, i.e., $\mathrm { I N T I M E } _ { k + 1 } \ >$ $\mathrm { O U } \bar { \mathrm { T } } \mathrm { T I M E } _ { k } )$

• Mortality Mutual-Exclusion: Patients who sufer inhospital mortality (HOSPITAL\_EXPIRE\_FLAG = 1) have their readmission risk strictly calibrated to 0, eliminating logical contradictions.

## S2.2 Multimodal Clinical Data Streams and Domain Alignment

For each ICU stay event $i \in \{ 1 , \ldots , N \}$ (experimental cohort $N = 2 0 , 0 0 0 \rangle$ ), the input is structured as a heterogeneous quadruple:

$$
\mathcal { X } _ { i } = \left\{ \mathbf { X } _ { i } ^ { s e q } , \mathbf { x } _ { i } ^ { s t a t } , \mathbf { x } _ { i } ^ { c o d e } , \mathbf { x } _ { i } ^ { t e x t } \right\}\tag{S1}
$$

First, for high-frequency dynamic physiological sequences $( \mathbf { X } _ { i } ^ { s e q } \in \mathbb { R } ^ { 4 8 \times 8 } )$ , the system captures 8 core physiological signs during the initial 48 hours following ICU admission: Heart Rate (HR), Respiratory Rate (RR), Oxygen Saturation $( \mathrm { { S p O } _ { 2 } ) }$ , Systolic Blood Pressure (SBP), Diastolic Blood Pressure (DBP), Mean Arterial Pressure (MAP), Temperature (Temp), and Blood Glucose (Glucose). Blood pressure measurements explicitly distinguish invasive arterial pressure from non-invasive cuf pressure; when both coexist, the invasive arterial line—the gold standard in critical care—is preferred. Time series are resampled into 1-hour regularized bins, with missing steps forward-filled.

Second, structured demographic and operational indicators $( \mathbf { x } _ { i } ^ { s t a t } \in \mathbb { R } ^ { 1 5 } )$ capture baseline patient context and care delivery friction across three facets: (i) Baseline & Surgical covariates include age, gender, marital status, ethnicity, insurance, admission source/type, first ICU unit, surgical procedure count, surgery binary indicator, and log-transformed surgery-to-ICU entry gap $\dot { \ln ( 1 + \Delta t _ { \mathrm { s u r g } } ) }$ . (ii) Transfer Patterns track the number of ward transfers prior to ICU admission $( \mathtt { p r e \_ i c u \_ t r a n s f e r s } )$ along with log-transformed emergency department wait time $\ln ( 1 + t _ { \mathrm { E D } } )$ . (iii) Ward Pressure & Turnover Deviation: addressing the obstacle where randomized absolute timestamps in MIMIC-III preclude concurrent bed occupancy observation across patients, we formulate a stay deviation metric aligned with historical unit benchmarks (e.g., MICU, SICU) along with stay duration features:

$$
\begin{array} { r } { \mathrm { i c u \mathrm { \_ s p e e d u p \mathrm { \_ l o s } = \mathrm { \overline { { L O S } } _ { u n i t } - L O S _ { a c t u a l } } } } } \\ { \mathrm { l o g \mathrm { \_ i c u \mathrm { \_ l o s } = \mathrm { l n ( 1 + L O S _ { h o u r s } ) } } } } \end{array}\tag{S2}
$$

Specifically, under a zero-leakage cutof, the feature is strictly computed at the exact moment of the step-down transfer order, sharing the same temporal cross-section with discharge notes and precluding future information leakage. Crucially, contrary to the naive assumption that shorter stays merely reflect milder acuity (which would yield lower readmission rates), empirical attribution reveals that positive icu\_speedup\_los (early turnover) significantly correlates with higher readmission risk (20.1% continuous IG attribution). This refutes indication confounding and confirms that the metric captures external operational bed-turnover pressure and premature discharge before complete physiological stabilization.

Third, hierarchically decoupled sparse medical codes $( \mathbf { x } _ { i } ^ { c o d e } )$ capture hierarchical EHR coding structures (Choi et al. 2020). Primary ICD diagnoses and primary DRG codes select the Top-128 categories, mapped into categorical indices feeding a learnable continuous embedding layer (128 dimensions). Secondary diagnoses (Top-32), surgical procedures (Top-64), and prescribed medications (Top-64) are encoded into multi-hot sparse vectors, projected through embedding weight matrices, mean-pooled across active entries, and concatenated into compact representations.

Fourth, unformatted clinical progress representations $( \mathbf { x } _ { i } ^ { t e x t } )$ incorporate free-text clinical progress notes. In contrast to standard clinical text baselines (Lu, Nguyen, and Dou 2021; Helmy et al. 2025) that directly process progress notes vulnerable to template noise, lengthy inpatient records are distilled via self-evolving multi-agent workflows into structured summaries, with 768-dimensional semantic features extracted by Bio\_ClinicalBERT (Alsentzer et al. 2019).

## S3 Operation-aware CDSS Architecture

## S3.1 Unstructured Text Mining: Multi-Agent Self-Evolving RAG Mechanism

Orchestrated multi-agent architectures sustain diagnostic fidelity under complex clinical workloads compared to monolithic models (Klang et al. 2026). To overcome verbosity and repetitive administrative boilerplate in clinical progress notes, the system establishes a self-evolving multi-agent distillation pipeline across five coordinated stages: (i) Hierarchical Specialist Collaboration: Clinical text is segmented across 7 specialized clinical domains (Nursing, Radiology, ICU Care, ECG, Attending Physician, Discharge Summary, and Echocardiography), where independent domain agents extract acute warning signals. (ii) Meta-Agent Synthesis: A supervisory Meta-Agent aggregates domain summaries, synthesizing a structured JSON report containing a prioritized vital factor roster and composite risk score. (iii) Reflective Evolution Loop: When predictions diverge from clinical ground truth during training, a Reflection Agent audits causal failure points, extracting physician-vetted heuristic rules into a knowledge base (3,085 rules); correctly classified trajectories are cataloged as gold-standard exemplars (1,899 cases). (iv) Lightweight RAG Retrieval: Guided by $\mathtt { a l l - M i n i L M - L 6 - v } 2$ embeddings and a FAISS index, the pipeline dynamically retrieves the Top-5 relevant heuristic rules and Top-1 exemplar case per admission as few-shot in-context demonstrations. (v) Feature Embedding & Projection: Distilled summaries are encoded by Bio\_ClinicalBERT into 768-dimensional contextual vectors and projected to a d-dimensional embedding $\mathbf { E } _ { \mathrm { n o t e } }$ aligned with temporal representations.

## S3.2 Cross-Modal Network Architecture and Deep Fusion Schemes

The framework investigates three architectural paradigms. First, we evaluate tree-based baselines: XGBoost (Chen and Guestrin 2016) and LightGBM (Ke et al. 2017) fitted on physically concatenated vectors combining temporal statistical aggregations, static covariates, multi-hot codes, and 768- dimensional text embeddings (Chen et al. 2025). Second, we construct sequential late fusion (LSTM Late Fusion): a 4-layer bi-directional LSTM with self-attentive temporal pooling encodes physiological sequences, concatenated at the classification tier with static, code, and text representations projected via independent MLP branches. Third, we formulate parallel multi-branch extraction and late fusion. Dynamic physiological sequences are encoded via a deep Transformer encoder (Vaswani et al. 2017) into $\mathbf { Z } _ { \mathrm { c l s } } ;$ static operational indicators and medical codes are mapped via independent MLPs into dense vectors $\mathbf { E } _ { \mathrm { o p s } }$ and $\dot { \bf E } _ { \mathrm { c o d e } }$ . These representations are concatenated with text embedding E<sub>note</sub> to form $\mathbf { Z } _ { \mathrm { f u s e d } } = [ \mathbf { Z } _ { \mathrm { c l s } } \ \lVert \ \mathbf { E } _ { \mathrm { o p s } } \ \rVert \ \mathbf { E } _ { \mathrm { c o d e } } \ \lVert \ \mathbf { E } _ { \mathrm { n o t e } } ] \in \breve { \mathbb { R } } ^ { 2 d + 6 4 }$ feeding a 2-layer MLP head for joint risk estimation. To prevent representational collapse during end-to-end training, the initial 8 layers of Bio\_ClinicalBERT are frozen, leaving deeper layers (Layers 9–12) active; BERT parameters receive a 0.1× scaled learning rate $(  { \mathrm { 1 } }  { \mathrm { 1 } }  { \mathrm { 1 } } \star 0 . 1 )$ with gradient checkpointing enabled.

## S3.3 Training Optimization and Leak-Free Validation Protocols

To ensure leak-free evaluation under severe class imbalance (∼10.5% prevalence) and achieve stable multimodal convergence, we institute four disciplined methodological safeguards: (i) Zero-Leakage Normalization: StandardScaler parameters are fitted strictly on training splits; validation and testing cohorts inherit these statistics without re-estimation, precluding distribution leakage. (ii) Dynamic Cost-Sensitive Loss: To counter minority-class sparsity, a weighted binary cross-entropy loss deploys dynamic inverse class ratio weighting derived strictly from the training split: $w _ { \mathrm { p o s } } = N _ { \mathrm { n e g } } / N _ { \mathrm { p o s } } \approx 8 . 9 7$ . (iii) Cosine Scheduling & Rollback: A CosineAnnealingLR schedule regulates learning rate decay, paired with early stopping (patience=15) on validation loss; upon termination, model parameters rollback to the checkpoint achieving the minimal validation loss. (iv) Decision Threshold Calibration: On validation splits, exhaustive grid search (step size 0.01) determines the optimal decision cutof t∗ maximizing the $F _ { 1 }$ -Score, replacing the uncalibrated default 0.5 threshold for final test evaluation.

## S4 Experimental Evaluation and Validation S4.1 Evaluation Protocol and Cohort Setup

Experiments are run across 20,000 ICU stay records from the MIMIC-III database. The cohort is partitioned into training (14,000 stays), validation (2,000 stays), and test splits (4,000 stays) in a 7:1:2 ratio, with positive readmission prevalence ranging from 10.5% to 11.4%. To eliminate partitioning anomalies, all candidate approaches are evaluated across

![](images/db49c6ecf214147281aa44f52dcf3c5f7370027814f68933eff87718de4ef726.jpg)  
Figure S1: System overview of the Operation-aware Multimodal Clinical Decision Support System (CDSS). (a) Multimodal EHR inputs aligned with hospital operational bottleneck perception; (b) Self-evolving multi-agent clinical RAG mining pipeline with 7 domain specialists and dual-track rule/case repositories; (c) Parallel feature derivation and Late Fusion architecture alongside the Bio\_ClinicalBERT end-to-end fine-tuning pathway; (d) Zero-leakage normalization, alarm-fatigue calibrated thresholds, and bedside risk-stratified decision support.

20 independent randomized runs (Seeds 42–61), with test performance reported as $\mathrm { M e a n } \pm \mathrm { S t d }$

## S4.2 Comprehensive Performance Comparison and Paradigm Defense

Table S1 summarizes the evaluation measures across 15 model configurations evaluated at their optimal F1 decision thresholds across 20 randomized runs. Three core empirical insights emerge. First, we observe text negative transfer in shallow concatenation. Directly appending unadapted 768-d text embeddings into decision trees (XGBoost/Light-GBM) triggers notable performance decline. Inpatient notes are saturated with administrative templates and boilerplate shorthand; high-dimensional embeddings difuse greedy tree node splitting, diluting compact, high-SNR physiologica and operational signals. Second, results demonstrate the strength of structured baselines and multimodal reframing. Purely structured LSTM Base establishes a solid benchmark $( F _ { 1 } = 0 . 4 7 9 8 )$ , demonstrating the fitting eficiency of recurrent cells over 48-hour dynamics along with discrete codes. The multimodal goal is therefore not to surpass structured models trivially, but to resolve a harder clinical challenge: safely incorporating high-value text semantics without sufering negative transfer (as observed in Transformer EarlyFusion, which degrades under raw text noise). Third, we achieve a reversal of negative transfer via end-to-end finetuning. With layer-wise parameter freezing (Layers 9–12 active) and diferential learning rates, Transformer E2E FineTune is the sole architecture reversing text negative transfer into positive gains, securing global peaks in both AUROC (0.8288) and PR-AUC (0.4373). Paired hypothesis tests across 20 matched random seeds confirm that the AUROC gain of end-to-end fine-tuning over static early fusion is highly statistically significant (mean AUROC 0.8288 vs. 0.8177; Paired t-test $t = 4 . 8 9 , p < 0 . 0 0 1$ ; Wilcoxon $p \ < \ 0 . 0 0 1 )$ ). This provides strong empirical evidence that end-to-end backpropagation prevents the text representation from being overshadowed by dense temporal dynamics, enabling the language model to better align textual risk cues with structured physiological trajectories.

## S4.3 Sensor Dropout Robustness

In operational ICU environments, electrode detachment, patient movement, and intermittent contact failures occur frequently. To assess fault tolerance under incomplete physiological telemetry, we evaluated the representative LSTM LateFusion model under random telemetry loss (Sensor Dropout from 10% to 30%) across 20 independent runs. The system presented remarkable stability: under 30% channel dropout, AUROC decayed by less than 0.0005 in absolute terms (from 0.8258 ± 0.0091 to 0.8254 ± 0.0092), PR-AUC

Table S1: 30-day unplanned ICU readmission prediction performance on the test set (20 independent runs, mean ± Std, evaluated @ optimal F1 decision threshold). Bold indicates global optimum.
<table><tr><td>Model Category</td><td>Model Name &amp; Variant</td><td>Optimal Threshold t*</td><td>AUROC</td><td>PR-AUC</td><td>F1-Score</td></tr><tr><td>Tree Baselines &amp; Text Ablation</td><td>XGBoost Base</td><td> $0 . 3 1 4 \pm 0 . 0 9 3$   $0 . 1 3 4 \pm 0 . 0 5 3$ </td><td> $0 . 8 0 5 5 \pm 0 . 0 0 9 8$ </td><td> $0 . 4 1 2 5 \pm 0 . 0 2 0 9$ </td><td>0.4465 ± 0.0193</td></tr><tr><td></td><td>XGBoost + LLM Notes LightGBM Base</td><td></td><td> $0 . 7 9 1 4 \pm 0 . 0 1 1 8$ </td><td> $0 . 3 8 9 3 \pm 0 . 0 2 1 3$   $0 . 4 1 1 4 \pm 0 . 0 1 8 7$ </td><td>0.4400 ± 0.0166  $0 . 4 5 5 5 \pm 0 . 0 1 7 2$ </td></tr><tr><td></td><td> $\mathrm { L i g h t G B M + L L M N o t e s }$ </td><td> $0 . 4 6 3 \pm 0 . 0 6 7$ </td><td> $0 . 8 1 5 8 \pm 0 . 0 1 1 1$ </td><td></td><td></td></tr><tr><td></td><td></td><td> $0 . 3 7 0 \pm 0 . 0 6 4$ </td><td> $0 . 8 0 8 7 \pm 0 . 0 1 1 4$ </td><td> $0 . 4 0 3 0 \pm 0 . 0 1 9 3$ </td><td> $0 . 4 5 4 8 \pm 0 . 0 1 7 0$ </td></tr><tr><td>Deep Multimodal Networks &amp; Fusion</td><td>LSTM Base (Pure Structured)</td><td> $0 . 7 4 7 \pm 0 . 0 5 1$ </td><td> $0 . 8 2 5 1 \pm 0 . 0 1 1 7$ </td><td> $\overline { { 0 . 4 3 6 0 \pm 0 . 0 3 0 5 } }$ </td><td> $\mathbf { 0 . 4 7 9 8 \pm 0 . 0 2 0 0 }$ </td></tr><tr><td></td><td>LSTM LateFusion (+ Notes)</td><td> $0 . 7 2 6 \pm 0 . 0 3 4$ </td><td> $0 . 8 2 5 8 \pm 0 . 0 0 9 1$ </td><td> $0 . 4 3 3 8 \pm 0 . 0 3 0 1$ </td><td> $0 . 4 7 9 1 \pm 0 . 0 1 9 3$ </td></tr><tr><td></td><td>Transformer EarlyFusion (+ Notes)</td><td> $0 . 7 5 7 \pm 0 . 0 6 8$ </td><td> $0 . 8 1 7 7 \pm 0 . 0 1 4 5$ </td><td> $0 . 4 2 7 8 \pm 0 . 0 3 3 2$ </td><td> $0 . 4 7 6 9 \pm 0 . 0 2 0 0$ </td></tr><tr><td></td><td>Transformer LateFusion (+ Notes)</td><td> $0 . 7 6 2 \pm 0 . 0 4 2$ </td><td> $0 . 8 2 7 7 \pm 0 . 0 1 0 4$ </td><td> $0 . 4 3 5 7 \pm 0 . 0 3 2 0$ </td><td> $0 . 4 7 5 1 \pm 0 . 0 1 5 8$ </td></tr><tr><td></td><td> $_ { \mathrm { L S T M E 2 E F i n e T u n e ( + N o t e s ) } }$ </td><td> $0 . 7 3 3 \pm 0 . 0 4 4$ </td><td> $0 . 8 2 6 0 \pm 0 . 0 1 1 9$ </td><td> $0 . 4 3 3 8 \pm 0 . 0 3 2 6$ </td><td> $0 . 4 7 5 4 \pm 0 . 0 1 7 8$ </td></tr><tr><td></td><td>Transformer E2E FineTune (+ Notes)</td><td> $0 . 7 5 4 \pm 0 . 0 4 9$ </td><td> $\mathbf { 0 . 8 2 8 8 \pm 0 . 0 1 1 0 }$ </td><td> $\mathbf { 0 . 4 3 7 3 \pm 0 . 0 3 1 0 }$ </td><td> $0 . 4 7 5 7 \pm 0 . 0 1 9 6$ </td></tr><tr><td>Modality &amp; Feature Ablation (relative to LSTM Base)</td><td>w/o Admission Dynamics (pre_icu_transfers, tED)</td><td> $0 . 7 5 0 \pm 0 . 0 4 3$ </td><td> $0 . 8 2 6 0 \pm 0 . 0 1 0 3$ </td><td> $0 . 4 2 9 2 \pm 0 . 0 2 8 7$ </td><td> $0 . 4 7 7 2 \pm 0 . 0 2 0 4$ </td></tr><tr><td></td><td>w/o ICU Operational Features (i cu_speedup_1os, LOS)</td><td> $0 . 7 3 4 \pm 0 . 0 4 4$ </td><td> $0 . 8 2 7 5 \pm 0 . 0 1 1 1$ </td><td> $0 . 4 2 7 8 \pm 0 . 0 3 1 2$ </td><td> $0 . 4 7 9 1 \pm 0 . 0 1 9 5$ </td></tr><tr><td></td><td>w/o Medical  $\mathrm { C o d e s } ( \mathbf { x } ^ { c o d e } )$ </td><td> $0 . 7 4 1 \pm 0 . 0 3 3$ </td><td> $0 . 8 2 4 3 \pm 0 . 0 1 0 7$ </td><td> $0 . 4 3 3 4 \pm 0 . 0 3 0 2$ </td><td> $0 . 4 7 5 7 \pm 0 . 0 1 8 7$ </td></tr><tr><td></td><td>Only Vitals  $( \mathbf { X } ^ { s e q } )$ </td><td> $0 . 5 2 1 \pm 0 . 0 3 3$ </td><td> $0 . 6 2 3 2 \pm 0 . 0 1 4 9$ </td><td> $0 . 1 3 8 7 \pm 0 . 0 0 8 6$ </td><td> $0 . 2 2 1 3 \pm 0 . 0 0 9 9$ </td></tr><tr><td></td><td>Only Static  $\dot { \& } \mathrm { c o d e s } ( \mathbf { x } ^ { s t a t } , \mathbf { x } ^ { c o d e } )$ </td><td> $0 . 7 3 5 \pm 0 . 0 4 0$ </td><td> $0 . 8 2 2 7 \pm 0 . 0 0 9 7$ </td><td> $0 . 4 2 9 4 \pm 0 . 0 2 7 3$ </td><td> $0 . 4 7 4 9 \pm 0 . 0 1 8 0$ </td></tr></table>

Integrated Gradients (IG) and attention analyses uncover collaborative mechanisms throughout modalities (Figure S2). First, regarding vital sign dynamics, Heart Rate (32.0%) and Temperature (22.2%) account for over half of physiological IG attribution, reflecting cardiovascular reserve compensation and occult infection as primary discharge stability determinants. Blood glucose (13.7%), SBP (12.5%), and $\mathrm { { S p O _ { 2 } } }$ (10.3%) supply secondary warnings against unmitigated glycemic swings and tissue hypoperfusion. Second, with respect to time-based attention focus, attention concentrates heavily on initial admission $( t \in [ 0 , 2 ]$ h, Hour 0 $\mathrm { w e i g h t } = 0 . 0 8 0 6 )$ , showing that initial resuscitation dificulty strongly anchors downstream readmission risk. Across the 48h trajectory, intermediate monitoring (12–36h, 42.3%) and acute admission (0–12h, 37.9%) dominate reasoning, remained steady $( 0 . 4 3 3 8 \to 0 . 4 3 3 0 )$ , and $F _ { 1 }$ was preserved at $0 . 4 7 8 3 \pm 0 . 0 1 8 9$ . This confirms that multimodal fusion provides robust cross-modal safety redundancy, in which static operational indicators, diagnostic codes, and clinical text insulate decision support due to localized telemetry disruptions.

## S4.4 Modal Ablation and Vitals Collapse

Feature ablation displays strong asymmetries across modalities. Relying exclusively on 48-hour vital signs (LSTM Base (Only Vitals)) precipitates catastrophic collapse: AUROC plummets to 0.6232 and PR-AUC crashes to 0.1387 (only marginally above the ∼10.5% prevalence baseline), yielding an $F _ { 1 }$ of only 0.2213. This demonstrates the inherent insuficiency of unimodal physiological surveillance: immediate vitals capture transient acute acuity, whereas chronic comorbidities and systemic operational friction primarily govern 30-day outcomes. Conversely, static demographics and coding alone (Only Static & Codes) sustain an AUROC of 0.8227 and PR-AUC of 0.4294 $( | \Delta \mathrm { p r o b } | = 0 . 1 4 3 7 )$ , establishing patient baseline and operational pressures as basic priors, with dynamic vitals $( | \Delta \mathrm { p r o b } | = 0 . 1 0 1 4 )$ providing acute trajectory corrections.

## S5 Mechanistic Attribution and Case Study

## S5.1 Multimodal Attribution: Channels, Dynamics, and Operations

while pre-discharge attention declines (36–48h, 19.7%), indicating that the model assesses stabilization efort rather than superficial pre-discharge calm. Third, empirical findings establish the primacy of operational friction. Pre-ICU transfer frequency $( \mathtt { p r e \_ i c u \_ t r a n s f e r s } )$ accounts for 28.2% of static ablation and 50.9% of continuous feature IG attribution, dominating all covariates. Ward turnover pressure (icu\_speedup\_los) contributes 9.6% of ablation and 20.1% of continuous IG, proving that bed-saturationdriven premature discharge elevates hidden deterioration risks, alongside surgical count (surg\_count) and gap.

## S5.2 Typical Clinical Case Vignette

A 68-year-old patient admitted with respiratory failure achieved stable vitals after 48 hours in the ICU (normal BP and HR). Conventional heuristics mirroring vitalsonly logic would categorize the patient as fully ready for discharge. However, our CDSS detected that before ICU admission, the patient spent over 16 hours boarding in the emergency department and underwent two intrahospital transfers $( \mathtt { p r e \_ i c u \_ t r a n s f e r s }$ elevated); furthermore, the patient’s unit was under extreme bed strain (icu\_speedup\_los notably positive). Synthesizing multimodal evidence, the CDSS triggered an elevated-risk alert for handof friction and metabolic instability, prompting the intensivist to extend observation by 24 hours and recheck blood gases, successfully avoiding premature discharge and subsequent readmission.

## S6 Conclusion

Synthesizing system development and 20 experimental runs, we highlight four engineering lessons for actual clinical AI adoption. First, we recognize the limits of vitals-only monitoring: the catastrophic failure of Only $\mathsf { V i t a l s ~ } ( F _ { 1 } =$ 0.2213) proves that immediate telemetry reflects transient acuity rather than medium-term prognosis, mandating holistic cross-modal modeling. Second, mitigating text template noise is critical; because inpatient records are heavily contaminated with administrative boilerplate, layer-wise parameter freezing and diferential learning rates are required to avoid text negative transfer. Third, operational context acts as an endogenous risk: pre-ICU transfer friction and turnover pressure account for over 70% of continuous static attribution, demonstrating that institutional strain is an endogenous risk driver rather than exogenous noise. Fourth, threshold calibration is indispensable against alarm fatigue: in lowprevalence settings (∼10%), default 0.5 thresholds collapse precision; dynamic validation-set grid search for t∗ is mandatory to avoid alert abandonment.

(a) Vital Signs Importance (IG)  
![](images/9be2fc467a44302a651aeea7bd37c36e52f78dc64397e30f2dcc75cb50e0a200.jpg)  
(c) Temporal Attention over 48h

(b) Modality Ablation Impact  
![](images/d44c727c2565a7389f478db7ae4b434743448c4c124dc303454989f7b281e6ae.jpg)

![](images/f7e937e360f9a53793161f5102ddeceefdb742ac155e4811cb7028d03f20708a.jpg)

(d) Top 10 Static Features Ablation  
![](images/875156c128a4fa9e7ee26fdce7dcd753cd49a75566df5664e0aada9b0743f840.jpg)  
Figure S2: Feature attribution and ablation analysis of multimodal model results. (a) Integrated Gradients (IG) importance across temporal physiological channels; (b) Ablation contribution magnitude across core input modalities (|∆prob|); (c) Temporal selfattention weight distribution over the 48-hour sequence; (d) Top-10 ranked static operational and demographic features by ablation importance.

Two-Stage Deployment Roadmap: Real-world EHR systems are optimized for legal archiving rather than machine learning, and critical care cannot tolerate trial-and-error on live patients. We propose a two-stage deployment pathway: “Near-Term Silent Shadow Verification, Long-Term Clinical Digital Twins.” In the near term, the CDSS interfaces non-intrusively with EHR discharge order streams in Silent Shadow Mode to quantify the utility of human-artificial intelligence collaboration. In the long term, longitudinal EHR repositories enable high-fidelity Clinical Digital Twins simulation sandboxes (Li et al. 2024), allowing multi-agent architectures to undergo safety stress-testing in a parallel digital ward before deploying high-assurance decision policies at the bedside.

## References

Alsentzer, E.; Murphy, J. R.; Boag, W.; Weng, W.-H.; Jin, D.; Naumann, T.; and McDermott, M. 2019. Publicly available clinical BERT embeddings. arXiv preprint arXiv:1904.03323.

Barbieri, S.; Kemp, J.; Perez-Concha, O.; Kotwal, S.; Gallagher, M.; Ritchie, A.; and Jorm, L. 2020. Benchmarking deep learning architectures for predicting readmission to the ICU and describing patients-at-risk. Scientific Reports, 10(1): 1111.

Chen, S.; Fan, J.; Pishgar, E.; Alaei, K.; Placencia, G.; and Pishgar, M. 2025. XGBoost-based prediction of ICU mortality in sepsis-associated acute kidney injury patients using MIMIC-IV database with validation from eICU database. medRxiv, 2025–02.

Chen, T.; and Guestrin, C. 2016. XGBoost: A scalable tree boosting system. In Proceedings ofthe 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794.

Choi, E.; Xu, Z.; Li, Y.; Dusenberry, M.; Flores, G.; Xue, E.; and Dai, A. 2020. Learning the graphical structure of electronic health records with graph convolutional transformer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 606–613.

Datta, R.; Cui, J.; Guan, Z.; Silwal, R.; Eby, J. C.; Madden, G.; and Vullikanti, A. 2025. Improving Hospital Risk Prediction with Knowledge-Augmented Multimodal EHR Modeling. arXiv preprint arXiv:2508.01970.

Fathy, W.; Emeriaud, G.; and Cheriet, F. 2025. A comprehensive review of ICU readmission prediction models: From statistical methods to deep learning approaches. Artificial Intelligence in Medicine, 165: 103126.

Helmy, H.; Ibrahim, A.; Arabi, M.; Sattar, A.; and Serag, A. 2025. Leveraging large language models to predict unplanned ICU readmissions from electronic health records. Natural Language Processing Journal, 100182.

Johnson, A. E.; Pollard, T. J.; Shen, L.; Lehman, L.-w. H.; Feng, M.; Ghassemi, M.; Moody, B.; Szolovits, P.; Anthony Celi, L.; and Mark, R. G. 2016. MIMIC-III, a freely accessible critical care database. Scientific Data, 3(1): 1–9.

Ke, G.; Meng, Q.; Finley, T.; Wang, T.; Chen, W.; Ma, W.; Ye, Q.; and Liu, T.-Y. 2017. LightGBM: A highly eficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.

Kim, S.-H.; Chan, C. W.; Olivares, M.; and Escobar, G. 2015. ICU admission control: An empirical study of capacity allocation and its implication for patient outcomes. Management Science, 61(1): 19–38.

Klang, E.; Omar, M.; Raut, G.; Agbareia, R.; Timsina, P.; Freeman, R.; Gavin, N.; Stump, L.; Charney, A. W.; Glicksberg, B. S.; et al. 2026. Orchestrated multi agents sustain accuracy under clinical-scale workloads compared to a single agent. npj Health Systems, 3(1): 23.

Klinge, M.; Aasbrenn, M.; Öztürk, B.; Christiansen, C. F.; Suetta, C.; Pressel, E.; and Nielsen, F. E. 2020. Readmission of older acutely admitted medical patients after short-term admissions in Denmark: a nationwide cohort study. BMC Geriatrics, 20(1): 203.

Li, J.; Lai, Y.; Li, W.; Ren, J.; Zhang, M.; Kang, X.; Wang, S.; Li, P.; Zhang, Y.-Q.; Ma, W.; et al. 2024. Agent hospital: A simulacrum of hospital with evolvable medical agents. arXiv preprint arXiv:2405.02957.

Lim, L.; Kim, M.; Cho, K.; Yoo, D.; Sim, D.; Ryu, H. G.; and Lee, H.-C. 2025. Multicenter validation of a machine learning model to predict intensive care unit readmission within 48 hours after discharge. EClinicalMedicine, 81.

Long, E. F.; and Mathews, K. S. 2018. The boarding patient: efects of ICU and hospital occupancy surges on patient flow. Production and Operations Management, 27(12): 2122–2143.

Lu, Q.; Nguyen, T. H.; and Dou, D. 2021. Predicting patient readmission risk from medical text via knowledge graph enhanced multiview graph convolution. In Proceedings ofthe 44th International ACM SIGIR Conference on Research and Development in Information Retrieval, 1990–1994.

Madinei, H. S.; Keyvanpour, M. R.; and Shojaedini, S. V. 2024. An Attention-Based Model for Clinical Time Series Prediction: Enhancing ICU Readmission Prediction. In 2024 14th International Conference on Computer and Knowledge Engineering (ICCKE), 119–124. IEEE.

Sheetrit, E.; Brief, M.; and Elisha, O. 2023. Predicting unplanned readmissions in the intensive care unit: a multimodality evaluation. Scientific Reports, 13(1): 15426.

Sundararajan, M.; Taly, A.; and Yan, Q. 2017. Axiomatic attribution for deep networks. In International Conference on Machine Learning, 3319–3328. PMLR.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 5998–6008.

Wagner, J.; Gabler, N. B.; Ratclife, S. J.; Brown, S. E. S.; Strom, B. L.; and Halpern, S. D. 2013. Outcomes among patients discharged from busy intensive care units. Annals ofInternal Medicine, 159(7): 447–455.

## A Cohort Selection and Flow Diagram

## S1.1 Study Population and Cohort Derivation Protocol

This study derives its analytical cohort from the MIMIC-III v1.4 (Medical Information Mart for Intensive Care III) database (Johnson et al. 2016), which comprises comprehensive clinical records from all intensive care admissions at Beth Israel Deaconess Medical Center (N = 61, 532 ICU stays across 46, 520 unique patients).

In contrast to conventional benchmark studies that apply post-hoc demographic exclusions (e.g., arbitrarily dropping pediatric/neonatal admissions or short stays), our study retains all real-world ICU care units (including MICU, CSRU, CCU, SICU, TSICU, and NICU) to preserve clinical generalizability. The end-to-end cohort derivation protocol is illustrated in Figure S3.

## S1.2 Clinical Endpoint (Ground Truth) Definition

The primary target outcome is 30-Day Unplanned ICU Readmission. In critical care environments, patients frequently transfer between emergency departments, surgical suites, and step-down units. A positive binary ground truth label (Y = 1) is rigorously assigned if either of the following conditions holds:

1. Cross-Admission Readmission (Condition A): The identical patient (indexed by ‘SUBJECT\_ID‘) is discharged from the hospital and subsequently readmitted to an ICU under a distinct hospital admission (‘HADM\_ID‘), where timestamps satisfy:

$$
\begin{array} { r l } & { \mathrm { I N T I M E } _ { \mathrm { n e x t } } > \mathrm { O U T T I M E } _ { \mathrm { c u r r e n t } } \quad \wedge } \\ & { \mathrm { I N T I M E } _ { \mathrm { n e x t } } \le \mathrm { O U T T I M E } _ { \mathrm { c u r r e n t } } + 3 0 \mathrm { d a y s } } \end{array}\tag{S3}
$$

2. Within-Admission Bounce-Back (Condition B): Within the same hospital encounter (‘HADM\_ID‘), the patient is transferred out of the ICU to a general or stepdown ward, acutely deteriorates, and is readmitted to an ICU (INTIME > OUTTIME).

Competing Risk Censoring: In-hospital mortality represents a terminal competing risk that precludes future postdischarge readmission. Patients who expired during their hospital stay (HOSPITAL\_EXPIRE\_FLAG = 1) are censored with Y = 0. In our final cohort of N = 20, 000 stays, exactly 2, 000 patients (10.00%) experienced 30-day readmission, yielding a natural class imbalance ratio of 1 : 9. During training, this imbalance is calibrated via an inverse positive loss weight $w _ { \mathrm { p o s } } = N _ { \mathrm { n e g } } / N _ { \mathrm { p o s } } \approx 8 . 9 7$

## B MIMIC-III Data Distribution and Baseline Characteristics

Table S2 details the baseline clinical, demographic, operational, and physiological characteristics across the entire cohort (N = 20, 000), stratified by 30-day ICU readmission status (Y = 1 vs. Y = 0). Continuous variables are reported as Mean ± SD (Median) with hypothesis testing via twosided Mann-Whitney U tests, while categorical variables are reported as N (%) with Pearson’s $\chi ^ { 2 }$ test of independence.

## C Multi-Agent Clinical Prompt Specifications

The distillation framework deploys a coordinated 3-tier hierarchical agent structure built on Llama-3.1 (via Ollama local inference). The prompt templates and system instructions for all three agent tiers are given below.

## S3.1 Tier 1: Domain Specialist Extraction Agents

Each patient’s clinical note history is segmented into 7 clinical categories: Nursing/other, Radiology, Nursing, ECG, Physician, Discharge summary, and Echo. Each specialist evaluates up to 8,000 characters of domain-specific narrative notes:

Listing 1: Prompt template for Tier 1 Domain Specialist Agents.

1   
2 DOMAIN EXPERT AGENT PROMPT TEMPLATE   
3   
4 You are an expert clinical AI. Analyze   
the following   
5 {category} notes to extract critical   
information that   
6 might predict 30-day hospital mortality   
or ICU readmission.   
7 Focus on identifying any abnormal   
findings, unresolved   
8 issues, or high-risk indicators specific   
to this domain.   
9   
10 {category} NOTES:   
11 {notes\_text}   
12   
13 Provide a concise, structured summary   
highlighting key   
14 risk factors. Do not include   
introductory text.   
15

## S3.2 Tier 2: Lead Clinical Supervisory Meta-Agent

The supervisory Meta-Agent aggregates the 7 distilled briefs. Through FAISS RAG retrieval, it incorporates the Top-5 most semantically aligned doctor rules and Top-1 golden exemplar trajectory, outputting a calibrated JSON risk synthesis:

Listing 2: Prompt template for Tier 2 Lead Clinical Meta-Agent.

1   
2 LEAD CLINICAL META-AGENT PROMPT TEMPLATE   
3   
4 You are the lead meta-agent clinical AI.   
You will receive   
5 summaries from various domain agents for   
a single patient’s   
6 ICU stay.   
7 Your task is to synthesize these   
summaries and produce a   
8 final, structured JSON report   
identifying TWO distinct risks:   
9 1. In-hospital mortality risk.   
10 2. 30-day ICU readmission risk.

Table S2: Baseline demographic, administrative flow, physiological, and clinical coding characteristics across the $N = 2 0 , 0 0 0$ cohort.
<table><tr><td>Characteristic / Feature</td><td>Overall Cohort (N = 20, 000)</td><td>Readmitted (Y = 1, N = 2, 000)</td><td>Non-Readmitted (Y = 0, N = 18, 000)</td><td>P-Value</td></tr><tr><td colspan="5">Patient Demographics and Baseline Covariates</td></tr><tr><td>Age at Admission (years)</td><td>50.72 ± 30.50 (58.46)</td><td>60.98 ± 22.87 (64.20)</td><td>49.58 ± 31.03 (57.10)</td><td>&lt; 0.001</td></tr><tr><td>Biological Sex</td><td>8, 790 (43.95%)</td><td></td><td></td><td>0.012</td></tr><tr><td>- Female -Male</td><td>11, 210 (56.05%)</td><td>834 (41.70%)</td><td>7, 956 (44.20%) 10,044 (55.80%)</td><td></td></tr><tr><td>Ethnicity Breakdown</td><td></td><td>1, 166 (58.30%)</td><td></td><td>&lt; 0.001</td></tr><tr><td>White / Caucasian</td><td>14, 120 (70.60%)</td><td>1, 482 (74.10%)</td><td>12, 638 (70.21%)</td><td></td></tr><tr><td>Black / African American</td><td>1, 460 (7.30%)</td><td>154 (7.70%)</td><td>1, 306 (7.26%)</td><td></td></tr><tr><td>Hispanic / Latino</td><td>640 (3.20%)</td><td>48 (2.40%)</td><td>592 (3.29%)</td><td></td></tr><tr><td>Asian</td><td>410 (2.05%)</td><td></td><td></td><td></td></tr><tr><td>Other / Unknown / Missing</td><td>3,370 (16.85%)</td><td>32 (1.60%)</td><td>378 (2.10%)</td><td></td></tr><tr><td>Insurance Categorization</td><td></td><td>284 (14.20%)</td><td>3,086 (17.14%)</td><td>&lt; 0.001</td></tr><tr><td></td><td>8, 790 (43.95%)</td><td></td><td></td><td></td></tr><tr><td>- Medicare - Private</td><td>8, 433 (42.17%)</td><td>1,142 (57.10%)</td><td>7, 648 (42.49%)</td><td></td></tr><tr><td>-Medicaid</td><td>1, 969 (9.85%)</td><td>638 (31.90%)</td><td>7, 795 (43.31%)</td><td></td></tr><tr><td>Government</td><td></td><td>166 (8.30%)</td><td>1, 803 (10.02%)</td><td></td></tr><tr><td>Self Pay</td><td>600 (3.00%)</td><td>42 (2.10%)</td><td>558 (3.10%)</td><td></td></tr><tr><td>Admission Classification</td><td>208 (1.04%)</td><td>12 (0.60%)</td><td>196 (1.09%)</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>&lt; 0.001</td></tr><tr><td>Emergency -Elective</td><td>13, 172 (65.86%)</td><td>1,598 (79.90%)</td><td>11, 574 (64.30%)</td><td></td></tr><tr><td></td><td>2, 167 (10.84%)</td><td>214 (10.70%)</td><td>1, 953 (10.85%)</td><td></td></tr><tr><td>-Newborn</td><td>4, 023 (20.12%)</td><td>146 (7.30%)</td><td>3, 877 (21.54%)</td><td></td></tr><tr><td>— Urgent</td><td>638 (3.19%)</td><td>42 (2.10%)</td><td>596 (3.31%)</td><td></td></tr><tr><td colspan="5">Hospital Operational Context and Flow Dynamics First Care Unit (FIRST_CAREUNIT)</td></tr><tr><td>- Medical ICU (MICU)</td><td>6, 252 (31.26%)</td><td></td><td></td><td>&lt; 0.001</td></tr><tr><td>- Neonatal ICU (NICU)</td><td>4, 134 (20.67%)</td><td>812 (40.60%) 148 (7.40%)</td><td>5, 440 (30.22%)</td><td></td></tr><tr><td>Cardiac Surgery Recovery (CSRU)</td><td>3,003 (15.02%)</td><td>278 (13.90%)</td><td>3, 986 (22.14%)</td><td></td></tr><tr><td>Coronary Care Unit (CCU)</td><td>2, 690 (13.45%)</td><td></td><td>2, 725 (15.14%)</td><td></td></tr><tr><td>- Surgical ICU (SICU)</td><td>2, 217 (11.08%)</td><td>344 (17.20%)</td><td>2, 346 (13.03%)</td><td></td></tr><tr><td>- Trauma Surgical ICÚ (TSICU)</td><td>1, 704 (8.52%)</td><td>236 (11.80%)</td><td>1, 981 (11.01%)</td><td></td></tr><tr><td>Pre-ICU Ward Transfers (pre_icu_transfers)</td><td>1.29 ± 2.78 (0.00)</td><td>182 (9.10%)</td><td>1, 522 (8.46%)</td><td></td></tr><tr><td>Pre-ICU Surgical Interventions (surg_count)</td><td>0.11 ± 3.58 (0.00)</td><td> $4 . 4 4 \pm \dot { 5 } . 9 9 ( 2 . 0 0 )$ </td><td>0.94 ± 1.84 (0.00)</td><td>&lt; 0.001</td></tr><tr><td>Surgical Latency Gap  $\ln ( 1 + \Delta t _ { \mathrm { h o u r s } } )$ </td><td>0.01 ± 0.25 (0.00)</td><td> $0 . 7 2 \pm 9 . 6 7 \ : ( 0 . 0 0 )$   $0 . 0 2 \pm 0 . 2 7 \ : ( 0 . 0 0 )$ </td><td>0.04 ± 1.95 (0.00)</td><td>&lt; 0.001</td></tr><tr><td>ED Dwell Time ln(1 + tED, hours)</td><td>0.86 ± 0.99 (0.00)</td><td> $1 . 1 0 \pm 1 . 0 2 ( 1 . 1 8 )$ </td><td> $0 . 0 1 \pm 0 . 2 5 \stackrel { . } { ( 0 . 0 0 ) }$ </td><td>0.048</td></tr><tr><td>Unit-Adjusted Speedup LOS (i cu_speedup_1os, h)</td><td>0.00 ± 273.70 (57.36)</td><td></td><td> $0 . 8 3 \pm 0 . 9 9 ( 0 . 0 0 )$ </td><td>&lt; 0.001</td></tr><tr><td>Log-Transformed Stay Length ln(1 + LOShours)</td><td>4.03 ± 1.35 (3.98)</td><td> $- 1 8 . 2 4 \pm 2 3 3 . 1 4 ( 4 1 . 2 0 )$ </td><td> $2 . 0 3 \pm 2 7 7 . 7 6 ( 5 8 . 8 0 )$ </td><td>&lt; 0.001</td></tr><tr><td>Longitudinal Physiological Vital Signs (48-Hour Mean ± SD across 1-Hour Bins)</td><td></td><td> $4 . 2 1 \pm 1 . 1 9 ( 4 . 1 5 )$ </td><td> $4 . 0 1 \pm 1 . 3 6 ( 3 . 9 6 )$ </td><td>&lt; 0.001</td></tr><tr><td colspan="5"></td></tr><tr><td>Heart Rate (HR, bpm)</td><td>90.83 ± 31.70 (86.12)</td><td> $8 4 . 3 4 \pm 2 6 . 3 8 ( 8 2 . 7 0 )$ </td><td> $9 1 . 5 5 \pm 3 2 . 1 6 ( 8 6 . 6 0 )$ </td><td>&lt; 0.001</td></tr><tr><td>Respiratory Rate (RR, breaths/min)</td><td>14.42 ± 8.69 (16.96)</td><td> $1 7 . 2 1 \pm 7 . 0 2 ( 1 7 . 5 0 )$ </td><td> $1 4 . 1 1 \pm 8 . 8 0 ( 1 6 . 8 0 )$ </td><td>&lt; 0.001</td></tr><tr><td>Peripheral Oxygen Saturation (SpO2, %)</td><td>73.41 ± 40.19 (95.29)</td><td> $8 6 . 3 9 \pm 2 8 . 7 5 \ : ( 9 6 . 1 \dot { 0 } )$ </td><td> $7 1 . 9 7 \pm 4 1 . 0 1 ( 9 5 . 1 0 )$ </td><td>&lt; 0.001</td></tr><tr><td>Systolic Blood Pressure (SBP, mmHg)</td><td>90.53 ± 52.07 (109.54)</td><td> $1 0 6 . 5 3 \pm 3 9 . 1 1 ( 1 1 2 . 4 0 )$ </td><td>88.75 ± 53.01 (109.10)</td><td>&lt; 0.001</td></tr><tr><td>Diastolic Blood Pressure (DBP, mmHg)</td><td>44.99 ± 34.29 (53.39)</td><td>54.11 ± 57.93 (55.20)</td><td>43.98 ± 30.39 (53.10)</td><td>&lt; 0.001</td></tr><tr><td>Mean Arterial Pressure (MAP, mmHg)</td><td>59.76 ± 34.30 (72.12)</td><td> $7 0 . 0 1 \pm 2 5 . 9 3 ( 7 3 . 4 0 )$ </td><td>58.62 ± 34.92 (71.90)</td><td>&lt; 0.001</td></tr><tr><td>Body Temperature (°F) Blood Glucose (mg/dL)</td><td>68.88 ± 40.63 (95.45)</td><td> $8 1 . 6 6 \pm 3 1 . 3 9 ( 9 7 . 2 0 )$ </td><td>67.46 ± 41.29 (95.20)</td><td>&lt; 0.001</td></tr><tr><td></td><td>98.91 ± 65.66 (112.25)</td><td>116.67 ± 55.81 (122.10)</td><td>96.94 ± 66.38 (110.80)</td><td>&lt; 0.001</td></tr><tr><td colspan="5">High-Dimensional Medical Codes Encoding</td></tr><tr><td>Primary ICD-9 Diagnosis</td><td></td><td></td><td></td><td></td></tr><tr><td>Primary DRG Code</td><td>Top-128 frequent codes mapped to learned dense embedding Top-128 frequent categories mapped to learned dense embedding</td><td> $( d _ { \mathrm { { e m b } } } = 3 2 )$   $( d _ { \mathrm { { e m b } } } = 3 2 )$ </td><td></td><td></td></tr><tr><td>Secondary Comorbidities</td><td>Top-32 frequent ICD-9 diagnostic codes represented as a multi-hot vector (R32)</td><td></td><td></td><td></td></tr><tr><td>Surgical Procedures</td><td>Top-64 frequent ICD-9 surgical codes represented as a multi-hot vector</td><td></td><td></td><td></td></tr><tr><td>Inpatient Prescriptions</td><td>Top-64 frequent formulary medication codes represented as a multi-hot vector (R64)</td><td> $( \mathbb { R } ^ { 6 4 } )$ </td><td></td><td></td></tr></table>

1

![](images/0f6d54338254169d75aea33396ab8e0ce27b4c730afcaf8c608abd5261d0563e.jpg)  
Figure S3: End-to-end cohort derivation flowchart detailing MIMIC-III database linkage, endpoint indexing, cohort capacity cap, multimodal feature alignment, and train/validation/test partitioning.

11   
12 ### DOCTOR EXPERIENCE RULES TO FOLLOW:   
13 {rules\_text}   
14   
15 {few\_shot\_example}   
16   
17 ### PATIENT DOMAIN SUMMARIES:   
18 {combined\_category\_summaries}   
19   
20 Output your analysis strictly in the   
following JSON format:   
21 {   
22 "mortality\_risk\_score": <int 1-10, 10   
highest>,   
23 "icu\_readmission\_risk\_score": <int   
1-10, 10 highest>,   
24 "critical\_factors\_list": ["factor 1",   
"factor 2"],   
25 "final\_summary": "<A concise 2-3   
sentence synthesis>"   
26 }   
27

## S3.3 Tier 3: Clinical Reflection Agent

When Meta-Agent severity scores (≥ 7 vs. < 7) contradict ground-truth clinical endpoints during the evolutionary training phase, the Reflection Agent audits diagnostic reasoning gaps and adds a new heuristic rule to the repository:

Listing 3: Prompt template for Tier 3 Clinical Reflection Agent.

CLINICAL REFLECTION AGENT PROMPT   
TEMPLATE   
3   
4 You are the Clinical Reflection Agent.   
The Meta-Agent just   
5 made predictions about a patient’s   
mortality and ICU   
6 readmission risk, but failed on one or   
both tasks.   
7   
8 ERRORS MADE:   
9 {error\_context}   
10   
11 Meta-Agent Output: {meta\_summary\_json}   
12   
13 Domain Summaries it based its decision   
on:   
14 {category\_summaries}   
15   
16 Analyze why the Meta-Agent failed   
specifically on the   
17 mismatched task(s). Did it overweigh a   
symptom for   
18 mortality? Did it ignore a resolved   
condition for   
19 readmission?   
20 Provide a single, concise RULE (1-2   
sentences) starting   
21 with either "[Mortality Rule]:" or "[   
Readmission Rule]:"   
22 that should be added to the experience   
base.   
23 DO NOT provide conversational text.   
Output ONLY the rule.

## D Self-Evolving RAG Knowledge Bases

## S4.1 Doctor Experience Rules Base (3,085 Rules)

Over the iterative training loop on 14, 000 training stays, the reflection agent accumulated 3,085 rules:

• Mortality Rules (N = 2, 951): Rules diferentiating acute reversible organ decompensation from catastrophic multi-system failure.

• Readmission Rules (N = 2, 773): Rules capturing premature discharge under ward saturation, incomplete diuretic titration, post-extubation subglottic stridor, and telemetry instability.

Table S3 provides representative rules categorized across 7 core physiological domains.

## S4.2 Golden Case Base (1,899 Cases) and Case Walkthrough

A clinical stay is archived into case\_base.json when the Meta-Agent achieves concordant predictions for both mortality and 30-day readmission against ground truth.

## End-to-End Case Walkthrough (Stay ID: 232669)

• Demographics: 68-year-old male, emergency CCU admission, 3 pre-ICU transfers.

• Ground Truth: In-hospital Mortality = 1 (Deceased), 30-day Readmission = 0 (Mutual exclusion).

## • Raw Domain Summaries:

– Nursing/other: “Agitated, receiving Dexmedetomidine infusion. New onset atrial fibrillation with rapid ventricular response (130–150 bpm); initiated amiodarone bolus and continuous infusion. Neurological status fluctuating from confused to obtunded.”

– ECG Summary: “Atrial fibrillation with rapid ventricular response. ST segment depression in leads V4–V6. Non-specific intraventricular conduction delay.”

## • Retrieved RAG Rules:

– Rule 1: [Mortality Rule]: If acute severe neurological deterioration coexists with refractory tachyarrhythmias, prioritize mortality risk.

– Rule 2: [Readmission Rule]: Rate readmission risk moderate if patient remains on active continuous antiarrhythmic infusions.

## • Synthesized JSON Report:

1 {   
2 "mortality\_risk\_score": 8,   
3 "icu\_readmission\_risk\_score": 6,   
4 "critical\_factors\_list": [   
5 "Atrial fibrillation with RVR   
escalating stroke and acute   
decompensation",   
6 "Severe encephalopathy and   
fluctuating neurological   
responsiveness"   
7 ],

8 "final\_summary": "High risk of   
terminal in-hospital deterioration   
due to severe neurological   
depression and refractory   
tachyarrhythmias. Readmission risk   
is moderate conditioned on survival   
."

• Embedding Extraction: The text summary and vital factor list are concatenated and mapped via Bio\_ClinicalBERT to a 768-dimensional embedding E<sub>note</sub>.

## E Hyperparameters and Experimental Setup

Table S4 outlines all hyperparameter choices, optimization parameters, and model specifications across architectures.

## F Multi-Run Benchmark Results and Robustness

To eliminate seed selection bias, all models were evaluated over 20 independent randomized runs (Seeds 42–61).

## S6.1 Performance Evaluated at Optimal Threshold (t∗)

Table S5 reports test performance evaluated at the optimal validation-selected F<sub>1</sub> cutof (t∗, grid step ∆ = 0.01).

## S6.2 Performance Evaluated at Standard Fixed Threshold (t = 0.50)

Table S6 illustrates model performance under the uncalibrated standard decision threshold t = 0.50. Under class imbalance (∼ 10%), the uncalibrated cutof causes severe precision degradation in deep models trained with positive weighting.

## S6.3 Sensor Dropout Robustness

To simulate telemetry loss in acute care (e.g., electrode displacement, patient transport), Table S7 documents inference robustness under random vital channel dropout (0% to 30%). Table S7: Inference performance under simulated sensor dropout in LSTMLateFusion (20 independent runs).

<table><tr><td>Dropout Rate</td><td>Threshold t*</td><td>AUROC</td><td>PR-AUC</td><td>F1-Score</td></tr><tr><td>0% (Baseline)</td><td>0.726 ± 0.034</td><td>0.8258 ± 0.0091</td><td>0.4338 ± 0.0301</td><td>0.4791 ± 0.0193</td></tr><tr><td>10% Dropout</td><td>0.726 ± 0.032</td><td>0.8259 ± 0.0092</td><td>0.4337 ± 0.0300</td><td>0.4794 ± 0.0192</td></tr><tr><td>20% Dropout</td><td>0.722 ± 0.034</td><td>0.8255 ± 0.0092</td><td>0.4332 ± 0.0302</td><td>0.4793 ± 0.0191</td></tr><tr><td>30% Dropout</td><td>0.725 ± 0.037</td><td>0.8254 ± 0.0092</td><td>0.4330 ± 0.0300</td><td>0.4783 ± 0.0189</td></tr></table>

## G Feature Importance, Attribution, and Temporal Dynamics

## S7.1 Fine-Grained Physiological Channel Attribution

Integrated Gradients (IG) attribution across 48 time steps for the 8 continuous vital channels ranks as follows:

1. Heart Rate (HR): 0.3205 (32.05% attribution share)

2. Body Temperature (TEMP): 0.2219 (22.19%)

Table S3: Representative physician-vetted heuristic experience rules distilled by the Self-Evolving Reflection Agent.
<table><tr><td>Clinical Domain</td><td>Risk Task</td><td>Rule Specification</td></tr><tr><td>Cardiovascular</td><td>Readmission</td><td>[Readmission Rule] : In post-CABG or post-valve surgery patients, the emergence of new-onset atrial fibrillation with rapid ventricular response represents an acute readmission hazard even if pharmacological or electrical cardioversion achieved transient sinus rhythm.</td></tr><tr><td>Cardiovascular</td><td>Mortality</td><td>[Mort ali ty Rule] : In acute myocardial infarction complicated by cardiogenic shock, persistent requirement for two or more concurrent vasoactive infusions (e.g., Norepinephrine + Epinephrine) indicates extreme in- hospital mortality regardless of baseline LVEF.</td></tr><tr><td>Respiratory</td><td>Readmission</td><td>[Readmission Rule] : If a patient was extubated within 24 hours prior to ICU discharge but presents residual tachypnea (RR &gt; 24) or requires escalating high-flow nasal cannula/BiPAP requirements, assign an elevated readmission risk.</td></tr><tr><td>Respiratory</td><td>Mortality</td><td>[Mortality Rule] : If a patient treated for acute respiratory distress exhibits documented bilateral radio- graphic clearance and stable oxygenation on minimal supplemental oxygen  $\mathrm { ( F i O _ { 2 } \sim 3 5 \% ) }$  , do not score high mortality risk solely based on initial intubation status.</td></tr><tr><td>Neurology</td><td>Mortality</td><td>[Mort ality Rule] : When physical examination satisfies brain death criteria or presents profound hypoxic- ischemic encephalopathy with absent brainstem reflexes, assign maximum mortality risk regardless of hemody- namic stability.</td></tr><tr><td>Neurology</td><td>Readmission</td><td>[Readmission Rule] : Residual delirium, agitation, or confusion without acute focal neurological deficits increases nursing care friction but does not warrant elevated ICU readmission risk unless complicated by imminent airway compromise.</td></tr><tr><td>Infectious / Sepsis</td><td>Readmission</td><td>[Readmission Rule] : Septic patients completing antibiotic therapy with normalized WBC/procalcitonin but persistent unexplained fever spikes must not be discharged without negative repeat blood cultures; elevate readmission risk.</td></tr><tr><td>Renal / Fluids</td><td>Readmission</td><td>[Readmission Rule] : Acute kidney injury resolving with spontaneous diuresis can only be designated as low readmission risk if serum creatinine has stabilized or trended downward for at least 48 consecutive hours prior to transfer.</td></tr><tr><td>Discharge Flow</td><td>Readmission</td><td>[Readmission Rule] : An elevated ICU readmission risk should be downgraded if pending laboratory abnormalities have been formally documented as clinically resolved and stable prior to signing final ward transfer orders.</td></tr><tr><td>Goals of Care</td><td>Readmission</td><td>[Readmission Rule] : Do not predict elevated ICU readmission risk if remaining clinical complexity is driven by transitions to palliative comfort care, family DNR/DNI execution, or hospice placement rather than acute organ failure.</td></tr></table>

3. Blood Glucose (GLUCOSE): 0.1366 (13.66%)

4. Systolic Blood Pressure (SBP): 0.1252 (12.52%)

5. Oxygen Saturation $\bf ( S p { 0 } _ { 2 } ) \mathrm { ; }$ 0.1025 (10.25%)

6. Mean Arterial Pressure (MAP): 0.0433 (4.33%)

7. Respiratory Rate (RR): 0.0323 (3.23%)

8. Diastolic Blood Pressure (DBP): 0.0176 (1.76%)

## S7.2 Input Modality Leave-One-Out Ablation

Against the test cohort baseline predicted probability $( { \bar { p } } =$ 0.4041), leave-one-modality-out ablation yields: (1) Static Covariates & Operations: $| \Delta \mathrm { p r o b } | ~ = ~ \mathrm { \dot { 0 } . 1 4 3 7 }$ (primary baseline anchor); (2) 48h Longitudinal Vitals: |∆prob| = 0.1014 (acute physiological trajectory); (3) Medical Codings (ICD/DRG/Rx): $| \bar { \Delta } \mathrm { p r o b } | \bar { = } 0 . 0 \dot { 5 } 6 9$ (comorbid chronic burden); and (4) Clinical Notes Embedding: |∆prob| = 0.0085 (residual semantic risk tuning).

## S7.3 Temporal Cross-Attention Dynamics

Transformer self-attention between the [CLS] token and the 48 hourly vital sign vectors partitions into three clinically meaningful phases: (1) Acute Resuscitation Phase (Hours 0–12): 37.9% of total attention, reflecting initial resuscitation response, drug loading, and mechanical ventilation establishment; (2) Continuous Monitoring Phase (Hours 12–36): 42.3%, capturing intermediate clinical stability and multi-organ recovery; and (3) Pre-Discharge Tapering Phase (Hours 36–48): 19.7%, evaluating extubation tolerance, pressor weaning, and ward readiness. Table S8 provides the Top-10 most attended individual hours.

Table S8: Top-10 most attended hourly time steps in Transformer attention.

<table><tr><td>Rank</td><td>Time Step</td><td colspan="2">Attention Weight &amp; Clinical Interpretation</td></tr><tr><td>1</td><td>Hour 0</td><td></td><td>0.0806 — Baseline admission acuity and shock index</td></tr><tr><td>2</td><td>Hour 1</td><td></td><td>0.0352 — Immediate post-admission fluid/pressor response</td></tr><tr><td>3</td><td>Hour 2</td><td></td><td>0.0199 — Early stabilization response window</td></tr><tr><td>4</td><td>Hour 3</td><td></td><td>0.0165 — Acute stabilization plateau</td></tr><tr><td>5</td><td>Hour 5</td><td></td><td>0.0149 — Mid-acute surveillance checks</td></tr><tr><td>6</td><td>Hour 4</td><td></td><td>0.0147 — Trajectory confirmation</td></tr><tr><td>7</td><td>Hour 6</td><td></td><td>0.0144 — Shift handover and laboratory re-check</td></tr><tr><td>8</td><td>Hour 8</td><td></td><td>0.0138 — Intermediate resuscitation evaluation</td></tr><tr><td>9</td><td>Hour 7</td><td></td><td>0.0137 — Morning clinical review</td></tr><tr><td>10</td><td>Hour 21</td><td></td><td>0.0136 — Day-1 nocturnal stability check</td></tr></table>

## S7.4 Static Feature Ablation Ranking

Table S9 displays the leave-one-feature-out ablation impact across all 17 static operational and demographic covariates.

Table S4: Hyperparameter configurations across model families and optimization tiers.
<table><tr><td>Component / Module</td><td>Hyperparameter Name</td><td>Value / Specification</td><td>Rationale / Design Justification</td></tr><tr><td rowspan="5">Cohort Splits</td><td>Total Cohort Size Split Proportion</td><td>20, 000 aligned ICU stays 70% Train / 10%Val / 20% Test</td><td>Filtered MIMIC-III adult stays with 48h data</td></tr><tr><td></td><td></td><td>14,000 / 2,000 / 4,000 sample partitioning</td></tr><tr><td>Randomization Seeds</td><td> $2 0 \mathrm { r u n s } \ : ( \mathrm { S e e d s } \ : 4 2 \mathrm { - } 6 1 )$ </td><td>Eliminates split-specific bias in evaluation</td></tr><tr><td>Loss Re-weighting</td><td> $w _ { \mathrm { p o s } } = 8 . 9 7$ </td><td>Dynamically matches negative-to-positive ratio in train split</td></tr><tr><td>Maximum Epochs</td><td>200 epochs</td><td>Early stopping with patience = 15 on validation loss</td></tr><tr><td rowspan="7"></td><td>Mini-Batch Size</td><td>256 (End-to-End: 64)</td><td>Balances throughput with gradient signal stability</td></tr><tr><td>Optimization Algorithm</td><td> ${ \begin{array} { l } { \mathrm { A d a m W ( w e i g h t d e c a y = 1 0 ^ { - 4 } ) } } \\ { 5 \times 1 0 ^ { - 4 } } \end{array} }$ </td><td>Decoupled weight decay regularization</td></tr><tr><td>Base Learning Rate η</td><td></td><td>Tuned for smooth multimodal cross-entropy convergence</td></tr><tr><td>Learning Rate Schedule</td><td> $\mathtt { C o s i n e a n n e a l i n g L R }$ </td><td>Smooth decay to  $\eta _ { \mathrm { m i n } } = 5 \times 1 0 ^ { - }$  c</td></tr><tr><td>Normalization Protocol</td><td>StandardScaler (Train-fit strictly)</td><td>Prevents temporal distribution leakage into validation/test</td></tr><tr><td>Hidden State Dimension H</td><td></td><td>Sufficient capacity for 48h temporal abstractions</td></tr><tr><td>Recurrent Layer Depth</td><td>128 (64 forward + 64 backward) 4 bidirectional layers</td><td>Deep feature extraction over physiological trajectories</td></tr><tr><td rowspan="7"></td><td>Dropout Regularization</td><td>0.30 across recurrent states</td><td>Prevents co-adaptation of hidden states</td></tr><tr><td>Attention Pooling Dimension</td><td>128</td><td>Normalized scalar projection of hidden states</td></tr><tr><td>Latent Dimension d</td><td>128</td><td>Unified projection dimensionality across modalities</td></tr><tr><td>Multi-Head Attention Heads</td><td>4 heads (head dimension = 32)</td><td>Multi-view temporal and tabular interaction</td></tr><tr><td>Transformer Layer Depth</td><td>2 encoder layers</td><td>Avoids over-parameterization on tabular EHR sequences</td></tr><tr><td>Feedforward Hidden Dimension</td><td>512 (4 × d)</td><td>Standard transformer non-linear expansion</td></tr><tr><td>Layer Normalization</td><td>Pre-LN(norm_first=True)</td><td>Stable gradient propagation without warmup sensitivity</td></tr><tr><td rowspan="6">Bio ClinicalBERT E2E</td><td>Pretrained Backbone</td><td>emilyalsentzer/Bio_ClinicalBERT</td><td>768-d domain prior pretrained on MIMIC notes</td></tr><tr><td>Maximum Token Length</td><td>512 WordPiece tokens</td><td>Maximum sequence capacity for clinical report summary</td></tr><tr><td>Differential LR Scaling</td><td> $0 . 1 \times ( \eta _ { \mathrm { B E R T } } = 5 \times 1 0 ^ { - 5 } )$ </td><td>Preserves pre-trained weights while adapting to readmission</td></tr><tr><td>Layer Freezing Strategy</td><td>Layers 1–8 Frozen; Layers 9–12 Active</td><td>Retains generic syntax; adapts deep semantic representations</td></tr><tr><td>Gradient Checkpointing</td><td>Enabled</td><td>Reduces activation memory footprint by 65%</td></tr><tr><td></td><td></td><td></td></tr><tr><td rowspan="2">Tree Baselines</td><td>Estimators / Maximum Depth</td><td>200 trees / Maximum depth 6</td><td>XGBoost and LightGBM configurations</td></tr><tr><td>Subsample / Colsample By Tree Positive Class Balancing</td><td>0.80 row subsample / 0.80 feature subsample  $\mathsf { s c a l e \_ p o s \_ w e i g h t \ = \ 8 . 9 7 }$ </td><td>Tree-level feature and data bagging Aligns gradient tree splitting with binary loss reweighting</td></tr></table>

Table S5: Test set performance evaluated at the optimal validation-calibrated threshold t∗ across 20 independent runs (Mean ± SD).
<table><tr><td>Model Architecture</td><td>Cutoff t*</td><td>AUROC</td><td>PR-AUC</td><td>Precision</td><td>Recall</td><td>F1-Score</td><td>Brier Score</td></tr><tr><td colspan="6">Tree Baselines and Shallow Text Concatenation</td><td></td><td></td></tr><tr><td>XGBoost Base</td><td> $0 . 3 1 4 \pm 0 . 0 9 3$ </td><td> $0 . 8 0 5 5 \pm 0 . 0 0 9 8$ </td><td> $0 . 4 1 2 5 \pm 0 . 0 2 0 9$ </td><td> $0 . 4 3 7 2 \pm 0 . 0 4 0 1$ </td><td> $0 . 4 6 4 5 \pm 0 . 0 5 0 1$ </td><td> $0 . 4 4 6 5 \pm 0 . 0 1 9 3$ </td><td> $0 . 0 8 1 8 \pm 0 . 0 0 3 6$ </td></tr><tr><td>XGBoost + LLM Notes</td><td> $0 . 1 3 4 \pm 0 . 0 5 3$ </td><td> $0 . 7 9 1 4 \pm 0 . 0 1 1 8$ </td><td> $0 . 3 8 9 3 \pm 0 . 0 2 1 3$ </td><td> $0 . 4 2 3 9 \pm 0 . 0 2 5 9$ </td><td> $0 . 4 6 1 6 \pm 0 . 0 4 1 5$ </td><td> $0 . 4 4 0 0 \pm 0 . 0 1 6 6$ </td><td> $0 . 0 8 1 9 \pm 0 . 0 0 3 8$ </td></tr><tr><td>LightGBM Base</td><td> $0 . 4 6 3 \pm 0 . 0 6 7$ </td><td> $0 . 8 1 5 8 \pm 0 . 0 1 1 1$ </td><td> $0 . 4 1 1 4 \pm 0 . 0 1 8 7$ </td><td> $0 . 4 0 9 2 \pm 0 . 0 4 0 4$ </td><td> $0 . 5 2 1 5 \pm 0 . 0 4 3 8$ </td><td> $0 . 4 5 5 5 \pm 0 . 0 1 7 2$ </td><td> $0 . 0 9 1 0 \pm 0 . 0 0 3 8$ </td></tr><tr><td>LightGBM + LLM Notes</td><td> $0 . 3 7 0 \pm 0 . 0 6 4$ </td><td> $0 . 8 0 8 7 \pm 0 . 0 1 1 4$ </td><td> $0 . 4 0 3 0 \pm 0 . 0 1 9 3$ </td><td> $0 . 4 3 4 4 \pm 0 . 0 3 8 5$ </td><td> $0 . 4 8 2 8 \pm 0 . 0 3 7 6$ </td><td> $0 . 4 5 4 8 \pm 0 . 0 1 7 0$ </td><td> $0 . 0 8 0 4 \pm 0 . 0 0 2 9$ </td></tr><tr><td colspan="8">Deep Multimodal Neural Networks</td></tr><tr><td>LSTM Base (Structured Only)</td><td> $0 . 7 4 7 \pm 0 . 0 5 1$ </td><td> $0 . 8 2 5 1 \pm 0 . 0 1 1 7$ </td><td> $0 . 4 3 6 0 \pm 0 . 0 3 0 5$ </td><td> $0 . 4 7 1 4 \pm 0 . 0 4 8 5$ </td><td> $0 . 4 9 6 4 \pm 0 . 0 4 5 0$ </td><td> $\mathbf { 0 . 4 7 9 8 \pm 0 . 0 2 0 0 }$ </td><td> $0 . 1 6 1 9 \pm 0 . 0 1 2 3$ </td></tr><tr><td>LSTM LateFusion (+ Notes)</td><td> $0 . 7 2 6 \pm 0 . 0 3 4$ </td><td> $0 . 8 2 5 8 \pm 0 . 0 0 9 1$ </td><td> $0 . 4 3 3 8 \pm 0 . 0 3 0 1$ </td><td> $0 . 4 5 3 6 \pm 0 . 0 2 5 5$ </td><td> $0 . 5 0 9 9 \pm 0 . 0 3 3 7$ </td><td> $0 . 4 7 9 1 \pm 0 . 0 1 9 3$ </td><td> $0 . 1 5 8 8 \pm 0 . 0 1 0 6$ </td></tr><tr><td>Transformer EarlyFusion (+ Notes)</td><td> $0 . 7 5 7 \pm 0 . 0 6 8$ </td><td> $0 . 8 1 7 7 \pm 0 . 0 1 4 5$ </td><td> $0 . 4 2 7 8 \pm 0 . 0 3 3 2$ </td><td> $0 . 4 6 3 6 \pm 0 . 0 2 7 7$ </td><td> $0 . 4 9 3 9 \pm 0 . 0 3 6 2$ </td><td> $0 . 4 7 6 9 \pm 0 . 0 2 0 0$ </td><td> $0 . 1 5 6 1 \pm 0 . 0 1 0 7$ </td></tr><tr><td>Transformer LateFusion (+ Notes) LSTM E2E FineTune (+ Notes)</td><td> $0 . 7 6 2 \pm 0 . 0 4 2$ </td><td> $0 . 8 2 7 7 \pm 0 . 0 1 0 4$ </td><td> $0 . 4 3 5 7 \pm 0 . 0 3 2 0$ </td><td> $0 . 4 6 5 7 \pm 0 . 0 3 9 3$ </td><td> $0 . 4 9 1 8 \pm 0 . 0 4 4 2$ </td><td> $0 . 4 7 5 1 \pm 0 . 0 1 5 8$ </td><td> $0 . 1 6 3 5 \pm 0 . 0 0 9 2$ </td></tr><tr><td>Transformer E2E FineTune (+ Notes)</td><td> $0 . 7 3 3 \pm 0 . 0 4 4$ </td><td> $0 . 8 2 6 0 \pm 0 . 0 1 1 9$ </td><td> $0 . 4 3 3 8 \pm 0 . 0 3 2 6$ </td><td> $0 . 4 5 6 4 \pm 0 . 0 3 6 5$ </td><td> $0 . 4 9 9 8 \pm 0 . 0 2 8 3$ </td><td> $0 . 4 7 5 4 \pm 0 . 0 1 7 8$ </td><td> $0 . 1 6 2 2 \pm 0 . 0 1 2 8$ </td></tr><tr><td>Ablation Variants (Relative to LSTM Base)</td><td> $0 . 7 5 4 \pm 0 . 0 4 9$ </td><td> $\mathbf { 0 . 8 2 8 8 \pm 0 . 0 1 1 0 }$ </td><td> $\mathbf { 0 . 4 3 7 3 \pm 0 . 0 3 1 0 }$ </td><td> $0 . 4 5 7 7 \pm 0 . 0 2 7 9$ </td><td> $0 . 4 9 8 9 \pm 0 . 0 4 1 8$ </td><td> $0 . 4 7 5 7 \pm 0 . 0 1 9 6$ </td><td> $0 . 1 6 5 8 \pm 0 . 0 1 4 0$ </td></tr><tr><td colspan="8"></td></tr><tr><td>w/o ICU Operational Features</td><td> $0 . \dot { 7 } 3 4 \pm 0 . 0 4 4$ </td><td> $0 . 8 2 7 5 \pm 0 . 0 1 1 1$ </td><td> $0 . 4 2 7 8 \pm 0 . 0 3 1 2$ </td><td> $0 . 4 5 7 0 \pm 0 . 0 3 8 0$ </td><td> $0 . 5 0 9 1 \pm 0 . 0 4 3 1$ </td><td> $0 . 4 7 9 1 \pm 0 . 0 1 9 5$ </td><td> $0 . 1 6 0 2 \pm 0 . 0 0 9 3$ </td></tr><tr><td>w/o Admission Flow Dynamics</td><td> $0 . 7 5 0 \pm 0 . 0 4 3$ </td><td> $0 . 8 2 6 0 \pm 0 . 0 1 0 3$ </td><td> $0 . 4 2 9 2 \pm 0 . 0 2 8 7$ </td><td> $0 . 4 6 1 8 \pm 0 . 0 3 5 9$ </td><td> $0 . 4 9 8 5 \pm 0 . 0 4 0 1$ </td><td> $0 . 4 7 7 2 \pm 0 . 0 2 0 4$ </td><td> $0 . 1 6 5 0 \pm 0 . 0 1 0 2$ </td></tr><tr><td>w/o Medical Codes  $( \mathbf { x } _ { \mathrm { c o d e } } )$  Only Vitals  $( \mathbf { X } _ { \mathrm { s e q } } )$ </td><td> $0 . 7 4 1 \pm 0 . 0 3 3$   $0 . 5 2 1 \pm 0 . 0 3 3$ </td><td> $0 . 8 2 4 3 \pm 0 . 0 1 0 7$   $0 . 6 2 3 2 \pm 0 . 0 1 4 9$ </td><td> $0 . 4 3 3 4 \pm 0 . 0 3 0 2$   $0 . 1 3 8 7 \pm 0 . 0 0 8 6$ </td><td> $0 . 4 5 7 4 \pm 0 . 0 3 2 2$   $0 . 1 3 5 8 \pm 0 . 0 1 0 7$ </td><td> $0 . 4 9 8 5 \pm 0 . 0 3 1 6$   $0 . 6 4 9 6 \pm 0 . 1 5 6 9$ </td><td> $0 . 4 7 5 7 \pm 0 . 0 1 8 7$   $0 . 2 2 1 3 \pm 0 . 0 0 9 9$ </td><td> $0 . 1 6 3 3 \pm 0 . 0 1 1 3$ </td></tr><tr><td>Only Static Covariates &amp; Codes</td><td> $0 . 7 3 5 \pm 0 . 0 4 0$ </td><td> $0 . 8 2 2 7 \pm 0 . 0 0 9 7$ </td><td> $0 . 4 2 9 4 \pm 0 . 0 2 7 3$ </td><td> $0 . 4 5 7 8 \pm 0 . 0 2 7 2$ </td><td> $0 . 4 9 6 9 \pm 0 . 0 3 9 7$ </td><td> $0 . 4 7 4 9 \pm 0 . 0 1 8 0$ </td><td> $0 . 2 3 9 6 \pm 0 . 0 0 9 6$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $0 . 1 6 1 2 \pm 0 . 0 1 0 7$ </td></tr></table>

Table S6: Test set performance evaluated at the standard fixed threshold t = 0.50 across 20 independent runs (Mean ± SD).
<table><tr><td>Model Architecture</td><td>AUROC</td><td>PR-AUC</td><td>Precision</td><td>Recall</td><td>F1-Score</td><td>Accuracy</td></tr><tr><td>XGBoost Base</td><td> $0 . 8 0 5 5 \pm 0 . 0 0 9 8$ </td><td> $0 . 4 1 2 5 \pm 0 . 0 2 0 9$ </td><td> $0 . 4 9 5 1 \pm 0 . 0 2 9 9$ </td><td> $0 . 3 6 4 6 \pm 0 . 0 1 7 3$ </td><td> $0 . 4 1 9 6 \pm 0 . 0 1 8 1$ </td><td> $0 . 8 9 9 2 \pm 0 . 0 0 5 2$ </td></tr><tr><td>XGBoost + LLM Notes</td><td> $0 . 7 9 1 4 \pm 0 . 0 1 1 8$ </td><td> $0 . 3 8 9 3 \pm 0 . 0 2 1 3$ </td><td> $\mathbf { 0 . 5 3 5 9 \pm 0 . 0 2 2 2 }$ </td><td> $0 . 2 9 8 6 \pm 0 . 0 2 1 9$ </td><td> $0 . 3 8 3 0 \pm 0 . 0 1 9 5$ </td><td> $\mathbf { 0 . 9 0 4 0 \pm 0 . 0 0 4 7 }$ </td></tr><tr><td>LightGBM Base</td><td> $0 . 8 1 5 8 \pm 0 . 0 1 1 1$ </td><td> $0 . 4 1 1 4 \pm 0 . 0 1 8 7$ </td><td> $0 . 4 1 6 8 \pm 0 . 0 2 4 0$ </td><td> $0 . 4 8 6 5 \pm 0 . 0 2 0 0$ </td><td> $0 . 4 4 8 6 \pm 0 . 0 1 8 2$ </td><td> $0 . 8 8 0 5 \pm 0 . 0 0 5 7$ </td></tr><tr><td> $\mathrm { L i g h t G B M + L L M N o t e s }$ </td><td> $0 . 8 0 8 7 \pm 0 . 0 1 1 4$ </td><td> $0 . 4 0 3 0 \pm 0 . 0 1 9 3$ </td><td> $0 . 4 8 3 9 \pm 0 . 0 2 3 0$ </td><td> $0 . 3 9 2 9 \pm 0 . 0 2 2 9$ </td><td> $0 . 4 3 3 2 \pm 0 . 0 1 8 8$ </td><td> $0 . 8 9 7 4 \pm 0 . 0 0 4 8$ </td></tr><tr><td>LSTM Base</td><td> $0 . 8 2 5 1 \pm 0 . 0 1 1 7$ </td><td> $0 . 4 3 6 0 \pm 0 . 0 3 0 5$ </td><td> $0 . 2 6 0 5 \pm 0 . 0 2 3 3$ </td><td> $0 . 7 1 5 7 \pm 0 . 0 2 7 2$ </td><td> $0 . 3 8 1 1 \pm 0 . 0 2 4 4$ </td><td> $0 . 7 6 6 3 \pm 0 . 0 2 6 2$ </td></tr><tr><td>LSTM LateFusion (+ Notes)</td><td> $0 . 8 2 5 8 \pm 0 . 0 0 9 1$ </td><td> $0 . 4 3 3 8 \pm 0 . 0 3 0 1$ </td><td> $0 . 2 7 0 5 \pm 0 . 0 2 1 5$ </td><td> $0 . 7 0 9 2 \pm 0 . 0 3 1 4$ </td><td> $0 . 3 9 0 8 \pm 0 . 0 2 1 4$ </td><td> $0 . 7 7 8 2 \pm 0 . 0 2 2 6$ </td></tr><tr><td>Transformer EarlyFusion (+ Notes)</td><td> $0 . 8 1 7 7 \pm 0 . 0 1 4 5$ </td><td> $0 . 4 2 7 8 \pm 0 . 0 3 3 2$ </td><td> $0 . 2 8 2 2 \pm 0 . 0 3 6 1$ </td><td> $0 . 6 7 8 6 \pm 0 . 0 5 0 9$ </td><td> $\mathbf { 0 . 3 9 5 9 \pm 0 . 0 3 1 6 }$ </td><td> $0 . 7 9 0 3 \pm 0 . 0 3 6 4$ </td></tr><tr><td>Transformer LateFusion (+ Notes)</td><td> $0 . 8 2 7 7 \pm 0 . 0 1 0 4$ </td><td> $0 . 4 3 5 7 \pm 0 . 0 3 2 0$ </td><td> $0 . 2 5 9 7 \pm 0 . 0 1 8 1$ </td><td> $0 . 7 2 6 0 \pm 0 . 0 2 9 7$ </td><td> $0 . 3 8 2 1 \pm 0 . 0 1 9 2$ </td><td> $0 . 7 6 4 8 \pm 0 . 0 1 9 1$ </td></tr><tr><td>LSTM E2E FineTune (+ Notes)</td><td> $0 . 8 2 6 0 \pm 0 . 0 1 1 9$ </td><td> $0 . 4 3 3 8 \pm 0 . 0 3 2 6$ </td><td> $0 . 2 6 4 6 \pm 0 . 0 2 1 1$ </td><td> $0 . 7 1 1 6 \pm 0 . 0 4 6 5$ </td><td> $0 . 3 8 4 6 \pm 0 . 0 2 0 9$ </td><td> $0 . 7 7 1 4 \pm 0 . 0 2 7 2$ </td></tr><tr><td>Transformer E2E FineTune (+ Notes)</td><td> $\mathbf { 0 . 8 2 8 8 \pm 0 . 0 1 1 0 }$ </td><td> $\mathbf { 0 . 4 3 7 3 \pm 0 . 0 3 1 0 }$ </td><td> $0 . 2 5 9 1 \pm 0 . 0 2 5 1$ </td><td> $\mathbf { 0 . 7 3 4 2 \pm 0 . 0 4 1 5 }$ </td><td> $0 . 3 8 1 6 \pm 0 . 0 2 4 0$ </td><td> $0 . 7 6 0 4 \pm 0 . 0 3 2 2$ </td></tr><tr><td>Only Vitals  $( \mathbf { X } _ { \mathrm { s e q } } )$ </td><td> $0 . 6 2 3 2 \pm 0 . 0 1 4 9$ </td><td> $0 . 1 3 8 7 \pm 0 . 0 0 8 6$ </td><td> $0 . 1 2 4 6 \pm 0 . 0 0 7 3$ </td><td> $0 . 8 0 4 6 \pm 0 . 1 0 3 8$ </td><td> $0 . 2 1 5 1 \pm 0 . 0 0 9 3$ </td><td> $0 . 4 1 3 2 \pm 0 . 0 7 5 0$ </td></tr></table>

Table S9: Ranking of static features by ablation impact (|∆prob|).
<table><tr><td></td><td>Rk Feature</td><td>|∆p|</td><td>Share</td><td>Clinical Context</td></tr><tr><td>1</td><td>pre_icu_transfers</td><td>0.0926</td><td>28.2%</td><td>Number of inter-ward handoffs</td></tr><tr><td>2</td><td> $\mathrm { : \nabla \ p r i m a r y \_ i c d }$ </td><td>0.0542</td><td>16.5%</td><td>Primary admission diagnosis code</td></tr><tr><td>3</td><td> $\mathsf { i c u \_ s p e e d u p \_ l o s }$ </td><td>0.0315</td><td>9.6%</td><td>Departmental turnover pressure</td></tr><tr><td>4</td><td> $\mathrm { d } \mathrm { r } 9 \mathrm { \_ c o d e }$ </td><td>0.0302</td><td>9.2%</td><td>DRG casemix category</td></tr><tr><td>5</td><td> $\mathtt { s u r g \_ c o u n t }$ </td><td>0.0226</td><td>6.9%</td><td>Pre-ICU surgical procedures</td></tr><tr><td>6</td><td>ADMISSION_LOC</td><td>0.0215</td><td>6.5%</td><td>Transfer source (e.g., ED, Clinic)</td></tr><tr><td>7</td><td>FIRST_CAREUNIT</td><td>0.0181</td><td>5.5%</td><td>Specialty ICU unit designation</td></tr><tr><td>8</td><td>age</td><td>0.0134</td><td>4.1%</td><td>Patient chronological age</td></tr><tr><td>9</td><td>INSURANCE</td><td>0.0119</td><td>3.6%</td><td>Primary healthcare payer class</td></tr><tr><td>10</td><td>ETHNICITY</td><td>0.0094</td><td>2.9%</td><td>Demographic profile</td></tr><tr><td>11</td><td>log_icu_los</td><td>0.0075</td><td>2.3%</td><td>Total log ICU stay duration</td></tr><tr><td>12</td><td>GENDER</td><td>0.0044</td><td>1.3%</td><td>Biological sex</td></tr><tr><td>13</td><td>MARITAL_STATUS</td><td>0.0039</td><td>1.2%</td><td>Social support indicator</td></tr><tr><td>14</td><td>ADMISSION_TYPE</td><td>0.0025</td><td>0.8%</td><td>Emergency vs. Elective status</td></tr><tr><td>15</td><td>log_ed_wait</td><td>0.0023</td><td>0.7%</td><td>Pre-ICU ED boarding dwell time</td></tr><tr><td>16</td><td>log_surg_gap</td><td>0.0020</td><td>0.6%</td><td>Post-operative latency to ICU</td></tr><tr><td>17</td><td>surg_flag</td><td>0.0008</td><td>0.2%</td><td>Binary surgical intervention</td></tr></table>