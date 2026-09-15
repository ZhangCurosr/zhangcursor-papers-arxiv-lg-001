# Knowledge-Enriched Structured EHR Features for 30-Day Hospital Readmission Prediction on MIMIC-IV

Mohamad Najafi<sup>1</sup>, Hongyun Fu<sup>1</sup>, Mathias Brochhausen<sup>2</sup>, Jian Wu<sup>1⋆</sup>, and Yaohang Li<sup>1⋆⋆</sup>

<sup>1</sup> Old Dominion University, Virginia, USA

<sup>2</sup> University of Arkansas for Medical Sciences, Arkansas, USA

Abstract. Recent approaches to 30-day hospital readmission prediction rely on pre-trained language models applied to discharge summaries. Although these methods achieve strong performance, they depend on the availability of clinical notes, incur substantial computational costs, and yield representations that lack interpretability. We propose a knowledge-enriched feature representation that augments structured Electronic Health Record (EHR) data with four medical knowledge sources: disease ontology mapping, procedure classification, drug ingredient vocabulary, and organ system laboratory aggregation, without using clinical notes. Each feature dimension corresponds to a named clinical concept, yielding a sparse and interpretable patient representation. The approach is evaluated with six classifiers on a MIMIC-IV v2.2 cohort. Under 20-fold cross-validation, the best configuration achieves an AUROC of 0.743. This performance is comparable to that of previously reported methods on this dataset—including both those using only structured data and those incorporating clinical notes—while requiring considerably less computational cost. Interpretability analysis shows that demographics, organ system labs, drug ingredient features, and first-level ontology disease categories drive prediction, while deeper hierarchy levels contribute negligibly. These findings indicate that knowledge-enriched structured features ofer a competitive and eficient alternative to embeddings from clinical notes for 30-day readmission prediction.

Keywords: Hospital readmission · Structured EHR · Medical ontology · MIMIC-IV · Knowledge-enriched prediction

## 1 Introduction

Unplanned 30-day hospital readmissions remain a persistent quality and cost challenge in healthcare systems. In the United States, the Hospital Readmissions Reduction Program (HRRP) imposes financial penalties on hospitals with excess readmission rates [25], and readmission rates serve as a widely adopted indicator of care quality across countries [37]. Accurate prediction of readmission risk at the point of discharge enables targeted interventions, such as transitional care programs and structured outpatient follow-up [5], which have proven efective at reducing avoidable readmissions [29].

Recently, several methods have incorporated unstructured clinical text to improve readmission prediction. Almeida et al. [1] achieved an area under the receiver operating characteristic curve (AUROC ∈ [0, 1]) of 0.727 on MIMIC-IV [19] by encoding discharge summaries with a pre-trained language model and combining the resulting embeddings with structured features in a graph neural network. Other text-based and multimodal approaches, including MM-STGNN [36] and a ClinicalT5-based model [28], report comparable AUROC values on restricted MIMIC-IV subsets.

Structured Electronic Health Record (EHR) data, by contrast, are available for all billed admissions and, because they are already recorded as discrete coded fields (diagnoses, procedures, laboratory results, and medications), require no natural language processing to encode [30]. In this work, we focus on diagnosis codes, procedure codes, laboratory results, and medication records, which cover the principal clinical drivers of readmission risk (comorbidity, interventions, physiological status, and treatment) and are populated for essentially all billed admissions in MIMIC-IV. Other structured signals (e.g., vital signs, microbiology) are sparser or less standardized and are left to future work. However, raw structured features encode clinical events as flat categorical variables without capturing the semantic relationships between them [10]. For instance, ICD-10 E11.9 (type 2 diabetes without complications) and E11.65 (type 2 diabetes with hyperglycemia) share no explicit similarity unless an external ontology links them through a common disease ancestor; Makohon et al. [24] showed that mapping ICD codes to SNOMED CT concepts via ontological definitions establishes such clinically meaningful links. Medical ontologies, including the Mondo Disease Ontology (MONDO) [38], the Clinical Classifications Software (CCS) [33], and the RxNorm drug terminology [26], organize clinical concepts into hierarchical structures that encode precisely these relationships.

This paper investigates whether augmenting structured EHR features with knowledge derived from four medical sources produces a readmission prediction model that is competitive with note-based approaches. The contributions are: (1) a knowledge-enriched feature representation integrating four medical knowledge sources with structured EHR data, producing a sparse patient representation in which each dimension corresponds to a named clinical concept; (2) a controlled three-way comparison of ontology-enriched, transformer-encoded, and combined features on a MIMIC-IV v2.2 cohort with six classifiers; (3) an interpretability analysis quantifying the contribution of each knowledge source; and (4) evidence that knowledge-enriched structured features achieve competitive 30-day readmission prediction without clinical notes.

## 2 Related Work

## 2.1 Readmission Prediction from Structured EHR

Early readmission risk models relied on manually curated scoring systems. The LACE index [37] combines length of stay, acuity, comorbidity, and ED visits into an additive score (AUROC ∼0.68), and the HOSPITAL score [11] uses seven structured variables with similar performance. Machine learning approaches extended these baselines: Rajkomar et al. [30] applied deep learning to EHR from 216,221 patients, achieving AUROC 0.75–0.76 for 30-day readmission; Ashfaq et al. [4] achieved AUROC 0.77 with a cost-sensitive LSTM for congestive heart failure (CHF) readmission.

## 2.2 Note-Based and Multimodal Readmission Prediction

Beyond structured features, pre-trained language models applied to clinical text have improved readmission prediction. Huang et al. [16] fine-tuned ClinicalBERT for discharge-based readmission prediction and demonstrated that clinical notes carry complementary signal beyond structured codes. Almeida et al. [1] combined BioClinicalBERT [2] embeddings of diagnosis and procedure titles and discharge summaries with a Facebook AI Similarity Search (FAISS) [21] patient graph and a GraphSAGE [14] classifier, achieving AUROC 0.727 on MIMIC-IV [19] (structured-only baseline 0.704). Tang et al. [36] proposed MM-STGNN, a spatiotemporal graph neural network fusing longitudinal chest radiographs and structured EHR data, achieving AUROC 0.79 on a chest-radiograph subset of MIMIC-IV. Pandey et al. [28] applied a fine-tuned ClinicalT5 model to clinical notes combined with structured features, reporting AUROC 0.68 on a dischargenote subset of MIMIC-IV. He et al. [15] compared structured features against discharge narratives on MIMIC-IV (AUROC 0.65–0.67 for classical ML vs. 0.72 for ClinicalLongformer), and Shakya et al. [34] found word2vec EHR embeddings outperformed one-hot and pre-trained BERT for heart-failure readmission.

## 2.3 Knowledge-Enriched Clinical Prediction

An alternative to note-based encoding is integrating external medical knowledge directly into structured representations, avoiding the dependency on discharge summary availability. Choi et al. [10] proposed GRAM, which learns medical concept representations by attending over ancestors in a medical ontology, and applied it to diagnosis prediction on MIMIC-III [20]. Rasmy et al. [31] pretrained Med-BERT on 28.5 million patient records, improving AUROC by 1.21–6.14% over base RNN models (GRU, Bi-GRU, RETAIN) on two disease-prediction tasks. Xu et al. [39] developed RAM-EHR, achieving a 3.4% average AUROC gain over prior knowledge-enhanced baselines on phenotype and cardiovascularoutcome prediction. Jiang et al. [18] proposed KARE, a KG community retrieval framework that improves MIMIC-IV readmission prediction by up to

12.7% in macro-F1 over the best baseline. Carvalho et al. [8] enriched MIMIC-III ICU stays with annotations from seven biomedical ontologies and learned dense KG embeddings (AUROC 0.827 for ICU readmission); their approach produces opaque learned vectors rather than explicit interpretable features and targets ICU rather than all-cause hospital readmission. These works establish the value of ontological structure for clinical modeling but have not applied multi-source knowledge enrichment as explicit sparse features to structured-only hospital readmission prediction, nor compared it against transformer-based feature extraction on an identical cohort.

## 3 Problem Formulation and Study Motivation

Note-based models can improve discrimination but depend on discharge summaries, which are not recorded for every admission and are costly to process, whereas structured-only models avoid this dependency at a possible accuracy cost. This trade-of motivates the structured-only setting we study next.

We address the standard 30-day readmission prediction task under a structured-only input setting; the task is unchanged from prior work, and only the input representation varies. Formally, it is a binary classification problem: given an admission $a _ { i }$ with structured feature vector $\mathbf { x } _ { i }$ , the goal is to predict $y _ { i } ~ \in ~ \{ 0 , 1 \}$ , where $y _ { i } ~ = ~ 1 \mathrm { { ~ i f ~ } }$ and only if the same patient has a subsequent admission within 30 calendar days of discharge from $a _ { i }$ . Because the instances (admissions) and target are identical to prior work, AUROC values remain directly comparable across feature configurations. Each admission is treated independently; we do not model temporal sequences across admissions.

To investigate the role of feature representation, we define three configurations evaluated under identical conditions, with each component detailed in Section 5:

– B1 (BERT-based): BioClinicalBERT [2] embeddings of ICD diagnosis and procedure titles, concatenated with raw demographic features and per-item lab abnormality rates.

– B2 (Ontology-enriched): demographic features concatenated with multilevel MONDO disease-hierarchy features (Section 5.2), CCS procedure categories, organ-system laboratory aggregates, RxNorm drug-ingredient indicators, and unmapped-ICD features.

– B3 (Combined): B1 reduced to 256 dimensions by truncated SVD (Section 5.6) concatenated with the full B2 matrix.

B1 is a transformer-encoded representation of structured features; B2 replaces these learned embeddings with explicit knowledge-enriched features; B3 tests complementarity. B2 is sparse, requiring approximately 24 times less storage than B1 (176 MB versus 4.2 GB for the full cohort), and each dimension maps to a named clinical concept.

## 4 Dataset and Cohort Construction

We use MIMIC-IV v2.2 [19], a de-identified single-center EHR dataset (2008–2019), extracting structured data from the admissions, patients, diagnoses\_icd, procedures\_icd, prescriptions, and labevents tables; diagnosis codes span ICD-9-CM and ICD-10-CM.

Cohort construction. Starting from all admissions, we exclude patients under 18 and in-hospital deaths, yielding 422,622 admissions across 176,901 patients. We impose no note-related inclusion criteria, so the cohort is not restricted to admissions with available discharge summaries.

Label definition. Following the definition in Section 3, the chronologically last admission of each patient has no subsequent admission on record and therefore cannot be a 30-day readmission, so it receives $y _ { i } = 0$ . The 30-day readmission rate is 20.1%.

Data split. We partition the data into training (60%), validation (20%), and test (20%) sets at the patient level: all admissions of a given patient are assigned to a single split, so no patient appears in more than one split and no information leaks between training and test. The split is stratified by each patient’s maximum label, preserving the readmission ratio across splits. The cohort has mean age $5 6 . 6 \pm 1 9 . 0$ years and is 52.3% female; it is split into 253,254 training, 84,328 validation, and 85,040 test admissions, and contains 25,809 unique ICD diagnosis codes, 12,575 procedure codes, and 9,610 prescription drug names.

## 5 Knowledge-Enriched Readmission Prediction Framework

The B2 ontology-enriched features are built from the same MIMIC-IV tables, mapping raw structured data through four knowledge sources (MONDO, CCS, RxNorm, and organ-system laboratory aggregation) into six feature blocks (demographics, MONDO disease hierarchy, CCS procedures, organ-system labs, RxNorm drugs, and unmapped ICD codes), concatenated into a single sparse matrix; the ablation (Section 8) reports the MONDO block at its three ancestor levels (L1–L3), giving eight rows. Figure 1 provides an overview.

## 5.1 Demographics

The demographic and administrative block encodes administrative fields (admission type, admission location, discharge location, insurance type) and demographic fields (race/ethnicity, marital status) as one-hot vectors, plus age at admission and length of stay as min-max scaled continuous values, yielding 93 features shared across B1, B2, and B3.

![](images/19311425be060f7ef704d7dd1c0d30cb41569ee19c5a8d234d43f83501260129.jpg)  
Fig. 1. Knowledge-enriched feature construction pipeline. Raw MIMIC-IV structured data is enriched through four medical knowledge sources via ontology-based mapping into the sparse B2 feature matrix. Tree and linear models operate directly on the full sparse matrix; neural models use SVD-2048 projections.

## 5.2 MONDO Disease Ontology Mapping

The Mondo Disease Ontology (MONDO) [38] provides a unified disease classification integrating OMIM [3], Orphanet [32], and other sources through the Open Biological and Biomedical Ontologies (OBO) Foundry framework [17]. Following the preprocessing of Almeida et al. [1], we retain the first 10 diagnosis codes per admission (ordered by sequence number), which keeps the structured diagnosis inputs consistent with the B1 comparator; we then map each ICD-9 or ICD-10 code to MONDO concepts via cross-reference (xref) links, attempting code truncation at three levels when an exact match fails. Here a MONDO concept denotes a disease entity (a node in the ontology) to which a diagnosis code is mapped.

For each mapped concept, three ancestor levels are extracted (L1: direct parent, L2: grandparent, L3: great-grandparent), as illustrated in Figure 2. This yields 5,413 L1, 892 L2, and 252 L3 binary features (6,557 mapped features in total). ICD codes with no cross-reference to any MONDO concept, typically administrative, symptom, or factor codes (e.g., ICD-10 Z-chapter “factors influencing health status” codes), are termed unmapped and retained as raw one-hot features in a separate block of 14,905 dimensions. Diagnosis-derived features thus total 21,462 dimensions (6,557 + 14,905); with the remaining blocks the full B2 matrix spans 23,430 (Section 5.6).

## 5.3 CCS Procedure Classification

Raw ICD procedure codes are high-cardinality and sparse. The AHRQ Clinical Classifications Software (CCS) [33] groups them into mutually exclusive, clinically coherent categories, reducing dimensionality while preserving procedural meaning; in our cohort it maps all 12,575 unique procedure codes into 527 singlelevel categories. As in Almeida et al. [1], we map the first five procedures per admission (by sequence number) to these categories as a multi-hot vector of 527 binary features; admissions with fewer than five yield a sparser vector with no padding.

![](images/cc355ae2fbf7490d7f599d6aec1c09ac558386996c7c33c511573c555d5a95ed.jpg)  
Fig. 2. MONDO disease ontology mapping (cardiovascular example). ICD codes are linked to three ancestor levels (L1–L3); a binary feature is generated for each mapped concept.

## 5.4 Organ system Laboratory Aggregation

Laboratory results from labevents are grouped into eight categories by keyword matching against MIMIC-IV’s d\_labitems table [19]: cardiac, hepatic, renal, hematologic, metabolic, inflammatory, coagulation, and a residual category for unmatched tests. For example, serum troponin-I results are assigned to the cardiac system and creatinine results to the renal system. For each system, three features are computed (total count, abnormal count, abnormal percentage), yielding 8 × 3 = 24 continuous features. A system here is an organ or physiological grouping of tests, distinct from the CCS procedure categories; an admission with no results for a system has those three features set to zero (no imputation).

## 5.5 RxNorm Drug Ingredient Mapping

Prescription free-text drug names are normalized and matched against the RxNorm CUI vocabulary [26]. Each matched term is collapsed to its ingredientlevel RxCUI by following has\_ingredient, tradename\_of, and consists\_of relations in the RxNorm relationship file. For example, “metformin 500 mg” normalizes and collapses to RxCUI 41493 (metformin, ingredient). Not every drug matches: 72.1% of prescription records (41.1% of unique strings) map to an ingredient RxCUI, and unmatched drugs are dropped. Each unique ingredient CUI present in an admission becomes a binary feature, yielding 1,324 dimensions.

## 5.6 Feature Integration

The six blocks are concatenated via sparse horizontal stacking into a CSR matrix of 23,430 dimensions. For neural models (MLP, GraphSAGE), the sparse matrix is projected to a dense representation via truncated singular value decomposition (SVD) [13] with k = 2,048 components (95.4% variance retained). Tree and linear models operate directly on the full sparse matrix.

B3 projects B1 (2,481 dimensions) to 256 via truncated SVD, then concatenates with full B2, yielding 23,686 dimensions for tree/linear models; neural models use SVD-2048 of B3 (95.2% variance retained). Preliminary experiments showed that SVD-256 retains only 71% of variance whereas SVD-2048 captures over 95%, yielding approximately two AUROC points improvement for neural models. Neural message-passing models can exploit relational structure between clinically similar patients, which a per-admission feature vector does not represent. We therefore construct a patient graph from the dense embeddings: indexing them with FAISS [21], we retrieve neighbors by exact inner-product (cosine) search and link patients with cosine similarity > 0.9 by an edge (17,401,420 edges over 422,622 patients). The resulting graph is passed to GraphSAGE for message passing.

## 6 Experimental Setup

Models. We evaluate six classifiers: LightGBM [22] (31 leaves, learning rate 0.05, 300 trees), XGBoost [9] (depth 6, learning rate 0.1, 300 trees), L2-regularized Logistic Regression (LR, C = 1), Random Forest [7] (RF, 200 trees, depth 20, min. leaf 5), a Multilayer Perceptron (MLP, hidden 512–256–128, dropout 0.3), and GraphSAGE [14] (two SAGE layers, hidden 64, neighbors [10, 10]). The two neural models use Adam (MLP learning rate $5 \times 1 0 ^ { - 4 }$ , GraphSAGE 10<sup>−5</sup>), batch 1024, and up to 150 epochs with early stopping (patience 10).

Each model is evaluated on all three feature sets (B1, B2, B3): tree and linear models (LightGBM, XGBoost, LR, RF) consume the full sparse matrix, the neural models (MLP, GraphSAGE) its SVD-2048 projection (Table 1, Section 5.6).

Evaluation metrics. The primary metric is AUROC. The secondary metric is Balanced Accuracy (BAcc), both reported for the held-out test set across all models. AUPRC is additionally reported for the best-performing model (Light-GBM on B2) to complement AUROC under class imbalance. The label is imbalanced (20.1% positive); all models address this via class weighting (balanced class weights for LR and RF, scale\_pos\_weight for the boosted trees, and a positive-weighted loss for the neural models). The classification threshold is selected on the validation set by maximizing balanced accuracy. Although the cohort is large, we report 20-fold cross-validation for the best model (LightGBM on B2) to quantify estimate variance and match the protocol of prior work [1]; its mean and 95% CI are reported in Table 2.

Feature configurations. Table 1 summarizes the three feature sets; all share the same cohort, label definition, and patient-level 60/20/20 split.

Hardware. Experiments were conducted on a workstation with an Intel Core Ultra 9 285 (24 cores), 64 GB RAM, and an NVIDIA RTX 4000 SFF Ada (20 GB).

Table 1. Feature set configurations. B1 follows the pipeline of Almeida et al. [1]; B2 is the proposed knowledge-enriched representation; B3 combines both. Neural models use SVD-2048 projections for B2 and B3.
<table><tr><td>Set Components</td><td>Dims Sparsity</td></tr><tr><td>B1 Demo + BERT diag/proc + labs</td><td>2,481 Dense</td></tr><tr><td>B2 Demo + MONDO + CCS + labs + drugs + ICD 23,430</td><td>99.78%</td></tr><tr><td>B3 B1(SVD-256) + B2</td><td>23,686 Mixed</td></tr></table>

Table 2. Held-out fixed-split test AUROC and Balanced Accuracy (BAcc) for six models across three feature configurations. Bold: best feature set per model. Underline: best model per feature set. Under 20-fold cross-validation at the patient level, the best configuration (LightGBM, B2) achieves AUROC 0.743 ± 0.007 (BAcc 0.676). For reference, on a single split Almeida et al. [1] report AUROC 0.704 (structured-only) and 0.727 (with notes), with BAcc 0.649 and 0.667 respectively, on a cohort conditioned on discharge-summary availability, with diferent inclusion criteria.
<table><tr><td>B1</td><td colspan="5">(BERT) B2 (Ontology) B3 (Combined)</td></tr><tr><td>Model</td><td>AUROC BAcc</td><td>AUROC</td><td>BAcc</td><td>AUROC</td><td>BAcc</td></tr><tr><td>LightGBM</td><td>0.7359 0.6726</td><td>0.7412</td><td>0.6739</td><td>0.7396</td><td>0.6769</td></tr><tr><td>XGBoost</td><td>0.7322 0.6674</td><td>0.7374</td><td>0.6714</td><td>0.7359</td><td>0.6732</td></tr><tr><td>MLP</td><td>0.7323 0.6701</td><td>0.7317</td><td>0.6685</td><td>0.7325</td><td>0.6693</td></tr><tr><td>GraphSAGE</td><td>0.7271 0.6679</td><td>0.7239</td><td>0.6626</td><td>0.7260</td><td>0.6627</td></tr><tr><td>LR</td><td>0.7164 0.6602</td><td>0.7049</td><td>0.6538</td><td>0.7100</td><td>0.6564</td></tr><tr><td>RF</td><td>0.7007 0.6450</td><td>0.6966</td><td>0.6421</td><td>0.7109</td><td>0.6552</td></tr></table>

## 7 Results

Table 2 reports AUROC and Balanced Accuracy for all six classifiers across the three feature configurations on the held-out test set.

Feature representation and model comparison. B2 (ontology-enriched) achieves the highest AUROC for gradient-boosted tree models: LightGBM reaches 0.7412 and XGBoost 0.7374, both significantly outperforming their B1 counterparts (bootstrap, 1,000 iterations; LightGBM: mean $\varDelta = + 0 . 0 2 8$ , 95% CI [+0.026, +0.031], $p < 0 . 0 0 1$ ; LR: mean $\varDelta = + 0 . 0 1 0$ , 95% CI [+0.006, +0.014], p < 0.001). We report the bootstrap test for LightGBM (best tree model) and LR (linear baseline) as family representatives; the other models follow the same B2-versus-B1 direction (Table 2). Dense-input models show the opposite pattern: Graph-SAGE (0.7271 on B1 vs. 0.7239 on B2) and LR (0.7164 vs. 0.7049) prefer B1. B3 does not consistently outperform either single-source configuration. LightGBM on B2 is the best result across all 18 configurations (AUROC 0.7412, AUPRC 0.441).

Cross-study context. The best fixed-split result (LightGBM B2, AUROC 0.7412), and the 20-fold-CV value of 0.743±0.007 (95% CI: 0.740–0.746), fall in the range of previously reported structured-only and note-inclusive values on this dataset (Table 2). This is a cross-study comparison rather than a controlled one: the reference cohort is conditioned on discharge-summary availability (303,571 vs. 422,622 admissions) and the absence of per-fold variance precludes a significance test, so we describe our result as competitive with, not superior to, note-based approaches (Section 10).

Table 3. Feature block importance for LightGBM on B2 (SVD-256, AUROC 0.7399 baseline). Solo: AUROC from one block alone. LOBO (leave-one-block-out): AUROC change when removing one block. Gain: the share of total model gain attributable to a block, where a feature’s gain is the summed reduction in training loss across all tree splits that use it, normalized by the total gain over all features. Perm.: AUROC change under random permutation.
<table><tr><td>Feature block</td><td>Dims Solo LOBO Δ Gain (%) Perm. ∆</td></tr><tr><td>Demographics 93 0.6985</td><td>-0.0369 48.5 -0.1117</td></tr><tr><td>Organ system labs</td><td>24 0.6273 -0.0019 11.4 -0.0227</td></tr><tr><td>Drug RxNorm 1,324 0.6576</td><td>-0.0041 12.8 -0.0192</td></tr><tr><td></td><td>-0.0040 14.4 -0.0142</td></tr><tr><td>Unmapped ICD 14,905 0.6569</td><td>-0.0049 8.5 -0.0105</td></tr><tr><td>MONDO L1 5,413 0.6499</td><td>-0.0013 4.3 -0.0035</td></tr><tr><td>CCS procedures 527 0.6130</td><td>+0.0004 0.05 -0.0001</td></tr><tr><td>MONDO L2 892 0.5534 MONDO L3 252 0.5298</td><td>-0.0000 0.07 -0.0001</td></tr></table>

## 8 Ablation and Comparative Analysis

To understand which knowledge sources drive performance, we apply four interpretability methods to LightGBM on B2. The ablation uses SVD-256 projected B2 features; relative block rankings are consistent with the full-sparse evaluation in Section 7.

Table 3 reports the results across eight feature blocks.

Demographics dominate. Demographics account for 48.5% of LightGBM gain and produce the largest permutation importance drop, as shown in Figure 3. Training on demographics alone yields AUROC 0.6985, comparable to published structured-only baselines such as LACE [37] and classical ML on MIMIC-IV [15].

Knowledge source contributions. Organ system labs, drug ingredient features, unmapped ICD codes, and MONDO L1 each contribute 1.0 to 2.3 percentage points of permutation importance, with partial redundancy under LOBO removal (Table 3). Gain-based importance is highest for unmapped ICD codes and drug ingredient features (Table 3). MONDO L2 produces a slight positive LOBO efect and L3 essentially zero; direct disease categories (L1) are therefore suficient, and broader levels are retained only to preserve the generality of the mapping.

![](images/a3023bf58106c16a2709fb95c990b550e5bbdca943d78cfa4bf74b5d4c601e0b.jpg)  
Fig. 3. Block-level permutation importance for LightGBM on B2. Bar length indicates the drop in test AUROC when the block’s features are randomly permuted.

Table 4. Subgroup test-set AUROC for LightGBM on B2 (n = 85,040), with 95% bootstrap confidence intervals (2,000 resamples, fixed seed). Prev. (prevalence) is the subgroup 30-day readmission rate. Race categories follow MIMIC-IV groupings, with “Other/Unk.” denoting other or unknown race/ethnicity.
<table><tr><td>Subgroup</td><td>Prev.</td><td>AUROC [95% CI]</td></tr><tr><td>Overall</td><td>0.202</td><td>0.741 [0.737, 0.745]</td></tr><tr><td>Sex</td><td></td><td></td></tr><tr><td>Female</td><td>0.189</td><td>0.746 [0.740, 0.752]</td></tr><tr><td>Male</td><td>0.215</td><td>0.734 [0.728, 0.740]</td></tr><tr><td>Race</td><td></td><td></td></tr><tr><td>White</td><td>0.202</td><td>0.727 [0.722, 0.732]</td></tr><tr><td>Black</td><td>0.220</td><td>0.753 [0.743, 0.763]</td></tr><tr><td>Hispanic</td><td>0.214</td><td>0.773 [0.758, 0.787]</td></tr><tr><td>Asian</td><td>0.186</td><td>0.777 [0.754, 0.798]</td></tr><tr><td>Other/Unk.</td><td>0.154</td><td>0.779 [0.763, 0.795]</td></tr></table>

<table><tr><td>Subgroup</td><td>Prev.</td><td>AUROC [95% CI]</td></tr><tr><td>Insurance</td><td></td><td></td></tr><tr><td>Medicare</td><td>0.217</td><td>0.711 [0.704, 0.718]</td></tr><tr><td>Medicaid</td><td>0.258</td><td>0.750 [0.738, 0.762]</td></tr><tr><td>Other</td><td>0.181</td><td>0.754 [0.748, 0.760]</td></tr><tr><td>Age</td><td></td><td></td></tr><tr><td>&lt;40</td><td>0.192</td><td>0.792 [0.783, 0.800]</td></tr><tr><td>40-54</td><td>0.221</td><td>0.752 [0.744, 0.761</td></tr><tr><td>55-64</td><td>0.214</td><td>0.734 [0.724, 0.743]</td></tr><tr><td>65-74</td><td>0.201</td><td>0.724 [0.714, 0.734]</td></tr><tr><td>75-84</td><td>0.188</td><td>0.674 [0.662, 0.686]</td></tr><tr><td>85+</td><td>0.164</td><td>0.668 [0.649, 0.687]</td></tr></table>

## 8.1 Subgroup performance

Because demographic blocks dominate model gain (Section 8), we audit whether discrimination varies across subgroups. Table 4 reports test-set AUROC with 95% bootstrap confidence intervals for LightGBM on B2 by sex, insurance, race, and age.

Discrimination is stable across sex, insurance, and race, with no subgroup near chance, and is not lower for racial-minority or Medicaid groups. The main gradient is by age: AUROC declines from 0.79 in the youngest band to 0.67 for patients aged 85 and over, indicating that readmission in the oldest patients depends on factors only partly captured by structured and ontology features. Comparable AUROC across groups does not imply equitable deployment: at a single operating threshold, false- and true-positive rates vary with subgroup prevalence, and detection is lowest in the oldest patients. Confidence intervals for the smallest groups are wide and limit interpretation.

## 9 Discussion

Why ontology features outperform BERT for tree models. Gradient-boosted trees partition the feature space through axis-aligned splits, so the sparse ontology features in B2, where each dimension is a single clinical concept, enable semantically meaningful splits, whereas BERT embeddings compress semantics into dense vectors that mix latent factors [12,35]. Conversely, GraphSAGE and LR perform better on B1 (0.3–1.2 AUROC points), benefiting from dense low-dimensional structure for gradient-based optimization; the knowledge-enrichment benefit is thus specific to the model family.

Hierarchy depth and demographic dominance. Direct parent-level categories (L1) capture suficient granularity; broader levels (L2, L3) add no discriminative value (Section 8). This does not conflict with the benefit of knowledge enrichment: the gain over demographics comes from coarse, interpretable groupings (L1 categories, CCS, organ-system labs, drug ingredients), not deep ontological depth, so a negligible L2/L3 contribution is consistent with a positive overall contribution. Demographics dominate prediction, reflecting the importance of admission type, insurance status, and length of stay [23]; the four knowledge sources add roughly four AUROC points above demographics alone, though this dominance limits isolation of the knowledge-enrichment contribution.

## 10 Limitations and Ethical Considerations

Limitations. All experiments use single-institution MIMIC-IV, so generalizability to other populations and coding practices is unvalidated. Cross-validation was run only for LightGBM on B2; other comparisons rely on fixed-split diferences without confidence intervals, and the LOBO analysis quantifies block-removal efects but not additive contributions of individual sources. A configuration combining ontology features with clinical notes was not evaluated; the all-cause label does not separate planned from unplanned readmissions (predominantly emergency and urgent here); and the cross-study comparison with Almeida et al. [1] is qualified by difering inclusion criteria (422,622 vs. 303,571 admissions) and cannot be tested for significance without their per-fold variance.

Ethical considerations and privacy compliance. MIMIC-IV is de-identified under the HIPAA Safe Harbor standard and released under a PhysioNet Credentialed Data Use Agreement [6,19]; our pipeline consumes only the de-identified release without re-identification. Demographic features, including insurance type and race/ethnicity, contribute to model gain and encode socioeconomic determinants that reflect documented systemic health inequities [27]. A subgroup audit (Section 8.1) shows discrimination is broadly stable across groups, while error rates at a fixed threshold difer by group; deploying such models for resource allocation therefore requires fairness auditing and prospective validation before clinical use.

## 11 Conclusion

We proposed a knowledge-enriched feature representation for 30-day readmission prediction that integrates four medical knowledge sources with structured EHR data, without clinical notes. Under 20-fold cross-validation, LightGBM on B2 achieves AUROC 0.743 without the language-model inference that note-based methods require, using a sparse representation that needs roughly 24 times less storage than the BERT features (Section 3). Interpretability analysis shows that demographics, together with the knowledge-derived features (direct L1 disease categories, organ-system labs, and drug ingredients), drive prediction, while only the deeper MONDO ontology levels (L2, L3) contribute negligibly. Overall, enriching structured EHR data with medical knowledge yields an interpretable, lightweight readmission model that performs on par with note-based approaches, without requiring clinical notes. Future work includes multi-site validation, evaluation of combined ontology and note features, and fairness auditing across demographic subgroups.

## References

1. Almeida, T., Moreno, P., Barata, C.: Prediction of 30-day hospital readmission with clinical notes and ehr information. In: Iberian Conference on Pattern Recognition and Image Analysis. pp. 220–232. Springer (2025)

2. Alsentzer, E., Murphy, J.R., Boag, W., Weng, W.H., Jin, D., Naumann, T., McDermott, M.: Publicly available clinical bert embeddings. arXiv preprint arXiv:1904.03323 (2019)

3. Amberger, J.S., Bocchini, C.A., Schiettecatte, F., Scott, A.F., Hamosh, A.: Omim. org: Online mendelian inheritance in man (omim®), an online catalog of human genes and genetic disorders. Nucleic acids research 43(D1), D789–D798 (2015)

4. Ashfaq, A., Sant’Anna, A., Lingman, M., Nowaczyk, S.: Readmission prediction using deep learning on electronic health records. Journal of biomedical informatics 97, 103256 (2019)

5. Balasubramanian, I., Andres, E.B., Malhotra, C.: Outpatient follow-up and 30-day readmissions: a systematic review and meta-analysis. JAMA Network Open 8(11), e2541272 (2025)

6. Benitez, K., Malin, B.: Evaluating re-identification risks with respect to the HIPAA privacy rule. Journal of the American Medical Informatics Association 17(2), 169– 177 (2010)

7. Breiman, L.: Random forests. Machine learning 45(1), 5–32 (2001)

8. Carvalho, R.M., Oliveira, D., Pesquita, C.: Knowledge graph embeddings for icu readmission prediction. BMC medical informatics and decision making 23(1), 12 (2023)

9. Chen, T., Guestrin, C.: Xgboost: A scalable tree boosting system. In: Proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining. pp. 785–794 (2016)

10. Choi, E., Bahadori, M.T., Song, L., Stewart, W.F., Sun, J.: Gram: graph-based attention model for healthcare representation learning. In: Proceedings of the 23rd ACM SIGKDD international conference on knowledge discovery and data mining. pp. 787–795 (2017)

11. Donzé, J.D., Williams, M.V., Robinson, E.J., Zimlichman, E., Aujesky, D., Vasilevskis, E.E., Kripalani, S., Metlay, J.P., Wallington, T., Fletcher, G.S., et al.: International validity of the hospital score to predict 30-day potentially avoidable hospital readmissions. JAMA internal medicine 176(4), 496–502 (2016)

12. Grinsztajn, L., Oyallon, E., Varoquaux, G.: Why do tree-based models still outperform deep learning on typical tabular data? Advances in neural information processing systems 35, 507–520 (2022)

13. Halko, N., Martinsson, P.G., Tropp, J.A.: Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. SIAM review 53(2), 217–288 (2011)

14. Hamilton, W., Ying, Z., Leskovec, J.: Inductive representation learning on large graphs. Advances in neural information processing systems 30 (2017)

15. He, Z., Li, H., Tian, S., Wen, J., Yuan, G., Li, A.: A comparative study of structured and narrative ehr data for 30-day readmission risk assessment. Electronics 14(20), 4033 (2025)

16. Huang, K., Altosaar, J., Ranganath, R.: Clinicalbert: Modeling clinical notes and predicting hospital readmission. arXiv preprint arXiv:1904.05342 (2019)

17. Jackson, R., Matentzoglu, N., Overton, J.A., Vita, R., Balhof, J.P., Buttigieg, P.L., Carbon, S., Courtot, M., Diehl, A.D., Dooley, D.M., et al.: Obo foundry in 2021: operationalizing open data principles to evaluate ontologies. Database 2021, baab069 (2021)

18. Jiang, P., Xiao, C., Jiang, M., Bhatia, P., Kass-Hout, T., Sun, J., Han, J.: Reasoning-enhanced healthcare predictions with knowledge graph community retrieval. arXiv preprint arXiv:2410.04585 (2024)

19. Johnson, A.E., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S., Pollard, T.J., Hao, S., Moody, B., Gow, B., et al.: Mimic-iv, a freely accessible electronic health record dataset. Scientific data 10(1), 1 (2023)

20. Johnson, A.E., Pollard, T.J., Shen, L., Lehman, L.w.H., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Anthony Celi, L., Mark, R.G.: Mimic-iii, a freely accessible critical care database. Scientific data 3(1), 1–9 (2016)

21. Johnson, J., Douze, M., Jégou, H.: Billion-scale similarity search with gpus. IEEE transactions on big data 7(3), 535–547 (2019)

22. Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., Liu, T.Y.: Lightgbm: A highly eficient gradient boosting decision tree. Advances in neural information processing systems 30 (2017)

23. Kum Ghabowen, I., Epane, J.P., Shen, J.J., Goodman, X., Ramamonjiarivelo, Z., Zengul, F.D.: Systematic review and meta-analysis of the financial impact of 30-day readmissions for selected medical conditions: a focus on hospital quality performance. In: Healthcare. vol. 12, p. 750. MDPI (2024)

24. Makohon, I., Najafi, M., Wu, J., Brochhausen, M., Li, Y.: Enhancing clinical note generation with ICD-10, clinical ontology knowledge graphs, and chain-of-thought prompting using GPT-4. Journal of Computational Biology p. 15578666251411390 (2025)

25. Muchiri, S., Azadeh-Fard, N., Pakdil, F.: The analysis of hospital readmission rates after the implementation of hospital readmissions reduction program. Journal of Patient Safety 18(3), 237–244 (2022)

26. Nelson, S.J., Zeng, K., Kilbourne, J., Powell, T., Moore, R.: Normalized names for clinical drugs: Rxnorm at 6 years. Journal of the American Medical Informatics Association 18(4), 441–448 (2011)

27. Obermeyer, Z., Powers, B., Vogeli, C., Mullainathan, S.: Dissecting racial bias in an algorithm used to manage the health of populations. Science 366(6464), 447–453 (2019)

28. Pandey, S.R., Tile, J.D., Oghaz, M.M.D.: Predicting 30-day hospital readmissions using clinicalt5 with structured and unstructured electronic health records. PLoS One 20(9), e0328848 (2025)

29. Pattar, B.S., Ackroyd, A., Sevinc, E., Hecker, T., Turino Miranda, K., McClurg, C., Weekes, K., James, M.T., Pannu, N., Ravani, P., et al.: Electronic health record interventions to reduce risk of hospital readmissions: a systematic review and metaanalysis. JAMA Network Open 8(7), e2521785 (2025)

30. Rajkomar, A., Oren, E., Chen, K., Dai, A.M., Hajaj, N., Hardt, M., Liu, P.J., Liu, X., Marcus, J., Sun, M., et al.: Scalable and accurate deep learning with electronic health records. NPJ digital medicine 1(1), 18 (2018)

31. Rasmy, L., Xiang, Y., Xie, Z., Tao, C., Zhi, D.: Med-bert: pretrained contextualized embeddings on large-scale structured electronic health records for disease prediction. NPJ digital medicine 4(1), 86 (2021)

32. Rath, A., Olry, A., Dhombres, F., Brandt, M.M., Urbero, B., Ayme, S.: Representation of rare diseases in health information systems: the orphanet approach to serve a wide range of end users. Human mutation 33(5), 803–808 (2012)

33. Salsabili, M., Kiogou, S., Adam, T.J.: The evaluation of clinical classifications software using the national inpatient sample database. AMIA Summits on Translational Science Proceedings 2020, 542 (2020)

34. Shakya, P., Khaneja, A., Wagholikar, K.B.: Predicting 30-days hospital readmission for patients with heart failure using electronic health record embeddings: Comparative evaluation. JMIR Medical Informatics 13, e73020 (2025)

35. Shwartz-Ziv, R., Armon, A.: Tabular data: Deep learning is not all you need. Information fusion 81, 84–90 (2022)

36. Tang, S., Tariq, A., Dunnmon, J.A., Sharma, U., Elugunti, P., Rubin, D.L., Patel, B.N., Banerjee, I.: Predicting 30-day all-cause hospital readmission using multimodal spatiotemporal graph neural networks. IEEE Journal of Biomedical and Health Informatics 27(4), 2071–2082 (2023)

37. Van Walraven, C., Dhalla, I.A., Bell, C., Etchells, E., Stiell, I.G., Zarnke, K., Austin, P.C., Forster, A.J.: Derivation and validation of an index to predict early death or unplanned readmission after discharge from hospital to the community. Cmaj 182(6), 551–557 (2010)

38. Vasilevsky, N.A., Matentzoglu, N.A., Toro, S., Flack IV, J.E., Hegde, H., Unni, D.R., Alyea, G.F., Amberger, J.S., Babb, L., Balhof, J.P., et al.: Mondo: unifying diseases for the world, by the world. MedRxiv pp. 2022–04 (2022)

39. Xu, R., Shi, W., Yu, Y., Zhuang, Y., Jin, B., Wang, M.D., Ho, J., Yang, C.: Ramehr: Retrieval augmentation meets clinical predictions on electronic health records. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). pp. 754–765 (2024)