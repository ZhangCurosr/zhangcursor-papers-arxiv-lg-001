# RETHINKING CORRECTNESS FOR UNCERTAINTY ESTIMATION IN CLINICALPREDICTION WITH VISION–LANGUAGE MODELS

Mingcheng Zhu

Jinning Liang

Tingting Zhu

University of Oxford, Oxford, United Kingdom mingcheng.zhu@eng.ox.ac.uk

## ABSTRACT

Vision–language models are increasingly explored for clinical prediction from electronic health records and medical images, where identifying unreliable predictions is important for safe deployment. Uncertainty estimation (UE) enables detecting such predictions, but its evaluation depends on a correctness criterion that determines whether each model output is correct. If this criterion disagrees with human judgement or distorts downstream UE performance, conclusions about model reliability can be misleading. We introduce a two-axis framework that evaluates correctness criteria by their agreement with human judgements and fidelity to human-referenced UE performance. We assess eight criteria across three clinical prediction tasks and three models using 450 predictions annotated by two reviewers. Across the audited tasks, canonical exact matching (EM) achieved the highest observed human agreement and lowest UE distortion, while the BERT-based matching (BEM) and LLM-judge also showed strong human agreement. Across four UE methods and 23,254 clinical predictions, criterion choice changed errordetection AUROC by up to 0.146 and reversed the relative ranking of UE methods. The LLM-judge also selectively accepted invalid or uncertain outputs, accepting 16 of 30 such human-identified errors. These results demonstrate that correctness assessment is an integral component of clinical UE evaluation and should be validated before UE methods are compared. Code is available at https://github.com/JasonZuu/EHR-Correctness.

Index Terms— Clinical prediction, multimodal signal processing, uncertainty estimation, correctness validation

## 1. INTRODUCTION

Vision–language models (VLMs) are increasingly used for clinical prediction by integrating information from medical images and textual electronic health records (EHRs) [1, 2]. As illustrated in Fig. 1(a), these models can leverage multimodal information to support prediction, while reliable assessment of their outputs remains essential for clinical use. Uncertainty estimation (UE) addresses this need by assigning an uncertainty score to each model prediction, as shown in Fig. 1(b). However, evaluating UE requires a correctness criterion that determines whether each prediction is correct, and the validity of this criterion is often assumed rather than explicitly validated [3]. The reliability of UE evaluation therefore depends not only on the uncertainty estimator itself, but also on how prediction correctness is defined.

Automating correctness assessment is challenging because lexical similarity does not necessarily reflect semantic equivalence between model-generated predictions and reference answers [4, 5]. Several methods have been proposed to provide correctness labels. Canonical exact matching (EM) provides a simple and transparent criterion but may reject valid reformulations, whereas the learned BERT-based matching (BEM) can better accommodate semantic variation [6]. Token-overlap metrics can be unreliable for verbose instruction-following responses [4]. LLM-based judges offer greater semantic flexibility and have shown strong agreement with human assessment in clinical summarisation [7], but broader studies have also identified systematic biases in LLM judging [8]. Prior work has established that correctness functions can affect UE evaluation [9, 10]. We instead study correctness criteria themselves as objects of validation, jointly assessing prediction-level human agreement and their downstream distortion of UE evaluation in clinical prediction. Importantly, similar label agreement does not imply similar UE fidelity: two criteria with identical confusion counts may mislabel predictions at different positions in the uncertainty ranking and therefore induce different AUROC distortions.

![](images/cb7271373483a474eda87fafbc045bc66c3f12d0a298d665a1adba69dde2f7be.jpg)  
Fig. 1. Overview of the proposed evaluation framework. (a) A clinical model predicts from an EHR and, where available, an image. (b) An answer-only UE method assigns an uncertainty score, while a correctness criterion compares the prediction with the reference answer to define the binary outcome used for UE evaluation. (c) Joint assessment of agreement with human judgements and preservation of human-referenced UE performance.

Conversely, preserving aggregate UE performance does not ensure that individual predictions are assessed correctly. We therefore jointly assess correctness criteria through human-label agreement and human-referenced UE performance fidelity. Our contributions are threefold:

• We provide a consensus-annotated audit of eight correctness criteria across three clinical prediction tasks and three models, using 450 patient-deduplicated predictions.

• We establish two complementary validity requirements for correctness criteria, agreement with human judgements and preservation of human-referenced UE performance, and operationalise them in a two-axis framework.

• We quantify criterion-dependent variation in UE estimates on 23,254 parseable predictions and characterise observed false acceptance across clinical response error categories, with explicit scope and uncertainty limitations.

## 2. METHODOLOGY

## 2.1. Problem Formulation

We consider clinical prediction from textual EHRs and, where available, medical images using a VLM. For case i, let $\boldsymbol { x } _ { i } ~ = ~ ( e _ { i } , v _ { i } )$ denote the textual EHR $e _ { i }$ and medical image v<sub>i</sub>, with $v _ { i } = \emptyset$ for cases without an image. Given a task prompt $q _ { t } , \mathbf { a }$ VLM with parameters θ generates a textual prediction $y _ { i } \sim p _ { \theta } ( \cdot \mid x _ { i } , q _ { t } )$

To estimate the uncertainty of the VLM’s generation, a UE method u assigns a scalar score u<sub>i</sub> to the generated answer $y _ { i } ,$ with larger values indicating greater uncertainty. Evaluating this score requires a binary outcome indicating whether the prediction is correct. Given a reference answer $\widehat { y } _ { i }$ , a correctness criterion f provides this outcome:

$$
c _ { i } = f ( y _ { i } , { \widehat { y } } _ { i } ) \in \{ 0 , 1 \} ,\tag{1}
$$

where $c _ { i } = 1$ denotes a correct prediction and $c _ { i } = 0$ denotes an incorrect prediction. The resulting correctness defines the reference outcomes used to evaluate whether higher uncertainty is associated with incorrect predictions. Because these outcomes depend on the chosen correctness criterion, different criteria $f$ may yield different estimates of UE performance even when the model predictions and uncertainty scores are the same. We therefore evaluate correctness criteria along two complementary dimensions: agreement with human judgements of prediction correctness and fidelity to UE performance measured using human-derived correctness labels.

## 2.2. Human Validation Consistency

Human Validation Consistency measures agreement between an automatic correctness criterion and human judgements using balanced accuracy. Let $c _ { i } ^ { ( h ) }$ and $c _ { i } ^ { ( f ) }$ denote the human and criterion-derived correctness labels, respectively. Let G contain the G dataset–model groups, and let $\mathcal { T } _ { g }$ contain the human-reviewed predictions in group $g .$ We characterise the audited sample using unit analysis weights, $w _ { i } = 1$ , within each dataset–model group. Within each group, we define

$$
\begin{array} { c } { { H _ { f , g } = \displaystyle \frac { \sum _ { i \in \mathcal { Z } _ { g } } w _ { i } c _ { i } ^ { ( h ) } c _ { i } ^ { ( f ) } } { 2 \cdot \sum _ { i \in \mathcal { Z } _ { g } } w _ { i } c _ { i } ^ { ( h ) } } } } \\ { { + \displaystyle \frac { \sum _ { i \in \mathcal { T } _ { g } } w _ { i } \big ( 1 - c _ { i } ^ { ( h ) } \big ) \big ( 1 - c _ { i } ^ { ( f ) } \big ) } { 2 \cdot \sum _ { i \in \mathcal { Z } _ { g } } w _ { i } \big ( 1 - c _ { i } ^ { ( h ) } \big ) } . } } \end{array}\tag{2}
$$

The two terms measure agreement on human-labelled correct and incorrect predictions, respectively. We average equally across groups to obtain $\begin{array} { r } { H _ { f } \ = \ G ^ { - \hat { 1 } } \sum _ { g \in \mathcal { G } } \mathbf { \overbar { H } } _ { f , g . } } \end{array}$ Both $H _ { f , g }$ and $H _ { f }$ lie in [0, 1], with larger values indicating stronger agreement with human judgements. Agreement at the prediction level does not necessarily imply that a criterion preserves the measured performance of UE methods. Balanced accuracy gives equal importance to sensitivity for human-correct responses and specificity for human-incorrect responses within each group. The subsequent macro-average also gives each dataset–model combination equal influence. This separates the summary from differences in group size, while retaining the task and model as the unit of comparison. These weights describe performance on the constructed audit; they do not recover population performance from the EM-stratified sampling design. Consequently, the reported human-agreement estimates should be interpreted together with the sampling protocol and the group-specific results.

## 2.3. UE Performance Fidelity

UE Performance Fidelity measures whether a correctness criterion preserves UE performance relative to human-derived correctness labels. For method u and group $^ { g , }$ let $A _ { u , g } ^ { ( f ) }$ and $A _ { u , g } ^ { ( h ) }$ denote AUROCs computed on the same human-reviewed predictions and uncertainty scores. These use criterion-derived error labels $1 - c _ { i } ^ { ( f ) }$ and humanderived error labels $1 - c _ { i } ^ { ( h ) }$ , respectively, with unit analysis weights. We quantify their difference using the following signed distortion:

$$
\Delta _ { f , u , g } = A _ { u , g } ^ { ( f ) } - A _ { u , g } ^ { ( h ) } .\tag{3}
$$

Positive values indicate higher measured AUROC than the human reference, while negative values indicate lower measured $\mathrm { { A U } } .$ ROC. Within each group, we average absolute distortion across $N _ { u }$ UE methods and define fidelity as

$$
D _ { f , g } = \frac { 1 } { N _ { u } } \sum _ { u = 1 } ^ { N _ { u } } | \Delta _ { f , u , g } | , \qquad F _ { f , g } = 1 - D _ { f , g } .\tag{4}
$$

We then average equally across groups to get the grouped score $\begin{array} { r } { D _ { f } = \frac { 1 } { G } \sum _ { q \in \mathcal { G } } { D _ { f , g } } } \end{array}$ and $F _ { f } = 1 - D _ { f }$ . We report $F _ { f }$ for visualisation so that both framework axes increase with criterion suitability. Since AUROC lies in $[ 0 , 1 ] , F _ { f } \in [ 0 , 1 ]$ , with larger values indicating closer agreement with human-referenced AUROCs. A fidelity of one indicates identical AUROCs for the evaluated methods and groups, but does not imply identical correctness labels. Balanced accuracy summarises label agreement but does not specify where disagreements occur in the uncertainty ordering. Because AUROC depends on the score ordering of correct–incorrect pairs, criteria with identical confusion counts can yield different AUROC distortions; conversely, a small mean distortion does not ensure that the underlying correctness labels agree with human judgements.

## 2.4. Human Annotation Protocol

We construct a human reference from 450 predictions across nine dataset–model groups. Each group contains 50 predictions, stratified equally between EM-accepted and EM-rejected cases to ensure coverage of both canonical-match outcomes rather than estimate their population prevalence. Two trained annotators with clinical-AI research experience independently annotated correctness and a primary error category using the task, candidate labels, reference labels, model answer, reasoning and available model output. Incorrect responses were categorised as missing/partial, extra/wrong or invalid/uncertain; when omissions and additional labels co-occurred, the response was assigned to extra/wrong. All disagreements were resolved jointly. Before discussion, correctness agreement was 445/450 (98.89%; Cohen’s κ=0.978), and error-category agreement was 206/216 (95.37%) among responses both annotators judged incorrect. The final reference combines 435 independent agreements and 15 joint decisions, covering five correctness disagreements and ten additional category disagreements. Joint decisions were recorded separately from independent annotations. The consensus reference contains 229 correct responses and 221 incorrect responses across the nine dataset–model groups.

![](images/dbb99be0af6b796e1c82b1fbda43c98077a1a4ce4f6c96e597c85071448dea03.jpg)  
(a) MCMED (ED disposition)

![](images/f606adc81437fe3fd1cb70401eb9994d054f58453526198d407ffbdd33cad785.jpg)  
(b) EHRSHOT (Diagnosis prediction)

![](images/4227b75080e3a14e9ac4d46c16f604b39c49cfe9ccc24bdccec5bf000750eec9.jpg)  
(c) MIMIC-IV (Phenotyping)  
Fig. 2. Task-specific joint assessment of correctness criteria against the consensus audit. H is balanced accuracy averaged equally across three models; $F$ is one minus absolute human-referenced AUROC distortion averaged across three models and four UE methods. Points use equal case weights within each group. Higher values indicate greater human agreement and UE performance fidelity, respectively.

Automatic correctness decisions were hidden during independent annotation. Reviewers assessed whether the answer satisfied the task and reference labels using the information presented in the annotation interface. They did not independently re-evaluate the original clinical records or images. Sampling equal numbers of EMaccepted and EM-rejected predictions ensured that both matching outcomes were examined.

## 3. RESULTS

## 3.1. Experimental Setup

We evaluate three tasks with predefined candidate-label spaces: prediction of first diagnoses within the following year on EHRSHOT [11] (six disease labels), ED disposition on MCMED [12] (four outcomes), and ICU phenotyping on MIMIC-IV [13] (five labels). EHRSHOT and MIMIC-IV permit multiple labels or none; MCMED requires one outcome. MIMIC-IV uses the first 24 hours of ICU events to predict phenotypes at ICU discharge. MCMED uses events recorded before the disposition decision and ECG images; only this task includes image inputs. Models return JSON with reasoning and a string answer naming candidate diagnoses or dispositions. EHRs are represented as textual event streams [14, 15]. We evaluate Gemma-4 E4B, Gemma-4 26B-A4B and MedGemma-1.5 4B on each task [16, 17].

We compare eight correctness criteria. Specifically, EM normalises Unicode, case, whitespace and punctuation, maps the complete extracted answer to canonical candidate names, and compares unordered label sets without synonym expansion. Literal “none” denotes the empty set. ROUGE-L uses thresholds ≥0.20, ≥0.30 and ≥0.40 [18]; BEM uses $\ge 0 . 5 0$ [6]. Microsoft and Potsawee NLI apply per-label entailment thresholds of ≥0.50 followed by exact label-set comparison [19, 20]. The Gemma-4 31B judge maps the extracted answer to candidate labels without seeing the reference labels; a deterministic step checks set equality and invalid-output flags [16, 21]. The judge receives the task question and candidate labels, but not the source clinical record, image or generation reasoning. Acceptance requires label-set equality and no out-of-space labels or contradictions; malformed judge outputs are not accepted as correct predictions.

We evaluate four answer-only UE methods: Self-Certainty [22], LogTokU [23], Answer Entropy [24] and Semantic Entropy [5]. We assess criterion sensitivity on the full prediction set using errordetection AUROC. This metric quantifies how well uncertainty scores rank incorrect predictions above correct ones, without requiring a decision threshold or a common score scale across UE methods. Our use of AUROC targets a specific question: whether changing the correctness labels changes the apparent ordering quality of a fixed uncertainty score. Predictions and UE scores are held fixed when criteria are compared, so the observed differences arise from the evaluation labels. This analysis does not assess whether uncertainty values are calibrated probabilities, nor does it select an operating threshold for clinical use.

## 3.2. RQ1: Human Agreement and UE Fidelity

This experiment assesses the suitability of correctness criteria through human agreement and UE fidelity. We compare eight criteria on the 450-prediction consensus audit using $H _ { f }$ and $F _ { f } = 1 - D _ { f } ,$ with unit case weights and nine equally weighted dataset–model groups. Figure 2 presents the two-axis results by task. We compare BEM and the judge with EM using 10,000 paired bootstrap replicates that preserve each group’s EM quotas and share sampled cases across criteria and UE methods. The intervals measure resampling stability within this audit, rather than full-population uncertainty.

EM achieved the highest macro balanced accuracy and lowest mean absolute AUROC distortion, with point estimates of 0.992 and 0.005, respectively. BEM and the judge achieved balanced accuracies of 0.980 and 0.962, with distortions of 0.017 and 0.021. Relative to EM, distortion increased by 0.012 for BEM (95% paired bootstrap interval: −0.001 to 0.026) and 0.016 for the judge (0.004 to 0.037). Using Bonferroni-adjusted 97.5% CIs for the two comparisons, the judge–EM difference remained positive, whereas the BEM–EM interval crossed zero, so a difference in distortion was not established.

Table 1. Criterion-conditioned error-detection AUROC on the full parseable prediction set.
<table><tr><td colspan="4">UE Method Correctness EHRSHOT MCMED MIMIC-IV</td></tr><tr><td>Self- Certainty</td><td>LLM-judge BEM EM</td><td>0.424 0.574 0.422 0.607 0.425 0.597</td><td>0.525 0.512 0.543</td></tr><tr><td>LogTokU</td><td>LLM-judge BEM EM</td><td>0.251 0.262 0.253</td><td>0.636 0.586 0.643 0.440 0.631 0.582</td></tr><tr><td>Answer Entropy</td><td>LLM-judge BEM EM</td><td>0.667 0.639 0.671</td><td>0.577 0.606 0.584 0.518 0.593 0.618</td></tr><tr><td>Semantic Entropy</td><td>LLM-judge BEM EM</td><td>0.634 0.611 0.637</td><td>0.573 0.604 0.582 0.522 0.589 0.616</td></tr></table>

The directional errors further distinguish these criteria. Relative to the human annotations, EM falsely rejected 4/229 correct responses and accepted 0/221 incorrect responses, compared with 0/229 and 9/221 for BEM and 0/229 and 16/221 for the LLM-judge. Among these three criteria, EM exhibited the strictest acceptance behaviour in the audit, rejecting all human-labelled incorrect responses at the cost of rejecting four correct responses. BEM and the judge accepted all correct responses but also accepted some incorrect ones. As EM, BEM and the LLM-judge achieved the highest human agreement and lowest UE distortion among the evaluated criteria, we retain these three criteria for the sensitivity analysis experiment.

## 3.3. RQ2: Criterion Sensitivity of UE Evaluation

This experiment examines how correctness-criterion choice affects the measured performance and relative ranking of UE methods at scale. We evaluate the four UE methods under EM, BEM and the judge on 23,254 predictions. Table 1 reports criterion-conditioned error-detection AUROC, with equal weighting of the three evaluated models within each dataset. Criterion choice changes both the magnitude and ordering of measured UE performance. The largest spread among the dataset–method means occurs for LogTokU on MIMIC-IV: AUROC ranges from 0.440 under BEM to 0.586 under the judge, a difference of 0.146. On the same dataset, LogTokU exceeds Self-Certainty under the judge (0.586 versus 0.525), but falls below it under BEM (0.440 versus 0.512). These reversals show that the correctness definition can affect which UE method appears preferable. Higher criterion-conditioned AUROC does not establish closer agreement with human-referenced performance, because human labels are unavailable for the full set.

These results motivate criterion validation before selecting UE methods for clinical prediction tasks. The spread across criteria is distinct from distortion relative to a human reference. A larger fullset AUROC under one criterion could reflect better error identification or merely a different assignment of correctness labels. Likewise, an AUROC below 0.5 describes the score ordering under the stated error labels and score direction, rather than a criterion-independent property of the UE method.

![](images/2b8c855efe6fb0c684641dce628809d640bce185a7a9c4ffd241d02ffa11b688.jpg)  
Fig. 3. Observed false-acceptance rates by human-annotated error category. All groups containing at least one error of the relevant category are included, covering 24/167/30 errors for missing/partial, extra/wrong and invalid/uncertain responses, respectively. Rates are averaged equally across included groups. Error bars show 95% confidence intervals derived from bootstrap sampling for 10,000 times.

## 3.4. RQ3: Observed False-Acceptance Patterns

This experiment examines which human-identified clinical errors are incorrectly accepted as correct by each criterion. Figure 3 reports false-acceptance rates for each category, averaged equally across dataset–model groups containing at least one error of that type.

Despite its high overall Human Validation Consistency, the evaluated judge accepts 16/30 invalid/uncertain responses, corresponding to an equal-group mean rate of 60.3%, with no observed false acceptance for missing/partial or extra/wrong predictions. Microsoft NLI also shows its highest false-acceptance rate for invalid/uncertain responses, at 30.1%, whereas Potsawee NLI shows its highest rate for extra/wrong predictions, at 8.1%. The three ROUGE-L variants most often accept missing/partial errors, with a mean rate of 35.0% for this category. BEM shows false acceptance in all three categories, while EM rejects all human-identified errors in the audit. For clinical UE evaluation, an answer’s apparent plausibility does not by itself establish that it meets the task’s correctness requirements. Accepting invalid or uncertain outputs as correct introduces labels that conflict with human judgements and can either inflate or reduce the evaluated metrics. Human validation should examine incomplete, additional and uncertain answers alongside aggregate agreement and the fidelity of downstream UE estimates.

The concentration of false acceptance in invalid/uncertain responses suggests an error-specific acceptance bias in the evaluated LLM-judge. In these clinical prediction tasks, an answer must provide a determinate response within the candidate-label space. Uncertainty or invalidity cannot be resolved simply by assigning a plausible label. The observed pattern indicates that the judge’s acceptance decisions do not consistently preserve this distinction, despite its high aggregate agreement with human annotations.

## 4. CONCLUSION

This study shows that reliable evaluation of uncertainty in clinical prediction requires validation of the correctness criteria. Across the evaluated candidate-label tasks, EM achieved the highest observed human agreement and lowest UE distortion. On the full prediction set, criterion choice changed measured AUROC by up to 0.146 and reversed UE-method rankings. Error analysis further revealed the bias of acceptance of invalid or uncertain predictions by LLM-judge. Together, these findings show that correctness assessment is an integral component of clinical UE evaluation and should be validated against both human judgements and downstream UE performance.

## 5. COMPLIANCE WITH ETHICAL STANDARDS

This study retrospectively analysed de-identified data from MIMIC-IV, MC-MED and EHRSHOT, accessed through PhysioNet and Stanford University under the respective data use agreements. The original MIMIC-IV data sharing was approved by the Beth Israel Deaconess Medical Center Institutional Review Board with a waiver of informed consent. The original MC-MED study was approved by the Stanford University Institutional Review Board, also with a waiver of informed consent. The EHRSHOT study reported that IRB approval was not required because the data were de-identified. Our study involved no patient recruitment or clinical intervention. Human annotation was limited to assessing model-generated responses against predefined task requirements and reference labels.

## 6. ACKNOWLEDGEMENTS

The authors declare no conflicts of interest.

## 7. REFERENCES

[1] W. Lou, Y. Wu, P. Xu, W. Zhang, X. Chen, J. Yang, M. He, and D. Shi, “Key concept learning for medical vision language model with reasoning capabilities,” npj Digital Medicine, 2026.

[2] T. Chen, M. Zhu, Z. Luo, and T. Zhu, “Cross-representation benchmarking in time-series electronic health records for clinical outcome prediction,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 7076–7080.

[3] M. Zhu, Y. Liu, and T. Zhu, “Towards generation-efficient uncertainty estimation in large language models,” arXiv preprint arXiv:2605.06053, 2026.

[4] V. Adlakha, P. BehnamGhader, X. H. Lu, N. Meade, and S. Reddy, “Evaluating correctness and faithfulness of instruction-following models for question answering,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 681–699, 2024.

[5] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, “Detecting hallucinations in large language models using semantic entropy,” Nature, vol. 630, no. 8017, pp. 625–630, 2024.

[6] J. Bulian, C. Buck, W. Gajewski, B. Borschinger, and T. Schus-¨ ter, “Tomayto, tomahto. beyond token-level answer equivalence for question answering evaluation,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, Abu Dhabi, United Arab Emirates, 2022, pp. 291–305, Association for Computational Linguistics.

[7] E. Croxford, Y. Gao, E. First, N. Pellegrino, M. Schnier, J. Caskey, M. Oguss, G. Wills, G. Chen, D. Dligach, M. M. Churpek, A. Mayampurath, F. Liao, C. Goswami, K. K. Wong, B. W. Patterson, and M. Afshar, “Evaluating clinical AI summaries with large language models as judges,” npj Digital Medicine, vol. 8, no. 1, pp. 640, 2025.

[8] G. H. Chen, S. Chen, Z. Liu, F. Jiang, and B. Wang, “Humans or LLMs as the judge? a study on judgement bias,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Miami, Florida, USA, 2024, pp. 8301–8327, Association for Computational Linguistics.

[9] A. Santilli, A. Golinski, M. Kirchhof, F. Danieli, A. Blaas, M. Xiong, L. Zappella, and S. Williamson, “Revisiting uncertainty quantification evaluation in language models: Spurious interactions with response length bias results,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), Vienna, Austria, 2025, pp. 743–759, Association for Computational Linguistics.

[10] M. Ielanskyi, K. Schweighofer, L. Aichberger, and S. Hochreiter, “Addressing pitfalls in the evaluation of uncertainty estimation methods for natural language generation,” in International Conference on Learning Representations, 2026.

[11] M. Wornow, R. Thapa, E. Steinberg, J. A. Fries, and N. H. Shah, “EHRSHOT: An EHR benchmark for few-shot evaluation of foundation models,” in Advances in Neural Information Processing Systems. 2023, vol. 36, pp. 67125–67137, Curran Associates, Inc.

[12] A. Kansal, E. Chen, B. T. Jin, P. Rajpurkar, and D. A. Kim, “MC-MED, multimodal clinical monitoring in the emergency department,” Scientific Data, vol. 12, no. 1, pp. 1094, 2025.

[13] A. E. W. Johnson et al., “MIMIC-IV, a freely accessible electronic health record dataset,” Scientific Data, vol. 10, no. 1, pp. 1, 2023.

[14] M. Zhu, Y. Liu, Z. Luo, and T. Zhu, “The taxonomies, training, and applications of event stream modelling for electronic health records,” arXiv preprint arXiv:2603.14003, 2026.

[15] M. Zhu, Z. Luo, Y. Liu, and T. Zhu, “From token to token pair: Efficient prompt compression for large language models in clinical prediction,” in Forty-third International Conference on Machine Learning, 2026.

[16] Gemma Team, “Gemma 4 technical report,” 2026.

[17] A. Sellergren et al., “Medgemma 1.5 technical report,” 2026.

[18] C.-Y. Lin, “ROUGE: A package for automatic evaluation of summaries,” in Text Summarization Branches Out, Barcelona, Spain, 2004, pp. 74–81, Association for Computational Linguistics.

[19] P. He, X. Liu, J. Gao, and W. Chen, “DeBERTa: Decodingenhanced BERT with disentangled attention,” in International Conference on Learning Representations, 2021.

[20] P. Manakul, A. Liusie, and M. Gales, “SelfCheckGPT: Zeroresource black-box hallucination detection for generative large language models,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore, 2023, pp. 9004–9017, Association for Computational Linguistics.

[21] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judging LLM-as-a-judge with MT-Bench and chatbot arena,” in Advances in Neural Information Processing Systems. 2023, vol. 36, Curran Associates, Inc.

[22] Z. Kang, X. Zhao, and D. Song, “Scalable best-of-n selection for large language models via self-certainty,” in Advances in Neural Information Processing Systems, 2025.

[23] H. Ma, J. Chen, J. T. Zhou, G. Wang, and C. Zhang, “Estimating llm uncertainty with evidence,” arXiv preprint arXiv:2502.00290, 2025.

[24] L. Liu, R. Pourreza, S. Panchal, A. Bhattacharyya, Y. Jian, Y. Qin, and R. Memisevic, “Enhancing hallucination detection through noise injection,” in International Conference on Learning Representations, 2026.