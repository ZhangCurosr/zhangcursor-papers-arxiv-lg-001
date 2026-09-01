# Balancing Privacy, Utility, and Safety in LLM Alignment through Preference Optimization

Dishu Yang\*†

Jingjing (May) Liu†

Jize Li

Khoury College of Computer Sciences

Department of Electrical Engineering

Department of Economics

Northeastern University

and Computer Sciences

Boston University

San Jose, USA

University of California, Berkeley

Boston, USA

yang.dis@northeastern.edu

Berkeley, USA

jizel@bu.edu

\*Corresponding author

ORCID: 0009-0009-6936-4017

†Equal contribution as a first

author

Abstract— Preference optimization is widely used to align large language models with human preferences, but preference-data composition may also influence privacy-relevant memorization. We examine whether adding synthetic privacy-preference pairs to Direct Preference Optimization (DPO) is associated with lower canary-based memorization signals without modifying the objective or introducing a formal privacy mechanism. We propose Privacy-Pressure Preference Mixing (P3M), a data-composition protocol that varies the amount of privacy-preference data while keeping helpfulness and harmlessness preference data fixed. We evaluate a non-privacy Baseline and privacy-mixing ratios of 0.5, 1.0, and 2.0 using Gemma 3 270M-IT across five random seeds and validate the same four conditions using 4-bit-quantized Gemma 2 2B-IT across three seeds. Overall, under the tested conditions, privacy-preference mixing is associated with lower mean canary suffix log-likelihood proxy values across both model settings and lower aggregate membership-inference attack performance relative to the Baseline in the mixed-source 2B evaluation. Specifically, across the privacy-aware 2B configurations, the mean area under the receiver operating characteristic curve (AUROC) ranges from 0.596 to 0.629, and the mean area under the precisionrecall curve (AUPRC) ranges from 0.541 to 0.575, compared with 0.804 and 0.790, respectively, for the Baseline. However, the reduction in membership distinguishability does not hold uniformly across data sources. Moreover, the relationship between the privacy ratio and harmlessness preference accuracy varies by model setting, whereas helpfulness preference accuracy remains broadly stable. These findings suggest that P3M should be viewed as a lightweight empirical protocol for examining privacy-utility-safety trade-offs rather than as a formal privacy guarantee or a defense against extraction attacks.

Keywords— DPO, privacy-aware LLM alignment, preference mixing, LoRA, trustworthy AI, privacy-utility-safety trade-off

## I. INTRODUCTION

Preference-based alignment methods train language models using human judgments over alternative outputs. In Reinforcement Learning from Human Feedback (RLHF), models are fine-tuned from human demonstrations or rankings to better follow instructions and align with human intent [1,2,3,4]. Direct Preference Optimization (DPO) bypasses the reward-modeling step by directly optimizing the language model on preference data [5]. However, alignment data may contain prompts, responses, preference labels, and user-derived context. Such data can be vulnerable to membership-inference attacks [6], and memorization may surface or persist across RLHF and direct preference-learning pipelines, raising privacy concerns when sensitive examples are later regurgitated [7].

Recent privacy-preserving alignment work addresses related risks through differential privacy for RLHF-style alignment [8], privacy-preserving synthetic instructions [9], federated RLHF [10], and differentially private synthetic preference data [12]. These approaches rely on formal privacy mechanisms, decentralized training, or additional synthesis pipelines. This paper studies a lighter and more practically accessible question: when the DPO objective, LoRA setup, base model, and nonprivacy data are fixed within each experiment, how do helpfulness, harmlessness, and memorization-related signals change as the amount of privacy-oriented preference data varies? We further evaluate the same data-composition conditions in a larger-model setting and conduct a complementary membership-inference analysis.

We propose Privacy-Pressure Preference Mixing (P3M), a data-composition protocol that treats privacy pressure as an experimental variable rather than as a new objective. P3M adds synthetic privacy-preference pairs to a standard helpfulnessand-harmlessness preference mixture. In each privacy pair, the chosen response does not disclose a synthetic private identifier, whereas the rejected response reveals it. Across configurations, we vary privacy pressure by changing the amount of privacypreference data in the DPO training pool while keeping the helpfulness and harmlessness data fixed.

Our controlled, resource-bounded evaluation compares a non-privacy Baseline with three privacy-mixing ratios (0.5, 1.0, and 2.0) using Gemma 3 270M-IT across five random seeds and 4-bit Gemma 2 2B-IT across three seeds. Both experiments use LoRA-based DPO and evaluate helpfulness preference accuracy, harmlessness preference accuracy, and a canary suffix log-likelihood proxy following the broader canary-based memorization methodology [13]. For the 2B validation, we additionally adapt PREMIA, a reference-based membershipinference framework for preference data [6]. We report attack performance using the area under the receiver operating characteristic curve (AUROC) and the area under the precisionrecall curve (AUPRC). Together, the canary-based proxy and PREMIA-based evaluation provide complementary empirical signals of memorization and membership distinguishability, but they do not provide a formal privacy guarantee or demonstrate resistance to extraction attacks.

Across both model settings, all privacy-aware configurations yield lower mean canary suffix log-likelihood proxy values than the corresponding Baselines. In the 2B validation, they also yield lower aggregate AUROC and AUPRC on the constructed mixed-source attack sets, although source-stratified results do not show uniform reductions across helpfulness, harmlessness, and privacy-preference tuples. Helpfulness preference accuracy remains broadly stable, whereas harmlessness preference accuracy decreases monotonically with the privacy ratio in the 270M experiment but not in the 2B validation. These results highlight the importance of jointly evaluating privacy, utility, and safety outcomes.

This paper makes three contributions. First, it introduces Privacy-Pressure Preference Mixing (P3M), a lightweight protocol for varying privacy-oriented preference data within DPO. Second, it evaluates the same four data-composition conditions using Gemma 3 270M-IT across five random seeds and 4-bit Gemma 2 2B-IT across three random seeds, jointly measuring helpfulness preference accuracy, harmlessness preference accuracy, and the canary suffix log-likelihood proxy. Third, it complements the canary-based evaluation with a PREMIA-based membership-inference evaluation on the 2B model and reports attack performance at both the aggregate mixed-source and source-stratified levels, showing that the reduction in aggregate attack performance observed for the privacy-aware configurations relative to the Baseline does not hold uniformly across helpfulness, harmlessness, and privacypreference tuples.

## II. RELATED WORK

## A. Preference Optimization for LLM Alignment

Preference-based learning uses human comparisons to learn reward functions [1] and has subsequently been applied to summarization [2], instruction following [3], and helpful-andharmless assistant training [4]. The helpful-and-harmless setting motivates the utility-safety framing adopted in this study, but privacy exposure is not a primary training or evaluation target in these alignment applications. Conventional RLHF typically involves separate reward-model training and policy optimization, whereas DPO directly optimizes the policy from preference pairs without fitting a separate reward model [5]. This simpler training structure makes DPO suitable for controlled studies of preference-data composition. Building on DPO, P3M keeps the optimization objective unchanged and examines whether adding privacy-preference pairs affects memorization-related signals as well as helpfulness and harmlessness outcomes.

## B. Memorization and Privacy Leakage in Language Models

Language models may memorize rare, unique, or duplicated training sequences, creating a risk of data exposure. Canarybased testing inserts synthetic sequences and quantifies their exposure [13], while extraction attacks show that models can reproduce verbatim training examples, including personally identifiable information [15]. Memorization increases with model capacity, example duplication, and the amount of prompt context [16]. These concerns extend to alignment: preference data can be vulnerable to membership-inference attacks in DPOand PPO-aligned models [6], while memorization can surface and persist through RLHF and direct preference-learning pipelines [7]. Accordingly, we use a canary suffix log-likelihood proxy in both model settings and a PREMIA-based membership-inference evaluation on the 2B model as complementary empirical diagnostics.

## C. Privacy-Aware Alignment and Safety Trade-offs

Privacy, utility, and safety do not necessarily improve together under preference alignment. Safe RLHF separates reward and safety objectives using distinct reward and cost models [17], while broader evaluations find uneven effects of preference alignment across trustworthiness dimensions, including privacy [18]. Privacy-aware alignment has explored the application of differential privacy to reinforcement-learningbased alignment [8], privacy-preserving synthetic instructions [9], federated RLHF [10], user-level private RLHF [11], and differentially private preference-data synthesis [12]. P3M complements these approaches by retaining DPO and treating privacy-preference mixing as a data-composition intervention evaluated across three dimensions: memorization-related signals, helpfulness, and harmlessness.

## D. Lightweight Alignment under Resource Constraints

LoRA reduces the number of trainable parameters by freezing the base model and optimizing low-rank adapters [14], while QLoRA further reduces memory usage by training LoRA adapters over a frozen 4-bit-quantized base model [19]. These methods make repeated controlled comparisons more feasible under limited compute. LoRA is used in both experiments, with 4-bit quantization applied only in the 2B validation. Both serve as computational tools rather than privacy mechanisms. The privacy intervention is the P3M data-composition protocol.

## III. METHODOLOGY

## A. Overview of the Framework

The proposed P3M framework treats privacy pressure as a configurable data-composition variable in DPO. Within each model experiment, the base model, DPO objective, LoRA configuration, and helpfulness and harmlessness pools remain fixed, while the privacy-preference data varies across mixing ratios. The resulting adapters are evaluated using helpfulness and harmlessness preference accuracy and a canary suffix loglikelihood proxy. Section III-F extends the evaluation with a larger-model validation and PREMIA-based membership inference. P3M changes data composition rather than the optimization objective and does not provide a formal privacy mechanism.

## B. Preference Data Construction

The fixed non-privacy data comprise helpfulness and harmlessness preference pairs from Anthropic HH-RLHF, representing utility and safety, respectively. Each example follows the standard preference-pair format used in DPO training:

$$
\{ p r o m p t , c h o s e n r e s p o n s e , r e j e c t e d r e s p o n s e \}\tag{1}
$$

## C. Synthetic Privacy-Preference Data and Canary Design

To examine whether adding privacy-preference data is associated with lower values of the canary suffix log-likelihood proxy, we construct synthetic preference pairs around artificial identifiers: the chosen response does not disclose the identifier, whereas the rejected response discloses it. The base privacy pool comprises 100 unique identifiers assigned to low-, medium-, and high-frequency tiers with 1 $( n = 3 3 ) , 5 ( n = 3 3 )$ , and $1 0 \ : ( n =$ 34) nominal occurrences, respectively, yielding 538 rows. The sampling pool at privacy ratio 0.5 contains 269 rows drawn from the base pool and covers 83 identifiers, whereas the pools at privacy ratios 1.0 and 2.0 contain all 100 identifiers. None of these identifiers appears in the Baseline training pool. The evaluation covers all 100 base-pool identifiers rather than a separate held-out set. Because each adapter is trained for only 100 update steps using examples sampled from its pool, inclusion in the nominal pool does not guarantee that a given identifier appears in the realized update stream. This design provides a controlled memorization-related signal without using real personal information.

## D. Privacy Mixing Strategy and Training Setup

Privacy pressure is introduced through dataset composition. Let $D _ { h }$ denote the helpfulness preference set, $D _ { s }$ denote the harmlessness preference set, and $D _ { p }$ denote the synthetic privacy-preference set. The main privacy-aware training set is constructed as:

$$
D ( r ) = D _ { h } \uplus D _ { s } \uplus M \big ( D _ { p } , r \big )\tag{2}
$$

where $M \left( D _ { p } , r \right)$ samples half of, includes all of, or duplicates the privacy-preference rows for privacy mixing ratio $r \in \{ 0 . 5 , 1 . 0 , \bar { 2 } . 0 \}$ , respectively. Thus, privacy pressure is implemented through data frequency rather than per-example weighting. Across conditions, we hold the base model, optimizer, DPO objective, and LoRA configuration fixed and train each adapter for 100 optimization steps. Only the amount of privacy-preference data is varied, and the nominal samplingpool size changes accordingly.

## E. Evaluation Protocol

We evaluate the trained models across three dimensions: utility, safety, and privacy.

Utility and safety are measured as preference accuracy on held-out helpfulness and harmlessness sets, respectively. For each preference pair, we compute the length-normalized conditional log probability of the chosen and rejected completions given the same prompt. A prediction is correct when the chosen completion receives the higher score.

Privacy is empirically evaluated using the canary suffix loglikelihood proxy, defined for each canary as the average conditional log-probability per token of its hidden suffix given a prompt containing its visible prefix. Scores are first averaged within the low-, medium-, and high-frequency tiers and then averaged equally across the three tiers. Lower values indicate a weaker measured memorization-related signal. This modeldependent proxy is an empirical diagnostic, not a formal privacy guarantee or extraction test. Results are reported as the mean ± sample standard deviation across five seeds for the primary experiment and three seeds for the 2B validation.

## F. Larger-Model and Membership-Inference Validation

For validation, we repeat the four-condition comparison using 4-bit-quantized Gemma 2 2B-IT across three seeds. Table 2 summarizes the LoRA-DPO settings. Reference logprobabilities are precomputed using the frozen pre-DPO model. Because the two model settings differ in model generation and precision, the 2B experiment is a larger-model validation rather than a controlled scaling study.

Following PREMIA’s target/reference probability-ratio construction [6], we use a length-normalized log-domain adaptation. For a preference tuple $\boldsymbol { z } = ( x , y _ { w } , y _ { l } )$ , where � denotes the prompt and $y _ { w }$ and $y _ { l }$ denote the chosen and rejected responses, respectively, the tuple-level attack score is defined as follows:

$$
\begin{array} { r } { s _ { \mathrm { p a i r } } ( z ) = \left[ \bar { \ell } _ { \theta } ( y _ { w } | x ) - \bar { \ell } _ { \mathrm { r e f } } ( y _ { w } | x ) \right] } \\ { - \left[ \bar { \ell } _ { \theta } ( y _ { l } | x ) - \bar { \ell } _ { \mathrm { r e f } } ( y _ { l } | x ) \right] } \end{array}\tag{3}
$$

where ℓ<sup>F</sup> denotes mean response-token log-probability, � denotes the DPO-aligned model, and ref denotes the frozen pre-DPO model. Higher scores indicate stronger evidence that the complete preference tuple was used during alignment.

A unique tuple is labeled as a member if it is sampled at least once during the 100 DPO optimization steps of a training run. Tuples available in the corresponding training pool but not sampled during these steps serve as non-member candidates. Each attack set contains equal numbers of member and nonmember tuples, with the source composition of the two groups matched where possible. Membership therefore refers to the realized update stream rather than a fixed train-test split. We report tuple-level AUROC and AUPRC. On these balanced attack sets, values closer to 0.5 indicate weaker membership distinguishability. This evaluation is an empirical privacy audit rather than a formal privacy guarantee.

## IV. EXPERIMENTS AND RESULTS

We compare the Baseline with privacy ratios 0.5, 1.0, and 2.0 on Gemma 3 270M-IT across five random seeds and on 4- bit Gemma 2 2B-IT across three random seeds. Both experiments evaluate helpfulness preference accuracy, harmlessness preference accuracy, and the canary suffix loglikelihood proxy. The 2B validation additionally reports PREMIA-based membership-inference AUROC and AUPRC. The canary suffix log-likelihood proxy and PREMIA metrics are interpreted as empirical privacy-risk signals rather than formal privacy guarantees.

## A. Experimental Setup

We evaluate the four configurations on Gemma 3 270M-IT across five random seeds and 4-bit-quantized Gemma 2 2B-IT across three random seeds. Table 1 summarizes the nominal training-pool composition, and Table 2 reports the model, training, and evaluation settings. For each trained 2B adapter, we construct the balanced attack sets following Section III-F. They contain 83 to 100 member tuples and an equal number of non-member tuples per run, with source composition matched where possible.

TABLE 1. TRAINING DATASET COMPOSITION
<table><tr><td>Run</td><td>Helpfulness examples</td><td>Harmlessness examples</td><td>Privacy- preference examples</td><td>Total examples</td></tr><tr><td>Baseline</td><td>500</td><td>500</td><td>0</td><td>1000</td></tr><tr><td>Privacy ratio 0.5</td><td>500</td><td>500</td><td>269</td><td>1269</td></tr><tr><td>Privacy ratio 1.0</td><td>500</td><td>500</td><td>538</td><td>1538</td></tr><tr><td>Privacy ratio 2.0</td><td>500</td><td>500</td><td>1076</td><td>2076</td></tr></table>

TABLE 2. HYPERPARAMETERS AND REPRODUCIBILITY SETTINGS FOR THE PRIMARY AND LARGER-MODEL EXPERIMENTS
<table><tr><td rowspan=2 colspan=1>Settings</td><td rowspan=1 colspan=2>Experimental Configuration</td></tr><tr><td rowspan=1 colspan=1>270M Primary</td><td rowspan=1 colspan=1>2B Validation</td></tr><tr><td rowspan=1 colspan=1>Base model</td><td rowspan=1 colspan=1>Gemma 3 270M-IT</td><td rowspan=1 colspan=1>Gemma 2 2B-IT</td></tr><tr><td rowspan=1 colspan=1>Random seeds</td><td rowspan=1 colspan=1>Five</td><td rowspan=1 colspan=1>Three</td></tr><tr><td rowspan=1 colspan=1>Base-modelweights</td><td rowspan=1 colspan=1>Frozen</td><td rowspan=1 colspan=1>Frozen, MLX 4-bit</td></tr><tr><td rowspan=1 colspan=1>LoRAconfiguration</td><td rowspan=1 colspan=1>Rank 8; scale 10.0;dropout 0.0; fourlayers</td><td rowspan=1 colspan=1>Rank 8; scale 10.0;dropout 0.0; fourlayers</td></tr><tr><td rowspan=1 colspan=1>Optimizerand DPO</td><td rowspan=1 colspan=1>Adam optimizer;learning rate $5 \times 1 0 ^ { - 6 } ;$ DPO beta 0.1</td><td rowspan=1 colspan=1>Adam optimizer;learning rate $5 \times 1 0 ^ { - 6 } ;$ DPO beta 0.1</td></tr><tr><td rowspan=1 colspan=1>Training schedule</td><td rowspan=1 colspan=1>100 optimization steps;maximum length 384</td><td rowspan=1 colspan=1>100 optimization steps;maximum length 384</td></tr><tr><td rowspan=1 colspan=1>Evaluation setsizes</td><td rowspan=1 colspan=1>150 helpful; 150harmless; 100 canaries</td><td rowspan=1 colspan=1>150 helpful; 150harmless; 100 canaries</td></tr></table>

## B. Evaluation Metrics

Under the evaluation protocol in Section III-E, higher preference accuracy indicates better performance on the corresponding helpfulness or harmlessness set, whereas lower canary suffix log-likelihood proxy values indicate a weaker measured memorization-related signal. For the 2B validation, tuple-level PREMIA attack performance is reported using AUROC and AUPRC, as described in Section III-F. On the balanced attack sets, values closer to 0.5 indicate weaker membership distinguishability. Results are reported as the mean ± sample standard deviation across five seeds for the 270M experiment and three seeds for the 2B validation, including the PREMIA-based metrics.

## C. Primary Gemma 3 270M-IT Results

Table 3 reports the primary results for Gemma 3 270M-IT across the non-privacy Baseline and the three privacy-aware configurations.

TABLE 3. FIVE-SEED GEMMA 3 270M-IT RESULTS (MEAN ± STANDARD DEVIATION)
<table><tr><td>Run</td><td>Helpfulness Pref. Acc.</td><td>Harmlessness Pref. Acc.</td><td>Canary Suffix Log-likelihood Proxy</td></tr><tr><td>Baseline</td><td> $\overline { { 0 . 5 7 9 \pm 0 . 0 0 9 } }$ </td><td> $\overline { { 0 . 4 6 1 \pm 0 . 0 2 0 } }$ </td><td> $- 7 . 9 5 2 \pm 1 . 1 6 4$ </td></tr><tr><td>Privacy ratio 0.5</td><td> $0 . 5 8 0 \pm 0 . 0 0 5$ </td><td> $0 . 4 4 1 \pm 0 . 0 1 7$ </td><td> $- 9 . 4 8 7 \pm 1 . 1 6 4$ </td></tr><tr><td>Privacy ratio 1.0</td><td> $0 . 5 8 0 \pm 0 . 0 1 2$ </td><td> $0 . 4 3 6 \pm 0 . 0 0 8$ </td><td> $- 9 . 7 0 6 \pm 1 . 2 5 5$ </td></tr><tr><td>Privacy ratio 2.0</td><td> $0 . 5 8 8 \pm 0 . 0 0 6$ </td><td> $0 . 4 2 7 \pm 0 . 0 0 7$ </td><td> $- 1 0 . 6 4 4 \pm 1 . 1 6 2$ </td></tr></table>

All privacy-aware configurations have lower mean canary suffix log-likelihood proxy values than the Baseline (-7.952), with the lowest value observed at ratio 2.0 (-10.644). Mean helpfulness preference accuracy remains stable across configurations (0.579 to 0.588), whereas mean harmlessness preference accuracy decreases monotonically from 0.461 to 0.427. Because all harmlessness accuracies are below 0.5, these results represent relative changes under matched settings rather than evidence of strong absolute harmlessness performance. Thus, in the primary experiment, higher privacy ratios are associated with lower canary proxy values and lower harmlessness accuracy, while helpfulness accuracy remains broadly stable.

## D. Gemma 2 2B-IT Validation and Membership-Inference Results

Table 4 reports the 2B validation results across three seeds. Because the canary suffix log-likelihood proxy is modeldependent, its absolute magnitude is interpreted within this model setting rather than compared directly with the 270M results.

TABLE 4. GEMMA 2 2B-IT VALIDATION RESULTS (THREE RANDOM SEEDS)  
(A) PREFERENCE ACCURACY AND CANARY SUFFIX LOG-LIKELIHOOD PROXY
<table><tr><td>Run</td><td>Helpfulness Pref. Acc.</td><td>Harmlessness Pref. Acc.</td><td>Canary Suffix Log-likelihood Proxy</td></tr><tr><td>Baseline</td><td> $0 . 6 1 1 \pm 0 . 0 2 1$ </td><td> $0 . 4 4 4 \pm 0 . 0 0 8$ </td><td> $- 7 . 3 4 8 \pm 1 . 2 8 9$ </td></tr><tr><td>Privacy ratio 0.5</td><td> $0 . 6 1 6 \pm 0 . 0 0 4$ </td><td> $0 . 4 3 1 \pm 0 . 0 1 7$ </td><td> $- 1 1 . 7 6 4 \pm 0 . 0 8 4$ </td></tr><tr><td>Privacy ratio 1.0</td><td> $0 . 6 0 9 \pm 0 . 0 1 7$ </td><td> $0 . 4 3 1 \pm 0 . 0 0 8$ </td><td> $- 1 2 . 6 1 6 \pm 1 . 0 0 9$ </td></tr><tr><td>Privacy ratio 2.0</td><td> $0 . 5 9 8 \pm 0 . 0 1 7$ </td><td> $0 . 4 8 9 \pm 0 . 0 7 3$ </td><td> $- 3 1 . 9 0 5 \pm 4 . 8 6 9$ </td></tr></table>

(B) PREMIA-BASED MEMBERSHIP-INFERENCE RESULTS
<table><tr><td>Run</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>Baseline</td><td> $\overline { { 0 . 8 0 4 \pm 0 . 0 6 1 } }$ </td><td> $\overline { { 0 . 7 9 0 \pm 0 . 0 5 5 } }$ </td></tr><tr><td>Privacy ratio 0.5</td><td> $0 . 6 0 8 \pm 0 . 0 1 7$ </td><td> $0 . 5 6 4 \pm 0 . 0 1 4$ </td></tr><tr><td>Privacy ratio 1.0</td><td> $0 . 6 2 9 \pm 0 . 0 2 5$ </td><td> $0 . 5 7 5 \pm 0 . 0 2 1$ </td></tr><tr><td>Privacy ratio 2.0</td><td> $0 . 5 9 6 \pm 0 . 0 5 9$ </td><td> $0 . 5 4 1 \pm 0 . 0 4 5$ </td></tr></table>

Table 4 shows that all privacy-aware configurations yield lower mean canary suffix log-likelihood proxy values than the Baseline (-7.348). The privacy ratio 2.0 configuration yields the lowest mean value (-31.905), although it also exhibits the greatest variability across seeds. Helpfulness preference accuracy remains broadly stable across configurations, ranging from 0.598 to 0.616. Mean harmlessness preference accuracy is modestly lower at privacy ratios 0.5 and 1.0 than in the Baseline but increases to $\bar { 0 . 4 8 9 } \dot { \pm } 0 . 0 7 3$ at ratio 2.0. Thus, unlike the 270M experiment, the 2B validation does not exhibit a monotonic decline in harmlessness preference accuracy as the privacy ratio increases.

On the mixed-source attack sets, the privacy-aware configurations yield lower aggregate AUROC (0.596 to 0.629) and AUPRC (0.541 to 0.575) than the Baseline (0.804 and 0.790, respectively), indicating weaker aggregate membership distinguishability. However, neither metric varies monotonically with the privacy ratio, all privacy-aware values remain above 0.5, and the aggregate pattern does not imply uniform reductions across data sources. Table 5 therefore reports the source-stratified results.

TABLE 5. SOURCE-STRATIFIED PREMIA-BASED MEMBERSHIP-INFERENCE RESULTS FOR GEMMA 2 2B-IT. VALUES ARE REPORTED AS THE MEAN ± STANDARD DEVIATION ACROSS THREE RANDOM SEEDS  
(A) AUROC
<table><tr><td rowspan=1 colspan=1>Configuration</td><td rowspan=1 colspan=3>Tuple source</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Helpfulness</td><td rowspan=1 colspan=1>Harmlessness</td><td rowspan=1 colspan=1>Privacypreference</td></tr><tr><td rowspan=1 colspan=1>Baseline</td><td rowspan=1 colspan=1>0.815 ± 0.120</td><td rowspan=1 colspan=1>0.795 ± 0.032</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Privacy ratio0.5</td><td rowspan=1 colspan=1>0.659 ± 0.063</td><td rowspan=1 colspan=1>0.681 ± 0.028</td><td rowspan=1 colspan=1>0.469 ± 0.014</td></tr><tr><td rowspan=1 colspan=1>Privacy ratio1.0</td><td rowspan=1 colspan=1> $0 . 7 6 7 \pm 0 . 0 2 4$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 7 3 7 \pm 0 . 0 4 5 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 5 3 9 \pm 0 . 1 0 6 } }$ </td></tr><tr><td rowspan=1 colspan=1>Privacy ratio2.0</td><td rowspan=1 colspan=1> $\overline { { 0 . 8 0 6 \pm 0 . 0 8 1 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 8 1 3 \pm 0 . 0 3 6 } }$ </td><td rowspan=1 colspan=1> $\overline { { 0 . 4 9 3 \pm 0 . 1 3 9 } }$ </td></tr></table>

(B) AUPRC
<table><tr><td rowspan=1 colspan=1>Configuration</td><td rowspan=1 colspan=3>Tuple source</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Helpfulness</td><td rowspan=1 colspan=1>Harmlessness</td><td rowspan=1 colspan=1>Privacypreference</td></tr><tr><td rowspan=1 colspan=1>Baseline</td><td rowspan=1 colspan=1> $\overline { { 0 . 8 2 0 \pm 0 . 1 0 3 } }$ </td><td rowspan=1 colspan=1>0.776 ± 0.053</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Privacy ratio0.5</td><td rowspan=1 colspan=1> $\overline { { 0 . 6 2 7 \pm 0 . 0 5 6 } }$ </td><td rowspan=1 colspan=1>0.707 ± 0.032</td><td rowspan=1 colspan=1>0.508 ± 0.016</td></tr><tr><td rowspan=1 colspan=1>Privacy ratio1.0</td><td rowspan=1 colspan=1> $0 . 7 6 9 \pm 0 . 0 4 3$ </td><td rowspan=1 colspan=1>0.714 ± 0.067</td><td rowspan=1 colspan=1>0.560 ± 0.077</td></tr><tr><td rowspan=1 colspan=1>Privacy ratio2.0</td><td rowspan=1 colspan=1>0.817 ± 0.076</td><td rowspan=1 colspan=1>0.830 ± 0.024</td><td rowspan=1 colspan=1> $\overline { { 0 . 5 0 4 \pm 0 . 0 7 0 } }$ </td></tr></table>

Each source-specific subset contains equal numbers of members and non-members, ranging from 17 to 55 per class across configurations and seeds. Because some subsets are small and the validation uses only three seeds, these results should be treated as descriptive source-level diagnostics. The Baseline contains no privacy-preference tuples and therefore has no corresponding source-specific result.

Ratios 0.5 and 1.0 generally have lower helpfulness and harmlessness AUROC/AUPRC values than the Baseline, whereas ratio 2.0 is comparable to or higher than the Baseline. Privacy-preference results remain near chance across the three privacy-aware configurations. Thus, the lower aggregate attack performance does not hold uniformly across sources and may partly reflect mixed-source attack-set composition. The aggregate results should therefore be interpreted as mixturelevel membership distinguishability rather than a uniform source-level reduction.

Overall, the 2B validation likewise shows lower mean canary proxy values than the Baseline for all privacy-aware configurations and broadly stable helpfulness accuracy. However, it does not reproduce the monotonic decline in harmlessness accuracy observed in the primary experiment, and the source-stratified results do not show a uniform reduction in membership distinguishability.

## V. DISCUSSION

Across both model settings, privacy-preference mixing is associated with lower mean canary suffix log-likelihood proxy values. Helpfulness remains broadly stable, whereas harmlessness trends vary by model setting and the 2B membership-inference results vary by data source. These findings position P3M as a data-composition protocol for examining privacy-utility-safety trade-offs rather than as a formal privacy defense or a uniformly beneficial intervention across metrics and data sources.

## A. Interpreting the Privacy Mixing Ratio

No privacy ratio is uniformly optimal. Higher ratios correspond to lower mean canary proxy values in both models, but PREMIA attack performance in the 2B validation is nonmonotonic across ratios and does not decrease uniformly across sources, while the monotonic decline in harmlessness preference accuracy observed in the 270M experiment does not recur in the 2B validation. Configurations should therefore be evaluated jointly across privacy, helpfulness, and harmlessness outcomes rather than by privacy-preference data volume alone.

With 100 optimization steps per adapter, higher ratios shift the expected update composition toward privacy-preference examples. Differences in Gemma generation, model capacity, quantization, and optimization dynamics may also contribute. Because the present experiments do not isolate these factors, this explanation remains a hypothesis.

## B. Scope and Limitations

This resource-bounded study evaluates the 270M and 2B models across five and three seeds, respectively, using LoRA-DPO with four adapted transformer layers. Because the settings differ in model generation, parameter scale, and quantization, the 2B experiment should be interpreted as a larger-model validation rather than a controlled scaling study. Future work should assess generalization to other model families and scales as computational resources permit.

The privacy ratio changes both data composition and nominal pool size because privacy-preference rows are added or repeated rather than substituted for non-privacy rows. Although all configurations use 100 optimization steps, the inclusion of an example in the nominal pool does not guarantee that it is sampled during training. Thus, the observed metric differences may reflect not only privacy-preference data composition but also pool size, duplicated privacy rows, and variation in the examples sampled across runs. A fixed-total-size control that replaces some non-privacy examples with privacy-preference examples would help isolate composition effects from changes in pool size.

The canary proxy and PREMIA-based evaluation provide complementary empirical diagnostics of memorization-related signals and membership distinguishability. They do not provide formal privacy guarantees or establish resistance to extraction attacks. In our PREMIA-based evaluation, membership is defined with respect to the realized update stream rather than a fixed train-test partition. Because the balanced attack sets vary across runs in size and source composition, aggregate comparisons may be sensitive to attack-set composition. Future work should evaluate fixed member and non-member sets across runs, additional membership-inference protocols, and generation-based extraction attacks.

Synthetic canaries avoid the need for real personal data but do not capture leakage involving realistic identifiers, contextual clues, or combinations of partial information. More representative privacy-sensitive formats and extraction prompts should therefore be evaluated under appropriate safeguards.

## VI. CONCLUSION

This paper introduces P3M, a DPO data-composition protocol that varies the amount of synthetic privacy-preference data while keeping the helpfulness and harmlessness pools fixed within each model experiment. Under the tested conditions, all privacy-aware configurations show lower mean canary suffix log-likelihood proxy values than the corresponding Baseline within each model setting. In the PREMIA-based evaluation, the privacy-aware 2B configurations also yield lower aggregate AUROC and AUPRC values than the Baseline on the constructed mixed-source attack sets, although the sourcestratified results do not show a uniform reduction in membership distinguishability across data sources. Helpfulness preference accuracy remains broadly stable, whereas harmlessness trends differ between the two model settings. In similar resourcebounded settings, privacy-mixing configurations should therefore be evaluated jointly across privacy, utility, and safety outcomes rather than selected on the basis of any single objective. P3M should be viewed as a lightweight empirical protocol for examining privacy-utility-safety trade-offs rather than as a formal privacy guarantee or evidence of resistance to extraction attacks. Future work should evaluate fixed-total-size controls, additional model families, more realistic privacysensitive formats, and broader membership-inference and extraction evaluations.

## REFERENCES

[1] P. F. Christiano, J. Leike, T. Brown, M. Martic, S. Legg, and D. Amodei, “Deep reinforcement learning from human preferences,” Advances in neural information processing systems, vol. 30, 2017.

[2] N. Stiennon et al., “Learning to summarize with human feedback,” Advances in neural information processing systems, vol. 33, pp. 3008– 3021, 2020.

[3] L. Ouyang et al., “Training language models to follow instructions with human feedback,” Advances in neural information processing systems, vol. 35, pp. 27730–27744, 2022.

[4] Y. Bai et al., “Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback,” Apr. 12, 2022, arXiv: arXiv:2204.05862. doi: 10.48550/arXiv.2204.05862.

[5] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in neural information processing systems, vol. 36, pp. 53728–53741, 2023.

[6] Q. Feng, S. R. Kasa, S. K. Kasa, H. Yun, C. H. Teo, and S. B. Bodapati, “Exposing Privacy Gaps: Membership Inference Attack on Preference Data for LLM Alignment,” Apr. 27, 2025, arXiv: arXiv:2407.06443. doi: 10.48550/arXiv.2407.06443.

[7] A. Pappu, B. Porter, I. Shumailov, and J. Hayes, “Measuring memorization in RLHF for code completion,” arXiv.org, Jun. 17, 2024. https://arxiv.org/abs/2406.11715

[8] F. Wu, H. A. Inan, A. Backurs, V. Chandrasekaran, J. Kulkarni, and R. Sim, “Privately Aligning Language Models with Reinforcement Learning,” May 03, 2024, arXiv: arXiv:2310.16960. doi: 10.48550/arXiv.2310.16960.

[9] D. Yu, P. Kairouz, S. Oh, and Z. Xu, “Privacy-Preserving Instructions for Aligning Large Language Models,” Jul. 02, 2024, arXiv: arXiv:2402.13659. doi: 10.48550/arXiv.2402.13659.

[10] F. X. Fan, C. Tan, Y.-S. Ong, R. Wattenhofer, and W.-T. Ooi, “FedRLHF: A Convergence-Guaranteed Federated Framework for Privacy-Preserving and Personalized RLHF,” Feb. 08, 2025, arXiv: arXiv:2412.15538. doi: 10.48550/arXiv.2412.15538.

[11] J. Zhang et al., “Towards User-level Private Reinforcement Learning with Human Feedback,” Feb. 22, 2025, arXiv: arXiv:2502.17515. doi: 10.48550/arXiv.2502.17515.

[12] F. Gao and J. Yang, “Differentially Private Preference Data Synthesis for Large Language Model Alignment,” May 29, 2026, arXiv: arXiv:2605.30808. doi: 10.48550/arXiv.2605.30808.

[13] N. Carlini, C. Liu, Ú. Erlingsson, J. Kos, and D. Song, “The secret sharer: Evaluating and testing unintended memorization in neural networks,” in 28th USENIX security symposium (USENIX security 19), 2019, pp. 267– 284.

[14] E. J. Hu et al., “Lora: Low-rank adaptation of large language models.,” Iclr, vol. 1, no. 2, p. 3, 2022.

[15] N. Carlini et al., "Extracting training data from large language models," in Proc. USENIX Security Symposium, 2021. doi: https://doi.org/10.48550/arXiv.2012.07805

[16] N. Carlini, D. Ippolito, M. Jagielski, K. Lee, F. Tramer, and C. Zhang, “Quantifying memorization across neural language models,” arXiv.org, Feb. 15, 2022. https://arxiv.org/abs/2202.07646

[17] J. Dai et al., “Safe rlhf: Safe reinforcement learning from human feedback,” in International Conference on Learning Representations, 2024

[18] A. J. Li, S. Krishna, and H. Lakkaraju, “More rlhf, more trust? on the impact of preference alignment on trustworthiness,” in International Conference on Learning Representations, 2025

[19] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “Qlora: Efficient finetuning of quantized llms,” Advances in neural information processing systems, vol. 36, pp. 10088–10115, 2023.