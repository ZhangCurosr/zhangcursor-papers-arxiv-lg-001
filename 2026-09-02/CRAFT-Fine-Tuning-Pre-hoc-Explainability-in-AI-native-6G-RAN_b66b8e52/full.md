# CRAFT: Fine-Tuning Pre-hoc Explainability in AI-native 6G RAN

1<sup>st</sup> Pranshav Gajjar

NextG Wireless Lab

NCSU

Raleigh USA

2<sup>nd</sup> Vijay K Shah

NextG Wireless Lab

NCSU

Raleigh USA

Abstract—The next generation of mobile networks is envisioned as fully AI-native, with AI-RAN architectures embedding small language models (SLMs) to perform reasoning over realtime telemetry. The state-of-the-art training paradigms for telecom LLMs, exemplified by RANSTRUCT-style supervised finetuning (SFT) on curated instruction data, are limited to post hoc rationalization. Here, the explanations, when produced at all, are generated after or independently of the decision, leaving the decision process unauditable. Pre-hoc reasoning, where a causal reasoning trace is produced before the output label, is preferable, and the broader LLM reasoning literature has made real progress toward it via reinforcement learning methods such as Group Relative Policy Optimization (GRPO). Here we observe that transplanting this recipe into the telecom setting runs into a cold-start barrier: SLMs either learn to output the desired format or learn to predict the label, but rarely both. We identify this barrier and propose CRAFT, which stands for Cold-start Reasoning Alignment via Fine-Tuning, a data-centric method to autonomously generate a verified dataset of (input, trace, label) triplets. CRAFT fine-tunes compact SLMs on this verified data using low-rank adaptation (LoRA), requiring substantially less compute and wall-clock time than GRPO-based methods. On the TRACTOR and Interference Classification (IC) xApp telecom datasets, CRAFT achieves up to 86.5% and 94.6% for accuracy and F1 with no parse failures, while direct GRPO and SFT+GRPO fail to exceed 28% and 53.5% F1 with multiple parse failures. We further show that CRAFT-initialized policies serve as a robust foundation for subsequent GRPO fine-tuning, as under diverse reward functions the performance remains consistent with no parse failures. Finally, we demonstrate that CRAFT consumes 59% less energy than GRPO-based baselines, making it a sustainable path to deployable, auditable AI in 6G RAN.

Index Terms—O-RAN, AI-RAN, LLM Training, xApps, rApps, Reasoning, Traffic Classification, Interference Classification

## I. INTRODUCTION

The evolution toward 6G networks is widely framed as a paradigm shift toward fully AI-native radio access networks (AI-RAN) [1], [2], [3], in which intelligence is not bolted onto the network but embedded within its control loops. Central to this vision is the O-RAN Alliance’s disaggregated architecture, which exposes RAN Intelligent Controllers (RICs) hosting xApps and rApps capable of automated, real-time decisionmaking about resource allocation, slicing, and interference mitigation [4]. As AI-RAN matures, these components are increasingly expected to operate agentically rather than as static classifiers, a trajectory reflected in emerging work such as Telecom Model Context Protocol (TeleMCP) [5], [4] and GENESIS [6], both of which open a pathway towards truly multi-agentic telecom systems that can solve end-to-end telecom tasks inside currently available hardware.

This shift places a new demand on the small language models (SLMs) which are designed for edge performance and fall between 100M-5B parameters [7] to populate xApps and rApps. They must not only read key performance indicator (KPI) telemetry and output a correct action, but do so with a human-readable justification that a network operator can audit before trusting the decision [4]. Existing state-of-theart recipes for telecom-specialized language models, however, are not built around this requirement. Methods such as RANSTRUCT [8] and TelecomGPT [9] fine-tune models to emit an answer directly, with no accompanying reasoning trace, while agentic systems such as AI5GTest [10] that do produce explanations, generate them only after the decision has already been made. This ordering is problematic: because the explanation is generated independently of, or after, the prediction, there is no guarantee that the stated justification is what actually drove the output, and the decision process itself remains unauditable. Worse, when these same models are instead prompted to reason before committing to a label, they tend to collapse, failing to output a valid label at all rather than reasoning productively toward one.

The broader LLM reasoning literature offers a template for avoiding this failure mode. Chain-of-thought prompting [11] and reinforcement learning methods such as Group Relative Policy Optimization (GRPO) [12] have made real progress toward eliciting genuine pre-hoc reasoning, in which a causal reasoning trace is generated first, and the label is derived from it, rather than the reverse. Naively transplanting this recipe into the telecom setting, however, runs into what we term a coldstart barrier: SLMs simply lack any prior over what a valid reasoning trace looks like to solve telecom tasks, so when GRPO’s reward signal is forced to jointly optimize for correct formatting, non-trivial reasoning content, and label accuracy, these objectives conflict early in training and exploration fails outright. In practice, we observe that SLMs subjected to this recipe learn to satisfy at most one of these objectives, either the output format or the label.

To address this, we propose CRAFT (Cold-start Reasoning Alignment via Fine-Tuning), a data-centric method that sidesteps the exploration difficulty of RL altogether by autonomously constructing a verified dataset of (input, trace, label) triplets and then fine-tuning a compact SLM on this dataset with ordinary low-rank supervised adaptation. Because verification happens once, offline, during dataset construction, rather than online during policy optimization, CRAFT requires substantially less compute and wall-clock time than GRPObased alternatives. On the TRACTOR and IC xApp telecom datasets, CRAFT achieves up to 86.5% and 94.6% macro-F1, respectively, with zero parse failures, while direct GRPO and SFT+GRPO baselines fail to exceed 28% and 53.5% F1 and exhibit substantial parse-failure rates. We further show that CRAFT-initialized policies provide a robust foundation for any subsequent GRPO fine-tuning a practitioner may still wish to apply, remaining stable in accuracy and format compliance across diverse reward formulations, and that CRAFT is up to 59% more energy-efficient than GRPO-based baselines, making it a sustainable path toward deployable, auditable AI in 6G RAN.

## II. BACKGROUND

## A. Large Language Models in Telecom

A growing body of work makes the case that language models are becoming a standard component of telecom R&D and operations rather than a research curiosity. On the evaluation side, ORAN-Bench-13K [13] contributes nearly 14,000 curated multiple-choice questions drawn from O-RAN specification documents and shows that general-purpose LLMs leave substantial headroom on O-RAN-specific knowledge, motivating retrieval-augmented and fine-tuned alternatives. TeleResilienceBench [14] probes a different, and for AI-RAN arguably more operationally relevant, capability: whether a model can recover when it inherits a partially completed or already-flawed reasoning trace from a prior step or upstream agent, rather than only being evaluated from a clean start; even the strongest models it tests recover correctly less than a third of the time, underscoring how far current reasoning behavior is from being trustworthy inside a live pipeline.

On the systems side, AI5GTest [10] and GENESIS [6] both use cooperative LLM agents to automate work that has traditionally required extensive manual engineering: AI5GTest for specification-aware conformance testing and validation of O-RAN components against 3GPP and O-RAN specifications, and GENESIS for a broader set of RAN research-anddevelopment tasks spanning feature synthesis, testing, hardening, optimization, and security, validated end-to-end against over-the-air experiments on a production O-RAN testbed. Finally, TelecomGPT [9] proposes a continual-pretraining, instruction-tuning, and alignment-tuning pipeline for adapting general-purpose LLMs to telecom, together with new benchmarks for telecom math modeling, open-ended QA, and code tasks, and shows the resulting model outperforming generalpurpose LLMs of comparable scale on telecom-specific evaluation. Taken together, this line of work makes a strong case that telecom-specialized language models are already delivering value and are becoming more central to AI-RAN pipelines. What remains comparatively underexplored, and what we address here, is not whether these models can produce a decision, but whether they can produce one whose reasoning trace is causally, rather than merely rhetorically, tied to that decision.

RANSTRUCT and TelecomGPT operationalize training an LM via Low-Rank Adaptation (LoRA). LoRA freezes pretrained weights $W _ { 0 } ~ \in ~ \bar { \mathbb { R } } ^ { d \times k }$ and injects trainable low-rank matrices $B \in \mathbb { R } ^ { d \times r } , A \in \mathbb { R } ^ { r \times k }$ with $r \ll$ min(d, k):

$$
h = W _ { 0 } x + { \frac { \alpha } { r } } B A x ,\tag{1}
$$

where α is a scaling factor. This approach yields strong labelprediction accuracy but provides no mechanism for auditable reasoning traces.

## B. Chain-of-Thought Reasoning

Chain-of-thought (CoT) [11] prompting is widely regarded as the first step towards reasoning in LLMs and improves accuracy on complex tasks by eliciting intermediate reasoning steps before the final answer. Formally, rather than modeling the label directly as $p _ { \theta } ( y \mid x )$ , a CoT-style model factorizes generation as $p _ { \theta } ( t , y \mid x ) = p _ { \theta } ( t \mid x ) p _ { \theta } ( y \mid x , t )$ , so that a reasoning trace t is sampled first and the label $y$ is decoded conditioned on both the input x and the trace. Self-consistency [15] decoding further boosts reliability by sampling K independent traces $t ^ { ( 1 ) } , \dots , t ^ { ( K ) } \sim p _ { \theta } ( t ^ { \cdot } | \mathbf { \Sigma } _ { x } )$ , decoding a label $y ^ { ( k ) } \sim p _ { \theta } ( y \mid x , t ^ { ( k ) } )$ for each, and aggregating over them via majority vote, $\begin{array} { r } { \hat { y } = \arg \operatorname* { m a x } _ { y } \sum _ { k = 1 } ^ { K } \bar { \mathcal { Y } } [ y ^ { ( k ) } = y ] } \end{array}$ . Because t is generated before $y$ and $y$ is conditioned on t, the trace can be causally upstream of the output label rather than a post-hoc gloss appended after the fact. In contrast to posthoc explanation techniques such as SHAP [16] or LIME [17], which fit a justification to an already-fixed prediction ${ \hat { y } } = f ( x )$ and therefore need not reflect the mechanism that actually produced yˆ, a pre-hoc trace can be made causally informative in a precise, testable sense: we retain a trace only if the label can be correctly recovered from the trace together with the input, i.e. if $p _ { \theta } ( y \mid x , t )$ assigns highest probability on the true label.

## C. Group Relative Policy Optimization

The transition from prompting chain-of-thought (CoT) to training models that reliably produce it has followed a welldocumented path in the LLM reasoning literature, going from SFT to Reinforcement learning. Now, RL for LLMs was popularized by Proximal Policy Optimization (PPO) [18], which trains a value network alongside the policy π to estimate the expected return of a partial generation and uses it as the baseline against which realized rewards are compared. That critic is comparable in size to the policy, so PPO can significantly increase the memory footprint of training. GRPO [12] removes the critic entirely and has become the de facto choice for reasoning-oriented training: for each input x it samples a group of $G$ outputs $\{ o _ { i } \} _ { i = { \ i } } ^ { G }$ from the behavior policy $\pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } .$ , scores each with a reward $r _ { i } ,$ , and uses the group itself as the baseline, giving a group-normalized advantage

$$
\hat { A } _ { i , \ell } = \frac { r _ { i } - \mathrm { m e a n } ( \mathbf { r } ) } { \mathrm { s t d } ( \mathbf { r } ) + \delta } ,\tag{2}
$$

where $\mathbf { r } ~ = ~ [ r _ { 1 } , \ldots , r _ { G } ]$ and $\delta > 0$ is a small stabilizing constant added by implementations of the original formulation. Under outcome supervision, this scalar is broadcast across every token ℓ of $o _ { i }$ . The policy is updated on a clipped surrogate objective inherited from PPO, with a KL penalty of weight $\beta$ added directly to the loss, anchoring π to a fixed reference policy $\pi _ { \mathrm { r e f } } ,$ typically the initial checkpoint. A reward function is usually a weighted composite of several verifiercomputed terms, which fall into two broad groups. The first is concerned with form: whether the output follows the required template and whether the reasoning trace it contains is nontrivial rather than empty or degenerate. These terms are taskagnostic, since the target structure is fixed by the prompt rather than by the problem. The second group scores correctness, and is the only part that changes with the task, which can be accuracy for classification problems or a bounded error-based score for a KPI regression or forecasting task, or by a pass rate over executable checks for a code-generating LLM.

## III. CRAFT

We now present CRAFT. At a high level, CRAFT breaks the cold-start cycle by synthetically constructing a dataset of (KPI input, reasoning trace, label) triples in which the trace is verified to be causally informative for the label, rather than merely plausible-sounding, and then fine-tuning a target SLM on this dataset with ordinary supervised learning. We first substantiate, empirically, why this bootstrapping step is necessary (Section III-A), and then formalise the verification procedure and the resulting training objective (Section III-B).

## A. Motivation

To motivate CRAFT empirically, we evaluate four naive training strategies on TRACTOR (see Section IV for full dataset details) using Qwen 3.5 2B as the base model, with each KPI window serialized into the prompt as per-feature summary statistics. The first strategy is label-only SFT, essentially the RANSTRUCT recipe, in which the model is trained on input-label pairs with no reasoning trace at all. The second augments the output template with a post-hoc <think> field, but the supervised training target still contains only the label, so any reasoning the model happens to produce is never actually supervised. The third is standalone GRPO applied directly to the base model, using a composite reward that jointly encourages correct output format, the presence of a non-trivial reasoning trace, and label accuracy. The fourth is SFT followed by GRPO, the common warm-start recipe in which the RL stage is initialized from a policy already tuned for label prediction.

The results of this comparison are stark. When simply asked to emit a label, the SFT model reaches 77% accuracy with no parse failures; however, once the same model is instead prompted to reason first and then answer, both accuracy and parse behavior collapse to 0% accuracy with a 100% parsefailure rate. Standalone GRPO exhibits the identical failure: 100% parse failures and 0% accuracy, as does SFT followed by GRPO. The example below illustrates a representative failure: rather than reasoning toward a label, the model becomes trapped repeating fragments of its own format instructions and never emits a parseable prediction.

## Cold-start Failure Example

Input KPI snippet:   
PRB\_req mean=424.7 (max=1.25e4, median=0),   
DL buffer max=151.7 kB (median=0) ...   
DL bitrate mean=0.119 Mbps (std=0.54), UL   
bitrate mean=0.115 Mbps ...   
Ground truth: eMBB.   
Parsed prediction: None (format collapse; no   
valid label).   
Model output (cold-start + reasoning prompt):   
<think>   
LABEL must be exactly one of: eMBB, mMTC, URLLC,   
ctrl. No other text, JSON, or extra <think>   
tags.   
</think>   
LABEL must be exactly one of: eMBB, mMTC, URLLC,   
ctrl. No other text, JSON, or extra <think>   
tags.   
user   
<think>   
The user wants me to classify the network slice from   
KPI window statistics using the provided rules   
- I need to analyze the KPI metrics (means, slopes,   
spread, etc.) across the 64 rows.   
- I must output exactly two blocks: first <SOLUTION>   
then </SOLUTION>.   
- The LABEL must be one of: eMBB, mMTC, URLLC, ctrl.   
- No other text, JSON, or extra <think> tags.   
</think>   
<think>   
LABEL must be exactly one of: eMBB, mMTC, URLLC,   
ctrl. No other text, JSON, or extra <think>   
tags.   
</think>   
LABEL must be exactly one of: eMBB, mMTC, URLLC,   
ctrl. No other text, JSON, or extra <think>   
tags.

## B. Algorithm

CRAFT breaks this cycle by synthetically constructing a dataset of (KPI input, reasoning trace, label) triples in which every trace is verified to be causally informative for its label. The construction procedure introduces two roles played by the same base model: an Oracle Reasoner $\mathcal { M } _ { \theta _ { R } }$ and a Predictor $\mathcal { M } _ { \theta _ { P } } ,$ both instantiated from the same underlying model but invoked with different prompts. The Oracle Reasoner is given the KPI window and its ground-truth label, and is asked to produce a detailed reasoning trace that justifies that label. The Predictor, by contrast, receives the KPI window together with the Oracle’s generated trace wrapped in <think> tags, and must recover the label from this pairing; an example is retained only if the Predictor’s output matches the ground truth. This verification step is what distinguishes CRAFT’s traces from ordinary rationalizations: because the Predictor has no access to the label and must derive it by leveraging the trace, a retained trace should contain the information needed for the decision, making it a viable pre-hoc and auditable reasoning path rather than a post-hoc gloss.

Algorithm 1 CRAFT Verified Reasoning Dataset Generation   
Require: Labeled dataset $\begin{array} { r c l } { \mathcal { D } } & { = } & { \{ ( \boldsymbol { x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N } ; } \end{array}$ Oracle Reasoner   
$\mathcal { M } _ { \theta _ { R } } \colon$ Predictor $\mathcal { M } _ { \theta _ { P } } \colon$ min trace length τ<sub>min</sub>;   
Ensure: Verified dataset $\mathbf { \dot { \boldsymbol { D } } } ^ { \prime }$   
1: $\mathcal { D } ^ { \prime }  \emptyset$   
2: for $i = 1$ to N do   
3: $x  \mathrm { F O R M A T K P I S } ( x _ { i } ) ; y  y _ { i }$   
4: $P _ { R } \gets \mathrm { O R A C L E P R O M P T } ( x , y )$   
5: $r \gets \mathrm { G E N E R A T E } ( \mathcal { M } _ { \theta _ { R } } , \dot { P } _ { R } )$   
6: t ← EXTRACTTHINKING(r)   
7: if t = ⊥ or $| t | < \tau _ { \operatorname* { m i n } }$ or ¬FORMATOK(r) then   
8: continue   
9: end if   
10: $P _ { P } $ PREDICTORPROMPT $( x , t )$   
11: $p \gets \mathrm { G E N E R A T E } ( \mathcal { M } _ { \theta _ { P } } , P _ { P } )$   
12: $\hat { y } \gets \mathrm { P A R S E L A B E L } ( p )$   
13: if $\hat { y } \neq \perp$ and $\hat { y } = \overset { \cdot } { y }$ then   
14: $\mathcal { D } ^ { \prime }  \mathcal { D } ^ { \prime } \cup \{ ( x , t , y ) \}$   
15: end if   
16: end for   
17: return $\mathcal { D } ^ { \prime }$

Algorithm 1 formalizes this procedure. For each labeled example $( x _ { i } , y _ { i } )$ in the source dataset D, the KPI window is first serialized into a textual prompt, and the Oracle is queried with both the window and the ground-truth label to generate a candidate reasoning trace. The trace is then subjected to three filtering conditions: minimum trace length $\tau _ { \mathrm { m i n } }$ , wellformedness of the surrounding output structure, and successful extraction, which is indicated by $\neq \bot ;$ any failure of which discards the example outright. Surviving traces are passed to the Predictor together with the original KPI window, and the predicted label is parsed and compared against the ground truth; the triple $( x , t , y )$ is added to the verified dataset $\mathcal { D } ^ { \prime }$ only if this comparison succeeds. Once $\mathcal { D } ^ { \prime }$ has been assembled, the target SLM is fine-tuned on these verified triples using LoRA (Equation 1) with a standard language-modeling loss over the concatenation of the reasoning trace and the label.

## IV. EXPERIMENTAL SETUP

Datasets. We evaluate CRAFT on two telecom datasets; TRACTOR [19] is an O-RAN near-real-time (near-RT) RIC network-slice traffic classification task built from nonoverlapping gNB KPI windows of 64 samples at 250ms resolution (16s). Each window is represented by per-KPI summary statistics over 17 KPIs and assigned to one of four classes: eMBB, mMTC, URLLC, or ctrl. The resulting dataset contains 1,575 windows. IC xApp [20] is a near-RT RIC RF interference detection dataset. Here we form non-overlapping 15- sample windows over four uplink KPIs (ul\_snr, ul\_mcs, ul\_bitrate, and $\mathtt { u l \_ b l e r } )$ and classify each window as clean or interference, resulting in 389 windows. Thus, IC differs from TRACTOR in its KPI composition, temporal granularity, and label cardinality, allowing us to assess whether CRAFT’s benefits extend beyond multiclass networkslice classification. Both datasets are split 70/15/15 into train, validation, and test partitions, with the same fixed split used across every method we compare.

Models. We fine-tune three target SLMs, Qwen 3.5 2B, Qwen 3.5 4B [21], and Nemotron-3-Nano 4B [22], chosen for their strong prior performance in TeleResilienceBench [14]. For CRAFT’s dataset-generation stage, both the Oracle Reasoner and the Predictor are instantiated as Gemma 4 31B [23], a larger model whose role is confined to offline dataset construction rather than deployment.

Baselines. We compare CRAFT against four alternative training paradigms. Zero Shot uses a chain-of-thought prompt with no fine-tuning at all. SFT performs ordinary supervised fine-tuning on input-label pairs with no reasoning-trace supervision. GRPO applies Group Relative Policy Optimization directly to the base model with a composite reward $r ~ = ~ w _ { \mathrm { f m t } } r _ { \mathrm { f m t } } + w _ { \mathrm { t h i n k } } r _ { \mathrm { t h i n k } } + w _ { \mathrm { a c c } } r _ { \mathrm { a c c } }$ that jointly rewards output format, reasoning presence, and label accuracy with equal weights for all three terms. SFT+GRPO applies the same composite reward via GRPO but initializes from a labelonly SFT checkpoint. CRAFT is our proposed method: LoRA fine-tuning on the verified reasoning dataset produced by Algorithm 1, with no reinforcement learning involved.

Training details. All LoRA adapters use rank 16, $\alpha = 3 2 ,$ and dropout 0, We optimise with AdamW [24] at learning rate $5 \times 1 0 ^ { - 6 }$ with 10% linear warmup and a linear decay schedule. Each GRPO step samples 4 completions per prompt, with gradient accumulation 1. Every run is conducted on a single RTX 4090 with 24GB VRAM under a fixed 12-hour compute budget, and all training and inference are performed through Unsloth [25]; the hyperparameters are chosen to avoid training failures in our limited compute setup.

Metrics. We report accuracy and macro-F1 for classification quality, parse-failure rate (PF%, the fraction of test outputs from which no valid label could be extracted), Think% (the fraction of outputs containing a non-empty <think> block), Solution% (the fraction of outputs containing a solution block), and training wall-time.

## A. Ablation Studies

To probe the robustness, generalizability, and efficiency of CRAFT beyond the main comparison, we design three ablation studies. The first examines reward robustness under continued GRPO: starting from the best CRAFT-tuned model (Qwen 3.5 4B on TRACTOR), we continue training with GRPO for the remainder of the 12-hour budget under three reward weightings; a balanced static setting with $w _ { \mathrm { f m t } } \ =$ w<sub>think</sub> $= \ w _ { \mathrm { a c c } } \ = \ 0 . 5$ , a correctness-priority static setting with $w _ { \mathrm { a c c } } ~ = ~ 0 . 9$ and $w _ { \mathrm { f m t } } = w _ { \mathrm { t h i n k } } = 0 . 3$ , and a random dynamic setting in which all three weights are resampled independently from [0.2, 1.0] at every step. The balanced setting reflects typical practice; the correctness-priority setting stresses whether emphasizing label accuracy erodes format compliance in the absence of a strong format prior; and the random dynamic setting, by continually shifting the objective, constitutes the hardest test of stability. We report macro-F1 and PF% for each, expecting CRAFT’s initialization to sustain high performance across all three schemes in a way that GRPO trained from scratch cannot.

The second ablation tests generalization to IC xApp: we repeat the comparison, all baselines and CRAFT with the bestperforming SLM from the TRACTOR experiments, on the IC xApp dataset to verify that the cold-start phenomenon and CRAFT’s remedy are not artefacts specific to TRACTOR. We present side-by-side bar plots of macro-F1 and PF% across methods to make the failure mode and CRAFT’s resolution of it directly comparable across datasets. The third ablation quantifies energy efficiency: using the NVIDIA Management Library, we measure total GPU energy consumption in Joules during training for SFT, GRPO, SFT+GRPO, and CRAFT (Zero Shot is excluded, as it involves no training). This lets us contextualize CRAFT’s wall-time advantage in terms of actual energy cost, which is directly relevant to the sustainability of deploying auditable reasoning models at scale in AI-RAN.

## V. RESULTS AND DISCUSSION

Main results. Table I reports accuracy, macro-F1, parsefailure rate, Think%, Solution%, and training wall-time for all five training paradigms across all three SLMs on TRACTOR for producing pre-hoc reasoning. Zero-shot prompting is weak and uneven across models: Qwen 3.5 4B reaches 32.9% accuracy and 26.3% macro-F1 with no parse failures, but Qwen 3.5 2B and Nemotron-3-Nano collapse entirely, with 0% accuracy and a 100% parse-failure rate, indicating that smaller models cannot reliably follow the required output format without any fine-tuning at all. Label-only SFT after one epoch reaches 54.0% accuracy and 51.1% macro-F1 on Qwen 3.5 4B, with a low 3.8% parse-failure rate and a 96.6% Think rate; notably this is not the total-collapse failure mode described in Section III, since Qwen 3.5 4B is large enough to produce some usable reasoning even when only the label is supervised, but the same recipe still leaves Qwen 3.5 2B and Nemotron-3-Nano at 0% accuracy with parse-failure rates near 100%. GRPO and SFT+GRPO under the balanced reward achieve 35.0%/28.0% and 56.5%/53.5% accuracy/macro-F1, respectively, on Qwen 3.5 4B, and while their parse-failure rates remain low (0% and 2.5%), their accuracy stays well below what CRAFT achieves. CRAFT itself, fine-tuned on the verified reasoning dataset for five epochs, reaches 83.1% accuracy and 86.5% macro-F1 with 0% parse failures and a 100% Think rate, training in maximum 5.8 hours when shared dataset-preparation time is included comfortably within the 12- hour budget. Averaged across all three SLMs, CRAFT attains approximately 81% accuracy with 0% parse failures, a result that holds even for Qwen 3.5 2B and Nemotron-3-Nano, both of which collapse entirely under every other training paradigm.

## A. Ablation 1: reward robustness under continued GRPO.

Figure 1 shows accuracy and macro-F1 for Qwen 3.5 4B under the three CRAFT+GRPO reward variants, and Figure 2 shows the corresponding training reward trajectories alongside GRPO and SFT+GRPO. All three reward weightings sustain a macro-F1 in the 85-86% range with 0% parse failures, confirming that CRAFT’s initialization is robust to substantial changes in reward shaping, including the deliberately unstable random-dynamic setting. Continued GRPO does not, however, improve on CRAFT alone: every CRAFT+GRPO variant scores slightly below the CRAFT baseline of 83.1% accuracy and 86.5% macro-F1, landing between 81.9-82.7% accuracy and 85.1-86.1% macro-F1. This suggests that once a policy already reasons correctly and consistently, additional reinforcement learning mainly redistributes probability mass within an already-strong policy, effectively overfitting to the reward signal rather than unlocking further gains.

TABLE I  
MAIN RESULTS ON TRACTOR. METRICS ARE IN PERCENTAGES (%) UNLESS OTHERWISE NOTED.
<table><tr><td>Model</td><td>Method</td><td>Acc.</td><td>M-F1</td><td>PF</td><td>Think</td><td>Sol.</td><td>Time(h)</td></tr><tr><td rowspan="5">Qwen 3.5 2B</td><td>Zero Shot</td><td>0.0</td><td>0.0</td><td>100.0</td><td>94.1</td><td>93.7</td><td>0</td></tr><tr><td>SFT</td><td>0.0</td><td>0.0</td><td>100.0</td><td>0.0</td><td>0.0</td><td>0.25</td></tr><tr><td>GRPO</td><td>0.0</td><td>0.0</td><td>100.0</td><td>90.7</td><td>89.9</td><td>12.0</td></tr><tr><td>SFT+GRPO</td><td>0.0</td><td>0.0</td><td>100.0</td><td>88.6</td><td>0.0</td><td>12.0</td></tr><tr><td>CRAFT (Ours)</td><td>81.0</td><td>84.0</td><td>0.0</td><td>100.0</td><td>100.0</td><td>4.39</td></tr><tr><td rowspan="5">Qwen 3.5 4B</td><td>Zero Shot</td><td>32.9</td><td>26.3</td><td>0.0</td><td>100.0</td><td>100.0</td><td>0</td></tr><tr><td>SFT</td><td>54.0</td><td>51.1</td><td>3.8</td><td>96.6</td><td>96.6</td><td>0.52</td></tr><tr><td>GRPO</td><td>35.0</td><td>28.0</td><td>0.0</td><td>100.0</td><td>100.0</td><td>12.0</td></tr><tr><td>SFT+GRPO</td><td>56.5</td><td>53.5</td><td>2.5</td><td>97.9</td><td>97.5</td><td>12.0</td></tr><tr><td>CRAFT (Ours)</td><td>83.1</td><td>86.5</td><td>0.0</td><td>100.0</td><td>100.0</td><td>5.82</td></tr><tr><td rowspan="5">Nemotron-3- Nano 4B</td><td>Zero Shot</td><td>0.0</td><td>0.0</td><td>100.0</td><td>46.4</td><td>100.0</td><td>0</td></tr><tr><td>SFT</td><td>0.0</td><td>0.0</td><td>99.2</td><td>0.0</td><td>17.3</td><td>0.17</td></tr><tr><td>GRPO</td><td>0.0</td><td>0.0</td><td>100.0</td><td>40.5</td><td>100.0</td><td>12.0</td></tr><tr><td>SFT+GRPO</td><td>0.0</td><td>0.0</td><td>99.6</td><td>0.8</td><td>16.9</td><td>12.0</td></tr><tr><td>CRAFT (Ours)</td><td>80.2</td><td>83.0</td><td>0.0</td><td>100.0</td><td>100.0</td><td>3.94</td></tr></table>

![](images/6b9c4870eb6067d8a0ea0c7c43860fd52d955c4c7c5436ec3af9f403b5061d86.jpg)  
Fig. 1. Qwen3.5-4B CRAFT+GRPO under three reward-weight modes (balanced, correctness, random): accuracy and F1.

## B. Ablation 2: generalisation to IC xApp.

Figure 3 presents the same comparison on IC xApp for Qwen 3.5 4B, and the failure mode is, if anything, sharper than on TRACTOR. Zero-shot prompting achieves 1.7% accuracy and 2.8% macro-F1 with a 98% parse-failure rate; labelonly SFT reaches 18.6% accuracy and 26.2% macro-F1 but still fails to parse 81% of outputs; GRPO matches zeroshot exactly, at 1.7%/2.8% with 98% parse failures; and SFT+GRPO reaches 11.9% accuracy and 16.7% macro-F1 with an 88% parse-failure rate. CRAFT, by contrast, achieves 94.9% accuracy, 94.6% macro-F1, 0% parse failures, and a 100% Think rate. Despite IC xApp’s different KPIs, shorter windows, and binary rather than four-way decision, the qualitative pattern is identical to TRACTOR: only CRAFT can simultaneously maintain high accuracy and valid reasoning format, confirming that the cold-start barrier and CRAFT’s remedy for it generalize across tasks rather than being an artefact of TRACTOR specifically.

![](images/820bedace590a83e29964f9ad3dc8be5d4d679d8c383d137e2553bc047ae2189.jpg)

Fig. 2. Qwen3.5-4B CRAFT+GRPO, SFT+GRPO, and GRPO training reward trajectories.  
![](images/ba623000f8e43e8efdadcc76a68b45e26778a26d1fb4f51f2d9c65e4fdae187c.jpg)  
Fig. 3. IC xApp ablation: F1 and parse-fail rate (PF) across training approaches.

## C. Ablation 3: energy efficiency.

Figure 4 reports training energy averaged across all three SLMs. SFT is cheapest at 0.08 kWh, reflecting its single epoch of label-only supervision, while GRPO and SFT+GRPO consume 2.73 kWh and 2.75 kWh, respectively, dominated by the 12-hour RL budget. CRAFT consumes 1.13 kWh in total, split between 0.70 kWh for verified dataset preparation and 0.43 kWh for the subsequent LoRA fine-tuning stage. This amounts to roughly 59% less energy than either GRPObased baseline, underscoring that CRAFT’s advantage is not merely one of wall-clock convenience but of substantially lower absolute compute and energy costs, a property directly relevant to deploying auditable reasoning models sustainably at the network edge.

![](images/621c1133d1fb4176ee4ecaa7d9f390fb44c6f5c45ebd3ea6eda7c51d40c8fe12.jpg)  
Fig. 4. Average training energy across three models, measured as mean GPU power × recorded train time. Stacked segments separate SFT, GRPO, CRAFT data prep (DP), and CRAFT train (T).

## VI. LIMITATIONS

Dataset coverage. Our evaluation spans two telecom datasets, TRACTOR and IC xApp. Both are established, widely used O-RAN near-RT RIC benchmarks that cover distinct tasks: four-way slice classification and binary interference detection over different KPI families and window lengths, and the consistency of CRAFT’s advantage across both is what gives us confidence that the cold-start barrier is not an artefact of either dataset individually. Even so, our evaluation is limited to classification-style decisions with discrete label sets; tasks that require continuous-valued outputs, such as KPI regression or forecasting, fall outside this scope, and extending CRAFT’s verification procedure to such settings is a natural direction for future work.

Model family diversity. We fine-tune three target SLMs, but these come from only two underlying model families, Qwen and Nemotron. Because CRAFT’s cold-start diagnosis and remedy could in principle interact with family-specific pretraining choices, such as tokenizer design or instructiontuning recipe, results on a broader set of model families would strengthen the claim.

## VII. CONCLUSION

This paper set out from the observation that pre-hoc reasoning that precedes and causally determines a model’s output is essential to building trustworthy, auditable AI for 6G RAN, yet existing training recipes for telecom SLMs fail to produce it. We traced this failure to a cold-start exploration barrier: when a small model with no prior over valid telecom reasoning traces is trained via GRPO, it has to simultaneously satisfy conflicting objectives over format, reasoning content, and accuracy, where it collapses rather than learning either objective robustly. We introduced CRAFT, a data-centric pipeline that sidesteps this barrier entirely by verifying reasoning traces offline using an Oracle Reasoner and a Predictor role played by the same base model before ever touching the target SLM, and then fine-tuning that SLM with ordinary, stable supervised learning rather than reinforcement learning. Across three SLMs and two telecom datasets, CRAFT delivers substantially higher accuracy, near-perfect format compliance, and large reductions in both training time and energy relative to GRPO-based baselines, while remaining robust to further RL fine-tuning under diverse reward signals rather than degrading under it. Taken together, these results position CRAFT-trained models as a practical, drop-in component for AI-RAN xApps and rApps that require both competent decision-making and a genuinely auditable rationale behind every decision.

Future work. We plan to directly address the scope restrictions discussed in the limitations section: broadening evaluation beyond TRACTOR and IC xApp to additional telecom decision tasks and potential hardware validation with context protocols like TeleMCP.

## ACKNOWLEDGMENT

Authors acknowledge the funding support from the Public Wireless Supply Chain Innovation Fund (PWSCIF) under Federal Award ID 51-60-IF007.

## REFERENCES

[1] N. A. Khan and S. Schmid, “Ai-ran in 6g networks: State-of-the-art and challenges,” IEEE Open Journal of the Communications Society, vol. 5, pp. 294–311, 2023.

[2] L. Kundu, X. Lin, R. Gadiyar, J.-F. Lacasse, and S. Chowdhury, “Airan: Transforming ran with ai-driven computing infrastructure,” IEEE Communications Magazine, 2025.

[3] M. Polese, N. Mohamadi, S. D’Oro, L. Bonati, and T. Melodia, “Beyond connectivity: An open architecture for ai-ran convergence in 6g,” IEEE Communications Magazine, 2026.

[4] P. Gajjar and V. K. Shah, “Agents should replace narrow predictive ai as the orchestrator in 6g ai-ran,” arXiv preprint arXiv:2605.11516, 2026.

[5] P. Gajjar, C. Shen, and V. K. Shah, “Tele-llm-hub: Building contextaware multi-agent llm systems for telecom networks,” arXiv preprint arXiv:2511.09087, 2025.

[6] T. Aghayev, M. Elkael, M. Polese, M. D. Nguyen, G. Gemmi, A. Lacava, A. Saeizadeh, R. Prasad, P. Testolina, A. Feraudo, et al., “Genesis: Harnessing ai agents for autonomous 6g ran synthesis, research, and testing,” arXiv preprint arXiv:2605.27360, 2026.

[7] Z. Lu, X. Li, D. Cai, R. Yi, F. Liu, X. Zhang, N. D. Lane, and M. Xu, “Small language models: Survey, measurements, and insights,” 2025.

[8] P. Gajjar and V. K. Shah, “Oransight-2.0: Foundational llms for oran,” IEEE Transactions on Machine Learning in Communications and Networking, 2025.

[9] H. Zou, Q. Zhao, Y. Tian, L. Bariah, F. Bader, T. Lestable, and M. Debbah, “Telecomgpt: A framework to build telecom-specific large language models,” IEEE Transactions on Machine Learning in Communications and Networking, 2025.

[10] A. Ganiyu, P. Gajjar, and V. K. Shah, “Ai5gtest: Ai-driven specificationaware automated testing and validation of 5g o-ran components,” in 18th ACM Conference on Security and Privacy in Wireless and Mobile Networks, pp. 53–64, 2025.

[11] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al., “Chain-of-thought prompting elicits reasoning in large language models,” Advances in neural information processing systems, vol. 35, pp. 24824–24837, 2022.

[12] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. Li, Y. Wu, et al., “Deepseekmath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[13] P. Gajjar and V. K. Shah, “Oran-bench-13k: An open source benchmark for assessing llms in open radio access networks,” arXiv preprint arXiv:2407.06245, 2024.

[14] P. Gajjar, E. Ojo, and V. K. Shah, “Teleresiliencebench: Quantifying resilience for llm reasoning in telecommunications,” arXiv preprint arXiv:2605.09929, 2026.

[15] X. Wang, J. Wei, D. Schuurmans, Q. Le, E. Chi, S. Narang, A. Chowdhery, and D. Zhou, “Self-consistency improves chain of thought reasoning in language models,” 2023.

[16] E. Mosca, F. Szigeti, S. Tragianni, D. Gallagher, and G. Groh, “Shapbased explanation methods: a review for nlp interpretability,” in Proceedings of the 29th international conference on computational linguistics, pp. 4593–4603, 2022.

[17] J. Dieber and S. Kirrane, “Why model why? assessing the strengths and limitations of lime,” arXiv preprint arXiv:2012.00093, 2020.

[18] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” 2017.

[19] J. Groen, M. Belgiovine, U. Demir, B. Kim, and K. Chowdhury, “Tractor: Traffic analysis and classification tool for open ran,” in ICC 2024-IEEE International Conference on Communications, pp. 4894– 4899, IEEE, 2024.

[20] A. Chiejina, B. Kim, K. Chowhdury, and V. K. Shah, “System-level analysis of adversarial attacks and defenses on intelligence in o-ran based cellular networks,” in Proceedings of the 17th ACM Conference on Security and Privacy in Wireless and Mobile Networks, pp. 237–247, 2024.

[21] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[22] A. Blakeman, A. Grattafiori, A. Basant, A. Gupta, A. Khattar, A. Renduchintala, A. Vavre, A. Shukla, A. Bercovich, A. Ficek, et al., “Nemotron 3 nano: Open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning,” arXiv preprint arXiv:2512.20848, 2025.

[23] G. Team, S. E. Abd, V. Aggarwal, R. Algayres, A. Andreev, O. Bachem, I. Ballantyne, C. Brick, V. Carbune, M. Casbon,˘ et al., “Gemma 4 technical report,” arXiv preprint arXiv:2607.02770, 2026.

[24] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” 2019.

[25] M. H. Daniel Han and U. team, “Unsloth,” 2023.