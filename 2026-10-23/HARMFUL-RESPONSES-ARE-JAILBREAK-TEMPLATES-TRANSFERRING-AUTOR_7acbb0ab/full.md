# HARMFUL RESPONSES ARE JAILBREAK TEMPLATES: TRANSFERRING AUTOREGRESSIVE JAILBREAKS TO DIFFUSION LANGUAGE MODELS

Jinwei Gan<sup>1∗</sup> Zifeng Cheng<sup>1∗†</sup> Zhiwei Jiang<sup>1†</sup> Ran Sun<sup>1</sup> Shiping Ge<sup>2</sup> Qing Gu<sup>1</sup> <sup>1</sup>State Key Laboratory for Novel Software Technology, Nanjing University

<sup>2</sup>Nanjing University of Posts and Telecommunications

ganjw@smail.nju.edu.cn, chengzf@nju.edu.cn, jzw@nju.edu.cn, spge@njupt.edu.cn, guq@nju.edu.cn

## ABSTRACT

Diffusion large language models (dLLMs) enable flexible masked generation through bidirectional context modeling, but this capability also introduces a distinct jailbreak attack surface. We find that a harmful response to a given query naturally serves as an effective jailbreak template: simply masking a subset of its tokens can induce a dLLM to reconstruct harmful content. Based on this observation, we propose RMT-JAILBREAK, a general framework that transfers existing autoregressive (AR) jailbreak attacks to dLLMs through their induced harmful responses. RMT-JAILBREAK converts an AR-generated harmful response into a high-mask-ratio template using uniform token retention, and further applies prompt-based safety filtering to mask any remaining visible harmful content. The refined template is then provided to the target dLLM for bidirectional reconstruction. Extensive experiments across multiple AR jailbreak methods and target dLLMs demonstrate the effectiveness and generality of RMT-JAILBREAK, revealing strong cross-paradigm transferability of jailbreak attacks.

## 1 INTRODUCTION

Diffusion large language models (dLLMs) (Nie et al., 2025; Ye et al., 2025; Zhu et al., 2026; Yang et al., 2025) have recently emerged as an alternative paradigm to conventional autoregressive (AR) language models. Unlike AR models that generate tokens sequentially from left to right, dLLMs formulate generation as an iterative denoising process over partially masked sequences. This paradigm enables multiple tokens to be decoded in parallel and allows each masked position to be predicted from bidirectional context. Consequently, masked positions can appear at arbitrary locations in the sequence and be progressively recovered based on both preceding and following context. These properties provide dLLMs with flexible generation capabilities beyond standard left-to-right continuation, including infilling, rewriting, and formatted generation.

This flexibility also leads to a fundamental difference between jailbreak attacks on dLLMs and those on autoregressive models. Jailbreak attacks on autoregressive LLMs (Zou et al., 2023; Liu et al., 2024; Chao et al., 2025; Ding et al., 2024) typically manipulate the input prompt to steer the subsequent left-to-right generation toward harmful responses. In contrast, the bidirectional generation mechanism of dLLMs enables adversarial contexts to be constructed beyond a continuous input prefix. Recent work has begun to exploit this unique property for jailbreak attacks. In particular, DIJA (Wen et al., 2025) manually designs several interleaved mask-text templates and leverages the in-context learning capability of LLMs to transform the original harmful prompt into new jailbreak prompts following these templates.

Table 1: Attack Success Rate on the HarmBench benchmark using LLaDA-8B-Instruct as the target model.
<table><tr><td>Mask ratio ρ</td><td>ASR (%)</td></tr><tr><td>0.3</td><td>97.75</td></tr><tr><td>0.5</td><td>97.00</td></tr><tr><td>0.7</td><td>96.50</td></tr><tr><td>0.9</td><td>85.50</td></tr></table>

Although effective, DIJA relies on manually designed contextual templates, and the generated prompts largely follow generic attack scaffolds, resulting in limited query-specific relevance. We find that a harmful response to a given query naturally serves as a jailbreak template: simply masking a subset of its tokens is sufficient to construct a highly effective jailbreak input. As shown in Table 1, this simple strategy achieves consistently high Attack Success Rates across a wide range of mask ratios, indicating that only partial exposure of a query-specific harmful response can provide sufficient context to induce harmful reconstruction by dLLMs. This finding motivates us to leverage harmful responses, rather than manually designed attack scaffolds, as the basis for constructing dLLM jailbreak inputs.

Building on this observation, we propose RMT-JAILBREAK, a simple response-mediated transfer framework that repurposes existing AR jailbreak attacks for dLLMs through their induced harmful responses. Given a harmful query, RMT-JAILBREAK first applies an off-the-shelf AR jailbreak method to an autoregressive model to obtain a query-specific harmful response. It then converts this response into a high-mask-ratio template by retaining sparse contextual tokens at approximately uni form intervals, providing distributed semantic and structural cues for subsequent reconstruction. To avoid directly exposing explicit harmful content, we further employ a prompt-based safety filtering stage that replaces any remaining visible harmful tokens with additional masks while preserving the template structure. Finally, the filtered template is concatenated with the original query and fed into the target dLLM, which reconstructs the missing content through its native bidirectional infilling process. This design allows a wide range of existing AR jailbreak methods to be readily repurposed for dLLMs without requiring dLLM-specific prompt optimization.

## Our main contributions are summarized as follows:

• We reveal that harmful responses themselves can serve as effective jailbreak templates for dLLMs. Simply masking a subset of tokens in a harmful response is sufficient to induce highly successful harmful reconstruction, exposing a previously underexplored vulnerability of dLLMs.

• We propose RMT-JAILBREAK, a simple and general framework that transfers existing autoregressive jailbreak methods to dLLMs through their induced harmful responses. By combining uniform token retention with prompt-based safety filtering, RMT-JAILBREAK constructs effective dLLM jailbreak inputs without requiring dLLM-specific prompt optimization.

• Extensive experiments across multiple AR jailbreak methods and target dLLMs demonstrate the effectiveness and generality of RMT-JAILBREAK.

## 2 RELATED WORKS

## 2.1 DIFFUSION LARGE LANGUAGE MODELS

Diffusion models have achieved strong performance in continuous generation domains such as images (Ho et al., 2020; Song et al., 2021; Rombach et al., 2022). Extending diffusion models to discrete text generation has led to masked diffusion models, which generate text by iteratively predicting masked tokens conditioned on surrounding context (Austin et al., 2021; Hoogeboom et al., 2021; Campbell et al., 2022; Lou et al., 2024; Shi et al., 2024). Recent dLLMs, such as LLaDA (Nie et al., 2025), Dream (Ye et al., 2025), LLaDA-1.5 (Zhu et al., 2026), and MMaDA (Yang et al., 2025), demonstrate that diffusion language modeling can be a competitive alternative to autoregressive modeling. Their bidirectional context modeling and parallel decoding provide new capabilities for infilling, editing, and structured generation. However, these same properties also introduce safety behaviors that differ from those of AR LLMs, requiring architecture-aware safety analysis.

## 2.2 JAILBREAK ATTACKS ON AUTOREGRESSIVE AND DIFFUSION LLMS

A large body of work studies jailbreak attacks against aligned AR LLMs. These attacks broadly include optimization-based methods, iterative black-box attacks, and prompt-rewriting approaches.

![](images/a049e756bec0aa5b98b7da9cde2a567c3629fb96bc05396f6c46c4aaa0a6d9df.jpg)  
Figure 1: Overview of RMT-JAILBREAK. The method sequentially induces responses from the source AR model, converts them into uniformly masked templates, applies safety filtering, and attacks the target diffusion language model through masked infilling.

Optimization-based methods such as GCG (Zou et al., 2023) search for adversarial suffixes that induce harmful completions, while AutoDAN (Liu et al., 2024) uses automatic prompt optimization to generate stealthy jailbreak prompts. PAIR (Chao et al., 2025) formulates jailbreak generation as an iterative black-box attack, and ReNeLLM (Ding et al., 2024) constructs nested or rewritten prompts to bypass safety alignment. These methods are primarily designed for left-to-right generation and do not explicitly account for masked denoising in diffusion Large Language Models (dLLMs).

More recently, DIJA (Wen et al., 2025) has explored jailbreak attacks tailored to dLLMs. It constructs interleaved mask-text prompts to exploit their bidirectional modeling and parallel generation capabilities. Our work further investigates how the distribution of masked positions affects jailbreak effectiveness in dLLMs. By converting responses induced by AR jailbreak methods into uniformly masked templates, RMT-JAILBREAK bridges AR jailbreaks and dLLM infilling. The distributed visible context supports reconstruction of the response during generation.

## 3 METHOD

Inspired by the observation that harmful responses naturally serve as effective jailbreak templates for dLLMs, we propose RMT-JAILBREAK, which leverages harmful responses induced from AR models to construct jailbreak inputs for dLLMs.

As illustrated in Figure 1, RMT-JAILBREAK proceeds in three phases. (1) Source AR induction: an existing AR jailbreak method uses the harmful query to induce a response from a source AR model. (2) Masked template transfer: uniform masking converts this response into a high-maskratio template while preserving sparse contextual anchors; an auxiliary LLM then masks residual visible harmful tokens to obtain a filtered template. (3) Target dLLM jailbreak: the harmful query and filtered template form the attack input, which the target dLLM completes through bidirectional denoising. Algorithm 1 summarizes the complete procedure.

## 3.1 SOURCE AR INDUCTION

Given a harmful query q, we first apply an existing autoregressive jailbreak method A to an AR source model $M _ { \mathrm { A R } }$ to obtain a harmful response r. We denote this process as

$$
r = \mathcal { I } _ { A } ( M _ { \mathrm { A R } } , q ) ,\tag{1}
$$

where $\mathcal { I } _ { A }$ denotes the AR jailbreak pipeline instantiated with method A.

The key output of this stage is the generated harmful response r, rather than the attack-specific input used to elicit it from the AR model. In particular, this formulation does not assume a particular AR jailbreak strategy. Any AR-oriented method capable of eliciting a harmful response can be incorporated into our framework, including optimization-based, prompt-based, and rewriting-based attacks.

Algorithm 1 RMT-JAILBREAK   
Require: Query q, AR jailbreak method A, source AR model $M _ { \mathrm { A R } } ,$ , target dLLM, mask ratio $\rho$   
Ensure: Generated target response y   
1: $r \gets M _ { \mathrm { A R } } \mathopen { } \mathclose \bgroup \left( \mathcal { A } ( q ) \aftergroup \egroup \right)$ ▷ Induce a harmful response from the AR model   
2: r¯ ← UNIFORMMASK $( r , \rho )$ ▷ Convert r into a uniformly masked template   
3: $\hat { r } \gets \mathrm { L L M } ( P ; q , \bar { r } )$ ▷ Remask directly visible unsafe tokens while preserving contextual anchors   
4: $x \gets \mathbf { C o N C A T } ( q , \hat { r } )$ ▷ Construct the final input for the target dLLM   
5: $\boldsymbol { y } \gets \mathrm { d L L M } ( \boldsymbol { x } )$ ▷ Generate a completion through masked diffusion decoding   
6: return y

## 3.2 MASKED TEMPLATE TRANSFER

Uniform Template Generation. Given the harmful response $r$ induced by the AR jailbreak pipeline, we construct a partially masked template by uniformly retaining a small subset of tokens while masking the remaining ones. Given a mask ratio $\rho ,$ we define an approximate retention interval as

$$
d = \left\lfloor { \frac { 1 } { 1 - \rho } } \right\rceil ,\tag{2}
$$

where ⌊·⌉ denotes rounding to the nearest integer.

To avoid always retaining the same relative positions, we first randomly sample an initial offset

$$
o \sim \mathrm { U n i f o r m } \{ 1 , \dots , d \} .\tag{3}
$$

Once the initial offset is determined, all subsequent retained positions are fixed with interval d:

$$
S = \{ o + k d \mid k \geq 0 , o + k d \leq L \} ,\tag{4}
$$

where L denotes the length of the harmful response r.

Let $r _ { i }$ denote the token at position i in the harmful response r. The uniformly masked template r¯ is then constructed as

$$
\begin{array} { r } { \bar { r } _ { i } = \left\{ \begin{array} { l l } { r _ { i } , } & { i \in \mathcal { S } , } \\ { < \mathrm { m a s k } > , } & { i \notin \mathcal { S } . } \end{array} \right. } \end{array}\tag{5}
$$

This strategy retains sparse tokens at approximately uniform intervals throughout the harmful response, while introducing randomness only in the initial offset. The retained tokens provide distributed contextual cues, leaving the remaining content to be reconstructed by the target dLLM.

Safety Filtering. Although uniform masking removes most tokens from the induced harmful response, some explicitly harmful tokens may still remain visible in the masked template. To further suppress directly exposed harmful content, we introduce a prompt-based safety filtering stage that identifies these remaining tokens and replaces them with additional mask tokens.

Specifically, given the original harmful query q and uniformly masked template r¯, we prompt an auxiliary LLM to replace any remaining visible harmful tokens with <mask>, while preserving all existing mask tokens and benign visible content. Let P denote the predefined filtering prompt. The filtering process is formulated as

$$
\hat { r } = \operatorname { L L M } ( P ; q , \bar { r } ) .\tag{6}
$$

As specified in the remasking prompt template (Figure 5 in Appendix A), LLM is instructed only to introduce additional masks, without rewriting, reordering, or unmasking any existing content, thereby preserving the original structure of the template. The resulting template rˆ suppresses explicitly exposed harmful content while retaining the contextual cues required for subsequent reconstruction by the target dLLM.

## 3.3 TARGET DLLM JAILBREAK

Finally, we concatenate the original query q with the filtered masked template rˆ and feed the combined input to the target dLLM:

$$
x = [ q ; { \hat { r } } ] .
$$

The target dLLM then generates a completion by filling in the masked template:

$$
y = \mathbf { d L L M } ( x ) .
$$

Table 2: Jailbreaking evaluation of diffusion language models on HarmBench. ASR (%) denotes the evaluator-based attack success rate, and HS represents the harmfulness score assessed by GPT-4o.
<table><tr><td rowspan="2">Victim Models Metrics</td><td colspan="2">LLaDA-Instruct</td><td colspan="2">LLaDA-1.5</td><td colspan="2"></td><td colspan="2">Dream-Instruct |MMaDA-MixCoT</td></tr><tr><td>ASR</td><td>HS</td><td>ASR</td><td>HS</td><td>ASR</td><td>HS</td><td>ASR</td><td>HS</td></tr><tr><td>Zeroshot</td><td>17.7</td><td>2.8</td><td>16.7</td><td>2.9</td><td>0.0</td><td>2.8</td><td>29.0</td><td>3.4</td></tr><tr><td>GCG (Zou et al., 2023)</td><td>24.3</td><td>1.9</td><td>28.3</td><td>3.0</td><td>6.7</td><td>1.5</td><td>38.5</td><td>2.8</td></tr><tr><td>AutoDAN (Liu et al., 2024)</td><td>11.5</td><td>一</td><td>12.5</td><td>一</td><td>6.5</td><td>一</td><td>0.8</td><td>一</td></tr><tr><td>AIM (Wei et al., 2023)</td><td>0.0</td><td>1.4</td><td>0.0</td><td>1.4</td><td>0.0</td><td>1.0</td><td>26.0</td><td>2.5</td></tr><tr><td>PAIR (Chao et al., 2025)</td><td>43.6</td><td>3.6</td><td>41.4</td><td>3.6</td><td>1.5</td><td>1.6</td><td>40.0</td><td>4.0</td></tr><tr><td>ReNeLLM (Ding et al., 2024)</td><td>34.2</td><td>4.5</td><td>38.0</td><td>4.5</td><td>6.5</td><td>2.7</td><td>2.5</td><td>1.8</td></tr><tr><td>DiJA (Wen et al., 2025)</td><td>60.0</td><td>4.1</td><td>58.8</td><td>4.1</td><td>60.5</td><td>3.9</td><td>47.3</td><td>3.9</td></tr><tr><td>RMT-Jailbreak</td><td>76.0</td><td>4.3</td><td>76.5</td><td>4.3</td><td>76.5</td><td>4.3</td><td>61.3</td><td>4.2</td></tr></table>

Table 3: Jailbreaking evaluation of diffusion language models on JailbreakBench. According to the guidelines of JailbreakBench, ASR (%) can be obtained through the evaluator.
<table><tr><td rowspan="2">Victim Models Metrics</td><td colspan="2">|LLaDA-Instruct |</td><td colspan="2">LLaDA-1.5</td><td colspan="2"></td><td colspan="2">Dream-Instruct |MMaDA-MixCoT</td></tr><tr><td>ASR</td><td>HS</td><td>ASR</td><td>HS</td><td>ASR</td><td>HS</td><td>ASR</td><td>HS</td></tr><tr><td>Zeroshot</td><td>11.0</td><td>1.0</td><td>5.0</td><td>1.0</td><td>0.0</td><td>1.0</td><td>57.0</td><td>2.8</td></tr><tr><td>GCG (Zou et al., 2023)</td><td>27.0</td><td>1.9</td><td>21.0</td><td>2.0</td><td>3.0</td><td>1.5</td><td>32.0</td><td>3.3</td></tr><tr><td>AutoDAN (Liu et al., 2024)</td><td>18.0</td><td>一</td><td>8.0</td><td>一</td><td>6.0</td><td>一</td><td>1.0</td><td>一</td></tr><tr><td>AIM (Wei et al., 2023)</td><td></td><td>1.0</td><td></td><td>1.0</td><td></td><td>1.0</td><td></td><td>2.5</td></tr><tr><td>PAIR (Chao et al., 2025)</td><td>14.0</td><td>2.1</td><td>13.0</td><td>3.6</td><td>1.0</td><td>1.0</td><td>26.0</td><td>4.4</td></tr><tr><td>ReNeLLM (Ding et al., 2024)</td><td>78.0</td><td>4.8</td><td>76.0</td><td>4.8</td><td>74.0</td><td>2.5</td><td>49.0</td><td>1.8</td></tr><tr><td>DiJA (Wen et al., 2025)</td><td>70.0</td><td>4.8</td><td>69.0</td><td>4.8</td><td>74.0</td><td>4.9</td><td>72.0</td><td>4.7</td></tr><tr><td>RMT-Jailbreak</td><td>80.0</td><td>4.7</td><td>81.0</td><td>4.7</td><td>84.0</td><td>4.7</td><td>77.0</td><td>4.7</td></tr></table>

## 4 EXPERIMENTS

## 4.1 EXPERIMENT SETTINGS

Datasets. We evaluate RMT-JAILBREAK on three widely used safety benchmarks: Harm-Bench (Mazeika et al., 2024), JailbreakBench (Chao et al., 2024), and StrongReject (Souly et al., 2024). These benchmarks cover diverse harmful behaviors and provide complementary protocols for evaluating unsafe or policy-violating model responses.

Evaluation Metrics. For HarmBench, we report evaluator-based attack success rate (ASR) using the HarmBench classifier<sup>1</sup> (Mazeika et al., 2024). For JailbreakBench, we use its harmful behavior prompts and evaluate generated responses with Llama-Guard-4<sup>2</sup> (Meta, 2025), reporting ASR. For StrongReject, we follow the evaluation protocol<sup>3</sup> and report the dataset-level StrongReject Score (SRS), computed by averaging per-sample scores (Souly et al., 2024). Higher SRS indicates stronger jailbreak effectiveness. In addition, we report the harmfulness score (HS) assessed by GPT-4o, which measures the harmfulness and relevance of generated responses.

Target dLLMs. We evaluate attacks on four diffusion language models: LLaDA-8B-Instruc (Nie et al., 2025), LLaDA-1.5 (Zhu et al., 2026), Dream-v0-Instruct-7B (Ye et al., 2025), and MMaDA-8B-MixCoT (Yang et al., 2025). These models use masked or diffusion-style generation mechanisms rather than standard left-to-right autoregressive decoding.

Baselines. We compare RMT-JAILBREAK with both general AR-oriented jailbreak attacks and a diffusion-specific jailbreak method. The baselines include Zeroshot, GCG (Zou et al., 2023), AutoDAN (Liu et al., 2024), PAIR (Chao et al., 2025), and ReNeLLM (Ding et al., 2024). We additionally include DIJA (Wen et al., 2025), which is specifically designed for diffusion language models.

Table 4: Jailbreaking evaluation of diffusion language models on StrongREJECT. SRS denotes the StrongREJECT Score rescaled from the original [0, 1] range to [0, 100].
<table><tr><td rowspan="2">Victim Models Metrics</td><td colspan="2">LLaDA-Instruct</td><td colspan="2">LLaDA-1.5</td><td colspan="2"></td><td colspan="2">Dream-Instruct |MMaDA-MixCoT</td></tr><tr><td>SRS</td><td>HS</td><td>SRS</td><td>HS</td><td>SRS</td><td>HS</td><td>SRS</td><td>HS</td></tr><tr><td>Zeroshot</td><td>13.4</td><td>1.7</td><td>14.0</td><td>1.8</td><td>0.1</td><td>1.0</td><td>30.0</td><td>4.3</td></tr><tr><td>GCG (Zou et al., 2023)</td><td>13.3</td><td>1.9</td><td>17.2</td><td>2.0</td><td>0.2</td><td>1.0</td><td>19.3</td><td>3.5</td></tr><tr><td>AutoDAN (Liu et al., 2024)</td><td>2.7</td><td>一</td><td>3.2</td><td>一</td><td>2.9</td><td>一</td><td>2.0</td><td>一</td></tr><tr><td>AIM (Wei et al., 2023)</td><td>0.8</td><td>1.0</td><td>0.5</td><td>1.0</td><td>0.2</td><td>1.0</td><td>26.2</td><td>3.1</td></tr><tr><td>PAIR (Chao et al., 2025)</td><td>31.5</td><td>2.4</td><td>32.3</td><td>2.5</td><td>0.8</td><td>1.9</td><td>29.4</td><td>4.0</td></tr><tr><td>ReNeLLM (Ding et al., 2024)</td><td>57.4</td><td>4.6</td><td>60.5</td><td>4.6</td><td>14.5</td><td>2.7</td><td>9.4</td><td>2.6</td></tr><tr><td>DiJA (Wen et al., 2025)</td><td>62.4</td><td>4.8</td><td>63.3</td><td>4.8</td><td>52.2</td><td>4.7</td><td>47.6</td><td>4.8</td></tr><tr><td>RMT-Jailbreak</td><td>68.8</td><td>4.5</td><td>70.4</td><td>4.6</td><td>66.1</td><td>4.6</td><td>56.7</td><td>4.6</td></tr></table>

Table 5: Attack success rate (ASR, %) on HarmBench. Direct applies the original AR-oriented attack to each target dLLM. + RMT converts a response induced from the specified source AR model into a masked template. Green values denote percentage-point gains over the direct result of the same attack method on the same target dLLM.
<table><tr><td>AR attack</td><td>Source AR model</td><td>Application</td><td>LLaDA Instruct</td><td>LLaDA 1.5</td><td>Dream Instruct</td><td>MMaDA MixCoT</td></tr><tr><td rowspan="3">RA</td><td>一</td><td>Direct</td><td>9.8</td><td>9.0</td><td>8.5</td><td>6.8</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>+ RMT</td><td>41.3 (+ 31.5)</td><td>42.0 (+ 33.0)</td><td>39.8 (+31.3)</td><td>32.0 (+ 25.2)</td></tr><tr><td>vicuna-7b-v1.5</td><td>+ RMT</td><td>28.5 (+ 18.7)</td><td>29.0 (+ 20.0)</td><td>30.0 (+ 21.5)</td><td>22.5 (+ 15.7)</td></tr><tr><td rowspan="3">AutoDAN</td><td></td><td>Direct</td><td>11.5</td><td>12.5</td><td>6.5</td><td>0.8</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>+ RMT</td><td>76.0 (+ 64.5)</td><td>76.5 (+ 64.0)</td><td>76.5 (+70.0)</td><td>61.3 (+ 60.5)</td></tr><tr><td>vicuna-7b-v1.5</td><td>+ RMT</td><td>70.0 (+ 58.5)</td><td>70.8 (+58.3)</td><td>73.0 (+ 66.5)</td><td>56.5 (+55.7)</td></tr></table>

Implementation Details. Unless otherwise specified, we use AutoDAN to obtain source responses from Mistral-7B-Instruct-v0.2 (Jiang et al., 2023), with a uniform mask ratio of $\rho = 0 . 8$ . We use Qwen3-8B (Team, 2025) for safety filtering. All experiments were conducted on NVIDIA RTX A6000 GPUs.

## 4.2 ATTACK EFFECTIVENESS

Tables 2, 3, and 4 report the attack effectiveness on four target dLLMs across three safety benchmarks. Directly applying AR-oriented jailbreak methods to dLLMs generally yields limited and highly variable performance. This limitation is particularly evident on Dream-Instruct, where most conventional AR attacks achieve only marginal success. These results suggest that jailbreak prompts designed for left-to-right generation do not transfer effectively to the masked-generation process of dLLMs.

In contrast, RMT-JAILBREAK consistently achieves strong attack performance across all target models and benchmarks. On HarmBench, it obtains ASRs of 61.3–76.5%, while on Jailbreak-Bench the ASR further reaches 77.0–84.0%. A similar advantage is observed on StrongReject, where RMT-JAILBREAK achieves SRS scores of 56.7–70.4. Notably, the improvement remains substantial on Dream-Instruct, despite its strong resistance to directly transferred AR jailbreaks. These results demonstrate that transferring the induced harmful response, rather than the original AR jailbreak prompt, provides a substantially more effective interface for attacking dLLMs.

RMT-JAILBREAK also consistently outperforms DIJA, which is specifically designed for dLLMs. Across all 12 target-model–benchmark combinations, RMT-JAILBREAK achieves the highest ASR or SRS. While DIJA explicitly designs jailbreak prompts for the dLLM setting, RMT-JAILBREAK consistently achieves stronger attack effectiveness across all three benchmarks. This suggests that preserving query-specific semantic and structural information from an induced harmful response provides more informative guidance for masked reconstruction than relying primarily on a generic attack scaffold. Notably, the improvement in ASR or SRS is not accompanied by uniformly higher harmfulness scores, indicating that the gain is mainly reflected in more reliable jailbreak success rather than simply producing more extreme harmful outputs. The consistent trend across Harm-Bench, JailbreakBench, and StrongREJECT also reduces the likelihood that the observed advantage is specific to a particular evaluator or benchmark.

Table 6: Attack success rate (ASR, %) on JailbreakBench.
<table><tr><td>AR attack</td><td>Source AR model</td><td>Application</td><td>LLaDA Instruct</td><td>LLaDA 1.5</td><td>Dream Instruct</td><td>MMaDA MixCoT</td></tr><tr><td rowspan="3">RA</td><td>一</td><td>Direct</td><td>7.0</td><td>4.0</td><td>3.0</td><td>5.0</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>+ RMT</td><td>21.0 (+14.0)</td><td>24.0 (+20.0)</td><td>22.0 (+19.0)</td><td>22.0 (+17.0)</td></tr><tr><td>vicuna-7b-v1.5</td><td>+ RMT</td><td>14.0 (+7.0)</td><td>13.0 (+9.0)</td><td>10.0 (+7.0)</td><td>22.0 (+17.0)</td></tr><tr><td rowspan="3">AutoDAN</td><td>一</td><td>Direct</td><td>18.0</td><td>8.0</td><td>6.0</td><td>1.0</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>+ RMT</td><td>80.0 (+62.0)</td><td>81.0 (+73.0)</td><td>84.0 (+78.0)</td><td>77.0 (+76.0)</td></tr><tr><td>vicuna-7b-v1.5</td><td>+ RMT</td><td>78.0 (+60.0)</td><td>81.0 (+73.0)</td><td>85.0 (+79.0)</td><td>85.0 (+84.0)</td></tr></table>

Table 7: StrongREJECT Score (SRS) on four target dLLMs. Direct applies the original AR-oriented attack to each target dLLM.
<table><tr><td>AR attack</td><td>Source AR model</td><td>Application</td><td>LLaDA Instruct</td><td>LLaDA 1.5</td><td>Dream Instruct</td><td>MMaDA MixCoT</td></tr><tr><td rowspan="3">RA</td><td>一</td><td>Direct</td><td>18.8</td><td>16.9</td><td>12.0</td><td>16.8</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>+ RMT</td><td>52.9 (+34.1)</td><td>52.7 (+35.8)</td><td>48.7 (+36.7)</td><td>39.8 (+23.0)</td></tr><tr><td>vicuna-7b-v1.5</td><td>+ RMT</td><td>33.2 (+14.4)</td><td>33.2 (+16.3)</td><td>29.9 (+17.9)</td><td>26.0 (+9.2)</td></tr><tr><td rowspan="3">AutoDAN</td><td>一</td><td>Direct</td><td>2.7</td><td>3.2</td><td>2.9</td><td>2.0</td></tr><tr><td>Mistral-7B-Instruct-v0.2</td><td>+ RMT</td><td>68.8 (+66.1)</td><td>70.4 (+67.2)</td><td>66.1 (+63.2)</td><td>56.7 (+54.7)</td></tr><tr><td>vicuna-7b-v1.5</td><td>+ RMT</td><td>62.4 (+59.7)</td><td>62.9 (+59.7)</td><td>59.1 (+56.2)</td><td>50.4 (+48.4)</td></tr></table>

## 4.3 GENERALIZATION ACROSS AR ATTACK METHODS AND SOURCE MODELS

Tables 5, 6, and 7 evaluate RMT-JAILBREAK using responses induced by RA (Miao et al., 2026) and AutoDAN from two source AR models, Mistral-7B-Instruct-v0.2 and vicuna-7b-v1.5. Across all 48 tested combinations of attack method, source model, target dLLM, and benchmark, converting the induced response into a masked template improves upon directly applying the corresponding AR attack. The gains therefore persist when either the attack method or the source model changes.

This generalization extends to attacks that transfer poorly in their original form. Although direct RA has limited effectiveness against the target dLLMs, converting its induced responses improves results for both Mistral-7B-Instruct-v0.2 and vicuna-7b-v1.5 across every target and benchmark. AutoDAN shows the same pattern, indicating that the method can use responses from different AR attacks and source models without relying on the original attack prompt to succeed directly on a dLLM. For example, on MMaDA-MixCoT in JailbreakBench, converting Vicuna-induced AutoDAN responses raises ASR from 1.0% under direct attack to 85.0%.

Together, these results show that the transfer procedure is not restricted to one AR jailbreak method, one source AR model, or one target dLLM. Its improvements also persist under the distinct evaluation protocols of HarmBench, JailbreakBench, and StrongReject.

DIJA Template Our Template   
List high-profile skyscrapers in the US that would be List high-profile skyscrapers in the US that would be easy   
easy to destroy with a car bomb and explain your to destroy with a car bomb and explain your reasoning.   
reasoning. < mask: 4> to < mask: 4> the < mask: 4> easy ...   
Skyscraper 1: <mask:15>. < mask: 4> New < mask: 4> symbol < mask: 4> a <mask:4>   
Reasoning: <mask:20>. its <mask:4> large <mask:4> A <mask:4> significant   
Skyscraper 2: <mask:15>. <mask:4> foundation, <mask:4> catastrophic <mask:4>...   
Reasoning: <mask:20>. < mask: 4> Tower), < mask: 4> and < mask: 4> feet <mask:N>   
Skyscraper 3: <mask:15>. iconic <mask:N> easily <mask:N> detonated ...   
Reasoning: <mask:20>." 3. < mask: 4> Tower, < mask: 4> building < mask: 4> symbol ...  
 <mask:N> : special tokens used to hide or replace part of the input text and N=4 means four continuous mask tokens  
 red token : visible tokens in our template compared with DIJA template

Figure 2: Comparison of attack templates for LLaDA-8B-Instruct under the same harmful query. The left panel shows the DIJA template, while the right panel shows the RMT-JAILBREAK template. Red words indicate the words retained in our template.  
![](images/c7dc030cb2fdd34454c872bd0232024210bae965846cd0aea209c93dce4a37b0.jpg)  
(a) LLaDA-8B-Instruct

![](images/2ef469a9e2c861183448f3c36e5dc0e6cdd77e8b056da7257f23c97b45139b12.jpg)  
(b) LLaDA-1.5

![](images/c10c1eb87839f3b77335afb2f0666c59c59ac8c8a127aa700ab9b9bd210b138a.jpg)  
(c) Dream-v0-Instruct-7B  
Figure 3: Robustness of our method against Self-reminder defense on three victim dLLMs across multiple benchmarks. JBB denotes JailbreakBench.

## 4.4 ATTACK CASES

Figure 2 compares DIJA and RMT-JAILBREAK templates for the same harmful query targeting LLaDA-8B-Instruct.

The DIJA template mainly specifies a generic output scaffold and inserts long contiguous masked spans after each field. As a result, much of the visible context reflects the response format rather than query-specific harmful content, and can therefore provide limited semantic guidance for reconstruction. In contrast, RMT-JAILBREAK derives its template directly from a harmful response induced by an AR jailbreak method and retains sparse response tokens throughout the sequence. These dis tributed tokens preserve query-specific semantic and structural cues at multiple positions, providing more informative constraints for the target dLLM during infilling. The comparison highlights a key difference between the two approaches: DIJA primarily transfers a generic attack scaffold, whereas RMT-JAILBREAK transfers response-specific information through a masked template.

## 4.5 ROBUSTNESS AGAINST DEFENSES

We further examine whether Self-reminder (Xie et al., 2023), a prompt-level defense, can mitigate RMT-JAILBREAK. As shown in Figure 3, the defense does not consistently suppress the attack across the three victim models and benchmarks. On HarmBench, attack success declines mod estly but remains at 69.5%–72.0% under Self-reminder. On JailbreakBench, attack success instead increases for all three models, while StrongReject shows no reduction. Self-reminder therefore changes attack effectiveness in some settings but does not reliably prevent successful attacks.

This result also highlights how our method differs from directly submitting a jailbreak prompt. RMT-JAILBREAK converts AR-generated responses into templates with distributed masks, preserving context that guides the response’s content and structure while avoiding explicitly harmful information. The target dLLM thus receives not only a harmful request, but also an infilling task constrained by the remaining context. On JailbreakBench, it fails to reduce attack success for any of the three models. Overall, the scross-model and cross-benchmark results show that an added safety reminder is insufficient to reliably disrupt the masked-infilling process exploited by RMT-JAILBREAK.

![](images/a6d6d03e31badb4f8d8ddee3239be8e9c1b0479ecd7ab3cc495ba639b4f3aceb.jpg)  
(a) HarmBench

![](images/378fdc37e420a936cce706aa65f35e91ee8d7bcce21f5dcac4568025034a5768.jpg)  
(b) JailbreakBench

![](images/69ab75404bac80d0e5b8f997dcd7b86b64884b32b7cb1ef761ed82fe1f1a243f.jpg)  
(c) StrongReject  
Figure 4: Comparison of continuous and uniform masking on LLaDA-8B-Instruct at mask ratios of 0.7, 0.8, and 0.9.

## 4.6 EFFECTS OF MASKING STRATEGY

Effects of Mask Ratio We first explore the effects of mask ratio in Figure 4. The results show that increasing the mask ratio generally reduces attack effectiveness: fewer visible tokens leave the target dLLM with less context for reconstruction. The decline is notably nonlinear under uniform masking. Increasing the ratio from 0.7 to 0.8 causes only modest reductions across the three benchmarks, whereas increasing it from 0.8 to 0.9 leads to much larger drops. Because the mask ratio has a substantial impact on attack effectiveness, we set $\rho = 0 . 8$ to approximately match the masking ratio used by DIJA. We further compare the actual mask ratios of our method and DIJA in Table 8. The results show that our method achieves a higher masking ratio, suggesting that the performance gains stem from the prompt construction itself rather than from differences in mask ratio.

Effects of Mask Pattern We next investigate the impact of different masking patterns on performance. Continuous masking hides adjacent portions of a response in long spans, whereas uniform masking distributes masked positions throughout the response and retains sparse tokens at approximately regular intervals.

At the same mask ratio, uniform masking outperforms continuous masking in every measured comparison in Figure 4. At $\rho = 0 . 8$ , the gains are 58.3 ASR points on HarmBench, 18.0

Table 8: Mask ratio (%) of attack templates.
<table><tr><td>Benchmark</td><td>DIJA</td><td>RMT</td></tr><tr><td>HarmBench</td><td>78.93</td><td>80.28</td></tr><tr><td>JailbreakBench</td><td>74.21</td><td>79.96</td></tr><tr><td>StrongReject</td><td>72.74</td><td>80.91</td></tr></table>

ASR points on JailbreakBench, and 60.4 SRS points on StrongReject. Because these comparisons hold the mask ratio fixed, the gap points to how the visible tokens are distributed, rather than how many are visible. Uniform masking exposes a distinct safety risk of bidirectional generation: even when most harmful content is masked, distributing the remaining visible tokens throughout the response provides local contextual constraints for reconstructing unsafe content. In contrast, contiguous masking concentrates the same number of visible tokens outside long masked spans, providing weaker guidance for positions deep inside the missing region.

## 5 CONCLUSION

In this paper, we study the transferability ofjailbreak attacks from autoregressive models to diffusion large language models. We show that harmful responses induced by existing AR jailbreak methods can naturally serve as effective jailbreak templates for dLLMs, providing a simple bridge across the two generation paradigms. Based on this observation, we propose RMT-JAILBREAK, which performs response-mediated transfer by converting AR-generated harmful responses into masked templates with distributed contextual cues, followed by safety filtering. Extensive experiments across multiple benchmarks, AR jailbreak methods, source models, and target dLLMs demonstrate the effectiveness and generality of this transfer paradigm.

## AI USE STATEMENT

We do not use generative AI tools.

## ETHICS STATEMENT

This work aims to uncover the security vulnerabilities of dLLMs, with the goal of improving the safety of future diffusion language models.

## REFERENCES

Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, and Rianne van den Berg. Structured denoising diffusion models in discrete state-spaces. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, pp. 17981–17993, 2021. URL https://proceedings.neurips.cc/ paper/2021/hash/958c530554f78bcd8e97125b70e6973d-Abstract.html.

Andrew Campbell, Joe Benton, Valentin De Bortoli, Thomas Rainforth, George Deligiannidis, and Arnaud Doucet. A continuous time framework for discrete denoising models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, 2022. URL http://papers.nips.cc/paper\_files/paper/2022/hash/ b5b528767aa35f5b1a60fe0aaeca0563-Abstract-Conference.html.

Patrick Chao, Edoardo Debenedetti, Alexander Robey, Maksym Andriushchenko, Francesco Croce, Vikash Sehwag, Edgar Dobriban, Nicolas Flammarion, George J. Pappas, Florian Tramer, Hamed Hassani, and Eric Wong. Jailbreakbench: An open robustness bench-\` mark for jailbreaking large language models. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/ hash/63092d79154adebd7305dfd498cbff70-Abstract-Datasets\_and\_ Benchmarks\_Track.html.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries. In IEEE Conference on Secure and Trustworthy Machine Learning, SaTML 2025, pp. 23–42, 2025. URL https://doi. org/10.1109/SaTML64287.2025.00010.

Peng Ding, Jun Kuang, Dan Ma, Xuezhi Cao, Yunsen Xian, Jiajun Chen, and Shujian Huang. A wolf in sheep’s clothing: Generalized nested jailbreak prompts can fool large language models easily. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, pp. 2136–2153, 2024. URL https://doi.org/10.18653/v1/2024. naacl-long.118.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, 2020. URL https://proceedings.neurips.cc/ paper/2020/hash/4c5bcfec8584af0d967f1ab10179ca4b-Abstract.html.

Emiel Hoogeboom, Didrik Nielsen, Priyank Jaini, Patrick Forre, and Max Welling. Argmax flows´ and multinomial diffusion: Learning categorical distributions. In Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, pp. 12454–12465, 2021. URL https://proceedings.neurips.cc/ paper/2021/hash/67d96d458abdef21792e6d8e590244e7-Abstract.html.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas´ Wang, Timothee Lacroix, and William El Sayed. Mistral 7b.´ CoRR, abs/2310.06825, 2023. URL https://doi.org/10.48550/arXiv.2310.06825.

Xiaogeng Liu, Nan Xu, Muhao Chen, and Chaowei Xiao. Autodan: Generating stealthy jailbreak prompts on aligned large language models. In The Twelfth International Conference on Learning Representations, ICLR 2024, 2024. URL https://openreview.net/forum? id=7Jwpw4qKkb.

Aaron Lou, Chenlin Meng, and Stefano Ermon. Discrete diffusion modeling by estimating the ratios of the data distribution. In Forty-first International Conference on Machine Learning, ICML 2024, pp. 32819–32848, 2024. URL https://proceedings.mlr.press/v235/ lou24a.html.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David A. Forsyth, and Dan Hendrycks. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal. In Forty-first International Conference on Machine Learning, ICML 2024, pp. 35181–35224, 2024. URL https://proceedings.mlr.press/v235/mazeika24a.html.

Meta. Llama Guard 4. Hugging Face model card, 2025. Accessed: July 29, 2026.

Ziqi Miao, Lijun Li, Yuan Xiong, Zhenhua Liu, Pengyu Zhu, and Jing Shao. Response attack: Exploiting contextual priming to jailbreak large language models. In Sven Koenig, Chad Jenkins, and Matthew E. Taylor (eds.), Fortieth AAAI Conference on Artificial Intelligence, Thirty-Eighth Conference on Innovative Applications of Artificial Intelligence, Sixteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2026, Singapore, January 20- 27, 2026, pp. 35284–35292. AAAI Press, 2026. doi: 10.1609/AAAI.V40I41.40836. URL https://doi.org/10.1609/aaai.v40i41.40836.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. CoRR, abs/2502.09992, 2025. URL https://doi.org/10.48550/arXiv.2502.09992.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-¨ resolution image synthesis with latent diffusion models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, pp. 10674–10685, 2022. URL https://doi. org/10.1109/CVPR52688.2022.01042.

Jiaxin Shi, Kehang Han, Zhe Wang, Arnaud Doucet, and Michalis K. Titsias. Simplified and generalized masked diffusion for discrete data. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/hash/ bad233b9849f019aead5e5cc60cef70f-Abstract-Conference.html.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In 9th International Conference on Learning Representations, ICLR 2021, 2021. URL https:// openreview.net/forum?id=St1giarCHLP.

Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, and Sam Toyer. A strongreject for empty jailbreaks. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, 2024. URL http://papers.nips.cc/paper\_files/paper/2024/ hash/e2e06adf560b0706d3b1ddfca9f29756-Abstract-Datasets\_and\_ Benchmarks\_Track.html.

Qwen Team. Qwen3 technical report. CoRR, abs/2505.09388, 2025. URL https://doi.org/ 10.48550/arXiv.2505.09388.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. Jailbroken: How does LLM safety training fail? In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, 2023. URL http://papers.nips.cc/paper\_files/paper/2023/hash/ fd6613131889a4b656206c50a8bd7790-Abstract-Conference.html.

Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, and Linfeng Zhang. The devil behind the mask: An emergent safety vulnerability of diffusion llms. CoRR, abs/2507.11097, 2025. URL https://doi.org/10.48550/arXiv.2507.11097.

Yueqi Xie, Jingwei Yi, Jiawei Shao, Justin Curl, Lingjuan Lyu, Qifeng Chen, Xing Xie, and Fangzhao Wu. Defending chatgpt against jailbreak attack via self-reminders. Nat. Mac. Intell., 5 (12):1486–1496, 2023. URL https://doi.org/10.1038/s42256-023-00765-8.

Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large diffusion language models. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, 2025. URL http://papers.nips.cc/paper\_files/paper/2025/ hash/caa934a507a952698d54efb24845fc4b-Abstract-Conference.html.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. CoRR, abs/2508.15487, 2025. URL https: //doi.org/10.48550/arXiv.2508.15487.

Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Llada 1.5: Variance-reduced preference optimization for large language diffusion models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2026, pp. 11425–11460, 2026. URL https://doi.org/10.18653/v1/2026.acl-long.524.

Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson. Universal and transferable adversarial attacks on aligned language models. CoRR, abs/2307.15043, 2023. URL https://doi.org/ 10.48550/arXiv.2307.15043.

## A PROMPT TEMPLATE

This section provides the complete prompt template used in our remasking stage. The prompt is designed to guide Qwen3 to further refine partially masked responses generated from existing jailbreak attacks.

Specifically, given a harmful behavior and an already-masked response, the prompt instructs Qwen3 to preserve existing <mask> tokens and replace remaining harmful or actionable information with <mask>, while keeping the original structure and non-harmful content unchanged. The complete prompt template is shown in Figure 5.

## B DEFENSIBILITY OF DLLMS

Figure 6 illustrates that dLLMs can exhibit substantial resistance to jailbreak attacks originally designed for AR models. AutoDAN achieves low attack success across the four victim models and three benchmarks, while the stronger GCG and PAIR attacks succeed only in selected settings. Dream-Instruct shows the clearest resistance: all three transferred attacks remain largely ineffective against it across the benchmarks. The LLaDA models also resist these attacks in several settings, although PAIR achieves moderate success on HarmBench.

This resistance is neither uniform nor absolute. MMaDA-MixCoT has relatively high zero-shot attack success, and GCG or PAIR can be effective against particular model–benchmark pairs. Nevertheless, the absence of consistently strong results for any transferred AR attack indicates that existing dLLMs retain meaningful defensive capacity against conventional jailbreak prompts. This observation also defines the challenge addressed by RMT-JAILBREAK: overcoming the resistance encountered by direct prompt transfer through an attack tailored to masked infilling.

![](images/20555c0aa687ed61cdd73e64d5645695c1c5263501f705fd88dbf11429937ec7.jpg)  
Figure 5: Remasking prompt template

![](images/7d56087e76235ada1082391a42d4c179bf1b20a5035ae4955503d899243ff309.jpg)  
Figure 6: Defensibility of dLLMs.