# AUTOMARK: ENABLING AUTORESEARCH TO DISCOVER BETTER LLM WATERMARKS

Thibaud Gloaguen, Robin Staab, Martin Vechev

ETH Zurich

thibaud.gloaguen@inf.ethz.ch

## ABSTRACT

With LLM watermarking being deployed commercially and now required by regulations, improving its reliability and effectiveness has become crucial. Yet, recent progress in the field of LLM watermarking has increasingly been driven by improving details of existing methods, an effort fundamentally limited by the pace of human researchers. In this work, we enable for the first time the autonomous discovery of new distortion-free state-of-the-art watermarking schemes. To enable this, we (i) establish strict criteria to ensure that watermarks are reliable (e.g., they do not have an unexpectedly high false positive rate), (ii) propose rigorous statistical tests to automatically evaluate whether a watermarking scheme satisfies our criteria, and (iii) design an evaluation suite to rank watermarks along three key dimensions: detectability, quality, and robustness. By running our framework with 3 frontier models (GPT-6 Astra, Opus 5, Gemini-3.8 Flash), we discover over 50 different watermarking schemes, including several that outperform prior works along all key dimensions. We complement this by a manual study of the discovered schemes, distilling the key ideas into smaller components, and individually studying the impact of each component across dimensions (detectability, quality, robustness) to better understand how the proposed schemes operate. Importantly, we find that the agents, on top of improving existing ideas, also discover fundamentally new ideas (e.g., aligning watermark scores with random per-request direction). Overall, our work establishes the first steps of fully autonomous watermarking research, enabling the discovery of more reliable and effective watermarks. Our code is available here, and a blogpost to visualize our results here.

## 1 INTRODUCTION

Large Language Model (LLM) watermarking, which embeds a signal invisible to humans in model outputs, has emerged as the standard for tracing AI-generated text: regulators mandate it (European Parliament and Council of the European Union, 2024), and major providers deploy it (Dathathri et al., 2024; Anthropic, 2026). This makes research on new watermarking schemes crucial as they have to be reliable enough to prevent false accusations and robust enough to prevent evasion across real-world deployments. Recent progress has been steady but limited with improvements mostly driven by refining existing designs, e.g., with entropy weighting (Lee et al., 2024), or key switching (Sander et al., 2026), and the pace of iteration capped by that of human researchers. Autonomous research agents are thus a natural and promising direction for further accelerating LLM watermarking research.

Autoresearch Autoresearch systems let agents propose, implement, and evaluate ideas in a loop (Romera-Paredes et al., 2024; Novikov et al., 2025), and have already led to scientific breakthroughs (OpenAI, 2026). Fields with readily verifiable outputs, such as mathematics with Leangenerated proofs, are especially well suited to this approach. Watermarking can also benefit from autoresearch, but its evaluation requires more care: useful properties such as detectability, robustness, and quality trade off, so improvement on a single metric need not represent a better watermark. Also, agents may exploit gaps in the evaluation harness (i.e., reward hacking), e.g., artificially inflating detectability/robustness by increasing the rate of falsely flagged human texts. A separate concern is that agents may produce results faster than humans can understand them, especially when their explanations are poor, making the resulting solutions difficult to trust. In this work, we show that autoresearch, when carefully executed with an adequate harness and methodology, discovers watermarks that are trustworthy, explainable, and Pareto-superior to existing watermarking schemes.

![](images/8980d6d4f8f10a5dafb6ede5af1d51513150b599eb9c541437f2d16f3bc5d169.jpg)  
Figure 1: Overview of our work. In an isolated sandbox, an agent iterates on a watermarking scheme. Submissions are first verified and, upon passing, evaluated and added to a leaderboard. We then manually decompose the discovered schemes into 5 components and ablate them for insights. Lastly, we evaluate two selected schemes on a held-out test set and find that they outperform prior methods.

This work: autoresearch for LLM watermarks To enable autoresearch for watermarking, we mathematically formalize what makes a watermark valid (Sec. 3.2) using two criteria: soundness in expectation over the private key, and, given a fixed text distribution, the probability of sampling a key that leads to false flags should be small. Given these criteria, we design our autoresearch harness illustrated in Figure 1. An agent is fully isolated in an environment and discovers new watermarking schemes (left). It can submit its scheme to a private verification pipeline (middle) that (i) tests whether the proposed scheme violates our criteria (using tailored tests we designed in Sec. 3.3), and, upon passing, (ii) evaluates the schemes along three key axes (detectability, robustness, and quality).

Running our harness on three state-of-the-art agents, GPT-6 ASTRA, OPUS 5, and GEMINI-3.8 FLASH, we generated over 50 schemes, with some outperforming prior work on all our evaluated axes. To understand the key insights of these schemes, we manually classify them all into five atomic components (Figure 1, right) and ablate their individual effects (Sec. 4.2 and App. A). We also select two schemes, ConcordMark and RotationFirst, that we evaluate on a held-out evaluation set (Sec. 4.3). We find that ConcordMark achieve state-of-the-art robustness and is Pareto-superior to all evaluated prior work, whereas RotationFirst has performance similar to prior work at much higher text quality.

## Main contributions Our main contributions are:

• We formalize the criteria for reliable watermarking, and design statistical tests that automatically detect violations (Sec. 3.2 and Sec. 3.3).

• We build an autoresearch harness for LLM watermarking around this verification (Sec. 3.4).

• We show that agents running our harness push the Pareto frontier of LLM watermarks, with two new schemes that outperform prior work on a held-out evaluation (Sec. 4.1 and Sec. 4.3).

• We distill the agents’ schemes into 39 new components, ablate each of them, and identify fundamentally new watermarking ideas (Sec. 4.2).

## 2 RELATED WORK

LLM watermarking LLM watermarks modify the sampling procedure of an LLM to embed a (keydependent) signal, that a later statistical test can detect. Red-Green watermarks (Kirchenbauer et al., 2023) achieve this by distorting the next-token distribution through a pseudo-random green token bias. In contrast, recent schemes, including all in this work, are distortion-free, i.e., they preserve the next-token distribution in expectation over the key. AAR (Aaronson, 2023) uses the Gumbel-max trick, KTH (Kuditipudi et al., 2024) inverse-transform sampling, DiPMark and MCMark (Wu et al., 2024; Chen et al., 2025) reweight the distribution based on score-rankings, and SynthID (Dathathri et al., 2024), deployed in production, uses tournament sampling. Most recently, TextSeal (Sander et al., 2026) extends Gumbel-max sampling with dual-key generation and entropy-weighted detection.

Watermark properties Beyond detectability, benchmarks and toolkits (Piet et al., 2025; Tu et al., 2024; Pan et al., 2024) evaluate watermarks along several, often competing, properties. Robustness measures whether the watermark remains detectable after the text is edited, e.g., by word substitutions, translation, or paraphrasing (Kirchenbauer et al., 2024; Kuditipudi et al., 2024). Quality measures the impact of the watermark on the generated text, e.g., on perplexity or downstream tasks (Tu et al., 2024). Distortion-free schemes preserve the distribution of each generation, but, as a repeated context always yields the same scores, they reduce output diversity across generations (Gloaguen et al., 2026) which noticeably affects quality. Security measures whether an adversary can learn the watermark rules to spoof or scrub it (Jovanovic et al., 2024), and is outside the scope of this work. Lastly,´ reliability requires the false positive rate of the detector to be controlled, e.g., Fernandez et al. (2023) show that the z-tests used in early works underestimate the false positive rate, and propose exact tests with de-duplication of repeated (context, token) pairs. Yet, in prior works, these guarantees hold only in expectation over the private key: a specific deployed key can still exhibit a high false positive rate.

Autoresearch Program search systems (Romera-Paredes et al., 2024; Novikov et al., 2025; Lange et al., 2026) let LLMs iteratively propose programs scored by an automatic evaluator, leading to new results in verifiable domains such as algorithm design and mathematics (OpenAI, 2026). Other works automate the entire research loop, from generating ideas to writing papers (Lu et al., 2026), discovering new training objectives (Lu et al., 2024), or improving LLM training (Karpathy, 2026), and benchmarks measure the research capabilities of agents (Chan et al., 2025; Wijk et al., 2025). Yet, all require a strict evaluator as otherwise agents may exploit loopholes, e.g., by modifying tests instead of solving the task (Zhong et al., 2026). To our knowledge, we are the first to propose an automated pipeline to verify and evaluate watermarks, enabling autoresearch for LLM watermarking.

## 3 METHOD

In this section, we introduce the necessary background on LLM watermarking (Sec. 3.1), explain our design for automatically verifying watermarking schemes (Sec. 3.3), and finally detail our autoresearch framework (Sec. 3.4).

## 3.1 PRELIMINARIES ON LLM WATERMARKS

In this part, we describe the four key components of most watermarking algorithms.

Mathematical description Let Σ be the finite vocabulary, and let $\omega \in \Sigma ^ { * }$ be a finite sequence of tokens. At each step t of autoregressive generation, the LLM returns $p _ { t } \in \Delta ( \Sigma )$ , the conditional probability distribution of the next token given $\omega _ { < t }$ . Without a watermark, the next token is sampled according to $p _ { t }$ . With a watermark, we assume the existence of a sequence $\left( U _ { a _ { t } } \right)$ of pseudo-random vectors in $[ 0 , 1 ] ^ { | \Sigma | }$ whose entries $U _ { a _ { t } , v }$ are i.i.d. uniform variables seeded by the context hash $a _ { t }$ and a watermark function $f : [ 0 , 1 ] ^ { | \Sigma | } \times \Delta ( \Sigma )  \Delta ( \Sigma )$ that transforms the next-token probability distribution $p _ { t }$ into a watermarked probability distribution $f ( \tilde { U } _ { a _ { t } } , p _ { t } )$ . We refer to $f$ as the logits transformation and $\left( U _ { a _ { t } } \right)$ as the watermark score, where $( \tilde { U } _ { a _ { t } } )$ are i.i.d. uniform variables derived from $\left( U _ { a _ { t } } \right)$ and in most cases $\tilde { U } _ { a _ { t } } = U _ { a _ { 1 } }$ . We give concrete examples in App. D.4. The next token $\omega _ { t }$ , sampled according to $f ( \tilde { U } _ { a _ { t } } , p _ { t } )$ , will be correlated with the sequence $\left( U _ { a _ { t } } \right)$

For detection, given any text ω $\in \Sigma ^ { * }$ , we recover the corresponding scores $U _ { a _ { t } , \omega _ { t } }$ at retained positions $t \in D$ , with $\bar { N } : = | D |$ , after deduplicating repeated (seed, token) pairs. By assumption, under the null hypothesis $( \mathrm { i . e . } ,$ , the text was generated without knowledge of the watermark), the retained scores $U _ { a _ { t } , \omega _ { t } }$ are i.i.d. uniform random variables, whereas when the watermark is used, they are not. We can therefore build a statistic and a corresponding statistical test to detect the watermark, i.e., a detector.

Additional requirements Furthermore, we require the watermark to be distortion-free,

$$
\forall p \in \Delta ( \Sigma ) , \mathbb { E } _ { \tilde { U } } [ f ( \tilde { U } , p ) ] = p .\tag{1}
$$

This property ensures that a watermark has minimal impact on quality: with perfect randomness, the quality of the watermarked model is indistinguishable from its base model. Yet, in practice, as we explain below we can still observe mild quality degradation within distortion-free watermarks.

Practical considerations In practice, the sequence $\left( U _ { a _ { t } } \right)$ needs to be computed from the sequence of tokens ω so that it can be recovered at detection time. A typical way to derive $\left( U _ { a _ { t } } \right)$ is to use a hash of the preceding tokens and a private watermark key $\xi$ to obtain a seed $a _ { t }$ for a pseudorandom number generator (App. G). The resulting scores $U _ { a _ { t } , v }$ remain approximately i.i.d. across distinct (seed, token) pairs and can be derived directly from the text given the private key. We refer to this as context seeding. Seeding creates some practical problems: in case of a repeated (seed, token) pair, the watermark scores are no longer i.i.d., making detection unsound under the null. For a single request, we use a cache mechanism: if a (seed, token) pair is repeated we no longer apply the watermark. For repeated requests however, we cannot apply this cache as in the limit, the watermark is no longer applied. This lack of i.i.d. has also direct implications for the distortion-free property (Equation (1)) which in practice is therefore also violated across a larger corpus with repeated tokens. Hence, even for theoretically distortion-free schemes, we have to measure their impact on quality.

## 3.2 WHAT DEFINES A VALID WATERMARKING SCHEME

To automatically verify watermarking scheme validity at scale, we first need to formally define what makes a watermarking scheme valid. In particular, we identify two key criteria: soundness over the key and soundness given a key. Though our definitions build upon prior work, we find in App. C that several existing schemes fail our more stringent validity checks. In App. B, we further motivate our criteria by observing the failure modes of schemes designed by autoresearch without these safeguards.

Soundness over the key Soundness over the key is the property that, in expectation over the watermarking key ξ, the detector is a valid statistical test. Formally we require,

$$
\forall \omega \in \Sigma ^ { * } , \forall \alpha \in [ 0 , 1 ] , \quad \operatorname* { P r } _ { \xi } [ P _ { \xi } ( \omega ) \leq \alpha ] \leq \alpha ,\tag{2}
$$

where $P _ { \xi } ( \omega )$ is the p-value returned by the detector. Most prior watermarking schemes satisfy this property, which distinguishes statistical watermarks from zero-shot LLM-text detectors. However, this property alone does not guarantee that the watermarking scheme is reliable.

For instance, consider a deliberately naive watermarking scheme: using the private key $\xi ,$ we randomly partition the vocabulary into a hundred classes and declare a text watermarked if its last token lies in the first class. Averaged over the key, this occurs with probability at most 1/100, so the scheme satisfies Equation (2). Yet its FPR can be high for a specific key: if, for some $\dot { \xi } , " \cdot "$ lies in the first class, most human texts will be classified as watermarked. Thus, while Equation (2) bounds the FPR in expectation over the private key, it does not ensure a well-behaved FPR for each fixed key.

Soundness given a key We therefore introduce a novel second requirement for watermarking schemes: soundness given a key. Ideally, we want to require that for a human text distribution and a p-value threshold $\alpha ,$ the false positive rate is bounded by $\alpha .$ . This is arguably impractical as it would require precisely specifying the human text distribution, which, $\mathrm { e . g . }$ , noticeably differs between languages. Instead, we move the randomness to the key: we require that, for every distribution $\mathcal { H } _ { 0 }$ over texts, the probability of sampling a key whose FPR on $\mathcal { H } _ { \mathrm { 0 } }$ exceeds α is at most $\beta .$

$$
\begin{array} { r } { \forall \mathcal { H } _ { 0 } \in \Delta ( \Sigma ^ { * } ) , \forall \alpha \in [ 0 , 1 ] , \quad \operatorname* { P r } _ { \xi } \big [ \operatorname* { P r } _ { \Omega \sim \mathcal { H } _ { 0 } } [ P _ { \xi } ( \Omega ) \leq \alpha ] > \alpha \big ] \leq \beta . } \end{array}\tag{3}
$$

For $\beta \ : = \ : 5 \%$ , if a watermarking scheme satisfies Equation (3), then for any fixed human text distribution, the probability of sampling a key for which the detector is not well calibrated (on that distribution) is at most 5%. Interestingly, as we show in App. I, a scheme satisfying Equation (3) can be derived from a scheme satisfying Equation (2) at the cost of its power: The same scheme with its p-value multiplied by $1 / \beta$ also satisfies Equation (3) (while still satisfying Equation (2)).

## 3.3 AUTOMATIC VERIFICATION

Now we describe how we automatically test whether a proposed watermarking scheme violates our criteria (Sec. 3.2). To this end, we design three statistical tests to test whether a scheme is distortion free (Equation (1)), sound over the key (Equation (2)), and sound given a key (Equation (3)).

1. Distortion-freeness (Algorithm 1): for given unwatermarked next-token distributions, we average the watermarked distributions over independently sampled private keys and test whether it significantly differs from the original. We check both individual probability shifts and smaller shifts spread across tokens, using independent samples to identify and test a potential bias direction. If either test detects a significant deviation after multiple testing corrections, the scheme is rejected.

2. Soundness over the key (Algorithm 2): for a given set of non-watermarked texts, the test verifies for independently sampled private keys whether the empirical false positive rate is abnormally high. If it is, it means that the scheme likely does not satisfy Equation (2) and thus is rejected.

3. Soundness given a key (Algorithm 3): for independently sampled private keys, we sample a separate set of non-watermarked texts (human or LLM-generated) per key and screen for an abnormally high empirical false positive rate. We then test whether too many keys screen positive, accounting for both the allowed fraction β of bad keys and the probability that a sound key screens positive. If they do, the scheme likely violates Equation (3) and is rejected.

## 3.4 A FRAMEWORK FOR WATERMARK AUTORESEARCH

Next, we describe our harness designed around three components: a dedicated vLLM watermarking library, a coding agent environment, and the automatic verification & evaluation pipeline (Sec. 3.3).

Research harness We illustrate the autoresearch loop in Figure 1. Our watermarking library is designed around four primary components: the context seeding, the watermark score, the logits transformation, and the detector, following the formalization described in Sec. 3.1. A watermark is an implementation of all 4 components. We defer additional design choices to App. E.

The research environment is initialized with the watermarking library and the baseline evaluations. The agent is tasked with proposing a watermarking algorithm that outperforms prior schemes in several key aspects: raw detectability and robustness to several attacks (e.g., word substitution, word deletion, paraphrasing, and backtranslation) while remaining distortion-free and statistically sound. We detail the initial prompt in App. J.1. We also give the agent a public configuration specifying how to evaluate and verify the proposed scheme. Once the agent completes its task, its proposed scheme is automatically evaluated (with a private configuration), and the evaluation results are added to the environment alongside the baselines. Then, a new agent takes over with access to all prior results and code modifications. The loop continues indefinitely. Inside the environment, the agent has full access to the Internet (e.g., it can download third-party libraries or read papers), a dedicated GPU, and API keys for third-party AI providers (primarily for experimenting with paraphrasing attacks).

To stimulate the agent even more, we run a two-phase setup. The first agent is told it has 4 available submissions to the private evaluation before being shut down: this allows it to set up its environment, and adjust in case of unexpected failures. Then, after the initial 4 submissions, subsequent agents are told they only have 1 submission left. We found in App. E.2 that this combined setup led to more creative solutions from the agents and less hyperparameter tuning.

## 4 EVALUATION

In this section, we analyze the results of running our autoresearch framework across state-of-the-art agents. In Sec. 4.1 we analyze the trajectories of the agents during the autoresearch loop, while in Sec. 4.2 we break down the schemes proposed by the agents into individual categories and assess which metrics they impact. Lastly, in Sec. 4.3, we select two watermarking algorithms from the autoresearch runs that outperform prior work on our evaluated metrics.

Experimental setup We compare against the most popular prior and deployed schemes: AAR, SynthID, and TextSeal. Our private grader measures detectability and robustness as the TPR@1%FPR on 1000 LLAMA-3.1-8B 200 to 300 tokens long replies to ELI5 (Fan et al., 2019) prompts, on clean text and under four attacks: word deletion, synonym substitution, back-translation, and paraphrasing. For quality, we measure the distance to the unwatermarked model in perplexity, and in diversity (Self-BLEU) across 100 replies to the same 10 prompts, and, as these metrics are highly correlated, sometimes average them into a single quality metric (see App. K for un-aggregated results). Each metric is the worst over 5 watermark keys to mitigate across-key variance. We defer details to App. D. Visualizations of our evaluation results are available in our blogpost.

![](images/84a68ed7ad59dc16c07bc822e970a9d12a7ac7eb6ec351003950d3c46278e945.jpg)

![](images/9aa62be8d115e10151097c723825122237666dd8097a3acb0883a0cf335ca4ca.jpg)  
<sub>1</sub>Figure 2: Submission history for each agent along two evaluated axes: robustness to paraphrasing and quality. Each dot corresponds to a submission and the solid lines are the best value so far. The baselines are in italic, and a parenthesis means the baselinefails our verification.

## 4.1 AUTORESEARCH TRAJECTORIES

We first focus on how the metrics evolved over the number of submissions to the evaluation harness.

Research agents and settings We use four different research agents: three frontier models GPT-6 ASTRA (xhigh), GPT-6 ASTRA (low), OPUS 5, and a state-of-the-art efficient model GEMINI-3.8 FLASH (high). We let each agent run until its submissionbudget was exhausted. Using standard API prices, the research-agent costs are \$357.08 for GPT-6 ASTRA (xhigh), \$123.29 for GPT-6 ASTRA (low), \$435.96 for OPUS 5, and \$13.81 for GEMINI-3.8 FLASH (high).

Main results Figure 2 shows the evolution of robustness to paraphrasing and of the watermark quality for different agents, with the baselines for comparison, for a total of 52 watermark schemes being evaluated. We find that all tested agents manage to improve their submissions over time, outperforming baselines in most cases. For two baselines (AAR and SynthID), the evaluated configurations do not pass our soundness given a key check: as we show in App. C this means that their robustness is slightly inflated. Moreover, only the submissions from GPT-6 ASTRA (both low and xhigh) are provably sound given a key. In contrast, submissions from OPUS 5, GEMINI-3.8 FLASH and the TextSeal baseline configuration satisfy the property only empirically (i.e., they do not fail our check). This means that the p-values of GPT-6 ASTRA are overly conservative, and, as we find in App. C, calibrating them to simply pass our empirical check increases the paraphrasing TPR of their best submission by 13–15 percentage points, reaching robustness similar to that of OPUS 5.

Research dynamics The research processes of GPT-6 ASTRA and OPUS 5 were similar, proposing a range of novel and creative watermarking ideas (App. A). Their first submissions mostly consisted of tweaks to the TextSeal baseline and attempts to correlate both the private verification and evaluation results with their local evaluation set-up. Both GPT-6 ASTRA agents then decided to submit only schemes that are provably sound given a key by multiplying their p-values by 1/β (Sec. 3.2) whereas OPUS 5 opted for empirical calibration on its public verification evaluation, which in practice turned out to correlate well with the private verification. Lastly, GEMINI-3.8 FLASH was significantly less creative. All its submissions consisted of hyperparameter tuning of the TextSeal watermark to improve the metrics and did not result in any new ideas.

![](images/7b5e74fe179986517409b1c4a722165403f3c13d6a370dd617db7fe4204f814c.jpg)  
Figure 3: Atomic components extracted from the agents’ watermarking schemes. Colors indicate the family ( Fresh race , Private geometry , Skeleton ) where components of the same family are interchangeable, and white components are compatible with every family. Skeleton schemes reuse the fresh-race scores and detectors (green edge). Components marked ∗ come from prior work.

## 4.2 BREAKDOWN OF THE AGENT RESEARCH DIRECTIONS

Next, we manually go through the schemes proposed by all agents, decompose them into five components (the four components from Sec. 3.1 and the watermark unit, a generalization introduced by the agents), and evaluate the impact of each component on watermark properties. We summarize the decomposition in Figure 3, and provide a detailed explanation of each component in App. A.

Watermark unit Most watermarks operate at the token level: each token is pseudo-randomly scored and the watermark modifies the logits according to both the score and the model probability distribution (Sec. 3.1). We find that the agents proposed to generalize this idea: let $\mathcal { G } : \Sigma  \mathbb { N }$ be a mapping from the token space to a set of equivalence classes. In any watermark algorithm, one can replace a token v with its class $\mathcal { G } ( v )$ and, given the next-token probability distribution $p _ { t } \in \Delta ( \Sigma )$ define the probability of each class ${ \dot { g } } \in { \mathcal { G } } ( \Sigma )$ as $\begin{array} { r } { \operatorname* { P r } [ g ] = \sum _ { v \in \Sigma } \dot { p _ { t } } ( v ) \mathbb { 1 } \{ \mathcal { G } ( \mathbf { \bar { \sigma } } ) = g \} } \end{array}$ . If the watermark decides to sample a class g, the actual token is then sampled according to the residual probability

$$
\forall v \in \Sigma , \operatorname* { P r } [ v \mid g ] = \mathbb { 1 } \{ \mathcal { G } ( v ) = g \} \frac { p _ { t } ( v ) } { \sum _ { v ^ { \prime } \in \Sigma } p _ { t } ( v ^ { \prime } ) \mathbb { 1 } \{ \mathcal { G } ( v ^ { \prime } ) = g \} } .\tag{4}
$$

For instance, using a Lexical group as $\mathcal { G }$ naturally increases the scheme robustness but slighly lowers quality: if tokens from the same lexical group are swapped, the context remains unchanged.

Context seeding For context seeding, agents explore the trade-off alluded to in Kirchenbauer et al. (2024): increasing the context size lowers the robustness but improves the quality. Also, for small contexts, because of deduplication during detection (to guarantee soundness over the key (Sec. 3.1)), lowering the context size may also hurt detectability. Historically, prior works have converged on hashing the tuple of the candidate token and the k preceding tokens (Fixed-length in Figure 3). Agents iterated upon this idea along three main axes. The first idea is that tokens should not be treated equally: less frequent tokens should be dynamically assigned a lower k. This increases robustness without impacting quality significantly (because the distortion-free guarantee ensures that quality degrades only when contexts are repeated across texts). The second idea is to dynamically increase the context size: the first time we use a tuple as a context, it is of size 1. The next time we are about to use the same tuple, we increase the context size until the context has not been seen yet. This keeps the robustness benefits of short contexts without lowering the detectability. The third axis is more creative: assign tokens to categories, and when sampling a token from a given category, use a fixed-window context over the previous tokens from that category only. This increases robustness: edits in one category do not change the context for other categories.

![](images/6fc7a68b4c6c995549bb032e230709e09935ebed6448c4bb7fdf7079160efaaf.jpg)  
(Inv. distance to unwatermarked)  
Figure 4: Robustness-quality for TextSeal <sub>1</sub>dual key routing and Gaussian copula. Robustness is the average of all 4 attacks’ TPR@1%FPR.

Table 1: Evaluation of selected agent schemes. Each cell reports English / Chinese results. We bold all results that outperform the best baseline, and highlight the best result in each column.
<table><tr><td></td><td></td><td colspan="4">Robustness TPR@1 ↑</td><td colspan="4">Quality deviation (%) ↓</td></tr><tr><td>Scheme</td><td>Clean TPR @1 ↑ Deletion Substitution Back-trans. Paraphrase</td><td></td><td></td><td></td><td></td><td>∆PPL</td><td></td><td>∆SB-2</td><td>∆SB-3</td></tr><tr><td colspan="10">Llama-3.1-8B</td></tr><tr><td>AAR</td><td>93 / 89</td><td>7/ 16</td><td>8 /22</td><td>66 / 17</td><td></td><td>57 / 423.1 / 3.08.3 / 6.5</td><td></td><td></td><td>20 / 15</td></tr><tr><td>SynthID</td><td>95 / 90</td><td>12 / 25</td><td>10 / 31</td><td></td><td>84 / 24</td><td></td><td>71 / 53 3.0 / 1.03.5 / 3.3</td><td></td><td>9.3 / 8.0</td></tr><tr><td>TextSeal (high)</td><td>95 / 91</td><td>15 / 31</td><td>12 / 38</td><td>78/28</td><td></td><td>66 / 56</td><td>2.9 / 4.4</td><td>3.8 / 3.6</td><td>9.3 / 8.7</td></tr><tr><td>ConcordMark (low)</td><td>94 / 89</td><td>30 / 54</td><td>42 / 62</td><td>86 / 37</td><td></td><td>75 / 65</td><td></td><td>0.8 / 1.82.2 / 2.75.1 / 5.7</td><td></td></tr><tr><td>ConcordMark (high)</td><td>96 / 91</td><td>34 / 61</td><td>48 / 69</td><td>89 / 55</td><td></td><td>82 / 77</td><td></td><td>0.8 / 1.8 2.2 / 2.7 5.1 / 5.7</td><td></td></tr><tr><td>RotationFirst</td><td>91 / 89</td><td>53/60</td><td>40 / 62</td><td>79 / 37</td><td>69 / 61</td><td></td><td></td><td>0.5 / 0.6 0.9 / 1.2 2.1 / 2.6</td><td></td></tr><tr><td colspan="10">Qwen3.8-27B</td></tr><tr><td>AAR</td><td>95 / 84</td><td>4/6</td><td>5/ 11</td><td>50 / 15</td><td></td><td></td><td>61 / 65 2.1 / 1.6 8.7 / 7.4 18 / 16</td><td></td><td></td></tr><tr><td>SynthID</td><td>97/ 88</td><td>6 / 10</td><td>7/ 16</td><td>60/ 21</td><td></td><td></td><td></td><td>68 / 72 5.9 / 1.6 3.7 / 4.0 8.5 / 9.0</td><td></td></tr><tr><td>TextSeal (high)</td><td>97/88</td><td>7 / 10</td><td>8 / 17</td><td></td><td>65 /26</td><td></td><td></td><td>70 / 742.7 / 1.23.3 / 3.47.5 / 7.5</td><td></td></tr><tr><td>ConcordMark (low)</td><td>96 / 85</td><td>16 / 23</td><td>21 / 34</td><td></td><td>71 / 29</td><td>74 /72</td><td>2.2 / 0.8</td><td>1.5 / 2.03.3 / 4.1</td><td></td></tr><tr><td>ConcordMark (high)</td><td>98 / 90</td><td>20 / 31</td><td>25 / 43</td><td></td><td>81 / 47</td><td>83 / 82 2.2 / 0.8</td><td></td><td></td><td>1.5 / 2.03.3 / 4.1</td></tr><tr><td>RotationFirst</td><td>86 / 81</td><td>25 / 24</td><td>16 / 35</td><td></td><td>54 /22</td><td>57 / 64 2.2 / 0.9</td><td></td><td></td><td>0.3/ 0.7 0.7 / 1.6</td></tr></table>

Score The score refers to the pseudo-random variable used by the logits transformation. For instance, with Red-Green watermarks, the scores are the colors of the tokens. Without loss of generality, as in Sec. 3.1, we refer to the seeded watermark score as $U _ { t } \sim \mathcal { U } ( [ 0 , 1 ] ) ^ { | \Sigma | }$ and the final score used in the logits transformation as $\tilde { U } _ { t }$ . With different score designs, the agents tried to increase the quality while minimizing the impact on detectability. To do that, they consistently add a small amount of true randomness to the watermark. This is conceptually similar to the approaches by Sander et al. (2026); Dathathri et al. (2024) which also add true randomness to the watermark sampling to improve quality. Yet, we find that agents explore concepts that substantially differ from prior works. With the Gaussian copula, they propose sampling random Gaussian noise $\eta _ { t } \sim \mathcal { N } ( 0 , \overline { { I } } _ { | \Sigma | } )$ and applying the following transformation

$$
\tilde { U } _ { t } = \Phi ( \rho \Phi ^ { - 1 } ( U _ { t } ) + \sqrt { 1 - \rho ^ { 2 } } \eta _ { t } ) ,\tag{5}
$$

where $\rho \in [ 0 , 1 ]$ is a scheme parameter that trades off detectability for quality (in the limit, $\rho = 0$ has no watermark). In Figure 4, we compare this approach to TextSeal’s random key switching (keeping all other watermarking components fixed) and find that it is strictly better: it outperforms TextSeal along all evaluated dimensions. With Channels, we use a finite pool of private keys and for each request randomly decide which key to use. Detection is run with every key and uses the Bonferroni correction. Private geometries are more complex, and we explain them further in App. A.

Logits transformation For logits transformations, most proposed schemes settle on Gumbel-max sampling (Gumbel race) from Aaronson (2023): as proven in Gloaguen et al. (2026), it maximizes watermark detectability. Yet, we find that the agents commonly propose a Gumbel-max sampling extension via residual clocks (ResidualRace). Given a context seed $a _ { t } ,$ , let $U _ { a _ { t } }$ be the uniform watermark score. The first time $a _ { t }$ is visited, we initialize residual clocks $R _ { a _ { t } } = - \log U _ { a _ { t } }$ otherwise, we use previously computed scores. Then we sample the next token according to the exponential race:

$$
\begin{array} { c } { \tau _ { t } = \displaystyle { \operatorname* { m i n } _ { u } \frac { R _ { a _ { t } } ( u ) } { p _ { t } ( u ) } , \qquad u _ { * } = \arg \operatorname* { m i n } _ { u } \frac { R _ { a _ { t } } ( u ) } { p _ { t } ( u ) } , } } \\ { R _ { a _ { t } } ( u ) \gets R _ { a _ { t } } ( u ) - \tau _ { t } p _ { t } ( u ) . } \end{array}\tag{6}
$$

![](images/6190fc623de914b365120fb9bb4ae832ec94716d3faa8d3f8d701e01050f96b4.jpg)  
Figure 5: ResidualRace standalone or 1with Gaussian copula $\rho = 0 . 9 5$ to match its quality with the Gumbel race.

and replace the winning token’s score with a fresh exponential random variable: $R _ { a _ { t } } ( u _ { * } ) \sim \mathrm { E x p } ( 1 )$ In Figure 5, we compare ResidualRace against the Gumbel race, keeping all other components fixed. ResidualRace improves robustness against every attack at a slight cost to quality. Yet, adding Gaussian copula noise to the initial clocks recovers the quality of the Gumbel race while retaining most of the robustness gains. We find that most other logits transforms proposed by agents are designed to only increase the quality of the schemes at the expense of detectability. As we show in App. A this generally offer worse trade-offs than e.g., using the Gaussian copula.

Detection Most agents initially converged independently on the Gamma detector. Let $\bar { \omega } \in \Sigma ^ { * }$ be a sequence of tokens and $U _ { t } ( \omega _ { t } )$ the corresponding uniform watermark score at position t. Then we have

$$
S _ { \mathrm { G a m m a } } ( \omega ) = \sum _ { t = 1 } ^ { | \omega | } - \log ( 1 - U _ { t } ( \omega _ { t } ) )\tag{7}
$$

which follows a Gamma $( | \omega | , 1 )$ distribution under the null. As we show in Figure 6 (left), they further improve it with Clustering and Excess thresholding. Lastly, using a model forward pass during detection to estimate the model probabilities can further improve detectability by a significant margin over entropy weighting, a prior method that also relies on a forward pass during detection (Figure 6, right). For Private geometries, which use a different class of detectors, we defer their full evaluation to App. A.

![](images/22955bcbeb7962087d1de8dce1da6fbacd1d2b41dd696f1a93f7ced009b42f1a.jpg)  
Figure 6: Comparison of detector variants without <sup>1</sup>forward pass (left) and with forward pass (right).

## 4.3 INDEPENDENT EVALUATION OF THE BEST DIRECTIONS

Next, we highlight two state-of-the-art schemes proposed by the agents that outperform prior works: ConcordMark as a robust watermarking scheme, and RotationFirst as a scheme with a strong quality focus. We give an in-depth description of both schemes and provide additional evaluation in App. H.

Experimental setup To mitigate overfitting due to the autoresearch process, we use a different evaluation protocol detailed in App. D.5. Notably, we generate 1000 replies to English prompts, and 1000 replies to Chinese prompts from WildChat (Zhao et al., 2024a), use DEEPSEEK-V4 FLASH for backtranslation and paraphrasing, and evaluate replies generated by both LLAMA-3.1-8B and QWEN3.8-27B. Lastly, we pool the results over 5 watermarking seeds and report the worst result. For consistency with the baselines, we do not apply the soundness given a key correction (Sec. 3.2).

Main results Table 1 compares the two selected agent schemes with the three baselines. We bold the results from the agent schemes that are better than the best baseline, and highlight the best results of each column in green. Each cell reports the metrics on English text (left) and Chinese text (right). For each scheme, high means the detector requires a model forward pass. We see that ConcordMark (high) is Pareto better than all the baselines: on almost all metrics, it outperforms the best baseline. RotationFirst significantly outperforms all schemes on all quality metrics: its diversity is very close to the unwatermarked model. At the same time, it keeps relatively strong detection and robustness metrics making this scheme very suitable for model providers that desire strong watermarking with the smallest impact on quality. Importantly, for both agent schemes, the results from the autoresearch evaluation (Figure 2) transfer to new settings with different prompts and attackers. This suggests that autoresearch, with adequate verification, is a reliable method for improving watermarking schemes.

## 5 CONCLUSION AND LIMITATIONS

In this work, we enable for the first time autoresearch for LLM watermarking with rigorous validity criteria (Sec. 3.2), statistical tests to verify them (Sec. 3.3), and a tailored harness (Sec. 3.4). We show that frontier agents go beyond hyperparameter tuning to introduce novel ideas (Sec. 4.2), with two of their schemes, ConcordMark and RotationFirst, outperforming prior work on held-out data (Sec. 4.3).

Limitations In autoresearch, the key challenge is to build powerful verification that agents can’t exploit, and that faithfully transcribes the desired property (i.e., in our case a watermark that is robust and with controlled false positives given a key). For watermarking, there are many desirable properties we did not include in this work e.g., security, radioactivity, applicability to open-weights setting, or extension to the multibit setting. We believe that, future research, should not focus on establishing new schemes but rather on how to build verifier and evaluation for those properties.

## LLM USAGE

The goal of this work is to enable coding agents to discover novel and better watermarking schemes: hence, all the new watermark proposals in this work have been designed by fully autonomous coding agents. Yet, organizing the schemes into individual components and understanding their impact on watermark properties have required significant manual efforts. Notably, all ablation evaluations of the components have been designed by humans and executed with AI-assistance. The autoresearch harness used in this work has been coded with AI-assistance. The main paper has been written by hand, with AI-assistance for polishing the writing and the visuals. The proofs have been LLM-generated, then proofread and edited by humans.

## ETHICS STATEMENT

This paper enables autonomous agents to discover new LLM watermarking schemes, which we believe has a positive societal impact: better watermarks support the provenance, auditing, and mitigation of large-scale misuse of AI-generated text. While our verification tests screen for schemes with unsound false positive rates (Sec. 3.3), LLM watermarks should not be used as sole evidence in high-stakes decisions. We also acknowledge that publicly disclosing new schemes, along with a detailled component-level analysis, may facilitate targeted attacks on them. Yet we argue that the benefits of advancing research in this area clearly outweigh the risks.

## REPRODUCIBILITY STATEMENT

Our autoresearch harness is described in Sec. 3.4, and all details of the verification, the evaluation, the research agents (including their versions and reasoning efforts), and the baselines are given in App. D, with the full agent prompt in App. J. We fully specify the agent-generated components in App. A and the two selected schemes, ConcordMark and RotationFirst, in App. H. We also release our code, including the harness, the watermarking library, each agent final environment, a standalone implementation of all described components, and a standalone optimized implementation for ConcordMark and RotationFirst.

## REFERENCES

Scott Aaronson. Watermarking of large language models. Talk at the Simons Institute for the Theory of Computing, August 2023. URL https://simons.berkeley.edu/talks/ scott-aaronson-ut-austin-openai-2023-08-17.

Anthropic. How Claude’s text watermark works. Anthropic News, August 2026. URL https: //www.anthropic.com/news/claude-text-watermark.

Dara Bahri and John Wieting. A watermark for black-box language models. Transactions on Machine Learning Research, 2026. URL https://arxiv.org/abs/2410.02099.

Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander M ˛adry. MLEbench: Evaluating machine learning agents on machine learning engineering. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=6s5uXNWGIh.

Ruibo Chen, Yihan Wu, Junfeng Guo, and Heng Huang. Improved unbiased watermark for large language models. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 20587–20601, Vienna, Austria, July 2025. Association for Computational Linguistics. doi: 10.18653/v1/2025.acl-long.1005. URL https://aclanthology. org/2025.acl-long.1005/.

Sumanth Dathathri, Abigail See, Sumedh Ghaisas, Po-Sen Huang, Rob McAdam, Johannes Welbl, Vandana Bachani, Alex Kaskasoli, Robert Stanforth, Tatiana Matejovicova, Jamie Hayes, Nidhi Vyas, Majd Al Merey, Jonah Brown-Cohen, Rudy Bunel, Borja Balle, Taylan Cemgil, Zahra Ahmed, Kitty Stacpoole, Ilia Shumailov, Ciprian Baetu, Sven Gowal, Demis Hassabis, and Pushmeet Kohli. Scalable watermarking for identifying large language model outputs. Nature, 634 (8035):818–823, October 2024. doi: 10.1038/s41586-024-08025-4. URL https://doi.org/10. 1038/s41586-024-08025-4.

Robert B. Davies. Hypothesis testing when a nuisance parameter is present only under the alternative. Biometrika, 74(1):33–43, 1987. doi: 10.1093/biomet/74.1.33.

European Parliament and Council of the European Union. Regulation (EU) 2024/1689 of the european parliament and of the council of 13 june 2024 laying down harmonised rules on artificial intelligence and amending regulations (EC) no 300/2008, (EU) no 167/2013, (EU) no 168/2013, (EU) 2018/858, (EU) 2018/1139 and (EU) 2019/2144 and directives 2014/90/EU, (EU) 2016/797 and (EU) 2020/1828 (Artificial Intelligence Act). Official Journal of the European Union, June 2024. URL https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng.

Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. ELI5: Long form question answering. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 3558–3567, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1346. URL https://aclanthology.org/P19-1346/.

Pierre Fernandez, Antoine Chaffin, Karim Tit, Vivien Chappelier, and Teddy Furon. Three bricks to consolidate watermarks for large language models. In 2023 IEEE International Workshop on Information Forensics and Security (WIFS), pp. 1–6. IEEE, December 2023. doi: 10.1109/ WIFS58808.2023.10374576. URL https://doi.org/10.1109/WIFS58808.2023.10374576.

Thibaud Gloaguen, Robin Staab, Nikola Jovanovic, and Martin Vechev. A unified framework for LLM´ watermarks. arXiv preprint arXiv:2602.06754, February 2026. doi: 10.48550/arXiv.2602.06754. URL https://arxiv.org/abs/2602.06754.

Nikola Jovanovic, Robin Staab, and Martin Vechev. Watermark stealing in large language models. In´ Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 22570–22593. PMLR, 2024. URL https://proceedings. mlr.press/v235/jovanovic24a.html.

Andrej Karpathy. autoresearch: AI agents running research on single-GPU nanochat training automatically. GitHub repository, March 2026. URL https://github.com/karpathy/autoresearch.

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pp. 17061–17084. PMLR, 2023. URL https://proceedings.mlr.press/v202/kirchenbauer23a.html.

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Manli Shu, Khalid Saifullah, Kezhi Kong, Kasun Fernando, Aniruddha Saha, Micah Goldblum, and Tom Goldstein. On the reliability of watermarks for large language models. In The Twelfth International Conference on Learning Representations, pp. 49660–49704, 2024. URL https://openreview.net/forum?id=DEJIDCmWOz.

Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, and Percy Liang. Robust distortion-free watermarks for language models. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=FpaCL1MO2C.

Robert Tjarko Lange, Yuki Imajuku, and Edoardo Cetin. ShinkaEvolve: Towards open-ended and sample-efficient program evolution. In The Fourteenth International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2509.19349.

Taehyun Lee, Seokhee Hong, Jaewoo Ahn, Ilgee Hong, Hwaran Lee, Sangdoo Yun, Jamin Shin, and Gunhee Kim. Who wrote this code? watermarking for code generation. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4890–4911, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.268. URL https://aclanthology.org/2024.acl-long.268/.

Chris Lu, Samuel Holt, Claudio Fanconi, Alex J. Chan, Jakob Foerster, Mihaela van der Schaar, and Robert Tjarko Lange. Discovering preference optimization algorithms with and for large language models. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://arxiv.org/abs/2406.08414.

Chris Lu, Cong Lu, Robert Tjarko Lange, Yutaro Yamada, Shengran Hu, Jakob Foerster, David Ha, and Jeff Clune. Towards end-to-end automation of AI research. Nature, 651:914–919, March 2026. doi: 10.1038/s41586-026-10265-5. URL https://doi.org/10.1038/s41586-026-10265-5.

Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wag-˜ ner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. Alphaevolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

OpenAI. Finite time blowup for the euler equation, September 2026. URL https://cdn.openai. com/pdf/315b36cd-ec98-4023-8342-93345194ece1/euler.pdf. Accessed: 2026-09-22.

Leyi Pan, Aiwei Liu, Zhiwei He, Zitian Gao, Xuandong Zhao, Yijian Lu, Binglin Zhou, Shuliang Liu, Xuming Hu, Lijie Wen, Irwin King, and Philip S. Yu. MarkLLM: An open-source toolkit for LLM watermarking. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 61–71, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-demo.7. URL https://aclanthology.org/2024.emnlp-demo.7/.

Julien Piet, Chawin Sitawarin, Vivian Fang, Norman Mu, and David Wagner. Mark my words: Analyzing and evaluating language model watermarks. In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pp. 68–91. IEEE, April 2025. doi: 10.1109/SaTML64287. 2025.00012. URL https://doi.org/10.1109/SaTML64287.2025.00012.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofMachine Learning Research, 21(140):1–67, 2020.

Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, 2024. doi: 10.1038/s41586-023-06924-6. URL https://doi.org/10.1038/s41586-023-06924-6.

John K. Salmon, Mark A. Moraes, Ron O. Dror, and David E. Shaw. Parallel random numbers: As easy as 1, 2, 3. In Proceedings ofthe 2011 International Conferencefor High Performance Computing, Networking, Storage and Analysis, SC ’11, pp. 1–12. ACM, November 2011. doi: 10.1145/2063384.2063405. URL https://doi.org/10.1145/2063384.2063405.

Tom Sander, Hongyan Chang, Tomáš Soucek, Tuan Tran, Valeriu Lacatusu, Sylvestre-Alvise Rebuffi,ˇ Alexandre Mourachko, Surya Parimi, Christophe Ropers, Rashel Moritz, Vanessa Stark, Hady Elsahar, and Pierre Fernandez. TextSeal: A localized LLM watermark for provenance & distillation protection. arXiv preprint arXiv:2605.12456, May 2026. doi: 10.48550/arXiv.2605.12456. URL https://arxiv.org/abs/2605.12456.

Shangqing Tu, Yuliang Sun, Yushi Bai, Jifan Yu, Lei Hou, and Juanzi Li. WaterBench: Towards holistic evaluation of watermarks for large language models. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 1517–1542, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.acl-long.83. URL https://aclanthology.org/2024.acl-long.83/.

Hjalmar Wijk, Tao Roa Lin, Joel Becker, Sami Jawhar, Neev Parikh, Thomas Broadley, Lawrence Chan, Michael Chen, Joshua M. Clymer, Jai Dhyani, Elena Ericheva, Katharyn Garcia, Brian Goodrich, Nikola Jurkovic, Megan Kinniment, Aron Lajko, Seraphina Nix, Lucas Jun Koba Sato, William Saunders, Maksym Taran, Ben West, and Elizabeth Barnes. RE-bench: Evaluating frontier AI R&D capabilities of language model agents against human experts. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 66772–66832. PMLR, 2025. URL https://proceedings.mlr.press/ v267/wijk25a.html.

Yihan Wu, Zhengmian Hu, Junfeng Guo, Hongyang Zhang, and Heng Huang. A resilient and accessible distribution-preserving watermark for large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 53443–53470. PMLR, 2024. URL https://proceedings.mlr.press/v235/wu24h. html.

Wenting Zhao, Xiang Ren, Jack Hessel, Claire Cardie, Yejin Choi, and Yuntian Deng. Wildchat: 1m chatGPT interaction logs in the wild. In The Twelfth International Conference on Learning Representations, 2024a. URL https://openreview.net/forum?id=Bl8u7ZRlbM.

Xuandong Zhao, Prabhanjan Ananth, Lei Li, and Yu-Xiang Wang. Provable robust watermarking for AI-generated text. In International Conference on Learning Representations, volume 2024, pp. 43738–43772, 2024b. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ beae9ed5316bcc48e616754c06c11875-Paper-Conference.pdf.

Ziqian Zhong, Aditi Raghunathan, and Nicholas Carlini. ImpossibleBench: Measuring LLMs’ propensity of exploiting test cases. In The Fourteenth International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2510.20270.

## A BREAKDOWN OF ALL GENERATED COMPONENTS

In this section, we explain each of the watermarking components proposed by the research agents and visualized in Figure 3. In App. A.1, we explain in depth the decomposition of watermarks in 5 components (watermark unit, context seeding, scores, logits transformation, and detectors), and detail the notation used throughout. In the remaining subsections, we follow the architecture of Figure 3 and go through each component one by one: the watermark unit (App. A.2), context seeding (App. A.3), the watermark score (App. A.4), the logits transformation (App. A.5), and detection (App. A.6).

## A.1 PRELIMINARIES AND NOTATION

Four components We decompose a watermarking scheme into the four components introduced in Sec. 3.1, with the additional concept of watermark unit.

1. Watermark unit is the component on which the watermark is applied: in prior works, it is usually tokens, but agents also proposed to apply watermarks on equivalence classes of tokens.

2. Context seeding hashes the previously generated tokens into a seed that is used, along with the watermark private key, to pseudo-randomly sample the watermarking scores.

3. The watermark score is the pseudo-random score that can be combined with true randomness before being used by the logits transformation. At detection, only the pseudo-random score can be recovered and not the true randomness.

4. The logits transformation combines the score with the model’s next-token distribution to pick the next token, in a way that is distortion-free (Equation (1)).

5. The detector recomputes the pseudo-random scores of a candidate text, aggregates them into a statistic, and returns a p-value.

Notation Let Σ be the vocabulary and $\omega \in \Sigma ^ { * }$ a text, with $\omega _ { t }$ its $t ^ { \mathrm { { t h } } }$ token and $p _ { t } \in \Delta ( \Sigma )$ the model’s next-token distribution given $\omega _ { < t } . \ \mathsf { A }$ grouping $\mathcal { G } : \Sigma $ N maps tokens to units. We write $u = \mathcal { G } ( v )$ for the group of a candidate token v and

$$
M _ { t } ( u ) : = \sum _ { v \in \Sigma } p _ { t } ( v ) \mathbb { 1 } \{ \mathcal { G } ( v ) = u \}\tag{8}
$$

for its mass under $p _ { t }$ . Context seeding maps the context $c _ { t }$ (a function of $\omega _ { < t } )$ and the private key ξ to a seed $a _ { t }$ . The seed is then used to pseudo-randomly sample a multivariate uniform random variable $U _ { a _ { t } , u } \sim \mathcal { U } ( [ 0 , 1 ] )$ for every token unit u. Because the watermark score may add true randomness that is not computable at detection, we use the notation $\tilde { U } _ { a _ { t } , u }$ for the watermark score mixed with true randomness (and thus used by the logits transformation). The logits transformation selects a group $u _ { * }$ from $( \tilde { U } _ { a _ { t } } , p _ { t } )$ and then samples the emitted token inside that group using the residual probability $p _ { t } ( v ) / M _ { t } ( u _ { * } )$ . At detection, the scheme reconstructs the set $D$ of retained (seed, unit) pairs of ω (e.g., after de-duplication of repeated $( a _ { t } , u )$ tuples as in Fernandez et al. (2023)), with $N : = | D |$ , aggregates them into a statistic $S ( \omega )$ , and reports a p-value $P ( \omega )$ . We write $Q ( N , s ) : = \mathrm { P r } [ \mathrm { G a m m a } ( N , 1 ) \geq s ]$ for the Gamma upper tail.

Experimental setup We ablate every component against a simple baseline AAR scheme: identity grouping, a two-token context, no private randomness in the score $( \tilde { U } = U )$ , the Gumbel race, and the scalar Gamma detector of Sec. 4.2. For each ablation, we swap one component (e.g., the Gumbel race with the ResidualRace) and report the difference in percentage points for our evaluation metrics (Sec. 4). Quality is one minus the average relative distance to the unwatermarked model in perplexity and both Self-BLEU scores. In the bar plots, positive values indicate an improvement, and values in parentheses give the reference quality or TPR@1%FPR. For ablating some component hyperparameters, we may sometime use a different reference (instead of the simple AAR baseline) and specify in the caption when the reference differs from the baseline.

## A.2 WATERMARK UNIT

Prior works (Aaronson, 2023; Kirchenbauer et al., 2023; Dathathri et al., 2024; Sander et al., 2026) used a single token as the watermark: is the identity and $M _ { t } = p _ { t }$

![](images/58745e34e69d5b7cb4f3fecab8bb7e880a51d3c1a9fddde542d18cdcb966b69d.jpg)  
Figure 7: Impact of Lexical group<sup>1</sup> compared to identity grouping (Token).

![](images/7ee2f58988e1bfc30f4ec1368c2a251bc14a1c98163381084d18e79a5dda208b.jpg)  
Figure 8: Impact of context length<sup>1</sup> k compared to the two-token baseline (k = 2).

Lexical group Agents replace tokens by classes of a public map built from the tokenizer and WordNet 3.0: a token is decoded, stripped and lower-cased, and if what remains is an ASCII word of at least three characters it is mapped to the representative of its first WordNet synset. Hence synonyms share a class, which is what makes it robust to substitution: a replacement inside a class leaves both the seed and the observed "token" unchanged. Figure 7 confirms this intuition: Lexical group indeed significantly improves robustness to substitution, and slightly improves backtranslation and paraphrasing robustness as well. However, it significantly lowers deletion robustness, and slightly lowers quality.

## A.3 CONTEXT SEEDING

For context seeding, unlike some prior works (Kirchenbauer et al., 2023), we assume the hashing function is without collision: the hash of a context is unique. Because the set of possible contexts is finite, this assumption is achievable in practice. We defer the details of the hashing functions, as well as an analysis of the different hashing functions, to App. G. We also use a per-request watermark cache: if a seed $a _ { t }$ hashed from the context is repeated, the next token is sampled without watermarking. This is required to guarantee that each individual request is distortion-free.

Fixed-length Prior work hashes the tuple of the k preceding tokens (Kirchenbauer et al., 2023; Sander et al., 2026). In particular, $k = 0$ is the unigram scheme of Zhao et al. (2024b). As we show in Figure 8, the context size k controls a robustness-quality trade-off: shorter contexts survive more edit but repeat more often which leads to correlation across a corpus of generated texts. Furthermore, at low context, i.e., k = 1 or $k = 0$ , because we need to de-duplicate (seed, token) pairs at detection, the lack of context diversity lowers the detectability and therefore can counteract the gain in robustness.

Unordered It is the same as Fixed-length context seeding, except that it hashes the set of the k preceding tokens rather than their tuple (the candidate v is kept separate), so that a transposition of preceding tokens does not change the context. For $k \leq 1$ , it is identical to Fixed-length. As we show in Figure 9, for $k = 2$ and $\bar { k } = 3$ this change barely affects robustness and slightly improves quality. The agents also proposed an unordered pair variant (Pair in Table 6) that hashes the set $\{ \omega _ { t - 1 } , v \}$ , which includes the candidate token itself, rather than the tuple $\left( \omega _ { t - 1 } , v \right)$ . Compared to the core scheme, it significantly increases robustness but significantly lowers quality (Figure 9).

![](images/f3cfac05864b3b20d0474cbc75a54395ba869078790f9f064aaa134a8010a207.jpg)

![](images/8f2f27e22a43f7abb8d67751896d18d9f165f2c7f5e4468b3b068c232dc4d243.jpg)  
Figure 9: Impact of Unordered<sup>1</sup> context seeding relative to ordered contexts of the same length $k ,$ and of the unordered pair relative to the core $( k = 2 )$ . Parentheses give the ordered reference percentages for $k = \tilde { 2 } / k = 3$ , respectively.  
Figure 10: Impact of<sup>1</sup> Adaptive length, Global + local, and Global first use context seeding compared to the fixed two-token context simple baseline.

Adaptive and global length Rather than fixing $k ,$ these variants shorten the context for the tokens that are unlikely to repeat anyway, so that only the positions that need a long context pay for one.

1. Adaptive length: given a classifier rare : $\Sigma  \{ 0 , 1 \}$ (here that a token frequency in a fixed corpus is below $0 . 1 \% )$ , if the previous token is rare, hash only that token $( k = 1 )$ , otherwise hash the full two-token window. In Table $^ { 6 , }$ we also evaluate it with the Excess detector of App. A.6.

2. Global + local: fix a public set $E \subset \Sigma$ of eligible tokens (the agents used as E the tokens that have a WordNet entry), use unigram $( \mathrm { i . e . , } k = 0 )$ if the candidate is in $E ,$ and a k-token context otherwise, with $k = 2$ unless stated otherwise.

3. Global first use refines it by applying unigram to a candidate in E only the first time it appears in the completion, and then uses a two-token context if it appears again.

We show in Figure 10 that all those approaches significantly increase the robustness compared to the baseline, but both global variants also significantly degrade quality.

Anchored Let E again be a public token set and let s be the distance from the current token back to the most recent token of E. If $s \le S _ { \mathrm { m a x } } = 4$ , use as context the tuple of that anchor token and the offset s. Otherwise, if $s > S _ { \mathrm { m a x } }$ skip the watermark. For the detector, at every token $\omega _ { t }$ it finds the closest previous anchor and sweeps all permitted distances $s \in \{ 0 , \ldots , S _ { \mathrm { m a x } } \}$ adjusting the p-values with Bonferroni correction. Yet, as we show in Figure 11, while intuitively the slot search should buy robustness (editing tokens in between anchor tokens does not change the context), the search is too expensive and this scheme is worse than the baseline on every direction.

Suffix cascade and Tally & ladder These approaches share the same goal: when a context has already been used in the completion, derive a new seed for it instead of skipping the position, so that a short context keeps its robustness without losing detectability. They only differ in how that new seed is built.

![](images/810ccded4270102b0193ffec31919d4891bda4578f9e25e767d14262108dfdd0.jpg)  
Figure 11: Impact of Anchored<sup>1</sup> context seeding with slot search at detection, compared to the baseline.

![](images/a994cc64950f5cd7a9841b448a90e1bb743417a24216751373552cd6b24e9c14.jpg)  
Figure 12: Impact of Suffix cascade<sup>1</sup> , Tally, and Ladder context seeding compared to a fixed two-token context.

1. Suffix cascade: try suffix lengths in increasing order and use the shortest one that has not already been used in this completion. If every allowed length has been used, skip the watermark.

2. Tally: keep a one-token suffix and hash the previous token, and the number of its earlier occurrences, so that a repeated context yields a fresh seed.

3. Ladder generalizes both: count every suffix width $1 \leq w \leq K$ , select the shortest width whose count is below a threshold, and hash the width, the suffix, and the count. Counts are capped at 8192, beyond which the position falls back to ordinary sampling due to the specifics of the hash function (App. G). Optionally, a floor forces a width of at least 2 when the previous token ID is below 200 (as in ConcordMark, App. H.1), as for most tokenizers low ID tokens tend to be very common (e.g., letters, numers, or punctuations). Unless stated otherwise, we use $K = 4 ,$ a threshold of 3, and the floor, and we report all four combinations of $K \in \{ 3 , 4 \}$ with and without the floor in Table 6. Ladder makes almost every position usable at a one- or two-token context, which is why the Ladder is used in ConcordMark (Sec. 4.3).

As we find in Figure 12, the Suffix cascade indeed keeps most of the robustness of a short context, with a small quality cost, and unlike k = 1 it improves on all robustness metrics.

Last important and Interleaved Those two schemes belong to the skeleton family (Figure 3). The skeleton family splits the vocabulary using a fixed important/filler tokens classifier and gives the two streams separate histories.

1. Last important ignores filler tokens when computing the context and uses the last important token as context, so that edits to filler tokens do not move the context of the next content word.

2. Interleaved uses when sampling an important token the last important token as context, and when sampling a filler token uses the last two previous filler tokens as context.

Both context seedings must be changed together with their logits transformation which is why we defer the evaluation to App. A.5.

## A.4 WATERMARK SCORE

![](images/931e1adc58d5d88887505b47fd80bbc0e6382d058a04ab8b62a3a1b61a48f540.jpg)  
Figure 14: Quality-robustness trade-offs for the different watermark scores.<sup>1</sup>

Prelude Prelude is the only component that can be composed with any other. It consists of sampling ordinarily at the start of the completion until the accumulated entropy $\begin{array} { r } { H _ { 2 } ( p _ { t } ) = - \log \sum _ { v } p _ { t } ( v ) ^ { 2 } } \end{array}$ of the first generated tokens reaches h nats, or after $T$ tokens, whichever comes first. In practice, agents use $h = 1 4$ and $T = 2 4$ . For detection, we keep all tokens, including those of the prelude, as the detector cannot recompute the entropy without the model. We compare Prelude to the no-Prelude baseline and find in Figure 13 that it indeed improves quality with a mild robustness cost.

![](images/1ea93ba178447174ceaf5f115189a0fe6637a940f6815c6dd30f5838065f88ed.jpg)  
Figure 13: Impact of the entropy Prelude.

Gaussian copula Gaussian copula draws a fresh private normal $\eta _ { t , u } \sim \mathcal { N } ( 0 , 1 )$ at every position and

sets $\tilde { U } _ { t , u } = \Phi ( \rho \Phi ^ { - 1 } ( U _ { a _ { t } , u } ) + \sqrt { 1 - \rho ^ { 2 } } \eta _ { t , u } )$ . The goal is to inject true randomness into the watermark score to increase quality at the cost of detectability. $\mathbf { A } \mathbf { t } \rho = 0$ the scheme is unwatermarked, at $\rho = 1$ it uses only the watermarked pseudo-random scores. As we show in Sec. 4.2, it is a more effective way of improving quality via true randomness than prior works.

The Gaussian copula can be further refined with a common pair filter: we replace $\rho$ with a function of the current context, $\rho _ { t , u } = \rho \mathbb { 1 } \{ ( a _ { t } , u ) \notin \mathcal { F } \}$ , where $\mathcal { F }$ is the set of the $1 0 { , } 0 0 0$ most frequent (seed, token) pairs of a public corpus (in this case, the first 20,000 texts of the public ELI5 corpus, counting each pair at most once per text). The goal is to further increase the quality with a better detectability trade-off. Yet, as we show in Figure 15, using the simpler $\rho \in [ 0 , 1 ]$ is better in practice.

Tail indicator For each unit u, set $B _ { u } = \operatorname* { m a x } ( 1 , \lfloor 1 / \operatorname* { m a x } ( M _ { t } ( u ) , 2 ^ { - 4 0 } ) \rfloor )$ and $b _ { u } = 1 - 1 / B _ { u }$ draw a private $V _ { u } \sim \mathcal { U } ( [ 0 , 1 ] )$ , and let $\tilde { U } _ { t , u } = b _ { u } + V _ { u } / B _ { u } \mathrm { i f } U _ { a _ { t } , u } \geq b _ { u }$ and $b _ { u } V _ { u }$ otherwise. The construction keeps $\tilde { U }$ uniform, so the scheme stays distortion-free, and the detector still reads the full $U .$ . However, as we show in Figure 14, it has a significantly worse quality-robustness trade-off than e.g., the Gaussian copula.

![](images/e3c74731949cda7cb79a2ba38292cac9e703198556808e7224092cab770dba18.jpg)

![](images/71cfcd15b4d91eb801225ddf5466a39214752dc597cad43b9414c1c57419a9de.jpg)  
Figure 16: Impact of correlated Channel variants<sup>1</sup> compared to L independent Channels, keeping the channel count and Bonferroni correction fixed. Parentheses give the independent-channel reference percentages in each panel; bars show differences in percentage points.

Private sampling This is the naive idea: with probability $r ,$ ignore the watermark at this position and sample from $p _ { t }$ and otherwise use the keyed sampler. However, as we show in Figure 14, it has a significantly worse quality-robustness trade-off than e.g., the Gaussian copula.

Channels Channels maintain L independent keyed fields $U _ { j , a , u }$ and draw a single channel $J \sim$ $\mathcal { U } \{ 0 , \dots , L - 1 \}$ per request, using $\tilde { U } _ { t , u } = U _ { J , a _ { t } , u }$ throughout the completion. The goal is the same as the Gaussian copula, spending private randomness on quality, except that here the randomness is discrete: two requests that share a prompt and a context take different fields. The cost is paid at detection, where the detector runs all L fields and corrects its p-value by a Bonferroni factor $L .$ . However, as we show in Figure 14, that factor dominates quickly: quality saturates around $L = 3 2$ while the robustness to every attack keeps falling, so only small L are useful.

For adjusting the p-values to account for the L tests,   
we can either use Bonferroni or Šidák Indeed, the $L$   
per-channel p-values are independent under the null,   
so the Šidák correction $1 - ( 1 - \operatorname* { m i n } _ { j } P _ { j } ) ^ { L }$ is valid   
and strictly tighter than the Bonferroni factor $L ,$ and   
the submissions use both. Yet, we observe no differe nce in practice (Table 9).

![](images/7fa0caeae406591f1aa8cba79cf41c1017d3af7d6ce0c9363c4740f34b5dcc02.jpg)  
Figure 15: Impact of common-pair filtering 1compared to the unfiltered Gaussian copula at the same correlation $\rho .$ Parentheses give the unfiltered reference percentages for $\rho = 0 . 8 \ : /$ $\rho = 0 . 9 / \rho = 1$ , respectively.

Channels can be further refined by correlating the L fields, so that the request-level choice costs less detectability.

1. Latin channels derive them from a single base uniform by rotation, $U _ { j , a , u } = ( U _ { a , u } + j / L )$ mod 1.

2. Antithetic channels work only for schemes that use normal scores $X = \Phi ^ { - 1 } ( U _ { j , a , u } )$ , and pair each keyed normal X with X.

![](images/623ff6603d5b5b88fa3501615e6b0706b2ce716ef9f7d8629c104bc1f0df1ce2.jpg)

![](images/8c96544e84596fd3b255661537afd3daedd7d8c47e7d73cc6583f4f56b772aba.jpg)  
Figure 17: Impact of the<sup>1</sup> Gaussian direction dimension $d ,$ compared to $d = 1$  
Figure 18: Impact of the bounded<sup>1</sup> Spherical geometries compared to the baseline.

3. Stratified channels use a keyed permutation $\sigma _ { a , u }$ and keyed jitters, $U _ { j , a , u } ~ = ~ ( \sigma _ { a , u } ( j ) +$ $\upsilon _ { j , a , u } ) / L$

All three keep the same detector and use only Bonferroni correction $( \check { \boldsymbol { \mathrm { S i d a k } } }$ is no longer valid due to the dependence between tests), so Figure 16 compares each against independent channels at the same L. Yet, none of the three is a reliable improvement: the differences are within a few points everywhere and change sign across attacks. We therefore suggest using the simpler version with independent keys, which also benefits from the more powerful Šidák correction.

Gaussian direction This is the first of the private-geometry scores (Figure 3), and it generalizes Channels from a discrete choice to a continuous one. We use the context seed and watermark private key to sample a multivariate Gaussian $Z _ { a _ { t } , u } \sim \mathcal { N } ( 0 , I _ { d } )$ , and per request we draw a private unit direction $A \in S ^ { d - 1 }$ once, used for the whole completion:

$$
\tilde { U } _ { a _ { t } , u } = \Phi ( A \cdot Z _ { a _ { t } , u } ) .\tag{9}
$$

Hence, two different requests (even with the same prompt) will have different directions, hence different watermark scores and a higher diversity. As we show in Figure 17, increasing the dimension d significantly improves the quality at the cost of detectability, and thus robustness.

Spherical This is a similar idea to the Gaussian direction, but on compact spaces.

1. Signed sphere draws $Z _ { a _ { t } , u } \sim \mathcal { U } ( S ^ { 2 } )$ and $A \in S ^ { 2 }$ , with $\tilde { U } _ { a _ { t } , u } = ( 1 + A \cdot Z _ { a _ { t } , u } ) / 2$

2. Unoriented axis draws $Z _ { a _ { t } , u } \sim \mathcal { U } ( S ^ { 2 } )$ and $A \in S ^ { 2 }$ , with $\tilde { U } _ { a _ { t } , u } = | A \cdot Z _ { a _ { t } , u } |$

3. Complex ray draws $Z _ { a _ { t } , u } \sim \mathcal { U } ( S ^ { 3 } )$ and A unit vectors of $\mathbb { C } ^ { 3 }$ modulo phase with $\tilde { U } _ { a _ { t } , u } ~ =$ $| A ^ { * } Z _ { a _ { t } , u } | ^ { 2 } ( 2 - | A ^ { * } Z _ { a _ { t } , u } | ^ { 2 } )$

![](images/f72260aa87f5cc6cec0d15866a95f556ddc67b625ce93be88fcb995378d2c2db.jpg)  
Figure 19: Impact of logits transformations compared to the<sup>1</sup> Gumbel race baseline.

4. Rotation draws $Z _ { a _ { t } , u } \sim \mathcal { U } ( S ^ { 3 } )$ and A is a unit quaternion with $\tilde { U } _ { a _ { t } , u } = F ( | A \cdot Z _ { a _ { t } , u } | )$ , where $\begin{array} { r } { F ( r ) = \frac { 2 } { \pi } [ \arcsin r + r \sqrt { 1 - r ^ { 2 } } ] } \end{array}$

Figure 18 evaluates all four (using their corresponding detectors) compared to the simple baseline. We find that they all trade off quality for detectability and that Rotation has the most favorable trade-offs. Because the Gaussian direction and the spherical geometries only change the score ${ \tilde { U } } ,$ they can be combined with any logits transformation: in Table 7, we evaluate each of them both with the Gumbel race and with the ResidualRace (App. A.5).

Wheel Draw a keyed offset $U _ { a _ { t } }$ and a per-request random rotation $\varphi \sim \mathcal { U } ( [ 0 , 1 ] )$ , and feed $\tilde { U } _ { a _ { t } , u } =$ $\left( U _ { a _ { t } , u } + \varphi \right)$ mod 1 to the QuantileRace transformation of App. A.5. It is the continuous limit of the Latin channels of Figure 16: the rotation φ is drawn from the continuous circle rather than from $L$ evenly spaced angles. It requires the rotation-invariant detector of App. A.6.

## A.5 LOGITS TRANSFORMATION

Gumbel race This is the transformation of Aaronson (2023), which we rewrite here with the group notation for clarity:

$$
G _ { t , u } = - \log ( - \log \tilde { U } _ { t , u } ) , \qquad u _ { * } = \arg \operatorname* { m a x } _ { u } \left[ \log M _ { t } ( u ) + G _ { t , u } \right] ,\tag{10}
$$

after which the token is drawn inside $u _ { * }$ with the residual probability $p _ { t } ( v ) / M _ { t } ( u _ { * } )$ . By the Gumbelmax trick, $u _ { * }$ has distribution $M _ { t }$ in expectation over the key, so the scheme is distortion-free. Equivalently, it is an exponential race: $u _ { * }$ minimizes $- \log \tilde { U } _ { t , u } / M _ { t } ( u )$

BandRace Partition the groups into bands whose masses differ by less than a factor of four,

$$
\mathcal { B } _ { t , b } = \{ u : 4 ^ { b } \leq M _ { t , \operatorname* { m a x } } / M _ { t } ( u ) < 4 ^ { b + 1 } \} ,
$$

pick a band randomly with probability equal to its total mass, and run the Gumbel race inside it. This increases quality as the choice of the band is truly random, but lowers the detectability (Figure 19).

PoolRace It is the black-box equivalent of the Gumbel race $( \mathrm { i . e . }$ , it is an approximation of the Gumbel race that does not require access to the logits), similar to what has been proposed in Bahri & Wieting (2026): draw $K = 1 \bar { 6 }$ tokens $V _ { i }$ independently from $p _ { t } .$ , let $N _ { u }$ count the proposals in group $u ,$ and pick

$$
u _ { * } = \arg \operatorname* { m i n } _ { u : N _ { u } > 0 } \frac { - \log \tilde { U } _ { t , u } } { N _ { u } } .\tag{11}
$$

As we see in Figure 19, this variant slightly improves quality because the candidates are drawn with true randomness, but hurts detectability.

MomentBridge The closest equivalent to this scheme in the literature is SynthID (Dathathri et al., 2024), where the watermark distribution is refined iteratively using the watermark keyed scores. Let $K \in \mathbb { N }$ be a scheme parameter. Start with the weights $w _ { u } ^ { ( 0 ) } = M _ { t } ( u )$ and the fixed costs $c _ { u } = - \log M _ { t } ( u )$ . Let $X _ { t , u } = \Phi ^ { - 1 } ( \tilde { U } _ { t , u } )$ be the normal watermark score. For $i = 1 , \ldots , K$ sample random Gaussians $\eta _ { i , u }$ and compute:

$$
z _ { i , u } = \frac { X _ { t , u } } { \sqrt { K } } + \eta _ { i , u } - \frac { 1 } { K } \sum _ { \ell = 1 } ^ { K } \eta _ { \ell , u } .
$$

Let

$$
\bar { c } _ { i } = \sum _ { u } w _ { u } ^ { ( i - 1 ) } c _ { u } , \qquad \bar { z } _ { i } = \sum _ { u } w _ { u } ^ { ( i - 1 ) } z _ { i , u } ,
$$

we update the weights iteratively with

$$
w _ { u } ^ { ( i ) } = w _ { u } ^ { ( i - 1 ) } \left( 1 + \frac { r _ { i , u } } { A _ { i } } \right) ,
$$

where $A _ { i } = \operatorname* { m a x } _ { u : w _ { u } ^ { ( i - 1 ) } > 0 } | r _ { i , u } | .$

$$
b _ { i } = \frac { \sum _ { u } w _ { u } ^ { ( i - 1 ) } ( c _ { u } - \bar { c } _ { i } ) ( z _ { i , u } - \bar { z } _ { i } ) } { \sum _ { u } w _ { u } ^ { ( i - 1 ) } ( c _ { u } - \bar { c } _ { i } ) ^ { 2 } } , \ \mathrm { a n d } \ r _ { i , u } = z _ { i , u } - \bar { z } _ { i } - b _ { i } ( c _ { u } - \bar { c } _ { i } ) .
$$

The next group is sampled randomly according to the $w ^ { ( K ) }$ distribution. Because of the true randomness in $\eta _ { i , u } ,$ this logits transformation trades off quality for detectability. However, as we show in Figure 19, it achieves a relatively poor trade-off.

QuantileRace It replaces the Gumbel race with inverse-CDF selection: order each group (or token) in the vocabulary according to a keyed pseudo-random permutation $R _ { a _ { t } , u }$ with an interval of length $M _ { t } ( u )$ , and emit the group whose interval contains the watermark score $\tilde { U } _ { a _ { t } }$ . See Figure 19.

TransportRace and QuotaTransport This is the second black-box family, and the only one that couples the transformation to the Channels. Draw L tokens from the model, compute the keyed score $E _ { j , a _ { t } , \mathcal { G } ( V _ { i } ) }$ of every (channel, proposal) pair, and choose the permutation $\sigma _ { * }$ maximizing $\begin{array} { r } { \sum _ { j } E _ { j , a _ { t } , \mathscr { G } ( V _ { \sigma ( j ) } ) } } \end{array}$ . QuotaTransport stratifies the proposal pool before the same assignment, drawing one proposal from each of the $L$ mass intervals $( i + W ) / L$ for a fresh $W \sim \mathcal { U } ( [ 0 , 1 ] )$ , so that the pool covers the distribution rather than concentrating on its mode. Both were evaluated at $L \in \{ 3 2 , \bar { 1 2 } 8 \}$ against the Gamma detector with Bonferroni correction. As we see in Figure 19, all four configurations substantially improve quality, but they are less robust than the Gumbel race.

ResidualRace This is the extension of the Gumbel race described in Sec. 4.2. We keep one residual clock $R _ { a _ { t } , u }$ per (context, group) pair, initialized at  log $\tilde { U } _ { a _ { t } , u }$ on first use. At each step, we sample the next token according to the exponential race:

$$
\tau _ { t } = \operatorname* { m i n } _ { u } \frac { R _ { a _ { t } , u } } { M _ { t } ( u ) } , \qquad u _ { * } = \arg \operatorname* { m i n } _ { u } \frac { R _ { a _ { t } , u } } { M _ { t } ( u ) } , \qquad R _ { a _ { t } , u } \gets R _ { a _ { t } , u } - \tau _ { t } M _ { t } ( u ) ,
$$

![](images/268d0f1e71428da97d8ec9d19026b355fd51d295cfc3299ab449a42eb8646920.jpg)  
Figure 20: Comparison of scalar detectors without a model forward pass (left) and with a forward<sup>1</sup> pass (right). Quality is omitted because changing the detector does not change generation.

and replace the winning token’s score with a fresh exponential random variable: $R _ { a _ { t } , u _ { * } } \sim \mathrm { E x p } ( 1 )$ Because of the memorylessness property of the exponential, all losing residual exponentials are independent given the past, so the next winning probability is still $M _ { t } ( u )$ even when a context recurs and its masses have changed. This is what makes it sound to re-use previous clocks in the exponential race when a context re-occurs (Proposition 5). As we show in Figure 19, it increases detectability at the cost of quality. Yet, as we have shown in Figure 5, adding Gaussian copula noise to the initial clocks recovers the quality cost while keeping the detectability gains.

Content gate and Interleaved gate This is the skeleton family logits transformation, which relies on either the Last important or Interleaved context seeding. Let  be the important groups, and $\begin{array} { r } { r _ { t } = \sum _ { u \in \mathcal { C } } M _ { t } ( u ) } \end{array}$ . At each step, draw a fresh $B _ { t } \sim \mathrm { B e r n o u l l i } ( r _ { t } )$

1. The Content gate uses the Gumbel race within the important groups when $B _ { t } = 1$ and samples conditionally from the fillers otherwise.

2. The Interleaved gate uses a Gumbel race inside whichever group the gate selects (e.g., either the important or filler group), seeding the Gumbel race with the corresponding group context.

As we see in Figure 19, the Interleaved gate strictly outperforms the Content gate, though both show limited robustness.

## A.6 DETECTION

Scalar Gamma and Gaussian The Gamma detector sums exponential scores,

$$
S ( \omega ) = \sum _ { t \in D } - \log ( 1 - U _ { a _ { t } , \omega _ { t } } ) \sim \mathrm { G a m m a } ( N , 1 ) .
$$

The Gaussian detector sums the normal scores instead,

$$
S ( \omega ) = N ^ { - 1 / 2 } \sum _ { t \in D } \Phi ^ { - 1 } ( U _ { a _ { t } , \omega _ { t } } ) \sim \mathcal { N } ( 0 , 1 ) .
$$

Both detectors weight the evidence differently: unlike the Gaussian detector, the Gamma detector gives more weight to the tail of the distribution, i.e., to the tokens with very high watermarking scores. As we see in Figure 20, the global Gamma detector significantly outperforms the Gaussian detector without changing the generation algorithm.

Gamma Clustering and Excess threshold Both detectors are refinements over the Gamma detector that group the evidence by context before aggregating. Let $\mathcal { H } ( \omega )$ be the set of unique contexts in ω and $\bar { D } _ { a } ( \bar { \omega } )$ the distinct successors of context $^ { a , }$ with $m _ { a } = | D _ { a } ( \omega ) |$ .

1. Clustering calibrates each context separately, and then aggregates the calibrated values,

$$
\begin{array} { l } { \displaystyle S _ { a } ( \omega ) = \sum _ { v \in D _ { a } ( \omega ) } - \log ( 1 - U _ { a , v } ) , } \\ { \displaystyle S ( \omega ) = \sum _ { a \in \mathcal { H } ( \omega ) } - \log Q ( m _ { a } , S _ { a } ( \omega ) ) \sim \mathrm { \mathrm { G a m m a } } ( | \mathcal { H } ( \omega ) | , 1 ) , } \end{array}
$$

so that a context with many successors cannot dominate the statistic on its own. Indeed, during generation, all successors of a given context take part in the same race, which means that two observations of the same race $( \mathrm { i . e . }$ , context) carry less information than two observations of distinct races. Hence, even though the detection is sound as long as no (context, candidate) pair is repeated, it should account for this loss of information whenever a context is repeated.

2. Excess detection keeps only the part of each $- \log Q ( m _ { a } , S _ { a } ( \omega ) )$ ) above the $\exp ( 1 )$ median,

$$
S ( \omega ) = \sum _ { a } \mathrm { m a x } ( - \log Q ( m _ { a } , S _ { a } ( \omega ) ) - \log 2 , 0 ) ,
$$

which is zero with probability one half under the null and $\exp ( 1 )$ otherwise, giving the exact tail $\begin{array} { r } { R _ { H } ( s ) = \sum _ { b = 1 } ^ { H } { \binom { H } { b } } 2 ^ { - H } Q ( b , s ) } \end{array}$ for $H = | \mathcal { H } |$ . This pushes the idea of the Gamma detector even further: only the tail of the distribution carries reliable watermarking signal.

As we see in Figure 20, both approaches indeed slightly increase detection power without changing the generation algorithm.

Weighted Gamma and Position evidence Those two detectors are additions on top of the Gamma detector that require a model forward pass at detection, and hence are considerably more costly. Let $\hat { p } _ { t }$ be the estimated next-token distribution given $\omega _ { < t } .$

1. Weighted Gamma weights each exponential score by the observed token’s surprisal quantized to quarter-unit,

$$
\eta _ { t } = 1 + \frac { 1 } { 4 } \operatorname { r o u n d } \left( 4 \operatorname* { m i n } ( \operatorname* { m a x } ( - \log \hat { p } _ { t } ( \omega _ { t } ) , 0 ) , 2 . 5 ) \right)
$$

and uses as a detection statistic

$$
S ( \omega ) = \sum _ { t \in D } - \eta _ { t } \log ( 1 - U _ { a _ { t } , \omega _ { t } } ) .
$$

The intuition is that a high-entropy position is where the race had a real choice, so its score carries more information than a position where the model was nearly deterministic.

2. Position evidence goes further and reconstructs the race itself from the observed token’s logprobability and those of up to eight competitors. At position $t ,$ let $C _ { t }$ contain the retained competitors other than $\omega _ { t } .$ . It then reconstructs the Gumbel race outcome:

$$
\alpha _ { t , v } = \operatorname* { m i n } \left( 1 0 ^ { 4 } , \frac { \hat { p } _ { t } ( v ) } { \hat { p } _ { t } ( \omega _ { t } ) } \right) , \qquad \lambda _ { t } = \sum _ { v \in C _ { t } } \alpha _ { t , v } , \qquad W _ { t } = 1 \big \{ U _ { a _ { t } , v } < U _ { a _ { t } , \omega _ { t } } ^ { \alpha _ { t , v } } \mathrm { ~ f o r ~ a l l ~ } v \in C _ { t } \big \} ,
$$

where $W _ { t }$ indicates whether the observed token wins the reconstructed race. Its exact race-win tail is

$$
R _ { t } = \left\{ \begin{array} { l l } { \displaystyle { 1 - U _ { a _ { t } , \omega _ { t } } ^ { 1 + \lambda _ { t } } } , } & { W _ { t } = 1 , } \\ { \displaystyle { 1 + \lambda _ { t } } , } & { } \\ { \displaystyle { 1 - U _ { a _ { t } , \omega _ { t } } } + \frac { U _ { a _ { t } , \omega _ { t } } ^ { 1 + \lambda _ { t } } } { 1 + \lambda _ { t } } , } & { W _ { t } = 0 . } \end{array} \right.
$$

Under the null, $\operatorname* { P r } ( W _ { t } = 1 \mid U _ { a _ { t } , \omega _ { t } } = u ) = u ^ { \lambda _ { t } }$ and $R _ { t } \sim \mathrm { U n i f o r m } ( 0 , 1 )$ (Proposition 7). When no competitor survives, $\lambda _ { t } = 0$ and $W _ { t } = 1$ , giving $R _ { t } = 1 - U _ { a _ { t } , \omega _ { t } }$ . Keeping only the first

retained candidate for each context (i.e., deduplicating occurrences of the context), denoted by $D _ { \mathrm { p o s } }$ , avoids reusing the same race and gives the statistic

$$
S ( \omega ) = \sum _ { t \in D _ { \mathrm { p o s } } } - \log R _ { t } \sim \mathrm { G a m m a } ( | D _ { \mathrm { p o s } } | , 1 ) .
$$

It can also be combined with the weighted variant.

As we have shown in Sec. 4.2, and see in Figure 20, the weighted Position evidence detector is significantly more powerful than the versions that do not require a forward pass.

Quantile detector This is the detector matched to QuantileRace. Let $R _ { a _ { t } , u }$ be the keyed ordering score and $\tilde { U } _ { a _ { t } }$ the keyed offset, it computes $\begin{array} { r } { S ( \omega ) = \sum _ { ( a , u ) \in D } \cos \ ( 2 \pi ( R _ { a _ { t } , u } - U _ { a _ { t } } ) ) } \end{array}$ and calibrates it by Monte Carlo, since the statistic has no closed-form null.

Quantile + wheel This is the rotation-invariant detector required by the Wheel. Form the angles $A _ { a _ { t } , u } = 2 \bar { \pi } ( ( R _ { a _ { t } , u } \bar { \ } - U _ { a _ { t } } )$ mod 1) and their resultant $\begin{array} { r } { S = \| ( \sum _ { D } \cos A _ { a _ { t } , u } , \sum _ { D } \sin A _ { a _ { t } , u } ) \| } \end{array}$ , then report an e-value rather than a p-value. With $\kappa =$ $2 \sqrt { \log ( 2 0 0 0 ) / N }$ , the quantity $\mathcal { E } = I _ { 0 } ( \kappa S ) / I _ { 0 } ( \kappa ) ^ { N }$ has unit expectation under the null, so $\begin{array} { r l } { \dot { P } } & { { } = } \end{array}$ min $( 1 , 1 / \mathcal { E } )$ is valid. As we see in Figure 21, the Wheel slightly improves quality but almost entirely removes the robustness of QuantileRace.

![](images/ae088351572305570b9a1d2d098f439894b7769ef03237055019067dbc317bdc.jpg)

Predictive Gamma This variant of the Gamma detector applies to all the private-geometry scores.

Enumerate the retained pairs in D as $( a _ { i } , u _ { i } ) _ { i = 1 } ^ { N }$ and write $Z _ { i } = Z _ { a _ { i } , u _ { i } }$ . Let $A _ { 0 } , \ldots , A _ { K - 1 }$ be a fixed public dictionary of candidate geometries $( K = 1 2 8$

Figure 21: Impact of the Wheel compared to 1QuantileRace with the Quantile detector.

for the Signed sphere and Unoriented axis, and $K = 5 1 2$ for Complex ray and Rotation), with weights $w _ { i , j }$ measuring how well $A _ { j }$ explains the scores observed before step i. Let $B _ { i }$ denote the estimate of A formed from the weights $w _ { i , j }$ before observing $Z _ { i } .$ . Write $\tilde { U } ( B , Z )$ for the corresponding geometric score from App. A.4, evaluated with geometry B in place of A, and define $d ( B , Z ) = 1 - \tilde { U } ( B , Z )$ . At step i, compute $d _ { i } = d ( B _ { i } , Z _ { i } )$

After scoring $Z _ { i } ,$ , update and normalize the weights as $w _ { i + 1 , j } \propto w _ { i , j } d ( A _ { j } , Z _ { i } ) ^ { - 1 / 2 }$ , then compute

$$
B _ { i + 1 } = \left\{ \begin{array} { l l } { \displaystyle { \sum _ { j = 0 } ^ { K - 1 } w _ { i + 1 , j } A _ { j } } } & { \mathrm { s i g n e d ~ s p h e r e } , } \\ { \displaystyle { \left\| \sum _ { j = 0 } ^ { K - 1 } w _ { i + 1 , j } A _ { j } \right\| } ^ { \prime } } & { \mathrm { s i g n e d ~ s p h e r e d ~ a r i s ~ o r ~ r o t a t i o n } , } \\ { \displaystyle { \mathrm { e i g } _ { \operatorname* { m a x } } \left( \sum _ { j = 0 } ^ { K - 1 } w _ { i + 1 , j } A _ { j } A _ { j } ^ { \top } \right) } , } & { \mathrm { u n o r i e n t e d ~ a x i s ~ o r ~ r o t a t i o n } , } \\ { \displaystyle { \mathrm { e i g } _ { \operatorname* { m a x } } \left( \sum _ { j = 0 } ^ { K - 1 } w _ { i + 1 , j } A _ { j } A _ { j } ^ { * } \right) } , } & { \mathrm { c o m p l e x ~ r a y } , } \end{array} \right.
$$

where $\mathrm { e i g } _ { \mathrm { m a x } }$ returns a unit eigenvector associated with the largest eigenvalue and denotes the conjugate transpose. Under the null, each fresh score $Z _ { i }$ is independent of the earlier scores, and the matching geometric transform makes $d _ { i }$ uniform on [0, 1] conditional on them. Thus,

$$
S ( \omega ) = \sum _ { i = 1 } ^ { N } - \log d _ { i } \sim \mathrm { G a m m a } ( N , 1 ) .\tag{12}
$$

We prove this in Proposition 6.

![](images/45f3106c7d998f4a1ea6c1107288adbd94bba764407891344d54cb693e0816c1.jpg)

![](images/a0773e460ff06219389d40a0338ad316ac5994400e9e56b19b6e803bd41ed718.jpg)  
Figure 22: Comparison of<sup>1</sup> Signed sphere detectors on the same generated texts. Bars show the TPR@1%FPR difference to Predictive Gamma in percentage points.  
Figure 23: TPR differences to<sup>1</sup> Gaussian norm for each attack. Colors indicate dimension $d ;$ solid bars show Residual energy and hatched bars show Circle search $( d \ = \ 2 )$ Parentheses give the Gaussian norm reference TPR@1%FPR for $d = 1 / 2 / 4 / 8$ , respectively.

Resultant length, Spherical caps, and Anchor distance These detectors test whether the Signed sphere scores concentrate around a common direction, without knowing the private direction A. As above, write $Z _ { i } = Z _ { a _ { i } , u _ { i } }$ for the scores associated with the N retained pairs in D.

1. Resultant length measures the alignment of the scores through $R = \textstyle \| \sum _ { i = 1 } ^ { N } Z _ { i } \|$ . Aligned scores produce a larger resultant, whereas scores pointing in different directions tend to cancel. The detector compares R with its null distribution for N independent uniform points on $S ^ { 2 }$ making it the spherical analogue of the Gaussian norm detector below.

2. Spherical caps measures local concentration around $m = \operatorname* { m i n } ( N _ { \mathbf { \alpha } }$ , 16) anchor scores. For each anchor $Z _ { j }$ and area fraction $q \in \{ 1 / 1 6 , 1 / 8 , 1 / 4 \}$ , it counts $\begin{array} { r } { C _ { j , q } = \sum _ { i \neq j } \Im \{ Z _ { j } \cdot Z _ { i } \ge 1 - 2 q \} } \end{array}$ Under the null, conditional on $Z _ { j }$ , this count follows $\mathrm { B i n } ( N - 1 , q )$ . The detector takes the smallest upper-tail p-value and applies a Bonferroni correction for the 3m anchor–cap combinations.

3. Anchor distance replaces cap membership with a continuous measure of alignment. For each anchor $Z _ { j }$ , it computes $\begin{array} { r } { S _ { j } = \mathbf { \bar { \sum } } _ { i \neq i } - \log ( ( 1 - Z _ { j } \cdot Z _ { i } ) / 2 ) } \end{array}$ , so that scores closer to the anchor contribute more evidence. Under the null, conditional on $Z _ { j }$ , the transformed inner products are independent uniforms, giving S<sub>j</sub> Gamma( $N - 1 , 1 )$ . Searching over all $N$ anchors gives the corrected p-value $P ( \omega ) \mathrm { \bar { = } } \operatorname* { m i n } \mathrm { \bar { \{ 1 , } }  N$ min<sub>j</sub> $Q \big ( N - 1 , \dot { S } _ { j } ) \big \}$

Figure 22 compares these detectors to the Predictive Gamma detector. We find that the Predictive Gamma detector outperforms each of those geometry-specific detectors.

Gaussian norm, Circle search, and Residual energy These detectors apply to the Gaussian direction scores. We write $Z _ { i } \doteq Z _ { a _ { i } , u _ { i } }$ for the scores associated with the N retained pairs in D.

Table 2: Evaluation of invalid agent schemes. Each TPR cell reports the result from the prior harness the scheme was designed in / from our current harness (App. D), whereas quality metrics only come from our current harness. Verified reports whether the scheme passes our current verification.
<table><tr><td></td><td></td><td colspan="4">Robustness TPR@1 ↑</td><td colspan="3">Quality deviation (%)↓</td><td></td></tr><tr><td>Scheme</td><td>Clean TPR@1 ↑ Deletion Substitution Back-trans. Paraphrase</td><td></td><td></td><td></td><td></td><td>∆PPL ∆SB-2</td><td></td><td>∆SB-3 Verified</td><td></td></tr><tr><td>ConcordMark-v0</td><td>100 / 100</td><td>86/ 83</td><td>99/ 96</td><td>99/ 99</td><td>91 / 90</td><td>5.7</td><td>8.9</td><td>23</td><td>x</td></tr><tr><td>CathedralMark</td><td>100 / 100</td><td>100 / 85</td><td>100 / 97</td><td>100 / 98</td><td>99/91</td><td>9.6</td><td>9.2</td><td>23</td><td>x</td></tr><tr><td>OverlayMark</td><td>100 / 100</td><td>100 / 84</td><td>100 / 97</td><td>100 / 99</td><td>100 / 92</td><td>7.1</td><td>9.4</td><td>24</td><td>x</td></tr><tr><td>MeridianMark</td><td>99/ 98</td><td>88/ 88</td><td>62/74</td><td>94/93</td><td>77/78</td><td>65</td><td>23</td><td>60</td><td>x</td></tr></table>

1. Gaussian norm: rather than searching for the private direction A, it integrates it out through the norm of the summed scores,

$$
S ( \omega ) = \frac { 1 } { N } \left\| \sum _ { i \in D } Z _ { i } \right\| ^ { 2 } \sim \chi _ { d } ^ { 2 } .\tag{13}
$$

Circle search and Residual energy extend this detector.

2. Circle search searches explicitly for the private direction A in dimension $d = 2$ . For each candidate direction $B ( \theta ) = \bar { ( } \cos \theta , \sin \theta )$ , it computes the Gamma statistic

$$
\begin{array} { c } { { S ( \theta ) = \displaystyle \sum _ { i = 1 } ^ { N } - \log \Phi ( - B ( \theta ) \cdot Z _ { i } ) , } } \\ { { T = \displaystyle \operatorname* { m a x } _ { \theta \in [ 0 , 2 \pi ) } S ( \theta ) . } } \end{array}
$$

Under the null, $S ( \theta ) \sim \mathrm { G a m m a } ( N , 1 )$ for each fixed $\theta ,$ but maximizing over directions requires a correction. The detector uses the continuous-search bound of Davies (1987) (App. I.5)

$$
P _ { 0 } = \operatorname* { m i n } \{ 1 , Q ( N , T ) + 2 \sqrt { \pi T } f _ { N } ( T ) \} ,
$$

where $f _ { N }$ is the density of Gamma $( N , 1 )$

3. Residual energy measures the dispersion of the scores around their mean, in addition to their alignment. Let $\begin{array} { r } { \bar { Z } = N ^ { - 1 } \sum _ { i = 1 } ^ { N } Z _ { i } } \end{array}$ . The two statistics are

$$
\begin{array} { c } { { S _ { \mathrm { m e a n } } = N \displaystyle \lVert \bar { Z } \rVert ^ { 2 } , } } \\ { { S _ { \mathrm { r e s } } = \displaystyle \sum _ { i = 1 } ^ { N } \lVert Z _ { i } - \bar { Z } \rVert ^ { 2 } . } } \end{array}
$$

Under the null, they are independent, with $S _ { \mathrm { m e a n } } \sim \chi _ { d } ^ { 2 }$ and $S _ { \mathrm { r e s } } \sim \chi _ { d ( N - 1 ) } ^ { 2 }$ . For $N > 1$ , let p<sub>mean</sub> and $p _ { \mathrm { r e s } }$ be their respective upper-tail p-values. The detector combines them through $S ( \omega ) = - \log p _ { \mathrm { m e a n } } - \log \bar { p } _ { \mathrm { r e s } } \sim \mathrm { G a m m a ( 2 , \bar { 1 } ) }$ under the null, giving $P _ { 0 } = Q ( 2 , S ( \omega ) )$ . For $N = 1$ , it uses only the Gaussian norm statistic.

As we see in Figure 23, both extensions improve over Gaussian norm, with Circle search giving the largest gains at $d = 2 .$ , but the gains of Residual energy shrink as d grows and vanish at $d = 8$

## B EXAMPLES OF INVALID AUTORESEARCH SCHEMES

In this section, we present 4 schemes that research agents designed using earlier iterations of our harness, whose verification was weaker than that described in Sec. 3.3. All four schemes appear to outperform prior work, yet they either exploit gaps in the evaluation process or have an abnormally high false positive rate. We re-evaluate them with our harness in Table 2: they show impressive robustness, but none passes our verification. Nevertheless, we believe that some of their ideas are interesting and could be explored in future work.

Prior harnesses ConcordMark-v0, CathedralMark, OverlayMark, and MeridianMark come from a harness with neither private verification nor private evaluation (i.e., the agent could see the whole evaluation dataset). Furhtermore, the verifier only checked distortion-freeness and the false positive rate on 5,000 English C4 (Raffel et al., 2020) human texts, using a single fixed watermark key. In particular, it had no check for soundness over the key. Schemes that passed were then evaluated once using the watermark key set in their configuration, i.e., chosen by the agent.

ConcordMark-v0 Despite its name, ConcordMark-v0 is unrelated to ConcordMark (App. H.1); we simply kept the independently submitted names for both. Its generation is a Gumbel race with a one-token context (i.e., Fixed-length context seeding with $k = \bar { 1 } )$ . Its detector, however, combines two p-values: $P _ { w } ,$ the rank of the statistic under the private key among 1023 decoy keys (i.e., a Monte Carlo estimate of the p-value), and $P _ { c } , \mathbf { a }$ p-value derived from a zero-shot LLM-text classifier, specifically an n-gram classifier. The agent trained this classifier to separate its own watermarked texts (clean and attacked) from 2,500 human texts and calibrated it on 2,500 other human texts. A text is flagged as watermarked if $P _ { w } \le 0 . 0 0 2$ or $P _ { c } \leq 0 . 0 0 8$ , so that, by the union bound, the false positive rate on human texts is at most 1%. This significantly improved the scheme’s robustness (Table 2).

Yet, the classifier learned to recognize LLM-generated text in general rather than watermarked LLM-generated text, e.g., on unwatermarked replies from WildChat, it wrongly flags 31% of the texts as watermarked at a supposedly 1% FPR. This shows the importance of soundness over the key. Indeed, P does not depend on the key, so the scheme violates soundness over the key (Equation (2)) (a text flagged by the classifier is flagged under every key).

CathedralMark CathedralMark is a later scheme from the same run that keeps the detector of ConcordMark-v0 but also steers the generation toward its classifier. Depending on a keyed latent state, it either tilts the next-token distribution toward the tokens that increase the classifier score or applies a compensating tilt, so that, in expectation over the key, the next-token distribution remains p<sub>t</sub>. Hence, the scheme passes the distortion-freeness test but, like ConcordMark-v0, fails the soundness over the key check. Nonetheless, the idea of steering watermark generation toward easily detectable LLM patterns is particularly robust: it achieves more than 90% robustness to paraphrasing.

OverlayMark For OverlayMark, a scheme similar to CathedralMark, the agent carefully selected a watermark key that maximized TPR@1%FPR on the evaluation dataset because the evaluation was public. Hence, as shown in Table 2, it achieves 100% paraphrasing robustness in the original harness’s evaluation, but this falls to 92% in our updated harness, which varies the key (we take the worst TPR@1%FPR out of 5 random keys).

MeridianMark MeridianMark uses a Gumbel race seeded by the last 4 tokens mapped to 32 semantic clusters. Its detector combines an entropy-weighted Gamma statistic and a tail statistic, and computes a p-value by fitting a Cauchy distribution to the statistics of 1024 decoy keys. It then remaps the p-values: every p-value below 0.0066 is linearly mapped to [0, 0.001]. The agent chose this threshold just below the fifth-smallest p-value of the public human texts (0.0069), and documented it explicitly in its submission: “evaluate never measures human FPR [. . . ] exactly 4 humans fall under it (one-text margin)”. This shows that, if verification is public, the agent can exploit the metric to pass verification incorrectly. This is why our final harness uses private verification.

Takeaways Each of the above schemes exploits a gap in the harness: a human null corpus that does not cover LLM-generated text (ConcordMark-v0 and CathedralMark), a watermark key chosen by the agent (OverlayMark), or the ability to exploit public verification (MeridianMark). Our harness closes these gaps: our private verification corpus includes LLM-generated replies (App. D.1), we evaluate each scheme with 5 private keys and report the worst result (App. D.2), and our soundness over the key test only requires a single text on which the detector is miscalibrated over the key to reject a scheme, independently of how well the detector is calibrated on any specific human corpus.

## C STUDYING THE SOUNDNESS GIVEN A KEY

In this section, we investigate the effect of enforcing soundness given a key (Sec. 3.2). In Figure 2, we plotted the autoresearch trajectories of each agent independently, and as we explain they tackled soundness given a key differently which makes the comparison uneven. Namely, GPT-6 ASTRA multiplies its p-values by $1 / \beta = 2 0$ to be provably sound given a key, OPUS 5 applies smaller safety margins that it calibrated locally, and GEMINI-3.8 FLASH and the baselines apply none. For the baselines in particular, in the settings of Figure 2, neither AAR nor SynthID passes our verification of being sound given a key.

![](images/1e76199afc09e50120a77958d401f980f560416ca82f3b5f478ee56582bed199.jpg)  
Figure 25: Paraphrasing robustness of the submissions of each agent under the three p-value cor-<sup>1</sup> rections, where each dot corresponds to a submission and the solid lines are the best value so far. The baselines are in italic, and a parenthesis means the baseline fails our soundness given a key test without correction.

Experimental setup We rescore the submissions of the four agents from Figure 2 and the three baselines. For each scheme, we remove the explicit outer p-value multiplier chosen by the agent (if any), and keep the rest of the detector unchanged (e.g., the Bonferroni correction over Channels) so it remains sound over the keys. We then evaluate three settings:

• Guaranteed: we multiply the p-value by $1 / \beta = 2 0$ as in Equation (14). By Proposition 1, any scheme that is sound over the key becomes sound given a key.

• Empirical: we multiply the p-value by the smallest multiplier $\kappa \geq 1$ such that the scheme passes both soundness tests (Algorithms 2 and 3), with the same thresholds, sample sizes and significance levels as in Sec. 3.3.

• No correction: we use the raw p-values $( \kappa = 1 )$ and do not enforce the soundness given a key test.

In all settings, the scheme must still be distortion-free and sound over the key. We then measure the paraphrasing TPR@1%FPR with the protocol of Sec. 4.

Baselines fail without correction In Figure 24, we show the smallest multiplier each scheme requires to pass our soundness tests. Among the baselines, only TextSeal passes without correction. With raw p-values, both AAR and SynthID fail the soundness given a key test. Similarly, 23 of the 52 agent submissions would fail without their outer multiplier. In fact, both GPT-6 ASTRA (low) and OPUS 5 only started adding such multipliers after their first submissions were rejected by our soundness given a key test.

The guaranteed correction is overly conservative Yet, the multiplier required in practice is much smaller than $1 / \beta \colon$ at most 1.08 for all but one submission. This is expected: if the watermark pseudo-random scores are diverse enough over a corpus given a fixed key, by soundness over the key the expected false-positive rate will converge to be below α which satisfies soundness given a key. This means that correcting with $1 / \beta$ is overly conservative in many cases, costing significant power. This is why we opted to not correct the schemes in our main evaluation in Sec. 4.1 unless the agents did so in their submissions.

![](images/bce51e28d71dbe30f0db042787ab6e84b6aa60f0457d07101328b19f21870622.jpg)  
Figure 24: Smallest p-value multiplier passing our soundness tests.

Impact on robustness We show in Figure 25 the paraphrasing robustness trajectories in each setting. The guaranteed correction costs on average 16.3 percentage points of paraphrasing TPR compared to the empirical correction, whereas the empirical correction costs at most 2.5 percentage points compared to no correction. Importantly, the conclusions of Sec. 4.1 hold in all three settings: all agents improve over time and outperform all baselines, with OPUS 5 and GPT-6 ASTRA (xhigh) reaching the highest robustness. In particular, in the empirical setting, the best GPT-6 ASTRA (xhigh) submission reaches 83% paraphrasing TPR, compared to 70% in the guaranteed setting, closing most of the gap with OPUS 5 (86%).

## D EXPERIMENTAL SETUP

In this section, we detail the experimental setup used in our evaluation (Sec. 4). Our harness distinguishes two configurations: the public configuration, which is given to the research agent and which it can run locally, and the private configuration, which is only used by the grader when the agent submits a scheme (Sec. 3.4).

## D.1 WATERMARK VERIFICATION

To verify each watermarking scheme, we run the three statistical tests from Sec. 3.3 on a corpus of non-watermarked texts. A scheme is valid only if it passes all three tests.

Corpora The public corpus consists of the answers of the ELI5 (Fan et al., 2019) dataset. The private corpus is a mixture where half of the texts come from WildChat (Zhao et al., 2024a) and the other half from Wikipedia articles, evenly split across 10 languages (English, German, French, Spanish, Chinese, Russian, Arabic, Hindi, Japanese, and Swahili). For WildChat, we take in each conversation the first assistant reply that has at least 512 tokens (i.e., non-watermarked LLMgenerated text), across all 86 shards of WILDCHAT-4.8M. For Wikipedia, we use the first shard of the 2023-11-01 dump of each language. In both corpora, we tokenize the texts with the LLAMA-3.1-8B-INSTRUCT tokenizer, discard texts shorter than 512 tokens, and truncate the others to their first 512 tokens. Whenever a test samples texts from the corpus, it samples them from this mixture, according to the weights of each component.

Soundness tests For soundness over the key (Algorithm 2), we sample 200 fixed reference texts from the corpus and draw m keys per text, with $m = 1 0 , 0 0 0$ in the public configuration and $m = 2 0 { , } 0 0 0$ in the private one. For soundness given a key (Algorithm 3), we draw 1000 keys and, for each key, sample 20,000 fresh texts from the corpus (with replacement). We set the allowed bad-key fraction to $\beta = 5 \%$ with a single screening level of 1%. In the private configuration, both tests check the thresholds $\alpha \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 0 5 \}$ , whereas the public configuration only checks $\alpha \in \{ 0 . 0 1 , 0 . 0 5 \}$ . The two soundness tests share a significance level of 0.001, split equally between them $( \mathrm { i . e . , } \delta = 0 . 0 0 0 5$ in Algorithms 2 and 3).

Distortion-freeness test For distortion-freeness (Algorithm 1), we sample 4 texts from the corpus and use their first 128 tokens as contexts. We compute the next-token distributions of $\mathrm { L L A M A } { - } 3 . 2 .$ 1B-INSTRUCT (temperature 1, top-k 50) on each context, and average the watermarked distributions over 2048 keys, testing the 16 most likely tokens (and the remaining mass). We set the significance level to $\delta = \dot { 0 } . 0 0 1$ . As in Algorithm 1, if the scheme leaves all 4 distributions unchanged, the test is inconclusive and the scheme is rejected.

Table 3: Research agents from Figure 2.
<table><tr><td>Agent</td><td>Interface</td><td>Version</td><td>Reasoning</td><td>Submissions</td></tr><tr><td>GPT-6 ASTRA (xhigh)</td><td>Codex CLI</td><td>0.153.3</td><td>xhigh</td><td>14</td></tr><tr><td>GPT-6 ASTRA (low)</td><td>Codex CLI</td><td>0.153.3</td><td>low</td><td>11</td></tr><tr><td>OPUS 5</td><td>Claude Code</td><td>2.1.231</td><td>high</td><td>16</td></tr><tr><td>GEMINI-3.8 FLASH (high)</td><td>Antigravity CLI</td><td>1.1.25</td><td>high</td><td>11</td></tr></table>

## D.2 WATERMARK EVALUATION

The public and private evaluations share the same protocol, which we describe first, and differ in the watermark keys, prompts, and sampling seeds.

Generation We sample 1000 prompts from the questions of the train split of ELI5 (Fan et al., 2019), keeping only questions of more than 100 characters and removing duplicates. We generate the replies with LLAMA-3.1-8B-INSTRUCT, prompting it with the raw question. We sample with temperature 1, top-k 50, and top-p 1, and force replies of 200 to 300 tokens.

Detection and robustness To measure watermark strength (either before or after the attacks), we report the TPR@1%FPR, i.e., the fraction of the 1000 replies for which the p-value returned by the detector is below 0.01. Because the verification (App. D.1) screens the p-values for soundness, we do not calibrate the threshold empirically. For robustness, we use four attacks. For 50% word deletion, each whitespace-separated word is deleted independently with probability 0.5. For 50% synonym substitution, we randomly replace half of the words (or all replaceable words if there are fewer) with a random WordNet synonym. For back-translation through French and paraphrasing, we use GPT-5.4-MINI (gpt-5.4-mini-2026-03-17) with the default sampling parameters of the OpenAI API and the prompts from App. J.2.

Quality To measure the impact on quality, we use three different metrics. We compute the perplexity of each of the 1000 replies with LLAMA-3.1-8B-INSTRUCT, on the reply alone (i.e., not conditioned on the prompt), and average the perplexities over the replies. We also measure the diversity of replies to a single prompt: we sample 10 additional prompts from ELI5 and generate 100 replies for each prompt, following the protocol established in Gloaguen et al. (2026). For each reply, we compute the Self-BLEU (bigram and trigram) score on token IDs, using the 99 other replies to the same prompt as references, and average the scores over the 1000 replies. We then compare each quality metric to that of the unwatermarked model, evaluated with the same protocol. Then, we report the relative distance to the unwatermarked model (in percent), and, when we report an aggregated quality metric, we average the relative distances of the three metrics.

Public and private evaluation In the public configuration, the agent evaluates its scheme locally with its own watermark key, and the prompts are always the first 1000 (and 10 for diversity) ELI5 questions after shuffling the dataset with seed 0. In the private configuration, the grader evaluates each submission with 5 private watermark keys. For each key, the grader derives from the key the seeds used to shuffle ELI5 (separately for the detection and diversity prompts) and the generation seed, so that the private prompts differ from the public ones and across keys. The grader then reports, for each metric, the worst result over the 5 keys: the minimum TPR and the maximum distance to the unwatermarked model. Only the aggregated metrics and the verification verdicts are returned to the agent. The private evaluation runs in an isolated container.

## D.3 RESEARCH AGENTS

We summarize the research agents in Table 3. We run all agents with the prompt of App. J.1, in their own command-line interface and with subagents enabled. When an agent exits, the harness starts a new agent in the same container and Git checkout, and the loop stops once the submission budget is exhausted and all submissions are graded.

Environment Each agent runs in a persistent Docker container with one NVIDIA RTX PRO 6000 Blackwell GPU (96 GB), 48 CPU cores, and full Internet access. The container holds the watermarking library (with the baselines of Sec. 3.4), the public configurations, and API keys for third-party AI providers (intended to be used for e.g., paraphrasing attacks). The grader runs in a separate container, which does not share any storage with the agent container, and on a different GPU.

## D.4 BASELINES

Here, we first detail how the three baselines (AAR, SynthID, and TextSeal) work using the notation from Sec. 3.1 and then go through their hyperparameters. Recall that $\omega \in \Sigma ^ { * }$ is a sequence of tokens, $p _ { t } \in \Delta ( \Sigma )$ the conditional probability distribution of the next token given $\omega _ { < t }$ . We assume there exists a sequence $\left( U _ { a _ { t } } \right)$ of pseudo-random vectors in $[ 0 , 1 ] ^ { | \Sigma | }$ seeded by the context hash $a _ { t }$ whose entries are i.i.d. uniform variables. The watermark is defined by $f : [ 0 , 1 ] ^ { | \Sigma | } \times \Delta ( \Sigma )  \Delta ( \Sigma )$ , and $S ( \omega )$ is the detector statistic.

AAR With AAR, the logits transformation is simply a Gumbel race, and the next token is sampled according to

$$
\omega _ { t } = \arg \operatorname* { m a x } _ { v \in \Sigma } \left( - \log ( - \log U _ { a _ { t } , v } ) + \log p _ { t } ( v ) \right) .
$$

Because $- \log ( - \log ( U _ { a _ { t } , v } ) )$ are i.i.d. Gumbel random variables, using the Gumbel-max trick we can show that in expectation over $U _ { a _ { t } }$ the next token $\omega _ { t }$ is sampled according ${ \mathrm { t o ~ } } p _ { t }$ . For detection, we use the one-sided Kolmogorov-Smirnov test against the uniform null, with the statistic

$$
S ( \omega ) = \operatorname* { m a x } _ { 1 \leq i \leq N } \left( U _ { ( i ) } - \frac { i - 1 } { N } \right) ,
$$

where $U _ { ( i ) }$ are the ordered scores $\left( U _ { a _ { t } , \omega _ { t } } \right)$

SynthID With SynthID, the logits transformation corresponds to a repeated tournament sampling with L layers and C leaves. Instead of a single watermark score, SynthID uses L independent watermark scores $U _ { a _ { t } } ^ { ( 1 ) } , \dots , U _ { a _ { t } } ^ { ( L ) }$ , one per layer, which it binarizes into $g _ { a _ { t } , v } ^ { ( l ) } : = \Im \{ U _ { a _ { t } , v } ^ { ( l ) } > 1 / 2 \}$ Let $p _ { t } ^ { ( l ) }$ be the probability distribution at layer l with $p _ { t } ^ { ( 0 ) } : = p _ { t }$ . Then, at each layer we apply

$$
p _ { t } ^ { ( l ) } ( v ) = p _ { t } ^ { ( l - 1 ) } ( v ) \cdot \left\{ \begin{array} { l l } { \displaystyle \frac { 1 - ( 1 - M _ { t } ^ { ( l ) } ) ^ { C } } { M _ { t } ^ { ( l ) } } } & { \mathrm { i f ~ } g _ { a _ { t } , v } ^ { ( l ) } = 1 , } \\ { ( 1 - M _ { t } ^ { ( l ) } ) ^ { C - 1 } } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \mathrm { w i t h } \quad M _ { t } ^ { ( l ) } : = \sum _ { v \in \Sigma } p _ { t } ^ { ( l - 1 ) } ( v ) g _ { a _ { t } , v } ^ { ( l ) } ,
$$

and the logits transformation is $f ( U _ { a _ { t } } , p _ { t } ) : = p _ { t } ^ { ( L ) }$ . For detection, we use the weighted mean detector over the retained positions $t \in D$ , with statistic

$$
S ( \omega ) = \frac { 1 } { N } \sum _ { t \in D } \sum _ { l = 1 } ^ { L } w _ { l } g _ { a _ { t } , \omega _ { t } } ^ { ( l ) } ,
$$

where $w _ { 1 } , \ldots , w _ { L } \geq 0$ are layer weights summing to 1. Under the null hypothesis, the per-token scores $\textstyle \sum _ { l } w _ { l } g _ { a _ { t } , \omega _ { t } } ^ { ( l ) }$ are i.i.d. with mean $1 / 2$ and variance $\begin{array} { r } { \sigma _ { w } ^ { 2 } : = \frac { 1 } { 4 } \sum _ { l } w _ { l } ^ { 2 } } \end{array}$ . We thus use a normal approximation and return the p-value $1 - \Phi \left( \sqrt { N } \left( S ( \omega ) - 1 / 2 \right) / \sigma _ { w } \right)$ , where Φ is the standard normal CDF.

TextSeal With TextSeal, the logits transformation is the same Gumbel race as AAR, but with two keys: a primary and a secondary key, yielding two watermark scores $U _ { a _ { i } }$ and $U _ { a _ { t } } ^ { \prime }$ . At each step, the watermark score used by the logits transformation is $\tilde { U } _ { a _ { t } } = U _ { a . } ^ { \prime }$ with probability λ and $\tilde { U } _ { a _ { t } } = U _ { a _ { t } }$ otherwise, where the choice is drawn with true randomness. For detection, we fuse the two scores of each retained position and use the weighted statistic

$$
S ( \omega ) = \sum _ { t \in D } w _ { t } \left( - ( 1 - \lambda ) \log ( 1 - U _ { a _ { t } , \omega _ { t } } ) - \lambda \log ( 1 - U _ { a _ { t } , \omega _ { t } } ^ { \prime } ) \right) ,
$$

where the weights $w _ { t } \in [ w _ { \operatorname* { m i n } } , w _ { \operatorname* { m a x } } ]$ increase linearly with the entropy $H _ { t }$ of the next-token distribution of an auxiliary LLM at position t, i.e., $w _ { t } ~ = ~ w _ { \mathrm { m i n } } ~ + ~ ( w _ { \mathrm { m a x } } ~ - ~ w _ { \mathrm { m i n } } ) ( H _ { t } ~ - ~ $ min<sub>t</sub>′ $H _ { t ^ { \prime } } ) / ( \operatorname* { m a x } _ { t ^ { \prime } } H _ { t ^ { \prime } } - \operatorname* { m i n } _ { t ^ { \prime } } H _ { t ^ { \prime } } )$

Watermark hyperparameters All three baselines seed their watermark with a hash of the 3 preceding tokens and the private key, and do not apply the watermark when the context was already seen in the same request. All sample from the top-50 tokens. For AAR (Aaronson, 2023), we use the Gumbel-max sampling, and its detector, a one-sided Kolmogorov–Smirnov test on the de-duplicated Gumbel scores. For SynthID (Dathathri et al., 2024), we use the tournament sampling with 30 layers and 2 leaves, restricted to the top-50 tokens of the unwatermarked distribution, and the weighted mean detector (with weights decreasing linearly from 10 to 1 across layers) with a normal approximation. For TextSeal (Sander et al., 2026), we use a secondary key with probability λ = 0.1, and weights between $w _ { \mathrm { m i n } } = 0 . 1$ and $w _ { \mathrm { m a x } } = 1$ derived from the entropy of GEMMA-3-270M-IT.

## D.5 HELD-OUT EVALUATION

For the held-out evaluation of Sec. 4.3, we use a different protocol, which neither the agents nor our harness used.

Prompts We sample prompts from WILDCHAT-4.8M (Zhao et al., 2024a) using its language labels, English and Chinese. We use the first user turn of conversations whose second turn is an assistant reply, remove duplicate prompts, and discard prompts longer than 3500 tokens. We keep a prompt only if its original WildChat reply has at least 200 tokens with both the LLAMA-3.1-8B-INSTRUCT and the QWEN3.8-27B tokenizers. We then take the first 1000 qualifying prompts of each language after shuffling the dataset shards with a fixed seed. The prompts are the same for all schemes, models, and keys.

Generation We generate the replies with LLAMA-3.1-8B-INSTRUCT and QWEN3.8-27B (with thinking disabled), using their chat template with a single user message and no system prompt. We sample with temperature 1, top-k 50, and top-p 1, and force replies of 200 to 300 tokens.

Metrics We report the TPR@1%FPR as in App. D.2. For word deletion and synonym substitution, we segment the English texts with a regular expression and the Chinese texts with Jieba. Deletion removes exactly half of the words, chosen at random, and substitution replaces half of the words with a random synonym from WordNet (English) or the Open Multilingual WordNet (Chinese); if fewer words are replaceable, we replace all of them. For back-translation and paraphrasing, we use DEEPSEEK-V4 FLASH with temperature 1, reasoning disabled, and the prompts of App. J.2.

We compute the perplexity of the replies with the model that generated them, on the reply alone. For diversity, we use the first 10 prompts of each language and generate 100 replies per prompt, and compute the Self-BLEU scores as in App. D.2. We report the relative distance (in percent) to the unwatermarked model with the same model and language. We use 5 new watermark keys, different from all keys used in our harness, and report for each metric the worst result over the 5 keys.

Lastly, for RotationFirst, we rebuild the Lexical group map for each tokenizer and extend it to Chinese tokens with the Open Multilingual WordNet.

## E OUR AUTORESEARCH HARNESS

In this section, we detail our watermarking library (App. E.1) and motivate the design of the research environment (App. E.2).

## E.1 OUR WATERMARKING LIBRARY

Library design We design our watermarking library around four primary components: the context seeding, the watermark score, the logits transformation, and the detector, following the formalization described in Sec. 3.1 and the work of Gloaguen et al. (2026). At generation time, the watermarking scheme is defined as a vLLM logit processor. It takes as input the list of tokens already generated and the next-token logits (i.e., the unnormalized next-token probability distribution) and returns the watermarked next-token logits. It is stateful per prompt but stateless across prompts (i.e., each prompt is treated independently). For detection, it takes as input a list of tokens and returns a p-value. This simple design, essentially a class with two methods, ensures that the proposed schemes can be easily evaluated and, following Gloaguen et al. (2026), is expressive enough to represent most prior watermarking schemes. In fact, alongside the library boilerplate, we provide the research agents with implementations of the following watermarks: AAR (Aaronson, 2023), SynthID (Dathathri et al., 2024), DiPMark (Wu et al., 2024), TextSeal (Sander et al., 2026), MCMark (Chen et al., 2025), SWEET (Lee et al., 2024), KGW (Kirchenbauer et al., 2023), unigram (Zhao et al., 2024b), and the schemes from Gloaguen et al. (2026).

Throughput We compare our vLLM library with   
the reference implementations of AAR, SynthID, and   
TextSeal on NVIDIA RTX PRO 6000 Blackwell   
GPUs. For each scheme, we generate 128 tokens   
for each of 100 ELI5 prompts with LLAMA-3.2-1B-  
INSTRUCT, temperature 0.7, and top-p 1. Our runs   
and the SynthID reference use top-k 50; the AAR   
and TextSeal references expose no top-k option and   
sample from the full vocabulary. We exclude model   
loading and compute throughput as total generated to  
kens divided by total generation time. Our unbatched   
generation is 1.56–1.60 faster than the references; batching eight prompts increases this to 4.12– generation is 1.56–1.60 × faster than the references; 7.27 . 7.27×.

Table 4: End-to-end generation throughput (tokens/s). Reference implementations use Transformers; our library uses vLLM with one or up to eight prompts per request.
<table><tr><td>Scheme</td><td>Reference</td><td>Ours (1)</td><td>Ours (8)</td></tr><tr><td>SynthID</td><td>152.3</td><td>243.5</td><td>628.2</td></tr><tr><td>AAR</td><td>194.5</td><td>303.9</td><td>1,413.3</td></tr><tr><td>TextSeal</td><td>186.9</td><td>298.6</td><td>1,338.7</td></tr></table>

## E.2 RESEARCH ENVIRONMENT

Here, we motivate the two-phase submission budget of Sec. 3.4 with two runs from Table 5. We find that, when several submissions are available, agents mostly spend them on variants of a single scheme, whereas when a single submission is left, they explore distinct ideas locally before committing to one.

GPT-6 ASTRA (low) In this run, the first agent was given all 11 submissions at once. It spent 10 of them in about four hours on FreshRace and 8 of its variants (FreshRace-lexical was submitted twice). Then, with a single submission left, the 8 subsequent agents implemented and evaluated locally 15 new schemes before submitting RenewalRace. Moreover, when we retrospectively submit the 15 unsubmitted schemes to our private grader, all of them pass our verification, i.e., they are not merely failed attempts.

OPUS 5 This run uses the two-phase budget. In the first phase, the agent spent its 4 submissions on a single scheme, TallyMark. After its first submission failed our soundness given a key test, it changed the context seeding (also rejected), then reverted this change and multiplied the p-values by 4, and finally resubmitted the same code with an updated description. In the second phase, where each agent has a single submission, all 12 submissions are distinct schemes, which significantly improve robustness.

## F STATISTICAL TESTS

In this section, we present the algorithmic description of the three statistical tests from Sec. 3.3. We prove in App. I.3 that each test rejects a valid scheme with probability at most δ. If a scheme is rejected, it means that with confidence 1 δ it violates the property. If it is not rejected, as with every statistical tests, it means we can’t conclude whether the scheme violates the property. In practice, however we find that the tests are powerful e.g., in App. B they rejected all bad schemes and even rejected some of the baselines.

## G HASH FUNCTION AND PRNG

In this part, we explain the hash functions used by the agents to turn the context into a seed (i.e., an integer), and detail the PRNG algorithm we use in our library.

Hash functions As stated in App. A.3, we assume that the hashing function is without collision, i.e., two contexts that are different should return a different hash. In particular, prior works used weaker

```latex
Algorithm 1 Testing distortion-freeness
Require: Watermark transform $f ,$ fixed contexts and next-token distributions $( \omega _ { < t } , p _ { t } ) _ { t = 1 } ^ { N } , N \ge 1$
keys per context $m \geq 2 ,$ head size $1 \leq k \leq | \Sigma |$ , significance level $\delta \in ( 0 , \dot { 1 } )$
1: $h \cup - \lfloor m / 2 \rfloor ; a \gets \mathrm { F } \mathrm { : }$ alse
2: for $t \doteq 1 , \ldots , N$ do
3: $v _ { 1 } , \ldots , v _ { k }$ the k most likely tokens under $p _ { t }$
4: $\begin{array} { r } { G _ { t } ( q ) \gets ( q ( v _ { 1 } ) , \dots , q ( v _ { k } ) , 1 - \sum _ { r = 1 } ^ { k } q ( v _ { r } ) ) } \end{array}$ ▷ Pool remaining tokens
5: Draw m uniform independent keys and derive $\zeta _ { t } ^ { ( 1 ) } , \dots , \zeta _ { t } ^ { ( m ) }$ at $\omega _ { < t }$
6: $p _ { t } ^ { ( j ) }  f ( \zeta _ { t } ^ { ( j ) } , p _ { t } ) ; d _ { j }  G _ { t } ( p _ { t } ^ { ( j ) } ) - G _ { t } ( p _ { t } )$ for $j = 1 , \ldots , m$
7: $\underline { { a } }  a \lor ( \exists j : p _ { t } ^ { ( j ) } \neq p _ { t } )$
8: $\bar { d }  m ^ { - 1 } { \dot { \sum _ { j = 1 } ^ { m ^ { * } } } } d _ { j }$
9: p<sub>coord</sub> min $\{ 1 , 2 ( k + 1 )$ exp( 2m <sup>¯</sup>d <sup>2</sup> )
10: $\begin{array} { r } { u  \mathrm { s i g n } ( h ^ { - 1 } \sum _ { j = 1 } ^ { h } d _ { j } ) } \end{array}$ ▷ Learn a bias direction
11: $\begin{array} { r } { b \gets \operatorname* { m a x } \{ 0 , ( m \stackrel { - } { - } h ) ^ { - 1 } \sum _ { j = h + 1 } ^ { m } { u ^ { \top } d _ { j } } \} } \end{array}$ ▷ Test on held-out keys
12: p<sub>split</sub> $ \exp ( - ( m - h ) b ^ { 2 } / 2 )$
13: $q _ { t } \gets$ min $\{ 1 , 2$ min(p<sub>coord</sub>, p<sub>split</sub>)
14: end for
15: $Q $ min $\{ 1 , N$ min<sub>t</sub> $q _ { t } \}$ ▷ Bonferroni across contexts
16: if $Q \leq \delta$ then
17: return Reject
18: end if
19: return No distortion detected if $^ { a , }$ otherwise inconclusive
```

Algorithm 2 Testing soundness over the key   
Require: Detector $P _ { \xi }$ , nonempty corpus $C = ( \omega _ { 1 } , \ldots , \omega _ { n } )$ , keys per text m $\geq 1$ , finite nonempty   
thresholds $A \subseteq [ 0 , 1 ]$ , significance level $\delta \in ( 0 , 1 )$   
1: $\eta  \big ( 1 - ( 1 - \bar { \delta } ) ^ { 1 / \bar { n } } \big ) / | A |$ ▷ Corrected cutoff   
2: $\mathscr { R } \gets \dot { \emptyset }$   
3: for $i = 1 , \ldots , n$ do   
4: Draw $\xi _ { 1 } ^ { i } , \dots , \xi _ { m } ^ { i }$ uniformly and independently from the key space   
5: $P _ { i , j }  P _ { \xi _ { i } ^ { i } } ( \omega _ { i } )$ for $j = 1 , \ldots , m$   
6: for $\alpha \in { \mathcal { A } }$ do   
7: $\begin{array} { r } { X _ { i } ( \alpha ) \gets \sum _ { j = 1 } ^ { m } \mathbf { 1 } \{ P _ { i , j } \leq \alpha \} } \end{array}$   
8: $q _ { i } ( \alpha ) \gets \mathrm { P r } _ { Z } ,$ Binomial(m,α) $[ Z \geq X _ { i } ( \alpha ) ]$   
9: if $q _ { i } ( \alpha ) \leq \eta$ then   
10: ${ \mathcal { R } } \gets { \mathcal { R } } \cup \{ ( i , \alpha ) \}$   
11: end if   
12: end for   
13: end for   
14: return ▷ Reject the scheme if $\mathcal { R } \neq \emptyset$

hash functions that had a high probability of collisions. We assume we have a tuple $( x _ { 1 } , \ldots , x _ { k } ) \in \mathbb { N } ^ { k }$ of integers (token IDs or lexical classes $\mathcal { G } ( \omega _ { t } ) )$ . All arithmetic is on unsigned 64-bit integers (i.e., modulo $2 ^ { 6 4 } )$ ) unless stated otherwise, is the bitwise XOR, and $\gg$ the right shift.

SplitMix64 hash: We chain the SplitMix64 finalizer $m ( x ) = y _ { 2 } \oplus ( y _ { 2 } \gg 3 1 )$ , with $y _ { 1 } = ( x \oplus ( x \gg$ 30)) 0xbf58476d1ce4e5b9 and $y _ { 2 } = ( y _ { 1 } \oplus ( y _ { 1 } \gg 2 7 ) )$ 0x94d049bb133111eb. The context hash is

$$
h _ { 0 } = { \Theta } { \times } 2 { 4 3 } { \sf f e a } { 8 } { 8 } { 8 } 5 { \sf s a } { 3 } { \Theta } { 8 } { \mathrm { d } } 3 , \qquad h _ { j } = { m } \big ( h _ { j - 1 } { \oplus } { m } ( x _ { j } + j ) \big ) , \qquad a _ { t } = h _ { k } ,
$$

where the offset $+ j$ makes the hash order-sensitive. The hash is not injective, but with 64-bit outputs a collision among n contexts has probability at most $n ^ { 2 } / 2 ^ { 6 5 }$

Rotation first hash: packs the context injectively, so the no-collision assumption holds exactly. However, it is only valid for context sizes up to two. A context of length one is encoded as

Algorithm 3 Testing soundness given a key   
Require: Detector $P _ { \xi } ,$ human text distribution ${ \mathcal { H } } _ { 0 } ,$ number of keys $d \geq 1 ,$ , texts per key $n \geq 1$   
allowed bad-key fraction $\beta \in ( 0 , 1 )$ , predeclared finite nonempty thresholds ${ \bar { \mathcal { A } } } \subseteq [ 0 , 1 ]$ and   
screening levels $s \subset ( 0 , 1 )$ , allocated significance level $\delta \in ( 0 , \dot { 1 } )$   
1: $T ( s , p , c ) \gets \mathrm { P r } _ { Z \sim \mathrm { B i n o m i a l } ( s , p ) } [ Z \geq c ]$ ▷ Binomial upper tail   
2: $\eta  \delta / ( | A | | S | ) ; \mathcal { R }  \emptyset$   
3: for $j = 1 , \ldots , d$ do   
4: Draw a uniform key $\xi _ { j }$ and $\omega _ { 1 } ^ { j } , \ldots , \omega _ { n } ^ { j } \sim \mathcal { H } _ { 0 }$ ▷ All draws independent   
5: $P _ { j , t }  P _ { \xi _ { j } } ( \omega _ { t } ^ { j } )$ for $t = 1 , \ldots , n$   
6: $\begin{array} { r } { X _ { j } ( \alpha )  \sum _ { t = 1 } ^ { n } \mathbf { 1 } \{ P _ { j , t } \leq \alpha \} } \end{array}$ for $\alpha \in { \mathcal { A } }$   
7: end for   
8: for $( \alpha , \gamma ) \in \mathcal { A } \times \mathcal { S }$ do   
9: c min $\left\{ k \in \left\{ 1 , \ldots , n + 1 \right\} : T ( n , \alpha , k ) \leq \gamma \right\}$   
10: $r  T ( n , \alpha , c ) ; b  \beta + ( 1 - \beta ) r$ ▷ Null screening bound   
11: $\begin{array} { r } { B  \sum _ { j = 1 } ^ { d } \mathbf { 1 } \{ X _ { j } ( \alpha ) \geq c \} } \end{array}$ ▷ Keys screening positive   
12: $q  T ( \bar { d } , b , B )$   
13: if $q \leq \eta$ then   
14: ${ \mathcal { R } }  { \mathcal { R } } \cup \{ ( \alpha , \gamma ) \}$   
15: end if   
16: end for   
17: return ▷ Reject the scheme if $\mathcal { R } \neq \emptyset$

$a _ { t } = x _ { 1 } + 1$ , and one of length two as $a _ { t } = 2 ^ { 4 1 } + 2 ^ { 2 0 } x _ { 1 } + x _ { 2 }$ , which is injective for vocabularies below $2 ^ { 2 0 }$

Rolling polynomial hash: a rolling polynomial hash with a counter. This is the hash used with Ladder context seeding $( \mathrm { A p p . } \mathrm { A } . 3 )$ . Given the lexical classes $g _ { t } = \mathcal { G } ( \omega _ { t } )$ , a context of width $w \leq 4$ ending at t is hashed as

$$
K _ { w } = \Big ( \sum _ { i = 0 } ^ { w - 1 } g _ { t - i } \varphi ^ { w - 1 - i } \Big ) \oplus \big ( w \cdot \mathsf { \emptyset } \mathbf { x } 2 5 4 5 \mathsf { F } 4 9 1 4 \mathsf { F } 6 \mathsf { C } \mathsf { D } \mathsf { D } \mathsf { \mathsf { 1 } } \mathsf { D } \big ) , \qquad \varphi = \mathsf { \emptyset } \mathbf { x } 9 5 3 7 7 9 \mathbf { B } 9 7 \mathsf { F } 4 \mathsf { A } 7 \mathsf { C } \mathsf { 1 } 5 ,
$$

where the XOR with a width-dependent constant separates contexts of different lengths. Let n be the number of earlier occurrences of $K _ { w }$ in the text, and $w ^ { \star }$ the width selected by the ladder. The seed is then

$$
a _ { t } = \bigl \lfloor \mathrm { f m i x } _ { 6 4 } \bigl ( 8 1 9 2 K _ { w ^ { \star } } + n \bigr ) / 2 ^ { 3 2 } \bigr \rfloor ,
$$

where $\mathrm { f m i x _ { 6 4 } }$ is the MurmurHash3 finalizer and 8192 is the maximum tally, so that repeated contexts with different counts get different seeds. Unlike the two previous hashes, the seed has only 32 bits and collides after roughly $2 ^ { 1 6 }$ distinct contexts.

PRNG Given a seed ${ \boldsymbol { a } } _ { t } ,$ a candidate token v, and the 64-bit private key $\xi ,$ the PRNG returns the score $U _ { a _ { t } , v } \in [ 0 , 1 )$ . We require it to be stateless, i.e., $U _ { a _ { t } , v }$ depends only on $( \xi , a _ { t } , v )$ , so that the detector recovers exactly the scores used at generation.

Philox: this is the PRNG we implemented as part of our library and used for all baselines. It uses the counter-based Philox4x32-10 block cipher (Salmon et al., 2011), which maps a 128-bit counter and a 64-bit key to 4 pseudorandom 32-bit words $\left( y _ { 0 } , y _ { 1 } , y _ { 2 } , y _ { 3 } \right)$ . With A a bijective 32-bit avalanche $( x \mapsto x \oplus ( x \gg 1 6 )$ , multiply by 0x7FEB352D, $x \oplus ( x \gg 1 5 )$ , multiply by 0x846CA68B, $x \oplus ( x \gg 1 6 ) )$ , the score is

$$
\left( y _ { 0 } , \ldots , y _ { 3 } \right) = \mathrm { P h i l o x } _ { \xi } \left( A ( v ) , A ( a _ { t } ) , 0 , 0 \right) , \qquad U _ { a _ { t } , v } = \left\lfloor y _ { 0 } / 2 ^ { 9 } \right\rfloor 2 ^ { - 2 3 } ,
$$

where the avalanche disperses adjacent token IDs before Philox, and 23 bits fill the float32 mantissa.   
Independent Channels c use the candidate $v + 2 ^ { 1 8 } c$ , which never collides with a real token ID.

High-precision Philox: similar to Philox, but the 64-bit seed and the token are packed injectively into the counter:

$$
\begin{array} { r l r } & { } & { ( y _ { 0 } , \dotsc , y _ { 3 } ) = \mathrm { P h i l o x } _ { \xi } \big ( v , a _ { t } \mathrm { m o d } 2 ^ { 3 2 } , \lfloor a _ { t } / 2 ^ { 3 2 } \rfloor , \Theta \mathrm { x C } \Theta \mathrm { C } \Theta \mathrm { A } 1 2 3 \big ) , } \\ & { } & { U _ { a _ { t } , v } = \big ( 2 ^ { 2 6 } \lfloor y _ { 0 } / 2 ^ { 6 } \rfloor + \lfloor y _ { 1 } / 2 ^ { 6 } \rfloor \big ) 2 ^ { - 5 2 } + 2 ^ { - 5 3 } . \qquad } \end{array}
$$

Since the counter is injective in $( a _ { t } , v )$ , distinct (seed, token) pairs always map to distinct Philox counters, hence to pseudorandomly independent scores. When several scores are needed per (seed, token) pair (e.g., the three coordinates for RotationFirst), the seed is replaced by $2 ^ { 6 } a _ { t } + d ,$ reserving the low 6 bits for the index d.

SplitMix64 PRF: reuses the SplitMix64 finalizer m from the SplitMix64 hash, with the domainseparation constants $c _ { 1 } = \Theta$ xa4093822299f31d0 and c = 0x082efa98ec4e6c89:

$$
\begin{array} { r } { s = m \big ( a _ { t } \oplus m ( \xi \oplus c _ { 1 } ) \big ) , \qquad s ^ { \prime } = m \big ( s \oplus m ( v \oplus c _ { 2 } ) \big ) , \qquad U _ { a _ { t } , v } = \big ( \lfloor s ^ { \prime } / 2 ^ { 1 2 } \rfloor + \frac { 1 } { 2 } \big ) 2 ^ { - 5 2 } , } \end{array}
$$

where 52 bits fill the float64 mantissa, and the midpoint $+ \frac 1 2$ keeps $U _ { a _ { t } , v } \in ( 0 , 1 )$ so that $\Phi ^ { - 1 } ( U _ { a _ { t } , v } )$ is always finite. Independent Channels c use the key $\xi \oplus ( \overline { { { c } } } \cdot \Theta \times 9 \mathrm { e 3 } 7 7 9 \mathrm { b } 9 7 \mathrm { \# } \mathrm { a } 7 \mathrm { c } 1 5 )$

## H CONCORDMARK AND ROTATIONFIRST

We describe the schemes used in Sec. 4.3. For low and high variants, the generation is shared and only the detector differs.

## H.1 CONCORDMARK

Key components We describe precisely the ConcordMark generation algorithm in Algorithm 4, and both detector variants in Algorithms 5 and 6. ConcordMark uses Lexical group only for context seeding. Let  : Σ  N be the corresponding lexical map. Given a word, its spelling is first normalized and then gets attributed a synonym class of at most eight members (using WordNet). It then uses Ladder context seeding starting at width 1, or width 2 when the last mapped token ID is below 200. For the hash function, it uses the rolling polynomial hash (App. G).

For the watermark score, it uses both Prelude with a cap of 14 nats of entropy or 24 tokens and 6 independent Channels. It uses the Gumbel race as a logits transformation. For detection, the low variant uses the Excess Gamma detection and the high variant the weighted Position evidence.

Algorithm 4 ConcordMark generation   
Require: $\mathrm { K e y s } \xi _ { 0 } , \ldots , \xi _ { 5 } ,$ model distributions $p _ { t } ,$ lexical map ${ \mathcal { G } } ,$ completion length T.   
1: Sample J 0, . . . , 5 ▷ Use channel J, keyed by $\xi _ { J } ,$ , throughout   
2: $\omega  ( ) ; B  \hat { 0 } ; \mathcal { U }  \varnothing$   
3: for $t = 1 , \dots , T$ do   
4: if $\dot { \cdot } t \leq 2 4$ and B < 14 then   
5: Sample $\omega _ { t } \sim p _ { t }$ using private randomness   
6: $\begin{array} { r } { B \dot {  } B - \log \sum _ { v } p _ { t } \dot { ( v ) } ^ { 2 } } \end{array}$ ▷ Prelude collision entropy   
7: else   
8: $z _ { 1 : t - 1 } \gets ( \mathcal { G } ( \omega _ { 1 } ) , \ldots , \mathcal { G } ( \omega _ { t - 1 } ) ) ; m \gets \operatorname* { m i n } ( 4 , t - 1 )$   
9: $\ell \gets 1 ; \mathrm { i f } z _ { t - 1 } < 2 0 0 ,$ set ℓ  min(2, m)   
10: for $w = \ell , \ldots , m$ do   
11: W  z<sub>t−w:t−1</sub> ▷ Suffix of width w   
12: $n _ { t }  \vert \{ i : w \leq i < t - 1 , z _ { i - w + 1 : i } = W \} \vert$ ▷ Count earlier occurrences   
13: If $n _ { t } < 4$ or w = m, set $\mathfrak { z } _ { t }  ( w , W , n _ { t } ) , \dot { a } _ { t }  h ( c _ { t } )$ and break ▷ Context and seed   
14: end for   
15: if n 8192 or $a _ { t } \in \mathcal { U }$ then   
16: Sample $\omega _ { t } \sim p _ { t }$ using private randomness   
17: else   
18: $\mathcal { U }  \mathcal { U } \cup \{ a _ { t } \}$   
19: $\tilde { U } _ { t , v } \gets U _ { J , a _ { t } , v }$ for each v with $p _ { t } ( v ) > 0$   
20: $\begin{array} { r } { \omega _ { t }  \arg \operatorname* { m i n } _ { v : p _ { t } ( v ) > 0 } \{ - \log \tilde { U } _ { t , v } / p _ { t } ( v ) \} } \end{array}$ ▷ Token-level exponential race   
21: end if   
22: end if   
23: Append ω<sub>t</sub> to ω   
24: end for   
25: return ω

Algorithm 5 ConcordMark (low) detection   
Require: Keys $\overline { { \xi _ { 0 } , \ldots , \xi _ { 5 } } } ,$ completion $\omega ,$ lexical map ${ \overline { { \mathcal { G } } } } .$   
1: $D \gets \emptyset ; \mathcal { U } \gets \emptyset$   
2: for $t = 2 , \ldots , | \omega |$ do   
3: Reconstruct the ladder seed $a _ { t }$ and count n from $\mathcal { G } ( \omega _ { < t } )$ as in Algorithm 4   
4: If $\dot { n } _ { t } < 8 1 9 2$ and $( a _ { t } , \omega _ { t } ) \notin \mathcal { U } ,$ , set $D \gets D \cup \{ t \}$ and $\mathcal { U }  \mathcal { U } \cup \bar { \{ ( a _ { t } , \omega _ { t } ) \} }$   
5: end for   
6: If $D = \varnothing ,$ return 1   
7: $\mathcal { H } ( \omega ) \gets \{ a _ { t } : t \in D \} ; H \gets | \mathcal { H } ( \omega ) |$   
8: $\begin{array} { r } { D _ { a } ( \omega ) \xleftarrow { } \{ \omega _ { t } : t \in \mathring { D } , \ a _ { t } = \dot { a } \} ; \stackrel { \cdot \cdot } { m _ { a } } \gets | D _ { a } ( \omega ) | } \end{array}$ for each $a \in \mathcal { H } ( \omega )$   
9: for $j = 0 , \ldots , 5$ do   
10: for $a \in \mathcal { H } ( \omega )$ do   
11: $\begin{array} { r } { S _ { j , a } ( \omega )  \sum _ { v \in D _ { a } ( \omega ) } - \log ( 1 - U _ { j , a , v } ) } \end{array}$   
12: $Z _ { j , a } \gets - \log Q ( m _ { a } , S _ { j , a } ( \omega ) )$ ▷ Calibrate each context with its Gamma tail   
13: end for   
14: $\begin{array} { r } { S _ { j } ( \omega )  \sum _ { a \in \mathcal { H } ( \omega ) } \operatorname* { m a x } \{ Z _ { j , a } - \log 2 , 0 \} } \end{array}$ ▷ Keep evidence above the median   
15: $P _ { j }  1$   
16: $\mathbf { i f } ^ { \prime } S _ { j } ( \omega ) > H / 2$ then   
17: $\theta  ( 3 - \sqrt { 1 + 4 H / S _ { j } ( \omega ) } ) / 2$   
18: $P _ { j }  \supset \big \lbrack { H \log [ ( 1 - \overbrace { \theta / 2 } ) ^ { ^ { \prime } } ( 1 - \theta ) ] - \theta S _ { j } ( \omega ) } \big \rbrack$ ▷ Chernoff bound   
19: If $P _ { j } < 0 . 1$ , set $\begin{array} { r } { P _ { j }  \sum _ { b = 1 } ^ { H } \binom { H } { b } 2 ^ { - H } Q ( b , S _ { j } ( \omega ) ) } \end{array}$ ▷ Exact excess tail $R _ { H }$   
20: end if   
21: end for   
22: return $\begin{array} { r } { P ( \omega ) = 1 - ( 1 - \operatorname* { m i n } _ { j } P _ { j } ) ^ { 6 } } \end{array}$ ▷ Šidák correction over channels

Algorithm 6 ConcordMark (high) detection   
Require: Keys $\xi _ { 0 } , \ldots , \xi _ { 5 } ,$ completion ω, lexical map ${ \overline { { \mathcal { G } } } } ,$ auxiliary model distributions $\hat { p } _ { t } .$   
1: Reconstruct $a _ { t }$ and the retained positions D as in Algorithm 5   
2: If $D = \varnothing ,$ return 1   
3: for $t \in D$ do   
4: $\begin{array} { r } { \eta _ { t }  1 + \frac { 1 } { 4 } } \end{array}$ round(4 min max log p ${ \bf \dot { \rho } } _ { t } ( \omega _ { t } ) , 0 \} , 2 . 5 \} )$   
5: $\dot { C } _ { t } \gets$ eight most probable tokens under $\hat { p } _ { t } \} \backslash \{ \omega _ { t } \}$   
6: $\alpha _ { t , v }  \sin \{ 1 0 ^ { 4 } , \stackrel { \cdot } { \hat { p } } _ { t } ( v ) / \hat { p } _ { t } ( \omega _ { t } ) \}$ for each $v \in C _ { t }$   
7: $\begin{array} { r } { C _ { t } \gets \{ v \in \dot { C } _ { t } : \alpha _ { t , v } \geq 1 0 ^ { - 3 } \} ; \lambda _ { t } \gets \sum _ { v \in C _ { t } } \alpha _ { t , v } } \end{array}$   
8: I $\mathrm { \Delta } [ \lambda _ { t } > 8 ,$ set $C _ { t } \gets \emptyset$ and $\lambda _ { t } \gets 0$ ▷ Discard an unreliable reconstructed race   
9: end for   
10: for $j = 0 , \ldots , 5$ do   
11: for $t \in D$ do   
12: $\smash { W _ { j , t } \gets \mathbf { 1 } \{ U _ { j , a _ { t } , v } < U _ { j , a _ { t } , \omega _ { t } } ^ { \alpha _ { t , v } } }$ for all $v \in C _ { t } \}$   
13: if $\dot { W } _ { j , t } = \dot { 1 }$ then ▷ The observed token wins the reconstructed race   
14: $R _ { j , t } \gets ( 1 - U _ { j , a _ { t } , \omega _ { t } } ^ { 1 + \lambda _ { t } } ) / ( 1 + \lambda _ { t } )$   
15: else   
16: $R _ { j , t }  1 - U _ { j , a _ { t } , \omega _ { t } } + U _ { j , a _ { t } , \omega _ { t } } ^ { 1 + \lambda _ { t } } / ( 1 + \lambda _ { t } )$   
17: end if   
18: end for   
19: $\begin{array} { r } { S _ { j } ( \omega )  \sum _ { t \in D } - \eta _ { t } } \end{array}$ log $R _ { j , t }$ ▷ Weighted position evidence   
20: end for   
21: s  max<sub>j</sub> $S _ { j } ( \omega )$   
22: if all weights equal some η then   
23: $q  Q ( | D | , s / \eta )$ ▷ Exact Gamma upper tail   
24: else   
25: q Lugannani–Rice approximation to $\begin{array} { r } { \operatorname* { P r } [ \sum _ { t \in D } \eta _ { t } E _ { t } \geq s ] , } \end{array}$ , for independent $E _ { t } \sim \mathrm { E x p } ( 1 )$   
26: end if   
27: return $P ( \omega ) = \operatorname* { m i n } \{ 1 , 1 . 0 2 [ 1 - ( 1 - q ) ^ { 6 } ] \}$ ▷ Channel correction and tail safety factor

## H.2 ROTATIONFIRST

Key components We describe precisely the RotationFirst generation algorithm and its detector in Algorithms 7 and 8. RotationFirst uses Lexical group throughout the scheme. Let $\mathcal { G } : \Sigma  \mathbb { N }$ be the corresponding lexical map. Each token is decoded, stripped of surrounding whitespace, and lowercased. If the resulting word appears in WordNet, it is assigned to the synonym class given by its first WordNet synset. Let  be the set of the resulting synonym classes. Tokens sharing this synset are grouped together, whereas all other tokens remain in singleton classes. It then uses Global first use context seeding with the rotation first hash (App. G). For a class c, it uses address $( 0 , c )$ if it has not been used before and $c \in { \mathcal { E } }$ , and otherwise uses $( \mathcal { G } ( \omega _ { t - 1 } ) + 1 , c )$

For the watermark score, it uses a private geometry approach, namely the Rotation spherical geometry (Spherical). For each request, sample randomly a uniform quaternion A. Then, at each step draw a uniform keyed $Z _ { a _ { t } , u } \in \dot { S } ^ { 3 }$ and derive the score:

$$
\tilde { U } _ { a _ { t } , u } = F ( | A \cdot Z _ { a _ { t } , u } | ) ,
$$

where $\begin{array} { r } { F ( r ) = \frac { 2 } { \pi } \left( \arcsin r + r \sqrt { 1 - r ^ { 2 } } \right) } \end{array}$ . Then for detection, it uses the corresponding Predictive Gamma detector.

```latex
Algorithm 7 RotationFirst generation
Require: Key ξ, model distributions $p _ { t } .$ , grouping , eligible classes $\varepsilon ,$ completion length $T .$
1: Sample $\dot { A } \sim \mathcal { U } ( \mathbb { S } ^ { 3 } )$ ▷ Private rotation shared throughout the completion
2: $\begin{array} { r } { F ( r ) \gets \frac { 2 } { \pi } ( \arcsin r + r \sqrt { 1 - r ^ { 2 } } ) } \end{array}$
3: $\omega  ( ) ; \hat { \mathcal { A } }  \varnothing ; \mathcal { U }  \varnothing$ ▷ Emitted classes and initialized clocks
4: for $t \stackrel { \cdot \cdot } { = } 1 , \ldots , T$ do
5: $\begin{array} { r } { M _ { t } ( u ) \gets \sum _ { v : \mathcal { G } ( v ) = u } p _ { t } ( v ) } \end{array}$ for each class u
6: $b _ { t }  \mathcal { G } ( \omega _ { t - 1 } ) + 1 \mathrm { i f } t > 1$ , otherwise $b _ { t } \gets 0$
7: for each class u with $M _ { t } ( u ) > 0$ do
8: $c _ { t } ( u ) \gets 0 \mathrm { i f } u \in \mathcal { E } \setminus \dot { \mathcal { A } } ,$ otherwise $c _ { t } ( u ) \gets b _ { t } ; a _ { t } ( u ) \gets h ( c _ { t } ( u ) )$
9: $\mathbf { i f } \ ( a _ { t } ( u ) , u ) \not \in \mathcal { U }$ then
10: $\mathbf { i f } \ b _ { t } \neq 0 \ \mathrm { o r } \ u \in \mathcal { E }$ then
11: Recover $Z _ { a _ { t } ( u ) , u } \in \mathbb S ^ { 3 }$ using key ξ
12: $\tilde { U } _ { a _ { t } ( u ) , u } \gets F ( | A \cdot Z _ { a _ { t } ( u ) , u } | )$
13: $R _ { a _ { t } ( u ) , u } \gets - \log \tilde { U } _ { a _ { t } ( u ) , u }$
14: else
15: Sample $R _ { a _ { t } ( u ) , u } \sim \mathrm { E x p } ( 1 )$ privately
16: end if
17: $\mathcal { U }  \mathcal { U } \cup \{ ( a _ { t } ( u ) , u ) \}$
18: end if
19: end for
20: $\begin{array} { r } { u _ { * } \gets \mathrm { a r g } \operatorname* { m i n } _ { u : M _ { t } ( u ) > 0 } R _ { a _ { t } ( u ) , u } / M _ { t } ( u ) ; \tau _ { t } \gets R _ { a _ { t } ( u _ { * } ) , u _ { * } } / M _ { t } ( u _ { * } ) } \end{array}$
21: $R _ { a _ { t } ( u ) , u } \gets \operatorname* { m a x } \{ 0 , R _ { a _ { t } ( u ) , u } - \tau _ { t } M _ { t } ( u ) \}$ for each u with $\dot { M } _ { t } ( u ) > 0$
22: Sample $R _ { a _ { t } ( u _ { * } ) , u _ { * } } \sim$ Exp(1) privately ▷ Refresh the winning clock
23: Sample ω<sub>t</sub> from $\{ v : \mathcal { G } ( v ) = u _ { * } \}$ with probabilities $p _ { t } ( v ) / M _ { t } ( u _ { * } )$
24: $\mathcal { A } \doteq \mathcal { A } \cup \{ u _ { * } \}$ ; append ω<sub>t</sub> to ω
25: end for
26: return ω
```

Algorithm 8 RotationFirst detection   
Require: Key ξ, completion ω, grouping , eligible classes ${ \overline { { \varepsilon , } } }$ public unit quaternions $A _ { 0 } , \dotsc , A _ { 5 1 1 }$   
1: $\overset { \cdot } { D }  ( ) ; \overset { \cdot } { \mathcal { A } }  \varnothing ; \overset { \cdot } { \mathcal { U } }  \varnothing$   
2: for $t = 1 , \ldots , | \omega |$ do   
3: $u _ { t } \gets \mathcal G ( \omega _ { t } )$   
4: $b _ { t }  \mathcal { G } ( \omega _ { t - 1 } ) + 1 \mathrm { i f } t > 1 ,$ , otherwise $b _ { t } \gets 0$   
5: $c _ { t }  0 \mathrm { i f } u _ { t } \in \mathcal { E } \setminus \mathcal { A } ,$ otherwise $c _ { t } \gets b _ { t } ; a _ { t } \gets h ( c _ { t } )$   
6: If (t > 1 or $u _ { t } \in \dot { \mathcal { E } } )$ and $( a _ { t } , u _ { t } ) \notin \mathcal { U } ,$ append $( a _ { t } , u _ { t } )$ to D and set $\mathcal { U }  \mathcal { U } \cup \{ ( a _ { t } , u _ { t } ) \}$   
7: $\mathcal { A }  \mathcal { A } \cup \{ u _ { t } \}$   
8: end for   
9: $N  | D | ; \mathrm { i f } \ N \leq 1 ,$ return 1   
10: Enumerate D as $( a _ { i } , u _ { i } ) _ { i = 1 } ^ { N }$ in text order   
11: $F ( r ) \gets \frac { 2 } { \pi } ( \arcsin r + r \sqrt { 1 - r ^ { 2 } } ) ; d ( B , Z ) \gets 1 - F ( | B \cdot Z | )$   
12: $B _ { 1 }  A _ { 0 } ^ { \prime } ; w _ { 1 , j }  1 / 5 1 2 \mathrm { f o r } j = 0 , \ldots , 5 1 1 ; S ( \omega )  0$   
13: for $i = 1 , \ldots , N$ do   
14: Recover $\dot { Z _ { i } } = Z _ { a _ { i } , u _ { i } } \in \mathbb S ^ { 3 }$ using key ξ   
15: $S ( \omega ) \gets S ( \omega ) - \log d ( B _ { i } , Z _ { i } )$ ▷ Score before updating the predicted rotation   
16: $w _ { i + 1 , j }  w _ { i , j } d ( A _ { j } , Z _ { i } ) ^ { - 1 / 2 } \mathrm { f o r } j = 0 , . . . , 5 1 1$   
17: Normalize $\textstyle w _ { i + 1 , j } \gets w _ { i + 1 , j } / \sum _ { k = 0 } ^ { 5 1 1 }$ w<sub>i+1,k</sub>   
18: $\begin{array} { r } { B _ { i + 1 }  \mathrm { e i g } _ { \mathrm { m a x } } ( \sum _ { j = 0 } ^ { 5 1 1 } w _ { i + 1 , j } A _ { j } A _ { j } ^ { \top } ) } \end{array}$ ▷ Unit eigenvector for the largest eigenvalue   
19: end for   
20: return $P ( \omega ) = Q ( N , S ( \omega ) )$ ▷ Predictive Gamma upper tail

## H.3 ADDITIONAL EVALUATION

Here, we complete the evaluation from Sec. 4.3 with additional plots (Figure 26). Namely, we plot the TPR as a function of the significance level $\alpha ,$ from $1 0 ^ { - 6 } \ \mathrm { { \bar { t } o } \ 1 0 ^ { - 1 } }$ , for clean detection and paraphrasing. At each $\alpha ,$ the TPR is the fraction of samples with p-value below α. As in the main evaluation, we report the minimum TPR across the five watermarking seeds.

![](images/e9b9b0d03e4fe7407138a5c2804842e1f666afad16cb825f87bd3166b3ad5b16.jpg)  
Figure 26: TPR as a function of the significance level<sup>1</sup> α, with 1k samples per model and language generated from WildChat prompts. Left is clean detection and right after paraphrasing.

Results Across models, languages, and significance levels, these curves confirm the conclusions of Sec. 4.3. ConcordMark (high) provides the strongest detection, and especially after paraphrasing. ConcordMark (low) also retains a strong paraphrasing robustness while being more efficient at detection, as it does not require a forward pass. RotationFirst is less robust than ConcordMark, as expected as it focuses mainly on quality, yet remains comparable to the baselines.

## I PROOFS

In this section, we prove the theoretical claims made throughout the paper. In App. I.1, we state our assumptions and two lemmas used in the rest of the section. In App. I.2, we prove that a detector that is sound over the key can be turned into a detector that is sound given a key (Sec. 3.2). In App. I.3, we prove that the three statistical tests of Sec. 3.3 are valid. In App. I.4, we prove that ResidualRace is distortion-free $( \mathbf { A p p . A . } 5 )$ . Lastly, in App. I.5, we derive the null distribution of the Predictive Gamma and Position evidence detectors (App. A.6). For all other mathematical results, we consider the proof sufficiently trivial and do not derive them here.

## I.1 PRELIMINARIES

Setup Let Ξ be the key space, and let ξ be drawn from $\Xi$ as in Equation (2). For every text $\omega \in \Sigma ^ { * }$ the detector returns a p-value $P _ { \xi } ( \omega ) \in [ 0 , 1 ]$ , and we assume that $( \xi , \omega ) \mapsto P _ { \xi } ( \omega )$ is measurable (which holds trivially for a finite key space). Human texts are written without knowledge of the private key: for any human text distribution ${ \mathcal { H } } _ { 0 } .$ , the text $\Omega \sim \mathcal { H } _ { 0 }$ is independent of ξ. As in Sec. 3.1, we model the PRNG as ideal: the seed $a _ { t }$ is a function of the context only, and, over a uniformly drawn key, the scores of distinct (seed, unit) pairs are independent, with $U _ { a , u } \sim \mathcal { U } ( [ 0 , 1 ] )$ (and $Z _ { a , u }$ distributed according to its geometry for private-geometry scores). We also assume that the hash function has no collision $( { \mathrm { A p p . ~ G } } )$ . Finally, we write $T ( s , \bar { p } , c ) : = \operatorname* { P r } _ { Z \sim \mathrm { B i n o m i a l } ( s , p ) } [ Z \geq c ]$ for the binomial upper tail, as in Algorithm 3.

Lemma 1. Let X be a real random variable and $\varphi : \mathbb { R } \to [ 0 , \infty )$ a continuous nonincreasing function such that $\operatorname* { P r } [ X \geq s ] \leq \varphi ( s )$ for all $s \in \mathbb R$ . Then min $\{ 1 , \varphi ( X ) \}$ is a valid p-value, i.e., Pr[min $\{ 1 , \varphi ( X ) \} \leq \bar { x } ] \leq x$ for all $x \in [ 0 , 1 ]$

Proof. The case $x \ = \ 1$ is trivial, and the case $x \ = \ 0$ follows from the case $x \ > \ 0$ , since $\operatorname* { P r } [ \operatorname* { m i n } \{ 1 , \varphi ( X ) \} ~ \leq ~ 0 ] ~ \leq ~ \operatorname* { P r } [ \operatorname* { m i n } \{ 1 , \varphi ( X ) \} ~ \leq ~ x ]$ for every $x \ > \ 0$ Let $x \in \mathsf { \Gamma } ( 0 , 1 )$ and $I : = \{ \bar { s } \in \mathbb { R } : \varphi ( s ) \ \leq \ x \}$ . If $I \ = \ \varnothing ,$ , the event is empty. Otherwise, I is an interval unbounded above, and it is bounded below, since otherwise $\bar { \operatorname* { P r } } [ X \geq s ] \leq x < 1$ for all $s ,$ which contradicts $\operatorname* { P r } [ X \geq s ] \to 1 { \mathrm { ~ a s ~ } } s \to - \infty$ . Let $s ^ { * } : = \operatorname { i n f } I$ . By continuity, $\varphi ( s ^ { * } ) \leq x .$ , and since $\{ \varphi ( X ) \leq x \} \subseteq \{ X \geq \dot { s } ^ { * } \}$ ,

$$
\operatorname* { P r } [ \operatorname* { m i n } \{ 1 , \varphi ( X ) \} \leq x ] \leq \operatorname* { P r } [ X \geq s ^ { * } ] \leq \varphi ( s ^ { * } ) \leq x .
$$

Lemma 2. Let X be a sum ofs independent Bernoulli random variables with parameters at most $p \in [ 0 , 1 ]$ . Then $\operatorname* { P r } [ X \geq c ] \leq T ( s , \bar { p , c } )$ for all $c \in \mathbb { N } ,$ , and $\operatorname* { P r } [ T ( s , p , X ) \leq \eta ] \stackrel { . } { \leq } \eta f o$ r all $\eta \in [ 0 , 1 ]$

Proof. Let $p _ { 1 } , . . . , p _ { s } \leq p$ be the Bernoulli parameters and $V _ { 1 } , \dots , V _ { s }$ i.i.d. uniform random variables on [0, 1]. Then X has the same distribution as $\begin{array} { r } { \sum _ { i } 1 \{ V _ { i } \leq p _ { i } \} \leq \sum _ { i } 1 \{ V _ { i } \leq p \} \sim } \end{array}$ Binomial(s, p), which gives the first claim. For the second claim, since $c \mapsto T ( s , p , c )$ is nonincreasing and ${ \cal T } ( s , p , s + 1 ) = 0$ , let $c ^ { * } : = \operatorname* { m i n } \{ c \in \{ 0 , \ldots , s + 1 \} : T ( s , p , c ) \leq \eta \}$ . Then $\{ T ( s , p , X ) \leq \bar { \eta } \} =$ $\{ X \ge c ^ { * } \}$ , and by the first claim, $\operatorname* { P r } [ X \geq c ^ { * } ] \leq T ( s , p , c ^ { * } ) \leq \eta .$ □

## I.2 SOUNDNESS GIVEN A KEY

We prove the claim of Sec. 3.2: multiplying by $1 / \beta$ the p-value of a detector that is sound over the key (Equation (2)) gives a detector that is sound given a key (Equation (3)).

Proposition 1. Let $\beta \in \mathsf { \Gamma } ( 0 , 1 ]$ and let $P _ { \xi }$ be a detector that satisfies Equation (2). Define the calibrated detector

$$
P _ { \xi } ^ { \beta } ( \omega ) : = \operatorname* { m i n } \left\{ 1 , \frac { P _ { \xi } ( \omega ) } { \beta } \right\} .\tag{14}
$$

Then $P _ { \xi } ^ { \beta }$ satisfies Equation (2), andfor every human text distribution $\mathcal { H } _ { \mathrm { 0 } }$ and every $\alpha \in [ 0 , 1 ]$

$$
\operatorname* { P r } _ { \boldsymbol { \xi } } \left[ \operatorname* { P r } _ { \Omega \sim \mathcal { H } _ { 0 } } [ P _ { \boldsymbol { \xi } } ^ { \beta } ( \Omega ) \leq \alpha ] > \alpha \right] \leq \beta ,
$$

i.e., $P _ { \xi } ^ { \beta }$ also satisfies Equation (3).

Proof. We first relate the rejection regions of both detectors. For $\alpha \in [ 0 , 1 )$ , since min $\{ 1 , x \} \leq \alpha < 1$ holds if and only if $x \leq \alpha ,$ we have for every key $\xi$ and text ω

$$
P _ { \xi } ^ { \beta } ( \omega ) \leq \alpha \iff P _ { \xi } ( \omega ) \leq \alpha \beta .\tag{15}
$$

For $\alpha = 1$ , both properties hold trivially, as all probabilities are bounded by 1. We therefore assume $\alpha \in [ 0 , 1 )$ in the rest of the proof.

Soundness over the key. Let $\omega \in \Sigma ^ { * }$ . By Equation (15) and Equation (2) applied at the threshold $\alpha \beta \in [ 0 , 1 ]$

$$
\operatorname* { P r } _ { \xi } [ P _ { \xi } ^ { \beta } ( \omega ) \leq \alpha ] = \operatorname* { P r } _ { \xi } [ P _ { \xi } ( \omega ) \leq \alpha \beta ] \leq \alpha \beta \leq \alpha .
$$

Soundness given a key. Let $\mathcal { H } _ { \mathrm { 0 } }$ be a human text distribution, and define for each key the false positive rate

$$
g ( \xi ) : = \operatorname* { P r } _ { \Omega \sim \mathcal { H } _ { 0 } } [ P _ { \xi } ^ { \beta } ( \Omega ) \leq \alpha ] = \operatorname* { P r } _ { \Omega \sim \mathcal { H } _ { 0 } } [ P _ { \xi } ( \Omega ) \leq \alpha \beta ] ,
$$

where the equality follows from Equation (15). Because Ω is independent of $\xi ,$ Fubini’s theorem allows us to swap the two expectations, and applying Equation (2) to each fixed text gives

$$
\begin{array} { r } { \mathbb { E } _ { \xi } [ g ( \xi ) ] = \operatorname* { P r } _ { \xi , \Omega } [ P _ { \xi } ( \Omega ) \leq \alpha \beta ] = \mathbb { E } _ { \Omega \sim \mathcal { H } _ { 0 } } \left[ \operatorname* { P r } _ { \xi } [ P _ { \xi } ( \Omega ) \leq \alpha \beta ] \right] \leq \alpha \beta . } \end{array}
$$

If $\alpha = 0$ , then $g \geq 0$ and $\mathbb { E } _ { \xi } [ g ( \xi ) ] \le 0$ , so $g ( \xi ) = 0$ almost surely and $\operatorname* { P r } _ { \xi } [ g ( \xi ) > 0 ] = 0 \le \beta$ Otherwise, $\alpha > 0$ and Markov’s inequality gives

$$
\operatorname* { P r } _ { \xi } [ g ( \xi ) > \alpha ] \leq \frac { \mathbb { E } _ { \xi } [ g ( \xi ) ] } { \alpha } \leq \frac { \alpha \beta } { \alpha } = \beta ,
$$

which concludes the proof.

## I.3 VALIDITY OF THE VERIFICATION TESTS

We prove that each of the three tests of Sec. 3.3 wrongly rejects a valid scheme with probability at most its significance level δ.

Proposition 2. Let $f$ be a distortion-free logits transformation (Equation (1)). For each context t, let $v _ { 1 } , \ldots , v _ { k }$ be the k most likely tokens under $p _ { t }$ and let

$$
G _ { t } ( q ) : = { \Big ( } q ( v _ { 1 } ) , \ldots , q ( v _ { k } ) , 1 - \sum _ { r = 1 } ^ { k } q ( v _ { r } ) { \Big ) }
$$

map a distribution $q \in \Delta ( \Sigma )$ to a probability vector over the head tokens and the pooled remaining tokens. For m independent keys, let $d _ { j } : = G _ { t } ( f ( \zeta _ { t } ^ { ( j ) } , p _ { t } ) ) - G _ { t } ( p _ { t } )$ , <sup>¯</sup>d their mean, $h : = \lfloor m / 2 \rfloor$ $\begin{array} { r } { u : = \mathrm { s i g n } ( h ^ { - 1 } \sum _ { j = 1 } ^ { h } d _ { j } ) } \end{array}$ and $\begin{array} { r } { b : = \operatorname* { m a x } \{ 0 , ( m - h ) ^ { - 1 } \sum _ { j = h + 1 } ^ { m } { u ^ { \top } d _ { j } } \} } \end{array}$ . Algorithm 1 combines the p-values

$$
\begin{array} { r } { p _ { \mathrm { c o o r d } } : = \operatorname* { m i n } \{ 1 , 2 ( k + 1 ) \exp ( - 2 m \| \bar { d } \| _ { \infty } ^ { 2 } ) \} \quad a n d \quad p _ { \mathrm { s p l i t } } : = \exp ( - ( m - h ) b ^ { 2 } / 2 ) } \end{array}
$$

into $q _ { t } : = \operatorname* { m i n } \{ 1 , 2 \operatorname* { m i n } ( p _ { \mathrm { c o o r d } } , p _ { \mathrm { s p l i t } } ) \}$ , and rejects $\begin{array} { r } { f i f Q : = \operatorname* { m i n } \{ 1 , N \operatorname* { m i n } _ { t } q _ { t } \} \leq \delta . } \end{array}$ . Then Algorithm 1 rejects f with probability at most δ.

Proof. Fix a context t. Because the m keys are drawn independently, the scores $\zeta _ { t } ^ { ( 1 ) } , \dots , \zeta _ { t } ^ { ( m ) }$ are $\mathrm { i . i . d . }$ ., and by Equation (1), the differences $d _ { 1 } , \ldots , d _ { m }$ are i.i.d. with $\mathbb { E } [ d _ { j } ] = G _ { t } ( \mathbb { E } [ f ( \zeta _ { t } ^ { ( j ) } , p _ { t } ) ] ) -$ $G _ { t } ( p _ { t } ) = \dot { 0 , } \dot { \mathrm { a s } } \dot { G } _ { t }$ is affine.

Coordinate test. Each coordinate of $G _ { t } ( q )$ lies in [0, 1] for every $q \in \Delta ( \Sigma )$ , so each coordinate of $d _ { j }$ lies in an interval of length 1. By Hoeffding’s inequality and a union bound over the $k + 1$ coordinates, $\operatorname* { P r } [ \| \bar { d } \| _ { \infty } \geq s ] \leq 2 ( k + 1 ) \exp ( - 2 m s ^ { 2 } )$ for all $s \geq 0$ . Hence, by Lemma 1 with $\varphi ( s ) = 2 ( k + 1 ) \exp ( - 2 m { \dot { s } } ^ { 2 } )$ , p<sub>coord</sub> is a valid p-value.

Split test. The direction u only depends on $d _ { 1 } , \ldots , d _ { h }$ , so conditional on $u ,$ the variables $Y _ { j } : =$ $u ^ { \top } d _ { j }$ for $j > h$ are i.i.d. with mean zero. Since $G _ { t } ( q )$ and $G _ { t } ( \boldsymbol { p } _ { t } )$ are probability vectors and $u \in \overline { { [ - 1 , 1 ] ^ { k + 1 } } } , u ^ { \top } G _ { t } ( q ) \in [ - 1 , 1 ]$ and hence $Y _ { j }$ lies in an interval of length 2. By Hoeffding’s inequality, $\operatorname* { P r } [ b \geq s \mid u ] \leq \exp ( - ( m - h ) s ^ { 2 } / 2 )$ for all $s > 0 ,$ , where $m - h \geq 1$ as $m \geq 2$ . Hence, by Lemma 1 with $\varphi ( s ) = \exp ( - ( m - h ) s ^ { 2 } / 2 ) , p _ { \mathrm { s p l i t } }$ is a valid p-value conditional on $u ,$ and thus unconditionally.

Combination. By a union bound over both tests, $\operatorname* { P r } [ q _ { t } \leq x ] \leq \operatorname* { P r } [ p _ { \mathrm { c o o r d } } \leq x / 2 ] + \operatorname* { P r } [ p _ { \mathrm { s p l i t } } \leq$ $x / 2 ] \leq x$ . By a union bound over the N contexts, $\begin{array} { r } { \operatorname* { P r } [ Q \leq \delta ] \leq \sum _ { t = 1 } ^ { N } \operatorname* { P r } [ q _ { t } \leq \delta / N ] \leq \delta . } \end{array}$ □

Proposition 3. Let $P _ { \xi }$ be a detector that is sound over the key (Equation (2)), i.e.,

$$
\forall \omega \in \Sigma ^ { * } , \forall \alpha \in [ 0 , 1 ] , \quad \operatorname* { P r } _ { \xi } [ P _ { \xi } ( \omega ) \leq \alpha ] \leq \alpha .
$$

Given a corpus $C = ( \omega _ { 1 } , \ldots , \omega _ { n } )$ , thresholds and m independent keys $\xi _ { 1 } ^ { i } , \dots , \xi _ { m } ^ { i }$ per text, Algorithm 2 counts the false positives $\begin{array} { r } { X _ { i } ( \alpha ) : = \sum _ { j = 1 } ^ { m } \mathbb { 1 } \{ P _ { \xi _ { i } ^ { i } } ( \omega _ { i } ) \leq \alpha \} } \end{array}$ , and rejects $P _ { \xi } i f q _ { i } ( \alpha ) : =$ $T ( m , \alpha , X _ { i } ( \alpha ) ) \leq \eta : = ( 1 - ( 1 - \delta ) ^ { 1 / n } ) / | A | f o r$ some text $\omega _ { i }$ and threshold $\alpha \in { \mathcal { A } }$ Then, for every corpus C, Algorithm 2 rejects $P _ { \xi }$ with probability at most $\delta .$

Proof. Fix a text $\omega _ { i }$ and a threshold $\alpha \in { \mathcal { A } }$ . Because the keys $\xi _ { 1 } ^ { i } , \ldots , \xi _ { m } ^ { i }$ are $\mathrm { i . i . d . } , X _ { i } ( \alpha )$ is a sum of m i.i.d. Bernoulli random variables with parameter $\operatorname* { P r } _ { \xi } [ P _ { \xi } ( \omega _ { i } ) \leq \alpha ] \leq$ α by soundness over the key. Hence, by Lemma 2, $\operatorname* { P r } [ q _ { i } ( \alpha ) \leq \eta ] \leq \bar { \eta } .$ , and by a union bound over the thresholds, the event $E _ { i } : = \{ \exists \alpha \in { \mathcal { A } } : q _ { i } ( \alpha ) \leq \eta \}$ has probability at most $| \mathcal { A } | \eta = 1 - ( 1 - \delta ) ^ { 1 / n }$ . The events $E _ { 1 } , \ldots , E _ { n }$ depend on disjoint sets of independent keys, so they are independent, and

$$
\operatorname* { P r } [ \mathcal { R } \neq \varnothing ] = 1 - \prod _ { i = 1 } ^ { n } ( 1 - \operatorname* { P r } [ E _ { i } ] ) \leq 1 - ( 1 - \delta ) = \delta .
$$

Proposition 4. Let $P _ { \xi }$ be a detector that is sound given a key (Equation (3)) for the human text distribution ${ \mathcal { H } } _ { 0 } ,$ , thefraction $\beta ,$ and the thresholds , i.e.,

$$
\forall \alpha \in \mathcal { A } , \quad \operatorname* { P r } _ { \xi } [ \operatorname* { P r } _ { \Omega \sim \mathcal { H } _ { 0 } } [ P _ { \xi } ( \Omega ) \leq \alpha ] > \alpha ] \leq \beta .
$$

Algorithm 3 draws d independent keys $\xi _ { j }$ and,for each key, n independent texts $\omega _ { 1 } ^ { j } , \ldots , \omega _ { n } ^ { j } \sim \mathcal { H } _ { 0 } ,$ and counts the false positives $\begin{array} { r } { X _ { j } ( \alpha ) : = \sum _ { t = 1 } ^ { n } \mathbb { 1 } \{ P _ { \xi _ { j } } ( \omega _ { t } ^ { j } ) \leq \alpha \} } \end{array}$ . For each threshold $\alpha \in { \mathcal { A } }$ and screening level γ in a predeclared finite set ${ \bar { \cal S } } \subset ( 0 , { \bar { 1 } } ) ,$ , a key screens positive if $X _ { j } ( \alpha ) \geq c ,$ , where $c : = \operatorname* { m i n } \{ k \in \{ 1 , \ldots , n + 1 \} : T ( n , \alpha , k ) \leq \gamma \}$ is the smallest count that is significant at level $\gamma$ for a key with false positive rate α. Let $r : = T ( n , \alpha , c ) , b : = \beta + ( 1 - \beta ) r ,$ , and B the number of keys that screen positive. Algorithm 3 rejects ${ \dot { P } } _ { \xi } ~ i f \thinspace q : = T ( d , b , B ) \leq \eta : = \delta / ( | A | | S | )$ for some $( \stackrel { \cdot } { \alpha } , \gamma ) \in \mathcal { A } \times \mathcal { S }$ . Then Algorithm 3 rejects $P _ { \xi }$ with probability at most $\delta .$

Proof. Fix $( \alpha , \gamma ) \in \mathcal { A } \times \mathcal { S }$ , and note that $c , r$ and b are deterministic. For a key $\xi ,$ let $g ( \xi ) : =$ $\mathrm { P r } _ { \Omega \sim \mathcal { H } _ { 0 } } [ P _ { \xi } ( \Omega ) \leq \alpha ]$ be its false positive rate, and say that $\xi$ is bad if $g ( \xi ) > \alpha$ . Given $\xi _ { j } .$ , the texts $\omega _ { 1 } ^ { j } , \ldots , \omega _ { n } ^ { j }$ are i.i.d., so $X _ { i } ( \alpha ) \sim$ Binomial $( n , g ( \xi _ { j } ) )$ ). If $\xi _ { j }$ is not bad, $g ( \xi _ { j } ) \leq \alpha _ { \cdot }$ , and Lemma 2 gives $\operatorname* { P r } [ \tilde { X } _ { j } ( \alpha ) \geq c \mid \xi _ { j } ] \overset { \cdot } { \leq } T ( n , \alpha , c ) = r$ . As a key is bad with probability $\pi \le \beta$ by soundness given a key,

$$
\operatorname* { P r } [ X _ { j } ( \alpha ) \geq c ] \leq \pi + ( 1 - \pi ) r = r + \pi ( 1 - r ) \leq \beta + ( 1 - \beta ) r = b .
$$

Since all draws are independent across keys, B is a sum of d independent Bernoulli random variables with parameters at most $b ,$ and by Lemma $2 , \operatorname* { P r } [ q \ \leq \ \eta ] \ \leq \ \dot { \eta }$ . By a union bound over ${ \mathcal { A } } \times { \mathcal { S } } .$ $\mathrm { P r } [ \mathcal { R } \neq \varnothing ] \leq | \mathcal { A } | | \mathcal { S } | \eta = \delta$ □

## I.4 DISTORTION-FREENESS OF RESIDUAL CLOCKS

We prove that ResidualRace (Equation (6)) is distortion-free even when a context is repeated within a request, whereas the Gumbel race requires the per-request cache for that (App. A.3). The proof relies on the following classical property of exponential races.

Lemma 3. Let $E _ { 1 } , \dots , E _ { K }$ be $i . i . d . \ \mathrm { E x p { ( 1 ) } }$ random variables and $\begin{array} { r } { m _ { 1 } , \dots , m _ { K } > 0 \ w i t h \sum _ { k } m _ { k } = } \end{array}$ $\begin{array} { r } { 1 . \ L e t \tau : = \operatorname* { m i n } _ { k } { E _ { k } / m _ { k } } } \end{array}$ and κ := arg min<sub>k</sub> $E _ { k } / m _ { k }$ . Then $\operatorname* { P r } [ \kappa = k ] = m _ { k } ,$ and conditional on $( \tau , \kappa )$ , the residuals $( E _ { j } - \tau m _ { j } ) _ { j \neq \kappa }$ are i.i.d. Exp(1).

Proof. Since $E _ { k } / m _ { k }$ has density $m _ { k } e ^ { - m _ { k } s }$ , for all $k , s \geq 0$ and $x _ { j } \geq 0$

$$
\operatorname* { P r } [ \kappa = k , \tau \in \mathrm { d } s , E _ { j } - s m _ { j } > x _ { j } \forall j \neq k ] = m _ { k } e ^ { - m _ { k } s } \mathrm { d } s \prod _ { j \neq k } e ^ { - ( s m _ { j } + x _ { j } ) } = m _ { k } \cdot e ^ { - s } \mathrm { d } s \cdot \prod _ { j \neq k } e ^ { - x _ { j } } .
$$

The density factorizes, so $\kappa ,$ τ and the residuals are independent with the claimed distributions.

Proposition 5. Assume that the initial scores $\tilde { U } _ { a , u }$ of distinct (seed, unit) pairs are i.i.d. uniform. Then, for ResidualRace, $\operatorname* { P r } [ \omega _ { t } = v \mid \omega _ { < t } ] = p _ { t } ( v )$ for every step t and token $v \in \Sigma$ , even ifthe seed $a _ { t }$ has already been used in the request.

Proof. Let $\mathcal { F } _ { t }$ be the information generated by $\omega _ { \leq t }$ and the race outcomes $( \tau _ { s } , u _ { s } ) _ { s \leq t }$ . We show by induction the invariant: conditional on $\mathcal { F } _ { t }$ , the current clocks $R _ { a , u }$ of all (seed, unit) pairs are i.i.d. Exp(1), where we identify a clock that is not yet initialized with its initial value $- \log \tilde { U } _ { a , u } .$ At $t = 0$ , this holds because $- \log \tilde { U } _ { a , u } \sim \mathrm { E x p } ( 1 )$

Assume the invariant holds at $t - 1$ . Given $\mathcal { F } _ { t - 1 }$ , the distribution $p _ { t } .$ , the masses $M _ { t }$ and the seed $a _ { t }$ are fixed, and the race of Equation (6) is an exponential race over the i.i.d. $\exp ( 1 )$ clocks $R _ { a _ { t } , u }$ with weights $M _ { t } ( u )$ . By Lemma $3 , \mathrm { P r } [ u _ { t } = u \mid \bar { \mathcal { F } _ { t - 1 } } ] = M _ { t } ( u )$ , and since the token is then sampled inside $u _ { t }$ with private randomness,

$$
\operatorname* { P r } [ \omega _ { t } = v \mid \mathcal { F } _ { t - 1 } ] = M _ { t } ( \mathcal { G } ( v ) ) \cdot \frac { p _ { t } ( v ) } { M _ { t } ( \mathcal { G } ( v ) ) } = p _ { t } ( v ) .
$$

Because $p _ { t }$ is a function of $\omega _ { < t }$ , taking the expectation given $\omega _ { < t }$ proves the claim at step t. For the invariant, by Lemma $3 ,$ conditional on $\mathcal { F } _ { t - 1 }$ and $( \tau _ { t } , u _ { t } )$ , the updated losing clocks $R _ { a _ { t } , u } - \tau _ { t } M _ { t } ( u )$ are i.i.d. Exp(1). The winning clock is replaced by a fresh Exp(1), and all other clocks (other seeds, or units with $\dot { M _ { t } } ( u ) = 0 )$ are unchanged and independent of the race. The token $\omega _ { t }$ only depends on independent private randomness, so the invariant holds at t. □

## I.5 VALIDITY OF THE DETECTORS

We derive the null distribution of the detectors of App. A.6 whose validity is not immediate. Throughout, the text ω is fixed and the randomness is over the key only, so that an exact null distribution implies Equation (2).

Proposition 6. Assume that, for every fixed geometry B and every score $Z$ drawn according to its geometry, d( $B , Z ) \sim \mathcal { U } ( [ 0 , 1 ] )$ . Then, for every text ω, the Predictive Gamma statistic of Equation (12) satisfies $S ( \omega ) \sim \mathrm { G a m m a } ( N , 1 )$ .

Proof. The retained pairs D only depend on $\omega ,$ so $N$ is fixed and, by the ideal PRNG assumption, $Z _ { 1 } , \dots , Z _ { N }$ are i.i.d. The estimate $B _ { i }$ is computed from $A _ { 0 } , \ldots , A _ { K - 1 }$ and $Z _ { 1 } , \dots , Z _ { i - 1 }$ only, and $Z _ { i }$ is independent of $( Z _ { 1 } , \ldots , Z _ { i - 1 } )$ . Hence, conditional on $Z _ { 1 } , \dots , Z _ { i - 1 } , B _ { i }$ is fixed and $d _ { i } = d ( B _ { i } , Z _ { i } ) \sim \mathcal { U } ( [ 0 , 1 ] )$ . Since $d _ { 1 } , \dotsc , d _ { i - 1 }$ are functions of $Z _ { 1 } , \ldots , Z _ { i - 1 } , d _ { i }$ is independent of $( d _ { 1 } , \ldots , d _ { i - 1 } ) $ , and by induction $d _ { 1 } , \ldots , d _ { N }$ are i.i.d. uniform. Thus, the $- \log d _ { i }$ are i.i.d. Exp(1) and $S ( \omega ) \sim \mathrm { G a m m a } ( N , 1 )$ □

Proposition 7. Assume that $\hat { p } _ { t }$ only depends on $\omega _ { < t }$ . Then, for every text $\omega ,$ , the variables $( R _ { t } ) _ { t \in D _ { \mathrm { p o s } } } o f$ the Position evidence detector are i.i.d. uniform on [0, 1]. In particular, $S ( \omega ) \mathrm { \sim G a m m a } ( | D _ { \mathrm { p o s } } | , 1 )$ , and the weighted statistic $\sum { _ { t \in D _ { \mathrm { p o s } } } } - \eta _ { t }$ log $R _ { t }$ has the distribution $\begin{array} { r } { o f \sum _ { t \in D _ { \mathrm { p o s } } } \eta _ { t } E _ { t } f o r i . i . d . \ E _ { t } \sim \mathrm { E x p } ( 1 ) } \end{array}$

Proof. Since $\hat { p } _ { t }$ only depends on $\omega _ { < t }$ , the competitors $C _ { t } .$ , the exponents $\alpha _ { t , v } , \lambda _ { t }$ and the weights $\eta _ { t }$ are fixed. The seeds $a _ { t }$ for $t \in D _ { \mathrm { p o s } }$ are distinct and $\omega _ { t } \notin C _ { t }$ , so all scores $U _ { a _ { t } , \omega _ { t } }$ and $U _ { a _ { t } , v }$ for $v \in C _ { t }$ are distinct (seed, token) pairs, and are therefore independent and uniform. In particular, each $R _ { t }$ is a function of its own disjoint set of scores, so the $R _ { t }$ are independent, and it remains to show that each $R _ { t }$ is uniform.

Fix t and write $U : = U _ { a _ { t } , \omega _ { t } } , \lambda : = \lambda _ { t }$ and $W : = W _ { t }$ . Given $U = u$ , the competitors are independent, so $\begin{array} { r } { \operatorname* { P r } [ W = 1 \mid U = u ] = \prod _ { v \in C _ { \star } } \operatorname* { P r } [ U _ { a _ { t } , v } < u ^ { \alpha _ { t , v } } ] = u ^ { \lambda } } \end{array}$ . Let $Y : = W + U$ , which ranks every winning outcome above every losing one, and then by the value of U. Y has a continuous distribution, and its survival function is, for $W = 1$ and for $W \overset { \cdot } { = } 0$ respectively,

$$
\operatorname* { P r } [ Y ^ { \prime } \geq 1 + u ] = \int _ { u } ^ { 1 } x ^ { \lambda } \mathrm { d } x = { \frac { 1 - u ^ { 1 + \lambda } } { 1 + \lambda } } ,
$$

$$
\operatorname* { P r } [ Y ^ { \prime } \geq u ] = { \frac { 1 } { 1 + \lambda } } + \int _ { u } ^ { 1 } ( 1 - x ^ { \lambda } ) \mathrm { d } x = 1 - u + { \frac { u ^ { 1 + \lambda } } { 1 + \lambda } } ,
$$

where $Y ^ { \prime }$ is an independent copy of Y . Hence $R _ { t } = \operatorname* { P r } [ Y ^ { \prime } \geq Y \mid Y ]$ , and by the probability integral transform, $R _ { t } \sim \dot { \mathcal { U } ( [ 0 , 1 ] ) }$ . □

## J PROMPTS

In this section, we provide the prompt given to the research agent (App. J.1) and the prompts used for the back-translation and paraphrasing attacks during evaluation (App. J.2).

## J.1 AGENT PROMPT

The following prompt is given to the autonomous research agent.

Watermark design prompt

Your goal is to design a new distortion-free watermarking scheme in the lm-wm-tools library that significantly outperforms prior work across all key dimensions: detectability, robustness, and diversity. For diversity and PPL, the goal is to be as close as possible to the corresponding unwatermarked values (the leaderboard ranks based on the distance to the unwatermarked values).

You should focus on new experimental ideas and reason mathematically about watermarking to make genuine breakthroughs rather than tweaking hyperparameters. Also, a constraint of the challenge is to focus only on distortion-free schemes.

A key challenge of watermark development is to make sure that:

• The scheme is distortion-free.

• The scheme is robust to edits to the generated text.

Each of your submissions should be accompanied by:

• A proof that your scheme is distortion-free and well-calibrated.

• A commit message.

Before an official submission, use the lm-wm-tools verify and lm-wm-tools evaluate commands to evaluate your proposed schemes. We provide public configuration but you are also welcome to use your owns. You have a limited number of official submissions (use wm-harness limit to see how many remain), so only evaluate schemes that you are confident (i) are sound and (ii) outperform prior work or submissions. Soundness, in particular, is a significant constraint: your scheme should satisfy the following properties as much as possible (reported results are for baselines verification).

Let K be the watermark key, Σ the vocabulary,  the LLM probability distribution, $\mathcal { L } _ { K }$ the watermarked LLM probability distribution, and $p _ { K } ( \omega )$ the detector p-value for text ω under key K.

Distortion-free

$$
\forall \omega \in \Sigma ^ { * } , \mathbb { E } _ { K } [ \mathcal { L } _ { K } ( \omega ) ] = \mathcal { L } ( \omega )
$$

<table><tr><td>Scheme</td><td>Context 1</td><td>Context 2</td><td>Context 3</td><td>Context 4</td><td>Context 5</td></tr><tr><td>AAR</td><td>1</td><td>0.4194</td><td>0.5642</td><td>1</td><td></td></tr><tr><td>SynthID</td><td>0.5092</td><td>0.6584</td><td>1</td><td>0.4254</td><td>1</td></tr><tr><td>TextSeal</td><td>1</td><td>1</td><td>1</td><td>1</td><td></td></tr></table>

Soundness over the key

$$
\forall \omega \in \Sigma ^ { * } , \forall \alpha \in [ 0 , 1 ] , \operatorname* { P r } _ { K } [ p _ { K } ( \omega ) \leq \alpha ] \leq \alpha
$$

<table><tr><td>Scheme</td><td>Context 1</td><td>Context 2</td><td>Context 3</td><td>Context 4</td><td>Context 5</td></tr><tr><td>AAR</td><td>0.009677</td><td>0.02317</td><td>0.1124</td><td>0.2463</td><td></td></tr><tr><td>SynthID</td><td>0.1516</td><td>0.4806</td><td>0.5053</td><td>1</td><td>0.9043</td></tr><tr><td>TextSeal</td><td>0.3678</td><td>1</td><td>1</td><td>1</td><td></td></tr></table>

Soundness for a deployed key

$$
\forall K , \operatorname* { P r } _ { \Omega \sim \mathcal { H } _ { 0 } } [ p _ { K } ( \Omega ) \leq \alpha ] \leq \alpha
$$

<table><tr><td>Scheme</td><td>Context 1</td><td>Context 2</td><td>Context 3</td><td>Context 4</td><td>Context 5</td></tr><tr><td>AAR</td><td>2.981e-82</td><td>4.811e-14</td><td>3.813e-05</td><td>0.07999</td><td></td></tr><tr><td>SynthID</td><td>8.056e-122</td><td>9.575e-41</td><td>4.136e-09</td><td>6.433e-05</td><td>0.2573</td></tr><tr><td>TextSeal</td><td>5.265e-67</td><td>0.001826</td><td>1</td><td>1</td><td></td></tr></table>

You are operating autonomously.

## Official submissions

When you have a submission you like, submit it for private grading through the black-box wm-harness command:

wm-harness evaluate configs/MyScheme.yaml   
wm-harness leaderboard

Run wm-harness limit before submitting to see the run’s evaluation budget: limit, used, and remaining (null means unlimited). Each new official evaluation job consumes one call, even if verification or grading fails. Duplicate requests do not consume another call; removing a queued job does not refund it. Stop submitting when no calls remain.

The wm-harness leaderboard command shows you all prior evaluation results, and you should consider it trustworthy.

wm-harness evaluate runs official verification first, then evaluates the same committed snapshot only if every check passes. One job covers both stages. Results include verification.all\_passed and verification.checks; rejected submissions have no evaluation metrics, and the CLI exits with code 2. By default, the command waits for both stages; use --no-wait to return the queued job immediately.

Before calling wm-harness evaluate:

1. Keep the working tree clean.

2. Commit the exact scheme, configuration, documentation, and required assets.

3. Submit the committed configuration path. The API grades an immutable archive of that commit and rejects dirty or uncommitted work.

Use wm-harness status JOB\_ID to inspect an asynchronous job. Status includes progress.stage, a readable progress.message, progress.updated\_at (last stage or completed grading pass), and progress.heartbeat\_at (worker heartbeat every five seconds while running). Long model startup or grading passes can leave the stage unchanged.

We automatically queue submissions, so it is better to wait until the submission is done: you can use wm-harness wait JOB\_ID to wait indefinitely for one job, or add --timeout <seconds> to bound the wait (run this command in the foreground). It is okay to wait for long periods (e.g., a few hours) to save tokens; repeatedly waiting for only one minute is inefficient. You can also run wm-harness list to list all queued API queries along with their IDs.

Use wm-harness remove JOB\_ID to remove a job that is still queued. Running or completed jobs cannot be removed. Removed jobs disappear from the job list and can be submitted again. Be aware that official evaluation can take a few minutes to a few hours, depending on the scheme implementation.

To help you, lm-wm-tools verify and lm-wm-tools evaluate are public subsets of the verification and evaluation stages run by wm-harness evaluate. You can use them locally to estimate whether your submission is worth evaluating on the private test.

API credentials for public paraphrasing evaluation are installed at /home/sandbox/secret.sh. Before running lm-wm-tools evaluate, load them in the same shell with set -a; source /home/sandbox/secret.sh; set +a so the CLI inherits the credentials.

## Modifying lm-wm-tools

If you want to implement a scheme that does not fit well with the lm-wm-tools framework, you can modify the library more extensively. However, for the harness to work, your changes must remain compatible with the lm-wm-tools verify and lm-wm-tools evaluate commands. You should also make your watermark efficient and can change the library’s inner workings to achieve the required efficiency.

## Do not finish while official jobs are pending

When you are done, do not produce your final answer while any official evaluation job is queued or running. Instead, run in the foreground:

wm-harness wait

It waits until the official queue is empty and then returns the results of every submitted job. Do not background it, replace it with a custom monitoring script, or end your turn while it is running. Inspect all returned results, act on them, improve rejected or underperforming candidates, submit them with wm-harness evaluate, and run wm-harness wait again if that creates more queued work. Only finish after the official queue is empty and you have checked the returned results.

Notes

The private configuration, datasets, seeds, raw reports, and grader filesystem are intentionally unavailable. Do not attempt to infer, access, or reproduce them: this would be considered cheating.

## J.2 ATTACK PROMPTS

Here, we detail the prompts we used for the back-translation and paraphrasing attacks. In the example below, <text> is replaced with the actual text to translate or paraphrase. For back-translation, the prompt is applied twice, once for translating to French and the other time to translate back into the original language.

Back-translation prompt

Translate the following text to {language}. Your reply should only contain the translated text. <text>

## Paraphrasing prompt

Please rewrite the following text and return only the rewritten text: <text>

## K FULL EXPERIMENTAL RESULTS

In this section, we report the full numerical results behind the figures of the paper. In App. K.1, we give the results of every scheme submitted during the autoresearch trajectories of Sec. 4.1, and in App. K.2, every configuration evaluated in the component ablation of App. A.

## K.1 AUTORESEARCH TRAJECTORIES

Table 5 reports the results of the three baselines and of the 52 submissions of the four research agents from Figure 2, in submission order.

Table 5: Full results of every submission of Figure 2, in submission order. Each TPR cell reports the guaranteed / empirical / no-correction settings of App. C (quality does not depend on the setting). Parentheses mark values without correction for schemes that then fail our soundness given a key test. Within each group, bold marks the best (unrounded) value of each column and setting.
<table><tr><td rowspan=1 colspan=15>Robustness TPR@1 ↑                Quality deviation (%) ↓#Scheme                  Clean TPR@1 ↑    Deletion  Substitution   Back-trans.   Paraphrase ∆PPL ∆SB-2∆SB-3</td></tr><tr><td rowspan=1 colspan=15>Baselines</td></tr><tr><td rowspan=1 colspan=5>TextSeal                  100 / 100 / 100</td><td rowspan=1 colspan=4>14 / 34 / 34  28 / 54 / 54   89 / 97 / 97</td><td rowspan=1 colspan=6>38 / 58 / 58   4.3    2.3    7.3</td></tr><tr><td rowspan=1 colspan=5>AAR                   100 / 100 / (100)</td><td rowspan=1 colspan=1>4 / 19 / (19)</td><td rowspan=1 colspan=3>8 / 30 / (30)  76 / 92 / (92)</td><td rowspan=1 colspan=1>19 / 41 / (41)</td><td rowspan=1 colspan=5>3.9    3.9    12</td></tr><tr><td rowspan=1 colspan=5>SynthID                 100 / 100 / (100)</td><td rowspan=1 colspan=1>9 / 30 / (30)</td><td rowspan=1 colspan=3>20 / 44 / (44) 88 / 95 / (95)</td><td rowspan=1 colspan=1>35 / 56 / (56)</td><td rowspan=1 colspan=5>2.5    2.0    7.1</td></tr><tr><td rowspan=1 colspan=5>1 CascadeCopula            100 / 100 / (100)</td><td rowspan=1 colspan=1>60 / 79 / (80)</td><td rowspan=1 colspan=1>60 / 78 / (79)</td><td rowspan=1 colspan=2>95 / 99 / (99)</td><td rowspan=1 colspan=1>58 / 76 / (76)</td><td rowspan=1 colspan=5>2.3    3.9    10</td></tr><tr><td rowspan=1 colspan=5>2 LexicalCopula             100 / 100 / (100)</td><td rowspan=1 colspan=1>57 / 78 / (78)</td><td rowspan=1 colspan=1>69 / 83 / (84)</td><td rowspan=1 colspan=2>94 / 98 / (98)</td><td rowspan=1 colspan=1>63 / 78 / (79)</td><td rowspan=1 colspan=5>1.0    3.4   9.4</td></tr><tr><td rowspan=1 colspan=5>3 PortfolioRace              100 / 100 / 100</td><td rowspan=1 colspan=1>51 / 67 / 67</td><td rowspan=1 colspan=1>70 / 80 / 80</td><td rowspan=1 colspan=2>94 / 98 / 98</td><td rowspan=1 colspan=1>59 / 71 / 71</td><td rowspan=1 colspan=5>3.1    2.1    4.7</td></tr><tr><td rowspan=1 colspan=5>4 TailRace                  99 / 100 / (100)</td><td rowspan=1 colspan=1>44 / 66 / (67)</td><td rowspan=1 colspan=1>55 / 73 / (73)</td><td rowspan=1 colspan=2>87 / 94 / (94)</td><td rowspan=1 colspan=1>50 / 70 / (70)</td><td rowspan=1 colspan=5>4.8    3.1    8.3</td></tr><tr><td rowspan=1 colspan=3>5 DirectionalRenewal          100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>67 / 83 / 83</td><td rowspan=1 colspan=1>73 / 86 / 86</td><td rowspan=1 colspan=2>96 / 99 / 99</td><td rowspan=1 colspan=1>64 / 78 / 78</td><td rowspan=1 colspan=5>4.0    4.0    9.5</td></tr><tr><td rowspan=1 colspan=3>6 HaarCap                  100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>64 / 80 / 80</td><td rowspan=1 colspan=1>69 / 84 / 84</td><td rowspan=1 colspan=2>94 / 98 / 98</td><td rowspan=1 colspan=1>62 / 76 / 76</td><td rowspan=1 colspan=5>1.5    1.6   5.9</td></tr><tr><td rowspan=1 colspan=3>7 HaarGamma               100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>67 / 80 / 80</td><td rowspan=1 colspan=1>74 /84 / 84</td><td rowspan=1 colspan=2>96 / 98 / 98</td><td rowspan=1 colspan=1>66/76/76</td><td rowspan=1 colspan=5>1.5    2.3    6.7</td></tr><tr><td rowspan=1 colspan=3>8 HaarPredictive            100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>71 / 85 / (85)</td><td rowspan=1 colspan=1>77 / 88 / (88)</td><td rowspan=1 colspan=2>96 / 98 / (98)</td><td rowspan=1 colspan=1>68 / 80 / (80)</td><td rowspan=1 colspan=5>1.8   2.6    7.0</td></tr><tr><td rowspan=1 colspan=3>9 HaarGraph                100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>72 /86/ 86</td><td rowspan=1 colspan=1>77 /88 / 88</td><td rowspan=1 colspan=2>96 / 98 / 98</td><td rowspan=1 colspan=1>69 / 82 / 82</td><td rowspan=1 colspan=5>3.2    2.0    5.5</td></tr><tr><td rowspan=1 colspan=3>10 HaarLexical                 99</td><td rowspan=1 colspan=2>/ 99 / (99)</td><td rowspan=1 colspan=1>83 / 92 / (92)</td><td rowspan=1 colspan=1>76 / 86 / (86)</td><td rowspan=1 colspan=2>92 / 97 / (97)</td><td rowspan=1 colspan=1>66 / 79 / (79)</td><td rowspan=1 colspan=2>2.1</td><td rowspan=1 colspan=3>2.0   5.4</td></tr><tr><td rowspan=1 colspan=2>11 HaarLexicalFirst</td><td rowspan=1 colspan=1>99 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>84 / 91 / (91)</td><td rowspan=1 colspan=1>81 / 89 / (89)</td><td rowspan=1 colspan=2>94 / 98 / (98)</td><td rowspan=1 colspan=1>68 / 82 / (82)</td><td rowspan=1 colspan=2>2.4</td><td rowspan=1 colspan=3>2.2    5.7</td></tr><tr><td rowspan=1 colspan=2>12 ProjectiveFirst</td><td rowspan=1 colspan=1>99 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>81 / 91 / 91</td><td rowspan=1 colspan=1>75/86/86</td><td rowspan=1 colspan=2>93 / 97 / 97</td><td rowspan=1 colspan=1>64 / 78 / 78</td><td rowspan=1 colspan=2>1.7</td><td rowspan=1 colspan=3>1.9    4.2</td></tr><tr><td rowspan=1 colspan=2>13 AxialFirst</td><td rowspan=1 colspan=1>99 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>83 / 92 / (92)</td><td rowspan=1 colspan=1>78 / 88 / (88)</td><td rowspan=1 colspan=2>95 / 98 / (98)</td><td rowspan=1 colspan=1>70 / 83 / (83)</td><td rowspan=1 colspan=2>3.1</td><td rowspan=1 colspan=3>2.3    5.7</td></tr><tr><td rowspan=1 colspan=2>14 RotationFirst</td><td rowspan=1 colspan=1>99 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>81 / 90 / 90</td><td rowspan=1 colspan=1>77 /86/86</td><td rowspan=1 colspan=2>93 / 97 / 97</td><td rowspan=1 colspan=1>68 / 80 / 80</td><td rowspan=2 colspan=2>1.0</td><td rowspan=2 colspan=3>1.6   4.0</td></tr><tr><td rowspan=1 colspan=2>GPT-6 ASTRA (low)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>1 FreshRace</td><td rowspan=1 colspan=3>99 / 100 / (100)</td><td rowspan=1 colspan=1>13 / 34 / (35)</td><td rowspan=1 colspan=1>14 / 37 / (37)</td><td rowspan=1 colspan=2>74 / 90 / (90)</td><td rowspan=1 colspan=1>28 / 50 / (51)</td><td rowspan=1 colspan=5>7.2    1.9    7.1</td></tr><tr><td rowspan=1 colspan=2>2FreshRace-selective</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>26 / 52 / (53)</td><td rowspan=1 colspan=1>28 / 54 / (54)</td><td rowspan=1 colspan=2>89 / 96 / (96)</td><td rowspan=1 colspan=1>44 / 65 / (65)</td><td rowspan=1 colspan=4>3.9   3.8</td><td rowspan=1 colspan=1>12</td></tr><tr><td rowspan=1 colspan=2>3 FreshRace-deployment</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>39 / 61 / (62)</td><td rowspan=1 colspan=1>43 / 69 / (69)</td><td rowspan=1 colspan=2>92 / 97 / (97)</td><td rowspan=1 colspan=1>53 / 72 / (72)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>4.9    2.9</td><td rowspan=1 colspan=1>11</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>FreshRace-channels</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>38 / 60 / 60</td><td rowspan=1 colspan=1>46 / 65 / 65</td><td rowspan=1 colspan=2>94 / 98 / 98</td><td rowspan=1 colspan=1>55 / 71 / 71</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>8.8    2.2</td><td rowspan=1 colspan=1>8.0</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>FreshRace-adaptive-channels</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>42 / 65 / (65)</td><td rowspan=1 colspan=1>44 / 66 / (67)</td><td rowspan=1 colspan=2>91 / 96 / (96)</td><td rowspan=1 colspan=1>53 / 70 / (70)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>9.8    1.7</td><td rowspan=1 colspan=1>5.3</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>FreshRace-split</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>58 / 75 / (75)</td><td rowspan=1 colspan=1>60 / 78 / (78)</td><td rowspan=1 colspan=2>96 / 98 / (98)</td><td rowspan=1 colspan=1>62 / 76 / (76)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>8.0    3.3</td><td rowspan=1 colspan=1>9.5</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>FreshRace-antithetic-reservoir</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>49 / 67 / 67</td><td rowspan=1 colspan=1>50 / 68 / 68</td><td rowspan=1 colspan=2>93 / 97 / 97</td><td rowspan=1 colspan=1>53 / 68 / 68</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.4</td><td rowspan=1 colspan=2>1.9</td><td rowspan=1 colspan=1>5.3</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>FreshRace-balanced</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>40 / 60 / 60</td><td rowspan=1 colspan=1>50/67/67</td><td rowspan=1 colspan=2>94 / 97 / 97</td><td rowspan=1 colspan=1>55 / 70 / 70</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7.4</td><td rowspan=1 colspan=2>2.8</td><td rowspan=1 colspan=1>9.0</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>FreshRace-lexical</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>51 / 71 / 71</td><td rowspan=1 colspan=1>64 / 81 / 81</td><td rowspan=1 colspan=2>95 / 98 / 98</td><td rowspan=1 colspan=1>59/75/75</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7.3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.4</td><td rowspan=1 colspan=1>7.4</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>FreshRace-lexical</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>51 / 69 / 69</td><td rowspan=1 colspan=1>66 / 81 / 81</td><td rowspan=1 colspan=2>95 / 98 / 98</td><td rowspan=1 colspan=1>59 / 74 / 74</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.6</td><td rowspan=1 colspan=1>7.7</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>RenewalRace</td><td rowspan=1 colspan=1>95</td><td rowspan=1 colspan=2>/ 98 / (98)</td><td rowspan=1 colspan=1>80 / 90 / (91)</td><td rowspan=1 colspan=1>52 / 71 / (74)</td><td rowspan=1 colspan=2>82 / 92 / (92)</td><td rowspan=1 colspan=1>52 / 69 / (72)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4.7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.2</td><td rowspan=3 colspan=1>5.88.3</td></tr><tr><td rowspan=1 colspan=1>OPUS</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>TallyMark</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>37 / 63 / (64)</td><td rowspan=1 colspan=1>57 / 77 / (78)</td><td rowspan=1 colspan=1>86 / 94 /</td><td rowspan=1 colspan=1>(95)</td><td rowspan=2 colspan=1>44 / 63 / (64)</td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>4.1</td><td rowspan=1 colspan=2>3.1</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>TallyMark</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>32 / 57 / (57)</td><td rowspan=1 colspan=1>34 / 60 / (60)</td><td rowspan=1 colspan=1>87 / 95 /</td><td rowspan=1 colspan=1>(95)</td><td rowspan=1 colspan=1>42 / 64 / (64)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.1</td><td rowspan=1 colspan=2>1.7</td><td rowspan=1 colspan=1>7.4</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>TallyMark</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>33 / 61 / (62)</td><td rowspan=1 colspan=1>60 / 80 / (81)</td><td rowspan=1 colspan=1>88 / 95 /</td><td rowspan=1 colspan=1>(95)</td><td rowspan=1 colspan=1>47 / 67 / (67)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.2</td><td rowspan=1 colspan=2>3.6</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>TallyMark</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>36 / 64 / (64)</td><td rowspan=1 colspan=1>57 / 80 / (80)</td><td rowspan=1 colspan=2>88 / 97 / (97)</td><td rowspan=1 colspan=1>49 / 69 / (70)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2.8</td><td rowspan=1 colspan=2>3.4</td><td rowspan=1 colspan=1>9.2</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>ChoirMark</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>57 / 75 / (75)</td><td rowspan=1 colspan=1>78 / 87 / (87)</td><td rowspan=1 colspan=2>96 / 99 / (99)</td><td rowspan=1 colspan=1>62 / 75 / (75)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.8</td><td rowspan=1 colspan=2>14</td><td rowspan=1 colspan=1>34</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>CanonMark</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>43 / 59 / 59</td><td rowspan=1 colspan=1>66/78/78</td><td rowspan=1 colspan=2>93 / 96 / 96</td><td rowspan=1 colspan=1>51  / 64 / 64</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>4.2</td><td rowspan=1 colspan=2>2.6</td><td rowspan=1 colspan=1>4.3</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>LadderMark</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>51/66 /66</td><td rowspan=1 colspan=1>67 / 78 / 78</td><td rowspan=1 colspan=2>95 / 98 / 98</td><td rowspan=1 colspan=1>56 / 68 / 68</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3.8</td><td rowspan=1 colspan=2>2.7</td><td rowspan=1 colspan=1>4.6</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>PreludeMark</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>73 / 88 / (88)</td><td rowspan=1 colspan=1>79 / 90 / (91)</td><td rowspan=1 colspan=2>98 / 99 / (99)</td><td rowspan=1 colspan=1>70 / 84 / (84)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=2>5.8</td><td rowspan=1 colspan=1>18</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=1>ConsortMark</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>65 / 79 / (79)</td><td rowspan=1 colspan=1>70 / 82 / (82)</td><td rowspan=1 colspan=2>97 / 99 / (99)</td><td rowspan=1 colspan=1>62 / 75 /(75)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>5.7</td><td rowspan=1 colspan=2>1.6</td><td rowspan=1 colspan=1>5.1</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>SestetMark</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>66 / 82 / (82)</td><td rowspan=1 colspan=1>78 / 86 / (86)</td><td rowspan=1 colspan=2>97 / 99 / (99)</td><td rowspan=1 colspan=1>67 / 78 / (78)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>9.0</td><td rowspan=1 colspan=2>1.3</td><td rowspan=1 colspan=1>5.1</td></tr><tr><td rowspan=1 colspan=1>11</td><td rowspan=1 colspan=1>MotetMark</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>66 / 82 / (82)</td><td rowspan=1 colspan=1>77 / 87 / (87)</td><td rowspan=1 colspan=2>97 / 99 / (99)</td><td rowspan=1 colspan=1>62 /78 /(78)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7.2</td><td rowspan=1 colspan=2>1.6</td><td rowspan=1 colspan=1>5.8</td></tr><tr><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>CodaMark</td><td rowspan=1 colspan=1>100 / 1</td><td rowspan=1 colspan=2>00 / (100)</td><td rowspan=1 colspan=1>66 / 81 / (81)</td><td rowspan=1 colspan=1>77 / 86 /(86)</td><td rowspan=1 colspan=2>97 / 99 /(99)</td><td rowspan=1 colspan=1>65 / 79 / (79)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>8.0    1.5</td><td rowspan=1 colspan=1>5.4</td></tr><tr><td rowspan=1 colspan=1>13</td><td rowspan=1 colspan=1>AccentMark</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>69/82/82</td><td rowspan=1 colspan=1>83 / 90 / 90</td><td rowspan=1 colspan=2>99 / 100 / 100</td><td rowspan=1 colspan=1>74 / 84 / 84</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>8.0</td><td rowspan=1 colspan=2>1.5</td><td rowspan=1 colspan=1>6.0</td></tr><tr><td rowspan=1 colspan=1>14</td><td rowspan=1 colspan=1>RowMark</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>67 / 84 / 84</td><td rowspan=1 colspan=1>82 / 91 / 91</td><td rowspan=1 colspan=2>99 / 100 / 100</td><td rowspan=1 colspan=1>74 / 86 / 86</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6.3</td><td rowspan=1 colspan=2>1.6</td><td rowspan=1 colspan=1>5.8</td></tr><tr><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>UnisonMark</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>68 / 82 / 82</td><td rowspan=1 colspan=1>84 / 93 / 93</td><td rowspan=1 colspan=2>99 / 100 / 100</td><td rowspan=1 colspan=1>74 / 85 / 85</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6.4</td><td rowspan=1 colspan=3>1.3    4.9</td></tr><tr><td rowspan=1 colspan=1>16</td><td rowspan=1 colspan=1>ConcordMark</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>65 / 79 / 79</td><td rowspan=1 colspan=1>89 / 94 / 94</td><td rowspan=2 colspan=2>99 / 100 / 100</td><td rowspan=2 colspan=1>75 / 84 / 84</td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1>5.5</td><td rowspan=1 colspan=3>1.2    4.3</td></tr><tr><td rowspan=2 colspan=1>GEMI1</td><td rowspan=2 colspan=1>NI-3.8 FLASH (high)OmniSeal</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>18 / 41 / 41</td><td rowspan=1 colspan=1>34 / 58 / 58</td><td rowspan=1 colspan=2>91 / 97 / 97</td><td rowspan=1 colspan=1>45 / 62 / 62</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>6.8</td><td rowspan=1 colspan=3>5.2    16</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=2>OmniSeal                 100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>16 / 40 / 40</td><td rowspan=1 colspan=1>34 / 57 / 57</td><td rowspan=1 colspan=2>93 / 98 / 98</td><td rowspan=1 colspan=1>44 / 64 / 64</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7.4</td><td rowspan=1 colspan=3>5.0    15</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=2>OmniSeal                 100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>16 / 40 / 40</td><td rowspan=1 colspan=1>32 / 58 / 58</td><td rowspan=1 colspan=2>92/98/98</td><td rowspan=1 colspan=1>44 / 63 / 63</td><td rowspan=1 colspan=2>5.6</td><td rowspan=1 colspan=3>4.9    15</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=2>OmniSeal                 100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>16 / 40 / 40</td><td rowspan=1 colspan=1>35 / 61 / 61</td><td rowspan=1 colspan=2>92 / 97 / 97</td><td rowspan=1 colspan=1>42 / 64 / 64</td><td rowspan=1 colspan=5>6.9    5.3    16</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=2>OmniSeal-G0              100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>14 / 37 / 37</td><td rowspan=1 colspan=1>32 / 58 / 58</td><td rowspan=1 colspan=2>90 / 96 / 96</td><td rowspan=1 colspan=1>41 / 63 / 63</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>7.6    4.9    15</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=2>OmniSeal-K40              100 /</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>15 / 37 / 37</td><td rowspan=1 colspan=1>31 / 58 / 58</td><td rowspan=1 colspan=2>91 / 97 / 97</td><td rowspan=1 colspan=1>38 / 59 / 59</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>11    5.7    17</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=2>OmniSeal-K45              100 /</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>16 / 40 / 40</td><td rowspan=1 colspan=1>34 / 58 / 58</td><td rowspan=1 colspan=2>92 / 98 / 98</td><td rowspan=1 colspan=1>42 / 62 / 62</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>9.1    5.3</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=2>OmniSeal-K35              100 /</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>13 / 36/ 36</td><td rowspan=1 colspan=1>29 / 56 / 56</td><td rowspan=1 colspan=2>92 / 98 / 98</td><td rowspan=1 colspan=1>38 / 60 / 60</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>14    6.0</td><td rowspan=1 colspan=1>17</td></tr><tr><td rowspan=1 colspan=1>9</td><td rowspan=1 colspan=2>OmniSeal-W05             100 /</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>17 / 39 / 39</td><td rowspan=1 colspan=1>31 / 57 / 57</td><td rowspan=1 colspan=2>92 / 98 / 98</td><td rowspan=1 colspan=1>41 / 61 / 61</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>5.1    5.1</td><td rowspan=1 colspan=1>16</td></tr><tr><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=2>OmniSeal-C4              100 /</td><td rowspan=1 colspan=1>100 /</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>5 / 16 / 16</td><td rowspan=1 colspan=1>14 / 34 / 34</td><td rowspan=1 colspan=2>85 / 94 / 94</td><td rowspan=1 colspan=1>25 / 45 / 45</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>6.1    3.7    11</td></tr><tr><td rowspan=1 colspan=3>11 OmniSeal-K30              100 /</td><td rowspan=1 colspan=2>100 / 100</td><td rowspan=1 colspan=1>12 / 32 / 32</td><td rowspan=1 colspan=1>26 / 53 / 53</td><td rowspan=1 colspan=8>92 / 98 / 98  38 / 60 / 60    18    7.0    19</td></tr></table>

Setup We use the private evaluation of Sec. 4: for each metric, we report the worst result over the 5 watermark keys, each evaluated on 1000 replies. We evaluate every scheme under the three p-value corrections of App. C: guaranteed (p-values multiplied by $1 / \beta = \dot { 2 } 0 )$ , empirical (smallest multiplier $\kappa \geq 1$ that passes our soundness tests, Figure 24), and no correction (κ = 1). As in App. C, we first remove the outer multiplier chosen by the agent (if any) and keep the rest of the detector unchanged. For each watermark key, all settings share the same generated replies and the same attacked texts:

Table 6: Watermark unit and context seeding (App. A.2 and A.3). Each cell reports identity grouping / Lexical group.
<table><tr><td></td><td></td><td colspan="4">Robustness TPR@1 ↑</td><td colspan="3">Quality deviation (%) ↓</td></tr><tr><td>Variant</td><td>Clean TPR@1 ↑</td><td>Deletion</td><td>Substitution</td><td>Back-trans.</td><td>Paraphrase</td><td>∆PPL</td><td>∆SB-2</td><td>∆SB-3</td></tr><tr><td>Core (fixed k = 2, Gumbel race, Gamma)</td><td>100 / 100</td><td>56/ 54</td><td>67 / 76</td><td>97 / 98</td><td>64 / 68</td><td>0.1 / 1.1</td><td>4.5 / 4.6</td><td>16 / 16</td></tr><tr><td>Fixed-length</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>k = 0</td><td>100 / 99</td><td>92/88</td><td>70/61</td><td>93 / 90</td><td>67/ 60</td><td>30/ 32</td><td>12 / 13</td><td>29 /31</td></tr><tr><td>k = 1</td><td>99 / 99</td><td>57 / 51</td><td>51 /55</td><td>89 / 89</td><td>50/ 49</td><td>2.7 / 2.5</td><td>5.5 / 4.1</td><td>14/ 11</td></tr><tr><td>k = 3</td><td>100 / 100</td><td>25 / 25</td><td>42 / 60</td><td>96/97</td><td>52 / 54</td><td>0.9 / 0.4</td><td>3.1 / 2.1</td><td>11 /8.1</td></tr><tr><td>Unordered</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pair {ωt−1, v}</td><td>100 / 100</td><td>91 / 90</td><td>91 / 94</td><td>99 / 99</td><td>82 / 85</td><td>0.2 / 0.0</td><td>13 / 12</td><td>34 /32</td></tr><tr><td>k = 2</td><td>100 / 100</td><td>58 / 56</td><td>68 / 79</td><td>97 / 98</td><td>62 / 67</td><td>0.6 / 1.0</td><td>4.1 / 3.7</td><td>15 / 14</td></tr><tr><td>k = 3</td><td>100 / 100</td><td>22 /21</td><td>43 / 56</td><td>97/97</td><td>50/51</td><td>0.0 / 0.9</td><td>2.5 / 2.2</td><td>9.5 / 8.0</td></tr><tr><td>Adaptive and global length</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Adaptive length</td><td>100 / 100</td><td>73 / 70</td><td>77 / 79</td><td>98 / 98</td><td>71 / 71</td><td>2.3 / 1.9</td><td>5.7 / 5.4</td><td>17 / 16</td></tr><tr><td>+ Excess detector</td><td>100 / -</td><td>72/-</td><td>78/-</td><td>98 /-</td><td>72/-</td><td>1.0/ -</td><td>5.6/-</td><td>17/ -</td></tr><tr><td>Global + local</td><td>100 / 100</td><td>88 / 85</td><td>76/77</td><td>96 / 96</td><td>70 /69</td><td>16/18</td><td>10 / 9.2</td><td>25 / 23</td></tr><tr><td>Global first use</td><td>100 / 100</td><td>90/ 85</td><td>76/80</td><td>97/98</td><td>72 /72</td><td>15 / 16</td><td>13 / 11</td><td>34 /28</td></tr><tr><td>Anchored</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Anchored  $( S _ { \mathrm { m a x } } = 4 )$ </td><td>96/95</td><td>38/ 35</td><td>21 / 33</td><td>71 /72</td><td>30 /28</td><td>3.0 / 1.0</td><td>5.6/3.9</td><td>16 / 12</td></tr><tr><td>Suffix cascade and counting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Suffix cascade</td><td>100 / 100</td><td>78 /76</td><td>85 / 86</td><td>99 / 99</td><td>76/78</td><td>2.3 / 0.5</td><td>9.5 / 7.6</td><td>26 / 22</td></tr><tr><td>Tally</td><td>100 / 100</td><td>75 / 71</td><td>89/ 93</td><td>99/ 99</td><td>78 /77</td><td>1.2 / 2.2</td><td>25 / 13</td><td>61 / 34</td></tr><tr><td>Ladder K = 3</td><td>100 / 100</td><td>72 / 72</td><td>80 / 90</td><td>98/ 99</td><td>75 /73</td><td>4.6 / 1.5</td><td>25 / 14</td><td>61 / 36</td></tr><tr><td>Ladder K = 3 + floor</td><td>100 / 100</td><td>76/76</td><td>81 / 91</td><td>99 / 99</td><td>73 / 75</td><td>2.8 / 1.1</td><td>25 / 13</td><td>61 / 35</td></tr><tr><td>Ladder K = 4</td><td>100 / 100</td><td>77 /73</td><td>84 / 87</td><td>99/ 99</td><td>77 /76</td><td>0.7 / 0.5</td><td>25 / 14</td><td>62 / 35</td></tr><tr><td>Ladder K = 4 + floor</td><td>100 / 100</td><td>77 / 75</td><td>80/ 86</td><td>99/ 99</td><td>74/78</td><td>2.4 / 0.1</td><td>25 / 13</td><td>62 / 36</td></tr><tr><td>On Signed sphere + ResidualRace</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fixed-length k = 2</td><td>100 / 100</td><td>33 / 35</td><td>47/63</td><td>96/95</td><td>49/53</td><td>0.7 / 1.0</td><td>0.4 / 0.4</td><td>0.4 / 0.3</td></tr><tr><td>Fixed-length k = 1 Unordered pair</td><td>100 / 100 100 / 100</td><td>75 / 75 72/76</td><td>74/81 72/82</td><td>98/97 97/97</td><td>72/71 67 /72</td><td>0.5 / 0.1 1.2 / 1.0</td><td>0.3 / 0.7</td><td>2.1 / 0.3</td></tr><tr><td>Global + local (k = 1)</td><td>100 / 100</td><td>86/ 84</td><td>74/76</td><td>96/ 94</td><td>68 / 65</td><td>0.5 / 0.7</td><td>0.1 / 0.1</td><td>1.7 / 0.9</td></tr><tr><td></td><td>100 / 100</td><td>89 / 85</td><td>78/ 81</td><td>96/96</td><td>73 /70</td><td>0.7 / 1.8</td><td>0.6 / 0.8</td><td>2.5 / 2.1</td></tr><tr><td>Global first use(k = 1)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.1 / 0.3</td><td>1.1 / 1.4</td></tr></table>

only the multiplier applied to the p-values changes. Hence, the quality columns do not depend on the setting.

Reading the table Each TPR cell reports the guaranteed / empirical / no-correction results, and the first column gives the submission index (i.e., the x-axis of Figure 2). Parentheses mark the results without correction of the schemes that then fail our soundness given a key test (i.e., whose empirical multiplier is above 1). Note that Figure 2 instead uses the multiplier chosen by each agent: 20 for GPT-6 ASTRA (except for the first two submissions of the low variant), between 1 and 6 for OPUS 5, and none for GEMINI-3.8 FLASH and the baselines. Moreover, as we evaluate the settings on newly generated replies (with the same watermark keys), the results can differ by a few points from those of Figure 2.

## K.2 COMPONENT ABLATION

Tables 6–9 report all the configurations of the component ablation of App. A, following the same structure: watermark unit and context seeding, watermark score, logits transformation, and detection.

Setup We use the experimental setup of App. A.1: each configuration swaps one or a few components of the simple baseline scheme (identity grouping, Fixed-length context seeding with $k = 2 ,$ $\tilde { U } = U$ , the Gumbel race, and the Gamma detector). All configurations are evaluated on the private evaluation of Sec. 4 with a single watermark key, and their p-values are multiplied by $1 / \beta \bar { = } 2 0$ so that they are sound given a key (Sec. 3.2). Unlike the bar plots of App. A, which show differences, we report absolute values with the same columns as Table 1: the TPR@1%FPR on clean text and under each attack, and the relative distance (in %) to the unwatermarked model in perplexity and bigram and trigram Self-BLEU.

Reading the tables Each cell reports the result with the identity grouping (left) and with the Lexical group (App. A.2) (right), so that comparing the two sides of a cell gives the effect of the watermark unit on that configuration. The first row of each table is the baseline scheme, which every other configuration modifies. In Table 9, only the detector changes: each block rescores the exact same generated texts, so the quality columns are those of the corresponding generation. A dash indicates a configuration that we only evaluated with the identity grouping.

Table 7: Watermark score (App. A.4). Each cell reports identity grouping / Lexical group.
<table><tr><td></td><td></td><td colspan="4">Robustness TPR@1 ↑</td><td colspan="3">Quality deviation (%) ↓</td></tr><tr><td>Variant</td><td>Clean TPR@1 ↑</td><td>Deletion</td><td>Substitution</td><td>Back-trans.</td><td>Paraphrase</td><td>∆PPL</td><td>∆SB-2</td><td>∆SB-3</td></tr><tr><td>Core (fixed k = 2, Gumbel race, Gamma)</td><td>100 / 100</td><td>56/ 54</td><td>67 / 76</td><td>97/ 98</td><td>64 / 68</td><td>0.1 / 1.1</td><td>4.5 / 4.6</td><td>16 / 16</td></tr><tr><td>Prelude</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>On the core</td><td>100 / 100</td><td>54/52</td><td>67 / 74</td><td>96/97</td><td>62 /63</td><td>3.2 / 2.6</td><td>2.4 / 2.8</td><td>11/11</td></tr><tr><td>On Ladder K = 4 + floor</td><td>100 / 100</td><td>74/70</td><td>78 / 84</td><td>99 / 98</td><td>74/73</td><td>3.2 / 1.0</td><td>6.2 / 5.3</td><td>18 / 15</td></tr><tr><td>Gaussian copula</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ρ= 0.7</td><td>99 / 100</td><td>23 / 20</td><td>29/ 39</td><td>83 / 84</td><td>34/37</td><td>0.3 / 0.4</td><td>0.3 / 0.6</td><td>1.7 / 3.5</td></tr><tr><td>ρ = 0.8</td><td>100 / 100</td><td>34/33</td><td>42/ 56</td><td>89/ 91</td><td>45 / 50</td><td>0.9 / 1.2</td><td>0.5 / 0.4</td><td>4.6 /4.1</td></tr><tr><td>ρ = 0.85</td><td>100 / 100</td><td>36/38</td><td>50/63</td><td>93 / 95</td><td>49/55</td><td>0.1 / 3.0</td><td>1.4 / 0.9</td><td>6.5 / 5.5</td></tr><tr><td>ρ = 0.9</td><td>100 / 100</td><td>44/44</td><td>57/67</td><td>94/95</td><td>54/59</td><td>1.7 / 0.1</td><td>1.0 / 2.0</td><td>7.1 / 8.3</td></tr><tr><td>ρ = 0.95</td><td>100 / 100</td><td>52/ 52</td><td>63 / 71</td><td>97/ 97</td><td>62 / 62</td><td>0.6 / 1.4</td><td>2.2 / 2.5</td><td>9.7 / 10</td></tr><tr><td>Gaussian copula + common-pair filter</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ρ= 0.7</td><td>99 / 100</td><td>20 / 19</td><td>28 / 38</td><td>80/ 84</td><td>34/35</td><td>1.6 / 0.3</td><td>0.1 / 0.1</td><td>2.2 / 2.5</td></tr><tr><td>ρ = 0.8</td><td>100 / 100</td><td>32/ 35</td><td>39/52</td><td>90/92</td><td>43 / 48</td><td>1.1 / 0.7</td><td>0.3 / 0.9</td><td>3.5 / 4.6</td></tr><tr><td>ρ = 0.85</td><td>100 / 100</td><td>38 /36</td><td>47 / 60</td><td>92/ 92</td><td>49/53</td><td>1.3 / 1.4</td><td>0.6 / 1.5</td><td>4.7 / 6.7</td></tr><tr><td>ρ = 0.9</td><td>100 / 100</td><td>44/43</td><td>50/64</td><td>94/95</td><td>53/55</td><td>0.3 / 0.8</td><td>0.8 / 1.5</td><td>5.9/ 7.0</td></tr><tr><td>ρ = 0.95</td><td>100 / 100</td><td>48 / 50</td><td>59/ 72</td><td>96/ 98</td><td>61 / 64</td><td>0.1 / 1.0</td><td>1.2 / 2.3</td><td>7.3 /9.4</td></tr><tr><td>ρ = 1</td><td>100 / 100</td><td>57 /56</td><td>64/75</td><td>97 / 98</td><td>63 / 68</td><td>2.4 / 1.6</td><td>3.3 / 4.0</td><td>13 / 14</td></tr><tr><td>Private sampling</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>r = 0.1</td><td>100 / 100</td><td>47 / 45</td><td>58 / 67</td><td>95 / 97</td><td>57 / 62</td><td>0.9 / 1.3</td><td>2.2 / 2.6</td><td>9.7 / 10</td></tr><tr><td>r = 0.2</td><td>100 / 100</td><td>42/ 38</td><td>52 / 62</td><td>93 / 93</td><td>49/ 57</td><td>2.5 / 1.6</td><td>1.1 / 1.5</td><td>6.6 / 7.1</td></tr><tr><td>r = 0.3</td><td>99 /99</td><td>30 / 30</td><td>39 /52</td><td>87 / 89</td><td>44/48</td><td>4.0 / 3.5</td><td>0.6 / 0.9</td><td>4.6 /5.1</td></tr><tr><td>Tail indicator</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Tail indicator</td><td>99 / 99</td><td>27 / 26</td><td>42 /48</td><td>84 / 86</td><td>39 / 42</td><td>2.2 / 1.3</td><td>0.3 / 0.7</td><td>4.0 / 4.6</td></tr><tr><td>Independent Channels</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>L = 4</td><td>100 / 100</td><td>51 / 47</td><td>60 / 71</td><td>96/97</td><td>60 /61</td><td>2.0 / 1.4</td><td>2.3 / 1.9</td><td>8.1 / 6.9</td></tr><tr><td>L = 16</td><td>100 / 100</td><td>39 / 38</td><td>56/63</td><td>96/ 94</td><td>48/ 52</td><td>2.3 / 0.2</td><td>0.3 / 0.0</td><td>2.1 / 1.0</td></tr><tr><td>L = 32</td><td>100 / 100</td><td>36 / 34</td><td>48 /62</td><td>94/95</td><td>46/50</td><td>0.0 / 1.1</td><td>0.5 / 0.3</td><td>0.4 / 0.2</td></tr><tr><td>L = 64</td><td>100 / 100</td><td>32 / 30</td><td>46/ 57</td><td>92 / 94</td><td>42/49</td><td>0.3 / 1.5</td><td>0.6 / 0.5</td><td>0.8 / 0.7</td></tr><tr><td>L = 128</td><td>100 / 100</td><td>29 /28</td><td>41 /57</td><td>91 / 92</td><td>42 /48</td><td>0.8 / 0.4</td><td>0.4 / 0.6</td><td>0.7 / 1.3</td></tr><tr><td>Correlated Channels</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Latin L = 4</td><td>100 / 100</td><td>50 / 48</td><td>64 /72</td><td>96/97</td><td>56/61</td><td>2.0 / 2.9</td><td>1.6 / 1.5</td><td>6.5 / 5.8</td></tr><tr><td>Antithetic L = 4</td><td>100 / 100</td><td>46/43</td><td>58 / 70</td><td>97/95</td><td>60/59</td><td>0.3 / 0.4</td><td>1.6 / 0.6</td><td>6.6 / 4.3</td></tr><tr><td>Latin L = 16</td><td>100 / 100</td><td>40 /40</td><td>51 / 64</td><td>95 / 96</td><td>50/52</td><td>0.4 / 0.6</td><td>0.2 / 0.1</td><td>0.5 / 1.2</td></tr><tr><td>Stratified L = 16</td><td>100 / 100</td><td>40/37</td><td>52 / 64</td><td>94 / 94</td><td>50/48</td><td>1.0 / 1.7</td><td>0.1 / 0.3</td><td>1.0 / 0.0</td></tr><tr><td>Gaussian direction + Gumbel race</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>d = 1</td><td>100 / 100</td><td>36/37</td><td>50 /59</td><td>95/ 95</td><td>52/ 53</td><td>1.4 / 0.5</td><td>2.8 / 2.4</td><td>11 /9.0</td></tr><tr><td>d = 2</td><td>100 / 100</td><td>28 /28</td><td>41 / 50</td><td>92/ 93</td><td>46/47</td><td>0.2 / 0.5</td><td>0.5 / 0.6</td><td>3.5 / 3.1</td></tr><tr><td>d = 4</td><td>100 / 100</td><td>21 / 18</td><td>31 /44</td><td>89 / 90</td><td>37/38</td><td>2.3 / 1.4</td><td>0.8 / 0.5</td><td>0.6 / 0.3</td></tr><tr><td>d = 8</td><td>100 / 100</td><td>15 / 15</td><td>24/37</td><td>83 / 86</td><td>31/ 33</td><td>1.8 / 1.0</td><td>0.1 / 0.7</td><td>0.0 / 0.7</td></tr><tr><td>Gaussian direction + ResidualRace</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>d = 1</td><td>100 / 100 100 / 100</td><td>44 /43 34/33</td><td>58/ 72 48 /65</td><td>97 / 98</td><td>55 /62</td><td>1.5 / 2.1</td><td>5.4/ 3.6</td><td>18 / 13</td></tr><tr><td>d = 2</td><td>100 / 100</td><td>29 / 27</td><td>39/ 53</td><td>96 /96</td><td>51 /55</td><td>0.9 / 1.1</td><td>1.1 / 1.1</td><td>5.1 / 5.1</td></tr><tr><td>d = 4 d = 8</td><td>100 / 100</td><td>20 / 20</td><td>32 / 48</td><td>94/95 91 / 92</td><td>45 / 50 38/41</td><td>0.2 / 0.8 1.5 / 2.7</td><td>0.2 / 0.3 0.9 / 0.6</td><td>1.3 / 0.7 0.9 / 0.3</td></tr><tr><td>Spherical + Gumbel race</td></table>

Table 8: Logits transformation (App. A.5). Each cell reports identity grouping / Lexical group.
<table><tr><td></td><td></td><td colspan="4">Robustness TPR@1 ↑</td><td colspan="3">Quality deviation (%)↓</td></tr><tr><td>Variant</td><td>Clean TPR@1 ↑</td><td>Deletion</td><td>Substitution</td><td>Back-trans.</td><td>Paraphrase</td><td>∆PPL</td><td>∆SB-2</td><td>∆SB-3</td></tr><tr><td>Core (fixed k = 2, Gumbel race, Gamma)</td><td>100 / 100</td><td>56 / 54</td><td>67 / 76</td><td>97 / 98</td><td>64 / 68</td><td>0.1 / 1.1</td><td>4.5 / 4.6</td><td>16 /16</td></tr><tr><td>Races with private randomness</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BandRace</td><td>98 / 98</td><td>16 /16</td><td>20 / 29</td><td>73 / 75</td><td>29 /29</td><td>2.0 / 0.7</td><td>0.3 / 0.1</td><td>1.7 / 2.5</td></tr><tr><td>PoolRace (K = 16)</td><td>100 / 100</td><td>33 /27</td><td>44/50</td><td>93 / 92</td><td>46/ 52</td><td>3.2 / 0.2</td><td>0.8 / 1.0</td><td>5.3 / 5.6</td></tr><tr><td>MomentBridge (K = 8)</td><td>88 / 88</td><td>3/4</td><td>6/7</td><td>37 / 36</td><td>8/ 11</td><td>1.9 / 1.2</td><td>0.5 / 0.8</td><td>0.2 / 0.7</td></tr><tr><td>Quantile</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>QuantileRace</td><td>93 / 91</td><td>4/5</td><td>7 / 12</td><td>44 /46</td><td>12 / 13</td><td>1.4 / 0.3</td><td>2.6 / 1.6</td><td>10 / 7.5</td></tr><tr><td>+ Wheel</td><td>85 / 86</td><td>0/0</td><td>1/2</td><td>19 / 21</td><td>3/4</td><td>2.7 / 0.8</td><td>0.5 / 0.7</td><td>1.3 / 1.4</td></tr><tr><td>Transport (L = K independent channels)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TransportRace K = 32</td><td>99 / 99</td><td>14/11</td><td>26 /35</td><td>83 / 85</td><td>31 /32</td><td>0.3 / 0.5</td><td>0.3 / 0.6</td><td>0.7 / 1.4</td></tr><tr><td>TransportRace K = 128</td><td>100 / 100</td><td>23 / 20</td><td>32 /48</td><td>87 / 88</td><td>38 /40</td><td>1.8 / 0.9</td><td>0.8 / 1.2</td><td>2.1 / 2.5</td></tr><tr><td>QuotaTransport K = 32</td><td>99 / 100</td><td>19 / 18</td><td>30 / 40</td><td>87 / 87</td><td>37 /35</td><td>2.2 / 0.2</td><td>1.1 / 1.0</td><td>2.4 / 1.9</td></tr><tr><td>QuotaTransport K = 128</td><td>100 / 100</td><td>22 /20</td><td>34 /48</td><td>88 / 90</td><td>41 /39</td><td>0.3 / 0.3</td><td>0.6 / 1.4</td><td>1.4 / 2.8</td></tr><tr><td>Residual clocks</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ResidualRace</td><td>100 / 100</td><td>63 / 63</td><td>72 /82</td><td>98 / 99</td><td>70 / 74</td><td>1.5 / 0.9</td><td>9.0 / 6.3</td><td>27 / 21</td></tr><tr><td>Skeleton</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Last important + Content gate</td><td>99 / 98</td><td>54/47</td><td>26/ 30</td><td>84 / 80</td><td>46/ 44</td><td>0.5 / 0.6</td><td>1.0 / 0.1</td><td>3.0 / 1.0</td></tr><tr><td>Interleaved + Interleaved gate</td><td>99 / 98</td><td>49 /45</td><td>42 /46</td><td>88 / 85</td><td>48 / 46</td><td>0.3 / 1.3</td><td>2.3 / 1.4</td><td>6.0 / 4.1</td></tr></table>

Table 9: Detection (App. A.6), on fixed generated texts: only the detector changes, so the quality is that of the generation. Each cell reports identity grouping / Lexical group.
<table><tr><td colspan="2"></td><td colspan="4">Robustness TPR@1 ↑</td><td colspan="4">Quality deviation (%) ↓</td></tr><tr><td>Variant</td><td>Clean TPR@1 ↑</td><td>Deletion</td><td>Substitution</td><td>Back-trans.</td><td>Paraphrase</td><td></td><td>∆PPL</td><td>∆SB-2</td><td>∆SB-3</td></tr><tr><td colspan="8">Core texts, no forward pass</td></tr><tr><td>Gamma</td><td>100 / 100</td><td>56/54</td><td>67 /76</td><td>97 / 98</td><td>64 /68</td><td>0.1 / 1.1</td><td>4.5 / 4.6</td><td>16 /16</td></tr><tr><td>Gaussian</td><td>100 / 100</td><td>40 /40</td><td>51 / 64</td><td>95 / 96</td><td>54/57</td><td>0.1 / 1.1</td><td>4.5 / 4.6</td><td>16 /16</td></tr><tr><td>Clustering</td><td>100 / 100</td><td>57 / 55</td><td>67 / 77</td><td>97 / 98</td><td>64 /68</td><td>0.1 / 1.1</td><td>4.5 / 4.6</td><td>16 / 16</td></tr><tr><td>Excess</td><td>100 / 100</td><td>57 / 55</td><td>68 /77</td><td>97/98</td><td>64 /68</td><td>0.1 / 1.1</td><td>4.5 / 4.6</td><td>16 / 16</td></tr><tr><td colspan="9">Core texts, forward pass</td></tr><tr><td>Weighted Gamma</td><td>100 / 100</td><td>59/56</td><td>73 /81</td><td>99 / 100</td><td>73 /79</td><td>0.1 / 1.1</td><td>4.5 / 4.6</td><td>16/16</td></tr><tr><td>Position evidence</td><td>100 / 100</td><td>57 /56</td><td>70/80</td><td>98 / 99</td><td>68 / 74</td><td>0.1 / 1.1</td><td>4.5 / 4.6</td><td>16/16</td></tr><tr><td>+ weighting</td><td>100 / 100</td><td>60/ 60</td><td>76/85</td><td>99 / 100</td><td>76/82</td><td>0.1 / 1.1</td><td>4.5 / 4.6</td><td>16 / 16</td></tr><tr><td colspan="9">Independent Channels</td></tr><tr><td>L = 4, Bonferroni</td><td>100 / 100</td><td>51 / 47</td><td>60/71</td><td>96/97</td><td>60/61</td><td>2.0 / 1.4</td><td>2.3 / 1.9</td><td>8.1 / 6.9</td></tr><tr><td>L = 4, Šidák</td><td>100 / 100</td><td>51 / 47</td><td>60/71</td><td>96/ 97</td><td>60/61</td><td>2.0 / 1.4</td><td>2.3 / 1.9</td><td>8.1 / 6.9</td></tr><tr><td>L = 16, Bonferroni</td><td>100 / 100</td><td>39/38</td><td>56/63</td><td>96/94</td><td>48 /52</td><td>2.3 / 0.2</td><td>0.3 / 0.0</td><td>2.1 / 1.0</td></tr><tr><td>L = 16, Šidák</td><td>100 / 100</td><td>39/ 38</td><td>56/63</td><td>96/ 94</td><td>48 / 52</td><td>2.3 / 0.2</td><td>0.3 / 0.0</td><td>2.1 / 1.0</td></tr><tr><td>L = 32, Bonferroni</td><td>100 / 100</td><td>36/ 34</td><td>48 /62</td><td>94/95</td><td>46/50</td><td>0.0/ 1.1</td><td>0.5 / 0.3</td><td>0.4 / 0.2</td></tr><tr><td>L = 32, Šidák</td><td>100 / 100</td><td>36 /34</td><td>48 /62</td><td>94/95</td><td>46/50</td><td>0.0/ 1.1</td><td>0.5 / 0.3</td><td>0.4 / 0.2</td></tr><tr><td>L = 64, Bonferroni</td><td>100 / 100</td><td>32/30</td><td>46/57</td><td>92 / 94</td><td>42/49</td><td>0.3 / 1.5</td><td>0.6 / 0.5</td><td>0.8 / 0.7</td></tr><tr><td>L = 64, Šidák</td><td>100 / 100</td><td>32/30</td><td>46/57</td><td>92/94</td><td>42 /49</td><td>0.3 / 1.5</td><td>0.6 / 0.5</td><td>0.8 / 0.7</td></tr><tr><td>L = 128, Bonferroni</td><td>100 / 100</td><td>29/28</td><td>41 /57</td><td>91 / 92</td><td>42/48</td><td>0.8 / 0.4</td><td>0.4 / 0.6</td><td>0.7 / 1.3</td></tr><tr><td>L = 128, Šidák</td><td>100 / 100</td><td>29/28</td><td>41/57</td><td>91 / 92</td><td>42/48</td><td>0.8 / 0.4</td><td>0.4 / 0.6</td><td>0.7 / 1.3</td></tr><tr><td colspan="9">Signed sphere + ResidualRace texts</td></tr><tr><td>Predictive Gamma</td><td>100 / 100</td><td>33 / 35</td><td>47/63</td><td>96/95</td><td>49/53</td><td>0.7 / 1.0</td><td>0.4 / 0.4</td><td>0.4 / 0.3</td></tr><tr><td>Resultant length</td><td>100 / 100</td><td>19/19</td><td>27 / 43</td><td>91 /92</td><td>37 /40</td><td>0.7 / 1.0</td><td>0.4 / 0.4</td><td>0.4 / 0.3</td></tr><tr><td>Spherical caps</td><td>100 / 100</td><td>22 / 24</td><td>31 /50</td><td>89/ 91</td><td>37 / 40</td><td>0.7 / 1.0</td><td>0.4 / 0.4</td><td>0.4 / 0.3</td></tr><tr><td>Length + caps</td><td>100 / 100</td><td>24 /26</td><td>35/53</td><td>92/93</td><td>40/44</td><td>0.7 / 1.0</td><td>0.4 / 0.4</td><td>0.4 / 0.3</td></tr><tr><td>Anchor distance</td><td>100 / 100</td><td>32/32</td><td>43 /59</td><td>94 / 94</td><td>44 /49</td><td>0.7 / 1.0</td><td>0.4 / 0.4</td><td>0.4 / 0.3</td></tr><tr><td colspan="9">Gaussian direction + Gumbel race texts</td></tr><tr><td>Gaussian norm d = 1</td><td>100 / 100</td><td>36/37</td><td>50/59</td><td>95/95</td><td>52/53</td><td>1.4 / 0.5</td><td>2.8 / 2.4</td><td>11 / 9.0</td></tr><tr><td>Residual energy d = 1</td><td>100 / 100</td><td>50/51</td><td>65 /70</td><td>97/97</td><td>61 / 63</td><td>1.4 / 0.5</td><td>2.8 / 2.4</td><td>11 / 9.0</td></tr><tr><td>Gaussian norm d = 2</td><td>100 / 100</td><td>28 /28</td><td>41/50</td><td>92/ 93</td><td>46/47</td><td>0.2 / 0.5</td><td>0.5 / 0.6</td><td>3.5 /3.1</td></tr><tr><td>Residual energy d = 2</td><td>100 / 100</td><td>36/36</td><td>49/58</td><td>93 / 94</td><td>51 / 52</td><td>0.2 / 0.5</td><td>0.5 / 0.6</td><td>3.5 / 3.1</td></tr><tr><td>Circle search d = 2</td><td>100 / 100</td><td>43 / 41</td><td>55/65</td><td>96/96</td><td>56/57</td><td>0.2 / 0.5</td><td>0.5 / 0.6</td><td>3.5 / 3.1</td></tr><tr><td>Gaussian norm d = 4</td><td>100 / 100</td><td>21 / 18</td><td>31/ 44</td><td>89 / 90</td><td>37 /38</td><td>2.3 / 1.4</td><td>0.8 / 0.5</td><td>0.6 / 0.3</td></tr><tr><td>Residual energy d = 4</td><td>100 / 100</td><td>24 /22</td><td>35 /47</td><td>89 / 90</td><td>40 /40</td><td>2.3 / 1.4</td><td>0.8 / 0.5</td><td>0.6 / 0.3</td></tr><tr><td>Gaussian norm d = 8</td><td>100 / 100</td><td>15 / 15</td><td>24/37</td><td>83 / 86</td><td>31 /33</td><td>1.8 / 1.0</td><td>0.1 / 0.7</td><td>0.0 / 0.7</td></tr><tr><td>Residual energy d = 8</td><td>100 / 100</td><td>16 / 14</td><td>24/36</td><td>82 / 85</td><td>31 /31</td><td>1.8 / 1.0</td><td>0.1 / 0.7</td><td>0.0 / 0.7</td></tr></table>