# Online Self-Weighted Fine-Tuning

Haiquan Wen<sup>1,\*</sup> Yiwei He<sup>1,\*</sup> Bei Peng<sup>2</sup> Guangliang Cheng<sup>1,†</sup>

<sup>1</sup>University of Liverpool <sup>2</sup>University of Sheffield

<sup>\*</sup>Equal contribution. <sup>†</sup>Correspondence: guangliang.cheng@liverpool.ac.uk

## Abstract

Standard supervised fine-tuning (SFT) assigns the same explicit loss weight to every expert demonstration, regardless of the model’s changing competence over training queries. Reinforcement learning (RL) based methods adapt update strength using model-generated rollouts, but often require substantially more sampling and can be unstable on hard tasks. We propose Online Self-Weighted Fine-Tuning (OSW-FT), a simple method that augments SFT with online, trajectory-level weighting. For each query, OSW-FT estimates the model’s current success rate using a small number of inferenceonly rollouts and rescales the standard SFT loss accordingly. The optimization direction remains anchored to the expert trajectory, while the update magnitude adapts online. For binaryverifiable reasoning, we connect this weighting to SFT and RL at the gradient level, inspired by variance-reduction principles. The resulting estimator is unbiased for the exact OSW-FT surrogate update for any finite rollout count, and we analyze convergence with respect to the corresponding surrogate objective. Evaluated across Qwen3 series ranging from 0.6B to 4B on multiple challenging benchmarks (e.g., AIME), OSW-FT consistently improves over SFT on small-to-medium scale models. OSW-FT offers a favorable compute-performance trade-off as a practical approach for fine-tuning small-to-medium LLMs on binary-verifiable reasoning tasks with only 2 online rollouts.

## 1 Introduction

Post-training a LLM raises a deceptively simple question: how much should each training sample update the model at its current stage of training? This question is especially consequential for verifiable reasoning tasks, where the model’s competence on individual training queries can vary substantially and evolves throughout training. Some queries may already be reliably solved by the model and provide limited additional learning signal, while others remain near its capability frontier and warrant concentrated optimization effort. Standard Supervised Fine-Tuning (SFT) (Ouyang et al., 2022), however, assigns a uniform sample weight to every expert demonstration throughout training, treating already-mastered and unresolved queries identically. While this design is simple and stable, it allocates optimization effort coarsely across examples with very different learning value.

Reinforcement learning (RL) based post-training methods (Guo et al., 2025; Schulman et al., 2017) address this issue from a different direction. By computing advantages from model-generated rollouts, they adapt update strength to the model’s current behavior and can focus learning on more informative examples. This adaptivity is appealing, but it comes with substantial online sampling cost and optimization instability, especially on hard reasoning tasks where failing to discover any correct trajectories entirely deprives the model of a viable learning direction. SFT avoids this failure mode by always providing a high-quality expert trajectory as the supervision target, but lacks a mechanism to modulate how strongly that trajectory should be learned from. This suggests a natural middle ground: preserve the stable, expert-anchored optimization direction of SFT, while introducing a lightweight online signal that adjusts update magnitude according to the model’s current competence.

We propose Online Self-Weighted Fine-Tuning (OSW-FT), a simple post-training method for verifiable reasoning that changes how much to learn from each expert trajectory without changing what to learn from. For each training query, OSW-FT performs a small number of inference-only rollouts to estimate the model’s current pass rate pˆ(q), and reweights the standard SFT loss by a factor of (1 − pˆ(q)). This weight is largest for queries the model consistently fails on, thereby concentrating optimization effort near the capability frontier, and naturally decreases toward zero as the model masters a query. Crucially, OSW-FT uses online rollouts only to estimate this scalar weight; it does not optimize on model-generated trajectories, does not perform token-level reward shaping, and does not modify the training distribution. This design targets settings with a limited online rollout budget while retaining expert-trajectory supervision.

For binary-verifiable reasoning, we construct an explanatory framework connecting this weighting to RL control variates at the gradient level. OSW-FT restores this quantity online through a Monte Carlo estimate from a small number of rollouts. Under our analysis, the resulting estimator is unbiased for the exact OSW-FT surrogate update for any finite rollout count K, and the convergence analysis applies to the corresponding surrogate objective under the stated assumptions. The method is naturally most useful when the training distribution remains challenging for the current model; when most training queries are already solved, the online weight correspondingly diminishes.

We evaluate OSW-FT on Qwen3 series (Yang et al., 2025) models ranging from 0.6B to 4B parameters across AIME (AIME, 2025), AMC (AMC, 2023), MATH-500 (Lightman et al., 2023), and GPQA-Diamond (Rein et al., 2024). Empirically, OSW-FT consistently improves over standard SFT on small-to-medium model scales. Compared with GRPO (Shao et al., 2024), the results vary across models and metrics: OSW-FT performs favorably in several limited-rollout settings, whereas GRPO remains stronger in some configurations. OSW-FT is therefore not intended as a general replacement for $\operatorname { R L } ;$ it targets binary-verifiable reasoning when the available online rollout budget is limited. Ablation studies further validate the benefit of continuous online success-rate weighting over alternative hard-thresholding or random heuristics.

In summary, this work makes the following core contributions: First, we introduce OSW-FT, an expert-trajectory-anchored method that modulates SFT updates using an online success-rate weight for verifiable reasoning. Second, we analyze finiterollout unbiasedness and convergence with respect to the proposed surrogate update. Third, evaluations show that OSW-FT consistently improves over standard SFT on small-to-medium scales and provides a favorable compute–performance tradeoff under a limited online rollout budget.

## 2 Online Self-Weighted Fine-Tuning

Supervised Fine-Tuning (SFT) aligns language models by maximizing the log-likelihood of expert demonstrations. Given a dataset $\mathcal { D } = \{ ( q , o ^ { * } ) \}$ of input queries $q$ and ground-truth output trajectories $o ^ { * }$ , SFT trains the model $\pi _ { \theta }$ by applying a uniform weight of 1 to the gradient of every ground-truth trajectory:

$$
\begin{array} { r l } & { - \nabla _ { \boldsymbol { \theta } } \mathcal { J } _ { \mathrm { S F T } } = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S F T } } } \\ & { \qquad = - \mathbb { E } _ { ( \boldsymbol { q } , \boldsymbol { o } ^ { * } ) \sim \mathcal { D } } \left[ 1 \cdot \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( \boldsymbol { o } ^ { * } \mid \boldsymbol { q } ) \right] } \end{array}\tag{1}
$$

From a classical estimation perspective, this uniform weighting is strictly principled as the standard maximum likelihood estimator over the data distribution. However, from an online optimization perspective, applying full gradient magnitude to queries the model has already mastered $( \pi _ { \theta } ( o ^ { * } \mid q )  1 )$ becomes statistically inefficient; it injects redundant gradient noise regarding the evolving policy and dampens the learning efficacy on unresolved frontier queries. To derive a principled variance-reduced gradient estimator that dynamically modulates this update intensity, we cast the problem within the framework of score function estimation.

## 2.1 Score Function Decomposition and Optimal Control Variate

For verifiable reasoning tasks, let $R ( o ) \in \{ 0 , 1 \}$ denote the binary correctness of a generated trajectory. The alignment objective is to maximize expected correctness:

$$
\mathcal { I } ( \theta ) = \mathbb { E } _ { q \sim \mathcal { D } } \mathbb { E } _ { o \sim \pi _ { \theta } ( \cdot | q ) } [ R ( o ) ]\tag{2}
$$

Applying the log-derivative trick and subtracting a query-dependent control variate $c ( q ) -$ which leaves the expected gradient unbiased since $\mathbb { E } _ { o \sim \pi _ { \theta } } [ \nabla _ { \theta } \log \pi _ { \theta } ( o \mid q ) ] = 0 -$ yields the standard variance-reduced score function estimator:

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { J } = \mathbb { E } _ { q \sim \mathcal { D } } \mathbb { E } _ { o \sim \pi _ { \theta } } \bigg [ } \\ & { \qquad \quad \ ( R ( o ) - c ( q ) ) \nabla _ { \theta } \log \pi _ { \theta } ( o \mid q ) \bigg ] } \end{array}\tag{3}
$$

We seek optimal control variate $c ^ { * } ( q )$ that minimizes the MSE of the scalar learning signal:

$$
\operatorname* { m i n } _ { c ( q ) } \ \mathbb { E } _ { o \sim \pi _ { \theta } } \left[ ( R ( o ) - c ( q ) ) ^ { 2 } \right]\tag{4}
$$

Setting the derivative to zero yields the exact analytical solution $c ^ { * } ( q ) = \mathbb { E } _ { o \sim \pi _ { \theta } } [ R ( o ) \mid q ]$ , which for binary tasks simplifies to the model’s current success probability $p _ { s } ( q )$ . Crucially, this optimal control variate is identical to the advantage of the oracle trajectory:

$$
\begin{array} { r l } & { A ( o ^ { * } , q ) \triangleq R ( o ^ { * } ) - \mathbb { E } _ { o \sim \pi _ { \theta } } [ R ( o ) \mid q ] } \\ & { \qquad = 1 - p _ { s } ( q ) } \end{array}\tag{5}
$$

The MSE-minimizing baseline and the unbiased advantage estimator for $o ^ { * }$ are therefore identical for binary rewards, providing a unified statistical justification for the signal magnitude derived below. Substituting $c ^ { * } ( q ) = p _ { s } ( q )$ and applying the Law of Total Expectation to decompose by reward outcome, we observe a dual role of $p _ { s } ( q ) \mathrel { \mathop : }$ : it represents both probability of generating correct trajectory $( \operatorname* { P r } [ R = 1 ] )$ and determines its advantage magnitude:

$$
\begin{array} { c } { { \nabla _ { \theta } \mathcal { I } = \mathbb { E } _ { q \sim \mathcal { D } } \Big [ \underbrace { p _ { s } ( q ) } _ { \mathrm { P r } [ R = 1 ] } \cdot \mathbb { E } _ { \sigma \sim \pi _ { \theta } \mid R ( o ) = 1 } \Big [ } } \\ { { \mathrm { ~ } } } \\ { { \underbrace { \left( 1 - p _ { s } ( q ) \right) } _ { A ( o ^ { * } , q ) } \nabla _ { \theta } \log \pi _ { \theta } ( o \mid q ) \Big ] } } \\ { { + \left( 1 - p _ { s } ( q ) \right) \cdot \mathbb { E } _ { \sigma \sim \pi _ { \theta } \mid R ( o ) = 0 } \Big [ } } \\ { { \left( 0 - p _ { s } ( q ) \right) \nabla _ { \theta } \log \pi _ { \theta } ( o \mid q ) \Big ] \Bigg ] _ { ( 6 ) } } } \end{array}
$$

## 2.2 Oracle Substitution in SFT

The equation highlights a fundamental tension: the positive learning signal is gated by the success probability $p _ { s } ( q )$ . In the low-resource or early-training regime where $p _ { s } ( q ) \to 0$ , pure reinforcement learning methods suffer from gradient starvation, as the agent cannot explore and discover valid reasoning paths from scratch.

SFT as Deterministic Oracle Substitution. The essence of SFT is to bypass this exploration bottleneck by replacing the intractable generative probability with the absolute certainty of the dataset D. By manually setting the probability of encountering a correct trajectory strictly to unity $( \mathrm { P r } [ R = 1 ]  1 )$ , SFT not only guarantees a continuous learning signal but also mathematically nullifies the negative gradient term:

$$
\begin{array} { r l } & { \nabla _ { \theta } \mathcal { I } \approx \mathbb { E } _ { q \sim \mathcal { D } } \Big [ \underbrace { 1 } _ { \mathrm { D a t a s e t C e r t a i n t y } } \cdot } \\ & { \mathbb { E } _ { o \sim \pi _ { \theta } | R ( o ) = 1 } \left[ \underbrace { \left( 1 - p _ { s } ( q ) \right) } _ { A ( o ^ { * } , q ) } \nabla _ { \theta } \log \pi _ { \theta } ( o \mid q ) \right] \Big ] } \end{array}\tag{7}
$$

The oracle trajectory $o ^ { * }$ serves as a deterministic surrogate for the intractable conditional expectation $\mathbb { E } _ { o \sim \pi _ { \theta } | R ( o ) = 1 } [ \cdot ]$ , translating to: $\nabla _ { \theta } \log \pi _ { \theta } ( o \mid q ) \approx$ $\nabla _ { \theta } \log \pi _ { \theta } ( o ^ { * } \mid q )$

Justification for Asymmetric Substitution. A technical subtlety arises in this transition: the success probability $p _ { s } ( q )$ is substituted by 1 in the empirical frequency slot $( \operatorname* { P r } [ R = 1 ] )$ but retained within the inner advantage term $( 1 - p _ { s } ( q ) )$ . We justify this asymmetric treatment through the lens of distributional decoupling:

• Directional Alignment (Data-centric): SFT performs a strict distributional shift, replacing the model’s stochastic generative distribution $\pi _ { \theta } ( \cdot \mid R ( o ) = 1 )$ with the deterministic expert dataset D. In this expert domain, the empirical frequency of encountering an oracle trajectory is unity $( P _ { \mathit { D } } ( R = 1 ) = 1 )$ , which nullifies the negative gradient term and anchors the optimization direction to $o ^ { * }$

• Magnitude Regulation (Model-centric): While the optimization direction is dictated by the static data distribution, the update magnitude must respect the variance-reduced baseline governed by the evolving policy. Retaining $( 1 - p _ { s } ( q ) )$ serves as a crucial regularizer to evaluate the competence gap between the current policy state and the oracle target.

Consequently, this formulation is not an algebraic identity, but a regularized Surrogate Gradient that decouples data-driven direction from policy-driven step-size. Standard SFT applies the distributional shift but simplifies the policy magnitude to a static value of 1, yielding:

$$
\nabla _ { \boldsymbol { \theta } } \mathcal { J } _ { \mathrm { S F T } } \propto \mathbb { E } _ { ( \boldsymbol { q } , \boldsymbol { o } ^ { * } ) \sim \mathcal { D } } \left[ 1 \cdot \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( \boldsymbol { o } ^ { * } \mid \boldsymbol { q } ) \right]\tag{8}
$$

## 2.3 OSW-FT Gradient

Advantage-Guided SFT Magnitude. We argue that while the update frequency and direction should be guided by the oracle $o ^ { * }$ , the magnitude must respect the variance-reduced baseline. Retaining the oracle’s deterministic frequency while restoring the advantage magnitude weight yields the OSW-FT gradient:

Algorithm 1 Online Self-Weighted Fine-Tuning   
Require: Dataset D, Model $\pi _ { \theta }$ , Rollout K, Batch   
Size B   
1: while not converged do   
2: $\mathcal { B } = \{ ( q _ { i } , o _ { i } ^ { * } ) \} _ { i = 1 } ^ { B } \sim \mathcal { D }$   
3: Initialize buffer $\mathcal { M }  \emptyset$   
4: for each $q _ { i }$ in B do   
5: Sample K outputs $\{ \hat { o } _ { i , k } \} \sim \pi _ { \theta } ( \cdot | q _ { i } )$   
6: Calculate rewards and success probability   
$\hat { p } _ { i }$   
7: Compute weight $w _ { i } = 1 - \hat { p } _ { i }$   
8: Store tuple $( q _ { i } , o _ { i } ^ { * } , w _ { i } )$ into $\mathcal { M }$   
9: end for   
10: Compute weighted SFT loss on current $\pi \theta \colon$   
$\mathcal { L } ( \boldsymbol { \theta } ) = - \frac { 1 } { | \mathcal { M } | } \sum _ { ( \boldsymbol { q } , \boldsymbol { o } ^ { * } , \boldsymbol { w } ) \in \mathcal { M } } \boldsymbol { w } \cdot \log \pi _ { \boldsymbol { \theta } } ( \boldsymbol { o } ^ { * } \mid \boldsymbol { q } )$   
11: Update parameters.   
12: end while

$$
\nabla _ { \theta } \mathcal { J } _ { \mathrm { 0 S W } } \propto \mathbb { E } _ { ( q , o ^ { * } ) \sim \mathcal { D } } \big [ ( 1 - \hat { p } ( q ) ) \cdot \nabla _ { \theta } \log \pi _ { \theta } ( o ^ { * } \mid q ) \big ]\tag{9}
$$

where $\begin{array} { r c l } { \hat { p } ( q ) } & { = } & { \frac { 1 } { K } \sum _ { k = 1 } ^ { K } R ( \hat { o } _ { k } ) } \end{array}$ is an online Monte Carlo estimate of the success probability. Consequently, $( 1 - \hat { p } ( q ) )$ serves as a principled heuristic that modulates the expert trajectory update using the model’s empirical advantage.

## 2.4 Discussion

OSW-FT preserves the original data distribution and dynamically assesses the learning value of each sample online via model rollouts, re-allocating only the gradient magnitude through the advantage weight. This ability to adaptively optimize directly on datasets with unlabeled difficulty eliminates the need for manual data sorting. Unlike rollout-optimized RFT methods such as GRPO, the gradient direction in OSW-FT is anchored by the high-quality expert demonstration $o ^ { * }$ ; only the magnitude is modulated by the model’s current capability. This retains supervised optimization while incorporating an online estimate of the model’s current success rate. The additional online sampling relative to standard SFT is K inference-only rollouts per query. In our experiments, $K = 2$ captures a useful advantage signal under a limited online rollout budget. Finally, OSW-FT has a gradient-level structural connection to SFT and RL. Appendix C presents this connection under policygradient theorem (Sutton et al., 1999; Schulman et al., 2017).

## 3 Theoretical Analysis

We establish two key theoretical properties of OSW-FT: robustness to finite rollout estimation and convergence guarantees with smaller gradient second moment than SFT. Full proofs are deferred to Appendix B.

## 3.1 Robustness to Finite Rollouts

Success probability $p _ { s } ( q )$ is estimated via K rollouts: $\begin{array} { r } { \hat { p } ( q ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } R ( \hat { o } _ { k } ) } \end{array}$ . Denoting $g _ { 0 } ( q ) : =$ $\nabla _ { \theta }$ log $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { o } ^ { * } | \boldsymbol { q } )$ , we analyze the OSW-FT estimator $\hat { g } _ { K } ( q ) = ( 1 - \hat { p } ( q ) ) \cdot g _ { 0 } ( q )$

Theorem 1 (Finite Rollout Properties). For a fixed query q with $p : = p _ { s } ( q ) \ \in \ ( 0 , 1 )$ and K i.i.d. rollouts, the OSW-FT estimator is unbiased: $\mathbb { E } _ { \hat { p } } [ \hat { g } _ { K } ( \boldsymbol { q } ) ] = ( 1 - p ) \cdot g _ { 0 } ( \boldsymbol { q } )$

Remark 1 (Choice of K). The signal loss proba-$b i l i t y p ^ { K }$ predicts a qualitative transition at $K = 2$ At $K = 1$ , the weight is binary $( \{ 0 , 1 \} )$ and a query with $p = 0 . 5$ loses its signal with 50% probability, reducing OSW-FT to hard example selection. At $K = 2$ , signal loss drops to $p ^ { 2 } = 2 5 \%$ and the intermediate weight $w = 0 . 5$ enables soft discrimination. For $K \geq 2 , p ^ { K }$ is already small andfurther increases yield diminishing returns.

## 3.2 Convergence Analysis

A subtlety of OSW-FT is that the weight $1 - p _ { s } ( q )$ depends on the current policy $\pi _ { \theta } ,$ yet Algorithm 1 holds it fixed within each batch of gradient updates. To handle this, we define a surrogate objective $J ^ { ( t ) } ( \theta )$ at each outer iteration t, which freezes the weights at their current values while optimizing only the log-likelihood (see Definition 1 in Appendix). Under standard smoothness and boundedgradient assumptions (Assumption 2 in Appendix), the following guarantee holds.

Theorem 2 (Convergence Rate). Suppose OSW-FT runs for T total gradient steps with learning rate $\eta = 1 / \sqrt { T }$ , updating weights every $S$ steps. Let L be the smoothness constant of $J ^ { ( t ) }$ , B an upper bound on | log $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { o } ^ { * } | \boldsymbol { q } ) |$ , and $\epsilon _ { p }$ the maximum change in $p _ { s } ( q )$ between consecutive weight updates. Then:

$$
\begin{array} { r l } & { \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \Big [ \| \nabla _ { \theta } J ^ { ( \lfloor t / S \rfloor ) } ( \theta _ { t } ) \| ^ { 2 } \Big ] } \\ & { \le \displaystyle \frac { 2 \Delta _ { 0 } + L \cdot \Sigma ^ { 2 } } { \sqrt { T } } + B \cdot \epsilon _ { p } , } \end{array}\tag{10}
$$

where $\Delta _ { 0 }$ is the initial optimality gap. The stochastic-gradient second moment is

$$
\begin{array} { r l } & { \Sigma ^ { 2 } : = \mathbb { E } _ { q \sim D } \Big [ \big ( ( 1 - p _ { s } ( q ) ) ^ { 2 } } \\ & { \quad \quad \quad \quad + \frac { p _ { s } ( q ) ( 1 - p _ { s } ( q ) ) } { K } \Big ) \| g _ { 0 } ( q ) \| ^ { 2 } \Big ] , } \end{array}\tag{11}
$$

which satisfies $\Sigma ^ { 2 } < \Sigma _ { \mathrm { S F T } } ^ { 2 } : = \mathbb { E } _ { q \sim D } [ \| g _ { 0 } ( q ) \| ^ { 2 } ]$ for any $K \geq 1$ , whenever the model has nonzero success probability on a positive-measure set of queries.

The first term vanishes at rate $O ( 1 / \sqrt { T } )$ ; the second term $\boldsymbol { B } \cdot \boldsymbol { \epsilon } _ { p }$ reflects the cost of holding weights fixed between updates, which is small under typical learning rates. The strict inequality $\Sigma ^ { 2 } < \Sigma _ { \mathrm { S F T } } ^ { 2 }$ implies that OSW-FT converges with smaller second moment than SFT under identical conditions.

## 4 Experiments

## 4.1 Experimental Setup

We evaluate our method across model scales with the Qwen3 series (0.6B, 1.7B, and 4B) (Yang et al., 2025) to comprehensively study the interaction between model capacity and our dynamic weighting mechanism. We compare OSW-FT against standard SFT and GRPO (Shao et al., 2024).

Training Dataset. To establish a rigorous testbed for reasoning capabilities, we curated a highquality training set of 10k multiple-choice questions (MCQs) sampled from AM-Qwen3-Distilled (Tian et al., 2025) — a NuminaMATH dataset (LI et al., 2024) distilled via Qwen3-235B. To ensure the provision of high-quality oracle trajectories $( o ^ { * } )$ required for both SFT and OSW-FT, we manually filter and clean the dataset. Specifically, we rigorously enforce a single-choice answer format and constrain the reasoning trajectories to a maximum length of 4k tokens with the vast majority around 2k tokens. This length distribution ensures that even small-capacity base models can learn from substantial reasoning steps efficiently.

Experiment Details. All models undergo fullparameter fine-tuning. We employ the AdamW optimizer (Loshchilov and Hutter, 2019) with a constant learning rate schedule, alongside DeepSpeed ZeRO3 and bfloat16 precision across all model scales. To minimize generation overhead, we integrate vLLM (Kwon et al., 2023) for both training rollouts and evaluation. Its high-throughput capabilities enable us to maintain a large training rollout batch size (128), making the online cost of OSW-FT highly efficient.

The maximum sequence length is set to 5k tokens, with a maximum completion length of 4k tokens allocated for online rollouts. For OSW-FT and GRPO, we use Math-Verify (Huggingface, 2025) for binary accuracy reward function.

• SFT: We fine-tune the models for 1 epoch with a learning rate of $1 \times 1 0 ^ { - 5 }$

• OSW-FT (Ours): Following the same 1- epoch schedule and $1 \times 1 0 ^ { - 5 }$ learning rate. For each query, we generate $K = 2$ trajectories at a temperature of $T = 1 . 0$ to estimate the online success probability $\hat { p } ( q )$

• GRPO: For each query, we generate $K = 8$ trajectories (group size of $G = 8 )$ at a temperature of $T ~ = ~ 1 . 0$ We adopt $1 \times 1 0 ^ { - 6 }$ learning rate, aligning with established community best practices for stable RL training. To mitigate variance from trajectory lengths, we employ token-level policy gradient loss (DAPO, (Yu et al., 2025)). We set KL penalty coefficient $\beta = 0$ (disabling KL) and apply clipping bound of [0.8, 1.28].

Evaluation Protocol. To thoroughly assess verifiable reasoning capabilities, we report performance on several challenging benchmarks, including advanced mathematical reasoning tasks such as AMC (AMC, 2023), AIME (AIME, 2025), and MATH-500 (Lightman et al., 2023), as well as the expert-level general question answering GPQA-Diamond (Rein et al., 2024). We report Pass@1/Pass@16 for competition benchmarks (AMC, AIME). We use Pass@16 as an empirical indicator of exploration: under the same 16- sample budget, a higher Pass@16 means that the model can discover a correct solution on more queries through sampling. Additionally, for benchmarks comprising hundreds of items (MATH-500 and GPQA-Diamond), we report Pass@1 as an auxiliary test to evaluate deterministic problemsolving accuracy on relatively simpler or out-ofdistribution (OOD) tasks. We apply sampling with $T o p _ { p } { = } 0 . 9 5 , T o p _ { k } = 2 0$ , and a sampling temperature of $T = 0 . 6 .$ allowing a maximum generation length of 8k tokens.

Table 1: Main Results. Comparison across different post-training paradigms. We report Pass@1 / Pass@16 for competition benchmarks (AIME, AMC) to highlight both stability and exploration potential. For MATH-500, GPQA-Diamond, we report Pass@1. Bold indicates the best performance. <sup>†</sup> indicates the base model shows lower performance due to limited instruction-following ability, which recovers after fine-tuning.
<table><tr><td>Model</td><td>AIME 24</td><td>AIME 25</td><td>AMC 22</td><td>AMC 23</td><td>AMC 24</td><td>MATH-500</td><td>GPQA-D</td></tr><tr><td>Qwen3-0.6B-Base</td><td>00.83 / 06.67</td><td>00.21 / 03.34</td><td>11.05 / 41.86</td><td>11.82 / 54.35</td><td>04.58 / 26.67</td><td>22.00</td><td>14.14</td></tr><tr><td>+ GRPO</td><td>00.83 / 10.00</td><td>00.00 / 00.00</td><td>08.14 / 25.58</td><td>16.30 / 43.48</td><td>08.61 / 31.11</td><td>36.20</td><td>26.77</td></tr><tr><td>+ SFT</td><td>00.00 / 00.00</td><td>00.21 / 03.34</td><td>09.01 / 32.56</td><td>11.55 / 50.00</td><td>03.89 / 24.44</td><td>24.20</td><td>15.66</td></tr><tr><td>+ OSW-FT</td><td>00.83 / 06.67</td><td>01.04 / 10.00</td><td>08.72 / 41.86</td><td>13.04 / 56.52</td><td>05.14 / 28.89</td><td>24.20</td><td>20.20</td></tr><tr><td>Qwen3-1.7B-Base</td><td>03.75 / 20.00</td><td>03.96 / 20.00</td><td>23.84 / 58.14</td><td>28.67 / 65.22</td><td>14.17 / 48.89</td><td>56.40</td><td>25.76</td></tr><tr><td>+ GRPO</td><td>06.25 / 23.33</td><td>02.08 / 13.33</td><td>20.20 / 48.84</td><td>34.10 / 65.22</td><td>13.06 / 42.22</td><td>59.40</td><td>28.79</td></tr><tr><td>+ SFT</td><td>02.92 / 20.00</td><td>03.12 / 10.00</td><td>18.31 / 55.81</td><td>26.90 / 65.22</td><td>13.75 / 48.89</td><td>52.20</td><td>17.17</td></tr><tr><td>+ OSW-FT</td><td>05.42 / 26.67</td><td>03.96 / 23.34</td><td>22.24 / 55.81</td><td>27.45 / 65.22</td><td>17.78 / 48.89</td><td>57.40</td><td>26.27</td></tr><tr><td>Qwen3-4B-Base</td><td>09.58 / 30.00</td><td>06.88 / 23.34</td><td>36.63 / 69.77</td><td>39.27 / 78.26</td><td>29.31 / 60.00</td><td>69.60</td><td>10.10 †</td></tr><tr><td>+ GRPO</td><td>05.63 / 20.00</td><td>05.20 / 23.33</td><td>39.53 / 58.14</td><td>34.92 / 67.39</td><td>22.92 / 53.33</td><td>71.40</td><td>30.32</td></tr><tr><td>+ SFT</td><td>12.29 / 26.67</td><td>15.42 / 26.67</td><td>40.70 / 69.77</td><td>44.57 / 73.91</td><td>35.00 / 75.56</td><td>76.40</td><td>27.27</td></tr><tr><td>+ OSW-FT</td><td>13.33 / 36.67</td><td>16.25 / 36.67</td><td>41.42 / 74.42</td><td>45.11 / 78.26</td><td>35.00 / 75.56</td><td>79.00</td><td>32.32</td></tr></table>

## 4.2 Main Results

Table 1 presents the main comparative results. Overall, OSW-FT improves over standard SFT across the evaluated small-to-medium model scales. Its comparison with GRPO varies across models and metrics, reflecting a trade-off rather than a general performance ordering between expertanchored training and RFT.

Enhancing Exploration in Small Language Models. For the 0.6B, 1.7B, and 4B models, OSW-FT consistently outperforms or matches standard SFT on Pass@16. For instance, on the 4B model, OSW-FT achieves a Pass@16 of 36.67% on both AIME 24 and AIME 25, compared with 26.67% for SFT. From these Pass@16 results, we observe that OSW-FT encourages more exploration under the same sampling budget: it discovers correct solutions for a larger fraction of queries across repeated samples. Relative to GRPO, OSW-FT performs favorably on several competition-math metrics, while GRPO remains stronger in some other settings, particularly on Pass@1 for the smaller models.

The Capacity-Data Mismatch Phenomenon. The optimization dynamics of OSW-FT further explain its scale-dependent behavior. As illustrated in Figure 1, the advantage weight $( 1 - { \hat { p } } )$ acts

![](images/5d1f17d3d8e8c144e34d30f2bc7e90c1b076d3d00b8ee383f644b2fe2e44c60e.jpg)  
Figure 1: OSW-FT weight (1 − pˆ) stratifies by model scale, validating the model-data matching hypothesis. as an implicit curriculum, decaying as the model masters the dataset. Crucially, these trajectories cleanly stratify by model capacity: the highly capable 4B model masters the dataset rapidly, causing its weights to plummet toward zero, whereas the 0.6B model maintains a sustained learning signal.

## 4.3 Performance under Limited Rollout Budgets

As established in Remark 1 (Section 3), our theoretical analysis of the signal loss probability predicts a critical qualitative transition at K = 2. At this minimal threshold, the advantage weight shifts from binary hard-example mining to fine-grained soft discrimination. Guided by this theoretical insight, a central computational consideration is empirically verifying this optimal trade-off against the overhead introduced by the online rollouts. Because these rollouts strictly serve to estimate the baseline for variance reduction rather than to discover the correct reasoning path from scratch, minimal generation overhead should theoretically suffice. To validate this, we conduct an ablation study on the Qwen3-0.6B-Base model, varying $K \in \{ 1 , 2 , 4 , 8 \}$ . The results are shown in Table 2.

Table 2: Ablation on the number of rollouts (K). We report Pass@16 for AIME 24, AIME 25 and Pass@1 for MATH-500, GPQA-D on Qwen3-0.6B-Base. OSW-FT maintains robust performance at K = 2, while GRPO is more sensitive to the rollout count on the AIME benchmarks in this setting.
<table><tr><td rowspan="2">Rollouts</td><td colspan="4">GRPO</td><td colspan="4">OSW-FT</td></tr><tr><td>AIME 24</td><td>AIME 25</td><td>MATH-500</td><td>GPQA-D</td><td>AIME 24</td><td>AIME 25</td><td>MATH-500</td><td>GPQA-D</td></tr><tr><td> $K = 1$ </td><td></td><td></td><td></td><td></td><td>06.67</td><td>06.67</td><td>22.80</td><td>15.15</td></tr><tr><td> $K = 2$ </td><td>00.00</td><td>00.00</td><td>29.00</td><td>22.72</td><td>06.67</td><td>10.00</td><td>24.20</td><td>20.20</td></tr><tr><td>K = 4</td><td>06.67</td><td>00.00</td><td>37.00</td><td>26.27</td><td>06.67</td><td>10.00</td><td>24.00</td><td>20.71</td></tr><tr><td>K = 8</td><td>10.00</td><td>00.00</td><td>36.20</td><td>26.77</td><td>06.67</td><td>10.00</td><td>24.60</td><td>20.71</td></tr></table>

Table 3: Ablation on alternative weighting strategies $( K = 4 )$ . We report Pass@16 for AIME 24, AIME 25 and Pass@1 for MATH-500, GPQA-D on Qwen3-0.6B-Base. OSW-FT consistently outperforms alternative weighting heuristics, validating the necessity of smooth online success-rate anchoring.
<table><tr><td>Strategy</td><td>AIME 24</td><td>AIME 25</td><td>MATH-500</td><td>GPQA-D</td></tr><tr><td>Hard-thresholded</td><td>06.67</td><td>06.67</td><td>23.80</td><td>15.15</td></tr><tr><td>Random</td><td>00.00</td><td>00.00</td><td>21.40</td><td>15.17</td></tr><tr><td>OSW-FT</td><td>06.67</td><td>10.00</td><td>24.00</td><td>20.71</td></tr></table>

OSW-FT. As shown in Table 2, setting K = 1 acts as a highly noisy, binarized estimator and yields the lowest performance across all metrics (e.g., 15.15% on GPQA-D). However, simply increasing the rollouts to K = 2 produces a sharp performance improvement, jumping to 20.20% on GPQA-D and matching or exceeding $K = 1$ on all other benchmarks. Critically, the performance plateaus for $K > 2$ . Increasing the rollouts to K = 4 or K = 8 yields only negligible marginal benefits. Consequently, a coarse estimate via $K = 2$ captures the necessary advantage signal without demanding excessive generation overhead.

Contrast with GRPO under Limited Rollout Budgets. To fully contextualize this computational efficiency, we contrast OSW-FT with the GRPO baseline under matched rollout budgets. Table 2 suggests a potential limitation of pure RL: while GRPO can optimize shorter-horizon tasks (MATH, GPQA) at minimal compute $( K = 2 )$ , it obtains zero scores on the most challenging reasoning benchmarks, scoring 0.00 on both AIME 24 and AIME 25. This outcome may stem from RL’s reliance on model-generated exploration. On highly complex tasks like AIME, the probability of finding a correct long-horizon reasoning path with only 2 random attempts can be low. Consequently, the advantage signal for these hard queries can become sparse, providing the model with limited learning signal. To alleviate this exploration bottleneck and achieve non-zero AIME performance, GRPO can benefit from larger group sizes (e.g., $K = 8 )$ .

Advantage of GT Anchoring. This contrast suggests a potential advantage of OSW-FT for capacity-constrained models. By deterministically anchoring the gradient direction to the ground-truth oracle trajectory (o<sup>∗</sup>), OSW-FT provides a stable, expert-anchored optimization path across the evaluated tasks. Thus, it can mitigate the exploration bottleneck that can affect pure RL on hard queries, showing consistent performance across the evaluated benchmarks while operating at a limited computational budget $( K = 2 )$

## 4.4 Ablation on Weighting Strategies

To thoroughly investigate whether the performance gains of OSW-FT stem from the mathematically derived online success-rate weighting rather than naive data filtering or generic optimization artifacts, we conduct a rigorous ablation study on Qwen3- 0.6B-Base under rollout budget of $K = 4$ . We implement two critical control baselines:

• Hard-thresholded Weighting: A discrete control scheme where a query receives a binary weight $w \in \{ 0 , 1 \}$ . Specifically, $w = 0$ if all K rollouts succeed (indicating the query is fully mastered), and w = 1 otherwise. This mimics hard rejection sampling or hard mistake-driven learning.

• Random Weighting: A control scheme that decouples the loss weights from the model’s actual performance. It assigns random weights drawn from a uniform distribution, calibrated to match the exact empirical mean weight of OSW-FT across the dataset, thereby testing if our method simply acts as an implicit learning rate decay.

![](images/8ab1cb962478bf258482feda538e6cb76e42438ea92942877e5979436d1e0b57.jpg)  
(a) GRPO learning rate of $1 \times 1 0 ^ { - 6 }$

![](images/884986c56d7f5900dd8d385d9c91730333049f998de90a6148af8e96c9b9e85b.jpg)  
(b) GRPO learning rate of $1 \times 1 0 ^ { - 5 }$  
Figure 2: GRPO training reward under two learning rates. In our setup, the reward collapses toward zero at $1 \times 1 0 ^ { - 5 }$ across the evaluated model scales, whereas $1 \times 1 0 ^ { - 6 }$ produces the stable runs used in the main comparison.

The empirical results are summarized in Table 3. We observe a severe performance degradation when utilizing alternative heuristics. In contrast, OSW-FT scales down gradient updates as capability increases, supporting the benefit of content-aware weighting under a limited online rollout budget.

## 4.5 On the Optimization Stability of GRPO

Learning Rate Sensitivity. We initially trained the GRPO baseline using the $1 \times 1 0 ^ { - 5 }$ learning rate employed by SFT and OSW-FT. In our setup, this learning rate causes the training reward to collapse toward zero across the evaluated model scales, as shown in Figure 2b. Once all sampled trajectories receive the same reward, the relative-advantage signal vanishes and optimization stalls.

We therefore use $1 \times 1 0 ^ { - 6 }$ for the reported GRPO baseline. This lower learning rate produces the stable training dynamics shown in Figure 2a.

Late-Stage Instability and Checkpoint Selection. We also observe late-stage degradation in some GRPO runs, particularly for the smaller-capacity models. To report a stable GRPO baseline, we evaluate the checkpoint at which the reward reaches a plateau, before subsequent degradation, rather than using the final checkpoint unconditionally.

## 5 Related Work

Offline Reweighting and Regularization. A growing body of work views standard SFT through an offline Reinforcement Learning lens, aiming to mitigate over-optimization without incurring the computational cost of online generation. One line of research focuses on reweighting (e.g., SoftDedup (He et al., 2024), iw-SFT (Qin and Springenberg, 2025), DFT (Wu et al., 2026)). Another line introduces structural regularization, such as ASFT (Zhu et al., 2026), which apply KL-divergence penalties against a static reference model to prevent catastrophic forgetting and stylistic overfitting. While these approaches improve upon uniform SFT, they operate entirely offline. OSW-FT bypasses these static heuristics by operating online, directly tracking the model’s continuously evolving capabilities via real-time rollouts.

Online Rollout-Driven Alignment. To incorporate real-time generative feedback, recent paradigms increasingly rely on online rollouts (Schulman et al., 2017; Shao et al., 2024; Yu et al., 2025; Zheng et al., 2025; Gao et al., 2025; Wang et al., 2025; Chen et al., 2025). STaR (Zelikman et al., 2022) filters and trains exclusively on selfgenerated rationales that yield correct final answers. RL or hybrid-RL methods like GRPO (Shao et al., 2024) and CHORD (Zhang et al., 2026) are inherently bottlenecked by massive compute overhead. Conversely, rollout-augmented SFT methods attempt to simplify this optimization but introduce new flaws. OSFT (Li et al., 2025) fine-tunes on model-generated trajectories without filtering by correctness, and lacks explicit mechanisms for accuracy-based gradient scaling. OTR (Ming et al., 2025) guides SFT using a localized policy gradient. However, its one-token lookahead provides only myopic, token-level credit signals, lacking explicit trajectory-level credit assignment for long-horizon reasoning. OSW-FT resolves these bottlenecks by unifying trajectory evaluation with deterministic optimization.

Data Selection and Curriculum Learning. Recent findings highlight that alignment depends heavily on data quality rather than sheer quantity (Zhou et al., 2023; Li et al., 2024). To maximize data efficiency, traditional curriculum learning (Bengio et al., 2009) presents training examples in a manually predefined order of increasing difficulty. This static paradigm has largely evolved into Automatic Curriculum Learning (Narvekar et al., 2020; Portelas et al., 2020), where intermediate tasks are dynamically selected based on the agent’s real-time competence. More recently, curriculum paradigms have been adapted for LLMs (Shi et al., 2025), alongside advanced data selection methods like FisherSFT (Deb et al., 2025). However, these LLM-specific approaches typically rely on static heuristics that struggle to accurately track the model’s continuously evolving capabilities during training. OSW-FT seamlessly shifts the optimization budget toward the model’s true capability frontier, entirely eliminating the need for offline filtering or manual data sorting.

## 6 Conclusion

We introduced Online Self-Weighted Fine-Tuning (OSW-FT), a simple method that augments supervised fine-tuning with online, trajectory-level weighting for verifiable reasoning. By estimating the model’s current success rate with a small number of inference-only rollouts, OSW-FT adjusts how strongly each expert trajectory should update the model while keeping the optimization direction anchored to expert demonstrations. Empirically, OSW-FT consistently improves over SFT on small-to-medium models and offers a favorable performance–rollout trade-off when the available online rollout budget is limited.

## Limitations

OSW-FT currently assumes a reliable binary verifier and high-quality expert trajectories. Its successrate weight is designed for binary-verifiable reasoning and has not been validated for continuous rewards or open-ended generation. Although GPQA-Diamond broadens the evaluation beyond mathematics, it remains a binary-verifiable multiplechoice task and therefore does not establish applicability to non-binary settings.

Our experiments also focus on Qwen3 model family and primarily mathematical reasoning tasks. Extending OSW-FT to code generation and other reward formulations remains future work.

## References

AIME. 2025. AIME problems and solutions 2025. https://artofproblemsolving.com/wiki/ index.php/AIME\_Problems\_and\_Solutions.

AMC. 2023. AMC problems and solutions.

Yoshua Bengio, Jérôme Louradour, Ronan Collobert, and Jason Weston. 2009. Curriculum learning. In ICML.

Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. 2025. Minimax-m1: Scaling test-time compute efficiently with lightning attention. Arxiv.

Rohan Deb, Kiran Koshy Thekumparampil, Kousha Kalantari, Gaurush Hiranandani, Shoham Sabach, and Branislav Kveton. 2025. FisherSFT: Dataefficient supervised fine-tuning of language models using information gain. In ICML.

Chang Gao, Chujie Zheng, Xiong-Hui Chen, Kai Dang, Shixuan Liu, Bowen Yu, An Yang, Shuai Bai, Jingren Zhou, and Junyang Lin. 2025. Soft adaptive policy optimization. Arxiv.

Saeed Ghadimi and Guanghui Lan. 2013. Stochastic first-and zeroth-order methods for nonconvex stochastic programming. SIAM Journal on Optimization.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Arxiv.

Nan He, Weichen Xiong, Hanwen Liu, Yi Liao, Lei Ding, Kai Zhang, Guohua Tang, Xiao Han, and Yang Wei. 2024. Softdedup: an efficient data reweighting method for speeding up language model pre-training. In ACL.

Huggingface. 2025. Math-Verify. https://github. com/huggingface/Math-Verify.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Jia LI, Edward Beeching, Lewis Tunstall, Ben Lipkin, Roman Soletskyi, Shengyi Costa Huang, Kashif Rasul, Longhui Yu, Albert Jiang, Ziju Shen, Zihan Qin, Bin Dong, Li Zhou, Yann Fleureau, Guillaume Lample, and Stanislas Polu. 2024. Numinamath.

Mengqi Li, Lei Zhao, Anthony Man-Cho So, Ruoyu Sun, and Xiao Li. 2025. Online sft for llm reasoning: Surprising effectiveness of self-tuning without rewards. Arxiv.

Ming Li, Yong Zhang, Zhitao Li, Jiuhai Chen, Lichang Chen, Ning Cheng, Jianzong Wang, Tianyi Zhou, and Jing Xiao. 2024. From quantity to quality: Boosting llm performance with self-guided data selection for instruction tuning. In NAACL.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. In ICLR.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In ICLR.

Rui Ming, Haoyuan Wu, Shoubo Hu, Zhuolun He, and Bei Yu. 2025. One-token rollout: Guiding supervised fine-tuning of llms with policy gradient. Arxiv.

Sanmit Narvekar, Bei Peng, Matteo Leonetti, Jivko Sinapov, Matthew E Taylor, and Peter Stone. 2020. Curriculum learning for reinforcement learning domains: A framework and survey. JMLR.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. In NeurIPS.

Rémy Portelas, Cédric Colas, Lilian Weng, Katja Hofmann, and Pierre-Yves Oudeyer. 2020. Automatic curriculum learning for deep rl: A short survey. Arxiv.

Chongli Qin and Jost Tobias Springenberg. 2025. Supervised fine tuning on curated data is reinforcement learning (and can be improved). Arxiv.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. 2024. Gpqa: A graduate-level google-proof q&a benchmark. In COLM.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. Arxiv.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Arxiv.

Taiwei Shi, Yiyang Wu, Linxin Song, Tianyi Zhou, and Jieyu Zhao. 2025. Efficient reinforcement finetuning via adaptive curriculum learning. Arxiv.

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. 1999. Policy gradient methods for reinforcement learning with function approximation. In NeurIPS.

Xiaoyu Tian, Yunjie Ji, Haotian Wang, Shuaiting Chen, Sitong Zhao, Yiping Peng, Han Zhao, and Xiangang Li. 2025. Not all correct answers are equal: Why your distillation source matters. Arxiv.

Jiakang Wang, Runze Liu, Lei Lin, Wenping Hu, Xiu Li, Fuzheng Zhang, Guorui Zhou, and Kun Gai. 2025. Aspo: Asymmetric importance sampling policy optimization. Arxiv.

Yongliang Wu, Yizhou Zhou, Zhou Ziheng, Yingzhe Peng, Xinyu Ye, Xinting Hu, Wenbo Zhu, Lu Qi, Ming-Hsuan Yang, and Xu Yang. 2026. On the generalization of SFT: A reinforcement learning perspective with reward rectification. In ICLR.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Arxiv.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, YuYue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, and 17 others. 2025. DAPO: An open-source LLM reinforcement learning system at scale. In NeurIPS.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. 2022. Star: Bootstrapping reasoning with reasoning. In NeurIPS.

Wenhao Zhang, Yuexiang Xie, Yuchang Sun, Yanxi Chen, Guoyin Wang, Yaliang Li, Bolin Ding, and Jingren Zhou. 2026. On-policy RL meets off-policy experts: Harmonizing supervised fine-tuning and reinforcement learning via dynamic weighting. In ICLR.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. 2025. Group sequence policy optimization. Arxiv.

Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, et al. 2023. Lima: Less is more for alignment. In NeurIPS.

He Zhu, Junyou Su, Peng Lai, Ren Ma, Wenjia Zhang, Linyi Yang, and Guanhua Chen. 2026. Anchored supervised fine-tuning. In ICLR.

## A Ethics Statement

We acknowledge the use of Large Language Models (LLMs) during the preparation of this manuscript. Specifically, LLMs were utilized solely as auxiliary tools for minor writing assistance (such as grammar checking, typo correction, and phrasing refinement) and for debugging code implementations. We have reviewed all LLMassisted output and take full responsibility for the accuracy, originality, and integrity of the work presented in this paper.

## B Theoretical Proofs

Throughout, we use $p : = p _ { s } ( q ) = \mathbb { E } _ { o \sim \pi _ { \theta } } [ R ( o ) | q ]$ and $g _ { 0 } ( q ) : = \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( o ^ { * } \vert q )$ for a fixed query q satisfying Assumption 1.

Assumption 1 (Binary Verifiable Reward). For each query $q ,$ the reward is binary: $R ( o ) \in$ $\{ 0 , 1 \}$ , and the success probability $p _ { s } ( q ) : =$ $\mathbb { E } _ { o \sim \pi _ { \theta } } [ R ( o ) | q ] \in ( 0 , 1 )$ is well-defined.

## B.1 Finite Rollout Analysis (Proof of Theorem 1)

Proof. Let $\begin{array} { r } { \hat { p } ( q ) \ = \ \frac { 1 } { K } \sum _ { k = 1 } ^ { K } R ( \hat { o } _ { k } ) } \end{array}$ where $\hat { o } _ { k } \sim$ $\pi _ { \theta } ( \cdot | q )$ are i.i.d. rollouts. Since $\begin{array} { r l } { R ( \hat { o } _ { k } ) } & { { } \sim } \end{array}$ Bernoulli(p), we have $\mathbb { E } [ \boldsymbol { \hat { p } } ] = \boldsymbol { p }$ and $\mathrm { V a r } [ \hat { p } ] =$ $p ( 1 - p ) / K$

Unbiasedness. Since $g _ { 0 } ( q )$ is deterministic for a fixed query q:

$$
\begin{array} { r } { \mathbb { E } _ { \hat { p } } [ \hat { g } _ { K } ( \boldsymbol { q } ) ] = \mathbb { E } [ 1 - \hat { p } ( \boldsymbol { q } ) ] \cdot \boldsymbol { g } _ { 0 } ( \boldsymbol { q } ) = ( 1 - p ) \cdot \boldsymbol { g } _ { 0 } ( \boldsymbol { q } ) . } \end{array}\tag{12}
$$

Signal loss. The event $w ( q ) = 0$ occurs iff ${ \hat { p } } ( q ) =$ 1, i.e., all K rollouts are correct. By independence:

$$
P ( w ( q ) = 0 ) = P ( \hat { p } ( q ) = 1 ) = p ^ { K } .\tag{13}
$$

The following results provide additional quantitative characterization of the estimator under finite rollouts.

Remark 2 (Total second moment). Using the identity $\mathbb { E } [ X ^ { 2 } ] = ( \mathbb { E } [ X ] ) ^ { 2 } + \operatorname { V a r } [ X ]$ with $X = 1 - { \hat { p } } { \mathrm { : } }$

$$
\mathbb { E } _ { \hat { p } } [ \| \hat { g } _ { K } ( \boldsymbol { q } ) \| ^ { 2 } ] = \left[ ( 1 - p ) ^ { 2 } + \frac { p ( 1 - p ) } { K } \right] \| g _ { 0 } ( \boldsymbol { q } ) \| ^ { 2 } .\tag{14}
$$

This decomposes into a signal term $( 1 - p ) ^ { 2 } \| g _ { 0 } \| ^ { 2 }$ and an estimation noise term $\frac { p ( 1 - p ) } { K } \| g _ { 0 } \| ^ { 2 }$ that vanishes as $K  \infty .$ . This expression is used in the convergence analysis (Theorem 2).

Remark 3 (Estimation-induced MSE). The MSE due to weight estimation can be isolated as:

$$
\begin{array} { r } { \mathbb { E } _ { \hat { p } } \big [ \lVert \hat { g } _ { K } ( \boldsymbol { q } ) - ( 1 - p ) g _ { 0 } ( \boldsymbol { q } ) \rVert ^ { 2 } \big ] = } \\ { \underline { { p ( 1 - p ) } } \lVert g _ { 0 } ( \boldsymbol { q } ) \rVert ^ { 2 } , } \end{array}\tag{15}
$$

which follows from $\hat { g } _ { K } - ( 1 - p ) g _ { 0 } = ( p - \hat { p } ) g _ { 0 }$ and $\mathbb { E } [ ( p - \hat { p } ) ^ { 2 } ] = \mathrm { V a r } [ \hat { p } ] = p ( 1 - p ) / K$

## B.2 Convergence Analysis (Proof of Theorem 2)

Definition 1 (OSW-FT Surrogate Objective). At outer iteration t with parameters $\theta _ { t } ,$ , define:

$$
J ^ { ( t ) } ( \theta ) = \mathbb { E } _ { q \sim \mathcal { D } } \left[ \underbrace { \left( 1 - p _ { \theta _ { t } } ( q ) \right) } _ { f i x e d w e i g h t } \cdot \log \pi _ { \theta } ( o ^ { * } \mid q ) \right] ,\tag{16}
$$

where the weight is computed from $\pi _ { \theta _ { t } }$ and held fixed during inner-loop updates.

Remark 4 (Necessity of stop-gradient). Differentiating the naive objective $\mathbb { E } _ { q } [ ( 1 - p _ { \theta } ( q ) )$ log $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { o } ^ { * } | \boldsymbol { q } ) ]$ through the weight would introduce a $\nabla _ { \theta } p _ { \theta } ( q )$ term that the OSW-FT estimator $\left( 1 ~ - ~ { \hat { p } } \right) g _ { 0 }$ does not capture. The surrogate $J ^ { ( t ) }$ makes explicit that $\left( 1 - \hat { p } \right) g _ { 0 }$ is an unbiased gradient of $J ^ { ( t ) } ( \theta )$ at $\theta = \theta _ { t } ,$ , which is the quantity optimized in practice. This matches Algorithm 1, where rollouts and weights are computed once per batch before gradient updates.

Assumption 2 (Regularity Conditions). (i)

L-smoothness: For each fixed weight configuration, $J ^ { ( t ) } ( \theta )$ is L-smooth in θ.

(ii) Bounded second moment: $\mathbb { E } _ { q , \hat { p } } [ \rVert ( 1 \ - $ $\hat { p } ( q ) ) g _ { 0 } ( q ) \rVert ^ { 2 } ] \le M$ for all θ.

(iii) Weight stability: $| p _ { \theta _ { t + 1 } } ( q ) - p _ { \theta _ { t } } ( q ) | \leq \epsilon _ { p } f o r$ all q between consecutive outer iterations.

(iv) Bounded log-likelihood: $| \log \pi _ { \theta } ( o ^ { * } | q ) | \le B$ for all θ, q.

Proof of Theorem 2. The proof proceeds in four steps.

Step 1: Inner-loop convergence. Within each outer iteration t, the weights are fixed, and by the unbiasedness established in Theorem 1, the stochastic gradient $( 1 - \hat { p } ( q ) ) g _ { 0 } ( q )$ is an unbiased estimator of $\nabla _ { \boldsymbol { \theta } } J ^ { ( t ) } ( \boldsymbol { \theta } _ { t } )$ . Applying the standard SGD convergence result for L-smooth non-convex objectives (Ghadimi and Lan, 2013) over the S inner steps:

$$
\begin{array} { r l } & { \frac { 1 } { S } \displaystyle \sum _ { s = 1 } ^ { S } \mathbb { E } [ \| \nabla J ^ { ( t ) } ( \theta _ { t , s } ) \| ^ { 2 } ] } \\ & { \leq \frac { 2 \left( J ^ { ( t ) } ( \theta ^ { * } ) - J ^ { ( t ) } ( \theta _ { t , 0 } ) \right) } { S \eta } + L \eta \Sigma ^ { 2 } , } \end{array}\tag{17}
$$

where

$$
\begin{array} { l } { \displaystyle \Sigma ^ { 2 } : = \mathbb { E } _ { q \sim \mathcal { D } } \bigg [ ( 1 - p ( q ) ) ^ { 2 } } \\ { \displaystyle + \frac { p ( q ) ( 1 - p ( q ) ) } { K } \bigg ] \| g _ { 0 } ( q ) \| ^ { 2 } } \end{array}\tag{18}
$$

follows from the total second moment expression in Remark 2 and linearity of expectation over q.

Step 2: Cross-iteration mismatch. When switching from outer iteration t to $t + 1$ , the surrogate objective changes. The mismatch is bounded by:

$$
\begin{array} { r l } & { | J ^ { ( t + 1 ) } ( \theta ) - J ^ { ( t ) } ( \theta ) | } \\ & { = \left| \mathbb { E } _ { q } \big [ ( p _ { \theta _ { t } } ( q ) - p _ { \theta _ { t + 1 } } ( q ) ) \log \pi _ { \theta } ( o ^ { * } | q ) \big ] \right| } \\ & { \leq \mathbb { E } _ { q } \big [ | p _ { \theta _ { t } } ( q ) - p _ { \theta _ { t + 1 } } ( q ) | \cdot | \log \pi _ { \theta } ( o ^ { * } | q ) | \big ] } \\ & { \leq \epsilon _ { p } \cdot B , } \end{array}
$$

using Assumption 2(iii) and (iv).

Step 3: Accumulating the switching cost. Since there are $M = T / S$ outer-iteration transitions, the mismatch accumulates as

$$
\begin{array} { r } { \bigg | \sum _ { m = 1 } ^ { T / S } \left[ J ^ { ( m ) } ( \theta ) - J ^ { ( m + 1 ) } ( \theta ) \right] \bigg | \leq \frac { T } { S } \epsilon _ { p } B , } \end{array}\tag{19}
$$

which, after dividing by $\eta T = { \sqrt { T } }$ , contributes a term of order $( \epsilon _ { p } B / S ) \sqrt { T }$ to the averaged bound. This term is bounded by the single-transition mismatch $\epsilon _ { p } B$ whenever $T \le S ^ { 2 }$ , i.e. over training horizons commensurate with the chosen update interval S:

$$
\frac { \epsilon _ { p } B } { S } \sqrt { T } \leq \epsilon _ { p } B .\tag{20}
$$

Step 4: Telescoping. Summing (17) over all $T / S$ outer iterations and accounting for the mismatch (19) at each transition (with the switching cost bounded as in Step 3), with $\eta = 1 / \sqrt { T }$

$$
\begin{array} { r l } & { \frac { 1 } { T } \displaystyle \sum _ { t = 1 } ^ { T } \mathbb { E } \Big [ \| \nabla _ { \theta } J ^ { ( \lfloor t / S \rfloor ) } ( \theta _ { t } ) \| ^ { 2 } \Big ] } \\ & { \le \displaystyle \frac { 2 \Delta _ { 0 } + L \cdot \Sigma ^ { 2 } } { \sqrt { T } } + B \cdot \epsilon _ { p } . } \end{array}\tag{21}
$$

□

Corollary 1 $( \Sigma ^ { 2 } < \Sigma _ { \mathrm { S F T } } ^ { 2 } ) .$ . The SFT gradient second moment is $\Sigma _ { \mathrm { S F T } } ^ { 2 } = \mathbb { E } _ { q } [ \| g _ { 0 } ( q ) \| ^ { 2 } ]$ . We show that $\Sigma ^ { 2 } < \Sigma _ { \mathrm { S F T } } ^ { 2 }$ for any $K \geq 1$

Decompose the difference:

$$
\begin{array} { r l } & { \Sigma _ { \mathrm { S F T } } ^ { 2 } - \Sigma ^ { 2 } } \\ & { = \mathbb { E } _ { q } \big [ \| g _ { 0 } \| ^ { 2 } \big ] } \\ & { - \mathbb { E } _ { q } \bigg [ \bigg ( ( 1 - p ) ^ { 2 } + \frac { p ( 1 - p ) } { K } \bigg ) \| g _ { 0 } \| ^ { 2 } \bigg ] } \\ & { = \mathbb { E } _ { q } \bigg [ \bigg ( p ( 2 - p ) - \frac { p ( 1 - p ) } { K } \bigg ) \| g _ { 0 } \| ^ { 2 } \bigg ] \ . } \end{array}
$$

We verify the integrand is strictly positive for $p \in ( 0 , 1 )$ and $K \geq 1 .$

$$
\begin{array} { l l l } { \displaystyle p ( 2 - p ) - \frac { p ( 1 - p ) } { K } = p \bigg [ ( 2 - p ) - \frac { 1 - p } { K } \bigg ] } \\ { \geq p [ ( 2 - p ) - ( 1 - p ) ] = p > 0 , } \end{array}\tag{22}
$$

where the inequality uses $K \geq 1$ . Therefore, whenever $P ( p ( q ) > 0 ) > 0$ , the expectation in (22) is strictly positive, giving $\Sigma ^ { 2 } < \Sigma _ { \mathrm { S F T } } ^ { 2 }$

## C Unifying SFT and RL via General Policy Gradient

To reveal the structural connection between Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), we unify them under the General Policy Gradient framework. In RL, the gradient update for a model $\pi _ { \theta }$ takes the standard form:

$$
\nabla _ { \boldsymbol { \theta } } J = \mathbb { E } \left[ \underbrace { \left( R ( \tau ) - b ( \boldsymbol { q } ) \right) } _ { \mathrm { A d v a n t a g e } / \mathrm { W e i g h t } } \cdot \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( \tau \mid \boldsymbol { q } ) \right]\tag{23}
$$

We view SFT as maximizing an objective function $J _ { \mathrm { S F T } } \ = \ - { \mathcal L } _ { \mathrm { S F T } }$ . Under this lens, SFT restricts the trajectory entirely to the deterministic ground-truth oracle $( \tau = o ^ { * }$ , where the reward $R ( o ^ { * } ) \equiv 1 )$ and implicitly operates with a zero baseline $( b ( q ) \equiv 0 )$ . Consequently, the SFT gradient collapses into a static, uniform update:

$$
\nabla _ { \boldsymbol { \theta } } J _ { \mathrm { S F T } } \propto \mathbb { E } \left[ \underbrace { \left( 1 - 0 \right) } _ { \mathrm { S t a t i c ~ W e i g h t } \equiv 1 } \cdot \nabla _ { \boldsymbol { \theta } } \log \pi _ { \boldsymbol { \theta } } ( o ^ { * } \mid q ) \right]\tag{24}
$$

The Missing Baseline Problem. While RL methods (such as GRPO) actively employ dynamic baselines (e.g., the group mean) to reduce variance and scale updates based on the policy’s evolving capability, SFT’s implicit zero baseline treats all ground-truth trajectories as having identical, maximal advantage. As a consequence, SFT assigns the exact same gradient coefficient to trivial, alreadymastered samples $( \pi _ { \boldsymbol { \theta } } ( \boldsymbol { o } ^ { * } \mid \boldsymbol { q } ) \approx 1 )$ as it does to completely unsolved ones. This leads to inefficient gradient allocation and unnecessary variance injection during training. Notably, this inefficiency arises not from noisy supervision, but from a systematic mathematical mismatch between the static gradient estimator and the policy’s true competence.

By formally deriving the variance-reduced baseline under the constraint of deterministic groundtruth trajectories, we naturally arrive at the formulation of Online Self-Weighted Fine-Tuning (OSW-FT). As established in Section 2, assigning this optimal baseline to the model’s current success probability $( b ( q ) = p _ { s } ( q ) )$ perfectly restores the dynamic advantage missing from standard SFT, ensuring optimal gradient reallocation while preserving the optimization stability inherent to supervised learning.