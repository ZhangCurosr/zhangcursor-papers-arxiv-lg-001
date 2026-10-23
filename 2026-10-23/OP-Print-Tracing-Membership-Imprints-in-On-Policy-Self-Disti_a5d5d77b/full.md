# OP-Print: Tracing Membership Imprints in On-Policy Self-Distillation

Qi Li Xinchao Wang

National University of Singapore liqi@u.nus.edu xinchao@nus.edu.sg

## Abstract

On-policy distillation has been widely adopted for alleviating the distribution mismatch of offpolicy distillation. Self-distillation further removes the need for an external teacher by using the student’s own outputs, enhanced with extra context or demonstrations, as supervision. This teacher-free design makes on-policy self-distillation practical, yet its privacy risks remain largely unexplored. In particular, using two coupled views of the same training example may strengthen its memorization. Motivated by this, we present the first study of membership inference attacks against on-policy self-distillation. We investigate both gray-box and black-box threat models and propose a unified attack framework, named OP-Print. OP-Print builds on our finding that the supervision imprint in on-policy self-distillation is sparse and structured, concentrating on uncertaintysensitive correction sites rather than across the entire rollout. Exploiting this property, we construct attack signals tailored to the two access levels: in the gray-box setting, OP-Print directly aggregates the privileged teacher-student residual correction along the same rollout; in the black-box setting, it measures restart-based output-basin stability as an observable surrogate for the hidden correction process. Across diverse models and distillation schemes, all baseline attacks remain essentially at random guessing even under gray-box access, whereas OP-Print is broadly validated to be effective in both gray-box and black-box settings.

## 1 Introduction

On-policy distillation has recently become an important paradigm for improving model performance, as it alleviates the distribution mismatch of off-policy distillation (Guo et al., 2025; Song and Zheng, 2026; Ye et al., 2026b). Its self-distillation variants go one step further by removing the need for an external teacher: the distillation signal is instead derived from the student model itself, augmented with additional context or demonstrations that bootstrap its own capabilities (Zhao et al., 2026a; Shenfeld et al., 2026; Sang et al., 2026; He et al., 2026; Yin et al., 2026a). This teacherfree property makes on-policy self-distillation especially attractive in practice. In particular, the model is exposed to two coupled views of the same example: a standard version without privileged information and a privileged version augmented with guidance such as a reference solution. While this design improves reasoning performance, its privacy implications have remained largely unexamined. Intuitively, processing both views of the same sample may produce a stronger and more distinctive training imprint, potentially making that sample easier to memorize than under ordinary training.

Motivated by this concern, in this work, we investigate the data privacy risks of on-policy selfdistillation through the lens of membership inference attacks (Shi et al., 2023; Li et al., 2024; Song et al., 2019; Li et al., 2026b; Wang et al., 2025). Membership inference asks whether an adversary can determine if a given sample was used in training (Shokri et al., 2017a; Carlini et al., 2022a; Yeom et al., 2018). We explore this problem under a unified gray-box and black-box threat model (Hu et al., 2022; Wu and Cao, 2025; Bai et al., 2024). We found this problem fundamentally nontrivial: the supervision signal in on-policy self-distillation is not globally visible, nor is it well captured by coarse output-level heuristics (Zhang et al., 2024; Hu et al., 2025; Li et al., 2026b). Instead, it is sparse, structured, and tied to the model’s internal correction process. More importantly, our analysis shows that this signal is organized primarily by token difficulty rather than absolute decoding position. In other words, privileged correction concentrates on uncertain reasoning steps, while position alone provides only weak evidence. This makes gray-box inference feasible but already nontrivial, and makes black-box inference substantially harder, since the adversary cannot observe the underlying uncertainty or correction process and only has access to the final outputs.

![](images/310f19bd8a908e1a41a8b71bb712b952c520546a27e576f4103a881182d24a86.jpg)  
(a) Gray-box Trajectory.

![](images/144efe8dddbe9eda6779231c2d92e9b1a3f2897df9d06d37b8715c699122de10.jpg)  
(b) TPR@Low FPR.  
Figure 1: Gray-box MIA signals from rollout-level dynamics and attack behavior (AUC) of Gray-box score.

To address this problem, we propose a unified framework, named OP-Print, that targets the same hidden supervision imprint at two levels of adversary observability. Our key insight is that onpolicy self-distillation leaves a sparse but structured correction imprint concentrated on uncertaintysensitive reasoning tokens. In the gray-box setting, this signature can be measured directly by comparing student-conditioned and privileged teacherconditioned behaviors along the same rollout. In the black-box setting, the correction process itself is hidden, but it still leaves observable traces in the geometry ofthe output basin: when generation is restarted from partial prefixes, member and nonmember examples differ in how stably their continuations remain within a local solution basin. Based on this observation, OP-Print instantiates two complementary attacks. Its gray-box variant aggregates teacher-student residual discrepancy along self-rollouts into a direct membership score, while its black-box variant probes restart-based outputbasin stability as a natural surrogate for the inaccessible correction signal. Together, they offer a unified view of privacy leakage: one captures the hidden correction process directly, while the other reveals its behavioral effects from outputs alone.

A gray-box example is shown in Figure 1. The gap between members and non-members is observable but with magnitude vary across decoding positions. OP-Print makes use of such signal and outperforms the baselines. Across diverse models and distillation schemes, we found that all baseline attacks, consistent with observations from prior works (Meeus et al., 2025a; Das et al., 2025; Duan et al., 2024), remain essentially at random guessing even under gray-box access, whereas OP-Print is broadly effective in both gray-box and black-box settings. These results reveal an overlooked privacy risk of on-policy self-distillation and show that its leakage surface differs from conventional distillation. Our contributions are as follows:

• We uncover the location of the membership signal in on-policy self-distillation: rather than being globally visible, it resides in a sparse and structured correction imprint concentrated on uncertainty-sensitive tokens.

• Building on this observation, we propose OP-Print, a unified attack framework that targets this hidden signal under both threat models: in the gray-box setting, it exploits privileged residual discrepancy directly, while in the black-box setting, it captures its restart-based output-basin traces, thereby enabling reliable membership inference.

• Extensive experiments across diverse LLMs consistently demonstrate the superiority of OP-Print. For example, on Qwen3-8B (Yang et al., 2025) under the black-box setting, OP-Print improves AUC by 7.7% and TPR@5% FPR by 26.5% over the strongest baseline.

## 2 Related Work

On-Policy Self-Distillation. On-policy distillation (OPD) trains the student on trajectories sampled from its own evolving policy, reducing the training-inference mismatch of off-policy distillation (Song and Zheng, 2026; Zhao et al., 2026a). Existing approaches include logit matching (Agarwal et al., 2024a; Ko et al., 2024; Jung et al., 2025; Wu et al., 2025b), outcome supervision (Ye et al., 2025; Jiang et al., 2023; Xiong et al., 2026; Singh et al., 2025; Mendes et al., 2026), and selfdistillation (Zhao et al., 2026a; Penaloza et al., 2026; Ye et al., 2026b,a; Dat et al., 2026). In selfdistillation, the same model acts as both teacher and student under asymmetric conditioning, eliminating the need for an external teacher. Early methods such as SPIN (Chen et al., 2024) rely on reference data, whereas recent approaches inject privileged information into the teacher context. A representative example is OPSD (Zhao et al., 2026a), where the teacher receives additional supervision while the student sees only the original input, and training minimizes token-level divergence on the student’s own rollouts. Recent extensions include GATES (Stein et al., 2026), SDPO (Hübotter et al., 2026), OPSDC (Sang et al., 2026), and SDFT (Shenfeld et al., 2026).

![](images/318b9dc4006229d1110a05bb122f584309f1ab6f9a3b2f80edef951d83f7cbbe.jpg)  
(a) Mean divergence across relative token positions.

![](images/bf12c55fd7621628f25e4a4b8becb066be410229c6af0a5502e95c4c766c07a0.jpg)

![](images/95d6909233b67b104f3eddc8847c71650156b9b902dfb469a6008648c7970fb8.jpg)  
(b) Mean divergence across student-(c) Reliability map of token-level teacheruncertainty quantiles. student divergence.  
Figure 2: (a). Divergence varies only mildly across token positions, indicating that position alone is not the main source of the signal. (b). Divergence increases with student uncertainty, showing that correction concentrates on difficult reasoning tokens. (c). The reliability map reveals localized rather than uniform divergence, confirming that the supervision imprint is sparse and structured.

Membership Inference Attack. Membership Inference Attacks (MIAs) (Shokri et al., 2017a; Carlini et al., 2022a; Li et al., 2026a; Yin et al., 2026b) determine whether a sample was used to train a target model, posing risks to sensitive training data. Existing MIAs mainly fall into metric-based and shadow model-based methods (Carlini et al., 2022a; Li et al., 2024). Metric-based attacks infer membership from statistics derived from model outputs (Yeom et al., 2018; Salem et al., 2018; Song and Mittal, 2021; Li et al., 2024), whereas shadow model-based attacks train auxiliary models to mimic the target (Shokri et al., 2017a; Zarifzadeh et al., 2023). Since the latter is computationally expensive and often impractical for LLMs (Li et al., 2024, 2026b), this work focuses exclusively on metric-based methods.

MIAs were first studied for classification models (Shokri et al., 2017a) and later extended to generative (Chen et al., 2020; Hayes et al., 2017) and embedding models (Mahloujifar et al., 2021; Song and Raghunathan, 2020). Recent work has further investigated MIAs against LLMs (Shi et al., 2023; Li et al., 2024), covering both pretraining (Shi et al., 2023; Zhang et al., 2024; Changpinyo et al., 2021) and post-training data (Li et al., 2024; Wu and Cao, 2025; Hu et al., 2025). The on-policy distillation process is commonly conducted during the posttraining process. This highlights the unique importance of this problem: The capability of LLMs to effectively interact with humans fundamentally depends on the post-training stage, as the strength of this capability is directly tied to the quality of the post-training dataset. Furthermore, developers often construct their own task-specific datasets for this stage (Song and Zheng, 2026; Zhao et al., 2026a), which introduces additional privacy risks. Motivated by these factors, we focus on the posttraining stage MIA of LLMs.

## 3 Method

Challenges. Figure 2 clarifies both the difficulty and the feasibility of MIA on on-policy selfdistillation. The key observation is that the supervision imprint is not organized primarily by token position, but by token difficulty. As shown in Figure 2a, teacher-student divergence varies only moderately across relative decoding positions, so position alone provides at most a weak signal. In contrast, Figure 2b shows a much sharper dependence on student uncertainty: the divergence rises substantially in higher-uncertainty regions, indicating that privileged correction is concentrated on difficult reasoning tokens rather than uniformly distributed throughout the rollout. Figure 2c further shows that this signal occupies only a restricted region of the reliability map, confirming that the resulting imprint is sparse and structured instead of globally exposed. This structure directly explains the gap between gray-box and black-box attacks. In the gray-box setting, adversaries can access the teacher-conditioned residual correction signal along the same rollout, and directly aggregates this privileged discrepancy. In the black-box setting, however, the most informative variable is fundamentally hidden, and the adversary only observes final output tokens. Since position is only weakly informative, this makes black-box MIA substantially harder.

Problem Setup. Let $x = ( q , r )$ denote a candidate training example, where q is the problem and r is its reference solution. Given a target checkpoint p<sub>θ</sub>, our goal is to infer whether x belongs to the member set used during self-distillation. We write $p _ { s } ( q )$ for the standard student prompt constructed from q alone, and $p _ { t } ( q , r )$ for the privileged prompt that additionally conditions on the reference solution r. For a rollout $y = ( y _ { 1 } , \dots , y _ { T } ) \sim p _ { \theta } ( \cdot \mid p _ { s } ( q ) )$ , let $y _ { < t }$ denote its prefix before token t.

![](images/dc431ea49694633a9710189ebcdba8c37ded8606df1b8708f989f7eb177ea90d.jpg)  
Figure 3: An illustration of the black-box attack process of OP-Print. Given multiple initial rollouts, OP-Print truncates each rollout at different prefix ratios, restarts generation from the partial prefix, and scores membership by measuring the similarity between restarted outputs. Members tend to produce more stable and similar restart trajectories, while non-members exhibit lower restart consistency.

A Unified View: Hidden Correction Imprint. The analysis in Figure 2 suggests that on-policy self-distillation leaves a structured correction imprint: membership-relevant signal is sparse, concentrated on uncertainty-sensitive reasoning tokens, and only weakly aligned with absolute decoding position. This motivates a unified view of both attacks. We posit a latent quantity measuring how much privileged correction the model would still require on a rollout:

$$
\begin{array} { r } { \Delta _ { t } ( x , y ) = \mathrm { K L } ( p _ { \theta } ( \cdot \mid p _ { t } ( q , r ) , y _ { < t } ) } \\ { \parallel p _ { \theta } ( \cdot \mid p _ { s } ( q ) , y _ { < t } ) ) . } \end{array}\tag{1}
$$

Intuitively, $\Delta _ { t } ( x , y )$ is large when the studentconditioned model still disagrees substantially with its privileged counterpart at token t, and small when the correction has already been internalized. Members should therefore exhibit smaller privileged correction on average, because self-distillation has already adapted the model on those examples. Our gray-box attack estimates this quantity directly, while our black-box attack targets its observable consequence in the output basin.

Gray-box Attack: Privileged Residual Aggregation. In the gray-box setting, we can evaluate the target model under both $p _ { s } ( q )$ and $p _ { t } ( q , r )$ along the same rollout. Given K sampled rollouts $\{ y ^ { ( k ) } \} _ { k = 1 } ^ { K }$ from $p _ { \theta } ( \cdot \mid p _ { s } ( q ) )$ , we define the gray-

```latex
Algorithm 1: OP-Print (Black-box)
Require: Target model p , query q, prefix ratios P, number
of full generations M, number of restarts R
Ensure: Membership score $S _ { \mathrm { b l a c k } } ( x )$
1: Sample M generations $\{ y ^ { ( m ) } \} _ { m = 1 } ^ { \dot { M } ^ { - } } \sim p _ { \theta } ( \cdot \mid p _ { s } ( q ) )$
2: for m = 1, . . . , M do
3: for each $\rho \in \mathcal P$ do
4: Truncate $y ^ { ( m ) }$ into prefix $\pi _ { \rho } ( \boldsymbol { y } ^ { ( m ) } )$
5: Restart decoding R times from $\pi _ { \rho } \big ( \boldsymbol { y } ^ { ( m ) } \big )$
6: Obtain restart set $\mathcal { R } _ { m , \rho } ( x )$ , where
$\mathcal { R } _ { m , \rho } ( x ) = \{ \tilde { y } ^ { ( m , \rho , 1 ) } , \dots , \tilde { y } ^ { ( m , \rho , R ) } \}$
7: Compute restart stability
$\operatorname { \mathrm { m p u v e ~ r c s t a t ~ s t a t ~ o u n t ~ s t a u p n u y } } _ { \mathrm { S t a b } _ { \mathrm { t e x t } } \left( \mathcal { R } _ { m , \rho } ( x ) \right) } = \frac { 2 } { R ( R - 1 ) } \sum _ { i < j }$
sim $\left( \tilde { y } ^ { ( m , \rho , i ) } , \tilde { y } ^ { ( m , \rho , j ) } \right)$
8: end for
9: end for
10: Aggregate over all full generations and prefix ratios:
$S _ { \mathrm { b l a c k } } ( x ) = { \frac { 1 } { M | \mathcal { P } | } } \sum _ { m = 1 } ^ { M } \sum _ { \rho \in \mathcal { P } } \operatorname { S t a b } _ { \mathrm { t e x t } } ( \mathcal { R } _ { m , \rho } ( x ) )$
11: return $S _ { \mathrm { b l a c k } } ( x )$
```

box score as

$$
S _ { \mathrm { g r a y } } ( x ) = - \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \frac { 1 } { T _ { k } } \sum _ { t = 1 } ^ { T _ { k } } \Delta _ { t } \Big ( x , y ^ { ( k ) } \Big ) .\tag{2}
$$

Here, a larger score means a smaller average privileged gap, hence stronger evidence that the example has already been absorbed by the self-distilled model. In implementation, we compute the divergence in fixed-size chunks (e.g., 64 tokens) for efficiency (ablations can be found in Section 4.2). In our experiments, we also introduce a grey-box scoring variant to assess the utility of the maximumdiscrepancy position. This variant can be written as $\begin{array} { r } { S _ { \mathrm { g r a y } } ^ { \mathrm { m a x } } \dot { ( \boldsymbol { x } ) } = - \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \operatorname* { m a x } _ { 1 \leq t \leq T _ { k } } \Delta _ { t } \big ( \boldsymbol { x } , \boldsymbol { y } ^ { ( k ) } \big ) } \end{array}$

Table 1: Gray-box and black-box results with the JSD objective. Colors indicate the top-2 and worst-2 results in the gray-box setting, and the top-1 and worst-1 results in the black-box setting.
<table><tr><td rowspan="2">Method</td><td colspan="3">Qwen3-1.7B</td><td colspan="3">Qwen3-4B</td><td colspan="3">Qwen3-8B</td></tr><tr><td>AUC</td><td>Acc.</td><td>TPR@5%FPR</td><td>AUC</td><td>Acc.</td><td>TPR@5%FPR</td><td>AUC</td><td>Acc.</td><td>TPR@5%FPR</td></tr><tr><td>Gray-box setting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Min-K% Prob</td><td>0.512</td><td>0.518</td><td>0.052</td><td>0.497</td><td>0.505</td><td>0.046</td><td>0.529</td><td>0.511</td><td>0.055</td></tr><tr><td>ModRényi (α = 0.5)</td><td>0.484</td><td>0.506</td><td>0.044</td><td>0.508</td><td>0.508</td><td>0.051</td><td>0.513</td><td>0.514</td><td>0.056</td></tr><tr><td>ModRényi (α = 2)</td><td>0.516</td><td>0.514</td><td>0.057</td><td>0.507</td><td>0.512</td><td>0.047</td><td>0.506</td><td>0.502</td><td>0.050</td></tr><tr><td>Rényi (α = 0.5)</td><td>0.503</td><td>0.509</td><td>0.049</td><td>0.517</td><td>0.511</td><td>0.054</td><td>0.495</td><td>0.503</td><td>0.045</td></tr><tr><td>Rényi (α = 1)</td><td>0.508</td><td>0.501</td><td>0.046</td><td>0.512</td><td>0.507</td><td>0.050</td><td>0.507</td><td>0.513</td><td>0.056</td></tr><tr><td>Rényi (α = 2)</td><td>0.517</td><td>0.518</td><td>0.055</td><td>0.506</td><td>0.508</td><td>0.044</td><td>0.491</td><td>0.503</td><td>0.052</td></tr><tr><td>Rényi (α = ∞)</td><td>0.489</td><td>0.502</td><td>0.048</td><td>0.514</td><td>0.511</td><td>0.057</td><td>0.511</td><td>0.509</td><td>0.046</td></tr><tr><td>OP-Print-G</td><td>0.579</td><td>0.586</td><td>0.108</td><td>0.580</td><td>0.583</td><td>0.095</td><td>0.556</td><td>0.543</td><td>0.088</td></tr><tr><td>OP-Print-G-Max</td><td>0.594</td><td>0.583</td><td>0.110</td><td>0.611</td><td>0.604</td><td>0.127</td><td>0.587</td><td>0.580</td><td>0.102</td></tr><tr><td>Black-box setting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>S-Len</td><td>0.505</td><td>0.516</td><td>0.068</td><td>0.419</td><td>0.507</td><td>0.060</td><td>0.505</td><td>0.516</td><td>0.068</td></tr><tr><td>RC-Rate</td><td>0.507</td><td>0.517</td><td>0.034</td><td>0.544</td><td>0.566</td><td>0.104</td><td>0.506</td><td>0.508</td><td>0.046</td></tr><tr><td>R-Maj</td><td>0.516</td><td>0.523</td><td>0.000</td><td>0.450</td><td>0.508</td><td>0.000</td><td>0.493</td><td>0.523</td><td>0.000</td></tr><tr><td>Avg. R-Len</td><td>0.495</td><td>0.508</td><td>0.000</td><td>0.458</td><td>0.508</td><td>0.000</td><td>0.489</td><td>0.506</td><td>0.000</td></tr><tr><td>R-Std</td><td>0.536</td><td>0.528</td><td>0.066</td><td>0.542</td><td>0.568</td><td>0.084</td><td>0.511</td><td>0.515</td><td>0.046</td></tr><tr><td>F-Rate</td><td>0.508</td><td>0.513</td><td>0.048</td><td>0.516</td><td>0.543</td><td>0.098</td><td>0.517</td><td>0.521</td><td>0.052</td></tr><tr><td>OP-Print-B</td><td>0.554</td><td>0.552</td><td>0.086</td><td>0.575</td><td>0.573</td><td>0.089</td><td>0.557</td><td>0.546</td><td>0.086</td></tr></table>

Black-box Attack: Restart-based Basin Stability. In the black-box setting, the informative variable in Figure 2, i.e., the teacher-conditioned residual correction, is hidden. The adversary sees only generated text, so a direct estimate of Equation 1 is unavailable. Our key design choice is therefore to attack the behavioral trace left by hidden correction, rather than the hidden correction itself.

As illustrated in Figure 3, for each input q, we first sample M full generations $\{ y ^ { ( m ) } \} _ { m = 1 } ^ { \bar { M } }$ from $p _ { \theta } ( \cdot \mid p _ { s } ( q ) )$ . For each generation $y ^ { ( m ) }$ and each prefix ratio $\rho \in \mathcal P$ , we truncate it to a partial prefix $\pi _ { \rho } ( y ^ { ( m ) } )$ and restart decoding from that prefix multiple times, obtaining a restart set

$$
\mathcal { R } _ { m , \rho } ( x ) = \left\{ \tilde { y } ^ { ( m , \rho , 1 ) } , \dots , \tilde { y } ^ { ( m , \rho , R ) } \right\} .\tag{3}
$$

We then measure how tightly these restarted continuations collapse in text space:

$$
\begin{array} { r } { \mathrm { S t a b } _ { \mathrm { t e x t } } ( \mathcal { R } _ { m , \rho } ( x ) ) = \displaystyle \frac { 2 } { R ( R - 1 ) } \sum _ { i < j } } \\ { \displaystyle \operatorname * { s i m } \Big ( \tilde { y } ^ { ( m , \rho , i ) } , \tilde { y } ^ { ( m , \rho , j ) } \Big ) } \end{array}\tag{4}
$$

where sim(·, ·) is a normalized string-level similarity. Our final black-box score is the average restarttext stability across sampled full generations and

prefix ratios:

$$
S _ { \mathrm { b l a c k } } ( x ) = \frac { 1 } { M | \mathcal { P } | } \sum _ { m = 1 } ^ { M } \sum _ { \rho \in \mathcal { P } } \mathrm { S t a b } _ { \mathrm { t e x t } } ( \mathcal { R } _ { m , \rho } ( x ) ) .\tag{5}
$$

A higher $S _ { \mathrm { b l a c k } } ( x )$ indicates a narrower and more stable output basin under prefix restarts, which we interpret as evidence that the hidden correction has already been more strongly internalized, and hence that x is more likely to be a member. The pseudo code is provided in Algorithm 1.

## 4 Experiments

Distillation Variants. We evaluate our attacks on two representative on-policy self-distillation variants. Both follow the same high-level training paradigm (Zhao et al., 2026a; Song and Zheng, 2026): the student first generates an on-policy completion from the original problem, and the resulting continuation is then paired with two conditioning contexts with teacher-side prompt contains additional privileged information derived from the reference solution. In the first variant, we follow the practice of previous works (Zhao et al., 2026a; Shenfeld et al., 2026) to use a generalized Jensen-Shannon divergence (JSD) objective that aligns the teacher and student distributions over the student-generated continuation. For the second variant, we follow a tinker-style (Lu and Lab,

![](images/f7fa5269619a4f9b8c2488893f3e8f960d3d542dc98fda9d9664040813aeef5d.jpg)  
(a) Gray-box member case.

![](images/9eb628b29a5a1198207fb3c7163ea18051a94d3b720099cf6d862cf2bdd4c316.jpg)

![](images/e891d3ebb2989440334d557c59032c3bad63e2ecaeef36da089667134c4d3789.jpg)

(b) Gray-box non-member case.  
![](images/58ac5b05ed06e9ef522209775864946b371b6ca754a821de3a3651fe3b1b0d74.jpg)  
(c) Black-box member case.

![](images/9f591208a63d9f75360133bb797abd173f40e5b4871f8ef80b2d419235193e7f.jpg)  
(d) Black-box non-member case.

![](images/6e93ef5be3751c326f3a38d8aa2f5341716479d80887d3758233c40143d7afaf.jpg)  
Figure 4: Representative gray-box and black-box OP-Print signals. (a)(b) In the gray-box setting, members show lower chunk-level KL gaps and higher scores than non-members. (c)(d) In the black-box setting, members exhibit more stable outputs under prefix restarts, whereas non-members are less stable.

2025) formulation. It preserves the same privileged teacher / standard student setup, but replaces distribution matching with a sampled-token policyimprovement objective. Concretely, the teacherstudent log-probability difference on the generated continuation is used as an advantage-like signal, and optimization increases the student probability on tokens preferred by the privileged teacher. This yields a more policy-oriented realization of on-policy self-distillation. These two variants provide a useful contrast between distribution-level and sampled-token-level self-distillation under the same on-policy training principle. All training hyperparameters are deferred to Appendix A.

Datasets and Models. A long-standing and highly debated issue in MIA against LLMs is data construction (Meeus et al., 2025a; Das et al., 2025; Duan et al., 2024). In prior work, the attack results are often rendered unreliable by severe distribution mismatch between the member and non-member sets, leading to overclaimed performance (Meeus et al., 2025a). On the other side, prior work has found that training a model too much on-policy selfdistillation can suffer from mode collapse (Song and Zheng, 2026; Zhao et al., 2026a). This also motivates controlling the data scale in evaluation. To avoid the above problems and to ensure a correct and fair evaluation of attack performance, we carefully select and construct the dataset and train the target models under a protocol that strictly follows the requirements of membership inference evaluation. For the dataset, e use the mathematical reasoning subset of OpenThoughts (Guha et al., 2025), sampling 1k problem-solution pairs with chain-ofthought reasoning, splitting them evenly into 500 members and 500 non-members. For the target models, we fine-tune three Qwen3 variants (Yang et al., 2025) (1.7B, 4B, 8B) on the sampled member set under the distillation variants described above. Training details are in Appenfix A.

Baselines. We consider both the general gray-box baselines and on-policy self-distillation specified black-box baselines. For gray-box baselines, we include Min-K% (Zhang et al., 2024) and MaxRényi-K% (Li et al., 2024). We report the best results among different variants of these methods. For black-box baselines, we construct rollout-derived metrics including solution length (S-Len), rollout correct-rate (RC-Rate), rollout majority-answer (R-Maj), average rollout length (Avg. R-Len), rolloutlength standard deviation (R-Std), and formattedanswer rate (F-Rate). These metrics capture output length, correctness, answer consistency, length stability, and formatting reliability, respectively.

Evaluation metric. As a binary classification problem, the performance can be evaluated with the AUC score (Carlini et al., 2021). We define the members as "positive" and the non-members as "negative". We also report True Positive Rate (TPR) at low False Positive Rate (FPR) (Carlini et al., 2022a), which is an important metric in MIAs and measures detection rate at a meaningful threshold. We set the threshold as 5% and evaluate all the methods under TPR@5% FPR. We also report the best classification accuracy achievable by sweeping over all possible thresholds on the attack scores.

![](images/067190194840594fb3f0becfa7d6f7a2691bfa1cefdcf25fbd745a2c205f2e44.jpg)

![](images/2c0ed28e056f8306a4c740f31d1bf053d08deb826cd71a4b7953ae09ff18b233.jpg)  
(a) Gray-box sensitivity to the # rollouts and the chunk size. The default setting is outlined with a dashed box.

![](images/b4f5acc822a636f87d38b8391368feb56501e9c2dbabd6cf52b248e85392adf0.jpg)  
(b) Black-box sensitivity to the number of prefix ratios.

![](images/66ae673f81a27edea3bfc92ce92166393f7d14836171b2883c1451a6fbdf8859.jpg)  
(c) Trade-off of restart cost and performance.  
Figure 5: Ablation studies of OP-Print on Qwen3-4B. (a) Gray-box performance across rollout numbers and chunk sizes. (b) Black-box performance with varying numbers of prefix ratios. (c) Restart cost-performance trade-off, where bubble size denotes TPR@5%FPR.

Specifically, this accuracy is computed as the maximum value of $\begin{array} { r } { 1 - \frac { \mathrm { F P R } \dot { + } ( 1 - \mathrm { T P R } ) } { 2 } } \end{array}$ across the ROC curve, representing the optimal classification accuracy between members and non-members. We use # frames to denote the number of frames.

## 4.1 Main Results

Results and Analysis. Table 1 reports membership inference results under both gray-box and blackbox settings. For gray-box attacks, we set the chunk size to 64 and $K = 3 2 ;$ for black-box attacks, we use $| \mathcal { P } | = 1 0$ with ρ ∼ Uniform(0, 1) (see Section 4.2 for ablations). Distillation uses the dense token-level JSD objective. Overall, OP-Print consistently outperforms existing baselines. In the gray-box setting, OP-Print-G-Max achieves the highest AUC across all model scales, improving the strongest non-OP-Print baseline from 0.517 to 0.594 on Qwen3-1.7B, 0.517 to 0.611 on Qwen3- 4B, and 0.529 to 0.587 on Qwen3-8B. It also achieves TPR@5%FPR scores of 0.110, 0.127, and 0.102, respectively, demonstrating strong performance in the low-FPR regime.

Standard likelihood- and entropy-based MIA baselines, including Min-K% Prob and Rényi scores, remain near random guessing, with AUCs around 0.49–0.52. In contrast, OP-Print-G and OP-Print-G-Max perform substantially better, indicating that on-policy self-distillation leaves a structured correction imprint that is better captured by teacher–student discrepancy than by raw likelihood statistics. The consistent advantage of OP-Print-G-Max further suggests that membership information is concentrated at the most discrepant positions.

In the black-box setting, OP-Print-B achieves the best overall AUC and accuracy across model scales. For example, on Qwen3-4B, it reaches 0.575 AUC and 0.573 accuracy, outperforming all black-box baselines on both metrics. Although RC-Rate attains the highest TPR@5%FPR on Qwen3-4B, its gains are less consistent across scales. These results suggest that black-box membership evidence is better captured by aggregating behavioral consequences of the hidden correction imprint than by any single surface heuristic.

Across scales, the results also show that membership leakage under on-policy self-distillation does not monotonically vanish as the backbone becomes larger. Both gray-box and black-box OP-Print variants remain effective on Qwen3-8B, demonstrating that the proposed attack captures a persistent training-induced signal rather than a scale-specific artifact. These findings validate our central claim: on-policy self-distillation introduces a distinct membership leakage channel, and exploiting the correction imprint yields stronger and more reliable inference than conventional MIA scores or simple rollout statistics.

Case-level Interpretation. Figure 4 illustrates this correspondence qualitatively. In the gray-box cases (K = 32), the member example exhibits a lower chunk-level KL gap throughout the rollout and therefore a higher $S _ { \mathrm { g r a y } }$ , while the non-member case shows the opposite pattern. In the black-box cases $( | \mathcal { P } | = 2 0 , \rho \sim \mathrm { U n i f o r m } ( 0 , 1 ) )$ , the member example produces a visibly more stable restart basin, whereas the non-member case is more dispersive under the same restart procedure. These examples are consistent with our design principle: gray-box MIA accesses privileged correction directly, while black-box MIA recovers its observable traces through restart-based output stability.

Influence of Distillation Variants. Table 2 shows that OP-Print remains effective under the tinkerstyle objective, indicating that the correctionimprint signal generalizes beyond a specific distillation formulation. In the gray-box setting, OP-Print-G and OP-Print-G-Max consistently outperform likelihood-based baselines. OP-Print-G-Max achieves the best AUC on Qwen3-1.7B and Qwen3- 8B, while OP-Print-G remains competitive on several metrics, further supporting that membership information is concentrated at highly informative discrepancy positions. In the black-box setting, OP-Print-B remains the strongest overall method in AUC and accuracy across model scales. While simple rollout heuristics such as R-Std and F-Rate occasionally achieve higher TPR@5%FPR on specific models, their performance is less consistent. In contrast, OP-Print-B delivers more stable gains, showing that the behavioral footprint of the hidden correction imprint remains observable even without privileged gray-box signals.

Table 2: Gray-box and black-box results with tinker-style objective. Colors indicate the top-2 and worst-2 results in the gray-box setting, and the top-1 and worst-1 results in the black-box setting.
<table><tr><td rowspan="2">Method</td><td colspan="3">Qwen3-1.7B</td><td colspan="3">Qwen3-4B</td><td colspan="3">Qwen3-8B</td></tr><tr><td>AUC</td><td>Acc.</td><td>TPR@5%FPR</td><td>AUC</td><td>Acc.</td><td>TPR@5%FPR</td><td>AUC</td><td>Acc.</td><td>TPR@5%FPR</td></tr><tr><td>Gray-box setting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Min-K% Prob</td><td>0.509</td><td>0.510</td><td>0.051</td><td>0.491</td><td>0.502</td><td>0.045</td><td>0.522</td><td>0.503</td><td>0.054</td></tr><tr><td>ModRényi (α = 0.5)</td><td>0.478</td><td>0.504</td><td>0.042</td><td>0.503</td><td>0.500</td><td>0.052</td><td>0.506</td><td>0.507</td><td>0.055</td></tr><tr><td>ModRényi (α = 2)</td><td>0.507</td><td>0.505</td><td>0.055</td><td>0.496</td><td>0.508</td><td>0.046</td><td>0.500</td><td>0.501</td><td>0.049</td></tr><tr><td>Rényi (α = 0.5)</td><td>0.495</td><td>0.502</td><td>0.050</td><td>0.510</td><td>0.504</td><td>0.056</td><td>0.484</td><td>0.500</td><td>0.043</td></tr><tr><td>Rényi (α = 1)</td><td>0.500</td><td>0.500</td><td>0.045</td><td>0.504</td><td>0.503</td><td>0.048</td><td>0.502</td><td>0.505</td><td>0.057</td></tr><tr><td>Rényi (α = 2)</td><td>0.505</td><td>0.512</td><td>0.053</td><td>0.498</td><td>0.501</td><td>0.042</td><td>0.481</td><td>0.500</td><td>0.051</td></tr><tr><td>Rényi (α = ∞)</td><td>0.481</td><td>0.500</td><td>0.047</td><td>0.506</td><td>0.509</td><td>0.054</td><td>0.506</td><td>0.504</td><td>0.048</td></tr><tr><td>OP-Print-G</td><td>0.571</td><td>0.579</td><td>0.076</td><td>0.563</td><td>0.575</td><td>0.094</td><td>0.548</td><td>0.553</td><td>0.071</td></tr><tr><td>OP-Print-G-Max</td><td>0.583</td><td>0.578</td><td>0.109</td><td>0.571</td><td>0.569</td><td>0.083</td><td>0.579</td><td>0.576</td><td>0.098</td></tr><tr><td>Black-box setting</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>S-Len</td><td>0.497</td><td>0.506</td><td>0.067</td><td>0.481</td><td>0.500</td><td>0.058</td><td>0.497</td><td>0.507</td><td>0.066</td></tr><tr><td>RC-Rate</td><td>0.500</td><td>0.510</td><td>0.033</td><td>0.533</td><td>0.534</td><td>0.082</td><td>0.500</td><td>0.505</td><td>0.044</td></tr><tr><td>R-Maj</td><td>0.507</td><td>0.514</td><td>0.000</td><td>0.489</td><td>0.500</td><td>0.000</td><td>0.512</td><td>0.504</td><td>0.000</td></tr><tr><td>Avg. R-Len</td><td>0.497</td><td>0.501</td><td>0.000</td><td>0.487</td><td>0.500</td><td>0.000</td><td>0.480</td><td>0.502</td><td>0.000</td></tr><tr><td>R-Std</td><td>0.527</td><td>0.520</td><td>0.064</td><td>0.539</td><td>0.540</td><td>0.093</td><td>0.505</td><td>0.507</td><td>0.044</td></tr><tr><td>F-Rate</td><td>0.503</td><td>0.502</td><td>0.046</td><td>0.508</td><td>0.534</td><td>0.096</td><td>0.530</td><td>0.532</td><td>0.072</td></tr><tr><td>OP-Print-B</td><td>0.548</td><td>0.550</td><td>0.085</td><td>0.567</td><td>0.562</td><td>0.091</td><td>0.553</td><td>0.539</td><td>0.064</td></tr></table>

## 4.2 Ablation Study

We further analyze the sensitivity of OP-Print to key hyperparameters, including rollout number, chunk size, prefix ratios, and restarts. These factors affect probing granularity, stability, and computational cost. Results are shown in Figure 5.

Effect of rollout number and chunk size. Figure 5a shows that performance improves as the rollout number increases to 32, after which gains become marginal. A chunk size of 64 achieves the best or near-best results, balancing local noise and over-smoothing. OP-Print-G-Max consistently outperforms OP-Print-G, indicating that membership signals are concentrated at highly informative discrepancy positions.

Effect of prefix-ratio selection. Figure 5b shows that using more prefix ratios improves AUC and TPR@5%FPR by probing the generation trajectory more densely. However, the gains are gradual, and the default setting already performs strongly. This suggests that the black-box signal is not limited to a narrow truncation point and remains robust under practical prefix selection.

Effect of restart count. Figure 5c shows that increasing the number of restarts improves the performance, but with diminishing returns and substantially higher cost. This suggests that additional restarts mainly reduce estimation variance rather than reveal new information. The default setting provides a favorable cost-performance trade-off.

## 5 Conclusion

We study membership inference risks in on-policy self-distillation and identify a sparse, structured correction imprint concentrated on uncertaintysensitive tokens. Based on this observation, we propose OP-Print, a unified gray-box and black-box attack framework that captures either privileged teacher–student discrepancies or their observable restart-based behavioral traces. Experiments across models and distillation schemes show that OP-Print consistently outperforms existing works.

## Limitaions

The black-box attack requires repeated restart generation, which introduces additional inference cost (but the adversaries can select the attack configuration themselves). We focus on attack effectiveness and leave defense design against correction-imprint leakage to future work.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. 2024a. On-policy distillation of language models: Learning from self-generated mistakes. In The twelfth international conference on learning representations.

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos Garea, Matthieu Geist, and Olivier Bachem. 2024b. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations.

Li Bai, Haibo Hu, Qingqing Ye, Haoyang Li, Leixia Wang, and Jianliang Xu. 2024. Membership inference attacks and defenses in federated learning: A survey. ACM Computing Surveys, 57(4):1–35.

Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer. 2022a. Membership inference attacks from first principles. In 2022 IEEE symposium on security and privacy (SP), pages 1897–1914. IEEE.

Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramèr. 2022b. Membership inference attacks from first principles. In 2022 IEEE Symposium on Security and Privacy, pages 1897–1914. IEEE.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. 2021. Extracting training data from large language models. In 30th USENIX security symposium (USENIX Security 21), pages 2633–2650.

Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. 2021. Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recogni tion, pages 3558–3568.

Dingfan Chen, Ning Yu, Yang Zhang, and Mario Fritz. 2020. Gan-leaks: A taxonomy of membership inference attacks against generative models. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, pages 343– 362.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. 2024. Self-play fine-tuning converts weak language models to strong language models. arXiv preprint arXiv:2401.01335.

Debeshee Das, Jie Zhang, and Florian Trantèr. 2025. Blind baselines beat membership inference attacks for foundation models. In 2025 IEEE Security and Privacy Workshops (SPW), pages 118–125. IEEE.

Phuong Tuan Dat, Qi Li, and Xinchao Wang. 2026. dopsd: On-policy self-distillation for diffusion language models. arXiv preprint arXiv:2607.04428.

Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, and Hannaneh Hajishirzi. 2024. Do membership inference attacks work on large language models? arXiv preprint arXiv:2402.07841.

Yingying Feng, Jie Li, Jie Hu, Yukang Zhang, Lei Tan, and Jiayi Ji. 2025a. Mdreid: Modalitydecoupled learning for any-to-any multi-modal object re-identification. Advances in Neural Information Processing Systems, 38:42584–42605.

Yingying Feng, Jie Li, Chi Xie, Lei Tan, and Jiayi Ji. 2025b. Multi-modal object re-identification via sparse mixture-of-experts. In International Conference on Machine Learning, pages 16796–16807. PMLR.

Etash Guha, Ryan Marten, Sedrick Keh, Negin Raoof, Georgios Smyrnis, Hritik Bansal, Marianna Nezhurina, Jean Mercat, Trung Vu, Zayne Sprague, et al. 2025. Openthoughts: Data recipes for reasoning models. arXiv preprint arXiv:2506.04178.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Jamie Hayes, Luca Melis, George Danezis, and Emiliano De Cristofaro. 2017. Logan: Membership inference attacks against generative models. arXiv preprint arXiv:1705.07663.

Yinghui He, Simran Kaur, Adithya Bhaskar, Yongjin Yang, Jiarui Liu, Narutatsu Ri, Liam Fowl, Abhishek Panigrahi, Danqi Chen, and Sanjeev Arora. 2026. Self-distillation zero: Self-revision turns binary rewards into dense supervision. arXiv preprint arXiv:2604.12002.

Hongsheng Hu, Zoran Salcic, Lichao Sun, Gillian Dobbie, Philip S Yu, and Xuyun Zhang. 2022. Membership inference attacks on machine learning: A survey. ACM Computing Surveys (CSUR), 54(11s):1–37.

Yuke Hu, Zheng Li, Zhihao Liu, Yang Zhang, Zhan Qin, Kui Ren, and Chun Chen. 2025. Membership inference attacks against vision-language models. arXiv preprint arXiv:2501.18624.

Jonas Hübotter, Frederike Lübeck, Lejs Behric, Anton Baumann, Marco Bagatella, Daniel Marta, Ido Hakimi, Idan Shenfeld, Thomas Kleine Buening, Carlos Guestrin, et al. 2026. Reinforcement learning via self-distillation. arXiv preprint arXiv:2601.20802.

Matthew Jagielski, Milad Nasr, Katherine Lee, Christopher A. Choquette-Choo, Nicholas Carlini, and Florian Tramèr. 2023. Students parrot their teachers: Membership inference on model distillation. In Advances in Neural Information Processing Systems, volume 36, pages 44382–44397.

Yuxin Jiang, Chunkit Chan, Mingyang Chen, and Wei Wang. 2023. Lion: Adversarial distillation of proprietary large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 3134–3154.

Seongryong Jung, Suwan Yoon, DongGeon Kim, and Hwanhee Lee. 2025. Todi: Token-wise distillation via fine-grained divergence control. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 8089–8102.

Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. 2024. Distillm: Towards streamlined distillation for large language models. arXiv preprint arXiv:2402.03898.

Qi Li, Cheng-Long Wang, Yinzhi Cao, and Di Wang. 2026a. Cola: A choice leakage attack framework to expose privacy risks in subset training. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16137–16149.

Qi Li, Runpeng Yu, and Xinchao Wang. 2026b. Vidsme: Membership inference attacks against large video understanding models. Advances in Neural Information Processing Systems, 38:111572–111596.

Zhan Li, Yongtao Wu, Yihang Chen, Francesco Tonin, Elias Abad Rocamora, and Volkan Cevher. 2024. Membership inference attacks against large visionlanguage models. Advances in Neural Information Processing Systems, 37:98645–98674.

Hangui Lin, Yan Shu, Zhengyang Liang, Chi Liu, Xiangrui Liu, Minghao Qin, Teng Long, Zheng Liu, and Nicu Sebe. 2026. Dyco-rl: Dynamic cross-modal coordination for visual reasoning. arXiv preprint arXiv:2606.08035.

Kevin Lu and Thinking Machines Lab. 2025. Onpolicy distillation. Thinking Machines Lab: Connectionism. Https://thinkingmachines.ai/blog/onpolicy-distillation.

Saeed Mahloujifar, Huseyin A Inan, Melissa Chase, Esha Ghosh, and Marcello Hasegawa. 2021. Membership inference on word embedding and beyond. arXiv preprint arXiv:2106.11384.

Matthieu Meeus, Igor Shilov, Shubham Jain, Manuel Faysse, Marek Rei, and Yves-Alexandre de Montjoye. 2025a. Sok: Membership inference attacks on llms are rushing nowhere (and how to fix it). In 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), pages 385–401. IEEE.

Matthieu Meeus, Igor Shilov, Shubham Jain, Manuel Faysse, Marek Rei, and Yves-Alexandre de Montjoye. 2025b. Sok: Membership inference attacks on llms are rushing nowhere (and how to fix it). In IEEE Conference on Secure and Trustworthy Machine Learning, pages 385–401. IEEE.

Ethan Mendes, Jungsoo Park, and Alan Ritter. 2026. Didactic to constructive: Turning expert solutions into learnable reasoning. arXiv preprint arXiv:2602.02405.

Ashwinee Panda, Xinyu Tang, Christopher A. Choquette-Choo, Milad Nasr, and Prateek Mittal. 2025. Privacy auditing of large language models. In International Conference on Learning Representations.

Emiliano Penaloza, Dheeraj Vattikonda, Nicolas Gontier, Alexandre Lacoste, Laurent Charlin, and Massimo Caccia. 2026. Privileged information distillation for language models. arXiv preprint arXiv:2602.04942.

Haritz Puerto, Martin Gubri, Sangdoo Yun, and Seong Joon Oh. 2025. Scaling up membership inference: When and how attacks succeed on large language models. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 4165–4182. Association for Computational Linguistics.

Ahmed Salem, Yang Zhang, Mathias Humbert, Pascal Berrang, Mario Fritz, and Michael Backes. 2018. Ml-leaks: Model and data independent membership inference attacks and defenses on machine learning models. arXiv preprint arXiv:1806.01246.

Hejian Sang, Yuanda Xu, Zhengze Zhou, Ran He, Zhipeng Wang, and Jiachen Sun. 2026. Crisp: Compressed reasoning via iterative self-policy distillation. arXiv preprint arXiv:2603.05433.

Idan Shenfeld, Mehul Damani, Jonas Hübotter, and Pulkit Agrawal. 2026. Self-distillation enables continual learning. arXiv preprint arXiv:2601.19897.

Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer. 2023. Detecting pretraining data from large language models. arXiv preprint arXiv:2310.16789.

Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. 2017a. Membership inference attacks against machine learning models. In 2017 IEEE symposium on security and privacy (SP), pages 3–18. IEEE.

Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. 2017b. Membership inference attacks against machine learning models. In 2017 IEEE Symposium on Security and Privacy, pages 3–18. IEEE.

Yan Shu, Hangui Lin, Yexin Liu, Yan Zhang, Gangyan Zeng, Yan Li, Yu Zhou, Ser Nam Lim, Harry Yang, and Nicu Sebe. 2026. When semantics mislead vision: Mitigating large multimodal models hallucinations in scene text spotting and understanding. Advances in Neural Information Processing Systems, 38:68952–68980.

Aasheesh Singh, Vishal Vaddina, and Dagnachew Birru. 2025. Orpo-distill: Mixed-policy preference optimization for cross-architecture llm distillation. arXiv preprint arXiv:2509.25100.

Congzheng Song and Ananth Raghunathan. 2020. Information leakage in embedding models. In Proceedings of the 2020 ACM SIGSAC conference on computer and communications security, pages 377– 390.

Liwei Song and Prateek Mittal. 2021. Systematic evaluation of privacy risks of machine learning models. In 30th USENIX Security Symposium (USENIX Security 21), pages 2615–2632.

Liwei Song, Reza Shokri, and Prateek Mittal. 2019. Membership inference attacks against adversarially robust deep learning models. In 2019 IEEE Security and Privacy Workshops (SPW), pages 50–56. IEEE.

Mingyang Song and Mao Zheng. 2026. A survey of onpolicy distillation for large language models. arXiv preprint arXiv:2604.00626.

Alex Stein, Furong Huang, and Tom Goldstein. 2026. Gates: Self-distillation under privileged context with consensus gating. arXiv preprint arXiv:2602.20574.

Lei Tan, Weihao Li, Pingyang Dai, Jie Chen, Liujuan Cao, and Rongrong Ji. 2026. Knowing where to focus: Attention-guided alignment for text-based person search. International Journal of Computer Vision, 134(2):72.

Lei Tan, Jiaer Xia, Wenfeng Liu, Pingyang Dai, Yongjian Wu, and Liujuan Cao. 2024a. Occluded person re-identification via saliency-guided patch transfer. In Proceedings ofthe AAAI conference on artificial intelligence, volume 38, pages 5070–5078.

Lei Tan, Yukang Zhang, Keke Han, Pingyang Dai, Yan Zhang, Yongjian Wu, and Rongrong Ji. 2024b. Rle: A unified perspective of data augmentation for crossspectral re-identification. Advances in Neural Information Processing Systems, 37:126977–126996.

Cheng-Long Wang, Qi Li, Zihang Xiang, Yinzhi Cao, and Di Wang. 2025. Towards lifecycle unlearning commitment management: Measuring sample-level unlearning completeness. In 34th USENIX Security Symposium (USENIX Security 25), pages 6481–6500.

Guangyang Wu, Youran Ding, Xinyu Che, Benyuan Sun, Yi Yang, and Xiaohong Liu. 2026. Matching every pair to track every point: Pairformer for allpairs tracking and video trajectory fields. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 35187– 35196.

Guangyang Wu, Huayu Zheng, Siqi Luo, Guangtao Zhai, and Xiaohong Liu. 2025a. Animateqr: Bridging aesthetics and functionality in dynamic qr code generation. In Advances in Neural Information Processing Systems, volume 38, pages 3861–3879.

Hengyu Wu and Yang Cao. 2025. Membership inference attacks on large-scale models: A survey. arXiv preprint arXiv:2503.19338.

Taiqiang Wu, Chaofan Tao, Jiahao Wang, Runming Yang, Zhe Zhao, and Ngai Wong. 2025b. Rethinking kullback-leibler divergence in knowledge distillation for large language models. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 5737–5755.

Jing Xiong, Hui Shen, Shansan Gong, Yuxin Cheng, Jianghan Shen, Chaofan Tao, Haochen Tan, Haoli Bai, Lifeng Shang, and Ngai Wong. 2026. Ovd: On-policy verbal distillation. arXiv preprint arXiv:2601.21968.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Tianzhu Ye, Li Dong, Zewen Chi, Xun Wu, Shaohan Huang, and Furu Wei. 2025. Black-box on-policy distillation of large language models. arXiv preprint arXiv:2511.10643.

Tianzhu Ye, Li Dong, Qingxiu Dong, Xun Wu, Shaohan Huang, and Furu Wei. 2026a. Online experiential learning for language models. arXiv preprint arXiv:2603.16856.

Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. 2026b. On-policy context distillation for language models. arXiv preprint arXiv:2602.12275.

Samuel Yeom, Irene Giacomelli, Matt Fredrikson, and Somesh Jha. 2018. Privacy risk in machine learning: Analyzing the connection to overfitting. In 2018 IEEE 31st computer security foundations symposium (CSF), pages 268–282. IEEE.

Bo Yin, Qi Li, and Xinchao Wang. 2026a. On-policy self-evolution via failure trajectories for agentic safety alignment. arXiv preprint arXiv:2605.11882.

Bo Yin, Qi Li, Runpeng Yu, and Xinchao Wang. 2026b. Refinement provenance inference: Detecting llmrefined training prompts from model behavior. arXiv preprint arXiv:2601.01966.

Sajjad Zarifzadeh, Philippe Liu, and Reza Shokri. 2023. Low-cost high-power membership inference attacks. arXiv preprint arXiv:2312.03262.

Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao Frank Yang, and Hai Li. 2024. Min-k%++: Improved baseline for detecting pre-training data from large language models. arXiv preprint arXiv:2404.02936.

Jingyang Zhang, Jingwei Sun, Eric Yeats, Yang Ouyang, Martin Kuo, Jianyi Zhang, Hao Frank Yang, and Hai Li. 2025a. Min-k%++: Improved baseline for pretraining data detection from large language models. In International Conference on Learning Representations.

Xueyi Zhang, Chengwei Zhang, Zheng Li, Xiyu Wang, Siqi Cai, Mingrui Lao, Yanming Guo, and Huiping Zhuang. 2026. Rep deep & machine learning: Exemplar-free continual video action recognition via slow-fast collaborative learning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 36075–36083.

Xueyi Zhang, Peiyin Zhu, Yuan Liao, Xiyu Wang, Mingrui Lao, Siqi Cai, Yanming Guo, and Haizhou Li. 2025b. Trustclip: Learning from noisy labels via semantic label verification and trust-aligned gradient projection. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 4388–4397.

Xueyi Zhang, Peiyin Zhu, Jinping Sui, Xiaoda Yang, Jiahe Tian, Mingrui Lao, Siqi Cai, Yanming Guo, and Jun Tang. 2025c. Choose your expert: Uncertaintyguided expert selection for continual deepfake detection. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 11502–11511.

Xueyi Zhang, Peiyin Zhu, Chengwei Zhang, Zhiyuan Yan, Jikang Cheng, Mingrui Lao, Siqi Cai, and Yanming Guo. 2025d. Generalization-preserved learning: Closing the backdoor to catastrophic forgetting in continual deepfake detection. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV), pages 3798–3808. IEEE.

Ziqi Zhang, Ali Shahin Shamsabadi, Hanxiao Lu, Yifeng Cai, and Hamed Haddadi. 2025e. Membership and memorization in LLM knowledge distillation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 20074–20084. Association for Computational Linguistics.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. 2026a. Self-distilled reasoner: On-policy selfdistillation for large language models. arXiv preprint arXiv:2601.18734.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. 2026b. Self-distilled reasoner: On-policy self-distillation for large language models. In Proceedings of the 43rd

International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pages 162433–162448. PMLR.

## A Training Details

Training data. All target models are trained on the mathematical reasoning subset of OpenThoughts (Guha et al., 2025). Following the standard protocol used in our evaluation, we first randomly partition the original training set into two disjoint halves using a fixed random seed $( \mathrm { i . e . , }$ 42), denoted as the train split and the $h e l d – o u t$ split. We then uniformly sample 500 examples from each split to construct the final member and non-member pools used throughout the paper. The member subset is used for target-model training, while the held-out subset is reserved for non-member evaluation only.

Teacher configuration. We use a fixed-teacher setup in all reported target-model training runs. Intuitively, the teacher corresponds to the initial policy augmented with privileged information, while the student is updated throughout training. This setting is used consistently across all model scales to isolate the privacy effects of on-policy selfdistillation under a stable teacher signal.

Optimization and generation. The training process is conducted on 8 NVIDIA A6000 GPUs. Training uses bfloat16 precision, gradient checkpointing, and on-policy generation with maximum completion length 1024. Sampling uses stochastic decoding with temperature, top-p, and top-k filtering. For efficient on-policy rollout generation, we use colocated vLLM during training. The distillation objective is computed on the generated continuation only, while prompt tokens are masked out from the loss.

Loss variants. For the standard OPSD variant, training uses a token-level generalized Jensen-Shannon divergence objective between teacher and student distributions on the sampled continuation. For the Tinker-style variant, the full-distribution matching loss is replaced by a sampled-token policy-improvement objective based on the teacherstudent log-probability difference on the same rollout. In both cases, the teacher signal is applied on the student’s own on-policy generations.

Implementation notes. All runs use the same random split seed for data construction. We save checkpoints periodically during training and use the resulting checkpoints in downstream membership inference evaluation. Unless explicitly stated otherwise, all attack results are reported on checkpoints trained under the settings summarized in Table 3.

## B Additional Analysis of the Black-box Score

In this appendix, we provide additional intuition and lightweight theoretical support for the blackbox component of OP-Print. Our goal is not to claim a fully general identifiability theorem, but to formalize why restart-based output stability is a reasonable surrogate for the hidden correction imprint, why repeated restarts reduce estimation variance, and why a more concentrated output basin leads to a larger black-box score.

## B.1 Restart-text Stability as a Surrogate for the Hidden Correction Imprint

Recall that in the main text, the key challenge of the black-box setting is that the adversary cannot directly observe the privileged teacher-student discrepancy along the rollout. We therefore introduce a latent variable that summarizes how strongly the correction process has already been internalized.

Latent correction strength. For an input example $x = ( q , r )$ and a prefix ratio $\rho ,$ let $C ( x , \rho ) \in \mathbb { R }$ denote the latent correction strength at that prefix. Intuitively, a larger $C ( x , \rho )$ means that by the time the model reaches the prefix $\pi _ { \rho } ( y )$ , the privileged correction relevant to x has already been more strongly absorbed into the model’s local decision geometry. We do not assume that $C ( x , \rho )$ is directly observable. Instead, we assume that the distribution of restarted continuations from the prefix $\pi _ { \rho } ( y ^ { ( m ) } )$ depends on $C ( x , \rho )$ through its effect on local basin concentration.

Population-level restart stability. Let $\tilde { Y }$ and ${ \tilde { Y } } ^ { \prime }$ be two independent restarted continuations sampled from the same prefix-conditioned restart distribution at ratio $\rho .$ . Define the population restart-text stability as

$$
\operatorname { S t a b } _ { \operatorname { t e x t } } ^ { \star } ( x , \rho ) : = \mathbb { E } \left[ \operatorname { s i m } ( \tilde { Y } , \tilde { Y } ^ { \prime } ) \Big | x , \rho \right] .\tag{6}
$$

This quantity measures the expected pairwise similarity between two independent restarts from the same truncated state.

Monotone surrogate assumption. The central assumption behind our black-box attack is that this population stability is monotone in the latent correction strength:

$$
{ \mathrm { S t a b } } _ { \mathrm { t e x t } } ^ { \star } ( x , \rho ) = g ( C ( x , \rho ) ) ,\tag{7}
$$

Table 3: Training configurations for target models. All models are trained under the same on-policy self-distillation pipeline; only model scale and a few runtime settings differ across runs.
<table><tr><td>Configuration</td><td>Qwen3-1.7B</td><td>Qwen3-4B</td><td>Qwen3-8B</td></tr><tr><td>Number of epochs</td><td> $^ 3$ </td><td>3</td><td> $^ 3$ </td></tr><tr><td>Learning rate</td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td><td> $5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Precision</td><td>bfloat16</td><td>bfloat16</td><td>bfloat16</td></tr><tr><td>Attention implementation</td><td>SDPA</td><td>SDPA</td><td>SDPA</td></tr><tr><td>Gradient checkpointing</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Max completion length</td><td>1024</td><td>1024</td><td>1024</td></tr><tr><td>Max sequence length</td><td>20000</td><td>20000</td><td>20000</td></tr><tr><td>Temperature</td><td>1.1</td><td>1.1</td><td>1.1</td></tr><tr><td>Top-p</td><td>0.95</td><td>0.95</td><td>0.95</td></tr><tr><td> $\mathrm { T o p } { \cdot } k$ </td><td>20</td><td>20</td><td>20</td></tr><tr><td> $\beta$ </td><td>0</td><td>0</td><td>0</td></tr><tr><td> $\lambda$ </td><td>1</td><td>1</td><td>1</td></tr><tr><td>vLLM GPU memory utilization</td><td>0.4</td><td>0.4</td><td>0.4</td></tr><tr><td>Max grad norm</td><td>0.1</td><td>0.1</td><td>0.1</td></tr><tr><td>JSD token clip</td><td>0.05</td><td>0.05</td><td>0.06</td></tr></table>

where $g ( \cdot )$ is non-decreasing. This assumption is natural: if the hidden correction has been more strongly internalized, then restarted decoding from an intermediate prefix should be more likely to remain in the same local solution basin, producing more stable continuations and hence higher pairwise text similarity.

From prefix-level surrogate to example-level surrogate. Suppose that the adversary uses a set of prefix ratios $\mathcal { P } \subset ( 0 , 1 )$ . Then the population blackbox score can be written as

$$
S _ { \mathrm { b l a c k } } ^ { \star } ( x ) : = \frac { 1 } { | \mathcal { P } | } \sum _ { \rho \in \mathcal { P } } \mathrm { S t a b } _ { \mathrm { t e x t } } ^ { \star } ( x , \rho ) .\tag{8}
$$

Substituting Equation (7) into Equation (8) gives

$$
S _ { \mathrm { b l a c k } } ^ { \star } ( x ) = \frac { 1 } { | \mathcal { P } | } \sum _ { \rho \in \mathcal { P } } g \bigl ( C ( x , \rho ) \bigr ) .\tag{9}
$$

Therefore, the population black-box score is an average monotone transform of the latent correction strength across prefix ratios. Now suppose that member examples tend to exhibit larger latent correction strength on average, i.e.,

$$
\begin{array} { r } { \mathbb { E } [ C ( x , \rho ) \mid x \in \mathcal { D } _ { \mathrm { m e m } } ] > \mathbb { E } [ C ( x , \rho ) \mid x \in \mathcal { D } _ { \mathrm { n o n } } ] } \\ { \mathrm { f o r ~ m a n y ~ } \rho \in \mathcal { P } . } \end{array}\tag{10}
$$

Because $g$ is non-decreasing, Equation (9) implies that

$$
\begin{array} { r l } & { \mathbb { E } [ S _ { \mathrm { b l a c k } } ^ { \star } ( x ) \mid x \in { \mathcal { D } } _ { \mathrm { m e m } } ] > } \\ & { \mathbb { E } [ S _ { \mathrm { b l a c k } } ^ { \star } ( x ) \mid x \in { \mathcal { D } } _ { \mathrm { n o n } } ] . } \end{array}\tag{11}
$$

Thus, under this weak monotonicity assumption, restart-text stability acts as an order-preserving surrogate for membership evidence in expectation.

Connection to the empirical score. In practice, we do not observe the population score in Equation (8); instead, we estimate it with finitely many full generations and finitely many restarts:

$$
S _ { \mathrm { b l a c k } } ( x ) = \frac { 1 } { M | \mathcal { P } | } \sum _ { m = 1 } ^ { M } \sum _ { \rho \in \mathcal { P } } \widehat { \mathrm { S t a b } } _ { \mathrm { t e x t } } ( \mathcal { R } _ { m , \rho } ( x ) ) .\tag{12}
$$

Hence, the black-box attack can be viewed as a Monte Carlo estimator of a population-level surrogate for the hidden correction imprint.

## B.2 Why Increasing the Number of Restarts Reduces Estimation Variance

We now formalize why increasing the restart count R stabilizes the black-box estimate.

Pairwise estimator. Fix an example $x ,$ a sampled full generation index $m ,$ and a prefix ratio $\rho .$ Let

$$
\tilde { Y } _ { 1 } , \dots , \tilde { Y } _ { R }
$$

be $R$ restarted continuations sampled independently from the same prefix-conditioned restart distribution. The empirical restart-text stability is

$$
\widehat { \mathrm { S t a b } } _ { \mathrm { t e x t } } = \frac { 2 } { R ( R - 1 ) } \sum _ { 1 \leq i < j \leq R } \sin ( \tilde { Y } _ { i } , \tilde { Y } _ { j } ) .\tag{13}
$$

Step 1: Unbiasedness. Define the kernel $h ( { \tilde { Y } } _ { i } , { \tilde { Y } } _ { j } ) =$ sim $( \tilde { Y } _ { i } , \tilde { Y } _ { j } )$ . Because all unordered pairs $( \tilde { Y } _ { i } , \tilde { Y } _ { j } )$ are identically distributed for $i \neq$ $j ,$ we have $\mathbb { E } [ h ( \tilde { Y } _ { i } , \tilde { Y } _ { j } ) ] ~ = ~ \mathbb { E } \Big [ \mathrm { s i m } ( \tilde { Y } , \tilde { Y } ^ { \prime } ) \Big ] ~ =$ $\mathrm { S t a b } _ { \mathrm { t e x t } } ^ { \star } ( x , \rho )$ , where $\tilde { Y } , \tilde { Y } ^ { \prime }$ are i.i.d. draws from the same restart distribution. Taking expectation on both sides of Equation (13),

$$
\begin{array} { r l r } {  { \mathbb { E } \Big [ \widehat { \mathrm { S t a b } } _ { \mathrm { t e x t } } \Big ] = \frac { 2 } { R ( R - 1 ) } \sum _ { 1 \le i < j \le R } \mathbb { E } [ h ( \tilde { Y } _ { i } , \tilde { Y } _ { j } ) ] } } \\ & { } & { = \frac { 2 } { R ( R - 1 ) } \binom { R } { 2 } \mathrm { S t a b } _ { \mathrm { t e x t } } ^ { \star } ( x , \rho ) } \\ & { } & { = \mathrm { S t a b } _ { \mathrm { t e x t } } ^ { \star } ( x , \rho ) . \qquad ( 1 4 } \end{array}
$$

So the empirical pairwise stability is an unbiased estimator of the population stability.

Step 2: Variance scaling. Equation (13) is a second-order U-statistic. Standard U-statistic theory implies that, under mild boundedness assumptions on the kernel h,

$$
\mathrm { V a r } \Big ( \widehat { \mathrm { S t a b } } _ { \mathrm { t e x t } } \Big ) = O \Big ( \frac { 1 } { R } \Big ) .\tag{15}
$$

The intuition is simple: although there are $\binom { R } { 2 }$ pairs, these pairs are not fully independent because they share restart samples; as a result, the variance decays more slowly than $O ( R ^ { - 2 } )$ but still decreases at rate $O ( R ^ { - 1 } )$

Step 3: Effect on the final black-box score. The final black-box score averages these estimators across multiple sampled full generations and prefix ratios:

$$
S _ { \mathrm { b l a c k } } ( x ) = \frac { 1 } { M | \mathcal { P } | } \sum _ { m = 1 } ^ { M } \sum _ { \rho \in \mathcal { P } } \widehat { \mathrm { S t a b } } _ { \mathrm { t e x t } } ( \mathcal { R } _ { m , \rho } ( x ) ) .\tag{16}
$$

If the restart estimators across $( m , \rho )$ pairs are weakly dependent or approximately independent, then averaging further reduces variance roughly by a factor of $M | \mathcal { P } |$ . Hence:

• increasing R reduces the variance of each prefix-level stability estimate;

• increasing M or $| \mathcal { P } |$ further stabilizes the final score through averaging.

This provides a simple theoretical explanation for the empirical trend observed in the ablation study: more restarts improve stability, but with diminishing returns.

## B.3 A Simple Two-State Toy Example

We finally give a toy model that explains why a more concentrated output basin leads to a larger black-box score.

Two-basin assumption. Suppose that for a fixed prefix, restarted decoding falls into one of two local basins:

• a dominant basin A, with probability p;

• an alternative basin B, with probability $1 - p .$

Assume that texts generated within the same basin are more similar than texts generated across different basins:

$$
\begin{array} { r l } & { \mathbb { E } [ \sin ( \tilde { Y } _ { i } , \tilde { Y } _ { j } ) \mid \tilde { Y } _ { i } , \tilde { Y } _ { j } \in A ] = s _ { \mathrm { i n } } , } \\ & { \mathbb { E } [ \sin ( \tilde { Y } _ { i } , \tilde { Y } _ { j } ) \mid \tilde { Y } _ { i } , \tilde { Y } _ { j } \in B ] = s _ { \mathrm { i n } } , } \\ & { \mathbb { E } [ \sin ( \tilde { Y } _ { i } , \tilde { Y } _ { j } ) \mid \tilde { Y } _ { i } \in A , \tilde { Y } _ { j } \in B ] = s _ { \mathrm { o u t } } , } \end{array}\tag{17}
$$

with

$$
s _ { \mathrm { i n } } > s _ { \mathrm { o u t } } .
$$

Step 1: Enumerate pair types. For two independent restarted outputs, there are three cases:

• both are in A, with probability $p ^ { 2 }$

• both are in B, with probability $( 1 - p ) ^ { 2 }$ ;

• one is in A and the other in B, with probability $2 p ( 1 - p )$

Step 2: Compute expected pairwise similarity. Using the law of total expectation,

$$
\begin{array} { r l } & { \mathbb { E } [ \sin ( \tilde { Y } _ { i } , \tilde { Y } _ { j } ) ] = \big [ p ^ { 2 } + ( 1 - p ) ^ { 2 } \big ] s _ { \mathrm { i n } } } \\ & { \qquad + 2 p ( 1 - p ) s _ { \mathrm { o u t } } . } \end{array}\tag{18}
$$

Now use the identity $p ^ { 2 } + ( 1 - p ) ^ { 2 } = 1 - 2 p ( 1 - p )$ to rewrite Equation (18) as

$$
\begin{array} { r l } & { \mathbb { E } [ \sin ( \tilde { Y } _ { i } , \tilde { Y } _ { j } ) ] = \big ( 1 - 2 p ( 1 - p ) \big ) s _ { \mathrm { i n } } } \\ & { \qquad + 2 p ( 1 - p ) s _ { \mathrm { o u t } } } \\ & { \qquad = s _ { \mathrm { i n } } - 2 p ( 1 - p ) ( s _ { \mathrm { i n } } - s _ { \mathrm { o u t } } ) . } \end{array}\tag{19}
$$

Equivalently,

$$
\mathbb { E } [ \sin ( \tilde { Y } _ { i } , \tilde { Y } _ { j } ) ] = s _ { \mathrm { o u t } } + \big ( p ^ { 2 } + ( 1 - p ) ^ { 2 } \big ) \big ( s _ { \mathrm { i n } } - s _ { \mathrm { o u t } } \big ) .\tag{20}
$$

Step 3: Interpret the shape. The quantity $p ^ { 2 } +$ $( 1 - p ) ^ { 2 }$ is minimized at $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ and increases as $p$ moves toward 0 or 1. Therefore, the expected pairwise similarity is smallest when restart trajectories are evenly split across the two basins, and largest when nearly all restarts collapse into one basin.

Since $s _ { \mathrm { i n } } > s _ { \mathrm { o u t } }$ , Equation (20) shows that:

• a dispersive output basin (mass spread across multiple competing continuations) yields lower restart-text stability;

• a concentrated output basin (mass concentrated on one dominant continuation mode) yields higher restart-text stability.

Step 4: Connection to membership inference. Suppose that member examples tend to have a more strongly internalized correction process, which makes their prefix-conditioned restart distribution more concentrated in one dominant basin. Then their corresponding p values are typically closer to 0 or 1, whereas non-member examples are more likely to have p values closer to $\hat { \frac { 1 } { 2 } }$ . By Equation (20), this implies

$$
\begin{array} { r l } { \mathbb { E } [ \mathrm { S t a b } _ { \mathrm { t e x t } } ] } & { \mathrm { i s ~ l a r g e r ~ f o r ~ m e m b e r s } } \\ & { \mathrm { t h a n ~ f o r ~ n o n { - } m e m b e r s } . } \end{array}
$$

This toy model therefore provides a simple interpretation of the black-box score: members are easier to attack because hidden correction makes their restart trajectories collapse into a tighter output basin.

## C Extended Related Work

On-Policy Distillation and Self-Distillation. Knowledge distillation transfers supervision from a teacher to a student, while on-policy distillation reduces training-inference mismatch by learning from student-generated trajectories (Agarwal et al., 2024b). Recent self-distillation approaches further eliminate external teachers by exploiting privileged contexts or reference solutions (Zhao et al., 2026b). Related research has explored reliable supervision through semantic label verification (Zhang et al., 2025b), uncertainty-guided expert selection (Zhang et al., 2025c), and generalization-preserving continual learning (Zhang et al., 2025d, 2026). Despite their different objectives, these approaches highlight how training signals influence learned representations and subsequent model behavior. However, the privacy implications of privilegedcontext self-distillation remain insufficiently understood. OP-Print investigates whether the coupled teacher-student conditioning process leaves distinctive membership-related imprints.

Membership Inference and Localized Information Leakage. Membership inference attacks (MIAs) determine whether a sample was used during model training (Shokri et al., 2017b; Carlini et al., 2022b). Existing LLM attacks exploit token probabilities, sequence likelihoods, and calibrated uncertainty statistics (Zhang et al., 2025a), while recent studies emphasize the importance of controlling distributional confounders (Meeus et al., 2025b; Puerto et al., 2025). The privacy risks of knowledge distillation have also attracted attention (Jagielski et al., 2023; Zhang et al., 2025e). Beyond privacy research, selective information modeling has been explored through cross-spectral augmentation (Tan et al., 2024b), modality-decoupled representations (Feng et al., 2025a), sparse mixtureof-experts (Feng et al., 2025b), saliency-guided feature transfer (Tan et al., 2024a), attention-based alignment (Tan et al., 2026), and dynamic crossmodal coordination (Lin et al., 2026). While addressing different problems, these studies illustrate the importance of identifying informative components rather than treating all information uniformly. OP-Print similarly examines localized training signals, revealing that membership information concentrates on uncertainty-sensitive correction sites rather than being uniformly distributed across generated trajectories.

Black-Box Behavioral Signals and Structured Dependencies. When internal model statistics are unavailable, membership inference must rely on observable generation behavior. Existing approaches investigate output consistency and memorization patterns, while privacy auditing further examines how training data influence model responses (Panda et al., 2025). Related challenges concerning structured information dependencies and output reliability arise in multimodal hallucination mitigation (Shu et al., 2026), dense temporal correspondence modeling (Wu et al., 2026), and constrained spatiotemporal generation (Wu et al., 2025a). OP-Print focuses on autoregressive generation, where intermediate prefixes constrain subsequent continuations. Its black-box attack measures restart-based output stability as an observable surrogate for hidden correction imprints, complementing the gray-box attack that directly aggregates privileged teacher-student discrepancies. Together, these approaches characterize membership leakage under different levels of model access.