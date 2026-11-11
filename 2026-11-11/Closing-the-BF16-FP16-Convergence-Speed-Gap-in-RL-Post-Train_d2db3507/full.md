# Closing the BF16-FP16 Convergence Speed Gap in RL Post-Training with Stochastic Rounding

Mahdi Nazemi Oleg Rybakov Tijmen Blankevoort Asit Mishra Carlo del Mundo NVIDIA

## Abstract

BF16 is widely used to pretrain large language models because it has the same dynamic range as FP32 and converges similarly to FP32 training. However, in reinforcement learning with verifiable rewards, where learning rates are typically low, we find that standard mixed-precision BF16 training converges more slowly than FP16 training at the same learning rate. We attribute this gap to weight sticking: when FP32 shadow weights are quantized to BF16 using round-to-nearest, small updates often leave the BF16 weights used for rollout generation and training unchanged. Consequently, several optimizer steps may be needed before updates to the shadow weights are reflected in the BF16 weights. Raising the BF16 learning rate reduces weight sticking and matches FP16's convergence speed, but it also increases the KL divergence from the reference model. We instead apply stochastic rounding when quantizing the FP32 shadow weights to BF16. This approach matches FP16's convergence speed across different models and datasets while reducing the KL divergence relative to BF16 training with a larger learning rate. Unlike FP16, it keeps BF16's wider dynamic range and needs no loss scaling.

![](images/a91e8c092796ad511d04cd645edcc5c172c6605502867a3ab68f71bd75a34ad0.jpg)

DeepSeek-R1-14B (DAPO)  
![](images/a660b99c54acb2f84084fab5f27849dd2e0f36764592176dec2ef2e0b57ed0dc.jpg)

DeepSeek-R1-14B (MATH)  
![](images/068d7ed334c3170d51dc1512fd1138c3de9d0c432d168c17066e7c642d6efa83.jpg)

DeepSeek-R1-1.5B (MATH)  
![](images/c88845a4944eb1d4d436366e3f8e79831046a43e68a5ede0eb09f05c4e733b18.jpg)  
---BF16 FP16BF16+SR

Figure 1: Applying stochastic rounding when quantizing the FP32 shadow weights to BF16 (BF16+SR) closes the BF16–FP16 convergence speed gap across different models and datasets. The FP16 and BF16+SR reward trajectories closely overlap.

## 1 Introduction

Mixed-precision BF16 training is widely used for pretraining large language models [Llama Team, 2024, Bi et al., 2024]. BF16 has the same dynamic range as FP32, and mixed-precision BF16 training converges similarly to FP32 training [Kalamkar et al., 2019]. BF16 is also broadly supported by modern AI accelerators, which typically offer substantially higher peak throughput for BF16 matrix multiplications than for FP32. These properties make BF16 a natural choice for reinforcement learning (RL) post-training as well.

However, at the low learning rates (LRs) common in reinforcement learning with verifiable rewards (RLVR), such as $1 0 ^ { - 6 }$ [Yu et al., 2025, Liu et al., 2025], we find that BF16 needs more RL iterations than FP16 to reach the same reward or downstream task score. This gap persists even when the rollout and training backends produce bit-exact log-probabilities for the same weights and inputs, so training-inference mismatch [Qi et al., 2025] alone cannot explain it. Throughout this paper, we measure convergence speed by the number of RL iterations needed to reach a given reward or score. We attribute the slower convergence to weight sticking: when FP32 shadow weights are quantized to BF16 using round-to-nearest, small updates often leave the BF16 weights used for rollout generation and training unchanged for several optimizer steps. Formats with fewer mantissa bits, such as BF16, are more prone to sticking [Miahi and Belilovsky, 2026, Dirhoussi et al., 2026].

Two simple fixes can reduce weight sticking: switching to FP16 or increasing the BF16 LR. Because of its narrower dynamic range, FP16 typically requires loss scaling to avoid gradient underflow. In preliminary FP16 RLVR experiments on agentic tasks, we observed frequent overflows even with loss scales much smaller than those common in pretraining. Each overflow skips the corresponding optimizer step, so the rollouts used for that step are wasted. Increasing the BF16 LR can recover FP16's convergence speed, but it also increases the policy's drift from the reference model, which we measure by KL divergence. Such drift is commonly penalized in RL post-training [DeepSeek-AI, 2025a], and it has been shown to predict how much a post-trained model forgets [Shenfeld et al., 2025].

We instead apply stochastic rounding (SR) when quantizing the FP32 shadow weights to BF16. SR rounds each shadow weight to one of its neighboring BF16 values, with probabilities set by the shadow weight's position between them. Even a small update to a shadow weight therefore changes the rounding probabilities, so the expected value of each BF16 weight moves with its FP32 shadow weight. BF16 with SR (BF16+SR) matches FP16's convergence speed (Fig. 1) with less drift than BF16 training at a higher LR. It also benefits from BF16's broad accelerator support and avoids the need for FP16 loss scaling.

Our contributions are as follows:

• We show that, at typical RLVR LRs, BF16 converges more slowly than FP16 even when the rollout and training backends produce bit-exact log-probabilities for the same weights and inputs.

• We identify weight sticking as a contributor to this slower convergence and show that increasing the number of mantissa bits in simulated BF16-like formats speeds up convergence.

• We show that applying SR when quantizing the FP32 shadow weights to BF16 matches FP16's convergence speed with less drift than raising the BF16 LR, and that applying SR in the forward passes alone is sufficient.

## 2 Weight Sticking and Stochastic Rounding

Standard mixed-precision training applies optimizer updates to FP32 shadow weights and quantizes weights to lower-precision formats, such as BF16 or FP16, for the forward and backward passes [Micikevicius et al., 2018]. In RL post-training, the rollout backend also uses the low-precision weights, so this weight quantization affects the policy that produces training data.

Let $w \in \mathbb { R }$ denote an FP32 shadow weight, and let $w ^ { - }$ and $w ^ { + }$ be adjacent representable values in the target format such that $w ^ { - } \leq w < w ^ { + }$ . Their spacing, $\mathrm { U L P } ( w ) = w ^ { + } - w ^ { - }$ , is one unit in the last place (ULP) of the target format at w. We consider quantizers Q that map w to either $w ^ { - } \mathrm { o r } w ^ { + }$

Round-to-nearest (RN) maps w to

$$
\begin{array} { r } { Q ^ { \mathrm { R N } } ( w ) = \left\{ \begin{array} { l l } { w ^ { - } , } & { w < \frac { w ^ { - } + w ^ { + } } { 2 } , } \\ { w ^ { + } , } & { w > \frac { w ^ { - } + w ^ { + } } { 2 } , } \\ { w _ { \mathrm { t i e } } , } & { w = \frac { w ^ { - } + w ^ { + } } { 2 } , } \end{array} \right. } \end{array}\tag{1}
$$

where $w _ { \mathrm { t i e } } \in \{ w ^ { - } , w ^ { + } \}$ depends on the tie-breaking rule.

Under RN quantization, weight sticking occurs when a nonzero update u to an FP32 shadow weight w leaves its quantized value unchanged, i.e., $Q ^ { \mathrm { R N } } ( w + u ) = \dot { Q } ^ { \mathrm { R N } } ( w )$ . Here, u may be a single optimizer update or the net update over several consecutive steps.

When post-training starts from a BF16 checkpoint, each initial shadow weight $w _ { 0 }$ lies on the BF16 grid, and, ignoring special cases, its quantized value stays at $w _ { 0 }$ as long as the net update is smaller than half a ULP:

$$
| u | < { \frac { \mathrm { U L P } ( w _ { 0 } ) } { 2 } } \implies Q ^ { \mathrm { R N } } ( w _ { 0 } + u ) = Q ^ { \mathrm { R N } } ( w _ { 0 } ) .\tag{2}
$$

BF16 has 7 mantissa bits compared to FP16's 10, so for values in the normal range of both formats, the BF16 ULP is 8× the FP16 ULP, and a BF16 weight needs an 8× larger net update to move. Equivalently, since the ULP grows with $| w _ { 0 } | ,$ a given update can move FP16 weights up to about 8× larger in magnitude than the largest BF16 weights it can move.

For a stuck weight, the RN quantization error exactly cancels the optimizer's net update to wo:

$$
e ^ { \mathrm { R N } } ( w _ { 0 } + u ) = Q ^ { \mathrm { R N } } ( w _ { 0 } + u ) - ( w _ { 0 } + u ) = - u .\tag{3}
$$

To remove this systematic error in expectation, SR quantizes a shadow weight according to

$$
Q ^ { \mathrm { S R } } ( w ) = \left\{ \begin{array} { l l } { w ^ { + } } & { \mathrm { w i t h ~ p r o b a b i l i t y } ~ \frac { w - w ^ { - } } { w ^ { + } - w ^ { - } } , } \\ { w ^ { - } } & { \mathrm { w i t h ~ p r o b a b i l i t y } ~ \frac { w ^ { + } - w } { w ^ { + } - w ^ { - } } . } \end{array} \right.\tag{4}
$$

Consequently, $\mathbb { E } \big [ Q ^ { \mathrm { S R } } ( w _ { 0 } + u ) \ | \ w _ { 0 } + u \big ] = w _ { 0 } + u ,$ , where the expectation is over the rounding randomness. SR replaces RN's systematic cancellation of updates in stuck weights with a rounding error that has zero mean but higher variance. We hypothesize that the added variance harms RLVR convergence less than RN's systematic error does, even when rollout generation and training use a single random realization of the quantized weights rather than their expectation. We test this hypothesis in Section 3 by applying SR when quantizing the FP32 shadow weights to BF16 while leaving the rest of the training recipe unchanged.

## 3 Experiments

## 3.1 Experimental Setup

Models and datasets. Most experiments train DeepSeek-R1-Distill-Qwen-1.5B (DeepSeek-R1- 1.5B) [DeepSeek-AI, 2025b] on MATH-1460 [Qi et al., 2025], a curated set of 1,460 competition math problems with binary correctness rewards. Larger-scale experiments train DeepSeek-R1-Distill-Qwen-14B (DeepSeek-R1-14B) [DeepSeek-AI, 2025b] on MATH-1460 or DAPO-Math-17k [Yu et al., 2025], and Qwen3-30B-A3B [Qwen Team, 2025] on MATH-1460. The TorchInductor study in Section 3.2 also trains Qwen3-1.7B [Qwen Team, 2025] on MATH-1460. We measure downstream accuracy on the AIME 2024 and AIME 2025 datasets and report mean@32, the accuracy averaged over 32 sampled responses per problem.

Training configuration. Unless noted otherwise, experiments use Dr. GRPO [Liu et al., 2025] in the verl framework [Sheng et al., 2025], with vLLM 0.12 [Kwon et al., 2023] for rollout generation and Hugging Face Transformers [Wolf et al., 2020] as the training backend. We use asymmetric clip ratios $\epsilon _ { \mathrm { l o } } = 0 . 2$ and $\epsilon _ { \mathrm { h i } } = 0 . 2 8$ [Yu et al., 2025], a dual-clip lower bound $c = 3 . 0$ [Ye et al., 2020], and a constant token-budget loss normalizer [Liu et al., 2025]. We use AdamW with FP32 shadow weights and a default LR of $\eta = 1 0 ^ { - 6 }$ . Each RL iteration samples 8 responses for each of 512 problems and splits the resulting 4,096 sequences into 4 mini-batches of 1,024, so each RL iteration takes 4 optimizer steps. The maximum response length is 8,192 tokens. FP16 runs use dynamic loss scaling with PyTorch's ShardedGradScaler and a growth interval of 400 steps.

Numerical parity and hardware. For tightly controlled experiments, we add run-to-run determinism and training-inference numerical parity for selected models in verl. Determinism makes runs reproducible, and parity removes training-inference mismatch as a confounder, so numerical formats and rounding schemes can be compared fairly. We achieve parity by using batch-invariant kernels [He and Lab, 2025] and by aligning the numerics of the rollout and training backends so that they produce bit-exact log-probabilities for the same weights and inputs. Enforcing parity reduces throughput, so we reserve it for the BF16–FP16 parity comparison and the weight sticking measurements in Section 3.2, which run on 64 NVIDIA H100 GPUs. All other experiments run on 64 NVIDIA B200 GPUs.

Data types. Regardless of the numerical format or rounding scheme, attention accumulation, RMS normalization, log-softmax, optimizer states, and gradient all-reduce use FP32. Matrix multiplications and their inputs (weights, activations, and gradients), attention outputs, and residual additions use BF16 or FP16.

## 3.2 Results and Discussion

BF16 converges more slowly than FP16 even after removing training-inference mismatch. Qi et al. [2025] reported reward collapse in BF16 RLVR and attributed it to training-inference mismatch. In our setup, the collapse occurs only when vLLM's TorchInductor compiler optimizations are enabled; disabling them prevents it (Fig. 2) and increases end-to-end step time by less than 2%. We observed this behavior with DeepSeek-R1-1.5B and Qwen3-1.7B on all vLLM releases we used during this work (0.12 to 0.17), but we have not tested later releases (the latest is 0.33 at the time of writing). We therefore disable TorchInductor optimizations in all remaining experiments. With TorchInductor disabled and parity enforced, FP16 still reaches a given reward level in fewer RL iterations than BF16 at the same LR (Fig. 3). In the FP16 run, ShardedGradScaler never skipped an optimizer step.

BF16 has a higher weight sticking rate than FP16 at RLVR's low LRs. Table 1 reports the percentage of weights that still equal their values in the BF16 checkpoint after each of the four optimizer steps in the first RL iteration. Across the transformer blocks of the DeepSeek-R1-1.5B model, the median weight magnitude is $2 . 5 \times 1 0 ^ { - 2 }$ , and the 25th and 75th percentiles are $1 . 2 \times 1 0 ^ { - 2 }$ and $4 . 5 \times 1 0 ^ { - 2 }$ . At the median magnitude, half a ULP is about $6 . 1 \times 1 0 ^ { - 5 }$ in BF16 and $7 . 6 \times 1 0 ^ { - 6 }$ in ${ \mathrm { F P } } 1 6 ,$ both of which are well above the Adam update of roughly $\eta = 1 0 ^ { - 6 }$ . Most weights therefore remain unchanged in both formats. However, because FP16's half-ULP threshold is 8× smaller, the same update changes a larger fraction of the small-magnitude weights, and FP16 changes roughly 8× as many weights as BF16 by the end of the first RL iteration (9.21% versus 1.16%).

Equation (2) suggests two ways to reduce weight sticking: shrinking the ULP, e.g., by increasing the number of mantissa bits, and increasing the optimizer's update size, e.g., by increasing the LR. Figure 4a tests the first with simulated formats that keep BF16's 8 exponent bits but have $m \in \{ \bar { 8 } , 9 , 1 0 \}$ mantissa bits. Each added mantissa bit speeds up convergence, and the $m = 1 0$ format tracks FP16 closely throughout training. These results support limited mantissa precision as a contributor to the convergence speed gap. Figure 4b tests the second with LRs from $1 0 ^ { - 6 }$ to

![](images/aaebe6abd33fea0f16f3b50417a60f05581e43cf357fc5219a7fa1ae51da590b.jpg)  
Figure 2: Disabling TorchInductor compiler optimizations in vLLM prevents BF16 reward collapse.

![](images/036142043b269246b1016a48176405aa5bf0a1ddad0463c926d4d7985b669320.jpg)  
Figure 3: Under training-inference numerical parity, FP16 reaches a given mean reward and AIME score (mean@32) in fewer RL iterations than BF16 (DeepSeek-R1-1.5B, MATH-1460).

Table 1: Percentage of quantized weights equal to their initial checkpoint values after each optimizer step in the first RL iteration (DeepSeek-R1-1.5B, MATH-1460 $\eta \doteq 1 0 ^ { - 6 } )$
<table><tr><td>Optimizer Step</td><td>BF16</td><td>FP16</td></tr><tr><td>1</td><td>99.42%</td><td>95.35%</td></tr><tr><td>2</td><td>99.23%</td><td>93.86%</td></tr><tr><td>3</td><td>98.93%</td><td>91.51%</td></tr><tr><td>4</td><td>98.84%</td><td>90.79%</td></tr></table>

$8 \times 1 0 ^ { - 6 }$ . A higher LR speeds up convergence for both formats, but FP16 stays ahead of BF16 at every LR.

Stochastic rounding closes the convergence speed gap. BF16+SR matches the FP16 reward trajectory across the evaluated model–dataset combinations, spanning 1.5B, 14B, and 30B-A3B models and two training datasets (Fig. 1). On AIME 2024 and AIME 2025, BF16 initially trails FP16 and BF16+SR but usually closes the gap later in training (Fig. 5). Early in training, FP16 and BF16+SR produce shorter responses than BF16 for all models and datasets, but all three configurations eventually converge to similar response lengths (Fig. 6). The only exception is the Qwen3-30B-A3B model, for which BF16+SR produces responses about 25% shorter than the other two configurations. We leave the study of this behavior to future work.

A higher BF16 LR closes the gap but increases drift. BF16 at $\eta = 2 \times 1 0 ^ { - 6 }$ follows a reward trajectory similar to FP16 at $\eta = \bar { 1 } 0 ^ { - 6 } ( \mathrm { F i g . 4 b } )$ , so tuning the LR is a viable way to close the gap.

![](images/dd32430acfe2acf0b126a1e7335e99aabd828b691465a30b259ae7a6f44b9741.jpg)  
(a) Mantissa bit-width sweep

![](images/e478f76f5030d08be1adcf78f3e66cf097b5afe08f36ee41ba632c70240898b3.jpg)  
(b) LR sweep  
Figure 4: Effect of mantissa bit-width and LR on convergence (DeepSeek-R1-1.5B, MATH-1460). (a) A simulated BF16-like format with 10 mantissa bits tracks FP16. (b) A higher LR speeds up convergence for both formats, but FP16 stays ahead at every LR.

![](images/89202146e4cd4a7a4361112e9855fc48af0473675ceb534eb3bac6ee01f2f131.jpg)  
Figure 5: AIME 2024 and AIME 2025 scores (mean@32). BF16 initially trails FP16 and BF16+SR on downstream task scores but usually closes the gap as training progresses.

However, a higher LR also changes the optimization dynamics, so we compare the resulting drift from the reference model using the k3 estimator of the KL divergence. Figure 7 plots the lowest measured drift at each reward level. For mean rewards above 0.7, BF16 at $\eta = 2 \times 1 0 ^ { - 6 }$ drifts about 1.5× as much as FP16, whereas BF16+SR stays close to FP16. Higher LRs can therefore close the convergence speed gap, but at the cost of more drift from the reference model.

Stochastic rounding of forward-pass weights is sufficient. We compared several SR variants on DeepSeek-R1-1.5B and MATH-1460. Drawing a fresh SR realization in every forward pass gives the same reward trajectory as drawing one realization after each optimizer step and using it for all forward passes until the next optimizer step, including rollout generation after weight synchronization. Also applying SR to the weights in the backward pass, or to the gradients, does not speed up convergence further. Applying SR only in the rollout backend, however, is not enough: the mean reward plateaus at a value well below FP16's. We therefore recommend the simplest variant: apply SR once after each optimizer step, use the resulting BF16 weights for all forward passes in both backends until the next optimizer step, and use round-to-nearest for all other conversions.

## 4 Conclusions and Future Work

We showed that, at the low LRs typical of RLVR, BF16 converges more slowly than FP16, and that this convergence speed gap persists under training-inference numerical parity. We attribute the gap to weight sticking: with RN and only 7 mantissa bits, a BF16 weight reflects small updates to its shadow weight only after the net update crosses a rounding boundary.

We compared three remedies. Switching to FP16 works well for the models and datasets in this paper, but in preliminary RLVR experiments with Qwen3.5-397B-A17B on an agentic software engineering task, FP16 training overflowed frequently even at a loss scale of 1,024, which is much smaller than the loss scales common in pretraining. Raising the BF16 LR matches FP16's convergence speed but increases drift from the reference model, which has been shown to predict how much a post-trained model forgets. Applying stochastic rounding when quantizing the FP32 shadow weights to BF16 closes the convergence speed gap with drift close to FP16's.

We expect weight sticking to worsen in formats with even fewer mantissa bits than BF16. AI accelerators typically provide higher throughput for matrix multiplications in lower-precision formats.

![](images/d832571ceea1880b5ce0799997fa81131cab994a18d07e6db7e97151ce1eda85.jpg)

Figure 6: Mean response length during training. After an initial phase in which FP16 and BF16+SR produce shorter responses, all three configurations converge to similar lengths, except on Qwen3- 30B-A3B.  
![](images/2e62ff49921f641c338e24c3a56d9c79c60a8dfa0f7d97f6a6d7b7a8b4abf398.jpg)  
Figure 7: Drift from the reference model at matched reward (DeepSeek-R1-1.5B, MATH-1460). BF16 with a higher LR closes the reward gap but incurs larger drift, while BF16+SR at the default LR stays close to FP16.

so weight sticking could limit their use for RLVR. Preliminary asynchronous RLVR experiments on Qwen3.5-397B-A17B, in which rollouts lag the trained policy by at most four policy versions, suggest that SR speeds up convergence in BF16 as well as in low-precision formats such as MXFP8. We leave a detailed study of RLVR using low-precision formats to future work.

## References

X. Bi, D. Chen, G. Chen et al., "DeepSeek LLM: scaling open-source language models with longtermism," CoRR, vol. abs/2401.02954, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2401.02954

DeepSeek-AI, “DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning," CoRR, vol. abs/2501.12948, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2501.12948

DeepSeek-AI, "DeepSeek-V3.2: Pushing the frontier of open large language models," CoRR, vol. abs/2512.02556, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2512.02556

A. Dirhoussi, Q. Gallouédec, E. Beeching, L. Tunstall, K. Rasul, and L. von Werra, "Defeating the trainer-generator precision mismatch in TRL," 2026. [Online]. Available: https://huggingface.co/spaces/ aminediroHF/trainer-generator-bf16-mismatch

H. He and T. M. Lab, "Defeating nondeterminism in LLM inference," Thinking Machines Lab: Connectionism, 2025, https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/.

D. D. Kalamkar, D. Mudigere, N. Mellempudi et al., “A study of BFLOAT16 for deep learning training," CoRR, vol. abs/1905.12322, 2019. [Online]. Available: http://arxiv.org/abs/1905.12322

W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model serving with PagedAttention," in Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023. ACM, 2023, pp. 611–626. [Online]. Available: https://doi.org/10.1145/3600006.3613165

Z. Liu, C. Chen, W. Li, P. Qi, T. Pang, C. Du, W. S. Lee, and M. Lin, "Understanding R1-Zero-like training: A critical perspective," CoRR, vol. abs/2503.20783, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2503.20783

Llama Team, "The Llama 3 herd of models," CoRR, vol. abs/2407.21783, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2407.21783

E. Miahi and E. Belilovsky, “"Understanding and exploiting weight update sparsity for communication-efficient distributed RL," CoRR, vol. abs/2602.03839, 2026. [Online]. Available: https://doi.org/10.48550/arXiv.2602. 03839

P. Micikevicius, S. Narang, J. Alben et al., “Mixed precision training," in 6th International Conference on Learning Representations, ICLR 2018, Vancouver, BC, Canada, April 30 - May 3, 2018, Conference Track Proceedings. OpenReview.net, 2018. [Online]. Available: https://openreview.net/forum?id=r1gs9JgRZ

P. Qi, Z. Liu, X. Zhou, T. Pang, C. Du, W. S. Lee, and M. Lin, "Defeating the training-inference mismatch via FP16," CoRR, vol. abs/2510.26788, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2510.26788

Qwen Team, “Qwen3 technical report," CoRR, vol. abs/2505.09388, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2505.09388

I. Shenfeld, J. Pari, and P. Agrawal, “RL's razor: Why online reinforcement learning forgets less," CoRR, vol. abs/2509.04259, 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2509.04259

G. Sheng, C. Zhang, Z. Ye, X. Wu, W. Zhang, R. Zhang, Y. Peng, H. Lin, and C. Wu, "HybridFlow: A flexible and efficient RLHF framework," in Proceedings of the Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 - 3 April 2025. ACM, 2025, pp. 1279–1297. [Online]. Available: https://doi.org/10.1145/3689031.3696075

T. Wolf, L. Debut, V. Sanh et al., “Transformers: State-of-the-art natural language processing," in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, EMNLP 2020 - Demos, Online, November 16-20, 2020. Association for Computational Linguistics, 2020, pp. 38–45. [Online]. Available: https://doi.org/10.18653/v1/2020.emnlp-demos.6

D. Ye, Z. Liu, M. Sun et al., “Mastering complex control in MOBA games with deep reinforcement learning," in The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020. AAAI Press, 2020, pp. 6672–6679. [Online]. Available: https://doi.org/10.1609/aaai.v34i04.6144

Q. Yu, Z. Zhang, R. Zhu et al., "DAPO: an open-source LLM reinforcement learning system at scale," in Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025. [Online]. Available: http://papers.nips.cc/paper\_files/paper/2025/ hash/a4277440d50f1f15d2cb4c14f7e0c0d2-Abstract-Conference.html