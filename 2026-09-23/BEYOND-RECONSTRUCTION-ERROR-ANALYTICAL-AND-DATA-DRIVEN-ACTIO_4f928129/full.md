# BEYOND RECONSTRUCTION ERROR: ANALYTICAL AND DATA-DRIVEN ACTIONTOKENIZATION FOR AUTOREGRESSIVE VISION-LANGUAGE-ACTION MODELS

Yuxin Yang<sup>1,†</sup>, Gaohan He<sup>2,†</sup>, Changxue Guan<sup>1</sup>, Hangming Liu<sup>3,∗</sup>

<sup>1</sup>College of Artificial Intelligence, Southwest University <sup>2</sup>Ocean College, Zhejiang University <sup>3</sup>Tianfu Securities Co., Ltd.

tianyiyoung127@gmail.com Hannah.he@zju.edu.cn gcx 0601@qq.com liuhm@tianfuzq.cn

## ABSTRACT

Discrete action tokenization is central to autoregressive visionlanguage-action (VLA) models, yet action representations are often evaluated primarily through reconstruction fidelity. We ask which representation properties actually matter for closedloop control by comparing fixed analytical, data-driven linear, and nonlinear neural representations under a unified tokenization interface. Across rate–distortion analysis, sequencemodeling diagnostics, and 3,500 LIBERO rollouts, representation rankings change with the evaluation criterion. PCA achieves lower nominal reconstruction error than Temporal-DCT, but produces less predictable token sequences and 3.0 percentage points lower mean seen-task success across three policy-training seeds, with the policy ordering reversing in one seed. In a matched seed-42 ablation, an autoencoder further reduces reconstruction error yet does not yield the strongest policy and exhibits greater sensitivity to discrete token perturbations. These findings show that reconstruction fidelity alone cannot reliably select action representations for autoregressive control, motivating joint evaluation of geometric fidelity, sequence predictability, decoder stability, and closed-loop performance.

Index Terms— Action Tokenization, Vision-Language-Action, Discrete Cosine Transform, Principal Component Analysis, Robot Learning.

## 1. INTRODUCTION

Autoregressive vision-language-action (VLA) models formulate robot control as sequence prediction by mapping continuous actions to discrete tokens [1–5]. This interface is consequential: an action tokenizer must preserve task-relevant motion information while producing sequences that remain compact and predictable for autoregressive decoding. Consequently, tokenizer quality can directly affect closed-loop execution rather than merely compression efficiency. Early VLA systems rely on per-dimension discretization [1, 2], while recent approaches increasingly tokenize multi-step action chunks to reduce sequence length and capture temporal structure.

FAST [3], for example, uses a fixed temporal Discrete Cosine Transform (DCT) [6] to exploit trajectory smoothness, whereas subsequent approaches explore learned vectorquantized codebooks [4] and mixture-of-experts tokenizers [5]. Despite these architectural advances, a basic representation question remains unresolved. Action tokenizers are often characterized by compression or reconstruction quality, yet autoregressive control imposes additional requirements: tokens must remain predictable to the policy, and discrete prediction errors should not induce disproportionately large changes in decoded actions. Thus, lower geometric reconstruction error does not necessarily imply better closed-loop control. This raises a fundamental question: what properties of an action representation actually matter for autoregressive robotic execution? This suggests viewing action tokenization not merely as a compression problem, but as an interface jointly coupling geometric representation, autoregressive sequence modeling, and error propagation into physical actions.

We investigate this question through three representative representation families under a unified tokenization interface (Fig. 1). Temporal-DCT provides a fixed analytical harmonic prior; Principal Component Analysis (PCA) provides a datadriven orthogonal basis that captures cross-axis correlations; and an MLP autoencoder provides a flexible nonlinear latent representation. By keeping normalization, quantization, token budget, VLA backbone, optimization, and rollout protocols matched within each comparison, we isolate how representation choice affects rate-distortion behavior, sequence predictability, quantization sensitivity, and closed-loop performance.

Our main findings are threefold:

• Representation Trade-offs: Across 20 rate–distortion regimes and cross-distribution tests, PCA attains lower nominal reconstruction error than Temporal-DCT, while DCT is more robust under coarse quantization and yields more predictable sequences.

• Closed-Loop Policy Performance: In 3,500 closedloop rollouts with Qwen3-VL-4B on LIBERO-Spatial, Temporal-DCT exceeds PCA by 3.0 pp on average across three seeds, despite its higher nominal reconstruction error; the ordering reverses in one seed.

• Reconstruction–Control Misalignment: In a matched ablation, an autoencoder attains the lowest reconstruction error but not the highest success rate. Its larger

![](images/80ac2bef1c864b6bc1f3f8ac97068808593e80fc07dc10dfcf9b5735c4c93836.jpg)

![](images/aa06085d63c4e391f0dbd8ea1a3aa2c8ef05af65812d45e371e0626905e4fb1a.jpg)  
Fig. 1. (a) Controlled evaluation pipeline: only the action representation is varied, while the VLA backbone, token budget, vocabulary, optimizer, and rollout protocol are matched. (b) Seed-42 reconstruction–control misalignment: the representation with the lowest reconstruction MSE is not the one with the highest seen-task success.

Table 1. Reconstruction MSE across 20 rate-distortion regimes on LIBERO-Spatial. P/D denotes PCA/Temporal-DCT as the lower-MSE representation; values indicate relative MSE reduction ∆ (%). Bold highlights nominal setting.
<table><tr><td>Tokens K</td><td>2-Bit</td><td>4-Bit</td><td>6-Bit</td><td>8-Bit</td><td>16-Bit</td></tr><tr><td>K = 7</td><td>D 13.3</td><td>P20.7</td><td>P 24.9</td><td>P 25.2</td><td>P 25.2</td></tr><tr><td>K = 14</td><td>D 10.2</td><td>P3.0</td><td>P7.7</td><td>P 8.3</td><td>P 8.3</td></tr><tr><td>K = 21</td><td>D 9.4</td><td>D 1.8</td><td>P1.6</td><td>P2.5</td><td>P2.5</td></tr><tr><td> $K = 2 8$ </td><td>D 9.1</td><td>D 1.7</td><td>P 2.8</td><td>P4.5</td><td>P4.4</td></tr></table>

perturbation gain further identifies decoder stability as a property hidden by reconstruction MSE.

## 2. BASIS ACTION TOKENIZATION PIPELINE

To isolate the effect of the underlying representation basis, we deliberately omit downstream BPE compression and expose all methods through an identical fixed-K scalar-token interface. Thus, our Temporal-DCT condition evaluates the analytical representation stage underlying FAST [3] rather than reproducing the complete FAST tokenizer.

## 2.1. Trajectory Chunking and Normalization

Following action chunking [3, 7, 8], continuous control trajectories are sliced into temporal windows of horizon $T = 1 6$ with action dimension D = 7 (3D Cartesian translation, 3D axis-angle rotation, and 1D binary gripper state). A single chunk is flattened into $\mathbf { a } \in \mathbb { R } ^ { M }$ , where $M = T \times D = 1 1 2$

To guarantee strictly leakage-free conditioning, normalizers are fitted exclusively on training demonstrations (Tasks 0–8): (1) Coordinate Scaling: 6-DoF end-effector pose commands are scaled to [−1, 1] based on training bounds, while binary gripper commands are kept unscaled in [0, 1]; (2) Standardization: the flattened 112D vector is z-score standardized as $\tilde { \mathbf { a } } = ( \mathbf { a } - \pmb { \mu } ) \oslash \pmb { \sigma }$

## 2.2. Analytical, Linear, and Neural Tokenizers

A linear tokenizer projects standardized chunk a˜ $\in \mathbb { R } ^ { M }$ onto K orthonormal basis vectors $\{ \mathbf { b } _ { k } \} _ { k = 1 } ^ { K } \left( \mathbf { B } \in \mathbb { R } ^ { K \times M } , \mathbf { B } \mathbf { B } ^ { \top } = \right.$

I<sub>K</sub>):

$$
\mathbf { c } = \mathbf { B } \tilde { \mathbf { a } } \in \mathbb { R } ^ { K } , \quad \hat { \mathbf { a } } = \mathbf { B } ^ { \top } \mathbf { c } \in \mathbb { R } ^ { M } .\tag{1}
$$

Temporal-DCT Basis: Following FAST [3], independent 1D DCT-II transforms are applied along the time horizon T for each actuator channel. The 1D DCT matrix $\mathbf { C } \in \mathbb { R } ^ { T \times T }$ is defined by:

$$
C _ { u , t } = \sqrt { \frac { 2 } { T } } \alpha _ { u } \cos \left( \frac { \pi ( 2 t + 1 ) u } { 2 T } \right) ,\tag{2}
$$

where $\alpha _ { 0 } = 1 / \sqrt { 2 }$ and $\alpha _ { u } = 1$ for $u \geq 1$ . To ensure all channels remain controllable, basis vectors are assigned in frequency priority: for $K = 7 m$ , we retain the first m frequency components across all $D = 7$ channels $( K \in \{ 7 , 1 4 , 2 1 , 2 8 \}$ corresponds to m $\in \ \{ 1 , 2 , 3 , 4 \}$ coefficients/channel, with nominal $K = 1 4 , m = 2 )$

PCA / Karhunen–Loeve Basis: The data-driven linear basis is obtained from the empirical covariance matrix ${ \boldsymbol { \Sigma } } =$ $\mathbb { E } [ ( \tilde { \mathbf { a } } - \bar { \mathbf { a } } ) ( \tilde { \mathbf { a } } - \bar { \mathbf { a } } ) ^ { \top } ]$ on the training trajectories. SVD yields eigenvectors $\mathbf { v } _ { 1 } , \ldots , \mathbf { v } _ { M }$ sorted by decreasing eigenvalue $\lambda _ { i } .$ The top K principal components form the rows of $\mathbf { B } _ { \mathrm { P C A } }$ While DCT treats actuators independently, PCA adapts to empirical cross-axis kinematic correlations across end-effector axes.

Nonlinear Autoencoder (AE): We train an MLP Autoencoder with symmetrical bottleneck architecture (112 → 64 → 14 → 64 → 112) using GELU activations to compress chunks into a 14D continuous latent bottleneck. The AE serves as a reconstruction-optimized nonlinear baseline rather than a quantization-aware learned tokenizer.

## 2.3. Symmetric Uniform Quantization

Let b denote the quantization precision with codebook vocabulary size $V = 2 ^ { b } \left( V = 2 5 6 \right.$ for nominal 8-bit quantization). Projection coefficients $\mathbf { c } \in \mathbb { R } ^ { K }$ are quantized into discrete tokens $\mathbf { z } \in \{ 0 , \ldots , V - 1 \} ^ { K }$ . Each coefficient $c _ { k }$ is standardized by its training standard deviation $\sigma _ { c , k } ,$ clipped to [−5, +5], and uniformly quantized:

Table 2. Closed-loop success (%) on LIBERO-Spatial over three seeds (50 paired initial states/task). Tasks 0–8 are seen; all six PCA/DCT policies score 0/50 on held-out Task 9. ± denotes standard deviation across policy-training seeds.
<table><tr><td>Condition</td><td>Task 0</td><td>Task 1</td><td>Task 2</td><td>Task 3</td><td>Task 4</td><td>Task 5</td><td>Task 6</td><td>Task 7</td><td>Task 8</td><td>Seen Avg (0–8)</td></tr><tr><td>PCA (seed 42)</td><td>96%</td><td>90%</td><td>94%</td><td>66%</td><td>44%</td><td>88%</td><td>94%</td><td>92%</td><td>56%</td><td>80.0% (360/450)</td></tr><tr><td>PCA (seed 43)</td><td>88%</td><td>84%</td><td>100%</td><td>70%</td><td>78%</td><td>92%</td><td>96%</td><td>90%</td><td>70%</td><td>85.3% (384/450)</td></tr><tr><td>PCA (seed 44)</td><td>94%</td><td>86%</td><td>100%</td><td>80%</td><td>60%</td><td>84%</td><td>92%</td><td>86%</td><td>76%</td><td>84.2% (379/450)</td></tr><tr><td>PCA (3-Seed Mean)</td><td>92.7%</td><td>86.7%</td><td>98.0%</td><td>72.0%</td><td>60.7%</td><td>88.0%</td><td>94.0%</td><td>89.3%</td><td>67.3%</td><td>83.2% ± 2.8%</td></tr><tr><td>DCT (seed 42)</td><td>90%</td><td>88%</td><td>96%</td><td>84%</td><td>56%</td><td>90%</td><td>94%</td><td>84%</td><td>94%</td><td>86.2% (388/450)</td></tr><tr><td>DCT (seed 43)</td><td>90%</td><td>96%</td><td>98%</td><td>92%</td><td>82%</td><td>90%</td><td>98%</td><td>88%</td><td>82%</td><td>90.7% (408/450)</td></tr><tr><td>DCT (seed 44)</td><td>92%</td><td>96%</td><td>86%</td><td>94%</td><td>26%</td><td>86%</td><td>88%</td><td>86%</td><td>80%</td><td>81.6% (367/450)</td></tr><tr><td>DCT (3-Seed Mean)</td><td>90.7%</td><td>93.3%</td><td>93.3%</td><td>90.0%</td><td>54.7%</td><td>88.7%</td><td>93.3%</td><td>86.0%</td><td>85.3%</td><td> $8 6 . 2 \% \pm 4 . 6 \%$ </td></tr><tr><td>∆ (PCA – DCT)</td><td>+2.0</td><td>-6.7</td><td>+4.7</td><td>-18.0</td><td>+6.0</td><td>-0.7</td><td>+0.7</td><td>+3.3</td><td>-18.0</td><td>-3.0 pp</td></tr></table>

$$
z _ { k } = \left\lfloor \left( \exp \left( \frac { c _ { k } } { \sigma _ { c , k } } , - 5 , 5 \right) + 5 \right) \cdot \frac { V - 1 } { 1 0 } \right\rceil .\tag{3}
$$

Dequantization maps $z _ { k }$ back to continuous approximation ${ \hat { c } } _ { k } = \left( z _ { k } \cdot { \frac { 1 0 } { V - 1 } } - 5 \right) \cdot \sigma _ { c , k }$ . For linear orthogonal models (PCA and Temporal-DCT), the continuous standardized chunk is recovered via transpose projection $\hat { \tilde { \mathbf { a } } } = \mathbf { B } ^ { \top } \hat { \mathbf { c } }$ . For the Autoencoder, the continuous latent vector h<sup>ˆ</sup> = cˆ is decoded via its nonlinear decoder network $\hat { \tilde { \mathbf { a } } } = g ( \hat { \mathbf { h } } )$ . Both are finally de-standardized to restore coordinate-scaled action commands $\hat { \mathbf { a } } \in [ - 1 , 1 ] ^ { 6 } \times [ 0 , 1 ]$ . All reported reconstruction MSEs are measured post-quantization in this coordinate-scaled action space.

## 3. OFFLINE REPRESENTATION ANALYSIS

## 3.1. Rate–Distortion and Quantization Sensitivity

We first examine how the representation basis interacts with token budget and quantization precision. Table 1 reveals a clear regime shift rather than a uniformly superior basis. At moderate and high precision, PCA benefits from concentrating trajectory variance into a compact data-adaptive subspace, outperforming Temporal-DCT in 14 of the 20 evaluated configurations. At the nominal setting (K = 14, 8-bit), this yields an 8.3% reduction in reconstruction MSE (0.0083 vs. 0.0091).

This advantage reverses under extreme 2-bit quantization, where Temporal-DCT is consistently better across all token budgets. A plausible structural explanation is that Temporal DCT preserves channel-wise harmonic structure, whereas each PCA component generally mixes multiple action dimensions. Consequently, coarse coefficient errors in PCA can be redistributed across several physical coordinates after inverse projection. These results indicate that a basis that is favored under nominal reconstruction need not remain optimal once quantization noise dominates the error budget.

## 3.2. Cross-Task and Cross-Dataset Generalization

We next test whether the learned PCA basis remains advantageous outside the distribution on which it is fitted. In leaveone-task-out evaluation within LIBERO-Spatial, PCA retains a modest advantage, winning 6 of 10 held-out tasks and reducing macro-average MSE by 6.2%. (For this offline diagnostic, the PCA basis is refitted on the remaining nine tasks in each fold; this is distinct from our downstream policy protocol, where Task 9 is strictly excluded from fitting and training.) This suggests that learned cross-axis structure transfers across related manipulation tasks.

Table 3. Token predictability and policy loss on LIBERO-Spatial (K = 14, $V = 2 5 6 ;$ lower is better). Policy loss is teacher-forced on seed-42 Task 0.
<table><tr><td colspan="5"></td><td rowspan="2">Policy Loss (nats/token)</td></tr><tr><td>Tokenizer</td><td>Entropy (bits)</td><td>PPL-1</td><td>PPL-2</td><td>PPL-3</td></tr><tr><td>PCA</td><td>6.665</td><td>105.61</td><td>108.20</td><td>180.81</td><td>3.513</td></tr><tr><td>Temporal-DCT</td><td>5.919</td><td>92.38</td><td>90.11</td><td>143.37</td><td>3.419</td></tr><tr><td>DCT Advantage</td><td>–0.746 bits</td><td>-12.5%</td><td>-16.7%</td><td>-20.7%</td><td>-2.7%</td></tr></table>

A different pattern emerges under zero-shot cross-dataset transfer. On LIBERO-Object, neither basis consistently dominates. On LIBERO-Goal, however, the preferred representation depends strongly on token budget: Temporal-DCT achieves 22.9–42.2% lower MSE at $K \in \{ 7 , 1 4 \}$ , whereas PCA recovers the advantage at $K \in \{ 2 1 , 2 8 \}$ and improves MSE by up to 28.3% at $K = 2 8$ . This crossover suggests that data-driven bases are most effective when sufficient representational capacity is available to capture domain-specific correlations, while a fixed harmonic prior can be more robust in this transfer setting when the representation budget is strongly constrained.

## 3.3. Sequence Predictability

Geometric fidelity is only one aspect of an action tokenizer for an autoregressive policy; the resulting discrete sequence must also be predictable from preceding tokens. We estimate token entropy and n-gram perplexity on disjoint episode splits using Lidstone smoothing (Table 3).

Despite higher nominal reconstruction error, Temporal-DCT yields an easier prediction problem: position-wise entropy is 0.75 bits lower and perplexity drops by 12.5–20.7% (Table 3). Teacher-forced policy loss on seed-42 Task 0 follows the same direction (3.419 vs. 3.513 nats/token). Thus, a representation may sacrifice geometric fidelity while producing statistically simpler sequences.

## 4. DOWNSTREAM ROBOTIC POLICY EVALUATION

## 4.1. Evaluation Protocol

We instantiate each tokenizer in the same Qwen3-VL-4B policy [9]. Action IDs $( V = 2 5 6 )$ map to dedicated tokens with dual 128 × 128 image inputs and no proprioception. Policies use LoRA [10] $( r = 6 4 , \alpha = 1 2 8 )$ on attention projections, trained for 30k steps with AdamW. Each step predicts $K = 1 4$ tokens, decoded to T = 16 actions. Normalizers and datadriven tokenizers are fitted on Tasks 0–8; Task 9 is strictly excluded. Paired evaluation over 50 initial states per task/seed across 7 runs yields 3,500 rollouts (113.6 H800 GPU-hours). Macro success uncertainty is estimated via 20,000-draw paired bootstrap.

## 4.2. Closed-Loop Control

Table 2 shows that PCA’s reconstruction advantage does not translate monotonically into control. Temporal-DCT attains 86.2% mean seen-task success, compared with 83.2% for PCA. It leads by 6.2 and 5.3 pp for seeds 42 and 43, respectively; the corresponding paired-bootstrap 95% intervals for PCA−DCT are [−10.7, −1.8] and [−9.3, −1.3] pp. Seed 44 reverses the ordering, with PCA ahead by 2.7 pp, but its interval [−1.8, +7.1] includes zero. Hence, the aggregate 3.0-pp DCT advantage is reproducible in direction for two seeds, but not invariant to policy optimization. Given only three independent policy-training seeds, we therefore treat the cross-seed mean gap as descriptive rather than as an inferential claim over optimization randomness.

PCA has a higher three-seed mean on five of nine seen tasks, but DCT gains 18.0 pp on Tasks 3 and 8, producing the higher macro-average. Both tokenizers score 0/50 on held-out Task 9 across all seeds, precluding cross-task comparison and reflecting a general floor effect.

## 4.3. Reconstruction–Control Misalignment

The nonlinear autoencoder provides a second counterexample to reconstruction-driven tokenizer selection. As shown in Table 4, it reduces post-quantization MSE by 36.7% relative to PCA and 41.9% relative to Temporal-DCT. Nevertheless, its 82.7% seen-task success lies between PCA (80.0%) and DCT (86.2%): the reconstruction ranking AE>PCA>DCT does not match the control ranking DCT>AE>PCA. Greater representational flexibility therefore improves geometric fidelity without determining the downstream policy ordering.

To probe this discrepancy, we measure sensitivity to discrete token perturbations. Let $\hat { \mathbf { a } } ( \mathbf { z } ) = \mathcal { D } ( Q ^ { - 1 } ( \mathbf { z } ) )$ denote the decoded action from token vector $\mathbf { z } ,$ where $\mathcal { D } ( \cdot ) = \mathbf { B } ^ { \top } ( \cdot )$ for linear bases (PCA/DCT) and $\mathcal { D } ( \cdot ) = g ( \cdot )$ for the autoencoder. We evaluate the perturbation gain:

Table 4. Tokenizer ablations on LIBERO-Spatial (seed 42, $K = 1 4 ) . G _ { z }$ is mean perturbation gain under $| \delta z | = 4 ;$ lower is better.
<table><tr><td>Condition</td><td>Representation Type</td><td>Recon. MSE ↓</td><td> $G _ { z } \downarrow$ </td><td>Seen ↑</td></tr><tr><td>PCA</td><td>Learned linear</td><td>0.00834</td><td>0.0336</td><td>80.0%</td></tr><tr><td>Temporal-DCT</td><td>Fixed analytical</td><td>0.00909</td><td>0.0337</td><td>86.2%</td></tr><tr><td>Autoencoder</td><td>Nonlinear neural</td><td>0.00528</td><td>0.0431</td><td>82.7%</td></tr></table>

$$
G _ { z } = \frac { \| \hat { \mathbf { a } } ( \mathbf { z } + \delta \mathbf { z } ) - \hat { \mathbf { a } } ( \mathbf { z } ) \| _ { 2 } } { \| \delta \mathbf { z } \| _ { 2 } } .\tag{4}
$$

We perturb token coordinates by $\delta z _ { j } \in \{ \pm 1 , \pm 2 , \pm 4 \}$ , clip to valid vocabulary range, and average $G _ { z }$ over chunks, all K coordinates, and both signs in coordinate-scaled action space $( \mathbf { a } \in [ - 1 , 1 ] ^ { 6 } \times [ 0 , 1 ] )$ . The autoencoder exhibits 26.5–30.3% larger mean amplification than linear bases; at $| \delta z _ { j } | = 4 , G _ { z }$ reaches 0.0431 vs. 0.0336 for PCA/DCT (Table 4). This diagnostic identifies decoder sensitivity omitted by reconstruction MSE.

## 4.4. Synthesis of Representation Trade-offs

Together, these findings establish a non-monotonic link between rate–distortion quality and control. PCA attains lower nominal reconstruction error among the linear bases, yet DCT produces more predictable sequences, remains robust under coarse quantization, and achieves higher mean seen-task success in our evaluation. The autoencoder minimizes reconstruction MSE but fails to lead control and incurs the highest perturbation gain. Action-tokenizer evaluation should therefore separate geometricfidelity, sequence predictability, and decoder stability.

## 5. LIMITATIONS

Our study evaluates LIBERO-Spatial with one VLA backbone, three training seeds, and fixed K = 14 scalar quantization. Morphological diversity, policy scale, learned codebooks, and BPE compression remain open; the observed ordering should not be assumed to transfer unchanged. Perturbation gain is local rather than causal, and the Task 9 floor precludes held-out conclusions.

## 6. CONCLUSION

In this controlled autoregressive VLA study, rankings diverge: DCT leads PCA by 3.0 pp in mean seen-task success despite worse reconstruction, while the autoencoder minimizes distortion without leading control. Reconstruction alone therefore cannot select a policy representation; tokenizers require sequence-aware diagnostics and paired closed-loop evaluation.

## 7. REFERENCES

[1] Brianna Zitkovich, Tianhe Yu, Sichun Xu, Peng Xu, Ted Xiao, Fei Xia, Jialin Wu, Paul Wohlhart, Stefan Welker, Ayzaan Wahid, Quan Vuong, Vincent Vanhoucke, Huong Tran, Radu Soricut, Anikait Singh, Jaspiar Singh, Pierre Sermanet, Pannag R. Sanketi, Grecia Salazar, Michael S. Ryoo, Krista Reymann, Kanishka Rao, Karl Pertsch, et al., “RT-2: Vision-language-action models transfer web knowledge to robotic control,” in Proc. Conference on Robot Learning (CoRL), 2023, pp. 2165–2183.

[2] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P. Foster, Pannag R. Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn, “Open-VLA: An open-source vision-language-action model,” in Proc. Conference on Robot Learning (CoRL), 2024, pp. 2679–2713.

[3] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine, “FAST: Efficient action tokenization for vision-language-action models,” in Proc. Robotics: Science and Systems (RSS), 2025.

[4] Yicheng Liu, Shiduo Zhang, Zibin Dong, Baijun Ye, Tianyuan Yuan, Xiaopeng Yu, Linqi Yin, Chenhao Lu, Junhao Shi, Luca Jiang-Tao Yu, Liangtao Zheng, Jingjing Gong, Tao Jiang, Xipeng Qiu, and Hang Zhao, “FASTer: Toward powerful and efficient autoregressive visionlanguage-action models with learnable action tokenizer and block-wise decoding,” in Proc. International Conference on Learning Representations (ICLR), 2026.

[5] Chunpu Xu, Zhixuan Liang, Tianshuo Yang, Chi-Min Chan, Yang Xiao, Jessie Wang, Xiaokang Yang, and Yao Mu, “MoEActok: A MoE-based action tokenizer for vision-language-action models,” in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2026, pp. 28042–28051.

[6] Nasir Ahmed, T. Natarajan, and Kamisetty R. Rao, “Discrete cosine transform,” IEEE Transactions on Computers, vol. C-23, no. 1, pp. 90–93, 1974.

[7] Tony Z. Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn, “Learning fine-grained bimanual manipulation with low-cost hardware,” in Proc. Robotics: Science and Systems (RSS), 2023.

[8] Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, and Shuran Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” in Proc. Robotics: Science and Systems (RSS), 2023.

[9] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, et al., “Qwen3-VL technical report,” arXiv preprint arXiv:2511.21631, 2025.

[10] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen, “LoRA: Low-rank adaptation of large language models,” in Proc. International Conference on Learning Representations (ICLR), 2022.