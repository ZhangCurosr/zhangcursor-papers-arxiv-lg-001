# Adversarial Post-Training of Foundation Models for High-Frequency Trading

Alexandre Bismuth   
alexandre.bismuth@cs.ox.ac.uk   
British Open-ended Learning and   
Discovery Lab (BOLD)   
Department of Computer Science   
University of Oxford   
Oxford, United Kingdom   
Harvard University   
Cambridge, MA, United States   
Valentin Mohl   
valentin.mohl@cs.ox.ac.uk   
British Open-ended Learning and   
Discovery Lab (BOLD)   
Department of Computer Science   
University of Oxford   
Oxford, United Kingdom   
Kang Li<sup>∗</sup>   
kang.li@stats.ox.ac.uk   
British Open-ended Learning and   
Discovery Lab (BOLD)   
Department of Statistics   
University of Oxford   
Oxford, United Kingdom   
Jakob Foerster   
jakob@robots.ox.ac.uk   
British Open-ended Learning and   
Discovery Lab (BOLD)   
Department of Engineering Science   
University of Oxford   
Oxford, United Kingdom   
Aramis Marti-Shahandeh   
aramis@robots.ox.ac.uk   
British Open-ended Learning and   
Discovery Lab (BOLD)   
Department of Engineering Science   
University of Oxford   
Oxford, United Kingdom   
Anisoara Calinescu   
ani.calinescu@cs.ox.ac.uk   
British Open-ended Learning and   
Discovery Lab (BOLD)   
Department of Computer Science   
University of Oxford   
Oxford, United Kingdom

## Abstract

While autoregressive foundation models trained on limit order book (LOB) data show state-of-the-art performance for market simulation, pre-training with cross-entropy neither penalizes the compounding of sampling errors over free-running rollouts nor targets desired market properties. We introduce a post-training method for LOB foundation models using a Wasserstein generative adversarial network (GAN) objective. A critic, trained online on rollouts through the true, non-diferentiable matching engine, guides updates from evolution strategies over rank-4 LoRA perturbations of the pre-trained generator, under a KL trust region. For eficiency, we use EGGROLL (Evolution Guided GeneRal Op timisation via Low-rank Learning), a memory-eficient variant of evolution strategies. On a held-out panel, adversarial post-training improves LOB-Bench realism by 17<sub>.</sub>6% (WS-21) and 17<sub>.</sub>0% (Mean-$L _ { 1 } )$ , while selection-matched random-perturbation and computematched continued-pre-training controls yield no improvement. Post-training is thus a necessary stage that more pre-training does not replace. Per-position divergence measurements show that it slows error accumulation over 500-message rollouts from 3<sub>.</sub>1 to 2<sub>.</sub>3 . Adversarial post-training is also inexpensive, requiring under 50 M training tokens and 12 GPU-hours per seed, three orders of magnitude below the 48 B-token pre-training budget. Ablations over critic inputs, optimizers, and the population–rollout trade-of, together with an analysis of reward hacking, yield practical design guidelines for adversarial post-training.

• Computing methodologies Neural networks; Continuous space search; • Applied computing Economics.

## Keywords

Foundation models, Limit order books, Market simulation, Evolution strategies, Post-training, Adversarial networks

Alexandre Bismuth, Kang Li, Aramis Marti-Shahandeh, Valentin Mohl,Jakob Foerster, and Anisoara Calinescu. 2026. Adversarial Post-Training of Foundation Models for High-Frequency Trading. In 7th ACM International Conference on AI in Finance (ICAIF ’26), November 14–17, 2026, Milan, AA, Italy. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3858063.3858908

## 1 Introduction

Recent work has established foundation models for high-frequency trading as the state-of-the-art for market simulation through autoregressive training on limit order book (LOB) messages, executed through a matching engine [20, 26, 34]. These models are pre-trained with cross-entropy, maximizing next-token likelihood over historical message streams and thus implicitly matching the market’s conditional next-message distribution. However, nexttoken prediction leaves a gap that more pre-training does not close: it creates compounding errors over long horizons, an efect known as exposure bias [3, 29], slowly drifting to unrealistic scenarios.

Equal contribution.

Here, adversarial methods ofer a promising solution: train a critic to distinguish real LOB sequences from generated rollouts, and update the pre-trained model to fool it. This method implicitly estimates a divergence between the two sequence distributions and, unlike hand-crafted statistics, continually reallocates capacity to whichever discrepancies the generator currently exhibits. The obstacle is that rollouts are executed through the exchange’s matching engine, which is non-diferentiable: no gradient connects the critic’s score back to the generator’s parameters. We therefore turn to zero-order optimization—methods such as EGGROLL [31] or GRPO [32] that require only scalar rewards evaluated on rollouts, never a backward pass through the simulator.

![](images/efd9cba46ac428eb3afdbf1de9e5cee4089f9898792ba9ed94886426dc1fb89f.jpg)  
Figure 1: Market simulation (left): a model continues LOB messages through a non-diferentiable exchange simulator. Adversar ial post-training (center): a critic trained online scores generated rollouts against real message streams, and EGGROLL converts scores into rank-4 LoRA [16] perturbations under a KL trust region. Evaluation (right): ofline selection with LOB-Bench and held-out evaluation to mitigate reward hacking. Solid arrows carry data; dashed arrows carry learning signals.

Furthermore, while this work focuses on market realism for LOB foundation models, our framework can be adapted to any desired metric. One promising target is market-impact fidelity, relevant to agentic trading (Section 7.1). Outside of the HFT literature, our approach also applies to other autoregressive models where compounding errors create long-horizon drifts and non-diferentiability prevents the use of gradient-based methods.

## 1.1 Contributions

(1) An adversarial post-training framework for HFT foundation models. A Wasserstein critic, trained online and from scratch on free-running rollouts executed through the true, non-diferentiable matching engine, steers gradientfree optimizers that update a pre-trained generator through rank-4 LoRA perturbations under a KL trust region.

(2) Improved market realism at negligible cost. Tested over held-out months, post-training improves WS-21 by 17<sub>.</sub>6% and Mean-�<sub>1</sub> by 17<sub>.</sub>0% while consuming under 50 M training tokens and 12 GPU-hours—a significant refinement after 170 GPU-hours and 48 B tokens of pre-training.

(3) An anatomy of adversarial post-training. Ablation ex periments allow us to define robust guidelines for pipeline design: raw data roughly doubles the gains of hand-designed inputs, small populations lose nothing at fixed compute, and reward-hacking requires careful evaluation protocols.

## 2 Background

Foundation models for market simulation. Electronic markets provide an ideal starting point for the development of foundation models thanks to the vast quantity of raw data. Every order submission, cancellation, deletion, and execution is a time-stamped message, and the exchange’s matching engine deterministically folds the message stream into the limit order book (LOB, Figure 2). Data feeds such as LOBSTER [17] preserve this stream losslessly, turning market simulation into a sequence-modeling problem: an autoregressive model over tokenized messages, pre-trained with crossentropy on historical flow and combined with a matching engine, defines a fully interactive synthetic market [20, 26]. GPU-native engines make such closed-loop rollouts fast enough to train and evaluate against [10]. Because next-message likelihood alone does not certify realism, generated markets are assessed distributionally over longer sequences: LOB-Bench [27] scores the discrepancy between real and generated rollouts across microstructure statistics (e.g. spreads, depths, volumes, order-flow imbalance, inter-arrival times) and is the evaluation standard we adopt.

Generative Adversarial Networks. A Generative Adversarial Network (GAN) trains a generator against a discriminator that learns to separate real from generated samples [13]. In the Wasserstein formulation, a 1-Lipschitz critic estimates the Wasserstein-1 distance between the two distributions via Kantorovich–Rubinstein duality, yielding a training signal that stays informative even when the two distributions are nearly disjoint [2]; the Lipschitz constraint is enforced by gradient penalties [14] or spectral normalization [25], and training is stabilized by two-time-scale updates [15] and the zero-centered �<sub>1</sub> penalty, which restores local convergence [24].

Zero-order optimization methods. Zero-order methods optimize objectives that can be evaluated but not diferentiated: instead of a gradient, they build an update from the scores of sampled candidates [33, 35]. On one hand, policy-gradient methods assign each sampled sequence an advantage, defined as how much better it scored than a baseline, and raise or lower its log-likelihood accordingly, shifting probability mass toward higher-reward samples. GRPO [32] replaces the learned value baseline with the group itself: it draws a group of rollouts per context, z-scores their rewards into group-relative advantages, and constrains the update with a KL penalty to a reference model. On the other hand, Evolution Strategies (ES) sample perturbations of the parameters, score each perturbed model on the objective, and combine the perturbations, weighted by rank-shaped antithetic scores, into an update [30]. EGGROLL [31] makes this eficient by drawing each perturbation as a rank-<sub>�</sub> outer product (collapsing per-member memory from � <sub>��</sub> to � <sub>� � �</sub> ), and is competitive with other policy optimization methods used to fine-tune modern large language models. Both families need only scalar rollout scores and therefore tolerate a non-diferentiable environment; they difer in where the update is estimated (weight space versus token space) and in what must be differentiated (nothing versus the generator’s own log-probabilities).

![](images/ed7e2951173f6129ff0887a4478d692477a8f8caf9b3fc4a578af75643e85c0e.jpg)  
Figure 2: The limit order book and its message stream from records in LOBSTER format. Bar length represents volume.

## 3 Related Work

Data-driven market simulation. Agent-based simulators such as ABIDES [6] compose hand-crafted trading agents around an exchange model; they are interactive by construction but struggle to match the fine structure of real order flow. A data-driven line therefore replaces the agent pool with a learned world agent: Stock-GAN generates order streams with a conditional Wasserstein GAN, sidestepping the non-diferentiable exchange by learning a neural approximation of the matching mechanism [21]; Coletta et al. develop CGAN world agents deployed inside multi-agent simulations [8, 9] and analyze their explainability and robustness [7]; Hultin et al. factorize message generation into a chain of conditional distributions [18]; and TRADES brings denoising difusion to the same world-agent setting [4]. Most recently, token-level autoregressive foundation models [20, 26, 34] have become the strongest generative simulators. Our work is orthogonal to this scaling approach: we take a pre-trained foundation model as given and ask how a second, adversarial training stage improves it.

Adversarial training for sequence models. For discrete sequences, discriminator gradients cannot flow through sampling; SeqGAN resolved this by treating generation as a policy updated with REIN-FORCE against discriminator rewards [36], a pattern our GRPO arm modernizes. Closest to us in spirit, adversarial post-training refines pre-trained difusion models against real data after likelihood-style pre-training [22]: pre-train for coverage, then optimize an adversarial objective for fidelity. We transport this two-stage pattern to market generation, where the adversarial stage must additionally traverse a hard exchange simulator, and pair it with gradient-free optimizers rather than backpropagation.

Reward-driven post-training. Post-training pre-trained models against scalar rewards is standard practice in the LLM literature, via policy-gradient methods such as GRPO [32] or, increasingly, evolution strategies [28, 30]. Two recent observations shape our design point. First, while hyperscale ES pre-training relies on enormous populations [31], the neighborhood of a large pre-trained model is dense with improving directions [11], suggesting that post-training needs orders of magnitude fewer. Second, any learned reward is eventually over-optimized [12]; our online critic inherits this in the form of a measurable hacking window. Unlike RLHFstyle pipelines whose reward model is frozen, our reward is a live adversary trained online, exploited on a documented schedule, and disciplined by external selection on held-out statistics.

## 4 Methods

## 4.1 Preprocessing

Dataset. The training corpus contains GOOG stock information from 2022 to 2025, covering about 1,000 trading days. Testing is completed over 28 trading days in January and February 2026. The pre-trained generator (the anchor in our study) learns from 50% of the training corpus. Post-training is performed on unseen training data, avoiding any unfair specialization or recency advantage. Un like existing literature, we train on the full message-stream instead of focusing on the top-k levels of the books.

The test period shows no dramatic regime shift. Event composition and volatility are stable across periods, while activity and price trend upward through the training years (Table 1).

Table 1: Statistics for the GOOG stock.
<table><tr><td>Period</td><td>Msgs/day (M) Exec. (%) Spread (bps) Daily RV (%)</td></tr><tr><td>2022-2025 4.41</td><td>1.45 1.18 1.44 157</td></tr><tr><td>2025 Q1 5.95</td><td>1.20 1.25 1.50</td></tr><tr><td>2025 Q2 6.87</td><td>183 1.80 165</td></tr><tr><td>1.55 2025 Q3 5.15</td><td>1.20 1.14 1.31 210</td></tr><tr><td>1.40 0.88</td><td>1.48 1.56 286</td></tr><tr><td>2025 Q4 8.03 Test 6.17</td><td>1.29 1.24 1.70 329</td></tr></table>

Limit order book. The LOBSTER [17] format contains messagelevel information about the GOOG stock. It provides event types, 10−<sup>4</sup> USD prices, volume, IDs, and nanosecond-resolution timestamps. We discard out-of-hours messages, hidden executions, cross trades, trading-halt indicators, and clip volumes at 9,999 shares.

Tokenization. LOBSTER messages are each encoded as 26 tokens over a shared vocabulary of   = 2<sub>,</sub>112, with four reserved special tokens (mask, hidden-event, not-applicable, start-of-sequence). Fields cover the message’s event type, direction, timing, price, and size, plus the referenced order’s attributes where applicable; sizes and price magnitudes are split into digit-token pairs (base-100 and base-1,000 respectively), shrinking the vocabulary while covering 999<sub>,</sub>999 ticks. Absolute timestamps are kept in the loss for temporal consistency but are filled deterministically at inference. Logit masking enforces each position’s valid-token set, guaranteeing syntactic validity (e.g., event types in 1<sub>,</sub> 2<sub>,</sub> 3<sub>,</sub> 4 ) though not semantic validity (e.g., canceling an order absent from the book).

Generator. The pre-trained generator is a 78<sub>.</sub>5 M-parameter Mamba-3 state-space model [19]. Two message-encoder layers embed the token stream and two book-encoder layers embed the order-book state; both feed six fused SSM layers (width $d _ { \mathrm { m o d e l } } = 1 , 0 2 4$ , state size 128) and a linear head over . The model conditions on 500-message contexts (13<sub>,</sub>000 tokens) and generates by sampling each token from its 50 most likely values (top-� sampling, �=50), at about 50k tokens per second on a single GH200 GPU.

Critic input. Generated data is fed into the critic to promote representation learning. We perform lossless transforms into channels (Figure 3): four one-hot fields for event type, a direction, a price, a size, a time diference, and four reference channels, defined for messages that reference another order. We intentionally discard absolute price, absolute time, and order IDs as trivial tells can separate real from fake without reflecting realism: NA in fake rollouts, non-stationary or pure bookkeeping in real data. These channels are combined with median normalization to mitigate heavy tails that dominate scales. The pool of 512 real context windows is redrawn every five steps, keeping the critic’s real set briefly stationary while preventing memorization of individual windows. At each step, <sub>�</sub>=128 contexts are sampled without replacement and shared across perturbations so fitness circumvents context-sampling noise.

## 4.2 Adversarial training

Wasserstein GAN (W-GAN). At each step, the generator rolls out $G \times Q = 1 6 { \times } 1 2 8 = 2 { , } 0 4 8$ perturbed continuations; the critic scores each rollout, and we apply EGGROLL to the generator’s interior projection kernels. The critic is trained from scratch and updated online alongside the generator on the W-GAN objective with $R _ { 1 }$ penalty $\scriptstyle ( \gamma = 1 0 )$ , averaged over the �=5 independent ensemble members, using Adam at a learning rate of $3 \times 1 0 ^ { - 4 }$ with $\beta _ { 1 } { = } 0 \ [ 1 5 ]$ . The $R _ { 1 }$ term is a zero-centered gradient penalty on real windows only, $\frac { \gamma } { 2 } \mathbb { E } _ { \boldsymbol { x } \sim p _ { \mathrm { r e a l } } } \left[ \| \nabla _ { \boldsymbol { x } } D ( \boldsymbol { x } ) \| ^ { 2 } \right]$ , which flattens the critic around the data manifold and keeps the online adversarial updates locally convergent [24]. Trained in this way, its decision boundary tracks the live discrepancies that free rollouts exhibit, including compounding errors that cross-entropy never penalizes.

EGGROLL. Since no gradient crosses the matching engine, the generator is updated with EGGROLL [31]: �=16 members (eight antithetic directions) of rank <sub>�</sub>=4 with <sub>�</sub>=0<sub>.</sub>003 (swept), evaluated on <sub>�</sub>=128 shared contexts. A direction’s fitness is its pessimistic ensemble score (mean std ), penalized by the KL trust region (below), centered-ranked per context, and averaged over contexts; the update is applied by Adam at $5 \times 1 0 ^ { - 4 }$ learning rate. For comparison, we swap the ES update for GRPO [32] under the same critic and contexts, with per-context groups (�=8), z-scored advantages, a single on-policy step, and a stronger trust region $( \lambda = 0 . 3 )$ , as raw advantages lack the rank transform’s bounding.

Generator updates. The pre-trained generator is frozen except for the interior projection kernels of the SSM blocks, which EGGROLL perturbs with rank-4 LoRA factors and updates in full. We intentionally exclude other parts of the SSM architecture to keep ES low-variance while preserving the pre-trained dynamics encoded in the SSM core, embeddings, and decoder head. On top of that, a Kullback-Leibler divergence trust region of $\lambda = 0 . 1$ is applied inside the fitness to prevent reward-hacking drift.

## 4.3 Evaluation

Metrics. Rollout realism is measured with LOB-Bench [27], the standard evaluation framework for generative message-level LOB models: real and generated continuations are mapped through a fixed battery of scoring functions covering microstructure statistics such as the bid–ask spread, returns, order-book volumes and imbalance, and message inter-arrival times. We aggregate the benchmark’s 21 unconditional scores into two summary numbers: WS-21, the mean over the 21 scores of the Wasserstein-1 distance between real and generated marginals, and Mean-�<sub>1</sub>, the mean of the corresponding $L _ { 1 }$ histogram distances. The $L _ { 1 }$ histogram distance reveals localized density mismatches, while the Wasserstein-1 distance responds to global distributional shape, including tails. Uncertainty is quantified with a paired window-level block bootstrap: whole evaluation windows are resampled with replacement (500 replicates), each replicate re-scores real and generated continuations from the same drawn windows, and model-versus-anchor deltas are formed replicate-wise from these shared draws, so window-composition noise cancels in the comparison. Day-clustered resampling over the 28 panel days yields consistent intervals.

![](images/e819432f26f3045e2b29666005a567ae6e0bd8755e9047561b843bd9906f2de0.jpg)  
Figure 3: Raw-message critic. Each 26-token message is decoded into 12 channels while absolute values and IDs are withheld as trivial tells. Channels are median-centered and scaled by 1.4826 MAD (median absolute deviation) to match a Gaussian distribution. Spectral-normalized TCN members, each behind its own fixed random projection, score the 500- message continuation and the generator receives the pessimistic ensemble score.

Selection. Any learned reward is eventually over-optimized [12], and ours is a live adversary: past a certain point the generator raises its critic score by exploiting the critic rather than by becoming more realistic, so the last checkpoint cannot be the deployed one. A threeseed pilot trained to 50 steps locates that turn (Section 5.1.1): realism improves monotonically through steps ${ \approx } 1 5 { - } 2 0 ,$ , then decays. We therefore train the headline seeds for 30 steps and score a checkpoint grid spaced every two steps. Each is scored ofline with WS-21 on 2,048 context windows (1,024 for comparative studies) drawn from the training timeframe, and the best is selected as the winner; the picks land at steps 10–20 (median 16). The criterion uses no critic signal by design, so a checkpoint cannot select itself through the same exploit that inflated its reward.

Sealed test. Selected checkpoints are scored once on 4,096 disjoint windows (2,048 windows for comparative studies) from a 28-tradingday panel of most recent unseen data (January–February 2026). Results are compared against the pre-trained model re-scored on the same windows.

Control. We compare against a selection-matched null in which training is replaced by drawing �=8 random LoRA perturbations of the base model through seed, matched to the deployed checkpoints in parameter norm, and passed the identical selection and sealedtest pipeline. This process bounds how much apparent improvement the pipeline can manufacture from selection bias.

## 5 Results

The optimized pipeline results are shown in Table 2: adversarial post-training improves held-out realism for every seed, with WS-21 falling by 17<sub>.</sub>6% against the base pre-trained model and $\mathrm { M e a n } { \cdot } L _ { 1 }$ by 17<sub>.</sub>0%; the paired 95% bootstrap interval excludes zero for the tenseed mean and for each seed individually (per-seed WS-21 range 11<sub>.</sub>3% to 23<sub>.</sub>9%), while the selection null ends above the anchor and the IsoFLOP control within noise of the pre-trained model.

Table 2: Sealed test at 4,096 held-out windows (28-day panel). Each Δ is paired against the frozen pre-trained generator on the same windows, with window-block bootstrap 95% CIs; the post-trained Δ is the mean over �=10 seeds.
<table><tr><td>Model</td><td>Δ WS-21 [95% CI] Δ Mean  $L _ { 1 }$  [95% CI]</td></tr><tr><td>Pre-Trained</td><td></td></tr><tr><td>Post-Trained</td><td> $- 1 7 . 6 \% [ - 1 9 . 7 , - 1 5 . 3 ]$   $- 1 7 . 0 \% [ - 1 8 . 9 , - 1 5 . 8 ]$ </td></tr><tr><td>Selection-null  $+ 5 . 3 \% [ + 2 . 5 , + 8 . 1 ]$ </td><td> $+ 7 . 8 \% [ + 5 . 9 , + 9 . 6 ]$ </td></tr><tr><td>IsoFLOP control</td><td> $+ 0 . 5 \% [ - 2 . 3 , + 3 . 4 ]$   $- 0 . 4 \% [ - 2 . 2 , + 1 . 4 ]$ </td></tr></table>

This adversarial pipeline consumes only about 50 M training <sup>tokens—a</sup> <sup>103</sup>× <sup>reduction</sup> <sup>from</sup> <sup>the</sup> ≈<sup>48</sup> <sup>B-token</sup> <sup>pre-training</sup> <sup>corpus.</sup> With 12 GPU-hours per seed, our pipeline shows that HFT foundation models can largely benefit from adversarial post-training as an inexpensive refinement stage.

The gains also reach the failure mode that motivates posttraining: error compounding over free-running rollouts. Measured position by position against the true continuation, the anchor’s divergence grows 3<sub>.</sub>1 over 500 generated messages; averaged over the ten sealed seeds, post-training slows the accumulation to 2<sub>.</sub>3 , accumulating 26% less excess divergence (paired day-bootstrap 95% CI 33%<sub>,</sub> 20% ), with every seed accumulating less than the anchor. Neither control reproduces the reduction—the selection-matched null accumulates 27% more than the anchor, while continued pre-training, at the IsoFLOP budget of the full experiment, accumulates at the anchor’s rate (Figure 4). Section 6.1 details the estimator, the controls, and their interpretation.

## 5.1 Training dynamics

5.1.1 Reward hacking. The per-seed training dynamics also reveal the need for external selection: with three seeds trained on 50 steps, all seeds improve monotonically to steps 15–20, reaching 16<sub>.</sub>4% to 21<sub>.</sub>7% against the anchor on the selection windows, then decay as the generator begins to exploit the critic—seed 1 returns past the anchor ( 25<sub>.</sub>4%) by step 50 and seeds 0 and 2 to roughly anchor level—and the WS-21 grid selection catches the turn in all three seeds. The softer sealed-test deltas ( 17<sub>.</sub>5% mean, against 19<sub>.</sub>4% at selection) reflect ordinary selection optimism; the sign is preserved out of sample.

The critic tells the same story from the other side. Its held-out real-versus-fake discrimination starts strong (AUC 0<sub>.</sub>62 at the first step) and declines steadily to chance level ( 0<sub>.</sub>51, calibration cross-entropy peaking near 0<sub>.</sub>80 nats) at steps 20–25: the generator genuinely closes the gap the critic can measure, exactly where grid selection places the turn. Past the turn the equilibrium destabilizes: discrimination recovers to AUC 0<sub>.</sub>64 and the cross-entropy drops below the ln 2 chance level for the only time in training, the online critic cleanly re-separating an adversary whose continued reward ascent has decoupled from realism. The ten 30-step headline seeds <sup>retrace</sup> <sup>the</sup> <sup>approach</sup> <sup>leg</sup> <sup>of</sup> <sup>the</sup> <sup>same</sup> <sup>arc</sup> <sup>(mean</sup> <sup>AUC</sup> <sup>0</sup>.<sup>62</sup> → <sup>0</sup>.<sup>52</sup> by step 20).

5.1.2 Loss behavior. Adversarial gains cost little likelihood. Teacher-forced next-token cross-entropy on 512 held-out real windows rises by 0<sub>.</sub>10% (95% CI 0<sub>.</sub>07%<sub>,</sub> 0<sub>.</sub>12% ) by step 20 across the ten headline seeds, where it stays through step 30. Nor is the concession a uniform drift: per-seed values at step 30 span 0<sub>.</sub>38% to 0<sub>.</sub>41%, while the KL divergence to the anchor grows monotonically in every seed $( 2 . 0  6 . 2 { \times } 1 0 ^ { - 3 }$ between steps 10 and 30). The update thus moves steadily away from the pre-trained model while leaving its likelihood almost untouched.

## 5.2 Compute eficiency

All of our results are obtained with populations of just �=16 members (eight antithetic direction pairs), five orders of magnitude below the 1M-member populations EGGROLL was introduced with for from-scratch pre-training [31]. The regimes difer fundamentally. Post-training explores the immediate neighborhood of a strong anchor. The scarce resource is instead the reliability of each direction’s score, since market rollouts are a noisy fitness signal. We therefore average each direction’s fitness over $Q { = } 1 2 8$ shared contexts, spending the rollout budget on update stability rather than population size. At a fixed budget $G { \times } Q ,$ , the outcome should then be governed by the total rather than the direction count, and small populations should lose nothing, as Section 5.2.1 confirms experimentally.

5.2.1 Compute-optimalfrontier. We ablate the split between direction count � and contexts per direction <sub>�</sub> at a fixed budget of 2<sub>,</sub>048 rollouts per step, three seeds per arm, holding everything else constant. On the sealed test, diferent allocations all sit well inside the within-arm seed spread. The split shows instead at selection, where larger-� arms tend to select earlier checkpoints (median picked step 20 at �=16 versus 15 at �=32 and �=64)—sharper per-step gradient estimates climb the critic’s proxy faster and approach the reward-hacking window sooner, while the co-trained critic receives the same number of updates to keep pace.

Restoring <sub>�</sub>=128 at �=64 quadruples the budget to 8<sub>,</sub>192 rollouts per step and buys nothing: its seed lands at 13<sub>.</sub>5% WS-21 and 15<sub>.</sub>8% Mean-�<sub>1</sub>, below the WS-21 seed range of the fixed-budget arms. Two conclusions follow. At a fixed budget, the sealed outcome is insensitive to how rollouts are split between directions and contexts. Furthermore, growing the population without shrinking the contexts per direction (to more closely resemble ES pre-training) buys no additional sealed realism either: against a critic trained online, the binding constraint is the critic’s robustness, not population size.

5.2.2 Benefits over pre-training. The improvement is not a tokenbudget efect that more pre-training could reproduce. As a control, we continue pre-training the anchor on the same unseen corpus under its original recipe, with compute matched to a single posttraining seed’s full experiment and a fresh cosine learning-rate schedule spanning that budget. Fully annealed, the control lands within noise of the pre-trained generator ( 0<sub>.</sub>5% WS-21 and 0<sub>.</sub>4% Mean $^ { - L _ { 1 } , }$ paired CIs spanning zero; Table 2). Additional likelihood training on the same corpus therefore reproduces none of the adver sarial gain, revealing the value of the critic’s sequence-level signal beyond data volume or extra compute. This benefit is also confirmed over the compounding error analysis of Section 6.1, where extra pre-training does not yield improvements over the pre-trained generator.

## 5.3 Critic and optimizer ablations

Holding the training loop fixed, we ablate the two central design choices of the pipeline: what the critic observes, and how its scores become generator updates.

5.3.1 Input features. The raw-message critic of Section 4 is the endpoint of an ablation over the critic’s input. Reusing the generator itself, scoring rollouts through the frozen backbone’s pooled hidden state with a shallow head, appears free but is structurally exploitable: the adversarial reward turns anti-aligned with heldout realism (correlation 0<sub>.</sub>5–0<sub>.</sub>8 between reward and LOB-Bench error), and two of three seeds decay back to the anchor. Hand designed per-step microstructure descriptors, scored by the same spectral-normalized TCN ensemble, restore a reliable signal but restrict the critic to discrepancies the descriptors can express. The losslessly decoded message stream removes both failure modes, leaving representation learning to the adversarial objective.

Table 3: Critic-input ablation. Single-program sealed test at 2,048 held-out windows (28-day panel), all rows sharing one anchor; Δ = mean paired change vs. that anchor across three seeds, with the per-seed range in brackets.
<table><tr><td>Critic input</td><td>ΔWS-21 ↓</td><td>∆Mean  $L _ { 1 } \downarrow$ </td></tr><tr><td>Generator backbone</td><td>-21.4% [-11.5,-28.2]-18.7% [−10.0,-23.7]</td><td></td></tr><tr><td>Microstructure descriptors -8.5% [-2.4, –11.6] -6.7% [+0.2, -13.3]</td><td></td><td></td></tr><tr><td>Raw messages</td><td>-17.5% [-15.7, -19.7] -18.9% [−17.2,-21.1]</td><td></td></tr></table>

Table 3 quantifies the comparison under the otherwise-identical protocol: three seeds per arm, $G { \times } Q = 1 6 { \times } 1 2 8 .$ , 50 steps, unseencorpus contexts, identical selection, and one single-program sealed test in which every row is generated under common random numbers and scored against the same anchor. The raw-message critic roughly doubles the descriptor critic’s sealed efect, improving WS-21 by 17<sub>.</sub>0 15<sub>.</sub>7 19<sub>.</sub>7% per seed (Mean-�<sub>1</sub> 17<sub>.</sub>2 18<sub>.</sub>4 21<sub>.</sub>1%) against a descriptor-critic mean of 8<sub>.</sub>5% (6<sub>.</sub>7%): every raw-critic seed beats every descriptor-critic seed on both metrics, and the descriptor arm’s weakest seed is statistically indistinguishable from the anchor ( 2<sub>.</sub>4% WS-21, paired CI spanning zero). The backbone critic’s selected checkpoints score well on average but are the least reliable of the three arms, spanning 11<sub>.</sub>5% to 28<sub>.</sub>2% across seeds—consistent with an exploitable critic rather than the reward remaining a faithful realism signal throughout training.

5.3.2 Optimization method. The critic’s signal is not specific to evolution strategies. As a comparison, we swap the ES update for GRPO under an otherwise-identical pipeline and repeat the full headline protocol: ten seeds, identical in-training-timeframe selection over the checkpoint grid, and one sealed scoring at 4<sub>,</sub>096 heldout windows with a norm-matched selection null of its own. Across ten seeds, GRPO shifts WS-21 by 16<sub>.</sub>0% and Mean- ${ \bf \cdot } L _ { 1 }$ by 17<sub>.</sub>3% against the same anchor (its selection null: 1<sub>.</sub>8%, not significant), compared with 17<sub>.</sub>6% and 17<sub>.</sub>0% for EGGROLL: the adversarial reward is optimizer-agnostic. We retain EGGROLL for the headline pipeline for two reasons. First, GRPO’s selection curves are jagged— sharp step-to-step swings rather than EGGROLL’s smooth rise into the hacking window—making results less predictable and harder to analyze. Second, the per-position decomposition of Section 6.1 shows contrasting results. GRPO slows accumulation even more than EGGROLL, but buys it with a large level ofset: its marginals start 75% above the anchor’s in the first 25-message bin, and the slower growth only wins that ofset back, leaving its total divergence at the 500-message horizon almost indistinguishable from the anchor ( 1<sub>.</sub>0%) and above EGGROLL’s total divergence ( 11<sub>.</sub>3%).

## 6 Discussion

## 6.1 Compounding error

We test the exposure-bias premise directly by measuring divergence as a function of continuation position. Over 4<sub>,</sub>096 held-out context windows on the 28-day test panel, we compare the true continuation against a free-running rollout from the same context, one rollout per window, all rows generated in a single common-randomnumbers program. At each token position we form the empirical marginal across windows over that position’s valid sub-alphabet and compute KL true gen , normalized by the entropy of the true marginal to make positions of diferent dificulty comparable, and averaged over the sampled phases of the new-message block (two near-deterministic phases with vanishing entropy are excluded). Figure 4 reports the accumulation view: each curve minus its own first 25-message bin, so every model starts at zero and its endpoint is the excess divergence accumulated over the rollout. Horizonindependent level ofsets, including the estimator’s finite-sample bias (a real-versus-real control of the same statistic is flat across the rollout), drop out of the comparison.

![](images/d1306ebcc0ec1c9107c7f57aba41ebe82687adf378e24ec67bef4c07ae06f64e.jpg)  
Figure 4: KL true gen � true accumulated since each curve’s first 25-message bin. Post-training accumulates 26% less excess divergence than the anchor, the selection null 27% more, continued pre-training the same. Shaded: 95% dayclustered bootstrap bands.

The pre-trained model’s divergence grows 3<sub>.</sub>1 from the first to the last 25-message bin, direct evidence that its errors compound over free-running generation. Post-training reduces exactly this accumulation: the ten-seed mean accumulates 26% less excess divergence than the anchor over the 500-message horizon (paired day-bootstrap, replicate-wise diferences on a draw matrix shared across rows; 95% CI 33%<sub>,</sub> 20% ). The controls locate the source of the reduction: the selection-luck null accumulates 27% more than the anchor (CI 15%<sub>,</sub> 37% ), and the continued-pre-training control accumulates at the anchor’s rate ( 3<sub>.</sub>9%, CI 6%<sub>,</sub> 14% ). Neither selection pressure nor additional likelihood training slows accumulation; the critic’s sequence-level signal does.

## 6.2 Market realism

6.2.1 Absolute value variations. Beyond the aggregate WS-21, per feature response follows a regular pattern: the larger a feature’s error in the base pre-trained model, the more post-training improves it. Fitting the 10-seed average relative change against the anchor’s per-feature Wasserstein level with a saturating power law $y = y _ { 0 } +$ $A x ^ { p }$ across the 21 features gives $R ^ { 2 } = 0 . 8 6$ and Spearman $\rho = - 0 . 7 6$ (Figure 5). The same compression holds across every critic variant of Section 5.3, indicating it is a property of the adversarial signal itself rather than of any particular critic input.

The mechanism follows from the objective rather than from any feature. The critic scores whole rollouts and carries no per-feature term; its discriminative power concentrates wherever the anchor deviates most from the data, and that is where the gradient-free update is steered hardest. As those deviations shrink they stop separating real from generated and the pressure moves elsewhere— a saturating dynamic, which is why a saturating power law and not a linear trend describes the response. Because nothing in the objective identifies a feature, the pattern is a consequence of adversarial pressure seeking the cheapest separation, not feature-specific tuning.

![](images/e0d31838ff6ac9573c5f6d100090cdbcba71c950bfba3d339327076afb317407.jpg)  
Figure 5: Per-feature error shift at test-time, averaged over the 10 post-training seeds. The saturating power-law fit $( R ^ { 2 } =$ 0.86, Spearman $\rho = - 0 . 7 6 )$ shows the largest anchor errors improving most; features near the evaluation noise floor (left) swing widely in relative terms, and spread (annotated) is the main outlier.

Two departures are informative. Spread—despite the 9th-largest anchor error—worsens by 34%. Unlike the compressed features it is not a marginal of the message stream but a joint property of the standing book, set by the interaction of placements and cancellations at the touch: the critic can improve each event type’s marginal realism while the book-level equilibrium it induces drifts. It is also self-reinforcing under free running, since a slightly wide spread changes which orders are marketable, which changes the next message, which widens it further. Since the critic scores the message stream but never observes the standing book, this drift is only indirectly visible to the reward; the exchangeable reward module of Section 7.1 ofers a potential countermeasure, retargeting a follow-up post-training pass at book-level properties such as the spread itself. Conversely, features near the noise floor worsen only in relative terms: their anchor errors are so small that tiny absolute changes read as large percentages, and their absolute errors remain small after post-training.

6.2.2 Beyond zero-sum shifts. Recent work also characterizes the neighborhood of large pretrained models as a thicket of directions that improve individual metrics by chance [11], so checkpoint selection over a training run could in principle manufacture apparent gains from undirected drift alone. Here, it could be possible that the generator moves error between statistics—improving whatever the critic currently prices at the expense of what it ignores—without improving general performance. Our selection-null control bounds this channel by passing an equivalent number of random LoRA perturbations through the identical selection pipeline: it ends significantly above the anchor, so selection alone manufactures no gain here—likely because neural thickets are observed at far larger scales (beyond 1<sub>.</sub>5 B parameters) than our 78<sub>.</sub>5 M-parameter generator.

## 6.3 Downstream performance

Our results show that post-training improves market realism and slows error accumulation over free-running rollouts. We therefore ask if that gain also shows in downstream performance or if that improvement is compensated by worse predictive capabilities. We test this on mid-price direction: over 2<sub>,</sub>048 held-out windows of the same 28-day panel, each model free-runs from a common context and we classify the sign of its mid-price change at horizons of 100 to 1<sub>,</sub>000 messages against the truth (three classes: up, flat, down). Deltas are paired against the anchor on identical windows, with a day-clustered bootstrap draw matrix shared across rows and horizons.

Table 4: Downstream mid-price direction at 2,048 held-out windows (28-day panel). Δ is the mean change in three-class directional accuracy across �=10 seeds relative to the frozen pre-trained anchor, with paired day-clustered bootstrap 95% CIs. Accuracy is preserved at every horizon: no interval separates from zero.
<table><tr><td colspan="2">Horizon (messages) ∆ Directional accuracy [95% CI]</td></tr><tr><td>100</td><td>+0.011 [-0.008, +0.031]</td></tr><tr><td>250</td><td>+0.012 [-0.007, +0.028]</td></tr><tr><td>500</td><td>-0.004 [-0.024, +0.016]</td></tr><tr><td>1,000</td><td>+0.009 [-0.010, +0.027]</td></tr></table>

Directional accuracy is preserved at every horizon (Table 4). This is consistent with the two quantities living on diferent axes: realism concerns the distribution of order flow, while direction concerns the conditional mean of a near-martingale. Eficient prices leave almost no predictable component to sharpen (the anchor’s own correlation with the true move decays to zero within 400 messages), so reshaping the flow distribution moves realism without disturbing direction.

Preservation is the informative outcome. No term in the critic or the ES reward targets directional prediction, so nothing protects this capability during the update: it is free to degrade, and does not. The compounding-error reduction is therefore not a zero-sum reallocation between capabilities. Rollouts stay closer to the data for longer (Section 6.1) while downstream predictive performance carries through intact, and that combination is what makes the refinement usable in practice.

## 7 Conclusion

We introduced an adversarial post-training framework for foundation models in high-frequency trading: a Wasserstein critic, trained online on rollouts executed through the true, non-diferentiable matching engine, steers gradient-free updates, built from low-rank perturbations, of a pre-trained generator under a KL trust region. For under 50 M training tokens and 12 GPU-hours per seed, three orders of magnitude below the pre-training token budget, posttraining improves sealed held-out realism by 17<sub>.</sub>6% (WS-21) and 17<sub>.</sub>0% (Mean-�<sub>1</sub>), every seed individually significant, and reduces the excess divergence accumulated over free-running 500-message rollouts by 26%. Selection-matched and compute-matched controls reproduce neither gain, isolating the critic’s sequence-level signal as the active ingredient, and the improvement costs nothing downstream: directional accuracy is preserved at every horizon. Alongside the headline result, we characterized the reward-hacking window that bounds productive adversarial training, the selection protocol that disciplines it, and distilled ablations over critic inputs, optimizers, and population geometry into reusable design guidelines.

## 7.1 Future work

Because nothing in the pipeline is specific to aggregate realism—the critic is an exchangeable reward module—our framework transfers to any property computable from free rollouts, inside or outside the high-frequency literature. Post-training thus forms its own stage between pre-training and fine-tuning: pre-train once for global market structure, post-train to sharpen a targeted property, and fine-tune the result into a specialized system such as an agentic trading framework or an information-decoding pipeline. The most consequential target is market-impact fidelity: empirical impact obeys tight regularities such as the square-root law [1, 5], and a simulator that reproduces the counterfactual price response to injected orders is the missing prerequisite for training and evaluating trading agents in realistic environments [6, 10]. Another idea could be to adapt our method towards diferent objectives: a GAN discriminator could, for example, post-train a variant of our model in order to perform market making, learning to score policies based on expected profits.

Two further directions concern the pipeline itself. First, all rollouts here span 500 messages; whether longer training horizons slow error accumulation further remains open. Second, the trained critic is a realism score in its own right: used in isolation, it could rank generators, flag anomalous order flow, or act as a frozen reward, enabling a controlled study of reward hacking apart from the co-adaptation of Section 5.1.1.

## Acknowledgments

This work was supported by the Engineering and Physical Sciences Research Council [grant number UKRI5149] and other sponsors of BOLD. The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR). Isambard-AI is operated by the University of Bristol and is funded by the UK Government’s Department for Business, Innovation, Science and Trade (DBIST) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023] [23].

AB gratefully acknowledges the support of the Anne Dias Griffin and Kenneth C. Grifin Harvard University Fellowship. VM gratefully acknowledges funding from the Department of Computer Science at the University of Oxford. AC acknowledges funding from a UKRI AI World Leading Researcher Fellowship (grant EP/W002949/1). AC and JF acknowledge funding from a JPMC Faculty Research Award.

## References

[1] Robert Almgren, Chee Thum, Emmanuel Hauptmann, and Hong Li. 2005. Direct Estimation of Equity Market Impact. Risk 18, 7 (2005), 58–62.

[2] Martin Arjovsky, Soumith Chintala, and Léon Bottou. 2017. Wasserstein Generative Adversarial Networks. In Proceedings ofthe 34th International Conference on Machine Learning, Vol. 70. 214–223.

[3] Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. 2015. Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks. In Advances in Neural Information Processing Systems 28.

[4] Leonardo Berti, Bardh Prenkaj, and Paola Velardi. 2025. TRADES: Generating Realistic Market Simulations with Difusion Models. In ECAI2025 – 28th European Conference on Artificial Intelligence. IOS Press, 3703–3710. doi:10.3233/FAIA251249

[5] Jean-Philippe Bouchaud, J. Doyne Farmer, and Fabrizio Lillo. 2009. How Markets Slowly Digest Changes in Supply and Demand. In Handbook of Financial Markets: Dynamics and Evolution. Elsevier, 57–160.

[6] David Byrd, Maria Hybinette, and Tucker Hybinette Balch. 2020. ABIDES: Towards High-Fidelity Multi-Agent Market Simulation. In Proceedings ofthe 2020 ACM SIGSIM Conference on Principles of Advanced Discrete Simulation (SIGSIM-PADS ’20). ACM, 11–22. doi:10.1145/3384441.3395986

[7] Andrea Coletta, Joseph Jerome, Rahul Savani, and Svitlana Vyetrenko. 2023. Conditional Generators for Limit Order Book Environments: Explainability, Chal lenges, and Robustness. In Proceedings ofthe Fourth ACM International Conference on AI in Finance (ICAIF ’23). ACM, 27–35. doi:10.1145/3604237.3626854

[8] Andrea Coletta, Aymeric Moulin, Svitlana Vyetrenko, and Tucker Balch. 2022. Learning to Simulate Realistic Limit Order Book Markets from Data as a World Agent. In Proceedings ofthe Third ACM International Conference on AI in Finance (ICAIF ’22). ACM, 428–436. doi:10.1145/3533271.3561753

[9] Andrea Coletta, Matteo Prata, Michele Conti, Emanuele Mercanti, Novella Bartolini, Aymeric Moulin, Svitlana Vyetrenko, and Tucker Balch. 2021. Towards Realistic Market Simulations: A Generative Adversarial Networks Approach. In Proceedings ofthe Second ACM International Conference on AI in Finance (ICAIF ’21). ACM, 1–9. doi:10.1145/3490354.3494411

[10] Sascha Frey, Kang Li, Peer Nagy, Silvia Sapora, Chris Lu, Stefan Zohren, Jakob Foerster, and Anisoara Calinescu. 2023. JAX-LOB: A GPU-Accelerated Limit Order Book Simulator to Unlock Large Scale Reinforcement Learning for Trading. In Proceedings ofthe Fourth ACM International Conference on AI in Finance (ICAIF ’23). ACM, 583–591. doi:10.1145/3604237.3626880

[11] Yulu Gan and Phillip Isola. 2026. Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights. arXiv preprint arXiv:2603.12228 (2026).

[12] Leo Gao, John Schulman, and Jacob Hilton. 2023. Scaling Laws for Reward Model Overoptimization. In Proceedings ofthe 40th International Conference on Machine Learning, Vol. 202. 10835–10866.

[13] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. 2014. Generative Adversarial Nets. In Advances in Neural Information Processing Systems 27.

[14] Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, and Aaron Courville. 2017. Improved Training of Wasserstein GANs. In Advances in Neural Information Processing Systems 30.

[15] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. 2017. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In Advances in Neural Information Processing Systems 30.

[16] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.

[17] Ruihong Huang and Tomas Polak. 2011. LOBSTER: Limit Order Book Reconstruction System. Technical Report. Available at SSRN 1977207.

[18] Hanna Hultin, Henrik Hult, Alexandre Proutiere, Samuel Samama, and Ala Tarighati. 2023. A Generative Model of a Limit Order Book Using Recurrent Neural Networks. Quantitative Finance 23, 6 (2023), 931–958. doi:10.1080/14697688. 2023.2205583

[19] Aakash Lahoti, Kevin Y. Li, Berlin Chen, Caitlin Wang, Aviv Bick, J. Zico Kolter, Tri Dao, and Albert Gu. 2026. Mamba-3: Improved Sequence Modeling using State Space Principles. In The Fourteenth International Conference on Learning Representations (ICLR).

[20] Junjie Li, Yang Liu, Weiqing Liu, Shikai Fang, Lewen Wang, Chang Xu, and Jiang Bian. 2025. MarS: a Financial Market Simulation Engine Powered by Generative Foundation Model. In The Thirteenth International Conference on Learning Representations (ICLR).

[21] Junyi Li, Xintong Wang, Yaoyang Lin, Arunesh Sinha, and Michael P. Wellman. 2020. Generating Realistic Stock Market Order Streams. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 34. 727–734. doi:10.1609/aaai.v34i01.5415

[22] Shanchuan Lin, Xin Xia, Yuxi Ren, Ceyuan Yang, Xuefeng Xiao, and Lu Jiang. 2025. Difusion Adversarial Post-Training for One-Step Video Generation. arXiv preprint arXiv:2501.08316 (2025). ICML 2025.

[23] Simon McIntosh-Smith, SadafR Alam, and Christopher Woods. 2024. Isambard-AI: a leadership class supercomputer optimised specifically for Artificial Intelligence. arXiv:2410.11199 [cs.DC] https://arxiv.org/abs/2410.11199

[24] Lars Mescheder, Andreas Geiger, and Sebastian Nowozin. 2018. Which Training Methods for GANs do actually Converge?. In Proceedings ofthe 35th International Conference on Machine Learning, Vol. 80. 3481–3490.

[25] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. 2018. Spectral Normalization for Generative Adversarial Networks. In International Conference on Learning Representations.

[26] Peer Nagy, Sascha Frey, Silvia Sapora, Kang Li, Anisoara Calinescu, Stefan Zohren, and Jakob Foerster. 2023. Generative AI for End-to-End Limit Order Book Mod elling: A Token-Level Autoregressive Generative Model of Message Flow Using a Deep State Space Network. In Proceedings ofthe Fourth ACM International Conference on AI in Finance (ICAIF ’23). ACM, 91–99. doi:10.1145/3604237.3626898

[27] Peer Nagy, Sascha Yves Frey, Kang Li, Bidipta Sarkar, Svitlana Vyetrenko, Stefan Zohren, Ani Calinescu, and Jakob Nicolaus Foerster. 2025. LOB-Bench: Bench marking Generative AI for Finance – an Application to Limit Order Book Data. In Proceedings ofthe 42nd International Conference on Machine Learning (ICML), Vol. 267. PMLR, 45437–45460.

[28] Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Yinggan Xu, Roberto Dailey, Elliot Meyerson, Babak Hodjat, and Risto Miikkulainen. 2025. Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning. arXiv preprint arXiv:2509.24372 (2025).

[29] Marc’Aurelio Ranzato, Sumit Chopra, Michael Auli, and Wojciech Zaremba. 2016. Sequence Level Training with Recurrent Neural Networks. In International Conference on Learning Representations.

[30] Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, and Ilya Sutskever. 2017. Evolution Strategies as a Scalable Alternative to Reinforcement Learning. arXiv preprint arXiv:1703.03864 (2017).

[31] Bidipta Sarkar, Mattie Fellows, Juan Agustin Duque, Alistair Letcher, Antonio León Villares, Anya Sims, Clarisse Wibault, Dmitry Samsonov, Dylan Cope, Jarek Liesen, Kang Li, Lukas Seier, Theo Wolf, Uljad Berdica, Valentin Mohl, Alexander David Goldie, Aaron Courville, Karin Sevegnani, Shimon Whiteson, and Jakob Nicolaus Foerster. 2025. Evolution Strategies at the Hyperscale. arXiv preprint arXiv:2511.16652 (2025).

[32] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300 (2024).

[33] James C. Spall. 1992. Multivariate Stochastic Approximation Using a Simultaneous Perturbation Gradient Approximation. IEEE Trans. Automat. Control 37, 3 (1992), 332–341.

[34] Aaron Wheeler and Jefrey D. Varner. 2024. MarketGPT: Developing a Pre-trained transformer (GPT) for Modeling Financial Time Series. arXiv:2411.16585

[35] Daan Wierstra, Tom Schaul, Tobias Glasmachers, Yi Sun, Jan Peters, and Jürgen Schmidhuber. 2014. Natural Evolution Strategies. Journal ofMachine Learning Research 15, 27 (2014), 949–980.

[36] Lantao Yu, Weinan Zhang, Jun Wang, and Yong Yu. 2017. SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient. In Proceedings ofthe Thirty-First AAAI Conference on Artificial Intelligence. 2852–2858.