# DashVMC

# Real-Time Discrete World Model Control in Geometry Dash

Contributors

Florent Tariolle<sup>1</sup>, Florian Yger<sup>1,2</sup>

<sup>1</sup>INSA Rouen Normandy <sup>2</sup>LITIS

World-model agents are usually evaluated in simulators that can wait for the policy; live games impose the opposite constraint, requiring capture, prediction, and action before the next frame. We present DashVMC, which learns a compact, action-conditioned world model from approximately two hours of recorded Geometry Dash gameplay. To test whether the learned dynamics are actionable, a controller is initialized by behavioural cloning (BC) and refined with Proximal Policy Optimization (PPO) entirely in frozen-model rollouts, without further interaction with the live game. Across three controller seeds, the refined policies survive longer than their BC initializations on all three oficial levels and a held-out community layout. At deployment, the baseline skips visual generation and sustains a 60-Hz captureto-action loop on a consumer GPU. Action-conditioned continuations and rollout diagnostics show that the model remains useful for control despite imperfect long-horizon fidelity.

Project page: tariolle.github.io/dash-vmc Code: github.com/Tariolle/dash-vmc Correspondence: florent.tariolle@insa-rouen.fr

## 1 Introduction

A world model learns transition dynamics $p _ { \theta } ( s _ { t + 1 } \mid s _ { t } , a _ { t } )$ and can be rolled forward so that an agent practices without further environment interaction (LeCun, 2022; Ha and Schmidhuber, 2018). DashVMC asks whether such practice can improve a policy that must then act in a live, non-paused game. It predicts a 64-token next grid in parallel for inexpensive imagined rollouts. The baseline deployment skips generation and reuses the transformer’s temporal summary; a matched ablation tests whether the transformer must remain in the live loop.

Most world-model agents are evaluated in synchronous environments that wait for the policy. A live, non paused application reverses that contract: screen capture, perception, inference, and action dispatch must finish before the action becomes stale.

Geometry Dash is a small but unforgiving test of this contract. The avatar moves continuously, the action space is binary, and a mistimed jump causes immediate death. Levels are deterministic, but the agent observes only captured pixels and acts through keyboard events; it does not receive emulator state during control. The apparent simplicity therefore isolates a demanding question: can an agent learn when to act from pixels, improve without touching the live game, and still react within narrow timing windows on modest hardware?

DashVMC follows the perception–dynamics–control organization of World Models (Ha and Schmidhuber, 2018): it compresses each frame into a discrete grid, learns action-conditioned grid dynamics, initializes a lightweight action model with BC, and refines it with PPO (Schulman et al., 2017) in the frozen model.

Our contributions are:

1. a compact discrete, action-conditioned world model learned from approximately two hours of ofline gameplay, whose rollouts support policy refinement without further live-game interaction;

2. a decoder-free baseline that sustains a 60-Hz capture-to-action loop on a consumer GPU, together with a spatial-only variant that removes the transformer at inference without a consistent live-performance deficit; and

3. evidence that the learned dynamics are actionable: controllers refined only in frozen-model rollouts improve over their exact BC parents on all three oficial levels across three seeds and on a held-out layout.

Diagnostics cover action sensitivity, rollout fidelity, controller input, dynamics losses, and latency.

## 2 Related work

World models and imagination. World Models demonstrated that a compact controller optimized in learned dreams can transfer to the real environment (Ha and Schmidhuber, 2018). IRIS combines a discrete autoencoder with an autoregressive transformer for data-eficient Atari learning (Micheli et al., 2023); ∆-IRIS reduces sequence cost with discrete delta tokens and continuous context tokens (Micheli et al., 2024); DIA-MOND uses difusion for imagination-trained control (Alonso et al., 2024); and DreamerV3 scales categorical recurrent latents across domains (Hafner et al., 2025). TWISTER adds action-conditioned CPC to transformer dynamics (Burchi and Timofte, 2025), which DashVMC adopts alongside parallel spatial prediction. Earlier screenshot-based Geometry Dash control used DQN and imitation under a fabricated timestep (Li and Raferty, 2017); here the game clock continues during capture and inference.

FSQ tokenization and policy initialization. FSQ replaces learned vector-quantization codebooks with bounded scalar levels, avoiding commitment losses and codebook collapse while exposing an integer coordinate lattice (Mentzer et al., 2024); we use the improved iFSQ bounding function (Lin et al., 2026). Demonstrationbased RL can reduce costly early exploration (Hester et al., 2018; Baker et al., 2022); here recorded actions initialize the controller, while the retained policy is selected only after imagined PPO refinement. Deterministic Sobel preprocessing removes appearance detail before tokenization, providing a compact task-specific alternative to richer continuous or generative representations.

## 3 DashVMC

DashVMC is organized around a simple loop: recorded play trains a world model, the action model is refined in rollouts from the frozen dynamics model, and the resulting controller is returned to the live game. The visual encoder, dynamics model, and action model are trained in sequence and then frozen for deployment. Figure 1 shows how the same learned state supports both imagined practice and live reaction.

## 3.1 Visual tokenization

Each RGB capture is cropped, converted to grayscale, filtered with Sobel edges, and resized to 64 × 64. A convolutional autoencoder maps the result through three stride-2 stages to a four-channel 8 × 8 latent map. Each spatial vector is independently quantized with iFSQ levels [8, 5, 5, 5], using the bounding function 2 sigmoid(1.6z) − 1, and yields 64 token IDs from 1000 implicit codes (Mentzer et al., 2024; Lin et al., 2026). The tokenizer is trained on adjacent frames with reconstruction, temporal-slowness, and latent-uniformity objectives (Xia et al., 2025). Once selected, its encoder is frozen; its decoder is used to visualize dreams but is not required by the action model.

(a) PPO policy improvement in imagination  
![](images/763f916a912c9a859d6e0ee0323c772c6d8b521a7fcbd5bc3bb378a5f047fe23.jpg)

![](images/a052b585dd5d8d7d7dedceaedef14ab951edf5bf4672994a6cac63ff597f28ac.jpg)  
Figure 1 Learning in imagination and acting in the live game. (a) A recorded context seeds the frozen world model; the action model chooses what to do, the world model predicts the consequence, and PPO updates only the action model. (b) In the baseline live path, the encoder and temporal context provide the state used for action selection, while visual generation is skipped.

## 3.2 Action-conditioned dynamics

The dynamics model is an eight-layer, width-384 transformer that receives four recent frame–action pairs. Each frame block contains 64 visual tokens and one alive/death status token, with action tokens inserted between blocks; a final masked block requests the next frame. Attention is bidirectional within a frame and causal across frame/action blocks, so all next-frame positions are predicted in one forward pass without seeing their targets. An explicit action token separates consecutive frames, so the model can answer the question central to policy learning: what changes if the action model jumps rather than idles? It predicts all 64 codes of the next grid in parallel, together with an alive/death score, and exposes a temporal summary h , taken from the most recent action-token position, to the action model. Parallel prediction keeps imagined steps inexpensive.

Training combines next-grid prediction with action-conditional contrastive predictive coding of weight 0.1 (Burchi and Timofte, 2025; van den Oord et al., 2018), context corruption, and an auxiliary death objective. For context corruption, 5% of context tokens are replaced uniformly and another 5% with adjacent FSQ codes. The token loss uses focal modulation (Lin et al., 2017) and structured label smoothing (SLS): for target lattice coordinate $\mathbf { c } _ { i } ,$ smoothing mass $\varepsilon = 0 . 1$ is distributed over non-target codes proportionally to $\exp ( - \Vert \mathbf { c } _ { i } - \mathbf { c } _ { j } \Vert _ { 2 } ^ { 2 } / 2 \sigma ^ { 2 } )$ , with $\sigma \ : = \ : 0 . 9$ Focal modulation is applied once to the aggregate soft-target cross-entropy, and the output projection is tied to the token embedding.

## 3.3 Controller and latent policy improvement

The 45,546-parameter actor–critic embeds the current token grid, processes it with two convolutions and max pooling, and concatenates the resulting 256-dimensional feature with the dynamics model’s temporal summary h<sub>t</sub>. Separate heads predict jump probability, value, and an eight-action auxiliary output covering the current and next seven actions. The matched spatial-only ablation removes $\mathbf { h } _ { t }$ from BC, PPO, and deployment, leaving a 41,706-parameter action model; the world model still generates its PPO rollouts. BC first reproduces recorded action timing; PPO then departs from imitation through repeated imagined consequences.

Each PPO iteration generates 512 imagined episodes of at most 45 steps from recorded four-frame contexts, with greedy next-grid predictions and sampled Bernoulli actions. The first four generated transitions refresh the context but are excluded because the recorded approach may make collision unavoidable; the remainder covers visible obstacles before invented geometry dominates. Rollouts terminate when the death logit exceeds the alive logit; because death frames are oversampled 5×, this is an operational boundary rather than a calibrated probability. Reward is +1 per survived step and −0.25 per jump, discouraging a local optimum in which an early jump merely delays collision while airborne and earns extra survival reward without clearing the obstacle. We use Generalized Advantage Estimation (GAE) (Schulman et al., 2015) with $\gamma = 0 . 9 9 5$ $\lambda = 0 . 9 5$ , PPO clip 0.2, and auxiliary-action loss weight 0.1.

## 3.4 Reactive deployment path

Training generates consequences; baseline live control only interprets recent observations. Desktop Duplication captures only the 1032 × 1032 model crop; CPU grayscale conversion followed by CUDA Sobel filtering and area resizing produces the 64 × 64 observation. Each frame is encoded and appended to a GPU-resident context, and the transformer runs context prefill only to produce h . The live context marks every observation alive; process-memory death detection only terminates and restarts an attempt outside the controller. The next-grid predictor and image decoder are skipped, reducing the deployed path to 15.57 million parameters; the encoder uses torch.compile, and transformer context encoding uses a CUDA Graph after warm-up. The accelerated preprocessing path was checked against the recorded-data pipeline on synthetic and fresh live cap tures and produced byte-identical observations. Spatial-only deployment loads only the frozen encoder and action model; both paths are reactive controllers whose policy improvement occurred during ofline imagined training.

## 4 Experimental setup

Data and training. Before augmentation, the corpus contains 4,264 action-labelled gameplay episodes, or 251,963 frames and approximately 2.33 hours at the nominal 30-FPS capture cadence. It includes deliberate failures, oficial Levels 1–7, several community levels, and 36 expert trajectories or segments totaling less than 20 minutes. BC uses action labels from both the deliberate-failure and expert corpora. Recording samples the current key state after each capture; deployment appends the new capture with the previously held state before dispatching the next, while process memory only segments deaths and restarts. Mechanics outside the selected scope, including gravity inversion, are excluded. Failure and expert trajectories are each split 90/10 with seed 42 at the base-episode level. During tokenizer training, each adjacent-frame pair receives a shared two-dimensional translation whose horizontal and vertical ofsets are sampled independently and uniformly from [−4, 4] pixels. For dynamics training, the frozen encoder instead tokenizes each base episode at vertical ofsets $\{ - 4 , - 2 , 0 , 2 , 4 \}$ , expanding the corpus to 21,320 episode variants. All windows and variants inherit their base-episode assignment; although no complete trajectories are duplicated, deterministic repeated attempts can share near-identical prefixes, so the 10% stratum is an in-distribution development set rather than a generalization test. The tokenizer, dynamics model, and BC controller use this split; PPO draws rollout seeds from the full corpus and uses 512 fixed development contexts for checkpoint selection. Tokenizer and dynamics training use seed 42, while controller training is repeated with seeds 43–45 under the same frozen representation and world model. The first two stages run sequentially in BF16 on one A100, controller training runs in BF16 on one H200, and live inference and local latency evaluation use the same RTX 2060 with PyTorch 2.11.0+cu126 (CUDA 12.6).

The retained full runs took 3.19 A100 hours for the tokenizer, 4.20 A100 hours for the dynamics model, and 17.41 H200 hours for the seed-43 H = 45 controller, including BC and 15,000 PPO iterations. Thus the retained pipeline required 24.80 hours of sequential training while occupying one GPU at a time—roughly one day end to end. The matched spatial-only controller took 17.12 H200 hours under the same iteration budget.

Checkpoint and live evaluation. PPO checkpoint selection uses survival on 512 fixed latent development contexts every ten iterations. All live checkpoints are then frozen and act deterministically from pixels alone; Auto-Retry standardizes restarts but provides no policy input. For each seed, we compare PPO with the exact BC checkpoint that initialized it over 25 scored attempts on Stereo Madness, Back on Track, and Polargeist. BC and PPO attempts are separate samples paired only by lineage and run until detected death. Background appearance cycles independently of level progress, while capture and dispatch phases vary near timing boundaries; repetitions measure this pixel-level deployment variability. The protocol also covers Stereo Madness Copy, which begins at Level 1’s first ship section, and post-freeze community layout Stereo INSANE Nerfed, a held-out test within supported mechanics. To validate the earlier 30-FPS evaluations, a cadence check compares the identical frozen PPO checkpoint on Stereo Madness Copy over 25 optimized-path attempts at 60 FPS and 20 earlier diagnostic-path attempts at 30 FPS.

Table 1 Training and checkpoint selection; tokenizer/dynamics learning rates use cosine decay.
<table><tr><td>Stage</td><td>Optimization</td><td>Selection</td></tr><tr><td>Tokenizer</td><td>Adam; 1,000 epochs; batch 2,048; LR  $1 0 ^ { - 3 } \to 1 0 ^ { - 5 } ;$  slowness 0.1; uniformity 0.01.</td><td>Min. loss (epoch 920).</td></tr><tr><td>Dynamics</td><td>AdamW; 200×500 steps; batch 512; LR  $2 \times 1 0 ^ { - 3 } \xrightarrow { } 5 \times 1 0 ^ { - 5 }$  weight decay  $0 . 0 1 ;$  dropout  $0 . 1 ;$  death oversampling 5×.</td><td>Max. death F1 (epoch 139).</td></tr><tr><td>BC</td><td>Three seeds; AdamW; 50 epochs/seed; batch 512; LR  $1 0 ^ { - 3 }$  ; weight decay positive jump weight 1.5.</td><td>Min. loss (10; 10; 11).</td></tr><tr><td>PPO</td><td> $1 0 ^ { - 4 } ;$  Three seeds; Adam; 15,000 iterations/seed; 512 rollouts/iteration; four update epochs; minibatch 512; LR  $1 0 ^ { - 4 } ;$  entropy/critic weights 0.01/0.5; ratio clip 0.2; max grad. norm 0.5.</td><td>Max. survival (12,420; 5,090; 12,280).</td></tr></table>

The endpoint is frames survived, except the cross-cadence probe uses wall time. PPO-minus-BC intervals independently resample each frozen policy’s 25 attempts 50,000 times and characterize deployment variability; the three seed diferences provide replication.

A post-freeze probe retrains seed 43 from the same BC parent with H = 20 for 30,000 iterations (15.61 H200 hours), versus H = 45 for 15,000 (17.41 hours), and scores 10 versus 25 attempts per layout. Because horizon and iteration count co-vary, this is a roughly compute-matched shorter-rollout probe, not an isolated ablation.

A second seed-43 probe keeps the H = 45 recipe and 15,000-iteration budget but removes h from controller training and inference. It scores the resulting BC and PPO checkpoints over 10 attempts on the three oficial levels and held-out layout; comparisons with the 25-attempt temporal-state reference independently bootstrap the two frozen-policy samples.

Latency and continuation protocols. Live latency covers capture through dispatch over 5,000 optimizedpath frames. The qualitative intervention starts immediately before a spike from one encoded four-frame prefix; greedy branches difer only in their first action and then idle. Quantitative diagnostics re-encode the unaugmented development trajectories, yielding 22,745 next-frame windows from 413 usable trajectories. The paired intervention flips only the final context action and bootstraps trajectories 5,000 times; autoregressive diagnostics greedily feed predictions back while replaying recorded actions. The standard cohort has 1,024 starts from 153 trajectories; the exploratory 200-step cohort has 256 starts from the only four suficiently long trajectories.

Targeted dynamics-loss ablations. We compare the reported structured-SLS+CPC dynamics model with three single-training-run variants: uniform label smoothing with the same ε = 0.1, no label smoothing, and structured SLS without CPC. They retain the tokenizer, corpus, split, architecture, seed 42, optimization, and maximum-development-death-F1 selection rule; all checkpoints share the exact evaluation windows and rollout starts.

External transition-latency protocol. We also time the native imagined transition of DashVMC and released Pong checkpoints of IRIS and DIAMOND on the same RTX 2060. Each runs in a fresh process at batch one with FP32 eager execution for 30 warm-ups and two repetitions of 100 synchronized transitions; difering native interfaces make this a cost, not quality, comparison.

## 5 Results

## 5.1 Policy refinement in the live game

The central question is whether policy improvement learned only through frozen-model rollouts transfers beyond imitation to the live game. The selected tokenizer reconstructs development edge maps at 34.10 dB PSNR, and the dynamics model reaches 29.58% visual-token accuracy. Deaths comprise 1.80% of its 22,745 development windows; at threshold 0.5 the death head obtains 0.721 precision, 0.883 recall, 0.794 F1, 0.994 AUROC, and 0.050 expected calibration error. Because this stratum selected the dynamics checkpoint, these are in-distribution development diagnostics rather than untouched test results. These component metrics establish a usable but imperfect imagined environment; the live comparison below tests whether it nevertheless teaches better actions.

Table 2 Live survival on all five layouts. No-op entries are mean frames standard deviation over 10 attempts and are shared across seeds. BC/PPO entries use 25 attempts; ∆ is PPO minus its exact parent BC [95% attempt-level bootstrap interval]. The first three are oficial levels, Stereo Madness Copy begins in ship form, and Stereo INSANE Nerfed is held out.
<table><tr><td>Level</td><td>Policy</td><td>Seed 43</td><td>Seed 44</td><td>Seed 45</td></tr><tr><td>Stereo Madness</td><td>No-op</td><td></td><td> $\mathrm { s h a r e d } \colon 4 6 . 2 \pm 0 . 7$ </td><td></td></tr><tr><td></td><td>BC</td><td> $1 2 0 . 2 \pm 9 1 . 8$ </td><td> $1 3 0 . 6 \pm 7 6 . 0$ </td><td> $1 5 6 . 0 \pm 1 0 0 . 2$ </td></tr><tr><td></td><td>PPO</td><td> ${ \bf 2 8 0 . 1 \pm 2 3 . 3 }$ </td><td> ${ \bf 2 8 0 . 4 \pm 3 3 . 5 }$ </td><td> $\mathbf { 3 0 8 . 8 \pm 6 8 . 6 }$ </td></tr><tr><td></td><td></td><td>∆ [95% CI] 159.9 [122.7, 195.4]</td><td> $1 4 9 . 8 \ [ 1 1 6 . 8 , 1 8 0 . 6 ]$ </td><td> $1 5 2 . 8 \ [ 1 0 6 . 7 , 1 9 9 . 8 ]$ </td></tr><tr><td>Back on Track</td><td>No-op</td><td></td><td> $\mathrm { s h a r e d } \colon 6 3 . 4 \pm 0 . 7$ </td><td></td></tr><tr><td></td><td>BC</td><td> $1 2 3 . 5 \pm 8 1 . 5$ </td><td> $1 1 1 . 1 \pm 6 0 . 3$ </td><td> $1 2 0 . 2 \pm 6 9 . 6$ </td></tr><tr><td></td><td>PPO</td><td> ${ \bf 2 6 0 . 0 \pm 5 5 . 5 }$ </td><td> $\mathbf { 1 9 5 . 9 \pm 1 2 6 . 5 }$ </td><td> $\mathbf { 2 0 9 . 4 \pm 1 3 0 . 0 }$ </td></tr><tr><td></td><td>∆ [95% CI]</td><td> $1 3 6 . 5 \ [ 9 8 . 8 , 1 7 4 . 4 ]$ </td><td> $8 4 . 8 \ [ 3 1 . 9 , 1 3 9 . 6 ]$ </td><td> $8 9 . 2 \ [ 3 4 . 6 , 1 4 7 . 2 ]$ </td></tr><tr><td>Polargeist</td><td>No-op</td><td></td><td> $\mathrm { s h a r e d : ~ } 4 1 . 5 \pm 0 . 7$ </td><td></td></tr><tr><td></td><td>BC</td><td> $5 2 . 4 \pm 1 6 . 7$ </td><td> $4 9 . 0 \pm 1 6 . 5$ </td><td> $4 6 . 2 \pm 9 . 2$ </td></tr><tr><td></td><td>PPO</td><td> ${ \bf 6 5 . 2 \pm 3 3 . 4 }$ </td><td> ${ \bf 6 8 . 3 \pm 4 4 . 1 }$ </td><td> ${ \bf 6 3 . 1 \pm 3 2 . 9 }$ </td></tr><tr><td></td><td>∆ [95% CI]</td><td> $1 2 . 8 \ [ - 0 . 2 , 2 8 . 6 ]$ </td><td> $1 9 . 4 \ [ 3 . 0 , 3 9 . 0 ]$ </td><td> $1 6 . 8 \ [ 5 . 7 , 3 1 . 5 ]$ </td></tr><tr><td>Stereo Madness Copy</td><td>No-op</td><td></td><td> $\mathrm { s h a r e d : ~ } 1 3 5 . 0 \pm 0 . 0$ </td><td></td></tr><tr><td></td><td>BC</td><td> $3 6 6 . 8 \pm 1 7 2 . 8$ </td><td> $2 8 5 . 7 \pm 1 5 3 . 5$ </td><td> $2 9 9 . 4 \pm 1 7 5 . 2$ </td></tr><tr><td></td><td>PPO</td><td></td><td> $\mathbf { 4 9 5 . 8 \pm 1 3 7 . 5 }$ </td><td> ${ \bf 5 4 6 . 8 \pm 9 4 . 1 }$ </td></tr><tr><td></td><td>∆ [95% CI]</td><td> $\mathbf { 4 3 5 . 6 \pm 1 9 6 . 1 }$   $6 8 . 8 \ [ - 3 3 . 3 , 1 6 6 . 1 ]$ </td><td> $2 1 0 . 1 \ [ 1 2 7 . 9 , 2 8 5 . 5 ]$ </td><td> $2 4 7 . 4 \ [ 1 6 7 . 6 , 3 2 0 . 7 ]$ </td></tr><tr><td>Stereo INSANE</td><td>No-op</td><td></td><td> $\mathrm { s h a r e d } \colon 4 6 . 1 \pm 0 . 8$ </td><td></td></tr><tr><td>Nerfed</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>BC</td><td> $1 0 4 . 8 \pm 7 2 . 1$ </td><td> $1 0 2 . 1 \pm 6 4 . 0$ </td><td> $9 7 . 8 \pm 5 7 . 9$ </td></tr><tr><td></td><td>PPO</td><td> $\mathbf { 2 9 0 . 4 \pm 4 8 . 6 }$ </td><td> $\mathbf { 3 4 2 . 8 \pm 8 0 . 8 }$ </td><td> $\mathbf { 2 6 6 . 7 \pm 1 2 8 . 9 }$ </td></tr></table>

In Table 2, BC exceeds no-op everywhere and PPO improves over its parent in all fifteen seed-level comparisons. Across the three controller seeds, the paired improvements are $1 5 4 . 2 \pm 5 . 2$ frames on Stereo Madness, $1 0 3 . 5 \pm 2 8 . 7$ on Back on Track, and $1 6 . 3 \pm 3 . 3$ on Polargeist, where the variation is the sample standard deviation across the three seed diferences. Every interval on the first two levels excludes zero; the smaller Polargeist gain is positive for all seeds, but seed 43’s interval marginally includes zero. Polargeist’s smaller gain may reflect its yellow-orb mechanic, which requires a second, precisely timed jump input while airborne and is absent from the preceding levels.

On Stereo Madness Copy, seed 43’s interval includes zero but seeds 44–45 exclude it (mean gain 175.4±94.2); on held-out Stereo INSANE Nerfed, all exclude zero (mean gain $1 9 8 . 3 \pm 3 7 . 6 )$ . The Copy is not held out, but exposes strong altitude control from Level 1’s first ship section. The 60-FPS cadence probe on Stereo Madness Copy yields similar observed real-time survival: mean recorded episode wall time is 17.72 s at 60 FPS versus 17.79 s at 30 FPS, with a 95% bootstrap interval of $[ - 1 . 0 4 , 0 . 8 9 ] \mathrm { s }$ for the mean diference (60 FPS minus 30 FPS). This limited one-checkpoint, one-layout probe supports cadence transfer there, not equivalence across the primary cube levels.

Runtime temporal-state ablation. The matched spatial-only controller gives up 1.11 frames on the fixed dream-selection metric, reaching 29.02/45 versus 30.13/45 for the seed-43 temporal-state reference.

Table 3 Seed-43 controller-input ablation (mean frames sample SD). Dream is best survival on the same 512 fixed contexts; live results use 25 attempts for the temporal-state reference and 10 for spatial-only.
<table><tr><td>Controller input Dream</td><td></td><td>Stereo</td><td>Back</td><td></td><td>Polargeist INSANE Nerfed</td></tr><tr><td> $\mathbf { z } _ { t } + \mathbf { h } _ { t }$ </td><td>30.13</td><td> $2 8 0 . 1 \pm 2 3 . 3$ </td><td> $2 6 0 . 0 \pm 5 5 . 5$ </td><td> $6 5 . 2 \pm 3 3 . 4$ </td><td> $2 9 0 . 4 \pm 4 8 . 6$ </td></tr><tr><td> $\mathbf { z } _ { t } \ \mathrm { o n l y }$ </td><td>29.02</td><td> $2 7 7 . 9 \pm 1 9 . 2$ </td><td> $2 9 4 . 9 \pm 6 4 . 4$ </td><td> $1 2 8 . 0 \pm 6 5 . 1$ </td><td> $2 4 8 . 1 \pm 7 7 . 3$ </td></tr></table>

Live performance is largely retained without $\mathbf { h } _ { t } .$ . This suggests that temporal state contributes little in this predominantly reactive game, where the current grid exposes nearby geometry and avatar position and recent history mainly disambiguates motion such as scrolling and vertical velocity. A memory-dependent environment would be expected to show a larger drop when $\mathbf { h } _ { t }$ is removed. Although $\mathbf { h } _ { t }$ improves dreamselection survival by 1.11 frames, the world model’s demonstrated role remains to provide PPO’s ofline training environment.

## 5.2 Live control at game speed

The complete perception-to-action loop must finish within 16.67 ms; at 60 Hz it can revise the binary key state every frame, not press 60 times per second. While the RTX 2060 also renders the game, 5,000 measurements have mean/median 12.018/12.083 ms, p95 14.068 ms, p99 14.913 ms, and maximum 24.059 ms. Eleven frames (0.22%) exceed the period; on a miss the loop skips its pacing sleep and immediately begins the next capture rather than reusing an old action. This doubles the temporal resolution of the 30-FPS demonstrations. Qualitatively, the ship policy uses sustained thrust and release intervals: ship motion remains correctable in flight, whereas a cube jump is largely ballistic until landing and therefore demands sparser, more precise timing. The smooth, low-duty-cycle behaviour is consistent with PPO’s per-step jump penalty, although it does not isolate that penalty’s efect.

## 5.3 Cost of an imagined step

DashVMC’s native imagined transition averages 11.05 ms, compared with 49.30 ms for DIAMOND and 177.83 ms for IRIS, corresponding to 4.5× and 16.1× lower latency, respectively. Only DashVMC fits the 16.67-ms 60-FPS period on this hardware.

Table 4 Batch-1 native imagined-transition latency on an RTX 2060 (synchronized run means in milliseconds). Interfaces difer, so this compares operational cost rather than quality.
<table><tr><td>Model</td><td>Run 1</td><td>Run 2</td><td>Mean</td></tr><tr><td>DashVMC</td><td>11.04</td><td>11.06</td><td>11.05</td></tr><tr><td>DIAMOND</td><td>50.13</td><td>48.46</td><td>49.30</td></tr><tr><td>IRIS</td><td>176.94</td><td>178.73</td><td>177.83</td></tr></table>

Parallel grid prediction and latent-space feedback avoid autoregressive token generation and pixel decoding at every imagined step, compounding the savings through PPO.

## 5.4 Action sensitivity and autoregressive fidelity

Beyond its 45-step training horizon, the model feeds predicted grids and new actions back into context. Interactive rollouts reproduce scrolling terrain, camera and height changes, collisions, and avatar transformations; long continuations may remain coherent or eventually repeat, empty, or invent impossible geometry. Video demonstrations of interactive level continuations are available on the project page.

Table 5 Recorded-action rollout diagnostics. Accuracy and decoder-space PSNR compare predicted with recorded grids; JS compares pooled token marginals. Standard: 1,024 starts from 153 trajectories; extended: 256 starts from four long trajectories.
<table><tr><td>Cohort</td><td>Horizon</td><td>Token acc. ↑</td><td>PSNR ↑</td><td>Token JS ↓</td></tr><tr><td>Standard</td><td>1</td><td>30.14%</td><td>32.83</td><td>0.0037</td></tr><tr><td></td><td>5</td><td>19.29%</td><td>28.25</td><td>0.0045</td></tr><tr><td></td><td>10</td><td>14.06%</td><td>25.23</td><td>0.0052</td></tr><tr><td></td><td>20</td><td>8.26%</td><td>22.45</td><td>0.0079</td></tr><tr><td></td><td>45</td><td>2.91%</td><td>19.99</td><td>0.0208</td></tr><tr><td>Extended</td><td>45</td><td>3.02%</td><td>19.78</td><td>0.0351</td></tr><tr><td></td><td>100</td><td>1.38%</td><td>19.49</td><td>0.0428</td></tr><tr><td></td><td>200</td><td>0.74%</td><td>19.39</td><td>0.0733</td></tr></table>

Table 5 is best read relative to screen replacement: scrolling replaces the entire visible screen in roughly 45 frames. The decline from 32.83 dB at horizon 1 to 19.99 dB at 45 therefore marks a transition to fully generated geometry, not only compounding error. Thereafter PSNR is nearly flat within the extended cohort (19.78/19.49/19.39 dB at horizons 45/100/200): long rollouts can remain coherent, but beyond one screen of travel they are not reliable counterfactual environments for PPO.

Figure 2 instead isolates action sensitivity: from one fixed context, changing one action makes the jump branch clear the spike while idle collides. Across all 22,745 development transitions, the factual action has lower next-frame negative log-likelihood than its flipped counterpart in 69.52% of contexts. The trajectory mean advantage is 0.165 nats per visual token (95% episode-bootstrap interval [0.150, 0.181]); factual-action token accuracy is 29.58% versus 27.78% after flipping, and 11.04% of argmax token predictions change.

![](images/eff18d8ab2359b9bf49b615ec9ce18ed4d1b418f83a9da148d3c18a0750c1879.jpg)  
Figure 2 Action-conditioning diagnostic from one shared four-frame context. The rows are aligned at t + 1 and $t + 5 \colon$ changing only the first future action makes the idle branch collide and terminate, while the jump branch clears the spike and lands by t + 13.

Table 6 Matched dynamics-loss ablations on fixed development data. Ours uses structured SLS and CPC; other row names denote the changed component. Action advantage is the episode-mean factual-versus-flipped NLL diference. Rollout columns average horizons $1 / 5 / 1 0 / 2 0 / 4 5$ . Each row is one run; bold is best.
<table><tr><td>Variant</td><td>Death F1 ↑</td><td>Action adv. ↑</td><td>Mean acc. (%) ↑</td><td>Mean PSNR ↑</td><td>Mean JS ↓</td></tr><tr><td>Ours</td><td>.794</td><td>.165</td><td>14.93</td><td>25.75</td><td>.0084</td></tr><tr><td>No SLS</td><td>.808</td><td>.196</td><td>14.33</td><td>25.65</td><td>.0115</td></tr><tr><td>No CPC</td><td>.787</td><td>.148</td><td>13.89</td><td>25.56</td><td>.0104</td></tr><tr><td>Uniform LS</td><td>.826</td><td>.156</td><td>12.90</td><td>25.41</td><td>.0141</td></tr></table>

Table 6 averages each rollout curve over the five horizons; Ours leads accuracy and PSNR while minimizing token-marginal JS, and removing CPC worsens every endpoint. Removing SLS improves death F1 and factual-action advantage but reduces all three rollout aggregates, whereas uniform smoothing attains the highest death F1 but the weakest rollout profile. These single-run diagnostics support CPC and associate FSQ-structured smoothing with better rollout fidelity, without establishing training variance or downstream policy efects.

Rollout horizon. The roughly compute-matched seed-43 H = 20 probe changes macro-average live survival by +1.9% relative to H = 45, with a 95% stratified-bootstrap interval of [−8.1, 13.5]%; every per-layout interval includes zero. Thus 20-step dreams preserve observed performance and show that late H = 45 states are not necessary for this run, but the single seed, unequal attempts, and co-varying iteration count preclude an isolated horizon claim.

## 6 Limitations

DashVMC studies one deterministic binary-action game, with one post-freeze layout within known mechanics. Repeated level prefixes make model diagnostics in-distribution; three controller seeds share one tokenizer and dynamics checkpoint, measuring controller rather than end-to-end uncertainty. A matched seed-43 spatial only probe finds no consistent live deficit after removing h , but its 10-attempt comparison does not establish equivalence; no direct frame-stack controller isolates the value of discrete visual tokens. Death calibration covers only recorded states, leaving recursive reward exploitation unquantified. Loss ablations are single run model diagnostics; controller-input and H = 20 probes are single-seed, and the latter is only roughly compute matched. One-GPU access constrained multi-seed ablations and scaling curves, while wall-clock live evaluation limited deployment tests. Finally, 60-FPS behavioural equivalence is probed on one checkpoint and layout, and reported speed depends on the RTX 2060 and compact edge representation.

## 7 Conclusion

DashVMC turns a fixed archive of roughly two hours of action-labelled gameplay into an interactive, actionconditioned predictive model of the game. Its latent rollouts form a learned environment for ofline model based reinforcement learning: PPO improves BC-initialized controllers without additional live interaction, and the resulting policies consistently outlive their BC parents after zero-shot transfer to the non-paused game. The deployed pipeline meets the game’s 60-Hz timing budget, while the spatial-only ablation shows that the learned dynamics can enable policy improvement without remaining in the live control path. Under data scarcity, the world model thus acts as an interface between past experience and future decisions without prescribing how it must be used: DashVMC uses that interface for imagined policy optimization, while planning-based agents could instead query it at evaluation time.

## Acknowledgments

We thank Clément Chatelain and Robin Condat for their guidance during the Representation Learning course at INSA Rouen Normandy, in which this project began, and CRIANN for providing the computing resources used for training. We thank Maël Planchot for contributing gameplay data. Geometry Dash is developed by RobTop Games.

## References

Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, and François Fleuret. Difusion for world modeling: Visual details matter in atari. In Advances in Neural Information Processing Systems, volume 37, 2024.

Bowen Baker, Ilge Akkaya, Peter Zhokhov, Joost Huizinga, Jie Tang, Adrien Ecofet, Brandon Houghton, Raul Sampedro, and Jef Clune. Video pretraining (VPT): Learning to act by watching unlabeled online videos. arXiv preprint arXiv:2206.11795, 2022.

Maxime Burchi and Radu Timofte. Learning transformer-based world models with contrastive predictive coding. In International Conference on Learning Representations, 2025.

David Ha and Jürgen Schmidhuber. Recurrent world models facilitate policy evolution. In Advances in Neural Information Processing Systems, volume 31, 2018.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640:647–653, 2025. doi: 10.1038/s41586-025-08744-2.

Todd Hester, Matej Vecerik, Olivier Pietquin, Marc Lanctot, Tom Schaul, Bilal Piot, Dan Horgan, John Quan, Andrew Sendonaris, Ian Osband, Gabriel Dulac-Arnold, John Agapiou, Joel Z. Leibo, and Audrunas Gruslys. Deep q-learning from demonstrations. In AAAI Conference on Artificial Intelligence, volume 32, 2018.

Yann LeCun. A path towards autonomous machine intelligence. OpenReview, 2022. URL https://openreview.net/ pdf?id=BZ5a1r-kVsf. Version 0.9.2.

Ted Li and Sean Raferty. Playing Geometry Dash with convolutional neural networks. Stanford CS231N course report, 2017. URL https://cs231n.stanford.edu/reports/2017/pdfs/605.pdf.

Bin Lin, Zongjian Li, Yuwei Niu, Kaixiong Gong, Yunyang Ge, Yunlong Lin, Mingzhe Zheng, JianWei Zhang, Miles Yang, Zhao Zhong, Liefeng Bo, and Li Yuan. iFSQ: Improving FSQ for image generation with 1 line of code. arXiv preprint arXiv:2601.17124, 2026.

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In International Conference on Computer Vision, pages 2980–2988, 2017.

Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: VQ-VAE made simple. In International Conference on Learning Representations, 2024.

Vincent Micheli, Eloi Alonso, and François Fleuret. Transformers are sample-eficient world models. In International Conference on Learning Representations, 2023.

Vincent Micheli, Eloi Alonso, and François Fleuret. Eficient world models with context-aware tokenization. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 35623–35638. PMLR, 2024.

John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter Abbeel. High-dimensional continuous control using generalized advantage estimation. arXiv preprint arXiv:1506.02438, 2015.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Zaishuo Xia, Yukuan Lu, Xinyi Li, Yifan Xu, and Yubei Chen. Cloning deterministic worlds: The critical role of latent geometry in long-horizon world models. arXiv preprint arXiv:2510.26782, 2025.