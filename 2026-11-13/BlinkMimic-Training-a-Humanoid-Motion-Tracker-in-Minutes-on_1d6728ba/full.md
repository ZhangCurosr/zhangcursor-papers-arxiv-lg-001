# BlinkMimic: Training a Humanoid Motion Tracker in Minutes on a Single GPU

Fuyi Chen

## Abstract

Whole-body motion tracking lets a humanoid robot follow reference motions, but training a tracking policy for a library of motions has usually taken hours of GPU time. SuperMimic showed that the policy can instead be trained by backpropagating through a world model learned from parallel simulation, and InstantTrack added Muon updates, a residual policy around the reference and an averaged deployed policy. We build on both and present BlinkMimic, a recipe that makes this training faster. On a simulated Unitree G1, BlinkMimic trains one policy for all 40 LAFAN1 clips in 3.4 minutes on a single RTX 4090, measured until three consecutive checkpoints reach a normalized return of at least 0.82 with no failed clip (mean of three seeds). We split training time into the cost of an iteration, the iterations needed to fit the motions, and the iterations needed to remove the last failures, and assign each component, inherited or new, to one of these terms. The new components are a step size that decreases once training has progressed, resets directed by failures of the deployed policy under deployment conditions, and GPU graph capture of the whole iteration. A leave-one-out ablation with three seeds per component shows which term each component afects, and further measurements of the gradient, the update rule and the failure cases explain these efects. Code and per-seed results are available at https://github.com/fuyichen0725/blinkmimic.

## 1 Introduction

A humanoid that can follow arbitrary reference motions is a building block for teleoperation, imitation and whole-body skills (Peng et al., 2018; He et al., 2024; Liao et al., 2025). Training such a tracker has been expensive: with on-policy reinforcement learning, a policy for a library of motions takes hours to days on several GPUs (Luo et al., 2023; Chen et al., 2025). SuperMimic (Liao et al., 2026) took a diferent route. It learns a diferentiable world model of the robot from parallel simulation and improves the policy by backpropagation through time (BPTT) in that model, replacing the high-variance policy-gradient estimate with a pathwise one. This cut the cost of training a tracker substantially and made training time a quantity worth optimizing in its own right. InstantTrack (Dao, 2026) then showed in an open implementation that a few changes, an averaged deployed policy, Muon updates (Jordan et al., 2024) and a policy that outputs a correction to the reference, bring training on the 40 LAFAN1 clips (Harvey et al., 2020) down to minutes. Training in minutes changes how such controllers are developed: a change to the reward, the retargeting or the robot model can be tested within the same hour on one workstation.

This paper builds on both and asks how far, and why, this training can be made faster. Our recipe, BlinkMimic, trains one policy for all 40 clips to a strict target, three consecutive checkpoints with a mean return of at least 0.82 and no failed clip, in 3.4 minutes on one RTX 4090 (Figure 1). Recipes like this tend to accumulate changes without a clear account of which ones matter. We organize ours around a simple identity: the time to the target is the cost of an iteration times the number of iterations, and the iterations split into a fit phase, until the mean return reaches 0.82, and a tail phase, until the last failing clips stop failing. The three terms are limited by diferent things. We assign each ingredient, inherited from SuperMimic and InstantTrack or new, to one of them, and add new ingredients where a term was not yet addressed.

The fit phase is limited by the update rule. The imagined gradient is accurate over the horizon on which the policy acts, and regression onto a perfect teacher needs no fewer updates than learning from the imagined gradient. What remains is how much each update achieves. Muon updates, here applied to both the policy and the world model, and the residual policy shorten this phase, and we add a step size that starts high and falls once training has progressed.

The tail phase is limited by where the simulator is spent. The last failures occur at a few states that the deployed policy reaches only after minutes of tracking. These states are learnable: the hardest clip trained alone stops failing early, but in the mix the states before its hard spot receive about 1% of the practice, and failure statistics taken during training point at the wrong places. A periodic probe with the deployed policy under deployment conditions finds these states directly.

![](images/5a69f2f63fd10c7baba60aa8ac08bbf7661e6cb0162b425a52e44204306eee89.jpg)  
b Time to target

![](images/9da233c3b17aa7d6ff1aad3f5b153de71dff65df06c11c7deb5fc24f6caa73b3.jpg)

c Mean return, 40 clips  
![](images/85bb9250e140fb0a3eca459dccc888ca019f2b81db4919dc26d72920326577d2.jpg)  
d Where the time goes

training time on one RTX 4090 (min)  
![](images/0032ab614a80d122374e8aa819d0f6e7c2b96d4f659de60067696897bdbed822.jpg)  
Figure 1 | One policy for 40 LAFAN1 clips in 3.4 minutes on one RTX 4090. a, The deployed policy of seed 0 at the first checkpoint that meets the target (iteration 1500, 3.32 minutes of training), simulated under evaluation settings; each row shows the 3.6 most dynamic seconds of one clip. Solid: robot. Translucent: reference. b, Training time to the first mean return of 0.82 (light) and to the target of three consecutive clean checkpoints (dark), on the same GPU with the same evaluator. InstantTrack and BlinkMimic: measured, mean of three seeds. SuperMimic (hatched): estimated as the iterations of InstantTrack’s SuperMimic runs times SuperMimic’s time per iteration measured on our GPU. c, Mean return over 40 clips × 4 replicas against training time; thin lines are seeds. d, Where the 3.4 minutes go (mean of three seeds), and the cost of one iteration by stage, measured before the collection step was captured as a graph.

The iteration is limited by the host. At these model sizes the GPU often waits for the CPU. We capture the whole iteration, including the MuJoCo-Warp (Google DeepMind and NVIDIA, 2025) physics step and episode resets, as CUDA graphs, and halve the imagination batch, which costs fewer extra iterations than it saves per iteration.

A leave-one-out ablation of the final recipe, with three seeds per ingredient, checks that each ingredient acts on the term it was chosen for. Our contributions are:

• BlinkMimic, a single-GPU recipe that trains one policy for 40 LAFAN1 clips in 3.4 minutes, with code and per-seed results (Table 1).

• A design organized by where training time goes, with a short analysis for each ingredient: how model error enters the gradient, why the reward favors a residual policy, why the step size should fall, why normalized updates produce a batch knee, and why failure statistics from training misdirect practice (Section 4).

• A leave-one-out ablation on a common baseline (Table 2) and measurements that explain it: gradient comparisons between the model and the simulator, a perfect-teacher reference, batch forks with a prediction registered before the run, and attribution tests for the tail (Section 5).

Section 3 describes the task and the training loop, Section 4 presents the ingredients term by term, and Section 5 reports the result and the ablation and then examines the fit phase and the tail phase.

## 2 Related Work

Humanoid motion tracking. Physics-based tracking grew out of DeepMimic (Peng et al., 2018) and was scaled to large libraries by PHC (Luo et al., 2023), MaskedMimic (Tessler et al., 2024), OmniH2O (He et al., 2024), GMT (Chen et al., 2025) and BeyondMimic (Liao et al., 2025). These works use on-policy RL with massively parallel simulation. SuperMimic (Liao et al., 2026) replaced the policy-gradient estimator with BPTT through a learned world model, and InstantTrack (Dao, 2026) introduced EMA deployment, Muon and a reference residual in an open implementation. We build on this line.

Diferentiable simulation and model-based gradients. Pathwise gradients through simulators (Xu et al., 2022; Georgiev et al., 2024) or learned models (Deisenroth and Rasmussen, 2011; Hafner et al., 2020, 2023) trade variance for bias, which can be large at contacts and in chaotic rollouts (Suh et al., 2022; Metz et al., 2021); model error compounds over the horizon (Janner et al., 2019; Buckman et al., 2018). Mixed estimators interpolate between the two gradient types (Parmas et al., 2018; Suh et al., 2022).

Optimizers, batch size and step size. Muon (Jordan et al., 2024; Liu et al., 2025) orthogonalizes the update of each weight matrix; SOAP (Vyas et al., 2024) preconditions in a Kronecker eigenbasis. The noise scale predicts a critical batch for SGD-like training (McCandlish et al., 2018), and its interaction with momentum and normalization is subtle (Zhang et al., 2019; Malladi et al., 2022; Cutkosky and Mehta, 2020). Adaptive methods sit near an edge of stability (Cohen et al., 2022); range tests (Smith, 2017) and warmup–stable–decay schedules (Hu et al., 2024) set the step size in practice.

Allocation and evaluation. Prioritized sweeping (Moore and Atkeson, 1993) and level replay (Jiang et al., 2021) spend computation where learning progresses or fails, and imitation learning requires practice on the states the learner itself reaches (Ross et al., 2011; Laskey et al., 2017). Time-to-target on stated hardware is the standard of DAWNBench (Coleman et al., 2017) and MLPerf (Mattson et al., 2020); RL comparisons need multiple seeds (Agarwal et al., 2021).

## 3 Problem Setup

Task. A Unitree G1 with 29 actuated joints tracks reference clips in MuJoCo-Warp (Google DeepMind and NVIDIA, 2025) at 50 Hz. We use the 40 LAFAN1 clips retargeted to the G1 (441,140 frames, 2.45 hours; the average clip lasts 3.7 minutes). The policy outputs a normalized action that is scaled and passed as a joint position target to PD controllers. The reward is a sum of Gaussian tracking kernels with an action-rate penalty, and an episode fails when the height or tilt error of the anchor body exceeds a threshold, following SuperMimic (Liao et al., 2026).

Evaluation. The deployed policy runs deterministically, without domain randomization or noise, from the first frame to the end of every clip, as in SuperMimic and InstantTrack. The normalized return $\bar { R } \in [ 0 , 1 ]$ accumulates until the first failure. A checkpoint meets the target when $\bar { R } \ge 0 . 8 2$ with zero failed clips; Section 5 requires this at three consecutive checkpoints.

Training loop. Figure 2 shows one iteration. In collection, 4096 simulated robots step the noisy training policy under domain randomization and store transitions in a replay bufer; finished episodes restart at a frame drawn from a reset distribution � over 1-second bins of the library. A learned world model $\hat { F } _ { \phi }$ is fit to short replay windows. In imagination, � start states from the bufer are rolled out through $\hat { F } _ { \phi }$ for $H = 3 2$ steps, and the policy follows the gradient, obtained by BPTT, of

$$
\begin{array} { r } { \hat { J } ( \theta ) = \mathbb { E } _ { z _ { 0 } } \bigg [ \sum _ { t < H } \gamma ^ { t } \hat { \alpha } _ { t } \hat { r } _ { t } + \gamma ^ { H } \hat { \alpha } _ { H } V _ { \psi } ( \hat { z } _ { H } ) \bigg ] , } \end{array}\tag{1}
$$

where $\hat { \alpha } _ { t }$ is a product of soft survival probabilities and $V _ { \psi }$ is a critic fit to �-returns of the imagined rollouts. As in InstantTrack, the deployed policy is an exponential moving average (EMA, $\beta = 0 . 9 9 5 )$ of $\pi _ { \theta }$ (Polyak and Juditsky, 1992).

Three distributions over the library should be kept apart: the reset distribution �, which we control; the visitation � of the training robots; and the imagination-start distribution �, which follows � with the delay of the replay bufer. Only � decides what the policy practises.

## 4 BlinkMimic

Let $c _ { i }$ be the wall-clock cost of iteration � and � the first iteration that meets the target. The training time is

$$
T = \sum _ { i < N } c _ { i } = \bar { c } ( N _ { \mathrm { f i t } } + N _ { \mathrm { t a i l } } ) ,\tag{2}
$$

where thefit phase $N _ { \mathrm { f i t } }$ ends when the mean evaluation return first reaches 0.82 and the tail phase $N _ { \mathrm { t a i l } } = N - N _ { \mathrm { f i t } }$ lasts until the last failing clips stop failing. The three terms respond to diferent changes, so we design for them separately.

![](images/9907089fdc244e521c925f00b7ae75557317588c26e62a1126f4f8752c60b55b.jpg)  
Figure 2 | One iteration of BlinkMimic. The real simulator (blue) feeds the learned proxies (orange), and the policy improves through the imagined gradient. Numbers mark the ingredients, inherited or new, in the order of Section 4: (1) Muon updates for the policy and the world model; (2) a step size that falls on progress; (3) the reference residual; (4) the semi-implicit, mirror-augmented world model; (5) the deployment-condition probe and reset allocation; (6) the imagination batch and the whole iteration captured as GPU graphs.

Sections 4.1 to 4.4 shorten the fit phase, Section 4.5 shortens the tail, and Section 4.6 makes each iteration cheaper; Figure 2 marks where each ingredient sits in the loop. The order follows the ablation in Table 2. The full configuration is in Appendix B and the derivations behind the short analyses are in Appendix A.

## 4.1 Muon for the policy and the world model

InstantTrack (Dao, 2026) introduced Muon (Jordan et al., 2024) to this setting. Muon replaces the update of each hidden weight matrix with � $U V ^ { \top }$ , where $U { \boldsymbol { \Sigma } } V ^ { \top }$ is the singular value decomposition of the momentum matrix. The step moves every singular direction by the same amount; it is steepest descent under a limit on the spectral norm of the step. Input, output and bias parameters keep Adam, and the learning rate is matched to Adam’s update magnitude (Liu et al., 2025).

Muon matters here because, as Section 5.3 shows, the fit phase is limited by how much each update achieves, not by the quality of the imagined gradient: regressing the same network onto a perfect teacher takes 2.3× fewer updates with Muon than with Adam. As in InstantTrack, the world model is also trained with Muon, which halves its loss early in training.

## 4.2 A step size that starts high and falls on progress

A larger step removes the initial error faster but leaves more noise at the end. On a quadratic $\begin{array} { r } { L = \frac { 1 } { 2 } h x ^ { 2 } } \end{array}$ with noisy gradients of variance $\sigma ^ { 2 }$ , the error contracts faster as the step � grows (up to $\eta = 1 / h )$ , while the remaining excess loss is $\eta \sigma ^ { 2 } / ( 2 ( 2 - \eta h ) )$ ), which grows with � (Appendix A.6). Averaging the iterates, as our EMA policy does, lowers this floor mainly along stif directions. In world-model training the objective also moves between iterations, which adds to the noise.

We therefore start at the largest policy step that a short screen admits, four times the base rate, and reduce it linearly to the base rate over 300 iterations once the training return first exceeds 0.75, or at iteration 1400 at the latest. The trigger adapts to how fast a seed learns, and the deadline covers seeds whose return rises late.

## 4.3 A residual policy

Following InstantTrack’s reference residual, the policy outputs a correction to the next reference joint target, $a = a ^ { \mathrm { r e f } } + N _ { \theta } .$ with the last layer initialized near zero; the observation and the MDP are unchanged. Here we explain why this helps, and the reason lies in the reward. A tracking kernel $r = \exp ( - e ^ { 2 } / \sigma ^ { 2 } )$ has the slope $| \partial r / \partial e | = ( 2 / \sigma ) \sqrt { x } e ^ { - x }$ with $x = e ^ { 2 } / \sigma ^ { 2 }$ . The slope peaks at $\begin{array} { r } { x = \frac { 1 } { \mathit { \Omega } } } \end{array}$ , falls to 30% of the peak at $x = 2 . 5$ and to $3 . 5 \%$ at $x = 5 .$ . A policy far from the reference therefore gets almost no gradient from its largest errors, while a residual policy starts where every kernel is informative (Figure 3c).

## 4.4 A world model with the simulator’s structure

The policy gradient flows through the model’s Jacobians, so what matters is how well these Jacobians match the simulator along imagined rollouts. Writing $x _ { t }$ for the sensitivity of the state to the policy parameters, the error of the model’s sensitivity is a sum of the Jacobian errors at each earlier step, each propagated forward by the model’s own dynamics (Appendix A.1):

$$
\hat { x } _ { t } - x _ { t } = \sum _ { k < t } \hat { \Phi } ( t , k + 1 ) \left[ \Delta A _ { k } x _ { k } + \Delta B _ { k } \Pi _ { k } + \hat { B } _ { k } \Delta \Pi _ { k } \right] .
$$

Here $\Delta A _ { k }$ and $\Delta B _ { k }$ are the model’s errors in the closed-loop state Jacobian and the action Jacobian, $\Pi _ { k } = \partial \pi / \partial \theta$ $\Delta \Pi _ { k }$ is the change of $\Pi _ { k }$ caused by the model’s diferent state, and $\hat { \Phi } ( t , k )$ is the product of the model’s closed-loop Jacobians from step � to �. Two consequences guide the design. Jacobian errors matter only where imagination goes and where later dynamics amplify them, so the model should be accurate on the policy’s own states. And a small one-step prediction error does not imply a small gradient error, so structure that shapes the Jacobians is worth more than extra accuracy of the prediction itself.

The model therefore uses the integrator’s structure, as does the implementation released with SuperMimic. It predicts a velocity change and integrates position semi-implicitly, plus a small learned position correction; if it shared the integration step exactly, its position Jacobian errors would be a factor Δ� smaller than its velocity errors. Half of every training window is replaced by its left–right mirror image, a symmetry of the robot that costs no simulation. In the final recipe, where the model is also trained with Muon, removing the structure and the mirror no longer lengthens the fit phase but lowers the return at which the policy settles, and with it the margin above the target (Section 5.2).

## 4.5 Spending the simulator on deployment failures

After the mean return reaches the target, a few clips still fail at a few places. Let $p _ { i }$ be the probability that the deployed policy fails in 1-second bin � under deployment conditions, given that it got there. When failures are rare, the number of failed clips is about $\sum _ { i } p _ { i }$ . If practice on a bin lowers its $p _ { i } ,$ , the fastest way to remove failures is to practise the bins with the highest $p _ { i }$ first (Appendix A.7). Transfer between clips makes this only a guide, but it says what should drive the reset distribution: failure rates of the deployed policy under deployment conditions.

Failure statistics collected during training measure something else. The training policy is noisy, randomized and not averaged. If resets are weighted by those statistics, bins that cannot be learned under randomization keep absorbing resets, while bins that fail only under deployment conditions receive no more than the uniform share (Appendix A.8). BlinkMimic therefore measures deployment failures directly. Every 125 iterations from iteration 1000, a separate simulator sweeps the whole library with the EMA policy under deployment conditions, using one world per 300-frame tile started on the reference 200–450 frames before the tile. The first failure of each world is counted in its bin, counts decay by $0 . 7$ per sweep, and each count is spread over the failing bin and the 10 bins before it, because a failure develops from the policy’s own drift. The resulting map receives 15% of all resets; the rest keep the sampler’s uniform, low-return and training-failure terms (Appendix B). The probe needs no prior knowledge of which clips are hard.

## 4.6 Making each iteration cheap

Imagination batch. Because Muon fixes the length of the step, a smaller batch does not make the step noisier in size; noise only rotates it away from the signal. A signal direction is lost once its strength falls below a detection threshold set by the noise, as known for spiked random matrices (Benaych-Georges and Nadakuditi, 2012). The number of steps to a target then stays nearly flat as the batch shrinks and rises sharply once the step’s overlap with the signal nears a curvature cost (Appendix A.4), a knee rather than the smooth $1 + B _ { \mathrm { n o i s e } } / B$ curve of SGD (McCandlish et al., 2018). Since the position of the knee depends on the model, we measure it with short forks and take $B = 2 0 4 8$ , the smallest batch that leaves the tail intact (Section 5.3)

Removing the host from the loop. At batch 2048, the updates and collection take 122 ms on an RTX 4090, but the MuJoCo-Warp kernels of a collection step add up to only about 11 ms of its 31 ms; the rest is launch gaps and host synchronizations. InstantTrack already captures the two update phases as CUDA graphs; we capture them from the first iteration and add the collection step, with the policy inference, physics step and reward in one CUDA graph with episode resets done by fixed-shape operations inside that graph. The probe sweep is also a graph and runs alongside training over 20 iterations. None of this changes the learning algorithm: after a masked reset every simulator field matches the original bit for bit, except joint velocities, which difer by at most $2 . 4 \times 1 0 ^ { - 7 }$ (Appendix C). In two alternating paired runs on the same GPU, without checkpoints or probes, the collection graph lowers the steady-state iteration from 127.5 to 118.5 ms (−7%) and shortens the first 700 iterations by 18 s.

Table 1 | Time to target on one RTX 4090. 40 clips × 4 replicas, a checkpoint every 100 iterations, training clock including compilation and probes; seeds 0 / 1 / 2 with the mean in parentheses. Times are in minutes. 3 clean checkpoints: the first of three consecutive checkpoints with return ≥ 0.82 and no failed rollout. Tailfailures: failed rollouts summed over checkpoints 2000–3500.
<table><tr><td>Configuration</td><td>ms/it</td><td>Iterations to 0.82</td><td>First ≥0.82</td><td>3 clean checkpoints</td><td>Tail failures</td><td>Final return</td></tr><tr><td>InstantTrack v1.2 (re-run)</td><td>147</td><td>2408 / 2741 / 2492</td><td>5.6 / 6.3 / 5.8 (5.9)</td><td>7.3 / 8.5 / 6.3 (7.4)</td><td>80 / 126 / 113</td><td>0.856</td></tr><tr><td>BlinkMimic (release)</td><td>135</td><td></td><td>1292 / 1456 / 1169 2.89 / 3.28 / 2.64 (2.94) 3.32 / 3.28 / 3.53 (3.38)</td><td></td><td>216/0</td><td>0.861</td></tr></table>

Table 2 | Leave-one-out ablation of the release recipe. Each row removes one ingredient from the full recipe (the last row restores the earlier batch); three seeds per row, all runs on the same GPU type with the same code and evaluator (40 × 4 rollouts, a checkpoint every 100 iterations). It. to 0.82: mean, interpolated, with the change against the full recipe. 3 clean: first of three consecutive clean checkpoints, per seed. Tail: failed rollouts summed over checkpoints 2000–3500, per seed. Return: mean over checkpoints 2000–3500. Times on the RTX 4090 are in Table 1.
<table><tr><td>Removed</td><td>Term</td><td>It. to 0.82</td><td>3 clean (iteration)</td><td>Tail</td><td>Return</td></tr><tr><td>Full recipe</td><td></td><td>1422</td><td>2100/2200/1900</td><td>1/8/0</td><td>0.860</td></tr><tr><td>– Muon on the policy (Adam instead)</td><td> $N _ { \mathrm { f i t } }$ </td><td>not by 6000</td><td>-1-1-</td><td>15 /3 /18</td><td>0.796</td></tr><tr><td>– Muon on the world model</td><td> $N _ { \mathrm { f i t } }$ </td><td>1768 (+24%)</td><td>3000 / 3100 / 2100 (+32%)</td><td>24/22/7</td><td>0.843</td></tr><tr><td>– Progress-triggered step size (constant ×1)</td><td> $N _ { \mathrm { f i t } }$ </td><td>1616 (+14%)</td><td>3000 / 1800 / 2300 (+15%)</td><td>20/3/6</td><td>0.856</td></tr><tr><td>– Reference residual</td><td> $N _ { \mathrm { f i t } }$ </td><td>1751 (+23%)</td><td>1800 / 2700 / 2200 (+8%)</td><td>0/20/8</td><td>0.845</td></tr><tr><td>– Semi-implicit world model and mirror</td><td>R</td><td>1428 (±0%)</td><td>1800 / 2900 / 1900 (+6%)</td><td>0/15/13</td><td>0.854</td></tr><tr><td>– Deployment-condition allocation</td><td> $N _ { \mathrm { t a i l } }$ </td><td>1350 (-5%)</td><td>2000 / 2000 / 1700 (-8%)</td><td>10/0/8</td><td>0.861</td></tr><tr><td>Imagination batch 4096 instead of 2048</td><td>č</td><td>1138 (-20%)</td><td>1700 / 2000 / 2000 (-8%)</td><td>1/0/0</td><td>0.864</td></tr></table>

## 5 Experiments

Protocol. Every checkpoint is evaluated on all 40 clips from the first frame, with four replicas per clip, because the simulator is not bitwise reproducible and a clip near its margin can flip between evaluations. A checkpoint is clean when the mean return is at least 0.82 and all 160 rollouts survive. The target is three consecutive clean checkpoints, reported at the first of them, as in InstantTrack (Dao, 2026); we also report the first checkpoint with return at least 0.82, which marks the end of the fit phase. We checkpoint every 100 iterations (about 13 s). Training time includes compilation, graph capture, checkpoints and probes and excludes the ofline evaluation. Runs that report time have the RTX 4090 to themselves, and baselines are re-run on the same GPU with the same evaluator and checkpoint interval. The ablation runs on a diferent GPU type and compares iterations, each row against the full recipe run on that same GPU type. Because the simulator is not bitwise reproducible across GPUs, the full recipe needs more iterations there than on the RTX 4090 (1422 against 1306 to 0.82, and 2067 against 1533 for the three-checkpoint rule), so numbers are compared only within a table. Seed counts are stated for every number; several diagnostics use one seed and are marked as such.

## 5.1 Main result

BlinkMimic reaches the target in 3.32, 3.28 and 3.53 minutes for seeds 0, 1 and 2 (Table 1). The first checkpoint at return 0.82 comes at 2.94 minutes on average, so the tail phase costs 26 seconds. One iteration costs 135 ms including probes and checkpoints. At the target, the RMS position error of the 14 tracked bodies is 4.1 cm and the mean joint-angle error 4.0°; by iteration 3500 the body error falls to 3.4 cm (Table 3). Meeting the target does not guarantee that later checkpoints stay clean: in seed 1, fallAndGetUp2\_subject2 fails in all four replicas at iteration 3100, long after the step size has settled. We return to such late flips in Section 6; Appendix E lists the earlier recipes.

## 5.2 What each ingredient contributes

We remove one ingredient at a time from the release recipe and keep everything else (Table 2). Outside their own rows, the step-size multiple and the batch stay at the values tuned with all ingredients present. Each row has three seeds, and all 24 runs share one GPU type.

Fit phase. Muon on the policy is the one ingredient the recipe cannot do without. With Adam in its place, the return reaches 0.80 by iteration 3000 and stays at 0.81 up to iteration 6000 in all three seeds, with no failed rollout at the end: the policy survives but does not track precisely enough. The row keeps the ×4 step and the batch of 2048, so it measures what Muon contributes inside the recipe, not the best Adam can do with its own settings. Removing Muon on the world model, the progress-triggered step size or the reference residual lengthens the fit phase by 24%, 14% and 23%, and every seed is slower than every seed of the full recipe. The three-checkpoint rule moves in the same direction but is noisier (+32%, +15%, +8%), because it also depends on the tail. Without these ingredients the policy also settles at a lower return (0.843–0.856 against 0.860).

![](images/fed1a02f9d5367666653076afa27411eeb78c3ed0c53690050db53cbd4b3c372.jpg)

![](images/4794987be31d93aaad22f89ed2079f8fb9087dfca19a782a29c00cda337df329.jpg)

![](images/fa557eb80dc172e41140683e4c9f77870f768ce539f51f0d4a190a42dc34d8fa.jpg)  
Figure 3 | The imagined gradient. a, Change of the 32-step return, in the simulator and in the world model, for policies a few iterations ahead of the current one, for a step backward along the path, and for random directions of equal norm (2048 on-policy states, 5 checkpoints from iteration 1000 to 3000, one seed). b, Correlation between the model’s directional derivatives of the open-loop return and central diferences in the simulator (48 states; ranges over checkpoints). c, A Gaussian tracking kernel and its slope. Inset: share of body-orientation errors past � = 2.5 at iteration 1000 under training conditions, with and without the reference residual (one seed).

Return level. Removing the semi-implicit structure and the mirror leaves the fit phase unchanged (1428 against 1422 iterations) but lowers the mean return over iterations 2000–3500 in every seed (0.8535–0.8548 against 0.8581–0.8628), and two of three seeds fail more often late. In the final recipe the structure therefore acts on the margin above the target. In earlier configurations without Muon on the world model it shortened the fit phase by 19% (Table 4). The row removes structure and mirror together, so their shares are not separated.

Tail phase. Without the probe, late failures double, from 9 to 18 rollouts over three seeds (1 / 8 / 0 against 10 / 0 / 8). This is the direction found in an earlier recipe (Section 5.4), but the counts are small and the seeds overlap, so three seeds do not establish the efect here, and the three-checkpoint rule is not slower without the probe. The probe also costs time: on the RTX 4090 its earlier synchronous form added 20 ms to 121 ms per iteration (Table 5).

Iteration cost. Restoring the batch of 4096 reduces the iterations to 0.82 by 20% but cost 41% more per iteration on the RTX 4090 in an earlier configuration (172 against 122 ms, Table 4), so the time to target grows. The paired comparison on the RTX 4090 in Section 5.3 gives the same answer.

The rest of this section asks why the ingredients act where they do: what limits the fit phase (Section 5.3) and where the tail comes from (Section 5.4).

## 5.3 Why the fit phase is limited by the update rule

The ablation shows that the update rule (Muon, step size) and the residual policy set the length of the fit phase. This section follows one line of argument: the imagined gradient is accurate where it matters, a perfect teacher is not faster, so what remains is how much each update achieves.

The imagined gradient is accurate where the policy acts. We test the model against the simulator in two ways. Along the actual training path (Figure 3a), the change of return predicted by the model matches the simulator, and at no checkpoint is the model more optimistic about the next policy than about the current one (model return minus simulator return 0.0151 against 0.0157 at iteration 1000, about 0.6% of the return). Online training thus leaves no measurable model exploitation at the scale of one update; it does appear when the model is frozen, where the return stops near 0.70 with 14 clips failing (one seed). A step backward along the path loses 0.04–0.10, while steps forward change the return by little after iteration 1000. Random directions of equal norm change nothing. The optimizer moves along stif directions and makes its net progress over hundreds of iterations, as expected near an edge of stability (Cohen et al., 2022). The EMA gains a further 0.012 over the raw policy.

![](images/e7e0f32e2f586b96246edc5b5ad2040458e8faca11f008c58313e196f87906a9.jpg)

![](images/2116b81df2cf2e9cfd3084f91dcf47059eab7ed2fa7cf353bc81e565515dde46.jpg)

![](images/fb52e228689c51052305b17340927f8e9c391c9a70fabdf196a578f2cdc632dc.jpg)

![](images/554fcd69641f47504360b03c0b5fdd4de79e64d189cbf56dd5d72595bba2d639.jpg)  
Figure 4 | The update rule limits the fit phase. a, �, RL updates to reach 0.82 divided by updates of a regression onto a perfect teacher with the same network and optimizer. Blue: changes to the signal; amber: an update-rule change alone, which helps the teacher more than RL. Run lengths and seeds are in Appendix E. b, Singular values of one policy gradient matrix (iterations 700–1600, one seed) and the detection threshold for three batch sizes. c, Iterations to 0.82 against batch, relative to 4096, for RL (2048: thre seeds, bar: range; 1024, 512: one seed) and for the teacher; shaded: range registered before the 512 run. d, Training return for step-size multiples (×1: three-seed mean; others one seed).

Figure 3b shows the limit of the model. The action gradient over 1 and 8 steps correlates at 0.95–0.98 and 0.80–0.89 with the simulator, also just before failures, but over 32 steps the correlation is low. As the sensitivity analysis in Section 4.4 suggests, the error compounds with the horizon. The critic carries much of the signal, and its slope is unreliable (correlation with the true change of return at most 0.11), yet removing it costs 50% more iterations. The gradient is thus accurate at the horizon where the policy can act, and the long horizon mostly delivers the value estimate.

A perfect teacher is not faster. To ask how much better the signal could be, we train the same network with the same optimizer by regression onto the actions of a converged policy, with DAgger-style data collection, and define � as the ratio of RL updates to teacher updates needed to reach 0.82. The teacher is a reference, not a bound. Figure 4a traces � along the history of the recipe. The first configuration needs 30% more updates than regression; the structured world model brings � to 1.04. Muon on the policy speeds up the teacher by 2.3× (3494 to 1478 updates, and SOAP matches it at 1505) but RL by less, so � rises to 1.83. Muon on the world model and the reference residual then bring it back to about 1.0. Before the step-size schedule, the recipe needed 1549 iterations at batch 4096, against 1478–1499 for the teacher; with the schedule, the release recipe needs 1306 at batch 2048, fewer than the teacher. A better signal therefore has little left to give, while a better update rule, as the teacher’s 2.3× gain from Muon shows, can still help.

The batch knee. In-batch noise is small. Cosines between the gradients of two disjoint halves of the batch lie mostly in 0.62–0.87 under Muon, and the teacher gains only 3% from a batch of ×2 or ×4. Most singular directions of the gradient that carry energy remain above the detection threshold at smaller batches (Figure 4b). A measurement of the resolved signal count alone did not explain the knee, however, so we fit the single parameter = 0.53 of the knee model (Appendix A.4) to the fork at � = 1024. Before the run at � = 512 we registered a prediction of 2280–2630 iterations; the run took 2301 (Figure 4c, one seed). A further fork that kept 4096 samples for the critic and used 1024 for the policy kept about 84% of the slowdown (1.17×), so the knee comes mostly from the policy gradient and not from critic data. At � = 2048 the wall time to 0.82 falls by 17% with every seed faster, and we stop there because smaller batches increase late failures. In the release recipe, with its larger step, the cost of halving the batch is higher: 1422 against 1138 iterations to 0.82 (Table 2), a 25% increase, still smaller than the 29% saved per iteration (122 against 172 ms in Table 4)

Step size. A step-size screen shows the same trade-of. At iteration 500 the return rises from 0.606 at ×2 to 0.659 at ×4 and falls to 0.603 at ×8, without divergence (Figure 4d). By iteration 1250, ×2 has caught up, consistent with the higher noise floor of a large step (Section 4.2). In an earlier recipe, a progress-triggered decay from ×4 reduced iterations by 19% against a constant step (three seeds; 14% in the final recipe, Table 2), whereas a cosine schedule with a fixed horizon gave no gain.

## 5.4 The tail phase

Where the tail comes from. Before allocation, most late failures came from one clip, a floor roll followed by a stand-up in jumps1\_subject2. Repairing reference frames, training 10% of episodes under deployment conditions, anchoring the policy (drift −78 to −84%), and adding real data at the failure points to the world model each left the failures unchanged. The test that separated the hypotheses is shown in Figure 5c. Trained alone, the clip fails in 16 of 16 rollouts up to iteration 1000 and in 0 of 128 from iteration 1250 on; in the mix, the 10 seconds before the hard spot receive 0.6–1.5% of resets. Forcing 10% of resets onto that window gives 0 of 256 failures at the six checkpoints from 1750 to 3000, while the same seed without the change fails 19%, 3.9% and 2.3% of the time at 2500, 2750 and 3000.

a  
![](images/315e7cb0c87e9c7cfd04cfe3e124c15cb84d573b63b1c7fc443564d48f675764.jpg)

![](images/a3b7e9dda15e97df3bf56f0e7ad805bd98d7ceb008ee30ab7a2239ace7842fda.jpg)

![](images/d3721b5e504d5eb9f71b25469dc623a0f30dd2715b39516d76ba29876c3b29be.jpg)  
Figure 5 | The tail is a practice problem. a, Failed rollouts per clip and checkpoint over iterations 1500–3500, summed over three seeds and four replicas (maximum 12 per cell), for recipe B without and with allocation; the tail counts in the text start at 2000. b, Share of a hard window in the reset distribution, the visited states and the imagination starts (one seed). c, Late failure rate of the hardest clip trained alone (iterations 1250–3000), in the 40-clip mix (mean over iterations 2500–3000), and in the mix with 10% of resets forced onto the window before the hard spot (iterations 1750–3000). One seed each.

Practice is lost on the way. An audit of one run traced a hard window from resets to imagination (Figure 5b). In steady state its share falls from 15.6% of resets to 10.3% of visited states and 9.0% of imagination starts, mostly because episodes started in the window run past it. When a failure is new, the bufer lags: 18.2% of resets become only 2.5% of imagination starts until the bufer turns over, about 400 iterations later.

Allocation in two recipes. We registered an ablation before running it on recipe B, which had a synchronous probe (Table 5): allocation is efective if tail failures fall by at least 40% on average and in at least two of three seeds. Tail failures fall from 23, 22 and 10 to 2, 10 and 2 (Figure 5a), so the criterion is met, and the three-checkpoint rule improves from 6.1 to 4.9 minutes. In the release recipe the other ingredients leave fewer late failures to remove: without the probe, 18 rollouts fail over three seeds, against 55 in recipe B without it, and the probe halves them to 9 (Section 5.2). The direction is the same, but at these counts three seeds no longer separate the two arms.

What did not help. Table 6 in the appendix lists the alternatives that did not shorten training, including critic variants, shorter or attenuated horizons, mixed gradient estimators, planning teachers, wider networks and prioritized replay.

## 6 Discussion and Limitations

What limits the recipe now. The fit phase takes more than 80% of the 3.4 minutes. It already runs at about the perfect-teacher rate, and the ablation shows that every fit-phase ingredient still matters, so further gains there must come from a better update rule, from less for the network to learn from scratch, or from cheaper updates. The clearest open problem is reliability. A checkpoint that is clean at one time can fail later (Section 5.1), and the ablation suggests tha the margin above the target, which the world-model structure raises, matters for this alongside the allocation of practice. We regard late flips, not mean speed, as the most valuable target for future work.

Scope of the evidence. We use one robot and one motion library, report training time on one GPU type, and do not report real-robot deployment. Three seeds cannot resolve diferences below about 15% in the strict rule, so we give every seed. In the final recipe this applies to the allocation: it halves late failures, as it reduced them in an earlier recipe, but the remaining counts are too small for three seeds to separate. The ablation keeps the settings tuned for the full recipe, so a removed ingredient may do better with settings of its own, and the structure and the mirror are removed together. Several diagnostics rest on one seed and were measured on earlier configurations of the recipe, as marked. The asynchronous probe assumes an otherwise idle GPU: when another process shares the GPU, training with it can crash, which the synchronous probe avoids at a cost in speed. The perfect-teacher ratio � is a reference and not a lower bound, since a diferent update rule changes both numerator and denominator. The analyses in Section 4 are idealized: they say what to measure and how to read it, and the design choices themselves rest on measurements.

Conclusion. BlinkMimic trains one policy for 40 LAFAN1 clips in 3.4 minutes on a single RTX 4090. Splitting the time into iteration cost, fit and tail shows what to work on: the update rule during the fit phase, the allocation of the real simulator during the tail, and the removal of host latency throughout. The split itself does not depend on the robot or the motion library, so the same accounting can be applied to other world-model training setups; whether the same ingredients matter there is a question for measurement, not something our single benchmark settles. We release the recipe and per-seed results to make such comparisons easy.

## References

Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, and Marc G. Bellemare. Deep reinforcement learning at the edge of the statistical precipice. In NeurIPS, 2021.

Florent Benaych-Georges and Raj Rao Nadakuditi. The singular values and vectors of low rank perturbations of large rectangular random matrices. Journal of Multivariate Analysis, 111:120–135, 2012.

Jacob Buckman, Danijar Hafner, George Tucker, Eugene Brevdo, and Honglak Lee. Sample-eficient reinforcement learning with stochastic ensemble value expansion. In NeurIPS, 2018.

Zixuan Chen, Mazeyu Ji, Xuxin Cheng, Xuanbin Peng, Xue Bin Peng, and Xiaolong Wang. GMT: General motion tracking for humanoid whole-body control. arXiv preprint arXiv:2506.14770, 2025.

Jeremy M. Cohen, Behrooz Ghorbani, Shankar Krishnan, Naman Agarwal, Sourabh Medapati, Michal Badura, Daniel Suo, David Cardoze, Zachary Nado, George E. Dahl, and Justin Gilmer. Adaptive gradient methods at the edge of stability. arXiv preprint arXiv:2207.14484, 2022.

Cody Coleman, Deepak Narayanan, Daniel Kang, Tian Zhao, Jian Zhang, Luigi Nardi, Peter Bailis, Kunle Olukotun, Chris Ré, and Matei Zaharia. DAWNBench: An end-to-end deep learning benchmark and competition. In NIPS ML Systems Workshop, 2017.

Ashok Cutkosky and Harsh Mehta. Momentum improves normalized SGD. In International Conference on Machine Learning, 2020.

Huy-Quang Dao. InstantTrack: Whole-body motion tracking for the Unitree G1 in minutes on one GPU. https: //github.com/Huy-Quang-Dao/instantTrack, v1.0–v1.2, 2026.

Marc Peter Deisenroth and Carl Edward Rasmussen. PILCO: A model-based and data-eficient approach to policy search. In ICML, 2011.

Ignat Georgiev, Krishnan Srinivasan, Jie Xu, Eric Heiden, and Animesh Garg. Adaptive horizon actor-critic for policy learning in contact-rich diferentiable simulation. In ICML, 2024.

Google DeepMind and NVIDIA. MuJoCo Warp: GPU-accelerated MuJoCo with NVIDIA Warp. https://github. com/google-deepmind/mujoco\_warp, 2025.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In ICLR, 2020.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.

Félix G. Harvey, Mike Yurick, Derek Nowrouzezahrai, and Christopher Pal. Robust motion in-betweening. ACM Transactions on Graphics, 39(4), 2020.

Tairan He, Zhengyi Luo, Xialin He, Wenli Xiao, Chong Zhang, Weinan Zhang, Kris Kitani, Changliu Liu, and Guanya Shi. OmniH2O: Universal and dexterous human-to-humanoid whole-body teleoperation and learning. In CoRL, 2024.

Shengding Hu, Yuge Tu, Xu Han, et al. MiniCPM: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. In NeurIPS, 2019.

Minqi Jiang, Edward Grefenstette, and Tim Rocktäschel. Prioritized level replay. In ICML, 2021.

Keller Jordan, Yuchen Jin, Vlado Boza, Jiacheng You, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An optimizer for hidden layers in neural networks. https://kellerjordan.github.io/posts/muon/, 2024.

Michael Laskey, Jonathan Lee, Roy Fox, Anca Dragan, and Ken Goldberg. DART: Noise injection for robust imitation learning. In CoRL, 2017.

Qiayuan Liao, Takara E. Truong, Xiaoyu Huang, Yuman Gao, Guy Tevet, Koushil Sreenath, and C. Karen Liu. Beyond-Mimic: From motion tracking to versatile humanoid control via guided difusion. arXiv preprint arXiv:2508.08241, 2025.

Qiayuan Liao, Yiyang Shao, Junfeng Long, Kevin Zakka, Roberto Horowitz, Pieter Abbeel, and Koushil Sreenath. Super-Mimic: Scaling humanoid whole-body control with one diferentiable world model. arXiv preprint arXiv:2610.28480, 2026.

Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, et al. Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982, 2025.

Zhengyi Luo, Jinkun Cao, Alexander Winkler, Kris Kitani, and Weipeng Xu. Perpetual humanoid control for real-time simulated avatars. In ICCV, 2023.

Sadhika Malladi, Kaifeng Lyu, Abhishek Panigrahi, and Sanjeev Arora. On the SDEs and scaling rules for adaptive gradient algorithms. In NeurIPS, 2022.

Peter Mattson et al. MLPerf training benchmark. In MLSys, 2020.

Sam McCandlish, Jared Kaplan, Dario Amodei, and OpenAI Dota Team. An empirical model of large-batch training. arXiv preprint arXiv:1812.06162, 2018.

Luke Metz, C. Daniel Freeman, Samuel S. Schoenholz, and Tal Kachman. Gradients are not all you need. arXiv preprint arXiv:2111.05803, 2021.

Andrew W. Moore and Christopher G. Atkeson. Prioritized sweeping: Reinforcement learning with less data and less time. Machine Learning, 13:103–130, 1993.

Paavo Parmas, Carl Edward Rasmussen, Jan Peters, and Kenji Doya. PIPPS: Flexible model-based policy search robust to the curse of chaos. In ICML, 2018.

Xue Bin Peng, Pieter Abbeel, Sergey Levine, and Michiel van de Panne. DeepMimic: Example-guided deep reinforcement learning of physics-based character skills. ACM Transactions on Graphics, 37(4), 2018.

Boris T. Polyak and Anatoli B. Juditsky. Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization, 30(4):838–855, 1992.

Stéphane Ross, Geofrey Gordon, and Drew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In AISTATS, 2011.

Leslie N. Smith. Cyclical learning rates for training neural networks. In WACV, 2017.

Hyung Ju Terry Suh, Max Simchowitz, Kaiqing Zhang, and Russ Tedrake. Do diferentiable simulators give better policy gradients? In ICML, 2022.

Chen Tessler, Yunrong Guo, Ofir Nabati, Gal Chechik, and Xue Bin Peng. MaskedMimic: Unified physics-based character control through masked motion inpainting. ACM Transactions on Graphics, 43(6), 2024.

Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, and Sham Kakade. SOAP: Improving and stabilizing Shampoo using Adam. arXiv preprint arXiv:2409.11321, 2024.

Jie Xu, Viktor Makoviychuk, Yashraj Narang, Fabio Ramos, Wojciech Matusik, Animesh Garg, and Miles Macklin. Accelerated policy learning with parallel diferentiable simulation. In ICLR, 2022.

Guodong Zhang, Lala Li, Zachary Nado, James Martens, Sushant Sachdeva, George E. Dahl, Christopher J. Shallue, and Roger Grosse. Which algorithmic choices matter at which batch sizes? insights from a noisy quadratic model. In NeurIPS, 2019.

## A Derivations

Notation. $z _ { t }$ is the augmented Markov state, $a _ { t } = \pi _ { \theta } ( z _ { t } ) , D _ { t } = F _ { z } , B _ { t } = F _ { a } , A _ { t } = D _ { t } + B _ { t } \pi _ { z }$ , and $x _ { t } = \partial z _ { t } / \partial \theta$ . A hat denotes quantities of the learned model along its own rollout.

## A.1 Sensitivity error of the world mode

Assume that both rollouts stay where $F , { \hat { F } }$ and $\pi _ { \theta }$ are diferentiable and start from the same state with zero sensitivity. Diferentiating gives $x _ { t + 1 } = A _ { t } x _ { t } + B _ { t } \Pi _ { t }$ and $\hat { x } _ { t + 1 } = \hat { A } _ { t } \hat { x } _ { t } + \hat { B } _ { t } \hat { \Pi } _ { t }$ . With $e _ { t } = \hat { x } _ { t } - x _ { t } , \Delta A _ { t } = \hat { A } _ { t } - A _ { t } , \Delta B _ { t } = \hat { B } _ { t } - B _ { t }$ $\Delta \Pi _ { t } = \hat { \Pi } _ { t } - \Pi _ { t }$ , subtraction gives $\boldsymbol { e } _ { t + 1 } = \hat { A } _ { t } \boldsymbol { e } _ { t } + \Delta A _ { t } \boldsymbol { x } _ { t } + \Delta B _ { t } \boldsymbol { \Pi } _ { t } + \hat { B } _ { t } \Delta \boldsymbol { \Pi } _ { t } ;$ unrolling from $e _ { 0 } = 0$ gives the expression in Section 4.4, with $\hat { \Phi } ( t , k ) = \hat { A } _ { t - 1 } \cdot \cdot \cdot \hat { A } _ { k }$

The last term cannot be dropped: take $F = a , \hat { F } = a + 1 , \pi _ { \theta } ( s ) = \theta s , s _ { 0 } = 1 , \theta = 1$ ; then $A = { \hat { A } }$ and $B = { \hat { B } } ;$ , yet the two-step sensitivities difer by 1. A small prediction error also does not imply a small gradient error: $\hat { F } = F + \varepsilon \sin ( z / \varepsilon ^ { 2 } )$ has function error � and derivative error $1 / \varepsilon$

## A.2 Semi-implicit structure

Suppose the simulator integrates $\nu _ { t + 1 } = \nu _ { t } + \Delta t f ( q _ { t } , \nu _ { t } , a _ { t } ) , q _ { t + 1 } = q _ { t } + \Delta t \nu _ { t + 1 }$ , and the model shares this step and learns only ${ \hat { f } } .$ . The Jacobian errors of $( q , \nu )$ then have the block form $\left[ \begin{array} { l } { \Delta t ^ { 2 } \delta } \\ { \Delta t \delta } \end{array} \right]$ with $\delta = \partial ( \hat { f } - f )$ , so position errors are a factor Δ� smaller than velocity errors. Our model adds a small learned position correction, so the structure is encouraged, not exact.

## A.3 Muon as steepest descent

For $\boldsymbol { G } = \boldsymbol { U \Sigma V } ^ { \top }$ , the minimizer of $\langle G , \Delta \rangle$ subject to $\| \Delta \| _ { \mathrm { o p } } \leq \eta \mathrm { i s } - \eta U V ^ { \top }$ (Jordan et al., 2024). Muon applies this step to the momentum matrix using an approximate orthogonalization.

## A.4 Batch knee under normalized steps

Write the Muon step as � polar(�) with polar $( M ) = U V ^ { \top }$ for $M = U \Sigma V ^ { \top }$ , and model the momentum as $M = { \bar { G } } + E$ with i.i.d. noise � whose per-entry variance scales as $1 / B$ . With a local quadratic model, one step decreases the loss by $\eta \langle \bar { G } , \mathrm { p o l a r } ( M ) \rangle - \frac { \eta ^ { 2 } } { 2 } \langle \mathrm { p o l a r } ( M ) , H \mathrm { p o l a r } ( M ) \rangle$ . Since polar(�) has unit singular values, we treat the curvature term as a constant � independent of $B ;$ this is the assumption that distinguishes the result from SGD, where noise inflates the step. Random-matrix results for spiked models (Benaych-Georges and Nadakuditi, 2012) give the overlaps. With $\tau ^ { 2 }$ the per-entry noise variance of �, $\theta _ { i } = s _ { i } / ( \tau \sqrt { q } )$ and $c = p / q$ for a $\mid p \times q$ matrix, $a _ { i } = 0$ for $\theta _ { i } ^ { 4 } \leq c$ , and otherwise

$$
\begin{array} { r } { a _ { i } ^ { 2 } = \left[ 1 - \frac { c ( 1 + \theta _ { i } ^ { 2 } ) } { \theta _ { i } ^ { 2 } ( \theta _ { i } ^ { 2 } + c ) } \right] \left[ 1 - \frac { c + \theta _ { i } ^ { 2 } } { \theta _ { i } ^ { 2 } ( \theta _ { i } ^ { 2 } + 1 ) } \right] . } \end{array}
$$

Noise directions contribute zero on average, so $\begin{array} { r } { \langle \bar { G } , \mathrm { p o l a r } ( M ) \rangle = \sum _ { i } { s _ { i } a _ { i } ( B ) } = : A ( B ) } \end{array}$ , where $\begin{array} { r } { \bar { G } = \sum _ { i } s _ { i } u _ { i } \nu _ { i } ^ { \top } } \end{array}$ and $a _ { i } \in [ 0 , 1 ]$ is the overlap of the step with signal direction �. Equal progress with per-step gain $\eta A ( B ) - C$ requires $S ( B ) \propto 1 / ( \eta A ( B ) - C )$ . With $f ( B ) = A ( B ) / A ( B _ { 0 } )$ and $\rho = C / ( \eta A ( B _ { 0 } ) )$ ), dividing by the value at $B _ { 0 }$ gives

$$
\frac { S ( B ) } { S ( B _ { 0 } ) } = \frac { 1 - \rho } { f ( B ) - \rho } ,
$$

which rises sharply once $f ( B )$ approaches $\rho .$ . We measured the singular spectra at $B _ { 0 } = 4 0 9 6$ and summed over layers. The model covers only directions detectable at $B _ { 0 } ,$ assumes � is independent of �, and is tested by a single registered prediction on one seed. With $\rho$ fitted to one fork, �(2048) is predicted to be 1.07, against a measured 1.006 for the single fork and 1.10 for the three-seed mean.

## A.5 Saturation of the tracking kernel

$\partial r / \partial e = - ( 2 e / \sigma ^ { 2 } ) \exp ( - e ^ { 2 } / \sigma ^ { 2 } )$ . Substituting $e = \sigma { \sqrt { x } }$ gives the magnitude $( 2 / \sigma ) \sqrt { x } e ^ { - x }$ , maximized at $\begin{array} { r } { x = \frac { 1 } { 2 } } \end{array}$ with value $( 2 / \sigma ) 0 . 4 2 9 . \mathrm { A t } x = 2 . 5$ the ratio to the peak is ${ \sqrt { 2 . 5 } } e ^ { - 2 . 5 } / 0 . 4 2 9 = 0 . 3 0 $ ; at $x = 5$ it is $\sqrt { 5 } e ^ { - 5 } / 0 . 4 2 9 = 0 . 0 3 5$

## A.6 Noise floor and averaging

For the noisy step $x _ { k + 1 } = ( 1 - \eta h ) x _ { k } + \eta \xi _ { k }$ with Var $\xi = \sigma ^ { 2 }$ , let $\phi = 1 - \eta h$ with $| \phi | < 1$ . The stationary variance � satisfies $V = \phi ^ { 2 } V + \dot { \eta } ^ { 2 } \sigma ^ { 2 }$ , so $V = \eta \sigma ^ { 2 } / ( h ( 2 - \eta h ) )$ and $\begin{array} { r } { \mathbb { E } L = \frac { 1 } { 2 } h V = \eta \sigma ^ { 2 } / ( 2 ( 2 - \eta h ) ) , } \end{array}$ . For the EMA $\begin{array} { r } { y _ { k } = \left( 1 - \beta \right) \sum _ { j \geq 0 } \beta ^ { j } x _ { k - j } } \end{array}$ use $\mathrm { C o v } ( x _ { k } , x _ { k + l } ) = V \phi ^ { | l | }$ to get

$$
\operatorname { V a r } y = ( 1 - \beta ) ^ { 2 } V \sum _ { i , j \geq 0 } \beta ^ { i + j } \phi ^ { | i - j | } = V \frac { ( 1 - \beta ) ( 1 + \beta \phi ) } { ( 1 + \beta ) ( 1 - \beta \phi ) } ,
$$

which is smallest relative to � when � is small, that is, along stif directions.

## A.7 Equal marginal gains

Minimize $\textstyle \sum _ { i } p _ { i } ( n _ { i } )$ subject to Í<sub>�</sub> $n _ { i } = R , n _ { i } \geq 0 .$ , with $p _ { i }$ convex and decreasing. The KKT conditions give $- p _ { i } ^ { \prime } ( n _ { i } ) = \mu$ for $n _ { i } > 0 \mathrm { ~ a n d } - p _ { i } ^ { \prime } ( 0 ) \leq \mu$ otherwise. For $p _ { i } ( n ) = p _ { i } ^ { 0 } e ^ { - n / \tau }$ the marginal gain is $p _ { i } ( n ) / \tau$ , so a greedy rule picks the bin with the highest failure probability. Transfer between clips violates separability, and consecutive hard spots violate convexity.

## A.8 Allocation by mismatched failure statistics

Let the reset weights be $w _ { i } \propto \left( 1 - \alpha \right) g ( \hat { p } _ { i } ^ { M } ) + \alpha / N$ with � increasing and $g ( 0 ) = 0 ,$ , where $\hat { p } _ { i } ^ { M }$ are failure estimates taken under a condition �, and let � be bins with failure probability at least � > 0 under � at all times and let the estimator keep $\hat { p } _ { i } ^ { M } \ge \delta ^ { \prime } > 0$ on �. If the remaining bins converge, $\dot { \hat { p } } _ { i } ^ { M }  0$ for � ∉ �, then $g ( \hat { p } _ { i } ^ { M } ) \to 0$ of � while $g ( \hat { p } _ { i } ^ { M } ) \geq g ( \delta ^ { \prime } ) > 0$ on �. The non-uniform part of � therefore concentrates on �, whose share tends to $\left( 1 - \alpha \right) + \alpha | S | / N$ A bin with $p _ { i } ^ { E } > 0 = p _ { i } ^ { M }$ has $g ( \hat { p } _ { i } ^ { M } ) \to 0$ and keeps only $\alpha / N$

## B Configuration

<table><tr><td>Simulator Replay World model Imagination Critic Policy</td><td>MuJoCo-Warp via mjlab, 50 Hz control, 4096 robots 1000 steps per robot (≈400 iterations); update-to-data 0.4 MLP 3 × 1024, semi-implicit, mirror 0.5, 5 batches of windows per iteration, Muon</td></tr></table>

## C Systems Details

Time in an iteration. Before the collection graph, an iteration at batch 2048 took 122 ms: policy and critic updates 70 ms, world-model updates 21 ms, and collection 31 ms. The GPU was fully idle for only 8–11% of the iteration, so overlapping simulation with learning on separate streams would gain little. Removing launch gaps gains more. Warp’s own step graph cannot be launched inside a PyTorch capture, so we capture Warp’s kernels directly into the PyTorch graph on a shared stream, and the constraint solver’s iteration loop becomes a conditional graph node. Episode ends are handled with fixed shapes: random reset frames are drawn for every robot and written only where the episode ended, which removes data-dependent indexing and host synchronizations.

Check of the reset path. We compare all simulator state fields after a masked reset with those after the original reset. They agree bit for bit except for joint velocities, which difer by at most $2 . 4 \times 1 0 ^ { - 7 }$

Time split of the release recipe. Over three seeds, the 3.38 minutes consist of about 9 s of startup and compilation, 167 s to the first checkpoint at return 0.82 and 26 s of tail.

D Tracking Accuracy

E Lineage of the Recipe

F Negative Results

Table 3 | Tracking accuracy of the release recipe. Three seeds, mean (range), each checkpoint rolled out four times under the evaluation protocol. Errors are averaged over all control steps of all 40 clips. Body position: RMS error of the 14 tracked bodies with the reference re-anchored to the robot root, as in the reward; root drift: global root position error, accumulated over clips of 3.7 minutes on average without any global position feedback; foot slip: horizontal foot speed in contact (the reference motion itself has 10.2 cm/s); contact agreement: share of foot frames whose contact state matches the reference.
<table><tr><td>Metric</td><td>At the target</td><td>Iteration 3500</td></tr><tr><td>Body position (cm)</td><td>4.10 (4.02–4.14)</td><td>3.45 (3.40–3.49)</td></tr><tr><td>Body orientation (deg)</td><td>8.1 (7.9–8.3)</td><td>6.9 (6.8–7.1)</td></tr><tr><td>Joint angle (deg)</td><td>4.0 (4.0–4.1)</td><td>4.2 (4.1–4.4)</td></tr><tr><td>Root drift (cm)</td><td>93 (89–97)</td><td>75 (73–77)</td></tr><tr><td>Foot slip (cm/s)</td><td>7.1 (7.0–7.2)</td><td>6.7 (6.6–6.7)</td></tr><tr><td>Contact agreement</td><td>0.747 (0.744–0.749)</td><td>0.764 (0.763–0.765)</td></tr><tr><td>Return</td><td>0.840 (0.837–0.843)</td><td>0.861 (0.858–0.863)</td></tr></table>

Table 4 | From InstantTrack v1.0 to BlinkMimic (RTX 4090, 40 × 4). Iterations and minutes to a mean return of 0.82, cost per iteration, and the iteration of the strict target (three consecutive clean checkpoints; per seed). <sup>†</sup>500-iteration grid. <sup>‡</sup>plus probe sweeps. From row A on, checkpoints every 100 iterations and a synchronous probe (about 20 ms) are included, so ms/it rises there although the graphs make each iteration cheaper (paired measurement in Section 4.6).
<table><tr><td>Configuration</td><td>seeds</td><td>it. 0.82</td><td>min</td><td>ms/it</td><td>strict it.</td></tr><tr><td>InstantTrack v1.0 (our run)</td><td>2+1</td><td>4997</td><td>15.3</td><td>195</td><td>6167 (19.6)†</td></tr><tr><td>+ semi-implicit WM, mirror</td><td>3+1</td><td>4066</td><td>≈12</td><td>190</td><td></td></tr><tr><td>+ Muon (policy, WM)</td><td>2</td><td>2598</td><td>7.6</td><td>194</td><td></td></tr><tr><td>+ reference residual, no warm-up</td><td>1</td><td>1732</td><td>5.8</td><td>195</td><td>2750 (9.1)</td></tr><tr><td>+ sampler/contact fixes (nb)</td><td>3</td><td>1587</td><td>5.5</td><td>195</td><td>2750 / 2500 / –</td></tr><tr><td>+ update graphs, allocation</td><td>3</td><td>1549</td><td>4.9</td><td>172</td><td>2500 / 3000 / 3000</td></tr><tr><td>+ batch 2048</td><td>3</td><td>1701</td><td>4.0</td><td>122</td><td>2500 / 2250 / 2750</td></tr><tr><td>+ graph capture of the whole iteration (A)</td><td>3</td><td>1742</td><td>4.0</td><td>142</td><td>2800 / 2900 / 2200</td></tr><tr><td>+ step-size schedule (B)</td><td>3</td><td>1407</td><td>3.2</td><td>141÷</td><td>2100 / 2500 / 1800</td></tr><tr><td>+ probe fixes, cutoff at 1400 (BlinkMimic)</td><td>3</td><td>1306</td><td>2.9</td><td>135÷</td><td>1500 / 1500 /1600</td></tr></table>

Table 5 | Earlier recipes on the RTX 4090 (same protocol as Table 1). A: whole-iteration graphs, constant step size; B: A with the progress-triggered step size; C: B without allocation. The probe of A–C was synchronous, started at iteration 1250 and swept every 250 iterations. 6 clean checkpoints: a stricter variant of the target rule; “–”: not reached within 3500 iterations.
<table><tr><td>Configuration</td><td>ms/it</td><td>Iterations to 0.82</td><td>First ≥0.82</td><td>3 clean checkpoints</td><td>6 clean checkpoints</td><td>Tail failures Final return</td><td></td></tr><tr><td>No allocation (C)</td><td>121</td><td>1439 / 1302 / 1479</td><td>3.2 / 3.0 / 3.2 (3.1)</td><td>6.4 / 6.2 / 5.6 (6.1)</td><td>-/-/ 5.6 (−)</td><td>23 / 22 / 10</td><td>0.862</td></tr><tr><td>Constant step size (A)</td><td>142</td><td>1778 / 1679 / 1768</td><td>4.2 / 3.9 / 4.3 (4.1)</td><td>6.6 / 6.7 / 5.1 (6.1)</td><td>6.6 / 6.7 / 5.1 (6.1)</td><td>9/14 /5</td><td>0.862</td></tr><tr><td>Progress-triggered step (B)</td><td>141</td><td>1344 /1331 / 1547</td><td>3.2 / 3.2 / 3.7 (3.3)</td><td>4.8 / 5.7 / 4.1 (4.9)</td><td>4.8 / 5.7 / 4.1 (4.9)</td><td>2/10/2</td><td>0.861</td></tr><tr><td>BlinkMimic (release)</td><td>135</td><td></td><td>1292 / 1456 / 1169 2.89 / 3.28 / 2.64 (2.94) 3.32 / 3.28 / 3.53 (3.38)</td><td></td><td>5.36 / –1 3.53 (→)</td><td>2/6/0</td><td>0.861</td></tr></table>

Table 6 | Interventions that did not shorten training. Iterations to 0.82 relative to the configuration they were added to, one seed unless stated.
<table><tr><td>Intervention</td><td>Effect</td><td>Reading</td></tr><tr><td>Batch ×2, ×4 (teacher regression)</td><td>-3% updates</td><td>noise is not limiting</td></tr><tr><td>Prioritized WM windows; prioritized imagi- within seed noise nation starts</td><td></td><td>reweighting changes variance only</td></tr><tr><td>4 variants</td><td>wall</td><td>Planning (CEM) teacher + behavior cloning, no strict-target speed-up; 1.6–1.7× κ ≈ 1; the student does not learn a margin</td></tr><tr><td>CEM data at failure points into the world cliff error -21 to -48%; policy model accuracy at failures is not the limit model</td><td>worse where taught</td><td></td></tr><tr><td>Critic slope supervision; critic variants Mixed pathwise/likelihood-ratio estimator</td><td>no gain; 2× slower per iteration</td><td>slope is not limiting offline λ = 0.01–0.06; zeroth- pathwise variance does not grow exponentially</td></tr><tr><td>λ-return actor; gradient decay; H ∈ {16, 24}</td><td>order term 100–1600× noisier +27% to neutral</td><td>trades model error for critic error</td></tr><tr><td>No critic</td><td>+50%, never reaches 0.84</td><td>the value beyond H is needed</td></tr><tr><td>Width ×2</td><td>-22% iterations, +27% wall</td><td>cost grows faster than updates fall</td></tr><tr><td>Policy 1024-512-256</td><td></td><td>+26% iterations, equal wall, more less capacity for the hardest motions</td></tr><tr><td>WM 5→2 batches, critic 5→2 epochs</td><td>tail failures (2 seeds) -25% cost per iteration; 0.82 not proxy training sets signal quality</td><td></td></tr><tr><td>Lookahead (k = 10, α = 0.5)</td><td>reached by 2000 +60%</td><td></td></tr><tr><td>Cosine step decay to 4000 iterations</td><td>+3%</td><td>parameter averaging in the loop hurts a fixed schedule ignores progress</td></tr><tr><td>Sharpness-weighted loss; Fourier features +17 to +21%</td><td></td><td>spectral bias is unchanged by reweighting</td></tr><tr><td>(teacher) Training-condition failure resets ×1.7; sam- tail unchanged</td><td></td><td>wrong measurement condition (Appendix A.8)</td></tr></table>

## G Full Experiment Record

Table 7 | All learning experiments behind this paper. Iterations to 0.82 on 40 × 4 unless stated; “1e” = single evaluation (about ±500 iterations of noise). Configurations as in Table 4. Seeds: number of training seeds.
<table><tr><td>Experiment</td><td>Seeds</td><td>Result</td><td>Reading</td></tr><tr><td colspan="4">Fit phase: signal side</td></tr><tr><td>Semi-implicit WM (on v1.0)</td><td>3 3+1</td><td>-13%</td><td>smaller Jacobian error</td></tr><tr><td>+ mirror augmentation (smir)</td><td></td><td>-18.6% for both together (4066 vs. 4997)</td><td>symmetry prior; WM direction cosine 0.73→0.80</td></tr><tr><td>WM Muon (alone 7 smir+policy Muon)</td><td>1/2</td><td>-21.9% / -48% total</td><td>WM loss at 500 halved</td></tr><tr><td>Reference residual smir+Muon)</td><td></td><td>-22% iterations, -27% updates</td><td>RL-only gain (teacher unchanged)</td></tr><tr><td>No WM-only warm-up</td><td></td><td>-11% iterations, +4% wall to 0.82</td><td>kept for simplicity</td></tr><tr><td>Oracle (late) world model, warm- up kept</td><td></td><td>-4%; strict unchanged</td><td>map quality not limiting</td></tr><tr><td>No critic</td><td></td><td>+50%; 0.84 never reached</td><td>critic value needed</td></tr><tr><td>Critic: alive-only / bowl prior / slope supervision</td><td>1 each</td><td>+4% / -6% / -3% at half speed</td><td>critic slope not limiting</td></tr><tr><td>λ-return actor; gradient decay</td><td>(1e) 1 (1e)</td><td>+27%; -8% with lower return</td><td>chain shortening trades errors</td></tr><tr><td>0.95 Imagination action noise 0.1</td><td>1 (1e)</td><td>±0</td><td></td></tr><tr><td>Mixed pathwise/LR estimator (offline)</td><td>3 ckpts</td><td>λ = 0.013–0.064; ZO term 100–1600×</td><td>no variance blow-up to fix</td></tr><tr><td>Critic state-gradient training for</td><td>2 ckpts</td><td>noisier; one-step gain negative ∂V correlation ≤ 0.41 (needed 0.5)</td><td>rejected</td></tr><tr><td>H = 8 (offline) Planning teacher (CEM) + BC,</td><td>1 each</td><td>strict not faster; 1.6–1.7× wall</td><td>teacher passes, student lacks margin</td></tr><tr><td>4 variants CEM data into WM</td><td></td><td>cliff error -11 to -55%; policy worse</td><td>better map ≠ better policy</td></tr><tr><td>FK/contact inputs to WM (of- 2 fline)</td><td></td><td>where taught violent-moment error -12 to -22% (needed -50%)</td><td>rejected</td></tr><tr><td>Fit phase: step side Policy Muon (on v1.0)</td><td></td><td>-36%; teacher regression 3494→1478</td><td>update geometry</td></tr><tr><td>SOAP vs. Muon (teacher regres- sion)</td><td></td><td>1505 vs. 1478</td><td>preconditioning exhausted</td></tr><tr><td>Two updates per iteration (teacher regression)</td><td></td><td>-38% iterations, +24% updates</td><td>update-bound</td></tr><tr><td>Teacher regression, 5/10/20/40 clips</td><td></td><td>998 / 1336 / 1427 / 1499 updates</td><td>time goes to shared skills</td></tr><tr><td>Width ×2 (teacher / RL)</td><td>2/1</td><td>-27% updates / -22% iterations, +27%</td><td>capacity helps per update, not per second</td></tr><tr><td>Width ×0.5 (teacher)</td><td></td><td>wall +87%</td><td></td></tr><tr><td>Policy 1024-512-256 (RL)</td><td></td><td>+26% iterations, equal wall, tail failures 8/13 vs. 2/5</td><td>capacity for hard motions</td></tr><tr><td>Residual-LN, trunk, Fourier fea- 1 each</td><td></td><td>±0 to +21%</td><td>network form not limiting</td></tr><tr><td>tures, sharpness weighting Lookahead (k=10, α=0.5)</td><td>1</td><td>+59%</td><td></td></tr><tr><td>LR ×2 (teacher/RL, batch 2048)</td><td>1/1</td><td>-12% / -25%</td><td>step size has headroom</td></tr><tr><td>LR screen ×2, 3, 4, 6,8 (750 it)</td><td>1 each</td><td>return at 500: 0.606 / 0.639 / 0.659 /</td><td>ceiling ×4</td></tr><tr><td>Cosine LR to 4000</td><td></td><td>0.630 / 0.603 +3%, strict unchanged</td><td>fixed schedule</td></tr><tr><td>LR ×4, progress-triggered decay</td><td>1 (+3</td><td>1349 vs. 1833 (-26%)</td><td>see Section 5.3</td></tr><tr><td></td><td>in Ta- ble 5)</td><td></td><td></td></tr><tr><td>Batch and cost Batch 2048</td><td>3</td><td>+6/+7/+16% iterations, -17% wall</td><td>above the knee</td></tr><tr><td>Batch 1024 / 512</td><td>1/1</td><td>+21% / +45% (512 predicted 1.44– knee; more late failures 1.66×)</td><td></td></tr><tr><td>Policy gradient on 1024, critic 1 on 4096</td><td></td><td>+17%</td><td>knee from policy-gradient noise</td></tr><tr><td>WM 2 batches, critic 2 epochs Batch ×2/×4 (teacher regres- 1</td><td>1</td><td>-25% cost, 0.82 not reached by 2000 -3%</td><td>proxy training sets signal quality not noise-limited</td></tr><tr><td>sion) Tail phase</td><td></td><td></td><td></td></tr><tr><td>Hardest clip trained alone</td><td>1</td><td>16/16 failures at 750–1000; 0/128 at 1250– learnable 3000</td><td></td></tr><tr><td>Forced 10% reset share before 1 its hard spot</td><td></td><td>0/256 at 6 checkpoints vs. 19% / 3.9% / tail = practice share 2.3%</td><td></td></tr><tr><td>Reference floor repair; deploy- ment conditions in 10% of train-</td><td>1 each</td><td>no improvement</td><td>Appendix A.8</td></tr><tr><td>ing; sampler cap; failure resets ×1.7 Policy anchoring to EMA</td><td></td><td>drift -78 to -84%, failures unchanged</td><td></td></tr><tr><td>Deployment-condition alloca- 3 tion (vs. none)</td><td></td><td>see Section 5.4</td><td>drift not the cause</td></tr><tr><td>Probe share 0.3 / 0.5 instead of 3 / 6 0.15 (release recipe)</td><td></td><td>3-checkpoint rule -8% / –6%; iterations within seed spread; not adopted to 0.82 −10% / −10%</td><td></td></tr><tr><td>Release recipe Leave-one-out ablation, 7 rows</td><td></td><td>see Table 2</td><td></td></tr><tr><td>Step-size cutoff at 1000 / 1200</td><td>3 each 3/3</td><td>iterations to 0.82 -3% / -6%</td><td>within seed spread</td></tr><tr><td>instead of 1400 Reference inverse-dynamics</td><td>3/3</td><td>iterations to 0.82 -6% / -8%</td><td>within seed spread; not adopted</td></tr><tr><td>feedforward on the residual (s = 1 / 0.5)</td><td></td><td></td><td></td></tr></table>