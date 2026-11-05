# Catastrophic Overfitting is Weight Drift the Training Loss Cannot See

Chao Pan

Department of Computer Science and Engineering,

Southern University of Science and Technology, Shenzhen 518055, China

11930665@mail.sustech.edu.cn

Abstract. Fast adversarial training (FAT) makes robust training afordable by replacing the multi-step attack with a single-step one, but it sufers from catastrophic overfitting (CO): partway through training, robustness to multi-step attacks collapses to near zero while the single-step attack used for training keeps being repelled. We show that CO is a quiet escape of the weights. Across the event the network’s clean behavior quietly improves, and moving from the weights just before CO to those just after it changes the training objective only marginally while destroying most of the robustness. The training objective is therefore nearly blind to the displacement it causes, so it exerts no force to undo it. The exponential moving average (EMA) of the weights, which FAT protocols already maintain, does not accumulate this drift and stays robust through the entire event. This observation suggests a nearly free fix: when validation robustness drops, reset the weights to their EMA. We build this guard on top of a coupling-based base method designed so that its average stays inside the robust region. The resulting training removes CO in every one of our runs and attains the strongest AutoAttack accuracy among fast methods on all three benchmark datasets, including a clear margin on CIFAR-100, where it also clears the multi-step reference at roughly half its cost. Code is available at https://github.com/fzjcdt/co-weight-drift.

## 1 Introduction

Adversarial training [15] is the most reliable known defense against adversarial examples, but it repeats a multi-step attack for every batch, an order of magnitude more computation than ordinary training. Fast adversarial training (FAT) [7, 22, 27] removes this bottleneck by training against a single-step attack. It works well, with one strange exception.

The exception is catastrophic overfitting (CO) [2, 12, 27]. Partway through an otherwise ordinary run, robustness to multi-step attacks collapses to near zero within a single epoch. The single-step FGSM attack used for training continues to be repelled, so from the inside of the training loop nothing appears to be wrong, and the run finishes with a model that is robust against nothing except the attack it trained on. Because CO is silent and cannot be predicted in advance, the strongest published fast methods are chosen as much for their luck as for their design.

Many fixes have been proposed: randomized initialization [27] and gradient-alignment regularizers [2], modified attacks and initializations [5, 12, 17], bi-level formulations [26], and others [9, 29]. They help empirically, but they all act on the attack or on the loss, and none of them answers the mechanistic question: where in weight space has the CO model gone, and why is the optimization that was robust an epoch earlier unable to find its way back?

Part of the reason nobody has answered it is that ordinary training logs cannot see the event: at epoch resolution CO is a clif with no consistent precursor. So we rebuilt the instrument. We retrained the classic victim, vanilla FGSM adversarial training, and every 100 batches saved two checkpoints: the training weights and their exponential moving average (EMA), an object the protocol already maintains. This turns a oneepoch mystery into a short film.

![](images/503d64a2b8a9e394b13bd2995bab8b256ed8e06822d35ca2017c82f104537cac.jpg)

![](images/60c0e8317a140093ba4ada7de0b959e5ecdbc19dbbe08a47775014d10a18e4ff.jpg)  
Figure 1: Catastrophic overfitting (CO) at two timescales. Top: at epoch resolution, CO in vanilla FGSM training is a clif without warning; validation PGD-10 robustness falls from 45% to 0.5% in a single epoch (shaded), while the FGSM attack used for training keeps being repelled, so nothing inside the training loop looks wrong. Bottom: the same method retrained with checkpoints every 25 batches. The clif is 75 batches wide, the EMA model �¯ (blue) holds its robustness throughout, and clean accuracy keeps rising. Our guard resets the weights to this EMA when validation robustness dips.

Three findings came out of watching that film (Fig. 1). First, CO barely touches the clean function: through the event, clean accuracy rises while PGD-10 robustness falls from 44% to 11%. Robustness leaves before accuracy notices. Second, the direction of travel is invisible to the training objective:

interpolating from the pre-CO EMA to the CO weights changes the training loss by a few percent while destroying more than two thirds of the robustness, so there is no restoring force along the direction that matters. Third, the weight average resists the drift: the EMA keeps its robustness through the entire event, and in the benchmark’s released runs its own CO, when it happens at all, arrives many epochs later or never.

These three facts point to a fix that is not another regularizer. If training cannot correct the drift, but the trajectory’s own average does not accumulate it, the natural move is to pull the weights back toward their EMA when the drift shows up. We build this idea in its simplest form: a guard that resets the weights to their EMA when validation robustness drops by more than a threshold, sitting on top of a strong coupling-based base method designed so that its EMA holds the robust region. On CIFAR-10 the unguarded base collapses in half of its runs; the guarded version removes the coin flip entirely and reaches the best fast-method AutoAttack accuracy we are aware of on all three benchmark datasets, including a clear gain on Tiny-ImageNet.

Contributions. (1) A measured mechanism: CO is a drift of the weights along directions the training objective can barely see, established by paired raw/EMA tracking at 100-batch resolution and by line searches along the drift direction (Figs. 2, 3). (2) Its dynamics: a short warning window that no epoch-level observable can see, ending in gradient masking (Fig. 4), and a staged, margin-ordered sweep through the evaluation set. (3) A fix derived from the mechanism: anchoring the weights to their own EMA, with a threshold chosen inside a measured gap and a detector-free control showing that the rescue comes from the anchor itself. (4) Results: the strongest fast-method AutoAttack accuracy on CIFAR-10 (50.45±0.20, 4 seeds, no CO), CIFAR-100 (27.15±0.04, +0.51 over the strongest fast method and +0.25 over the multi-step reference), and Tiny-ImageNet (+0.59); the benchmark’s second strong-attack suite (CR) confirms the ordering on all three.

## 2 Related Work

Catastrophic overfitting. CO was identified with single-step adversarial training in [27]. Later work attributed it to the degradation of the single-step attack and proposed randomized initialization [27] and gradient-alignment regularizers [2], attack and initialization modifications [5, 12, 17], bi-level formulations [26], and self-fitting analyses [9]. All of these act on the attack or on the loss, and each new fix has been followed by the failure reappearing in a new form. Our analysis acts on the weights instead: it locates the CO solution relative to the robust one and shows that the direction between them is nearly flat for the training objective, which explains both why attack-side interventions help without explaining and why the failure keeps returning.

Fast adversarial training. Single-step methods with consistency or spectral couplings define the current FAT frontier: FGSM-PGI [11], N-FGSM [5], LIET [18], NuAT [24], GAT [23], SSAT [12], and the 21-method benchmark [16] whose protocol and released results we reuse throughout. We differ in what the couplings are for: previous methods use them to prevent CO, while we use them to keep the weight average inside the robust region so that a reset has somewhere to return to, and we separate that role from the guard that performs the reset. All baseline numbers in this paper are the benchmark’s released results; nothing is retrained for comparison.

Weight averaging. EMA and SWA [10, 21] average weights after or alongside training; Lookahead [28] unconditionally pulls fast weights toward slow weights; EWC [13] anchors to old tasks by Fisher importance. We use the average as a live control target, gated by a robustness measurement rather than applied unconditionally, and the diference matters: an always-on anchor at moderate pull strength destroys training, because it also pulls back the component the optimizer is legitimately optimizing (Appendix C).

## 3 Anatomy of Catastrophic Overfitting

## 3.1 What we trained and what we saved

We retrain the classic victim, vanilla FGSM adversarial training (FGSM-AT in the benchmark of [16]), under the benchmark’s exact protocol (ResNet18, CIFAR-10, �=8/255; Appendix A). Every 100 batches we save two checkpoints together: the raw training weights � and their EMA �¯ (decay 0.9995, the value the benchmark protocol itself uses). All evaluations in this section recalibrate BatchNorm statistics first; without this step the entire efect disappears (Appendix A). On this setup CO lands at epochs 19–21 across three seeds, and we track both checkpoints through the window.

## 3.2 Robustness leaves before accuracy notices

Fig. 2 shows the paired timeline. Between the last healthy checkpoint and the first CO one, 100 batches apart, the raw weights change dramatically for robustness and hardly at all for everything the training loop can observe. Clean accuracy moves from 73% to 80%. The cosine between the trainingattack direction and a PGD-10 direction falls from 0.82 to 0.00. Over the same interval, PGD-10 robustness falls from 44% to 11%. Robustness leaves before accuracy notices, and nothing the optimizer monitors moves in a way that would call the drift back.

The EMA tells the other half of the story. Evaluated on the same grid, it holds PGD-10 robustness at 44–46% and its gradient cosine at 0.75–0.83 for the entire event. This is a general property, not an artifact of one run: across the benchmark’s released per-epoch curves, the EMA’s own CO arrives 6–24 epochs after the raw CO in 4 of the 11 CIFAR-10 CO runs and never within training in the other 7 (Fig. 6 in the appendix). The drift reaches the average only slowly.

![](images/2318afdb4e3e403fae73a53851e7217fbe763854b0ed6737dfecb57c603e1841.jpg)

![](images/2cffeff4d4f056a262bd1721dba6db82f221264b59b7487e56fe81fd9d7120bd.jpg)

![](images/7f2ae8f09face1508751db5bbd79c7603601e73f7ec13922329e9fa7db5a4dce.jpg)  
Figure 2: What collapses and what does not. Every 100 batches through the CO window we evaluate the raw weights � (red) and their EMA �¯ (blue), BatchNorm recalibrated; the gray band marks the 100 batches in which CO happens. Top: the weights’ PGD-10 accuracy falls from 44% to 11% while the EMA holds 44–46%. Middle: over the same interval the clean accuracy of the weights rises from 73% to 80%. Bottom: the cosine between the training-attack direction and a PGD direction falls from 0.82 to 0.00 in the weights and stays high in the EMA.

## 3.3 A flat path for training, a clif for robustness

To make “invisible” precise, let L be the training objective (the empirical risk under the training attack) and � the PGD-10 robustness. We call a direction � blind if moving along it leaves L nearly unchanged while destroying �:

$$
| { \mathcal { L } } ( w + \alpha \nu ) - { \mathcal { L } } ( w ) | \leq \varepsilon \alpha , \quad | R ( w + \alpha \nu ) - R ( w ) | \geq \delta \alpha ,\tag{1}
$$

with $\delta / \varepsilon \gg 1$ . The term is operational: it means flatness by measurement, normalized relative to each function’s own level, and makes no claim about null spaces of Hessians or NTK kernels.

We then measure how blind the actual drift is (Fig. 3). We take 12 (EMA, raw) checkpoint pairs through the CO window, one every 100 batches, and for each pair interpolate $w ( \alpha ) =$ $\bar { w } + \alpha ( w - \bar { w } )$ from the EMA (�=0) to the weights (�=1), recomputing the attack at every �. The training objective varies within ±7.5% of its level along the entire path, and the CO endpoint is not even an improvement. Over the same path, robustness loses 69–77%, with a knee at $\begin{array} { r } { \alpha \approx 0 . 6 – 0 . 8 \colon } \end{array}$ the sensitivity gap along this single direction is roughly 10×.

There is no barrier between the robust and the CO solutions; they are connected by a path that is flat for training and a clif for robustness. The model did not find a better minimum; it wandered sideways of a plateau.

## 3.4 Why training cannot undo it

The measurements above compress into three short statements about the dynamics. We state them informally, with assumptions inline; each is backed by a measurement in the preceding sections.

Proposition 1 (no restoring force along blind directions). If � satisfies Eq. (1) at �, the expected SGD update along � is $\mathbb { E } \langle \Delta w , \nu \rangle = - \eta \langle \nabla \mathcal { L } , \nu \rangle = O ( \eta \varepsilon )$ : the dynamics exert essentially no force on the blind component, which instead drifts under gradient noise.

Proposition 2 (the time-average compresses the drift). Ifthe blind displacement is stationary with no persistent bias, its running average satisfies E∥ $\lvert \bar { \nu } _ { T } \rvert \rvert ^ { 2 } = O ( \sigma ^ { \bar { 2 } } / T )$ while the raw displacement grows as $\mathbb { E } \Vert \nu _ { T } \Vert ^ { 2 } = O ( \sigma ^ { 2 } T )$

Proposition 3 (resetting to the average is safe). Replacing � by �¯ keeps the visible component of the trajectory and discards the accumulated blind displacement; by Eq. (1) the replacement perturbs L by at most �(�) (measured: +6 points of clean accuracy) while restoring � toward its pre-CO level.

Together they say: training cannot correct the drift, the average does not accumulate it, and resetting to the average is cheap for the training objective. That is exactly the fix we build in Sec. 4.

Blind directions are not exotic. A perturbation � of the last layer that increases the logit margin $z _ { \mathrm { y } } - \operatorname* { m a x } _ { k \neq \mathrm { y } } z _ { k }$ without changing any argmax preserves clean predictions exactly; the softmax gradient of the cross-entropy decays exponentially in the margin, so $\langle \nabla \mathcal { L } , \nu \rangle \to 0 .$ , and the saturated logit surface creates a flat region in which few-step attacks stall while longer attacks still find the true boundary. This is the gradient masking we measure next. The assumption of no persistent bias (Proposition 2) draws the method’s applicability boundary: it holds for coupled FAT methods, whose EMA holds the robust region through the event, and fails for vanilla FGSM training, whose biased drift drags the average along; anchoring indeed fails there (Appendix C).

## 3.5 The last warning window

Why has this geometry gone unnoticed? Because at epoch resolution the event broadcasts nothing reliable: none of the epoch-level statistics we tried fires consistently before the clif. At 50-batch resolution the picture changes (Fig. 4). The cosine between the training-attack direction and a PGD direction decays from 0.79 to 0.32 over 200–300 batches, a soft ramp rather than a clif, and only then does PGD-10 robustness fall. This ramp is the only warning window that exists, and every per-epoch observable misses it.

![](images/51efeda91b2ab969bdffb63ef008d6ac463d81c27494b1266ee6d97509ca8944.jpg)  
Figure 3: Moving along the drift direction, $w ( \alpha ) = \bar { w } + \alpha \left( w - \bar { w } \right)$ , for 12 (EMA, weights) checkpoint pairs sampled every 100 batches through the CO window, with the training attack recomputed at each �; each curve runs from the EMA at �=0 (left) to the weights at �=1 (right). Gray: the 6 pairs from before CO. Red, darkening with time: the 6 pairs from after CO. Left: the training objective varies within ±7.5% of its level along every path, and the CO endpoint is not a better training loss. Right: along the paths after CO, PGD-10 loses $6 9 - 7 7 \%$

![](images/3b0628a78e59dfb1fadbf3671fd8cd7be4779ef3ea4c6b2a269527cb0e166c03.jpg)  
Figure 4: CO at batch resolution. The cosine between the trainingattack direction and a PGD direction (three seeds, smoothed; faint trace: raw values of one seed) decays over 200–300 batches before the robustness clif. At epoch resolution the same event is a featureless clif.

After the ramp, the model fakes robustness to weak attacks. PGD-3 (three steps of 2/255) reports robust accuracy rising to 86% while PGD-10 destroys the model, and the FGSM probe accuracy rises as well. This is textbook gradient masking [3], now measured batch-by-batch as the end state of CO: the training attack has become a no-op, so the method trains on efectively clean data while believing it trains adversarially. Two practical consequences follow: weak iterative attacks are unreliable probes of CO, and evaluation protocols that mix attack budgets can mis-rank models across the transition (precursor statistics in Appendix D).

## 3.6 The clif, sample by sample

The aggregate clif is 75 batches wide, which invites the picture of a simultaneous global event. The samples tell a diferent story. Re-evaluating the checkpoints of the window on the 1,000 validation images, with the attack seed fixed so that every checkpoint sees the identical PGD-10 initialization, we call a sample dead at the first checkpoint where it stays fragile for three consecutive checkpoints. Deaths are staged: a cohort of 16% of the initially robust samples dies 100 or more batches before the median, so the aggregate clif is the visible end of a slower sweep (Fig. 7 in the appendix). The death order is written in the healthy model: early deaths have a smaller clean logit margin at the window’s first checkpoint, and the order is a property of the samples rather than of one training path, since an independently trained replica reproduces it (cross-replica Spearman $\rho { = } 0 . 4 8 $ full statistics in Appendix D). CO sweeps the evaluation set in a largely preserved order, starting from the samples whose correct-class margin was smallest all along. This structure does not yield an earlier trigger; its value is to say what the drift destroys first, and to ground the clif shape in a sweep rather than a simultaneous global event.

## 4 Pulling the Weights Back to Their Average

The three propositions point to a simple rule rather than a new regularizer. The drift cannot be corrected by training. The average does not accumulate it. Resetting is nearly free for the training objective. So: watch for the drift, and when it appears, reset the weights to their own EMA (Eq. 4). Before the control itself, we specify the training it sits on top of, because the base method’s properties are what make anchoring possible at all.

## 4.1 A base method built for anchoring

The guard needs something to guard: a training configuration whose EMA actually holds a robust region. The configuration is dictated by what the coupling terms must measure. Write $x _ { a } = x + \delta _ { a }$ for the adversarial example of a batch. The perturbation $\delta _ { a }$ is what the training attack itself produces at that step: it starts from a fresh random initialization $\delta _ { 0 }$ (random half-sign at $0 . 5 \varepsilon ,$ , following NuAT [24]), takes one sign step of the cross-entropy loss evaluated at $x { + } \delta _ { 0 }$ , and is clipped to the threat model, $\| \delta _ { a } \| _ { \infty } \leq \varepsilon$ (Algorithm 1, lines 3–4). It is an input-space pattern, recomputed for every batch, not a weight-space direction. Writing the clean and adversarial logits as $z _ { c } = f ( x )$ and $z _ { a } = f ( x + \delta _ { a } )$ , the coupling $\| z _ { c } - z _ { a } \| _ { 2 } ^ { 2 }$ is a finite-diference estimate of the directional derivative of � along the adversarial displacement, anchored at the clean input �. The anchor matters: � lies on the data manifold and does not move with the weights, so the estimate is made where the labels live and is statistically independent of the current parameters. Anchoring elsewhere, for example at the attack’s initialization point $x { + } \delta _ { 0 }$ , makes the penalty self-referential: it rewards flattening exactly the gradient direction the attack uses, a masking feedback loop that we measured and that destroys the robust region (Appendix B). The clean branch is not regularization; it is information.

The full objective couples the two branches at two scales:

$$
\mathcal { L } \ = \ \mathrm { C E } _ { \mathrm { l s } } ( z _ { c } , y ) + \frac { \gamma ( t ) } { B } \left\| Z _ { c } - Z _ { a } \right\| _ { * } + m ( t ) \left\| z _ { c } - z _ { a } \right\| _ { 2 } ^ { 2 } ,\tag{2}
$$

where $Z _ { c } , Z _ { a }$ stack the batch’s clean and adversarial logits, $\| \cdot \| _ { * }$ is the nuclear norm (sum of singular values), � is the batch size, $\mathrm { C E _ { l s } }$ is label-smoothed cross-entropy, and �(�), �(�) ramp linearly over training (values per dataset in Sec. 5). The two coupling terms act at diferent scales and cover each other’s failure modes. The MSE term follows the per example consistency idea of FGSM-PGI [11], which penalizes the distance between the current adversarial prediction and a prior-guided initialization, but anchors the penalty at the clean input: it aligns each image’s clean and adversarial logits, and a setup with only this term is stable but plateaus below the frontier. The nuclear term, the idea behind NuAT [24], acts on the batch as a matrix: it penalizes concentration of the clean-minus-adversarial logit diference into a few directions, and alone it is toxic at the same strength on the harder datasets, exactly where the published nuclear-norm method fails (Appendix B). Together the setup is the strongest fast configuration we measured when it survives. Both terms need the clean logits in the computation graph, so each step carries three forward passes; we also tried to recover the clean branch at two-pass cost and failed in every arrangement we could think of, which we read as the price of true-label supervision on the data manifold (Appendix B).

Two properties of this setup matter for the rest of the paper. First, it is strong but drift-prone: on CIFAR-10 half of its runs sufer CO, which makes it the honest testbed for the anchor, since a method that never collapsed would need no guard. Second, its drift carries no persistent bias, the assumption of Proposition 2: the coupling terms suppress the biased drift of vanilla FGSM training, which is why its EMA holds the robust region through a CO event. The base and the anchor are designed together; neither works as well alone.

Algorithm 1 Guarded coupled FAT (one training run). Lines 3–   
4 define $\delta _ { a } \colon$ a fresh half-sign initialization, one sign step,   
clipped to the threat model. The guard reads only the validation   
PGD-10 the protocol already computes.   
1: �¯ ← �; �<sup>★</sup> ← 0; ����� ← 0 ⊲ �¯ : EMA, �=0.9995   
2: for epoch $t = 0 , \ldots , T { - } 1$ do   
3: for each minibatch $( x , y )$ of size � do   
4: $\delta _ { 0 } \sim$ fresh half-sign init at 0.5�; �<sub>0</sub> =   
$f ( x + \delta _ { 0 } ; w ) , z _ { c } = f ( x ; w )$   
5: $\delta _ { a } ~ = ~ \mathrm { c l i p } \big ( \delta _ { 0 } ~ + ~ \varepsilon \mathrm { s i g n } \nabla _ { \delta _ { 0 } } \mathrm { C E } ( z _ { 0 } , y ) \big ) ; ~ z _ { a } ~ = ~$   
$f ( x { + } \delta _ { a } { ; } w )$   
6: L ← Eq. 2 with �(�), � (�) ⊲ the coupled step   
7: $w  w - \eta \nabla _ { w } \mathcal { L } ; \quad \bar { w }  \tau \bar { w } + ( 1 - \tau ) w$   
8: end for   
9: �<sub>�</sub> ← validation PGD-10 of � ⊲ protocol already   
computes it for selection   
10: $b ^ { \star } \gets \operatorname* { m a x } ( b ^ { \star } , p _ { w } )$   
11: if $b ^ { \star } - p _ { w } > \theta$ and ����� < 5 and cooldown ≥ 3   
epochs then   
12: � ← �¯ ; � ���� ← � ���� + 1 ⊲ reset to the   
anchor; zero cost   
13: end if   
14: end for   
15: return weights selected by best validation PGD-10 (stan  
dard protocol)

## 4.2 The guard: a discrete reset

Every method in the benchmark already evaluates PGD-10 on a held-out validation split every epoch, for checkpoint selection. The guard reads exactly this number for the raw weights. Writing $p _ { t }$ for the epoch-� validation PGD-10 of $w _ { t } .$ it maintains the running best and the deficit

$$
b _ { t } ^ { \star } = \operatorname* { m a x } _ { s \leq t } p _ { s } , ~ g _ { t } ~ = ~ b _ { t } ^ { \star } - p _ { t } ,\tag{3}
$$

and fires the reset

$$
w _ { t + 1 } = \left\{ \begin{array} { l } { { \bar { w } _ { t } , g _ { t } > \theta \land t - t _ { \mathrm { f i r e } } \geq 3 \land F _ { t } < 5 , } } \\ { { w _ { t } , \mathrm { o t h e r w i s e } , } } \end{array} \right.\tag{4}
$$

where $t _ { \mathrm { f i r e } }$ is the last reset epoch and $F _ { t }$ the number of resets so far (Algorithm 1). The two bookkeeping constants exist for failure modes, not for normal operation: the cooldown stops the guard from firing every epoch right after a reset (the anchor lands below the pre-CO peak, so the deficit stays positive by construction), and the fire budget is a stop-loss against a sinking anchor (on vanilla FGSM training an uncapped guard fires 68 times in 100 epochs). In our formal runs the bookkeeping does its work quietly: three of the four CIFAR-10 seeds never fire at all, and the fourth fires three times inside its CO window, each dip spaced exactly at the cooldown, recovering to the same level as the others (50.61). The cost is zero additional passes: the deployed method is a static network trained and evaluated under the unchanged protocol.

![](images/9db080f5e82ee2d7782577e1cb22c7fcdddad075af1cce3a62bf368cc79d504c.jpg)

![](images/8a07c7982a90dddcc63dc69a4097a19ac7f6a63d7698adf1979cfd8e17d48468.jpg)  
Figure 5: Left: choosing the threshold. Over 600 epochs of healthy runs the guard statistic never dips more than 6.5 points below its best; the first abnormal epoch of every CO dips at least 9.6. We place �=8 points in the gap. Right: the same seed under two thresholds (logged validation curves; arrows mark guard resets). The trajectories are bit-identical through epoch 18; �=8 catches the first abnormal epoch and recovers (AA 50.61); �=10 misses it, fires one epoch later on a contaminated anchor, and dies (7.94).

## 4.3 Choosing the threshold

The guard’s one parameter has a natural value, because healthy and drifting trajectories are well separated (Fig. 5). Across 600 epochs of healthy guarded runs the deficit $g _ { t }$ never exceeds 6.5 points, while at the onset of every CO we have observed the first abnormal epoch dips at least 9.6 points. Any threshold in (6.5, 9.6) therefore never fires on healthy trajectories and always fires on the first abnormal epoch; we choose the center, �=8 points. The choice is also forgiving in one direction and sharp in the other, and one controlled experiment shows how sharp: on a seed whose trajectory is identical through epoch 18, �=8 catches the first abnormal epoch and the run finishes at 50.61 AutoAttack, while �=10 fires one epoch later on an already contaminated anchor and the run dies at 7.94 (Fig. 5, right).

Two families of controls pin down that the rescue comes from the anchor itself rather than from the detection, and where anchoring stops helping; both are relegated to the appendix. A detector-free control that resets at a fixed epoch rescues the same seed essentially as well (Sec. C), and an always-on continuous pull of the weights toward the EMA either sleeps through slow drift or, at higher strength, destroys training (Sec. C). Anchoring requires a robust region to anchor to: it cannot save vanilla FGSM training, whose average drifts along (Sec. C).

## 5 Experiments

Protocol. We follow the benchmark protocol [16] exactly and reproduce no baselines: ResNet18 [8] on CIFAR-10/100 and PreActResNet18 on Tiny-ImageNet, �=8/255, 100 epochs, perepoch validation PGD-10 for checkpoint selection, AutoAttack [4] and the benchmark’s constraint-relaxation (CR) attack [20] on the full test set; full details in Appendix A. Our method is the base setup of Eq. (2) plus the guard, with $\gamma _ { \mathrm { m a x } } { = } 4 . 0 , m _ { \mathrm { m a x } } { = } 2 . 5$ on CIFAR-10, and $\gamma _ { \mathrm { m a x } } { = } 1 . 2 5 , m _ { \mathrm { m a x } } { = } 4 . 5$ on CIFAR-100 (the coupling strength must be re-balanced on the harder dataset; Sec. 5). All ± values are sample standard deviations over the listed seeds.

CIFAR-10. The base method is the strongest fast configuration we know when it survives, and it survives half the time. The guard removes the coin flip: four runs, zero CO, 50.45±0.20 AutoAttack, +0.46 over LIET, the strongest published fast method, and +0.35 over NuAT. On healthy runs the guard statistic never came near its threshold, so the guarded number is the base method’s own strength with the coin flip removed. The complete AutoAttack suite (the targeted FAB and square attacks in addition to the two APGD variants) finds nothing beyond APGD on our models, consistent with the absence of gradient masking. The CR attack, the benchmark’s second strong-attack suite, preserves the ordering: ours 50.44±0.16, ahead of NuAT (50.11±0.19) and LIET (49.93±0.15) and within 0.1 of the multi-step PGD-AT-WA (50.51±0.16). The only better number in Table 1 is PGD-AT-WA, a multi-step method at several times the cost. Table 2 (Appendix B) collects the threshold, control, and failure-mode probes.

CIFAR-100 and Tiny-ImageNet: the coupling weight flips. The nuclear-norm term that helps on CIFAR-10 is toxic at the same strength on the harder datasets: at weight 4.0 the base method collapses to 11.2 on CIFAR-100 and 7.8 on Tiny ImageNet, the same failure as the published nuclear-norm method NuAT. Reducing the weight to 1.0 repairs the setup, and the response on CIFAR-100 is an inverted U over the weight: 1.0 gives 26.6, the peak at 1.25 with MSE weight 4.5 gives 27.15±0.04 over four seeds, 1.5 gives 27.1, and 2.0 falls back to 25.5. At the peak we clear LIET (26.64) by +0.51 and the multi-step PGD-AT-WA (26.90, roughly twice our cost) by +0.25; the CR column shows the same margins (27.20±0.07, +0.57 over LIET and +0.33 over the multi-step reference). On Tiny-ImageNet the same re-balancing brings no further gain $( m _ { \mathrm { m a x } } { = } 5 . 0 $ scores 20.69 on its single seed, level with the $m _ { \mathrm { m a x } } { = } 2 . 5$ mean), so the setup stays at $\gamma _ { \mathrm { { m a x } } } { = } 1 . 0 $ $m _ { \mathrm { m a x } } { = } 2 . 5$ and reaches 20.47±0.11, +0.59 over LIET, the best fast-method number we are aware of on this dataset. The response is monotone on both sides of the flip, which identifies the nuclear weight, not the guard, as the load-bearing choice on these datasets. The guard never fires on either one, so here the contribution is the repair of the nuclear-norm family’s known failure on harder datasets, with the guard carried as insurance. Per-seed values for every number in the table are listed in Appendix B.

Table 1: Robust accuracy (%) under AutoAttack (AA) and the benchmark’s constraint-relaxation attack (CR) on all three benchmark datasets, �=8/255, mean±sample std over the released seeds. Baselines are the benchmark’s released results [16]; nothing is retrained for comparison. Ours: base method plus guard, 4 seeds on CIFAR-10/CIFAR-100 and 3 on Tiny-ImageNet. Bold: best fast method per column. The multi-step rows (PGD-AT, PGD-AT-WA) are shown for reference at several times the fast-method cost.
<table><tr><td></td><td colspan="2">CIFAR-10</td><td colspan="2">CIFAR-100</td><td colspan="2">Tiny-ImageNet</td></tr><tr><td>Method</td><td>AA</td><td>CR</td><td>AA</td><td>CR</td><td>AA</td><td>CR</td></tr><tr><td>multi-step (reference)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PGD-AT [15]</td><td>44.33±0.08</td><td> $4 4 . 5 0 { \pm } 0 . 1 3 $ </td><td> $2 1 . 4 4 { \pm } 0 . 2 3 $ </td><td> $2 0 . 8 1 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $1 7 . 0 4 { \pm } 0 . 0 8 $ </td><td> $1 6 . 8 4 { \pm } 0 . 1 2 $ </td></tr><tr><td>PGD-AT-WA [10]</td><td>50.49±0.15</td><td> $5 0 . 5 1 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $2 6 . 9 0 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $2 6 . 8 7 { \scriptstyle \pm 0 . 1 7 }$ </td><td> $1 9 . 7 7 { \scriptstyle \pm 0 . 1 6 }$ </td><td> $1 9 . 7 5 { \pm } 0 . 1 3 $ </td></tr><tr><td>fast</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>NuAT [24]</td><td>50.10±0.18</td><td> $5 0 . 1 1 { \scriptstyle \pm 0 . 1 9 }$ </td><td> $1 4 . 4 1 \pm 3 . 2 9$ </td><td> $1 7 . 6 6 { \pm } 0 . 4 2 $ </td><td> $1 0 . 2 0 { \scriptstyle \pm 0 . 4 4 }$ </td><td>10.19±0.44</td></tr><tr><td>LIET [18]</td><td>49.99±0.15</td><td>49.93±0.15</td><td>26.64±0.13</td><td>26.63±0.14</td><td>19.88±0.09</td><td>19.86±0.07</td></tr><tr><td>FGSM-PGI [11]</td><td>49.89±0.08</td><td>49.89±0.12</td><td>26.29±0.16</td><td>26.31±0.12</td><td>19.55±0.11</td><td>19.52±0.12</td></tr><tr><td>FGSM-FUAP [17]</td><td>49.73±0.12</td><td>49.71±0.12</td><td>25.95±0.08</td><td>25.94±0.05</td><td>19.10±0.18</td><td>19.09±0.19</td></tr><tr><td>FGSM-CUAP [17]</td><td>49.55±0.22</td><td>49.54±0.22</td><td>25.85±0.50</td><td>25.83±0.49</td><td>19.19±0.09</td><td>19.20±0.11</td></tr><tr><td>FGSM-UAP [17]</td><td>49.44±0.14</td><td>49.37±0.15</td><td>25.84±0.24</td><td>25.82±0.23</td><td>18.90±0.28</td><td>18.96±0.21</td></tr><tr><td>GAT [23]</td><td>49.15±0.15</td><td>49.15±0.17</td><td>22.06±0.08</td><td>21.97±0.07</td><td>11.33±0.06</td><td>11.27±0.06</td></tr><tr><td>FGSM-RS-CS [30]</td><td>48.64±0.38</td><td> $4 8 . 7 0 { \scriptstyle \pm 0 . 3 4 }$ </td><td> $2 6 . 2 0 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $2 6 . 2 0 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $1 8 . 7 4 { \scriptstyle \pm 0 . 1 1 }$ </td><td>18.69±0.11</td></tr><tr><td>FGSM-PCO [26]</td><td>48.52±0.32</td><td> $4 8 . 4 8 { \pm } 0 . 3 4$ </td><td> $1 9 . 9 7 { \scriptstyle \pm 0 . 4 3 }$ </td><td>19.91±0.39</td><td> $1 1 . 2 4 { \pm } 0 . 2 1 $ </td><td>11.20±0.23</td></tr><tr><td>GradAlign [2]</td><td>47.99±0.62</td><td> $4 8 . 0 2 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $2 5 . 3 7 { \scriptstyle \pm 0 . 1 4 }$ </td><td>25.39±0.15</td><td>17.87±0.19</td><td>17.81±0.22</td></tr><tr><td>N-FGSM [5]</td><td> $4 7 . 7 7 { \scriptstyle \pm 0 . 4 1 }$ </td><td> $4 7 . 8 4 { \pm } 0 . 4 1 $ </td><td> $2 4 . 8 6 { \pm } 0 . 4 6 $ </td><td> $2 4 . 8 9 { \scriptstyle \pm 0 . 4 9 }$ </td><td> $1 8 . 7 9 { \scriptstyle \pm 0 . 0 8 }$ </td><td>18.76±0.06</td></tr><tr><td>N-AAER [14]</td><td>47.38±0.18</td><td> $4 7 . 4 8 { \pm } 0 . 2 0 $ </td><td> $2 2 . 7 4 { \pm } 1 . 1 7$ </td><td> $2 2 . 8 1 { \pm } 1 . 1 8$ </td><td> $8 . 7 4 { \pm } 1 1 . 6 5$ </td><td>8.74±11.65</td></tr><tr><td>ZeroGrad [6]</td><td> $4 5 . 9 4 \pm 2 . 1 9$ </td><td> $4 6 . 0 4 { \pm } 2 . 2 0 $ </td><td> $2 5 . 0 0 { \pm } 0 . 1 4$ </td><td> $2 4 . 9 5 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $1 8 . 1 5 { \pm } 1 . 1 1 $ </td><td>18.10±1.14</td></tr><tr><td>FGSM-MEP-CS [30]</td><td> $4 5 . 3 5 { \pm } 0 . 8 1 $ </td><td> $4 5 . 3 7 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $2 4 . 8 6 { \pm } 0 . 2 8 $ </td><td> $2 4 . 8 0 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $1 8 . 2 3 { \pm } 0 . 1 9 $ </td><td>18.13±0.16</td></tr><tr><td>Free-AT [22]</td><td>44.34±0.17</td><td>44.32±0.24</td><td>21.76±0.25</td><td>21.74±0.27</td><td> $1 3 . 5 1 { \pm } 0 . 1 2 $ </td><td> $1 3 . 5 1 { \pm } 0 . 1 5 $ </td></tr><tr><td>FGSM-RS [27]</td><td> $4 3 . 0 9 { \scriptstyle \pm 5 . 3 3 }$ </td><td> $4 3 . 1 4 \pm 5 . 2 9$ </td><td> $2 1 . 4 2 { \scriptstyle \pm 2 . 2 4 }$ </td><td> $2 1 . 4 6 { \pm } 2 . 2 4$ </td><td> $1 6 . 3 3 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $1 6 . 2 6 { \scriptstyle \pm 0 . 7 1 }$ </td></tr><tr><td>SSAT [12]</td><td>38.33±1.55</td><td> $3 8 . 3 2 { \pm } 1 . 5 3 $ </td><td>18.83±0.73</td><td> $1 8 . 7 9 { \scriptstyle \pm 0 . 7 9 }$ </td><td>11.94±0.12</td><td>11.92±0.14</td></tr><tr><td>FGSM-AT [7]</td><td>37.28±0.30</td><td>37.27±0.27</td><td> $1 6 . 6 2 { \scriptstyle \pm 0 . 5 4 }$ </td><td> $1 6 . 5 8 { \pm } 0 . 5 0 $ </td><td> $1 4 . 8 7 { \scriptstyle \pm 3 . 7 7 }$ </td><td> $1 4 . 8 4 { \pm } 3 . 7 5 $ </td></tr><tr><td>ELLE [1]</td><td> $3 2 . 4 6 { \pm } 2 5 . 2 7$ </td><td> $3 9 . 7 7 { \scriptstyle \pm 1 2 . 7 7 }$ </td><td> $2 1 . 2 1 { \scriptstyle \pm 2 . 3 7 }$ </td><td> $2 1 . 2 3 { \pm } 2 . 3 7$ </td><td> $1 8 . 8 1 { \scriptstyle \pm 0 . 2 1 }$ </td><td> $1 8 . 8 0 { \scriptstyle \pm 0 . 2 1 }$ </td></tr><tr><td>Base + guard (ours)</td><td> ${ \bf 5 0 . 4 5 { \scriptstyle \pm 0 . 2 0 } }$ </td><td>50.44±0.16</td><td> $\mathbf { 2 7 . 1 5 { \pm } 0 . 0 4 }$ </td><td> $\mathbf { 2 7 . 2 0 { \pm } 0 . 0 7 }$ </td><td>20.47±0.11</td><td> $\mathbf { 2 0 . 5 4 \pm 0 . 1 5 }$ </td></tr></table>

The unguarded base is not deployable: on CIFAR-10 it matches the guarded row on the two seeds that survive (AutoAttack 50.67±0.08) but returns gradient-masked artifacts on the two that collapse, for a 4-seed mean of 29.1; the guard’s efect is exactly the removal of this failure mode. On CIFAR-100 and Tiny-ImageNet the guard never fires; those columns are the re-balanced base method’s own performance with the guard carried as zero-cost insurance. Under the iterative attacks of the released suite the ordering is similar (PGD-50: LIET 55.97, NuAT 54.63, ours 54.63); full columns in the supplementary material.

Is the gain attack-specific? No. Which samples a model solves under AutoAttack is strongly conserved across healthy methods (mean pairwise Kendall $\tau = 0 . 8 2 )$ , and our anchored models follow the same conserved pattern $( \tau = 0 . 8 4 7 )$ ; the only models that rewrite the sample-dificulty spectrum are CO ones $( \tau = 0 . 3 8 6 )$ . The gain is therefore a genuine robustness gain, not an artifact of the evaluation attack. The check is worth reporting in its own right, since evaluation pipelines can silently favor some methods when test-side information leaks into training choices [19]; solving the same conserved set of hard samples is the opposite signature. Details in Appendix E.

runs is that the guard statistic never approached its threshold, which is the operative claim. (2) The per-dataset coupling weights $\scriptstyle ( \gamma _ { \mathrm { m a x } } = 4 . 0 / 1 . 2 5 / 1 . 0 )$ were found by small sweeps, not derived from the mechanism; the dose response is broad near the peak, but the weight is not universal across datasets. (3) Not every failure of the base method is a validation-visible CO: one collapsed seed kept high validation PGD-10 while test robustness died (Appendix B), and no validation-side statistic we know sees that flavor. (4) The propositions are informal, and the logit-margin construction is a sketch, not a proof of existence at network scale. (5) Instrumentation perturbs the random streams: adding batch-level probes changes which seeds collapse, even though the dynamics for a given draw are deterministic.

## 6 Limitations

## 7 Conclusion

(1) Seed counts are small (4/4/3 per dataset). Zero CO in four seeds is consistent with a CO rate as high as 53% (one-sided 95% binomial bound); what we actually observe on healthy

Catastrophic overfitting is not an optimization failure in the usual sense: nothing about the training objective got worse. It is an escape along directions the training loss cannot see, by a model whose clean behavior quietly improves while its robustness disappears. Once measured, the geometry dictates the response: anchor the drifting weights to their own timeaverage, with a trigger that is itself sensitive to the drift. The anchor is attack-agnostic (no evaluation signal enters training), costs nothing beyond what the protocol already computes, and fails informatively wherever there is no robust region to anchor to.

## References

[1] Elias Abad Rocamora, Fanghui Liu, Grigorios G Chrysos, Pablo M Olmos, and Volkan Cevher. Eficient local linearity regularization to overcome catastrophic overfitting, 2024.

[2] Maksym Andriushchenko and Nicolas Flammarion. Understanding and improving fast adversarial training. In Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and Hsuan-Tien Lin, editors, Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ b8ce47761ed7b3b6f48b583350b7f9e4-Abstract.html.

[3] Anish Athalye, Nicholas Carlini, and David Wagner. Obfuscated gradients give a false sense of security: Circumventing defenses to adversarial examples. In International Conference on Machine Learning (ICML), 2018.

[4] Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameterfree attacks. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings ofMachine Learning Research, pages 2206–2216. PMLR, 2020. URL http: //proceedings.mlr.press/v119/croce20b.html.

[5] Pau de Jorge Aranda, Adel Bibi, Riccardo Volpi, Amartya Sanyal, Philip Torr, Guillaume Rogez, and Puneet Dokania. Make some noise: Reliable and eficient single-step adversarial training. In Advances in Neural Information Processing Systems, volume 35, pages 12881–12893, 2022.

[6] Zeinab Golgooni, Mehrdad Saberi, Masih Eskandar, and Mohammad Hossein Rohban. Zerograd: Costless conscious remedies for catastrophic overfitting in the FGSM adversarial training. Intell. Syst. Appl., 19:200258, 2023. doi: 10.1016/J.ISWA.2023.200258. URL https://doi.org/10. 1016/j.iswa.2023.200258.

[7] Ian Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. arXiv preprint arXiv:1412.6572, 2014.

[8] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015. URL https://arxiv.org/abs/1512.03385.

[9] Zhengbao He, Tao Li, Sizhe Chen, and Xiaolin Huang. Investigating catastrophic overfitting in fast adversarial training: A self-fitting perspective. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2023 - Workshops, Vancouver, BC, Canada, June 17-24, 2023, pages 2314–2321.

IEEE, 2023. doi: 10.1109/CVPRW59228.2023.00227. URL https://doi.org/10.1109/CVPRW59228.2023.00227.

[10] Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson. Averaging weights leads to wider optima and better generalization. In UAI, 2018.

[11] Xiaojun Jia, Yong Zhang, Xingxing Wei, Baoyuan Wu, Ke Ma, Jue Wang, and Xiaochun Cao. Prior-guided adversarial initialization for fast adversarial training. In European Conference on Computer Vision (ECCV), pages 567–584, 2022.

[12] Hoki Kim, Woojin Lee, and Jaewook Lee. Understanding catastrophic overfitting in single-step adversarial training. In Thirty-Fifth AAAI Conference on Artificial Intelligence, AAAI 2021, Thirty-Third Conference on Innovative Applications of Artificial Intelligence, IAAI 2021, The Eleventh Symposium on Educational Advances in Artificial Intelligence, EAAI 2021, Virtual Event, February 2-9, 2021, pages 8119–8127. AAAI Press, 2021. doi: 10.1609/AAAI.V35I9.16989. URL https: //doi.org/10.1609/aaai.v35i9.16989.

[13] James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. Overcoming catastrophic forgetting in neural networks. In PNAS, volume 114, pages 3521–3526, 2017.

[14] Runqi Lin, Chaojian Yu, and Tongliang Liu. Eliminating catastrophic overfitting via abnormal adversarial examples reg ularization. In Advances in Neural Information Processing Systems, volume 36, pages 67866–67885, 2023.

[15] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations (ICLR), 2018.

[16] Chao Pan and Xin Yao. Fastat benchmark: A comprehensive framework for fair evaluation of fast adversarial training methods. arXiv preprint arXiv:2604.22853, 2026.

[17] Chao Pan, Qing Li, and Xin Yao. Adversarial initialization with universal adversarial perturbation: A new approach to fast adversarial training. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 21501–21509, 2024.

[18] Chao Pan, Ke Tang, Qing Li, and Xin Yao. Mitigating catastrophic overfitting in fast adversarial training via label information elimination. In IEEE/CVF International Conference on Computer Vision, ICCV 2025, Honolulu, HI, USA, October 19-25, 2025, pages 2991–3000. IEEE, 2025. doi: 10.1109/ICCV51701.2025.00287. URL https://doi.org/ 10.1109/ICCV51701.2025.00287.

[19] Chao Pan, Ke Tang, Qing Li, and Xin Yao. Rethinking RobustBench: Is high synthetic-test data similarity an implicit information advantage inflating robustness scores? In IEEE In ternational Conference on Data Science andAdvancedAnalytics (DSAA), pages 1–10, 2025.

[20] Chao Pan, Yu Wu, Ke Tang, Qing Li, and Xin Yao. Eficient robustness evaluation via constraint relaxation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 6263–6271, 2025.

[21] Boris T Polyak and Anatoli B Juditsky. Acceleration ofstochastic approximation by averaging. SIAM Journal on Control and Optimization, 30(4):838–855, 1992.

[22] Ali Shafahi, Mahyar Najibi, Amin Ghiasi, Zheng Xu, John Dickerson, Christoph Studer, Larry S Davis, Gavin Taylor, and Tom Goldstein. Adversarial training for free! In NeurIPS, 2019.

[23] Gaurang Sriramanan, Sravanti Addepalli, Arya Baburaj, and R Venkatesh Babu. Guided adversarial attack for evaluating and enhancing adversarial defenses. In Advances in Neural Information Processing Systems, volume 33, pages 20297– 20308, 2020.

[24] Gaurang Sriramanan, Sravanti Addepalli, Arya Baburaj, and Venkatesh Babu R. Towards eficient and efective adversarial training. In Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual, pages 11821–11833, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 62889e73828c756c961c5a6d6c01a463-Abstract.html.

[25] Christian Szegedy, Vincent Vanhoucke, Sergey Iofe, Jonathon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In 2016 IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2016, Las Vegas, NV, USA, June 27-30, 2016, pages 2818–2826. IEEE Computer Society, 2016. doi: 10.1109/CVPR.2016.308. URL https://doi.org/10.1109/CVPR.2016.308.

[26] Zhaoxin Wang, Handing Wang, Cong Tian, and Yaochu Jin. Preventing catastrophic overfitting in fast adversarial training: A bi-level optimization perspective. In Ales Leonardis, Elisa Ricci, Stefan Roth, Olga Russakovsky, Torsten Sattler, and G¨ul Varol, editors, Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part XXVIII, volume 15086 of Lecture Notes in Computer Science, pages 144–160. Springer, 2024. doi: 10.1007/978-3-031-73390-1\ 9. URL https://doi.org/10. 1007/978-3-031-73390-1\_9.

[27] Eric Wong, Leslie Rice, and J. Zico Kolter. Fast is better than free: Revisiting adversarial training. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. OpenReview.net, 2020. URL https://openreview.net/forum?id=BJx040EFvH.

[28] Michael R Zhang, James Lucas, Geofrey Hinton, and Jimmy Ba. Lookahead optimizer: k steps forward, 1 step back. In NeurIPS, 2019.

[29] Jie-Chao Zhao, Jin Ding, Yong-Zhi Sun, Ping Tan, Ji-En Ma, and You-Tong Fang. Avoiding catastrophic overfitting in fast adversarial training with adaptive similarity step size. PLOS ONE, 20(1):e0317023, 2025. doi: 10.1371/journal.pone.0317023.

[30] Mengnan Zhao, Lihe Zhang, Yuqiu Kong, and Baocai Yin. Fast adversarial training with smooth convergence. In IEEE/CVF International Conference on Computer Vision, ICCV 2023, Paris, France, October 1-6, 2023, pages 4697–4706. IEEE, 2023. doi: 10.1109/ICCV51070.2023.00435. URL https: //doi.org/10.1109/ICCV51070.2023.00435.

## A Protocol and measurement details

Training follows the benchmark [16] exactly: ResNet18 on CIFAR-10/100 and PreActResNet18 on Tiny-ImageNet; �=8/255 for training and evaluation; 100 epochs, OneCycle SGD, label smoothing 0.4 [25]; a 49,000/1,000 train/validation split; per-epoch validation PGD-10 for checkpoint selection; the complete standard AutoAttack suite (apgd-ce, apgd-t, FAB-T, square) and the constraint-relaxation (CR) attack [20], both in the benchmark’s released configuration on the full test set. The main-table and per-seed rows use the complete suite; the dose-probe rows keep the two-attack development variant, which on our models difers by at most 0.02 (the targeted FAB and square attacks find no additional points on any of our checkpoints, consistent with the absence of gradient masking). A run of our method takes 1.05 hours against 0.7–0.8 for the strongest two-pass fast methods and about 2h for PGD-AT. Measurement detail. An EMA checkpoint evaluated with its own stale BatchNorm statistics scores at chance level, so every EMA evaluation in Sec. 3 recalibrates BatchNorm statistics on 4,096 training samples first; without this step the entire efect disappears. The recalibration is an analysis-time tool only and is not part of the deployed method. A CO event is defined on the per-epoch validation curves so that it can be re-derived from any log: validation PGD-10 drops more than 30 points in one epoch, stays below 10% for the following three epochs, while validation FGSM accuracy remains above 30%. The CIFAR-10 runs of Eq. (2) additionally carry a small wrong-logit dispersion term (weight 0.3) inherited from the method’s development lineage; it is disabled on CIFAR-100/Tiny-ImageNet and plays no role in the anchoring mechanism. Odd steps add the same nuclear term inside the attack objective, following NuAT.

## B Per-seed values

Tables 2–5 list the per-seed values behind every mean in the paper. Complete training logs, checkpoints, and the analysis outputs behind Sec. 3 ship with the supplementary material.

## C Controls and failure modes

The rescue comes from the anchor, not the detector. Three controls separate the mechanism from the bookkeeping. First, a detector-free control: reset at a fixed epoch (18), ignoring all signals. It rescues the same seed to 49.99, essentially the guarded result. The detection signal contributes timing, not magic. (The fixed epoch presupposes knowledge of when that seed collapses, so this is a control, not a method.) Second, a strength control: an always-on anchor at 2.5× the pull strength $( \eta _ { a } { = } 0 . 0 5 )$ destroys training (8.15). Only the blind component should ever be pulled; pulling always fights the optimizer itself. Third, falling back to the coupled loss when the guard fires instead of resetting also fails (7.05), confirming that it is the reset that helps, not any extra regularization. A re-nucleating draw (five CO events in 31 epochs) isolates the bookkeeping choices on a failure case: the raw-reading trigger is decisionidentical to the two-reading version (7.54 versus 7.53 on the same draw, the trajectories bit-identical until the budget binds), and without the fire budget checkpoint selection alone lifts the masked endpoint to 15.14.

Table 2: CIFAR-10: per-seed AutoAttack (%) for our configurations and controls. Guard fires per seed in parentheses where nonzero.
<table><tr><td>Configuration</td><td>Per-seed AA (%)</td><td>Note</td></tr><tr><td>ours Base + guard, θ=8 guard  $\scriptstyle { \bar { \theta = 5 } }$  guard θ=3</td><td>50.19/50.40/50.61/50.60 50.72/50.43/50.57/50.63 50.42/49.90/50.54</td><td>fires 0/0/4/0 no change 2/5/3 spurious</td></tr><tr><td>threshold and trigger controls guard θ=10, same traj. guard θ=12 guard, no cooldown (s2) trigger on raw reading only without budget</td><td>7.94 50.40 / 7.84 50.18 7.54 15.14 7.53</td><td>misses epoch 19 saves 1 of 2 seeds 15 consecutive fires same draw, with budget selection artifact decision-identical</td></tr><tr><td>cont. pull  $\eta _ { a } { = } 0 . 0 2$  cont. pull  $\eta _ { a } { = } 0 . 0 5$  coupled-loss fallback two-pass base (no clean branch), seed 0</td><td>50.45/50.29/50.39/7.14/50.11 8.15 7.05 49.92 / 49.93</td><td>1 of 5 fails too strong on fire</td></tr><tr><td>EMA-weights anchor high-weight base + guard earlier two-pass ports</td><td>collapses 7.13 / 7.80 / 5.07 48.57-49.84</td><td>adv. acc. 0.94 by ep. 8 3 seeds, no region</td></tr><tr><td></td><td></td><td></td></tr><tr><td>continuous pull / guard</td><td>0.00 / 38.34</td><td>matches the own-EMA</td></tr><tr><td>decoupled victim (vanilla FGSM-AT)</td><td></td><td></td></tr></table>

Table 3: CIFAR-10: full test-set columns per model. The unguarded base either matches the guarded row (survivors) or returns a gradientmasked artifact (CO seeds: inflated clean, collapsed PGD-50 and AutoAttack).
<table><tr><td>Model</td><td>Seed</td><td>Clean</td><td>PGD-10</td><td>PGD-50</td><td>AA</td></tr><tr><td>Base + guard</td><td>0</td><td>79.46</td><td>55.43</td><td>54.60</td><td>50.19</td></tr><tr><td></td><td>1</td><td>80.62</td><td>55.18</td><td>54.43</td><td>50.40</td></tr><tr><td></td><td>2</td><td>79.35</td><td>55.20</td><td>54.52</td><td>50.61</td></tr><tr><td></td><td>3</td><td>79.28</td><td>55.67</td><td>54.98</td><td>50.60</td></tr><tr><td>Base, no guard</td><td>0</td><td>79.84</td><td>一</td><td>54.94</td><td>survived</td></tr><tr><td>(seed status)</td><td>3</td><td>79.28</td><td></td><td>54.98</td><td>survived</td></tr><tr><td></td><td>1</td><td>92.58</td><td></td><td>25.62</td><td>7.57</td></tr><tr><td></td><td>2</td><td>92.37</td><td>_</td><td>24.61</td><td>7.48</td></tr></table>

PGD-20 for the guarded seeds: 54.74/54.58/54.65/55.05. Unguarded survivors score the same level as the guarded row under checkpoint selection restricted to healthy epochs; the two CO seeds select masked artifacts.

A continuous variant, and why we did not choose it. The same idea can be made continuous: every �=8 steps, measure on the current batch the KL divergence $\Delta _ { t }$ between the EMA’s predictions and the adversarial predictions, and update

$$
w _ { t + 1 } = w _ { t } - \eta _ { a } \operatorname* { m a x } ( 0 , \Delta _ { t } - \delta ) \big ( w _ { t } - \bar { w } _ { t } \big ) .\tag{5}
$$

The pull is exactly zero while the two checkpoints agree and grows with the disagreement; it costs one extra forward pass per 8 steps. It failed on one of its five seeds (7.14): that seed’s degradation was slow, and its trigger, the KL between clean and adversarial output distributions, stayed inside its dead zone the whole time. The reason is geometric and worth stating as a design principle: a trigger measured on the clean function inherits the clean function’s blindness (Sec. 3.2) and can sleep through the very drift it is meant to catch. The guard’s trigger is validation PGD, a robustness measurement rather than a clean one, and it fired and recovered the same seed. This is why the guard, not the continuous pull, is our method.

Table 4: CIFAR-100: per-seed AutoAttack (%) along the weight response. $\gamma _ { \mathrm { m a x } }$ is the nuclear weight, $m _ { \mathrm { m a x } }$ the MSE weight.
<table><tr><td>Setting</td><td>Per-seed AA (%)</td><td>Mean</td></tr><tr><td>γ=4.0 (one seed)</td><td>11.20</td><td></td></tr><tr><td>γ=0.5 (one seed)</td><td>22.44</td><td></td></tr><tr><td> $\stackrel { \cdot } { \gamma } = 1 . 0 , m = 2 . 5$ </td><td>25.53/26.41/26.23/26.24</td><td>26.10±0.39</td></tr><tr><td>no nuclear term (3 seeds)</td><td>26.36/25.82/24.93</td><td></td></tr><tr><td> $m { = } 3 . 5$ </td><td>26.35/26.74/26.27/26.42</td><td>26.45</td></tr><tr><td> $\mathrm { \Delta } m { = } 4 . 5$ </td><td>26.61/26.35/26.66/26.64</td><td>26.57±0.14</td></tr><tr><td> $m { = } 5 . 0 / 5 . 5 ( \mathrm { s e e d } 0 )$ </td><td>26.44 / 25.98</td><td>turnover above 4.5</td></tr><tr><td> $\gamma { = } 1 . 5 \ ( \mathrm { s e e d } \ 0 )$ </td><td>27.12</td><td></td></tr><tr><td> $\gamma { = } 2 . 0 \ ( \mathrm { s e e d } \ 0 )$ </td><td>25.45</td><td>past the peak</td></tr><tr><td> $\gamma { = } 1 . 2 5 , m { = } 4 . 5 \mathrm { ( o u r s ) }$ </td><td>27.20/27.13/27.13/27.13</td><td>27.15±0.04</td></tr><tr><td colspan="3">single-seed probes that did not move the result</td></tr><tr><td>Bernoulli-uniform init</td><td>25.47</td><td></td></tr><tr><td>CE at the adversarial point, mix</td><td>26.06</td><td></td></tr><tr><td>tail-only nuclear</td><td>26.05</td><td></td></tr><tr><td>label smoothing 0.25</td><td>25.35</td><td></td></tr><tr><td>CE mix + tail-only nuclear</td><td>25.53</td><td></td></tr><tr><td>θ=5 / θ=3 (seed 0)</td><td>26.63 / 26.75</td><td>0 / 2 fires</td></tr></table>

Table 5: Tiny-ImageNet: per-seed AutoAttack (%).
<table><tr><td>Setting</td><td>Per-seed AA (%)</td><td>Note</td></tr><tr><td> $\gamma { = } 1 . 0 , m { = } 2 . 5 \mathrm { ( o u r s ) }$ </td><td>20.49/20.57/20.36</td><td>20.47±0.11</td></tr><tr><td>γ=4.0 (one seed)</td><td>7.80</td><td>toxic</td></tr><tr><td> $m { = } 5 . 0 \ ( \mathrm { o n e \ s e e d } )$ </td><td>20.69</td><td>one guard fire</td></tr><tr><td> $\theta { = } 5 / \theta { = } 3 ( \mathrm { s e e d } 0 )$ </td><td>20.42 / 20.40</td><td>fire-free, level</td></tr></table>

Where anchoring cannot help. An anchor needs a robust region to anchor to; the failures are as instructive as the successes. (i) Vanilla FGSM training cannot be saved: its drift carries a persistent bias that drags the EMA along; anchoring yields 0.00 (continuous pull) or 38.34 (guard, merely matching the benchmark’s own EMA-rescued FGSM-AT), and removing the guard’s two constants does worse, not better, because it resets every epoch to an anchor that is itself sinking. The guard is not a plug-in cure for arbitrary fast methods; it protects coupled setups whose average stays put. (ii) A high-weight coupled configuration at two-pass cost never establishes a stationary robust region in the first place (best validation PGD 36–43%); the guard fires, but the anchor it resets to is itself adrift, and all three seeds die. (iii) Our base, with the clean branch in the graph, holds both properties at once: strong coupling and a stationary robust region. The third forward pass is not just cost; it is what makes the method anchorable.

![](images/0241aded043583099b35f4c45f5dc87d8c0ac2e36ecf063bce8f0b6a14a8e944.jpg)  
Figure 6: The average collapses late, or never. Epochs between the raw CO and the EMA’s own CO, over the CO runs recoverable from the released benchmark curves. Bars count only runs in which the EMA also collapses within 100 epochs: 4 of the 11 CIFAR-10 CO runs (lags 6–24) and all 9 recoverable CIFAR-100 CO runs (lags 1– 29). In the remaining 7 CIFAR-10 CO runs the EMA never collapses within 100 epochs, so no finite lag exists.

## D Additional analyses

Death-order statistics. Early deaths have a smaller clean logit margin at the window’s first checkpoint (median 1.02 versus 1.62; two-sided Mann–Whitney $\scriptstyle p = 9 \times 1 0 ^ { - 1 6 }$ ; Spearman −0.54). The order is a property of the samples, not of one training path: an independently trained replica, instrumented on a fourfold coarser checkpoint grid, reproduces the same statistics $\scriptstyle ( p = 2 \times 1 0 ^ { - 2 4 }$ , Spearman −0.58), its death order correlates with the first replica’s at Spearman $\rho { = } 0 . 4 8 \ ( p { = } 2 \times$ $1 0 ^ { - 2 1 } )$ , and one replica’s margins predict the other’s death order (−0.45). Stratifying the trigger by margin does not fire earlier than the aggregate mean.

Precursor statistics. At epoch resolution, variance and lag-1 autocorrelation statistics of the released curves fire in only 3 of 11 benchmark CO runs, and the FGSM−PGD gap is not a warning (15-point gaps occur in healthy runs). At batch resolution the ramp of Fig. 4 also carries the statistical signature of a system approaching a transition point: in the final 3,000 batches before the clif the variance of the cosine rises by 150– 280× and its lag-1 autocorrelation climbs from 0.69–0.73 to 0.88 in both instrumented seeds, the pattern known as critical slowing down.

Drift anatomy. The weight diference across the seed-1 CO pair concentrates moderately in late blocks (energy Gini 0.79; the eight largest tensors, all convolutions in stages 3–4, hold 58% of the energy) but is not surgically localizable: rolling back the top-20%-energy tensors to the healthy checkpoint recovers only 0.043 → 0.133 of the 0.409 validation PGD-10, and a random equal-size rollback is harmful (0.010). The CO load is difuse across shared features, which is why the guard rolls back the whole weight vector rather than a submodule.

![](images/c7a533cb08df6a6b9b473c4dfb99ac49246044857399823cc981716b786ff6a9.jpg)

![](images/bf5be1bab6219c2da9e32802507b7a5b29ef477f028e08506f1252e39d8c12e2.jpg)  
Clean-margin quartile at the healthy checkpoint  
Figure 7: CO in sample space. Top: validation PGD-10 (mean, left axis) and the number of samples dying per checkpoint (right axis) across the seed-1 window of Fig. 4; a cohort of samples dies 100 or more batches before the aggregate clif. Bottom: fraction of each clean-margin quartile dying at least 100 batches before the median, for two independently trained seeds; low-margin samples die first in both.

Two flavors of validation blindness. On saved checkpoints we compared the protocol’s selection statistic (validation PGD-10 on 1,000 held-out images) with the attack-direction cosine used in Fig. 4. A collapsed checkpoint selected by validation PGD (the hard draw of the no-budget ablation: validation PGD 0.432, AutoAttack 7.5) is caught by geometry: its FGSM accuracy stays high (0.740) while the cosine collapses to 0.055, so multiplying the selection score by the cosine costs nothing and rejects it. A second collapsed seed shows the opposite flavor: validation PGD-10 stays high (0.592) and the cosine healthy (0.854) to the end of training while test PGD-50 falls to 25.6 and clean accuracy inflates to 92.6%; the validation distribution itself has been memorized. No validation-side statistic can see this flavor, which is why the guard’s rescue is limited to CO events that are visible in validation robustness.

## E Which samples do the methods solve?

We score per-sample AutoAttack survival (500-sample common validation subset) and compare orderings across models by Kendall � (Table 6). Healthy methods, benchmark and ours alike, agree strongly on which samples are hard (pairwise � 0.79–0.91, mean 0.82); a clean-dificulty baseline reaches 0.79, so much of the agreement is generic dificulty. PGD-20 with 5 restarts reproduces the AutoAttack ordering almost perfectly (per-model agreement 0.984), so the conservation is not an AutoAttack artifact. The only spectrum rewriters are failure modes: a collapsed seed $( \tau = 0 . 3 8 6 )$ and a logitsurgery variant (0.624). The strongest surviving base variant (AA 50.67) does not move the spectrum (0.847): high mean accuracy here is earned on the same dificulty structure as every other healthy method, not by gaming a particular attack.

Table 6: Kendall � of per-sample AutoAttack survival. All seven measured values are listed.
<table><tr><td>Quantity</td><td>T</td></tr><tr><td>Pairwise mean over 16 healthy models</td><td>0.82</td></tr><tr><td>PGD-20 (5 restarts), pairwise PGD-20 vs. AutoAttack ordering, per model</td><td>0.870</td></tr><tr><td>Clean-difficulty baseline</td><td>0.984</td></tr><tr><td>Collapsed seed</td><td>0.79</td></tr><tr><td>Logit-surgery variant</td><td>0.386</td></tr><tr><td>Strongest surviving base variant (AA 50.67)</td><td>0.624 0.847</td></tr></table>