# Efective Learning Rate Governs Loss Dynamics in Language Model Pretraining

Zihan Liu<sup>1,2,∗</sup> Ruiheng Zheng<sup>1,∗</sup> Shaobo Zhang<sup>1</sup> Changxin Tian<sup>2</sup> Kunlong Chen<sup>2</sup> Zhiqiang Zhang<sup>2</sup> Lei Wu<sup>1,†</sup> <sup>1</sup>Peking University <sup>2</sup>Ant Group

## Abstract

We uncover ELR collapse in language model pretraining: learning rate (LR) and parameter norm govern loss dynamics primarily through their ratio, the efective learning rate (ELR). When ELR is matched across runs, their loss trajectories collapse throughout training despite substantially diferent LRs and parameter norms. Across optimizers, architectures, datasets, and model scales, mean collapse errors are typically a few $\times 1 0 ^ { - 3 }$ , below the seedto-seed variation measured in a representative configuration. Systematic ablations identify normalization design and the timescale of LR–norm variation as key determinants of collapse precision. Controlled interventions further show that weight decay and Hyperball shape loss dynamics primarily through the ELR schedules they induce. Replacing LR with ELR enables a fitted functional scaling law (FSL) (Li et al., 2025a) to transfer across norm-control methods. The resulting ELR-based FSL also explains delayed acceleration, a recurring efect of norm control. Together, these results establish ELR as a common coordinate linking LR scheduling, norm control, and loss dynamics.

![](images/de966f641e943d81bdbdcab120d3b9e8a40f4c14dee2733a71c5389f2bade88b.jpg)

(a) Llama-124M / FineWeb  
![](images/d1343e43820111ffd293f72178cadcf892661fc595e71ae5cde40b6a364ef0f5.jpg)

![](images/b20e6bc53cb33edf866301d04296a892d385c167916d2d5b8883b2544a44302a.jpg)

(b) Qwen3-MoE-586M / FineWeb  
![](images/fd2a5a802ccae0d63a884eb7cc0af4d9b0fb03f6f8a9443e60b004f6988b2c42.jpg)  
Figure 1: ELR matching collapses loss trajectories despite diferent LR and norm schedules. In each panel, three nonconstant LR schedules are paired with prescribed norm schedules to match the ELR of a reference run. Legend values report the post-warmup mean absolute loss discrepancy from the reference in units of $1 0 ^ { - 3 }$ . Loss panels are clipped above 4.2 in (a) and 4.0 in (b) for display. See Section 4 and Appendices B and C for experimental details and additional results.

## 1 Introduction

Training dynamics are primarily governed by hyperparameters such as the learning rate (LR) and batch size. In large language model (LLM) pretraining, norm control provides another axis, encompassing mechanisms that shape parameter-norm evolution either implicitly through weight decay (Krogh and Hertz, 1991; Loshchilov and Hutter, 2019) or explicitly through norm constraints (Wen et al., 2026; Loshchilov, 2023; Xie et al., 2026). It can improve training stability (D’Angelo et al., 2024) and hyperparameter transfer across scales (Kosson et al., 2026), but it also reshapes loss dynamics and often accelerates convergence. This raises a basic question:

Does norm control introduce an independent degree of freedom governing loss dynamics?

ELR collapse: a precise macroscopic law. Within the regimes studied here, our experiments answer this question in the negative. We construct runs whose LR and parameter-norm trajectories difer substantially while matching a common ELR schedule step by step, where $\eta _ { k } ^ { \mathrm { e f f } } : = \eta _ { k } / \Vert \mathbf { W } _ { k } \Vert _ { F }$ . Figure 1 presents the central experiment:

## When ELR schedules are matched, full loss trajectories collapse with mean absolute discrepancies of only a few $\times 1 0 ^ { - 3 }$

For comparison, changing only the initialization or data-order seed in a representative configuration yields mean pairwise loss discrepancies of order $1 0 ^ { - 2 } { \mathrm { ; } }$ ; see Appendix A.2. We call this phenomenon ELR collapse. It suggests a simple macroscopic picture: LR scheduling and norm control shape loss dynamics primarily through the ELR schedule they jointly induce, rather than acting as independent controls.

ELR collapse is nontrivial and conditional. The precision of ELR collapse does not follow from scale invariance. Transformers are not exactly scale invariant. In a pre-norm residual block,

$$
\mathbf { h } ^ { \ell + 1 } = \mathbf { h } ^ { \ell } + \mathrm { F F N } \left( \mathrm { N o r m } ( \mathbf { h } ^ { \ell } ) \right) ,
$$

rescaling $\mathbf { h } ^ { \ell }$ afects the identity and residual branches diferently. Learnable normalization gains, nonhomogeneous activations, and the embedding and output layers introduce further scale dependence. Consequently, ELR collapse is not implied by the exact reparameterization symmetry that motivates ELR in scale-invariant models (van Laarhoven, 2017; Li et al., 2020); see Appendix A.1 for a detailed discussion.

High-precision collapse is also not automatic. Our ablations identify three factors that strongly afect its precision: QK-Norm (Henry et al., 2020), learnable RMSNorm gains, and suficiently gradual variation in the LR–norm realization of a prescribed ELR schedule. Removing QK-Norm weakens the collapse even though training remains stable. Fixing the RMSNorm gains weakens it further, despite making the parameterization more scale invariant. These findings rule out static scale symmetry as a complete explanation and delineate an empirical regime for high-precision collapse, while its microscopic dynamical origin remains open.

Beyond collapse: mediation and transfer. The prescribed-ELR experiments establish the central phenomenon: diferent LR and norm realizations of the same ELR schedule produce nearly identical loss trajectories. We next subject the claim that ELR governs loss dynamics to two stronger tests.

• From prescribed norms to practical norm control. The prescribed-ELR experiments establish collapse by directly controlling the parameter norm trajectory. We next ask whether the same principle holds under practical norm-control mechanisms. For weight decay and Hyperball (Wen et al., 2026), we leave the norm control in place and adapt only the LR of a comparison run to match the target ELR schedule. This recovers the target loss dynamics with collapse errors of $4 . 8 \times 1 0 ^ { - 3 }$ and $1 . 2 \times 1 0 ^ { - 3 }$ , respectively. Thus, ELR collapse extends beyond prescribed norm trajectories to practical norm-control methods, whose efects on loss are largely captured by the ELR schedule they induce.

• From trajectory matching to scaling-law transfer. We next ask whether ELR enables a predictive scaling law to transfer across norm-control methods. We fit the functional scaling law (FSL) (Li et al., 2025a) using only non-Hyperball runs and evaluate it, without refitting, on Hyperball runs absent from the fitting set. When parameterized by ELR, the fitted FSL remains accurate; when parameterized by LR, it develops a large systematic bias, increasing the prediction error by 12.9×. Thus, ELR does more than align individual trajectories: it enables FSL to transfer across distinct norm-control methods.

ELR explains delayed acceleration. Finally, we use the ELR picture to explain the efect of delayed acceleration: a norm-controlled run can initially have higher loss, yet later overtake an uncontrolled baseline (Hofmann et al., 2022; Liu et al., 2025; Wen et al., 2026). We show that this reversal is captured by the ELR schedule induced by norm control and can be reproduced through direct norm control. Thus, delayed acceleration emerges as a consequence of ELR reshaping rather than a mechanism specific to weight decay or Hyperball.

Taken together, our results establish ELR collapse as a precise, conditional, and operational macroscopic law for pretraining loss dynamics. At the level of training loss, LR scheduling and norm control are distinct mechanisms for realizing a common ELR schedule, rather than independent dynamical coordinates.

## 2 Related Work

ELR under scale invariance. Normalization has motivated extensive study of scale-invariant objectives satisfying $\mathcal { L } ( c \mathbf { w } ) = \mathcal { L } ( \mathbf { w } )$ for all $c > 0 .$ . Such objectives depend on W only through its direction $\widehat { \bf W } = { \bf W } / \| { \bf W } \|$ . For gradient descent, the ELR governing the dynamics of $\widehat { \mathbf { W } } _ { k }$ is $\eta _ { k } / \| \mathbf { W } _ { k } \| ^ { 2 }$ (Hofer et al., 2018). For scale-insensitive optimizers such as signSGD and Adam, the corresponding ELR is $\eta _ { k } / \lVert \mathbf { W } _ { k } \rVert$ (van Laarhoven, 2017). ELR has since been widely used to analyze the optimization dynamics of normalized models (Li and Arora, 2020; Heo et al., 2021; Li et al., 2020; Wan et al., 2021; Kodryan et al., 2022). This line of work derives ELR from exact scale invariance. In contrast, we show that ELR governs loss dynamics to high precision in language model pretraining despite the lack of such symmetry.

Norm control in LLM pretraining. Weight decay is increasingly understood as a dynamical mechanism that regulates norms and thereby controls efective angular updates for scale-invariant models, rather than solely as a regularizer (Zhang et al., 2019; Wan et al., 2021; Kosson et al., 2024). In LLM pretraining, weight decay has been found important for training stability and hyperparameter transfer across scales (D’Angelo et al., 2024; Wortsman et al., 2024; Kosson et al., 2026). Loshchilov (2023) proposed weight norm control, which generalizes decoupled weight decay by steering parameter norms toward prescribed targets. Recent methods impose explicit norm constraints: Hyperball constrains parameters to prescribed Frobenius-norm spheres (Wen et al., 2026), whereas SSO imposes spectral-norm constraints (Xie et al., 2026).

Prior experiments also provide evidence that norm control afects loss through ELR. D’Angelo et al. (2024) adjusted the LR of runs without weight decay to reproduce the ELR induced by weight decay, obtaining approximate loss alignment. Li et al. (2025b) later reported late-stage loss alignment when $\eta \lambda ,$ an equilibrium proxy for ELR, is matched. We extend these observations into a quantitative empirical law: ELR matching yields collapse errors of order $1 0 ^ { - 3 }$ across diverse training configurations, and we identify the conditions required for collapse at this precision. We further test whether the efects of both weight decay and explicit norm control on loss dynamics are indeed governed by ELR.

Functional scaling laws. Li et al. (2025a) showed that functional scaling laws (FSL) provide a reliable model for predicting loss dynamics in LLM pretraining under general LR schedules. We show that replacing LR with ELR improves predictive accuracy and, more importantly, enables the fitted FSL to transfer across norm-control regimes. The resulting elr-FSL explains why norm-controlled runs can initially have higher loss than the uncontrolled baselines, yet later overtake them and attain lower final loss—a phenomenon we call delayed acceleration.

Concurrent work. Two concurrent studies examine norm-aware efective step sizes in LLM pretraining. Yang et al. (2026) show that expressing the optimal LR in terms of ELR reduces its nonlinear scaling and improves extrapolation across model sizes and data budgets. Xiao et al. (2026) derive an angular ELR and match it between weight-decayed and Hyperball-controlled Muon runs, approximately recovering the corresponding loss dynamics. Our work addresses the broader trajectory-level hypothesis that ELR governs loss dynamics across training configurations. We test this hypothesis quantitatively and systematically: across optimizers, architectures, datasets, and model scales, matching ELR aligns full loss trajectories with mean discrepancies of only a few $\times 1 0 ^ { - 3 }$ . We further identify the conditions controlling this precision and show that ELR enables FSL to transfer across norm-control regimes.

## 3 ELR, Norm Control, and Collapse Metrics

Efective learning rate and norm control. Let $\mathbf { W } _ { k }$ be a matrix-valued parameter in a Transformer updated with LR $\eta _ { k }$ at step k. We define its efective learning rate (ELR) as

$$
\eta _ { k } ^ { \mathrm { e f f } } : = \frac { \eta _ { k } } { \Vert \mathbf { W } _ { k } \Vert _ { F } } ,\tag{1}
$$

and refer to $\{ \eta _ { k } ^ { \mathrm { e f f } } \} _ { k }$ as its ELR schedule. Decoupled weight decay updates $\mathbf { W } _ { k }$ according to

$$
\mathbf { W } _ { k + 1 } = ( 1 - \eta _ { k } \lambda ) \mathbf { W } _ { k } - \eta _ { k } \mathbf { U } _ { k } ,
$$

where $\mathbf { U } _ { k }$ denotes the optimizer update. Rather than prescribing $\| \mathbf { W } _ { k } \| _ { F } .$ , weight decay regulates its evolution through the shrinkage factor $1 - \eta _ { k } \lambda$ , thereby shaping ELR implicitly. Hyperball (Wen et al., 2026), by contrast, constrains each matrix parameter to a sphere of prescribed radius $R \colon$

$$
\mathbf { W } _ { k + 1 } = \mathtt { N o r m } _ { R } ( \mathbf { W } _ { k } - \eta _ { k } \mathtt { N o r m } _ { R } ( \mathbf { U } _ { k } ) ) , \qquad \mathtt { N o r m } _ { R } ( \mathbf { Q } ) : = R \frac { \mathbf { Q } } { \lVert \mathbf { Q } \rVert _ { F } } .
$$

Note that the coeficient multiplying $\mathbf { U } _ { k }$ before the outer projection is $\tilde { \eta } _ { k } = \eta _ { k } R / \| \mathbf { U } _ { k } \| _ { F }$ . Since the weight norm is $R ,$ , the corresponding ELR is $\eta _ { k } ^ { \mathrm { e f f } } = \tilde { \eta } _ { k } / \lVert \mathbf { W } _ { k } \rVert _ { F } = \eta _ { k } / \lVert \mathbf { U } _ { k } \rVert _ { F }$ for Hyperball.

Quantifying ELR collapse. Each collapse experiment is a paired controlled comparison. The target and comparison runs use the same parameter initialization and data order, with all other sources of training randomness held fixed. They difer only in the LR and/or norm-control intervention under study, resulting in diferent LR and parameter-norm trajectories while their ELR trajectories are matched. Let $\{ L _ { k } \}$ and $\{ L _ { k } ^ { \mathrm { r e f } } \}$ denote the loss trajectories of a matched run and its reference run, respectively. We define the loss residual and mean collapse error as

$$
r _ { k } : = L _ { k } - L _ { k } ^ { \mathrm { r e f } } , \quad \quad \Delta _ { \mathrm { c o l l } } : = \frac { 1 } { | T | } \sum _ { k \in \mathcal { T } } | r _ { k } | ,\tag{2}
$$

where $\tau$ denotes the set of training steps over which collapse is evaluated.

Calibrating the scale of collapse error. There is no configuration-independent threshold for when two pretraining loss trajectories difer meaningfully. Nevertheless, recent LLM optimizer studies routinely use 10<sup>−2</sup>-scale loss diferences to distinguish optimizer and hyperparameter choices (Liu et al., 2024, 2025; Wang et al., 2025a,b; Chen et al., 2026). This is a practical convention rather than a universal threshold.

For an internal calibration, we measure run-to-run variation in our most extensively tested configuration, Llama-124M trained on FineWeb. Repeating the same configuration while changing only the initialization seed or data-order seed yields variations of $1 . 1 1 \times 1 0 ^ { - 2 }$ and $1 . 6 2 \times 1 0 ^ { - 2 }$ respectively, under the same metric; see Appendix A.2. Both substantially exceed the collapse errors of a few $\times 1 0 ^ { - 3 }$ observed in this setting.

Together, these references place the typical collapse error of a few $\times 1 0 ^ { - 3 }$ well below both seed-induced variation in our reference configuration and the $1 0 ^ { - 2 }$ -scale improvements regarded as meaningful in LLM optimizer studies.

## 4 The ELR Collapse

Matching ELR. Consider a reference run A with LR schedule $\{ \eta _ { k } ^ { A } \} _ { k }$ and norm evolution $R _ { k } ^ { A } : = \| \mathbf { W } _ { k } ^ { A } \| _ { F }$ , which induce the ELR schedule $\gamma _ { k } : = \eta _ { k } ^ { A } / R _ { k } ^ { A }$ . For a second run B with a diferent LR schedule $\{ \eta _ { k } ^ { B } \} _ { k }$ , matching $\gamma _ { k }$ requires the target norm $R _ { k } ^ { B } : = \eta _ { k } ^ { B } / \gamma _ { k }$ . We enforce $\| \mathbf { W } _ { k } ^ { B } \| _ { F } = R _ { k } ^ { B }$ at every optimization step. By construction, runs A and B have diferent LR and norm schedules but the same ELR schedule. More details are deferred to Appendix B.2.

## 4.1 Quantitative Validation across Training Configurations

We begin with two representative systems: a dense Llama model with 124M parameters and a Qwen3-MoE model with 586M parameters, both trained with AdamW. In each setting, we prescribe a common warmup–stable–decay (WSD) ELR schedule and realize it through four diferent LR and target-norm schedules. The reference run uses the WSD LR schedule, while the other three use linear-up, linear-down, and sinusoidal LRs, with their target norms adjusted to preserve the prescribed ELR.

Figure 1 presents the central result. Despite substantial diferences in the LR and parameternorm trajectories, the corresponding loss curves nearly coincide throughout training. In the order linear-up, linear-down, and sinusoidal, the mean collapse errors are

$$
\begin{array} { r l } { \mathrm { L l a m a ~ ( 1 2 4 M ) } { \mathrm { : } } } & { \quad \Delta _ { \mathrm { c o l l } } = ( 1 . 8 , 2 . 5 , 2 . 6 ) \times 1 0 ^ { - 3 } , } \\ { \mathrm { Q w e n 3 - M o E ~ ( 5 8 6 M ) } { \mathrm { : } } } & { \quad \Delta _ { \mathrm { c o l l } } = ( 3 . 0 , 4 . 1 , 3 . 3 ) \times 1 0 ^ { - 3 } . } \end{array}
$$

Figure 2 further shows the corresponding residuals $r _ { k }$ throughout training. They generally fluctuate around zero without systematic drift, although transient deviations occur. ELR collapse is therefore a trajectory-level quantitative agreement with mean discrepancies of only a few $\times 1 0 ^ { - 3 }$ , rather than exact pointwise equality of the loss curves.

![](images/3314463e89637e06dd8fc2ca7d79f42a083c6b720144ef1f78d194f19c9645cd.jpg)

![](images/d61ef55bdc2f5f2e20ea08de2faecbf2819773b81601d2045d2872a682cb5397.jpg)  
Figure 2: ELR-matched residuals remain centered near zero without systematic drift. Signed loss residuals $r _ { k } = L _ { k } - L _ { k } ^ { \mathrm { r e f } }$ for the linear-up, linear-down, and sinusoidal LR schedules over the post-warmup evaluation window. Both panels use the prescribed-ELR protocol of Figure 1; the dashed line marks zero residual.

We next test the same protocol across a broader range of configurations:

• Models: Transformer architectures spanning dense (Llama and Qwen3), MoE (Qwen3- MoE), and linear attention based on Kimi Delta Attention (KDA) (Zhang et al., 2025), with model sizes from 100M to 1B parameters;

• Datasets: FineWeb, C4, and OpenWebText;

• Optimizers: AdamW, Muon, and Signum.

Experimental details and complete results are provided in Appendices B and C, respectively. Across the 26 ELR-matched comparisons summarized in Table C.1, the median collapse error is $2 . 5 \times 1 0 ^ { - 3 }$ , and all comparisons remain below $5 \times 1 0 ^ { - 3 }$ . Comparable collapse precision is also observed for ViTs trained on ImageNet in Appendix G. Together, these results demonstrate the robustness of ELR collapse across architectures, model scales, datasets, and optimizers.

## 4.2 When Is ELR Collapse Precise?

As discussed in the introduction and detailed in Appendix A.1, transformers are not exactly scale invariant, so high-precision ELR collapse does not follow from an exact architectural symmetry. We therefore explored a range of model parameterizations and training configurations to identify which factors control ELR collapse. Two sensitivities emerged consistently: normalization design and the timescale of the LR–norm realization.

QK-Norm and learnable RMSNorm gains unexpectedly improve collapse precision. Although QK-Norm is not required for stable training in our settings, our exploration across a broad range of module configurations revealed that it substantially improves collapse precision. Our default models therefore apply QK-Norm (Henry et al., 2020) after the query and key projections and RMSNorm (Zhang and Sennrich, 2019), with learnable gains, before the attention and FFN blocks. To assess their roles, we repeat the four-schedule prescribed-ELR protocol used in Figures 1 and 2 under three nested normalization configurations: the default model, the model without QK-Norm, and the model without QK-Norm and with fixed RMSNorm gains.

![](images/4b11f0c3370f6222773d33a0f4a2bac77a62dd4d527e8004e2759f140b1d37b4.jpg)

![](images/5ab5a56a58fbf7f9a750339bceb9ef15cb511415381e392f7ea38ecd7e98bf53.jpg)  
Figure 3: QK-Norm and learnable RMSNorm gains sharpen ELR collapse. Experiments use Llama-124M trained on FineWeb with AdamW. Starting from the default configuration, the ablations sequentially remove QK-Norm and then fix RMSNorm gains. Left: mean absolute loss residual at each step, averaged over the three ELR-matched schedules. Right: Mean collapse error; bars show the mean across schedules, and error bars show their range. Fixing the RMSNorm gains produces the largest error despite making the parameterization more scale invariant. In the plot labels, RMS denotes RMSNorm.

Figure 3 shows that removing QK-Norm increases the mean collapse error from $2 . 3 \times 1 0 ^ { - 3 }$ to $5 . 2 \times 1 0 ^ { - 3 }$ . Starting from the model without QK-Norm, fixing the remaining RMSNorm gains further increases the error to $1 . 8 4 \times 1 0 ^ { - 2 }$ . The combined ablation therefore increases the collapse error by nearly an order of magnitude. Thus, both QK-Norm and learnable RMSNorm gains substantially improve collapse precision, although neither is designed for this purpose.

The gain ablation is particularly counterintuitive. Fixing the gains makes the parameterization more scale invariant, yet increases the collapse error by a factor of approximately 3.5 relative to the learnable-gain counterpart. High-precision ELR collapse therefore cannot be explained solely by static architectural scale invariance. Instead, this result suggests an additional dynamical role for normalization and gain adaptation, whose mechanism remains to be understood.

Faster LR–norm variation degrades ELR collapse. Starting from a WSD LR schedule $\eta _ { k } ^ { \mathrm { W S D } }$ , we introduce the sinusoidal modulation

$$
\eta _ { k } ^ { ( m ) } = s _ { m } ( k ) \eta _ { k } ^ { \mathrm { W S D } } , \qquad s _ { m } ( k ) = 1 + 0 . 5 \sin \left( \frac { 2 \pi m k } { N } \right) ,
$$

where N is the training horizon and m controls the number of oscillation cycles. For each $m ,$ we multiply the target-norm schedule by the same factor $s _ { m } ( k )$ , so that the prescribed ELR remains unchanged. Varying m therefore changes the timescale of the LR–norm realization while preserving the ELR schedule.

Figure 4(left) shows that, over the tested range, the mean collapse error increases monotonically from $2 . 8 \times 1 0 ^ { - 3 }$ at two cycles to $7 . 5 \times 1 0 ^ { - 3 }$ at 32 cycles. Figure $4 ( \mathrm { r i g h t } )$ reveals that the residuals oscillate at the imposed frequency, with substantially larger amplitude under faster modulation. The degradation is therefore primarily a short-timescale response rather than a systematic divergence of the loss trajectories.

![](images/863bde85e993713c522cc0a895758564ccc57114cdadb2362496647a52b09b6a.jpg)

![](images/b3224188a9b52fadc9f98ebbfb65900fbc00bb698d22ec7d43bc8fdc39e661ad.jpg)

![](images/c42c835b2ba1b56e14f20cf40184539a675c940de550b08cf6be0ed734b0b3a0.jpg)  
Figure 4: Rapid LR–norm variation degrades ELR collapse. Experiments use Llama-124M trained on FineWeb with AdamW. Left: The collapse error $\Delta _ { \mathrm { c o l l } }$ increases monotonically with the number of modulation cycles. Right: Loss residuals for representa tive 2- and 32-cycle modulations. The right insets show the corresponding LR schedules together with the WSD reference. In each run, the target-norm schedule is modulated by the same factor as the LR, preserving the prescribed ELR schedule. Faster modulation produces larger oscillatory residuals at the imposed frequency.

Together, these ablations show that ELR is an accurate but not exact coordinate for loss dynamics: collapse precision depends on normalization design and the timescale of joint LR–norm variation. Its origin must therefore involve training dynamics beyond static scale invariance, but the underlying mechanism remains open.

## 5 Weight Decay and Hyperball Act through ELR

The preceding experiments establish ELR collapse using prescribed LR and norm schedules. We now test whether ELR also mediates the efects of two practical norm-control methods: weight decay and Hyperball. For each method, we train a norm-controlled target run and record the ELR trajectory it induces. We then train a comparison run under a diferent norm-control setting, leaving its norm trajectory unprescribed and adapting only its LR to match the target ELR. If this LR-only intervention recovers the target loss dynamics despite diferent norm trajectories and norm-control mechanisms, then the method’s efect on loss is mediated by the ELR it induces. The optimizer-specific matching rules are given in Appendix B.2.

• Weight decay. We take an AdamW run with λ = 0.1 as the target and compare it with a run using λ = 0. Under the original LR schedule, removing weight decay changes the norm evolution and causes the loss trajectory to deviate substantially from the target. When we instead adapt only the LR of the λ = 0 run to match the target ELR, its entire loss trajectory closely follows the target throughout training, with a mean collapse error of $4 . 8 \times 1 0 ^ { - 3 }$ ; see Figure 5(a).

![](images/925d2133a9032c3dae8e63bfea385a7495083f940fd50924feecbac3ce3aebc7.jpg)  
Figure 5: Weight decay and Hyperball afect loss dynamics through ELR. Experiments use Llama-124M trained on FineWeb. Dashed gray curves denote the normcontrolled targets, blue curves use the original LR, and red curves adapt the LR to match the target ELR. (a) An AdamW run with weight decay $\lambda = 0 . 1$ serves as the target. Removing weight decay changes the loss dynamics under the original LR, whereas ELR matching recovers the target loss with a collapse error of $4 . 8 \times 1 0 ^ { - 3 }$ . (b) A MuonH run serves as the target. Replacing Hyperball with weight decay changes the loss dynamics under the original LR, whereas ELR matching reduces the collapse error (steps ⩾ 7.5k) to $1 . 2 \times 1 0 ^ { - 3 }$

• Hyperball. We take MuonH (Muon with Hyperball) as the target and adapt only the LR of a MuonW run (Muon with weight decay) to match the target ELR. Figure 5(b) shows that MuonW with its original LR deviates substantially from MuonH, whereas ELR matching brings the loss trajectories into close alignment after a brief initial transient. This transient occurs at the start of matching, when the adapted LR is largest. We therefore evaluate collapse from step 7.5k onward, obtaining a mean error of only $1 . 2 \times 1 0 ^ { - 3 }$

Appendix D provides full trajectory diagnostics for both experiments and reports the reversedirection interventions, with each norm-control setting serving in turn as the target. These reverse interventions yield the same conclusion.

Together, these LR-only interventions show that weight decay and Hyperball afect loss dynamics primarily through the ELR trajectories they induce. Apart from the brief Hyperball onset transient, matching ELR reduces the loss discrepancies between diferent norm-control mechanisms to the $1 0 ^ { - 3 }$ scale. This does not imply that norm control necessarily improves convergence: its efect depends on the ELR trajectory it induces.

## 6 ELR Enables Scaling-Law Transfer

Pairwise ELR matching shows that individual loss trajectories can be aligned across diferent norm-control methods. This does not yet establish that these methods obey a common predictive law. We therefore ask a stronger question: can a scaling law fitted without Hyperball transfer to Hyperball when expressed in ELR coordinates?

Functional scaling laws. The functional scaling law (FSL) of Li et al. (2025a) models loss dynamics under a general LR schedule. Given an LR schedule $\pmb { \eta } = ( \eta _ { 0 } , \dots , \eta _ { N } )$ , define the intrinsic training time $\begin{array} { r } { t _ { k } : = \sum _ { j = 0 } ^ { k } \eta _ { j } } \end{array}$ . FSL models the loss at step k as

$$
L _ { k } = L _ { \infty } + S ( t _ { k } ) + N _ { k } , \qquad N _ { k } = \sum _ { j = 0 } ^ { k } K ( t _ { k } - t _ { j } ) \eta _ { j } ^ { 2 } ,\tag{3}
$$

where $L _ { \infty }$ is the irreducible loss. For large t, the signal term and memory kernel satisfy $S ( t ) \asymp t ^ { - s }$ and $K ( t ) \asymp t ^ { - \gamma }$ for some $s , \gamma > 0$ . The signal term decreases with the accumulated training time, whereas $N _ { k }$ captures the optimization noise accumulated along the trajectory. Increasing the LR therefore accelerates signal learning by advancing the intrinsic time, but also increases noise injection through the factor $\eta _ { j } ^ { 2 }$ . We refer to this original LR-based formulation as lr-FSL.

Motivated by ELR collapse, we define elr-FSL by replacing $\eta _ { j }$ with $\eta _ { j } ^ { \mathrm { e f f } }$ throughout Equation (3), including both the intrinsic training time and the noise term, while leaving the functional form unchanged. The two formulations therefore difer only in whether loss dynamics are parameterized by LR or ELR. In both cases, the realized LR or ELR schedule is supplied as input. We test scaling-law transfer across norm-control methods in this section and use elr-FSL again in Section 7 to analyze delayed acceleration.

Transfer across norm-control methods. We train a 124M-parameter Llama model on FineWeb under ten configurations: four without weight decay, four with weight-decay coeficient $\lambda = 0 . 1$ , and two with Hyperball. Both FSL variants are fitted on the same four non-Hyperball trajectories. Four additional non-Hyperball trajectories evaluate generalization to held-out LR schedules under norm-control methods represented in the fitting set; we refer to these as in-distribution (ID) runs. The two Hyperball trajectories test out-of-distribution (OOD) transfer to a norm-control method absent from the fitting set. Appendix E provides the complete data split, parameterization, and fitting procedure.

Figure 6 summarizes the results. On the two unseen Hyperball runs, elr-FSL remains accurate without refitting, with a mean RMSE of 0.0212, whereas lr-FSL develops a large systematic bias and reaches 0.2508, a 11.83× larger error. ELR also improves prediction on the held-out ID runs, reducing the mean RMSE from 0.0239 to 0.0133. Thus, the advantage of ELR is not limited to pairwise trajectory alignment: it enables a scaling law fitted under one set of norm-control methods to transfer to another:

(a) Prediction error across data splits
<table><tr><td>Evaluation</td><td># Runs lr-FSL RMSE ↓ elr-FSL RMSE ↓</td><td></td><td> $\mathrm { R M S E _ { \mathrm { l r } } / R M S E _ { \mathrm { e l r } } }$  ↑</td></tr><tr><td>Fit</td><td>4</td><td>0.0183</td><td>0.0131 1.40×</td></tr><tr><td>ID: held-out</td><td>4</td><td>0.0239</td><td>0.0133 1.80×</td></tr><tr><td>OOD: Hyperball</td><td>2</td><td>0.2508</td><td>0.0212 11.83×</td></tr></table>

![](images/806bed525442d01cce6138ff6c55158f727cd46bf4419ecf89d9e64af69b1cae.jpg)

(c) Transfer to Hyperball  
![](images/6afd101f48504c582594225537b8841c8d3a822ca25f396319694b7702ebf176.jpg)  
Figure 6: ELR enables FSL to transfer to unseen Hyperball runs. Experiments use Llama-124M trained on FineWeb with Adam under three norm-control regimes: no weight decay, decoupled weight decay, and Hyperball. (a) Mean RMSE on four fitting runs, four held-out ID runs, and two OOD-Hyperball runs. Both FSL variants are fitted on the same four non-Hyperball curves and evaluated without refitting. The lr-FSL/elr-FSL RMSE ratio increases from 1.8× on the held-out ID runs to 12.9× on the OOD-Hyperball runs. (b) A representative held-out ID using weight-decay coeficient $\lambda = 0 . 1$ . (c) A representative OOD-Hyperball run; no Hyperball data are used for fitting. Without refitting, elr-FSL continues to track the observed loss, whereas lr-FSL breaks down. Table entries are averages across runs; legend values report RMSE for the individual runs shown.

## 7 Delayed Acceleration: Explanation and Control

Having shown that ELR provides a common predictive coordinate across norm-control regimes, we now use it to explain a recurring late-stage benefit of norm control. Weight decay can be crucial for sustaining optimizer eficiency as LLM training scales; see Liu et al. (2025, Figure 2) and Hofmann et al. (2022, Figure A7). Yet its benefit often emerges only late. As shown in Figure 7(left), the weight-decayed run initially has higher loss than the unregularized baseline, but later overtakes it and attains a lower final loss. Hyperball exhibits the same qualitative behavior; see Appendix F. We call this late-emerging gain delayed acceleration and show that it is governed by the ELR schedule induced by norm control.

![](images/96c4cb2690be97bc27b6898585dfb9b253880e5857cc85c82815af04272e7c5f.jpg)  
Figure 7: ELR explains and controls delayed acceleration. Experiments use Llama-124M trained on FineWeb with AdamW. Left: Weight decay initially yields higher loss than the unregularized baseline, but overtakes it during the terminal phase. ELR-guided norm control further lowers the final loss. Middle: Corresponding norm evolution. Explicit norm control approximately follows the weight-decay run early, then increases the norm more rapidly after the vertical dashed line. Right: Corresponding ELR trajectories, with the shared LR shown in the inset. The accelerated late-stage norm growth produces a faster ELR decay.

Norm growth prematurely suppresses ELR. Figure 7(middle and right) reveals the origin of the reversal. Without weight decay, the parameter norm grows steadily, causing the ELR $\eta _ { k } ^ { \mathrm { e f f } } = \eta _ { k } / \Vert \mathbf { W } _ { k } \Vert _ { F }$ to decay substantially faster than the nominal LR. Weight decay restrains this norm growth and thereby sustains a larger ELR for longer. Thus, the two runs follow markedly diferent efective optimization trajectories despite sharing the same LR schedule. From the ELR perspective, the role of weight decay in loss dynamics is to prevent premature ELR decay.

The gain is acquired early and revealed late. The signal–noise decomposition of FSL explains why the benefit of this larger ELR is delayed. Let

$$
t _ { k } ^ { \mathrm { e f f } } = \sum _ { j = 0 } ^ { k } \eta _ { j } ^ { \mathrm { e f f } }
$$

denote the efective training time. By sustaining a larger ELR, weight decay accumulates $t _ { k } ^ { \mathrm { e f f } }$ more rapidly and lowers the signal-learning term $S ( t _ { k } ^ { \mathrm { e f f } } )$ . At the same time, the larger ELR produces greater noise accumulation, which can initially mask this signal-learning advantage. As ELR decays later in training, new noise injection diminishes while previously accumulated noise continues to be forgotten. The advantage acquired earlier is then revealed as a lower training loss. Delayed acceleration therefore reflects a temporal separation: the gain is acquired early but revealed late.

ELR reshaping controls delayed acceleration. The previous explanation suggests improving the ELR schedule induced by weight decay with λ = 0.3: preserve its relatively large ELR early in training, but reduce ELR more aggressively during the late phase. We realize this modification through a prescribed norm schedule that approximately follows the $\lambda = 0 . 3$ run early and grows faster later; see Figure 7(middle). Under the shared LR schedule, this produces a similar early ELR but a smaller late-stage ELR; see Figure 7(right). The resulting run preserves delayed acceleration and attains a lower final loss than the weight-decay baseline; see Figure 7(left). Crucially, this improvement is achieved by increasing the parameter norm more rapidly in the late phase, showing that the benefit does not arise from maintaining a uniformly smaller norm. Rather, norm control afects loss dynamics through the ELR schedule it realizes. Although the prescribed schedule is heuristic rather than optimized, the intervention turns the ELR explanation into a control principle: delayed acceleration can be shaped directly through ELR and is not specific to weight decay itself.

Appendix F provides additional experiments, including analogous delayed acceleration under Hyperball and its explanation through the same ELR-based mechanism.

## 8 Conclusion and Discussion

Our main result is a macroscopic empirical law for language model pretraining: LR and parameter norm govern loss dynamics primarily through their ratio, ELR. Across the regimes studied, distinct LR schedules and norm-control mechanisms produce nearly identical loss dynamics when they induce the same ELR schedule. This reduction is analogous to the role of density in a macroscopic physical system: mass and volume may vary substantially, yet the relevant bulk behavior is governed primarily by density—their ratio—rather than by either quantity alone. Likewise, LR scheduling and norm control shape pretraining loss dynamics primarily through the ELR trajectory they jointly induce.

Microscopic mechanism and scope. ELR collapse is a macroscopic law, not a consequence of exact scale symmetry. Transformers are not exactly scale invariant, and learnable RMSNorm gains sharpen collapse even though they make the parameterization less scale invariant. Rapid LR– norm variation also produces structured oscillatory deviations at the imposed frequency. These observations suggest a dynamical compensation mechanism with a finite response timescale. A satisfactory theory should explain both why the many coupled degrees of freedom in Transformer training admit such an accurate low-dimensional description at the level of loss and why its precision depends on normalization design and the timescale of the LR–norm realization. Our claims are correspondingly limited to loss dynamics: matched loss trajectories need not imply matched parameters, representations, or downstream performance.

ELR as a coordinate for cross-scale transfer. Hyperparameter transfer is commonly posed as separate rules for the LR, weight-decay coeficient, and norm target (Yang et al., 2021; Ghosh et al., 2026; Kosson et al., 2026). ELR suggests a more structured two-stage problem: first determine the ELR schedule appropriate for a target scale, then choose a stable LR–norm realization. This separates two questions that current tuning practice often conflates: what efective trajectory should training follow, and how should that trajectory be realized? A useful transfer theory should determine how the desired ELR schedule changes with model size, batch size, data budget, and training horizon, while allowing the LR and norm-control mechanism to be chosen according to the constraints of each setting. Such a formulation would replace method-specific transfer heuristics with a common dynamical target.

ELR-first pretraining. Current practice often treats the LR schedule, weight decay, and norm constraints as separate design choices. Our results suggest a diferent abstraction: the ELR schedule is the design object (Li et al., 2026; Wang et al., 2026), whereas LR scheduling and norm control are mechanisms for realizing it. Norm control is useful not because a particular norm is intrinsically preferable, but because it expands and shapes the set of ELR schedules that training can realize. The explicit norm-control experiment makes this distinction concrete: increasing the parameter norm more rapidly late in training improves the final loss by producing a more favorable ELR decay. This does not mean that ELR captures every aspect of training; stability, numerical precision, and transferability remain additional constraints on how an ELR schedule should be realized. For loss dynamics, however, the practical principle is simple: design the ELR schedule first, then choose the LR and norm-control mechanism that realizes it robustly.

## Acknowledgement

Lei Wu is supported by the National Natural Science Foundation of China (NSFC 12522120, NSFC 92470122, and NSFC 12288101). Zihan Liu is supported by Ant Group Research Intern Program. We thank Shengtao Guo and Binghui Li for helpful discussions.

## References

Bernstein, J., Wang, Y.-X., Azizzadenesheli, K. and Anandkumar, A. (2018). signSGD: Compressed optimisation for non-convex problems. In International Conference on Machine Learning. (cited on page 20)

Chen, L., Li, J., Liang, K., Su, B., Xie, C., Pierse, N. W., Liang, C., Lao, N. and Liu, Q. (2026). Cautious weight decay. In International Conference on Learning Representations. (cited on page 4)

D’Angelo, F., Andriushchenko, M., Varre, A. and Flammarion, N. (2024). Why do we need weight decay in modern deep learning? In Advances in Neural Information Processing Systems, vol. 37. (cited on pages 2 and 3)

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K. and Fei-Fei, L. (2009). ImageNet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition. (cited on page 29)

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J. and Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations. (cited on page 29)

Ghosh, N., Wu, D. and Bietti, A. (2026). Understanding the mechanisms of fast hyperparameter transfer. In International Conference on Learning Representations, vol. 2026. (cited on page 11)

Gokaslan, A. and Cohen, V. (2019). OpenWebText corpus. http://Skylion007.github.io/ OpenWebTextCorpus. (cited on page 19)

Henry, A., Dachapally, P. R., Pawar, S. S. and Chen, Y. (2020). Query-key normalization for transformers. In Findings of the Association for Computational Linguistics: EMNLP 2020. (cited on pages 2 and 6)

Heo, B., Chun, S., Oh, S. J., Han, D., Yun, S., Kim, G., Uh, Y. and Ha, J.-W. (2021). AdamP: Slowing down the slowdown for momentum optimizers on scale-invariant weights. In International Conference on Learning Representations. (cited on page 3)

Hofer, E., Banner, R., Golan, I. and Soudry, D. (2018). Norm matters: Eficient and accurate normalization schemes in deep networks. In Advances in Neural Information Processing Systems, vol. 31. (cited on page 3)

Hofmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L. A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Vinyals, O., Rae, J. W. and Sifre, L. (2022). Training compute-optimal large language models. In Advances in Neural Information Processing Systems, vol. 35. (cited on pages 3 and 10)

Hu, S., Tu, Y., Han, X., He, C., Cui, G., Long, X., Zheng, Z., Fang, Y., Huang, Y., Zhao, W., Zhang, X., Thai, Z. L., Zhang, K., Wang, C., Yao, Y., Zhao, C., Zhou, J., Cai, J., Zhai, Z., Ding, N., Jia, C., Zeng, G., Li, D., Liu, Z. and Sun, M. (2024). MiniCPM: Unveiling the potential of small language models with scalable training strategies. In Conference on Language Modeling. (cited on page 20)

Kodryan, M., Lobacheva, E., Nakhodnov, M. and Vetrov, D. P. (2022). Training scale-invariant neural networks on the sphere can happen in three regimes. In Advances in Neural Information Processing Systems, vol. 35. (cited on page 3)

Kosson, A., Messmer, B. and Jaggi, M. (2024). Rotational equilibrium: How weight decay balances learning across neural networks. In International Conference on Machine Learning. PMLR. (cited on page 3)

Kosson, A., Welborn, J., Liu, Y., Jaggi, M. and Chen, X. (2026). Weight decay may matter more than µP for learning rate transfer in practice. In International Conference on Learning Representations. (cited on pages 2, 3, and 11)

Krogh, A. and Hertz, J. A. (1991). A simple weight decay can improve generalization. In Advances in Neural Information Processing Systems, vol. 4. (cited on page 1)

Li, B., Chen, F., Huang, Z., Wang, L. and Wu, L. (2025a). Functional scaling laws in kernel regression: Loss dynamics and learning rate schedules. In Advances in Neural Information Processing Systems, vol. 38. (cited on pages 1, 2, 3, and 8)

Li, B., Wang, Z., Chen, F., Zhao, S., Zheng, R. and Wu, L. (2026). Optimal learning-rate schedules under functional scaling laws: Power decay and warmup-stable-decay. arXiv preprint arXiv:2602.06797. (cited on page 12)

Li, B., Wen, J., Zhou, Z., Zhu, J. and Chen, J. (2025b). Eficient hyperparameter tuning via trajectory invariance principle. arXiv preprint arXiv:2509.25049. (cited on page 3)

Li, Z. and Arora, S. (2020). An exponential learning rate schedule for deep learning. In International Conference on Learning Representations. (cited on page 3)

Li, Z., Lyu, K. and Arora, S. (2020). Reconciling modern deep learning with traditional optimization analyses: The intrinsic learning rate. In Advances in Neural Information Processing Systems, vol. 33. (cited on pages 2 and 3)

Liu, H., Li, Z., Hall, D., Liang, P. and Ma, T. (2024). Sophia: A scalable stochastic secondorder optimizer for language model pre-training. In International Conference on Learning Representations. (cited on page 4)

Liu, J., Su, J., Yao, X., Jiang, Z., Lai, G., Du, Y., Qin, Y., Xu, W., Lu, E., Yan, J., Chen, Y., Zheng, H., Liu, Y., Liu, S., Yin, B., He, W., Zhu, H., Wang, Y., Wang, J., Dong, M., Zhang, Z., Kang, Y., Zhang, H., Xu, X., Zhang, Y., Wu, Y., Zhou, X. and Yang, Z. (2025). Muon is scalable for LLM training. arXiv preprint arXiv:2502.16982. (cited on pages 3, 4, 10, and 20)

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L. and Stoyanov, V. (2019). RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692. (cited on page 19)

Loshchilov, I. (2023). Weight norm control. arXiv preprint arXiv:2311.11446. (cited on pages 1 and 3)

Loshchilov, I. and Hutter, F. (2019). Decoupled weight decay regularization. In International Conference on Learning Representations. (cited on page 1)

Penedo, G., Kydlíček, H., Ben Allal, L., Lozhkov, A., Mitchell, M., Rafel, C., von Werra, L. and Wolf, T. (2024). The FineWeb datasets: Decanting the web for the finest text data at scale. In Advances in Neural Information Processing Systems, vol. 37. (cited on page 19)

Rafel, C., Shazeer, N., Roberts, A., Lee, K., Narang, S., Matena, M., Zhou, Y., Li, W. and Liu, P. J. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21 1–67. (cited on page 19)

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E. and Lample, G. (2023). LLaMA: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971. (cited on page 19)

van Laarhoven, T. (2017). L2 regularization versus batch and weight normalization. arXiv preprint arXiv:1706.05350. (cited on pages 2 and 3)

Wan, R., Zhu, Z., Zhang, X. and Sun, J. (2021). Spherical motion dynamics: Learning dynamics of normalized neural network using SGD and weight decay. In Advances in Neural Information Processing Systems, vol. 34. (cited on page 3)

Wang, J., Li, B., Zhou, Z., Wang, M., Zhang, J., Cai, X., Wu, L. et al. (2026). Fast catch-up, late switching: Optimal batch size scheduling via functional scaling laws. In International Conference on Learning Representations, vol. 2026. (cited on page 12)

Wang, J., Wang, M., Zhang, J., Wang, W., Pei, P., Cai, X., Wu, L. et al. (2025a). Gradpower: Powering gradients for faster language model pre-training. arXiv preprint arXiv:2505.24275. (cited on page 4)

Wang, J., Wang, M., Zhou, Z., Yan, J., E, W. and Wu, L. (2025b). The sharpness disparity principle in transformers for accelerating language model pre-training. In International Conference on Machine Learning. (cited on page 4)

Wen, K., Dang, X., Lyu, K., Ma, T. and Liang, P. (2026). Fantastic pretraining optimizers and where to find them II: Hyperball optimization. arXiv preprint arXiv:2606.16899. (cited on pages 1, 2, 3, 4, and 28)

Wortsman, M., Liu, P. J., Xiao, L., Everett, K. E., Alemi, A. A., Adlam, B., Co-Reyes, J. D., Gur, I., Kumar, A., Novak, R., Pennington, J., Sohl-Dickstein, J., Xu, K., Lee, J., Gilmer, J. and Kornblith, S. (2024). Small-scale proxies for large-scale transformer training instabilities. In International Conference on Learning Representations. (cited on page 3)

Xiao, Y., Sun, J., Gao, Z., Wei, Z., Wang, C., Tao, R., Teng, J. and Dai, B. (2026). Hyperball may not be a free lunch. arXiv preprint arXiv:2607.22444. (cited on page 4)

Xie, T., Luo, H., Tang, H., Hu, Y., Liu, J. K., Ren, Q., Wang, Y., Zhao, W. X., Yan, R., Su, B., Luo, C. and Guo, B. (2026). Controlled LLM training on spectral sphere. arXiv preprint arXiv:2601.08393. (cited on pages 1 and 3)

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z. and Qiu, Z. (2025). Qwen3 technical report. arXiv preprint arXiv:2505.09388. (cited on page 19)

Yang, G., Hu, E., Babuschkin, I., Sidor, S., Liu, X., Farhi, D., Ryder, N., Pachocki, J., Chen, W. and Gao, J. (2021). Tuning large neural networks via zero-shot hyperparameter transfer. Advances in Neural Information Processing Systems, 34 17084–17097. (cited on page 11)

Yang, Z., Zhang, H., Xu, J. and Zhang, J. (2026). On the nonlinearity of learning rate scaling for LLM training. arXiv preprint arXiv:2606.29158. (cited on page 4)

Zhang, B. and Sennrich, R. (2019). Root mean square layer normalization. In Advances in Neural Information Processing Systems, vol. 32. (cited on page 6)

Zhang, G., Wang, C., Xu, B. and Grosse, R. (2019). Three mechanisms of weight decay regularization. In International Conference on Learning Representations. (cited on page 3)

Zhang, Y., Lin, Z., Yao, X., Hu, J., Meng, F., Liu, C., Men, X., Yang, S., Li, Z., Li, W., Lu, E., Liu, W., Chen, Y., Xu, W., Yu, L., Wang, Y., Fan, Y., Zhong, L., Yuan, E., Zhang, D., Zhang, Y., Liu, T. Y., Wang, H., Fang, S., He, W., Liu, S., Li, Y., Su, J., Qiu, J., Pang, B., Yan, J., Jiang, Z., Huang, W., Yin, B., You, J., Wei, C., Wang, Z., Hong, C., Chen, Y., Chen, G., Wang, Y., Zheng, H., Wang, F., Liu, Y., Dong, M., Zhang, Z., Pan, S., Wu, W., Wu, Y., Guan, L., Tao, J., Fu, G., Xu, X., Wang, Y., Lai, G., Wu, Y., Zhou, X., Yang, Z. and Du, Y. (2025). Kimi Linear: An expressive, eficient attention architecture. arXiv preprint arXiv:2510.26692. (cited on pages 5 and 19)

## Appendix

A Nontriviality and Precision of ELR Collapse 17   
A.1 Transformers Lack Scale Invariance . 17   
A.2 Interpreting Collapse Errors at the 10<sup>−3</sup> Scale 17   
B Experimental Details 18   
B.1 Training Configurations 18   
B.2 ELR-Matching Protocols 20   
C Quantitative Validation (Supplement to Section 4) 21   
D Weight Decay and Hyperball (Supplement to Section 5) 23   
E Functional Scaling Laws (Supplement to Section 6) 25   
F Delayed Acceleration (Supplement to Section 7) 27   
F.1 Acquired Early, Revealed Late: A Controlled WSD Test . 27   
F.2 Delayed Acceleration under Hyperball . 28   
G ELR Collapse in Vision Transformers 29

## A Nontriviality and Precision of ELR Collapse

## A.1 Transformers Lack Scale Invariance

Our ELR is defined using the norm of the matrix-valued parameters. Transformers, however, also contain trainable vector-valued parameters, most notably the biases of FFN layers and gains in normalization layers. Let $\mathcal { L } ( \mathbf { W } , \phi )$ denote the loss, where W collects the matrix-valued parameters and $\phi$ the vector-valued parameters. The strongest relevant symmetry one might posit is matrix-only scale invariance:

$$
\begin{array} { r } { \boldsymbol { \mathcal { L } } ( c \mathbf { W } , \phi ) = \boldsymbol { \mathcal { L } } ( \mathbf { W } , \phi ) , \qquad \forall c > 0 . } \end{array}\tag{A.1}
$$

However, even if this symmetry held exactly, matching the ELR would not by itself imply matching loss trajectories since the loss is sensitive to the scale of $\phi .$

More seriously, the matrix-only symmetry in Eq. (A.1) also fails for several architectural reasons:

• Pre-norm residual blocks. A pre-norm residual block updates the hidden state as

$$
\mathbf h ^ { \ell + 1 } = \mathbf h ^ { \ell } + F _ { \ell } \left( \mathrm { R M S N o r m } _ { \gamma _ { \ell } } ( \mathbf h ^ { \ell } ) ; \mathbf W _ { \ell } \right) .\tag{A.2}
$$

The identity path preserves the scale of $\mathbf { h } ^ { \ell }$ , whereas RMSNorm removes it before the residual branch. Thus, rescaling the upstream matrices that determine $\mathbf { h } ^ { \ell }$ afects the two paths diferently. Rescaling $\mathbf { W } _ { \ell }$ likewise changes the residual branch relative to the identity path. Hence, the block is not invariant under a rescaling of its matrix parameters.

• Value and output projections. The attention value–output path contains the product

$$
( \mathbf { H } \mathbf { W } _ { V } ) \mathbf { W } _ { O } .
$$

Jointly rescaling $\mathbf { W } _ { V }$ and $\mathbf { W } _ { O }$ by c scales this path by $c ^ { 2 } .$ , even if the attention weights remain fixed. Its magnitude relative to the residual stream therefore changes.

• SwiGLU feed-forward networks. The SwiGLU FFN combines several matrix scales:

$$
\mathrm { M L P } ( \mathbf { H } ) = \left[ \mathrm { S i L U } ( \mathbf { H } \mathbf { W } _ { G } ) \odot ( \mathbf { H } \mathbf { W } _ { U } ) \right] \mathbf { W } _ { D } .\tag{A.3}
$$

Its multiplicative gate couples the scales of multiple projections and moreover, SiLU is not homogeneous. Consequently, a common rescaling of $\mathbf { W } _ { G } , \mathbf { W } _ { U }$ , and $\mathbf { W } _ { D }$ cannot be canceled by a single global scale factor.

• Token embeddings and output logits. Rescaling the token embeddings changes the scale of the initial residual stream, which is not eliminated by subsequent residual blocks. Rescaling the head layer changes the logit scale and hence the efective softmax temperature, to which cross-entropy is not invariant. When the embedding and output matrices are tied, the same parameter introduces scale dependence at both ends of the network.

## A.2 Interpreting Collapse Errors at the $1 0 ^ { - 3 }$ Scale

Fix a source of stochasticity c, and let $\mathit { L } _ { i , k }$ denote the smoothed loss of run i at step $k .$ To match the definition of collapse error, we measure run-to-run variation using the pairwise mean absolute discrepancy

$$
d _ { i j } : = \frac { 1 } { | \mathcal { T } | } \sum _ { k \in \mathcal { T } } \left| L _ { i , k } - L _ { j , k } \right| , \qquad \mathrm { f o r } ~ 1 \leqslant i < j \leqslant n ,
$$

where $\tau$ is the set of evaluation steps. We then define the stochastic variation associated with c by averaging over all unordered pairs:

$$
\Delta _ { \mathrm { s t o c h } } ( c ) : = \frac { 2 } { n ( n - 1 ) } \sum _ { 1 \leqslant i < j \leqslant n } d _ { i j } .
$$

This definition uses the same mean absolute trajectory discrepancy as $\Delta _ { \mathrm { c o l l } }$ , allowing a direct comparison between ELR-collapse errors and ordinary run-to-run variation.

Figure A.1 reports run-to-run variation for Llama-124M trained on FineWeb with AdamW. Repeating the same configuration while changing only the initialization seed or data-order seed gives

$$
\Big | \Delta _ { \mathrm { s t o c h } } ( \mathrm { i n i t } ) = 1 . 1 1 \times 1 0 ^ { - 2 } , \qquad \Delta _ { \mathrm { s t o c h } } ( \mathrm { d a t a \mathrm { - } o r d e r } ) = 1 . 6 2 \times 1 0 ^ { - 2 } . \Big |
$$

For comparison, the three ELR-matched runs in Figure 1 under the same training configurations, which keep both seeds fixed, have collapse errors between $1 . 8 \times 1 0 ^ { - 3 }$ and $2 . 6 \times 1 0 ^ { - 3 }$ . Thus, even the largest collapse error is more than four times smaller than the lower stochastic-variation baseline; across the reported values, the diference is a factor of 4.3–9.0.

![](images/df8bc571732c2cfb09a97d29287ee9e1fb2b587fa363b3866d578a209af85253.jpg)

![](images/77c55fa74a4a2b981e0078712d9a036e1e28bca241fac0a40f3c532796703ec8.jpg)

![](images/b6d1f8d33801a5e80fc6fefc4b6ebf888a47dbb49238fd3ece466e7f1d95cf26.jpg)  
Figure A.1: Run-to-run loss variation substantially exceeds ELR-collapse error. (a,b) Training-loss trajectories from repeated runs difering only in initialization seed (a) or data-order seed (b). (c) The step-wise mean pairwise loss discrepancy over training. Legend values report the mean and standard deviation of the pairwise time-averaged discrepancies across unordered run pairs, in units of $1 0 ^ { - 2 }$ . The resulting time-averaged variations are $\Delta _ { \mathrm { s t o c h } } ( \mathrm { i n i t } ) = 1 . 1 1 \times 1 0 ^ { - 2 }$ and $\Delta _ { \mathrm { s t o c h } } ( \mathrm { d a t a - o r d e r } ) = 1 . 6 2 \times 1 0 ^ { - 2 }$ . By comparison, the three ELR-matched Llama-124M runs in Figures 1 and 2, with both seeds fixed, have collapse errors of $1 . 8 \times 1 0 ^ { - 3 } , 2 . 5 \times 1 0 ^ { - 3 }$ , and $2 . 6 \times 1 0 ^ { - 3 }$

## B Experimental Details

This appendix specifies the shared training configurations and the ELR matching protocols used throughout the paper. Appendix C then reports the corresponding trajectory-level results.

## B.1 Training Configurations

Tables B.1 and B.2 summarize the training configurations used. Below we provide further details.

Models. We consider several Transformer architectures in our pretraining experiments.

• Llama. Llama (Touvron et al., 2023) is a popular dense decoder-only Transformer architecture, incorporating Rotary Positional Encoding (RoPE), Swish-Gated Linear Units (SwiGLU), and root mean square layer normalization (RMSNorm). Our 124M and 1B parameter implementations are adapted from Modded-NanoGPT<sup>1</sup>. They use tied embeddings, QK-Norm, and a GPT-2 tokenizer with a padded vocabulary of 50,304 tokens.

• Qwen3. Qwen3 (Yang et al., 2025) is a family of decoder-only language models that includes both dense and mixture-of-experts (MoE) variants. Our Qwen3 Dense and Qwen3-MoE implementations are adapted from the oficial Qwen3 release. Compared with Llama, Qwen3 uses Grouped-Query Attention and QK-Norm. Our Qwen3-MoE model replaces the dense MLP with a sparse 32-expert MoE layer using top-4 routing.

• Kimi Delta Attention. Kimi Delta Attention (KDA) is a gated linear-attention mechanism introduced in Kimi Linear (Zhang et al., 2025). Our local implementation replaces three out of every four standard self-attention layers with KDA, giving a 3:1 KDA to standard attention ratio.

Datasets. Models are pretrained on the following datasets:

• FineWeb. FineWeb (Penedo et al., 2024) is a large-scale English web dataset derived from Common Crawl.

• Colossal Clean Crawled Corpus (C4) (Rafel et al., 2020). It is a large-scale public language dataset widely used for LLM pretraining, including T5 (Rafel et al., 2020).

• OpenWebText (Gokaslan and Cohen, 2019). It is an open-source recreation of the WebText corpus and is extensively used for LLM pretraining, including RoBERTa (Liu et al., 2019) and nanoGPT<sup>2</sup>.

Table B.1: Dense model configurations and peak learning rates used in the ELR-collapse experiments. Q/KV lists the numbers of query and key/value heads.
<table><tr><td>Model</td><td>size</td><td> $d _ { \mathrm { m o d e l } }$ </td><td> $d _ { \mathrm { F F } }$ </td><td> $\mathrm { Q } / \mathrm { K V }$ </td><td>depth</td><td>peak lr</td><td>final lr</td></tr><tr><td>Llama-124M</td><td>124M</td><td>768</td><td>3072</td><td>6</td><td>12</td><td> $1 . 8 \times 1 0 ^ { - 3 }$ </td><td>0</td></tr><tr><td>Llama-135M</td><td>135M</td><td>576</td><td>1536</td><td>9/3</td><td>30</td><td> $6 \times 1 0 ^ { - 4 }$ </td><td> $6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Llama-1B</td><td>1.07B</td><td>2560</td><td>10240</td><td>20</td><td>12</td><td> $3 \times 1 0 ^ { - 4 }$ </td><td>0</td></tr><tr><td>Qwen3-145M</td><td>145M</td><td>768</td><td>2304</td><td>12/6</td><td>12</td><td> $1 . 8 \times 1 0 ^ { - 3 }$ </td><td>0</td></tr><tr><td>Kimi Delta Attention-127M</td><td>127M</td><td>768</td><td>3072</td><td>6</td><td>12</td><td> $1 . 8 \times 1 0 ^ { - 3 }$ </td><td>0</td></tr></table>

Table B.2: Qwen3-MoE configuration and peak learning rate used in the ELRcollapse experiment. Q/KV lists the numbers of query and key/value heads; top-k gives the number of active experts per token.
<table><tr><td>Model</td><td>size</td><td> $d _ { \mathrm { m o d e l } }$ </td><td> $d _ { \mathrm { F F } }$ </td><td> $\mathrm { Q } / \mathrm { K V }$ </td><td>depth</td><td> $n _ { \mathrm { e x p e r t s } }$ </td><td>top-k</td><td>peak lr</td><td>final lr</td></tr><tr><td>Qwen3-MoE</td><td>586M</td><td>768</td><td>576</td><td>12/4</td><td>12</td><td>32</td><td>4</td><td> $1 . 8 \times 1 0 ^ { - 3 }$ </td><td>0</td></tr></table>

Training recipe. Unless stated otherwise, we use the following configuration throughout. All runs within each comparison share the same initialization and data-order seeds, eliminating the seed-induced variability discussed in Appendix A.2. We use AdamW with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 5$ , the MoonShot variant of Muon (Liu et al., 2025) with momentum coeficient $\beta = 0 . 9 5$ and Signum (Bernstein et al., 2018) with momentum coeficient $\beta = 0 . 9 5$ . Unless otherwise specified, weight decay is set to zero; nonzero weight decay is used only when explicitly indicated. The default LR schedule is warmup–stable–decay (WSD) (Hu et ${ \mathrm { a l . } }$ , 2024). Unless otherwise specified, all language model experiments use batch size $B = 1 2 8$ and sequence length 1024.

Loss smoothing. For all collapse and ELR-matching experiments, we apply an exponential moving average with $\alpha _ { \mathrm { E M A } } = 0 . 9 9$ to the raw loss at each training step. The smoothing is performed independently for each run before computing the loss residual $r _ { k }$ and mean collapse error $\Delta _ { \mathrm { c o l l } }$ . Both the plotted loss curves and the reported collapse errors are therefore based on the same smoothed losses. Unless otherwise stated, $\tau$ comprises all post-warmup training steps.

## B.2 ELR-Matching Protocols

The conversion from LR to ELR depends on how the optimizer parameterizes its update. We write $\mathbf { W } _ { k }$ for a matrix-valued parameter immediately before training step $k , \mathbf { U } _ { k }$ for the unscaled optimizer update, and $\eta _ { k }$ for the LR. When $\eta _ { k }$ directly multiplies $\mathbf { U } _ { k }$ , the ELR is

$$
\eta _ { k } ^ { \mathrm { e f f } } = \frac { \eta _ { k } } { \Vert \mathbf { W } _ { k } \Vert _ { F } } .
$$

However, Hyperball requires a diferent conversion because it normalizes the update:

$$
\mathbf { W } _ { k + 1 } = \mathtt { N o r m } _ { R } ( \mathbf { W } _ { k } - \eta _ { k } \mathtt { N o r m } _ { R } ( \mathbf { U } _ { k } ) ) , \qquad \mathtt { N o r m } _ { R } ( \mathbf { Q } ) : = R \frac { \mathbf { Q } } { \lVert \mathbf { Q } \rVert _ { F } } .
$$

The inner update can be written as

$$
\mathbf { W } _ { k } - \eta _ { k } \mathbb { N } \mathrm { o r m } _ { R } ( \mathbf { U } _ { k } ) = \mathbf { W } _ { k } - \frac { \eta _ { k } R } { \| \mathbf { U } _ { k } \| _ { F } } \mathbf { U } _ { k } .
$$

Because Hyperball maintains $\| \mathbf { W } _ { k } \| _ { F } = R .$ , its ELR is therefore

$$
\eta _ { k } ^ { \mathrm { e f f } } = \frac { \eta _ { k } R / \Vert \mathbf { U } _ { k } \Vert _ { F } } { R } = \frac { \eta _ { k } } { \Vert \mathbf { U } _ { k } \Vert _ { F } } .
$$

We use these optimizer-specific definitions in both matching protocols below.

Intervention timing. Let $k _ { 0 }$ denote the first training step after warmup. All matching controls and interventions are activated at $k _ { 0 }$ . Before $k _ { 0 }$ , the compared runs follow the same warmup configuration and enter the controlled phase from the same parameter state. All ELR schedules below refer to the post-warmup phase $k \geqslant k _ { 0 }$

We use two matching protocols. The first explicitly controls the LR and parameter norm schedule to realize a prescribed ELR schedule. The second adapts only the LR to reproduce the ELR induced by a reference run.

Matching a prescribed ELR schedule. Given a target ELR schedule $\{ \eta _ { k } ^ { \mathrm { e f f } } \} _ { k \geqslant k _ { 0 } }$ , we construct multiple runs with diferent LR and norm schedules. For an optimizer whose LR directly multiplies its update, we choose an LR schedule $\{ \eta _ { k } \} _ { k \geqslant k _ { 0 } }$ and define the corresponding norm schedule by

$$
\rho _ { k } : = \frac { \eta _ { k } } { \eta _ { k } ^ { \mathrm { e f f } } } .
$$

We scale the LR schedule so that $\rho _ { k _ { 0 } } = \| \mathbf { W } _ { k _ { 0 } } \| _ { F }$ , ensuring continuity at the start of the controlled phase. At step k, we first apply the underlying optimizer update with LR $\eta _ { k }$ , obtaining a provisional parameter $\widetilde { \mathbf { W } } _ { k + 1 }$ , and then project it onto the next target norm:

$$
\mathbf { W } _ { k + 1 }  \rho _ { k + 1 } \frac { \widetilde { \mathbf { W } } _ { k + 1 } } { \vert \vert \widetilde { \mathbf { W } } _ { k + 1 } \vert \vert _ { F } } .
$$

Consequently, $\| \mathbf { W } _ { k } \| _ { F } = \rho _ { k }$ and $\begin{array} { r } { \frac { \eta _ { k } } { \| \mathbf { W } _ { k } \| _ { F } } = \frac { \eta _ { k } } { \rho _ { k } } = \eta _ { k } ^ { \mathrm { e f f } } } \end{array}$ throughout the controlled phase. Repeating this construction with diferent LR schedules produces distinct LR and norm trajectories that realize the same prescribed ELR schedule.

For a Hyperball run, its prescribed radius supplies the norm control. After computing $\mathbf { U } _ { k }$ we instead set

$$
\eta _ { k } = \eta _ { k } ^ { \mathrm { e f f } } \Vert \mathbf { U } _ { k } \Vert _ { F } ,
$$

which realizes the same prescribed ELR under the Hyperball definition above.

Reference-run ELR matching. This protocol does not prescribe the norm trajectory of the matched run. Its optimizer and norm-control mechanism are left unchanged, and only its LR is adapted. This allows us to test whether a norm-control method afects loss dynamics through the ELR it induces.

Let run A be the reference run and run B the matched run. We first train run A and record its ELR using the appropriate definition:

$$
\eta _ { k } ^ { \mathrm { e f f } , A } = \left\{ \eta _ { k } ^ { A } / \| \mathbf { W } _ { k } ^ { A } \| _ { F } , \quad \mathrm { i f ~ r u n ~ A ~ u s e s ~ a ~ d i r e c t - u p d a t e ~ o p t i m i z e r } , \right.
$$

When training run B, we set

$$
\eta _ { k } ^ { B } = \left\{ \begin{array} { l l } { \eta _ { k } ^ { \mathrm { e f f } , A } \lVert \mathbf { W } _ { k } ^ { B } \rVert _ { F } , } & { \mathrm { i f ~ r u n ~ B ~ u s e s ~ a ~ d i r e c t - u p d a t e ~ o p t i m i z e r } , } \\ { \eta _ { k } ^ { \mathrm { e f f } , A } \lVert \mathbf { U } _ { k } ^ { B } \rVert _ { F } , } & { \mathrm { i f ~ r u n ~ B ~ u s e s ~ H y p e r b a l l } . } \end{array} \right.
$$

Thus, the matched run tracks the reference ELR while retaining its own norm evolution and norm-control mechanism.

## C Quantitative Validation (Supplement to Section 4)

Appendix B specifies the shared training configurations and ELR-matching protocols used throughout the experiments. For a matched run with smoothed loss $L _ { k }$ and a reference run with smoothed loss $L _ { k } ^ { \mathrm { r e f } }$ , we quantify collapse precision using the collapse residual and mean collapse error introduced in Section 3:

$$
r _ { k } : = L _ { k } - L _ { k } ^ { \mathrm { r e f } } , \quad \quad \Delta _ { \mathrm { c o l l } } : = \frac { 1 } { | \mathcal { T } | } \sum _ { k \in \mathcal { T } } | r _ { k } | ,
$$

where $\tau$ denotes the set of training steps over which collapse is evaluated. We use these quantities throughout this appendix to report both the trajectory-level deviations and their average magnitude.

Table C.1 summarizes the mean collapse errors and principal configurations for all prescribed-ELR validation experiments, including those presented in Section 4.1. Unless otherwise noted, each setting compares three LR schedules—linear-up, linear-down, and sinusoidal—with the target-norm schedule adjusted so that all runs realize the same prescribed ELR schedule.

Table C.1: Collapse errors across prescribed-ELR validation settings. Each row compares runs with diferent LR and parameter-norm schedules that realize the same prescribed ELR schedule. Entries report the mean collapse error $\Delta _ { \mathrm { c o l l } } .$ , in units of $1 0 ^ { - 3 } .$ , for linear-up, linear-down, and sinusoidal LR schedules. An em dash denotes a schedule that was not evaluated.
<table><tr><td rowspan="2" colspan="2">No. Validation setting Model</td><td rowspan="2">Dataset</td><td colspan="3"> $\Delta _ { \mathrm { c o l l } } ~ ( \times 1 0 ^ { - 3 } )$ </td><td rowspan="2">Figure</td></tr><tr><td></td><td>Up Down Sine</td><td></td></tr><tr><td colspan="8">Experiments reported in the main text</td></tr><tr><td>1 Dense Llama</td><td>Llama-124M</td><td>FineWeb</td><td>1.8</td><td>2.5</td><td></td><td>2.6 Fig. 1</td><td></td></tr><tr><td>2 MoE</td><td>Qwen3-MoE 586M FineWeb</td><td></td><td>3.0</td><td>4.1</td><td>3.3</td><td>Fig. 1</td><td></td></tr><tr><td colspan="8">Additional experiments reported in the appendix</td></tr><tr><td>3 1B scale</td><td>Llama-1B</td><td>FineWeb</td><td>2.0</td><td>2.0</td><td></td><td>Fig. C.1</td><td></td></tr><tr><td>4 Qwen3 Dense</td><td>Qwen3-145M</td><td>FineWeb</td><td>1.2</td><td>1.9</td><td>2.7</td><td></td><td>Fig. C.2(a)</td></tr><tr><td>5 Linear attention</td><td>KDA-127M</td><td>FineWeb</td><td>0.9</td><td>1.7</td><td></td><td>3.2</td><td>Fig. C.2(b)</td></tr><tr><td>6 C4</td><td></td><td>Qwen3-145M C4</td><td></td><td>2.1</td><td>2.1</td><td>2.5</td><td>Fig. C.2(c)</td></tr><tr><td>7</td><td>OpenWebText</td><td>Qwen3-145M</td><td>OpenWebText 3.5</td><td></td><td>3.4</td><td>3.5</td><td>Fig. C.2(d)</td></tr><tr><td>8 Signum</td><td></td><td>Llama-124M</td><td>FineWeb</td><td>4.8 1.6</td><td></td><td></td><td>4.9 Fig. C.2(e)</td></tr><tr><td>9 Muon</td><td></td><td>Llama-124M</td><td>FineWeb</td><td>1.3 1.8</td><td></td><td></td><td>2.5 Fig. C.2(f)</td></tr></table>

1B-scale stress test. As a scale stress test, we repeat the prescribed-ELR experiment with a Llama-1B model trained on FineWeb. Figure C.1 reports the full LR, ELR, loss, and collapse residual trajectories for the linear-up and linear-down realizations. Each yields a mean collapse error of $2 . 0 \times 1 0 ^ { - 3 }$ , showing that high-precision ELR collapse persists at the 1B scale.

Broader validation. Figure C.2 reports trajectory-level diagnostics for the remaining experiments. These extend the validation to dense Qwen3 and Kimi Delta Attention, to C4 and OpenWebText, and to the Signum and Muon optimizers. Each panel follows the LR–ELR–loss layout of Figure 1.

![](images/39703ec2e42550266a8c96fc4334a96c74386564c1cc6a9839ccd865d68d897d.jpg)  
Figure C.1: ELR collapse in Llama-1B on FineWeb. The compact panels show LR and ELR. The loss panel compares the two varied schedules with the constant reference and reports their mean absolute collapse errors.

![](images/bbeb662edb4776f6f1ca588762a244a5ab0e7bebc5c701b29a14b3d728c6a8e0.jpg)  
(a) Qwen3-Dense-145M / FineWeb

![](images/36178169920d3768bd9e03b7f85e6a87a6ab63b9abb3b6a760ecd19b50ec4444.jpg)

![](images/43afd50681a0a1a8043eb5de3512e9cf58aa39933e1450f31579095c89331187.jpg)  
(c) Qwen3-145M / C4

![](images/7a1f389449d74bace6d10e94792c53d2bbac27d7616e3e88cf02c905040c4406.jpg)  
(b) Kimi Delta Attention / FineWeb

![](images/9e9f362a9a06a4c4d4964620c93e8889a33b3323890852c5d127528731ed6390.jpg)  
(d) Qwen3-145M / OpenWebText

![](images/83307db64916de9ae762967defe77a7bea96b458ffa80e1922bb6d54500eaddd.jpg)

![](images/3f7e3c3fae5bd26d3d66f7c97acedf69b70eec54e1b9fb0d4ea57f42169b5cda.jpg)  
(e) Signum / Llama-124M / FineWeb

![](images/2e356192db2e57bca0aa69d133b1b9817ae3d3fb9329c81ba1ce95eebd11d19e.jpg)  
(f) Muon / Llama-124M / FineWeb

![](images/8ec098e46689386d637805a10c5200f7ce718ba96b7fb036c2598048b237c771.jpg)

![](images/c414260151db87dc5d451b48f27ae64c92e2b12e973660ebabb963b2ea907c67.jpg)

![](images/5cebdc4cba5e2b0bcaa8d5e7874e00d6e623ee6059bed53b0a660725eb20a23a.jpg)

![](images/5de6b72d258d0351f2b6f85befa434f19855b8f41cad08a0c43dd06642994ccb.jpg)  
Figure C.2: Additional validation of ELR collapse across training configurations. (a) Replicates the prescribed-ELR experiment with Qwen3-145M on FineWeb. (b) Applies the same protocol to Kimi Delta Attention (KDA). (c,d) Repeat the Qwen3-145M experiment on C4 and OpenWebText, respectively. (e,f) Evaluate Signum and Muon with Llama-124M, respectively. Together, these experiments test ELR collapse across model architectures, pretraining datasets, and optimizers. In each subfigure, the loss-panel legend reports the mean absolute collapse error relative to the constant-schedule reference run.

## D Weight Decay and Hyperball (Supplement to Section 5)

This section reports the LR and ELR trajectories underlying the intervention experiments in Section 5 and repeats each intervention with the matching direction reversed. All adapted LRs follow the reference-run ELR-matching protocol in Appendix B.2.

Weight decay with AdamW. Figure D.1(a) supplements Figure 5(a) with the corresponding LR and ELR trajectories. The run with λ = 0.1 serves as the target, while the LR of the run without weight decay is adapted to track its ELR schedule. Because the norm of the λ = 0 run grows more rapidly, its adapted LR increases through most of training and reaches approximately

![](images/33f04452edb6d495eb317a9415a4f22c56f5c42a56fd11af8e473a72888297a6.jpg)  
Figure D.2: LR-only ELR matching aligns MuonW and MuonH loss dynamics. Experiments use Llama-124M trained on FineWeb. MuonW denotes Muon with weight decay, whereas MuonH uses Hyperball. In each subfigure, only the LR of the comparison run is adapted to match the target ELR, while its norm-control method remains unchanged. The compact panels show the LR and ELR trajectories, and the large panel compares the resulting losses. (a) MuonH serves as the target, and the LR of MuonW is adapted. (b) The matching direction is reversed: MuonW serves as the target, and the LR of MuonH is adapted. For MuonH, we use the optimizer-specific ELR defined in Appendix B.

$4 \times 1 0 ^ { - 2 }$ before the terminal decay. Figure D.1(b) reverses the direction: the $\lambda = 0$ run serves as the target, and the LR of the $\lambda = 0 . 1$ run is adapted accordingly. The loss trajectories remain closely aligned in both directions.

![](images/a88f34218215a0deff2d717380f63b7855bec17e389c956d6f9db7dd7f30bdce.jpg)

![](images/cdc60b8869c8a74653976d4e6913bd87e01436ad64df357a09304f21ffc4a5c6.jpg)

![](images/f6fabcdf76893e2ac447c1a93a93add3d18592d867e9b0ba34ac3fe3dc935944.jpg)  
Figure D.1: LR-only ELR matching recovers loss dynamics across AdamW weight-decay settings. Experiments use Llama-124M trained on FineWeb with AdamW. In each subfigure, only the LR of the comparison run is adapted to match the target ELR, while its parameter norm evolves freely. The compact panels show the LR and ELR trajectories; the large panel compares the losses and reports the mean collapse error. (a) The run with weight decay $\lambda = 0 . 1$ serves as the target, and the LR of the $\lambda = 0$ run is adapted. (b) The matching direction is reversed: the $\lambda = 0$ run serves as the target, and the LR of the $\lambda = 0 . 1$ run is adapted. The loss trajectories remain closely aligned in both directions.

Hyperball with Muon. Figure $\mathrm { { D . 2 ( a ) } }$ supplements Figure 5(b) with the corresponding LR and ELR trajectories. MuonH, which uses Hyperball, serves as the target, while the LR of MuonW, which uses weight decay, is adapted to track its ELR schedule. Figure D.2(b) reverses the direction: MuonW serves as the target, and the LR of MuonH is adapted to track its ELR. The resulting loss trajectories remain closely aligned, with a mean collapse error of $2 . 5 \times 1 0 ^ { - 3 }$ Thus, the agreement is not specific to the direction of ELR matching.

Learnable RMSNorm gains remain important under LR-only intervention. Figure D.3 repeats the AdamW LR-only intervention in Figure D.1, with the RMSNorm gains fixed throughout training as the only change. As in the prescribed-ELR experiments, fixing these gains substantially weakens the collapse and increases the collapse error to the 10<sup>−2</sup> scale. Thus, the dependence on learnable gains is not specific to prescribed norm schedules; it persists when the parameter norm evolves freely and only the LR is adapted to match ELR.

![](images/fd21cf2e06bb4e1fd5ccdd1d2334581d5c21186902267091c303658b1fd373a8.jpg)

![](images/780c093fd480931b52c1821b390b08349d034028420955c6322064d6c75e4ac2.jpg)  
Figure D.3: Fixing RMSNorm gains degrades LR-only ELR matching. Dashed black curves show the target losses, red curves show the comparison runs whose LRs are adapted to match the target ELR schedules, and blue curves show the collapse residuals $r _ { k }$ on the right axis. (a) The run with $\lambda = 0 . 1$ serves as the target, and the LR of the $\lambda = 0$ run is adapted. (b) The matching direction is reversed. In both panels, all RMSNorm gains are fixed and no norm trajectory is prescribed. Compared with the learnable-gain results in Figure D.1, fixing the gains produces sustained $1 0 ^ { - 2 } \cdotp$ -scale residuals in both directions.

## E Functional Scaling Laws (Supplement to Section 6)

We provide the FSL specification and per-trajectory results complementing Figure 6 in Section 6.

Trajectories and evaluation split. We train Llama-124M on FineWeb for 42,000 optimizer steps, including 1,000 warmup steps. Standard training uses wd $\in \{ 0 , 0 . 1 \}$ and a peak learning rate of $6 \times 1 0 ^ { - 4 }$ , whereas Hyperball directly controls the update norm and uses a peak learning rate of $2 \times 1 0 ^ { - 3 }$ . We consider four LR schedules. WSD-10% and WSD-30% decay LR over the final 10% and 30% of training, respectively. The 7111 schedule divides the post-warmup steps into four consecutive phases containing 70%, 10%, 10%, and 10% of the steps, with LR reduced by a factor of $1 0 ^ { 1 / 3 }$ at each phase boundary. The complete assignment of trajectories to the fitting, in-distribution (ID), and out-of-distribution (OOD) splits is reported in Table E.1. Figure E.1 shows that ELR places the diferent norm-control regimes on a comparable scale.

FSL parameterization and fitting. Given a rate schedule $\{ r _ { j } \} _ { j \geqslant 0 }$ , we define the corresponding intrinsic time as $\begin{array} { r } { t _ { k } = \sum _ { j = 0 } ^ { k } r _ { j } } \end{array}$ . FSL models the loss trajectory as

$$
\widehat { L } _ { k } = L _ { \infty } + C _ { 1 } ( B _ { 1 } + t _ { k } ) ^ { - a _ { 1 } } + C _ { 2 } \sum _ { j = 0 } ^ { k } ( B _ { 2 } + t _ { k } - t _ { j } ) ^ { - a _ { 2 } } r _ { j } ^ { 2 } .\tag{E.1}
$$

The lr-FSL and elr-FSL models share this parameterization and difer only in the rate schedule used as input. Let $\eta _ { k }$ denote the learning rate and $n _ { k }$ the measured norm scale at step k. The two input schedules are

$$
r _ { k } ^ { \mathrm { l r } } = \eta _ { k } , \qquad r _ { k } ^ { \mathrm { e l r } } = \frac { \eta _ { k } } { n _ { k } } .\tag{E.2}
$$

For standard training trajectories, $n _ { k }$ is the Frobenius norm of the parameters; for Hyperball trajectories, it is the mean update-norm scale.

![](images/0b283471793e39235a0284bfd3fe31797e019c6de7480eecf35e1e0bd2e79457.jpg)  
Figure E.1: LR schedule, ELR-denominator evolution, and ELR schedule under distinct norm-control regimes. All curves use WSD-10%. Standard training uses weight decay with $\lambda \in \{ 0 , 0 . 1 \}$ . For standard runs, the ELR denominator is the parameter Frobenius norm; for Hyperball, it is the average update norm over the post-warmup steps.

Following the FSL specification in the main text, we jointly fit the seven parameters

$$
\Theta = ( L _ { \infty } , C _ { 1 } , B _ { 1 } , a _ { 1 } , C _ { 2 } , B _ { 2 } , a _ { 2 } )
$$

across the four fitting trajectories. For both variants, we smooth the loss trajectories using an exponential moving average with decay 0.99, exclude the first 2,000 steps from fitting, and use the same parameter initialization. We optimize Θ for 800 Adam steps, followed by 10 L-BFGS refinement steps.

Prediction Results. Table E.1 reports every trajectory, and Figure E.2 shows the corresponding predictions. elr-FSL has lower RMSE on nine of ten trajectories; under Hyperball, the improvements reach 5.61× for cosine and 19.96× for WSD-10%.  
Table E.1: Per-trajectory prediction error. Improvement = $\mathrm { R M S E } _ { l r } / \mathrm { R M S E } _ { e l r } ;$ values above one favor elr-FSL. Bold marks the lower error.
<table><tr><td>Split</td><td>Trajectory</td><td>lr-FSL ↓</td><td>elr-FSL ↓</td><td>Improvement ↑</td></tr><tr><td rowspan="4">Fit</td><td>wd = 0, WSD-30%</td><td>0.0251</td><td>0.0150</td><td>1.68×</td></tr><tr><td>wd = 0, cosine</td><td>0.0069</td><td>0.0138</td><td>0.50×</td></tr><tr><td>wd = 0.1, cosine</td><td>0.0180</td><td>0.0122</td><td>1.48×</td></tr><tr><td>wd = 0.1, WSD-10%</td><td>0.0184</td><td>0.0112</td><td>1.65×</td></tr><tr><td rowspan="4">ID</td><td>wd = 0, WSD-10%</td><td>0.0327</td><td>0.0160</td><td>2.05×</td></tr><tr><td>wd = 0, 7111</td><td>0.0233</td><td>0.0143</td><td>1.63×</td></tr><tr><td>wd = 0.1, WSD-30%</td><td>0.0172</td><td>0.0110</td><td>1.56×</td></tr><tr><td>wd = 0.1, 7111</td><td>0.0193</td><td>0.0113</td><td>1.70×</td></tr><tr><td rowspan="2">OOD</td><td>Hyperball, cosine</td><td>0.1353</td><td>0.0241</td><td>5.61×</td></tr><tr><td>Hyperball, WSD-10%</td><td>0.3662</td><td>0.0183</td><td>19.96×</td></tr></table>

![](images/e97f7e8fc736b14ecd39f466323c33c12679081b5a706d61c149a26736476d82.jpg)  
Figure E.2: Predictions for all fitted and held-out trajectories. Rows show fit, ID schedule transfer, and OOD Hyperball trajectories. Black denotes ground truth; blue and red denote lr-FSL and elr-FSL. We see that elr-FSL is substantially more accurate on both held-out ID and OOD schedules.

## F Delayed Acceleration (Supplement to Section 7)

## F.1 Acquired Early, Revealed Late: A Controlled WSD Test

The cosine-schedule experiments in the main text establish delayed acceleration. However, under cosine decay, the learning rate decreases continuously, making it dificult to vary the duration of the late low-ELR phase independently. We therefore conduct a complementary experiment using WSD, whose terminal decay window can be controlled explicitly. Holding the peak LR and training budget fixed, we vary the WSD decay ratio from 0.1 to 0.3. This provides a targeted test of the temporal mechanism predicted by FSL.

FSL predicts that the benefit of weight decay is acquired early but revealed late. By suppressing norm growth, weight decay sustains a larger ELR during the early phase. The resulting faster accumulation of efective training time improves the signal-learning term, but the larger ELR simultaneously produces greater noise accumulation. Consequently, the signal-learning advantage need not be visible immediately in the observed loss. As ELR decays, new noise injection falls while previously accumulated noise continues to be forgotten, revealing the signal-learning advantage acquired earlier. The decisive prediction is therefore that a short decay phase may end before this advantage becomes visible, whereas a longer decay phase should reveal it and allow the weight-decayed run to overtake the unregularized baseline earlier.

Figure F.1 confirms this prediction. With a WSD decay ratio of 0.1, the λ = 0.1 run maintains a larger ELR than the λ = 0 run, but the terminal decay phase is too short to reveal its signal-learning advantage: it remains above the unregularized baseline at the end of training. When the decay ratio is increased to 0.3, the longer low-ELR phase changes the outcome. The $\lambda = 0 . 1$ run overtakes $\lambda = 0$ during decay and attains a lower final loss. In other words, the gain is accumulated before the crossover but becomes observable only during a suficiently long decay phase.

The predicted crossover under a longer decay window supports the FSL account: an early signal-learning gain is initially masked by noise and revealed only late.

![](images/7aaae8864841cd47fa296f3894b545438de0ee3a86c1287f6aa50453789f3206.jpg)  
(a) WSD decay ratio 0.1

![](images/06545f6d40b1399d620a60a94d1786dcfb779336d0b6cddc3a9698d96f5ef2c8.jpg)  
(b) WSD decay ratio 0.3  
Figure F.1: A longer WSD decay phase reveals the signal-learning gain acquired earlier. Using Llama-135M trained on FineWeb, we explicitly vary the duration of the terminal WSD decay phase. Each panel compares $\lambda \in \{ 0 , 0 . 1 \}$ at a fixed WSD decay ratio; black and red denote $\lambda = 0$ and 0.1, respectively. With a decay ratio of 0.1 (left), the decay phase is too short for the $\lambda = 0 . 1$ run to overtake the unregularized baseline. Increasing the decay ratio to 0.3 (right) extends the low-ELR phase, allowing the accumulated noise to be forgotten and revealing the earlier signal-learning advantage: the $\lambda = 0 . 1$ run overtakes $\lambda = 0$ and attains a lower final loss.

## F.2 Delayed Acceleration under Hyperball

![](images/985209f998cdeba7891be3893a6d9732b73729c50fae8a6c510d248705eae5d4.jpg)

![](images/7d27f51725cc5ff27daf38e0ca132dfdea12f617fc433498218fc9f4ab125b50.jpg)  
Figure F.2: Delayed acceleration under Hyperball. On Llama-124M trained on FineWeb, MuonH initially has higher loss than MuonW, but later overtakes it and reaches a lower late-stage loss. Both runs use linear schedules.

The same qualitative phenomenon also appears under Hyperball. A delayed gain is already visible in the MuonH trajectory reported by Wen et al. (2026, Figure 3, right). In our experiment, Figure F.2 compares MuonH with MuonW when both use linear decay. MuonH has higher loss in early training, but subsequently catches up with and overtakes MuonW, reaching a lower loss near the end of training. Delayed acceleration is therefore not specific to weight decay; it also arises when the parameter norm is controlled explicitly by Hyperball.

The same ELR mechanism explains this reversal: Hyperball’s norm control slows ELR decay relative to MuonW, maintaining a larger ELR before the terminal decay. This accelerates signal learning but also increases noise accumulation, initially masking the advantage. During the terminal decay, the ELR decreases and the accumulated noise is gradually forgotten, revealing the advantage acquired earlier. This is consistent with Figure 5(b), where matching the MuonH normalized update coeficient reproduces its loss trajectory with a mean collapse error of 1 $. 2 \times 1 0 ^ { - 3 }$

## G ELR Collapse in Vision Transformers

Setup and ELR matching. To test whether ELR collapse extends beyond language modeling, we repeat the norm-control matching experiment of Appendix B.2 on a 9.8M-parameter pre-norm Vision Transformer (ViT) (Dosovitskiy et al., 2021) trained on ImageNet-1K (Deng et al., 2009) with $6 4 \times 6 4$ inputs. The model has 12 layers, hidden size 256, four attention heads, an MLP expansion ratio of 4, and $8 \times 8$ patches; it uses RMSNorm, QK-Norm, and ReLU activations. We train in FP32 for 50,000 steps using AdamW with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ , no weight decay, global batch size 512, and a WSD schedule that warms up to $6 \times 1 0 ^ { - 4 }$ over 1,250 steps, remains constant until step 45,000, and decays to $6 \times 1 0 ^ { - 5 }$

We use the prescribed-ELR matching construction in Appendix B.2. For the ViT experiments, the controlled parameter collections comprise the patch-projection afine parameters, the classtoken and positional embeddings, and the afine parameters in all attention and MLP blocks. The RMSNorm gains and classifier head retain their base learning rates and are excluded from the prescribed-norm projection.

Result. We compute the mean collapse error as in the main text using the smoothed training loss. Figure G.1 shows that the three nonconstant branches remain aligned with the constant branch, with $\Delta _ { \mathrm { c o l l } } \in [ 3 . 1 6 , 3 . 1 9 ] \times 1 0 ^ { - 3 }$ . This extends the 10<sup>−3</sup>-scale ELR collapse observed in language model pretraining to this ViT setting.

![](images/5a331ee98ce317472cd6afb7e610512757623d9ec36d1b88f75dd98336d090bf.jpg)  
Figure G.1: ELR collapse for a 9.8M-parameter ViT trained on ImageNet-1K. At step 6,250, training is branched into four runs with distinct LR and parameter-norm schedules but a matched ELR schedule. Their smoothed loss trajectories remain collapsed throughout the post-branch phase. The legend reports the collapse error $\Delta _ { \mathrm { c o l l } }$ of each run relative to the constant-LR reference.

Normalization also afects collapse precision in ViTs. We test whether the normalization dependence identified in Section 4.2 also appears in ViTs. Starting from the preceding configuration, we consider three nested variants: (i) QK-Norm with learnable gains, (ii) no QK-Norm with learnable gains, and (iii) no QK-Norm with all remaining RMSNorm gains fixed at initialization. For each variant, we repeat the prescribed-ELR protocol using the reference, linear-up, linear-down, and sinusoidal schedules.

Figure G.2 shows the same qualitative ordering observed in language models. Averaged over the three nonreference schedules, the mean collapse error increases from $3 . 1 8 \times 1 0 ^ { - 3 }$ in the default configuration to $3 . 9 0 \times 1 0 ^ { - 3 }$ after removing QK-Norm, and further to $6 . 4 6 \times 1 0 ^ { - 3 }$ after fixing the RMSNorm gains. This result shows that the normalization dependence of ELR collapse is not confined to our language model experiments, but also appears in the ViT/ImageNet setting studied here.

![](images/a9252b26ffb5f511a31b1362cf5a7c7c4c2ca4572c6eed6e206cd3f9597f4bf2.jpg)

![](images/a2ea614791964d25c83132e79f24448429a11df3eaedf8f060e0d0b1e348c4c8.jpg)  
Figure G.2: Normalization improves the precision of ELR collapse in ViTs. Left: Mean absolute loss residual across the linear-up, linear-down, and sine-wave ELR-matched schedules relative to the constant-LR reference during training. Right: Corresponding trajectory-averaged collapse error. Removing QK-Norm weakens the collapse, while additionally fixing the remaining gains produces the largest discrepancy.