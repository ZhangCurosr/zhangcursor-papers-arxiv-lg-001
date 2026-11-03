# CONTROLLING SHARPENING IN LLM POST-TRAINING

Simon Matrenok<sup>1</sup>, Nicolas Menet<sup>2</sup>, Nicolas Flammarion<sup>1</sup>, Andreas Krause<sup>2</sup>

<sup>1</sup>School of Computer and Communication Sciences (IC), EPFL

<sup>2</sup>Department of Computer Science, ETH Zurich¨

## ABSTRACT

LLM post-training via reinforcement learning has shown consistent improvements in accuracy at the cost of reducing behavioral coverage—a phenomenon known as sharpening that can stall progress on difficult tasks. Repeated-sampling objectives such as pass@k and max@k are known to mitigate sharpening, but the mechanism behind it remains poorly understood. Moreover, existing approaches for @k optimization use dedicated algorithms, and no general framework exists for adapting established policy optimization methods from @1 to @k. We derive the optimal policies for KL-regularized pass@k and max@k objectives in closed form. Our explicit solutions reveal that @k optimization naturally redistributes learning pressure based on prompt difficulty, directing it toward harder problems. Crucially, we show that standard @1 objectives achieve these exact same optimal policies through prompt-adaptive regularization or reward transformation. Ultimately, this yields an optimizer-independent explanation for why @k objectives reduce sharpening on easy tasks while prioritizing success on harder ones. Building on these insights, we introduce @k Adaptive Pressure for Policy Optimization (KAPPO). Following a one-time reference calibration, KAPPO seamlessly adapts existing @1 methods to @k via prompt-adaptive regularization or reward transformation, without otherwise altering their training procedures. Experiments with several RL methods on mathematical reasoning and code generation demonstrate that KAPPO provides direct control over sharpening. Crucially, we demonstrate that @k pressure can be fully realized with single-completion training, matching the performance of multi-completion @k optimizers at a fraction of the cost.

## 1 INTRODUCTION

Reinforcement learning (RL) has become a central component of LLM post-training, producing sub stantial performance improvements in mathematical reasoning, code generation, and other verifiable domains (Shao et al., 2024; Chen et al., 2025a; Yu et al., 2025). Yet, an increase in average performance does not by itself reveal how the model has improved. Pretrained LLMs contain extensive latent capabilities, and RL can affect them through two distinct mechanisms. It may uncover successful behavior that is rarely expressed by the base policy, or it may concentrate probability on solutions that are already readily available. The latter phenomenon, referred to as sharpening, improves single-trial success but often reduces the range of problems solvable by the model under repeated tri als (Yue et al., 2025). Further capability progress thus depends on effectively controlling sharpening.

Recent studies confirm this exact limitation: RL post-training dynamics are indeed dominated by sharpening (Yue et al., 2025; Song et al., 2025). The impressive single-trial gains frequently diminish or reverse in multi-trial settings, as the generated reasoning paths are confined to a narrow subset of the base model’s original distribution. Counteracting this over-concentration is notoriously difficult. For example, simply injecting randomness or diversity at the token level rarely translates into meaningful diversity in high-level reasoning strategies (Walder & Karkhanis, 2025; Chen et al., 2025b).

Multi-trial metrics provide a practical lens for quantifying the extent of sharpening (Yue et al., 2025). For binary rewards, pass@k measures the probability that at least one of k completions succeeds; for continuous rewards, max@k measures the expected maximum reward among k completions. Gains driven by sharpening concentrate at small k and diminish as the sampling budget grows, whereas gains that persist at large k indicate broader coverage of successful behavior. While these metrics do not prove that the model has discovered reasoning strategies outside its base distribution, they remain the clearest operational proxies for diverse reasoning. These observations motivate using @k metrics not just for evaluation, but also as training objectives to mitigate sharpening (Walder & Karkhanis, 2025). However, the mechanism that makes @k objectives less prone to sharpening remains poorly understood. Moreover, existing approaches rely on dedicated algorithms for @k optimization, precluding a direct transfer of standard @1 reinforcement learning advancements to the @k setting.

We derive the closed-form optimal policies for KL-regularized pass@k and max@k. Our analysis reveals that @k optimization acts as prompt-adaptive pressure: it restricts optimization on tasks the model already solves reliably and redirects pressure toward tasks on which successful behavior is rare. We show that this behavior is achievable with ordinary @1 objectives via prompt-adaptive regularization or reward transformation. Leveraging this connection, we introduce @k Adaptive Pressure for Policy Optimization (KAPPO), a framework for turning existing @1 policy-optimization methods into @k optimizers. After a one-time reference calibration, KAPPO introduces prompt-adaptive pressure while preserving all other characteristics of the base method. Our contributions are:

1. We derive closed-form optimal policies for KL-regularized pass@k and max@k objectives.

2. We prove that @k optimization redistributes optimization pressure according to task difficulty via prompt-adaptive regularization or reward transformation, prioritizing hard tasks.

3. We introduce KAPPO, a framework that turns existing @1 methods into @k optimizers. In KAPPO, k directly controls sharpening without requiring online groups of k completions.

4. We instantiate KAPPO across several policy optimization methods, matching the performance of a direct multi-completion @k-optimization baseline at a fraction of its rollouts.

## 2 RELATED WORK

Exploration mechanisms in RLVR. Beyond the sharpening evidence discussed above, RL has been shown to consistently reduce generation diversity (Kirk et al., 2024), including on problems the model has not yet solved (Song et al., 2025). Application of standard exploration strategies, such as entropy regularization (Mnih et al., 2016; Haarnoja et al., 2018; Ahmed et al., 2019) and noisy reward perturbations (Lu & Van Roy, 2017), has shown limited or sensitive improvements relative to methods designed specifically for multi-attempt reasoning (Walder & Karkhanis, 2025; Chen et al., 2025b). More structured approaches are guided by epistemic uncertainty via policy ensembles (Menet et al., 2026a), or introduce exploration bonuses based on final outcomes (Song et al., 2025), hidden representations (Tuyls et al., 2026; Menet et al., 2026b), or sequence-level diversity (Hu et al., 2026). These methods rely on different proxies for novelty and have been evaluated in heterogeneous settings; no established exploration mechanism for LLM post-training has yet emerged.

Pass@k and max@k optimization. Originally introduced as an evaluation metric for code generation (Chen et al., 2021), pass@k has recently been adopted as a training objective. Walder & Karkhanis (2025) derive unbiased policy-gradient estimators, called PKPO, for binary pass@k and continuous max@k rewards. They show that increasing k during training preserves higher entropy, increases the number of problems solved at least once, and unblocks learning on difficult tasks where pass@1 optimization stalls. max@k objectives are submodular (Krause & Golovin, 2014), and encourage “hedging the bets” given limited expressivity, a principle associated with exploration. Subsequent work studies analytical advantage design (Chen et al., 2025b), token-level mechanisms for preventing probability over-concentration (Peng et al., 2025), improved policy-gradient estimators for order-statistic objectives (Parmas et al., 2026), and a pass@k interpretation of maximum likelihood objectives (Tajwar et al., 2026). These approaches primarily analyze training gradients or construct multi-completion learning signals. We instead derive the corresponding KL-regularized optimal policies and show that the effect of @k optimization can be reproduced through promptadaptive @1 objectives. This provides a general route for turning existing @1 methods into @k variants without constructing expensive @k-dependent multi-completion learning signals.

## 3 PRELIMINARIES

KL-regularized reward optimization. Let $x \sim \mathcal { D }$ denote a prompt, $y \sim \pi ( \cdot \mid x )$ a completion, and $R ( y , x )$ the corresponding reward. Standard KL-regularized RL optimizes the objective

$$
\begin{array} { r } { \mathcal { I } ( \pi ) : = \mathbb { E } _ { x \sim \mathcal { D } } \left[ \mathbb { E } _ { y \sim \pi ( \cdot | x ) } [ R ( y , x ) ] - \beta \operatorname { K L } \left( \pi ( \cdot | x ) \| \pi _ { \mathrm { r e f } } ( \cdot | x ) \right) \right] , } \end{array}\tag{1}
$$

where $\pi _ { \mathrm { r e f } }$ is a fixed reference policy and $\beta > 0$ controls the strength of regularization. While some recent RLVR methods omit the KL penalty to allow larger policy shifts (Yu et al., 2025), other works suggest that carefully designed KL-regularized objectives enhance both reasoning performance and training stability (Liu et al., 2025). Beyond potential empirical advantages, KL regularization is a prerequisite for non-degenerate optimal policies. Indeed, the optimal policy is given in closed form:

$$
\begin{array} { r } { \pi ^ { * } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp ( R ( y , x ) / \beta ) . } \end{array}\tag{2}
$$

Methods such as REINFORCE (Williams, 1992), RLOO (Ahmadian et al., 2024), PPO (Schulman et al., 2017), and GRPO (Shao et al., 2024) approximate gradients of Eq. 1. In contrast, methods like DPO (Rafailov et al., 2023), A<sup>∗</sup>PO (Brantley et al., 2025), and QRPO (Matrenok et al., 2025) directly fit a parameterized policy π<sub>θ</sub> to the optimal policy $\pi ^ { * }$ given in Eq. 2. Our analysis and contributions are agnostic to the choice of optimizer, separating @k capabilities from other advancements.

Pass@k and max@k. Let $y _ { 1 } , \dotsc , y _ { k } \stackrel { \mathrm { i . i . d . } } { \sim } \pi ( \cdot | \ x )$ . We define max@k and pass@k as

$$
\operatorname* { m a x } \ @ \mathsf { k } ( \pi ; x ) = \mathbb { E } \big [ \operatorname* { m a x } _ { i = 1 , \ldots , k } R ( y _ { i } , x ) \big ] , \qquad \mathrm { p a s s @ k } ( \pi ; x ) = \mathbb { P } \left[ \exists i \in \{ 1 , \ldots , k \} : R ( y _ { i } , x ) = 1 \right] .
$$

Note that for binary rewards $R ( y , x ) \in \{ 0 , 1 \}$ , max@k coincides with pass@k. Moreover, for $k = 1$ , both expressions recover the ordinary @1 expected reward $\mathbb { E } _ { y \sim \pi ( \cdot | x ) } [ R ( y , x ) ]$ used in the standard RL objective $\mathcal { I }$ in Eq. 1. Let $V _ { k } ( \pi ; x )$ denote either of these @k objectives. Substituting $V _ { k } ( \pi ; x )$ into $\mathcal { I }$ in place of the @1 reward yields the KL-regularized objective studied in this work:

$$
\begin{array} { r } { \mathcal { I } _ { k } ( \pi ) = \mathbb { E } _ { x \sim \mathcal { D } } \left[ V _ { k } ( \pi ; x ) - \beta \mathrm { K L } \left( \pi ( \cdot \mid x ) \| \pi _ { \mathrm { r e f } } ( \cdot \mid x ) \right) \right] . } \end{array}\tag{3}
$$

## 4 @k OPTIMAL POLICIES AND ADAPTIVE PRESSURE

## 4.1 BINARY REWARDS: pass@k

Proofs of all results in this subsection, including boundary cases, are provided in Appendix B. We first consider binary rewards, i.e., we set $V _ { k } ( \pi ; x ) = \mathrm { p a s s } @ \mathrm { k } ( \pi ; x )$ in $\operatorname { E q . 3 }$ for $\mathcal { T } _ { k }$ , and derive the closed-form optimal policy which maximizes $\mathcal { T } _ { k }$ . It takes the familiar exponential-tilt form.

## Proposition 4.1: Closed-form optimal policy for the pass@k objective

Let $R ( y , x ) \in \{ 0 , 1 \}$ . Then the optimal policy for $\mathcal { T } _ { k }$ is

$$
\pi ^ { * } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { R ( y , x ) } { \beta _ { \mathrm { e f f } } ( x ) } \right) ,\tag{4}
$$

$$
\beta _ { \mathrm { e f f } } ( x ) = { \frac { \beta } { k ( 1 - p ^ { * } ( x ) ) ^ { k - 1 } } } { \mathrm { ~ a n d ~ } } Z ( x ) = 1 - p _ { \mathrm { r e f } } ( x ) + p _ { \mathrm { r e f } } ( x ) \exp \left( { \frac { 1 } { \beta _ { \mathrm { e f f } } ( x ) } } \right)\tag{5}
$$

Here, $p _ { \mathrm { r e f } } ( x ) = \mathbb { P } _ { y \sim \pi _ { \mathrm { r e f } } ( \cdot | x ) } [ R ( y , x ) = 1 ]$ and $p ^ { * } ( x ) = \mathbb { P } _ { y \sim \pi ^ { * } ( \cdot | x ) } [ R ( y , x ) = 1 ]$ are, respectively, the success probabilities of the reference model and optimal model.

Compared with the optimal policy (Eq. 2) for the single-reward objective (Eq. 1), the constant regularization strength $\beta$ is replaced by the prompt-dependent effective regularization strength $\beta _ { \mathrm { e f f } } ( x )$ The obtained expression for the optimal policy, however, remains implicit since $p ^ { * } ( x )$ depends on the unknown optimal policy $\pi ^ { * }$ . Nevertheless, it can be made fully explicit by summing $\operatorname { E q . }$ . 4 over successful completions, which yields a scalar equation relating $p ^ { * } ( x )$ directly to $p _ { \mathrm { r e f } } ( x )$

## Proposition 4.2: Scalar fixed-point characterization

For $0 < p _ { \mathrm { r e f } } ( x ) < 1$ , the optimal success probability $p ^ { * } ( x )$ is the unique solution in (0, 1) of

$$
k ( 1 - p ^ { * } ( x ) ) ^ { k - 1 } = \beta \log \frac { p ^ { * } ( x ) ( 1 - p _ { \mathrm { r e f } } ( x ) ) } { p _ { \mathrm { r e f } } ( x ) ( 1 - p ^ { * } ( x ) ) } .\tag{6}
$$

The reference success probability $p _ { \mathrm { r e f } } ( x )$ can be estimated directly from reference-model samples,<sup>1</sup>, while $p ^ { * } ( x )$ is computed by solving Eq. 6. For an error analysis, see Appendix K. Proposition 4.1 suggests two equivalent ways to recover the pass@k optimal policy with single-reward optimization.

## Corollary 4.3: pass@k as prompt-adaptive regularization

Optimizing the ordinary single-reward objective in Eq. 1 using the prompt-adaptive regularization strength $\beta _ { \mathrm { e f f } } ( x )$ from Eq. 5 results exactly in the pass@k optimal policy provided in Eq. 4.

This equivalence holds because both objectives decompose over prompts and $\beta _ { \mathrm { e f f } } ( x )$ is constant w.r.t. the completion y for each fixed x. As the optimal policy given in Eq. 4 depends on the reward and KL coefficient only through their ratio, we obtain a complementary reward-side interpretation.

## Corollary 4.4: pass@k as prompt-adaptive reward scaling

Optimizing the ordinary single-reward objective J given in Eq. 1 with constant regularization strength β but applying a prompt-adaptive reward transformation

$$
\begin{array} { r } { \widetilde { R } _ { k } ( y , x ) = \frac { \beta } { \beta _ { \mathrm { e f f } } ( x ) } R ( y , x ) = k ( 1 - p ^ { * } ( x ) ) ^ { k - 1 } R ( y , x ) } \end{array}\tag{7}
$$

results exactly in the pass@k optimal policy provided in Eq. 4.

Two views of adaptive pressure. Corollary 4.3 and Corollary 4.4 show that pass@k optimization can be expressed as ordinary single-reward optimization with either prompt-adaptive regularization or, equivalently, prompt-adaptive reward scaling. Taking $1 - p _ { \mathrm { r e f } } ( x )$ as a simple measure of prompt complexity, easy tasks x have high $p _ { \mathrm { r e f } } ( x )$ and consequently high $p ^ { * } ( x )$ , resulting in stronger effective regularization, while hard tasks receive weaker effective regularization and therefore stronger optimization pressure (see Figure 1a). Equivalently, under the reward-side view, a successful completion receives a smaller reward multiplier on easy tasks and a larger multiplier on difficult ones.

Intuition: reducing sharpening. Consider optimizing the usual single-reward objective (Eq. 1) on a binary-reward task. The reward gain depends only on the increase in success probability, regardless of initial performance—for example, moving from near zero to 0.1 on a hard task is worth as much as moving from 0.9 to 1 on an already-easy task. For an LLM, however, the latter is often much easier: the model already knows how to solve the task and only needs to concentrate probability on known successful behavior. In contrast, increasing a near-zero success probability requires acquiring entirely novel reasoning strategies. The adaptive pressure induced by pass@k objectives counteracts this tendency towards sharpening by imposing stronger effective regularization on already-easy tasks or, equivalently, through the prompt-adaptive reward scaling described above.

![](images/851abaf8b11c24b0dd4329630920eda774034c51a9341fa68c5f812ead54c4c8.jpg)  
(a) pass@k: effective regularization

![](images/3a33980b1612373710cfc1e656a649e56e46dc3dad21fc708e9d61bac81683e4.jpg)  
(b) max@k: effective regularization

![](images/65e58af98e30e0169b3d71fae4bdd9e1fd8307ed6edf6100a2e3ed35779fbb5d.jpg)  
(c) max@k: reward transformation

Figure 1: Adaptive pressure induced by @k objectives. (a) For binary rewards, increasing k strengthens regularization on easy tasks while retaining greater optimization pressure on difficult tasks. (b–c) For continuous rewards, we illustrate the same effect at $k = 4$ using synthetic “easy” Beta(8, 2) and “difficult” Beta(2, 8) reference reward distributions on $[ 0 , 1 ]$ . The easy task receives stronger effective regularization and smaller transformed rewards than the difficult task. All panels use $\bar { \beta } = 0 . 5$

## 4.2 CONTINUOUS REWARDS: max@k

Next, we consider real-valued rewards, i.e., we set $V _ { k } ( \pi ; x ) = \mathrm { m a x @ k } ( \pi ; x )$ in Eq. 3 for $\mathcal { T } _ { k }$ , and derive the associated closed-form optimal policy which maximizes $\mathcal { T } _ { k }$ . For $k > 1$ , we obtain:

## Proposition 4.5: Closed-form optimal policy for the max@k objective

The optimal policy for the max@k objective $\mathcal { T } _ { k }$ under real-valued rewards evaluates to

$$
\pi ^ { * } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { k } { \beta } \int _ { - \infty } ^ { R ( y , x ) } F _ { \pi ^ { * } } ( r \mid x ) ^ { k - 1 } d r \right) ,\tag{8}
$$

where $F _ { \pi ^ { * } } ( r \mid x ) : = \mathbb { P } _ { y ^ { \prime } \sim \pi ^ { * } ( \cdot \mid x ) } [ R ( y ^ { \prime } , x ) \le r ]$ is the reward CDF under the optimal policy.

The optimal policy identified above remains implicit because the CDF $F _ { \pi ^ { * } }$ on its right-hand side is itself induced by the unknown optimal policy $\pi ^ { * }$ . In contrast to the binary-reward case, computing $F _ { \pi ^ { * } }$ ∗ is more difficult and requires a sufficiently accurate estimate of the entire reference reward distribution $F _ { \pi _ { r e f } } .$ . It can, nevertheless, be computed numerically before training; we provide the procedure in Appendix D. All derivations and regularity conditions are explained in Appendix C.

Following the derivation for pass@k, a natural question is whether we can also identify a task-dependent $\beta _ { \mathrm { e f f } }$ here that reduces @k optimization to @1. A simple substitution shows that:

$$
\begin{array} { r } { \pi ^ { * } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { R ( y , x ) } { \beta _ { \mathrm { e f f } } ( R ( y , x ) , x ) } \right) \mathrm { ~ f o r ~ } \beta _ { \mathrm { e f f } } ( r , x ) = \frac { \beta } { k } r \displaystyle \int _ { - \infty } ^ { r } F _ { \pi ^ { * } } ( u \mid x ) ^ { k - 1 } d u . } \end{array}\tag{9}
$$

Note that, to ensure well-definedness, one can assume $R ( y , x ) > 0$ without loss of generality given (lower) bounded rewards, since the objectives in Eqs. 1 and 3 are invariant to constant reward shifts.

Eq. 9 provides an interpretation analogous to the binary-reward case: max@k replaces constant regularization with a strength that adapts to task complexity. In the max@k case, however, the effective strength also varies with the reward of the individual completion. As an illustrative example, Figure 1b uses two synthetic reference reward distributions on the [0, 1] interval representing an “easy” and a “difficult” task. This completion dependence means that substituting $\beta _ { \mathrm { e f f } } ( R ( y , x ) , x )$ into the ordinary single-reward objective would produce a different optimal policy; hence, the exact adaptive-regularization equivalence established in Corollary 4.3 for binary rewards does not extend to continuous rewards. Still, β $\beta _ { \mathrm { e f f } }$ provides a useful interpretation of how max@k redistributes optimization pressure and could be adapted in practice to design completion-adaptive regularization. Alternatively, we may follow our derivations for pass@k and instead consider adaptive reward transformations to redistribute optimization pressure. We obtain the following objective-level equivalence:

## Corollary 4.6: max@k as prompt-adaptive reward transformation

Optimizing the ordinary single-reward objective J given in Eq. 1 with the transformed reward

$$
\widetilde { R } ( y , x ) = S _ { x } ( R ( y , x ) ) { \mathrm { ~ f o r ~ } } S _ { x } ( r ) = k \int _ { - \infty } ^ { r } F _ { \pi ^ { * } } ( u \mid x ) ^ { k - 1 } d u\tag{10}
$$

results in the max@k optimal policy derived in Proposition 4.5.

The transformed-reward view shows the same prompt-adaptive effect. In the illustration in Figure 1c, the same raw reward is assigned a larger transformed value on the difficult task than on the easy task, providing a complementary reward-space view of how max@k can be used to control sharpening.

Explicit closed-form optimal policy for win-rate reward. As discussed, the optimal policy in Eq. 8 remains implicit because it depends on the unknown CDF $F _ { \pi ^ { * } \cdot } \mathbf { A }$ workaround is to consider a different but closely related reward: the reference quantile $F _ { \mathrm { r e f } } ( r \mid x ) = \mathbb { P } _ { y ^ { \prime } \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x ) } [ R ( y ^ { \prime } , x ) \leq r ]$ which preserves reward ordering and represents the win rate against reference-policy completions. Indeed, the max@k optimal policy for these win-rate rewards admits a fully explicit expression.

## Proposition 4.7: Explicit optimal policy for win-rate max@k

For a continuous reference reward distribution with CDF $\boldsymbol { F } _ { \mathrm { r e f } } ( \boldsymbol { r } \mid \boldsymbol { x } )$ , the objective $\mathcal { T } _ { k }$ with win-rate rewards $F _ { \mathrm { r e f } } ( R ( y , x ) \mid x )$ admits the following optimal policy:

$$
\pi _ { q } ^ { * } ( y \mid x ) = \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { S _ { x } ^ { \mathrm { q } } ( R ( y , x ) ) } { \beta } \right) ,\tag{11}
$$

where

$$
S _ { x } ^ { \mathrm { q } } ( r ) = \beta \log \left( C + \frac { 1 } { \beta } \Psi ^ { - 1 } ( F _ { \mathrm { r e f } } ( r \mid x ) ) ^ { k } \right) , \Psi ( z ) = \int _ { 0 } ^ { z } \frac { d u } { C + \frac { 1 } { \beta } u ^ { k } } , \mathrm { ~ a n d ~ } C \mathrm { ~ s . t . ~ } \Psi ( 1 ) = 1 .\tag{12}
$$

For a proof, see Appendix F. The scalar C and mapping Ψ depend only on k and $\beta ,$ and can be numerically pre-computed prior to training. The only prompt-dependent quantity in $S _ { x } ^ { \mathrm { q } }$ is $F _ { \mathrm { r e f } } ( r \mid x )$ which, similarly $\mathrm { t o } p _ { \mathrm { r e f } } ( x )$ for binary-rewards, can be estimated once from a fixed bank of referencemodel completions (see Appendix K for an error analysis). Thus, the adaptive-reward principle of Corollary 4.6 admits a directly computable realization for win-rate-based optimal policies.

## 5 KAPPO: @k ADAPTIVE PRESSURE FOR POLICY OPTIMIZATION

We introduce @k Adaptive Pressure for Policy Optimization (KAPPO), a framework for adapting existing @1 policy-optimization methods to @k settings. The parameter k can be tuned directly to control sharpening without otherwise changing the optimization algorithm. Given a base optimizer, KAPPO redistributes optimization pressure either through adaptive reward transformation or adap tive regularization, following our derivations in Section 4. The required prompt-dependent transformations and coefficients are estimated once before training at negligible cost and reused across runs; we call this stage reference calibration (see Appendix J). We instantiate both interfaces below to demonstrate $\mathrm { \langle A P P O ^ { \circ } s }$ generality and effectiveness across several popular RLVR optimizers.

## 5.1 REGULARIZATION-SIDE ADAPTATION: @k-DPO

The regularization-side interface is particularly useful for methods that do not directly optimize scalar rewards, most prominently DPO (Rafailov et al., 2023). We apply it in the binary-reward setting,<sup>2</sup> using the connection established in Corollary 4.3 to replace DPO’s global KL coefficient

with the prompt-dependent $\beta _ { \mathrm { e f f } } ( x )$ from Eq. 5. We call the resulting method @k-DPO:

$$
\begin{array} { r } { \mathcal { L } _ { \mathtt { G k } \mathrm { - } \mathrm { D P O } } ( \theta ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim \mathcal { D } _ { \mathtt { p r e f } } } \log \sigma \left( \beta _ { \mathrm { e f f } } ( x ) \left[ \log \frac { \pi _ { \theta } ( y _ { w } | x ) } { \pi _ { \mathtt { r e f } } ( y _ { w } | x ) } - \log \frac { \pi _ { \theta } ( y _ { l } | x ) } { \pi _ { \mathtt { r e f } } ( y _ { l } | x ) } \right] \right) . } \end{array}\tag{13}
$$

Before training, we perform reference calibration by generating a fixed bank of M reference-model completions for each prompt x and use the associated binary rewards to estimate $p _ { \mathrm { r e f } } ( x )$ . We then solve Eq. 6 for $p ^ { * } ( x )$ and use it to compute $\beta _ { \mathrm { e f f } } ( x )$ . Each online update subsequently requires only a pair of completions per prompt. Except for $\beta _ { \mathrm { e f f } }$ , the DPO training procedure remains unchanged.

## 5.2 REWARD-SIDE ADAPTATION: @k-A<sup>⋆</sup>PO AND @k-GRPO

To demonstrate the reward-side interface, we consider two complementary instantiations. First, we choose $A ^ { \star } \mathrm { P O }$ (Brantley et al., 2025), a method that uses only a single online completion per prompt. This choice isolates a central implication of our analysis: online tuples of k completions are not necessary to induce @k optimization pressure. Second, we also apply the interface to GRPO (Shao et al., 2024). Following common practice in RLVR, we use GRPO without a KL regularization term (Yu et al., 2025); this additionally tests whether KAPPO remains effective beyond the KL regularized setting assumed by our theory.

@k-A<sup>⋆</sup>PO. A<sup>⋆</sup>PO optimizes the squared-regression loss

$$
\begin{array} { r } { \mathcal { L } _ { A ^ { \star } \mathrm { P O } } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \mathrm { o l d } } ( \cdot | x ) } \left[ \left( R ( y , x ) - \beta \log Z ( x ) - \beta \log \frac { \pi _ { \theta } ( y | x ) } { \pi _ { \mathrm { r e f } } ( y | x ) } \right) ^ { 2 } \right] , } \end{array}\tag{14}
$$

with the partition function $Z ( x )$ estimated from reference-model completions. Replacing $R ( y , x )$ with the prompt-adaptive reward yields @k-A<sup>⋆</sup>PO. We treat both binary and continuous rewards:

For binary rewards, we substitute the adaptive reward $\frac { \beta R ( y , x ) } { \beta _ { \mathrm { e f f } } ( x ) }$ into Eq. 14. We obtain $\beta _ { \mathrm { e f f } }$ and $Z ( x )$ through the same one-time reference calibration procedure as for the binary @k-DPO instantiation.

For continuous rewards, we apply the win-rate-based adaptive reward transformation S<sup>q</sup> from Eq. 12 and set $Z ( x ) = 1$ since normalization follows automatically (see Proposition 4.7). We conduct a one-time reference calibration to approximate $\boldsymbol { F } _ { \mathrm { r e f } } ( \boldsymbol { r } \mid \boldsymbol { x } )$ As in the binary case, we sample a fixed bank $\{ y _ { j } ^ { \mathrm { r e f } } \} _ { j = 1 } ^ { M }$ of reference-model completions for each prompt. We then use this bank to construct the approximation $\begin{array} { r } { \widehat { F } _ { \mathrm { r e f } } ( r \mid x ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbb { I } [ R ( y _ { j } ^ { \mathrm { r e f } } , x ) \leq r ] } \end{array}$ for each prompt x.

@k-GRPO. Here, we construct @k-GRPO for the continuous-reward setting. For the corresponding binary-reward adaptation as well as additional properties of GRPO instantiations, see Appendix G. Given a group of n completions with rewards $\mathbf { \bar { \xi } } _ { R _ { i } } \mathbf { \bar { \xi } } = R ( y _ { i } , x )$ , we replace the raw rewards by the win-rate-based transformed rewards $S _ { x } ^ { \mathrm { q } } ( R _ { i } )$ in the GRPO loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathtt { Q k - G R P O } } ( \theta ) = - \mathbb { E } _ { x \sim \mathcal { D } , y _ { 1 } , \ldots , y _ { n } \sim \pi _ { \mathrm { o l d } } ( \cdot | x ) } \left[ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { 1 } { | y _ { i } | } \sum _ { t } u _ { i , t } ^ { \mathrm { c l i p } } ( \theta ) \right] , } \end{array}\tag{15}
$$

$$
\begin{array} { r } { u _ { i , t } ^ { \mathrm { c l i p } } ( \theta ) = \operatorname* { m i n } \bigl ( \frac { \pi _ { \theta } ( y _ { i } ^ { \epsilon } | y _ { i } ^ { \epsilon } , x ) } { \pi _ { \mathrm { o l d } } ( y _ { i } ^ { \epsilon } | y _ { i } ^ { \epsilon } , x ) } \widehat { A } _ { i } ^ { k } , \mathrm { c l i p } \left( \frac { \pi _ { \theta } ( y _ { i } ^ { \epsilon } | y _ { i } ^ { \epsilon } , x ) } { \pi _ { \mathrm { o l d } } ( y _ { i } ^ { \epsilon } | y _ { i } ^ { \epsilon } , x ) } , 1 - \epsilon , 1 + \epsilon \right) \widehat { A } _ { i } ^ { k } \bigr ) , \widehat { A } _ { i } ^ { k } = \frac { S _ { x } ^ { \mathrm { q } } ( R _ { i } ) - \frac { 1 } { n } \sum _ { j = 1 } ^ { n } S _ { x } ^ { \mathrm { q } } ( R _ { j } ) } { \mathrm { s t d } _ { j = 1 , \dots , n } [ S _ { x } ^ { \mathrm { q } } ( R _ { j } ) ] } . } \end{array}
$$

@k-GRPO uses the same one-time reference calibration as the continuous $\ @ \mathrm { k } { - } A ^ { \star } \mathrm { P O }$ instantiation.   
Apart from the reward transformation $S _ { x } ^ { \mathrm { q } }$ , the standard GRPO procedure remains unchanged.

## 6 EXPERIMENTS

## 6.1 SETUP

Tasks and scope. We evaluate KAPPO on mathematical reasoning with binary rewards (correctness) and code generation with continuous rewards (test case pass rate). We investigate all instantiations of KAPPO presented in Section 5, but note that KAPPO could be applied to other algorithms.

Model selection. Two principles guide our choice of initial checkpoints. First, we avoid models that have undergone reasoning-focused preference optimization or RL. This reflects the practical setting in which RL for reasoning is applied directly after supervised fine-tuning (SFT), and separates the effect of our training from reasoning behavior acquired via an undocumented prior curriculum. Second, we aim to minimize possible contamination of the training and evaluation datasets. These criteria lead us to OLMo-3-7B-Instruct-SFT and Qwen3-4B-Base. The former is an SFTonly, pre-RL checkpoint with a fully open training pipeline including explicit data decontamination procedures (Team Olmo et al., 2025). For a second model family, we use the Qwen pre-trained rather than a post-trained checkpoint to limit exposure to undocumented post-training data, although its pretraining corpus cannot be fully audited (Yang et al., 2025). Hyperparameters and checkpoints are selected on a validation split; complete experimental details are provided in Appendix H.

Baselines. As elaborated in Section 5, KAPPO can be applied to a wide range of RL optimizers by adjusting the effective regularization or transforming the reward. Since applying KAPPO with $k = 1$ leaves the underlying @1 optimizer unchanged, we use $k = 1$ as the primary baseline to isolate the effect of adaptive @k pressure. We also compare a KAPPO-adjusted A<sup>∗</sup>PO algorithm with PKPO (Walder & Karkhanis, 2025), a dedicated pass@k optimization algorithm that requires at least $n \geq k$ new on-policy completions per prompt.

## 6.2 BINARY REWARDS: MATHEMATICAL REASONING

Data and evaluation. We design the train–validation–test split to mirror a common model development setting: training data and validation benchmarks are closely coupled during development, while the final test consists of newer and often more difficult problems, appearing after model release. To closely match this setting, we perform RL on the training split of MATH and use its test split for in-distribution validation (Hendrycks et al., 2021). Final evaluation is performed on AIME 2026 (Mathematical Association of America, 2026), a markedly harder dataset whose problems predominantly exceed MATH’s highest difficulty tier. AIME 2026 also postdates both model initialization checkpoints, ruling out direct contamination of our test benchmark. A completion receives re ward one if its extracted final answer is correct and zero otherwise, i.e., we consider binary rewards.

Controlling sharpening by adjusting k. Figure 2 plots $\mathrm { p a s s  @ k _ { \mathrm { e v a l } } }$ against the number of completions $k _ { \mathrm { e v a l } }$ . We examine the improvement over the initial checkpoint as a function of the sampling budget. Gains exclusively at small $k _ { \mathrm { e v a l } }$ indicate that training has further concentrated probability on already-accessible behavior, whereas gains at large $k _ { \mathrm { e v a l } }$ point toward acquisition of novel behavior. Figure 2a considers reward-side adaptation. The A<sup>⋆</sup>PO baseline (pass@1 training) closely tracks GRPO, showing that single-completion optimization performs competitive to multi-completion optimization for pass@1 training. However, the gains achieved for both models diminish as $k _ { \mathrm { e v a l } }$ increases. Training with $k > 1$ typically (but not always) retains less of the pass@1 gain achieved by $k = 1$ training, but consistently overtakes the single-reward baselines at larger evaluation budgets. Figure 2b exhibits the same effect for the regularization-side interface. Standard DPO produces gains concentrated at small evaluation budgets, whereas @k-DPO trained with $k = 2$ overtakes it as $k _ { \mathrm { e v a l } }$ increases. This pattern holds for both OLMo and Qwen. Together, these results corroborate the effect predicted by our analysis across two base optimizers and both KAPPO interfaces:

![](images/034685dabc30eb79c56b0ca6ca0518786f2f68fb7c0335df752a23e3234f4d02.jpg)  
(a) A<sup>⋆</sup>PO and @k-A<sup>⋆</sup>PO.

![](images/b83c4918741cbcc49fb6c1d0a53b1a1fac19b1e1a1fad0baa9ab3722f49afbe4.jpg)  
(b) DPO and @k-DPO.  
Figure 2: pass@k on AIME 2026 as the sampling budget at test time $\left( \mathrm { k _ { e v a l } } \right)$ increases. Setting $k > 1$ during training consistently reduces sharpening. In Figure (a), we plot standard GRPO and $\ @ \mathrm { k } { - } A ^ { \star } \mathrm { P O }$ trained with $k \in \{ 1 , 2 , \overset { \cdot } { 4 } \}$ . In Figure (b), we plot @k-DPO trained with $k \in \{ 1 , 2 \}$

![](images/aa006242a0cb73a65c71846e96a7291fa7205548f0205058bfff93f3e5580b9b.jpg)  
Figure 3: MATH Level 5 validation performance of the starting checkpoints, @k-A<sup>⋆</sup>PO checkpoints, and PKPO checkpoints. @k-A<sup>⋆</sup>PO matches the performance of PKPO at a fraction of the on-policy training rollouts, requiring only a single-time reference calibration a $: \le 5 \%$ training time overhead.

@k adaptation counteracts excessive sharpening and shifts optimization pressure from already-easy tasks toward difficult tasks where successful behavior remains rare.

Agreement with dedicated multi-completion pass@k optimization. We next verify that a KAPPO adaptation of a single-completion optimizer can match the performance of a state-of-the-art dedicated multi-completion pass@k algorithm. To that end, we compare @k-A<sup>⋆</sup>PO with PKPO for $k \in \{ 2 , 4 \}$ using a matched number of prompt batches and the minimum feasible PKPO rollout mul tiplicity $( n = k )$ . Across both models, $\ @ \mathrm { k } - A ^ { \star } \mathrm { P O }$ achieves validation performance comparable to PKPO (see Figure 3) while requiring only a fraction of the on-policy rollouts, reducing training cost. This supports our central prediction: @k optimization can be achieved without relying on online kcompletion learning signals, but by injecting adaptive pressure into single-completion optimizers.

## 6.3 CONTINUOUS REWARDS: CODE GENERATION

Data and evaluation. We follow the same data selection strategy as in the binary-reward experiments: training and validation use closely related data, while final evaluation uses a newer and substantially harder problem source to approximate the distribution shift encountered after model deployment. We conduct our max@k experiments using Qwen3-4B-Base. Specifically, we train on the training split of LeetCodeDataset (Xia et al., 2025) and use its test split for validation. The two splits share the same source and task format, with a temporal cutoff on July 2024 (Xia et al., 2025). Each completion receives a continuous reward in [0, 1] equal to the fraction of test cases it passes. For final evaluation, we use the hard subset of LiveCodeBench v6, a substantially harder collection of competitive-programming problems (Jain et al., 2025) released between January and April 2025, limiting the opportunity for their inclusion in Qwen3 pretraining (released April 28th). Nevertheless, since Qwen’s pretraining cutoff is not documented, contamination cannot be ruled out completely.

Controlling sharpening by adjusting k. We observe the same qualitative effect as in the binary-reward experiments. Figure 4 compares the initial checkpoint with win-rate-based @k-A<sup>⋆</sup>PO and @k-GRPO trained with $k \in$ {1, 2, 4}. The $k \ = \ 1$ configuration typically provides the largest improvement for max@1, but its advantage contracts as the evaluation budget increases and the starting checkpoint eventually catches up. Training with $k > 1$ retains (slightly) less of the max@1 gain achieved by $k = 1$ training, but overtakes it at larger evaluation budgets, maintaining a greater advantage over the starting checkpoint. Thus, our explicit win-rate-based reward transformation provides analogous control over sharpening for continuous rewards as our binary-reward adaptations.

![](images/177ac0cb042d980b2c925ef6ad3114f9db3fcbb5758967575034d79c6a5dabbc.jpg)  
Figure $4 \colon \operatorname* { m a x } \ @ k _ { \mathrm { e v a l } }$ on the hard subset of Live-CodeBench v6 for Qwen3-4B. Win-rate-based $\uplus k \uplus \uplus \uplus$ (left) and @k-GRPO (right) mitigate excessive sharpening for $k > 1$

## 7 CONCLUSION

We derive closed-form optimal policies for KL-regularized pass@k and max@k optimization, proving that @k optimization redistributes pressure according to task difficulty. For binary rewards, this pressure can be expressed equivalently through prompt-adaptive reward scaling or regularization strength. For continuous rewards, it appears as an adaptive reward transformation or a completiondependent regularization strength, with a fully explicit construction for win-rate-based rewards. Leveraging these insights, we introduce KAPPO, a framework that adapts existing @1 policyoptimization methods to @k settings without requiring online multi-completion rollouts. In particular, our @k-A<sup>⋆</sup>PO instantiation requires only one new completion per prompt, yet achieves performance comparable to a SoTA multi-completion baseline. Validated across mathematical reasoning and code generation, @k-A<sup>⋆</sup>PO, @k-DPO, and @k-GRPO successfully control sharpening, confirming that diverse reasoning can be effectively encouraged via prompt-adaptive optimization pressure.

Limitations and future work. Our reference calibration relies on a static bank of M completions for every prompt instead of, e.g., optimizing the budget based on the noise transfer of the equations in Propositions 4.2 and 4.7, resulting in noisier estimates for $p ^ { * }$ and S<sup>q</sup>. Future work may develop better budget allocations. Second, our continuous-reward experiments optimize the win-rate surrogate S<sup>q</sup> rather than raw-reward max@k, leaving empirical evaluation of numerical $F _ { \pi ^ { * } }$ reconstruction (see Appendix D) to future work. Lastly, our win-rate solution assumes a continuous reference reward distribution; its application to discrete rewards with ties is an approximation.

## ACKNOWLEDGEMENTS

This work was supported as part of the Swiss AI Initiative by a grant from the Swiss National Supercomputing Centre (CSCS) under project ID infra01 on Alps. It was partially funded by an unrestricted gift from Coefficient Giving and by the Swiss National Science Foundation under grant number 212111. We are grateful to Robert West for his supervision and support.

## AI USE STATEMENT

We used generative AI tools to check proofs, improve their exposition, edit the manuscript, and assist with literature search and reference checking. AI also supported the implementation of methods, training infrastructure, evaluation, and plotting, following author-specified designs and guidance. For data processing, AI assisted with deduplication and analysis of dataset diversity, task difficulty, and metadata. The central research ideas and essential proof steps were developed by the authors. All experimental decisions, including experimental design, hyperparameters, comparisons, and all interpretations of results were determined solely by the authors. The authors reviewed the AIassisted code, verified mathematical claims and proofs, checked cited sources and their relevance, and checked figures against experimental outputs. We take responsibility for the final content of this work, including all AI-assisted text, claims, and artifacts.

## ETHICS STATEMENT

We improve understanding of pass@k and max@k optimization and introduce a method to adapt existing reinforcement learning methods from @1 to @k. As a result, post-training of frontier models may be accelerated, which can result in all kinds of secondary effects, both economic and societal. Our experiments use only publicly available models and datasets in accordance with their licenses.

## REPRODUCIBILITY STATEMENT

All considered datasets and models are publicly available. Instead of proposing a novel RL optimization method with many degrees of freedom to be specified, we identify a minimalist intervention to turn any standard @1 RL method into an @k method. We provide pseudocode for reference calibration with binary and continuous win-rate rewards in Appendix J.1. As a result, all of our experiments can be easily reproduced. For completeness, we describe the exact experimental details in Appendix H. Our entire code base is available in the supplementary materials.

## REFERENCES

Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin,´ Ahmet Ust<sup>¨</sup> un, and Sara Hooker. Back to basics: Revisiting REINFORCE-style optimiza-¨ tion for learning from human feedback in LLMs. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12248–12267, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.662. URL https://aclanthology.org/2024.acl-long.662/.

Zafarali Ahmed, Nicolas Le Roux, Mohammad Norouzi, and Dale Schuurmans. Understanding the impact of entropy on policy optimization. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 151–160. PMLR, 2019. URL https://proceedings.mlr.press/v97/ahmed19a.html.

Kiante Brantley, Mingyu Chen, Zhaolin Gao, Jason Lee, Wen Sun, Wenhao Zhan, and´ Xuezhou Zhang. Accelerating RL for LLM reasoning with optimal advantage regression. In Advances in Neural Information Processing Systems, volume 38, pp. 151492–151531. Curran Associates, Inc., 2025. doi: 10.52202/085713-5067. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ de8e1aebdb1e8a96970a58a641eac3cc-Abstract-Conference.html.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. doi: 10.48550/arXiv.2107.03374. URL https://arxiv.org/abs/2107.03374.

Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. AceReason-Nemotron: Advancing math and code reasoning through reinforcement learning. arXiv preprint arXiv:2505.16400, 2025a. doi: 10.48550/arXiv.2505.16400. URL https://arxiv.org/abs/2505.16400.

Zhipeng Chen, Xiaobo Qin, Youbin Wu, Yue Ling, Qinghao Ye, Wayne Xin Zhao, and Guang Shi. Pass@k training for adaptively balancing exploration and exploitation of large reasoning models. arXiv preprint arXiv:2508.10751, 2025b. doi: 10.48550/arXiv.2508.10751. URL https:// arxiv.org/abs/2508.10751.

Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Jennifer Dy and Andreas Krause (eds.), Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1861–1870. PMLR, 2018. URL https://proceedings.mlr.press/v80/haarnoja18b.html.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Joaquin Vanschoren and Serena Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https: //datasets-benchmarks-proceedings.neurips.cc/paper\_files/paper/ 2021/hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html.

Zican Hu, Shilin Zhang, Yafu Li, Jianhao (Elliott) Yan, Xuyang Hu, Leyang Cui, Xiaoye Qu, Chunlin Chen, Yu Cheng, and Zhi Wang. Diversity-incentivized exploration for versatile reasoning. In Carl Vondrick, Bharath Hariharan, Colin Raffel, Lerrel Pinto, Diyi Yang, and Aleksandra Faust

(eds.), International Conference on Learning Representations, volume 2026, pp. 71175–71205, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 74164872c24945a391473a8da5415843-Paper-Conference.pdf.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. In Yisong Yue, Animesh Garg, Nanyun Peng, Fei Sha, and Rose Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 58791–58831, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 94074dd5a072d28ff75a76dabed43767-Paper-Conference.pdf.

Robert Kirk, Ishita Mediratta, Christoforos Nalmpantis, Jelena Luketina, Eric Hambro, Edward Grefenstette, and Roberta Raileanu. Understanding the effects of RLHF on LLM generalisation and diversity. In International Conference on Learning Representations, pp. 20620–20653, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 5a68d05006d5b05dd9463dd9c0219db0-Paper-Conference.pdf.

Andreas Krause and Daniel Golovin. Submodular function maximization. In Lucas Bordeaux, Youssef Hamadi, and Pushmeet Kohli (eds.), Tractability: Practical Approaches to Hard Problems, chapter 3, pp. 71–104. Cambridge University Press, 2014. doi: 10.1017/ CBO9781139177801.004. URL https://doi.org/10.1017/CBO9781139177801. 004.

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. ProRL: Prolonged reinforcement learning expands reasoning boundaries in large language models. In Advances in Neural Information Processing Systems, volume 38, pp. 17998–18031. Curran Associates, Inc., 2025. doi: 10.52202/085713-0608. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 1a22b912945fb7c0bdd079e792b31b6f-Abstract-Conference.html.

Xiuyuan Lu and Benjamin Van Roy. Ensemble sampling. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Ad vances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/ file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf.

Pascal Massart. The tight constant in the Dvoretzky–Kiefer–Wolfowitz inequality. The Annals of Probability, 18(3):1269–1283, 1990. doi: 10.1214/aop/1176990746. URL https://doi. org/10.1214/aop/1176990746.

Mathematical Association of America. American invitational mathematics examination (AIME), 2026. URL https://maa.org/maa-invitational-competitions/.

Simon Matrenok, Skander Moalla, and Caglar Gulcehre. Quantile reward policy optimization: Alignment with pointwise regression and exact partition functions. In Advances in Neural Information Processing Systems, volume 38, pp. 102693–102750. Curran Associates, Inc., 2025. doi: 10.52202/085713-3434. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 94840c41497ead6a84f493f029eba7fa-Abstract-Conference.html.

Nicolas Menet, Andreas Krause, and Abbas Rahimi. POETS: Uncertainty-aware LLM optimization via compute-efficient policy ensembles. Advances in Neural Information Processing Systems, 39, 2026a. URL https://arxiv.org/abs/2605.07775.

Nicolas Menet, Aleksandar Terzic, Michael Hersche, Andreas Krause, and Abbas Rahimi. Thompson sampling via fine-tuning of LLMs. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 68794–68822, 2026b. URL https://proceedings.iclr.cc/paper\_files/paper/ 2026/file/707a2d58641b2192203b4bf4c532cfe1-Paper-Conference.pdf.

Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, and Koray Kavukcuoglu. Asynchronous methods for deep reinforcement learning. In Maria Florina Balcan and Kilian Q. Weinberger (eds.), Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pp. 1928–1937, New York, New York, USA, 2016. PMLR. URL https://proceedings.mlr.press/v48/mniha16.html.

Paavo Parmas, Yongmin Kim, Kohsei Matsutani, Shota Takashiro, Soichiro Nishimori, Takeshi Kojima, Yusuke Iwasawa, and Yutaka Matsuo. OrderGrad: Optimizing beyond the mean with order-statistic policy gradient estimation. arXiv preprint arXiv:2606.06096, 2026. doi: 10.48550/arXiv.2606.06096. URL https://arxiv.org/abs/2606.06096.

Ruotian Peng, Yi Ren, Zhouliang Yu, Weiyang Liu, and Yandong Wen. Beyond the sampled token: Preserving candidate support in RLVR. arXiv preprint arXiv:2510.14807, 2025. doi: 10.48550/ arXiv.2510.14807. URL https://arxiv.org/abs/2510.14807.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pp. 53728–53741. Curran Associates, Inc., 2023. doi: 10.52202/075280-2338. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017. doi: 10.48550/arXiv.1707. 06347. URL https://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024. doi: 10.48550/arXiv.2402.03300. URL https://arxiv.org/abs/2402.03300.

Yuda Song, Julia Kempe, and Remi Munos. Outcome-based exploration for LLM reasoning.´ arXiv preprint arXiv:2509.06941, 2025. doi: 10.48550/arXiv.2509.06941. URL https://arxiv. org/abs/2509.06941.

Fahim Tajwar, Guanning Zeng, Yueer Zhou, Yuda Song, Daman Arora, Yiding Jiang, Jeff Schneider, Ruslan Salakhutdinov, Haiwen Feng, and Andrea Zanette. Maximum likelihood reinforcement learning. In International Conference on Machine Learning, 2026. doi: 10.48550/arXiv.2602. 02710. URL https://icml.cc/virtual/2026/poster/65332.

Team Olmo, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lam bert, Pete Walsh, Pradeep Dasigi, Robert Berry, Saumya Malik, Saurabh Shah, Scott Geng, Shane Arora, Shashank Gupta, Taira Anderson, Teng Xiao, Tyler Murray, Tyler Romero, Victoria Graf, Akari Asai, Akshita Bhagia, Alexander Wettig, Alisa Liu, Aman Rangapur, Chloe Anastasiades, Costa Huang, Dustin Schwenk, Harsh Trivedi, Ian Magnusson, Jaron Lochner, Jiacheng Liu, Lester James V. Miranda, Maarten Sap, Malia Morgan, Michael Schmitz, Michal Guerquin, Michael Wilson, Regan Huff, Ronan Le Bras, Rui Xin, Rulin Shao, Sam Skjonsberg, Shannon Zejiang Shen, Shuyue Stella Li, Tucker Wilde, Valentina Pyatkin, Will Merrill, Yapei Chang, Yuling Gu, Zhiyuan Zeng, Ashish Sabharwal, Luke Zettlemoyer, Pang Wei Koh, Ali Farhadi, Noah A. Smith, and Hannaneh Hajishirzi. Olmo 3. arXiv preprint arXiv:2512.13961, 2025. doi: 10.48550/arXiv.2512.13961. URL https://arxiv.org/abs/2512.13961.

Jens Tuyls, Dylan Foster, Akshay Krishnamurthy, and Jordan Ash. Representation-based exploration for language models: From test-time to post-training. In Carl Vondrick, Bharath Hariharan, Colin Raffel, Lerrel Pinto, Diyi Yang, and Aleksandra Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 120621–120645, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ c420aca62da1d668e8e79e0bcae24e6b-Paper-Conference.pdf.

Christian Walder and Deep Tejas Karkhanis. Pass@K policy optimization: Solving harder reinforcement learning problems. In Advances in Neural Information Processing Systems, volume 38, pp. 152416–152445. Curran Associates, Inc., 2025. doi: 10.52202/085713-5096. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ df8a1a6306c3bcc89f6078759feadf62-Abstract-Conference.html.

Ronald J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine Learning, 8(3–4):229–256, 1992. doi: 10.1007/BF00992696. URL https: //doi.org/10.1007/BF00992696.

Yunhui Xia, Wei Shen, Yan Wang, Jason Klein Liu, Huifeng Sun, Siyue Wu, Jian Hu, and Xiaolong Xu. LeetCodeDataset: A temporal dataset for robust evaluation and efficient training of code LLMs. arXiv preprint arXiv:2504.14655, 2025. doi: 10.48550/arXiv.2504.14655. URL https: //arxiv.org/abs/2504.14655.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025. doi: 10.48550/arXiv.2505. 09388. URL https://arxiv.org/abs/2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pp. 113222–113244. Curran Associates, Inc., 2025. doi: 10.52202/085713-3775. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ a4277440d50f1f15d2cb4c14f7e0c0d2-Abstract-Conference.html.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In Advances in Neural Information Processing Systems, volume 38, pp. 57654–57689. Curran Associates, Inc., 2025. doi: 10.52202/085713-1933. URL https://proceedings.neurips.cc/paper\_files/paper/2025/hash/ 537d5aa768c2d534016a4d06f87bc8fb-Abstract-Conference.html.

## APPENDIX

## CONTENTS

A A Variational Optimality Lemma 16   
B Binary-Rewards Derivation 18   
B.1 Optimal Policy 18   
B.2 Scalar Equation and Uniqueness 19   
B.3 Boundary Cases . 20   
B.4 Objective-Level Equivalences 21   
C Continuous-Rewards Derivation 21   
C.1 Optimal Policy 22   
C.2 Objective-Level Equivalence 23   
D Numerical Reconstruction of the Raw-Reward Optimal CDF 24   
D.1 Continuous Reference Reward Distribution 24   
D.2 Discrete Reconstruction . 24   
D.3 Empirical Approximation . 25   
E Reward-Scale Assumptions for Continuous Rewards 26   
F Explicit max@k Solution for Win-Rate Rewards 27   
G KAPPO Adaptation of GRPO 29   
H Experimental Details 31   
I Compute Resources 31   
J Reference Calibration 31   
J.1 Calibration Algorithms 32   
J.2 Boundary Handling 32   
K Sensitivity to Finite Reference Calibration 34   
K.1 Binary Rewards . 35   
K.2 Continuous Rewards 39   
L Comparison with Direct Multi-Completion Optimization 41

## A A VARIATIONAL OPTIMALITY LEMMA

We first establish a general result used in the derivations of both optimal policies. Since all objectives considered in this work decompose over prompts, we fix a prompt x throughout this section. Let

$$
S _ { x } = \{ y : \pi _ { \mathrm { r e f } } ( y \mid x ) > 0 \}\tag{16}
$$

denote the support of the reference policy. Any policy assigning positive probability outside $S _ { x }$ has infinite KL divergence, so it is sufficient to optimize over distributions supported on $S _ { x }$

Consider an objective of the form

$$
\begin{array} { r } { \mathcal { I } _ { \boldsymbol { x } } ( \pi ) = U _ { \boldsymbol { x } } ( \pi ) - \beta \mathrm { K L } \left( \pi ( \cdot  { | \boldsymbol { \ x } ) } \| \pi _ { \mathrm { r e f } } ( \cdot  { | \boldsymbol { \ x } ) } \right) , } \end{array}\tag{17}
$$

where $U _ { x }$ is a concave reward functional and $\beta > 0$

For a feasible policy π, consider a perturbation

$$
\pi _ { \varepsilon } ( y \mid x ) = \pi ( y \mid x ) + \varepsilon h ( y \mid x ) ,\tag{18}
$$

where

$$
\sum _ { y \in S _ { x } } \left| h ( y \mid x ) \right| < \infty , \qquad \sum _ { y \in S _ { x } } h ( y \mid x ) = 0 ,\tag{19}
$$

and there exists $\varepsilon _ { 0 } > 0$ such that $\pi _ { \varepsilon } ( y \mid x ) \ge 0$ for every $\varepsilon \in [ 0 , \varepsilon _ { 0 } ]$ and $y \in S _ { x }$ . The first condition ensures that the perturbation has finite total variation, the second preserves normalization, and the third ensures that the perturbed policy remains feasible in the direction under consideration.

We assume that the corresponding one-sided directional derivative of $U _ { x }$ exists and can be written as

$$
\left. \frac { d } { d \varepsilon } U _ { x } ( \pi + \varepsilon h ) \right| _ { \varepsilon = 0 ^ { + } } = \sum _ { y \in S _ { x } } h ( y \mid x ) A _ { \pi } ( y , x )\tag{20}
$$

for some score $A _ { \pi } ( y , x )$ , with all displayed sums finite. The specific score corresponding to each @k objective is derived in the following sections.

We additionally assume that an optimal policy exists and has full support on $S _ { x } .$ Under the regularity conditions used below, the reward change produced by transferring probability ε between two completions is $O ( \varepsilon )$ . Suppose, however, that π $( y \mid x ) = 0$ for some $y \in S _ { x }$ and transfer probability ε to y. The resulting change in the negative KL term contains

$$
- \beta \varepsilon \log \varepsilon = \beta \varepsilon \log { \frac { 1 } { \varepsilon } } .\tag{21}
$$

Although this quantity tends to zero as $\varepsilon  0 .$ , its ratio to ε diverges. Therefore, for sufficiently small $\varepsilon ,$ the KL gain dominates the $O ( \varepsilon )$ reward change, and a policy assigning zero probability to part of $S _ { x }$ cannot be optimal.

Lemma A.1 (Variational form of a KL-regularized optimum). Under the assumptions above, any maximizer π<sup>∗</sup> ofEq. 17 satisfies

$$
\pi ^ { * } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { A _ { \pi ^ { * } } ( y , x ) } { \beta } \right) ,\tag{22}
$$

where

$$
Z ( x ) = \sum _ { y \in { \cal S } _ { x } } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { A _ { \pi ^ { * } } ( y , x ) } { \beta } \right) .\tag{23}
$$

Conversely, any feasible policy satisfying Eq. 22 is the unique maximizer of Eq. 17.

Proof. For compactness, define

$$
B _ { \pi ^ { * } } ( y , x ) = A _ { \pi ^ { * } } ( y , x ) - \beta \left( \log \frac { \pi ^ { * } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } + 1 \right) .\tag{24}
$$

Consider any perturbation h such that $\pi ^ { * } + \varepsilon h$ remains feasible for both positive and negative sufficiently small ε. Applying Eq. 20 to h and −h shows that the corresponding two-sided derivative exists. Since $\varepsilon = 0$ is a local maximum of $\mathcal { I } _ { x } ( \pi ^ { * } + \varepsilon h )$ , this derivative must vanish:

$$
\left. \frac { d } { d \varepsilon } \mathcal { I } _ { x } ( \pi ^ { * } + \varepsilon h ) \right| _ { \varepsilon = 0 } = 0 .\tag{25}
$$

The directional derivative of the KL term is

$$
\frac { d } { d \varepsilon } \mathrm { K L } \left( \pi ^ { * } + \varepsilon h \| \pi _ { \mathrm { r e f } } \right) \bigg | _ { \varepsilon = 0 } = \sum _ { y \in S _ { x } } h ( y \mid x ) \left( \log \frac { \pi ^ { * } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } + 1 \right) .\tag{26}
$$

Together with Eq. 20, this gives

$$
\sum _ { y \in S _ { x } } h ( y \mid x ) B _ { \pi ^ { * } } ( y , x ) = 0 .\tag{27}
$$

Now consider any two completions $y , z \in S _ { x }$ and choose

$$
h = \mathbf { e } _ { y } - \mathbf { e } _ { z } ,\tag{28}
$$

where $\mathbf { e } _ { y }$ denotes a unit point mass at $y .$ . Since $\pi ^ { * }$ has full support, this perturbation remains feasible for both signs of sufficiently small ε. Substituting it into Eq. 27 yields

$$
B _ { \pi ^ { * } } ( y , x ) - B _ { \pi ^ { * } } ( z , x ) = 0 .\tag{29}
$$

Because $y$ and z were arbitrary, $B _ { \pi ^ { * } } ( \boldsymbol { y } , \boldsymbol { x } )$ is constant with respect to y on $S _ { x }$ . Hence, there exists a prompt-dependent scalar $c ( x )$ such that

$$
A _ { \pi ^ { * } } ( y , x ) - \beta \left( \log \frac { \pi ^ { * } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } + 1 \right) = c ( x )\tag{30}
$$

for every $y \in S _ { x }$

Rearranging Eq. 30 gives

$$
\pi ^ { * } ( y \mid x ) = \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { A _ { \pi ^ { * } } ( y , x ) } { \beta } - \frac { c ( x ) } { \beta } - 1 \right) .\tag{31}
$$

The final two terms are independent of $y$ and can therefore be absorbed into the normalization constant. Imposing $\scriptstyle \sum _ { y } \pi ^ { * } ( y \mid x ) = 1$ gives Eqs. 22 and 23.

It remains to show that this stationary policy is the unique global maximizer. Let π be any other feasible policy and define

$$
h ( y \mid x ) = \pi ( y \mid x ) - \pi ^ { * } ( y \mid x ) .\tag{32}
$$

This perturbation satisfies

$$
\sum _ { y } \left| h ( y \mid x ) \right| \leq 2 , \qquad \sum _ { y } h ( y \mid x ) = 0 ,\tag{33}
$$

and

$$
\pi ^ { * } + \varepsilon h = ( 1 - \varepsilon ) \pi ^ { * } + \varepsilon \pi\tag{34}
$$

is a valid policy for every $\varepsilon \in [ 0 , 1 ]$

Define

$$
g ( \varepsilon ) = U _ { x } ( \pi ^ { * } + \varepsilon h ) .\tag{35}
$$

Because $U _ { x }$ is concave, $g$ is concave on [0, 1]. A concave function lies below its supporting line at the origin, and hence

$$
g ( 1 ) - g ( 0 ) \leq g ^ { \prime } ( 0 ^ { + } ) .\tag{36}
$$

Using Eq. 20, this gives

$$
U _ { x } ( \pi ) - U _ { x } ( \pi ^ { * } ) \leq \sum _ { y \in S _ { x } } \left( \pi ( y \mid x ) - \pi ^ { * } ( y \mid x ) \right) A _ { \pi ^ { * } } ( y , x ) .\tag{37}
$$

The KL difference admits the exact decomposition

$$
\begin{array} { r l r } {  { \mathrm { K L } ( \pi \| \pi _ { \mathrm { r e f } } ) - \mathrm { K L } ( \pi ^ { * } \| \pi _ { \mathrm { r e f } } ) } } \\ & { } & { = \displaystyle \sum _ { y \in \mathcal { S } _ { x } } ( \pi ( y \mid x ) - \pi ^ { * } ( y \mid x ) ) ( \log \frac { \pi ^ { * } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } + 1 ) + \mathrm { K L } ( \pi \| \pi ^ { * } ) . } \end{array}\tag{38}
$$

To verify this identity, expand

$$
\log { \frac { \pi ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } } = \log { \frac { \pi ( y \mid x ) } { \pi ^ { * } ( y \mid x ) } } + \log { \frac { \pi ^ { * } ( y \mid x ) } { \pi _ { \mathrm { r e f } } ( y \mid x ) } }\tag{39}
$$

inside the first KL term and use $\begin{array} { r } { \sum _ { y } ( \pi ( y \mid x ) - \pi ^ { * } ( y \mid x ) ) = 0 . } \end{array}$

Combining Eqs. 37 and 38 yields

$$
\begin{array} { l } { \mathcal { I } _ { x } ( \pi ) - \mathcal { J } _ { x } ( \pi ^ { * } ) \leq \displaystyle \sum _ { y \in \mathcal { S } _ { x } } \left( \pi ( y \mid x ) - \pi ^ { * } ( y \mid x ) \right) B _ { \pi ^ { * } } ( y , x ) } \\ { \displaystyle - \beta \operatorname { K L } \left( \pi \| \pi ^ { * } \right) . } \end{array}\tag{40}
$$

By Eq. 30, $B _ { \pi ^ { * } } ( y , x ) = c ( x )$ for every y. Since both policies have total mass one,

$$
\sum _ { y \in S _ { x } } \left( \pi ( y \mid x ) - \pi ^ { * } ( y \mid x ) \right) B _ { \pi ^ { * } } ( y , x ) = c ( x ) \sum _ { y \in S _ { x } } \left( \pi ( y \mid x ) - \pi ^ { * } ( y \mid x ) \right) = 0 .\tag{41}
$$

Consequently,

$$
\begin{array} { r } { \mathcal { I } _ { x } ( \pi ) - \mathcal { I } _ { x } ( \pi ^ { * } ) \leq - \beta \mathrm { K L } \left( \pi \| \pi ^ { * } \right) \leq 0 . } \end{array}\tag{42}
$$

Equality holds only if $\pi = \pi ^ { * }$ , proving that $\pi ^ { * }$ is the unique global maximizer.

The score $A _ { \pi } ( y , x )$ is defined only up to an additive term that depends on x but not on y. Such a term vanishes in Eq. 20 because every permitted perturbation has zero total mass, and it is absorbed into $Z ( x )$ in Eq. 22. The following sections derive the relevant score for the pass@k and max@k reward functionals.

## B BINARY-REWARDS DERIVATION

We now apply Lemma A.1 to the pass@k objective. As before, the objective decomposes over prompts, so we fix a prompt x. For any policy π, define its success probability

$$
p _ { \pi } ( x ) = \mathbb { P } _ { y \sim \pi ( \cdot | x ) } [ R ( y , x ) = 1 ] = \sum _ { y } \pi ( y \mid x ) R ( y , x ) .\tag{43}
$$

For binary rewards, the probability that at least one of k independent completions succeeds is

$$
U _ { x } ( \pi ) = 1 - \left( 1 - p _ { \pi } ( x ) \right) ^ { k } .\tag{44}
$$

The prompt-wise KL-regularized objective is therefore

$$
\begin{array} { r } { \mathcal { I } _ { k , x } ( \pi ) = 1 - \left( 1 - p _ { \pi } ( x ) \right) ^ { k } - \beta \mathrm { K L } \left( \pi ( \cdot \mid x ) \| \pi _ { \mathrm { r e f } } ( \cdot \mid x ) \right) . } \end{array}\tag{45}
$$

## B.1 OPTIMAL POLICY

Proposition 4.1 (restated). For $R ( y , x ) \in \{ 0 , 1 \}$ , the unique maximizer of $\mathcal { T } _ { k , x }$ satisfies

$$
\pi ^ { * } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { R ( y , x ) } { \beta _ { \mathrm { e f f } } ( x ) } \right) ,
$$

where

$$
\beta _ { \mathrm { e f f } } ( x ) = \frac { \beta } { k ( 1 - p ^ { * } ( x ) ) ^ { k - 1 } } , \qquad Z ( x ) = 1 - p _ { \mathrm { r e f } } ( x ) + p _ { \mathrm { r e f } } ( x ) e ^ { 1 / \beta _ { \mathrm { e f f } } ( x ) } ,
$$

with $p _ { \mathrm { r e f } } ( x ) = p _ { \pi _ { \mathrm { r e f } } } ( x )$ and $p ^ { * } ( x ) = p _ { \pi ^ { * } } ( x )$

Proof. We first consider $0 < p _ { \mathrm { r e f } } ( x ) < 1 $ ; the boundary cases are treated below. The mapping $p \mapsto \mathsf { \bar { 1 } } - ( 1 - p ) ^ { k }$ is concave on $[ 0 , 1 ]$ for $k \geq 1 { : }$ it is affine for $k = 1$ , while for $k \geq 2$

$$
{ \frac { d ^ { 2 } } { d p ^ { 2 } } } \left[ 1 - ( 1 - p ) ^ { k } \right] = - k ( k - 1 ) ( 1 - p ) ^ { k - 2 } \leq 0 .\tag{46}
$$

Since $p _ { \pi } ( x )$ is linear in $\pi ,$ the reward functional $U _ { x } ( \pi )$ is concave. We can therefore apply Lemma A.1 under its stated regularity assumptions.

Consider a perturbation $\pi _ { \varepsilon } = \pi + \varepsilon h$ satisfying the conditions of Appendix A. Its success probability is

$$
p _ { \pi _ { \varepsilon } } ( x ) = p _ { \pi } ( x ) + \varepsilon \sum _ { y } h ( y \mid x ) R ( y , x ) .\tag{47}
$$

Consequently, the directional derivative of the reward functional is

$$
\frac { d } { d \varepsilon } U _ { x } ( \pi + \varepsilon h ) \bigg \vert _ { \varepsilon = 0 ^ { + } } = k ( 1 - p _ { \pi } ( x ) ) ^ { k - 1 } \sum _ { y } h ( y \mid x ) R ( y , x ) .\tag{48}
$$

Thus, the score in Eq. 20 is

$$
A _ { \pi } ( y , x ) = k \left( 1 - p _ { \pi } ( x ) \right) ^ { k - 1 } R ( y , x ) .\tag{49}
$$

Applying Lemma A.1 gives

$$
\pi ^ { * } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { k } { \beta } \left( 1 - p ^ { * } ( x ) \right) ^ { k - 1 } R ( y , x ) \right) ,\tag{50}
$$

where

$$
p ^ { * } ( x ) = p _ { \pi ^ { * } } ( x ) .\tag{51}
$$

Define

$$
p _ { \mathrm { r e f } } ( x ) = \mathbb { P } _ { y \sim \pi _ { \mathrm { r e f } } ( \cdot | x ) } [ R ( y , x ) = 1 ] .\tag{52}
$$

Since the reward is binary, the reference policy assigns total mass $1 - p _ { \mathrm { r e f } } ( x )$ to unsuccessful completions and total mass $p _ { \mathrm { r e f } } ( x )$ to successful completions. Normalizing Eq. 50 therefore gives

$$
Z ( x ) = 1 - p _ { \mathrm { r e f } } ( x ) + p _ { \mathrm { r e f } } ( x ) \exp \left( \frac { k } { \beta } \left( 1 - p ^ { * } ( x ) \right) ^ { k - 1 } \right) .\tag{53}
$$

Introducing

$$
\beta _ { \mathrm { e f f } } ( x ) = \frac { \beta } { k \left( 1 - p ^ { * } ( x ) \right) ^ { k - 1 } } ,\tag{54}
$$

we recover the form reported in Eq. 4:

$$
\pi ^ { * } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { R ( y , x ) } { \beta _ { \mathrm { e f f } } ( x ) } \right) .\tag{55}
$$

The solution remains implicit because its right-hand side depends on the success probability of the unknown optimal policy.

## B.2 SCALAR EQUATION AND UNIQUENESS

It remains to determine $p ^ { * } ( x )$

Proposition 4.2 (restated). For $0 < p _ { \mathrm { r e f } } ( x ) < 1$ , the optimal success probability $p ^ { * } ( x )$ is the unique solution in (0, 1) of

$$
k ( 1 - p ^ { * } ( x ) ) ^ { k - 1 } = \beta \log \frac { p ^ { * } ( x ) ( 1 - p _ { \mathrm { r e f } } ( x ) ) } { p _ { \mathrm { r e f } } ( x ) ( 1 - p ^ { * } ( x ) ) } .
$$

□

Proof. Summing Eq. 50 over successful completions gives

$$
p ^ { * } ( x ) = \frac { p _ { \mathrm { r e f } } ( x ) \exp { \left( \frac { k } { \beta } ( 1 - p ^ { * } ( x ) ) ^ { k - 1 } \right) } } { 1 - p _ { \mathrm { r e f } } ( x ) + p _ { \mathrm { r e f } } ( x ) \exp { \left( \frac { k } { \beta } ( 1 - p ^ { * } ( x ) ) ^ { k - 1 } \right) } } .\tag{56}
$$

Taking the odds ratio and then the logarithm yields

$$
k \left( 1 - p ^ { * } ( x ) \right) ^ { k - 1 } = \beta \log \frac { p ^ { * } ( x ) \left( 1 - p _ { \mathrm { r e f } } ( x ) \right) } { p _ { \mathrm { r e f } } ( x ) \left( 1 - p ^ { * } ( x ) \right) } ,\tag{57}
$$

which is Eq. 6 in the main text.

To establish uniqueness, fix $q = p _ { \mathrm { r e f } } ( x ) \in ( 0 , 1 )$ and define, for $p \in ( 0 , 1 )$

$$
\Phi _ { q } ( p ) = \beta \log \frac { p ( 1 - q ) } { q ( 1 - p ) } - k ( 1 - p ) ^ { k - 1 } .\tag{58}
$$

Equation 57 is equivalent to $\Phi _ { q } ( p ) = 0$ . The function satisfies

$$
\operatorname * { l i m } _ { p \to 0 ^ { + } } \Phi _ { q } ( p ) = - \infty , \qquad \operatorname * { l i m } _ { p \to 1 ^ { - } } \Phi _ { q } ( p ) = + \infty ,\tag{59}
$$

and

$$
\Phi _ { q } ^ { \prime } ( p ) = \frac { \beta } { p ( 1 - p ) } + k ( k - 1 ) ( 1 - p ) ^ { k - 2 } > 0 .\tag{60}
$$

Hence, $\Phi _ { q }$ is continuous and strictly increasing from −∞ to +∞, so it has exactly one root in (0, 1). Moreover,

$$
\Phi _ { q } ( q ) = - k ( 1 - q ) ^ { k - 1 } < 0 ,\tag{61}
$$

which implies that the unique solution satisfies

$$
p ^ { * } ( x ) > p _ { \mathrm { r e f } } ( x ) .\tag{62}
$$

For $k = 1$ , Eq. 57 reduces to

$$
1 = \beta \log \frac { p ^ { * } ( x ) \left( 1 - p _ { \mathrm { r e f } } ( x ) \right) } { p _ { \mathrm { r e f } } ( x ) \left( 1 - p ^ { * } ( x ) \right) } ,\tag{63}
$$

whose solution is

$$
p ^ { * } ( x ) = \frac { p _ { \mathrm { r e f } } ( x ) e ^ { 1 / \beta } } { 1 - p _ { \mathrm { r e f } } ( x ) + p _ { \mathrm { r e f } } ( x ) e ^ { 1 / \beta } } .\tag{64}
$$

In this case, $\beta _ { \mathrm { { e f f } } } ( x ) = \beta$ , and Eq. 50 reduces to the standard KL-regularized single-reward optimal policy.

## B.3 BOUNDARY CASES

The scalar equation above assumes $0 < p _ { \mathrm { r e f } } ( x ) < 1$ . The two boundary cases follow directly from the support constraint imposed by the KL term.

Boundary cases. $\mathrm { I f } p _ { \mathrm { r e f } } ( x ) \in \{ 0 , 1 \}$ , the unique maximizer of $\mathcal { T } _ { k , x }$ is the reference policy itself:

$$
\pi ^ { * } ( \cdot \mid x ) = \pi _ { \mathrm { r e f } } ( \cdot \mid x ) , \qquad p ^ { * } ( x ) = p _ { \mathrm { r e f } } ( x ) .
$$

Proof. If $p _ { \mathrm { r e f } } ( x ) = 0$ , the reference policy assigns no probability to successful completions. Every policy with finite KL divergence is therefore also supported only on unsuccessful completions, so

$$
p ^ { * } ( x ) = 0 , \qquad \pi ^ { * } ( \cdot \mid x ) = \pi _ { \mathrm { r e f } } ( \cdot \mid x ) .\tag{65}
$$

The reward term is identically zero over the feasible set, and the reference policy uniquely minimizes the KL term.

$\mathrm { I f } p _ { \mathrm { r e f } } ( x ) = 1$ , every completion in the support of the reference policy is successful. Hence, every finite-KL policy has success probability one and the pass@k reward is constant:

$$
p ^ { * } ( x ) = 1 , \qquad \pi ^ { * } ( \cdot \mid x ) = \pi _ { \mathrm { r e f } } ( \cdot \mid x ) .\tag{66}
$$

The reference policy again uniquely minimizes the KL term.

For $k > 1$ , the case $p _ { \mathrm { r e f } } ( x ) = 1$ corresponds to $\beta _ { \mathrm { e f f } } ( x ) = + \infty \colon$ once the task is solved with probability one, the pass@k reward provides no further optimization pressure. For $k = 1$ , the effective regularization remains $\beta$

□

## B.4 OBJECTIVE-LEVEL EQUIVALENCES

We finally prove the two objective-level interpretations stated in the main text.

Corollary 4.3 (restated). Replacing the constant KL coefficient $\beta$ in the ordinary single-reward objective with $\beta _ { \mathrm { e f f } } ( x )$ from Eq. 54 recovers the pass@k optimal policy, with $\beta _ { \mathrm { e f f } } ( x ) = + \infty$ interpreted in the limiting sense.

Proof. For each prompt x with finite $\beta _ { \mathrm { e f f } } ( x )$ , consider the ordinary single-reward objective with this prompt-dependent KL coefficient. Its unique optimal policy is

$$
\pi _ { \mathrm { r e g } } ^ { * } ( y \mid x ) = \frac { 1 } { Z _ { \mathrm { r e g } } ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { R ( y , x ) } { \beta _ { \mathrm { e f f } } ( x ) } \right) .\tag{67}
$$

This is exactly the pass@k optimal policy derived in Eq. 50. Since the objective decomposes over prompts and $\beta _ { \mathrm { e f f } } ( x )$ is constant with respect to $y$ for each fixed x, the two objectives have the same optimal policy. When $\beta _ { \mathrm { e f f } } ( x ) = + \infty$ , the limiting policy is $\pi _ { \mathrm { r e f } }$ , agreeing with the boundary case above. □

Corollary 4.4 (restated). The ordinary single-reward objective with constant KL coefficient $\beta$ and transformed reward

$$
\widetilde { R } _ { k } ( y , x ) = \frac { \beta } { \beta _ { \mathrm { e f f } } ( x ) } R ( y , x ) = k ( 1 - p ^ { * } ( x ) ) ^ { k - 1 } R ( y , x )
$$

has the same unique optimal policy as the pass@k objective.

Proof. Retain the constant KL coefficient $\beta$ and replace the original binary reward with

$$
\widetilde { R } _ { k } ( y , x ) = \frac { \beta } { \beta _ { \mathrm { e f f } } ( x ) } R ( y , x ) = k \left( 1 - p ^ { * } ( x ) \right) ^ { k - 1 } R ( y , x ) .\tag{68}
$$

The corresponding single-reward optimal policy is

$$
\pi _ { \mathrm { r e w } } ^ { * } ( y \mid x ) = \frac { 1 } { Z _ { \mathrm { r e w } } ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { \widetilde { R } _ { k } ( y , x ) } { \beta } \right)\tag{69}
$$

$$
= \frac { 1 } { Z _ { \mathrm { r e w } } ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { R ( y , x ) } { \beta _ { \mathrm { e f f } } ( x ) } \right) .\tag{70}
$$

This again coincides exactly with the pass@k optimal policy.

## C CONTINUOUS-REWARDS DERIVATION

We now derive the optimal policy for the max@k objective with real-valued rewards. As before, the objective decomposes over prompts, so we fix a prompt x. For the derivation, it is sufficient to assume that

$$
R ( y , x ) \in [ a _ { x } , b _ { x } ]\tag{71}
$$

for every $y$ in the support of the reference policy, where the endpoints may depend on x. A common reward range across prompts is not required for the optimal-policy derivation; it becomes relevant only for the cross-prompt interpretation discussed in Appendix E.

Boundedness guarantees that the relevant expectations and directional derivatives are finite and permits differentiation under the expectation. The result extends to unbounded rewards whenever the corresponding expectations, integrals, and directional derivatives are finite, the required interchange of differentiation and expectation is valid, and the optimum is attained.

No continuity assumption on the reward distribution is required in this section. In particular, the reward CDF may contain atoms.

For a policy π, define

$$
U _ { k , x } ( \pi ) = \mathbb { E } _ { \boldsymbol { y } _ { 1 } , \dots , \boldsymbol { y } _ { k } ^ { \mathrm { ~ i ~ . i . d . } } \pi ( \cdot | \boldsymbol { x } ) } \left[ \operatorname* { m a x } _ { i = 1 , \dots , k } R ( \boldsymbol { y } _ { i } , \boldsymbol { x } ) \right] .\tag{72}
$$

The prompt-wise KL-regularized objective is

$$
\begin{array} { r } { \mathcal { I } _ { k , x } ( \pi ) = U _ { k , x } ( \pi ) - \beta \mathrm { K L } \left( \pi ( \cdot \mid x ) \| \pi _ { \mathrm { r e f } } ( \cdot \mid x ) \right) . } \end{array}\tag{73}
$$

## C.1 OPTIMAL POLICY

Proposition 4.5 (restated). For $k > 1$ and $\beta > 0$ , under the assumptions above, the unique maximizer of $\mathcal { T } _ { k , x }$ satisfies

$$
\pi ^ { * } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { k } { \beta } \int _ { - \infty } ^ { R ( y , x ) } F _ { \pi ^ { * } } ( r \mid x ) ^ { k - 1 } d r \right) ,
$$

where $F _ { \pi ^ { * } } ( r \mid x ) = \mathbb { P } _ { y \sim \pi ^ { * } ( \cdot | x ) } [ R ( y , x ) \leq r ]$ is the optimal reward CDF and $Z ( x )$ normalizes the policy.

Proof. Step 1: Concavity and existence. Let

$$
F _ { \pi } ( r \mid x ) = \mathbb { P } _ { y \sim \pi ( \cdot \mid x ) } [ R ( y , x ) \leq r ]\tag{74}
$$

denote the reward CDF under $\pi .$ Since the maximum of k independent rewards is bounded below by $a _ { x }$ , the tail-integral identity gives

$$
U _ { k , x } ( \pi ) = a _ { x } + \int _ { a _ { x } } ^ { b _ { x } } \left[ 1 - F _ { \pi } ( r \mid x ) ^ { k } \right] d r .\tag{75}
$$

For each fixed $r , F _ { \pi } ( r \mid x )$ is linear in π, while the mapping

$$
u \longmapsto 1 - u ^ { k }\tag{76}
$$

is concave on [0, 1]. Integration preserves concavity, so $U _ { k , x } ( \pi )$ is a concave functional of π. Combined with the strict concavity of the negative KL term, this implies that Eq. 73 has at most one maximizer.

For a finite completion space, existence follows from compactness of the probability simplex. For a countably infinite space, boundedness of the reward allows a maximizing sequence to be chosen with uniformly bounded KL divergence from $\pi _ { \mathrm { r e f } }$ . The corresponding KL sublevel sets are tight, so this sequence has a subsequence converging in total variation. The bounded reward functional is continuous under this convergence, while KL is lower semicontinuous, ensuring that the supremum is attained. Bounded rewards also satisfy the full-support condition discussed in Appendix A.

Step 2: Directional derivative. Consider a perturbation

$$
\pi _ { \varepsilon } = \pi + \varepsilon h\tag{77}
$$

satisfying the conditions of Appendix A. Expanding the product measure in $U _ { k , x } ( \pi _ { \varepsilon } )$ and differentiating at $\varepsilon = 0$ gives one term for each of the k sampled completions. By symmetry,

$$
{ \frac { d } { d \varepsilon } } U _ { k , x } ( \pi + \varepsilon h ) { \Big | } _ { \varepsilon = 0 ^ { + } } = k \sum _ { y } h ( y \mid x ) \mathbb { E } _ { y _ { 2 } , \ldots , y _ { k } ^ { \lfloor \cdot \rfloor , \ldots , n } ( \cdot \vert x ) } \left[ \operatorname* { m a x } \left\{ R ( y , x ) , R ( y _ { 2 } , x ) , \ldots , R ( y _ { k } , x ) \right\} \right] .\tag{78}
$$

Thus, one valid score in Eq. 20 is

$$
\Gamma _ { \pi } ( y , x ) = k \mathbb { E } _ { y _ { 2 } , \ldots , y _ { k } \to \ldots \pi ( \cdot | x ) } \left[ \operatorname* { m a x } \left\{ R ( y , x ) , R ( y _ { 2 } , x ) , \ldots , R ( y _ { k } , x ) \right\} \right] .\tag{79}
$$

To isolate the part that depends on $y ,$ let

$$
M _ { k - 1 } = \operatorname* { m a x } _ { i = 2 , \ldots , k } R ( y _ { i } , x ) .\tag{80}
$$

Using

$$
\mathrm { m a x } \{ a , b \} = b + ( a - b ) _ { + } , \qquad ( u ) _ { + } = \mathrm { m a x } \{ u , 0 \} ,\tag{81}
$$

we obtain

$$
\Gamma _ { \pi } ( y , x ) = k \mathbb { E } _ { \pi } [ M _ { k - 1 } ] + k \mathbb { E } _ { \pi } \left[ \left( R ( y , x ) - M _ { k - 1 } \right) _ { + } \right] .\tag{82}
$$

The first term depends on π and x but not on the candidate completion $y .$ It can therefore be omitted from the score by the additive-constant invariance established after Lemma A.1.

For any real number a and random variable M,

$$
( a - M ) _ { + } = \int _ { - \infty } ^ { a } \mathbb { I } [ M \leq r ] d r .\tag{83}
$$

Taking expectations and applying Tonelli’s theorem gives

$$
{ \mathbb E } [ ( a - M ) _ { + } ] = \int _ { - \infty } ^ { a } { \mathbb P } [ M \leq r ] d r .\tag{84}
$$

This identity holds for arbitrary reward distributions and does not require their CDFs to be continuous.

Because $M _ { k - 1 }$ is the maximum of $k - 1$ independent rewards sampled from $\pi ,$

$$
\mathbb { P } [ M _ { k - 1 } \leq r ] = F _ { \pi } ( r \mid x ) ^ { k - 1 } .\tag{85}
$$

Substituting this into Eq. 84, with $a = R ( y , x )$ , gives

$$
\mathbb { E } _ { \boldsymbol \pi } \left[ ( R ( y , x ) - M _ { k - 1 } ) _ { + } \right] = \int _ { - \infty } ^ { R ( y , x ) } F _ { \boldsymbol \pi } ( r \mid x ) ^ { k - 1 } d r .\tag{86}
$$

We may therefore choose the completion-dependent score

$$
A _ { \pi } ( y , x ) = k \int _ { - \infty } ^ { R ( y , x ) } F _ { \pi } ( r \mid x ) ^ { k - 1 } d r .\tag{87}
$$

Step 3: Optimal policy. Applying Lemma A.1 with the score in Eq. 87 gives

$$
\pi ^ { * } ( y \mid x ) = \frac { 1 } { Z ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { k } { \beta } \int _ { - \infty } ^ { R ( y , x ) } F _ { \pi ^ { * } } ( r \mid x ) ^ { k - 1 } d r \right) ,\tag{88}
$$

where

$$
Z ( x ) = \sum _ { y } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( { \frac { k } { \beta } } \int _ { - \infty } ^ { R ( y , x ) } F _ { \pi ^ { * } } ( r \mid x ) ^ { k - 1 } d r \right) .\tag{89}
$$

This is Eq. 8 from the main text. By the converse statement of Lemma A.1, any feasible policy satisfying this characterization is the unique global maximizer. □

The characterization remains implicit because the CDF $F _ { \pi ^ { * } }$ appearing on its right-hand side is itself induced by the unknown optimal policy. A numerical procedure for reconstructing $F _ { \pi ^ { * } }$ ∗ from the reference reward distribution is provided in Appendix D.

## C.2 OBJECTIVE-LEVEL EQUIVALENCE

Corollary 4.6 (restated). For $k > 1$ , define the fixed reward transformation

$$
S _ { x } ( r ) = k \int _ { - \infty } ^ { r } F _ { \pi ^ { * } } ( u \mid x ) ^ { k - 1 } d u .
$$

The ordinary single-reward objective with constant KL coefficient $\beta$ and transformed reward $\widetilde { R } ( y , x ) = S _ { x } ( R ( y , x ) )$ has the same unique optimal policy as the max@k objective.

Proof. Treating $S _ { x }$ as a fixed reward transformation, the corresponding single-reward optimal policy is

$$
\pi _ { \mathrm { r e w } } ^ { * } ( y \mid x ) = \frac { 1 } { Z _ { \mathrm { r e w } } ( x ) } \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { S _ { x } ( R ( y , x ) ) } { \beta } \right) .
$$

Substituting the definition of $S _ { x }$ gives exactly the exponent in Eq. 88. Normalization therefore yields $Z _ { \mathrm { r e w } } ( x ) = \overline { { Z } } ( x )$ and $\pi _ { \mathrm { r e w } } ^ { * } = \pi ^ { * }$ . Since both objectives decompose over prompts, the equivalence holds for the full objectives as well. □

## D NUMERICAL RECONSTRUCTION OF THE RAW-REWARD OPTIMAL CDF

The optimal policy in Eq. 88 depends on the unknown reward CDF $F _ { \pi ^ { * } }$ ∗ . In this section, we show how this CDF can be reconstructed numerically from the reference reward distribution. As in the previous derivations, we fix a prompt x and suppress its dependence where doing so does not cause ambiguity.

Define the optimal-policy score

$$
H ( r ) = k \int _ { - \infty } ^ { r } F _ { \pi ^ { * } } ( u \mid x ) ^ { k - 1 } d u\tag{90}
$$

and let

$$
c ( x ) = { \frac { 1 } { Z ( x ) } } .\tag{91}
$$

Because the likelihood ratio in Eq. 88 depends on a completion only through its reward, the reward distributions under the optimal and reference policies satisfy

$$
d P _ { \pi ^ { * } } ( r \mid x ) = c ( x ) \exp \left( \frac { H ( r ) } { \beta } \right) d P _ { \mathrm { r e f } } ( r \mid x ) .\tag{92}
$$

Thus, once the scalar $c ( x )$ is known, the optimal reward distribution can be reconstructed progressively from lower to higher rewards. The normalization condition determines the unique value of c(x).

## D.1 CONTINUOUS REFERENCE REWARD DISTRIBUTION

Suppose first that the reference reward distribution has density $f _ { \mathrm { r e f } } ( r \mid x )$ on a bounded interval $[ a _ { x } , b _ { x } ]$ . Let $f _ { \pi ^ { * } } ( r \mid x )$ denote the corresponding optimal density. Equation 92 gives

$$
f _ { \pi ^ { * } } ( r \mid x ) = c ( x ) f _ { \mathrm { r e f } } ( r \mid x ) \exp \left( \frac { H ( r ) } { \beta } \right) .\tag{93}
$$

Since

$$
\frac { d } { d r } F _ { \pi ^ { * } } ( r \mid x ) = f _ { \pi ^ { * } } ( r \mid x ) , \frac { d } { d r } H ( r ) = k F _ { \pi ^ { * } } ( r \mid x ) ^ { k - 1 } ,\tag{94}
$$

the unknown functions satisfy the coupled system

$$
\frac { d } { d r } F _ { \pi ^ { * } } ( r \mid x ) = c ( x ) f _ { \mathrm { r e f } } ( r \mid x ) \exp \left( \frac { H ( r ) } { \beta } \right) ,
$$

$$
\frac { d } { d r } H ( r ) = k F _ { \pi ^ { * } } ( r \mid x ) ^ { k - 1 } ,\tag{95}
$$

with initial conditions

$$
F _ { \pi ^ { * } } ( a _ { x } \mid x ) = 0 , \qquad H ( a _ { x } ) = 0 .\tag{96}
$$

For a candidate value of $c ( x )$ , the system can be integrated forward from $a _ { x }$ to $b _ { x }$ . The correct value is the one satisfying the terminal normalization condition

$$
F _ { \pi ^ { * } } ( b _ { x } \mid x ) = 1 .\tag{97}
$$

This gives a one-dimensional shooting problem. The terminal value $F _ { \pi ^ { * } } \left( b _ { x } \mid x \right)$ increases monotonically with $c ( x )$ , so the normalizing value can be found using bisection or another scalar root-finding method.

## D.2 DISCRETE RECONSTRUCTION

In practice, the reference reward distribution is estimated from a finite bank of reference-model completions and is therefore discrete. Let

$$
r _ { 1 } < r _ { 2 } < \cdots < r _ { m }\tag{98}
$$

be its distinct observed reward values, and define

$$
q _ { j } = \mathbb { P } _ { \pi _ { \mathrm { r e f } } } \left[ R ( y , x ) = r _ { j } \right] , \qquad j = 1 , \ldots , m .\tag{99}
$$

Let

$$
p _ { j } = \mathbb { P } _ { \pi ^ { * } } \left[ R ( y , x ) = r _ { j } \right] , \qquad P _ { j } = \sum _ { \ell = 1 } ^ { j } p _ { \ell }\tag{100}
$$

denote the optimal probability mass at reward $r _ { j }$ and the corresponding cumulative probability.

Define

$$
H _ { j } = H ( r _ { j } ) = k \int _ { - \infty } ^ { r _ { j } } F _ { \pi ^ { * } } ( u \mid x ) ^ { k - 1 } d u .\tag{101}
$$

Since the optimal CDF is constant between consecutive reward values,

$$
H _ { 1 } = 0 , \qquad H _ { j } = H _ { j - 1 } + k ( r _ { j } - r _ { j - 1 } ) P _ { j - 1 } ^ { k - 1 } , \quad j = 2 , \ldots , m .\tag{102}
$$

Grouping Eq. 88 by reward value gives

$$
p _ { j } = c ( x ) q _ { j } \exp \left( \frac { H _ { j } } { \beta } \right) .\tag{103}
$$

Equations 102 and 103 yield a forward recursion for any candidate $c = c ( x )$

$$
\begin{array} { r l } & { H _ { 1 } ( c ) = 0 , } \\ & { p _ { 1 } ( c ) = c q _ { 1 } , } \\ & { P _ { 1 } ( c ) = p _ { 1 } ( c ) , } \\ & { H _ { j } ( c ) = H _ { j - 1 } ( c ) + k ( r _ { j } - r _ { j - 1 } ) P _ { j - 1 } ( c ) ^ { k - 1 } , } \\ & { p _ { j } ( c ) = c q _ { j } \exp \left( \displaystyle \frac { H _ { j } ( c ) } { \beta } \right) , } \\ & { P _ { j } ( c ) = P _ { j - 1 } ( c ) + p _ { j } ( c ) , \qquad j = 2 , \ldots , m . } \end{array}\tag{104}
$$

The required normalization is

$$
P _ { m } ( c ) = 1 .\tag{105}
$$

Each quantity in the recursion is continuous and strictly increasing in c. Moreover,

$$
P _ { m } ( 0 ) = 0 ,\tag{106}
$$

while, for $q _ { 1 } > 0$

$$
P _ { m } \left( { \frac { 1 } { q _ { 1 } } } \right) \geq p _ { 1 } \left( { \frac { 1 } { q _ { 1 } } } \right) = 1 ,\tag{107}
$$

with strict inequality whenever $m > 1$ . Therefore, Eq. 105 has a unique solution in

$$
0 < c \leq { \frac { 1 } { q _ { 1 } } } ,\tag{108}
$$

which can be found efficiently by bisection. Equivalently, one may treat the optimal probability of the lowest reward, $p _ { 1 } = c q _ { 1 }$ , as the scalar search variable.

Once c has been determined, the recursion recovers all optimal masses $p _ { j }$ , cumulative probabilities $P _ { j }$ , and scores $H _ { j }$ . For $r \in [ r _ { j } , r _ { j + 1 } )$ , the reconstructed CDF and score are

$$
F _ { \pi ^ { * } } ( r \mid x ) = P _ { j } , \qquad H ( r ) = H _ { j } + k ( r - r _ { j } ) P _ { j } ^ { k - 1 } .\tag{109}
$$

Thus, the score is linearly interpolated between adjacent reward levels, with slope determined by the reconstructed optimal CDF.

## D.3 EMPIRICAL APPROXIMATION

Given M reference-model completions, the discrete reference masses are estimated as

$$
\widehat { q } _ { j } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { I } \left[ R ( y _ { i } ^ { \mathrm { r e f } } , x ) = r _ { j } \right] .\tag{110}
$$

Substituting these empirical masses into Eq. 104 produces an approximation of $F _ { \pi } { : }$ ∗ and the rawreward transformation H. The reconstruction must be performed once for each prompt and each pair (k, β).

Unlike the binary-reward case, where the reference distribution is summarized by the single scalar $p _ { \mathrm { r e f } } ( x )$ , the continuous case requires estimating the complete reference reward distribution. Errors in its estimated masses and reward locations propagate through the forward recursion and the subsequent exponential reweighting. This motivates the explicit reference-win-rate construction derived in Appendix F.

## E REWARD-SCALE ASSUMPTIONS FOR CONTINUOUS REWARDS

The derivation in Appendix C requires only prompt-wise bounded rewards, or suitable integrability conditions in the unbounded case. The stronger shared-range assumption used in the main text is needed only for interpreting the resulting adaptation across prompts.

Shift invariance. Both objectives studied in this work are invariant, at the level of their optimizer, to prompt-dependent constant reward shifts. For

$$
\begin{array} { r } { R _ { c } ( y , x ) = R ( y , x ) + c ( x ) , } \end{array}\tag{111}
$$

we have

$$
\begin{array} { r } { \mathbb { E } _ { y \sim \pi ( \cdot | x ) } [ R _ { c } ( y , x ) ] = \mathbb { E } _ { y \sim \pi ( \cdot | x ) } [ R ( y , x ) ] + c ( x ) } \end{array}\tag{112}
$$

and

$$
\mathbb { E } \left[ \operatorname* { m a x } _ { i = 1 , \dots , k } R _ { c } ( y _ { i } , x ) \right] = \mathbb { E } \left[ \operatorname* { m a x } _ { i = 1 , \dots , k } R ( y _ { i } , x ) \right] + c ( x ) .\tag{113}
$$

The shift therefore adds a policy-independent constant to Eqs. 1 and 3, leaving their optimal policies unchanged. Consequently, if rewards are bounded below by $r _ { \mathrm { m i n } } .$ , we may replace R with $R - r _ { \operatorname* { m i n } }$ and work with nonnegative rewards without loss of generality.

Nonnegative rewards and effective regularization. Suppose that the reward support for prompt x has lower endpoint $a _ { x }$ . Since $F _ { \pi ^ { * } } ( u \mid x ) = 0$ below $a _ { x }$ , the exponent in Eq. 8 can be written, for $r > a _ { x }$ , as

$$
\frac k { \beta } \int _ { a _ { x } } ^ { r } F _ { \pi ^ { * } } ( u \mid x ) ^ { k - 1 } d u = \frac { r - a _ { x } } { \beta _ { \mathrm { e f f } } ( r , x ) } ,\tag{114}
$$

where

$$
\beta _ { \mathrm { e f f } } ( r , x ) = \frac { \beta ( r - a _ { x } ) } { k \int _ { a _ { x } } ^ { r } F _ { \pi ^ { * } } ( u \mid x ) ^ { k - 1 } d u } .\tag{115}
$$

Shifting the lower endpoint to zero gives the simpler expression in Eq. 9. The nonnegative convention therefore provides a natural origin relative to which reward magnitude and effective regularization can be compared. Without such an origin, a definition based directly on $r / \beta _ { \mathrm { e f f } }$ would depend on an arbitrary reward shift and could become singular or negative.

This convention is not required for the transformed-reward interpretation. More generally, for any anchor a,

$$
S _ { x , a } ( r ) = k \int _ { a } ^ { r } F _ { \pi ^ { * } } ( u \mid x ) ^ { k - 1 } d u .\tag{116}
$$

Changing a adds the same constant to $S _ { x , a } ( r )$ for every completion of a fixed prompt. This constant is absorbed into the partition function and therefore does not affect the optimal policy.

Common reward range. Our cross-prompt interpretation treats the location of the reference reward distribution — and, informally, its mean

$$
\begin{array} { r } { \mu _ { \mathrm { r e f } } ( x ) = \mathbb { E } _ { y \sim \pi _ { \mathrm { r e f } } ( \cdot | x ) } [ R ( y , x ) ] } \end{array}\tag{117}
$$

— as an indicator of task difficulty: prompts receiving lower rewards under the reference policy are regarded as more difficult. This interpretation is meaningful only when rewards can be compared

across prompts. In particular, a reward value $r _ { 1 }$ must represent a better outcome than $r _ { 2 } ~ < ~ r _ { 1 }$ independently of which prompts produced the two values.

We therefore assume a shared, semantically meaningful fixed range,

$$
R ( y , x ) \in [ r _ { \operatorname* { m i n } } , r _ { \operatorname* { m a x } } ] \qquad \mathrm { f o r } \mathrm { e v e r y } x .\tag{118}
$$

Without common anchors, the average reward cannot define prompt difficulty consistently. Indeed, because the objectives are invariant to prompt-dependent reward shifts, one could change the mean reward of any prompt arbitrarily without changing either its optimization problem or its optimal policy. A difference between two means would then reflect only a difference in reward coordinates rather than a difference in task difficulty. Prompt-dependent rescalings create the same comparability problem and additionally alter the balance between reward and KL regularization unless $\beta$ is rescaled accordingly. A common fixed range rules out these ambiguities, as in normalized verifier scores and test-case pass rates.

The mean provides an intuitive summary of difficulty but does not, by itself, guarantee a pointwise ordering of the adaptive transformations. A sufficient condition for the ordering illustrated in Figures 1b and 1c can be stated in terms of the optimal reward CDFs. Consider an easy prompt $x _ { \mathrm { e } }$ and a difficult prompt $x _ { \mathrm { h } }$ on a common range $[ 0 , r _ { \mathrm { m a x } } ]$ such that

$$
F _ { \pi ^ { * } } ( r \mid x _ { \mathrm { e } } ) \leq F _ { \pi ^ { * } } ( r \mid x _ { \mathrm { h } } ) \qquad { \mathrm { f o r ~ a l l ~ } } r \in [ 0 , r _ { \operatorname* { m a x } } ] .\tag{119}
$$

Then, for every common raw reward value,

$$
S _ { x _ { \mathrm { e } } } ( r ) \leq S _ { x _ { \mathrm { h } } } ( r ) , \qquad \beta _ { \mathrm { e f f } } ( r , x _ { \mathrm { e } } ) \geq \beta _ { \mathrm { e f f } } ( r , x _ { \mathrm { h } } ) .\tag{120}
$$

Thus, the easy prompt receives a smaller transformed reward and stronger effective regularization. This condition formalizes the representative easy–difficult comparison used in the main text; it is not a universal definition of task difficulty. If the reward CDFs cross, the corresponding transformation and effective-regularization curves may also cross, and no global ordering is implied.

## F EXPLICIT max@k SOLUTION FOR WIN-RATE REWARDS

We now derive the explicit solution in Eqs. 11–12. Fix a prompt x and suppress its dependence where unambiguous. Let

$$
F _ { \mathrm { r e f } } ( r \mid x ) = \mathbb { P } _ { y ^ { \prime } \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x ) } [ R ( y ^ { \prime } , x ) \leq r ]\tag{121}
$$

denote the reference reward CDF, and define the reference quantile, or win rate, of a completion y as

$$
Q ( y ) = F _ { \mathrm { r e f } } ( R ( y , x ) \mid x ) .\tag{122}
$$

Proposition 4.7 (restated). Suppose that the reference reward distribution is continuous. For $k \geq 1$ and $\beta > 0$ , the KL-regularized max@k objective with reward $Q ( y )$ has the unique optimal policy

$$
\pi _ { q } ^ { * } ( y \mid x ) = \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { S _ { x } ^ { \mathrm { q } } ( R ( y , x ) ) } { \beta } \right) ,
$$

where

$$
S _ { x } ^ { \mathrm { q } } ( r ) = \beta \log \left( C + \frac { [ \Psi ^ { - 1 } ( F _ { \mathrm { r e f } } ( r \mid x ) ) ] ^ { k } } { \beta } \right) , \qquad \Psi ( z ) = \int _ { 0 } ^ { z } \frac { d u } { C + u ^ { k } / \beta } ,
$$

and $C > 0$ is uniquely determined by $\Psi ( 1 ) = 1$ . The partition function for this transformed reward is exactly one.

Proof. Step 1: Deriving the optimal win-rate density. Since the reference reward distribution is atomless, the probability integral transform gives

$$
Q ( y ) \sim \mathrm { U n i f } [ 0 , 1 ] , \qquad y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x ) .\tag{123}
$$

Let

$$
G ( q ) = \mathbb { P } _ { y \sim \pi _ { q } ^ { * } ( \cdot | x ) } [ Q ( y ) \leq q ]\tag{124}
$$

be the win-rate CDF under the optimal policy, and let $g ( q ) = G ^ { \prime } ( q )$ denote its density. The optimality condition from Appendix C, applied with $Q ( y )$ as the reward, gives

$$
\pi _ { q } ^ { * } ( y \mid x ) = \pi _ { \mathrm { r e f } } ( y \mid x ) w ( Q ( y ) ) , \qquad w ( q ) = \frac { 1 } { Z ( x ) } \exp \left( \frac { k } { \beta } \int _ { 0 } ^ { q } G ( u ) ^ { k - 1 } d u \right) .
$$

For $k = 1$ , the same expression follows directly from the ordinary KL-regularized optimal policy in Eq. 2.

The multiplier $w ( Q ( y ) )$ depends on the completion only through its win rate. Since $Q$ is uniform under the reference policy, its CDF under the reweighted policy satisfies

$$
G ( q ) = \mathbb { E } _ { y \sim \pi _ { \mathrm { r e f } } ( \cdot | x ) } \left[ w ( Q ( y ) ) \mathbb { I } \{ Q ( y ) \leq q \} \right] = \int _ { 0 } ^ { q } w ( u ) d u .
$$

Differentiating gives $g ( q ) = w ( q )$ , and hence $\pi _ { q } ^ { * } ( y \mid x ) = \pi _ { \mathrm { r e f } } ( y \mid x ) g ( Q ( y ) )$ . In particular,

$$
g ( q ) = \frac 1 { Z ( x ) } \exp \left( { \frac k \beta } \int _ { 0 } ^ { q } G ( u ) ^ { k - 1 } d u \right) .\tag{125}
$$

Differentiating Eq. 125 yields

$$
\frac { g ^ { \prime } ( q ) } { g ( q ) } = \frac { k } { \beta } G ( q ) ^ { k - 1 } .\tag{126}
$$

Using $G ^ { \prime } ( q ) = g ( q ) > 0$ and treating g as a function of G, we obtain

$$
\frac { d g } { d G } = \frac { g ^ { \prime } ( q ) } { G ^ { \prime } ( q ) } = \frac { k } { \beta } G ^ { k - 1 } .\tag{127}
$$

Integration therefore gives

$$
g ( q ) = C + \frac { 1 } { \beta } G ( q ) ^ { k } ,\tag{128}
$$

where $C = g ( 0 ) > 0$ . Thus, the optimal win-rate CDF satisfies

$$
G ^ { \prime } ( q ) = C + \frac { 1 } { \beta } G ( q ) ^ { k } , \qquad G ( 0 ) = 0 , \qquad G ( 1 ) = 1 .\tag{129}
$$

Step 2: Determining the constant and the CDF. Separating variables gives

$$
q = \int _ { 0 } ^ { G ( q ) } \frac { d u } { C + u ^ { k } / \beta } = \Psi ( G ( q ) ) .\tag{130}
$$

The terminal condition $G ( 1 ) = 1$ requires

$$
1 = \int _ { 0 } ^ { 1 } \frac { d u } { C + u ^ { k } / \beta } .\tag{131}
$$

The right-hand side is continuous and strictly decreasing in $C > 0$ . It diverges as $C \downarrow 0$ for every $k \geq 1$ and converges to zero as $C  \infty$ . Hence Eq. 131 has a unique solution $C > 0$ . For this value of $C .$ , Ψ is a strictly increasing bijection from [0, 1] to [0, 1], and therefore

$$
G = \Psi ^ { - 1 } .\tag{132}
$$

Step 3: Recovering and verifying the optimal policy. As established in Step 1, the optimal completion policy satisfies $\pi _ { q } ^ { * } ( y \mid \bar { x } ) \stackrel { - } { = } \pi _ { \mathrm { r e f } } \bar { ( } y \mid x ) \bar { g } ( Q ( y ) )$ . Substituting the density from Eq. 128 and $Q ( y ) = F _ { \mathrm { r e f } } ( R ( y , x ) \mid x )$ gives

$$
\pi _ { q } ^ { * } ( y \mid x ) = \pi _ { \mathrm { r e f } } ( y \mid x ) \left( C + \frac { [ G ( F _ { \mathrm { r e f } } ( R ( y , x ) \mid x ) ) ] ^ { k } } { \beta } \right) .\tag{133}
$$

Equivalently, defining

$$
S _ { x } ^ { \mathrm { q } } ( r ) = \beta \log { \left( C + \frac { [ G ( F _ { \mathrm { r e f } } ( r \mid x ) ) ] ^ { k } } { \beta } \right) }\tag{134}
$$

yields

$$
\pi _ { q } ^ { * } ( y \mid x ) = \pi _ { \mathrm { r e f } } ( y \mid x ) \exp \left( \frac { S _ { x } ^ { \mathrm { q } } ( R ( y , x ) ) } { \beta } \right) .\tag{135}
$$

Normalization is already incorporated because

$$
\mathbb { E } _ { y \sim \pi _ { \mathrm { r e f } } ( \cdot | x ) } \left[ \exp \left( \frac { S _ { x } ^ { \mathrm { q } } ( R ( y , x ) ) } { \beta } \right) \right] = \int _ { 0 } ^ { 1 } g ( q ) d q = G ( 1 ) - G ( 0 ) = 1 .\tag{136}
$$

To verify sufficiency, observe that the constructed policy induces win-rate density $g$ and hence CDF G. Moreover, differentiating $g = C + G ^ { k } / \beta$ with $\bar { G } ^ { \prime } = \bar { g }$ gives $g ^ { \prime } / g = ( k / \beta ) G ^ { k - 1 }$ . Integrating and using $g ( 0 ) = C$ recovers Eq. 125, with normalizing constant $1 / C$ . The policy therefore satisfies the sufficient optimality condition from Appendix C. Strict concavity of the KL-regularized objective establishes uniqueness. □

The quantities $C , \Psi$ , and G depend only on k and $\beta$ and can therefore be precomputed independently of the prompt. All prompt dependence enters through the reference CDF $F _ { \mathrm { r e f } }$

For $k = 1$ , the construction reduces to the standard exponential tilt:

$$
g ( q ) = \frac { \exp ( q / \beta ) } { \displaystyle \int _ { 0 } ^ { 1 } \exp ( u / \beta ) d u } .\tag{137}
$$

Ties and discrete rewards. In practical settings such as test-case pass-rate rewards, the reference reward distribution is discrete and contains ties. The deterministic win rate $Q ( y ) = F _ { \mathrm { r e f } } ( R ( y , x )$ x) is then not exactly uniform under the reference policy, so the continuous solution provides an approximation to the optimal policy for this win-rate objective. We retain this deterministic convention in our experiments and evaluate the resulting adaptation empirically. The identity $Z ( x ) = 1$ likewise holds for the true CDF of a continuous reference reward distribution, but need not hold for discrete rewards or finite-bank CDF estimates. Our implementation retains $Z ( x ) = 1$ as part of the same continuous-reference approximation.

## G KAPPO ADAPTATION OF GRPO

GRPO constructs its learning signal by centering and standardizing rewards within each prompt’s completion group. This section explains how that normalization interacts with the adaptive reward transformations derived in Section 4. For a group of rewards $R _ { 1 } , \ldots , R _ { n }$ , write $\begin{array} { r } { \bar { R } = \frac { 1 } { n } \sum _ { i } R _ { i } } \end{array}$ and $s _ { R } = \mathrm { s t d } _ { i } ( R _ { i } )$ . For clarity, we omit the numerical denominator stabilizer and write the expressions below for nonzero-variance groups; constant-reward groups receive zero advantages.

Cancellation of binary reward scaling. For binary rewards, Corollary 4.4 gives the transformed reward $\widetilde { R } _ { i } = a _ { k } ( x ) R _ { i }$ , where

$$
a _ { k } ( x ) = { \frac { \beta } { \beta _ { \mathrm { e f f } } ( x ) } } = k ( 1 - p ^ { * } ( x ) ) ^ { k - 1 } .\tag{138}
$$

Since this multiplier is shared by all completions of the same prompt, it scales both the centered rewards and their standard deviation. For $a _ { k } ( x ) > 0$ , the resulting advantages satisfy

$$
\frac { \widetilde { R } _ { i } - \overline { { \widetilde { R } } } } { \mathrm { s t d } _ { j } ( \widetilde { R } _ { j } ) } = \frac { a _ { k } ( x ) ( R _ { i } - \bar { R } ) } { a _ { k } ( x ) s _ { R } } = \frac { R _ { i } - \bar { R } } { s _ { R } } .\tag{139}
$$

Thus, substituting the transformed binary rewards before group standardization cancels the intended prompt-dependent scaling.

Preserving the binary adaptation. The core issue here is that standardized GRPO does not generally estimate the policy gradient of the underlying expected-reward objective. Dividing each group by its empirical reward standard deviation reweights each prompt’s update by a factor that depends on the sampled completions and the current policy; consequently, the expected GRPO update generally differs from the policy gradient of the original objective. For the same reason, KAPPO’s adaptive reward scaling is discarded, as shown in Eq. 139. Therefore, the failure of ordinary reward substitution in this case arises precisely because standardized GRPO does not follow the objective to which that connection applies.

Accordingly, we treat the KAPPO adaptation of standardized GRPO as an empirical extension of our theoretical result. To retain the derived redistribution of pressure while preserving GRPO’s normalization, we apply the adaptive multiplier after standardization:

$$
\widehat { A } _ { i } ^ { k } = a _ { k } ( x ) \frac { R _ { i } - \bar { R } } { s _ { R } } .\tag{140}
$$

Since the clipped GRPO surrogate is positively homogeneous in the advantage, this construction multiplies each prompt’s contribution to the GRPO loss and gradient by exactly $a _ { k } ( x )$ . It therefore preserves the theoretically derived cross-prompt pressure redistribution within GRPO’s existing learning signal, without claiming an exact objective-level equivalence. This provides a natural reward-side adaptation for GRPO without KL regularization.

Two alternative implementations recover the theoretically grounded interfaces more directly. Removing standard-deviation normalization while retaining group centering allows the ordinary KAPPO reward scaling to survive. Alternatively, for KL-regularized GRPO, the regularization-side interface can be applied through the adaptive coefficient $\beta _ { \mathrm { e f f } } ( x )$ , as established in Corollary 4.3.

Continuous reward transformations. For continuous rewards, group standardization does not cause the same cancellation. The transformation S<sup>q</sup> is nonlinear for $k > 1$ , whereas centering and standardization eliminate only affine changes in reward. The transformed rewards are therefore affected by GRPO’s normalization but are not generally discarded by it. This is consistent with the purpose of the adaptation: KAPPO supplies the @k reward transformation, while GRPO retains its defining group-relative reweighting of the resulting learning signal.

For the win-rate-based transformation used in our experiments, the interaction is particularly clear. At the beginning of training, when the current policy is close to the reference policy, the population mean and standard deviation of the transformed rewards are approximately the same for every prompt. GRPO normalization is therefore trivial with respect to prompt adaptivity: it applies the same affine transformation across prompts and does not alter the redistribution of optimization pressure induced by KAPPO.

Indeed, write

$$
\begin{array} { r } { Q _ { x } ( Y ) = F _ { \mathrm { r e f } } ( R ( Y , x ) \mid x ) , \qquad S _ { x } ^ { \mathrm { q } } ( R ( Y , x ) ) = T _ { k , \beta } ( Q _ { x } ( Y ) ) , } \end{array}\tag{141}
$$

where $T _ { k , \beta }$ is shared across prompts. Under the continuous-reference assumption from Appendix F, at initialization

$$
Q _ { x } ( Y ) \sim \mathrm { U n i f } [ 0 , 1 ] , \qquad Y \sim \pi _ { \mathrm { r e f } } ( \cdot \mid x ) .\tag{142}
$$

Hence,

$$
\mu _ { k , \beta } = \mathbb { E } _ { U \sim \operatorname { U n i f } [ 0 , 1 ] } [ T _ { k , \beta } ( U ) ] , \qquad \sigma _ { k , \beta } ^ { 2 } = \operatorname { V a r } _ { U \sim \operatorname { U n i f } [ 0 , 1 ] } [ T _ { k , \beta } ( U ) ]\tag{143}
$$

do not depend on the prompt, and GRPO applies the common normalization

$$
S _ { x } ^ { \mathrm { q } } ( r ) \longmapsto \frac { S _ { x } ^ { \mathrm { q } } ( r ) - \mu _ { k , \beta } } { \sigma _ { k , \beta } } .\tag{144}
$$

Thus, the prompt-adaptive structure of $S _ { x } ^ { \mathrm { q } }$ is preserved up to a common shift and scale.

As training proceeds, the current policy’s win-rate distribution can evolve differently across prompts, so the corresponding means and standard deviations become prompt dependent. GRPO normalization then becomes nontrivial and interacts with the KAPPO transformation, introducing GRPO’s usual prompt-dependent reweighting on top of the adaptive pressure supplied by KAPPO. Since S<sup>q</sup><sub>x</sub> is nonlinear, this normalization affects but does not cancel the @k transformation.

## H EXPERIMENTAL DETAILS

General setup. We perform full-parameter training in bfloat16 using AdamW with zero weight decay. We use a cosine learning-rate schedule with a ${ \bar { 3 } \% }$ warmup, a maximum prompt and response length of 4096 tokens, and sample training completions with temperature 1 and top-p = 1. Hyperparameters and intermediate checkpoints are selected using the corresponding validation sets. All KAPPO adaptations use $M = 3 0$ reference completions per prompt, generated once before training.

Mathematical reasoning. We train for at most 1000 steps. For $\ @ \mathrm { k } { - } A ^ { \star } \mathrm { P O }$ , we search learning rates in $\{ 5 { \times } 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 2 { \times } 1 0 ^ { - 5 } \}$ and $\beta$ in $\{ 2 . 5 { \times } 1 0 ^ { - 4 } , 5 { \times } 1 0 ^ { - 4 } , 1 0 ^ { - 3 } \}$ . For PKPO, we search learning rates in $\{ 1 0 ^ { - 6 } , 5 { \times } 1 0 ^ { - 6 } , \dot { 1 } 0 ^ { - 5 } \}$ and use the minimum rollout multiplicity $n = k .$ . GRPO uses $n = 8$ completions per prompt and learning rates in $\{ 5 { \times } 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 2 { \times } \dot { 1 0 } ^ { - 5 } \}$ . It uses standard group-relative advantage normalization without an additional KL penalty.

For standard DPO, we search learning rates in $\{ 5 { \times } 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 2 { \times } 1 0 ^ { - 5 } \}$ and $\beta$ in {0.01, 0.02, 0.05}. For @k-DPO, we use the same learning-rate grid and search the base coefficient used to construct $\beta _ { \mathrm { e f f } } ( x )$ in $\left\{ 1 0 ^ { - 4 } , 2 { \times } 1 0 ^ { - 4 } , 5 { \times } 1 0 ^ { - 4 } \right\}$ . Both variants construct a preference pair by sampling eight completions and selecting the highest- and lowest-reward candidates; ties receive zero loss weight. We use 64 prompts per DPO step, yielding 128 selected completions after pair construction, and disable log-probability length normalization. For @k-DPO, we add a failure pseudo-count of $\alpha = 0 . 5$ and cap $\beta _ { \mathrm { e f f } } ( x )$ at $ { \bar { 2 } } ;$ the precise boundary handling is described in Appendix J.2. The other mathematical-reasoning experiments use 128 prompts per training step.

Code generation. We train Qwen3-4B for at most 500 steps using 128 prompts per step. For win rate $\ @ \mathrm { k } { - } A ^ { \star } \mathrm { P O }$ , we search learning rates in $\{ 1 0 ^ { - 5 } , 2 { \times } 1 0 ^ { - 5 } \}$ and $\beta$ in $\{ 5 \times 1 { \dot { 0 } } ^ { - 4 } , 1 0 ^ { - 3 } , 5 \times 1 0 ^ { - 3 } \}$ For PKPO and GRPO, we search learning rates in $\{ 1 0 ^ { - 6 } , \overset { \vartriangle } { 5 } \times 1 0 ^ { - 6 } , 1 0 ^ { - 5 } \}$ . PKPO uses $n = k ,$ whereas GRPO uses $n = 8$ . During training, rewards are computed using up to 64 held-out test cases per problem; validation and final evaluation use the complete available test suite.

For win-rate @k-GRPO, we search learning rates in $\{ 5 { \times } 1 0 ^ { - 6 } , 1 0 ^ { - 5 } \}$ and the reward-transformation parameter $\beta$ in $\{ 5 { \times } 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 5 { \times } 1 0 ^ { - 3 } \}$ . As in standard GRPO, we use $n = 8$ completions per prompt and no $\dot { \mathrm { K L } }$ penalty.

## I COMPUTE RESOURCES

All training runs used 32 NVIDIA GH200 GPUs. Both mathematical reasoning and code-generation runs took approximately 10–12 hours, corresponding to 320–384 GPU-hours per run. The overall computational cost of the study, including hyperparameter sweeps, exploratory runs, and evaluation exceeded 100,000 GPU-hours.

## J REFERENCE CALIBRATION

KAPPO requires prompt-level statistics of the reference policy. We estimate them once before training from a fixed bank of M reference completions per prompt. Since the reference policy is fixed, the entire bank can be generated as an inference-only job using high-throughput engines such as vLLM or SGLang. Generation can be continuously batched and performed asynchronously, without backward passes, optimizer updates, parameter offloading, or repeated weight synchronization between the actor and generation workers. The resulting rewards are stored and reused across all training runs that share the same starting model and dataset.

For binary rewards, calibration estimates

$$
\widehat { p } _ { \mathrm { r e f } } ( \boldsymbol { x } ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbb { I } [ R ( y _ { j } ^ { \mathrm { r e f } } , \boldsymbol { x } ) = 1 ] .\tag{145}
$$

For each choice of k and $\beta ,$ , we then solve Eq. 6 for $p ^ { * } ( x )$ and compute $\beta _ { \mathrm { e f f } } ( x )$ and $Z ( x )$ (defined in Eq. 5). The @k-DPO instantiation uses $\beta _ { \mathrm { e f f } } ( x )$ , whereas binary $\ @ \bar { \mathrm { k } } { - } A ^ { \star } \mathrm { P O }$ additionally uses $Z ( x )$ and the reward multiplier $\beta / \beta _ { \mathrm { e f f } } ( x )$

For continuous rewards, the same bank defines the empirical reference CDF

$$
\widehat { F } _ { \mathrm { r e f } } ( r \mid x ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } \mathbb { I } [ R ( y _ { j } ^ { \mathrm { r e f } } , x ) \leq r ] .\tag{146}
$$

It is substituted into the explicit transformation $S _ { x } ^ { \mathrm { q } }$ in Eq. 12. The remaining quantities $C$ and Ψ depend only on $k$ and $\beta$ and are precomputed once. Both continuous $\ @ \mathrm { k } - \bar { A } ^ { \star } \bar { \mathrm { P O } }$ and @k-GRPO therefore reuse exactly the same calibrated reference rewards; changing $k , \beta ,$ the learning rate, or the random seed requires no additional reference generation.

We use $M = 3 0$ throughout our experiments. For MATH, this corresponds to $3 0 \times 7 , 5 0 0 = 2 2 5 , 0 0 0$ reference completions per model. Generating the complete bank took approximately 0.5 hours in our setup, whereas a single 1,000-step training run took approximately 10–12 hours. Thus, even if charged entirely to one run, calibration contributes only

$$
{ \frac { 0 . 5 } { 1 0 { - } 1 2 } } \approx 4 . 2 \% { - } 5 . 0 \%\tag{147}
$$

of its wall-clock time. More importantly, this is a one-time cost shared by every run using the same model and dataset. Amortized over $N$ runs, its relative contribution is only $( 4 . 2 \% - 5 . 0 \% ) / N$ ; over six runs, for example, it adds about five minutes per run, or less than $1 \%$ . The bank is typically reused across substantially more configurations, making the practical cost of reference calibration negligible relative to training.

## J.1 CALIBRATION ALGORITHMS

Algorithms 1 and 2 summarize reference calibration for binary and continuous win-rate rewards, respectively. Both start from the same reusable reference bank: for each training prompt $x \in \mathcal { X }$ draw $y _ { x , 1 } ^ { \mathrm { r e f } } , \ldots , y _ { x , M } ^ { \mathrm { r e f } } \stackrel { \mathrm { i . i . d . } } { \sim } \pi _ { \mathrm { r e f } } ( \cdot \mid x )$ and store their rewards $r _ { x , j } ^ { \mathrm { r e f } } = R ( y _ { x , j } ^ { \mathrm { r e f } } , x )$ . This sampling is performed only once per reference model and dataset. Changing k or $\beta$ requires only recomputing the numerical tables below, not generating additional completions.

Binary rewards. The calibration depends on each prompt only through its reference success count $N ( x )$ . We therefore precompute the required quantities for all $M + 1$ possible counts. Algorithm 1 includes the method-specific boundary conventions used in our experiments: no smoothing for $A ^ { \star } \mathrm { P O }$ , and a failure pseudo-count with a coefficient cap for DPO. We store $\ell ( x ) = \beta \log \widehat { Z } ( x )$ rather than $\widehat Z ( x )$

During training, binary $\ @ \mathrm { k } { - } A ^ { \star } \mathrm { P O }$ uses the regression target $\widehat { a } _ { k } ( x ) R ( y , x ) - \ell ( x )$ , while @k-DPO uses ${ \widehat \beta } _ { \mathrm { e f f } } ( x )$ in its preference loss. Neither requires further calibration during training. The fixedpoint solve and partition computation are implemented in numerically stable coordinates, as described in Appendix J.2.

Continuous win-rate rewards. An empirical CDF constructed from M reference rewards takes values only in $\{ 0 , 1 / M , \ldots , 1 \}$ . Consequently, the entire win-rate-based transformation can be precomputed on this grid. The resulting table is shared by all prompts and by both continuous $\ @ \mathrm { k } { - } A ^ { \star } \mathrm { P O }$ and @k-GRPO.

For an online completion with reward r, TRANSFORMREWARD returns $\widehat { S } _ { x } ^ { \mathrm { q } } ( r ) = T _ { M \widehat { F } _ { \mathrm { r e f } } ( r | x ) }$ using only comparisons with stored rewards and a table lookup. No numerical integration or root finding is performed during training. The inclusive comparison preserves our convention for ties and includes both empirical-CDF endpoints without smoothing.

Continuous $\ @ \mathrm { k } { - } A ^ { \star } \mathrm { P O }$ uses this value as its regression target, retaining $Z ( x ) ~ = ~ 1$ under the continuous-reference approximation discussed in Appendix F. @k-GRPO uses the same transformed rewards before its usual group normalization. The parameter $\beta$ determines the transformation in both cases; using it for calibration does not require adding a KL penalty to GRPO.

## J.2 BOUNDARY HANDLING

A finite reference bank can contain no successes or only successes even when the true success probability lies strictly between zero and one. The implementation therefore needs explicit conventions

Algorithm 1 Reference calibration for binary rewards   
Require: Reference rewards $\{ r _ { x , j } ^ { \mathrm { r e f } } \}$ , bank size M, training parameter $k \geq 1 , \beta > 0 .$ , method   
$m \in \{ A ^ { \star } \mathrm { P O } , \mathrm { D P O } \}$   
1: $\alpha \gets \bar { 0 }$ for $A ^ { \star } \mathrm { P O }$ , or α $ 0 . 5$ for DPO   
2: for $j = 0 , \ldots , M$ do   
3: $q _ { j }  j / ( M + \alpha )$   
4: if $q _ { j } = 0$ then   
5: $\overset { \vartriangle } { p _ { j } }  \boldsymbol { 0 } ; a _ { j }  k ; \ell _ { j }  \boldsymbol { 0 }$   
6: else if $q _ { j } = 1$ then   
7: $p _ { j } \gets 1$   
8: $a _ { j } \gets 1$ if $k = 1 ,$ , otherwise $a _ { j }  0$   
9: $\ell _ { j } \gets a _ { j }$   
10: else   
11: Solve Eq. 6 with $p _ { \mathrm { r e f } } = q _ { j }$ for $p _ { j }$   
12: $a _ { j }  k ( 1 - p _ { j } ) ^ { k - 1 }$   
13: $\ell _ { j } ^ { ' } \gets \beta \log ( 1 - q _ { j } + q _ { j } \exp ( a _ { j } / \beta ) )$   
14: end if   
15: if $m = \mathrm { D P O }$ then   
16: $b _ { j } \gets \operatorname* { m i n } \{ \beta / a _ { j } , 2 \}$   
17: end if   
18: end for   
19: for each prompt $x \in \mathcal { X }$ do   
20: $\begin{array} { r } { N ( x ) \gets \sum _ { j = 1 } ^ { M } \mathbb { I } [ r _ { x , j } ^ { \mathrm { r e f } } = 1 ] } \end{array}$   
21: $\widehat { p } ^ { * } ( x )  p _ { N ( x ) }$   
22: if $m = A ^ { \star } \mathrm { { P O } }$ then   
23: $\widehat { a } _ { k } ( x ) \gets a _ { N ( x ) } ; \ell ( x ) \gets \ell _ { N ( x ) }$   
24: else   
25: $\widehat { \beta } _ { \mathrm { e f f } } ( x ) \gets b _ { N ( x ) }$   
26: end if   
27: end for   
28: return $\{ { \widehat { a } } _ { k } ( x ) , \ell ( x ) \} _ { x \in \mathcal { X } }$ for $A ^ { \star } \mathrm { P O } .$ , or $\{ \widehat { \beta } _ { \mathrm { e f f } } ( x ) \} _ { x \in \mathcal { X } }$ for DPO

for these empirical endpoints, distinct from the population boundary cases in Appendix B. Let $N ( x )$ denote the number of successful completions among the M reference samples.

Binary $\scriptstyle ( \mathrm { a } ) \mathbf { k } = A ^ { \star } \mathbf { P } \mathbf { 0 }$ . We use the unsmoothed estimate $\widehat { p } _ { \mathrm { r e f } } ( x ) \ = \ N ( x ) / M$ . Writing $a _ { k } ( x ) =$ $\beta / \beta _ { \mathrm { e f f } } ( x )$ , the endpoint values for $k > 1$ are

$$
\begin{array} { r } { \left( \widehat { p } ^ { * } ( x ) , \widehat { a } _ { k } ( x ) , \beta \log \widehat { Z } ( x ) \right) = \left\{ \begin{array} { l l } { ( 0 , k , 0 ) , } & { N ( x ) = 0 , } \\ { ( 1 , 0 , 0 ) , } & { N ( x ) = M . } \end{array} \right. } \end{array}\tag{148}
$$

We compute the regression target $\widehat { a } _ { k } ( x ) R ( y , x ) - \beta \log \widehat { Z } ( x )$ directly, avoiding an explicit infinite $\beta _ { \mathrm { e f f } } ( x )$ at $N ( x ) = M$ Thus, a zero-success bank gives target $k R ( y , x )$ , whereas an all-success bank gives target zero for $k > 1$ . For $k = 1$ , the multiplier remains one; the targets at $N ( x ) = 0$ and $N ( x ) = M$ are $R ( y , x )$ and $R ( y , x ) - 1$ , respectively. No probability smoothing is used for this method.

Binary @k-DPO. Unlike the regression target, the DPO loss requires a finite coefficient. We therefore add a failure pseudo-count and solve the fixed-point equation using $\widehat { p } _ { \mathrm { r e f } } ( x ) = N ( x ) / ( M +$ α) with $\alpha ~ = ~ 0 . 5$ This leaves the zero-success endpoint unchanged but moves the all-success estimate below one. The formula applies to every count, not only to all-success banks. We then use

$$
\widehat { \beta } _ { \mathrm { e f f } } ( x ) = \operatorname* { m i n } \left\{ \frac { \beta } { k ( 1 - \widehat { p } ^ { * } ( x ) ) ^ { k - 1 } } , 2 \right\} .\tag{149}
$$

Smoothing prevents the infinite coefficient at an empirical success probability of one, while the cap limits very large coefficients near this boundary.

Algorithm 2 Reference calibration for continuous win-rate rewards   
Require: Reference rewards $\{ r _ { x , j } ^ { \mathrm { r e f } } \}$ }, bank size M, training parameter $k \geq 1 , \beta > 0$   
1: if k = 1 then   
2: log $C \gets - \log \beta - \beta ^ { - 1 } - \log ( 1 - e ^ { - 1 / \beta } )$   
3: for $j = 0 , \ldots , \bar { M }$ do   
4: $T _ { j } \gets j / M + \beta$ log C   
5: end for   
6: else   
7: Solve $\int _ { 0 } ^ { 1 } { \frac { d u } { C + u ^ { k } / \beta } } = 1$ for $C > 0$   
8: Define $\Psi ( z ) = \int _ { 0 } ^ { z } \frac { d u } { C + u ^ { k } / \beta }$   
9: $T _ { 0 } \gets \beta \log C ; \breve { T _ { M } } \gets \beta \log ( C + 1 / \beta )$   
10: for $j = 1 , \ldots , M - 1$ do   
11: Solve $\Psi ( z _ { j } ) = j / M$ for $z _ { j } \in ( 0 , 1 )$   
12: $T _ { j } \gets \beta \log ( C + z _ { j } ^ { k } / \beta )$   
13: end for   
14: end if   
15: return Stored reference rewards and the shared table $\{ T _ { j } \} _ { j = 0 } ^ { M }$   
16: function TRANSFORMREWARD(x, r)   
17: $j \gets \sum _ { i = 1 } ^ { M } \mathbb { I } [ r _ { x , i } ^ { \mathrm { r e f } } \leq r ]$   
18: return $T _ { j }$   
19: end function

Win-rate endpoints and ties. For continuous $\ @ \mathrm { k } { - } A ^ { \star } \mathrm { P O }$ and @k-GRPO, we use the inclusive empirical CDF without smoothing or clipping its values away from zero and one. The transformation remains finite at both endpoints:

$$
S _ { x } ^ { \mathrm { q } } ( r ) = \left\{ \begin{array} { l l } { \beta \log C , } & { \widehat { F } _ { \mathrm { r e f } } ( r \mid x ) = 0 , } \\ { \beta \log ( C + 1 / \beta ) , } & { \widehat { F } _ { \mathrm { r e f } } ( r \mid x ) = 1 . } \end{array} \right.\tag{150}
$$

Rewards below the smallest reference reward receive the first value; rewards at or above the largest receive the second. Ties retain the inclusive-CDF convention, including for a constant-reward reference bank. For $A ^ { \star } \mathrm { P O }$ , we retain $Z ( x ) = 1$ under the continuous-reference approximation discussed in Appendix F.

Numerical stability. For interior binary estimates, we solve the scalar equation in log-odds coordinates and compute $a _ { k }$ directly from that solution, avoiding loss of precision in $1 - p ^ { * }$ near one. Partition terms and win-rate transformations are evaluated in the log domain, with a separate closedform computation for $k = 1$ , to avoid overflow at small $\beta .$ . These numerical evaluations do not introduce additional probability smoothing.

Degenerate online groups. In @k-GRPO, we divide centered transformed rewards by their group standard deviation plus $1 \bar { 0 } ^ { - 6 }$ ; constant-reward groups therefore have zero advantages. For both DPO variants, selected pairs with equal rewards receive zero loss weight.

## K SENSITIVITY TO FINITE REFERENCE CALIBRATION

Reference calibration computes the quantities and mappings required by KAPPO before training. For binary rewards, we first estimate $p _ { \mathrm { r e f } } ( x )$ from a fixed bank of reference completions, then solve Eq. 6 for $p ^ { * } ( x )$ and compute the adaptive reward multiplier $a _ { k } ( x )$ or KL coefficient $\beta _ { \mathrm { e f f } } ( x )$ . For continuous rewards, we estimate the reference CDF $\bar { F _ { \mathrm { r e f } } } \bar { ( r \mid x ) }$ and substitute it into the win-ratebased transformation $S _ { x } ^ { \mathrm { q } } ( r )$ . This section analyzes how finite-sample noise in the estimates of $p _ { \mathrm { r e f } }$ and Fref $F _ { \mathrm { r e f } }$ propagates to these final quantities. We distinguish noise, measured by the standard deviation across independently drawn reference banks, from bias, measured by the difference between the mean estimate and the exact value obtained from the true reference distribution. The figures use our experimental bank size $M = 3 0$ and an illustrative coefficient $\beta = 0 . 1$

![](images/3331a594165e4b3a909a2d9014181fa088bf2a8cf350b1e7689bc8fb73c509f7.jpg)  
Figure 5: Exact optimal success probability as a function of the true reference success probability, for $\beta = 0 . 1$ . Markers show the attainable empirical estimates $j / M$ for $M = 3 0 ;$ ; the dotted line marks the first positive estimate $1 / M .$ The map is continuous, but its steep rise near zero is compressed on the linear horizontal scale.

Exact finite-sample calculation. Fix a prompt and let q denote the true reference quantity being estimated: $q = p _ { \mathrm { r e f } } ( x )$ for binary rewards, or $q = F _ { \mathrm { r e f } } ( r \mid x )$ at a fixed reward threshold $r$ for continuous rewards. Each reference completion then produces a binary observation—success in the first case and the event $R ( y , x ) \leq r$ in the second. Hence, the number N of positive observations in a bank of M i.i.d. reference completions satisfies $N \sim$ Binomial $. ( M , q )$ , and the empirical estimate is $\widehat { q } = N / M$

Let $\theta ( q )$ denote any exact downstream quantity computed from $q ,$ such as $p ^ { * } , a _ { k } , \beta _ { \mathrm { e f f } } , \mathrm { o r } S _ { x } ^ { \mathrm { q } } ( r )$ . Let $\widehat { \theta } _ { j }$ be the value produced by the implemented calibration rule when $N = j$ , including any smoothing or clipping. Since $\textstyle \operatorname* { P r } ( N = j ) = w _ { j } ( q ) : = { \binom { M } { j } } q ^ { j } ( 1 - q ) ^ { M - j }$ , its exact finite-bank mean, noise, and bias are

$$
\mu _ { \theta } ( q ) = \sum _ { j = 0 } ^ { M } w _ { j } ( q ) \widehat { \theta } _ { j } , \qquad \sigma _ { \theta } ( q ) = \left( \sum _ { j = 0 } ^ { M } w _ { j } ( q ) [ \widehat { \theta } _ { j } - \mu _ { \theta } ( q ) ] ^ { 2 } \right) ^ { 1 / 2 } , \qquad b _ { \theta } ( q ) = \mu _ { \theta } ( q ) - \theta ( q ) .\tag{151}
$$

We report absolute noise and bias for $p ^ { * }$ and $a _ { k }$ . Because the scale of $\beta _ { \mathrm { e f f } }$ varies substantially, for this quantity we define relative noise and relative bias as $\sigma _ { \theta } ( q ) / \theta ( q )$ and $b _ { \theta } ( q ) / \theta ( q )$ , respectively, using the exact coefficient in the denominator. Equation 151 sums over all $M + 1$ possible reference-bank outcomes, without Monte Carlo simulation or a local approximation. The reported errors account for finite-bank estimation and, where applicable, smoothing and clipping; numerical solver error is not included in this analysis.

## K.1 BINARY REWARDS

Suppressing the prompt argument, set $q = p _ { \mathrm { r e f } } ( x )$ and let $g ( q )$ be the optimal success probability determined by Eq. 6. The exact maps applied during reference calibration are

$$
p ^ { * } = g ( q ) , \qquad a _ { k } ( q ) = k [ 1 - g ( q ) ] ^ { k - 1 } , \qquad \beta _ { \mathrm { e f f } } ( q ) = \frac { \beta } { a _ { k } ( q ) } ,\tag{152}
$$

where $a _ { k }$ is the adaptive reward multiplier. For $k > 1$ , the boundary values are $g ( 0 ) = 0 , g ( 1 ) = 1$ $a _ { k } ( 0 ) = k$ , and $a _ { k } ( 1 ) = 0$ . For $k = 1$ , the multiplier is identically one.

Sensitive regions. Figure 5 shows that $g$ rises sharply near zero and changes more gradually in the interior. A finite bank may therefore fail to resolve a narrow region over which the inferred $p ^ { * }$ changes substantially. For example, at $k = 4 , g ( 0 ) = 0$ whereas $g ( 1 / 3 0 ) \approx 0 . 5 5 3$ , so observing one success instead of none produces a large change in ${ \widehat { p } } ^ { * }$ . The probability of observing no successes is $( 1 - q ) ^ { M }$ , which for $M = 3 0$ is approximately 0.740 when $q = 0 . 0 1$ and 0.042 when $q = 0 . 1$

![](images/e7148911e2eaab94335955572fb74ba727bd420ef2a8d1e177d132c1ccdc7b12.jpg)

![](images/1d323dc1ac82ad522647c82354aa6c05313c9fb0667eb35baef67911eed086a7.jpg)

(a) Optimal success probability $p ^ { * }$  
![](images/35b41535da8f5edf062dd58da236b8ecf72071a8f49b9f7ce59b89edff3e1477.jpg)

![](images/7a03fe0891748f50e47a6af00511688809e8ffb50ff3cc2c7e6b72a75b5453a2.jpg)

(b) Adaptive reward multiplier $a _ { k } .$  
![](images/af1cdd2535923eb05492b2f4575c5add6d1c146d55850252c2f9404bfb9f4f0a.jpg)

![](images/880f970677cb64e78662b96896c050dcb85db3ea81aabcf8205bde8002eec201.jpg)  
(c) Adaptive KL coefficient $\beta _ { \mathrm { e f f } }$ (relative errors).  
Figure 6: Exact finite-bank noise (left) and signed bias (right) for $M = 3 0$ and $\beta \ : = \ : 0 . 1$ , plotted against the true reference success probability. Panels (a) and (b) use $\widehat { q } \ = \ N / M ;$ panel (c) uses Eq. 153. Coefficient errors are divided by the exact $\beta _ { \mathrm { e f f } } ( q ) $ ; the endpoint $q = 1$ , where this coefficient diverges for $k > 1$ , is omitted. Noise uses logarithmic axes, while signed bias uses symmetric-logarithmic axes that are linear within $[ - 1 0 ^ { - 6 } , \bar { 1 } 0 ^ { - 6 } ]$ . Zero noise at deterministic endpoints is omitted from the logarithmic axes.

Estimators used in the analysis. For $p ^ { * }$ and $a _ { k }$ , we use the unsmoothed estimate $\widehat { q } \ = \ N / M$ employed in our binary $\ @ \mathrm { k } { - } A ^ { \star } \mathrm { P O }$ runs, giving ${ \widehat { p } } ^ { * } = g ( { \widehat { q } } )$ and $\widehat { a } _ { k } = a _ { k } ( \widehat { q } )$ . Thus, each horizontal axis value in Figure 6 fixes the true $p _ { \mathrm { r e f } }$ , while the plotted noise describes how the inferred quantities vary across independently drawn reference banks. For $\beta _ { \mathrm { e f f } }$ , we use the one-sided smoothing and cap employed by @k-DPO:

$$
{ \widehat { q } } _ { \mathrm { D } } = { \frac { N } { M + 0 . 5 } } , \qquad { \widehat { \beta } } _ { \mathrm { e f f } } = \operatorname * { m i n } \left\{ { \frac { \beta } { a _ { k } ( { \widehat { q } } _ { \mathrm { D } } ) } } , 2 \right\} .\tag{153}
$$

The smoothing keeps $\widehat { q } _ { \mathrm { D } } < 1$ while preserving $\widehat { q } _ { \mathrm { { D } } } = 0$ for a zero-success bank. Without smoothing or clipping, $N = M$ would produce an infinite coefficient for $k > 1$ , making its finite-bank moments undefined. We measure bias relative to the exact, unclipped $\beta _ { \mathrm { e f f } } ( q )$ , so the reported bias includes the effect of these implementation safeguards.

## Consistency and convergence rates.

Lemma K.1 (Consistency of binary calibration). Fix an integer $k \geq 1 , \beta > 0 ;$ , and $q \in [ 0 , 1 ]$ Under i.i.d. reference sampling, the unsmoothed estimates $\widehat { p } ^ { * } = g ( \widehat { q } )$ and $\widehat { \boldsymbol { a } } _ { k } = \boldsymbol { a } _ { k } ( \widehat { \boldsymbol { q } } )$ are strongly consistent. Forfixed $q \in ( 0 , 1 )$ and $\theta \in \{ g , a _ { k } \}$ , their noise and bias satisfy

$$
\sigma _ { \theta } ( q ) = | \theta ^ { \prime } ( q ) | \sqrt { \frac { q ( 1 - q ) } { M } } + o ( M ^ { - 1 / 2 } ) , \qquad b _ { \theta } ( q ) = \frac { q ( 1 - q ) } { 2 M } \theta ^ { \prime \prime } ( q ) + o ( M ^ { - 1 } ) .\tag{154}
$$

At q = 0 and $q = 1$ , both estimates are exact for every M.

Proof. By the strong law of large numbers, ${ \widehat { q } } \to q$ almost surely. The fixed-point characterization in Eq. 6 implies that $g$ is continuous on $[ 0 , 1 ]$ , including the limits $g ( q ) \to 0$ as $q \downarrow 0$ and $g ( q ) \to 1$ as $q \uparrow 1$ . Thus, $a _ { k }$ is also continuous, and applying either map to $\widehat { q }$ gives almost-sure convergence to its exact value.

For interior $q ,$ smoothness of $g$ follows from the implicit function theorem. Indeed, the derivative of the right-hand side minus the left-hand side of Eq. $^ { 6 , }$ with respect to the candidate success probability $p ,$ is

$$
\frac { \beta } { p ( 1 - p ) } + k ( k - 1 ) ( 1 - p ) ^ { k - 2 } > 0 .
$$

Consequently, both $g$ and $a _ { k }$ are smooth on (0, 1).

For either map θ, a Taylor expansion around the fixed interior point $q ,$ together with

$$
\mathbb { E } [ \widehat { q } - q ] = 0 , \qquad \mathbb { E } [ ( \widehat { q } - q ) ^ { 2 } ] = \frac { q ( 1 - q ) } { M } ,
$$

gives

$$
\mathbb { E } [ \theta ( \widehat { q } ) ] - \theta ( q ) = \frac { \theta ^ { \prime \prime } ( q ) q ( 1 - q ) } { 2 M } + o ( M ^ { - 1 } ) , \qquad \mathrm { V a r } ( \theta ( \widehat { q } ) ) = \frac { \theta ^ { \prime } ( q ) ^ { 2 } q ( 1 - q ) } { M } + o ( M ^ { - 1 } ) .
$$

These expansions remain valid despite possible boundary behavior of the derivatives: outside a fixed interior neighborhood of $q ,$ binomial deviations have exponentially small probability, while g and $a _ { k }$ are bounded. Taking the square root of the variance gives Eq. 154. Finally, when $q \in \{ 0 , 1 \}$ ${ \widehat { q } } = q$ almost surely for every ${ \dot { M } } ,$ so both errors vanish. □

Thus, sampling noise is $O ( M ^ { - 1 / 2 } )$ and bias is $O ( M ^ { - 1 } )$ at each fixed interior reference success probability. The asymptotic constants depend on the local slope and curvature of the calibration maps, which can be large near sensitive regions. Consequently, these rates describe the large-bank limit but need not accurately predict the errors at $M = { \bar { 3 } } 0 ;$ ; the figures instead use the exact finitebank calculation in Eq. 151.

## Effect of coefficient stabilization.

Lemma K.2 (Consistency under smoothing and clipping). Fix an integer $k \ge 1 , \beta > 0$ , and $q \in ( 0 , 1 )$ . The estimator in Eq. 153 satisfies

$$
\widehat { \beta } _ { \mathrm { e f f } } \ \frac { \mathrm { a . s . } } { M  \infty } \operatorname* { m i n } \{ \beta _ { \mathrm { e f f } } ( q ) , 2 \} .
$$

$I f \beta _ { \mathrm { e f f } } ( q ) < 2 ,$ , it is consistentfor the exact coefficient, with $O ( M ^ { - 1 / 2 } )$ noise and $O ( M ^ { - 1 } )$ bias. If $\beta _ { \mathrm { e f f } } ( q ) > 2$ , its noise vanishes and its bias relative to the exact coefficient converges to $2 - \beta _ { \mathrm { e f f } } ( q ) ;$ the probability of obtaining an estimate below the cap decays exponentially in M. At an interior clipping threshold $\beta _ { \mathrm { e f f } } ( q ) \stackrel { - } { = } 2$ with $k > 1$ , the estimator remains consistent, but its downward bias is oforder $M ^ { - 1 / 2 }$

Proof. The smoothed estimate can be written as

$$
\widehat { q } _ { \mathrm { D } } = \frac { M } { M + 0 . 5 } \widehat { q } .
$$

It therefore converges almost surely to $q ,$ and its bias is

$$
\mathbb { E } [ \widehat { q } _ { \mathrm { D } } ] - q = - \frac { q } { 2 M + 1 } = O ( M ^ { - 1 } ) .
$$

Continuity of the capped coefficient at every interior q gives the stated almost-sure limit. Since the estimator is bounded by two, its mean and variance also converge to those of this deterministic limit.

When $\beta _ { \mathrm { e f f } } ( q ) \ < \ 2$ , the cap is inactive throughout a neighborhood of $q .$ In this neighborhood, the coefficient is a smooth function of $\widehat { q } _ { \mathrm { { D } } }$ , so the same Taylor argument as in Lemma K.1 gives $O ( M ^ { - 1 / 2 } )$ noise and $O ( M ^ { - 1 } )$ bias. The smoothing changes the leading bias coefficient but not its order. Excursions outside this neighborhood have exponentially small probability and do not change these rates.

When $\beta _ { \mathrm { e f f } } ( q ) > 2$ , the estimator equals two throughout a neighborhood of $q .$ Obtaining a value below two therefore requires a deviation of $\widehat { q } _ { \mathrm { D } }$ from q by a fixed positive amount, whose probability decays exponentially by binomial concentration.

Finally, suppose that $k > 1$ and $\beta _ { \mathrm { e f f } } ( q ) = 2$ . The coefficient is smooth and strictly increasing at this interior point. The central limit theorem and the local form of the clipping operation give

$$
\begin{array} { r } { \sqrt { M } \big ( \widehat { \beta } _ { \mathrm { e f f } } - 2 \big ) \stackrel { \mathrm { d } } {  } \beta _ { \mathrm { e f f } } ^ { \prime } ( q ) \operatorname* { m i n } \{ Z , 0 \} , \qquad Z \sim \mathcal { N } ( 0 , q ( 1 - q ) ) . } \end{array}
$$

Binomial moment bounds and local smoothness also justify convergence of the first moments. Hence,

$$
\mathbb { E } [ \widehat { \beta } _ { \mathrm { e f } } ] - 2 = - \beta _ { \mathrm { e f f } } ^ { \prime } ( q ) \sqrt { \frac { q ( 1 - q ) } { 2 \pi M } } + o ( M ^ { - 1 / 2 } ) ,
$$

establishing the downward bias of order $M ^ { - 1 / 2 }$

The failure pseudo-count thus vanishes asymptotically, whereas the fixed coefficient cap does not. Dividing the noise and bias by the fixed, finite $\beta _ { \mathrm { e f f } } ( q )$ preserves the corresponding rates for the relative errors.

Implications. For $p ^ { * }$ and $a _ { k }$ , finite-bank error is concentrated near the low-success boundary and becomes much smaller in the interior. $\mathrm { A t } k = 4$ and $p _ { \mathrm { r e f } } = 0 . 1 , \widehat { p } ^ { \ast }$ has standard deviation 0.123 and bias $- 0 . 0 2 8 ;$ at $p _ { \mathrm { r e f } } = 0 . 5$ , these decrease to 0.026 and 0.001. The same boundary events affect the adaptive multiplier in the opposite direction: a bank containing no successes yields $\widehat { p } ^ { * } = 0$ and hence the maximal multiplier $\widehat { a } _ { k } = k ,$ , producing an upward bias in the pressure assigned to prompts with small but nonzero success probability. As $M  \infty .$ , both noise and bias vanish, and the probability $( 1 - p _ { \mathrm { r e f } } ) ^ { M }$ of unresolved zero-success banks decays exponentially. This improvement need not be monotone in $M \colon$ near the low-success boundary, larger banks can initially increase variability as successful completions begin to appear. Additional reference completions are therefore most valuable for hard prompts for which $M p _ { \mathrm { r e f } }$ is small. At the exact boundary $p _ { \mathrm { r e f } } = 0 ,$ there is no estimation error: every bank contains zero successes and the maximal multiplier is the correct value.

For $k > 1$ , the effective KL coefficient has a different limitation near $p _ { \mathrm { r e f } } = 1$ . Because the exact $\beta _ { \mathrm { e f f } }$ diverges while the implemented estimator is capped, its sampling noise can be small simply because nearly all estimates have saturated at the cap, even though its downward bias relative to the exact coefficient is large. Increasing M reduces finite-bank noise asymptotically but cannot remove this cap-induced bias; doing so requires changing the cap or the stabilization rule. Thus, larger reference banks address the main uncertainty on low-success prompts, whereas accuracy on extremely easy prompts is ultimately limited by the chosen coefficient stabilization.

![](images/207f4a42683ed956c06675e883b8ca73148e8624a0c67fd25d93117097f1bc30.jpg)  
(a) Win-rate-based reward transformation.

![](images/1e121ebdc2a19b70c8e44e6e7d581aaeede60d39af91ad4ab316807b03889f53.jpg)

![](images/c8728ea45f262bf54ec20dc6997a9843de503cae375d616c972be6ce8f6cef0b.jpg)  
(b) Noise in the transformed reward.  
(c) Bias in the transformed reward.  
Figure 7: Win-rate transformation and exact finite-bank noise and bias for $M = 3 0$ and $\beta = 0 . 1$ To compare different $k ,$ values are normalized by $\Delta _ { T } = T ( 1 ) - T ( 0 )$ ; this normalization is only for display, and the transformation panel additionally subtracts $T ( 0 )$ . The horizontal axis is the true reference quantile of a fixed raw reward. Noise and bias use the same axis conventions as Figure $^ { 6 . }$

## K.2 CONTINUOUS REWARDS

For continuous rewards, reference calibration estimates the CDF value at each reward threshold and substitutes it into the win-rate-based transformation. At a fixed prompt and raw reward $r ,$ write Eq. 12 as

$$
S _ { x } ^ { \mathrm { q } } ( r ) = T ( q ) , \qquad q = F _ { \mathrm { r e f } } ( r \mid x ) , \qquad T ( q ) = \beta \log \left( C + \frac { G ( q ) ^ { k } } { \beta } \right) , \quad G = \Psi ^ { - 1 } .\tag{155}
$$

Here, N counts the reference completions whose rewards do not exceed $r ,$ so $N \sim$ Binomial $( M , q )$ and $\widehat { q } = N / M$ . The estimated transformed reward is therefore $T ( \widehat { q } )$ , and its exact noise and bias follow from Eq. 151 by setting $\theta ( q ) = T ( q )$ and $\widehat { \theta } _ { j } = T ( j / M )$ . This pointwise calculation requires no model of the full raw-reward distribution. To compare different $\begin{array} { r } { \dot { k } , } \end{array}$ we divide the plotted noise and bias by the transformation range $\Delta _ { T } = T ( 1 ) - T ( 0 )$ .

## Sensitivity and convergence guarantees.

Lemma K.3 (Consistency and error bounds for win-rate calibration). Fix an integer $k \geq 1$ and $\beta > 0$ . The transformation T in $E q .$ . 155 is convex and k-Lipschitz. Under i.i.d. reference sampling, for every $q \in [ 0 , 1 ]$ and bank size $M ,$ , the estimator $T ( \widehat { q } )$ satisfies

$$
\sigma _ { T } ( q ) \le k \sqrt { \frac { q ( 1 - q ) } { M } } , \qquad 0 \le b _ { T } ( q ) \le \frac { k ( k - 1 ) ( C + \beta ^ { - 1 } ) q ( 1 - q ) } { 2 M } .\tag{156}
$$

At fixed interior $q ,$ the asymptotic expressions in $E q .$ . 154 apply with $\theta = T$ . At $q = 0$ and $q = 1$ both errors are zero.

For each fixed prompt, the estimated mapping $\widehat { S } _ { x } ^ { \mathrm { q } } ( r ) : = T ( \widehat { F } _ { \mathrm { r e f } } ( r \mid x ) )$ converges uniformly almost surely to $S _ { x } ^ { \mathrm { q } } ( r )$ . Moreover,for every $\delta \in \bar { ( 0 , 1 ) }$

$$
\operatorname* { P r } \left( \operatorname* { s u p } _ { r } \left| \widehat { S } _ { x } ^ { \mathrm { q } } ( r ) - S _ { x } ^ { \mathrm { q } } ( r ) \right| > k \sqrt { \frac { \log ( 2 / \delta ) } { 2 M } } \right) \leq \delta .\tag{157}
$$

These estimation guarantees hold for both continuous and discrete reference reward distributions.

Proof. Shape of the transformation. Using $G ^ { \prime } ( q ) = C + G ( q ) ^ { k } / \beta$ , differentiation gives, for $k > 1$

$$
T ^ { \prime } ( q ) = k G ( q ) ^ { k - 1 } , \qquad T ^ { \prime \prime } ( q ) = k ( k - 1 ) G ( q ) ^ { k - 2 } \left( C + \frac { G ( q ) ^ { k } } { \beta } \right) .\tag{158}
$$

Since $G ( q ) \in [ 0 , 1 ]$ , these derivatives satisfy

$$
0 \leq T ^ { \prime } ( q ) \leq k , \qquad 0 \leq T ^ { \prime \prime } ( q ) \leq k ( k - 1 ) ( C + \beta ^ { - 1 } ) .
$$

Thus, $T$ is convex and k-Lipschitz. For $k = 1$ , direct differentiation gives $T ^ { \prime } ( q ) = 1$ and $T ^ { \prime \prime } ( q ) = 0$ so the same conclusions hold.

Pointwise noise and bias. The Lipschitz property gives

$$
\operatorname { V a r } ( T ( { \widehat { q } } ) ) \leq \mathbb { E } \left[ ( T ( { \widehat { q } } ) - T ( q ) ) ^ { 2 } \right] \leq k ^ { 2 } \mathbb { E } [ ( { \widehat { q } } - q ) ^ { 2 } ] = { \frac { k ^ { 2 } q ( 1 - q ) } { M } } .
$$

Taking square roots proves the noise bound. Convexity and $\mathbb { E } [ \widehat { q } ] = q$ give $b _ { T } ( q ) \geq 0$ by Jensen’s inequality. A second-order Taylor expansion with the uniform bound on $T ^ { \prime \prime }$ gives

$$
T ( \widehat { q } ) \leq T ( q ) + T ^ { \prime } ( q ) ( \widehat { q } - q ) + \frac { k ( k - 1 ) ( C + \beta ^ { - 1 } ) } { 2 } ( \widehat { q } - q ) ^ { 2 } .
$$

Taking expectations proves the upper bound on bias. At fixed interior $q ,$ the Taylor argument from Lemma K.1 yields Eq. 154. At either endpoint, ${ \widehat { q } } = q$ almost surely, so both errors vanish.

Uniform consistency. The Lipschitz property also implies

$$
\operatorname* { s u p } _ { r } \left| \widehat S _ { x } ^ { \mathrm { q } } ( r ) - S _ { x } ^ { \mathrm { q } } ( r ) \right| \leq k \operatorname* { s u p } _ { r } \left| \widehat F _ { \mathrm { r e f } } ( r \mid x ) - F _ { \mathrm { r e f } } ( r \mid x ) \right| .
$$

Applying the Dvoretzky–Kiefer–Wolfowitz–Massart inequality (Massart, 1990) proves Eq. 157. More generally, for every $\varepsilon > 0 .$

$$
\operatorname* { P r } \left( \operatorname* { s u p } _ { r } \Big | \widehat { S } _ { x } ^ { \mathrm { q } } ( r ) - S _ { x } ^ { \mathrm { q } } ( r ) \Big | > \varepsilon \right) \leq 2 \exp \left( - \frac { 2 M \varepsilon ^ { 2 } } { k ^ { 2 } } \right) .
$$

The right-hand side is summable over $M$ , so the Borel–Cantelli lemma gives uniform almost-sure convergence. Neither the binomial calculation nor the empirical-CDF inequality requires a continuous reference distribution. □

Where quantile error matters. Equation 158 shows that sensitivity increases toward the upper reference quantiles. Convexity also explains the upward bias: although the empirical $\mathrm { C D F }$ is unbiased at each fixed threshold, the transformed reward generally is not. For $k = \bar { 1 }$ , T is affine, so the bias is zero and the standard deviation is exactly $\sqrt { q ( 1 - q ) / M }$

The finite-sample bounds establish $O ( M ^ { - 1 / 2 } )$ noise and $O ( M ^ { - 1 } )$ bias for fixed k and $\beta ,$ while Eq. 154 gives the leading constants at fixed interior quantiles. These guarantees concern estimation of the mapping $T ( \check { F _ { \mathrm { r e f } } } ( r \mid x ) )$ ; they do not require the reference win rates themselves to be uniformly distributed.

![](images/67cc934ea4069da98bf58d82b78bae31d2726064fc460957d85415da2e07258e.jpg)  
Starting checkpoint @2-A<sup>⋆</sup>PO PKPO, k<sub>train</sub> = 2

![](images/c5cf9cc2c04252208fbb8b0a1603d852b0a5522b443b7fec831677cd18ce6902.jpg)  
Starting checkpoint @4-A<sup>⋆</sup>PO PKPO, k<sub>train</sub> = 4  
(a) $k _ { \mathrm { t r a i n } } = 2 .$  
(b) $k _ { \mathrm { t r a i n } } = 4 .$  
Figure 8: AIME 2026 test performance of the starting checkpoints and the selected $\ @ k { - } A ^ { \star } \mathrm { P O }$ and PKPO checkpoints across evaluation sampling budgets. Each subfigure reports Qwen3-4B (left) and OLMo-3-7B (right).

Implications. The win-rate transformation has the opposite sensitive region: both noise and bias are small at low and intermediate quantiles but increase toward the upper tail, where $T$ is steepest, before vanishing at $q = 1 . { \mathrm { A t } } k = 4$ , the standard deviation and bias of the transformed reward are 2.62% and 0.66% of its range at $q = 0 . 5 ,$ but increase to 18.27% and 4.43% at $q = 0 . 9$ . The bias is positive because the convex transformation amplifies upward empirical-CDF errors more than downward ones. Near the upper boundary, the main finite-sample event is assigning the maximal empirical quantile to a reward whose true quantile is below one; for example, at $q = 0 . 9 9$ , this occurs with probability $\operatorname* { P r } ( \widehat { q } = 1 ) = q ^ { 3 0 } \approx \overset { } { 0 } . 7 4 0$

Both effects vanish as $M \to \infty$ , as $\widehat { q }$ concentrates around q and the probability $q ^ { M }$ of assigning a premature maximal quantile decreases. Additional reference completions are therefore most useful when accurate upper-tail ranks are important, particularly where $M ( 1 - q )$ is small. Since the same reference bank is reused throughout training, its error is a fixed distortion of the estimated transformation rather than fresh noise at every update. Overall, $M = 3 0$ provides substantially better resolution away from the sensitive boundaries, while larger banks can be targeted to hard binary prompts or to applications requiring more accurate upper-tail quantiles. For continuous rewards, this analysis isolates uncertainty from estimating the reference CDF; the separate approximation introduced by tied or discrete rewards is discussed in Appendix F.

## L COMPARISON WITH DIRECT MULTI-COMPLETION OPTIMIZATION

Checkpoint selection. For each model, method, and training value $k _ { \mathrm { t r a i n } } \in \{ 2 , 4 \}$ , we select the checkpoint that maximizes pass $\mathbb { Q } \mathrm { k } _ { \mathrm { t r a i n } }$ on the MATH Level 5 validation problems. Figure 3 reports the complete validation curves of the selected checkpoints, and Table 1 reports the corresponding numerical values.

At the evaluation budget matching the training objective, @k-A<sup>⋆</sup>PO is within 0.5 percentage points of PKPO in every comparison. The two methods also produce similar performance across the remaining evaluation budgets. This supports the central prediction of our analysis: adaptive pressure applied through an ordinary single-completion optimizer can reproduce direct multi-completion pass@k optimization.

Figure 8 extends this comparison to AIME 2026. Across both model families and training values of k, the two methods achieve comparable performance across evaluation budgets, showing that their agreement extends beyond the in-distribution validation set to harder, unseen problems.

Rollout-compute comparison. With a prompt batch of 128 and 1,000 training steps, @k-A<sup>⋆</sup>PO generates 128,000 on-policy completions per run. PKPO generates n completions for every prompt and therefore requires 128,000n completions. At its minimum rollout multiplicity $n = k ,$ , this gives $2 5 6 { , } 0 0 0$ completions for $k = 2$ and 512,000 for $k = 4$ . Thus, $\ @ \mathrm { k } { - } A ^ { \star } \mathrm { P O }$ reduces online rollout generation by 50% and 75%, respectively; larger PKPO groups increase this difference proportionally.

Table 1: MATH Level 5 validation performance (in %) of the starting checkpoints and selected @k-A<sup>⋆</sup>PO and PKPO checkpoints, corresponding to Figure 3.
<table><tr><td>Model</td><td>Method</td><td>pass@1</td><td>pass@2</td><td>pass@4</td><td>pass@8</td><td>pass@16</td></tr><tr><td>OLMo-3-7B</td><td>Starting checkpoint</td><td>37.38</td><td>50.94</td><td>63.16</td><td>73.29</td><td>81.80</td></tr><tr><td></td><td> $\ @ 2 \ – A ^ { \star } \mathrm { P O }$ </td><td>65.34</td><td>76.98</td><td>84.38</td><td>89.11</td><td>92.22</td></tr><tr><td></td><td>PKPO  $( k _ { \mathrm { t r a i n } } = 2 )$ </td><td>66.08</td><td>77.38</td><td>84.73</td><td>89.46</td><td>92.52</td></tr><tr><td></td><td> $\ @ 4 - A ^ { \star } \mathrm { P O }$ </td><td>64.46</td><td>76.51</td><td>84.39</td><td>89.33</td><td>92.45</td></tr><tr><td></td><td> $\mathrm { P K P O } \left( k _ { \mathrm { t r a i n } } = 4 \right)$ </td><td>64.31</td><td>76.18</td><td>83.92</td><td>89.04</td><td>92.45</td></tr><tr><td>Qwen3-4B</td><td>Starting checkpoint</td><td>23.93</td><td>38.32</td><td>54.32</td><td>68.16</td><td>77.79</td></tr><tr><td></td><td>@2-A*PO</td><td>57.27</td><td>70.48</td><td>79.39</td><td>85.34</td><td>89.43</td></tr><tr><td></td><td> $\mathrm { P K P O } \left( k _ { \mathrm { t r a i n } } = 2 \right)$ </td><td>59.88</td><td>70.77</td><td>78.77</td><td>84.59</td><td>88.75</td></tr><tr><td></td><td> $\ @ 4 - A ^ { \star } \mathrm { P O }$ </td><td>57.22</td><td>70.14</td><td>78.63</td><td>84.22</td><td>88.29</td></tr><tr><td></td><td> $\mathrm { P K P O } \left( k _ { \mathrm { t r a i n } } = 4 \right)$ </td><td>57.94</td><td>69.91</td><td>78.65</td><td>84.81</td><td>89.20</td></tr></table>

@k-A<sup>⋆</sup>PO additionally requires one-time reference calibration, whose implementation and cost are detailed in Appendix J. In our setup, generating the complete reference bank took approximately half an hour, compared with 10–12 hours for one training run, and the same bank is reused across all configurations of a given model and dataset. Even when amortized over only six runs, calibration therefore adds approximately five minutes per run, or less than 1% of training time. Its contribution is consequently small relative to the savings from removing PKPO’s k-dependent online rollout multiplicity.