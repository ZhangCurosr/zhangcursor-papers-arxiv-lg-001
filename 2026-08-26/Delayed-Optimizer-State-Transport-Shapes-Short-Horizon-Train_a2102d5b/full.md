# Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions

Jinhui Guo<sup>1,</sup> <sup>∗</sup>

<sup>1</sup>School of Physics, Beihang University, Beijing 100191, China

Adaptive optimizers retain gradient history in moment variables, allowing a local change in loss weighting to alter later updates. We examine whether this delayed transport is large enough to change prospective short-horizon decisions. On committed future-minibatch sequences, we diferentiate eight step AdamW trajectories through the complete model–optimizer state and select exposure-matched Math–Code loss schedules before independent evaluation. Across 12 unused 0.3M Transformer histories, full transport lowers token-disjoint loss relative to an optimizer-aware immediate derivative in 10/12 histories (mean benefit $4 . 7 1 \times 1 0 ^ { - 4 } ;$ ; exact one-sided sign test, $p = 0 . 0 1 9 3 )$ The two controllers act equally often but select diferent schedules in $6 0 / 9 6$ windows. Crossed checkpoint– future-path tests attribute this reordering to the interaction between optimizer state and near-future data, while an independent Ising–CNN experiment shows that deleting moment-state transport destroys accurate response prediction. Full-transport scores also concentrate exact-rollout winners in larger candidate libraries, focusing finite-amplitude evaluation on a shortlist. On these committed short paths, optimizer memory and near-future data order are therefore actionable components of the training state, providing a mechanism-based criterion for when finite-horizon rather than one-step intervention is required.

## I. INTRODUCTION

Adaptive optimizers endow neural network training with internal dynamical state and memory. A change in data or task weighting alters both the immediate parameter update and the optimizer moments that shape later updates [1, 2]. A local intervention can therefore propagate along the subsequent training trajectory: interventions that look similar immediately may rank differently under an objective evaluated several steps later. Choosing when to intervene is consequently a dynamical question rather than a purely local one.

Training-data composition is an important controllable component of modern learning systems. Datamixture and curriculum methods optimize global or timedependent domain exposure using proxy losses, current losses, rewards, or gradients [3–11]. Unrolled and bilevel diferentiation relate downstream objectives to earlier training choices [12, 13], while influence and trajectory methods quantify how examples or updates afect later predictions [14–17]. Domain order can produce noncommuting updates, with optimizer memory promoting ordering efects to leading order [18, 19]; optimizer-aware attribution can likewise difer from parameter-gradient attribution [20]. We resolve the response to a short intervention into injection, subsequent model–optimizer transport, and later readout, and quantify how transport changes the ranking of finite actions.

Nonequilibrium response theory provides useful language for this problem. In a driven system, a perturbation propagates through internal degrees of freedom before afecting a later observable [21, 22]; eliminating those degrees of freedom can introduce memory into the reduced dynamics [23, 24]. AdamW has the same structural feature: its moment bufers retain past-gradient information, leaving parameter-only dynamics with hidden-history dependence. Diferentiating the realized optimization trajectory traces the perturbation from its source to a later objective, without assuming equilibrium or fluctuation– dissipation.

Let $x _ { t }$ denote the augmented training state required for a locally Markovian update, including the model parameters and optimizer variables. For a control perturbation applied at step s, the linear response of an observable y<sub>t</sub> evaluated at a later step t is

$$
\chi _ { y } ( t , s ) = C _ { y , t } \Phi ( t , s + 1 ) B _ { s } ,\tag{1}
$$

where $B _ { s }$ injects the intervention into the training state, $\Phi ( t , s + 1 )$ is the ordered product of intervening update Jacobians, and $C _ { y , t }$ projects the propagated perturbation onto the observable. Equation (1) is the standard unrolled hypergradient in source–transport–readout form. Here $B _ { s }$ is set by a controlled domain-loss contrast, Φ transports parameter and optimizer perturbations, and $C _ { y , t }$ measures a later held-out objective. Deleting selected optimizer-coordinate tangents while keeping the nominal trajectory fixed tests how those coordinates contribute to transmission of the finite-time response.

We study this question in a paired two-domain training setup, using exposure-matched, eight-step Math–Code loss schedules in a byte-level Transformer. Both domain minibatches are evaluated at every step, fixing sample allocation so that interventions difer only in the temporal placement of their loss weights. Actions are selected on committed future-minibatch paths and locked before an independent token-disjoint readout. A crossed checkpoint– future-path experiment separates the current optimizer state from the subsequent training drive. An Ising–CNN benchmark tests the same transport mechanism under a diferent architecture, data-generating process, and scientific readout. For larger schedule libraries, finite-time scores rank candidates before exact finite-amplitude rollouts choose among a shortlist.

We find that delayed optimizer-state transport materially reorders finite interventions and improves held-out outcomes under prospectively locked decisions. Optimizer memory thereby becomes an actionable component of the training state, with consequences for finite schedule selection. For short committed paths, the results motivate a concrete design principle: candidate schedules should be evaluated using both the augmented model–optimizer state and the near-future data path, while transportbased scores can focus exact finite-amplitude evaluation on a small shortlist. Observing the same mechanism in a Transformer language-modeling system and an independently anchored Ising–CNN benchmark supports its relevance beyond a single architecture, although its range at larger scales and under unknown future paths remains to be established. Figure 1 summarizes the dynamical decomposition and experimental design.

## II. FINITE-TIME RESPONSE AND ACTIONS

This section connects a local loss-weight perturbation to a finite schedule decision. The full-state tangent defines the source–transport–readout response along a fixed future path; coordinate deletion then isolates transmission through optimizer moments. Contracting the resulting source-time derivatives with exposure-matched schedules gives the local scores used for finite action selection.

## A. Full optimizer-state response

Consider a fixed realization of the training path. Let $x _ { t } = ( \theta _ { t } , z _ { t } ^ { \mathrm { o p t } } )$ denote the augmented state needed to make a single optimizer step locally Markovian. For AdamW, $z _ { t } ^ { \mathrm { o p t } }$ contains the first and second moments and the optimizer clock. The update can be written as

$$
x _ { t + 1 } = F _ { t } ( x _ { t } , p _ { t } , \xi _ { t } ) ,\tag{2}
$$

where $p _ { t }$ controls the relative weight of two paired domains and $\xi _ { t }$ specifies the realized minibatches and any remaining stochasticity. Here $F _ { t }$ denotes the complete one-step AdamW update map, including the moment updates, bias correction, weight decay, clipping branch, and parameter update. Its explicit t-dependence accommodates the optimizer clock and any prescribed learning-rate schedule. With the minibatch pair fixed, the training loss is

$$
\ell _ { t } ( \theta , p ) = p \ell _ { A , t } ( \theta ) + ( 1 - p ) \ell _ { B , t } ( \theta ) ,\tag{3}
$$

Here θ denotes the model parameters, while $\ell _ { A , t }$ and $\ell _ { B , t }$ are the losses on the paired domain-A and domain-B minibatches presented at step t. The symbol $p$ is the scalar loss-weight argument, and $p _ { t }$ is its realized value at step t. Because both minibatches are evaluated at every step, $p _ { t }$ changes their relative loss weights without changing sample allocation.

Linearization about a nominal trajectory, within a fixed diferentiable branch of the update map, gives

$$
\delta x _ { t + 1 } = A _ { t } \delta x _ { t } + B _ { t } \delta p _ { t } , \qquad A _ { t } = \frac { \partial F _ { t } } { \partial x _ { t } } , \quad B _ { t } = \frac { \partial F _ { t } } { \partial p _ { t } } ,\tag{4}
$$

where $\delta \boldsymbol { x } _ { t }$ and $\delta p _ { t }$ denote first-order variations and all derivatives are evaluated on the nominal path. For a finite displacement, the omitted remainder is $O ( \parallel ( \delta x _ { t } , \delta p _ { t } ) \parallel ^ { 2 } )$ provided that the perturbation stays on the same smooth branch. For a scalar observable $y _ { t } = h _ { t } ( x _ { t } )$ , its readout covector is

$$
C _ { y , t } = \left( \nabla _ { x } h _ { t } ( x _ { t } ) \right) ^ { \mathsf { T } } .\tag{5}
$$

For $\delta x _ { t _ { 0 } } = 0$ , the response of the linearized system at time t is

$$
\begin{array} { l } { { \displaystyle \delta y _ { t } = \sum _ { s = t _ { 0 } } ^ { t - 1 } C _ { y , t } \Phi ( t , s + 1 ) B _ { s } \delta p _ { s } , } } \\ { { \displaystyle \Phi ( t , s + 1 ) = A _ { t - 1 } \cdot \cdot \cdot A _ { s + 1 } , } } \end{array}\tag{6}
$$

with $\Phi ( s + 1 , s + 1 ) = I$ . In this factorization, $B _ { s } \delta p _ { s }$ is the perturbation injected into the augmented state, $\Phi ( t , s + 1 )$ is its transport through the intervening updates, and $C _ { y , t }$ is the final projection onto the observable.

For a terminal objective ${ \mathcal { I } } _ { T } = { \mathcal { I } } _ { T } ( x _ { T } )$ that depends on earlier controls only through $x _ { T } ,$ , define

$$
C _ { T } = \left( \nabla _ { x } \mathcal { I } _ { T } ( x _ { T } ) \right) ^ { \mathsf { T } } .\tag{7}
$$

The derivative with respect to a control applied at step s is then

$$
\frac { \partial \mathcal { I } _ { T } } { \partial p _ { s } } = C _ { T } \Phi ( T , s + 1 ) B _ { s } .\tag{8}
$$

Equation (8) is the standard unrolled hypergradient in source–transport–readout form. The explicit AdamW tangent and its coordinate-resolved closure checks are given in Supps. A.1 and A.2, respectively.

The ordered product Φ depends on the source time and on the realized future path. Moving a signed reweighting within a control window changes the Jacobians that follow it and hence its terminal efect. Each $A _ { t }$ is set by the evolving model–optimizer state and the minibatch at that step, so a diferent ordering of future updates generally produces a diferent response. The exposure-matched schedules below probe this source-time dependence.

Let $L _ { A }$ and $L _ { B }$ be terminal losses evaluated on paired domain-A and domain-B readout data. Their antisymmetric and symmetric combinations are

$$
m = \frac { L _ { A } - L _ { B } } { 2 } , \qquad e = \frac { L _ { A } + L _ { B } } { 2 } .\tag{9}
$$

The readout m measures redistribution between the domains, and e measures the common-mode loss. The underlying state perturbation is the same; only the terminal projection difers.

![](images/9dcc7dd9f71955645905dca6c1c935473a57a7a532d123f0a2e9d733fa882d01.jpg)  
FIG. 1. Pathwise training response separates source, transport, and readout. A local loss-weight perturbation enters the augmented model–optimizer state through $B _ { s } ,$ propagates through the finite-time Jacobian product Φ, and is projected by the observable-specific readout $C _ { y , t }$ . The same decomposition supports the three tests used below: deleting moment-state tangent channels, comparing full transport with an optimizer-aware immediate response, and ranking finite temporal schedules before exact reranking in large candidate libraries. Actions and terminal states are locked before the independent readout.

## B. Optimizer memory as a transport channel

To further identify which state coordinates transmit the response, we selectively remove perturbations in the Adam moment coordinates while retaining the same nominal trajectory and future minibatches. The resulting memory-deleted tangent isolates their contribution to the transport factor Φ and provides a coordinate-level diagnostic of moment-mediated transmission in the Adam state representation.

The origin of this contribution is already visible in the first-moment equation. Let $r _ { s } = \nabla _ { \theta } \ell _ { s }$ be the parameter gradient, $q _ { s } ~ = ~ \partial r _ { s } / \partial p _ { s }$ its control derivative, and $\mu _ { s }$ Adam’s first moment. In AdamW, $\beta _ { 1 }$ and $\beta _ { 2 }$ are the exponential retention factors for the first and second gradient moments, respectively; values closer to unity produce slower decay and hence longer optimizer memory. Omitting Hessian feedback for this illustration, a control pulse at step s gives

$$
\begin{array} { r } { \delta \mu _ { s + k } \simeq ( 1 - \beta _ { 1 } ) \beta _ { 1 } ^ { k - 1 } q _ { s } \delta p _ { s } , \qquad k \geq 1 . } \end{array}\tag{10}
$$

The first moment therefore retains an exponentially decaying record of the pulse. With the moments and optimizer clock included in $x _ { t } ,$ the AdamW update is Markovian in the augmented state; a prescribed schedule may still enter $F _ { t }$ explicitly through t. Removing the moments from the state leaves history-dependent parameter dynamics.

Our AdamW configurations use $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } =$ 0.999. The corresponding discrete e-folding times,

$$
\tau _ { \beta } = - \frac { 1 } { \ln \beta } ,\tag{11}
$$

are approximately 9.5 and 999.5 steps. The eight-step horizon therefore samples a substantial fraction of the firstmoment relaxation while covering only a small fraction of the second-moment timescale. The present window is consequently more sensitive to first-moment transport than to second-moment relaxation.

## C. Finite-action scoring

Having identified how a perturbation is transported through the augmented state, we now use the source-time derivatives to compare finite temporal schedules. Each candidate redistributes a fixed total domain weight across several source times. An H-step intervention is written as

$$
u = ( u _ { 0 } , \ldots , u _ { H - 1 } ) , \qquad p _ { t + j } = p _ { 0 } + a u _ { j } ,\tag{12}
$$

where $p _ { 0 }$ is the baseline domain weight and a is the intervention amplitude. The neutral action is $u = 0$ . In the primary Math–Code library, the nonneutral schedules satisfy

$$
\sum _ { j = 0 } ^ { H - 1 } u _ { j } = 0 ,\tag{13}
$$

and consequently have the same aggregate domain weight over the window. They difer only in the temporal placement of the reweighting.

For a window beginning at t, the full source-time derivative and the optimizer-aware immediate derivative are

$$
\begin{array} { r l } & { g _ { t + j } ^ { \mathrm { f u l l } } = C _ { t + H } \Phi ( t + H , t + j + 1 ) B _ { t + j } , } \\ & { g _ { t + j } ^ { \mathrm { m y } } = C _ { t + j + 1 } B _ { t + j } . } \end{array}\tag{14}
$$

The coeficient $g _ { t + j } ^ { \mathrm { f u l l } }$ is the single-source term of Eq. (6), evaluated with source time $s = t + j$ and terminal time $t + H ;$ ; equivalently, $g _ { t + j } ^ { \mathrm { f u l l } } = \partial \mathcal { T } _ { t + H } / \partial p _ { t + j }$ . It contains no summation because it describes the terminal response to a perturbation at one source time. The sum appears when the H source-time contributions are contracted with a complete schedule in Eq. (15). The immediate coeficient is the one-step case, for which $\Phi ( t + j + 1 , t + j + 1 ) =$ = I. Here $C _ { t + j + 1 }$ is the gradient of the same controller objective evaluated at the next state. The two derivatives share the source term from the current AdamW update. The immediate derivative is read out at that next state, whereas full transport carries the same source through the remaining specified updates and uses the common terminal objective. Their comparison isolates the efect of the remaining training window on the decision obtained from the current-step response.

Following $\operatorname { E q . } \ ( 8 )$ , the corresponding first-order terminal score can be expressed as

$$
\delta \mathcal { T } _ { t + H } ( u ) = a \sum _ { j = 0 } ^ { H - 1 } g _ { t + j } ^ { \mathrm { f u l l } } u _ { j } .\tag{15}
$$

Equal-exposure schedules can therefore receive diferent scores when the response varies across source times.

The executed schedules have finite amplitude, and their tangent scores are used as local decision statistics. Every candidate is evaluated on the same committed future minibatches. A clipping-branch screen removes comparisons that leave the sampled smooth branch of the piecewisesmooth update [25]; its finite-amplitude motivation and sampled branch test are detailed in Supp. D.1. In the primary seven-action experiment, the tangent-selected branch-compatible schedule is executed directly. For larger libraries, one reverse adjoint supplies the sourcetime derivatives, the resulting scores define a shortlist, and exact finite-amplitude rollouts select an action within that shortlist, as specified in Supp. D.2.

## III. EXPERIMENTAL DESIGN

The response construction above yields a full-transport score, a memory-deleted diagnostic, and an optimizeraware immediate comparator for exposure-matched schedules. We test these quantities prospectively in two systems, the Math–Code Transformer and the Ising–CNN model. In each case, the future minibatch path is fixed before scoring, and actions are locked before independent evaluation.

## A. Math–Code Transformer

The primary experiments use a byte-level causal Transformer [26] trained on paired Math and Code minibatches. Math examples come from 56 medium-entropy modules of the DeepMind Mathematics Dataset [27]; Code examples are complete files from the Python standard library [28]. Training and validation files are disjoint, and the two domains have matched byte-token budgets; corpus construction and split definitions are given in Supp. B.1. The primary model has 296,504 parameters, sequence length 64, and batch size 8, and is trained with AdamW [2]. A 1,055,648-parameter model is used for the scale check. Normalized-age checkpoint matching and the tested duration boundary are reported in Supp. B.1 and Fig. S2.

Control is applied in windows of $H = 8$ updates. Six zero-sum schedules with amplitude $a = 0 . 0 2$ compete with the neutral action, using the symmetric loss e as the objective. Nonoverlapping token positions divide the validation corpus into controller, audit, and test thirds, separating action selection from the reported terminal efect. After each eight-step block, a new action is computed from the resulting checkpoint, giving blockwise receding-horizon replanning [29].

Confirmation uses 12 held-out 0.3M training histories that played no role in method development. Full transport is compared with neutral training, validation-gradient alignment (VGA), and the optimizer-aware immediate derivative in Eq. (14). The immediate derivative includes the moment update, bias correction, weight decay, and sampled clipping branch of the current AdamW step, and ends before subsequent state propagation. Checkpoints, candidate schedules, amplitudes, future minibatches, branch screens, and readout partitions are identical across arms; the complete four-arm protocol is specified in Supp. B.1. Actions and terminal states are locked before token-disjoint evaluation.

## B. Ising–CNN benchmark

The second system is a 252,642-parameter GroupNorm CNN [30] trained on configurations of the two-dimensional Ising model generated by Wolf-cluster Monte Carlo [31]. Relative to Math–Code, this benchmark changes the architecture, data-generating process, and scientific readout while retaining the same optimizer-state perturbation. Neural classifiers and crossing constructions have previously been used to diagnose phases and transitions [32– 34].

The controlled neural observable, $\widehat { T } _ { N }$ , is the zerocrossing temperature of a linear fit to the mean logit diference. If $\bar { z } _ { \theta } ^ { \mathrm { f i t } } ( T ) = \alpha + \beta T$ , then $\widehat { T } _ { N } = - \alpha / \beta$ , the temperature at which the fitted mean logit contrast between the two phase labels vanishes. Binder-cumulant crossings over several lattice sizes define an independent finite-size anchor $T _ { B }$ , which is locked before neural action selection [35, 36]. The exact infinite-lattice critical temperature obtained from Kramers–Wannier duality and the Onsager solution is reserved for post-lock evaluation [37, 38]. During control, $T _ { B }$ supplies the reference physical measurement and $\widehat { T } _ { N }$ the diferentiable training readout. Details of the crossing fit, Binder construction, resampling, and model averaging are given in Supp. B.2.

And then three analyses are performed in the Ising system: an optimizer-memory ablation, a prospective control test against the locked Binder anchor, and a crossed checkpoint–future-path audit. In the crossed design, a future tape is a pre-generated sequence of eight paired minibatches; varying the checkpoint and tape independently separates state, future-drive, and interaction contributions to schedule preference.

## C. Prospective protocol and statistical inference

The two systems serve complementary roles. Math– Code supplies the primary decision-level confirmation and scale check, while Ising tests the same source–transport– readout mechanism with a diferent architecture, data source, and physically anchored readout.

Development and confirmation histories remain separate throughout. Predictions and branch screens are locked before finite diferences or independent readouts are examined. The independently seeded training history is the primary inferential unit; windows from the same history are repeated conditions. The corresponding experimental definitions and estimands are given in Supps. B–C. Replication across histories is summarized by history-level bootstrap intervals and exact sign tests [39], with block re sampling used to retain within-chain or contiguous-token dependence [40].

## IV. RESULTS

With the response construction and prospective protocols in place, we now examine finite-time transport in the Math–Code and Ising systems. We begin with its efect on prospectively locked decisions, then resolve the roles of the future minibatch path and optimizer-state coordinates, and finally evaluate transport-based ranking of finite-amplitude schedules.

## A. Decision value of delayed transport

Using the prospective protocol above, we quantify the efect of propagation beyond the current AdamW update on locked held-out decisions. Across 12 held-out 0.3M Math–Code histories, full eight-step transport outperforms the optimizer-aware immediate controller in $1 0 / 1 2$ histories (Fig. 2). The mean token-disjoint benefit is $4 . 7 0 8 \times 1 0 ^ { - 4 } ;$ ; its history-bootstrap 95% interval is $\left[ 2 . 6 6 0 , 6 . 5 4 5 \right] \times 1 0 ^ { - 4 }$ , and the exact one-sided sign-test probability is $7 9 / 4 0 9 6 = 0 . 0 1 9 3$ . Full transport also outperforms neutral training in $1 1 / 1 2$ histories, with mean benefit $7 . 8 0 5 \times 1 0 ^ { - 4 }$ , and VGA in $1 2 / 1 2$

Both controllers intervene in all 96 windows, yet they select the same temporal schedule in only 36. Their outcome diference therefore arises from schedule selection rather than action frequency. The full-versus-immediate increment is 60.3% of the full-versus-neutral benefit. A postlock evaluation over essentially all available target tokens retains the sign and magnitude of the efect, with positive diferences in $1 1 / 1 2$ histories. The domain-resolved token audit and same-state decision geometry are reported in Supp. E.2 and Fig. S5; the latter shows that the schedule rankings difer even when both scores are evaluated at the same state.

These comparisons isolate the decision value of propagation beyond the current AdamW update. Both controllers include the same optimizer-aware immediate derivative and intervene in every control window; they difer in how the perturbation is transported and read out over the remainder of the specified path. The resulting schedule changes and positive held-out efect show that delayed propagation contributes to prospective action quality. Because exposure-matched candidates often have similar terminal losses, this contribution can move the selected schedule across a discrete decision boundary while producing a small absolute loss diference.

The 1M extension separates local response fidelity from closed-loop decision reliability. Its pathwise tangent remains accurate, while repeated action selection gives a positive mean and $9 / 1 2$ positive histories, below the prespecified $1 0 / 1 2$ history-consistency criterion. These measurements probe diferent regimes: reconstruction concerns one eight-step neighborhood, whereas closedloop utility accumulates finite-amplitude remainders and policy-induced state shifts over successive windows. A nested duration study observes this progressive separation, with model scale and sufix length varied together. The 1M experiment therefore supports the one-window response mechanism; decision-level confirmation comes from the 0.3M cohort. The reconstruction and scale evidence are documented in Supps. E.1 and F and Fig. S2.

## B. Future-path conditioning

Building on the Math–Code decision result, the crossed Ising design separates the contribution of the checkpoint from its interaction with future minibatches. In the resulting decomposition, centered schedule-score variance comprises 10.3% state–schedule, 17.8% tape–schedule, and 71.9% state–tape–schedule interaction $\left( \mathrm { F i g . 3 } \right)$ . The large interaction shows that the efect of a future-minibatch sequence depends strongly on its starting checkpoint.

Consistent with this interaction, a state-only leaveone-tape-out controller does not transfer reliably: its mean benefit is $- 1 . 6 1 \times 1 0 ^ { - 5 }$ , with $1 / 4$ positive histories. Because most schedule-score variation resides in the state–tape–schedule interaction, averaging over other tapes suppresses the path-specific component needed to rank actions on the held-out tape. By contrast, when the next eight paired minibatches are committed before action selection, full control improves $6 0 / 7 2$ Ising cells and all six history means. The crossed and specified-path protocols are defined in Supp. B.3, and the complete comparator results appear in Supp. E.3 and Fig. S6. For the tested setting, the decision rule takes the form

$$
\boldsymbol { u } ^ { * } = \boldsymbol { u } ^ { * } ( \boldsymbol { x } _ { t } , \boldsymbol { \xi } _ { t : t + H } ) ,\tag{16}
$$

so the checkpoint alone is insuficient to determine the preferred action: the relevant control state comprises both the augmented optimizer state and the specified near-future drive.

![](images/cda634d9080e862499ab7d7b910883042c78105e9b1b171d03c7ff380c04bca7.jpg)

![](images/4acf07fa006b76e79ed60922d043d92451f034ae6c90ce72256ed750c2c212ce.jpg)

![](images/66e30442ee98a0020bc52c9566e0e2ad86580d242ab46a666fb3347b4c757ff8.jpg)  
Action agreement: full-myopic 36/96; full-VGA 15/96

FIG. 2. Delayed transport changes finite actions and improves the independent readout. (a) Token-disjoint history-level benefit of full eight-step transport relative to the optimizer-aware immediate derivative, VGA, and neutral training. (b) Paired benefit relative to neutral training. (c) Both full and immediate controllers intervene in all 96 windows, yet select the same temporal schedule in only 36. Actions and terminal states are locked before the test readout.  
![](images/928aeb2014232e5f8877b341b76b83bbe8006780d7b20e4a468804801326be64.jpg)

![](images/32d74eb628737b76dac1758f081fb4d7f6e68db97b3578998221b9e3b50dc1e1.jpg)

![](images/e818b67f302256331ceb736e584e91ce3c8f3c55e30210b85afafbad1cdc95ba.jpg)  
FIG. 3. The preferred short-horizon action depends on the future minibatch path. Here a tape is a pre-generated sequence of eight paired future minibatches. (a) Schedule-score variation is dominated by checkpoint–tape interaction. (b) On the realized tape, full transport retains positive decision value relative to memory-deleted and immediate controllers. (c) The value collapses when actions are predicted from other tapes, exposing the boundary of state-only control.

## C. Optimizer-moment contributions to the response

To identify the components that carry the pathconditioned response, we resolve the perturbation across the parameter and Adam-moment coordinates of the augmented state. In Math–Code, diferentiation through this full state reconstructs paired finite-amplitude responses without a fitted response model. Reconstruction is measured by the normalized root-mean-square error (NRMSE) and cosine similarity between predicted and observed response vectors. At 0.3M parameters, the full-state prediction gives NRMSE = 0.0239 and cosine similarity 0.9997 for m, and NRMSE = 0.0490 and cosine similarity 0.9988 for e. At 1M, the corresponding NRMSE/cosine pairs are 0.0138/0.9999 and 0.0304/0.9995 (Fig. 4a).

Within this reconstruction, the signed block-projection fraction defined in Supp. A.2 resolves contributions to the next-parameter tangent by state coordinate. The first Adam moment contributes 0.209 at 0.3M and 0.207 at 1M. In a separate optimizer intervention, reducing $\beta _ { 1 }$ lowers this projection to 0.424 of baseline, whereas reducing $\beta _ { 2 }$ increases the second-moment norm ratio by a factor of 20.7. The full tangent outperforms its memory-deleted counterpart in all 62 eligible momentum and AdamWfamily cells. Under SGD, the same nominal pulse produces a substantially larger finite response and an inaccurate local reconstruction. The optimizer-family results and this amplitude boundary are documented in Supps. E.1 and F.

Ensemble averaging provides a separate test of path specificity (Fig. 4c,d). Averaging paths before comparing scales gives cosine similarity 0.978 for m and 0.083 for $e ,$ although individual 1M e paths remain accurately reconstructed $( \mathrm { N R M S E } = 0 . 0 3 0 4 )$ The pooled profile combines scale dependence in the terminal projection C and the transported tangent ΦB, so weak transfer of the averaged e profile can coexist with accurate prediction on each path. Together with the coordinate interventions, this contrast identifies optimizer moments as a measurable path-level transmission channel; the ensemble-averaged response is not a scale-independent kernel.

![](images/0c3283423b94bd575decae0e71bb2b868ba0b8ed15f6a5cede1897c56cfca228.jpg)

![](images/43467e5fc4e4eff0901703abed222f690b9360a4cd1c38e9fe30b28117535ce0.jpg)

![](images/5b5b8698dae4dd020d6ae944a0031dfb63ef1641c64ef58aea2a4bfc79cf7d56.jpg)

![](images/d78350e951c7cb5d0dfe1dd6b94dff70e8855013d30375399f703626143062a1.jpg)  
FIG. 4. Optimizer moments carry delayed response across Transformer scales. (a) Direct pathwise tangent reconstruction remains accurate at approximately 0.3M and 1M parameters. (b) Mean signed block-projection fractions quantify contributions to the next parameter tangent; the definition and closure check are given in Supp. A.2. (c,d) Individual-path transport remains accurate even when an ensemble-mean response profile transfers weakly across scales.

## D. Ising–CNN validation of moment-state transport

The Math–Code analysis identifies moment-mediated transport, which we further examine in the Ising–CNN system. Full-state predictions on the Ising–CNN paths give NRMSE = 0.002115 for m and 0.001580 for e. On the same nominal trajectories, deleting moment-state tangents raises these errors to 0.6923 and 0.6871 (Fig. S4). Accurate reconstruction in this setting therefore requires transmission through the moment coordinates, consistent with the degradation observed in Math–Code.

Beyond response reconstruction, the prospective Ising test uses the independently locked Binder anchor. Its final confirmation retains the fixed $T _ { B }$ and replaces the neural histories, future tapes, calibration chains, and test chains. Full and exhaustive-rollout policies agree in 11/12 cells. On the new test chains, the mean absolute deviation $| \widehat { T } _ { N } - T _ { B } |$ falls from 0.02435 to 0.01880. All six histories improve, with a block-aware interval of [0.00258, 0.00839] (Fig. 5). Binder finite-size analysis remains the criticalpoint reference, while the neural crossing serves as the controlled training readout. The calibration boundary and fresh-chain confirmation are reported in Supp. E.3; the anchored protocol is defined in Supp. B.2. Together, these results reproduce moment-mediated transport under a diferent architecture and readout, with the Binder construction providing an external physical reference for the controlled neural observable.

## E. Transport-guided ranking of finite actions

With the local mechanism established, we evaluate transport scores as a way to reduce exact finite-amplitude search in larger schedule libraries. For a scalar terminal objective, one reverse adjoint gives derivatives at every source time. A schedule library can then be ranked by scalar products. Because finite amplitude may change the ordering, exact rollouts evaluate the shortlisted candidates. Ranking quality is measured by winner recall: the fraction of settings in which the exhaustive finiteamplitude winner lies among the top-q candidates of a ranking rule.

On 54 fresh Ising settings, a development-calibrated adaptive shortlist includes the exhaustive finite-amplitude winner in every case, and exact reranking selects it without a clipping-branch mismatch (Fig. 6). At an equal budget of neutral plus eight nonneutral exact rollouts, winner recall for full, memory-deleted, immediate, and random ranking is 0.889, 0.796, 0.370, and 0.296. The corresponding Math–Code values are 1.000, 0.907, 0.407, and 0.155. With fixed $q = 1 6 ,$ , exact reranking recovers all 108 exhaustive winners on fresh Math–Code settings. The adaptive rule recovers 100/108 overall; one library gives 32/36, compared with its prespecified criterion of 33/36. The missed candidates occur where the finiteamplitude remainder and local score spacing exceed the calibrated cutof, quantities not determined by amplitude and library size alone. Its locked actions still improve the token-disjoint objective in all six fresh histories, with mean benefit $6 . 4 6 \times 1 0 ^ { - 4 }$ The transfer audit and calibration boundary are reported in Supp. E.4 and Fig. S7; measured costs appear in Tables S1–S2.

![](images/12886af6828f87863bf108395c41326893cb59339124f627fed45cea80ad60a1.jpg)

b  
![](images/95e506b9e3ba6a9102efc5b3e830fc26d5f1f9da042b4e7e8bb0559f3013ebaa.jpg)

![](images/cd9005cddb6521fef17dcf48f2603ac10c8577158c3ae1195cbeae899da7ae84.jpg)

![](images/0a97000f9df8fe6cbb7bdf83493eaea9ecd07960d9c8b09927d532a5b0f6ebd2.jpg)  
FIG. 5. The optimizer-state mechanism transfers to a Binder-anchored Ising benchmark. (a) Binder crossings define the physical anchor before neural action selection; the final confirmation reuses this anchor while replacing neural paths and test chains. (b) Candidate-contrast NRMSE, cosine similarity, and discrete action agreement across the staged tests. The elevated absolute-calibration NRMSE marks a calibration boundary, while action agreement remains high. (c) The selected action reduces the absolute deviation of the neural crossing from the Binder anchor on new test chains. (d) Full transport outperforms the immediate controller relative to neutral training.

These results support full transport as a screening statistic that concentrates strong finite actions, while exact rollout reranking resolves finite-amplitude competition within the shortlist. The fixed-q result is stable across the tested libraries, whereas the adaptive width remains system-calibrated.

## V. DISCUSSION

Across the two systems, short-horizon loss-weight control emerges as a path-conditioned dynamical problem on the augmented model–optimizer state. Delayed propagation changes prospectively locked decisions, Adam moments carry a measurable part of the perturbation, and the preferred schedule depends jointly on the current optimizer state and the near-future drive. These findings also distinguish accurate local response prediction from reliable finite-amplitude action selection.

These results connect recent analyses of noncommuting domain updates, optimizer-memory order efects, and optimizer-aware attribution to prospective decisions over finite temporal actions [18–20]. They also complement truncated hypergradients [41] and dynamic data-mixture methods that modify sample allocation or the subsequent training path [5, 6, 10, 11]. In the paired-loss setting studied here, the diferentiated object is a specified finite future map. Aggregate domain exposure remains fixed, so the action changes the temporal placement of domain weights rather than their total allocation. The Ising benchmark tests the same dynamical mechanism in a diferent architecture, data-generating system, and physically anchored readout.

![](images/051d4d4d5f9ced41d4edd3469f7d30c3a1b4bed1b4f693c79af48b56b856607d.jpg)

![](images/7162b6bf39e378cc48b6cbe91a0cd1d7eaaa6aacf1a9a002ac9d63190bd84107.jpg)

![](images/916734d02e6a3a4ca4b7e47668f432f601fae44336ef2fb545cfa06de3caf77b.jpg)  
FIG. 6. Full transport concentrates finite-amplitude winners at equal exact-rollout budget. (a) Each colored point is the winner recall obtained after holding out one Ising path, Ising history, or Math–Code history cluster; black segments denote the mean within each cluster type. (b) With neutral plus q = 8 nonneutral rollouts, full transport has the highest exhaustive-winner recall in both systems, followed by memory deletion, immediate ranking, and random selection. (c) Charging the one-time development calibration gives break-even points of 7.32 Ising and 12.22 Math–Code future-path bundles fo repeated reuse of the same library structures. This accounting counts candidate rollouts rather than total training time.

Finite amplitude introduces an additional ranking problem because curvature can reorder candidates that are close under the local tangent. The small candidate library therefore tests the tangent-selected action directly, whereas the larger libraries use full transport to form a shortlist before exact rollout reranking. At equal exactrollout budget, full transport places the exhaustive winner near the top more often than immediate, memorydeleted, or random ranking. The adaptive shortlist requires system-specific empirical calibration and therefore complements, rather than replaces, fixed-budget shortlist selection. The computational tradeof depends on how the derivatives are evaluated. Forward full transport propagates source sensitivities through the augmented state, whereas the reverse adjoint obtains all source-time derivatives in one backward sweep while retaining intermediate trajectory states. In the measured CPU implementation, forward full transport takes 1.33 times the optimizer-aware immediate-controller total; reverseadjoint full transport takes 0.80 times that total but 2.72 times its peak resident memory. These measurements explain the observed time–memory tradeof and identify the regime in which transport-guided screening is most useful: candidate libraries large enough to benefit from shortlist reranking, especially when their structure is reused across future paths.

The present evidence defines a local operating regime for this description. It is strongest for eight-step control with Adam-like optimizers and the 0.3M Transformer. At 1M parameters, the mean decision efect remains positive, while accumulated finite-amplitude and closed-loop deviations accompany a missed history-consistency criterion. Under SGD, the common pulse produces a much larger response, confounding optimizer family with efective perturbation scale. The assumption of a specified near-future minibatch sequence is also substantive. Unknown future drives would require averaging over possible paths, while changes in sample allocation would introduce additional gradient-covariance efects. Larger models, longer horizons, amplitude-matched optimizer comparisons, and future-path marginalization provide direct tests of how far the finite-time description extends.

## VI. CONCLUSION

Adaptive optimizers retain gradient history, but how this internal state changes finite training decisions has been unclear. We addressed this problem by diferentiating specified eight-step AdamW paths through the complete model–optimizer state, selecting exposure-matched temporal schedules, and locking each action before independent evaluation. Math–Code supplied the primary prospective decision test, while an Ising–CNN benchmark examined the same mechanism under a diferent architecture, data-generating system, and physically anchored readout.

In the primary 0.3M Math–Code confirmation, full transport improved held-out decisions relative to an optimizer-aware immediate derivative and reordered schedules despite matched intervention frequency. Crossed checkpoint–path experiments showed that the preferred action depends jointly on the current optimizer state and the near-future minibatches, while targeted tangent deletion identified Adam moments as a measurable transmission channel in both systems. For larger candidate libraries, transport scores concentrated exactrollout winners and focused finite-amplitude evaluation

on a shortlist.

These results identify short-horizon loss-weight control as a path-conditioned dynamical problem on the augmented training state. Along committed paths, full-state derivatives can therefore serve both as mechanistic probes of training dynamics and as screening scores for finite interventions. Future-path marginalization, longer horizons, and larger models provide direct tests of how far this local description extends.

## ACKNOWLEDGMENTS

The work of J.G. is supported by the Postdoctoral Fellowship Program (Grade C) of China Postdoctoral

[1] D. P. Kingma and J. Ba, in International Conference on Learning Representations (2015) arXiv:1412.6980.

[2] I. Loshchilov and F. Hutter, in International Conference on Learning Representations (2019) arXiv:1711.05101.

[3] A. Albalak, Y. Elazar, S. M. Xie, S. Longpre, N. Lambert, X. Wang, N. Muennighof, B. Hou, L. Pan, H. Jeong, C. Rafel, S. Chang, T. Hashimoto, and W. Y. Wang, Transactions on Machine Learning Research (2024), arXiv:2402.16827.

[4] Y. Liu, C. Chen, J. Yang, and R. Sun, arXiv preprint arXiv:2505.21598 10.48550/arXiv.2505.21598 (2025).

[5] S. M. Xie, H. Pham, X. Dong, N. Du, H. Liu, Y. Lu, P. Liang, Q. V. Le, T. Ma, and A. W. Yu, in Advances in Neural Information Processing Systems, Vol. 36 (2023) pp. 69798–69818, arXiv:2305.10429.

[6] S. Fan, M. Pagliardini, and M. Jaggi, in Proceedings of the 41st International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 235 (2024) pp. 12895–12915.

[7] Q. Liu, X. Zheng, N. Muennighof, G. Zeng, L. Dou, T. Pang, J. Jiang, and M. Lin, in International Conference on Learning Representations (2025) pp. 38305–38339, arXiv:2407.01492.

[8] M. F. Chen, N. Roberts, K. Bhatia, J. Wang, C. Zhang, F. Sala, and C. Ré, in Advances in Neural Information Processing Systems, Vol. 36 (2023) pp. 36000–36040, arXiv:2307.14430.

[9] A. Albalak, L. Pan, C. Rafel, and W. Y. Wang, arXiv preprint arXiv:2312.02406 10.48550/arXiv.2312.02406 (2023).

[10] S. Fan, D. Grangier, and P. Ablin, arXiv preprint arXiv:2410.02498 10.48550/arXiv.2410.02498 (2024).

[11] Z. Li, Y. Deng, P. Zhong, M. Razaviyayn, and V. Mirrokni, in Advances in Neural Information Processing Systems, Vol. 38 (2025) pp. 169400–169440, arXiv:2502.06244.

[12] D. Maclaurin, D. Duvenaud, and R. P. Adams, in Proceedings of the 32nd International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 37 (2015) pp. 2113–2122.

[13] L. Franceschi, P. Frasconi, S. Salzo, R. Grazzi, and M. Pontil, in Proceedings of the 35th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 80 (2018) pp. 1568–1577.

Science Foundation under Grant No. GZC20252775. The author used OpenAI ChatGPT and Codex to assist with code debugging, literature searches, and language editing. All AI-generated suggestions were reviewed by the author, who assumes full responsibility for the content of this work.

## DATA AVAILABILITY AND CODE

The code required to reproduce the analyses in this work is publicly available in the project’s GitHub repository. All relevant input parameters and plotting scripts are included there.

[14] P. W. Koh and P. Liang, in Proceedings of the 34th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 70 (2017) pp. 1885–1894.

[15] G. Pruthi, F. Liu, S. Kale, and M. Sundararajan, in Advances in Neural Information Processing Systems, Vol. 33 (2020) pp. 19920–19930, arXiv:2002.08484.

[16] Y. Chen, B. Li, H. Yu, P. Wu, and C. Miao, in Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 35 (2021) pp. 7081–7089, arXiv:2102.02515.

[17] S. M. Park, K. Georgiev, A. Ilyas, G. Leclerc, and A. Madry, in Proceedings of the 40th International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 202 (2023) pp. 27074–27113.

[18] A. Rukhovich, A. Podolskiy, and I. Piontkovskaya, arXiv preprint arXiv:2501.15556 (2025), arXiv:2501.15556 [cs.LG].

[19] J. Sweeney, arXiv preprint arXiv:2606.29554 (2026), arXiv:2606.29554 [cs.LG].

[20] M. Ding, Z. Zhang, D. Wang, and L. Hu, arXiv preprint arXiv:2602.00329 (2026), arXiv:2602.00329 [cs.LG].

[21] R. Kubo, Journal of the Physical Society of Japan 12, 570 (1957).

[22] D. Ruelle, Physics Letters A 245, 220 (1998).

[23] R. Zwanzig, Physical Review 124, 983 (1961).

[24] H. Mori, Progress of Theoretical Physics 33, 423 (1965).

[25] M. di Bernardo, C. J. Budd, A. R. Champneys, and P. Kowalczyk, Piecewise-Smooth Dynamical Systems: Theory and Applications, Applied Mathematical Sciences, Vol. 163 (Springer, London, 2008).

[26] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, in Advances in Neural Information Processing Systems, Vol. 30 (2017) pp. 5998–6008.

[27] D. Saxton, E. Grefenstette, F. Hill, and P. Kohli, in International Conference on Learning Representations (2019) arXiv:1904.01557.

[28] Python Software Foundation, The Python standard library, Python 3.12 (2026), accessed August 24, 2026.

[29] D. Q. Mayne, J. B. Rawlings, C. V. Rao, and P. O. M. Scokaert, Automatica 36, 789 (2000).

[30] Y. Wu and K. He, in Computer Vision – ECCV 2018 (2018) pp. 3–19, arXiv:1803.08494.

[31] U. Wolf, Physical Review Letters 62, 361 (1989).

[32] J. Carrasquilla and R. G. Melko, Nature Physics 13, 431 (2017).

[33] E. P. L. van Nieuwenburg, Y.-H. Liu, and S. D. Huber, Nature Physics 13, 435 (2017).

[34] D. Bachtis, G. Aarts, and B. Lucini, Physical Review E 102, 053306 (2020).

[35] K. Binder, Zeitschrift für Physik B Condensed Matter 43, 119 (1981).

[36] B. Kastening, Physical Review E 87, 044101 (2013), arXiv:1209.0105.

[37] H. A. Kramers and G. H. Wannier, Physical Review 60, 252 (1941).

[38] L. Onsager, Physical Review 65, 117 (1944).

[39] B. Efron, The Annals of Statistics 7, 1 (1979).

[40] H. R. Künsch, The Annals of Statistics 17, 1217 (1989).

[41] A. Shaban, C.-A. Cheng, N. Hatch, and B. Boots, in Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, Proceedings of Machine Learning Research, Vol. 89 (2019) pp. 1723–1732.

[42] B. A. Pearlmutter, Neural Computation 6, 147 (1994).

[43] H. Akaike, IEEE Transactions on Automatic Control 19, 716 (1974).

[44] G. Shafer and V. Vovk, Journal of Machine Learning Research 9, 371 (2008).

[45] R. I. Oliveira, P. Orenstein, T. Ramos, and J. V. Romano, Journal of Machine Learning Research 25, 1 (2024).

[46] Y. Jin and E. J. Candès, Journal of Machine Learning Research 24, 1 (2023).

# Supplemental Material for Delayed Optimizer-State Transport Shapes Short-Horizon Training Decisions

Jinhui Guo

School of Physics, Beihang University, Beijing 100191, China

This Supplemental Material provides detailed derivations, experimental definitions, statistical procedures, additional results, and reproducibility information supporting the main text. The notation follows that of the main text. Supp. A gives the AdamW-specific tangent equations and coordinate-resolved comparator implementation. Supps. B–D specify the experimental systems, statistical estimands, branch safeguards, and finite-action protocol. Supp. E presents the supporting results and cost measurements, while Supps. F and G summarize the empirical boundaries and reproducibility record.

Throughout, locked before evaluation means that the configuration, implementation, candidate set, predictions or actions, and any terminal-state hashes were fixed before the designated audit or independent test outcome was examined. This ordering keeps the stated comparisons outcome-blind. The lock records are internal protocol records and were not externally preregistered.

## SUPP. A. ADAMW OPTIMIZER-STATE TANGENT AND COORDINATE TRANSPORT

The source–transport–readout factorization and finite-action scores are defined in Sec. II. This section supplies the AdamW-specific tangent equations, coordinate decomposition, and implementation checks used for optimizer-memory attribution.

## A.1. AdamW tangent equations

For AdamW $[ 1 , 2 ]$ , let ${ r } _ { t } = \nabla _ { \boldsymbol { \theta } } \ell _ { t } ( \boldsymbol { \theta } _ { t } , p _ { t } )$ be the raw gradient and let $\widetilde { \boldsymbol { r } } _ { t } = \boldsymbol { \mathcal { C } } _ { t } ( \boldsymbol { r } _ { t } )$ denote the gradient after the active clipping map. The variables $\mu _ { t }$ and $v _ { t }$ are the uncorrected exponential moving averages of the clipped gradient and its elementwise square, respectively. Here ⊙ and $\oslash$ denote elementwise multiplication and division, and square roots of vector-valued moment quantities are also evaluated elementwise. The moment updates are

$$
\begin{array} { r l } & { \mu _ { t + 1 } = \beta _ { 1 } \mu _ { t } + ( 1 - \beta _ { 1 } ) \widetilde { r } _ { t } , } \\ & { v _ { t + 1 } = \beta _ { 2 } v _ { t } + ( 1 - \beta _ { 2 } ) \widetilde { r } _ { t } \odot \widetilde { r } _ { t } . } \end{array}\tag{S1}
$$

If $n _ { t + 1 }$ is the optimizer clock after the update, the hats denote bias-corrected moments,

$$
\widehat { \mu } _ { t + 1 } = \frac { \mu _ { t + 1 } } { 1 - \beta _ { 1 } ^ { n _ { t + 1 } } } , \qquad \widehat { v } _ { t + 1 } = \frac { v _ { t + 1 } } { 1 - \beta _ { 2 } ^ { n _ { t + 1 } } } .\tag{S2}
$$

With $d _ { t + 1 } = \sqrt { \widehat { v } _ { t + 1 } } + \epsilon$ , the parameter update is

$$
\theta _ { t + 1 } = ( 1 - \eta _ { t } \lambda ) \theta _ { t } - \eta _ { t } \widehat { \mu } _ { t + 1 } \oslash d _ { t + 1 } .\tag{S3}
$$

Here $\eta _ { t }$ is the learning ra $\mathrm { { ; e , } } \lambda$ is the decoupled weight-decay coeficient, and ϵ is the denominator stabilizer.

On a fixed diferentiable clipping branch, let $H _ { t } = \partial r _ { t } / \partial \theta$ and $q _ { t } = \partial r _ { t } / \partial p$ . Then

$$
\delta \widetilde { \boldsymbol { r } } _ { t } = D \mathcal { C } _ { t } ( \boldsymbol { r } _ { t } ) \left( H _ { t } \delta \boldsymbol { \theta } _ { t } + q _ { t } \delta p _ { t } \right) .\tag{S4}
$$

For the paired weighted loss, $q _ { t } = \nabla _ { \theta } \ell _ { A , t } - \nabla _ { \theta } \ell _ { B , t }$ . The moment tangents are

$$
\begin{array} { r l } & { \delta \mu _ { t + 1 } = \beta _ { 1 } \delta \mu _ { t } + ( 1 - \beta _ { 1 } ) \delta \widetilde { r } _ { t } , } \\ & { \delta v _ { t + 1 } = \beta _ { 2 } \delta v _ { t } + 2 ( 1 - \beta _ { 2 } ) \widetilde { r } _ { t } \odot \delta \widetilde { r } _ { t } . } \end{array}\tag{S5}
$$

Because the control does not perturb the optimizer clock,

$$
\delta \widehat { \mu } _ { t + 1 } = \frac { \delta \mu _ { t + 1 } } { 1 - \beta _ { 1 } ^ { n _ { t + 1 } } } , \qquad \delta \widehat { v } _ { t + 1 } = \frac { \delta v _ { t + 1 } } { 1 - \beta _ { 2 } ^ { n _ { t + 1 } } } .\tag{S6}
$$

The parameter tangent is therefore

$$
\delta \theta _ { t + 1 } = ( 1 - \eta _ { t } \lambda ) \delta \theta _ { t } - \eta _ { t } \left[ \delta { \widehat { \mu } } _ { t + 1 } \oslash { d } _ { t + 1 } - ( { \widehat { \mu } } _ { t + 1 } \odot \delta { \widehat { v } } _ { t + 1 } ) \oslash \left( 2 { \sqrt { { \widehat { v } } _ { t + 1 } } } \odot d _ { t + 1 } \odot d _ { t + 1 } \right) \right] .\tag{S7}
$$

The implementation evaluates the clipping diferential and Hessian–vector products by matrix-free automatic diferentiation [42]; no dense Hessian is formed.

## A.2. Coordinate-resolved transport and comparator implementation

The equations above propagate the complete state tangent. To resolve the contribution of individual state coordinates, let $P _ { j }$ project onto $j \in \{ \theta , \mu , v \}$ , let $\dot { x } _ { k }$ and $\dot { p } _ { k }$ denote the state and control tangents along the specified intervention direction, and write $\dot { x } _ { i , k + 1 } = P _ { i } \dot { x } _ { k + 1 }$ . At a propagation step after the source injection, where $\dot { p } _ { k } = 0$ , the contribution from input block $j$ to output block i after one update is

$$
w _ { i  j , k } = P _ { i } A _ { k } P _ { j } \dot { x } _ { k } , \qquad c _ { i  j , k } = \frac { \langle \dot { x } _ { i , k + 1 } , w _ { i  j , k } \rangle } { \| \dot { x } _ { i , k + 1 } \| ^ { 2 } } .\tag{S8}
$$

If a control is applied at step $k , P _ { i } B _ { k } { \dot { p } } _ { k }$ enters separately as a new source contribution. The denominator in Eq. (S8) is the squared norm of the target tangent, so $c _ { i  j , k }$ is a signed reconstruction contribution. For $i = \theta ,$ the mean $( \theta , \mu , v )$ contributions are $( 0 . 7 9 3 0 , 0 . 2 0 8 9 , - 0 . 0 0 1 8 7 )$ at 0.3M and $( 0 . 7 9 3 6 , 0 . 2 0 6 8 , - 0 . 0 0 0 3 7 )$ at 1M. Their near-unit sums give an internal closure check. In the 1M implementation audit, the maximum block-closure and readout-factorization errors are $5 . 9 6 \times 1 0 ^ { - 8 }$ and $1 . 5 4 \times 1 0 ^ { - 8 }$ , respectively. The approximately 21% value in the main text is the first-moment block’s signed projection onto the next parameter tangent. Supp. E.1 reports direct finite-response reconstruction; the timed forward and adjoint implementations select the same action on all four checked paths (Table S1).

The memory-deleted diagnostic projects the propagated tangent after every update as $( \delta \theta , \delta \mu , \delta v ) \mapsto ( \delta \theta , 0 , 0 )$ while leaving the nominal parameters, moments, and future minibatches unchanged. The optimizer-aware immediate comparator uses the same complete one-step tangent but terminates before subsequent propagation, as defined in Eq. (14). These two constructions separately test moment-coordinate transmission and the decision value of the remaining specified future map.

## SUPP. B. EXPERIMENTAL SYSTEMS AND PROSPECTIVE CONTROLS

Section III introduces the two experimental systems and their prospective tests. Here we give the model configurations, data partitions, candidate schedules, physical readouts, and crossed future-path construction needed to interpret and reproduce those tests.

## B.1. Math–Code Transformer

Domain A contains 56 medium-entropy modules of the DeepMind Mathematics Dataset; domain B contains complete Python standard-library files. Training and validation files are disjoint, with matched byte-token budgets. The causal byte-level Transformers [26] use sequence length 64, batch size $^ { 8 , }$ zero dropout, four attention heads, and AdamW with $\beta _ { 1 } = 0 . 9$ and $\beta _ { 2 } = 0 . 9 9 9$ . The smaller model has 296,504 parameters in three width-104 layers, and the larger model has 1,055,648 parameters in four width-176 layers.

Training progress is indexed by an efective weighted token exposure per model parameter. With t optimizer steps, per-domain minibatch size $B ,$ sequence length S, and parameter count $P ,$ this gives

$$
\tau = { \frac { t B S } { P } } , \qquad \Delta \tau _ { \mathrm { c o n t r o l } } = { \frac { H _ { \mathrm { t o t a l } } B S } { P } } ,\tag{S9}
$$

where tBS is the exposure of the unit-normalized mixture loss. Each paired step evaluates B sequences from each domain, so the computational token count is $2 t B S ; \tau$ uses the weighted-mixture convention to provide a common checkpoint coordinate. Dividing by P compares how much weighted data exposure each parameter has received. Steps 1750 at 0.3M and 6200 at 1M both lie near $\tau \simeq 3 .$ . The same coordinate maps a 64-step controlled sufix at 0.3M to approximately 228 steps at 1M; the long 1M development trajectory uses 224 steps.

The primary library contains neutral and six zero-sum eight-step schedules, with $u _ { j } \in \{ - 1 , + 1 \} , \sum _ { i } u _ { j } = 0$ , and amplitude $a = 0 . 0 2$ . A plus sign increases the Math weight to $p _ { 0 } + a$ and a minus sign increases the Code weight to $1 - p _ { 0 } + a$ . The schedule names used in $\mathrm { F i g }$ . S1c correspond to

$$
\begin{array} { r } { u ^ { \mathrm { e a r l y ~ A } } = ( + , + , + , + , + , - , - , - , - ) , \qquad u ^ { \mathrm { l a t e ~ A } } = ( - , - , - , - , + , + , + , + ) , } \\ { u ^ { \mathrm { m i d d l e ~ A } } = ( - , - , + , + , + , + , - , - ) , \quad u ^ { \mathrm { e d g e s ~ A } } = ( + , + , - , - , - , - , + , + ) , } \\ { u ^ { \mathrm { a l t . ~ A } } = ( + , - , + , - , + , - , + , - ) , \qquad u ^ { \mathrm { a l t . ~ B } } = ( - , + , - , + , - , + , - , + ) . } \end{array}\tag{S10}
$$

All six therefore have the same aggregate Math and Code weight over a window and difer only in temporal order. Guarded token positions divide the validation corpus into controller, audit, and test thirds. The controller partition is used to score actions; audit and test readouts are evaluated after the actions and terminal-state hashes have been locked.

The primary four-arm experiment compares neutral, VGA, the Adam-aware immediate derivative, and full transport on 12 held-out histories under identical checkpoints, future minibatches, schedules, branch screens, and readout partitions; Supp. E.2 gives the full comparison. An earlier independent cohort used the same 0.3M model and schedule library to compare full transport with neutral training over eight replanning windows. Denote their terminal symmetric test losses by $\bar { e } _ { * }$ and $\bar { e } _ { 0 } .$ . The benefit $\bar { e } _ { 0 } - \bar { e } _ { * }$ is positive in $1 1 / 1 2$ histories, and its mean remains positive throughou the sufix (Fig. S1a,b). The selected schedules span the complete nonneutral library (Fig. S1c), so the ordering is recomputed as the controlled trajectory evolves.

![](images/58ac0a01efa272eedfe94bcf8a3ff401f5c009d8d87530e61227770e0cedff18.jpg)

![](images/fbc654d4b2e3909efd7493df75261fb4176d90f77e2bce564181fce37fe16218.jpg)

![](images/db0aeeaa9396706985e7931e57ac59d45abb5e68d36669c7580cc96f034afed4.jpg)  
FIG. S1. Receding-horizon behavior in the independent 0.3M full-versus-neutral cohort. (a) Terminal test benefit $\bar { e } _ { 0 } - \bar { e } _ { * }$ by training history; the dashed line marks the history mean. (b) Controller-readout benefit over the eight replanning windows; thin lines show histories, and the dark curve and band show the mean and its standard error. (c) Selection counts for the schedules in $\operatorname { E q . }$ (S10).

![](images/d5b1ac9109b09114e65129671d9569b914c0796618a59d2c94c3c44d26512205.jpg)  
FIG. S2. Math–Code scale and duration observations. (a) History-level 1M benefits on the locked and dense test readouts. (b) Mean 64-step benefit at 0.3M and 1M; error bars are history-bootstrap 95% intervals. (c) Controller, audit, and test benefits after 64 and 224 controlled steps in one nested 1M development trajectory.

The scale extension keeps the training age near $\tau = 3$ and first retains the same 64-step sufix. Its mean benefit is $2 . 3 2 9 \times 1 0 ^ { - 4 }$ with history-bootstrap interval $[ 0 . 3 9 9 , 4 . 3 8 4 ] \times 1 0 ^ { - 4 } ; 9 / 1 2$ histories are positive, compared with the prespecified consistency threshold of 10/12. The dense token readout preserves this count $( \mathrm { F i g . \ S 2 a , b ) }$ . In the single 224-step development trajectory, controller-readout benefit remains positive while independent-test benefit approaches zero (Fig. S2c). The longer sufix accumulates finite-amplitude displacement and changes the states at which later windows are linearized. At 1M, accurate one-window response therefore coexists with less uniform utility after repeated closed-loop interventions.

## B.2. Ising–CNN system and physical anchor

Two-dimensional Ising configurations are generated by Wolf cluster updates [31]. The supervised class label and the controlled data domain are distinct. Class 0 contains ordered-phase anchors at temperatures {1.50, 1.80, 2.10, 2.18}, and class 1 contains disordered-phase anchors at {2.36, 2.44, 2.80, 3.20}. Domain A contains the four anchors nearer the transition, {2.10, 2.18, 2.36, 2.44}, whereas domain B contains the four farther anchors, {1.50, 1.80, 2.80, 3.20}. The temperatures 2.24 and 2.30 are excluded from supervised training and appear only in the neural transition readout. The 252,642-parameter $L = 1 6$ CNN uses GroupNorm [30], zero dropout, and AdamW. Its temperatureresolved logit crossing is the diferentiable training observable, following earlier machine-learning studies of phases and transitions [32–34].

For a configuration σ at temperature T, let $z _ { 0 } ( \sigma ; \theta )$ and $z _ { 1 } ( \sigma ; \theta )$ be the ordered- and disordered-class logits. For $N _ { T }$ readout configurations $\{ \sigma _ { i } ^ { ( T ) } \} _ { i = 1 } ^ { N _ { T } }$ , define the mean disordered-minus-ordered contrast

$$
\bar { z } _ { \theta } ( T ) = \frac { 1 } { N _ { T } } \sum _ { i = 1 } ^ { N _ { T } } \left[ z _ { 1 } ( \sigma _ { i } ^ { ( T ) } ; \theta ) - z _ { 0 } ( \sigma _ { i } ^ { ( T ) } ; \theta ) \right] .\tag{S11}
$$

A line $\bar { z } _ { \theta } ( T ) = \alpha + \beta T$ fitted at $T \in \{ 2 . 1 8 , 2 . 2 4 , 2 . 3 0 , 2 . 3 6 \}$ defines $\begin{array} { r } { \widehat { T } _ { N } = - \alpha / \beta , } \end{array}$ , the temperature at which the fitted classifier preference changes sign. This $L = 1 6$ neural crossover is the diferentiable controller readout. The Binder construction below provides the thermodynamic reference.

Let $m _ { \mathrm { s p i n } } = L ^ { - 2 } \bar { \sum _ { i } } \sigma _ { i }$ be the magnetization per spin. The Binder cumulant and each ratio-two crossing are defined by

$$
U _ { L } ( T ) = 1 - \frac { \langle m _ { \mathrm { s p i n } } ^ { 4 } \rangle _ { T } } { 3 \langle m _ { \mathrm { s p i n } } ^ { 2 } \rangle _ { T } ^ { 2 } } , \qquad U _ { L } ( T _ { \times } ) = U _ { 2 L } ( T _ { \times } ) ,\tag{S12}
$$

and their finite-size drift is modeled as

$$
T _ { \times } ( L , 2 L ) = T _ { B } + A L ^ { - r } .\tag{S13}
$$

Here $T _ { B }$ is the extrapolated crossing temperature, A the drift amplitude, and r a correction exponent. The Monte Carlo bank contains eight independent chains, 256 configurations per temperature, eight temperatures from 2.10 to $2 . 4 5 ,$ and sizes $L \in \{ 8 , 1 2 , 1 6 , \bar { 2 4 } , 3 2 , 4 8 \}$ . Crossings from (8, 16), (12, 24), (16, 32), and (24, 48) are fitted at frozen $r \in \{ 0 . 5 , 1 . 0 , \ldots , 4 . 0 \}$ and combined using Akaike weights [43]. Circular-block resampling propagates within-chain dependence, and variation across the fitted exponents is included as between-model uncertainty [40]. The resulting $T _ { B }$ distribution is frozen before neural action selection. Post-lock evaluation uses the exact infinite-lattice value $2 / \log ( 1 + { \sqrt { 2 } } ) \ [ 3 7 , 3 8 ]$

For eight independent $L = 1 6$ neural calibration chains, the controlled objective is

$$
\mathcal { I } _ { \mathrm { a n c h o r } } = ( \overline { { \widehat { T } _ { N } } } - T _ { B } ) ^ { 2 } + \mathrm { V a r } _ { c } ( \widehat { T } _ { N , c } ) + 0 . 1 \frac { \overline { { R _ { c } } } } { ( \overline { { \beta _ { c } } } ) ^ { 2 } + 1 0 ^ { - 1 2 } } ,\tag{S14}
$$

where c indexes the eight neural calibration chains, $\widehat { T } _ { N , c }$ is the fitted crossing on chain $c , \beta _ { c }$ is its fitted slope, and $R _ { c }$ is its mean squared residual. Overbars denote the mean across chains, and $\mathrm { V a r } _ { c }$ is the population variance across the eight chains (divisor eight). The three terms penalize displacement from the Binder anchor, variation across neura chains, and poor conditioning of the fitted crossing. The coeficient 0.1 fixes the relative weight of the fit-quality term, and $1 \dot { 0 } ^ { - 1 2 }$ stabilizes its denominator. At baseline $p _ { 0 } = 0 . 5$ , the nine-candidate library contains neutral, the six zero-sum schedules in Eq. (S10) with A denoting the near-transition domain, and two constant schedules: near-heavy $= ( + , + , + , + , + , + , + , + )$ and far-heavy $= ( - , - , - , - , - , - , - , - )$ . The anchored protocol uses $a = 0 . 0 8$ . Binder scaling fixes the physical reference used in ${ \mathcal { I } } _ { \mathrm { a n c h o r } } ,$ which is diferentiated to score the candidate actions.

## B.3. Crossed future-tape and specified-path designs

The two designs resolve how the current checkpoint and the next minibatches jointly determine a schedule. The crossed audit measures transfer of a checkpoint-conditioned schedule across possible future minibatch sequences. It

uses four independent histories, checkpoints 60, 100, and 140, and the same eight future tapes at each of the resulting 12 states, giving $1 2 \times 8 = 9 6$ state–tape cells at amplitude $a = 0 . 0 0 2$ . A future tape is a pre-generated sequence of eight paired near- and far-domain minibatches.

Let $S _ { i \nu k }$ be the full-transport score of schedule k from state i on tape $\nu ,$ and let $Z _ { i \nu k } = S _ { i \nu k } - S _ { i \nu 0 }$ be its contrast with the neutral schedule. The balanced crossed design decomposes this contrast as

$$
Z _ { i \nu k } = \overline { { { Z } } } . . _ { k } + \Delta _ { i k } ^ { \mathrm { s t a t e } } + \Delta _ { \nu k } ^ { \mathrm { t a p e } } + \Delta _ { i \nu k } ^ { \mathrm { i n t } } ,\tag{S15}
$$

where $\Delta _ { i k } ^ { \mathrm { s t a t e } } = \overline { { Z } } _ { i \cdot k } - \overline { { Z } } _ { \cdot \cdot k }$ is the state–schedule term, $\Delta _ { \nu k } ^ { \mathrm { t a p e } } = \overline { { Z } } _ { \cdot \nu k } - \overline { { Z } } _ { \cdot \cdot k }$ the tape–schedule term, and $\Delta _ { i \nu k } ^ { \mathrm { i n t } }$ the remaining state–tape–schedule interaction. Each component is broadcast to the full $( i , \nu , k )$ array before its entries are squared and summed to give $S S _ { r }$ . The reported percentage for $r \in$ {state, tape, int} is $S S _ { r } / ( S S _ { \mathrm { s t a t e } } + S S _ { \mathrm { t a p e } } + S S _ { \mathrm { i n t } } )$

For the leave-one-tape-out test, each tape in turn is excluded. Scores and branch eligibility are aggregated over the other seven tapes, producing a checkpoint-conditioned action for the held-out tape. Comparators are neutral, memory-deleted, Adam-aware immediate, random matched-weight schedules, and a globally fixed schedule selected from the same seven tapes.

The specified-path design conditions the decision on the committed eight-minibatch tape. The same tape therefore determines the propagator and the executed finite action. Six new histories, three checkpoints per history, and four tapes per checkpoint yield 72 cells. Predictions, actions, branch records, checkpoints, and code are locked before generation of a new independent test chain. Its decision rule is $\boldsymbol { u } ^ { * } ( x _ { t } , \xi _ { t : t + H } )$ . Results for the crossed and specified-path designs are reported in Sec. IV B and Supp. E.3.

## SUPP. C. ESTIMANDS AND STATISTICAL METHODOLOGY

Section IV reports response reconstruction, history-level control benefits, and recovery of finite-amplitude winners. This section defines the corresponding estimands and their units of replication.

Let $y _ { t } ^ { ( s , + a ) }$ and $y _ { t } ^ { ( s , - a ) }$ denote the same scalar or vector observable at time t after changing the loss weight only at source time s from its nominal value $p _ { 0 }$ to $p _ { 0 } + a \mathrm { o r } p _ { 0 } - a$ . The two trajectories use identical minibatches and difer only in the sign of this perturbation. Their central finite-amplitude response is

$$
r _ { y } ( t , s ; a ) = \frac { y _ { t } ^ { ( s , + a ) } - y _ { t } ^ { ( s , - a ) } } { 2 a } ,\tag{S16}
$$

where $a > 0$ is the perturbation amplitude and $\ell = t - s$ is the response lag. On a smooth path, this central secant approaches $\partial { y _ { t } } / \partial { p _ { s } }$ as $a \to 0$ . The measured response can decay while the transported state tangent remains nonzero, because $y _ { t }$ observes only its projection onto the chosen readout.

Let r collect these finite responses over the observation times or readout components being compared, and let $\widehat { r }$ be the corresponding tangent prediction. Reconstruction is summarized by

$$
\mathrm { N R M S E } = \frac { \Vert r - \widehat { r } \Vert _ { 2 } } { \Vert r \Vert _ { 2 } } , \qquad \cos ( r , \widehat { r } ) = \frac { r ^ { \mathsf { T } } \widehat { r } } { \Vert r \Vert _ { 2 } \Vert \widehat { r } \Vert _ { 2 } } .\tag{S17}
$$

Here $\| \cdot \| _ { 2 }$ is the Euclidean norm. NRMSE measures prediction error relative to the observed response norm, while the cosine measures directional alignment independently of overall scale. These are the reconstruction measures used in Figs. 4a and 5b. Finite action selection depends on candidate ordering, so the response metrics are accompanied by discrete action agreement and exhaustive-winner recall.

For an independent training history $h ,$ define the paired decision efect

$$
\Delta _ { h } = \mathcal { T } _ { h } ^ { \mathrm { c o m p } } - \mathcal { T } _ { h } ^ { \mathrm { f u l l } } ,\tag{S18}
$$

where $\mathcal { I } _ { h } ^ { \mathrm { f u l l } }$ is the locked terminal objective under full transport and $\mathcal { I } _ { h } ^ { \mathrm { c o m p } }$ is the objective under the designated comparator. Positive $\Delta _ { h }$ favors full transport; Fig. 2a plots this efect for each comparator. Checkpoints, future tapes, candidate sizes, and control windows within the same history are repeated conditions; the independent history is therefore the inferential unit. Mean-efect intervals resample the history-level values $\{ \Delta _ { h } \}$ with the ordinary nonparametric bootstrap [39]. The exact sign test uses only the number of positive history efects. For 10 positive efects among the 12 four-arm histories, its one-sided probability under equiprobable independent signs is

$$
\operatorname* { P r } \{ { \mathrm { B i n o m i a l } } ( 1 2 , 1 / 2 ) \geq 1 0 \} = { \frac { 7 9 } { 4 0 9 6 } } = 0 . 0 1 9 2 8 7 1 .\tag{S19}
$$

Monte Carlo chains and contiguous token readouts contain local dependence, which is retained by circular-block or contiguous-block resampling [40].

For candidate setting $^ { g , }$ let $k _ { g } ^ { * }$ be the exact finite-amplitude winner from exhaustive rollout and let $\mathcal { S } _ { g , q }$ contain the neutral action and the q nonneutral candidates retained by a ranking rule. Winner recall at nonneutral rollout budge $q$ is

$$
\operatorname { R e c a l l } ( { q } ) = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } \mathbf { 1 } \big \{ k _ { g } ^ { * } \in S _ { g , q } \big \} ,\tag{S20}
$$

where $\mathbf { 1 } \{ \cdot \}$ is the indicator function and $G$ is the number of evaluated settings. Figure 6b reports this quantity at $q = 8$ Candidate-size rows are nested within libraries, multiple libraries share a path, and early and late paths can share a training history. Pooled recall summarizes all 54 Ising and 108 Math–Code settings, while held-path, held-history, and cluster-maximum summaries measure sensitivity to the shared paths and histories. Because these settings are not exchangeable calibration units, the split-conformal framework does not apply directly [44–46]; the reported summarie are empirical calibration diagnostics.

## SUPP. D. FINITE ACTIONS, BRANCH COMPATIBILITY, AND EXACT RERANKING

Section II ranks the seven-action library directly and screens larger libraries before exact reranking. The finite amplitude remainder, clipping-branch criterion, and shortlist rule are specified below.

## D.1. Finite-amplitude remainder and branch screen

Let $\delta x _ { t } = x _ { t } ^ { \mathrm { c a n d } } - x _ { t } ^ { \mathrm { n e u t r a l } }$ be the state displacement between a candidate and the neutral trajectory, and let $\delta { p } _ { t }$ be their loss-weight diference. Expanding the update map $F _ { t }$ around the neutral path gives

$$
\delta x _ { t + 1 } = A _ { t } \delta x _ { t } + B _ { t } \delta p _ { t } + \rho _ { t } , \qquad \| \rho _ { t } \| \leq \frac { M _ { t } } { 2 } \big ( d _ { t } + \vert \delta p _ { t } \vert \big ) ^ { 2 } ,\tag{S21}
$$

where $A _ { t } = \partial _ { x } F _ { t } , B _ { t } = \partial _ { p } F _ { t } , \rho _ { t }$ is the one-step Taylor remainder, $d _ { t } = \| \delta x _ { t } \|$ , and $M _ { t }$ is a local Lipschitz constant for the derivative of $F _ { t }$ . Vector norms are Euclidean, and $\| A _ { t } \|$ is the induced operator norm. If $\left\| A _ { t } \right\| \le \alpha _ { t }$ repeated propagation bounds the terminal state error by

$$
\epsilon _ { x , T } \leq \sum _ { s = t } ^ { T - 1 } \left( \prod _ { r = s + 1 } ^ { T - 1 } \alpha _ { r } \right) \frac { M _ { s } } { 2 } ( d _ { s } + | \delta p _ { s } | ) ^ { 2 } .\tag{S22}
$$

Here $T$ is the terminal step and $\epsilon _ { x , T }$ bounds the diference between the finite state displacement and its tangent prediction. The expression isolates three sources of finite-amplitude error: intervention size, accumulated candidate– neutral separation, and amplification by subsequent Jacobians. Its qualitative dependence guides the choice of small amplitudes and the exact reranking of close candidates.

Gradient clipping makes the update piecewise smooth [25]. Each candidate schedule is connected to neutral by the homotopy

$$
\lambda \in \{ 0 , 1 / 8 , \ldots , 1 \} , \qquad p _ { t + j } ( \lambda ) = p _ { 0 } + \lambda a u _ { j } ,\tag{S23}
$$

where $j \in \{ 0 , \ldots , H - 1 \}$ indexes the update within the window and λ interpolates from neutral to the full action. A candidate is branch-compatible when every sampled point follows the neutral clipping branch and its gradient norm remains at least 0.002 from the unit clipping threshold. The screen separates sampled changes of clipping regime from smooth curvature within a fixed branch. Response reconstruction measures the latter: the tested SGD condition passes the sampled screen but has a large finite-amplitude reconstruction error.

## D.2. Ranking, exact reranking, and empirical shortlist calibration

For candidate schedule $u _ { k }$ , let $\mathcal { T } _ { k } ( a )$ be the exact terminal objective after executing it at amplitude a. Its tangent decomposition is

$$
\begin{array} { r } { \mathcal { I } _ { k } ( \boldsymbol { a } ) = L _ { k } ( \boldsymbol { a } ) + R _ { k } ( \boldsymbol { a } ) , \qquad L _ { k } ( \boldsymbol { a } ) = \mathcal { I } _ { 0 } + \boldsymbol { a } \boldsymbol { g } ^ { \mathsf { T } } \boldsymbol { u } _ { k } , } \end{array}\tag{S24}
$$

where $\mathcal { I } _ { 0 }$ is the neutral terminal objective, $g = ( g _ { t } ^ { \mathrm { f u l l } } , \dots , g _ { t + H - 1 } ^ { \mathrm { f u l l } } )$ contains the source-time derivatives from one reverse adjoint, $L _ { k }$ is the tangent score, and $R _ { k }$ is its finite-amplitude remainder. Dot products $g ^ { \mathsf { T } } u _ { k }$ rank the library before exact rollouts evaluate the retained candidates. If $| R _ { k } | \le \epsilon$ for every candidate, the tangent ordering $L _ { u } < L _ { v }$ is preserved whenever $L _ { v } - L _ { u } > 2 \epsilon$ . Candidates closer than this margin require finite-amplitude comparison.

A narrow fixed shortlist succeeds when tangent-score gaps dominate the candidate-dependent remainders. In the completed development libraries (mixed seeds A and B and the balanced library), $q = 8$ gives exact-action agreement of 15/18, 16/18, and 13/18; the diagnostic width $q = 1 6$ gives 17/18, 17/18, and 18/18. Since the second stage evaluates every retained action exactly, these misses occur when the exhaustive winner lies outside the shortlist. The development results motivate a width that adapts to the observed finite-amplitude ranking error.

Let K be the number of nonneutral candidates, $L _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { k } L _ { k }$ , and $k ^ { * }$ the exhaustive finite-amplitude winner on a completed development setting. The normalized tangent-score gap of that winner is

$$
Z = \frac { \operatorname* { m a x } ( 0 , L _ { k ^ { * } } - L _ { \operatorname* { m i n } } ) } { a ^ { 2 } \sqrt { 2 \log K } } .\tag{S25}
$$

The one-sided empirical order statistic $c _ { 0 . 9 0 }$ of the development values sets the fresh-path tolerance

$$
\epsilon ( K , a ) = c _ { 0 . 9 0 } a ^ { 2 } \sqrt { 2 \log K } , \qquad S ( K , a ) = \{ k : L _ { k } - L _ { \operatorname* { m i n } } \leq \epsilon ( K , a ) \} .\tag{S26}
$$

Exact rollout then selects the lowest-objective action in $\mathcal { S } ( K , a )$ together with the neutral action. The factor $a ^ { 2 }$ follows the leading Taylor remainder, while $\sqrt { 2 \log K }$ supplies an empirical candidate-count correction motivated by lighttailed extreme-value scaling. Recalibrated alternatives proportional to $a ^ { 2 } , a ^ { 2 } { \sqrt { 2 \log K } }$ , and $a ^ { 2 } ( 2 \log K )$ give the same outcome on the available fresh rows. The calibration therefore fixes an empirical shortlist rule; the dependence-aware interpretation is given in Supp. C.

Fixed $q = 1 6$ already agrees with exhaustive search in 52 of the 54 fresh settings, leaving two large-library actions unresolved. Figure S3 shows what the adaptive calibration adds. The 50th of 54 development gaps fixes $c _ { 0 . 9 0 }$ and recalls 50/54 development winners [Fig. S3(a)]. Applied unchanged to new histories, tapes, and library seeds or orderings, the rule reaches $1 8 / 1 8$ agreement in each fresh library [Fig. S3(c)]. Both improvements occur at $K = 1 2 8$ on the same early path, where the exhaustive winners have tangent ranks 23 and 22. The adaptive shortlists expand to 43 and 44 candidates on these cases and include both winners [Fig. S3(b)]. Across all fresh settings, the library-level median widths are 7.5, 7.0, and 9.0. The result is a targeted correction of two large-library ranking failures, obtained by concentrating additional exact rollouts on the afected path.

![](images/c83e16b1550304147eda263bf7fca7ba85256795785e2016df608680bd00dbf1.jpg)

![](images/340c3eefa8b8641d08ad12932e9b284088ad55fef298d56167728645ffa9de47.jpg)

![](images/3ba55b64ee4b8738d2a4dc14fb810f374e654dd4b30bd0f2c4e775b54bb9d58b.jpg)  
FIG. S3. Ising shortlist calibration and fresh-path evaluation. (a) The 50th of 54 ordered development gaps fixes $c _ { 0 . 9 0 } . ( \mathrm { b } )$ Applying this coeficient on fresh paths gives the median and range of the adaptive shortlist size for each K and library. (c) Adaptive reranking agrees with the exhaustive action in $1 8 / 1 8$ settings per library; fixed $q = 1 6$ misses one $K = 1 2 8$ action in each mixed library.

## SUPP. E. ADDITIONAL EVIDENCE FOR RESPONSE, CONTROL, AND RANKING

The response, control, and cost measurements underlying Sec. IV and Figs. 2–6 of the main text are collected here. Each subsection expands one main-text result and relates its numerical diagnostics to the corresponding dynamical or decision-level interpretation.

## E.1. Response reconstruction, optimizer memory, and scale

The Math–Code full-state tangent reconstructs paired finite responses without a fitted response kernel. At 0.3M, the pathwise NRMSE and cosine similarity are 0.0239 and 0.9997 for the antisymmetric readout $m = ( L _ { A } - L _ { B } ) / 2$ and 0.0490 and 0.9988 for the symmetric readout $e = ( L _ { A } + L _ { B } ) / 2$ . The corresponding 1M values are 0.0138/0.9999 and 0.0304/0.9995. Thus, on a specified eight-step path, the local derivative remains accurate at both tested scales.

The optimizer interventions identify how this response is carried. Among the 62 moment-bearing cells that remain on a common smooth clipping branch, full-state transport outperforms moment-tangent deletion in every case. When $\beta _ { 1 }$ is reduced, the projection of the first-moment response onto its baseline direction falls to 0.4238 of the baseline value. Reducing $\beta _ { 2 }$ increases the norm of the second-moment response by a factor of 20.70. The common-amplitude SGD comparison behaves diferently: its symmetric-response RMS is 11.5 times the AdamW baseline and its NRMSE is 1.46, even though the clipping branch does not change. The fixed perturbation therefore produces a much larger efective displacement under SGD, for which the local linearization is no longer accurate. The finite-amplitude scale of this displacement is distinct from the AdamW coordinate-deletion result.

The accurate local derivative at 1M does not by itself ensure consistent control over a long sufix. Repeated decisions continually separate the controlled and neutral trajectories, changing the subsequent Jacobians about which each window is linearized. Nonlinear remainders can therefore accumulate even when every individual eight-step derivative is accurate, which accounts for the lower history consistency observed in the longer 1M closed-loop run.

A direct lag-resolved view comes from the Ising–CNN experiment. Exact finite responses and the full-state tangent remain nearly indistinguishable across all eight lags in Fig. S4. Moment deletion agrees at the first lag, before retained moment perturbations have propagated through later updates, and then decays too rapidly. Its NRMSE rises from 0.002115 to 0.6923 for m and from 0.001580 to 0.6871 for e. The separation after the first update locates the delayed response in the AdamW moment coordinates.

![](images/956e386cfb76b94f402587d938b95eedaed7f6844670cb4c253df057a9ad50c6.jpg)  
FIG. S4. Lag-resolved Ising–CNN response and moment-tangent deletion for (a) the antisymmetric readout m and (b) the symmetric readout e. The full-state tangent follows the exact finite response in both panels. Deleting perturbations in the AdamW moment coordinates leaves the first lag unchanged but removes most of the response transmitted to later updates. Curves show path averages; NRMSE is computed pathwise.

## E.2. Four-arm delayed-transport confirmation

The prospective four-arm experiment compares full transport with the Adam-aware immediate derivative, neutral training, and VGA on the same 12 histories. The primary full-minus-immediate efects, expressed as reductions in held-out loss, are

$$
( 6 . 0 9 8 , 9 . 8 6 6 , 7 . 3 9 5 , 5 . 2 5 5 , 6 . 0 2 6 , 3 . 4 1 8 , - 2 . 0 1 2 , 5 . 9 3 5 , 8 . 6 2 8 , 4 . 8 0 0 , - 1 . 4 3 0 , 2 . 5 2 2 ) \times 1 0 ^ { - 4 } .\tag{S27}
$$

Their mean is $4 . 7 0 8 2 \times 1 0 ^ { - 4 }$ , with 10/12 positive efects and a history-bootstrap 95% interval of $\left[ 2 . 6 5 9 6 , 6 . 5 4 4 7 \right] \times 1 0 ^ { - 4 }$ The full-minus-neutral mean is $7 . 8 0 5 2 \times 1 0 ^ { - 4 }$ , with 11/12 positive histories and an interval of $\left[ 4 . 9 6 8 5 , 1 0 . 9 4 3 0 \right] \times 1 0 ^ { - 4 }$

Full exceeds VGA in all 12 histories, with a mean diference of $7 . 7 1 9 2 \times 1 0 ^ { - 4 }$ and an interval of $[ 5 . 1 3 8 8 , 1 1 . 2 4 0 0 ] \times 1 0 ^ { - 4 }$ The prespecified full–immediate and full–neutral criteria both pass.

The immediate controller also improves over neutral, by $3 . 0 9 7 0 \times 1 0 ^ { - 4 }$ on average, and has positive history efects in $9 / 1 2$ cases. VGA improves by $8 . 6 1 \times 1 0 ^ { - 6 }$ and is positive in $4 / 1 2 .$ Full and immediate both act in all 96 windows, yet choose the same schedule in only 36. Their diferent outcomes therefore arise from the ordering of temporal schedules under matched intervention frequency. The full–immediate increment is 60.3% of the full–neutral benefit; this ratio summarizes the combined efect of delayed propagation, terminal readout, and discrete action selection.

Readout sampling is checked separately. The locked 16-batch-per-domain evaluation gives mean full-minus-immediate benefits of $7 . 3 9 0 5 \times 1 0 ^ { - 4 }$ on Math and $2 . 0 2 5 9 \times 1 0 ^ { - 4 }$ on Code. A subsequent audit covers 99.984% of Math and 99.989% of Code target tokens using disjoint 64-token windows grouped into contiguous 4096-token superblocks. The nearly exhaustive mean benefits are $6 . 2 5 4 8 \times 1 0 ^ { - 4 }$ on Math, $\bar { 2 } . 7 3 \bar { 1 } 8 \times 1 0 ^ { - 4 }$ on Code, and $4 . 4 9 3 3 \times 1 0 ^ { - 4 }$ for their equal average; $1 1 / 1 2$ combined history efects are positive. A 10,000-replicate contiguous-superblock bootstrap gives $[ 4 . 0 2 6 , 4 . 9 8 \dot { 2 } ] \times 1 \dot { 0 } ^ { - 4 }$ , while joint resampling of histories and superblocks gives $[ 2 . 5 9 \bar { 6 } , 6 . 1 5 5 ] \times 1 0 ^ { - 4 }$ . Agreement with the locked result shows that the measured benefit extends across the test partition.

To distinguish local score changes from state divergence caused by earlier actions, a deterministic replay evaluates both complete candidate-score vectors at every state along the locked full and immediate trajectories. The replay recovers all 96 actions of each controller. Let $i = \arg \operatorname* { m i n } _ { k } S _ { k } ^ { \mathrm { i m m } }$ be the immediate-score winner over branch-compatible candidates, and define its strongest challenger under the full score by

$$
k ^ { * } = \arg \operatorname* { m i n } _ { k \neq i } \left( S _ { k } ^ { \mathrm { f u l l } } - S _ { i } ^ { \mathrm { f u l l } } \right) .\tag{S28}
$$

Define

$$
M = S _ { k ^ { * } } ^ { \mathrm { i m m } } - S _ { i } ^ { \mathrm { i m m } } , \qquad D = \left( S _ { k ^ { * } } ^ { \mathrm { f u l l } } - S _ { i } ^ { \mathrm { f u l l } } \right) - M .\tag{S29}
$$

Here M is the immediate-score margin separating the selected action from the challenger, and D is the change in that relative score produced by delayed transport. With no score ties, the immediate winner changes exactly when $M + D < 0$ . Ranking and executed-action changes coincide: the scores disagree in $6 0 / 9 6$ states along either locked path, and 59 changed history–window locations are shared. At the 12 common initial states, before closed-loop trajectories diverge, the two scores already choose diferent actions in $6 / 1 2$ cases.

The replayed states separate exactly at $D = - M$ in Fig. S5(a): points below this line receive a future-map correction large enough to overturn the immediate winner. The normalized immediate margins have similar medians when the winner is retained or changed, 1.60 and 1.55, whereas the median centered full–immediate score cosine falls from 0.814 to $0 . 1 3 0 \ [ \mathrm { F i g . \ S 5 ( b ) } ]$ . Comparable margins together with the loss of score-vector alignment associate the action changes with a broader reordering of the candidate library. The common-state disagreements in Fig. S5(c) occur before the closed-loop paths separate and isolate the direct contribution of delayed transport to the ranking.

![](images/ee9959ec6e0d89cb3dc366945dae71bd0bca522359ec68840b53c8334d48aa3f.jpg)

![](images/0dc278e17cf71494eec608e101c2fb1d87c87bd987500a0493c556f2940baad8.jpg)

![](images/0a173a2b68571b7fe045123f29e1e8de08791df9284b82ee772ddf67d8af120c.jpg)  
FIG. S5. Same-state decision geometry in the four-arm Math–Code experiment. (a) Gray points retain the immediate winner, orange points change it, and the dashed line is the exact multicandidate boundary $D = - M$ . (b) Centered full–immediate score cosines for the retained- and changed-winner groups; box lines give medians and quartiles. (c) The two rankings disagree at 60/96 states along each replayed trajectory and at 6/12 common initial states.

## E.3. Future-path conditioning and anchored Ising control

The crossed audit varies the current checkpoint and the next eight minibatches independently. Centered schedule score variation decomposes into 10.3% state–schedule, 17.8% tape–schedule, and 71.9% state–tape–schedule interaction.

On each realized tape, full actions improve $8 8 / 9 6$ cells and all four histories, and beat the immediate comparator in 57/61 informative cells, where “informative” denotes a nonzero pairwise benefit diference. Averaging over the other tapes gives a leave-one-tape-out mean benefit $\mathrm { o f - 1 . 6 1 \times 1 0 ^ { - 5 } }$ and positive history means in only 1/4 cases. The same optimizer state thus assigns diferent schedule orderings to diferent upcoming minibatch sequences, and averaging those orderings can cancel the preference relevant to the realized path.

With the future path fixed, full actions improve $6 0 / 7 2$ cells and all six history means. They beat the immediate comparator in 30/41 informative cells and the memory-deleted policy in $2 2 / 3 2 .$ , with positive history means in five of six cases for each comparison. Candidate-contrast amplitudes transfer imperfectly from validation to test data [NRMSE 0.474 and cosine 0.892 in Fig. S6(a)], yet the selected actions have positive test utility [Fig. S6(b)]. Action selection uses the local ordering near the best candidate, while NRMSE accumulates errors over the entire contrast vector. The history-level diferences in Fig. S6(c) show that delayed propagation and moment transport improve the selected action in five histories; the sixth gives small negative diferences for both comparisons.

![](images/8d77cadcee8d243ae910550e1487f07e59615960bd1a81f4c21652685270726d.jpg)

![](images/dd239443bb0593e2996779a0a27558b0c203c9ff62f16514b4964db23706177b.jpg)

![](images/9f3eb843dbfdbfa8af4dae1c7a228eebe578de0093f018bb9737bed7e343e9fa.jpg)  
FIG. S6. Specified-future-path confirmation. (a) Validation scores preserve useful local action ordering despite imperfect calibration of the full test-contrast vector; red points denote the locked actions. (b) Full transport gives the largest mean independent-test benefit among the four policies. (c) Full exceeds memory deletion and the immediate derivative in five of six history means. Error bars in (b) show standard errors across histories.

The specified-path Ising design also permits a prospective test against the independently constructed Binder anchor. This confirmation retains the fixed Binder artifact while replacing the neural histories, future tapes, calibration chains, and test chains. Full and exhaustive policies agree in 11/12 cells. On the new chains, the mean absolute neural-root error $| \widehat { T } _ { N } - T _ { B } |$ falls from 0.02435 to 0.01880; all six histories improve, with a block-aware interval of [0.00258, 0.00839] for the reduction. Candidate-contrast NRMSE remains above its calibration threshold because it measures the amplitudes of all candidate efects, whereas action agreement depends on the lowest-risk candidate. The more precise Binder estimate combines a direct thermodynamic observable across several lattice sizes; the classifier-dependent $L = 1 6$ crossing $\widehat { T } _ { N }$ supplies the diferentiable training readout controlled here.

## E.4. Equal-budget winner recall, transfer, and cost

For larger libraries, each score retains neutral and selects q nonneutral candidates for exact rollout. At $q = 8 .$ exhaustive-winner recall for full, memory-deleted, immediate, and random ranking is $4 8 / 5 4 , 4 3 / 5 4 , 2 0 / 5 4$ , and 0.2959 in Ising, and 108/108, 98/108, 44/108, and 0.1550 in Math–Code. This ordering holds for every tested $q \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ On fresh Ising paths, the adaptive shortlist also includes and recovers the exhaustive winner in all 54 settings.

Three fresh Math–Code libraries test transfer to two random zero-sum spheres and the complete balanced binary library, all with matched aggregate exposure. Fixed $q = 1 6$ recovers all 108 exhaustive winners; the adaptive coeficient recovers 100/108, with 32/36 in sphere A and $3 4 / 3 6$ in each remaining library. Their upper-tail gaps extend beyond the development distribution $[ \mathrm { F i g . \ S 7 ( a , b ) } ]$ , showing that calibrated width depends on local score spacing and library geometry. The locked adaptive actions nevertheless improve token-disjoint loss in all six histories, with mean benefit $6 . 4 6 1 2 \times 1 0 ^ { - 4 }$ and a history-bootstrap interval of $[ 5 . 4 4 5 6 , 7 . 6 5 4 5 ] \times 1 0 ^ { - 4 } ~ [ \mathrm { F i g . ~ S 7 ( c ) } ]$

![](images/02ea1c48b8325c27d3e3570c4d40b3c97b4c05b8d4524e63a520232ead4cee87.jpg)

![](images/47ad0b6b4dd8ddf0076b1b9b7e6154fa322ff0090e4386b8ef6affdb35cf0a96.jpg)

![](images/8297c5777ec6d05db086c31467dc218897853f0954c705de5bd781eefd878d12.jpg)  
FIG. S7. Math–Code transfer of transport-guided exact reranking. (a) Ordered normalized winner gaps for development and fresh settings, together with the frozen Math–Code calibration coeficient. (b) Fixed q = 16 recovers every exhaustive winner; the adaptive rule gives 88.9%, 94.4%, and 94.4% agreement in the three libraries. (c) Every adaptive action improves token-disjoin test loss; the horizontal line and band show the history mean and its 95% bootstrap interval.

Winner recall does not include the one-time cost of calibrating the adaptive cutof. To account for it, one future-path bundle is defined as the three largest libraries at a checkpoint–tape pair, comprising $1 2 8 + 1 2 8 + 7 1 = 3 2 7$ candidate paths. Ising calibration uses 1962 exact paths and the deployed adaptive rule evaluates 59 paths per bundle on average; Math–Code calibration uses 3924 exact paths and its adaptive rule evaluates six. Charging the complete calibration stage up front gives

$$
N _ { \mathrm { b r e a k } } ^ { \mathrm { I s i n g } } = \frac { 1 9 6 2 } { 3 2 7 - 5 9 } = 7 . 3 2 , \qquad N _ { \mathrm { b r e a k } } ^ { \mathrm { M C } } = \frac { 3 9 2 4 } { 3 2 7 - 6 } = 1 2 . 2 2\tag{S30}
$$

future-path bundles for the reused library structures. After these points, the saved exact candidate paths, 327 − 59 or $3 2 7 - 6$ per bundle, exceed the one-time calibration count. This accounting concerns nonlinear candidate rollouts; the full per-window timing below also includes diferentiation, action execution, and readout.

TABLE S1. Direct 0.3M four-arm CPU cost. Entries are medians of 12 Apple-M4 CPU measurements in seconds; RSS is peak resident memory in GiB from fresh worker processes. “Screen” is the branch-compatibility evaluation, “action” is the selected eight-step update, and “readout” is the held-out evaluation. Component medians need not sum to the median total.
<table><tr><td>Method</td><td>differentiation</td><td>screen</td><td>action</td><td>readout</td><td>total</td><td>RSS</td></tr><tr><td>Ordinary training</td><td>0</td><td>0</td><td>0.115</td><td>0.031</td><td>0.145</td><td>0.433</td></tr><tr><td>VGA</td><td>0.930</td><td>5.269</td><td>0.123</td><td>0.031</td><td>6.360</td><td>0.785</td></tr><tr><td>Adam-aware immediate</td><td>1.540</td><td>5.382</td><td>0.123</td><td>0.031</td><td>7.073</td><td>0.560</td></tr><tr><td>Full, forward</td><td>4.676</td><td>4.603</td><td>0.106</td><td>0.029</td><td>9.402</td><td>0.566</td></tr><tr><td>Full, adjoint</td><td>0.600</td><td>4.848</td><td>0.107</td><td>0.030</td><td>5.657</td><td>1.522</td></tr></table>

Forward full, used in the confirmation, takes 1.33 times the total time of the immediate controller because it propagates each source sensitivity through the complete eight-step state map. The adjoint obtains all source-time derivatives in one backward sweep and selects the same action as forward mode on all four timed paths. It takes 0.80 times the immediate total, while retaining more trajectory state and reaching 2.72 times its peak RSS. The two implementations therefore exchange forward-mode time for adjoint memory on this CPU benchmark.

Table S2 reports the complete per-window costs from the earlier deployment-oriented audit, with $T _ { \mathrm { c t r l } } = T _ { \mathrm { a d j o i n t } } +$ $T _ { \mathrm { e x a c t } } + T _ { \mathrm { a c t i o n } } + T _ { \mathrm { r e a d o u t } }$ and $T _ { 0 } = T _ { \mathrm { t r a i n } } + T _ { \mathrm { r e a d o u t } }$

TABLE S2. Complete measured Math–Code CPU cost per eight-step window. Times are seconds.
<table><tr><td>Scale</td><td>train</td><td>readout</td><td>adjoint</td><td>exact</td><td>action</td><td>control</td><td> $T _ { \mathrm { c t r l } } / T _ { 0 }$ </td></tr><tr><td>0.3M</td><td>0.0937</td><td>0.0275</td><td>0.4859</td><td>0.7031</td><td>0.0945</td><td>1.3110</td><td>10.82</td></tr><tr><td>1M</td><td>0.1751</td><td>0.0494</td><td>0.9201</td><td>1.2380</td><td>0.1806</td><td>2.3881</td><td>10.64</td></tr></table>

The roughly 10.7-fold per-window ratio is dominated by the adjoint and the exact candidate rollouts, which evaluate the specified future map several times before an action is chosen; ordinary training executes that map only once. This ratio measures the complete prospective comparison performed at every window, of which the AdamW update is only one component.

Three additional measurements clarify the comparator and cost results in Fig. S8. The early full–VGA diference changes sign across histories, although full transport improves over neutral on average [Fig. S8(a)]. VGA measures validation-gradient alignment in parameter space, so its diference from full transport combines current AdamW update geometry with subsequent state propagation. The main confirmation uses the Adam-aware immediate derivative to hold the current update fixed. In the complete timing, exact screening and adjoint diferentiation dominate the per-window cost at both scales [Fig. S8(b)]. Separately calibrated constant, $\sqrt { 2 }$ log $\overline { { K } }$ , and 2 log K factors give the same fresh-winner recall in each system [Fig. S8(c)]. For the tested values of K and the correlated library families, recalibration absorbs their numerical diferences, leaving the candidate-count dependence unidentified.

![](images/8b29cd83efe763b9c81d20cd9a13ca9d5bb61593948daf5acc1e047482b39bed.jpg)

![](images/87beea8f3f62f3d8aa616283e157cc866bac8207aab07f73811545a168a9c1e4.jpg)

![](images/70089e7d43fdd441cf5325ceca95b199f2abc87e5b5acc6b16517b722ec3556d.jpg)  
FIG. S8. External comparator, complete cost, and candidate-count sensitivity. (a) History-level test-loss benefits in the early full, VGA, and neutral comparison; squares and intervals show means and 95% history-bootstrap intervals. (b) Measured CPU time per eight-step window, decomposed into adjoint, exact screening, selected-action training, and readout. Labels give the ratio to ordinary training and readout. (c) Fresh-winner recall after separately calibrating three candidate-count factors; the dashed line marks 90% recall.

## SUPP. F. EMPIRICAL SCOPE AND OBSERVED BOUNDARIES

The main experiments establish local decisions on specified eight-step paths. Additional tests vary the training scale and duration, the future path and readout, the optimizer, the candidate library, and the evaluation cost. Table S3 groups the resulting boundaries by their observed source.

The dynamical boundaries arise at diferent points in the response construction. One-window accuracy at 1M concerns a derivative about the current path; repeated control moves the state and changes the Jacobians used in later windows. Removing the realized future tape changes the drive entering the propagator, and the 71.9% state–tape– schedule interaction then dominates the action ordering. Changing the readout alters the terminal projection of the transported tangent, so useful local ordering can coexist with inaccurate transfer of the full contrast vector. In the SGD comparison, the same nominal pulse produces a symmetric-response RMS 11.5 times the AdamW value, moving the finite experiment outside the accurately reconstructed amplitude range.

The two Math–Code comparator cohorts answer diferent questions. In the early six-history experiment, full transport exceeds VGA in $3 / 6$ histories, below the stated history-consistency criterion. VGA measures validation-gradient alignment in parameter-space geometry, whereas the executed update contains AdamW preconditioning and moment state. Its diference from full transport therefore mixes current-update geometry with delayed propagation. The later 12-history experiment uses new histories and an Adam-aware derivative of the same current update, leaving subsequent state transport as the principal distinction. This follow-up protocol was fixed before those new histories were evaluated.

Dependence within a training history also afects shortlist calibration. Leave-one-history-out winner recall is $4 4 / 5 4$ for Ising and 97/108 for Math–Code, below the corresponding pooled summaries. For a 90% empirical order statistic, the index $\lceil ( n + 1 ) 0 . 9 0 \rceil$ is 7 when $n = 6 .$ , exceeding the number of Ising paths. At n = 12 for Math–Code, the index is 12, so the cluster-level value is the sample maximum and is substantially larger than the setting-level coeficient. The shortlist threshold is consequently treated as an empirical risk budget, with no finite-sample coverage level assigned.

TABLE S3. Empirical boundaries of the reported results and their observed sources.
<table><tr><td>Dimension</td><td>Observed boundary</td><td>Supported here</td><td>Source of boundary</td></tr><tr><td>Scale and duration</td><td>1M: accurate local response; 9/12 One-window reconstruction ex- Scale and suffix length vary positive histories; one path drifts fur- tends to 1M ther by 224 steps</td><td></td><td>together; repeated actions change later states and Jaco- bians</td></tr><tr><td>Unknown future</td><td>Leave-one-tape-out benefit -1.61 × Utility on committed future State-tape-schedule interac-  $1 0 ^ { - 5 }$  1/4 histories positive</td><td>paths</td><td>tion contributes 71.9% of score variation</td></tr><tr><td>Cross-readout</td><td>NRMSE/cosine = 0.474/0.892; cali- Locked actions can improve test Action choice uses local order- bration gates missed</td><td>loss</td><td>ing; NRMSE weights all con- trast amplitudes</td></tr><tr><td>Optimizer family</td><td>SGD response is inaccurate at the Moment transport in the tested The same pulse gives 11.5× the common amplitude</td><td>momentum and AdamW set- AdamW symmetric-response tings</td><td>RMS</td></tr><tr><td></td><td>External comparator Early full-VGA difference is positive VGA mixes current-update ge- The VGA score omits AdamW in 3/6 histories</td><td>ometry with delayed transport preconditioning and moment-</td><td>state updates</td></tr><tr><td></td><td>candidate-count forms tie</td><td>Shortlist calibration Sphere A gives 32/36; recalibrated Fixed q = 16 exact reranking Winner-gap geometry changes gives 108/108</td><td>across libraries; recalibration absorbs the tested K factors</td></tr><tr><td>Physical benchmark</td><td>ceeds the Binder error</td><td>Neural-root error decreases but ex- Control of a physics-anchored Binder uses a direct observable neural readout</td><td>at several L; the neural root uses one L = 16 classifier</td></tr><tr><td>Efficiency</td><td>10.8× ordinary training per window date rollouts</td><td>Complete CPU control costs 10.6- Ranking reduces exact candi- Adjoint construction and ex-</td><td>act screening repeat the future map before each action</td></tr></table>

## SUPP. G. REPRODUCIBILITY AND DATA PROVENANCE

The public repository provides the training and analysis code, configuration files containing the reported input parameters, and the plotting scripts used for the figures. Run identifiers and file hashes bind each result to its training state, future minibatch path, candidate library, selected action, independent readout, and implementation.

Each confirmation fixes the protocol and implementation before unused histories and future tapes are assigned; the resulting checkpoints are hashed. Controller predictions, branch records, actions, and terminal states enter the experiment lock before the independent readout is evaluated. Separate identifiers for histories, checkpoints, tapes, libraries, and readouts keep repeated conditions grouped by training history. Sensitivity analyses have separate records linked to the same lock.

A claim-to-artifact map links each figure and table to its configuration, lock, machine-readable result, and plotting script. Environment records, checkpoint and data manifests, verification tests, and a protocol ledger record the provenance of the released analyses and the inputs used to regenerate the reported summaries.

Late shortlist and timing records use PyTorch 2.12.0; several earlier response records use PyTorch 2.13.0, with the version stored per run. Timing uses float32, deterministic algorithms where supported, four CPU threads, and an Apple M4 CPU. Dynamic CPU frequency makes absolute seconds hardware-specific; component times and ratios support comparisons within the recorded environment. Original code uses the MIT License; upstream licenses govern corpora and checkpoints.