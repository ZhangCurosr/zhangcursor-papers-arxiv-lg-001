# Characterizing Replay Retention Under Dynamics Shift in Model-Based Reinforcement Learning

Everest Yang<sup>∗</sup>, Skye Thompson, George D. Konidaris Brown University, Providence, RI, USA

Abstract— Adapting to changes in robot dynamics requires learning from new data without discarding experience that may still be useful. In continual model-based reinforcement learning (RL), replay collected before a dynamics change can slow adaptation, while removing it unnecessarily reduces available training data and can be especially costly if earlier dynamics return. We study when recent transitions are preferable to the full replay history. Two quantities characterize this tradeoff: change magnitude and age–staleness area under the curve (AUC), measuring how well transition age separates stale from fresh data. Forgetting stale data helps after large permanent shifts but hurts when dynamics recur and older data becomes useful again. Choosing a replay strategy therefore depends on predicting when older data will help or hurt. We test these effects across two locomotion morphologies, two model-based RL algorithms, and Real-World RL benchmark perturbations. Because ground-truth staleness labels are unavailable on deployed robots, we evaluate whether an estimator built from interaction data can still provide the quantities needed to choose a replay strategy after permanent changes. Our results show that replay retention depends on change magnitude and on how the dynamics evolve.

## I. INTRODUCTION

Reinforcement learning (RL) with learned world models is a common approach to continuous control when an agent must adapt as its dynamics change. Methods such as DreamerV3 [1] and TD-MPC2 [2] learn from replayed experience [3] to predict the consequences of actions. Adapting a learned world model can become difficult when the robot’s dynamics shift during continued training, for example from actuator damage or a change in payload. Older transitions may then no longer reflect the robot’s current behavior. Training on outdated transitions can slow down adaptation, while removing them may discard data that could still be useful under the current dynamics.

The value of older replay data depends on how the dynamics evolve over time (Fig. 1). After a lasting change, such as a new payload, transitions collected beforehand can remain mismatched. If earlier dynamics later return, those same transitions can become useful again. The utility of older replay data therefore depends on both the size and temporal structure of the dynamics change.

We study how transition staleness affects replay usefulness by comparing replay over the full history with fixed windows of recent data. We focus on two properties of the dynamics change. The first is its magnitude, which captures how much the underlying dynamics change relative to those represented by older transitions. The second is the relationship between transition age and continued relevance, which we measure using age–staleness AUC. Age is useful for replay selection when older transitions are more likely to come from outdated dynamics, but it becomes less informative when the dynamics recur and older transitions once again match the dynamics the agent is experiencing.

We study three dynamics patterns: permanent changes, recurring changes, and a no-change control. Within these settings, we vary change magnitude and recurrence. Separate controls vary the size of the replay window and the total number of transitions the replay buffer can store, testing how each affects performance across these dynamics settings. We test these effects across two locomotion morphologies, two model-based RL algorithms, and perturbations from the Real-World RL (RWRL) benchmark. We then ask whether interaction data can provide enough information to decide when older replay should be discarded. For permanent changes, we also show that we can estimate change magni tude and age–staleness from actuator response alone, without access to the simulator’s dynamics parameters, suggesting that these signals could guide replay selection on real-world systems. Our contributions are as follows:

1) A characterization of when older replay data should be retained based on change magnitude and the relationship between transition age and staleness.

2) Evidence that the same dynamics change can favor forgetting when it persists and retention when it recurs. This shows that change magnitude alone is not enough to select an effective replay strategy.

3) A retrospective actuator-response analysis that detects dynamics changes and decides whether older replay should be discarded using estimated change magnitude and age–staleness, with staleness inferred from observed actuator responses rather than ground-truth dynamics parameters.

## II. RELATED WORK

When dynamics change during continued learning, adaptation depends on how the model is updated and which past transitions remain available for replay. Prior studies have examined replay retention and sampling, including the effects of transition age and buffer size on control [4], [5], [6]. Local forgetting removes samples near parts of the state space that have changed so a model-based agent can adapt without discarding data elsewhere [7]. Curious Replay instead prioritizes transitions that are still informative for improving the world model [8], and Continual-Dreamer studies selective replay as a way to reduce forgetting while preserving transfer across tasks [9]. World Models with Augmented Replay (WMAR) maintains a more diverse long-term replay distribution [10]. Our work focuses on when age-based forgetting is useful under permanent versus recurring dynamics changes. More selective replay strategies can avoid some of the losses from a fixed window of recent data. We focus on the separate question of when transition age is useful for deciding which transitions to retain.

![](images/437d3741fd3b2868a3fb8d2e76e51518530d1359179c1f7836be134058fc0eb0.jpg)  
Fig. 1. Overview of replay retention under changing dynamics. Interaction data populate replay and can also be used retrospectively to estimate whether older transitions should be retained or discarded. The resulting replay choice determines which data would be used to update the world model and policy. Under permanent changes, older data may remain mismatched, while under recurring dynamics it can become useful again.

Another approach is to adapt the model or policy directly rather than modify replay. Context-aware dynamics models infer latent information about the current dynamics [11]. Rapid motor adaptation instead uses recent interaction history to adapt locomotion policies online [12]. Meta-learned dynamics models can adapt from recent data after changes in morphology, terrain, or payload [13]. Dynamics randomization exposes policies to variations in physical parameters during training to improve robustness [14], while other meta-RL methods learn priors or latent representations that support rapid adaptation [15], [16]. Cully et al. use a map of highperforming behaviors to recover from robot damage [17]. Our setting instead keeps the learning procedure fixed and studies which previously collected transitions should remain available as training continues. The same model continues learning as the dynamics change, rather than relying on domain randomization or a separate model for each regime.

Prioritized replay samples transitions according to temporal-difference error [18]. Density-ratio importance weighting favors transitions that better match the current policy distribution [19]. Methods with adaptive windows choose the window size from the data stream [20].

## III. PROBLEM FORMULATION

## A. Continual Learning and Retention

We consider a model-based RL agent that continues updating its world model and policy after the environment dynamics change [21]. At each step, the agent acts, stores the resulting transition, and samples from replay for further updates, while the learned model itself is never reset. The replay strategy determines which stored transitions remain available for training.

Passive replay samples from the full available buffer, with a capacity of one million transitions. Recency restricts replay to the newest $w = 1 0 , 0 0 0$ transitions. Let $J _ { s }$ denote the mean episodic return for replay strategy s, averaged over evaluations during the 90k frames following the dynamics change. Each evaluation uses ten episodes. Our primary comparison is the paired difference in return after the dynamics shift

$$
D = J _ { \mathrm { r e c e n c y } } - J _ { \mathrm { p a s s i v e } } .\tag{1}
$$

$D > 0$ favors Recency. We call D the Recency advantage.

## B. Staleness Labels and Age Ranking

In our controlled experiments, the dynamics take one of two known regimes. Each transition is labeled by the $\mathrm { { d y } \mathrm { { - } } }$ namics regime in which it was collected, such as the original dynamics or a shifted regime with reduced actuator gain. Let $r _ { t }$ denote the regime active at time $t . \mathrm { ~ A ~ }$ transition collected at time τ is stale at time t when $r _ { \tau } \neq r _ { t } .$ . Here, staleness is defined from the known dynamics regime, giving us groundtruth staleness labels for the controlled experiments.

Age-based forgetting is useful only when transition age is informative about staleness. We measure how well transition age $A _ { t } ( \tau ) = t - \tau$ separates stale from fresh data using the area under the receiver operating characteristic curve (AUC), which we define for a buffer at time t as

$$
\begin{array} { r l } & { \mathrm { A U C } _ { \mathrm { a g e } } ( t ) = \operatorname* { P r } [ A _ { t } ( x _ { \mathrm { s t a l e } } ) > A _ { t } ( x _ { \mathrm { f r e s h } } ) ] } \\ & { \phantom { \quad \quad } + \frac { 1 } { 2 } \operatorname* { P r } [ A _ { t } ( x _ { \mathrm { s t a l e } } ) = A _ { t } ( x _ { \mathrm { f r e s h } } ) ] . } \end{array}\tag{2}
$$

This is the probability that a random stale transition is older than a random fresh one [22]. An AUC of 1 means stale transitions are always older than fresh ones, while 0.5 corresponds to chance ranking. The measure is undefined when the buffer contains only stale or only fresh transitions.

Under recurring dynamics, a transition can be stale at one point in training and useful again later when its dynamics return. Because this relationship changes over time, we report age–staleness AUC averaged over the evaluation window after the change. Each episode is assigned to the dynamics regime active when that episode ends.

## IV. CHARACTERIZING REPLAY RETENTION UNDER DYNAMICS SHIFT

## A. Bias–Variance Trade-offs in Replay Retention

Consider estimating a scalar component of the current dynamics from $n _ { o }$ observations of the old regime and $n _ { f }$ fresh observations, each with noise variance $\sigma ^ { 2 }$ . Let the old observations be offset from the current parameter by M. If the old data are weighted by $\beta ,$ the resulting mean-squared estimation error is

$$
\mathcal { E } ( \beta ) = \beta ^ { 2 } M ^ { 2 } + \beta ^ { 2 } \frac { \sigma ^ { 2 } } { n _ { o } } + ( 1 - \beta ) ^ { 2 } \frac { \sigma ^ { 2 } } { n _ { f } } .\tag{3}
$$

The optimal weight on old data is

$$
\beta ^ { \star } = \frac { \sigma ^ { 2 } / n _ { f } } { M ^ { 2 } + \sigma ^ { 2 } / n _ { o } + \sigma ^ { 2 } / n _ { f } } .\tag{4}
$$

When the mismatch is small, retaining old data mainly reduces variance. As M grows, the optimal weight on old data falls toward zero, so larger changes favor forgetting. For a fixed replay window, the benefit of forgetting also depends on how much stale data remains inside that window. Let $\rho$ denote the stale fraction in the full buffer and $\rho _ { w }$ the stale fraction in a window of size w. Relative to training on all N observations in the full buffer, restricting training to the window changes the mean-squared estimation error by

$$
\Delta \mathcal { E } = ( \rho ^ { 2 } - \rho _ { w } ^ { 2 } ) M ^ { 2 } - \sigma ^ { 2 } \left( \frac { 1 } { w } - \frac { 1 } { N } \right) .\tag{5}
$$

The first term captures the reduction in bias from removing stale data, while the second captures the variance cost of using fewer samples. Forgetting is beneficial when the former outweighs the latter. Related analyses of moving windows after a change capture the same bias–variance trade-off between retaining stale data and reducing the number of training samples [23]. For $\rho > \rho _ { w }$ , the error difference is zero at the mismatch threshold

$$
M ^ { \star } = \sigma \sqrt { \frac { 1 / w - 1 / N } { \rho ^ { 2 } - \rho _ { w } ^ { 2 } } } .\tag{6}
$$

Algorithm 1 Retrospective Replay Selection   
1: Input: Interaction trajectory H   
2: Output: Replay strategy S   
3: K ← ESTIMATEACTUATORRESPONSE(H)   
4: ˆs ← DETECTSTALESEGMENTS(K)   
5: $\widehat { \mathrm { A U C } } _ { \mathrm { a g e } }$ ← AGESTALENESSAUC(H, ˆs)   
6: mˆ ← ESTIMATEMAGNITUDE(K, ˆs)   
7: if $\widehat { \mathrm { A U C } } _ { \mathrm { a g e } } \geq 0 . 8 7$ and mˆ ≥ 0.40 then   
8: S ← RECENCY   
9: else   
10: S ← PASSIVE   
11: end if   
12: return S

Recency is useful only when removing stale data outweighs the cost of training on fewer samples. Small dynamics changes leave older transitions relatively useful, while low age–staleness separation means a window of recent data discards useful experience along with stale data. Age–staleness AUC does not determine $\rho _ { w }$ directly, but it measures how well age distinguishes stale from fresh transitions.

We vary change magnitude, recurrence, window size, and buffer capacity separately to test how the performance tradeoff between retaining potentially stale data and training on fewer samples changes across these conditions. The model abstracts away policy updates, nonlinear function approximation, and the changing data distribution during training. Its role is to motivate when removing stale data should outweigh the cost of training on fewer samples, which we test in the experiments that follow.

## B. Estimating Regime Changes from Actuator Response

In simulation, we first evaluate replay retention using ground-truth regime labels to measure staleness directly. Outside simulation, those labels would be unavailable, so we separately test whether estimates from the robot’s own interaction data can support similar replay decisions without privileged dynamics information. We estimate dynamics changes from the agent’s actions and measured joint velocities. For each episode e and joint $j ,$ we compute

$$
K _ { e , j } = \frac { \sum _ { t } ( v _ { j , t + 1 } - v _ { j , t } ) a _ { j , t } } { \sum _ { t } a _ { j , t } ^ { 2 } + \epsilon } .\tag{7}
$$

$K _ { e , j }$ measures actuator response rather than physical gain, since it can also vary with contacts, state visitation, and policy behavior. We evaluate the mean response across joints, a variant restricted to large actions, $| a _ { j , t } | > 0 . 5$ , and the full vector of per-joint responses.

Each response channel is standardized using its median and median absolute deviation. Pruned Exact Linear Time (PELT) [24], [25] then segments the episode-level series using squared-error cost, a minimum segment length of five episodes, and penalty 3d log n, where d is the number of channels and n the number of episodes. We use changepoint detection as a tool for estimating when the interaction dynamics shift, rather than as a replay strategy itself [26], [27]. Earlier segments are labeled stale when their difference from the final segment gives a two-sample z-statistic greater than three, a threshold set before the evaluation and not tuned on these runs. These labels provide an estimated AUC at the point where the replay decision is made, and the change magnitude is then estimated from the medians of the stale and fresh segments as

![](images/bb952f413f58ea310bac9540d04f05ea789c8725e70c741feb9fe56d82a15459.jpg)  
Fig. 2. Passive and Recency trajectories under permanent and recurring gain changes. Frames are taken from matched training episodes at the indicated steps. Values at right report evaluation return after the shift over five seeds, mean ± standard error (s.e.).

$$
\hat { m } = 1 - \frac { \mathrm { m e d i a n } ( K _ { \mathrm { f r e s h } } ) } { \mathrm { m e d i a n } ( K _ { \mathrm { s t a l e } } ) } .\tag{8}
$$

The replay decision uses two estimated quantities: age– staleness AUC and change magnitude. We set the AUC threshold to 0.87, corresponding to the sampled recurrence condition where the Recency advantage becomes uncertain for the main 10k replay window. The threshold is not recalibrated for the 20k window, and the selector is evaluated only against the 10k window. We set the magnitude threshold to 0.40, between the Walker sweep points where Recency hurt $( g = 0 . 7 )$ and helped $( g \ : = \ : 0 . 5 )$ , and fixed it before the $g = 0 . 6$ run that later placed the sign change between magnitudes 0.4 and 0.5 (Section VI-B). The selector chooses Recency, meaning it discards older replay, only when both thresholds are exceeded. Algorithm 1 summarizes the selector. Because regime segmentation uses the completed trajectory, the current selector is retrospective. An online version could apply the same estimates to the trajectory observed so far, making reliable online change detection a direction for future work.

## V. EXPERIMENTS

## A. Setup and Dynamics Changes

We use proprioceptive observations from the DeepMind Control Suite (DMC) [28]. Walker with DreamerV3 is our main setting, where we study the effects of forgetting under permanent and recurring dynamics changes while varying change magnitude, recurrence period, window size, and buffer capacity. We use Cheetah to test whether the same replay trends hold across different agent morphologies, and TD-MPC2 to test whether they persist with a different modelbased RL algorithm. DreamerV3 uses action repeat two and 500-transition episodes, so the 10,000-transition recency window spans about twenty episodes. Each branch runs for 130k environment frames. The dynamics change begins after 40k frames, and we evaluate return over the next 90k frames.

We study three controlled dynamics changes: a permanent gain reduction, permanent actuator damage, and a recurring gain change. We also run a no-change control. These capture permanent changes such as reduced actuator effectiveness or damage, and recurring changes such as adding and later removing a payload. The gain reduction lowers actuator gain to $g = 0 . 5 { \ : } .$ , actuator damage sets one actuator’s gain to zero, and the recurring condition alternates between the original and reduced-gain dynamics every 20k frames after the first change, so each regime lasts 20k frames.

We use separate sweeps to test how recurrence and change magnitude affect whether forgetting older replay helps or hurts performance. The recurrence sweep varies how often the original dynamics return at a fixed gain reduction. The magnitude sweep varies g under a permanent change to identify when Recency begins to outperform Passive. It uses $g \in \{ 1 . 0 , 0 . 9 , 0 . 7 , 0 . 5 , 0 . 3 \}$ and an additional run at $g = 0 . 6$ near the observed sign change.

TABLE I  
WALKER RETURN AFTER THE SHIFT AND AGE–STALENESS AUC (FIVE SEEDS, MEAN ± S.E.).  
RECENCY HELPS AFTER PERMANENT CHANGES BUT HURTS UNDER RECURRING DYNAMICS AND WITH NO CHANGE.
<table><tr><td>Change</td><td> $\mathrm { \ A U C _ { a g e } }$ </td><td>Passive</td><td>Recency</td><td>Graded</td><td>WMAR</td></tr><tr><td>Gain reduction</td><td>1.00</td><td> $1 7 7 \pm 2 1$ </td><td> $3 2 3 \pm 3 0$ </td><td> $1 6 3 \pm 1 3$ </td><td> $3 8 7 \pm 2 7$ </td></tr><tr><td>Actuator damage</td><td>1.00</td><td> $2 3 2 \pm 4 0$ </td><td> $4 0 4 \pm 3 4$ </td><td> $2 4 6 \pm 4 8$ </td><td> $4 1 8 \pm 2 8$ </td></tr><tr><td>Recurring</td><td>0.54</td><td> $4 8 1 \pm 1 7$ </td><td> $2 9 0 \pm 1 9$ </td><td> $4 6 4 \pm 1 1$ </td><td> $4 8 4 \pm 2 4$ </td></tr><tr><td>No change</td><td>一</td><td> $9 3 6 \pm 3$ </td><td> $7 3 5 \pm 2 7$ </td><td> $9 3 6 \pm 4$ </td><td> $8 1 6 \pm 2 2$ </td></tr></table>

(a) Permanent Gain Change  
![](images/425052436deb940dc044a64e9ffe57c0046696435edbe6fe4231454a60b81625.jpg)

![](images/d1a57ffdca9bc6a70c509ea22770b1365cfea2d04439621e0be0c5beab6ca9e7.jpg)  
Fig. 3. Evaluation return before and after the dynamics change for (a) permanent and (b) recurring gain reduction, averaged over five seeds with s.e. bands. The recency window is applied from the branch point, 40k frames before the change. The vertical line marks the initial change, and the shaded intervals mark the periods in which the reduced gain is active. After a permanent change, Recency reaches a higher return than Passive, while under recurrence Passive recovers more strongly whenever the original dynamics return.

All experiments use five paired seeds. To control for differences in training history before the shift, all replay strategies within a seed branch from the same checkpoint saved before the shift [29]. The window size, capacity, and $g = 0 . 6$ controls use a separate set of training runs initialized from independently trained pre-shift checkpoints. We report means and standard errors for all results.

## B. Controls, Transfer, and External Perturbations

The main recency window contains 10k transitions. We repeat the recurrence experiment with a 20k window to test the dependence on window size in (5). We also compare Recency with a 10k reservoir buffer, which keeps a uniform sample from the full replay history rather than only the newest transitions [30], [31]. Recency and reservoir therefore have the same capacity but different age distributions, allowing us to test whether any benefit comes from keeping recent data specifically rather than simply using a smaller buffer. We also compare against graded replay, which downweights transitions according to the current model’s prediction error, to test whether prediction error alone is enough to identify stale data. We include a WMAR-style diversity replay baseline [10] to test whether preserving a broader range of past experience avoids the costs of strict recency. Finally, we use a reset-once control that clears replay only at the initial change to test whether one-time forgetting is enough, or whether older data needs to be continually discarded.

To test transfer across morphology, we repeat the actuator damage and recurring conditions on Cheetah. We then repeat the same comparison with TD-MPC2 to test a different model-based RL algorithm.

To test whether the replay trends extend beyond our controlled gain changes, we also evaluate two perturbations from the Real-World RL benchmark [32]. Walker joint damping follows a cyclic schedule. Contact friction decreases from 0.7 toward 0.01. We do not assign age–staleness AUC values to them because we have not derived validated staleness labels from their perturbation histories.

Estimator evaluation: To test whether interaction data can predict when older replay should be discarded without access to ground-truth staleness labels, we evaluate the actuatorresponse estimator from Section IV-B on saved trajectories. For the permanent gain change, we measure detection delay. We then evaluate the selector on 24 saved replay histories, one per training run, spanning three recurrence periods and permanent changes at gain factors 0.5 and 0.6. We compare estimated staleness with the ground-truth label at each endpoint and check whether the selector chooses the strategy that performed better. The runs with a 20k recurrence period use the same 314-episode cutoff as the return sweep over recurrence periods in Section VI-B.

Implementation details: We use an open-source PyTorch implementation of DreamerV3 [33] with its default proprioceptive DMC configuration. Training uses batches of 16 sequences of length 64, a train ratio of 512, and a model learning rate of $1 0 ^ { - 4 }$ . Replay capacity is one million transitions. We evaluate for ten episodes every 10k frames, with evaluation trajectories kept separate from training replay, and average return over the 90k frames following the change.

(a) Varying Recurrence  
![](images/3e210b5cc06add8c4f53845a91990d1aaacf08be41656f0ba7180a1f800b8931.jpg)

(b) Permanent Change Magnitude  
![](images/e4ed3bb6dcc836b0c03af033ced81c3b33be0ddd96ca11f1058bfa23e8fb988f.jpg)  
Actuator gain g (smaller = larger change)

(c) Varying Window Size  
![](images/e267f351bafda6362d06f63c094fff4c1017e9962b308dd50cdbebe107dbbf9b.jpg)  
Fig. 4. Recency–Passive return difference under (a) varying recurrence, (b) varying permanent gain reduction, and (c) varying window size, with positive values favoring Recency. Points show paired means over five seeds with s.e. bars. Conditions shown with diamond-shaped markers were rerun from a separate set of independently trained pre-shift checkpoints.

## VI. RESULTS

## A. Permanent vs. Recurring Dynamics

Recency helps under permanent changes and hurts under recurrence (Table I, Fig. 2). After the permanent gain reduction, Recency improves return over Passive replay by $1 4 6 \pm 3 8$ . Permanent actuator damage gives the same sign, with Recency improving return over Passive by $1 7 2 \pm 4 2$ In both cases, the measured age–staleness AUC is 1.00, meaning every stale transition is older than every fresh transition in the buffer.

The direction reverses under recurrence. At a timeaveraged AUC of 0.54, Recency’s return is $1 9 1 \pm 1 7$ below Passive. With no dynamics change, it is $2 0 1 \pm 2 5$ below Passive. Fig. 3 shows how the two replay strategies diverge over training, with Recency already below Passive before the change once its replay window is applied. Graded replay stays close to Passive under recurrence and avoids the large loss from the fixed recency window. WMAR obtains the highest mean return of the four strategies in the Walker gain reduction condition (Table I).

## B. Recurrence, Magnitude, and Window Size

We first fix the gain reduction and vary the recurrence period. As the dynamics return more frequently, the timeaveraged age–staleness AUC falls from 1.00 to 0.53, while the Recency advantage moves from +146 to −287 (Fig. 4a). At the sampled AUC of 0.87, the Recency advantage is $- 1 9 \pm 1 4$ , making the sign uncertain. Changing the recurrence period affects both age–staleness AUC and how long the agent trains under each regime. The resulting return differences therefore reflect both factors rather than age– staleness AUC alone.

The magnitude sweep matches the prediction of (6): small permanent changes do not justify discarding older data, while larger changes can. The Recency advantage is negative at $g \ = \ 1 . 0 , \ 0 . 9 .$ , and 0.7 and becomes positive at $g \ = \ 0 . 5$ (Fig. 4b). At $g = 0 . 3$ , all methods are near the task floor, so the comparison is less informative. The runs from separately trained checkpoints narrow the transition between harmful and useful forgetting to between $g = 0 . 6$ and $g = 0 . 5$ . At $g = 0 . 6 ,$ the Recency advantage is $- 1 5 4 \pm 3 6 ,$ while at $g =$ 0.5 it is $+ 6 8 \pm 2 6 .$ . The estimator recovers this magnitude from actuator response alone, with mean estimates of 0.43 at $g = 0 . 5$ and 0.35 at $g = 0 . 6$ , so the quantity that separates these two cases is available to the selector’s magnitude gate.

TABLE II  
WALKER REPLAY CONTROLS, SEPARATELY TRAINED CHECKPOINTS (FIVE SEEDS, MEAN ± S.E.).
<table><tr><td>Change</td><td>Passive</td><td>Recency</td><td>Reservoir</td><td>Reset-once</td></tr><tr><td>Gain reduction</td><td> $2 2 0 \pm 2 1$ </td><td> $2 8 8 \pm 2 2$ </td><td> $1 2 8 \pm 1 4$ </td><td> $4 2 5 \pm 6 6$ </td></tr><tr><td>Recurring</td><td> $4 8 7 \pm 1 9$ </td><td> $2 8 7 \pm 1 2$ </td><td> $2 4 9 \pm 1 4$ </td><td> $3 1 2 \pm 3 8$ </td></tr></table>

TABLE III

REPLAY COMPARISONS IN ADDITIONAL SETTINGS (FIVE SEEDS, MEAN ± S.E.).
<table><tr><td>Setting</td><td>Change</td><td>Recency — Passive</td><td>WMAR — Passive</td><td></td></tr><tr><td>Cheetah</td><td>Damage</td><td> $+ 1 8 4 \pm 3 5$ </td><td> $+ 1 1 1 \pm 3 9$ </td><td rowspan="4"></td></tr><tr><td>Cheetah</td><td>Recurring</td><td> $- 1 2 6 \pm 1 2$ </td><td> $- 4 2 \pm 2 2$ </td></tr><tr><td>TD-MPC2</td><td>Damage</td><td> $+ 8 \pm 1 5$ </td><td></td></tr><tr><td>TD-MPC2</td><td>Recurring</td><td> $- 1 2 3 \pm 1 3$ </td><td></td></tr><tr><td>RWRL</td><td>Recurring</td><td> $- 2 2 2 \pm 3 2$ </td><td></td><td rowspan="2"></td></tr><tr><td>RWRL</td><td>Friction</td><td> $+ 9 4 \pm 5 8$ </td><td></td></tr></table>

Under recurrence, increasing the replay window reduces Recency’s loss relative to Passive at every sampled period (Fig. 4c), consistent with (5). A larger window keeps more data from earlier regimes, which becomes useful when those dynamics return. For the replay decision, this means that when previously seen dynamics are likely to recur, retaining older data or widening the window is the better choice, because the transitions being discarded will be needed again.

## C. Replay Controls and Transfer

Table II tests whether Recency helps because it retains recent data or simply because it trains on fewer transitions. Recency and the reservoir buffer both retain 10k transitions, but the reservoir keeps a uniform sample of the full replay history rather than only the newest data. The 10k reservoir performs worse than Passive replay under both permanent and recurring dynamics, while Recency improves return after the permanent gain change. This shows that reducing replay capacity alone does not explain Recency’s benefit. Which transitions are retained also matters. Reset-once outperforms Recency after the permanent shift but remains $1 7 5 \pm 3 2$ below Passive under recurrence, and all five of its paired differences there are negative. The recurrence loss persists even without a continually applied window, so discarding data from dynamics that later return is itself costly.

(b) Selector Agreement  
(a) Actuator-Response Traces  
![](images/ec16ed6f0caf0f9dcc88ee2dbb818c9be0f287822adde70a62e558151a5edf24.jpg)

![](images/57c103cda0243cbcad3cb3663fff0bc7b0b4370230129d6778a42b53b183dbfd.jpg)  
Fig. 5. Retrospective replay decisions. (a) Actuator-response traces with recorded and detected dynamics changes for permanent, recurring, and unchanged dynamics. The first five random-action episodes are omitted from the plot but included in detection. (b) Agreement between the selector and the betterperforming strategy across 24 saved replay histories. Fractions at the end of each bar give the number of correct decisions.

Cheetah reproduces both Walker results (Table III). On Cheetah, Recency improves return over Passive by 184 ± 35 after actuator damage and is $1 2 6 \pm 1 2$ below Passive under recurrence. WMAR preserves a broader mix of past experience instead of selecting primarily by age. On Cheetah, it improves less than Recency after damage but avoids part of Recency’s loss under recurrence. This supports the main result that aggressive age-based forgetting helps most after permanent shifts, while retaining a broader history is more useful when earlier dynamics can return.

TD-MPC2 shows the same cost under recurrence, with Recency 123 ± 13 below Passive, but a weaker benefit after actuator damage, where the Recency advantage is $+ 8 \pm 1 5$ with only three of five seeds positive. The benefit of forgetting after a permanent shift appears to depend more on the learning algorithm than the cost of discarding data under recurrence. One reason may be that TD-MPC2’s Passive return after damage stays near 800, against 232 for DreamerV3 in Table I, leaving little room for Recency to improve on the full history.

On the Real-World RL benchmark, under cyclic damping, Recency is $2 2 2 \pm 3 2$ below Passive, with all five paired differences negative, extending our recurrence result to an external benchmark perturbation. Under drifting friction, the Recency advantage is $+ 9 4 \pm 5 8 .$ , but its paired bootstrap interval includes zero, so we cannot conclude that it helps. Across these settings, the cost of forgetting under recurrence is the more consistent effect, and its benefit under changes that do not recur depends on the condition.

## D. Retrospective Replay Decisions

The per-joint actuator-response detector finds the permanent gain change within three episodes in all five seeds. Each detection occurs one episode after the recorded change. The selector chooses Recency in three of the five runs with a permanent change at $g = 0 . 5$ and, across all 24 histories, chooses the replay strategy that actually performed better in 19 of them (Fig. 5b).

Estimated AUC does not recover every recurring endpoint accurately. Some correct decisions to retain older data instead come from the estimated magnitude falling below the magnitude threshold. The permanent change is detected reliably, while later recurring switches are not always recovered, and the actuator-response estimate can drift over time even when the underlying dynamics do not change.

## VII. DISCUSSION

## A. Implications for Replay Design

A fixed window of recent data can hurt performance even without a dynamics change, because it discards useful older experience and reduces the available training data. Recency’s return is $2 0 1 \pm 2 5$ below Passive with no dynamics change and $1 9 1 \pm 1 7$ below under recurrence. Large permanent changes can make discarding older data worthwhile, while recurrence makes those transitions useful again. Replay choice therefore depends on both change magnitude and how well transition age separates stale from current data. A larger window reduces Recency’s loss under recurrence, while a reservoir with the same capacity, which keeps a uniform sample of the replay history, does not reproduce Recency’s gain after a permanent change.

## B. From Regime Detection to Replay Selection

Detecting a change does not determine what to do with older replay data. The actuator-response detector identifies a shift in the interaction statistics. The replay decision also depends on the estimated change magnitude and on how well age separates stale from current data. The results for the permanent change in Fig. 5 make this distinction clear. The gain change is detected in every seed one episode after it occurs, yet the selector chooses the better-performing strategy in only three of the five runs.

In both runs with a wrong decision, the change is still detected at the same delay as in the successful seeds. In one seed, the estimated AUC is 0.72 and the estimated magnitude is 0.36, so neither gate is passed. In the other, the magnitude estimate passes at 0.43, but the estimated AUC falls to 0.47. In these cases, detecting the change is easier than deciding whether older replay should be discarded.

Limitations: Age–staleness AUC summarizes how well age ranks stale and fresh transitions, but it does not specify how much stale data falls inside a particular replay window. In the controlled experiments, staleness is defined from known dynamics regimes and reconstructed episode timelines. We do not report AUC for the Real-World RL perturbations because validated staleness labels have not been derived from those perturbation histories.

The actuator-response analysis is retrospective, and its thresholds and the histories it is evaluated on both come from Walker. The actuator-response estimate can also vary with policy behavior and state visitation. An online version would need to make each decision from past observations only, for example with online change-point detection [34], and be evaluated on previously unseen conditions.

## VIII. CONCLUSION

We show that the value of older replay data depends on how the dynamics evolve over time. Restricting replay to recent data improves return after large permanent changes, while older data becomes useful again when the dynamics recur. Recency’s loss under recurrence also appears on Cheetah, with TD-MPC2, and under an external benchmark perturbation. Our retrospective actuator-response estimator suggests that these replay decisions can be made from interaction data without access to simulator dynamics parameters. An online version could use the same signals to adjust replay retention as a robot’s dynamics change over time.

## ACKNOWLEDGMENTS

We thank Rohan Bandaru, Brian Wei, Sergio Orozco, David Paulius, and members of the Brown Intelligent Robot Lab for helpful discussions and feedback.

## REFERENCES

[1] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse domains through world models,” arXiv preprint arXiv:2301.04104, 2023.

[2] N. Hansen, H. Su, and X. Wang, “TD-MPC2: Scalable, robust world models for continuous control,” in International Conference on Learning Representations (ICLR), 2024.

[3] L.-J. Lin, “Self-improving reactive agents based on reinforcement learning, planning and teaching,” Machine Learning, vol. 8, pp. 293– 321, 1992.

[4] T. de Bruin, J. Kober, K. Tuyls, and R. Babuška, “Experience selection in deep reinforcement learning for control,” Journal of Machine Learning Research, vol. 19, pp. 1–56, 2018.

[5] S. Zhang and R. S. Sutton, “A deeper look at experience replay,” arXiv preprint arXiv:1712.01275, 2017.

[6] W. Fedus, P. Ramachandran, R. Agarwal, Y. Bengio, H. Larochelle, M. Rowland, and W. Dabney, “Revisiting fundamentals of experience replay,” in International Conference on Machine Learning (ICML), 2020, pp. 3061–3071.

[7] A. Rahimi-Kalahroudi, J. Rajendran, I. Momennejad, H. van Seijen, and S. Chandar, “Replay buffer with local forgetting for adapting to local environment changes in deep model-based reinforcement learning,” in Conference on Lifelong Learning Agents (CoLLAs), 2023.

[8] I. Kauvar, C. Doyle, L. Zhou, and N. Haber, “Curious replay for model-based adaptation,” in International Conference on Machine Learning (ICML), 2023.

[9] S. Kessler, M. Ostaszewski, M. Bortkiewicz, M. Zarski, M. Wołczyk,<sup>˙</sup> J. Parker-Holder, S. J. Roberts, and P. Miłos, “The effectiveness of´ world models for continual reinforcement learning,” in Conference on Lifelong Learning Agents (CoLLAs), 2023.

[10] L. Yang, L. Kuhlmann, and G. Kowadlo, “Augmenting replay in world models for continual reinforcement learning,” arXiv preprint arXiv:2401.16650, 2024.

[11] K. Lee, Y. Seo, S. Lee, H. Lee, and J. Shin, “Context-aware dynamics model for generalization in model-based reinforcement learning,” in International Conference on Machine Learning (ICML), 2020.

[12] A. Kumar, Z. Fu, D. Pathak, and J. Malik, “RMA: Rapid motor adaptation for legged robots,” in Robotics: Science and Systems (RSS), 2021.

[13] A. Nagabandi, I. Clavera, S. Liu, R. S. Fearing, P. Abbeel, S. Levine, and C. Finn, “Learning to adapt in dynamic, real-world environments through meta-reinforcement learning,” in International Conference on Learning Representations (ICLR), 2019.

[14] X. B. Peng, M. Andrychowicz, W. Zaremba, and P. Abbeel, “Sim-toreal transfer of robotic control with dynamics randomization,” in IEEE International Conference on Robotics and Automation (ICRA), 2018.

[15] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in International Conference on Machine Learning (ICML), 2017.

[16] K. Rakelly, A. Zhou, D. Quillen, C. Finn, and S. Levine, “Efficient offpolicy meta-reinforcement learning via probabilistic context variables,” in International Conference on Machine Learning (ICML), 2019.

[17] A. Cully, J. Clune, D. Tarapore, and J.-B. Mouret, “Robots that can adapt like animals,” Nature, vol. 521, pp. 503–507, 2015.

[18] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized experience replay,” in International Conference on Learning Representations (ICLR), 2016.

[19] S. Sinha, J. Song, A. Garg, and S. Ermon, “Experience replay with likelihood-free importance weights,” in Learning for Dynamics and Control (L4DC), 2022.

[20] A. Bifet and R. Gavaldà, “Learning from time-changing data with adaptive windowing,” in SIAM International Conference on Data Mining (SDM), 2007, pp. 443–448.

[21] K. Khetarpal, M. Riemer, I. Rish, and D. Precup, “Towards continual reinforcement learning: A review and perspectives,” Journal of Artificial Intelligence Research (JAIR), vol. 75, 2022.

[22] J. A. Hanley and B. J. McNeil, “The meaning and use of the area under a receiver operating characteristic (ROC) curve,” Radiology, vol. 143, no. 1, pp. 29–36, 1982.

[23] L. I. Kuncheva and I. Žliobaite, “On the window size for classification˙ in changing environments,” Intelligent Data Analysis, vol. 13, no. 6, pp. 861–872, 2009.

[24] R. Killick, P. Fearnhead, and I. A. Eckley, “Optimal detection of changepoints with a linear computational cost,” Journal of the American Statistical Association, vol. 107, no. 500, pp. 1590–1600, 2012.

[25] C. Truong, L. Oudre, and N. Vayatis, “Selective review of offline change point detection methods,” Signal Processing, vol. 167, p. 107299, 2020.

[26] L. N. Alegre, A. L. C. Bazzan, and B. C. da Silva, “Minimum-delay adaptation in non-stationary reinforcement learning via online highconfidence change-point detection,” in International Conference on Autonomous Agents and Multiagent Systems (AAMAS), 2021.

[27] G. Zollicoffer, K. Eaton, J. Balloch, J. Kim, W. Zhou, R. Wright, and M. Riedl, “Novelty detection in reinforcement learning with world models,” in International Conference on Machine Learning (ICML), 2025.

[28] Y. Tassa, Y. Doron, A. Muldal, T. Erez, Y. Li, D. de Las Casas, D. Budden, A. Abdolmaleki, J. Merel, A. Lefrancq, T. Lillicrap, and M. Riedmiller, “DeepMind control suite,” arXiv preprint arXiv:1801.00690, 2018.

[29] P. Henderson, R. Islam, P. Bachman, J. Pineau, D. Precup, and D. Meger, “Deep reinforcement learning that matters,” in AAAI Conference on Artificial Intelligence, 2018.

[30] J. S. Vitter, “Random sampling with a reservoir,” ACM Transactions on Mathematical Software, vol. 11, no. 1, pp. 37–57, 1985.

[31] D. Isele and A. Cosgun, “Selective experience replay for lifelong learning,” in AAAI Conference on Artificial Intelligence, 2018.

[32] G. Dulac-Arnold, N. Levine, D. J. Mankowitz, J. Li, C. Paduraru, S. Gowal, and T. Hester, “An empirical investigation of the challenges of real-world reinforcement learning,” arXiv preprint arXiv:2003.11881, 2020.

[33] NM512, “dreamerv3-torch: Implementation of DreamerV3 in Py-Torch,” https://github.com/NM512/dreamerv3-torch, 2023.

[34] R. P. Adams and D. J. C. MacKay, “Bayesian online changepoint detection,” arXiv preprint arXiv:0710.3742, 2007.