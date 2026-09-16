# Adapting to Decision-Relevant Non-Stationarity in Decentralized Heterogeneous Bandits

Zhaojun Peng Hantsukipzj@gmail.com

## Abstract

Decentralized bandit systems often contain heterogeneous agents: rewards can change at individual agents even when the best action for the network stays the same. These local changes may cancel when rewards are averaged across agents, so the number of local changes $S _ { \mathrm { l o c } }$ can be much larger than the number of changes in the best common arm $S _ { \mathrm { d e c } }$ . We introduce Decision-Relevant Fresh Comparison (DRFC), which uses new, balanced samples from all agents to compare arms at the network level and switches only when fresh global evidence indicates that the common best arm has changed. We prove a high-probability dynamic regret bound with no adaptation term depending on $S _ { \mathrm { l o c } } .$ , and show that every algorithm must still pay for identifying genuine decision switches and propagating them through the communication graph. Under a distinct time-average benchmark, an anytime-valid sliding-window extension handles gradual drift; experiments on synthetic, semi-real, and MovieLens-1M replays show that DRFC ignores decision-irrelevant local changes while the extension avoids false switches.

## Introduction

Many online learning systems are both distributed and heterogeneous. Examples include federated recommendation, edge decision systems, and regional pricing. Diferent agents face diferent local populations, so the same action can yield diferent reward distributions at diferent agents. At the same time, raw observations cannot be centralized, so agents must communicate over sparse, timevarying networks. In these systems, non-stationarity is often local before it is global: one population, device, or region may drift while the action preferred by the network as a whole stays unchanged.

Combining decentralized and non-stationary bandits naturally suggests detecting local shifts and refreshing estimates. This works in homogeneous systems, where a local change informs the common decision, but fails under heterogeneity: local movement need not be decision-relevant.

The common decision depends on the agent-average reward, so local changes can cancel. One population may move toward arm 1 while another moves toward arm 2, leaving the average gap and best common arm unchanged. Resetting on each local shift then spends samples and communication on changes with no decision consequence.

This separation is the point of the paper. The right adaptation unit is not a local distribution change, but a change in the network-level best action. We distinguish the number of local distribution changes, $S _ { \mathrm { l o c } }$ , from the number of decision switches, $S _ { \mathrm { d e c } }$ , where a decision switch is a change in the best common arm under the agent-average reward. In heterogeneous systems $S _ { \mathrm { l o c } }$ can be much larger than $S _ { \mathrm { d e c } } ,$ so $S _ { \mathrm { l o c } }$ counts environmental motion that the learner should ignore.

A two-agent illustration. Two agents and two arms already show the gap. If one agent alternates toward arm 1 while the other moves in the opposite phase, the local means change every L rounds but the agent-average best arm can remain fixed. Then $S _ { \mathrm { l o c } } = \Theta ( T / L )$ while $S _ { \mathrm { d e c } } = 0$ : local-changereactive methods keep resetting, whereas a decision-level method should certify the common arm once and keep it.

We propose Decision-Relevant Fresh Comparison (DRFC) to implement this decision-level view. DRFC does not try to detect every local shift. It periodically performs fresh equal-agent comparisons of global arm gaps: each active arm is sampled under the same time distribution, each agent contributes equally to the aggregate estimate, and source-traceable gossip makes the comparison common knowledge over the communication graph. The algorithm switches only when a fresh globalgap certificate says that the best common arm has changed.

The resulting regret bound has no adaptation term in $S _ { \mathrm { l o c } }$ . Its switch-dependent terms scale with $S _ { \mathrm { d e c } }$ , plus the sampling and communication needed to certify fresh global gaps. DRFC-Probe uses cheap probes to trigger full confirmation only when the cached decision appears stale. We also prove that this dependence is not a proof artifact. Protocols that react to local changes pay $\Omega ( S _ { \mathrm { l o c } } )$ on cancellation instances, and any rate-consistent algorithm must pay a switch-induced certification and communication-delay cost when the best common arm really changes.

Contributions. (1) Decision-relevant non-stationarity. We distinguish local changes $S _ { \mathrm { l o c } }$ from decision switches $S _ { \mathrm { d e c } }$ and identify $S _ { \mathrm { d e c } } \ll S _ { \mathrm { l o c } }$ as the regime where local-change adaptation is wasteful.

(2) Fresh global-gap certification. We give DRFC, a fresh equal-agent comparison protocol that certifies the best common arm over the graph and attains high-probability dynamic consensus regret with no $S _ { \mathrm { l o c } }$ adaptation term (Theorem 1). (3) Switch-axis necessity. We prove unavoidable switch-induced certification and communication-delay costs for rate-consistent algorithms when $S _ { \mathrm { d e c } } > 0$ (Theorem 2), and show that local-change-reactive protocols pay $\Omega ( S _ { \mathrm { l o c } } )$ even when $S _ { \mathrm { d e c } } = 0$ (Theorem 3). (4) Drift-robust decision monitoring. Under a distinct time-average decision benchmark, we extend the principle to within-regime drift via DRFC-Seq, an anytimevalid sliding-window monitor that avoids decision-irrelevant false switches while adapting under a window-average margin condition (Theorem 4).

After formalizing decision-relevant non-stationarity, we develop DRFC and DRFC-Probe and establish switch-axis upper and lower bounds together with a local-reactive separation. We then introduce DRFC-Seq for the distinct time-average drift benchmark before evaluating both settings.

## Related Work

Non-stationary bandits. Passive methods forget old observations through sliding windows, discounting, or variation budgets [Auer et al., 2002, Garivier and Moulines, 2011, Besbes et al., 2014, Cheung et al., 2019]. Active methods restart after detected changes [Liu et al., 2018, Cao et al., 2019, Besson et al., 2022, Komiyama et al., 2024], and adaptive reductions remove prior knowledge of the amount of non-stationarity [Auer et al., 2019, Wei and Luo, 2021]. Recent work refines these ideas for smooth drift, constrained feedback, heavy tails, and path-length variation [Suk, 2024, Li and Li, 2025, Genalti et al., 2025, Hu et al., 2026]. These results measure reward-model variation or best-arm switches in a centralized stream. Our setting separates local reward changes from changes in the decentralized common decision.

Decentralized heterogeneous bandits. Cooperative bandits quantify the cost of communication through network, spectral, or flooding-time parameters [Landgren et al., 2016, Kolla et al., 2018, Martinez-Rubio et al., 2019, Xu and Klabjan, 2023, Liu et al., 2025, Cheng and Maghsudi, 2023].

Heterogeneous and federated bandits study local biases, client sampling, gossip, asynchronous communication, robustness, heavy-tailed graph and reward models, and common-arm objectives [Yang et al., 2022, Chawla et al., 2020, Wang et al., 2020, Shi and Shen, 2021, Chawla et al., 2023, Xu et al., 2025, Mirfakhar et al., 2025, Wang et al., 2025, Hu et al., 2025, Wang and $\mathrm { X u }$ , 2026, 2025]. These works are primarily stationary. Recent multi-agent work also studies combinatorial allocation with evolving rewards [Adams et al., 2025], but it does not distinguish $S _ { \mathrm { l o c } }$ from $S _ { \mathrm { d e c } }$ . DRFC targets this gap: agents certify fresh equal-agent global gaps over the graph and adapt to decision switches rather than local distribution changes.

Sequential and distributed change detection. Classical quickest-detection theory characterizes optimal single-stream procedures under false-alarm constraints [Moustakides, 1986, Pollak, 1985, Lai, 1995]. Distributed variants combine local sensor tests through fusion rules or consensus [Veeravalli, 2001, Tartakovsky and Veeravalli, 2008, Braca et al., 2011]. Such detectors are designed to alarm when a monitored stream changes. Under heterogeneous rewards, that event can be decision-irrelevant because local changes can cancel in the agent average. We use confidencesequence tools [Howard et al., 2021, Kaufmann and Koolen, 2021, Waudby-Smith and Ramdas, 2024] for a diferent purpose: certifying fresh global comparisons and, later, the drift extension.

## Problem Formulation

Reward and communication model. There are N agents and K arms. At round t, agent i pulls arm $A _ { i , t }$ and observes bounded reward $X _ { i , A _ { i , t } , t } \in [ 0 , 1 ]$ with

$$
\mathbb { E } [ X _ { i , a , t } \mid { \mathcal { F } } _ { t - 1 } ] = \mu _ { i , a , t } .
$$

Local means are heterogeneous and time-varying. Agents communicate over a random graph $G _ { t } ;$ the only graph property used in the main theorem is the flooding time below. A notation table is in the supplement.

Assumption 1 (Bounded oblivious stochastic rewards). Let $\mathcal { F } _ { t }$ be the σ-algebra generated by past rewards, actions, graphs, and fresh randomization. The mean path $\{ \mu _ { i , a , t } \}$ is $\mathcal { F } _ { 0 }$ -measurable. Fresh randomization used by the learner is drawn independently across agents and blocks, independent of $\mathcal { F } _ { t - 1 }$ and of the reward noise process. Conditional on $\mathcal { F } _ { t - 1 }$ and all learner randomization, the centered noise $X _ { i , a , t } - \mu _ { i , a , t } \in [ - 1 , 1 ]$ has zero mean, and the noises observed by diferent agents in the same round are mutually independent.

The exogeneity and cross-agent clauses are what give equal-agent averaging its $1 / \sqrt { N }$ radius in Lemma 1. The lower-bound families of Theorem 2 use independent Bernoulli rewards and lie inside this class, so the hardness they establish applies to every algorithm covered by the upper bounds.

Assumption 2 (Random graph flooding). There exists $D _ { G } ( \delta )$ such that, with probability at least $1 - \delta _ { i }$ , every source-traceable record relevant to the regret analysis floods to all agents within $D _ { G } ( \delta )$ rounds. For Erdős–Rényi $G _ { t }$ with $p \ge c \log ( N T / \delta ) / N , D _ { G } ( \delta ) \le N - 1$

The equal-agent global mean is

$$
\bar { \mu } _ { a , t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mu _ { i , a , t } .
$$

Decision-relevant non-stationarity. The instantaneous best common arm is

$$
a _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { a } \bar { \mu } _ { a , t } .
$$

We assume this maximizer is unique at every round. This removes artificial decision switches caused only by tie-breaking. The local-change and decision-switch counts are

$$
\begin{array} { l } { { \displaystyle { \cal S } _ { \mathrm { l o c } } = \sum _ { t = 2 } ^ { T } { \bf 1 } \{ \exists i , a : \mu _ { i , a , t } \neq \mu _ { i , a , t - 1 } \} } , }  \\ { { \displaystyle { \cal S } _ { \mathrm { d e c } } = \sum _ { t = 2 } ^ { T } { \bf 1 } \{ a _ { t } ^ { \star } \neq a _ { t - 1 } ^ { \star } \} . } } \end{array}\tag{1}
$$

The regime of interest is $S _ { \mathrm { d e c } } \ll S _ { \mathrm { l o c } }$ : local rewards may change often, while the best common arm changes rarely.

## Decision regimes and regret.

Definition 1 (Sign-stable decision regimes). There exist switch times $1 = \rho _ { 0 } < \cdot \cdot \cdot < \rho _ { S _ { \mathrm { d e c } } } <$ $\rho _ { S _ { \mathrm { d e c } } + 1 } = T + 1$ with $a _ { t } ^ { \star } = a _ { r } ^ { \star }$ for all $t \in [ \rho _ { r } , \rho _ { r + 1 } )$ . Write $L _ { r } = \rho _ { r + 1 } - \rho _ { r }$ for the length of regime $^ { r , }$ so $\begin{array} { r } { \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } L _ { r } = T } \end{array}$ . For every $a \neq a _ { r } ^ { \star }$ , the instantaneous global gap in regime r is sign-stable:

$$
\begin{array} { r l } & { \Delta _ { a } ^ { ( r ) } = \underset { t \in [ \rho _ { r } , \rho _ { r + 1 } ) } { \operatorname* { i n f } } \big ( \bar { \mu } _ { a _ { r } ^ { \star } , t } - \bar { \mu } _ { a , t } \big ) > 0 , } \\ & { \Gamma _ { a } ^ { ( r ) } = \underset { t \in [ \rho _ { r } , \rho _ { r + 1 } ) } { \operatorname* { s u p } } \big ( \bar { \mu } _ { a _ { r } ^ { \star } , t } - \bar { \mu } _ { a , t } \big ) \leq 1 . } \end{array}
$$

The main benchmark is dynamic consensus regret against the instantaneous best common arm:

$$
R _ { T } = \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { N } \bigl ( \mu _ { i , a _ { t } ^ { \star } , t } - \mu _ { i , A _ { i , t } , t } \bigr ) .\tag{2}
$$

Local means may change arbitrarily often inside a regime. The main model only requires the global best arm and the signs of the global gaps to remain stable between decision switches.

Environment assumptions. The stochastic reward and graph assumptions above, together with the minimum length condition below, are the only environment assumptions for Theorem 1. Scanbudget choices are algorithmic and stated with the certification guarantee.

Assumption 3 (Minimum decision-regime length). Let $L _ { \operatorname* { m i n } } = \operatorname* { m i n } _ { r } L _ { r }$ . The main guarantee requires $L _ { \mathrm { m i n } }$ to exceed the monitoring, certification, and flooding time needed after a decision switch. The concrete inequality is stated with Theorem 1. This spacing is required only between decision switches, not between local changes.

## Fresh Decision Certification

The basic primitive is a fresh comparison of two arms at the equal-agent level. It does not test whether any local distribution changed. Instead, it discards stale evidence and estimates the current global gap from a balanced sample in which every agent contributes symmetrically. This primitive is what lets DRFC react to $S _ { \mathrm { d e c } }$ rather than $S _ { \mathrm { l o c } }$

Fresh equal-agent comparison. For a scan window W, define the equal-agent time-average gap

$$
\bar { G } _ { a , b } ( \mathcal { W } ) = \frac { 1 } { \left| \mathcal { W } \right| } \sum _ { t \in \mathcal { W } } \left( \bar { \mu } _ { a , t } - \bar { \mu } _ { b , t } \right) .
$$

Each scan block pulls every active arm the same number of times at every agent, with rewardindependent random order. The resulting pairwise estimator is

$$
\hat { G } _ { a , b } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bigl ( \hat { g } _ { i , a } - \hat { g } _ { i , b } \bigr ) ,
$$

where each agent has the same number of fresh samples from arms a and b.

Two symmetries are essential. Reward-independent random arm orders give every arm the same exposure to each slot of an oblivious mean path, so within-scan drift cannot systematically favor one arm. Equal-agent averaging assigns weight $1 / N$ to each agent regardless of local sample counts, while source-keyed deduplication preserves those weights under gossip. The certificate therefore targets the fresh global gap rather than a sample-weighted or stale surrogate.

Lemma 1 (Fresh equal-agent comparison). Fix a scan window W contained in one decision regime. On the high-probability event used in Theorem 1, simultaneously for all active pairs,

$$
\begin{array} { r } { | \hat { G } _ { a , b } - \bar { G } _ { a , b } ( \mathcal { W } ) | \leq \beta ( \mathcal { W } ) , ~ } \\ { \beta ( \mathcal { W } ) = \widetilde { O } \left( \sqrt { \frac { \log \left( K ^ { 2 } T ^ { 3 } / \delta \right) } { N h B } } \right) , } \end{array}
$$

where B is the number of balanced blocks in W. Hence a positive lower-confidence gap certifies a positive equal-agent gap over the same fresh window.

Corollary 1 (Scan-budget calibration). Suppose lower bounds $\underline { { \Delta } } _ { a } \leq \Delta _ { a } ^ { ( r ) }$ are known for the active challengers and set

$$
B _ { a } = \left\lceil \frac { 1 6 C ^ { 2 } \log ( K ^ { 2 } T ^ { 3 } / \delta ) } { N h \underline { { \Delta } } _ { a } ^ { 2 } } \right\rceil .
$$

A fresh active-set scan with budget

$$
H _ { \mathrm { s c a n } } \geq C _ { \mathrm { s c a n } } h \sum _ { a } B _ { a } + D _ { G } ( \delta / 3 )
$$

certifies the unique best common arm whenever the scan is contained in one decision regime. Constants and the active-set proof are in supplement A.1.

## DRFC and Its Regret Guarantee

Algorithm. DRFC is the conservative instantiation of the fresh certificate. It maintains a candidate ${ \hat { a } } ,$ an epoch id, a scan schedule, and source-traceable records. Between scans, agents exploit aˆ. Every M rounds, DRFC runs an active-set scan using only fresh balanced records, with no local-change alarm or local reset rule. After each block, arm a is eliminated once

$$
\operatorname* { m a x } _ { b \in \boldsymbol { A } \backslash \{ a \} } \left( \hat { \boldsymbol { G } } _ { b , a } - \boldsymbol { \beta } _ { b , a } \right) > 0 .
$$

After certification, switch records propagate over the random graph and agents synchronize for $D _ { G } ( \delta / 3 )$ rounds.

Algorithm 1 DRFC active-set scan   
1: Input: block size $h ,$ monitoring period M, scan budget $H _ { \mathrm { s c a n } }$ , confidence $\delta ,$ flooding $D _ { G }$   
2: Initialize aˆ by one scan in rounds $[ 1 , H _ { \mathrm { s c a n } } ]$ , charged to $\mathcal { C } _ { 0 }$ , then set $e \gets 0$   
3: for $t = H _ { \mathrm { s c a n } } + 1$ to $T$ do   
4: Exploit aˆ if no phase active.   
5: if $t \equiv 0$ mod M and idle then   
6: Set $A  [ K ] .$   
7: while $| { \mathcal { A } } | > 1$ and budget $H _ { \mathrm { s c a n } }$ not exhausted do   
8: Run synchronized balanced active-set block with per-agent independent random orders   
(h slots per active arm).   
9: Gossip source-keyed records in background (dedup by source, epoch, active set, arm,   
block), sampling continues.   
10: On completed equal-agent records, eliminate identically at all agents each a with   
max<sub>b</sub> $( \hat { G } _ { b , a } - \beta _ { b , a } ) > 0$   
11: end while   
12: if $A = \{ w \}$ and w $\neq \hat { a }$ then   
13: Emit switch record, set $\hat { a }  w ,$ , set $e + { = } 1$ , sync $D _ { G } ( \delta / 3 )$ rounds.   
14: end if   
15: end if   
16: end for

Certification complexity. For regime r, define

$$
\mathcal { C } _ { r } = \sum _ { a \neq a _ { r } ^ { \star } } \frac { \Gamma _ { a } ^ { ( r ) } } { ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } ,
$$

which is the gap-dependent cost of certifying all challengers in regime r and reduces to $\textstyle \sum _ { a } 1 / \Delta _ { a } ^ { ( r ) }$ for stationary gaps.

Theorem 1 (DRFC dynamic consensus regret). Under Assumptions 1, 2, and 3, and the decision regimes of Definition 1, choose $H _ { \mathrm { s c a n } }$ by Corollary 1. If $L _ { \mathrm { m i n } } \ge M + 2 H _ { \mathrm { s c a n } } + D _ { G } ( \delta / 3 )$ , then DRFC satisfies $\mathrm { w . p . } \geq 1 - \delta$

$$
R _ { T } = \widetilde O \Big ( \frac { N K ( h + D _ { G } ( \delta / 3 ) ) T } { M } + \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } ( 1 + \frac { L _ { r } } { M } ) \mathcal C _ { r }
$$

$$
+ N S _ { \mathrm { d e c } } ( M + H _ { \mathrm { s c a n } } + D _ { G } ( \delta / 3 ) ) \Big ) .
$$

Proof sketch. Lemma 1 gives a uniform radius for every fresh balanced scan. Sign stability makes certified lower-confidence gaps preserve the true equal-agent signs, so each suboptimal arm is removed once $\beta ( \mathcal { W } ) \leq \Delta _ { a } ^ { ( r ) } / 4$ , contributing $\mathcal { C } _ { r }$ up to logs. The other terms are periodic scans, monitoring delay, one conservative boundary scan, and graph synchronization. Full proof in supplement A.

The three terms are scheduled certification, within-regime statistical certification, and switchonly delay/synchronization. No term scales with $S _ { \mathrm { l o c } }$ , so local changes that cancel in the equal-agent average do not force adaptation. The probe-triggered variant below replaces periodic full scans by probe-triggered confirmation and removes the conservative boundary term $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$

Gap adaptivity. A geometric scan-budget schedule removes prior knowledge of $\Delta _ { \mathrm { m i n } }$ up to logarithmic factors; the full Gap-Adaptive DRFC statement and proof are in supplement B.

## Probe-Triggered Monitoring and Lower Bound

## DRFC-Probe

DRFC-Probe is a modification of the monitoring mechanism, not a separate learning algorithm. It replaces

$$
\begin{array} { c } { { \mathrm { p e r i o d i c ~ f u l l ~ s c a n } } } \\ { { \Downarrow } } \\ { { \mathrm { c h e a p ~ p r o b e } + \mathrm { t r i g g e r e d ~ c o n f i r m i n g ~ s c a n } } } \end{array}
$$

Every M rounds, a balanced all-arm probe tests whether the cached candidate may be stale; only a suspicious probe launches the same active-set scan as Algorithm 1. Let $d _ { \mathrm { p r o b e } }$ be the worst-case delay from a decision boundary to completion of the first suspicious probe (for the fixed schedule in the supplement, $d _ { \mathrm { p r o b e } } \leq M + q _ { p } K h _ { \mathrm { p r o b e } } )$ . To keep the confirming scan inside the new regime, we require explicitly

$$
L _ { r } \ge d _ { \mathrm { p r o b e } } + H _ { \mathrm { s c a n } } + D _ { G } ( \delta / 3 ) .
$$

Corollary 2 (Probe-triggered monitoring). Under the assumptions of Theorem $^ { 1 , }$ suppose $( 2 +$ $\lambda _ { p } ) \beta _ { p } < \Delta _ { \mathrm { m i n } } ,$ , so a correct cached candidate causes no false trigger, and the regime-length condition above holds. Then, $w . p . \geq 1 - \delta$ , every confirming scan is contained in one decision regime and

$$
\begin{array} { r l r } {  { R _ { T } = \widetilde { O } \Big ( \frac { N K q _ { p } h _ { \mathrm { p r o b e } } T } { M } + \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } \mathcal { C } _ { r } } } \\ & { } & { + N S _ { \mathrm { d e c } } \big ( d _ { \mathrm { p r o b e } } + D _ { G } ( \delta / 3 ) \big ) \Big ) . } \end{array}
$$

Thus probe-triggered monitoring removes the conservative boundary term $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$ . The trigger calibration and proof are in supplement C.

## Information-Theoretic Lower Bound

Theorem 2 (Switch-axis lower bound). Fix margin $\Delta \in ( 0 , 1 / 8 )$ and suficiently long regimes. There are constants $c _ { \mathrm { c } } , c _ { \mathrm { g } } > 0$ and two diferent homogeneous instance families with lower-bound scales

certification:

$$
A : = c _ { \mathrm { c } } S _ { \mathrm { d e c } } \frac { K - 1 } { \Delta } ,
$$

graph propagation:

$$
B : = c _ { \mathrm { g } } N S _ { \mathrm { d e c } } D _ { G } \Delta .
$$

The first applies to rate-consistent algorithms by a per-arm change-of-measure argument when the regimes provide Ω(K log $T / ( N \Delta ^ { 2 } ) )$ ) certification room. The second applies to decentralized graphcommunication algorithms on a diameter- $. D _ { G }$ broom graph under the conditions stated in supplement G. Because the constructions are diferent, every algorithm in the intersection faces an instance in their union satisfying

$$
\begin{array} { r } { \mathbb { E } [ R _ { T } ] \ge \operatorname* { m a x } \{ A , B \} \ge \frac { 1 } { 2 } ( A + B ) . } \end{array}
$$

This is a two-instance minimax statement, not an additive lower bound on one instance. Full conditions and proofs are in supplement G.

<table><tr><td>Upper-bound cost</td><td>Corresponding lower-bound scale</td></tr><tr><td>Certification</td><td> $S _ { \mathrm { d e c } } ( K - 1 ) / \Delta$ </td></tr><tr><td>Graph propagation</td><td> $N S _ { \mathrm { d e c } } D _ { G } \Delta$ </td></tr><tr><td></td><td>Periodic monitoring Not constrained by Theorem 2</td></tr></table>

Table 1: Axis-wise lower-bound comparison after Corollary 2 removes the boundary term $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$

On homogeneous equal-gap instances, certification matches in $S _ { \mathrm { d e c } } , K$ , and $1 / \Delta$ . Propagation matches in $S _ { \mathrm { d e c } }$ and $D _ { G } ;$ the upper bound is worst-case in the per-round gap whereas the lower bound carries $\Delta .$ , so the table does not claim full gap-wise optimality.

## Local-Reactive Protocol Separation

We complete the instantaneous-benchmark analysis with an illustrative, class-relative separation.

Theorem 3 (Illustrative separation from local-reactive protocols). For the class of protocols required to reset after suficiently many agents experience local jumps, there is a cancellation instance with $S _ { \mathrm { d e c } } = 0$ and $S _ { \mathrm { l o c } } = \Theta ( T / L )$ on which every such protocol pays $\Omega ( S _ { \mathrm { l o c } } )$ reset regret or controlcommunication cost, while Theorem 1 has no $S _ { \mathrm { l o c } ^ { - } } d e p e n d e n t$ adaptation term. This is a class-relative illustrative separation, not the algorithm-independent lower bound of Theorem 2. The formal reactiveclass definition, zero-sum sign construction, SW-UCB and CUSUM membership proofs, and per-reset cost argument are in supplement $E { - } F .$

## Extension: Within-Regime Drift

The preceding sections use the instantaneous best common arm and count its switches by $S _ { \mathrm { d e c } }$ . We now study a distinct comparator: the best fixed common arm after averaging over each drift regime.

## Time-Average Decision Benchmark

Let $1 = \rho _ { 0 } < \cdots < \rho _ { S _ { \scriptscriptstyle \mathrm { d e s } } ^ { \mathrm { a v g } } } < \rho _ { S _ { \scriptscriptstyle \mathrm { d e s } } ^ { \mathrm { a v g } } + 1 } = T + 1$ partition the horizon into maximal average-decision regimes. For regime r, let $L _ { r } = \bar { \rho } _ { r + 1 } - \rho _ { r }$ dec and define

$$
a _ { r } ^ { \mathrm { a v g } } = \arg \operatorname* { m a x } _ { a } \frac { 1 } { L _ { r } } \sum _ { t = \rho _ { r } } ^ { \rho _ { r + 1 } - 1 } \bar { \mu } _ { a , t } .
$$

Adjacent regimes have diferent unique maximizers; hence $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } }$ counts average-decision switches, distinct from the instantaneous count $S _ { \mathrm { d e c } }$ . The corresponding decision regret is

$$
R _ { T } ^ { \mathrm { { d e c } } } = \sum _ { r } \sum _ { t = \rho _ { r } } ^ { \rho _ { r + 1 } - 1 } \sum _ { i = 1 } ^ { N } \left( \mu _ { i , a _ { r } ^ { \mathrm { { a v g } } } , t } - \mu _ { i , A _ { i , t } , t } \right) .
$$

This benchmark difers from the instantaneous regret $R _ { T }$ in (2). Because $a _ { r } ^ { \mathrm { a v g } }$ is optimal only after averaging over the regime, an individual summand—and even a short partial sum—may be negative. Thus $R _ { T } ^ { \mathrm { d e c } }$ measures regime-level decision quality, not tracking of an instantaneous oracle.

## Window-Average Margin

Assumption 4 (Window-stable decision margin). Under the average-decision regimes above, there are a window length W in balanced probe blocks and a margin $\bar { \Delta } > 0$ such that, for every regime r, every length-W window $\mathcal { W } \subseteq [ \rho _ { r } , \rho _ { r + 1 } )$ , and every $a \neq a _ { r } ^ { \mathrm { a v g } }$ 2

$$
\bar { G } _ { a _ { r } ^ { \mathrm { a v g } } , a } ( \mathcal { W } ) \geq \bar { \Delta } .
$$

The instantaneous gap may change sign inside $\mathcal { W } ;$ only the window-average decision is required to remain stable.

## DRFC-Seq

For a window starting at probe block $s , \hat { G } _ { b , a } ^ { ( s ) } ( n )$ and $\bar { G } _ { b , a } ^ { ( s ) } ( n )$ denote the empirical and corresponding equal-agent time-average gaps over the same n per-arm observations.

Lemma 2 (Uniform window correctness). Under Assumptions 1 and 2, with probability at least $1 - \alpha _ { i }$ , simultaneously for every ordered arm pair $( b , a )$ , every window start s, and every per-arm count $n \geq 1$ 2

$$
\begin{array} { r l } & { \left| \hat { G } _ { b , a } ^ { ( s ) } ( \boldsymbol { n } ) - \bar { G } _ { b , a } ^ { ( s ) } ( \boldsymbol { n } ) \right| \leq r _ { n } , } \\ & { \qquad \quad u _ { n , s } : = \log \log ( e n ) } \\ & { \qquad \quad + \log ( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha ) , } \\ & { \qquad r _ { n } = \sqrt { a _ { 0 } u _ { n , s } / ( N n ) } . } \end{array}
$$

The guarantee is uniform over window starts, not merely valid for one fixed window. Its blockmartingale proof is in supplement H.

DRFC-Seq applies Lemma 2 to a continuously sliding window. Every $\delta _ { \mathrm { p } }$ rounds, agents collect one balanced all-arm probe block; a challenger replaces the candidate only when its lower-confidence window-average gap is positive. The monitor acts only after the window contains W fully flooded blocks.

## DRFC-Seq Regret

Theorem 4 (DRFC-Seq decision regret under within-regime drift). Run DRFC-Seq with a sliding window of W balanced blocks, probe gap $\delta _ { \mathrm { p } }$ , and radius $r _ { n }$ . Let $n = W h$ be the per-arm sample count in a full window. Suppose Assumptions 1, 2, and 4 hold. The length-W window must satisfy the time-average margin condition with margin ∆<sup>¯</sup> inside each regime; the full-window sample size must satisfy $W h \ge n _ { 0 } ( \bar { \Delta } ) = \widetilde { \cal O } ( 1 / ( N \bar { \Delta } ^ { 2 } ) )$ ; and every regime must contain at least 2W probe blocks, equivalently at least $2 W \delta _ { \mathrm { p } } + D _ { G } ( \alpha )$ rounds. The monitor is evaluated only on a full window.

Correctness. With probability at least $1 - \alpha$ , once DRFC-Seq holds $a _ { r } ^ { \mathrm { a v g } }$ , no full in-regime window can induce a false switch away from it, even when the instantaneous gap is negative. This follows simultaneously over all window starts from Lemma 2.

Adaptivity and regret. Under $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } }$ average-decision switches satisfying the spacing above, the window slides fully into each new regime within W probe blocks and DRFC-Seq adopts the new best arm. Its decision regret satisfies

$$
\begin{array} { r l r } {  { R _ { T } ^ { \mathrm { d e c } } = \widetilde { O } \Big ( N K h T / \delta _ { \mathrm { p } } + \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } } \mathcal { C } _ { r } ^ { \mathrm { s e q } } } } \\ & { } & { + N ( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } + 1 ) ( W \delta _ { \mathrm { p } } + D _ { G } ( \alpha ) ) \Big ) , } \end{array}
$$

Algorithm 2 DRFC-Seq: anytime-valid drift-robust monitor   
1: Input: block size $h ,$ probe gap $\delta _ { \mathrm { { p } } } .$ , window W (blocks), level $\alpha ,$ flooding $D _ { G } ( \alpha )$   
2: Initialize aˆ arbitrarily to any arm, then set $\mathcal { Q }  \emptyset$ (probe recovery adopts the best arm within   
W blocks).   
3: for $t = 1$ to $T$ do   
4: if t is a probe time then   
5: Run one synchronized balanced block with per-agent independent orders (h   
pulls/arm/agent) over all arms, then flood records.   
6: Append the fully-flooded equal-agent reward/pull sums to $\mathcal { Q } ,$ and if $| \mathcal { Q } | > W$ drop oldest.   
7: Form windowed means $\hat { \mu } _ { a }$ and counts $n _ { a }$ , then set b ← arg max<sub>a̸=ˆa</sub> $\hat { \mu } _ { a }$   
8: Compute the anytime-valid radius $r _ { n }$ at the current window start s from the stitched   
boundary above.   
9: $/ /$ act only on a full window $( | \mathcal { Q } | = W )$ , aligning with the length-W margin (Assumption 4)   
10: if $| \mathcal { Q } | = W$ and $\left( \hat { \mu } _ { b } - \hat { \mu } _ { \hat { a } } \right) - \left( r _ { n _ { b } } + r _ { n _ { \hat { a } } } \right) > 0$ then   
11: Emit switch, set $\hat { a }  b ,$ sync $D _ { G } ( \alpha )$ rounds. $/ /$ relabel only, sliding window (line 5)   
forgets, no flush   
12: end if   
13: else   
14: Exploit aˆ.   
15: end if   
16: end for

where $\mathcal { C } _ { r } ^ { \mathrm { s e q } } = \widetilde { O } ( K / \bar { \Delta } _ { r } ^ { 2 } )$ is keyed to the time-average margin. This is not the instantaneous $\mathcal { C } _ { r }$ of Theorem 1, which need not exist under drift. No term depends on $S _ { \mathrm { l o c } }$ or on the number of instantaneous sign changes.

## Memory–Adaptation Tradeof

For a monitor M, let $\ell _ { \mathrm { e f f } }$ be the shortest observation window such that, conditional on the current candidate, the law of its adopt decision is determined by the last $\ell _ { \mathrm { e f f } }$ rounds.

Proposition 1 (Memory–adaptation tradeof under cancellation drift). There exists a single-regime cancellation-drift instance $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } \ : = \ : 0 )$ of period P whose time-average gap is $\bar { \Delta } > 0$ but whose instantaneous gap is negative during each down-phase. Any monitor with efective memory

$$
\ell _ { \mathrm { e f f } } < P / 2
$$

that adopts a genuine ∆<sup>¯</sup> -margin switch within $\ell _ { \mathrm { e f f } }$ rounds with probability at least q uniformly over prehistory false-adopts during a down-phase with probability at least q<sub>0</sub>. A drift-safe monitor can escape only by using efective memory at least $P / 2$ and paying the corresponding adaptation delay. A sample-suficient DRFC-Seq window spanning a full period operates on this drift-safe side of the tradeof. The formal detector class and proof are in supplement H.1.

## Unknown Drift Period

Doubling-Window DRFC-Seq removes exact prior knowledge of the drift period by running a geometric ladder of window lengths and operating on the shortest window whose certificate agrees with a drift-safe reference window. The guarantee assumes that the ladder’s largest window covers the unknown period, is drift-safe, and is sample-suficient; it may be chosen from a conservative upper range. Its one-time calibration and per-switch latency are given in supplement H.2. This extension adapts the memory scale $W$ to an unknown period $P ;$ it is distinct from the gap-adaptive construction above, which adapts the scan budget $H _ { \mathrm { s c a n } }$ to an unknown gap $\Delta _ { \mathrm { m i n } }$

![](images/79afe5ab26ef0ef6147f7bf89170bb7f1316ac3afeac7fea3b9c8c2d57401dd2.jpg)

![](images/7b22b430d0b17e098e10981ebe0007e65b4d8a6bc915211990359819059297dd.jpg)

![](images/ee21cc8c47e2228d13a7b160d6becc82b8a7b19296cd07302944630466db7cb5.jpg)

![](images/22cdb6a7716872367611b0e911fb25c1b0bf64dcb277f15f8690371742e222e1.jpg)  
Figure 1: Decision-relevant scaling. (a) At $S _ { \mathrm { d e c } } = 0$ , DRFC is insensitive to $S _ { \mathrm { l o c } }$ , while local-reactive baselines grow. (b) At fixed spacing, regret is approximately linear in $S _ { \mathrm { d e c } }$ . (c) Narrow gaps increase post-switch delay through harder certification. (d) Denser communication shortens recovery; labels give source-record reach, a flooding proxy rather than direct $D _ { G }$ . Curves show means and 95% confidence intervals where available (100 runs in panel a; 20 seeds in panels b–d).

## Experiments

We compare DRFC with local-reactive monitors and with decision-level baselines that observe the same equal-agent signal: SW-UCB-Dec, DecCUSUM, DL-GLR-klUCB, and DL-ADR-bandit. The Oracle has centralized communication but no drift-period side information. Full configurations, confidence intervals, and tuning sweeps are in supplement I.

## $S _ { \mathrm { l o c } } { - } S _ { \mathrm { d e c } }$ Separation

Figure 1a fixes $S _ { \mathrm { d e c } } = 0$ while increasing $S _ { \mathrm { l o c } }$ . At the largest sweep point $( S _ { \mathrm { l o c } } = 2 4 9 )$ , mean regret is 180 for DRFC versus 18,016 for CUSUM-Reset and 18,992 for LocalReset, jointly testing Theorem 1 and the class-relative separation of Theorem 3.

## Switch-Axis Scaling

Figures 1b–d isolate switch count, certification dificulty, and communication delay. Regret is approximately linear in $S _ { \mathrm { d e c } }$ , narrower gaps increase recovery delay consistently with harder certification, and denser graphs shorten recovery. The connectivity sweep is a proxy for $D _ { G } ,$ not a direct measurement. These tests correspond to the two lower-bound components in Theorem $2 ;$ periodic monitoring remains a separate, unconstrained axis. Supplement I.2 directly compares periodic and probe-triggered monitoring through their regret–source-record trade-of; full axis sweeps are in supplement I.5.

## Within-Regime Drift

At $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0$ we sweep drift amplitude b, so every emitted average-decision switch is false, and report both $R _ { T } ^ { \mathrm { d e c } }$ and the false-switch rate. Once $b > \Delta$ , the instantaneous gap changes sign while the window-average gap remains positive. DRFC-Seq alone stays at zero false switches, as predicted by Theorem 4. Supplement I reports the full drift-amplitude sweep and additionally varies the window length and drift period, testing Proposition 1 under both underspecified and conservative memory.

## Real-Data Replay

On the raw, unanchored MovieLens-1M [Harper and Konstan, 2015] genre replay $( \bar { \Delta } \approx 0 . 0 1 1 5 )$ DRFC-Seq attains the lowest decision regret and zero false switches among adopt-capable monitors (Table 2). Across a gain sweep, its false-switch separation from fast detectors is Holm-significant over 30 seeds for $g \geq 2 . 0$ . This validates cancellation and window memory; supplement I.4–I.8 reports full results and topology, asynchrony, period-change, and $K = 3 2$ checks.
<table><tr><td>Method</td><td> $R _ { T } ^ { \mathrm { { d e c } } }$ </td><td>False-switch rate</td></tr><tr><td>DRFC-Seq</td><td> ${ \bf 2 7 1 0 \pm 6 }$ </td><td>0.00</td></tr><tr><td>DecCUSUM</td><td> $3 0 5 3 \pm 4 7 5$ </td><td>1.00</td></tr><tr><td>DL-GLR-klUCB</td><td> $3 4 5 9 \pm 7 3 6$ </td><td>0.90</td></tr><tr><td>DL-ADR-bandit</td><td> $3 8 6 8 \pm 5 0 0$ </td><td>0.90</td></tr></table>

Table 2: Raw MovieLens drift $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } \ : = \ : 0$ , 30 seeds; $\mathrm { m e a n } \pm \mathrm { s . d . } )$ ; lower is better. Full results: supplement I.

## Discussion

In heterogeneous networks, adaptation complexity follows common-decision switches $S _ { \mathrm { d e c } }$ , not local changes $S _ { \mathrm { l o c } } \mathrm { : }$ fresh equal-agent certification ignores changes that cancel globally. DRFC-Probe replaces scheduled scans with cheap probes and triggered confirmation. The lower bounds make certification and graph propagation unavoidable; periodic monitoring remains a design choice. The drift extension changes the comparator: under the time-average benchmark, DRFC-Seq trades adoption speed for robustness to instantaneous sign changes. Thus both models certify only network-level decision changes. Extending them through push-sum averaging to directed or unreliable graphs [Kempe et al., 2003, Nedić and Olshevsky, 2015] is a natural next step.

Limitations. (i) Obliviousness excludes adaptive drift or participation. (ii) Rewards are bounded and sub-Gaussian; bounded-variance extensions can use block median-of-means. (iii) The guarantees require a unique common best arm and a positive regime/window margin; latency becomes vacuous near ties. (iv) Communication is $\widetilde { \cal O } ( N K ( h + D _ { G } ) T / M )$ source records, within 1.3× of decision-level detectors; sharper edge bounds need finer flooding (supplement I.2). (v) Synchronized balanced blocks are demanding asynchronously; margin-dependent degradation is in supplement I.4–I.7.

## Supplementary Material

This supplement contains: (A) the full proof of the dynamic regret theorem, including the timevarying concentration lemma, sign preservation, active-set scan certification, dynamic regret decomposition, and the random-graph flooding instantiation; (B) the Gap-Adaptive DRFC proposition, which removes prior knowledge of $\Delta _ { \mathrm { m i n } }$ through a persistent geometric scan-budget schedule; (C) the probe-triggered monitoring guarantee with explicit false-trigger budget; (D) the distinct sourcerecord complexity bound; (E) the construction and proof of the class-relative separation theorem; (F) verification that sliding-window UCB and CUSUM-style detectors lie inside the local-changereactive comparator class on the construction; (G) the full proof of the information-theoretic lower bound; (H) the full anytime-valid correctness and adaptivity proof of DRFC-Seq under within-regime drift (the main DRFC-Seq theorem), via Ville’s inequality and a law-of-the-iterated-logarithm confidence sequence, including (H.1) the drift-separation proof of the dichotomy proposition (main Proposition 1); (I) additional experimental figures (component ablations, communication ablations, graph-probability sweep, user-cluster semi-real, monitor-period sensitivity); (J) the proof of the personalized (per-cluster) regret theorem for DRFC-Pers; (K) a reproducibility checklist.

Notation. Table 3 collects the symbols used in the main paper and this supplement.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $N , K$ </td><td>number of agents, number of arms</td></tr><tr><td> $\Delta$ </td><td>per-regime decision margin, the equal-agent gap of the best common arm</td></tr><tr><td> $S _ { \mathrm { l o c } } , \ S _ { \mathrm { d e c } }$ </td><td>local-mean changes, instantaneous decision switches</td></tr><tr><td> $\mathbf { \boldsymbol { C } } ^ { \mathrm { a v g } }$   $S _ { \mathrm { d e c } } ^ { \mathrm { } \mathrm { } \cdots \mathrm { } \mathrm { } s }$ </td><td>switches of the regime-average decision in the drift model</td></tr><tr><td> $M$ </td><td>monitor period, rounds between certification triggers</td></tr><tr><td> $h$ </td><td>block size, per-arm per-agent pulls in one balanced block</td></tr><tr><td> $H _ { \mathrm { s c a n } }$ </td><td>scan budget, rounds spent certifying the best arm at a trigger</td></tr><tr><td> $D _ { G } ( \delta )$ </td><td>flooding time, rounds for a source record to reach all agents w.p.  $1 - \delta$ </td></tr><tr><td> $W$ </td><td>DRFC-Seq sliding-window length, in probe blocks</td></tr><tr><td> $\delta _ { \mathrm { p } }$ </td><td>probe gap, rounds between consecutive probe blocks</td></tr><tr><td> $\alpha$ </td><td> $\mathrm { D R F C - S e q }$  anytime false-switch level</td></tr><tr><td> $r _ { n }$ </td><td>anytime-valid confidence radius on the windowed equal-agent gap</td></tr><tr><td> $\scriptstyle { \mathcal { C } } _ { r }$ </td><td>certification complexity  $\textstyle \sum _ { a \neq a _ { r } ^ { \star } } \Gamma _ { a } ^ { ( r ) } / ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 }$ </td></tr><tr><td> $L _ { r }$ </td><td>length of decision regime r</td></tr></table>

Table 3: Notation used throughout the main paper and supplement.

## A. Proof Details

This supplement expands the proof sketches in the main paper. The purpose is to make explicit the sample-count convention, the finite good event, and the reason local distribution changes inside a decision regime do not enter the adaptation term. All constants below are universal and may change from line to line. The controlling constants of the three quantitative results are listed explicitly in the following subsection. Theorem, lemma, and equation references inside this supplement use independent numbering; cross-references to the main paper use result names whenever possible.

## Controlling Constants and Hidden Dependencies

For a theory reader we collect the controlling constants of the three quantitative results, so that no constant named in a theorem statement is left implicit. Every value below is the one used in the corresponding proof.

DRFC radius constant. The good-event radius of the equal-agent time-average gap over a window of $| \mathcal { W } |$ balanced blocks is

$$
\beta ( \mathcal { W } ) = C \sqrt { \frac { \log ( K ^ { 2 } T ^ { 3 } / \delta ) } { N h \left| \mathcal { W } \right| } } ,
$$

where $C$ is the universal constant of the two-term concentration bound of Lemma 3, combining reward noise by Hoefding–Azuma with schedule-selection noise by sampling-without-replacement (Bardenet–Maillard). It satisfies $C \ \leq \ 8$ , and the scan-budget calibration uses the per-arm block count $B _ { a } ^ { ( r ) } = \lceil 1 6 C ^ { 2 } \log ( K ^ { 2 } T ^ { 3 } / \delta ) / ( N h ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } ) \rceil$ . The only factor the $\widetilde O$ of the main DRFC theorem suppresses is the single logarithm $\log ( K ^ { 2 } T ^ { 3 } / \delta )$

DRFC-Seq radius constants. The anytime-valid radius is

$$
r _ { n } = { \sqrt { \frac { a _ { 0 } { \bigl ( } \log \log ( e n ) + \log ( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha ) { \bigr ) } } { N n } } } ,
$$

with $a _ { 0 } = \Theta ( c _ { 1 } )$ , where $c _ { 1 }$ is the sub-Gaussian variance proxy of one balanced block $( \sigma _ { \ell } ^ { 2 } \le c _ { 1 } / ( N h ) )$ and $C _ { 0 }$ the stitched-boundary constant of Howard et al. [2021], Theorem 1. Both are absolute. Here s indexes the window start, and the term log $( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha )$ carries a summable $6 / ( \pi ^ { 2 } s ^ { 2 } )$ schedule over starts together with the $K ( K - 1 )$ ordered arm pairs, so the guarantee holds with no predeclared horizon (Section H), while log $\log ( e n )$ is the iterated-logarithm price of the anytime guarantee. These are the only factors the $\widetilde O$ of the main DRFC-Seq theorem suppresses.

Certification length. The per-arm sample count at which the windowed estimator separates the time-average-best arm from every challenger is

$$
n _ { 0 } ( \bar { \Delta } ) = \operatorname* { i n f } \{ n \geq 1 : r _ { n } \leq \bar { \Delta } / 4 \} = \widetilde O \big ( 1 / ( N \bar { \Delta } ^ { 2 } ) \big ) ,
$$

so a window of W blocks certifies once $W h \ge n _ { 0 } ( \bar { \Delta } )$ , giving $\mathcal { C } _ { r } ^ { \mathrm { s e q } } = \widetilde { O } ( K / \bar { \Delta } _ { r } ^ { 2 } )$ keyed to the regime margin $\bar { \Delta } _ { r }$ . The block budget h is a fixed design constant and enters $r _ { n }$ only through $n = W h$ Because $r _ { n }$ carries log $( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha )$ , the certification length $n _ { 0 }$ is start-indexed; the recovery and regret statements use the worst start in the horizon, ${ n _ { 0 } ( \bar { \Delta } ) = \widetilde O ( \log ( T ) / ( N \bar { \Delta } ^ { 2 } ) ) }$ through $s \leq T$ the log T absorbed into $\widetilde O$ . The no-false-switch guarantee of part (a) needs no such cap and holds at every start by the summable schedule, so only the adaptivity rate, not the validity, sees the horizon.

## Comparison Windows

Fix an ordered arm pair $( a , b )$ and a tested comparison window W. The window consists of active-set blocks in which both a and b are active. Let $B = | \mathcal { W } |$ be the number of such fresh blocks and let

$$
m = h B
$$

be the number of fresh pulls per arm, per agent. For each block $\ell \in \mathcal W$ , write

$$
\mathcal { U } _ { \ell } = \{ t _ { \ell , 1 } , \dots , t _ { \ell , q _ { \ell } } \}
$$

for the common global-round macro-slot pool in that active-set block. In a pure pairwise block, $q _ { \ell } = 2 h$ . In an active-set block, $q _ { \ell }$ also includes slots assigned to other active arms. The active-set randomization assigns exactly h slots to a and exactly h slots to b from this block-level common pool, independently of rewards. Equivalently, for each agent and block it draws disjoint subsets

$$
S _ { i , a } ^ { ( \ell ) } , S _ { i , b } ^ { ( \ell ) } \subset \mathcal { U } _ { \ell } , \qquad | S _ { i , a } ^ { ( \ell ) } | = | S _ { i , b } ^ { ( \ell ) } | = h , \qquad S _ { i , a } ^ { ( \ell ) } \cap S _ { i , b } ^ { ( \ell ) } = \emptyset ,
$$

with uniform marginals over size-h subsets of $\mathcal { U } _ { \ell }$ . The local and global estimators are

$$
\hat { g } _ { i , a , b } ( \mathcal { W } ) = \frac { 1 } { B } \sum _ { \ell \in \mathcal { W } } \frac { 1 } { h } \sum _ { t \in S _ { i , a } ^ { ( \ell ) } } X _ { i , a , t } - \frac { 1 } { B } \sum _ { \ell \in \mathcal { W } } \frac { 1 } { h } \sum _ { t \in S _ { i , b } ^ { ( \ell ) } } X _ { i , b , t } , \qquad \hat { G } _ { a , b } ( \mathcal { W } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \hat { g } _ { i , a , b } ( \mathcal { W } ) .
$$

The concentration target is the equal-agent time-average pairwise gap

$$
\bar { G } _ { a , b } ( \mathcal { W } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { B } \sum _ { \ell \in \mathcal { W } } \frac { 1 } { q \ell } \sum _ { t \in \mathcal { U } _ { \ell } } \left( \mu _ { i , a , t } - \mu _ { i , b , t } \right) .
$$

This is the key object: it is well defined even when the local means vary within the window. The pure pairwise comparison in the main text is the special case $q _ { \ell } = 2 h$ for every block; the active-set scan uses the same concentration statement for pairwise projections of larger block-level macro-slot pools.

Let W be the family of windows on which DRFC may evaluate a decision rule. A window is indexed by an epoch id $( \leq S _ { \mathrm { d e c } } + 1 \leq T$ since each accepted switch increments the epoch), an ordered arm pair $( K ( K - 1 )$ choices), and a contiguous block interval (at most $T ^ { 2 }$ start–end pairs of global rounds). This gives the conservative count

$$
| \mathfrak { M } | \le K ( K - 1 ) \cdot T \cdot T ^ { 2 } = K ( K - 1 ) T ^ { 3 } .
$$

The confidence radius used below is

$$
\beta ( \mathcal { W } ) = C \sqrt { \frac { \log ( K ^ { 2 } T ^ { 3 } / \delta ) } { N h | \mathcal { W } | } } .
$$

## Time-Varying Concentration

Lemma 3 (Uniform time-varying comparison concentration). For any fixed synchronized pairwise or active-set-projection comparison window W and ordered pair $( a , b )$

$$
\mathbb { P } \left[ \left| \hat { G } _ { a , b } ( \mathcal { W } ) - \bar { G } _ { a , b } ( \mathcal { W } ) \right| > C \sqrt { \frac { \log ( 1 / \delta ) } { N m } } \right] \le \delta .
$$

Consequently, after increasing C, the bound

$$
\left| \hat { G } _ { a , b } ( \mathcal { W } ) - \bar { G } _ { a , b } ( \mathcal { W } ) \right| \leq \beta ( \mathcal { W } )
$$

holds simultaneously for all tested windows with probability at least $1 - \delta / 3$

Proof. Throughout, let $m = h B$ be the per-arm per-agent sample count in the window. By main $\mathrm { A s } -$ sumption 1, the local mean path is $\mathcal { F } _ { 0 }$ -measurable and each block’s fresh randomization $\{ \bar { S } _ { i , a } ^ { ( \ell ) } , S _ { i , b } ^ { ( \ell ) } \} _ { i , \ell }$ is independent of $\mathcal { F } _ { t - 1 }$ . Let S denote the entire collection of schedules used in the window. Decompose the estimation error as

$$
\hat { G } _ { a , b } ( \mathcal { W } ) - \bar { G } _ { a , b } ( \mathcal { W } ) = A + R ,
$$

where

$$
A = \hat { G } _ { a , b } ( \mathcal { W } ) - \mathbb { E } \big [ \hat { G } _ { a , b } ( \mathcal { W } ) \mid \mathcal { S } , \mathcal { F } _ { 0 } \big ]
$$

is the reward noise conditional on schedules and the mean path, and

$$
R = \mathbb { E } \Big [ \hat { G } _ { a , b } ( \mathcal { W } ) \mid \mathcal { S } , \mathcal { F } _ { 0 } \Big ] - \bar { G } _ { a , b } ( \mathcal { W } )
$$

is the schedule-randomization error in the conditional mean.

Reward noise term A. Conditional on $\boldsymbol { s }$ and $\mathcal { F } _ { 0 }$

$$
A = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { B } \sum _ { \ell \in \mathcal { W } } \frac { 1 } { h } \left( \sum _ { t \in S _ { i , a } ^ { ( \ell ) } } ( X _ { i , a , t } - \mu _ { i , a , t } ) - \sum _ { t \in S _ { i , b } ^ { ( \ell ) } } ( X _ { i , b , t } - \mu _ { i , b , t } ) \right) .
$$

Order the $2 N h B \ : = \ : 2 N m$ summands lexicographically, first by round t and within a round by agent index i, and let $\mathcal { G } _ { t , i }$ be $\mathcal { F } _ { t - 1 } \vee \sigma ( \mathcal { S } )$ enlarged by the round-t observed noises of the agents preceding i in this order. By the exogeneity clause of main Assumption 1, the reward noises are independent of the learner randomization $\boldsymbol { s }$ and each summand has zero mean conditional on $\mathcal { F } _ { t - 1 }$ and all learner randomization, and by the cross-agent clause the round-t noises of diferent agents are mutually independent under the same conditioning, so conditioning additionally on the preceding same-round noises leaves the mean at zero, $\mathbb { E } [ X _ { i , a , t } - \mu _ { i , a , t } \mid { \mathcal { G } } _ { t , i } ] = 0$ . Without that clause the $1 / \sqrt { N }$ rate below can fail, since a round-level noise common to all agents satisfies the perround martingale property yet makes A concentrate only at rate $1 / \sqrt { m }$ . The ordered summands are therefore a bounded martingale diference sequence in $[ - 1 , 1 ]$ , and A combines them with coeficients $1 / ( N h B ) = 1 / ( N m )$ . Hoefding-Azuma Azuma [1967] therefore gives

$$
\mathbb { P } ( | A | > x \mid \mathcal { S } , \mathcal { F } _ { 0 } ) \le 2 \exp \left( - \frac { 2 x ^ { 2 } } { 2 N m \cdot ( 1 / ( N m ) ) ^ { 2 } \cdot ( 2 ) ^ { 2 } } \right) = 2 \exp \left( - \frac { N m x ^ { 2 } } { 4 } \right) .
$$

Marginalizing over $s$ preserves the bound.

Schedule-randomization term R. For each agent i and block $\ell \in { \mathcal { W } }$ , define the per-block, per-arm population block means

$$
\bar { M } _ { i , a } ^ { ( \ell ) } = \frac { 1 } { q \ell } \sum _ { t \in \mathcal { U } _ { \ell } } \mu _ { i , a , t } , \qquad \bar { M } _ { i , b } ^ { ( \ell ) } = \frac { 1 } { q \ell } \sum _ { t \in \mathcal { U } _ { \ell } } \mu _ { i , b , t } ,
$$

and their empirical counterparts on the block schedule,

$$
\widehat { M } _ { i , a } ^ { ( \ell ) } = \frac { 1 } { h } \sum _ { t \in S _ { i , a } ^ { ( \ell ) } } \mu _ { i , a , t } , \qquad \widehat { M } _ { i , b } ^ { ( \ell ) } = \frac { 1 } { h } \sum _ { t \in S _ { i , b } ^ { ( \ell ) } } \mu _ { i , b , t } .
$$

The block-level deviations are

$$
Z _ { i , a } ^ { ( \ell ) } = \widehat { M } _ { i , a } ^ { ( \ell ) } - \bar { M } _ { i , a } ^ { ( \ell ) } , \qquad Z _ { i , b } ^ { ( \ell ) } = \widehat { M } _ { i , b } ^ { ( \ell ) } - \bar { M } _ { i , b } ^ { ( \ell ) } ,
$$

and the agent-level deviation aggregates to

$$
R _ { i } = \frac { 1 } { B } \sum _ { \ell \in \mathcal { W } } \left( Z _ { i , a } ^ { ( \ell ) } - Z _ { i , b } ^ { ( \ell ) } \right) , \qquad R = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } R _ { i } .
$$

Block-level concentration. Fix $( i , \ell )$ . The active-set randomization draws a uniform random ordered partition of $\mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { { } } \mathrm { } \mathrm { { } } \mathrm { { } } \mathrm { } \mathrm { { } } \mathrm { } \mathrm { { } } \mathrm { } \mathrm { } \mathrm { { } } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm \mathrm { } \mathrm { } \mathrm { } \mathrm { } \mathrm \mathrm { } \mathrm { } \mathrm { } \mathrm \mathrm { } \mathrm { } \mathrm \mathrm { } \mathrm { } \mathrm \mathrm { } \mathrm \mathrm { } \mathrm { } \mathrm \mathrm { } \mathrm \mathrm { } \mathrm \mathrm { } \mathrm \mathrm { } \mathrm \mathrm { } \mathrm \mathrm { } \mathrm \mathrm { } \mathrm \mathrm { } \mathrm \mathrm \mathrm { } \mathrm \mathrm { \mathrm } $ into $( S _ { i , a } ^ { ( \ell ) } , S _ { i , b } ^ { ( \ell ) } , \mathcal { U } _ { \ell } \backslash ( S _ { i , a } ^ { ( \ell ) } \cup S _ { i , b } ^ { ( \ell ) } ) )$ ) subject to the size constraints $| S _ { i , a } ^ { ( \ell ) } | = | S _ { i , b } ^ { ( \ell ) } | = h$ By symmetry of this partition, the marginal law of $S _ { i , a } ^ { ( \ell ) }$ is uniform over size-h subsets of $\mathcal { U } _ { \ell }$ , and similarly for $S _ { i , b } ^ { ( \ell ) }$ . Each marginal therefore satisfies the Hoefding–Serfling inequality for sampling without replacement [Bardenet and Maillard, 2015, Cor. 2.4]:

$$
\mathbb { P } \Big ( \vert Z _ { i , a } ^ { ( \ell ) } \vert > x \Big ) \le 2 \exp \left( - \frac { 2 h x ^ { 2 } } { 1 - ( h - 1 ) / q \ell } \right) \le 2 \exp \bigl ( - 2 h x ^ { 2 } \bigr ) ,
$$

and identically for $Z _ { i , b } ^ { ( \ell ) }$ . Hence both $Z _ { i , a } ^ { ( \ell ) }$ and $Z _ { i , b } ^ { ( \ell ) }$ are sub-Gaussian with proxy variance $\sigma _ { 0 } ^ { 2 } = 1 / ( 4 h )$ ， regardless of the joint dependence between them.

Aggregation across blocks. Diferent blocks $\ell \in \mathcal W$ use independently drawn schedules (main Assumption 1), so the pairs $\{ ( Z _ { i , a } ^ { ( \ell ) } , Z _ { i , b } ^ { ( \ell ) } ) \} _ { \ell \in \mathcal { W } }$ are independent across ℓ for fixed i. For each ℓ, the diference $Z _ { i , a } ^ { ( \ell ) } - Z _ { i , b } ^ { ( \ell ) }$ is the sum of two sub-Gaussian random variables and is therefore sub-Gaussian with proxy variance at most $2 \cdot 4 \sigma _ { 0 } ^ { 2 } = 2 / h$ (using the elementary bound that the sub-Gaussian proxy variance of a sum of two zero-mean variables is at most twice the sum of their individual proxies). Summing B independent copies and dividing by B gives that $R _ { i }$ is sub-Gaussian with proxy variance $2 / ( h B ) = 2 / m$ , so

$$
\mathbb { P } ( | R _ { i } | > t ) \le 2 \exp \biggl ( - \frac { m t ^ { 2 } } { 4 } \biggr ) .
$$

Aggregation across agents. The agent-level schedules are independent across i (main Assumption 1), so the $\{ R _ { i } \} _ { i = 1 } ^ { N }$ are independent sub-Gaussian random variables with the same proxy variance $2 / m$ . Therefore $\begin{array} { r } { R = \mathbf { \bar { \Phi } } N ^ { - 1 } \sum _ { i } R _ { i } } \end{array}$ is sub-Gaussian with proxy variance $2 / ( N m )$ , giving

$$
\mathbb { P } ( | R | > x ) \leq 2 \exp \left( - \frac { N m x ^ { 2 } } { 4 } \right) .
$$

Combining. By the union bound,

$$
\mathbb P \Big ( | \hat { G } _ { a , b } ( \mathcal W ) - \bar { G } _ { a , b } ( \mathcal W ) | > 2 x \Big ) \le \mathbb P ( | A | > x ) + \mathbb P ( | R | > x ) \le 4 \exp \left( - \frac { N m x ^ { 2 } } 4 \right) .
$$

Setting the right-hand side equal to δ yields the fixed-window deviation 2x $= 4 \sqrt { \log ( 4 / \delta ) / ( N m ) }$ ， which is at most $C \sqrt { \log ( 1 / \delta ) / ( N m ) }$ for a universal constant $C \ ( \mathrm { e . g . } \ C \ = \ 8 $ for $\delta \le 1 / 2$ , since $\log ( 4 / \delta ) / \log ( 1 / \delta ) \leq \dot { 2 }$ in that range).

Simultaneous bound over W. Apply the fixed-window bound with failure probability $\delta ^ { \prime } \ =$ $\delta / ( 3 | \mathfrak { W } | )$ where $| \mathfrak { V } | \le K ( K - 1 ) T ^ { 3 }$ . This replaces log $( 1 / \delta )$ by log $( K ( K - 1 ) T ^ { 3 } / \delta ) +$ log $3 \ \leq$ $2 \log ( K ^ { 2 } T ^ { 3 } / \delta )$ , which is absorbed into C in the radius $\beta ( \mathcal W )$ . Union bounding over W gives the simultaneous bound with probability at least $1 - \delta / 3$ , as claimed. □

## A Random-Graph Flooding Instantiation

The main theorem treats flooding through the abstract uniform quantity $D _ { G } ( \delta )$ ; we give one conservative instantiation. Let $\left\{ \boldsymbol { G } _ { t } \right\}$ be independent Erdős–Rényi graphs on $[ N ]$ with edge probability $p ,$ and let ${ \mathcal { T } } \subseteq [ T ] , | T | \leq T$ , be the analyzed communication rounds.

We claim that for a universal constant $c ,$

$$
p ~ \geq ~ c \frac { \log ( N T / \delta ) } { N } ~ \Longrightarrow ~ \mathbb { P } \Big [ \forall t \in { \mathcal { T } } : G _ { t } \mathrm { ~ c o n n e c t e d } \Big ] ~ \geq ~ 1 - \delta / 3 ,
$$

and that on this event every relevant source-traceable record reaches all agents within $D _ { G } ( \delta / 3 ) \leq$ $N - 1$ rounds.

Connectivity. For fixed t, a disconnection is witnessed by a cut $S$ with $s : = | S | \le N / 2$ and no crossing edge, so by a union bound over cuts,

$$
\mathbb { P } [ G _ { t } \mathrm { ~ d i s c o n n e c t e d } ] \ \leq \ \sum _ { s = 1 } ^ { \lfloor N / 2 \rfloor } { \binom { N } { s } } ( 1 - p ) ^ { s ( N - s ) } \ \leq \ \sum _ { s = 1 } ^ { \lfloor N / 2 \rfloor } \exp \bigl \{ s \log ( e N / s ) - p s ( N - s ) \bigr \} .
$$

For $s \leq N / 2$ one has $N - s \geq N / 2$ , so $\begin{array} { r } { p s ( N - s ) \geq \frac { p N } { 2 } s \geq 2 s \log ( e N / s ) } \end{array}$ once $p \ge c \log ( N T / \delta ) / N$ with c large enough, whence each summand is at most $\begin{array} { c c l } { { e ^ { - s \log ( e N / s ) } } } & { { \le } } & { { ( \delta / ( 3 N T ) ) } } \end{array}$ and $\mathbb { P } [ G _ { t }$ disconnected] $\leq ~ \delta / ( 3 T )$ . A union bound over $\smash { t \in \tau }$ gives the claimed connectivity event with probability at least $1 - \delta / 3$

Flooding time. Fix a record with informed set $S _ { \tau } \subsetneq [ N ]$ at round τ . On the connectivity event $G _ { \tau + 1 }$ has at least one edge between $S _ { \tau }$ and $[ N ] \setminus S _ { \tau }$ , so $| S _ { \tau + 1 } | \geq | S _ { \tau } | + 1$ . Iterating, $| S _ { \tau + k } | \geq$ min $\{ N , | S _ { \tau } | + k \}$ , hence $S _ { \tau + k } = [ N ]$ for some $k \leq N - 1$ , i.e. $D _ { G } ( \delta / 3 ) \leq N - 1$ . Sharper randomgraph gossip bounds may replace this instantiation without altering the regret proof, which uses only the resulting $D _ { G } ( \delta )$

## Uniform Good Event

Lemma 4 (Uniform good event). Suppose the random graph process satisfies the uniform flooding assumption with budget $\delta / 3$ , and agents use deterministic epoch tie-breaking for switch records. $T h e n ,$ with probability at least $1 - \delta _ { \cdot }$ , the following events hold simultaneously:

1. every tested estimator is within its radius $\beta ( \mathcal W )$ ;

2. every source-traceable comparison or switch record relevant to the regret analysis reaches all agents within $D _ { G } ( \delta / 3 )$ rounds;

3. every accepted switch produces a common candidate and epoch id within one synchronization window.

Proof. We track the δ-budget explicitly.

Concentration event $( \delta / 3 )$ . By Lemma 3 with parameter $\delta / 3 .$ , the simultaneous bound $| \hat { G } _ { a , b } ( \mathcal { W } )$ $\bar { G } _ { a , b } ( \mathcal { W } ) | \leq \beta ( \mathcal { W } )$ holds for all $\mathcal { W } \in \mathfrak { W }$ and all ordered active pairs with probability at least $1 - \delta / 3$ Under the Gap-Adaptive DRFC schedule of supplement B, the same bound applies simultaneously across the $k _ { \mathrm { m a x } } + 1 \le 1 + \log _ { 2 } ( \Delta ^ { ( 0 ) } / \Delta )$ doubling levels by replacing $\delta / 3$ by $\delta / ( 3 ( k _ { \operatorname* { m a x } } + 1 ) )$ inside Lemma $s ;$ the additional $\log ( k _ { \operatorname* { m a x } } + 1 )$ factor is absorbed into the $\bar { O }$

Flooding event $( \delta / 3 )$ . By the main random-graph flooding assumption with parameter $\delta / 3$ every flooding window in $\mathcal { T } _ { \mathrm { f l o o d } }$ propagates its source record to all agents within $D _ { G } ( \delta / 3 )$ rounds, simultaneously with probability at least $1 - \delta / 3$ . The cardinality $| \mathcal { T } _ { \mathrm { f l o o d } } | \le N \cdot K ( K - 1 ) T ^ { 3 }$ is absorbed in $D _ { G }$ through the union bound described in the assumption.

Synchronization event $( \delta / 3 )$ . Condition on the previous two events. Switch records contain source agent id, epoch id, old candidate, new candidate, and triggering comparison id. Agents adopt the highest received epoch id; ties at the same epoch are broken by a fixed deterministic rule $\left( \mathrm { e . g . } \right.$ smallest source agent id). On the concentration event, Lemma 6 (proven below) shows that pairwise lower-confidence gaps cannot eliminate $a _ { r } ^ { \star }$ in regime $r ,$ and they eliminate each non-best arm a once the radius is below $\Delta _ { a } ^ { ( r ) } / 4$ . A scan that does not yet have a unique active arm is treated as inconclusive and does not emit a switch record. Once a switch record is emitted, it reaches all agents within $D _ { G } ( \delta / 3 )$ rounds on the flooding event. Two simultaneous switch records with the same epoch are resolved by the deterministic tie-breaking rule. Therefore the synchronization event—all agents share the same $( \hat { a } , e )$ after one synchronization window—fails with conditional probability zero. The remaining $\delta / 3$ slack is reserved for the doubling-level union bound above.

Conclusion. By the union bound, all three events hold simultaneously with probability at least $1 - \delta$ □

## Sign Preservation and Pairwise Decisions

Lemma 5 (Sign preservation). $I f \mathcal { W }$ is contained in decision regime $^ { r , }$ then for every $b \neq a _ { r } ^ { \star }$

$$
\bar { G } _ { a _ { r } ^ { \star } , b } ( \mathcal { W } ) \geq \Delta _ { b } ^ { ( r ) } .
$$

For every $a \neq a _ { r } ^ { \star }$

$$
{ { \bar { G } } _ { a , a _ { r } ^ { \star } } } ( \mathcal { W } ) \le - \Delta _ { a } ^ { ( r ) } .
$$

Proof. Fix $\mathcal { W } \subseteq [ \rho _ { r } , \rho _ { r + 1 } )$ and $b \neq a _ { r } ^ { \star }$

$$
\begin{array} { l } { \displaystyle \bar { G } _ { a _ { r } ^ { \star } , b } ( \mathcal W ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { B } \sum _ { \ell \in \mathcal W } \frac { 1 } { q _ { \ell } } \sum _ { t \in \mathcal W _ { \ell } } \bigl ( \mu _ { i , a _ { r } ^ { \star } , t } - \mu _ { i , b , t } \bigr ) } \\ { \displaystyle \qquad \stackrel { \mathrm { ( ) } } { \geq } \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { B } \sum _ { \ell \in \mathcal W } \frac { 1 } { q _ { \ell } } \sum _ { t \in \mathcal W _ { \ell } } \Delta _ { b } ^ { ( r ) } } \\ { \displaystyle \qquad = \Delta _ { b } ^ { ( r ) } . } \end{array}
$$

Symmetrically, for $a \neq a _ { r } ^ { \star }$

$$
\begin{array} { c l c r } { \displaystyle \bar { G } _ { a , a _ { r } ^ { \star } } ( \mathcal { W } ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { B } \sum _ { \ell \in \mathcal { W } } \frac { 1 } { q \ell } \sum _ { t \in \mathcal { U } _ { \ell } } \big ( \mu _ { i , a , t } - \mu _ { i , a _ { r } ^ { \star } , t } \big ) } \\ { \displaystyle \bigodot } \\ { \displaystyle \bigtriangleup _ { \mathrm { ~  ~ } } ^ { \mathrm { ~ \scriptsize ~ \bigodot ~ } } - \Delta _ { a } ^ { ( r ) } . } \end{array}
$$

Here ○1 applies the decision-regime margin bound $\mu _ { i , a _ { r } ^ { \star } , t } - \mu _ { i , b , t } \geq \Delta _ { b } ^ { ( r ) }$ at every (i, t) with $t \in$ $[ \rho _ { r } , \rho _ { r + 1 } )$ , ○2 applies $\mu _ { i , a , t } - \mu _ { i , a _ { r } ^ { \star } , t } \leq - \Delta _ { a } ^ { ( r ) }$ at every such $( i , t )$ , and the closing equalities use the convex weights $\begin{array} { r } { \frac { 1 } { N } \cdot \frac { 1 } { B } \cdot \frac { \mathrm { i } } { q _ { \ell } } \geq 0 } \end{array}$ summing to 1. □

Lemma 6 (Pairwise correctness). On the uniform good event, consider a tested window inside decision regime r.

1. ( Reverse direction, unconditional.) For every $a \neq a _ { r } ^ { \star }$ , the lower-confidence comparison from a to $a _ { r } ^ { \star }$ is strictly negative:

$$
\hat { G } _ { a , a _ { r } ^ { \star } } ( \mathcal { W } ) - \beta ( \mathcal { W } ) < 0 .
$$

No hypothesis on $\beta ( \mathcal W )$ is needed beyond the good event.

2. (Forward direction, threshold.) For every $b \neq a _ { r } ^ { \star } , \ i f \beta ( \mathcal { W } ) \leq \Delta _ { b } ^ { ( r ) } / 4$ , then the lower-confidence comparison from $a _ { r } ^ { \star }$ to b is strictly positive:

$$
\hat { G } _ { a _ { r } ^ { \star } , b } ( \mathcal { W } ) - \beta ( \mathcal { W } ) > 0 .
$$

Proof. Part 1 (reverse).

$$
\begin{array} { r } { \hat { G } _ { a , a _ { r } ^ { \star } } ( \mathcal { W } ) - \beta ( \mathcal { W } ) \overset { \mathbb { O } } { \leq } - \Delta _ { a } ^ { ( r ) } } \\ { \overset { \mathbb { O } } { < } 0 , } \end{array}
$$

where ○1 is Lemma 5 on the good event, $\hat { G } _ { a , a _ { r } ^ { \star } } ( \mathcal { W } ) \leq - \Delta _ { a } ^ { ( r ) } + \beta ( \mathcal { W } )$ , subtracting $\beta ( \mathcal W )$ , and ○2 uses $\Delta _ { a } ^ { ( r ) } > 0 ;$ ; no upper bound on $\beta ( \mathcal W )$ is needed.

Part 2 (forward).

$$
\begin{array} { r l } & { \hat { G } _ { { a } _ { r } ^ { \star } , { b } } ( \mathcal { W } ) - \beta ( \mathcal { W } ) \overset { \mathbb { O } } { \geq } \Delta _ { { b } } ^ { ( r ) } - 2 \beta ( \mathcal { W } ) } \\ & { \qquad \overset { \mathbb { O } } { \geq } \Delta _ { { b } } ^ { ( r ) } / 2 } \\ & { \qquad > 0 , } \end{array}
$$

where ○1 is Lemma 5 on the good event, $\hat { G } _ { a _ { r } ^ { \star } , b } ( \mathcal { W } ) \geq \Delta _ { b } ^ { ( r ) } - \beta ( \mathcal { W } )$ , subtracting $\beta ( \mathcal W )$ , and $\textcircled{2}$ uses the hypothesis $\beta ( \mathcal { W } ) \leq \Delta _ { b } ^ { ( r ) } / 4$ □

Lemma 7 (Active-set scan certification). On the uniform good event, consider an active-set full-certification scan whose blocks are contained in decision regime r. The pairwise lowerconfidence elimination rule never eliminates $a _ { r } ^ { \star }$ . Moreover, each active arm a $\neq a _ { r } ^ { \star }$ is eliminated once its pairwise projection against $a _ { r } ^ { \star }$ has radius at most $\Delta _ { a } ^ { ( r ) } / 4$ . Let $\tau _ { a }$ be the number of active-set blocks in which arm a remains active. Here $\tau _ { a }$ counts active-set blocks until a’s synchronous elimination, i.e. until the balanced count reaches $B _ { a } ^ { ( r ) }$ , so $1 \leq \tau _ { a } \leq B _ { a } ^ { ( r ) } = \lceil x _ { a } \rceil$ with $x _ { a } \ = \ 1 6 C ^ { 2 } \log ( K ^ { 2 } T ^ { 3 } / \delta ) / ( N h ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } )$ . Under the nonblocking-gossip convention the eliminating equal-agent records flood with la $g \le D _ { G } ( \delta / 3 )$ , so beyond its $B _ { a } ^ { ( r ) }$ certification blocks each arm stays sampled for at most $D _ { G } ( \delta / 3 )$ further rounds, a delayed-elimination over-sampling of at most $K D _ { G } ( \delta / 3 )$ group-rounds per scan (the same budget added to $H _ { \mathrm { s c a n } }$ below) at group regret $\leq N$ each. Using $\lceil x _ { a } \rceil \leq x _ { a } + 1$ , the one completed scan contributes group regret

$$
( o n e \it { - s c a n ~ g r o u p ~ r e g r e t } ) \leq \widetilde { O } ( \mathcal { C } _ { r } ) + N h \sum _ { a \neq a _ { r } ^ { \ast } } \Gamma _ { a } ^ { ( r ) } + N K D _ { G } ( \delta / 3 ) , \qquad \mathcal { C } _ { r } = \sum _ { a \neq a _ { r } ^ { \ast } } \frac { \Gamma _ { a } ^ { ( r ) } } { ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } .
$$

The gap-dependent term collects the certification cost $N h x _ { a } \Gamma _ { a } ^ { ( r ) } = \widetilde { O } ( \Gamma _ { a } ^ { ( r ) } / ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } )$ of each arm, the floor $\begin{array} { r } { N h \sum _ { a \neq a _ { \star } ^ { \star } } \Gamma _ { a } ^ { ( r ) } \leq N h ( K - 1 ) } \end{array}$ is the mandatory single balanced block of each arm, and the last term is the nonblocking flooding-lag over-sampling. Summed over the $1 + L _ { r } / M$ scans of every regime, the floor and the flooding lag together give the periodic-monitoring term $\widetilde { O } ( N K ( h + D _ { G } ( \delta / 3 ) ) T / M )$ of Theorem 5, the per-switch part of the lag merging into the $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$ boundary term.

Proof. While arms a and $a _ { r } ^ { \star }$ are both active, each active-set block samples them over the same randomized time distribution. The pairwise projection of those blocks is therefore a valid comparison window in the sense of Lemma 3.

Step 1: best arm is never eliminated. For any active non-best arm b and any tested projection between b and $a _ { r } ^ { \star }$ contained in regime $^ { r , }$ Lemma 6 gives

$$
\hat { G } _ { b , a _ { r } ^ { \star } } ( \mathcal { W } ) - \beta ( \mathcal { W } ) < 0 .
$$

Thus no active arm has a positive lower-confidence gap against $a _ { r } ^ { \star }$ , and the elimination rule never removes $a _ { r } ^ { \star }$

Step 2: non-best arms eliminated at explicit threshold. Fix $a \neq a _ { r } ^ { \star }$ . By the definition of $\beta$ from Appendix ,

$$
\beta ( \mathcal { W } ) = C \sqrt { \frac { \log ( K ^ { 2 } T ^ { 3 } / \delta ) } { N h | \mathcal { W } | } } \le \frac { \Delta _ { a } ^ { ( r ) } } { 4 } \iff N h | \mathcal { W } | \ge \frac { 1 6 C ^ { 2 } \log ( K ^ { 2 } T ^ { 3 } / \delta ) } { ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } .
$$

Define

$$
B _ { a } ^ { ( r ) } = \left\lceil \frac { 1 6 C ^ { 2 } \log ( K ^ { 2 } T ^ { 3 } / \delta ) } { N h ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } \right\rceil ,
$$

which is the per-arm block count used by the scan-budget calibration after absorbing the constant $1 6 C ^ { 2 }$ into the unspecified constant in that calibration. Once arm a has participated in $| \boldsymbol { \mathcal { W } } | = B _ { a } ^ { ( r ) }$ pairwise-projection blocks against $a _ { r } ^ { \star } .$ , Lemma 6 gives

$$
\hat { G } _ { a _ { r } ^ { \star } , a } ( \mathcal { W } ) - \beta ( \mathcal { W } ) > 0 ,
$$

so the lower-confidence pairwise gap from $a _ { r } ^ { \star }$ to a is positive and the elimination rule removes $a .$ Therefore $\tau _ { a } \leq B _ { a } ^ { ( r ) }$ , and

$$
h \tau _ { a } \leq h B _ { a } ^ { ( r ) } = \frac { 1 6 C ^ { 2 } \log ( K ^ { 2 } T ^ { 3 } / \delta ) } { N ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } = \widetilde { O } \left( \frac { 1 } { N ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } \right) .
$$

Step 3: regret accounting. Fix arm a $\neq a _ { r } ^ { \star }$ . Because the scan is contained in regime r, $a _ { u } ^ { \star } = a _ { r } ^ { \star }$ at every slot u in its blocks. Let $\mathcal { W } _ { a }$ index the $\tau _ { a }$ active-set blocks in which a remains active, and write the group regret charged to arm a as

$$
\mathcal { R } _ { a } = \sum _ { \ell \in \mathcal { W } _ { a } } R _ { a } ^ { ( \ell ) } , \qquad R _ { a } ^ { ( \ell ) } = \sum _ { i = 1 } ^ { N } \sum _ { u \in S _ { i , a } ^ { ( \ell ) } } \big ( \mu _ { i , a _ { r } ^ { \star } , u } - \mu _ { i , a , u } \big ) \in [ - N h , N h ] .
$$

Note that the per-pull integrand $\mu _ { i , a _ { r } ^ { \star } , u } - \mu _ { i , a , u }$ is a per-agent gap that can exceed the groupaverage gap ${ \Gamma } _ { a } ^ { \left( r \right) }$ , so a deterministic per-pull bound by $\Gamma _ { a } ^ { ( r ) }$ is not available; we instead control $\textstyle { \mathcal { R } } _ { a }$ in expectation through the schedule randomization and add a Hoefding slack.

Expected regret. The marginal of $S _ { i , a } ^ { ( \ell ) }$ is uniform over size-h subsets of $\mathcal { U } _ { \ell }$ and is drawn independently across (i, ℓ). By uniform sampling without replacement,

$$
\mathbb { E } \Big [ R _ { a } ^ { ( \ell ) } \Big | \mathcal { F } _ { 0 } \Big ] = \frac { N h } { q _ { \ell } } \sum _ { u \in \mathcal { U } _ { \ell } } \big ( \bar { \mu } _ { a _ { r } ^ { \star } , u } - \bar { \mu } _ { a , u } \big ) \leq N h \Gamma _ { a } ^ { ( r ) } ,
$$

where the inequality uses $\bar { \mu } _ { a _ { r } ^ { \star } , u } - \bar { \mu } _ { a , u } \leq \Gamma _ { a } ^ { ( r ) }$ for every $u \in [ \rho _ { r } , \rho _ { r + 1 } )$ from the main sign-stable decision-regime definition.

Concentration slack. Conditional on $\mathcal { F } _ { 0 }$ , the per-agent block contribution $\begin{array} { r } { \sum _ { u \in S _ { i , a } ^ { ( \ell ) } } ( \mu _ { i , a _ { r } ^ { \star } , u } - \mu _ { i , a , u } ) } \end{array}$ lies in $[ - h , h ]$ and is independent across $i ;$ block-level schedules are independent across ℓ (main Assumption 1). By Hoefding’s inequality applied to the $N \tau _ { a }$ independent bounded summands,

$$
\mathbb { P } \Big ( \mathcal { R } _ { a } - \mathbb { E } [ \mathcal { R } _ { a } \mid \mathcal { F } _ { 0 } ] > t \Big | \mathcal { F } _ { 0 } \Big ) \leq \exp \left( - \frac { t ^ { 2 } } { 2 N h ^ { 2 } \tau _ { a } } \right) .
$$

Choosing $t = h \sqrt { 2 N \tau _ { a } \log ( 3 K ^ { 2 } T ^ { 3 } / \delta ) }$ makes the right-hand side at most $\delta / ( 3 K ^ { 2 } T ^ { 3 } )$ , and a union bound over the at most $| \mathfrak { W } | \le K ^ { 2 } T ^ { 3 }$ tested $( a , \mathcal W _ { a } )$ pairs absorbs the failure event into the $\delta / 3$ slack already reserved in Lemma 3. Hence, on the good event,

$$
\mathcal { R } _ { a } \leq N h \tau _ { a } \Gamma _ { a } ^ { ( r ) } + h \sqrt { 2 N \tau _ { a } \log ( 3 K ^ { 2 } T ^ { 3 } / \delta ) } .
$$

Substitution. From Step $2 , \tau _ { a } \leq B _ { a } ^ { ( r ) } = \lceil x _ { a } \rceil \leq x _ { a } + 1$ with $x _ { a } = 1 6 C ^ { 2 } \log ( K ^ { 2 } T ^ { 3 } / \delta ) / ( N h ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } )$ so $N h \tau _ { a } \Gamma _ { a } ^ { ( r ) } \leq N h x _ { a } \Gamma _ { a } ^ { ( r ) } + N h \Gamma _ { a } ^ { ( r ) }$ splits into a certification part and one mandatory block:

$$
N h x _ { a } \Gamma _ { a } ^ { ( r ) } = \frac { 1 6 C ^ { 2 } \Gamma _ { a } ^ { ( r ) } \log ( K ^ { 2 } T ^ { 3 } / \delta ) } { ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } = \widetilde { O } \left( \frac { \Gamma _ { a } ^ { ( r ) } } { ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } \right) , \qquad N h \Gamma _ { a } ^ { ( r ) } \le N h ,
$$

and

$$
h \sqrt { N \tau _ { a } \log ( \cdot ) } = \sqrt { N h \cdot h \tau _ { a } \log ( \cdot ) } \leq \frac { 4 C \sqrt { h } \log ( K ^ { 2 } T ^ { 3 } / \delta ) } { \Delta _ { a } ^ { ( r ) } } = \widetilde { O } \left( \frac { \sqrt { h } } { \Delta _ { a } ^ { ( r ) } } \right) .
$$

Since $\Gamma _ { a } ^ { ( r ) } \ge \Delta _ { a } ^ { ( r ) }$ (sup vs. inf of the same gap), $1 / \Delta _ { a } ^ { ( r ) } \leq \Gamma _ { a } ^ { ( r ) } / ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 }$ , so the Hoefding slack is dominated by the expected-regret term up to a $\sqrt { h }$ factor; treating h as a fixed scheme parameter absorbs $\sqrt { h }$ into the ${ \widetilde { O } } .$ Summing the per-arm certification bound $\widetilde { O } \big ( \Gamma _ { a } ^ { ( r ) } / ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } \big )$ and the per-arm floor ${ \cal N h } \Gamma _ { a } ^ { ( r ) }$ over $a \neq a _ { r } ^ { \star }$

$$
( \mathrm { o n e - s c a n ~ g r o u p ~ r e g r e t } ) \le \widetilde O ( \mathcal C _ { r } ) + N h \sum _ { a \ne a _ { r } ^ { \star } } \Gamma _ { a } ^ { ( r ) } , \qquad \mathcal C _ { r } = \sum _ { a \ne a _ { r } ^ { \star } } \frac { \Gamma _ { a } ^ { ( r ) } } { ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } .
$$

The floor obeys Nh $\begin{array} { r } { , \sum _ { a \ne a _ { * } ^ { \star } } \Gamma _ { a } ^ { ( r ) } \le N h ( K - 1 ) \mathrm { b y } \Gamma _ { a } ^ { ( r ) } \le 1 } \end{array}$ , the cost of one mandatory balanced block per arm. There are $1 + L _ { r } ^ { ' } / M$ scans in regime r and $\textstyle \sum _ { r } L _ { r } \leq T$ , so summed over every scan this floor totals

$$
\begin{array} { r } { N h ( K - 1 ) \displaystyle \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } \Bigl ( 1 + \frac { L _ { r } } { M } \Bigr ) \leq N h ( K - 1 ) \Bigl ( S _ { \mathrm { d e c } } + 1 + \frac { T } { M } \Bigr ) = \widetilde { O } \Bigl ( \frac { N K h T } { M } + N K h S _ { \mathrm { d e c } } \Bigr ) . } \end{array}
$$

The first term is the periodic-monitoring term of Theorem 5, and the second is absorbed into $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$ because $H _ { \mathrm { s c a n } } \geq 2 h ( K - 1 )$ . Hence $\mathcal { C } _ { r }$ remains purely gap-dependent, with no blockgranularity condition required.

Step $\it 4 .$ wall-clock budget. Let $\mathbf { \mathcal { A } } _ { \ell }$ be the active set in scan block ℓ, and let $\tau _ { \star }$ be the number of blocks in which $a _ { r } ^ { \star }$ remains active. Since $a _ { r } ^ { \star }$ stays active until certification, $\tau _ { \star } \le \operatorname* { m a x } _ { a \ne a _ { r } ^ { \star } } \tau _ { a } \le$ $\sum { _ { a \ne a _ { r } ^ { \star } } \tau _ { a } }$ . The total wall-clock sampling length of the scan is

$$
h \sum _ { \ell } | \mathscr { A } _ { \ell } | = h \sum _ { a \in [ K ] } \tau _ { a } \leq 2 h \sum _ { a \neq a _ { r } ^ { \star } } \tau _ { a } \leq 2 h \sum _ { a \neq a _ { r } ^ { \star } } B _ { a } ^ { ( r ) } .
$$

Adding the final flooding lag $D _ { G } ( \delta / 3 )$ for source records, together with the within-scan delayedelimination overhead bounded by $K D _ { G } ( \delta / 3 )$ additional group samples (each non-best arm can stay active for at most $D _ { G } ( \delta / 3 )$ extra rounds beyond its synchronous-elimination time, and the per-block group sample count is at most $| \mathcal { A } _ { \ell } | \leq K )$ , gives a contained-scan wall-clock budget bounded by

$$
2 h \sum _ { a \ne a _ { r } ^ { \star } } B _ { a } ^ { ( r ) } + ( K + 1 ) D _ { G } ( \delta / 3 ) \le H _ { \mathrm { s c a n } } ,
$$

which is exactly the calibration $\begin{array} { r } { H _ { \mathrm { s c a n } } \geq 2 h \sum _ { a \neq a _ { \ast } ^ { \star } } B _ { a } ^ { ( r ) } + ( K + 1 ) D _ { G } ( \delta / 3 ) } \end{array}$ used by the main DRFC theorem (the leading factor 2 accounting for $a _ { r } ^ { \star }$ staying active until the last elimination). On the good event, the scan therefore certifies a unique winner within $H _ { \mathrm { s c a n } }$ slots whenever it is contained in a decision regime. □

## Dynamic Regret

Theorem 5 (Dynamic consensus regret under margin-known calibration, detailed form). Under the assumptions in the main text, with the block budget h a fixed constant, active-set DRFC satisfies on the good event

$$
R _ { T } = \widetilde { \cal O } \left( \frac { N K ( h + D _ { G } ( \delta / 3 ) ) T } { M } + \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } \left( 1 + \frac { L _ { r } } { M } \right) \mathcal { C } _ { r } + N S _ { \mathrm { d e c } } \left( M + H _ { \mathrm { s c a n } } + D _ { G } ( \delta / 3 ) \right) \right) .
$$

The same display holds with probability at least $1 - \delta$ . The periodic-monitoring term carries $h +$ $D _ { G } ( \delta / 3 )$ rather than h because each interior scan pays, beyond its mandatory balanced block, the nonblocking flooding-lag over-sampling of Lemma 7.

Proof. Work on the good event of Lemma 4. There is no failure-event regret on this event because the complement has probability at most δ.

Conventions.

1. A scan is interior if all of its fresh sampling blocks lie inside a single decision regime, and boundary-crossing otherwise; Lemmas 5 and 6 apply to interior scans only.

2. DRFC runs at most one active scan at a time and skips monitoring triggers while a scan or synchronization phase is active, and uncertified scans are aborted at the $H _ { \mathrm { s c a n } }$ wall-clock budget.

3. Under the nonblocking-gossip convention, comparison records propagate while later comparison slots are collected, with the final comparison-record availability lag included in $H _ { \mathrm { s c a n } } ;$ ; only accepted switch records create an explicit synchronization delay.

4. The initial fresh comparison tournament that initializes aˆ (main Algorithm 1, line 3) is treated as the first interior scan of regime $r = 0$ . It is contained in $[ 1 , H _ { \mathrm { s c a n } } ] \subset [ \rho _ { 0 } , \rho _ { 1 } )$ by the main minimum-regime-length assumption, so Lemma 7 applies and its regret is absorbed into the $\mathcal { C } _ { 0 }$ summand with the 1 in $( 1 + L _ { 0 } / M )$

We decompose the total group regret $R _ { T }$ into five additive terms:

$$
R _ { T } = R _ { \mathrm { s c a n } } + R _ { \mathrm { m o n } } + R _ { \mathrm { c r o s s } } + R _ { \mathrm { d e l a y } } + R _ { \mathrm { s y n c } } ,
$$

where each term is bounded below.

Term 1: interior scan regret $R _ { \mathrm { s c a n } }$ . By Lemma $^ { 7 , }$ one completed interior scan in regime r contributes at most $\widetilde { O } ( \mathcal { C } _ { r } ) + N h ( K - 1 ) + N K D _ { G } ( \delta / 3 )$ group regret, the second summand the mandatory single balanced block of each arm and the third the nonblocking flooding-lag over-sampling. DRFC starts a new scan every M rounds outside of synchronization windows, so regime r contains at most $\lceil L _ { r } / M \rceil + O ( 1 )$ interior scans (the $O ( 1 )$ slack absorbs the first scan after the post-switch monitoring wait and the at-most-one boundary-crossing scan split of into $R _ { \mathrm { c r o s s } } )$ . Therefore, using $\textstyle \sum _ { r } L _ { r } \leq T$

$$
R _ { \mathrm { s c a n } } \leq \widetilde O \Big ( \frac { N K ( h + D _ { G } ( \delta / 3 ) ) T } { M } \Big ) + \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } \bigg ( 1 + \frac { L _ { r } } { M } \bigg ) \widetilde O ( \mathcal { C } _ { r } ) ,
$$

where the periodic-monitoring term $\widetilde { O } ( N K ( h + D _ { G } ( \delta / 3 ) ) T / M )$ collects the mandatory one-block per-arm floor and the flooding-lag over-sampling across all ${ \widetilde { \cal O } } ( T / M )$ scans, its residual $N K ( h +$ $D _ { G } ( \delta / 3 ) ) S _ { \mathrm { d e c } }$ piece is absorbed into the $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$ term (Term 3) via $H _ { \mathrm { s c a n } } \geq 2 h ( K - 1 ) +$ $( K + 1 ) D _ { G } ( \delta / 3 )$ , and the universal $O ( 1 )$ slack is absorbed into the Oe. The post-switch statistical comparison delay is absorbed here because the first interior scan after a switch certifies the new best arm against the old candidate in the new regime, which is exactly the scan analyzed in Lemma 7.

Term 2: monitoring delay $R _ { \mathrm { m o n } }$ . After each decision switch, DRFC waits at most M rounds until the next monitoring trigger before starting a contained scan in the new regime. During this wait the algorithm exploits the previous candidate $a _ { r - 1 } ^ { \star }$ , which is now suboptimal by at most its new-regime gap $\begin{array} { r } { \Gamma ^ { ( r ) } : = \operatorname* { m a x } _ { a } \Gamma _ { a } ^ { ( r ) } \leq 1 } \end{array}$ per pull (the main sign-stable decision-regime definition bounds $\Gamma _ { a } ^ { ( r ) } \leq 1 )$ . Group regret is therefore at most $N \Gamma ^ { ( r ) }$ per round, and across the $S _ { \mathrm { d e c } }$ switches,

$$
R _ { \mathrm { m o n } } \leq N \sum _ { r = 1 } ^ { S _ { \mathrm { d e c } } } M \Gamma ^ { ( r ) } \ \leq \ N S _ { \mathrm { d e c } } M .
$$

Term 3: boundary-crossing scan regret $R _ { \mathrm { c r o s s } }$ . At most one scan straddles each decision switch boundary by the at-most-one-active-scan rule. Sign preservation does not apply to such a scan, so the conservative bound charges $H _ { \mathrm { s c a n } }$ wall-clock rounds at group regret N per round:

$$
R _ { \mathrm { c r o s s } } \leq N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } } .
$$

The boundary-crossing scan is then aborted or completes; either way, the next contained scan starts within M rounds, absorbed into $R _ { \mathrm { m o n } }$

Possible incorrect emit on a boundary-crossing scan. A boundary-crossing scan that certifies may emit a switch record for an arm w that is not $a _ { r + 1 } ^ { \star }$ , because the sign-preservation lemma (Lemma 5) does not constrain mixed-regime windows. On the good event, three observations close this case without changing the regret display. (i) Until the next monitoring trigger, the algorithm exploits w rather than the correct candidate; the duration is at most M rounds, contributing at most NM group regret per decision boundary, already counted inside $R _ { \mathrm { m o n } } \leq N S _ { \mathrm { d e c } } M$ . (ii) The first interior scan in regime $r + 1$ is contained in $[ \rho _ { r + 1 } , \rho _ { r + 2 } )$ by the main minimum-regime-length assumption, so Lemma 6 certifies $a _ { r + 1 } ^ { \star }$ against the current candidate w and emits a correct switch record; its group regret is already counted in $R _ { \mathrm { s c a n } }$ via the $r = r + 1$ summand $( 1 + L _ { r + 1 } / M ) \mathcal { C } _ { r + 1 }$ (using the additive 1 to charge this first contained scan). (iii) At most two switch records per decision boundary may be propagated. To see this, observe that each accepted switch record advances the epoch id, and the at-most-one-active-scan rule prevents a fresh scan from starting before the current scan terminates and its synchronization window of length $D _ { G } ( \delta / 3 )$ closes. A boundary-crossing scan emits at most one (possibly incorrect) switch record before the boundary’s synchronization window opens; the next contained scan in regime $r + 1$ is launched no earlier than the following monitoring trigger, contained in $[ \rho _ { r + 1 } , \rho _ { r + 1 } + M ]$ by the main minimum-regime-length assumption, and emits at most one (correct) switch record by Lemma 6. Any third switch record would require a third scan to start before $M + H _ { \mathrm { s c a n } } + D _ { G } ( \delta / 3 )$ has elapsed past the boundary, which is excluded by the at-most-one-active-scan rule and the regime-length condition $L _ { r + 1 } \ge M + 2 H _ { \mathrm { s c a n } } + D _ { G } ( \delta / 3 )$ Hence the synchronization cost is at most $2 N D _ { G } ( \delta / 3 )$ per boundary, and the universal factor of 2 is absorbed into the $\widetilde O$ of $R _ { \mathrm { s y n c } } \leq N S _ { \mathrm { d e c } } D _ { G } ( \delta / 3 )$

Term $\it 4 \mathrm { : }$ post-switch decision delay $R _ { \mathrm { d e l a y } }$ . Between the boundary-crossing scan and the first contained scan in the new regime, the algorithm exploits the previous candidate. The chain consists of $\mathrm { ( a ) }$ at most $H _ { \mathrm { s c a n } }$ rounds for the boundary-crossing scan (already charged to $R _ { \mathrm { c r o s s } } )$ , (b) at most M rounds of monitoring wait (already charged to $R _ { \mathrm { m o n } } )$ , and (c) the first contained scan, whose regret is charged by sign preservation to the $\mathcal { C } _ { r }$ term in $R _ { \mathrm { s c a n } }$ . No additional exploitation period exists between these phases. Therefore $R _ { \mathrm { d e l a v } } = 0$ with no double-counting.

Term 5: switch synchronization $R _ { \mathrm { s y n c } }$ . On the good event, every accepted switch record prop agates to all agents within $D _ { G } ( \delta / 3 )$ rounds (Lemma 4). During this synchronization window the agents may still play heterogeneous candidates; each lagging agent exploits the stale candidate $a _ { r - 1 } ^ { \star }$ whose per-round group regret in the new regime is at most its gap $\begin{array} { r } { \Gamma ^ { ( r ) } : = \operatorname* { m a x } _ { a } \Gamma _ { a } ^ { ( r ) } \leq 1 - } \end{array}$ not the crude constant 1. Across the $S _ { \mathrm { d e c } }$ switches,

$$
R _ { \mathrm { s y n c } } \leq N \sum _ { r = 1 } ^ { S _ { \mathrm { d e c } } } D _ { G } ( \delta / 3 ) \Gamma ^ { ( r ) } \ \leq \ N S _ { \mathrm { d e c } } D _ { G } ( \delta / 3 ) .
$$

The gap-scaled form $\begin{array} { r } { N \sum _ { r } D _ { G } ( \delta / 3 ) \Gamma ^ { ( r ) } } \end{array}$ is the quantity compared against the communication-delay part of the main information-theoretic lower bound: it matches the $\Omega ( N S _ { \mathrm { d e c } } D _ { G } \Delta )$ bound up to logs precisely when $\Gamma ^ { ( r ) } = \Theta ( \Delta )$ , removing the $1 / \Delta$ looseness that a crude per-round-1 accounting would carry on that instance family.

Combining. Summing the five terms,

$$
R _ { T } \le \widetilde O \Big ( \frac { N K ( h + D _ { G } ( \delta / 3 ) ) T } { M } \Big ) + \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } \bigg ( 1 + \frac { L _ { r } } M \bigg ) \widetilde O ( \mathcal { C } _ { r } ) + N S _ { \mathrm { d e c } } ( M + H _ { \mathrm { s c a n } } + D _ { G } ( \delta / 3 ) ) .
$$

This is the display in the theorem, with the periodic-monitoring term $\widetilde { O } ( N K ( h + D _ { G } ( \delta / 3 ) ) T / M )$ supplied by the per-scan one-block floor and flooding-lag over-sampling (Term 1) and the $\bar { O }$ absorbing the universal constants from Lemma 7 and the $\log ( K ^ { 2 } T ^ { 3 } / \delta )$ factor inside $\mathcal { C } _ { r }$ . The highprobability statement follows from Lemma 4. □

The gap-regular corollary in the main text follows immediately from

$$
\mathcal { C } _ { r } = \sum _ { a \neq a _ { r } ^ { \star } } \frac { \Gamma _ { a } ^ { ( r ) } } { ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } \leq \kappa \sum _ { a \neq a _ { r } ^ { \star } } \frac { 1 } { \Delta _ { a } ^ { ( r ) } } .
$$

## Probe-Triggered Monitoring

The main theorem analyzes conservative full-certification scans. The probe-triggered monitoring modification instead runs cheap probes and launches a full active-set scan only when a probe suggests that the cached candidate may be stale. This subsection records the corresponding guarantee; the learning and certification rules are unchanged.

Monitoring rule. Every M rounds, DRFC-Probe runs $q _ { p } \geq 1$ 1 randomized balanced probe blocks over all arms, with $h _ { \mathrm { p r o b e } }$ pulls per arm per agent in each block. Define $d _ { \mathrm { p r o b e } }$ as the worst-case delay from a decision boundary to completion of the first post-boundary suspicious probe; for this fixed schedule,

$$
d _ { \mathrm { p r o b e } } \leq M + q _ { p } K h _ { \mathrm { p r o b e } } .
$$

Let $\hat { \mu } _ { a } ^ { p }$ be the equal-agent probe estimate after aggregating the $q _ { p }$ probe blocks. All arms have the same probe pull count, so write the common radius as $\beta _ { p }$ . For a trigger parameter $\lambda _ { p } \geq 0$ , the probe is suspicious if some challenger $b \neq { \hat { a } }$ satisfies

$$
\hat { \mu } _ { b } ^ { p } \geq \hat { \mu } _ { \hat { a } } ^ { p } - \lambda _ { p } \beta _ { p } .
$$

The choice $\lambda _ { p } = 2$ is the standard interval-overlap trigger $\hat { \mu } _ { b } ^ { p } + \beta _ { p } \ge \hat { \mu } _ { \hat { a } } ^ { p } - \beta _ { p }$ . Smaller $\lambda _ { p }$ is more conservative and can reduce false triggers at the cost of requiring more probe accuracy after a true switch. When the probe is suspicious, DRFC launches the same active-set full-certification scan analyzed above. Switches are allowed only after that full scan certifies a winner. If no challenger is suspicious, DRFC keeps the cached certificate and continues exploiting aˆ.

Lemma 8 (Probe trigger soundness and completeness). Consider aggregated probe blocks contained in decision regime r, and let $\beta _ { p }$ be the common probe radius after aggregation. Define the equal-agent probe-window time-average mean

$$
\bar { \mu } _ { a } ^ { p } : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { q _ { p } h _ { \mathrm { p r o b e } } } \sum _ { t \in \mathcal { U } _ { p } } \mu _ { i , a , t } ,
$$

where $\mathcal { U } _ { p }$ is the union of the $q _ { p }$ probe-block macro-slot pools. By the main sign-stable decision-regime definition, $\bar { \mu } _ { a _ { r } ^ { \star } } ^ { p } - \bar { \mu } _ { b } ^ { p } \geq \Delta _ { b } ^ { ( r ) }$ for every $b \neq a _ { r } ^ { \star }$ . On the uniform concentration event:

1. if the cached candidate aˆ is not $a _ { r } ^ { \star }$ and $\Delta _ { \hat { a } } ^ { ( r ) } > ( 2 - \lambda _ { p } ) _ { + } \beta _ { p }$ , then the probe is suspicious for challenger $a _ { r } ^ { \star }$ ;

2. if the cached candidate is $a _ { r } ^ { \star }$ and $( 2 + \lambda _ { p } ) \beta _ { p } < \Delta _ { b } ^ { ( r ) }$ for every challenger $b \neq a _ { r } ^ { \star }$ , then the probe is not suspicious.

Proof. Because an all-arm probe samples every arm over the same randomized time distribution, $\hat { \mu } _ { a } ^ { p }$ concentrates around $\bar { \mu } _ { a } ^ { p }$ with radius $\beta _ { p }$ on the good event. If $\hat { a } \neq a _ { r } ^ { \star }$ , take $b = a _ { r } ^ { \star } ;$ on the good event,

$$
\begin{array} { r l } & { \hat { \mu } _ { b } ^ { p } \overset { \textcircled { 1 } } { \geq } \bar { \mu } _ { b } ^ { p } - \beta _ { p } } \\ & { \qquad \overset { \textcircled { 2 } } { \geq } \bar { \mu } _ { \hat { a } } ^ { p } + ( 1 - \lambda _ { p } ) \beta _ { p } } \\ & { \qquad \overset { \textcircled { 3 } } { \geq } \hat { \mu } _ { \hat { a } } ^ { p } - \lambda _ { p } \beta _ { p } , } \end{array}
$$

where $\textcircled{1}$ is the good-event lower confidence, $\textcircled{2}$ uses $\bar { \mu } _ { b } ^ { p } - \bar { \mu } _ { \hat { a } } ^ { p } = \Delta _ { \hat { a } } ^ { ( r ) } > ( 2 - \lambda _ { p } ) _ { + } \beta _ { p }$ , and ○3 is the good-event bound $\bar { \mu } _ { \hat { a } } ^ { p } \geq \hat { \mu } _ { \hat { a } } ^ { p } - \beta _ { p } ;$ the suspicious-probe condition therefore holds. If $\hat { a } = a _ { r } ^ { \star }$ , then for each $b \neq a _ { r } ^ { \star }$ 2

$$
\begin{array} { r l } & { \hat { \mu } _ { b } ^ { p } \overset { \mathbb { O } } { \leq } \bar { \mu } _ { b } ^ { p } + \beta _ { p } } \\ & { \qquad \overset { \mathbb { O } } { < } \bar { \mu } _ { a _ { r } ^ { \star } } ^ { p } - ( 1 + \lambda _ { p } ) \beta _ { p } } \\ & { \qquad \overset { \mathrm { O } } { \leq } \hat { \mu } _ { a _ { r } ^ { \star } } ^ { p } - \lambda _ { p } \beta _ { p } , } \end{array}
$$

where $\textcircled{1}$ is the good-event upper confidence, $\textcircled{2}$ uses $\bar { \mu } _ { a _ { r } ^ { \star } } ^ { p } - \bar { \mu } _ { b } ^ { p } \geq \Delta _ { b } ^ { ( r ) } > ( 2 + \lambda _ { p } ) \beta _ { p }$ , and $\textcircled{3}$ is the good-event bound $\bar { \mu } _ { a _ { r } ^ { \star } } ^ { p } \leq \hat { \mu } _ { a _ { r } ^ { \star } } ^ { p } + \beta _ { p } ;$ no challenger satisfies the trigger rule. □

The lemma turns the false-trigger budget into an explicit quantity. Let $F _ { r }$ denote the number of suspicious probes in regime r that occur while the cached candidate is already $a _ { r } ^ { \star } . \mathrm { ~ I f ~ } ( 2 + \lambda _ { p } ) \beta _ { p } <$ $\mathrm { m i n } _ { b \neq a _ { r } ^ { \star } } \Delta _ { b } ^ { ( r ) }$ for all probe blocks in regime r, then $F _ { r } = 0$ on the good event. When the probe is intentionally smaller than this conservative threshold, the theorem below keeps $F _ { r }$ as an explicit price for cheap monitoring.

Theorem 6 (Probe-triggered monitoring guarantee). Under the main reward, flooding, and signstable regime assumptions, replace the minimum-regime condition by

$$
L _ { r } \ge d _ { \mathrm { p r o b e } } + H _ { \mathrm { s c a n } } + D _ { G } ( \delta / 3 ) \qquad f o r ~ e v e r y ~ r e g i m e ~ r .
$$

Then DRFC-Probe satisfies on the good event

$$
R _ { T } = \widetilde { O } \left( \frac { N K q _ { p } h _ { \mathrm { p r o b e } } T } { M } + \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } ( 1 + F _ { r } ) \mathcal { C } _ { r } + N S _ { \mathrm { d e c } } \left( d _ { \mathrm { p r o b e } } + D _ { G } ( \delta / 3 ) \right) \right) + F _ { \partial } N H _ { \mathrm { s c a n } } ,
$$

where $F _ { \partial } \leq \textstyle \sum _ { r } F _ { r }$ counts the boundary-straddling false-triggered scans. In particular, under the no-false-trigger calibration of Lemma 8, namely $\begin{array} { r } { ( 2 + \lambda _ { p } ) \beta _ { p } < \operatorname* { m i n } _ { b \neq a _ { r } ^ { \star } } \Delta _ { b } ^ { ( r ) } } \end{array}$ for every probe block (so $F _ { r } = 0$ for every regime on the good event), we have $F _ { \partial } = 0$ and no full scan straddles a decision boundary, so the boundary-crossing term $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$ of Theorem 5 is absent:

$$
R _ { T } = \widetilde { \cal O } \left( \frac { N K q _ { p } h _ { \mathrm { p r o b e } } T } { M } + \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } { \mathcal C } _ { r } + N S _ { \mathrm { d e c } } \left( d _ { \mathrm { p r o b e } } + D _ { G } ( \delta / 3 ) \right) \right) .
$$

The full-certification monitoring factor $\left( 1 + L _ { r } / M \right)$ of Theorem $^ { 5 }$ is thereby replaced by the probe cost $N K q _ { p } h _ { \mathrm { p r o b e } } T / M$ plus one contained scan per decision regime.

Proof. The periodic probes use $O ( K q _ { p } h _ { \mathrm { p r o b e } } )$ slots per agent per monitoring period and there are at most T /M periods, giving the first term.

No full scan straddles a decision boundary (no-false-trigger case). Between probes DRFC-Probe exploits the cached candidate ${ \hat { a } } ;$ a full active-set scan is launched only in response to a suspicious probe. Fix a switch at $\rho _ { r + 1 }$ and work on the no-false-trigger event, where $\begin{array} { r } { ( 2 + \lambda _ { p } ) \beta _ { p } < \operatorname* { m i n } _ { b \neq a _ { s } ^ { \star } } \Delta _ { b } ^ { ( s ) } } \end{array}$ holds for every probe block in every regime s. By Lemma 8(2) (soundness), no probe whose aggregation window is contained in regime r is suspicious once the cached candidate equals $a _ { r } ^ { \star } .$ so no full scan is launched in $( \rho _ { r } , \rho _ { r + 1 } )$ after the regime-r certificate is in place. Hence at the switch instant $\rho _ { r + 1 }$ the algorithm is exploiting $\hat { a } = a _ { r } ^ { \star }$ or collecting a cheap probe block, never running a full scan. Probe-events occur on a fixed schedule, one every M rounds at times $\{ t _ { 0 } + j M \} _ { j \geq 0 }$ , each occupying $q _ { p } K h _ { \mathrm { p r o b e } }$ rounds. Let s be the first scheduled start with $s \geq \rho _ { r + 1 } ;$ then $s \in [ \rho _ { r + 1 } , \rho _ { r + 1 } + M )$ and its aggregation window $[ s , s + q _ { p } K h _ { \mathrm { p r o b e } } ]$ lies in regime $r + 1$ (a probe-event ongoing at $\rho _ { r + 1 }$ , if any, is the cheap straddling block noted above and is not needed for detection). By Lemma 8(1) (completeness) this probe-event is suspicious for $a _ { r + 1 } ^ { \star }$ , so detection occurs by round $s + q _ { p } K h _ { \mathrm { p r o b e } } < \rho _ { r + 1 } + M + q _ { p } K h _ { \mathrm { p r o b e } }$ . We impose the calibration $q _ { p } K h _ { \mathrm { p r o b e } } \le H _ { \mathrm { s c a n } }$ (a probe-event is no costlier than one full scan); it is satisfiable for $q _ { p } = O ( 1 )$ because under the no-false-trigger size $h _ { \mathrm { p r o b e } } = \Theta ( \log ( K ^ { 2 } T ^ { 3 } / \delta ) / ( N \Delta ^ { 2 } ) )$ both $q _ { p } K h _ { \mathrm { p r o b e } }$ and $H _ { \mathrm { s c a n } }$ are $\Theta \big ( K \log ( K ^ { 2 } T ^ { 3 } / \delta ) / ( N \Delta ^ { 2 } ) \big )$ up to the additive flooding term in $H _ { \mathrm { s c a n } }$ . The confirmation scan is launched at the detection time $\geq \rho _ { r + 1 }$ and runs for at most $H _ { \mathrm { s c a n } }$ rounds, so it ends by

$$
\rho _ { r + 1 } + d _ { \mathrm { p r o b e } } + H _ { \mathrm { s c a n } } \ \leq \ \rho _ { r + 2 } - D _ { G } ( \delta / 3 ) \ < \ \rho _ { r + 2 }
$$

by the stated min-regime-length condition $L _ { r + 1 } \ge d _ { \mathrm { p r o b e } } + H _ { \mathrm { s c a n } } + D _ { G } ( \delta / 3 )$ . The confirmation scan is thus contained in $[ \rho _ { r + 1 } , \rho _ { r + 2 } )$ , and every launched full scan is therefore interior to a single decision regime, so none straddles a boundary and the sign-preservation Lemmas 5–6 (whose hypothesis is exactly single-regime containment, Lemma 7) apply to it. The only boundary-crossing object is at most one cheap probe block per switch, of group cost $O ( N K h _ { \mathrm { p r o b e } } )$ , already inside the first term. If the no-false-trigger calibration is relaxed, a false-triggered scan launched late in regime r may straddle $\rho _ { r + 1 } ;$ each such scan is charged the conservative transition overhead $O ( N H _ { \mathrm { s c a n } } )$ of Theorem 5, contributing the $F _ { \partial } N H _ { \mathrm { s c a n } }$ term with $F _ { \partial } \leq \sum _ { r } F _ { r }$

Certification. Each interior full scan is exactly the active-set scan analyzed in Lemma $7$ and contributes $\widetilde O ( \mathcal { C } _ { r } )$ group regret. There is one necessary confirmation scan per decision regime, plus $F _ { r }$ false-triggered interior scans by the definition of the false-trigger budget, giving $\textstyle \sum _ { r } ( 1 + F _ { r } ) { \mathcal { C } } _ { r }$

Detection delay and synchronization. Until the detecting probe fires, the algorithm exploits the stale candidate $a _ { r } ^ { \star }$ , suboptimal in regime $r + 1$ by at most $\Gamma ^ { ( r + 1 ) } \leq 1$ per pull (the main sign-stable decision-regime definition), for at most $d _ { \mathrm { p r o b e } }$ rounds; this contributes at most $N S _ { \mathrm { d e c } } d _ { \mathrm { p r o b e } }$ group regret (gap-scaled, $\begin{array} { r } { N \sum _ { r } d _ { \mathrm { p r o b e } } \Gamma ^ { ( r + 1 ) } ) } \end{array}$ . Each accepted switch triggers one synchronization window of $D _ { G } ( \delta / 3 )$ rounds, contributing at most $N S _ { \mathrm { d e c } } D _ { G } ( \delta / 3 )$ (gap-scaled, $\begin{array} { r } { N \sum _ { r } D _ { G } ( \delta / 3 ) \Gamma ^ { ( r ) } } \end{array}$ , as in the proof of Theorem 5).

Summing the four contributions gives the first display. On the no-false-trigger event $F _ { r } = 0$ for every r, hence $F _ { \partial } = 0$ , no full scan straddles a boundary, and the boundary-crossing term $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$ vanishes, giving the second display. □

## Minimax Optimality of the Switch-Induced Rate

Pairing the tightened probe upper bound (Theorem 6) with the main information-theoretic lower bound pins the switch-induced, gap-dependent part of the instantaneous-benchmark regret $R _ { T }$ to a single rate.

Corollary 3 (Minimax optimality in $( S _ { \mathrm { d e c } } , K , \Delta , N , D _ { G } ) )$ . Fix $K \ge 3 , N \ge 4 , S _ { \mathrm { d e c } } \ge 1 , \Delta \in$ $( 0 , 1 / 8 )$ , a decentralized communication graph of diameter $D _ { G } \leq N / 4$ with $\Delta \le 1 / ( 8 \sqrt { N D _ { G } } )$ (the conditions of the communication-delay lower bound), and a stationary-margin family in which each regime has constant global gaps with $\Gamma ^ { ( r ) } = \Theta ( \Delta )$ , minimum margin $\Delta _ { i }$ and length $\geq \operatorname* { m a x } \{ d _ { \mathrm { p r o b e } } +$ $H _ { \mathrm { s c a n } } + D _ { G } ( \delta / 3 )$ , c<sub>0</sub>K log $T / ( N \Delta ^ { 2 } ) \}$ (the upper-bound confirming-scan condition and the lowerbound certification-room condition, respectively). Holding the monitoring period M and the probe size $( q _ { p } , h _ { \mathrm { p r o b e } } )$ fixed, and assuming the scan-budget calibration used by the main DRFC theorem together with the probe calibration $q _ { p } K h _ { \mathrm { p r o b e } } \le H _ { \mathrm { s c a n } . }$ , no-false-trigger DRFC-Probe attains on the good event

$$
\begin{array} { r } { R _ { T } = \widetilde { O } \Big ( \underbrace { S _ { \mathrm { d e c } } \frac { K - 1 } { \Delta } } _ { c e r t i f i c a t i o n } + \underbrace { N S _ { \mathrm { d e c } } D _ { G } \Delta } _ { s y n c h r o n i z a t i o n } + \underbrace { N S _ { \mathrm { d e c } } d _ { \mathrm { p r o b e } } } _ { d e t e c t i o n \ d e l a y } + \underbrace { \frac { N K q _ { p } h _ { \mathrm { p r o b e } } T } { M } } _ { b a c k g r o u n d \ p r o b i n g } \Big ) , } \end{array}
$$

the last two terms being the parameter-governed monitoring cost: $d _ { \mathrm { p r o b e } } \leq M + q _ { p } K h _ { \mathrm { p r o b e } }$ , and the background probing scales with $T / M$ and the probe accuracy $h _ { \mathrm { p r o b e } } = \Theta ( \log ( K ^ { 2 } T ^ { 3 } / \delta ) / ( N \Delta ^ { 2 } ) )$ required by the no-false-trigger calibration (so the probing term is not gap-free). Every algorithm that is both rate-consistent and in the decentralized graph-communication class satisfies

$$
\begin{array} { r } { \mathbb { E } [ R _ { T } ] \ \geq \ \operatorname* { m a x } \{ A , B \} \ \geq \ \frac { 1 } { 2 } ( A + B ) , \qquad A = c S _ { \mathrm { d e c } } \frac { K - 1 } { \Delta } , \quad B = c N S _ { \mathrm { d e c } } D _ { G } \Delta . } \end{array}
$$

Hence the switch-induced, gap-dependent part of $R _ { T }$ is

$$
R _ { T } ^ { \mathrm { { s w i t c h } } } = \widetilde { \Theta } \big ( S _ { \mathrm { { d e c } } } ( K - 1 ) / \Delta + N S _ { \mathrm { { d e c } } } D _ { G } \Delta \big ) ,
$$

matched up to factors polylogarithmic in $( K , N , T , 1 / \delta )$ : on this axis DRFC-Probe is rate-optimal in $( S _ { \mathrm { d e c } } , K , \Delta , N , D _ { G } )$ , where “matched” is in the two-instance sense max $\{ A , B \} \ge \frac { 1 } { 2 } ( A + B )$ of the main information-theoretic lower bound, not a single-instance additive lower bound. The boundaryscan term $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$ ofTheorem 5, the one switch-induced gap-dependent term previously without a matching lower bound, is removed (Theorem 6); the residual detection-delay and background-probing costs lie on the separate period/horizon axis governed by M and are not the subject of the lower bound.

Proof. Upper side. In a stationary regime $\Gamma _ { a } ^ { ( r ) } ~ = ~ \Theta ( \Delta _ { a } ^ { ( r ) } )$ , so $\begin{array} { r l r } { \mathcal { C } _ { r } } & { { } = } & { \sum _ { a \ne a _ { r } ^ { \star } } \Gamma _ { a } ^ { ( r ) } / ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } = } \end{array}$ $\Theta ( ( K - 1 ) / \Delta )$ and $\begin{array} { r c l } { \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } \mathcal { C } _ { r } } & { = } & { \Theta ( S _ { \mathrm { d e c } } ( K \textrm { -- } 1 ) / \Delta ) } \end{array}$ . The gap-scaled synchronization cost is $N \sum _ { r } D _ { G } ( \delta / 3 ) \Gamma ^ { ( r ) } ~ = ~ \Theta ( N S _ { \mathrm { d e c } } D _ { G } \Delta )$ (proof of Theorem 5). The no-false-trigger calibration $( 2 + \lambda _ { p } ) \beta _ { p } < \Delta$ is met by a single probe block of size $h _ { \mathrm { p r o b e } } = \Theta ( \log ( K ^ { 2 } T ^ { 3 } / \delta ) / ( N \Delta ^ { 2 } ) )$ , which forces $F _ { r } = 0$ . Substituting into the second (no-false-trigger) display of Theorem 6, whose boundary term $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$ is absent, gives the stated upper bound.

Lower side. This is the main information-theoretic lower bound: the certification cost A holds for every rate-consistent algorithm on the K-arm homogeneous family (Part 1), and the delay cost B holds for every decentralized graph-communication algorithm on the broom family (Part 2). An algorithm in the intersection meets an instance in the union on which $\mathbb { E } [ R _ { T } ] \ \geq \ \operatorname* { m a x } \{ A , B \} \ \geq$ ${ \frac { 1 } { 2 } } ( A + B )$

Matching. The two switch-induced gap-dependent terms (certification and synchronization) are $\widetilde { O } ( S _ { \mathrm { d e c } } ( K - 1 ) / \Delta + N S _ { \mathrm { d e c } } D _ { G } \Delta )$ , and the lower bound is $\Omega ( S _ { \mathrm { d e c } } ( K - 1 ) / \Delta + N S _ { \mathrm { d e c } } D _ { G } \Delta )$ since max $\{ A , B \} \ge \frac { 1 } { 2 } ( A + B )$ , so this part is pinned to $\widetilde { \Theta } ( S _ { \mathrm { d e c } } ( K - 1 ) / \Delta + N S _ { \mathrm { d e c } } D _ { G } \Delta )$ . The remaining detection-delay term $N S _ { \mathrm { d e c } } d _ { \mathrm { p r o b e } }$ is parameter-governed, and the background-probing term $N K q _ { p } h _ { \mathrm { p r o b e } } T / M$ carries the probe accuracy $h _ { \mathrm { p r o b e } } = \Theta ( \log ( K ^ { 2 } T ^ { 3 } / \delta ) / ( N \Delta ^ { 2 } ) )$ ; neither is constrained by the main information-theoretic lower bound. □

Remark 1 (The boundary term $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$ is a conservative artifact, not a minimax floor). The term $N S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } }$ enters Theorem 5 only through $R _ { \mathrm { c r o s s } }$ , which charges the at-most-one boundarystraddling scan per switch at the full group rate N per round for its entire $H _ { \mathrm { s c a n } }$ budget. This is a per-round-1 overcharge of the same kind that the gap-scaled synchronization accounting avoids (proof of Theorem 5): a balanced active-set block pulls every active arm equally, so in expectation over the schedule randomization its group regret is the gap-scaled $h N \sum _ { a \in \mathcal { A } } \Gamma _ { a }$ rather than $h N | \mathcal { A } |$ Recasting $R _ { \mathrm { c r o s s } }$ in this gap-scaled form on a stationary-margin family $\overrightarrow { ( \Gamma ) } = \Theta ( \Delta ) )$ replaces the per-block factor 1 by $\Theta ( \Delta ) _ { ; }$ ; the one delicacy is that a straddling window can postpone a single elimination, and a postponed arm is then cheap exactly when it is slow (a runner-up of gap $\Delta$ costs $\Theta ( \Delta )$ per pull) and fast exactly when it is costly (a sign-flipped arm of gap Θ(1) is eliminated within $\Theta ( n _ { \mathrm { p r e } } \Delta )$ post-boundary blocks), so in both cases the straddle adds only $\widetilde { O } ( ( K - 1 ) / \Delta )$ per switch, not $1 / \Delta ^ { 2 }$ . The clean, fully rigorous route to optimality is nonetheless the probe variant (Theorem 6), where cheap detection guarantees that no full scan ever straddles a boundary, so the term is simply absent and DRFC-Probe is minimax-optimal (Corollary 3). Either way the $1 / \Delta$ gap above the lower bound is not fundamental on this family: no algorithm-independent $\Omega ( 1 / \Delta ^ { 2 } )$ certification lower bound can hold on the gap-regular family $( \Gamma ^ { ( r ) } = \Theta ( \Delta ) )$ , because there the per-switch certification regret is $\Theta \big ( ( K - 1 ) / \Delta \big )$ (Part 1 of the Information-Theoretic Lower Bound subsection, attained by DRFC-Probe): $\Omega ( 1 / \Delta ^ { 2 } )$ samples certify while each suboptimal pull costs only $\Delta .$ This scoping is essential. For irregular gaps $( \Gamma ^ { ( r ) } = \Theta ( 1 )$ , within-regime drift) the certification cost is genuinely $\mathcal { C } _ { r } = \Theta ( ( K - 1 ) / \Delta ^ { 2 } )$ and per-pull regret can be Θ(1), so the $1 / \Delta ^ { 2 }$ there is not an artifact and tightness is not claimed; the present minimax statement (Corollary 3) is confined to the gap-regular family.

## Removing Prior Gap Knowledge

This subsection gives the full Gap-Adaptive DRFC statement and proof. Its sole purpose is to remove prior knowledge of $\Delta _ { \mathrm { m i n } }$ from the scan-budget calibration. The construction uses a persistent-k schedule and a geometric scan-budget schedule. Together they introduce no $\log _ { 2 } ( 1 / \Delta _ { \operatorname* { m i n } } ^ { ( r ) } )$ multiplicative overhead; the only price is an additive log log $T$ term inside the concentration radius, absorbed by $\widetilde { O } .$

Schedule. For uniform notation in this subsection, write the scan-budget function

$$
H _ { \mathrm { s c a n } } ( \Delta ) : = C _ { \mathrm { s c a n } } h \sum _ { a } \left[ \frac { C \log ( K ^ { 2 } T ^ { 3 } / \delta _ { d } ) } { N h \Delta ^ { 2 } } \right] + D _ { G } ( \delta _ { d } / 3 ) ,
$$

so that the fixed-budget $H _ { \mathrm { s c a n } }$ of the main DRFC theorem is the special case $H _ { \mathrm { s c a n } } ( \underline { { \Delta } } _ { \mathrm { f i x } } )$ at a known margin lower bound $\Delta _ { \mathrm { f i x } }$ (with $\delta _ { d }$ replaced by $\delta )$ ; the same symbol $H _ { \mathrm { s c a n } }$ is used as a scalar in the main text and as the function $H _ { \mathrm { s c a n } } ( \cdot )$ in this subsection only. Fix an optimistic initial margin guess $\Delta ^ { ( 0 ) } \in ( 0 , 1 ) \ ( \mathrm { e . g . } \ \Delta ^ { ( 0 ) } = 1 / 2 )$ and a margin floor $\underline { { \Delta } } \in ( 0 , \Delta ^ { ( 0 ) } ] \ ( \mathrm { e . g . } \ \underline { { \Delta } } = 1 / T )$ . Set $k _ { \mathrm { m a x } } = \lceil \log _ { 2 } ( \Delta ^ { ( 0 ) } / \underline { { \Delta } } ) ^ { - }$ ⌉ and choose the per-level confidence allotment $\delta _ { d } = \delta / ( 3 ( k _ { \operatorname* { m a x } } + 1 ) )$ . For each level $0 \leq k \leq k _ { \operatorname* { m a x } }$ , define

$$
\Delta ^ { ( k ) } = \Delta ^ { ( 0 ) } \cdot 2 ^ { - k } , \qquad H _ { \mathrm { s c a n } } ^ { ( k ) } = H _ { \mathrm { s c a n } } ( \Delta ^ { ( k ) } ) .
$$

Gap-Adaptive DRFC runs the active-set scan at level $k = 0$ (small budget, optimistic margin); if a scan exits without unique certification within $H _ { \mathrm { s c a n } } ^ { ( k ) }$ wall-clock slots, the agent shrinks the margin guess by setting $k \gets k + 1$ and reattempting on the next monitoring trigger, until either a unique winner is certified or k reaches $k _ { \mathrm { m a x } }$ . At level $k _ { \operatorname* { m a x } } .$ , the budget is large enough to certify any regime with $\Delta _ { \mathrm { m i n } } ^ { ( r ) } \ge 2 \Delta$

Proposition 2 (Gap-Adaptive DRFC). Suppose the main reward, flooding, and sign-stable decisionregime conditions hold, and replace the main minimum-regime-length assumption by the anytime version

$$
\rho _ { r + 1 } - \rho _ { r } \geq M + C _ { \mathrm { d b } } H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } ) + D _ { G } ( \delta / 3 ) , \qquad \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } = \operatorname* { m i n } _ { r ^ { \prime } \leq r } \operatorname* { m i n } _ { a \neq a _ { r ^ { \prime } } ^ { \star } } \Delta _ { a } ^ { ( r ^ { \prime } ) } ,
$$

for a universal constant $C _ { \mathrm { d b } } \ ( e . g . , C _ { \mathrm { d b } } = 4 )$ . Here $\tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) }$ is the smallest gap seen up to regime $^ { r , }$ the gap the persistent level tracks, and $C _ { \mathrm { d b } } H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } )$ covers three pieces. The first is the geometric walk-up to the persistent level $\tilde { k } _ { r }$ , whose summed wall-clock budget telescopes to $O ( H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } ) )$ by the $4 ^ { - k }$ schedule with no level factor. The second is one boundary-crossing scan at the persistent level. The third is one contained certifying scan at level $\tilde { k } _ { r }$ . Then Gap-Adaptive DRFC, run with arbitrary optimistic guess $\Delta ^ { ( 0 ) } \in ( 0 , 1 )$ and floor $\begin{array} { r } { \underline { { \Delta } } \le \operatorname* { m i n } _ { r } \Delta _ { \operatorname* { m i n } } ^ { ( r ) } / 2 } \end{array}$ , and which retains the current level k across monitoring triggers (incrementing only on a failed scan), satisfies with probability at least $1 - \delta$

$$
R _ { T } = \tilde { \cal O } \left( \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } \left( 1 + \frac { L _ { r } } { M } \right) \mathcal C _ { r } + N M S _ { \mathrm { d e c } } + N D _ { G } ( \delta / 3 ) S _ { \mathrm { d e c } } + N \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } ) \right) .
$$

The running-minimum gap $\begin{array} { r } { \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } = \operatorname* { m i n } _ { r ^ { \prime } \leq r } \Delta _ { \mathrm { m i n } } ^ { ( r ^ { \prime } ) } } \end{array}$ replaces the per-regime $\Delta _ { \mathrm { m i n } } ^ { ( r ) }$ because the optimistic level is never lowered, so an easy regime following a hard one inherits the harder regime’s budget. The two coincide when the gaps are non-increasing across regimes, and in general $\tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } \leq \Delta _ { \mathrm { m i n } } ^ { ( r ) }$

Proof. The good-event union bound now ranges over the doubling levels $k = 0 , \ldots , k _ { \mathrm { m a x } }$ as well as over W. Each level uses confidence $\delta _ { d } = \delta / ( 3 ( k _ { \operatorname* { m a x } } + 1 ) )$ , so the union bound across levels closes within the budget $\delta / 3$ already reserved for the concentration event in Lemma 4. Adding $k _ { \mathrm { m a x } } + 1 = O ( \log _ { 2 } ( \Delta ^ { ( 0 ) } / \Delta ) )$ levels increases each $\log ( K ^ { 2 } T ^ { 3 } / \delta _ { d } )$ factor by an additive log log $T$ term, absorbed by Oe.

For a regime r with true minimum margin $\Delta _ { \operatorname* { m i n } } ^ { ( r ) } .$ , let $k _ { r } = \lceil \log _ { 2 } ( 2 \Delta ^ { ( 0 ) } / \Delta _ { \operatorname* { m i n } } ^ { ( r ) } ) \rceil$ be the smallest level at which $\Delta ^ { ( k _ { r } ) } \leq \Delta _ { \operatorname* { m i n } } ^ { ( r ) } / 2$ , equivalently the first level whose scan budget $H _ { \mathrm { s c a n } } ( \Delta ^ { ( k _ { r } ) } )$ is suficient for the regime by Lemma 7. The floor assumption $\Delta \le \Delta _ { \mathrm { m i n } } ^ { ( r ) } / 2$ guarantees $k _ { r } \leq k _ { \operatorname* { m a x } }$

Persistent-k schedule. The level k is retained across monitoring triggers and across regime boundaries, and is incremented only when a scan exits without a unique winner, so it is non-decreasing in time. Write $\begin{array} { r } { \tilde { k } _ { r } = \operatorname* { m a x } _ { r ^ { \prime } \leq r } k _ { r ^ { \prime } } = \lceil \log _ { 2 } ( 2 \Delta ^ { ( 0 ) } / \tilde { \Delta } _ { \operatorname* { m i n } } ^ { ( r ) } ) \rceil } \end{array}$ for the running-maximum certifying level, which corresponds to the running-minimum gap $\tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) }$ . Once the schedule first reaches level $\tilde { k } _ { r }$ it never falls back, so every subsequent interior scan in regime r runs at level $\tilde { k } _ { r }$ , and this level certifies regime r because $\Delta ^ { ( \tilde { k } _ { r } ) } \leq \Delta ^ { ( k _ { r } ) } \leq \Delta _ { \operatorname* { m i n } } ^ { ( r ) } / 2$ , a smaller optimistic margin only enlarging the scan budget and remaining suficient for the regime’s true gap by Lemma 7. At a regime boundary $\rho _ { r + 1 }$ the carried-over level is $\tilde { k } _ { r }$ , so a walk-up at regime $r + 1$ occurs only when the new regime sets a deeper running minimum, that is when $k _ { r + 1 } > \tilde { k } _ { r }$ equivalently $\Delta _ { \mathrm { m i n } } ^ { ( r + 1 ) } < \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } ,$ , and only the increment from $\tilde { k } _ { r }$ to $\tilde { k } _ { r + 1 }$ is paid. When the new regime is easier, with $\Delta _ { \mathrm { m i n } } ^ { ( r + 1 ) } \ge \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) }$ , the persistent level is already suficient and no walk-up occurs, at the price of running that regime’s scans at the conservative budget $H _ { \mathrm { s c a n } } ( \Delta ^ { ( \tilde { k } _ { r } ) } ) = O ( H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } ) )$ rather than the cheaper $H _ { \mathrm { { s c a n } } } ( \Delta _ { \mathrm { { m i n } } } ^ { ( r + 1 ) } )$ .

Wasted-level cost. A wasted-level attempt at level $k < \tilde { k } _ { S _ { \mathrm { d e c } } }$ exits without certification within $H _ { \mathrm { { s c a n } } } ( \Delta ^ { ( k ) } )$ wall-clock slots and contributes at most $H _ { \mathrm { { s c a n } } } ( \Delta ^ { ( \tilde { k } ) } )$ · N group regret. By the $\Delta ^ { ( k ) } =$ $\Delta ^ { ( 0 ) } 2 ^ { - k }$ schedule and $H _ { \mathrm { s c a n } } ( \Delta ) \propto 1 / \Delta ^ { 2 }$ (up to $\widetilde O$ factors absorbed in $H _ { \mathrm { s c a n } } ) , ~ H _ { \mathrm { s c a n } } ( \Delta ^ { ( k ) } ) ~ \asymp$ $4 ^ { k - \tilde { k } _ { S _ { \mathrm { d e c } } } } H _ { \mathrm { s c a n } } ( \Delta ^ { ( \tilde { k } _ { S _ { \mathrm { d e c } } } ) } )$ , so the geometric sum of all walk-up budgets across the whole horizon telescopes to

$$
\sum _ { k < \tilde { k } _ { S _ { \mathrm { d e c } } } } H _ { \mathrm { s c a n } } ( \Delta ^ { ( k ) } ) \le \frac { 4 } { 3 } H _ { \mathrm { s c a n } } ( \Delta ^ { ( \tilde { k } _ { S _ { \mathrm { d e c } } } ) } ) = O ( H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( S _ { \mathrm { d e c } } ) } ) ) ,
$$

the single deepest level, with no $k _ { \mathrm { m a x } }$ factor. Because k is non-decreasing it passes through each level at most once, so the walk-up is paid at most once globally rather than once per regime, and is bounded by the last summand of $\begin{array} { r } { N \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } ) } \end{array}$

Certifying-level scans. At the persistent level $\tilde { k } _ { r } .$ , each interior scan certifies on the good event by Lemma 7 with margin $\Delta ^ { ( \tilde { k } _ { r } ) } \leq \Delta _ { \operatorname* { m i n } } ^ { ( r ) }$ , contributing $\widetilde O ( \mathcal { C } _ { r } )$ group regret. The certification cost stays at the true-gap level $\mathcal { C } _ { r }$ rather than the conservative $\tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) }$ , because the anytime scan exits as soon as the unique winner is certified, set by the regime’s own gaps, while the margin guess $\Delta ^ { ( \tilde { k } _ { r } ) }$ only caps the give-up budget. The number of such scans per regime is at most $1 + L _ { r } / M$ , exactly as in Theorem 5.

Boundary-crossing accounting. At most one scan straddles each decision switch by the atmost-one-active-scan rule, and it runs at the persistent level $\tilde { k } _ { r }$ . Its wall-clock budget is at most $H _ { \mathrm { s c a n } } ( \Delta ^ { ( \tilde { k } _ { r } ) } ) = O ( H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } ) )$ , so summed over the $S _ { \mathrm { d e c } } + 1$ regimes the boundary-crossing scans give the $\begin{array} { r } { N \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } ) } \end{array}$ term of the display. Earlier failed-level attempts at the same regime have already terminated under the persistent-k schedule and are charged to the wasted-level term, not double-counted into $R _ { \mathrm { c r o s s } }$

Minimum-regime-length suficiency. The condition $\rho _ { r + 1 } - \rho _ { r } \ge M + C _ { \mathrm { d b } } H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } ) + D _ { G } ( \delta / 3 )$ is enough to fit, in order: one monitoring wait (M), the geometric walk-up to the persistent level plus one certifying scan and one boundary-crossing scan (all $O ( H _ { \mathrm { s c a n } } ( \tilde { \Delta } _ { \mathrm { m i n } } ^ { ( r ) } ) )$ , with $C _ { \mathrm { d b } } = 4 )$ , and one synchronization window $\left( D _ { G } ( \delta / 3 ) \right)$ . The running-minimum gap enters because the persistent level never falls back, so even an easy regime must budget for the conservative scan length set by the hardest regime seen so far. Hence the doubling sequence completes inside each regime without crossing a decision boundary more than once.

Summing the wasted-level, certifying, monitoring, sync, and boundary-crossing contributions yields the display. □

Remark 2 (Margin-free anytime form). Choosing the floor $\underline { { \Delta } } = 1 / T$ together with an optimistic guess $\Delta ^ { ( 0 ) } = 1 / 2$ gives $k _ { \mathrm { m a x } } = \lceil \log _ { 2 } ( T / 2 ) \rceil = O ( \log T )$ and the regret bound holds simultaneously for every regime r with $\Delta _ { \mathrm { m i n } } ^ { ( r ) } \geq 2 / T$ . The only level-related overhead in the display is the additive log log T inside $\widetilde O$ coming from the union bound over the $k _ { \operatorname* { m a x } { } } + 1$ doubling levels; no $\log _ { 2 } ( 1 / \Delta _ { \operatorname* { m i n } } ^ { ( r ) } )$ factor multiplies $\mathcal { C } _ { r }$ or $H _ { \mathrm { s c a n } }$ thanks to the geometric walk-up and persistent-k schedule. This is the form one would deploy in practice when no margin lower bound is available: doubling shrinks $\Delta ^ { ( k ) }$ from $\Delta ^ { ( 0 ) }$ downward, growing the scan budget geometrically, until a level whose budget exceeds the true regime requirement is reached.

## Distinct Source-Record Complexity

This subsection counts unique source-traceable payloads before graph-level retransmission. In the active-set implementation, each source, active arm, and block creates one keyed comparison record. Let

$$
B _ { r } = \sum _ { a \ne a _ { r } ^ { \star } } \frac { 1 } { h ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } .
$$

Proposition 3 (Distinct source-record complexity). On the good event, conservative active-set DRFC creates, up to logarithmic factors, at most

$$
\widetilde { O } \left( \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } \left( 1 + \frac { L _ { r } } { M } \right) \mathcal { B } _ { r } + N K S _ { \mathrm { d e c } } \frac { H _ { \mathrm { s c a n } } } { h } + N S _ { \mathrm { d e c } } \right)
$$

distinct source-traceable comparison and switch records. The probe-triggered variant creates at most

$$
\widetilde { \cal O } \left( \frac { N K q _ { p } T } { M } + \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } } ( 1 + F _ { r } ) \mathcal { B } _ { r } + N K S _ { \mathrm { d e c } } \frac { H _ { \mathrm { s c a n } } } { h } + N S _ { \mathrm { d e c } } \right)
$$

distinct source records. If one distinct source record is flooded using at most $Q _ { G } ( \delta )$ unit edge transmissions with high probability, multiplying the displays by $Q _ { G } ( \delta )$ gives the corresponding edge-level trafic bound. For full-neighbor flooding, the crude deterministic bound $Q _ { G } ( \delta ) \leq N ^ { 2 } D _ { G } ( \delta )$ always holds.

Proof. During one interior full-certification scan in regime r, the number of active blocks charged to a non-best arm a is

$$
\tau _ { a } = \widetilde { O } \left( \frac { 1 } { N h ( \Delta _ { a } ^ { ( r ) } ) ^ { 2 } } \right)
$$

by Lemma 7. The best arm remains active until the scan ends, but its active-block count is at most the maximum non-best active-block count and is therefore dominated, up to constants, by $\textstyle \sum _ { a \neq a _ { r } ^ { \star } } \tau _ { a }$ . Thus the number of distinct comparison records in one scan is

$$
N \sum _ { a \in [ K ] } \tau _ { a } = \widetilde { \cal O } ( B _ { r } ) .
$$

Summing over at most $1 + L _ { r } / M$ scans per regime gives the conservative DRFC comparison-record term. Boundary-crossing scans are not certified by the regime-contained argument. There is at most one such scan per decision switch, each is aborted or completed within $H _ { \mathrm { s c a n } }$ wall-clock slots, and each active-set block creates at most NK source-arm records. The conservative contribution is therefore

$$
O \left( N K S _ { \mathrm { d e c } } \frac { H _ { \mathrm { s c a n } } } { h } \right) ,
$$

where the division by h only converts the slot budget into a crude block-count upper bound. Switch synchronization creates at most $O ( N )$ distinct control records per decision switch under the deterministic epoch-tie-breaking convention, so switch records contribute $O ( N S _ { \mathrm { d e c } } )$

For DRFC-Probe, every monitoring probe creates $O ( N K q _ { p } )$ distinct probe records, and there are at most $T / M$ probe times. Every suspicious probe launches the same full-certification scan as above. The number of necessary scans is one per decision regime, and the number of unnecessary scans in regime $r$ is $F _ { r }$ by definition. Boundary-crossing confirmation scans are again charged by the conservative $O ( N K S _ { \mathrm { d e c } } H _ { \mathrm { s c a n } } / h )$ term. This gives the second display. The edge-level trafic statement follows by multiplying the source-record count by the per-record flooding bound $Q _ { G } ( \delta )$ Under full-neighbor flooding, a record is sent over at most all directed edges in each of at most $D _ { G } ( \delta )$ rounds, giving the conservative $N ^ { 2 } D _ { G } ( \delta )$ bound. □

## Separation from Local-Change-Reactive Protocols

Definition 2 (Local-change-reactive protocol). For constants $\alpha , \eta \in ( 0 , 1 ) , \gamma \in ( 0 , 1 / 2 )$ , and $\tau \geq 0$ a protocol is $( \alpha , \gamma , \eta , \tau )$ -reactive $i f ,$ whenever at least αN agents experience local mean jumps of magnitude at least $2 \gamma$ within a window of length $\tau ,$ , it initiates a global reset, fresh tournament, or control synchronization with probability at least $1 - \eta$ . Each initiated reaction costs at least $c _ { \mathrm { r e s e t } }$ group regret or at least $c _ { \mathrm { m s g } }$ control messages.

Fix $\alpha , \eta \in ( 0 , 1 ) , \gamma \in ( 0 , 1 / 2 )$ , and tolerance $\tau \geq 0$ , and choose N even. Consider $K = 2$ . Choose ϵ satisfying $\gamma \le \epsilon < 1 / 2$ , and then choose $0 < \Delta < \mathrm { m i n } \{ 2 \epsilon , ~ 1 / 2 - \epsilon \}$ . The constraint $\epsilon < 1 / 2$ keeps this interval nonempty, since $\epsilon \geq \gamma > 0$ gives $2 \epsilon > 0$ while $\epsilon < 1 / 2$ gives $1 / 2 - \epsilon > 0$ , so min $\{ 2 \epsilon , 1 / 2 - \epsilon \} > 0$ . Such an ϵ exists for every $\gamma \in ( 0 , 1 / 2 )$ , for instance $\epsilon = \gamma$ , so the construction covers the full range of Definition 2 with no restriction beyond $\gamma < 1 / 2$ . The requirement $\Delta < 2 \epsilon$ is what guarantees that a sign flip $\sigma _ { i } ^ { ( s ) } \mapsto - \sigma _ { i } ^ { ( s ) }$ swaps the per-agent best arm at agent i: with $\sigma _ { i } = - 1$ the local means become $\mu _ { i , 1 } = 1 / 2 + \Delta - \epsilon$ and $\mu _ { i , 2 } = 1 / 2 + \epsilon$ , so arm 2 becomes locally best if $2 \epsilon > \Delta$ . For segment s, define

$$
\mu _ { i , 1 } ^ { ( s ) } = \frac { 1 } { 2 } + \Delta + \sigma _ { i } ^ { ( s ) } \epsilon , \qquad \mu _ { i , 2 } ^ { ( s ) } = \frac { 1 } { 2 } - \sigma _ { i } ^ { ( s ) } \epsilon ,
$$

where $\sigma _ { i } ^ { ( s ) } \in \{ - 1 , + 1 \}$ and $\begin{array} { r } { \sum _ { i } \sigma _ { i } ^ { ( s ) } = 0 } \end{array}$ . The choices of $\epsilon$ and $\Delta$ keep all means in $[ 0 , 1 ]$ . At each local-change boundary, flip the signs of at least αN agents while preserving the zero-sum condition by flipping equal numbers of +1 and −1 signs; this is possible after increasing $N$ by a constant factor if needed. Then

$$
\bar { \mu } _ { 1 } ^ { ( s ) } = \frac { 1 } { 2 } + \Delta , \quad \quad \bar { \mu } _ { 2 } ^ { ( s ) } = \frac { 1 } { 2 } ,
$$

so $S _ { \mathrm { d e c } } = 0$ while $S _ { \mathrm { l o c } } = \Theta ( T / L )$ if signs flip every L rounds.

For every flipped agent, both local arm means change by $2 \epsilon \geq 2 \gamma$ . The flips happen at a common boundary across all flipped agents, so all relevant local-jump boundaries lie within a window of length $0 \leq \tau$ regardless of the tolerance τ . By Definition 2, an $( \alpha , \gamma , \eta , \tau )$ -local-change-reactive

protocol initiates a reset, fresh tournament, or control synchronization at each such boundary with probability at least $1 - \eta$ . If each initiated reaction costs at least $c _ { \mathrm { r e s e t } }$ group regret or $c _ { \mathrm { m s g } }$ control messages, linearity of expectation gives

$$
\Omega ( ( 1 - \eta ) S _ { \mathrm { l o c } } c _ { \mathrm { r e s e t } } )
$$

expected reset regret or

$$
\Omega ( ( 1 - \eta ) S _ { \mathrm { l o c } } c _ { \mathrm { m s g } } )
$$

expected control communication.

For DRFC, the global pairwise gap is constant and positive throughout the instance. By Lemmas 5 and 4, local sign flips do not create a decision-relevant switch, so the regret bound has no $S _ { \mathrm { l o c } }$ adaptation term.

## Reactive-Class Instantiations

This subsection verifies that two standard non-stationary bandit baselines fall inside the comparator class of Definition 2 when applied to the construction in Appendix , and derives explicit lower bounds on their reaction cost. Both examples treat the per-agent estimator as the natural object; group-level aggregation does not avoid the cost because the protocol is required to take a coordinated action whenever a suficiently large agent subset triggers.

Example 1: per-agent sliding-window UCB. Each agent i runs SW-UCB with window length W, maintaining a local index

$$
I _ { i , a , t } = \hat { \mu } _ { i , a , t } ^ { W } + \sqrt { \frac { \xi \log t } { n _ { i , a , t } ^ { W } } } ,
$$

where $\hat { \mu } _ { i , a , t } ^ { W }$ is the empirical mean over the last W pulls of arm a and $n _ { i , a , t } ^ { W }$ is the corresponding pull count. A protocol-level reaction (group reset of the candidate arm, fresh tournament, or contro synchronization) is triggered whenever the per-agent best arm changes at $\alpha N$ or more agents within a time window of length $\tau \geq W$

Membership. At any local-flip boundary in the construction, each flipped agent has its local arm means swap roles by margin $2 \epsilon \geq 2 \gamma$ . A standard SW-UCB analysis Garivier and Moulines [2011] shows that on the high-probability event, the empirical index ordering at a flipped agent flips within W rounds whenever the per-agent gap exceeds 4 $\sqrt { \xi \log T / W } ;$ since the per-agent gap is 2γ, the condition $2 \gamma \ge 4 \sqrt { \xi \log T / W }$ holds as soon as $W \ge 4 \xi \log T / \gamma ^ { 2 }$ . Thus for $W \ge 1 6 \xi \log T / \gamma ^ { 2 }$ and $\tau \geq W$ , SW-UCB triggers a per-agent index switch at every flipped agent within the tolerance window with probability at least $1 - 1 / T$ . Setting $\eta = 1 / T$ shows SW-UCB lies in the $( \alpha , \gamma , \eta , \tau )$ reactive class for the construction.

Reaction cost. Each triggered reaction invalidates the SW-UCB sliding-window estimator at the afected agents, so the next $\Theta ( W )$ rounds per agent are spent re-exploring instead of exploiting. On the construction, the average per-agent pull regret per re-exploration block is at least a constant fraction of $\Delta$ (the global decision margin), since the per-agent best arm flips but the global best arm does not. This yields $c _ { \mathrm { r e s e t } } \geq c _ { 1 } \alpha N W \Delta$ group regret per local-flip boundary for a universal constant $c _ { 1 }$ , and substituting the minimum W above gives

$$
c _ { \mathrm { r e s e t } } \geq c _ { 1 } \alpha N \Delta \cdot \frac { 4 \xi \log T } { \gamma ^ { 2 } } .
$$

Across $S _ { \mathrm { l o c } } = \Theta ( T / L )$ boundaries this is $\Omega ( ( 1 - \eta ) S _ { \mathrm { l o c } } c _ { \mathrm { r e s e t } } )$ expected reset regret as in the main class-relative separation theorem.

Example 2: per-agent CUSUM change-point detector. Each agent i runs CUSUM on the local reward stream of its currently played arm: maintain $S _ { i , t } = \operatorname* { m a x } ( 0 , S _ { i , t - 1 } + X _ { i , A _ { i , t } , t } - \hat { \mu } _ { i , A _ { i , t } } - \zeta )$ declare a per-agent change when $S _ { i , t } > h _ { \mathrm { c u s u m } }$ , and on detection emit a control message that triggers a global fresh tournament if a fraction αN of agents declare a change inside a time window of length $\tau .$

Membership. On the construction, a local-flip boundary creates a mean shift of magnitude $2 \epsilon \geq$ $2 \gamma$ at every flipped agent for the currently played arm. Classical CUSUM bounds Page [1954], Lai [1995] show that for a single-agent mean shift of magnitude $2 \gamma$ , the expected detection delay is $E _ { \theta } [ \tau _ { \mathrm { c u s u m } } ] = O ( h _ { \mathrm { c u s u m } } / ( 2 \gamma ) ^ { 2 } )$ with false-alarm rate $1 / T$ when $h _ { \mathrm { c u s u m } } = \Theta ( \log T )$ . Setting $\tau \geq$ $\Theta ( \log T / \gamma ^ { 2 } )$ then guarantees that at least αN agents declare a change within the tolerance window with probability at least $1 - 1 / T$ . Thus CUSUM lies in the $( \alpha , \gamma , \eta , \tau )$ -reactive class with $\eta = 1 / T$ on the construction.

Reaction cost. A triggered fresh tournament across N agents and $K = 2$ arms requires at least $\Theta ( 1 / \Delta ^ { 2 } )$ group samples to certify the global best arm (any pairwise certification across heterogeneous local means needs $\Omega ( 1 / \Delta ^ { 2 } )$ group samples by the standard fresh-comparison lower bound), and a constant fraction of these samples pay group regret $\Theta ( \Delta )$ per pull because the tournament probes both arms. This yields $c _ { \mathrm { r e s e t } } \geq c _ { 2 } N / \Delta$ group regret per detection for a universal constant $c _ { 2 } .$ , where the factor N comes from running the tournament across all N agents. For the controlmessage cost, the tournament also broadcasts at least $\Omega ( N )$ source-traceable records before the candidate is committed, giving $c _ { \mathrm { m s g } } \geq c _ { 3 } N$ per detection. Across $S _ { \mathrm { l o c } }$ boundaries these accumulate to $\Omega ( ( 1 - \eta ) S _ { \mathrm { l o c } } c _ { \mathrm { r e s e t } } )$ or $\Omega ( ( 1 - \eta ) S _ { \mathrm { l o c } } c _ { \mathrm { m s g } } )$ as in the main class-relative separation theorem.

Remark. Both examples derive their reaction cost from the per-detection cost of any natural global response that a local-change detector can trigger. A protocol that detects local changes but refuses to react would exit the comparator class trivially, but it would also pay $\Theta ( T )$ regret under any true decision switch and is therefore not a meaningful baseline. A protocol that monitors a decision-level signal directly—such as $\mathrm { D R F C } ,$ which monitors certified pairwise lower-confidence gaps on the equal-agent global projection—does not trigger on local jumps in the first place and is outside the class by design.

## Information-Theoretic Lower Bound

This subsection proves the main information-theoretic lower bound. The argument has two parts: a per-switch Fano-method certification cost, and a graph-communication delay cost.

## Part 1: Certification Cost

Construction. Fix $K \geq 2 , N \geq 1 , S _ { \mathrm { d e c } } \geq 1 .$ , and $\Delta \in ( 0 , 1 / 8 )$ (matching the main informationtheoretic lower bound; the single-arm shift below reaches $\textstyle { \frac { 1 } { 2 } } + 2 \Delta < \frac { 3 } { 4 } )$ . Partition the horizon [T] into $S _ { \mathrm { d e c } } + 1$ regimes of equal length $L = \lfloor T / ( S _ { \mathrm { d e c } } + 1 ) \rfloor$ , with the convention that the last regime absorbs any leftover rounds. For regime $r \in \{ 0 , \ldots , S _ { \mathrm { d e c } } \}$ , define the regime’s best arm as $a _ { r } ^ { \star } = ( r \ \mathrm { m o d } \ K ) { + } 1$ For $K \geq 2$ consecutive values difer, so every regime boundary is a genuine decision switch and the instance has exactly $S _ { \mathrm { d e c } }$ of them. Set all agents homogeneous: for every agent $i ,$ arm $^ { a , }$ and round t in regime $r ,$

$$
\mu _ { i , a , t } = \left\{ { \begin{array} { l l } { { \frac { 1 } { 2 } } + \Delta } & { { \mathrm { i f ~ } } a = a _ { r } ^ { \star } , } \\ { { \frac { 1 } { 2 } } } & { { \mathrm { i f ~ } } a \neq a _ { r } ^ { \star } . } \end{array} } \right.
$$

Rewards are independent Bernoulli with these means. This is a valid homogeneous instance P with $S _ { \mathrm { d e c } }$ decision switches, minimum margin $\Delta ,$ and $S _ { \mathrm { l o c } } = S _ { \mathrm { d e c } }$ . The relevant family is $P$ together with the single-arm alternatives $\{ Q _ { a } \}$ defined below, one per suboptimal arm and post-switch regime. Rate-consistency (defined below) is imposed on every member of this family, which is what forbids hard-coding. An algorithm pinned to the cyclic sequence plays a suboptimal arm for an entire regime on each $Q _ { a }$ where another arm is best, violating $\mathbb { E } [ n _ { b } ^ { ( r ) } ] \le C _ { \mathrm { a l g } }$ log T there, so it is not rate-consistent. The bound below therefore constrains every rate-consistent algorithm on the fixed instance $P _ { \mathrm { : } }$ , with no Bayes prior on the construction. Because agents are homogeneous, a centralized oracle that pools all observations faces the same statistical task as the decentralized system, so any lower bound for a centralized algorithm applies a fortiori to decentralized algorithms.

Per-regime Fano argument. Fix a decision-switch boundary at $\rho _ { r + 1 }$ and consider the new regime $r + 1$ . Although P fixes $a _ { r + 1 } ^ { \star }$ , a rate-consistent algorithm must perform well on the whole family of regime- $( r { + } 1 )$ variants that share the regime-≤ r prefix and difer only in which arm is best after the switch. We lower-bound its worst-case per-regime cost by a Yao averaging device, a warm-up later sharpened to the log-free per-arm bound. Consider $K$ hypotheses $\{ H _ { k } \} _ { k = 1 } ^ { K }$ , indexed by the identity of the new best arm: under $H _ { k }$ , arm $k$ has global mean $1 / 2 + \Delta$ and all other arms have global mean $1 / 2$ throughout regime $r + 1$ . Put the uniform prior $\theta \sim \mathrm { U n i f } [ K ]$ over which arm is best in regime $r + 1$ . By Yao’s principle the worst-case cost over $\{ H _ { k } \}$ is at least this Bayes average, so a bound under the prior is a valid worst-case lower bound on the family.

Let $\theta \in [ K ]$ denote the random hypothesis, and let $\hat { \theta }$ be any estimator of θ based on all observations up to the end of regime $r + 1$ . By Fano’s inequality Tsybakov [2009],

$$
\mathbb { P } ( \hat { \theta } \neq \theta ) \geq 1 - \frac { I ( \theta ; \mathbf { X } _ { r + 1 } ) + \log 2 } { \log K } ,
$$

where $I ( \theta ; { \bf X } _ { r + 1 } )$ is the mutual information between the hypothesis and all reward observations in regime $r + 1$ across all agents.

Mutual-information bound. Under $H _ { k } .$ , agent i pulls arm $A _ { i , t }$ at round t and observes $X _ { i , A _ { i , t } , t } \sim$ Ber $\cdot ( \mu _ { A _ { i , t } } )$ where $\mu _ { a } = 1 / 2 + \Delta \cdot { \bf 1 } [ a = k ]$ . The observations of arm a by all agents across regime $r + 1$ are i.i.d. Bernoulli conditional on $H _ { k }$ . The conditional distribution difers across hypotheses only for the arm that is special under each hypothesis. By the chain rule and the data-processing inequality,

$$
I ( \theta ; \mathbf { X } _ { r + 1 } ) \leq \sum _ { a = 1 } ^ { K } \mathbb { E } \Big [ n _ { a } ^ { ( r + 1 ) } \Big ] \cdot d ( a ) ,
$$

where $\begin{array} { r } { n _ { a } ^ { ( r + 1 ) } = \sum _ { i = 1 } ^ { N } \sum _ { t \in \mathrm { r e g i m e } \ r + 1 } \mathbf { 1 } [ A _ { i , t } = a ] } \end{array}$ is the total number of pulls of arm a by all agents in regime $r + 1$ , the expectation is under the mixture $\theta \sim \mathrm { U n i f } [ K ]$ , and

$$
d ( a ) = \operatorname* { m a x } _ { k \neq j } \mathrm { k l } \left( \mu _ { a } ^ { ( k ) } , \mu _ { a } ^ { ( j ) } \right) ,
$$

with $\mathrm { k l } ( p , q ) = p \log ( p / q ) + ( 1 - p ) \log ( ( 1 - p ) / ( 1 - q ) )$ the binary KL divergence. Under the construction:

• I $\dot { \mathrm { \boldsymbol { \cdot } } } \ k = a \ \mathrm { a n d } \ j \neq a \mathrm { : } \ \mathrm { k l } ( 1 / 2 + \Delta , 1 / 2 )$

• If $k \neq a$ and $j = a \colon \mathrm { k l } ( 1 / 2 , 1 / 2 + \Delta )$

• If $k \neq a$ and $j \neq a \colon \operatorname { k l } ( 1 / 2 , 1 / 2 ) = 0$

Since k $1 ( 1 / 2 + \Delta , 1 / 2 ) \le 4 \Delta ^ { 2 } / ( 1 - 4 \Delta ^ { 2 } ) \le 8 \Delta ^ { 2 }$ for $\Delta \in ( 0 , 1 / 4 )$ and similarly $\mathrm { k l } ( 1 / 2 , 1 / 2 + \Delta ) \le 8 \Delta ^ { 2 }$ we have $d ( a ) \leq 8 \Delta ^ { 2 }$ for every arm a. Therefore

$$
I ( \theta ; { \bf X } _ { r + 1 } ) \le 8 \Delta ^ { 2 } \sum _ { a = 1 } ^ { K } \mathbb { E } [ n _ { a } ^ { ( r + 1 ) } ] = 8 \Delta ^ { 2 } \cdot N L .
$$

Regret extraction. Fix any algorithm and any realization of θ. Under $H _ { \theta }$ , the group regret in regime $r + 1$ is

$$
R _ { r + 1 } = \Delta \sum _ { a \neq \theta } n _ { a } ^ { ( r + 1 ) } .
$$

Taking the expectation over $\theta \sim \mathrm { U n i f } [ K ]$ and the algorithm’s randomness,

$$
\mathbb { E } [ R _ { r + 1 } ] = \Delta \mathbb { E } \left[ \sum _ { a \neq \theta } n _ { a } ^ { ( r + 1 ) } \right] .
$$

By symmetry of the construction (all arms are exchangeable under the mixture),

$$
\mathbb { E } \left[ \sum _ { a \neq \theta } n _ { a } ^ { ( r + 1 ) } \right] = \frac { K - 1 } { K } \cdot N L ,
$$

minus the reduction from the algorithm’s ability to identify θ. More precisely, let $p _ { e } = \mathbb { P } ( \hat { \boldsymbol { \theta } } \neq \boldsymbol { \theta } )$ . The next display is a non-binding Fano-style heuristic, superseded by the rigorous single-arm argument below and included only to motivate the regime split, so we do not rely on it. Heuristically, if the algorithm commits to the estimated best arm, the number of suboptimal group pulls scales as

$$
\mathbb { E } \left[ \sum _ { a \neq \theta } n _ { a } ^ { ( r + 1 ) } \right] \gtrsim p _ { e } \cdot \frac { N L } { 2 } ,
$$

the factor $\frac { 1 } { 2 }$ standing in for the constant fraction of committed rounds spent on a wrong arm under $\{ \hat { \theta } \neq \theta \}$ . The binding lower bound below is derived without this step.

Substituting the Fano bound: if $8 \Delta ^ { 2 } N L ~ \leq ~ ( \log K ) / 4$ , i.e. $L ~ \le ~ \log K / ( 3 2 N \Delta ^ { 2 } )$ , then $I ( \theta ; \mathbf { X } _ { r + 1 } ) \le ( \log K ) / 4$ and

$$
p _ { e } \geq 1 - { \frac { ( \log K ) / 4 + \log 2 } { \log K } } \geq { \frac { 1 } { 4 } } \qquad { \mathrm { ( f o r ~ } } K \geq 4 { \mathrm { ) } } .
$$

In this short-regime case, $\mathbb { E } [ R _ { r + 1 } ] \ge \Delta \cdot ( 1 / 4 ) \cdot N L / 2 = \Delta N L / 8$ , and summing over $S _ { \mathrm { d e c } }$ switches gives $\mathbb { E } [ R _ { T } ] \geq S _ { \mathrm { d e c } } \Delta N L / 8 .$ , which is $\Omega ( S _ { \mathrm { d e c } } \Delta T / S _ { \mathrm { d e c } } ) = \Omega ( \Delta T )$ and is not useful when $\Delta T$ is large. The interesting case is $L \ge c _ { 0 } K / ( N \Delta ^ { 2 } )$ for a suficiently large constant $c _ { 0 }$ . In this regime, we use a per-arm change-of-measure argument instead of Fano directly. We first give a Fano-style warm-up, superseded by the single-arm argument below and not the family alternative of the theorem, so here $Q _ { a }$ denotes only a local two-arm comparison. For each suboptimal arm $a \neq a _ { r + 1 } ^ { \star } .$ , define $Q _ { a }$ to agree with the original instance $P$ on all arms except a and $a _ { r + 1 } ^ { \star }$ in regime $r + 1 { : }$ : under $Q _ { a }$ , arm a has mean $1 / 2 + \Delta$ and arm $a _ { r + 1 } ^ { \star }$ has mean $1 / 2$ in regime $r + 1$ . In this alternative, arm a is the best arm.

Under $P \colon$ the regret charged to arm a is $R _ { a } ^ { P } = \Delta \cdot n _ { a } ^ { ( r + 1 ) }$

Under $Q _ { a } \mathrm { { : } }$ the regret charged to arm $a _ { r + 1 } ^ { \star }$ (now suboptimal) is $R _ { a ^ { \star } } ^ { Q _ { a } } = \Delta \cdot n _ { a _ { r + 1 } ^ { \star } } ^ { ( r + 1 ) }$

$\mathrm { B y }$ the Bretagnolle–Huber inequality [Bretagnolle and Huber, 1978] applied to the event $\{ n _ { a } ^ { ( r + 1 ) } \ge N L / ( 2 K ) \}$ :

$$
P \bigg ( n _ { a } ^ { ( r + 1 ) } < \frac { N L } { 2 K } \bigg ) + Q _ { a } \bigg ( n _ { a } ^ { ( r + 1 ) } \geq \frac { N L } { 2 K } \bigg ) \geq \frac { 1 } { 2 } \exp ( - \mathrm { K L } ( P _ { r + 1 } \| Q _ { a , r + 1 } ) ) ,
$$

where

$$
\mathrm { K L } ( P _ { r + 1 } | | Q _ { a , r + 1 } ) = \mathbb { E } _ { P } [ n _ { a } ^ { ( r + 1 ) } ] \cdot \mathrm { k l } ( 1 / 2 , 1 / 2 + \Delta ) + \mathbb { E } _ { P } [ n _ { a _ { r + 1 } ^ { * } } ^ { ( r + 1 ) } ] \cdot \mathrm { k l } ( 1 / 2 + \Delta , 1 / 2 ) \le 8 \Delta ^ { 2 } \cdot N L .
$$

Under $P \colon \mathbb { E } _ { P } [ R _ { a } ^ { P } ] = \Delta \cdot \mathbb { E } _ { P } [ n _ { a } ^ { ( r + 1 ) } ]$

Under $\textstyle Q _ { a } .$ on the event $\{ n _ { a } ^ { ( r + 1 ) } \ge N L / ( 2 K ) \}$ , arm a is pulled at least $N L / ( 2 K )$ times while arm $a _ { r + 1 } ^ { \star }$ is pulled at most $N L - N L / ( 2 K ) = N L ( 2 K - 1 ) / ( 2 K )$ times. But under $Q _ { a }$ , arm a is optimal, so regret comes from pulling arms other than a. We have $\mathbb { E } _ { Q _ { a } } [ R _ { r + 1 } ^ { Q _ { a } } ] \ge \Delta \cdot \mathbb { E } _ { Q _ { a } } [ N L - n _ { a } ^ { ( r + 1 ) } ]$

Combining: for each $a \neq a _ { r + 1 } ^ { \star }$

$$
\mathbb { E } _ { P } [ n _ { a } ^ { ( r + 1 ) } ] + \mathbb { E } _ { Q _ { a } } [ N L - n _ { a } ^ { ( r + 1 ) } ] \ge \frac { N L } { 2 K } \cdot \frac { 1 } { 2 } \exp ( - 8 \Delta ^ { 2 } N L ) .
$$

Therefore

$$
\operatorname* { m a x } \biggl ( \mathbb { E } _ { P } [ R _ { a } ^ { P } ] , \mathbb { E } _ { Q _ { a } } [ R _ { r + 1 } ^ { Q _ { a } } ] \biggr ) \geq \frac { \Delta \cdot N L } { 4 K } \cdot \frac { 1 } { 2 } \exp ( - 8 \Delta ^ { 2 } N L ) .
$$

For the gap-dependent bound, we do not need this exponential term. Instead, the bound used in the sequel—and the family alternative $Q _ { a }$ of the theorem—comes from a single-arm change-of-measure argument (valid for $\Delta \in ( 0 , 1 / 8 ) )$ ; from here on $Q _ { a }$ denotes this single-arm alternative. Fix a nonbest arm a under $P$ and let $Q _ { a }$ be the single-arm alternative that raises only arm a to global mean $1 / 2 + 2 \Delta$ throughout regime $r + 1$ , leaving $a _ { r + 1 } ^ { \star }$ at $1 / 2 + \Delta$ and every other arm unchanged. Under $Q _ { a } ,$ , arm a is the unique best arm in regime $r + 1$ , by margin $\Delta$ . The base instance P realizes exactly $S _ { \mathrm { d e c } }$ switches. The alternative $Q _ { a }$ changes only regime $r + 1$ ’s best arm, so it preserves every switch except possibly at the two boundaries $\rho _ { r + 1 }$ and $\rho _ { r + 2 }$ . When the raised arm a equals an adjacent regime’s best arm, each such coincidence merges that boundary. At $K = 2$ the sole non-best arm in an interior regime $r + 1$ is the best of both neighbours, so both boundaries merge and $Q _ { a }$ carries $S _ { \mathrm { d e c } } - 2$ switches, one fewer merge at a terminal regime. In all cases $Q _ { a }$ has between $S _ { \mathrm { d e c } } - 2$ and $S _ { \mathrm { d e c } }$ switches. We therefore take the instance family to be all instances with at most $S _ { \mathrm { d e c } }$ switches and margin $\Delta _ { i }$ , so that every $Q _ { a }$ is a member and rate-consistency applies to it. The regret bound below is realized on $P$ , which has exactly $S _ { \mathrm { d e c } }$ switches, so the stated $\Omega ( S _ { \mathrm { d e c } } ( K { - } 1 ) / \Delta )$ is unafected. For $K \geq 3$ one may instead draw a distinct from both neighbouring bests, keeping all $S _ { \mathrm { d e c } }$ switches, exactly as the communication-delay construction does. The likelihood ratio between $P _ { r + 1 }$ and $Q _ { a , r + 1 }$ , restricted to the reward observations in regime $r + 1$ , factorizes over the perpull observations because rewards are conditionally independent given the action sequence (main Assumption 1). The KL divergence decomposes over the single changed arm as

$$
\begin{array} { r } { \mathrm { K L } ( P _ { r + 1 } \| Q _ { a , r + 1 } ) = \mathbb { E } _ { P } [ n _ { a } ^ { ( r + 1 ) } ] \cdot \mathrm { k l } ( \frac { 1 } { 2 } , \frac { 1 } { 2 } + 2 \Delta ) \le C _ { 1 } \Delta ^ { 2 } \mathbb { E } _ { P } [ n _ { a } ^ { ( r + 1 ) } ] , } \end{array}
$$

since $P$ and $Q _ { a }$ difer only on arm a in regime $r + 1$ and rewards at all other arms have identical Bernoulli laws; here $\mathrm { k l } ( \frac 1 2 , \frac 1 2 + 2 \Delta ) \le ( 2 \Delta ) ^ { 2 } / ( ( \frac 1 2 + 2 \Delta ) ( \frac 1 2 - 2 \Delta ) ) \le C _ { 1 } \Delta ^ { 2 }$ with $C _ { 1 } \leq 2 2$ for $\Delta \le 1 / 8$

We now invoke the Kaufmann–Cappé–Garivier divergence-decomposition lemma [Kaufmann et al., 2016, Lemma 1]: for any event E that is P-likely and $Q _ { a } .$ -unlikely,

$$
\begin{array} { r } { \mathrm { K L } ( P _ { r + 1 } | | Q _ { a , r + 1 } ) \geq \mathrm { k l } ( P ( \mathcal { E } ) , Q _ { a } ( \mathcal { E } ) ) . } \end{array}
$$

We restrict attention to the rate-consistent algorithm class: those whose expected per-regime pul count of every suboptimal arm satisfies $\mathbb { E } [ n _ { b } ^ { ( r ) } ] \le C _ { \mathrm { a l g } }$ log $T$ on every instance of the family, where $C _ { \mathrm { a l g } }$ may depend on $( K , \Delta )$ but not on T (UCB-style algorithms qualify with $C _ { \mathrm { a l g } } = \Theta ( 1 / \Delta ^ { 2 } )$ ; this is strictly weaker than asymptotic optimality and includes any centralized oracle not hard-coded to the instance). Take $\mathcal { E } = \{ n _ { a } ^ { ( r + 1 ) } \geq N L / 2 \}$ . Both tail estimates below follow from rate-consistency alone via Markov’s inequality; we never invoke the upper bound of the main DRFC theorem, which would make the argument circular. Crucially, rate-consistency pins the two tails away from 0 and 1 by an absolute constant margin—it does not drive them to $1 / T \mathrm { - s o }$ the resulting bound is genuinely finite-time and non-asymptotic.

Tail under P. Arm a is suboptimal under $P ,$ so by Markov,

$$
p : = P ( \mathcal { E } ) = P \big ( n _ { a } ^ { ( r + 1 ) } \geq N L / 2 \big ) \leq \frac { \mathbb { E } _ { P } [ n _ { a } ^ { ( r + 1 ) } ] } { N L / 2 } \leq \frac { 2 C _ { \mathrm { a l g } } \log T } { N L } \leq \frac { 1 } { 4 } ,
$$

the last step holding once $L \ge 8 C _ { \mathrm { a l g } }$ log $T / N$ , which is implied by the regime-length condition $L \ge c _ { 0 } ( C _ { \mathrm { a l g } } ) K$ log $T / ( N \Delta ^ { 2 } )$ for $\Delta < 1 / 8$ . The two factors are kept separate on purpose. The $1 / \Delta ^ { 2 }$ is the room a regime leaves for the $\Omega ( 1 / \Delta ^ { 2 } )$ certification pulls of each arm, while $C _ { \mathrm { a l g } }$ is the rateconsistency constant, itself $\Theta ( 1 / \Delta ^ { 2 } )$ for a UCB-type algorithm, so the condition reads $\widetilde \Omega ( K / ( N \Delta ^ { 4 } ) )$ for such algorithms. Keeping $C _ { \mathrm { a l g } }$ explicit in $c _ { 0 } ( C _ { \mathrm { a l g } } )$ makes this ∆-dependence transparent rather than hidden.

Tail under $Q _ { a }$ . Under $Q _ { a }$ arm a is the unique optimal arm, so every other arm is suboptimal; on the event $\mathcal { E } ^ { c } = \{ n _ { a } ^ { ( r + 1 ) } < N L / 2 \}$ the remaining $\sum _ { b \ne a } n _ { b } ^ { ( r + 1 ) } > N L / 2$ pulls land on suboptimal arms. By Markov and rate-consistency under $Q _ { a }$

$$
\begin{array} { r } { q : = Q _ { a } ( \mathcal { E } ^ { c } ) \leq Q _ { a } \Big ( \sum _ { b \neq a } n _ { b } ^ { ( r + 1 ) } \geq N L / 2 \Big ) \leq \frac { \sum _ { b \neq a } \mathbb { E } _ { Q a } [ n _ { b } ^ { ( r + 1 ) } ] } { N L / 2 } \leq \frac { 2 ( K - 1 ) C _ { \mathrm { a l g } } \log T } { N L } \leq \frac { 1 } { 4 } , } \end{array}
$$

again under the regime-length condition (the factor $K$ in $c _ { 0 }$ absorbs the K −1 here). Hence $Q _ { a } ( \mathcal { E } ) =$ $1 - q \ge 3 / 4$

Conclusion. By the monotonicity of $\operatorname { k l } ( \cdot , \cdot )$ in its arguments and $p \leq 1 / 4 \leq 3 / 4 \leq Q _ { a } ( \mathcal { E } )$ , the Kaufmann–Cappé–Garivier lemma yields

$$
\begin{array} { r } { C _ { 1 } \Delta ^ { 2 } \mathbb { E } _ { P } [ n _ { a } ^ { ( r + 1 ) } ] \ge \mathrm { K L } ( P _ { r + 1 } \| Q _ { a , r + 1 } ) \ge \mathrm { k l } \big ( P ( \mathcal { E } ) , Q _ { a } ( \mathcal { E } ) \big ) \ge \mathrm { k l } \Big ( \frac { 1 } { 4 } , \frac { 3 } { 4 } \Big ) = \frac { 1 } { 2 } \log 3 = : \kappa > 0 . } \end{array}
$$

Hence the finite-time, log-free per-arm bound

$$
\mathbb { E } _ { P } [ n _ { a } ^ { ( r + 1 ) } ] \ge \frac { \kappa } { C _ { 1 } \Delta ^ { 2 } } = \Omega \bigg ( \frac { 1 } { \Delta ^ { 2 } } \bigg ) ,
$$

valid for every regime long enough to permit these pulls, i.e. $L \ge c _ { 0 } K$ log $T / ( N \Delta ^ { 2 } )$ . The log $T$ enters only through this regime-length feasibility condition—guaranteeing the certification has room to occur—never through the tail probabilities, so no circular appeal to a log T regret rate is made.

Each group pull of suboptimal arm a costs group regret $\Delta$ (homogeneous instance, margin $\Delta )$ so

$$
\mathbb { E } _ { P } [ \mathrm { g r o u p ~ r e g r e t ~ f r o m ~ a r m ~ } a ] = \Delta \mathbb { E } _ { P } [ n _ { a } ^ { ( r + 1 ) } ] \geq \frac { \kappa } { C _ { 1 } \Delta } .
$$

Summing over the $K - 1$ suboptimal arms and the $S _ { \mathrm { d e c } }$ post-switch regimes,

$$
\mathbb { E } [ R _ { T } ] \ge c ^ { \prime } S _ { \mathrm { d e c } } \frac { K - 1 } { \Delta } , \qquad c ^ { \prime } = \frac { \kappa } { C _ { 1 } } ,
$$

the log-free certification term stated in the certification part of the main information-theoretic lower bound.

## Part 2: Communication Delay

Scope of this term. The communication-delay term is established for the decentralized graphcommunication class: protocols in which agent v’s action at round t is measurable with respect to the information that has reached v along communication paths of length $\leq t$ in the random graph process. A centralized oracle is not a member of this class and is exempt from this term; the certification term of Part 1 holds for it without exemption. This split is intrinsic—no quantity proportional to the graph diameter $D _ { G }$ can lower-bound an algorithm that is free of the graph constraint.

Why a homogeneous gap is required. Under the per-agent accounting $\begin{array} { r } { R _ { T } = \sum _ { t } \sum _ { i } ( \mu _ { i , a _ { t } ^ { \star } , t } - } \end{array}$ $\mu _ { i , A _ { i , t } , t } )$ (main, regret definition), the benchmark arm $a _ { t } ^ { \star }$ is the global-consensus optimum but each agent pays in its own local means. A construction in which the lagging agents are flat $\begin{array} { r } { ( \mu _ { i , a } \equiv \frac { 1 } { 2 } ) } \end{array}$ charges them nothing: their per-round term is 0 for every action, so no delay regret accrues at exactly the agents that wait for flooding. We therefore use a homogeneous-gap instance and localize information by a genie plus a sub-threshold $\mathrm { g a p }$

Construction. Fix $K \ge 3 , \ N \ge 4$ , and assume the feasibility condition $D _ { G } ~ \le ~ N / 4$ (when $D _ { G } > N / 4$ the delay term is dominated by the certification term of Part 1 and need not be proved separately). Take a broom graph G of diameter exactly $D _ { G } \colon$ a handle path $1 = v _ { 0 } , v _ { 1 } , \dotsc , v _ { D _ { G } - 1 }$ of $D _ { G } - 1$ edges, with a clique $\mathcal { V } _ { \mathrm { f a r } }$ on the remaining $N - D _ { G }$ agents all attached to the endpoint $v _ { D _ { G } - 1 }$ . Then $\operatorname { l i s t } _ { G } ( 1 , v ) = D _ { G }$ exactly for every $v \in \mathcal { V } _ { \mathrm { f a r } }$ (the $D _ { G } - 1$ handle edges plus one edge into the clique), the graph diameter is $D _ { G }$ (realized between agent 1 and $\mathcal { V } _ { \mathrm { f a r } } ;$ far–far distance is 1), and $| \mathcal { V } _ { \mathrm { f a r } } | = N - D _ { G } \ge N - N / 4 \ge N / 2$ since $D _ { G } \leq N / 4$ . The construction respects the flooding model of Assumption 3 with $D _ { G } ( \delta ) = D _ { G }$ on this fixed graph. At each switch r nature draws the new best arm $a _ { r } ^ { \star }$ uniformly from two arms $\{ b _ { r } , c _ { r } \}$ , both distinct from the previous best $a _ { r - 1 } ^ { \star }$ (possible since $K \geq 3 )$ , so every boundary is a genuine decision switch and the instance realizes exactly $S _ { \mathrm { d e c } }$ switches; it sets, for every agent $i \in [ N ]$ ，

$$
\begin{array} { r } { \mu _ { i , a , t } = \frac { 1 } { 2 } + \Delta ( a = a _ { r } ^ { \star } ) , \qquad \mu _ { i , a , t } = \frac { 1 } { 2 } - \Delta ( a \neq a _ { r } ^ { \star } ) . } \end{array}
$$

All means lie in [0, 1] for $\Delta \le { \frac { 1 } { 2 } }$ . Every agent’s local best equals the global best $a _ { r } ^ { \star }$ with local gap $2 \Delta$ , and the global margin is $\bar { \mu } _ { a _ { r } ^ { \star } , t } - \bar { \mu } _ { a , t } = 2 \Delta _ { \mathrm { ~ \scriptsize ~ \cdot ~ } }$ hence a lagging agent that plays the stale arm incurs per-round regret $2 \Delta$ regardless of i. Genie. At each switch $t _ { r _ { \mathrm { i } } }$ , nature reveals $a _ { r } ^ { \star }$ to agent 1 only. Side information can only help an algorithm, so a lower bound surviving the genie is stronger; it also strips the per-source statistical-identification cost and isolates the pure propagation delay. Gap calibration. Require

$$
\Delta \le \Delta _ { \mathrm { m a x } } : = \frac { 1 } { 8 \sqrt { N D _ { G } } } .
$$

Speed-of-information lemma. For any decentralized protocol, the history $H _ { v , \tau }$ of node v at any round $\tau < t _ { r } + \mathrm { d i s t } _ { G } ( 1 , v )$ is conditionally independent of the genie variable $a _ { r } ^ { \star }$ given the rewards observed within graph-radius $( \tau - t _ { r } )$ of v. Indeed information travels at most one hop per round, and agent 1’s only excess knowledge is $a _ { r } ^ { \star } ;$ ; induction on rounds gives the claim. In particular, for $v \in \mathcal { V } _ { \mathrm { f a r } }$ and $\tau < t _ { r } + D _ { G }$ , the genie value has not reached v.

Indistinguishability. Let $P \left( a _ { r } ^ { \star } = b _ { r } \right)$ and $Q \ ( a _ { r } ^ { \star } = c _ { r } )$ denote the two instances at switch $^ { r , }$ which difer only by which of the two unrevealed candidates $b _ { r } , c _ { r }$ carries mean $\begin{array} { l } { \frac { 1 } { 2 } + \Delta } \end{array}$ (every other arm, including the previous best $a _ { r - 1 } ^ { \star }$ , sits at $\begin{array} { l } { { \frac { 1 } { 2 } - \Delta } } \end{array}$ in both). The per-pull divergence is

$$
\begin{array} { r } { \mathrm { k l } \bigl ( \frac 1 2 + \Delta , \frac 1 2 - \Delta \bigr ) \le \frac { ( 2 \Delta ) ^ { 2 } } { \bigl ( \frac 1 2 - \Delta \bigr ) \bigl ( \frac 1 2 + \Delta \bigr ) } \le 3 2 \Delta ^ { 2 } \qquad ( \Delta \le \frac 1 4 ) . } \end{array}
$$

For $v \in \mathcal { V } _ { \mathrm { f a r } }$ and $\tau < t _ { r } + D _ { G }$ , the observations available to determine $A _ { v , \tau }$ number at most N agents $\times D _ { G }$ rounds $= N D _ { G }$ pulls (the reachable neighborhood within the window), so

$$
\begin{array} { r } { { \mathrm { K L } } \bigl ( P _ { H _ { v , \tau } } \parallel Q _ { H _ { v , \tau } } \bigr ) \leq 3 2 N D _ { G } \Delta ^ { 2 } \leq 3 2 N D _ { G } \Delta _ { \operatorname* { m a x } } ^ { 2 } = \frac { 1 } { 2 } . } \end{array}
$$

By Bretagnolle–Huber and the arm-relabeling symmetry of $P , Q$ , averaging over the uniform genie,

$$
\begin{array} { r } { \mathbb { E } \big [ \mathbf { 1 } \{ A _ { v , \tau } \neq a _ { r } ^ { \star } \} \big ] \geq \frac { 1 } { 2 } e ^ { - 1 / 2 } \cdot \frac { 1 } { 2 } \geq 0 . 1 5 = : p _ { 0 } . } \end{array}
$$

Aggregation. For each switch r and each $v \in \mathcal { V } _ { \mathrm { f a r } }$ , the genie has not reached v for all $\tau \ \in$ $[ t _ { r } , t _ { r } + D _ { G } )$ (since di $\mathfrak { s t } _ { G } ( 1 , v ) \geq D _ { G } )$ , so the expected per-agent regret over this window is at least $2 \Delta \cdot p _ { 0 } \cdot D _ { G } = 0 . 3 \Delta D _ { G }$ . Summing over th $\gtrsim N / 2$ far agents and the exactly $S _ { \mathrm { d e c } }$ genuine switches,

$$
\begin{array} { r } { \mathbb { E } [ R _ { T } ] \ge 0 . 1 5 N S _ { \mathrm { d e c } } D _ { G } \Delta = \Omega ( N S _ { \mathrm { d e c } } D _ { G } \Delta ) , \qquad \Delta \in \left( 0 , \frac { 1 } { 8 \sqrt { N D _ { G } } } \right] . } \end{array}
$$

This expectation is taken over the uniform genie draw of each new best arm, so by the probabilistic method there is a fixed realized sequence of new arms, a single deterministic instance of the family with exactly $S _ { \mathrm { d e c } }$ switches, on which $\mathbb { E } [ R _ { T } ] \ge \Omega ( N S _ { \mathrm { d e c } } D _ { G } \Delta )$ with the expectation now only over the algorithm and graph randomness. This de-randomization is the worst-case form claimed in the communication-delay part of the main information-theoretic lower bound. The range restriction is intrinsic: for a constant gap, agents identify the new best arm from their own observations and need not await the flood, so the diameter-delay regret is a genuinely small-gap phenomenon.

Combined bound (two-instance minimax, not additive). Both parts use homogeneous instances, but they are proved on separate families: Part 1 on a K-arm certification family (every rate-consistent algorithm, including a centralized oracle), Part 2 on a broom-graph family (the decentralized graph-communication class, $\Delta \le \Delta _ { \mathrm { m a x } } \ )$ . The two costs are therefore not additive on a single instance. Writing

$$
A : = c S _ { \mathrm { d e c } } \frac { K - 1 } { \Delta } , \qquad B : = c N S _ { \mathrm { d e c } } D _ { G } \Delta ,
$$

the correct combined statement is the two-instance minimax: for every algorithm in the intersection of the two classes (rate-consistent and decentralized-graph), there exists an instance in the union of the two families on which

$$
\mathbb { E } [ R _ { T } ] \ \geq \ \operatorname* { m a x } \{ A , B \} \ \geq \ \frac { 1 } { 2 } ( A + B ) .
$$

The factor $\textstyle { \frac { 1 } { 2 } }$ is the only sense in which the sum $A + B$ is a lower bound; we do not claim an additive bound on a single instance (a centralized oracle, in particular, pays A but is exempt from B). This matches the statement of the main information-theoretic lower bound and concludes its proof.

## H. DRFC-Seq: Anytime-Valid Guarantee under Within-Regime Drift

We prove anytime-valid correctness (a) and the adaptivity bound (b) of the main DRFC-Seq theorem. Throughout, $a _ { r } ^ { \mathrm { a v g } }$ denotes the regime-average best arm, abbreviated as $a ^ { \mathrm { a v g } }$ within a fixed regime, and $\bar { S } _ { \mathrm { d e c } } ^ { \mathrm { a v g } }$ counts changes in this comparator. The variable n counts the per-arm, per-agent samples in the current sliding window, and for an ordered pair $( b , a )$ we write ${ \hat { G } } _ { b , a } ( n )$ for the windowed equal-agent balanced estimator and $\bar { G } _ { b , a } ( n )$ for its time-average target. The main paper’s window-average margin condition states that, for some window length W and margin $\bar { \Delta } > 0$ , every length-W window inside a regime has $\bar { G } _ { a _ { r } ^ { \mathrm { a v g } } , a } ( \mathcal { W } ) \geq \bar { \Delta }$ for each $a \neq a _ { r } ^ { \mathrm { a v g } }$ , while the instantaneous gap may be negative on a constant fraction of the window.

Step 0: synchronized evaluation makes the windowed estimator complete. The equalagent estimator ${ \hat { G } } _ { b , a } ( n )$ averages the balanced contrasts of all N agents, so a block can be scored only after its source-traceable reward and pull records reach every agent. Each probe runs one synchronized balanced block and then floods its records (main Algorithm 2, line 4). Let $E _ { \mathrm { f l o o d } }$ be the event that every such block reaches all agents within $D _ { G } ( \alpha )$ rounds; by the main random-graph flooding assumption at level $\alpha ,$

$$
\mathbb { P } [ E _ { \mathrm { f l o o d } } ] \ \geq \ 1 - \alpha ,
$$

and the timing convention $\delta _ { \mathrm { p } } \geq K h + D _ { G } ( \alpha )$ opens the next probe only after the current block has fully flooded. On $E _ { \mathrm { f l o o d } }$ , every block appended to Q carries the complete equal-agent average

$$
g _ { i } = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } c _ { i , j } ,
$$

identical at every agent, which is the quantity the confidence sequence of Step 1 is built on. Evaluation lags each block by at most $D _ { G } ( \alpha )$ rounds, a latency already charged in the $D _ { G } ( \alpha )$ terms of the regret bound and in the regime spacing $\ge 2 W \delta _ { \mathrm { p } } + D _ { G } ( \alpha )$ of the main DRFC-Seq theorem, so no rate changes. Running $E _ { \mathrm { f l o o d } }$ and the confidence-sequence event of Step 1 each at level $\alpha / 2$ and intersecting them gives the good event $\mathcal { E }$ with $\mathbb { P } [ \mathcal { E } ] \geq 1 - \alpha$ , the halved level entering $r _ { n }$ only through an absorbed log 2.

Step 1: a time-uniform confidence sequence via Ville’s inequality. Fix an ordered pair $( b , a )$ and a window start $s \in \{ 1 , \ldots , T \}$ . We index the confidence sequence at the per-arm, peragent sample level: let n count the balanced pulls of each arm accumulated inside the window opened at s (a full W-block window carries $n \ = \ W h$ such samples per arm per agent, since each block contributes h pulls/arm/agent). For the i-th sample $( i = 1 , \ldots , n )$ , let $c _ { i , j } ~ \in ~ [ - 1 , 1 ]$ be agent $j ^ { \prime } \mathrm { s }$ balanced $( b , a )$ -contrast at that pull, and let the equal-agent per-sample contrast be $\begin{array} { r } { g _ { i } \ = \ \frac { 1 } { N } \sum _ { j = 1 } ^ { N } c _ { i , j } } \end{array}$ . Then $\begin{array} { r } { \hat { G } _ { b , a } ^ { ( s ) } ( n ) \ = \ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } g _ { i } } \end{array}$ and $\begin{array} { r } { \bar { G } _ { b , a } ^ { ( s ) } ( n ) \ = \ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \mathbb { E } _ { 0 } [ g _ { i } ] } \end{array}$ , with $\mathbb { E } _ { 0 }$ the conditional expectation given the F -measurable mean path. Define the sample-level filtration $\mathcal { F } _ { n } ^ { ( s ) } =$ σ randomized schedules and rewards of samples $1 , \ldots , n$ inside window $s , \ F _ { 0 } )$ ; the block grouping governs only when the monitor acts, on full W-block windows, not the confidence-sequence index.

Why block-level indexing. A sample-level cumulative sum $\begin{array} { r } { \sum _ { i , j } ( c _ { i , j } - \mathbb { E } _ { 0 } [ g _ { i } ] ) } \end{array}$ is not the correct martingale. Under within-regime drift the equal-agent per-sample target $\mathbb { E } _ { 0 } [ g _ { i } ]$ depends on which time slot the schedule assigned to the i-th pull, so a sample-indexed centering drops the schedulerandomization error R of Lemma $^ { 3 , }$ which is exactly the term that does not vanish when the means vary inside the window. Indexing the confidence sequence at the block level carries that term inside the increment. Enumerate the blocks of the window opened at s as $\ell = 1 , 2 , \ldots ,$ each a full balanced probe block of h pulls per arm per agent over all arms, so that after B blocks the per-arm sample count is $n = h B$ , and let

$$
\hat { G } _ { \ell } : = \hat { G } _ { b , a } ( \{ \ell \} ) , \qquad \bar { G } _ { \ell } : = \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \frac { 1 } { q _ { \ell } } \sum _ { t \in \mathcal { U } _ { \ell } } \bigl ( \mu _ { j , b , t } - \mu _ { j , a , t } \bigr )
$$

be the single-block estimator and its time-average target (§ notation), with per-block error $Y _ { \ell } : =$ $\hat { G } _ { \ell } - \bar { G } _ { \ell }$

(i) Mean-zero increment. Each block draws, for every agent j, disjoint uniform size-h pull-slot subsets $S _ { j , a } ^ { ( \ell ) } , S _ { j , b } ^ { ( \ell ) } \subset \mathcal { U } _ { \ell }$ independently of $\mathcal { F } _ { t - 1 }$ (main Assumption 1). The uniform-subset identity

$$
\begin{array} { r } { \mathbb { E } _ { \boldsymbol { \mathcal { S } } } \Big [ \frac { 1 } { h } \sum _ { t \in S _ { j , a } ^ { ( \ell ) } } \mu _ { j , a , t } \Big ] = \frac { 1 } { q _ { \ell } } \sum _ { t \in \mathcal { U } _ { \ell } } \mu _ { j , a , t } , } \end{array}
$$

together with the conditional mean-zero of the reward noise, gives

$$
{ \mathbb E } \big [ \hat { G } _ { \ell } \big | \mathcal F _ { 0 } , \{ Y _ { \ell ^ { \prime } } \} _ { \ell ^ { \prime } < \ell } \big ] = \bar { G } _ { \ell } , \qquad \mathrm { h e n c e } \qquad { \mathbb E } \big [ Y _ { \ell } \big | \mathcal F _ { 0 } , \{ Y _ { \ell ^ { \prime } } \} _ { \ell ^ { \prime } < \ell } \big ] = 0 .
$$

Unlike the sample-level error, $Y _ { \ell }$ therefore contains the schedule-randomization term R rather than discarding it.

(ii) Block-count martingale. By the conditional mean-zero property of step (i), which holds given $\mathcal { F } _ { 0 }$ and all earlier blocks, the centered partial sum

$$
Z _ { B } ^ { ( s ) } : = \sum _ { \ell = 1 } ^ { B } Y _ { \ell } = B \big ( \hat { G } _ { b , a } ^ { ( s ) } ( n ) - \bar { G } _ { b , a } ^ { ( s ) } ( n ) \big )
$$

is a martingale in the block count B with respect to $\mathcal { G } _ { B } ^ { ( s ) } = \sigma ( \mathcal { F } _ { 0 } , Y _ { 1 } , \ldots , Y _ { B } )$ . The windowed sample mean error is not a martingale, since $\begin{array} { r } { \mathbb { E } [ Z _ { B } ^ { ( s ) } / B \mid \mathcal { G } _ { B - 1 } ^ { ( s ) } ] = \frac { B - 1 } { B } \left( Z _ { B - 1 } ^ { ( s ) } / B \right) } \end{array}$ ; the machinery is applied to $Z _ { B } ^ { ( s ) }$ and converted back by dividing by B. The start s is fixed because a sliding window that drops its oldest block is not a martingale in the running count, and the moving window is handled by the union over starts below.

(iii) Bounded, conditionally sub-Gaussian increments. Each $| Y _ { \ell } | \le 2$ , and splitting $Y _ { \ell }$ into reward noise (Hoefding–Azuma) and schedule-selection noise (sampling without replacement, Bardenet– Maillard), exactly the two-term decomposition of Lemma 3, makes $Y _ { \ell }$ conditionally sub-Gaussian with variance proxy

$$
\sigma _ { \ell } ^ { 2 } \le c _ { 1 } / ( N h )
$$

for a universal $c _ { 1 }$ , the equal-agent average over N agents and h balanced pulls. The reward-noise half reaches its $1 / ( N h )$ proxy through the same lexicographic martingale ordering as in Lemma 3, which relies on the cross-agent independence clause of main Assumption 1.

(iv) Supermartingale and Ville. The process

$$
\begin{array} { r } { M _ { B } ^ { ( s ) } ( \lambda ) = \exp \Bigl ( \lambda Z _ { B } ^ { ( s ) } - \frac { c _ { 1 } } { 2 } \lambda ^ { 2 } \frac { B } { N h } \Bigr ) } \end{array}
$$

is a nonnegative supermartingale with $M _ { 0 } ^ { ( s ) } = 1$ for every $\lambda \in \mathbb { R }$ . Ville’s inequality, geometric peeling over λ, and the stitched time-uniform boundary of Howard–Ramdas–Sékely–McAulife Howard et al.

[2021] (Theorem 1, explicit constants $a _ { 0 } , C _ { 0 }$ with $a _ { 0 } = \Theta ( c _ { 1 } ) )$ give a time-uniform envelope $| Z _ { B } ^ { ( s ) } | \le$ $\sqrt { \left( c _ { 1 } B / ( N h ) \right) \left( 2 \log \log ( e B ) + 2 \log ( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha ) \right) }$ valid simultaneously over all $B \geq 1$ , where the window starting at step s spends level $\alpha _ { s } = 6 \alpha / ( \pi ^ { 2 } s ^ { 2 } )$ over the pair union. Dividing by B and writing $n = h B$ for the per-arm sample count (so the block size $h$ enters $r _ { n }$ through $n = h B$ , and log $\log ( e B ) \leq \log \log ( e n ) )$ yields, for the fixed start s,

$$
\begin{array} { r l r } {  { \mathbb { P } \big [ \exists n \geq 1 : \ \big | \hat { G } _ { b , a } ^ { ( s ) } ( n ) - \bar { G } _ { b , a } ^ { ( s ) } ( n ) \big | > r _ { n } \bigg ] \leq \frac { 6 \alpha } { \pi ^ { 2 } K ( K - 1 ) s ^ { 2 } } , } } \\ & { } & { r _ { n } = \sqrt { \frac { a _ { 0 } \big ( \log \log ( e n ) + \log ( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha ) \big ) } { N n } } . } \end{array}
$$

Two points make this radius honest where the naive $\sqrt { 2 ( \log ( 1 / \alpha ) + 2 \log \log n ) / ( N n ) }$ fails. (i) Defined for all $n \geq 1 { : } \log ( e n ) \geq 1$ so log $\log ( e n ) \geq 0$ , and $\log ( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha ) > 0$ for every $s \geq 1$ The bare log log n is negative or undefined at $n \leq 2$ , exactly when a window is filling. (ii) Sliding, all comparisons, and horizon-free: the monitor may compare any ordered pair $( b , a )$ as the candidate aˆ changes, so we union over all $K ( K - 1 )$ ordered arm pairs and over window starts. Rather than split $\alpha$ uniformly over $\leq T$ starts, which would need the horizon $T$ in advance, we spend a summable schedule $\alpha _ { s } = \alpha 6 / ( \pi ^ { 2 } s ^ { 2 } )$ on the window starting at step s. Because $\begin{array} { r } { \sum _ { s > 1 } 6 / ( \pi ^ { 2 } s ^ { 2 } ) = 1 } \end{array}$ the guarantee holds simultaneously over all starts $s \geq 1$ with no predeclared horizon, the displayed radius at start s carrying log $( C _ { 0 } K ( K { - } 1 ) s ^ { 2 } / \alpha )$ . For downstream regret bounds we use $s \leq T$ , so this is at most $\log ( C _ { 0 } K ( K { - } 1 ) T ^ { 2 } / \alpha )$ and absorbed into $\widetilde { O } .$ , while the construction itself is genuinely anytime-valid and uniform in $T _ { i }$ not merely valid up to a predeclared horizon. The estimator’s target is the within-window time-average gap $\bar { G } ^ { ( s ) }$ throughout—no instantaneous quantity ever enters. (An equivalent route uses a discounted confidence sequence Waudby-Smith and Ramdas [2024] with forgetting factor $\eta ;$ we take the union-over-starts form for transparent constants.) Call the simultaneous good event $\mathcal { E }$ (over all ordered pairs and all window starts), with $\mathbb { P } [ \mathcal { E } ] \geq 1 - \alpha$ . To lighten notation we drop the superscript s below, with n the count in the currently active window.

Step 2: anytime correctness (a), stated about the true best arm. We prove correctness without assuming the algorithm’s candidate is already optimal—that assumption is exactly the proof–algorithm misalignment the monitor must avoid. Define a false switch as displacing the timeaverage-best arm $a ^ { \mathrm { a v g } }$ during a regime with no average-decision switch $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0 )$ ; adopting $a ^ { \mathrm { a v g } }$ from a suboptimal candidate is a correct adoption, not a false switch.

Work on E. The $| \mathcal { Q } | = W$ gate of main Algorithm 2 enforces only that the monitor acts on a full window of W blocks; the algorithm cannot see regime boundaries. For claim $( a )$ this is enough: in a regime with $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0$ there is no average-decision boundary inside the analyzed segment, so every full window automatically lies inside the regime, $\mathcal { W } \subseteq [ \rho _ { r } , \rho _ { r + 1 } )$ , and the main windowaverage margin assumption applies to it directly. (Boundary-straddling full windows arise only at a true average-decision switch, handled in Step 3.) Because each probe runs a balanced block of h pulls/arm/agent over all arms, the per-arm windowed sample counts are exactly equal, $n _ { a } = n = W h$ for all $a .$ . For the time-average-best arm $a ^ { \mathrm { a v g } }$ and any challenger $b \neq a ^ { \mathrm { a v g } }$ , the main window-average margin assumption gives $\bar { G } _ { b , a ^ { \mathrm { a v g } } } ( \mathcal { W } ) \leq - \bar { \Delta }$ , so on $\mathcal { E }$

$$
\begin{array} { r } { \hat { \mu } _ { b } - \hat { \mu } _ { a ^ { \mathrm { a v g } } } = \hat { G } _ { b , a ^ { \mathrm { a v g } } } ( n ) \leq - \bar { \Delta } + 2 r _ { n } , \qquad \mathrm { L C B } _ { b } = ( \hat { \mu } _ { b } - \hat { \mu } _ { a ^ { \mathrm { a v g } } } ) - 2 r _ { n } \leq - \bar { \Delta } < 0 . } \end{array}
$$

This holds simultaneously for every full window (the CS event $\mathcal { E }$ is uniform over all $\leq T$ window starts). No challenger ever attains a positive lower-confidence gap against $a ^ { \mathrm { a v g } }$ , independently of which arm the algorithm currently holds. In particular, once $a ^ { \mathrm { a v g } }$ is the candidate it is never displaced in a $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0$ regime, so DRFC-Seq performs no drift-induced false switch, however often or far the instantaneous gap dips negative—only the time-average target enters. This proves (a).

We must, however, match the implemented switch rule (main Algorithm 2): each probe tests only the single empirical-argmax challenger $b = \arg \operatorname* { m a x } _ { a \neq \hat { a } } \hat { \mu } _ { a } .$ , and an accepted switch relabels aˆ without flushing Q (the sliding window in line 5 is the only forgetting mechanism). Two facts make this rule recover $a ^ { \mathrm { a v g } }$

Fact 1 (the argmax is $a ^ { \mathrm { a v g } }$ on a full in-regime window). Because every probe runs a balanced block of h pulls/arm/agent over all arms (main Algorithm 2, line 4), the per-arm windowed counts are exactly equal, $n _ { a } = n = W h$ for all a (so $r _ { n _ { a } } = r _ { n }$ is common). On a full window $\mathcal { W } \subseteq [ \rho _ { r } , \rho _ { r + 1 } )$ the main window-average margin assumption gives $\bar { G } _ { a ^ { \mathrm { a v g } } , a } ( \mathcal { W } ) \geq \bar { \Delta }$ for every $a \ne a ^ { \mathrm { a v g } }$ (not only against the current candidate). On $\mathcal { E } , | \hat { G } _ { a ^ { \mathrm { a v g } } , a } - \bar { G } _ { a ^ { \mathrm { a v g } } , a } | \leq 2 r _ { n } ,$ so once $2 r _ { n } < \bar { \Delta }$ we have $\hat { \mu } _ { a } \mathrm { a v g } - \hat { \mu } _ { a } =$ $\hat { G } _ { a ^ { \mathrm { a v g } } , a } \geq \bar { \Delta } - 2 r _ { n } > 0$ simultaneously for all $a \neq a ^ { \mathrm { a v g } }$ . Hence $a ^ { \mathrm { a v g } }$ is the unique empirical maximizer and equals the tested challenger b whenever ${ \hat { a } } \neq a ^ { \mathrm { a v g } }$

Fact 2 (the full-window gate bounds the transient). The $| \mathcal { Q } | = W$ gate forbids any switch until the window is full; since an accepted switch does not flush Q, once the window is full it stays full, with per-arm count fixed at $n = W h$ . Starting from the regime boundary $\rho _ { r }$ , the sliding window replaces one old block per probe, so after at most W probe-blocks the window holds only regime-r blocks and is a full in-regime window.

Lemma 9 (Initialization-free recovery). Let $n _ { 0 } ( \bar { \Delta } ) : = \operatorname* { i n f } \{ n \geq 1 : r _ { n } \leq \bar { \Delta } / 4 \} = \widetilde O \big ( 1 / ( N \bar { \Delta } ^ { 2 } ) \big )$ and assume $W h \ge n _ { 0 } ( \bar { \Delta } )$ and each regime spans at least 2W probe-blocks. On E, from an arbitrary candidate aˆ (including a mis-initialized one), DRFC-Seq adopts $a ^ { \mathrm { a v g } }$ within W probe-blocks of the regime start, and once adopted never leaves it for the remainder of the regime. (During $t h e \le W$ pre-adoption probe-blocks the candidate may change; this is charged to the Step-3 transient, not to correctness.)

Proof. By Fact 2, within W probe-blocks of the regime start the window is a full in-regime window with $n = W h \ge n _ { 0 } ( \bar { \Delta } )$ , so $2 r _ { n } \le \bar { \Delta } / 2 < \bar { \Delta }$ . (During those $\leq W$ probe-blocks the gate may permit transient switches among arms; this afects only the Step-3 transient regret, not correctness, since no displacement of $a ^ { \mathrm { a v g } }$ can occur once $a ^ { \mathrm { a v g } }$ is held, by (a).) On this first full in-regime window, by Fact 1 the tested challenger is $b = a ^ { \mathrm { a v g } }$ (unless ${ \hat { a } } = a ^ { \mathrm { a v g } }$ already, in which case apply (a)), and the main window-average margin assumption against the current candidate gives, with the direct balanced pairwise contrast $\hat { G } _ { a ^ { \mathrm { a v g } } , \hat { a } }$ of deviation $r _ { n }$

$$
\begin{array} { r } { \mathrm { L C B } _ { a ^ { \mathrm { a v g } } } = \hat { G } _ { a ^ { \mathrm { a v g } } , \hat { a } } ( n ) - 2 r _ { n } \geq ( \bar { \Delta } - r _ { n } ) - 2 r _ { n } = \bar { \Delta } - 3 r _ { n } \geq \frac { \bar { \Delta } } { 4 } > 0 , } \end{array}
$$

using $r _ { n } \leq \bar { \Delta } / 4 ,$ so the trigger fires on $a ^ { \mathrm { a v g } }$ and it is adopted. After adoption, (a) shows $a ^ { \mathrm { a v g } }$ is never displaced on any subsequent full in-regime window. □

Lemma 9 closes the alignment gap against the implemented algorithm: the arbitrary initial candidate of main Algorithm 2 need not be assumed optimal, the argmax-only test provably selects $a ^ { \mathrm { a v g } }$ on the first full in-regime window (Fact 1), and the full-window gate confines all suboptimal/transient switching to the $\leq W$ post-boundary probes (Fact 2). The anytime guarantee (a) then holds for the true $a ^ { \mathrm { a v g } }$ rather than for a candidate certified only by hypothesis.

Step 3: adaptivity (b). At a true average-decision switch the post-switch state is an arbitrary stale candidate aˆ against the new best $a _ { r } ^ { \mathrm { a v g } }$ , so the recovery bound of Lemma 9 applies verbatim: on $\mathcal { E } ,$ DRFC-Seq adopts $a _ { r } ^ { \mathrm { a v g } }$ within W probe-blocks of the switch (the stitched numerator log log $( e W h ) + \log ( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha )$ absorbed into $\widetilde { O } )$ . Those $\leq W$ probe-blocks span $\le \ W \delta _ { \mathrm { p } }$ rounds, during which each of N agents plays a possibly stale arm at per-round cost $\leq 1$ , an $O ( N W \delta _ { \mathrm { p } } )$ transient per switch, plus a further $O ( N D _ { G } ( \alpha ) )$ synchronization cost. The opening regime $r = 0$ is itself entered from an arbitrary initial candidate, so Lemma 9 applied at the start $\rho _ { 0 }$ contributes one further $O ( N ( W \delta _ { \mathrm { p } } + D _ { G } ( \alpha ) ) )$ recovery transient that no switch pays for. The $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } }$ post-switch recoveries and this single initialization recovery total $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } + 1$ recovery events, which is why the transient term below carries the factor $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } + 1$ rather than $\bar { S } _ { \mathrm { d e c } } ^ { \mathrm { a v g } }$ and remains nonzero even at $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0$ Certification cost, derived directly from the probe process (not from the Algorithm 1 active-set scan). Algorithm 2 runs no active-set scan at all, initializing arbitrarily, so certification of $a _ { r } ^ { \mathrm { a v g } }$ is achieved purely by the balanced probe blocks reaching confidence. By Lemma 9, the windowed estimator separates $a _ { r } ^ { \mathrm { a v g } }$ from every challenger once $n \geq n _ { 0 } ( \bar { \Delta } _ { r } ) = \widetilde O ( 1 / ( N \bar { \Delta } _ { r } ^ { 2 } ) )$ per-arm samples, i.e. within $W _ { r } = \lceil n _ { 0 } ( \bar { \Delta } _ { r } ) / h \rceil$ probe-blocks, with each probe-block pulling all K arms at h pulls/arm/agent, so the certification exploration of the $K - 1$ non-candidate arms over those $W _ { r }$ blocks costs

$$
\widetilde { O } \big ( N ( K - 1 ) h W _ { r } \big ) = \widetilde { O } \big ( N K h \cdot n _ { 0 } ( \bar { \Delta } _ { r } ) / h \big ) = \widetilde { O } \big ( K / \bar { \Delta } _ { r } ^ { 2 } \big ) = : \widetilde { O } ( \mathcal { C } _ { r } ^ { \mathrm { s e q } } ) ,
$$

which we take as the definition of $\mathcal { C } _ { r } ^ { \mathrm { s e q } }$ in the DRFC-Seq bound (keyed to the time-average margin $\Delta _ { r } ,$ , which always exists under the main time-average margin assumption, unlike the instantaneousgap $\mathcal { C } _ { r }$ of the main DRFC theorem). It coincides with the active-set scan cost of Lemma 7 up to constants, but is here a property of the probe CS, eliminating any algorithm/proof mismatch. Finally, the monitor runs one balanced probe block over all K arms every $\delta _ { \mathrm { p } }$ rounds; the steadystate exploration cost of pulling the $K - 1$ non-candidate arms is $\widetilde { O } ( N K h T / \delta _ { \mathrm { p } } )$ , absorbed into the periodic term by choosing $\delta _ { \mathrm { p } } = \Theta ( M )$ (matching the $T / M$ monitoring cadence of the main DRFC theorem). Summing,

$$
R _ { T } ^ { \mathrm { { d e c } } } = \widetilde { O } \Big ( \frac { N K h T } { \delta _ { \mathrm { p } } } + \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } } \mathcal { C } _ { r } ^ { \mathrm { s e q } } + N ( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } + 1 ) \big ( W \delta _ { \mathrm { p } } + D _ { G } ( \alpha ) \big ) \Big ) ,
$$

with no term depending on $S _ { \mathrm { l o c } }$ or on the number of instantaneous sign changes of the gap, because the trigger reads only the drift-invariant time-average target G<sup>¯</sup>. □

## H.1 Drift Separation: Proof of the Drift Proposition (main Proposition 1)

We prove the main memory–adaptation proposition: a single homogeneous, average-decisionstationary instance $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0 )$ on which (a) the windowed time-average reader never adopts a suboptimal arm, while (b) every responsive monitor with efective memory below half the drift period adopts the wrong arm with probability bounded below by an absolute constant. The two claims exhibit the stated memory–adaptation tradeof.

The square-wave instance. Fix $K = 2$ arms, N agents, and a single regime [T] (so $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0 )$ Let P be an even drift period in rounds and define the period-P square wave

$$
\operatorname { s q } _ { P } ( t ) = { \left\{ \begin{array} { l l } { + 1 , } & { ( t { \bmod { P } } ) < P / 2 \quad ( \mathrm { ^ { \boldsymbol { \mathfrak { c } } } u p - p h a s e ^ { \boldsymbol { \mathfrak { 7 } } } } ) , } \\ { - 1 , } & { ( t { \bmod { P } } ) \geq P / 2 \quad ( \mathrm { ^ { \boldsymbol { \mathfrak { c } } } d o w n - p h a s e ^ { \boldsymbol { \mathfrak { 7 } } } } ) . } \end{array} \right. }
$$

Pick a mean drift $\bar { \Delta } \in ( 0 , 1 / 3 )$ and amplitude b with

$$
b > 2 \bar { \Delta } \qquad \mathrm { a n d } \qquad \bar { \Delta } + b \leq 1 ,
$$

e.g. $( \bar { \Delta } , b ) = ( 0 . 1 , 0 . 3 )$ . Set the time-varying Bernoulli means, identical across all N agents (homogeneous instance),

$$
\begin{array} { r } { \mu _ { 1 } ( t ) = \frac { 1 } { 2 } + \frac { 1 } { 2 } \big ( \bar { \Delta } + b \mathrm { s q } _ { P } ( t ) \big ) , \qquad \mu _ { 2 } ( t ) = \frac { 1 } { 2 } - \frac { 1 } { 2 } \big ( \bar { \Delta } + b \mathrm { s q } _ { P } ( t ) \big ) , } \end{array}
$$

so both means lie in [0, 1] (since $\bar { \Delta } + b \leq 1 )$ and the instantaneous gap is

$$
g _ { t } : = \mu _ { 1 } ( t ) - \mu _ { 2 } ( t ) = \bar { \Delta } + b \mathrm { s q } _ { P } ( t ) = \left\{ \begin{array} { l l } { \bar { \Delta } + b > 0 , } & { \mathrm { u p \mathrm { - } p h a s e } , } \\ { \bar { \Delta } - b < 0 , } & { \mathrm { d o w n \mathrm { - } p h a s e } , } \end{array} \right.
$$

where $\bar { \Delta } - b < 0$ because $b > 2 \bar { \Delta } > \bar { \Delta }$ . Thus arm 1 leads on the time average but loses the instantaneous comparison on the entire down half of every period — a constant fraction $1 / 2$ of every window.

Time-average margin holds for every window. A contiguous window of exactly P rounds over a period-P square wave contains exactly $P / 2$ up-steps and $P / 2$ down-steps, regardless of phase $o f f s e t .$ Hence for every window start $s ,$

$$
\bar { G } _ { 1 , 2 } ^ { ( s ) } ( P ) = \frac { 1 } { P } \sum _ { t = s } ^ { s + P - 1 } g _ { t } = \bar { \Delta } + \frac { b } { P } \sum _ { t = s } ^ { s + P - 1 } \mathrm { s q } _ { P } ( t ) = \bar { \Delta } > 0 .
$$

So the instance satisfies the main time-average margin condition with margin $\bar { \Delta }$ and period $P ,$ while its instantaneous gap is negative on half of every window. This is exactly the regime in which the hypotheses of the main DRFC-Seq theorem hold but those of any instantaneous-gap guarantee fail.

(a) The period-spanning windowed reader never errs. The guarantee is a property of the window length, not of the DRFC family per se: claim (a) holds for any reader whose averaging window spans at least one full drift period $P$ (in rounds), and it is $D R F C { - } S e q$ that is constructed to do so. When its W-probe-block sliding window $W \delta _ { \mathrm { p } }$ covers an integer number of periods the window average is exactly $\bar { \Delta }$ by the display above. For a general window of $W \delta _ { \mathrm { p } } = ( m + f ) P$ rounds $( m \ge 1$ integer, $f \in [ 0 , 1 ) ,$ the leftover partial period contributes a bounded windowing bias

$$
\left| \bar { G } _ { 1 , 2 } ^ { ( s ) } ( W \delta _ { \mathrm { p } } ) - \bar { \Delta } \right| \ \le \ \varepsilon _ { W } : = \frac { b P } { 2 W \delta _ { \mathrm { p } } } = \frac { b } { 2 ( m + f ) } ,
$$

since at most $P / 2$ same-sign steps fall in the partial period. The windowed margin is thus $\bar { \Delta } - \varepsilon _ { W } .$ still strictly positive once $W \delta _ { \mathrm { p } } > b P / ( 2 \bar { \Delta } )$ ; this is the window-stable margin of the main extension, and the integer-multiple case is the special point $\varepsilon _ { W } = 0$ . The experiments operate here: a window of $\approx 5 . 7$ periods gives $\varepsilon _ { W } \leq b / 1 1 . 4$ , below the headline margin (Section I). A monitor whose adopt test has efective memory below $P / 2$ belongs to the responsive class of part (b), regardless of its nominal monitoring period; the binding condition is $\ell _ { \mathrm { e f f } } < P / 2$ , not $M < P$ . DRFC-Seq reads the full-window balanced estimator $\hat { G } _ { 1 , 2 } ( n )$ over $n = W h$ per-arm, per-agent samples, whose target is $\bar { G } _ { 1 , 2 } ^ { ( s ) } ( P ) = \bar { \Delta }$ by the display above. By the time-uniform confidence sequence of Section H (Step 1), on the good event E with $\mathbb { P } [ \mathcal { E } ] \geq 1 - \alpha$ , simultaneously over all window starts,

$$
\begin{array} { r } { \big | \hat { G } _ { 1 , 2 } ( n ) - \bar { \Delta } \big | \leq r _ { n } , \qquad r _ { n } = \widetilde O \big ( 1 / \sqrt { N n } \big ) . } \end{array}
$$

Once $n \geq n _ { 0 } ( \bar { \Delta } ) = \widetilde O ( 1 / ( N \bar { \Delta } ^ { 2 } ) )$ we have $2 r _ { n } \le \bar { \Delta } / 2$ , so the challenger arm 2 has $\mathrm { L C B _ { 2 } } = \hat { G } _ { 2 , 1 } ( n ) -$ $2 r _ { n } \le - \bar { \Delta } + 2 r _ { n } \le - \bar { \Delta } / 2 < 0$ on every full window. The wrong arm never attains a positive lower-confidence $\mathrm { g a p } ;$ by Section H, Step 2, arm 1 is held and never displaced. No drift-induced false switch occurs, however deep or frequent the instantaneous dips. This is claim (a).

A waveform-class bias bound. The bound $\varepsilon _ { W } = b P / ( 2 L ( W ) )$ was read of the square wave, but it is the worst case over a natural waveform class, which is the only property the period-agnostic monitor of Section H.2 uses. Let the within-regime drift be any P-periodic excursion $g _ { t } = \bar { \Delta } { + } b s _ { P } ( t )$ whose normalized shape $s _ { P }$ has period $P _ { \mathrm { : } }$ , per-period mean zero $\begin{array} { r } { ( \sum _ { t = s } ^ { s + P - 1 } s _ { P } ( t ) = 0 } \end{array}$ for every start $s )$ , and bounded swing $| s _ { P } ( t ) | \le 1$ . The square, triangle, and sine drifts of Section I all lie in this class. For a window of $L$ rounds the full periods cancel, so $\begin{array} { r } { \bar { G } ^ { ( s ) } ( L ) - \bar { \Delta } = ( b / L ) \sum _ { t \in A } s _ { P } ( t ) } \end{array}$ for a contiguous index set $A$ inside one period. Per-period mean zero makes the positive mass equal half the total variation, $\begin{array} { r } { \sum _ { t : s _ { P } ( t ) > 0 } s _ { P } ( t ) = \frac { 1 } { 2 } \sum _ { t } | s _ { P } ( t ) | \leq P / 2 } \end{array}$ , and any partial sum is at most this positive mass. Hence

$$
| \bar { G } ^ { ( s ) } ( L ) - \bar { \Delta } | \ \leq \ \frac { b P } { 2 L } \ = : \ \varepsilon _ { W } \qquad \mathrm { f o r ~ e v e r y ~ w a v e f o r m ~ i n ~ t h e ~ c l a s s ~ a n d ~ e v e r y ~ p h a s e } ,
$$

with the square wave attaining the bound. Drift-safety of a window therefore depends only on $L ( W )$ versus $b P / ( 2 \bar { \Delta } )$ , uniformly over the class.

(b) Every short-memory responsive monitor false-adopts. We make the comparator class precise. A linear short-memory monitor maintains, for the ordered pair (arm 2 over arm 1), a statistic

$$
\widehat { C } _ { t } = \sum _ { u \geq 0 } w _ { u } \xi _ { t - u } , \qquad w _ { u } \geq 0 , \quad \sum _ { u \geq 0 } w _ { u } = 1 , \quad \sum _ { u \geq 0 } w _ { u } { \bf 1 } [ u \geq P / 2 ] \leq \epsilon _ { 0 } ,
$$

where $\xi _ { t }$ is the per-round candidate-vs-challenger reward contrast (mean −g in our instance) and the weights $\{ w _ { u } \}$ are the monitor’s memory kernel; the last condition says an at-most-ϵ fraction of the kernel mass reaches beyond half a drift period (the efective memory is $< P / 2 )$ . The monitor adopts arm 2 when $\widehat { C } _ { t } \geq \theta$ . This class contains: (i) sliding sub-window means of length $w \le P / 2$ (uniform kernel on $[ 0 , w ) , \epsilon _ { 0 } = 0 )$ ; (ii) exponentially discounted/forgetting confidence sequences with factor $\rho$ and efective horizon $1 / ( 1 - \rho ) < P / 2$ (geometric kernel, $\epsilon _ { 0 } = \rho ^ { P / 2 } )$ ; and (iii) the perincrement test inside a CUSUM/Page detector, whose alarm at the end of a down-phase requires its most recent run-length statistic—a non-negatively weighted sum of recent increments with the same efective-memory bound—to cross the alarm level θ (a CUSUM that integrated mass uniformly over a full period would be exactly the windowed reader of part (a), not a short-memory monitor). The only excluded monitor is one whose kernel spans a full drift period, which is precisely DRFC-Seq.

Responsiveness forces $\theta \leq \bar { \Delta }$ . To be adaptive in the sense of the main DRFC-Seq theorem — i.e. to adopt the new best arm after a genuine decision switch, which changes the time-average gap to magnitude $\bar { \Delta } -$ the monitor’s statistic, an average of post-switch local contrasts whose mean is the post-switch time-average gap $\bar { \Delta }$ , must be able to exceed $\theta ;$ this requires $\theta \leq \bar { \Delta }$ (a monitor with $\theta > \bar { \Delta }$ never crosses on a margin-∆<sup>¯</sup> switch and is non-adaptive). Fix any such $\theta \in [ 0 , \bar { \Delta } ]$

Down-phase false adoption. Evaluate the statistic at a time $t ^ { \star }$ that is $P / 2$ rounds into a downphase, so the entire efective-memory window of $\widehat { C } _ { t ^ { \star } }$ lies in the current down-phase except for the $\leq \epsilon _ { 0 }$ kernel mass that leaks into the preceding up-phase. During the down-phase $\mathbb { E } [ \xi _ { t } ] = - g _ { t } = b - \bar { \Delta }$ during the leaked up-phase $\mathbb { E } [ \xi _ { t } ] ~ = ~ - ( \bar { \Delta } + b ) ~ \geq ~ - 1$ . Hence, assuming a small leakage budget $\epsilon _ { 0 } \leq ( b - 2 \bar { \Delta } ) / 4$ 2

$$
\begin{array} { r } { \mathbb { E } [ \widehat { C } _ { t ^ { \star } } ] \geq ( 1 - \epsilon _ { 0 } ) ( b - \bar { \Delta } ) - \epsilon _ { 0 } ( \bar { \Delta } + b ) \geq ( b - \bar { \Delta } ) - 2 \epsilon _ { 0 } b \geq \bar { \Delta } + \frac { 1 } { 2 } ( b - 2 \bar { \Delta } ) > \bar { \Delta } \geq \theta , } \end{array}
$$

using $b > 2 \bar { \Delta }$ . The statistic is a non-negatively weighted average $( \textstyle \sum _ { u } w _ { u } = 1 )$ of the per-round, peragent bounded contrasts in $[ - 1 , 1 ] ;$ ; writing m<sup>−</sup> $\begin{array} { r } { \mathbf { \Sigma } ^ { - 1 } : = \sum _ { u } w _ { u } ^ { 2 } / ( h N ) } \end{array}$ for the kernel’s efective sample size $( m = \Theta ( \tau _ { \mathrm { e f f } } h N )$ for any kernel with efective memory $\tau _ { \mathrm { e f f } } )$ , the bounded-diference (Azuma– Hoefding) inequality for weighted sums gives

$$
\begin{array} { r } { \mathbb { P } \big [ \widehat { C } _ { t ^ { \star } } < \theta \big ] \leq \exp \big ( - \frac { 1 } { 2 } m \big ( \mathbb { E } [ \widehat { C } _ { t ^ { \star } } ] - \theta ) ^ { 2 } \big ) \leq \exp \big ( - \frac { 1 } { 8 } m \big ( b - 2 \bar { \Delta } \big ) ^ { 2 } \big ) , } \end{array}
$$

using $\begin{array} { r } { \mathbb { E } [ \widehat { C } _ { t ^ { \star } } ] - \theta \geq \frac { 1 } { 2 } ( b - 2 \bar { \Delta } ) } \end{array}$ . Hence the monitor fires on arm $2$ (a false adoption, since arm 1 is the time-average best and $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0 )$ with probability

$$
\mathbb { P } [ \mathrm { f a l s e ~ a d o p t i o n ~ i n ~ t h i s ~ d o w n - p h a s e } ] \ge p _ { 0 } : = 1 - \exp \bigl ( - \frac { 1 } { 8 } ( b - 2 \bar { \Delta } ) ^ { 2 } \bigr ) > 0 ,
$$

a constant independent of $T$ and of the threshold $\theta \in [ 0 , \bar { \Delta } ]$ (and → 1 as the per-phase sample count m grows). Each regime contains $\Theta ( T / P )$ down-phases, so the expected number of false adoptions is $\Omega ( p _ { 0 } T / P )$ , each charging $\Omega ( \bar { \Delta } )$ instantaneous regret on switch-back; the monitor’s drift-induced regret is therefore $\Omega ( p _ { 0 } \bar { \Delta } T / P )$ , linear in the number of drift periods.

(b<sup>′</sup>) The general impossibility: any responsive monitor, linear or not. The linear-kernel bound above is convenient, but the separation needs no linearity: it follows from a single black-box property—bounded detection delay—by an exact change of measure. Call a (possibly randomized, possibly nonlinear) monitor M $( \bar { \Delta } , d )$ -responsive if, on every instance in which arm 2 becomes the time-average-best arm by margin $\ge \bar { \Delta }$ starting at some round $\tau , \mathcal { M }$ adopts arm 2 by round $\tau + d$ with probability at least a fixed constant q<sub>0</sub> $\textstyle ( \mathrm { e . g . } q _ { 0 } = { \frac { 1 } { 2 } } )$ . Bounded detection delay d is exactly what switch-adaptivity in the sense of the main $\mathrm { D R F C - S e q }$ theorem demands: a monitor whose delay is unbounded never adopts a genuine ∆<sup>¯</sup> -margin new best arm in time and is non-adaptive.

We make the responsiveness hypothesis precise, including its dependence on prehistory, so that the lemma is not vacuously strong. Call M $( \bar { \Delta } , d )$ -responsive (uniformly over prehistory) if, for every instance—including one whose pre-change rounds oscillate—on which arm 2 becomes timeaverage-best by margin $\ge \bar { \Delta }$ at some round $\tau , \mathcal { M }$ adopts arm 2 by $\tau + d$ with probability $\geq q _ { 0 }$ The qualifier “uniformly over prehistory” is what the lemma uses and is exactly the property a short-efective-memory monitor has automatically: if the adopt-decision at $\tau + d$ depends only on the last $< P / 2$ rounds of observations, the monitor cannot condition on the oscillatory prefix, so its responsiveness is necessarily uniform. A monitor that escapes the conclusion must therefore make its adopt-decision depend on a window of length $\geq P / 2 \mathrm { - } \mathrm { i . e }$ . have efective memory $\geq P / 2$ —which is precisely the non-adaptivity regime quantified in the dichotomy below. The hypothesis is thus a genuine property of bounded-memory detectors, not a smuggled assumption.

We make the resource on which the dichotomy turns precise, so that “efective memory” is an operational quantity rather than an intuition.

Definition 3 (Efective memory of a switch monitor). Model a monitor M as carrying a latched candidate arm $\hat { a } _ { t }$ and, at each round t, emitting an adopt-decision D<sub>t</sub> ∈ {hold} ∪ {switch $\boldsymbol { b } : \boldsymbol { b } \neq \hat { \boldsymbol { a } } _ { t } \}$ as a possibly randomized function of its observation history $\mathcal { H } _ { t }$ . Write $\mathcal { H } _ { ( t - m , t ] }$ for the restriction of that history to the last m rounds. The monitor has efective memory at most m if, for every t and conditioned on the current candidate $\hat { a } _ { t } ,$ the conditional law of $D _ { t }$ is determined by $\mathcal { H } _ { ( t - m , t ] }$ alone, that is any two histories agreeing on $\mathcal { H } _ { ( t - m , t ] }$ and on $\hat { a } _ { t }$ induce the same conditional law of $D _ { t }$ The efective memory mem(M) is the least such m, measured in rounds, and +∞ when no finite m works. The candidate $\hat { a } _ { t }$ is a $\lceil \log _ { 2 } K \rceil$ -bit latch, so a finite mem(M) bounds only the observation window the adopt-test reads, not the candidate state a monitor may carry indefinitely; equivalently $\mathrm { m e m } ( { \mathcal M } ) < m$ is exactly the “responsive uniformly over prehistory” property above, since a decision measurable with respect to the last m rounds cannot condition on anything earlier.

Lemma 10 (Responsiveness forces drift false alarms: uniform sub-P/2 responsiveness). On the square-wave instance above, every monitor that is $( \bar { \Delta } , d )$ -responsive uniformly over prehistory with detection delay $d < P / 2$ equivalently, every monitor whose adopt-decision depends only on a window $o f < P / 2$ rounds and that is switch-adaptive on clean instances—false-adopts arm 2 within each down-phase with probability at least $q _ { 0 }$ . The binding hypothesis is the efective memory $( < P / 2$ , uniform over prehistory), not the monitor’s internal form: within that class the bound is indiferent to whether the statistic is nonlinear $( e . g .$ a CUSUM/Page run-length test, a GLR statistic, or an adaptive threshold) or randomized. Classical CUSUM/Page/GLR detectors fall in the class precisely when tuned to detect within half a drift period—i.e. in the switch-adaptive regime that is the only regime of interest here—while a variant whose efective memory reaches $\geq P / 2$ is exempt from the conclusion but then, by the dichotomy below, fails to adopt genuine ∆<sup>¯</sup> -margin switches occurring within $P / 2$

Proof. Fix a down-phase with onset $t ^ { \star }$ . Build a genuine-switch comparison instance $\mathcal { T } _ { 1 }$ that (i) is identical to the drift instance $\mathcal { T } _ { 0 }$ at every round $t < t ^ { \star } + P / 2$ , and (ii) for $t \geq t ^ { \star }$ freezes the down-phase means—arm 2 at $\begin{array} { r } { \frac { 1 } { 2 } + \frac { 1 } { 2 } ( b - \bar { \Delta } ) } \end{array}$ , arm 1 at $\begin{array} { r } { \frac { 1 } { 2 } - \frac { 1 } { 2 } ( b - \bar { \Delta } ) } \end{array}$ so that under $\mathcal { T } _ { 1 }$ arm 2 is the time-average-best arm from $\tau = t ^ { \star }$ onward, with margin $b - \bar { \Delta } \geq \bar { \Delta }$ (using $b > 2 \bar { \Delta } )$ . Because the drift instance $\mathcal { T } _ { 0 }$ is itself at the down-phase means throughout $[ t ^ { \star } , t ^ { \star } + P / 2 )$ , the two instances have identical per-round mean paths—hence identical observation laws—on the entire prefix $[ 0 , t ^ { \star } + P / 2 )$

Apply responsiveness to $\mathcal { T } _ { 1 }$ at $\tau = t ^ { \star } \colon$ : M adopts arm 2 by round $t ^ { \star } + d$ with probability $\geq q _ { 0 }$ This adoption event is measurable with respect to the observations in $[ 0 , t ^ { \star } + d ) \subseteq [ 0 , t ^ { \star } + P / 2 )$ (since $d < P / 2 )$ , a prefix on which $\mathcal { T } _ { 0 }$ and $\mathcal { T } _ { 1 }$ are equal in law. Hence the same event has probability $\geq q _ { 0 }$ under $\mathcal { T } _ { 0 }$ . Under $\mathcal { T } _ { 0 }$ arm 1 is the time-average-best arm $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0 )$ , so adopting arm 2 is a false switch. Each down-phase therefore carries false-adoption probability $\geq q _ { 0 }$ , independent of $T$ and of the monitor’s internal form. □ □

The escape clause is now visible in full generality: a monitor avoids the down-phase false alarm only with detection delay $d \ge P / 2$ —memory spanning at least half a drift period, enough to net the down-phase against the preceding up-phase and recover the time-average sign—which is precisely non-responsiveness to a true ∆<sup>¯</sup> -margin switch within half a period. No monitor with sub-$P / 2$ detection delay, linear or nonlinear, escapes; this strictly generalizes part (b), which is the special case of a linear normalized kernel.

The dichotomy. Combining: for the square-wave instance, any responsive monitor with sub- $P / 2$ detection delay—a linear short-memory monitor with threshold $\theta \le \bar { \Delta }$ (claim (b)), or any nonlinear/randomized monitor (Lemma 10)—false-adopts with constant probability per down-phase, whereas detection delay $\geq P / 2$ (equivalently $\theta > \bar { \Delta }$ in the linear case) makes it non-adaptive to genuine $\bar { \Delta } { \cdot } \mathrm { m a r g i n }$ switches. No single threshold yields both drift-robustness and switch-adaptivity. DRFC-Seq does not escape the dichotomy—no monitor can—but it attains its drift-robust corner at the minimal cost the impossibility permits: its memory spans an integer number of full periods (its W probe-blocks covering $W \delta _ { \mathrm { p } }$ , a multiple of $P _ { \mathrm { : } }$ , rounds), so its target is the drift-invariant average $\bar { \Delta }$ (not a phase-dependent local contrast), giving claim (a); and because a single period sufices to recover the time-average sign, it still adopts a genuine ∆<sup>¯</sup> -margin switch within W probe-blocks (Section H, adaptivity (b))—one period of detection latency, the price the dichotomy charges any drift-robust monitor, rather than an evasion of it. This is the separation asserted in main Proposition 1. □

Corollary 4 (Memory–latency dichotomy on the efective-memory axis). Fix the period-P squarewave cancellation instance above, with time-average margin $\bar { \Delta } > 0$ and amplitude $b > 2 \bar { \Delta }$ . For every monitor M in the sense of Definition 3:

(i) Short memory false-alarms. If mem $( \mathcal { M } ) < P / 2$ and M is switch-adaptive, adopting a genuine $\bar { \Delta }$ -margin switch within mem(M) rounds with probability at least $q _ { 0 }$ , then in every down-phase M false-adopts with probability at least $q _ { 0 }$

(ii) Drift safety costs $\geq P / 2$ latency. If M is drift-safe on this instance, performing no false adopt with probability at least $1 - \alpha _ { i }$ , then a genuine switch placed at a down-phase onset is not adopted within $P / 2$ rounds with probability at least $1 - \alpha$ , so its worst-case genuine-switch detection latency is at least $P / 2$

Contrapositively, (i) shows any drift-safe and switch-adaptive monitor has $\mathrm { m e m } ( { \mathcal M } ) \ge P / 2$ , and (ii) charges any drift-safe monitor at least $P / 2$ latency regardless. The two horns are exhaustive over the memory–latency plane, so no monitor is both drift-safe and switch-adaptive with latency below $P / 2$ . DRFC-Seq attains the floor of $( i i ) .$ : its efective memory is the W probe-blocks spanning $W \delta _ { \mathrm { p } } \geq P$ rounds, and it adopts a genuine ∆<sup>¯</sup> -margin switch within that window (main DRFC-Seq theorem), one drift period of latency, the least (ii) permits up to the factor 2 and the probe-block granularity.

Proof. Part (i) is Lemma 10 read through Definition 3: $\mathrm { m e m } ( \mathcal { M } ) < P / 2$ is precisely uniform-overprehistory responsiveness with detection delay below $P / 2 ,$ , so the exact change of measure of that lemma transfers the genuine-switch adoption to a down-phase false adoption.

For part (ii), let s be the onset of a down-phase of the drift instance $\mathcal { T } _ { 0 }$ (no decision switch, arm 1 time-average-best). Construct the genuine-switch instance $\mathcal { T } _ { 1 }$ that is identical in law to $\mathcal { T } _ { 0 }$ on the entire prefix $[ 0 , s + P / 2 )$ and, from round s on, freezes the down-phase means so that arm 2 is time-average-best by margin $b - \bar { \Delta } \geq \bar { \Delta }$ from $\tau = s$ onward, exactly the construction in the proof of Lemma 10. The event $^ { 6 6 } { \mathcal { M } }$ performs no adopt before round $s + P / 2 "$ is measurable with respect to the observations on $[ 0 , s + P / 2 )$ , on which $\mathcal { T } _ { 0 }$ and $\mathcal { T } _ { 1 }$ have identical laws, so it carries the same probability under both. Under $\mathcal { T } _ { 0 }$ the regime has no switch, so any adoption is a false switch and drift safety gives this event probability at least $1 - \alpha$ . Transferring to $\mathcal { T } _ { 1 }$ , where a genuine switch occurred at $\tau = s$ , the monitor fails to adopt within $P / 2$ rounds with probability at least $1 - \alpha ;$ since the switch phase is adversarial, the worst-case detection latency is at least $P / 2$ . The argument is an exact two-instance coupling and uses no concentration inequality and no appeal to any upper bound of this paper. □ □

Remark 3 (Scope of the dichotomy vs. classical sequential change detection). We state precisely what Proposition 1 does and does not claim, to avoid conflation with the classical sequential changedetection literature (Page; Lorden; Lai). Our result is a dichotomy on the efective-memory axis of a detector, not a false-alarm-rate-constrained minimax over arbitrary stateful procedures. Concretely: (i) we do not prove a lower bound on the average-run-length-to-false-alarm of an optimally tuned Page/GLR procedure with full state; $( i i )$ we do prove that on the cancellation-drift instance, any detector whose adopt-decision is determined by a window $o f < P / 2$ rounds (uniformly over prehistory) false-adopts with constant probability per down-phase, while a detector that nets the cancellation, that is whose adopt-decision delay reaches $\geq P / 2$ , is correspondingly slow to adopt a genuine switch occurring within that half-period. A fully stateful, optimally tuned Page/GLR detector is not claimed to fail, instead paying this $\geq P / 2$ latency to stay drift-robust, which is the second horn. The two horns are exhaustive over the responsiveness–delay plane, which is why the trade-of cannot be tuned away within the switch-adaptive regime. This is the property the experiments confirm (the tuning sweep of Section I.1 traces both horns empirically). The novelty relative to the classical theory is the instance: cancellation drift makes the time-average and instantaneous decision signals disagree, so the memory length—not the false-alarm threshold—becomes the binding resource, a regime the single-stream change-detection minimaxes do not address.

## H.2 Doubling-Window DRFC-Seq: Period-Agnostic Monitoring

The main drift proposition charges any drift-robust monitor one drift period of memory, and Section H attains that corner with a single window W tuned so that $W \delta _ { \mathrm { p } }$ spans the period P. That tuning is the one benign knob of DRFC-Seq, and the main time-average margin assumption simply postulates it. The construction below removes the knob: it learns the shortest drift-safe window online, with no knowledge of $P ,$ paying only a one-time calibration and an additive log J inside the confidence radius from the union over the ladder. Write $L ( W ) : = W \delta _ { \mathrm { p } }$ for the in-rounds length of a W-block window.

The algorithm. Doubling-Window DRFC-Seq runs the Section H probe process once and reads it through a geometric ladder of sliding windows

$$
W _ { 0 } < W _ { 1 } < \cdots < W _ { J - 1 } = W _ { \mathrm { m a x } } , \qquad W _ { i } = 2 ^ { i } W _ { 0 } , \qquad J = \left\lceil \log _ { 2 } ( W _ { \mathrm { m a x } } / W _ { 0 } ) \right\rceil + 1 . 6 \left( \mathrm { W a x } / W _ { 0 } \right\rceil - 1 . 5 \left( \mathrm { W } - \mathrm { W } _ { 0 } \right) ^ { 2 } - 6 W _ { 0 } . 5 \left( \mathrm { W } - \mathrm { W } _ { 0 } \right) ^ { 3 } - 1 . 5 \left( \mathrm { W } - \mathrm { W } _ { 0 } \right) ^ { 4 } - 1 . 5 \left( \mathrm { W } _ { 0 } < \mathrm { W } _ { 0 } \right) ^ { 5 } - 1 .
$$

All windows share the one probe stream, so the sampling cost is that of the single window $W _ { \mathrm { m a x } } .$ not the sum. Window i runs the Section H confidence sequence at level $\alpha / J ;$ let $a _ { i } ( t )$ be its absolute best-arm certificate at probe-step $t ,$ the empirically best arm whose lower-confidence gap against the runner-up is positive (and hence positive against every other arm, the runner-up being the closest), or ⊥ if no arm separates from the field. This is exactly the top-versus-runner-up test the implemented monitor runs, so the certificate is an all-pairs dominance statement, not a comparison against the held candidate only. Window i is anointed at the first step at which $a _ { i }$ has equalled the top window’s certificate $a J { - 1 }$ on every one of the previous $W _ { \mathrm { m a x } }$ consecutive full-window steps, where matching requires the same non-⊥ arm at each step. The monitor’s operating window is the shortest anointed one, $W _ { \mathrm { o p } } ( t ) = \operatorname* { m i n } \{ W _ { i } : i$ anointed by t}, and it adopts a switch exactly when $a _ { \mathrm { o p } }$ moves of the held candidate. No adoption is made before the first anointment.

Theorem 7 (Period-agnostic anytime-valid monitoring). Run Doubling-Window DRFC-Seq on a within-regime drift instance from the waveform class of Section H.1 with time-average margin $\bar { \Delta } > 0$ (the main time-average margin condition) and unknown period P and amplitude b. Define a window W to be drift-safe when its windowing bias is strictly below the margin, ε<sub>W</sub> $< \bar { \Delta }$ , equivalently $L ( W ) > b P / ( 2 \bar { \Delta } )$ , and write its windowed margin $\rho _ { W } : = \bar { \Delta } - \varepsilon _ { W } > 0$ . Suppose the ladder top covers the period, is drift-safe with a constant-fraction margin, and is sample-suficient as the calibration authority,

$$
\begin{array} { r } { L ( W _ { \mathrm { m a x } } ) \geq \operatorname* { m a x } \biggr ( P , ~ \frac { b P } { \Delta } \biggr ) ( s o \varepsilon _ { W _ { \mathrm { m a x } } } \leq \frac { \bar { \Delta } } { 2 } ) , \qquad W _ { \mathrm { m a x } } h \geq n _ { 0 } ( \bar { \Delta } / 2 ) , } \end{array}
$$

where $n _ { 0 } ( \rho ) : = \operatorname* { i n f } \{ n : r _ { n } \leq \rho / 4 \} = \widetilde O ( 1 / ( N \rho ^ { 2 } ) )$ is the per-arm count of the recovery Lemma ${ \mathit { 9 , } }$ so that at margin ρ a window’s absolute certificate is non-⊥ with a strict gap (the balanced pairwise estimator has deviation $r _ { n }$ , giving $\mathrm { L C B } = \hat { G } - 2 r _ { n } \geq \rho - 3 r _ { n } \geq \rho / 4 > 0 )$ . The window advances one probe block per evaluation, the only times the monitor acts, so the relevant phases are the evaluated block-phases, and we take the period to be a whole number of probe blocks (a benign coarsening, since a block spans far fewer rounds than a period) so that a contiguous run of $P / s _ { \mathrm { b l k } }$ blocks enumerates every block-phase exactly, where $s _ { \mathrm { b l k } } = L ( W ) / W$ is the per-block round span (Remark $\it 4$ removes this coarsening for arbitrary real P at an explicit 2b/W of-grid slack). Let W <sup>⋆</sup> be the shortest ladder window the period-oracle would run, the shortest that is both drift-safe and sample-suficient, $L ( W ^ { \star } ) \ge b P / \bar { \Delta }$ and $W ^ { \star } h \ge n _ { 0 } ( \bar { \Delta } / 2 )$ ; such a window exists with $L ( W ^ { \star } ) \leq L ( W _ { \operatorname* { m a x } } )$ because $W _ { \mathrm { m a x } }$ meets both. Then, with $r _ { n }$ the Section H stitched radius:

(a) (Anytime-valid drift safety.) With probability at least $1 - \alpha ,$ simultaneously over all rounds $t \leq T$ , the monitor performs no false switch, that is no displacement of the time-average-best arm once it is held, during any pure-drift regime $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0 )$ ; the false-switch probability is $\leq \alpha _ { ; }$ , uniformly in $T$ and independent of $S _ { \mathrm { l o c } }$

(b) (Period-agnostic competitive latency.) After a calibration of at most $2 L ( W _ { \mathrm { m a x } } )$ rounds the shortest anointed window satisfies $W _ { \mathrm { o p } } \leq W ^ { \star }$ , and any genuine ∆<sup>¯</sup> -margin switch whose new regime spans at least $2 W _ { \mathrm { o p } }$ probe-blocks (the recovery-lemma spacing of the main DRFC-Seq theorem) is adopted within $O \left( L ( W ^ { \star } ) \right) = O \left( W ^ { \star } \delta _ { \mathrm { p } } \right)$ rounds, the per-switch latency of the oracle told P that runs the single window $\dot { W } ^ { \star }$ , up to the doubling factor 2 and the radius inflation of (c), and using no knowledge of P.

(c) (Union cost.) The ladder inflates the radius over a single tuned window only by the additive log $J = \log \log _ { 2 } ( W _ { \mathrm { m a x } } / W _ { 0 } )$ inside the stitched logarithm of $r _ { n } ;$ since log J is dominated by $\log ( C _ { 0 } K ( K { - } 1 ) s ^ { 2 } / \alpha )$ this is a $\left( 1 + o ( 1 ) \right)$ inflation of the radius, not a standalone multiplicative factor.

Proof. Step 0 (union confidence sequence; part $( c ) )$ . Apply the Section H, Step 1 confidence sequence to each of the J windows at level $\alpha / J$ and union-bound. On a good event $\mathcal { E } _ { J }$ with $\mathbb { P } [ \mathcal { E } _ { J } ] \ge 1 - \alpha$ every window’s balanced estimate obeys $| \hat { G } - \bar { G } | \leq r _ { n } ^ { ( J ) }$ for all per-arm counts n simultaneously, where $r _ { n } ^ { ( J ) }$ is $r _ { n }$ with α replaced by $\alpha / J _ { ; }$ , i.e. with the stitched term $\log ( C _ { 0 } K ( K { - } 1 ) s ^ { 2 } / \alpha )$ raised by the additive log J. Since log $J = \log \log _ { 2 } ( W _ { \mathrm { m a x } } / W _ { 0 } )$ is dominated by the log $( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha )$ already present, $r _ { n } ^ { ( J ) } = r _ { n } \big ( 1 + o ( 1 ) \big )$ ; this is (c). Write $r _ { n }$ for $r _ { n } ^ { ( J ) }$ and work on $\mathcal { E } _ { J }$ throughout.

Step 1 (anointment certifies true safety and a usable margin). For a window W write its worstphase margin $m ( W ) : = \mathrm { m i n } _ { s } \bar { G } _ { a ^ { \mathrm { a v g } } } ^ { ( s ) } ( L ( W ) )$ , the smallest over phases of the time-average-best arm’s windowed gap to its nearest challenger. The waveform-class bound of Section H.1 is used only in its safe direction, as an upper bound on the bias: $m ( W ) \geq \bar { \Delta } - \varepsilon _ { W }$ with $\varepsilon _ { W } = b P / ( 2 L ( W ) )$ , for every waveform in the class. We never need the bias to be attained, so the argument makes no squarewave-specific claim. The top window is drift-safe with a constant-fraction margin, $m ( W _ { \mathrm { m a x } } ) \ \geq$ $\bar { \Delta } - \varepsilon _ { W _ { \mathrm { m a x } } } \geq \bar { \Delta } / 2 > 0$ , and once filled is sample-suficient $( W _ { \mathrm { m a x } } h \ge n _ { 0 } ( \bar { \Delta } / 2 )$ , so $r _ { n } \leq \bar { \Delta } / 8$ and its certified gap $\hat { G } - 2 r _ { n } \ge m ( W _ { \mathrm { m a x } } ) - 3 r _ { n } > 0 )$ ; hence on $\mathcal { E } _ { J }$ its absolute certificate is the time-average-best arm at every phase and is never ⊥, $a _ { J - 1 } \equiv a ^ { \mathrm { a v g } }$ . Now consider any window W anointed at t. The monitor evaluates only at probe blocks and the window start advances one block per evaluation, so the phases the monitor ever sees form the block-grid. Anointment requires $W \mathrm { { s } }$ certificate to equal $W _ { \mathrm { m a x } } \mathrm { ^ { * } s }$ on $W _ { \mathrm { m a x } }$ consecutive full-window blocks, a span over which the start sweeps $L ( W _ { \mathrm { m a x } } ) \ge P$ rounds, at least one full drift period. Under the whole-block period convention this contiguous run enumerates every block-phase the monitor will ever evaluate, $W \mathrm { { s } }$ own worst block-phase $s _ { W }$ included (Remark 4 removes the convention for arbitrary real $P .$ , carrying an explicit of-grid slack $2 b / W )$ . Since $W _ { \mathrm { m a x } } \mathrm { ^ { * } s }$ certificate there is the non-⊥ arm $a ^ { \mathrm { a v g } }$ , matching forces W to certify the same $a ^ { \mathrm { a v g } }$ at full count $n _ { W } = W h$ , so its lower-confidence gap of $a ^ { \mathrm { a v g } }$ over the runner-up was positive, $\hat { G } - 2 r _ { n _ { W } } > 0$ , whence on $\mathcal { E } _ { J }$

$$
m ( W ) = \bar { G } _ { a ^ { \mathrm { a v g } } } ^ { ( s _ { W } ) } ~ \geq ~ \hat { G } - r _ { n _ { W } } ~ > ~ r _ { n _ { W } } ~ > ~ 0 ,
$$

the worst-phase margin against the nearest challenger, not merely against the held candidate. Thus every anointed window is truly drift-safe $( m ( W ) > 0$ , its candidate-arm target positive at every phase, for any waveform in the class) and carries the self-certified margin $m ( W ) > r _ { n _ { W } }$ . No appeal to a drift-unsafe window producing a wrong certificate is made, so the step holds for sine and triangle drifts exactly as for the square wave.

Step 2 (part $( a ) )$ . The monitor adopts only through the operating window $W _ { \mathrm { o p } } = \operatorname* { m i n } \{ W _ { i }$ anointed} (and adopts nothing before the first anointment), which by Step 1 is truly drift-safe, with candidate-arm target $\ge m ( W _ { \mathrm { o p } } ) > 0$ at every phase. In a pure-drift regime $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0 )$ the timeaverage-best arm $a ^ { \mathrm { a v g } }$ holds, and any challenger $a ^ { \prime }$ has windowed gap $\bar { G } _ { a ^ { \prime } , a ^ { \mathrm { a v g } } } \leq - m ( W _ { \mathrm { o p } } ) < 0 ;$ using only $m ( W _ { \mathrm { o p } } ) ~ > ~ 0$ from Step 1 and the anytime confidence sequence, its lower-confidence gap is $\hat { G } _ { a ^ { \prime } , a ^ { \mathrm { a v g } } } - 2 r _ { n } \le ( - m ( W _ { \mathrm { o p } } ) + r _ { n } ) - 2 r _ { n } < 0$ at every full-window count n the monitor evaluates, with no $n _ { 0 }$ needed. No challenger ever attains a positive lower-confidence gap, so the operating window never moves of $a ^ { \mathrm { a v g } }$ and no adoption occurs. A handof to a newly anointed shorter window is between two truly drift-safe windows that both certify $a ^ { \mathrm { a v g } }$ , so it is silent. The false-switch probability is at most $\mathbb { P } [ \mathcal { E } _ { J } ^ { \mathrm { c } } ] \leq \alpha$ , uniformly over all $t \leq T$ and independent of $S _ { \mathrm { l o c } }$ . This is the proof–algorithm alignment point: adoption is gated on anointment, the empirically checkable certificate that Step 1 turns into true drift-safety, and the proof never assumes a window safe that the algorithm has not anointed.

Step 3 (part (b)). The top window fills in $W _ { \mathrm { m a x } }$ blocks and, being drift-safe and sample-suficient, agrees with itself thereafter, so it is anointed by $2 L ( W _ { \mathrm { m a x } } )$ rounds. Consider $W ^ { \star }$ , the shortest ladder window that is both drift-safe and sample-suficient $( L ( W ^ { \star } ) \ge b P / \bar { \Delta }$ and $W ^ { \star } h \geq n _ { 0 } ( \bar { \Delta } / 2 ) )$ ; by the safe direction of the bias bound $m ( W ^ { \star } ) \ge \bar { \Delta } - \varepsilon _ { W ^ { \star } } \ge \bar { \Delta } / 2$ , and its sample-suficiency holds by definition. Since $L ( W ^ { \star } ) \leq L ( W _ { \operatorname* { m a x } } )$ it fills inside the calibration window; on $\mathcal { E } _ { J }$ it certifies $a ^ { \mathrm { a v g } }$ at every probe-step phase $( \mathrm { m a r g i n } \ge \bar { \Delta } / 2 , 2 r _ { n } \le \bar { \Delta } / 2$ once filled) and so equals $W _ { \mathrm { m a x } } \mathrm { ^ { * } s }$ certificate over any period, hence is anointed by $2 L ( W _ { \mathrm { m a x } } )$ . Thus the shortest anointed window satisfies $W _ { \mathrm { o p } } \leq W ^ { \star }$ Whatever $W _ { \mathrm { o p } }$ is, Step 1 gives it a self-certified margin $m ( W _ { \mathrm { o p } } ) > r _ { n _ { \mathrm { o p } } }$ at $n _ { \mathrm { o p } } = W _ { \mathrm { o p } } h$ . By the polynomial dependence of the stitched radius, $\rho > r _ { n }$ at n forces $n _ { 0 } ( \rho ) = { \cal O } ( n )$ , so $n _ { 0 } ( m ( W _ { \mathrm { o p } } ) ) =$ $O ( W _ { \mathrm { o p } } h )$ : the operating window already holds enough samples to recover. Now take a genuine average-decision switch at $\tau$ past calibration, where $a _ { r } ^ { \mathrm { { a v g } } }$ becomes time-average-best by margin $\bar { \Delta }$ and the new regime spans at least $2 W _ { \mathrm { o p } }$ probe-blocks. The recovery lemma (Lemma 9), whose regime-spacing hypothesis is thus met, applied to $W _ { \mathrm { o p } }$ adopts $a _ { r } ^ { \mathrm { a v g } }$ once the post-switch window fills and reaches $n _ { 0 } ( m ( W _ { \mathrm { o p } } ) ) = O ( W _ { \mathrm { o p } } h )$ samples, i.e. within $O ( W _ { \mathrm { o p } } \delta _ { \mathrm { p } } ) = O ( L ( W _ { \mathrm { o p } } ) ) \le O ( L ( W ^ { \star } ) )$ rounds. This is the per-switch latency of the oracle told P that runs $W ^ { \star }$ , up to the doubling granularity and the $r _ { n }$ inflation of (c). The latency is in rounds, set by the window length in probeblocks times $\delta _ { \mathrm { p } }$ , with no spurious division by $N$ . No knowledge of $P$ enters and the calibration $2 L ( W _ { \mathrm { m a x } } )$ is paid once. □

Remark 4 (Arbitrary real periods, of the block grid). The whole-block period convention used in Step 1 $( P$ an integer multiple of the probe-block span $\delta _ { \mathrm { p } } )$ is a simplifying device, not a requirement, and the guarantee degrades gracefully for an arbitrary real P. The windowed time-average gap is Lipschitz in the start phase, since advancing the window by one block replaces a single boundary block of per-block gap within b of the mean,

$$
\vert \bar { G } _ { a ^ { \mathrm { a v g } } } ^ { ( s + \delta _ { \mathrm { p } } ) } ( L ( W ) ) - \bar { G } _ { a ^ { \mathrm { a v g } } } ^ { ( s ) } ( L ( W ) ) \vert \leq \frac { 2 b } { W } = : \varepsilon _ { W } ^ { \mathrm { o f f } } .
$$

The monitor evaluates only block-grid phases, so the worst evaluated phase lies within one block of the true worst phase s and underestimates the worst-phase margin by at most $\varepsilon _ { W } ^ { \mathrm { o f f } }$ . Anointment over a $W _ { \mathrm { m a x } } – b l o c k$ run still sweeps $L ( W _ { \mathrm { m a x } } ) \ge P$ rounds, hence an evaluated phase within $\delta _ { \mathrm { p } }$ of $s _ { W }$ , and matching $W _ { \mathrm { m a x } } \ ' _ { s }$ non-⊥ certificate there gives, in place of the exact Step 1 conclusion, $m ( W ) \geq r _ { n _ { W } } - \varepsilon _ { W } ^ { \mathrm { o f f } }$ . Every anointed window is therefore truly drift-safe whenever $\varepsilon _ { W } ^ { \mathrm { o f f } } < r _ { n _ { W } }$ , that is $W \gtrsim { b ^ { 2 } N h } / \log ( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha )$ , an explicit lower bound the operating window meets once $W _ { 0 }$ is enlarged by a constant factor. Under this mild strengthening part $( a ) \mathit { \Pi } _ { s }$ no-false-switch guarantee holds $f o r$ arbitrary real $P _ { z }$ with the whole-block case recovered as the $\varepsilon _ { W } ^ { \mathrm { o f f } } = 0$ specialization, and the latency (b) is unchanged up to the additive $\varepsilon ^ { \mathrm { o f f } }$ inside the margin. The experiment of Section I samples periods of the block grid and reports the predicted zero false-switch rate, so it matches this analysis rather than relying on the convention.

Remark 5 (The price of period-agnosticism is honest, not zero). Theorem 7 does not evade the Proposition 1 dichotomy. It still spends one drift period of memory at the operating window $W _ { \mathrm { o p } } \leq$ $W ^ { \star }$ , exactly the charge the dichotomy levies, and it adds a one-time $O ( L ( W _ { \mathrm { m a x } } ) )$ calibration that is the information cost of discovering the safe scale without being told P: any monitor that adopts a $\bar { \Delta } \cdot$ margin switch faster than $L ( W _ { \mathrm { m a x } } )$ could not yet have ruled out a drift period just below $L ( W _ { \mathrm { m a x } } )$ so some calibration is unavoidable. What the theorem buys is that this cost is paid once rather than per regime, after which the monitor runs at the oracle’s per-switch latency. Section I reports the matching experiment: across randomized periods, phases, and square, triangle, and sine drift shapes, Doubling-Window DRFC-Seq holds the oracle’s false-switch rate and tracks its detection latency without the period as input, while any single fixed window is either drift-unsafe at the long end or needlessly slow at the short end.

Remark 6 (Relation to multi-scale selection). The ladder-of-windows shape recalls Lepski’s method for bandwidth selection Lepski˘ı [1991], but the selection principle difers in three ways the driftmonitoring problem forces. First, the objects compared are decisions, the certified argmax arms $a _ { i ; }$ not estimator values within overlapping confidence intervals, with agreement on which arm is best rather than on a real-valued estimate. Second, the agreement test is temporal, a match over a full $W _ { \mathrm { m a x } } – b l o c k$ sweep of one drift period rather than at a single time, which is what converts a pointwise certificate into the anytime-valid drift-safety of part $( a )$ by forcing a candidate window to certify the best arm at its own worst phase. Third, the authority is anchored to safety, the longest drift-safe window, not merely to the largest scale, and the whole construction is anytime-valid through the union confidence sequence over the ladder. The output therefore carries a uniform-in-time no-falseswitch guarantee, a property estimation-risk bandwidth selection does not target.

## I. Additional Experimental Figures

This section contains the experimental figures referenced in the main paper as “supplement.” All figures and tables are reproduced from the anonymized code-and-data package accompanying the paper. The simulators use only the Python standard library, and the figure layer uses numpy and matplotlib; all tabular inputs needed for deterministic reproduction are included in that package. The descriptions below preserve the parameters, seeds, and statistical aggregation needed to reproduce each reported result.

## Full versions of main-paper experiment figures

These are the full-size figures and tables summarized in the main paper’s Experiments and Related Work sections (positioning, real-data corroboration, and secondary scaling results). Main-result references below use result names whenever possible so they remain stable under section reordering.

The comparator choice does not manufacture the result. With one arm played per agent per round, instantaneous dynamic regret and decision regret difer by a single instance constant, $\begin{array} { r } { R _ { T } ( \pi ) - R _ { T } ^ { \mathrm { d e c } } ( \pi ) = \sum _ { t } { N \left[ \bar { \mu } _ { a _ { t } ^ { \star } } ( t ) - \bar { \mu } _ { a _ { r } ^ { \mathrm { a v g } } } ( t ) \right] } = : D \geq 0 } \end{array}$ , because each policy’s own action terms cancel. D does not depend on the policy π, so replacing the time-average comparator $a _ { r } ^ { \mathrm { a v g } }$ by the instantaneous best arm $a _ { t } ^ { \star }$ adds the same D to every method and leaves all rankings and gaps unchanged. For this instance $D \approx 4 2 5 2$ , computed from the equal-agent means (8 of 16 real bins reverse, minimum instantaneous global gap −0.080, time-average margin 0.0115). The $R _ { T }$ column applies this ofset. DRFC-Seq stays the lowest-regret method under the instantaneous comparator as well (6962 against DecCUSUM 7305 and the Oracle 8103), so the separation is a property of the methods, not of the comparator. The diagnostic is included in the reproducibility package.

![](images/07277e743985bb8b5adc9e71b741bf32384d5587799755fd7845b09cb7639b82.jpg)  
Figure 2: Full stochastic decision-switch sweep. Regret grows linearly with $S _ { \mathrm { d e c } }$ across all baselines; DRFC tracks the switch count with low false-switch rate.

## I.1 Decision-level baseline tuning (fairness)

The two decision-level change detectors we add—DL-GLR-klUCB (GLR on the global aggregated gap) and DL-ADR-bandit (multi-scale adaptive restart on the global mean)—are run at single operating points in the main paper (GLR threshold 6.0; ADR threshold scale 1.0 and minimum window 2). A natural concern is whether these points were chosen to make the baselines look weak, i.e. whether a more conservative setting would let them keep their switch-adaptivity while avoiding the within-regime-drift false alarms. Table 10 sweeps each detector’s primary sensitivity knob and, at every setting, measures both the genuine-switch adopt rate (a real mid-horizon switch, no drift) and the drift-only false-switch rate (no switch, live drift $b = 0 . 2 0 )$ , at the same live operating point $( N = 2 4 , \Delta = 0 . 1 0$ , horizon 24000, 30 seeds) as main Table 2 (the adaptivity battery).

The sweep is not a search for a better threshold; it is a direct test of the dichotomy of main Proposition 1. No setting of either detector reaches the both-good corner. Holding the detector switch-adaptive (adopt rate ≈ 1) pins its drift false-switch rate at 0.60–0.90; the only way to drive the false-switch rate down is to make the detector progressively non-adaptive—GLR at threshold 20 already misses half the genuine switches (adopt 0.50) while still false-alarming at rate 0.27, and ADR reaches a low false-switch rate only by collapsing to adopt rate 0.30 (scale 2.5) or total inertia (adopt 0, scale 4.0). The single points used in the main paper sit in the adaptive regime, which is the setting most favourable to the baselines (they are compared where they actually function as switch detectors, not where they are inert). DRFC-Seq is of this trade-of curve entirely: adopt rate

![](images/9a2f03692e0575820bd201e6b7d9472408b56a95d352bec7dc4cf9eaf895b344.jpg)  
Figure 3: Component ablations. Sample-weighted aggregation flips sign on heterogeneous-samplecount instances; sequential block estimation flips sign under within-window drift. Equal-agent averaging and randomized interleaving avoid both failure modes.

1.00 and false-switch rate 0.00 simultaneously, at a fixed level $\alpha = 0 . 1 0$ with no threshold to tune.

## I.2 Per-round communication cost

The source-record complexity bound (Proposition 3) is here made concrete and compared to the baselines, answering the question of whether source-traceable gossip and periodic scanning are bandwidth-heavy in practice. Table 11 reports the mean source-record transmissions per agent per round at the standard K-ary operating point $( N = 1 2 , K = 4 , S _ { \mathrm { d e c } } = 2$ , edge probability 0.50, horizon 1200, 20 seeds), read from the archived per-seed communication logs under the simulator’s common gossip accounting, where every method is charged for each record it floods over the random graph.

The conservative DRFC certifier transmits about 1.13 records per agent per round, within a factor 1.3 of the decision-level change detectors (DecCUSUM, CUSUM-Reset, LocalReset, all near 0.9) and about 2.3 times the passive index sharing of SW-UCB-Dec, which only broadcasts a $K \mathfrak { - }$ vector each gossip cycle. The light-probe and Oracle variants spend more trafic (1.36 and 1.30) to buy lower regret, the regret-trafic trade of the probe variant (Appendix C) and the Pareto figure. So the periodic certifier is comparable to a change detector rather than an order heavier, and its cost is governed by the monitor period M and block size h through the $\widetilde { O } ( N K ( h + D _ { G } ) T / M )$ rate of Proposition 3, with the per-scan and per-probe source-record displays $\ddot { O } ( B _ { r } )$ and $O ( N K q _ { p } )$ of that proposition giving the explicit per-event message complexity the deployment can tune. Dividing the total rate by NT, the per-agent per-round cost is $\widetilde { O } ( K ( h + D _ { G } ) / M )$ , which depends on the network size N only through the flooding time $D _ { G } .$ so the per-agent bandwidth grows with the network only through $D _ { G }$ and is otherwise flat in N, while the only N-dependence of the total is the unavoidable linear factor from having N agents. The component split is therefore a constant certifier overhead set by M and h plus a graph term that scales with $D _ { G }$ , the same $D _ { G }$ the topology and switching-stress sweeps (Sections , ) vary directly.

![](images/5fdb989d84728d5ebe4348f353a2c3df072cc35fa9504508fbdfe7165e818a52.jpg)

b  
![](images/e39145776954de3ddae7c6cb63a639c2cfc869f1c0d7dd701bc92838c59ad11f.jpg)  
Figure 4: Communication ablations. Naive gossip over-counts high-propagation records and biases the reconstructed gap; source-traceable deduplication preserves the equal-source estimator. Synchronization windows reduce agent disagreement after switches.

## I.3 Consolidated parameter-tuning guidelines

This subsection collects the practical guidance for choosing the parameters $( M , h , H _ { \mathrm { s c a n } } , \delta _ { \mathrm { p } } , W , \alpha , C )$ in one place, including what each controls, the reported default, and what to do when the gap $\Delta$ or the drift period is unknown. Table 12 is the summary; the main text carries a distilled version. The recurring principle is that the two margin-dependent knobs, the scan budget $H _ { \mathrm { s c a n } }$ (conservative DRFC) and the window W (DRFC-Seq), are the only inputs that need prior knowledge, and each has a knowledge-free doubling replacement that pays only a log log or log J overhead.

When $\Delta$ is unknown. Use Gap-Adaptive DRFC (Proposition 2), which walks the scan budget up a geometric ladder $H _ { \mathrm { s c a n } } ^ { ( k ) } \propto 4 ^ { k }$ and attains the Theorem 5 regret at the running-minimum gap with only an additive log log T overhead, so no margin input is needed above a floor.

When the drift period is unknown. Use Doubling-Window DRFC-Seq (Theorem 7), which runs a geometric ladder of sliding windows on one shared probe stream and adopts on the shortest window that has matched the longest drift-safe window, recovering the period-oracle’s latency within the doubling factor and a $( 1 + o ( 1 ) )$ union inflation, with no period input.

The benign knobs. The monitor period M trades periodic monitoring cost $( \widetilde O ( N K ( h \ +$ $D _ { G } ) T / M ) )$ against post-switch detection delay (O(M)), so M scales with the communication budget and the tolerable switch latency, not with $\Delta$ . The block size h sets the per-block radius and is a small constant. The probe gap $\delta _ { \mathrm { p } }$ need only satisfy $\delta _ { \mathrm { p } } \geq K h + D _ { G } ( \alpha )$ so each block fully floods before the next. The level α is the horizon-uniform false-switch budget and is robust across a 20× range (Table 14). The radius constant $C$ errs only conservatively as it grows (Table 19), so the practical $C \ = \ 1 . 1 5$ and the worst-case theorem value bracket a range over which the zero-false-switch conclusion is preserved.

![](images/868f8cca73762e6e6c394981cb1bf5e14e83549d7d27005e12f1375f27a63ccd.jpg)  
Figure 5: User-cluster semi-real stress test. Left: with $S _ { \mathrm { d e c } } = 0$ and varying $S _ { \mathrm { l o c } } ,$ conservative DRFC and DRFC-Probe are insensitive to the drift rate, while LocalReset, SW-UCB-Dec, and CUSUM-Reset all grow with $S _ { \mathrm { l o c } }$ . Right: with $S _ { \mathrm { l o c } }$ fixed, DRFC tracks $S _ { \mathrm { d e c } }$

## I.4 Beyond Erdős–Rényi: topology robustness

The main paper’s random-graph sweep uses Erdős–Rényi activation. To test whether the result depends on that choice we re-run DRFC at $S _ { \mathrm { d e c } } = 2$ over five structurally distinct base graphs, each made time-varying by activating every base edge independently per gossip round, with 20 seeds. The families are the regular low-diameter ring-with-chords (paper default), a Watts–Strogatz smallworld graph, a random geometric graph, a Barabási–Albert scale-free graph, and the path graph of maximal diameter $N - 1$

Figure 13 reports group regret and post-switch detection delay against the source-record reach rate, the fraction of flooded records that reach every agent and the empirical proxy for 1 minus the flooding-failure probability, hence for the flooding time $D _ { G }$ . The reading is that what governs DRFC is the realized reach rate, not the topology label. At a sparse activation probability 0.30 the reach rate ranges from 0.68 on the small-world graph down to 0.22 on the geometric graph, and DRFC regret tracks it monotonically, from 922 at the highest reach to 952 at the lowest, with the four connected families collapsing onto a single reach-versus-regret curve. As the activation probability rises to 0.80 every connected family reaches a reach rate of 1.00 and a regret near 900 within noise, regardless of whether the graph is a regular ring, a clustered geometric graph, or a hub-dominated scale-free graph. The path graph is the one persistent stress case. Its single-line base graph fails to flood $\left( \mathrm { r e a c h } \approx 0 \right)$ even at activation 0.80, because a single inactive edge severs the line, so DRFC cannot recover switches and its detection delay stays censored at the horizon. This is exactly the $D _ { G } \to N$ corner where the $N S _ { \mathrm { d e c } } D _ { G }$ term of Theorem 5 blows up, the honest boundary of the guarantee rather than a counterexample to it. The experiment therefore supports the claim that only $D _ { G }$ no finer topology functional, enters the regret, and it directly answers the topology generalization question by spanning small-world, clustered, scale-free, and maximal-diameter graphs.

## I.5 Switching-regime stress: validating the $\Delta ,$ spacing, and $D _ { G }$ dependence

To validate the way Theorem 5 and the main information-theoretic lower bound depend on the perregime certification $\mathcal { C } _ { r } .$ , the connectivity $D _ { G } ,$ and the per-regime scan count, we sweep three axes in the genuine-switching regime $S _ { \mathrm { d e c } } > 0$ , with 20 seeds, each isolating one predicted dependence (Figure 14).

![](images/7a3032ccf6aaf2710d67bf51cf021a0d1b7006407648043ed3dc9a0691d4662f.jpg)  
Figure 6: Graph-probability sweep. As Erdős–Rényi edge probability p rises from 0.22 to 0.50, sourcerecord reach rate climbs from 0.04 to 0.99, decision-switch recovery delay shrinks, and DRFC regret falls accordingly.

Gap ∆. Holding $S _ { \mathrm { d e c } } = 2$ and edge probability 0.50, as the gap grows over {0.08, 0.12, 0.18, 0.26} the post-switch detection delay falls from 382 rounds to 117, because a wider gap is certified with fewer balanced blocks, the empirical face of the $\mathcal { C } _ { r } \sim ( K - 1 ) / \Delta$ certification cost. Group regret rises over the same range from 1111 to 3262, because in this frequently-monitored horizon the dominant cost is the periodic monitoring term, whose each wasted probe pull of a suboptimal arm costs about $\Delta ,$ so the two branches of the bound appear on the two metrics, certification on the delay and monitoring on the regret.

Connectivity $D _ { G }$ . Holding $S _ { \mathrm { d e c } } ~ = ~ 2$ and $\Delta \ = \ 0 . 1 0$ , raising the edge probability over {0.25, 0.40, 0.60, 0.85} lifts the reach rate from 0.07 to 1.00 and collapses the detection delay from 1667 rounds to about 200, the direct empirical signature of the $N S _ { \mathrm { d e c } } D _ { G }$ delay term, since higher connectivity shrinks the flooding time $D _ { G }$

Switch count $S _ { \mathrm { d e c } }$ . Holding $\Delta = 0 . 1 0$ , edge probability 0.50, and the inter-switch spacing fixed at a regime length of 600 rounds (horizon scaled as $6 0 0 ( S _ { \mathrm { d e c } } + 1 ) )$ , group regret grows linearly in $S _ { \mathrm { d e c } }$ over {1, 2, 3, 4}, from 845 to 1388 to 1877 to 2414, with near-constant increments of about 520. This isolates the $N S _ { \mathrm { d e c } }$ switch term, showing the per-switch cost is constant once the spacing is held fixed, so the linear growth is in the switch count and not an artifact of a shrinking regime.

Across all three axes the false-switch rate stays 0, and the qualitative scalings match the bound, not its constants, as elsewhere in this supplement.

## I.6 Asynchrony and partial participation

The main algorithm schedules a synchronized balanced block in which every agent participates. We now show the equal-agent estimator degrades gracefully when only a random subset of agents participates in each block, the partial-participation model of asynchronous cooperative bandits where in each round an unknown time-varying subset of agents is active [Wang et al., 2025], and that the natural re-normalization keeps the estimator unbiased, answering the question of whether equal-agent averaging can absorb dropouts.

![](images/d81fff7e3b35551f2b1c7308894940d8abac58f445e38c396ea0eee570dc26bd.jpg)  
Figure 7: Monitor period M sensitivity at $S _ { \mathrm { d e c } } = 2 , K = 4 , \Delta = 0 . 1 0$ . Left: regret mean ± std vs $M \in \{ 8 0 , 1 2 0 , 2 0 0 \}$ . Right: detection delay vs M. The expected trade-of appears: larger M reduces regret from periodic certification cost but lengthens detection delay after a true switch. DRFC-Probe and Oracle dominate at large $M ;$ LocalReset is insensitive to M because it triggers on local-change boundaries, not monitoring schedule.

Oblivious dropout model. At each balanced block let $\Pi \subseteq [ N ]$ be the participating set, each agent included independently with probability $p ,$ drawn independently of the rewards and of the mean path (the oblivious analogue of the main reward-obliviousness assumption). A participating agent pulls every active arm on the block’s balanced schedule, a non-participating one contributes no pulls. The re-normalized equal-agent block estimator of arm a averages over the participants, $\begin{array} { r } { \hat { \mu } _ { a } ^ { \Pi } = | \Pi | ^ { - 1 } \sum _ { i \in \Pi } \hat { g } _ { i , a } } \end{array}$ , where $\hat { g } _ { i , a }$ is agent i’s within-block empirical mean. The naive estimator keeps dividing by the full count, $\begin{array} { r } { \hat { \mu } _ { a } ^ { \mathrm { n a i v e } } = N ^ { - 1 } \sum _ { i \in \Pi } \hat { g } _ { i , a } } \end{array}$

Proposition 4 (Re-normalization is unbiased under oblivious dropout). Fix a block with active arms $a , b$ and per-agent means $\mu _ { i , a }$ . Under the oblivious dropout model, conditioned on $| \Pi | = m$ for any $m \geq 1$ 2

$$
\mathbb { E } \big [ \hat { \mu } _ { a } ^ { \Pi } - \hat { \mu } _ { b } ^ { \Pi } \big | | \Pi | = m \big ] = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( \mu _ { i , a } - \mu _ { i , b } ) = \bar { G } _ { a , b } ,
$$

the all-agent equal-agent gap, so the re-normalized gap estimator is exactly unbiased for $\bar { G } _ { a , b }$ for every realized participation level. Its anytime radius uses the realized pull count $| \Pi | h ,$ , so over a length-W window with mean participation p the efective per-arm count is pNWh and the radius $r _ { n }$ inflates by a factor $1 / { \sqrt { p } }$ over full participation. The naive estimator instead satisfies $\mathbb { E } [ \hat { \mu } _ { a } ^ { \mathrm { n a i v e } } - \hat { \mu } _ { b } ^ { \mathrm { n a i v e } } \mid \lvert \Pi \rvert =$ $m ] = ( m / N ) \bar { G } _ { a , b }$ , so its gap is biased by the participation fraction, shrinking to $p \bar { G } _ { a , b }$ in expectation.

Table 4: Scaling of decision regret with arms K and agents N (mean ± std over 12 seeds for this sweep, mixed topologies). For the decision-level methods regret grows with K and with N, consistent with the $N S _ { \mathrm { d e c } }$ term of the upper bound. Because this is group regret, a sum over the N agents, the near-linear growth in N is expected from the summation alone and is not by itself a test of the communication term $N S _ { \mathrm { d e c } } D _ { G }$ , which is the smaller higher-order departure isolated separately by the graph-probability sweep below. DRFC-KGraph tracks Oracle (the perfect-communication reference, not a lower bound) to within noise. DRFC-Probe spends source trafic to stay nearly flat in N (the probe trade-of of Appendix C), so it is non-monotone in N and lowest at large N. This sweep isolates $K / N$ scaling of the decision-level mechanism, the separation from local-changereactive protocols is the distinct experiment in Figure 8. Top block DRFC family, bottom block decision-level baselines.
<table><tr><td colspan="5">(a) Arms K (N = 12)</td></tr><tr><td>Method</td><td>K=2</td><td> $K { = } 4$ </td><td> $K { = } 8$ </td><td> $K { = } 1 6$ </td></tr><tr><td rowspan="3">DRFC-KGraph DRFC-Probe Oracle</td><td> $3 3 1 { \pm } 4 0 $ </td><td> $8 2 3 { \pm } 2 2$ </td><td> $1 0 3 7 { \pm } 3 7$ </td><td> $1 1 5 4 { \pm } 2 8 $ </td></tr><tr><td> $1 1 8 { \pm } 4 2$ </td><td> $5 9 0 { \pm } 1 2 7$ </td><td> $9 8 1 { \pm } 5 6 $ </td><td> $1 1 3 8 { \pm } 2 5 $ </td></tr><tr><td> $3 1 7 { \pm } 2 9$ </td><td> $8 5 6 \pm 2 5$ </td><td> $1 0 5 6 { \pm } 2 0 $ </td><td> $1 1 5 6 { \pm } 1 5$ </td></tr><tr><td rowspan="3">SW-UCB-Dec CUSUM-Reset</td><td> $3 5 6 \pm 2 3$ </td><td> $7 9 1 { \pm } 2 3 $ </td><td> $1 1 2 4 { \pm } 1 9$ </td><td> $1 3 0 9 { \pm } 7 $ </td></tr><tr><td> $4 5 8 { \pm } 1 1$ </td><td>821±14</td><td> $1 0 7 3 { \pm } 1 8$ </td><td> $1 2 3 6 { \pm } 2 2$ </td></tr><tr><td>(b) Agents</td><td> $N \ ( K = 4 )$ </td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Method</td><td>N=6</td><td> $N { = } 1 2$ </td><td> $N { = } 2 4$ </td><td> $N { = } 4 8$ </td></tr><tr><td rowspan="3">DRFC-KGraph DRFC-Probe Oracle</td><td> $4 0 8 { \pm } 1 3$ </td><td> $8 2 3 { \pm } 2 2$ </td><td> $1 5 8 8 { \pm } 3 4$ </td><td> $3 1 1 0 { \pm } 0$ </td></tr><tr><td> $3 4 9 { \pm } 6 9$ </td><td> $5 9 0 { \pm } 1 2 7$ </td><td> $4 9 0 { \pm } 1 3 1$ </td><td>778±0</td></tr><tr><td> $4 1 8 { \pm } 1 3$ </td><td> $8 5 6 \pm 2 5$ </td><td> $1 6 1 2 { \pm } 5 2$ </td><td> $3 1 1 0 { \pm } 0$ </td></tr><tr><td>SW-UCB-Dec CUSUM-Reset</td><td> $4 4 2 { \pm } 1 3$   $3 9 7 { \pm } 1 2 $ </td><td> $7 9 1 { \pm } 2 3 $   $8 2 1 { \pm } 1 4$ </td><td> $1 3 1 5 { \pm } 5 0 $   $1 6 8 1 { \pm } 1 1$ </td><td>1984±50  $3 3 8 5 { \pm } 1 7$ </td></tr></table>

Proof. Conditioned on $| \Pi | = m$ , the set Π is a uniformly random m-subset of [N] because the inclusions are i.i.d. and exchangeable, so ${ \mathbb { P } } [ i \in \Pi \mid | \Pi | = m ] = m / N$ for every i. The block schedule is fixed independently of the rewards, so $\mathbb { E } [ \hat { g } _ { i , a } - \hat { g } _ { i , b } ] = \mu _ { i , a } - \mu _ { i , b }$ . Hence

$$
\begin{array} { r } { \mathbb { E } \big [ \frac { 1 } { m } \sum _ { i \in \Pi } ( \hat { g } _ { i , a } - \hat { g } _ { i , b } ) \big | | \Pi | = m \big ] = \frac { 1 } { m } \displaystyle \sum _ { i = 1 } ^ { N } \frac { m } { N } ( \mu _ { i , a } - \mu _ { i , b } ) = \bar { G } _ { a , b } , } \end{array}
$$

independent of $m ,$ , which is the first claim. The radius statement is the concentration radius of Lemma 3 with N replaced by the realized participant count, whose window mean is $p N$ . For the naive estimator the $1 / m$ is replaced by $1 / N$ , multiplying the conditional mean by $m / N$ , and taking expectations over |Π| with $\mathbb { E } | \Pi | = p N$ gives the factor p. □ □

The re-normalized estimator therefore carries the within-regime drift guarantee of the main DRFC-Seq theorem to partial participation with no bias, only a $1 / { \sqrt { p } }$ radius inflation that slows adoption gracefully, while the naive estimator reads a gap shrunk by p and stalls. The records that do flood are deduplicated by source exactly as before, so a dropped agent is simply absent from the window rather than double-counted, and the source-traceable scheme is unchanged. This is robustness to missed probe blocks and temporary disconnection at the estimator level, complementary to the lossy-gossip robustness already modeled by the main random-graph flooding assumption.

<table><tr><td>Closest prior line</td><td>Non-stat. unit</td><td></td><td>Dec. Regret scales with On</td><td> $S _ { \mathrm { l o c } } \gg S _ { \mathrm { d e c } }$ </td></tr><tr><td>Garivier-Moulines</td><td>opt-arm switches</td><td>no</td><td> $\#$  switches</td><td>resets per local change</td></tr><tr><td>Besbes et al.</td><td>variation  $V _ { T }$ </td><td>no</td><td> $V _ { T } ^ { 1 / 3 } T ^ { 2 / 3 }$ </td><td>local shift inflates  $V _ { T }$ </td></tr><tr><td>Cao; Besson et al.</td><td>change points</td><td>no</td><td> $\#$  changes</td><td>restarts each local change</td></tr><tr><td>Abbasi-Yadkori et al.</td><td>best-arm switches no</td><td></td><td># switches</td><td>single stream, no average</td></tr><tr><td>M.-Rubio; Chawla et al. none (stationary) yes</td><td></td><td></td><td>spec. gap,  $N , K$ </td><td>no temporal-drift model</td></tr><tr><td>Komiyama et al.</td><td>global changes</td><td>no</td><td> $\#$  changes</td><td>local counted as global</td></tr><tr><td>DRFC (ours)</td><td>decision  $S _ { \mathrm { d e c } }$ </td><td>yes</td><td> $S _ { \mathrm { d e c } }$  not  $S _ { \mathrm { l o c } }$ </td><td>adapts to  $S _ { \mathrm { d e c } }$  only</td></tr></table>

Table 5: Why the closest accepted lines cannot exploit heterogeneous cancellation. Each either counts every local change, so its adaptation term scales with $S _ { \mathrm { l o c } }$ rather than $S _ { \mathrm { d e c } }$ , or assumes stationary rewards; only DRFC certifies non-stationarity at the agent-average decision level and adapts to $S _ { \mathrm { d e c } } \ll S _ { \mathrm { l o c } }$ . The cancellation instance of the two-agent illustration (main paper) has $S _ { \mathrm { l o c } } = \Theta ( T / L )$ yet $S _ { \mathrm { d e c } } = 0$ , so every $S _ { \mathrm { l o c } }$ -scaling row pays $\Omega ( T / L )$ while DRFC pays only monitoring cost. (Mainpaper Related Work.)

Empirical degradation. Figure 15 sweeps the participation probability p from 1.0 down to 0.30 on the within-regime drift instance and on a genuine-switch instance $( N = 2 4 , K = 4 , \Delta = 0 . 1 0$ amplitude $b = 0 . 2 0$ , period-spanning window, 30 seeds), and confirms Proposition 4. On the drift instance the re-normalized windowed gap stays flat at 0.095 to 0.097, within noise of the true margin $\Delta = 0 . 1 0$ , across the whole participation range, while the naive gap shrinks linearly from 0.096 at full participation to 0.028 at $p = 0 . 3 0$ , exactly the $p \bar { G }$ bias the proposition predicts. DRFC-Seq holds a zero false-switch rate at every participation level under both estimators, so drift safety survives dropout, whereas the decision-level DecCUSUM false-alarms at rate one throughout. On the genuine-switch instance the re-normalized monitor adopts the switch in every run down to $p = 0 . 9 0$ , with the adopt latency growing from 4980 to 5557 rounds as the $1 / { \sqrt { p } }$ radius inflation slows certification, and still in two thirds of runs at $p = 0 . 7 5$ , stalling only once $p \leq 0 . 6$ drives the inflated radius above the margin, a safe failure that never mis-adopts. The naive estimator, reading a gap shrunk by $p ,$ already collapses to a zero adopt rate by $p = 0 . 7 5$ . Re-normalization therefore extends reliable adoption to far lower participation, and the only price of dropout is the gracefu $1 / { \sqrt { p } }$ latency growth the analysis predicts, not a bias or a false switch.

<table><tr><td>Scenario</td><td>DRFC-Seq DecCUSUM DL-GLR DL-ADR</td><td></td><td></td><td></td></tr><tr><td colspan="5">Genuine-switch adopt rate  $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } { = } 1$  , no drift)</td></tr><tr><td> $\Delta { = } 0 . 1 0$ </td><td>1.00</td><td>1.00</td><td>0.87</td><td>0.97</td></tr><tr><td> $\Delta { = } 0 . 2 0$ </td><td>1.00</td><td>1.00</td><td>0.93</td><td>0.93</td></tr><tr><td colspan="5">False-switch rate (drift present)</td></tr><tr><td> $\Delta { = } 0 . 1 0 .$  drift  $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } { = } 1 )$ </td><td>0.00</td><td>0.90</td><td>0.83</td><td>0.63</td></tr><tr><td> $\Delta { = } 0 . 2 0 ,$  drift  $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } { = } 1 )$ </td><td>0.00</td><td>0.03</td><td>0.63</td><td>0.20</td></tr><tr><td>drift only  $( b { = } 0 . 2 \bar { 0 } , S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } { = } 0 )$ </td><td>0.00</td><td>0.90</td><td>0.90</td><td>0.60</td></tr></table>

Table 6: Adaptivity / non-vacuity battery at the live operating point $( N = 2 4 , 2 r _ { n } \approx 0 . 0 8 5 < \Delta ;$ 30 seeds, horizon 24000, switch at 12000); DL-GLR is DL-GLR-klUCB, DL-ADR is DL-ADRbandit. Top block: genuine-switch adopt rate (fraction of runs adopting the correct new arm)—all four decision-level monitors are alive, including the two strong change detectors. Bottom block: false-switch rate—DRFC-Seq is the only one that stays at 0 under drift, while DecCUSUM, DL-GLR-klUCB, and even the gradual-drift-specialized DL-ADR-bandit all false-alarm, exactly the dichotomy of main Proposition 1. Detection latencies (rounds): DRFC-Seq ∼ 5000–6800, DL-GLR ∼ 2200–4200, DL-ADR ∼ 1500–2300, DecCUSUM ∼ 300—faster detection buys the false alarms. (Main-paper “Non-vacuity: DRFC-Seq does switch on genuine switches.”)

![](images/401ad898df91ee37c9585e73a124f7587b0a6f009b206c181fdfe0610290ca29.jpg)  
Figure 8: Separation stress test $( S _ { \mathrm { d e c } } = 0 _ { \mathrm { : } }$ , varying $S _ { \mathrm { l o c } } )$ . LocalReset regret grows linearly in $S _ { \mathrm { l o c } } ;$ DRFC and Oracle pay only monitoring cost. (Main-paper “Separation under decision-irrelevant drift.”)

![](images/ef926bb928444f935bf5e536569e1d54d04d0f1493b96257d16f975a9fdf1c9c.jpg)  
Figure 9: Within-regime drift on MovieLens-1M genre dynamics, semi-synthetic (real measured fluctuation, anchored margin) $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0 , \ : \Delta = 0 . 1 0$ , arms {Romance, Thriller, Comedy, Action}, $8 / 1 6$ real bins reverse). x-axis: gain g scaling the empirical fluctuation (dotted: real sign-flip onset). (a) Decision regret $R _ { T } ^ { \mathrm { d e c } }$ : the periodic-scan family blows up to ∼ 2× DRFC-Seq; the strong decisionlevel detectors DL-GLR-klUCB and DL-ADR-bandit are higher regret than DRFC-Seq under strong drift while DecCUSUM reaches marginally lower regret only by false-switching. (b) False-switch rate: DRFC-Seq (adopt-capable) stays at 0; all change-detection competitors—including DL-GLR-klUCB and $\mathrm { D L } { \mathrm { - A D R } }$ -bandit—climb to 1. $N = 2 4$ agents at a live operating point $( 2 r _ { n } < \Delta )$ ; 30 seeds.

<table><tr><td>Method</td><td> $R _ { T } ^ { \mathrm { d e c } }$ </td><td> $R _ { T }$ </td><td>(inst.) False-switch Switches</td><td></td></tr><tr><td>DRFC-Seq</td><td> ${ \bf 2 7 1 0 \pm 6 }$ </td><td>6962</td><td>0.00</td><td>0.0</td></tr><tr><td>DecCUSUM</td><td> $3 0 5 3 \pm 4 7 5$ </td><td>7305</td><td>1.00</td><td>4.7</td></tr><tr><td>AdaptivePeriodicTourn.</td><td> $3 0 7 5 \pm 7 8 6$ </td><td>7326</td><td>1.00</td><td>5.1</td></tr><tr><td>DL-GLR-klUCB</td><td> $3 4 5 9 \pm 7 3 6$ </td><td>7710</td><td>0.90</td><td>3.4</td></tr><tr><td>DL-ADR-bandit</td><td> $3 8 6 8 \pm 5 0 0$ </td><td>8120</td><td>0.90</td><td>4.1</td></tr><tr><td>Oracle (perfect comm.)</td><td> $3 8 5 1 \pm 2 6 5$ </td><td>8103</td><td>0.00</td><td>0.0</td></tr><tr><td>DRFC-KGraph</td><td> $4 3 3 5 \pm 2 8 1$ </td><td>8587</td><td>1.00</td><td>10.1</td></tr><tr><td>PeriodicTournament</td><td> $4 6 0 6 \pm 2 4 3$ </td><td>8857</td><td>1.00</td><td>9.6</td></tr></table>

Table 7: Raw, un-anchored MovieLens-1M within-regime drift $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0$ , raw time-average margin $\bar { \Delta } \approx 0 . 0 1 1 5 , \ 8 / 1 6$ bins reverse, N = 24, 30 seeds, horizon 12000). No constructed square wave and no anchored margin, the drift being the measured rating fluctuation. DRFC-Seq is the only deployable monitor that is both lowest-regret and never false-switches, while every fast decision-level change detector (DecCUSUM, DL-GLR-klUCB, DL-ADR-bandit, AdaptivePeriodicTournament) false-alarms, consistent with main Proposition 1. The conservative decentralized DRFC variants also false-switch on stale snapshots, and only the perfect-communication Oracle reference matches DRFC-Seq’s zero rate—DRFC-Seq attains it under realistic communication. The passive SW-UCB-Dec index $( 3 3 5 6 \pm 7 8$ , no adopt-event) is omitted from the false-switch comparison.

<table><tr><td>Cluster</td><td>Drama</td><td>Romance</td><td>Thriller</td><td>Comedy</td><td>Action</td></tr><tr><td>age25 occ17</td><td>0.680</td><td>0.627</td><td>0.582</td><td>0.592</td><td>0.548</td></tr><tr><td>age25 occ0</td><td>0.618</td><td>0.564</td><td>0.470</td><td>0.515</td><td>0.472</td></tr><tr><td>age35_occ17</td><td>0.537</td><td>0.480</td><td>0.461</td><td>0.449</td><td>0.452</td></tr><tr><td>Population avg</td><td>0.620</td><td>0.538</td><td>0.526</td><td>0.510</td><td>0.494</td></tr></table>

Table 8: MovieLens-1M cluster-by-genre breakdown (mean binarised rating, ≥ 4). Drama (bold) is the highest-rated genre in every cluster, with population-average margin 0.08 over the runnerup. Within each cluster, however, the ranking of the four non-best genres varies (see e.g. Comedy/Thriller/Action interchange across rows), producing high $S _ { \mathrm { l o c } }$ on the time-binned replay even though $S _ { \mathrm { d e c } } = 0$ . Three representative clusters and the population average are shown; Drama is the top genre in all 12 clusters in the released data table.

![](images/10f419454082bb5acbaf36648f81a214082f9e156418e9ab372a11e08375cf06.jpg)  
Figure 10: MovieLens-1M replay $( S _ { \mathrm { d e c } } = 0 , N = 1 2$ clusters, K = 5 genres). DRFC-Probe slope is ∼ 2.4× smaller than LocalReset.

![](images/6f1f5d8b0686cf9de09b2d09e2ab7da7296ea2164f043de99977e39097472c91.jpg)  
Figure 11: K-ary stress check. Probe-triggered DRFC trades regret against source-record count under tight regimes $( S _ { \mathrm { d e c } } \in \{ 1 , 2 , 4 \} )$ .

![](images/97f15885edead34518ab615de19bd95eeeb16df8ad0eb7d4006996db30a133be.jpg)  
Figure 12: Geometric scan-budget walk-up for Gap-Adaptive DRFC (labeled Doubling-DRFC in the plot). Mean $\mathrm { w a l k - u p } \leq 1 . 5$ levels across a 5× range of $\Delta ,$ matching oracle-tuned regret.

<table><tr><td>Method</td><td>MV slope</td><td>MV final</td><td>K-ary  $S _ { 4 }$ </td><td>K-ary f</td></tr><tr><td>DRFC</td><td>12.4</td><td>950</td><td>951</td><td>0.10</td></tr><tr><td>DRFC-Probe</td><td>8.6</td><td>818</td><td>881</td><td>0.35</td></tr><tr><td>SW-UCB-Dec</td><td>5.9</td><td>852</td><td>787</td><td>0.00</td></tr><tr><td>CUSUM-Reset</td><td>10.5</td><td>894</td><td>961</td><td>0.55</td></tr><tr><td>LocalReset</td><td>20.8</td><td>1009</td><td>1007</td><td>0.20</td></tr><tr><td>Oracle</td><td>12.7</td><td>979</td><td>925</td><td>0.15</td></tr></table>

Table 9: Headline numbers. MV slope: $\mathrm { r e g r e t - p e r } { - } S _ { \mathrm { l o c } }$ slope on MovieLens replay $( S _ { \mathrm { l o c } } { = } 3$ to 29, $S _ { \mathrm { d e c } } { = } 0 )$ . MV final: regret at $S _ { \mathrm { l o c } } { = } 2 9$ . K-ary $S _ { 4 } \colon$ regret at $S _ { \mathrm { d e c } } { = } 4$ on the K-ary stress check (horizon 1200). f: false-switch rate at $S _ { \mathrm { d e c } } { = } 4 .$ . All entries are read directly from the archived aggregate tables. DRFC-Probe’s MovieLens slope (8.6) is $\sim 2 . 4 \times$ smaller than the explicit local-reset baseline LocalReset (20.8), with lower absolute regret than LocalReset at every $S _ { \mathrm { l o c } } ;$ the passive SW-UCB index has a flatter slope but a higher regret floor and no adopt-event $( f { = } 0 )$ . On the K-ary check, conservative DRFC has the lowest false-switch rate among switching methods (0.10).

<table><tr><td>Detector/ /setting</td><td>genuine adopt</td><td>drift FSR</td></tr><tr><td>DL-GLR-klUCB (threshold sweep)</td><td></td><td></td></tr><tr><td>threshold 2.0</td><td>1.00</td><td>0.87</td></tr><tr><td>threshold 4.0</td><td>1.00</td><td>0.90</td></tr><tr><td>threshold 6.0 (paper)</td><td>0.83</td><td>0.90</td></tr><tr><td>threshold 9.0</td><td>0.60</td><td>0.83</td></tr><tr><td>threshold 14.0 threshold 20.0</td><td>0.67</td><td>0.50 0.27</td></tr><tr><td>DL-ADR-bandit (threshold-scale sweep)</td><td>0.50</td><td></td></tr><tr><td>scale 0.50</td><td></td><td></td></tr><tr><td>scale 0.75</td><td>1.00</td><td>0.63</td></tr><tr><td></td><td>1.00</td><td>0.60</td></tr><tr><td>scale 1.00 (paper)</td><td>1.00</td><td>0.63</td></tr><tr><td>scale 1.50</td><td>0.80</td><td>0.80</td></tr><tr><td>scale 2.50</td><td>0.30</td><td>0.17</td></tr><tr><td>scale 4.00</td><td>0.00</td><td>0.00</td></tr><tr><td>DRFC-Seq (no threshold,  $\scriptstyle \alpha = 0 . 1 0 )$ </td><td>1.00</td><td>0.00</td></tr></table>

Table 10: Decision-level baseline tuning sweep. “genuine adopt” is the genuine-switch adopt rate (higher is more switch-adaptive); “drift FSR” is the drift-only false-switch rate (lower is more driftrobust). Across the full sweep of either detector, no setting achieves both a high adopt rate and a low false-switch rate—suppressing the drift false alarms requires giving up switch-adaptivity, exactly the main Proposition 1 dichotomy. The paper operating points sit in the adaptive (baseline-favourable) regime. DRFC-Seq attains both simultaneously. Reproduced by the baseline-tuning sweep in the anonymized package, with archived per-seed outputs. 30 seeds.

<table><tr><td>Method</td><td>records/agent/round</td><td>total records</td></tr><tr><td>SW-UCB-Dec</td><td>0.50</td><td>7200</td></tr><tr><td>DecCUSUM</td><td>0.90</td><td>12957</td></tr><tr><td>CUSUM-Reset</td><td>0.87</td><td>12492</td></tr><tr><td>LocalReset</td><td>0.88</td><td>12740</td></tr><tr><td>DRFC-KGraph</td><td>1.13</td><td>16298</td></tr><tr><td>DRFC-Probe</td><td>1.36</td><td>19604</td></tr><tr><td>Oracle</td><td>1.30</td><td>18732</td></tr></table>

Table 11: Per-round source-record communication at the standard operating point $( N = 1 2 , K = 4$ $S _ { \mathrm { d e c } } = 2$ , edge probability 0.50, horizon 1200, 20 seeds). Records per agent per round is the total flooded source records divided by N and the horizon. The conservative DRFC certifier is within a factor 1.3 of the decision-level change detectors and the probe and Oracle variants spend more trafic for lower regret, so source-traceable certification is bandwidth-comparable to a change detector and tunable through M and h by Proposition 3. Read from the archived communication logs in the reproducibility package.

<table><tr><td>Knob</td><td>Controls</td><td>Default</td><td>If  $\Delta _ { \prime }$  /period unknown</td></tr><tr><td>M</td><td>monitor cost vs delay</td><td>120</td><td>comm. budget, not  $\Delta$ </td></tr><tr><td> $h$ </td><td>per-block radius</td><td>6/24</td><td>small constant</td></tr><tr><td> $H _ { \mathrm { s c a n } }$ </td><td>scan budget</td><td></td><td>scan calib. Gap-Adaptive DRFC (App. B)</td></tr><tr><td> $\delta _ { \mathrm { p } }$ </td><td>probe cadence</td><td>20</td><td> $\geq K h + D _ { G }$ </td></tr><tr><td> $W$ </td><td>drift-period span</td><td>120</td><td>Doubling-Window (Thm. H.2)</td></tr><tr><td>α</td><td>false-switch level</td><td>0.10</td><td>robust 20×</td></tr><tr><td> $C$ </td><td>radius constant</td><td>1.15</td><td>conservative as  $C \uparrow$ </td></tr></table>

Table 12: Parameter-tuning summary. The only margin-dependent inputs are the scan budget $H _ { \mathrm { s c a n } }$ and the window $W ,$ and each has a knowledge-free doubling replacement (Proposition $^ { 2 , }$ Theorem $7 )$ The remaining knobs are governed by the communication budget and the tolerable latency rather than by $\Delta$ or the drift period, and the confidence inputs $\alpha , C$ are robust or err conservatively (Tables 14, 19).

![](images/107b4c787e982c3e459943ad6839dd5f53e666ab5b06bdcc12e23ed85e3f10d6.jpg)  
Figure 13: Topology robustness beyond Erdős–Rényi $( S _ { \mathrm { d e c } } = 2 , N = 1 2 , K = 4 , 2 0$ seeds). Each base graph (ring+chord, small-world, geometric, scale-free, path) is made time-varying by per-round edge activation. (a) Group regret against the source-record reach rate. Across the four connected families the points collapse onto one curve, so regret is governed by the realized reach rate (the $D _ { G }$ proxy) and not by the topology family. (b) Post-switch detection delay against reach rate. The path graph (maximal diameter) never floods and sits at censored delay, the $D _ { G } \to N$ corner of the bound. Reproduced by the topology-robustness sweep.

![](images/c7d845bcabe1ff79831e51cfef52f96a030f8c9373b078c930a762646cc39430.jpg)

![](images/f8f89bef201d9d1aaa7b58a6a1da9bbb377f345fb8312b6cfb51767e72555fdd.jpg)

![](images/d90819145a23d32cd04ba3e3b395df61ae8c7e32169fec9c04574b5fdcf33a9d.jpg)  
Figure 14: Switching-regime stress at $S _ { \mathrm { d e c } } > 0 \ ( N = 1 2 , K = 4 , 2 0$ seeds), validating the bound’s three dependences. (a) Detection delay falls with the gap $\Delta$ (easier certification, the $\mathcal { C } _ { r } \sim ( K - 1 ) / \Delta$ branch) while regret rises with $\Delta$ (the monitoring branch, each wasted pull costing $\Delta )$ . (b) Detection delay collapses as connectivity rises and the reach rate climbs, the $N S _ { \mathrm { d e c } } D _ { G }$ delay term. (c) Group regret grows linearly in $S _ { \mathrm { d e c } }$ at fixed inter-switch spacing, the $N S _ { \mathrm { d e c } }$ switch term. Reproduced by the switching-regime stress sweep.

![](images/985d7c70363ad5f5cd6db72d58c57fbce079c20fbc4ede1d8a6165b4d8dbd9f7.jpg)

![](images/a0b9edb5255010f74f567c00b8e9d3712c070be5f98f975261a7c168bdc6dfff.jpg)

![](images/d41ece03a7a4efa1e6e7a9d6603106e3dd37f8b43a3fc2384e59c760d0bfa19a.jpg)  
Figure 15: Asynchrony and partial participation $( N = 2 4 , K = 4 , \Delta = 0 . 1 0$ , 30 seeds), sweeping the per-block participation probability $p .$ (a) On the drift instance $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0 )$ the re-normalized windowed gap stays at the true margin $\Delta$ across all p (unbiased), while the naive estimator shrinks to $p \Delta$ (biased), the empirical face of Proposition 4. (b) DRFC-Seq holds a zero false-switch rate under dropout while DecCUSUM false-alarms at rate one. (c) On a genuine-switch instance $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 1 )$ the re-normalized monitor keeps adopting down to about 75% participation with a gracefully growing latency, while the naive estimator stalls. Reproduced by the partial-participation sweep.

## I.7 Structured and persistent participation bias

Section assumes dropout is oblivious, independent of the rewards. The harder question is structured participation, where which agents drop out is correlated with their private arm preferences and the pattern persists across the horizon. In a heterogeneous network the equal-agent global mean of an arm is the average of per-agent means, so if participation correlates with those means the surviving subset is no longer a uniform sample and the re-normalized subset mean is biased. The headline is an exact, operating-point-independent characterization of that bias as a covariance identity. A finite-sample corollary then shows the margin absorbs the bias once it exceeds the confidence radius by a constant factor, a regime reached with a longer window or a larger gap, and a final part shows that knowing the propensities removes the bias outright.

Persistent-propensity model. Each agent i participates in every block independently with a fixed propensity $p _ { i } \in [ p _ { \operatorname* { m i n } } , p _ { \operatorname* { m a x } } ]$ , reward-independent given the block but correlated with the peragent means $\mu _ { i , a }$ . Write $\bar { p } = N ^ { - 1 } \sum _ { i } p _ { i }$ . The re-normalized subset-mean estimator of arm a over a window of W blocks pools the surviving agent-block means, $\begin{array} { r } { \hat { \mu } _ { a } = \left( \sum _ { w , i } \mathbf { 1 } _ { i , w } \hat { g } _ { i , a , w } \right) / \left( \sum _ { w , i } \mathbf { 1 } _ { i , w } \right) } \end{array}$ where $\mathbf { 1 } _ { i , w }$ indicates that agent i joined block w.

Proposition 5 (Subset-mean bias under structured participation). Under the persistent-propensity model, as the window length $W \to \infty$ the re-normalized subset mean converges almost surely to the propensity-weighted mean, and its bias relative to the all-agent equal-agent mean $\bar { \mu } _ { a } = N ^ { - 1 } \sum _ { i } \mu _ { i , a }$ is the exact identity

$$
\mathrm { \Delta b i a s } _ { a } = \frac { \sum _ { i } p _ { i } \mu _ { i , a } } { \sum _ { i } p _ { i } } - \bar { \mu } _ { a } = \frac { \mathrm { C o v } _ { i } ( p , \mu _ { \cdot , a } ) } { \bar { p } } ,
$$

where Cov<sub>i</sub> and p¯ are the empirical covariance and mean over the N agents. Consequently

$$
\left| { \mathrm { b i a s } _ { a } } \right| ~ \leq ~ \frac { \left( p _ { \operatorname* { m a x } } - p _ { \operatorname* { m i n } } \right) \left( \operatorname* { m a x } _ { i } \mu _ { i , a } - \operatorname* { m i n } _ { i } \mu _ { i , a } \right) } { 4 \bar { p } } ,
$$

and the contested-pair gap estimate is biased by $B _ { a , b } = [ \mathrm { C o v } _ { i } ( p , \mu _ { \cdot , a } ) - \mathrm { C o v } _ { i } ( p , \mu _ { \cdot , b } ) ] / \bar { p }$ . DRFC-Seq’s adopt rule preserves the true decision ordering, with neither a false switch nor a missed genuine switch, whenever $\Delta > 4 r _ { n }$ and

$$
| B _ { a , b } | < \Delta - 4 r _ { n } ,
$$

so structured participation bias is absorbed up to a margin-dependent boundary and only beyond it can flip a decision. If the propensities are known, the self-normalized inverse-propensity estimator $\begin{array} { r } { \hat { \mu } _ { a } ^ { \mathrm { I P W } } = \left( \sum _ { i \in \Pi } \hat { g } _ { i , a } / p _ { i } \right) / \left( \sum _ { i \in \Pi } 1 / p _ { i } \right) } \end{array}$ is asymptotically unbiased for $\bar { \mu } _ { a }$ , at the price of a variance inflation by the Kish efective-sample-size factor folded into $r _ { n }$

Proof. Participation is drawn independently across the W blocks, and the within-block means $\hat { g } _ { i , a , w }$ are bounded in [0, 1], so by the law of large numbers the block-averaged numerator $\begin{array} { r } { W ^ { - 1 } \sum _ { w , i } { \bf 1 } _ { i , w } \hat { g } _ { i , a , w } \ \to \ \sum _ { i } p _ { i } \mu _ { i , a } } \end{array}$ and the block-averaged denominator $\begin{array} { r } { W ^ { - 1 } \sum _ { w , i } { \bf 1 } _ { i , w } \ \to \ \sum _ { i } p _ { i } } \end{array}$ almost surely, hence $\begin{array} { r } { \hat { \mu } _ { a }  ( \sum _ { i } p _ { i } \mu _ { i , a } ) / ( \sum _ { i } p _ { i } ) } \end{array}$ . Clearing denominators,

$$
\frac { \sum _ { i } p _ { i } \mu _ { i , a } } { \sum _ { i } p _ { i } } - \bar { \mu } _ { a } = \frac { N \sum _ { i } p _ { i } \mu _ { i , a } - ( \sum _ { i } p _ { i } ) ( \sum _ { i } \mu _ { i , a } ) } { N \sum _ { i } p _ { i } } = \frac { \mathrm { C o v } _ { i } ( p , \mu _ { \cdot , a } ) } { \bar { p } } ,
$$

with $\begin{array} { r } { \mathrm { C o v } _ { i } ( p , \mu . , a ) \ = \ N ^ { - 1 } \sum _ { i } p _ { i } \mu _ { i , a } \ - \ \bar { p } \bar { \mu } _ { a } } \end{array}$ , which is the identity. The range bound follows from $\begin{array} { r } { | \mathrm { C o v } _ { i } ( p , \mu _ { \cdot , a } ) | \leq \sqrt { \mathrm { V a r } _ { i } ( p ) \mathrm { V a r } _ { i } ( \mu _ { \cdot , a } ) } \leq \frac { 1 } { 4 } ( p _ { \operatorname* { m a x } } - p _ { \operatorname* { m i n } } ) ( \operatorname* { m a x } _ { i } \mu _ { i , a } - \operatorname* { m i n } _ { i } \mu _ { i , a } ) } \end{array}$ by Cauchy–Schwarz and Popoviciu’s inequality Var $\leq ( \mathrm { r a n g e } / 2 ) ^ { 2 }$ . For the decision boundary, on the anytime-valid event of probability $1 - \alpha$ each arm’s windowed estimate lies within $r _ { n }$ of its biased limit, so the empirical gap lies within $2 r _ { n }$ of $\bar { G } _ { a , b } + B _ { a , b }$ , with $\bar { G } _ { a , b }$ the all-agent gap of magnitude $\Delta .$ . The adopt rule certifies the pair only when this empirical gap, minus a further $2 r _ { n } ,$ is positive. When a genuine switch makes $\bar { G } _ { a , b } = + \Delta$ the worst-case empirical gap is $\Delta + B _ { a , b } - 2 r _ { n }$ , so adoption is guaranteed once $\Delta + B _ { a , b } - 2 r _ { n } > 2 r _ { n } ,$ , that is $\Delta + B _ { a , b } > 4 r _ { n }$ . When the candidate is truly best the certified lower bound never exceeds the biased limit $- \Delta + B _ { a , b } ,$ so a false switch needs $B _ { a , b } > \Delta$ . The symmetric suficient condition $\left| B _ { a , b } \right| < \Delta - 4 r _ { n }$ , which presumes $\Delta > 4 r _ { n }$ , rules out both a missed genuine switch and a false switch. For IPW the same argument on the reweighted sums gives the limit $\begin{array} { r } { \big ( \sum _ { i } p _ { i } \mu _ { i , a } / p _ { i } \big ) / ( \sum _ { i } p _ { i } / p _ { i } ) = N ^ { - 1 } \sum _ { i } \mu _ { i , a } = \bar { \mu } _ { a } } \end{array}$ , and the reweighted window’s efective sample size is the Kish quantity $\textstyle ( \sum _ { i \in \Pi } 1 / p _ { i } ) ^ { 2 } / \sum _ { i \in \Pi } 1 / p _ { i } ^ { 2 }$ □ □

Empirical boundary and the IPW repair. Figure 16 is a controlled diagnostic that sweeps the structural skew s of a persistent preference-correlated dropout on the same drift and genuineswitch instances $( N = 2 4$ , K = 4, $\Delta = 0 . 1 0$ , 30 seeds, a two-group zero-sum tilt on the contested arm). The measured bias is the emergent diference between the subset and IPW windowed gaps, not the closed form. On the drift instance the subset-mean windowed gap falls almost exactly along the covariance prediction of Proposition 5, from 0.090 at $s = 0$ through zero near $s = 0 . 5 \mathrm { ~ t o ~ } \mathrm { - 0 . 0 9 7 }$ at $s = 0 . 9$ , the measured bias matching $\mathrm { C o v } ( p , \mu ) / \bar { p }$ to within 0.02 (for instance 0.105 against the predicted 0.111 at $s = 0 . 6 )$ , while the IPW estimator holds the gap flat at the margin across all skews. The decision tracks the boundary, the subset monitor holding a zero false-switch rate well past the guaranteed-safe point and breaking only once the bias clears the margin by the radius headroom, to 0.17 at $s = 0 . 9$ and 1 at $s = 1$ , whereas IPW stays at zero throughout. This drift instance is the clean test of the proposition, since its false-switch boundary depends on the bias sign alone, not on the radius headroom. On the genuine-switch instance the headroom is already thin at this partial-participation operating point, where $4 r _ { n }$ is comparable to $\Delta .$ , so adoption is radius limited even at $s = 0$ (subset 0.50, IPW 0.60), consistent with the $\Delta > 4 r _ { n }$ premise not being comfortably met. The structure then suppresses the new best arm, so the biased subset monitor’s adopt rate collapses to zero by $s = 0 . 1 5$ , while IPW, which removes the bias but pays a variance inflation, keeps adopting at 0.43 and 0.23 for $s = 0 . 1 5$ and 0.3 before its inflated radius censors adoption. To show the decision-safety corollary is not vacuous, we rerun the same instances at a wider-margin point where $\Delta > 4 r _ { n }$ holds $( \Delta = 0 . 2 5$ at the same window). Genuine-switch adoption then returns to 1 at $s = 0 ,$ , and the biased subset monitor adopts in every run up to $s = 0 . 6$ before the bias clears the boundary and adoption falls to zero by $s = 0 . 7 5$ , the predicted transition, while the IPW monitor adopts in every run through $s = 0 . 9$ and falls only at $s = 1$ . The wider margin also holds the drift-instance false-switch rate at zero throughout, the margin now absorbing the full swept bias. Structured participation bias is therefore real but bounded by the margin, and its mean is removable by inverse-propensity weighting when the propensities are known or estimable from participation logs, at a variance cost that limits finite-sample adoption power.

## I.8 Scaling to large action spaces

A referee asked for the cost of block-based certification at large K, where each DRFC scan pulls Kh samples per agent per block. We sweep the arm count $K \in \{ 4 , 8 , 1 6 , 3 2 \}$ at $N = 1 2$ agents, with two decision switches $( S _ { \mathrm { d e c } } = 2 )$ , moderate decision-irrelevant local drift $( S _ { \mathrm { l o c } } = 2 6 )$ , decision gap $\Delta = 0 . 2 0$ , and edge probability 0.5. The monitor period is held fixed at $M = 1 5 0 0$ so that a full K-arm scan, which costs Kh pulls per agent and ranges from 32 to 256 at $h = 8$ , completes within one period even at $K = 3 2$ , using the disclosed practical radius constant $C = 1 . 1 5$ . Each cell averages 12 seeds over horizon 15000 with full source-record reach.

![](images/1d673e18d82f31330fe25c9698d383e0126516043dffc61e5edef3dfbe9f285d.jpg)

![](images/7ed55e0588f2e14e29a6cba26359b901ae369a38897e1fcf27855f7963dd3246.jpg)

![](images/066808541b4f642d68e50bfdb5224a10682a8abfd8da4286cb42dcc50bc88d8b.jpg)  
Figure 16: Structured and persistent participation bias $( N = 2 4 $ 1 $K = 4 .$ ， $\Delta = 0 . 1 0 , \ 3 0$ seeds), sweeping the structural skew s of a persistent preference-correlated dropout. (a) On the drift instance the subset-mean windowed gap follows the covariance prediction of Proposition 5 (dotted) and crosses zero, while IPW with known propensities stays at the margin ∆. The $s = 1$ subset point is post-breakdown, measured against the switched candidate. (b) The subset monitor holds falseswitch rate 0 through the guaranteed-safe boundary (vertical line, bias = ∆) and breaks only past it, once the bias clears the margin by the radius headroom, while IPW stays at 0. (c) On a genuineswitch instance IPW preserves adoption to lower skew than the biased subset mean. Reproduced by the structured-participation sweep.

Figure 17 shows two facts. First, DRFC regret tracks the centralized oracle at every arm count, from 4336 against the oracle 4218 at $K = 4$ to 9896 against 9605 at $K = 3 2$ , a graceful growth of about 2.3 times across an eightfold larger action space, and it stays well below every reactive baseline, the margin widening with K, so that at $K = 1 6 \mathrm { D R F C }$ reaches 6612 against CUSUM-Reset 31171 and SW-UCB-Dec 31482. This growth is the governed Kh certification cost the block scan pays for a larger action space, not a failure mode. Second, DRFC per-agent source-record trafic stays nearly flat in $K$ , from 0.145 to 0.193 records per agent per round, because source-traceable gossip deduplicates by source and there are only N sources regardless of $K .$ . The index-sharing SW-UCB-Dec instead broadcasts a K-vector each cycle, so its per-agent trafic grows linearly in $K$ , through 0.333, 0.667, 1.333, and 2.667. The realized DRFC trafic sits far below the per-agent $\widetilde { O } ( K ( h + D _ { G } ) / M )$ worst case of Section , since steady-state source dedup leaves it N-dominated rather than K-dominated, so large-action-space communication is governed by the number of sources and the monitor period rather than by the action-space width.

Subsampled blocks for very large K. The Kh per-block pull cost is the one quantity that grows with the action space. A subsampled active-set block that samples a uniform size-r subset of the active arms each block replaces Kh by rh per block at the price of about $K / r$ more blocks to cover every arm, leaving the elimination guarantee intact because each arm is still certified once it has been covered enough times. This keeps the per-block budget bounded for very large K while the decision-level certification is unchanged. A companion agent-count sweep confirms that the per-agent trafic grows with the network only through the flooding rounds, the $D _ { G }$ dependence of Section , with the per-agent regret order-constant.

![](images/bcf4e4ca3d780e8fbae41f90dde7008bc8c1a71855ffe52ed7d916178c1bfc8b.jpg)

![](images/80bee6fd8f59016cd4c2cb2de3441ca4830f50cec2cc67fd2c9211bdf7aa57d6.jpg)  
Figure 17: Scaling to large action spaces $( N = 1 2 , S _ { \mathrm { d e c } } = 2 , \Delta = 0 . 2 0$ , 12 seeds, monitor period fixed at $M \ = \ 1 5 0 0 )$ . (a) Group regret against the arm count K. DRFC tracks the centralized oracle at every K and stays far below the reactive and decision-level change detectors, the margin widening with K. (b) Per-agent source-record trafic against K. DRFC stays flat near 0.17 because source-traceable gossip deduplicates by source, while the index-sharing SW-UCB-Dec broadcasts a K-vector and grows linearly in K. Reproduced by the large-action-space sweep.

## J. Proof of the Personalized Benchmark Theorem

This section proves the personalized-benchmark theorem for DRFC-Pers, a supplement-only result distinct from the four numbered main-paper theorems. The main paper refers to the personalized benchmark only in its limitations. The argument is a direct per-cluster instantiation of the global proof (Sections A–B of this supplement) plus a union bound over clusters; it does not introduce new concentration machinery.

Setup. Recall the partition $\{ I _ { c } \} _ { c = 1 } ^ { G }$ with $\left| I _ { c } \right|$ agents in cluster c, the cluster-personalized best arm $a _ { t } ^ { \star , c } =$ arg max<sub>a</sub> $\textstyle | I _ { c } | ^ { - 1 } \sum _ { i \in I _ { c } } \mu _ { i , a , t }$ , and the personalized regret $\begin{array} { r } { R _ { T } ^ { \mathrm { p e r s } } = \sum _ { c } \sum _ { i \in I _ { c } } \sum _ { t } ( \mu _ { i , a _ { t } ^ { \star , c } , t } - } \end{array}$ $\mu _ { i , A _ { i , t } , t } )$ . DRFC-Pers runs G logically independent active-set scans, one per cluster; cluster c’s scan uses only the agents in $I _ { c }$ and the cluster-mean $\bar { \mu } _ { a } ^ { ( c ) , t }$ as the elimination object. Define per-cluster decision-switch times $\rho _ { r } ^ { \left( c \right) }$ and per-cluster gap $\begin{array} { r } { \Delta _ { a } ^ { ( r , c ) } = \operatorname* { i n f } _ { t \in [ \rho _ { r } ^ { ( c ) } , \rho _ { r + 1 } ^ { ( c ) } ) } ( \bar { \mu } _ { a _ { r } ^ { \star , c } , t } - \bar { \mu } _ { a , t } ) } \end{array}$ in direct analogy to the main sign-stable decision-regime definition.

Per-cluster sign preservation. The block-level estimator inside cluster c is $\hat { G } _ { a , b } ^ { ( c ) } ( \mathcal { W } ) ~ =$ $\begin{array} { r } { | I _ { c } | ^ { - 1 } \sum _ { i \in I _ { c } } \hat { g } _ { i , a , b } ( \mathcal { W } ) } \end{array}$ , a direct restriction of the global object to $I _ { c }$ . Equation (3)-style decomposition (reward noise + schedule randomization) goes through with N replaced by $\left| I _ { c } \right|$ throughout, so the radius is $\beta ^ { ( c ) } ( \mathcal { W } ) = C \sqrt { \log ( K ^ { 2 } T ^ { 3 } / \delta _ { c } ) / ( | I _ { c } | h | \mathcal { W } | ) }$ with $\delta _ { c } = \delta / ( 3 G )$ (the budget per cluster from the union bound below). The sign-preservation lemma (Lemma 5) and the pairwise-correctness lemma (Lemma 6) hold verbatim with $\Delta _ { a } ^ { ( r ) }$ replaced by $\Delta _ { a } ^ { ( r , c ) }$ and the global mean replaced by the cluster mean.

Lemma 11 (Per-cluster sign preservation). $I f \mathcal { W }$ is contained in cluster c’s decision regime r, then for every $b \neq a _ { r } ^ { \star , c } , { \bar { G } } _ { a _ { r } ^ { \star , c } , b } ^ { ( c ) } ( \mathcal { W } ) \geq \Delta _ { b } ^ { ( r , c ) }$

Proof. Fix $\mathcal { W } \subseteq [ \rho _ { r } ^ { ( c ) } , \rho _ { r + 1 } ^ { ( c ) } )$ and $b \neq a _ { r } ^ { \star , c }$

$$
\begin{array} { r l } & { \bar { G } _ { a _ { r } ^ { \epsilon , c } , b } ^ { ( c ) } ( \mathcal { W } ) = \displaystyle \frac { 1 } { B } \sum _ { \ell \in \mathcal { W } } \frac { 1 } { q \ell } \sum _ { t \in \mathcal { U } _ { \ell } } \bigl ( \bar { \mu } _ { a _ { r } ^ { \star , c } , t } - \bar { \mu } _ { b , t } \bigr ) } \\ & { \qquad \stackrel { \mathrm { ( ) } } { \geq } \displaystyle \frac { 1 } { B } \sum _ { \ell \in \mathcal { W } } \frac { 1 } { q \ell } \sum _ { t \in \mathcal { U } _ { \ell } } \Delta _ { b } ^ { ( r , c ) } } \\ & { \qquad = \Delta _ { b } ^ { ( r , c ) } , } \end{array}
$$

where $\textcircled{1}$ applies the cluster-mean margin $\bar { \mu } _ { a _ { r } ^ { \star , c } , t } - \bar { \mu } _ { b , t } \geq \Delta _ { b } ^ { ( r , c ) }$ at every $t \in [ \rho _ { r } ^ { ( c ) } , \rho _ { r + 1 } ^ { ( c ) } )$ and the closing equality uses the convex weights $\begin{array} { r } { \frac { 1 } { B } \cdot \frac { 1 } { q _ { \ell } } \geq 0 } \end{array}$ summing to 1. □

Per-cluster active-set scan certification. By the same argument as Lemma 7 but with N replaced by $\left| I _ { c } \right|$ , the per-cluster scan in regime r certifies $a _ { r } ^ { \star , c }$ once its radius is below $\Delta _ { a } ^ { ( r , c ) } / 4$ requiring at most $B _ { a } ^ { ( r , c ) } = O ( \log ( K ^ { 2 } T ^ { 3 } / \delta _ { c } ) / ( | I _ { c } | h ( \Delta _ { a } ^ { ( r , c ) } ) ^ { 2 } ) )$ blocks per active arm. The one-scan group regret is $\begin{array} { r } { \mathcal { C } _ { r } ^ { ( c ) } = \sum _ { a \neq a _ { r } ^ { \star , c } } \Gamma _ { a } ^ { ( r , c ) } / ( \Delta _ { a } ^ { ( r , c ) } ) ^ { 2 } } \end{array}$

Per-cluster dynamic regret. Applying the five-term decomposition of Theorem 5 per cluster gives

$$
\begin{array}{c} R _ { T } ^ { ( c ) } \leq \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } ^ { ( c ) } } ( 1 + \frac { L _ { r } ^ { ( c ) } } { M } ) \mathcal { C } _ { r } ^ { ( c ) }  \\ { + | I _ { c } | S _ { \mathrm { d e c } } ^ { ( c ) } \big ( M + H _ { \mathrm { s c a n } } + D _ { G } ( \delta _ { c } / 3 ) \big ) . } \end{array}
$$

The boundary-crossing scan accounting and synchronization term carry over verbatim with $N $ $\left| I _ { c } \right|$

Union bound across clusters. Reserve the concentration budget $\delta / 3$ for the per-cluster concentration events, the flooding budget $\delta / 3$ for the propagation events, and the synchronization budget $\delta / 3$ for the per-cluster epoch-tie-breaking events—exactly as in Lemma 4. Each cluster consumes $1 / G$ of each budget; the per-cluster efective confidence is $\delta _ { c } = \delta / ( 3 G )$ . The $\log ( K ^ { 2 } T ^ { 3 } / \delta _ { c } )$ factor inside $\mathcal { C } _ { r } ^ { ( c ) }$ absorbs the extra log G inside ${ \widetilde { O } } .$

Summing the per-cluster regrets and applying linearity of expectation gives

$$
R _ { T } ^ { \mathrm { p e r s } } = \widetilde { \cal O } ( \sum _ { c = 1 } ^ { G } [ \sum _ { r = 0 } ^ { S _ { \mathrm { d e c } } ^ { ( c ) } } 1 + \frac { L _ { r } ^ { ( c ) } } { M } ) \mathcal { C } _ { r } ^ { ( c ) } + | I _ { c } | S _ { \mathrm { d e c } } ^ { ( c ) } \big ( M + H _ { \mathrm { s c a n } } + D _ { G } ( \delta / ( 3 G ) ) \big ) ] ) ,
$$

which is the per-cluster personalized-benchmark bound for DRFC-Pers. The high-probability statement holds with probability at least $1 - \delta$ by the union bound. □

Per-cluster separation. The main class-relative separation theorem also lifts per-cluster: on an instance family where cluster c has $S _ { \mathrm { l o c } } ^ { ( c ) } = \Theta ( T / L )$ but $S _ { \mathrm { d e c } } ^ { ( c ) } = 0$ , any $( \alpha , \gamma , \eta , \tau )$ -reactive protoco triggered by per-cluster local jumps incurs $\Omega ( ( 1 - \eta ) S _ { \mathrm { l o c } } ^ { ( c ) } c _ { \mathrm { r e s e t } } )$ reset regret within cluster c, while DRFC-Pers pays no $S _ { \mathrm { l o c } } ^ { ( c ) }$ adaptation term. The construction reuses the sign-flip family of supplementary Section E restricted to a single cluster.

When the bound is tight. For each cluster, the per-cluster bound is tight up to the same log factors as the global bound (matched by the homogeneous-family certification lower bound applied per cluster). The G-fold parallelism cost shows up as a factor G in the additive log log G inside Oe from the union bound, but no multiplicative G on the leading $\mathcal { C } _ { r } ^ { ( c ) }$ term.

## K. Reproducibility Checklist

All experiments are implemented in Python (standard library only for the simulators; the figure layer uses matplotlib and numpy). Each run is deterministic given its recorded seed and frozen experiment configuration, and writes tabular outputs consumed by the figure-generation routines. The anonymized code-and-data archive submitted as supplementary material contains the reproduction instructions, per-seed logs, aggregate tables, and significance outputs needed to reproduce every figure, table, and statistical test; the archive is released publicly upon publication.

Environment. CPython 3.13 with numpy 2.2 and matplotlib 3.10 for the figure layer; the simulators use the standard library only. Experiments run on a single commodity x86-64 CPU core under Windows 11, with under 2 GB RAM and no GPU or external services. The MovieLens-1M table used by the real-data drift experiment (∼6 MB) is auto-downloaded and cached on first use; the cache is not included in the supplementary archive.

Mapping experiments to results. The reproducibility package maps each reported artifact to a documented experiment entry:

• Within-regime drift phase transition (main DRFC-Seq theorem, synthetic square-wave instance): simulator outputs, aggregate tables, and rendered figures for the synthetic drift panel.

• Real-data within-regime drift (MovieLens-1M, Drama-excluded genre set with 8/16 signchanging bins): the corresponding replay outputs and rendered real-data drift figures.

• Decision-switch, ablation, gossip, graph-probability, user-cluster, and doubling sweeps: one documented experiment entry per sweep, each paired with its aggregate table and rendered figure.

• Headline significance tests of Table 13, the one-sided Mann-Whitney U on decision regret and Fisher exact on the per-seed false-switch event with Holm correction: the analysis reuses the exact figure pipeline and re-aggregates its per-seed values against the archived aggregate tables as a self-check.

Statistics. Every reported number is a mean over independent runs with distinct seeds: 20 runs for the K-ary, MovieLens-replay, user-cluster, and doubling sweeps (the frozen configuration default), and 30 runs for the within-regime-drift sweeps. Shaded bands in figures are 95% confidence intervals <sub>of the mean (1.96 σ/</sub>√<sub>runs). Table ± ranges are one standard deviation across runs. The false-switch</sub> rate is the per-seed event rate, the fraction of seeds in which the monitor adopts a non-time-averagebest arm at least once inside a no-decision-switch regime. This is the quantity the per-seed Fisher test below operates on, and it is the definition used in every false-switch table and figure caption.

<table><tr><td>Data Baseline</td><td></td><td>regret ratio MWU Holm p FSR (DRFC vs base) Fisher Holm p</td><td></td></tr><tr><td colspan="4">Synthetic square-wave drift,  $b \in \{ 0 . 1 5 , 0 . 2 0 , 0 . 2 5 \}$  (post-margin)</td></tr><tr><td> $\mathrm { D e c C U S U M }$ </td><td> $1 . 4 1 { - } 2 . 0 1 $   $\leq 2 . 6 \times 1 0 ^ { - 1 0 }$ </td><td> $\mathrm { 0 ~ v s ~ 0 . 1 7  – 0 . 8 3 }$ </td><td> $\leq 2 . 6 \times 1 0 ^ { - 2 }$ </td></tr><tr><td> $\mathrm { D L \mathrm { - G L R \mathrm { - k l U C B } } }$ </td><td> $1 . 7 6 { - 2 . 1 9 }$   $\leq 2 . 6 \times 1 0 ^ { - 1 0 }$ </td><td> $\mathrm { 0 ~ v s ~ 0 . 5 7  – 0 . 7 3 }$ </td><td> $\leq 1 . 5 \times 1 0 ^ { - 6 }$ </td></tr><tr><td> $\mathrm { D L - A D R - b a n d i t }$ </td><td>1.60-1.86  $\leq 2 . 6 \times 1 0 ^ { - 1 0 }$ </td><td> $\mathrm { 0 \ v s \ 0 . 4 3  – 0 . 5 3 }$ </td><td> $\leq 4 . 6 \times 1 0 ^ { - 5 }$ </td></tr><tr><td colspan="4">Real MovieLens drift, g ∈ {1.5, 2.0, 2.5, 3.0} (post-onset)</td></tr><tr><td>DecCUSUM</td><td>0.70-0.95 n.s.</td><td> $0 ~ \mathrm { { v s } ~ 0 . 1 7  – 1 . 0 0 }$ </td><td> $\leq 1 . 3 \times 1 0 ^ { - 1 }$ </td></tr><tr><td> $\mathrm { D L \mathrm { - G L R \mathrm { - k l U C B } } }$ </td><td>1.09-1.42  $\leq 3 . 1 \times 1 0 ^ { - 8 }$ </td><td> $0 ~ \mathrm { v s } ~ 0 . 4 0 { - } 1 . 0 0$ </td><td> $\leq 3 . 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>DL-ADR-bandit</td><td>1.14-1.38  $\leq 2 . 6 \times 1 0 ^ { - 1 0 }$ </td><td> $0 ~ \mathrm { { v s } ~ 0 . 5 7  – 1 . 0 0 }$ </td><td> $\leq 2 . 2 \times 1 0 ^ { - 6 }$ </td></tr></table>

Table 13: Significance of the headline drift separation, 30 seeds per cell, one-sided Mann-Whitney U on decision regret and Fisher’s exact test on the per-seed false-switch event, with Holm-Bonferroni family-wise correction within each dataset. Regret ratio is median base regret over median DRFC-Seq regret, so a value above one means DRFC-Seq is lower. Each cell reports the range over the swept amplitudes and the most conservative Holm-adjusted p-value in that range. On synthetic drift every separation is significant. On real MovieLens drift the false-switch separation is significant against all three detectors at $g \geq 2 . 0$ (down to $p \leq 2 \times 1 0 ^ { - 1 6 } )$ and marginal only against DecCUSUM at the onset gain $g \ = \ 1 . 5$ . DecCUSUM reaches marginally lower real-data regret (ratio below one) by harvesting decision-irrelevant reward through false switches, so its regret comparison is not significant while its false-switch rate climbs to one. Re-aggregated per-seed values match the archived aggregate tables.

Statistical significance of the headline separation. The headline drift separation is backed by a significance analysis over the same 30 seeds, not a visual gap. Each method draws an independent policy seed per run, so the 30 per-method values form an independent sample, and we test the two headline quantities directly. Decision regret is compared with a one-sided Mann-Whitney U test and the per-seed false-switch event with Fisher’s exact test, both against each adopt-capable decision-level detector. Holm-Bonferroni controls the family-wise error across all comparisons within each dataset, and a 10,000-resample bootstrap interval for the median regret diference accompanies every regret test. A self-check re-aggregates the per-seed values used here and matches the archived aggregate tables to within $1 0 ^ { - 3 }$ , so these are provably the same runs that underlie the synthetic and MovieLens drift phase-transition figures. Table 13 reports, for each baseline, the range over the post-margin amplitudes together with the most conservative Holm-adjusted p-value in that range.

On the synthetic instance every separation is significant. Past the margin DRFC-Seq attains 1.4 to 2.2 times lower median decision regret than each detector at Holm-adjusted p below $3 \times 1 0 ^ { - 1 0 }$ with a zero false-switch rate against rates of 0.17 to 0.83 at Holm-adjusted Fisher p below 0.03, and the bootstrap median-diference interval excludes zero in every case. On real MovieLens drift the false-switch separation is the strong result, with DRFC-Seq holding rate zero while each detector climbs toward one at Holm-adjusted p down to $2 \times 1 0 ^ { - 1 6 }$ . The regret picture is reported faithfully rather than overclaimed. DRFC-Seq has significantly lower regret than the two restart detectors DL-GLR-klUCB and DL-ADR-bandit, while DecCUSUM reaches a marginally lower median regret by harvesting decision-irrelevant reward through false switches, so the one-sided regret test against DecCUSUM is not significant on real data. This is the honest content of the real-data panel, where the decisive quantity is the false-switch rate.

Hyperparameters. All hyperparameters are the frozen defaults in the recorded experiment configurations; runtime overrides are limited to the number of runs, horizon, and seed. The drift experiments fix the exploitable decision margin $\Delta \ : = \ : 0 . 1 0$ and sweep the drift gain over {0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0} with sliding-window confidence level $\alpha = 0 . 1 0$

DRFC-Seq live operating point (non-vacuity). For the within-regime-drift experiments DRFC-Seq runs with $N = 2 4$ agents and a period-spanning sliding window of 120 probe-blocks (each 6 balanced samples/arm, probe gap 20), so the per-arm window count is $n = 7 2 0$ and the stitched anytime radius is $2 r _ { n } \approx 0 . 0 8 5$ , strictly below the headline decision margin $\Delta = 0 . 1 0$ . This is deliberate: a monitor whose radius exceeds the margin can never trigger, so a zero false-switch rate would be vacuous. At this live point the monitor is capable of switching, which we verify with an adaptivity battery (genuine $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } { = } 1$ switch at the horizon midpoint, with and without concurrent drift): DRFC-Seq adopts a genuine switch in every run at false-switch rate $0 ,$ at both the headline margin $\Delta = 0 . 1 0$ and a separated margin $\Delta = 0 . 2 0$ and with or without concurrent drift, and never adopts under a drift-only instance $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } { = } 0 , b = 0 . 2 0 )$ . The adaptivity battery uses horizon 24000, so the ∼ 5000–6800-round detection latency is measured rather than censored, and is summarized in the adaptivity table of Section I. The window (≈ 6240 rounds, 120 probe-blocks of 52 rounds) spans ≈ 12.5 drift periods (period 500 rounds). The MovieLens-1M clipping ([0.05, 0.95]) is verified to preserve the time-average-best arm with strictly positive margin at every gain by an archived preprocessing check.

Robustness to window and confidence misspecification. The period-spanning window is not a hand-tuned requirement that presumes oracle knowledge of the drift period. Holding the true period fixed at $P = 5 0 0$ rounds and the post-margin amplitude at $b = 0 . 2 0$ , we sweep the slidingwindow length over a 16× range, from 1.0 to 16.6 drift periods, and the confidence level α over a 20× range (Table 14). DRFC-Seq holds a zero false-switch rate at every setting, with decision regret flat to within 0.1%. At the fixed agent count N = 24 part of this safety is conservative inertia. Shortening the window lowers the per-arm count $n = W h$ , which inflates the anytime radius $r _ { n } \propto 1 / { \sqrt { n } } ,$ , so the radius is live $( 2 r _ { n } < \Delta )$ only for windows above about nine periods and a shorter window is also too conservative to trigger.

To separate genuine drift-robustness from this inertia we repeat the sweep holding the radius live and constant $( 2 r _ { n } \approx 0 . 0 8 5 < \Delta )$ by scaling the agent count inversely with the window, $N \approx 2 8 8 0 / W$ (Table 15). The false-switch rate stays zero across windows from 1.25 to 12.5 periods even at this live radius, so the drift-robustness is genuine time-average tracking and not the conservative inertia of a wide radius. All these windows sit at or above one drift period, where the windowing bias $\varepsilon _ { W } = b P / ( 2 | W | )$ stays within the margin and oscillates in sign as the window slides, absorbed by the time-uniform radius. Holding the radius live at shorter windows by adding agents is not free on the switch-adaptive side of this agent-scaled sweep. The genuine-switch adopt latency grows from 5165 rounds at 12.5 periods to 8997 at 9.4 periods and exceeds the horizon below about nine periods, but the fixed-agent control next shows that this growth is the cost of the added agents, not of the shorter window.

To isolate the window length from the agent count we hold $N = 2 4$ and the communication setup fixed and instead pin the radius live by deepening each probe block, $h \approx 7 2 0 / W$ , so the per-arm window count $n = W h$ and the radius stay fixed while only the number of probe-blocks changes (Table 16). Across the reachable live band, from about two to 12.5 drift periods, DRFC-Seq holds a zero false-switch rate and adopts the genuine switch in every run, and now the adopt latency falls as the window shortens, from 5193 rounds at 12.5 periods to 3073 at about two periods. The window length is therefore a wide and forgiving design choice rather than a hand-picked value. A window of about two drift periods already gives the drift-robust good corner and adopts fastest at this agent count. At a fixed agent count the radius-liveness condition couples to the window length, since a deeper block consumes rounds, so the in-rounds window cannot fall below about two periods here, which is why the agent-scaled sweep above is needed to reach shorter live windows.

<table><tr><td>Sweep setting FSR</td><td>regret</td></tr><tr><td>Window length (drift periods),</td><td> $\alpha = 0 . 1 0$ </td></tr><tr><td>1.04 0.00 0.00</td><td>9972</td></tr><tr><td>1.66 2.18 0.00</td><td>9972</td></tr><tr><td>3.12 0.00</td><td>9973</td></tr><tr><td>4.68 0.00</td><td>9974</td></tr><tr><td>6.24 0.00</td><td>9972</td></tr><tr><td>9.36 0.00</td><td>9970</td></tr><tr><td></td><td>9972</td></tr><tr><td>12.48 0.00</td><td>9971</td></tr><tr><td>16.64 0.00</td><td>9975</td></tr><tr><td>Confidence level 0.01 0.00 0.05 0.00 0.10 0.00 0.20 0.00</td><td>α, window = 12.5 periods 9974 9972 9973</td></tr></table>

Table 14: DRFC-Seq under window and confidence misspecification within-regime drift $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0$ $b = 0 . 2 0 > \Delta = 0 . 1 0$ , period $P = 5 0 0$ rounds, N = 24, 30 seeds). The false-switch rate (fraction of seeds with at least one false switch) stays 0 and decision regret stays flat across a 16× window range and a 20× confidence range. Reproduced by the window-sensitivity sweep.

## Removing the drift-period knob with Doubling-Window DRFC-Seq

The window sweeps above hold the sliding window matched to a known drift period. Doubling-Window DRFC-Seq (Theorem 7) removes that knowledge. It runs a geometric ladder of windows $W \in \{ 8 , 1 6 , 3 2 , 6 4 \}$ probe-blocks on one shared probe stream at union level $\alpha / J$ , anoints the shortest window whose certificate has matched the longest drift-safe window $W _ { \mathrm { m a x } }$ over a full $W _ { \mathrm { m a x } } .$ -block horizon, and adopts on the shortest anointed window. The anointment is the empirically checkable certificate that Theorem 7 turns into drift-safety, so the monitor adopts only through a window proven to track the time-average best arm, while paying the per-switch latency of the shortest such window rather than of $W _ { \mathrm { m a x } }$

We stress this against an unknown drift scale. Each of 30 seeds (10 per shape) draws a random drift period P uniform in [600, 4000] rounds, a random phase, and a random waveform shape from square, triangle, and sine, all with margin $\Delta = 0 . 1 0$ and amplitude $b = 0 . 2 2$ (so $b > 2 \Delta$ , the cancellation regime of Proposition 1). Each draw is run in two cells, a drift-only cell $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0$ measuring the false-switch rate) and a drift-plus-genuine-switch cell $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 1$ with the same live drift, the average-best arm flipping at the horizon midpoint, measuring the adopt rate and detection latency). Four monitors are compared on identical instances. DW-Adaptive is the doubling-window policy with no knowledge of P. Oracle-W<sup>⋆</sup> is single-window DRFC-Seq told P and using the shortest grid window with a constant-fraction drift-safety margin, $L ( W ^ { \star } ) \geq 2 P _ { \mathrm { { i } } }$ ; this cover factor 2 is close to the $b / \Delta = 2 . 2$ scale at which Theorem 7 places its constant-fraction-margin $W ^ { \star }$ , though the experiment does not tie the two exactly. This is the unfair upper reference. Fixed-Short and Fixed-Long pin the single window to the shortest and the longest grid entry. The probe-block span is 212 rounds, so the grid windows span 1696 to 13568 rounds and the period range exercises a drift-safe window $W ^ { \star }$ that ranges across the whole ladder as P varies. The drawn periods are arbitrary reals, not multiples of the block span, so the whole-block period convention of Theorem 7 does not hold exactly here. Remark 4 covers exactly this case, replacing the convention with an explicit of-grid slack $2 b / W$ that the fine block grid keeps well below the confidence radius, and the zero false-switch rate below confirms that analysis rather than relying on the alignment.

<table><tr><td>window (periods)</td><td>N</td><td> $2 r _ { n }$ </td><td>drift FSR  $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } ( \mathrm { = } 0 )$ </td><td>genuine adopt (latency)</td></tr><tr><td>1.25</td><td>240</td><td>0.085</td><td>0.00</td><td></td></tr><tr><td>1.56</td><td>192</td><td>0.085</td><td>0.00</td><td></td></tr><tr><td>1.87</td><td>160</td><td>0.085</td><td>0.00</td><td></td></tr><tr><td>2.18</td><td>138</td><td>0.085</td><td>0.00</td><td>censored</td></tr><tr><td>3.12</td><td>96</td><td>0.085</td><td>0.00</td><td>censored</td></tr><tr><td>4.68</td><td>64</td><td>0.085</td><td>0.00</td><td>censored</td></tr><tr><td>6.24</td><td>48</td><td>0.085</td><td>0.00</td><td>censored</td></tr><tr><td>9.36</td><td>32</td><td>0.085</td><td>0.00</td><td>0.85 (8997)</td></tr><tr><td>12.48</td><td>24</td><td>0.085</td><td>0.00</td><td>1.00 (5165)</td></tr></table>

Table 15: DRFC-Seq at a held-live radius. The anytime radius is pinned at $2 r _ { n } \approx 0 . 0 8 5 < \Delta = 0 . 1 0$ across all window lengths by scaling the agent count inversely with the window $( N \approx 2 8 8 0 / W )$ , so safety cannot come from a conservatively wide radius. The drift-only false-switch rate $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0$ $b = 0 . 2 0$ , 20 seeds) stays 0 across the full range of windows from about one to twelve drift periods, all on the safe side of the one-period bias boundary. The genuine-switch adopt rate (separated margin $\Delta = 0 . 2 0$ , latency in rounds in parentheses) shows the trade-of of holding the radius live at short windows, where the larger agent count slows coordination and the adopt latency grows until it exceeds the horizon (censored). The period-spanning window adopts fastest. Reproduced by the held-live-radius sweep.

Table 17 and Figure 18 report the result. DW-Adaptive holds a zero false-switch rate in both cells and adopts every genuine switch, matching Oracle-W<sup>⋆</sup>, and its detection latency tracks the oracle that is told the period at a mean ratio of 1.36 and a median ratio of 1.19, well inside the doubling-factor-2 slack of Theorem 7, using no period input. The mean is the looser of the two because a few seeds anoint a conservatively long window, while the median ratio 1.19 sits near the no-overhead floor the theorem predicts, the operating window matching the oracle’s up to the $( 1 + o ( 1 ) )$ union inflation of the radius. DW-Adaptive is 27 percent faster in mean latency than the drift-safe Fixed-Long window, which must use $W _ { \mathrm { m a x } }$ at every period. Fixed-Short is the only monitor that violates safety, false-switching on 1 of the 30 drift-plus-switch seeds, the one with the longest drift period $P = 3 8 8 9$ , because its window is below the period at the long end of the range. The pattern is the Pareto statement of the theorem. No single fixed window is both safe and fast across the randomized range, Fixed-Long is safe but slow and Fixed-Short is fast but drift-unsafe, while DW-Adaptive recovers the period-oracle’s operating point online and across all three waveform shapes.

## Robustness to a changing drift period

The randomized sweep above keeps the drift period fixed within each run and only unknown across runs. A natural follow-up question is a period that itself changes during a run. We switch the drift half-period once at the horizon midpoint, so every run contains two distinct periods, and we test both directions, a grow case where the full period rises from 700 to 2800 rounds and a shrink case where it falls from 2800 to 700, with the genuine decision switch placed at 0.7 of the horizon inside the post-change phase. The waveform phase is carried continuously across the change, so the time-average gap stays $\Delta = 0 . 1 0$ on both sides and the time-average best arm is unchanged. The four monitors match Table 17, except that Oracle-W<sup>⋆</sup> is now told the post-change period and Fixed-Early pins the window matched to the pre-change period.

<table><tr><td>window (periods)</td><td>h</td><td>drift FSR genuine  $( S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } { = } 0 )$  adopt</td><td>latency (rounds)</td></tr><tr><td>2.18</td><td>34</td><td>0.00 1.00</td><td>3073</td></tr><tr><td>3.12</td><td>24</td><td>0.00 1.00</td><td>3243</td></tr><tr><td>4.68</td><td>16</td><td>0.00 1.00</td><td>3512</td></tr><tr><td>6.24</td><td>12</td><td>0.00 1.00</td><td>3879</td></tr><tr><td>9.36</td><td>8</td><td>0.00 1.00</td><td>4486</td></tr><tr><td>12.48</td><td>6</td><td>0.00 1.00</td><td>5193</td></tr></table>

Table 16: DRFC-Seq at a held-live radius with the agent count fixed $( N = 2 4 )$ and the communication setup unchanged. The radius is pinned live $( 2 r _ { n } \approx 0 . 0 8 5 < \Delta = 0 . 1 0 )$ by deepening each probe block, $h \approx 7 2 0 / W$ , so the per-arm window count $n = W h \approx 7 2 0$ and only the number of probe-blocks W varies. Across the reachable live band DRFC-Seq holds a zero false-switch rate (drift-only, $S _ { \mathrm { d e c } } ^ { \mathrm { a v g } } = 0 , b = 0 . 2 0$ , 20 seeds) and adopts the genuine switch in every run (separated margin $\Delta = 0 . 2 0 )$ , with the adopt latency falling as the window shortens. This removes the agentcount confound of Table 15, leaving window length the only varying factor. Reproduced by the fixed-agent held-live-radius sweep.
<table><tr><td>Monitor</td><td>knows  $P$ </td><td>drift FSR</td><td></td><td>adopt latency</td><td>switch FSR</td></tr><tr><td>DW-Adaptive</td><td>no</td><td>0.00</td><td>1.00</td><td>8031</td><td>0.00</td></tr><tr><td>Oracle  $. W ^ { \star }$ </td><td>yes</td><td>0.00</td><td>1.00</td><td>5894</td><td>0.00</td></tr><tr><td>Fixed-Short</td><td>no</td><td>0.00</td><td>1.00</td><td>2714</td><td>0.03</td></tr><tr><td>Fixed-Long</td><td>no</td><td>0.00</td><td>1.00</td><td>11004</td><td>0.00</td></tr></table>

Table 17: Period-agnostic window robustness (Doubling-Window DRFC-Seq, 30 seeds, randomized period $P \in [ 6 0 0 , 4 0 0 0 ]$ , randomized phase, square/triangle/sine drift, $\Delta = 0 . 1 0 , b = 0 . 2 2 , N = 2 4 )$ Columns report the drift-only false-switch rate, the genuine-switch adopt rate, the detection latency in rounds, and the false-switch rate in the drift-plus-switch cell. DW-Adaptive matches the periodoracle on false switches and adopt rate and tracks its latency without knowing $P ,$ is $2 7$ percent faster than the drift-safe Fixed-Long window, and unlike Fixed-Short never violates safety. Reproduced by the period-agnostic window sweep.

Table 18 reports the outcome over 10 seeds per case. DW-Adaptive holds a zero drift false-switch rate and adopts every genuine switch in both directions, and its mean detection latency tracks the post-change oracle at a ratio of 0.91 on the grow case and 0.96 on the shrink case, recovering the post-change oracle operating point online with no period input. This regime is a robustness check rather than a separation. Both fixed-window baselines also stay drift-safe here, and Fixed-Early is in fact faster, because the period change stays inside the anytime radius tolerance of the shortest window, so no fixed window is forced to false-switch. The separating regime, where a too-short fixed window does false-switch, is the randomized sweep of Table 17 in which Fixed-Short is the single monitor that breaks. The narrow and specific point here is that the doubling ladder recovers the post-change oracle’s safety and latency without being told either period, not that it dominates a fixed window in this easy setting.

![](images/2b30f0b6ee9118bc596fa9224937b6e6b52e6383ec332bb32998c043b57ee43c.jpg)

![](images/bb276b6e5694c2f689c8c7f87b1eaf739cdafd4e4cb422d4399933a35f0cf5b4.jpg)  
Figure 18: Doubling-Window DRFC-Seq is period-agnostic. (a) Per-seed genuine-switch detection latency against the realized drift period $P ,$ across 30 seeds and the square, triangle, and sine drift shapes. DW-Adaptive (blue) sits just above the period-oracle Oracle- $W ^ { \star }$ (gray) and well below the drift-safe Fixed-Long window (green) at every period, while Fixed-Short (red) is fastest but is the one monitor that false-switches. (b) The false-switch versus latency Pareto frontier, one point per monitor. DW-Adaptive, Oracle- $W ^ { \star }$ , and Fixed-Long all hold a zero genuine-phase false-switch rate, and DW-Adaptive reaches that safe frontier far to the left of Fixed-Long, while Fixed-Short pays a nonzero false-switch rate for its low latency. The proposed monitor recovers the period-oracle’s operating point without the period as input.

Confidence-scale sweep and theorem-calibrated regime. The confidence-radius constant $C = 1 . 1 5$ in the K-ary and user-cluster simulators is the result of practical tuning, not the worst-case theorem-calibrated value. Table 19 quantifies the gap by sweeping the constant for the conservative certifier across the practical value, the per-pair union-bound value $\sqrt { 2 \log ( K ^ { 2 } / \delta ) } \approx 3 . 1 9$ , and the worst-case appendix value $\sqrt { 2 \log ( K ^ { 2 } T ^ { 3 } / \delta ) }$ , which is $\approx 7 . 2 6$ at horizon 1200 and $\approx 8 . 4 1$ at horizon 24000. Three facts emerge. First, raising C errs only in the conservative direction. The false-switch rate never exceeds the practical point’s 0.03 and is 0 at every $C \ \geq \ 2$ . Second, at the default operating point the radius after one comparison block is $\approx 0 . 4 3$ for the worst-case constant against a decision margin of 0.10, so elimination cannot finish inside the 400-round regime and certification stalls, which is why the reported experiments use the practical value. Third, in a theorem-calibrated regime whose regime length respects the $c _ { 0 } K$ log $T / ( N \Delta ^ { 2 } )$ floor of the main information-theoretic lower bound, every swept constant becomes operational and certifies at zero false-switch rate, with certification cost scaling as $C ^ { 2 } / \Delta ^ { 2 }$ exactly as in the $\mathcal { C } _ { r }$ term of the main DRFC theorem. $\mathrm { D R F C - }$ Seq’s radius uses the same functional form as the certified object, so the zero-false-switch conclusion under drift is scale-robust in the same conservative direction. The reported implementation uses this radius in the finite-horizon form log $K ( K - 1 ) T / \alpha )$ rather than the horizon-free stitched form log $( C _ { 0 } K ( K - 1 ) s ^ { 2 } / \alpha )$ of the DRFC-Seq anytime-validity proof, a practical proxy at a known run length. The two forms difer only by the start-time peeling factor, and the sweep above shows the zero-false-switch conclusion is preserved across radius constants from the practical value up to the worst-case appendix value, so it does not depend on which valid radius form is used.

<table><tr><td>Case</td><td>Monitor</td><td>knows P</td><td>drift FSR</td><td></td><td>adopt latency</td><td>ratio</td></tr><tr><td>grow</td><td>DW-Adaptive</td><td>no</td><td>0.00</td><td>1.00</td><td>6278</td><td>0.91</td></tr><tr><td>grow</td><td>Oracle-W*</td><td>yes</td><td>0.00</td><td>1.00</td><td>6872</td><td>1.00</td></tr><tr><td>grow</td><td>Fixed-Early</td><td>no</td><td>0.00</td><td>1.00</td><td>2229</td><td>0.32</td></tr><tr><td>grow</td><td>Fixed-Long</td><td>no</td><td>0.00</td><td>1.00</td><td>7148</td><td>1.04</td></tr><tr><td>shrink</td><td>DW-Adaptive</td><td>no</td><td>0.00</td><td>1.00</td><td>7173</td><td>0.96</td></tr><tr><td>shrink</td><td>Oracle-W*</td><td>yes</td><td>0.00</td><td>1.00</td><td>7508</td><td>1.00</td></tr><tr><td>shrink</td><td>Fixed-Early</td><td>no</td><td>0.00</td><td>1.00</td><td>6490</td><td>0.86</td></tr><tr><td>shrink</td><td>Fixed-Long</td><td>no</td><td>0.00</td><td>1.00</td><td>6618</td><td>0.88</td></tr></table>

Table 18: Robustness to a mid-run drift-period change (Doubling-Window DRFC-Seq, 10 seeds per case, N = 24, K = 4, $\Delta = 0 . 1 0$ , b = 0.22, square, triangle, and sine drift). The drift half-period switches once at the horizon midpoint, the grow case from full period 700 to 2800 rounds and the shrink case from 2800 to 700, with the genuine switch in the post-change phase. Columns report whether the monitor is told the post-change period, the drift-only false-switch rate, the genuineswitch adopt rate, the detection latency in rounds, and the latency ratio to the post-change oracle. All monitors stay drift-safe in this regime, so it is a robustness check rather than a separation, and Fixed-Early is in fact faster. The point is that DW-Adaptive recovers the post-change oracle’s safety and latency without knowing either period, while the separating regime where a too-short fixed window breaks is Table 17. Reproduced by the changing-period sweep.

Dataset license. MovieLens-1M [Harper and Konstan, 2015] is distributed by GroupLens Research under the dataset’s own license, which permits research use.

Ethical considerations. This work studies cooperative decision-making in decentralized systems with heterogeneous agents. We use MovieLens-1M as a publicly released, fully anonymized benchmark for cooperative learning, and no human-subject experiments or personally identifying data are involved. The decentralized framework can support privacy-preserving deployments, since no raw observations leave an agent, though deployment in user-facing recommender systems requires further audit for fairness across demographic strata.

Additional related references. Recent work further studies switch-count non-stationarity, formal definitions of non-stationary bandits, smooth or heavy-tailed change processes, constrained feedback, linear dynamic regret, and heterogeneous or robust multi-agent bandits [Abbasi-Yadkori et al., 2023, Liu et al., 2023, Suk, 2024, Genalti et al., 2025, Li and Li, 2025, Hu et al., 2026, Xu et al., 2025, Mirfakhar et al., 2025, Adams et al., 2025, Wang and Xu, 2025, Hu et al., 2025].

<table><tr><td>Regime</td><td>C</td><td>adopt false</td><td></td><td>delay regret</td></tr><tr><td>default</td><td>1.15 practical</td><td>0.90 0.03</td><td>210</td><td>904</td></tr><tr><td>default</td><td>2.00</td><td>0.70 0.00</td><td>486</td><td>927</td></tr><tr><td>default</td><td>3.19 per-pair</td><td>0.00 0.00</td><td>1173</td><td>1000</td></tr><tr><td>default</td><td>5.00</td><td>0.00 0.00</td><td>1200</td><td>977</td></tr><tr><td>default</td><td>7.26 worst-case</td><td>0.00 0.00</td><td>1200</td><td>977</td></tr><tr><td></td><td>calibrated 1.15 practical</td><td>1.00 0.00</td><td>224</td><td>751</td></tr><tr><td></td><td>calibrated 3.19 per-pair</td><td>1.00 0.00</td><td>749</td><td>2532</td></tr><tr><td></td><td>calibrated 8.41 worst-case</td><td>0.90 0.00</td><td>4855</td><td>12076</td></tr></table>

Table 19: Confidence-scale sweep for the conservative certifier (30 seeds, $S _ { \mathrm { d e c } } = 2 , \Delta = 0 . 1 0 )$ . The default operating point is the reported K-ary stress configuration (horizon 1200, monitor period 120, regime length 400). The theorem-calibrated regime stretches the horizon to 24000 with monitor period 4000, so each 8000-round regime is long enough for the worst-case constant to finish its elimination scans. Columns report the fraction of runs adopting every decision switch, the falseswitch rate, mean detection delay in rounds (censored at the horizon when no switch is ever certified), and mean group regret. Raising C never creates false switches, and in the calibrated regime even the worst-case constant adopts every switch in 90 percent of runs (at least one switch in every run) at false-switch rate 0, at a certification cost that grows as $C ^ { 2 } / \Delta ^ { 2 }$ in line with the $\mathcal { C } _ { r }$ term of the main DRFC theorem.

## References

Yasin Abbasi-Yadkori, Andras Gyorgy, and Nevena Lazic. A new look at dynamic regret for nonstationary stochastic bandits. Journal of Machine Learning Research, 24(288):1–37, 2023. URL https://www.jmlr.org/papers/v24/22-0387.html.

Katherine B. Adams, Justin J. Boutilier, Qinyang He, and Yonatan Mintz. Finite-time guarantees for multi-agent combinatorial bandits with nonstationary rewards. arXiv preprint arXiv:2508.20923, 2025. doi: 10.48550/arXiv.2508.20923. URL https://arxiv.org/abs/2508.20923.

Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47(2–3):235–256, 2002. doi: 10.1023/A:1013689704352.

Peter Auer, Pratik Gajane, and Ronald Ortner. Adaptively tracking the best bandit arm with an unknown number of distribution changes. In Proceedings of the 32nd Conference on Learning Theory (COLT), volume 99 of Proceedings of Machine Learning Research, pages 138–158, 2019. URL https://proceedings.mlr.press/v99/auer19a.html.

Kazuoki Azuma. Weighted sums of certain dependent random variables. Tohoku Mathematical Journal, 19(3):357–367, 1967. doi: 10.2748/tmj/1178243286.

Rémi Bardenet and Odalric-Ambrym Maillard. Concentration inequalities for sampling without replacement. Bernoulli, 21(3):1361–1385, 2015. doi: 10.3150/14-BEJ605.

Omar Besbes, Yonatan Gur, and Assaf Zeevi. Stochastic multi-armed-bandit problem with nonstationary rewards. In Advances in Neural Information Processing Systems, volume 27, pages 199–207, 2014. URL https://papers.nips.cc/paper/5378-stochastic-multi-armed-bandit-problemwith-non-stationary-rewards.

Lilian Besson, Emilie Kaufmann, Odalric-Ambrym Maillard, and Julien Seznec. Eficient changepoint detection for tackling piecewise-stationary bandits. Journal of Machine Learning Research, 23(77):1–40, 2022. URL https://jmlr.org/papers/v23/20-1384.html.

Paolo Braca, Stefano Marano, Vincenzo Matta, and Peter Willett. Consensus-based page’s test in sensor networks. Signal Processing, 91(4):919–930, 2011. doi: 10.1016/j.sigpro.2010.09.011.

Jean Bretagnolle and Catherine Huber. Estimation des densités: risque minimax. In Séminaire de Probabilités XII, volume 649 of Lecture Notes in Mathematics, pages 342–363. Springer Berlin Heidelberg, 1978. doi: 10.1007/BFb0064610.

Yang Cao, Zheng Wen, Branislav Kveton, and Yao Xie. Nearly optimal adaptive procedure with change detection for piecewise-stationary bandit. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics, pages 418–427, 2019. URL http: //proceedings.mlr.press/v89/cao19a.html.

Ronshee Chawla, Abishek Sankararaman, Ayalvadi Ganesh, and Sanjay Shakkottai. The gossiping insert-eliminate algorithm for multi-agent bandits. In Proceedings of the 23rd International Conference on Artificial Intelligence and Statistics (AISTATS), volume 108 of Proceedings of Machine Learning Research, pages 3471–3481. PMLR, 2020. URL https://proceedings.mlr.press/ v108/chawla20a.html.

Ronshee Chawla, Daniel Vial, Sanjay Shakkottai, and R. Srikant. Collaborative multi-agent heterogeneous multi-armed bandits. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 4189–4217. PMLR, 2023. URL https://proceedings.mlr.press/v202/chawla23a.html.

Xiaotong Cheng and Setareh Maghsudi. Distributed consensus algorithm for decision-making in multi-agent multi-armed bandit. arXiv preprint arXiv:2306.05998, 2023. doi: 10.48550/arXiv. 2306.05998. URL https://arxiv.org/abs/2306.05998.

Wang Chi Cheung, David Simchi-Levi, and Ruihao Zhu. Learning to optimize under nonstationarity. In Proceedings of the 22nd International Conference on Artificial Intelligence and Statistics (AISTATS), volume 89 of Proceedings of Machine Learning Research, pages 1079–1087. PMLR, 2019. URL https://proceedings.mlr.press/v89/cheung19b.html.

Aurélien Garivier and Eric Moulines. On upper-confidence bound policies for switching bandit problems. In Algorithmic Learning Theory (ALT), pages 174–188, 2011. doi: 10.1007/978-3-642- 24412-4\_16. URL https://arxiv.org/abs/0805.3415. Preprint version 2008, arXiv:0805.3415.

Gianmarco Genalti, Sujay Bhatt, Nicola Gatti, and Alberto Maria Metelli. Catoni-style change point detection for regret minimization in non-stationary heavy-tailed bandits. arXiv preprint arXiv:2505.20051, 2025. URL https://arxiv.org/abs/2505.20051.

F. Maxwell Harper and Joseph A. Konstan. The MovieLens datasets: History and context. ACM Transactions on Interactive Intelligent Systems, 5(4):19:1–19:19, 2015. doi: 10.1145/2827872.

Steven R. Howard, Aaditya Ramdas, Jon McAulife, and Jasjeet Sekhon. Time-uniform, nonparametric, nonasymptotic confidence sequences. The Annals of Statistics, 49(2):1055–1080, 2021. doi: 10.1214/20-AOS1991. URL https://arxiv.org/abs/1810.08240.

Zicheng Hu, Yuchen Wang, and Cheng Chen. Robust decentralized multi-armed bandits: From corruption-resilience to byzantine-resilience. arXiv preprint arXiv:2511.10344, 2025. URL https: //arxiv.org/abs/2511.10344.

Zihao Hu, Yuan Yao, Jiheng Zhang, and Zhengyuan Zhou. Dynamic regret for non-stationary linear bandits via misspecification reductions. arXiv preprint arXiv:2607.02891, 2026. URL https://arxiv.org/abs/2607.02891.

Emilie Kaufmann and Wouter M. Koolen. Mixture martingales revisited with applications to sequential tests and confidence intervals. Journal of Machine Learning Research, 22(246):1–44, 2021. URL https://jmlr.org/papers/v22/18-798.html.

Emilie Kaufmann, Olivier Cappé, and Aurélien Garivier. On the complexity of best-arm identification in multi-armed bandit models. Journal of Machine Learning Research, 17(1):1–42, 2016.

David Kempe, Alin Dobra, and Johannes Gehrke. Gossip-based computation of aggregate information. In Proceedings of the 44th Annual IEEE Symposium on Foundations of Computer Science (FOCS), pages 482–491, 2003. doi: 10.1109/SFCS.2003.1238221.

Ravi Kumar Kolla, Krishna Jagannathan, and Aditya Gopalan. Collaborative learning of stochastic bandits over a social network. IEEE/ACM Transactions on Networking, 26(4):1782–1795, 2018. doi: 10.1109/TNET.2018.2852361.

Junpei Komiyama, Edouard Fouché, and Junya Honda. Finite-time analysis of globally nonstationary multi-armed bandits. Journal of Machine Learning Research, 25, 2024. URL https: //www.jmlr.org/papers/v25/21-0916.html.

Tze Leung Lai. Sequential changepoint detection in quality control and dynamical systems. Journal of the Royal Statistical Society. Series B (Methodological), 57(4):613–644, 1995. doi: 10.1111/j. 2517-6161.1995.tb02052.x.

Peter Landgren, Vaibhav Srivastava, and Naomi Ehrich Leonard. On distributed cooperative decision-making in multiarmed bandits. In 2016 European Control Conference (ECC), pages 243–248, 2016. doi: 10.1109/ECC.2016.7810293.

O. V. Lepski˘ı. On a problem of adaptive estimation in Gaussian white noise. Theory of Probability & Its Applications, 35(3):454–466, 1991. doi: 10.1137/1135065.

Shaoang Li and Jian Li. Constrained feedback learning for non-stationary multi-armed bandits. arXiv preprint arXiv:2509.15073, 2025. URL https://arxiv.org/abs/2509.15073.

Fang Liu, Joohyun Lee, and Ness Shrof. A change-detection based framework for piecewisestationary multi-armed bandit problem. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32, 2018. doi: 10.1609/aaai.v32i1.11746. URL https://ojs.aaai.org/index. php/AAAI/article/view/11746.

Jingyuan Liu, Hao Qiu, Lin Yang, and Mengfan Xu. Distributed multi-agent bandits over Erdős– Rényi random networks. arXiv preprint arXiv:2510.22811, 2025. URL https://arxiv.org/abs/ 2510.22811.

Yueyang Liu, Xu Kuang, and Benjamin Van Roy. A definition of non-stationary bandits. arXiv preprint arXiv:2302.12202, 2023. URL https://arxiv.org/abs/2302.12202.

David Martinez-Rubio, Varun Kanade, and Patrick Rebeschini. Decentralized cooperative stochastic bandits. In Advances in Neural Information Processing Systems, volume 32, 2019. URL https: //proceedings.neurips.cc/paper/2019/hash/85353d3b2f39b9c9b5ee3576578c04b7-Abstract.html.

Amirmahdi Mirfakhar, Xuchuang Wang, Jinhang Zuo, Yair Zick, and Mohammad Hajiesmaili. Heterogeneous multi-agent bandits with parsimonious hints. arXiv preprint arXiv:2502.16128, 2025. URL https://arxiv.org/abs/2502.16128.

George V. Moustakides. Optimal stopping times for detecting changes in distributions. The Annals of Statistics, 14(4):1379–1387, 1986. doi: 10.1214/aos/1176350164.

Angelia Nedić and Alex Olshevsky. Distributed optimization over time-varying directed graphs. IEEE Transactions on Automatic Control, 60(3):601–615, 2015. doi: 10.1109/TAC.2014.2364096.

E. S. Page. Continuous inspection schemes. Biometrika, 41(1/2):100–115, 1954. doi: 10.1093/ biomet/41.1-2.100.

Moshe Pollak. Optimal detection of a change in distribution. The Annals of Statistics, 13(1): 206–227, 1985. doi: 10.1214/aos/1176346587.

Chengshuai Shi and Cong Shen. Federated multi-armed bandits. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pages 9603–9611, 2021. URL https://ojs.aaai. org/index.php/AAAI/article/view/17156.

Joe Suk. Adaptive smooth non-stationary bandits. arXiv preprint arXiv:2407.08654, 2024. URL https://arxiv.org/abs/2407.08654.

Alexander G. Tartakovsky and Venugopal V. Veeravalli. Asymptotically optimal quickest change detection in distributed sensor systems. Sequential Analysis, 27(4):441–475, 2008. doi: 10.1080/ 07474940802446236.

Alexandre B. Tsybakov. Introduction to Nonparametric Estimation. Springer, 2009. doi: 10.1007/ b13794.

Venugopal V. Veeravalli. Decentralized quickest change detection. IEEE Transactions on Information Theory, 47(4):1657–1665, 2001. doi: 10.1109/18.923755.

John Wang and Mengfan Xu. Multi-objective multi-agent bandits: From learning eficiency to fairness optimization. arXiv preprint arXiv:2605.06864, 2026. URL https://arxiv.org/abs/2605. 06864.

Xingyu Wang and Mengfan Xu. Multi-agent multi-armed bandit with fully heavy-tailed dynamics. arXiv preprint arXiv:2501.19239, 2025. doi: 10.48550/arXiv.2501.19239. URL https://arxiv.org/ abs/2501.19239.

Xuchuang Wang, Yu-Zhen Janice Chen, Xutong Liu, Lin Yang, Mohammad Hajiesmaili, Don Towsley, and John C. S. Lui. Asynchronous multi-agent bandits: Fully distributed vs. leadercoordinated algorithms. Proceedings of the ACM on Measurement and Analysis of Computing Systems, 9(1):1–39, 2025. doi: 10.1145/3711696. URL https://dl.acm.org/doi/10.1145/3711696.

Yuanhao Wang, Jiachen Hu, Xiaoyu Chen, and Liwei Wang. Distributed bandit learning: Nearoptimal regret with eficient communication. In International Conference on Learning Representations (ICLR), 2020. URL https://arxiv.org/abs/1904.06309.

Ian Waudby-Smith and Aaditya Ramdas. Estimating means of bounded random variables by betting. Journal of the Royal Statistical Society Series B: Statistical Methodology, 86(1):1–27, 2024. doi: 10.1093/jrsssb/qkad009. URL https://arxiv.org/abs/2010.09686.

Chen-Yu Wei and Haipeng Luo. Non-stationary reinforcement learning without prior knowledge: An optimal black-box approach. In Proceedings of the 34th Conference on Learning Theory (COLT), volume 134 of Proceedings of Machine Learning Research, pages 4300–4354. PMLR, 2021. URL https://proceedings.mlr.press/v134/wei21b.html.

Mengfan Xu and Diego Klabjan. Decentralized randomly distributed multi-agent multi-armed bandit with heterogeneous rewards. In Advances in Neural Information Processing Systems, volume 36, 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/ ec795aeadae0b7d230fa35cbaf04c041-Abstract-Conference.html.

Mengfan Xu, Liren Shan, Fatemeh Ghafari, Xuchuang Wang, Xutong Liu, and Mohammad Hajiesmaili. Heterogeneous multi-agent multi-armed bandits on stochastic block models. arXiv preprint arXiv:2502.08003, 2025. doi: 10.48550/arXiv.2502.08003. URL https://arxiv.org/abs/2502.08003.

Lin Yang, Yu-Zhen Janice Chen, Mohammad H. Hajiesmaili, John C. S. Lui, and Don Towsley. Distributed bandits with heterogeneous agents. In IEEE INFOCOM 2022 - IEEE Conference on Computer Communications, pages 200–209, 2022. doi: 10.1109/INFOCOM48880.2022.9796901. URL https://arxiv.org/abs/2201.09353.