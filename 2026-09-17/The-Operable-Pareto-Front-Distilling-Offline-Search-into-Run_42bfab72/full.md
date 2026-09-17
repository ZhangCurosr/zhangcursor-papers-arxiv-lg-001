# The Operable Pareto Front: Distilling Offline Search into Run-Time Control for Multi-Objective UAV Edge-Computing Scheduling

Qiao Liao, Zhiyong Feng, Bin Wu, Guodong Fan

Abstract—A UAV mobile edge computing (MEC) fleet trades energy against delay, and its schedules form a Pareto front; we call a scheduler operable when the fleet can be asked for any point on that front at run time. We propose PrefDT, to the best of our knowledge the first preference-conditioned Decision Transformer for the problem of joint trajectory, association and offloading scheduling. Its idea comes from language modeling: we hand the model the desired trade-off as an input, such that a single model only needs to be trained once offline to return any desired point on the curve in one rollout. The network state is summarized by attention pooling with a per-user bypass, so the scheduler keeps working when user reports are lost. The energy target is a running budget decremented by what the fleet actually spends. As a result, when wind or load pushes consumption off the plan, the policy can track the difference and hold its budget. Because no corpus of preference-labeled flights exists, we design a distillation pipeline and build the corpus by ourselves. In simulation against 26 method variants, PrefDT produces the best trade-off curve of any learned method and holds its energy budget to within 0.6% when propulsion cost rises by half in mid-flight.

Index Terms—UAV mobile edge computing, multi-objective scheduling, Decision Transformer, offline reinforcement learning, preference conditioning.

## I. INTRODUCTION

U <sup>NMANNED</sup> <sup>Aerial</sup> <sup>Vehicles</sup> <sup>(UAVs)</sup> <sup>carrying</sup> <sup>edge</sup> servers extend mobile edge computing (MEC) to places where ground infrastructure is absent, damaged, or overloaded. A fleet is expected to balance two costs: the time users wait and the energy the aircraft burn, and the balance it should strike is not fixed for the mission. In a disaster response, delay matters most while survivors are being located, but energy matters most once the fleet must stay aloft until relief arrives; the same fleet may be asked for two different balances in one day. Generally, the schedule is computed for one balance before takeoff, and a different balance costs another optimization or training run. The schedule is also fixed once the fleet is in the air. It is built from predicted energy consumption and link quality, and cannot notice the difference when wind raises what a maneuver costs, or reports from users fail to arrive. Deployment therefore asks for a scheduler that takes the balance as a request at run time, holds an energy budget against what the fleet actually spends, and keeps deciding when reports are lost.

Solving this problem means planning three quantities at the same time. In every time slot the scheduler decides where each aircraft flies, which users it serves, and how much of each user’s task it takes on. These decisions are tightly coupled with each other and cannot be made separately [5], [6]. The decisions also serve two goals set against each other: finishing tasks quickly costs energy, and conserving energy costs delay. As a result, there is no single best schedule, and one must instead choose a single point on a trade-off curve called the Pareto front. The scheduling problem is NP-hard.

Three categories of schedulers dominate the current literature: model-based optimization, learned schedulers and population methods. They differ from each other mainly in when they commit to a trade-off. In particular, modelbased optimization commits at solve time: based on a chosen weight, the two conflicting goals are collapsed into one, and the resulting program is solved by block-coordinate descent with successive convex approximation (SCA) over detailed channel and propulsion models [5], [1], [6]. As a result, any weight change needs another run, and collapsing the goals into a single optimization objective puts part of the Pareto front permanently out of reach [2]. As the second category, learned schedulers commit at training time: a deep offloading policy such as DROO [7], or a graph-attention scheduler [22], is trained for one weighting and one scenario. In contrast, population methods (i.e., the third category) postpone the commitment by returning many solutions in a single run. For example, PGMORL [8] evolves a population of policies, while the classical dominance-based evolutionary search NSGA-II [23] evolves the solutions themselves. Unfortunately, this category of schedulers generally converges to a fixed menu of configurations rather than a continuum. To obtain a trade-off that is not on the menu, the search process must be run again.

In current literature, all three categories above are evaluated by how good a Pareto front they can produce. Such a criterion says nothing about the cost of changing to a different tradeoff, and what happens when a flight does not go as predicted. Moreover, the schedule is computed from predicted consumption, and none of these schedulers tracks the energy already spent; nothing is corrected when the actual flying cost departs from the predicted one.

A conditioned scheduler addresses the first issue above (i.e., the cost of changing to a different trade-off). It takes two inputs: the state of the network, and the energy–delay trade-off the fleet is asked to strike. Three requirements are desirable for a UAV fleet: (A) When the energy–delay trade-off is set differently, the fleet’s behavior follows in proportion, across the whole range the physics permits, and without becoming unstable when more is asked for than can be delivered. (B) Finish the mission within a given total energy budget, measured against the energy actually consumed rather than a predicted plan—even when conditions change mid-flight (e.g., a headwind or a heavier task load). (C) Accept a mid-flight revision of that budget, issued as a single new total, without re-planning or retraining. A scheduler is said to be operable if it meets all three requirements above, and Sec. IV-C makes the term measurable.

This paper builds an operable scheduler without sacrificing the quality of the Pareto front. The idea comes from language modeling. In particular, Decision Transformer [3] and Trajectory Transformer [12] recast control as conditional sequence generation. A trajectory becomes a stream of tokens; a causal model learns to continue that stream. The decisive part is where the goal goes: the desired outcome is supplied as an input token, which can be set at inference time, instead of being compiled into a training objective. The idea has since been extended in several directions—online adaptation [13], prompting [14], and multi-agent execution [15]—and two lines of work carry it toward our setting: conditioning on more than one objective, and use as a resource manager in wireless networks. PEDA [4] and related multi-objective variants [21] condition on a preference over vector-valued returns, evaluated on locomotion benchmarks. A value-based line conditions a critic on the preference rather than a sequence [11], [20], [24]. In wireless networking, Decision Transformers have begun to appear as resource managers [16], though so far with a scalar cost signal and a single objective. If what a policy should achieve can be supplied as an input, then needing a fresh solve for every trade-off follows from where the trade-off enters the pipeline, not from the problem itself.

Directly borrowing the idea from language modeling does not match the physical network, and three gaps exist. The first is the input. The set of users present changes from slot to slot, and each user needs a decision of its own, while a sequence model wants a fixed layout of inputs. Pooling all users into one summary accepts any number of them but erases their identity, which per-user decisions cannot afford. Our encoder therefore keeps the pooled summary for what the fleet has in common and adds a bypass: each user’s raw token is routed around the summary and directly into that user’s own association and offloading heads. One frozen model can then serve any number of users and still decide for each of them separately. An additional benefit comes for free: the summary is defined on any subset of the users, so a lost report leaves a smaller set rather than an incomplete input, and the scheduler keeps working on a link that drops reports.

The second gap is the setting. A Decision Transformer is steered by a single scalar, the return-to-go; folding delay and energy into one number leaves no part of it in joules. We give one value per objective instead, one for delay and the other for energy. The energy entry starts at the mission’s budget and is decremented, at each slot, by what the fleet actually consumed. By arithmetic, that entry is then the remaining allowance itself, and the policy reads it before every decision. The budget is thereby enforced against real consumption rather than against a predicted plan. A mid-flight revision is then a single overwrite of that entry, and, because each objective has an entry of its own, delay and energy targets can be set independently.

The third gap is the training data. This kind of model learns by imitation from recorded runs labeled with the outcome achieved, but no such records exist for our problem. We produce them ourselves: a compact control rule is searched once, and its archive of solutions spans the front; each sampled preference is matched to the archive member whose realized delay–energy share is nearest, and that member’s flight is recorded under the corresponding label. The label is therefore a realized trade-off rather than a weight, so a requested setting denotes an outcome the fleet can actually produce. The front’s extreme members are demonstrated more often than uniform sampling would, because imitation regresses toward the behavioral mean; the whole corpus costs one offline search, with no per-preference training.

Addressing all three gaps leads to PrefDT, the scheduler proposed in this paper. PrefDT turns the Pareto front from something a search has to produce into something a model can be asked for: one frozen model, trained entirely offline from preference-annotated logs and never updated again, returns any trade-off at the cost of a single rollout.

Our algorithmic contributions are summarized below.

(1) We propose PrefDT, an operable scheduler: one frozen model returns any point on the Pareto front at run time, so a new trade-off costs one rollout instead of another solve, search, or training run. It holds an energy budget against actual consumption and accepts a revised budget in flight.

(2) We bring the Decision Transformer of language modeling to UAV scheduling and adapt it to a physical network. The desired trade-off is supplied as an input at inference time rather than compiled into the training objective, so the scheduler takes its trade-off at run time. We add a per-user bypass to the pooled state encoder, so the scheduler decides for each user separately while the pooled summary accepts any number of users. We replace the scalar goal input with a vector, one value per objective, so every point on the front can be named and energy has a channel of its own. No preference-labeled flight data exists, so we build a distillation pipeline to produce it: a searched control rule supplies the flights, and each flight is labeled with the trade-off it actually struck.

(3) Three mechanisms carry these designs into deployment. Lost reports are tolerated because the pooled summary is defined on any subset of the active users: a lost report leaves a smaller set, and nothing is retrained. Energy is tracked because the energy entry of the goal vector starts at the mission’s budget and is decremented every slot by what was actually consumed: the policy reads the remaining budget, and a revision is one overwrite of that entry. The model learns the whole front including extremes, because the distillation pipeline matches each sampled preference to the archived rule whose realized delay–energy share is nearest and oversamples the extreme members.

Across 26 method variants under one protocol, PrefDT produces the best trade-off curve of any learned method, and a new trade-off costs one 0.41-second rollout, 620–2400× less than a re-solve or a retraining. When propulsion cost rises by half in mid-flight, the same frozen model stays within 0.6% of its energy budget.

The paper is organized as follows. Section II formulates the system model and the scheduling problem. Section III presents PrefDT: the encoder, the conditioning channel, and the distillation pipeline. Section IV describes the experimental setup and the operability criterion, and Section V reports the experiments. Section VI concludes the paper and states the limitations of this work. Appendices A–D are provided in the supplementary material.

## II. SYSTEM MODEL AND PROBLEM FORMULATION

Physical models follow the multi-UAV MEC literature [1], [6], [17] so that all baselines share an identical environment; Appendix A collects the symbols.

## A. Network model and decision variables

M rotary-wing UAVs, indexed by $m \in { \mathcal { M } }$ , fly at a fixed altitude H above a square service area and act as edge servers for U ground users during T slots of length $\delta _ { t } \colon \mathbf { q } _ { m } ( t ) \in  { \mathbb { R } } ^ { 2 }$ denotes the horizontal projection of UAV m and $\mathbf { w } _ { u } \in \mathbb { R } ^ { 2 }$ the (static) position of user u. In each slot an active set $\kappa ( t )$ requests service, its size drawn uniformly from $\{ 4 , \ldots , 8 \}$ and its members uniformly at random from the user pool, and each member carries a task with probability $p _ { a }$ . The active set therefore has time-varying cardinality. UAV kinematics follow

q<sub>m</sub>(t+1) = q<sub>m</sub>(t) + d<sub>m</sub>(t)- cos φ<sub>m</sub>(t), sin φ<sub>m</sub>(t)<sup>⊤</sup>, (1) with step length $0 ~ \le ~ d _ { m } ( t ) ~ \le ~ D _ { \mathrm { m a x } } ~ = ~ V _ { \mathrm { m a x } } \delta _ { t }$ , heading $\varphi _ { m } ( t ) \in [ 0 , 2 \pi )$ ), and speed $v _ { m } ( t ) = d _ { m } ( t ) / \delta _ { t }$ . In every slot the scheduler jointly selects

$$
\mathbf { a } _ { t } = \{ d _ { m } ( t ) , \varphi _ { m } ( t ) \} _ { m \in \mathcal { M } } \cup \big \{ x _ { u , m } ( t ) \big \} \cup \big \{ \rho _ { u } ( t ) \big \} _ { u \in K ( t ) , \mathfrak { n } }\tag{2}
$$

i.e., continuous flight controls, the binary association $x _ { u , m } ( t ) \in \{ 0 , 1 \}$ (each active user served by at most one UAV), and the continuous fraction $\rho _ { u } ( t ) \ \in \ [ 0 , 1 ]$ of task u offloaded to its serving UAV. Edge compute is shared equally among the users a UAV serves, $f _ { m , u } ( t ) ~ = ~ f _ { m } ^ { \operatorname* { m a x } } / | \mathcal { U } _ { m } ( t ) |$ where $f _ { m } ^ { \mathrm { m a x } }$ is UAV m’s total CPU frequency and ${ { \mathcal { U } } _ { m } } ( t ) =$ $\{ u \in \mathcal { K } ( t ) : x _ { u , m } ( t ) = 1 \}$ is the set of users it serves in slot t—a deliberate simplification; $f _ { m , u }$ can be promoted to a fourth decision without changing the framework.

## B. Air–ground channel

The channel power gain $h _ { u , m } ( t )$ follows the standard probabilistic air–ground model [17]: the elevation-dependent LoS probability of [17] mixes LoS and NLoS states, which share a free-space path loss and differ by a state-dependent excess attenuation $\eta _ { X }$ , by the variance of their log-normal shadowing, and by their Nakagami fading parameter; the channel is quasistatic within a slot. Since this component is standard, we omit its equations and take its constants from [17], whose urban parameter set we adopt; the link operates at a 2 GHz carrier, and the rate below uses B=1 MHz, $P _ { u } { = } 0 . 1 \ : \mathrm { W }$ and $\sigma ^ { 2 } { = } { \mathrm { - } } 1 1 0 \mathrm { d B m }$ . Each served user transmits on its own channel of bandwidth B, so the uplink rate is

$$
R _ { u , m } ( t ) = B \log _ { 2 } \Bigl ( 1 + \frac { P _ { u } h _ { u , m } ( t ) } { \sigma ^ { 2 } } \Bigr ) ,\tag{3}
$$

with user transmit power $P _ { u }$ and noise power $\sigma ^ { 2 }$ . Bandwidth contention among the users of one UAV is a deliberate simplification: the cost of adding a user to a UAV falls on the shared edge compute alone.

## C. Task and computation model

Task u is described by the tuple $( D _ { u } , C _ { u } , \tau _ { u } ^ { \mathrm { d l } } ) \mathrm { ; }$ : input size $D _ { u }$ bits, computing density $C _ { u }$ cycles/bit, and deadline $\tau _ { u } ^ { \mathrm { d l } }$ our instantiation holds $C _ { u }$ at a single value (Appendix A). The split $\rho _ { u }$ executes the two parts in parallel, the local part on the user’s own CPU at frequency $f _ { u } ^ { \mathrm { l o c } }$

$$
T _ { u } ^ { \mathrm { l o c } } ( t ) = \frac { \left( 1 - \rho _ { u } ( t ) \right) D _ { u } C _ { u } } { f _ { u } ^ { \mathrm { l o c } } } ,\tag{4}
$$

$$
T _ { u } ^ { \mathrm { o f f } } ( t ) = \frac { \rho _ { u } ( t ) D _ { u } } { R _ { u , m } ( t ) } + \frac { \rho _ { u } ( t ) D _ { u } C _ { u } } { f _ { m , u } ( t ) } ,\tag{5}
$$

with task completion $T _ { u } ( t ) = \mathrm { m a x } \{ T _ { u } ^ { \mathrm { l o c } } ( t ) , T _ { u } ^ { \mathrm { o f f } } ( t ) \}$ , where (5) sums uplink transmission and edge execution (result feedback is negligible, as standard). Tasks settle within their arrival slot (no cross-slot queueing in this model); a task missing its deadline counts as a violation. The per-slot delay increment $\Delta T ( t )$ sums completion times over active users plus a penalty χ per deadline violation, with χ sized so that penalties do not dominate the controllable part of the objective.

## D. Energy model

UAV propulsion uses the full rotary-wing model of [1],

$$
\begin{array} { r } { P ^ { \mathrm { f l y } } ( v ) = \underbrace { P _ { 0 } \bigg ( 1 + \frac { 3 v ^ { 2 } } { U _ { \mathrm { t i p } } ^ { 2 } } \bigg ) } _ { \mathrm { b l a d e ~ p r o f l e } } + \underbrace { P _ { i } \bigg ( \sqrt { 1 + \frac { v ^ { 4 } } { 4 v _ { 0 } ^ { 4 } } } - \frac { v ^ { 2 } } { 2 v _ { 0 } ^ { 2 } } \bigg ) ^ { 1 / 2 } } _ { \mathrm { i n d u c e d } } } \\ { + \underbrace { \frac { 1 } { 2 } d _ { 0 } \rho _ { a } s _ { r } A v ^ { 3 } } _ { \mathrm { p a r a s i t e } } , } \end{array}\tag{6}
$$

with the standard rotor and airframe constants of [1] $( P _ { 0 } { = } 7 9 . 9 \ : \mathrm { W }$ blade profile, $P _ { i } { = } 8 8 . 6 \mathrm { W }$ induced, $U _ { \mathrm { t i p } } { = } 1 2 0 \mathrm { m } / \mathrm { s }$ $v _ { 0 } { = } 4 . 0 3 \mathrm { m / s } .$ , and parasite coefficient $\scriptstyle { \frac { 1 } { 2 } } d _ { 0 } \rho _ { a } s _ { r } A = \mathrm { { \bar { 0 } } } . 0 2 )$ . Two properties matter. The parasite $v ^ { 3 }$ term: truncating (6) after two terms makes hovering and cruising nearly isoenergetic, destroying the energy–delay tension the problem is about. And the full curve’s interior minimum at $v ^ { * } { = } 8 .$ .4 m/s (hovering costs 25% more; 30 m/s costs 4.8× more): the physically optimal way to save energy is to fly at the valley and trade through offloading—and which methods find, hold, or abandon this point explains most of the performance ordering in our study (Sec. V-D). Edge computation obeys the standard cubic power law, at $E _ { m , u } ^ { \mathrm { c o m p } } ( t ) = \kappa _ { c } f _ { m , u } ( t ) ^ { 2 } \rho _ { u } ( t ) D _ { u } C _ { u }$ per offloaded task, with $\kappa _ { c }$ the effective switched capacitance of the edge processor, and the per-slot system energy increment $\Delta E ( t )$ sums flight power over the slot and edge-computation energy over each $\mathrm { U A V } _ { \mathrm { \Delta } }$ served users (UAV platform energy; user-side consumption is not an objective here).

## E. Constrained multi-objective scheduling problem

Collecting the models above, the offline design problem is

$$
( \mathrm { P 1 } ) \operatorname* { m i n } _ { \{ \mathbf { a } _ { t } \} _ { t = 1 } ^ { T } } \Big ( T _ { \mathrm { t o t } } , E _ { \mathrm { t o t } } \Big ) = \Big ( \sum _ { t = 1 } ^ { T } \Delta T ( t ) , \sum _ { t = 1 } ^ { T } \Delta E ( t ) \Big )\tag{7}
$$

$$
\mathrm { s . t . } ( 1 ) , \quad 0 \leq d _ { m } ( t ) \leq D _ { \operatorname* { m a x } } ,\tag{C1}
$$

$$
\| { \bf q } _ { m } ( t ) - { \bf q } _ { m ^ { \prime } } ( t ) \| \geq D _ { \mathrm { m i n } } , ~ \forall m \neq m ^ { \prime } ,\tag{C2}
$$

$$
\begin{array} { r } { \sum _ { m \in \mathcal { M } } x _ { u , m } ( t ) \leq 1 , x _ { u , m } ( t ) \in \{ 0 , 1 \} , } \end{array}\tag{C3}
$$

$$
\begin{array} { r } { 0 \leq \rho _ { u } ( t ) \leq \sum _ { m } x _ { u , m } ( t ) , } \end{array}\tag{C4}
$$

$$
\begin{array} { r } { \sum _ { u } x _ { u , m } ( t ) f _ { m , u } ( t ) \leq f _ { m } ^ { \operatorname* { m a x } } , } \end{array}\tag{C5}
$$

$$
\begin{array} { r } { \sum _ { t = 1 } ^ { T } E _ { m } ( t ) \leq E _ { m } ^ { \operatorname* { m a x } } , } \end{array}\tag{C6}
$$

where C1–C2 bound motion and enforce collision avoidance, C3–C4 couple offloading to association, C5 caps edge compute, and C6 is a long-term energy budget, $E _ { m } ( t )$ being UAV m’s energy spent in slot t and $E _ { m } ^ { \mathrm { m a x } }$ its onboard capacity. Deadlines enter softly, through the penalty $\chi .$ P1 is a non-convex mixed-integer nonlinear program whose singleobjective scalarizations are already NP-hard [6]: continuous flight and offloading variables couple with binary association through the non-convex rate (3) and the non-monotone propulsion curve (6).

C6 is where capability (B)’s budget contract lives in the mathematics, and it is the only constraint of P1 that spans the horizon: C1–C5 bind slot by slot, while C6 couples all slots through the running sum $\textstyle \sum _ { t ^ { \prime } < t } E _ { m } ( t ^ { \prime } )$ . No memoryless per-slot rule can enforce it, so honoring a budget requires the policy to carry its own cumulative consumption. The standard remedy has a cost of its own: a Lagrangian treatment bakes one budget into training and must retrain for the next.

Because the two objectives conflict with each other, P1 has no single minimizer but a Pareto set. From here on we negate the objectives and work with the return $\mathbf { G } = - ( T _ { \mathrm { t o t } } , E _ { \mathrm { t o t } } )$ so that larger is better throughout the paper. A return G dominates $\mathbf { G } ^ { \prime } \mathbf { \Sigma } ( \mathbf { G } \succ \mathbf { G } ^ { \prime } )$ iff $G _ { j } \geq G _ { j } ^ { \prime }$ for all j with at least one strict inequality; the Pareto front is

$$
\mathcal { P } = \big \{ \mathbf { G } \in \mathcal { G } : \exists \mathbf { G } ^ { \prime } \in \mathcal { G } , \mathbf { G } ^ { \prime } \succ \mathbf { G } \big \} ,\tag{8}
$$

with ${ \mathcal { G } } \subset \mathbb { R } ^ { n _ { o } }$ the attainable return set over $n _ { o }$ objectives. Our goal is not one point of $\mathcal { P }$ but a single policy able to reach any of them on request.

## F. Multi-objective MDP reformulation

P1 induces a multi-objective MDP $\langle S , \mathcal { A } , P , { \bf R } , \Omega , f , \gamma \rangle$ [10] with γ=1 over the finite horizon. The state $\mathbf { s } _ { t }$ gathers pernode indicators—each UAV’s position, residual energy, load and speed, and each active user’s UAV-centric position, task tuple, queue and channel indicators; the action is (2); the vector reward is the negative objective increment, $\begin{array} { r l } { \mathbf { r } _ { t } } & { { } = } \end{array}$ $- ( \Delta T ( t ) , \Delta E ( t ) ) \in \mathbb { R } ^ { n _ { o } }$ , so the expected vector return $\begin{array} { r } { \textbf G ^ { \pi } = \mathbb { E } _ { \pi } \big [ \sum _ { t = 1 } ^ { T } \mathbf { r } _ { t } \big ] } \end{array}$ equals $- ( T _ { \mathrm { t o t } } , E _ { \mathrm { t o t } } )$ in expectation. Preferences live on the simplex $\begin{array} { r } { \Omega = \{ \omega \in \mathbb { R } _ { \geq 0 } ^ { n _ { o } } : \sum _ { j } \omega _ { j } = 1 \} } \end{array}$ with linear utility $f ( \mathbf { G } , \omega ) = \omega ^ { \top } \mathbf { G }$ . Not every preference is physically expressible: the achievable front has ends, and preferences past them saturate rather than producing more extreme outcomes—which band is attainable is a property of the corpus, settled in Sec. III-C. Rather than solving one scalarized instance of P1 per $\omega ,$ we seek a single conditional policy

$$
\begin{array} { r } { \pi _ { \boldsymbol { \theta } } ( \mathbf { a } _ { t } \mid \mathbf { s } _ { 1 : t } , \omega ) , \qquad \omega \in \Omega , } \end{array}\tag{9}
$$

whose induced return approaches the Pareto-optimal return at every attainable preference. The aliasing regime of Prop. 2 begins at three objectives and is outside this paper’s scope.

## G. Why not scalarize first

Proposition 1 (Convex-hull limitation): Let ${ \mathcal { G } } \subset \mathbb { R } ^ { n _ { o } }$ be the attainable return set. For any $\omega$ in the simplex, maximizers of $\omega ^ { \top } \mathbf G$ over $\mathcal { G }$ lie on the boundary of conv(G); hence unsupported Pareto-optimal points—those not on the convex hull—are unattainable by a priori linear scalarization for every ω.

Proof. Linear objectives attain their maxima over $\mathcal { G }$ and over conv(G) at the same value, and over a compact convex set at supporting points of the boundary. A point strictly inside the hull’s Pareto face has, for every ω, a strictly larger-utility supporting point, so it is never a maximizer. See also [2]. ■ Discrete offloading choices in (2) and the non-monotone propulsion power curve (6) make G non-convex in this problem, so this is not a hypothetical concern. PrefDT sidesteps it because it never scalarizes at training time: it imitates preference-labeled behavior directly, so its reachable set is bounded by corpus coverage, not by hull geometry. The same design keeps training offline: the entire pipeline consumes preference-annotated trajectories, whereas the offline search that produces the rule those trajectories come from (the teacher of Sec. III-C) costs 606,000 slots of live interaction, and a scalarized learner pays a live (re)training per preference. Sec. V-C does the cost accounting and states the boundary: what offline training removes is live interaction at deployment, not the value of a competent teacher having existed.

## III. THE PROPOSED PREFDT SCHEDULER

One decision organizes this section: the setting is an input the policy reads at run time, not a term compiled into what the policy was trained to maximize. The pipeline that results runs at two time scales, and Fig. 1 separates them. One scale runs once, before the fleet flies, and ends in a frozen artifact. The other runs in every slot and updates no weights.

The offline scale builds the corpus and trains on it. A teacher’s rule family is searched once, and each of its rollouts is paired with the setting whose outcome that rollout realized; the pairs are the training corpus. A causal sequence model is trained on the corpus by imitation, and a regressor is fitted beside it that turns a setting into the return the corpus attains there. Training then stops. The weights, the regressor and the per-objective scales are archived together, and none of them changes again.

The per-slot scale is a loop. At the start of a flight the requested setting becomes a return target, and a requested energy budget replaces that target’s energy entry. In each slot the network state becomes one token per UAV and one per active user, and each UAV pools the active users into a summary of fixed width. The summary, the preference and the current return target enter the token stream. The sequence model reads the most recent slots of that stream and returns one output per slot, from which the flight controls are decoded; each user’s association and offload fraction are decoded from that output together with that user’s own token. The environment executes the joint action and returns the delay and energy actually consumed in that slot. Both values are subtracted from the return target before the next slot starts. One forward pass per slot, with nothing solved or searched, and no weight updated.

![](images/fafcf43ae1fe55bd5cf5b716ad8997315b79d1ef573f23f9a41516920c8daacc.jpg)  
Fig. 1. The structure of PrefDT. The pipeline runs at two time scales. At run time and in every slot (top): the network state is read as one token per UAV and one per active user; each UAV pools every active user into one summary, and each user’s own features bypass the pooling to rejoin the Transformer output (⊕) for that user’s decisions. A frozen causal Transformer reads the stream of return, state and action tokens and emits the slot’s decisions, which the network executes; the return token’s two entries, the delay target set by the requested trade-off and the energy budget, are decremented by the delay and energy the environment reports back. No weight changes at run time. Offline and once (bottom): the teacher, an archive of scheduling rules tuned by NSGA-II, is flown to record 50,000 flights, each labelled with the trade-off it struck and the delay and energy it cost; the Transformer is trained to imitate these flights and is then frozen into the model that runs above.

## A. The state encoder: tokens, pooling, and the per-user path

The active set K(t) changes size every slot, while a sequence model consumes a fixed layout of inputs. The association and offloading of (2) are decided per member, so a description of the fleet as a whole is not sufficient. The encoder must accept a set of any size and still return a decision for every member.

The encoder therefore has two paths. One condenses $\kappa ( t )$ into a summary of fixed width. That summary is symmetric in its members and therefore retains no identity, so no permember decision can be read from it. The other path supplies each member’s features to the heads that decide for it, without passing through the summary.

One frozen model then serves an active set of any size, and still decides per member. Changing the number of users requires no retraining, because no part of the network is sized by that number. The summary is defined on any subset of K(t), so a report lost on the uplink leaves a smaller set rather than an incomplete input. The scheduler therefore keeps deciding on a link that drops reports, with no substitute value invented for the members that are missing.

Each slot is encoded as one token per node. A UAV token holds that aircraft’s normalized position, residual energy, number of served users and speed. A user token holds that user’s position in the serving UAV’s polar frame, its task tuple, its queue indicator, its channel power gain, and a block $\tilde { \psi } _ { u }$ of four closed-form features,

$$
\begin{array} { r l } & { \mathbf { z } _ { m } ^ { \mathrm { u a v } } ( t ) = \big [ \tilde { \mathbf { q } } _ { m } ( t ) , ~ \tilde { e } _ { m } ^ { \mathrm { r e s } } ( t ) , ~ | \mathcal { U } _ { m } ( t ) | , ~ \tilde { v } _ { m } ( t ) \big ] , \quad ( \mathrm { l } } \\ & { \mathbf { z } _ { u } ^ { \mathrm { u s r } } ( t ) = \big [ \tilde { r } _ { u , m } , ~ \tilde { \phi } _ { u , m } , ~ \tilde { D } _ { u } , ~ \tilde { C } _ { u } , ~ \tilde { \tau } _ { u } ^ { \mathrm { d l } } , ~ \tilde { q } _ { u } , ~ \tilde { h } _ { u , m } , ~ \tilde { \psi } _ { u } \big ] , } \end{array}\tag{10}
$$

(11)

where every entry is standardized by statistics archived with the training set, $\tilde { x } = ( x - \mu _ { \mathcal { D } } ) / \sigma _ { \mathcal { D } }$ , so that training and rollout use identical scales. The UAV tokens number M in every slot, while the user tokens number $| \mathcal { K } ( t ) |$ and therefore change in

count from slot to slot.

The four features in $\psi _ { u }$ are the two latencies that decide user u’s current task and their two decision-relevant ratios,

$$
\psi _ { u } = \big [ T _ { u } ^ { \mathrm { l o c } } , ~ T _ { u } ^ { \mathrm { o f f } } , ~ T _ { u } ^ { \mathrm { l o c } } / \tau _ { u } ^ { \mathrm { d l } } , ~ T _ { u } ^ { \mathrm { o f f } } / T _ { u } ^ { \mathrm { l o c } } \big ] ,\tag{12}
$$

with $T _ { u } ^ { \mathrm { l o c } }$ and $T _ { u } ^ { \mathrm { o f f } }$ the local and offloading latencies of $( 4 ) \AA { - } ( 5 )$ evaluated at the current channel state and edge load. Both are closed-form functions of quantities the state already contains: whatever the physical models compute in closed form, the policy is not required to learn.

Each UAV condenses the active users into a single summary by masked cross-attention. For each of $H _ { a }$ heads of width $d _ { h } ,$ the query $\mathbf { q } _ { m } ^ { ( i ) }$ is a linear projection of the UAV token, and the key $\mathbf { \dot { k } } _ { u } ^ { ( i ) }$ and the value $\mathbf { v } _ { u } ^ { ( i ) }$ are linear projections of the user token; the attention weights are

$$
\alpha _ { u , m } ^ { ( i ) } ( t ) = \frac { \exp \bigl ( \langle \mathbf { q } _ { m } ^ { ( i ) } , \mathbf { k } _ { u } ^ { ( i ) } \rangle / \sqrt { d _ { h } } + \mu _ { u } ( t ) \bigr ) } { \sum _ { u ^ { \prime } } \exp \bigl ( \langle \mathbf { q } _ { m } ^ { ( i ) } , \mathbf { k } _ { u ^ { \prime } } ^ { ( i ) } \rangle / \sqrt { d _ { h } } + \mu _ { u ^ { \prime } } ( t ) \bigr ) } ,\tag{13}
$$

with the mask $\mu _ { u } ( t ) = 0$ for $u \in \mathcal { K } ( t )$ and −∞ otherwise, so that inactive users receive exactly zero weight. Summing the values under these weights and projecting the concatenated heads gives the per-UAV summary $\mathbf { p } _ { m } ( t )$ , and the encoder’s representation of the slot is

$$
s _ { t } = \left[ \mathbf { z } _ { 1 } ^ { \mathrm { u a v } } ; \mathbf { p } _ { 1 } ( t ) ; \mathbf { \beta } \cdot \cdot \cdot ; \mathbf { z } _ { M } ^ { \mathrm { u a v } } ; \mathbf { p } _ { M } ( t ) \right] .\tag{14}
$$

This representation has one width for every value of $| \mathcal { K } ( t ) |$ and it contains no padded positions. Each summary is a weighted sum over the user tokens, so it is unchanged by any permutation π of the active users,

$$
\mathbf { p } _ { m } \bigl ( \{ \mathbf { z } _ { \pi ( u ) } ^ { \mathrm { u s r } } \} \bigr ) = \mathbf { p } _ { m } \bigl ( \{ \mathbf { z } _ { u } ^ { \mathrm { u s r } } \} \bigr ) ,\tag{15}
$$

and carries no information about which user occupies which index.

Each user’s own token is therefore supplied directly to that user’s two decision heads, concatenated with the sequence model’s output $\mathbf { e } _ { t }$ at slot t. Writing $x _ { u } ( t ) \in \{ 0 , 1 , \ldots , M \}$ for user $u \mathrm { { s } }$ categorical association, with 0 denoting local execution,

$$
\begin{array} { r } { \boldsymbol { \lambda } _ { u } ( t ) = \operatorname { M L P } _ { \boldsymbol { c } } \big ( [ \mathbf { e } _ { t } ; ~ \mathbf { z } _ { u } ^ { \mathrm { u s r } } ( t ) ] \big ) \in \mathbb { R } ^ { M + 1 } , } \end{array}\tag{16}
$$

$$
\begin{array} { r } { \hat { \rho } _ { u } ( t ) = \frac { 1 } { 2 } \Big ( 1 + \operatorname { t a n h } \operatorname { M L P } _ { \rho } \big ( [ { \mathbf { e } } _ { t } ; ~ { \mathbf { z } } _ { u } ^ { \operatorname { u s r } } ( t ) ] \big ) \Big ) , } \end{array}\tag{17}
$$

while the flight controls are decoded from $\mathbf { e } _ { t }$ alone, $\hat { \mathbf { a } } _ { t } ^ { \mathrm { { f l y } } } =$ tanh $\mathrm { M L P } _ { \mathrm { f } } ( \mathbf { e } _ { t } )$ . Permuting the active users permutes the outputs of (16)–(17) in the same order, so this second path is equivariant.

## B. The return channel: the vector return-to-go, its decrement, and the sequence model

This subsection builds the value that carries the setting into the policy. Two kinds of request arrive through it. One is the energy–delay trade-off. The other is a total energy budget in joules. Both are chosen before the flight and cannot be derived by the policy from the network state, so both have to be supplied as an input the policy reads. The second one also has to change as the fleet spends.

The value is a vector with one entry per objective rather than a single total. Each entry is the return still to be collected on that objective, so the energy entry can be initialized at the requested budget and reduced at every slot by what that slot consumed. The sequence model reads a window of slots rather than a single slot, so it sees that reduction as it happens. A single total does neither of these things. Under three or more objectives it names a whole family of front points at once, so it cannot say which of them is being asked for. And at any number of objectives, including two, no part of it is a quantity in joules, so a budget can neither be requested through it nor reduced inside it.

Every setting therefore names a single outcome the fleet can produce. A total energy budget can be requested in joules and enforced against what the fleet has actually spent: an overspend enters the value the policy reads one slot after it begins, with nothing re-issued and no outer loop.

The conditioning value at slot t is the per-objective returnto-go, the suffix sum $\begin{array} { r } { \mathbf { g } _ { t } = \sum _ { t ^ { \prime } > t } \mathbf { r } _ { t ^ { \prime } } \in \mathbb { R } ^ { n _ { o } } } \end{array}$ , the componentwise analogue of DT’s scalar return-to-go [3], [4]. Raw delay and energy returns differ by about two orders of magnitude, so each entry is rescaled by a constant of its own,

$$
\hat { g } _ { t , j } = \frac { g _ { t , j } } { \eta _ { j } } , \qquad \eta _ { j } = \operatorname* { m a x } _ { \mathcal { D } } g _ { 1 , j } - \operatorname* { m i n } _ { \mathcal { D } } g _ { 1 , j } ,\tag{18}
$$

where η collects the constants and $\oslash$ denotes element-wise division. The rescaling is a pure ratio: no shift is subtracted, and no normalization layer follows the projection. The constants are per-objective, fixed, and archived with the dataset, so that training and rollout use identical scales.

The preference enters the token stream at three sites,

$$
\begin{array} { r } { s _ { t } ^ { * } = s _ { t } \oplus \omega , \qquad a _ { t } ^ { * } = a _ { t } \oplus \omega , \qquad g _ { t } ^ { * } = \hat { \mathbf { g } } _ { t } \odot \omega , } \end{array}\tag{19}
$$

by concatenation to the state and action tokens and by elementwise weighting of the normalized return-to-go [4]. Concatenation before any layer makes the preference part of every position’s representation, rather than a separate token that attention may or may not route to. The mechanism is PEDA’s, unchanged; what PrefDT adds on top of it is the pipeline of Sec. III-C, and Sec. V-D removes each addition singly.

Trajectories are serialized in DT’s returns-first token order $( g _ { t } ^ { * } , s _ { t } ^ { * } , a _ { t } ^ { * } )$ [3], so a context of $K _ { \mathrm { c t x } }$ slots spans $3 K _ { \mathrm { c t x } }$ tokens, each summed with a learned per-slot timestep embedding. A causal GPT backbone (3 layers, 4 heads, width 256, $K _ { \mathrm { c t x } } { = } 2 0 )$ predicts $\mathbf { a } _ { t }$ at the position of $s _ { t } ^ { * }$ , and its output at that position is the $\mathbf { e } _ { t }$ of (16)–(17).

Training minimizes DT’s standard mixed imitation loss [3]: L2 on the continuous actions, the flight controls and the offload ratios, and cross-entropy on the per-user association, equally weighted. Actions are mapped to $[ - 1 , 1 ]$ beforehand by the standard rescaling $\tilde { \mathbf { a } } = 2 ( \mathbf { a } - \mathbf { a } _ { \mathrm { m i n } } ) / ( \mathbf { a } _ { \mathrm { m a x } } - \mathbf { a } _ { \mathrm { m i n } } ) - 1$ . AdamW at $1 0 ^ { - 4 }$ with warmup, batch 256; convergence in 40k steps. The policy head is deterministic by default, and sampling the association head is an inference-time option rather than a training change: it draws from the existing softmax, and no parameter is refit.

Across the slots of one context the return-to-go decrements by the reward each slot realized,

$$
\hat { \mathbf { g } } _ { t + 1 } = \hat { \mathbf { g } } _ { t } - \mathbf { r } _ { t } \oslash \pmb { \eta } ,\tag{20}
$$

so an energy entry requested at a budget, $\hat { g } _ { 1 , E } = - B / \eta _ { E }$ stands after t slots at

$$
\hat { g } _ { t , E } = - \frac { { B } - \sum _ { t ^ { \prime } < t } \Delta E ( t ^ { \prime } ) } { \eta _ { E } } = - \frac { { B } - \mathrm { c u m } E ( t ) } { \eta _ { E } } .\tag{21}
$$

The energy entry of the conditioning vector is the budget not yet spent, up to the fixed scale $\eta _ { E }$ . A revision of the total to $B ^ { \prime }$ is a single assignment to (21), and requires no further change to the loop.

How many entries the conditioning value needs is set by the dimension of the front.

Proposition 2 (Utility aliasing): Let $\mathcal { P } \subset \mathbb { R } ^ { n _ { o } }$ be a compact $( n _ { o } - 1 )$ -manifold and let ω have strictly positive entries. The level set $\{ \mathbf { g } : { \boldsymbol { \omega } } ^ { \top } \mathbf { g } = c \}$ meets $\mathcal { P }$ transversally in a set of dimension $n _ { o } - 2 \colon$ a point for $n _ { o } = 2 .$ , and a positive-dimensional family for $n _ { o } \geq 3$

The proof is a transversality argument on the dimension of the intersection; the sketch is in Appendix B. One entry therefore names a single front point at two objectives and a whole family of them at three or more, while $n _ { o }$ entries name one point at every $n _ { o }$

Two further differences hold at every $n _ { o } ,$ including the $n _ { o } = 2$ case the proposition leaves sufficient. A scalar utility has no component that is the energy budget, so the identity (21) is available to the vector form alone. And the vector form supplies $n _ { o }$ supervision targets per timestep where the scalar supplies one, with the decrement (20) propagating perobjective rather than aggregate progress.

## C. The distillation pipeline: corpus construction and inference

Conditional imitation needs recorded runs labeled with the outcome each achieved, and scheduling supplies none: no archive of flights exists in which each flight carries the tradeoff it struck. The corpus therefore has to be produced. Which flight is stored under which setting is what a setting comes to mean, and no loss function decides that afterwards.

The corpus is distilled from a teacher: a compact physical rule whose ten parameters are searched once, offline, and whose rollouts become the training data. A policy trained by imitation can only produce behaviors its corpus contains, so the teacher sets the ceiling, and choosing it is part of the algorithm rather than a tuning detail. That choice is made on physics: the cheapest way to save energy is to hold the speed at the propulsion power minimum and trade through partial offloading, so a teacher qualifies only if its members sweep the front that way. Each setting is then paired with the member whose realized cost composition matches it, which is what makes a setting an outcome share rather than an importance weight over a training reward. Settings outside the band that family spans are refused before any data is generated, and the extreme members are demonstrated more heavily than uniform sampling would demonstrate them, because imitation regresses toward the behavioral mean.

The search is paid once. Afterwards a frozen model answers every setting the archive could have answered, and three requests the archive could not: a point between two of its members, a total energy budget, and a revision of that budget in flight. The setting axis is calibrated onto the front the trained model actually achieves, so distinct settings name distinct attainable outcomes in an exact order and at the requested density. A denser front costs further rollouts and no further search.

The shipped corpus is distilled from the teacher’s archive: rollouts of the NSGA-II-searched rule family of Sec. IV-B, whose members hold $v ^ { * }$ throughout and sweep the front through the offload ratio. The archive also fixes which preferences are physically expressible. With $\hat { \bf G } ( \omega )$ the return the family realizes at $\omega ,$ the attainable band is

$$
\Omega ^ { \ast } = \Big \{ \omega \in \Omega : \varsigma _ { 1 } \big ( \hat { \mathbf { G } } ( \omega ) \big ) \in \big [ 0 . 0 5 2 , 0 . 6 7 6 \big ] \Big \} ,\tag{22}
$$

where $\varsigma _ { 1 } ( \cdot )$ is the delay share of the normalized return and the interval is the span the family covers. Preferences outside $\Omega ^ { * }$ are rejected at data generation, so the corpus never annotates a setting the physics cannot express.

Dataset generation follows the D4MORL recipe [4] in three steps: sample preferences from Dirichlet distributions at three entropy tiers $( \alpha \in \{ 1 , 3 , 8 \}$ , one third of the draws each) and reject ω $\notin \Omega ^ { * }$ , the band of (22) widened by a tolerance of $0 . 0 5 ;$ match each accepted ω to the archive member whose realized return share is nearest,

$$
i ^ { \star } ( \omega ) = \arg \operatorname* { m i n } _ { i } \ \left\| \varsigma \big ( \mathbf { G } ^ { \pi _ { i } } \big ) - \varsigma \big ( \hat { \mathbf { G } } ( \omega ) \big ) \right\| _ { 2 } ;\tag{23}
$$

and roll out $\pi _ { i ^ { \star } }$ , storing the preference-annotated trajectory— 50,000 trajectories in all. Because the match is on realized share, ω denotes an outcome composition: a front point whose normalized cost is $\omega _ { 1 }$ parts delay. Two refinements act at the extremes. Corner emphasis redirects a quarter of the draws into the share window of the archive’s four lowest-delay members. Quality diversity mixes archive-member and scripted-heuristic rollouts in an 80/20 ratio, the heuristic pool spanning three fixed modes, which varies outcome quality at fixed ω.

Three maps stand between a requested setting and the fleet’s behavior: a conditioner $\omega \mapsto \hat { \mathbf { g } } _ { 1 }$ , a remap $c \mapsto \omega$ , and the decrement loop that carries $\hat { \bf g } _ { 1 }$ through the horizon. The conditioner is a preference-to-return regressor fitted offline on the corpus’s clean expert episodes and frozen with the weights,

$$
f _ { \mathrm { r e g } } = \arg \operatorname* { m i n } _ { \mathbf { A } } \sum _ { ( \omega _ { i } , \mathbf { G } _ { i } ) \in \mathcal { D } _ { \mathrm { e x p } } } \big \| \mathbf { G } _ { i } - \mathbf { A } \phi ( \omega _ { i } ) \big \| _ { 2 } ^ { 2 } ,\tag{24}
$$

with quadratic features $\phi .$ Its targets are returns the corpus contains at that preference, never an ideal point, which lies outside the attainable set.

Two gates run before any training: the binned map $\omega \mapsto$ $\mathbb { E } [ \mathbf { G } | \boldsymbol { \omega } ]$ is checked for monotonicity, and (24) is checked for fit. Each conditioner is fitted per corpus and archived with it, and a corpus build that would overwrite another corpus’s conditioner aborts by construction.

The remap is fitted after training. It is a monotone map $c \mapsto$ $\omega ( c )$ carrying an abstract setting $c \in [ 0 , 1 ]$ onto the measured, non-dominated portion of the front the trained model achieves, and it is pure inference: no weight, conditioner or checkpoint changes.

At rollout the normalized return-to-go initializes from the conditioner,

$$
\hat { \mathbf { g } } _ { 1 } = f _ { \mathrm { r e g } } ( \omega ) \oslash \pmb \eta ,\tag{25}
$$

and from there decrements by (20); actions decode deterministically. Algorithm 1 composes the three maps into one sweep. It takes the frozen $\pi _ { \theta } .$ , a setting grid $\mathcal { C } \subset [ 0 , 1 ]$ , the remap, the conditioner and the scales $\mathbf { \eta } _ { \eta } ,$ and returns the non-dominated subset of the returns the sweep realizes. Its outer loop turns one setting into one target: the remap and the conditioner are applied once, and a requested budget is written into $\hat { g } _ { E }$ in place of the conditioner’s energy target. Its inner loop carries that target through the horizon: one forward pass per slot, the decrement of (20), and the same $\hat { g } _ { E }$ open to a mid-flight revision. A K-point front therefore costs KT forward passes against one frozen $\pi _ { \theta }$

Algorithm 1 PrefDT front generation with a recalibrated   
setting axis (generalizing DT’s return-conditioned evaluation   
loop [3])   
Require: frozen $\pi _ { \boldsymbol { \theta } } ;$ setting grid $\mathcal { C } \subset [ 0 , 1 ] ;$ remap $c \mapsto \omega ( c )$   
(fitted on calibration seeds); conditioner $f _ { \mathrm { r e g } } ;$ scales $\eta$   
1: for $c \in { \mathcal { C } }$ do   
2: $\omega  \omega ( c ) ; \hat { \mathrm { ~ \bf ~ g ~ } }  f _ { \mathrm { r e g } } ( \omega ) \oslash \eta$ (budget override:   
gˆ<sub>E</sub> $ - B / \eta _ { E } ) ;$ reset env; $\xi  \emptyset$   
3: for $t = 1 , \dots , T$ do   
4: build $( g _ { t } ^ { * } , s _ { t } ^ { * } )$ from (19) with $\hat { \bf g } _ { t } = \hat { \bf g } ;$ append to $\xi$   
5: $\mathbf { a } _ { t } \gets \pi _ { \theta } \big ( \cdot \mid \xi _ { t - K _ { \mathrm { c t x } } + 1 : t } , \omega \big ) ;$ ; step $\mathbf { e n v }  \mathbf { r } _ { t }$   
6: $\hat { \mathbf { g } }  \hat { \mathbf { g } } - \mathbf { r } _ { t } \oslash \pmb { \eta }$ (revision = overwrite gˆ<sub>E</sub>; (21))   
7: end for   
8: record $\begin{array} { r } { \mathbf G ( c ) = \sum _ { t } \mathbf r _ { t } } \end{array}$   
9: end for   
10: return non-dominated subset (8) of $\{ \mathbf G ( c ) \} _ { c \in \mathcal { C } }$

## IV. EXPERIMENTAL SETUP

## A. Scenario

The environment implements Sec. II: M=2 UAVs at H=100 m over a 1000×1000 m area, U=10 users of whom four to eight are active in any slot, and T=100 slots of 1 s. Tasks, deadlines, compute rates, partial offloading, rotary-wing propulsion and the probabilistic LoS air–ground channel are as modelled there, with every constant listed in Appendix A. All methods share this environment and identical evaluation seeds, with one exception: the degradation study of Sec. V-B adds a mid-episode propulsion multiplier, pinned so that it cannot have perturbed anything else (Sec. IV-D).

## B. Methods under comparison: three declared classes

Twenty-six method variants are evaluated under one shared protocol. The comparison set spans methods with incompatible access assumptions and roles, so we declare three classes before any of their numbers are quoted (Table I). Table II reports the variants the main text reads; Appendix C lists every variant with its configuration.

Class 1 is the class the front-quality claim is scoped to. ω-conditioned IQL is trained on our own corpus, so that the comparison isolates architecture from data, and the Lagrangian CMDP is retrained once per budget level.

Class 2 is instantiated over our own channel and propulsion models, at the corpus’s per-objective scales. The SCA oracle is restarted from hover, from the demand centroid, and from jittered variants of each, since BCD is local and a bound is only a bound if it was given a fair search; SCA-MPC’s iteration budget is deliberately smaller than the oracle’s, because an MPC given the oracle’s budget would not be the method this baseline exists to represent. Neither can take a setting: a new preference is a fresh solve rather than a fresh token.

TABLE I  
THE THREE DECLARED CLASSES OF THE COMPARISON SET.
<table><tr><td>Class</td><td>Members</td><td></td><td>Provenance</td><td>A new trade-off costs</td></tr><tr><td>1 Learned sched- ulers</td><td>PrefDT;  $\partial \omega \mathbf { - B } { \cal C } ;$  Lagrangian CMDP [9]; single-</td><td>problem; per-preference PPO; by us unconditioned DT;</td><td>IQL [24]; PGMORL [8]; methods run by conditioned; one PEDA [4] ported to this us; the rest built retraining</td><td>ω-conditioned Three published One rollout if per preference or per budget otherwise</td></tr><tr><td>2 Model- based refer- ence</td><td>SCA oracle, supplied the full Literature future arrival sequence; SCA- method [5], [6], per setting MPC over a five-slot window our instantiation</td><td></td><td></td><td>One fresh solve</td></tr><tr><td>bounds 3 tillation teacher</td><td>Dis- Teacher: NSGA-II [23] over Search operator One re-search; our ten-parameter rule; the published, source of the training corpus</td><td>ours</td><td></td><td>rule front size is the population</td></tr></table>

Class 3 exists because its rule is our own composition. No published UAV-MEC baseline exposes a search space small enough for a 2000-evaluation budget, so we composed one from standard elements of this literature—a latencycomparison offload gate, a service-radius association rule, and centroid-tracking motion (four global genes plus three per-UAV motion gains, ten in all)—and searched it with NSGA-II, whose published contribution is the search operator rather than the rule.

## C. Metrics and protocol

Front quality is hypervolume (HV [19]) with IGD, both perseed against a reference point and reference front shared by every method compared.

Front quality does not measure whether settings are followed—a scheduler can produce an excellent front while ignoring every request—so the conditioning channel carries its own pair of readings. A setting is followed when different settings produce different outcomes, and the outcomes arrive in the requested order. Fidelity measures the order: per seed, the rank correlation $| \rho |$ between the settings and the front positions they produce, with |ρ|=1 meaning every outcome arrives exactly in the requested order. Fidelity alone is insufficient, because perfect ordering can be held over a narrow segment of the front. Reach closes that gap: the span, in seconds of delay, of the outcomes the settings produce on one seed. Reach alone is insufficient in turn, because a wide span can be produced with no order at all. Both defects occur among the systems evaluated in Sec. V, so the two readings are used only together, against a criterion fixed before the decisive experiments:

$$
\mathrm { o p e r a b l e } \longleftrightarrow \vert \rho \vert \geq 0 . 9 \wedge \mathrm { r e a c h } \geq 0 . 8 \cdot W _ { \mathrm { t e a c h e r } } ,\tag{26}
$$

where $W _ { \mathrm { t e a c h e r } }$ is the span that the teacher’s archive itself attains per seed. The requirement therefore reads: near-perfect ordering, over at least 80% of the span the teacher covers— 493 s in this study. Both clauses are per-seed statistics, because mixing a pooled threshold with a per-seed statistic biases the test. Neither constant was chosen after the results were seen; the pair is this study’s operating point, not a proposed universal. |ρ| and reach are reported throughout on the full uniform-ω grid; what the recalibrated setting axis of Sec. III-C adds is reported in Sec. V-D.

A budget’s violation rate is read with its tracking error, since a policy that ignores its setting downward scores a flattering violation rate. Every inferential comparison uses 100 paired evaluation seeds and a two-sided Wilcoxon signedrank test with the exact null distribution, computed by a subset-sum dynamic program, since Monte-Carlo nulls cannot resolve the 10<sup>−7</sup>-scale p-values a paired design at this sample produces; relative effects carry bootstrap confidence intervals. Appendix C gives the machinery and the controls behind it.

## D. Experiment configurations

How every experiment in Sec. V was run—its grid, its perturbation and its seed count—is tabulated in Appendix C of the supplementary material; seven conventions hold throughout and are not repeated there. An ablation toggles one component at a time, with architecture, training schedule and seeds otherwise fixed. Readings on the PPO-corpus configuration and on the archive corpus, which is the shipped path, are never differenced against each other. Every arm is evaluated on a frozen checkpoint with no retraining. Inferential rows are read at n=100; descriptive rows carry the n the table prints, and the two are never differenced against each other—where a headline comparison needs a row that Table II prints as descriptive, that row is re-evaluated at n=100 first. Calibration and evaluation seeds are disjoint wherever a fitted component—the recalibration remap—could otherwise be scored on its own fitting data. Hypervolume is read against one shared reference point and pooled reference front, recomputed whenever the method set changes; the reference file carries a hash of the contributing method set, so silent drift is structurally impossible. And the degradation hook of Sec. V-B reproduces the unmodified simulator bit for bit at multiplier 1.0—a test on exact array equality, not a tolerance—so the one study that perturbs the environment cannot have moved a number in any other. The cost study alone is not evaluated on the shared grid: it is timed on one Apple M1 core with OMP\_NUM\_THREADS, MKL\_NUM\_THREADS and VECLIB\_NUM\_THREADS set to 1, the routes run one after another and the machine otherwise idle, the unit being one setting—one episode of T slots for PrefDT and for both solvers, one 2M-step training run for the scalarized learner, one complete search for the teacher—so that a K-point front is K times the per-setting value and is never timed as a whole. Everything else ran on one rented cloud instance (one NVIDIA RTX 4090D, 24 GB, on a 24-vCPU allocation of a 256-core host); the Apple M1 machine served development, smoke tests, and the timing study alone.

## V. EXPERIMENTAL EVALUATION

A. Front quality, fidelity, and reach: comparing the field and selecting from it

Three quantities decide a scheduler here, and no one of them decides it alone. Front quality says how good the tradeoff curve is; fidelity says whether the outcomes arrive in the order the settings requested; reach says over how much of the curve that order holds. Table II reads every method in the study on all three. Read together, PrefDT stands best: it produces the best trade-off curve of any learned method, and on the corpus it ships with it is the only learned method that clears both clauses of (26).

TABLE II  
METHODS UNDER ONE SHARED PROTOCOL, GROUPED BY THE CLASSES OF SEC. IV-B. HV (×10<sup>6</sup>, MEAN±STD) AND IGD ARE READ AGAINST A REFERENCE POINT AND REFERENCE FRONT SHARED BY EVERY ROW; LOWER IGD IS BETTER. |ρ| AND REACH ARE CO-REPORTED PER SEC. IV-C, AND THE LAST COLUMN READS EACH ROW AGAINST THE PRE-REGISTERED CRITERION (26) AND NAMES THE CLAUSE IT FAILS.
<table><tr><td>Method</td><td>HV</td><td>IGD</td><td>|ρ|</td><td>Reach</td><td>n</td><td>(26)</td></tr><tr><td colspan="7">Class 1: learned, preference-conditioned</td></tr><tr><td>PrefDT (shipped) te</td><td>99.47±1.18</td><td>402</td><td>0.93</td><td>601</td><td>100</td><td>yes</td></tr><tr><td>PrefDT (PPO corpus)†</td><td>90.28±2.48</td><td>1457</td><td>0.91</td><td>540</td><td>100</td><td>yes</td></tr><tr><td>PEDA, as published†9</td><td>82.87±2.17</td><td>1594</td><td>0.76</td><td>398</td><td>100</td><td>no (both)</td></tr><tr><td>PEDA + attention pooling †g</td><td>84.18±1.98</td><td>1298</td><td>0.68</td><td>491</td><td>100</td><td>no (both)</td></tr><tr><td>IQL+ω (archive corpus)c</td><td>96.23±1.82</td><td>1587</td><td>0.94</td><td>413</td><td>10</td><td>no (reach)</td></tr><tr><td>IQL+ω (PPO corpus)c</td><td>91.83±1.48</td><td>1221</td><td>0.95</td><td>528</td><td>10</td><td>yes</td></tr><tr><td>PGMORL</td><td>87.34±1.23</td><td>4501</td><td>0.84</td><td>15</td><td>10</td><td>no (both)</td></tr><tr><td>Per-preference PPO†</td><td>90.72±1.54</td><td>1369</td><td>0.95</td><td>396</td><td>100</td><td>no (reach)</td></tr><tr><td>ω-BC†</td><td>89.23±2.88</td><td>1237</td><td>0.49</td><td>458</td><td>100</td><td>no (both)</td></tr><tr><td>PrefDT, ideal-point targets</td><td>90.21±2.21</td><td>1475</td><td>0.98</td><td>170</td><td>10</td><td>no (reach)</td></tr><tr><td colspan="7">Class 2: model-based reference bounds</td></tr><tr><td>SCA oracleª</td><td>99.91±1.25</td><td>314</td><td>0.88</td><td>607</td><td>10</td><td>no (|ρ|)</td></tr><tr><td>SCA-MPCb</td><td>99.94±1.27</td><td>597</td><td>0.93</td><td>606</td><td>10</td><td>yesf</td></tr><tr><td colspan="7">Class 3: distillation teacher (ours; corpus source)</td></tr><tr><td>Teacherd</td><td>99.80±1.31</td><td>351</td><td></td><td>616</td><td>10</td><td></td></tr><tr><td colspan="7">Floors</td></tr><tr><td>Unconditioned DT</td><td>79.96±5.26</td><td>3162</td><td></td><td>100</td><td>10</td><td></td></tr><tr><td>Random</td><td>62.40</td><td>18529</td><td></td><td>193</td><td>1</td><td></td></tr></table>

Reach is defined for any method that offers more than one front point; |ρ| is a correlation between requested settings and the outcomes they produce and is undefined (—) for a method that takes no preference input.  
1 Inferential rows, n=100 paired seeds; the others are descriptive, n=10.  
Non-causal: supplied the full future arrival sequence.  
b Not real-time schedulable as implemented (Sec. V-C; qualified in Sec. VI).  
Printed descriptive, but this row’s verdict and the comparison of Sec. V-D are read at n=100 (Appendix C).  
The search operator is published [23]; the rule it searches is our own composition, and this row is the source of the training corpus.  
Scored against an archived reference pool that does not contain it (Appendix C).  
Meets the numeric criterion, but its setting is a fresh solve per point rather than a run-time input (note <sup>b</sup>).  
g PEDA [4] ported to this problem on the PPO corpus with its own conditioning kept: a fixed-dimension input by zero padding and no per-user path; the second row adds the pooled encoder of Sec. III-A and still omits the per-user path.

Three configurations reach higher front quality, and the three quantities do not all read for them. The teacher offers the widest span in the study, 616 s—the reference the reach clause is set against, since 0.8 × 616 is the required 493—and no fidelity, since fidelity is a correlation between requested settings and the outcomes they produce and the teacher’s archive is not requested. The non-causal oracle reads on both and fails the ordering clause, at |ρ|=0.88. The rolling-horizon solver reads on both and passes, at |ρ|=0.93 over 606 s. What the three share is the narrow band they occupy: they span 0.14 HV, and the full-information oracle buys only 0.11 over the ten-parameter rule. Front quality in this environment saturates against physical structure that ten parameters already capture, so it has little room left in which to separate methods, and the separation falls to the other two quantities.

The closest published method reads on the same corpus as the PPO-corpus row. PEDA [4] ported as published—a fixed-dimension input by zero padding, association decoded from the pooled token alone, its own conditioning kept— scores 82.87 against PrefDT’s 90.28 on the same hundred seeds $( p { = } 3 . 2 { \times } 1 0 ^ { - 3 0 } )$ , at $| \rho | { = } 0 . 7 6$ over 398 s, failing both clauses. Adding the pooled encoder without the per-user path recovers 1.31; the remaining 6.10 is the per-user bypass, which Table VI isolates. What separates PrefDT from the method it descends from is the interface to the physical network, not the sequence model.

TABLE III  
PREFDT AGAINST THE TEACHER’S POPULATION SCAN. HV $( \times 1 0 ^ { 6 } )$ IS READ OVER n=100 PAIRED SEEDS AGAINST A SHARED REFERENCE; EVERY NSGA-II ROW IS A SEARCH OF ABOUT 2000 POLICY EVALUATIONS.
<table><tr><td>Method</td><td>HV</td><td>Corner (s) Settable</td><td></td><td>Densify</td></tr><tr><td>NSGA-II pop 101 (best at fixed budget)</td><td>100.096</td><td></td><td>no</td><td>re-search</td></tr><tr><td>NSGA-II pop  $1 0 1 ^ { \ddagger }$  (2× budget)</td><td>100.120</td><td></td><td>no</td><td>re-search</td></tr><tr><td>NSGA-II pop 21</td><td>99.887</td><td>102.8</td><td>no</td><td>re-search</td></tr><tr><td>NSGA-II pop 50 (corpus source)</td><td>99.786</td><td>102.0</td><td>no</td><td>re-search</td></tr><tr><td>PrefDT (shipped), K=101</td><td>99.824§</td><td>103.6</td><td>yes</td><td>rollout</td></tr><tr><td>PrefDT (shipped), K=21</td><td>99.466</td><td>104.3</td><td>yes</td><td>rollout</td></tr><tr><td>PrefDT (archive corpus, no corner/physics),  $K { = } 2 1$ </td><td>99.229</td><td>105.0</td><td>yes</td><td>rollout</td></tr></table>

Corner is the deepest delay a fixed setting or archive member attains, the depth a request can reach; Settable is whether a different point on the front can be asked for at run time; Densify is what adding points to the fron costs.  
‡ Given twice the search budget. The corpus-source row is the teacher row of Table II, re-evaluated here at n=100.  
The headline deficit is measured against the strongest configuration, pop 101: −0.27 HV $( p { = } 1 . 8 { \times } 1 0 ^ { - 6 } ) ;$ against pop 21 the difference is −0.06 (p=0.12).

PrefDT retains 99.7% of its teacher’s front quality, and the teacher that figure is measured against is not the one it learned from but the strongest we could find. The teacher’s population is also the size of the front it returns, and re-run at the same ∼2000-evaluation budget the largest population is the strongest: 100.10 at 101, against 99.79 for the pop-50 archive the student was actually distilled from (Table III). Against that stronger teacher, PrefDT at 101 settings reaches 99.82, while the teacher has no fidelity at all, its front size is its population, and a different point on it requires searching again. The 0.27 HV between them is not spread over the front: at matched cardinality the 0–120 s band alone closes 127% of the gap, more than all of it, because above 380 s PrefDT’s points are the better ones. Appendix D decomposes that corner into what the corpus withheld, what deterministic decoding costs, and a residual limit of imitation, and what survives the first two is about a second of corner depth on a 900-s axis. Distillation therefore delivers a settable, densifiable front at 99.7% of a teacher stronger than its own corpus source, and what separates them reduces to one second at one end.

Against the solvers the quantity cannot be read at the sample the rest of this paper uses. Their solve cost fixes it: they exist only on the first ten seeds, since an $n { = } 1 0 0$ sweep is 150–200 hours of single-core compute per solver (Sec. V-C), and on those ten seeds the comparison must be paired— PrefDT scores 99.27 there, not its hundred-seed 99.47. Seed for seed SCA-MPC leads by 0.67 and the SCA oracle by 0.64, on 7 of 10 seeds, p=0.16, 95% CI [−0.07, +1.36]. That is an underpowered comparison, not a measured tie and not a measured loss, and we report it as such.

Read on front quality alone, the ranking selects against the property the conditioning was added for. The highest front quality in the whole ablation family (92.44) belongs to a variant with the return token removed and the shortest context window, at fidelity 0.643: it covers the front without following the settings. ω-conditioned ${ \mathrm { I Q L } } ,$ trained on the same corpus as PrefDT, holds that clause and fails the other— $- | \rho | { = } 0 . 9 4$ over 413 s, 67% of the teacher’s span against the required 493, and no advantage-weighting temperature closes it—swept over two and a half orders of magnitude, the reach deficit stays at least 97 s at every setting. PGMORL fails both, offering six policies that collapse to about three distinct behaviors. The sharpest case is PrefDT’s own: removing the preference token raises front quality above every arm of the family while fidelity and reach fail together (Sec. V-D). Selected by front quality alone, three of these could have been the final configuration; under the criterion, none of them is. The attribution of the teacher’s own quality is Appendix D.

## B. Operability: setting fidelity, budget tracking, and midepisode revision

Operability was defined in Sec. I as accepting three requests at run time: a preference over delay and energy, a total energy budget, and a mid-flight revision of that budget. The first is read against the criterion of Sec. IV-C. PrefDT posts $| \rho | { = } 0 . 9 3$ over a reach of 601 s, clearing both clauses, and it is the only learned method in the study that clears them on the corpus it ships with (Sec. V-A). The second and third are read against the energy coordinate of the return token, which by (21) always equals the energy still allowed to be spent.

Under unchanged physics the budget is followed and never overrun. Realized consumption is monotone in the request across the whole range, with a gain dE/dB between 0.75 and 0.99 on all four trained variants of the pipeline (the archive and corner-emphasis corpora, each with and without the physics features), and a violation rate of 0.00 at a budget tight enough to constrain the flight. Below the floor no policy in this study can fly under, the request saturates rather than being pursued. Under a propulsion cost raised by half in mid-flight, PrefDT finishes +0.6% over its contract with no re-issue and no outer loop, where the fixed rule of matching appetite finishes +17% and violates on every episode, and a supervisory controller built to repair exactly that blindness recovers less than one point of the overshoot (Fig. 2). A revision is exact where it should be exact and monotone where it should be monotone: re-issuing the same total is exact to floating point—at most $4 \times 1 0 ^ { - 4 } \mathrm { J }$ against the 34–41 kJ consumed—tightening tightens and relaxing relaxes, and a revision below the physical floor is ignored rather than pursued. The one other method in this study that holds a budget is a Lagrangian CMDP retrained for each budget. Read at the three budgets it was trained for, it tracks more accurately than PrefDT—0.107 against 0.178— and it exists nowhere else, each of the three bought with 2M environment steps.

The three readings establish that the contract is enforced against consumption rather than against a plan. The token is the remaining allowance itself, so a propulsion cost raised by half enters the quantity the policy reads one slot after the rise. The vector structure is what makes this possible: collapsed to a scalar, the same requests reverse, and asking for a larger budget yields less consumption (Sec. V-D). The violation rate under the degraded contract is 0.70 rather than zero, and the rate and the magnitude are consistent: the token requests a target, not a ceiling, so a policy that tracks one tightly finishes on both sides of it. A one-sided guarantee is obtained by requesting a margin instead: at 0.9B the same frozen model runs at violation rate 0.238, from 0.305, at relative error 0.187 from 0.172.

![](images/4833445c3847a468427c97787815d133aae58f13104866177ee1273bdfc68df0.jpg)  
Fig. 2. Capabilities (B) and (C) under a mid-episode change of physics. Each trace is cumulative mission energy against the slot index, with the band one standard deviation across seeds; the horizontal lines are the requested budget and the revised total, the straight dashed line is the pace that would land exactly on the budget, and the vertical line marks the slot at which propulsion cost rises. The archive member is a fixed rule that reads no consumption state; PrefDT receives the budget as the energy coordinate of its return token and no other instruction; the third trace rewrites that coordinate mid-episode to a total below what the changed physics allows.

TABLE IV  
WHAT ONE NEW SETTING COSTS, BY ROUTE, TIMED UNDER THESINGLE-CORE PROTOCOL OF SEC. IV-D.
<table><tr><td>Route to one new setting</td><td>Interaction</td><td>Core-s</td><td>Wall-clock (s)</td><td>n</td></tr><tr><td>PrefDT, one rollout</td><td>100 slots</td><td>0.41</td><td>0.41±0.02</td><td>20</td></tr><tr><td>SCA oracle, one convex re-solve</td><td>100 slots</td><td>251.7</td><td>252 (155–378)</td><td>3</td></tr><tr><td>SCA-MPC, one convex re-solve</td><td>100 slots</td><td>340.0</td><td>341±7</td><td>3</td></tr><tr><td>Per-preference PPO, one training run</td><td>2M steps</td><td>986.7</td><td>988±3</td><td>3</td></tr><tr><td>Teacher, one re-search*</td><td>606,000 slots</td><td>133.8</td><td>135</td><td>1</td></tr></table>

Interaction is the simulated slots or environment steps the route consumes for one setting, and does not depend on the machine; core-s is CPU time including child processes; wall-clock is elapsed time on an otherwise idle machine; n is the repetitions behind the row. The SCA oracle’s spread is across environment seeds at one preference, so its range is printed rather than a standard deviation.  
The teacher’s search is paid once and returns its archive rather than one setting; the row times the corpus-source configuration.  
Wall-clock is machine-dependent and not uniformly so—this machine runs the teacher’s search faster than the evaluation server and the solvers’ convex programs slower—while the interaction column does not move.

## C. What a new trade-off costs: one rollout against a re-solve, a re-search, and a retraining

A setting supplied at run time is worth what a new setting costs. Four routes to a new point on the trade-off curve are priced here against each other: one rollout of the frozen model, one convex re-solve, one fresh search of the teacher’s rule, and one training run of a scalarized learner. All four are timed under the single-core protocol of Sec. IV-D.

TABLE V  
THE SCHEDULING NODE’S PER-SLOT COST, ON THE SHIPPED FROZEN CHECKPOINT AND WITH NO RETRAINING.
<table><tr><td>Quantity</td><td>Shipped checkpoint</td></tr><tr><td>Parameters</td><td>3.23 M</td></tr><tr><td>Size (half precision)</td><td>6.5MB</td></tr><tr><td>Decision latency, p95</td><td>3.7 ms</td></tr><tr><td>as a share of a 1 s slot</td><td>0.37%</td></tr><tr><td>Hardware</td><td>one Apple M1 CPU core, batch 1  $1 . 2 \times 1 0 ^ { - 3 }$ </td></tr></table>

The control-channel row compares the state the node receives and the actions it returns against the offloaded traffic the fleet already carries; the closed-form features of Sec. III-A add nothing to it, being functions of raw state the node computes for itself.

Two of the four routes pay in environment interaction and two pay in computation per slot (Table IV). A rollout of the frozen model is one episode of 100 slots and 0.41 s. The solvers consume the same 100 slots and 252 and 341 s, because every slot of that episode is a fresh convex program. The scalarized learner consumes 2M environment steps and 988 s before it can serve its first setting, and the second setting costs the same again, for a front 8.8 HV below the swept one. The teacher’s search consumes 606,000 slots once and returns its archive—60× the interaction a 101-point sweep of the frozen model costs, a ratio between interaction counts and not between elapsed times. Per setting, on one core, the retraining route costs 2,400× the rollout and the solvers 620– 830×; a 21-point front follows by multiplication, at 8.6 s of rollouts against 5.8 hours of retraining and two hours of resolving for each evaluation seed. Densifying is asymmetric in the same way: 21 → 101 settings buys +0.36 HV for 80 further rollouts, while the teacher’s front size is its population and changes only by searching again (Table III).

The asymmetry does not depend on which route reaches the better front. It rests on where the search sits. The teacher must interact with the environment for all 606,000 slots and the scalarized learner for 2M steps per setting, while PrefDT trains from stored trajectories—zero environment steps, 40,000 gradient steps—and pays the search cost once, before any setting is requested (the axes this does not cover—the teacher’s own quality, and the designer who knows which ten parameters to write—are recorded in Sec. VI). What remains at run time is one forward pass per slot, priced in Table V: nothing is solved, searched, or explored in the field. What offline training removes is the search at run time; what it cannot remove is the need for competent behavior to exist somewhere in the corpus.

D. Attribution by ablation: corpus, encoder, return channel, pipeline

Sec. III argued for each component of PrefDT before any experiment was run. Eleven of them are removed here, one at a time, with architecture, training schedule and seeds otherwise fixed, so that what leaves with each component can be attributed to it rather than to model capacity, to data volume, or to some other component. Table VI is the result, module by module, with the configuration each removal was run under.

Three readings run across the table. The per-user bypass is not a choice between alternatives: the loss its removal produces is the one (15) predicts—front quality, fidelity and reach fall together—so what the measurement establishes is that a pooled encoder cannot supervise per-entity decisions without a per-entity path. The pipeline that follows the corpus is a short ladder: the archive-corpus model reads 98.65, the quadratic conditioner carries it to 99.04, corner emphasis buys the front’s corner at a fidelity of 0.895, and the physics features restore fidelity and land the shipped 99.47. Among the choices that were genuinely open, the corpus moves front quality further than any architectural one—8.42 HV—because the teacher’s search settled at the propulsion power minimum and imitation preserves that operating point where policy improvement disturbs it (Fig. 3). And the removals fall into two kinds: those that cost front quality, and those that cost the settable front while leaving front quality alone or raising it—removing the preference token raises hypervolume above the un-ablated configuration with both clauses of (26) failing.

TABLE VI  
EVERY COMPONENT OF SEC. III, THE ABLATION THAT REMOVES IT, WHAT THE ABLATION RETURNS, AND WHAT THAT ESTABLISHES. ARMS, CORPORA AND SEED COUNTS ARE IN APPENDIX C OF THE SUPPLEMENTARY MATERIAL.
<table><tr><td>Module</td><td>How it is removed</td><td>Result</td><td>What it establishes</td></tr><tr><td>Attention pooling</td><td>zero padding of a fixed user count</td><td> $- 2 . 9 0 \ \mathrm { H V } \ ( p { = } 2 . 0 { \times } 1 0 ^ { - 7 } )$ </td><td>the invariant summary is the better repre- sentation, not merely the convenient one</td></tr><tr><td>Per-user bypass (16)</td><td>decision head reads the pooled summary only</td><td>-6.10 HV  $( p { = } 3 . 0 { \times } 1 0 ^ { - 2 9 } ) ;$  |ρ|  $0 . 9 1 $  0.68 and reach  $5 4 0  4 9 1 \mathrm { s } ,$  (26) failing</td><td>the collapse (15) predicts, produced on de- both clauses of mand: a pooled encoder cannot supervise per-entity decisions without a per-entity</td></tr><tr><td>Injection site (19)</td><td>ω into the state token only, or into the return token only</td><td> $9 2 . 1 7 / 0 . 9 2 / 4 6 4 \qquad \mathrm { a n d } \qquad 9 0 . 5 6 / 0 . 8 0 / 5 5 2$  against the two-site model&#x27;s 89.91/0.90/695 quality (HV / |ρ| / reach)</td><td>path two injection points buy reach, not front</td></tr><tr><td></td><td>Vector return-to-go (19) scalar return, or no return at all</td><td>scalar +0.08% HV  $\scriptstyle ( p = 0 . 9 5 5 )$  at  $| \rho | = 0 . 6 9 ,$  vector  $+ 1 . 1 0 \% ~ ( p { = } 1 . 1 { \times } 1 0 ^ { - 4 } )$  at 0.91, both against no return at all; under the scalar, real- ized consumption reverses as the request rises,  $5 4 . 1  5 1 . { \overset { . } { 4 } } $  45.2 kJ at requests of 70, 80</td><td>a scalar carries the setting and cannot be trained from</td></tr><tr><td>Preference token</td><td>removed, with the conditioner&#x27;s targets supplied directly</td><td>and 90 kJ HV 92.02 against the un-ablated 90.28; the token buys the settable front, not front |ρ|=0.805 and reach 274 s, both below criterion quality</td><td></td></tr><tr><td>Context window</td><td> $K _ { \mathrm { c t x } } ~ \in ~ \{ 1 , 1 0 , 5 0 \}$  crossed with the return token</td><td>+3.64 HV with the token and —3.44 without it; interaction +7.08 (p=1.9× 10−24)</td><td>context tracks the decrementing target (20); it is not a model of the world</td></tr><tr><td>Teacher and corpus</td><td>distil from per-preference PPO experts in- stead of the teacher&#x27;s archive</td><td>—8.42 HV; the field reorders, with a corpus- learner interaction of +4.87 HV</td><td>imitation preserves the teacher&#x27;s operating point where policy improvement leaves it</td></tr><tr><td>Conditioner (24)</td><td>linear preference-to-return fit; ideal-point targets</td><td>the quadratic fit moves |ρ| from 0.809 to 0.932; A4 posts |ρ|=0.98 over a reach of 170 s</td><td>the conditioner decides whether the target a setting names is one the model can reach</td></tr><tr><td>Corner emphasis</td><td>uniform preference sampling</td><td> $\scriptstyle | \rho | = 0 . 8 9 5 ,$  under the criterion&#x27;s 0.90</td><td>reaching the front&#x27;s low-delay corner costs obedience until it is repaired</td></tr><tr><td>Physics features (12)</td><td>raw tokenizer features only</td><td>|ρ| restored,  $0 . 8 9 5 \ :  \ : 0 . 9 3 1 ;$  emphasis the same features buy corner depth corpus, and the pair carries the shipped</td><td>without corner what the features buy depends on the</td></tr><tr><td>Setting-axis recalibration</td><td>uniform-ω sweep</td><td>instead against 12 of 21 spoiled;  $| \rho | { = } 1 . 0 0 0$  0.931; front quality unmoved</td><td>configuration back over the criterion 21 of 21 settings distinct and non-dominated recalibration changes what a setting against names, not what the model can do</td></tr></table>

![](images/0fc3587da6cc903a29914a7d619e48d525a0980604cd5318f53dba2659233476.jpg)

![](images/d742a0288f0a9649f25ad9156a413e1c63033a99636a4dac6d42ae082ddd3dab.jpg)  
Fig. 3. (a) Propulsion power against flight speed, with hover and the interior minimum $v ^ { * }$ marked. (b) Realized mean flight speed against the requested setting, for two learners trained on the same archive corpus: PrefDT with the quadratic conditioner, and ω-conditioned IQL. The dashed line is v<sup>∗</sup> carried over from (a).

Selecting by front quality alone, in this design space, selects a model that does not take settings.

## E. Deployment: what the link need not guarantee, and what scale does not widen

Execution is centralized: one node—a ground station, or a designated leader aircraft—receives fleet state once per slot and returns the joint action. That link loses reports, and it takes time. This section measures what the scheduler needs it to guarantee, and reports three requirements it does not impose; what it does impose is Sec. V-F.

The first requirement is delivery. We drop a fraction of the user reports before they reach the policy—5, 10, 20 and 40%— and read what fraction of the front survives against the same model’s own nominal condition; the registered predictions this study made, and the one that failed, are in Appendix C. At one report in twenty, 96.7% of the front survives; at two in five, with no retraining and no special handling, 71.1% does, where the scripted per-user rule run on identical loss masks keeps 60.8%. The scheduler therefore keeps deciding on a link that drops reports, and the reason is the pooled summary of Sec. III-A: it is defined on any subset of the active set, so a lost report delivers a smaller set rather than an incomplete input, and no substitute value is invented for the members that are missing. Nothing in training saw a dropped report.

The second requirement is recency of the individual report. Substituting the last report received for a missing one lifts retention at every loss rate—by 1.0, 2.1, 4.3 and 8.9 points at 5, 10, 20 and 40%, so that at two fifths of all reports lost

80.0% of the front still survives—while the same substitution applied to the per-user rule costs it 1.3, 2.3, 3.7 and 3.3 points at the same four rates, each direction significant on the same 100 paired seeds. A stale value entering the pooled summary is one member among those that did arrive and is diluted accordingly, while a rule that compares this user’s own numbers against a threshold acts on the stale value directly. The obvious mitigation is therefore available to this scheduler and unavailable to the alternative, and one buffered report per user recovers most of what heavy loss costs.

The third requirement is speed. We displace the snapshot the policy acts on by k slots and read retention in both directions, so that staleness is separated from information: a snapshot from one slot in the future damages exactly as much as one from a slot in the past (paired difference 0.007 percentage points, p=0.85), and retention tracks the overlap between the snapshot’s activity set and the current one, about 0.43 at every displacement in both directions. What the scheduler requires is therefore that the snapshot be the current slot’s, not that it arrive quickly: within a slot the active set cannot change, by the discrete-slot formulation of Sec. II, so any transport completing inside the slot costs nothing and no latency guarantee is needed. What it costs when the snapshot is not the current slot’s is Sec. V-F.

A fourth condition is the size of the fleet. The encoder’s parameter shapes bind M, so a different fleet is served by retraining at that size rather than by one model serving every size, and what that retraining preserves is the ratio to the teacher: 78–80% at $\scriptstyle { M = 2 / 4 / 8 }$ . The distillation gap therefore does not widen with scale, and the per-slot decision stays under 2% of the slot at every size.

F. The boundaries: state synchronization, and the trained operating point

Sec. V-E established that what the link must deliver is a snapshot of the current slot, not a fast one. This experiment measures what happens when it does not: the policy decides slot t from the state of slot t−k, for k up to five. One slot of displacement costs about half the front, retention 49.5%, and further displacement costs no more—between 49.0 and 50.0% out to five slots. The loss is a cliff at the first slot rather than a slope, so there is no operating point at which a late snapshot is partially acceptable. State synchronization is a requirement of this scheduler, and it is the first boundary of the envelope.

The second boundary is the operating point itself. Every result above is measured at the configuration the model was trained on, while a deployed fleet meets channel constants, task loads and propulsion parameters that differ from it, so what this experiment asks is how far those results carry. Reevaluated frozen at fifteen operating points away from the training configuration, PrefDT’s worst-case front quality drops 4.3% on average, against 2.6% for the searched rule its corpus came from (Appendix D): imitation inherits the rule’s behavior at the trained point and not its tolerance away from it. Zeroshot transfer to shifted physics is therefore outside what this paper claims. An in-episode switch of the arrival process, by contrast, degrades neither at any context length, which locates the budget contract of Sec. V-B precisely: what it survives is a disturbance to the integral of consumption, the quantity its token carries, not a change in the environment’s statistics.

These two boundaries bound the envelope. Inside it—at the trained operating point, with the snapshot of the current slot— reports may fail and transport may take milliseconds, and every result of Sec. V-B applies as stated.

## VI. CONCLUSION

Unlike existing UAV-MEC schedulers, which compile the energy–delay trade-off into the objective before the fleet flies, PrefDT takes it as an input the scheduler reads at inference. One frozen model then serves any point on the curve at one rollout, 0.41 s per setting, where the alternatives pay a fresh solve, a fresh search, or a fresh training run.

The same channel carries a physical contract. Its energy entry is the joules still allowed, so a total budget is honored against what the fleet actually spends, is revisable in flight by a single write, and saturates safely when more is asked for than can be delivered. A denser front costs further rollouts and no further search. Three components make this work in a physical network: a pooled-token encoder with a per-user decision bypass, a vector-valued conditioning channel carried by a rollout decrement, and a seven-step distillation pipeline that generates the preference-labeled corpus that scheduling does not supply.

Across 26 method variants under one protocol, PrefDT produces the best trade-off curve of any learned method and, on the corpus it ships with, is the only one that clears a criterion on setting–outcome order and span fixed before the decisive runs. One model, unchanged, also holds an energy budget to 0.6% when the physics change in mid-flight, where the open-loop rule it learned from overruns on every episode; and its front reaches 99.7% of a teacher stronger than the one it was distilled from. Front quality does not say whether a setting was followed. We report two readings that do—the order in which the outcomes arrive, and the span over which that order holds—and the configuration this paper ships was selected on them, not on front quality.

Four qualifications bound what this paper establishes. Every result is measured in one physics-based simulator family, whose i.i.d. arrival process the training corpora share. Uplink bandwidth is uncontended by construction (Sec. II), so one source of pressure on the association decision is absent. The solver latency of Sec. V-C times a per-slot re-compilation that parameterized caching would largely amortize—only 79 ms per slot is irreducible—so the schedulability comparison is a statement about our implementation, not about the methods; what no caching removes is the structure the cost table prices, a fresh convex program every slot and a fresh solve per preference. Finally, the cost accounting does not cover design knowledge at all: the teacher’s quality is ours to have composed, and a designer who already knows which ten parameters to write has paid a cost this paper does not price, incurred once when the system is built and never again at a setting.

[1] Y. Zeng, Q. Wu, and R. Zhang, “Accessing from the sky: A tutorial on UAV communications for 5G and beyond,” Proc. IEEE, vol. 107, no. 12, pp. 2327–2375, 2019.

[2] I. Das and J. E. Dennis, “A closer look at drawbacks of minimizing weighted sums of objectives for Pareto set generation in multicriteria optimization problems,” Structural Optimization, vol. 14, no. 1, pp. 63– 69, 1997.

[3] L. Chen et al., “Decision Transformer: Reinforcement learning via sequence modeling,” in Proc. NeurIPS, 2021.

[4] B. Zhu, M. Dang, and A. Grover, “Scaling Pareto-efficient decision making via offline multi-objective RL,” in Proc. ICLR, 2023.

[5] Q. Wu, Y. Zeng, and R. Zhang, “Joint trajectory and communication design for multi-UAV enabled wireless networks,” IEEE Trans. Wireless Commun., vol. 17, no. 3, pp. 2109–2121, 2018.

[6] G. Sun, Y. Wang, Z. Sun, Q. Wu, J. Kang, D. Niyato, and V. C. M. Leung, “Multi-objective optimization for multi-UAV-assisted mobile edge computing,” IEEE Trans. Mobile Comput., vol. 23, no. 12, pp. 14803–14820, 2024.

[7] L. Huang, S. Bi, and Y.-J. A. Zhang, “Deep reinforcement learning for online computation offloading in wireless powered mobile-edge computing networks,” IEEE Trans. Mobile Comput., vol. 19, no. 11, pp. 2581–2593, 2020.

[8] J. Xu et al., “Prediction-guided multi-objective reinforcement learning for continuous robot control,” in Proc. ICML, 2020.

[9] C. Tessler, D. J. Mankowitz, and S. Mannor, “Reward constrained policy optimization,” in Proc. ICLR, 2019.

[10] D. M. Roijers, P. Vamplew, S. Whiteson, and R. Dazeley, “A survey of multi-objective sequential decision-making,” J. Artif. Intell. Res., vol. 48, pp. 67–113, 2013.

[11] R. Yang, X. Sun, and K. Narasimhan, “A generalized algorithm for multi-objective reinforcement learning and policy adaptation,” in Proc. NeurIPS, 2019.

[12] M. Janner, Q. Li, and S. Levine, “Offline reinforcement learning as one big sequence modeling problem,” in Proc. NeurIPS, 2021.

[13] Q. Zheng, A. Zhang, and A. Grover, “Online Decision Transformer,” in Proc. ICML, 2022.

[14] M. Xu et al., “Prompting Decision Transformer for few-shot policy generalization,” in Proc. ICML, 2022.

[15] L. Meng et al., “Offline pre-trained multi-agent decision transformer,” Mach. Intell. Res., vol. 20, no. 2, pp. 233–248, 2023.

[16] C. Lu et al., “Attention-enhanced prompt decision transformers for AAVassisted communications with AoI,” IEEE Wireless Commun. Lett., 2025.

[17] A. Al-Hourani, S. Kandeepan, and S. Lardner, “Optimal LAP altitude for maximum coverage,” IEEE Wireless Commun. Lett., vol. 3, no. 6, pp. 569–572, 2014.

[18] J. Schulman et al., “Proximal policy optimization algorithms,” arXiv:1707.06347, 2017.

![](images/a93bd5762e4daea0f1b985f8e8099dce12acc5c0031124f02c7e507d3ca9695e.jpg)

[19] E. Zitzler and L. Thiele, “Multiobjective evolutionary algorithms: A comparative case study and the strength Pareto approach,” IEEE Trans. Evol. Comput., vol. 3, no. 4, pp. 257–271, 1999.

[20] T. Basaklar, S. Gumussoy, and U. Y. Ogras, “PD-MORL: Preferencedriven multi-objective reinforcement learning algorithm,” in Proc. ICLR, 2023.

[21] A. Ghanem et al., “Multi-objective decision transformers for offline reinforcement learning,” arXiv preprint, 2023.

Qiao Liao is currently pursuing a Ph.D. degree in Computer Science and Technology with the College of Intelligence and Computing, Tianjin University. She is working on mobile edge computing. Her main research interests are reinforcement learning and service computing.

[22] L. Feng et al., “Graph-attention-based reinforcement learning for trajectory design and resource assignment in multi-UAV-assisted communication,” IEEE Internet Things J., 2024.

[23] K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput., vol. 6, no. 2, pp. 182–197, 2002.

[24] I. Kostrikov, A. Nair, and S. Levine, “Offline reinforcement learning with implicit Q-learning,” in Proc. ICLR, 2022.

Zhiyong Feng (Member, IEEE) was born in 1965. He received the Ph.D. degree from Tianjin University, Tianjin, China, in 1996.

![](images/e717e2b46be000f0033d2c2f09d1dc59e7ff6e6f76457af262a228477d87d8d8.jpg)

He is currently a Professor with the College of Intelligence and Computing, Tianjin University, Tianjin, China. He has authored more than 200 articles, one book and 39 patents. His research interests include service computing, knowledge engineering, and software engineering.

Dr. Feng is a distinguished member of China Computer Federation (CCF), a member of the Asso-

ciation for Computing Machinery (ACM), and the Chairman of ACM China Tianjin Branch.

![](images/3184e2bd2c40d334aff61e680556c68575b4e4928182e4c103ee59c60c135578.jpg)

Bin Wu received the Ph.D. degree in Electrical and Electronic Engineering from the University of Hong Kong, Hong Kong, in 2007. From 2007 to 2012, he was a Postdoctoral Research Fellow with the Department of Electrical and Computer Engineering, University of Waterloo, Waterloo, ON, Canada. He is currently a Professor with the College of Intelligence and Computing, Tianjin University, Tianjin, China. His research interests include computer systems and networking, and communication system design.

![](images/f78afa6ea0e5c11ec27b54b3e355dc109eec1299c301d35e201799e8b0cf460c.jpg)

Guodong Fan received the Ph.D. degree in Computer Science and Technology with the College of Intelligence and Computing, Tianjin University, Tianjin, China. He is working on cognitive services. His main research interests include representation learning, service computing, and software repository mining.