# CROSS-ROLLOUT BELLMAN CLOSURE FOR LONG-HORIZON AGENTIC REINFORCEMENT LEARNING

Yangyang Ren<sup>1,2∗</sup> Haodong Zhu<sup>1,2∗</sup> Linlin Yang<sup>3†</sup> Sheng Xu<sup>3</sup> Peichao Lai<sup>4</sup> Baochang Zhang<sup>1,5</sup>

<sup>1</sup>Beihang University <sup>2</sup>Zhongguancun Academy <sup>3</sup>Communication University of China <sup>4</sup>Peking University <sup>5</sup>Hangzhou Innovation Institute of Beihang University

## ABSTRACT

Group-based reinforcement learning such as GRPO has become a standard recipe for post-training LLM agents, replacing a learned critic with relative comparison among rollouts sampled for each task. In long-horizon settings, these rollouts revisit shared anchor states, offering cross-rollout evidence for step-level credit. Ideally, step-level credit should incorporate evidence beyond the realized suffixes observed at an anchor while aggregating alternative continuations according to their empirical frequencies. Existing estimators capture only one of these properties: visit-local averaging pools realized suffix returns at shared anchors and respects observed frequencies, but does not recursively propagate evidence across rollouts; whereas shortest-path estimators have global reach but allow a rarely observed route to dominate an anchor’s value. Instead, we introduce a Cross-Rollout Bellman Closure (CRBC) method, which merges each rollout group into a finite empirical process with absorbing success and failure boundaries, and evaluates its behaviour-policy Bellman fixed point with one linear solve. This fixed point uses the same empirical action and transition frequencies to propagate evidence through shared anchors and aggregate alternative continuations. Backing up the resulting state values through observed transitions yields action values, whose gain over the corresponding state value provides step-level credit. A corresponding finite-depth family recovers visit-local return averaging at zero depth and converges to the exact closure as depth increases. The normalized closure credit is combined with the trajectory-level group advantage for policy optimization, without additional environment rollouts. Across ALFWorld, WebShop, and Sokoban benchmarks with multiple model scales, CRBC consistently improves final performance and learning efficiency. For example, CRBC outperforms the strongest baseline by 5.59 percentage points on ALFWorld with Qwen2.5-1.5B-Instruct, achieving a new state-of-the-art performance.

## 1 INTRODUCTION

Large Language Models (LLMs) (OpenAI, 2023; Yang et al., 2024; DeepSeek-AI, 2025) increasingly serve as interactive agents in embodied worlds, web interfaces, and tool-augmented workflows, where task completion requires planning across multiple interdependent decisions (Shridhar et al., 2021; Yao et al., 2022; Furuta et al., 2024; Schick et al., 2023). Reinforcement learning (RL) provides a post-training approach to improving these capabilities (Ziegler et al., 2020; Ouyang et al., 2022; Team, 2025). Group-based methods such as RLOO and GRPO replace PPO’s learned critic (Schulman et al., 2017) with relative comparisons among same-task rollouts, avoiding the cost of training a separate value network (Ahmadian et al., 2024; Shao et al., 2024; Yu et al., 2025).

A single trajectory-level advantage, when broadcast to every step, cannot distinguish the contributions of individual actions (Wang et al., 2025; Jin et al., 2025; Chen et al., 2025). Yet rollouts sampled from the same task and initial condition often visit shared intermediate situations, or anchors, providing transition evidence beyond terminal outcomes. Step-level methods exploit this structure by comparing visits at shared anchors, with some further refining comparison groups by the consistency of preceding histories (Feng et al., 2025; Wang et al., 2026c; He et al., 2026). A further line pools transitions across rollouts into a merged graph and scores actions on it (Cheng et al., 2026; Wang et al., 2026d). Together, these approaches expose a reusable cross-rollout structure within each task group (Figure 1(a)).

![](images/9b756d4b3d93d38a64c7810bc3a7b83f5fdcc8ff5fb2b4db303065949d7e412e.jpg)  
Figure 1: Reach and aggregation for step-level credit. For the rollout trajectory τ, node s denotes an anchor: an environment-relevant state representation that may be shared by multiple visits, and the edge represents an observed transition after executing semantic action a. (a) Shared anchors merge rollouts into a finite empirical process. (b) Shortest-path estimation aggregates by an extremum, ignoring transition frequencies; a rarely executed continuation can dominate anchor value. (c) The finite-depth family on this process, illustrated at $s _ { 2 } ,$ uses behaviour-consistent expectations: $K = 0$ pools visit-local suffix returns; $K = 1$ adds one cross-rollout backup; and $K  \infty$ evaluates all supported continuations at the behaviour-policy Bellman fixed point (CRBC), giving global reach.

Yet exposing this structure determines which observed transitions may be recombined, not how they should be evaluated. Two choices remain open: how far evidence may travel (reach), and how the evidence that arrives is combined (aggregation). Visit-local averaging (Feng et al., 2025) aggregates by an expectation over the observed behaviour, averaging realized suffix returns across visits to an anchor, but its reach is local: each occurrence is scored by its own continuation, so evidence pooled at shared successors is not recursively propagated back to the current anchor (Figure 1(c)). A shortest-path estimator (Cheng et al., 2026) has global reach over the merged graph, but aggregates by an extremum (Figure 1(b)): it values the shortest route to success without accounting for how frequently its transitions were observed, so a rarely observed continuation can dominate the estimate. Intuitively, an action’s credit should account for both successful and failed continuations revealed by other rollouts at a shared successor. These limitations motivate combining global reach with behaviour-consistent aggregation on the group’s empirical process, leading to our central question: once a rollout group has been merged into an empirical process, how should step credit propagate through and aggregate its supported continuations?

To address this issue, we introduce Cross-Rollout Bellman Closure (CRBC), which merges each rollout group into a finite empirical process over shared anchors and evaluates it at its behaviourpolicy Bellman fixed point with one linear solve (Figure 2). Specifically, the Bellman closure repeatedly propagates successor values backward through the observed transitions until each anchor value is consistent with the values of its possible continuations. This closure is also the limit of Bellman backups initialized with visit-local suffix averages, allowing anchors to incorporate evidence from rollouts that never visited them. With success and failure as absorbing boundaries, this fixed point is the state value V, the exact discounted probability of reaching success under the group’s own behaviour. Backing V up through the observed transitions gives the action value Q, and Q − V measures how much an action improves on that behaviour at the same anchor. On this empirical process, feasible policy reweighting along the closure-credit direction does not decrease discounted success value (Proposition 2). We combine the normalized closure credit with the trajectory-level group advantage for critic-free policy optimization, without additional environment rollouts. Our main contributions are summarized as follows:

![](images/f611a9f3b0e47d826a2502a07a7243d429fab856eed24be730ed3b707190dee6.jpg)  
Figure 2: Overview of CRBC. (a) Shared anchors merge rollouts into an empirical process with absorbing success and failure boundaries. (b) A Bellman fixed-point solve yields the state value V, followed by action backups to obtain the action value Q and normalized Q − V credit, which gives relative action credit. (c) Step-level credit and trajectory-level group-relative advantages are combined for policy optimization.

• We construct a finite empirical process from each rollout group’s transition counts, allowing credit at an anchor to use downstream evidence from rollouts that never visited it. This evidence is weighted by observed action and transition frequencies.

• We introduce CRBC, which computes the group’s behaviour-policy Bellman fixed point with one linear solve and scores each action by its gain over the state value, giving step credit with global reach and behaviour-consistent aggregation.

• We show that feasible policy reweighting along the closure-credit direction does not decrease discounted success value on the fixed empirical process (Proposition 2). Experiments on ALFWorld, WebShop, and Sokoban demonstrate gains over the evaluated baselines, including a 5.59-percentage-point improvement over the strongest baseline on ALF-World with Qwen2.5-1.5B-Instruct.

## 2 RELATED WORK

Reinforcement learning for LLMs and agents. LLM post-training uses human feedback (Ziegler et al., 2020; Stiennon et al., 2022; Ouyang et al., 2022), preference optimization (Rafailov et al., 2024), and verifiable rewards (DeepSeek-AI, 2025; Team, 2025). Group-based methods replace PPO’s critic (Schulman et al., 2017) with comparisons among same-prompt responses, including RLOO (Ahmadian et al., 2024), GRPO (Shao et al., 2024), Dr. GRPO, DAPO, and CPPO (Liu et al., 2025; Yu et al., 2025; Lin et al., 2025). Agent training progressed from DQN in text games (Mnih et al., 2015; Narasimhan et al., 2015) to PPO- and AWR-style updates (Peng et al., 2019) for embodied and device-control tasks (Zhai et al., 2024; Bai et al., 2024). ArCHer (Zhou et al., 2024) learns hierarchical values for tasks such as WebShop (Yao et al., 2022), while Agent Q (Putta et al., 2024) uses Monte Carlo tree search (Silver et al., 2017). LOOP (Chen et al., 2025) combines leave-one-out estimation with PPO-style updates on AppWorld (Trivedi et al., 2024), and RAGEN (Wang et al., 2025) studies multi-turn trajectory optimization. Sparse rewards in long-horizon tasks such as ALFWorld (Shridhar et al., 2021) motivate finer credit assignment.

Step-level credit assignment in agentic RL. Step-level credit can use process reward models (Lightman et al., 2024), LLM judges (Wang et al., 2026b), learned action-boundary critics (Wang et al., 2026a), or rollout-group statistics. GiGPO (Feng et al., 2025) compares repeated anchor visits,

HGPO (He et al., 2026) refines groups by history consistency, and RTMC (Wang et al., 2026c) averages returns over rollout trees. GraphGPO (Cheng et al., 2026) scores merged graphs by shortestpath distance to success, while G2PO (Wang et al., 2026d) derives edge credit from one-step TD errors. Other methods calibrate credit under limited rollout evidence (Li et al., 2026), add progress signals for all-failing groups (Yang et al., 2026), derive credit from state potentials (Fan & Liu, 2026), or extract attribution from model reasoning (Tan et al., 2026). Unlike return averaging, shortest-path scoring, or one-step TD credit, CRBC explicitly solves for the exact behavior-policy Bellman fixed point of the group’s empirical process, using observed action and transition frequencies for cross-rollout propagation and aggregation.

## 3 PRELIMINARIES

Rollout groups and anchors. We consider an LLM agent with policy $\pi _ { \theta } .$ , parameterized by $\theta ,$ solving a task g through G rollouts from the same initial condition. At step t of rollout i, it observes a language context $h _ { i , t } ,$ , generates a response $y _ { i , t } \sim \pi _ { \theta } ( \cdot \mid h _ { i , t } )$ , and executes the semantic action $a _ { i , t } = \operatorname { E x e c } ( y _ { i , t } )$ , after which the environment returns $h _ { i , t + 1 }$ . Rollout i ends after $T _ { i } \leq T _ { \operatorname* { m a x } }$ steps with a binary success indicator $R _ { i } \in \{ 0 , 1 \}$ and no intermediate reward. An environment-aware map $\phi : \mathcal { H } \to S$ from language contexts to anchors defines $s _ { i , t } = \phi ( h _ { i , t } )$ , with anchors matched only within each task group (Feng et al., 2025; Cheng et al., 2026). We represent the group by

$$
\mathcal { D } _ { g } = \{ \tau _ { i } \} _ { i = 1 } ^ { G } , \qquad \tau _ { i } = \big ( ( s _ { i , t } , a _ { i , t } , x _ { i , t } ) _ { t = 1 } ^ { T _ { i } } , R _ { i } \big ) ,\tag{1}
$$

where $x _ { i , t } = s _ { i , t + 1 }$ for $t < T _ { i }$ , and the terminal successor $x _ { i , T _ { i } }$ is the absorbing success boundary $z ^ { + } \mathrm { i f } R _ { i } = 1$ , or the absorbing failure boundary $z ^ { - }$ otherwise. We write $\scriptstyle { \mathcal { S } } _ { q }$ for the observed nonterminal anchors, $\mathcal { A } _ { g } ( s )$ for the actions observed at $s ,$ and $\mathcal { X } _ { g } = \mathcal { S } _ { g } \cup \{ z ^ { + } , z ^ { - } \}$ . Contexts and responses are retained for policy optimization. Time-limit truncations are treated as failures in this representation.

Group-based RL. Let $A _ { i , t }$ <sub>t</sub> denote the advantage assigned to environment step (i, t). The optimizer maximizes a clipped surrogate with a reference-policy penalty,

$$
\begin{array} { r } { \mathcal { I } ( \theta ) = \mathbb { E } \Big [ \operatorname* { m i n } \big ( \rho _ { i , t } ( \theta ) A _ { i , t } , ~ \mathrm { c l i p } ( \rho _ { i , t } ( \theta ) , 1 \pm \epsilon ) A _ { i , t } \big ) \Big ] - \lambda _ { \mathrm { K L } } \mathbb { D } _ { \mathrm { K L } } \big [ \pi _ { \theta } ~ \big \| ~ \pi _ { \mathrm { r e f } } \big ] . } \end{array}\tag{2}
$$

Here E averages uniformly over sampled groups and, within each group, over environment steps. ϵ sets the clipping range, and $\lambda _ { \mathrm { K L } }$ weights the KL penalty against the fixed reference policy $\pi _ { \mathrm { r e f } }$ The importance ratio is $\rho _ { i , t } ( \theta ) = \pi _ { \theta } ( y _ { i , t } \mid h _ { i , t } ) / \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { i , t } \mid h _ { i , t } )$ , where $\pi _ { \theta _ { \mathrm { o l d } } }$ is the rollout policy. Group-relative methods form the advantage by comparing outcomes within the group rather than against a learned value function; GRPO (Shao et al., 2024) uses

$$
A _ { i } ^ { \mathrm { G } } = \frac { R _ { i } - \operatorname * { m e a n } ( R _ { 1 } , . . . , R _ { G } ) } { \operatorname { s t d } ( R _ { 1 } , . . . , R _ { G } ) } ,\tag{3}
$$

and RLOO (Ahmadian et al., 2024) a leave-one-out variant. Ours is agnostic to the choice, and we denote by $A _ { i } ^ { \mathrm { G } }$ whatever the underlying optimizer outputs. Setting $\bar { A _ { i , t } } = A _ { i } ^ { \mathrm { G } }$ for every step of $\tau _ { i }$ gives one scalar per trajectory, broadcast uniformly, which cannot distinguish the actions inside it.

Step-level credit. Step-level methods exploit shared anchors to replace uniform trajectory-level credit with signals that vary within a trajectory. Let $\mathcal { T } _ { g } ( s ) : = \{ ( i , t ) : s _ { i , t } = s \}$ collect occurrences of anchor $s ,$ and let $G _ { i , t } = \beta ^ { T _ { i } - t + 1 } R _ { i }$ <sub>i</sub> denote the realized discounted return, with discount $\beta \in$ (0, 1). The step-level advantage is

$$
A _ { i , t } ^ { \mathrm s t e p } = \frac { G _ { i , t } - \operatorname* { m e a n } \{ G _ { i ^ { \prime } , t ^ { \prime } } : ( i ^ { \prime } , t ^ { \prime } ) \in \mathcal { T } _ { g } ( s _ { i , t } ) \} } { \mathrm { s t d } \{ G _ { i ^ { \prime } , t ^ { \prime } } : ( i ^ { \prime } , t ^ { \prime } ) \in \mathcal { T } _ { g } ( s _ { i , t } ) \} } ,\tag{4}
$$

combined with $A _ { i } ^ { \mathrm { G } }$ by a weighted sum (He et al., 2026; Feng et al., 2025). Here equation 4 scores each occurrence by its own realized suffix return, so evidence from other rollouts at shared successors never propagates back to s.

## 4 CROSS-ROLLOUT BELLMAN CLOSURE

To propagate and aggregate evidence from the observed continuations of rollouts that share anchors, this section constructs a finite empirical process from a rollout group (Section 4.1), computes its Bellman closure (Section 4.2), analyses credit alignment across propagation depths (Section 4.3), and integrates closure credit into policy optimization (Section 4.4), as summarized in Figure 2. Throughout the section and its proofs, $P _ { g } , \mu _ { g } ,$ , and all associated Bellman operators, values, and credits refer to the process induced by group g.

## 4.1 THE EMPIRICAL PROCESS

We first turn the rollouts in a group into a common empirical process over shared anchors. This representation preserves the observed action and transition frequencies needed to propagate and aggregate cross-rollout evidence.

Empirical model. Let $N _ { q } ( s , a , x )$ count occurrences of transition $( s , a , x )$ in $\mathcal { D } _ { g }$ , with marginals $\begin{array} { r } { N _ { g } ( \bar { s } , a ) = \sum _ { x \in \mathcal { X } _ { a } } N _ { g } ( s , \bar { a } , x ) } \end{array}$ and $\begin{array} { r } { N _ { g } ( s ) = \sum _ { a \in \mathcal { A } _ { a } ( s ) } N _ { g } ( s , a ) } \end{array}$ . The group induces an empirical transition kernel and an empirical behaviour policy over anchors,

$$
P _ { g } ( x \mid s , a ) : = \frac { N _ { g } ( s , a , x ) } { N _ { g } ( s , a ) } , \qquad \mu _ { g } ( a \mid s ) : = \frac { N _ { g } ( s , a ) } { N _ { g } ( s ) } .\tag{5}
$$

The unsmoothed frequencies in equation 5 define the group-induced process, with zero probability assigned to unobserved transitions. Our analysis concerns this empirical process rather than the unknown environment dynamics.

The visit-local estimator. Let $\mathcal { T } _ { g } ( s , a ) : = \{ ( i , t ) \in \mathbb { Z } _ { g } ( s ) : a _ { i , t } = a \}$ collect occurrences of $( s , a )$ within group $^ { g , }$ so that $| \mathcal { T } _ { g } ( s , a ) | = N _ { g } ( s , a )$ . Using the realized discounted returns $G _ { i , t }$ defined in Section 3, we define the visit-local estimates

$$
Q _ { g } ^ { ( 0 ) } ( s , a ) : = \frac { 1 } { | \mathcal { Z } _ { g } ( s , a ) | } \sum _ { ( i , t ) \in \mathcal { T } _ { g } ( s , a ) } G _ { i , t } , \qquad V _ { g } ^ { ( 0 ) } ( s ) : = \sum _ { a \in \mathcal { A } _ { g } ( s ) } \mu _ { g } ( a \mid s ) Q _ { g } ^ { ( 0 ) } ( s , a ) .\tag{6}
$$

Weighting these action values by their empirical action frequencies makes $V _ { g } ^ { ( 0 ) } ( s )$ equal to the mean return over all visits to s. Each occurrence contributes only the discounted outcome of its own complete continuation to termination.

## 4.2 THE BELLMAN CLOSURE

Given the empirical process, we now evaluate how success evidence propagates through its ob served transitions. We achieve this by defining Bellman backups and their fixed point, which yields behaviour-consistent state and action values.

Backup operators. Let $\nu _ { g }$ denote the value functions $V : \mathcal { X } _ { g }  [ 0 , 1 ]$ with $V ( z ^ { + } ) = 1$ and $V ( z ^ { - } ) = 0$ . The action backup averages discounted successor values under the empirical transition kernel, and the state backup averages these action values under the empirical behaviour policy:

$$
\begin{array} { r l } & { ( \mathcal { B } _ { g } V ) ( s , a ) : = \beta \displaystyle \sum _ { x \in \mathcal { X } _ { g } } P _ { g } ( x \mid s , a ) V ( x ) , \qquad s \in \mathcal { S } _ { g } , \ a \in \mathcal { A } _ { g } ( s ) , } \\ & { \quad ( \mathcal { T } _ { g } V ) ( s ) : = \displaystyle \sum _ { a \in \mathcal { A } _ { g } ( s ) } \mu _ { g } ( a \mid s ) ( \mathcal { B } _ { g } V ) ( s , a ) , \quad s \in \mathcal { S } _ { g } . } \end{array}\tag{7}
$$

The state backup keeps both boundary values fixed. Each application of $\mathcal { T } _ { g }$ performs one round of cross-rollout recombination: evidence at successor anchors propagates back to s across one additional observed edge, including evidence from rollouts that did not visit s. By contrast, equation 6 pools complete realized suffix returns without recombining continuations across rollouts. Repeated backups extend this propagation along longer observed paths.

The behaviour-policy fixed point. We define the closure of this propagation as a value assignment unchanged by further Bellman backups. Each anchor’s value then equals the empirical average of its discounted successor values, yielding the fixed-point condition $\begin{array} { r } { \dot { \mathcal { T } } _ { g } V _ { g } = V _ { g } . } \end{array}$ We express this condition as a linear system by collecting the transitions among non-terminal anchors and the onestep contribution from the success boundary into

$$
M _ { g } ( s , s ^ { \prime } ) : = \sum _ { a \in A _ { g } ( s ) } \mu _ { g } ( a \mid s ) P _ { g } ( s ^ { \prime } \mid s , a ) , \qquad b _ { g } ( s ) : = \beta \sum _ { a \in A _ { g } ( s ) } \mu _ { g } ( a \mid s ) P _ { g } ( z ^ { + } \mid s , a ) ,\tag{8}
$$

where $s , s ^ { \prime } \in S _ { g }$ . Restricting values to these non-terminal anchors, the state backup is affine, $\mathcal { T } _ { g } V =$ $\beta M _ { g } V + b _ { g }$ . The matrix $M _ { g }$ is substochastic: its row sums are at most one because probability mass may leave to the absorbing boundaries. The fixed-point condition therefore becomes $( I - \beta M _ { g } ) V _ { g } =$ $b _ { g }$ , so well-posedness reduces to the invertibility of $I - \beta M _ { g }$

Proposition 1 (Well-posedness). For any group and any $\beta \in ( 0 , 1 ) , \mathcal { T } _ { g }$ is a β-contraction on $\mathcal { V } _ { g }$ in $\| \cdot \| _ { \infty }$ and admits a unique fixed point $\bar { V } _ { g } \in \mathcal { V } _ { g } .$ . Bellman iteration converges to $V _ { g }$ from every initializer in $\gamma _ { g }$ , and $I - \beta M _ { g }$ is invertible.

The closure is therefore independent of the initializer and has the closed form

$$
V _ { g } = ( I - \beta M _ { g } ) ^ { - 1 } b _ { g } , \qquad Q _ { g } : = \mathcal { B } _ { g } V _ { g } .\tag{9}
$$

In practice, we obtain $V _ { g }$ by solving $( I - \beta M _ { g } ) V _ { g } = b _ { g }$ directly, without forming the inverse or iterating $\mathcal { T } _ { q } \ ( \mathrm { A }$ ppendix B.1). With no intermediate rewards and boundary values $\bar { V } ( z ^ { + } ) = 1$ and $V ( z ^ { - } ) \bar { = 0 } , \bar { V _ { g } ( s ) }$ gives the exact β-discounted probability of reaching success from s under $\mu _ { g }$ on the empirical process induced by the current group.

## 4.3 FINITE DEPTH AND THE REFERENCE DIRECTION

We next relate visit-local estimation to the full closure through a finite-depth family. We then characterize when feasible policy reweighting using these credits does not decrease discounted success value on the fixed empirical process.

The finite-depth family. We initialize Bellman iteration at the visit-local estimator in equation 6, with the same fixed boundary values. For integers $K \geq 1$

$$
V _ { g } ^ { ( K ) } : = \mathcal T _ { g } ^ { K } V _ { g } ^ { ( 0 ) } , \qquad Q _ { g } ^ { ( K ) } : = \mathcal B _ { g } V _ { g } ^ { ( K - 1 ) } , \qquad A _ { g } ^ { ( K ) } : = Q _ { g } ^ { ( K ) } - V _ { g } ^ { ( K ) } ,\tag{10}
$$

with $A _ { g } ^ { ( 0 ) } : = Q _ { g } ^ { ( 0 ) } - V _ { g } ^ { ( 0 ) } . \mathrm { A s } K  \infty ,$ , the value estimates converge to $V _ { g }$ and $Q _ { g }$ in equation 9, and the limiting credit is denoted by $A _ { g } ^ { ( \infty ) } : = Q _ { g } - V _ { g }$ . Here K counts rounds of recombination rather than steps of lookahead, since equation 6 already carries complete realized suffixes at $K = 0$ . All members remain centred under $\mu _ { g }$ at each anchor, and varying K changes reach while holding the empirical process and aggregation fixed. The $K = 0$ member uses visit-local suffix averaging (Feng et al., 2025), whereas shortest-path estimators (Cheng et al., 2026) lie outside this family because they aggregate by an extremum rather than an expectation.

On the fixed empirical process, let $V _ { a } ^ { \nu }$ denote the β-discounted success value under a policy ν on the observed action support, and let ${ \dot { M } } _ { g } ^ { \nu }$ be obtained from $M _ { g }$ in equation 8 by replacing $\mu _ { g }$ with $\nu .$ In particular, $V _ { g } ^ { \mu _ { g } } = V _ { g }$

Proposition 2 (The closure credit as a reference direction). Let A<sup>˜</sup> be centred under $\mu _ { g }$ at each anchor, let w $: S _ { g }  [ 0 , \infty )$ , and define $\xi ( s , a ) : = \mu _ { g } ( a \mid s ) w ( s ) \tilde { A } ( s , a )$ . For any $\alpha \geq 0$ such that $\nu _ { \alpha } : = \mu _ { g } + \alpha \xi$ is a valid policy,

$$
V _ { g } ^ { \nu _ { \alpha } } - V _ { g } = \alpha \big ( I - \beta M _ { g } ^ { \nu _ { \alpha } } \big ) ^ { - 1 } \Big ( w \odot \big < \tilde { A } , A _ { g } ^ { ( \infty ) } \big > _ { \mu _ { g } } \Big ) ,\tag{11}
$$

where $\begin{array} { r } { \langle f , h \rangle _ { \mu _ { g } } ( s ) : = \sum _ { a } \mu _ { g } ( a \mid s ) f ( s , a ) h ( s , a ) } \end{array}$ and ⊙ denotes componentwise multiplication. For $\tilde { A } = A _ { q } ^ { ( \infty ) }$ , the inner product is the anchor-wise variance $\begin{array} { r } { \sigma _ { g } ( s ) ^ { 2 } : = \sum _ { a } \mu _ { g } ( a \mid s ) [ A _ { g } ^ { ( \infty ) } ( s , a ) ] ^ { 2 } \ge } \end{array}$ 0, giving $V _ { g } ^ { \nu _ { \alpha } } \succeq V _ { g }$ componentwise.

Finite-depth credits inherit this guarantee when their deviation from the closure is sufficiently small.

Corollary 3 (Finite-depth alignment). For $K \geq 1$ , let $\bar { \delta } _ { g } ^ { ( 0 ) } : = \mathcal { T } _ { g } V _ { g } ^ { ( 0 ) } - V _ { g } ^ { ( 0 ) }$ be the empirical Bellman residual on non-terminal anchors. Then

$$
\big \| \boldsymbol { A } _ { g } ^ { ( \infty ) } - \boldsymbol { A } _ { g } ^ { ( K ) } \big \| _ { \infty } \leq \frac { 2 \beta ^ { K } } { 1 - \beta } \big \| \bar { \boldsymbol { \delta } } _ { g } ^ { ( 0 ) } \big \| _ { \infty } .\tag{12}
$$

Under the assumptions of Proposition 2, taking $\tilde { A } = A _ { g } ^ { ( K ) }$ gives $V _ { q } ^ { \nu _ { \alpha } } \succeq V _ { g }$ whenever the right-hand side of equation 12 does not exceed $\sigma _ { g } ( s )$ at every anchor with $\bar { w } ( s ) > 0$ and $\sigma _ { g } ( s ) > 0 ,$ ; anchors with $\sigma _ { g } ( s ) = 0$ contribute a zero inner product in equation 11.

Together, Proposition 2 and Corollary 3 establish closure credit as a reference direction on the fixed empirical process: feasible policy reweighting along this direction does not decrease discounted success value, and sufficiently accurate finite-depth credits inherit the same guarantee. Proofs are provided in Appendix B, with further analysis of Bellman residuals and finite-depth errors in $\mathsf { A p - }$ pendix B.3.

## 4.4 STEP CREDIT AND POLICY OPTIMIZATION

Finally, we convert closure values into normalized step-level credit and combine it with the trajectory-level group advantage for critic-free policy optimization.

Step-level advantage. For CRBC, we gate closure credit on anchors with at least two distinct observed actions and normalise it by the anchor-wise standard deviation (Figure 2(b)):

$$
A _ { g } ^ { \mathrm { S } } ( s , a ) : = \frac { \mathbf { 1 } \{ | A _ { g } ( s ) | \geq 2 \} } { \operatorname* { m a x } \{ \sigma _ { g } ( s ) , \sigma _ { \mathrm { m i n } } \} } \big ( Q _ { g } ( s , a ) - V _ { g } ( s ) \big ) .\tag{13}
$$

Here $\sigma _ { g } ( s )$ is defined in Proposition 2, and $\sigma _ { \operatorname* { m i n } { } } > 0$ guards nearly flat rows. Denote the nonnegative prefactor by $c _ { g } ( s ) ;$ the same proposition then applies with $w = c _ { g } ,$ preserving alignment. All occurrences of the same (anchor, semantic action) pair share this credit.

Policy optimization. CRBC retains the trajectory-level advantage $A _ { i } ^ { \mathrm { G } }$ (Figure 2(c)) so that steps without an identifiable anchor comparison remain covered. The advantage substituted into equation 2 at step (i, t) is

$$
\begin{array} { r } { A _ { i , t } ^ { \mathrm { f i n a l } } = w _ { \mathrm { G } } A _ { i } ^ { \mathrm { G } } + w _ { \mathrm { S } } A _ { g } ^ { \mathrm { S } } ( s _ { i , t } , a _ { i , t } ) , } \end{array}\tag{14}
$$

with weights $w _ { \mathrm { G } } , w _ { \mathrm { S } } \geq 0 .$ , giving the CRBC objective

$$
\mathcal { I } _ { \mathrm { C R B C } } ( \theta ) = \mathbb { E } \bigg [ \frac { 1 } { \sum _ { i } T _ { i } } \sum _ { i , t } \operatorname* { m i n } \bigl ( \rho _ { i , t } A _ { i , t } ^ { \mathrm { f i n a l } } , \ \mathrm { c l i p } ( \rho _ { i , t } , 1 \pm \epsilon ) A _ { i , t } ^ { \mathrm { f i n a l } } \bigr ) \bigg ] - \lambda _ { \mathrm { K L } } \mathbb { D } _ { \mathrm { K L } } \big [ \pi _ { \theta } \| \pi _ { \mathrm { r e f } } \big ] ,\tag{15}
$$

where $\textstyle \sum _ { i } T _ { i }$ is the total number of environment steps in the group, and the remaining notation $( \rho _ { i , t } ,$ $\epsilon , \lambda _ { \mathrm { K L } } )$ is that of equation 2. Only the advantage differs from equation 2, so CRBC introduces no parametric critic and no additional rollouts. The advantage and the empirical process behind it receive no gradient.

## 5 EXPERIMENTS

Our experiments answer three questions: (1) does cross-rollout Bellman closure improve LLMagent training over trajectory-level and step-level baselines; (2) how does it behave over the course of training; and (3) how do its two design choices (combining step- and trajectory-level credit, and taking the closure depth to $K \to \infty )$ ) affect performance.

## 5.1 EXPERIMENT SETUP

Benchmarks and baselines. We evaluate CRBC on text-based ALFWorld (Shridhar et al., 2021) and WebShop (Yao et al., 2022) for household interaction and web shopping, respectively, and on visual $6 \times 6$ Sokoban (Schrader, 2018). Baselines include closed-source models GPT-4o (OpenAI, 2023) and Gemini-2.5-Pro (Team, 2023); prompting baselines Qwen2.5 (Qwen et al., 2025), ReAct

Table 1: Performance comparison on ALFWorld and WebShop. For ALFWorld, we report the average success rate (%) for each subtask and the overall success rate. For WebShop, we report the average task score and average success rate (%). Most results are averaged over three random seeds. The best results are highlighted in bold.
<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="7">ALFWorld</td><td colspan="2">WebShop</td></tr><tr><td>Pick</td><td>Clean</td><td>Cool</td><td>Look</td><td>Heat</td><td>Pick2</td><td>All</td><td>Score</td><td>Succ.</td></tr><tr><td colspan="2">Closed-Source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prompting</td><td>GPT-4o</td><td>75.30</td><td>60.80</td><td>31.20</td><td>56.70</td><td>21.60</td><td>49.80</td><td>48.00</td><td>31.80</td><td>23.70</td></tr><tr><td>Prompting</td><td>Gemini-2.5-Pro</td><td>92.80</td><td>63.30</td><td>62.10</td><td>69.00</td><td>26.60</td><td>58.70</td><td>60.30</td><td>42.50</td><td>35.90</td></tr><tr><td colspan="2">Qwen2.5-1.5B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prompting</td><td>Qwen2.5</td><td>5.90</td><td>5.50</td><td>3.30</td><td>9.70</td><td>4.20</td><td>0.00</td><td>4.10</td><td>23.10</td><td>5.20</td></tr><tr><td>Prompting</td><td>ReAct</td><td>17.40</td><td>20.50</td><td>15.70</td><td>6.20</td><td>7.70</td><td>2.00</td><td>12.80</td><td>40.10</td><td>11.30</td></tr><tr><td>Prompting</td><td>Reflexion</td><td>35.30</td><td>22.20</td><td>21.70</td><td>13.60</td><td>19.40</td><td>3.70</td><td>21.80</td><td>55.80</td><td>21.90</td></tr><tr><td>RL Training</td><td>PPO</td><td> $6 4 . 8 0 _ { ( 3 . 5 0 ) }$ </td><td> $4 0 . 5 0 _ { ( 6 . 9 0 ) }$ </td><td> $5 7 . 1 0 _ { ( 4 . 9 0 ) }$ </td><td> $6 0 . 6 0 _ { ( 6 . 6 0 ) }$ </td><td> $4 6 . 4 0 _ { ( 4 . 0 0 ) }$ </td><td> $4 7 . 4 0 _ { ( 1 . 9 0 ) }$ </td><td>54.30(3.10)</td><td></td><td>73.80(3.00) 51.50(2.90)</td></tr><tr><td>RL Training</td><td>RLOO</td><td> $8 8 . 3 0 _ { ( 3 . 0 0 ) }$ </td><td> $5 2 . 8 0 _ { ( 8 . 6 0 ) }$ </td><td> $7 1 . 0 0 _ { ( 5 . 9 0 ) }$ </td><td> $6 2 . 8 0 _ { ( 8 . 7 0 ) }$ </td><td> $6 6 . 4 0 _ { ( 5 . 5 0 ) }$ </td><td> $5 6 . 9 0 _ { ( 4 . 7 0 ) }$ </td><td> $6 9 . 7 0 _ { ( 2 . 5 0 ) }$ </td><td></td><td> $7 3 . 9 0 _ { ( 5 . 6 0 ) } \ 5 2 . 1 0 _ { ( 6 . 7 0 ) }$ </td></tr><tr><td>RL Training</td><td>GRPO</td><td> $8 5 . 2 7 _ { ( 1 . 4 0 ) }$ </td><td> $6 5 . 7 9 _ { ( 2 . 6 3 ) }$ </td><td> $8 8 . 0 8 _ { ( 8 . 0 8 ) }$ </td><td> $5 8 . 3 3 _ { ( 0 . 0 0 ) }$ </td><td> $7 8 . 5 7 _ { ( 7 . 1 4 ) }$ </td><td> $7 5 . 0 0 _ { ( 5 . 0 0 ) }$ </td><td> $7 7 . 7 3 _ { ( 1 . 9 5 ) }$ </td><td></td><td> $8 3 . 3 3 _ { ( 3 . 5 2 ) } ~ 7 1 . 0 9 _ { ( 0 . 7 8 ) }$ </td></tr><tr><td>RL Training</td><td>GiGPO</td><td> $9 6 . 6 7 _ { ( 4 . 7 1 ) }$ </td><td> $8 4 . 2 1 _ { ( 4 . 3 0 ) }$ </td><td> $8 8 . 2 1 _ { ( 3 . 0 2 ) }$ </td><td> $7 2 . 2 2 _ { ( 3 . 9 3 ) }$ </td><td> $9 5 . 2 4 _ { ( 3 . 8 9 ) }$ </td><td> $8 8 . 3 3 _ { ( 2 . 3 6 ) }$ </td><td> $8 9 . 3 2 _ { ( 1 . 9 5 ) }$ </td><td></td><td> $8 8 . 9 3 _ { ( 1 . 5 5 ) } 7 5 . 7 8 _ { ( 2 . 2 1 ) }$ </td></tr><tr><td>RL Training</td><td>GraphGPO</td><td> $9 5 . 6 3 _ { ( 3 . 0 9 ) }$ </td><td> $8 4 . 2 1 \dot { _ { ( 4 . 3 0 ) } }$ </td><td> $8 8 . 3 1 _ { ( 0 . 2 2 ) }$ </td><td> $8 8 . 8 9 _ { ( 3 . 9 3 ) }$ </td><td> $9 2 . 0 6 _ { ( 5 . 9 4 ) }$ </td><td> $8 8 . 3 3 _ { ( 2 . 3 6 ) }$ </td><td> $9 0 . 1 0 _ { ( 1 . 8 4 ) } ^ { \cdot }$ </td><td></td><td> $8 9 . 6 0 _ { ( 1 . 4 6 ) } \ 7 9 . 6 9 _ { ( 1 . 2 8 ) }$ </td></tr><tr><td>RL Training</td><td>HGPO</td><td> $9 4 . 3 0 _ { ( 2 . 9 0 ) }$ </td><td> $6 9 . 1 0 _ { ( 9 . 2 0 ) }$ </td><td> $\mathbf { 9 5 . 9 0 } _ { ( 2 . 8 0 ) } ^ { \cdot }$ </td><td> $\mathbf { 9 7 . 4 0 } _ { ( 3 . 6 0 ) }$ </td><td> $9 2 . 0 0 \dot { _ { ( 2 . 3 0 ) } }$ </td><td> $8 2 . 3 0 _ { ( 3 . 2 0 ) }$ </td><td> $9 0 . 5 0 _ { ( 0 . 4 0 ) } ^ { \cdot }$ </td><td></td><td> $8 5 . 5 0 _ { ( 0 . 5 0 ) } \ 7 0 . 5 0 _ { ( 1 . 7 0 ) }$ </td></tr><tr><td>RL Training</td><td>CRBC (Ours)</td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $9 2 . 9 8 _ { ( 6 . 5 6 ) }$ </td><td> $9 2 . 1 5 _ { ( 3 . 3 3 ) }$ </td><td> $9 1 . 6 7 _ { ( 0 . 0 0 ) }$ </td><td> $\mathbf { 9 6 . 8 3 } _ { ( 4 . 4 9 ) }$ </td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $\mathbf { 9 6 . 0 9 } _ { ( 0 . 6 4 ) }$ </td><td></td><td> $\mathbf { 9 0 . 1 1 } _ { ( 1 . 2 4 ) } \ \mathbf { 8 1 . 5 1 } _ { ( 0 . 7 4 ) }$ </td></tr><tr><td colspan="2">Qwen2.5-7B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prompting</td><td>Qwen2.5</td><td>33.40</td><td>21.60</td><td>19.30</td><td>6.90</td><td>2.80</td><td>3.20</td><td>14.80</td><td>26.40</td><td>7.80</td></tr><tr><td>Prompting</td><td>ReAct</td><td>48.50</td><td>35.40</td><td>34.30</td><td>13.20</td><td>18.20</td><td>17.60</td><td>31.20</td><td>46.20</td><td>19.50</td></tr><tr><td>Prompting</td><td>Reflexion</td><td>62.00</td><td>41.60</td><td>44.90</td><td>30.90</td><td>36.30</td><td>23.80</td><td>42.70</td><td>58.10</td><td>28.80</td></tr><tr><td>RL Training</td><td>PPO</td><td> $9 2 . 3 0 _ { ( 4 . 0 0 ) }$ </td><td> $6 4 . 0 0 _ { ( 8 . 4 0 ) }$ </td><td> $9 2 . 5 0 _ { ( 2 . 4 0 ) }$ </td><td> $8 9 . 5 0 _ { ( 7 . 0 0 ) }$ </td><td> $8 0 . 3 0 _ { ( 2 . 0 0 ) }$ </td><td> $6 8 . 8 0 _ { ( 8 . 3 0 ) }$ </td><td>80.40(2.70)</td><td>81.40(3.10)</td><td>68.70(5.10)</td></tr><tr><td>RL Training</td><td>RLOO</td><td> $8 7 . 6 0 \dot { _ { ( 4 . 3 0 ) } }$ </td><td> $7 8 . 2 0 _ { ( 8 . 3 0 ) }$ </td><td> $8 7 . 3 0 _ { ( 5 . 8 0 ) }$ </td><td> $8 1 . 3 0 _ { ( 7 . 6 0 ) }$ </td><td> $7 1 . 9 0 _ { ( 5 . 2 0 ) }$ </td><td> $4 8 . 9 0 _ { ( 8 . 4 0 ) }$ </td><td> $7 5 . 5 0 _ { ( 4 . 6 0 ) }$ </td><td>80.30(3.20)</td><td>65.70(4.00)</td></tr><tr><td>RL Training</td><td>GRPO</td><td> $8 8 . 9 8 _ { ( 5 . 3 0 ) }$ </td><td> $9 1 . 9 8 _ { ( 4 . 4 3 ) }$ </td><td> $7 7 . 8 9 _ { ( 4 . 5 8 ) }$ </td><td> $7 8 . 5 7 _ { ( 0 . 0 0 ) }$ </td><td> $\mathbf { 9 0 . 7 4 } _ { ( 5 . 2 4 ) }$ </td><td> $7 1 . 4 3 _ { ( 3 . 8 9 ) }$ </td><td> $8 3 . 3 3 _ { ( 2 . 0 5 ) }$ </td><td></td><td> $8 4 . 3 1 _ { ( 1 . 2 7 ) } 7 5 . 0 0 _ { ( 2 . 7 8 ) }$ </td></tr><tr><td>RL Training</td><td>GiGPO</td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $9 2 . 1 1 _ { ( 7 . 8 9 ) }$ </td><td> $8 0 . 7 7 _ { ( 7 . 6 9 ) }$ </td><td> $9 5 . 8 3 _ { ( 4 . 1 7 ) }$ </td><td> $9 0 . 4 8 _ { ( 0 . 0 0 ) }$ </td><td> $9 0 . 0 0 _ { ( 5 . 0 0 ) }$ </td><td> $9 1 . 4 1 _ { ( 2 . 3 4 ) }$ </td><td></td><td> $8 6 . 6 0 _ { ( 1 . 1 7 ) } \ 7 6 . 5 6 _ { ( 1 . 5 6 ) }$ </td></tr><tr><td>RL Training</td><td>GraphGPO</td><td> $9 7 . 8 5 _ { ( 3 . 0 4 ) }$ </td><td> $8 9 . 4 7 _ { ( 7 . 4 4 ) }$ </td><td> $8 8 . 3 1 _ { ( 0 . 2 2 ) }$ </td><td> $9 4 . 4 4 _ { ( 3 . 9 3 ) }$ </td><td> $8 2 . 5 4 _ { ( 8 . 0 9 ) }$ </td><td> $8 6 . 6 7 _ { ( 6 . 2 4 ) }$ </td><td> $9 0 . 1 0 _ { ( 3 . 5 1 ) }$ </td><td></td><td> $8 7 . 2 4 _ { ( 1 . 5 7 ) } ~ 7 8 . 1 3 _ { ( 0 . 0 0 ) }$ </td></tr><tr><td>RL Training</td><td>CRBC (Ours)</td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td>94.23(5.77)</td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $9 0 . 4 8 _ { ( 0 . 0 0 ) }$ </td><td> $\mathbf { 9 7 . 5 0 } _ { ( 2 . 5 0 ) }$ </td><td> $\mathbf { 9 6 . 8 8 } _ { ( 1 . 5 6 ) }$ </td><td></td><td> $9 0 . 4 3 _ { ( 1 . 2 5 ) } \ 8 1 . 5 1 _ { ( 3 . 5 1 ) }$ </td></tr></table>

(Yao et al., 2023), and Reflexion (Shinn et al., 2023); and RL methods PPO (Schulman et al., 2017), RLOO (Ahmadian et al., 2024), GRPO (Shao et al., 2024), GiGPO (Feng et al., 2025), GraphGPO (Cheng et al., 2026), and HGPO (He et al., 2026). For Sokoban, group-based RL baselines share the same VLM backbone. Benchmark details and baseline descriptions, including implementation provenance, appear in Appendices C.2 and C.1, respectively.

Implementation details. We use Qwen2.5-1.5B-Instruct and Qwen2.5-7B-Instruct (Qwen et al., 2025) for ALFWorld and WebShop, and Qwen2.5-VL-3B-Instruct (Bai et al., 2025) for Sokoban. Within each benchmark and model scale, reproduced RL methods share the same data, rollout, and optimization protocol, differing only in the advantage estimator and its method-specific settings. Each update uses $G = 8$ rollouts per group, with 16 groups for ALFWorld/WebShop and 32 for Sokoban, and episodes capped at 50, 15, and 15 environment steps, respectively. Rollout and validation temperatures are 1.0 and 0.4, the actor learning rate is $1 0 ^ { - 6 }$ , and the KL coefficient is 0.01; we train for 150 updates and validate every five. All local runs use one node of eight 80 GB NVIDIA A100 GPUs. Unless stated otherwise, CRBC uses $\beta = 0 . 9 8 ,$ full closure $( K \to \infty )$ , and $( w _ { \mathrm { G } } , w _ { \mathrm { S } } ) = ( 1 , 5 )$ without auxiliary predictors. Full configurations appear in Appendix C.3.

## 5.2 EXPERIMENTAL RESULTS

Performance on agentic benchmarks. Table 1 reports the comparison between CRBC and the baselines on ALFWorld and WebShop, and Table 3 reports the results on Sokoban. Across both model sizes and all three environments, CRBC consistently outperforms every baseline in overall success. On ALFWorld with Qwen2.5-1.5B-Instruct, CRBC reaches an overall success rate of 96.09%, improving over GRPO by 18.36 points and over the strongest baseline (HGPO, 90.50%) by 5.59 points; it improves over GRPO on every subtask, ranks first on four of the six, and solves Pick and Pick2 perfectly. On WebShop, CRBC attains the best task score (90.11) and success rate (81.51%). The advantage persists at the 7B scale, where CRBC achieves 96.88% on ALFWorld and 81.51% success on WebShop, surpassing all baselines on both overall metrics. On the visionlanguage Sokoban benchmark (Table 3), CRBC reaches 82.81% success, exceeding GraphGPO by 5.21 points, GiGPO by 6.25 points, and GRPO by a large margin. These results show that recovering high-fidelity step-level credit through cross-rollout Bellman closure translates into consistent end-task gains across embodied, web, and visual multi-turn settings.

Table 2: Closure-depth ablation on ALFWorld with Qwen2.5-1.5B-Instruct. Only K varies; all other settings are fixed. Best values are in bold.  
Table 3: Test performance on $6 \times 6$ Sokoban with Qwen2.5-VL-3B-Instruct. Results are averaged over three seeds; best values are in bold.
<table><tr><td>Depth K</td><td>Step-credit</td><td>Score</td><td>Success (%)</td></tr><tr><td>0</td><td> $A ^ { ( 0 ) }$ </td><td> $7 . 0 0 _ { ( 0 . 2 7 ) }$ </td><td> $9 2 . 9 7 _ { ( 0 . 0 0 ) }$ </td></tr><tr><td>1</td><td> $A ^ { ( 1 ) }$ </td><td> $7 . 0 5 _ { ( 0 . 5 3 ) }$ </td><td> $9 2 . 9 7 _ { ( 1 . 5 6 ) }$ </td></tr><tr><td>2</td><td> $A ^ { ( 2 ) }$ </td><td> $7 . 1 6 _ { ( 1 . 2 0 ) }$ </td><td> $9 3 . 7 5 _ { ( 3 . 1 2 ) }$ </td></tr><tr><td>3</td><td> $A ^ { ( 3 ) }$ </td><td> $7 . 2 1 _ { ( 0 . 1 3 ) }$ </td><td> $9 3 . 7 5 _ { ( 0 . 7 8 ) }$ </td></tr><tr><td>∞</td><td> $A ^ { ( \infty ) }$ </td><td> $\mathbf { 8 . 0 1 } _ { ( 0 . 2 7 ) }$ </td><td> $\mathbf { 9 6 . 0 9 } _ { ( 0 . 6 4 ) }$ </td></tr></table>

<table><tr><td>Type</td><td>Method</td><td>Score</td><td>Success (%)</td></tr><tr><td>Prompting</td><td> $_ { \mathrm { Q w e n } 2 . 5 - \mathrm { V L } ^ { * } }$ </td><td></td><td>11.70</td></tr><tr><td>RL Training</td><td>GRPO{†</td><td> $1 . 8 1 _ { ( 2 . 5 0 ) }$ </td><td> $4 6 . 4 8 _ { ( 3 . 6 4 ) }$ </td></tr><tr><td>RL Training</td><td>GiGPO</td><td> $4 . 0 0 _ { ( 1 . 1 2 ) }$ </td><td> $7 6 . 5 6 _ { ( 5 . 4 7 ) }$ </td></tr><tr><td>RL Training</td><td>GraphGPO</td><td> $4 . 1 3 _ { ( 0 . 3 4 ) }$ </td><td> $7 7 . 6 0 _ { ( 2 . 8 8 ) }$ </td></tr><tr><td>RL Training</td><td>CRBC (Ours)</td><td> ${ \pmb 5 . 0 6 } _ { ( 0 . 5 4 ) }$ </td><td> ${ \bf 8 2 . 8 1 } _ { ( 2 . 3 0 ) }$ </td></tr></table>

Table 4: Credit and weight ablations on ALFWorld with Qwen2.5-1.5B-Instruct. We report final validation success rates (%) for six subtasks and the overall success rate (All) at step 150, averaged over three random seeds. Best values are in bold.
<table><tr><td>WG</td><td>wS</td><td>Credit assignment</td><td>Pick</td><td>Clean</td><td>Cool</td><td>Look</td><td>Heat</td><td>Pick2</td><td>All</td></tr><tr><td>0</td><td>5</td><td>Step-level only</td><td> $9 8 . 8 9 _ { ( 1 . 5 7 ) }$ </td><td> $\mathbf { 9 8 . 2 5 } _ { ( 2 . 4 8 ) }$ </td><td> $9 3 . 5 4 _ { ( 1 . 7 4 ) }$ </td><td> $\mathbf { 9 4 . 4 4 } _ { ( 3 . 9 3 ) }$ </td><td> $8 8 . 8 9 _ { ( 2 . 2 4 ) }$ </td><td> $9 6 . 6 7 _ { ( 2 . 3 6 ) }$ </td><td> $9 5 . 3 1 _ { ( 1 . 6 9 ) }$ </td></tr><tr><td>1</td><td>0</td><td>Trajectory-level only</td><td> $8 4 . 6 2 _ { ( 1 . 4 6 ) }$ </td><td> $6 4 . 9 1 _ { ( 2 . 4 8 ) }$ </td><td> $8 6 . 9 2 _ { ( 6 . 7 9 ) }$ </td><td> $5 8 . 3 3 _ { ( 0 . 0 0 ) }$ </td><td> $7 7 . 7 8 _ { ( 5 . 9 4 ) }$ </td><td> $7 3 . 3 3 _ { ( 4 . 7 1 ) }$ </td><td> $7 6 . 5 6 _ { ( 2 . 3 0 ) }$ </td></tr><tr><td>1</td><td>1</td><td>Step + trajectory</td><td> $9 8 . 8 9 _ { ( 1 . 5 7 ) }$ </td><td> $8 7 . 7 2 _ { ( 2 . 4 8 ) }$ </td><td> $9 4 . 8 2 _ { ( 1 . 7 8 ) }$ </td><td> $\mathbf { 9 4 . 4 4 } _ { ( 3 . 9 3 ) }$ </td><td> $8 8 . 8 9 _ { ( 2 . 2 4 ) }$ </td><td> $9 3 . 3 3 _ { ( 2 . 3 6 ) }$ </td><td> $9 3 . 7 5 _ { ( 1 . 6 9 ) }$ </td></tr><tr><td>1</td><td>2.5</td><td>Step + trajectory</td><td> $9 8 . 8 9 _ { ( 1 . 5 7 ) }$ </td><td> $9 2 . 9 8 _ { ( 4 . 9 6 ) }$ </td><td> $\mathbf { 9 7 . 3 8 } _ { ( 1 . 8 5 ) }$ </td><td> $8 6 . 1 1 _ { ( 3 . 9 3 ) }$ </td><td> $9 2 . 0 6 _ { ( 2 . 2 4 ) }$ </td><td> $9 3 . 3 3 _ { ( 2 . 3 6 ) }$ </td><td> $9 4 . 5 3 _ { ( 1 . 9 1 ) }$ </td></tr><tr><td>1</td><td>5</td><td>Step + trajectory (Ours)</td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $9 2 . 9 8 _ { ( 6 . 5 6 ) }$ </td><td> $9 2 . 1 5 _ { ( 3 . 3 3 ) }$ </td><td> $9 1 . 6 7 _ { ( 0 . 0 0 ) }$  </td><td> $\mathbf { 9 6 . 8 3 } _ { ( 4 . 4 9 ) }$ </td><td>100.00(0.00)</td><td> $\mathbf { 9 6 . 0 9 } _ { ( 0 . 6 4 ) }$ </td></tr><tr><td>1</td><td>10</td><td> $\mathbf { S t e p } + \mathbf { t r a j e c t o r y }$ </td><td> $9 8 . 8 9 _ { ( 1 . 5 7 ) }$ </td><td> $9 2 . 9 8 _ { ( 2 . 4 8 ) }$ </td><td> $9 4 . 8 7 _ { ( 3 . 6 3 ) }$ </td><td> $9 1 . 6 7 _ { ( 0 . 0 0 ) }$ </td><td> $8 8 . 8 9 _ { ( 2 . 2 4 ) }$ </td><td> $9 8 . 3 3 _ { ( 2 . 3 6 ) }$ </td><td> $9 4 . 7 9 _ { ( 1 . 9 5 ) }$ </td></tr></table>

Training dynamics. Figure 3 shows mean trajectory reward for CRBC, GraphGPO, GiGPO, and GRPO, with validation success rates in Figure 6 (appendix). Across the three environments, steplevel methods improve faster and attain higher rewards than trajectory-level GRPO. CRBC rises fastest during early and middle training and achieves the highest final reward; on ALFWorld with Qwen2.5-1.5B-Instruct, it matches GraphGPO’s final reward in roughly half the updates. Similar validation trends show improved held-out success alongside the reward gains, supporting the learning-efficiency benefit of cross-rollout Bellman closure.

## 5.3 ABLATION STUDY

Closure depth. Table 2 varies only the closure depth K, keeping the aggregation rule, anchor map ϕ, and other settings fixed. Figure 7 shows that reward, validation success, and validation score generally improve faster as K increases, with $K = 3$ and full closure showing the strongest early trends. Final success rates are 92.97% for $K = 0 , 1$ , 93.75% for $K = 2 , { \bar { 3 } } ,$ and 96.09% for full closure, a 3.12-point gain over the visit-local estimator; validation score also rises from 7.00 at $K = 0$ to 8.01 under full closure. Figure 5 provides a structural explanation using rollouts from an early-training checkpoint: the proportion of eligible visits with access to all eight rollouts increases from $1 2 . 9 \%$ at $\bar { K _ { \mathbf { \theta } } } = \mathbf { \epsilon } _ { 0 }$ to $6 1 . 5 \%$ under full closure, while the paired heatmap shows expanded coverage for 63.0% of eligible visits. These results support the benefit of deeper Bellman propagation and illustrate how closure makes additional cross-rollout evidence accessible beyond the current anchor.

Credit assignment and weight sensitivity. Table 4 evaluates the trajectory-level and step-level weights $w _ { \mathrm { G } }$ and w<sub>S</sub>. Trajectory-level credit alone (1, 0) achieves 76.56% overall success, whereas step-level closure credit alone (0, 5) reaches 95.31%. With $w _ { \mathrm { G } } = 1$ , increasing w from 1 to 2.5 and 5 raises success from 93.75% to 94.53% and 96.09%; a further increase to 10 reduces it to 94.79%. These results suggest that closure credit provides the main gains, complemented by appropriately weighted trajectory-level credit. The default (1, 5) achieves the highest overall success, exceeding step-level credit alone by 0.78 points.

![](images/14322629c869c0157655109712d2fbf6531d1557efff29360cbf8cb207eb8db5.jpg)

![](images/1782d4a5c45678bef1ad0071d4f4a38c4d7632870dd204255fb1f1aca60a080c.jpg)

![](images/d4c19ee96926bbe5c0494a1e65508db321db48fa47845d506e9292c2b0300e29.jpg)  
Figure 3: Training reward curves for CRBC (Ours, red), GraphGPO (blue), GiGPO (green), and GRPO (gray) on ALFWorld, WebShop, and Sokoban. Light and dark curves denote per-update and EMA-smoothed rewards $( \alpha = 0 . 9 5 )$ , respectively. Qwen2.5-1.5B-Instruct is used for ALF-World/WebShop and Qwen2.5-VL-3B-Instruct for Sokoban; panel scales are independent.

Computational overhead. Figure 4 presents the per-update runtime breakdown of CRBC on ALFWorld with Qwen2.5-1.5B-Instruct, averaged over 120 non-validation updates from a profiling run. An update takes 187.46 s on average, dominated by rollout (154.99 s) and the policy update (20.35 s). The trajectory-level group advantage $A ^ { \mathrm { G } }$ and the instrumented CRBC step-credit core $A ^ { \mathrm { S } }$ take only 0.044 s and 0.057 s, respectively, corresponding to 0.02% and 0.03% of the total update time. Including tensor preparation and advantage assembly, the complete advantage block takes 0.330 s, or 0.18% of the update. Since CRBC reuses the same rollout

![](images/2ffa3c905add3fdfb260e9a89ede4e0f8293e02780dfd935e59aab3b3d5c6f9d.jpg)  
Figure 4: Per-update runtime breakdown of CRBC on ALFWorld with Qwen2.5-1.5B-Instruct. Blue bars denote shared stages, while red bars denote trajectory-level and step-level credit construction.

and actor-update pipeline without introducing additional model rollouts or a critic network, its credit construction remains negligible relative to the main training stages.

## 6 CONCLUSION

We presented CRBC, which evaluates each rollout group’s empirical process at its behavior-policy Bellman fixed point to derive critic-free $Q - V$ credit with global cross-rollout propagation and behavior-consistent aggregation. Feasible policy reweighting along this credit does not decrease discounted success value on the fixed process. Combined with trajectory-level advantages, the normalized credit improves performance and learning efficiency on ALFWorld, WebShop, and Sokoban without additional rollouts, supported by depth and credit ablations.

## REFERENCES

Arash Ahmadian, Chris Cremer, Matthias Galle, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin,´ Ahmet Ust<sup>¨</sup> un, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learn-¨ ing from human feedback in llms, 2024. URL https://arxiv.org/abs/2402.14740.

Hao Bai, Yifei Zhou, Mert Cemri, Jiayi Pan, Alane Suhr, Sergey Levine, and Aviral Kumar. Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning, 2024. URL https://arxiv.org/abs/2406.11896.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.

Kevin Chen, Marco Cusumano-Towner, Brody Huval, Aleksei Petrenko, Jackson Hamburger, Vladlen Koltun, and Philipp Krahenb ¨ uhl. Reinforcement learning for long-horizon interactive ¨ llm agents, 2025. URL https://arxiv.org/abs/2502.01600.

Xin Cheng, Shuo He, Lang Feng, HaiYang Xu, Ming Yan, Lei Feng, and Bo An. Beyond trajectorylevel attribution: Graph-based credit assignment for agentic reinforcement learning, 2026. URL https://arxiv.org/abs/2605.26684.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948, 2025. doi: 10.48550/ARXIV.2501.12948. URL https://doi.org/ 10.48550/arXiv.2501.12948.

Mingxuan Fan and Peiyang Liu. Progress- and reliability-oriented group policy optimization for agentic reinforcement learning, 2026. URL https://arxiv.org/abs/2607.04242.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training, 2025. URL https://arxiv.org/abs/2505.10978.

Hiroki Furuta, Kuang-Huei Lee, Ofir Nachum, Yutaka Matsuo, Aleksandra Faust, Shixiang Shane Gu, and Izzeddin Gur. Multimodal web navigation with instruction-finetuned foundation models, 2024. URL https://arxiv.org/abs/2305.11854.

Shuo He, Lang Feng, Qi Wei, Xin Cheng, Lei Feng, and Bo An. Hierarchy-of-groups policy optimization for long-horizon agentic tasks, 2026. URL https://arxiv.org/abs/2602. 22817.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning, 2025. URL https://arxiv.org/abs/2503.09516.

Yuanfan Li, Qi Zhou, Wenjing Duan, and Lu Chen. When denser credit is not enough: Evidencecalibrated policy optimization for long-horizon llm agent training, 2026. URL https:// arxiv.org/abs/2606.05885.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id= v8L0pN6EOi.

Zhihang Lin, Mingbao Lin, Yuan Xie, and Rongrong Ji. Cppo: Accelerating the training of group relative policy optimization-based reasoning models, 2025. URL https://arxiv.org/ abs/2503.22342.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective, 2025. URL https://arxiv. org/abs/2503.20783.

Volodymyr Mnih, Koray Kavukcuoglu, David Silver, Andrei A Rusu, Joel Veness, Marc G Bellemare, Alex Graves, Martin Riedmiller, Andreas K Fidjeland, Georg Ostrovski, et al. Human-level control through deep reinforcement learning. Nature, 518(7540):529–533, 2015.

Karthik Narasimhan, Tejas Kulkarni, and Regina Barzilay. Language understanding for text-based games using deep reinforcement learning. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pp. 1–11, 2015.

OpenAI. GPT-4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303.08774. URL https://doi.org/10.48550/arXiv.2303.08774.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback, 2022. URL https://arxiv.org/abs/2203.02155.

Xue Bin Peng, Aviral Kumar, Grace Zhang, and Sergey Levine. Advantage-weighted regression: Simple and scalable off-policy reinforcement learning, 2019. URL https://arxiv.org/ abs/1910.00177.

Pranav Putta, Edmund Mills, Naman Garg, Sumeet Motwani, Chelsea Finn, Divyansh Garg, and Rafael Rafailov. Agent q: Advanced reasoning and learning for autonomous ai agents, 2024. URL https://arxiv.org/abs/2408.07199.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https://arxiv.org/abs/2305.18290.

Timo Schick, Jane Dwivedi-Yu, Roberto Dess\`ı, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools, 2023. URL https://arxiv.org/abs/2302.04761.

Max-Philipp B. Schrader. gym-sokoban. https://github.com/mpSchrader/ gym-sokoban, 2018.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402. 03300.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL https://arxiv.org/abs/2303.11366.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cotˆ e, Yonatan Bisk, Adam Trischler, and Matthew´ Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning, 2021. URL https://arxiv.org/abs/2010.03768.

David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. Nature, 550(7676):354–359, 2017.

Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul Christiano. Learning to summarize from human feedback, 2022. URL https://arxiv.org/abs/2009.01325.

Hui-Ze Tan, Xiao-Wen Yang, Hao Chen, Jie-Jing Shao, Yi Wen, Yuteng Shen, Weihong Luo, Xiku Du, Lan-Zhe Guo, and Yu-Feng Li. Hindsight credit assignment for long-horizon llm agents, 2026. URL https://arxiv.org/abs/2603.08754.

Gemini Team. Gemini: A family of highly capable multimodal models. CoRR, abs/2312.11805, 2023. doi: 10.48550/ARXIV.2312.11805. URL https://doi.org/10.48550/arXiv. 2312.11805.

Kimi Team. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025. doi: 10.48550/ARXIV.2501.12599. URL https://doi.org/10.48550/arXiv.2501. 12599.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. Appworld: A controllable world of apps and people for benchmarking interactive coding agents, 2024. URL https://arxiv. org/abs/2407.18901.

Daoyu Wang, Qingchuan Li, Mingyue Cheng, Jie Ouyang, Shuo Yu, Chunli Liu, Shijin Wang, Qi Liu, and Enhong Chen. Capo: Critic-guided action-aligned policy optimization for advancing llm agent capabilities, 2026a. URL https://arxiv.org/abs/2604.18401.

Junzhe Wang, Zhiheng Xi, Yajie Yang, Hao Luo, Shihan Dou, Tao Gui, and Qi Zhang. Enhancing llm-based search agents via contribution weighted group relative policy optimization, 2026b. URL https://arxiv.org/abs/2604.14267.

Tao Wang, Suhang Zheng, and Xiaoxiao Xu. Rtmc: Step-level credit assignment via rollout trees, 2026c. URL https://arxiv.org/abs/2604.11037.

Yunan Wang, Minghui Song, Zihan Zhang, Shaohan Huang, Haizhen Huang, Furu Wei, Weiwei Deng, Feng Sun, and Qi Zhang. Group-graph policy optimization for long-horizon agentic reinforcement learning, 2026d. URL https://arxiv.org/abs/2606.22995.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei-Fei, Lijuan Wang, Yejin Choi, and Manling Li. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning, 2025. URL https://arxiv.org/abs/ 2504.20073.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. Qwen2 technical report. CoRR, abs/2407.10671, 2024. doi: 10.48550/ARXIV.2407.10671. URL https://doi.org/10.48550/arXiv.2407. 10671.

Kaibing Yang, Guangfeng Cai, Shengtian Yang, Shuo He, Yu Li, Mengyi Liu, Pengwei Chen, Jun Xu, and Lei Feng. Progress-conditioned group policy optimization for long-horizon agentic tasks, 2026. URL https://arxiv.org/abs/2607.22724.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 20744–20757. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-1508. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/82ad13ec01f9fe44c01cb91814fd7b8c-Paper-Conference.pdf.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. URL https://arxiv. org/abs/2210.03629.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https://arxiv.org/abs/2503.14476.

Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, and Sergey Levine. Fine-tuning large vision-language models as decision-making agents via reinforcement learning, 2024. URL https://arxiv.org/abs/ 2405.10292.

Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. Archer: Training language model agents via hierarchical multi-turn rl, 2024. URL https://arxiv.org/abs/2402. 19446.

Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences, 2020. URL https://arxiv.org/abs/1909.08593.

## A ALGORITHM

Algorithm 1 CRBC with full Bellman closure   
Require: Initial policy $\pi _ { \theta } ,$ , fixed reference policy $\pi _ { \mathrm { r e f } }$ , task distribution $p ,$ anchor map $\phi ,$ discount   
${ \boldsymbol { \beta } } \in ( 0 , 1 )$ , advantage weights w<sub>G</sub>, w<sub>S</sub>, clipping parameter ϵ, KL penalty $\lambda _ { \mathrm { K L } }$ , group size $G ,$   
normalization floor $\sigma _ { \operatorname* { m i n } } > 0$   
1: for each training iteration do   
2: Set $\theta _ { \mathrm { o l d } }  \theta$ and sample a batch of tasks from $p$   
3: for each task $g$ in the batch do   
4: // Group rollout   
5: Initialise $G$ environments from the same task condition   
6: Collect complete rollouts using $y _ { i , t } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid h _ { i , t } )$ and $a _ { i , t } = \operatorname { E x e c } ( y _ { i , t } ) ;$ ; retain contexts,   
responses, and terminal outcomes $R _ { i }$   
7: // Empirical process and Bellman closure   
8: Map $s _ { i , t } = \phi ( h _ { i , t } )$ and construct $\mathcal { D } _ { g }$ with successors $x _ { i , t }$ and absorbing boundaries as in   
Section 3   
9: Build $( P _ { g } , \mu _ { g } )$ from transition counts using equation 5   
10: Form $M _ { g } , b _ { g }$ using equation 8 and solve $( \breve { I } - \dot { \beta } M _ { g } ) V _ { g } = b _ { g }$   
11: Set $V _ { g } ( z ^ { + } ) = 1$ and $\begin{array} { r } { \dot { V } _ { g } ( z ^ { - } ) = 0 } \end{array}$ , then compute $\bar { Q _ { g } } = \bar { B _ { g } } \bar { V _ { g } }$   
12: // Credit construction and combination   
13: Compute trajectory-level group-relative advantages $A _ { i } ^ { \mathrm { G } }$ using the underlying optimizer   
14: Compute $A _ { g } ^ { ( \infty ) } ( s , a ) = Q _ { g } ( s , a ) - V _ { g } ( s )$ and its standard deviation $\sigma _ { g } ( s )$ under $\mu _ { g } ( \cdot \mid s )$   
at each anchor   
15: Compute the gated and normalized step-level credit $A _ { g } ^ { \mathrm { S } }$ using equation 13   
16: Set $A _ { i , t } ^ { \mathrm { f i n a l } } = w _ { \mathrm { G } } A _ { i } ^ { \mathrm { G } } + w _ { \mathrm { S } } A _ { g } ^ { \mathrm { S } } ( s _ { i , t } , a _ { i , t } )$ for every step in the group   
17: end for   
18: // Policy update   
19: Hold advantages and graph statistics fixed, and update θ using the objective ${ \mathcal { I } } _ { \mathrm { C R B C } } ( \theta )$ in   
equation 15   
20: end for

Algorithm 1 summarizes the full-closure training procedure. At each iteration, CRBC constructs an empirical process for each rollout group, solves its behaviour-policy Bellman fixed point, and combines the resulting normalized step credit with trajectory-level group-relative advantages for policy optimization.

## B PROOFS

All value functions, including $V _ { g } ^ { ( 0 ) }$ , have fixed boundary values $V ( z ^ { + } ) = 1$ and $V ( z ^ { - } ) = 0$ . Matrix expressions act on their restrictions to $\begin{array} { r } { {  { \boldsymbol { S } } } _ { g } . } \end{array}$ , where $\mathcal { T } _ { g } V \dot { = } \beta M _ { g } V \dot { + } b _ { g }$ and $\| M _ { g } \| _ { \infty } \leq 1$ . For a state vector u and a policy ν on the observed action support, define

$$
\begin{array} { c l } { { ( { \cal P } _ { g } ^ { S } u ) ( s , a ) : = \displaystyle \sum _ { s ^ { \prime } \in { \cal S } _ { g } } { \cal P } _ { g } ( s ^ { \prime } \mid s , a ) u ( s ^ { \prime } ) , \qquad ( L u ) ( s , a ) : = u ( s ) , } } \\ { { { } } } \\ { { M _ { g } ^ { \nu } ( s , s ^ { \prime } ) : = \displaystyle \sum _ { a \in { \cal A } _ { g } ( s ) } \nu ( a \mid s ) { \cal P } _ { g } ( s ^ { \prime } \mid s , a ) . } } \end{array}\tag{16}
$$

These operators are non-expansive in the sup norm. Let ${ \mathcal { T } } _ { g } ^ { \nu }$ be the state backup with $\mu _ { g }$ replaced by $\nu ,$ and $\dot { V } _ { g } ^ { \nu }$ its fixed point. Since boundary differences vanish, for $U , W \in \mathcal { V } _ { g } .$

$$
( \mathcal { B } _ { g } U ) ( s , a ) - ( \mathcal { B } _ { g } W ) ( s , a ) = \beta \big ( P _ { g } ^ { S } ( U - W ) \big ) ( s , a ) .\tag{17}
$$

## B.1 PROOF OF PROPOSITION 1

Invariance. For $V \in \mathcal { V } _ { g }$ and $s \in S _ { g } ,$

$$
0 \leq ( \mathcal { T } _ { g } V ) ( s ) = \beta \sum _ { a } \mu _ { g } ( a \mid s ) \sum _ { x } P _ { g } ( x \mid s , a ) V ( x ) \leq \beta < 1 .
$$

Together with the fixed boundary values, this shows that $\mathcal { T } _ { g }$ maps $\mathcal { V } _ { g }$ into itself.

Contraction and conclusion. For $U , V \in \mathcal { V } _ { g }$ , substochasticity gives

$$
\begin{array} { r } { \| \mathcal { T } _ { g } U - \mathcal { T } _ { g } V \| _ { \infty } = \beta \| M _ { g } ( U - V ) \| _ { \infty } \leq \beta \| U - V \| _ { \infty } . } \end{array}
$$

The space $\mathcal { V } _ { g }$ is closed in the finite-dimensional sup-norm space and hence complete. The Banach fixed point theorem therefore gives a unique fixed point, and Bellman iteration converges to it from every initializer with error at most $\beta ^ { K }$ times the initial error. Moreover, $V _ { g } ( s ) \in [ 0 , \breve { \beta } ]$ for $s \in \mathcal S _ { g }$ Since $\| \beta M _ { g } \| _ { \infty } \le \beta < 1$ , the inverse $\begin{array} { r } { ( I - \beta M _ { g } ) ^ { - 1 } = \sum _ { k > 0 } ( \beta M _ { g } ) ^ { k } } \end{array}$ exists, is entrywise nonnegative, and has norm at most $( 1 - \beta ) ^ { - 1 }$ . The fixed-point equation then yields equation 9. □

## B.2 PROOF OF PROPOSITION 2

Feasible direction. Centring gives $\begin{array} { r } { \sum _ { a } \xi ( s , a ) = w ( s ) \sum _ { a } \mu _ { g } ( a \mid s ) \tilde { A } ( s , a ) = 0 , } \end{array}$ , so $\nu _ { \alpha }$ sums to one at each anchor. Non-negativity holds for $0 \leq \alpha \leq \alpha \xi$ , where $\alpha _ { \xi } : = \operatorname* { m i n } \{ \mu _ { g } ( a \mid s ) / | \xi ( s , a ) |$ $\xi ( s , a ) < 0 \}$ , with $\alpha _ { \xi } = \infty$ when no negative entries exist.

Value difference. Using the fixed-point equation and centring of ${ \tilde { A } } ,$

$$
\begin{array} { l } { { \displaystyle \left( { \cal T } _ { g } ^ { \nu _ { \alpha } } V _ { g } - V _ { g } \right) ( s ) = \sum _ { a } \left[ \nu _ { \alpha } ( a \mid s ) - \mu _ { g } ( a \mid s ) \right] Q _ { g } ( s , a ) } } \\ { { { } } } \\ { { { } = \alpha w ( s ) \langle \tilde { A } , A _ { g } ^ { ( \infty ) } \rangle _ { \mu _ { g } } ( s ) . } } \end{array}\tag{18}
$$

Subtracting the two Bellman equations gives $\left( I - \beta M _ { q } ^ { \nu _ { \alpha } } \right) \left( V _ { q } ^ { \nu _ { \alpha } } - V _ { g } \right) = \mathcal { T } _ { q } ^ { \nu _ { \alpha } } V _ { g } - V _ { g } ,$ establishing equation 11. The resolvent uses the perturbed policy $\nu _ { \alpha }$ and is entrywise non-negative by its Neumann-series representation.

Self-alignment. For $\tilde { A } = A _ { g } ^ { ( \infty ) }$

$$
\begin{array} { r } { \big \langle A _ { g } ^ { ( \infty ) } , A _ { g } ^ { ( \infty ) } \big \rangle _ { \mu _ { g } } ( s ) = \operatorname { V a r } _ { a \sim \mu _ { g } ( \cdot \vert s ) } \big [ Q _ { g } ( s , a ) \big ] = \sigma _ { g } ( s ) ^ { 2 } \geq 0 . } \end{array}
$$

The forcing and resolvent are therefore non-negative, yielding $V _ { q } ^ { \nu _ { \alpha } } \succeq V _ { g }$ . Improvement is strict at s when $\alpha > 0$ and an anchor with $w > 0$ and $\sigma _ { g } > 0$ is reachable from s under $\nu _ { \alpha }$ , including a path of length zero. □

## B.3 RESIDUAL DECOMPOSITION AND FINITE-DEPTH TRANSPORT

Residual definitions. Define the action-level residual and its state-wise average:

$$
\begin{array} { l } { { \displaystyle \delta _ { g } ^ { ( 0 ) } ( s , a ) : = ( \mathcal { B } _ { g } V _ { g } ^ { ( 0 ) } ) ( s , a ) - Q _ { g } ^ { ( 0 ) } ( s , a ) = Q _ { g } ^ { ( 1 ) } ( s , a ) - Q _ { g } ^ { ( 0 ) } ( s , a ) , } } \\ { { \displaystyle \bar { \delta } _ { g } ^ { ( 0 ) } ( s ) : = \sum _ { a } \mu _ { g } ( a \mid s ) \delta _ { g } ^ { ( 0 ) } ( s , a ) = ( \mathcal { T } _ { g } V _ { g } ^ { ( 0 ) } ) ( s ) - V _ { g } ^ { ( 0 ) } ( s ) . } } \end{array}\tag{19}
$$

Action-level residuals can cancel in this average, so $\bar { \delta } _ { g } ^ { ( 0 ) } \equiv 0$ need not imply $\delta _ { g } ^ { ( 0 ) } \equiv 0$

Provenance. For an observed transition $( s , a , x )$ , define the suffix evidence arriving from $( s , a )$ by

$$
V _ { g  ( s , a ) } ^ { ( 0 ) } ( x ) : = \frac { 1 } { N _ { g } ( s , a , x ) } \sum _ { ( i , t ) : s _ { i , t } = s , \ a _ { i , t } = a , \atop x _ { i , t } = x } \beta ^ { T _ { i } - t } R _ { i } .\tag{20}
$$

Set this quantity to zero for unobserved successors, whose transition probabilities are zero. Factoring $\beta ^ { T _ { i } - t + 1 } \dot { R } _ { i } = \dot { \beta } ( \beta ^ { T _ { i } - t } R _ { i } )$ and grouping occurrences by successor gives

$$
Q _ { g } ^ { ( 0 ) } ( s , a ) = \beta \sum _ { x \in \mathcal { X } _ { g } } P _ { g } ( x \mid s , a ) V _ { g  ( s , a ) } ^ { ( 0 ) } ( x ) .
$$

Subtracting this from the action backup yields

$$
\delta _ { g } ^ { ( 0 ) } ( s , a ) = \beta \sum _ { x \in \mathcal { X } _ { g } } P _ { g } ( x \mid s , a ) \Big [ V _ { g } ^ { ( 0 ) } ( x ) - V _ { g  ( s , a ) } ^ { ( 0 ) } ( x ) \Big ] .\tag{21}
$$

Boundary contributions vanish. For $\begin{array} { r } { x \in \mathcal S _ { g } , V _ { g } ^ { ( 0 ) } ( x ) = N _ { g } ( x ) ^ { - 1 } \sum _ { ( i , t ) \in \mathcal { T } _ { q } ( x ) } G _ { i , t } } \end{array}$ averages all occurrences at $x ,$ whereas the incoming average includes only those reached from $( s , a )$ . Thus the residual vanishes if every observed non-terminal successor has no initial occurrences and receives transitions only from $( s , a )$ , or if the weighted discrepancies cancel. Acyclicity alone does not suffice, since different state-action pairs may share a successor.

Transport. Since $( I - \beta M _ { g } ) ( V _ { g } - V _ { g } ^ { ( 0 ) } ) = \bar { \delta } _ { g } ^ { ( 0 ) }$ , set $d _ { g } : = ( I - \beta M _ { g } ) ^ { - 1 } \bar { \delta } _ { g } ^ { ( 0 ) } = V _ { g } - V _ { g } ^ { ( 0 ) }$ . The affine Bellman recurrence gives, by induction,

$$
V _ { g } - V _ { g } ^ { ( K ) } = ( \beta M _ { g } ) ^ { K } d _ { g } , \qquad K \geq 0 .\tag{22}
$$

For $K \geq 1$ , equation 17 then gives

$$
Q _ { g } - Q _ { g } ^ { ( K ) } = \beta P _ { g } ^ { S } ( \beta M _ { g } ) ^ { K - 1 } d _ { g } ,\tag{23}
$$

$$
A _ { g } ^ { ( \infty ) } - A _ { g } ^ { ( K ) } = \Big [ \beta P _ { g } ^ { S } ( \beta M _ { g } ) ^ { K - 1 } - L ( \beta M _ { g } ) ^ { K } \Big ] d _ { g } .
$$

At $K = 0$ , adding and subtracting $B _ { g } V _ { g } ^ { ( 0 ) }$ instead yields

$$
\begin{array} { r l } & { \quad Q _ { g } - Q _ { g } ^ { ( 0 ) } = \delta _ { g } ^ { ( 0 ) } + \beta P _ { g } ^ { S } d _ { g } , } \\ & { \quad A _ { g } ^ { ( \infty ) } - A _ { g } ^ { ( 0 ) } = \delta _ { g } ^ { ( 0 ) } + \big ( \beta P _ { g } ^ { S } - L \big ) d _ { g } . } \end{array}\tag{24}
$$

Consequently, the $K = 0$ deviation cannot be controlled solely by $\| \bar { \delta } _ { g } ^ { ( 0 ) } \| _ { \infty } .$ when $\bar { \delta } _ { g } ^ { ( 0 ) } \equiv 0 $ , we have $d _ { g } = 0$ , but $A _ { g } ^ { ( \infty ) } - A _ { g } ^ { ( 0 ) } = \delta _ { g } ^ { ( 0 ) }$ may remain nonzero. Considering the sequences from $K = 0$ onward, the state-value sequence is constant if and only if $\bar { \delta } _ { g } ^ { ( 0 ) } \equiv 0$ , whereas the paired sequence $( V _ { g } ^ { ( K ) } , Q _ { g } ^ { ( K ) } )$ is constant if and only if $\delta _ { g } ^ { ( 0 ) } \equiv 0$

## B.4 PROOF OF COROLLARY 3

Depth bound. For $K \geq 1$ , non-expansiveness of the operators in equation 23 and the bound on $( I - \beta M _ { g } ) ^ { - 1 }$ give

$$
\bigl \| \boldsymbol A _ { g } ^ { ( \infty ) } - \boldsymbol A _ { g } ^ { ( K ) } \bigr \| _ { \infty } \leq 2 \beta ^ { K } \| \boldsymbol d _ { g } \| _ { \infty } \leq \frac { 2 \beta ^ { K } } { 1 - \beta } \| \bar { \delta } _ { g } ^ { ( 0 ) } \| _ { \infty } = : E _ { K } ,
$$

establishing equation 12. The restriction $K \geq 1$ follows from the additional action-level residual in equation 24.

Alignment and value improvement. For a credit $\tilde { A }$ centred under $\mu _ { g } .$ , write $e : = A _ { g } ^ { ( \infty ) } - \tilde { A }$ and use the weighted norm $\begin{array} { r } { \Vert e ( s , \cdot ) \Vert _ { \mu _ { g } } : = \left( \sum _ { a } \mu _ { g } ( a \ | \ s ) e ( s , a ) ^ { 2 } \right) ^ { 1 / 2 } } \end{array}$ . Cauchy–Schwarz gives, anchor-wise,

$$
\big \langle \tilde { A } , A _ { g } ^ { ( \infty ) } \big \rangle _ { \mu _ { g } } = \sigma _ { g } ^ { 2 } - \big \langle e , A _ { g } ^ { ( \infty ) } \big \rangle _ { \mu _ { g } } \ge \sigma _ { g } \big ( \sigma _ { g } - \| e \| _ { \mu _ { g } } \big ) .\tag{25}
$$

Taking $\tilde { A } = A _ { g } ^ { ( K ) }$ , which is centred under $\mu _ { g } ,$ , gives $\| e ( s , \cdot ) \| _ { \mu _ { g } } \leq \| e \| _ { \infty } \leq E _ { K }$ . Hence the inner product is non-negative whenever $E _ { K } \le \sigma _ { g } ( s )$ . If $\sigma _ { g } ( s ) = 0$ , then $A _ { g } ^ { ( \infty ) } ( s , \cdot ) = 0$ on the observed support, so the inner product is zero regardless of $E _ { K } ^ { - } . \mathrm { A t }$ anchors with $w ( s ) = 0$ , the forcing term in equation 11 is also zero. Thus, if $E _ { K } \le \sigma _ { g } ( s )$ at every anchor with $w ( s ) > 0$ and $\sigma _ { g } ( s ) > 0 .$ the forcing is componentwise non-negative. For any feasible reweighting from Proposition 2, the non-negative resolvent in equation 11 then yields $V _ { g } ^ { \bar { \nu } _ { \alpha } } \succeq V _ { g }$

Existence of a finite depth. Let $S _ { g , + } ( w ) : = \{ s \in S _ { g } : w ( s ) > 0 , \sigma _ { g } ( s ) > 0 \}$ . If this set is nonempty, its finiteness gives min ${ \cdot } s \in  S _ { g , + } ( w ) \sigma _ { g } ( s ) > 0$ . Since $E _ { K } \to 0$ , the sufficient condition holds for all sufficiently large finite K. If the set is empty, all forcing terms are zero and the condition holds vacuously. Failure to satisfy this sufficient condition does not imply misalignment. The decay of $E _ { K }$ improves the certified lower bound $\sigma _ { g } ( \sigma _ { g } - E _ { K } )$ , but does not imply monotone alignment, preserved action rankings, or a particular training-curve shape. □

Table 5: Environment-level training settings used in our reproduced experiments. Settings are shared across methods within each model scale and environment.
<table><tr><td>Setting</td><td>ALFWorld</td><td>WebShop</td><td>Sokoban</td></tr><tr><td>Maximum environment steps</td><td>50</td><td>15</td><td>15</td></tr><tr><td>Maximum prompt tokens</td><td>2048</td><td>5120</td><td>1024</td></tr><tr><td>Maximum response tokens</td><td>512</td><td>512</td><td>512</td></tr><tr><td>Training groups per update</td><td>16</td><td>16</td><td>32</td></tr><tr><td>Rollouts per group</td><td>8</td><td>8</td><td>8</td></tr><tr><td>Validation trajectories</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Base model</td><td>Qwen2.5-1.5B/7B</td><td>Qwen2.5-1.5B/7B (Qwen et al., 2025)</td><td>Qwen2.5-VL-3B (Bai et al., 2025)</td></tr></table>

## C EXPERIMENT DETAILS

## C.1 COMPARING METHODS

We compare CRBC with the following baselines.

• GPT-4o: a closed-source, general-purpose LLM used as a reference for multi-turn agent capability (OpenAI, 2023).

• Gemini-2.5-Pro: a second closed-source LLM reference point with strong general-purpose reasoning ability (Team, 2023).

• ReAct: an in-context prompting agent that interleaves reasoning and acting (Yao et al., 2023).

• Reflexion: a prompting agent that uses verbal reflection and iterative self-improvement without parameter updates (Shinn et al., 2023).

• PPO: the standard clipped policy-gradient method with a learned value critic and GAE (Schulman et al., 2017).

• RLOO: a critic-free, trajectory-level leave-one-out baseline (Ahmadian et al., 2024).

• GRPO: a critic-free group objective that normalizes one outcome advantage over the rollouts of a task (Shao et al., 2024).

• GiGPO: a step-level method that compares occurrences at repeated anchor states (Feng et al., 2025).

• GraphGPO: a graph-based method that scores transitions by a shortest-path/extremal readout (Cheng et al., 2026).

• HGPO: a hierarchical grouping method that refines step comparisons using historicalcontext consistency (He et al., 2026).

• CRBC (ours): the predictor-free, full-closure setting described in Appendix C.3.

## C.2 ENVIRONMENT DETAILS

ALFWorld. ALFWorld (Shridhar et al., 2021) is a text-based embodied environment in which an agent must complete a household task through a long-horizon sequence of admissible commands. We use the six task families reported by the environment evaluator: Pick, Clean, Cool, Look, Heat, and Pick2. The training wrapper is AlfredTWEnv; it augments the textual observation with the current location, holding/item status, item-transition history, and the sorted admissible-command list. Success and failure are terminal outcomes, and a trajectory is truncated after at most 50 environment steps. Each validation pass evaluates 128 in-distribution trajectories selected by the environment seed.

WebShop. WebShop (Yao et al., 2022) is a text-based interactive shopping environment in which the agent searches, navigates, and selects an item that matches a natural-language request. We use the repository’s text-rich, small-catalogue configuration (observation mode=text rich and use small=True, using the repository’s items shuffle 1000.json/items ins v2 1000.json files) with two retained interaction records. The WebAgentTextEnv wrapper limits a trajectory to 15 environment steps; the validation goals use indices 0–499 and training starts at index 500 in the launcher. We report both the raw task score and the binary task success rate.

![](images/4a5dd40017cc49aac8c55767adb5d6a74094b15aaff157e4cfbe90ddb00eae94.jpg)

![](images/754d3aa7cf798d11855d78c1d59383ae04aa650207989665afa3f9dd7ca3e559.jpg)  
Figure 5: Cross-rollout evidence coverage measured on rollouts from an early-training checkpoint on ALFWorld. Left: distribution of the number of reachable rollouts at each closure depth K. Right: paired coverage at K = 0 and full closure, with color indicating the percentage of all eligible visits.

Sokoban. Sokoban (Schrader, 2018) tests transfer to a visual interactive setting. We use the $6 \times 6$ one-box configuration with Qwen2.5-VL-3B-Instruct (Bai et al., 2025); the checkpoint is pinned to the revision used by the releazed launcher. The environment returns an RGB observation and the agent chooses among the four box-pushing directions. Trajectories are limited to 15 environment steps. We report the final raw task score (“text score”) and the percentage of solved boards (“success rate”).

Anchor construction. Anchors are matched by exact equality of their environment-specific rep resentations within each task group. ALFWorld uses the current textual observation augmented with location, held-item status, item-transition records, and sorted admissible commands; WebShop uses the formatted current-page observation; Sokoban uses the current RGB observation. The recent interaction records appended to policy prompts are excluded from the matching key.

## C.3 DETAILS OF TRAINING

Shared protocol. For each update, text experiments sample 16 task groups with $G = 8$ rollouts per group, giving 128 training rollouts. Sokoban uses 32 groups with the same group size. All settings use 128 validation trajectories, validate before training and every five updates, and train for 150 updates. The rollout temperature is 1.0 and the validation temperature is 0.4 with sampling enabled. The actor learning rate is $1 0 ^ { - 6 }$ , the reference-policy KL-loss coefficient is 0.01 (the low-variance KL form), and gradient checkpointing is enabled. For ALFWorld and WebShop, a successful trajectory receives reward 10 and failure receives 0. An invalid-action penalty of 0.1 is applied before advantage computation. PPO alone uses a learned critic with learning rate $\mathrm { i 0 ^ { - 5 } } ;$ ; group-based methods do not train an auxiliary critic. Environment and data-loader seeds are set to the same value for each run, and means and standard deviations are computed only over completed independent seeds.

Resource and batching settings. The batching choices are adjusted only to fit the model and environment; within each comparison they are identical across methods. For ALFWorld, the 1.5B and 7B runs use tensor parallel sizes 1 and 4, actor micro-batches 32 and 4 per GPU, and PPO mini-batches 256 and 64, respectively. For WebShop, the corresponding tensor parallel sizes are 2 and 4, actor micro-batches are 8 and 4, and the PPO mini-batch is 64 for both scales. Sokoban uses tensor parallel size 2, actor micro-batch 8, and PPO mini-batch 64. The rollout and reference log-probability micro-batches are 32 (ALFWorld 1.5B), 16 (ALFWorld 7B and WebShop), and 16 (Sokoban). All reproduced experiments are conducted on a single node equipped with eight NVIDIA A100 GPUs (80 GB each); tensor parallelism and micro-batches are adjusted within this fixed budget as specified above.

Table 6: Bellman-discount ablation on ALFWorld with Qwen2.5-1.5B-Instruct. Only $\beta$ varies; all other CRBC settings are fixed. Results are averaged over three random seeds; best values are in bold.
<table><tr><td>Bellman discount β</td><td>Pick</td><td>Clean</td><td>Cool</td><td>Look</td><td>Heat</td><td> $\mathrm { P i c k } 2$ </td><td>All</td></tr><tr><td>0.95</td><td> $9 8 . 8 9 _ { ( 1 . 5 7 ) }$ </td><td> $9 8 . 2 5 _ { ( 2 . 4 8 ) }$ </td><td> ${ \bf 9 4 . 8 2 } _ { ( 1 . 7 8 ) }$ </td><td> $9 1 . 6 7 _ { ( 0 . 0 0 ) }$ </td><td> $8 8 . 8 9 _ { ( 2 . 2 4 ) }$ </td><td> $9 8 . 3 3 _ { ( 2 . 3 6 ) }$ </td><td> $9 5 . 5 7 _ { ( 0 . 3 7 ) }$ </td></tr><tr><td>0.98</td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $9 2 . 9 8 _ { ( 6 . 5 6 ) }$ </td><td> $9 2 . 1 5 _ { ( 3 . 3 3 ) }$ </td><td> $9 1 . 6 7 _ { ( 0 . 0 0 ) }$ </td><td> $\mathbf { 9 6 . 8 3 } _ { ( 4 . 4 9 ) }$ </td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $9 6 . 0 9 _ { ( 0 . 6 4 ) }$ </td></tr><tr><td>0.99</td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $8 6 . 8 4 _ { ( 2 . 6 3 ) }$ </td><td> $9 4 . 2 3 _ { ( 5 . 7 7 ) }$ </td><td> ${ \bf 9 5 . 8 3 } _ { ( 4 . 1 7 ) }$ </td><td> $9 2 . 8 6 _ { ( 2 . 3 8 ) }$ </td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $9 5 . 3 1 _ { ( 1 . 5 6 ) }$ </td></tr><tr><td>1.00</td><td> $\mathbf { 1 0 0 . 0 0 } _ { ( 0 . 0 0 ) }$ </td><td> $8 9 . 4 7 _ { ( 1 0 . 5 3 ) }$ </td><td> $9 2 . 2 3 _ { ( 3 . 7 7 ) }$ </td><td> $8 3 . 3 3 _ { ( 8 . 3 3 ) }$ </td><td> $9 5 . 2 4 _ { ( 4 . 7 6 ) }$ </td><td> $9 2 . 5 0 _ { ( 2 . 5 0 ) }$ </td><td> $9 3 . 3 6 _ { ( 0 . 3 9 ) }$ </td></tr></table>

Table 7: Rollout group size comparison on ALFWorld with Qwen2.5-1.5B-Instruct at step 150. We report validation success rates (%) for six subtasks and overall (All), together with task score. Best reported values within each group size are in bold.
<table><tr><td>Group size G Method</td><td></td><td>Pick</td><td>Clean</td><td>Cool</td><td>Look</td><td>Heat</td><td>Pick2</td><td>All</td><td>Score</td></tr><tr><td>4</td><td>GRPO</td><td>60.00</td><td>36.84</td><td>46.15</td><td>50.00</td><td>38.10</td><td>50.00</td><td>47.66</td><td>2.27</td></tr><tr><td>4</td><td>GiGPO</td><td>90.00</td><td>63.16</td><td>88.46</td><td>66.67</td><td>80.95</td><td>75.00</td><td>79.69</td><td>5.34</td></tr><tr><td>4</td><td>GraphGPO</td><td>86.67</td><td>68.42</td><td>84.62</td><td>58.33</td><td>76.19</td><td>75.00</td><td>77.34</td><td>4.38</td></tr><tr><td>4</td><td>CRBC (Ours)</td><td>100.00</td><td>94.74</td><td>88.46</td><td>91.67</td><td>100.00</td><td>80.00</td><td>92.97</td><td>7.12</td></tr><tr><td>16</td><td>GRPO</td><td>93.33</td><td>73.68</td><td>80.77</td><td>58.33</td><td>85.71</td><td>80.00</td><td>81.25</td><td>5.08</td></tr><tr><td>16</td><td>GiGPO</td><td>100.00</td><td>89.47</td><td>88.46</td><td>83.33</td><td>100.00</td><td>90.00</td><td>92.97</td><td>7.01</td></tr><tr><td>16</td><td>GraphGPO</td><td>100.00</td><td>78.95</td><td>88.46</td><td>66.67</td><td>90.48</td><td>85.00</td><td>87.50</td><td>5.30</td></tr><tr><td>16</td><td>CRBC (Ours)</td><td>100.00</td><td>94.74</td><td>96.15</td><td>100.00</td><td>90.48</td><td>100.00</td><td>96.88</td><td>8.34</td></tr></table>

Method-specific settings. GraphGPO uses its default shortest-path readout with graph discounts 0.10, 0.20, and 0.80 on ALFWorld, WebShop, and Sokoban, respectively, and step/trajectory weights $1 / 1$ GiGPO uses a return discount of 0.95 and step/trajectory weights $1 / 1$ (the final Sokoban launcher uses its mean norm variant). GRPO uses the standard group-normalized trajectory ad vantage, while PPO and RLOO use GAE and a trajectory-level leave-one-out baseline, respectively. Unless otherwise stated, CRBC uses the full-closure configuration with $K = \infty , \beta = 0 . 9 8 , w _ { \mathrm { G } } = 1$ $w _ { \mathrm { S } } = 5 ,$ , and a normalization floor $\sigma _ { \mathrm { m i n } } = 0 . 1$ . The empirical process is formed from the observed rollout transitions, and the resulting $Q - V$ signal is normalized per anchor before being combined with the trajectory-level advantage. All other training settings follow the common protocol above.

## D ADDITIONAL EXPERIMENTS

## D.1 BELLMAN DISCOUNT

Table 6 evaluates the Bellman discount $\beta$ on ALFWorld with Qwen2.5-1.5B-Instruct, keeping all other CRBC settings fixed. All results are averaged over three random seeds, with the default $\beta =$ 0.98 using the main CRBC result. Across the tested discounted settings $\beta \in \{ 0 . 9 5 , 0 . 9 8 , 0 . 9 9 \}$ overall success remains between 95.31% and 96.09%, with $\beta = 0 . 9 8$ achieving the highest observed mean. Removing discounting $( \beta = 1 . 0 0 )$ reduces overall success to 93.36%, a decrease of 2.73 percentage points relative to the default.

Subtask preferences vary: $\beta = 0 . 9 5$ performs best on Clean and Cool, $\beta = 0 . 9 9$ on Look, and $\beta = 0 . 9 8$ on Heat. All configurations use full closure, so $\beta$ controls how strongly distant evidence is attenuated along the merged graph. These results suggest that mild discounting is beneficial in this setting and support $\beta = 0 . 9 8$ as a practical default, with limited sensitivity among the tested discounted values.

## D.2 CLOSURE DEPTH AND EVIDENCE COVERAGE

Figure 5 examines cross-rollout evidence coverage using rollouts from an early-training checkpoint on ALFWorld. We consider visits to anchors with at least two distinct observed actions. For each eligible visit, coverage counts the distinct rollouts whose evidence is structurally accessible at depth K within its eight-rollout group, without weighting their contributions. The left panel shows the coverage distribution: increasing K allows evidence from shared successors to propagate back through the merged graph, shifting visits toward broader coverage. The proportion of eligible visits with access to all eight rollouts increases from 12.9% at K = 0 to 61.5% under full closure.

![](images/907e9e85328c658a59aac2b6f8def543004a9fc99efd500fbb8597afd5303999.jpg)

![](images/603f2180fc2a4c90225b4919673ff874f9b39d4a95574e228d09d7379dd92cdc.jpg)

![](images/9d8ee61ec6c6c52be7a78093099f0d9d12e260f3ac54580eb2f7313084fea820.jpg)  
Figure 6: Validation success-rate curves for CRBC (Ours, red), GraphGPO (purple), GiGPO (green), and GRPO (gray) on ALFWorld, WebShop, and Sokoban. Light curves show validation measurements recorded every five training updates, while dark curves show EMA-smoothed trends (α = 0.95 per update). Qwen2.5-1.5B-Instruct is used for ALFWorld and WebShop, and Qwen2.5- VL-3B-Instruct for Sokoban.

![](images/3f4431ffa12f9d1d4ab9fe6229b75d4340ab035cc3c3ffde7f600ab8cab4e46b.jpg)

![](images/eaa97b090ead33c07bdeebb0bc67f14f4b0ab77d93626932a428aa1a755c452d.jpg)

![](images/bcddaf4a26b3ea9c33eae03424501ff7d1b4ee8e643941da3ff0bb4a8289300e.jpg)  
V(0) (K = 0)  V(1) (K = 1)  V(2) (K = 2)  ν(3) (K = 3)  V(∞) (K = ∞, CRBC)  
Figure 7: Training dynamics for the closure-depth ablation on ALFWorld with Qwen2.5-1.5B-Instruct. Only the closure depth K varies; all other settings are fixed. From left to right, we show episode reward, validation success rate, and validation score. Light and dark curves denote raw and EMA-smoothed trajectories (α = 0.95); deeper closure gives stronger late-training performance.

The right panel pairs coverage before and after closure for the same visits, with rows representing $K = 0$ , columns representing full closure, and color intensity indicating the percentage of all eligible visits. Diagonal cells indicate unchanged coverage; cells with larger column values indicate expansion. Overall, coverage increases for 63.0% of eligible visits, including those whose expanded coverage remains below eight rollouts. This structural view complements the final performance comparison in Table 2 and the corresponding training curves in Figure 7, illustrating how closure makes additional cross-rollout evidence available.

## D.3 VALIDATION SUCCESS

Figure 6 reports validation success rates on ALFWorld, WebShop, and Sokoban, measured every five training updates. CRBC generally reaches high success rates earlier and achieves stronger final performance than the compared baselines. These trends complement the training reward curves in Figure 3, showing that the observed learning-efficiency gains are also reflected in task success on the validation sets.

## D.4 ROLLOUT GROUP SIZE

Table 7 compares group-based RL methods with rollout group sizes $G \in \{ 4 , 1 6 \}$ on ALFWorld using Qwen2.5-1.5B-Instruct. At G = 4, CRBC achieves 92.97% success, compared with 47.66% for GRPO, 79.69% for GiGPO, and 77.34% for GraphGPO. At $G = 1 6 ,$ , CRBC reaches 96.88%, versus 92.97% for GiGPO and 87.50% for GraphGPO. CRBC therefore improves over the strongest baseline by 13.28 and 3.91 percentage points, respectively, while also obtaining the highest task scores of 7.12 and 8.34. These comparisons show that CRBC retains an advantage at both tested group sizes.

## E PROMPT TEMPLATES

Figure 8 shows the prompt templates for ALFWorld, WebShop, and Sokoban. For the text environments, prompts retain at most the two most recent observation–action records. For Sokoban, the prompt contains the current RGB image, task rules, and four movement directions, without textual interaction history. In all three environments, the agent is instructed to place its reasoning inside <think>...</think> and its selected action inside <action>...</action>. Within each environment, all reproduced RL methods share the same prompt templates and memory protocol.

![](images/070b1e043605c474eee9cfc0f84e4f25e268b0855ad1bbb537275602b9907147.jpg)  
(c) Sokoban prompt  
Figure 8: Prompt templates for ALFWorld, WebShop, and Sokoban. Coloured placeholders denote runtime inputs. Text environments retain up to two observation–action records; Sokoban uses the current RGB observation without textual interaction history. Panel (c) includes an example RGB input.