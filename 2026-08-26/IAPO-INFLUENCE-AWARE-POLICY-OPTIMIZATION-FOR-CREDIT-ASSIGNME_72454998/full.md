# IAPO: INFLUENCE-AWARE POLICY OPTIMIZATION FOR CREDIT ASSIGNMENT IN MULTI-TURN SERVICE AGENTS

A PREPRINT

Bo Ren<sup>1,2,†</sup>, Yirong Mao<sup>2,†</sup>, Yi Yang<sup>2</sup>, Wenhui Que<sup>2,∗</sup> <sup>1</sup>Fudan University <sup>2</sup>WeChat, Tencent Inc.

August 26, 2026

## ABSTRACT

Large Language Model (LLM) agents increasingly solve long-horizon tasks through multi-turn interactions with users and external tools. In these settings, relevant task information often unfolds over time rather than being fully specified at the initial prompt. Service agents make this challenge especially concrete: users may clarify or revise their goals, while tool responses provide information needed for subsequent decisions. Thus, a final reward alone cannot indicate which actions contributed to resolving the task. Recent methods rely on comparative evidence from other trajectories or resampled continuations, or on separately constructed step-level learning signals, to refine credit. However, a completed rollout already records how information and errors flow between agent actions. We introduce Influence-Aware Policy Optimization (IAPO), which represents each rollout as a typed influence-dependency graph over trainable agent actions, with user and tool observations serving as evidence. IAPO converts support-use and failed-use structure into routing weights that redistribute the same trajectory-level advantage. Experiments with Qwen3-4B and Qwen3- 8B demonstrate superior performance over multi-turn reinforcement learning (RL) baselines across three service-agent benchmarks: τ<sup>2</sup>-Bench, UserBench, and AgentChangeBench. BFCL-v4 Multi-Turn further shows that these gains do not compromise multi-turn function-calling performance. This work advances the understanding of credit assignment in multi-turn user interactions and provides a principled approach to training service agents from sparse outcome feedback.

## Introduction

Large Language Models (LLMs) (OpenAI et al., 2024; Team et al., 2023; Yang et al., 2025; DeepSeek-AI et al., 2025) have moved from single-turn responders to interactive agents that plan, call tools, and act over multi-turn interactions with users and environments (Yao et al., 2022b; Schick et al., 2023; Liu et al., 2024). In particular, multi-turn service agents often solve tasks whose relevant information is not available at the initial prompt. Instead, users may reveal preferences or revise goals over successive turns, while tool calls expose operational states that determine which actions are valid or useful. For example, in an after-sales service task (Figure 1), before issuing the refund, the agent may need to clarify whether the user wants the money returned to the original card or converted into store balance, and then retrieve the order record needed to execute that choice. Thus, the contribution of an early action may become identifiable only after a later user clarification or tool response reveals how its information is used. We refer to this setting as evolving-state credit assignment: task goals, constraints, and operational states are progressively instantiated through interaction rather than fully specified at the outset.

Group-based RL algorithms such as GRPO (Shao et al., 2024) and its variants (Yu et al., 2026; Zheng et al., 2025) apply the same group-relative advantage to every trainable token of a sampled rollout. However, this trajectory-level signal does not distinguish how different intermediate actions contribute to the final outcome (Figure 1, GRPO columns). This limitation motivates finer-grained credit assignment for evolving-state interactions.

![](images/12e98fc9205150118ac37f87cce4b4971bd1f7625c726e840d6e5ee2acf0df45.jpg)  
Figure 1: Per-action advantage allocation for a successful (left, positive-advantage) and a failed (right, negativeadvantage) refund rollout. Refunding to the original card or store balance is correct only if it matches the user’s clarified intent, so an early clarifying turn decides whether later actions are right. GRPO applies the same advantage to every action, whereas IAPO routes credit according to realized support use and error propagation.

Existing approaches obtain finer-grained credit by supplementing the final outcome with additional evidence. One line compares recurring or matched states and resampled continuations across sampled trajectories (Feng et al., 2026; Ji et al., 2025; Samanta et al., 2026). Another line constructs separate step-level signals through additional evaluation, using masked-context policy scoring, a teacher model, or a learned critic on training-time information (Kong et al., 2026; Xie et al., 2026; Zhou et al., 2025; Tan et al., 2026).

However, obtaining this finer credit adds a requirement that the final outcome does not. 1) Cross-rollout methods need comparable structure across sampled trajectories. GiGPO (Feng et al., 2026) relies on environment states that recur across rollouts, and branch- or reset-based methods (Ji et al., 2025; Samanta et al., 2026) rely on a forkable or resettable environment with additional continuation samples. Free-form user and tool dialogue rarely provides either, because distinct user turns and tool returns leave few comparable state groups and a live user cannot be faithfully reset or branched. 2) Separately-scored methods add a scoring channel outside the reward, such as masked-context policy scoring (Kong et al., 2026), a teacher model (Xie et al., 2026), or a learned critic on privileged information (Zhou et al., 2025). The resulting step signal is computed apart from task completion and can drift from it. No existing method redistributes outcome credit using only the dependency structure already realized within a single completed interaction.

In this work, we observe that a completed user–tool rollout already records more structure than its outcome reward exposes: a later action may consume information elicited or supplied by an earlier one, or reuse an earlier invalid output and fall back because of an observable error. Therefore, we propose IAPO, which represents each completed rollout as a typed influence-dependency graph over trainable agent actions and routes the original trajectory-level advantage according to observed support-use and failed-use dependencies, using user and tool observations as evidence without assigning them policy-gradient credit. Since this dependency structure is already realized in the transcript and tool log, it can redistribute the same trajectory-level advantage without changing the reward, rollout process, group normalization, or clipped-loss implementation. IAPO keeps the critic-free, group-based update intact while introducing finer, action-level credit. The change lies in which action tokens carry the feedback, not in the value of the final task outcome.

## Contributions.

• We identify realized downstream information use as a credit object for evolving-state service interactions.

• We introduce IAPO, which extracts typed within-rollout dependencies and routes the trajectory-level advantage through support-use and failed-use structure.

• We prove mass conservation, sign preservation, and uniform fallback, and show that IAPO leads on the three service-agent benchmarks across two Qwen3 scales and maintains competitive out-of-distribution retention on BFCL-v4 Multi-Turn.

IAPO: Influence-Aware Policy Optimization  
![](images/e1b827314ed332686404d97e5dfaf8663d4a3674517d9d509b4d1aa696023259.jpg)

![](images/07d60ed1709d91059c59a2ab98fbdda7bfe47e911145eecb95d219521996c67a.jpg)

![](images/a606355a447663e8c1a4f5a0ee1897ab5f489079309876a5b019c2251b15a543.jpg)  
Figure 2: Overview of IAPO. (A) Multiple rollouts are sampled per task and scored by the outcome reward. (B) A frozen annotator extracts signed influence edges among trainable actions, using user and tool observations as evidence. (C) The resulting graph features are normalized into bounded per-action weights that replace GRPO’s uniform advantage with sign-conditioned routing.

## Related Work

Interactive benchmarks for user-interacting agents. Task-oriented dialogue has long been treated as sequential decision making (Budzianowski et al., 2018; Ultes et al., 2017). Interactive benchmarks have progressively incorporated simulated users, stateful tools, and multi-turn feedback (Yao et al., 2024; Barres et al., 2025; Qian et al., 2025a; Wang et al., 2023). Beyond dialogue, agent benchmarks cover web navigation, embodied environments, and APIbased tool use (Yao et al., 2022a; Shridhar et al., 2021; Li et al., 2023; Qin et al., 2023; Zhou et al., 2024a; Deng et al., 2023; Liu et al., 2024; Mialon et al., 2024). More recent testbeds further expose incremental preference acquisition and explicit goal shifts across interaction episodes (Qian et al., 2025a; Rana et al., 2025). These settings expose evolving user goals and tool-mediated state, but leave open how a final outcome should be attributed to the actions that gathered, used, or corrupted intermediate evidence.

RL for LLMs and credit assignment. Early LLM agents used frozen models with prompting and tool use, such as ReAct (Yao et al., 2022b) and Toolformer (Schick et al., 2023); more recent work optimizes agents directly with reinforcement learning, building on PPO (Schulman et al., 2017), GRPO (Shao et al., 2024), and their variants (Zheng et al., 2025; Yu et al., 2026; Liu et al., 2025b). Since outcome-only rewards leave intermediate steps unattributed, a growing body of work assigns credit more finely, from classical return decomposition and process supervision (Arjona-Medina et al., 2019; Harutyunyan et al., 2019; Lightman et al., 2024; Cui et al., 2025; Ng, Harada, and Russell, 1999) to recent agent-oriented methods. On reasoning and agent tasks, GiGPO (Feng et al., 2026) groups recurring states across rollouts, tree- or reset-based methods search over branches (Ji et al., 2025; Samanta et al., 2026), hierarchical and trajectory-level RL frameworks restructure the update itself (Zhou et al., 2024b; Wang et al., 2025), and Fission-GRPO (Zhang et al., 2026b) synthesizes new recovery rollouts from failed trajectories. Denser step rewards instead come from process or hindsight evaluators (Zhang et al., 2026a; Tan et al., 2026), turn-level shaping (Xie et al., 2026; Wei et al., 2025; Ding et al., 2026; Parthasarathi et al., 2025), or token-level reshaping (Li et al., 2026; Mesnard et al., 2021; Lai et al., 2024). A separate line places the user inside the RL loop: MUA-RL (Zhao et al., 2025) first trains agents against a simulated user interacting with live tools, and later methods add finer learning signals on top, such as UserRL (Qian et al., 2025b) shaping turn- and trajectory-level rewards, SWEET-RL (Zhou et al., 2025) training a step-level critic from training-time information, and InfoPO (Kong et al., 2026) adding an information-gain reward from masked-context counterfactuals. All of these add signals beyond the sampled rollout: shaped rewards, a trained critic, an auxiliary intrinsic objective, or synthesized recovery rollouts. IAPO instead uses the dependencies already realized among the agent’s own actions. It reweights the tokens of a rollout the baseline already sampled, using only the influence links realized inside that rollout.

## Problem Setup

We consider a multi-turn service episode in which user messages and tool returns progressively expose task state. Policy $\pi _ { \theta }$ controls assistant replies and tool calls; user messages, tool returns, and rule feedback are observations used for provenance but receive no policy-gradient credit.

A trajectory alternates between observations and assistant actions:

$$
\tau = ( o _ { 1 } , a _ { 1 } , o _ { 2 } , a _ { 2 } , \ldots , o _ { T } , a _ { T } ) ,\tag{1}
$$

where $o _ { t }$ is an observation, $a _ { t }$ is an assistant action, $L _ { i }$ is the token length of action $i ,$ and $L = \textstyle \sum _ { i } L _ { i }$ . The environment returns terminal reward $R _ { \tau }$ . GRPO samples $\mathcal { G } = \{ \tau _ { k } \} _ { k = 1 } ^ { K }$ for the same task and computes

$$
\hat { A } _ { \tau } ~ = ~ \frac { R _ { \tau } - \mu _ { \mathcal G } } { \sigma _ { \mathcal G } + \varepsilon } ,\tag{2}
$$

then broadcasts ${ \hat { A } } .$ <sub>τ</sub> to every action token.

Evolving-state credit assignment. IAPO redistributes this scalar using only dependency evidence in the completed rollout: $A _ { i , \ell } = \hat { A } _ { \tau } w _ { i }$ , with $w _ { i } > 0 , \sum _ { i } L _ { i } w _ { i } = L$ , and $w _ { i } = 1$ for an uninformative graph. The annotator extracts dependency labels but neither scores success nor creates an additional reward.

## Method

IAPO scores rollouts as GRPO does, extracts a signed influence graph from each completed rollout, and routes the trajectory advantage to each token as $A _ { i , \ell } = \hat { A } _ { \tau } w _ { i }$ (Figure 2). The routing weight $\mathbf { w } = ( w _ { 1 } , \hdots , w _ { T } )$ redistributes the advantage across actions according to their influence: toward information that is used downstream, and toward the observable errors behind a negative outcome. w is positive, bounded, and length-normalized, so routing reshapes how the advantage is shared without changing its total or its sign. The rest of this section builds w to this shape.

## Influence-Dependency Graph

The map w is computed from observable evidence in the completed rollout. For $\tau ,$ , let $V _ { \tau } = \{ a _ { 1 } , . . . , a _ { T } \}$ contain the trainable assistant actions. IAPO builds the influence-dependency graph $G _ { \tau } = ( V _ { \tau } , E _ { \tau } ^ { + } , E _ { \tau } ^ { - } )$ , where $E _ { \tau } ^ { + } , E _ { \tau } ^ { - } \subseteq$ $\{ ( i , j ) : 1 \le i < j \le T \}$ and the sign of an edge is independent of the sign of $\hat { A } _ { \tau }$ . We use + for support-use quantities and − for error-evidence quantities throughout this section $( \phi ^ { \pm } , \mathbf { m } ^ { \pm }$ , and $\mathbf { w } ^ { \pm }$ below). An edge $i  ^ { + }$ j is a support-use edge: $a _ { j }$ consumes information supplied or elicited by $a _ { i }$ and the annotation type is input or success\_utilization. An edge $i  ^ { - } j$ is afailed-use edge: $a _ { j }$ repeats an observable error from $a _ { i }$ , uses its invalid output, or must fall back because of it, with annotation type failed\_utilization. The error indicator $e _ { i } \in \{ 0 , 1 \}$ marks whether $a _ { i }$ is error-bearing: a tool, schema, runtime, or rule failure, or an explicit user or environment rejection. A goal revision is not an error unless it rejects the preceding action, so $d _ { i } ^ { - } > 0$ implies $e _ { i } = 1$

An edge $i \to j$ is accepted when three conditions are jointly met:

1. Necessity. $a _ { j }$ references a specific information unit (an entity value, a tool-returned field, or an error trace) that originates from $a _ { i }$ or from an observation elicited by $a _ { i }$ . Sharing the same topic or domain is insufficient: removing the referenced information unit must leave $\boldsymbol { a } _ { j } { \cdot } \mathbf { \boldsymbol { s } }$ argument or decision unsupported.

2. Explicit binding. The referenced information unit is bound to a concrete element in $a _ { j } \colon$ an intent parameter, a tool-call argument, a conditional branch, or an error-recovery action. Abstract thematic similarity does not qualify.

3. Source priority. When multiple prior actions supply overlapping information, the most comprehensive source is kept; ties are broken by earliest occurrence. Multiple sources are cited only when they contribute non-overlapping information units.

User turns and tool returns provide evidence but never receive advantage. Their source type is retained for provenance resolution. A user reply is resolved to the assistant action that elicited it, while a tool return is resolved to the tool call that produced it. If a cited user turn has no assistant elicitor, it contributes only a direct-input count for the consuming action, written $u _ { i } ^ { \mathrm { d i r } }$ . Duplicate user/elicitor evidence is counted once. Writing $d _ { i } ^ { + } = | \{ j : i \stackrel { + } { \to } j \} |$ and $d _ { i } ^ { - } = | \{ j :$ $i \xrightarrow { - } j \} |$ for the number of outgoing support-use and failed-use edges from $a _ { i }$ , and $d _ { i } ^ { \mathrm { i n } } = u _ { i } ^ { \mathrm { d i r } } + | \{ k : k \stackrel { + } {  } i \} |$ for the support evidence $a _ { i }$ consumes, whether from a direct user turn or an incoming support-use edge, these counts feed the graph features

$$
\phi _ { i } ^ { + } = \log ( 1 + d _ { i } ^ { + } + d _ { i } ^ { \mathrm { i n } } ) , \quad \phi _ { i } ^ { - } = \log \bigl ( 1 + e _ { i } ( 1 + d _ { i } ^ { - } + d _ { i } ^ { \mathrm { i n } } ) \bigr ) .\tag{3}
$$

The support score $\phi _ { i } ^ { + }$ grows with the number of downstream consumers and upstream sources of $a _ { i }$ . The error score ${ \phi } _ { i } ^ { - }$ is zero unless $e _ { i } { = } 1 { \mathrm { : } }$ ; when an observable error is present, the inner unit term counts the error itself, $d _ { i } ^ { - }$ counts its downstream failed use, and $d _ { i } ^ { \mathrm { i n } }$ counts the support evidence consumed by the erroneous action.

## From Features to Bounded Weights

Additive token corrections can change the sign or the total of the advantage, so IAPO uses multiplicative weights and turns each feature into a bounded weight around one. For a feature $\phi \in \breve { \mathbb { R } _ { > 0 } ^ { T } }$ , we standardize it by its length-weighted mean and deviation, clip at $c ,$ and renormalize so the length-weighted mean is one:

$$
m _ { i } = \frac { \exp \bigl ( \exp ( z _ { i } , - c , c ) \bigr ) } { \frac { 1 } { L } \sum _ { j } L _ { j } \exp \bigl ( \mathrm { c l i p } ( z _ { j } , - c , c ) \bigr ) } , \qquad z _ { i } = \frac { \phi _ { i } - \bar { \phi } } { \sigma _ { \phi } + \varepsilon } ,\tag{4}
$$

where $\begin{array} { r } { \bar { \phi } = \frac 1 L \sum _ { i } L _ { i } \phi _ { i } , \sigma _ { \phi } ^ { 2 } = \frac 1 L \sum _ { i } L _ { i } ( \phi _ { i } - \bar { \phi } ) ^ { 2 } } \end{array}$ , and $\varepsilon = 1 0 ^ { - 6 }$ . A feature above its mean gets weight above one and below its mean below one, and c caps how far a step can move so no single hub dominates the update. Applying this to the graph features gives the support weight $m _ { i } ^ { + }$ from $\phi ^ { + }$ and the error weight $\boldsymbol m _ { i } ^ { - }$ from $\phi ^ { - }$ . Every token in action i receives the same $w _ { i }$ , so the action carries token mass $L _ { i } w _ { i }$ . The length-aware properties are proved in Appendix B.

## Sign-Conditioned Routing

Positive-advantage rollouts $( \hat { A } _ { \tau } > 0 )$ . A positive-advantage rollout can still contain steps that made errors later steps had to repair, and IAPO gives them less credit than clean steps with the same support role. Starting from the support weight $m _ { i } ^ { + }$ , it discounts a step by its above-average error evidence ${ \psi } _ { i } ^ { - } = { \phi } _ { i } ^ { - } - \bar { \phi } ^ { - }$ , with $\begin{array} { r } { \bar { \phi } ^ { - } = \frac { 1 } { L } \sum _ { i } L _ { i } \phi _ { i } ^ { - } } \end{array}$ and renormalizes:

$$
w _ { i } ^ { + } ( \beta _ { + } ) = \frac { L m _ { i } ^ { + } \exp ( - \beta _ { + } \psi _ { i } ^ { - } ) } { \sum _ { j } L _ { j } m _ { j } ^ { + } \exp ( - \beta _ { + } \psi _ { j } ^ { - } ) } .\tag{5}
$$

This is the closed-form solution of the length-mean-constrained routing problem derived in Appendix B. $\mathrm { A t } ~ \beta _ { + } \mathrm { = } 0$ it is $w _ { i } ^ { + } { = } m _ { i } ^ { + }$ (support only); a larger $\beta _ { + }$ moves credit off repaired-error steps, and the denominator restores the length-weighted mean one.

Negative-advantage rollouts $( \hat { A } _ { \tau } < 0 )$ . A negative-advantage rollout concentrates its advantage on the steps that produced the errors: with many errors and hard attribution, support reuse need not be at fault, so IAPO conservatively routes blame to error evidence only, riding on the error feature directly:

$$
w _ { i } ^ { - } = m _ { i } ^ { - } .\tag{6}
$$

The error score $\phi ^ { - }$ is nonzero only after an observable error, and the normalizer keeps every multiplier positive, so more of the negative advantage reaches the steps with stronger error evidence. A projected support-prior alternative is evaluated in ablation and defined in Appendix B.

Routed advantage. The final weight follows the trajectory’s sign, $w = w ^ { + }$ when $\hat { A } _ { \tau } > 0$ and $\mathbf { \nabla } w = w ^ { - }$ when $\hat { A } _ { \tau } < 0$ , so the token advantage

$$
A _ { i , \ell } = \hat { A } _ { \tau } w _ { i }\tag{7}
$$

is zero when $\hat { A } _ { \tau } { = } 0$ and is passed to the same clipped-loss implementation, replacing the flat per-token advantage used by $\mathrm { G R P O . ~ } \beta _ { + }$ and c are set in the experimental setup, and the projected negative branch is an ablation. After each rollout, a frozen annotator extracts the dependency graph, cached for the corresponding policy update.

<table><tr><td>Model</td><td>Method</td><td>Think</td><td> $\overline { { \tau ^ { 2 } \left( \% \right) } }$ </td><td>UserBench (%)</td><td>AgentChangeBench (%)</td><td>BFCL-MT (%)</td><td>Avg (%)</td></tr><tr><td>GPT-4o-mini</td><td>一</td><td>一</td><td>23.75</td><td>28.30</td><td>35.50</td><td>37.00</td><td>31.14</td></tr><tr><td>Qwen3.5-397B</td><td>Base</td><td>w/</td><td>84.38</td><td>73.92</td><td>43.50</td><td>64.00</td><td>66.45</td></tr><tr><td rowspan="6">Qwen3-4B</td><td>Base</td><td>w/o</td><td>19.17</td><td>22.04</td><td>17.50</td><td>31.38</td><td>22.52</td></tr><tr><td>Base</td><td>w/</td><td>31.46</td><td>25.33</td><td>25.00</td><td>30.50</td><td>28.07</td></tr><tr><td>GRPO</td><td>w/o</td><td> $3 4 . 7 9 \pm 1 . 0 5$ </td><td> $2 3 . 2 7 \pm 0 . 4 7$ </td><td> $2 1 . 5 0 \pm 1 . 8 1$ </td><td> $3 1 . 6 9 \pm 0 . 1 3$ </td><td>27.81</td></tr><tr><td>GiGPO</td><td>w/o</td><td> $3 7 . 2 2 \pm 1 . 0 4$ </td><td> $2 5 . 0 6 \pm 0 . 2 5$ </td><td> $2 4 . 4 2 \pm 1 . 1 3$ </td><td> $3 2 . 4 2 \pm 0 . 5 6$ </td><td>29.78</td></tr><tr><td>InfoPO</td><td>w/o</td><td> $3 7 . 9 9 \pm 1 . 7 5$ </td><td> $2 0 . 4 7 \pm 1 . 1 0$ </td><td> $1 7 . 5 6 \pm 1 . 3 2$ </td><td> $2 9 . 9 4 \pm 0 . 3 4$ </td><td>26.49</td></tr><tr><td>IAPO (Ours)</td><td>w/o</td><td> $\overline { { 3 8 . 4 0 \pm 1 . 6 3 } }$ </td><td> $\overline { { { 2 6 . 8 9 \pm 0 . 6 1 } } }$ </td><td> $\overline { { 2 8 . 0 6 \pm 2 . 5 8 } }$ </td><td> $\overline { { 3 1 . 8 9 \pm 0 . 9 7 } }$ </td><td>31.31</td></tr><tr><td rowspan="7">Qwen3-8B ToolACE-2-8B</td><td>Base</td><td>w/o</td><td>24.17</td><td>18.04</td><td>27.50</td><td>39.38</td><td>27.27</td></tr><tr><td>Base</td><td>w/</td><td>37.29</td><td>22.08</td><td>29.00</td><td>39.00</td><td>31.84</td></tr><tr><td>MUA-RL†</td><td>w/o</td><td>35.63</td><td>17.31</td><td>30.05</td><td>24.75</td><td>26.94</td></tr><tr><td>GRPO</td><td>w/o</td><td> $2 9 . 6 1 \pm 3 . 5 6$ </td><td> $1 9 . 8 3 \pm 0 . 1 9$ </td><td> $2 6 . 0 3 \pm 1 . 7 9$ </td><td> $3 9 . 0 4 \pm 0 . 7 1$ </td><td>28.63</td></tr><tr><td>GiGPO</td><td>w/o</td><td> $3 5 . 8 3 \pm 1 . 0 4$ </td><td> $2 2 . 7 7 \pm 0 . 2 1$ </td><td> $2 9 . 2 5 \pm 0 . 8 8$ </td><td> $\mathbf { 4 0 . 0 0 \pm 0 . 4 4 }$ </td><td>31.96</td></tr><tr><td>InfoPO</td><td>w/o</td><td> $3 3 . 2 9 \pm 0 . 7 7$ </td><td> $2 3 . 3 7 \pm 0 . 3 8$ </td><td> $2 8 . 6 1 \pm 1 . 4 4$ </td><td> $3 8 . 0 1 \pm 0 . 6 3$ </td><td>30.82</td></tr><tr><td>IAPO (Ours)</td><td>w/o</td><td> $\overline { { 4 2 . 1 8 \pm 2 . 5 5 } }$ </td><td> $\overline { { 2 4 . 4 4 \pm 0 . 1 5 } }$ </td><td> $\overline { { 3 0 . 4 4 \pm 2 . 2 0 } }$ </td><td> $3 9 . 3 6 \pm 0 . 9 5$ </td><td>34.11</td></tr><tr><td colspan="8">External reported numbers (quoted from prior work trained on larger, synthetically constructed corpora)</td></tr><tr><td></td><td>SFT</td><td></td><td>20.53</td><td></td><td></td><td>37.00</td><td></td></tr><tr><td>BitAgent-8B</td><td>SFT</td><td>一</td><td>16.70</td><td></td><td></td><td>37.75</td><td></td></tr><tr><td>Fission-GRPO-4B</td><td>RL</td><td>w/</td><td>34.50</td><td></td><td></td><td>40.87</td><td></td></tr><tr><td>Fission-GRPO-8B</td><td>RL</td><td>w/</td><td>41.27</td><td></td><td></td><td>46.75</td><td></td></tr></table>

Table 1: Main results (%). ± is the standard deviation over three seeds. <sup>†</sup>: results re-evaluated using publicly available checkpoint. <sup>‡</sup>: results copied from the published papers. IAPO uses only the 178-task $\tau ^ { 2 } .$ -Bench training split, while Fission-GRPO (Zhang et al., 2026b) uses 630 tasks across 11 domains grounded in BFCL characteristics, designed and cleaned with Claude Sonnet 4. ToolACE-2-8B (Liu et al., 2025a) and BitAgent-8B (BitAgent, 2025) are specialized function-calling agents.

## Properties

The routing layer has four safeguards; proofs are in Appendix B.

Proposition 1 (Mass conservation). $\begin{array} { r } { \sum _ { i } \sum _ { \ell \in a _ { i } } A _ { i , \ell } = L \hat { A } _ { \tau } } \end{array}$ . IAPO redistributes the existing rollout advantage without changing its total token mass.

Proposition 2 (Uniform fallback). If the active graph features are constant, then $m _ { i } ^ { \pm } = 1 , w _ { i } = 1$ , and Eq. (7) assigns the same advantage to every token. With no observable error, ${ \phi } _ { i } ^ { - }$ is constant and the negative branch remains uniform.

Proposition 3 (Sign preservation). Because all routing multipliers are positive, every token advantage has the same sign as A<sup>ˆ</sup><sub>τ</sub>.

Proposition 4 (Optimization-interface preservation). IAPO preserves the terminal reward, group normalization, prompts, tools, rollout sampling, and optimization interface while intentionally changing the token-weighted policygradient estimator.

## Experiments

We evaluate IAPO through a controlled comparison with GRPO (Shao et al., 2024) and complementary comparisons with GiGPO (Feng et al., 2026) and InfoPO (Kong et al., 2026). GRPO (Shao et al., 2024) and IAPO share the same training protocol; their only difference is the trajectory-to-token advantage map. The evaluation covers success rate, per-domain breakdown, out-of-domain transfer, and annotator stability.

## Experimental Setup

Benchmarks. We train on the $\tau ^ { 2 } .$ -Bench (Barres et al., 2025) training split (airline/retail/telecom, 178 tasks) and evaluate on:

• $\tau ^ { 2 } .$ -Bench held-out test (20/40/40 tasks × 4 trials; domain-macro pass<sup>1</sup>);

• UserBench (Qian et al., 2025a) travel22/33/44 (101/87/67 cases; official multi-choice Score);

<table><tr><td>Benchmark</td><td>Split</td><td>GRPO</td><td>IAPO</td><td> $\Delta$ </td></tr><tr><td rowspan="3"> $\tau ^ { 2 }$ </td><td>Airline</td><td>22.78</td><td>29.72</td><td>+6.94</td></tr><tr><td>Retail</td><td>44.31</td><td>56.88</td><td>+12.57</td></tr><tr><td>Telecom</td><td>21.74</td><td>39.93</td><td>+18.19</td></tr><tr><td rowspan="3">UserBench</td><td>T22</td><td>20.68</td><td>25.65</td><td>+4.97</td></tr><tr><td>T33</td><td>19.41</td><td>24.50</td><td>+5.09</td></tr><tr><td>T44</td><td>19.10</td><td>22.55</td><td>+3.45</td></tr><tr><td rowspan="2">AgentChange</td><td>Banking</td><td>28.23</td><td>32.67</td><td>+4.44</td></tr><tr><td>Education</td><td>23.82</td><td>28.22</td><td>+4.40</td></tr></table>

Table 2: Per-split results at 8B (three-seed mean).

• AgentChangeBench (Rana et al., 2025) banking/education (100/100 tasks; TSR);

• BFCL-v4 Multi-Turn (function-calling retention) (Patil et al., 2024).

Baselines. GRPO (Shao et al., 2024) is the controlled baseline, differing from IAPO only in the trajectory-to-token advantage map. GiGPO (Feng et al., 2026) adds cross-rollout step credit by grouping actions with similar preceding observations; we use the authors’ official implementation with similarity-based grouping (γ=0.95, threshold 0.9, and ω=1.0), without further tuning. InfoPO (Kong et al., 2026) estimates per-turn information acquisition by comparing policy likelihoods under observed and masked contexts, fusing the resulting intrinsic reward with outcome advantage. The released MUA-RL-8B (Zhao et al., 2025) is included as an external checkpoint comparator; it was trained on $\tau ^ { 1 }$ retail/airline with a GPT-4o user simulator and is evaluated without further optimization. For broader context, we also report Fission-GRPO (Zhang et al., 2026b), ToolACE-2-8B (Liu et al., 2025a), and BitAgent-8B (BitAgent, 2025). Fission-GRPO (Zhang et al., 2026b) adds an error-recovery loop to GRPO: an error simulator diagnoses failed on-policy trajectories, constructs corrective samples, and triggers additional rollout batches for corrective training. ToolACE-2-8B (Liu et al., 2025a) and BitAgent-8B (BitAgent, 2025) are fine-tuned tool-use agents.

Training and evaluation. We train Qwen3-4B and Qwen3-8B with thinking disabled on 8× NVIDIA H20 GPUs using a verl/FSDP GRPO stack (Sheng et al., 2025; Zhao et al., 2023), N=16 samples/prompt, max 30 turns. Dependency annotation uses Qwen3-32B at temperature 0. Interactive evaluation follows the τ<sup>2</sup>-Bench protocol with held-out GPT-5.2 user simulators (Barres et al., 2025). IAPO defaults: $\beta _ { + } { = } 0 . 0 5 , w ^ { - } { = } m ^ { - } , c { = } 0 . 2 5$ . For each seed, we report the mean over the final three checkpoints; full hyperparameters and compute budget are in Appendix F.

## Main Results

Table 1 shows IAPO’s largest improvement on $\tau ^ { 2 } .$ -Bench: at 8B, it raises domain-macro pass<sup>1</sup> from 29.61% to 42.18%, a +12.57 pp gain over matched GRPO. The same update improves UserBench by +4.61 pp and AgentChangeBench by +4.41 pp, while BFCL-MT remains comparable (39.36% vs. 39.04%). Crucially, these in-domain gains do not come at the cost of out-of-distribution behavior: BFCL-MT stays effectively flat relative to GRPO at both scales, indicating that IAPO’s service-agent advantage reflects sharper credit assignment rather than overfitting away function-calling ability (Table 1).

Delayed state use yields the largest gain. Table 2 resolves the aggregate results by split. On $\tau ^ { 2 } .$ -Bench, IAPO improves airline from 22.78 to 29.72 (+6.94 pp), retail from 44.31 to 56.88 (+12.57 pp), and telecom from 21.74 to 39.93 (+18.19 pp). Telecom requires account or line state to persist until a later clarification determines its use; this is the setting in which a flat rollout advantage provides the least specific feedback about earlier actions.

Generalization. IAPO is trained only on $\tau ^ { 2 }$ -Bench, yet the same table shows it improving every out-of-domain split. UserBench requires accumulating preferences that arrive turn by turn: IAPO gains +4.97, +5.09, and +3.45 pp across the three travel splits. AgentChangeBench changes the desired outcome mid-interaction: IAPO gains +4.44 pp on banking and +4.40 pp on education, remaining ahead where account state must survive the goal shift. The higher success also comes with less redundant tool use: at 4B, IAPO lowers macro tool-call redundancy from GRPO’s 46.00 to 29.74 (Appendix F), so the gain reflects reuse of acquired state instead of repeated queries.

The router is far from flat. The improvement comes from a routing map that departs substantially from uniform credit. Across the audited action steps, the positive weight $w ^ { + }$ has standard deviation 0.222 and spans 0.637–1.505, and 86.5% of steps move by more than 10% from flat credit (Appendix E). The negative branch is deliberately sparser, with 27.1% of steps exceeding the same threshold, consistent with $\phi ^ { - }$ activating only after an observable error. IAPO changes the per-action advantage while departing substantially from the flat GRPO update.

<table><tr><td>Axis</td><td>Setting</td><td> $\tau ^ { 2 }$ </td><td>UB</td><td>AC</td></tr><tr><td>Ref.</td><td>GRPO</td><td>29.61 ± 3.56 19.83 ± 0.19 26.03 ± 1.79 32.50</td><td>20.92</td><td>27.58</td></tr><tr><td> $\beta _ { + }$ </td><td>0 .05 def. .1 .3</td><td>42.18 ± 2.55 24.44 ± 0.15 30.44 ± 2.20 33.96 31.39</td><td>25.33 24.08</td><td>30.92 28.83</td></tr><tr><td>C</td><td>.125 .25 def. .5</td><td>32.92 42.18 ± 2.55 24.44 ± 0.15 30.44 ± 2.20 42.29</td><td>19.11 17.78</td><td>30.17 25.67</td></tr><tr><td>w</td><td>cons. def. proj. 0 proj. .05</td><td>42.18 ± 2.55 24.44 ± 0.1530.44 ± 2.20 41.88 41.46</td><td>23.87 23.16</td><td>28.33 28.58</td></tr><tr><td></td><td>Annot. Q3-32B def. Gemma4-31B</td><td>42.18 ± 2.5524.44 ± 0.15 30.44 ± 2.20 40.90</td><td>24.45</td><td>33.25</td></tr></table>

Table 3: Ablation of routing hyperparameters and the frozen annotator on Qwen3-8B (%).

<table><tr><td>Annotator</td><td colspan="3">Edge agreement</td><td colspan="3">Routing agreement</td></tr><tr><td></td><td> $\operatorname { F } 1 _ { e }$ </td><td>Jaccard</td><td>Cohen&#x27;s κ</td><td> $\rho ( m ^ { + } )$ </td><td> $\rho ( m ^ { - } )$ </td><td>Sign agreement</td></tr><tr><td>Qwen3-32B (ref.)</td><td>0.852</td><td>0.822</td><td>0.812</td><td></td><td>一</td><td></td></tr><tr><td>Qwen3.5-27B</td><td>0.875</td><td>0.846</td><td>0.833</td><td>0.769</td><td>1.000</td><td>0.858</td></tr><tr><td>Gemma4-31B</td><td>0.878</td><td>0.857</td><td>0.849</td><td>0.745</td><td>0.996</td><td>0.858</td></tr><tr><td>DeepSeek-V4-Flash</td><td>0.879</td><td>0.852</td><td>0.843</td><td>0.758</td><td>0.975</td><td>0.877</td></tr><tr><td>Qwen3-235B-A22B</td><td>0.860</td><td>0.836</td><td>0.809</td><td>0.732</td><td>0.984</td><td>0.848</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>0.852</td><td>0.818</td><td>0.805</td><td>0.726</td><td>0.962</td><td>0.834</td></tr><tr><td>Mean</td><td>0.869</td><td>0.842</td><td>0.828</td><td>0.746</td><td>0.983</td><td>0.855</td></tr></table>

Table 4: Cross-model annotation consistency. Edge agreement reports F1, Jaccard, and Cohen’s κ; routing agreement reports Spearman correlations with the Qwen3-32B reference and sign agreement of the routed advantages.

## Ablation and Sensitivity

Each panel in Table 3 isolates one routing decision, either by disabling it or by sweeping its value: $\beta _ { + }$ controls how repaired errors affect positive-advantage rollouts, c bounds the weight spread, $w ^ { - }$ determines whether failure credit begins from support or observed error, and the annotator swap tests transfer across model families.

Support and error each carry useful signal. With $\beta _ { + } = 0 ,$ positive-advantage rollouts route credit by support use and negative-advantage ones route it by error propagation. This stripped rule already raises $\tau ^ { 2 }$ from 29.61 to 32.50. A small positive-branch gate $( \beta _ { + } { = } 0 . 0 5 )$ reaches 42.18 by reducing positive credit on actions whose earlier errors were repaired downstream. Within the $\beta _ { + }$ sweep, larger gates yield 33.96 and 31.39 on $\tau ^ { 2 }$ , indicating that error evidence refines support routing without taking its place.

Clipping prevents hubs from dominating the update. At c=0.125, routing stays near flat credit and reaches 32.92 on $\bar { \tau } ^ { 2 }$ . Increasing the bound to c=0.5 raises the in-domain score to 42.29, but UserBench and AgentChangeBench fall to 17.78 and 25.67. The default c=0.25 gives the strongest joint in-domain and transfer profile.

Failure routing follows error propagation, not support centrality. The conservative branch begins from observable error and reaches 42.18 on $\tau ^ { 2 }$ and 30.44 on AgentChangeBench. Both projected variants start from the support prior and are lower on these measures. The two panels rule out degree-based explanations: positive and negative advantage follow different evidence.

The gain transfers across annotator families. Replacing Qwen3-32B with Gemma4-31B yields $4 0 . 9 0 \ \mathrm { o n } \ \tau ^ { 2 }$ , 24.45 on UserBench, and 33.25 on AgentChangeBench. This transfer and the routing correlations in Table 4 indicate that IAPO does not rely on the default Qwen annotator’s particular outputs.

![](images/9334d7852029b91e9dc6b2f50d011d3ecb0923bb54fd54a073912b0d787cfdb5.jpg)  
Figure 3: Banking case study with three successive goal shifts. (Left) GRPO anchors on a schema-example identity and propagates the wrong account state. (Right) IAPO grounds the correct customer from supplied credentials and retains the linked state across goals.

## Annotator Stability and Behavioral Analysis

Cross-annotator stability. We audit all 178 $\tau ^ { 2 } .$ -Bench training tasks by rolling out the untrained Qwen3-8B policy once per task, yielding 2,349 annotated action nodes across airline, retail, and telecom. Six frozen annotators achieve 100% parse rate. Their canonical edge sets show substantial set-level agreement (mean edge-typed F1 0.87, source Jaccard 0.84, mean pairwise Cohen’s κ=0.83). The routing quantities used by IAPO are also stable: against the Qwen3-32B reference, $m ^ { + }$ has Spearman $\rho = 0 . 7 3 – 0 . 7 7$ , while $m ^ { - }$ has $\rho = 0 . 9 6 \mathrm { - 1 . 0 0 ( T a b l e 4 ) }$ . We further audit the lowest-agreement nodes in Appendix E. Disagreements fall mainly into benign, interpretable sources: interchangeable boilerplate turns (greetings, closings) where annotators pick different but near-identical preceding turns, and attribution breadth on already-failed steps (e.g., hallucinated tool calls) where annotators differ in attribution breadth, while generally agreeing that a fault exists. Neither source changes the routing multipliers, so agreement holds at both the edge and induced-routing levels.

Behavioral case study. Figure 3 illustrates a banking episode in AgentChangeBench (Rana et al., 2025) whose goals shift from authentication to transaction review and fraud response. Before receiving identity information, GRPO uses the example phone exposed in the tool schema, obtains Maria Santos (cust\_101), and propagates the resulting account state (acc\_101) into its first transaction query. When the user later supplies the actual phone number, GRPO does not re-ground the customer and instead transfers to a human. IAPO first grounds Jordan Smith from the supplied name and date of birth, then retains the linked state (cust\_202, acc\_202, and card\_202) across goal shifts, eventually locking card\_202. The case study isolates state grounding and cross-goal reuse.

Function-calling retention evaluation. We use BFCL-v4 Multi-Turn (Patil et al., 2024) as a function-calling retention check. Table 5 reports BFCL sub-splits at both scales. Relative to the untrained base, IAPO’s BFCL-MT score is essentially unchanged at 8B (39.36 vs. 39.38) and higher at 4B (31.89 vs. 31.38), so interactive training leaves out-ofdistribution function calling intact.

<table><tr><td>Model</td><td>Method</td><td>Multi- Turn</td><td>Base</td><td>Multi- Function</td><td>Multi- Parameter</td><td>Long</td></tr><tr><td>4B</td><td>Base (w/)</td><td>31.38</td><td>43.50</td><td>27.00</td><td>28.50</td><td>26.50</td></tr><tr><td></td><td>GRPO</td><td>31.69</td><td>39.61</td><td>30.67</td><td>28.28</td><td>28.22</td></tr><tr><td></td><td>GiGPO</td><td>32.42</td><td>31.83</td><td>32.33</td><td>32.67</td><td>32.83</td></tr><tr><td></td><td>InfoPO</td><td>29.94</td><td>37.94</td><td>29.00</td><td>26.50</td><td>26.33</td></tr><tr><td></td><td>IAPO</td><td>31.89</td><td>40.06</td><td>30.78</td><td>28.28</td><td>28.44</td></tr><tr><td>8B</td><td>Base (w/)</td><td>39.38</td><td>49.00</td><td>37.50</td><td>41.50</td><td>29.50</td></tr><tr><td></td><td>MUA-RL†</td><td>24.75</td><td>35.50</td><td>21.50</td><td>22.00</td><td>20.00</td></tr><tr><td></td><td>GRPO</td><td>39.04</td><td>48.94</td><td>40.67</td><td>34.78</td><td>31.78</td></tr><tr><td></td><td>GiGPO</td><td>40.00</td><td>40.17</td><td>39.67</td><td>40.33</td><td>39.83</td></tr><tr><td></td><td>InfoPO</td><td>38.01</td><td>49.39</td><td>37.44</td><td>30.78</td><td>34.44</td></tr><tr><td></td><td>IAPO</td><td>39.36</td><td>48.33</td><td>39.61</td><td>36.00</td><td>33.50</td></tr></table>

Table 5: BFCL-v4 Multi-Turn sub-split accuracy (%).

## Conclusion

In this work, we proposed IAPO, a novel RL algorithm to tackle the credit assignment challenge in evolving-state multi-turn service-agent training. IAPO introduces a typed influence-dependency graph that enables fine-grained per-step credit assignment while retaining the original trajectory-level outcome signal. By extracting support-use and failed-use dependencies from completed rollouts, it achieves this without replacing the final outcome with an independently learned local reward or requiring counterfactual continuations. Across three interactive service-agent benchmarks, IAPO consistently improves over GRPO and competitive credit-assignment baselines. The gains are largest in interactions with rich user and tool exchanges, consistent with the intuition that dense dependencies among actions are where uniform trajectory-level credit is least informative. These results establish within-rollout dependency structure as a practical source of selective credit for sparse-outcome multi-turn agent training.

## References

Arjona-Medina, J. A.; Gillhofer, M.; Widrich, M.; Unterthiner, T.; Brandstetter, J.; and Hochreiter, S. 2019. RUDDER: Return Decomposition for Delayed Rewards. In Advances in Neural Information Processing Systems, volume 32.

Barres, V.; Dong, H.; Ray, S.; Si, X.; and Narasimhan, K. 2025. τ<sup>2</sup>-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982.

BitAgent. 2025. BitAgent-8B. https://huggingface.co/BitAgent/BitAgent-8B. Commit hash ca31a77.

Budzianowski, P.; Wen, T.-H.; Tseng, B.-H.; Casanueva, I.; Ultes, S.; Ramadan, O.; and Gasic, M. 2018. MultiWOZ – A Large-Scale Multi-Domain Wizard-of-Oz Dataset for Task-Oriented Dialogue Modelling. In Proceedings of the 2018 conference on empirical methods in natural language processing, 5016–5026.

Cui, G.; Yuan, L.; Wang, Z.; Wang, H.; Zhang, Y.; Chen, J.; Li, W.; He, B.; Fan, Y.; Yu, T.; et al. 2025. Process reinforcement through implicit rewards. arXiv:2502.01456.

DeepSeek-AI; Liu, A.; Feng, B.; Xue, B.; Wang, B.; Wu, B.; Lu, C.; Zhao, C.; Deng, C.; Zhang, C.; Ruan, C.; et al. 2025. DeepSeek-V3 Technical Report. arXiv:2412.19437.

Deng, X.; Gu, Y.; Zheng, B.; Chen, S.; Stevens, S.; Wang, B.; Sun, H.; and Su, Y. 2023. Mind2web: Towards a generalist agent for the web. In Advances in Neural Information Processing Systems, volume 36, 28091–28114.

Ding, Y.; Le, H.; Han, S.; Ruan, K.; Jin, Z.; Kumar, V.; Wang, Z.; and Deoras, A. 2026. Empowering multi-turn tool-integrated agentic reasoning with group turn policy optimization. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 42409–42423.

Feng, L.; Xue, Z.; Liu, T.; and An, B. 2026. Group-in-group policy optimization for llm agent training. Advances in Neural Information Processing Systems, 38: 46375–46408.

Harutyunyan, A.; Dabney, W.; Mesnard, T.; Azar, M. G.; Piot, B.; Heess, N.; van Hasselt, H.; Wayne, G.; Singh, S.; Precup, D.; et al. 2019. Hindsight Credit Assignment. In Advances in Neural Information Processing Systems.

Ji, Y.; Ma, Z.; Wang, Y.; Chen, G.; Chu, X.; and Wu, L. 2025. Tree Search for LLM Agent Reinforcement Learning. arXiv:2509.21240.

Kong, F.; Zhang, J.; Deng, M.; Wu, C.; Luo, Y.; and Liu, B. 2026. InfoPO: Information-Driven Policy Optimization for User-Centric Agents. In Forty-third International Conference on Machine Learning.

Lai, X.; Tian, Z.; Chen, Y.; Yang, S.; Peng, X.; and Jia, J. 2024. Step-DPO: Step-Wise Preference Optimization for Long-Chain Reasoning of LLMs. arXiv:2406.18629.

Li, M.; Zhao, Y.; Yu, B.; Song, F.; Li, H.; Yu, H.; Li, Z.; Huang, F.; and Li, Y. 2023. API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. In Proceedings of the 2023 conference on empirical methods in natural language processing, 3102–3116.

Li, Z.; Kang, L.; Xiao, F.; Xing, L.; Si, Q.; Li, Z.; Gong, W.; Yang, D.; Xiao, Y.; and Guo, H. 2026. Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment in Mathematical Reasoning. arXiv:2601.07408.

Lightman, H.; Kosaraju, V.; Burda, Y.; Edwards, H.; Baker, B.; Lee, T.; Leike, J.; Schulman, J.; Sutskever, I.; and Cobbe, K. 2024. Let’s Verify Step by Step. In International Conference on Learning Representations, volume 2024, 39578–39601.

Liu, W.; Huang, X.; Zeng, X.; Hao, X.; Yu, S.; Li, D.; Wang, S.; Gan, W.; Liu, Z.; Yu, Y.; Wang, Z.; Wang, Y.; Ning, W.; Hou, Y.; Wang, B.; Wu, C.; Wang, X.; Liu, Y.; Wang, Y.; Tang, D.; Tu, D.; Shang, L.; Jiang, X.; Tang, R.; Lian, D.; Liu, Q.; and Chen, E. 2025a. ToolACE: Winning the Points of LLM Function Calling. In The Thirteenth International Conference on Learning Representations.

Liu, X.; Yu, H.; Zhang, H.; Xu, Y.; Lei, X.; Lai, H.; Gu, Y.; Ding, H.; Men, K.; Yang, K.; et al. 2024. Agentbench: Evaluating llms as agents. In International Conference on Learning Representations, volume 2024, 52989–53046.

Liu, Z.; Chen, C.; Li, W.; Qi, P.; Pang, T.; Du, C.; Lee, W. S.; and Lin, M. 2025b. Understanding r1-zero-like training: A critical perspective. arXiv:2503.20783.

Mesnard, T.; Weber, T.; Viola, F.; Thakoor, S.; Saade, A.; Harutyunyan, A.; Dabney, W.; Stepleton, T.; Heess, N.; Guez, A.; et al. 2021. Counterfactual Credit Assignment in Model-Free Reinforcement Learning. In International Conference on Machine Learning.

Mialon, G.; Fourrier, C.; Wolf, T.; LeCun, Y.; and Scialom, T. 2024. GAIA: A Benchmark for General AI Assistants. In International Conference on Learning Representations, volume 2024, 9025–9049.

Ng, A. Y.; Harada, D.; and Russell, S. 1999. Policy invariance under reward transformations: Theory and application to reward shaping. In International Conference on Machine Learning, volume 99, 278–287. Citeseer.

OpenAI; Achiam, J.; Adler, S.; Agarwal, S.; Ahmad, L.; Akkaya, I.; Aleman, F. L.; Almeida, D.; Altenschmidt, J.; Altman, S.; Anadkat, S.; et al. 2024. GPT-4 Technical Report. arXiv:2303.08774.

Parthasarathi, P.; Reymond, M.; Chen, B.; Cui, Y.; and Chandar, S. 2025. GRPO-λ: Credit Assignment improves LLM Reasoning. arXiv:2510.00194.

Patil, S. G.; Zhang, T.; Wang, X.; and Gonzalez, J. E. 2024. Gorilla: Large language model connected with massive apis. In Advances in Neural Information Processing Systems, volume 37, 126544–126565.

Qian, C.; Liu, Z.; Prabhakar, A.; Liu, Z.; Zhang, J.; Chen, H.; Ji, H.; Yao, W.; Heinecke, S.; Savarese, S.; Xiong, C.; and Wang, H. 2025a. UserBench: An Interactive Gym Environment for User-Centric Agents. arXiv:2507.22034.

Qian, C.; Liu, Z.; Prabhakar, A.; Qiu, J.; Liu, Z.; Chen, H.; Kokane, S.; Ji, H.; Yao, W.; Heinecke, S.; et al. 2025b. UserRL: Training Interactive User-Centric Agent via Reinforcement Learning. arXiv:2509.19736.

Qin, Y.; Liang, S.; Ye, Y.; Zhu, K.; Yan, L.; Lu, Y.; Lin, Y.; Cong, X.; Tang, X.; Qian, B.; et al. 2023. ToolLLM: Facilitating Large Language Models to Master 16000+ Real-World APIs. In The twelfth international conference on learning representations.

Rana, M.; Man, C.; Msiiwa, A. E.; Paine, J.; Zhu, K.; Dev, S.; Sharma, V.; and R, A. M. 2025. AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness in Conversational AI. arXiv:2510.18170.

Samanta, A.; Magesh, A.; Jain, A.; Yu, Y.; Jiang, D.; Asadi, K.; Hassani, K.; Sajda, P.; Bhandari, J.; and Efroni, Y. 2026. Credit Assignment with Resets in Language Model Reasoning. arXiv:2605.25507.

Schick, T.; Dwivedi-Yu, J.; Dessì, R.; Raileanu, R.; Lomeli, M.; Hambro, E.; Zettlemoyer, L.; Cancedda, N.; and Scialom, T. 2023. Toolformer: Language Models Can Teach Themselves to Use Tools. In Advances in Neural Information Processing Systems, volume 36, 68539–68551.

Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal policy optimization algorithms. arXiv:1707.06347.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y.; Wu, Y.; et al. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Sheng, G.; Zhang, C.; Ye, Z.; Wu, X.; Zhang, W.; Zhang, R.; Peng, Y.; Lin, H.; and Wu, C. 2025. HybridFlow: A Flexible and Efficient RLHF Framework. In Proceedings ofthe Twentieth European Conference on Computer Systems, EuroSys ’25, 1279–1297. New York, NY, USA: Association for Computing Machinery. ISBN 9798400711961.

Shridhar, M.; Yuan, X.; Côté, M.-A.; Bisk, Y.; Trischler, A.; and Hausknecht, M. 2021. ALFWorld: Aligning Text and Embodied Environments for Interactive Learning. In Proceedings ofthe International Conference on Learning Representations.

Tan, H.-Z.; Yang, X.-W.; Chen, H.; Shao, J.-J.; Wen, Y.; Shen, Y.; Luo, W.; Du, X.; Guo, L.-Z.; and Li, Y.-F. 2026. Hindsight credit assignment for long-horizon llm agents. arXiv:2603.08754.

Team, G.; Anil, R.; Borgeaud, S.; Alayrac, J.-B.; Yu, J.; Soricut, R.; Schalkwyk, J.; Dai, A. M.; Hauth, A.; Millican, K.; et al. 2023. Gemini: a family of highly capable multimodal models. arXiv:2312.11805.

Ultes, S.; Barahona, L. M. R.; Su, P.-H.; Vandyke, D.; Kim, D.; Casanueva, I.; Budzianowski, P.; Mrkšic, N.; Wen,´ T.-H.; Gasic, M.; et al. 2017. PyDial: A Multi-Domain Statistical Dialogue System Toolkit. In Proceedings of the Annual Meeting ofthe Associationfor Computational Linguistics, System Demonstrations, 73–78.

Wang, X.; Wang, Z.; Liu, J.; Chen, Y.; Yuan, L.; Peng, H.; and Ji, H. 2023. Mint: Evaluating llms in multi-turn interaction with tools and language feedback. arXiv:2309.10691.

Wang, Z.; Wang, K.; Wang, Q.; Zhang, P.; Li, L.; Yang, Z.; Jin, X.; Yu, K.; Nguyen, M. N.; Liu, L.; et al. 2025. RA-GEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning. arXiv:2504.20073.

Wei, Q.; Zeng, S.; Li, C.; Brown, W.; Frunza, O.; Deng, W.; Schneider, A.; Nevmyvaka, Y.; Zhao, Y. K.; Garcia, A.; et al. 2025. Reinforcing multi-turn reasoning in llm agents via turn-level reward design. arXiv:2505.11821.

Xie, Y.; Thomas, N.; Hansen, N.; Fu, Y.; Li, L. E.; and Wang, X. 2026. Tips: Turn-level information-potential reward shaping for search-augmented llms. arXiv:2603.22293.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 Technical Report. arXiv:2505.09388.

Yao, S.; Chen, H.; Yang, J.; and Narasimhan, K. 2022a. WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents. In Advances in Neural Information Processing Systems, volume 35, 20744–20757.

Yao, S.; Shinn, N.; Razavi, P.; and Narasimhan, K. 2024. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045.

Yao, S.; Zhao, J.; Yu, D.; Du, N.; Shafran, I.; Narasimhan, K.; and Cao, Y. 2022b. ReAct: Synergizing Reasoning and Acting in Language Models. arXiv:2210.03629.

Yu, Q.; Zhang, Z.; Zhu, R.; Yuan, Y.; Zuo, X.; Yue, Y.; Dai, W.; Fan, T.; Liu, G.; Liu, L.; et al. 2026. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38: 113222–113244.

Zhang, Y.; Tang, S.; Li, Z.; Han, Z.; and Tresp, V. 2026a. WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents. arXiv:2601.21872.

Zhang, Z.; Zhao, F.; Wang, R.; Wang, Z.; Liang, B.; Wang, J.; Hu, Y.; Cao, S.; and Wong, K.-F. 2026b. Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors. In Proceedings of the Conference of the Associationfor Computational Linguistics.

Zhao, W.; Wang, X.; Ma, C.; Kong, L.; Yang, Z.; Tuo, M.; Shi, X.; Zhai, Y.; and Cai, X. 2025. MUA-RL: Multi-Turn User-Interacting Agent Reinforcement Learning for Agentic Tool Use. arXiv:2508.18669.

Zhao, Y.; Gu, A.; Varma, R.; Luo, L.; Huang, C.-C.; Xu, M.; Wright, L.; Shojanazeri, H.; Ott, M.; Shleifer, S.; et al. 2023. PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel. arXiv:2304.11277.

Zheng, C.; Liu, S.; Li, M.; Chen, X.-H.; Yu, B.; Gao, C.; Dang, K.; Liu, Y.; Men, R.; Yang, A.; et al. 2025. Group sequence policy optimization. arXiv:2507.18071.

Zhou, S.; Xu, F. F.; Zhu, H.; Zhou, X.; Lo, R.; Sridhar, A.; Cheng, X.; Ou, T.; Bisk, Y.; Fried, D.; et al. 2024a. Webarena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, volume 2024, 15585–15606.

Zhou, Y.; Jiang, S.; Tian, Y.; Weston, J.; Levine, S.; Sukhbaatar, S.; and Li, X. 2025. SWEET-RL: Training Multi-Turn LLM Agents on Collaborative Reasoning Tasks. arXiv:2503.15478.

Zhou, Y.; Zanette, A.; Pan, J.; Levine, S.; and Kumar, A. 2024b. Archer: Training language model agents via hierarchical multi-turn rl. arXiv:2402.19446.