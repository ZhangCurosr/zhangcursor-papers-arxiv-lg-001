# Ask the Expert: LLM-Guided Reinforcement Learning for Autonomous Cyber Defense

Fernando Martinez<sup>†</sup>, Abhishek Satyam<sup>†</sup>, Tao Li, Junaid Farooq, Ying Wang, and Juntao Chen

Abstract—Policy-based reinforcement learning (RL) approaches have produced promising results for autonomous cyber defense; however, they are sample-inefficient in settings where defenders must respond under delayed, partial observations with actions from large action spaces. While large language models (LLMs) may reason semantically about security state space, high latency and trust assumptions prevent attractive in-line deployment models. We introduce Ask the Expert, a training-time guidance framework which first summarizes hard cyber-defense states, then intermittently queries an LLM for host-level defensive recommendations via a constrained action interface, and finally transforms those recommendations into tiered reward shaping for use with PPO. Because the LLM is discarded after training, deployment is a pure RL policy. Across TTCP CAGE CC1 and CC2 and both attacker types, this asymmetric design improves sample efficiency over PPO and outperforms the evaluated potential-based reward shaping (PBRS) baselines, while retaining the strongest terminal mean and requiring no LLM dependency at deployment time.

Index Terms—Autonomous cyber defense, large language models, reinforcement learning, reward shaping.

## I. INTRODUCTION

Enterprise networks are subject to continual risk from postcompromise lateral movement campaigns that slowly infiltrate their networks by reusing credentials and pivoting across hosts [1]. Embedded autonomous agents intended to respond to these intrusions must act under tight time constraints and with incomplete network observability. Successfully responding requires agents to reason over local alerts and choose from a large, semantically dense action space (e.g., conducting host forensics, terminating sessions, and deploying remediation), while being predominantly guided by dense but tactically weak reward signals.

Reinforcement learning (RL), and policy gradient methods like PPO [2] in particular, scale to learning policies over large structured discrete action spaces [3], [4]. Unfortunately, applying RL solutions directly to the response problem can behave destructively during early exploration: unfettered RL agents will drive significant service degradation for normal users until eventually stumbling onto reasonable behavior [5]. Moreover, the scalar reward emitted by the environment represents global network health, but does not provide meaningful, tactically valid feedback for granular actions an agent takes [6]. The bulk of early agent training time is spent performing no-knowledge exploration until converging on a reasonable policy, if at all. Classical potential-based reward shaping (PBRS) provides the theoretical backdrop for altering rewards while preserving optimal policies under restricted transformations [7], but practical cyber defense still relies heavily on heuristic shaping rules. We address this gap by using an empirically inspired, training-time shaping scheme derived from LLM advice rather than claiming policy invariance.

Separately, Large Language Models (LLMs) have been applied to qualitative cybersecurity reasoning tasks [8]–[10] as well as reward generation for automated RL problems [11], [12]. But interacting with an LLM directly as an online runtime agent would incur unacceptable inference latency and present dangerous trust-boundary risks. Fortunately, we can leverage similar asymmetric training regimes from SCORER [13] and CORAL [14] that allow us to inject enriched auxiliary knowledge during training-time. This leaves us with the question: how can we harness the zero-shot reasoning power of a pre-trained LLM as an asymmetric teacher?

We present Ask the Expert, an LLM-guided RL framework that intermittently samples hard states in which the defender first observes a newly spreading compromise. These states are passed to a zero-shot LLM, which returns host-focused remediation advice. The advice is distilled into a persistent tiered reward-shaping policy used only during training. After training, the LLM is discarded, leaving a fast and fully autonomous deployment policy.

Our contributions are as follows:

• We introduce Ask the Expert, a training-time-only LLMguided reward-shaping framework for autonomous cyber defense that improves RL learning without requiring any deployment-time LLM calls or added inference latency.

• We design a recent-frontier prompting mechanism that stores hard states involving newly spreading compromises, prioritizes recent severe failures, caps prompt concentration on any one spreading host, and distills host-level LLM guidance into persistent tiered priors over remediation actions.

• We conduct a multi-seed evaluation on CAGE CC1 and CC2 against two attacker types, showing improved sample efficiency over PPO and observation-based PBRS baselines.

## II. RELATED WORK

RL for autonomous cyber defense. Past research has explored the use of RL for broader network intrusion detection and response efforts [3], [4], [15]. These environments must balance attack containment against availability under partial observability. Existing benchmarks like CybORG helped standardize the problem [16]. Prior solutions often relied on exploration through heuristic reward shaping or action masking [5], [6]. However, these schemes rely on static rules curated by domain experts. Our recent-frontier prompt sampler is closer in spirit to prioritized selection [17] than to uniform replay, although it prioritizes LLM prompt candidates rather than TD updates. We then convert the resulting advice into dynamically tiered reward bonuses rather than fixed hand-designed heuristics.

Auxiliary signals and structured advice for RL. Structured training signals can improve RL sample efficiency and generalization without deployment overhead. Policy Shaping [18] was among the first to translate external advice into policy guidance rather than reward relabeling. Other asymmetrictraining methods use auxiliary signals to guide representations or messages, including SCORER [13] and CORAL [14]. Wang et al. [19] combine LLM-generated action demonstrations and reward shaping with constrained PPO for cloud-edge resource allocation. In contrast, we use intermittent host-level advice only as training-time reward bonuses, without demonstrations or deployment-time LLM calls.

LLMs for RL and Security. Large language models have been applied to RL reward design: Eureka [11] and Text2Reward [20] generate reward code from natural-language descriptions, while ELLM [12] and PRM4RL [21] prompt for subgoals or progress rewards; CARD [22] and LLMHRL [23] apply LLMs to reward reshaping and hierarchical control. In security, deployment latency and trust concerns make direct runtime LLM use unattractive [8]–[10]. In contrast, we use intermittent LLM advice only at training time, operationalized as reward bonuses, then discard the LLM at deployment.

## III. PROBLEM FORMULATION

## A. Modeling Network Defense as a POMDP

We model autonomous intrusion response as a singleagent partially observable Markov decision process (POMDP) $( S , \mathcal { O } , \mathcal { A } , \mathcal { T } , r , \gamma )$ , in which an autonomous Blue defender interacts with a concurrent adversarial Red agent. The true global state of the world $s _ { t } ~ \in ~ S$ is unknown. Instead, the defender receives delayed, partial observations $o _ { t } \in \mathcal { O }$ . These observations are based on host alerts and network telemetry, indicative of the consequences of compromise rather than direct access to the attacker’s location. At no point during training or deployment does the defender receive observations of Red’s internal state or intended future plan.

## B. Threat Model

We focus on adversarial post-compromise lateral movement [24] in CAGE CC1 and CC2. Red starts every episode with a foothold at User0. We benchmark against two attacker policies: B\_lineAgent, which attempts a direct pivot toward the operational server, and RedMeanderAgent, which sweeps subnets before penetrating inward. Regardless, Blue reacts solely to defender-visible aftermath signals.

## C. Action Space and Semantics

The defender operates over a large discrete verb–host action space A. The PPO agent acts over the entire environment action space, while the LLM-facing action interface only exposes a much smaller host-level remediation subspace. This allows us to maintain a much simpler language interface while still retaining full fidelity of the learned policy.

## D. Reward and Objective

The environment reward function $r _ { t }$ penalizes compromised hosts based on their criticality to operations. Negative effects of being overly aggressive, such as unnecessarily disrupting benign services that were not threats, are captured in the resulting aggregate reward. The policy aims to maximize discounted return:

$$
J ( \pi ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { T } \gamma ^ { t } r _ { t } \right] .\tag{1}
$$

Hidden penalties continuously obfuscate the tactical value of immediate actions, leading rudimentary agents either to hesitate during active pivoting or to blindly disrupt critical services or subnets. The role of our framework is therefore to add semantic training-time guidance without changing the underlying environment dynamics or deployment-time policy class.

## IV. ASK THE EXPERT FRAMEWORK

## A. Framework Architecture

Our framework supplements a standard policy-gradient RL agent with three modules run at training time (Figure 1): (i) a semantic encoder $\phi$ that converts defender reward-breakdown telemetry into sets of compromised hosts, newly spreading hosts, and clean hosts; (ii) a fixed-capacity candidate bag $\mathcal { C }$ and recent-frontier sampler that stores recent spreading-state transitions and periodically queries an LLM; and (iii) a tiered reward-shaping policy $\Pi _ { \mathrm { L L M } }$ that produces host-specific bonus weights for semantically related actions. PPO always takes its own actions; the LLM never intervenes on action selection and can only affect training via reward shaping. The LLM is discarded after training, leaving only $\pi _ { \theta }$

## B. Semantic State Encoder

The semantic encoder φ is deterministic. At each timestep, it inputs defender-visible reward-breakdown telemetry along with the previously known set of compromised hosts, and outputs three summaries: compromised hosts, newly detected hosts, and clean hosts. Newly detected hosts comprise the spreading set

$$
S _ { t } = { \mathrm { C o m p r o m i s e d } } _ { t } \setminus { \mathrm { C o m p r o m i s e d } } _ { t - 1 } .\tag{2}
$$

The produced semantic summary $\zeta _ { t }$ is compact and easily digestible by a general-purpose language model. The summary

![](images/ab3492e77166a86e052599a2d97d44bc6b9beeccad87dc2fd3631d8d9a50f565.jpg)  
Fig. 1. The Ask the Expert architecture shared across both B\_lineAgent and RedMeanderAgent experiments. Reward-breakdown telemetry is encoded into compromised, spreading, and clean host sets; spreading hard states are stored in the candidate bag; a recent-frontier sampler builds the LLM prompt; and the returned host-level advice is merged into a tiered reward-shaping policy applied at rollout end before PPO advantage computation. Deployment remains pure RL with no LLM calls.

exposes defender-available information and contains no testtime dependencies on the simulator’s hidden state.

## C. Hard-State Queue and LLM Querying

We maintain a fixed-capacity candidate bag C of spreadingstate transitions. This bag stores $( \zeta _ { t } , a _ { t } , r _ { t } , \Delta r _ { t } , n )$ whenever $S _ { t } \neq \emptyset .$ , where $\Delta r _ { t } = r _ { t } - r _ { t - 1 }$ is stored as context for the LLM but is unused as a storage gate. Entries are dropped from the head of C whenever $| { \mathcal { C } } | > B _ { \mathrm { m a x } }$

Once per LLM query iteration, the sampler filters C down to a frontier of recent failures, ranks entries by the most negative reward deltas, caps any single spreading host at three prompt slots, and selects up to $n _ { s }$ entries subject to that host cap. The resulting prompt $p$ conditions on both network context and a maintained history set H of past LLM suggestions and their observed outcomes, and for each sampled hard state appends the semantic summary, PPO’s selected action, and a qualitative outcome tag indicating reward deterioration, no change, or reward improvement. The LLM is constrained to a fixed host-level remediation vocabulary at host granularity, and invalid outputs are rejected during parsing; in CC2 this vocabulary contains Analyse, Remove, and Restore, while in CC1 it additionally includes Misinform, which deploys a decoy service on a host to delay and expose attacker lateral movement. This frontier-sampled host-level guidance mechanism is shared across attacker configurations, while allowing the prompt doctrine itself to be specialized to attacker structure (e.g. B\_lineAgent’s direct pivots vs. RedMeanderAgent’s subnet sweeps).

## D. Tiered LLM-Guided Reward Shaping

To eliminate deployment-time dependence on the LLM, we utilize its suggestions only through reward-shaping during

training. For each host $h ,$ the shaping module stores a tiered policy:

$$
\Pi _ { \mathrm { L L M } } ( h ) = \{ ( a _ { i } , w _ { i } ) \} ,\tag{3}
$$

where $a _ { i }$ is an action string and $w _ { i } \in ( 0 , 1 ]$ is that action’s bonus weight. Each host is initialized with weak default tiers over conservative remediation actions. After receiving a valid LLM response, its suggestion becomes $\Pi _ { \mathrm { L L M } } ( h ) ^ { \ }$ s single highweight primary tier, while semantically-related actions pertaining to host h are inserted as fallback tiers with lower weight. Specific tier weights are determined below as part of experimental instantiation.

After rollout concludes and before generalized advantage estimation (GAE), the method revisits each buffered timestep associated with a spreading host $h _ { t } .$ . If PPO chose action $a _ { t }$ which appears in $\Pi _ { \mathrm { L L M } } ( h _ { t } )$ , it reshapes the reward logged at that timestep via

$$
\tilde { r } _ { t } = r _ { t } + \beta \cdot w ( a _ { t } )\tag{4}
$$

where $\beta ~ > ~ 0$ is a global scalar bonus parameter and $w ( a _ { t } ) ~ \in ~ ( 0 , 1 ]$ is the weight of the tier matched by $a _ { t } .$ . We fix $\beta = 0 . 0 1$ for all experiments, so $\tilde { r } _ { t }$ always satisfies $0 \leq$ $\tilde { r } _ { t } - r _ { t } \le \beta$ . Note that the LLM is never invoked during action selection. It can only influence training through reward shaping given our constrained host-level action interface: unparsable recommendations are discarded, and PPO solely determines which action will be executed. If PPO chooses an action which is not matched by any tier, then $\tilde { r } _ { t } = r _ { t }$ . Otherwise, the original environment reward is kept but can be supplemented by up to $\beta .$ Thus, an erroneous recommendation cannot overwrite PPO’s selected action directly nor add more than $\beta$ to a single reward. Reward shaping also only happens at logged spreading-state timesteps, and host policies eventually expire after some fixed number of batches and revert to weak defaults.

## E. Policy Optimization and Advice Merging

The RL agent maintains standard policy $\pi _ { \boldsymbol { \theta } } ( \cdot | _ { O _ { t } } )$ and value network $V _ { \psi } ( o _ { t } )$ . Given a trajectory buffer $B ,$ the agent computes generalized advantage estimates $\hat { A } _ { t }$ using the shaped rewards $\tilde { r } _ { t }$ and updates its parameters with a clipped surrogate policy gradient objective [2] at every iteration.

To maximize the value of infrequent LLM consultations, newly generated host policies are merged into a global advice dictionary rather than replacing earlier entries. The current guidance mechanism also applies a conservative fallback when a spreading host in the current rollout is not explicitly mentioned by the latest validated LLM response: that host is refreshed with a full-weight Restore tier. In our method, stale host policies expire after a fixed number of batches and revert to the weak default priors, ensuring the active advice dictionary remains focused on the current failure frontier. The complete training procedure is outlined in Algorithm 1.

## V. EXPERIMENTAL EVALUATION

## A. Experimental Details

We evaluate on two TTCP CAGE Challenges [25] built on the CybORG simulation environment [16]: CAGE Challenge 1 (CC1) and Challenge 2 (CC2). Both environments model a medium-sized enterprise network with 13 hosts across user, enterprise, and operational subnets, with episodes of 100 timesteps and partial defender observations from host alerts and session telemetry. CC1 provides a 54-action discrete space comprising two environment-wide actions (Sleep, Monitor) and four host-targeted action types (Analyse, Remove, Restore, Misinform) applied across 13 hosts, where Misinform deploys decoy services intended to delay Red and expose attacker activity. CC2 provides a larger 145-action space in which Misinform is replaced by service-specific Decoy actions. All results are evaluated against B\_lineAgent and RedMeanderAgent [25].

All reported runs used Stable-Baselines3 PPO [26] with an MLP policy, learning rate $3 \times 1 0 ^ { - 4 } , n _ { \mathrm { s t e p s } } = 2 0 4 8 .$ , batch size 128, 6 PPO epochs per rollout, $\gamma = 0 . 9 9$ , clip range 0.2, and entropy coefficient 0.01, with synchronized train/evaluation normalization. As a reward-shaping control, we additionally evaluate an observation-based PBRS baseline [7], $r _ { t } ^ { \prime } = r _ { t } +$ $\lambda [ \gamma \Phi ( o _ { t + 1 } ) - \Phi ( o _ { t } ) ]$ , for $\lambda \ \in \ \{ 0 . 1 , 0 . 3 , 1 . 0 \}$ , where $\Phi ( o _ { t } )$ is the negative defender-visible threat score, constructed from compromise severity and host criticality such that more severe compromise of higher-criticality hosts yields lower potential. For both attacker settings, the shared Ask the Expert mechanism used $B _ { \mathrm { m a x } } = 2 0 0 , n _ { s } = 2 0 , W _ { \mathrm { w a r m } } = 2 0$ rollouts, a cadence of 10 rollouts per LLM query, and a fixed shaping bonus $\beta = 0 . 0 1$ with Gemini 3.1 Flash-Lite. Default host priors were weak Restore and Remove tiers with weights (0.1, 0.05); post-LLM host policies used weights $( 1 . 0 , 0 . 5 , 0 . 3 , 0 . 1 5 )$ for the primary, restore, remove, and analyse tiers, respectively. Prompt construction used a recent-failure frontier with host diversity capping, as well as a host-policy time-to-live after which stale advice reverted to defaults. In CC2, PPO operated in the full 145-action space and the LLM was restricted to a hostlevel Analyse/Remove/Restore remediation vocabulary; in CC1, PPO operated in the 54-action space and the LLM vocabulary additionally included Misinform. In both cases only the prompt doctrine was specialized to attacker structure. All reported evaluations were deterministic over 20 episodes.

```latex
Algorithm 1 Ask the Expert: LLM-Guided RL via Reward
Shaping
Require: Env ${ \overline { { \varepsilon } } } ,$ policy $\pi _ { \theta } ,$ value net $\overline { { V _ { \psi } } } .$ warmup W<sub>warm</sub>,
interval $K _ { \mathrm { u p d a t e } } ,$ bag size $B _ { \mathrm { m a x } } .$ , shaping parameter $\beta$
1: Init $\Pi _ { \mathrm { L L M } }  \mathrm { W }$ eak Default Tiers, $\mathcal { B }  \emptyset , \mathcal { C }  \emptyset , \mathcal { H }  \emptyset$
2: for iteration $n = 1 , 2 , \ldots$ do
3: for $t = 0$ to $T - 1$ do
4: Observe o<sub>t</sub>; $a _ { t } \sim \pi _ { \theta } ( \cdot | o _ { t } )$
5: Execute $a _ { t } ;$ observe $r _ { t } , ~ o _ { t + 1 }$
6: Encode reward-breakdown telemetry via $\phi$ into
compromised, spreading, and clean host sets
7: Construct semantic summary $\zeta _ { t }$ from the encoded
host sets
8: $S _ { t } \gets$ spreading hosts recovered by $\phi$
9: Store $\left( o _ { t } , a _ { t } , r _ { t } \right)$ in $\boldsymbol { B }$
10: if $S _ { t } \neq \emptyset$ then
11: $h _ { t } \gets$ first host in $S _ { t }$
12: Record $( t , h _ { t } )$ as a hard-state position for this
rollout
13: Add $\left( { \zeta _ { t } , a _ { t } , r _ { t } , \Delta r _ { t } , n } \right)$ to C
14: end if
15: end for
16: Mark outcome of previous LLM call in H using this
rollout’s mean reward
17: Expire stale host policies in $\Pi _ { \mathrm { L L M } }$ back to weak defaults
18: for each recorded hard-state position $( t , h _ { t } )$ in the
rollout do
19: if $a _ { t } ^ { B } \in \Pi _ { \mathrm { L L M } } ( h _ { t } )$ then
20: $\bar { \boldsymbol { r } } _ { t } \gets \boldsymbol { r } _ { t } + \boldsymbol { \beta } \cdot \boldsymbol { w } ( a _ { t } ^ { y } )$ in B
21: end if
22: end for
23: if $n \ge W _ { \mathrm { w a r m } }$ and $( n - W _ { \mathrm { w a r m } } )$ mod $K _ { \mathrm { u p d a t e } } = 0$
and $| { \mathcal { C } } | > 0$ then
24: entries ← SampleRecentFrontie $\cdot ( \mathcal { C } , n _ { s } )$
25: p ← BuildPrompt(entries, H)
26: $\mathcal { M } { \gets } \mathrm { V a l i d a t e } ( \mathrm { L L M } ( p ) )$
27: Π<sub>LLM</sub> ← Merge(Π<sub>LLM</sub>, ExpandTiered(M))
28: For spreading hosts from this rollout not addressed
by M, set $\Pi _ { \mathrm { L L M } } ( h ) $ full-weight Restore tier
29: Append validated suggestions to H
30: end if
31: Compute $\hat { A } _ { t }$ via GAE on the shaped rollout rewards;
update π<sub>θ</sub>, $V _ { \psi }$
32: B ← ∅
33: end for
```

## B. Performance Results

Shown in Figure 2 are the CC1 results. Against $\mathbb { B } _ { - } \bot .$ ineAgent, Ask the Expert improves early learning over

![](images/b02eb59c60b45d602beafb75515335a2b94abf5eaa6b88afacf802378d00193e.jpg)  
Fig. 2. Mean checkpoint evaluation reward for PPO, Ask the Expert, and PBRS variants in CC1. Top: B\_lineAgent. Bottom: RedMeanderAgent. Shaded bands denote ±1 standard error across independent training seeds.

PPO, reaching −86.7 vs. −122.9 at 50k steps and −59.6 vs. −80.5 at 75k. It also outperforms all evaluated PBRS variants at every reported checkpoint and finishes at −22.3, compared with −25.9 for PPO and −34.4 for the best PBRS mean at 300k. Against RedMeanderAgent, the method reaches −27.9 at 100k compared with −36.8 for PPO and −93.8 for the best PBRS mean at that checkpoint, and retains the best terminal mean (−9.7 vs. −27.7 for PPO and −18.0 for PBRS).

Figure 3 shows the same pattern on CC2. Against B\_lineAgent, Ask the Expert reaches −61.3 at 100k versus −88.1 for the best PBRS mean at that checkpoint and −89.7 for PPO, and finishes at −17.9 compared with −24.7 and −21.5, respectively. Against RedMeanderAgent, the method reaches −98.8 at 100k versus −132.7 for the best PBRS mean at that checkpoint and −139.9 for PPO; the gap narrows later, but Ask the Expert retains the best mean at 300k (−28.6 vs. −34.2 for PBRS and −30.4 for PPO). In all four settings, Ask the Expert achieves higher mean reward than every evaluated PBRS variant throughout the reported mean learning curves.

## C. Mechanistic Analysis of Frontier Sampling

Figure 4 shows another perspective on how the recentfrontier sampler behaves on one representative B\_lineAgent run. Panel (a) shows that the sampled prompt remains consistently more diverse than the raw rollout stream over the course of training. In other words, the sampler is exposing the LLM to a wider variety of hosts than the raw rollout frequencies would suggest. Panel (b) explains how this happens. Early in training, heavily represented hosts such as Op\_Server0 and Enterprise2 are deemphasized relative to their rollout frequency, while other hosts receive more prompt mass; later, the emphasis shifts as the active frontier moves. This figure is not intended as a separate performance result, but to illustrate that frontier sampling keeps successive LLM queries focused on a wider set of frontier-relevant states instead of repeatedly mirroring the most common host in the raw rollout stream.

![](images/256a56f46e341af36c3353729451350f99989e10eb4b3985deace123a8dc94d7.jpg)  
Fig. 3. Mean checkpoint evaluation reward for PPO, Ask the Expert, and PBRS variants in CC2. Top: B\_lineAgent. Bottom: RedMeanderAgent. Shaded bands denote ±1 standard error across independent training seeds.

![](images/4c79633cb037d13641d77bd4db7d92ecbb2cd9fdfa04fceb5eb42783d1948fd9.jpg)  
Fig. 4. Frontier-sampling mechanics from one representative B\_lineAgent run. Panel (a) compares the normalized Shannon entropy [27] of host frequencies in the raw rollout stream and in the sampled prompt over training. Panel (b) shows the host-wise reweighting induced by the sampler, measured as prompt share minus rollout share at each query. Positive values indicate that a host is emphasized more in the prompt than in the rollout stream.

## VI. DISCUSSION

Across CC1 and CC2, Ask the Expert attains higher mean reward than every evaluated PBRS variant at each reported checkpoint for both attacker types, while also improving substantially over PPO during early and intermediate training. The separation is especially pronounced in CC1 and against RedMeanderAgent; in CC2, the PBRS gap narrows as training progresses, but Ask the Expert retains the strongest terminal mean in both attacker settings. These results show that the observed gains are not reproduced by the evaluated observationbased potential shaping baselines, while deployment remains fully policy-based.

The frontier-sampling diagnostic (Figure 4) validates that prompt construction continues to expose the LLM to diverse frontier-relevant states rather than collapsing onto the most frequent host in the raw rollout stream, confirming that the diversity benefit of the sampler persists throughout training.

Ask the Expert’s results set it as a viable asymmetrictraining approach for self-defense systems: semantic reasoning is provided only where it can do the most good, at training time, while deployment remains lightweight and fully policy-based.

## VII. CONCLUSION

We introduced Ask the Expert, a training-time LLM-guided reward-shaping framework for autonomous cyber defense. The framework converts defender-visible compromise signals into semantic summaries, intermittently queries an LLM, and distills constrained host-level advice into tiered reward bonuses while leaving deployment as a pure RL policy. On CAGE CC1 and CC2 under two attacker types, the method improves sample efficiency over PPO and achieves higher mean reward than every evaluated PBRS variant throughout the reported learning curves. Although the gap to PBRS narrows during late training in some settings, Ask the Expert retains the strongest terminal mean in all four settings.

## REFERENCES

[1] C. Smiliotopoulos, G. Kambourakis, and C. Kolias, “Detecting lateral movement: A systematic survey,” Heliyon, vol. 10, no. 4, 2024.

[2] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[3] T. T. Nguyen and V. J. Reddi, “Deep reinforcement learning for cyber security,” IEEE Transactions on Neural Networks and Learning Systems, vol. 34, no. 8, pp. 3779–3795, 2021.

[4] W. Hu, X. Liao et al., “A survey on reinforcement learning for cyber security,” IEEE Communications Surveys & Tutorials, vol. 23, no. 2, pp. 1353–1381, 2021.

[5] E. Bates, V. Mavroudis, and C. Hicks, “Reward shaping for happier autonomous cyber security agents,” in Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security, 2023, pp. 221–232.

[6] G. Palmer, C. Parry, D. J. Harrold, and C. Willis, “Deep reinforcement learning for autonomous cyber defence: A survey,” arXiv preprint arXiv:2310.07745, 2023.

[7] A. Y. Ng, D. Harada, and S. Russell, “Policy invariance under reward transformations: Theory and application to reward shaping,” in Proceedings of the Sixteenth International Conference on Machine Learning, 1999, pp. 278–287.

[8] H. Xu, S. Wang, N. Li, K. Wang, Y. Zhao, K. Chen, T. Yu, Y. Liu, and H. Wang, “Large language models for cyber security: A systematic literature review,” ACM Trans. Softw. Eng. Methodol., Sep. 2025, just Accepted. [Online]. Available: https://doi.org/10.1145/3769676

[9] T. Purves, K. G. Kyriakopoulos, S. Jenkins, I. Phillips, and T. Dudman, “Causally aware reinforcement learning agents for autonomous cyber defence,” Knowledge-Based Systems, vol. 304, p. 112521, 2024.

[10] Y. Zhang et al., “Large language models for cyber security: A systematic review,” IEEE Security & Privacy, 2023.

[11] Y. J. Ma, W. Liang, G. Wang, D.-A. Huang, O. Bastani, D. Jayaraman, Y. Zhu, L. Fan, and A. Anandkumar, “Eureka: Human-level reward design via coding large language models,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview.net/forum?id=IEduRUO55F

[12] Y. Du, O. Watkins, Z. Wang, C. Colas, T. Darrell, P. Abbeel, A. Gupta, and J. Andreas, “Guiding pretraining in reinforcement learning with large language models,” in Proceedings of the 40th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, 23–29 Jul 2023, pp. 8657–8677. [Online]. Available: https://proceedings.mlr.press/v202/du23f.html

[13] F. Martinez, T. Li, Y. Lu, and J. Chen, “Stackelberg coupling of online representation learning and reinforcement learning,” in The Fourteenth International Conference on Learning Representations, 2026.

[14] F. Martinez-Lopez, T. Li, Y. Lu, and J. Chen, “In-context reinforcement learning via communicative world models,” arXiv preprint arXiv:2508.06659, 2025.

[15] K. Hammar, T. Li, R. Stadler, and Q. Zhu, “Adaptive security response strategies through conjectural online learning,” IEEE Transactions on Information Forensics and Security, vol. 20, pp. 4055–4070, 2025.

[16] J. Standen et al., “Cyborg: A cyber operations research gym for autonomous agents,” Proceedings of the AAAI Conference on Artificial Intelligence, 2021.

[17] T. Schaul, J. Quan, I. Antonoglou, and D. Silver, “Prioritized experience replay,” in International Conference on Learning Representations, 2016. [Online]. Available: https://dblp.org/rec/journals/corr/SchaulQAS15

[18] S. Griffith, K. Subramanian, J. Scholz, C. L. Isbell, and A. L. Thomaz, “Policy shaping: Integrating human feedback with reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 26, 2013. [Online]. Available: http://hdl.handle.net/1853/53270

[19] Y. Wang, X. Wu, J. Farooq, T. Li, and J. Chen, “Collaborative cloudedge computing via llm-guided constrained reinforcement learning,” TechRxiv, 2026. [Online]. Available: https://www.techrxiv.org/doi/full/10. 36227/techrxiv.177155994.44918685/v1

[20] T. Xie, S. Zhao, C. H. Wu, Y. Liu, Q. Luo, V. Zhong, Y. Yang, and T. Yu, “Text2reward: Reward shaping with language models for reinforcement learning,” in The Twelfth International Conference on Learning Representations, 2024. [Online]. Available: https://openreview. net/forum?id=tUM39YTRxH

[21] X. Zhang, N. Gao, X. Jiang, Y. Chen, Y. Pan, M. Zhang, and Y. Deng, “Progress reward model for reinforcement learning via large language models,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. [Online]. Available: https://openreview.net/forum?id=TJhHb6CscW

[22] S. Sun, R. Liu, J. Lyu, J.-W. Yang, L. Zhang, and X. Li, “A large language model-driven reward design framework via dynamic feedback for reinforcement learning,” Knowledge-Based Systems, 2025. [Online]. Available: https://www.sciencedirect.com/science/article/ pii/S0950705125011104

[23] Q. Li, B. J. Pang, Y. Song, H. Fu, Q. Xu, X. Yuan, X. Xu, and C. Zhang, “Large language model assisted hierarchical reinforcement learning training,” Information Sciences, 2026. [Online]. Available: https://colab.ws/articles/10.1016%2Fj.ins.2025.122688

[24] I. Homoliak, F. Toffalini, J. Guarnizo, Y. Elovici, and M. Ochoa, “Insight into insiders and it: A survey of insider threat taxonomies, analysis, modeling, and countermeasures,” ACM Computing Surveys, vol. 52, no. 2, p. 1–40, Apr. 2019. [Online]. Available: http: //dx.doi.org/10.1145/3303771

[25] M. Kiely, D. Bowman, M. Standen, and C. Moir, “On autonomous agents in a cyber defence environment,” arXiv preprint arXiv:2309.07388, 2023.

[26] A. Raffin, A. Hill, A. Gleave, A. Kanervisto, M. Ernestus, and N. Dormann, “Stable-baselines3: Reliable reinforcement learning implementations,” Journal of Machine Learning Research, vol. 22, no. 268, pp. 1–8, 2021. [Online]. Available: http://jmlr.org/papers/v22/20-1364.html

[27] C. E. Shannon, “A mathematical theory of communication,” Bell System Technical Journal, vol. 27, no. 3–4, pp. 379–423, 623–656, 1948.