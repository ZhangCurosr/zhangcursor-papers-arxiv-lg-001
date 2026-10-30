# DYNAS: A Dynamic Network Penetration Testing Framework Using Deep Hierarchical Reinforcement Learning

Md. Rakibul Hassan<sup>1</sup>, Majedul Islam<sup>1</sup>, Md. Ashraful Alam<sup>1</sup>, Muhammad Iqbal Hossain<sup>1</sup>, and Swakkhar Shatabda<sup>1</sup>

Accepted at the NSS 2026: 20th International Conference on Network and System Security (submission version)

BRAC University, Bangladesh

md.rakibul.hassan@g.bracu.ac.bd, majedul.islam@g.bracu.ac.bd, md.ashraful.alam1@g.bracu.ac.bd, iqbal.hossain@bracu.ac.bd, swakkhar.shatabda@bracu.ac.bd

Abstract. As network scale and uncertainty complicate automated penetration testing, reinforcement learning (RL) environments must support controlled evaluation under defensive and non-stationary conditions. We present DYNAS (Dynamic Network Attack Simulator), which combines parameterized large-scale scenario generation with stateful intrusion detection responses, explicit scan false-positive and false-negative noise, temporary service unavailability, and duration-based action timeouts. We further introduce Pen-DHRL, a recurrent manager–worker policy that selects persistent subgoals using IDS-conditioned global context. We compare Pen-DHRL with GNN, Invariant, and Attention baselines across four scenario variants and multiple training regimes. Under IDS-disabled training followed by IDS-enabled evaluation, Pen-DHRL obtains approximately 64% and 24% higher reward per step on Bridge and Dynamic, respectively, and approximately 26% and 30% more captured hosts on VarA and VarB than the next-best baseline. With all defensive dynamics active from the beginning of training, Pen-DHRL achieves positive mean episode returns in all four scenarios, whereas the flat baselines produce negative mean returns in all 12 architecture–scenario combinations; staged curriculum training produces positive mean returns across all 16 evaluated combinations.

Keywords: Automated penetration testing · AI for security · Hierarchical reinforcement learning · IDS-aware attack planning · Network attack simulation · Curriculum learning

## 1 Introduction

Automated security assessment provides repeatability and broad coverage, but automatically generated findings can require expert validation, whereas fully manual penetration testing is resource-intensive and dificult to scale [22,24]. This complementarity motivates learning-based methods for sequential attack planning, but their evaluation depends strongly on the assumptions represented by the training environment. Established

RL environments such as NASim and NASimEmu provide eficient interaction, partially observable attack planning, and scenario variation, yet their published simulation protocols simplify stateful defensive reactions and several sources of observation and execution uncertainty [25,13]. DYNAS targets this evaluation gap through a controlled, defense-aware simulator for large parameterized network scenarios.<sup>1</sup>

NASimEmu was selected as the implementation base because its variable-size graph observations, parameterized scenario representation, and released RL baselines align directly with our questions concerning structural transfer and hierarchical control. Its common agent interface also permits policies to be evaluated across changing network instances while preserving a comparable action vocabulary. Together, these properties provide a technically suitable and reproducible foundation for DYNAS [13].

Table 1. Abstraction-level and environmental-feature comparison of RL-based pentesting frameworks
<table><tr><td>Feature</td><td>NASim IAPTF</td><td></td><td>WCE</td><td></td><td>DynPen NASimEmu PenGym DYNAS</td><td></td><td></td></tr><tr><td>Simulation-Based Training</td><td>√</td><td>√</td><td>Conceptual</td><td>V</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Structural Scenario Variation</td><td>Limited</td><td></td><td>Conceptual</td><td>√</td><td>Limited</td><td>Limited</td><td>√</td></tr><tr><td>Firewall-Aware Reachability</td><td>√</td><td></td><td>Limited Conceptual</td><td></td><td>√</td><td>√</td><td>√</td></tr><tr><td>IDS Representation/ Response</td><td></td><td>Limited</td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>Curriculum Scheduling</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>Explicit Scan FP/FN Noise</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr><tr><td>Service-State Dynamics</td><td></td><td></td><td></td><td>Limited</td><td></td><td></td><td>√</td></tr><tr><td>Duration-Based Action Timeouts</td><td></td><td></td><td></td><td></td><td></td><td></td><td>√</td></tr></table>

Per NASim [25], IAPTF [11], WCE [6,7], DynPen [15], NASimEmu [13], and PenGym [19]. ✓ denotes explicit support in the cited implementation; Limited denotes related support at a diferent abstraction or granularity; Conceptual denotes a proposed framework or NASim-based case study; and – indicates that the feature is not documented in the cited protocol.

NASimEmu already supports scenario randomization for evaluating transfer across network sizes and configurations. DYNAS extends this simulation protocol by jointly introducing stateful IDS responses, explicit scan errors, temporary service-state changes, duration-based action timeouts, and curriculum-controlled dificulty. The reviewed systems provide complementary subsets of these capabilities, but their published protocols do not document their joint use within one repeatable attack-focused simulator. Pen-DHRL is designed for this setting by combining recurrent global context, persistent high-level subgoals, and a goal-conditioned worker policy. Against this background, the primary contributions of this research are as follows:

– Controlled Defense-Dynamics Layer: A configurable simulation layer that jointly represents stateful IDS responses, explicit scan false-positive and false-negative noise, temporary service unavailability and recovery, and duration-based action timeouts under firewall-constrained reachability.

– Parameterized Scenario Generation: A template-based generator that varies subnet sizes, host configurations, sensitive-host placement, and topology while preserving a stable action vocabulary.

– Hierarchical Policy: A recurrent manager uses IDS context to select persistent subgoals that guide attack planning over long horizons.

– Regime-Specific Evaluation: A comparison with flat RL baselines under defensedisabled training, immediate defended training, curriculum training, and transfer from IDS-disabled training to IDS-enabled evaluation.

## 2 Motivation and Background

DYNAS and Pen-DHRL are shaped by three requirements.

R1: defensive reaction must be stateful (memory across steps, not a per-action penalty or scripted opponent) and configurable. DYNAS abstracts reaction into four mechanisms: stateful intrusion detection, triaged into isolation, patching, or heavier monitoring; scan false positives/negatives that corrupt observations; service churn, which can vanish a service on return; and action timeouts, where failure need not mean error. Sect. 4 details this layer.

R2: the agent must observe its own footprint. A penalty with no observable correlate makes stealth unlearnable, not merely dificult. DYNAS models an assumed-breach engagement and exposes each host’s accumulated detection level, alerting threshold, and monitoring multiplier, so the agent can read its exact distance to a response rather than only a vague sense of noise; authorized engagements routinely grant this level of access, precisely so that an assessment can measure detection as well as compromise. The same reasoning gates host-internal actions, which require prior user-level access rather than remote capability. This motivates the IDS-conditioned context of Pen-DHRL (Sect. 4.2).

R3: exposure to defence must be staged. With all four mechanisms active from episode one, a randomly initialized policy accumulates heavy penalty before reaching any sensitive host, favouring early termination over exploration—the regime in which every flat baseline we evaluate records negative reward per step, recovering only once training is staged (Table 6). This motivates the three-stage curriculum of Sect. 4.1, ordered easy to hard [4].

DYNAS addresses R1 and R3; Pen-DHRL is the policy designed for R2.

Preliminaries. Penetration testing is a partially observable Markov decision process: the observation and action set grow as subnets are discovered [25], motivating the graphstructured, permutation-equivariant policies used throughout [13]. Credit assigned over a single timescale dilutes across a long trajectory, since a sensitive host pays of only after discovery, exploitation, and privilege escalation all succeed; we address this with the feudal formulation [9,27], where a manager emits a subgoal at a coarse timescale and a worker conditions on it. Subgoals may be semantic or latent; Pen-DHRL adopts the latter, so the decomposition is discovered from reward rather than imposed by hand.

## 3 Related Work

Existing RL-based penetration-testing environments emphasize complementary research problems. NASim introduced an eficient simulator for learning attack paths but identified scalability and environment-fidelity as continuing challenges [25]. IAPTF applied hierarchical decomposition to improve planning over large networks and represented security-aware clustering involving firewalls and IDSs [11]. WCE proposed a broader continuous cyber-testing research framework and illustrated it through a NASim-based case study [6,7]. DynPen addressed changing host attributes and network connectivity [15], while NASimEmu emphasized scenario variation and transfer to unseen network configurations [13]. PenGym subsequently demonstrated the importance of testing learned policies under system-level action execution [19]. Complementary eforts include automated red-team and post-exploitation methods [10,21], deep reinforcement learning for IoT security [23], and analyses of the shift from manual to automated testing [14,18]; broader studies further motivate integrating learning-based and automated approaches [1,2,3].

A parallel line of work builds environments for autonomous cyber operations more broadly. CyberBattleSim models lateral movement, typically over a fixed, hand-specified topology, and includes a defender that detects and evicts the attacker [17], acting as a scripted opponent rather than a parameterized detection layer. CybORG provides a gym for autonomous cyber agents [26] and underpins the CAGE challenges [8], which are oriented toward training defensive agents. DYNAS is attack-focused, like CyberBattleSim, but exposes its defensive mechanisms as configurable, curriculumschedulable parameters rather than a fixed opponent.

A further line of work evaluates policies on real infrastructure. Cyber ranges such as CyRIS [20] and KYPO [28] instantiate real hosts for security training. Such testbeds ofer execution fidelity, while simulation ofers the large, repeatable interaction budget and independent control over defensive mechanisms that a controlled evaluation framework requires. DYNAS is positioned deliberately on the simulation side.

Within learning-based attack planning, Hu et al. combine network-structure analysis with a Deep Q-Network to derive attack paths [12], and Bu et al. apply a dual-stage DRL model to multi-subnet environments [5]. Closest to this work, Li et al. incorporate expert prior knowledge into a hierarchical DRL model for penetration testing [16]; the underlying manager–worker feudal formulation was established by Vezhnevets et al. [27]. Pen-DHRL instead conditions its manager on an accumulated, per-host detection signal, so subgoal selection responds to the network’s defensive state rather than to attacker-side progress alone.

These frameworks address distinct aspects of scalability, scenario variation, execution, and defensive uncertainty; Table 1 summarizes their complementary coverage.

DYNAS integrates these capabilities within one controlled evaluation framework, while Pen-DHRL provides the corresponding hierarchical policy.

## 4 Framework and Methodology

![](images/4491709dfdadf9aa0dc4128ac86acb6872400ea5c3ce28f6279528b4eb9e17d5.jpg)  
Fig. 1. Overview of Dynamic Scenario Generation and HRL Agent Development. Part 1 (left, detailed in Part 1 below) is the scenario pipeline, from the cached template through the NASimEmu runtime. Part 2 (right, detailed in Part 2 below) is the Pen-DHRL agent, from graph encoding through the value and termination heads. Solid arrows carry data and decisions during an episode; dashed arrows carry gradients during the recurrent PPO update.

DYNAS extends NASimEmu’s simulation layer to generate larger parameterized scenarios and expose agents to controlled defensive and non-stationary efects. Our contribution extends the simulation side of NASimEmu. As shown in Fig. 1, the methodology has two connected parts.

Part 1: Dynamic Scenario Generation (1a–1d). (1a) A template YAML defines fixed primitives (OS, services, processes, exploits, and privilege-escalation rules) and is cached. (1b) An auto-generator samples subnet sizes, topology style, host configurations, sensitive-host placement, and firewall constraints to create large dynamic scenarios. (1c) Environment-dynamics parameters are configured through scenario or curriculum settings (when enabled), including IDS responses, scan noise, temporary service-state changes, and duration-based action timeouts. (1d) The NASimEmu runtime executes state transitions, rewards, and constraints under these settings.

Part 2: Pen-DHRL Agent Pipeline (2a–2f). (2a) The agent receives graph observations with node/subnet features and positional encoding. (2b) A recurrent manager GNN builds global context. (2c) The manager selects persistent subgoals from a goal bank via a learned switch policy. (2d) A goal-conditioned worker outputs node-action decisions. (2e) Value and termination heads estimate return and stopping. (2f) Parameters are optimized with recurrent PPO replay, and updates are fed back to both manager and worker.

We evaluate policy behavior across dynamically generated training and test instances and across the four defined scenario configurations using reward per step, episode return, episode length, and captured-host metrics. The experiments include defense-disabled, IDS-enabled curriculum, and defended-transfer conditions.

## 4.1 Dynamic Scenario Generation

The dynamic scenario generator of DYNAS extends the simulation layer of NASimEmu by combining a cached template layer with a parameterized auto-generator and a curriculum-controlled environment-dynamics stage. The template layer preserves a fixed action signature and observation dimensions across all generated instances, while the auto-generator resamples structural and parametric components per episode or per epoch.

Extension to Dynamic Scenarios DYNAS generates scenarios from a YAML template that caches fixed primitives, including OS types, service and process probability maps, exploit definitions, and privilege escalation rules. This preserves an identical action signature across instances. The generator samples subnet counts and sizes from configurable ranges, selects a topology style (mesh, chain, or random), and instantiates each host by drawing its OS, services, and processes from the template probability maps. Sensitive host placement follows configurable host probabilities. Firewall rules are derived from the chosen connectivity style, and the curriculum or scenario settings configure IDS responses, scan noise, temporary service changes, and action timeouts at runtime.

Domain Randomization To diversify training conditions, the pipeline independently varies exploit and privilege escalation success probabilities, action costs, and scan costs. Feature dropout additionally zeros observed service and process bits to simulate partial observability during reconnaissance. Template caching and action signature verification keep the exploit and privilege escalation space consistent across generated scenarios, preventing mismatches during batched rollouts. Curriculum injection activates environment parameters by stage.

Curriculum Schedule Training uses three curriculum stages defined over the training horizon. The baseline stage covers the first 30% of training and disables IDS responses, scan noise, service churn, and action timeouts. The medium stage covers the next 30% and jointly introduces low-intensity settings for all four mechanisms. The final stage covers the remaining 40% and increases their configured dificulty. In the final stage, scan false-positive and false-negative rates range from 1.5% to 7% depending on scan type, the timeout probability is 1.5%, the service-churn probability is 1%, and IDS thresholds are sampled between 0.65 and 0.88. Evaluation uses the final stage, providing a consistent high-dificulty condition for policy comparison. The staged parameterization supports reproducible analysis of policy adaptation as defensive pressure and execution uncertainty increase.

## 4.2 Pen-DHRL Model Design

As shown in Fig. 2, Pen-DHRL is implemented as a simulation-first pipeline in DYNAS, combining controlled dynamic scenarios with hierarchical graph control. Relative to the original NASimEmu, the DYNAS simulation layer configures IDS responses, scan noise, temporary service-state changes, and duration-based action timeouts through scenario or curriculum settings, while the agent layer performs recurrent manager-worker decision making over graph observations.

Main Processing Pipeline Given a graph-observation batch, the recurrent manager produces node features and global context, to which it adds the graph-wise mean of the host IDS fields observable to the agent: the accumulated detection level, alerting threshold, and monitoring multiplier (Sect. 2). The baseline represents its subgoals as � = 8 learned latent slots in a goal bank $\mathbf { Q } \in \mathbb { R } ^ { 8 \times 1 2 8 }$ . A learned switch and a counter initialized to � = 4 govern slot refresh, after which the selected goal is refined by the current context and supplied to the worker.

Side and Control Mechanisms Pen-DHRL persists the latent-slot index and vector, counter, manager hidden state, batch map, and episode counters. Reset, masked-reset, and clone operations support recurrent training. The recurrent PPO update replays the recorded worker action, stopping decision, and active goal index from the initial state with done masks, forms bootstrapped returns, and jointly updates both controllers.

## 4.3 Training and Evaluation Protocol

For the defended comparisons, all policies were optimized using a common vectorized PPO training loop. Each optimization iteration collected an eight-step rollout from 128 parallel environments, yielding 1,024 transitions, followed by three clipped PPO passes.

1. Main Processing Pipeline  
2. Side & Control  
![](images/f4da57fb41312d82e1ba74ac2f185c4f60d7c6b0aa9fb832b5828d9806a118b9.jpg)  
Fig. 2. Pen-DHRL Architecture Implemented in DYNAS. Left: the forward path for one decision. Node features and positional encodings are embedded and passed through the recurrent manager GNN to obtain per-node features and a graph-level global context; the graph-wise mean of the per-host IDS fields is projected and added to that context. The manager either keeps its active goal or draws a new slot from the goal bank, refines it with the global context, and passes the result to a GAT worker that emits a (host, action) decision. Value and termination heads read the concatenated global context, refined goal, and normalized episode progress. Right: the state carried across steps, and the recurrent PPO update, which snapshots that state, replays the recorded decisions with done-masked resets, and applies the clipped policy, value, and entropy losses.

The full training schedule comprised 200 reporting intervals of 100 PPO iterations, corresponding to 20,000 optimization iterations. PPO used a discount factor of 0.99, a clipping parameter of 0.2, and gradient-norm clipping at 3.0. Parameters were optimized using AdamW with an initial learning rate of $7 \times 1 0 ^ { - 4 }$ , weight decay of $1 0 ^ { - \bar { 4 } }$ , and an initial entropy coeficient of 0.02. Bootstrap targets were computed using a target network updated by soft parameter averaging with coeficient $\rho = 0 . 1$ . Pen-DHRL additionally preserved its recurrent manager state during rollout and replay, as described in the preceding model-design subsection.

At each episode reset, a scenario was selected uniformly from the configured training pool and regenerated by resampling subnet sizes, sensitive-host placement, and host OS, service, and process configurations while preserving a common action vocabulary. Curriculum dificulty was updated once per reporting interval. Evaluation was performed without parameter updates using 64 fresh parallel environments, with recurrent state reset at episode boundaries. Curriculum-enabled evaluation environments always used the final configured defense-dynamics stage. Each evaluation pass used a target of 100 completed episodes and a 400-step interaction limit. Episodes terminated when the policy selected its terminal action or reached this limit, and action generation followed each architecture’s implemented inference rule. Episode-level metrics were normalized by the number of completed episodes, while reward per step was normalized by the number of executed interactions.

## 5 Experimental Evaluation

## 5.1 Evaluation Metrics and Interaction Scale

Evaluation Metrics Rewards follow the implemented scaled transition rule $r = ( v -$ � 10, using scenario-defined host values and action costs. We evaluate each policy using four complementary metrics: reward per step measures mean utility per interaction, mean episode return measures cumulative utility, episode length reflects interaction eficiency, and the mean number of distinct sensitive hosts captured measures attack coverage separately from return. All metrics follow the protocol in Sect. 4.3.

Here � is non-zero only when an action first raises access to root on a sensitive host or when an intrusion-detection response fires, and � is the scenario-defined cost of the executed action. The divisor of ten is a fixed global scaling, applied to all reported returns and penalties.

Interaction-Budget Reference To characterize the interaction scale of each scenario, consider a complete-network sweep requiring one scan per subnet, two reconnaissance actions per host, one process scan per sensitive host, and expected exploit and privilegeescalation retries. Assuming success probabilities of approximately 0.8, the resulting heuristic action budget is

$$
A _ { \mathrm { h e u r i s t i c } } = S + 2 H + 3 . 5 N .\tag{1}
$$

Here, �, �, and � denote the numbers of subnets, hosts, and sensitive hosts, respectively. The heuristic combines subnet discovery, host reconnaissance, and expected compromise attempts into a common interaction-scale reference across scenarios.

Table 2 lists the scenario dimensions and heuristic action budget alongside the observed capture and episode-return values in the defense-disabled setting.

Table 2. Scenario dimensions, heuristic interaction budget, and observed Pen-DHRL outcomes in the defense-disabled setting
<table><tr><td>Scenario</td><td>(N, H, S)</td><td>Aheuristic</td><td></td><td>Captured hosts Episode return</td></tr><tr><td></td><td>Corp-Dyn (39, 83, 12)</td><td>314.5</td><td>36.13</td><td>49.41</td></tr><tr><td>VarA</td><td>(41, 86, 12)</td><td>327.5</td><td>36.68</td><td>48.43</td></tr></table>

## 5.2 Scenario Characterization

Reference and Dynamic Scenarios Tables 3 and 4 contrast original NASimEmu configurations with DYNAS dynamic configurations. The original configurations are corp and uni; the dynamic configurations are corp\_dynamic and corp\_dynamic\_varA. The reference configurations establish the original scenario scale, while the dynamic configurations introduce larger network instances and broader host, service, process, and exploit catalogs.

– corp.v2: Enterprise baseline with 8 subnets, 21 hosts, and 6 services.

– uni.v2: Academic-like topology with 9 subnets, 27 hosts, and 6 services.

– corp\_dynamic.v2: Expanded network (12 subnets, 83 hosts) with 32 services, 13 processes, and 8 exploits across Linux/Windows stacks.

– corp\_dynamic\_varA: Variant of corp\_dynamic with lower exploit probabilities and higher costs, producing stronger stochastic behavior.

Service, Process, and Structural Comparison. Relative to the reference configurations, the dynamic scenarios contain more hosts, services, processes, exploits, and sensitive hosts (Tables 3 and 4), increasing both graph size and the range of attack-relevant configurations encountered during policy learning.

Table 3. Service and process distribution across network scenarios
<table><tr><td>Scenario</td><td>LinSvc WinSvc AnySvc SensSvc LinProc WinProc UserExp</td><td></td><td></td><td></td><td></td><td></td><td></td><td>RootExp</td><td>PrivEsc</td></tr><tr><td>Corp</td><td>3</td><td>2</td><td>1</td><td>1</td><td>0</td><td>0</td><td>4</td><td>1</td><td>1</td></tr><tr><td>Uni</td><td>3</td><td>2</td><td>1</td><td>1</td><td>0</td><td>0</td><td>4</td><td>1</td><td>1</td></tr><tr><td>Corp-Dyn</td><td>23</td><td>8</td><td>1</td><td>3</td><td>6</td><td>7</td><td>7</td><td>1</td><td>6</td></tr><tr><td>VarA</td><td>23</td><td>8</td><td>1</td><td>3</td><td>6</td><td>7</td><td>8</td><td>0</td><td>6</td></tr></table>

Table 4. Structural and security characteristics of network scenarios
<table><tr><td>Scenario</td><td>Subnets Hosts Exploits SensHosts Density</td><td></td><td></td><td></td></tr><tr><td>Corp</td><td>8</td><td>21 5</td><td>7</td><td>0.25</td></tr><tr><td>Uni</td><td>9</td><td>27 5</td><td>3</td><td>0.27</td></tr><tr><td>Corp-Dyn</td><td>12</td><td>83 8</td><td>39</td><td>0.19</td></tr><tr><td>VarA</td><td>12</td><td>86 8</td><td>41</td><td>0.19</td></tr></table>

## 5.3 RL Architecture Comparison

Dynamic Baseline The original NASimEmu baselines include GNN, Attention, and Invariant agents, which provide useful comparisons for flat policy learning across training and test scenarios. With defenses disabled, Table 5 shows that Pen-DHRL obtains the highest reward per step and mean episode return. GNN achieves slightly greater host capture, while Invariant produces shorter episodes. The results therefore indicate improved return under hierarchical control while retaining a measurable eficiency and coverage trade-of.

Table 5. Baseline performance on dynamically generated scenarios without IDS or curriculum
<table><tr><td>Metric</td><td>Architecture Training</td><td></td><td>Testing</td></tr><tr><td rowspan="4">Reward/step</td><td>GNN</td><td>0.27</td><td>0.27</td></tr><tr><td>Invariant</td><td>0.34</td><td>0.33</td></tr><tr><td>Attention</td><td>0.26</td><td>0.26</td></tr><tr><td>Pen-DHRL</td><td>0.49</td><td>0.48</td></tr><tr><td rowspan="4">Episode return GNN</td><td></td><td>29.90</td><td>30.02</td></tr><tr><td>Invariant</td><td>27.21</td><td>27.06</td></tr><tr><td>Attention</td><td>28.21</td><td>28.09</td></tr><tr><td>Pen-DHRL</td><td>49.41</td><td>48.43</td></tr><tr><td rowspan="4">Episode length GNN</td><td></td><td>111.50</td><td>112.63</td></tr><tr><td>Invariant</td><td>81.10</td><td>80.96</td></tr><tr><td>Attention</td><td>109.90</td><td>109.42</td></tr><tr><td>Pen-DHRL</td><td>103.40</td><td>106.40</td></tr><tr><td rowspan="4">Captured hosts GNN</td><td></td><td>36.91</td><td>37.10</td></tr><tr><td>Invariant</td><td>32.23</td><td>32.13</td></tr><tr><td>Attention</td><td>35.63</td><td>35.46</td></tr><tr><td>Pen-DHRL</td><td>36.13</td><td>36.68</td></tr></table>

Defended Evaluation Scenarios The defended evaluation uses four complementary configurations with a shared action vocabulary. Dynamic provides the reference largenetwork setting; Bridge interpolates host, service, process, exploit, and action-cost parameters; VarA combines lower exploit probabilities, higher costs, and a redistributed sensitive-host profile; and VarB applies both a topology change and a further parameter shift. Together, these configurations form a structured progression from interpolation to joint structural–parametric transfer while retaining comparable action semantics.

IDS-Enabled Training: Curriculum vs. No-Curriculum Tables 6 and 7 compare all four architectures under IDS-enabled evaluation on Dynamic, Bridge, VarA, and VarB, reporting each metric under no-curriculum and curriculum training together with the change Δ (curriculum minus no-curriculum). In training without a curriculum, all four mechanisms are active from the outset. In that regime, the flat baselines produce negative reward per step, capture few hosts, and frequently reach the step limit; Pen-DHRL remains positive. The curriculum comprises three stages: the baseline stage disables these mechanisms, the medium stage jointly introduces them at low intensity, and the final stage increases their configured dificulty. Across this comparison, each flat baseline moves to positive reward per step and gains approximately 9–28 captured hosts, while Pen-DHRL improves more modestly because its no-curriculum performance is already positive. Table 8 evaluates a distinct transfer regime in which IDS is disabled during training and enabled during evaluation. In this regime, Pen-DHRL obtains the highest reward per step on Bridge and Dynamic and the highest capture on VarA and VarB; Attention produces the shortest episodes on VarA and VarB but captures fewer than one host on average.

Table 6. IDS-enabled evaluation of reward per step and episode return: no-curriculum vs. curriculum. Episode returns are rounded to integers; each Δ uses unrounded values
<table><tr><td rowspan="2">Scenario</td><td rowspan="2">Model</td><td colspan="3">Reward/step</td><td colspan="3">Episode return</td></tr><tr><td>No</td><td>Curr</td><td>Δ</td><td>No</td><td>Curr</td><td>Δ</td></tr><tr><td rowspan="4">Bridge</td><td>Pen-DHRL</td><td>0.40</td><td>1.51</td><td>+1.11</td><td>161</td><td>213</td><td>+53</td></tr><tr><td>GNN</td><td>-0.08</td><td>1.37</td><td>+1.45</td><td>-31</td><td>189</td><td>+220</td></tr><tr><td>Invariant</td><td>-1.30</td><td>1.22</td><td>+2.52</td><td>-511</td><td>206</td><td>+717</td></tr><tr><td>Attention</td><td>-1.23</td><td>1.24</td><td>+2.47</td><td>-490</td><td>205</td><td>+695</td></tr><tr><td rowspan="4">Dynamic</td><td>Pen-DHRL</td><td>0.54</td><td>2.25</td><td>+1.71</td><td>217</td><td>249</td><td>+32</td></tr><tr><td>GNN</td><td>-0.08</td><td>2.17</td><td>+2.25</td><td>-30</td><td>254</td><td>+284</td></tr><tr><td>Invariant</td><td>-1.41</td><td>1.89</td><td>+3.30</td><td>-560</td><td>263</td><td>+822</td></tr><tr><td>Attention</td><td>-1.31</td><td>2.01</td><td>+3.32</td><td>-523</td><td>253</td><td>+776</td></tr><tr><td rowspan="4">VarA</td><td>Pen-DHRL</td><td>0.17</td><td>0.61</td><td>+0.44</td><td>68</td><td>134</td><td>+66</td></tr><tr><td>GNN</td><td>-0.10</td><td>0.64</td><td>+0.74</td><td>-41</td><td>139</td><td>+180</td></tr><tr><td>Invariant</td><td>-1.29</td><td>0.63</td><td>+1.92</td><td>-514</td><td>140</td><td>+654</td></tr><tr><td>Attention</td><td>-1.20</td><td>0.67</td><td>+1.87</td><td>-479</td><td>134</td><td>+613</td></tr><tr><td rowspan="4">VarB</td><td>Pen-DHRL</td><td>0.08</td><td>0.38</td><td>+0.30</td><td>31</td><td>83</td><td>+52</td></tr><tr><td>GNN</td><td>-0.06</td><td>0.40</td><td>+0.46</td><td>-28</td><td></td><td> $8 8 \ + 1 1 5$ </td></tr><tr><td>Invariant</td><td>-1.18</td><td>0.40</td><td>+1.58</td><td>-458</td><td></td><td> $9 8 + 5 5 6$ </td></tr><tr><td>Attention</td><td>-0.56</td><td>0.19</td><td>+0.75</td><td>-224</td><td></td><td> $4 8 + 2 7 1$ </td></tr></table>

The Δ columns report the curriculum-associated change for each metric. Every flat baseline’s reward per step changes from negative to positive under the curriculum; for example, Invariant on Dynamic increases by 3.30, from 1.41 to 1.89, whereas Pen-DHRL, already positive, records smaller gains. Episode-return changes reach 822, and episode length decreases by approximately 140–290 steps for the flat baselines. Captured-host counts increase by approximately 9–28 for the flat baselines, while the corresponding Pen-DHRL changes remain below six.

## 6 Discussion

The results show that the training regime materially changes model ranking. When all defensive dynamics are active from the beginning of training, Pen-DHRL remains positive across all four scenarios, while the flat baselines obtain negative returns and very low capture. Curriculum scheduling improves every architecture and makes the flat baselines competitive, indicating that progressive exposure is particularly important for non-hierarchical policies.

Table 7. IDS-enabled evaluation of episode length and captured hosts: no-curriculum vs. curriculum. Episode lengths are rounded to integers; each Δ uses unrounded values
<table><tr><td colspan="2"></td><td colspan="3">Episode length</td><td colspan="3">Captured hosts</td></tr><tr><td rowspan="2">Scenario Bridge</td><td>Model</td><td>No</td><td>Curr</td><td>Δ</td><td>No</td><td>Curr</td><td>Δ</td></tr><tr><td>Pen-DHRL</td><td>400</td><td>141</td><td>-259</td><td>24.95</td><td>25.72</td><td>+0.77</td></tr><tr><td rowspan="4"></td><td>GNN</td><td>400</td><td>138</td><td>-262</td><td>0.99</td><td>23.68</td><td>+22.69</td></tr><tr><td>Invariant</td><td>392</td><td>170</td><td>-222</td><td>2.24</td><td>25.56</td><td>+23.32</td></tr><tr><td>Attention</td><td>400</td><td>165</td><td>-235</td><td>1.07</td><td>23.72</td><td>+22.65</td></tr><tr><td>Pen-DHRL</td><td>400</td><td>113</td><td>-287</td><td>29.59</td><td>30.37</td><td>+0.78</td></tr><tr><td rowspan="4"></td><td>GNN</td><td>400</td><td>117</td><td>-283</td><td>1.00</td><td>28.49</td><td>+27.49</td></tr><tr><td>Invariant</td><td>397</td><td>139</td><td>-258</td><td>2.15</td><td>29.80</td><td>+27.65</td></tr><tr><td>Attention</td><td>400</td><td>126</td><td>-274</td><td>1.06</td><td>27.44</td><td>+26.38</td></tr><tr><td>Pen-DHRL</td><td>400</td><td>238</td><td>-162</td><td>13.73</td><td>19.62</td><td>+5.89</td></tr><tr><td rowspan="4">VarA</td><td>GNN</td><td>400</td><td>218</td><td>-182</td><td>0.00</td><td>18.46</td><td>+18.46</td></tr><tr><td>Invariant</td><td>397</td><td>223</td><td>-175</td><td>0.00</td><td>18.13</td><td>+18.13</td></tr><tr><td>Attention</td><td>400</td><td>200</td><td>-200</td><td>0.00</td><td>17.29</td><td></td></tr><tr><td>Pen-DHRL</td><td>400</td><td>249</td><td></td><td></td><td></td><td>+17.29</td></tr><tr><td rowspan="4">VarB</td><td></td><td></td><td></td><td>-151</td><td>9.39</td><td>14.93</td><td>+5.54</td></tr><tr><td>GNN</td><td>400</td><td>220</td><td>-180</td><td>0.00</td><td>13.83</td><td> $+ 1 3 . 8 3$ </td></tr><tr><td>Invariant</td><td>387</td><td>243</td><td>-144</td><td>0.00</td><td>14.52</td><td> $+ 1 4 . 5 2$ </td></tr><tr><td>Attention</td><td>400</td><td>255</td><td>-145</td><td>0.00</td><td>9.48</td><td>+9.48</td></tr></table>

Table 8. Per-scenario model performance with IDS disabled during training but enabled during evaluation
<table><tr><td></td><td></td><td>Scenario Architecture Reward/step Episode return Ep Len Capture</td><td></td><td></td></tr><tr><td rowspan="3">Bridge</td><td>Pen-DHRL</td><td>1.15</td><td>161.96141.20</td><td>23.36</td></tr><tr><td>GNN</td><td>0.18</td><td>66.20368.86</td><td>26.17</td></tr><tr><td>Invariant Attention</td><td>0.01 0.70</td><td>5.20367.77 120.04170.39</td><td>25.69 17.58</td></tr><tr><td rowspan="4"></td><td>Dynamic Pen-DHRL</td><td>1.49</td><td>200.17 134.56</td><td>27.80</td></tr><tr><td>GNN</td><td>0.26</td><td>91.32 356.31</td><td>30.59</td></tr><tr><td>Invariant</td><td>0.19</td><td>66.58348.37</td><td>30.88</td></tr><tr><td>Attention</td><td>1.20</td><td>204.54170.28</td><td>26.87</td></tr><tr><td rowspan="4">VarA</td><td>Pen-DHRL</td><td>0.14</td><td>46.42 323.19</td><td>16.53</td></tr><tr><td>GNN</td><td>0.38</td><td>62.45162.44</td><td>10.53</td></tr><tr><td>Invariant</td><td>0.14</td><td>46.08327.06</td><td>13.17</td></tr><tr><td>Attention</td><td>-0.11</td><td>-8.50 74.20</td><td>0.34</td></tr><tr><td rowspan="4">VarB</td><td>Pen-DHRL</td><td>0.11</td><td>28.66 266.58</td><td>11.42</td></tr><tr><td>GNN</td><td>0.22</td><td>36.52169.74</td><td>8.26</td></tr><tr><td>Invariant</td><td>0.08</td><td>19.89264.46</td><td>8.79</td></tr><tr><td>Attention</td><td>-0.13</td><td>-5.24 39.64</td><td>0.10</td></tr></table>

With curriculum training, Pen-DHRL obtains the highest reward per step on Bridge and Dynamic by approximately 10% and 4%, respectively. On VarA and VarB, Pen-DHRL retains the highest capture, while the flat models obtain slightly higher reward per step. Under IDS-disabled training followed by IDS-enabled evaluation, Pen-DHRL’s advantage is larger: it provides approximately 64% and 24% higher reward per step on Bridge and Dynamic and approximately 26% and 30% greater capture on VarA and VarB. Together, these results identify defended transfer as the regime in which Pen-DHRL provides its strongest reward–coverage profile.

The evaluation assesses Pen-DHRL as an integrated hierarchical architecture in which recurrent global context, IDS conditioning, and persistent subgoals jointly coordinate long-horizon decisions. The resulting policy behaviour is consistent with the stronger reward–coverage profile observed under defended transfer.

## 7 Conclusion and Future Work

This work presents DYNAS, a Dynamic Network Attack Simulator that combines parameterized large-network generation with stateful IDS responses, explicit scan noise, temporary service-state changes, duration-based action timeouts, and curriculum-controlled dificulty. Within this environment, Pen-DHRL integrates recurrent global context, persistent high-level subgoals, and an IDS-conditioned worker policy. The evaluation shows that Pen-DHRL provides its clearest advantage when transferring from defense-disabled training to defended evaluation, while curriculum scheduling substantially improves all evaluated architectures and narrows the performance gap. These findings support hierarchical control as a strong approach for reward–coverage balance under defended and non-stationary conditions. Future work will extend DYNAS across additional topology families and integrate active defender policies and host-level firewall rules.

## References

1. Alhamed, M., Rahman, M.M.H.: A systematic literature review on penetration testing in networks: Future research directions. Applied Sciences 13(12), 6986 (2023).

2. Ashawa, M., Kpelai, S.J.: Penetration testing: Analysis of emerging technologies and their impact on pen testing. International Journal on Engineering Technologies and Informatics 4(4), 1–4 (2023).

3. Badarneh, H.J., Attiany, L.A., Asassfeh, M., Al-Shaikh, A., Afaneh, S., Shquier, M.M.A., et al.: The power of network penetration testing. In: 25th International Arab Conference on Information Technology (ACIT). pp. 1–6. IEEE (2024)

4. Bengio, Y., Louradour, J., Collobert, R., Weston, J.: Curriculum learning. In: Proceedings of the 26th International Conference on Machine Learning. pp. 41–48 (2009)

5. Bu, H., Wen, H., Zhu, H., Li, H., Song, X., Ren, Y.: Automated penetration on multi-subnet environments with dual-stage drl models. In: 2025 IEEE International Conference on Systems, Man, and Cybernetics (SMC). pp. 5054–5059. IEEE, Vienna, Austria (2025).

6. Cody, T., Beling, P., Freeman, L.: Towards continuous cyber testing with reinforcement learning for whole campaign emulation. In: IEEE AUTOTESTCON. pp. 1–5. IEEE (2022).

7. Cody, T., Meno, E., Beling, P., Freeman, L.: Whole campaign emulation with reinforcement learning for cyber test. IEEE Instrumentation & Measurement Magazine 26(5), 25–30 (2023).

8. Cyber Autonomy Gym for Experimentation: Cage challenge 2. https://github.com/ cage-challenge/cage-challenge-2 (2022)

9. Dayan, P., Hinton, G.E.: Feudal reinforcement learning. In: Advances in Neural Information Processing Systems 5. pp. 271–278 (1992)

10. Filiol, E., Mercaldo, F., Santone, A.: A method for automatic penetration testing and mitigation: A Red Hat approach. Procedia Computer Science 192, 2039–2046 (2021).

11. Ghanem, M.C., Chen, T.M., Nepomuceno, E.G.: Hierarchical reinforcement learning for eficient and efective automated penetration testing of large networks. Journal of Intelligent Information Systems 60(2), 281–303 (2023).

12. Hu, Z., Beuran, R., Tan, Y.: Automated penetration testing using deep reinforcement learning. In: 2020 IEEE European Symposium on Security and Privacy Workshops (EuroS&PW). pp. 2–10. IEEE (2020).

13. Janisch, J., Pevný, T., Lisý, V.: NASimEmu: Network attack simulator & emulator for training agents generalizing to novel scenarios. In: Computer Security. ESORICS 2023 International Workshops. LNCS, vol. 14399, pp. 589–608. Springer, Cham (2024).

14. Kongara, D., Krishnama, S.: A process of penetration testing using various tools. Mesopotamian Journal of Cybersecurity 2023, 93–103 (2023).

15. Li, Q., Wang, R., Li, D., Shi, F., Zhang, M., Chattopadhyay, A., et al.: DynPen: Automated penetration testing in dynamic network scenarios using deep reinforcement learning. IEEE Transactions on Information Forensics and Security 19, 8966–8981 (2024).

16. Li, Q., Zhang, M., Shen, Y., Wang, R., Hu, M., Li, Y., et al.: A hierarchical deep reinforcement learning model with expert prior knowledge for intelligent penetration testing. Computers & Security 132, 103358 (2023)

17. Microsoft Defender Research Team: Cyberbattlesim. https://github.com/microsoft/ CyberBattleSim (2021)

18. Moreno, A.C., Hernandez-Suarez, A., Sanchez-Perez, G., Toscano-Medina, L.K., Perez Meana, H., Portillo-Portillo, J., et al.: Analysis of autonomous penetration testing through reinforcement learning and recommender systems. Sensors 25(1), 211 (2025).

19. Nguyen, H.P.T., Hasegawa, K., Fukushima, K., Beuran, R.: PenGym: Realistic training environment for reinforcement learning pentesting agents. Computers & Security 148, 104140 (2025).

20. Pham, C., Tang, D., Chinen, K.i., Beuran, R.: Cyris: A cyber range instantiation system for facilitating security training. In: Proceedings of the 7th International Symposium on Information and Communication Technology (SoICT). pp. 251–258 (2016)

21. Pham, V.H., Do Hoang, H., Trung, P.T., Quoc, V.D., To, T.N., Duy, P.T.: Raiju: Reinforcement learning-guided post-exploitation for automating security assessment of network systems. Computer Networks 253, 110706 (2024).

22. Rane, N., Qureshi, A.: Comparative analysis of automated scanning and manual penetration testing for enhanced cybersecurity. In: 12th International Symposium on Digital Forensics and Security (ISDFS). pp. 1–6. IEEE (2024).

23. Samad, A., Altaf, S., Arshad, M.J.: Advancements in automated penetration testing for IoT security by leveraging reinforcement learning. International Journal of Computer Science and Telecommunications 15(1), 29–34 (2024)

24. Scarfone, K., Souppaya, M., Cody, A., Orebaugh, A.: Technical guide to information security testing and assessment. Tech. Rep. NIST Special Publication 800-115, National Institute of Standards and Technology (2008)

25. Schwartz, J., Kurniawati, H.: Autonomous penetration testing using reinforcement learning. arXiv preprint arXiv:1905.05965 (2019)

26. Standen, M., Lucas, M., Bowman, D., Richer, T.J., Kim, J., Marriott, D.: Cyborg: A gym for the development of autonomous cyber agents (2021), arXiv:2108.09118

27. Vezhnevets, A.S., Osindero, S., Schaul, T., Heess, N., Jaderberg, M., Silver, D., et al.: Feudal networks for hierarchical reinforcement learning. In: Proceedings of the 34th International Conference on Machine Learning. PMLR, vol. 70, pp. 3540–3549 (2017)

28. Vykopal, J., Oslejsek, R., Celeda, P., Vizvary, M., Tovarnak, D.: Kypo cyber range: Design and use cases. In: Proceedings of the 12th International Conference on Software Technologies (ICSOFT). pp. 310–321 (2017)