# EvoSignal: LLM-Guided Evolutionary Design of Modular Traffic Signal Control Programs

Leizhen Wang<sup>1</sup> Peibo Duan<sup>1,∗</sup> Zhenlin Qin<sup>2</sup> Yancheng Ling<sup>2</sup> Jian Xu<sup>3</sup> Yue Wang<sup>4</sup> Hao Wang<sup>5</sup> Zhenliang Ma<sup>2,∗</sup>

<sup>1</sup>Monash University <sup>2</sup>KTH Royal Institute of Technology <sup>3</sup>Chinese Academy of Sciences <sup>4</sup>Zhejiang Dahua Technology <sup>5</sup>Southeast University <sup>∗</sup>Corresponding authors

## Abstract

Effective traffic signal control (TSC) requires policies that respond to changing traffic demand and network conditions while meeting different control objectives. However, adapting existing strategies often involves repeated manual design and adjustment, making it difficult to systematically explore better control rules for a target network. Large language models (LLMs) can automate this process, but directly using them to select signal phases leaves decision rules embedded in black-box models and incurs recurring inference costs and latency. This paper formulates TSC as a modular program design problem and proposes EvoSignal, an LLM-guided evolutionary framework using traffic knowledge and performance feedback. The modular representation separates traffic feature extraction, local phase prioritization, and optional network-based priority adjustment. Starting from several established strategies, EvoSignal improves programs through feedback on congestion and signal operation, retaining strategies with different performance trade-offs. The resulting programs operate without online LLM inference. Simulation experiments across five scenarios on two realworld road networks show that the selected default EvoSignal program reduces waiting time by 16.8–49.2% relative to the lowest waiting time achieved by the 20 conventional, reinforcement learning-based, and LLMbased baselines in each scenario. A program prioritizing travel time and queue length outperforms all 20 baselines on all three metrics in the search scenario and remains among the top three on each metric when transferred unchanged to the other four scenarios. These findings support automated design of inspectable control programs that transfer across the evaluated road networks and traffic demands. Code is available at https://github.com/georgewanglz2019/EvoSignal.

## 1 Introduction

Traffic signal control (TSC) is an important component of urban traffic management. By selecting signal phases that serve competing traffic movements, a controller can reduce vehicle queues and waiting times at intersections. Effective control must account for changing traffic demand and interactions between intersections: releasing vehicles from one intersection may increase congestion on downstream roads and affect traffic conditions elsewhere in the network. These characteristics make the design of effective control policies challenging, particularly when road networks, traffic demand, and control objectives differ across scenarios [1, 2].

Existing TSC methods include conventional traffic engineering approaches and reinforcement learning (RL)-based approaches. Conventional methods, such as fixed-time control, adaptive signal control, and max-pressure control, provide well-established principles for determining signal settings [3–5]. Their control rules can respond to traffic conditions, but selecting and extending these rules for a target network still requires engineering effort. RL-based methods learn policies through interactions with a traffic environment and have achieved strong performance in signal coordination and network-wide control [6–8]. However, training requires interaction data and computation, and neural policies can be difficult to inspect and modify when control requirements change [2]. Programmatic policies offer explicit decision rules that can be examined and adjusted [9], motivating automated design methods that retain this form of control.

Recent studies use large language models (LLMs) to bring traffic knowledge and reasoning into signal control. LLMLight and Traffic-R1 analyze traffic conditions and select signal phases through LLM inference [10, 11]. This places language generation within the repeated control loop, adding inference latency and computational demand across intersections. Traffic-specialized models, including LightGPT and Traffic-R1, also require data collection and model training, although smaller models and efficient inference can mitigate deployment costs [10, 11]. Their language-based rationales help explain individual actions, but the complete decision rule remains encoded in model parameters and cannot be directly inspected and edited as a control program. These considerations motivate using LLM capabilities to improve controller design while keeping online phase selection computationally simple and explicit.

An alternative is to use an LLM to generate and revise a control program offline. The resulting program selects phases directly from traffic observations, removing repeated LLM inference from operation and exposing the implemented decision rules for inspection. Program revision can also adapt the controller without updating the LLM’s parameters. LLM-guided program evolution provides a mechanism for this process: candidate code is generated, evaluated, and improved using feedback. FunSearch, AlphaEvolve, OpenEvolve, and ShinkaEvolve establish this approach for program optimization [12–15]. Their general search mechanisms leave the traffic control representation and domain-specific evaluation guidance to the application designer. In TSC, the earlier EvolveSignal study explores fixed-time strategy design at a single intersection [16], while SignalClaw evolves interpretable skills using structured traffic feedback and eventbased composition [17]. Building on these advances, an important question is how to organize the evolution of state-responsive control programs across a road network, using existing strategies and observed traffic behavior to meet different control objectives.

This design problem involves four connected challenges. First, control decisions depend on local queues and approaching vehicles as well as downstream and network conditions. Evolving an undifferentiated program makes it harder to revise these functions separately or match them to available observations. Second, a single initial program represents only one control strategy, leaving other established approaches to pressure, demand, and service fairness unused at initialization. Third, aggregate performance scores rank programs but do not identify where queues accumulate or which phase-selection patterns contribute to long waiting times. Effective revision requires diagnostic feedback that connects observed traffic behavior to control logic. Finally, travel time, queue length, and waiting time can favor different strategies. Selecting only by a combined score can overlook useful alternatives and their reusable control ideas. These challenges call for a traffic-specific organization of program structure, initialization, diagnostic feedback, and strategy reuse within the evolutionary process.

We formulate TSC as a modular program design problem: an executable policy maps traffic observations to signal phase decisions, and its control logic is optimized for specified performance objectives. To address this problem, we propose EvoSignal, an LLM-guided evolutionary framework. Its three-module representa tion (3M) separates traffic feature extraction, local phase prioritization, and optional network-based priority adjustment, allowing control logic to reflect the available traffic information. Multi-strategy initialization (Multi-init) supplies programs based on established control strategies and complementary design principles. The simulation performance analyzer (SPA) diagnoses congestion and phase-selection behavior and suggests program revisions, while a Pareto strategy archive (PSA) supplies experience from recently evaluated strategies with different performance trade-offs. The LLM uses these inputs to revise candidate programs, whose measured control performance guides further evolution. Feedback consists of traffic observations and performance measurements; in this study, these are obtained through simulation. The selected program subsequently determines signal phases directly from traffic observations.

The main contributions are summarized as follows:

1. We formulate TSC as a modular program design problem, where an executable policy maps traffic observations to signal phase decisions under specified control objectives. The representation connects traffic feature extraction, local phase prioritization, and optional network-based priority adjustment through explicit interfaces, enabling targeted revision and inspection of the control logic.

2. We develop EvoSignal, an evolutionary framework that combines multiple initial control strategies, diagnostic feedback on traffic control performance, and reuse of strategies with different objective tradeoffs to guide LLM-based program revision.

3. We evaluate EvoSignal across five scenarios on two real-world road networks against conventional, RL-based, and LLM-based controllers, and LLM-guided program evolution frameworks. The results demonstrate competitive control performance and more consistent search outcomes. A program evolved in one scenario transfers unchanged to the other four scenarios, ranking among the top three on travel time, queue length, and waiting time against 20 reference controllers.

The remainder of this paper is organized as follows. Section 2 reviews related work. Section 3 formulates the modular program design problem and presents the EvoSignal search framework. Section 4 describes the experimental setup and evaluates control performance, search behavior, and component contributions. Section 5 analyzes the evolved programs and discusses their implications and limitations. Section 6 concludes the paper.

## 2 Related Work

We review four groups of methods: conventional TSC, RL-based TSC, LLM-based TSC, and LLM-guided program design and evolution. The discussion follows how these methods construct and execute control policies, emphasizing their ability to respond to traffic conditions, the transparency of their decision rules, and their computational requirements.

## 2.1 Conventional Traffic Signal Control

Conventional TSC methods determine signal settings using traffic engineering models and predefined control rules [1]. Fixed-time control allocates green time according to estimated demand; Webster’s formulation provides a classical basis for cycle length and green-split design [3]. Adaptive systems such as SCOOT adjust signal settings using measured traffic conditions [4]. Their implementation must account for phase structure, detector availability, and timing constraints [18].

Pressure-based control uses queue information to determine which movements to serve. Max-pressure control selects phases according to upstream–downstream queue differentials and provides network stability guarantees under its modeling assumptions [5]. Capacity-aware backpressure accounts for limited downstream storage [19], while cyclical max-pressure control incorporates an operational phase structure [20]. Distributed adaptive control also uses shockwave-based delay estimates and downstream queues to guide phase activation [21]. Conventional methods offer explicit rules, modest computational requirements, and, for certain pressure-based policies, theoretical stability guarantees. Adaptive and pressure-based controllers already respond to changing traffic conditions, but their responses remain constrained by predefined control rules. Extending these rules to capture more complex traffic interactions or changing objectives requires further engineering effort. This motivates learning control policies from traffic experience.

## 2.2 Reinforcement Learning-Based Traffic Signal Control

RL learns decision policies through interaction with an environment, with transportation applications in cluding traffic assignment [22, 23], and adaptive signal control [24, 25]. For TSC, this addresses the rule design limitation by learning how signal actions should depend on traffic conditions. Early work investigates adaptive intersection control [26], multi-agent coordination [27], and integrated control over large road networks through MARLIN-ATSC [28]. Deep RL extends these approaches by learning policies from richer traffic observations [29, 30].

A central design choice is how to represent traffic demand and compare competing phases. FRAP models phase competition [31], PressLight incorporates pressure into policy learning [6], and MPLight supports shared decentralized control across intersections [8]. AttendLight uses lane and phase attention to accommodate different intersection configurations [32]. Efficient-CoLight uses efficient pressure to represent traf fic movements [33], while Advanced-CoLight also considers approaching vehicles [34]. TransformerLight captures temporal traffic patterns through sequence modeling [35]. Control objectives extend beyond vehicle efficiency to person-based waiting costs across transport modes [36], while action designs include joint phase selection and duration optimization [37] and cycle-level optimization of cycle length and green splits [38].

Network coordination and adaptation introduce further challenges. Attention mechanisms support network coordination through graph-based modeling of neighboring intersections [7] and the selection of relevant intersection information for centralized critics [39]. Multi-agent actor–critic control [40] and knowledge sharing in KS-DDPG [41] support cooperation, while MetaLight reuses experience to adapt to new scenarios [42], and adversarial training improves robustness to environmental mismatch in integrated expressway and intersection control [43]. Reference-guided action selection also incorporates established timing policies to improve early training performance [44]. Programmatically interpretable RL [45] and π-Light [9] further address policy transparency using explicit programs. Overall, RL expands the ability to learn trafficresponsive behavior and coordination, but requires interaction data and optimization. For neural controllers, inspecting or modifying a particular decision rule remains difficult. Generalization beyond the training conditions is also a continuing challenge, and unfamiliar networks or traffic patterns may require additional adaptation [2, 24]. These limitations motivate incorporating broader traffic knowledge and more accessible explanations into control decisions.

## 2.3 LLM-Based Traffic Signal Control

LLMs are increasingly applied to transportation prediction, information analysis, and decision support [46]. These applications use traffic context and travel histories for mobility prediction [47, 48] and day-to-day route choice [49], or connect language interfaces to traffic analysis tools [50]. In TSC, pretrained knowledge and language reasoning support the interpretation of traffic conditions and signal decisions. Here, the LLM acts as the decision maker: traffic observations are expressed in a prompt, and model inference determines the signal action. LLMLight applies this approach to phase selection and develops LightGPT through traffic-specific training [10]. Its generalist models and LightGPT variants are included in our experiments. The GPT, Llama, and Qwen families provide representative backbones [51–54]. More generally, chain-ofthought prompting [55] and self-consistency [56] study how language models can reason through decisions and combine multiple reasoning paths.

Domain adaptation has been investigated for transferable mobility prediction [57], information extraction from crash narratives [58], and traffic-specific phase decisions [10, 11]. Within TSC, methods also extend the information and learning mechanisms available to the LLM decision maker. LLM-Assisted Light places the LLM at the center of control and supplies perception and decision tools [59]. Traffic-R1 trains traffic reasoning through RL and supports communication between intersections [11]. These developments improve the use of traffic knowledge, support transfer to unfamiliar conditions, and make individual decisions easier to discuss in language. Nevertheless, repeated model inference introduces computational cost and latency into online control. Compact models reduce this burden [11], but resource requirements remain a deployment consideration. A verbal rationale alone does not expose the complete neural decision rule or establish that it faithfully accounts for the selected action, limiting direct inspection and modification. These considerations motivate using LLMs to design executable control programs offline, so that online phase selection can use explicit code without repeated language-model inference.

## 2.4 LLM-Guided Program Design and Evolution

This approach separates the use of LLM reasoning from online control execution. Code language models can generate executable programs from task descriptions [60], and AlphaCode demonstrates large-scale generation and filtering of candidate solutions [61]. LLM-guided program evolution further combines code generation with repeated performance evaluation. FunSearch retains and improves promising programs using computational feedback [12], while AlphaEvolve develops an evolutionary coding workflow around automated evaluators [13]. OpenEvolve provides the open-source infrastructure used in this study [14]. ShinkaEvolve improves search through parent sampling, novelty-based filtering, and adaptive model selection, while retaining metrics and textual feedback for subsequent revisions [15].

Related approaches differ in the programs they produce and the feedback they use. Evolution of Heuristics jointly improves descriptions and code for optimization rules [62], while Eureka generates reward functions for training RL policies [63]. Voyager stores executable skills and refines them using environmental feedback [64]. Self-Refine iterates between feedback and revision [65], Reflexion retains verbal reflections from task outcomes [66], and ReEvo incorporates reflection into LLM-based evolutionary search [67]. These studies motivate using both measured performance and textual feedback to guide code improvement without updating the LLM’s parameters.

For traffic signals, the earlier EvolveSignal study applies a coding agent to fixed-time strategy design at a single intersection [16]. SignalClaw evolves interpretable traffic skills using structured feedback and event based composition [17]. These approaches demonstrate the value of evaluating and revising executable policies. However, general program-evolution frameworks leave the organization of traffic control logic and domain-specific guidance to the application designer. Traffic-specific studies address particular control settings and skill structures, leaving room to investigate how local decisions and network conditions can be combined with multiple existing strategies and different control objectives. EvoSignal addresses this design question through modular control programs, multiple initial strategies, traffic diagnostics, and reuse of strategies with complementary performance trade-offs. The resulting rules can be inspected and executed without online LLM inference, while their effectiveness and transfer still require evaluation across the intended traffic conditions.

## 3 Methodology

## 3.1 TSC as a Modular Program Design Problem

We formulate TSC policy design as the search for an executable program that maps traffic observations to signal phase decisions. We first define the control task and phase-priority-based decision rule, then specify the modular program representation and its performance objective.

## 3.1.1 Traffic observations and phase decisions

Consider a road network with signalized intersections. At decision step t, intersection i selects a phase from its admissible phase set P . A phase serves a group of non-conflicting traffic movements, following the phase-selection setting used in LLMLight [10]. The input $x _ { i , t }$ includes the current phase, elapsed phase time, vehicle counts and queues by traffic movement, and available downstream traffic information. It can also include decision history and information shared by other intersections.

An executable control program p maps this input to the selected phase $a _ { i , t }$

$$
a _ { i , t } = p ( x _ { i , t } ) \in P _ { i } .\tag{1}
$$

The same program runs at different intersections using their respective observations and state. Its code remains fixed during control evaluation, while its decisions respond to changing traffic conditions.

We adopt phase-priority-based control: at each decision step, the program computes a priority $s _ { k } ( x _ { i , t } )$ for each admissible phase k and selects

$$
a _ { i , t } \in \arg \operatorname* { m a x } _ { k \in P _ { i } } s _ { k } ( x _ { i , t } ) .\tag{2}
$$

Ties are resolved by the first phase in the configured ordering. Selecting the current phase again extends service to the same movements; selecting another phase invokes the environment’s signal-transition procedure. During a transition, the controller preserves its previous action. Section 4.1 specifies the phase configuration and timing settings used in the experiments.

This decision structure connects traffic observations to explicit phase preferences. Recomputing priorities allows the service order to respond to changing demand. For automated program design, the structure makes competing phase preferences available for inspection and permits the scoring logic to be revised while retaining a simple selection operation. It also accommodates pressure, approaching demand, and service history within a common control interface. EvoSignal organizes these computations through a three-module representation (3M).

## 3.1.2 3M: Modular control program representation

Traffic signal control requires a controller to interpret observations, prioritize competing movements, and consider surrounding traffic conditions. The 3M representation separates these tasks into traffic feature extraction (M1), local phase prioritization (M2), and network-based priority adjustment (M3). A candi date program contains the three functions $F _ { 1 } , F _ { 2 } , F _ { 3 }$ ; their input–output interfaces are specified, while their internal computations are revised during evolution.

Figure 1 shows execution at decision step t. Each of the I intersections runs the same program using its own observations and state; the selected phases $( a _ { 1 , t } , \ldots , a _ { I , t } )$ jointly control the traffic environment, whose next observations continue the loop. Program code stays fixed during execution; Figure 2 shows how it is revised during design.

![](images/f33f11fc9f5550d4d8a976448bde606df631aaa0b9956b00f52fdb38802110cb.jpg)  
Figure 1. Modular control program at decision step t. One controller is instantiated per intersection, with controller i expanded; the selected phases jointly control the traffic environment.

For one intersection at one decision step, omit the intersection and time subscripts and let x denote the available observations, history, and shared state. The main computations are

$$
( z , c ) = F _ { 1 } ( x ) ,\tag{3}
$$

$$
q = F _ { 2 } ( z ) ,\tag{4}
$$

$$
\begin{array} { r } { r = F _ { 3 } ( c , z ) . } \end{array}\tag{5}
$$

Here z contains features for phase prioritization, c contains additional traffic context, and $q , r \in \mathbb { R } ^ { | P | }$ give the base scores and adjustments for the admissible phases in P. Figure 1 omits t for readability: $x _ { i } , a _ { i }$ denote $x _ { i , t } , a _ { i , t }$ , while its remaining variables follow these interfaces and Eq. (6). These equations specify the module interfaces. The functions may also use controller history and shared state; the features and scoring expressions within each function can evolve.

M1: Traffic feature extraction M1 transforms movement-level observations into features for phase selection, such as pressure, waiting and approaching demand, downstream congestion, and service history. These form z, while c can include adjacency information, shared features, and network summaries. Evolution can revise the feature definitions and combinations. Appendix F.1 gives a concrete pressure–demand example.

M2: Local phase prioritization M2 assigns a base score $q _ { k }$ to each phase using the extracted features. Evolution can revise coefficients and introduce nonlinear, conditional, or history-dependent rules; the interface does not restrict candidates to a fixed scoring expression.

M3: Network-based priority adjustment M3 computes $r _ { k }$ from the traffic context and extracted features. Its role is to adjust a local phase preference in response to conditions beyond the current intersection. A gate g controls the contribution of M3. The final phase priorities are

$$
s _ { k } = \mathrm { s a f e } ( q _ { k } + g r _ { k } ) , \qquad k \in P ,\tag{6}
$$

where safe clips scores to the configured range and replaces nonfinite values before selection. The evolved program may adjust g. At non-transition decision steps, these priorities determine the phase through Eq. (2). Together, the three functions and the phase-selection operation implement the executable policy in Eq. (1).

The three stages make the information requirements and decision components explicit. A controller can use features supported by its available observations and include an additional network-based adjustment when appropriate. The generated code determines its actual dependencies. Section 5.1 compares two selected programs, with detailed feature extraction and priority rules in Appendix C.

## 3.1.3 Program design objective

Let Π denote the class of executable control programs that follow the 3M interfaces and return admissible phase decisions. We evaluate control performance using travel time (TT), queue length (QL), and waiting time (WT), all of which are minimized. For a given road network, traffic demand, and evaluation horizon, the performance of a program $p \in \Pi$ is represented by

$$
\mathbf { f } ( p ) = \left( f _ { T } ( p ) , f _ { Q } ( p ) , f _ { W } ( p ) \right) = ( \mathrm { T T } , \mathrm { Q L } , \mathrm { W T } ) ,\tag{7}
$$

TT averages the accumulated entry-to-exit durations in recorded vehicle trajectories. QL is the time average of the total number of queued vehicles on network approaches. WT is the time average of the mean elapsed duration of active stopping episodes at each sample, with zero assigned to an empty sample.

To express the control objectives during program design, we use the bounded scalar score

$$
S ( p ) = \sum _ { j \in \{ T , Q , W \} } w _ { j } \frac { 1 } { 1 + f _ { j } ( p ) / C _ { j } } ,\tag{8}
$$

where $C _ { j } > 0$ is a metric-specific scale. The weights satisfy $w _ { j } \geq 0$ and $\begin{array} { r } { \sum _ { j } w _ { j } = 1 } \end{array}$ and remain fixed within a search run. A larger score indicates better performance under the chosen weights. Section 4.1 specifies the experimental weights, and Appendix F lists the metric scales.

The modular program design problem is

$$
\operatorname* { m a x } _ { p \in \Pi } S ( p ) .\tag{9}
$$

The design variables are the executable implementations of $F _ { 1 } , F _ { 2 } , F _ { 3 }$ , including their feature computations, scoring expressions, conditional logic, and numerical coefficients. The module interfaces and the phaseselection operation remain specified. The priority $s _ { k }$ determines which phase to serve at one decision step, whereas $S ( p )$ evaluates the traffic consequences of running the program over the evaluation horizon. Improving a program therefore requires evaluating its successive decisions across the network. Each search uses a specified scenario and fixed objective weights; reuse under other networks and demands is evaluated separately.

## 3.2 EvoSignal: LLM-Guided Evolutionary Search

EvoSignal addresses Eq. (9) through the revision–evaluation loop in Figure 2. Multi-init supplies initial programs; the program database selects parents and references; and the LLM revises control logic within the 3M interfaces. The traffic evaluator measures each candidate’s performance, SPA diagnoses its traffic behavior, and PSA supplies summaries of strategies with different performance trade-offs. These records guide subsequent revisions. EvoSignal builds on the program database and code-editing mechanisms of OpenEvolve v0.2.26 [14], a community-developed open-source implementation of AlphaEvolve [13]. In this study, traffic feedback is obtained through simulation.

![](images/122926414d0a248dfc3dd2f59712893011e97db1dfb851e37376abf07c112600.jpg)  
Figure 2. EvoSignal framework for evolving modular control programs from multiple initial strategies, guided by traffic diagnostics and reusable strategy summaries. The selected program controls signals without online LLM inference.

## 3.2.1 Evolution procedure

Algorithm 1 summarizes the procedure. The database D stores code, parent identities, execution status, and evaluation records. A successful program has record $\boldsymbol { m } = ( \mathbf { f } , S , d , b )$ : traffic metrics, combined score, SPA feedback, and a saved PSA summary. The inherited database uses islands, elite retention, and exploratory sampling to select a parent $p _ { a }$ and reference programs $R _ { n }$ . The guidance $G _ { n }$ specifies the task, interfaces, and current revision emphasis. The prompt constructor H combines these inputs, and the fixed-parameter LLM distribution $P _ { \theta }$ generates edits. PSA memory M retains the latest L strategy records. Appendix F.2 gives storage and update details.

Let $V _ { N }$ contain all valid evaluated programs from initialization and N evolution attempts. EvoSignal returns the best observed program:

$$
p ^ { \star } \in \arg \operatorname* { m a x } _ { p \in V _ { N } } S ( p ) .\tag{10}
$$

Selection uses the source scenario and does not assume a global optimum over $\Pi ;$ the selected code remains fixed during control execution.

Algorithm 1 EvoSignal evolutionary design procedure   
Require: Initial programs $p ^ { ( 1 ) } , \ldots , p ^ { ( K ) } ;$ ; traffic scenario; weights w ; guidance G; budget N   
Ensure: Selected executable program $p ^ { \star }$   
1: Initialize database D and PSA memory M   
2: Evaluate initial programs; store their code, metrics, scores, and status in D   
3: Summarize successful initial programs and update M   
4: for $n = 1 , \ldots , N$ do   
5: Select parent $p _ { a }$ with record $m _ { a }$ and references $R _ { n }$ from D   
6: Construct prompt $u _ { n } = H ( p _ { a } , m _ { a } , R _ { n } , G _ { n } )$   
7: Generate $y _ { n } \sim P _ { \theta } ( \cdot \mid u _ { n } )$ ; apply edits $p _ { n } = \operatorname { E d i t } ( p _ { a } , y _ { n } )$   
8: Check and evaluate $p _ { n } ;$ record failures and continue if unsuccessful   
9: Obtain metrics $\mathbf { f } ( p _ { n } )$ and trajectory $\tau _ { n } ;$ compute $S ( p _ { n } )$   
10: Construct SPA feedback $d _ { n }$ from the trajectory and metrics   
11: Summarize the modules, update M, and construct PSA summary $b _ { n }$   
12: Store $p _ { n } ,$ its parent identity, and record $m _ { n } = ( \mathbf { f } ( p _ { n } ) , S ( p _ { n } ) , d _ { n } , b _ { n } )$ in D   
13: end for   
14: return $p ^ { \star }$ selected by Eq. (10)

For each revision, the prompt combines control code with measured outcomes and saved feedback:

$$
u _ { n } = H ( p _ { a } , m _ { a } , R _ { n } , G _ { n } ) ,\tag{11}
$$

It includes the parent’s saved PSA summary and can emphasize different modules at different search stages without restricting edits to those modules. The LLM then proposes changes:

$$
y _ { n } \sim P _ { \boldsymbol \theta } ( \cdot \mid u _ { n } ) , \qquad p _ { n } = \mathrm { E d i t } ( p _ { a } , y _ { n } ) .\tag{12}
$$

Here Edit applies SEARCH/REPLACE instructions (Appendix D.4), while the LLM parameters θ remain fixed. Successful evaluation yields traffic metrics and a trajectory $\tau _ { n }$ containing observations, phase decisions, and module outputs. SPA and PSA convert these outcomes into feedback stored with the candidate for later revision. Appendix D shows the actual prompt, response, code edits, and measured outcome of one iteration.

## 3.3 Multi-init: Initialization with Multiple Control Strategies

Multi-strategy initialization (Multi-init) introduces established control knowledge and complementary design principles into the initial population. Seeds can adapt traffic-control methods such as max-pressure or use LLM-generated implementations inspired by feedback control, fair service, and spatial coordination. Each expresses a control idea through the same 3M interfaces. These starting points are intended to diversify decision rules, reduce dependence on a single strategy, and supply alternative ideas for later revisions; the framework does not prescribe their number or types.

Let $p ^ { ( 1 ) } , \ldots , p ^ { ( K ) }$ denote the seeds. Each is evaluated under the same scenario and objective weights:

$$
\begin{array} { r } { \big ( \mathbf { f } ( p ^ { ( k ) } ) , \tau ^ { ( k ) } \big ) = E ( p ^ { ( k ) } ) , \qquad k = 1 , \dots , K . } \end{array}\tag{13}
$$

The evaluator E returns traffic metrics and a trajectory for each successful program. The evaluated seeds initialize $D _ { 0 } ,$ from which parents and references are selected. Seed evaluations precede the N evolution attempts. Appendices F.1 and F.2 describe the experimental seeds and their initialization records.

## 3.4 SPA: Simulation Performance Analyzer

The scalar score ranks overall performance but does not show where queues accumulate or how phase selection relates to demand. SPA converts recorded traffic behavior into diagnostics and revision suggestions using deterministic statistics and predefined rules. The LLM uses this report together with the program code to propose changes.

For trajectory $\tau _ { n } ,$ , let I be the number of intersections, T the number of diagnostic samples, $\varrho _ { i , t , m }$ the waitingvehicle count for movement $m ,$ and $a _ { i , t }$ the recorded phase action. With $B _ { k }$ denoting movements served by phase k, SPA relates average queued demand to selection frequency:

$$
\overline { { Q } } _ { k } ^ { \mathrm { p h a s e } } = \frac { 1 } { I T } \sum _ { i = 1 } ^ { I } \sum _ { t = 1 } ^ { T } \sum _ { m \in B _ { k } } Q _ { i , t , m } ,
$$

$$
\nu _ { k } = \frac { 1 } { I T } \sum _ { i = 1 } ^ { I } \sum _ { t = 1 } ^ { T } \mathbf { 1 } ( a _ { i , t } = k ) .\tag{14}
$$

Here $\mathbf { 1 } ( \cdot )$ is the indicator function. The pair $( \overline { { \boldsymbol { Q } } } _ { k } ^ { \mathrm { p h a s e } } , \nu _ { k } )$ helps identify mismatches between demand and phase service, although selection frequency alone does not establish whether green time is adequate. SPA also examines switching, queue changes, and congestion across locations and time windows; Appendix F.3 gives these statistics and diagnostic settings.

The feedback is

$$
d _ { n } = \mathrm { S P A } \big ( \tau _ { n } , \mathbf { f } ( p _ { n } ) \big ) ,\tag{15}
$$

The report suggests directions such as checking phase-continuation rules, demand–priority alignment, or service under late congestion. Subsequent evaluation determines whether an edit improves performance. Feedback $d _ { n }$ is stored with the candidate and enters Eq. (11) when it becomes a parent. Appendix D.2 shows a report and its corresponding revision.

## 3.5 PSA: Pareto Strategy Archive

A program with a lower aggregate score may contain useful rules for reducing queues or waiting. PSA retains module summaries and measured outcomes from recent successful evaluations so that such control ideas can inform later revisions. An auxiliary LLM call summarizes each program’s three modules as $s _ { n } =$ $( s _ { n } ^ { 1 } , s _ { n } ^ { 2 } , s _ { n } ^ { 3 } )$ . The updated memory $M ^ { + }$ retains the latest L records; it includes successful seed evaluations.

All three traffic metrics are minimized. Program $p _ { a }$ dominates $p _ { b }$ when

$$
\begin{array} { r l } & { f _ { j } ( p _ { a } ) \leq f _ { j } ( p _ { b } ) \quad \mathrm { f o r ~ a l l ~ } j \in \{ T , Q , W \} , } \\ & { f _ { j } ( p _ { a } ) < f _ { j } ( p _ { b } ) \quad \mathrm { f o r ~ a t ~ l e a s t ~ o n e ~ } j . } \end{array}\tag{16}
$$

Writing $e _ { a } \prec e _ { b }$ for the corresponding records, the nondominated set is

$$
F = \{ e \in M ^ { + } : \exists e ^ { \prime } \in M ^ { + } \mathrm { ~ s u c h ~ t h a t ~ } e ^ { \prime } \prec e \} .\tag{17}
$$

This set represents trade-offs among the retained recent evaluations, rather than the complete search history. PSA combines high-scoring nondominated records with a small number of recent dominated records, pairing their metrics with module summaries. Appendix F.4 specifies memory updates and retrieval settings.

The resulting summary $b _ { n }$ is saved with the candidate and supplied when it later becomes a parent. PSA provides strategy information for revision; parent and final-program selection continue to follow the database rules and scalar score. It therefore supports reuse without assuming recovery of a globally optimal Pareto frontier. Appendix D.3 shows a recorded strategy summary and the LLM response proposing its reuse.

## 4 Experiments

Section 4.1 describes the experimental setup. Section 4.2 compares the selected evolved programs with reference controllers, and Section 4.3 examines their transfer across road networks and traffic demands. Section 4.4 evaluates evolution performance and component ablations. Finally, Section 4.5 analyzes how objective weights affect program performance and the observed search distributions.

(e) Hangzhou

## 4.1 Experimental Setup

## 4.1.1 Traffic Networks and Simulation Environment

Networks and traffic demand We use two real-world road networks and demand profiles from public TSC benchmarks [68, 69], with roadnet and flow files distributed by LLMLight [10]. Jinan’s Dongfeng network contains 12 signalized intersections, and Hangzhou’s Gudang network contains 16. Each approach has three incoming lanes for left-turn, through, and right-turn traffic. Figure 3 shows the layouts, phases, and demand profiles; Table 1 defines all scenario identifiers.

The main search uses J1, and the principal control comparison covers the five observed-demand scenarios J1–J3 and H1–H2. An auxiliary search on H1 examines transfer in the reverse direction. J-high, H-high, and J-24h provide supplementary tests under synthetic high and day-long demand, reported in Appendix B.

(a) Jinan  
![](images/9f774ae19f9b43db00dc58756c9aba531071f305933403dfae4e30bf932ef9e3.jpg)  
(b) Hangzhou

![](images/aad9c8aa16936c49f9f44c03a10b13d4625dc97cda71b8d21ed5733f90937158.jpg)  
● Signalized intersection  
● Boundary  
(c) Signal phases

![](images/f2f3e005bfc474b9daf6fe3f4adcbe61d4199bb9fd1b7c3b3634f7fb2a753bdc.jpg)

![](images/68354ec8427791e599e19900e0998f41abdf8514d59ff5710e20f569bcefaa6e.jpg)

![](images/e7cc409dcc6b1321885a1cf4cf73d35387e560b49e1ccfe98a4ab57356cdc4f4.jpg)

![](images/027afc0f34a1222c77c3966dc0a8e0d8bb33a7a6eb67ecb2e5c12b360cef33f2.jpg)

![](images/3ce1d12815e9d360081eae2fecdab3557733a208ce60d01606bda343e20eed27.jpg)

![](images/f9214f21de25f81ab4a1529f10134d317d643cdd794e7c4099a8c91b01ddf7df.jpg)

![](images/24f2eecd2120264424e9b8fffe2df373a0cfb246a46595f22401a4eb598d439c.jpg)  
Figure 3. Traffic networks, selectable phases, and demand profiles. (a,b) Jinan and Hangzhou layouts. (c) Four selectable phases. (d,e) Observed demand on a common scale. (f) J-high on a separate scale. Arrivals use five-minute intervals.

Simulation and signal control Experiments use CityFlow [70] through the LLMLight environment [10], with 1-s simulation steps. Controllers choose among east–west through (ETWT), north–south through (NTST), east–west left-turn (ELWL), and north–south left-turn (NLSL) phases every 30 s. Changing phase incurs a 5-s yellow transition within this interval; retaining the current phase incurs none. Right turns remain available. The environment enforces this timing independently of program-level priority adjustments.

Table 1. Traffic demand statistics for the two road networks.
<table><tr><td rowspan="2">Demand scenario</td><td rowspan="2">ID</td><td rowspan="2">Signalized intersections</td><td rowspan="2"></td><td rowspan="2">Hours Vehicles</td><td colspan="4">Arrivals (vehicles / 5 min)</td></tr><tr><td>Mean</td><td>SD</td><td>Min</td><td>Max</td></tr><tr><td colspan="10">Core demand profiles</td></tr><tr><td>Jinan 1</td><td>J1</td><td></td><td></td><td></td><td>6,295 524.58</td><td>98.53</td><td>256</td><td>672</td></tr><tr><td>Jinan 2</td><td>J2</td><td>12</td><td></td><td>4,365</td><td>363.75</td><td>74.66</td><td>237</td><td>493</td></tr><tr><td>Jinan 3</td><td>J3</td><td></td><td>1</td><td>5,494</td><td>457.83</td><td>46.22</td><td>363</td><td>544</td></tr><tr><td>Jinan Extreme†</td><td>J-high</td><td></td><td>1</td><td>24,000</td><td>2000.00</td><td>44.21</td><td>1,933</td><td>2,098</td></tr><tr><td>Hangzhou 1</td><td>H1</td><td>16</td><td>1</td><td>2,983</td><td>248.58</td><td>40.45</td><td>212</td><td>333</td></tr><tr><td>Hangzhou 2</td><td>H2</td><td></td><td>1</td><td>6,984</td><td>582.00</td><td>318.52</td><td>203</td><td>1,146</td></tr><tr><td colspan="9">Supplementary demand profiles</td></tr><tr><td>Hangzhou Extreme† H-high</td><td></td><td>16</td><td>1</td><td>24,000</td><td>2000.00</td><td>40.79</td><td></td><td>1,9342,074</td></tr><tr><td>Jinan 24-hour†</td><td>J-24h</td><td>12</td><td>24</td><td>58,200</td><td>202.08</td><td>160.88</td><td>3</td><td>570</td></tr></table>

†: synthetic demand. Statistics are computed from scheduled departures in the actual flow files, using consecutive, non-overlapping five-minute bins over the full horizon SD is the population standard deviation across these bins; vehicle totals are scheduled departures, not completed trips.

Evaluation metrics We report travel time (TT), queue length (QL), and waiting time (WT), all minimized. Their definitions follow Section 3.1.

## 4.1.2 Compared Methods

We compare with 20 reference controllers and two LLM-guided program evolution frameworks, and evaluate four component ablations.

Conventional controllers. Three conventional policies provide reference performance.

• Random. Selects an admissible phase at random.

• FixedTime. Follows a predetermined phase schedule with fixed service durations.

• Maxpressure. Selects the phase with the largest pressure, using upstream and downstream queue differences to prioritize service [5].

RL-based controllers. Six methods cover pressure-based learning, attention-based phase selection, and coordination between intersections.

• MPLight. Combines pressure-based state and reward information with a phase-competition network and shared policy parameters [8].

• AttendLight. Aggregates lane information using attention and selects an available phase [32].

• PressLight. Incorporates intersection pressure into RL to support network traffic control [6].

• CoLight. Uses graph attention to exchange information among neighboring intersections [7].

• Efficient-CoLight. Augments CoLight with an efficient-pressure representation of traffic movements [33].

• Advanced-CoLight. Represents both queuing pressure and approaching traffic through advanced traffic-state features [34].

## LLM-based controllers. These methods use LLM inference to produce signal decisions.

• LLMLight generalist models. Select phases from descriptions of observed traffic [10]. The five backbones are Qwen2-72B, Llama2-70B, Llama3-70B, ChatGPT-3.5, and GPT-4.

• LightGPT. Uses traffic-specialized models developed through imitation fine-tuning and criticguided policy refinement [10]. The five variants are Qwen2-0.5B, Qwen2-7B, Llama2-7B, Llama3- 8B, and Llama2-13B.

• Traffic-R1. Applies two-stage reinforcement learning post-training to Qwen2.5-3B, using expertguided offline learning followed by interaction with simulated traffic environments [11]. At deployment, the resulting LLM processes traffic observations and generates signal-phase decisions.

LLM-guided program evolution frameworks. Both frameworks use the same 3M control interface. We compare their selected programs and their search performance, referring to the adapted implementations as OpenEvolve and ShinkaEvolve.

• OpenEvolve. Adapts an open-source framework for LLM-guided program evolution with execution-based feedback [14].

• ShinkaEvolve. Combines LLM-based program revisions with evolutionary search and adaptive proposal mechanisms [15].

EvoSignal configurations. We evaluate three configurations of EvoSignal, distinguished by their objective weights and source scenarios:

• EvoSignal. Evolves programs on J1 using the combined-score objective in Eq. (8), with $( w _ { T } , w _ { Q } , w _ { W } ) = ( 0 . 4 , 0 . 3 , 0 . 3 )$ for travel time (TT), queue length (QL), and waiting time (WT), respectively.

• EvoSignal (TT–QL). Uses the same objective with $( w _ { T } , w _ { Q } , w _ { W } ) = ( 0 . 4 8 , 0 . 4 8 , 0 . 0 4 )$ , increasing the emphasis on TT and QL and reducing the weight assigned to WT. Evolution is also conducted on J1.

• EvoSignal (H1, TT–QL). Uses the same TT–QL weights but evolves programs on H1 to examine transfer in the reverse direction.

Each configuration supplies a fixed program for the five-scenario comparison. We compare traffic metrics rather than scores computed under different objectives.

Ablation variants. We evaluate four variants of EvoSignal, each removing one component from the full framework.

• Without 3M. Represents the controller as a single editable code block and removes module-specific revision guidance.

• Without Multi-init. Initializes evolution with only the pressure–demand program in Table 11, instead of the full set of 11 initial programs.

• Without SPA. Removes traffic diagnostics and revision suggestions from the feedback supplied to the LLM, while retaining numerical evaluation results.

• Without PSA. Removes the Pareto strategy memory and its summaries from revision prompts, while retaining the underlying program database.

## 4.1.3 Evolution Settings and Evaluation Protocol

Control results and program selection The control tables use 19 reported LLMLight-benchmark controllers [10] and Traffic-R1 [11], whose results average three evaluations per scenario. For each J1 configuration, we select the returned program with the highest combined score on J1 across independent runs; an H1 program provides the reverse-transfer comparison. Each is evaluated unchanged on all five scenarios, without target-specific tuning or reselection, and ranked independently against the 20 references. Rank is one plus the number of references with a strictly lower metric value.

Initial control programs Following Section 3.3, we construct 11 seeds with LLM assistance from pressure-based control, feedback control, service fairness, congestion management, and coordination principles. Each uses the 3M interfaces and is checked for successful execution before evolution. Appendix F.1 lists their design principles and control rules; the pressure–demand seed alone initializes the without-Multiinit ablation.

Evolution budget and repeated runs Full EvoSignal starts from 11 programs and runs 200 evolution iterations after initialization. Following AlphaEvolve’s use of fast and more capable models [13], EvoSignal and its ablations use DeepSeek-V4-Flash and DeepSeek-V4-Pro [71], sampled with probabilities 70% and 30%, respectively, both at temperature 0.7. The primary framework and ablation comparisons use three independent runs per configuration; J1 TT–QL uses two and H1 TT–QL uses three. Appendix F provides the remaining settings.

Framework comparisons measure progress toward higher combined scores, final score quality, and variation across runs. The framework comparison uses progress relative to each framework’s configured iteration or generation budget; the ablation comparison uses evolution iterations. Neither axis measures elapsed time. Fixed control programs are evaluated separately by TT, QL, and WT across traffic scenarios.

## 4.2 Traffic Control Performance of Selected Evolved Programs

Tables 2 and 3 report TT, QL, and WT for the selected fixed control programs on the Jinan and Hangzhou networks. For EvoSignal, these are the programs selected by their source-scenario combined scores after 200 evolution iterations, following Section 4.1.3. Each program is ranked independently against the same 20 reference controllers. For each scenario and metric, the best reference value is the lowest value achieved by any of the 20 reference controllers.

The selected programs from all three evolution frameworks achieve lower WT than the best reference value in all five scenarios. The selected default EvoSignal program reduces WT by 16.78–49.24%, including on demands and a road network not used for search. TT and QL vary more across evolved programs.

On H2, the selected default EvoSignal program outperforms all 20 reference controllers on all three metrics. The EvoSignal (TT–QL) program improves on every reference controller on all three metrics in J1 and ranks first against these references on TT and QL in J2, with TT tied at the reported precision. These results show complementary performance profiles under different design objectives, examined in Section 4.5. Section 4.3 examines transfer in both directions for EvoSignal (TT–QL) and EvoSignal (H1, TT–QL).

Performance relative to Maxpressure Figure 4 highlights performance gains over Maxpressure. Like the evolved programs, Maxpressure uses explicit control rules and lightweight online computation. This comparison therefore assesses improvements within the same white-box, computationally lightweight form of control. Relative improvement is $1 0 0 ( m _ { \mathrm { M P } } - m _ { \mathrm { E v o } } ) / m _ { \mathrm { M P } }$ for metric m. The default EvoSignal program improves all three metrics in every scenario, with WT reductions of 17.10–55.62%. The TT–QL program improves TT and QL further on J1–J3 and H1, but its WT on J2 is 0.39% higher than Maxpressure. The main tables retain comparisons with all 20 reference controllers.

## 4.3 Transfer across Road Networks and Traffic Demands

Transfer in both directions All five scenarios share the observation structure and four-phase control interface described in Section 4.1. Tables 2 and 3 compare EvoSignal (TT–QL), evolved on J1, with EvoSignal (H1, TT–QL), evolved on H1 under the same objective weights. EvoSignal (TT–QL) is reused on J2 and J3, and on the different road network in H1 and H2. Conversely, EvoSignal (H1, TT–QL) is reused on H2 and on all three Jinan scenarios. Neither program undergoes further search or target-specific tuning. Both programs rank within the top three against the 20 reference controllers on every metric across the five scenarios. EvoSignal (H1, TT–QL) outperforms all 20 references on TT, QL, and WT in both J3 and H2. These results support reuse of the evolved control logic across both road networks and traffic demands.

Influence of the search scenario On J1, EvoSignal (TT–QL) reduces TT and QL by 0.97% and 2.68% relative to EvoSignal (H1, TT–QL), while WT is 4.12% higher. On H1, EvoSignal (H1, TT–QL) improves TT, QL, and WT by 0.19%, 2.30%, and 19.14%, respectively, relative to EvoSignal (TT–QL). Source scenario advantages are not uniform: EvoSignal (H1, TT–QL) also improves all three metrics on J3. Thus, strong transfer can coexist with scenario-dependent performance, and differences between evolved rules cannot be attributed to the search scenario alone.

Table 2. Traffic control performance on Jinan. Lower metric values and ranks are better.
<table><tr><td>Method</td><td></td><td>J1</td><td></td><td></td><td>J2</td><td></td><td>J3</td><td></td></tr><tr><td></td><td>TT (s)</td><td>QL (veh)</td><td>WT (s)</td><td>TT (s)</td><td>QL (veh)</td><td>WT (s)</td><td>TT (s) QL (veh)</td><td>WT (s)</td></tr><tr><td>Traditional controllers</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>597.62</td><td>687.35</td><td>99.46 555.23</td><td>428.38</td><td>100.40</td><td>552.74</td><td>529.63</td><td>99.33</td></tr><tr><td>FixedTime</td><td>481.79</td><td>491.03</td><td>70.99</td><td>441.19</td><td>294.14 66.72</td><td>450.11</td><td>394.34</td><td>69.19</td></tr><tr><td>Maxpressure</td><td>281.58</td><td>170.71</td><td>44.53</td><td>273.20</td><td>106.58 38.25</td><td>265.75</td><td>133.90</td><td>40.20</td></tr><tr><td>Reinforcement learning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MPLight</td><td>307.82</td><td>215.93</td><td>97.88</td><td>304.51</td><td>142.25 90.91</td><td>291.79</td><td>171.70</td><td>89.93</td></tr><tr><td>AttendLight</td><td>291.29</td><td>186.25</td><td>61.73</td><td>280.94</td><td>115.52</td><td>52.46 273.02</td><td>144.05</td><td>55.93</td></tr><tr><td>PressLight</td><td>291.57</td><td>185.46</td><td>50.53</td><td>281.46</td><td>115.99</td><td>47.27 275.85</td><td>148.18</td><td>54.81</td></tr><tr><td>CoLight</td><td>279.60</td><td>168.53</td><td>58.87</td><td>274.77</td><td>108.28</td><td>54.14 266.39</td><td>135.08</td><td>53.33</td></tr><tr><td>Efficient-CoLight</td><td>277.11</td><td>163.60</td><td>43.41</td><td>269.24</td><td>102.98</td><td>39.74 262.25</td><td>129.72</td><td>39.99</td></tr><tr><td>Advanced-CoLight</td><td>274.67</td><td>160.85</td><td>49.30</td><td>268.25</td><td>102.12</td><td>41.11 260.66</td><td>127.83</td><td>43.54</td></tr><tr><td colspan="9">LLMLight: generalist LLMs</td></tr><tr><td>Qwen2-72B</td><td>291.95</td><td>185.20</td><td>58.65</td><td>277.78</td><td>112.13</td><td>53.12 274.33</td><td>144.95</td><td>54.94</td></tr><tr><td>Llama2-70B</td><td>353.03</td><td>286.52</td><td>85.96</td><td>324.52</td><td>162.34</td><td>99.87 320.41</td><td>210.13</td><td>100.90</td></tr><tr><td>Llama3-70B</td><td>290.19</td><td>186.43</td><td>61.09</td><td>277.49</td><td>112.86</td><td>51.76 271.60</td><td>142.95</td><td>54.55</td></tr><tr><td>ChatGPT-3.5</td><td>536.79</td><td>952.93</td><td>155.62</td><td>524.81</td><td>393.21</td><td>179.34 501.36</td><td>479.50</td><td>154.19</td></tr><tr><td>GPT-4</td><td>275.26</td><td>160.93</td><td>46.61</td><td>271.34</td><td>105.22</td><td>47.55 264.70</td><td>132.53</td><td>46.16</td></tr><tr><td colspan="9">LLMLight: LightGPT</td></tr><tr><td>LightGPT (Qwen2-0.5B)</td><td>296.71</td><td>193.93</td><td>46.80</td><td>292.99</td><td>129.93</td><td>80.63 290.08</td><td>169.56</td><td>91.20</td></tr><tr><td>LightGPT (Qwen2-7B)</td><td>275.92</td><td>163.34</td><td>48.56</td><td>270.41</td><td>104.40</td><td>44.93 263.10</td><td>130.94</td><td>45.75</td></tr><tr><td>LightGPT (Llama2-7B)</td><td>275.11</td><td>161.35</td><td>46.38</td><td>269.01</td><td>102.92</td><td>43.06 260.53</td><td>127.75</td><td>41.84</td></tr><tr><td>LightGPT (Llama3-8B)</td><td>275.10</td><td>161.92</td><td>48.25</td><td>268.81</td><td>102.54</td><td>42.74 262.29</td><td>130.32</td><td>44.96</td></tr><tr><td>LightGPT (Llama2-13B)</td><td>274.03</td><td>159.39</td><td>43.24</td><td>266.94</td><td>100.46 40.34</td><td>260.17</td><td>127.08</td><td>41.00</td></tr><tr><td colspan="2">Additional LLM controller</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Traffic-R1</td><td>277.92</td><td>165.13</td><td></td><td>44.21 268.90</td><td>102.32</td><td>41.00 262.73</td><td>130.21</td><td>41.89</td></tr><tr><td colspan="2">Evolved control programs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OpenEvolve Rank vs. references</td><td>278.20 9</td><td>163.38</td><td>34.36 268.30</td><td></td><td>101.28</td><td>32.03 261.86</td><td>128.40</td><td>33.18</td></tr><tr><td>ShinkaEvolve</td><td>282.38</td><td>7</td><td>1</td><td>3</td><td>2</td><td>1</td><td>4 129.41</td><td>4 1 31.91</td></tr><tr><td>Rank vs. references</td><td>11</td><td>168.91</td><td>32.64 267.16</td><td></td><td>100.39</td><td>29.76 262.49</td><td>6</td><td>4 1</td></tr><tr><td>EvoSignal</td><td>277.74</td><td>10 163.29</td><td>1</td><td>2</td><td>1</td><td>1</td><td>129.09</td><td>33.28</td></tr><tr><td>Rank vs. references</td><td>8</td><td></td><td>34.31 268.52</td><td></td><td>102.30</td><td>31.71 262.20</td><td></td><td>4 1</td></tr><tr><td>EvoSignal (TT-QL)</td><td></td><td>6</td><td>1</td><td>3</td><td>3</td><td>1</td><td>4</td><td></td></tr><tr><td></td><td>271.85</td><td>155.53</td><td></td><td>41.98 266.94</td><td>100.08</td><td>38.40 260.58</td><td>127.79</td><td>39.72</td></tr><tr><td>Rank vs. references</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>2</td><td>3</td><td>3 1</td></tr><tr><td>EvoSignal (H1, TT-QL) 274.51</td><td></td><td>159.81</td><td>40.32 268.34</td><td></td><td>102.08</td><td>36.61 259.95</td><td>126.13</td><td>36.88</td></tr><tr><td>Rank vs. references</td><td>2</td><td>2</td><td>1</td><td>3</td><td>2</td><td>1</td><td>1 1</td><td>1</td></tr></table>

References comprise 19 reported LLMLight-benchmark controllers [10] and Traffic-R1 (mean of three evaluations). Each evolved program is ranked independently against these 20 references (1–21); other evolved programs are excluded. Bold marks the best reference values and evolved values that match or improve on them. Ranks use the displayed two-decimal metrics. Blue rows denote EvoSignal. –: unavailable. EvoSignal (H1, TT–QL) was searched on H1; the other evolved programs were searched on J1. Objective variants are defined in Section 4.1.2.

Supplementary evaluations under high and day-long demand show that these performance advantages do not extend uniformly to heavier traffic. Appendix B reports the complete fixed-program group results and their between-program variation.

## 4.4 Evolution Performance and Component Ablations

Comparison with evolution frameworks Figure 5 compares the three evolution frameworks on J1; Appendix A provides their exact final-score statistics. Curves show the mean best-so-far combined score during evolution over three independent runs, with bands indicating sample standard deviations (SDs). EvoSignal achieves a mean final best score of $0 . 3 9 7 3 1 4 \pm 0 . 0 0 0 4 8 9$ , compared with 0.394427 ± 0.002986 for OpenEvolve and 0.395840 ± 0.001751 for ShinkaEvolve. It has the highest mean score and the lowest variation across these runs.

Table 3. Traffic control performance on Hangzhou. Lower metric values and ranks are better.
<table><tr><td rowspan="2">Method</td><td colspan="3">H1</td><td colspan="3">H2</td></tr><tr><td>TT (s)</td><td>QL (veh)</td><td>WT (s)</td><td>TT (s)</td><td>QL (veh)</td><td>WT (s)</td></tr><tr><td>Traditional controllers</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random</td><td>621.14</td><td>295.81</td><td>96.06</td><td>504.28</td><td>432.92</td><td>92.61</td></tr><tr><td>FixedTime</td><td>616.02</td><td>301.33</td><td>73.99</td><td>486.72</td><td>425.15</td><td>72.80</td></tr><tr><td>Maxpressure</td><td>325.33</td><td>68.99</td><td>49.60</td><td>347.74</td><td>215.53</td><td>70.58</td></tr><tr><td>Reinforcement learning</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MPLight</td><td>345.60</td><td>84.70</td><td></td><td>81.97 358.56</td><td>237.17</td><td>100.16</td></tr><tr><td>AttendLight</td><td>322.94</td><td>66.96</td><td>55.19</td><td>358.81</td><td>239.05</td><td>72.88</td></tr><tr><td>PressLight</td><td>364.13</td><td>98.67</td><td>90.33</td><td>417.01</td><td>349.25</td><td>150.46</td></tr><tr><td>CoLight</td><td>322.85</td><td>66.94</td><td>61.82</td><td>342.90</td><td>212.09</td><td>99.74</td></tr><tr><td>Efficient-CoLight</td><td>311.96</td><td>58.20</td><td>36.83</td><td>333.27</td><td>189.65</td><td>61.70</td></tr><tr><td>Advanced-CoLight</td><td>304.47</td><td>52.94</td><td>41.75</td><td>329.16</td><td>186.34</td><td>76.59</td></tr><tr><td>LLMLight: generalist LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2-72B</td><td>321.91</td><td>65.75</td><td>66.52</td><td>342.37</td><td>206.07</td><td>100.55</td></tr><tr><td>Llama2-70B</td><td>357.95</td><td>93.21</td><td>106.63</td><td>361.53</td><td>250.69</td><td>121.58</td></tr><tr><td>Llama3-70B</td><td>325.85</td><td>69.42</td><td>71.51</td><td>339.23</td><td>198.82</td><td>78.81</td></tr><tr><td>ChatGPT-3.5</td><td>463.04</td><td>181.95</td><td>191.87</td><td>418.75</td><td>336.47</td><td>130.56</td></tr><tr><td>GPT-4</td><td>318.71</td><td>62.84</td><td>58.09</td><td>335.81</td><td>193.32</td><td>66.02</td></tr><tr><td>LLMLight: LightGPT</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LightGPT (Qwen2-0.5B)</td><td>328.11</td><td>70.90</td><td>84.49</td><td>351.21</td><td>233.16</td><td>102.39</td></tr><tr><td>LightGPT (Qwen2-7B)</td><td>313.37</td><td>59.03</td><td>50.65</td><td>335.30</td><td>198.47</td><td>72.93</td></tr><tr><td>LightGPT (Llama2-7B)</td><td>314.24</td><td>59.59</td><td>39.66</td><td>333.94</td><td>191.63</td><td>65.49</td></tr><tr><td>LightGPT (Llama3-8B)</td><td>311.72</td><td>58.40</td><td>47.60</td><td>333.85</td><td>192.06</td><td>69.75</td></tr><tr><td>LightGPT (Llama2-13B)</td><td>310.78</td><td>56.93</td><td>38.64</td><td>330.71</td><td>189.09</td><td>64.16</td></tr><tr><td>Additional LLM controller</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Traffic-R1</td><td>310.66</td><td>57.14</td><td></td><td>42.65 330.30</td><td>184.79</td><td>62.46</td></tr><tr><td>Evolved control programs</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OpenEvolve</td><td>310.55</td><td>57.66</td><td></td><td>29.39 328.98</td><td>178.87</td><td>36.22</td></tr><tr><td>Rank vs. references</td><td>2</td><td>4</td><td></td><td>1</td><td>1</td><td>1</td></tr><tr><td>ShinkaEvolve</td><td>315.67</td><td>60.90</td><td></td><td>27.93 329.30</td><td>176.78</td><td>30.95</td></tr><tr><td>Rank vs. references</td><td>8</td><td>8</td><td></td><td>2</td><td>1</td><td>1</td></tr><tr><td>EvoSignal</td><td>315.43</td><td>61.24</td><td></td><td>29.59 327.82</td><td>174.61</td><td>31.32</td></tr><tr><td>Rank vs. references</td><td>8</td><td>8</td><td></td><td>1</td><td>1</td><td>1</td></tr><tr><td>EvoSignal (TT-QL)</td><td>307.92</td><td>56.64</td><td></td><td>38.53 329.81</td><td>185.67</td><td>61.53</td></tr><tr><td>Rank vs. references</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>1</td></tr><tr><td>EvoSignal (H1, TT-QL) 307.33</td><td></td><td>55.34</td><td></td><td>31.15 328.68</td><td>180.92</td><td>54.70</td></tr><tr><td>Rank vs. references</td><td>2</td><td>2</td><td>1</td><td>1</td><td>1</td><td>1</td></tr></table>

Ranking, boldface, and color conventions follow Table 2. Each evolved program is compared independently with the same 20 references. EvoSignal (H1, TT–QL) was searched on H1; the others were searched on J1.

Component ablations Figure 6 compares the four ablations. Full EvoSignal has the highest mean final score and lowest final SD among these settings; Table 5 in Appendix A reports the values.

Without 3M. The monolithic controller reaches a similar best single-run score to full EvoSignal but has a lower mean score and substantially greater variation across runs. This supports more consistent search with 3M and its revision guidance. Explicit interfaces also support inspection and targeted revision around available sensing and communication resources.

Without Multi-init. Removing Multi-init yields the lowest mean final score, $0 . 3 9 2 1 9 0 \pm 0 . 0 0 5 4 2 8 .$ Full EvoSignal’s three final programs descend from different seeds—fairness-aware, efficient-pressure, and corridor-inspired control—showing use of multiple initial ideas. Initial scores and valid-record differences are reported in Appendix A.

![](images/122643f51e43418c0cd6c2b44d1363db44f9b682c28b575f42311a66dafd078b.jpg)

![](images/dbaec9be34c10b5689d9e608b53e14a4f6a2b2f90b11888df8d961a36270e373.jpg)

![](images/c5e7fe86127c8a6327f0a3e602edfc03762d79b9693dcb0c1f706cc2bdd9b06f.jpg)

Figure 4. Relative improvement over Maxpressure for two EvoSignal programs evolved on J1 and reused unchanged across five scenarios. Positive values indicate improvement.  
![](images/406fa7baff666d17c4179e97a2b9c618bb4a739800599cf699731885db728f85.jpg)

![](images/a07edb853e5519fd3756c5e84329352b13f520c0b6ea983c30f5f74074add097.jpg)

Figure 5. Evolution-framework comparison on J1 under the default objective. (a) Mean best-so-far score with SD bands over three runs. (b) Distribution of final best scores.  
![](images/28cff9c57a596295b69145716d8e829c952b838296d4b6c8ee332bd7eefe2ae7.jpg)

![](images/ebdd804788536b4b45f4986eb075cb2c21b2013b0181cc407d88b0b5b014e68e.jpg)  
Figure 6. Component ablations on J1 under the default objective: (a) best-so-far score over 200 evolution iterations; (b) final best scores. Each setting uses three runs, with the same plotting conventions as Figure 5.

Without SPA. Without SPA, the mean final score falls to 0.394508 and SD rises to 0.002684, consistent with the intended role of traffic diagnostics. Appendix D illustrates how diagnostics and archived strategies enter a recorded revision.

Without PSA. Removing PSA has a smaller effect, yielding 0.396607 ± 0.000639.

## 4.5 Influence of Objective Weights

Objective weights express operational priorities The two J1 programs in Figure 4 also illustrate the effect of objective weights. On J1, the TT–QL program reduces TT from 277.74 to 271.85 s and QL from 163.29 to 155.53 vehicles relative to the default program, while WT rises from 34.31 to 41.98 s. The weights and metric scales in Eq. (8) determine how these changes contribute to the combined score. Increasing the TT and QL weights reverses the ranking of these two programs; Appendix E gives the score decomposition. Scores computed under different objectives are not directly comparable.

Weights also change the observed search distribution Figure 7 shows the searches producing these programs. The TT–QL search concentrates on lower TT and QL, while the default search reaches lower WT; their selected programs occupy different parts of this trade-off. These observations describe the two recorded searches, rather than an optimal weight setting. Appendix E reports distribution statistics and additional views.

![](images/a0526b2a9a159262db426bae63e35e5951c352885b6d3b32ee8a9cf6f87575ec.jpg)

![](images/3fd8430c4741fbcc09519137ce4d2c6dbd2c7686dd054d4c1c9ba1763aa083d3.jpg)  
Figure 7. Objective-space distributions under the default and TT–QL objectives on J1: (a) travel time versus queue length; (b) queue length versus waiting time. Additional views and plotting details are provided in Appendix E.

The TT–QL program also has lower TT and QL but higher WT on J2, J3, and H1. On H2, the default program performs better on all three metrics. Objective weights guide the design of a new program rather than changing a fixed program’s decisions; the resulting trade-offs must therefore be assessed under the intended traffic conditions.

## 5 Program Analysis and Discussion

## 5.1 Control Rules of the Selected Programs

The two programs evolved on J1 and evaluated in Tables 2 and 3 illustrate the executable rules produced by EvoSignal. Both follow the same decision process: M1 extracts phase-level traffic features, M2 computes a priority for each candidate phase, and M3 supplies an optional network-based adjustment before the controller selects the phase with the highest priority. Table 4 summarizes the rules in their three modules.

Both programs use pressure, waiting vehicles, approaching vehicles, and recent phase-service history. Their main differences lie in how M2 combines this information. Default EvoSignal increases compensation for phases not recently requested as their waiting demand grows, while reducing the pressure contribution when downstream traffic is heavy. EvoSignal (TT–QL) adjusts the weight of waiting vehicles according to intersection congestion, limits service-age compensation, and reduces the combined priority when downstream occupancy is high. It also adds a preference for continuing the current phase when approaching demand remains relatively high.

The explicit program representation makes the implemented decision rules available for inspection. They provide a concrete interpretation of the complementary performance profiles in Section 4.5; the aggregate results do not isolate the contribution of each rule. Appendix C details the feature definitions, priority calculations, conditional rules, coefficients, and service-history updates. Appendix D.1 shows the corresponding editable code structure.

Table 4. Control rules in the three modules of the two selected J1 programs. Details are provided in Appendix C.
<table><tr><td>Component</td><td>EvoSignal</td><td>EvoSignal (TT-QL)</td></tr><tr><td>M1: features</td><td>Pressure, waiting and approaching traffic, upstream occupancy, downstream load, and time since a phase was requested.</td><td>Pressure, waiting and approaching traffic, max- imum downstream occupancy, and time since a phase was observed active.</td></tr><tr><td>M2: service</td><td>Service-age compensation grows with both age and waiting demand.</td><td>Service-age compensation is capped; the waiting weight adapts to congestion.</td></tr><tr><td>M2: congestion</td><td>Downstream load attenuates the pressure term; up- stream occupancy amplifies demand.</td><td>High downstream occupancy attenuates the com- bined phase score.</td></tr><tr><td>M2: persistence</td><td>Applies different discounts to current and compet- ing phase priorities at regular decision points.</td><td>Adds a continuation bonus when approaching de- mand remains relatively high.</td></tr><tr><td>M3: network adjust- ment</td><td>Favors the through phase with greater network di- rectional pressure; adds a common offset based on neighboring intersection loads.</td><td>No additional adjustment.</td></tr></table>

## 5.2 How Existing Strategies Are Revised

The default program provides an example of how evolution extends an initial control idea. Its fairnessaware seed, summarized in Table 11, increases priority for phases not recently selected and adjusts the preference for retaining the current phase. Evolution retains this service-history idea while incorporating pressure, waiting and approaching demand, and downstream traffic conditions. Later revisions change the demand weights and strengthen compensation for phases that have both a long service age and many waiting vehicles. Thus, the final rule combines service fairness with current traffic demand and downstream conditions. Appendix C.4 traces the code changes and their measured outcomes.

Appendix D provides a separate recorded example showing how such edits are produced: traffic diagnostics and archived strategies enter the prompt, and the LLM revises feature extraction and phase scoring through SEARCH/REPLACE edits. The appendix links the input, response, saved code, and evaluation results. Together, these records illustrate changes to the structure and use of traffic features, beyond adjusting coefficients in a fixed rule; they do not establish that each revision improves every metric or achieves its stated intent.

## 5.3 Implications for Traffic Control Design

The four components address different needs in automated traffic control design. 3M separates the processing of traffic observations from local phase prioritization and optional network-based adjustment. Multi-init supplies several established control ideas as executable starting points. SPA helps identify traffic conditions and service patterns that require attention. PSA makes earlier strategies and their measured trade-offs available for reuse. Together, these components guide the generation and evaluation of control rules whose source code remains available for inspection.

The selected program uses current traffic observations to choose signal phases without online LLM inference. LLM computation is therefore concentrated in the offline design stage, while repeated control decisions execute explicit rules. The experiments demonstrate traffic performance and traceable changes to these rules.

## 5.4 Limitations and Future Directions

Dependence on LLM interpretation and strategy reuse SPA computes traffic statistics and rule-based diagnostics, and PSA determines dominance from measured objectives. However, the quality of strategy summaries and the use of diagnostics and archived strategies during revision depend on the LLM. A useful observation may be overlooked or translated into a code change whose behavior differs from the generated explanation. Post-training on traffic control code and verified diagnostic–revision records could improve this capability, but its benefits remain to be evaluated. A complementary direction is to combine language based strategy reuse with explicit selection and sampling rules that account for objective-space coverage, strategy diversity, and measured performance. Such mechanisms could make the choice of reference pro grams less dependent on the LLM’s interpretation of a limited prompt.

Limited understanding and reuse of evolved control rules EvoSignal evaluates complete programs, while a single LLM revision may introduce several interacting changes to features, scoring rules, or conditional logic. A higher program score therefore does not establish which change is useful, under which traffic conditions, or whether its benefit transfers to another program. Future work should remove or replace individual changes while holding the remaining code fixed, examine their interactions across traffic conditions, and test their reuse in other controllers. Inspired by library learning in program synthesis [72], validated code fragments could be retained with their required inputs, tested conditions, failure cases, and performance trade-offs. Together, these steps could turn useful local improvements into reusable control knowledge and support the construction of better programs in subsequent searches.

## 6 Conclusion

This paper formulated TSC as a modular program design problem and presented EvoSignal to address it through LLM-guided evolution. Executable policies map traffic observations to signal phase decisions through connected modules for feature extraction, local phase prioritization, and optional network-based adjustment. Multiple initial strategies, traffic diagnostics, and reuse of strategies with different performance trade-offs guide the search. Control performance feedback guides code revision during design. The selected program then responds to traffic observations without online LLM inference.

Experiments on real-world road networks and traffic demand show that the selected default EvoSignal program reduces WT by 16.8–49.2% relative to the lowest WT achieved by the 20 reference controllers in each of the five scenarios. With greater emphasis on TT and QL, the framework produces a program that outperforms all references on all three metrics in J1 and ranks within the top three on every metric across all five scenarios without further revision. EvoSignal also achieves higher mean search scores and lower variation than the evaluated LLM-guided program evolution frameworks. The ablations and program records provide evidence of the roles of the four components. These findings support automated design of explicit traffic control rules, with their suitability assessed against the intended objectives and operating conditions.

## Declaration of Generative AI and AI-assisted technologies in the writing process

During the preparation of this work, the authors used ChatGPT to improve the clarity and readability of the manuscript. The authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

## References

[1] Markos Papageorgiou, Christina Diakaki, Vaya Dinopoulou, Apostolos Kotsialos, and Yibing Wang. Review of road traffic control strategies. Proceedings ofthe IEEE, 91(12):2043–2067, 2003. doi: 10.1109/jproc.2003.819610.

[2] Rex Chen, Fei Fang, and Norman Sadeh. The real deal: A review of challenges and opportunities in moving reinforcement learning-based traffic signal control systems towards reality. arXiv preprint arXiv:2206.11996, 2022.

[3] F. V. Webster. Traffic signal settings. Road Research Technical Paper 39, Road Research Laboratory, London, U.K., 1958. URL https://trid.trb.org/view/113579.

[4] P. B. Hunt, D. I. Robertson, R. D. Bretherton, and M. C. Royle. The SCOOT on-line traffic signal optimisation technique. Traffic Engineering & Control, 23(4):190–192, 1982. URL https://trid.trb.org/View/186640.

[5] Pravin Varaiya. Max pressure control of a network of signalized intersections. Transportation Research Part C: Emerging Technologies, 36:177–195, 2013. doi: 10.1016/j.trc.2013.08.014.

[6] Hua Wei, Chacha Chen, Guanjie Zheng, Kan Wu, Vikash Gayah, Kai Xu, and Zhenhui Li. PressLight: Learning max pressur control to coordinate traffic signals in arterial network. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1290–1298, 2019. doi: 10.1145/3292500.3330949.

[7] Hua Wei, Nan Xu, Huichu Zhang, Guanjie Zheng, Xinshi Zang, Chacha Chen, Weinan Zhang, Yanmin Zhu, Kai Xu, and Zhenhui Li. CoLight: Learning network-level cooperation for traffic signal control. In Proceedings ofthe 28th ACM International Conference on Information and Knowledge Management, pages 1913–1922, 2019. doi: 10.1145/3357384.3357902.

[8] Chacha Chen, Hua Wei, Nan Xu, Guanjie Zheng, Ming Yang, Yuanhao Xiong, Kai Xu, and Zhenhui Li. Toward a thousand lights: Decentralized deep reinforcement learning for large-scale traffic signal control. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 34, pages 3414–3421, 2020. doi: 10.1609/aaai.v34i04.5744.

[9] Yin Gu, Kai Zhang, Qi Liu, Weibo Gao, Longfei Li, and Jun Zhou. π-Light: Programmatic interpretable reinforcement learn ing for resource-limited traffic signal control. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 21107–21115, 2024. doi: 10.1609/aaai.v38i19.30103.

[10] Siqi Lai, Zhao Xu, Weijia Zhang, Hao Liu, and Hui Xiong. LLMLight: Large language models as traffic signal contro agents. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pages 2335– 2346, 2025. doi: 10.1145/3690624.3709379.

[11] Xingchen Zou, Yuhao Yang, Zheng Chen, Xixuan Hao, Yiqi Chen, Chao Huang, and Yuxuan Liang. Traffic-R1: Reinforced LLMs bring human-like reasoning to traffic signal control systems. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 21823–21838. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.995. URL https://aclanthology.org/2026.acl-long.995/.

[12] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M Pawan Kumar, Emilien Dupont, Francisco JR Ruiz, Jordan S Ellenberg, Pengming Wang, Omar Fawzi, et al. Mathematical discoveries from program search with large language models. Nature, 625(7995):468–475, 2024. doi: 10.1038/s41586-023-06924-6.

[13] Alexander Novikov, Ngân Vu, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov,˜ Borislav Kozlovskii, Francisco JR Ruiz, Abbas Mehrabian, et al. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv preprint arXiv:2506.13131, 2025.

[14] Asankhaya Sharma. OpenEvolve: an open-source evolutionary coding agent, 2025. URL https://github.com/ algorithmicsuperintelligence/openevolve. Base implementation: version 0.2.26.

[15] Robert Tjarko Lange, Yuki Imajuku, and Edoardo Cetin. ShinkaEvolve: Towards open-ended and sample-efficient program evolution. In The Fourteenth International Conference on Learning Representations, 2026. URL https://proceedings. iclr.cc/paper\_files/paper/2026/hash/7886b9bafe76c52fd568db10ff9772df-Abstract-Conference.html.

[16] Leizhen Wang, Peibo Duan, Hao Wang, Yue Wang, Jian Xu, Nan Zheng, and Zhenliang Ma. EvolveSignal: A large language model powered coding agent for discovering traffic signal control strategies. arXiv preprint arXiv:2509.03335, 2025.

[17] Da Lei, Feng Xiao, Lu Li, and Yuzhan Liu. SignalClaw: LLM-guided evolutionary synthesis of interpretable traffic signa control skills. arXiv preprint arXiv:2604.05535, 2026.

[18] Peter Koonce, Lee Rodegerdts, Kevin Lee, Shaun Quayle, Scott Beaird, Cade Braud, Jim Bonneson, Phil Tarnoff, and Tom Urbanik. Traffic signal timing manual. Technical Report FHWA-HOP-08-024, Federal Highway Administration, 2008. URL https://ops.fhwa.dot.gov/publications/fhwahop08024/fhwa\_hop\_08\_024.pdf.

[19] Jean Gregoire, Xiangjun Qian, Emilio Frazzoli, Arnaud de La Fortelle, and Tichakorn Wongpiromsarn. Capacity-Awar Backpressure Traffic Signal Control. IEEE Transactions on Control ofNetwork Systems, 2(2):164–173, 2015. doi: 10.1109 tcns.2014.2378871.

[20] Michael W. Levin, Jeffrey Hu, and Michael Odell. Max-pressure signal control with cyclical phase structure. Transportation Research Part C: Emerging Technologies, 120:102828, 2020. doi: 10.1016/j.trc.2020.102828.

[21] Ning Xie and Hao Wang. Distributed adaptive traffic signal control based on shockwave theory. Transportation Research Part C: Emerging Technologies, 173:105052, 2025. doi: 10.1016/j.trc.2025.105052.

[22] Leizhen Wang, Peibo Duan, Cheng Lyu, and Zhenliang Ma. Reinforcement learning-based sequential route recommendation for system-optimal traffic assignment. IEEE Open Journal of Intelligent Transportation Systems, 7:1483–1491, 2026. doi: 10.1109/OJITS.2026.3696779.

[23] Leizhen Wang, Peibo Duan, Cheng Lyu, Zewen Wang, Zhiqiang He, Nan Zheng, and Zhenliang Ma. Scalable and reliable multi-agent reinforcement learning for traffic assignment. Communications in Transportation Research, 5:100225, 2025. doi: 10.1016/j.commtr.2025.100225.

[24] Hua Wei, Guanjie Zheng, Vikash Gayah, and Zhenhui Li. Recent Advances in Reinforcement Learning for Traffic Signal Control. ACM SIGKDD Explorations Newsletter, 22(2):12–18, 2021. doi: 10.1145/3447556.3447565.

[25] Ammar Haydari and Yasin Yilmaz. Deep Reinforcement Learning for Intelligent Transportation Systems: A Survey. IEEE Transactions on Intelligent Transportation Systems, 23(1):11–32, 2022. doi: 10.1109/tits.2020.3008612.

[26] Baher Abdulhai, Rob Pringle, and Grigoris J. Karakoulas. Reinforcement Learning for True Adaptive Traffic Signal Control. Journal ofTransportation Engineering, 129(3):278–285, 2003. doi: 10.1061/(asce)0733-947x(2003)129:3(278).

[27] I. Arel, C. Liu, T. Urbanik, and A.G. Kohls. Reinforcement learning-based multi-agent system for network traffic signal control. IET Intelligent Transport Systems, 4(2):128–135, 2010. doi: 10.1049/iet-its.2009.0070.

[28] Samah El-Tantawy, Baher Abdulhai, and Hossam Abdelgawad. Multiagent Reinforcement Learning for Integrated Network of Adaptive Traffic Signal Controllers (MARLIN-ATSC): Methodology and Large-Scale Application on Downtown Toronto. IEEE Transactions on Intelligent Transportation Systems, 14(3):1140–1150, 2013. doi: 10.1109/tits.2013.2255286.

[29] Li Li, Yisheng Lv, and Fei-Yue Wang. Traffic signal timing via deep reinforcement learning. IEEE/CAA Journal ofAutomatica Sinica, 3(3):247–254, 2016. doi: 10.1109/jas.2016.7508798.

[30] Hua Wei, Guanjie Zheng, Huaxiu Yao, and Zhenhui Li. IntelliLight: A reinforcement learning approach for intelligent traffic light control. In Proceedings ofthe 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2496–2505, 2018. doi: 10.1145/3219819.3220096.

[31] Guanjie Zheng, Yuanhao Xiong, Xinshi Zang, Jie Feng, Hua Wei, Huichu Zhang, Yong Li, Kai Xu, and Zhenhui Li. Learning phase competition for traffic signal control. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, pages 1963–1972, 2019. doi: 10.1145/3357384.3357900.

[32] Afshin Oroojlooy, Mohammadreza Nazari, Davood Hajinezhad, and Jorge Silva. AttendLight: Universal attention-based reinforcement learning model for traffic signal control. In Advances in Neural Information Processing Systems, volume 33, pages 4079–4090, 2020. URL https://arxiv.org/abs/2010.05772.

[33] Qiang Wu, Liang Zhang, Jun Shen, Linyuan Lü, Bo Du, and Jianqing Wu. Efficient pressure: Improving efficiency for signalized intersections. arXiv preprint arXiv:2112.02336, 2021. URL https://arxiv.org/abs/2112.02336.

[34] Liang Zhang, Qiang Wu, Jun Shen, Linyuan Lü, Bo Du, and Jianqing Wu. Expression might be enough: Representing pressure and demand for reinforcement learning based traffic signal control. In Proceedings of the 39th International Conference on Machine Learning, volume 162, pages 26645–26654. PMLR, 2022. URL https://proceedings.mlr.press/v162/ zhang22ah.html.

[35] Qiang Wu, Mingyuan Li, Jun Shen, Linyuan Lü, Bo Du, and Ke Zhang. TransformerLight: A Novel Sequence Modeling Based Traffic Signaling Mechanism via Gated Transformer. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pages 2639–2647, 2023. doi: 10.1145/3580305.3599530.

[36] Leizhen Wang, Zhenliang Ma, Changyin Dong, and Hao Wang. Human-centric multimodal deep (HMD) traffic signal control. IET Intelligent Transport Systems, 17(4):744–753, 2023. doi: 10.1049/itr2.12300.

[37] Jihong Jin, Shangming Wu, Pengwei Zhang, Xiaorui Zhang, Chaoen Yin, and Changyin Dong. Hybrid action space approach to traffic signal optimization using deep reinforcement learning. Computers & Operations Research, 189:107391, 2026. doi: 10.1016/j.cor.2026.107391.

[38] Gengyue Han, Xiaohan Liu, Yu Han, Xianyue Peng, and Hao Wang. CycLight: Learning traffic signal cooperation with a cycle-level strategy. Expert Systems with Applications, 255:124543, 2024. doi: 10.1016/j.eswa.2024.124543.

[39] Gengyue Han, Xiaohan Liu, Hao Wang, Changyin Dong, and Yu Han. An Attention Reinforcement Learning-Based Strategy for Large-Scale Adaptive Traffic Signal Control System. Journal of Transportation Engineering, Part A: Systems, 150(3): 04024001, 2024. doi: 10.1061/JTEPBS.TEENG-8261.

[40] Tianshu Chu, Jie Wang, Lara Codecà, and Zhaojian Li. Multi-agent deep reinforcement learning for large-scale traffic signal control. IEEE Transactions on Intelligent Transportation Systems, 21(3):1086–1095, 2020. doi: 10.1109/TITS.2019.2901791.

[41] Zhenning Li, Hao Yu, Guohui Zhang, Shangjia Dong, and Cheng-Zhong Xu. Network-wide traffic signal control optimization using a multi-agent deep reinforcement learning. Transportation Research Part C: Emerging Technologies, 125:103059, 2021. doi: 10.1016/j.trc.2021.103059.

[42] Xinshi Zang, Huaxiu Yao, Guanjie Zheng, Nan Xu, Kai Xu, and Zhenhui Li. MetaLight: Value-based meta-reinforcement learning for traffic signal control. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 1153– 1160, 2020. doi: 10.1609/aaai.v34i01.5467.

[43] Gengyue Han, Yu Han, Hao Wang, Tiancheng Ruan, and Changze Li. Coordinated Control of Urban Expressway Integrating Adjacent Signalized Intersections Using Adversarial Network Based Reinforcement Learning Method. IEEE Transactions on Intelligent Transportation Systems, 25(2):1857–1871, 2024. doi: 10.1109/TITS.2023.3314409.

[44] Yunxue Lu, Andreas Hegyi, A. Maria Salomons, and Hao Wang. Reference RL: Reinforcement learning with reference mechanism and its application in traffic signal control. Information Sciences, 689:121485, 2025. doi: 10.1016/j.ins.2024. 121485.

[45] Abhinav Verma, Vijayaraghavan Murali, Rishabh Singh, Pushmeet Kohli, and Swarat Chaudhuri. Programmatically Inter pretable Reinforcement Learning. In Proceedings of the 35th International Conference on Machine Learning, volume 80, pages 5045–5054. PMLR, 2018. URL https://proceedings.mlr.press/v80/verma18a.html.

[46] Zhenliang Ma, Leizhen Wang, Zhenlin Qin, and Yancheng Ling. Large language models for urban transportation. In Mobility Patterns, Big Data and Transport Analytics: Tools and Applications for Modeling, chapter 10, pages 287–318. Elsevier, second edition, 2026. doi: 10.1016/B978-0-443-26789-5.00011-0.

[47] Xinglei Wang, Meng Fang, Zichao Zeng, and Tao Cheng. Where Would I Go Next? Large Language Models as Human Mobility Predictors. arXiv preprint arXiv:2308.15197, 2023. URL https://arxiv.org/abs/2308.15197.

[48] Zhenlin Qin, Pengfei Zhang, Leizhen Wang, and Zhenliang Ma. LingoTrip: Spatiotemporal context prompt driven large language model for individual trip prediction. Journal of Public Transportation, 27:100117, 2025. doi: 10.1016/j.jpubtr. 2025.100117.

[49] Leizhen Wang, Peibo Duan, Zhengbing He, Cheng Lyu, Xin Chen, Nan Zheng, Li Yao, and Zhenliang Ma. Agentic Large Language Models for day-to-day route choices. Transportation Research Part C: Emerging Technologies, 180:105307, 2025. doi: 10.1016/j.trc.2025.105307.

[50] Siyao Zhang, Daocheng Fu, Wenzhe Liang, Zhao Zhang, Bin Yu, Pinlong Cai, and Baozhen Yao. TrafficGPT: Viewing, processing and interacting with traffic foundation models. Transport Policy, 150:95–105, 2024. doi: 10.1016/j.tranpol.2024. 03.006.

[51] OpenAI. GPT-4 Technical Report. arXiv preprint arXiv:2303.08774, 2023. URL https://arxiv.org/abs/2303.08774.

[52] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint arXiv:2307.09288, 2023. URL https://arxiv.org/abs/2307.09288.

[53] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783, 2024. URL https://arxiv.org/abs/2407.21783.

[54] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 Technical Report. arXiv preprint arXiv:2407.10671, 2024. URL https://arxiv.org/abs/2407. 10671.

[55] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems, volume 35, 2022. URL https://arxiv.org/abs/2201.11903.

[56] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-Consistency Improves Chain of Thought Reasoning in Language Models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2203.11171.

[57] Zhenlin Qin, Leizhen Wang, Yancheng Ling, Francisco Camara Pereira, and Zhenliang Ma. A foundational individual mobility prediction model based on open-source large language models. Transportation Research Part C: Emerging Technologies, 185:105562, 2026. doi: 10.1016/j.trc.2026.105562.

[58] Xixi Wang, Jordanka Kovaceva, Miguel Costa, Shuai Wang, Francisco Camara Pereira, and Robert Thomson. Domainadapted pre-trained language models for implicit information extraction in crash narratives. Transportation Research Part C: Emerging Technologies, 192:105863, 2026. doi: 10.1016/j.trc.2026.105863.

[59] Maonan Wang, Aoyu Pang, Yuheng Kan, Man-On Pun, Chung Shue Chen, and Bo Huang. LLM-Assisted Light: Leveraging Large Language Model Capabilities for Human-Mimetic Traffic Signal Control in Complex Urban Environments. arXiv preprint arXiv:2403.08337, 2024. URL https://arxiv.org/abs/2403.08337.

[60] Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating Large Language Models Trained on Code. arXiv preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

[61] Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with AlphaCode. Science, 378(6624):1092–1097, 2022. doi: 10.1126/science.abq1158.

[62] Fei Liu, Xialiang Tong, Mingxuan Yuan, Xi Lin, Fu Luo, Zhenkun Wang, Zhichao Lu, and Qingfu Zhang. Evolution of heuristics: Towards efficient automatic algorithm design using large language model. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235, pages 32201–32223. PMLR, 2024. URL https://proceedings.mlr. press/v235/liu24bs.html.

[63] Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-Level Reward Design via Coding Large Language Models. In International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2310.12931.

[64] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An Open-Ended Embodied Agent with Large Language Models. arXiv preprint arXiv:2305.16291, 2023. URL https://arxiv.org/abs/2305.16291.

[65] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shri mai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Self-Refine: Iterative refinement with self-feedback. In Advances in Neural Information Processing Systems, volume 36, 2023. doi: 10.52202/075280-2019.

[66] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, 2023. doi: 10.52202/ 075280-0377.

[67] Haoran Ye, Jiarui Wang, Zhiguang Cao, Federico Berto, Chuanbo Hua, Haeyeon Kim, Jinkyoo Park, and Guojie Song. ReEvo: Large language models as hyper-heuristics with reflective evolution. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-1381.

[68] Hua Wei, Guanjie Zheng, Vikash V. Gayah, and Zhenhui Li. A survey on traffic signal control methods. arXiv preprint arXiv:1904.08117, 2019. URL https://arxiv.org/abs/1904.08117.

[69] Traffic Signal Control. Reinforcement learning for traffic signal control. Public benchmark website, n.d. URL https: //traffic-signal-control.github.io/. Accessed September 28, 2026.

[70] Huichu Zhang, Siyuan Feng, Chang Liu, Yaoyao Ding, Yichen Zhu, Zihan Zhou, Weinan Zhang, Yong Yu, Haiming Jin, and Zhenhui Li. CityFlow: A multi-agent reinforcement learning environment for large scale city traffic scenario. In The World Wide Web Conference, pages 3620–3624, 2019. doi: 10.1145/3308558.3314139.

[71] DeepSeek-AI. DeepSeek-V4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026. doi: 10.48550/arXiv.2606.19348. URL https://arxiv.org/abs/2606.19348.

[72] Lionel Wong, Kevin M Ellis, Joshua Tenenbaum, and Jacob Andreas. Leveraging language to learn program abstractions and search heuristics. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pages 11193–11204. PMLR, 2021. URL https://proceedings.mlr.press/v139/ wong21a.html.

## Appendix A. Search Statistics

Table 5 reports the primary three-run comparison used in Figures 5 and 6.

Table 5. Final best search scores on J1 under the default objective (Section 4.1.2).
<table><tr><td>Configuration</td><td>Runs</td><td>Mean ± SD</td><td>Best run</td></tr><tr><td>EvoSignal</td><td>3</td><td>0.397314 ± 0.000489</td><td>0.397742</td></tr><tr><td>OpenEvolve</td><td>3</td><td> $0 . 3 9 4 4 2 7 \pm 0 . 0 0 2 9 8 6$ </td><td>0.397484</td></tr><tr><td>ShinkaEvolve</td><td>3</td><td> $0 . 3 9 5 8 4 0 \pm 0 . 0 0 1 7 5 1$ </td><td>0.397672</td></tr><tr><td>Without SPA</td><td>3</td><td> $0 . 3 9 4 5 0 8 \pm 0 . 0 0 2 6 8 4$ </td><td>0.396541</td></tr><tr><td>Without PSA</td><td>3</td><td> $0 . 3 9 6 6 0 7 \pm 0 . 0 0 0 6 3 9$ </td><td>0.397309</td></tr><tr><td>Without 3M</td><td>3</td><td> $0 . 3 9 3 9 8 8 \pm 0 . 0 0 3 2 3 0$ </td><td>0.397682</td></tr><tr><td>Without Multi-init</td><td>3</td><td> $0 . 3 9 2 1 9 0 \pm 0 . 0 0 5 4 2 8$ </td><td>0.395863</td></tr></table>

SD is the sample standard deviation across independent search runs. Without 3M uses the archived monolithic controller and its corresponding search guidance.

Additional component statistics The mean best initial score is 0.385995 with Multi-init and 0.384187 with the single seed; subsequent evolution adds 0.011319 and 0.008002 on average, respectively. Singleseed runs also contain fewer valid evolutionary records, which should be considered when interpreting final performance. Averaged over configured search steps, the best-so-far score is 0.395385 with PSA and 0.394389 without it.

## Appendix B. Complete Evaluations of Fixed Programs

Tables 6 and 7 report the performance of all fixed program groups across the eight scenarios in Table 1. The code is unchanged across scenarios. Means and SDs summarize differences between independently obtained programs within each group. The H1 TT–QL group comprises three independently evolved programs; its group statistics are distinct from the single representative in the main tables.

High and day-long demand The three default EvoSignal programs have lower mean WT than the two EvoSignal (TT–QL) programs on J-high, H-high, and J-24h: 197.02 versus 332.66 s, 81.09 versus 312.72 s, and 26.77 versus 29.29 s. Default EvoSignal also has lower TT and QL under both high-flow profiles, while TT–QL has slightly lower TT and QL over the full day. Compared with these two groups, ShinkaEvolve has lower means on all three metrics under both high-flow profiles; on J-24h, default EvoSignal has slightly lower TT and QL, with similar WT. These results show that advantages on the five main scenarios do not extend uniformly to heavier demand. Each fixed program is evaluated once per scenario; SDs describe variation between programs.

Table 6. Fixed-program results: EvoSignal, EvoSignal (TT–QL).

$$
\mathrm { Q L } \left( \mathrm { v e h } \right)
$$

$$
2 7 5 . 9 7 \pm 1 . 8 6
$$

$$
1 6 0 . 6 4 \pm 2 . 5 0
$$

$$
3 5 . 6 4 \pm 1 . 5 2
$$

$$
2 6 8 . 2 1 \pm 0 . 5 1
$$

$$
1 0 1 . 7 5 \pm 0 . 5 0
$$

$$
3 3 . 0 6 \pm 1 . 4 3
$$

$$
2 6 1 . 5 7 \pm 1 . 0 0
$$

$$
1 2 8 . 5 6 \pm 0 . 5 6
$$

$$
3 4 . 3 4 \pm 0 . 9 6
$$

$$
3 1 6 . 1 5 \pm 2 . 1 8
$$

$$
6 1 . 7 6 \pm 1 . 3 2
$$

$$
2 9 . 8 8 \pm 0 . 4 7
$$

$$
3 2 9 . 6 9 \pm 1 . 6 2
$$

$$
1 7 7 . 7 1 \pm 2 . 6 9
$$

$$
3 4 . 4 5 \pm 4 . 9 2
$$

$$
8 5 5 . 2 0 \pm 1 4 . 1 0
$$

$$
2 6 5 4 . 5 0 \pm 6 0 . 3 8
$$

$$
1 9 7 . 0 2 \pm 6 6 . 5 6
$$

$$
6 8 1 . 2 7 \pm 2 6 . 3 2
$$

$$
1 4 4 6 . 0 6 \pm 8 8 . 5 2
$$

$$
8 1 . 0 9 \pm 2 8 . 4 8
$$

$$
2 7 1 . 3 1 \pm 1 . 2 4
$$

$$
5 6 . 2 7 \pm 0 . 7 2
$$

$$
2 6 . 7 7 \pm 0 . 3 5
$$

$$
2 7 2 . 2 7 \pm 0 . 5 9
$$

$$
1 5 6 . 6 8 \pm 1 . 6 1
$$

$$
4 1 . 6 4 \pm 0 . 4 8
$$

$$
2 6 7 . 2 4 \pm 0 . 4 2
$$

$$
1 0 0 . 5 0 \pm 0 . 6 1
$$

$$
3 7 . 7 3 \pm 0 . 9 6
$$

$$
2 6 0 . 8 2 \pm 0 . 3 4
$$

$$
1 2 8 . 1 0 \pm 0 . 4 4
$$

$$
3 9 . 7 3 \pm 0 . 0 2
$$

$$
3 1 0 . 9 4 \pm 4 . 2 7
$$

$$
5 8 . 2 5 \pm 2 . 2 7
$$

$$
3 8 . 3 5 \pm 0 . 2 5
$$

$$
3 3 0 . 1 3 \pm 0 . 4 5
$$

$$
1 8 4 . 2 5 \pm 2 . 0 0
$$

$$
5 6 . 0 5 \pm 7 . 7 5
$$

$$
9 0 2 . 2 3 \pm 1 7 . 7 5
$$

$$
2 8 7 0 . 4 0 \pm 7 6 . 0 9
$$

$$
3 3 2 . 6 6 \pm 1 0 4 . 6 7
$$

$$
7 2 6 . 6 8 \pm 5 2 . 6 5
$$

$$
1 6 6 1 . 2 4 \pm 2 3 5 . 4 7
$$

$$
3 1 2 . 7 2 \pm 2 7 5 . 9 1
$$

$$
2 6 9 . 4 7 \pm 0 . 4 4
$$

$$
5 5 . 1 8 \pm 0 . 1 6
$$

$$
2 9 . 2 9 \pm 1 . 0 2
$$

Mean ± sample SD across different fixed programs. Each program contributes one evaluation to each scenario. EvoSignal (H1, TT–QL) programs were searched on H1;   
the other groups were searched on J1.

Table 7. Fixed-program results: EvoSignal (H1, TT–QL), ShinkaEvolve.
<table><tr><td>Scenario Programs</td><td></td><td>TT (s) QL (veh)</td><td>WT (s)</td></tr><tr><td>EvoSignal (H1, TT–QL)</td><td></td><td></td><td></td></tr><tr><td>J1</td><td>3  $2 7 6 . 6 6 \pm 3 . 7 4$ </td><td> $1 6 2 . 5 4 \pm 4 . 2 6$ </td><td> $4 0 . 4 4 \pm 1 . 6 7$ </td></tr><tr><td>J2 3</td><td> $2 6 8 . 4 0 \pm 0 . 2 6$ </td><td> $1 0 2 . 0 8 \pm 0 . 0 0$ </td><td> $3 7 . 2 9 \pm 1 . 3 6$ </td></tr><tr><td>J3</td><td>3  $2 6 0 . 3 1 \pm 1 . 0 6$ </td><td> $1 2 7 . 3 6 \pm 1 . 4 3$ </td><td> $3 8 . 9 7 \pm 2 . 0 6$ </td></tr><tr><td>H1 3</td><td> $3 0 7 . 7 7 \pm 0 . 5 9$ </td><td> $5 5 . 8 6 \pm 0 . 5 9$ </td><td> $3 2 . 3 4 \pm 2 . 1 7$ </td></tr><tr><td>H2</td><td>3  $3 3 2 . 1 2 \pm 3 . 7 1$ </td><td> $1 8 6 . 2 5 \pm 6 . 3 6$ </td><td> $5 2 . 9 2 \pm 2 . 3 4$ </td></tr><tr><td>J-high</td><td>3  $8 1 5 . 1 6 \pm 7 . 9 4$ </td><td> $2 6 2 7 . 9 9 \pm 8 0 . 9 0$ </td><td> $1 3 2 . 6 3 \pm 3 9 . 2 2$ </td></tr><tr><td>H-high</td><td>3  $6 9 5 . 3 8 \pm 1 1 . 8 9$ </td><td> $1 5 6 1 . 0 1 \pm 1 5 . 1 9$ </td><td> $1 0 3 . 7 2 \pm 7 . 7 2$ </td></tr><tr><td>J-24h</td><td>3  $2 6 9 . 7 8 \pm 0 . 6 0$ </td><td> $5 5 . 4 9 \pm 0 . 4 7$ </td><td> $2 8 . 1 5 \pm 1 . 2 2$ </td></tr><tr><td>ShinkaEvolve</td><td></td><td></td><td></td></tr><tr><td>J1</td><td>3  $2 7 9 . 3 2 \pm 3 . 2 9$ </td><td> $1 6 4 . 9 1 \pm 4 . 2 8$ </td><td> $3 4 . 7 3 \pm 2 . 1 5$ </td></tr><tr><td>J2</td><td>3  $2 6 8 . 0 3 \pm 1 . 6 9$ </td><td> $1 0 1 . 3 9 \pm 2 . 0 1$ </td><td> $3 1 . 6 1 \pm 1 . 9 5$ </td></tr><tr><td>J3</td><td>3  $2 6 2 . 3 8 \pm 0 . 2 8$ </td><td> $1 2 9 . 6 8 \pm 0 . 4 7$ </td><td> $3 3 . 6 9 \pm 2 . 2 6$ </td></tr><tr><td>H1</td><td>3  $3 1 7 . 4 3 \pm 1 . 5 5$ </td><td> $6 2 . 1 3 \pm 1 . 0 7$ </td><td> $3 1 . 1 1 \pm 6 . 2 6$ </td></tr><tr><td>H2</td><td>3  $3 2 9 . 7 1 \pm 0 . 8 1$ </td><td> $1 7 7 . 3 5 \pm 1 . 7 2$ </td><td> $3 4 . 5 2 \pm 8 . 6 9$ </td></tr><tr><td>J-high</td><td>3  $8 2 3 . 0 4 \pm 8 . 5 2$ </td><td> $2 5 2 0 . 1 8 \pm 2 2 . 8 9$ </td><td> $1 1 5 . 7 9 \pm 1 2 . 7 8$ </td></tr><tr><td>H-high</td><td>3  $6 5 8 . 0 3 \pm 1 4 . 6 2$ </td><td> $1 3 6 2 . 2 1 \pm 5 3 . 3 2$ </td><td> $5 9 . 3 4 \pm 8 . 8 6$ </td></tr><tr><td>J-24h</td><td>3  $2 7 2 . 4 2 \pm 0 . 2 9$ </td><td> $5 6 . 7 9 \pm 0 . 2 0$ </td><td> $2 6 . 6 8 \pm 2 . 7 8$ </td></tr></table>

Mean ± sample SD across different fixed programs. Each program contributes one evaluation to each scenario. EvoSignal (H1, TT–QL) programs were searched on H1; the other groups were searched on J1.

## Appendix C. Feature Extraction and Priority Rules of the Selected Programs

This appendix describes the two fixed J1 programs used in the main comparison: default EvoSignal and EvoSignal (TT–QL). The equations summarize the computations used for phase selection under the evaluated control protocol, including feature extraction, priority adjustments, and service-history updates. They assume finite, nonnegative vehicle observations and an admissible current phase. The overall phaseselection interface is defined in Section 3.1.2. Coefficients below are constants in the selected control programs; they are distinct from the search-objective weights.

## C.1 Shared Notation and Feature Aggregation

We follow Section 3.1.2 and omit intersection and time subscripts for a single decision. Phase $k \in P$ serves movements $B _ { k }$ , and $K = | P | = 4$ in the experiments. Let $e _ { m }$ be the observed efficient-pressure feature, $Q _ { m }$ the waiting-vehicle count, $V _ { m }$ the approaching-vehicle feature, and $n _ { m } , n _ { m } ^ { d }$ the upstream and downstream vehicle counts. Both programs use

$$
E _ { k } = \sum _ { m \in B _ { k } } e _ { m } , \qquad Q _ { k } = \sum _ { m \in B _ { k } } Q _ { m } , \qquad V _ { k } = \sum _ { m \in B _ { k } } \operatorname* { m a x } ( V _ { m } , 0 ) .\tag{18}
$$

Thus, $E _ { k }$ aggregates the supplied efficient-pressure feature, rather than assuming it equals a newly defined pressure formula. Let $k _ { 0 }$ be the current phase after converting its observation to the program’s zero-based index, and h the observed time\_this\_phase. The simulation updates h in seconds. Under the evaluation protocol, the first decision observes $h = 0$ . Subsequent decisions occur every 30 s: a phase change includes 5 s of yellow followed by 25 s of green, so $h \geq 2 5$ at subsequent decision points. Retaining a phase increases its duration by another 30 s. The priority equations below describe regular decision points after initialization; initialization is stated separately. In contrast, the service-age variables below count nontransition calls to the controller; their thresholds are not durations in seconds. Define $[ \nu ] _ { + } = \operatorname* { m a x } ( \nu , 0 )$ and $[ \nu ] _ { 0 } ^ { 1 } = \operatorname* { m i n } ( \operatorname* { m a x } ( \nu , 0 ) , 1 )$

Core demand terms For each phase, both programs combine the features defined above into a core demand term. Superscripts D and T denote default EvoSignal and EvoSignal (TT–QL), respectively:

$$
\begin{array} { r l } & { b _ { k } ^ { \mathrm { D } } = ( 1 - d _ { k } ) E _ { k } + 0 . 8 Q _ { k } + 0 . 4 V _ { k } , } \\ & { b _ { k } ^ { \mathrm { T } } = E _ { k } + ( 0 . 4 + 0 . 3 \rho ) Q _ { k } + 0 . 5 V _ { k } . } \end{array}\tag{19}
$$

The default program uses $d _ { k } = \operatorname* { m i n } ( \operatorname* { m a x } ( D _ { k } / 1 5 , 0 ) , 1 )$ , where $\begin{array} { r } { D _ { k } = \sum _ { m \in B _ { k } } n _ { m } ^ { d } } \end{array}$ is the downstream vehicle count for the phase. The TT–QL program uses $\begin{array} { r } { \rho = \operatorname* { m i n } ( \sum _ { k \in P } Q _ { k } / ( 5 | P | ) , \tilde { 1 } ) } \end{array}$ to adjust its waiting-vehicle weight with intersection congestion. These are intermediate demand terms, rather than complete phase priorities. The following subsections specify the additional adjustments and local scoring rules.

## C.2 Default EvoSignal

M1: occupancy, downstream load, and service history In addition to Eq. (18), M1 computes

$$
D _ { k } = \sum _ { m \in B _ { k } } n _ { m } ^ { d } , ~ O _ { k } = \operatorname* { m a x } _ { m \in B _ { k } } \bigl ( n _ { m } / 2 0 \bigr ) ,
$$

$$
N = \sum _ { m = 1 } ^ { 1 2 } n _ { m } , C = N / ( 1 2 \times 2 0 ) .\tag{20}
$$

The value 20 is a normalization constant in the program, not an independently estimated physical lane capacity. Let u be the controller’s counter of non-transition decision calls. With $\ell _ { k }$ initialized to zero, M1 forms $A _ { k } = \operatorname* { m a x } ( u - \ell _ { k } , 0 )$ . After choosing action $^ { a , }$ the controller sets $\ell _ { a } = u .$ Hence $A _ { k }$ measures calls since phase k was last requested, not the accumulated waiting time of its vehicles.

M2: demand scoring and service-age compensation Using $b _ { k } ^ { \mathrm { D } }$ and $d _ { k }$ from Eq. (19), the priority before current-phase adjustments is

$$
\tilde { q } _ { k } = \left[ b _ { k } ^ { \mathrm { D } } ( 1 + 0 . 5 [ O _ { k } ] _ { 0 } ^ { 1 } ) ( 1 + 0 . 3 C ) + 0 . 8 [ A _ { k } - 4 ] _ { + } ( 1 + 0 . 2 Q _ { k } ) \right] ( 1 + 0 . 0 2 N ) .\tag{21}
$$

The downstream factor attenuates the pressure term, upstream occupancy amplifies the demand term, and service-age compensation increases with waiting demand. At regular decision points, the local priorities are

$$
q _ { k } = \left\{ { \begin{array} { l l } { 0 . 4 \widetilde { q } _ { k } , } & { k = k _ { 0 } , } \\ { 0 . 9 5 \widetilde { q } _ { k } , } & { k \neq k _ { 0 } . } \end{array} } \right.\tag{22}
$$

These factors reduce the preference for retaining the current phase relative to competing phases. At initialization, the program instead sets $q _ { k _ { 0 } } = 1 0 ^ { 9 }$ and $q _ { k } = \widetilde { q } _ { k }$ for $k \neq k _ { 0 }$ . Signal timing follows the evaluation protocol in Section 4.1.

M3: network-based priority adjustment Let $P _ { E }$ and $P _ { N }$ denote the queue-pressure features summed over the east–west and north–south movement groups across available shared intersection records, and let $M = \operatorname* { m a x } ( P _ { E } , P _ { N } )$ . These queue-pressure summaries are distinct from the efficient-pressure feature $E _ { k }$ . Let $J _ { i }$ count available adjacency entries whose total vehicle count exceeds the current intersection’s N by more than five; it is zero when adjacency or the current intersection’s shared record is unavailable. The adjustment is

$$
r _ { k } = - 0 . 0 5 J _ { i } + \left\{ \begin{array} { l l } { 0 . 3 [ P _ { E } - P _ { N } ] _ { + } / M , } & { M > 1 , k = 0 , } \\ { 0 . 3 [ P _ { N } - P _ { E } ] _ { + } / M , } & { M > 1 , k = 1 , } \\ { 0 , } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{23}
$$

Phases 0 and 1 serve east–west and north–south through traffic, respectively. The directional term favors the through phase associated with greater network pressure, while the neighbor-load term applies the same offset to all phase scores. The code updates $g \gets \operatorname* { m a x } ( 0 . 1 , g )$ and combines the adjustment with local priorities through Eq. (6).

## C.3 EvoSignal (TT–QL)

M1: downstream occupancy and service history Alongside Eq. (18), the features used by the final rule are

$$
O _ { k } ^ { d } = \operatorname* { m a x } _ { m \in { \cal B } _ { k } } ( n _ { m } ^ { d } / 2 0 ) , \qquad Q _ { \mathrm { t o t } } = \sum _ { k \in { \cal P } } Q _ { k } , \qquad \rho = \operatorname* { m i n } \left( \frac { Q _ { \mathrm { t o t } } } { 5 K } , 1 \right) .\tag{24}
$$

Here $Q _ { \mathrm { t o t } }$ is the sum of phase waiting features used by this controller. At call $u ,$ M1 first sets $\ell _ { k _ { 0 } } = u$ for the phase currently observed active, and then computes $A _ { k } = \operatorname* { m a x } ( u - \ell _ { k } , 0 )$ , with initial $\ell _ { k } = 0$ . Unlike the default program, it updates this history from the observed current phase before scoring, rather than from the requested action after scoring. In particular, $A _ { k _ { 0 } } = 0$

M2: bounded service compensation and downstream attenuation Using $b _ { k } ^ { \mathrm { T } }$ from Eq. (19), define the service-age bonus and the downstream factor as

$$
\begin{array} { r l } & { B _ { k } ^ { \mathrm { a g e } } = \operatorname* { m i n } \left( 8 , 0 . 3 [ A _ { k } - 1 0 ] _ { + } + \frac { 0 . 2 [ \rho - 0 . 5 ] _ { + } [ A _ { k } - 1 0 ] _ { + } } { \operatorname* { m a x } ( A _ { k } , 1 ) } \right) , } \\ & { F _ { k } = \left\{ 1 , \qquad \begin{array} { l l } { 1 , } & { O _ { k } ^ { d } \leq 0 . 8 , } \\ { \operatorname* { m a x } \left( 1 - 2 ( O _ { k } ^ { d } - 0 . 8 ) , 0 . 1 \right) , } & { O _ { k } ^ { d } > 0 . 8 . } \end{array} \right. } \end{array}\tag{25}
$$

The intermediate priority is

$$
\widetilde { q } _ { k } = \operatorname* { m a x } \left( ( b _ { k } ^ { \mathrm { T } } + B _ { k } ^ { \mathrm { a g e } } ) F _ { k } , 0 \right) .\tag{26}
$$

In contrast to the default program’s pressure-only attenuation, $F _ { k }$ scales the combined score. The service-age contribution is bounded by eight before this attenuation. These rules express a preference against serving

movements with heavily occupied downstream links while retaining a bounded compensation for phases not recently active.

For $k \neq k _ { 0 }$ , let $q _ { k } = \widetilde { q } _ { k }$ . Set $V _ { \operatorname* { m a x } } = \operatorname* { m a x } _ { k } V _ { k }$ when any $V _ { k } > 0$ , and $V _ { \mathrm { m a x } } = 1$ otherwise. The continuation term and current-phase priority are

$$
\begin{array} { c } { { L = 0 . 5 + 0 . 3 V _ { k _ { 0 } } { \bf 1 } ( V _ { k _ { 0 } } \geq 0 . 5 V _ { \mathrm { m a x } } ) , } } \\ { { q _ { k _ { 0 } } = \left\{ \begin{array} { l l } { { 0 . 6 ( \widetilde { q } _ { k _ { 0 } } + L ) , } } & { { V _ { k _ { 0 } } < 2 , } } \\ { { \widetilde { q } _ { k _ { 0 } } + L , } } & { { \mathrm { o t h e r w i s e } . } } \end{array} \right. } } \end{array}\tag{27}
$$

The continuation bonus uses approaching demand relative to other phases. At regular decision points, the 0.6 factor reduces the current-phase priority when fewer than two approaching vehicles are observed. At initialization, the program instead sets $q _ { k _ { 0 } } = C _ { \mathrm { c l i p } } .$ , the configured clipping bound (default $1 0 ^ { 6 } )$ , while competing phases retain $q _ { k } = \widetilde { q } _ { k }$

Phase selection Both programs combine scores and apply the clipping and nonfinite-value handling in Eq. (6), then select an argmax as in Eq. (2). The implementation resolves a tie by the first phase index. During a yellow-transition observation, the controller returns its stored action without recomputing these rules. Appendix D.1 shows the corresponding executable interfaces, and Appendix D.4 illustrates how a recorded edit replaces part of such a program.

## C.4 Revision History of the Default Program

The default program develops the fairness-aware seed through substantive revisions at iterations 35, 130, and 144. The first adds efficient pressure, downstream-load attenuation, and demand terms; later edits adjust waiting and approaching weights and strengthen service-age compensation with waiting demand. Relative to the seed, the final program reduces TT, QL, and WT by 2.24%, 5.52%, and 7.91%. Improvement is not monotonic: iteration 130 improves TT and QL but worsens WT and the combined score. The highestscoring evaluation, at iteration 156, uses code identical to iteration 144.

A separate SPA event in the same search illustrates revision guided by traffic diagnostics. Its parent report identifies peak congestion in the third quarter and a 95.5% switching rate. The revision adds efficient pressure and downstream information: TT falls from 286.51 to 280.68 s, QL from 175.17 to 167.19 vehicles, and WT from 37.20 to 34.18 s.

## Appendix D. An Example of Feedback-Guided Program Revision

This appendix follows a recorded EvoSignal revision at iteration 92 on J1. The example links the actual generation input, the recorded LLM response, and the resulting evaluation. Excerpts retain the recorded wording; [...] marks omitted material. Line wrapping, indentation, and typographic punctuation are normalized for readability. Explanatory text outside the boxes is supplied by the authors.

## D.1 Editable Program Structure

The following structural excerpt shows how the controller invokes its three editable modules. Dots omit module bodies, transition handling, shared-feature publication, diagnostics, and program-specific history updates. The names match the implementation. Evolution revises the computations within M1–M3 while retaining their interfaces and the prescribed phase-selection operation. The functions may update controller state, including the M3 gate. Detailed feature extraction and priority rules for the two selected programs are given in Appendix C.

```python
def choose_action(self, count, state):
# Transition handling and counter update omitted.
local_feats, global_feats = self.feature_extractor(state)
# Publish local features to the shared store.
priority_local = self.local_priority(local_feats)
priority_delta = self.global_modifier(global_feats, local_feats)
priority_final = priority_local + self.global_gate * priority_delta
priority_final = np.clip(priority_final,
-self.priority_clip, self.priority_clip)
priority_final = np.nan_to_num(priority_final, nan=0.0,
posinf=0.0, neginf=0.0)
self.action = int(np.argmax(priority_final))
# History updates omitted.
return self.action
# EVOLVE-BLOCK-START
def feature_extractor(self, state): # M1
... # Extract traffic features and context.
return local_feats, global_feats
def local_priority(self, local_feats): # M2
... # Compute one base priority per phase.
return priority
def global_modifier(self, global_feats, local_feats): # M3
... # Compute one adjustment per phase.
return delta
# EVOLVE-BLOCK-END
```

Feature aggregation and history construction can change in M1; coefficients, nonlinear expressions, and conditional phase-scoring rules can change in M2; network-context corrections can change in M3. These are program-design variables, rather than fixed parameters tuned within a single scoring formula.

## D.2 Input: Task and Traffic Diagnostics

Table 8 maps the prompt components to the method notation. The full input also includes parent code and reference programs; the following excerpts focus on task guidance, SPA, and PSA.

Table 8. Components of the recorded revision prompt.
<table><tr><td>Input component</td><td>Role in program revision</td></tr><tr><td>Task and interfaces  $( G _ { n } )$ </td><td>Define the control task, editable modules, and current M2 focus.</td></tr><tr><td>Parent and references  $( p _ { a } , R _ { n } )$ </td><td>Supply executable code and alternative control examples.</td></tr><tr><td>Performance and SPA  $\left( m _ { a } \right)$ </td><td>Report parent metrics, congestion patterns, and suggested changes.</td></tr><tr><td>PSA summary  $( b _ { a }$  in  $m _ { a } )$ </td><td>Describe retained strategies and their measured trade-offs.</td></tr><tr><td>Output instructions</td><td>Request explanations followed by exact SEARCH/REPLACE edits.</td></tr></table>

## System prompt: control task and revision focus

You are an expert traffic signal control engineer and algorithm designer.   
Your task is to improve a traffic signal controller called   
PriorityAgent that runs inside the CityFlow traffic simulator.   
The algorithm must be \*\*general-purpose\*\* -- it should work across different   
road networks and traffic demand patterns, not overfit to one scenario.   
[...]   
## Current Focus: PRIORITY LOGIC REFINEMENT (M2 focus)   
M1 should already provide useful features. \*\*Focus primarily on M2   
(local\_priority)\*\* -- refine how features are combined into priority.   
[...]   
- Keep M3 conservative unless you are confident coordination will help.

## SPA: parent performance and congestion feedback

[Overall] avg\_tt=277.7s | avg\_queue=163.9 | avg\_wait=36.5s | reward=0.0 | steps=120 | global\_gate   
=0.0   
[...]   
[Temporal Trend]   
Q1(0-900s)=129.4 -> Q2(900-1800s)=145.7 -> Q3(1800-2700s)=199.0 -> Q4(2700-3600s)=175.3   
Peak congestion at Q3 (1800-2700s)   
[...]   
## Mutation Suggestions (from diagnostic analysis)   
1. Congestion peaks in Q3 (1800-2700s), suggesting the algorithm struggles under sustained load.   
Consider adaptive mechanisms: increase priority sensitivity when total queue grows, or add a   
time-varying urgency multiplier.

## D.3 Input: Strategy Memory; Output: Revision Rationale

The PSA excerpt below is from the summary actually supplied to this generation. It describes an earlier strategy whose score exceeded the parent’s current score. The displayed iter -1 is a missing iteration identifier in the saved summary, not the iteration of the current revision. The archive heading and diagnostic suggestions are reproduced as recorded.

## PSA: one retained strategy

## Strategy Archive (71 evaluated, 8 on Pareto front)   
### Pareto Front (non-dominated solutions)   
Study their module designs to understand trade-offs.   
1. [iter -1] TT=276.4s QL=161.7 WT=35.6s (combined=0.3961)   
Strengths: none   
M1: Computes per-phase {phase\_pressure\_queue, phase\_pressure\_queue\_efficient,   
phase\_vehicle\_count, phase\_waiting, phase\_running, phase\_vc\_downstream,   
phase\_time\_since\_green} plus per-lane {lane\_occupancy} and temporal {time\_this\_phase}.   
M2: Computes local priority per phase as base = ph\_pqe + 0.4\*ph\_wt + 0.1\*ph\_run - 0.15\*   
ph\_vc\_down + 0.8\*max\_lane\_occ; adds starvation bonus if ph\_tsg > STARVATION\_THRESHOLD (4):   
0.5\*(ph\_tsg-4)\*(1+0.1\*ph\_wt); reduces current phase priority by factor 0.3 if ttp >=   
MAX\_GREEN\_STEPS (6).   
[...]

The response explicitly refers to the archive and proposes simplifying M1 and restoring a local-priority rule. These are the model’s recorded explanations; the code and measured outcomes are examined separately on the next page. Duration labels in the response retain the generated wording; the controller receives elapsed phase time in seconds and tests it only at scheduled decision points.

## LLM response: stated revision and planned edits

We need to revert local\_priority to the simpler, proven formula from the Pareto front --   
efficient pressure + fixed 0.4 \* waiting + 0.1 \* running - 0.15 \* downstream\_count + 0.8 \*   
maximum lane occupancy per phase, plus a linear anti-starvation bonus for phases starved > 4   
steps, and only a max-green safeguard (reduce priority by 0.3 after 6 steps). This matches   
the best-performing agents and removes all the unnecessary complexity (adaptive weight,   
spillback penalty, delta-waiting reward, urgency multiplier, min-green override) that   
degraded fitness from 0.3961 to 0.3930.   
We also simplify M1 by eliminating the temporal derivative (ph\_wt\_delta, prev\_ph\_wt),   
total\_waiting, overall\_occupancy, step\_urgency, and downstream occupancy features, keeping   
only the essential phase-level aggregates that M2 actually uses and that are required by the   
network-summary diagnostic.   
Three targeted replacements:   
1. \*\*M1 aggregation block\*\* - drop all extra computations.   
2. \*\*M1 local\_feats dict\*\* - keep only the keys needed by M2 and shared diagnostics.   
3. \*\*Entire M2 local\_priority method\*\* - replace with the simple, robust formula.

SPA suggests congestion-sensitive adaptation, while PSA supplies a simpler alternative with a stronger recorded score. In this event, the response chooses the latter direction. This illustrates how diagnostic feedback and retained strategies provide context for revision; the LLM is not required to implement every suggestion.

## D.4 Output: Code Revision and Measured Outcome

The response contains three SEARCH/REPLACE edits: two simplify M1 feature construction and one replaces M2. Below are excerpts from the original M2 and the replacement. The replacement is present in both the response and the saved child program. For compactness, the surrounding method definitions, feature retrieval, and other unchanged code are omitted.

## Recorded SEARCH/REPLACE format (M2 excerpt)

```lisp
<<<<<<< SEARCH
def local_priority(self, local_feats):
[...]
base = (ph_pqe[p] + w_wait * ph_wt[p] + 0.1 * ph_run[p]
+ 0.8 * ph_max_occ[p])
[...]
def local_priority(self, local_feats):
[...]
base = ph_pqe[p] + 0.4 * ph_wt[p] + 0.1 * ph_run[p] - 0.15 * ph_vc_down[p]
[...]
>>>>>>> REPLACE
```

The SEARCH block identifies the original method text, and the REPLACE block supplies its replacement. The recorded edit contains the complete method on each side; the displayed [...] omissions are for presentation and are not part of the executable edit. The excerpts below explain the changed computation. This intermediate revision illustrates the editing mechanism; it is distinct from the two final programs analyzed in Appendix C.

## Before: congestion-dependent weighting

congestion\_factor = min(1.0, total\_waiting / 50.0) # 0-1   
w\_wait = 0.4 + 0.4 \* congestion\_factor # 0.4 -> 0.8   
[...]   
base = (ph\_pqe[p] + w\_wait \* ph\_wt[p] + 0.1 \* ph\_run[p]   
+ 0.8 \* ph\_max\_occ[p])

After: the replacement priority computation

```python
for p in range(n):
base = ph_pqe[p] + 0.4 * ph_wt[p] + 0.1 * ph_run[p] - 0.15 * ph_vc_down[p]
max_occ = 0.0
for m in self.phase_movements.get(p, []):
if m < len(lane_occ):
max_occ = max(max_occ, lane_occ[m])
base += 0.8 * max_occ
if ph_tsg[p] > STARVATION_THRESHOLD:
starvation_bonus = 0.5 * (ph_tsg[p] - STARVATION_THRESHOLD) * (1.0 + ph_wt[p] * 0.1)
base += starvation_bonus
priority[p] = base
cur_0idx = cur - 1 if cur > 0 else 0
if 0 <= cur_0idx < n and ttp >= MAX_GREEN_STEPS:
priority[cur_0idx] *= 0.3
```

Here ph\_pqe, ph\_wt, ph\_run, and ph\_vc\_down represent efficient pressure, waiting vehicles, approaching vehicles, and downstream vehicle counts. The code adds occupancy and service-age terms, then adjusts the current phase’s priority. The recorded constants are STARVATION\_THRESHOLD = 4 and MAX\_GREEN\_STEPS = 6.

Table 9. Recorded parent and child performance for the illustrated revision.
<table><tr><td>Metric</td><td>Parent</td><td>Revised program</td></tr><tr><td>TT (s)</td><td>277.70</td><td>276.37</td></tr><tr><td>QL (veh)</td><td>163.89</td><td>161.73</td></tr><tr><td>WT (s)</td><td>36.52</td><td>35.64</td></tr><tr><td>Search score</td><td>0.392953</td><td>0.396055</td></tr></table>

All three traffic metrics improve relative to the parent. The resulting score recovers the level of an earlier archived strategy. This is a documented example of strategy reuse and program revision, rather than an isolated estimate of SPA or PSA’s causal contribution.

## Appendix E. Objective-Weight Analysis and Additional Views

Score contributions and candidate distributions For the two selected J1 programs, scoring under the default weights gives a combined TT and QL gain of about 0.00514 but a WT cost of 0.01483 when moving from default EvoSignal to TT–QL. Under the TT–QL weights, the gain becomes 0.00755 and the cost falls to 0.00198, reversing their ranking. Each comparison applies one common objective to both programs. Among evolved candidates, median TT decreases from 279.22 to 274.22 s and median QL from 165.06 to 159.77 vehicles under the TT–QL objective, while median WT increases from 35.34 to 43.86 s. These statistics describe the two recorded searches.

EvoSignal EvoSignal (TT–QL)

![](images/40e9dc33c00bb844b4cae3517e5350f959ed250f14325e2b53d5de66e7bddac6.jpg)

![](images/582ca099e34159ad2b2d552e55dddf8e1bcc9cbce820b4bcdc422b5e51dc65a9.jpg)

(c) Queue length and waiting time  
![](images/4fed84f069c939ab139c31d10b6c9c80231355d821bd44cf22672330119ade5a.jpg)

![](images/e548181cd3f4ff25de0e23659f127a365ac1598cf2bee13c525ce53bfe6d8e6f.jpg)  
Figure 8. Additional objective-space views of the default and TT–QL searches on J1: (a) three objectives; (b–d) pairwise projections over wider display ranges.

Records and display ranges The EvoSignal and TT–QL searches provide 197 and 200 valid records, respectively. Each search includes 11 initial programs, which enter the plots and non-dominated sets; the reported medians exclude initialization. A program is non-dominated if no other evaluated program in the same search is no worse on every displayed metric and strictly better on at least one. Fronts are computed jointly for the three metrics in the three-dimensional view and independently for each metric pair in the two-dimensional views. This does not establish global Pareto optimality.

Figure 7 uses TT of 270.5–286 s, QL of 153–176 vehicles, and WT of 32–51 s; its two panels show 187/188 EvoSignal records and 195/190 TT–QL records. Figure 8 expands these limits to 270–290 s, 150–185 vehicles, and 30–60 s, respectively. Panels (a)–(d) show 190/190/191/190 EvoSignal records and 197 TT–QL records each. All empirical non-dominated points and both selected programs remain visible; out-of-range records remain in the front calculations.

## Appendix F. Implementation Details and Hyperparameters

Table 10 summarizes the core settings from the archived main EvoSignal runs. The objective weights for the three evaluated configurations are defined in Section 4.1.2. The full framework enables 3M, Multi-init, SPA, and PSA; each ablation removes the corresponding component.

Table 10. Core evolution and traffic-control settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Program evolution and LLM generation</td><td></td></tr><tr><td>Evolution attempts after initialization, N</td><td>200</td></tr><tr><td>Initial programs (full / without Multi-init)</td><td>11 /1</td></tr><tr><td>LLM sampling probabilities (Flash / Pro)</td><td>0.7 / 0.3</td></tr><tr><td>Generation temperature / maximum output tokens</td><td>0.7 / 16,000</td></tr><tr><td>Population size / database archive size / islands</td><td>40 / 15 / 2</td></tr><tr><td>Elite-selection / exploitation ratios</td><td>0.2 / 0.6</td></tr><tr><td>Top / diverse reference programs in the prompt</td><td>2/1</td></tr><tr><td>Concurrent evaluations / evaluation timeout</td><td>2 / 600 s</td></tr><tr><td>Code revision format</td><td>SEARCH/REPLACE within editable blocks</td></tr><tr><td>Objective scaling and strategy memory</td><td></td></tr><tr><td>Score scales  $\left( C _ { T } , C _ { Q } , C _ { W } \right)$ </td><td>(100 s, 100 vehicles, 50 s)</td></tr><tr><td>PSA memory capacity, L</td><td>50 records</td></tr><tr><td>Maximum front / recent dominated entries shown</td><td>8/5</td></tr><tr><td>Traffic simulation and signal control</td><td></td></tr><tr><td>Simulation step / phase-decision interval</td><td>1 s / 30 s</td></tr><tr><td>Yellow transition when changing phase</td><td>5 s within the decision interval</td></tr><tr><td>Selectable phases per intersection</td><td>4</td></tr><tr><td>Evaluation horizon</td><td>3,600 s; 86,400 s for J-24h</td></tr></table>

Flash and Pro denote DeepSeek-V4-Flash and DeepSeek-V4-Pro. The database archive stores programs; the separate PSA memory retains strategy records, with limits on the front and recent entries supplied to the LLM.

The pressure–demand program is both a member of the full initial set and the sole seed for the without-Multi-init ablation.

## F.1 Initial Programs and Feature Examples

Table 11 lists the 11 experimental seeds. They were generated with LLM assistance from complementary control ideas, implemented through the same 3M interfaces, and tested for successful execution. Their names indicate design inspiration rather than complete reproductions of classical controllers.

M1: Traffic feature extraction M1 transforms movement-level observations into information useful for phase selection. For example, in the base initial program, let $B _ { k }$ denote the movements served by phase k.

Table 11. Initial control programs and their complementary design principles.
<table><tr><td>Initial program</td><td>Design emphasis</td><td>Main control idea</td></tr><tr><td>Pressure-demand</td><td>Pressure and demand</td><td>Combine queue pressure, waiting vehicles, and approaching vehicles to rank phases.</td></tr><tr><td>Efficient pressure</td><td>Pressure-based control</td><td>Use efficient queue pressure directly as a simple initial priority rule.</td></tr><tr><td>Continuation-aware pressure Phase continuation</td><td></td><td>Favor the current phase when approaching demand justifies continuation; otherwise follow pressure priorities.</td></tr><tr><td>Demand-supply balance</td><td>Demand and capacity</td><td>Combine estimated demand-supply ratios, saturation, and waiting vehicles to assess service urgency.</td></tr><tr><td>PID-inspired</td><td>Feedback control</td><td>Combine pressure, an accumulated-waiting proxy, and queue growth to respond to current and past congestion.</td></tr><tr><td>Fairness-aware</td><td>Service fairness</td><td>Increase priorities for phases not recently selected and reduce preference for a phase retained for a long period.</td></tr><tr><td>Corridor coordination</td><td>Neighbor coordination</td><td>Adjust local priorities using neighboring traffic information to encourage coordinated service.</td></tr><tr><td>Webster-inspired</td><td>Capacity-based allocation</td><td>Use estimated volume-to-capacity ratios and waiting demand to determine phase priorities.</td></tr><tr><td>Lane balancing</td><td>Lane congestion</td><td>Prioritize congested lanes while penalizing movements with high downstream occupancy.</td></tr><tr><td>Intersection adaptation</td><td>Congestion adaptation</td><td>Adjust the relative contributions of pressure and waiting vehicles according to intersection congestion.</td></tr><tr><td>Network trends</td><td>Network coordination</td><td>Modify local priorities according to dominant directional demand across the network.</td></tr></table>

Names indicate design inspiration; the programs implement these ideas within the 3M phase-priority structure.

For movement $m ,$ let $\pi _ { m }$ denote the observed queue-pressure feature, $Q _ { m }$ the waiting-vehicle count, and $V _ { m }$ the moving-vehicle demand feature. M1 aggregates them as

$$
\pi _ { k } = \sum _ { m \in { \cal B } _ { k } } \pi _ { m } , \qquad Q _ { k } = \sum _ { m \in { \cal B } _ { k } } Q _ { m } , \qquad V _ { k } = \sum _ { m \in { \cal B } _ { k } } \operatorname * { m a x } ( V _ { m } , 0 ) .\tag{28}
$$

These phase features are included in $z ,$ alongside movement-level information and the current signal state. M1 can be revised to introduce downstream congestion, demand changes, service history, or different feature combinations. The program also publishes features to a shared intersection store. Since intersections update sequentially, shared entries may come from different decision steps. Context c can include adjacency information, shared features, and network summaries.

M2: Local phase prioritization M2 assigns each phase a base score $q _ { k }$ . In the same initial program, this is a weighted combination of the features in Eq. (28):

$$
q _ { k } = \alpha \pi _ { k } + \beta Q _ { k } + \gamma V _ { k } ,\tag{29}
$$

where $\alpha , \beta ,$ , and $\gamma$ are control-rule coefficients. They are distinct from the search-objective weights $w _ { j }$ in Eq. (8). This example shows how pressure, waiting traffic, and approaching demand can influence phase priority. Evolution can revise the coefficients and replace the linear expression with nonlinear, conditional, or history-dependent rules. Thus, Eq. (29) illustrates one starting rule rather than restricting all candidates to a fixed functional form. The pressure–demand seed uses mean phase-pressure features from available neighbors for its network-adjustment rule. Setting its initial gate to zero removes this additive effect; the scaffold still computes r.

## F.2 Database Records and Evolution Updates

For each successfully evaluated program, the database stores

$$
m _ { j } = \big ( \mathbf { f } ( p _ { j } ) , S ( p _ { j } ) , d _ { j } , b _ { j } \big ) ,\tag{30}
$$

Here $d _ { j }$ and $b _ { j }$ are SPA feedback and the saved PSA summary; unavailable feedback is omitted. Code, execution status, and parent identity are also retained. Selection uses the inherited island-based database, combining elite retention with exploratory and random sampling:

$$
\left( p _ { a } , R _ { n } \right) = \operatorname { S e l e c t } ( D _ { n - 1 } ) ,\tag{31}
$$

References include high-scoring and alternative programs. A successful evaluation returns

$$
\left( \mathbf { f } ( p _ { n } ) , \tau _ { n } \right) = E ( p _ { n } ) ,\tag{32}
$$

where E uses the specified network, demand, and horizon. After forming $m _ { n } .$ , the database update is

$$
D _ { n } = U { \left( D _ { n - 1 } , ( p _ { n } , m _ { n } ) , a \right) } ,\tag{33}
$$

The operator $U$ stores the candidate and parent identity a under the database retention rules. Construction and evaluation failures retain separate status records. Workers can evaluate concurrently, so updates follow completion order.

Successful seed evaluations initialize the database through

$$
\begin{array} { r } { m ^ { ( k ) } = \big ( \mathbf { f } ( p ^ { ( k ) } ) , S ( p ^ { ( k ) } ) , \emptyset , \emptyset \big ) , \qquad D _ { 0 } = \mathrm { I n i t } \big ( \{ ( p ^ { ( k ) } , m ^ { ( k ) } ) \} _ { k = 1 } ^ { K } \big ) . } \end{array}\tag{34}
$$

Here Init inserts evaluated seeds under the same retention rules. Seeds can update PSA, but the initialization path does not save SPA reports or PSA snapshots in their database records; the empty fields reflect that implementation. Failed seed evaluations retain their execution status.

## F.3 Additional SPA Statistics and Diagnostic Rules

Using the notation in Section 3.4, SPA computes the switching rate for $T > 1$ as

$$
\nu _ { \mathrm { s w } } = { \frac { 1 } { I ( T - 1 ) } } \sum _ { i = 1 } ^ { I } \sum _ { t = 2 } ^ { T } \mathbf { 1 } ( a _ { i , t } \neq a _ { i , t - 1 } ) .\tag{35}
$$

The mean queue at intersection i and the mean network queue in temporal window W are

$$
\overline { { Q } } _ { i } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \sum _ { m } Q _ { i , t , m } , \qquad \overline { { Q } } _ { W } = \frac { 1 } { | W | } \sum _ { t \in W } \sum _ { i = 1 } ^ { I } \sum _ { m } Q _ { i , t , m } .\tag{36}
$$

The implementation compares four consecutive temporal windows and groups intersections by location. It also examines phase-score variation, demand–priority agreement, and observations following module changes. These statistics guide inspection rather than estimate causal effects. The report contains up to four revision suggestions. For example, $f _ { W } ( p _ { n } ) > 1 . 5 C _ { W }$ prompts examination of green-hold or switchingmargin rules; other rules address uneven service, demand alignment, and late congestion.

## F.4 PSA Storage and Retrieval

With module summaries $s _ { n }$ defined in Section 3.5, the archive record and memory update are

$$
\begin{array} { c } { { e _ { n } = \left( \mathbf { f } ( p _ { n } ) , \mathrm { r o u n d } ( S ( p _ { n } ) , 4 ) , s _ { n } \right) , } } \\ { { M ^ { + } = \mathrm { T a i l } _ { L } ( M \parallel e _ { n } ) . } } \end{array}\tag{37}
$$

Here ∥ appends a record and $\mathrm { T a i l } _ { L }$ retains the latest L records, with $L = 5 0$ . Records are linked to program identities, stored scores are rounded to four decimal places, and summaries are cached when identical code reappears. Successful seed evaluations also enter this memory. Given the nondominated set $F .$ , the saved text summary is

$$
b _ { n } = \operatorname { R e n d e r } \bigl ( \operatorname { T o p } _ { 8 } ( F ) , \operatorname { R e c e n t } _ { 5 } ( M ^ { + } \setminus F ) \bigr ) .\tag{38}
$$

The first operator selects up to eight nondominated records by stored scalar score in descending order. The second selects up to five recent dominated records, newest first. Rendering combines their metrics and module summaries. The summary is stored with its candidate, so a later parent revision uses that saved snapshot rather than necessarily the latest memory state.