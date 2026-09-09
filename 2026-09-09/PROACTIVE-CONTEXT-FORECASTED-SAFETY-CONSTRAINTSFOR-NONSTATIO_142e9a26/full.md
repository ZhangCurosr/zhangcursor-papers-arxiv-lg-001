# PROACTIVE CONTEXT-FORECASTED SAFETY CONSTRAINTSFOR NONSTATIONARY REINFORCEMENT LEARNING

Tim Tomashevskiy

Department of Computing and Software

McMaster University

Hamilton, Ontario, Canada

tomashet@mcmaster.ca

## ABSTRACT

Ensuring safety in reinforcement learning under nonstationarity requires anticipating changes in risk before they lead to unsafe behavior. Existing approaches typically rely on safety constraints defined at design time or updated reactively during execution, assuming that such constraints remain valid over time. However, in nonstationary environments with evolving contexts and changing driving layouts, these assumptions may fail.

We propose a framework for proactive safety constraint generation based on context forecasting. The approach infers latent environmental context from observations, predicts its future evolution, and constructs safety constraints adapted to anticipated conditions. This enables the agent to proactively avoid unsafe regions instead of reacting only after safety violations occur.

We evaluate the method in driving environments with structured context variation. The experiments include a sweep over nonstationarity intensities and additional held-out driving layouts, including highway, intersection, and racetrack scenarios. Results show that proactive constraint generation substantially reduces collisions under both seen and out-of-training nonstationarity intensities and generally remains effective across held-out driving layouts while maintaining usable task performance.

These findings suggest that context-based constraint generation is a promising approach for safe reinforcement learning under nonstationarity.

## 1 INTRODUCTION

Reinforcement learning (RL) is a general framework for learning sequential decision-making policies by interacting with an environment (Sutton & Barto, 2018). Over the past decades, RL algorithms have shown remarkable empirical success in a broad variety of applications, including robotics, control, and complex games. Nevertheless, the application of RL algorithms to real-world systems also poses significant safety issues.

In real-world applications, an RL agent is often required to meet strict safety constraints. Failure to meet these constraints can result in expensive system failures or undesired system behavior. For this reason, there is a growing interest in the research community in safe reinforcement learning, where RL agents are required to meet safety constraints both during learning and at deployment time. Some existing approaches include constrained Markov decision processes, Lyapunov-based verification, and algorithms that ensure safe exploration with high probability (Berkenkamp et al., 2017; Wachi & Sui, 2020). These methods enable agents to improve their policies while avoiding unsafe states and actions.

Most existing safe RL methods assume that the environment is stationary and that safety constraints are known in advance. However, in real-world systems, these assumptions are often violated. Operating conditions, external distur bances, and system dynamics may change over time, introducing various forms of non-stationarity into the learning problem (Chandak, 2022). As a result, safety constraints specified at design time may become outdated, overly conservative, or inconsistent with the current environment.

The problem becomes even more challenging in continual or lifelong learning, where agents interact with changing environments over extended time horizons and continuously update their policies. In such settings, fixed safety constraints may become inconsistent with evolving environmental conditions and safety requirements. Developing safety constraints that adapt to environmental change remains a major challenge.

This problem brings us to a very important question: How should one formulate safety constraints during the design phase when reinforcement learning agents are operating in environments that have the potential to change over time?

To address this problem, this paper presents a goal-oriented framework for formulating constraints in the context of continual reinforcement learning. In this framework, instead of assuming that the constraint functions are known in advance, the framework formulates safety constraints based on higher-level safety objectives. These objectives determine how the constraints can be modified based on the additional information obtained by the agent about the environment.

This formulation is referred to as the FC-Goal framework, where FC-Goal denotes Future Constraint Goals. The main idea is to formulate safety constraints based on design-time safety goals and adapt the constraints as new information about the environment becomes available. In this paper, FC-Goal is instantiated as a proactive context-forecasted safety-constraint framework: latent environmental context is extracted from observations, future context trajectories are predicted, and the predicted context is used to synthesize anticipatory safety constraints before unsafe behavior occurs.

The contributions of this paper can be summarized as follows:

• We formalize proactive safety-constraint generation for reinforcement learning under episodic nonstationarity, where constraints depend on evolving latent context.

• We introduce the FC-Goal framework, instantiated through context discovery, multi-step context forecasting, and anticipatory safety-constraint generation.

• We provide a conditional high-probability horizon-level safety analysis showing how calibrated context uncertainty, tail-safe constraint prediction, and MPC-style filtering can jointly control violation probability over a forecast horizon.

• We evaluate the method across nonstationarity intensities and held-out highway-env layouts, showing substantial collision-rate reduction with usable task performance.

The rest of this paper is organized as follows. Section 2 discusses related work in safe reinforcement learning and non-stationary learning. Section 3 explains the background of this work. Section 4 presents the problem formulation. Section 5 describes the proposed FC-Goal framework and its proactive context-forecasted safety-constraint instantiation. Section 6 explains implementation details. Appendix D analyzes the theoretical properties of the method, and Section 7 presents experimental evaluation.

## 2 RELATED WORK

Safe reinforcement learning has been studied through constrained optimization, risk-sensitive objectives, safe exploration, and robust control. Our work is most closely related to approaches that address uncertainty in safety constraints and adaptation under nonstationarity.

Safe reinforcement learning under fixed safety models. A common formulation models safety through constrained MDPs (CMDPs), solved using Lagrangian or primal–dual optimization (Altman, 1999; Achiam et al., 2017; Chow et al., 2019; Tessler et al., 2019). Related approaches use risk-sensitive criteria such as VaR and CVaR to control rare but catastrophic failures (Tamar et al., 2015; Chow et al., 2015; Prashanth, 2014). Model-based safe exploration methods further account for epistemic uncertainty using confidence bounds or Gaussian processes (Turchetta et al., 2016; Berkenkamp et al., 2016; 2017), while robust MDPs address uncertainty through ambiguity sets and worst-case optimization (Iyengar, 2005; Nilim & El Ghaoui, 2005). Although these approaches provide principled notions of safety, they typically assume that the safety model, cost structure, or uncertainty set can be specified in advance and remains valid over time.

Adaptation under nonstationarity. A separate line of work addresses nonstationarity through meta-learning, latent context inference, and online adaptation (Finn et al., 2017; Nagabandi et al., 2019; Rakelly et al., 2019; Zintgraf et al., 2020). These methods improve adaptation to changing environments, but focus primarily on performance recovery rather than proactive safety under context uncertainty.

Safe RL under nonstationarity and context-aware adaptation. Recent work combines safety with nonstationary or task-varying RL using context inference, meta-learning, or online adaptation. CASRL is one of the closest contextaware safe RL methods: it infers latent environment context with a probabilistic model and evaluates safety using uncertainty-aware trajectory sampling under prior safety constraints and nonstationary disturbances (Chen et al., 2021). However, CASRL uses inferred context mainly for safe adaptation and planning in the current regime; it does not explicitly forecast the future evolution of context and convert that forecast into a horizon of future safety constraints. Meta-safe RL methods provide reward and constraint-violation guarantees across related CMDP tasks (Khattar et al., 2024), while PEARL<sup>+</sup> improves pre-adaptation safety by regularizing the prior policy before the agent observes a new task (Wen et al., 2022). Online safe meta-RL under Markov task transitions provides mechanisms for all-time safety and sample-efficient meta-updates (Yuan et al., 2025). Safe continual RL under nonstationarity has also been surveyed and organized by constraint formulation, safety mechanism, and adaptation speed, including reactive, quick, and proactive safety adaptation (Tomashevskiy, 2026). Overall, this literature shows that context and task structure are important for safe adaptation, but existing approaches generally use inferred context or task-transition structure to adapt policies, update safety mechanisms, or provide violation guarantees for the current or newly encountered regime. In contrast, our framework explicitly forecasts latent context dynamics and uses the predicted context trajectory to synthesize anticipatory safety constraints before unsafe behavior occurs.

Positioning of our work. Our approach combines context-aware adaptation with proactive safety-constraint generation by treating safety constraints as context-dependent and explicitly modeling context evolution under episodic nonstationarity. Instead of assuming a fixed safety model, we infer latent context from observations, predict its future evolution, and generate safety constraints proactively before action selection. In this sense, the proposed framework differs from CMDP and risk-sensitive approaches that rely on static safety specifications, from safe exploration methods that focus on stationary uncertainty, and from meta-learning approaches that adapt performance or current-task safety without synthesizing safety constraints for anticipated future conditions.

## 3 BACKGROUND AND MOTIVATION

Safety in reinforcement learning is typically enforced through explicit constraint formulations. Constraints can be specified either as a safe set (reach–avoid specification) or as a constraintfunction with a threshold,

$$
f _ { C } ( s , a ) \leq \sigma .\tag{1}
$$

Safe sets are commonly used in reach–avoid and safe exploration settings (Berkenkamp et al., 2017; Hsu et al., 2021), while constraint functions are standard in CMDPs and practical safe RL (Achiam et al., 2017; García & Fernández, 2015). In most existing approaches, these constraints are defined at design time and treated as fixed.

Constraint formulations differ in the strength of safety guarantees. Hard constraints enforce safety at every time step, while expected (CMDP-style) constraints bound cumulative cost in expectation (Altman, 1999; Achiam et al., 2017). Intermediate formulations include chance constraints, which limit violation probability, and risk-sensitive criteria such as CVaR, which control tail risks (Tamar et al., 2015; Chow et al., 2015). Robust constraints further account for model uncertainty through worst-case guarantees (Iyengar, 2005; Nilim & El Ghaoui, 2005).

A key limitation of these approaches is the assumption that safety constraints remain valid after deployment. In nonstationary environments, this assumption breaks down. Constraint specifications may be incomplete at design time, and distribution shifts can invalidate learned safety relationships between states, actions, and constraint signals. As a result, fixed constraints may become either unsafe or overly conservative as conditions evolve.

These limitations motivate adaptive, context-dependent safety mechanisms that can update constraints in response to changing environments. In this work, we address this challenge by generating safety constraints proactively from predicted context, enabling the agent to anticipate and avoid unsafe regions under nonstationarity.

## 4 PROBLEM FORMULATION

We consider an agent interacting with an environment over episodes $i = 1 , 2 , \dots$ . Each episode corresponds to a stationary MDP instance $M _ { i }$ , but the environment is nonstationary across episodes:

$$
M _ { 1 } , M _ { 2 } , \ldots , M _ { i } , M _ { i + 1 } , \ldots , M _ { i } \neq M _ { i + 1 } .
$$

We assume that each $M _ { i }$ can be summarized by a latent context variable $z _ { i }$ that captures stationary properties of the episode (e.g., traffic density, aggressiveness, observation noise). Thus, $z _ { i }$ is constant within episode $i ,$ and changes between episodes. Given a context time series $z _ { 1 } , \ldots , z _ { i }$ , we assume the sequence has a transition pattern, so that $z _ { i + 1 }$ is predictable from history.

## 4.1 GOAL-BASED SAFETY SIGNAL

We assume the agent is given a high-level safety goal, while the exact low-level constraint function is unknown at design time. In driving, the natural safety goal is collision avoidance. We operationalize this goal through a clearance margin $d _ { t } ,$ which measures distance to the nearest obstacle relative to a speed-dependent safe distance. Safety requires $d _ { t } > 0 ,$ , and the constraint function is defined as a prediction of future clearance.

This goal-based view provides a practical way to define safety without enumerating all low-level requirements explicitly. Instead, the algorithm learns how clearance depends on latent context and uses forecasts of context to update constraints proactively.

## 5 METHOD OVERVIEW: PROACTIVE CONTEXT-FORECASTED CONSTRAINTS

This section describes the FC-Goal framework and its proactive context-forecasted safety-constraint instantiation. The method operationalizes the high-level idea of deriving safety constraints from evolving safety goals by using three stages: context discovery, context prediction, and constraint forecasting. We first summarize the full pipeline and then provide the implementation details in Section 6.

## 5.1 HIGH-LEVEL OVERVIEW

The proactive context-forecasted instantiation of FC-Goal is designed for episodic nonstationarity in driving. Each episode i is treated as a stationary MDP instance summarized by an episode-level latent context $z _ { i } ,$ while nonstationarity occurs through changes in context between episodes. The pipeline consists of three tiers.

Proactive Safety Foreasting Framework  
![](images/447e82f50592c499becfa314a194de3946b222655f5248f15131c4fc6500aa3d.jpg)  
Figure 1: High-level pipeline of the proactive context-forecasted safety framework. Tier 1 extracts an episode context $\hat { z } _ { i }$ , Tier 2 forecasts future contexts and forms calibrated uncertainty sets, and Tier 3 predicts multi-step tail-safe clearance constraints. An MPC-style action filter enforces robust constraints over the forecast horizon, enabling controlaware proactive safety.

Tier 1: Context Discovery. The first tier infers an episode-level context embedding $\hat { z } _ { i }$ from recent transition windows $( s _ { t } , a _ { t } , s _ { t + 1 } )$ . This representation is intended to capture stationary episode properties such as traffic density, otheragent aggressiveness, observation noise, or other latent factors that affect safety and dynamics.

Tier 2: Context Prediction. The second tier models the context time series $\{ \hat { z } _ { 1 } , \hdots , \hat { z } _ { i } \}$ and predicts an n-step future context trajectory for $n \geq 1 0$ . The forecaster outputs a probabilistic prediction $( \mu _ { i + 1 : i + n } , \Sigma _ { i + 1 : i + n } )$ . To account for context uncertainty, the forecast is calibrated using conformal prediction, yielding ellipsoidal uncertainty sets $\mathcal { Z } _ { i + k } ( \rho )$ around plausible future contexts.

Tier 3: Constraint Forecasting and Enforcement. The third tier converts the predicted context trajectory into a horizon of safety constraints. In the driving setting, this is done by forecasting conservative tail estimates of future clearance, such as lower quantiles or CVaR-style margins, and forming proactive constraints $c _ { t + k } ^ { \prime } = \hat { d } _ { t + k } ^ { ( q ) } - \epsilon$ for $k = 1 , \ldots , n$ . These predicted constraints are then enforced through a control-aware MPC-style safety filter. Candidate action sequences are rolled out with a fast ego model, and only actions satisfying robust multi-step constraints under the calibrated context uncertainty set are accepted. If no feasible action is found, the filter executes a conservative fallback action.

This pipeline turns context prediction into actionable safety decisions: rather than waiting for violations or adapting only after a shift is observed, the agent uses predicted context evolution to synthesize anticipatory safety constraints before unsafe behavior occurs.

## 6 IMPLEMENTATION DETAILS

This section describes an implementable instantiation of the proposed framework for driving under episodic nonstationarity. We use i to index episodes and t to index time steps within an episode. Each episode i corresponds to a stationary MDP instance characterized by an episode-level latent context $z _ { i } ,$ while nonstationarity occurs only across episodes through changes in $z _ { i }$ . The method consists of three tiers: Tier 1 (context extraction), Tier 2 (multi-step context forecasting with conformal ellipsoids and meta-learning updates), and Tier 3 (tail-safe constraint forecasting). Predicted constraints are enforced via a control-aware MPC-style safety filter. The prediction horizon is $n \geq 1 0$

## 6.1 TIER 1: CONTEXT EXTRACTION VIA REPRESENTATION LEARNING

Inputs and outputs. Tier 1 receives short transition windows $\tau _ { i } = \{ ( s _ { t } , a _ { t } , s _ { t + 1 } ) \} _ { t = 0 } ^ { K - 1 }$ collected at the beginning of episode i. Its output is an episode-level context embedding $\hat { z } _ { i } \in \mathbb { R } ^ { d } \ ( \mathrm { t y p i c a l l y } \ d \in \ [ 8 , 3 2 ] )$ , which is treated as constant within the episode. In driving, zˆ<sub>i</sub> is intended to capture stationary episode properties such as traffic density, other-agent aggressiveness, and observation noise.

Architecture. For vector observations, we implement the context encoder $g _ { \phi }$ as a GRU (or Transformer) over the transition sequence, followed by an MLP head producing $\hat { z } _ { i }$ . For image observations, we prepend a convolutional backbone to encode each $s _ { t }$ before the sequence model. Optionally, the encoder outputs Gaussian parameters $( \mu _ { i } ^ { z } , \Sigma _ { i } ^ { z } )$ , and we set $\hat { z } _ { i } = \mu _ { i } ^ { z }$

Training objective. We train Tier 1 jointly with a context-conditioned dynamics model $p _ { \theta } ( s _ { t + 1 } \mid s _ { t } , a _ { t } , \hat { z } _ { i } )$ to ensure that the learned context is predictive of the episode dynamics. We use a next-state prediction loss (Gaussian NLL or MSE) and an episode-consistency regularizer:

$$
\mathcal { L } _ { \mathrm { T i e r l } } ( \phi , \theta ) = - \sum _ { t } \log p _ { \theta } ( s _ { t + 1 } \mid s _ { t } , a _ { t } , \hat { z } _ { i } ) + \lambda _ { \mathrm { c o n s } } \sum _ { u , v \in i } \| \hat { z } _ { i } ^ { ( u ) } - \hat { z } _ { i } ^ { ( v ) } \| _ { 2 } ^ { 2 } ,\tag{2}
$$

where $\hat { z } _ { i } ^ { ( u ) }$ and $\hat { z } _ { i } ^ { ( v ) }$ are computed from different windows within the same episode. This encourages $\hat { z } _ { i }$ to encode stationary MDP properties rather than transient state variation.

## 6.2 TIER 2: MULTI-STEP CONTEXT FORECASTING WITH REGRET FEEDBACK

Tier 2 models the time series of extracted contexts $\{ \hat { z } _ { 1 } , \dots , \hat { z } _ { i } \}$ and predicts a multi-step forecast for future episode contexts. We implement the forecaster $h _ { \psi }$ as a GRU/Transformer over the last L contexts, producing Gaussian predictions

$$
\begin{array} { r } { ( \mu _ { i + 1 : i + n } , \Sigma _ { i + 1 : i + n } ) = h _ { \psi } ( \widehat { z } _ { i - L + 1 : i } ) , } \end{array}
$$

where Σ is diagonal for efficiency. We train the forecaster using a discounted multi-step prediction loss:

$$
\mathcal L ^ { ( z ) } ( \psi ) = \sum _ { k = 1 } ^ { n } w _ { k } \| \mu _ { i + k } - \hat { z } _ { i + k } \| _ { 2 } ^ { 2 } , \qquad w _ { k } = \gamma ^ { k - 1 } .\tag{3}
$$

Horizon prediction and regret. Tier 2 predicts contexts for $n \geq 1 0$ steps ahead. At the start of episode i + 1, Tier 1 extracts the realized context $\hat { z } _ { i + 1 }$ , and Tier 2 computes a regret signal

$$
r _ { i + 1 } ^ { ( z ) } = \| \mu _ { i + 1 } - \hat { z } _ { i + 1 } \| .\tag{4}
$$

This regret is used to track forecasting performance and to drive online adaptation.

Off-policy meta-learning updates. To enable rapid adaptation to changing context-transition patterns, Tier 2 is updated using off-policy meta-learning on replayed episode subsequences stored in an episode buffer $\mathcal { D } = \{ \hat { z } _ { i } \}$ . Each meta-task samples a short subsequence and constructs a support/query split: the support set emulates limited new evidence (e.g., one newly observed episode), while the query set evaluates multi-step forecasting performance. We apply a MAML/Reptile-style update to learn an initialization of ψ that reduces regret after regime changes.

## 6.3 CONFORMAL CALIBRATION: ELLIPSOIDAL CONTEXT UNCERTAINTY SETS

Context forecasts are uncertain, and downstream safety evaluation should remain valid when the predicted context is slightly wrong. We therefore construct calibrated uncertainty sets around the Tier 2 forecast. For each calibration episode, we compute the Mahalanobis score

$$
\begin{array} { r } { s _ { i + 1 } = ( \hat { z } _ { i + 1 } - \mu _ { i + 1 } ) ^ { \top } \Sigma _ { i + 1 } ^ { - 1 } ( \hat { z } _ { i + 1 } - \mu _ { i + 1 } ) , } \end{array}\tag{5}
$$

and choose $\rho ^ { 2 }$ as the $( 1 - \alpha )$ quantile of scores stored in a calibration buffer C. The resulting calibrated ellipsoid for episode $i + k$ is

$$
\begin{array} { r } { \mathcal { Z } _ { i + k } ( \rho ) = \left\{ z : ( z - \mu _ { i + k } ) ^ { \top } \Sigma _ { i + k } ^ { - 1 } ( z - \mu _ { i + k } ) \leq \rho ^ { 2 } \right\} . } \end{array}\tag{6}
$$

In practice, we update $\rho$ using a sliding window of recent episodes to track slow drift while maintaining empirical coverage.

## 6.4 TIER 3: TAIL-SAFE CONSTRAINT FORECASTING (QUANTILES/CVAR)

Driving safety signal. We define the safety objective as collision avoidance and operationalize it using a clearance margin relative to a speed-dependent safe distance:

$$
d _ { t } \ = \ \operatorname* { m i n } _ { j \in { \mathcal { O } } _ { t } } \Big ( \mathrm { d i s t } ( \mathrm { e g o } , j ) - \big ( d _ { 0 } + h v _ { t } \big ) \Big ) ,\tag{7}
$$

where ${ \mathcal { O } } _ { t }$ is the set of surrounding vehicles and $v _ { t }$ is ego speed. A safety violation occurs when $d _ { t } \leq 0$

Quantile forecasting. Tier 3 predicts a conservative lower-quantile of future clearance over horizon n:

$$
\hat { d } _ { t + 1 : t + n } ^ { ( q ) } = f _ { \omega } ( x _ { t } , \hat { z } _ { i } ) ,
$$

where $x _ { t }$ includes ego kinematics, lane geometry features, and pooled obstacle features (e.g., attention-based set encoding). The quantile predictor is trained using the pinball loss:

$$
\mathcal { L } _ { \mathrm { T i e r 3 } } ( \omega ) = \sum _ { k = 1 } ^ { n } w _ { k } \rho _ { q } \Big ( d _ { t + k } - \hat { d } _ { t + k } ^ { ( q ) } \Big ) , \qquad \rho _ { q } ( u ) = u \left( q - \mathbb { I } \{ u < 0 \} \right) .\tag{8}
$$

We then define proactive constraints as

$$
c _ { t + k } ^ { \prime } = \hat { d } _ { t + k } ^ { ( q ) } - \epsilon , \qquad k = 1 , \dots , n ,\tag{9}
$$

so that $c _ { t + k } ^ { \prime } > 0$ enforces a tail-safe clearance margin with quantile level q and safety margin ϵ.

CVaR alternative. For higher conservatism, Tier 3 can be extended to a CVaR formulation by predicting a distribution (or samples) of $d _ { t + k }$ and enforcing $\mathrm { C V a R } _ { \alpha } ( d _ { t + k } ) > \epsilon$ . In this work, quantile prediction provides a simple and effective tail-risk baseline.

## 6.5 CONTROL-AWARE ENFORCEMENT VIA MPC-STYLE SAFETY FILTERING

Predicted constraints are enforced using an MPC-style safety filter. At each step t, we sample M candidate control sequences $\mathbf { a } _ { t : t + H - 1 }$ (steering/throttle/brake) and roll out a fast ego dynamics model (kinematic bicycle) to obtain predicted states $s _ { t + 1 : t + n } ^ { \prime } .$ . For each candidate, we evaluate robust multi-step safety under context uncertainty:

$$
\operatorname* { m i n } _ { k \le n } \operatorname* { i n f } _ { z \in { \mathcal Z } _ { i + k } ( \rho ) } \left( \hat { d } ^ { ( q ) } ( s _ { t + k } ^ { \prime } , z ) - \epsilon \right) \ge 0 .\tag{10}
$$

The inner infimum is approximated using a small set of adversarial samples on the ellipsoid boundary. Among feasible candidates, we select the sequence maximizing progress and smoothness; if none is feasible, we execute a conservative fallback action (e.g., braking while maintaining lane).

## 6.6 END-TO-END EXECUTION LOOP

At a high level, each episode proceeds as follows: (i) Tier 1 extracts the episode-level context $\hat { z } _ { i }$ from an initial transition window; (ii) Tier 2 forecasts future contexts and constructs conformal uncertainty sets; (iii) at each step, Tier 3 predicts tail-safe clearance constraints, and an MPC-style safety filter selects a feasible action under worst-case context realizations; (iv) data is stored and all tiers are periodically updated using replay buffers.

The full implementation pseudocode is provided in Appendix A.

## 7 EXPERIMENTS

We evaluate the proposed safety framework on merge-v0 from highway-env. The main goal is to assess whether context-based safety constraints reduce unsafe behavior under stationary and increasingly nonstationary dynamics, while preserving task performance. The evaluation uses a sweep over switching frequencies, reports safety and taskperformance metrics together, includes a compact component analysis, and provides full held-out-layout results in Appendix C. We additionally evaluate highway-v0, intersection-v0, and racetrack-v0 as held-out stress-test layouts to assess whether the safety layer remains effective beyond the main merge-v0 benchmark.

## 7.1 SETUP AND METRICS

We train DQN and PPO agents with three random seeds on the main merge-v0 benchmark. Nonstationarity is controlled by $p _ { \mathrm { s t a y } }$ , the probability that the current environment context persists to the next episode; $p _ { \mathrm { s t a y } } ~ = ~ 1 . 0$ corresponds to the stationary setting, while smaller values indicate more frequent context changes. We evaluate $p _ { \mathrm { s t a y } } \in \{ 1 . 0 , 0 . 9 5 , 0 . 8 5 , 0 . 7 0 , 0 . 5 0 \}$ . Unless otherwise stated, model selection and hyperparameter tuning use the stationary and milder switching regimes up to $p _ { \mathrm { s t a y } } = 0 . 8 5$ , while $p _ { \mathrm { s t a y } } = 0 . 7 0$ and $p _ { \mathrm { s t a y } } = 0 . 5 0$ are treated as out-oftraining nonstationarity intensities. For each algorithm and nonstationarity level, we compare the unconstrained agent (safety off) with the full context-based safety mechanism (safety on).

The primary metric is collision rate. We also report final reward as a task-performance metric and minimum distance as an auxiliary proximity diagnostic. Minimum distance should not be interpreted as the sole safety criterion: a policy can reduce collisions while allowing smaller clearances in some maneuvers, reflecting a trade-off between strict collision avoidance, mobility, and flexibility.

## 7.2 RESULTS ACROSS NONSTATIONARITY LEVELS

Figure 2 shows that the unconstrained baselines degrade as context switches become more frequent: DQN collision rate increases from 0.0555 at $p _ { \mathrm { s t a y } } ~ = ~ 1 . 0$ to 0.1912 at $p _ { \mathrm { s t a y } } ~ = ~ 0 . 5 0$ , and PPO increases from 0.0544 to 0.1656. In contrast, the safety-enabled agents remain close to zero over the same sweep. At $p _ { \mathrm { s t a y } } = 0 . 7 0$ , DQN drops from 0.1883 to 0.0020, and PPO drops from 0.1621 to 0.0021. Reward changes are modest relative to the collision-rate reduction, while minimum distance is mixed; we therefore interpret minimum distance as a diagnostic of proximity and mobility rather than as the primary safety target.

Figure 3 summarizes the relative changes between safety-on and safety-off runs. Across all evaluated settings, enabling safety reduces collision rates by more than 90% and usually by more than 97%. This supports the central empirical claim that context-based constraints primarily improve the target safety outcome while preserving usable task performance.

![](images/7cf5128984fccb36c8d1d96e6fc370b0b47970ab2c483062449df2e54e751225.jpg)  
Figure 2: Effect of increasing nonstationarity on the main merge-v0 benchmark. The horizontal axis shows $p _ { \mathrm { s t a y } }$ where lower values indicate more frequent context switching. Collision rate is the primary safety metric: unconstrained agents become substantially more collision-prone as nonstationarity increases, while safety-enabled agents remain near zero. Error bars denote standard deviation across three seeds.

![](images/ae0635515fc6d05d8d06596f78416814e24ec31f39f1babf02f226e078b47cd7.jpg)  
Figure 3: Relative effect of enabling safety on the main merge-v0 benchmark. Collision-rate reduction is consistently large, while reward and minimum-distance changes are smaller and mixed, indicating a realistic safety–mobility tradeoff rather than uniform improvement on all metrics.

## 7.3 HELD-OUT LAYOUT STRESS TEST

We further evaluate the safety layer on held-out highway-env layouts: highway-v0, intersection-v0, and racetrack-v0. These layouts test whether the same safety mechanism remains effective beyond the main merge-v0 benchmark and under different interaction geometries. Across the evaluated nonstationarity levels, the safety-enabled method reduces collision rates relative to the unconstrained baseline. Full per-layout results are reported in Appendix C.

## 7.4 DISCUSSION AND LIMITATIONS

The results strengthen the empirical evidence by evaluating multiple nonstationarity levels, reporting both DQN and PPO, separating collision reduction from auxiliary proximity diagnostics, adding held-out-layout stress tests, and including a compact component analysis. Figure 2 shows that decreasing $p _ { \mathrm { s t a y } }$ increases collision rates for unconstrained agents, while the safety-enabled method remains effective even under out-of-training nonstationarity intensities. The held-out-layout stress test further suggests that the safety layer is not restricted to the main merge-v0 layout.

<table><tr><td>Variant</td><td>Collision rate ↓</td><td>Final reward ↑</td><td>Min. distance ↑</td></tr><tr><td>No safety</td><td>17.52%</td><td>10.47</td><td>19.36</td></tr><tr><td>Fixed constraint</td><td>6.72%</td><td>9.89</td><td>27.52</td></tr><tr><td>Context-only constraint</td><td>2.74%</td><td>10.91</td><td>16.33</td></tr><tr><td>Forecasting without conformal calibration</td><td>2.03%</td><td>10.51</td><td>15.75</td></tr><tr><td>Full safety method</td><td>0.21%</td><td>11.73</td><td>13.78</td></tr></table>

Table 1: Component analysis under strong nonstationarity on the main merge-v0 benchmark $( p _ { \mathrm { s t a y } } ~ = ~ 0 . 7 0 )$ , aggregated across DQN and PPO. Collision rate is the primary safety metric; reward and minimum distance quantify mobility/proximity trade-offs. The full method achieves the lowest collision rate among all variants while maintaining usable reward; the minimum distance reflects the expected mobility/proximity trade-off.

Table 1 shows that fixed constraints already reduce collisions relative to no safety, but context-dependent and forecasted constraints reduce collisions further. This supports the claim that the benefit is not merely due to adding a static safety layer, but comes from adapting constraints to changing context. At the same time, reward and minimum distance show a realistic safety–mobility trade-off: the goal of the method is to prevent collisions rather than maximize clearance in every maneuver.

The evaluation remains limited in the range of nonstationarity mechanisms considered. Although the experiments vary the intensity of context switching through $p _ { \mathrm { s t a y } }$ , they do not systematically isolate which part of the context changes. In our formulation, visible and latent contexts jointly describe the state; future work should therefore distinguish shifts in visible context, shifts in latent context, and simultaneous shifts in both. These context changes may correspond to different statistical forms of distribution shift, such as covariate shift, label shift, or concept shift. Future work should study how each type of context shift affects the validity, conservativeness, and responsiveness of context-dependent constraints, especially when multiple shifts occur simultaneously.

## 8 CONCLUSION

This paper studied proactive safety constraint generation for reinforcement learning under episodic nonstationarity. The key idea is to treat safety constraints as context-dependent objects rather than fixed design-time specifications. By extracting latent context, forecasting its future evolution, and enforcing tail-safe clearance constraints through a control-aware safety filter, the proposed framework aims to prevent violations before they occur rather than only reacting after unsafe behavior is observed

The experiments show that safety-enabled agents substantially reduce collision rates across a sweep of nonstationarity levels. As $p _ { \mathrm { s t a y } }$ decreases, unsafe baselines generally become more collision-prone, while the safety-enabled runs maintain much lower collision rates for both DQN and PPO. The auxiliary reward and minimum-distance metrics indicate that this improvement is accompanied by a realistic safety–mobility trade-off rather than a uniform improvement on every metric. Additional held-out driving layouts, including highway, intersection, and racetrack scenarios, provide a stronger stress test than varying the switching intensity alone and support the more precise claim that the approach remains useful under both out-of-training nonstationarity intensities and held-out environment layouts.

These results support the central claim that context-dependent safety constraints are useful for RL systems operating in changing environments. At the same time, the current evaluation is not exhaustive. Future work should include larger-scale per-algorithm ablations, stronger reactive and fixed-constraint baselines, and more detailed diagnostics of context forecasting, calibration coverage, and intervention frequency.

## REFERENCES

Joshua Achiam, David Held, Aviv Tamar, and Pieter Abbeel. Constrained policy optimization. In International Conference on Machine Learning (ICML), 2017.

Eitan Altman. Constrained Markov decision processes. CRC press, 1999.

Felix Berkenkamp, Riccardo Moriconi, Angela P Schoellig, and Andreas Krause. Safe learning of regions of attraction for uncertain, nonlinear systems with gaussian processes. In 2016 IEEE 55th Conference on Decision and Control (CDC), pp. 4661–4666. IEEE, 2016.

Felix Berkenkamp, Matteo Turchetta, Angela P. Schoellig, and Andreas Krause. Safe model-based reinforcement learning with stability guarantees. In Advances in Neural Information Processing

Systems, volume 30, 2017. URL https://proceedings.neurips.cc/paper/2017/hash/ 766ebcd59621e305170616ba3d3dac32-Abstract.html.

Yash Chandak. Reinforcement Learning for Non-stationary problems. PhD thesis, University of Massachusetts Amherst, 2022.

Baiming Chen, Zuxin Liu, Jiacheng Zhu, Mengdi Xu, Wenhao Ding, Liang Li, and Ding Zhao. Context-aware safe reinforcement learning for non-stationary environments. In 2021 IEEE International Conference on Robotics and Automation (ICRA), pp. 10689–10695. IEEE, 2021.

Yinlam Chow, Aviv Tamar, Shie Mannor, and Marco Pavone. Risk-sensitive and robust decisionmaking: a CVaR optimization approach. In Advances in Neural Information Processing Systems, volume 28, pp. 1522–1530, 2015. URL https://proceedings.neurips.cc/paper/2015/hash/ 64223ccf70bbb65a3a4aceac37e21016-Abstract.html.

Yinlam Chow, Ofir Nachum, Aleksandra Faust, Edgar Duenez-Guzman, and Mohammad Ghavamzadeh. Lyapunovbased safe policy optimization for continuous control. arXiv preprint arXiv:1901.10031, 2019. URL https: //arxiv.org/abs/1901.10031.

Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pp. 1126–1135. PMLR, 2017.

Javier García and Fernando Fernández. A comprehensive survey on safe reinforcement learning. Journal ofMachine Learning Research, 16(42):1437–1480, 2015.

Kai-Chieh Hsu, Vicenç Rúbies-Royo, Claire J. Tomlin, and Jaime F. Fisac. Safety and liveness guarantees through reach-avoid reinforcement learning. In Robotics: Science and Systems (RSS), 2021. doi: 10.15607/RSS.2021.XVII. 077. URL https://arxiv.org/abs/2112.12288. arXiv:2112.12288.

Garud N. Iyengar. Robust dynamic programming. Mathematics ofOperations Research, 30(2):257–280, 2005. doi: 10. 1287/moor.1040.0129. URL https://pubsonline.informs.org/doi/10.1287/moor.1040.0129.

Vanshaj Khattar, Yuhao Ding, Bilgehan Sel, Javad Lavaei, and Ming Jin. A cmdp-within-online framework for metasafe reinforcement learning. arXiv preprint arXiv:2405.16601, 2024.

Anusha Nagabandi, Ignasi Clavera, Simin Liu, Ronald S. Fearing, Pieter Abbeel, Sergey Levine, and Chelsea Finn. Learning to adapt in dynamic, real-world environments through meta-reinforcement learning. In International Conference on Learning Representations, 2019. URL https://arxiv.org/abs/1803.11347.

Arnab Nilim and Laurent El Ghaoui. Robust control of markov decision processes with uncertain transition matri ces. Operations Research, 53(5):780–798, 2005. doi: 10.1287/opre.1050.0216. URL https://pubsonline. informs.org/doi/10.1287/opre.1050.0216.

L. A. Prashanth. Policy gradients for CVaR-constrained MDPs. In Algorithmic Learning Theory, volume 8776 of Lecture Notes in Computer Science, pp. 155–169. Springer, 2014. doi: 10.1007/978-3-319-11662-4\_12. URL https://link.springer.com/chapter/10.1007/978-3-319-11662-4\_12.

Kate Rakelly, Aurick Zhou, Chelsea Finn, Sergey Levine, and Deirdre Quillen. Efficient off-policy meta-reinforcement learning via probabilistic context variables. In International conference on machine learning, pp. 5331–5340. PMLR, 2019.

Richard S Sutton and Andrew G Barto. Reinforcement learning: An introduction. MIT press, 2 edition, 2018.

Aviv Tamar, Yinlam Chow, Mohammad Ghavamzadeh, and Shie Mannor. Policy gradient for coherent risk measures. In Advances in Neural Information Processing Systems, volume 28, pp. 1468–1476, 2015. URL https://proceedings.neurips.cc/paper/2015/hash/ 024d7f84fff11dd7e8d9c510137a2381-Abstract.html.

Chen Tessler, Daniel J. Mankowitz, and Shie Mannor. Reward constrained policy optimization. In International Conference on Learning Representations (ICLR), 2019. URL https://openreview.net/forum?id= SkfrvsA9FX.

Timofey Tomashevskiy. Safe continual reinforcement learning methods for nonstationary environments. towards a survey of the state of the art. arXiv preprint arXiv:2601.05152, 2026.

Matteo Turchetta, Felix Berkenkamp, and Andreas Krause. Safe exploration in finite markov decision processes with gaussian processes. Advances in neural information processing systems, 29, 2016.

Akifumi Wachi and Yanan Sui. Safe reinforcement learning in constrained markov decision processes. In International Conference on Machine Learning, pp. 9797–9806. PMLR, 2020.

Lu Wen, Songan Zhang, H Eric Tseng, Baljeet Singh, Dimitar Filev, and Huei Peng. Improved robustness and safety for pre-adaptation of meta reinforcement learning with prior regularization. In 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 8987–8994. IEEE, 2022.

Zhenyuan Yuan, Siyuan Xu, and Minghui Zhu. All-time safety and sample-efficient meta update for online safe meta reinforcement learning under markov task transition. Machine Learning, 114(8):173, 2025.

Luisa Zintgraf, Kyriacos Shiarlis, Maximilian Igl, Sebastian Schulze, Yarin Gal, Katja Hofmann, and Shimon Whiteson. VariBAD: A very good method for bayes-adaptive deep RL via meta-learning. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=Hkl9JlBYvr.

## APPENDIX

A ALGORITHM DETAILS

A.1 END-TO-END PROCEDURE

Algorithm 1 Proactive Context-Based Safety Constraint Generation   
Require: Forecast horizon $n ,$ context encoder $g _ { \phi } ,$ context forecaster $h _ { \psi } .$ , constraint forecaster $f _ { \omega } ,$ safety margin $\epsilon ,$   
conformal radius $\rho$   
1: for episode $i = 1 , 2 , \dots$ do   
2: Collect an initial transition window $\tau _ { i } = \{ ( s _ { t } , a _ { t } , s _ { t + 1 } ) \}$   
3: Infer episode context ${ \hat { z } } _ { i } \gets g _ { \phi } ( \tau _ { i } )$   
4: Forecast future contexts $( \mu _ { i + 1 : i + n } , \Sigma _ { i + 1 : i + n } )  h _ { \psi } ( \hat { z } _ { 1 : i } )$   
5: Construct calibrated context uncertainty sets $\mathcal { Z } _ { i + 1 : i + n } ( \rho )$   
6: for step $t = 0 , \ldots , T _ { i } - 1$ do   
7: Observe current state $s _ { t }$   
8: Predict tail-safe future clearances $\hat { d } _ { t + 1 : t + n } ^ { ( q ) } \gets f _ { \omega } ( s _ { t } , \hat { z } _ { i } )$   
9: Form proactive constraints $c _ { t + k } ^ { \prime } = \hat { d } _ { t + k } ^ { ( q ) } - \epsilon ,$ , for $k = 1 , \dots , n$   
10: Sample candidate action sequences $a _ { t : t + H - 1 } ^ { ( m ) } , m = 1 , \dots , M$   
11: Roll out candidate sequences using a fast ego-dynamics model   
12: Keep only candidates satisfying   
$\operatorname* { m i n } _ { k \le n } \operatorname* { i n f } _ { z \in \mathcal { Z } _ { i + k } ( \rho ) } \left( \hat { d } _ { t + k } ^ { ( q ) } ( z ) - \epsilon \right) \ge 0$   
13: if at least one feasible candidate exists then   
14: Execute the first action of the feasible sequence with best progress/smoothness score   
15: else   
16: Execute conservative fallback action   
17: end if   
18: Store transition, realized clearance $d _ { t + 1 }$ , and violation indicator   
19: end for   
20: Extract realized next context $\hat { z } _ { i + 1 }$ when available   
21: Update context forecaster using prediction error $\| \mu _ { i + 1 } - \hat { z } _ { i + 1 } \|$   
22: Update constraint forecaster using clearance prediction error   
23: Update conformal calibration buffer and radius $\rho$   
24: end for

## A.2 CONSTRAINT FORMULATION

We define buffered constraints as:

$$
c _ { t + 1 } = d _ { t + 1 } + b _ { t + 1 } ,\tag{11}
$$

where $b _ { t + 1 } \leq 0$ is a signed conservative correction accounting for uncertainty due to prediction error, context shift, and epistemic uncertainty.

A generic formulation is:

$$
b _ { t + 1 } = - B ( \mathcal { U } _ { d } , \mathcal { U } _ { z } , \mathcal { U } _ { \mathrm { e p i } } , \mathcal { V } ) , \qquad B ( \cdot ) \geq 0 ,\tag{12}
$$

where these terms represent distance uncertainty, context uncertainty, epistemic uncertainty, and violation history, respectively. Thus, larger uncertainty produces a more negative correction and therefore a more conservative buffered clearance constraint.

## A.3 LEARNING SIGNALS

We define:

$$
r _ { i + 1 } ^ { ( z ) } = \| \mu _ { i + 1 } - \hat { z } _ { i + 1 } \| ,
$$

$$
r _ { t + 1 } ^ { ( c ) } = | c _ { t + 1 } ^ { \prime } - c _ { t + 1 } | .\tag{13}
$$

(14)

These signals are used to update context forecasting and constraint prediction models.

## B ADDITIONAL EXPERIMENTAL DIAGNOSTICS

The main paper emphasizes compact visual summaries. Table 2 provides the full numerical results used to generate Figures 2 and 3. These values are included in the appendix to keep the main experimental section focused. Figures 4, 5, and 6 split the appendix metric sweep into collision rate, final reward, and minimum distance, respectively.
<table><tr><td>Setting</td><td>Algo</td><td>Safety</td><td></td><td>Collision rate (%) ↓ Minimum distance ↑</td><td>Final reward ↑</td></tr><tr><td>Stationary  $( p _ { \mathrm { s t a y } } = 1 . 0 0 ) $ </td><td>DQN</td><td>off</td><td> $\overline { { 5 . 5 5 \pm 0 . 2 8 0 } }$ </td><td> $\overline { { 1 2 . 0 8 \pm 0 . 4 2 } }$ </td><td> $\overline { { 1 5 . 3 5 \pm 0 . 2 0 8 } }$ </td></tr><tr><td>Stationary  $( p _ { \mathrm { s t a y } } = 1 . 0 0 ) $ </td><td>DQN</td><td>on</td><td> $0 . 2 1 \pm 0 . 0 0 5$ </td><td> $1 2 . 0 1 \pm 0 . 0 6$ </td><td> $1 5 . 9 3 \pm 0 . 2 6 0$ </td></tr><tr><td>Stationary  $( p _ { \mathrm { s t a y } } = 1 . 0 0 ) $ </td><td>PPO</td><td>off</td><td> $5 . 4 4 \pm 0 . 4 3 0$ </td><td> $1 7 . 8 5 \pm 0 . 8 4$ </td><td> $1 2 . 4 2 \pm 0 . 1 9 1$ </td></tr><tr><td>Stationary  $( p _ { \mathrm { s t a y } } = 1 . 0 0 ) $ </td><td>PPO</td><td>on</td><td> $0 . 2 1 \pm 0 . 0 0 4$ </td><td> $1 7 . 5 8 \pm 0 . 6 7$ </td><td> $1 3 . 9 3 \pm 0 . 1 2 1$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 9 5 )$ </td><td>DQN</td><td>off</td><td> $\overline { { 9 . 3 4 \pm 0 . 3 7 0 } }$ </td><td> $\overline { { 1 9 . 8 2 \pm 0 . 7 9 } }$ </td><td> $\overline { { 1 4 . 5 2 \pm 0 . 0 6 9 } }$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 9 5 )$ </td><td>DQN</td><td>on</td><td> $0 . 1 2 \pm 0 . 0 0 3$ </td><td> $1 5 . 1 1 \pm 0 . 0 8$ </td><td> $1 3 . 7 1 \pm 0 . 5 2 0$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 9 5 )$ </td><td>PPO</td><td>off</td><td> $5 . 7 2 \pm 0 . 5 1 0$ </td><td> $1 8 . 1 6 \pm 0 . 6 9$ </td><td> $1 3 . 0 7 \pm 0 . 7 1 0$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 9 5 )$ </td><td>PPO</td><td>on</td><td> $0 . 2 8 \pm 0 . 0 1 0$ </td><td> $1 2 . 7 4 \pm 0 . 5 9$ </td><td> $1 3 . 7 2 \pm 0 . 4 1 6$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 8 5 )$ </td><td>DQN</td><td>off</td><td> $\overline { { 9 . 9 0 \pm 0 . 5 0 0 } }$ </td><td> $\overline { { 1 3 . 7 8 \pm 0 . 6 9 } }$ </td><td> $\overline { { 1 3 . 8 6 \pm 0 . 1 5 6 } }$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 8 5 )$ </td><td>DQN</td><td>on</td><td> $0 . 7 4 \pm 0 . 0 0 5$ </td><td> $1 2 . 5 5 \pm 0 . 0 9$ </td><td> $1 3 . 9 3 \pm 0 . 0 6 9$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 8 5 )$ </td><td>PPO</td><td>off</td><td> $1 0 . 9 5 \pm 0 . 5 6 0$ </td><td> $1 7 . 3 9 \pm 1 . 0 1$ </td><td> $1 2 . 4 2 \pm 0 . 0 4 2$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 8 5 )$ </td><td>PPO</td><td>on</td><td> $0 . 3 1 \pm 0 . 0 0 3$ </td><td> $1 8 . 1 7 \pm 0 . 7 6$ </td><td> $1 3 . 9 3 \pm 0 . 2 4 2$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 7 0 )$ </td><td>DQN</td><td>off</td><td> $\overline { { 1 8 . 8 3 \pm 1 . 8 1 0 } }$ </td><td> $\overline { { 2 3 . 6 9 \pm 1 . 6 8 } }$ </td><td> $\overline { { 1 1 . 2 1 \pm 0 . 0 6 9 } }$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 7 0 )$ </td><td>DQN</td><td>on</td><td> $0 . 2 0 \pm 0 . 0 0 6$ </td><td> $1 5 . 7 3 \pm 0 . 1 4$ </td><td> $1 2 . 5 6 \pm 0 . 1 0 4$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 7 0 )$ </td><td>PPO</td><td>off</td><td> $1 6 . 2 1 \pm 1 . 1 3 0$ </td><td> $1 5 . 0 3 \pm 1 . 0 8$ </td><td> $9 . 7 3 \pm 0 . 2 2 5$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 7 0 )$ </td><td>PPO</td><td>on</td><td> $0 . 2 1 \pm 0 . 0 0 3$ </td><td> $1 1 . 8 3 \pm 0 . 3 4$ </td><td> $1 0 . 8 9 \pm 0 . 1 0 4$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 5 0 )$ </td><td>DQN</td><td>off</td><td> $\overline { { 1 9 . 1 2 \pm 1 . 4 7 0 } }$ </td><td> $\overline { { 1 3 . 9 0 \pm 0 . 9 2 } }$ </td><td> $\overline { { 8 . 9 8 \pm 0 . 0 6 6 } }$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 5 0 )$ </td><td>DQN</td><td>on</td><td> $0 . 9 5 \pm 0 . 0 0 8$ </td><td> $1 1 . 5 0 \pm 0 . 0 8$ </td><td> $8 . 7 9 \pm 0 . 1 5 8$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 5 0 )$ </td><td>PPO</td><td>off</td><td> $1 6 . 5 6 \pm 0 . 0 7 0$ </td><td> $1 6 . 7 2 \pm 1 . 2 9$ </td><td> $7 . 7 6 \pm 0 . 1 1 8$ </td></tr><tr><td>Nonstationary  $( p _ { \mathrm { s t a y } } = 0 . 5 0 )$ </td><td>PPO</td><td>on</td><td> $0 . 2 0 \pm 0 . 0 0 2$ </td><td> $3 0 . 3 8 \pm 1 . 0 9$ </td><td> $8 . 9 7 \pm 0 . 3 1 2$ </td></tr></table>

Table 2: Training summary across three random seeds. Values are reported as mean ± standard deviation. Collision rate is reported as a percentage. Lower collision rate indicates better safety, higher final reward indicates better task performance, and minimum distance is reported as an auxiliary proximity diagnostic rather than the primary safety metric.

![](images/cac07e20cfcaae6ca7ef233c3f16fd2cd5becd7f867e2c75d34bfe90c354ebe6.jpg)  
Figure 4: Collision-rate sweep across the full $p _ { \mathrm { s t a y } }$ range on the main merge-v0 benchmark. This figure reports the primary safety metric and shows that safety-enabled runs remain close to zero, while unconstrained baselines become more collision-prone as context switching becomes more frequent.

![](images/1f2e1d470c8dbb685d650fb2ec1065b98bfdf0863ead8d30211fbd84295029f4.jpg)

![](images/a51223154996d1c179a8ffcb769ef9fb6807233c4745b3d2c1cb6d9b078e4357.jpg)  
Figure 5: Final-reward sweep across the full $p _ { \mathrm { s t a y } }$ range. This figure reports task performance and shows that the safety layer preserves usable reward despite large collision-rate reductions, with some degradation under stronger nonstationarity.

![](images/c3ec24afe474ca1c895840ef3b23a0d534ab5aedbf7db8e72ddc5e4f67a264fe.jpg)

![](images/fa54d940c8c7aed7c9d71ae79ed6b0ff3d0ee12284b61a4c578d4caa96124945.jpg)  
Figure 6: Minimum-distance sweep across the full $p _ { \mathrm { s t a y } }$ range. This figure reports an auxiliary proximity diagnostic rather than the primary safety target, illustrating that collision reduction is accompanied by realistic mobility/proximity trade-offs.

## C HELD-OUT LAYOUT RESULTS

This appendix reports the held-out-layout stress test used to evaluate whether the safety layer remains effective beyond the main merge-v0 benchmark. The held-out environments are three highway-env layouts: highway-v0, intersectionv0, and racetrack-v0. These layouts introduce different road geometries and interaction patterns from the main merge setting.

Table 3 reports aggregate collision rates and relative reductions for each held-out layout, while Figure 7 visualizes the corresponding safety-off and safety-on collision rates. Across all three held-out layouts, the safety-enabled method reduces collision rates relative to the unconstrained baseline, suggesting that the safety layer is not restricted to the main merge-v0 environment.
<table><tr><td>Environment</td><td>Safety off (%)</td><td>Safety on (%)</td><td>Reduction</td></tr><tr><td>merge</td><td>14.76%</td><td>1.88%</td><td>87.2%</td></tr><tr><td>highway</td><td>11.76%</td><td>0.34%</td><td>97.1%</td></tr><tr><td>intersection</td><td>6.18%</td><td>0.49%</td><td>92.0%</td></tr><tr><td>racetrack</td><td>7.94%</td><td>0.35%</td><td>95.7%</td></tr></table>

Table 3: Main and held-out layout collision summary averaged over DQN/PPO and all evaluated $p _ { \mathrm { s t a y } }$ values. Collision rates are reported as percentages. The merge-v0 row corresponds to the main benchmark, while highway-v0, intersection-v0, and racetrack-v0 are held-out stress-test layouts. Safety-enabled runs reduce collisions across all layouts; the magnitude varies by road topology, so these results are presented as a stress test rather than a claim of universal transfer.

![](images/b95b02dc44377ba18a31682ef1475722349697dffcc023b451175cda4bffe5b4.jpg)  
Figure 7: Held-out layout stress test on highway-v0, intersection-v0, and racetrack-v0. Bars show mean collision rate across DQN/PPO and evaluated nonstationarity levels. The safety-enabled method reduces collisions in each evaluated held-out layout.

## D THEORETICAL SAFETY GUARANTEES

Because the proposed method relies on learned context extraction, context forecasting, and tail-risk constraint prediction, its guarantees are conditional on calibration and predictive coverage assumptions. We therefore provide high-probability, horizon-level guarantees rather than unconditional hard-safety guarantees. The analysis focuses on multi-step proactive safety over a forecast horizon $n \geq 1 0$ , using calibrated context uncertainty sets and tail-safe clearance prediction.

## D.1 SETUP

We assume episodic nonstationarity: each episode i corresponds to a stationary MDP instance with latent context $z _ { i } \in \mathbb { R } ^ { d }$ , while $z _ { i }$ changes only between episodes. Within an episode, at each step t the agent observes a clearance margin

$$
d _ { t } \ = \ \operatorname* { m i n } _ { j \in { \mathcal { O } } _ { t } } \Big ( \mathrm { d i s t } ( \mathrm { e g o } , j ) - \big ( d _ { 0 } + h v _ { t } \big ) \Big ) ,\tag{15}
$$

and safety requires $d _ { t } > 0$ . The algorithm predicts an n-step tail-safe clearance forecast using Tier 3 and enforces a proactive constraint

$$
c _ { t + k } ^ { \prime } = \hat { d } _ { t + k } ^ { ( q ) } - \epsilon , \qquad k = 1 , \dots , n ,\tag{16}
$$

where $\hat { d } _ { t + k } ^ { ( q ) }$ is a predicted q-quantile of clearance and $\epsilon > 0$ is a fixed safety margin. Context uncertainty is handled via a conformal ellipsoid $\mathcal { Z } _ { i } ( \rho )$ derived from Tier 2.

## D.2 ASSUMPTIONS

We state the minimal conditions needed for high-probability safety.

A1 (Conformal context coverage). Tier 2 outputs Gaussian forecasts $\left( \mu _ { i } , \Sigma _ { i } \right)$ and a conformal radius $\rho$ is chosen such that the ellipsoid

$$
\mathcal { Z } _ { i } ( \rho ) = \left\{ z : ( z - \mu _ { i } ) ^ { \top } \Sigma _ { i } ^ { - 1 } ( z - \mu _ { i } ) \leq \rho ^ { 2 } \right\}
$$

satisfies the coverage property

$$
\begin{array} { r } { \mathbb { P } ( z _ { i } \in \mathcal { Z } _ { i } ( { \boldsymbol \rho } ) ) ~ \geq ~ 1 - \alpha . } \end{array}\tag{17}
$$

A2 (Robust quantile coverage). Tier 3 produces a context-conditioned quantile predictor $\hat { d } ^ { ( q ) } ( s , z )$ such that for each $k \leq n$

$$
\begin{array} { r } { \mathbb { P } \Big ( d _ { t + k } \geq \hat { d } _ { t + k } ^ { ( q ) } ( z ) \mid s _ { t } , \pi \Big ) \geq 1 - q , \forall z \in \mathcal { Z } _ { i } ( \rho ) , } \end{array}\tag{18}
$$

where the probability is over environment stochasticity and (optionally) model randomness. This assumption corresponds to a calibrated lower-quantile predictor, which can be approached in practice via quantile regression plu empirical calibration.

A3 (MPC feasibility). At each step, the MPC filter selects an action sequence whose predicted trajectory satisfies the robust constraint condition:

$$
\operatorname* { m i n } _ { k \le n } \operatorname* { i n f } _ { z \in \mathcal { Z } _ { i } ( \rho ) } \left( \hat { d } _ { t + k } ^ { ( q ) } ( z ) - \epsilon \right) \ge 0 .\tag{19}
$$

If no feasible candidate exists, a conservative fallback is applied.

## D.3 MULTI-STEP HIGH-PROBABILITY SAFETY

We now state a horizon-level safety guarantee for the executed trajectory.

Theorem 1 (Proactive n-step safety under calibrated context and quantile prediction). Assume A1–A3. Suppose the MPCfilter enforces equation 19 at time t. Then, with probability at least

$$
1 - \alpha - n q ,\tag{20}
$$

the realized clearance satisfies

$$
d _ { t + k } \geq \epsilon \qquad f o r a l l k = 1 , \dots , n .\tag{21}
$$

Proof sketch. By A1, with probability at least $1 - \alpha$ the true episode context satisfies $z _ { i } \in \mathcal { Z } _ { i } ( \rho )$ . Conditioned on this event, the robust constraint enforcement equation 19 implies $\hat { d } _ { t + k } ^ { ( q ) } ( z _ { i } ) \geq \epsilon$ for all $k \leq n . \mathrm { B y } \mathrm { A } 2$ , for each fixed k, the probability that the realized clearance violates this bound is at most $q ,$ i.e., $\mathbb { P } ( d _ { t + k } < \hat { d } _ { t + k } ^ { ( q ) } ( z _ { i } ) ) \le q$ . Applying a union bound across $k = 1 , \dots , n$ yields

$$
\mathbb { P } \left( \exists k \le n : \ d _ { t + k } < \epsilon \right) \le \alpha + n q ,
$$

which proves the claim.

## D.4 INTERPRETATION: WHAT TYPE OF GUARANTEES ARE PROVIDED?

Theorem 1 provides a probabilistic (chance-style) guarantee over the next n steps. Importantly, the guarantee is proactive and control-aware: it applies to the trajectory induced by the selected MPC-filtered actions rather than to passive rollouts. The guarantee is not an unconditional hard constraint in the classical control-theoretic sense, since it depends on calibrated coverage properties of learned predictors. However, the failure probability is explicitly controlled by $( \alpha , q , n ) \colon$ decreasing α enlarges the context ellipsoid, decreasing q increases conservatism in the quantile predictor, and increasing n makes the guarantee stricter but more challenging.

## D.5 FROM PER-STEP GUARANTEES TO CUMULATIVE SAFETY

Although the algorithm enforces a per-step margin, it also implies a cumulative bound on the probability of any violation over a time horizon T:

$$
\mathbb { P } ( \exists t \le T : \ d _ { t } < \epsilon ) \le \left\lceil \frac { T } { n } \right\rceil ( \alpha + n q ) ,
$$

by applying Theorem 1 over consecutive blocks of length n and union bounding the failure events. Thus, per-step proactive constraints yield a cumulative risk control interpretation, where the total violation probability grows at most linearly with time.

## D.6 DISCUSSION

The guarantee above highlights the role of the three upgrades: conformal calibration provides explicit context uncertainty coverage, quantile forecasting controls tail risk in clearance prediction, and MPC-style filtering ensures that constraints are enforced for the executed controls. Together, these components provide a principled high-probability safety guarantee for driving under episodic nonstationarity, while allowing the method to become less conservative as predictive accuracy improves.