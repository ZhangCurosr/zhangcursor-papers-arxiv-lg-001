# MA-UITD: OFFLINE MULTI-AGENT REWARD INFER-ENCE FOR CHARACTERIZING CONTINUOUS SOCIAL BE-HAVIOR

Zeyuan Zhang   
Institute of Neuroscience, Chinese Academy of Sciences   
zhangzeyuan@ion.ac.cn

Yuan Cheng Icahn School of Medicine at Mount Sinai yuan.cheng@mssm.edu

Yusi Chen   
Institute of Neuroscience, Chinese Academy of Sciences   
chenyusi@ion.ac.cn   
Yiyun Yang   
Nankai University   
1120210035@mail.nankai.edu.cn   
Herbert Zheng Wu   
Icahn School of Medicine at Mount Sinai   
zheng.wu@mssm.edu

## ABSTRACT

Multi-agent inverse reinforcement learning offers a powerful framework not only for imitation learning but also for understanding the neural computations supporting social interactions and decision-making. Inferring rewards from interacting agents is challenging because observed choices reflect both personal preferences and expectations about others. Existing methods often require additional environment interaction, repeated policy optimization as an inner loop, or restrictive assumptions about environment dynamics and action spaces, limiting their applicability to fixed behavioral datasets. We introduce Multi-Agent Unilateral Inverse Temporal Difference Learning (MA-UITD), an offline method for inferring explicit agent-specific rewards from joint demonstrations in continuous-state environments with either discrete or bounded continuous actions. Its core idea is to represent the long-term consequences of each agent’s own actions through belief-conditioned successor features that average over other agents’ behavior. Shared reward features and agent-specific preference vectors connect these unilateral values to the target reward function, while inverse temporal-difference consistency and noise-contrastive behavioral learning enable inference without known dynamics or an inner policyoptimization loop. Across simulated cooperative, competitive, and general-sum tasks, MA-UITD outperforms baselines in discrete-action settings and infers behaviorally compatible reward profiles in continuous-action settings. When applied to experimentally collected social-foraging trajectories, MA-UITD inferred distinct spatial and interaction-dependent reward structures for leaders and followers, providing a quantitative characterization of latent behavioral objectives that may be relevant for understanding the neural computations underlying social behavior. We make our code available at https://github.com/ZaynZh/MA-UITD

## 1 INTRODUCTION

A central goal of behavioral science and neuroscience is to understand the objectives that organize individual behavior. Although accurate action prediction is an important benchmark, an explicit reward function provides a more interpretable computational account by specifying which outcomes an animal values, how it trades off competing outcomes, and how these valuations change with experience. Subjective-value signals have been identified in orbitofrontal cortex and across a broader valuation network involving ventromedial prefrontal cortex and ventral striatum (Padoa-Schioppa & Assad, 2006; Bartra et al., 2013), whereas phasic activity of midbrain dopamine neurons has been linked to quantitative reward-prediction errors and corresponding distribution (Schultz et al., 1997; Bayer & Glimcher, 2005; Dabney et al., 2020). Consequently, inferred rewards—and the values and prediction errors they imply—can serve as trial-by-trial computational variables for modelbased analyses of neural recordings, enabling tests of latent-computation hypotheses beyond action prediction alone (O’Doherty et al., 2007).

Inverse reinforcement learning (IRL) provides a principled framework for recovering such latent objectives by inferring reward functions under which observed behavior is likely. In single-agent settings, IRL has been used to recover both stable and dynamically evolving goals from behavior (Ziebart et al., 2008; Ashwood et al., 2022; Wu et al., 2025; Ke et al., 2025; Wang et al., 2025). Model-derived reward and value representations have also begun to provide computational variables for characterizing neural activity and examining how valuation-related computations may be altered in neuropsychiatric disorders (Bromberg-Martin et al., 2024; Moneta et al., 2024). However, conventional IRL was developed primarily for single-agent settings and treated each individual as an isolated decision-maker, making it difficult to distinguish intrinsic preferences from socially contingent choices that depend on other individuals.

Inferring these objectives becomes substantially more challenging when individuals interact. In social settings, an animal’s choice depends on both its own preferences and its expectations about others. Approaching a shared resource, for example, may reflect one’s own resource preference and anticipated interaction with other agents such as cooperation and competition. Similar observed actions can consequently arise from different incentives. Computational models have framed such social inference as IRL in a partially observable Markov decision process (POMDP), where a focal agent acts according to beliefs about an incompletely observed environment (Jara-Ettinger, 2019; Baker et al., 2017; Velez-Ginorio et al., 2017). Although this formulation captures uncertainty and false beliefs, other individuals are typically incorporated into the focal agent’s environment rather than represented as autonomous decision-makers with their own actions, policies, and rewards. By contrast, a multi-agent MDP, or Markov game, explicitly represents multiple decision-makers whose joint actions determine state transitions and whose policies are mutually interdependent. Multiagent inverse reinforcement learning (MAIRL) builds on this formulation to infer agent-specific rewards while accounting for how each agent anticipates and responds to the behavior of others (Yu et al., 2019; Tian et al., 2021; Neumeyer et al., 2021; Chen et al., 2026). MAIRL thus offers a systematic framework for separating what each individual values from what it expects others to do, providing interpretable computational variables for studying cooperation, competition, and general-sum interactions.

Applying MAIRL to animal behavior imposes several practical constraints: experiments typically yield a fixed dataset of joint trajectories without reward labels, known transition dynamics, or opportunities to collect additional behavior under a learned policy, and the recorded states and actions are often continuous. Existing methods address parts of this setting: Multi-Agent Adversaria Inverse Reinforcement Learning(MA-AIRL) combines reward estimation with adversarial policy optimization, requiring repeated policy updates and environment interaction during training (Yu et al., 2019). Multi-Agent Marginal Q-Learning(MAMQL) avoids adversarial training by learning through opponent-marginalized values, using discrete actions and additional policy rollouts in its reported training procedure (Haynam et al., 2025). These constraints motivate a fully offline approach that infers explicit agent-specific rewards directly from fixed joint demonstrations, without known dynamics or an inner policy-optimization loop, while supporting continuous behavior.

We introduce Multi-Agent Unilateral Inverse Temporal Difference Learning (MA-UITD), an offline method for inferring explicit agent-specific rewards from joint demonstrations. We focus on arrivalstate rewards that describe the value individuals assign to attained environmental and social states. The inferred rewards provide interpretable explanations of behavior while supporting expert-level behavioral imitation. Building on successor features and inverse temporal-difference learning (Filos et al., 2021), our main contributions are:

• We introduce belief-conditioned successor features (BCSF), which marginalize over other agents’ anticipated behavior to represent unilateral long-term values and recover interpretable agent-specific rewards.

• We develop an offline learning procedure that combines inverse temporal-difference consistency with noise-contrastive behavioral learning, requires no known dynamics or policyoptimization loop, and supports continuous states with either discrete or continuous actions.

• Across cooperative, competitive, and general-sum simulations, MA-UITD outperforms discrete-action baselines and captures task-relevant reward structure in continuous-action settings.

• Applied to animal social-interaction data, MA-UITD reveals distinct spatial and interactiondependent incentives underlying leader–follower dynamics.

## 2 RELATED WORK

Multi-agent inverse reward learning. In multi-agent settings, MA-AIRL (Yu et al., 2019) couples agent-specific reward estimation to an adversarial discriminator and policy optimization. Such adversarial approaches involve an inner policy loop and outer loop, which could be computationally expensive and unstable. MAMQL (Haynam et al., 2025) instead learns opponent-marginalized reward through a non-adversarial inverse soft-Q objective; however, it assumes a discrete action set and obtains explicit rewards through a separate reconstruction loss, with its reported training procedure drawing additional policy rollouts. Other work models strategic beliefs through latent quantal level-k reasoning (Tian et al., 2021), but requires recursive value iteration and identifies continuous state and action spaces as a limitation. GS-CIOC (Neumeyer et al., 2021) supports non-adversarial reward inference with continuous states and actions, yet presumes known dynamics and relies on local linear-quadratic approximations around demonstrations under a specified equilibrium model.

Successor-feature-based inverse reward learning. A complementary approach connects explicit rewards to long-term values through a shared feature representation. Filos et al. (2021) introduce inverse temporal-difference (ITD) learning, which jointly estimates shared cumulants, demonstratorspecific successor features, and preference vectors from fixed demonstrations without reward labels. By combining behavioral likelihood fitting with successor-feature Bellman consistency, ITD yields explicit rewards from the learned cumulants and preferences. However, its formal setup considers different reward tasks within a common controlled Markov process with discrete actions, without an explicit Markov-game formulation of strategic interactions.

Offline reward inference for animal behavior. These considerations are particularly important for studies of animal behavior, where joint trajectories can be recorded experimentally but constructing a faithful interactive simulator is often difficult because the underlying behavioral dynamics are only partially known (Fujii et al., 2025). In such settings, methods that require repeated environment interaction, additional policy rollouts, or an accurate model of the dynamics may therefore be impractical. Offline reward-inference methods can recover explicit rewards from fixed demonstrations without known dynamics or additional environment interaction, but are formulated for single-agent settings with discrete actions (Chan & Schaar, 2021). This motivates methods that can infer explicit, agent-specific rewards directly from fixed datasets of continuous joint trajectories in multi-agent systems.

Building on these, our method learns belief-conditioned successor features directly from a fixed dataset of joint transitions and uses noise-contrastive estimation to avoid the intractable continuousaction normalizer. It thereby recovers an explicit agent-specific reward profile in continuous state and action spaces, without adversarial optimization, additional environment interaction, or a forward policy-optimization loop; moreover, equilibrium and finite-depth reasoning enter as restrictions on a common predictive belief rather than as hard-coded solution concepts.

## 3 PROBLEM FORMULATION

Markov game and offline demonstrations. We consider an n-agent discounted Markov game without known reward functions $\mathcal { G } ^ { - R } = ( \mathcal { N } , \mathcal { S } , \{ A _ { i } \} _ { i = 1 } ^ { n } , P , p _ { 0 } , \gamma )$ , where $\mathcal { N } = \{ 1 , \ldots , n \}$ is the set of agents, S is the joint state space, $A _ { i }$ is the action space of agent $\begin{array} { r } { i , \mathcal { A } = \prod _ { i = 1 } ^ { n ^ { - } } \mathcal { A } _ { i } } \end{array}$ is the joint action space, $P ( s ^ { \prime } \mid s , \pmb { a } )$ denotes the transition dynamics, $p _ { 0 }$ is the initial-state distribution, and $\gamma \in [ 0 , 1 )$ is the discount factor. We allow each $\mathcal { \bar { A } } _ { i }$ to be either discrete or a bounded continuous subset of $\mathbb { R } ^ { d _ { i } }$

At time t, all agents simultaneously choose a joint action $\mathbf { \delta } \mathbf { a } _ { t } = ( a _ { 1 , t } , \ldots , a _ { n , t } )$ , after which the environment transitions according to $s _ { t + 1 } \sim P ( \cdot \mid s _ { t } , \mathbf { a } _ { t } )$ . We are given only an offline dataset of joint expert trajectories

$$
\mathcal { D } = \left\{ \tau ^ { ( m ) } \right\} _ { m = 1 } ^ { M } , \qquad \tau ^ { ( m ) } = \left( s _ { 0 } ^ { ( m ) } , { \pmb a } _ { 0 } ^ { ( m ) } , s _ { 1 } ^ { ( m ) } , { \pmb a } _ { 1 } ^ { ( m ) } \ldots , s _ { T _ { m } } ^ { ( m ) } , { \pmb a } _ { T _ { m } } ^ { ( m ) } \right) ,
$$

without observing the rewards that generated the demonstrated behavior. No additional interaction with the environment is assumed during inverse learning.

Arrival-state rewards. Motivated by behavioral and neuroscience settings in which the inferred reward is interpreted as the value assigned to attained outcomes or configurations, we focus on agent-specific arrival-state rewards. That is, the reward received by agent i following a transition $\left( { { s _ { t } } , { \bf { a } } _ { t } , { s _ { t + 1 } } } \right)$ is assumed to take the form

$$
R _ { i } ( s _ { t } , \pmb { a } _ { t } , s _ { t + 1 } ) = r _ { i } ( s _ { t + 1 } ) , \qquad r _ { i } : \mathcal { S }  \mathbb { R } .
$$

Because s is the joint environmental state, $r _ { i } ( s )$ may still encode both individual preferences and interactions among agents. Thus, this assumption restricts where reward is assigned—to the resulting state rather than directly to the executed action—without requiring agents to share the same reward function. When deterministic choices produce distinguishable outcome states, action-contingent rewards can be represented equivalently as arrival-state rewards. This includes common cognitive tasks such as multi-armed bandits and the prisoner’s dilemma when the outcome state records the selected action or joint action.

Reward compatibility rather than reward identification. Inverse reinforcement learning is generally non-identifiable, and this ambiguity is further compounded in multi-agent settings. In particular, rewards in general-sum Markov games are not uniquely identifiable from demonstrations without additional assumptions (Freihaut & Ramponi, 2025). We therefore seek a representative agent-specific reward profile $\widehat { R } = ( \widehat { r } _ { 1 } , \ldots , \widehat { r } _ { n } )$ within the arrival-state reward class, rather than claim recovery of the unique ground truth. We assess compatibility operationally: under agent i’s predictive belief $\mu _ { i , - i }$ , the reward-induced belief-conditioned score should assign higher relative likelihood to demonstrated actions than to counterfactual alternatives. Equilibrium and finite-depth reasoning enter only through the choice of this belief, rather than as imposed solution concepts.

## 4 MULTI-AGENT UNILATERAL INVERSE TEMPORAL DIFFERENCE LEARNING (MA-UITD)

We introduce MA-UITD, an offline method for inferring agent-specific rewards from fixed joint demonstrations. It involves three key components: a shared reward feature map Φ, an agent-specific preference vector $w _ { i } .$ , and agent-specific belief-conditioned successor features (BCSFs). BCSFs represent the long-term consequences of an agent’s own action after averaging over its prediction of the other agents’ behavior. This separates strategic assumptions, encoded by the predictive belief, from reward preferences, encoded by w .

Learning combines two complementary objectives. A contrastive behavioral loss makes the inferred values assign greater relative likelihood to demonstrated actions than to sampled alternatives, while a unilateral inverse temporal-difference loss enforces Bellman consistency between the BCSFs and the learned reward features. Both objectives are estimated from the fixed dataset, without a transition model, additional environment interaction, or an inner policy-optimization loop. We first define the BCSF representation and then present the offline learning procedure.

## 4.1 BELIEF-CONDITIONED SUCCESSOR FEATURES (BCSF)

For a stationary joint policy π, define agent $i \mathbf { \ ' } _ { \mathbf { S } }$ joint-action value as

$$
Q _ { i } ^ { \pi } ( s , \pmb { a } ) \triangleq \mathbb { E } _ { \pmb { \pi } , \pmb { P } } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r _ { i } ( s _ { t + 1 } ) \Bigg | s _ { 0 } = s , \pmb { a } _ { 0 } = \pmb { a } \right] .\tag{1}
$$

This value conditions on the complete joint action $\pmb { a } = ( a _ { i } , \pmb { a } _ { - i } )$ . The initial joint action is fixed, while subsequent joint actions are generated by π and states evolve according to $P .$ . In our setting, the demonstrations consist of trajectories generated by an unknown stationary joint policy $\pi ^ { D }$

The joint-action value $Q _ { i } ^ { \pi } ( s , a _ { i } , { \pmb a } _ { - i } )$ evaluates a fully specified joint action. In a simultaneous-move interaction, however, agent i must select $a _ { i }$ without observing the other agents’ current actions. Its policy therefore cannot respond directly to the realized $\mathbf { \delta } _ { \mathbf { a } - i } .$ . Instead, evaluating $a _ { i }$ requires a belief about the actions of the other agents. Let $\pmb { \mu } _ { i , - i } ( \pmb { a } _ { - i } \mid s )$ denote its belief about the other agents’ actions, and let $\pmb { \pi } ^ { ( i ) } = ( \pi _ { i } , \pmb { \mu } _ { i , - i } )$ denote its perceived continuation profile. The value relevant to its choice is therefore

$$
\bar { Q } _ { i } ^ { \pi _ { i } , \mu _ { i , - i } } \big ( s , a _ { i } \big ) \triangleq \mathbb { E } _ { a _ { - i } \sim \pmb { \mu } _ { i , - i } ( \cdot | s ) } \left[ Q _ { i } ^ { \pi ^ { ( i ) } } \big ( s , a _ { i } , \pmb { a } _ { - i } \big ) \right] .\tag{2}
$$

We assume an arrival-state linear reward representation,

$$
r _ { i } ( s _ { t + 1 } ) = \Phi ( s _ { t + 1 } ) ^ { \top } w _ { i } .\tag{3}
$$

The corresponding joint-action successor features are

$$
\Psi _ { i } ^ { \pi ^ { ( i ) } } ( s , a ) \triangleq \mathbb { E } _ { \pi ^ { ( i ) } , P } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \Phi ( s _ { t + 1 } ) \Bigg | s _ { 0 } = s , \pmb { a } _ { 0 } = \pmb { a } \right] ,\tag{4}
$$

so that $Q _ { i } ^ { \pi ^ { ( i ) } } ( s , \pmb { a } ) = \Psi _ { i } ^ { \pi ^ { ( i ) } } ( s , \pmb { a } ) ^ { \top } w _ { i }$ . Marginalizing these successor features over the agent’s belief gives the belief-conditioned successorfeatures (BCSFs):

$$
\begin{array} { r } { \bar { \Psi } _ { i } ^ { \pi _ { i } , \mu _ { i , - i } } \bigl ( s , a _ { i } \bigr ) \triangleq \mathbb { E } _ { a _ { - i } \sim \pmb { \mu } _ { i , - i } ( \cdot | s ) } \left[ \Psi _ { i } ^ { \pi ^ { ( i ) } } \bigl ( s , a _ { i } , \pmb { a } _ { - i } \bigr ) \right] . } \end{array}\tag{5}
$$

By linearity,

$$
\bar { Q } _ { i } ^ { \pi _ { i } , \mu _ { i , - i } } ( s , a _ { i } ) = \bar { \Psi } _ { i } ^ { \pi _ { i } , \mu _ { i , - i } } ( s , a _ { i } ) ^ { \top } w _ { i } .\tag{6}
$$

We model agent i’s policy as a logit response to this unilateral value:

$$
\pi _ { i } ( a _ { i } \mid s ) = \frac { \exp { \left( \lambda _ { i } \bar { \Psi } _ { i } ^ { \pi _ { i } , \mu _ { i , - i } } ( s , a _ { i } ) ^ { \top } \boldsymbol { w } _ { i } \right) } } { \displaystyle \int _ { A _ { i } } \exp { \left( \lambda _ { i } \bar { \Psi } _ { i } ^ { \pi _ { i } , \mu _ { i , - i } } ( s , \tilde { a } _ { i } ) ^ { \top } \boldsymbol { w } _ { i } \right) } d \tilde { a } _ { i } } .\tag{7}
$$

For discrete action spaces, the integral is replaced by a sum over $A _ { i }$

The BCSFs satisfy the Bellman recursion

$$
\begin{array} { r } { \bar { \Psi } _ { i } ^ { \pi _ { i } , \mu _ { i , - i } } ( s , a _ { i } ) = \mathbb { E } _ { a _ { - i } \sim \mu _ { i , - i } ( \cdot | s ) , \atop s ^ { \prime } \sim P ( \cdot | s , a _ { i } , a _ { - i } ) , } \left[ \Phi ( s ^ { \prime } ) + \gamma \bar { \Psi } _ { i } ^ { \pi _ { i } , \mu _ { i , - i } } ( s ^ { \prime } , a _ { i } ^ { \prime } ) \right] . } \end{array}\tag{8}
$$

This recursion integrates over the other agents’ current actions, the next state, and agent i’s continuation action. It is the basis of the offline temporal-difference objective introduced below.

The belief $\pmb { \mu } _ { i , - i }$ determines the strategic interpretation of the representation. For a logit quantalresponse equilibrium (LQRE) (McKelvey & Palfrey, 1995), there is a common policy profile $\pi ^ { \star }$ such that, for every agent,

$$
\pmb { \mu } _ { i , - i } = \pmb { \pi } _ { - i } ^ { \star } = \prod _ { j \neq i } \pi _ { j } ^ { \star } , \qquad \pi _ { i } = \pi _ { i } ^ { \star } .\tag{9}
$$

All perceived continuation profiles then coincide with $\pi ^ { \star }$ , and Equation (7) holds simultaneously for every agent. Under quantal level-k reasoning, the same representation instead uses $\mu _ { i , - i } ^ { ( k ) } = \pi _ { - i } ^ { ( \dot { k } - 1 ) }$ Thus, equilibrium and finite-depth reasoning correspond to different restrictions on the predictive belief rather than different reward representations.

## 4.2 OFFLINE REWARD INFERENCE WITH MA-UITD

Given demonstrations D, we parameterize the shared reward feature map as $\Phi \big ( s ; \theta ^ { \Phi } \big )$ , agent $i \ ' s$ BCSFs as $\bar { \Psi } _ { i } ( s , a _ { i } ; \eta _ { i } )$ , and its preference vector as $w _ { i }$ . For offline estimation, we take the predictive belief to be calibrated to the demonstrated conditional behavior of the other agents $\mu _ { i , - i } ( { \pmb a } _ { - i } \mid s )$ ≈ $\pi _ { - i } ^ { D } ( a _ { - i } \mid s )$ . Observed joint transitions then provide Monte Carlo samples of the expectations in Equation (8). We fit the parameterized representation using a contrastive behavioral objective and a BCSF temporal-difference objective.

## 4.2.1 CONTRASTIVE BEHAVIORAL LEARNING

The normalizing integral in Equation (7) is generally intractable for continuous actions. We therefore fit the unnormalized BCSF scores using noise-contrastive estimation (NCE) (Gutmann & Hyvarinen¨ , 2010). At a demonstrated state $s ,$ let $a _ { i } ^ { - + }$ denote agent $i \mathbf { \ ' } _ { \mathbf { S } }$ observed action and sample K alternative actions $a _ { i , 1 : K } ^ { - } \sim \nu _ { i } ( \cdot \mid s )$ from a declared proposal distribution over $A _ { i }$ . Their scores are $q _ { i } ^ { + } =$ $\bar { \Psi } _ { i } ( s , a _ { i } ^ { + } ; \pmb { \eta } _ { i } ) ^ { \top } \boldsymbol { w } _ { i } , q _ { i , k } ^ { - } = \bar { \Psi } _ { i } ( s , a _ { i , k } ^ { - } ; \pmb { \eta } _ { i } ) ^ { \top } \boldsymbol { w } _ { i }$

We minimize the contrastive loss

$$
\begin{array} { r } { \mathcal { L } _ { i } ^ { \mathrm { N C E } } ( \eta _ { i } , w _ { i } ) \triangleq - \mathbb { E } _ { \mathbf { \Phi } _ { ( s , a _ { i } ^ { + } ) \sim \mathcal { D } , \mathbf { \Phi } } } \left[ \log \frac { \exp ( q _ { i } ^ { + } / \tau _ { \mathrm { N C E } } ) } { \exp ( q _ { i } ^ { + } / \tau _ { \mathrm { N C E } } ) + \sum _ { k = 1 } ^ { K } \exp ( q _ { i , k } ^ { - } / \tau _ { \mathrm { N C E } } ) } \right] . } \end{array}\tag{10}
$$

where $\tau _ { \mathrm { N C E } } > 0$ controls the assumed stochasticity of the demonstrated behavior and absorbs the rationality scale $\lambda _ { i }$ in Equation (7). A lower temperature emphasizes the highest-valued candidate, whereas a higher temperature represents noisier action selection. This objective compares only unilateral alternatives for agent i; the other agents’ behavior remains marginalized through $\bar { \Psi } _ { i }$

## 4.2.2 UNILATERAL INVERSE TEMPORAL-DIFFERENCE LEARNING

Inverse temporal-difference learning connects reward features to successor features through their Bellman recursion (Filos et al., 2021). In our setting, the recursion must be enforced on BCSFs because these are the unilateral quantities that determine the behavioral scores in Equation (10).

Consider consecutive demonstrated transitions $\left( s _ { t } , \mathbf { a } _ { t } , s _ { t + 1 } , \mathbf { a } _ { t + 1 } \right) \sim \mathcal { D }$ . Under Section 4.2, the observed ${ \mathbf { } } _ { \mathbf { \Gamma } } \mathbf { \Delta } \mathbf { a } _ { - i , t }$ samples the current belief, $s _ { t + 1 }$ samples the unknown transition kernel, and $\boldsymbol { a } _ { i , t + 1 }$ samples agent i’s continuation policy. The one-step target

$$
y _ { i , t } = \Phi ( s _ { t + 1 } ; \pmb { \theta } ^ { \Phi } ) + \gamma \mathrm { s t o p g r a d } \big ( \bar { \Psi } _ { i } \big ( s _ { t + 1 } , a _ { i , t + 1 } ; \tilde { \eta } _ { i } \big ) \big )\tag{11}
$$

is therefore a Monte Carlo estimate of the right-hand side of Equation (8). Here $\tilde { \eta } _ { i }$ denotes targetnetwork parameters, which are treated as fixed when differentiating the loss. We minimize

$$
\mathcal { L } _ { i } ^ { \mathrm { U T D } } ( \theta ^ { \Phi } , \eta _ { i } ) \triangleq \mathbb { E } _ { ( s _ { t } , a _ { t } , s _ { t + 1 } , a _ { t + 1 } ) \sim \mathcal { D } } \Bigg [ \Bigg \lVert \bar { \Psi } _ { i } \big ( s _ { t } , a _ { i , t } ; \eta _ { i } \big ) - y _ { i , t } \Bigg \rVert ^ { 2 } \Bigg ] .\tag{12}
$$

This sample-based objective implicitly averages over the other agents’ actions and the environment transition, so neither the transition kernel nor an explicit opponent model must be estimated.

## 4.2.3 OPTIMIZATION AND REWARD RECOVERY

Training alternates between the two objectives. See Algorithm 1 for a detailed implementation. The NCE step updates $\mathbf { \eta } _ { \eta _ { i } }$ and $w _ { i }$ to fit the demonstrated action preferences, with an $\ell _ { 1 }$ penalty $\lambda _ { w } \| w _ { i } \| _ { 1 }$ The UITD step updates $\smash { \theta ^ { \Phi } }$ and every $\eta _ { i }$ to enforce temporal consistency, followed by an update of the target parameters $\tilde { \eta } _ { i }$ . After convergence, the agent-specific rewards are recovered directly as $\widehat { r } _ { i } ( s ) \stackrel { - } { = } \bar { \Phi ( } s ; \pmb { \theta } ^ { \Phi } ) ^ { \top } w _ { i }$ . Because each agent has its own $w _ { i } .$ , the same representation applies to cooperative, competitive, and general-sum games.

## 5 RESULTS

We evaluate MA-UITD in three simulated multi-agent environments spanning cooperative, competitive, and general-sum interactions. Our experiments examine 1) whether MA-UITD reproduces expert-level behavior and 2) recovers interpretable rewards compatible with the demonstrations. We first compare MA-UITD with baselines in discrete-action settings and then evaluate it in continuous action settings.

## 5.1 EXPERIMENTAL SETUP

Environments. We evaluate three games: (1) cooperative Simple Spread from MPE2 (Lowe et al., 2017), where two agents cooperate to cover two landmarks; (2) competitive Simple Push, where a good agent attempts to reach a target landmark while an adversary tries to prevent it from doing so through physical interaction; and (3) general-sum Gems, adapted from Haynam et al. (2025). In Gems, Red and Blue maximize individual returns by collecting their respective private gems for 1 reward or jointly collecting a purple gem for $6 ,$ after which the episode terminates. Agents choose among nine directional actions in the discrete setting or bounded two-dimensional displacements in the continuous setting, with an 80-step horizon.

Baselines. We evaluate MA-UITD against two baselines. The first is MA-AIRL (Yu et al., 2019), an adversarial learning-based multi-agent IRL method. Then, we benchmark against MAMQL (Haynam et al., 2025), an IQ-Learn-style MAIRL algorithm that similarly marginalizes the joint-action value function over the opponents’ continuation policies but is designed only for discrete action spaces.

## 5.2 POLICY AND REWARD RECOVERY

Policy imitation. MA-UITD achieved higher recovered-policy returns than both baselines in Simple Spread and for both roles in Simple Push, while most closely approaching expert performance in Gems (Table 1). In Gems, MA-UITD and MAMQL recovered both cooperative and competitive behaviors, whereas MA-AIRL converged exclusively to competitive behavior (Table 8).

Table 1: Policy imitation performance measured as mean true return on discrete-action tasks
<table><tr><td rowspan="2">Method</td><td rowspan="2">Simple Spread Shared return ↑</td><td colspan="2">Simple Push cross-play</td><td rowspan="2">Gems Agent-mean return ↑</td></tr><tr><td>Learned adversary ↑</td><td>Learned good ↑</td></tr><tr><td>Expert reference</td><td>-21.592</td><td>0.722</td><td>-7.228</td><td>3.581</td></tr><tr><td>MA-AIRL</td><td> $- 2 2 . 1 3 5 \pm 0 . 0 7 1$ </td><td> $- 0 . 0 4 1 \pm 0 . 0 0 5$ </td><td> $- 7 . 8 8 5 \pm 0 . 0 8 1$ </td><td> $0 . 5 1 6 \pm 0 . 0 0 4$ </td></tr><tr><td>MAMQL</td><td> $- 2 8 . 0 9 8 \pm 0 . 4 1 5$ </td><td> $0 . 4 6 8 \pm 0 . 0 0 3$ </td><td> $- 7 . 4 4 3 \pm 0 . 0 0 8$ </td><td> $2 . 5 8 7 \pm 0 . 5 4 2$ </td></tr><tr><td>MA-UITD</td><td> $\mathbf { - 2 1 . 8 4 8 \pm 0 . 0 3 3 }$ </td><td> $\mathbf { 0 . 5 8 6 \pm 0 . 0 2 1 }$ </td><td> $\mathbf { - 7 . 3 1 9 \pm 0 . 0 2 2 }$ </td><td> $\mathbf { 2 . 9 5 3 \pm 0 . 0 9 9 }$ </td></tr></table>

Values are reported as mean ± sample standard deviation across three training seeds, each evaluated over 500 stochastic-policy episodes.

Reward recovery. We evaluate reward recovery on held-out transitions using Pearson’s correlation coefficient (PCC), which measures linear agreement, and Spearman’s rank correlation coefficient (SCC), which measures agreement in reward ordering. For each agent, we compare the ground-truth immediate reward with the reward predicted by the learned reward function (Table 7). Figure 1 shows that MA-UITD outperforms the baselines on both measures across all three simulated environments.

![](images/f5589a38a31f810e93f899a6adf40d595f0dc8b12c20e4ece00d531100bc235f.jpg)  
Figure 1: Reward recovery for MA-UITD and baselines. Each method is independently trained with three random seeds, and correlations are computed on 500 held-out test episodes for each trained model.

Interpretable reward visualization. We next visualize the reward functions recovered by each method. In Simple Spread (Fig. 2), MA-UITD captures the task’s underlying coverage structure: once one agent is close to a landmark, the other landmark becomes relatively more rewarding for the other agent, encouraging the agents to distribute across landmarks rather than collapse onto the same target. Consistent with the task’s symmetric cooperative objective, MA-UITD recovers nearly identical spatial reward structures for both agents. By contrast, MA-AIRL produces highly localized peaks around individual landmarks, whereas MAMQL assigns high rewards to regions inconsistent with the ground-truth structure and recovers markedly different reward functions across agents. Additional visualizations for Simple Spread and the other tasks are provided in Appendix C.3.1.

![](images/1e64123d12669ef93c456997ca253b94a5d893b9e29638c13762dbc75085f441.jpg)

![](images/7bc5612d2cf0fe4b236fd1062d680d1bae509a8a63cd998fa03a694e4fca7d78.jpg)

![](images/71cd4db1797612493bb21becf46b71394277770fc0b5cdb70ab693fef2d2ffda.jpg)

![](images/2077199296577ba49ca952aa845afc1ac41091085dd0937bf12bb10d34df4d23.jpg)

![](images/09a7236ac1abfa345ddf50005fdb5f55de7fd55c162b56867f315ebee396fe70.jpg)

![](images/e56c83d04544977f65eb250fc7051bf03ca42ef219a940a7c58c673406bd7cac.jpg)

![](images/175bc8edb0e84f766a334982d9f983c21dc4c570b17e0254dba9be597f4cddb3.jpg)  
Agent 1 arrival position, x

![](images/7f7403b759271c4752ab6643da41b9ad5f98a03fcc2c5aa60f26f13714e6be54.jpg)  
Figure 2: Visualization of the ground-truth and recovered reward landscapes in Simple Spread. Agent 1’s arrival position is varied over the spatial grid while agent 2 (red circle), the landmarks (stars), and arrival velocities are held fixed. The top and bottom rows show the rewards recovered for agents 1 and 2, respectively.

Continuous action spaces. We next evaluated MA-UITD in continuous-action versions of all three environments. The recovered policies achieved returns matching the expert reference in Gems and approached expert performance in Simple Spread, whereas performance in the competitive Simple Push task remained role dependent (Table 2). On held-out trajectories, the recovered rewards showed consistent positive agreement with the ground-truth rewards across all three tasks, with the strongest linear agreement observed in Simple Spread (agent-mean $\operatorname { P C C } \ 0 . 8 2 8 \pm 0 . 0 3 3 ; \operatorname { T a b l e } \ 3 )$ . The comparatively modest SCC values likely reflect the prevalence of transitions with similar immediate rewards in continuous environments, where small prediction errors can reverse local rankings. We also conducted a qualitative visual analysis by comparing the recovered and ground-truth reward landscapes (Appendix C.3.2). Together, these quantitative and visual results provide evidence that MA-UITD can infer behaviorally compatible reward profiles directly from offline demonstrations with bounded continuous actions.

Table 2: Policy imitation performance of MA-UITD on continuous-control tasks.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Simple Spread Shared return ↑</td><td colspan="2">Simple Push cross-play</td><td rowspan="2">Gems Agent-mean return ↑</td></tr><tr><td>Learned adversary ↑</td><td>Learned good ↑</td></tr><tr><td>Expert reference</td><td>-33.338</td><td>-0.063</td><td>-6.707</td><td>3.337</td></tr><tr><td>MA-UITD</td><td> $- 3 6 . 7 8 3 \pm 0 . 1 0 0$ </td><td> $- 2 . 2 8 6 \pm 0 . 1 7 9$ </td><td> $- 8 . 6 1 1 \pm 0 . 3 9 6$ </td><td> $3 . 9 8 3 \pm 0 . 0 2 6$ </td></tr></table>

Values are reported as mean ± sample standard deviation across three training seeds, each evaluated over 500 stochastic-policy episodes.

Table 3: Reward recovery of MA-UITD on continuous-control tasks with held-out demonstrations.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Simple Spread Agent mean</td><td colspan="2">Simple Push</td><td colspan="2">Gems</td></tr><tr><td>Adversary</td><td>Good</td><td>Red</td><td>Blue</td></tr><tr><td>PCC↑</td><td> $0 . 8 2 8 \pm 0 . 0 3 3$ </td><td> $0 . 3 3 9 \pm 0 . 0 2 2$ </td><td> $0 . 7 1 7 \pm 0 . 2 1 9$ </td><td> $0 . 6 6 3 \pm 0 . 1 2 9$ </td><td> $0 . 6 4 3 \pm 0 . 1 3 9$ </td></tr><tr><td>SCC↑</td><td> $0 . 3 8 8 \pm 0 . 0 8 9$ </td><td> $0 . 3 2 2 \pm 0 . 0 9 3$ </td><td> $0 . 3 9 6 \pm 0 . 0 3 5$ </td><td> $0 . 2 9 4 \pm 0 . 0 2 8$ </td><td> $0 . 3 0 5 \pm 0 . 0 1 0$ </td></tr></table>

Correlations are computed over all transitions in each seed’s held-out 500-episode test split and reported as mean ± sample standard deviation across three seeds.

Application to real animal trajectories without trajectory binning. We next evaluated MA-UITD using joint trajectories from pairs of mice performing a social-foraging task (Cheng et al., 2025). In this task, the two animals had to cooperate by navigating to the same reward zone to obtain reward (Fig. 3a). Over training, pairs developed stable asymmetric behavioral roles, with one animal acting as a leader and the other as a follower, potentially reflecting distinct internal reward functions shaped by uncertainty under decentralized, simultaneous movement. Because MA-UITD models a separate reward function for each agent, it can directly represent this form of behavioral asymmetry. Without prior knowledge, however, an animal’s internal reward could depend on arbitrary task variables, leaving inverse reward inference severely underconstrained. We therefore restricted the reward representation to the two animals’ positions in the arena and their inter-animal distance, defining reward functions on separate self-position, other-position, and distance components.

Figure 3b,c shows distinct learned reward structures for a representative leader–follower pair. The leader’s learned reward depended primarily on its own spatial location, whereas the follower’s position-dependent contributions were weaker relative to its proximity-dependent component. This pattern was consistent across mouse pairs: follower rewards increased strongly at short inter-animal distances, while leader rewards showed substantially weaker modulation by distance (Fig. 3d,e). These observations support leader–follower dynamics in which the leader navigates according to spatial goals and the follower is motivated to remain close to the leader.

![](images/71e81c773261f76d90a80f4d34de61e39c4612f912adfff6258f8170aba9d568.jpg)

b  
![](images/6aaa8e828d5369877f95e27b39c6060f79f13f3c73315ff22146fcb192804569.jpg)

c  
![](images/435ddc96de42af03c5c45b201c5a92d4a2a90414e633a6b97efbc69187743e84.jpg)

![](images/4907560a735af8895a672b62bc5c5f57c8e0789aaec70f17b6006425d440562a.jpg)

e  
![](images/bee42f1268846514404ef7de0bb767e9bf857ee219af1851b45df81b374067fa.jpg)  
Figure 3: Learned position and proximity reward components. a, Social-foraging task. b,c, Example self- and other-position reward maps, scaled with the distance component using a common within-animal reward range. d,e, Inferred distance-dependent rewards for individual pairs (thin) and their mean (thick). Gray shading marks the range below the common 5th-percentile threshold, where data were sparsely sampled. Followers valued proximal partner locations more highly than leaders (two-sided paired t-test at d = 0.07, P = 0.026; n = 5 pairs).

## 6 DISCUSSION

In summary, we developed MA-UITD, an offline multi-agent inverse reinforcement learning method that uses belief-conditioned successor features to recover explicit agent-specific rewards from fixed demonstrations in continuous-state environments with discrete or bounded continuous actions. The current method conditions reward inference on predictive beliefs estimated from observed behavior;

an important direction for future work is to infer each agent’s internal beliefs jointly with its reward function and to extend the framework to interactions involving larger numbers of agents. One limitation is that demonstrations provide little evidence about rewards outside the support of the observed states. Reward estimates in these regions therefore depend largely on regularization and the inductive biases of the chosen function approximator, and should be interpreted cautiously until supported by additional data.

## AI USE STATEMENT

In this work, we used generative AI tools to assist with language polishing, literature retrieval and discovery (e.g., identifying related work), and coding. It is important to note that generative AI tools were not involved in the ideation, research methodology, or experimental design. We have reviewed all AI-assisted work. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## REFERENCES

Zoe Ashwood, Aditi Jha, and Jonathan W Pillow. Dynamic inverse reinforcement learning for characterizing animal behavior. Advances in Neural Information Processing Systems, 35:29663– 29676, 2022.

Chris L. Baker, Julian Jara-Ettinger, Rebecca Saxe, and Joshua B. Tenenbaum. Rational quantitative attribution of beliefs, desires and percepts in human mentalizing. Nature Human Behaviour, 1(4):0064, March 2017. ISSN 2397-3374. doi: 10.1038/s41562-017-0064. URL https: //www.nature.com/articles/s41562-017-0064.

Oscar Bartra, Joseph T. McGuire, and Joseph W. Kable. The valuation system: A coordinatebased meta-analysis of BOLD fMRI experiments examining neural correlates of subjective value. NeuroImage, 76:412–427, 2013. doi: 10.1016/j.neuroimage.2013.02.063.

Hannah M. Bayer and Paul W. Glimcher. Midbrain dopamine neurons encode a quantitative reward prediction error signal. Neuron, 47(1):129–141, 2005. doi: 10.1016/j.neuron.2005.05.020.

Matteo Bettini, Amanda Prorok, and Vincent Moens. Benchmarl: Benchmarking multi-agent reinforcement learning. Journal of Machine Learning Research, 25(217):1–10, 2024. URL http://jmlr.org/papers/v25/23-1612.html.

Ethan S. Bromberg-Martin, Yang-Yang Feng, Takaya Ogasawara, J. Kael White, Kaining Zhang, and Ilya E. Monosov. A neural mechanism for conserved value computations integrating information and rewards. Nature Neuroscience, 27(1):159–175, January 2024. ISSN 1546-1726. doi: 10.1038/ s41593-023-01511-4. URL https://doi.org/10.1038/s41593-023-01511-4.

Alex J. Chan and Mihaela van der Schaar. Scalable Bayesian Inverse Reinforcement Learning, March 2021. URL http://arxiv.org/abs/2102.06483. arXiv:2102.06483 [cs.LG].

Yusi Chen, Yuan Cheng, Myungji Kwak, Angela Radulescu, and Herbert Zheng Wu. Unveiling value functions in social cognition with multi-agent inverse reinforcement learning. bioRxiv, 2026. doi: 10.1101/2024.10.09.617461. URL https://www.biorxiv.org/content/early/ 2026/04/08/2024.10.09.617461.

Yuan Cheng, Yusi Chen, Myungji Kwak, Ross P. Kempner, Rudramani Singha, Jared Winslow, Runqi Liu, Umais Khan, Tessa Spangler, Alvi Khan, Talmo Pereira, Matthew Whiteway, Evan S. Schaffer, Nuttida Rungratsameetaweemana, Nan Yang, and Herbert Zheng Wu. Asymmetric social representations in the prefrontal cortex for cooperative behavior. bioRxiv, 2025. doi: 10. 1101/2025.08.27.672249. URL https://www.biorxiv.org/content/early/2025/ 08/31/2025.08.27.672249.full.pdf.

Will Dabney, Zeb Kurth-Nelson, Naoshige Uchida, Clara Kwon Starkweather, Demis Hassabis, Remi´ Munos, and Matthew Botvinick. A distributional code for value in dopamine-based reinforcement learning. Nature, 577(7792):671–675, 2020. doi: 10.1038/s41586-019-1924-6. URL https: //doi.org/10.1038/s41586-019-1924-6.

Angelos Filos, Clare Lyle, Yarin Gal, Sergey Levine, Natasha Jaques, and Gregory Farquhar. PsiPhi-Learning: Reinforcement Learning with Demonstrations using Successor Features and Inverse Temporal Difference Learning. In Proceedings ofthe 38th International Conference on Machine Learning, pp. 3305–3317. PMLR, July 2021. URL https://proceedings.mlr.press/ v139/filos21a.html.

Till Freihaut and Giorgia Ramponi. On Feasible Rewards in Multi-Agent Inverse Reinforcement Learning, November 2025. URL http://arxiv.org/abs/2411.15046. arXiv:2411.15046 [cs.LG].

Keisuke Fujii, Kazushi Tsutsui, Yu Teshima, Makoto Itoh, Naoya Takeishi, Nozomi Nishiumi, Ryoya Tanaka, Shunsuke Shigaki, and Yoshinobu Kawahara. Data-driven simulator of multi-animal behavior with unknown dynamics via offline and online reinforcement learning, October 2025. URL http://arxiv.org/abs/2510.10451. arXiv:2510.10451 [cs.LG].

Michael Gutmann and Aapo Hyvarinen. Noise-contrastive estimation: A new estimation principle¨ for unnormalized statistical models. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 297–304. JMLR Workshop and Conference Proceedings, 2010.

Nathaniel Haynam, Adam Khoja, Dhruv Kumar, Vivek Myers, and Erdem Bıyık. Multi-agent inverse Q-learning from demonstrations. In 2025 IEEE International Conference on Robotics and Automation (ICRA), pp. 15021–15028. IEEE, 2025. doi: 10.1109/ICRA55743.2025.11128370. URL https://doi.org/10.1109/ICRA55743.2025.11128370.

Julian Jara-Ettinger. Theory of mind as inverse reinforcement learning. Current Opinion in Behavioral Sciences, 29:105–110, 2019.

Jingyang Ke, Feiyang Wu, Jiyi Wang, Jeffrey Markowitz, and Anqi Wu. Inverse reinforcement learning with switching rewards and history dependency for characterizing animal behaviors. arXiv preprint arXiv:2501.12633, 2025. doi: 10.48550/arXiv.2501.12633.

Diederik P. Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization, January 2017. URL http://arxiv.org/abs/1412.6980. arXiv:1412.6980 [cs.LG].

Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization, January 2019. URL http://arxiv.org/abs/1711.05101. arXiv:1711.05101 [cs.LG].

Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-agent actorcritic for mixed cooperative-competitive environments. Neural Information Processing Systems (NIPS), 2017.

Richard D. McKelvey and Thomas R. Palfrey. Quantal Response Equilibria for Normal Form Games. Games and Economic Behavior, 10(1):6–38, July 1995. ISSN 0899-8256. doi: 10.1006/game.1995.1023. URL https://www.sciencedirect.com/science/ article/pii/S0899825685710238.

Nir Moneta, Shany Grossman, and Nicolas W. Schuck. Representational spaces in orbitofrontal and ventromedial prefrontal cortex: task states, values, and beyond. Trends in Neurosciences, 47(12):1055–1069, December 2024. ISSN 0166-2236. doi: 10.1016/j.tins.2024.10.005. URL https://doi.org/10.1016/j.tins.2024.10.005.

Christian Neumeyer, Frans A. Oliehoek, and Dariu M. Gavrila. General-Sum Multi-Agent Continuous Inverse Optimal Control. IEEE Robotics and Automation Letters, 6(2):3429–3436, April 2021. ISSN 2377-3766, 2377-3774. doi: 10.1109/LRA.2021.3060411. URL https://ieeexplore. ieee.org/document/9357891/.

John P. O’Doherty, Alan Hampton, and Hackjin Kim. Model-based fMRI and its application to reward learning and decision making. Annals ofthe New York Academy ofSciences, 1104:35–53, 2007. doi: 10.1196/annals.1390.022.

Camillo Padoa-Schioppa and John A. Assad. Neurons in the orbitofrontal cortex encode economic value. Nature, 441(7090):223–226, 2006. doi: 10.1038/nature04676. URL https://doi. org/10.1038/nature04676.

Wolfram Schultz, Peter Dayan, and P. Read Montague. A neural substrate of prediction and reward. Science, 275(5306):1593–1599, 1997. doi: 10.1126/science.275.5306.1593. URL https:// doi.org/10.1126/science.275.5306.1593.

Ran Tian, Masayoshi Tomizuka, and Liting Sun. Learning Human Rewards by Inferring Their Latent Intelligence Levels in Multi-Agent Games: A Theory-of-Mind Approach with Application to Driving Data, March 2021. URL http://arxiv.org/abs/2103.04289. arXiv:2103.04289 [cs.AI].

Joey Velez-Ginorio, Max H Siegel, Joshua B Tenenbaum, and Julian Jara-Ettinger. Interpreting actions by attributing compositional desires. In CogSci, 2017.

Jiyi Wang, Jingyang Ke, Bo Dai, and Anqi Wu. Learning task-agnostic motifs to capture the continuous nature of animal behavior. arXiv preprint arXiv:2506.15190, 2025. doi: 10.48550/ arXiv.2506.15190. URL https://arxiv.org/abs/2506.15190.

Feiyang Wu, Ye Zhao, and Anqi Wu. Distributional inverse reinforcement learning. arXiv preprint arXiv:2510.03013, 2025. doi: 10.48550/arXiv.2510.03013. URL https://arxiv.org/abs/ 2510.03013.

Lantao Yu, Jiaming Song, and Stefano Ermon. Multi-Agent Adversarial Inverse Reinforcement Learning. In Proceedings of the 36th International Conference on Machine Learning, pp. 7194– 7201. PMLR, May 2019. URL https://proceedings.mlr.press/v97/yu19e.html.

Brian D Ziebart, Andrew L Maas, J Andrew Bagnell, Anind K Dey, et al. Maximum entropy inverse reinforcement learning. In Aaai, volume 8, pp. 1433–1438. Chicago, IL, USA, 2008.

## A ALGORITHMS

Algorithm 1 Multi-Agent Unilateral Inverse Temporal Difference Learning (MA-UITD)   
Input: Reward-free joint demonstrations $\overline { { \mathcal { D } ; } }$ number of agents $n ;$ discount $\gamma ;$ negatives ${ \overline { { K } } } ;$   
temperature τ<sub>NCE</sub>; proposal distributions $\{ \nu _ { i } \} _ { i = 1 } ^ { n } ;$ learning rate α; regularization coefficient $\lambda _ { w } .$   
Output: Shared reward feature parameters $\pmb { \theta } ^ { \Phi }$ and agent-specific parameters $\{ \eta _ { i } , w _ { i } \} _ { i = 1 } ^ { n } .$   
1: Initialize $\theta ^ { \Phi } , \{ \eta _ { i } , w _ { i } \} _ { i = 1 } ^ { n }$ , and target parameters $\{ \tilde { \eta } _ { i } \} _ { i = \cdot } ^ { n }$ 1   
2: while training budget remains do   
3: Sample a minibatch $\boldsymbol { B } \sim \mathcal { D }$   
4: for $i = 1 , \ldots , n$ do   
5: Sample $K$ unilateral alternatives from $\nu _ { i } ( \cdot \mid s )$   
6: Compute $\mathcal { L } _ { i } ^ { \mathrm { N C E } }$ using Equation (10)   
7: $\pmb { \eta } _ { i }  \pmb { \eta } _ { i } - \mathrm { \dot { \alpha } } \nabla _ { \pmb { \eta } _ { i } } \mathcal { L } _ { i } ^ { \mathrm { N C E } }$   
8: $w _ { i }  w _ { i } - \alpha \nabla _ { w _ { i } } ^ { \cdot } \big ( \mathcal { L } _ { i } ^ { \mathrm { N C E } } + \lambda _ { w } \| w _ { i } \| _ { 1 } \big )$   
9: end for   
10: Compute $\begin{array} { r } { { \mathcal { L } } ^ { \mathrm { U I T D } } = n ^ { - 1 } \sum _ { i = 1 } ^ { n } { \mathcal { L } } _ { i } ^ { \mathrm { U I T D } } } \end{array}$ using Equation (12)   
11: $\pmb { \theta } ^ { \Phi }  \pmb { \theta } ^ { \Phi } - \alpha \nabla _ { \pmb { \theta } ^ { \Phi } } \mathcal { L } ^ { \mathrm { U I T D } }$   
12: for $i = 1 , \ldots , n$ do   
13: $\pmb { \eta } _ { i } \gets \pmb { \eta } _ { i } - \alpha \nabla _ { \pmb { \eta } _ { i } } \mathcal { L } ^ { \mathrm { U I T D } }$   
14: Update $\tilde { \eta } _ { i }$ from $\eta _ { i }$   
15: end for   
16: end while   
17: return $\widehat { r } _ { i } ( s ) = \Phi ( s ; \pmb { \theta } ^ { \Phi } ) ^ { \top } w _ { i }$ for all i

## B IMPLEMENTATION DETAILS

For Simple Spread, the reward-feature network takes the scalar agent–landmark and inter-agent distances as input, whereas the BCSF networks take the agent and landmark positions, agent velocities, and each agent’s own action. For Simple Push, the reward-feature network takes the agent and landmark positions as input, while the BCSF networks additionally receive the agent velocities and each agent’s own action. For Gems, a shared torso network, MLP([32, 32]), encodes the agent positions. The reward-feature head operates on this shared state representation, while each BCSF head additionally receives the corresponding agent’s own action. To ensure a fair comparison, all algorithms were provided with the same task-specific state and action information.

## B.1 EXPERT DEMONSTRATIONS.

For the discrete-action versions of Simple Spread and Simple Push, we generate demonstrations using expert policies trained with the multi-agent ACKTR implementation of Yu et al. (2019). For their continuous-action versions and for both versions of the Gems game, we use expert policies trained with MASAC (Bettini et al., 2024). The Gems datasets contain an equal mixture of fully cooperative and fully competitive expert behavior. Across tasks, the demonstration datasets contain between 500 and 4,000 episodes.

## B.2 MA-UITD ON DISCRETE ACTION DATASETS

MA-UITD naturally generalizes to datasets with discrete action spaces by simply replacing NCE-loss in Eq. (10) with a Maximum Likelihood Estimation(MLE) loss

$$
\mathcal { L } _ { i } ^ { \mathrm { M L E } } ( \eta _ { i } , w _ { i } ) = - \mathbb { E } _ { ( s , a _ { i } ) \sim \mathcal { D } } \left[ \log \frac { \exp ( \bar { \Psi } _ { i } ( s , a _ { i } ; \eta _ { i } ) ^ { \top } w _ { i } / \tau _ { \mathrm { M L E } } ) } { \sum _ { \tilde { a } _ { i } \in A _ { i } } \exp ( \bar { \Psi } _ { i } ( s , \tilde { a } _ { i } ; \eta _ { i } ) ^ { \top } w _ { i } / \tau _ { \mathrm { M L E } } ) } \right] .\tag{13}
$$

## B.3 NOISE DISTRIBUTION

Let $\rho _ { E } ( s , \mathbf { a } )$ denote the state–joint-action occupancy induced by the expert demonstrations, and let

$$
\rho _ { E , i } ( s , a _ { i } ) = \int \rho _ { E } ( s , \mathbf { a } ) \mathrm { d } \mathbf { a } _ { - i }\tag{14}
$$

be its marginal for agent i. We draw a transition $( s ^ { \prime } , \mathbf { a } ^ { \prime } )$ uniformly from $\mathcal { D } _ { \mathrm { t r a i n } }$ and use $a _ { i } ^ { \prime }$ as a negative action at the current state s. Equivalently, the noise distribution is the expert’s marginal action distribution,

$$
\zeta _ { i } ( a _ { i } ) = \int { \rho _ { E , i } ( s ^ { \prime } , a _ { i } ) \mathrm { d } s ^ { \prime } } ,\tag{15}
$$

approximated empirically using the training demonstrations.

## B.4 CONTINUOUS POLICY ROLLOUT

Rollout initial states were generated using a fixed task-specific random seed, independently of training, and were shared across all model seeds. At each state, we approximated each agent’s continuous Boltzmann policy using the same candidate-based procedure. Specifically, we generated K = 256 low-discrepancy candidate actions from the agent’s bounded action domain and evaluated

$$
\begin{array} { r } { \widehat Q _ { i } ( s , a _ { i } ) = \overline { { \Psi } } _ { i } ( s , a _ { i } ) ^ { \top } w _ { i } . } \end{array}\tag{16}
$$

The executed action was sampled from the categorical approximation

$$
\widehat { \pi } _ { i } ( a _ { i } \mid s ) = \frac { \exp \Bigl ( \widehat { Q } _ { i } ( s , a _ { i } ) / \tau \Bigr ) } { \sum _ { a _ { i } ^ { \prime } \in A _ { i , K } } \exp \Bigl ( \widehat { Q } _ { i } ( s , a _ { i } ^ { \prime } ) / \tau \Bigr ) } , \qquad \tau = 1 ,\tag{17}
$$

where $\mathcal { A } _ { i , K }$ denotes the candidate set. Candidate-generation and policy-sampling seeds were fixed across model seeds, yielding paired evaluation conditions.

## B.5 HYPERPARAMETERS

We conducted a comprehensive hyperparameter search for all three algorithms included in the comparison. Hyperparameters used by MA-UITD implementations are listed in the table 4, 5, 6.

Table 4: MA-UITD hyperparameters for discrete action datasets
<table><tr><td>Hyperparameter</td><td>Simple Spread</td><td>Simple Push</td><td>Gems</td></tr><tr><td>Reward-feature approximator Φ</td><td>MLP([128, 128])</td><td>MLP([128, 128])</td><td>MLP([64, 32])</td></tr><tr><td>BCSF approximator  $\bar { \Psi } _ { i }$ </td><td> $\mathbf { M L P } ( [ 2 5 6 , 2 5 6 ] )$ </td><td>MLP([256, 256])</td><td>MLP([64, 64])</td></tr><tr><td>Activation function</td><td>SiLU</td><td>SiLU</td><td>LeakyReLU, slope 0.01</td></tr><tr><td> $\ell _ { 1 }$  coefficient</td><td>0.05</td><td>0.05</td><td>0.001 0.3</td></tr><tr><td>Dropout rate</td><td>0</td><td>0</td><td></td></tr><tr><td>Number of dimensions in Φ</td><td>6</td><td>128</td><td>10</td></tr><tr><td>n-step return</td><td>4</td><td>4</td><td>1</td></tr><tr><td>Discount factor γ</td><td> $_ { 0 . 9 5 }$ </td><td> $_ { 0 . 9 5 }$ </td><td> $_ { 0 . 9 9 }$ </td></tr><tr><td>Polyak coefficient τ</td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Rationality τMLE</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Optimiser</td><td>AdamW(Loshchilov &amp; Hutter, AdamW(Loshchilov &amp; Hutter, AdamW(Loshchilov 2019) for  $\Phi , { \bar { \Psi } } _ { i }$  (lr=  $1 0 ^ { - 4 } ,$  2019) for wd=  $1 0 ^ { - 5 } ) ;$  Adam(Kingma  $\&$  wd= 10 Ba, 2017) for  $w _ { i } ( \mathrm { l r } = 1 0 ^ { - \overline { { 4 } } } )$  Ba, 2017) f  $\mathrm { o r } w _ { i } ( \mathrm { l r } = \mathrm { i } 0 ^ { - 4 } )$ </td><td>&amp;  $\Phi , { \bar { \Psi } } _ { i }$  2019) for  $\Phi , { \bar { \Psi } } _ { i }$   $^ { - 5 } ) ;$  Adam(Kingma &amp; Adam(Kingma &amp; Ba,  $\mathrm { l r = \delta \ 1 0 ^ { - 4 } , }$ </td><td> $\mathrm { ( l r = }  &  \ 1 0 ^ { - 4 } ,$  ter,</td></tr></table>

Table 5: MA-UITD hyperparameters for continuous-action datasets
<table><tr><td>Hyperparameter</td><td>Simple Spread</td><td>Simple Push</td><td>Gems</td></tr><tr><td>Reward-feature approximator Φ</td><td>MLP([128, 128])</td><td> $\mathtt { M L P } ( [ 1 2 8 , 1 2 8 ] )$ </td><td> $\mathbf { M L P } ( [ 6 4 , 3 2 ] )$ </td></tr><tr><td>BCSF approximator ī i</td><td> $\mathrm { M L P } ( [ 2 5 6 , 2 5 6 ] )$ </td><td> $\mathrm { M L P } ( [ 2 5 6 , 2 5 6 ] )$ </td><td>MLP([128, 64])</td></tr><tr><td>Activation function</td><td>SiLU</td><td>SiLU</td><td>LeakyReLU, slope 0.02</td></tr><tr><td>l1 coefficient</td><td>0.05</td><td>0.05</td><td>0.001 0.3</td></tr><tr><td>Dropout rate</td><td>0</td><td>0</td><td></td></tr><tr><td>Number of dimensions in Φ</td><td>64</td><td>128</td><td>16</td></tr><tr><td>n-step return</td><td></td><td></td><td>1</td></tr><tr><td>Discount factor γ</td><td>0.95</td><td></td><td>0.99</td></tr><tr><td>Polyak coefficient τ</td><td> $1 0 ^ { - 4 }$ </td><td> $_ { 0 . 9 5 }$   $1 0 ^ { - 4 }$ </td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Rationality TNCE</td><td></td><td></td><td>1</td></tr><tr><td>Optimiser</td><td>1 AdamW(Loshchilov &amp; Hutter, AdamW(Loshchilov &amp; Hutter, AdamW(Loshchilov 2019) for  $\Phi , { \bar { \Psi } } _ { i } ~ ( \mathrm { l r } = ~ 1 0 ^ { - 4 }$  2019) for  $\mathrm { w d } = \ 1 0 ^ { - 5 } ) ;$  Adam(Kingma &amp;  $\mathrm { w d } = \ 1 0 ^ { - 5 } ) ;$ </td><td> $\Phi , { \bar { \Psi } } _ { i } ( \mathrm { l r } = 1 0 ^ { - 4 } ,$  Adam(Kingma &amp; Adam(Kingma &amp; Ba,</td><td>ter, 2019) for</td></tr></table>

Table 6: MA-UITD hyperparameters for real animal trajectories
<table><tr><td colspan="2">Hyperparameter</td></tr><tr><td>State representation</td><td>Physical-order neck positions  $( x _ { 1 } , y _ { 1 } , x _ { 2 } , y _ { 2 } )$ </td></tr><tr><td>Action representation</td><td>Own observed 2-D neck displacement</td></tr><tr><td>Reward-feature approximator Φ</td><td>Position MLP([128, 64]); distance MLP([64, 32])</td></tr><tr><td>BCSF approximator ī i</td><td>MLP([256, 128])</td></tr><tr><td>Activation function</td><td>LeakyReLU, slope 0.01</td></tr><tr><td> $\ell _ { 1 }$  coefficient</td><td>0.05</td></tr><tr><td>Dropout rate</td><td>0.3 for both Φ and īi</td></tr><tr><td>Number of dimensions in Φ</td><td> $2 0 ( 8 + 8 + 4 )$ </td></tr><tr><td>n-step return</td><td>1</td></tr><tr><td>Discount factor γ</td><td> $_ { 0 . 9 9 }$ </td></tr><tr><td>Polyak coefficient τ</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Optimiser</td><td>AdamW(Loshchilov &amp; Hutter, 2019) for Φ, Ψi  $( \mathrm { l r } = 1 0 ^ { - 4 } , \mathrm { w d } = 1 0 ^ { - 5 } ) ;$  Adam(Kingma  $\& B a , 2 0 1 7 )$  for  $w _ { i } ( \mathrm { l r } = 1 0 ^ { - 4 } )$ </td></tr></table>

## B.6 COMPUTE RESOURCES

All experiments were conducted on a workstation equipped with a 12-core AMD Ryzen 9 9900X CPU, 64 GB of system memory, and a single NVIDIA GeForce RTX 5080 GPU with 16 GB of memory. Each training run used one GPU. Experiments were implemented in Python 3.10.20 using PyTorch 2.11.0 and CUDA 12.8.

## C EXTENDED RESULTS

## C.1 REWARD RECOVERY FOR MA-UITD AND BASELINES

Details for reward recovery performance across MA-UITD and baselines are shown in Table 7

Table 7: Held-out immediate-reward recovery.
<table><tr><td>Method</td><td>Simple Spread</td><td colspan="2">Simple Push</td><td>Gems</td></tr><tr><td></td><td>PCC ↑ SCC↑</td><td>Adversary PCC ↑ Adversary SCC↑</td><td>Good PCC ↑ Good SCC↑</td><td>PCC ↑ SCC↑</td></tr><tr><td></td><td> $0 . 6 4 1 \pm 0 . 0 1 4$ </td><td> $0 . 1 7 8 \pm 0 . 0 2 2$ </td><td> $0 . 5 8 1 \pm 0 . 0 5 7$ </td><td> $0 . 2 3 0 \pm 0 . 0 2 1$ </td></tr><tr><td>MA-AIRL</td><td> $0 . 7 4 5 \pm 0 . 0 1 3$ </td><td> $0 . 1 4 5 \pm 0 . 0 2 4$ </td><td> $0 . 3 7 5 \pm 0 . 0 7 0$ </td><td> $0 . 1 4 5 \pm 0 . 0 2 1$ </td></tr><tr><td></td><td> $- 0 . 1 4 0 \pm 0 . 0 1 6$ </td><td> $0 . 0 8 1 \pm 0 . 0 1 2$ </td><td> $- 0 . 0 0 3 \pm 0 . 0 0 3$ </td><td> $0 . 1 4 2 \pm 0 . 0 2 9$ </td></tr><tr><td>MAMQL</td><td> $- 0 . 1 5 4 \pm 0 . 0 1 9$ </td><td> $- 0 . 0 0 0 \pm 0 . 0 1 9$ </td><td> $- 0 . 0 7 4 \pm 0 . 0 1 4$ </td><td> $0 . 1 7 4 \pm 0 . 0 3 7$ </td></tr><tr><td></td><td> $\mathbf { 0 . 9 4 0 \pm 0 . 0 0 8 }$ </td><td> $\mathbf { 0 . 5 6 4 \pm 0 . 1 8 4 }$ </td><td> $\mathbf { 0 . 8 5 4 \pm 0 . 0 1 6 }$ </td><td> $\mathbf { 0 . 7 1 1 \pm 0 . 0 1 4 }$ </td></tr><tr><td>MA-UITD</td><td> $\mathbf { 0 . 9 4 2 \pm 0 . 0 0 7 }$ </td><td> $\mathbf { 0 . 3 7 2 \pm 0 . 1 3 8 }$ </td><td> $\mathbf { 0 . 5 5 4 \pm 0 . 0 4 0 }$ </td><td> $\mathbf { 0 . 3 2 1 \pm 0 . 0 0 9 }$ </td></tr></table>

Entries are mean ± sample standard deviation across three training seeds on the fixed 500-episode test split; the upper and lower values in each cell are PCC and SCC, respectively.

## C.2 POLICY IMITATION FOR THE Gems

Details for policy imitation in Gems are shown in Table 8

Table 8: Policy imitation for the Gems
<table><tr><td>Method</td><td>Purple gems (%)</td><td>Red/Blue gems (%)</td></tr><tr><td>Ground Truth</td><td>50</td><td>50</td></tr><tr><td>MA-AIRL</td><td> $0 . 0 \pm 0 . 0$ </td><td> $1 0 0 . 0 \pm 0 . 0$ </td></tr><tr><td>MAMQL</td><td> $5 0 . 9 \pm 1 0 . 7$ </td><td> $4 9 . 1 \pm 1 0 . 7$ </td></tr><tr><td>MA-UITD</td><td> $4 4 . 8 \pm 1 . 4$ </td><td> $5 5 . 2 \pm 1 . 4$ </td></tr></table>

Percentage of gems collected. Values are reported as the mean ± sample standard deviation across three training seeds, each evaluated over 500 stochastic-policy episodes.

## C.3 REWARD VISUALIZATION

In this section, we provide a comprehensive visual analysis of the reward functions recovered from both discrete- and continuous-action datasets(Fig. 4, 5, 6, 7).

![](images/862b8b6c5653bb158b740b13b70e6311d6c89c702670e6237db08fd1b1ddd1ad.jpg)

## C.3.1 RECOVERED REWARD IN DISCRETE-ACTION ENVIRONMENTS

![](images/eddce031bc03f2b5e97a62d863cfb6ef19da2e45804a1afbd391664b9bf0d8c2.jpg)

![](images/e61875c92c1a58751723fbca8a321cea2aef69b23447b95412a5443873f0ed5b.jpg)

![](images/e4eaa85e692eac5b249fc15117abcd285116bab2987860e6ee824b3383ceb6c1.jpg)

![](images/078d3d386dfbe307a9df8c48e59ff6a97e616cd35530cdf6b1c1c9f97cfd5942.jpg)  
Figure 4: Reward recovery in discrete Simple Spread. a,b, Agent 1 reward versus each agent’s nearest-landmark distance; c,d, corresponding Agent 2 results. e,f, Agent 1 and Agent 2 rewards versus the five indicated arrival-state distances. Curves show the true and raw MA-UITD, MA-AIRL, and MAMQL rewards, averaged in equal-width bins over 4,836 held-out transitions (500 episodes).

![](images/0c474667bafeda8b88c5503003b4e96f8913b0eb1814cac0df4c09cc64f6a9b8.jpg)  
Figure 5: Reward recovery in discrete Simple Push. a,b, Adversary reward versus the good agent’s and its own landmark distances, respectively. c, Good-agent reward versus its landmark distance. Curves show the true and raw MA-UITD, MA-AIRL, and MAMQL rewards, averaged in equal-width bins over 12,500 held-out transitions (500 episodes).

![](images/6883fe58834fb43f397e02fcd0a96ca96ddcb7087803ea75a2ab0a706329146a.jpg)

![](images/e4eac5c85a9ce8a9b5d79f77b59261954769602a99e72a7a080b31fbe4d3a95c.jpg)

![](images/680c2e9de21e441674afbec94094193435495f9fbdeac3cdbb0c4c90350d3658.jpg)

![](images/29218865dee5e715c03c14c49a4057199850ee6cf5788f6b688b54b8a0839cfc.jpg)

![](images/09f4f3ba8ab5f691c24aad0040afff9db55ea5272f9d65a53f3f6f51e7cfde91.jpg)

![](images/2ccd34a391039a17f098108a0d5ed308119edf49c40810ee8eba3b35d95981a0.jpg)

g Reward maps while moving the red agent  
![](images/1f193fe6615a35d8886d51cb60cc947cd7561c8ba90b44f943ad7f5aeeb70bc9.jpg)

![](images/47990288ade29b2bd6a86cb0a2a87a6157e68268c9d02ebe174aa479e171b09f.jpg)

![](images/df69fd3f4191843759e38b8982ae6689a38baea586bd078e9492e3e41daa42eb.jpg)

![](images/92355a6f5dfe6d21aaf27d061b8081cbbc9072c2f99454fd99a47dc9c0a60b1f.jpg)

![](images/6e6219188b2402858b2f5438f46fe34aa4cd31d8b020b9a9cddc766e0aa72b71.jpg)

![](images/425e7761a2251cca510ba6cf57472f73d1e8c0837062da9089b0e1a3a1f26e9e.jpg)

![](images/ec33a8e644a4c44f31acc5e468cefee6adc1cb1923776d296db99bd873971347.jpg)

![](images/be8eaef2d466562a012578a63d55091105ee7d2f7f9b2ff216433587a04bf61a.jpg)  
Figure 6: Reward recovery in discrete Gems. a–c, Red-agent reward versus arrival distance to the red, blue, and purple gems; d–f, corresponding blue-agent results. Curves compare ground truth with raw MA-UITD, MA-AIRL, and MAMQL rewards, averaged in equal-width bins over 10,274 held-out transitions (500 episodes). g, Spatial reward maps obtained by moving the red agent while fixing the blue agent at the purple gem. Rows show red- and blue-agent rewards; columns show ground truth and the three recovered rewards. Circles mark gem locations and collection radii. Each map uses an independent symmetric scale centered at zero (blue, negative; red, positive).

![](images/6f1aec58e43fb6818059cc8bdcf6644511f5b48faf3da6be62faa24e328b9c23.jpg)  
Figure 7: Reward recovery across continuous-action environments. a, Reward–distance relationships in Simple Push. b,c, Red- and blue-agent relationships in Gems. d,e, Agent 1 and Agent 2 rewards versus five arrival-state distances in Simple Spread. f, Spatial rewards obtained by moving Agent 2 while fixing Agent 1 and the landmarks. Black and orange denote true and raw MA-UITD rewards. Curves are binned means from 500 held-out episodes per task; Spatial maps use independent raw color scales.