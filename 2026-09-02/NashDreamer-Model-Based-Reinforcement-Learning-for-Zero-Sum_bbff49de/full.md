# NashDreamer: Model-Based Reinforcement Learning for Zero-Sum Imperfect-Information Games

Tomáš Holecekˇ

Viliam Lisý

Artificial Intelligence Center, Department of Computer Science Faculty of Electrical Engineering, Czech Technical University in Prague holecto6@fel.cvut.cz, viliam.lisy@agents.fel.cvut.cz

## Abstract

Model-based reinforcement learning (MBRL) has achieved remarkable results in single-agent domains, yet its extension to competitive imperfect information games (IIGs) remains underexplored. In multi-agent settings, opponent-induced non-stationarity complicates the learning process, and decentralized model learning faces severe identifiability barriers, which we argue make centralized model learning a mathematical necessity. Building on this analysis, we propose NashDreamer, a principled MBRL framework for two-player zero-sum IIGs. It introduces a centralized Multi-Agent Recurrent State-Space Model (MARSSM) that decouples environment dynamics from the effect of players’ strategies on their individual observations. NashDreamer is designed to use arbitrary policy gradient algorithms and inherits their convergence guarantees towards Nash equilibria under an idealized model. Empirical evaluations across four benchmark games demonstrate that NashDreamer substantially improves sample efficiency over model-free baselines early in the training. Finally, we theoretically analyze the architecture’s optimization landscape, identifying the vulnerability of the Dreamer family of algorithms to posterior collapse in stochastic environments. We leave it as an open challenge.

## 1 Introduction

Model-based reinforcement learning (MBRL) has emerged as a powerful paradigm for sampleefficient learning. MuZero [34] and its extensions demonstrate that learned latent models can serve as effective planning abstractions, while the Dreamer series [12–15] shows that world models can generate imagination trajectories that reduce sample requirements. These approaches have matched or surpassed the state of the art across diverse single-agent domains, including robotics [41], autonomous driving [11, 18], and nuclear fusion control [7].

Complex imperfect information adversarial domains currently force a trade between fidelity and tractability. Automated cyber defense is studied on emulators made fast by simplifying the attacker, the defender or the network itself [19], and dogfight simulators, such as the one in Lu and Chen [25] hand each aircraft a complete frame rather than only what lies within its own sensor range. Sample-efficient generative MBRL offers the opposite trade, recovering speed from a small budget of interactions with the unsimplified environment. To lay the foundation, we investigate the Dreamerstyle paradigm for multi-player imperfect information games (IIGs), leveraging generative world models purely to synthesize large artificial datasets for scalable, sample-based policy optimization.

Adapting MBRL to IIGs, however, exposes fundamental identifiability barriers. In a standard decentralized setting, a learning agent cannot distinguish between the objective stochastic dynamics of the environment and the shifting policy of its opponent [1, 28]. We argue that this “policy-physics entanglement” dictates that Centralized Training with Decentralized Execution (CTDE) is not merely an architectural convenience, but a mathematical necessity to learn a sound, stationary world model.

We further provide the following contributions: (1) We propose NashDreamer, a principled CTDE framework for two-player zero-sum IIGs. It introduces a centralized Multi-Agent Recurrent State-Space Model (MARSSM), which can be combined with game-theoretically sound policy-gradient methods, to facilitate convergence towards the Nash Equilibrium. (2) We show that DreamerV3 “KL-balancing” can cause posterior collapse, highlighting the challenge of mitigating latent space nonstationarity while maintaining a theoretically sound framework. (3) We provide thorough empirical evaluations showing that NashDreamer improves sample efficiency compared to model-free baselines across various domains.

NashDreamer is not meant to replace well-tuned model-free policy-gradient methods, which reach excellent policies given abundant simulator access [33]. It is a sample-efficiency layer agnostic to the optimizer running inside imagination. The appropriate comparison is thus a policy-gradient method $\bar { X }$ against NashDreamer[X] at an equal budget of real environment interactions. This is the regime of our motivating applications, such as automated cyber defense or expensive simulations, where a real interaction, rather than a gradient step, is the dominant cost.

## 2 Background

We formalize our target domains as two-player zero-sum (2p0s) simultaneous-move games. We use a custom variant of Partially Observable Stochastic Games (POSGs) inspired by [20]. We denote $\mathcal { N } = \{ 1 , 2 \}$ the player set with c for the chance player; S the state space with initial state s ; $\mathcal { A } = \mathcal { A } _ { 1 } \times \mathcal { A } _ { 2 }$ the joint action space, whose elements we write as $\boldsymbol { a } = ( a _ { 1 } , a _ { 2 } ) ; \mathbb { O } = \mathbb { O } _ { 1 } \times \mathbb { O } _ { 2 }$ is the joint observation space with per-player observations. $T : \mathcal { S } \times \mathcal { A }  \Delta \mathcal { S }$ is the (partial) transition function ; $O : S  \mathbb { O }$ is the (deterministic) observation function $; R : \mathcal { S } \times \mathcal { A }  \mathbb { R }$ is the zero-sum reward function $( R _ { 1 } = - R _ { 2 } ) ; P : { \mathcal { S } } \to 2 ^ { \mathcal { N } }$ identifies the acting players at each state; and $L _ { i } : { \mathcal { S } }  2 ^ { A _ { i } }$ <sup>i</sup> specifies legal actions for player i. States where $P ( s ) \bar { = } \bar { \varnothing }$ and the transition function is defined are chance nodes <sup>2</sup>; states with undefined transitions are terminal and other nodes are called decision nodes, where we assume $P ( s ) = \{ 1 , 2 \}$ . Turn-based games, such as Leduc Poker, are embedded by assigning a noop action to the inactive player (Appendix F).

A history $h = ( s _ { 0 } , a ^ { 0 } , \ldots , s _ { t } )$ is the full sequence of states and joint actions from the initial state, the ground-truth object no player observes directly; s(h) is its last state, through which we lift the state-indexed functions, writing $T ( h , a ) : = T ( s ( h ) , a )$ and likewise for O and ${ \check { L } } _ { i } .$ Player i observes only their action-observation history (AOH) $\vec { \tau _ { i } } ( \dot { h } ) = \big ( o _ { i } ^ { 0 } , a _ { i } ^ { 0 } , \ldots , o _ { i } ^ { t } \big )$ , and we assume perfect recall. The information set (infoset) of player i at h is $I _ { i } ( h ) \dot { = } \{ h ^ { \prime } : \tau _ { i } ( \ddot { h } ^ { \prime } ) = \tau _ { i } ( h ) \}$ , with $\mathcal { T } _ { i }$ the set of all of them. A policy $\pi _ { i } : \mathcal { T } _ { i }  \bar { \Delta } ( \mathcal { \bar { A } } _ { i } )$ conditions on $\tau _ { i }$ alone and never on $h ,$ legal-action masks entering only as an implementation-level restriction of its support. We use π for optimized policies and reserve $\mu$ for the joint sampling policy that collects data. We assume collective observability: the joint action-observation history identifies the true underlying history, i.e., $I _ { 1 } ( h ) \cap I _ { 2 } ( h ) = \{ h \} . \ ^ { 3 }$ It is a technical convenience rather than a restriction, since chance outcome eventually observed by some player can be deferred to the point where it is first observed [40]. It is also the assumption that makes public belief states well defined [5], and all our benchmarks satisfy it.

A Nash equilibrium (NE) [27] is a joint policy $\pi ^ { * }$ where no player can improve their utility by unilateral deviation. We measure distance to NE using NashConv [23]: NashConv(π) = $\begin{array} { r } { \dot { u ( { \mathbf B } { \mathbf R } ( \pi _ { 2 } ) , \pi _ { 2 } ) } - u ( \pi _ { 1 } , { \mathbf B } { \mathbf { R } } ( \pi _ { 1 } ) ) } \end{array}$ , where $\operatorname { B R } ( \pi _ { i } )$ is the best response of the player i’s opponent.

## 2.1 DreamerV3

DreamerV3 [14] learns a latent world model, the Recurrent State-Space Model (RSSM), for acting in POMDPs. The model state $m _ { t } = [ \hat { h } _ { t } , z _ { t } ]$ concatenates a deterministic recurrent state $\hat { h } _ { t }$ produced by a sequence model $\hat { h } _ { t } = \mathrm { s e q } ( m _ { t - 1 } , a _ { t - 1 } )$ and a stochastic categorical state $z _ { t }$ sampled from either

the posterior encoder $q _ { t } ^ { \theta } = \operatorname { e n c } ( { \hat { h } } _ { t } , o _ { t } )$ (during training with real observations) or the prior dynamics predictor $p _ { t } ^ { \theta } = \mathrm { d y n } ( \hat { h } _ { t } )$ (during imagination). The RSSM is trained end-to-end by optimizing:

$$
\mathcal { L } ^ { \mathrm { m o d e l } } = \sum _ { t } ^ { T } \beta ^ { \mathrm { e n c } } \mathcal { L } _ { t } ^ { \mathrm { e n c } } + \beta ^ { \mathrm { d y n } } \mathcal { L } _ { t } ^ { \mathrm { d y n } } + \beta ^ { \mathrm { p r e d } } \mathcal { L } _ { t } ^ { \mathrm { p r e d } } ,\tag{1}
$$

The prediction loss $\mathcal { L } _ { t } ^ { \mathrm { p r e d } }$ trains decoders to reconstruct rewards, continuation flags, and observations from the model state. The encoder and dynamics losses $\mathcal { L } _ { t } ^ { \mathrm { e n c } } = \mathrm { K L } ( q _ { t } ^ { \theta } \parallel \mathbf { s } \mathbf { g } ( p _ { t } ^ { \theta } ) )$ and $\mathcal { L } _ { t } ^ { \mathrm { d y n } } =$ $\mathrm { K L } ( \operatorname { s g } ( q _ { t } ^ { \theta } ) \parallel p _ { t } ^ { \theta } )$ are both KL divergences between the posterior and prior (sg denotes stop-gradient).

These asymmetric losses perform “KL balancing”: $\mathcal { L } _ { t } ^ { \mathrm { d y n } }$ trains the dynamics to match the encoder, while $\mathcal { L } _ { t } ^ { \mathrm { e n c } }$ serves as an encoder regularizer to remain predictable by the dynamics.

From trained model states, Dreamer generates imagination trajectories using the dynamics predictor, effectively amplifying the training data. For policy optimization, the actor uses REINFORCE with an entropy bonus, and the critic uses TD(λ).

## 2.2 Game-Theoretic Policy Optimization

Policy-gradient methods developed for single-agent problems are often not sound in imperfectinformation games: the online-learning dynamics they descend from are known to cycle rather than converge to an NE, in matrix games [26] and in sequential IIGs [31] alike. Regularizing toward a reference policy has proven the successful remedy, recovering last-iterate convergence.

Regularized Nash Dynamics (RNaD) [32] regularizes in reward space, subtracting a KL penalty against a reference policy $\pi ^ { \mathrm { r e g } }$ from the payoff, which preserves the zero-sum structure. Periodically resetting $\pi ^ { \mathrm { r e g } }  \pi$ then solves a sequence of regularized games whose solutions converge to an NE of the original one. Magnetic Mirror Descent (MMD) [36] instead regularizes in policy space, pairing a KL term toward a fixed magnet policy with a trust-region term on consecutive iterates. The difference in guarantees matters for the comparisons that follow: RNaD converges to an NE, whereas MMD under the fixed uniform magnet [33] that we adopt has convergence guarantees only to a quantal response equilibrium, so its asymptotic exploitability is bounded away from zero. Appendix A details both algorithms, including the off-policy simultaneous-move variant of RNaD [21] that we use.

## 3 World Models in Games: The Necessity of Centralized Training

Learning a world model without observing the opponent’s actions faces the following fundamental barriers preventing the use of contemporary decentralized approaches in adversarial environments (see Appendix D).

Action identifiability. For a given state s and ego action $a _ { 1 }$ , different opponent actions $a _ { 2 } \neq a _ { 2 } ^ { \prime }$ must produce distinct next state transition distributions: $T ( s , ( a _ { 1 } , a _ { 2 } ) ) \neq \hat { T } ( s , ( a _ { 1 } , a _ { 2 } ^ { \prime } ) )$ . When violated, action aliasing makes the data insufficient to disambiguate the causes of state changes [1, 24]. While this aliasing is benign for a purely forward-predictive model (as the resulting state is identical), it poses a fundamental identifiability barrier for any decentralized model attempting to reverse-engineer the objective environment dynamics independent of the opponent’s policy.

Policy-physics entanglement. Without observing $a _ { 2 }$ , a learner cannot distinguish true environment dynamics $\dot { T }$ from a modified dynamic $T ^ { \prime }$ where the opponent’s strategy artificially mimics the effects of T [1]. A decentralized model is therefore a joint-system model rather than an environment model. It captures how the world works against that specific opponent, failing to capture the objective environment dynamics [28].

Adversarial exploitation. In strictly competitive environments, a strong opponent will actively exploit this identifiability gap, deliberately selecting actions that maximize the mismatch between true physical reality and the ego-agent’s corrupted internal representation.

Centralized training bypasses these barriers entirely. By conditioning the world model on the joint action $( a _ { 1 } , a _ { 2 } )$ during training, the objective environment physics is perfectly decoupled from the opponent’s shifting strategy. We intentionally structure the world model so that it can be used with only the ego-agent partial observations at test time.

The barrier is directly observable: in perturbed Rock-Paper-Scissors, a decentralized model trained with the identical objective and optimizer learns the components of the game that do not depend on the opponent’s action, driving its error on observations, terminal flags and legal-action masks towards zero, while its reward error grows over training instead of falling, since without $a _ { 2 }$ the reward is not a function of its inputs. Appendix L reports the comparison.

Centralization also amplifies the usual benefits of a world model. Disentangling the stochasticity of the opponent’s policy from the environment dynamics reduces the real samples needed to learn the model: in Rock-Paper-Scissors, a tabular world model would require only 9 transitions, after which policy training can proceed entirely in imagination. A world model further decouples data collection from policy optimization, so agents may collect with highly exploratory policies while training purely on-policy inside imagination, avoiding the off-policy corrections causing prohibitive variance in IIGs.

## 4 NashDreamer

As established in Section 3, applying decentralized world models to adversarial environments inevitably results in policy-physics entanglement. To resolve this, we introduce the centralized Multi-Agent Recurrent State-Space Model (MARSSM) to learn the objective environment dynamics, while a separate recurrent network handles information aggregation for each player in a decentralized manner. Crucially, to preserve the causal chain of strategic deductions required for accurate information aggregation, we unroll complete trajectories starting at the initial state $t = 0$ for the world model training. We detail possible mechanisms allowing training on shorter, mid-trajectory chunks and their respective pitfalls in Appendix E.2

Definition 1 A centralized MARSSM is a tuple (seq, enc, dyn, iset), where: $\begin{array} { r l } { \hat { h } _ { t } } & { { } = } \end{array}$ $\mathrm { s e q } ( \mathrm { m } _ { \mathrm { t } - 1 } , ( \mathrm { a } _ { 1 } ^ { \mathrm { t } - 1 } , \mathrm { a } _ { 2 } ^ { \mathrm { t } - 1 } ) )$ is the sequence model, receiving the joint action; $q _ { t } ^ { \theta } = \mathrm { e n c } ( \hat { \mathrm { h } } _ { \mathrm { t } } , ( \mathrm { o } _ { 1 } ^ { \mathrm { t } } , \mathrm { o } _ { 2 } ^ { \mathrm { t } } ) )$ is the posterior encoder, inferring the stochastic state $z _ { t } \sim q _ { t } ^ { \theta }$ from the joint current observations; $p _ { t } ^ { \theta } = \mathrm { d y n } ( \hat { \mathrm { h } } _ { \mathrm { t } } )$ is the prior dynamics predictor, predicting the stochastic state $z _ { t } \sim p _ { t } ^ { \theta }$ without access to the current observations; $\hat { I } _ { i } ^ { t } = \mathrm { i s e t } ( \hat { \mathrm { I } } _ { \mathrm { i } } ^ { \mathrm { t } - 1 } , \mathrm { a } _ { \mathrm { i } } ^ { \mathrm { t } - 1 } , \mathrm { o } _ { \mathrm { i } } ^ { \mathrm { t } } )$ is the infoset model for player $i ,$ recurrently building the latent infoset embeddingfrom their own $A O H \tau _ { i }$ (Section 2).

The central model state $m _ { t } = [ \hat { h } _ { t } , z _ { t } ]$ aggregates the private observations of both players and handles the environment dynamics. The infoset model embeds each player’s AOH $\tau _ { i }$ into a fixed-size vector $\hat { I } _ { i } ^ { t }$ , serving as our latent infoset representation.

## 4.1 World Model Objective

Our predictor loss extends DreamerV3 with legal action prediction and per-player observation reconstruction. Given the centralized model state $m _ { t } = [ \hat { h } _ { t } , z _ { t } ]$ , the predictors decode the reward $R _ { 1 } ^ { t }$ for Player 1, a termination flag $d _ { t }$ , and, for each player i, the legal action mask $\ell _ { i } ^ { t }$ and observation $o _ { i } ^ { \hat { t } . }$

$$
\mathcal { L } _ { t } ^ { \mathrm { m p r e d } } = \mathcal { L } _ { t } ^ { \mathrm { r e w } } ( R _ { 1 } ^ { t } \mid m _ { t } ) + \mathcal { L } _ { t } ^ { \mathrm { B C E } } ( d _ { t } \mid m _ { t } ) + \sum _ { i \in \{ 1 , 2 \} } \left[ \mathcal { L } _ { t } ^ { \mathrm { B C E } } ( \ell _ { i } ^ { t } \mid m _ { t } ) + \mathcal { L } _ { t } ^ { \mathrm { M S E } } ( o _ { i } ^ { t } \mid m _ { t } ) \right] ,\tag{2}
$$

where $\mathcal { L } ^ { \mathrm { r e w } }$ follows the DreamerV3 categorical parameterization, BCE denotes binary cross-entropy (per element for the legal action mask), and MSE is mean squared error.

The latent infoset embeddings $\hat { I } _ { i } ^ { t }$ are trained by two further losses. The unifier loss requires the joint embeddings $( \hat { I } _ { 1 } ^ { t } , \hat { I } _ { 2 } ^ { t } )$ to reconstruct the centralized model state $m _ { t } ,$ , a target well defined precisely because of collective observability (Section $2 ) ;$ the individual loss requires each embedding to reconstruct its own player’s observation $o _ { i } ^ { t }$ and previous action $a _ { i } ^ { t - 1 }$ , so that it captures local context:

$$
\begin{array} { r l } { \mathscr { L } _ { t } ^ { \mathrm { i s t } } = \underbrace { \mathscr { L } _ { t } ^ { \mathrm { M S E } } ( \mathrm { s g } ( \hat { h } _ { t } ) \mid \hat { I } _ { 1 } ^ { t } , \hat { I } _ { 2 } ^ { t } ) + \mathscr { L } _ { t } ^ { \mathrm { C E } } ( \mathrm { s g } ( z _ { t } ) \mid \hat { I } _ { 1 } ^ { t } , \hat { I } _ { 2 } ^ { t } ) } _ { \mathrm { U i i f i e r ~ L o s s } } } & { + \displaystyle \sum _ { i \in \{ 1 , 2 \} } \underbrace { \left[ \mathscr { L } _ { t } ^ { \mathrm { M S E } } ( o _ { i } ^ { t } \mid \hat { I } _ { i } ^ { t } ) + \mathscr { L } _ { t } ^ { \mathrm { C E } } ( a _ { i } ^ { t - 1 } \mid \hat { I } _ { i } ^ { t } ) \right] } _ { \mathrm { I o d i s i a n a l ~ I _ { o e } } } } \end{array}
$$

Individual Loss

(3)

where CE is categorical cross-entropy.

The total model loss combines the standard DreamerV3 KL-balancing losses $\mathcal { L } _ { t } ^ { \mathrm { e n c } }$ and $\mathcal { L } _ { t } ^ { \mathrm { d y n } }$ (Section 2.1) with our prediction and infoset losses:

$$
\mathcal { L } ^ { \mathrm { m o d e l } } = \sum _ { t } ^ { T } \beta ^ { \mathrm { e n c } } \mathcal { L } _ { t } ^ { \mathrm { e n c } } + \beta ^ { \mathrm { d y n } } \mathcal { L } _ { t } ^ { \mathrm { d y n } } + \beta ^ { \mathrm { p r e d } } \mathcal { L } _ { t } ^ { \mathrm { m p r e d } } + \beta ^ { \mathrm { i s e t } } \mathcal { L } _ { t } ^ { \mathrm { i s e t } } ,\tag{4}
$$

where we set $\beta ^ { \mathrm { d y n } } = \beta ^ { \mathrm { p r e d } } = \beta ^ { \mathrm { i s e t } } = 1$ and $\beta ^ { \mathrm { e n c } } = 0 . 1$ . Following the DreamerV3 convention, all task losses carry weight 1 while the encoder regularizer is down-weighted, so the posterior is shaped primarily by the prediction targets and only mildly pulled toward the prior. We deliberately do not tune these coefficients per environment, with the exception of Leduc, where we showcase failure modes of the KL balancing loss in stochastic environments (Appendix J). Figure 1 visualizes two steps of world-model training.

## 4.2 Imagination

Imagination supplies the synthetic data on which the actor-critic is trained. As in DreamerV3, we start one imagined trajectory for every non-chance node of trajectories collected from the real environment, terminal nodes included. Goofspiel-5, for instance, yields five imagined trajectories per real one. The imagined volume per gradient step therefore scales with the length of the game rather than with the batch size alone. We deviate from DreamerV3 in where a rollout begins: instead of branching from the encountered node, every rollout restarts from the root state $t = 0$ and runs for the maximum trajectory length of the underlying game, for the reasons detailed in Appendix E.2.

Starting from a model state $m _ { 0 }$ obtained during world model training, each imagined step reconstructs the per-player legal actions, the reward and the termination flag from $m _ { t } ;$ samples each actor’s action from its policy conditioned on that player’s latent infoset and legal actions $^ 4 ;$ advances the sequence model on the joint action; samples the next stochastic state from the prior to form $m _ { t + 1 } ;$ and updates each player’s latent infoset from the observations reconstructed at $m _ { t + 1 }$ (see Appendix C.1).

That last reconstruction is a key difference from standard Dreamer, where the decoder is merely a training signal. Here the infoset model consumes observations, so the decoder becomes an essential generative component of the imagination pipeline. We acknowledge the risks introduced by explicit reconstruction of high-dimensional observations and discuss their mitigation in Appendix E.1.

Following the CTDE paradigm, the decentralized actor for player i is conditioned on $\hat { I } _ { i } ^ { t }$ alone, ensuring it cannot access the opponent’s private observations. The centralized critic, used exclusively during training, operates on the joint representation $( \hat { I } _ { 1 } ^ { t } , \hat { I } _ { 2 } ^ { t } )$ , grounding its value estimates in the full game history. All MARSSM parameters are frozen during the imagination phase.Figure 2 visualizes two imagined steps; Appendix C.1 states the step in full.

## 4.3 Actor-Critic

We choose not to use the DreamerV3 standard REINFORCE, as its fragility towards hyperparameter tuning is unsuitable for MBRL (Appendix N). Among the game-theoretically sound alternatives, such as MMD [36], we suggest RNaD because: (i) its last-iterate convergence guarantee is what our model-convergence result composes with (Corollary 2); (ii) it reportedly needs less domain-specific tuning, which matters here because the game it optimizes against is a learned, non-stationary object; and (iii) its off-policy simultaneous-move variant [21] is a natural fit for the replay/exploratory model learning. The choice is nevertheless modular as we demonstrate by also testing with MMD.

We train the actor-critic networks concurrently on both real and imagined trajectories.

While the MARSSM imagination pipeline theoretically permits purely on-policy optimization, we deliberately employ the off-policy formulation in our empirical setup to ensure exact parity to our (necessarily) off-policy model-free RNaD with replay buffer and to permit exploratory policies for the model learning.

![](images/7968f935bc888c59912b32f71f06eaddf8d9be7c9eb5faca021e71a3dd1f5cb0.jpg)  
Figure 1: Two steps of NashDreamer world-model training on a real trajectory. The centralized RSSM is unrolled on the joint action and joint observation, the infoset model separately for each player. All stochastic states are posterior. The KL losses between posterior and prior are omitted; $a _ { d }$ denotes the initial dummy action.

## 4.4 Model Convergence

To understand how NashDreamer captures environment dynamics, we analyzed the global minimum of the world model objective (all formal definitions and proofs are deferred to Appendix B). We define an infoset-isomorphic model $( \mathcal { M } ^ { \ast } )$ as one that establishes a one-to-one mapping between latent states and true histories, and clusters latent states into latent infosets if and only if corresponding true histories belong to the same true infoset.

Our first result is that this structure-preserving model is strictly better, in expected loss, than models with two structural deviations: a “branch duplication”, which maps one history to several hallucinated latent branches with a distribution over them, and a model that violates the information partitioning of the game<sup>5</sup>. The same result fixes the value of the optimum, $\mathbb { E } _ { \mu } [ \mathcal { L } ^ { \mathrm { m o d e l } } ( \mathcal { M } ^ { * } ) ] =$ $\left( \beta ^ { \mathrm { e n c } } + \beta ^ { \mathrm { d y n } } \right) \mathbb { E } _ { h , a \sim \mu } [ H ( T ( h , a ) ) ]$ ], so the optimal loss of the “perfect” model scales linearly with environment stochasticity (Theorem 2). Because $\mathcal { M } ^ { * }$ captures the POSG structure, which unrolls into an EFG [20], and because branch duplication is indistinguishable from it from the actor-critic’s view, RNaD’s convergence result composes with it: an infoset-isomorphic model held fixed, together with exactly realized RNaD dynamics, yields convergence to a Nash equilibrium of the underlying game (Corollary 2). That guarantee is idealized in a way worth stating explicitly, since the practical algorithm approximates every one of its assumptions: neural function approximation, sampled trajectories, off-policy corrections, and a learned model updated concurrently with the policy.

![](images/e3b88c25aeb8de71f6a7d10c183b4115806b01c8651d6571761f7769bfcde129.jpg)  
Figure 2: Two imagined steps. Imagination is grounded in a posterior model state taken from world-model training and proceeds with prior stochastic states. The observation decoder is required during imagination, since the infoset model consumes observations.

However, our second result exposes a profound vulnerability of two-way KL-balancing objective in a stochastic environment :

Theorem 1 (Vulnerability to Posterior Collapse) Let M<sup>collapse</sup> be a model, where $K L ( e n c ( \hat { h } _ { t } , o _ { t } ) \parallel d y n ( \hat { h } _ { t } ) ) = 0 .$ for all reachable $\hat { h } _ { t } ,$ , and $o _ { t }$ where $p ( o _ { t } | \hat { h } _ { t } ) > 0$ . There exists an environment, sampling policy $\mu ,$ , and loss coefficients, such that $\mathbb { E } _ { \mu } [ \dot { \mathcal { L } } ( \dot { \mathcal { M } } ^ { c o l l a p s e } ) ] < \mathbb { E } _ { \mu } [ \mathcal { L } ( \mathcal { M } ^ { * } ) ]$

Specifically, this can happen in environments with high stochasticity and small magnitude changes for the predictor, turning disambiguation of the chance outcomes into a suboptimal solution. We note that this theorem also holds for a single-agent environment and the standard DreamerV3 architecture.

The dependence of the isomorphism loss bound on the (typically) non-stationary $\mu$ renders static coefficient tuning unsound. However, this degenerate minimum can be mitigated by using a “one-way” KL loss (applying a stop-gradient to the posterior), which drives convergence to either $\mathcal { M } ^ { * }$ or $\mathcal { M } ^ { b }$ if our sampling policy is sufficiently exploratory (Appendix B).

Unfortunately, removing this posterior regularization allows for representation drift (Section 2.1). Consequently, Dreamer-style MBRL faces a trade-off: the regularizer serving to mitigate representation non-stationarity and unpredictability can drive the optimization towards a degenerate solution.

Finally, this artifact does not conflict with DreamerV3’s empirical success on stochastic domains such as Minecraft, Atari and BSuite. What makes collapse harmful is not stochasticity as such, but chance that is discrete, many-outcome, and strategically decisive. Minecraft’s procedural generation is highly stochastic, yet the next frames are largely predictable from those already seen, the initial-only randomness is closer to epistemic uncertainty than aleatoric, and pixel-level error is strategically benign where collapse does occur. The randomness injected into Atari and BSuite is genuinely aleatoric but either few-outcome (sticky actions) or noise-like. Chance in games is the opposite on every count, which is why the same artifact becomes damaging, as we demonstrate in Leduc.

## 5 Empirical Evaluation

To avoid imagination in a highly inaccurate model, a world model warm-up precedes imagination: 1000 gradient steps for Imperfect-Information Goofspiel-5 (GS5) and Leduc Poker (LP), and 2000 for the larger environments: Stochastic Imperfect Information Goofspiel-13 (GS13), Battleship 5 × 5 (BS5) and Phantom Tic-Tac-Toe (PTTT). Appendix R ablates this choice. Goofspiel and Battleship are natively simultaneous-move; Leduc Poker and Phantom Tic-Tac-Toe are turn-based, embedded by the dummy-action construction of Section 2. The world model has no turn-based mode. We refer the reader to Appendix F for detailed environment rules, G.3 for extended hyperparameters, and M for comparison with DreamerV3 in a single-agent domain. All the experiments were carried out with 10 distinct training seeds.

Evaluation protocol. All reported NashConv values are computed exactly against the true game by best-response dynamic programming over the full game tree, not estimated by sampling. For Phantom Tic-Tac-Toe, we utilize the exp-a-spiel library [33]. Appendix H details the protocol, including the handling of chance nodes and simultaneous moves.

Baselines and positioning. We compare against model-free RNaD (with and without replay) and against MMD [36], the strongest-performing model-free policy-gradient method in the evaluation of Rudolph et al. [33]. MMD serves both as a baseline and, run inside imagination as Nash-Dreamer[MMD], as the demonstration that the policy optimizer is interchangeable (Section 4.3). Its hyperparameters are the generic setting of Rudolph et al. [33] and are held fixed across all domains, detailed in Appendix G.5. We do not compare directly against published multi-agent Dreamer systems, because each is structurally inapplicable to 2p0s IIGs : cooperative communication-based methods lose their central mechanism in a zero-sum game, opponent-embedding models define no fixed joint policy. Appendix D argues each case in detail. However, we do perform the comparison against a decentralized model and show its unidentifiability in Appendix L.

## 5.1 Sample Efficiency: NashDreamer vs. Model-Free Baselines

We compare against the identical model-free versions of our policy optimization algorithms and compare in terms of real environment steps seen. To compensate for the additional steps seen in imagination, we also evaluate against RNaD augmented with a replay buffer.

Our environments directly provide infoset representation, used as an input for our model-free baseline. To ensure exact input parity for NashDreamer, we use these in place of observations. We validate the capabilities of NashDreamer to learn directly from partial observations via an ablation in Imperfect Information Goofspiel 5 (Appendix O)

Imperfect Information Goofspiel 5: While both methods achieved comparable asymptotic performance, NashDreamer reaches it utilizing substantially fewer environment interactions (Figure 3). The off-policy replay failed to improve RNaD’s sample efficiency. As theorized in Appendix E.2, we attribute this failure to the variance injected by unbiasing the off-policy data.

Leduc Poker: While NashDreamer initially outpaced the model-free baseline, it exhibited late-stage performance deterioration (Figure 3). Investigating the learned dynamics reveals a structural error present throughout training: the model hallucinated physically impossible events at the deeper public chance node, such as dealing a public card already present in a player’s hand. We also tested the one-way KL loss of Theorem 3, which sets β<sup>enc</sup> = 0, removing the posterior regularization and signal towards collapse, it fails to prevent the structural error and deteriorates furthest of all variants we ran, because of representation drift (Appendix J), confirming out tradeoff claim from Section 4.4.Appendix K reports quantitative world-model diagnostics evidencing this failure.

Phantom Tic-Tac-Toe. Phantom TTT serves as a benchmark of a large. yet tractable game. In this game, both optimizers improve when wrapped in NashDreamer, with NashDreamer[RNaD] attains model-free RNaD’s final exploitability using roughly a quarter of the gradient steps (Figure 3). Replay improves substantially on plain RNaD here, as it also does in Leduc, but it stays above NashDreamer[RNaD] throughout training.

![](images/068d414d1373cdd4184c2935badc01c243aa1328b246c9fe409ea181be3ae378.jpg)  
Figure 3: NashConv against real environment interaction on the three domains where it is exactly computable. In Goofspiel-5 NashDreamer[RNaD] reaches a given NashConv with fewer environment steps than either RNaD variant. In Leduc it converges faster initially and then deteriorates, the world model retaining a structural error at the public chance node (Appendix I). In Phantom Tic-Tac-Toe both optimizers improve inside imagination, and NashDreamer[RNaD] continues to improve while the two MMD variants settle together. Bold curves are means over the 10 training seeds, faint curves the individual seeds; the dotted line marks the end of the world-model warm-up.

MMD as a model-free baseline. On small games, model-free MMD converges faster than RNaD. Wrapping MMD in NashDreamer changes little, as MMD already converges within the world-model warm-up phase. This phenomenon is observed in Goofspiel 5 and Leduc, however in Phantom Tic-Tac-Toe the model already brings visible acceleration. Appendix R tests this explanation directly by shortening the warm-up in Leduc, which recovers the acceleration even in Leduc.

## 5.2 Approximate NashConv on Larger Domains

Exact NashConv is intractable in stochastic Goofspiel-13 and Battleship, so we approximate it by budgeted best-response search. We freeze a trained checkpoint’s policy and, from each seat in turn, train 10 PPO seeds against it for 3,000 gradient steps, evaluating each over 100,000 games. The largest best-response value obtained by any PPO seed is taken as that seat’s estimate, and the two seats are summed. Appendix H gives the full protocol.

We have trained each of the algorithms for 30k gradient steps, giving a world-model warm-up of 2k gradient steps for the NashDreamer variants and we evaluate the approximate NashConv each 10k steps.

Since a fixed search budget need not recover the true best response it is a budgeted lower bound on NashConv: values are comparable across methods evaluated at the same budget, but not against the exact NashConv of Section 5.1 and not against results obtained at any other budget. Intervals are ±2σ over the 10 training seeds, not over PPO seeds.
<table><tr><td></td><td colspan="3">Goofspiel (13 cards)</td><td colspan="3">Battleship  $( 5 \times 5 )$ </td></tr><tr><td>Method</td><td>10k</td><td>20k</td><td>30k</td><td>10k</td><td>20k</td><td>30k</td></tr><tr><td>MMD</td><td> $0 . 8 2 7 \pm 0 . 0 7 6$ </td><td> $0 . 7 4 1 \pm 0 . 0 9 3$ </td><td> $0 . 7 3 2 \pm 0 . 0 7 5$ </td><td> $1 . 3 0 6 \pm 0 . 0 7 0$ </td><td> $1 . 1 6 1 \pm 0 . 0 9 5$ </td><td> $1 . 0 2 2 \pm 0 . 1 5 0$ </td></tr><tr><td>ND[MMD]</td><td> $0 . 8 5 3 \pm 0 . 0 7 4$ </td><td> $0 . 7 9 1 \pm 0 . 0 8 5$ </td><td> $0 . 7 6 3 \pm 0 . 0 7 5$ </td><td> $1 . 2 0 9 \pm 0 . 1 2 6$ </td><td> $0 . 9 6 9 \pm 0 . 0 9 7$ </td><td> $\mathbf { 0 . 8 1 7 \pm 0 . 1 6 1 }$ </td></tr><tr><td>RNaD</td><td> $0 . 7 4 2 \pm 0 . 3 8 8$ </td><td> $0 . 4 4 8 \pm 0 . 2 4 7$ </td><td> $0 . 3 5 6 \pm 0 . 2 9 1$ </td><td> $1 . 4 1 6 \pm 0 . 0 6 6$ </td><td> $1 . 3 2 6 \pm 0 . 1 3 2$ </td><td> $1 . 1 5 5 \pm 0 . 1 8 3$ </td></tr><tr><td>RNaD w/ replay</td><td> $0 . 7 5 6 \pm 0 . 0 7 4$ </td><td> $0 . 5 3 1 \pm 0 . 0 6 1$ </td><td> $0 . 4 9 9 \pm 0 . 0 7 4$ </td><td> $1 . 4 3 8 \pm 0 . 0 8 6$ </td><td> $1 . 3 9 5 \pm 0 . 1 2 2$ </td><td> $1 . 2 2 3 \pm 0 . 1 2 0$ </td></tr><tr><td>ND[RNaD]</td><td> $0 . 4 6 9 \pm 0 . 1 3 8$ </td><td> $0 . 2 5 0 \pm 0 . 1 4 5$ </td><td> $\mathbf { 0 . 1 6 8 \pm 0 . 1 3 9 }$ </td><td> $1 . 1 9 9 \pm 0 . 1 3 1$ </td><td> $1 . 1 8 4 \pm 0 . 1 6 8$ </td><td> $1 . 1 6 9 \pm 0 . 2 5 6$ </td></tr></table>

Table 1: Budgeted approximate NashConv (lower is better) at three checkpoints, as mean ±2σ over the 10 training seeds. Values are a lower bound on NashConv at a fixed best-response budget and are not comparable with exact NashConv or with other budgets. ND denotes NashDreamer with the given actor-critic inside imagination; best value per domain in bold.

In Goofspiel-13 NashDreamer[RNaD] reaches lower approximate NashConv after 20k gradient steps than model-free RNaD reaches only at 30k, and it leads at every checkpoint. In Battleship the same holds for the other optimizer, NashDreamer[MMD] at 20k surpassing MMD at 30k. Which optimizer is the stronger one is domain-dependent, but our model nevertheless accelerates it further (Table 1).

## 6 Related Work

Model-based RL. MuZero [34] and Stochastic MuZero [2] learn latent models for planning in perfect-information games. The Dreamer series [12–15] trains world models for imagination-based policy learning in single-agent settings. NashDreamer specifically builds upon the online architecture of DreamerV3, as adapting the offline learning paradigm of DreamerV4 to adversarial multi-agent settings introduces overfitting to a particular opponent set, as well as prohibitive requirements of dataset size. LAMIR Kubicek and Lisý [21] extends MuZero-style look-ahead reasoning to IIGs, but focuses on search-time planning rather than pure RL and assumes deterministic games.

Multi-agent Dreamer. MAMBA [9] and DreamerCF [6] extend Dreamer to cooperative settings with inter-agent communication. In competitive ones, Lu and Chen [25] condition the world model on ego and opponent embeddings, the latter randomly initialized at test time and so requiring zero-shot generalization, and Orsula [29] use single-player modeling with population-based self-play. All use policy-gradient actor-critics without game-theoretic convergence guarantees.

Game-theoretic RL. CFR [42] and its variants are the gold standard for solving IIGs but require explicit game tree access. DREAM [39] removes this requirement via Monte Carlo sampling, enabling model-free deep CFR. RNaD/DeepNash [32] achieves human-level Stratego play using model-free RL with regularized reward dynamics. PSRO [22] maintains a population of best-response policies. MMD [36] achieves superhuman performance in Stratego [37]. All mentioned model-free approaches are potential alternatives to RNaD as the actor-critic component within NashDreamer’s learned world model, and we substantiate this for MMD in Section 5.1 and Appendix Q.

## 7 Conclusion

We introduced NashDreamer, a principled MBRL framework for two-player zero-sum imperfect information games. By pairing a centralized latent world model (MARSSM) with a decentralized, game-theoretic actor-critic (such as RNaD), NashDreamer overcomes the identifiability barriers inherent to multi-agent learning. Empirically, it significantly improves convergence in early-to-mid stages of training.

However, as formalized theoretically and validated in Leduc Poker, environments with high chance stochasticity remain challenging due to the tradeoff between stability and collapse. Resolving this tension without inducing degenerate minima is the most critical direction for future work. Additional limitations to overcome include relaxing the t = 0 unrolling constraint and validating the architecture in massive environments, closer to the desired real-world deployment.

## 8 Broader impacts

We envision the primary application of our framework to be a building block for automated cyber defense. We acknowledge the dual-use nature of this research, as the same methodology could be leveraged to train sophisticated adversarial agents. We maintain, however, that the most effective countermeasure is making the algorithm widely accessible, allowing the community to train robust defenders with symmetric capabilities, rather than relying on the inherently fragile paradigm of security through obscurity.

## Acknowledgments and Disclosure of Funding

We thank the anonymous reviewers, whose thorough and constructive feedback prompted several additional experiments and analyses that substantially improved both the empirical evaluation and the presentation of this work.

This research is supported by Czech Science Foundation (GA25-18353S). Computational resources were supplied by (e-INFRA CZ LM2018140) supported by the Ministry of Education, Youth and Sports of the Czech Republic.

## References

[1] Eyal Amir and Allen Chang. Learning partially observable deterministic action models. In Journal ofArtificial Intelligence Research, volume 33, pages 349–402, 2008.

[2] Ioannis Antonoglou, Julian Schrittwieser, Sherjil Ozair, Thomas K Hubert, and David Silver. Planning in stochastic environments with a learned model. In International Conference on Learning Representations, 2022.

[3] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016

[4] James Bradbury, Roy Frostig, Peter Hawkins, Matthew James Johnson, Chris Leary, Dougal Maclaurin, George Necula, Adam Paszke, Jake VanderPlas, Skye Wanderman-Milne, and Qiao Zhang. JAX: Composable transformations of Python+NumPy programs, 2018. URL http://github.com/google/jax.

[5] Noam Brown, Anton Bakhtin, Adam Lerer, and Qucheng Gong. Combining deep reinforcement learning and search for imperfect-information games. In Advances in Neural Information Processing Systems, volume 33, pages 17057–17069, 2020.

[6] Jiajun Chai, Yuqian Fu, Dongbin Zhao, and Yuanheng Zhu. Aligning credit for multi-agent cooperation via model-based counterfactual imagination. In Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems, pages 281–289, 2024.

[7] Ian Char, Joseph Abbate, László Bardóczi, Mark Boyer, Youngseog Chung, Rory Conlin, Keith Erickson, Viraj Mehta, Nathan Richner, Egemen Kolemen, et al. Offline model-based reinforcement learning for tokamak control. In Learningfor Dynamics and Control Conference, pages 1357–1372. PMLR, 2023.

[8] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using rnn encoderdecoder for statistical machine translation. In Proceedings ofthe 2014 Conference on Empirical Methods in Natural Language Processing, pages 1724–1734, 2014.

[9] Vladimir Egorov and Alexei Shpilman. Scalable multi-agent model-based reinforcement learning. In Proceedings of the 21st International Conference on Autonomous Agents and Multiagent Systems, pages 381–390, 2022.

[10] Lasse Espeholt, Hubert Soyer, Remi Munos, Karen Simonyan, Vlad Mnih, Tom Ward, Yotam Doron, Vlad Firoiu, Tim Harley, Iain Dunning, et al. Impala: Scalable distributed deep-rl with importance weighted actor-learner architectures. In International Conference on Machine Learning, pages 1407–1416. PMLR, 2018.

[11] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. In Advances in Neural Information Processing Systems, 2024.

[12] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In International Conference on Learning Representations, 2020.

[13] Danijar Hafner, Timothy P Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering atari with discrete world models. In International Conference on Learning Representations, 2021.

[14] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640(8059):647–653, 2025.

[15] Danijar Hafner, Wilson Yan, and Timothy Lillicrap. Training agents inside of scalable world models. In Advances in Neural Information Processing Systems, 2025.

[16] Jonathan Heek, Anselm Levskaya, Avital Oliver, Marvin Ritter, Bertrand Rondepierre, Andreas Steiner, and Marc van Zee. Flax: A neural network library and ecosystem for jax. 2020. URL http://github.com/google/flax.

[17] Daniel Hennes, Dustin Morrill, Shayegan Omidshafiei, Rémi Munos, Julien Perolat, Marc Lanctot, Audrunas Gruslys, Jean-Baptiste Lespiau, Paavo Parmas, Edgar Duéñez-Guzmán, and Karl Tuyls. Neural replicator dynamics: Multiagent learning via hedging policy gradients. In Proceedings of the 19th International Conference on Autonomous Agents and Multiagent Systems, 2020.

[18] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

[19] Jaromír Janisch, Tomáš Pevný, and Viliam Lisý. Nasimemu: Network attack simulator & emulator for training agents generalizing to novel scenarios, 2023. URL https://arxiv. org/abs/2305.17246.

[20] Vojtech Kova ˇ ˇrík, Martin Schmid, Neil Burch, Michael Bowling, and Viliam Lisý. Rethinking formal models of partially observable multiagent decision making. Artificial Intelligence, 303: 103645, 2022.

[21] Ondrej Kubicek and Viliam Lisý. Look-ahead reasoning with a learned model in imperfect information games. In International Conference on Learning Representations, volume 2026, pages 53533–53547, 2026.

[22] Marc Lanctot, Vinicius Zambaldi, Audrunas Gruslys, Angeliki Lazaridou, Karl Tuyls, Julien Pérolat, David Silver, and Thore Graepel. A unified game-theoretic approach to multiagent reinforcement learning. Advances in Neural Information Processing Systems, 30, 2017.

[23] Marc Lanctot, Edward Lockhart, Jean-Baptiste Lespiau, Vinicius Zambaldi, Satyaki Upadhyay, Julien Pérolat, Sriram Srinivasan, Finbarr Timbers, Karl Tuyls, Shayegan Omidshafiei, et al. Openspiel: A framework for reinforcement learning in games. arXiv preprint arXiv:1908.09453, 2019.

[24] Lennart Ljung. System Identification: Theory for the User. Prentice Hall, 2nd edition, 1999.

[25] Tianyu Lu and Bing Chen. Empowering aerial maneuver games through model-based constrained reinforcement learning. 2025.

[26] Panayotis Mertikopoulos, Christos Papadimitriou, and Georgios Piliouras. Cycles in adversarial regularized learning. In Proceedings ofthe Twenty-Ninth Annual ACM-SIAM Symposium on Discrete Algorithms, pages 2703–2717. SIAM, 2018.

[27] John F Nash Jr. Equilibrium points in n-person games. Proceedings of the National Academy of Sciences, 36(1):48–49, 1950.

[28] Frans A Oliehoek. Decentralized pomdps. In Reinforcement Learning: State-of-the-Art, pages 471–503. Springer, 2012.

[29] Andrej Orsula. Learning to play air hockey with model-based deep reinforcement learning. arXiv preprint arXiv:2406.00518, 2024.

[30] Georgios Papoudakis, Filippos Christianos, Arrasy Rahman, and Stefano V Albrecht. Dealing with non-stationarity in multi-agent deep reinforcement learning. arXiv e-prints, pages arXiv– 1906, 2019.

[31] Julien Perolat, Remi Munos, Jean-Baptiste Lespiau, Shayegan Omidshafiei, Mark Rowland, Pedro Ortega, Neil Burch, Thomas Anthony, David Balduzzi, Bart De Vylder, et al. From poincaré recurrence to convergence in imperfect information games: Finding equilibrium via regularization. In International Conference on Machine Learning, pages 8525–8535. PMLR, 2021.

[32] Julien Perolat, Bart De Vylder, Daniel Hennes, Eugene Tarassov, Florian Strub, Vincent de Boer, Paul Muller, Jerome T Connor, Neil Burch, Thomas Anthony, et al. Mastering the game of stratego with model-free multiagent reinforcement learning. Science, 378(6623):990–996, 2022.

[33] Max Rudolph, Nathan Lichtle, Sobhan Mohammadpour, Alexandre M Bayen, J Zico Kolter, Amy Zhang, Gabriele Farina, Eugene Vinitsky, and Samuel Sokota. Reevaluating policy gradient methods for imperfect-information games. In The Fourteenth International Conference on Learning Representations, 2026.

[34] Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839):604–609, 2020.

[35] Shai Shalev-Shwartz and Yoram Singer. A primal-dual perspective of online learning algorithms. Machine Learning, 69(2–3):115–142, 2007.

[36] Samuel Sokota, Ryan D’Orazio, J Zico Kolter, Nicolas Loizou, Marc Lanctot, Ioannis Mitliagkas, Noam Brown, and Christian Kroer. A unified approach to reinforcement learning, quantal response equilibria, and two-player zero-sum games. In The Eleventh International Conference on Learning Representations.

[37] Samuel Sokota, Eugene Vinitsky, Hengyuan Hu, J Zico Kolter, and Gabriele Farina. Superhuman ai for stratego using self-play reinforcement learning and test-time search. arXiv preprint arXiv:2511.07312, 2025.

[38] Finnegan Southey, Michael P Bowling, Bryce Larson, Carmelo Piccione, Neil Burch, Darse Billings, and Chris Rayner. Bayes’ bluff: Opponent modelling in poker. In Proceedings of the Twenty-First Conference on Uncertainty in Artificial Intelligence, pages 550–558. AUAI Press, 2005.

[39] Eric Steinberger, Adam Lerer, and Noam Brown. DREAM: Deep regret minimization with advantage baselines and model-free learning. In International Conference on Learning Representations, 2020.

[40] Frederick B Thompson. Equivalence of games in extensive form. In Harold W Kuhn, editor, Classics in Game Theory, pages 36–45. Princeton University Press, 1997. Originally RAND Research Memorandum RM-759, 1952.

[41] Philipp Wu, Alejandro Escontrela, Danijar Hafner, Pieter Abbeel, and Ken Goldberg. Daydreamer: World models for physical robot learning. In Conference on Robot Learning, pages 2226–2240. PMLR, 2023.

[42] Martin Zinkevich, Michael Johanson, Michael Bowling, and Carmelo Piccione. Regret minimization in games with incomplete information. In Advances in Neural Information Processing Systems, 2007.

[43] Liu Ziyin, Zhikang T Wang, and Masahito Ueda. Laprop: Separating momentum and adaptivity in adam. In International Conference on Machine Learning, 2020.

## A Regularized Policy Optimization in Imperfect-Information Games

This appendix expands Section 2.2 with the full statement of the two actor-critic algorithms that Nash-Dreamer runs inside imagination. Their implementations and hyperparameters are given separately in Appendices G.4 and G.5.

## A.1 Why Policy Gradients Cycle

Follow the Regularized Leader (FoReL) [35] is a general framework from online learning. When the regularizer is entropy, FoReL yields Replicator Dynamics, which is closely connected to policygradient methods via Neural Replicator Dynamics (NeuRD) [17]. However, FoReL has been proven to cycle and fail to converge to a Nash Equilibrium in both matrix games [26] and sequential IIGs [31].

## A.2 Regularized Nash Dynamics

Regularized Nash Dynamics (RNaD) [32] overcomes this by introducing policy-dependent reward regularization:

$$
\begin{array} { r } { R ^ { \mathrm { r e g } } ( h , a , \pi , \pi ^ { \mathrm { r e g } } ) = R ( h , a ) - \eta \log { \frac { \pi _ { 1 } ( a _ { 1 } \vert I _ { 1 } ( h ) ) } { \pi _ { 1 } ^ { \mathrm { r e g } } ( a _ { 1 } \vert I _ { 1 } ( h ) ) } } + \eta \log { \frac { \pi _ { 2 } ( a _ { 2 } \vert I _ { 2 } ( h ) ) } { \pi _ { 2 } ^ { \mathrm { r e g } } ( a _ { 2 } \vert I _ { 2 } ( h ) ) } } , } \end{array}\tag{5}
$$

where $\eta > 0$ is the regularization coefficient. This formulation naturally preserves the zero-sum property of the game. By periodically updating the regularization policy $\dot { \pi } ^ { \mathrm { r e g } }  \pi$ , RNaD provably converges to an NE. The practical deep reinforcement learning instantiation, DeepNash [32], combines RNaD with a V-trace estimator [10] for the actor-critic. We adapt the simultaneous-move variant of RNaD [21], which introduces counterfactual importance weighting to enable off-policy learning.

## A.3 Magnetic Mirror Descent

Magnetic Mirror Descent (MMD) [36] is the second optimizer we employ. It is a mirror-descent method whose update carries two regularizers, one anchoring the policy to a magnet policy $\pi ^ { \mathrm { m a g } }$ and one to the previous iterate:

$$
\begin{array} { r } { \pi _ { t + 1 } = \underset { \pi } { \arg \operatorname* { m i n } } \left\{ - \langle Q ^ { \pi _ { t } } , \pi \rangle + \alpha \mathrm { K L } ( \pi \parallel \pi ^ { \mathrm { m a g } } ) + \frac { 1 } { \eta } \mathrm { K L } ( \pi \parallel \pi _ { t } ) \right\} . } \end{array}\tag{6}
$$

Both methods thus regularize toward a reference policy, but through different mechanisms: RNaD transforms the reward $( \mathrm { E q . } 5 )$ , which preserves the zero-sum property of the game, whereas MMD constrains the policy update itself. We use the instantiation of Rudolph et al. [33], in which the magnet is heldfixed and uniform, so that α $\mathrm { K L } ( \pi \parallel \pi ^ { \mathrm { m a g } } )$ reduces to an entropy bonus and the proximal term is realized by PPO-style clipped updates. One consequence matters for the comparisons that follow: with a fixed magnet and $\alpha > 0$ , MMD converges to a quantal response equilibrium rather than to an NE, so its asymptotic exploitability is bounded away from zero. Unlike RNaD, there are, to the best of our knowledge no convergence results for MMD when switching the magnet policy or annealing the magnet regularization.

## B Proofs and Theoretical Analysis

## B.1 Definitions and Formal Setup

We first formalize the concepts necessary to state and prove our theoretical results. We make use of the fact that a POSG can be unrolled into an EFG structure [20]. Because games of this class form a tree, every non-root history has a unique predecessor.

As in Section 2, we lift the state-indexed functions of the game (transition, observation, legal actions) to histories through the last state $s ( h )$ of the history, writing e.g. $T ( h , a ) : = T ( s ( h ) , a )$ ). Accordingly, $H ( T ( h , a ) )$ below denotes the entropy of the chance outcome at $h .$

Definition 2 (Predecessor Function) Let h be a non-root true history in the game tree. We define the predecessorfunction $P r e \nu ( h ) = ( h _ { p r e \nu } , a _ { p r e \nu } , r _ { p r e \nu } )$ , which returns the unique immediately preceding history $h _ { p r e \nu } ,$ , the joint action $a _ { p r e \nu }$ taken at $h _ { p r e \nu }$ that led to $h ,$ and the immediate environment reward $r _ { p r e \nu }$ generated by this transition.

Definition 3 (Latent Trajectory) A latent trajectory is a sequence

$$
\hat { \tau } = ( m _ { 0 } , a _ { 0 } , \hat { o } _ { 0 } , \hat { I } _ { 0 } , m _ { 1 } , a _ { 1 } , \hat { o } _ { 1 } , \hat { I } _ { 1 } , \dots , m _ { t } , \hat { o } _ { t } )
$$

produced by unrolling the model, where the latent model state $m _ { l } = ( \hat { h } _ { l } , z _ { l } )$ encapsulates both the deterministic recurrent state $\hat { h } _ { l }$ and the stochastic categorical state z<sub>l</sub>, $a _ { l } = \left( a _ { 1 , l } , a _ { 2 , l } \right)$ is the joint action, $\hat { o } _ { l }$ is the joint observation reconstruction of $m _ { l } ,$ and $\hat { I } _ { l } = ( \hat { I } _ { 1 , l } , \hat { I } _ { 2 , l } )$ are the latent infoset embeddings. Latent trajectories are the objects over which tree-structure isomorphism is stated (Definition 7).

Definition 4 (Latent Reach Probability) Let τˆ be a latent trajectory as in Definition 3. We define the reach probability of a specific latent model state $m _ { t }$ under a joint sampling policy $\mu$ as:

$$
\hat { \nu } _ { \mu } ( m _ { t } ) = p ^ { \theta } ( z _ { t } \mid \hat { h } _ { t } ) \prod _ { l = 0 } ^ { t - 1 } p ^ { \theta } ( z _ { l } \mid \hat { h } _ { l } ) \prod _ { i = 1 } ^ { 2 } \mu _ { i } ( a _ { i , l } \mid \hat { I } _ { i , l } )
$$

Similarly, the reach probability of the deterministic state $\hat { h } _ { t } ,$ , which strictly isolates the probability prior to thefinal chance node resolution, is:

$$
\hat { \nu } _ { \mu } ( \hat { h } _ { t } ) = \prod _ { l = 0 } ^ { t - 1 } p ^ { \theta } ( z _ { l } \mid \hat { h } _ { l } ) \prod _ { i = 1 } ^ { 2 } \mu _ { i } ( a _ { i , l } \mid \hat { I } _ { i , l } )
$$

Definition 5 (Valid State and History Sets) Let M be the set of all possible latent model states generated by the architecture. We define $\mathbb { M } ^ { * } \subseteq \mathbb { M }$ as the subset oflatent states that are reachable $( \hat { \nu } _ { \mu } ( m ) > 0 )$ under a sufficiently exploratory sampling policy $\mu$ with full support over all legal actions.

Furthermore, let H be the set ofall valid histories in the true game tree. We define $\mathbb { H } _ { n c } \subseteq$ H as the subset oftrue histories where the last node is not a chance node.

We restrict our mapping to $\mathbb { H } _ { n c }$ because our architecture inherently models environmental stochasticity as latent transitions rather than explicit nodes.

Definition 6 (ϵ-Distance Matching) Let $\hat { r } ( m )$ and $\hat { d } ( m )$ denote the predicted reward and terminationflagfrom the latent state m. Let $\hat { l } _ { i } ( m )$ and $\hat { o } _ { i } ( m )$ be the predicted legal actions and reconstructed observationfor player i. A reachable latent model state m $\bar { \bf \Phi } \in \mathbb { M } ^ { * }$ is considered within ϵ-distance of a true history $h \in \mathbb { H } _ { n c }$ (where $P r e \nu ( h ) = ( h _ { p r e \nu } , a _ { p r e \nu } , r _ { p r e \nu } ) )$ if its predictions match the deterministic true environment emissions within an arbitrarily small error bound $\epsilon ,$ where $\| \cdot \|$ denotes the max-norm over vector components:

1. $\| \hat { r } ( m ) - r _ { p r e \nu } \| < \epsilon$

2. $\| \hat { d } ( m ) - d ( h ) \| < \epsilon$

3. $F o r i \in \{ 1 , 2 \} : \| \hat { l } _ { i } ( m ) - L _ { i } ( h ) \| < \epsilon$

4. For $i \in \{ 1 , 2 \} \colon \| \hat { o } _ { i } ( m ) - O _ { i } ( h ) \| < \epsilon$

where $d ( h ) = 1$ if the history is terminal and 0 otherwise.

To make condition 4 concrete, consider Leduc Poker, where each $O _ { i } ( h )$ is a binary vector whose blocks encode the player’s private card, the public card (if already dealt), and the current bets. Take a history h in which player 1 holds the jack and no public card has been dealt, so the private-card block of $O _ { 1 } ( h )$ is one-hot on the jack and the public-card block is all zeros. A latent state $m$ whose decoder outputs $\hat { o } _ { 1 } ( m )$ with 0.99 on the jack, at most 0.01 on every other private card and at most 0.01 on every public-card entry is within ϵ-distance for $\epsilon = 0 . 0 2 \mathrm { : }$ : the reconstruction is unambiguous both about which private card was dealt and about no public card being present. A latent state that instead splits mass between two private cards, say 0.5 and $0 . 5 ,$ violates condition 4 for every $\epsilon < 0 . 5$ Such a state has aliased two distinct true histories into one latent state, which is exactly the structural failure excluded from $\mathcal { M } ^ { * }$ , and which we observe empirically at the round-2 public chance node in Leduc (Appendix J).

Definition 7 (Tree-Structure Isomorphism) A generative world model is tree-structure isomorphic if there exists a bijective mapping $f _ { \epsilon } : \bar { \mathbb { H } } _ { n c } \to \mathbb { M } ^ { * }$ , parameterized by the tolerance ϵ of Definition 6 and evaluated at a history h, that satisfies thefollowing conditionsfor all $h \in \mathbb { H } _ { n c } .$

1. $f _ { \epsilon } ( h )$ is within ϵ-distance of h.

2. $f _ { \epsilon } ( h )$ is the ending state ofa latent trajectory τˆ (Definition 3) that contains the exact same joint action sequence as the true history h, and $\hat { o } _ { l }$ is within ϵ-distance of o<sub>l</sub> for each timestep l.

Definition 8 (Infoset Isomorphism) Let $\hat { I } _ { i } ( m )$ denote the latent infoset embedding of player i for latent state m. Assume a world model is tree-structure isomorphic with bijective mapping $\overline { { f _ { \epsilon } } } : \mathbb { H } _ { n c } \to \mathbb { M } ^ { * }$ . The model is infoset-isomorphic if,for any player $i \in \{ 1 , 2 \}$ and any pair ofvalid true histories $h , h ^ { \prime } \in \mathbb { H } _ { n c } ,$ their corresponding mapped model states $m = f _ { \epsilon } ( h )$ and $m ^ { \prime } = f _ { \epsilon } ( h ^ { \prime } )$ satisfy thefollowing equivalence relation:

$$
\| \hat { I } _ { i } ( m ) - \hat { I } _ { i } ( m ^ { \prime } ) \| < \epsilon \iff I _ { i } ( h ) = I _ { i } ( h ^ { \prime } )
$$

where $I _ { i } ( h )$ denotes the true infoset of player i at history h.

A world model exhibits branch duplication if the encoder introduces artificial stochasticity, mapping a single true history to multiple distinct latent model states. Semi-formally, instead of a bijection $f _ { \epsilon } ,$ the model induces a mapping $f _ { \epsilon } ^ { \prime } : \mathbb { H } _ { n c } \to \mathcal { P } ( \mathbb { M } ^ { * } )$ where a true history $h \in \mathbb { H } _ { n c }$ maps to a set of reachable model states $\mathbb { M } _ { h } \subseteq \mathbb { M } ^ { * }$ . We assume these sets are disjoint for distinct histories and cover the entire reachable model space $( \mathbb { M } _ { h } \cap \mathbb { M } _ { h ^ { \prime } } = \emptyset { \mathrm { ~ f o r ~ } } h \neq h ^ { \prime } )$ , all states within $\mathbb { M } _ { h }$ are within ϵ-distance for all predictor targets. Crucially, all states within $\mathbb { M } _ { h }$ yield the exact same joint latent infoset representation $( \hat { I } _ { 1 } , \hat { I } _ { 2 } )$ (this condition will be satisfied directly, since all the model states were created by the same AOH and all satisfy the ϵ-distance accuracy).

Definition 9 (Information Partition Violation) A world model violates the game’s information partitioning ifit is tree-structure isomorphic but not infoset-isomorphic

This means the model correctly captures the objective physical dynamics, but its mapped latent infosets either alias distinct true infosets together, or improperly separate histories belonging to the same true infoset.

## B.2 Proofs

We will now analyze the loss for isomorphic model, leading up to a proof for Theorem 2. We assume that the KL losses are unclipped.

Lemma 1 Let $\hat { h }$ be a deterministic recurrent state induced by the true history and joint action $\left( h _ { p r e \nu } , a _ { p r e \nu } \right)$ . For any model thatperfectly reconstructs deterministic environment emissions (including both infoset-isomorphic models and models exhibiting branch duplication), the expected dynamics loss over the chance outcome is exactly equal to the scaled inherent Shannon entropy of the true environment transition: $H ( T ( h _ { p r e \nu } , a _ { p r e \nu } ) )$ .

Proof 1 The dynamics loss is defined as the expected KL divergence between the posterior encoder $\boldsymbol q ^ { \theta } ( \boldsymbol z \mid o , \hat { \boldsymbol h } )$ and the prior dynamics predictor $p ^ { \theta } ( z \mid \hat { h } )$ across all possible observations o sampled from the true environment distribution $p ( o \mid \hat { h } )$ .

Assuming an optimal prior that perfectly matches the aggregated marginal posterior, we have $p ^ { \theta } ( z \mid \hat { h } ) = \mathbb { E } _ { o \sim p } [ q ^ { \theta } ( z \mid o , \hat { h } ) ]$ ]. We define the induced true marginal distribution of the latent state as $p ( z \mid \hat { h } ) = p ^ { \theta } ( z \mid \hat { h } )$ , and the joint distribution over observations and latent chance states as $p ( o , z \mid \hat { h } ) = p ( o \mid \hat { h } ) q ^ { \theta } ( z \mid o , \hat { h } )$ ).

Using this joint distribution, we expand the expected KL divergence:

$$
\begin{array} { l } { { { \mathbb { E } } _ { o \sim p } \left[ K L \left( q ^ { \theta } ( z \mid o , \hat { h } ) \parallel p ^ { \theta } ( z \mid \hat { h } ) \right) \right] = \displaystyle \sum _ { o } p ( o \mid \hat { h } ) \sum _ { z } q ^ { \theta } ( z \mid o , \hat { h } ) \log \frac { q ^ { \theta } ( z \mid o , \hat { h } ) } { p ^ { \theta } ( z \mid \hat { h } ) } \ ~ } } \\ { { = \displaystyle \sum _ { o , z } p ( o , z \mid \hat { h } ) \log \frac { p ( o , z \mid \hat { h } ) / p ( o \mid \hat { h } ) } { p ( z \mid \hat { h } ) } \ ~ } } \\ { { = \displaystyle \sum _ { o , z } p ( o , z \mid \hat { h } ) \log \frac { p ( o , z \mid \hat { h } ) } { p ( o \mid \hat { h } ) p ( z \mid \hat { h } ) } \ ~ } } \end{array}
$$

Because the final expression is the expected log-ratio of the joint distribution to the product of its marginals, it is exactly theformal definition ofConditional Mutual Information (CMI), $I ( z , o \mid \hat { h } )$ Applying standard information-theoretic identities, we rewrite this CMI as the difference between conditional entropies:

$$
I ( z , o \mid \hat { h } ) = H ( o \mid \hat { h } ) - H ( o \mid z , \hat { h } )
$$

Because both an isomorphic model and a branch-duplicated model map true histories to states within an ϵ-distance boundfor predictor targets, the combination ofthe deterministic history representation h<sup>ˆ</sup> and the chosen latent chance state z must perfectly reconstruct the true environment observation o. Since o is fully and deterministically defined by the tuple $( \hat { h } , z )$ , there is zero remaining uncertainty about the observation. Therefore, the conditional entropy $H ( o \mid z , { \hat { h } } ) = 0 .$

Consequently, the expected dynamics loss simplifies exactly to $H ( o \mid \hat { h } )$ . Because the recurrent state h<sup>ˆ</sup> uniquely identifies the predecessor history and joint action $\left( h _ { p r e \nu } , a _ { p r e \nu } \right)$ , the entropy of the subsequent observation o is exactly the inherent aleatoric entropy of the true environment chance node:

$$
\mathbb { E } _ { o \sim p } \left[ K L \left( q ^ { \theta } ( \boldsymbol { z } \mid o , \hat { h } ) \mid \mid p ^ { \theta } ( \boldsymbol { z } \mid \hat { h } ) \right) \right] = H ( T ( h _ { p r e \nu } , a _ { p r e \nu } ) )
$$

Because this equality holds as long as $H ( o \mid z , { \hat { h } } ) = 0$ (which is strictly required to maintain the ϵ-distance boundsfor valid reconstruction), it is invariant to how the encoder maps the observation into the latent space. Thus, branch duplication cannot decrease the expected KL penalty below the true chance entropy.

With the dynamics loss lower-bounded by the inherent environment stochasticity, we now establish the convergence properties of the remaining objective terms under the assumption of unbounded network capacity.

Corollary 1 (Deterministic Posterior) By the definition oftree-structure isomorphism, the bijective mapping $f _ { \epsilon } ( h )$ requires the model state to be induced by the exact same AOH as the true history. Because the recurrent state h<sup>ˆ</sup> contains all historical context up to the current observation, the stochastic state z must necessarily encode the chance-induced observation. To satisfy $H ( o \mid z , { \hat { h } } ) = 0$ (as established in Lemma 1), the posterior encoder $\boldsymbol q ^ { \theta } ( \boldsymbol z \mid o , \hat { \boldsymbol h } )$ must collapse into a deterministic (Dirac delta/one-hot) distribution, mapping each distinct chance outcome of $T ( h _ { p r e \nu } , a _ { p r e \nu } )$ to a unique z.

Lemma 2 (Predictor and Unifier Loss Minimization) Let M<sup>∗</sup> be an infoset-isomorphic model. $A s \epsilon \to 0$ in the ϵ-distance bound, the expected lossfor all deterministic predictors (reward, continuation, legal actions, observation reconstruction) and all latent infoset targets decreases to zero.

Proof 2 By the definition of ϵ-distance matching, all predictions generated by a mapped state m $\in \mathbb { M } ^ { * }$ match the true environment emissions within an arbitrary bound ϵ. As $\epsilon  0 ,$ , the Mean Squared Error (MSE) and Cross-Entropy losses for these predictive targets trivially evaluate to zero.

Furthermore, the decentralized infoset networks process the true action-observation history (AOH) to generate $\hat { I } _ { 1 }$ and ${ \hat { I } } _ { 2 } .$ Given unbounded capacity, the auto-encoding targets (decoding the current observation and previous action from the infoset embedding) are perfectly reconstructed, achieving zero loss.

Crucially, under our assumptions of perfect recall and immediately observable chance outcomes, the intersection of the true infosets of all players uniquely identifies the true underlying history (collective observability, Section 2): $I _ { 1 } ( h ) \bar { \cap } I _ { 2 } ( h ) = \{ h \}$ . Because the model is infoset-isomorphic, its latent infosets perfectly preserve this partitioning $\begin{array} { r } { ( \| \hat { I } _ { i } ( m ) - \hat { I } _ { i } ( m ^ { \prime } ) \| < \epsilon \iff I _ { i } ( h ) = I _ { i } ( h ^ { \prime } ) ) } \end{array}$ . Therefore, the joint latent infoset embedding $( \hat { I } _ { 1 } , \hat { I } _ { 2 } )$ uniquely identifies the mapped latent state m. Because this mapping is uniquely resolvable, the centralized unifier network can perfectly reconstruct mfrom $( \hat { I } _ { 1 } , \hat { I } _ { 2 } )$ , allowing the unifier loss to minimize to $z e r o .$

Using these, we can prove the first result of Section 4.4.

Theorem 2 Let µ be an arbitrary joint sampling policy. Let $\mathcal { M } ^ { \ast }$ be an infoset-isomorphic world model, $\dot { \mathcal { M } } ^ { b }$ be a world model exhibiting branch duplication, and $\mathcal { M } ^ { l }$ be a model violating the game’s information partitioning. It holds that $\mathbb { E } _ { \mu } [ \dot { \mathcal { L } } ^ { m o d e l } ( \mathcal { M } ^ { * } ) ] < \mathbb { E } _ { \mu } [ \mathcal { L } ^ { m o d e l } ( \mathcal { M } ^ { b } ) ]$ and $\mathbb { E } _ { \mu } [ \mathcal { L } ^ { \mathrm { m o d e l } } ( \mathcal { M } ^ { * } ) ] < \mathbb { E } _ { \mu } [ \mathcal { L } ^ { \mathrm { m o d e l } } ( \mathcal { M } ^ { l } ) ]$ . Furthermore, under any $\mu , \mathcal { M } ^ { * }$ achieves an expected loss:

$$
\mathbb { E } _ { \mu } [ \mathcal { L } ^ { \mathrm { m o d e l } } ( \mathcal { M } ^ { * } ) ] = ( \beta ^ { e n c } + \beta ^ { d y n } ) \mathbb { E } _ { h , a \sim \mu } [ H ( T ( h , a ) ) ] ,
$$

where H denotes Shannon entropy.

Proof 3 (Proof of Theorem 2) We first establish the exact expected loss for the infoset-isomorphic model, M<sup>∗</sup>, under the sampling policy $\mu .$

By Lemma 2, because $\mathcal { M } ^ { \ast }$ is both tree-structure isomorphic and infoset-isomorphic, its ϵ-distance bounds approach zero, allowing all deterministic predictor losses (reward, continuation, legal actions, observation reconstruction) and all unifier/infoset losses to minimize to zero.

The only remaining non-zero terms in the objective are the KL divergence losses. As established in Lemma 1, the expected KL divergence at a given recurrent state h<sup>ˆ</sup> induced by the true predecessor $( h , a )$ evaluates exactly to the chance entropy $H ( T ( h , a ) )$ . Because the objective utilizes two separate KL penalties—onefor updating the dynamics prior (scaled by $\beta ^ { d y n } )$ and onefor updating the posterior encoder (scaled by $\beta ^ { e n \bar { c } } )$ —the total expected loss at this specific step is:

$$
\mathcal { L } ^ { K L } ( \hat { h } ) = ( \beta ^ { e n c } + \beta ^ { d y n } ) H ( T ( h , a ) )
$$

To compute the expected loss ofthe entire model under the joint sampling policy µ, we aggregate this per-step loss over all reachable latent recurrent states. We denote the set of all true chance nodes as $\mathbb { H } _ { c } = \mathbb { H } \setminus \mathbb { H } _ { n c }$ and the set of all reachable recurrent states as $\hat { \mathbb { H } } ^ { * }$ . Because the bijective mapping $f _ { \epsilon } : \mathbb { H } _ { n c } \to \mathbb { M } ^ { * }$ guarantees a one-to-one correspondence between true non-chance histories and latent states, ifwe reparametrize deterministic transitions to be represented via a trivial chance node with a single outcome, each latent recurrent state $\hat { h } _ { t }$ corresponds perfectly to a true chance node transitionfrom some $( h , a )$ , i.e. $f _ { \epsilon }$ induces also a bijective mapping $g : \hat { \mathbb { H } } ^ { * } \to \mathbb { H } _ { c }$

Furthermore, because the optimal prior exactly captures the true marginal chance distribution, the latent reach probability of any recurrent state, ${ \hat { \nu } } _ { \mu } ( \dot { h } )$ , exactly equals the true environment reach probability ofits corresponding chance node, $\nu _ { \mu } ( h , a )$ . Consequently, we can translate the latent summation directly into an expectation over the true game space:

$$
\begin{array} { r l } & { \mathbb { E } _ { \boldsymbol { \mu } } [ \mathcal { L } ( \mathcal { M } ^ { * } ) ] = \displaystyle \sum _ { \hat { h } \in \mathbb { \hat { H } } ^ { * } } \hat { \nu } _ { \boldsymbol { \mu } } ( \hat { h } ) \mathcal { L } _ { K L } ( \hat { h } ) } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ &  \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \ \end{array}
$$

Next, we establish the strict suboptimality of the model exhibiting branch duplication, $\mathcal { M } ^ { b }$ $B y$ Lemma $I , \mathcal { M } ^ { b }$ achieves the same KL loss . However, by definition, $\mathcal { M } ^ { b }$ maps a single true history $h \in \mathbb { H } _ { n c }$ to a set of distinct reachable latent states $\mathbb { M } _ { h } \subseteq \mathbb { M } ^ { * }$ . Crucially, all duplicated states $m \in \mathbb { M } _ { h }$ yield the exact same joint latent infoset $( \hat { I } _ { 1 } , \hat { I } _ { 2 } )$

The centralized unifier network is a deterministicfunction trained to predict the central model state $m = ( \hat { h } , z )$ from the joint latent infosets. In $\mathcal { M } ^ { b } ;$ , the input $( \hat { I } _ { 1 } , \hat { I } _ { 2 } )$ remains constant across the set $\mathbb { M } _ { h }$ , but the target state m varies as the encoder distributes the observation across different stochastic outcomes z. Because a deterministic neural network mapping a constant input to multiple varying targets incurs an irreducible variance penalty, it must yield a strictly positive unifier loss. Therefore, $\mathbb { E } _ { \mu } \mathsf { \bar { [ } \mathcal { L ( M ^ { b } ) ] } > \mathbb { E } _ { \mu } [ \mathcal { L ( M ^ { * } ) } ] }$ ].

Finally, we establish the strict suboptimality ofthe model violating the game’s information partitioning, $\mathcal { M } ^ { l }$ . By definition, M<sup>l</sup> is tree-structure isomorphic butfails the infoset isomorphism condition. This means there exists at least one player i and two valid true histories $h , h ^ { \prime } \in \bar { \mathbb { H } } _ { n c }$ mapped to latent states $m , m ^ { \prime }$ such that the equivalence relation $\| \hat { I } _ { i } ( m ) - \hat { I } _ { i } ( m ^ { \prime } ) \| < \epsilon \iff I _ { i } ( h ) = I _ { i } ( h ^ { \prime } )$ is broken. This violation must manifest in one oftwo cases:

Case 1; $I _ { i } ( h ) = I _ { i } ( h ^ { \prime } )$ but $\| \hat { I } _ { i } ( m ) - \hat { I } _ { i } ( m ^ { \prime } ) \| \geq \epsilon .$ In games with perfect recall, histories in the same true infoset share the exact same Action-Observation History (AOH)for player i. Because the decentralized infoset network processes this AOH to generate ${ \hat { I } } _ { i } ,$ , it acts as a deterministic function. It is impossiblefor a deterministicfunction to output distinct embeddings for identical inputs. Thus, this case cannot practically occur.

Case 2: $I _ { i } ( h ) \neq I _ { i } ( h ^ { \prime } )$ but $\| \hat { I } _ { i } ( m ) - \hat { I } _ { i } ( m ^ { \prime } ) \| < \epsilon .$ . Because the true infosets are distinct, perfect recall dictates that the underlying AOHfor player i must differ in at least one previous action or observation. However, the model aliases them to the same latent infoset embedding. The architecture utilizes decentralized auto-encoding targets,forcing the network to decode the current observation andprevious actionfrom ${ \hat { I } } _ { i } .$ . Because a deterministic decoder cannot map a single, aliased embedding to two distinct true AOH targets, it necessarily incurs a strictly positive reconstruction error.

We note thatforcing this individual infoset reconstruction loss is a theoretical necessity, not merely an empirical stabilization trick. In an asymmetric game where Player 1 possesses perfect information, Player 1’s infoset alone could uniquely identify the history h. Without individual decentralized targets, the centralized unifier loss could perfectly minimize to zero relying exclusively on Player 1, entirely failing to constrain Player 2’s latent partition.

Because Case 1 is structurally impossible, the violation in $\mathcal { M } ^ { l }$ must manifest as Case 2, incurring a strictly positive individual infoset loss. Therefore, $\mathbb { E } _ { \mu } [ \mathcal { L } ( \mathcal { M } ^ { l } ) ] > \mathbb { E } _ { \mu } [ \mathcal { L } ( \dot { \mathcal { M } } ^ { * } ) ]$ .

Remark 1 (Optimization and Benign Branch Duplication) While Theorem 2 establishes that the branch-duplicated model $\mathcal { M } ^ { b }$ has a strictly higher global minimum than the perfectly isomorphic model $\mathcal { M } ^ { \ast }$ , standard one-way optimization (where the centralized unifier target is treated as a constant with stop-gradients) does not provide a direct gradient signal to the encoder to actively merge these duplicated branches. Consequently, in practice, the model may converge to $\mathcal { M } ^ { b }$ rather than M<sup>∗</sup>. However, because all duplicated states within $\mathbb { M } _ { h }$ accurately reconstruct the true environment emissions and preserve the true marginal chance probabilities, this branch duplication is a benign structural artifact that does not degrade downstream policy optimization, unlike information partition violations or posterior collapse.

Because $\mathcal { M } ^ { * }$ accurately captures the underlying POSG structure, which can be “unrolled” into an EFG [20], and because ${ \dot { \mathcal { M } } } ^ { b }$ is indistinguishable from it from the actor-critic’s view, the RNaD convergence result applies directly.

Corollary 2 (Nash Equilibrium Convergence) Assume that (i) the world model has converged to the infoset-isomorphic model $\mathcal { M } ^ { * }$ (or to $\bar { \mathcal { M } } ^ { b } )$ and is heldfixed during policy optimization, and (ii) the RNaD dynamics are realized exactly, i.e., with exact policy evaluation and without functionapproximation or sampling error. Then NashDreamer converges to a Nash Equilibrium of the underlying game.

Proof 4 (Proof of Theorem 1) To prove the existence of this degenerate global minimum, we construct a specific counterexample environment and evaluate the expected losses of the isomorphic model M<sup>∗</sup> and the collapsed model $\mathcal { M } ^ { c o l l a p s e }$

Assume the standard Dreamer loss scaling coefficients: $\beta ^ { e n c } = 0 . 1$ , and all other coefficients (dynamics, reward, observation, and infoset) are 1.0.

Counterexample Environment Construction: Consider an environment that begins with a single chance node with 101 uniformly distributed outcomes. The remainder ofthe environment is determin istic, and a deterministic joint policy µ guarantees reaching the terminal state. The initial chance node outcome has two distinct effects:

1. It dictates a continuous component of a subsequent observation for Player $I , o _ { 1 } ^ { t } ,$ , selecting a value such that its transformed representation in symlog spacefalls uniformly in the set $\{ 0 , 0 . 0 1 , 0 . 0 2 , \ldots , 1 \}$

2. It penalizes the final terminal reward, subtracting a value such that the final reward in symlog spacefalls uniformly in the same set $\{ 0 , 0 . 0 1 , 0 . 0 2 , \ldots , 1 \}$

Expected Loss of the Isomorphic Model (M<sup>∗</sup>): By Theorem 2, the isomorphic model incurs zero prediction and unifier losses, leaving only the KL divergence penalty. The inherent entropy of the 101-outcome uniform chance node is log(101) ≈ 4.605. Applying the loss coefficients, the expected loss is exactly:

$$
\mathbb { E } _ { \mu } [ \mathcal { L } ( \mathcal { M } ^ { * } ) ] = ( \beta ^ { e n c } + \beta ^ { d y n } ) H ( T ) = ( 0 . 1 + 1 . 0 ) \log ( 1 0 0 ) \approx 5 . 0 6
$$

Expected Loss of the Collapsed Model $( \mathcal { M } ^ { c o l l a p s e } ) { : }$ Now consider a degenerate model where the posterior completely ignores the observation and collapses to the prior. Because $q ^ { \theta } ( z \mid o _ { t } , \hat { h } _ { t } ) =$ $p ^ { \theta } ( z \mid \hat { h } _ { t } )$ , the KL divergence loss is exactly 0. The encoder maps all 101 chance outcomes to a single model state. Consequently, the model must rely on this static representation to predict the varying observations and rewards.

For the observation reconstruction (optimized via Mean Squared Error), the target symlog values span $\{ 0 , \ldots , 1 \}$ . A sufficiently trained predictor will output the mean value of 0.5, the maximum possible squared error for any outcome is $( 1 - 0 . 5 ) ^ { 2 } = { \dot { 0 } } . 2 5$ . Therefore, the expected observation loss is strictly bounded: $\mathcal { L } ^ { o b s } \le 0 . 2 5$

For the reward prediction, we utilize the standard two-hot cross-entropy loss over discretized bins. The true targets in symlog space fall uniformly in $\{ 0 , \ldots , 1 \}$ . In the two-hot encoding scheme, a target $y \in [ 0 , 1 ]$ is represented as a linear mixture: $( 1 - y )$ probability mass on the bin for 0, and y probability mass on the binfor 1. A sufficiently trained predictor thus outputs a uniform distribution over the two adjacent bins (predicting 0.5 for bin 0 and 0.5 for bin 1), the cross-entropy loss for any target y is:

$$
- \left( ( 1 - y ) \log ( 0 . 5 ) + y \log ( 0 . 5 ) \right) = - \log ( 0 . 5 ) = \log ( 2 ) \approx 0 . 6 9
$$

Thus, the expected reward loss is bounded: $\begin{array} { r } { \mathcal { L } ^ { r e w a r d } \leq 0 . 7 . } \end{array}$

The individual infoset losses can all minimize to zero. For Player 2, all true historiesfall into the same infoset, requiring only a constant embedding. For Player 1, the decentralized network can simply produce a distinct latent infoset embedding for each unique observation o<sup>t</sup> . Finally, because the encoder collapses all outcomes into a single latent state, the centralized unifier network only needs to map these varyingjoint latent infosets to a single, constant target state. Learning this trivial surjective mapping allows the unifier loss to reach 0. We explicitly note that this degenerate mapping does not contradict our previous result (Theorem 2) regarding the strict suboptimality ofinformation partition violations. That penaltyfundamentally relied on the model maintaining tree-structure isomorphism (perfect deterministic reconstruction), whereas this collapsed model deliberately abandons structural isomorphism to save on KL divergence.

Summing the bounded predictor penalties, the total expected loss ofthe collapsed model is:

$$
\mathbb { E } _ { \mu } [ \mathcal { L } ^ { m o d e l } ( \mathcal { M } ^ { c o l l a p s e } ) ] \leq 0 . 2 5 + 0 . 6 9 = 0 . 9 4
$$

Because $\mathbb { E } _ { \mu } [ \mathcal { L } ^ { m o d e l } ( \mathcal { M } ^ { c o l l a p s e } ) ] \le 0 . 9 4$ and $\mathbb { E } _ { \mu } [ \mathcal { L } ^ { m o d e l } ( \mathcal { M } ^ { * } ) ] \approx 5 . 0 6 ,$ , the degenerate collapsed model achieves a strictly lower objective loss than the perfectly structure-preserving isomorphic model. This proves that under standard static KL-balancing, the global minimum is vulnerable to posterior collapse in environments with high chance entropy and low-magnitude predictor variance.

We note that, in our counterexample environment, we let only one player’s observations and reward be determined by the chance outcome. Since our world model is also centralized, an analogous counterexample single-agent environment could be constructed, proving this property also for DreamerV3.

Finally, we discuss that with the encoder/representation loss disabled, our model will, in theory, converge to an infoset-isomorphic model or a model with duplicated branches, both of which enable sound actor-critic learning.

Theorem 3 (Convergence of One-Way Optimization) Let ${ \mathcal { L } } ^ { \mathrm { o n e - w a y } }$ denote the optimization objective where the posterior encoder receives no gradientfrom the KL divergence penalty (i.e. setting $\beta ^ { e n c } = 0 )$ . Under the assumptions of unbounded network capacity and a sufficiently exploratory sampling policy µ, the global minimum ofL<sup>one−way</sup> is achieved strictly by the set ofmodels that are either perfectly infoset-isomorphic (M<sup>∗</sup>) or exhibit benign branch duplication $( \mathcal { M } ^ { b } )$

Proof 5 Under L<sup>one-way</sup>, the parameters of the posterior encoder $q ^ { \theta } ( z \mid o _ { t } , \hat { h } _ { t } )$ are updated exclusively by the deterministic predictor losses (reward, continuation, observation, legal actions).

To achieve a loss of0 on the deterministic environment predictors, the encoder must map true histories to latent states that maintain ϵ-distance matching. This enforces either tree-structure isomorphic model, or duplicated branches. Even though Theorem 2 shows that the isomorphic model achieves a lower total loss, as stated in Remark 1 the model lacks a gradient signal toforce this convergence. However, as branch duplication is a benign artifact and all the branches are structurally identical and have the same joint infoset embedding, we can merge them for the purpose of this proof and consider only the infoset isomorphic model M<sup>∗</sup>.

With the encoder forced to maintain tree-structure isomorphism to satisfy the predictors, we consider the remaining structural losses. By Theorem 2, any model that maintains tree-structure isomorphism but violates the game’s true information partitioning (M<sup>l</sup>) will incur a strictly positive, reducible penalty in the infoset individual loss

Concurrently, the dynamics prior $p ^ { \theta } ( z \mid \hat { h } _ { t } )$ is updated to minimize the KL divergence against this fixed, structurally sound posterior target. As established in Lemma 1, the optimal prior will perfectly match the aggregated marginal posterior, achieving a KL divergence exactly equal to the true chance entropy.

M<sup>∗</sup> and M<sup>b</sup> perfectly minimize all reconstruction losses to 0, reduce the dynamics loss to the real environment entropy (the lowest obtainable non-degenerate value) and achieve a global/local minimum of the infoset loss, respectively. So, M<sup>∗</sup> forms the global learnable minimum under L<sup>one−way</sup> (but not necessarily the global minimum in general, as demonstrated in Theorem 1) and M<sup>b</sup> forms a local minimum. Thus, given sufficient data and compute, the architecture is mathematically guaranteed to converge to an environmentally and game-theoretically sound representation.

## C Training Algorithm

Algorithm 1 states the complete NashDreamer training loop, collecting in one place the world-model objective of Section 4.1, the imagination procedure of Section 4.2 and the actor-critic of Section 4.3. Three properties of the loop are worth making explicit, as they are easy to miss in the prose. Real environment data is collected at every iteration. The world model is updated at every iteration and is not frozen during the warm-up period. The warm-up period gates the imagination phase alone: during it the actor-critic trains on real trajectories only, and the world model trains exactly as it does afterwards. All hyperparameter values are listed in Table 2.

The centralized critic v is evaluated on the joint representation $( \hat { I } _ { 1 } ^ { t } , \hat { I } _ { 2 } ^ { t } )$ and is used only during training; each actor $\pi _ { i }$ conditions on $\hat { I } _ { i } ^ { t }$ alone, so the policies remain executable from one player’s action-observation history at test time. Note that phase (iii) generates one full rollout from the root per non-chance node of the real minibatch, terminal nodes included, so the number of imagined trajectories per gradient step grows with the length of the game rather than being a fixed multiple of |B|: Goofspiel-5 yields five imagined trajectories per real trajectory, Goofspiel-13 thirteen, and Battleship $5 \times 5$ twenty-eight. The resulting counts for the head-to-head domains are given in Appendix Q.

On the warm-up period. The warm-up of phase (iii) is a stabilizing mechanism rather than a theoretical requirement, and nothing about it is specific to imperfect information. Policy-gradient actor-critics, including the REINFORCE actor of DreamerV3, are not designed for a non-stationary optimization target, so a warm-up would be equally defensible in a single-agent POMDP. What the imperfect-information setting adds is that the non-stationarity arrives from several directions at once, and each family of methods available here pays for it somewhere: through careful hyperparameter tuning (MMD, PPO-style methods), through a regularization mechanism that is itself a source of non-stationarity (the periodic $\pi ^ { \mathrm { r e g } }$ update of RNaD, Section 2.2), or by requiring explicit access to the true game/estimating the true distributions via importance sampling (CFR and its variants). Imagination from a world model that is still changing rapidly adds a further source on top of these, and the warm-up removes it from the phase of training where it is most harmful. Appendix R sweeps its length in Leduc Poker and finds the method not to be sensitive to it.

Algorithm 1 NashDreamer[RNaD]. $N ^ { \mathrm { w a r m } }$ is the world-model warm-up length, $N ^ { \mathrm { r e g } }$ the   
regularization-policy update period, $| B |$ the batch size and $T ^ { \mathrm { m a x } }$ the maximum trajectory length of   
the game (Table 2). Phase (ii) corresponds to Figure 1 and phase (iii) to Figure 2.   
Require: game $\mathcal { G } ;$ world model parameters $\theta$ comprising the MARSSM (seq, enc, dyn, iset) and the   
predictors; decentralized actors $\pi _ { 1 } , \pi _ { 2 } ;$ centralized critic v   
1: ${ \bar { \pi } _ { i } ^ { \mathrm { r e g } } }  \pi _ { i }$ for $i \in \{ 1 , 2 \}$   
2: for $n = 1 , \ldots , N ^ { \mathrm { s t e p s } }$ do   
3: (i) Real environment interaction — at every iteration   
4: collect a minibatch $B$ of $| B |$ complete trajectories from $\mathcal { G }$ under the joint sampling policy $\mu ,$   
each unrolled from $s _ { 0 }$ to termination   
5: (ii) World model update — at every iteration; $\theta$ is neverfrozen   
6: for all trajectories in $B ,$ , for $t = 0 , \ldots , T$ do   
7: $\hat { h } _ { t } \gets \mathrm { s e q } ( m _ { t - 1 } , ( a _ { 1 } ^ { t - 1 } , a _ { 2 } ^ { t - 1 } ) )$ {joint action}   
8: $q _ { t } ^ { \theta }  \operatorname { e n c } ( \hat { h } _ { t } , ( o _ { 1 } ^ { t } , o _ { 2 } ^ { t } ) ) , \quad p _ { t } ^ { \theta }  \operatorname { d y n } ( \hat { h } _ { t } ) , \quad z _ { t } \sim q _ { t } ^ { \theta }$ {posterior}   
9: $m _ { t } \gets [ \hat { h } _ { t } , z _ { t } ]$   
10: $\hat { I } _ { i } ^ { t } \gets$ iset $( \hat { I } _ { i } ^ { t - 1 } , a _ { i } ^ { t - 1 } , o _ { i } ^ { t } )$ for $i \in \{ 1 , 2 \}$ {own AOH only}   
11: end for   
12: take a gradient step on $\theta$ with $\mathcal { L } ^ { \mathrm { m o d e l } }$ of Eq. 4   
13: (iii) Imagination — gated by the warm-up; $\theta _ { \ J }$ frozen throughout   
14: $\hat { B } \gets \emptyset$   
15: if $n > N ^ { \mathrm { w a r m } }$ then   
16: for all non-chance nodes of each trajectory in $B ,$ terminals included do   
17: $m _ { 0 } , \hat { I } _ { 1 } ^ { 0 } , \hat { I } _ { 2 } ^ { 0 } \gets$ posterior states at $t = 0$ computed in (ii) {all rollouts start at the root}   
18: for $t = 0 , \ldots , T ^ { \mathrm { m a x } } - 1$ do   
19: $\hat { l } _ { 1 } ^ { t } , \hat { l } _ { 2 } ^ { t } , \hat { r } _ { t } , \hat { b } _ { t }$ ← predictors $( m _ { t } )$   
20: $a _ { i } ^ { t } \sim \pi _ { i } ( \cdot | \hat { I } _ { i } ^ { t } , \hat { l } _ { i } ^ { t } )$ for $i \in \{ 1 , 2 \}$ {decentralized}   
21: $\hat { h } _ { t + 1 }  \mathrm { s e q } ( m _ { t } , ( a _ { 1 } ^ { t } , a _ { 2 } ^ { t } ) )$   
22: $p _ { t + 1 } ^ { \theta }  \mathrm { d y n } ( \hat { h } _ { t + 1 } ) , \quad z _ { t + 1 } \sim p _ { t + 1 } ^ { \theta }$ {prior}   
23: $m _ { t + 1 } \gets [ \hat { h } _ { t + 1 } , z _ { t + 1 } ]$   
24: $\hat { o } _ { i } ^ { t + 1 } \gets$ observation\_decoder $\cdot ( m _ { t + 1 } )$ for $i \in \{ 1 , 2 \}$ {decoder is generative here}   
25: $\hat { I } _ { i } ^ { t + 1 } \gets \mathrm { i s e t } ( \hat { I } _ { i } ^ { t } , a _ { i } ^ { t } , \hat { o } _ { i } ^ { t + 1 } )$ for $i \in \{ 1 , 2 \}$   
26: end for   
27: append the rollout to $\hat { B }$   
28: end for   
29: end if   
30: (iv) Actor-critic — one RNaD step on real and imagined data jointly   
31: update $\pi _ { 1 } , \pi _ { 2 } ,$ v by off-policy simultaneous-move RNaD on $B \cup { \hat { B } } ,$ , with rewards regularized   
toward $\pi ^ { \mathrm { r e g } }$ as in $\dot { \mathrm { E q . } } 5$   
32: if n mod $N ^ { \mathrm { r e g } } = 0$ then   
33: $\pi _ { i } ^ { \mathrm { r e g } }  \pi _ { i }$ for $i \in \{ 1 , 2 \}$   
34: end if   
35: end for

The warm-up should not be read as realizing assumption (i) of Corollary 2, which requires a world model that has converged and is held fixed during policy optimization. As Algorithm 1 makes explicit, the model is updated for the entire run; the warm-up narrows the gap to that assumption without closing it, and it is one of the approximations discussed in Section 4.4.

## C.1 The Imagination Step

Section 4.2 summarizes the imagined step in prose. Stated in full, and starting from a model state $m _ { 0 }$ obtained during world model training, each step proceeds as follows.

1. Reconstruct current per-player legal actions $\hat { l } _ { i } ^ { t } .$ , the reward $\hat { r _ { t } }$ and the termination flag $\hat { b _ { t } }$ from $m _ { t }$

2. Each actor samples an action $a _ { i } ^ { t }$ from its policy conditioned on the player’s latent infoset and legal actions: $a _ { i } ^ { t } \sim \pi _ { i } ( \cdot | \hat { I } _ { i } ^ { t } , \hat { l } _ { i } ^ { t } )$ . The legal actions are used to explicitly mask out the policy output, rather then letting the actor learn the illegal actions.

3. The sequence model advances the recurrent state: $\hat { h } _ { t + 1 } = \mathrm { s e q } ( m _ { t } , ( a _ { 1 } ^ { t } , a _ { 2 } ^ { t } ) )$

4. The dynamics predictor produces the prior distribution $p _ { t + 1 } ^ { \theta } = \mathrm { d y n } ( \hat { h } _ { t + 1 } )$ , from which a stochastic state $z _ { t + 1 }$ is sampled, yielding the next model state $m _ { t + 1 } = [ \hat { h } _ { t + 1 } , z _ { t + 1 } ]$

5. We obtain per-player observation reconstructions $\hat { o } _ { i } ^ { t + 1 }$ from $m _ { t + 1 }$ using the predictor.

6. The infoset model updates each player’s latent infoset: $\hat { I } _ { i } ^ { t + 1 } = \mathrm { i s e t } ( \hat { I } _ { i } ^ { t } , a _ { i } ^ { t } , \hat { o } _ { i } ^ { t + 1 } )$

Step 4 is the key difference from standard Dreamer, where the decoder is merely a training signal. Here, the infoset model requires observation input (step 5), so the decoder becomes an essential generative component of the imagination pipeline.

## D Contemporary approaches

So far, Dreamer family algorithms in multi-agent setting employ standard MARL techniques for non-stationarity mitigation [30] and decentralized models. We argue that these fail in adversarial settings. Explicit communication [9, 6] is incompatible with zero-sum games, as rational players will learn to ignore it. Alternatively, population self-play or explicit opponent modeling [25, 29] suffers from the aforementioned policy-physics entanglement, which can have catastrophic results upon encountering a novel opponent.

This is also why we do not report these systems as empirical baselines (Section 5): each is structurally inapplicable to 2p0s IIGs rather than merely inconvenient to run.

• MAMBA [9] and DreamerCF [6] are cooperative methods built around inter-agent communication. In a zero-sum game, a communication channel carries no equilibrium value, as rational opponents ignore or exploit it, so porting these methods amounts to deleting their central mechanism.

• The opponent-embedding approach of Lu and Chen [25] conditions the world model on an embedding of the opponent’s identity that must be randomly initialized at test time against an unseen opponent. There is then no fixed joint policy to evaluate, and NashConv, which requires one, is not well defined for such a model.

• Population self-play with single-agent models [29] instantiates exactly the policy-physics entanglement described above: what is learned is a joint-system model of the environment together with the historical opponent population, not a model of the game. Our decentralized MARSSM ablation is a controlled instance of this family, with the same optimizer and capacity and only the centralization removed, which we consider a fairer comparison than re-implementing any single such system together with its confounding design choices.

• LAMIR [21] extends MuZero-style search to IIGs, but assumes deterministic transitions, which excludes both Leduc and Goofspiel-13, and targets search-time planning rather than model learning for sample efficiency, so the comparison would be ill-posed along the axis we measure.

As a motivating example, consider the Battleship game. Since a decentralized model is, from a particular player point of view, effectively a POMDP marginalized over past opponents, the player might hallucinate the opponent’s fleet placement based purely on historical correlations with the ego-agent’s own layout. Observing a hit or miss that contradicts this entrenched hallucination breaks the internal belief representation, rendering all subsequent latent planning nonsensical.

## E Architectural design

## E.1 Mitigating Compounding Errors in High-Dimensional Spaces

As noted in Section 4.2, our current MARSSM formulation reconstructs the explicit observation $\hat { o } _ { i , t }$ at each imagination step to update the individual latent infosets $\hat { I } _ { i , t }$ . Because our empirical evaluations isolate structural learning by using compact infoset representation vectors, this explicit decoding is computationally inexpensive and highly interpretable.

However, scaling this architecture to high-dimensional raw observations (such as images) presents a vulnerability. Standard Dreamer architectures transition strictly within the latent space to avoid the compounding auto-regressive errors and computational overhead of decoding raw pixels at every step of imagination.

To resolve this when scaling MARSSM, the architecture can seamlessly adopt a latent bottleneck approach. Rather than decoding the raw observation directly, the centralized world model would predict a low-dimensional observation embedding $e _ { i , t }$ (this approach is already employed by the encoder, before predicting the posterior latent stochastic state). The decentralized infoset networks would then aggregate these compact embeddings instead of raw observations. The actual highdimensional observation reconstruction would be decoupled from the autoregressive imagination loop, acting solely as an auxiliary loss on $e _ { i , t }$ during training. This two-stage approach preserves the causal information chain required for IIGs while maintaining the computational efficiency and robustness to compounding errors characteristic of the broader Dreamer family.

## E.2 Replay Buffer Design

Our replay buffer stores complete trajectories and unrolls sequences strictly from the initial state t = 0. We store fixed-size trajectories and pad with invalid data if needed. This differs fundamentally from standard MBRL practice of sampling random mid-episode chunks. In IIGs, games often have rigid starting configurations, and uncertainty is generated iteratively by hidden opponent actions. Initializing a recurrent state at t > 0 severs the causal chain of strategic deductions, making it impossible for the infoset networks to reconstruct accurate beliefs. While this presents a scaling limitation for very long-horizon games, it ensures rigorous adherence to the game’s informationtheoretic properties.

While it is theoretically possible to bypass the t = 0 requirement by storing the latent model states and infoset vectors directly into the buffer [14], doing so in an IIG introduces two fatal issues. First, due to representation drift, latent states stored in the buffer are generated by stale network parameters and lose their semantic meaning as the networks evolve.

Second, and more fundamentally, because a player’s policy conditions solely on their infoset, optimizing that policy requires the empirical distribution of the underlying histories within that infoset to perfectly match their true reach probabilities under the current joint policy. Sampling mid-episode histories from an off-policy replay buffer intrinsically violates this property, as the sampled histories reflect stale opponent strategies. Correcting this mismatch requires importance sampling or counterfactual unbiasing, which injects prohibitively high variance into the learning process.

How costly that correction is for a replay buffer of full trajectories, which is the design we do use, is a game-dependent matter rather than a uniform verdict on replay. The importance ratio accumulates along a trajectory, so the variance grows with the length of the game, and it grows with how far the policies travel between the moment a trajectory enters the buffer and the moment it is replayed. We consequently clip the counterfactual importance-sampling ratio (Table 2); in Battleship the gradients diverge without it. The empirical picture follows suit: replay does not improve on plain RNaD in Goofspiel-5, whereas in Leduc and in Phantom Tic-Tac-Toe it does, and in the two largest domains it is the weaker of the two RNaD baselines under both head-to-head play (Appendix Q) and approximate exploitability (Table 1). We therefore offer the variance argument as an explanation of the cases where replay does not help, not as a prediction that it never will.

## F Environments

Point Card Matching (PCM). A single-agent environment inspired by Goofspiel. The agent holds M cards $( 1 , \ldots , M ) ;$ ; point cards are revealed in descending order. Matching the point card scores a point. Reward is sparse (total score at termination). Features dynamic action masking and strict sequentiality.

Imperfect Information Goofspiel. A two-player simultaneous-move card game. Players simultaneously play cards; the higher card wins the revealed point card (ties discard it). We use an imperfect-information variant where players observe round outcomes (win/loss/tie) but not the oppo nent’s specific card. We evaluate on 3-card, 5-card (descending point order), and 13-card (random point order) variants.

Leduc Poker [38]. A condensed Texas Hold’em with a 6-card deck (J, Q, K in two suits). Features private cards, a public community card, and two betting rounds. As this is a turn-based game, it is adapted to our framework via dummy actions. Trajectories reach up to 8 steps with multiple chance nodes.

Battleship. A two-player board game with placement and battle phases. Players simultaneously and secretly place ships on their grid (ships may not overlap or be adjacent), then simultaneously target opponent tiles each round, receiving hit/miss feedback. We evaluate on a 5×5 board, with two size-2 ships and one size-3 ship for head-to-head comparison.

Phantom Tic-Tac-Toe. A turn-based, imperfect-information variant of Tic-Tac-Toe on $\mathrm { ~ a ~ 3 ~ } \times \mathrm { ~ 3 ~ }$ board in which neither player observes the opponent’s marks. A player attempting to claim a tile already held by the opponent is informed of the collision and replays their move, repeating until they place on a free tile, so a turn can consist of several attempts. Following the OpenSpiel [23] and exp-a-spiel [33] implementations, we additionally forbid a player from attempting a tile they already know the opponent occupies. This keeps the game length bounded, and it is without loss of information under our perfect-recall assumption (Section 2), since a player remembers every collision they have encountered. This is a deterministic turn-based game, with trajectories reaching up to 17 steps. It is also adapted via dummy actions.

Perturbed Rock-Paper-Scissors. The canonical one-shot matrix game, modified so that the rewards for playing Scissors are multiplied by two, which moves the equilibrium off the uniform strategy; the uniform policy has a NashConv of 2/3. It is the smallest simultaneous-move game we use, and serves as an isolated testbed for the identifiability barriers of Section 3 (Appendix L).

## G Implementation Details

All models and environments were implemented in Python 3.12 using the JAX [4] ecosystem and the newest version of Flax [16] (NNX). To keep consistent with the DreamerV3 architecture [14], we use a single LaProp optimizer [43] with a constant learning rate of $\alpha = 3 \times 1 0 ^ { - 4 }$ for all network parameters. Training utilizes a batch size of 64 complete trajectories per gradient step. All experiments were conducted on a cluster, utilizing 2 cores of AMD server processors with up to 30 GB of system RAM for the PPO budgeted approximate NashConv across 10 stored model seeds and 12 GB for other experiments (experiment dependent) and a single NVIDIA RTX A4000 GPU, running on Debian 12 with CUDA 13.0. To upper-bound the complexity, our most time-demanding experiments (Battleship 5x5) required a maximum of 24 hours of total compute time to train for 30,000 minibatches across 10 seeds. The estimated compute time for all the experiments (including preliminaries not reported in the paper) is around 4 weeks of CPU/GPU time.

## G.1 Network Architecture

As our work focuses on algorithmic adaptations rather than novel network structures, we use the standard DreamerV3 network architecture throughout. Each predictor component is implemented as a Multi-Layer Perceptron (MLP) utilizing layer normalization [3] and SiLU activations. The sequential network embeds input components separately, creates a skip connection with the recurrent state, concatenates the features, passes them through an MLP, and uses a GRU cell [8] as the gate unit. The posterior encoder consists of two primary pathways: one embedding the joint observation, and one predicting the stochastic latent distribution conditioned on both the recurrent state and this observation embedding.

## G.2 Capacity Constraints

MLP Dimensions: Every MLP predictor across the architecture consists of an input layer, a single hidden layer of 256 units, and an output layer (with layer normalization and SiLU applied after the input and hidden layers). The dimensions of the input and output layers are determined based on the specific input and target sizes of the respective environment.

To test the model’s ability to capture dynamics under heavily constrained capacity, we strictly the size of latent state representations.

Deterministic States: We constrain the dimensionalities of the deterministic latent vectors to correspond directly to the true information content of the environment. Specifically, the central recurrent state $\hat { h } _ { t }$ matches the dimensionality of the true joint infoset representation, and each decentralized latent infoset $\hat { I } _ { i , t }$ matches the true individual player’s infoset dimensionality. The specific per-environment infoset representation sizes are:

• Point Card Matching with 3 cards (PCM 3): 16

• Goofspiel: 35 (3 cards), 87 (5 cards), 535 (13 cards)

• Leduc Poker: 39

• Phantom Tic-Tac-Toe: 104

• Battleship: 716 (for a 5x5 board with 3 ships each)

Stochastic Latent Space: Rather than relying on the excessively large stochastic capacity of standard DreamerV3 $( \mathrm { e . g . , 3 2 \times 3 2 }$ classes), we deliberately restrict the categorical distribution to the minimal capacity required to uniquely identify chance outcomes without forcing the network to “temporally multiplex” representations.

Temporal multiplexing occurs when the latent capacity is so constrained that the network must assign the exact same latent code $z _ { t }$ to completely distinct physical chance outcomes, relying entirely on the recurrent state $\hat { h } _ { t }$ to disambiguate them $( \mathrm { e . g . }$ ., class 1 representing a private card at step 1, but a public card at step 3). This places immense optimization strain on the dynamics MLP.

To prevent this, we allocate exactly enough capacity to assign a distinct latent configuration to each unique class of chance node. For Imperfect Information stochastic Goofspiel-13, we use a single distribution of 13 categories. While a single distribution of 34 categories should be sufficient for Leduc, we find that under our capacity constraints, the model cannot learn the underlying environment, due to gradient dilution for a single, large distribution. So, our main results use a factored stochastic state of 2 categorical distributions with 6 categories. We present a comparison in Appendix J. We explicitly note that utilizing domain knowledge to size the latent space is an experimental choice designed to evaluate the framework under minimal capacity conditions. It is not a structural requirement of MARSSM. For environments with unknown or intractably large chance dynamics, the framework natively supports scaling back up to latent spaces with many distinct outcomes (e.g., 32 × 32 classes) typical of the Dreamer family.

In deterministic environments, the free-nats KL clipping threshold is set to 0, while in stochastic environments it is maintained at 0.2.

## G.3 Hyperparameters

Unless explicitly stated otherwise in the environment descriptions, all NashDreamer experiments share the hyperparameters listed in Table 2. The architecture is trained entirely online for 30, 000 gradient steps, where, at each step, a new minibatch is collected from the real environment, used to update the world model, which then generates imagination trajectories minibatch. Then, the actor-critic performs a step on the concatenation of the imagined minibatch with the real minibatch. For the significantly larger state spaces of Goofspiel-13, Battleship $5 \times 5$ and Phantom Tic-Tac-Toe, the initial world model warm-up phase is extended from the default 1, 000 to 2, 000 gradient steps to ensure stable structural representations before policy learning begins.

To ensure numerical stability during training, we apply several targeted regularizations following [14]. First, we add a 0.01 uniform mixture to both the prior and posterior categorical distributions to prevent KL divergence spikes to infinity. Second, during imagination, we apply a state sampling threshold of 0.05 (thresholding individual category probabilities and renormalizing before sampling the stochastic state $z _ { t } )$ to prevent the actor from exploring highly improbable hallucinated branches. The only exception to this threshold is Leduc under the flat stochastic space, where this threshold is set to 0.01 (as the initial chance outcomes have probability $\textstyle { \frac { 1 } { 3 0 } } < 0 . 0 5 )$ . Finally, the target predictors for terminal states and legal actions are thresholded at 0.5 to yield binary masks during execution. For the off-policy RNaD optimization, we also clip the counterfactual importance sampling weights to prevent exploding gradients. We further adopt the DreamerV3 practice of zero-initializing the output layers of the reward and critic predictors, so that both heads predict exactly zero at the start of training; this keeps large, arbitrary early predictions from dominating the world-model loss and the imagined returns during the first iterations, and complements the warm-up period discussed in Appendix C.

Table 2: Shared NashDreamer hyperparameters used across all experiments.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>General Training Total gradient steps</td><td>30,000</td></tr><tr><td>World model warm-up period</td><td>1, 000 (2, 000 for large envs)</td></tr><tr><td>Batch size Optimizer (Shared) Learning rate</td><td>64  $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning rate warmup steps Betas  $( { \bar { \beta } } _ { 1 } , \beta _ { 2 } )$  Epsilon (€) Adaptive Gradient Clipping (AGC)</td><td>1,000 (0.9,0.999)  $\mathrm { i \times 1 0 ^ { - 8 } }$  0.3</td></tr><tr><td>World Model Targets &amp; Regularization Reward symlog bins (one-sided) Categorical uniform mixture State sampling threshold</td><td>20 0.01 0.05</td></tr><tr><td>Terminal / Legal prediction threshold</td><td>0.5</td></tr><tr><td>Dynamics KL scale  $( \beta ^ { \mathrm { d y n } } )$ </td><td>1.0</td></tr><tr><td>Encoder KL scale  $( \beta ^ { \mathrm { e n c } } )$ </td><td>0.1</td></tr><tr><td>Prediction loss scales Infoset loss scale Actor-Critic &amp; RNaD</td><td>1.0 1.0</td></tr></table>

## G.4 Standalone RNaD Baseline Configuration

To ensure a rigorous and fair empirical comparison, the model-free standalone RNaD baseline utilizes the exact same Actor-Critic hyperparameters and optimizer configurations as NashDreamer (listed in Table 2).

Because the standalone baseline relies on offline experience replay rather than online model-based imagination, we maintain a per-trajectory replay buffer with a capacity of 10, 000 trajectories. The baseline is trained using an environment-dependent replay ratio (the number of trajectories sampled

from the buffer per newly collected online trajectory) to compensate for the additional computational overhead of the model-based variant. For Goofspiel-13, this ratio is set to 26, and for Battleship 5 × 5, it is set to 40.

We train the baseline for 30, 000 gradient steps, matching NashDreamer. Each gradient step entails the collection of a new minibatch from the environment, concatenation with replayed trajectories (if used) and performing the step on this larger minibatch.

## G.5 MMD Baseline Configuration

The MMD baseline [36] shares the network architecture, optimizer and batch size of the RNaD baseline; the two differ only in the policy-optimization rule, with MMD replacing RNaD’s policydependent reward regularization by magnet-policy regularization. This parity is what allows the NashDreamer[MMD] and NashDreamer[RNaD] comparison of Appendix Q to isolate the effect of the world model rather than of the surrounding implementation. Concretely, we follow the OpenSpiel PPO implementation, with an entropy exploration bonus and the additional inverse-KL penalty described by Rudolph et al. [33], and we adopt their “generic” hyperparameter setting. All network architectures and layer sizes are identical to RNaD (Appendix G). The MMD-specific values are listed in Table 3.

Table 3: MMD-specific hyperparameters, shared by the model-free MMD baseline and Nash-Dreamer[MMD], and held fixed across all domains.
<table><tr><td>Hyperparameter</td><td>Value</td><td>Role</td></tr><tr><td>PPO epochs</td><td>7</td><td>optimization epochs per batch</td></tr><tr><td>PPO clipping coefficient</td><td>0.1</td><td>realizes the proximal term of Eq. 6</td></tr><tr><td>Inverse-KL coefficient</td><td>0.05</td><td>loss multiplicand of the inverse KL penalty</td></tr><tr><td>Magnet coefficient (α)</td><td>0.2</td><td>fixed uniform magnet, i.e. an entropy bonus</td></tr><tr><td>Advantage-normalization € Discount / bootstrapping (γ, λ)</td><td>1 × 10−8 (1.0, 1.0)</td><td>numerical guard, not a tuned quantity as for RNaD (Table 2)</td></tr></table>

The magnet is fixed and uniform, so the magnet term contributes an entropy bonus rather than an anchor that moves with the policy; as noted in Section 2.2, this means MMD targets a quantal response equilibrium rather than an NE. Advantages are normalized in the standard PPO fashion over the valid (non-padded) entries of the batch,

$$
\tilde { A } = \frac { A - \mathbb { E } [ A ] } { \sqrt { \operatorname { V a r } [ A ] } + \epsilon } ,
$$

with γ and λ set to 1.0 to match the RNaD configuration rather than as independent choices.

We hold these values fixed across domains and do not tune them per environment. The primary reason is that tuning would not affect the claim under test: the model-free MMD baseline and NashDreamer[MMD] use the same MMD hyperparameters, so whichever values are chosen, the comparison isolates the effect of the world model on a given optimizer. An untuned MMD is thus a conservative baseline for the modularity claim of Section 4.3 and a fair one for the sample-efficiency comparison. The values are moreover not arbitrary: they are the generic configuration reported by Rudolph et al. [33] for exactly this family of games. Finally, the domains that motivate this work are those in which environment interaction is the dominant cost, which raises the value of an optimizer that performs acceptably without per-domain tuning; we note as a secondary point that tuning against a world model that is itself still changing is of questionable soundness. We acknowledge that per-domain tuning would likely improve MMD’s absolute numbers, and we make no claim that the values used are optimal for it.

## H Exploitability Evaluation Protocols

## H.1 Exact NashConv

All NashConv values reported in Section 5 are computed exactly against the true game. They are not computed against the learned world model, and they are not estimated by sampling.

At evaluation time we first extract a behavioral policy over the true game. We traverse every infoset of the true game and query the trained actor network at it; for NashDreamer this means replaying that infoset’s AOH $\tau _ { i }$ through the infoset model to obtain the latent embedding $\hat { I } _ { i }$ and reading off $\pi _ { i } ( \cdot | \hat { I } _ { i } )$ with the true legal-action mask applied. The evaluated object is therefore a well-defined joint policy over the true game’s infosets, which is what makes NashConv meaningful as an equilibrium metric

Given this joint policy, each player’s best response is computed by dynamic programming over the full true game tree, following the OpenSpiel exploitability routine [23]; for Phantom Tic-Tac-Toe we instead use the exp-a-spiel library of Rudolph et al. [33], whose implementation of the game matches the variant described in Appendix F. Chance nodes are enumerated exactly with their true probabilities rather than sampled. Simultaneous-move nodes are expanded in the standard turn-based form, in which the second player’s infoset does not contain the first player’s action at that node, so the best-response computation respects the original information structure. The reported values are exact for the extracted policies; the only approximation in the pipeline is the extraction itself, i.e., the network’s own numerical output at each infoset. NashConv curves report values per training seed, evaluated at fixed intervals of environment interaction.

## H.2 Best-Response Learner

Section 5.2 states the budgeted best-response protocol and reports its results; this section documents the learner that protocol runs. Taking the maximum over the ten PPO seeds rather than their mean makes the estimate the tightest lower bound the search budget supports, which is the conservative choice for a metric on which we claim an improvement.

The best-response learner is plain PPO: the same implementation and hyperparameters as the MMD baseline of Appendix G.5, with the inverse-KL penalty and the entropy exploration bonus removed. It is run identically in both games, for 3,000 gradient steps with batch size 64, $\gamma = 1 . 0 , \lambda = 1 . 0$ clipping coefficient 0.1 and 7 inner epochs. The budget of 3,000 steps is a tenth of that of the largest evaluated checkpoint, which suffices because computing a best response against a fixed opponent is a single-agent problem and materially easier than equilibrium finding.

A budget that is too small manifests as a best response failing to reach the value the evaluated profile achieves against itself, which indicates that the search fell short rather than that the checkpoint is unexploitable. We mark any such cell with †. No cell of Table 1 is affected, so the budget is inside the useful range everywhere the main text draws a conclusion from it.

## H.3 Aggregation over PPO Seeds

Table 4 reports the same experiment aggregated by averaging over the 10 PPO seeds instead of taking the worst case. The values are uniformly smaller, as expected of a looser bound, and the least exploitable method is the same one under both aggregations at every checkpoint of both games. The full ranking is identical throughout Goofspiel-13; in Battleship a single adjacent pair exchanges places at each checkpoint, the two RNaD baselines at 10k, MMD and NashDreamer[RNaD] at 20k, and NashDreamer[RNaD] and model-free RNaD at 30k. Each of those exchanges is between values lying well inside one another’s intervals, and every claim made in Section 5.2 holds under either aggregation. One cell is flagged under this looser bound, NashDreamer[RNaD] at 30k in Goofspiel-13, on two of the ten training seeds.

<table><tr><td></td><td colspan="3">Goofspiel (13 cards)</td><td colspan="3">Battleship (5×5)</td></tr><tr><td>Method</td><td>10k</td><td>20k</td><td>30k</td><td>10k</td><td>20k</td><td>30k</td></tr><tr><td>MMD</td><td> $0 . 7 8 9 \pm 0 . 0 7 6$ </td><td> $0 . 6 9 5 \pm 0 . 0 9 8$ </td><td> $0 . 6 8 4 \pm 0 . 0 6 7$ </td><td> $1 . 2 0 2 \pm 0 . 0 5 8$ </td><td> $0 . 9 8 6 \pm 0 . 1 0 0$ </td><td> $0 . 8 3 1 \pm 0 . 1 4 7$ </td></tr><tr><td>ND[MMD]</td><td> $0 . 8 0 5 \pm 0 . 0 6 9$ </td><td> $0 . 7 4 4 \pm 0 . 0 8 5$ </td><td> $0 . 7 0 8 \pm 0 . 0 8 2$ </td><td> $1 . 0 6 9 \pm 0 . 1 1 0$ </td><td> $0 . 7 8 1 \pm 0 . 0 9 9$ </td><td> $0 . 6 1 2 \pm 0 . 1 2 7$ </td></tr><tr><td>RNaD</td><td> $0 . 6 7 5 \pm 0 . 3 6 8$ </td><td> $0 . 3 8 1 \pm 0 . 2 5 8$ </td><td> $0 . 2 7 9 \pm 0 . 2 7 6$ </td><td> $1 . 3 1 2 \pm 0 . 0 6 7$ </td><td> $1 . 1 3 5 \pm 0 . 1 7 2$ </td><td> $0 . 9 3 7 \pm 0 . 1 6 3$ </td></tr><tr><td>RNaD w/ replay</td><td> $0 . 7 0 1 \pm 0 . 0 7 5$ </td><td> $0 . 4 7 6 \pm 0 . 0 7 3$ </td><td> $0 . 4 3 4 \pm 0 . 0 8 1$ </td><td> $1 . 3 0 6 \pm 0 . 0 5 7$ </td><td> $1 . 2 3 5 \pm 0 . 0 8 9$ </td><td> $1 . 0 0 7 \pm 0 . 1 5 2$ </td></tr><tr><td>ND[RNaD]</td><td> $0 . 4 0 8 \pm 0 . 1 3 3$ </td><td> $0 . 1 8 4 \pm 0 . 1 4 3$ </td><td> $0 . 0 9 3 \pm 0 . 1 3 4 ^ { \dagger }$ </td><td> $0 . 9 7 9 \pm 0 . 1 0 5$ </td><td> $0 . 9 5 1 \pm 0 . 1 4 0$ </td><td> $0 . 9 1 2 \pm 0 . 2 3 5$ </td></tr></table>

Table 4: Budgeted approximate exploitability averaged over the 10 PPO seeds, rather than taking the worst case as in Table 1. Mean ±2σ over the 10 training seeds. <sup>†</sup> marks the one cell in which the best response failed to reach the profile’s own value, on two of the ten training seeds, i.e. where the budget was too small.

![](images/e5cf9d5af3b7a19ec6e1fe946efb375becebc93ec24713be845d21a3528c5e38.jpg)  
Figure 4: NashDreamer LE-Tree for round 1 of Leduc Poker (private cards 1 and 2 dealt). Utilizing the factored latent space, the model accurately learns the underlying game dynamics.  
Figure 5: NashDreamer LE-Tree for round 1 of Leduc Poker under a flat latent space (1 × 34). The single categorical distribution suffers from severe mode collapse, failing to learn the true environment structure.

## I Latent Error Trees

To qualitatively assess the structural fidelity of the learned world model, we introduce Latent Error Trees (LE-Trees). Starting from the initial state, we recursively unroll the game tree following the agent’s current joint policy, pruning any branch where a player’s action probability falls below a threshold of 0.05.

In these visualizations, blue diamond nodes represent the ground-truth game states, while the adjacent rectangular nodes represent their corresponding latent reconstructions. These latent nodes are color-coded based on the maximum prediction error across all deterministic decoder heads (reward, continuation, legal actions, and individual infoset reconstruction): green indicates low error (error < 0.2), orange indicates moderate error $( 0 . 2 \leq \mathrm { e r r o r } < 0 . 5 )$ , and red indicates severe error (error ≥ 0.5). Any non-green node is explicitly annotated with the specific predictor component responsible for the maximum error. Finally, all transitions (edges) are visually weighted according to their corresponding chance or policy probabilities.

## J Leduc Failure Modes and Latent Space Capacity

As described in Section 5, while the world model accurately matches the dynamics for round 1 of Leduc Poker (Figure 4), it mistakenly predicts the wrong public card in round 2 (the LE-Tree for round 2 is contained in the Supplementary Material due to size constraints).

Our Leduc implementation utilizes binary observation and infoset tensors, alongside rewards divided by the maximal bet (13), effectively rescaling the reward signal into the [−1, 1] range. This bounded structure meets the criteria of low prediction magnitude changes combined with high stochasticity, which inherently exacerbates the tension between posterior collapse and representation drift in the late stages of the game.

Figure 6 compares the three configurations of NashDreamer[RNaD] we ran on this domain. Replacing the factored latent with a minimal flat one, a single categorical with 34 categories, costs surprisingly little in NashConv, but the world model then fails to capture the environment’s topological structure (Figure 5), which Appendix K.4 quantifies as posterior collapse at the initial deal. The actorcritic evidently compensates for that inaccuracy; since our objective is a structurally accurate and interpretable model, the factored latent remains the better choice.

The one-way variant sets $\beta ^ { \mathrm { e n c } } = 0 ;$ , removing the posterior regularizer as Theorem 3 prescribes, and is the worst of the three in the later stages of training. It does not prevent the structural error at the public chance node, as its prior instead spreads over far more latent configurations than the node has outcomes (Appendices K.4 and K.5), which is what representation drift looks like once nothing pulls the posterior toward the prior. The variant our theory endorses is therefore the one that can perform the worst in practice, which is the tension stated in Section 4.4.

![](images/4b4885a89e61537738761067c7a38ef921f221ab18f1f83a77a798b6f1dd5c2c.jpg)  
Figure 6: NashConv in Leduc Poker for the three latent and loss configurations of Nash-Dreamer[RNaD]: the factored latent, the flat single-categorical latent, and the one-way variant with $\beta ^ { \mathrm { e n c } } = 0$ . All three deteriorate after the initial descent, the one-way variant furthest and the factored latent least. Bold curves are means over the 10 training seeds, faint curves the individual seeds; the dotted line marks the end of the world-model warm-up.

## K Quantitative World-Model Diagnostics

The Latent Error Trees of Appendix I are qualitative. To attribute the policy results of Section 5 to world-model quality rather than to actor-critic dynamics alone, we measure the learned model directly under three protocols, defined in Appendices K.1, K.2 and K.3: rollout validity, the per-step emission error of a trajectory generated entirely by the model; the posterior-collapse indicator, the expected ${ \mathrm { K L } } ( q ^ { \theta } \parallel p ^ { \theta } )$ at chance nodes; and chance-outcome calibration, the agreement between the prior’s distribution over successors and the true one. Results are reported in Appendices K.4 to K.7.

All three protocols filter the model’s categorical stochastic state before use, applying the thresholding used during imagination (Appendix G.3), so that the model is scored on the states it generates at run time. The filter acts on each categorical distribution independently: within a distribution, categories below the threshold are set to zero and the remainder renormalized, so a factored state whose two categoricals read [0.9, 0.0, 0.01] and [0.01, 0.0, 0.9] becomes $[ 1 , 0 , 0 ] \otimes [ 0 , 0 , 1 ]$ at a threshold of 0.05. The threshold must lie below the smallest outcome probability the game presents, or real outcomes are removed before they can be scored; we use 0.05 for the factored variants and 0.01 for the flat one, whose single categorical has to resolve the 1/30 of Leduc’s initial deal.

In Goofspiel-5, which has no chance nodes, every emission converges: terminal flags and legal actions within 1,000 training steps, observations about an order of magnitude later, and the reward last of all. Appendix K.7 reports the four curves and the ordering in full.

In Leduc Poker, the picture is qualitatively different and matches the deterioration visible in Figure 3. Terminal flags and legal actions are again identified without error within 1,000 steps, and deterministic transitions are reconstructed exactly by the factored and one-way variants, but the observation and reward errors never reach zero and the chance transitions stay miscalibrated for the whole of training. Because the three tested model variants in Leduc fail in visibly different ways, we report each diagnostic separately for all three rather than summarizing them here: Appendix K.4 for the posteriorcollapse indicator, Appendix K.5 for chance-outcome calibration and Appendix K.6 for the prediction accuracies under rollout.

These diagnostics confirm the mechanism our theory predicts. NashDreamer learns models of deterministic game components without major issues, whereas in the presence of chance stochasticity the world model can settle on an incorrect representation, or even degrade with further training, and the degradation is localized in the chance transitions rather than in the deterministic emissions. This is the empirical counterpart of Theorem 1: the objective itself rewards a model that declines to disambiguate chance outcomes.

## K.1 Rollout Validity

This protocol measures whether a trajectory generated entirely by the model corresponds to a realizable trajectory of the true game. Unlike a single-step emission score, errors compound along the rollout, as they do when the actor-critic trains on imagined data.

The model rollout and the true game are advanced with the same joint actions, sampled from the model. The recurrent state is initialized at $h _ { 0 }$ and the latent infoset embeddings at their initial values; the initial stochastic state is sampled from the prior $p ^ { \theta }$ . Each step proceeds as follows.

1. At a chance node of the true game, the outcome is not sampled from the true chance distribution but chosen to minimize the $L ^ { \infty }$ distance between the true joint observation and the one reconstructed from the current model state. Divergence later in the rollout is then attributable to the learned dynamics rather than to the two trajectories having drawn different chance outcomes.

2. Each player’s action is sampled as $a _ { i } ^ { t } \sim \pi _ { i } ( \cdot | \hat { I } _ { i } ^ { t } )$ from the latent infoset embedding, or from the infoset reconstructed from the model state when the model was trained on observations alone.

3. $( a _ { 1 } ^ { t } , a _ { 2 } ^ { t } )$ advances the recurrent state through the centralized sequence model and advances the true game; $z _ { t + 1 }$ is then sampled from the prior at the new recurrent state.

4. The observations reconstructed from $m _ { t + 1 }$ update each $\hat { I } _ { i } ^ { t + 1 }$

The rollout terminates when the true game reaches a terminal state.

Error statistics. At each step the model’s emissions are compared with the true ones and the discrepancy accumulated: $L ^ { 1 ^ { * } }$ for the reward, $L ^ { \infty }$ for the joint observation, and 1 per boolean mismatch for the terminal flag and the legal-action mask. Each statistic is the accumulated magnitude divided by the number of valid steps, so all four are mean per-step magnitudes, on different scales.

## K.2 Posterior Collapse at Chance Nodes

By Theorem 2, an infoset-isomorphic model attains an expected $E _ { o } \operatorname { K L } ( q ^ { \theta } \parallel p ^ { \theta } )$ at a chance node equal to the entropy of that node, whereas posterior collapse drives the same quantity to zero.

The model is conditioned on the true action sequence (teacher forcing). At each non-chance node the stochastic state is taken from the posterior, selecting the outcome whose reconstructed joint observation is closest to the true one in $\bar { L } ^ { \infty }$ norm, which keeps the model state aligned with the true history. At a chance node we enumerate the outcomes, evaluate the posterior $q ^ { \delta } ( \cdot | o )$ induced by each, and record their expected divergence from the single prior $p ^ { \theta }$ at that node,

$$
{ \mathcal { C } } = \sum _ { o } p ( o ) \operatorname { K L } \left( q ^ { \theta } ( \cdot \mid o ) \parallel p ^ { \theta } \right) .\tag{7}
$$

Chance nodes are grouped by their number of outcomes $K ,$ and C is averaged, unweighted, over all nodes with the same K. Every chance node in our environments is uniform, so an isomorphic model attains ${ \mathcal { C } } = \log K$

Deviation from log K has two distinct causes. $\mathcal { C } <$ log K indicates posterior collapse: the posterior depends only weakly on the chance outcome. ${ \mathcal { C } } > \log K$ indicates that the prior assigns less mass to the realized posterior mode than a calibrated prior over K equiprobable outcomes would, i.e. that it is spread over more latent configurations than the node has outcomes. A modest excess is expected when the latent space is larger than K and distinct histories reaching the same outcome occupy distinct configurations; a large excess indicates that the prior does not track the posterior.

## K.3 Chance-Outcome Calibration

This protocol measures the prior’s distribution over successors: what fraction of its mass corresponds to realizable outcomes, and whether those outcomes carry the correct probabilities.

Teacher forcing is applied as in Appendix K.2, but the stochastic states are drawn from the prior $p ^ { \theta }$ At each node the model’s stochastic states are enumerated and assigned to true successors: a state is assigned to the successor whose joint observation is nearest its reconstruction in $L ^ { \infty }$ norm, or to a separate unmatched bucket if that distance exceeds 0.3 for every successor. Summing model probabilities within each bucket gives a marginal over the true successors, plus the unmatched mass. Deterministic nodes are included, their true successor distribution being a Dirac.

Two statistics are reported: the $L ^ { 1 }$ distance between the bucketed, unnormalized marginal (so the model marginal need not sum up to 1) and the true successor distribution and the unmatched mass, both bounded by 1. Both are averaged over all nodes of a kind, with chance and deterministic nodes reported separately, and are not grouped by K. The two isolate different failures: unmatched mass measures prior mass on latent states with no corresponding true successor, whereas a large $L ^ { 1 }$ distance with little unmatched mass indicates successors that are represented but mis-weighted.

## K.4 Posterior Collapse in Leduc Poker

Leduc has chance nodes of two sizes: the initial private deal, with $K = 3 0$ outcomes, and the second-round public card, with $K = 4$ . We measure the statistic of Appendix K.2 at both over 10 seeds and 30,000 gradient steps, for the three configurations the paper uses: the flat latent, a single categorical with 34 categories (Appendix J); the factored latent, with 36 configurations; and the one-way variant, which keeps the factored latent but sets $\beta ^ { \mathrm { e n c } } = 0$ as Theorem 3 prescribes.

At the initial node $( K = 3 0 )$ the flat latent collapses and the other two do not (Figure 7). It settles near 1.7, almost exactly half of the $\log 3 0 = \bar { 3 . 4 0 1 }$ baseline, so it cannot distinguish all thirty outcomes; since it also struggles at deterministic nodes (Appendix K.5), the difficulty lies with the flat stochastic state rather than with this node. The factored latent recovers most of the gap at 2.9, leaving a residual collapse, and the one-way variant sits just above the baseline at 3.45, which follows from its 36 configurations: a prior spread over all of them exceeds log 30.

At the public node $( K = 4 )$ the ordering inverts (Figure 8). The flat latent sits on the log $4 = 1 . 3 8 6$ baseline, the factored latent just above at 1.45, and the one-way variant far above at 2.4, with a visibly wider spread across seeds. Its prior is therefore spread over many more configurations than the node has outcomes, which we attribute to the prior being unable to track the moving posterior — the mechanism Section 5.1 holds responsible for the one-way variant’s poor performance.

No configuration is faithful at both nodes: the flat latent is calibrated at small K and collapses at large, the factored latent is close at both and exact at neither, and dropping the posterior regularizer trades collapse for drift. This is the tension of Section 7, and why the structural error at the deeper chance node survives every variant we ran.

![](images/78e9ef1c7ad0e7944a9e3ba6c2e430801198821576404b66104572dab0143b99.jpg)

![](images/783dc6c89a05126bf17cc9fde39fcb9d36f47d2a503881798a64b3de8ffe7f8d.jpg)  
Figure 7: Expected ${ \mathrm { K L } } ( q ^ { \theta } \parallel p ^ { \theta } )$ at the $K = 3 0$ private-deal chance node of Leduc Poker, averaged over nodes of that size. Bold curves are means over 10 seeds, faint curves the individual seeds. The flat latent collapses to roughly half the uniform baseline.  
Figure 8: The same statistic at the $K = 4$ publiccard chance nodes of Leduc Poker. Here the flat latent is calibrated and the one-way variant overshoots the baseline by a wide margin, indicating a prior that is not tracking the posterior.

## K.5 Chance-Outcome Calibration in Leduc Poker

The calibration statistics of Appendix K.3 localize the Leduc failure more sharply than the collapse statistic can, because they separate the nodes at which the model must represent a distribution from those at which it must represent a single successor. We measure them for the same three configurations over 10 seeds and 30,000 gradient steps; both statistics are bounded by 1.

At deterministic nodes the factored and one-way variants drive both statistics to zero within roughly 10,000 steps, while the flat latent settles near 0.11 on both (Figures 11 and 12). At chance nodes the three separate: the factored variant reaches an $L ^ { 1 }$ distance of 0.16 with 0.07 of its mass unmatched, the flat latent 0.45 and 0.44, and the one-way variant 0.63 and 0.63 (Figures 9 and 10).

For the one-way and flat variants the two statistics coincide, so no successor is over-weighted and their whole discrepancy is deficit. They reach that signature for different reasons, which the collapse statistic separates: at $\dot { K } = 3 0$ the one-way posterior still resolves every deal whereas the flat one sits at about half of log K (Appendix K.4). The one-way variant therefore represents every outcome and only mis-weights it, diverting two thirds of its prior mass to configurations the game cannot produce, while the flat variant’s deficit includes outcomes it no longer distinguishes at all.

The flat latent is also the only configuration that fails to resolve a deterministic successor. We attribute this to a single one-hot supplying the sequence model with less usable structure than a pair of them, so the factored latent earns its place by the form in which it presents the outcomes rather than by how many it can represent.

The factored variant is the only one whose $L ^ { 1 }$ distance exceeds its unmatched mass, so beyond a small leakage it misallocates probability among the outcomes it does represent. It is nevertheless the best configuration at chance nodes and exact at deterministic ones: the structural error of Section 5.1 is, for this configuration, a misallocation rather than a failure to represent.

![](images/4e9bb0ba60adc7d37532026249e5db28e8658aeaba2c7c5e34d4a1907ef2f793.jpg)

![](images/66edb0a44f16574d19dda7444ed192dc4ea8a68ec3f0b0b255a5b1fd5c67e626.jpg)

Figure 9: $L ^ { 1 }$ distance between the bucketed marginal of the prior and the true outcome distribution at the chance nodes of Leduc Poker. Bold curves are means over 10 seeds, faint curves the individual seeds.  
![](images/ccf1c3c8e4d69cb2173290935da320ab19259c3ca557386b02f4f44f97fe7d25.jpg)  
Figure 11: The same $L ^ { 1 }$ statistic at the deterministic nodes of Leduc Poker, where the true distribution is a Dirac on the single successor. The factored and one-way variants reach zero; the flat latent does not.

Figure 10: Unmatched probability mass at the chance nodes of Leduc Poker: the share of the prior that decodes to no real outcome. The oneway variant leaves roughly 0.63 unmatched, the flat latent 0.44 and the factored latent 0.07.  
![](images/64adf5441478e9bce26964be980e33dc24b25e74f89687057db73afd94a891a0.jpg)  
Figure 12: Unmatched probability mass at the deterministic nodes of Leduc Poker. It vanishes for the factored and one-way variants but plateaus near 0.11 for the flat latent.

## K.6 Rollout Validity in Leduc Poker

Rollout validity (Appendix K.1) removes the teacher forcing of the two statistics above, so errors compound as they do during imagination. We report it for the same three configurations over 10 seeds and 30,000 gradient steps. Leduc rewards are divided by the maximal bet of 13 into [−1, 1] (Appendix J), so reward errors are on that normalized scale.

The deterministic emissions survive the rollout (Figures 15 and 16): terminal flags and legal-action masks reach zero within roughly 1,000 gradient steps for all three configurations and stay there. The Leduc failure is not a loss of track of when the game ends or which actions are available.

Neither observations nor rewards reach zero (Figures 13 and 14), and they disagree on which configuration is best. On observations the one-way variant leads at 0.36, ahead of the factored at 0.54 and the flat at 0.83, all down from an initial 4.1; on rewards the first two reverse, at 0.024 and 0.029, with the flat latent again worst at 0.037. The flat latent is the only one worst on both, consistent with its failure to resolve even deterministic successors (Appendix K.5).

That the one-way variant leads here despite leaving 0.63 of its chance-node mass unmatched follows from what each protocol measures: rollout validity scores whether a generated trajectory is realizable, not the probability with which it is generated, and the one-way defect is in the probabilities. The two protocols are therefore complementary, since a model can score well here while visiting realizable states at incorrect frequencies.

That the factored variant nonetheless trails it on observations, despite being far better calibrated at chance nodes, reflects two different notions of a latent state supporting an outcome. Bucketing (Appendix K.3) credits an outcome whenever some configuration in the prior’s support decodes close to it, whereas a rollout uses the configuration actually sampled. The size of the gap suggests the factored variant’s best supporter for an outcome is considerably better than its typical sampled one.

![](images/988cd741a85d0069c8f3b0df131cf04b0ed1767189b72fcffa24d6336258b77d.jpg)

![](images/cbce2ac1f6b5404de7ab5c5f828f783691564aaac410b675d0adbd2c5573cfd7.jpg)  
Figure 13: Rollout observation error in Leduc Poker, as the mean per-step $L ^ { \infty }$ error of the joint observation. Bold curves are means over 10 seeds, faint curves the individual seeds. No configuration reaches zero.  
Figure 14: Rollout reward error in Leduc Poker, as the mean per-step $L ^ { 1 }$ error on rewards normalized to [−1, 1]. The ordering of the factored and one-way variants is the reverse of the one on observations.

## K.7 Rollout Validity in Goofspiel-5

Goofspiel-5 reveals its point cards in descending order (Appendix F) and therefore contains no chance nodes, so the failure mode of Theorem 1 cannot arise. It serves as the positive control for the Leduc results: it establishes that the architecture can recover the rules of a game exactly. We report rollout validity for NashDreamer[RNaD] over 10 seeds and 30,000 gradient steps.

Every statistic converges. Terminal flags and legal-action masks reach zero error within 1,000 gradient steps and remain there (Figures 19 and 20). The observation error falls from roughly 4.4 to 0.25 over the same 1,000 steps and to near zero by 10,000 (Figure 17). The reward converges last (Figure 18): it reaches roughly 0.04 within 1,000 steps, then descends slowly with individual seeds spiking to 0.1, and approaches zero only in the final third of training.

The reward is also the slowest statistic in Leduc (Appendix K.6), which suggests a property of the reward signal rather than of either game. We attribute it to the reward being terminal: it is zero at every step except the one ending the game, and small in magnitude, so it contributes little to the prediction loss and is resolved last.

![](images/213cb8513ea304f411112c7e976b8fa6f7c6e2774bd4ad14fb31f021dc30bdfa.jpg)  
Figure 15: Rollout terminal-flag error in Leduc Poker. All three configurations reach zero within roughly 1,000 gradient steps and the curves coincide thereafter.

![](images/9e5831c7cb1e934c9daebd7320a2427f931594c6bd5da8c3105aedaf74f18259.jpg)  
Figure 16: Rollout legal-action-mask error in Leduc Poker, which likewise reaches zero for all three configurations and stays there.

![](images/90cfa0bbf48c1df7e208bc1d42448407c2e2c6daa71980313714ef18630ddb70.jpg)

![](images/d3ebc10a3eb0887b7f7a5ffbd880d6c9a56dacee2bea64eed3f7316ad6f5096f.jpg)  
Figure 17: Rollout observation error in Goofspiel-5, as the mean per-step $L ^ { \infty }$ error of the joint observation. Bold curve is the mean over 10 seeds, faint curves the individual seeds.  
Figure 18: Rollout reward error in Goofspiel-5, the slowest of the four to converge, with individual seeds spiking before falling back.

![](images/978675b2a6a1122902726efe5ff3018ca536291e1adab002412204af162d3d1c.jpg)  
Figure 19: Rollout terminal-flag error in Goofspiel-5, at zero within 1,000 gradient steps.

![](images/d175e5d42617cfa0756bf94316f105393ed35ee11c09066485a8610b4604b947.jpg)  
Figure 20: Rollout legal-action-mask error in Goofspiel-5, likewise at zero within 1,000 gradient steps.

## L Centralized versus Decentralized World Models

Section 3 argues that a decentralized world model faces identifiability barriers rather than merely an accuracy penalty. This appendix tests that claim directly by ablating the one property under dispute: we train a decentralized MARSSM in which each player’s world model conditions only on that player’s own actions and observations, leaving the architecture, the objective and the optimizer untouched.

Setup. The domain is perturbed Rock-Paper-Scissors (Appendix F) a matrix game with non-uniform equilibrium. Both variants use RNaD as the actor-critic and run for 1,000 gradient steps without switching the regularization policy, over 10 seeds. At each evaluated checkpoint we collect 4 batches of 2,058 rollouts and score them by the protocol of Appendix K.1.

The decentralized model learns everything except the reward. Figures 21–24 report the four error statistics. On the three quantities that do not depend on the opponent’s action, the two variants are essentially the same model: terminal flags and legal-action masks reach zero error within 100 gradient steps for both, and the joint observation error falls to zero for both, the decentralized variant simply taking longer to get there, around 600 steps against roughly 200 for the centralized one.

The reward separates them completely. The centralized model error in predicting the reward actually increases with training, due to it being unidentifiable for the ego-centric model.

![](images/bced5daa3dfc98e59eb86657984067c06b32be1601811cdbf8453e0b438c982c.jpg)

![](images/113d90e6f141b49813e8d7ed26c15c28ec7f852ca10c1f62120bb5c88bb6e379.jpg)  
Figure 21: Terminal-flag error over rollouts in perturbed Rock-Paper-Scissors. Bold curves are means over the 10 seeds, faint curves the individual seeds. The two variants coincide.  
Figure 22: Legal-action-mask error in perturbed Rock-Paper-Scissors, on the same convention. The two variants again coincide, both reaching zero within 100 gradient steps.

![](images/8e21b4b4ea2c8083fd9db17bb40800b2b7557f613040e2f531c716effa07f576.jpg)  
Figure 23: Joint-observation error in perturbed Rock-Paper-Scissors. Both variants converge to zero; the decentralized model is slower but not obstructed.

![](images/8d8ec3ca6481f844eb92c6322408bdfb980afe2e508079f59954875026a94971.jpg)  
Figure 24: Reward error in perturbed Rock-Paper-Scissors. The centralized model converges to zero, whereas the decentralized model diverges to approximately 1.0: without observing $a _ { 2 }$ the reward is unidentifiable from its inputs.

## M Single-Agent Validation

We validated NashDreamer against the reference DreamerV3 implementation on PCM with 3 cards. To ensure a fair comparison, we adapted the reference baseline to accommodate the specific constraints of the environment. Standard DreamerV3 utilizes a burn-in period to approximate the recurrent state from sampled trajectory chunks. However, in a strictly sequential game like PCM, a short burn-in period discards critical historical context, causing structure learning to fail. To resolve this, we configured the reference implementation to sample full trajectories starting from the root node, exactly mirroring the NashDreamer setup. Furthermore, rather than relying on standard reward penalties and episode truncation for invalid moves, we explicitly masked illegal actions during reference sampling.

Under these matched conditions, both models successfully learned the optimal policy within 1000 gradient steps with a minibatch size of 32. However, deeper analysis via Latent Error Trees (LE-Trees; see Appendix I for the visualization methodology and full figures) revealed a NashDreamer structural advantage.

NashDreamer’s LE-Tree was an exact match with the true reachable dynamics (Figure 26), while DreamerV3 retained artificial stochasticity in its transitions (Figure 25) (exactly the aforementioned branch duplication). We attribute this structural precision to two factors: (1) the additional legal action prediction signal, and (2) the centralized unifier loss, which penalizes unpredictable variance in the stochastic component. Crucially, as established in our theoretical analysis, without the unifier loss acting as a structural “tiebreaker,” the perfectly isomorphic model and a model exhibiting benign branch duplication would incur the exact same dynamics loss.

![](images/e3c5452899f6ffa8aa4619b7b5797481fe9457eb9e89677d58fae372d6689383.jpg)  
Figure 25: DreamerV3 LE-Tree on PCM 3 (1000 gradient steps). Predictions are accurate along the optimal path, but the model fails to learn that the environment is deterministic.

![](images/2f67f1a0c5790a7320344acd04dee9cf35818e6204c554a5adba02d7da4c618a.jpg)  
Figure 26: NashDreamer LE-Tree on PCM 3 (1000 gradient steps). The learned model is an exact match of the true reachable dynamics within the 0.2 accuracy threshold.

## N REINFORCE vs. RNaD

Since REINFORCE has no convergence guarantees in two-player zero-sum IIGs and statically tuning it is not sound in the context of changing world model, we opted not to use it as our policy optimization algorithm. Nevertheless, we test in in Rock-Paper-Scissors (RPS) as a standalone algorithm, and in Goofspiel 3 descending wrapped in NashDreamer to validate our choice.

## N.1 REINFORCE in RPS

Since every decision node in a POSG 2 is effectively a matrix game, we tested model-free RE-INFORCE without neural network function approximation in the canonical matrix game Rockpaper-scissors (RPS). We evaluated a low entropy bonus of 3e − 4, characteristic of single-agent environments and a higher entropy bonus of 0.2, typical of multi-agent environments.

We tested two distinct initializations: a uniform initialization, which initializes the algorithm in the equilibrium of the game, and a pure (Rock, Rock) initialization. The results are presented for 10 distinct seeds, with a 1-σ bar interval.

The algorithm always diverged away from equilibrium, even with the higher entropy bonus, which only slowed the divergence (Figure 27). Under the deterministic strategy initialization, low entropy bonus causes the algorithm to remain trapped in that determinism (Figure 28).

![](images/5da3755b4bc829734ef6a1126862c09554bc51b534b64c833bb1c58d4f1b0e31.jpg)

![](images/ee5aeaec00ffbdd86bc55ac3a64b66ca5efd19390fb32d6028af4cfec5382a4e.jpg)  
Figure 28: REINFORCE in RPS with pure initialization. Low entropy bonus is trapped in determinism.

Figure 27: REINFORCE in RPS with uniform initialization. Even at equilibrium, low entropy bonus causes divergence; high bonus slows but does not prevent it.  
![](images/40d1cb883d3e80696df5390380b730ded7205b0fce999f048fadbc367b146297.jpg)  
Figure 29: NashConv on Goofspiel 3. RNaD converges to the pure NE; REINFORCE with $\eta = 0 . 2$ entropy bonus cannot.

## N.2 Goofspiel 3

To evaluate REINFORCE as a component of the full NashDreamer architecture, we perform experiments in Goofspiel 3 (which has a pure equilibrium).

We test a reimplementation of the DreamerV3 version of REINFORCE [14], with identical hyperparameters, except the entropy bonus, which we set to 0.2 to match the RNaD regularization strength.

RNaD successfully converged to the pure NE, while REINFORCE, burdened by the entropy exploration bonus, was unable to collapse into the required pure strategies (Figure 29). Analysis of the LE-Trees (Figure 30 for RNaD; REINFORCE is contained in the supplementary material due to size) confirmed that REINFORCE spread probability mass across all actions rather than concentrating on the equilibrium.

## O Learning from partial observations

In our primary experiments (Section 5), we provide both NashDreamer and the standalone RNaD baseline with pre-aggregated infoset representation vectors. This design choice was made to isolate the primary experimental variable: the sample efficiency of the centralized world model versus an off-policy replay buffer and prevent confounding the difference by differing input representations.

However, the MARSSM architecture is designed to handle aggregating sequences of raw, partial observations via the decentralized recurrent infoset model. To empirically validate this capability, we conducted an ablation on the Imperfect Information Goofspiel-5 environment.

![](images/793ee481064c8392566e2f35c15ed46ea8ec1875a5205540ccd7ca5c247a6ffc.jpg)  
Figure 30: NashDreamer with RNaD LE-Tree on Goofspiel 3 (1000 gradient steps). The model learns both the pure equilibrium and accurate predictions along the equilibrium path.

In this experiment, we tested NashDreamer both with pre-aggregated infoset input and with only the observation in the current state (e.g., the current point card and win/loss/tie information). As demonstrated in Figure 31, the difference in performance is marginal, validating the capability of NashDreamer to aggregate per-player information and model the environment dynamics simultaneously.

![](images/1da5912ed13a4f238f18568c5db22f32fc997e59210674e6110a605a581c8406.jpg)  
Figure 31: Comparison of NashConv in Imperfect Information Goofspiel 5 when using pre-aggregated infoset representation versus latent infoset modelling from partial observations. The model demonstrates the capability to model accurate latent infosets.

## P NashConv Against Total Actor-Critic Steps

The NashConv curves of Section 5.1 are plotted against real environment interaction, because that is the quantity our sample-efficiency claim concerns. For completeness, we report here the same runs plotted against the total number of transitions consumed by the actor-critic, counting real, imagined and replayed transitions identically. Figure 32 gives this view for Imperfect Information Goofspiel-5, Leduc Poker and Phantom Tic-Tac-Toe. In every panel, the horizontal axis is logarithmic, evaluated checkpoints are marked on the curves, and the plotted range begins at the end of the world-model warm-up, before which no imagined data exists. The Leduc runs use the configuration that performed best empirically: the factored latent stochastic state of Appendix J with $\beta ^ { \mathrm { d y n } } = 0 . 1$ and $\beta ^ { \mathrm { { e n c } } } = 0 . 0 1$

Two observations are worth drawing from these plots. First, the model-based runs consume roughly an order of magnitude more total transitions than model-free RNaD to reach a comparable NashConv. This is expected, and it is precisely why the main text reports real interaction: imagined transitions are cheap in the regime we target, so counting them equally with real ones measures a different quantity than the one we claim to improve. Second, the comparison that is meaningful on this axis is against RNaD with replay, which was deliberately configured to consume more total transitions than NashDreamer generates by imagination (Appendix Q). In Goofspiel-5 and in Phantom Tic-Tac-Toe alike, NashDreamer[RNaD] reaches a lower NashConv than the replay baseline at comparable total step counts, so the improvement is not attributable to step volume alone. In Leduc the Nash-Dreamer[RNaD] curve additionally exhibits the late-training instability reported in Section 5.1 and localized in Appendix K.

How to read these plots. Every run is evaluated at 30 checkpoints spaced 1,000 gradient steps apart, and each evaluated checkpoint is marked on its curve. Because the horizontal axis counts transitions consumed rather than gradient steps, an equal number of gradient steps maps to a different horizontal position for each method, so the curves neither begin nor end together. This is most visible for the replay-augmented baseline, whose first checkpoint lies well to the right of the others because it already consumes replayed transitions during the warm-up period; the model-based runs separate from their model-free counterparts in the same manner once imagination begins. At the first checkpoint, by contrast, NashDreamer[RNaD] and model-free RNaD coincide, as they must: up to the end of the warm-up the two consume exactly the same real transitions, and the same holds for the MMD pair.

![](images/6fd06819f22c7258b4e04989abb1645ff86c7c3c0f0773abc555901c1116f7dd.jpg)  
Figure 32: NashConv against the total number of transitions consumed by the actor-critic, counting real, imagined and replayed transitions identically, in Imperfect Information Goofspiel-5 (left), Leduc Poker (middle) and Phantom Tic-Tac-Toe (right). Markers indicate evaluated checkpoints. Bold curves are means over the 10 training seeds, faint curves the individual seeds. The Leduc runs use the factored latent stochastic state with $\mathrm { \bar { \beta } ^ { d y n } = 0 . 1 }$ and $\beta ^ { \mathrm { e n c } } = 0 . 0 1$

## Q Head-to-Head Evaluation

We also report statistics of head-to-head play on larger games, using the same checkpoints as reported in Section 5.2, which were obtained by playing 1M games per seed pairing, in each seat (so 20M games per ego-opponent pair in general). The results are averaged across the seats.

Battleship 5 × 5. NashDreamer[RNaD] wins against every other method at all three checkpoints (Table 5), but its margin shrinks monotonically as interactions accumulate: against model-free RNaD it falls from 66.7% at 10K to 62.6% at 20K and 56.7% at 30K, and against RNaD with replay from 71.9% to 66.6% to 61.5%. NashDreamer[MMD] shows the same pattern against model-free MMD, 61.5% falling to 56.5%. This is the early-to-mid-training scope we claim throughout: the world model buys an advantage that the model-free baseline gradually closes given enough environment interaction. Between the two RNaD baselines, the one with replay is the weaker at every checkpoint, losing to plain RNaD in all three matrices, which is consistent with the variance argument of Appendix E.2.

Stochastic Goofspiel (13 cards). Here head-to-head play does not separate a NashDreamer variant from its model-free counterpart at all (Table 6). NashDreamer[RNaD] and RNaD are at parity across all three checkpoints, the model-based side winning 50.3%, 49.9% and 49.7% of matches, and NashDreamer[MMD] and MMD likewise (49.0%, 48.6%, 48.9%). RNaD consistently beats MMD across the tested checkpoints, and RNaD with replay is sligthly weaker than without it.

In Goofspiel-13, head-to-head play attributes no benefit at all to the world model: Nash-Dreamer[RNaD] and RNaD are indistinguishable in direct play, and each fares the same against the rest of the pool. Under best-response search the two are far apart, 0.168 against 0.356 at 30K.

<table><tr><td rowspan="2">Method</td><td colspan="5">Win rate (%) vs.</td></tr><tr><td>MMD</td><td>ND[MMD] ND[RNaD]</td><td></td><td>RNaD</td><td>RNaD w/ replay</td></tr><tr><td colspan="6">10K gradient steps</td></tr><tr><td>MMD</td><td></td><td> $3 8 . 5 \pm 2 . 6$ </td><td> $2 5 . 6 \pm 2 . 5$ </td><td> $4 1 . 4 \pm 2 . 7$ </td><td> $4 8 . 3 \pm 2 . 3$ </td></tr><tr><td>ND[MMD]</td><td> $6 1 . 5 \pm 2 . 6$ </td><td></td><td> $3 5 . 7 \pm 2 . 2$ </td><td> $5 2 . 7 \pm 2 . 2$ </td><td> $5 8 . 8 \pm 1 . 1$ </td></tr><tr><td>ND[RNaD]</td><td> $7 4 . 4 \pm 2 . 5$ </td><td> $6 4 . 3 \pm 2 . 2$ </td><td></td><td> $6 6 . 7 \pm 2 . 1$ </td><td> $7 1 . 9 \pm 1 . 6$ </td></tr><tr><td>RNaD</td><td> $5 8 . 6 \pm 2 . 7$ </td><td> $4 7 . 3 \pm 2 . 2$ </td><td> $3 3 . 3 \pm 2 . 1$ </td><td></td><td> $5 6 . 9 \pm 1 . 7$ </td></tr><tr><td>RNaD w/ replay</td><td> $5 1 . 7 \pm 2 . 3$ </td><td> $4 1 . 2 \pm 1 . 1$ </td><td> $2 8 . 1 \pm 1 . 6$ </td><td> $4 3 . 1 \pm 1 . 7$ </td><td></td></tr><tr><td colspan="6">20K gradient steps</td></tr><tr><td>MMD</td><td></td><td> $4 0 . 7 \pm 1 . 8$ </td><td> $2 9 . 0 \pm 1 . 6$ </td><td> $4 2 . 0 \pm 2 . 0$ </td><td> $4 5 . 1 \pm 1 . 8$ </td></tr><tr><td>ND[MMD]</td><td> $5 9 . 3 \pm { 1 . 8 }$ </td><td></td><td> $3 7 . 3 \pm 1 . 5$ </td><td> $5 1 . 0 \pm 2 . 2$ </td><td> $5 3 . 4 \pm 1 . 4$ </td></tr><tr><td>ND[RNaD]</td><td> $7 1 . 0 \pm 1 . 6$ </td><td> $6 2 . 7 \pm 1 . 5$ </td><td></td><td> $6 2 . 6 \pm 1 . 2$ </td><td> $6 6 . 6 \pm 0 . 9$ </td></tr><tr><td>RNaD</td><td> $5 8 . 0 \pm 2 . 0$ </td><td> $4 9 . 0 \pm 2 . 2$ </td><td> $3 7 . 4 \pm 1 . 2$ </td><td></td><td> $5 5 . 0 \pm 1 . 2$ </td></tr><tr><td>RNaD w/ replay</td><td> $5 4 . 9 \pm 1 . 8$ </td><td> $4 6 . 6 \pm 1 . 4$ </td><td> $3 3 . 4 \pm 0 . 9$ </td><td> $4 5 . 0 \pm 1 . 2$ </td><td></td></tr><tr><td colspan="6">30K gradient steps</td></tr><tr><td>MMD</td><td></td><td> $4 3 . 5 \pm 2 . 0$ </td><td> $3 2 . 2 \pm 1 . 7$ </td><td> $3 9 . 3 \pm 1 . 9$ </td><td> $4 3 . 2 \pm 1 . 7$ </td></tr><tr><td>ND[MMD]</td><td> $5 6 . 5 \pm 2 . 0$ </td><td></td><td> $3 8 . 5 \pm { 1 . 8 }$ </td><td> $4 6 . 3 \pm 2 . 5$ </td><td> $4 9 . 9 \pm 1 . 9$ </td></tr><tr><td>ND[RNaD]</td><td> $6 7 . 8 \pm 1 . 7$ </td><td> $6 1 . 5 \pm { 1 . 8 }$ </td><td></td><td> $5 6 . 7 \pm 2 . 3$ </td><td> $6 1 . 5 \pm 0 . 8$ </td></tr><tr><td>RNaD</td><td> $6 0 . 7 \pm 1 . 9$ </td><td> $5 3 . 7 \pm 2 . 5$ </td><td> $4 3 . 3 \pm 2 . 3$ </td><td></td><td> $5 4 . 9 \pm 1 . 9$ </td></tr><tr><td>RNaD w/ replay</td><td> $5 6 . 8 \pm 1 . 7$ </td><td> $5 0 . 1 \pm 1 . 9$ </td><td> $3 8 . 5 \pm 0 . 8$ </td><td> $4 5 . 1 \pm 1 . 9$ </td><td></td></tr></table>

Table 5: Head-to-head play of 1M games for each seed pairing and seat in Battleship 5 × 5: win rate (%) of the row method against the column method, mean ±2σ over training seeds. ND denotes NashDreamer with the given actor-critic inside imagination.

In Battleship the disagreement runs the other way. NashDreamer[RNaD] wins the cross-play table outright, yet at 30K it is the second most exploitable of the five methods (1.169), whereas MMD, which loses to every other method in direct play, is harder to exploit (1.022), and NashDreamer[MMD] is the hardest of all (0.817).

Neither observation contradicts the results of Section 5.2, since head-to-head measures performance across a particular opponent pool, rather than how hard the policy is to exploit overall. We therefore report exact NashConv wherever it is tractable (Section 5.1), budgeted approximate exploitability where it is not (Section 5.2), and treat the tables above as evidence about pool performance alone.

<table><tr><td rowspan="2">Method</td><td colspan="5">Win rate (%) vs.</td></tr><tr><td>MMD</td><td>ND[MMD]</td><td>ND[RNaD]</td><td>RNaD</td><td>RNaD w/ replay</td></tr><tr><td colspan="6">10K gradient steps</td></tr><tr><td>MMD</td><td></td><td> $5 1 . 0 \pm 1 . 8$ </td><td> $3 7 . 4 \pm 1 . 8$ </td><td> $3 6 . 6 \pm 1 . 9$ </td><td> $4 2 . 5 \pm 1 . 6$ </td></tr><tr><td>ND[MMD]</td><td> $4 9 . 0 \pm 1 . 8$ </td><td></td><td> $3 6 . 5 \pm 2 . 0$ </td><td> $3 5 . 2 \pm 1 . 5$ </td><td> $4 1 . 3 \pm 1 . 7$ </td></tr><tr><td>ND[RNaD]</td><td> $6 2 . 6 \pm 1 . 8$ </td><td> $6 3 . 5 \pm 2 . 0$ </td><td></td><td> $5 0 . 3 \pm 1 . 6$ </td><td> $5 6 . 4 \pm 1 . 0$ </td></tr><tr><td>RNaD</td><td> $6 3 . 4 \pm 1 . 9$ </td><td> $6 4 . 8 \pm 1 . 5$ </td><td> $4 9 . 7 \pm 1 . 6$ </td><td></td><td> $5 6 . 4 \pm 2 . 2$ </td></tr><tr><td>RNaD w/ replay</td><td> $5 7 . 5 \pm 1 . 6$ </td><td> $5 8 . 7 \pm 1 . 7$ </td><td> $4 3 . 6 \pm 1 . 0$ </td><td> $4 3 . 6 \pm 2 . 2$ </td><td></td></tr><tr><td colspan="6">20K gradient steps</td></tr><tr><td>MMD</td><td></td><td> $5 1 . 4 \pm 3 . 9$ </td><td> $3 8 . 4 \pm 1 . 7$ </td><td> $3 7 . 2 \pm 2 . 4$ </td><td> $4 1 . 6 \pm 1 . 6$ </td></tr><tr><td>ND[MMD]</td><td> $4 8 . 6 \pm 3 . 9$ </td><td></td><td> $3 7 . 4 \pm 2 . 5$ </td><td> $3 6 . 0 \pm 1 . 9$ </td><td> $4 0 . 4 \pm 2 . 4$ </td></tr><tr><td>ND[RNaD]</td><td> $6 1 . 6 \pm 1 . 7$ </td><td> $6 2 . 6 \pm 2 . 5$ </td><td></td><td> $4 9 . 9 \pm 1 . 4$ </td><td> $5 4 . 9 \pm 1 . 3$ </td></tr><tr><td>RNaD</td><td> $6 2 . 8 \pm 2 . 4$ </td><td> $6 4 . 0 \pm 1 . 9$ </td><td> $5 0 . 1 \pm 1 . 4$ </td><td></td><td> $5 5 . 7 \pm 2 . 4$ </td></tr><tr><td>RNaD w/ replay</td><td> $5 8 . 4 \pm 1 . 6 $ </td><td> $5 9 . 6 \pm 2 . 4$ </td><td> $4 5 . 1 \pm 1 . 3$ </td><td> $4 4 . 3 \pm 2 . 4$ </td><td></td></tr><tr><td colspan="6">30K gradient steps</td></tr><tr><td>MMD</td><td></td><td> $5 1 . 1 \pm 1 . 4$ </td><td> $3 7 . 9 \pm 2 . 3$ </td><td> $3 7 . 1 \pm 1 . 6$ </td><td> $4 1 . 5 \pm 1 . 5$ </td></tr><tr><td>ND[MMD]</td><td> $4 8 . 9 \pm 1 . 4$ </td><td></td><td> $3 7 . 1 \pm 1 . 8$ </td><td> $3 6 . 1 \pm 1 . 9$ </td><td> $4 0 . 6 \pm 1 . 7$ </td></tr><tr><td>ND[RNaD]</td><td> $6 2 . 1 \pm 2 . 3$ </td><td> $6 2 . 9 \pm 1 . 8$ </td><td></td><td> $4 9 . 7 \pm 1 . 3$ </td><td> $5 4 . 8 \pm 1 . 4$ </td></tr><tr><td>RNaD</td><td> $6 2 . 9 \pm 1 . 6$ </td><td> $6 3 . 9 \pm 1 . 9$ </td><td> $5 0 . 3 \pm 1 . 3$ </td><td></td><td> $5 5 . 8 \pm 1 . 4$ </td></tr><tr><td>RNaD w/ replay</td><td> $5 8 . 5 \pm 1 . 5$ </td><td> $5 9 . 4 \pm 1 . 7$ </td><td> $4 5 . 2 \pm 1 . 4$ </td><td> $4 4 . 2 \pm 1 . 4$ </td><td></td></tr></table>

Table 6: Head-to-head play of 1M games for each seed pairing and seat in stochastic 13-card Imperfect Information Goofspiel: win rate (%) of the row method against the column method, mean $\pm 2 \sigma$ over training seeds. ND denotes NashDreamer with the given actor-critic inside imagination.

## R World-Model Warm-Up Ablation

The warm-up is a stabilizing mechanism rather than a theoretical requirement (Appendix C). It supplements the DreamerV3 practice of zero-initializing the reward and critic heads (Appendix G.3), compensating for the additional non-stationarity that the opponent introduces in the multi-agent setting. We sweep its length over {0, 250, 500, 1000, 2000} gradient steps in Leduc Poker, 1000 being the value used in every other Leduc experiment, running both actor-critics inside imagination at each setting with model-free RNaD and MMD for reference (Figure 33). Runs are 10,000 gradient steps, since the parameter acts only early in training; the remaining configuration is unchanged.

NashDreamer[RNaD] is invariant to the setting: no value, zero included, separates from the others by more than the spread across training seeds. Nor does a longer warm-up improve the asymptote, as expected given that the Leduc model remains badly learned at every setting (Appendix K). The warm-up is therefore a stability parameter, not a lever on final policy quality.

For MMD, warm-ups of 0 and 250 steps accelerate early convergence relative both to the longer settings and to model-free MMD, while 500 and above are indistinguishable from the model-free baseline. This supports the explanation in Section 5.1 for NashDreamer[MMD]’s lack of gain on the small games: MMD converges inside the warm-up window, leaving imagination nothing to accelerate.

![](images/383d380080e232fbf9d15304b264917962bdd6edf46144f17aa5d4308ae25690.jpg)  
Figure 33: World-model warm-up sweep in Leduc Poker over {0, 250, 500, 1000, 2000} gradient steps, applied to both actor-critics inside imagination, with model-free RNaD and MMD for reference. Legend entries are labelled by the warm-up length in gradient steps. Bold curves are means over training seeds, faint curves the individual seeds. NashDreamer[RNaD] is largely unaffected by the setting, whereas NashDreamer[MMD] separates from model-free MMD in the early phase only at the two shortest warm-ups.