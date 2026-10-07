# Independent Multi-Agent Reinforcement Learning with Counterfactual Semantic-Social World Models

Fernando Martinez, Tao Li†, Yingdong Lu, and Juntao Chen

Abstract—Fully decentralized multi-agent reinforcement learning (MARL), also referred to as independent learning, requires each agent to learn and act using only its local information and experience, without a centralized critic or inter-agent communication. Such a stringent information structure renders the conventional reward signal ambiguous. A poor return may result from an ineffective ego action, an incompatible teammate response, or an effective opponent response, yet scalar rewards alone do not reveal which explanation is responsible. We argue that agents can learn more effectively by prospectively comparing the consequences of candidate actions rather than diagnosing failures only from realized returns. Based on this intuition, we introduce CASTLE (Counterfactual Action-conditioned Semantic Tokens for Local Execution in Decentralized MARL), an offline-training, online-in-context guidance framework with two complementary world models. A Local Dynamics World Model, offline pre-trained over agents’ local trajectories, summarizes the agent’s local trajectory dynamics and partial observability, while a Semantic-Social World Model predicts compact short-horizon task and social consequences for each candidate ego action. The latter is trained from counterfactual simulator rollouts that expose plausible teammate and opponent responses to alternative actions taken from the same logged rollout state. During online learning and execution, both world models remain frozen and are queried by agents using only locally available information. Their prediction logits provide in-context guidance to an independent PPO policy. Across 30 matched seeds on Tag, Spread, and Adversary in the benchmark multi-particle environments, our proposed CASTLE achieves the highest mean final score among the evaluated methods, exceeding the strongest baseline on each task by 10.67, 6.46, and 0.33 normalized points, respectively.

Impact Statement—Agents can act independently while still depending on another to complete a task. This is relevant for autonomous teams that cannot rely on communication: each agent learns from its observations, but the effect of an action can also depend on how other agents act. This work studies whether world models trained offline can help in this setting. These models provide information on environmental dynamics and the potential short-term consequences of each available action. In simulated tasks involving cooperation and competition, our method outperformed the decentralized baselines evaluated without sharing observations or weight updates during online learning. These results indicate that prior experience can promote independent learning and may be useful for autonomous teams that cannot rely on or utilize communication.

Index Terms—Multi-agent reinforcement learning, independent learning, world models, counterfactual rollout

## I. INTRODUCTION

Multi-agent reinforcement learning (MARL) is a general learning paradigm for cooperative and competitive decision making. A key and natural aspect of MARL is that each agent acquires non-stationary experience as other agents learn to improve their policies concurrently, posing a challenge for the design and analysis of learning algorithms. Centralized training with decentralized execution (CTDE) mitigates this challenge by allowing a centralized critic or model to access the global state and joint actions during learning [1]. An alternative set of methods focuses on learning to communicate with another by exchanging observations, latent states, or optimization variables [2]–[4]. Both of these strategies have proven successful, but neither is valid if agents must learn solely from their private local experience and then act entirely on their own. The corresponding learning paradigm is referred to as fully decentralized MARL or independent learning [5].

Since fully decentralized MARL enforces more challenging information structures [6], the study of agents’ independent learning dynamics, both empirically, such as Independent PPO (IPPO) [7] and theoretically, such as independent policy gradient [8], has been a recurring theme in MARL. Recent explorations include adding value transformations, decentralized policy regularization, or centralized context obtained from local observation histories [9]–[11]. However, a local trajectory does not distinguish between world dynamics and the unknown policies of other agents. An action may fail because it was poor, because a teammate chose a conflicting action, or because an opponent was coordinated. Learning these differences solely from returns can be slow and brittle.

Since the reward signal is retrospective and tied to a single realized consequence, the independent learning agents cannot diagnose their past failures and improve future learning. Inspired by recent developments in world models [12], this work proposes equipping each independent agent with prospective, action-conditioned predictions from world models that incorporate latent predictive structures that compress history and predict the consequences of potential actions before they occur. Our intuition is that comparing predicted futures across different ego actions provides the agent with counterfactual evidence beyond what scalar reward signals can provide.

Compared with existing multi-agent world models that commonly predict a centralized global state, aggregate local models centrally, or exchange predictions through a shared information cache [13]–[16], our proposed method achieves fully decentralized operations for both agent policy execution and world model prediction. Hence, our work applies to cooperative, competitive, and mixed-cooperative-competitive scenarios.

Specifically, we introduce CASTLE (Counterfactual Action-conditioned Semantic Tokens for Local Execution in Decentralized MARL), an offline-training, online-guidance framework for fully decentralized MARL. In offline, each agent learns two complementary world models from offline datasets. A Local Dynamics World Model (LDWM) maps the agent’s local history to a predictive representation of its own trajectory, capturing local dynamics and partial observability. A Semantic-Social World Model (SSWM) takes the local history, along with each candidate ego action, and outputs a compact token representing that action’s predicted shorthorizon task and social consequences. The SSWM learns these predictions from counterfactual outcomes generated by applying alternative ego actions from the same logged rollout state. Because these alternatives share a common context, they reveal how changes in ego action affect its future interactions with teammates and opponents, rather than requiring the agent to infer such differences from unrelated scalar returns. During the online stage, both world models are frozen and queried using only locally available information: the LDWM supplies predictive context to the policy, while the SSWM provides action-specific guidance to the agent’s policy learning.

Our contributions are as follows:

• We introduce CASTLE, a dual-world-model framework for fully decentralized MARL. A Local Dynamics World Model (LDWM) summarizes dynamics and partial observability, while an action-conditioned SSWM learns counterfactual tokens that encode social consequences.

• We develop an offline-training, online-guidance procedure that incorporates frozen LDWM context and transient SSWM consequence guidance into conventional independent MARL.

• We evaluate CASTLE using IPPO over 30 matched seeds on three cooperative and competitive MPE tasks under both in-domain and held-out pretraining. CASTLE surpasses the strongest fully decentralized baseline on Tag, Spread, and Adversary by 10.67, 6.46, and 0.33, respectively.

The rest of the paper is organized as follows. Section II introduces MARL basics and notations. Section III presents the proposed CASTLE framework, followed experimental evaluation in Section IV. In Section V, we position our CASTLE framework within the literature on decentralized MARL (multi-agent) world models and in-context RL. Finally, VI concludes the paper.

## II. PRELIMINARY

We model the multi-agent environment as a partially observable Markov game [17]:

$$
\mathcal { G } = \langle \mathcal { N } , \mathcal { S } , \{ \mathcal { O } ^ { i } , \mathcal { A } ^ { i } \} _ { i \in \mathcal { N } } , T , \{ P ^ { i } , r ^ { i } \} _ { i \in \mathcal { N } } , \rho , \gamma \rangle ,
$$

where $\mathcal { N } = \{ 1 , \ldots , n \}$ is the set of agents. S denotes the state space and $\mathcal { O } ^ { i }$ and $\mathcal { A } ^ { i }$ denote the observation space and action space of agent i respectively. We assume that each $\mathcal { A } ^ { i }$ is finite and discrete. Given joint action $a _ { t } = ( a _ { t } ^ { 1 } , \ldots , a _ { t } ^ { n } ) $

$T ( s _ { t + 1 } \mid s _ { t } , a _ { t } )$ defines the state transition, $P ^ { i } ( o _ { t } ^ { i } \mid s _ { t } )$ defines the local observation kernel, and $r ^ { i } ( s _ { t } , a _ { t } )$ is the reward for agent i. The initial-state distribution is denoted by $\rho ,$ and $\gamma \in$ $[ 0 , 1 )$ is the discount factor. Agent i selects actions based on its local history $\tau _ { t } ^ { i } = ( o _ { 0 } ^ { i } , a _ { 0 } ^ { i } , r _ { 0 } ^ { i } , \ldots , a _ { t - 1 } ^ { i } , r _ { t - 1 } ^ { i } , o _ { t } ^ { i } )$ using policy $\pi _ { \theta _ { i } } ^ { i } ( a _ { t } ^ { i } \mid \tau _ { t } ^ { i } ) . \tau _ { t } = ( \bar { \tau _ { t } ^ { 1 } } , . . . , \bar { \tau _ { t } ^ { n } } )$ and $\theta = ( \theta _ { 1 } , \ldots , \theta _ { n } )$ denote the tuple for all local histories collected and policy parameters respectively. For compactness, let $o _ { t } = ( o _ { t } ^ { 1 } , \ldots , o _ { t } ^ { n } )$ and define the joint observation kernel as $\begin{array} { r } { P ( o _ { t } \mid s _ { t } ) = \prod _ { i \in \mathcal { N } } P ^ { i } ( o _ { t } ^ { i } \mid s _ { t } ) } \end{array}$ Because each agent samples its action from its own local policy, the induced joint action distribution and global trajectory distribution are

$$
\begin{array} { l } { { \displaystyle p _ { \theta } ( a _ { t } \mid \tau _ { t } ) = \prod _ { i \in \mathcal { N } } \pi _ { \theta _ { i } } ^ { i } ( a _ { t } ^ { i } \mid \tau _ { t } ^ { i } ) , } } \\ { { \displaystyle q _ { \theta } ( \xi ) = \rho ( s _ { 0 } ) \prod _ { t = 0 } ^ { \infty } [ P ( o _ { t } \mid s _ { t } ) p _ { \theta } ( a _ { t } \mid \tau _ { t } ) T ( s _ { t + 1 } \mid s _ { t } , a _ { t } ) ] . } } \end{array}
$$

where ${ \boldsymbol { \xi } } = ( s _ { 0 } , o _ { 0 } , a _ { 0 } , s _ { 1 } , o _ { 1 } , a _ { 1 } , \dots )$ denotes the global trajectory. Although q<sub>θ</sub> depends on the policies of all agents, agent i controls only its own parameters $\theta _ { i }$ and seeks to maximize $J ^ { i } ( \theta ) = \mathbb { E } _ { \xi \sim q _ { \theta } } \big [ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } \bar { r } _ { t } ^ { i } \big ]$ using only its local trajectory. The global trajectory $\xi$ and its distribution $q _ { \theta }$ are analytical objects and are not available to the agents. During online learning and execution, each agent receives neither the global state, other agents’ private histories, inter-agent messages, nor gradients or parameters from others.

Independent PPO. Our online optimizer relies on independent PPO (IPPO) [18]. For agent i, let

$$
w _ { t } ^ { i } ( \theta _ { i } ) = \frac { \pi _ { \theta _ { i } } ^ { i } ( a _ { t } ^ { i } \mid \tau _ { t } ^ { i } ) } { \pi _ { \theta _ { i } ^ { \mathrm { o l d } } } ^ { i } ( a _ { t } ^ { i } \mid \tau _ { t } ^ { i } ) }
$$

be the policy ratio and $\widehat { A } _ { t } ^ { i }$ the advantage estimated from its local rollout [19]. The clipped actor objective is

$$
\begin{array} { r l r } & { } & { \bar { w } _ { t } ^ { i } = \mathrm { c l i p } ( w _ { t } ^ { i } , 1 - \epsilon , 1 + \epsilon ) , } \\ & { } & { \mathcal { L } _ { \mathrm { c l i p } } ^ { i } ( \theta _ { i } ) = \mathbb { E } _ { t } \left[ \operatorname* { m i n } \left( w _ { t } ^ { i } \widehat { A } _ { t } ^ { i } , \bar { w } _ { t } ^ { i } \widehat { A } _ { t } ^ { i } \right) \right] . } \end{array}\tag{1}
$$

where $\epsilon > 0$ is the PPO clipping radius.

Agents individually calculate their own advantage and bootstrapped return target using local rewards and critic $V _ { \omega _ { i } } ( \tau _ { t } ^ { i } )$ They maximize Eq. (1) with entropy regularization, and fit their critic to their local return target; actor and critic parameters $( \theta _ { i } , \omega _ { i } )$ are learned independently for each agent. CAS-TLE maintains this IPPO update and information structure, but alters only how the policy logits are formed: the trainable actor receives frozen predictive context with an annealed consequence correction, while both pretrained world models remain frozen.

Recurrent state-space world models. An RSSM represents a partially observed trajectory using a deterministic recurrent state $x _ { t }$ and a stochastic latent state $\zeta _ { t }$ . Given the previous latent state and action, the recurrent dynamics first update the deterministic state. A dynamics predictor estimates the current stochastic state without observing $o _ { t }$ , whereas a second distribution also uses the encoded current observation [20]:

$$
\begin{array} { r l r } & { x _ { t } = f _ { \phi } ( x _ { t - 1 } , \zeta _ { t - 1 } , a _ { t - 1 } ) , } & { p _ { \phi } ( \zeta _ { t } \mid x _ { t } ) , } \\ & { e _ { t } = e _ { \phi } ( o _ { t } ) , } & { q _ { \phi } ( \zeta _ { t } \mid x _ { t } , e _ { t } ) . } \end{array}
$$

![](images/9eca6bcc34a30ea96fd137177485538b7fa9559a216d7be99f58068a34b3df56.jpg)  
Fig. 1. Architecture of CASTLE. Offline pretraining produces frozen LDWM checkpoints and an SSWM that emits one consequence token $z _ { t , k } ^ { i }$ for each candidate ego action. During decentralized IPPO, a trainable actor combines base policy logits with an annealed gated utility branch computed from the frozen SSWM tokens. Online gradients update only the orange PPO components; blue world-model modules remain frozen.

Thus, $q _ { \phi }$ does not update the recurrent state $( x _ { t } ) ;$ it estimates the current stochastic state using both the recurrent state and current observation. Prediction heads applied to the joint latent state $( x _ { t } , \zeta _ { t } )$ reconstruct observations and predict rewards and continuation. Algorithms such as Dreamer use the learned prior to generate imagined latent trajectories for control [12]. In our work, the pretrained RSSM remains frozen and its observation-conditioned latent state supplies local predictive context during online learning.

## III. METHODOLOGY

Overview. Our CASTLE separates offline world-model pretraining from online decentralized control. Figure 1 provides a schematic overview of the CASTLE framework. From local trajectories, the Local Dynamics World Model (LDWM) learns a predictive representation of each agent’s local dynamics. Meanwhile, the Semantic-Social World Model (SSWM) learns to predict the short-horizon consequences of every action available to the agent, given plausible behavior from its teammates and opponents. Both world models are frozen before online learning. The actor uses the LDWM context to form its base action logits and converts the SSWM consequence tokens into a temporary correction to those logits. Each agent then updates only its own actor, guidance branch, and critic through IPPO, without global state, a centralized critic, or communication.

## A. Counterfactual Semantic-Social World Model

The Semantic-Social World Model (SSWM) is one of the two central components of CASTLE. Let $\tau _ { t } ^ { i }$ denote agent i’s local history, which is then encoded by a transformer. For each available action $k \in \mathcal { A } ^ { i }$ , an embedding of k conditions a latent state that is advanced for H steps using the same learned transition. Each updated latent state predicts task progress, distances to relevant entities, interaction events, and accumulated return at that horizon. These predictions concern both the agent’s task and its interactions with teammates (cooperative) and opponents (competitive). A projection of the final latent state produces the consequence token $z _ { t , k } ^ { i } = F _ { \eta } ( \tau _ { t } ^ { i } , k )$ . The parameters η are learned offline and remain fixed during online learning and execution.

We construct counterfactual training targets by restarting the simulator from logged states and evaluating every available ego action from the same starting point. We refer to each rollout initialized from a restored logged state as a counterfactual simulator rollout. For each candidate ego action k, we sample a behavior-policy profile from the offline population. At the first step, agent i executes k, while the other agents sample actions from their respective behavior policies using their local information. All agents then follow the sampled behavior-policy profile for the remaining rollout steps, allowing teammate and opponent actions to adapt to the state changes induced by k. We run M such closed-loop counterfactual rollouts per candidate action and average their outcomes at each horizon to obtain the targets:

$$
y _ { t , k } ^ { i , h } = \left( \Delta _ { t , k } ^ { i , h } , d _ { t , k } ^ { i , h } , c _ { t , k } ^ { i , h } , R _ { t , k } ^ { i , h } \right) , \qquad h = 1 , \ldots , H .\tag{2}
$$

Here, ∆ measures role-dependent progress in the distancebased task objective; d records the minimum task-relevant distance observed up to horizon h; $c ~ \in ~ [ 0 , 1 ] ^ { 3 }$ collects event indicators and smooth task-relevant interaction signals; and R is the discounted return over the first h steps. Let $\mathcal { U } \ = \ \{ \Delta , d , c , R \}$ denote the target components, let $\mathbf { y } _ { t } ^ { i } \mathbf { \Sigma } = \mathbf { \Sigma }$ $\{ y _ { t , k } ^ { i , h } : k \in \mathcal { A } ^ { i } , h = 1 , \ldots , H \}$ , and let $\mathcal { D } _ { \mathrm { c f } }$ denote the offline dataset of pairs $( \tau _ { t } ^ { i } , \mathbf { y } _ { t } ^ { i } )$ . We supervise these quantities at intermediate horizons and at the final horizon using the following multi-task objective:

$$
\mathcal { L } _ { \mathrm { S S W M } } ( \eta ) = \mathcal { L } _ { \mathrm { s e m } } + \lambda _ { a } \mathcal { L } _ { \mathrm { a c t } } + \lambda _ { \mathrm { r e p } } \mathcal { R } _ { \mathrm { r e p } }\tag{3}
$$

The semantic prediction loss is

$$
\mathcal { L } _ { \mathrm { s e m } } ( \eta ) = \mathbb { E } _ { \mathcal { D } _ { \mathrm { c f } } , k \sim \mathrm { U n i f } ( \mathcal { A } ^ { i } ) } \left[ \sum _ { h = 1 } ^ { H } \sum _ { u \in \mathcal { U } } \lambda _ { u , h } \ell _ { u } \left( \widehat { u } _ { \eta , t , k } ^ { i , h } , u _ { t , k } ^ { i , h } \right) \right] .
$$

Here, $\hat { \cdot }$ denotes model predictions. The expectation averages over the counterfactual dataset and a uniformly sampled candidate action, which is equivalent to averaging over all $k \in \mathcal { A } ^ { i }$ . In $\mathcal { L } _ { \mathrm { s e m } } , \ell _ { u }$ is mean squared error for the continuous targets $u \in \{ \Delta , d , R \}$ and binary cross-entropy for the event vector c; the two sums supervise every target component at every rollout horizon. The candidate-independent action loss is $\mathcal { L } _ { \mathrm { a c t } } = \mathrm { C E } _ { \mathrm { m a s k } } ( \widehat { p } _ { - i , t } , a _ { - i , t } )$ . It predicts the logged actions of visible non-ego agents from the local-history representation and, because it does not depend on candidate action $k ,$ is not part of the target in Eq. (2). The fixed weights $\lambda _ { u , h }$ also account for the additional supervision of final-horizon progress, distance, and events. We use the representation regularizer $\mathcal { R } _ { \mathrm { r e p } } = \operatorname* { m a x } ( \delta - s _ { z } , 0 ) ^ { 2 } - \lambda _ { \mathrm { e n t } } \mathcal { H } _ { \mathrm { i n t } }$ , where $s _ { z }$ is the standard deviation over all candidate actions and is averaged over all minibatch samples and token dimensions. The first term penalizes variation below the threshold $\delta .$ We compute ${ \mathcal { H } } _ { \mathrm { i n t } }$ by averaging the auxiliary interaction head’s category probabilities over the minibatch and non-ego agent slots, then taking the entropy of this distribution. Weighted by $\lambda _ { \mathrm { e n t } }$ , this term discourages concentration of the average probabilities in a single category. The auxiliary head contributes only to offline pretraining.

## B. Local Dynamics World Model

In addition to the SSWM, each agent runs a frozen local copy of the Local Dynamics World Model (LDWM) LDWM<sub>ϕ</sub>. The LDWM specializes the RSSM in Section II to agent $i \mathrm { \ ' } _ { \mathrm { s } }$ local observation–action stream. Its deterministic state, prior, and posterior are

$$
\begin{array} { r l r l } & { x _ { t } ^ { i } = f _ { \phi } ( x _ { t - 1 } ^ { i } , \zeta _ { t - 1 } ^ { i } , a _ { t - 1 } ^ { i } ) , } & { p _ { t } ^ { i } = p _ { \phi } ( \zeta _ { t } ^ { i } \mid x _ { t } ^ { i } ) , } \\ & { e _ { t } ^ { i } = e _ { \phi } ( o _ { t } ^ { i } ) , } & { q _ { t } ^ { i } = q _ { \phi } ( \zeta _ { t } ^ { i } \mid x _ { t } ^ { i } , e _ { t } ^ { i } ) . } \end{array}
$$

The posterior distribution $\zeta _ { t } ^ { i } \sim q _ { t } ^ { i }$ incorporates the current local observation, and the LDWM context supplied to the actor is $\psi _ { t } ^ { i } = [ x _ { t } ^ { i } ; \zeta _ { t } ^ { i } ] = \mathrm { L D W M } _ { \phi } ( \tau _ { t } ^ { i } )$ . Heads conditioned on $\psi _ { t } ^ { i }$ reconstruct $o _ { t } ^ { i }$ and predict the local reward $r _ { t } ^ { i }$ and continuation indicator $c _ { t } ^ { i } .$ Let $\mathcal { D } _ { \mathrm { l o c } }$ denote the offline dataset of agents’ local trajectories and let ¯· indicate a stop-gradient. We pretrain the LDWM with the Dreamer-style objective

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { L D W M } } ( \phi ) = \mathbb { E } _ { \tau ^ { i } \sim \mathcal { D } _ { \mathrm { l o c } } } \sum _ { t } \bigg [ \lambda _ { o } \mathcal { L } _ { \mathrm { o b s } } ^ { t } + \lambda _ { r } \mathcal { L } _ { \mathrm { r e w } } ^ { t } + \lambda _ { c } \mathcal { L } _ { \mathrm { c o n t } } ^ { t } } \\ & { \qquad + \lambda _ { \mathrm { d y n } } D _ { \mathrm { K L } } \left( \bar { q } _ { t } ^ { i } \parallel p _ { t } ^ { i } \right) + \lambda _ { \mathrm { r e p } } D _ { \mathrm { K L } } \left( q _ { t } ^ { i } \parallel \bar { p } _ { t } ^ { i } \right) \bigg ] , } \end{array}\tag{4}
$$

The observation and reward losses use mean squared error (MSE), and continuation prediction uses binary crossentropy (BCE). For the reward head we use the target $\widetilde { r } _ { t } ^ { i } =$ $\mathrm { s g n } ( r _ { t } ^ { i } ) \log ( 1 + | r _ { t } ^ { i } | )$ to compress reward magnitudes while keeping their signs. The three prediction objectives are

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { o b s } } ^ { t } = \mathrm { M S E } ( \widehat { o } _ { t } ^ { i } , o _ { t } ^ { i } ) , \mathcal { L } _ { \mathrm { r e w } } ^ { t } = ( \widehat { r } _ { t } ^ { i } - \widetilde { r } _ { t } ^ { i } ) ^ { 2 } , \mathcal { L } _ { \mathrm { c o n t } } ^ { t } = \mathrm { B C E } ( \widehat { c } _ { t } ^ { i } , c _ { t } ^ { i } ) . } \end{array}
$$

In these objectives, the quantities with $\widehat { \cdot }$ are outputs of the prediction heads conditioned on $\psi _ { t } ^ { i }$ , with the reward head predicting $\widetilde { r } _ { t } ^ { i } .$ The continuation target $c _ { t } ^ { i }$ equals one if the episode continues after step t and zero otherwise. The first KL term trains the prior using $\hat { q } _ { t } ^ { i }$ as its target, while the second trains the posterior using $\hat { p } _ { t } ^ { i } .$ , balancing dynamics and representation learning [12], [21]. During online interaction, each agent updates $( x _ { t } ^ { i } , \zeta _ { t } ^ { i } )$ from only its own observations and actions and passes $\psi _ { t } ^ { i }$ to the actor encoder. The parameters $\phi$ remain fixed, and LDWM inference uses no global state, centralized critic, communication, or other agents’ private information.

## C. Decentralized Agent Consequence Guidance

During online learning, the base actor learns action preferences from local experience, while the guidance branch learns how the predicted consequence of each candidate should temporarily shift those preferences. A local gate controls the overall strength of this correction. For each agent $i ,$ let $\theta _ { i }$ collect the trainable parameters of its attention encoder, base actor, gate, and utility head. The attention encoder $T _ { \theta }$ first combines the current local observation with the frozen LDWM context:

$$
h _ { t } ^ { i } = T _ { \theta } \left( o _ { t } ^ { i } , \bar { \psi } _ { t } ^ { i } \right) .\tag{5}
$$

The base actor outputs the action-logit vector $\begin{array} { r l } { \ell _ { t } ^ { b , i } } & { { } = } \end{array}$ $\ell _ { \theta _ { i } } ^ { b } ( h _ { t } ^ { i } ) \ \in \ \mathbb { R } ^ { | \mathcal { A } ^ { i } | }$ . In parallel, the frozen SSWM is queried with the same local history $\tau _ { t } ^ { i }$ and each candidate action $k ,$ producing $z _ { t , k } ^ { i } = F _ { \eta } ( \tau _ { t } ^ { i } , k )$ . A trainable branch converts these tokens into a utility-logit vector:

$$
\begin{array} { r l } & { g _ { t } ^ { i } = g _ { \sigma } \big ( g _ { \theta } ( \bar { h } _ { t } ^ { i } ) \big ) , \quad u _ { t , k } ^ { i } = u _ { \theta } \big ( \bar { h } _ { t } ^ { i } , \bar { z } _ { t , k } ^ { i } , e _ { k } \big ) , } \\ & { \ell _ { t , k } ^ { u , i } = g _ { t } ^ { i } u _ { t , k } ^ { i } , \quad \ell _ { t } ^ { u , i } = [ \ell _ { t , k } ^ { u , i } ] _ { k = 1 } ^ { | A ^ { i } | } , } \\ & { \ell _ { t } ^ { i } = \ell _ { t } ^ { b , i } + \alpha _ { n } \ell _ { t } ^ { u , i } . } \end{array}\tag{6}
$$

Here, $g _ { t } ^ { i }$ is a scalar gate computed from the agent’s local actor representation, $g _ { \sigma }$ denotes a scaled logistic sigmoid (see Table II for details), $u _ { t , k } ^ { i }$ is an action-specific utility score, and $e _ { k }$ is a one-hot encoding of candidate action $k .$ The SSWM token $z _ { t , k } ^ { i }$ is a vector-valued representation of predicted consequences of taking action $k ,$ while a categorical policy requires one scalar preference per action. The candidate utility branch thus maps the token and local actor representation to a scalar correction to apply in the actor’s logit space. The stop-gradient on $h _ { t } ^ { i }$ prevents the guidance branch from reshaping the shared actor representation through this path; the base actor continues to train that representation through PPO. The gate and utility head are learned solely by backpropagating the IPPO actor objective through the final policy. They have no auxiliary loss providing target values for $u _ { t , k } ^ { i }$ . In this way, the frozen SSWM provides predictive features while online experiences determine how those features influence action preferences.

Let $\pi _ { b } ^ { i } ( \cdot  { \mid } \tau _ { t } ^ { i } ) = \mathrm { s o f t m a x } ( \ell _ { t } ^ { b , i } )$ be the base IPPO policy conditioned on the LDWM context. After normalization over the available actions, the policy after the $n ^ { t h }$ PPO update satisfies

$$
\pi _ { \theta } ^ { i } ( k \mid \tau _ { t } ^ { i } ) \propto \pi _ { b } ^ { i } ( k \mid \tau _ { t } ^ { i } ) \exp \left( \alpha _ { n } \ell _ { t , k } ^ { u , i } \right) , \qquad k \in \mathcal { A } ^ { i } .\tag{7}
$$

Consequence guidance, therefore, updates the distribution over actions with learned utility-logit vector from online experience, instead of learning a separate policy. When $\alpha _ { n } = 0 ,$ , this additive correction vanishes, and Eq. (7) reduces to the base actor policy after normalization.

The utility logit $\ell _ { t . k } ^ { u , i }$ is the action-specific score obtained by mapping the SSWM token through the learned utility branch. Because categorical logits represent relative action preferences, adding $\bar { \ell _ { t } ^ { u , i } }$ makes it a residual correction to the base actor rather than a second independently normalized policy. In particular, Eq. (7) gives

$$
\log \frac { \pi _ { \theta } ^ { i } ( k \mid \tau _ { t } ^ { i } ) } { \pi _ { \theta } ^ { i } ( j \mid \tau _ { t } ^ { i } ) } = \log \frac { \pi _ { b } ^ { i } ( k \mid \tau _ { t } ^ { i } ) } { \pi _ { b } ^ { i } ( j \mid \tau _ { t } ^ { i } ) } + \alpha _ { n } \left( \ell _ { t , k } ^ { u , i } - \ell _ { t , j } ^ { u , i } \right) .
$$

The SSWM branch therefore reranks the base actor’s candidate actions according to differences in their predicted consequence scores, while preserving the base preferences when those scores are equal.

For $\alpha _ { n } > 0$ , this logit addition is equivalently the solution of

$$
\pi _ { \boldsymbol { \theta } } ^ { i } ( \cdot  { \mid } \tau _ { t } ^ { i } ) = \underset { q \in \Delta ( \mathcal { A } ^ { i } ) } { \arg \operatorname* { m a x } } \left\{ \alpha _ { n } \sum _ { k \in \mathcal { A } ^ { i } } q ( k ) \ell _ { t , k } ^ { u , i } - D _ { \mathrm { K L } } \big ( q  { \parallel } \pi _ { b } ^ { i } ( \cdot  { \mid } \tau _ { t } ^ { i } ) \big ) \right.
$$

Thus, the correction has the geometry of a KL-regularized policy-improvement step: it favors actions with higher consequence scores while penalizing departure from the locally informed base policy. To see when this interpretation corresponds to genuine improvement, suppose that $\ell _ { t , k } ^ { u , i } \ =$ $A ^ { \pi _ { b } ^ { i } } ( \tau _ { t } ^ { i } , k )$ is the exact advantage of the base policy. Since the base policy is feasible and has zero expected advantage, optimality of the guided policy implies

$$
\mathbb { E } _ { k \sim \pi _ { \theta } ^ { i } } \Big [ A ^ { \pi _ { b } ^ { i } } ( \tau _ { t } ^ { i } , k ) \Big ] \geq \frac { 1 } { \alpha _ { n } } D _ { \mathrm { K L } } \big ( \pi _ { \theta } ^ { i } \| \pi _ { b } ^ { i } \big ) \geq 0 .
$$

Holding the other agents’ policies fixed, an exact advantage correction would therefore yield a local policy-improvement step. In our method, however, $\ell _ { t , k } ^ { u , i }$ is not supervised to equal this advantage. Instead, IPPO learns from online returns how the frozen counterfactual features should affect action preferences. Consequently, logit addition supplies the conservative update geometry, while improvement depends on whether the learned correction becomes advantage-aligned. The context gate and the annealing coefficient limit the effect of inaccurate corrections, and $\alpha _ { n }  0$ returns the policy to $\pi _ { b } ^ { i }$

The guiding signal coming from the utility head is gated by both the local-context gate $g _ { t } ^ { i }$ and $\alpha _ { n }$ , instead of having heterogeneous semantic predictions directly added to the actor logits. We linearly anneal $\alpha _ { n }$ from $\alpha _ { 0 }$ to zero over $N _ { \mathrm { a n n } }$ PPO updates. This way, the SSWM heavily influences action selection at the beginning of training, but its direct impact on the policy logits disappears once annealed. Both world models continue to run during the forward pass, and the LDWM continues to inject predictive context into the base actor. For each agent, we jointly optimize actor, guidance branch, and local critic with IPPO using only each agent’s rollout, while keeping both pretrained world models fixed [22].

Algorithm 1 summarizes offline pretraining and each agent’s independent online updates.

Algorithm 1 CASTLE   
Require: Offline local trajectories, logged simulator states and joint   
actions, behavior-policy population; H, M, α<sub>0</sub>, $N _ { \mathrm { a n n } } .$   
Offline pretraining   
1: for each logged $( s _ { t } ^ { - } , i )$ and $k \in \mathcal { A } ^ { i }$ do   
2: for $m = 1 , \ldots , M$ do   
3: Sample behavior-policy profile $\beta ^ { ( m ) } ;$ restore $s _ { t } .$   
4: Execute $a _ { t } ^ { i } \gets k$ and sample $a _ { t } ^ { - i } \sim \beta _ { - i } ^ { ( m ) }$   
5: Roll out the remaining $\bar { H } - 1$ steps with all agents   
following $\beta ^ { ( m ) }$   
6: Compute $( \Delta , d , c , R )$ through each horizon $\begin{array} { r l } { h } & { { } = } \end{array}$   
$1 , \ldots , H .$   
7: end for   
8: $y _ { t , k } ^ { i , h } \gets$ mean counterfactual-rollout targets at each h   
$( \operatorname { E q . } ( 2 ) ) .$   
9: end for   
10: Minimize ${ \mathcal { L } } _ { \mathrm { L D W M } } ( \phi )$ on local trajectories $( \operatorname { E q . } \ ( 4 ) ) .$   
11: Minimize $\mathcal { L } _ { \mathrm { S S W M } } ( \eta )$ using offline targets (Eq. (3)).   
12: Freeze $\phi , \eta .$   
Online, independently for each agent i   
13: Initialize $\theta _ { i } , \omega _ { i } ,$ , optimizers, local history, and recurrent state.   
14: for PPO updates $n = 0 , 1 , \ldots$ within the interaction budget do   
15: $\alpha _ { n }  \alpha _ { 0 }$ ma $\mathrm { \Omega } \mathrm { : } \{ 0 , 1 - \mathrm { \hat { n } / \mathrm { N } _ { a n n } } \}$   
16: for each local rollout step t do   
17: $\psi _ { t } ^ { i }  \mathrm { L D W M } _ { \phi } ( \tau _ { t } ^ { i } ) .$   
) .18: $\begin{array} { r } { z _ { t , k } ^ { i }  F _ { \eta } ( \tau _ { t } ^ { i } , \dot { k } ) , \dot { k } \in \mathcal { A } ^ { i } . } \end{array}$   
19: $h _ { t } ^ { i }  T _ { \theta _ { i } } ( o _ { t } ^ { i } , \bar { \psi } _ { t } ^ { i } ) ;$ compute $\ell _ { t } ^ { i }$ using Eq. (6).   
20: $a _ { t } ^ { \imath } \sim$ softmax(ℓ<sup>i</sup><sub>t</sub>).   
21: Execute $a _ { t } ^ { i } ;$ observe $r _ { t } ^ { i } , o _ { t + 1 } ^ { i } ,$ and termination.   
22: Store local transition and log $\pi _ { \theta _ { i } } ^ { i } \left( a _ { t } ^ { i } \ \middle | \ \tau _ { t } ^ { i } \right)$   
23: Advance local history; reset history and recurrent state at   
episode boundaries.   
24: end for   
25: Compute $\widehat { A } _ { t } ^ { i }$ and local return targets using $V _ { \omega _ { i } }$   
26: for PPO epochs and local minibatches $( \alpha _ { n }$ fixed) do   
27: Update $\theta _ { i }$ via Eq. (1) with entropy regularization.   
28: Fit $V _ { \omega _ { i } }$ to the local return targets.   
29: end for   
30: end for

## IV. EXPERIMENTS

We evaluate three questions: whether offline counterfactual pretraining improves fully decentralized online learning, whether the learned consequence representation transfers when the target environment is excluded from pretraining, and how the composition of the offline population affects performance. All results use fixed online training budgets and are reported without checkpoint selection.

## A. Offline Data and Pretraining

We generate task-specific offline populations by running random policies and policy checkpoints spanning different levels of competence. Tag and Spread populations combine random, decentralized IPPO, and MAPPO sources, while Adversary populations combine random and MAPPO sources. We retain these behavior-policy checkpoints for counterfactual label generation. Recorded for each transition and acting agent are the local observation, previous and current action, reward, termination flag, visible-entity features, and policysource metadata. The simulator state, policy-source metadata, and features used to compute $\mathbf { y } _ { t } ^ { i }$ support only offline label construction; the SSWM itself receives only $( \tau _ { t } ^ { i } , k )$ . MAPPO checkpoints are likewise used only to generate offline trajectories and counterfactual rollout labels; centralized training information is never available to the online agents. Random policies sample actions uniformly. Medium and expert labels follow the training stage for MAPPO and recorded sourcepolicy performance for decentralized IPPO, as summarized in Appendix Table V.

![](images/96820b4ff2fbc0421508e5381a9a9613e21e14714020d92978068cee50dce820.jpg)

![](images/85771bbe2332681cca5f67d4c076a771b760c966c81a18c20396b791a421c1f9.jpg)

![](images/727d8d94649886b6062e79361fd7f6dad537a81dcfcb57ecc10c7520eaa97a4f.jpg)  
Fig. 2. Target-task predictive accuracy over the SSWM rollout horizon. Solid lines indicate target-included pretraining and dashed lines held-out pretraining, evaluated on identical target-task histories. Correlations are computed across candidate actions; shading denotes 95% confidence intervals.

For each logged state and acting agent, we enumerate all five possible actions and use counterfactual simulator rollouts to construct the labels in Eq. (2). Each action is evaluated through four closed-loop counterfactual rollouts using a fivestep lookahead. In each rollout, we sample a behavior-policy profile from the offline population. The acting agent executes the candidate action at the first step, while the other agents act according to the sampled profile; thereafter, every agent follows that profile using its updated local information. Results are averaged across the four rollouts. These behavior policies need not match the agents encountered online, but their diversity exposes the SSWM to plausible teammate and opponent responses to the state changes induced by each candidate ego action.

Both frozen models are pretrained separately. The LDWM is pretrained to reconstruct local observations, predict rewards and continuation, and regularize its latent dynamics using only local transitions; it is never conditioned on joint observations nor given access to the global state. The SSWM optimizer is run with Eq. (3) on source-stratified offline batches. One deterministic subset of offline examples is held out from gradient updates and used exclusively for validation. We check a series of prespecified validation metrics before deploying each checkpoint to confirm that it preserves candidate-wise progress and distance rankings, predicts interaction events and non-ego actions, and maintains non-degenerate candidatetoken variation and auxiliary interaction-head usage. Online returns are used to neither train nor select either world model.

We present the semantic prediction quality at each stage of the rollout horizon in Figure 2. When including target-task data in pretraining, candidate-distance rankings remain informative through horizon h = 5, although event-prediction error increases with rollout depth. Predictive ranking is strongest on Adversary, remains stable on Tag, and improves over the early horizons on Spread. When the target task is excluded from pretraining, candidate-distance correlation approaches zero on Tag and Spread, while event-prediction error increases on all three tasks. Because both target-task included and held-out checkpoints are validated on the exact same held-out targettask histories, this diagnostic isolates predictive transfer from downstream online policy performance.

## B. Experiment Setup

We use three Multi-Agent Particle Environment (MPE) tasks [23]: Tag, Spread, and Adversary. In Tag, three cooperative predators try to catch one faster prey agent. In Spread, three homogeneous cooperative agents cover three landmarks while avoiding colliding into each other. In Adversary, two informed cooperative agents move towards a target landmark while an uninformed adversary tries to infer which landmark is the target and move there as well. These tasks require learning pursuit behavior, cooperative allocation behavior, and deceptive partial-information cooperative behavior, respectively.

Across all methods, we use the same decentralized information setting: agents cannot communicate with each other, learning is restricted to information from the locally available history, and no centralized critic is used. We compare against IPPO, IPPO with a Transformer encoder, IQL, IQL with a Transformer encoder based on JaxMARL implementation [23], DPO [10], and a decentralized Dreamer implementation in which each agent uses separate parameters and maintains its own replay buffer. For each target task, heldout SSWM pretraining employs random, medium, and expert trajectories drawn from the other two MPE tasks, excluding any from the task itself. The in-domain (R) method integrates only random trajectories from the target task into the source pool, contrasting with in-domain (R+M), which also leverages intermediate target-task trajectories.

## C. Online Performance

Table I shows D4RL-normalized final-window results. The CASTLE columns consist of the framework pictured in Figure 1: LDWM context frozen along with annealed SSWM consequence guidance. The three columns differ by the three CASTLE variants differ in the target-task behavior available during offline pretraining, as defined in Section IV-B.

Our in-domain (R+M) variant reaches the highest mean across all three tasks and the highest mean normalized score across tasks. Compared to the strongest decentralized baseline in each row, it performs 10.67, 6.46, and 0.33 points higher on Tag, Spread, and Adversary, respectively.

Figure 3 shows that consequence guidance alters learning across regimes of interaction. During Tag, CASTLE’s in domain learning starts slowly but enters a higher-performance region after about a million steps. During Spread, learning accelerates immediately, and the transient around one million steps results from the conclusion of the guidance schedule; then CASTLE continues fully on decentralized IPPO and maintains its lead. These curves are closer together for Adversary, indicating that guidance is honing in on the terminal policy more for this task. Held-out guidance shows a broadly similar pattern but does not always confer an advantage.

TABLE I  
D4RL-NORMALIZED FINAL PERFORMANCE OVER 30 MATCHED SEEDS, REPORTED AS MEAN ± STANDARD DEVIATION. HIGHER IS BETTER. IN-DOMAIN (R) ADDS TARGET-TASK RANDOM TRAJECTORIES TO THE HELD-OUT SOURCE POOL, WHILE IN-DOMAIN (R+M) ADDS TARGET-TASK RANDOM AND INTERMEDIATE TRAJECTORIES. BOLD AND UNDERLINED ENTRIES ARE THE BEST AND SECOND-BEST MEANS IN EACH COLUMN.
<table><tr><td>Method</td><td>tag</td><td>spread</td><td>adversary</td><td>Total Average</td></tr><tr><td colspan="5">Decentralized baselines</td></tr><tr><td>IPPO</td><td> $7 8 . 0 7 \pm 8 . 6 7$ </td><td> $7 0 . 6 9 \pm 1 2 . 2 9$ </td><td> $1 0 0 . 3 4 \pm 0 . 9 2$ </td><td>83.03</td></tr><tr><td>IPPO + Transformer</td><td> $7 0 . 8 3 \pm 1 0 . 5 6$ </td><td> $7 9 . 6 4 \pm 5 . 5 8$ </td><td> $1 0 2 . 9 1 \pm 1 . 4 1$ </td><td>84.46</td></tr><tr><td>IQL</td><td> $7 6 . 7 5 \pm 7 . 4 8$ </td><td> $7 6 . 4 1 \pm 9 . 2 4$ </td><td> $9 5 . 5 5 \pm 1 . 8 4$ </td><td>82.90</td></tr><tr><td>IQL + Transformer</td><td> $6 5 . 9 1 \pm 8 . 0 4$ </td><td> $\underline { { 8 6 . 1 1 } } \pm 4 . 3 3$ </td><td> $1 0 1 . 0 5 \pm 1 . 3 3 $ </td><td>84.36</td></tr><tr><td>DPO</td><td> $7 4 . 4 6 \pm 7 . 3 1$ </td><td> $8 2 . 0 3 \pm 4 . 4 7$ </td><td> $9 8 . 7 7 \pm 1 . 2 2$ </td><td>85.09</td></tr><tr><td>Dreamer</td><td> $5 7 . 6 7 \pm 1 1 . 3 1$ </td><td> $2 4 . 2 6 \pm 1 5 . 1 5$ </td><td> $\underline { { 1 0 5 . 4 8 } } \pm 0 . 9 4$ </td><td>62.47</td></tr><tr><td colspan="5">CASTLE (ours)</td></tr><tr><td>Held-out</td><td> $7 2 . 4 1 \pm 1 1 . 4 5$ </td><td> $8 4 . 7 2 \pm 6 . 9 7$ </td><td> $1 0 5 . 1 9 \pm 1 . 1 2$ </td><td>87.44</td></tr><tr><td>In-domain (R)</td><td> $7 2 . 1 9 \pm 1 0 . 9 8$ </td><td> $8 1 . 4 8 \pm 7 . 1 0$ </td><td> $1 0 3 . 4 0 \pm 0 . 8 3$ </td><td>85.69</td></tr><tr><td>In-domain (R+M)</td><td> ${ \bf 8 8 . 7 4 \pm 9 . 1 9 }$ </td><td> ${ \bf 9 2 . 5 7 \pm 1 . 6 5 }$ </td><td> ${ \bf 1 0 5 . 8 1 \pm 0 . 7 6 }$ </td><td>95.70</td></tr></table>

## D. Held-Out and In-Domain Transfer

Target-task exposure alone is not enough to see gains in transfer. Compared to held-out pretraining, adding only random and weak target-task trajectories changes normalized score by −0.22 on Tag, −3.24 on Spread, and −1.79 on Adversary. Improving on this baseline by adding intermediateexpert target-task behavior on top of random data further raises scores by 16.55, 11.09, and 2.41 points, respectively. The resulting in-domain (R+M) improvements over held-out pretraining are therefore 16.33 points on Tag, 7.85 points on Spread, and 0.62 points on Adversary. The large Tag gap shows that pursuit particularly benefits from coverage of target-task interaction. By contrast, the small gaps on Spread and Adversary indicate that the consequence representation learned by interacting with the other MPE tasks already includes much of the structure needed to solve those tasks.

## E. Ablation Studies

The primary comparison shows the efficacy of the entire method but does not attribute which design choices provide the gain. Here we compare CASTLE in-domain (R+M) against three controls: one that removes LDWM context, one that removes SSWM consequence guidance, or one that keeps the SSWM correction module active throughout training rather than annealing it out. All the hyperparameters are kept the same, and all applicable pretrained components are from the same in-domain (R+M) offline population. Figure 4 shows the resulting learning curves.

These curves demonstrate different benefits between interaction regimes. CASTLE improves more slowly before separating after about a million steps on Tag, thereafter having the strongest return through the rest of training. On Spread it experiences a large early improvement before producing an aligned deviation upon removal of guidance. This behavior is exhibited across all 30 seeds. Decentralized IPPO then recovers to maintain an advantage later in training. On Adversary, both CASTLE and the configuration without LDWM show continued improvement while the no-SSWM and fixedguidance configurations stay at levels close to their early plateau.

## F. Mechanism Diagnostics

The ablations demonstrate whether the SSWM branch impacts learning, however they leave open the question of how the branch alters action selection behavior. To answer this question, we compare two policy distributions at every update. The guided policy $\pi _ { t } ^ { i }$ is the distribution used to sample actions from the final logits of Eq. (6). The corresponding base policy $\pi _ { t } ^ { b , i }$ is obtained from the same actor given the same local input, but after removing the SSWM utility correction. We quantify their difference with $D _ { \mathrm { K L } } ( \pi _ { t } ^ { i } \parallel \pi _ { t } ^ { b , \bar { i } } )$ , averaged across local rollout samples. If this quantity is zero, then the branch leaves the base policy unchanged, whereas a larger value indicates a stronger redistribution of action probability.

We next measure whether the policy redistribution induced by guidance is aligned with the subsequent local learning signal. For the sampled action, we compute $\Delta \log \pi _ { t } ^ { i } =$ log $\pi _ { t } ^ { i } ( a _ { t } ^ { i } ) - \log \pi _ { t } ^ { b , i } ( a _ { t } ^ { i } )$ and correlate it with the local GAE estimate $\widehat { A } _ { t } ^ { i }$ obtained from the completed rollout [19]. A positive correlation indicates that guidance tends to increase the probability of actions assigned higher local advantage. This correlation measures alignment with PPO’s online learning signal rather than an independent causal effect; causal evidence is provided by the matched ablations in Section IV-E. This diagnostic does not control the utility gate or update either frozen world model. Guidance is disabled after update 244, after which the policy is optimized exclusively through decentralized IPPO.

Despite monotonically decreasing throughout training, in Figure 5 policy KL first increases until updates 126, 111, and 130 for Tag, Spread, and Adversary, respectively, before decreasing again. The peak KL values are 0.069, 0.233, and 0.141. Intervention is therefore not solely based on the annealing schedule. Intervention also depends on how the trainable gate and utility head learn to utilize the frozen tokens given the evolving policy and visited state distribution. The subsequent drop happens when the diminishing coefficient begins to outweigh this learned scaling. Finally, the near-zero KL following update 244 exists mostly to verify that annealing actually cuts off the branch as intended.

![](images/92b45aa5e1fef031fb2f2fd752f2ef0f056c192ed67208a668b859982033e804.jpg)

![](images/5204f51501b09fd4f0911dc24b9ac01aee7fcb411bd273ce17bc52a84beadd5a.jpg)

![](images/7970634aea57a59925dc96474f4588b98407efeeb3080881fa727b9d6a14046d.jpg)  
IPPO IPPO + Transformer CASTLE held-out CASTLE in-domain (R+M)

Fig. 3. Raw online returns over 30 matched seeds. Lines show means and shaded regions 95% confidence intervals. Curves use a 25-update moving average.  
![](images/aa8df5669a44c8ad8b12a47453534c6cff4342d17a07ffb99c9c7731992087d8.jpg)  
Environment steps (M)

![](images/15f944702aaf67943e15adf92bb6eb008f0febaee3d2fd88724a150c77d3cbe1.jpg)  
Environment steps (M)

![](images/8d879c99f3f491ea40a97620da5fb452ae3ba382b9892f373e2828c4f82434e1.jpg)  
Environment steps (M)  
Fig. 4. Raw online returns over 30 matched seeds. Lines show means and shaded regions 95% confidence intervals. Curves use a 25-update moving average.

![](images/b1101f5973707f786dea132c338f9d29612abd6f943198145eb4a901cd74fae0.jpg)  
Fig. 5. Effect of consequence guidance over 30 matched seeds. Top: KL divergence between the guided policy and the base IPPO actor. Bottom: correlation between the guidance-induced change in the sampled action’s log probability and its realized local GAE advantage.

Alignment shows a correlated result. Looking solely at positive or aggregate values, alignment stays positive throughout this period, showing mean alignment scores of .0359 on Tag, .0584 on Spread, and .0383 on Adversary. Spread shows both the highest intervention and alignment makes sense given its quick early gains. However, Tag achieves the highest final performance improvement compared to the decentralized baselines while having the lowest KL by far, so intervention size does not track final performance across tasks. Figure 5, therefore, verifies that the branch becomes active, adapts to the online PPO signal, and is removed by annealing. Whether SSWM guidance improves learning is a separate question addressed by the matched component controls in Section IV-E.

## G. Discussion

The results indicate that SSWM guidance is most effective here when it shapes early action selection rather than maintaining a persistent influence throughout training. Maintaining SSWM correction performs worse than annealed guidance; once the correction weight decays to zero, subsequent policy updates rely exclusively on local experience, while the frozen LDWM continues to provide context. While the idea of distilling inductive bias into RL agents is not new [24]–[26], our world model approach provides in-context bias on the logits level. Although evaluated with decentralized IPPO, the framework is not tied to PPO’s clipped objective and could be adapted to other decentralized policy-gradient methods with categorical policies.

These results should be understood in the context of our experimental setup. We benchmark three specific discreteaction MPE tasks, and our current label-construction pipeline requires offline access to simulator states from which counterfactual simulator rollouts can be initialized. Its semantic targets rely on geometric relations and interaction events provided by these tasks. As both world models’ weights remain unchanged during online learning, their predictions can degrade when the policy experiences local histories not frequently encountered in the offline data. Annealing limits how long the SSWM has direct effect over action selection; however, it is agnostic to whether a prediction is trustworthy. Future work can estimate predictive uncertainty from local information and scale down guidance as the prediction becomes less certain, while maintaining decentralized execution.

## V. RELATED WORKS

## A. Fully Decentralized Multi-Agent Reinforcement Learning

CTDE methods optimize with access to privileged joint information and execute local policies after training [22]. Fully decentralized methods train without a centralized signal. IPPO decentralizes by independently applying PPO [22]. I2Q and Best Possible Q-Learning reformulate the local valuelearning objective to lessen transitions’ dependence on other agents’ actions [9], [27]. DPO backpropagates a decentralized surrogate signal for multi-agent policy improvement [10]. RAC conditions each local policy on return-derived context to help infer unobserved changes in joint behavior [11]. The term decentralized has also been used in a separate literature to refer to networked optimization where agents can exchange gradients or parameters with neighbors [28], [29]. Our online learning setting rules out both centralized and networked training signals, CASTLE, freezes and locally computes actionconsequence tokens for each independent learner and diminishes their direct influence as online experience is collected.

## B. Multi-Agent World Models

Single-agent world models learn compact latent states for planning or learning policies [12], [20], [30]. Multi-agent variants additionally need to model how agents affect the transition function. MAMBA and MABL learn compositional multi-agent latent dynamics with CTDE [13], [15], MARIE performs decentralized sequence modeling with centralized aggregation [31]. MAG instead studies agent-wise local models and explicitly optimizes how their prediction errors interact over multi-step rollouts [32]. Recent work also leverages diffusion models [33], mixture-of-experts [34], and compositional models [35] for joint or agent-structured imagination for control and planning. Other decentralized world-model approaches rely on explicit information sharing during training [14]. In contrast, CASTLE does not use a learned global latent state or online messages. Its world models are pretrained offline, queried from local histories, and frozen throughout decentralized policy optimization.

Counterfactual models of other agents have been used to measure social influence as well. Jaques et al. predict how alternative actions of one agent change the action distribution of another agent and use this causal influence as an intrinsic reward [36]. CASTLE instead predicts the full range of physical, task, and social consequences of every action and lets the online actor learn how those predictions should alter its action preferences. It neither defines an influence reward nor uses online messages or a global latent state.

## C. Offline Pretraining and In-Context Reinforcement Learning

Inspired by recent advances in large language models, in-context learning lets a model use examples provided in the input to condition its predictions without changing its parameters [37]. In RL, trajectories collected during recent experience can provide this conditioning context, allowing a pretrained policy to reuse information between trials or adapt to changes in the task and other agents. Recent methods accomplish this with hindsight-organized experience, hierarchical in-context decisions, locally consistent predictions of other agents, or obtain context from a pretrained communicative world model [38]–[41].

Offline MARL has likewise leveraged centralizedto-decentralized distillation, diffusion policies, and sequence models to extract coordinated behavior from fixed datasets [42]–[44]. These approaches place offline knowledge directly into agents’ action-generating policies, leaving coverage and quality of the pretraining dataset as primary determinants of learned behavior [45]–[47].

## VI. CONCLUSION

We presented CASTLE, an offline-to-online world-model based method for fully decentralized MARL. A frozen LDWM provides predictive dynamics context to each agent’s actor, while the frozen SSWM encodes the short-horizon consequences of each candidate action under locally plausible teammate and opponent behavior. Each IPPO actor uses online learning to discover how to translate these consequence tokens into an annealed correction of its action preferences. This enables knowledge transfer from offline populations without a centralized critic, communication, or shared online optimizer.

Across 30 matched seeds on Tag, Spread, and Adversary, our in-domain (R+M) variant yields the highest normalized mean performance on every task. Held-out pretraining also yields improved performance over the decentralized baselines on Spread, and remains competitive with the strongest decentralized baselines on Tag and Adversary. Our population comparisons and ablations show that intermediate target-task behavior is more beneficial than using random target-task data alone, and that transient consequence guidance outperforms retaining the model correction throughout training. These results indicate a promising use case for pretrained world models as ephemeral, action-level priors for decentralized online learning instead of persistent controllers or replacements for real experience.

## REFERENCES

[1] C. Amato, “An introduction to centralized training for decentralized execution in cooperative multi-agent reinforcement learning,” arXiv preprint arXiv:2409.03052, 2024.

[2] C. Zhu, M. Dastani, and S. Wang, “A survey of multi-agent deep reinforcement learning with communication,” Autonomous Agents and Multi-Agent Systems, vol. 38, no. 1, p. 4, 2024.

[3] Z. Pan, H. Lei, F. Zuo, Z. Bian, and T. Li, “Distributed online convex optimization with nonseparable costs and constraints,” IEEE Control Systems Letters, vol. 10, pp. 391–396, 2026.

[4] Y. Pan, T. Li, and Q. Zhu, “Model-agnostic hessian-free meta-policy optimization via zeroth-order estimation: A linear quadratic regulator perspective,” Dynamic Games and Applications, vol. 16, no. 1, pp. 386– 425, 2026.

[5] K. Su and Z. Lu, “A fully decentralized surrogate for multi-agent policy optimization,” Transactions on Machine Learning Research, 2024.

[6] T. Li, Y. Zhao, and Q. Zhu, “The role of information structures in gametheoretic multi-agent learning,” Annual Reviews in Control, vol. 53, pp. 296–314, 2022.

[7] C. S. De Witt, T. Gupta, D. Makoviichuk, V. Makoviychuk, P. H. Torr, M. Sun, and S. Whiteson, “Is independent learning all you need in the starcraft multi-agent challenge?,” arXiv preprint arXiv:2011.09533, 2020.

[8] C. Daskalakis, D. J. Foster, and N. Golowich, “Independent policy gradient methods for competitive reinforcement learning,” in Advances in Neural Information Processing Systems, 2020.

[9] J. Jiang and Z. Lu, “I2Q: A fully decentralized q-learning algorithm,” in Advances in Neural Information Processing Systems, vol. 35, pp. 20469– 20481, 2022.

[10] K. Su and Z. Lu, “A fully decentralized surrogate for multi-agent policy optimization,” Transactions on Machine Learning Research, 2024.

[11] C. Li, B. Bao, and Y. Gao, “In-context fully decentralized cooperative multi-agent reinforcement learning,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[12] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap, “Mastering diverse control tasks through world models,” Nature, vol. 640, pp. 647–653, 2025.

[13] V. Egorov and A. Shpilman, “Scalable multi-agent model-based reinforcement learning,” in Proceedings ofthe 21st International Conference on Autonomous Agents and Multiagent Systems, AAMAS ’22, (Richland, SC), p. 381–390, International Foundation for Autonomous Agents and Multiagent Systems, 2022.

[14] X. Zeng and Q. Zhang, “Efficient information sharing for training decentralized multi-agent world models,” Reinforcement Learning Journal, vol. 6, pp. 909–922, 2025.

[15] A. Venugopal, S. Milani, F. Fang, and B. Ravindran, “Mabl: Bi-level latent-variable world model for sample-efficient multi-agent reinforcement learning,” in Proceedings of the 23rd International Conference on Autonomous Agents and Multiagent Systems, AAMAS ’24, (Richland, SC), p. 1865–1873, International Foundation for Autonomous Agents and Multiagent Systems, 2024.

[16] D. Xue, J. Jiang, S. Zhang, W. Guo, L. Yuan, Z. Zhang, and Y. Yu, “Learning disentangled multi-agent world model for decentralized control,” in Proceedings of the 43rd International Conference on Machine Learning, vol. 306 of Proceedings of Machine Learning Research, PMLR, 2026.

[17] T. Li, G. Peng, Q. Zhu, and T. Baar, “The confluence of networks, games, and learning a game-theoretic framework for multiagent decision making over networks,” IEEE Control Systems, vol. 42, no. 4, pp. 35–67, 2022.

[18] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[19] J. Schulman, P. Moritz, S. Levine, M. I. Jordan, and P. Abbeel, “Highdimensional continuous control using generalized advantage estimation,” in International Conference on Learning Representations, 2016.

[20] D. Hafner, T. Lillicrap, I. Fischer, R. Villegas, D. Ha, H. Lee, and J. Davidson, “Learning latent dynamics for planning from pixels,” in Proceedings of the 36th International Conference on Machine Learning (K. Chaudhuri and R. Salakhutdinov, eds.), vol. 97 of Proceedings of Machine Learning Research, pp. 2555–2565, PMLR, 2019.

[21] F. Martinez, T. Li, Y. Lu, and J. Chen, “Stackelberg coupling of online representation learning and reinforcement learning,” in International Conference on Learning Representations, vol. 2026, pp. 2410–2447, 2026.

[22] C. Yu, A. Velu, E. Vinitsky, J. Gao, Y. Wang, A. Bayen, and Y. Wu, “The surprising effectiveness of PPO in cooperative multi-agent games,” in Advances in Neural Information Processing Systems, vol. 35, pp. 24611– 24624, 2022.

[23] A. Rutherford, B. Ellis, M. Gallici, J. Cook, A. Lupu, G. Ingvarsson, T. Willi, R. Hammond, A. Khan, C. S. de Witt, A. Souly, S. Bandyopadhyay, M. Samvelyan, M. Jiang, R. T. Lange, S. Whiteson, B. Lacerda, N. Hawes, T. Rocktaschel, C. Lu, and J. N. Foerster, “Jaxmarl: Multi-¨ agent rl environments and algorithms in jax,” in The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024.

[24] M. Hessel, H. v. Hasselt, J. Modayil, and D. Silver, “On inductive biases in deep reinforcement learning,” arXiv, 2019.

[25] T. Li, H. Lei, M. Yin, and Y. Hu, “Reinforcement learning with physics-informed symbolic program priors for zero-shot wireless indoor navigation,” in Inductive Biases in Reinforcement Learning Workshop, Reinforcement Learning Conference 2025, 2025.

[26] T. Li, H. Lei, H. Guo, M. Yin, Y. Hu, Q. Zhu, and S. Rangan, “Digital twin-enhanced wireless indoor navigation: Achieving efficient environment sensing with zero-shot reinforcement learning,” IEEE Open Journal of the Communications Society, vol. 6, pp. 2356–2372, 2025.

[27] J. Jiang and Z. Lu, “Best possible q-learning,” in Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence (S. Chiappa and S. Magliacane, eds.), vol. 286 of Proceedings of Machine Learning Research, pp. 1895–1908, PMLR, 2025.

[28] K. Zhang, Z. Yang, H. Liu, T. Zhang, and T. Basar, “Fully decentralized multi-agent reinforcement learning with networked agents,” in Proceedings of the 35th International Conference on Machine Learning, vol. 80, pp. 5872–5881, 2018.

[29] Z. Chen, Y. Zhou, R.-R. Chen, and S. Zou, “Sample and communicationefficient decentralized actor-critic algorithms with finite-time analysis,” in Proceedings of the 39th International Conference on Machine Learning (K. Chaudhuri, S. Jegelka, L. Song, C. Szepesvari, G. Niu, and S. Sabato, eds.), vol. 162 of Proceedings ofMachine Learning Research, pp. 3794–3834, PMLR, 2022.

[30] J. Schrittwieser, I. Antonoglou, T. Hubert, K. Simonyan, L. Sifre, S. Schmitt, A. Guez, E. Lockhart, D. Hassabis, T. Graepel, T. Lillicrap, and D. Silver, “Mastering atari, go, chess and shogi by planning with a learned model,” Nature, vol. 588, pp. 604–609, 2020.

[31] Y. Zhang, C. Bai, B. Zhao, J. Yan, X. Li, and X. Li, “Decentralized transformers with centralized aggregation are sample-efficient multiagent world models,” Transactions on Machine Learning Research, 2025.

[32] Z. Wu, C. Yu, C. Chen, J. Hao, and H. H. Zhuo, “Models as agents: Optimizing multi-step predictions of interactive local models in modelbased multi-agent reinforcement learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, pp. 10435–10443, 2023.

[33] Y. Zhang, X. Li, J. Ye, S. Qiu, D. Qu, X. Li, C. Zhang, and C. Bai, “Revisiting multi-agent world modeling from a diffusion-inspired perspective,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[34] Z. Zhao, Z. Zhao, K. Xu, Y. Fu, J. Chai, Y. Zhu, and D. Zhao, “Learning and planning multi-agent tasks via an MoE-based world model,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[35] H. Zhang, Z. Wang, Q. Lyu, Z. Zhang, S. Chen, T. Shu, B. Dariush, K. Lee, Y. Du, and C. Gan, “COMBO: Compositional world models for embodied multi-agent cooperation,” in International Conference on Learning Representations, 2025.

[36] N. Jaques, A. Lazaridou, E. Hughes, C. Gulcehre, P. Ortega, D. J. Strouse, J. Z. Leibo, and N. de Freitas, “Social influence as intrinsic motivation for multi-agent deep reinforcement learning,” in Proceedings of the 36th International Conference on Machine Learning, vol. 97 of Proceedings of Machine Learning Research, pp. 3040–3049, PMLR, 2019.

[37] Q. Dong, L. Li, D. Dai, et al., “A survey on in-context learning,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, (Miami, Florida, USA), pp. 1107–1128, Association for Computational Linguistics, Nov. 2024.

[38] H. Liu and P. Abbeel, “Emergent agentic transformer from chain of hindsight experience,” in International Conference on Machine Learning, pp. 21362–21374, PMLR, 2023.

[39] S. Huang, J. Hu, Z. Yang, et al., “Decision mamba: Reinforcement learning via hybrid selective sequence modeling,” in NeurIPS, 2024.

[40] T. Li, J. Guevara, X. Xie, and Q. Zhu, “Self-confirming transformer for belief-conditioned adaptation in offline multi-agent reinforcement learning,” in Proceedings of the Seventh Workshop on Adaptive and Learning Agents, pp. 1–10, 2025.

[41] F. Martinez, T. Li, Y. Lu, and J. Chen, “In-context reinforcement learning via communicative world models,” in Proceedings of the thirtyfifth international joint conference on artificial intelligence, IJCAI-26, p. 253–261, 2026.

[42] W.-C. Tseng, T.-H. J. Wang, Y.-C. Lin, and P. Isola, “Offline multiagent reinforcement learning with knowledge distillation,” in Advances in Neural Information Processing Systems, vol. 35, 2022.

[43] Z. Zhu, M. Liu, L. Mao, B. Kang, M. Xu, Y. Yu, S. Ermon, and W. Zhang, “MADiff: Offline multi-agent learning with diffusion models,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[44] J. Formanek, O. Mahjoub, L. Nessir, S. Abramowitz, R. J. de Kock, W. Khlifi, D. Rajaonarivonivelomanantsoa, S. Du Toit, A. Fokam, S. Singh, U. A. Mbou Sob, F. Chalumeau, and A. Pretorius, “Oryx: A scalable sequence model for many-agent coordination in offline MARL,” in Advances in Neural Information Processing Systems, vol. 38, 2025.

[45] C. Formanek, C. R. Tilbury, L. Beyers, J. Shock, and A. Pretorius, “Dispelling the mirage of progress in offline MARL through standardised baselines and evaluation,” in Advances in Neural Information Processing Systems, vol. 37, 2024.

[46] T. Li, K. Hammar, R. Stadler, and Q. Zhu, “Conjectural online learning with first-order beliefs in asymmetric information stochastic games,” in 2024 IEEE 63rd Conference on Decision and Control (CDC), IEEE CDC, pp. 6780–6785, 2024.

[47] K. Hammar, T. Li, R. Stadler, and Q. Zhu, “Adaptive security response strategies through conjectural online learning,” IEEE Transactions on Information Forensics and Security, vol. 20, pp. 4055–4070, 2025.

[48] J. Fu, A. Kumar, O. Nachum, G. Tucker, and S. Levine, “D4RL: Datasets for deep data-driven reinforcement learning,” arXiv preprint arXiv:2004.07219, 2020.

## APPENDIX

## A. Software, hardware, and Experimental Setup

Our implementation used: Python 3.13.7, JAX and jaxlib 0.5.3, Flax 0.10.6, Optax 0.2.6, Distrax 0.1.5, Chex 0.1.90, NumPy 1.26.4, Gymnax 0.0.9. The MPE environments use JaxMARL 0.1.0. GPU support used the JAX CUDA 12 plugin 0.5.3. All our experiments used an NVIDIA L40 GPU.

TABLE II  
MODEL ARCHITECTURE AND LDWM REFERENCE PRETRAINING.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Local Dynamics World Model (LDWM)</td><td></td></tr><tr><td>Deterministic / hidden dimension</td><td>256 /  256</td></tr><tr><td>Stochastic variables / categories</td><td>32 / 32</td></tr><tr><td>Context dimension / uniform mixture</td><td>1280 / 0.01</td></tr><tr><td>Adam learning rate / epsilon</td><td> $3 \times 1 0 ^ { - 4 } / 1 0 ^ { - 5 }$ </td></tr><tr><td>Gradient norm limit / KL free nats</td><td>100 / 0.5</td></tr><tr><td>Obs. / reward / continuation weights</td><td>1 / 1 / 1</td></tr><tr><td>Dynamics / representation KL weights</td><td>0.8 / 0.2</td></tr><tr><td>Semantic-Social World Model (SSWM)</td><td></td></tr><tr><td>History length / hidden dimension</td><td>8 / 128</td></tr><tr><td>Transformer layers / attention heads</td><td>1/4</td></tr><tr><td>Token width / interaction categories</td><td>16 /  4</td></tr><tr><td>Padded observation dimension</td><td>18</td></tr><tr><td>Entity slots / features per entity</td><td>7/ 2</td></tr><tr><td>Actor, Critic, and Guidance</td><td></td></tr><tr><td>Attention / feed-forward width</td><td>64 / 128</td></tr><tr><td>Transformer layers / attention heads</td><td>1/ 4</td></tr><tr><td>Actor / critic output dimension</td><td>5 / 1</td></tr><tr><td>Utility hidden width / activation</td><td>64  / SiLU</td></tr><tr><td>Utility output per action / gate</td><td>1 / 0.5 σ(·)</td></tr></table>

TABLE III

ONLINE PPO AND CONSEQUENCE GUIDANCE.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Parallel environments / rollout steps</td><td>32 /  128</td></tr><tr><td>Optimizer / epsilon</td><td>Adam / 10−5</td></tr><tr><td>Initial learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Learning-rate decay / lower bound</td><td>Linear / 10% of initial</td></tr><tr><td>PPO epochs / minibatches per epoch</td><td>4/ 4</td></tr><tr><td>Discount / GAE parameter</td><td>0.99 / 0.95</td></tr><tr><td>Policy clip / gradient norm limit</td><td>0.2 / 0.5</td></tr><tr><td>Value loss / entropy coefficients</td><td>1.0 / 0.03</td></tr><tr><td>Guidance coefficient, initial to final</td><td>0.10 to 0</td></tr><tr><td>Annealing duration / schedule</td><td>244 updates / linear</td></tr><tr><td>Token supplied to the actor</td><td>Final horizon</td></tr></table>

World models remain frozen. Guidance gradients do not enter the actor representation. Actor and critic parameters and optimizers are private to each agent.

TABLE IV  
SSWM PRETRAINING, LABEL CONSTRUCTION, VALIDATION.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Optimization and Loss Weights</td><td></td></tr><tr><td>Optimizer / learning rate</td><td>AdamW / 3 × 10−4</td></tr><tr><td>Weight decay / batch size</td><td>10−4 / 256</td></tr><tr><td>Gradient steps / training and split seed</td><td>10,000 / 0</td></tr><tr><td>Random / decentralized / CTDE sampling</td><td>0.10 / 0.60 / 0.30</td></tr><tr><td>Additional final ∆ / d / c weights</td><td>1 / 1  / 1</td></tr><tr><td>Horizon-wise ∆ / d / c weights</td><td>0.5 / 0.5 / 0.5</td></tr><tr><td>Horizon-wise R / action loss weights</td><td>0.25 / 0.5</td></tr><tr><td>Token variation / entropy weights</td><td>0.05 / 0.01</td></tr><tr><td>Token standard-deviation floor</td><td>0.01</td></tr><tr><td>Label Configuration</td><td></td></tr><tr><td>Actions / rollouts (M) / horizon (H)</td><td>5 /4/5</td></tr><tr><td>Counterfactual return discount</td><td>0.95</td></tr><tr><td>Candidate ego action</td><td>1st step</td></tr><tr><td>Non-ego actions</td><td>Sampled behavior policies</td></tr><tr><td>Subsequent actions, all agents</td><td>Sampled behavior policies</td></tr><tr><td>Rollout aggregation</td><td>Mean at each h</td></tr><tr><td>Validation</td><td></td></tr><tr><td>Training / validation split</td><td>90% / 10%</td></tr><tr><td>Split procedure</td><td>Random permutation</td></tr><tr><td>Checkpoint evaluated</td><td>Final pretraining checkpoint</td></tr></table>

Sampling is source-stratified with replacement. Weights multiply mean losses; final-horizon terms are additional. The effective entropy weight is 0.01: $\lambda _ { \mathrm { r e p } } = 0 . 0 5 , \lambda _ { \mathrm { e n t } } = 0 . 2 .$ . The split is by example, not trajectory.

For each behavior-policy source, we collect 128 trajectories of 25 steps using eight parallel environments.

TABLE V  
OFFLINE BEHAVIOR-POLICY CATEGORIES.
<table><tr><td>Source</td><td>Category definition</td></tr><tr><td>Random</td><td>Uniform action sampling (R).</td></tr><tr><td>MAPPO</td><td>Initial, medium (M), and expert: 0%, 30%, and 100% of the training budget.</td></tr><tr><td></td><td>Decentralized IPPO Weak, medium (M), and expert labels assigned from existing performance logs.</td></tr></table>

## B. Task and Evaluation Protocol

Table VI presents the number of agents, online training budget, and random and expert return references for each task. TABLE VI

TASK BUDGETS AND RAW NORMALIZATION REFERENCES.
<table><tr><td>Task</td><td></td><td>Agents Timesteps Random</td><td></td><td>Expert</td></tr><tr><td>Tag</td><td>4</td><td>5M</td><td>-0.57</td><td>36.12</td></tr><tr><td>Spread</td><td>3</td><td>5M</td><td>-79.14</td><td>-28.46</td></tr><tr><td>Adversary</td><td>3</td><td>10M</td><td>-13.26</td><td>-1.75</td></tr></table>

We evaluate each method using 30 matched seeds. For each seed, we average the return logged over the last 200 policy updates to obtain its final raw score. Let $J _ { m , e , s }$ denote this score for method m, task e, and seed s. To compare tasks with different reward scales, we normalize each score using the fixed references in Table VI, following the formula used in D4RL [48]:

$$
S _ { m , e , s } = 1 0 0 \frac { J _ { m , e , s } - J _ { \mathrm { r a n d o m } , e } } { J _ { \mathrm { e x p e r t } , e } - J _ { \mathrm { r a n d o m } , e } } .\tag{8}
$$

We report the mean and standard deviation of these scores across seeds. The Total Average is the mean of the three task means. Scores above 100 exceed the corresponding expert reference.