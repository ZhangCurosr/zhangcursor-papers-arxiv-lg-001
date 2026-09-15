# Safe Meta-Reinforcement Learning via Information Space Reachability

Zeyang Li, Sunbochen Tang, Navid Azizan

Abstract— Meta-reinforcement learning (meta-RL) enables agents to adapt to unseen tasks with limited experience. Despite its promise, the application of meta-RL in real-world tasks is hindered by safety requirements, which have been underexplored in prior work. In this paper, we propose a safe meta-RL framework that explicitly accounts for safety during adaptation. Our key insight is to reason about safety in the information space, which captures both the physical state and the agent’s belief over the underlying task. Within this space, we introduce a safety value function that measures the probability of the agent avoiding unsafe regions indefinitely. We show that this function satisfies a self-consistency condition and a Bellman equation, which make it learnable via meta-RL. Based on this formulation, we develop a safe meta-RL algorithm that learns the safety value function and leverages it for safety filtering and constrained policy optimization. Experiments on meta-RL benchmarks demonstrate the effectiveness of the proposed method.

## I. INTRODUCTION

Meta-reinforcement learning (meta-RL) [1] is a powerful paradigm for learning to learn in sequential decision-making problems. It enables agents to adapt quickly to previously unseen tasks using only limited data [2], [3]. Specifically, an agent is trained over a distribution of tasks and learns an effective adaptation strategy, namely a meta-policy, that can be transferred to new tasks drawn from a similar distribution. This allows the agent to adapt to a new task with only a small amount of experience, significantly improving efficiency compared with learning a new policy from scratch.

Existing meta-RL methods can be broadly categorized according to how they achieve rapid adaptation. Gradientbased approaches, exemplified by MAML [4], formulate learning to learn as a bilevel optimization problem, in which the meta-learner acquires an initialization that can be quickly adapted to a new task through a small number of gradient updates. More recently, latent-belief approaches such as PEARL [5] and VariBAD [6] adopt a Bayes-adaptive perspective, inferring a latent representation of the underlying task from observed experience and conditioning the policy on this inferred representation. By explicitly modeling task uncertainty, these methods enable efficient adaptation without requiring online gradient updates, and have demonstrated strong empirical performance across a range of meta-RL benchmarks.

Despite the widely recognized effectiveness of meta-RL, its application to real-world decision-making problems remains limited, largely due to safety requirements. In many practical settings, success depends not only on reward maximization but also on maintaining safety; otherwise, catastrophic failures may occur. This motivates the study of safe meta-RL, which aims to learn a safe meta-policy that can adapt to new tasks such that the resulting policy not only achieves task objectives, such as reward maximization, but also satisfies safety constraints.

Existing safe meta-RL methods have notable limitations. First, most existing works adopt the constrained Markov decision process (CMDP) framework [7], in which the constraint requires the expected trajectory cost to remain below a predefined threshold. This formulation is often ill-suited to real-world applications, since constraints are enforced only in expectation, allowing the agent to violate safety requirements at particular states. Khattar et al. [8] formulate safe meta-RL in a CMDP-within-online framework and establish taskaveraged guarantees for reward maximization and constraint satisfaction. Cho and Sun [9] employ successive convexconstrained policy updates across multiple tasks via differentiable convex programming. Xu and Zhu [10] propose safe policy adaptation and safe meta-policy training, and establish an anytime safety guarantee for policy adaptation. Second, these approaches largely inherit the optimizationcentric perspective of methods such as MAML [4], where both learning and adaptation take place in the policy parameter space. Although this viewpoint is conceptually clean and theoretically appealing, it has been shown to be inefficient for complex control tasks [5], [6], due to expensive inner-loop optimization, on-policy data requirements, and the inability to explicitly capture task uncertainty, which is only indirectly reflected in policy parameters.

In this paper, we propose a novel safe meta-RL framework that explicitly addresses these limitations. First, we consider the state-wise constraint formulation, which requires the agent to satisfy safety constraints at every state it visits [11]. Second, we adopt the Bayes-adaptive perspective in meta-RL [5], [6] and reason about safety in the information space, rather than in the policy parameter space. The information space is defined by the physical state together with the posterior belief over tasks. We then formulate safety preservation as a reachability problem in this space. This viewpoint offers both conceptual and practical advantages. It makes explicit that safety under task uncertainty is inherently belief-dependent: the same physical state may admit different safety guarantees under different task posteriors. By learning safety-preserving behavior directly in information space, the agent can better handle task uncertainty, leading to policies that are both safe and performant on meta-RL tasks. The main contributions of this paper are summarized as follows.

• We formulate meta-RL for safety preservation as a Bayes-adaptive reachability problem in information space. By introducing safety value functions and establishing their self-consistency conditions and Bellman equations, we develop a meta-RL framework for statewise safety.

• We propose a practical safe meta-RL algorithm for complex high-dimensional systems, in which value functions and policies are approximated with neural networks, and belief updates are approximated by a neural encoder that infers latent task representations from online interactions.

• We demonstrate the effectiveness of the proposed algorithm on widely used meta-RL benchmarks.

## II. PROBLEM STATEMENT

Consider a family of Markov decision processes (MDPs),

$$
\boldsymbol { \mathcal { M } } _ { z } = ( \mathcal { X } , \mathcal { U } , P _ { z } , r _ { z } , h _ { z } , \gamma ) ,
$$

parameterized by a task descriptor $z \in { \mathcal { Z } }$ . Here, $\mathcal { Z }$ denotes the task space, X denotes the state space, and $\mathcal { U }$ denotes the action space. For each task z, $P _ { z } ( \cdot | x , u )$ defines the transition kernel on $\mathcal { X } , r _ { z } : \mathcal { X } \times \mathcal { U }  \mathbb { R }$ defines the reward function, and $h _ { z } : \mathcal { X } $ R defines the constraint function. $\gamma \in ( 0 , 1 )$ is the reward discount factor.

We consider a meta-reinforcement learning (meta-RL) setting, where a task $z \in \mathcal { Z }$ is sampled from a prior distribution $p ( z )$ at the beginning of each episode and then remains fixed for the duration of that episode. For clarity, we assume throughout the theoretical development that ${ \mathcal { Z } } ,$ $x ,$ and U are finite. The extension to continuous spaces is standard.

Given a sampled task $z ,$ the environment evolves according to the transition kernel $x _ { t + 1 } \sim P _ { z } ( \cdot | x _ { t } , u _ { t } )$ , with instantaneous reward $r _ { t } ~ = ~ r _ { z } ( x _ { t } , u _ { t } )$ and constraint value $h _ { t } = h _ { z } ( x _ { t } )$ . The goal of meta-RL is to leverage experience collected across training tasks drawn from $p ( z )$ in order to learn a controller that can rapidly adapt to a new task using online interaction. Concretely, at each decision time t, the policy may depend on the interaction history observed so far. Let

$$
c _ { t } = ( x _ { 0 } , h _ { 0 } , u _ { 0 } , r _ { 0 } , \cdot \cdot \cdot ~ , x _ { t - 1 } , h _ { t - 1 } , u _ { t - 1 } , r _ { t - 1 } , x _ { t } , h _ { t } )
$$

denote the context available up to time t. A history-dependent policy π then selects actions according to $c _ { t }$ . The standard meta-RL objective is to maximize the task-averaged discounted cumulative reward

$$
J ( \pi ) = \mathbb { E } _ { z \sim p ( z ) , \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } r _ { z } ( x _ { t } , u _ { t } ) \right] .
$$

In this paper, we additionally require safety, in the sense that the state constraints $h _ { z } ( x _ { t } ) \geq 0$ are satisfied for all t with high probability under the task distribution and the policy.

At the theoretical level, meta-RL can be formulated $\mathbf { \vec { e } X \tilde { \mathbf { \theta } } } -$ actly as a Bayes-adaptive control problem [12], [13]. Define the posterior belief over tasks by $b _ { t } ( \boldsymbol { z } ) = \mathbb { P } ( \boldsymbol { z } | \boldsymbol { c } _ { t } )$ , which we refer to as the belief state. Here, $b _ { t } \in \Delta ( \mathcal { Z } )$ is a probability distribution over the task space $\mathcal { Z } .$ . The information state is defined as the pair consisting of the physical state and the belief state:

$$
s _ { t } = ( x _ { t } , b _ { t } ) \in \mathcal S = \mathcal X \times \Delta ( \mathcal Z ) .
$$

The information state $s _ { t }$ constitutes an exact sufficient statistic for decision-making under task uncertainty: there exists a deterministic Markov policy on $s$ that achieves optimality.

Let O denote the post-action observation space, and define the post-action observation at time t by

$$
o _ { t + 1 } = ( r _ { t } , x _ { t + 1 } , h _ { t + 1 } ) \in \mathcal { O } .
$$

The task-conditioned observation kernel is

$$
\begin{array} { l } { { O _ { z } ( r , x ^ { + } , h ^ { + } | x , u ) } } \\ { { \ } } \\ { { = P _ { z } ( x ^ { + } | x , u ) { \bf 1 } \{ r = r _ { z } ( x , u ) \} { \bf 1 } \{ h ^ { + } = h _ { z } ( x ^ { + } ) \} } . } \end{array}\tag{1}
$$

The exact posterior update is then given by $\mathrm { \Delta B a y e s ^ { \prime } }$ rule

$$
b _ { t + 1 } ( z ) = \frac { O _ { z } ( o _ { t + 1 } | x _ { t } , u _ { t } ) b _ { t } ( z ) } { \sum _ { \bar { z } \in \mathcal { Z } } O _ { \bar { z } } ( o _ { t + 1 } | x _ { t } , u _ { t } ) b _ { t } ( \bar { z } ) } .
$$

Although the Bayes-adaptive formulation provides a conceptually clean characterization of the meta-RL problem, maintaining the exact posterior $b _ { t }$ is generally intractable in complex settings. As a result, an important class of modern meta-RL algorithms approximates the posterior update via amortized inference from trajectory context. Specifically, one introduces an encoder $q _ { \phi } ( \xi | c _ { t } )$ and uses a finite-dimensional parameterization as a proxy for the belief $b _ { t } .$ In this paper, we adopt this perspective: the exact safety analysis is carried out in the information state $\left( { { x } _ { t } } , { { b } _ { t } } \right)$ , whereas the practical method replaces $b _ { t }$ with a learned latent representation inferred online from context.

## III. REACHABILITY ANALYSIS ON INFORMATION SPACE

In this section, we develop a meta-RL framework to characterize and optimize safety directly over the information space $\begin{array} { r } { S = \mathcal { X } \times \Delta ( \mathcal { Z } ) } \end{array}$ . As discussed earlier, it is necessary to reason on the information state $\boldsymbol { s } = \left( \boldsymbol { x } , \boldsymbol { b } \right)$ , rather than the physical state x alone, since the belief b encodes epistemic uncertainty about the underlying task and therefore fundamentally affects the safety guarantees that can be established.

Define the task-dependent safety indicator as $g _ { z } ( x ) \ =$ ${ \bf 1 } \{ h _ { z } ( x ) { \mathrm { ~  ~ \geq ~ } } 0 \}$ and the corresponding information-state safety indicator as $\begin{array} { r } { g ( s ) \ = \ \sum _ { z \in \mathcal { Z } } b ( z ) g _ { z } ( x ) } \end{array}$ . Since the current constraint value $h _ { t }$ is part of the context $c _ { t } .$ , every task in the support of $b _ { t }$ is consistent with the observed current safety label. Therefore $g ( s _ { t } ) \in \{ 0 , 1 \}$ for every reachable information state. Accordingly, all statements below are on the reachable part of S.

We consider stochastic policies $\pi : S \to \Delta ( { \mathcal { U } } )$ . For $s =$ $( x , b ) \in S$ , define the predictive observation kernel by

$$
O ( o ^ { + } | s , u ) = \sum _ { z \in \mathcal { Z } } b ( z ) O _ { z } ( o ^ { + } | x , u ) ,
$$

where $O _ { z }$ is the task-conditioned observation kernel in (1). The Bayesian update map is

$$
\Phi ( b , x , u , o ^ { + } ) ( z ) = \frac { O _ { z } ( o ^ { + } | x , u ) b ( z ) } { \sum _ { \bar { z } \in \mathcal { Z } } O _ { \bar { z } } ( o ^ { + } | x , u ) b ( \bar { z } ) } .
$$

The corresponding next information state is

$$
F ( s , u , o ^ { + } ) = ( x ^ { + } , \Phi ( b , x , u , o ^ { + } ) ) ,
$$

where $o ^ { + } = ( r , x ^ { + } , h ^ { + } )$

Fix $s = ( x , b ) \in \mathcal { S }$ and $u \in \mathcal { U } .$ . Under a stochastic policy π, let $\mathbb { P } _ { s , u } ^ { \pi }$ and $\mathbb { E } _ { s , u } ^ { \pi }$ denote the probability law and expectation of the controlled process defined as follows: a task descriptor $z \sim b$ is drawn once and kept fixed; $s _ { 0 } = s$ and $u _ { 0 } = u ;$ conditional on $z , x _ { t } , u _ { t } .$ , the post-action observation $o _ { t + 1 }$ is sampled according to $O _ { z } ( \cdot | x _ { t } , u _ { t } )$ ; the next information state is $s _ { t + 1 } = F ( s _ { t } , u _ { t } , o _ { t + 1 } )$ ; and for all $t \geq 1$ , the action $u _ { t }$ is sampled from $\pi ( \cdot | s _ { t } )$

Define the first-violation time by

$$
\tau = \operatorname* { i n f } \{ t \geq 0 : \ h _ { z } ( x _ { t } ) < 0 \} = \operatorname* { i n f } \{ t \geq 0 : \ g ( s _ { t } ) = 0 \} ,
$$

with the convention $\tau = \infty$ if the trajectory remains safe forever. The quantity of primary interest is the probability that the process never reaches the unsafe region. This leads naturally to the following safety value functions.

Definition 1 (safety value functions): For a stochastic policy π, define its safety action-value function by

$$
Q _ { h } ^ { \pi } ( s , u ) = \mathbb { P } _ { s , u } ^ { \pi } ( \tau = \infty ) = \mathbb { E } _ { s , u } ^ { \pi } \left[ \prod _ { t = 0 } ^ { \infty } g ( s _ { t } ) \right] .\tag{2}
$$

The associated safety state-value function is

$$
V _ { h } ^ { \pi } ( s ) = \sum _ { u \in \mathcal { U } } \pi ( u | s ) Q _ { h } ^ { \pi } ( s , u ) .
$$

The optimal safety values are defined by $Q _ { h } ^ { * } ( s , u ) \ =$ $\operatorname* { s u p } _ { \pi } Q _ { h } ^ { \pi } ( s , u )$ and $V _ { h } ^ { * } ( s ) = \operatorname* { s u p } _ { \pi } V _ { h } ^ { \pi } ( s )$

Equation (2) is the probability, under the posterior belief b, that safety is maintained for all time after taking action u at information state s and then following π. By maximizing this quantity, we obtain policies that are optimal for rendering the system safe, together with the corresponding optimal values.

The safety value function admits a self-consistency condition: once the current safety label and the next observation are revealed, the remainder of the problem has exactly the same form as the original one. This idea is inspired by Hamilton–Jacobi reachability [14] and its RL formulations [15], [16].

Proposition 1 (safety self-consistency condition): For every stochastic policy π and every $( s , u ) \in \mathcal { S } \times \mathcal { U }$

$$
Q _ { h } ^ { \pi } ( s , u ) = g ( s ) \sum _ { o ^ { + } \in \mathcal { O } } O ( o ^ { + } | s , u ) V _ { h } ^ { \pi } ( F ( s , u , o ^ { + } ) ) .\tag{3}
$$

Proof: By definition,

$$
Q _ { h } ^ { \pi } ( s , u ) = \mathbb { E } _ { s , u } ^ { \pi } \left[ \prod _ { t = 0 } ^ { \infty } g ( s _ { t } ) \right] = g ( s ) \mathbb { E } _ { s , u } ^ { \pi } \left[ \prod _ { t = 1 } ^ { \infty } g ( s _ { t } ) \right] .
$$

Conditioning on the next observation $o _ { 1 } = o ^ { + }$ gives

$$
\begin{array} { l } { Q _ { h } ^ { \pi } ( s , u ) } \\ { \displaystyle = g ( s ) \sum _ { o ^ { + } \in \mathcal O } O ( o ^ { + } | s , u ) \mathbb { E } _ { s , u } ^ { \pi } \left[ \prod _ { t = 1 } ^ { \infty } g ( s _ { t } ) \ \Bigg | \ o _ { 1 } = o ^ { + } \right] . } \end{array}
$$

Once $o _ { 1 } = o ^ { + }$ is revealed, the next information state is $s _ { 1 } =$ $F ( s , u , o ^ { + } )$ , and the future control law is again π, so

$$
\mathbb { E } _ { s , u } ^ { \pi } \left[ \prod _ { t = 1 } ^ { \infty } g ( s _ { t } ) \ \Bigg | \ o _ { 1 } = o ^ { + } \right] = V _ { h } ^ { \pi } ( F ( s , u , o ^ { + } ) ) .
$$

Substituting this identity proves (3).

The self-consistency condition immediately suggests a Bellman-type characterization for the optimal safety values. The next theorem shows that the optimal perpetual-safety probability can indeed be computed recursively on information space.

Theorem 1 (safety Bellman equation): The optimal safety values satisfy the following recursive characterization:

$$
\left\{ \begin{array} { l } { { Q _ { h } ^ { \ast } ( s , u ) = g ( s ) \sum _ { o ^ { + } \in \mathcal { O } } O \left( o ^ { + } | s , u \right) V _ { h } ^ { \ast } \left( F ( s , u , o ^ { + } ) \right) } } \\ { { V _ { h } ^ { \ast } ( s ) = \operatorname* { m a x } _ { u \in \mathcal { U } } Q _ { h } ^ { \ast } ( s , u ) . } } \end{array} \right.\tag{4}
$$

Proof: Fix $( s , u ) \in \mathcal { S } \times \mathcal { U } .$ . By Proposition 1, for every stochastic policy π,

$$
Q _ { h } ^ { \pi } ( s , u ) = g ( s ) \sum _ { o ^ { + } \in \mathcal { O } } O ( o ^ { + } | s , u ) V _ { h } ^ { \pi } ( F ( s , u , o ^ { + } ) ) .
$$

Since $V _ { h } ^ { \pi } ( F ( s , u , o ^ { + } ) ) \leq V _ { h } ^ { * } ( F ( s , u , o ^ { + } ) )$ for every $o ^ { + }$ , we obtain

$$
Q _ { h } ^ { * } ( s , u ) \leq g ( s ) \sum _ { o ^ { + } \in \mathcal { O } } O ( o ^ { + } | s , u ) V _ { h } ^ { * } ( F ( s , u , o ^ { + } ) ) .\tag{5}
$$

Conversely, conditioned on the first post-action observation $o ^ { + }$ , the remaining problem is again the same safetymaximization problem started from the successor information state $F ( s , u , o ^ { + } )$ . By Bellman’s principle of optimality for the information-state controlled Markov process, an optimal continuation policy must therefore attain value $V _ { h } ^ { * } ( F ( s , u , o ^ { + } ) )$ at each reachable successor state. Hence

$$
Q _ { h } ^ { * } ( s , u ) \geq g ( s ) \sum _ { o ^ { + } \in \mathcal { O } } O ( o ^ { + } | s , u ) V _ { h } ^ { * } ( F ( s , u , o ^ { + } ) ) .\tag{6}
$$

Combining (5) and (6) proves the first equality in (4).

For the state-value function, the first decision at state s is the choice of an action distribution $\mu \in \Delta ( \mathcal { U } )$ . Therefore

$$
V _ { h } ^ { * } ( s ) = \operatorname* { s u p } _ { \mu \in \Delta ( \mathcal { U } ) } \sum _ { u \in \mathcal { U } } \mu ( u ) Q _ { h } ^ { * } ( s , u ) .
$$

Since the right-hand side is linear in $\mu ,$ its supremum over the simplex $\Delta ( \mathcal { U } )$ is attained at an extreme point, i.e., at a deterministic action. Hence $V _ { h } ^ { * } ( s ) = \operatorname* { m a x } _ { u \in \mathcal { U } } Q _ { h } ^ { * } ( s , u )$

The proposed safety value function (2) characterizes perpetual safety. However, the corresponding self-consistency condition (3) and Bellman equation (4) generally do not induce contraction mappings. We therefore introduce a discounted surrogate that retains a clear probabilistic interpretation while yielding contraction mappings and stable fixedpoint characterizations.

Definition 2 (discounted safety value functions): Fix a safety discount factor $\gamma _ { h } \in ( 0 , 1 )$ . For a stochastic policy π, define the discounted safety action-value function by

$$
Q _ { h , \gamma _ { h } } ^ { \pi } \left( s , u \right) = ( 1 - \gamma _ { h } ) \mathbb { E } _ { s , u } ^ { \pi } \left[ \sum _ { t = 0 } ^ { \infty } \gamma _ { h } ^ { t } \prod _ { k = 0 } ^ { t } g ( s _ { k } ) \right] .\tag{7}
$$

The discounted safety state-value function is

$$
V _ { h , \gamma _ { h } } ^ { \pi } \left( s \right) = \sum _ { u \in \mathcal { U } } \pi ( u | s ) Q _ { h , \gamma _ { h } } ^ { \pi } ( s , u ) .
$$

The optimal discounted safety values are defined by $\begin{array} { l l l } { Q _ { h , \gamma _ { h } } ^ { * } ( s , u ) } & { = } & { \operatorname* { s u p } _ { \pi } Q _ { h , \gamma _ { h } } ^ { \pi } ( s , u ) } \end{array}$ and $\begin{array} { r l } { V _ { h , \gamma _ { h } } ^ { * } ( s ) } & { { } = } \end{array}$ sup<sub>π</sub> $V _ { h , \gamma _ { h } } ^ { \pi ^ { * } } ( s )$

The discounted quantity has an interesting probabilistic interpretation. Suppose that, independently of the task, the episode terminates after each step with probability $1 - \gamma _ { h }$ . If $T$ denotes the resulting geometric horizon, then $Q _ { h , \gamma _ { h } } ^ { \pi } ( s , u )$ is exactly the probability that the trajectory remains safe throughout the episode, i.e., that no safety violation occurs before termination. The next proposition makes this interpretation precise.

Proposition 2 (geometric-horizon interpretation): Let T be independent of the controlled process and geometrically distributed on $\{ 0 , 1 , 2 , \ldots \}$ with $\mathbb { P } ( T = t ) = ( 1 - \gamma _ { h } ) \gamma _ { h } ^ { t }$ Then, for every (s, u),

$$
Q _ { h , \gamma _ { h } } ^ { \pi } ( s , u ) = \mathbb { P } _ { s , u } ^ { \pi } ( \tau > T ) .
$$

Proof: For each $t \geq 0 ,$ , we have

$$
\prod _ { k = 0 } ^ { t } g ( s _ { k } ) = \mathbf { 1 } \{ \tau > t \} .
$$

Substituting this identity into (7) yields

$$
\begin{array} { r l } & { \displaystyle Q _ { h , \gamma _ { h } } ^ { \pi } ( s , u ) = ( 1 - \gamma _ { h } ) \sum _ { t = 0 } ^ { \infty } \gamma _ { h } ^ { t } \mathbb { P } _ { s , u } ^ { \pi } ( \tau > t ) } \\ & { \quad \quad \quad \quad = \displaystyle \sum _ { t = 0 } ^ { \infty } \mathbb { P } ( T = t ) \mathbb { P } _ { s , u } ^ { \pi } ( \tau > t ) } \\ & { \quad \quad \quad = \mathbb { P } _ { s , u } ^ { \pi } ( \tau > T ) , } \end{array}
$$

where the last equality uses the independence of T and the controlled process. ■

The discounted analogue of the self-consistency condition has the same basic structure as in the undiscounted case, except that the current safety indicator contributes an immediate term weighted by $1 - \gamma _ { h }$

Proposition 3 (discounted self-consistency condition): For every stochastic policy π and every $( s , u ) \in \mathcal { S } \times \mathcal { U } _ { : }$

$$
\begin{array} { r l } & { \displaystyle Q _ { h , \gamma _ { h } } ^ { \pi } \left( s , u \right) = \left( 1 - \gamma _ { h } \right) g ( s ) } \\ & { \quad + \gamma _ { h } g ( s ) \displaystyle \sum _ { o ^ { + } \in \mathcal { O } } O ( o ^ { + } \vert s , u ) V _ { h , \gamma _ { h } } ^ { \pi } ( F ( s , u , o ^ { + } ) ) . } \\ & { \displaystyle P r o o f ; ~ \mathrm { S t a r t i n g ~ f r o m } \left( 7 \right) , } \end{array}\tag{8}
$$

$$
\begin{array} { l } { { Q _ { h , \gamma _ { h } } ^ { \pi } ( s , u ) = ( 1 - \gamma _ { h } ) \mathbb { E } _ { s , u } ^ { \pi } \left[ g ( s _ { 0 } ) + \displaystyle \sum _ { t = 1 } ^ { \infty } \gamma _ { h } ^ { t } \displaystyle \prod _ { k = 0 } ^ { t } g ( s _ { k } ) \right] } } \\ { { = ( 1 - \gamma _ { h } ) g ( s ) + \gamma _ { h } g ( s ) \left( 1 - \gamma _ { h } \right) \mathbb { E } _ { s , u } ^ { \pi } \left[ \displaystyle \sum _ { \ell = 0 } ^ { \infty } \gamma _ { h } ^ { \ell } \displaystyle \prod _ { k = 1 } ^ { \ell + 1 } g ( s _ { k } ) \right] . } } \end{array}
$$

Conditioning on the next observation $o _ { 1 } = o ^ { + }$ gives

$$
\begin{array} { r l } & { ( 1 - \gamma _ { h } ) \mathbb { E } _ { s , u } ^ { \pi } \left[ \displaystyle \sum _ { \ell = 0 } ^ { \infty } \gamma _ { h } ^ { \ell } \prod _ { k = 1 } ^ { \ell + 1 } g ( s _ { k } ) \ \Bigg | \ o _ { 1 } = o ^ { + } \right] } \\ & { = V _ { h , \gamma _ { h } } ^ { \pi } ( F ( s , u , o ^ { + } ) ) . } \end{array}
$$

Averaging with respect to $O ( o ^ { + } | s , u )$ proves (8).

We now turn to optimality. As in the undiscounted case, the optimal discounted safety values satisfy a Bellman equation.

Theorem 2 (discounted safety Bellman equation): The optimal discounted safety values satisfy the following recursive characterization:

$$
\left\{ \begin{array} { l l } { \displaystyle Q _ { h , \gamma _ { h } } ^ { * } ( s , u ) = \left( 1 - \gamma _ { h } \right) g ( s ) } \\ { \displaystyle + \gamma _ { h } g ( s ) \sum _ { o ^ { + } \in \mathcal { O } } O \left( o ^ { + } | s , u \right) V _ { h , \gamma _ { h } } ^ { * } \left( F ( s , u , o ^ { + } ) \right) } \\ { \displaystyle V _ { h , \gamma _ { h } } ^ { * } ( s ) = \operatorname* { m a x } _ { u \in \mathcal { U } } Q _ { h , \gamma _ { h } } ^ { * } ( s , u ) . } \end{array} \right.
$$

Proof: The proof follows a similar idea to that of the undiscounted safety Bellman equation, so we omit it here.

The discounted formulation is especially convenient since it admits fixed-point characterizations. These operators also motivate the critic and actor updates used in the practical algorithm later.

Definition 3 (safety operators): Let $Q : S \times \mathcal { U }  \mathbb { R }$ be a bounded function. Define the safety self-consistency operator as

$$
\begin{array} { l } { { ( { \mathcal T } _ { h , \gamma _ { h } } ^ { \pi } Q ) ( s , u ) = \left( 1 - \gamma _ { h } \right) g ( s ) + \gamma _ { h } g ( s ) \displaystyle \sum _ { o ^ { + } \in { \mathcal O } } O ( o ^ { + } \vert s , u ) } } \\ { { \displaystyle \sum _ { u ^ { + } \in { \mathcal U } } \pi ( u ^ { + } \vert F ( s , u , o ^ { + } ) ) Q ( F ( s , u , o ^ { + } ) , u ^ { + } ) } , } \end{array}
$$

and the safety Bellman operator as

$$
\begin{array} { r l } {  { ( \mathcal T _ { h , \gamma _ { h } } Q ) ( s , u ) = ( 1 - \gamma _ { h } ) g ( s ) + \gamma _ { h } g ( s ) } } \\ & { \sum _ { o ^ { + } \in \mathcal O } O ( o ^ { + } | s , u ) \operatorname* { m a x } _ { u ^ { + } \in \mathcal U } Q ( F ( s , u , o ^ { + } ) , u ^ { + } ) . } \end{array}\tag{9}
$$

Based on these definitions, the discounted self-consistency condition and Bellman equation can be written compactly as $Q _ { h , \gamma _ { h } } ^ { \pi } ~ = ~ T _ { h , \gamma _ { h } } ^ { \pi } Q _ { h , \gamma _ { h } } ^ { \pi }$ and $Q _ { h , \gamma _ { h } } ^ { * } ~ = ~ { \cal T } _ { h , \gamma _ { h } } Q _ { h , \gamma _ { h } } ^ { * }$ . The next result shows that these operators are monotone contractions, which guarantees uniqueness of the corresponding fixed points.

Theorem 3 (monotone contractions): Let $Q , \widetilde { Q } : { \cal S } \times \mathcal { U } $ R be bounded functions. Then:

1) If $Q \leq { \widetilde { Q } }$ , then $\begin{array} { r } { \mathcal { T } _ { h , \gamma _ { h } } ^ { \pi } Q \ \leq \ \mathcal { T } _ { h , \gamma _ { h } } ^ { \pi } \widetilde Q } \end{array}$ and $\mathcal { T } _ { h , \gamma _ { h } } Q \ \leq$ ${ \mathcal { T } } _ { h , \gamma _ { h } } { \widetilde { Q } } .$

2) The operators $\mathcal { T } _ { h , \gamma _ { h } } ^ { \pi }$ and $\mathcal { T } _ { h , \gamma _ { h } }$ are γ<sub>h</sub>-contractions in the infinity norm:

$$
\begin{array} { r } { \left. \mathcal { T } _ { h , \gamma _ { h } } ^ { \pi } Q - \mathcal { T } _ { h , \gamma _ { h } } ^ { \pi } \widetilde { Q } \right. _ { \infty } \leq \gamma _ { h } \left. Q - \widetilde { Q } \right. _ { \infty } , } \end{array}\tag{10}
$$

and

$$
\begin{array} { r } { \left. \mathcal { T } _ { h , \gamma _ { h } } Q - \mathcal { T } _ { h , \gamma _ { h } } \widetilde { Q } \right. _ { \infty } \leq \gamma _ { h } \left. Q - \widetilde { Q } \right. _ { \infty } . } \end{array}\tag{11}
$$

Consequently, each operator has a unique fixed point. In particular, the unique fixed points are $Q _ { h , \gamma _ { h } } ^ { \pi }$ for $\mathcal { T } _ { h , \gamma _ { h } } ^ { \pi }$ and $Q _ { h , \gamma _ { h } } ^ { * }$ for $\mathcal { T } _ { h , \gamma _ { h } }$

Proof: Monotonicity is straightforward. We prove contraction below. Fix $( s , u ) \ \in \ S \times \mathcal { U } .$ For the evaluation

operator,

$$
\begin{array} { r l } & { \displaystyle | ( \mathcal { T } _ { h , \gamma _ { h } } ^ { \pi } Q ) ( s , u ) - ( \mathcal { T } _ { h , \gamma _ { h } } ^ { \pi } \widetilde { Q } ) ( s , u ) | } \\ & { = \gamma _ { h } g ( s ) | \displaystyle \sum _ { o + \in \mathcal { O } } Q ( o ^ { + } | s , u ) \displaystyle \sum _ { u ^ { + } \in \mathcal { U } } \pi ( u ^ { + } | F ( s , u , o ^ { + } ) )  } \\ & { \qquad \cdot ( Q ( F ( s , u , o ^ { + } ) , u ^ { + } ) - \widetilde Q ( F ( s , u , o ^ { + } ) , u ^ { + } ) )  } \\ & { \displaystyle  \sum _ { o + \in \mathcal { O } } Q ( o ^ { + } | s , u ) \displaystyle \sum _ { u ^ { + } \in \mathcal { U } } \pi ( u ^ { + } | F ( s , u , o ^ { + } ) ) \| Q - \widetilde Q \| _ { \infty } } \\ & { \displaystyle \leq \gamma _ { h } \| Q - \widetilde Q \| _ { \infty } . } \end{array}
$$

Taking the supremum over (s, u) proves (10).

For the optimality operator, we use the elementary inequality

$$
\left| \operatorname* { m a x } _ { u ^ { + } } a _ { u ^ { + } } - \operatorname* { m a x } _ { u ^ { + } } \tilde { a } _ { u ^ { + } } \right| \leq \operatorname* { m a x } _ { u ^ { + } } | a _ { u ^ { + } } - \tilde { a } _ { u ^ { + } } | .
$$

Applying this to (9), we obtain

$$
\begin{array} { l } { \displaystyle \left| ( \mathcal { T } _ { h , \gamma _ { h } } Q ) ( s , u ) - ( \mathcal { T } _ { h , \gamma _ { h } } \widetilde { Q } ) ( s , u ) \right| } \\ { \displaystyle \leq \gamma _ { h } g ( s ) \sum _ { \sigma ^ { + } \in \mathcal { O } } O ( o ^ { + } | s , u ) } \\ { \displaystyle \operatorname* { m a x } _ { u ^ { + } \in \mathcal { U } } \left| Q ( F ( s , u , \sigma ^ { + } ) , u ^ { + } ) - \widetilde { Q } ( F ( s , u , \sigma ^ { + } ) , u ^ { + } ) \right| } \\ { \displaystyle \leq \gamma _ { h } \left\| Q - \widetilde { Q } \right\| _ { \infty } . } \end{array}
$$

Taking the supremum over $( s , u )$ proves (11). Banach’s fixed-point theorem then gives uniqueness of the fixed points.

The discounted quantity $Q _ { h , \gamma _ { h } } ^ { \pi }$ is introduced for algorithmic and analytical convenience, but it remains faithful to the original perpetual-safety objective. The next result shows that, for every fixed policy, the discounted surrogate converges to the original value as $\gamma _ { h }  1$

Theorem 4 (recovery of the undiscounted safety value): For every stochastic policy $\pi$ and every $( s , u ) \in \mathcal { S } \times \mathcal { U } _ { : }$

$$
\operatorname* { l i m } _ { \gamma _ { h }  1 } Q _ { h , \gamma _ { h } } ^ { \pi } ( s , u ) = Q _ { h } ^ { \pi } ( s , u ) .\tag{12}
$$

Consequently, for every $s \in S$

$$
\operatorname* { l i m } _ { \gamma _ { h } \to 1 } V _ { h , \gamma _ { h } } ^ { \pi } ( s ) = V _ { h } ^ { \pi } ( s ) .
$$

Proof: Fix a stochastic policy π and an initial pair $( s , u )$ . Define

$$
a _ { t } = \mathbb { P } _ { s , u } ^ { \pi } \big ( \tau > t \big ) = \mathbb { E } _ { s , u } ^ { \pi } \left[ \prod _ { k = 0 } ^ { t } g ( s _ { k } ) \right] , \qquad t \ge 0 .
$$

Since the events $\{ \tau > t \}$ are decreasing in t, the sequence $( a _ { t } ) _ { t \geq 0 }$ is nonincreasing and bounded in [0, 1]. Moreover, by continuity of probability for decreasing events,

$$
\operatorname* { l i m } _ { t \to \infty } a _ { t } = \mathbb { P } _ { s , u } ^ { \pi } \bigg ( \bigcap _ { t = 0 } ^ { \infty } \{ \tau > t \} \bigg ) = \mathbb { P } _ { s , u } ^ { \pi } ( \tau = \infty ) = Q _ { h } ^ { \pi } ( s , u ) .\tag{13}
$$

On the other hand, by (7),

$$
Q _ { h , \gamma _ { h } } ^ { \pi } ( s , u ) = ( 1 - \gamma _ { h } ) \sum _ { t = 0 } ^ { \infty } \gamma _ { h } ^ { t } a _ { t } .
$$

Let $a _ { \infty } : = Q _ { h } ^ { \pi } ( s , u )$ . Given $\varepsilon > 0 ,$ , choose N such that $| a _ { t } - a _ { \infty } | \leq \varepsilon$ for all $t \geq N$ , which is possible by (13). Then

$$
\begin{array} { r l } & { \displaystyle \left| { { \cal Q } } _ { n , \gamma _ { h } } ^ { n } ( s , u ) - a _ { \infty } \right| } \\ & { = \displaystyle \left| ( 1 - \gamma _ { h } ) \displaystyle \sum _ { t = 0 } ^ { \infty } \gamma _ { h } ^ { t } ( a _ { t } - a _ { \infty } ) \right| } \\ & { \leq \displaystyle ( 1 - \gamma _ { h } ) \sum _ { t = 0 } ^ { N - 1 } \gamma _ { h } ^ { t } | a _ { t } - a _ { \infty } | + ( 1 - \gamma _ { h } ) \displaystyle \sum _ { t = N } ^ { \infty } \gamma _ { h } ^ { t } | a _ { t } - a _ { \infty } | } \\ & { \leq \displaystyle ( 1 - \gamma _ { h } ) \sum _ { t = 0 } ^ { N - 1 } | a _ { t } - a _ { \infty } | + \varepsilon ( 1 - \gamma _ { h } ) \displaystyle \sum _ { t = N } ^ { \infty } \gamma _ { h } ^ { t } } \\ & { \leq \displaystyle ( 1 - \gamma _ { h } ) \sum _ { t = 0 } ^ { \infty } | a _ { t } - a _ { \infty } | + \varepsilon . } \end{array}
$$

The first term on the right tends to zero as $\gamma _ { h }  1$ , since it is a finite constant multiplied by $1 - \gamma _ { h }$ . Therefore

$$
\operatorname* { l i m } _ { \gamma _ { h } \to 1 } \left. Q _ { h , \gamma _ { h } } ^ { \pi } ( s , u ) - Q _ { h } ^ { \pi } ( s , u ) \right. \le \varepsilon .
$$

Since $\varepsilon > 0$ is arbitrary, (12) follows. The remaining proof for state-value function is straightforward. ■

The results in this section provide a theoretical foundation for designing meta-RL algorithms that satisfy safety constraints. The discounted safety Bellman equation suggests a policy-iteration-style approach to computing the discounted safety value function. In practice, this naturally leads to an actor-critic algorithm, in which the actor and critic serve as function approximators for the policy and safety value function, respectively.

## IV. ALGORITHM DESIGN

In this section, we present a practical safe meta-RL algorithm that seeks to maximize cumulative reward while satisfying given safety constraints. As noted earlier, exact belief updates are generally intractable. We therefore introduce an encoder $q _ { \phi } \left( \xi | c _ { t } \right)$ that provides a finite-dimensional parameterization serving as a proxy for the belief $b _ { t }$ . Building on this representation, we employ two actor-critic pairs. The first, consisting of a safety actor and a safety critic, is derived from the theoretical framework developed in the previous section and is dedicated to safety preservation. The second is a standard actor-critic pair for task performance, namely reward maximization. In this module, the safety critic is further used as a safety filter and in constrained policy optimization for training the actor.

The encoder follows the probabilistic latent-variable design introduced in [5]. At time t, the context is given by $c _ { t } \ = \ ( x _ { 0 } , h _ { 0 } , u _ { 0 } , r _ { 0 } , \ldots , x _ { t - 1 } , h _ { t - 1 } , u _ { t - 1 } , r _ { t - 1 } , x _ { t } , h _ { t } )$ Given this context, the encoder produces an approximate Gaussian posterior over the latent task representation $\xi \sim$ $q _ { \phi } ( \xi | c _ { t } ) = \mathcal { N } ( \mu _ { t } , \Sigma _ { t } )$ , where $\phi$ denotes the parameters of the encoder network. Specifically, taking $c _ { t }$ as input, the network outputs the mean $\mu _ { t }$ and covariance $\Sigma _ { t }$ of the latent representation ξ.

For both the safety critic and the performance critic, we adopt the double Q-network design [17] and use two networks for each. The safety critic networks are denoted by $Q _ { h } ( x , u , \xi ; \psi _ { 1 } )$ and $Q _ { h } ( x , u , \xi ; \psi _ { 2 } )$ , while the performance critic networks are denoted by $Q ( x , u , \xi ; \omega _ { 1 } )$ and $Q ( x , u , \xi ; \omega _ { 2 } )$ . For a cleaner presentation, we use the shorthand notations $\begin{array} { r } { Q _ { h } ( x , u , \xi ; \psi ) = \operatorname* { m i n } _ { i \in \{ 1 , 2 \} } Q _ { h } ( x , u , \xi ; \psi _ { i } ) } \end{array}$ and $\begin{array} { r } { Q ( x , u , \xi ; \omega ) = \ \operatorname* { m i n } _ { i \in \{ 1 , 2 \} } Q ( x , u , \xi ; \omega _ { i } ) } \end{array}$ . The same convention applies to the corresponding target networks, whose parameters are denoted by $\hat { \psi } _ { 1 } , \hat { \psi } _ { 2 } , \hat { \omega } _ { 1 }$ , and $\hat { \omega } _ { 2 }$ . Since the safety value lies in [0, 1], we apply a sigmoid activation at the output layer of the safety critic to enforce this range.

We denote the task actor, which is responsible for maximizing task performance, by $\pi ( x , \xi ; \theta )$ , and the safety actor, which is responsible for maximizing the safety value, by $\pi _ { h } ( x , \xi ; \varphi )$ . The task actor is stochastic in order to encourage exploration and improve performance, whereas the safety actor is deterministic to support safety preservation. Accordingly, we write $u \sim \pi ( x , \xi ; \theta )$ and $u = \pi _ { h } ( x , \xi ; \varphi )$

Following the soft actor-critic algorithm [17], the performance critic loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { Q } ( \omega _ { i } ) = \mathbb { E } _ { ( x , u , r , x ^ { \prime } ) \sim \mathcal { B } } \left[ \left( Q ( x , u , \xi ; \omega _ { i } ) - \hat { Q } \right) ^ { 2 } \right] , } \end{array}
$$

where B denotes the replay buffer and the target value is given by

$$
\begin{array} { r } { \hat { Q } = r + \gamma \left( Q ( x ^ { \prime } , u ^ { \prime } , \xi ; \hat { \omega } ) - \alpha \log \pi ( u ^ { \prime } | x ^ { \prime } , \xi ; \theta ) \right) , } \end{array}
$$

with $u ^ { \prime } \sim \pi ( x ^ { \prime } , \xi ; \theta )$ . α denotes the regularization parameter. Based on Definition 3, the safety critic loss is defined as

$$
\begin{array} { r } { \mathcal { L } _ { Q _ { h } } ( \psi _ { i } ) = \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { u } , \boldsymbol { g } , \boldsymbol { x } ^ { \prime } ) \sim \mathcal { B } } \left[ \left( Q _ { h } ( \boldsymbol { x } , \boldsymbol { u } , \xi ; \psi _ { i } ) - \hat { Q } _ { h } \right) ^ { 2 } \right] , } \end{array}
$$

where the target safety value is given by

$$
\hat { Q } _ { h } = ( 1 - \gamma _ { h } ) g + \gamma _ { h } g Q _ { h } ( x ^ { \prime } , u ^ { \prime } , \xi ; \hat { \psi } ) ,
$$

with $u ^ { \prime } = \pi _ { h } ( x ^ { \prime } , \xi ; \varphi )$

The safety actor $\pi _ { h } ( x , \xi ; \varphi )$ is designed to maximize the safety value. Its loss function is therefore defined as

$$
\mathcal { L } _ { \pi _ { h } } ( \varphi ) = - \mathbb { E } _ { x \sim \mathcal { B } , \xi \sim q _ { \phi } ( \xi \vert c _ { t } ) } \left[ Q _ { h } ( x , \pi _ { h } ( x , \xi ; \varphi ) , \xi ; \psi ) \right] .
$$

The task actor aims to maximize reward while satisfying the safety constraint. To this end, we perform constrained policy optimization with the constraint $Q _ { h } ( x , u , \xi ; \psi ) \ge$ $1 - \delta ,$ , where $u \sim \pi ( x , \xi ; \theta )$ and $\delta > 0$ is a small positive constant included for numerical stability. We adopt a primaldual approach, which alternates between updating the task actor parameters and the Lagrange multiplier $\lambda .$ The loss function for the task actor is given by

$$
\begin{array} { r l } & { \mathcal { L } _ { \pi } ( \boldsymbol { \theta } ) = - \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { B } , \boldsymbol { \xi } \sim \boldsymbol { q } _ { \phi } ( \boldsymbol { \xi } | \boldsymbol { c } _ { t } ) } \left[ Q ( \boldsymbol { x } , \boldsymbol { u } , \boldsymbol { \xi } ; \boldsymbol { \omega } ) \right. } \\ & { \qquad \left. + \lambda Q _ { h } ( \boldsymbol { x } , \boldsymbol { u } , \boldsymbol { \xi } ; \boldsymbol { \psi } ) - \alpha \log \pi ( \boldsymbol { u } | \boldsymbol { x } , \boldsymbol { \xi } ; \boldsymbol { \theta } ) \right] , } \end{array}
$$

where $u \sim \pi ( x , \xi ; \theta )$ . The loss function for the Lagrange multiplier is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \lambda } = \mathbb { E } _ { x \sim \mathcal { B } , \xi \sim q _ { \phi } ( \xi | c _ { t } ) } \left[ \lambda \left( Q _ { h } ( x , u , \xi ; \psi ) - ( 1 - \delta ) \right) \right] , } \end{array}
$$

where $u \sim \pi ( x , \xi ; \theta )$ , and the multiplier is constrained to satisfy $\lambda \geq 0 .$

In addition to its role in constrained policy optimization, the safety critic is also used as a safety filter. By Definitions 1 and 2, the safety values provide an assessment of how safe it is to take action u under the current information state $( x , \xi )$ . Accordingly, if the nominal action proposed by the task actor is deemed unsafe by the safety critic, we override it with the action produced by the safety actor.

Following [5], the encoder is trained jointly with both the safety critic and the performance critic, and is regularized toward a standard Gaussian prior $p ( \xi ) = \mathcal { N } ( 0 , I )$ to prevent collapse. Its loss function is defined as

$$
\mathcal { L } _ { q } ( \phi ) = \mathcal { L } _ { Q _ { h } } ( \psi , \phi ) + \mathcal { L } _ { Q } ( \omega , \phi ) + \beta _ { \mathrm { K L } } D _ { \mathrm { K L } } ( q _ { \phi } ( \xi | c _ { t } ) \parallel p ( \xi ) ) .
$$

where $\beta _ { \mathrm { K L } }$ is a positive weighting coefficient. Thus, the encoder learns a latent representation that is informative for both reward prediction and safety certification.

We call our algorithm Information Safety Dual Actor-Critic (ISDAC). The overall method consists of two phases, meta-training and meta-testing, summarized in Algorithms 1 and 2, respectively. During meta-training, the agent is exposed to a distribution of tasks and jointly learns the encoder, the safety and performance critics, the safety actor, and the task actor. In this way, the latent variable ξ captures task uncertainty, while the learned policies achieve high return under safety constraints across tasks drawn from the training distribution. Note that, following the design in [5], we maintain two buffers during meta-training: a standard RL replay buffer B and a smaller context buffer $\textstyle B _ { c } .$ . The latter is introduced because encoder learning benefits from data that is closer to on-policy. During meta-testing, all learned parameters are kept fixed and the agent is deployed on a previously unseen task. Adaptation then proceeds through the accumulation of context and the corresponding posterior update. Once sufficient context has been collected, the adaptation process is complete, and the agent can resample the latent variable $\xi$ for formal deployment or evaluation.

## V. EXPERIMENTS

In this section, we evaluate the proposed algorithm on two continuous-control safe meta-RL benchmarks based on the HalfCheetah robot in the MuJoCo simulator [18], adapted from standard meta-RL benchmarks [5], [6].

HalfCheetah-Fwd-Back. This environment consists of two tasks: moving forward $( d = + 1 )$ and backward $( d =$ $^ { - 1 ) }$ . The task-specific reward is $r ( \boldsymbol { x } , \boldsymbol { u } ) = d \cdot v - 0 . 0 5 \| \boldsymbol { u } \| _ { 2 } ^ { 2 } ,$ where $v$ denotes the robot’s velocity, $d \in \{ - 1 , + 1 \}$ specifies the desired direction, and the second term penalizes control effort. The safety constraint requires the velocity magnitude to remain below a prescribed threshold: $| v | \leq v _ { \operatorname* { m a x } }$ with $v _ { \mathrm { m a x } } ~ = ~ 6 . 0$ . Since the reward encourages high-speed locomotion while the constraint limits velocity, there is an inherent conflict between performance and safety in both tasks.

HalfCheetah-Vel. In this environment, the robot is required to track a task-specific target velocity $v ^ { \star }$ sampled uniformly from [0, 3]. The reward penalizes deviation from the target: $r ( x , u ) = - | v - v ^ { \star } | - 0 . 0 5 \| u \| _ { 2 } ^ { 2 }$ . The same form of safety constraint is imposed, namely $| v | \leq v _ { \operatorname* { m a x } }$ , with the threshold $v _ { \mathrm { m a x } } = 1 . 5$ . The environment contains 100 training tasks and 30 test tasks, each associated with a distinct target velocity. This setting is particularly challenging since the relationship between the reward and the safety constraint depends on the task: when $v ^ { \star } \leq v _ { \mathrm { m a x } }$ , the two are naturally aligned, whereas when $v ^ { \star } > v _ { \mathrm { m a x } } .$ , they are in direct conflict. The agent must therefore learn to distinguish among tasks and adapt its behavior accordingly.

Algorithm 1: ISDAC: Meta-Training.   
Input: batch of training tasks $\{ z _ { i } \} _ { i = 1 } ^ { N } \sim p ( z ) ;$   
network parameters $\theta , \varphi , \omega _ { 1 } , \omega _ { 2 } , \psi _ { 1 } , \psi _ { 2 } , \phi ,$   
target network parameters $\hat { \omega } _ { 1 }  \omega _ { 1 }$   
$\hat { \omega } _ { 2 }  \omega _ { 2 } , \hat { \psi } _ { 1 }  \psi _ { 1 } , \hat { \psi } _ { 2 }  \psi _ { 2 }$ , target   
smoothing coefficient $\tau ,$ regularization   
coefficient $\alpha ,$ learning rate $\eta ,$ multiplier $\lambda ,$   
safety threshold $\delta .$   
Initialize replay buffers $B ^ { i }$ and $B _ { c } ^ { i }$ for training tasks.   
for each outer step do   
for each task $z _ { i }$ do   
Initialize context $c ^ { i } ;$   
for each system step do   
Sample $\xi \sim q _ { \phi } ( \xi | c ^ { i } ) ;$   
Sample nominal action $u \sim \pi ( x , \xi ; \theta )$   
if $Q _ { h } ( x , u , \xi ; \psi ) < 1 - \delta$ then   
$| \mathrm { ~  ~ \psi ~ } u  \pi _ { h } ( x , \xi ; \varphi ) ;$   
end   
Execute u. Update $B ^ { i }$ and $B _ { c } ^ { i } ;$   
Update context $c ^ { i }$   
end   
end   
for each inner step do   
for each task $z _ { i }$ do   
Sample context batch $c ^ { i } \sim B _ { c } ^ { i }$ and data   
batch $b ^ { i } \sim B ^ { i } ;$   
Sample $\xi \sim q _ { \phi } ( \xi | c ^ { i } ) ;$   
Compute per-task losses $\mathcal { L } _ { Q } ^ { i } ( \omega ) , \mathcal { L } _ { Q _ { h } } ^ { i } ( \psi )$   
$\mathcal { L } _ { \pi } ^ { i } ( \theta ) , \mathcal { L } _ { \pi _ { h } } ^ { i } ( \varphi ) , \mathcal { L } _ { \lambda } ^ { i } ,$ , and $\dot { \mathcal { L } } _ { q } ^ { i } ( \phi )$   
end   
for each gradient step do   
$\begin{array} { r } { \omega _ { j }  \omega _ { j } - \eta \nabla _ { \omega _ { j } } \sum _ { i } \mathcal { L } _ { Q } ^ { i } , \mathrm { f o r } j \in \{ 1 , 2 \} ; } \end{array}$   
$\begin{array} { r } { \psi _ { j }  \psi _ { j } - \eta \nabla _ { \psi _ { j } } \sum _ { i } \mathcal { L } _ { Q _ { h } } ^ { \bar { i } } , \mathrm { f o r } j \in \{ 1 , 2 \} } \end{array}$   
$\begin{array} { r } { \theta  \theta - \eta \nabla _ { \theta } \sum _ { i } \mathcal { L } _ { \pi } ^ { i } ; } \end{array}$   
$\begin{array} { r } { \varphi  \varphi - \eta \nabla _ { \varphi } \sum _ { i } \mathcal { L } _ { \pi _ { h } } ^ { i } ; } \end{array}$   
λ ← max $\{ \lambda ^ { \dot { \mathbf { \Gamma } } } - \overline { { \eta } } \mathbf { \tilde { \nabla } } _ { \lambda } \dot { \sum _ { i } } \mathcal { L } _ { \lambda } ^ { i } , 0 \}$   
$\begin{array} { r } { \phi  \phi - \eta \mathbf { \dot { V } } _ { \phi } \sum _ { i } \mathcal { L } _ { q } ^ { i } ; } \end{array}$   
$\hat { \omega } _ { j }  \tau \omega _ { j } + ( 1 - \tau ) \hat { \omega } _ { j }$ , for $j \in \{ 1 , 2 \} ;$   
$\hat { \psi } _ { j }  \tau \psi _ { j } + ( 1 - \tau ) \hat { \psi } _ { j } , \mathrm { f o r } j \in \{ 1 , 2 \}$   
end   
end   
end

```prolog
Algorithm 2: ISDAC: Meta-Testing.
Input: Test task $z _ { I } \sim p ( z )$ , safety threshold δ.
Initialize context $c ^ { I }$
for each system step do
Sample $\xi \sim q _ { \phi } ( \xi | c ^ { I } ) ;$
Sample nominal action $u \sim \pi ( x , \xi ; \theta )$
if $Q _ { h } ( x , u , \xi ; \psi ) < 1 - \delta$ then
$u  \pi _ { h } ( x , \xi ; \varphi ) ;$
end
Execute u. Gather data;
Accumulate context $c ^ { I } .$
end
```

![](images/8cd82660a6d3ee60b61076034409fdf3d1b980e852439d4bd465d8d6b49479a5.jpg)  
Fig. 1. Meta-testing performance on HalfCheetah-Fwd-Back as metatraining progresses. The solid lines correspond to the mean and the shaded regions correspond to ±1 standard deviation over three seeds.

Baselines. We compare our method against two baseline algorithms. The first is PEARL [5], a widely adopted standard meta-RL algorithm. The second, which we call PEARL-Lagrangian, combines PEARL with SAC-Lagrangian [19], a widely used safe RL algorithm under the CMDP framework. This approach enforces trajectory-cost constraints in expectation through a learned cost value function. All three methods use multilayer perceptrons as function approximators and share the same hyperparameter settings. We choose not to include the optimization-based safe meta-RL methods [8], [9], [10], as we were unable to obtain satisfactory performance within our interaction budget, which may reflect limitations in their sample efficiency.

During meta-training, we report two meta-testing metrics: episode return (cumulative reward) and episode violation (the number of unsafe timesteps). Fig. 1 shows the results on HalfCheetah-Fwd-Back, with the left column reporting averages across both tasks and the remaining columns showing the backward and forward tasks separately. Fig. 2 shows the results on HalfCheetah-Vel, with the left column reporting the average over all 30 test tasks and the remaining columns showing five representative tasks spanning target velocities in [0, 3], from the aligned regime (low $v ^ { \star } )$ to the conflicting regime (high $v ^ { \star } )$

PEARL achieves high return but incurs severe constraint violations, as it lacks any explicit safety mechanism. PEARL-Lagrangian reduces the number of violations, but fails to drive them close to zero, suggesting that a naive application of CMDP-based techniques to meta-RL is insufficient. In contrast, ISDAC maintains near-zero violations while achieving competitive return in both environments. The results on HalfCheetah-Vel further highlight the adaptivity of ISDAC. For low-velocity tasks, where the target velocity lies within the safety threshold, ISDAC performs comparably to PEARL. As the target velocity exceeds $v _ { \mathrm { m a x } }$ , ISDAC consistently maintains a low violation rate by learning to trade off return against safety. These results demonstrate that ISDAC can infer the task-dependent tension between reward maximization and safety from limited experience, and adapt its policy accordingly.

![](images/edbdfdaed2112d3e529b40ce593d94c1c4890c2346f6a73616d96f33c5ad356a.jpg)  
Fig. 2. Meta-testing performance on HalfCheetah-Vel as meta-training progresses. The solid lines correspond to the mean and the shaded regions correspond to ±1 standard deviation over three seeds.

## VI. CONCLUSION

In this paper, we presented a safe meta-RL framework for learning a meta-policy that can rapidly adapt to unseen tasks while satisfying safety requirements. We introduced the safety value function and established its theoretical properties. Based on these results, we further developed a practical safe meta-RL algorithm for complex high-dimensional tasks, in which the safety value function is learned and used to guide constrained policy optimization.

## REFERENCES

[1] J. Beck, R. Vuorio, E. Z. Liu, Z. Xiong, L. Zintgraf, C. Finn, and S. Whiteson, “A tutorial on meta-reinforcement learning,” Foundations and Trends in Machine Learning, vol. 18, no. 2-3, pp. 224–384, 2025.

[2] A. Nagabandi, I. Clavera, S. Liu, R. S. Fearing, P. Abbeel, S. Levine, and C. Finn, “Learning to adapt in dynamic, real-world environments through meta-reinforcement learning,” in International Conference on Learning Representations, 2019.

[3] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine, “Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning,” in Conference on robot learning. PMLR, 2020, pp. 1094–1100.

[4] C. Finn, P. Abbeel, and S. Levine, “Model-agnostic meta-learning for fast adaptation of deep networks,” in International Conference on Machine Learning. PMLR, 2017, pp. 1126–1135.

[5] K. Rakelly, A. Zhou, C. Finn, S. Levine, and D. Quillen, “Efficient offpolicy meta-reinforcement learning via probabilistic context variables,” in International Conference on Machine Learning. PMLR, 2019, pp. 5331–5340.

[6] L. Zintgraf, S. Schulze, C. Lu, L. Feng, M. Igl, K. Shiarlis, Y. Gal, K. Hofmann, and S. Whiteson, “Varibad: Variational bayes-adaptive deep rl via meta-learning,” Journal of Machine Learning Research, vol. 22, no. 289, pp. 1–39, 2021.

[7] E. Altman, Constrained Markov Decision Processes: Stochastic Modeling. Routledge, 1999.

[8] V. Khattar, Y. Ding, B. Sel, J. Lavaei, and M. Jin, “A CMDPwithin-online framework for meta-safe reinforcement learning,” in The Eleventh International Conference on Learning Representations, 2023.

[9] M. Cho and C. Sun, “Constrained meta-reinforcement learning for adaptable safety guarantee with differentiable convex programming,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 19, 2024, pp. 20 975–20 983.

[10] S. Xu and M. Zhu, “Efficient safe meta-reinforcement learning: Provable near-optimality and anytime safety,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[11] W. Zhao, T. He, R. Chen, T. Wei, and C. Liu, “State-wise safe reinforcement learning: A survey,” in International Joint Conference on Artificial Intelligence. IJCAI, 2023.

[12] M. O. Duff, Optimal Learning: Computational procedures for Bayesadaptive Markov decision processes. University of Massachusetts Amherst, 2002.

[13] M. Ghavamzadeh, S. Mannor, J. Pineau, and A. Tamar, “Bayesian reinforcement learning: A survey,” Foundations and Trends® in Machine Learning, vol. 8, no. 5-6, pp. 359–483, 2015.

[14] S. Bansal, M. Chen, S. Herbert, and C. J. Tomlin, “Hamilton-Jacobi reachability: A brief overview and recent advances,” in 2017 IEEE 56th Annual Conference on Decision and Control (CDC). IEEE, 2017, pp. 2242–2253.

[15] J. F. Fisac, N. F. Lugovoy, V. Rubies-Royo, S. Ghosh, and C. J. Tomlin, “Bridging hamilton-jacobi safety analysis and reinforcement learning,” in 2019 International Conference on Robotics and Automation (ICRA). IEEE, 2019, pp. 8550–8556.

[16] Z. Li, C. Hu, Y. Wang, Y. Yang, and S. E. Li, “Safe reinforcement learning with dual robustness,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[17] T. Haarnoja, A. Zhou, P. Abbeel, and S. Levine, “Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor,” in International Conference on Machine Learning. PMLR, 2018, pp. 1861–1870.

[18] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems. IEEE, 2012, pp. 5026–5033.

[19] S. Ha, P. Xu, Z. Tan, S. Levine, and J. Tan, “Learning to walk in the real world with minimal human effort,” in Conference on Robot Learning, vol. 155. PMLR, 2021, pp. 1110–1120.